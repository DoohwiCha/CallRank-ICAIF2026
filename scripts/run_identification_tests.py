#!/usr/bin/env python3
"""Run hard-news, own-history placebo, and issuer-identity tests for CallRank.

The reported CallRank representation, training boundary, PCA/Ridge head, target,
and release-aligned evaluation months remain fixed. Consensus fields are
reported as diagnostic proxies because the licensed snapshot does not
establish their historical estimate vintage. Seasonal actual-based controls
use only values reported by the current and earlier events.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


LOCAL_SCRIPTS = Path(__file__).resolve().parent
if str(LOCAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(LOCAL_SCRIPTS))

from common import (  # noqa: E402
    build_panel,
    hac_pvalue,
    load_targets,
    monthly_rank_ic,
    rank_target,
)
from callrank_features import (  # noqa: E402
    SECTOR_MONTH_KEYS,
    company_month_sections,
    group_mean,
    normalize_with_missing,
    prior_residual,
)


EVENT_MAP: Path
TARGETS: Path
AUTHORITATIVE_PREDICTIONS: Path
REPORTED_EARNINGS: Path
EMBEDDING_DIRS: dict[int, Path]
EMBEDDING_COLUMNS = [f"e{index:03d}" for index in range(768)]
LATEST_TRAINING_OUTCOME = pd.Timestamp("2020-12-01")
EVALUATION_START = pd.Timestamp("2023-11-01")
PCA_COMPONENTS = 32
RIDGE_ALPHA = 10_000.0
MINIMUM_PRIOR = 2
PLACEBO_RANDOMIZATIONS = 1_000
PLACEBO_SEED = 20260725
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--event-map", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument(
        "--authoritative-predictions",
        type=Path,
        required=True,
    )
    parser.add_argument("--earnings", type=Path, required=True)
    parser.add_argument(
        "--embedding-128",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--embedding-160",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--embedding-200",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--placebo-randomizations",
        type=int,
        default=PLACEBO_RANDOMIZATIONS,
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_event_map() -> pd.DataFrame:
    frame = pd.read_parquet(EVENT_MAP)
    usable = frame[frame["cleanroom_usable"].fillna(False)].copy()
    usable["issuer_id"] = usable["canonical_issuer_id"].astype(str)
    usable["sector_clean"] = usable["canonical_sector"].astype(str)
    usable["month_clean"] = pd.to_datetime(usable["verified_month"])
    usable["call_date_clean"] = pd.to_datetime(usable["verified_call_date"])
    usable["ticker_clean"] = (
        usable["canonical_asset_symbol"]
        .astype(str)
        .str.upper()
        .str.replace(".", "-", regex=False)
    )
    if usable["issuer_call_id"].duplicated().any():
        raise RuntimeError("Clean-room event map has duplicate call identifiers")
    if usable.empty:
        raise RuntimeError("The clean-room event map has no usable calls")
    return usable


def load_company_month_qa(
    chunk_words: int, event_map: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    path = EMBEDDING_DIRS[chunk_words] / f"call_qa_all{chunk_words}.parquet"
    frame = pd.read_parquet(path)
    required = {"issuer_call_id", *EMBEDDING_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{path.name} is missing columns: {sorted(missing)}")
    merged = frame.merge(
        event_map[
            [
                "issuer_call_id",
                "issuer_id",
                "sector_clean",
                "month_clean",
            ]
        ],
        on="issuer_call_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(event_map):
        raise RuntimeError(
            f"Clean embedding join returned {len(merged):,}, expected {len(event_map):,}"
        )
    call_keys = merged[["issuer_call_id", "issuer_id"]].copy()
    call_keys["sector"] = merged["sector_clean"].astype(str)
    call_keys["month"] = pd.to_datetime(merged["month_clean"])
    values = merged[EMBEDDING_COLUMNS].to_numpy(dtype=np.float64)
    dummy = np.zeros_like(values)
    company_keys, _, company_qa, _ = company_month_sections(
        call_keys,
        dummy,
        values,
    )
    return company_keys, company_qa


def prior_state(
    values: np.ndarray,
    identifiers: pd.Series,
    dates: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    means = np.full_like(values, np.nan, dtype=np.float64)
    counts = np.zeros(len(values), dtype=np.int64)
    order = pd.DataFrame(
        {
            "identifier": identifiers.astype(str).to_numpy(),
            "date": pd.to_datetime(dates).to_numpy(),
            "position": np.arange(len(values), dtype=int),
        }
    ).sort_values(["date", "identifier", "position"], kind="mergesort")
    sums: dict[str, np.ndarray] = {}
    seen: dict[str, int] = defaultdict(int)
    for month, month_rows in order.groupby("date", sort=True):
        del month
        for row in month_rows.itertuples(index=False):
            identifier = str(row.identifier)
            position = int(row.position)
            count = seen[identifier]
            counts[position] = count
            if count:
                means[position] = sums[identifier] / count
        for row in month_rows.itertuples(index=False):
            identifier = str(row.identifier)
            position = int(row.position)
            if identifier not in sums:
                sums[identifier] = np.zeros(values.shape[1], dtype=np.float64)
            sums[identifier] += values[position]
            seen[identifier] += 1
    return means, counts


def fit_linear_head(
    panel: pd.DataFrame, values: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    usable = np.isfinite(values).all(axis=1) & panel["excess_return"].notna().to_numpy()
    train = usable & panel["exit_date"].le(LATEST_TRAINING_OUTCOME).to_numpy()
    train_index = np.flatnonzero(train)
    if len(train_index) < 40:
        raise RuntimeError("Insufficient training rows")
    scaler = StandardScaler()
    train_x = scaler.fit_transform(values[train_index])
    pca = PCA(
        n_components=min(PCA_COMPONENTS, train_x.shape[0] - 1, train_x.shape[1]),
        svd_solver="full",
        random_state=42,
    )
    train_z = pca.fit_transform(train_x)
    ridge = Ridge(alpha=RIDGE_ALPHA).fit(
        train_z,
        rank_target(panel.iloc[train_index]),
    )
    score_index = np.flatnonzero(usable)
    scored = panel.iloc[score_index].copy()
    scored["pred_raw"] = ridge.predict(
        pca.transform(scaler.transform(values[score_index]))
    )
    raw_weight = (pca.components_.T @ ridge.coef_) / scaler.scale_
    return scored, raw_weight, {
        "train_rows": int(len(train_index)),
        "score_rows": int(len(score_index)),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
    }


def build_callrank_scores(
    event_map: pd.DataFrame,
    targets: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    states: dict[int, dict[str, Any]] = {}
    for chunk_words in sorted(EMBEDDING_DIRS):
        company_keys, qa = load_company_month_qa(chunk_words, event_map)
        residual = prior_residual(
            qa,
            company_keys["issuer_id"],
            company_keys["month"],
            MINIMUM_PRIOR,
        )
        direction = normalize_with_missing(residual)
        sector_keys, sector_direction, firm_counts = group_mean(
            company_keys,
            direction,
            SECTOR_MONTH_KEYS,
        )
        panel = build_panel(sector_keys, targets)
        scored, raw_weight, fit_audit = fit_linear_head(panel, sector_direction)
        scored["pred_rank"] = scored.groupby("month")["pred_raw"].rank(
            method="average", pct=True
        )
        scored["chunk_words"] = chunk_words
        frames.append(
            scored[
                [
                    "sector",
                    "month",
                    "entry_date",
                    "exit_date",
                    "excess_return",
                    "pred_rank",
                    "chunk_words",
                ]
            ]
        )
        history_mean, history_count = prior_state(
            qa,
            company_keys["issuer_id"],
            company_keys["month"],
        )
        states[chunk_words] = {
            "company_keys": company_keys,
            "qa": qa,
            "history_mean": history_mean,
            "history_count": history_count,
            "direction": direction,
            "raw_weight": raw_weight,
            "fit_audit": fit_audit,
            "firm_counts": firm_counts,
        }
    stacked = pd.concat(frames, ignore_index=True)
    ensemble = (
        stacked.groupby(
            ["sector", "month"],
            sort=True,
            observed=True,
        )
        .agg(
            entry_date=("entry_date", "first"),
            exit_date=("exit_date", "first"),
            excess_return=("excess_return", "first"),
            pred=("pred_rank", "mean"),
            chunk_models=("chunk_words", "nunique"),
        )
        .reset_index()
    )
    if not ensemble["chunk_models"].eq(len(EMBEDDING_DIRS)).all():
        raise RuntimeError("A reconstructed CallRank row lacks a chunk model")
    return ensemble, states


def verify_authoritative_scores(reconstructed: pd.DataFrame) -> dict[str, Any]:
    digest = sha256_file(AUTHORITATIVE_PREDICTIONS)
    authoritative = pd.read_parquet(AUTHORITATIVE_PREDICTIONS)
    authoritative = authoritative[
        authoritative["candidate"].eq("firm_conditioned_qa_direction")
    ][["sector", "month", "pred", "excess_return"]].copy()
    authoritative["month"] = pd.to_datetime(authoritative["month"])
    evaluation = reconstructed[
        reconstructed["month"].ge(EVALUATION_START)
    ][["sector", "month", "pred", "excess_return"]]
    merged = authoritative.merge(
        evaluation,
        on=["sector", "month"],
        suffixes=("_authoritative", "_reconstructed"),
        validate="one_to_one",
    )
    if len(merged) != len(authoritative):
        raise RuntimeError("Reconstructed evaluation keys differ from authoritative keys")
    prediction_diff = (
        merged["pred_authoritative"] - merged["pred_reconstructed"]
    ).abs()
    return {
        "authoritative_sha256": digest,
        "rows": int(len(merged)),
        "max_prediction_difference": float(prediction_diff.max()),
        "max_target_difference": float(
            (
                merged["excess_return_authoritative"]
                - merged["excess_return_reconstructed"]
            )
            .abs()
            .max()
        ),
    }


def reported_earnings_features() -> pd.DataFrame:
    frame = pd.read_parquet(REPORTED_EARNINGS).copy()
    frame["ticker_clean"] = (
        frame["ticker"]
        .astype(str)
        .str.upper()
        .str.replace(".", "-", regex=False)
    )
    frame["event_date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in (
        "epsActual",
        "epsEstimated",
        "revenueActual",
        "revenueEstimated",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["event_date"]).sort_values(
        ["ticker_clean", "event_date"]
    )
    frame = frame.drop_duplicates(["ticker_clean", "event_date"], keep="last")
    eps_denom = frame["epsEstimated"].abs().clip(lower=0.05)
    revenue_denom = frame["revenueEstimated"].abs().clip(lower=1.0)
    frame["eps_consensus_surprise"] = (
        frame["epsActual"] - frame["epsEstimated"]
    ) / eps_denom
    frame["revenue_consensus_surprise"] = (
        frame["revenueActual"] - frame["revenueEstimated"]
    ) / revenue_denom
    grouped = frame.groupby("ticker_clean", sort=False)
    eps_lag4 = grouped["epsActual"].shift(4)
    revenue_lag4 = grouped["revenueActual"].shift(4)
    date_lag4 = grouped["event_date"].shift(4)
    annual_gap = (frame["event_date"] - date_lag4).dt.days.between(300, 430)
    frame["eps_seasonal_change"] = np.where(
        annual_gap,
        (frame["epsActual"] - eps_lag4) / eps_lag4.abs().clip(lower=0.05),
        np.nan,
    )
    frame["revenue_seasonal_growth"] = np.where(
        annual_gap & revenue_lag4.abs().gt(1.0),
        frame["revenueActual"] / revenue_lag4 - 1.0,
        np.nan,
    )
    return frame


def match_structured_features(
    event_map: pd.DataFrame,
    company_keys: pd.DataFrame,
    history_count: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    earnings = reported_earnings_features()
    left = event_map[
        [
            "issuer_call_id",
            "issuer_id",
            "sector_clean",
            "month_clean",
            "call_date_clean",
            "ticker_clean",
        ]
    ].sort_values(["call_date_clean", "ticker_clean"])
    feature_columns = [
        "eps_consensus_surprise",
        "revenue_consensus_surprise",
        "eps_seasonal_change",
        "revenue_seasonal_growth",
    ]
    right = earnings[
        ["ticker_clean", "event_date", *feature_columns]
    ].sort_values(
        ["event_date", "ticker_clean"]
    )
    matched = pd.merge_asof(
        left,
        right,
        left_on="call_date_clean",
        right_on="event_date",
        by="ticker_clean",
        direction="nearest",
        tolerance=pd.Timedelta(days=3),
    )
    matched["date_gap_days"] = (
        matched["call_date_clean"] - matched["event_date"]
    ).abs().dt.days
    company = (
        matched.groupby(
            ["issuer_id", "sector_clean", "month_clean"],
            sort=True,
            observed=True,
        )[feature_columns]
        .mean()
        .reset_index()
        .rename(columns={"sector_clean": "sector", "month_clean": "month"})
    )
    eligible = company_keys.copy()
    eligible["history_count"] = history_count
    eligible = eligible[eligible["history_count"].ge(MINIMUM_PRIOR)]
    company = eligible.merge(
        company,
        on=["issuer_id", "sector", "month"],
        how="left",
        validate="one_to_one",
    )
    rows: list[dict[str, Any]] = []
    for (sector, month), group in company.groupby(
        ["sector", "month"], sort=True, observed=True
    ):
        row: dict[str, Any] = {
            "sector": sector,
            "month": pd.Timestamp(month),
            "eligible_firms": int(len(group)),
        }
        for column in feature_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row[column] = float(values.mean()) if values.notna().any() else np.nan
            row[f"{column}_coverage"] = float(values.notna().mean())
        rows.append(row)
    sector_features = pd.DataFrame(rows)
    evaluation_calls = matched["call_date_clean"].ge(EVALUATION_START)
    training_calls = matched["call_date_clean"].lt("2021-01-01")
    audit = {
        "reported_earnings_rows": int(len(earnings)),
        "clean_calls": int(len(matched)),
        "evaluation_event_match_share": float(
            matched.loc[evaluation_calls, "event_date"].notna().mean()
        ),
        "training_event_match_share": float(
            matched.loc[training_calls, "event_date"].notna().mean()
        ),
        "evaluation_consensus_pair_share": float(
            matched.loc[
                evaluation_calls,
                ["eps_consensus_surprise", "revenue_consensus_surprise"],
            ]
            .notna()
            .all(axis=1)
            .mean()
        ),
        "training_consensus_pair_share": float(
            matched.loc[
                training_calls,
                ["eps_consensus_surprise", "revenue_consensus_surprise"],
            ]
            .notna()
            .all(axis=1)
            .mean()
        ),
        "consensus_vintage_status": (
            "diagnostic_only: the licensed snapshot does not prove the estimate "
            "was frozen before each historical announcement"
        ),
    }
    return sector_features, audit


def winsorize_from_training(
    frame: pd.DataFrame,
    columns: list[str],
    train_mask: pd.Series,
) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        source = pd.to_numeric(output.loc[train_mask, column], errors="coerce")
        lower, upper = source.quantile([0.01, 0.99])
        output[column] = pd.to_numeric(output[column], errors="coerce").clip(
            lower=lower,
            upper=upper,
        )
    return output


def frozen_structured_score(
    panel: pd.DataFrame,
    feature_columns: list[str],
    label: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = panel.copy()
    train_mask = work["exit_date"].le(LATEST_TRAINING_OUTCOME)
    work = winsorize_from_training(work, feature_columns, train_mask)
    train = work[train_mask].dropna(subset=feature_columns + ["excess_return"])
    score = work[work["month"].ge(EVALUATION_START)].dropna(
        subset=feature_columns + ["excess_return"]
    )
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train[feature_columns])
    score_x = scaler.transform(score[feature_columns])
    model = Ridge(alpha=1.0).fit(train_x, rank_target(train))
    scored = score[
        ["sector", "month", "entry_date", "exit_date", "excess_return"]
    ].copy()
    scored["pred"] = model.predict(score_x)
    scored["structured_model"] = label
    return scored, {
        "model": label,
        "features": feature_columns,
        "train_rows": int(len(train)),
        "evaluation_rows": int(len(score)),
        "coefficients": {
            column: float(value)
            for column, value in zip(feature_columns, model.coef_, strict=True)
        },
    }


def residualize(y: pd.Series, x: pd.Series) -> pd.Series:
    frame = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    output = pd.Series(np.nan, index=y.index, dtype=float)
    if len(frame) < 4 or frame["x"].nunique() < 2:
        return output
    design = sm.add_constant(frame["x"].to_numpy(dtype=float))
    fit = sm.OLS(frame["y"].to_numpy(dtype=float), design).fit()
    output.loc[frame.index] = fit.resid
    return output


def partial_ic(
    callrank: pd.DataFrame,
    structured: pd.DataFrame,
) -> pd.Series:
    merged = callrank.merge(
        structured[["sector", "month", "pred"]],
        on=["sector", "month"],
        suffixes=("_callrank", "_structured"),
        validate="one_to_one",
    )
    rows: dict[pd.Timestamp, float] = {}
    for month, group in merged.groupby("month", sort=True):
        if len(group) < 6:
            continue
        callrank_rank = group["pred_callrank"].rank(method="average", pct=True)
        target_rank = group["excess_return"].rank(method="average", pct=True)
        structured_rank = group["pred_structured"].rank(
            method="average", pct=True
        )
        callrank_resid = residualize(callrank_rank, structured_rank)
        target_resid = residualize(target_rank, structured_rank)
        valid = callrank_resid.notna() & target_resid.notna()
        if valid.sum() >= 6:
            rows[pd.Timestamp(month)] = float(
                np.corrcoef(
                    callrank_resid[valid],
                    target_resid[valid],
                )[0, 1]
            )
    return pd.Series(rows, dtype=float).sort_index()


def monthly_encompassing_coefficients(
    callrank: pd.DataFrame,
    structured: pd.DataFrame,
) -> pd.DataFrame:
    merged = callrank.merge(
        structured[["sector", "month", "pred"]],
        on=["sector", "month"],
        suffixes=("_callrank", "_structured"),
        validate="one_to_one",
    )
    rows: list[dict[str, Any]] = []
    for month, group in merged.groupby("month", sort=True):
        if len(group) < 6:
            continue
        y = group["excess_return"].rank(method="average", pct=True)
        x = pd.DataFrame(
            {
                "callrank": group["pred_callrank"].rank(
                    method="average", pct=True
                ),
                "structured": group["pred_structured"].rank(
                    method="average", pct=True
                ),
            }
        )
        fit = sm.OLS(y.to_numpy(dtype=float), sm.add_constant(x)).fit()
        rows.append(
            {
                "month": pd.Timestamp(month),
                "callrank_coefficient": float(fit.params["callrank"]),
                "structured_coefficient": float(fit.params["structured"]),
            }
        )
    return pd.DataFrame(rows)


def equal_rank_ensemble(
    callrank: pd.DataFrame,
    structured: pd.DataFrame,
) -> pd.DataFrame:
    merged = callrank.merge(
        structured[["sector", "month", "pred"]],
        on=["sector", "month"],
        suffixes=("_callrank", "_structured"),
        validate="one_to_one",
    )
    merged["pred"] = 0.5 * (
        merged.groupby("month")["pred_callrank"].rank(
            method="average", pct=True
        )
        + merged.groupby("month")["pred_structured"].rank(
            method="average", pct=True
        )
    )
    return merged[
        [
            "sector",
            "month",
            "entry_date",
            "exit_date",
            "excess_return",
            "pred",
        ]
    ].copy()


def summarize_ic(label: str, values: pd.Series) -> dict[str, Any]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "test": label,
        "months": int(len(clean)),
        "mean_ic": float(clean.mean()),
        "newey_west_p": hac_pvalue(clean, 1),
        "positive_month_share": float(clean.gt(0).mean()),
    }


def structured_news_tests(
    reconstructed: pd.DataFrame,
    states: dict[int, dict[str, Any]],
    targets: pd.DataFrame,
    event_map: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    state = states[160]
    sector_features, data_audit = match_structured_features(
        event_map,
        state["company_keys"],
        state["history_count"],
    )
    panel = reconstructed[
        [
            "sector",
            "month",
            "entry_date",
            "exit_date",
            "excess_return",
            "pred",
        ]
    ].merge(
        sector_features,
        on=["sector", "month"],
        how="left",
        validate="one_to_one",
    )
    target_panel = targets.merge(
        sector_features,
        on=["sector", "month"],
        how="inner",
        validate="one_to_one",
    )
    models = {
        "consensus_proxy": [
            "eps_consensus_surprise",
            "revenue_consensus_surprise",
        ],
        "seasonal_pit_proxy": [
            "eps_seasonal_change",
            "revenue_seasonal_growth",
        ],
    }
    result_rows: list[dict[str, Any]] = []
    monthly_frames: list[pd.DataFrame] = []
    model_audits: list[dict[str, Any]] = []
    callrank_eval = panel[panel["month"].ge(EVALUATION_START)][
        ["sector", "month", "entry_date", "exit_date", "excess_return", "pred"]
    ].copy()
    for label, columns in models.items():
        scored, model_audit = frozen_structured_score(
            target_panel,
            columns,
            label,
        )
        model_audits.append(model_audit)
        structured_ic = monthly_rank_ic(scored)
        partial = partial_ic(callrank_eval, scored)
        encompassing = monthly_encompassing_coefficients(callrank_eval, scored)
        combined = equal_rank_ensemble(callrank_eval, scored)
        combined_ic = monthly_rank_ic(combined)
        paired_increment = pd.concat(
            [
                combined_ic.rename("combined"),
                structured_ic.rename("structured"),
            ],
            axis=1,
        ).dropna()
        incremental_ic = (
            paired_increment["combined"] - paired_increment["structured"]
        )
        result_rows.append(summarize_ic(f"{label}: structured only", structured_ic))
        result_rows.append(
            summarize_ic(f"{label}: equal-rank combined", combined_ic)
        )
        result_rows.append(
            {
                "test": f"{label}: combined minus structured",
                "months": int(len(incremental_ic)),
                "mean_ic": float(incremental_ic.mean()),
                "newey_west_p": hac_pvalue(incremental_ic, 1),
                "positive_month_share": float(
                    incremental_ic.gt(0).mean()
                ),
            }
        )
        result_rows.append(summarize_ic(f"{label}: CallRank partial IC", partial))
        result_rows.append(
            {
                "test": f"{label}: CallRank encompassing coefficient",
                "months": int(len(encompassing)),
                "mean_ic": float(encompassing["callrank_coefficient"].mean()),
                "newey_west_p": hac_pvalue(
                    encompassing["callrank_coefficient"], 1
                ),
                "positive_month_share": float(
                    encompassing["callrank_coefficient"].gt(0).mean()
                ),
            }
        )
        result_rows.append(
            {
                "test": f"{label}: structured encompassing coefficient",
                "months": int(len(encompassing)),
                "mean_ic": float(encompassing["structured_coefficient"].mean()),
                "newey_west_p": hac_pvalue(
                    encompassing["structured_coefficient"], 1
                ),
                "positive_month_share": float(
                    encompassing["structured_coefficient"].gt(0).mean()
                ),
            }
        )
        frame = pd.DataFrame(
            {
                "month": structured_ic.index,
                "structured_ic": structured_ic.to_numpy(),
            }
        )
        frame["model"] = label
        frame = frame.merge(encompassing, on="month", how="left")
        frame = frame.merge(
            partial.rename("callrank_partial_ic").rename_axis("month").reset_index(),
            on="month",
            how="left",
        )
        monthly_frames.append(frame)
    full_callrank_ic = monthly_rank_ic(callrank_eval)
    result_rows.insert(0, summarize_ic("CallRank full evaluation", full_callrank_ic))
    return (
        pd.DataFrame(result_rows),
        pd.concat(monthly_frames, ignore_index=True),
        {
            "data": data_audit,
            "models": model_audits,
            "sector_feature_rows": int(len(sector_features)),
        },
    )


def sector_history_excluding_firm(
    keys: pd.DataFrame,
    qa: np.ndarray,
    own_counts: np.ndarray,
) -> np.ndarray:
    sector_reference = np.full_like(qa, np.nan)
    frame = keys.copy().reset_index(drop=True)
    frame["position"] = np.arange(len(frame), dtype=int)
    firm_sum: dict[str, np.ndarray] = {}
    firm_count: dict[str, int] = defaultdict(int)
    sector_sum: dict[str, np.ndarray] = {}
    sector_count: dict[str, int] = defaultdict(int)
    for month, month_rows in frame.groupby("month", sort=True):
        del month
        for row in month_rows.itertuples(index=False):
            position = int(row.position)
            if own_counts[position] < MINIMUM_PRIOR:
                continue
            issuer = str(row.issuer_id)
            sector = str(row.sector)
            own_sum = firm_sum.get(
                issuer, np.zeros(qa.shape[1], dtype=np.float64)
            )
            own_count = firm_count[issuer]
            local_sector_count = sector_count[sector] - own_count
            if local_sector_count >= MINIMUM_PRIOR:
                sector_reference[position] = (
                    sector_sum[sector] - own_sum
                ) / local_sector_count
        for row in month_rows.itertuples(index=False):
            position = int(row.position)
            issuer = str(row.issuer_id)
            sector = str(row.sector)
            if issuer not in firm_sum:
                firm_sum[issuer] = np.zeros(qa.shape[1], dtype=np.float64)
            if sector not in sector_sum:
                sector_sum[sector] = np.zeros(qa.shape[1], dtype=np.float64)
            firm_sum[issuer] += qa[position]
            firm_count[issuer] += 1
            sector_sum[sector] += qa[position]
            sector_count[sector] += 1
    return sector_reference


def fixed_head_sector_scores(
    keys: pd.DataFrame,
    directions: np.ndarray,
    raw_weight: np.ndarray,
) -> pd.DataFrame:
    contribution = directions @ raw_weight
    frame = keys[["sector", "month"]].copy()
    frame["contribution"] = contribution
    frame = frame[np.isfinite(frame["contribution"])]
    return (
        frame.groupby(["sector", "month"], sort=True, observed=True)[
            "contribution"
        ]
        .mean()
        .rename("pred")
        .reset_index()
    )


def wrong_firm_candidate_values(
    keys: pd.DataFrame,
    qa: np.ndarray,
    history_counts: np.ndarray,
    raw_weight: np.ndarray,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    frame = keys.copy().reset_index(drop=True)
    frame["position"] = np.arange(len(frame), dtype=int)
    firm_sum: dict[str, np.ndarray] = {}
    firm_count: dict[str, int] = defaultdict(int)
    firm_sector: dict[str, str] = {}
    candidates: list[np.ndarray] = [
        np.asarray([], dtype=np.float64) for _ in range(len(frame))
    ]
    eligible_rows = 0
    rows_with_candidates = 0
    candidate_counts: list[int] = []
    for month, month_rows in frame.groupby("month", sort=True):
        del month
        for row in month_rows.itertuples(index=False):
            position = int(row.position)
            if history_counts[position] < MINIMUM_PRIOR:
                continue
            eligible_rows += 1
            issuer = str(row.issuer_id)
            sector = str(row.sector)
            own_count = int(history_counts[position])
            donor_ids = [
                donor
                for donor, count in firm_count.items()
                if donor != issuer
                and count >= MINIMUM_PRIOR
                and firm_sector.get(donor) == sector
            ]
            if not donor_ids:
                continue
            differences = np.asarray(
                [abs(firm_count[donor] - own_count) for donor in donor_ids],
                dtype=int,
            )
            minimum_difference = int(differences.min())
            matched_ids = [
                donor
                for donor, difference in zip(
                    donor_ids, differences, strict=True
                )
                if int(difference) == minimum_difference
            ]
            donor_means = np.vstack(
                [firm_sum[donor] / firm_count[donor] for donor in matched_ids]
            )
            residuals = qa[position][None, :] - donor_means
            norms = np.linalg.norm(residuals, axis=1)
            valid = norms > 1e-12
            values = (
                residuals[valid] / norms[valid, None]
            ) @ raw_weight
            if len(values):
                candidates[position] = values.astype(np.float64)
                rows_with_candidates += 1
                candidate_counts.append(int(len(values)))
        for row in month_rows.itertuples(index=False):
            position = int(row.position)
            issuer = str(row.issuer_id)
            firm_sum.setdefault(
                issuer, np.zeros(qa.shape[1], dtype=np.float64)
            )
            firm_sum[issuer] += qa[position]
            firm_count[issuer] += 1
            firm_sector[issuer] = str(row.sector)
    return candidates, {
        "eligible_rows": eligible_rows,
        "rows_with_candidates": rows_with_candidates,
        "candidate_coverage": (
            rows_with_candidates / eligible_rows if eligible_rows else float("nan")
        ),
        "median_matched_donors": (
            float(np.median(candidate_counts)) if candidate_counts else 0.0
        ),
    }


def placebo_draw_scores(
    keys: pd.DataFrame,
    candidate_values: list[np.ndarray],
    targets: pd.DataFrame,
    randomizations: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(PLACEBO_SEED)
    base = keys[["sector", "month"]].copy().reset_index(drop=True)
    target = targets[
        targets["month"].ge(EVALUATION_START)
    ][["sector", "month", "excess_return"]]
    evaluation_positions = np.flatnonzero(base["month"].ge(EVALUATION_START))
    rows: list[dict[str, Any]] = []
    for iteration in range(randomizations):
        sampled = np.full(len(base), np.nan, dtype=np.float64)
        for position in evaluation_positions:
            local = candidate_values[position]
            if len(local):
                sampled[position] = local[int(rng.integers(0, len(local)))]
        scored = base.iloc[evaluation_positions].copy()
        scored["pred"] = sampled[evaluation_positions]
        scored = (
            scored.dropna(subset=["pred"])
            .groupby(["sector", "month"], sort=True, observed=True)["pred"]
            .mean()
            .reset_index()
            .merge(
                target,
                on=["sector", "month"],
                how="inner",
                validate="one_to_one",
            )
        )
        ic = monthly_rank_ic(scored)
        rows.append(
            {
                "iteration": iteration,
                "mean_rank_ic": float(ic.mean()),
                "months": int(len(ic)),
            }
        )
    draws = pd.DataFrame(rows)
    return draws, {
        "randomizations": int(len(draws)),
        "seed": PLACEBO_SEED,
        "mean": float(draws["mean_rank_ic"].mean()),
        "std": float(draws["mean_rank_ic"].std(ddof=1)),
        "q95": float(draws["mean_rank_ic"].quantile(0.95)),
        "maximum": float(draws["mean_rank_ic"].max()),
    }


def identity_probe(
    keys: pd.DataFrame,
    qa: np.ndarray,
    direction: np.ndarray,
) -> pd.DataFrame:
    frame = keys.copy().reset_index(drop=True)
    train_mask = frame["month"].lt("2021-01-01").to_numpy()
    eval_mask = frame["month"].ge(EVALUATION_START).to_numpy()
    rows: list[dict[str, Any]] = []
    for label, values in (
        ("current_qa", qa),
        ("own_history_direction", direction),
    ):
        valid = np.isfinite(values).all(axis=1)
        train_frame = frame[train_mask & valid].copy()
        eval_frame = frame[eval_mask & valid].copy()
        common = sorted(
            set(train_frame["issuer_id"]) & set(eval_frame["issuer_id"])
        )
        train_frame = train_frame[train_frame["issuer_id"].isin(common)]
        eval_frame = eval_frame[eval_frame["issuer_id"].isin(common)]
        train_positions = train_frame.index.to_numpy(dtype=int)
        eval_positions = eval_frame.index.to_numpy(dtype=int)
        centroid_rows: list[np.ndarray] = []
        centroid_ids: list[str] = []
        for issuer, group in train_frame.groupby("issuer_id", sort=True):
            mean = values[group.index.to_numpy(dtype=int)].mean(axis=0)
            norm = np.linalg.norm(mean)
            if norm > 1e-12:
                centroid_rows.append(mean / norm)
                centroid_ids.append(str(issuer))
        centroids = np.vstack(centroid_rows)
        evaluation = values[eval_positions]
        evaluation = evaluation / np.linalg.norm(
            evaluation, axis=1, keepdims=True
        )
        similarities = evaluation @ centroids.T
        order = np.argsort(-similarities, axis=1)
        truth = eval_frame["issuer_id"].astype(str).to_numpy()
        ranks: list[int] = []
        for row_index, issuer in enumerate(truth):
            position = centroid_ids.index(issuer)
            rank = int(np.flatnonzero(order[row_index] == position)[0]) + 1
            ranks.append(rank)
        rank_array = np.asarray(ranks)
        rows.append(
            {
                "representation": label,
                "train_firms": int(len(centroid_ids)),
                "evaluation_rows": int(len(eval_positions)),
                "top1_accuracy": float(np.mean(rank_array <= 1)),
                "top5_accuracy": float(np.mean(rank_array <= 5)),
                "mean_reciprocal_rank": float(np.mean(1.0 / rank_array)),
                "median_rank": float(np.median(rank_array)),
            }
        )
    return pd.DataFrame(rows)


def history_placebo_tests(
    state: dict[str, Any],
    targets: pd.DataFrame,
    randomizations: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    keys = state["company_keys"]
    qa = state["qa"]
    counts = state["history_count"]
    raw_weight = state["raw_weight"]
    own_direction = state["direction"]
    sector_reference = sector_history_excluding_firm(
        keys,
        qa,
        counts,
    )
    sector_direction = normalize_with_missing(qa - sector_reference)
    target_eval = targets[
        targets["month"].ge(EVALUATION_START)
    ][["sector", "month", "excess_return"]]
    rows: list[dict[str, Any]] = []
    monthly_frames: list[pd.DataFrame] = []
    for label, values in (
        ("own_firm_history", own_direction),
        ("sector_history_excluding_firm", sector_direction),
    ):
        scored = fixed_head_sector_scores(keys, values, raw_weight).merge(
            target_eval,
            on=["sector", "month"],
            how="inner",
            validate="one_to_one",
        )
        scored = scored[scored["month"].ge(EVALUATION_START)]
        ic = monthly_rank_ic(scored)
        rows.append(summarize_ic(label, ic))
        monthly = ic.rename("rank_ic").rename_axis("month").reset_index()
        monthly["reference"] = label
        monthly_frames.append(monthly)
        if label != "own_firm_history":
            sector_keys, sector_values, _ = group_mean(
                keys,
                values,
                SECTOR_MONTH_KEYS,
            )
            refit_panel = build_panel(sector_keys, targets)
            refit_scored, _, _ = fit_linear_head(
                refit_panel,
                sector_values,
            )
            refit_scored = refit_scored[
                refit_scored["month"].ge(EVALUATION_START)
            ].copy()
            refit_scored["pred"] = refit_scored["pred_raw"]
            refit_ic = monthly_rank_ic(refit_scored)
            refit_label = f"{label}_refit"
            rows.append(summarize_ic(refit_label, refit_ic))
            refit_monthly = (
                refit_ic.rename("rank_ic")
                .rename_axis("month")
                .reset_index()
            )
            refit_monthly["reference"] = refit_label
            monthly_frames.append(refit_monthly)
    candidates, donor_audit = wrong_firm_candidate_values(
        keys,
        qa,
        counts,
        raw_weight,
    )
    draws, draw_summary = placebo_draw_scores(
        keys,
        candidates,
        targets,
        randomizations,
    )
    own_ic = float(
        pd.DataFrame(rows)
        .set_index("test")
        .loc["own_firm_history", "mean_ic"]
    )
    draw_summary["own_history_ic"] = own_ic
    draw_summary["upper_tail_p"] = float(
        (1 + draws["mean_rank_ic"].ge(own_ic).sum()) / (len(draws) + 1)
    )
    rows.append(
        {
            "test": "same_sector_wrong_firm_placebo_mean",
            "months": int(draws["months"].median()),
            "mean_ic": float(draws["mean_rank_ic"].mean()),
            "newey_west_p": float("nan"),
            "positive_month_share": float(draws["mean_rank_ic"].gt(0).mean()),
        }
    )
    identity = identity_probe(keys, qa, own_direction)
    return (
        pd.DataFrame(rows),
        pd.concat(monthly_frames, ignore_index=True),
        draws,
        {
            "donors": donor_audit,
            "draws": draw_summary,
            "identity": identity.to_dict(orient="records"),
        },
    )


def write_report(
    output_dir: Path,
    structured: pd.DataFrame,
    placebos: pd.DataFrame,
    identity: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        printable = frame.copy()
        for column in printable.select_dtypes(include=[np.number]).columns:
            printable[column] = printable[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.4f}"
            )
        headers = [str(column) for column in printable.columns]
        rows = [
            [str(value) for value in row]
            for row in printable.itertuples(index=False, name=None)
        ]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(lines)

    lines = [
        "# CallRank Identification Expansion",
        "",
        "## Structured-news tests",
        "",
        markdown_table(structured),
        "",
        "The consensus-proxy rows are diagnostic because the licensed snapshot "
        "does not establish historical estimate vintage. Seasonal-PIT proxies "
        "use current and prior reported actuals only.",
        "",
        "## Own-history placebos",
        "",
        markdown_table(placebos),
        "",
        "## Issuer identity retrieval",
        "",
        markdown_table(identity),
        "",
        "## Key audit",
        "",
        f"- Authoritative score max difference: "
        f"{summary['authoritative_verification']['max_prediction_difference']:.3e}",
        f"- Wrong-firm placebo p-value: "
        f"{summary['history_placebo']['draws']['upper_tail_p']:.4f}",
        f"- Estimate-vintage status: "
        f"{summary['structured_news']['data']['consensus_vintage_status']}",
        "",
    ]
    (output_dir / "IDENTIFICATION_EXPANSION_RESULTS.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    global EVENT_MAP
    global TARGETS
    global AUTHORITATIVE_PREDICTIONS
    global REPORTED_EARNINGS
    global EMBEDDING_DIRS

    args = parse_args()
    EVENT_MAP = args.event_map.expanduser().resolve()
    TARGETS = args.targets.expanduser().resolve()
    AUTHORITATIVE_PREDICTIONS = (
        args.authoritative_predictions.expanduser().resolve()
    )
    REPORTED_EARNINGS = args.earnings.expanduser().resolve()
    EMBEDDING_DIRS = {
        128: args.embedding_128.expanduser().resolve(),
        160: args.embedding_160.expanduser().resolve(),
        200: args.embedding_200.expanduser().resolve(),
    }
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    events = clean_event_map()
    targets = load_targets(TARGETS)
    reconstructed, states = build_callrank_scores(events, targets)
    authoritative_audit = verify_authoritative_scores(reconstructed)

    structured, structured_monthly, structured_audit = structured_news_tests(
        reconstructed,
        states,
        targets,
        events,
    )
    (
        placebo_summary,
        placebo_monthly,
        placebo_draws,
        placebo_audit,
    ) = history_placebo_tests(
        states[160],
        targets,
        int(args.placebo_randomizations),
    )
    identity = identity_probe(
        states[160]["company_keys"],
        states[160]["qa"],
        states[160]["direction"],
    )

    structured.to_csv(output_dir / "structured_news_tests.csv", index=False)
    structured_monthly.to_csv(
        output_dir / "structured_news_monthly.csv", index=False
    )
    placebo_summary.to_csv(
        output_dir / "history_reference_tests.csv", index=False
    )
    placebo_monthly.to_csv(
        output_dir / "history_reference_monthly.csv", index=False
    )
    placebo_draws.to_csv(
        output_dir / "wrong_firm_placebo_draws.csv", index=False
    )
    identity.to_csv(output_dir / "issuer_identity_probe.csv", index=False)
    reconstructed.to_parquet(
        output_dir / "reconstructed_callrank_scores.parquet",
        index=False,
    )
    summary = {
        "status": "PASS",
        "research_boundary": (
            "Post-hoc identification diagnostics with the reported signal, head, "
            "training boundary, target, and evaluation period held fixed."
        ),
        "authoritative_verification": authoritative_audit,
        "structured_news": structured_audit,
        "history_placebo": placebo_audit,
        "input_hashes": {
            "event_map": sha256_file(EVENT_MAP),
            "targets": sha256_file(TARGETS),
            "authoritative_predictions": sha256_file(AUTHORITATIVE_PREDICTIONS),
            "reported_earnings": sha256_file(REPORTED_EARNINGS),
        },
    }
    (output_dir / "identification_expansion_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_report(
        output_dir,
        structured,
        placebo_summary,
        identity,
        summary,
    )
    print(structured.to_string(index=False))
    print()
    print(placebo_summary.to_string(index=False))
    print()
    print(identity.to_string(index=False))
    print()
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
