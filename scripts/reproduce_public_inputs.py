#!/usr/bin/env python3
"""Refit the accepted main rankers from public numerical sufficient statistics.

The input boundary is author-generated sector-month Gram matrices, not raw
earnings-call text or issuer embeddings. Training labels are recomputed from
normalized total-return indices; PCA and Ridge are actually refitted. The
existing published result tables are used only as a final verification oracle.
No network, external data directory, saved predictions, or saved holdings are
needed. Optional output files are newly computed and are not model inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import linalg, stats

from common import SECTOR_ETF, load_config, monthly_rank_ic, rank_summary, rank_target
from portfolio import (ETFS, build_sleeves, daily_returns, momentum_scores,
                       prepare_prices, strategy_summary)

ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIONS = {"raw": "firm_conditioned_qa_raw", "unit": "firm_conditioned_qa_direction"}
KEY_COLUMNS = ["sector", "month", "entry_date", "exit_date", "excess_return"]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    return value


def csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def read_keys(path: Path) -> pd.DataFrame:
    frame = csv(path)
    if frame.columns.tolist() != KEY_COLUMNS:
        raise ValueError(f"Unexpected key schema: {path.name}")
    for column in ("month", "entry_date", "exit_date"):
        frame[column] = pd.to_datetime(frame[column])
    if frame.duplicated(["sector", "month"]).any():
        raise ValueError("Repeated sector-month key")
    return frame


def return_labels(keys: pd.DataFrame, prices: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Recompute target labels; check the supplied numerical snapshot labels."""
    frame = prepare_prices(prices).set_index("Date")
    dates = frame.index
    recomputed = []
    for row in keys.itertuples():
        entry = dates.get_loc(row.entry_date)
        exit_ = dates.get_loc(row.exit_date)
        if exit_ - entry != 42 or row.entry_date <= row.month + pd.offsets.MonthEnd(0):
            raise ValueError("Target horizon or post-month-end timing changed")
        etf = SECTOR_ETF[row.sector]
        r_sector = frame.loc[row.exit_date, etf] / frame.loc[row.entry_date, etf] - 1
        r_spy = frame.loc[row.exit_date, "SPY"] / frame.loc[row.entry_date, "SPY"] - 1
        recomputed.append(float(r_sector - r_spy))
    labels = np.asarray(recomputed)
    error = float(np.max(np.abs(labels - keys.excess_return.to_numpy(float))))
    if error > 2e-14:
        raise RuntimeError("Normalized indices do not reproduce supplied target labels")
    output = keys.copy()
    output["excess_return"] = labels
    return output, error


def spectral_basis(gram: np.ndarray, components: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    n = len(gram)
    if gram.shape != (n, n) or not np.isfinite(gram).all():
        raise ValueError("Invalid training Gram matrix")
    asymmetry = float(np.max(np.abs(gram - gram.T)))
    if asymmetry > 1e-9:
        raise ValueError("Training Gram matrix is not symmetric")
    # Eigenvalue order is reversed so PC 1 has the largest variance. Eigenvector
    # signs cancel in the prediction expression, as in an ordinary PCA/Ridge
    # pipeline. The extra eigenvalue records separation at the retained cutoff.
    eigenvalues, vectors = linalg.eigh(gram, subset_by_index=[n - components - 1, n - 1])
    eigenvalues, vectors = eigenvalues[::-1], vectors[:, ::-1]
    smallest = float(linalg.eigh(gram, subset_by_index=[0, 0], eigvals_only=True)[0])
    if smallest < -1e-8 * max(1.0, float(eigenvalues[0])):
        raise ValueError("Training Gram matrix is not numerically positive semidefinite")
    retained, basis = eigenvalues[:components], vectors[:, :components]
    if retained[-1] <= 0:
        raise ValueError("Insufficient positive principal components")
    diagnostic = {"train_rows": n, "retained_components": components,
                  "minimum_eigenvalue": smallest, "symmetry_max_error": asymmetry,
                  "largest_eigenvalue": float(retained[0]),
                  "last_retained_eigenvalue": float(retained[-1]),
                  "first_excluded_eigenvalue": float(eigenvalues[components]),
                  "explained_variance_fraction": float(retained.sum() / np.trace(gram))}
    return retained, basis, diagnostic


def ridge_predictions(cross: np.ndarray, eigenvalues: np.ndarray, basis: np.ndarray,
                      labels: np.ndarray, alpha: float) -> np.ndarray:
    """Exact dual form of a PCA-score Ridge fit with an unpenalized intercept.

    Z_train = U sqrt(L), Z_eval = G_eval_train U / sqrt(L).
    Multiplying the Ridge coefficient cancels sqrt(L), giving this expression.
    """
    mean = float(labels.mean())
    coefficients = (basis.T @ (labels - mean)) / (eigenvalues + alpha)
    return cross @ basis @ coefficients + mean


def average_rank_ic(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    valid = np.isfinite(predictions) & np.isfinite(targets)
    count = valid.sum(axis=-1)
    rp = stats.rankdata(np.where(valid, predictions, np.inf), axis=-1, method="average")
    rt = stats.rankdata(np.where(valid, targets, np.inf), axis=-1, method="average")
    centre = (count + 1) / 2
    xp = np.where(valid, rp - centre[..., None], 0.)
    xt = np.where(valid, rt - centre[..., None], 0.)
    denominator = np.sqrt((xp * xp).sum(axis=-1) * (xt * xt).sum(axis=-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (xp * xt).sum(axis=-1) / denominator
    return np.where((count >= 6) & (denominator > 0), result, np.nan)


def calendar_null(scored: pd.DataFrame, draws: int, seed: int) -> dict[str, Any]:
    p = scored.pivot(index="month", columns="sector", values="pred").sort_index()
    y = scored.pivot(index="month", columns="sector", values="excess_return").reindex_like(p)
    values, targets = p.to_numpy(), y.to_numpy()
    observed = float(np.nanmean(average_rank_ic(values, targets)))
    groups = [(j, np.flatnonzero(np.isfinite(values[:, j]))) for j in range(values.shape[1])]
    groups = [(j, ids) for j, ids in groups if len(ids) > 1]
    rng = np.random.default_rng(seed)
    offsets = rng.integers(1, np.array([len(ids) for _, ids in groups]), size=(draws, len(groups)))
    null = np.empty(draws)
    for first in range(0, draws, 1000):
        last = min(first + 1000, draws)
        shifted = np.broadcast_to(values, (last - first, *values.shape)).copy()
        for column, (j, ids) in enumerate(groups):
            lookup = (np.arange(len(ids))[None, :] - offsets[first:last, column, None]) % len(ids)
            shifted[:, ids, j] = values[ids, j][lookup]
        null[first:last] = np.nanmean(average_rank_ic(shifted, targets[None, :, :]), axis=1)
    return {"observed_rank_ic": observed, "randomizations": draws,
            "seed": seed, "null_mean": float(null.mean()),
            "upper_tail_p": float((1 + (null >= observed).sum()) / (draws + 1))}


def spread_sleeves(top: list[dict], bottom: list[dict]) -> list[dict]:
    by_month = {s["month"]: s for s in bottom}
    output = []
    for long in top:
        short = by_month[long["month"]]
        sleeve = dict(long)
        sleeve["id"] = f"spread-{long['id']}"
        sleeve["weights"] = {t: long["weights"][t] - short["weights"][t] for t in ETFS}
        output.append(sleeve)
    return output


def equal_universe_sleeves(scored: pd.DataFrame, prices: pd.DataFrame) -> list[dict]:
    px = prepare_prices(prices)
    indices = {pd.Timestamp(d): i for i, d in enumerate(px.Date)}
    output = []
    for month, group in scored.groupby("month", sort=True):
        if len(group) < 6:
            continue
        entry = indices[pd.Timestamp(group.entry_date.iloc[0])]
        output.append({"id": str(month.date()), "month": month,
                       "entry_index": entry, "exit_index": entry + 42,
                       "entry_date": px.loc[entry, "Date"], "exit_date": px.loc[entry + 42, "Date"],
                       "weights": {t: 1 / len(ETFS) for t in ETFS}})
    return output


def low_vol_scores(scored: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    px = prepare_prices(prices)
    indices = {pd.Timestamp(d): i for i, d in enumerate(px.Date)}
    pieces = []
    for _, group in scored.groupby("month", sort=True):
        end = indices[pd.Timestamp(group.entry_date.iloc[0])] - 1
        if end < 63:
            raise ValueError("Insufficient pre-entry volatility history")
        group = group.copy()
        group["pred"] = [-float(px.loc[end - 63:end, SECTOR_ETF[s]].astype(float).pct_change().dropna().std(ddof=1)) for s in group.sector]
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def factor_fit(daily: pd.DataFrame, factors: pd.DataFrame, model: str,
               lag: int, spread: bool) -> dict[str, Any]:
    frame = daily[["date", "net_return"]].merge(factors, on="date", validate="one_to_one")
    columns = ["MKT_RF"] if model == "capm" else ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]
    frame = frame.dropna(subset=["net_return", "RF", *columns])
    y = frame.net_return.to_numpy() - (0 if spread else frame.RF.to_numpy())
    fit = sm.OLS(y, sm.add_constant(frame[columns])).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    return {"model": model, "maxlags": lag, "observations": len(frame),
            "annualized_alpha": float((1 + fit.params["const"]) ** 252 - 1),
            "alpha_p": float(fit.pvalues["const"]), "market_beta": float(fit.params["MKT_RF"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "public_inputs")
    parser.add_argument("--output-dir", type=Path,
                        help="Save newly computed scores/portfolios/checks here; default prints the report only.")
    args = parser.parse_args()
    inputs = args.input_dir.resolve()
    manifest = json.loads((inputs / "manifest.json").read_text())
    for name, info in manifest["files"].items():
        if sha(inputs / name) != info["sha256"]:
            raise RuntimeError(f"Public input hash mismatch: {name}")
    config = load_config()
    settings = manifest["fixed_settings"]
    if (settings["pca_components"] != config["ranking_head"]["pca_components"]
            or settings["ridge_alpha"] != config["ranking_head"]["ridge_alpha"]
            or settings["one_way_cost_bps"] != config["portfolio"]["one_way_cost_bps"]):
        raise ValueError("Frozen public-input and public-code settings differ")
    train = read_keys(inputs / "train_keys.csv")
    evaluate = read_keys(inputs / "eval_keys.csv")
    if len(train) != 703 or len(evaluate) != 265 or not train.exit_date.le("2020-12-01").all() or not evaluate.month.ge("2023-11-01").all():
        raise ValueError("Accepted training/evaluation boundary changed")
    prices = csv(inputs / "sector_total_return_indices.csv")
    prices["Date"] = pd.to_datetime(prices.Date)
    train, train_target_error = return_labels(train, prices)
    evaluate, eval_target_error = return_labels(evaluate, prices)
    y = rank_target(train)
    components, alpha = int(settings["pca_components"]), float(settings["ridge_alpha"])
    pieces, models = [], []
    perturbation = None
    for short, candidate in REPRESENTATIONS.items():
        for chunk in (128, 160, 200):
            with np.load(inputs / f"{short}_{chunk}.npz", allow_pickle=False) as arrays:
                if set(arrays.files) != {"train_gram", "eval_train_gram"}:
                    raise ValueError("Unexpected sufficient-statistics arrays")
                gram, cross = arrays["train_gram"], arrays["eval_train_gram"]
            if gram.dtype != np.float64 or cross.dtype != np.float64 or cross.shape != (265, 703) or not np.isfinite(cross).all():
                raise ValueError("Invalid Gram data type/shape")
            eigenvalues, basis, diagnostic = spectral_basis(gram, components)
            pred = ridge_predictions(cross, eigenvalues, basis, y, alpha)
            scored = evaluate.copy()
            scored["pred"] = pred
            scored.insert(0, "chunk_words", chunk)
            scored.insert(0, "candidate", candidate)
            pieces.append(scored)
            models.append({"candidate": candidate, "passage_words": chunk, "ridge_alpha": alpha, **diagnostic})
            if perturbation is None:
                permuted_y = np.random.default_rng(20261004).permutation(y)
                permuted = ridge_predictions(cross, eigenvalues, basis, permuted_y, alpha)
                stronger_penalty = ridge_predictions(cross, eigenvalues, basis, y, alpha * 2)
                perturbation = {"candidate": candidate, "passage_words": chunk,
                                "label_permutation_prediction_l2_change": float(np.linalg.norm(permuted - pred)),
                                "doubled_penalty_prediction_l2_change": float(np.linalg.norm(stronger_penalty - pred)),
                                "purpose": "Demonstrate computed fits respond to training labels and penalty; perturbations are never used to choose reported settings."}
                if min(perturbation["label_permutation_prediction_l2_change"], perturbation["doubled_penalty_prediction_l2_change"]) <= 1e-8:
                    raise RuntimeError("Fit did not respond to changed labels/penalty")
    by_chunk = pd.concat(pieces, ignore_index=True)
    by_chunk["pred_rank"] = by_chunk.groupby(["candidate", "chunk_words", "month"]).pred.rank(method="average", pct=True)
    ensemble = by_chunk.groupby(["candidate", "sector", "month"], sort=True).agg(
        entry_date=("entry_date", "first"), exit_date=("exit_date", "first"),
        excess_return=("excess_return", "first"), pred=("pred_rank", "mean")).reset_index()
    rank_rows, nulls = [], {}
    for candidate, scored in ensemble.groupby("candidate"):
        rank_rows.append({"candidate": candidate, **rank_summary(scored)})
        nulls[candidate] = calendar_null(scored, int(config["evaluation"]["circular_shift_randomizations"]),
                                         int(config["evaluation"]["circular_shift_seed"]))
    rates = csv(inputs / "DGS3MO.csv")
    factors = csv(inputs / "daily_factors.csv")
    factors["date"] = pd.to_datetime(factors.date)
    # build_sleeves accepts a caps table even for equal weights. An empty schema
    # supplies no capitalization observations; a single selected ETF weighs 1.
    no_caps = pd.DataFrame(columns=["Date", "sector", "market_cap"])
    portfolios, daily_frames, factor_rows, weight_frames = [], [], [], []
    for candidate, scored in ensemble.groupby("candidate"):
        top, top_weights = build_sleeves(scored, prices, no_caps, side="top", count=1, weighting="equal")
        bottom, bottom_weights = build_sleeves(scored, prices, no_caps, side="bottom", count=1, weighting="equal")
        paths = {"top1": daily_returns(top, prices), "bottom1": daily_returns(bottom, prices),
                 "top1_minus_bottom1": daily_returns(spread_sleeves(top, bottom), prices)}
        if candidate == REPRESENTATIONS["unit"]:
            paths["one_over_n_11_sectors"] = daily_returns(equal_universe_sleeves(scored, prices), prices)
            for name, rule_scores in (("momentum_42d_top1", momentum_scores(scored, prices)),
                                      ("low_volatility_63d_top1", low_vol_scores(scored, prices))):
                sleeves, _ = build_sleeves(rule_scores, prices, no_caps, side="top", count=1, weighting="equal")
                paths[name] = daily_returns(sleeves, prices)
            spy_daily = paths["top1"].copy()
            spy_daily["gross_return"] = spy_daily["spy_return"]
            spy_daily["net_return"] = spy_daily["spy_return"]
            spy_daily["turnover"] = spy_daily["cost"] = 0.
            paths["spy"] = spy_daily
        for side, weights in (("top1", top_weights), ("bottom1", bottom_weights)):
            saved = weights.drop(columns="market_cap")
            saved.insert(0, "strategy", side)
            saved.insert(0, "candidate", candidate)
            weight_frames.append(saved)
        for strategy, daily in paths.items():
            summary = strategy_summary(daily, risk_free=rates)
            for lag in (5, 42):
                fit = sm.OLS(daily.net_return.to_numpy(), sm.add_constant(daily.spy_return.to_numpy())).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
                summary[f"spy_alpha_p_nw{lag}"] = float(fit.pvalues[0])
                if strategy in {"top1", "bottom1", "top1_minus_bottom1"}:
                    for model in ("capm", "ff5_mom"):
                        factor_rows.append({"candidate": candidate, "strategy": strategy,
                                            **factor_fit(daily, factors, model, lag, strategy == "top1_minus_bottom1")})
            portfolios.append({"candidate": candidate, "strategy": strategy, **summary})
            saved = daily.copy()
            saved.insert(0, "strategy", strategy)
            saved.insert(0, "candidate", candidate)
            daily_frames.append(saved)
    rank_table, portfolio_table, factor_table = map(pd.DataFrame, (rank_rows, portfolios, factor_rows))
    comparisons = []
    def compare(label: str, observed: float, expected: float, tolerance: float = 1e-9) -> None:
        comparisons.append({"endpoint": label, "observed": float(observed), "expected": float(expected),
                            "absolute_error": float(abs(observed - expected)),
                            "pass": bool(np.isclose(observed, expected, atol=tolerance, rtol=1e-8))})
    # Verification reads published aggregate summaries only after the fitted
    # model and portfolio paths have been computed from public numerical inputs.
    published = ROOT / "aggregate_results"
    expected_rank = csv(published / "representation_results.csv")
    expected_rank = expected_rank[expected_rank.resolution.eq("multiresolution")].set_index("candidate")
    for row in rank_table.itertuples():
        compare(f"{row.candidate}:rank_ic", row.rank_ic, expected_rank.loc[row.candidate, "rank_ic"])
        compare(f"{row.candidate}:rank_ic_hac_p", row.rank_ic_hac_p, expected_rank.loc[row.candidate, "rank_ic_hac_p"])
    expected_null = csv(published / "circular_shift_tests.csv").set_index("candidate")
    for candidate, row in nulls.items():
        compare(f"{candidate}:calendar_shift_p", row["upper_tail_p"], expected_null.loc[candidate, "upper_tail_p"])
    for short, candidate in REPRESENTATIONS.items():
        original = csv(published / ("portfolio_summary.csv" if short == "unit" else "raw_portfolio_summary.csv")).set_index("strategy")
        for row in portfolio_table[portfolio_table.candidate.eq(candidate) & portfolio_table.strategy.isin(["top1", "bottom1", "top1_minus_bottom1"])].itertuples():
            for field in ("annualized_return", "spy_return", "spy_alpha", "spy_alpha_p", "spy_beta", "sharpe", "mdd", "annualized_turnover", "annualized_cost"):
                compare(f"{candidate}:{row.strategy}:{field}", getattr(row, field), original.loc[row.strategy, field])
    baselines = csv(published / "baseline_portfolio_summary.csv").set_index("strategy")
    for row in portfolio_table[portfolio_table.strategy.isin(["one_over_n_11_sectors", "momentum_42d_top1", "low_volatility_63d_top1"])].itertuples():
        for field in ("annualized_return", "spy_alpha", "spy_alpha_p", "spy_beta", "sharpe", "mdd", "annualized_turnover"):
            compare(f"baseline:{row.strategy}:{field}", getattr(row, field), baselines.loc[row.strategy, field])
    original_factors = csv(published / "factor_regressions.csv")
    for row in factor_table[factor_table.candidate.eq(REPRESENTATIONS["unit"]) & factor_table.strategy.isin(["top1", "top1_minus_bottom1"])].itertuples():
        original = original_factors[original_factors.strategy.eq(row.strategy) & original_factors.model.eq(row.model) & original_factors.maxlags.eq(row.maxlags)].iloc[0]
        compare(f"unit:{row.strategy}:{row.model}:nw{row.maxlags}:alpha", row.annualized_alpha, original.annualized_alpha)
        compare(f"unit:{row.strategy}:{row.model}:nw{row.maxlags}:p", row.alpha_p, original.alpha_p)
    report = {"status": "PASS" if all(x["pass"] for x in comparisons) else "FAIL",
              "input_boundary": manifest["data_boundary"], "offline": True,
              "training_rows": len(train), "evaluation_sector_months": len(evaluate),
              "train_target_max_error": train_target_error, "eval_target_max_error": eval_target_error,
              "refitted_models": models, "perturbation_check": perturbation,
              "rank_summary": rank_rows, "calendar_shift_tests": nulls,
              "portfolio_summary": portfolios, "factor_regressions": factor_rows,
              "published_comparison": comparisons,
              "scope": "Six refitted raw/unit heads and their main rank/allocation endpoints, SPY, equal-sector, momentum and low-volatility benchmarks. LM text-tone and identity-control rebuilds still require their upstream licensed inputs; aggregate-result checks remain available separately."}
    if args.output_dir:
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        by_chunk.to_csv(output / "predictions_by_chunk.csv", index=False, float_format="%.17g")
        ensemble.to_csv(output / "predictions.csv", index=False, float_format="%.17g")
        rank_table.to_csv(output / "rank_summary.csv", index=False)
        portfolio_table.to_csv(output / "portfolio_summary.csv", index=False)
        factor_table.to_csv(output / "factor_regressions.csv", index=False)
        pd.concat(daily_frames).to_csv(output / "portfolio_daily.csv", index=False, float_format="%.17g")
        pd.concat(weight_frames).to_csv(output / "portfolio_weights.csv", index=False)
        (output / "replay_report.json").write_text(json.dumps(clean_json(report), indent=2, allow_nan=False) + "\n")
    print(json.dumps(clean_json(report), indent=2, allow_nan=False))
    if report["status"] != "PASS":
        raise RuntimeError("Public-input replay differs from published numerical results")


if __name__ == "__main__":
    main()
