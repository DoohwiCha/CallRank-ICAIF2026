#!/usr/bin/env python3
"""Shared transformations for the CallRank public release."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "frozen_config.json"

SECTOR_ETF = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Information Technology": "XLK",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".gz"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path.suffix}")


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
    elif path.suffix.lower() in {".csv", ".gz"}:
        frame.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported table format: {path.suffix}")


def l2_normalize(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    denominator = np.linalg.norm(matrix, axis=1, keepdims=True)
    denominator[denominator <= 1e-12] = 1.0
    return matrix / denominator


def embedding_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        column
        for column in frame.columns
        if len(column) == 4 and column.startswith("e") and column[1:].isdigit()
    )


def load_targets(path: Path) -> pd.DataFrame:
    frame = read_table(path).copy()
    required = {"sector", "month", "entry_date", "exit_date", "horizon", "excess_return"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Target columns are missing: {sorted(missing)}")
    for column in ("month", "entry_date", "exit_date"):
        frame[column] = pd.to_datetime(frame[column])
    horizon = int(load_config()["target"]["horizon_trading_days"])
    frame = frame[frame["horizon"].eq(horizon)].copy()
    if frame.duplicated(["sector", "month"]).any():
        raise ValueError("Target keys are not unique")
    return frame.sort_values(["month", "sector"]).reset_index(drop=True)


def build_panel(keys: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    key_frame = keys[["sector", "month"]].copy().reset_index(drop=True)
    if key_frame.duplicated(["sector", "month"]).any():
        raise ValueError("Embedding keys are not unique")
    output = key_frame.merge(
        targets,
        on=["sector", "month"],
        how="left",
        sort=False,
        validate="one_to_one",
    )
    # Feature matrices use the embedding-key row order. Preserve that order
    # even when a target is unavailable because an ETF has not started trading
    # or the 42-day forward window extends beyond the price snapshot.
    if not output[["sector", "month"]].equals(key_frame):
        raise RuntimeError("Target alignment changed the embedding row order")
    output["target_available"] = output["excess_return"].notna()
    return output


def rank_target(frame: pd.DataFrame) -> np.ndarray:
    ranked = frame.groupby("month")["excess_return"].rank(method="average", pct=True)
    return (ranked - 0.5).to_numpy(dtype=np.float64)


def fit_fixed_head(
    panel: pd.DataFrame,
    values: np.ndarray,
    *,
    evaluation_start: str,
    excluded_sector: str | None = None,
    pca_components: int | None = None,
    ridge_alpha: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = load_config()
    head = config["ranking_head"]
    latest_outcome = pd.Timestamp(head["latest_training_outcome"])
    start = pd.Timestamp(evaluation_start)
    usable = np.isfinite(values).all(axis=1) & panel["excess_return"].notna().to_numpy()
    train_mask = panel["exit_date"].le(latest_outcome).to_numpy() & usable
    if excluded_sector is not None:
        train_mask &= panel["sector"].ne(excluded_sector).to_numpy()
    score_mask = panel["month"].ge(start).to_numpy() & usable
    train_index = np.flatnonzero(train_mask)
    score_index = np.flatnonzero(score_mask)
    if len(train_index) < 3 or not len(score_index):
        raise ValueError("Training or evaluation sample is empty")

    scaler = StandardScaler()
    train_x = scaler.fit_transform(values[train_index])
    score_x = scaler.transform(values[score_index])
    components = head["pca_components"] if pca_components is None else pca_components
    pca: PCA | None = None
    if components is not False and components is not None:
        components = min(int(components), train_x.shape[0] - 1, train_x.shape[1])
        pca = PCA(
            n_components=components,
            svd_solver=str(head["pca_solver"]),
            random_state=int(head["pca_random_state"]),
        )
        train_x = pca.fit_transform(train_x)
        score_x = pca.transform(score_x)

    alpha = float(head["ridge_alpha"] if ridge_alpha is None else ridge_alpha)
    model = Ridge(alpha=alpha).fit(train_x, rank_target(panel.iloc[train_index]))
    scored = panel.iloc[score_index].copy()
    scored["pred"] = model.predict(score_x)
    metadata = {
        "train_rows": int(len(train_index)),
        "score_rows": int(len(score_index)),
        "latest_training_outcome": str(panel.iloc[train_index]["exit_date"].max().date()),
        "input_dimensions": int(values.shape[1]),
        "effective_features": int(train_x.shape[1]),
        "ridge_alpha": alpha,
        "pca_components": int(train_x.shape[1]) if pca is not None else None,
        "pca_explained_variance": (
            float(pca.explained_variance_ratio_.sum()) if pca is not None else None
        ),
    }
    return scored, metadata


def monthly_rank_ic(frame: pd.DataFrame) -> pd.Series:
    values: dict[pd.Timestamp, float] = {}
    minimum = int(load_config()["evaluation"]["minimum_sectors_per_month"])
    for month, group in frame.dropna(subset=["pred", "excess_return"]).groupby("month"):
        if len(group) < minimum:
            continue
        if group["pred"].nunique() < 2 or group["excess_return"].nunique() < 2:
            continue
        values[pd.Timestamp(month)] = float(
            stats.spearmanr(group["pred"], group["excess_return"]).statistic
        )
    return pd.Series(values, dtype=float).sort_index()


def hac_pvalue(values: pd.Series, maxlags: int) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(clean) < 8 or clean.std(ddof=1) <= 1e-12:
        return float("nan")
    fit = sm.OLS(clean.to_numpy(), np.ones((len(clean), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags}
    )
    return float(fit.pvalues[0])


def rank_summary(scored: pd.DataFrame) -> dict[str, Any]:
    values = monthly_rank_ic(scored)
    lag = int(load_config()["evaluation"]["newey_west_rank_lag"])
    return {
        "months": int(len(values)),
        "rank_ic": float(values.mean()),
        "rank_ic_hac_p": hac_pvalue(values, lag),
        "positive_month_share": float(values.gt(0.0).mean()),
    }


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
