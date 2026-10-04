#!/usr/bin/env python3
"""Firm-conditioned Q&A representations used by CallRank."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import embedding_columns, l2_normalize, read_table


CALL_KEYS = ["issuer_call_id", "issuer_id", "sector", "month"]
FIRM_MONTH_KEYS = ["issuer_id", "sector", "month"]
SECTOR_MONTH_KEYS = ["sector", "month"]
UNIT_DIRECTION = "firm_conditioned_qa_direction"
RAW_DISPLACEMENT = "firm_conditioned_qa_raw"


def normalize_with_missing(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    output = np.full_like(matrix, np.nan)
    valid = np.isfinite(matrix).all(axis=1) & (norms[:, 0] > 1e-12)
    output[valid] = matrix[valid] / norms[valid]
    return output


def group_mean(
    keys: pd.DataFrame,
    values: np.ndarray,
    group_columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    work = keys[group_columns].copy().reset_index(drop=True)
    groups = work.groupby(group_columns, sort=True, observed=True).indices
    key_rows: list[dict[str, Any]] = []
    means: list[np.ndarray] = []
    counts: list[int] = []
    for group_key in sorted(groups):
        positions = np.asarray(groups[group_key], dtype=np.int64)
        local = values[positions]
        valid = np.isfinite(local).all(axis=1)
        if valid.any():
            means.append(local[valid].mean(axis=0))
            counts.append(int(valid.sum()))
        else:
            means.append(np.full(values.shape[1], np.nan))
            counts.append(0)
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        key_rows.append(dict(zip(group_columns, group_key, strict=True)))
    return pd.DataFrame(key_rows), np.vstack(means), np.asarray(counts, dtype=int)


def prior_residual(
    values: np.ndarray,
    identifiers: pd.Series,
    dates: pd.Series,
    minimum_prior: int,
) -> np.ndarray:
    output = np.full_like(values, np.nan, dtype=np.float64)
    order = pd.DataFrame(
        {
            "identifier": identifiers.astype(str).to_numpy(),
            "date": pd.to_datetime(dates).to_numpy(),
            "position": np.arange(len(values), dtype=int),
        }
    ).sort_values(["identifier", "date", "position"], kind="mergesort")
    for _, group in order.groupby("identifier", sort=False):
        cumulative = np.zeros(values.shape[1], dtype=np.float64)
        count = 0
        for position in group["position"].to_numpy(dtype=int):
            if count >= minimum_prior:
                output[position] = values[position] - cumulative / count
            cumulative += values[position]
            count += 1
    return output


def load_call_embeddings(
    prepared_path: Path,
    qa_path: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    prepared = read_table(prepared_path)
    qa = read_table(qa_path)
    required = set(CALL_KEYS)
    for label, frame in (("prepared", prepared), ("qa", qa)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{label} call keys are missing: {sorted(missing)}")
        frame["month"] = pd.to_datetime(frame["month"])
        frame.sort_values(CALL_KEYS, inplace=True)
        frame.reset_index(drop=True, inplace=True)
        if frame.duplicated(CALL_KEYS).any():
            raise ValueError(f"{label} call keys are not unique")
    if not prepared[CALL_KEYS].equals(qa[CALL_KEYS]):
        raise ValueError("Prepared and Q&A call keys differ")
    columns = embedding_columns(prepared)
    if columns != embedding_columns(qa):
        raise ValueError("Prepared and Q&A embedding coordinates differ")
    return (
        prepared[CALL_KEYS].copy(),
        prepared[columns].to_numpy(dtype=np.float64),
        qa[columns].to_numpy(dtype=np.float64),
    )


def company_month_sections(
    call_keys: pd.DataFrame,
    prepared: np.ndarray,
    qa: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    company_keys, prepared_mean, prepared_calls = group_mean(
        call_keys, prepared, FIRM_MONTH_KEYS
    )
    qa_keys, qa_mean, qa_calls = group_mean(
        call_keys, qa, FIRM_MONTH_KEYS
    )
    if not company_keys.equals(qa_keys):
        raise RuntimeError("Prepared and Q&A company-month keys differ")
    if not np.array_equal(prepared_calls, qa_calls):
        raise RuntimeError("Prepared and Q&A company-month call counts differ")
    return (
        company_keys,
        l2_normalize(prepared_mean),
        l2_normalize(qa_mean),
        prepared_calls,
    )


def firm_conditioned_representations(
    company_keys: pd.DataFrame,
    prepared: np.ndarray,
    qa: np.ndarray,
    *,
    minimum_prior: int = 2,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    raw_residual = prior_residual(
        qa,
        company_keys["issuer_id"],
        company_keys["month"],
        minimum_prior,
    )
    eligible = np.isfinite(raw_residual).all(axis=1)
    direction = normalize_with_missing(raw_residual)
    history = qa - raw_residual
    history_unit = normalize_with_missing(history)
    angle = np.arccos(
        np.clip(np.sum(qa * history_unit, axis=1), -1.0, 1.0)
    ).reshape(-1, 1)
    matched_qa = qa.copy()
    matched_prepared = prepared.copy()
    same_call = normalize_with_missing(qa - prepared)
    for values in (matched_qa, matched_prepared, same_call):
        values[~eligible] = np.nan

    direction_keys, sector_direction, firms = group_mean(
        company_keys, direction, SECTOR_MONTH_KEYS
    )
    candidates: dict[str, np.ndarray] = {UNIT_DIRECTION: sector_direction}
    specifications = {
        "firm_conditioned_qa_raw": raw_residual,
        "firm_conditioned_qa_distance": angle,
        "qa_level": matched_qa,
        "prepared_level": matched_prepared,
        "firm_history_level": history_unit,
        "same_call_firm_direction": same_call,
    }
    sector_values: dict[str, np.ndarray] = {}
    for name, values in specifications.items():
        keys, mean, counts = group_mean(
            company_keys, values, SECTOR_MONTH_KEYS
        )
        if not direction_keys.equals(keys) or not np.array_equal(firms, counts):
            raise RuntimeError(f"Sector aggregation differs for {name}")
        sector_values[name] = mean

    candidates.update(
        {
            "firm_conditioned_qa_raw": sector_values[
                "firm_conditioned_qa_raw"
            ],
            "firm_conditioned_qa_distance": sector_values[
                "firm_conditioned_qa_distance"
            ],
            "qa_level": normalize_with_missing(sector_values["qa_level"]),
            "prepared_level": normalize_with_missing(
                sector_values["prepared_level"]
            ),
            "qa_and_history_concat": np.hstack(
                [
                    normalize_with_missing(sector_values["qa_level"]),
                    normalize_with_missing(
                        sector_values["firm_history_level"]
                    ),
                ]
            ),
            "same_call_firm_direction": normalize_with_missing(
                sector_values["same_call_firm_direction"]
            ),
            "same_call_section_direction": normalize_with_missing(
                normalize_with_missing(sector_values["qa_level"])
                - normalize_with_missing(sector_values["prepared_level"])
            ),
        }
    )
    audit = direction_keys.copy()
    audit["eligible_firms"] = firms
    return direction_keys, candidates, audit
