#!/usr/bin/env python3
"""Collect identity-control rank ICs from three completed result directories.

Each directory must be a fresh ``run_core.py`` output containing row-level
multiresolution predictions.  The command enforces identical sector-month
support and targets, reports raw-displacement and unit-direction ICs, and
computes the HAC paired correct-body-minus-donor-body gap.  The masked arm's
already generated calendar-shift result is carried into the summary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd


LOCAL_SCRIPTS = Path(__file__).resolve().parent
if str(LOCAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(LOCAL_SCRIPTS))

from build_identity_body_placebo import CORRECT_ARM, DONOR_ARM  # noqa: E402
from callrank_features import RAW_DISPLACEMENT, UNIT_DIRECTION  # noqa: E402
from common import hac_pvalue, load_config, monthly_rank_ic, read_table, write_table  # noqa: E402
from compare_identity_control_ics import (  # noqa: E402
    canonical_predictions,
    compare_monthly_ics,
    paired_panels,
)


REPRESENTATIONS = {
    RAW_DISPLACEMENT: "raw_residual",
    UNIT_DIRECTION: "unit_direction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masked-results", type=Path, required=True)
    parser.add_argument("--correct-results", type=Path, required=True)
    parser.add_argument("--donor-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--monthly-output", type=Path, default=None)
    parser.add_argument("--hac-lag", type=int, default=None)
    return parser.parse_args()


def load_multiresolution_predictions(path: Path) -> pd.DataFrame:
    source = path / "row_level" / "multiresolution_predictions.parquet"
    if not source.is_file():
        raise FileNotFoundError(f"Missing completed prediction table: {source}")
    frame = read_table(source)
    required = {"candidate", "sector", "month", "pred", "excess_return"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Completed prediction columns are missing: {sorted(missing)}")
    return frame


def load_masked_shift_results(path: Path) -> pd.DataFrame:
    source = path / "representation_circular_shift_tests.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Missing calendar-shift summary: {source}")
    frame = read_table(source)
    required = {"candidate", "randomizations", "upper_tail_p"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Calendar-shift columns are missing: {sorted(missing)}")
    if frame["candidate"].duplicated().any():
        raise ValueError("Calendar-shift candidate keys are not unique")
    return frame.set_index("candidate")


def _arm_predictions(
    frame: pd.DataFrame,
    candidate: str,
    arm: str,
) -> pd.DataFrame:
    subset = frame[frame["candidate"].eq(candidate)][
        ["sector", "month", "pred", "excess_return"]
    ].copy()
    if subset.empty:
        raise ValueError(f"Completed results omit candidate: {candidate}")
    subset.insert(0, "arm", arm)
    return canonical_predictions(subset)


def collect_identity_results(
    masked: pd.DataFrame,
    correct: pd.DataFrame,
    donor: pd.DataFrame,
    shift: pd.DataFrame,
    *,
    hac_lag: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lag = (
        int(load_config()["evaluation"]["newey_west_rank_lag"])
        if hac_lag is None
        else int(hac_lag)
    )
    if lag < 0:
        raise ValueError("HAC lag must be nonnegative")
    summary_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    for candidate, representation in REPRESENTATIONS.items():
        masked_arm = _arm_predictions(masked, candidate, "full_pipeline_identity_mask")
        correct_arm = _arm_predictions(correct, candidate, CORRECT_ARM)
        donor_arm = _arm_predictions(donor, candidate, DONOR_ARM)
        paired_panels(
            pd.concat([masked_arm, correct_arm], ignore_index=True),
            "full_pipeline_identity_mask",
            CORRECT_ARM,
        )
        pair_input = pd.concat([correct_arm, donor_arm], ignore_index=True)
        pair_monthly, pair_summary = compare_monthly_ics(
            pair_input,
            correct_arm=CORRECT_ARM,
            donor_arm=DONOR_ARM,
            hac_lag=lag,
        )
        masked_ic = monthly_rank_ic(masked_arm)
        correct_ic = pair_monthly.set_index("month")["correct_body_ic"]
        donor_ic = pair_monthly.set_index("month")["cross_sector_donor_body_ic"]
        for test, values in (
            ("full_pipeline_identity_mask", masked_ic),
            ("correct_masked_body", correct_ic),
            ("wrong_body_donor", donor_ic),
        ):
            monthly_rows.extend(
                {
                    "test": test,
                    "representation": representation,
                    "month": pd.Timestamp(month),
                    "rank_ic": float(value),
                }
                for month, value in values.dropna().items()
            )
        if candidate not in shift.index:
            raise ValueError(f"Calendar-shift summary omits candidate: {candidate}")
        shifted = shift.loc[candidate]
        pair_lookup = pair_summary.set_index("comparison")
        correct_stats = pair_lookup.loc[CORRECT_ARM]
        donor_stats = pair_lookup.loc[DONOR_ARM]
        gap_stats = pair_lookup.loc[f"{CORRECT_ARM}_minus_{DONOR_ARM}"]
        summary_rows.extend(
            [
                {
                    "test": "full_pipeline_identity_mask",
                    "representation": representation,
                    "months": int(len(masked_ic)),
                    "estimate": float(masked_ic.mean()),
                    "estimate_kind": "mean_monthly_rank_ic",
                    "newey_west_lag": lag,
                    "newey_west_p": hac_pvalue(masked_ic, lag),
                    "calendar_shift_draws": int(shifted["randomizations"]),
                    "calendar_shift_p": float(shifted["upper_tail_p"]),
                },
                {
                    "test": "correct_masked_body",
                    "representation": representation,
                    "months": int(correct_stats["months"]),
                    "estimate": float(correct_stats["mean"]),
                    "estimate_kind": "mean_monthly_rank_ic",
                    "newey_west_lag": lag,
                    "newey_west_p": float(correct_stats["hac_two_sided_p"]),
                    "calendar_shift_draws": pd.NA,
                    "calendar_shift_p": pd.NA,
                },
                {
                    "test": "wrong_body_donor",
                    "representation": representation,
                    "months": int(donor_stats["months"]),
                    "estimate": float(donor_stats["mean"]),
                    "estimate_kind": "mean_monthly_rank_ic",
                    "newey_west_lag": lag,
                    "newey_west_p": float(donor_stats["hac_two_sided_p"]),
                    "calendar_shift_draws": pd.NA,
                    "calendar_shift_p": pd.NA,
                },
                {
                    "test": "paired_correct_minus_donor",
                    "representation": representation,
                    "months": int(gap_stats["months"]),
                    "estimate": float(gap_stats["mean"]),
                    "estimate_kind": "paired_monthly_rank_ic_gap",
                    "newey_west_lag": lag,
                    "newey_west_p": float(gap_stats["hac_two_sided_p"]),
                    "calendar_shift_draws": pd.NA,
                    "calendar_shift_p": pd.NA,
                },
            ]
        )
    summary = pd.DataFrame(summary_rows)
    order = {
        "full_pipeline_identity_mask": 0,
        "correct_masked_body": 1,
        "wrong_body_donor": 2,
        "paired_correct_minus_donor": 3,
    }
    summary["_order"] = summary["test"].map(order)
    summary.sort_values(["_order", "representation"], inplace=True)
    summary.drop(columns="_order", inplace=True)
    monthly = pd.DataFrame(monthly_rows).sort_values(
        ["test", "representation", "month"], kind="mergesort"
    )
    return summary.reset_index(drop=True), monthly.reset_index(drop=True)


def default_monthly_path(output: Path) -> Path:
    stem = output.stem
    base = stem[:-6] if stem.endswith("_tests") else stem
    return output.with_name(f"{base}_monthly_ic{output.suffix}")


def main() -> None:
    args = parse_args()
    summary, monthly = collect_identity_results(
        load_multiresolution_predictions(args.masked_results),
        load_multiresolution_predictions(args.correct_results),
        load_multiresolution_predictions(args.donor_results),
        load_masked_shift_results(args.masked_results),
        hac_lag=args.hac_lag,
    )
    monthly_output = args.monthly_output or default_monthly_path(args.output)
    write_table(summary, args.output)
    write_table(monthly, monthly_output)
    print(f"Wrote identity-control summary to {args.output}")
    print(f"Wrote monthly identity-control ICs to {monthly_output}")


if __name__ == "__main__":
    main()
