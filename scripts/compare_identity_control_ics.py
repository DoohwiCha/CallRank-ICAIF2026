#!/usr/bin/env python3
"""Compare paired monthly rank ICs from identity-control prediction arms.

The input contains sector-month predictions after each text arm has passed
through the same embedding, firm-conditioning, and fixed-head pipeline.  This
script requires exact common sector-month support and identical return targets,
then reports each arm's monthly IC and the paired correct-minus-donor HAC gap.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


LOCAL_SCRIPTS = Path(__file__).resolve().parent
if str(LOCAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(LOCAL_SCRIPTS))

from common import (  # noqa: E402
    hac_pvalue,
    load_config,
    monthly_rank_ic,
    read_table,
    write_table,
)
from build_identity_body_placebo import CORRECT_ARM, DONOR_ARM  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--monthly-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--correct-arm", default=CORRECT_ARM)
    parser.add_argument("--donor-arm", default=DONOR_ARM)
    parser.add_argument("--arm-column", default="arm")
    parser.add_argument("--month-column", default="month")
    parser.add_argument("--sector-column", default="sector")
    parser.add_argument("--prediction-column", default="pred")
    parser.add_argument("--return-column", default="excess_return")
    parser.add_argument("--hac-lag", type=int, default=None)
    return parser.parse_args()


def canonical_predictions(
    predictions: pd.DataFrame,
    *,
    arm_column: str = "arm",
    month_column: str = "month",
    sector_column: str = "sector",
    prediction_column: str = "pred",
    return_column: str = "excess_return",
) -> pd.DataFrame:
    source = {
        "arm": arm_column,
        "month": month_column,
        "sector": sector_column,
        "pred": prediction_column,
        "excess_return": return_column,
    }
    missing = set(source.values()).difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction columns are missing: {sorted(missing)}")
    frame = predictions[list(source.values())].copy()
    frame.columns = list(source)
    frame["arm"] = frame["arm"].astype(str)
    frame["month"] = pd.to_datetime(frame["month"]).dt.to_period("M").dt.to_timestamp()
    frame["sector"] = frame["sector"].astype(str)
    frame["pred"] = pd.to_numeric(frame["pred"], errors="coerce")
    frame["excess_return"] = pd.to_numeric(frame["excess_return"], errors="coerce")
    if frame[["month", "sector", "pred", "excess_return"]].isna().any().any():
        raise ValueError("Predictions or target keys contain missing values")
    if frame.duplicated(["arm", "month", "sector"]).any():
        raise ValueError("Prediction keys are not unique within an arm")
    return frame


def paired_panels(
    frame: pd.DataFrame,
    correct_arm: str,
    donor_arm: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if correct_arm == donor_arm:
        raise ValueError("The two arm labels must differ")
    available = set(frame["arm"])
    missing = {correct_arm, donor_arm}.difference(available)
    if missing:
        raise ValueError(f"Prediction arms are missing: {sorted(missing)}")
    extra = available.difference({correct_arm, donor_arm})
    if extra:
        frame = frame[frame["arm"].isin([correct_arm, donor_arm])].copy()
    correct = frame[frame["arm"].eq(correct_arm)].copy()
    donor = frame[frame["arm"].eq(donor_arm)].copy()
    keys = ["month", "sector"]
    correct_keys = set(map(tuple, correct[keys].itertuples(index=False, name=None)))
    donor_keys = set(map(tuple, donor[keys].itertuples(index=False, name=None)))
    if correct_keys != donor_keys:
        raise ValueError("Prediction arms do not have exact common sector-month support")
    check = correct[keys + ["excess_return"]].merge(
        donor[keys + ["excess_return"]],
        on=keys,
        how="inner",
        suffixes=("_correct", "_donor"),
        validate="one_to_one",
    )
    if not np.allclose(
        check["excess_return_correct"],
        check["excess_return_donor"],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("Return targets differ across prediction arms")
    return correct, donor


def hac_mean_summary(values: pd.Series, lag: int) -> dict[str, float | int]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    output: dict[str, float | int] = {
        "months": int(len(clean)),
        "mean": float(clean.mean()) if len(clean) else float("nan"),
        "hac_se": float("nan"),
        "hac_two_sided_p": float("nan"),
        "ci95_low": float("nan"),
        "ci95_high": float("nan"),
    }
    if len(clean) < 8 or clean.std(ddof=1) <= 1.0e-12:
        return output
    fit = sm.OLS(clean.to_numpy(), np.ones((len(clean), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lag}
    )
    mean = float(fit.params[0])
    standard_error = float(fit.bse[0])
    critical = float(stats.norm.ppf(0.975))
    output.update(
        {
            "mean": mean,
            "hac_se": standard_error,
            "hac_two_sided_p": hac_pvalue(clean, lag),
            "ci95_low": mean - critical * standard_error,
            "ci95_high": mean + critical * standard_error,
        }
    )
    return output


def compare_monthly_ics(
    predictions: pd.DataFrame,
    *,
    correct_arm: str = CORRECT_ARM,
    donor_arm: str = DONOR_ARM,
    hac_lag: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    correct, donor = paired_panels(predictions, correct_arm, donor_arm)
    correct_ic = monthly_rank_ic(correct)
    donor_ic = monthly_rank_ic(donor)
    monthly = pd.concat(
        [
            correct_ic.rename("correct_body_ic"),
            donor_ic.rename("cross_sector_donor_body_ic"),
        ],
        axis=1,
    ).sort_index()
    monthly.index.name = "month"
    monthly["paired_gap"] = (
        monthly["correct_body_ic"] - monthly["cross_sector_donor_body_ic"]
    )
    monthly = monthly.reset_index()
    lag = (
        int(load_config()["evaluation"]["newey_west_rank_lag"])
        if hac_lag is None
        else int(hac_lag)
    )
    if lag < 0:
        raise ValueError("HAC lag must be nonnegative")
    records: list[dict[str, Any]] = []
    for label, column in (
        (correct_arm, "correct_body_ic"),
        (donor_arm, "cross_sector_donor_body_ic"),
        (f"{correct_arm}_minus_{donor_arm}", "paired_gap"),
    ):
        records.append(
            {
                "comparison": label,
                "hac_lag": lag,
                **hac_mean_summary(monthly[column], lag),
            }
        )
    return monthly, pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    predictions = canonical_predictions(
        read_table(args.predictions),
        arm_column=args.arm_column,
        month_column=args.month_column,
        sector_column=args.sector_column,
        prediction_column=args.prediction_column,
        return_column=args.return_column,
    )
    monthly, summary = compare_monthly_ics(
        predictions,
        correct_arm=args.correct_arm,
        donor_arm=args.donor_arm,
        hac_lag=args.hac_lag,
    )
    write_table(monthly, args.monthly_output)
    write_table(summary, args.summary_output)
    gap = summary.iloc[-1]
    print(
        "Paired correct-minus-donor IC gap: "
        f"{gap['mean']:.4f} (HAC p={gap['hac_two_sided_p']:.4g})"
    )


if __name__ == "__main__":
    main()
