#!/usr/bin/env python3
"""Build the fixed forward sector-ETF excess-return target."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import SECTOR_ETF, load_config, read_table


def build_targets(prices: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    config = load_config()
    horizon = int(config["target"]["horizon_trading_days"])
    frame = prices.rename(columns={"index": "Date"}).copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.sort_values("Date").reset_index(drop=True)
    required = {"Date", "SPY", *SECTOR_ETF.values()}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Price columns are missing: {sorted(missing)}")
    month_keys = keys[["sector", "month"]].copy()
    month_keys["month"] = pd.to_datetime(month_keys["month"])
    month_keys["period"] = month_keys["month"].dt.to_period("M")
    month_keys = month_keys.drop_duplicates(["sector", "period"])
    rows: list[dict[str, object]] = []
    for period, group in month_keys.groupby("period", sort=True):
        month_positions = frame.index[frame["Date"].dt.to_period("M").eq(period)]
        if not len(month_positions):
            continue
        entry_index = int(month_positions.max()) + 1
        exit_index = entry_index + horizon
        if entry_index >= len(frame) or exit_index >= len(frame):
            continue
        entry_date = pd.Timestamp(frame.loc[entry_index, "Date"])
        exit_date = pd.Timestamp(frame.loc[exit_index, "Date"])
        for sector in sorted(group["sector"].astype(str).unique()):
            ticker = SECTOR_ETF.get(sector)
            if ticker is None:
                raise ValueError(f"Unknown sector: {sector}")
            values = frame.loc[[entry_index, exit_index], [ticker, "SPY"]]
            if values.isna().any().any():
                continue
            sector_growth = float(frame.loc[exit_index, ticker] / frame.loc[entry_index, ticker])
            spy_growth = float(frame.loc[exit_index, "SPY"] / frame.loc[entry_index, "SPY"])
            rows.append(
                {
                    "sector": sector,
                    "month": period.to_timestamp(),
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "horizon": horizon,
                    "excess_return": sector_growth - spy_growth,
                }
            )
    return pd.DataFrame(rows).sort_values(["month", "sector"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--months", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = build_targets(pd.read_csv(args.prices), read_table(args.months))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output)} target rows")


if __name__ == "__main__":
    main()
