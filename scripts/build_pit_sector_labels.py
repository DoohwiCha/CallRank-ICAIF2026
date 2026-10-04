#!/usr/bin/env python3
"""Apply dated index membership and dated or reconstructed sector labels."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import SECTOR_ETF, read_table, write_table


REAL_ESTATE_EFFECTIVE = pd.Timestamp("2016-09-19")
COMMUNICATION_EFFECTIVE = pd.Timestamp("2018-09-24")
GICS_2023_EFFECTIVE = pd.Timestamp("2023-03-20")

TELECOM_INDUSTRIES = {
    "Alternative Carriers",
    "Integrated Telecommunication Services",
    "Wireless Telecommunication Services",
}
PRE_2018_TECH = {"ATVI", "EA", "GOOG", "GOOGL", "META", "TTWO", "TWTR"}
PRE_2018_DISCRETIONARY = {
    "CHTR", "CMCSA", "DIS", "IPG", "LYV", "MTCH", "NFLX", "NWS", "NWSA",
    "OMC", "PARA", "PSKY", "WBD",
}
PRE_2023_PAYMENTS = {
    "ADP", "BR", "CPAY", "FI", "FIS", "FISV", "GPN", "JKHY", "MA", "PAYX",
    "PYPL", "V",
}
PRE_2023_RETAIL = {"DG", "DLTR", "TGT"}


def base_symbol(symbol: str) -> str:
    return re.sub(r"-\d{6}$", "", str(symbol).upper()).replace(".", "-")


def reconstruct_sector(
    ticker: str,
    current_sector: str,
    industry: str,
    date: pd.Timestamp,
) -> str:
    symbol = str(ticker).upper()
    base = base_symbol(symbol)
    sector = str(current_sector)
    if symbol == "JCI-201609":
        return "Industrials"
    if date < REAL_ESTATE_EFFECTIVE and sector == "Real Estate":
        return "Financials"
    if date < COMMUNICATION_EFFECTIVE and sector == "Communication Services":
        if base in PRE_2018_DISCRETIONARY:
            return "Consumer Discretionary"
        if str(industry) in TELECOM_INDUSTRIES:
            return "Telecommunication Services"
        if base in PRE_2018_TECH:
            return "Information Technology"
        return "Consumer Discretionary"
    if date < GICS_2023_EFFECTIVE:
        if sector in {"Financials", "Industrials"} and base in PRE_2023_PAYMENTS:
            return "Information Technology"
        if sector == "Consumer Staples" and base in PRE_2023_RETAIL:
            return "Consumer Discretionary"
    return sector


def attach_historical_sectors(
    calls: pd.DataFrame, sector_history: pd.DataFrame
) -> pd.DataFrame:
    required = {"security_id", "effective_date", "sector"}
    missing = required.difference(sector_history.columns)
    if missing:
        raise ValueError(f"Missing historical sector columns: {sorted(missing)}")
    history = sector_history[list(required)].copy()
    history["effective_date"] = pd.to_datetime(history["effective_date"])
    if history.duplicated(["security_id", "effective_date"]).any():
        raise ValueError("Historical sector keys are not unique")
    left = calls.sort_values(["earnings_date", "security_id"]).copy()
    right = history.sort_values(["effective_date", "security_id"]).copy()
    output = pd.merge_asof(
        left,
        right,
        left_on="earnings_date",
        right_on="effective_date",
        by="security_id",
        direction="backward",
        allow_exact_matches=True,
    )
    if output["effective_date"].gt(output["earnings_date"]).any():
        raise RuntimeError("A sector classification follows its earnings call")
    output["sector_mapping_date"] = output["effective_date"]
    output["sector_mapping_source"] = "dated_sector_history"
    return output.drop(columns="effective_date")


def attach_membership(
    calls: pd.DataFrame,
    membership: pd.DataFrame,
    sector_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required_calls = {"security_id", "ticker", "earnings_date"}
    if sector_history is None:
        required_calls.update({"current_sector", "industry"})
    required_membership = {"security_id", "Date", "InIndex"}
    missing_calls = required_calls.difference(calls.columns)
    missing_membership = required_membership.difference(membership.columns)
    if missing_calls or missing_membership:
        raise ValueError(
            f"Missing call columns {sorted(missing_calls)} or membership columns {sorted(missing_membership)}"
        )
    output = calls.copy()
    output["earnings_date"] = pd.to_datetime(output["earnings_date"])
    history = membership.copy()
    history["Date"] = pd.to_datetime(history["Date"])
    histories = {
        security_id: group.sort_values("Date")
        for security_id, group in history.groupby("security_id", sort=False)
    }
    membership_dates: list[pd.Timestamp | pd.NaT] = []
    membership_flags: list[bool] = []
    for security_id, date in output[["security_id", "earnings_date"]].itertuples(
        index=False, name=None
    ):
        group = histories.get(security_id)
        if group is None:
            membership_dates.append(pd.NaT)
            membership_flags.append(False)
            continue
        dates = group["Date"].to_numpy(dtype="datetime64[ns]")
        position = int(np.searchsorted(dates, np.datetime64(date), side="right") - 1)
        if position < 0:
            membership_dates.append(pd.NaT)
            membership_flags.append(False)
            continue
        membership_dates.append(pd.Timestamp(dates[position]))
        membership_flags.append(bool(group.iloc[position]["InIndex"]))
    output["membership_date"] = membership_dates
    output["in_index"] = membership_flags
    output = output[output["in_index"] & output["membership_date"].notna()].copy()
    if sector_history is not None:
        output = attach_historical_sectors(output, sector_history)
    else:
        output["sector"] = [
            reconstruct_sector(ticker, sector, industry, date)
            for ticker, sector, industry, date in output[
                ["ticker", "current_sector", "industry", "earnings_date"]
            ].itertuples(index=False, name=None)
        ]
        output["sector_mapping_date"] = pd.NaT
        output["sector_mapping_source"] = "broad_gics_reconstruction"
    # The predecessor telecom category has no matching sector ETF and is not
    # backfilled into Communication Services before its effective date.
    return output[output["sector"].isin(SECTOR_ETF)].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=Path, required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument(
        "--sector-history",
        type=Path,
        help="Optional dated security_id/effective_date/sector table",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    history = read_table(args.sector_history) if args.sector_history else None
    output = attach_membership(
        read_table(args.calls), read_table(args.membership), history
    )
    if output["membership_date"].gt(output["earnings_date"]).any():
        raise RuntimeError("Membership matching crossed the call date")
    write_table(output, args.output)
    print(f"Wrote {len(output)} point-in-time call rows")


if __name__ == "__main__":
    main()
