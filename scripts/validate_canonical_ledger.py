#!/usr/bin/env python3
"""Validate the clean-room event map and its time-alignment invariants."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from common import load_config, read_table


REQUIRED = {
    "issuer_call_id",
    "canonical_issuer_id",
    "verified_call_date",
    "canonical_sector",
    "cleanroom_usable",
    "canonical_pit_membership_verified",
    "fiscal_year",
    "period",
    "content_sha256",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(
    rows: list[dict[str, Any]],
    label: str,
    observed: Any,
    expected: Any,
) -> None:
    rows.append(
        {
            "check": label,
            "status": "PASS" if observed == expected else "FAIL",
            "observed": observed,
            "expected": expected,
        }
    )


def validate_ledger(
    path: Path, *, enforce_frozen_hash: bool = True
) -> pd.DataFrame:
    config = load_config()["data_integrity"]
    ledger = read_table(path)
    missing = REQUIRED.difference(ledger.columns)
    if missing:
        raise ValueError(f"Ledger columns are missing: {sorted(missing)}")
    ledger["verified_call_date"] = pd.to_datetime(
        ledger["verified_call_date"]
    )
    usable = ledger[ledger["cleanroom_usable"].fillna(False)].copy()
    rows: list[dict[str, Any]] = []
    if enforce_frozen_hash:
        check(
            rows,
            "ledger_sha256",
            sha256(path),
            str(config["cleanroom_event_map_sha256"]),
        )
    check(rows, "ledger_rows", int(len(ledger)), int(config["ledger_rows"]))
    check(
        rows,
        "canonical_usable_calls",
        int(len(usable)),
        int(config["canonical_usable_calls"]),
    )
    check(
        rows,
        "canonical_issuers",
        int(usable["canonical_issuer_id"].nunique()),
        int(config["canonical_issuers"]),
    )
    duplicate_keys = {
        "duplicate:issuer_call_id": ["issuer_call_id"],
        "duplicate:issuer_content": [
            "canonical_issuer_id",
            "content_sha256",
        ],
        "duplicate:issuer_event_date": [
            "canonical_issuer_id",
            "verified_call_date",
        ],
    }
    for label, columns in duplicate_keys.items():
        check(
            rows,
            label,
            int(usable.duplicated(columns).sum()),
            0,
        )
    cross_issuer = (
        usable.groupby("content_sha256")["canonical_issuer_id"]
        .nunique()
        .gt(1)
        .sum()
    )
    check(
        rows,
        "cross_issuer:content_sha256",
        int(cross_issuer),
        0,
    )
    check(
        rows,
        "usable_verified_call_dates_missing",
        int(usable["verified_call_date"].isna().sum()),
        0,
    )
    check(
        rows,
        "usable_point_in_time_membership_failures",
        int(
            (~usable["canonical_pit_membership_verified"].fillna(False)).sum()
        ),
        0,
    )
    check(
        rows,
        "usable_sector_labels_missing",
        int(usable["canonical_sector"].fillna("").eq("").sum()),
        0,
    )
    gross_year_error = (
        usable["verified_call_date"]
        .dt.year.sub(
            pd.to_numeric(usable["fiscal_year"], errors="coerce")
        )
        .abs()
        .gt(1)
        .sum()
    )
    check(
        rows,
        "gross_event_fiscal_year_mismatch",
        int(gross_year_error),
        0,
    )
    invalid_period = int(
        (~usable["period"].isin({"Q1", "Q2", "Q3", "Q4"})).sum()
    )
    check(rows, "invalid_fiscal_period", invalid_period, 0)
    output = pd.DataFrame(rows)
    failed = output[output["status"].ne("PASS")]
    if len(failed):
        raise RuntimeError(
            "Canonical ledger validation failed:\n"
            + failed.to_string(index=False)
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--skip-frozen-hash",
        action="store_true",
        help="Run structural checks on a reconstructed ledger with different Parquet bytes.",
    )
    args = parser.parse_args()
    result = validate_ledger(
        args.ledger.resolve(),
        enforce_frozen_hash=not args.skip_frozen_hash,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
    print(
        json.dumps(
            {
                "status": "PASS",
                "checks": int(len(result)),
                "ledger": str(args.ledger.name),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
