#!/usr/bin/env python3
"""Build paired correct-body and return-blind donor-body text corpora.

Each target keeps the same visible company/ticker/speaker identity shell in
both arms.  The correct arm appends its own identity-masked Q&A body; the
placebo arm appends a deterministic same-month, cross-sector donor body.
Donors are assigned one-to-one by Q&A word-count proximity with a hash-based
tie breaker.  No return, target, price, or portfolio field enters matching.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


LOCAL_SCRIPTS = Path(__file__).resolve().parent
if str(LOCAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(LOCAL_SCRIPTS))

from common import read_table, save_json, write_table  # noqa: E402


CORRECT_ARM = "correct_body"
DONOR_ARM = "cross_sector_donor_body"
DONOR_ASSIGNMENT_PREFIX = "callrank-identity-placebo-v1"
DONOR_ASSIGNMENT_SEED = 20260804
MATCH_FIELDS = (
    "call_id",
    "issuer_id",
    "month",
    "sector",
    "masked_body",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--shell-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mapping-output", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--call-id-column", default="issuer_call_id")
    parser.add_argument("--issuer-column", default="issuer_id")
    parser.add_argument("--month-column", default="month")
    parser.add_argument("--sector-column", default="sector")
    parser.add_argument("--body-column", default="qa_text")
    parser.add_argument(
        "--match-word-count-column",
        required=True,
        help=(
            "Required masking-before capped-Q&A whitespace word count used "
            "only for donor matching."
        ),
    )
    parser.add_argument("--shell-call-id-column", default="issuer_call_id")
    parser.add_argument("--canonical-company-column", default="canonical_company")
    parser.add_argument("--canonical-ticker-column", default="canonical_query_symbol")
    parser.add_argument("--speaker-column", default="speaker_aliases")
    parser.add_argument("--seed", type=int, default=20260804)
    return parser.parse_args()


def canonical_calls(
    calls: pd.DataFrame,
    *,
    call_id_column: str = "issuer_call_id",
    issuer_column: str = "issuer_id",
    month_column: str = "month",
    sector_column: str = "sector",
    body_column: str = "qa_text",
    body_word_count_column: str | None = None,
) -> pd.DataFrame:
    source_columns = {
        "call_id": call_id_column,
        "issuer_id": issuer_column,
        "month": month_column,
        "sector": sector_column,
        "masked_body": body_column,
    }
    if body_word_count_column is not None:
        source_columns["body_words"] = body_word_count_column
    missing = set(source_columns.values()).difference(calls.columns)
    if missing:
        raise ValueError(f"Call columns are missing: {sorted(missing)}")
    frame = calls[list(source_columns.values())].copy()
    frame.columns = list(source_columns)
    if frame[["call_id", "issuer_id", "month", "sector"]].isna().any().any():
        raise ValueError("Matching keys contain missing values")
    frame["call_id"] = frame["call_id"].astype(str)
    frame["issuer_id"] = frame["issuer_id"].astype(str)
    frame["month"] = pd.to_datetime(frame["month"]).dt.to_period("M").dt.to_timestamp()
    frame["sector"] = frame["sector"].astype(str)
    frame["masked_body"] = frame["masked_body"].fillna("").astype(str)
    if frame["call_id"].duplicated().any():
        raise ValueError("Call identifiers must be unique")
    if frame["masked_body"].str.strip().eq("").any():
        raise ValueError("A masked Q&A body is empty")
    if body_word_count_column is None:
        frame["body_words"] = frame["masked_body"].str.split().str.len().astype(int)
    else:
        frame["body_words"] = pd.to_numeric(
            frame["body_words"], errors="raise"
        ).astype(int)
        if frame["body_words"].le(0).any():
            raise ValueError("A donor-matching word count is nonpositive")
    return frame.sort_values(["month", "call_id"], kind="mergesort").reset_index(drop=True)


def _jitter(target: str, donor: str) -> float:
    payload = f"{DONOR_ASSIGNMENT_PREFIX}|{target}|{donor}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value / float(2**64) * 1.0e-9


def match_donors(calls: pd.DataFrame, *, seed: int = 20260804) -> pd.DataFrame:
    if seed != DONOR_ASSIGNMENT_SEED:
        raise ValueError(
            f"The frozen donor assignment seed is {DONOR_ASSIGNMENT_SEED}"
        )
    required = {*MATCH_FIELDS, "body_words"}
    missing = required.difference(calls.columns)
    if missing:
        raise ValueError(f"Canonical call columns are missing: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for month, group in calls.groupby("month", sort=True):
        local = group.sort_values("call_id", kind="mergesort").reset_index(drop=True)
        if local["sector"].nunique() < 2:
            raise ValueError(f"Month {pd.Timestamp(month).date()} has fewer than two sectors")
        size = len(local)
        cost = np.full((size, size), 1.0e12, dtype=float)
        log_words = np.log1p(local["body_words"].to_numpy(dtype=float))
        for target_index, target in enumerate(local.itertuples(index=False)):
            for donor_index, donor in enumerate(local.itertuples(index=False)):
                eligible = (
                    target.call_id != donor.call_id
                    and target.issuer_id != donor.issuer_id
                    and target.sector != donor.sector
                )
                if eligible:
                    cost[target_index, donor_index] = (
                        abs(log_words[target_index] - log_words[donor_index])
                        + _jitter(target.call_id, donor.call_id)
                    )
        target_indices, donor_indices = linear_sum_assignment(cost)
        if len(target_indices) != size or (cost[target_indices, donor_indices] >= 1.0e11).any():
            raise ValueError(
                f"Month {pd.Timestamp(month).date()} has no one-to-one cross-sector assignment"
            )
        for target_index, donor_index in zip(target_indices, donor_indices, strict=True):
            target = local.iloc[int(target_index)]
            donor = local.iloc[int(donor_index)]
            rows.append(
                {
                    "month": pd.Timestamp(month),
                    "target_call_id": target["call_id"],
                    "target_issuer_id": target["issuer_id"],
                    "target_sector": target["sector"],
                    "target_body_words": int(target["body_words"]),
                    "donor_call_id": donor["call_id"],
                    "donor_issuer_id": donor["issuer_id"],
                    "donor_sector": donor["sector"],
                    "donor_body_words": int(donor["body_words"]),
                    "absolute_log1p_word_gap": float(
                        abs(log_words[target_index] - log_words[donor_index])
                    ),
                }
            )
    mapping = pd.DataFrame(rows).sort_values(
        ["month", "target_call_id"], kind="mergesort"
    ).reset_index(drop=True)
    if mapping.duplicated(["month", "target_call_id"]).any():
        raise RuntimeError("A target received more than one donor")
    if mapping.duplicated(["month", "donor_call_id"]).any():
        raise RuntimeError("A donor was reused within a month")
    if mapping["target_sector"].eq(mapping["donor_sector"]).any():
        raise RuntimeError("A donor shares the target sector")
    if mapping["target_issuer_id"].eq(mapping["donor_issuer_id"]).any():
        raise RuntimeError("A donor shares the target issuer")
    return mapping


def identity_shell(
    company: str,
    ticker: str,
    speakers: list[str],
) -> str:
    speaker_text = " | ".join(speakers) if speakers else "(no parsed speakers)"
    return (
        "[TARGET IDENTITY SHELL]\n"
        f"Canonical company: {company}\n"
        f"Canonical ticker: {ticker}\n"
        f"Current speakers (ordered unique): {speaker_text}\n"
        "[/TARGET IDENTITY SHELL]"
    )


def composite_document(shell: str, scrubbed_body: str) -> str:
    return f"{shell}\n\n[SCRUBBED Q&A BODY]\n{scrubbed_body}"


def shell_lookup_from_metadata(
    shell_metadata: pd.DataFrame,
) -> dict[str, tuple[str, str, list[str]]]:
    required = {"call_id", "canonical_company", "canonical_ticker", "speakers"}
    missing = required.difference(shell_metadata.columns)
    if missing:
        raise ValueError(f"Identity-shell metadata columns are missing: {sorted(missing)}")
    if shell_metadata["call_id"].astype(str).duplicated().any():
        raise ValueError("Identity-shell call identifiers are not unique")
    output: dict[str, tuple[str, str, list[str]]] = {}
    for row in shell_metadata.itertuples(index=False):
        if isinstance(row.speakers, (list, tuple)):
            source_speakers = list(row.speakers)
        else:
            source_speakers = str(row.speakers or "").split("|")
        speakers: list[str] = []
        for value in source_speakers:
            clean = " ".join(str(value).split())
            if clean and clean not in speakers:
                speakers.append(clean)
        company = " ".join(str(row.canonical_company).split())
        ticker = " ".join(str(row.canonical_ticker).split())
        if not company or not ticker:
            raise ValueError("A canonical identity-shell field is empty")
        output[str(row.call_id)] = (company, ticker, speakers)
    return output


def normalize_shell_metadata_input(
    frame: pd.DataFrame,
    *,
    call_id_column: str,
    company_column: str,
    ticker_column: str,
    speaker_column: str,
) -> pd.DataFrame:
    required = {call_id_column, company_column, ticker_column, speaker_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Identity-shell input columns are missing: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        raw = row[speaker_column]
        if isinstance(raw, (list, tuple)):
            speakers = list(raw)
        else:
            text = "" if pd.isna(raw) else str(raw).strip()
            if text.startswith("["):
                decoded = json.loads(text)
                if not isinstance(decoded, list):
                    raise ValueError("Speaker roster JSON must be a list")
                speakers = decoded
            else:
                speakers = text.split("|") if text else []
        rows.append(
            {
                "call_id": str(row[call_id_column]),
                "canonical_company": row[company_column],
                "canonical_ticker": row[ticker_column],
                "speakers": speakers,
            }
        )
    return pd.DataFrame(rows)


def build_paired_corpus(
    calls: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    shell_metadata: pd.DataFrame,
) -> pd.DataFrame:
    call_lookup = calls.set_index("call_id")
    if not call_lookup.index.is_unique:
        raise ValueError("Call identifiers must be unique")
    metadata_lookup = shell_lookup_from_metadata(shell_metadata)
    rows: list[dict[str, Any]] = []
    for match in mapping.itertuples(index=False):
        target = call_lookup.loc[str(match.target_call_id)]
        donor = call_lookup.loc[str(match.donor_call_id)]
        if str(match.target_call_id) not in metadata_lookup:
            raise ValueError("Identity-shell metadata omits a target call")
        shell_values = metadata_lookup[str(match.target_call_id)]
        shell = identity_shell(*shell_values)
        shell_hash = hashlib.sha256(shell.encode("utf-8")).hexdigest()
        shared = {
            "target_call_id": str(match.target_call_id),
            "target_issuer_id": str(match.target_issuer_id),
            "month": pd.Timestamp(match.month),
            "sector": str(match.target_sector),
            "identity_shell": shell,
            "identity_shell_sha256": shell_hash,
        }
        for arm, source_call_id, source in (
            (CORRECT_ARM, str(match.target_call_id), target),
            (DONOR_ARM, str(match.donor_call_id), donor),
        ):
            body = str(source["masked_body"])
            document = composite_document(shell, body)
            rows.append(
                {
                    **shared,
                    "arm": arm,
                    "body_source_call_id": source_call_id,
                    "body_source_issuer_id": str(source["issuer_id"]),
                    "body_source_sector": str(source["sector"]),
                    "document_text": document,
                    "document_sha256": hashlib.sha256(
                        document.encode("utf-8")
                    ).hexdigest(),
                }
            )
    output = pd.DataFrame(rows).sort_values(
        ["month", "target_call_id", "arm"], kind="mergesort"
    ).reset_index(drop=True)
    counts = output.groupby("target_call_id")["arm"].nunique()
    if not counts.eq(2).all():
        raise RuntimeError("Every target must have both text arms")
    shell_counts = output.groupby("target_call_id")["identity_shell_sha256"].nunique()
    if not shell_counts.eq(1).all():
        raise RuntimeError("Paired documents do not have identical identity shells")
    return output


def main() -> None:
    args = parse_args()
    calls = canonical_calls(
        read_table(args.input),
        call_id_column=args.call_id_column,
        issuer_column=args.issuer_column,
        month_column=args.month_column,
        sector_column=args.sector_column,
        body_column=args.body_column,
        body_word_count_column=args.match_word_count_column,
    )
    shell_metadata = normalize_shell_metadata_input(
        read_table(args.shell_metadata),
        call_id_column=args.shell_call_id_column,
        company_column=args.canonical_company_column,
        ticker_column=args.canonical_ticker_column,
        speaker_column=args.speaker_column,
    )
    mapping = match_donors(calls, seed=args.seed)
    corpus = build_paired_corpus(
        calls,
        mapping,
        shell_metadata=shell_metadata,
    )
    write_table(mapping, args.mapping_output)
    write_table(corpus, args.output)
    audit = {
        "calls": int(len(calls)),
        "months": int(calls["month"].nunique()),
        "paired_documents": int(len(corpus)),
        "donors_are_one_to_one_within_month": True,
        "all_donors_are_cross_sector": True,
        "all_donors_are_cross_issuer": True,
        "matching_fields": ["month", "sector", "issuer_id", "body_words", "call_id"],
        "matching_uses_outcomes": False,
        "mean_absolute_log1p_word_gap": float(mapping["absolute_log1p_word_gap"].mean()),
        "seed": int(args.seed),
    }
    if args.audit:
        save_json(args.audit, audit)
    print(f"Wrote {len(corpus):,} paired identity-shell documents to {args.output}")


if __name__ == "__main__":
    main()
