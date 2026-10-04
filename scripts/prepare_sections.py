#!/usr/bin/env python3
"""Split a licensed transcript snapshot into prepared and Q&A-side sections."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
SPEAKER_RE = re.compile(r"([A-Z][A-Za-z0-9 .,&'’()/\\-]{1,90})\s*:\s+")
OPERATOR_RE = re.compile(r"\boperator\b", re.I)
ANALYST_LABEL_RE = re.compile(r"\s+-\s+")
QA_PATTERNS = [
    re.compile(
        r"Operator\s*:\s*(?:Thank you[.,]\s*)?(?:Your|Our|The|And the)\s+"
        r"(?:first|next)\s+question",
        re.I,
    ),
    re.compile(
        r"Operator\s*:\s*(?:Thank you[.,]\s*)?(?:And\s+)?(?:the\s+)?"
        r"(?:first|next)?\s*question\s+(?:comes|is)",
        re.I,
    ),
    re.compile(r"\b(?:first|next)\s+question\s+comes\s+from\b", re.I),
    re.compile(r"\bquestion[-\s]and[-\s]answer session\b", re.I),
    re.compile(r"\bwe\s+(?:will\s+)?now\s+(?:begin|open).{0,80}question", re.I),
    re.compile(
        r"\b(?:happy|ready|pleased)\s+to\s+(?:take|answer)\s+(?:your\s+)?questions\b",
        re.I,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".gz"}:
        return pd.read_csv(path)
    raise ValueError("Input must be CSV or Parquet")


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
    elif path.suffix.lower() in {".csv", ".gz"}:
        frame.to_csv(path, index=False)
    else:
        raise ValueError("Output must be CSV or Parquet")


def word_count(text: str) -> int:
    return len(TOKEN_RE.findall(str(text)))


def find_qa_start(text: str) -> tuple[int | None, str]:
    minimum_position = max(1_500, int(len(text) * 0.20))
    candidates: list[tuple[int, str]] = []
    for pattern in QA_PATTERNS:
        for match in pattern.finditer(text):
            if match.start() >= minimum_position:
                candidates.append((match.start(), pattern.pattern[:44]))
    if candidates:
        return min(candidates)
    for match in SPEAKER_RE.finditer(text):
        if match.start() >= minimum_position and ANALYST_LABEL_RE.search(match.group(1)):
            return match.start(), "analyst_speaker_label_after_20pct"
    return None, "qa_marker_not_found"


def speaker_segments(text: str) -> list[tuple[str, str]]:
    matches = list(SPEAKER_RE.finditer(text))
    segments = []
    for index, match in enumerate(matches):
        speaker = re.sub(r"\s+", " ", match.group(1)).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            segments.append((speaker, body))
    return segments


def split_call(text: str) -> dict[str, Any]:
    qa_start, qa_method = find_qa_start(text)
    if qa_start is None:
        return {
            "prepared_text": text,
            "qa_text": "",
            "qa_start_char": -1,
            "qa_method": qa_method,
        }

    prepared_text = text[:qa_start]
    full_qa_text = text[qa_start:]
    answers: list[str] = []
    for speaker, body in speaker_segments(full_qa_text):
        if OPERATOR_RE.search(speaker) or ANALYST_LABEL_RE.search(speaker):
            continue
        answers.append(body)

    qa_text = " ".join(answers).strip()
    if word_count(qa_text) < 50 and word_count(full_qa_text) >= 50:
        qa_text = full_qa_text
        qa_method = f"{qa_method}|all_qa_fallback"
    return {
        "prepared_text": prepared_text,
        "qa_text": qa_text,
        "qa_start_char": qa_start,
        "qa_method": qa_method,
    }


def prepare_sections(transcripts: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        "security_id",
        "ticker",
        "sector",
        "earnings_date",
        "membership_date",
        "in_index",
        "content",
    }
    missing = required.difference(transcripts.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")

    frame = transcripts[list(required)].copy()
    frame["earnings_date"] = pd.to_datetime(frame["earnings_date"], errors="coerce")
    frame["membership_date"] = pd.to_datetime(frame["membership_date"], errors="coerce")
    frame["ticker"] = (
        frame["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False)
    )
    frame["content"] = frame["content"].fillna("").astype(str).str.replace("\ufeff", "")
    frame = frame.dropna(subset=["sector", "earnings_date", "membership_date"])
    if frame["membership_date"].gt(frame["earnings_date"]).any():
        raise ValueError("A membership observation follows its earnings call")
    if not frame["in_index"].astype(bool).all():
        raise ValueError("Input includes calls outside point-in-time membership")
    frame = frame[frame["content"].str.len().ge(1_000)]
    frame = (
        frame.sort_values(["ticker", "earnings_date"])
        .drop_duplicates(["security_id", "earnings_date"], keep="last")
        .reset_index(drop=True)
    )

    rows = []
    methods: dict[str, int] = {}
    for row in frame.itertuples(index=False):
        split = split_call(row.content)
        method = str(split["qa_method"])
        methods[method] = methods.get(method, 0) + 1
        rows.append(
            {
                "security_id": row.security_id,
                "ticker": row.ticker,
                "sector": row.sector,
                "earnings_date": row.earnings_date,
                "membership_date": row.membership_date,
                "in_index": bool(row.in_index),
                "month": row.earnings_date.to_period("M").to_timestamp(),
                "prepared_text": split["prepared_text"],
                "qa_text": split["qa_text"],
                "qa_start_char": split["qa_start_char"],
                "qa_method": method,
            }
        )

    output = pd.DataFrame(rows)
    audit = {
        "input_rows": int(len(transcripts)),
        "deduplicated_rows": int(len(frame)),
        "section_rows": int(len(output)),
        "qa_method_counts": methods,
    }
    return output, audit


def main() -> None:
    args = parse_args()
    sections, audit = prepare_sections(read_table(args.input))
    write_table(sections, args.output)
    if args.audit:
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Wrote {len(sections)} licensed section rows to {args.output}")


if __name__ == "__main__":
    main()
