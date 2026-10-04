#!/usr/bin/env python3
"""Build call-level SEC-domain embeddings at the three reported resolutions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import l2_normalize, load_config, read_table, write_table


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
KEY_COLUMNS = [
    "issuer_call_id",
    "issuer_id",
    "ticker",
    "sector",
    "earnings_date",
    "month",
]
IDENTITY_CONTROL_MARKER_COLUMN = "identity_control_protocol"
IDENTITY_CONTROL_MARKER_VALUE = "capped_then_identity_transformed_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--preprocessed-identity-control",
        action="store_true",
        help=(
            "Validate a marked identity-control table and chunk its prepared/Q&A "
            "text without applying the ordinary length filter or character cap again."
        ),
    )
    return parser.parse_args()


def word_count(value: Any) -> int:
    return len(TOKEN_RE.findall(str(value or "")))


def cap_then_clean(value: Any, character_cap: int) -> str:
    capped = str(value or "")[:character_cap]
    return re.sub(r"\s+", " ", capped).strip()


def call_id(issuer_id: Any, date: pd.Timestamp) -> str:
    payload = f"{issuer_id}|{date.date()}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def prepare_calls(source: pd.DataFrame) -> pd.DataFrame:
    required = {
        "security_id",
        "ticker",
        "sector",
        "earnings_date",
        "membership_date",
        "in_index",
        "prepared_text",
        "qa_text",
    }
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Input columns are missing: {sorted(missing)}")
    config = load_config()["text_construction"]
    frame = source.copy()
    frame["earnings_date"] = pd.to_datetime(frame["earnings_date"])
    frame["membership_date"] = pd.to_datetime(frame["membership_date"])
    if frame["membership_date"].gt(frame["earnings_date"]).any():
        raise ValueError("A membership observation follows its call")
    if not frame["in_index"].astype(bool).all():
        raise ValueError("Input contains a call outside dated membership")
    minimum = config["minimum_words_before_cap"]
    frame = frame[
        frame["prepared_text"]
        .map(word_count)
        .ge(int(minimum["prepared_remarks"]))
        & frame["qa_text"].map(word_count).ge(int(minimum["qa_side"]))
    ].copy()
    caps = config["per_call_character_caps"]
    frame["prepared_text"] = frame["prepared_text"].map(
        lambda value: cap_then_clean(
            value, int(caps["prepared_remarks"])
        )
    )
    frame["qa_text"] = frame["qa_text"].map(
        lambda value: cap_then_clean(value, int(caps["qa_side"]))
    )
    frame["issuer_id"] = frame["security_id"].astype(str)
    frame["month"] = (
        frame["earnings_date"].dt.to_period("M").dt.to_timestamp()
    )
    frame["issuer_call_id"] = [
        call_id(issuer, date)
        for issuer, date in zip(
            frame["issuer_id"], frame["earnings_date"], strict=True
        )
    ]
    frame.sort_values(
        ["issuer_id", "earnings_date", "ticker"], inplace=True
    )
    frame.drop_duplicates(
        ["issuer_id", "earnings_date"], keep="last", inplace=True
    )
    frame.reset_index(drop=True, inplace=True)
    if frame["issuer_call_id"].duplicated().any():
        raise RuntimeError("issuer_call_id is not unique")
    return frame


def prepare_preprocessed_identity_calls(source: pd.DataFrame) -> pd.DataFrame:
    """Validate, but never re-filter or re-cap, a transformed text table."""
    required = {
        "security_id",
        "ticker",
        "sector",
        "earnings_date",
        "membership_date",
        "in_index",
        "prepared_text",
        "qa_text",
        "issuer_call_id",
        "issuer_id",
        "month",
        IDENTITY_CONTROL_MARKER_COLUMN,
    }
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(
            f"Preprocessed identity-control columns are missing: {sorted(missing)}"
        )
    frame = source.copy()
    marker = frame[IDENTITY_CONTROL_MARKER_COLUMN].astype(str)
    if not marker.eq(IDENTITY_CONTROL_MARKER_VALUE).all():
        raise ValueError("The identity-control protocol marker is absent or invalid")
    for column in ("earnings_date", "membership_date", "month"):
        frame[column] = pd.to_datetime(frame[column], errors="raise")
    key_columns = [
        "security_id",
        "ticker",
        "sector",
        "earnings_date",
        "membership_date",
        "issuer_call_id",
        "issuer_id",
        "month",
    ]
    if frame[key_columns].isna().any().any():
        raise ValueError("A preprocessed identity-control key is missing")
    if frame["membership_date"].gt(frame["earnings_date"]).any():
        raise ValueError("A membership observation follows its call")
    if not frame["in_index"].astype(bool).all():
        raise ValueError("Input contains a call outside dated membership")
    if not frame["issuer_id"].astype(str).eq(frame["security_id"].astype(str)).all():
        raise ValueError("Issuer identifiers disagree with security identifiers")
    expected_month = frame["earnings_date"].dt.to_period("M").dt.to_timestamp()
    if not frame["month"].eq(expected_month).all():
        raise ValueError("A preprocessed call has an invalid calendar month")
    if frame["issuer_call_id"].astype(str).str.strip().eq("").any():
        raise ValueError("A preprocessed call identifier is empty")
    for column in ("prepared_text", "qa_text"):
        frame[column] = frame[column].fillna("").astype(str)
        if frame[column].str.strip().eq("").any():
            raise ValueError(f"A preprocessed {column} value is empty")
    if frame["issuer_call_id"].astype(str).duplicated().any():
        raise ValueError("Preprocessed call identifiers are not unique")
    if frame.duplicated(["issuer_id", "earnings_date"]).any():
        raise ValueError("Preprocessed issuer-date keys are not unique")
    frame.sort_values(
        ["issuer_id", "earnings_date", "ticker", "issuer_call_id"],
        inplace=True,
        kind="mergesort",
    )
    return frame.reset_index(drop=True)


def chunks(value: str, words_per_chunk: int) -> list[str]:
    words = str(value).split()
    return [
        " ".join(words[start : start + words_per_chunk])
        for start in range(0, len(words), words_per_chunk)
    ]


def encode_calls(
    model: Any,
    calls: pd.DataFrame,
    *,
    text_column: str,
    section_label: str,
    words_per_chunk: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    template = str(load_config()["encoder"]["prefix_template"])
    texts: list[str] = []
    owners: list[int] = []
    for owner, value in enumerate(calls[text_column].astype(str)):
        local = chunks(value, words_per_chunk)
        if not local:
            raise RuntimeError(f"No chunks for call row {owner}")
        for chunk in local:
            texts.append(
                template.format(section_label=section_label, text=chunk)
            )
            owners.append(owner)
    encoded = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float64)
    output = np.zeros((len(calls), encoded.shape[1]), dtype=np.float64)
    counts = np.zeros(len(calls), dtype=np.int64)
    np.add.at(output, np.asarray(owners, dtype=int), encoded)
    np.add.at(counts, np.asarray(owners, dtype=int), 1)
    if (counts <= 0).any():
        raise RuntimeError("A call has no encoded passage")
    return l2_normalize(output / counts[:, None]).astype(np.float32), counts


def embedding_frame(
    calls: pd.DataFrame, values: np.ndarray
) -> pd.DataFrame:
    columns = [f"e{index:03d}" for index in range(values.shape[1])]
    return pd.concat(
        [
            calls[KEY_COLUMNS].reset_index(drop=True),
            pd.DataFrame(values, columns=columns),
        ],
        axis=1,
    )


def main() -> None:
    args = parse_args()
    from sentence_transformers import SentenceTransformer

    config = load_config()
    encoder = config["encoder"]
    source = read_table(args.input)
    calls = (
        prepare_preprocessed_identity_calls(source)
        if args.preprocessed_identity_control
        else prepare_calls(source)
    )
    model = SentenceTransformer(
        encoder["name"], revision=encoder["revision"], device=args.device
    )
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = min(512, int(model.max_seq_length))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict[str, Any]] = []
    for words_per_chunk in config["chunking"]["word_chunk_sizes"]:
        section_outputs: dict[str, pd.DataFrame] = {}
        for section, column in (
            ("prepared", "prepared_text"),
            ("qa", "qa_text"),
        ):
            values, counts = encode_calls(
                model,
                calls,
                text_column=column,
                section_label=encoder["section_labels"][
                    "prepared_remarks"
                    if section == "prepared"
                    else "qa_side"
                ],
                words_per_chunk=int(words_per_chunk),
                batch_size=args.batch_size,
            )
            if values.shape[1] != int(config["feature_dimensions"]):
                raise RuntimeError("Pinned encoder dimension changed")
            output = embedding_frame(calls, values)
            section_outputs[section] = output
            path = (
                args.output_dir
                / f"call_{section}_all{int(words_per_chunk)}.parquet"
            )
            write_table(output, path)
            audit_rows.append(
                {
                    "section": section,
                    "chunk_words": int(words_per_chunk),
                    "calls": int(len(output)),
                    "passages": int(counts.sum()),
                }
            )
        if not section_outputs["prepared"][KEY_COLUMNS].equals(
            section_outputs["qa"][KEY_COLUMNS]
        ):
            raise RuntimeError("Prepared and Q&A output keys differ")
    audit = {
        "calls": int(len(calls)),
        "preprocessed_identity_control": bool(
            args.preprocessed_identity_control
        ),
        "date_min": str(calls["earnings_date"].min().date()),
        "date_max": str(calls["earnings_date"].max().date()),
        "rows": audit_rows,
    }
    (args.output_dir / "embedding_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
