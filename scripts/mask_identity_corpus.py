#!/usr/bin/env python3
"""Mask explicit identity aliases in every row of a licensed Q&A corpus.

The calls table must contain one row per call.  The alias registry is a long
table with a call identifier, ``alias_type`` (company, ticker, or speaker),
and ``alias``.  Company and ticker aliases are applied corpus-wide; speaker
aliases are applied to their associated call.  Supplying ``--spacy-model``
also masks selected named-entity labels.  The script never branches on a
training/evaluation split, so the same transformation is applied throughout.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd


LOCAL_SCRIPTS = Path(__file__).resolve().parent
if str(LOCAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(LOCAL_SCRIPTS))

from common import read_table, save_json, write_table  # noqa: E402


KIND_ALIASES = {
    "company": "company",
    "issuer": "company",
    "organization": "company",
    "org": "company",
    "ticker": "ticker",
    "symbol": "ticker",
    "speaker": "speaker",
    "person": "speaker",
    "executive": "speaker",
}
EXPLICIT_PLACEHOLDERS = {
    "company": "[ISSUER]",
    "ticker": "[TICKER]",
    "speaker": "[PERSON]",
}
NER_PLACEHOLDERS = {
    "PERSON": "[PERSON]",
    "ORG": "[ORG]",
    "PRODUCT": "[PRODUCT]",
    "NORP": "[NORP]",
    "FAC": "[FAC]",
}


@dataclass(frozen=True)
class Alias:
    value: str
    kind: str


@dataclass(frozen=True)
class MaskSpan:
    start: int
    end: int
    placeholder: str
    priority: int
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--aliases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--call-id-column", default="issuer_call_id")
    parser.add_argument("--text-column", default="qa_text")
    parser.add_argument("--output-text-column", default="qa_text")
    parser.add_argument("--alias-call-id-column", default="issuer_call_id")
    parser.add_argument("--alias-column", default="alias")
    parser.add_argument("--alias-type-column", default="alias_type")
    parser.add_argument("--split-column", default=None)
    parser.add_argument("--spacy-model", default=None)
    parser.add_argument(
        "--ner-labels",
        default="PERSON,ORG,PRODUCT,NORP,FAC",
        help="Comma-separated spaCy entity labels to mask.",
    )
    parser.add_argument("--ner-batch-size", type=int, default=32)
    return parser.parse_args()


def _clean_alias(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_alias_registry(
    aliases: pd.DataFrame,
    *,
    call_id_column: str = "issuer_call_id",
    alias_column: str = "alias",
    alias_type_column: str = "alias_type",
) -> pd.DataFrame:
    required = [call_id_column, alias_column, alias_type_column]
    missing = set(required).difference(aliases.columns)
    if missing:
        raise ValueError(f"Alias registry columns are missing: {sorted(missing)}")
    frame = aliases[required].copy()
    frame.columns = [
        "call_id" if column == call_id_column else
        "alias" if column == alias_column else "alias_type"
        for column in frame.columns
    ]
    if frame["call_id"].isna().any():
        raise ValueError("Alias registry contains a missing call identifier")
    frame["call_id"] = frame["call_id"].astype(str)
    frame["alias"] = frame["alias"].map(_clean_alias)
    raw_kinds = frame["alias_type"].astype(str).str.strip().str.lower()
    unknown = sorted(set(raw_kinds).difference(KIND_ALIASES))
    if unknown:
        raise ValueError(f"Unsupported alias types: {unknown}")
    frame["alias_type"] = raw_kinds.map(KIND_ALIASES)
    lengths = frame["alias"].str.len()
    frame = frame[
        (frame["alias_type"].eq("ticker") & lengths.ge(1))
        | (~frame["alias_type"].eq("ticker") & lengths.ge(2))
    ].copy()
    frame.drop_duplicates(["call_id", "alias_type", "alias"], inplace=True)
    return frame.reset_index(drop=True)


def alias_indices(
    registry: pd.DataFrame,
) -> tuple[tuple[Alias, ...], dict[str, tuple[Alias, ...]]]:
    global_rows = registry[registry["alias_type"].isin(["company", "ticker"])]
    global_aliases = tuple(
        Alias(row.alias, row.alias_type)
        for row in global_rows.drop_duplicates(["alias_type", "alias"]).itertuples()
    )
    local: dict[str, list[Alias]] = defaultdict(list)
    for row in registry[registry["alias_type"].eq("speaker")].itertuples():
        local[str(row.call_id)].append(Alias(row.alias, row.alias_type))
    return global_aliases, {
        key: tuple(values) for key, values in local.items()
    }


@lru_cache(maxsize=None)
def _literal_pattern(value: str) -> re.Pattern[str]:
    escaped = re.escape(value).replace(r"\ ", r"\s+")
    left = r"(?<![A-Za-z0-9])" if value[0].isalnum() else ""
    right = r"(?![A-Za-z0-9])" if value[-1].isalnum() else ""
    return re.compile(left + escaped + right, re.IGNORECASE)


@lru_cache(maxsize=None)
def _short_ticker_patterns(value: str) -> tuple[re.Pattern[str], ...]:
    escaped = re.escape(value)
    return (
        re.compile(
            rf"(?i)\b(?:ticker|symbol|nyse|nasdaq|amex)\b"
            rf"\s*(?:[:=]|\(|\[)?\s*(?P<alias>{escaped})(?![A-Za-z0-9])"
        ),
        re.compile(rf"(?i)\$(?P<alias>{escaped})(?![A-Za-z0-9])"),
    )


def _short_ticker_matches(text: str, value: str) -> Iterable[tuple[int, int]]:
    for pattern in _short_ticker_patterns(value):
        for match in pattern.finditer(text):
            yield match.span("alias")


def explicit_spans(text: str, aliases: Sequence[Alias]) -> list[MaskSpan]:
    spans: list[MaskSpan] = []
    for alias in aliases:
        compact = re.sub(r"[^A-Za-z0-9]", "", alias.value)
        if alias.kind == "ticker" and len(compact) <= 2:
            matches = _short_ticker_matches(text, alias.value)
        else:
            matches = (
                match.span() for match in _literal_pattern(alias.value).finditer(text)
            )
        for start, end in matches:
            spans.append(
                MaskSpan(
                    start,
                    end,
                    EXPLICIT_PLACEHOLDERS[alias.kind],
                    100,
                    alias.kind,
                )
            )
    return spans


def entity_spans(
    document: Any,
    allowed_labels: set[str],
) -> list[MaskSpan]:
    if document is None:
        return []
    output = []
    for entity in getattr(document, "ents", ()):
        label = str(entity.label_).upper()
        if label in allowed_labels and label in NER_PLACEHOLDERS:
            output.append(
                MaskSpan(
                    int(entity.start_char),
                    int(entity.end_char),
                    NER_PLACEHOLDERS[label],
                    10,
                    label,
                )
            )
    return output


def resolve_spans(spans: Sequence[MaskSpan]) -> list[MaskSpan]:
    selected: list[MaskSpan] = []
    ordered = sorted(
        spans,
        key=lambda span: (-span.priority, -(span.end - span.start), span.start, span.label),
    )
    for candidate in ordered:
        if candidate.start >= candidate.end:
            continue
        if any(
            candidate.start < accepted.end and accepted.start < candidate.end
            for accepted in selected
        ):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda span: span.start)


def mask_text(
    text: Any,
    aliases: Sequence[Alias],
    *,
    document: Any = None,
    ner_labels: set[str] | None = None,
) -> tuple[str, Counter[str]]:
    value = str(text or "")
    allowed = set() if ner_labels is None else {label.upper() for label in ner_labels}
    spans = resolve_spans(
        explicit_spans(value, aliases) + entity_spans(document, allowed)
    )
    pieces: list[str] = []
    cursor = 0
    counts: Counter[str] = Counter()
    for span in spans:
        pieces.extend((value[cursor:span.start], span.placeholder))
        cursor = span.end
        counts[span.label] += 1
    pieces.append(value[cursor:])
    return "".join(pieces), counts


def load_spacy_model(name: str) -> Any:
    try:
        import spacy
    except ImportError as error:
        raise RuntimeError(
            "spaCy is optional; install spaCy and the requested model to use NER"
        ) from error
    nlp = spacy.load(
        name,
        disable=["tagger", "parser", "attribute_ruler", "lemmatizer"],
    )
    nlp.max_length = max(int(nlp.max_length), 100_000)
    if "ner" not in nlp.pipe_names:
        raise RuntimeError("The requested spaCy model was loaded without NER")
    available = set(nlp.get_pipe("ner").labels)
    missing = sorted(set(NER_PLACEHOLDERS).difference(available))
    if missing:
        raise RuntimeError(f"The spaCy model is missing NER labels: {missing}")
    return nlp


def mask_identity_frame(
    calls: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    call_id_column: str = "issuer_call_id",
    text_column: str = "qa_text",
    output_text_column: str = "qa_text",
    nlp: Any = None,
    ner_labels: set[str] | None = None,
    ner_batch_size: int = 32,
) -> tuple[pd.DataFrame, Counter[str]]:
    required = {call_id_column, text_column}
    missing = required.difference(calls.columns)
    if missing:
        raise ValueError(f"Call columns are missing: {sorted(missing)}")
    if calls[call_id_column].astype(str).duplicated().any():
        raise ValueError("Call identifiers must be unique")
    if calls[call_id_column].isna().any():
        raise ValueError("A call identifier is missing")
    output = calls.copy()
    call_ids = output[call_id_column].astype(str).tolist()
    known_calls = set(call_ids)
    unknown = sorted(set(registry["call_id"]).difference(known_calls))
    if unknown:
        raise ValueError("Alias registry contains call identifiers absent from input")
    global_aliases, local_aliases = alias_indices(registry)
    texts = output[text_column].fillna("").astype(str).tolist()
    if nlp is None:
        documents: Iterable[Any] = (None for _ in texts)
    else:
        documents = nlp.pipe(texts, batch_size=ner_batch_size)
    masked: list[str] = []
    totals: Counter[str] = Counter()
    for call_id, text, document in zip(call_ids, texts, documents, strict=True):
        value, counts = mask_text(
            text,
            global_aliases + local_aliases.get(call_id, ()),
            document=document,
            ner_labels=ner_labels,
        )
        masked.append(value)
        totals.update(counts)
    if output_text_column != text_column:
        output.drop(columns=[text_column], inplace=True)
    output[output_text_column] = masked
    return output, totals


def main() -> None:
    args = parse_args()
    calls = read_table(args.input)
    registry = normalize_alias_registry(
        read_table(args.aliases),
        call_id_column=args.alias_call_id_column,
        alias_column=args.alias_column,
        alias_type_column=args.alias_type_column,
    )
    labels = {
        label.strip().upper()
        for label in str(args.ner_labels).split(",")
        if label.strip()
    }
    unsupported = labels.difference(NER_PLACEHOLDERS)
    if unsupported:
        raise ValueError(f"Unsupported NER labels: {sorted(unsupported)}")
    nlp = load_spacy_model(args.spacy_model) if args.spacy_model else None
    output, counts = mask_identity_frame(
        calls,
        registry,
        call_id_column=args.call_id_column,
        text_column=args.text_column,
        output_text_column=args.output_text_column,
        nlp=nlp,
        ner_labels=labels if nlp is not None else set(),
        ner_batch_size=args.ner_batch_size,
    )
    write_table(output, args.output)
    split_counts: dict[str, int] = {}
    if args.split_column:
        if args.split_column not in calls.columns:
            raise ValueError(f"Split column is missing: {args.split_column}")
        split_counts = {
            str(key): int(value)
            for key, value in calls[args.split_column].value_counts(dropna=False).items()
        }
    audit = {
        "rows": int(len(output)),
        "alias_rows": int(len(registry)),
        "explicit_global_aliases": len(alias_indices(registry)[0]),
        "spacy_model": args.spacy_model,
        "placeholder_counts": dict(sorted(counts.items())),
        "split_row_counts": split_counts,
        "source_text_retained": False,
    }
    if args.audit:
        save_json(args.audit, audit)
    print(f"Wrote {len(output):,} identity-masked call rows to {args.output}")


if __name__ == "__main__":
    main()
