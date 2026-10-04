#!/usr/bin/env python3
"""Build the three licensed section tables for identity-control evaluation.

The input follows the canonical section schema used by ``build_embeddings.py``
and additionally supplies company and current-call speaker aliases, either in
``company_aliases``/``speaker_aliases`` cells or in a long alias registry.
This entry point writes ``full_identity_mask.parquet``, ``correct_body.parquet``,
and ``donor_body.parquet``.  Every output retains the target call keys so the
ordinary embedding and fixed-head runners can process it unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


LOCAL_SCRIPTS = Path(__file__).resolve().parent
if str(LOCAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(LOCAL_SCRIPTS))

from build_embeddings import (  # noqa: E402
    IDENTITY_CONTROL_MARKER_COLUMN,
    IDENTITY_CONTROL_MARKER_VALUE,
    prepare_calls,
)
from build_identity_body_placebo import (  # noqa: E402
    CORRECT_ARM,
    DONOR_ARM,
    build_paired_corpus,
    canonical_calls,
    match_donors,
)
from common import read_table, save_json, write_table  # noqa: E402
from mask_identity_corpus import (  # noqa: E402
    NER_PLACEHOLDERS,
    load_spacy_model,
    mask_identity_frame,
    normalize_alias_registry,
)
from identity_masking import (  # noqa: E402
    NER_LABELS as FROZEN_NER_LABELS,
    SPEAKER_RE,
    VARIANT as FROZEN_MASK_VARIANT,
    IdentityMasker,
    ascii_fold,
    build_call_identities,
    build_global_aliases,
    clean_value,
    extract_person_name_variants,
    load_pair_rosters,
    registry_summary,
    sha256_text,
)


SECTION_COLUMNS = {
    "issuer_call_id",
    "security_id",
    "ticker",
    "sector",
    "earnings_date",
    "membership_date",
    "in_index",
    "prepared_text",
    "qa_text",
}
COMPANION_PREPARED_TEXT = " ".join(
    [
        "This matched mechanism-control document keeps the target call keys and a fixed prepared-section scaffold."
    ]
    * 8
)
FROZEN_EXPECTED_CALLS = 21_421
FROZEN_SPACY_RUNTIME = "3.8.14"
FROZEN_SPACY_MODEL = "en_core_web_sm"
FROZEN_SPACY_MODEL_VERSION = "3.8.0"
FROZEN_SHELL_TEMPLATE_VERSION = "target_identity_shell_v5_strict_line_name_shape"
FROZEN_COMPOSITE_TEMPLATE_VERSION = "identity_shell_plus_scrubbed_qa_v1"
FROZEN_SECTION_COLUMNS = {
    "issuer_call_id",
    "asset_symbol",
    "query_symbol",
    "company",
    "canonical_company",
    "canonical_asset_symbol",
    "canonical_query_symbol",
    "canonical_issuer_id",
    "canonical_sector",
    "original_asset_symbol",
    "original_query_symbol",
    "verified_call_date",
    "verified_month",
    "prepared_doc",
    "qa_doc",
    "opening_claimed_issuer",
    "qa_capped_word_count",
    "qa_text_sha256",
    "cleanroom_usable",
}
FROZEN_PAIR_COLUMNS = {
    "issuer_call_id",
    "pair_index",
    "question_speaker",
    "answer_speakers",
}

# The broad masker deliberately accepts every ``Label:`` span for recall.  The
# identity shell uses only strict line-start prepared speaker headers.
FROZEN_SPEAKER_HEADER_RE = re.compile(r"(?m)^[ \t]*" + SPEAKER_RE.pattern)
LOWERCASE_NAME_PARTICLES = frozenset(
    {
        "al",
        "bin",
        "da",
        "de",
        "del",
        "della",
        "den",
        "der",
        "di",
        "dos",
        "du",
        "el",
        "la",
        "le",
        "van",
        "von",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--frozen-inputs",
        action="store_true",
        help=(
            "Rebuild the reported experiment from canonical capped qa_doc bytes, "
            "the frozen historical identity metadata, and participant pair roster."
        ),
    )
    parser.add_argument(
        "--pair-rosters",
        type=Path,
        default=None,
        help="Frozen-input participant-pair parquet.",
    )
    parser.add_argument(
        "--historical-metadata",
        type=Path,
        default=None,
        help="Frozen-input historical company-name/ticker metadata table.",
    )
    parser.add_argument(
        "--frozen-expected-calls",
        type=int,
        default=FROZEN_EXPECTED_CALLS,
        help="Expected clean rows; 21,421 for the reported experiment.",
    )
    parser.add_argument(
        "--aliases",
        type=Path,
        default=None,
        help="Optional long registry: issuer_call_id, alias_type, alias.",
    )
    parser.add_argument("--company-alias-column", default="company_aliases")
    parser.add_argument("--speaker-alias-column", default="speaker_aliases")
    parser.add_argument("--ticker-alias-column", default="ticker_aliases")
    parser.add_argument("--canonical-company-column", default="canonical_company")
    parser.add_argument("--canonical-ticker-column", default="canonical_query_symbol")
    parser.add_argument("--spacy-model", default=None)
    parser.add_argument("--ner-labels", default="PERSON,ORG,PRODUCT,NORP,FAC")
    parser.add_argument("--ner-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260804)
    return parser.parse_args()


def parse_alias_cell(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        if bool(pd.isna(value)):
            return []
        text = str(value).strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, list):
                values = decoded
            else:
                values = re.split(r"\s*[|;]\s*", text)
        else:
            values = re.split(r"\s*[|;]\s*", text)
    output: list[str] = []
    for item in values:
        alias = re.sub(r"\s+", " ", str(item)).strip()
        if alias and alias not in output:
            output.append(alias)
    return output


def load_frozen_spacy(name: str = FROZEN_SPACY_MODEL) -> tuple[Any, dict[str, Any]]:
    """Load and validate the exact NER runtime used by the reported mask."""

    try:
        import spacy
    except ImportError as error:
        raise RuntimeError(
            "The frozen path requires requirements-embedding.txt, including spaCy."
        ) from error
    if str(spacy.__version__) != FROZEN_SPACY_RUNTIME:
        raise RuntimeError(
            f"spaCy {spacy.__version__} is installed; the frozen path requires "
            f"{FROZEN_SPACY_RUNTIME}"
        )
    disabled = ["tagger", "parser", "attribute_ruler", "lemmatizer"]
    nlp = spacy.load(name, disable=disabled)
    nlp.max_length = max(int(nlp.max_length), 100_000)
    if "ner" not in nlp.pipe_names:
        raise RuntimeError("The frozen spaCy model was loaded without NER")
    actual_model_version = str(nlp.meta.get("version"))
    if actual_model_version != FROZEN_SPACY_MODEL_VERSION:
        raise RuntimeError(
            f"{name} version {actual_model_version} was loaded; expected "
            f"{FROZEN_SPACY_MODEL_VERSION}"
        )
    available = set(nlp.get_pipe("ner").labels)
    missing = sorted(FROZEN_NER_LABELS.difference(available))
    if missing:
        raise RuntimeError(f"The frozen spaCy model lacks labels: {missing}")
    return nlp, {
        "spacy": str(spacy.__version__),
        "model_package": name,
        "model_name": nlp.meta.get("name"),
        "model_version": actual_model_version,
        "pipeline": list(nlp.pipe_names),
        "disabled": disabled,
        "labels": sorted(FROZEN_NER_LABELS),
    }


def prepare_frozen_sections(
    sections: pd.DataFrame,
    *,
    expected_calls: int | None = FROZEN_EXPECTED_CALLS,
) -> pd.DataFrame:
    """Validate canonical capped bytes without calling ``prepare_calls``."""

    missing = FROZEN_SECTION_COLUMNS.difference(sections.columns)
    if missing:
        raise ValueError(f"Frozen clean-room columns are missing: {sorted(missing)}")
    frame = sections[sections["cleanroom_usable"].fillna(False)].copy()
    if expected_calls is not None and len(frame) != int(expected_calls):
        raise ValueError(
            f"Expected {int(expected_calls):,} usable frozen calls; found {len(frame):,}"
        )
    frame["issuer_call_id"] = frame["issuer_call_id"].astype(str)
    if frame["issuer_call_id"].str.strip().eq("").any():
        raise ValueError("A frozen issuer_call_id is empty")
    if frame["issuer_call_id"].duplicated().any():
        raise ValueError("Frozen issuer_call_id values are not unique")
    frame["verified_call_date"] = pd.to_datetime(
        frame["verified_call_date"], errors="raise"
    )
    frame["verified_month"] = pd.to_datetime(
        frame["verified_month"], errors="raise"
    )
    required_keys = [
        "canonical_company",
        "canonical_asset_symbol",
        "canonical_query_symbol",
        "canonical_issuer_id",
        "canonical_sector",
        "verified_call_date",
        "verified_month",
        "qa_text_sha256",
    ]
    if frame[required_keys].isna().any().any():
        raise ValueError("A frozen canonical key or source hash is missing")
    frame["prepared_doc"] = frame["prepared_doc"].fillna("").astype(str)
    frame["qa_doc"] = frame["qa_doc"].fillna("").astype(str)
    if frame["prepared_doc"].str.strip().eq("").any():
        raise ValueError("A frozen prepared document is empty")
    if frame["qa_doc"].str.strip().eq("").any():
        raise ValueError("A frozen Q&A document is empty")
    if frame["qa_doc"].str.len().gt(9_000).any():
        raise ValueError("A frozen Q&A document exceeds the 9,000-character cap")
    expected_month = (
        frame["verified_call_date"].dt.to_period("M").dt.to_timestamp()
    )
    if not frame["verified_month"].eq(expected_month).all():
        raise ValueError("A frozen verified_month disagrees with verified_call_date")
    supplied_words = pd.to_numeric(
        frame["qa_capped_word_count"], errors="raise"
    ).astype(int)
    calculated_words = frame["qa_doc"].map(lambda value: len(value.split())).astype(int)
    if not supplied_words.eq(calculated_words).all():
        raise ValueError("qa_capped_word_count disagrees with capped qa_doc bytes")
    calculated_hashes = frame["qa_doc"].map(sha256_text)
    if not calculated_hashes.eq(frame["qa_text_sha256"].astype(str)).all():
        raise ValueError("qa_text_sha256 disagrees with capped qa_doc bytes")
    frame["qa_capped_word_count"] = supplied_words

    # This mirrors the frozen private loader exactly: source company remains
    # available as original_company before canonical values become model keys.
    frame["original_company"] = frame["company"]
    frame["asset_symbol"] = frame["canonical_asset_symbol"]
    frame["query_symbol"] = frame["canonical_query_symbol"]
    frame["issuer_id"] = frame["canonical_issuer_id"]
    frame["company"] = frame["canonical_company"]
    frame["sector"] = frame["canonical_sector"]
    frame["earnings_date"] = frame["verified_call_date"]
    frame["month"] = frame["verified_month"]
    frame.sort_values(
        ["issuer_id", "earnings_date", "issuer_call_id"],
        kind="mergesort",
        inplace=True,
    )
    return frame.reset_index(drop=True)


def _speaker_display_name(label: Any) -> str:
    variants = extract_person_name_variants(label)
    if not variants:
        return ""
    return sorted(
        variants,
        key=lambda value: (
            -sum(ord(character) > 127 for character in value),
            -len(value),
            value.casefold(),
            value,
        ),
    )[0]


def _speaker_key(name: str) -> str:
    folded = ascii_fold(clean_value(name)).casefold()
    folded = re.sub(r"[^a-z0-9]+", " ", folded)
    return re.sub(r"\s+", " ", folded).strip()


def _prepared_speaker_display_name(label: Any) -> str:
    raw_label = clean_value(label)
    name_segment = re.split(
        r"\s+[\-\N{EN DASH}\N{EM DASH}]\s+", raw_label, maxsplit=1
    )[0]
    if any(character.isdigit() for character in name_segment):
        return ""
    name = _speaker_display_name(raw_label)
    if not name:
        return ""
    tokens = re.findall(
        r"[A-Za-z][A-Za-z'\N{RIGHT SINGLE QUOTATION MARK}.-]*", name
    )
    for token in tokens[1:]:
        stripped = token.strip(".'-\N{RIGHT SINGLE QUOTATION MARK}")
        if not stripped or not stripped[0].islower():
            continue
        if stripped.casefold() in LOWERCASE_NAME_PARTICLES:
            continue
        if any(character.isupper() for character in stripped[1:]):
            continue
        return ""
    return name


def ordered_unique_speakers(
    prepared_doc: Any,
    pair_rows: pd.DataFrame,
) -> tuple[str, ...]:
    """Recover the frozen first-occurrence speaker roster for one call."""

    labels: list[Any] = []
    for match in FROZEN_SPEAKER_HEADER_RE.finditer(str(prepared_doc or "")):
        name = _prepared_speaker_display_name(match.group(1))
        if name:
            labels.append(name)
    if len(pair_rows):
        ordered = pair_rows.sort_values("pair_index", kind="mergesort")
        for row in ordered.itertuples(index=False):
            labels.append(row.question_speaker)
            labels.extend(clean_value(row.answer_speakers).split("|"))
    names: list[str] = []
    seen: set[str] = set()
    for label in labels:
        name = _speaker_display_name(label)
        key = _speaker_key(name)
        if name and key and key not in seen:
            names.append(name)
            seen.add(key)
    return tuple(names)


def frozen_shell_metadata(
    sections: pd.DataFrame,
    pair_rows: pd.DataFrame,
) -> pd.DataFrame:
    missing = FROZEN_PAIR_COLUMNS.difference(pair_rows.columns)
    if missing:
        raise ValueError(f"Frozen participant-pair columns are missing: {sorted(missing)}")
    pairs = pair_rows[list(FROZEN_PAIR_COLUMNS)].copy()
    pairs["issuer_call_id"] = pairs["issuer_call_id"].astype(str)
    allowed = set(sections["issuer_call_id"].astype(str))
    pairs = pairs[pairs["issuer_call_id"].isin(allowed)].copy()
    groups = {
        str(call_id): group
        for call_id, group in pairs.groupby("issuer_call_id", sort=False)
    }
    empty = pd.DataFrame(columns=list(FROZEN_PAIR_COLUMNS))
    rows: list[dict[str, Any]] = []
    for row in sections.itertuples(index=False):
        call_id = str(row.issuer_call_id)
        rows.append(
            {
                "call_id": call_id,
                "canonical_company": row.canonical_company,
                "canonical_ticker": row.canonical_query_symbol,
                "speakers": list(
                    ordered_unique_speakers(
                        row.prepared_doc,
                        groups.get(call_id, empty),
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def frozen_public_base(sections: pd.DataFrame) -> pd.DataFrame:
    """Map frozen canonical keys to the no-recap embedding input contract."""

    output = pd.DataFrame(
        {
            "security_id": sections["canonical_issuer_id"].astype(str),
            "ticker": sections["canonical_query_symbol"].astype(str),
            "sector": sections["canonical_sector"].astype(str),
            "earnings_date": pd.to_datetime(sections["verified_call_date"]),
            # The clean-room row verifies membership on the call date.  This
            # compatibility field is consumed only by the public input gate.
            "membership_date": pd.to_datetime(sections["verified_call_date"]),
            "in_index": True,
            "prepared_text": sections["prepared_doc"].astype(str),
            "qa_text": sections["qa_doc"].astype(str),
            "issuer_call_id": sections["issuer_call_id"].astype(str),
            "issuer_id": sections["canonical_issuer_id"].astype(str),
            "month": pd.to_datetime(sections["verified_month"]),
            "qa_capped_word_count": sections["qa_capped_word_count"].astype(int),
            "qa_text_sha256": sections["qa_text_sha256"].astype(str),
            "canonical_company": sections["canonical_company"].astype(str),
            "canonical_query_symbol": sections["canonical_query_symbol"].astype(str),
        }
    )
    return output


def _assert_frozen_registry_checkpoints(summary: Mapping[str, Any]) -> None:
    expected = {
        "global_aliases": 6_001,
        "call_aliases": 965_051,
        "global_plain_single_issuer_aliases_case_sensitive": 610,
        "registry_sha256": "6b39d56284eb499c0de4aa6ecc392128707f2ccb9f4a4da948b7b3f8bd440686",
    }
    failed = {
        key: {"actual": summary.get(key), "expected": value}
        for key, value in expected.items()
        if summary.get(key) != value
    }
    expected_sources = {
        "call_issuer": 69_476,
        "call_ticker": 22_871,
        "opening_claimed_issuer": 19_089,
        "pair_speaker": 254_486,
        "pair_speaker_component": 459_915,
        "prepared_speaker": 49_403,
        "prepared_speaker_component": 89_811,
    }
    if summary.get("call_aliases_by_source") != expected_sources:
        failed["call_aliases_by_source"] = {
            "actual": summary.get("call_aliases_by_source"),
            "expected": expected_sources,
        }
    if failed:
        raise RuntimeError(f"Frozen identity registry checkpoints changed: {failed}")


def _frozen_mask_coverage(audits: pd.DataFrame) -> dict[str, Any]:
    type_columns = [column for column in audits if column.startswith("selected_")]
    source_columns = [column for column in audits if column.startswith("source_")]
    return {
        "calls": int(len(audits)),
        "calls_without_selected_spans": int(audits["selected_spans"].eq(0).sum()),
        "selected_spans": int(audits["selected_spans"].sum()),
        "deterministic_candidates": int(audits["deterministic_candidates"].sum()),
        "ner_candidates": int(audits["ner_candidates"].sum()),
        "overlap_dropped": int(audits["overlap_dropped"].sum()),
        "original_words": int(audits["original_words"].sum()),
        "masked_words": int(audits["masked_words"].sum()),
        "residual_alias_calls": int(audits["residual_alias_count"].gt(0).sum()),
        "empty_after_mask_calls": int(audits["empty_after_mask"].fillna(False).sum()),
        "selected_by_type": {
            column.removeprefix("selected_").upper(): int(audits[column].sum())
            for column in type_columns
            if column not in {"selected_spans"}
        },
        "selected_by_source": {
            column.removeprefix("source_"): int(audits[column].sum())
            for column in source_columns
        },
    }


def _assert_frozen_coverage_checkpoints(coverage: Mapping[str, Any]) -> None:
    expected = {
        "calls": 21_421,
        "calls_without_selected_spans": 114,
        "selected_spans": 599_008,
        "deterministic_candidates": 387_465,
        "ner_candidates": 554_155,
        "overlap_dropped": 342_612,
        "original_words": 33_095_194,
        "masked_words": 32_922_398,
        "residual_alias_calls": 0,
        "empty_after_mask_calls": 0,
    }
    failed = {
        key: {"actual": coverage.get(key), "expected": value}
        for key, value in expected.items()
        if coverage.get(key) != value
    }
    if failed:
        raise RuntimeError(f"Frozen identity-mask coverage changed: {failed}")


def build_frozen_identity_control_tables(
    sections: pd.DataFrame,
    metadata: pd.DataFrame,
    pair_rosters_path: Path,
    pair_rows: pd.DataFrame,
    *,
    nlp: Any,
    ner_batch_size: int = 32,
    seed: int = 20260804,
    enforce_reported_checkpoints: bool = True,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply the reported identity controls without normalizing source bytes."""

    if ner_batch_size <= 0:
        raise ValueError("--ner-batch-size must be positive")
    call_ids = sections["issuer_call_id"].astype(str).tolist()
    global_aliases = build_global_aliases(sections, metadata)
    known_company_aliases = frozenset(
        alias.text.casefold()
        for alias in global_aliases
        if alias.entity_type == "ISSUER"
    )
    mask_rosters = load_pair_rosters(pair_rosters_path, set(call_ids))
    identities = build_call_identities(
        sections,
        mask_rosters,
        known_company_aliases,
    )
    registry = registry_summary(global_aliases, identities)
    if enforce_reported_checkpoints:
        _assert_frozen_registry_checkpoints(registry)
    masker = IdentityMasker(nlp, global_aliases, FROZEN_NER_LABELS)
    documents = nlp.pipe(
        sections["qa_doc"].astype(str),
        batch_size=int(ner_batch_size),
        n_process=1,
    )
    masked_bodies: list[str] = []
    audit_rows: list[dict[str, Any]] = []
    for row, document in zip(sections.itertuples(index=False), documents, strict=True):
        call_id = str(row.issuer_call_id)
        result = masker.mask(str(row.qa_doc), identities[call_id], document)
        if result.audit["original_sha256"] != str(row.qa_text_sha256):
            raise RuntimeError(f"Frozen source hash changed for call {call_id}")
        if int(result.audit["original_words"]) != int(row.qa_capped_word_count):
            raise RuntimeError(f"Frozen source word count changed for call {call_id}")
        if result.audit["residual_alias_count"] or result.audit["empty_after_mask"]:
            raise RuntimeError(f"Frozen identity mask failed for call {call_id}")
        masked_bodies.append(result.masked_text)
        audit_rows.append(result.audit)
    mask_audit = pd.DataFrame(audit_rows)
    coverage = _frozen_mask_coverage(mask_audit)
    if enforce_reported_checkpoints:
        _assert_frozen_coverage_checkpoints(coverage)

    base = frozen_public_base(sections)
    masked = base.copy()
    masked["qa_text"] = masked_bodies
    masked["mask_variant"] = FROZEN_MASK_VARIANT
    masked["masked_qa_sha256"] = [sha256_text(value) for value in masked_bodies]
    calls = canonical_calls(
        masked,
        body_word_count_column="qa_capped_word_count",
    )
    mapping = match_donors(calls, seed=seed)
    shell_metadata = frozen_shell_metadata(sections, pair_rows)
    corpus = build_paired_corpus(
        calls,
        mapping,
        shell_metadata=shell_metadata,
    )
    outputs = {
        "full_identity_mask": masked,
        CORRECT_ARM: companion_sections(base, corpus, CORRECT_ARM),
        "donor_body": companion_sections(base, corpus, DONOR_ARM),
    }
    expected_ids = set(call_ids)
    for label, output in outputs.items():
        output[IDENTITY_CONTROL_MARKER_COLUMN] = IDENTITY_CONTROL_MARKER_VALUE
        if set(output["issuer_call_id"].astype(str)) != expected_ids:
            raise RuntimeError(f"The frozen {label} output changed call support")
    audit = {
        "mode": "reported_frozen_inputs",
        "calls": int(len(sections)),
        "months": int(sections["month"].nunique()),
        "outputs": {key: int(len(value)) for key, value in outputs.items()},
        "registry": registry,
        "coverage": coverage,
        "all_rows_transformed_before_model_fitting": True,
        "donor_matching_uses_outcomes": False,
        "donors_are_one_to_one_same_month_cross_sector": True,
        "source_qa_doc_masked_without_pre_normalization": True,
        "qa_capped_word_count_verified": True,
        "canonical_call_ids_preserved": True,
        "identity_shell_template_version": FROZEN_SHELL_TEMPLATE_VERSION,
        "composite_template_version": FROZEN_COMPOSITE_TEMPLATE_VERSION,
        "companion_composite_is_not_recapped": True,
        "protocol_marker": IDENTITY_CONTROL_MARKER_VALUE,
        "seed": int(seed),
    }
    return outputs, mapping, mask_audit, audit


def prepare_section_keys(sections: pd.DataFrame) -> pd.DataFrame:
    missing = SECTION_COLUMNS.difference(sections.columns)
    if missing:
        raise ValueError(f"Canonical section columns are missing: {sorted(missing)}")
    frame = sections.copy()
    key_columns = [
        "issuer_call_id",
        "security_id",
        "ticker",
        "sector",
        "earnings_date",
        "membership_date",
    ]
    if frame[key_columns].isna().any().any():
        raise ValueError("A canonical section key is missing")
    if frame["issuer_call_id"].astype(str).duplicated().any():
        raise ValueError("Supplied clean-room call identifiers are not unique")
    frame["_source_call_id"] = frame["issuer_call_id"].astype(str)
    prepared = prepare_calls(frame)
    prepared["issuer_call_id"] = prepared.pop("_source_call_id")
    if prepared["issuer_call_id"].duplicated().any():
        raise ValueError("Retained clean-room call identifiers are not unique")
    return prepared


def registry_from_section_columns(
    sections: pd.DataFrame,
    *,
    company_column: str,
    ticker_column: str,
    speaker_column: str,
) -> pd.DataFrame:
    missing = {company_column, ticker_column, speaker_column}.difference(sections.columns)
    if missing:
        raise ValueError(
            "Section alias columns are missing; supply --aliases or add: "
            f"{sorted(missing)}"
        )
    rows: list[dict[str, str]] = []
    for row in sections.itertuples(index=False):
        values = row._asdict()
        call = str(values["issuer_call_id"])
        grouped = {
            "company": parse_alias_cell(values[company_column]),
            "ticker": parse_alias_cell(values[ticker_column]),
            "speaker": parse_alias_cell(values[speaker_column]),
        }
        if not grouped["company"] or not grouped["ticker"]:
            raise ValueError(f"Call {call} lacks a canonical company or ticker alias")
        for kind, aliases in grouped.items():
            rows.extend(
                {"issuer_call_id": call, "alias_type": kind, "alias": alias}
                for alias in aliases
            )
    return normalize_alias_registry(pd.DataFrame(rows))


def shell_metadata_from_sections(
    sections: pd.DataFrame,
    *,
    canonical_company_column: str,
    canonical_ticker_column: str,
    speaker_column: str,
) -> pd.DataFrame:
    required = {
        "issuer_call_id",
        canonical_company_column,
        canonical_ticker_column,
        speaker_column,
    }
    missing = required.difference(sections.columns)
    if missing:
        raise ValueError(f"Identity-shell columns are missing: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for _, row in sections.iterrows():
        company = re.sub(r"\s+", " ", str(row[canonical_company_column])).strip()
        ticker = re.sub(r"\s+", " ", str(row[canonical_ticker_column])).strip()
        if not company or company.lower() == "nan":
            raise ValueError("A canonical shell company is missing")
        if not ticker or ticker.lower() == "nan":
            raise ValueError("A canonical shell ticker is missing")
        rows.append(
            {
                "call_id": str(row["issuer_call_id"]),
                "canonical_company": company,
                "canonical_ticker": ticker,
                "speakers": parse_alias_cell(row[speaker_column]),
            }
        )
    output = pd.DataFrame(rows)
    if output["call_id"].duplicated().any():
        raise ValueError("Identity-shell call identifiers are not unique")
    return output


def validate_mask_registry(
    registry: pd.DataFrame,
    call_ids: set[str],
) -> None:
    unknown = set(registry["call_id"]).difference(call_ids)
    if unknown:
        raise ValueError("Mask alias registry contains calls absent from retained sections")
    kinds = {
        str(call): set(group["alias_type"].astype(str))
        for call, group in registry.groupby("call_id", sort=False)
    }
    incomplete = sorted(
        call for call in call_ids if not {"company", "ticker"}.issubset(kinds.get(call, set()))
    )
    if incomplete:
        raise ValueError("A retained call lacks company or ticker mask aliases")


def companion_sections(
    sections: pd.DataFrame,
    corpus: pd.DataFrame,
    arm: str,
) -> pd.DataFrame:
    documents = corpus[corpus["arm"].eq(arm)][
        ["target_call_id", "document_text"]
    ].rename(columns={"target_call_id": "issuer_call_id"})
    output = sections.merge(
        documents,
        on="issuer_call_id",
        how="left",
        validate="one_to_one",
    )
    if output["document_text"].isna().any():
        raise RuntimeError(f"The {arm} corpus lacks a target call")
    output["qa_text"] = output.pop("document_text")
    output["prepared_text"] = COMPANION_PREPARED_TEXT
    return output


def build_identity_control_tables(
    sections: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    nlp: Any = None,
    ner_labels: set[str] | None = None,
    ner_batch_size: int = 32,
    seed: int = 20260804,
    canonical_company_column: str = "canonical_company",
    canonical_ticker_column: str = "canonical_query_symbol",
    speaker_column: str = "speaker_aliases",
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    keyed = prepare_section_keys(sections)
    validate_mask_registry(registry, set(keyed["issuer_call_id"].astype(str)))
    calculated_words = (
        keyed["qa_text"].astype(str).str.split().str.len().astype(int)
    )
    if "qa_capped_word_count" in keyed.columns:
        supplied_words = pd.to_numeric(
            keyed["qa_capped_word_count"], errors="raise"
        ).astype(int)
        if not supplied_words.eq(calculated_words).all():
            raise ValueError("qa_capped_word_count disagrees with capped Q&A text")
        keyed["_donor_match_words"] = supplied_words
    else:
        keyed["_donor_match_words"] = calculated_words
    shell_metadata = shell_metadata_from_sections(
        keyed,
        canonical_company_column=canonical_company_column,
        canonical_ticker_column=canonical_ticker_column,
        speaker_column=speaker_column,
    )
    masked, placeholder_counts = mask_identity_frame(
        keyed,
        registry,
        call_id_column="issuer_call_id",
        text_column="qa_text",
        output_text_column="qa_text",
        nlp=nlp,
        ner_labels=ner_labels,
        ner_batch_size=ner_batch_size,
    )
    calls = canonical_calls(
        masked,
        body_word_count_column="_donor_match_words",
    )
    mapping = match_donors(calls, seed=seed)
    corpus = build_paired_corpus(
        calls,
        mapping,
        shell_metadata=shell_metadata,
    )
    outputs = {
        "full_identity_mask": masked,
        CORRECT_ARM: companion_sections(keyed, corpus, CORRECT_ARM),
        "donor_body": companion_sections(keyed, corpus, DONOR_ARM),
    }
    expected_ids = set(keyed["issuer_call_id"])
    for label, output in outputs.items():
        output.drop(columns=["_donor_match_words"], errors="ignore", inplace=True)
        output[IDENTITY_CONTROL_MARKER_COLUMN] = IDENTITY_CONTROL_MARKER_VALUE
        if set(output["issuer_call_id"]) != expected_ids:
            raise RuntimeError(f"The {label} output changed target call support")
    audit = {
        "calls": int(len(keyed)),
        "months": int(keyed["month"].nunique()),
        "outputs": {key: int(len(value)) for key, value in outputs.items()},
        "placeholder_counts": dict(sorted(placeholder_counts.items())),
        "all_rows_transformed_before_model_fitting": True,
        "donor_matching_uses_outcomes": False,
        "donors_are_one_to_one_same_month_cross_sector": True,
        "ordinary_character_cap_precedes_identity_masking": True,
        "companion_composite_is_not_recapped": True,
        "protocol_marker": IDENTITY_CONTROL_MARKER_VALUE,
        "seed": int(seed),
    }
    return outputs, mapping, audit


def main() -> None:
    args = parse_args()
    labels = {
        label.strip().upper()
        for label in str(args.ner_labels).split(",")
        if label.strip()
    }
    unsupported = labels.difference(NER_PLACEHOLDERS)
    if unsupported:
        raise ValueError(f"Unsupported NER labels: {sorted(unsupported)}")
    mask_audit: pd.DataFrame | None = None
    if args.frozen_inputs:
        if args.aliases is not None:
            raise ValueError("--aliases is only for the portable reference path")
        if args.pair_rosters is None or args.historical_metadata is None:
            raise ValueError(
                "--frozen-inputs requires --pair-rosters and --historical-metadata"
            )
        if labels != set(FROZEN_NER_LABELS):
            raise ValueError(
                f"The frozen path requires NER labels {sorted(FROZEN_NER_LABELS)}"
            )
        if args.frozen_expected_calls <= 0:
            raise ValueError("--frozen-expected-calls must be positive")
        sections = prepare_frozen_sections(
            read_table(args.sections),
            expected_calls=int(args.frozen_expected_calls),
        )
        metadata = read_table(args.historical_metadata)
        if args.pair_rosters.suffix.lower() in {".parquet", ".pq"}:
            pair_rows = pd.read_parquet(
                args.pair_rosters,
                columns=sorted(FROZEN_PAIR_COLUMNS),
            )
        else:
            pair_rows = read_table(args.pair_rosters)
        nlp, spacy_meta = load_frozen_spacy(
            args.spacy_model or FROZEN_SPACY_MODEL
        )
        outputs, mapping, mask_audit, audit = build_frozen_identity_control_tables(
            sections,
            metadata,
            args.pair_rosters,
            pair_rows,
            nlp=nlp,
            ner_batch_size=args.ner_batch_size,
            seed=args.seed,
            enforce_reported_checkpoints=(
                int(args.frozen_expected_calls) == FROZEN_EXPECTED_CALLS
            ),
        )
        audit["spacy"] = spacy_meta
    else:
        if args.pair_rosters is not None or args.historical_metadata is not None:
            raise ValueError(
                "--pair-rosters/--historical-metadata require --frozen-inputs"
            )
        sections = prepare_section_keys(read_table(args.sections))
        if args.aliases:
            registry = normalize_alias_registry(read_table(args.aliases))
        else:
            registry = registry_from_section_columns(
                sections,
                company_column=args.company_alias_column,
                ticker_column=args.ticker_alias_column,
                speaker_column=args.speaker_alias_column,
            )
        nlp = load_spacy_model(args.spacy_model) if args.spacy_model else None
        outputs, mapping, audit = build_identity_control_tables(
            sections,
            registry,
            nlp=nlp,
            ner_labels=labels if nlp is not None else set(),
            ner_batch_size=args.ner_batch_size,
            seed=args.seed,
            canonical_company_column=args.canonical_company_column,
            canonical_ticker_column=args.canonical_ticker_column,
            speaker_column=args.speaker_alias_column,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for label, output in outputs.items():
        write_table(output, args.output_dir / f"{label}.parquet")
    write_table(mapping, args.output_dir / "donor_mapping.csv")
    if mask_audit is not None:
        write_table(mask_audit, args.output_dir / "mask_coverage_audit.parquet")
    save_json(args.output_dir / "identity_control_build_audit.json", audit)
    print(f"Wrote three identity-control section tables to {args.output_dir}")


if __name__ == "__main__":
    main()
