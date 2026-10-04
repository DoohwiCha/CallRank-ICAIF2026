#!/usr/bin/env python3
"""Common identity masker for the 2026-08-04 CallRank mechanism audit.

This module deliberately separates text transformation from return/model code.
It combines deterministic issuer, ticker, and participant aliases with a broad
spaCy NER layer.  The resulting variant is a stress test, not a claim that all
product or brand identity has been removed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


VARIANT = "broad_identity_mask_v1"
NER_LABELS = frozenset({"PERSON", "ORG", "PRODUCT", "NORP", "FAC"})
PLACEHOLDERS = {
    "ISSUER": "[ISSUER]",
    "TICKER": "[TICKER]",
    "PERSON": "[PERSON]",
    "ORG": "[ORG]",
    "PRODUCT": "[PRODUCT]",
    "NORP": "[NORP]",
    "FAC": "[FAC]",
}

# Deterministic, call-local aliases dominate broad statistical NER spans.
PRIORITIES = {
    "call_ticker": 120,
    "call_issuer": 115,
    "opening_claimed_issuer": 112,
    "pair_speaker": 110,
    "pair_speaker_component": 109,
    "prepared_speaker": 108,
    "prepared_speaker_component": 107,
    "global_ticker": 100,
    "global_issuer": 95,
    "ner_PRODUCT": 75,
    "ner_PERSON": 72,
    "ner_ORG": 70,
    "ner_NORP": 68,
    "ner_FAC": 66,
}

COMPANY_COLUMNS = (
    "canonical_company",
    "original_company",
    "company",
)
TICKER_COLUMNS = (
    "canonical_asset_symbol",
    "canonical_query_symbol",
    "canonical_issuer_id",
    "asset_symbol",
    "query_symbol",
    "original_asset_symbol",
    "original_query_symbol",
)
NON_PERSON_LABELS = frozenset(
    {
        "analysts",
        "analyst",
        "executives",
        "executive",
        "company participants",
        "corporate participants",
        "conference call participants",
        "participants",
        "presentation",
        "question and answer session",
        "questions and answers",
        "editor",
        "unknown speaker",
        "unknown analyst",
        "unidentified analyst",
        "unidentified participant",
        "unidentified speaker",
        "operator",
    }
)
LEGAL_SUFFIX_RE = re.compile(
    r"(?:\s*[,.-]?\s*)\b(?:incorporated|corporation|company|holdings?|group|"
    r"limited|ordinary shares?|common stock|common|class [a-z]|inc|corp|co|"
    r"ltd|plc|n\.?v\.?)\.?$",
    re.IGNORECASE,
)
SPEAKER_RE = re.compile(
    r"([A-Z][A-Za-z0-9 .,&'\N{RIGHT SINGLE QUOTATION MARK}()/\\-]{1,100})\s*:\s+"
)
PERSON_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\N{RIGHT SINGLE QUOTATION MARK}.-]*")
SYMBOL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9.-]{0,15}$")
CORPORATE_TITLE_RE = re.compile(
    r",\s*(?:chief|co-chief|president|vice president|senior vice president|"
    r"executive vice president|chair(?:man|woman)?|director|founder|partner|"
    r"analyst|investor relations|controller|treasurer)\b.*$",
    re.IGNORECASE,
)
TRADE_DESCRIPTOR_TOKENS = frozenset(
    {
        "airlines",
        "automation",
        "bancorp",
        "bank",
        "beverage",
        "brands",
        "communications",
        "energy",
        "financial",
        "health",
        "healthcare",
        "industries",
        "materials",
        "networks",
        "payments",
        "pharmaceuticals",
        "resources",
        "semiconductor",
        "semiconductors",
        "services",
        "software",
        "solutions",
        "stores",
        "systems",
        "technology",
        "technologies",
        "therapeutics",
    }
)
UNSAFE_SINGLE_COMPANY_ALIASES = frozenset(
    {
        "advanced",
        "american",
        "capital",
        "central",
        "common",
        "digital",
        "eastern",
        "federal",
        "first",
        "general",
        "global",
        "home",
        "international",
        "key",
        "national",
        "new",
        "northern",
        "one",
        "our",
        "principal",
        "public",
        "real",
        "service",
        "southern",
        "target",
        "the",
        "three",
        "today",
        "two",
        "united",
        "universal",
        "western",
        "your",
    }
)
OPENING_NONIDENTITY_TOKENS = frozenset(
    {"our", "the", "today", "your", "quarterly", "fiscal"}
)
OPENING_NONIDENTITY_RE = re.compile(
    r"(?i)^(?:(?:first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter|"
    r"q[1-4]|quarterly|fiscal(?:\s+year)?|(?:19|20)\d{2})"
    r"(?:\s+(?:19|20)\d{2})?$"
)
OPENING_PREFIX_RE = re.compile(
    r"(?i)^(?:welcome\s+to\s+)?(?:today['\N{RIGHT SINGLE QUOTATION MARK}]s\s+|"
    r"earnings\s+call\s+for\s+|this\s+review\s+of\s+|"
    r"(?:(?:first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter|q[1-4])"
    r"(?:\s+(?:19|20)\d{2})?\s+)"
)
OPENING_TRAILING_RE = re.compile(
    r"(?i)\s+(?:teleconference|webcast|results?|reports?|fiscal(?:\s+year)?|"
    r"(?:first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter|q[1-4]|"
    r"(?:19|20)\d{2})\b.*$"
)
GENERIC_PERSON_COMPONENTS = frozenset(
    {
        "analyst",
        "answer",
        "company",
        "editor",
        "executive",
        "management",
        "operator",
        "participant",
        "question",
        "representative",
        "research",
        "speaker",
        "unidentified",
        "unknown",
    }
)
AMBIGUOUS_PERSON_COMPONENTS = frozenset(
    {
        "april",
        "art",
        "august",
        "bill",
        "black",
        "brown",
        "chip",
        "faith",
        "grant",
        "green",
        "guy",
        "hall",
        "hope",
        "joy",
        "june",
        "king",
        "lane",
        "long",
        "mark",
        "may",
        "page",
        "price",
        "stone",
        "white",
        "will",
        "wood",
        "young",
    }
)


@dataclass(frozen=True, order=True)
class AliasSpec:
    text: str
    entity_type: str
    source: str
    priority: int


@dataclass(frozen=True)
class CallIdentity:
    issuer_call_id: str
    aliases: tuple[AliasSpec, ...]
    opening_claimed_present: bool = False
    opening_claimed_alias_count: int = 0


@dataclass(frozen=True)
class SpanCandidate:
    start: int
    end: int
    entity_type: str
    source: str
    priority: int
    surface: str


@dataclass(frozen=True)
class MaskResult:
    masked_text: str
    selected_spans: tuple[SpanCandidate, ...]
    audit: dict[str, Any]


def sha256_text(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip(" \t\r\n,;:")
    if text.lower() in {"", "nan", "none", "null", "<na>"}:
        return ""
    return text


def ascii_fold(value: str) -> str:
    return unicodedata.normalize("NFKD", value).encode(
        "ascii", "ignore"
    ).decode("ascii")


def is_plain_single_company_alias(value: str) -> bool:
    """Flag ordinary-word-shaped issuer aliases for case-sensitive matching."""

    return bool(re.fullmatch(r"[A-Za-z]{3,}", clean_value(value)))


def _identity_tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", ascii_fold(value).lower()))


def _tokens_are_contiguous_subset(
    candidate: tuple[str, ...], reference: tuple[str, ...]
) -> bool:
    if not candidate or len(candidate) > len(reference):
        return False
    width = len(candidate)
    return any(
        reference[start : start + width] == candidate
        for start in range(0, len(reference) - width + 1)
    )


def company_alias_variants(value: Any) -> tuple[str, ...]:
    raw = clean_value(value)
    if not raw:
        return ()
    candidates = {raw}
    without_exchange = re.sub(
        r"\s+\([^)]*(?:CIK|NYSE|NASDAQ|[A-Z]{1,6})[^)]*\)\s*$",
        "",
        raw,
        flags=re.IGNORECASE,
    ).strip()
    if without_exchange:
        candidates.add(without_exchange)
    stripped = without_exchange
    while stripped:
        updated = LEGAL_SUFFIX_RE.sub("", stripped).strip(" ,.-")
        if updated == stripped:
            break
        stripped = updated
        if stripped:
            candidates.add(stripped)
    # Add a conservative trade-name stem for names such as "Agilent
    # Technologies".  Generic one-token stems (for example "Global") are
    # explicitly excluded because they would destroy ordinary semantics.
    trade_tokens = stripped.split()
    while (
        len(trade_tokens) > 1
        and re.sub(r"[^a-z]", "", trade_tokens[-1].lower())
        in TRADE_DESCRIPTOR_TOKENS
    ):
        trade_tokens.pop()
    if len(trade_tokens) == 1:
        trade_name = trade_tokens[0].strip(" ,.-")
        normalized_trade = re.sub(r"[^a-z0-9]", "", trade_name.lower())
        if (
            len(normalized_trade) >= 4
            and normalized_trade not in UNSAFE_SINGLE_COMPANY_ALIASES
            and not normalized_trade.isdigit()
        ):
            candidates.add(trade_name)
    # Preserve punctuation-bearing legal names while adding a plain-ASCII form.
    for candidate in tuple(candidates):
        folded = clean_value(ascii_fold(candidate))
        if folded:
            candidates.add(folded)
    return tuple(
        sorted(
            {
                candidate
                for candidate in candidates
                if 3 <= len(re.sub(r"[^A-Za-z0-9]", "", candidate))
                and len(candidate) <= 120
            },
            key=lambda item: (-len(item), item.casefold(), item),
        )
    )


def opening_company_alias_variants(
    value: Any,
    known_company_aliases: frozenset[str] | None = None,
    call_company_aliases: frozenset[str] | None = None,
    call_ticker_aliases: frozenset[str] | None = None,
) -> tuple[str, ...]:
    """Return safe issuer aliases from the noisy opening-claim parser.

    The source field sometimes contains pronouns or timing phrases rather than
    an issuer.  Aliases are retained only when corroborated by the global
    historical-name registry or the call's own company/ticker aliases.
    """

    raw = clean_value(value)
    if not raw:
        return ()
    normalized = re.sub(r"[^a-z0-9 ]+", " ", ascii_fold(raw).lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if OPENING_NONIDENTITY_RE.fullmatch(normalized):
        return ()
    normalized_tokens = normalized.split()
    if not normalized_tokens or normalized in OPENING_NONIDENTITY_TOKENS:
        return ()
    if normalized_tokens[0] in {"our", "the", "your"}:
        return ()
    cleaned = OPENING_PREFIX_RE.sub("", raw).strip(" ,.-")
    cleaned = OPENING_TRAILING_RE.sub("", cleaned).strip(" ,.-")
    cleaned = re.sub(r"['\N{RIGHT SINGLE QUOTATION MARK}]s?$", "", cleaned).strip(
        " ,.-"
    )
    if len(cleaned) > 120 or len(cleaned.split()) > 10:
        return ()
    variants = company_alias_variants(cleaned)
    compact = re.sub(r"[^A-Za-z0-9]", "", cleaned)
    if (
        not variants
        and 2 <= len(compact) <= 5
        and compact.isupper()
        and compact not in {"AI", "IT", "US"}
    ):
        variants = (cleaned,)
    known = known_company_aliases or frozenset()
    call_companies = call_company_aliases or frozenset()
    call_tickers = call_ticker_aliases or frozenset()
    company_token_sets = tuple(_identity_tokens(alias) for alias in call_companies)
    corroborated: list[str] = []
    for alias in variants:
        folded = alias.casefold()
        tokens = _identity_tokens(alias)
        if (
            folded in known
            or folded in call_tickers
            or any(
                _tokens_are_contiguous_subset(tokens, reference)
                for reference in company_token_sets
            )
        ):
            corroborated.append(alias)
    return tuple(corroborated)


def ticker_alias_variants(value: Any) -> tuple[str, ...]:
    raw = clean_value(value).upper()
    if not raw or not SYMBOL_RE.fullmatch(raw):
        return ()
    candidates = {raw}
    base = re.sub(r"-\d{6}$", "", raw)
    if base:
        candidates.add(base)
    for candidate in tuple(candidates):
        if "." in candidate:
            candidates.add(candidate.replace(".", "-"))
        if "-" in candidate and not re.search(r"-\d{6}$", candidate):
            candidates.add(candidate.replace("-", "."))
    return tuple(sorted(candidates, key=lambda item: (-len(item), item)))


def extract_person_name_variants(value: Any) -> tuple[str, ...]:
    label = clean_value(value)
    if not label:
        return ()
    label = re.split(r"\s+[\-\N{EN DASH}\N{EM DASH}]\s+", label, maxsplit=1)[0]
    label = CORPORATE_TITLE_RE.sub("", label).strip(" ,.-")
    label = re.sub(r"^(?:mr|mrs|ms|dr|prof)\.?\s+", "", label, flags=re.I)
    normalized_label = re.sub(r"[^a-z ]+", " ", label.lower())
    normalized_label = re.sub(r"\s+", " ", normalized_label).strip()
    if normalized_label in NON_PERSON_LABELS or "operator" in normalized_label:
        return ()
    tokens = PERSON_TOKEN_RE.findall(label)
    if not 2 <= len(tokens) <= 7:
        return ()
    if any(token.lower() in {"session", "participants", "presentation"} for token in tokens):
        return ()
    if any(
        re.sub(r"[^a-z]", "", token.lower()) in GENERIC_PERSON_COMPONENTS
        for token in tokens
    ):
        return ()
    candidates = {" ".join(tokens)}
    folded = clean_value(ascii_fold(" ".join(tokens)))
    if folded:
        candidates.add(folded)
    return tuple(
        sorted(candidates, key=lambda item: (-len(item), item.casefold(), item))
    )


def extract_prepared_speaker_names(text: Any) -> tuple[str, ...]:
    names: set[str] = set()
    for match in SPEAKER_RE.finditer(str(text or "")):
        names.update(extract_person_name_variants(match.group(1)))
    return tuple(sorted(names, key=lambda item: (-len(item), item.casefold(), item)))


def person_component_variants(value: Any) -> tuple[str, ...]:
    """Extract case-sensitive first/last name clues from a full roster name."""

    full_names = extract_person_name_variants(value)
    components: set[str] = set()
    for full_name in full_names:
        tokens = [token.strip(".-") for token in PERSON_TOKEN_RE.findall(full_name)]
        tokens = [token for token in tokens if len(re.sub(r"[^A-Za-z]", "", token)) >= 4]
        if not tokens:
            continue
        for token in {tokens[0], tokens[-1]}:
            normalized = re.sub(r"[^a-z]", "", token.lower())
            if normalized in GENERIC_PERSON_COMPONENTS or normalized in (
                AMBIGUOUS_PERSON_COMPONENTS
            ):
                continue
            components.add(token)
            folded = clean_value(ascii_fold(token))
            if folded:
                components.add(folded)
    return tuple(
        sorted(components, key=lambda item: (-len(item), item.casefold(), item))
    )


def load_pair_rosters(
    path: Path,
    allowed_call_ids: set[str] | None = None,
) -> dict[str, tuple[str, ...]]:
    columns = ["issuer_call_id", "question_speaker", "answer_speakers"]
    frame = pd.read_parquet(path, columns=columns)
    if allowed_call_ids is not None:
        frame = frame[frame["issuer_call_id"].astype(str).isin(allowed_call_ids)]
    rosters: dict[str, set[str]] = {}
    for row in frame.itertuples(index=False):
        call_id = str(row.issuer_call_id)
        names = rosters.setdefault(call_id, set())
        names.update(extract_person_name_variants(row.question_speaker))
        for label in clean_value(row.answer_speakers).split("|"):
            names.update(extract_person_name_variants(label))
    return {
        call_id: tuple(
            sorted(names, key=lambda item: (-len(item), item.casefold(), item))
        )
        for call_id, names in rosters.items()
    }


def _row_value(row: Mapping[str, Any], column: str) -> Any:
    try:
        return row[column]
    except (KeyError, TypeError):
        return ""


def build_call_identities(
    calls: pd.DataFrame,
    pair_rosters: Mapping[str, Sequence[str]],
    known_company_aliases: frozenset[str] | None = None,
) -> dict[str, CallIdentity]:
    identities: dict[str, CallIdentity] = {}
    for _, series in calls.iterrows():
        row = series.to_dict()
        call_id = str(row["issuer_call_id"])
        aliases: set[AliasSpec] = set()
        call_company_aliases: set[str] = set()
        call_ticker_aliases: set[str] = set()
        for column in COMPANY_COLUMNS:
            for alias in company_alias_variants(_row_value(row, column)):
                call_company_aliases.add(alias)
                aliases.add(
                    AliasSpec(
                        alias,
                        "ISSUER",
                        "call_issuer",
                        PRIORITIES["call_issuer"],
                    )
                )
        for column in TICKER_COLUMNS:
            for alias in ticker_alias_variants(_row_value(row, column)):
                call_ticker_aliases.add(alias.casefold())
                aliases.add(
                    AliasSpec(
                        alias,
                        "TICKER",
                        "call_ticker",
                        PRIORITIES["call_ticker"],
                    )
                )
        opening_value = clean_value(_row_value(row, "opening_claimed_issuer"))
        opening_aliases = opening_company_alias_variants(
            opening_value,
            known_company_aliases,
            frozenset(call_company_aliases),
            frozenset(call_ticker_aliases),
        )
        for alias in opening_aliases:
            aliases.add(
                AliasSpec(
                    alias,
                    "ISSUER",
                    "opening_claimed_issuer",
                    PRIORITIES["opening_claimed_issuer"],
                )
            )
        for alias in extract_prepared_speaker_names(
            _row_value(row, "prepared_doc")
        ):
            aliases.add(
                AliasSpec(
                    alias,
                    "PERSON",
                    "prepared_speaker",
                    PRIORITIES["prepared_speaker"],
                )
                )
            for component in person_component_variants(alias):
                aliases.add(
                    AliasSpec(
                        component,
                        "PERSON",
                        "prepared_speaker_component",
                        PRIORITIES["prepared_speaker_component"],
                    )
                )
        for alias in pair_rosters.get(call_id, ()):
            for variant in extract_person_name_variants(alias):
                aliases.add(
                    AliasSpec(
                        variant,
                        "PERSON",
                        "pair_speaker",
                        PRIORITIES["pair_speaker"],
                    )
                )
            for component in person_component_variants(alias):
                aliases.add(
                    AliasSpec(
                        component,
                        "PERSON",
                        "pair_speaker_component",
                        PRIORITIES["pair_speaker_component"],
                    )
                )
        identities[call_id] = CallIdentity(
            issuer_call_id=call_id,
            aliases=tuple(
                sorted(
                    aliases,
                    key=lambda item: (
                        -item.priority,
                        -len(item.text),
                        item.entity_type,
                        item.text.casefold(),
                        item.text,
                        item.source,
                    ),
                )
            ),
            opening_claimed_present=bool(opening_value),
            opening_claimed_alias_count=len(opening_aliases),
        )
    return identities


def build_global_aliases(
    calls: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
) -> tuple[AliasSpec, ...]:
    aliases: set[AliasSpec] = set()
    for column in COMPANY_COLUMNS:
        if column not in calls:
            continue
        for value in calls[column].dropna().unique():
            for alias in company_alias_variants(value):
                aliases.add(
                    AliasSpec(
                        alias,
                        "ISSUER",
                        "global_issuer",
                        PRIORITIES["global_issuer"],
                    )
                )
    if metadata is not None and "Name" in metadata:
        for value in metadata["Name"].dropna().unique():
            for alias in company_alias_variants(value):
                aliases.add(
                    AliasSpec(
                        alias,
                        "ISSUER",
                        "global_issuer",
                        PRIORITIES["global_issuer"],
                    )
                )
    for column in TICKER_COLUMNS:
        if column not in calls:
            continue
        for value in calls[column].dropna().unique():
            for alias in ticker_alias_variants(value):
                aliases.add(
                    AliasSpec(
                        alias,
                        "TICKER",
                        "global_ticker",
                        PRIORITIES["global_ticker"],
                    )
                )
    if metadata is not None and "Symbol" in metadata:
        for value in metadata["Symbol"].dropna().unique():
            for alias in ticker_alias_variants(value):
                aliases.add(
                    AliasSpec(
                        alias,
                        "TICKER",
                        "global_ticker",
                        PRIORITIES["global_ticker"],
                    )
                )
    return tuple(
        sorted(
            aliases,
            key=lambda item: (
                item.entity_type,
                -len(item.text),
                item.text.casefold(),
                item.text,
                item.source,
                item.priority,
            ),
        )
    )


def identity_registry_sha256(
    global_aliases: Sequence[AliasSpec],
    call_identities: Mapping[str, CallIdentity],
) -> str:
    digest = hashlib.sha256()
    for alias in global_aliases:
        digest.update(
            json.dumps(asdict(alias), sort_keys=True).encode("utf-8") + b"\n"
        )
    for call_id in sorted(call_identities):
        identity = call_identities[call_id]
        digest.update(
            json.dumps(
                {
                    "issuer_call_id": call_id,
                    "opening_claimed_present": identity.opening_claimed_present,
                    "opening_claimed_alias_count": identity.opening_claimed_alias_count,
                },
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
        for alias in identity.aliases:
            digest.update(
                json.dumps(asdict(alias), sort_keys=True).encode("utf-8")
                + b"\n"
            )
    return digest.hexdigest()


@lru_cache(maxsize=256)
def _short_ticker_pattern(tickers: tuple[str, ...]) -> re.Pattern[str]:
    # One- and two-character tickers are ordinary English tokens.  They are
    # masked only with a ticker/symbol label or a punctuated exchange label.
    alternatives = "|".join(
        re.escape(ticker)
        for ticker in sorted(set(tickers), key=lambda item: (-len(item), item))
    )
    return re.compile(
        rf"(?i:(?:(?:\bticker|\bsymbol)\s*[:=]?\s*)|"
        rf"(?:(?:\bnyse|\bnasdaq|\bamex)\s*[:=]\s*))"
        rf"(?P<ticker>(?:{alternatives}))(?![A-Za-z0-9])",
        re.IGNORECASE,
    )


def _explicit_short_ticker_matches(text: str, ticker: str) -> Iterable[re.Match[str]]:
    return _short_ticker_pattern((ticker.upper(),)).finditer(text)


def _explicit_short_ticker_matches_many(
    text: str, tickers: tuple[str, ...]
) -> Iterable[re.Match[str]]:
    if not tickers:
        return ()
    return _short_ticker_pattern(tickers).finditer(text)


@lru_cache(maxsize=1_024)
def _compiled_identity_aliases(
    aliases: tuple[AliasSpec, ...],
) -> tuple[tuple[re.Pattern[str], dict[str, AliasSpec], bool], ...]:
    """Compile call-local aliases into three scans rather than N scans."""

    groups: dict[tuple[str, bool], list[AliasSpec]] = {
        ("long_ticker", False): [],
        ("case_sensitive", False): [],
        ("case_insensitive", True): [],
    }
    for alias in aliases:
        if alias.entity_type == "TICKER":
            if len(alias.text) >= 3:
                groups[("long_ticker", False)].append(alias)
        elif (
            alias.source.endswith("_speaker_component")
            or alias.source == "opening_claimed_issuer"
            or (
                alias.entity_type == "ISSUER"
                and is_plain_single_company_alias(alias.text)
            )
        ):
            groups[("case_sensitive", False)].append(alias)
        else:
            groups[("case_insensitive", True)].append(alias)

    output: list[tuple[re.Pattern[str], dict[str, AliasSpec], bool]] = []
    for (_, ignore_case), specs in groups.items():
        if not specs:
            continue
        lookup: dict[str, AliasSpec] = {}
        for spec in specs:
            key = spec.text.casefold() if ignore_case else spec.text
            incumbent = lookup.get(key)
            if incumbent is None or (
                spec.priority,
                len(spec.text),
                spec.source,
                spec.text,
            ) > (
                incumbent.priority,
                len(incumbent.text),
                incumbent.source,
                incumbent.text,
            ):
                lookup[key] = spec
        alternatives = sorted(
            {spec.text for spec in lookup.values()},
            key=lambda item: (-len(item), item.casefold(), item),
        )
        flags = re.IGNORECASE if ignore_case else 0
        pattern = re.compile(
            r"(?<![A-Za-z0-9])(?:"
            + "|".join(re.escape(item) for item in alternatives)
            + r")(?![A-Za-z0-9])",
            flags,
        )
        output.append((pattern, lookup, ignore_case))
    return tuple(output)


class IdentityMasker:
    """Resolve deterministic aliases and broad NER spans on one source text."""

    def __init__(
        self,
        nlp: Any,
        global_aliases: Sequence[AliasSpec],
        ner_labels: frozenset[str] = NER_LABELS,
    ) -> None:
        from spacy.matcher import PhraseMatcher

        self.nlp = nlp
        self.ner_labels = frozenset(ner_labels)
        self.global_aliases = tuple(global_aliases)
        self.company_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        company_texts = sorted(
            {
                alias.text
                for alias in global_aliases
                if alias.entity_type == "ISSUER"
                and not is_plain_single_company_alias(alias.text)
            },
            key=lambda item: (-len(item), item.casefold(), item),
        )
        if company_texts:
            self.company_matcher.add(
                "GLOBAL_ISSUER",
                [nlp.make_doc(alias) for alias in company_texts],
            )
        ambiguous_company_texts = sorted(
            {
                variant
                for alias in global_aliases
                if alias.entity_type == "ISSUER"
                and is_plain_single_company_alias(alias.text)
                for variant in (alias.text, alias.text.upper())
            },
            key=lambda item: (-len(item), item),
        )
        self.ambiguous_company_pattern = (
            re.compile(
                r"(?<![A-Za-z0-9])(?:"
                + "|".join(re.escape(alias) for alias in ambiguous_company_texts)
                + r")(?![A-Za-z0-9])"
            )
            if ambiguous_company_texts
            else None
        )
        long_tickers = sorted(
            {
                alias.text
                for alias in global_aliases
                if alias.entity_type == "TICKER" and len(alias.text) >= 3
            },
            key=lambda item: (-len(item), item),
        )
        self.global_ticker_pattern = (
            re.compile(
                r"(?<![A-Za-z0-9])(?:"
                + "|".join(re.escape(alias) for alias in long_tickers)
                + r")(?![A-Za-z0-9])"
            )
            if long_tickers
            else None
        )
        self.global_short_tickers = tuple(
            sorted(
                {
                    alias.text
                    for alias in global_aliases
                    if alias.entity_type == "TICKER" and len(alias.text) <= 2
                },
                key=lambda item: (-len(item), item),
            )
        )

    def _deterministic_candidates(
        self,
        text: str,
        doc: Any,
        identity: CallIdentity,
    ) -> tuple[list[SpanCandidate], int]:
        candidates: list[SpanCandidate] = []
        short_ticker_aliases = 0
        for _, start, end in self.company_matcher(doc):
            span = doc[start:end]
            candidates.append(
                SpanCandidate(
                    span.start_char,
                    span.end_char,
                    "ISSUER",
                    "global_issuer",
                    PRIORITIES["global_issuer"],
                    span.text,
                )
            )
        if self.ambiguous_company_pattern is not None:
            for match in self.ambiguous_company_pattern.finditer(text):
                candidates.append(
                    SpanCandidate(
                        match.start(),
                        match.end(),
                        "ISSUER",
                        "global_issuer",
                        PRIORITIES["global_issuer"],
                        match.group(0),
                    )
                )
        if self.global_ticker_pattern is not None:
            for match in self.global_ticker_pattern.finditer(text):
                candidates.append(
                    SpanCandidate(
                        match.start(),
                        match.end(),
                        "TICKER",
                        "global_ticker",
                        PRIORITIES["global_ticker"],
                        match.group(0),
                    )
                )
        for match in _explicit_short_ticker_matches_many(
            text, self.global_short_tickers
        ):
            candidates.append(
                SpanCandidate(
                    match.start("ticker"),
                    match.end("ticker"),
                    "TICKER",
                    "global_ticker",
                    PRIORITIES["global_ticker"],
                    match.group("ticker"),
                )
            )
        for alias in identity.aliases:
            if alias.entity_type == "TICKER":
                if len(alias.text) <= 2:
                    short_ticker_aliases += 1
                    for match in _explicit_short_ticker_matches(text, alias.text):
                        candidates.append(
                            SpanCandidate(
                                match.start("ticker"),
                                match.end("ticker"),
                                alias.entity_type,
                                alias.source,
                                alias.priority,
                                match.group("ticker"),
                            )
                        )
                    continue
        for pattern, lookup, ignore_case in _compiled_identity_aliases(
            identity.aliases
        ):
            for match in pattern.finditer(text):
                surface = match.group(0)
                key = surface.casefold() if ignore_case else surface
                alias = lookup[key]
                candidates.append(
                    SpanCandidate(
                        match.start(),
                        match.end(),
                        alias.entity_type,
                        alias.source,
                        alias.priority,
                        surface,
                    )
                )
        return candidates, short_ticker_aliases

    def _ner_candidates(self, doc: Any) -> list[SpanCandidate]:
        candidates: list[SpanCandidate] = []
        for entity in doc.ents:
            if entity.label_ not in self.ner_labels:
                continue
            source = f"ner_{entity.label_}"
            candidates.append(
                SpanCandidate(
                    entity.start_char,
                    entity.end_char,
                    entity.label_,
                    source,
                    PRIORITIES[source],
                    entity.text,
                )
            )
        return candidates

    @staticmethod
    def resolve_overlaps(
        candidates: Sequence[SpanCandidate],
    ) -> tuple[tuple[SpanCandidate, ...], int]:
        unique = {
            (
                item.start,
                item.end,
                item.entity_type,
                item.source,
                item.priority,
            ): item
            for item in candidates
            if item.end > item.start
        }
        ordered = sorted(
            unique.values(),
            key=lambda item: (
                -item.priority,
                -(item.end - item.start),
                item.start,
                item.end,
                item.entity_type,
                item.source,
            ),
        )
        selected: list[SpanCandidate] = []
        for item in ordered:
            if any(
                item.start < accepted.end and accepted.start < item.end
                for accepted in selected
            ):
                continue
            selected.append(item)
        selected.sort(key=lambda item: (item.start, item.end))
        return tuple(selected), len(unique) - len(selected)

    @staticmethod
    def apply_spans(text: str, spans: Sequence[SpanCandidate]) -> str:
        pieces: list[str] = []
        cursor = 0
        for span in spans:
            pieces.append(text[cursor : span.start])
            pieces.append(PLACEHOLDERS[span.entity_type])
            cursor = span.end
        pieces.append(text[cursor:])
        return re.sub(r"[ \t]+", " ", "".join(pieces)).strip()

    def residual_aliases(
        self,
        masked_text: str,
        identity: CallIdentity,
        limit: int = 12,
    ) -> tuple[int, list[str]]:
        residual: list[str] = []
        doc = self.nlp.make_doc(masked_text)
        for _, start, end in self.company_matcher(doc):
            residual.append(doc[start:end].text)
            if len(residual) >= limit:
                return len(residual), residual
        if self.ambiguous_company_pattern is not None:
            for match in self.ambiguous_company_pattern.finditer(masked_text):
                residual.append(match.group(0))
                if len(residual) >= limit:
                    return len(residual), residual
        if self.global_ticker_pattern is not None:
            for match in self.global_ticker_pattern.finditer(masked_text):
                residual.append(match.group(0))
                if len(residual) >= limit:
                    return len(residual), residual
        for match in _explicit_short_ticker_matches_many(
            masked_text, self.global_short_tickers
        ):
            residual.append(match.group("ticker"))
            if len(residual) >= limit:
                return len(residual), residual
        for alias in identity.aliases:
            if alias.entity_type == "TICKER":
                if len(alias.text) <= 2:
                    matches = _explicit_short_ticker_matches(
                        masked_text, alias.text
                    )
                    for match in matches:
                        residual.append(match.group("ticker"))
                        if len(residual) >= limit:
                            return len(residual), residual
                    continue
        for pattern, _, _ in _compiled_identity_aliases(identity.aliases):
            for match in pattern.finditer(masked_text):
                residual.append(match.group(0))
                if len(residual) >= limit:
                    return len(residual), residual
        return len(residual), residual

    def mask(
        self,
        text: str,
        identity: CallIdentity,
        doc: Any | None = None,
    ) -> MaskResult:
        source_text = str(text or "")
        if doc is None:
            doc = self.nlp(source_text)
        deterministic, short_ticker_aliases = self._deterministic_candidates(
            source_text, doc, identity
        )
        ner = self._ner_candidates(doc)
        selected, overlap_drops = self.resolve_overlaps([*deterministic, *ner])
        masked = self.apply_spans(source_text, selected)
        residual_count, residual_examples = self.residual_aliases(
            masked, identity
        )
        type_counts = Counter(span.entity_type for span in selected)
        source_counts = Counter(span.source for span in selected)
        masked_characters = sum(span.end - span.start for span in selected)
        audit: dict[str, Any] = {
            "variant": VARIANT,
            "issuer_call_id": identity.issuer_call_id,
            "original_sha256": sha256_text(source_text),
            "masked_sha256": sha256_text(masked),
            "original_characters": len(source_text),
            "masked_characters": len(masked),
            "replaced_source_characters": int(masked_characters),
            "original_words": len(source_text.split()),
            "masked_words": len(masked.split()),
            "candidate_spans": len(deterministic) + len(ner),
            "selected_spans": len(selected),
            "overlap_dropped": overlap_drops,
            "deterministic_candidates": len(deterministic),
            "ner_candidates": len(ner),
            "short_ticker_aliases_context_only": short_ticker_aliases,
            "residual_alias_count": residual_count,
            "residual_alias_examples": json.dumps(
                residual_examples, ensure_ascii=True
            ),
            "empty_after_mask": not bool(masked.split()),
            "opening_claimed_present": identity.opening_claimed_present,
            "opening_claimed_alias_count": identity.opening_claimed_alias_count,
        }
        for entity_type in PLACEHOLDERS:
            audit[f"selected_{entity_type.lower()}"] = int(
                type_counts[entity_type]
            )
        for source in PRIORITIES:
            audit[f"source_{source}"] = int(source_counts[source])
        return MaskResult(masked, selected, audit)


def registry_summary(
    global_aliases: Sequence[AliasSpec],
    call_identities: Mapping[str, CallIdentity],
) -> dict[str, Any]:
    global_types = Counter(alias.entity_type for alias in global_aliases)
    call_sources: Counter[str] = Counter()
    calls_with_speakers = 0
    calls_with_opening = 0
    calls_with_opening_claim = 0
    calls_with_rejected_opening_claim = 0
    calls_with_short_tickers = 0
    for identity in call_identities.values():
        sources = {alias.source for alias in identity.aliases}
        call_sources.update(alias.source for alias in identity.aliases)
        calls_with_speakers += int(
            bool({"prepared_speaker", "pair_speaker"}.intersection(sources))
        )
        calls_with_opening += int("opening_claimed_issuer" in sources)
        calls_with_opening_claim += int(identity.opening_claimed_present)
        calls_with_rejected_opening_claim += int(
            identity.opening_claimed_present
            and identity.opening_claimed_alias_count == 0
        )
        calls_with_short_tickers += int(
            any(
                alias.entity_type == "TICKER" and len(alias.text) <= 2
                for alias in identity.aliases
            )
        )
    return {
        "variant": VARIANT,
        "global_aliases": len(global_aliases),
        "global_aliases_by_type": dict(sorted(global_types.items())),
        "global_plain_single_issuer_aliases_case_sensitive": sum(
            alias.entity_type == "ISSUER"
            and is_plain_single_company_alias(alias.text)
            for alias in global_aliases
        ),
        "call_identities": len(call_identities),
        "call_aliases": int(sum(call_sources.values())),
        "call_aliases_by_source": dict(sorted(call_sources.items())),
        "calls_with_speaker_aliases": calls_with_speakers,
        "calls_with_opening_claimed_alias": calls_with_opening,
        "calls_with_opening_claim": calls_with_opening_claim,
        "calls_with_rejected_opening_claim": calls_with_rejected_opening_claim,
        "calls_with_short_ticker_context_rule": calls_with_short_tickers,
        "registry_sha256": identity_registry_sha256(
            global_aliases, call_identities
        ),
        "warning": (
            "Broad automated identity masking; product/brand coverage is not "
            "gold-annotated or guaranteed complete."
        ),
    }
