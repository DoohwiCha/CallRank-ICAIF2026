from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from identity_masking import (  # noqa: E402
    PRIORITIES,
    AliasSpec,
    IdentityMasker,
    SpanCandidate,
    build_call_identities,
    build_global_aliases,
    company_alias_variants,
    identity_registry_sha256,
)


class FrozenIdentityMaskerTests(unittest.TestCase):
    def test_alias_construction_uses_frozen_sources_and_priorities(self) -> None:
        calls = pd.DataFrame(
            [
                {
                    "issuer_call_id": "call-1",
                    "canonical_company": "Example Technologies, Inc.",
                    "original_company": "Example Technologies",
                    "canonical_asset_symbol": "EXM",
                    "canonical_query_symbol": "EXM-US",
                    "canonical_issuer_id": "issuer-1",
                    "prepared_doc": "Alex Rivera: Thank you for joining us.",
                    "opening_claimed_issuer": "Example Technologies",
                }
            ]
        )
        global_aliases = build_global_aliases(calls)
        identities = build_call_identities(
            calls,
            {"call-1": ("Jordan Morgan",)},
            frozenset(
                alias.text.casefold()
                for alias in global_aliases
                if alias.entity_type == "ISSUER"
            ),
        )
        identity = identities["call-1"]
        sources = {alias.source for alias in identity.aliases}
        self.assertIn("call_issuer", sources)
        self.assertIn("call_ticker", sources)
        self.assertIn("prepared_speaker", sources)
        self.assertIn("pair_speaker", sources)
        self.assertIn("opening_claimed_issuer", sources)
        self.assertTrue(
            all(alias.priority == PRIORITIES[alias.source] for alias in identity.aliases)
        )
        self.assertIn("Example", company_alias_variants("Example Technologies, Inc."))

    def test_overlap_resolution_prefers_call_local_identity(self) -> None:
        issuer = "Example Technologies"
        person = "Alex Rivera"
        person_start = len(issuer) + len(" by ")
        candidates = [
            SpanCandidate(0, len(issuer), "ORG", "ner_ORG", 70, issuer),
            SpanCandidate(
                0,
                len(issuer),
                "ISSUER",
                "call_issuer",
                PRIORITIES["call_issuer"],
                issuer,
            ),
            SpanCandidate(
                person_start,
                person_start + len(person),
                "PERSON",
                "ner_PERSON",
                72,
                person,
            ),
        ]
        selected, dropped = IdentityMasker.resolve_overlaps(candidates)
        self.assertEqual(dropped, 1)
        self.assertEqual(selected[0].source, "call_issuer")
        self.assertEqual(
            IdentityMasker.apply_spans(
                f"{issuer} by {person}",
                selected,
            ),
            "[ISSUER] by [PERSON]",
        )

    def test_registry_hash_is_order_stable(self) -> None:
        calls = pd.DataFrame(
            [
                {
                    "issuer_call_id": "call-2",
                    "canonical_company": "Second Systems Corp.",
                    "canonical_asset_symbol": "SND",
                    "prepared_doc": "",
                },
                {
                    "issuer_call_id": "call-1",
                    "canonical_company": "First Systems Corp.",
                    "canonical_asset_symbol": "FST",
                    "prepared_doc": "",
                },
            ]
        )
        aliases = build_global_aliases(calls)
        identities = build_call_identities(calls, {})
        first = identity_registry_sha256(aliases, identities)
        reversed_aliases = tuple(reversed(aliases))
        self.assertNotEqual(first, identity_registry_sha256(reversed_aliases, identities))
        reordered_identities = {
            key: identities[key] for key in reversed(list(identities))
        }
        self.assertEqual(first, identity_registry_sha256(aliases, reordered_identities))

    def test_alias_spec_is_safe_serializable_protocol_state(self) -> None:
        alias = AliasSpec("EXM", "TICKER", "call_ticker", PRIORITIES["call_ticker"])
        self.assertEqual(alias.entity_type, "TICKER")
        self.assertGreater(alias.priority, PRIORITIES["global_ticker"])


if __name__ == "__main__":
    unittest.main()
