from __future__ import annotations

import json
import hashlib
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_identity_body_placebo import (  # noqa: E402
    CORRECT_ARM,
    DONOR_ASSIGNMENT_PREFIX,
    DONOR_ARM,
    _jitter,
    build_paired_corpus,
    canonical_calls,
    match_donors,
)
from build_embeddings import (  # noqa: E402
    IDENTITY_CONTROL_MARKER_COLUMN,
    IDENTITY_CONTROL_MARKER_VALUE,
    chunks,
    prepare_calls,
    prepare_preprocessed_identity_calls,
)
from compare_identity_control_ics import (  # noqa: E402
    canonical_predictions,
    compare_monthly_ics,
)
from build_identity_controls import (  # noqa: E402
    frozen_public_base,
    frozen_shell_metadata,
    ordered_unique_speakers,
    prepare_frozen_sections,
)
from mask_identity_corpus import (  # noqa: E402
    Alias,
    mask_identity_frame,
    mask_text,
    normalize_alias_registry,
)


class FakeEntity:
    def __init__(self, start: int, end: int, label: str) -> None:
        self.start_char = start
        self.end_char = end
        self.label_ = label


class FakeDocument:
    def __init__(self, entities: list[FakeEntity]) -> None:
        self.ents = entities


class IdentityControlTests(unittest.TestCase):
    def test_public_entry_points_have_cli_help(self) -> None:
        for name in ("build_identity_controls.py", "compare_identity_controls.py"):
            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / name), "--help"],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("usage:", completed.stdout)

    def alias_registry(self) -> pd.DataFrame:
        rows = []
        for call_id, company, ticker, speaker in (
            ("c1", "Acme Holdings", "ACM", "Alice Stone"),
            ("c2", "Beta Industries", "B", "Bob Reed"),
            ("c3", "Cedar Systems", "CDR", "Carol Lane"),
            ("c4", "Delta Energy", "DLE", "David Cole"),
        ):
            rows.extend(
                [
                    {"issuer_call_id": call_id, "alias_type": "company", "alias": company},
                    {"issuer_call_id": call_id, "alias_type": "ticker", "alias": ticker},
                    {"issuer_call_id": call_id, "alias_type": "speaker", "alias": speaker},
                ]
            )
        return normalize_alias_registry(pd.DataFrame(rows))

    def frozen_section_row(self, *, qa_doc: str | None = None) -> dict[str, object]:
        body = qa_doc or "First  line remains.\nSecond\tline remains."
        return {
            "issuer_call_id": "stable-call-1",
            "asset_symbol": "OLD",
            "query_symbol": "OLD-US",
            "company": "Example Systems, Inc.",
            "canonical_company": "Example Systems",
            "canonical_asset_symbol": "EXM",
            "canonical_query_symbol": "EXM-US",
            "canonical_issuer_id": "issuer-stable-1",
            "canonical_sector": "Information Technology",
            "original_asset_symbol": "OLD",
            "original_query_symbol": "OLD-US",
            "verified_call_date": "2024-02-12",
            "verified_month": "2024-02-01",
            "prepared_doc": "Alex Rivera: Welcome to the call.",
            "qa_doc": body,
            "opening_claimed_issuer": "Example Systems",
            "qa_capped_word_count": len(body.split()),
            "qa_text_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "cleanroom_usable": True,
        }

    def test_frozen_section_path_preserves_capped_bytes_and_call_ids(self) -> None:
        source = pd.DataFrame([self.frozen_section_row()])
        prepared = prepare_frozen_sections(source, expected_calls=1)
        self.assertEqual(prepared.loc[0, "qa_doc"], source.loc[0, "qa_doc"])
        self.assertIn("\n", prepared.loc[0, "qa_doc"])
        self.assertIn("\t", prepared.loc[0, "qa_doc"])
        self.assertEqual(prepared.loc[0, "issuer_call_id"], "stable-call-1")
        self.assertEqual(prepared.loc[0, "issuer_id"], "issuer-stable-1")
        public = frozen_public_base(prepared)
        self.assertEqual(public.loc[0, "issuer_call_id"], "stable-call-1")
        self.assertEqual(public.loc[0, "issuer_id"], public.loc[0, "security_id"])
        self.assertEqual(public.loc[0, "qa_text"], source.loc[0, "qa_doc"])

    def test_frozen_section_path_rejects_changed_source_bytes(self) -> None:
        row = self.frozen_section_row()
        row["qa_text_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "qa_text_sha256"):
            prepare_frozen_sections(pd.DataFrame([row]), expected_calls=1)

    def test_frozen_speaker_roster_preserves_first_occurrence_order(self) -> None:
        prepared = "Alex Rivera: Welcome.\nJordan Morgan: Thank you."
        pairs = pd.DataFrame(
            [
                {
                    "issuer_call_id": "stable-call-1",
                    "pair_index": 2,
                    "question_speaker": "Taylor Brooks",
                    "answer_speakers": "Alex Rivera|Casey Monroe",
                },
                {
                    "issuer_call_id": "stable-call-1",
                    "pair_index": 1,
                    "question_speaker": "Jordan Morgan",
                    "answer_speakers": "Morgan Patel",
                },
            ]
        )
        names = ordered_unique_speakers(prepared, pairs)
        self.assertEqual(
            names,
            (
                "Alex Rivera",
                "Jordan Morgan",
                "Morgan Patel",
                "Taylor Brooks",
                "Casey Monroe",
            ),
        )
        sections = prepare_frozen_sections(
            pd.DataFrame([self.frozen_section_row()]), expected_calls=1
        )
        shell = frozen_shell_metadata(sections, pairs)
        self.assertEqual(shell.loc[0, "speakers"], list(names))

    def test_literal_and_optional_entity_masking(self) -> None:
        text = "Acme Holdings launched Orbit after Alice Stone spoke."
        start = text.index("Orbit")
        document = FakeDocument([FakeEntity(start, start + len("Orbit"), "PRODUCT")])
        masked, counts = mask_text(
            text,
            [Alias("Acme Holdings", "company"), Alias("Alice Stone", "speaker")],
            document=document,
            ner_labels={"PRODUCT"},
        )
        self.assertEqual(
            masked,
            "[ISSUER] launched [PRODUCT] after [PERSON] spoke.",
        )
        self.assertEqual(counts["company"], 1)
        self.assertEqual(counts["speaker"], 1)
        self.assertEqual(counts["PRODUCT"], 1)

    def test_same_masking_rule_covers_training_and_evaluation(self) -> None:
        calls = pd.DataFrame(
            [
                {
                    "issuer_call_id": "c1",
                    "split": "training",
                    "qa_text": "Acme Holdings (ticker ACM) was discussed by Alice Stone.",
                },
                {
                    "issuer_call_id": "c2",
                    "split": "evaluation",
                    "qa_text": "Beta Industries compared itself with Acme Holdings. NASDAQ: B. Bob Reed spoke.",
                },
            ]
        )
        registry = self.alias_registry()
        registry = registry[registry["call_id"].isin(["c1", "c2"])].copy()
        masked, totals = mask_identity_frame(calls, registry)
        self.assertEqual(set(masked["split"]), {"training", "evaluation"})
        self.assertNotIn("Acme Holdings", " ".join(masked["qa_text"]))
        self.assertNotIn("Beta Industries", " ".join(masked["qa_text"]))
        self.assertIn("[TICKER]", masked.loc[0, "qa_text"])
        self.assertIn("NASDAQ: [TICKER]", masked.loc[1, "qa_text"])
        self.assertGreaterEqual(totals["company"], 3)
        self.assertEqual(totals["speaker"], 2)

    def paired_source(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"issuer_call_id": "c1", "issuer_id": "i1", "month": "2024-01-01", "sector": "A", "qa_text": "one " * 90},
                {"issuer_call_id": "c2", "issuer_id": "i2", "month": "2024-01-01", "sector": "A", "qa_text": "two " * 120},
                {"issuer_call_id": "c3", "issuer_id": "i3", "month": "2024-01-01", "sector": "B", "qa_text": "three " * 95},
                {"issuer_call_id": "c4", "issuer_id": "i4", "month": "2024-01-01", "sector": "B", "qa_text": "four " * 125},
            ]
        )

    def test_donor_mapping_is_deterministic_return_blind_and_cross_sector(self) -> None:
        source = self.paired_source()
        source["future_return"] = [10.0, -5.0, 20.0, -15.0]
        calls = canonical_calls(source)
        first = match_donors(calls)
        changed = source.copy()
        changed["future_return"] = [-900.0, 800.0, -700.0, 600.0]
        second = match_donors(canonical_calls(changed))
        pd.testing.assert_frame_equal(first, second)
        self.assertNotIn("future_return", calls.columns)
        self.assertTrue(first["target_sector"].ne(first["donor_sector"]).all())
        self.assertTrue(first["target_issuer_id"].ne(first["donor_issuer_id"]).all())
        self.assertFalse(first["donor_call_id"].duplicated().any())
        payload = f"{DONOR_ASSIGNMENT_PREFIX}|c1|c3".encode("utf-8")
        expected = int.from_bytes(__import__("hashlib").sha256(payload).digest()[:8], "big") / float(2**64) * 1.0e-9
        self.assertEqual(_jitter("c1", "c3"), expected)

    def test_paired_corpus_holds_identity_shell_fixed(self) -> None:
        calls = canonical_calls(self.paired_source())
        mapping = match_donors(calls)
        shell_metadata = pd.DataFrame(
            [
                {
                    "call_id": f"c{index}",
                    "canonical_company": f"Canonical {index}",
                    "canonical_ticker": f"T{index}",
                    "speakers": (["Zed Person", "Alice Person"] if index == 1 else []),
                }
                for index in range(1, 5)
            ]
        )
        corpus = build_paired_corpus(
            calls,
            mapping,
            shell_metadata=shell_metadata,
        )
        self.assertEqual(len(corpus), 2 * len(calls))
        self.assertEqual(set(corpus["arm"]), {CORRECT_ARM, DONOR_ARM})
        self.assertTrue(
            corpus.groupby("target_call_id")["identity_shell_sha256"]
            .nunique()
            .eq(1)
            .all()
        )
        donor = corpus[corpus["arm"].eq(DONOR_ARM)]
        self.assertTrue(donor["target_call_id"].ne(donor["body_source_call_id"]).all())
        first_document = corpus[corpus["target_call_id"].eq("c1")]["document_text"].iloc[0]
        self.assertTrue(
            first_document.startswith(
                "[TARGET IDENTITY SHELL]\n"
                "Canonical company: Canonical 1\n"
                "Canonical ticker: T1\n"
                "Current speakers (ordered unique): Zed Person | Alice Person\n"
                "[/TARGET IDENTITY SHELL]\n\n[SCRUBBED Q&A BODY]\n"
            )
        )
        empty_document = corpus[corpus["target_call_id"].eq("c2")]["document_text"].iloc[0]
        self.assertIn("Current speakers (ordered unique): (no parsed speakers)", empty_document)

    def test_build_entry_point_writes_three_embedding_ready_tables(self) -> None:
        prepared = "Prepared operating discussion remains fixed for this synthetic call. " * 12
        rows = []
        for index, (company, ticker, speaker, sector) in enumerate(
            (
                ("Acme Holdings", "ACM", "Alice Stone", "A"),
                ("Beta Industries", "BET", "Bob Reed", "A"),
                ("Cedar Systems", "CDR", "Carol Lane", "B"),
                ("Delta Energy", "DLE", "David Cole", "B"),
            )
        ):
            rows.append(
                {
                    "issuer_call_id": f"actual-call-{index}",
                    "security_id": f"issuer-{index}",
                    "ticker": ticker,
                    "sector": sector,
                    "earnings_date": f"2024-01-{10 + index:02d}",
                    "membership_date": "2024-01-01",
                    "in_index": True,
                    "prepared_text": prepared,
                    "qa_text": (
                        f"{company} " + "operating detail " * 900 + "AFTER_CAP_SENTINEL"
                        if index == 0
                        else f"{company} {speaker} described operations. " * (20 + index)
                    ),
                    "canonical_company": f"Canonical {company}",
                    "canonical_query_symbol": f"{ticker}.Q",
                    "company_aliases": json.dumps([company, f"Legacy {company}"]),
                    "ticker_aliases": json.dumps([f"{ticker}.Q", ticker]),
                    "speaker_aliases": (
                        json.dumps([speaker, "Aaron Second"])
                        if index == 0
                        else (json.dumps([]) if index == 3 else json.dumps([speaker]))
                    ),
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sections.csv"
            output = root / "controls"
            pd.DataFrame(rows).to_csv(source, index=False)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_identity_controls.py"),
                    "--sections",
                    str(source),
                    "--output-dir",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            paths = {
                name: output / f"{name}.parquet"
                for name in ("full_identity_mask", "correct_body", "donor_body")
            }
            self.assertTrue(all(path.is_file() for path in paths.values()))
            masked = pd.read_parquet(paths["full_identity_mask"])
            self.assertNotIn("Acme Holdings", " ".join(masked["qa_text"]))
            self.assertIn("[ISSUER]", " ".join(masked["qa_text"]))
            self.assertNotIn("AFTER_CAP_SENTINEL", masked.loc[masked["security_id"].eq("issuer-0"), "qa_text"].iloc[0])
            self.assertEqual(
                set(masked[IDENTITY_CONTROL_MARKER_COLUMN]),
                {IDENTITY_CONTROL_MARKER_VALUE},
            )
            correct = pd.read_parquet(paths["correct_body"])
            donor = pd.read_parquet(paths["donor_body"])
            self.assertEqual(
                set(correct["issuer_call_id"]),
                {f"actual-call-{index}" for index in range(4)},
            )
            self.assertEqual(set(correct["issuer_call_id"]), set(donor["issuer_call_id"]))
            correct_shells = correct.set_index("issuer_call_id")["qa_text"].str.split("[SCRUBBED Q&A BODY]", regex=False).str[0]
            donor_shells = donor.set_index("issuer_call_id")["qa_text"].str.split("[SCRUBBED Q&A BODY]", regex=False).str[0]
            pd.testing.assert_series_equal(correct_shells.sort_index(), donor_shells.sort_index())
            first_composite = correct.loc[
                correct["issuer_call_id"].eq("actual-call-0"), "qa_text"
            ].iloc[0]
            self.assertGreater(len(first_composite), 9_000)
            self.assertIn(
                "Current speakers (ordered unique): Alice Stone | Aaron Second",
                first_composite,
            )
            validated = prepare_preprocessed_identity_calls(correct)
            retained = validated.loc[
                validated["issuer_call_id"].eq("actual-call-0"), "qa_text"
            ].iloc[0]
            self.assertEqual(retained, first_composite)
            self.assertGreater(len(" ".join(chunks(retained, 160))), 9_000)
            ordinarily_reprepared = prepare_calls(correct)
            self.assertLessEqual(ordinarily_reprepared["qa_text"].str.len().max(), 9_000)
            self.assertTrue((output / "donor_mapping.csv").is_file())

    def test_preprocessed_mode_requires_protocol_marker(self) -> None:
        row = {
            "security_id": "issuer",
            "ticker": "TST",
            "sector": "A",
            "earnings_date": "2024-01-10",
            "membership_date": "2024-01-01",
            "in_index": True,
            "prepared_text": "prepared text",
            "qa_text": "Q&A text",
            "issuer_call_id": "actual-call",
            "issuer_id": "issuer",
            "month": "2024-01-01",
        }
        with self.assertRaisesRegex(ValueError, "columns are missing"):
            prepare_preprocessed_identity_calls(pd.DataFrame([row]))
        row[IDENTITY_CONTROL_MARKER_COLUMN] = "wrong"
        with self.assertRaisesRegex(ValueError, "marker"):
            prepare_preprocessed_identity_calls(pd.DataFrame([row]))


    def synthetic_predictions(self) -> pd.DataFrame:
        rng = np.random.default_rng(41)
        rows = []
        for month_index, month in enumerate(pd.date_range("2024-01-01", periods=12, freq="MS")):
            returns = np.arange(6, dtype=float)
            correct = returns.copy()
            if month_index % 3 == 0:
                correct[[0, 1]] = correct[[1, 0]]
            donor = rng.permutation(returns)
            for arm, predictions in ((CORRECT_ARM, correct), (DONOR_ARM, donor)):
                for sector, (prediction, target) in enumerate(zip(predictions, returns, strict=True)):
                    rows.append(
                        {
                            "arm": arm,
                            "month": month,
                            "sector": f"Sector {sector}",
                            "pred": prediction,
                            "excess_return": target,
                        }
                    )
        return pd.DataFrame(rows)

    def test_monthly_ic_comparison_reports_hac_paired_gap(self) -> None:
        predictions = canonical_predictions(self.synthetic_predictions())
        monthly, summary = compare_monthly_ics(predictions, hac_lag=1)
        self.assertEqual(len(monthly), 12)
        self.assertTrue(monthly["paired_gap"].notna().all())
        gap = summary.iloc[-1]
        self.assertGreater(float(gap["mean"]), 0.5)
        self.assertTrue(np.isfinite(float(gap["hac_se"])))
        self.assertTrue(np.isfinite(float(gap["hac_two_sided_p"])))

    def test_monthly_ic_comparison_rejects_unmatched_support(self) -> None:
        predictions = canonical_predictions(self.synthetic_predictions())
        shortened = predictions.drop(predictions.index[-1])
        with self.assertRaisesRegex(ValueError, "common sector-month support"):
            compare_monthly_ics(shortened)

    def test_compare_entry_point_collects_both_signed_forms(self) -> None:
        base = self.synthetic_predictions()
        candidates = {
            "firm_conditioned_qa_raw": 0.0,
            "firm_conditioned_qa_direction": 0.05,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_dirs = {}
            for arm_name, arm_label in (
                ("masked", CORRECT_ARM),
                ("correct", CORRECT_ARM),
                ("donor", DONOR_ARM),
            ):
                local = base[base["arm"].eq(arm_label)].copy()
                frames = []
                for candidate, offset in candidates.items():
                    piece = local[["sector", "month", "pred", "excess_return"]].copy()
                    piece["pred"] = piece["pred"] + offset
                    piece.insert(0, "candidate", candidate)
                    frames.append(piece)
                result_dir = root / arm_name
                (result_dir / "row_level").mkdir(parents=True)
                pd.concat(frames, ignore_index=True).to_parquet(
                    result_dir / "row_level" / "multiresolution_predictions.parquet",
                    index=False,
                )
                result_dirs[arm_name] = result_dir
            pd.DataFrame(
                [
                    {"candidate": candidate, "randomizations": 20_000, "upper_tail_p": 0.01 + index * 0.01}
                    for index, candidate in enumerate(candidates)
                ]
            ).to_csv(
                result_dirs["masked"] / "representation_circular_shift_tests.csv",
                index=False,
            )
            output = root / "identity_control_tests.csv"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "compare_identity_controls.py"),
                    "--masked-results",
                    str(result_dirs["masked"]),
                    "--correct-results",
                    str(result_dirs["correct"]),
                    "--donor-results",
                    str(result_dirs["donor"]),
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = pd.read_csv(output)
            self.assertEqual(len(summary), 8)
            self.assertEqual(set(summary["representation"]), {"raw_residual", "unit_direction"})
            paired = summary[summary["test"].eq("paired_correct_minus_donor")]
            self.assertTrue(paired["estimate"].gt(0.5).all())
            monthly = pd.read_csv(root / "identity_control_monthly_ic.csv")
            self.assertEqual(len(monthly), 72)


if __name__ == "__main__":
    unittest.main()
