from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_sections.py"
SPEC = importlib.util.spec_from_file_location("prepare_sections", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SectionParserTests(unittest.TestCase):
    def test_extracts_management_side_text(self) -> None:
        prepared = "Prepared discussion. " * 120
        transcript = (
            prepared
            + "\nOperator: We will now begin the question-and-answer session.\n"
            + "Analyst - Research: What changed in demand?\n"
            + "Executive: Demand improved while costs remained stable.\n" * 12
        )
        split = MODULE.split_call(transcript)
        self.assertIn("Prepared discussion", split["prepared_text"])
        self.assertIn("Demand improved", split["qa_text"])
        self.assertNotIn("What changed", split["qa_text"])

    def test_falls_back_to_full_qa_when_roles_are_missing(self) -> None:
        prepared = "Prepared discussion. " * 120
        qa = "\nOperator: We will now begin the question-and-answer session.\n" + (
            "Unlabeled discussion continues with operating and market details.\n" * 12
        )
        split = MODULE.split_call(prepared + qa)
        self.assertIn("all_qa_fallback", split["qa_method"])
        self.assertIn("Unlabeled discussion", split["qa_text"])

    def test_rejects_membership_observed_after_call(self) -> None:
        frame = __import__("pandas").DataFrame(
            {
                "security_id": ["id-1"],
                "ticker": ["AAA"],
                "sector": ["Industrials"],
                "earnings_date": ["2024-01-15"],
                "membership_date": ["2024-01-16"],
                "in_index": [True],
                "content": ["Prepared discussion. " * 200],
            }
        )
        with self.assertRaisesRegex(ValueError, "follows"):
            MODULE.prepare_sections(frame)


if __name__ == "__main__":
    unittest.main()
