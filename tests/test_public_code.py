from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common import (  # noqa: E402
    build_panel,
    fit_fixed_head,
    monthly_rank_ic,
)
from portfolio import (  # noqa: E402
    build_sleeves,
    daily_risk_free_returns,
    daily_returns,
    momentum_scores,
    strategy_summary,
)
from callrank_features import (  # noqa: E402
    RAW_DISPLACEMENT,
    UNIT_DIRECTION,
    firm_conditioned_representations,
    prior_residual,
)


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PublicCodeTests(unittest.TestCase):
    def test_frozen_configuration(self) -> None:
        config = json.loads((ROOT / "config" / "frozen_config.json").read_text())
        self.assertIs(config["raw_text_included"], False)
        self.assertIs(config["firm_level_rows_included"], False)
        self.assertEqual(config["feature_dimensions"], 768)
        self.assertEqual(config["encoder"]["name"], "llmware/industry-bert-sec-v0.1")
        self.assertEqual(
            config["data_integrity"]["evaluation_exact_official_date"],
            2848,
        )
        self.assertEqual(
            config["data_integrity"]["evaluation_exact_official_date"]
            + config["data_integrity"]["evaluation_official_month"],
            config["data_integrity"]["evaluation_cleanroom_calls"],
        )
        self.assertTrue(
            config["research_boundary"].startswith(
                "Fixed-head temporal evaluation"
            )
        )
        self.assertIn("reported fixed-head", config["snapshot_status"])
        self.assertIn(
            "identity-controlled mechanism tests",
            config["snapshot_status"],
        )
        self.assertNotIn(
            "before firm-conditioned alternatives were evaluated",
            config["ranking_head"]["provenance"],
        )
        self.assertIs(config["universe"]["membership_is_point_in_time"], True)
        self.assertIs(
            config["universe"]["firm_specific_historical_reclassifications_complete"],
            False,
        )
        self.assertIn("Top 1", config["portfolio"]["rule"])
        self.assertEqual(config["ranking_head"]["latest_training_outcome"], "2020-12-01")
        self.assertIs(
            config["ranking_head"]["refit_after_training_boundary"],
            False,
        )
        self.assertEqual(
            config["feature_definition"]["minimum_prior_firm_months"], 2
        )
        self.assertEqual(
            config["evaluation"]["circular_shift_randomizations"], 20_000
        )
        self.assertEqual(
            config["feature_definition"]["full_rank_signed_form"],
            "raw signed residual",
        )
        self.assertEqual(RAW_DISPLACEMENT, "firm_conditioned_qa_raw")
        self.assertEqual(
            config["feature_definition"]["bounded_allocation_signed_form"],
            "L2-normalized signed residual",
        )
        self.assertIs(
            config["feature_definition"]["empirical_dominance_claim"],
            False,
        )

    def test_release_validator_ignores_local_virtualenv(self) -> None:
        module = load_script("validate_release")
        self.assertFalse(
            module.is_public_path(ROOT / ".venv" / "lib" / "site.py")
        )
        self.assertFalse(
            module.is_public_path(ROOT / "scripts" / "__pycache__" / "code.pyc")
        )
        self.assertTrue(module.is_public_path(ROOT / "scripts" / "run_core.py"))
        if (ROOT / ".git").exists():
            self.assertFalse(
                any(".venv" in path.parts for path in module.tracked_files())
            )

    def test_firm_history_uses_only_prior_observations(self) -> None:
        values = np.array(
            [
                [1.0, 0.0],
                [1.0, 1.0],
                [2.0, 1.0],
                [100.0, 100.0],
            ]
        )
        identifiers = pd.Series(["A", "A", "A", "A"])
        dates = pd.Series(pd.date_range("2020-01-01", periods=4, freq="MS"))
        residual = prior_residual(values, identifiers, dates, 2)
        self.assertTrue(np.isnan(residual[:2]).all())
        self.assertTrue(np.allclose(residual[2], [1.0, 0.5]))
        changed = values.copy()
        changed[3] = [-500.0, 900.0]
        changed_residual = prior_residual(
            changed, identifiers, dates, 2
        )
        self.assertTrue(np.allclose(residual[2], changed_residual[2]))

    def test_firm_history_is_invariant_to_input_order(self) -> None:
        values = np.array(
            [[1.0, 0.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
        )
        identifiers = pd.Series(["A"] * 4)
        dates = pd.Series(pd.date_range("2020-01-01", periods=4, freq="MS"))
        expected = prior_residual(values, identifiers, dates, 2)
        order = np.array([2, 0, 3, 1])
        shuffled = prior_residual(
            values[order],
            identifiers.iloc[order].reset_index(drop=True),
            dates.iloc[order].reset_index(drop=True),
            2,
        )
        restored = np.empty_like(shuffled)
        restored[order] = shuffled
        self.assertTrue(
            np.allclose(expected, restored, equal_nan=True)
        )

    def test_sector_signal_equal_weights_eligible_firms(self) -> None:
        months = pd.date_range("2020-01-01", periods=3, freq="MS")
        keys = pd.DataFrame(
            [
                {
                    "issuer_id": issuer,
                    "sector": "Industrials",
                    "month": month,
                }
                for issuer in ("A", "B")
                for month in months
            ]
        )
        prepared = np.tile(np.array([[1.0, 0.0]]), (len(keys), 1))
        qa = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        qa = qa / np.linalg.norm(qa, axis=1, keepdims=True)
        sector_keys, candidates, audit = firm_conditioned_representations(
            keys, prepared, qa, minimum_prior=2
        )
        final = sector_keys["month"].eq(months[-1])
        self.assertEqual(int(audit.loc[final, "eligible_firms"].iloc[0]), 2)
        direction = candidates[UNIT_DIRECTION][final.to_numpy()][0]
        firm_residuals = np.vstack(
            [qa[2] - qa[0], qa[5] - qa[3]]
        )
        firm_directions = firm_residuals / np.linalg.norm(
            firm_residuals, axis=1, keepdims=True
        )
        self.assertTrue(
            np.allclose(direction, firm_directions.mean(axis=0))
        )

    def test_monthly_rank_ic_on_synthetic_panel(self) -> None:
        rows = []
        for month in pd.to_datetime(["2024-01-01", "2024-02-01"]):
            for index in range(6):
                rows.append(
                    {"month": month, "pred": float(index), "excess_return": float(index * 2)}
                )
        values = monthly_rank_ic(pd.DataFrame(rows))
        self.assertEqual(len(values), 2)
        self.assertTrue(np.allclose(values.to_numpy(), 1.0))

    def test_fixed_head_never_uses_late_outcomes(self) -> None:
        rng = np.random.default_rng(7)
        rows = []
        for month in pd.date_range("2020-01-01", periods=8, freq="MS"):
            for sector in range(8):
                rows.append(
                    {
                        "sector": f"Sector {sector}",
                        "month": month,
                        "entry_date": month + pd.offsets.BDay(1),
                        "exit_date": month + pd.offsets.BDay(50),
                        "horizon": 42,
                        "excess_return": float(sector),
                    }
                )
        for month in pd.date_range("2024-01-01", periods=2, freq="MS"):
            for sector in range(8):
                rows.append(
                    {
                        "sector": f"Sector {sector}",
                        "month": month,
                        "entry_date": month + pd.offsets.BDay(1),
                        "exit_date": month + pd.offsets.BDay(50),
                        "horizon": 42,
                        "excess_return": float(sector),
                    }
                )
        panel = pd.DataFrame(rows)
        values = rng.normal(size=(len(panel), 768))
        scored, metadata = fit_fixed_head(panel, values, evaluation_start="2023-11-01")
        self.assertEqual(len(scored), 16)
        self.assertLessEqual(pd.Timestamp(metadata["latest_training_outcome"]), pd.Timestamp("2020-12-01"))
        self.assertEqual(metadata["effective_features"], 32)

    def synthetic_prices(self) -> pd.DataFrame:
        dates = pd.bdate_range("2024-01-01", periods=100)
        frame = pd.DataFrame({"Date": dates, "SPY": np.linspace(100, 110, len(dates))})
        module = load_script("build_targets")
        for index, ticker in enumerate(module.SECTOR_ETF.values()):
            frame[ticker] = np.linspace(100, 112 + index, len(dates))
        return frame

    def test_target_builder_records_entry_and_exit(self) -> None:
        module = load_script("build_targets")
        prices = self.synthetic_prices()
        keys = pd.DataFrame(
            {"sector": list(module.SECTOR_ETF), "month": pd.Timestamp("2024-01-01")}
        )
        output = module.build_targets(prices, keys)
        self.assertEqual(len(output), 11)
        self.assertTrue(output["entry_date"].lt(output["exit_date"]).all())
        self.assertTrue(output["excess_return"].gt(0).all())

    def test_target_alignment_preserves_every_embedding_row(self) -> None:
        keys = pd.DataFrame(
            {
                "sector": ["A", "B", "C"],
                "month": pd.to_datetime(["2024-01-01"] * 3),
            }
        )
        targets = pd.DataFrame(
            {
                "sector": ["A", "C"],
                "month": pd.to_datetime(["2024-01-01"] * 2),
                "entry_date": pd.to_datetime(["2024-02-01"] * 2),
                "exit_date": pd.to_datetime(["2024-04-01"] * 2),
                "horizon": [42, 42],
                "excess_return": [0.1, 0.2],
            }
        )
        panel = build_panel(keys, targets)
        self.assertEqual(panel["sector"].tolist(), ["A", "B", "C"])
        self.assertEqual(len(panel), len(keys))
        self.assertTrue(np.isnan(panel.loc[1, "excess_return"]))
        self.assertEqual(panel["target_available"].tolist(), [True, False, True])

    def test_character_cap_precedes_whitespace_cleanup(self) -> None:
        module = load_script("build_embeddings")
        text = "alpha" + (" " * 20) + "omega"
        self.assertEqual(module.cap_then_clean(text, 10), "alpha")

    def test_spy_alpha_uses_compound_annualization(self) -> None:
        spy = np.linspace(-0.01, 0.01, 40)
        daily = pd.DataFrame(
            {
                "date": pd.bdate_range("2024-01-01", periods=len(spy)),
                "net_return": 0.001 + 0.5 * spy,
                "spy_return": spy,
                "turnover": 0.0,
                "cost": 0.0,
            }
        )
        summary = strategy_summary(daily)
        self.assertAlmostEqual(
            float(summary["spy_alpha"]),
            (1.001**252) - 1.0,
            places=10,
        )

    def test_top_one_capitalization_is_prior_close(self) -> None:
        module = load_script("build_targets")
        prices = self.synthetic_prices()
        entry_date = pd.Timestamp("2024-02-01")
        sectors = list(module.SECTOR_ETF)[:6]
        scored = pd.DataFrame(
            {
                "sector": sectors,
                "month": pd.Timestamp("2024-01-01"),
                "entry_date": entry_date,
                "exit_date": pd.Timestamp("2024-04-01"),
                "excess_return": np.arange(6, dtype=float),
                "pred": np.arange(6, dtype=float),
            }
        )
        cap_date = prices.loc[prices["Date"].lt(entry_date), "Date"].max()
        caps = pd.DataFrame(
            {
                "Date": cap_date,
                "sector": sectors,
                "market_cap": np.arange(1, 7, dtype=float) * 100.0,
            }
        )
        sleeves, weights = build_sleeves(
            scored,
            prices,
            caps,
            count=1,
        )
        self.assertEqual(len(sleeves), 1)
        self.assertEqual(set(weights["sector"]), {sectors[-1]})
        self.assertTrue(weights["cap_date"].lt(weights["entry_date"]).all())
        self.assertAlmostEqual(float(weights["weight"].sum()), 1.0)
        path = daily_returns(sleeves, prices)
        self.assertGreater(len(path), 42)
        self.assertEqual(float(path.iloc[0]["spy_return"]), 0.0)
        self.assertAlmostEqual(
            float(path.iloc[0]["net_return"]),
            -float(path.iloc[0]["cost"]),
        )

    def test_three_overlapping_sleeves_are_equal_capital_weighted(self) -> None:
        module = load_script("build_targets")
        tickers = tuple(module.SECTOR_ETF.values())
        dates = pd.bdate_range("2024-01-01", periods=80)
        rates = {
            tickers[0]: 0.01,
            tickers[1]: 0.02,
            tickers[2]: 0.03,
        }
        prices = pd.DataFrame({"Date": dates, "SPY": 100.0})
        for ticker in tickers:
            rate = rates.get(ticker, 0.0)
            prices[ticker] = 100.0 * (1.0 + rate) ** np.arange(len(dates))

        def sleeve(identifier: str, entry: int, ticker: str) -> dict:
            weights = {asset: 0.0 for asset in tickers}
            weights[ticker] = 1.0
            return {
                "id": identifier,
                "entry_index": entry,
                "exit_index": entry + 42,
                "weights": weights,
            }

        path = daily_returns(
            [
                sleeve("a", 10, tickers[0]),
                sleeve("b", 20, tickers[1]),
                sleeve("c", 30, tickers[2]),
            ],
            prices,
        )
        self.assertEqual(int(path["active_sleeves"].max()), 3)

        # Targets are installed after the entry-day return is calculated, so
        # the next trading day is the first return earned at one-third each.
        next_day = path.loc[path["date"].eq(dates[31])].iloc[0]
        self.assertEqual(int(next_day["active_sleeves"]), 3)
        self.assertAlmostEqual(
            float(next_day["gross_return"]),
            (0.01 + 0.02 + 0.03) / 3.0,
            places=12,
        )

    def test_cutoff_ties_can_be_split_without_return_information(self) -> None:
        module = load_script("build_targets")
        prices = self.synthetic_prices()
        entry_date = pd.Timestamp("2024-02-01")
        sectors = list(module.SECTOR_ETF)[:6]
        scored = pd.DataFrame(
            {
                "sector": sectors,
                "month": pd.Timestamp("2024-01-01"),
                "entry_date": entry_date,
                "exit_date": pd.Timestamp("2024-04-01"),
                "excess_return": np.arange(6, dtype=float),
                "pred": [0.0, 1.0, 2.0, 3.0, 5.0, 5.0],
            }
        )
        cap_date = prices.loc[prices["Date"].lt(entry_date), "Date"].max()
        caps = pd.DataFrame(
            {
                "Date": cap_date,
                "sector": sectors,
                "market_cap": np.arange(1, 7, dtype=float) * 100.0,
            }
        )
        _, canonical = build_sleeves(
            scored,
            prices,
            caps,
            count=1,
            tie_policy="canonical_sector",
        )
        _, all_ties = build_sleeves(
            scored,
            prices,
            caps,
            count=1,
            tie_policy="all_cutoff",
        )
        self.assertEqual(len(canonical), 1)
        self.assertEqual(len(all_ties), 2)
        self.assertAlmostEqual(float(all_ties["weight"].sum()), 1.0)
        self.assertEqual(set(all_ties["sector"]), set(sectors[-2:]))

    def test_sharpe_subtracts_strictly_prior_risk_free_rate(self) -> None:
        dates = pd.Series(pd.to_datetime(["2024-01-03", "2024-01-04"]))
        risk_free = pd.DataFrame(
            {
                "observation_date": pd.to_datetime(
                    ["2024-01-02", "2024-01-03", "2024-01-04"]
                ),
                "DGS3MO": [5.0, 6.0, 99.0],
            }
        )
        observed = daily_risk_free_returns(dates, risk_free)
        expected = np.array(
            [
                (1.05 ** (1.0 / 252.0)) - 1.0,
                (1.06 ** (1.0 / 252.0)) - 1.0,
            ]
        )
        self.assertTrue(np.allclose(observed, expected))

    def test_momentum_formation_ends_before_entry(self) -> None:
        module = load_script("build_targets")
        prices = self.synthetic_prices()
        sectors = list(module.SECTOR_ETF)[:6]
        scored = pd.DataFrame(
            {
                "sector": sectors,
                "month": pd.Timestamp("2024-03-01"),
                "entry_date": pd.Timestamp("2024-04-01"),
                "exit_date": pd.Timestamp("2024-05-29"),
                "excess_return": 0.0,
                "pred": 0.0,
            }
        )
        momentum = momentum_scores(scored, prices)
        self.assertTrue(momentum["momentum_formation_end"].lt(momentum["entry_date"]).all())

    def test_broad_gics_changes_follow_effective_dates(self) -> None:
        module = load_script("build_pit_sector_labels")
        self.assertEqual(
            module.reconstruct_sector(
                "GOOGL", "Communication Services", "Interactive Media", pd.Timestamp("2018-01-01")
            ),
            "Information Technology",
        )
        self.assertEqual(
            module.reconstruct_sector(
                "V", "Financials", "Transaction Processing", pd.Timestamp("2022-01-01")
            ),
            "Information Technology",
        )
        self.assertEqual(
            module.reconstruct_sector(
                "AAA", "Real Estate", "REIT", pd.Timestamp("2015-01-01")
            ),
            "Financials",
        )
        self.assertEqual(
            module.reconstruct_sector(
                "GOOGL", "Communication Services", "Interactive Media", pd.Timestamp("2019-01-01")
            ),
            "Communication Services",
        )

    def test_dated_sector_history_uses_latest_prior_label(self) -> None:
        module = load_script("build_pit_sector_labels")
        calls = pd.DataFrame(
            {
                "security_id": [1, 1],
                "ticker": ["AAA", "AAA"],
                "earnings_date": pd.to_datetime(["2020-02-01", "2021-02-01"]),
            }
        )
        history = pd.DataFrame(
            {
                "security_id": [1, 1],
                "effective_date": pd.to_datetime(["2019-01-01", "2021-01-01"]),
                "sector": ["Industrials", "Information Technology"],
            }
        )
        output = module.attach_historical_sectors(calls, history)
        self.assertEqual(
            output.sort_values("earnings_date")["sector"].tolist(),
            ["Industrials", "Information Technology"],
        )
        self.assertTrue(output["sector_mapping_date"].le(output["earnings_date"]).all())


if __name__ == "__main__":
    unittest.main()
