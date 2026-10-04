#!/usr/bin/env python3
"""Run the CallRank representation and portfolio calculations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from common import (
    build_panel,
    fit_fixed_head,
    hac_pvalue,
    load_config,
    load_targets,
    monthly_rank_ic,
    rank_summary,
    save_json,
)
from portfolio import (
    ETFS,
    build_sleeves,
    daily_returns,
    prepare_caps,
    prepare_prices,
    strategy_summary,
)
from callrank_features import (
    RAW_DISPLACEMENT,
    UNIT_DIRECTION,
    company_month_sections,
    firm_conditioned_representations,
    load_call_embeddings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--randomizations",
        type=int,
        help="Override the configured number of circular time shifts.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        help="Override the configured circular-shift seed for a declared control.",
    )
    parser.add_argument(
        "--risk-free",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "DGS3MO_2023-10-01_2026-06-05.csv",
        help="Public three-month Treasury yield history used for Sharpe ratios.",
    )
    return parser.parse_args()


def combine_spread(
    top: list[dict[str, Any]], bottom: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    bottom_by_month = {
        pd.Timestamp(sleeve["month"]): sleeve for sleeve in bottom
    }
    output: list[dict[str, Any]] = []
    for long_sleeve in top:
        month = pd.Timestamp(long_sleeve["month"])
        short_sleeve = bottom_by_month.get(month)
        if short_sleeve is None:
            raise RuntimeError(f"Missing bottom sleeve for {month:%Y-%m}")
        combined = dict(long_sleeve)
        combined["id"] = f"spread-{month.date()}"
        combined["weights"] = {
            ticker: float(long_sleeve["weights"][ticker])
            - float(short_sleeve["weights"][ticker])
            for ticker in ETFS
        }
        output.append(combined)
    return output


def prediction_piece(
    scored: pd.DataFrame, candidate: str, chunk_words: int
) -> pd.DataFrame:
    output = scored[
        [
            "sector",
            "month",
            "entry_date",
            "exit_date",
            "excess_return",
            "pred",
        ]
    ].copy()
    output.insert(0, "candidate", candidate)
    output.insert(0, "chunk_words", int(chunk_words))
    return output


def ensemble_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    work = predictions.copy()
    work["pred_rank"] = work.groupby(
        ["candidate", "chunk_words", "month"], observed=True
    )["pred"].rank(method="average", pct=True)
    output = (
        work.groupby(
            ["candidate", "sector", "month"], sort=True, observed=True
        )
        .agg(
            entry_date=("entry_date", "first"),
            exit_date=("exit_date", "first"),
            excess_return=("excess_return", "first"),
            pred=("pred_rank", "mean"),
            chunk_models=("chunk_words", "nunique"),
        )
        .reset_index()
    )
    expected = len(load_config()["chunking"]["word_chunk_sizes"])
    if not output["chunk_models"].eq(expected).all():
        raise RuntimeError("An ensemble row lacks one or more chunk models")
    return output


def paired_ic_tests(predictions: pd.DataFrame) -> pd.DataFrame:
    reference = predictions[predictions["candidate"].eq(UNIT_DIRECTION)]
    unit_ic = monthly_rank_ic(reference).rename("reference")
    rows: list[dict[str, Any]] = []
    alternatives = sorted(set(predictions["candidate"]) - {UNIT_DIRECTION})
    for candidate in alternatives:
        comparison = predictions[predictions["candidate"].eq(candidate)]
        comparison_ic = monthly_rank_ic(comparison).rename("alternative")
        paired = pd.concat([unit_ic, comparison_ic], axis=1).dropna()
        difference = paired["reference"] - paired["alternative"]
        test = stats.ttest_1samp(difference.to_numpy(), popmean=0.0)
        rows.append(
            {
                "comparison": candidate,
                "months": int(len(paired)),
                "unit_direction_ic": float(paired["reference"].mean()),
                "comparison_ic": float(paired["alternative"].mean()),
                "difference": float(difference.mean()),
                "paired_t_p": float(test.pvalue),
                "newey_west_p": hac_pvalue(difference, 1),
            }
        )
    return pd.DataFrame(rows).sort_values("comparison")


def circular_shift_null(
    scored: pd.DataFrame,
    *,
    randomizations: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = scored[
        ["sector", "month", "entry_date", "exit_date", "excess_return", "pred"]
    ].copy()
    source.sort_values(["sector", "month"], inplace=True)
    source.reset_index(drop=True, inplace=True)
    groups = [
        group.index.to_numpy(dtype=int)
        for _, group in source.groupby("sector", sort=True)
        if len(group) > 1
    ]
    observed = float(monthly_rank_ic(source).mean())
    rng = np.random.default_rng(seed)
    values = source["pred"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for iteration in range(randomizations):
        shifted_values = values.copy()
        for positions in groups:
            offset = int(rng.integers(1, len(positions)))
            shifted_values[positions] = np.roll(
                values[positions], offset
            )
        shifted = source.copy()
        shifted["pred"] = shifted_values
        rows.append(
            {
                "iteration": iteration,
                "rank_ic": float(monthly_rank_ic(shifted).mean()),
            }
        )
    null = pd.DataFrame(rows)
    upper_tail_p = float(
        (1 + null["rank_ic"].ge(observed).sum()) / (len(null) + 1)
    )
    return null, {
        "observed_rank_ic": observed,
        "randomizations": int(len(null)),
        "null_mean": float(null["rank_ic"].mean()),
        "null_std": float(null["rank_ic"].std(ddof=1)),
        "upper_tail_p": upper_tail_p,
    }


def economic_results(
    candidate: str,
    scored: pd.DataFrame,
    prices: pd.DataFrame,
    caps: pd.DataFrame,
    risk_free: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    rank = rank_summary(scored)
    top1, _ = build_sleeves(
        scored,
        prices,
        caps,
        side="top",
        count=1,
        weighting="pit_market_cap",
    )
    bottom1, _ = build_sleeves(
        scored,
        prices,
        caps,
        side="bottom",
        count=1,
        weighting="pit_market_cap",
    )
    top1_ties, _ = build_sleeves(
        scored,
        prices,
        caps,
        side="top",
        count=1,
        weighting="pit_market_cap",
        tie_policy="all_cutoff",
    )
    bottom1_ties, _ = build_sleeves(
        scored,
        prices,
        caps,
        side="bottom",
        count=1,
        weighting="pit_market_cap",
        tie_policy="all_cutoff",
    )
    strategies = {
        "top1": top1,
        "bottom1": bottom1,
        "top1_minus_bottom1": combine_spread(top1, bottom1),
        "top1_all_cutoff_ties": top1_ties,
        "bottom1_all_cutoff_ties": bottom1_ties,
        "top1_minus_bottom1_all_cutoff_ties": combine_spread(
            top1_ties,
            bottom1_ties,
        ),
    }
    rows: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    for strategy, sleeves in strategies.items():
        daily = daily_returns(sleeves, prices)
        rows.append(
            {
                "model": "PCA-32 + Ridge",
                "representation": candidate,
                "resolution": "multiresolution",
                "strategy": strategy,
                **rank,
                **strategy_summary(daily, risk_free=risk_free),
                "signal_months": int(len(sleeves)),
            }
        )
        if candidate == UNIT_DIRECTION:
            saved = daily.copy()
            saved.insert(0, "strategy", strategy)
            daily_frames.append(saved)
    return rows, daily_frames


def required_input_paths(data_dir: Path) -> dict[str, Path]:
    config = load_config()
    inputs = config["external_inputs"]
    paths = {
        "targets": data_dir / inputs["targets"],
        "prices": data_dir / inputs["adjusted_prices"],
        "caps": data_dir / inputs["sector_market_caps"],
    }
    pattern = str(inputs["call_embedding_pattern"])
    for words in config["chunking"]["word_chunk_sizes"]:
        for section in ("prepared", "qa"):
            paths[f"{section}_{words}"] = data_dir / pattern.format(
                section=section, chunk_words=words
            )
    missing = [
        str(path.name) for path in paths.values() if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing licensed or derived inputs: " + ", ".join(missing)
        )
    return paths


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    row_level = results_dir / "row_level"
    results_dir.mkdir(parents=True, exist_ok=True)
    row_level.mkdir(parents=True, exist_ok=True)

    config = load_config()
    paths = required_input_paths(data_dir)
    targets = load_targets(paths["targets"])
    prices = prepare_prices(pd.read_csv(paths["prices"]))
    caps = prepare_caps(pd.read_csv(paths["caps"]))
    risk_free = pd.read_csv(args.risk_free.expanduser().resolve())
    evaluation_start = str(config["evaluation"]["release_aligned_signal_start"])

    prediction_frames: list[pd.DataFrame] = []
    fit_rows: list[dict[str, Any]] = []
    audit_frames: list[pd.DataFrame] = []
    for words in config["chunking"]["word_chunk_sizes"]:
        call_keys, prepared, qa = load_call_embeddings(
            paths[f"prepared_{words}"], paths[f"qa_{words}"]
        )
        company_keys, prepared_month, qa_month, call_counts = (
            company_month_sections(call_keys, prepared, qa)
        )
        sector_keys, candidates, audit = firm_conditioned_representations(
            company_keys,
            prepared_month,
            qa_month,
            minimum_prior=int(
                config["feature_definition"][
                    "minimum_prior_firm_months"
                ]
            ),
        )
        audit.insert(0, "chunk_words", int(words))
        audit["company_month_calls"] = int(call_counts.sum())
        audit_frames.append(audit)
        panel = build_panel(sector_keys, targets)
        for candidate, values in candidates.items():
            scalar = candidate == "firm_conditioned_qa_distance"
            scored, metadata = fit_fixed_head(
                panel,
                values,
                evaluation_start=evaluation_start,
                pca_components=False if scalar else None,
            )
            prediction_frames.append(
                prediction_piece(scored, candidate, int(words))
            )
            fit_rows.append(
                {
                    "candidate": candidate,
                    "chunk_words": int(words),
                    **metadata,
                }
            )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    ensemble = ensemble_predictions(predictions)
    result_rows: list[dict[str, Any]] = []
    unit_daily: list[pd.DataFrame] = []
    for candidate, scored in ensemble.groupby("candidate", sort=True):
        rows, daily = economic_results(
            str(candidate), scored, prices, caps, risk_free
        )
        result_rows.extend(rows)
        unit_daily.extend(daily)

    randomizations = (
        int(args.randomizations)
        if args.randomizations is not None
        else int(config["evaluation"]["circular_shift_randomizations"])
    )
    seed = (
        int(args.random_seed)
        if args.random_seed is not None
        else int(config["evaluation"]["circular_shift_seed"])
    )
    circular_rows: list[dict[str, Any]] = []
    raw_displacement_null: pd.DataFrame | None = None
    for candidate, scored in ensemble.groupby("candidate", sort=True):
        null, summary = circular_shift_null(
            scored, randomizations=randomizations, seed=seed
        )
        circular_rows.append({"candidate": candidate, **summary})
        if candidate == RAW_DISPLACEMENT:
            raw_displacement_null = null

    pd.DataFrame(result_rows).to_csv(
        results_dir / "representation_results.csv", index=False
    )
    paired_ic_tests(ensemble).to_csv(
        results_dir / "paired_ic_tests.csv", index=False
    )
    pd.DataFrame(circular_rows).to_csv(
        results_dir / "representation_circular_shift_tests.csv",
        index=False,
    )
    pd.DataFrame(fit_rows).to_csv(
        results_dir / "fit_audit.csv", index=False
    )
    pd.concat(audit_frames, ignore_index=True).to_csv(
        results_dir / "sector_month_audit.csv", index=False
    )
    predictions.to_parquet(
        row_level / "resolution_predictions.parquet", index=False
    )
    ensemble.to_parquet(
        row_level / "multiresolution_predictions.parquet", index=False
    )
    pd.concat(unit_daily, ignore_index=True).to_parquet(
        row_level / "unit_direction_daily_returns.parquet", index=False
    )
    if raw_displacement_null is None:
        raise RuntimeError("Raw-displacement circular-shift null was not generated")
    raw_displacement_null.to_csv(
        results_dir / "temporal_circular_shift_null.csv", index=False
    )
    save_json(
        results_dir / "run_summary.json",
        {
            "status": "PASS",
            "full_rank_signed_form": RAW_DISPLACEMENT,
            "bounded_allocation_signed_form": UNIT_DIRECTION,
            "chunk_words": config["chunking"]["word_chunk_sizes"],
            "evaluation_start": evaluation_start,
            "randomizations": randomizations,
            "random_seed": seed,
            "research_boundary": config["research_boundary"],
        },
    )
    results = pd.DataFrame(result_rows)
    allocation_row = results[
        results["representation"].eq(UNIT_DIRECTION)
        & results["strategy"].eq("top1")
    ].iloc[0]
    rank_row = results[
        results["representation"].eq(RAW_DISPLACEMENT)
        & results["strategy"].eq("top1")
    ].iloc[0]
    print(
        json.dumps(
            {
                "rank_ic": float(rank_row["rank_ic"]),
                "rank_ic_hac_p": float(rank_row["rank_ic_hac_p"]),
                "annualized_return": float(
                    allocation_row["annualized_return"]
                ),
                "spy_alpha": float(allocation_row["spy_alpha"]),
                "circular_randomizations": randomizations,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
