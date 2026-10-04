#!/usr/bin/env python3
"""Recompute published metrics from aggregate, non-licensed paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import load_config
from portfolio import strategy_summary


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_FIELDS = (
    "annualized_return",
    "spy_return",
    "spy_alpha",
    "spy_alpha_p",
    "spy_beta",
    "sharpe",
    "mdd",
    "annualized_turnover",
    "annualized_cost",
)


def error_map(
    reported: pd.DataFrame,
    daily: pd.DataFrame,
    risk_free: pd.DataFrame,
    *,
    prefix: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, Any]] = []
    errors: dict[str, float] = {}
    indexed = reported.set_index("strategy")
    for strategy, frame in daily.groupby("strategy", sort=True):
        summary = strategy_summary(
            frame.sort_values("date").reset_index(drop=True),
            risk_free=risk_free,
        )
        rows.append({"strategy": strategy, **summary})
        expected = indexed.loc[strategy]
        fields = (
            tuple(
                field
                for field in SUMMARY_FIELDS
                if not (strategy == "spy" and field == "spy_alpha_p")
            )
        )
        for field in fields:
            errors[f"{prefix}:{strategy}:{field}"] = abs(
                float(summary[field]) - float(expected[field])
            )
        errors[f"{prefix}:{strategy}:risk_free_flag"] = abs(
            float(bool(summary["sharpe_uses_risk_free"]))
            - float(bool(expected["sharpe_uses_risk_free"]))
        )
    missing = set(indexed.index.astype(str)).difference(
        daily["strategy"].astype(str).unique()
    )
    if missing:
        errors[f"{prefix}:missing_daily_strategies"] = float(len(missing))
    return pd.DataFrame(rows), errors


def recompute_yearly(
    aggregate_dir: Path,
) -> dict[str, float]:
    daily = pd.read_csv(
        aggregate_dir / "portfolio_daily_returns.csv",
        parse_dates=["date"],
    )
    reported = pd.read_csv(aggregate_dir / "portfolio_yearly.csv")
    errors: dict[str, float] = {}
    for (strategy, year), group in daily.groupby(
        ["strategy", daily["date"].dt.year],
        sort=True,
    ):
        selected = reported[
            reported["strategy"].eq(strategy)
            & reported["year"].eq(int(year))
        ]
        if selected.empty:
            errors[f"yearly:{strategy}:{year}:missing"] = 1.0
            continue
        expected = selected.iloc[0]
        realized = float((1.0 + group["net_return"]).prod() - 1.0)
        spy = float((1.0 + group["spy_return"]).prod() - 1.0)
        errors[f"yearly:{strategy}:{year}:return"] = abs(
            realized - float(expected["return"])
        )
        errors[f"yearly:{strategy}:{year}:spy"] = abs(
            spy - float(expected["spy_return"])
        )
    return errors


def alpha_with_lag(daily: pd.DataFrame, lag: int) -> dict[str, float]:
    values = daily["net_return"].astype(float).to_numpy()
    market = daily["spy_return"].astype(float).to_numpy()
    fit = sm.OLS(values, sm.add_constant(market)).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": int(lag)},
    )
    return {
        "annualized_alpha": float(
            (1.0 + float(fit.params[0])) ** 252.0 - 1.0
        ),
        "alpha_p": float(fit.pvalues[0]),
        "beta": float(fit.params[1]),
    }


def recompute_alpha_lags(aggregate_dir: Path) -> dict[str, float]:
    daily = pd.read_csv(
        aggregate_dir / "portfolio_daily_returns.csv",
        parse_dates=["date"],
    )
    reported = pd.read_csv(aggregate_dir / "alpha_lag_sensitivity.csv")
    errors: dict[str, float] = {}
    for row in reported.itertuples(index=False):
        frame = daily[daily["strategy"].eq(row.strategy)].sort_values("date")
        result = alpha_with_lag(frame, int(row.hac_lag))
        for field in ("annualized_alpha", "alpha_p", "beta"):
            errors[f"alpha_lag:{row.strategy}:{row.hac_lag}:{field}"] = abs(
                float(result[field]) - float(getattr(row, field))
            )
    return errors


def recompute_cost_sensitivity(
    aggregate_dir: Path,
    risk_free: pd.DataFrame,
    *,
    daily_name: str = "portfolio_daily_returns.csv",
    sensitivity_name: str = "cost_sensitivity.csv",
    error_prefix: str = "cost",
) -> dict[str, float]:
    daily = pd.read_csv(
        aggregate_dir / daily_name,
        parse_dates=["date"],
    )
    reported = pd.read_csv(aggregate_dir / sensitivity_name)
    errors: dict[str, float] = {}
    fields = (
        "annualized_return",
        "spy_return",
        "spy_alpha",
        "spy_beta",
        "sharpe",
        "mdd",
        "annualized_turnover",
        "annualized_cost",
    )
    for row in reported.itertuples(index=False):
        frame = daily[
            daily["strategy"].eq(row.strategy)
        ].sort_values("date").copy()
        frame["cost"] = (
            frame["turnover"].astype(float)
            * float(row.one_way_cost_bps)
            / 10_000.0
        )
        frame["net_return"] = (
            frame["gross_return"].astype(float) - frame["cost"]
        )
        summary = strategy_summary(frame, risk_free=risk_free)
        prefix = (
            f"{error_prefix}:{row.strategy}:"
            f"{int(row.one_way_cost_bps)}"
        )
        for field in fields:
            errors[f"{prefix}:{field}"] = abs(
                float(summary[field]) - float(getattr(row, field))
            )
        errors[f"{prefix}:alpha_p_nw5"] = abs(
            alpha_with_lag(frame, 5)["alpha_p"]
            - float(row.spy_alpha_p_nw5)
        )
        errors[f"{prefix}:alpha_p_nw42"] = abs(
            alpha_with_lag(frame, 42)["alpha_p"]
            - float(row.spy_alpha_p_nw42)
        )
    return errors


def recompute_factor_alphas(aggregate_dir: Path) -> dict[str, float]:
    daily = pd.read_csv(
        aggregate_dir / "portfolio_daily_returns.csv",
        parse_dates=["date"],
    )
    factors = pd.read_csv(
        aggregate_dir / "factor_daily_ff5_momentum.csv",
        parse_dates=["date"],
    )
    reported = pd.read_csv(aggregate_dir / "factor_regressions.csv")
    errors: dict[str, float] = {}
    model_columns = {
        "capm": ["MKT_RF"],
        "ff5_mom": ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"],
    }
    for row in reported.itertuples(index=False):
        columns = model_columns[row.model]
        frame = (
            daily[daily["strategy"].eq(row.strategy)]
            .merge(factors, on="date", how="inner")
            .dropna(subset=["net_return", "RF", *columns])
        )
        y = frame["net_return"].astype(float)
        if row.strategy != "top1_minus_bottom1":
            y = y - frame["RF"].astype(float)
        fit = sm.OLS(
            y.to_numpy(),
            sm.add_constant(frame[columns]),
        ).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": int(row.maxlags)},
        )
        annualized_alpha = float(
            (1.0 + float(fit.params["const"])) ** 252.0 - 1.0
        )
        prefix = f"factor:{row.strategy}:{row.model}:{row.maxlags}"
        errors[f"{prefix}:observations"] = abs(
            float(len(frame)) - float(row.observations)
        )
        errors[f"{prefix}:annualized_alpha"] = abs(
            annualized_alpha - float(row.annualized_alpha)
        )
        errors[f"{prefix}:alpha_p"] = abs(
            float(fit.pvalues["const"]) - float(row.alpha_p)
        )
        errors[f"{prefix}:adjusted_r2"] = abs(
            float(fit.rsquared_adj) - float(row.adjusted_r2)
        )
        errors[f"{prefix}:market_beta"] = abs(
            float(fit.params["MKT_RF"]) - float(row.market_beta)
        )
        if row.model == "ff5_mom":
            errors[f"{prefix}:momentum_beta"] = abs(
                float(fit.params["MOM"]) - float(row.momentum_beta)
            )
    return errors


def recompute_random_null(aggregate_dir: Path) -> dict[str, float]:
    frame = pd.read_csv(aggregate_dir / "random_sector_null.csv")
    reported = json.loads(
        (aggregate_dir / "random_sector_null_summary.json").read_text(
            encoding="utf-8"
        )
    )
    observed = reported["observed"]
    draws = int(reported["randomizations"])
    values: dict[str, float] = {}
    for metric in ("annualized_return", "spy_alpha", "sharpe"):
        values[f"{metric}_upper"] = float(
            (1 + frame[metric].ge(float(observed[metric])).sum())
            / (draws + 1)
        )
        values[f"{metric}_percentile"] = float(
            frame[metric].lt(float(observed[metric])).mean()
        )
    values["mdd_lower"] = float(
        (1 + frame["mdd"].le(float(observed["mdd"])).sum())
        / (draws + 1)
    )
    values["mdd_percentile"] = float(
        frame["mdd"].gt(float(observed["mdd"])).mean()
    )
    errors = {
        f"random_null:{key}": abs(
            value - float(reported["p_values"][key])
        )
        for key, value in values.items()
    }
    errors["random_null:draw_count"] = abs(float(len(frame) - draws))
    return errors


def verify_temporal_null(aggregate_dir: Path) -> dict[str, float]:
    null = pd.read_csv(aggregate_dir / "temporal_circular_shift_null.csv")
    if "candidate" in null.columns:
        null = null[
            null["candidate"].eq("firm_conditioned_qa_raw")
        ].copy()
    tests = pd.read_csv(aggregate_dir / "circular_shift_tests.csv")
    main = tests[
        tests["candidate"].eq("firm_conditioned_qa_raw")
    ].iloc[0]
    observed = float(main["observed_rank_ic"])
    pvalue = float(
        (1 + null["rank_ic"].ge(observed).sum()) / (len(null) + 1)
    )
    return {
        "temporal:draw_count": abs(
            float(len(null)) - float(main["randomizations"])
        ),
        "temporal:null_mean": abs(
            float(null["rank_ic"].mean()) - float(main["null_mean"])
        ),
        "temporal:pvalue": abs(
            pvalue - float(main["upper_tail_p"])
        ),
    }


def verify_identification(aggregate_dir: Path) -> dict[str, float]:
    history = pd.read_csv(
        aggregate_dir / "history_reference_tests.csv"
    ).set_index("test")
    identity = pd.read_csv(
        aggregate_dir / "issuer_identity_probe.csv"
    ).set_index("representation")
    structured = pd.read_csv(
        aggregate_dir / "structured_news_tests.csv"
    ).set_index("test")
    draws = pd.read_csv(aggregate_dir / "wrong_firm_placebo_draws.csv")
    summary = json.loads(
        (
            aggregate_dir / "identification_expansion_summary.json"
        ).read_text(encoding="utf-8")
    )

    expected_history = {
        "own_firm_history": 0.135064935064935,
        "same_sector_wrong_firm_placebo_mean": 0.0743007034632034,
        "sector_history_excluding_firm": 0.0475378787878787,
        "sector_history_excluding_firm_refit": 0.0788961038961039,
    }
    errors = {
        f"identification:history:{key}": abs(
            float(history.loc[key, "mean_ic"]) - value
        )
        for key, value in expected_history.items()
    }

    own_ic = float(history.loc["own_firm_history", "mean_ic"])
    placebo_p = float(
        (1 + draws["mean_rank_ic"].ge(own_ic).sum()) / (len(draws) + 1)
    )
    reported_placebo = summary["history_placebo"]["draws"]
    errors["identification:placebo:draw_count"] = abs(
        float(len(draws)) - float(reported_placebo["randomizations"])
    )
    errors["identification:placebo:pvalue"] = abs(
        placebo_p - float(reported_placebo["upper_tail_p"])
    )
    errors["identification:placebo:mean"] = abs(
        float(draws["mean_rank_ic"].mean())
        - float(reported_placebo["mean"])
    )

    expected_identity = {
        "current_qa": 0.4140958113870078,
        "own_history_direction": 0.031762295081967214,
    }
    for key, value in expected_identity.items():
        errors[f"identification:identity:{key}"] = abs(
            float(identity.loc[key, "top1_accuracy"]) - value
        )

    reported_earnings = structured.loc[
        "seasonal_pit_proxy: CallRank partial IC"
    ]
    errors["identification:reported_earnings:ic"] = abs(
        float(reported_earnings["mean_ic"]) - 0.1122198177476402
    )
    errors["identification:reported_earnings:pvalue"] = abs(
        float(reported_earnings["newey_west_p"])
        - 0.0517495916927989
    )
    expected_history_p = {
        "own_firm_history": 0.0131230032792955,
        "sector_history_excluding_firm_refit": 0.0976642305759542,
    }
    for key, value in expected_history_p.items():
        errors[f"identification:history_p:{key}"] = abs(
            float(history.loc[key, "newey_west_p"]) - value
        )
    return errors


def verify_identity_controls(aggregate_dir: Path) -> dict[str, float]:
    frame = pd.read_csv(
        aggregate_dir / "identity_control_tests.csv"
    ).set_index(["test", "representation"])
    expected = {
        ("full_pipeline_identity_mask", "raw_residual"): (
            0.19147190821507917,
            0.0024031958666063118,
            0.005949702514874256,
        ),
        ("full_pipeline_identity_mask", "unit_direction"): (
            0.18317187804517912,
            0.0007036732147919635,
            0.0038498075096245186,
        ),
        ("correct_masked_body", "raw_residual"): (
            0.13833637976618604,
            0.018240087382669665,
            None,
        ),
        ("correct_masked_body", "unit_direction"): (
            0.12259544949502317,
            0.023201826465473276,
            None,
        ),
        ("wrong_body_donor", "raw_residual"): (
            -0.05692117708882669,
            0.4056979568015978,
            None,
        ),
        ("wrong_body_donor", "unit_direction"): (
            -0.05395103409270826,
            0.4323752477235936,
            None,
        ),
        ("paired_correct_minus_donor", "raw_residual"): (
            0.19525755685501278,
            0.019554851891726847,
            None,
        ),
        ("paired_correct_minus_donor", "unit_direction"): (
            0.17654648358773145,
            0.026687724286546203,
            None,
        ),
    }
    errors: dict[str, float] = {
        "identity_controls:rows": abs(float(len(frame)) - 8.0),
    }
    monthly = pd.read_csv(
        aggregate_dir / "identity_control_monthly_ic.csv",
        parse_dates=["month"],
    )
    shift_null = pd.read_csv(
        aggregate_dir / "identity_control_shift_null.csv"
    )
    errors["identity_controls:monthly_rows"] = abs(
        float(len(monthly)) - 144.0
    )
    errors["identity_controls:shift_rows"] = abs(
        float(len(shift_null)) - 40_000.0
    )

    def hac_intercept_p(values: pd.Series) -> float:
        clean = values.astype(float).to_numpy()
        fit = sm.OLS(clean, np.ones((len(clean), 1))).fit(
            cov_type="HAC", cov_kwds={"maxlags": 1}
        )
        return float(fit.pvalues[0])

    for key, (estimate, nw_p, shift_p) in expected.items():
        row = frame.loc[key]
        prefix = f"identity_controls:{key[0]}:{key[1]}"
        errors[f"{prefix}:months"] = abs(float(row["months"]) - 24.0)
        errors[f"{prefix}:estimate"] = abs(
            float(row["estimate"]) - estimate
        )
        errors[f"{prefix}:nw_p"] = abs(
            float(row["newey_west_p"]) - nw_p
        )
        if shift_p is not None:
            errors[f"{prefix}:shift_draws"] = abs(
                float(row["calendar_shift_draws"]) - 20_000.0
            )
            errors[f"{prefix}:shift_p"] = abs(
                float(row["calendar_shift_p"]) - shift_p
            )

        if key[0] != "paired_correct_minus_donor":
            series = monthly[
                monthly["test"].eq(key[0])
                & monthly["representation"].eq(key[1])
            ].sort_values("month")
            errors[f"{prefix}:monthly_count"] = abs(
                float(len(series)) - 24.0
            )
            errors[f"{prefix}:monthly_mean"] = abs(
                float(series["rank_ic"].mean()) - estimate
            )
            errors[f"{prefix}:monthly_hac_p"] = abs(
                hac_intercept_p(series["rank_ic"]) - nw_p
            )

        if key[0] == "full_pipeline_identity_mask":
            draws = shift_null[
                shift_null["representation"].eq(key[1])
            ]
            randomization_p = float(
                (1 + draws["rank_ic"].ge(estimate).sum())
                / (len(draws) + 1)
            )
            errors[f"{prefix}:released_shift_count"] = abs(
                float(len(draws)) - 20_000.0
            )
            errors[f"{prefix}:released_shift_p"] = abs(
                randomization_p - float(row["calendar_shift_p"])
            )

    for representation in ("raw_residual", "unit_direction"):
        correct = float(
            frame.loc[("correct_masked_body", representation), "estimate"]
        )
        donor = float(
            frame.loc[("wrong_body_donor", representation), "estimate"]
        )
        gap = float(
            frame.loc[
                ("paired_correct_minus_donor", representation), "estimate"
            ]
        )
        errors[f"identity_controls:{representation}:paired_gap_identity"] = abs(
            correct - donor - gap
        )
        correct_series = monthly[
            monthly["test"].eq("correct_masked_body")
            & monthly["representation"].eq(representation)
        ].set_index("month")["rank_ic"]
        donor_series = monthly[
            monthly["test"].eq("wrong_body_donor")
            & monthly["representation"].eq(representation)
        ].set_index("month")["rank_ic"]
        difference = correct_series.subtract(donor_series).dropna()
        errors[f"identity_controls:{representation}:paired_months"] = abs(
            float(len(difference)) - 24.0
        )
        errors[f"identity_controls:{representation}:paired_mean"] = abs(
            float(difference.mean()) - gap
        )
        errors[f"identity_controls:{representation}:paired_hac_p"] = abs(
            hac_intercept_p(difference)
            - float(
                frame.loc[
                    ("paired_correct_minus_donor", representation),
                    "newey_west_p",
                ]
            )
        )
    return errors


def verify_static_benchmark(aggregate_dir: Path) -> dict[str, float]:
    frame = pd.read_csv(
        aggregate_dir / "static_sector_benchmark.csv"
    ).set_index("ticker")
    expected = {
        ("SPY", "annualized_return"): 0.2401568933275866,
        ("SPY", "mdd"): 0.1875522942522178,
        ("XLU", "annualized_return"): 0.1743894659505798,
        ("XLU", "mdd"): 0.1050452279031481,
    }
    errors = {
        f"static_benchmark:{ticker}:{field}": abs(
            float(frame.loc[ticker, field]) - value
        )
        for (ticker, field), value in expected.items()
    }
    errors["static_benchmark:rows"] = abs(float(len(frame)) - 2.0)
    errors["static_benchmark:trading_days"] = abs(
        float(frame["trading_days"].min()) - 626.0
    ) + abs(float(frame["trading_days"].max()) - 626.0)
    return errors


def verify_raw_robustness(aggregate_dir: Path) -> dict[str, float]:
    frame = pd.read_csv(aggregate_dir / "raw_robustness.csv").set_index("key")
    expected = {
        "main": (24, 0.15782499767414498, 0.007408294451868525),
        "held_out": (24, 0.1575888019324557, 0.0008036094772112977),
        "momentum": (24, 0.1815720418422829, 0.0014394883769302635),
        "market_beta": (24, 0.16882924739366145, 0.007034921659106296),
        "reported_earnings": (24, 0.17086577811682738, 0.0031089583624310073),
        "complete": (19, 0.10747267320941493, 0.10127127289495616),
        "five_firms": (19, 0.06457728056179796, 0.35001766805393886),
        "firm_count": (24, 0.09275000897730191, 0.13448232840781563),
        "history_8": (23, 0.14423613413116868, 0.014040735135563597),
    }
    errors: dict[str, float] = {
        "raw_robustness:rows": abs(float(len(frame)) - float(len(expected))),
        "raw_robustness:positive": 0.0 if frame["ic"].gt(0).all() else 1.0,
        "raw_robustness:significant_intervals": abs(
            float(frame["low"].gt(0).sum()) - 6.0
        ),
    }
    for key, (months, ic, pvalue) in expected.items():
        errors[f"raw_robustness:{key}:months"] = abs(
            float(frame.loc[key, "months"]) - float(months)
        )
        errors[f"raw_robustness:{key}:ic"] = abs(
            float(frame.loc[key, "ic"]) - ic
        )
        errors[f"raw_robustness:{key}:pvalue"] = abs(
            float(frame.loc[key, "p"]) - pvalue
        )
    return errors


def verify_integrity_summaries(aggregate_dir: Path) -> dict[str, float]:
    config = load_config()
    canonical = json.loads(
        (aggregate_dir / "canonical_data_summary.json").read_text(
            encoding="utf-8"
        )
    )
    cleanroom = json.loads(
        (aggregate_dir / "cleanroom_integrity_summary.json").read_text(
            encoding="utf-8"
        )
    )
    aggregate = json.loads(
        (aggregate_dir / "aggregate_summary.json").read_text(
            encoding="utf-8"
        )
    )
    integrity = config["data_integrity"]
    population = cleanroom["population"]
    coverage = cleanroom["coverage_and_sections"]
    hashes = cleanroom["output_sha256"]
    return {
        "integrity:ledger_rows": abs(
            float(canonical["source_rows"]) - float(integrity["ledger_rows"])
        ),
        "integrity:clean_calls": abs(
            float(population["cleanroom_usable"])
            - float(integrity["canonical_usable_calls"])
        ),
        "integrity:evaluation_calls": abs(
            float(population["evaluation_cleanroom_usable"])
            - float(integrity["evaluation_cleanroom_calls"])
        ),
        "integrity:evaluation_exact_dates": abs(
            float(population["evaluation_call_date_exact"])
            - float(integrity["evaluation_exact_official_date"])
        ),
        "integrity:evaluation_confirmed_months": abs(
            float(population["evaluation_call_date_month_confirmed"])
            - float(integrity["evaluation_official_month"])
        ),
        "integrity:evaluation_unresolved_excluded": abs(
            float(population["evaluation_external_unresolved"])
            - float(integrity["evaluation_unresolved_excluded"])
        ),
        "integrity:evaluation_date_partition": abs(
            float(population["evaluation_call_date_exact"])
            + float(population["evaluation_call_date_month_confirmed"])
            - float(population["evaluation_cleanroom_usable"])
        ),
        "integrity:issuers": abs(
            float(coverage["canonical_issuers"])
            - float(integrity["canonical_issuers"])
        ),
        "integrity:event_map_hash": float(
            hashes["cleanroom_event_map.parquet"]
            != integrity["cleanroom_event_map_sha256"]
        ),
        "integrity:portfolio_event_map_hash": float(
            aggregate["input_hashes"]["cleanroom_event_map"]
            != hashes["cleanroom_event_map.parquet"]
        ),
    }


def verify_aggregate(
    aggregate_dir: Path,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    aggregate_dir = aggregate_dir.resolve()
    risk_free = pd.read_csv(
        ROOT / "data" / "DGS3MO_2023-10-01_2026-06-05.csv"
    )
    portfolio_reported = pd.read_csv(
        aggregate_dir / "portfolio_summary.csv"
    )
    portfolio_daily = pd.read_csv(
        aggregate_dir / "portfolio_daily_returns.csv",
        parse_dates=["date"],
    )
    portfolio, portfolio_errors = error_map(
        portfolio_reported,
        portfolio_daily,
        risk_free,
        prefix="portfolio",
    )
    raw_reported = pd.read_csv(
        aggregate_dir / "raw_portfolio_summary.csv"
    )
    raw_daily = pd.read_csv(
        aggregate_dir / "raw_portfolio_daily_returns.csv",
        parse_dates=["date"],
    )
    raw_portfolio, raw_portfolio_errors = error_map(
        raw_reported,
        raw_daily,
        risk_free,
        prefix="raw_portfolio",
    )

    baseline_reported = pd.read_csv(
        aggregate_dir / "baseline_portfolio_summary.csv"
    )
    baseline_daily = pd.read_csv(
        aggregate_dir / "baseline_portfolio_daily_returns.csv",
        parse_dates=["date"],
    )
    callrank_top1 = portfolio_daily[
        portfolio_daily["strategy"].eq("top1")
    ].copy()
    callrank_top1["strategy"] = "callrank_top1"
    baseline_daily = pd.concat(
        [baseline_daily, callrank_top1],
        ignore_index=True,
    )
    _, baseline_errors = error_map(
        baseline_reported,
        baseline_daily,
        risk_free,
        prefix="baseline",
    )

    all_errors = {
        **portfolio_errors,
        **raw_portfolio_errors,
        **baseline_errors,
        **recompute_yearly(aggregate_dir),
        **recompute_alpha_lags(aggregate_dir),
        **recompute_cost_sensitivity(aggregate_dir, risk_free),
        **recompute_cost_sensitivity(
            aggregate_dir,
            risk_free,
            daily_name="raw_portfolio_daily_returns.csv",
            sensitivity_name="raw_cost_sensitivity.csv",
            error_prefix="raw_cost",
        ),
        **recompute_factor_alphas(aggregate_dir),
        **recompute_random_null(aggregate_dir),
        **verify_temporal_null(aggregate_dir),
        **verify_identification(aggregate_dir),
        **verify_identity_controls(aggregate_dir),
        **verify_static_benchmark(aggregate_dir),
        **verify_raw_robustness(aggregate_dir),
        **verify_integrity_summaries(aggregate_dir),
    }
    maximum_error = max(all_errors.values())
    top1 = portfolio.set_index("strategy").loc["top1"]
    raw_indexed = raw_portfolio.set_index("strategy")
    raw_top1 = raw_indexed.loc["top1"]
    encoders = pd.read_csv(aggregate_dir / "encoder_summary.csv")
    costs = pd.read_csv(aggregate_dir / "cost_sensitivity.csv").set_index(
        ["strategy", "one_way_cost_bps"]
    )
    paired = pd.read_csv(aggregate_dir / "paired_ic_tests.csv").set_index(
        "comparison"
    )
    structured = pd.read_csv(
        aggregate_dir / "structured_news_tests.csv"
    ).set_index("test")
    corrections = pd.read_csv(
        aggregate_dir / "training_correction_ablation.csv"
    ).set_index(["scenario", "candidate"])
    static = pd.read_csv(
        aggregate_dir / "static_sector_benchmark.csv"
    ).set_index("ticker")
    identification = json.loads(
        (
            aggregate_dir / "identification_expansion_summary.json"
        ).read_text(encoding="utf-8")
    )
    concentration = json.loads(
        (aggregate_dir / "aggregate_summary.json").read_text(encoding="utf-8")
    )["top1_selection"]
    adverse = pd.read_csv(
        aggregate_dir / "grouped_signal_month_exclusions.csv"
    ).set_index("test")
    checks = {
        "aggregate_paths_recompute": maximum_error <= tolerance,
        "top1_return_matches": abs(
            float(top1["annualized_return"]) - 0.4212124618092991
        )
        <= tolerance,
        "top1_alpha_matches": abs(
            float(top1["spy_alpha"]) - 0.2704345799139105
        )
        <= tolerance,
        "top1_sharpe_uses_risk_free": bool(
            top1["sharpe_uses_risk_free"]
        ),
        "raw_top1_return_matches": abs(
            float(raw_top1["annualized_return"]) - 0.256502268485077
        )
        <= tolerance,
        "raw_top1_alpha_matches": abs(
            float(raw_top1["spy_alpha"]) - 0.1310372243516391
        )
        <= tolerance,
        "raw_spread_alpha_matches": abs(
            float(
                raw_indexed.loc["top1_minus_bottom1", "spy_alpha"]
            )
            - 0.1731693935257914
        )
        <= tolerance,
        "identity_controls_match": max(
            verify_identity_controls(aggregate_dir).values()
        )
        <= tolerance,
        "twenty_thousand_temporal_draws": len(
            pd.read_csv(
                aggregate_dir / "temporal_circular_shift_null.csv"
            )
        )
        == 20_000,
        "two_thousand_portfolio_nulls": len(
            pd.read_csv(aggregate_dir / "random_sector_null.csv")
        )
        == 2_000,
        "three_encoder_summaries": set(encoders["model"])
        == {"sec_bert", "minilm", "mpnet"},
        "one_thousand_wrong_firm_placebos": len(
            pd.read_csv(
                aggregate_dir / "wrong_firm_placebo_draws.csv"
            )
        )
        == 1_000,
        "leader_spans_ten_sectors_and_switches_sixteen_times": (
            int(concentration["distinct_sectors"]) == 10
            and int(concentration["sector_switches"]) == 16
            and int(concentration["transitions"]) == 23
        ),
        "adverse_stress_check_matches": (
            abs(
                float(
                    portfolio.set_index("strategy").loc[
                        "top1_ex_utilities", "spy_alpha"
                    ]
                )
                - 0.1120088876960676
            )
            <= tolerance
            and abs(
                float(
                    adverse.loc[
                        "exclude_2024_signal_months:top1", "spy_alpha"
                    ]
                )
                - 0.11750580689586498
            )
            <= tolerance
            and abs(
                float(
                    adverse.loc[
                        "exclude_three_largest_alpha_contributors:top1",
                        "annualized_return",
                    ]
                )
                - 0.3214921959730128
            )
            <= tolerance
            and abs(
                float(
                    adverse.loc[
                        "exclude_three_largest_alpha_contributors:top1",
                        "spy_alpha",
                    ]
                )
                - 0.17891870917017427
            )
            <= tolerance
            and abs(
                float(
                    adverse.loc[
                        "exclude_three_largest_alpha_contributors:top1",
                        "spy_alpha_p_nw42",
                    ]
                )
                - 0.019357881063845263
            )
            <= tolerance
        ),
        "top1_outperforms_spy_at_20bps": float(
            costs.loc[("top1", 20), "annualized_return"]
        )
        > float(costs.loc[("top1", 20), "spy_return"]),
        "top1_alpha_positive_at_20bps": float(
            costs.loc[("top1", 20), "spy_alpha"]
        )
        > 0.0,
        "spread_alpha_positive_at_20bps": float(
            costs.loc[("top1_minus_bottom1", 20), "spy_alpha"]
        )
        > 0.0,
        "raw_unit_paired_test_matches": abs(
            float(
                paired.loc[
                    "firm_conditioned_qa_raw",
                    "newey_west_p",
                ]
            )
            - 0.20038172970246038
        )
        <= tolerance,
        "seasonal_structured_combination_matches": (
            abs(
                float(
                    structured.loc[
                        "seasonal_pit_proxy: structured only",
                        "mean_ic",
                    ]
                )
                - 0.06463744588744588
            )
            <= tolerance
            and abs(
                float(
                    structured.loc[
                        "seasonal_pit_proxy: equal-rank combined",
                        "mean_ic",
                    ]
                )
                - 0.10902909668638272
            )
            <= tolerance
            and abs(
                float(
                    structured.loc[
                        "seasonal_pit_proxy: equal-rank combined",
                        "newey_west_p",
                    ]
                )
                - 0.025078027403692506
            )
            <= tolerance
            and abs(
                float(
                    structured.loc[
                        "seasonal_pit_proxy: CallRank encompassing coefficient",
                        "newey_west_p",
                    ]
                )
                - 0.03889727863785275
            )
            <= tolerance
        ),
        "return_blind_correction_matches": (
            abs(
                float(
                    corrections.loc[
                        ("old", "firm_conditioned_qa_direction"),
                        "rank_ic",
                    ]
                )
                - 0.14178664668652827
            )
            <= tolerance
            and abs(
                float(
                    corrections.loc[
                        (
                            "full_correction",
                            "firm_conditioned_qa_direction",
                        ),
                        "rank_ic",
                    ]
                )
                - 0.10037082760565695
            )
            <= tolerance
            and abs(
                float(
                    corrections.loc[
                        ("full_correction", "firm_conditioned_qa_raw"),
                        "rank_ic",
                    ]
                )
                - 0.15782499767414498
            )
            <= tolerance
        ),
        "wrong_firm_placebo_p_matches": abs(
            float(
                identification["history_placebo"]["draws"][
                    "upper_tail_p"
                ]
            )
            - 0.004995004995004995
        )
        <= tolerance,
        "static_xlu_comparison_matches": (
            abs(
                float(static.loc["XLU", "annualized_return"])
                - 0.1743894659505798
            )
            <= tolerance
            and abs(
                float(static.loc["SPY", "annualized_return"])
                - 0.2401568933275866
            )
            <= tolerance
        ),
    }
    return {
        "status": "PASS"
        if all(checks.values())
        else "FAIL",
        "maximum_numeric_error": maximum_error,
        "checks": checks,
        "errors": all_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregate-dir",
        type=Path,
        default=ROOT / "aggregate_results",
    )
    parser.add_argument("--tolerance", type=float, default=1e-10)
    args = parser.parse_args()
    result = verify_aggregate(args.aggregate_dir, args.tolerance)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
