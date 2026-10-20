#!/usr/bin/env python3
"""Point-in-time sector selection and overlapping-sleeve accounting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import SECTOR_ETF, load_config


ETFS = tuple(SECTOR_ETF.values())


def prepare_prices(prices: pd.DataFrame) -> pd.DataFrame:
    output = prices.rename(columns={"index": "Date"}).copy()
    output["Date"] = pd.to_datetime(output["Date"])
    output = output.sort_values("Date").reset_index(drop=True)
    missing = {"Date", "SPY", *ETFS}.difference(output.columns)
    if missing:
        raise ValueError(f"Price columns are missing: {sorted(missing)}")
    return output


def prepare_caps(caps: pd.DataFrame) -> pd.DataFrame:
    output = caps.rename(columns={"cap_date": "Date"}).copy()
    required = {"Date", "sector", "market_cap"}
    missing = required.difference(output.columns)
    if missing:
        raise ValueError(f"Capitalization columns are missing: {sorted(missing)}")
    output["Date"] = pd.to_datetime(output["Date"])
    output["market_cap"] = pd.to_numeric(output["market_cap"], errors="coerce")
    if output.duplicated(["Date", "sector"]).any():
        raise ValueError("Sector capitalization keys are not unique")
    return output.sort_values(["Date", "sector"]).reset_index(drop=True)


def daily_risk_free_returns(
    dates: pd.Series,
    risk_free: pd.DataFrame,
) -> np.ndarray:
    """Map each return date to the latest strictly prior Treasury yield."""
    frame = risk_free.copy()
    date_column = next(
        (column for column in ("observation_date", "Date", "date") if column in frame),
        None,
    )
    rate_column = next(
        (column for column in ("DGS3MO", "annual_yield_pct") if column in frame),
        None,
    )
    if date_column is None or rate_column is None:
        raise ValueError("Risk-free data require a date and DGS3MO yield column")
    frame = frame[[date_column, rate_column]].rename(
        columns={date_column: "rf_date", rate_column: "annual_yield_pct"}
    )
    frame["rf_date"] = pd.to_datetime(frame["rf_date"])
    frame["annual_yield_pct"] = pd.to_numeric(
        frame["annual_yield_pct"], errors="coerce"
    )
    frame = (
        frame.dropna()
        .sort_values("rf_date")
        .drop_duplicates("rf_date", keep="last")
    )
    requested = pd.DataFrame(
        {
            "position": np.arange(len(dates), dtype=int),
            "date": pd.to_datetime(dates).to_numpy(),
        }
    ).sort_values("date")
    matched = pd.merge_asof(
        requested,
        frame,
        left_on="date",
        right_on="rf_date",
        direction="backward",
        allow_exact_matches=False,
    ).sort_values("position")
    if matched["annual_yield_pct"].isna().any():
        raise ValueError("Risk-free history does not precede every return date")
    return (
        (1.0 + matched["annual_yield_pct"].to_numpy(dtype=float) / 100.0)
        ** (1.0 / 252.0)
        - 1.0
    )


def selection_weights(
    holdings: pd.DataFrame,
    side: str,
    weighting: str,
    cap_values: np.ndarray | None,
) -> np.ndarray:
    count = len(holdings)
    if weighting == "pit_market_cap":
        if cap_values is None or not np.isfinite(cap_values).all() or np.any(cap_values <= 0):
            raise ValueError("Selected sectors lack positive prior-close capitalization")
        raw = cap_values
    elif weighting == "equal":
        raw = np.ones(count, dtype=float)
    elif weighting == "score_rank":
        raw = np.arange(1, count + 1, dtype=float)
        if side == "bottom":
            raw = raw[::-1]
    else:
        raise ValueError(f"Unknown weighting: {weighting}")
    return raw / raw.sum()


def build_sleeves(
    scored: pd.DataFrame,
    prices: pd.DataFrame,
    caps: pd.DataFrame,
    *,
    side: str = "top",
    count: int = 1,
    weighting: str = "pit_market_cap",
    tie_policy: str = "canonical_sector",
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    if side not in {"top", "bottom"}:
        raise ValueError("side must be top or bottom")
    if tie_policy not in {"canonical_sector", "all_cutoff"}:
        raise ValueError("Unknown tie policy")
    config = load_config()
    horizon = int(config["portfolio"]["holding_days"])
    minimum = int(config["evaluation"]["minimum_sectors_per_month"])
    price_frame = prepare_prices(prices)
    cap_frame = prepare_caps(caps)
    price_index = {pd.Timestamp(date): index for index, date in enumerate(price_frame["Date"])}
    cap_lookup = cap_frame.set_index(["Date", "sector"])["market_cap"]
    sleeves: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    for month, group in scored.groupby("month", sort=True):
        if len(group) < max(minimum, count):
            continue
        entry_date = pd.Timestamp(group["entry_date"].iloc[0])
        entry_index = price_index.get(entry_date)
        if entry_index is None or entry_index == 0 or entry_index + horizon >= len(price_frame):
            continue
        if group["entry_date"].nunique() != 1:
            raise ValueError("A signal month has multiple entry dates")
        cap_date = pd.Timestamp(price_frame.loc[entry_index - 1, "Date"])
        ordered = group.sort_values(["pred", "sector"], kind="mergesort")
        if side == "top":
            holdings = ordered.tail(count)
            if tie_policy == "all_cutoff":
                cutoff = float(holdings["pred"].min())
                holdings = ordered[ordered["pred"].ge(cutoff)]
        else:
            holdings = ordered.head(count)
            if tie_policy == "all_cutoff":
                cutoff = float(holdings["pred"].max())
                holdings = ordered[ordered["pred"].le(cutoff)]
        cap_values = np.asarray(
            [cap_lookup.get((cap_date, sector), np.nan) for sector in holdings["sector"]],
            dtype=float,
        )
        weights_array = selection_weights(holdings, side, weighting, cap_values)
        weights = {ticker: 0.0 for ticker in ETFS}
        for sector, cap_value, weight in zip(
            holdings["sector"], cap_values, weights_array, strict=True
        ):
            ticker = SECTOR_ETF[str(sector)]
            weights[ticker] = float(weight)
            weight_rows.append(
                {
                    "month": pd.Timestamp(month),
                    "cap_date": cap_date,
                    "entry_date": entry_date,
                    "sector": str(sector),
                    "etf": ticker,
                    "market_cap": float(cap_value),
                    "weight": float(weight),
                    "tie_policy": tie_policy,
                }
            )
        exit_index = entry_index + horizon
        sleeves.append(
            {
                "id": str(pd.Timestamp(month).date()),
                "month": pd.Timestamp(month),
                "entry_index": entry_index,
                "exit_index": exit_index,
                "entry_date": entry_date,
                "exit_date": pd.Timestamp(price_frame.loc[exit_index, "Date"]),
                "weights": weights,
            }
        )
    weight_frame = pd.DataFrame(weight_rows)
    if len(weight_frame):
        if not weight_frame["cap_date"].lt(weight_frame["entry_date"]).all():
            raise RuntimeError("A capitalization date does not precede entry")
        if not weight_frame.groupby("month")["weight"].sum().sub(1.0).abs().lt(1e-12).all():
            raise RuntimeError("Sleeve weights do not sum to one")
    return sleeves, weight_frame


def daily_returns(sleeves: list[dict[str, Any]], prices: pd.DataFrame) -> pd.DataFrame:
    if not sleeves:
        raise ValueError("No eligible sleeves were built")
    config = load_config()
    price_frame = prepare_prices(prices)
    start = min(int(sleeve["entry_index"]) for sleeve in sleeves)
    end = max(int(sleeve["exit_index"]) for sleeve in sleeves)
    positions = {ticker: 0.0 for ticker in ETFS}
    previous_active: tuple[str, ...] = ()
    cost_rate = float(config["portfolio"]["one_way_cost_bps"]) / 10_000.0
    rows: list[dict[str, Any]] = []
    for index in range(start, end + 1):
        asset_returns: dict[str, float] = {}
        for ticker in ETFS:
            previous = float(price_frame.loc[index - 1, ticker])
            current = float(price_frame.loc[index, ticker])
            if np.isfinite(previous) and np.isfinite(current) and previous > 0:
                asset_returns[ticker] = current / previous - 1.0
            elif abs(positions[ticker]) <= 1e-15:
                asset_returns[ticker] = 0.0
            else:
                raise RuntimeError(f"Missing held-asset price for {ticker}")
        spy_previous = float(price_frame.loc[index - 1, "SPY"])
        spy_current = float(price_frame.loc[index, "SPY"])
        if not np.isfinite([spy_previous, spy_current]).all() or spy_previous <= 0:
            raise RuntimeError("Missing SPY price")
        # The portfolio is established at the first date's close, so the
        # benchmark must begin at that same close.
        spy_return = (
            0.0 if index == start else spy_current / spy_previous - 1.0
        )
        gross_return = float(sum(positions[t] * asset_returns[t] for t in ETFS))
        denominator = 1.0 + gross_return
        drifted = {
            ticker: positions[ticker] * (1.0 + asset_returns[ticker]) / denominator
            for ticker in ETFS
        }
        active = [
            sleeve
            for sleeve in sleeves
            if int(sleeve["entry_index"]) <= index < int(sleeve["exit_index"])
        ]
        active_ids = tuple(str(sleeve["id"]) for sleeve in active)
        if active_ids != previous_active:
            target = {
                ticker: (
                    sum(float(sleeve["weights"][ticker]) for sleeve in active) / len(active)
                    if active
                    else 0.0
                )
                for ticker in ETFS
            }
        else:
            target = drifted
        turnover = float(sum(abs(target[t] - drifted[t]) for t in ETFS))
        cost = turnover * cost_rate
        rows.append(
            {
                "date": pd.Timestamp(price_frame.loc[index, "Date"]),
                "gross_return": gross_return,
                "net_return": gross_return - cost,
                "spy_return": spy_return,
                "turnover": turnover,
                "cost": cost,
                "active_sleeves": len(active),
            }
        )
        positions = target
        previous_active = active_ids
    return pd.DataFrame(rows)


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.astype(float)).cumprod()
    return float(-(wealth / wealth.cummax() - 1.0).min())


def strategy_summary(
    daily: pd.DataFrame,
    risk_free: pd.DataFrame | None = None,
) -> dict[str, float | int | str | bool]:
    config = load_config()
    annualization = float(config["portfolio"]["annualization_days"])
    values = daily["net_return"].astype(float)
    years = len(values) / annualization
    annualized_return = float((1.0 + values).prod() ** (1.0 / years) - 1.0)
    if risk_free is None:
        excess_values = values.to_numpy(dtype=float)
    else:
        excess_values = (
            values.to_numpy(dtype=float)
            - daily_risk_free_returns(daily["date"], risk_free)
        )
    spy = daily["spy_return"].astype(float)
    spy_return = float((1.0 + spy).prod() ** (1.0 / years) - 1.0)
    x = sm.add_constant(spy.to_numpy())
    lag = int(config["evaluation"]["newey_west_daily_alpha_lag"])
    fit = sm.OLS(values.to_numpy(), x).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    daily_alpha = float(fit.params[0])
    return {
        "start": str(pd.Timestamp(daily["date"].min()).date()),
        "end": str(pd.Timestamp(daily["date"].max()).date()),
        "trading_days": int(len(values)),
        "annualized_return": annualized_return,
        "spy_return": spy_return,
        "spy_alpha": float((1.0 + daily_alpha) ** annualization - 1.0),
        "spy_alpha_p": float(fit.pvalues[0]),
        "spy_beta": float(fit.params[1]),
        "sharpe": float(
            np.sqrt(annualization)
            * excess_values.mean()
            / values.std(ddof=1)
        ),
        "sharpe_uses_risk_free": bool(risk_free is not None),
        "mdd": max_drawdown(values),
        "annualized_turnover": float(daily["turnover"].sum() / years),
        "annualized_cost": float(daily["cost"].sum() / years),
    }


def momentum_scores(scored: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    config = load_config()
    lookback = int(config["target"]["horizon_trading_days"])
    price_frame = prepare_prices(prices)
    date_index = {pd.Timestamp(date): index for index, date in enumerate(price_frame["Date"])}
    rows: list[pd.DataFrame] = []
    for _, group in scored.groupby("month", sort=True):
        entry_date = pd.Timestamp(group["entry_date"].iloc[0])
        entry_index = date_index.get(entry_date)
        if entry_index is None or entry_index - 1 - lookback < 0:
            continue
        end = entry_index - 1
        start = end - lookback
        output = group.copy()
        scores = []
        for sector in output["sector"]:
            ticker = SECTOR_ETF[str(sector)]
            sector_return = price_frame.loc[end, ticker] / price_frame.loc[start, ticker] - 1.0
            spy_return = price_frame.loc[end, "SPY"] / price_frame.loc[start, "SPY"] - 1.0
            scores.append(float(sector_return - spy_return))
        output["pred"] = scores
        output["momentum_formation_end"] = pd.Timestamp(price_frame.loc[end, "Date"])
        rows.append(output)
    result = pd.concat(rows, ignore_index=True)
    if not result["momentum_formation_end"].lt(result["entry_date"]).all():
        raise RuntimeError("Momentum formation reaches or follows entry")
    return result
