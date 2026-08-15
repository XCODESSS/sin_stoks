"""Walk-forward portfolio optimization backtesting engine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from config import (
    DEFAULT_MAX_WEIGHT,
    DEFAULT_TRANSACTION_COST_BPS,
    MIN_ESTIMATION_WEEKS,
    REBALANCE_YEARS,
    RISK_FREE_RATE,
    STARTING_VALUE,
    WEEKS_PER_YEAR,
)
from portfolio_strategies import StrategyCallable, StrategyConfig


@dataclass
class BacktestConfig:
    """Parameters governing walk-forward backtest execution."""

    rebalance_years: list[int] = field(default_factory=lambda: list(REBALANCE_YEARS))
    max_weight: float = DEFAULT_MAX_WEIGHT
    risk_free_rate: float = RISK_FREE_RATE
    min_estimation_weeks: int = MIN_ESTIMATION_WEEKS
    starting_value: float = STARTING_VALUE
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS


@dataclass
class BacktestResult:
    """Strongly-typed container for walk-forward execution results."""

    period_returns: pd.DataFrame
    portfolio_values: pd.DataFrame
    weights: pd.DataFrame
    turnover: pd.DataFrame
    config: BacktestConfig


def get_rebalance_dates(rebalance_years: list[int]) -> list[pd.Timestamp]:
    """Generate annual rebalance date timestamps."""
    return [pd.Timestamp(f"{year}-01-01") for year in rebalance_years]


def get_training_window(
    returns: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    min_periods: int = MIN_ESTIMATION_WEEKS,
) -> pd.DataFrame:
    """Slice historical training observations strictly prior to rebalance date."""
    train = returns.loc[returns.index < rebalance_date]
    if len(train) < min_periods:
        raise ValueError(
            f"Training window before {rebalance_date.date()} has only {len(train)} periods; "
            f"needs at least {min_periods}."
        )
    return train


def get_holding_period(
    returns: pd.DataFrame | pd.Series, rebalance_date: pd.Timestamp
) -> pd.DataFrame | pd.Series:
    """The 1-year out-of-sample holding period starting at rebalance_date."""
    next_rebalance_date = rebalance_date + pd.DateOffset(years=1)
    return returns.loc[(returns.index >= rebalance_date) & (returns.index < next_rebalance_date)]


def build_covariance(train: pd.DataFrame, periods_per_year: int = WEEKS_PER_YEAR) -> pd.DataFrame:
    """Annualized Ledoit-Wolf shrinkage covariance matrix from weekly log returns."""
    simple_returns = np.expm1(train)
    shrunk = LedoitWolf().fit(simple_returns.to_numpy())
    return pd.DataFrame(
        shrunk.covariance_ * periods_per_year,
        index=train.columns,
        columns=train.columns,
    )


def build_expected_returns(train: pd.DataFrame, periods_per_year: int = WEEKS_PER_YEAR) -> pd.Series:
    """Annualized arithmetic expected return per asset from historical simple returns."""
    simple_returns = np.expm1(train)
    return simple_returns.mean() * periods_per_year


def apply_weights(weights: pd.Series, holding_period: pd.DataFrame) -> pd.Series:
    """Calculate portfolio out-of-sample weekly returns with intra-year asset price drift."""
    if holding_period.empty:
        return pd.Series(dtype=float)
    simple_returns = np.expm1(holding_period[weights.index])
    asset_growth = (1.0 + simple_returns).cumprod()
    portfolio_growth = asset_growth @ weights
    period_returns = portfolio_growth.pct_change()
    period_returns.iloc[0] = portfolio_growth.iloc[0] - 1.0
    return period_returns

def calculate_turnover(old_weights: pd.Series, new_weights: pd.Series) -> float:
    """One-way turnover: half the sum of absolute weight changes."""
    aligned_old = old_weights.reindex(new_weights.index, fill_value=0.0)
    return float((new_weights - aligned_old).abs().sum() / 2.0)


def calculate_drifted_weights(weights: pd.Series, holding_period: pd.DataFrame) -> pd.Series:
    """Asset-level weights at the end of the holding period, after price drift — no rebalancing."""
    if holding_period.empty:
        return weights
    simple_returns = np.expm1(holding_period[weights.index])
    asset_growth = (1.0 + simple_returns).cumprod().iloc[-1]
    drifted_value = weights * asset_growth
    return drifted_value / drifted_value.sum()

def build_spy_benchmark(
    spy_log_returns: pd.Series,
    rebalance_dates: list[pd.Timestamp],
    starting_value: float = STARTING_VALUE,
) -> tuple[pd.Series, pd.Series]:
    """Compute SPY benchmark out-of-sample weekly returns and cumulative value."""
    spy_holding_chunks = [get_holding_period(spy_log_returns, r_date) for r_date in rebalance_dates]
    spy_log_series = pd.concat(spy_holding_chunks)
    spy_returns = np.expm1(spy_log_series)
    spy_returns.name = "SPY"
    spy_values = starting_value * (1.0 + spy_returns).cumprod()
    spy_values.name = "SPY"
    return spy_returns, spy_values


def run_walk_forward_backtest(
    returns: pd.DataFrame,
    spy_log_returns: pd.Series,
    strategies: Mapping[str, StrategyCallable],
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Execute the full walk-forward portfolio optimization and backtesting loop."""
    cfg = config or BacktestConfig()
    rebalance_dates = get_rebalance_dates(cfg.rebalance_years)
    strategy_cfg = StrategyConfig(max_weight=cfg.max_weight, risk_free_rate=cfg.risk_free_rate)

    weight_rows: dict[tuple[pd.Timestamp, str], pd.Series] = {}
    return_chunks: dict[str, list[pd.Series]] = {name: [] for name in strategies}
    turnover_rows: list[dict[str, object]] = []
    previous_drifted_weights: dict[str, pd.Series] = {}
    cost_fraction = cfg.transaction_cost_bps / 10_000.0

    for rebalance_date in rebalance_dates:
        train = get_training_window(returns, rebalance_date, min_periods=cfg.min_estimation_weeks)
        test = get_holding_period(returns, rebalance_date)

        covariance = build_covariance(train, periods_per_year=WEEKS_PER_YEAR)
        expected_returns = build_expected_returns(train, periods_per_year=WEEKS_PER_YEAR)

        for strat_name, strat_fn in strategies.items():
            weights = strat_fn(expected_returns, covariance, strategy_cfg)
            weight_rows[(rebalance_date, strat_name)] = weights

            if strat_name in previous_drifted_weights:
                turnover = calculate_turnover(previous_drifted_weights[strat_name], weights)
            else:
                turnover = 1.0  # first rebalance: full investment from cash

            cost = turnover * cost_fraction
            turnover_rows.append(
                {"Rebalance Date": rebalance_date, "Strategy": strat_name, "Turnover": turnover, "Cost": cost}
            )

            strat_returns = apply_weights(weights, test)
            if not strat_returns.empty:
                strat_returns.iloc[0] -= cost
            return_chunks[strat_name].append(strat_returns)

            previous_drifted_weights[strat_name] = calculate_drifted_weights(weights, test)

    weights_df = pd.DataFrame(weight_rows).T
    weights_df.index = weights_df.index.set_names(["Rebalance Date", "Strategy"])
    turnover_df = pd.DataFrame(turnover_rows)

    period_returns = pd.DataFrame(
        {strat_name: pd.concat(chunks) for strat_name, chunks in return_chunks.items()}
    )

    spy_returns, _ = build_spy_benchmark(spy_log_returns, rebalance_dates, cfg.starting_value)
    overlap = period_returns.index.intersection(spy_returns.index)
    if overlap.empty:
        raise ValueError("SPY benchmark and portfolio return indices do not overlap.")

    aligned_spy = spy_returns.reindex(period_returns.index).fillna(0.0)
    period_returns["SPY"] = aligned_spy

    # Build compounded portfolio values
    portfolio_values = cfg.starting_value * (1.0 + period_returns).cumprod()
    start_date = pd.Timestamp(f"{min(cfg.rebalance_years)}-01-01")
    if start_date not in portfolio_values.index:
        start_row = pd.DataFrame(cfg.starting_value, index=[start_date], columns=portfolio_values.columns)
        portfolio_values = pd.concat([start_row, portfolio_values])

    return BacktestResult(
        period_returns=period_returns,
        portfolio_values=portfolio_values,
        weights=weights_df,
        turnover=turnover_df,
        config=cfg,
    )
