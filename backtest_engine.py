"""Walk-forward portfolio optimization backtesting engine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from config import (
    DEFAULT_MAX_WEIGHT,
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


@dataclass
class BacktestResult:
    """Strongly-typed container for walk-forward execution results."""

    period_returns: pd.DataFrame
    portfolio_values: pd.DataFrame
    weights: pd.DataFrame
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


def build_covariance(train: pd.DataFrame) -> pd.DataFrame:
    """Annualized Ledoit-Wolf shrinkage covariance matrix from weekly log returns."""
    simple_returns = np.expm1(train)
    shrunk = LedoitWolf().fit(simple_returns.to_numpy())
    return pd.DataFrame(
        shrunk.covariance_ * WEEKS_PER_YEAR,
        index=train.columns,
        columns=train.columns,
    )


def build_expected_returns(train: pd.DataFrame) -> pd.Series:
    """Annualized arithmetic expected return per asset from historical simple returns."""
    simple_returns = np.expm1(train)
    return simple_returns.mean() * WEEKS_PER_YEAR


def apply_weights(weights: pd.Series, holding_period: pd.DataFrame) -> pd.Series:
    """Calculate portfolio out-of-sample weekly returns with intra-year asset price drift."""
    simple_returns = np.expm1(holding_period[weights.index])
    asset_growth = (1.0 + simple_returns).cumprod()
    portfolio_growth = asset_growth @ weights
    period_returns = portfolio_growth.pct_change()
    period_returns.iloc[0] = portfolio_growth.iloc[0] - 1.0
    return period_returns


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

    for rebalance_date in rebalance_dates:
        train = get_training_window(returns, rebalance_date, min_periods=cfg.min_estimation_weeks)
        test = get_holding_period(returns, rebalance_date)

        covariance = build_covariance(train)
        expected_returns = build_expected_returns(train)

        for strat_name, strat_fn in strategies.items():
            weights = strat_fn(expected_returns, covariance, strategy_cfg)
            weight_rows[(rebalance_date, strat_name)] = weights
            return_chunks[strat_name].append(apply_weights(weights, test))

    weights_df = pd.DataFrame(weight_rows).T
    weights_df.index = weights_df.index.set_names(["Rebalance Date", "Strategy"])

    period_returns = pd.DataFrame(
        {strat_name: pd.concat(chunks) for strat_name, chunks in return_chunks.items()}
    )

    # Align SPY benchmark returns
    spy_returns, _ = build_spy_benchmark(spy_log_returns, rebalance_dates, cfg.starting_value)
    aligned_spy = spy_returns.reindex(period_returns.index)
    period_returns["SPY"] = aligned_spy

    # Build compounded portfolio values
    portfolio_values = cfg.starting_value * (1.0 + period_returns).cumprod()
    start_date = pd.Timestamp(f"{cfg.rebalance_years[0]}-01-01")
    if start_date not in portfolio_values.index:
        start_row = pd.DataFrame(cfg.starting_value, index=[start_date], columns=portfolio_values.columns)
        portfolio_values = pd.concat([start_row, portfolio_values])

    return BacktestResult(
        period_returns=period_returns,
        portfolio_values=portfolio_values,
        weights=weights_df,
        config=cfg,
    )
