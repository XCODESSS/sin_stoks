"""Walk-forward evaluation of the portfolio strategies from backtest_portfolios.py.

Uses weekly returns (not monthly) so early rebalance years have enough
observations for a stable covariance estimate — see MIN_ESTIMATION_WEEKS
below for why this actually fixes the original shortfall.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from backtest_portfolios import (
    compute_expected_returns,
    inverse_vol_weights,
    load_weekly_returns,
    max_diversification_weights,
    max_sharpe_weights,
    min_variance_weights,
    risk_parity_weights,
    save_to_csv,
)

OUTPUT_DIR = Path("outputs/portfolio_backtest")
REBALANCE_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
WEEKS_PER_YEAR = 52

# 104 weeks (~2 years) of observations — a genuine "enough data for
# Ledoit-Wolf to be meaningful" floor, not a literal conversion of the old
# 36-month threshold. Converting 36 months into weeks (~156) would just
# reproduce the same calendar-time shortfall: the 2020 rebalance only has
# ~139 weeks of history before it either way.
MIN_ESTIMATION_WEEKS = 104

STARTING_VALUE = 10_000  # USD notional

Strategy = Callable[[pd.Series, pd.DataFrame], pd.Series]


def _ignore_expected_returns(strategy_fn: Callable[[pd.DataFrame], pd.Series]) -> Strategy:
    """Adapt a covariance-only strategy to the uniform (expected_returns, covariance) signature."""
    def adapted(expected_returns: pd.Series, covariance: pd.DataFrame) -> pd.Series:
        del expected_returns  # unused by this strategy
        return strategy_fn(covariance)
    return adapted


def equal_weight_weights(covariance: pd.DataFrame) -> pd.Series:
    """Equal weight across every asset in the covariance matrix — no estimation needed."""
    asset_count = len(covariance)
    return pd.Series(1.0 / asset_count, index=covariance.index)


STRATEGIES: dict[str, Strategy] = {
    "Equal Weight": _ignore_expected_returns(equal_weight_weights),
    "Max Sharpe": max_sharpe_weights,
    "Inverse Volatility": _ignore_expected_returns(inverse_vol_weights),
    "Minimum Variance": _ignore_expected_returns(min_variance_weights),
    "Risk Parity": _ignore_expected_returns(risk_parity_weights),
    "Maximum Diversification": _ignore_expected_returns(max_diversification_weights),
}


def load_returns() -> pd.DataFrame:
    """Full weekly log-return history used for both training and testing."""
    return load_weekly_returns()


def get_rebalance_dates() -> list[pd.Timestamp]:
    return [pd.Timestamp(f"{year}-01-01") for year in REBALANCE_YEARS]


def get_training_window(
    returns: pd.DataFrame, rebalance_date: pd.Timestamp, min_periods: int
) -> pd.DataFrame:
    """Expanding window: every observation strictly before the rebalance date."""
    train = returns.loc[returns.index < rebalance_date]
    if len(train) < min_periods:
        raise ValueError(
            f"Training window before {rebalance_date.date()} has only {len(train)} periods; "
            f"needs at least {min_periods}."
        )
    return train


def get_holding_period(returns: pd.DataFrame, rebalance_date: pd.Timestamp) -> pd.DataFrame:
    """The one-year window starting at the rebalance date — the out-of-sample test period."""
    next_rebalance_date = rebalance_date + pd.DateOffset(years=1)
    return returns.loc[(returns.index >= rebalance_date) & (returns.index < next_rebalance_date)]


def build_covariance(train: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf shrinkage covariance, annualized, from a training window of weekly log returns."""
    simple_returns = np.expm1(train)
    shrunk = LedoitWolf().fit(simple_returns.to_numpy())
    return pd.DataFrame(
        shrunk.covariance_ * WEEKS_PER_YEAR,
        index=train.columns,
        columns=train.columns,
    )


def build_expected_returns(train: pd.DataFrame) -> pd.Series:
    """Annualized arithmetic expected return per asset from the training window."""
    simple_returns = np.expm1(train)
    return compute_expected_returns(simple_returns, periods_per_year=WEEKS_PER_YEAR)


def apply_weights(weights: pd.Series, holding_period: pd.DataFrame) -> pd.Series:
    """Buy-and-hold: weights are set once at rebalance and drift with prices — no intra-year rebalancing."""
    simple_returns = np.expm1(holding_period[weights.index])
    asset_growth = (1 + simple_returns).cumprod()
    portfolio_growth = asset_growth @ weights
    portfolio_returns = portfolio_growth.pct_change()
    portfolio_returns.iloc[0] = portfolio_growth.iloc[0] - 1
    return portfolio_returns

def run_backtest(
    returns: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    strategies: dict[str, Strategy],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the walk-forward loop across every rebalance date and strategy.

    Returns:
        weights: rows indexed by (rebalance date, strategy), columns = tickers.
        period_returns: rows indexed by date, columns = strategy names.
    """
    weight_rows: dict[tuple[pd.Timestamp, str], pd.Series] = {}
    return_chunks: dict[str, list[pd.Series]] = {name: [] for name in strategies}

    for rebalance_date in rebalance_dates:
        train = get_training_window(returns, rebalance_date, MIN_ESTIMATION_WEEKS)
        test = get_holding_period(returns, rebalance_date)

        covariance = build_covariance(train)
        expected_returns = build_expected_returns(train)

        for strategy_name, strategy in strategies.items():
            weights = strategy(expected_returns, covariance)
            weight_rows[(rebalance_date, strategy_name)] = weights
            return_chunks[strategy_name].append(apply_weights(weights, test))

    weights = pd.DataFrame(weight_rows).T
    weights.index = weights.index.set_names(["Rebalance Date", "Strategy"])

    period_returns = pd.DataFrame({
        strategy_name: pd.concat(chunks)
        for strategy_name, chunks in return_chunks.items()
    })

    return weights, period_returns


def build_portfolio_values(period_returns: pd.DataFrame, starting_value: float) -> pd.DataFrame:
    """Cumulative portfolio value per strategy, compounding period returns from a common start."""
    return starting_value * (1.0 + period_returns).cumprod()


def save_results(
    weights: pd.DataFrame, period_returns: pd.DataFrame, portfolio_values: pd.DataFrame
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_to_csv(weights, OUTPUT_DIR / "walk_forward_weights.csv")
    save_to_csv(period_returns, OUTPUT_DIR / "walk_forward_returns.csv")
    save_to_csv(portfolio_values, OUTPUT_DIR / "walk_forward_values.csv")


def main() -> None:
    returns = load_returns()
    rebalance_dates = get_rebalance_dates()

    weights, period_returns = run_backtest(returns, rebalance_dates, STRATEGIES)
    portfolio_values = build_portfolio_values(period_returns, STARTING_VALUE)

    save_results(weights, period_returns, portfolio_values)


if __name__ == "__main__":
    main()