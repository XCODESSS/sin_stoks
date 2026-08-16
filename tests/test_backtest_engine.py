"""Unit tests for the walk-forward backtesting engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest_engine import (
    BacktestConfig,
    apply_weights,
    get_holding_period,
    get_training_window,
    run_walk_forward_backtest,
)


def test_two_asset_hand_calculated_growth():
    """Verify holding period return and compounding math against analytical calculations."""
    dates = pd.date_range("2021-01-01", periods=2, freq="W-FRI")
    # Weekly simple returns: A=[+10%, -5%], B=[+2%, +4%]
    simple_rets = pd.DataFrame(
        {"A": [0.10, -0.05], "B": [0.02, 0.04]},
        index=dates,
    )
    # Convert to log returns for input
    log_rets = np.log1p(simple_rets)

    weights = pd.Series({"A": 0.50, "B": 0.50})
    rets = apply_weights(weights, log_rets)

    # Week 0 portfolio growth = 0.5*1.10 + 0.5*1.02 = 1.06 -> return = +6.0%
    assert np.isclose(rets.iloc[0], 0.06)

    # Week 1 asset A value = 1.10 * 0.95 = 1.045
    # Week 1 asset B value = 1.02 * 1.04 = 1.0608
    # Week 1 portfolio value = 0.5*1.045 + 0.5*1.0608 = 1.0529
    # Week 1 return = (1.0529 / 1.06) - 1 = -0.00669811
    expected_w1_ret = (1.0529 / 1.06) - 1.0
    assert np.isclose(rets.iloc[1], expected_w1_ret)


def test_training_and_holding_window_isolation():
    """Training window and holding periods must strictly not overlap."""
    dates = pd.date_range("2018-01-01", "2024-12-31", freq="W-FRI")
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.normal(size=(len(dates), 4)), index=dates, columns=["A", "B", "C", "D"])

    rebalance_date = pd.Timestamp("2022-01-01")
    train = get_training_window(df, rebalance_date, min_periods=52)
    hold = get_holding_period(df, rebalance_date)

    assert (train.index < rebalance_date).all()
    assert (hold.index >= rebalance_date).all()
    assert (hold.index < rebalance_date + pd.DateOffset(years=1)).all()
    assert len(train.index.intersection(hold.index)) == 0


def test_run_walk_forward_backtest_prepends_start_row_and_applies_initial_cost():
    """Walk-forward results keep the start row and charge the initial investment once."""
    dates = pd.date_range("2018-01-05", "2020-12-25", freq="W-FRI")
    returns = pd.DataFrame(
        {
            "A": np.log1p(np.full(len(dates), 0.01)),
            "B": np.log1p(np.full(len(dates), 0.02)),
        },
        index=dates,
    )
    spy_dates = dates[dates >= pd.Timestamp("2020-01-01")]
    spy_returns = pd.Series(np.log1p(np.full(len(spy_dates), 0.03)), index=spy_dates, name="SPY")

    def strategy_a(expected_returns: pd.Series, covariance: pd.DataFrame, config) -> pd.Series:
        return pd.Series([0.6, 0.4], index=covariance.index)

    def strategy_b(expected_returns: pd.Series, covariance: pd.DataFrame, config) -> pd.Series:
        return pd.Series([0.5, 0.5], index=covariance.index)

    result = run_walk_forward_backtest(
        returns,
        spy_returns,
        {"Strategy A": strategy_a, "Strategy B": strategy_b},
        BacktestConfig(rebalance_years=[2020]),
    )

    assert isinstance(result.weights.index, pd.MultiIndex)
    assert result.weights.index.names == ["Rebalance Date", "Strategy"]
    assert list(result.weights.index.get_level_values("Strategy")) == ["Strategy A", "Strategy B"]

    first_period = result.period_returns.index[0]
    assert first_period in spy_returns.index
    assert np.isclose(result.period_returns.loc[first_period, "SPY"], 0.03)
    assert (result.turnover.loc[result.turnover["Rebalance Date"] == pd.Timestamp("2020-01-01"), "Turnover"] == 1.0).all()

    gross_first_return_a = 0.6 * 0.01 + 0.4 * 0.02
    expected_net_first_return_a = gross_first_return_a - (10.0 / 10_000.0)
    assert np.isclose(result.period_returns.loc[first_period, "Strategy A"], expected_net_first_return_a)
    assert result.portfolio_values.index[0] == pd.Timestamp("2020-01-01")


def test_run_walk_forward_backtest_rejects_missing_spy_dates():
    """Benchmark gaps must fail instead of being converted into artificial zero returns."""
    dates = pd.date_range("2018-01-05", "2020-12-25", freq="W-FRI")
    returns = pd.DataFrame(
        {
            "A": np.log1p(np.full(len(dates), 0.01)),
            "B": np.log1p(np.full(len(dates), 0.02)),
        },
        index=dates,
    )
    spy_dates = dates[dates >= pd.Timestamp("2020-01-10")]
    spy_returns = pd.Series(np.log1p(np.full(len(spy_dates), 0.03)), index=spy_dates, name="SPY")

    def equal_strategy(expected_returns: pd.Series, covariance: pd.DataFrame, config) -> pd.Series:
        return pd.Series([0.5, 0.5], index=covariance.index)

    with pytest.raises(ValueError, match="SPY benchmark returns are missing"):
        run_walk_forward_backtest(
            returns,
            spy_returns,
            {"Equal": equal_strategy},
            BacktestConfig(rebalance_years=[2020]),
        )
