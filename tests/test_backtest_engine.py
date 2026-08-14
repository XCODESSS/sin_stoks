"""Unit tests for the walk-forward backtesting engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_engine import (
    apply_weights,
    get_holding_period,
    get_training_window,
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
    df = pd.DataFrame(np.random.randn(len(dates), 4), index=dates, columns=["A", "B", "C", "D"])

    rebalance_date = pd.Timestamp("2022-01-01")
    train = get_training_window(df, rebalance_date, min_periods=52)
    hold = get_holding_period(df, rebalance_date)

    assert (train.index < rebalance_date).all()
    assert (hold.index >= rebalance_date).all()
    assert (hold.index < rebalance_date + pd.DateOffset(years=1)).all()
    assert len(train.index.intersection(hold.index)) == 0
