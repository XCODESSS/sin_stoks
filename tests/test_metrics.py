"""Unit tests for quantitative risk and return performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from reporting.metrics import (
    calculate_alpha,
    calculate_beta,
    calculate_cagr,
    calculate_calmar_ratio,
    calculate_drawdown,
    calculate_information_ratio,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_total_return,
    calculate_tracking_error,
    calculate_volatility,
)


def test_total_return_and_cagr():
    """Verify total return and CAGR against known geometric compounding values."""
    dates = pd.date_range("2020-01-01", periods=5, freq="YE")
    # $10,000 doubling to $20,000 over 4 years
    values = pd.Series([10_000.0, 12_000.0, 15_000.0, 18_000.0, 20_000.0], index=dates)

    tot_ret = calculate_total_return(values)
    cagr = calculate_cagr(values)

    assert np.isclose(tot_ret, 1.0)  # +100%
    expected_cagr = (2.0) ** (1.0 / 4.0) - 1.0  # ~18.92%
    assert np.isclose(cagr, expected_cagr, atol=1e-3)


def test_max_drawdown_and_calmar():
    """Verify max drawdown and Calmar ratio on a known underwater path."""
    dates = pd.date_range("2020-01-01", periods=4, freq="YE")
    # Peak at 120, trough at 90 -> -25% drawdown
    values = pd.Series([100.0, 120.0, 90.0, 110.0], index=dates)
    dd_series = calculate_drawdown(values)
    max_dd = calculate_max_drawdown(values)
    calmar = calculate_calmar_ratio(values)

    assert np.isclose(max_dd, -0.25)
    assert np.isclose(dd_series.iloc[1], 0.0)
    assert np.isclose(dd_series.iloc[2], -0.25)
    assert np.isfinite(calmar)


def test_volatility_and_sharpe():
    """Verify annualized volatility and Sharpe ratio scaling."""
    # Constant 1% weekly returns with zero vol
    constant_rets = pd.Series([0.01] * 52)
    vol = calculate_volatility(constant_rets, periods_per_year=52)
    assert np.isclose(vol, 0.0)

    # Weekly returns with known variance
    np.random.seed(42)
    rets = pd.Series(np.random.normal(0.002, 0.02, 260))  # ~5 years weekly
    sharpe = calculate_sharpe_ratio(rets, risk_free_rate=0.04, periods_per_year=52)
    sortino = calculate_sortino_ratio(rets, risk_free_rate=0.04, periods_per_year=52)
    assert np.isfinite(sharpe)
    assert np.isfinite(sortino)


def test_beta_alpha_and_tracking_error():
    """Verify Beta, Alpha, Tracking Error, and Information Ratio calculations."""
    dates = pd.date_range("2021-01-01", periods=104, freq="W-FRI")
    np.random.seed(42)
    bench_rets = pd.Series(np.random.normal(0.002, 0.02, 104), index=dates)

    # Perfect tracking (identical portfolio)
    beta_ident = calculate_beta(bench_rets, bench_rets)
    alpha_ident = calculate_alpha(bench_rets, bench_rets, risk_free_rate=0.04, periods_per_year=52)
    te_ident = calculate_tracking_error(bench_rets, bench_rets, periods_per_year=52)
    assert np.isclose(beta_ident, 1.0)
    assert np.isclose(alpha_ident, 0.0)
    assert np.isclose(te_ident, 0.0)

    # 2x leveraged portfolio with 1% periodic alpha
    port_rets = 2.0 * bench_rets + 0.001
    beta_2x = calculate_beta(port_rets, bench_rets)
    alpha_2x = calculate_alpha(port_rets, bench_rets, risk_free_rate=0.04, periods_per_year=52)
    te_2x = calculate_tracking_error(port_rets, bench_rets, periods_per_year=52)
    ir_2x = calculate_information_ratio(port_rets, bench_rets, periods_per_year=52)

    assert np.isclose(beta_2x, 2.0)
    assert np.isfinite(alpha_2x)
    assert te_2x > 0.0
    assert np.isfinite(ir_2x)
