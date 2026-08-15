"""Quantitative risk and return performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import RISK_FREE_RATE, WEEKS_PER_YEAR


def calculate_total_return(values: pd.Series) -> float:
    """Total cumulative return: (V_end / V_start) - 1."""
    if values.empty or len(values) < 2:
        return np.nan
    return float((values.iloc[-1] / values.iloc[0]) - 1.0)


def calculate_cagr(values: pd.Series) -> float:
    """Compound Annual Growth Rate using exact calendar day span."""
    if values.empty or len(values) < 2:
        return np.nan
    days = (values.index[-1] - values.index[0]).days
    if days <= 0:
        return np.nan
    years = days / 365.25
    return float((values.iloc[-1] / values.iloc[0]) ** (1.0 / years) - 1.0)


def calculate_volatility(returns: pd.Series, periods_per_year: int = WEEKS_PER_YEAR) -> float:
    """Annualized standard deviation of returns."""
    if returns.empty or len(returns) < 2:
        return np.nan
    return float(returns.std() * np.sqrt(periods_per_year))


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
    periods_per_year: int = WEEKS_PER_YEAR,
) -> float:
    """Annualized Sharpe ratio: (Annualized Return - Rf) / Annualized Volatility."""
    if returns.empty or len(returns) < 2:
        return np.nan
    ann_return = returns.mean() * periods_per_year
    ann_vol = calculate_volatility(returns, periods_per_year)
    if ann_vol == 0 or np.isnan(ann_vol):
        return np.nan
    return float((ann_return - risk_free_rate) / ann_vol)


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
    periods_per_year: int = WEEKS_PER_YEAR,
) -> float:
    """Annualized Sortino ratio penalizing only downside deviations below periodic Rf target."""
    if returns.empty or len(returns) < 2:
        return np.nan
    target_period_rf = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    downside = target_period_rf - returns[returns < target_period_rf]
    if downside.empty:
        return np.nan
    downside_dev = np.sqrt(np.mean(np.square(downside))) * np.sqrt(periods_per_year)
    if downside_dev == 0 or np.isnan(downside_dev):
        return np.nan
    ann_return = returns.mean() * periods_per_year
    return float((ann_return - risk_free_rate) / downside_dev)


def calculate_drawdown(values: pd.Series) -> pd.Series:
    """Cumulative underwater drawdown series: (V_t / CumMax(V_t)) - 1."""
    running_max = values.cummax()
    return values / running_max - 1.0


def calculate_max_drawdown(values: pd.Series) -> float:
    """Peak-to-trough maximum drawdown (negative float)."""
    return np.nan if values.empty else float(calculate_drawdown(values).min())


def calculate_calmar_ratio(values: pd.Series) -> float:
    """Calmar ratio: CAGR / |Max Drawdown|."""
    max_dd = abs(calculate_max_drawdown(values))
    if max_dd == 0 or np.isnan(max_dd):
        return np.nan
    return float(calculate_cagr(values) / max_dd)


def calculate_beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate portfolio Beta against the benchmark"""
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if aligned.empty or len(aligned) < 2:
        return np.nan
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    bench_var = np.var(aligned.iloc[:, 1], ddof=1)
    if bench_var == 0 or np.isnan(bench_var):
        return np.nan
    return float(cov[0, 1] / bench_var)


def calculate_alpha(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
    periods_per_year: int = WEEKS_PER_YEAR,
) -> float:
    """Annualized Jensen's Alpha: (R_p - R_f) - Beta * (R_b - R_f)."""
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if aligned.empty or len(aligned) < 2:
        return np.nan
    beta = calculate_beta(aligned.iloc[:, 0], aligned.iloc[:, 1])
    if np.isnan(beta):
        return np.nan
    r_p = aligned.iloc[:, 0].mean() * periods_per_year
    r_b = aligned.iloc[:, 1].mean() * periods_per_year
    return float((r_p - risk_free_rate) - beta * (r_b - risk_free_rate))


def calculate_tracking_error(
    returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: int = WEEKS_PER_YEAR
) -> float:
    """Annualized Tracking Error: StdDev(R_p - R_b) * sqrt(periods)."""
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if aligned.empty or len(aligned) < 2:
        return np.nan
    diff = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    return float(diff.std() * np.sqrt(periods_per_year))


def calculate_information_ratio(
    returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: int = WEEKS_PER_YEAR
) -> float:
    """Information Ratio: Annualized Excess Return / Annualized Tracking Error."""
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if aligned.empty or len(aligned) < 2:
        return np.nan
    te = calculate_tracking_error(aligned.iloc[:, 0], aligned.iloc[:, 1], periods_per_year)
    if te == 0 or np.isnan(te):
        return np.nan
    excess_return = (aligned.iloc[:, 0].mean() - aligned.iloc[:, 1].mean()) * periods_per_year
    return float(excess_return / te)
