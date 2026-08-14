"""Compatibility wrapper for legacy single-period portfolio allocation callers.

DEPRECATION NOTICE:
Direct use of backtest_portfolios.py is deprecated.
Use portfolio_strategies.py for allocation functions and backtest_engine.py for walk-forward mechanics.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from backtest_engine import build_covariance, build_expected_returns
from config import COVARIANCE_END, COVARIANCE_START, DATA_DIR, DEFAULT_MAX_WEIGHT
from portfolio_strategies import (
    calculate_portfolio_volatility,
    hierarchical_risk_parity,
    inverse_volatility,
    max_sharpe,
    maximum_diversification,
    minimum_variance,
    project_to_capped_simplex,
    risk_parity,
    validate_projected_weights,
)

MAX_WEIGHT = DEFAULT_MAX_WEIGHT

warnings.warn(
    "backtest_portfolios.py is deprecated. Use run_backtest.py and report.py instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Aliases for backward compatibility
_project_to_capped_simplex = project_to_capped_simplex
_validate_projected_weights = validate_projected_weights
portfolio_volatility = calculate_portfolio_volatility
max_sharpe_weights = max_sharpe
inverse_vol_weights = inverse_volatility
min_variance_weights = minimum_variance
risk_parity_weights = risk_parity
max_diversification_weights = maximum_diversification
hrp_weights = hierarchical_risk_parity


def print_strategy_weights(strategy_name: str, weights: pd.Series, covariance: pd.DataFrame) -> None:
    """Print a strategy's weights (sorted, rounded) and its portfolio volatility."""
    print(f"\n{strategy_name} weights:")
    print(weights.sort_values(ascending=False).round(4))
    print(f"{strategy_name} portfolio volatility: {portfolio_volatility(weights, covariance):.2%}")


def print_cap_hits(strategy_name: str, weights: pd.Series, max_weight: float, tol: float = 1e-4) -> None:
    at_cap = weights[weights >= max_weight - tol]
    if at_cap.empty:
        return
    print(f"\n{strategy_name} hit the {max_weight:.0%} cap: {list(at_cap.index)}")


def compute_expected_returns(rets: pd.DataFrame, periods: int = 12) -> pd.Series:
    return rets.mean() * periods


def load_monthly_returns() -> pd.DataFrame:
    """Load monthly return history restricted to the covariance estimation window."""
    returns = pd.read_csv(DATA_DIR / "monthly_returns.csv", index_col=0)
    returns.index = pd.to_datetime(returns.index)
    window = returns.loc[(returns.index >= COVARIANCE_START) & (returns.index <= COVARIANCE_END)]
    return window.fillna(0.0)


def load_weekly_returns() -> pd.DataFrame:
    """Load weekly log-return history."""
    returns = pd.read_csv(DATA_DIR / "weekly_returns.csv", index_col=0)
    raw_index = returns.index.astype(str)
    if raw_index.str.contains("/").any():
        returns.index = pd.to_datetime(raw_index.str.split("/").str[-1])
    else:
        returns.index = pd.to_datetime(raw_index)
    returns = returns.loc[(returns.index >= COVARIANCE_START) & (returns.index <= COVARIANCE_END)]
    return returns.fillna(0.0)


def save_to_csv(data: pd.DataFrame | pd.Series, path: Path, header: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if header is not None:
        data.to_csv(path, header=header)
    else:
        data.to_csv(path)
    print(f"Saved -> {path}")


def main() -> None:
    monthly_rets = load_monthly_returns()
    cov = build_covariance(monthly_rets, periods_per_year=12)
    mu = build_expected_returns(monthly_rets, periods_per_year=12)

    save_to_csv(cov, DATA_DIR / "covariance_matrix.csv")
    save_to_csv(mu, DATA_DIR / "expected_returns.csv", header=["Expected Annual Return"])
    print("Legacy covariance and expected return artifacts refreshed.")


if __name__ == "__main__":
    main()
