"""Portfolio asset allocation strategies, constraints, and strategy registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import DEFAULT_MAX_WEIGHT, RISK_FREE_RATE


@dataclass(frozen=True)
class StrategyConfig:
    """Parameters governing portfolio optimization strategies."""

    max_weight: float = DEFAULT_MAX_WEIGHT
    risk_free_rate: float = RISK_FREE_RATE


# Standardized strategy callable interface
StrategyCallable = Callable[[pd.Series, pd.DataFrame, StrategyConfig], pd.Series]


# ============================================================================
# CONSTRAINT & PROJECTION HELPERS
# ============================================================================


def project_to_capped_simplex(
    weights: np.ndarray, max_weight: float, tol: float = 1e-12, max_iter: int = 200
) -> np.ndarray:
    """Project raw unconstrained weights onto the capped simplex: {w | 0 <= w <= max_weight, sum(w) = 1}."""
    n = len(weights)
    if n * max_weight < 1.0:
        raise ValueError(f"Infeasible cap: {n} assets with max_weight={max_weight:.4f} sum to < 1.0")

    lo = weights.min() - max_weight
    hi = weights.max()
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        projected = np.clip(weights - mid, 0.0, max_weight)
        total = projected.sum()
        if abs(total - 1.0) <= tol:
            return projected
        if total > 1.0:
            lo = mid
        else:
            hi = mid

    return np.clip(weights - (lo + hi) / 2.0, 0.0, max_weight)


def validate_projected_weights(
    weights: np.ndarray, max_weight: float, strategy_name: str, tol: float = 1e-6
) -> None:
    """Ensure weights are finite, non-negative, respect the cap, and sum to 1."""
    if not np.isfinite(weights).all():
        raise ValueError(f"{strategy_name} produced non-finite weights (NaN or Inf)")
    if (weights < -tol).any():
        raise ValueError(f"{strategy_name} produced negative weights: min={weights.min():.6f}")
    if (weights > max_weight + tol).any():
        raise ValueError(f"{strategy_name} violated position cap {max_weight:.2%}: max={weights.max():.6f}")
    if not np.isclose(weights.sum(), 1.0, atol=tol, rtol=0.0):
        raise ValueError(f"{strategy_name} weights do not sum to 1.0: sum={weights.sum():.6f}")


def calculate_portfolio_volatility(weights: pd.Series, covariance: pd.DataFrame) -> float:
    """Annualized portfolio standard deviation given weights and covariance."""
    w = weights.to_numpy()
    cov = covariance.loc[weights.index, weights.index].to_numpy()
    return float(np.sqrt(max(w @ cov @ w, 0.0)))


# ============================================================================
# STRATEGY IMPLEMENTATIONS (ORIGINAL 6 STRATEGIES)
# ============================================================================


def equal_weight(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    config: StrategyConfig | None = None,
) -> pd.Series:
    """Equal-weight (1/N) allocation across all eligible assets."""
    del expected_returns, config
    tickers = covariance.index
    n = len(tickers)
    return pd.Series(1.0 / n, index=tickers)


def max_sharpe(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    config: StrategyConfig | None = None,
) -> pd.Series:
    """Long-only Max Sharpe ratio portfolio solved via SLSQP quadratic optimization."""
    cfg = config or StrategyConfig()
    tickers = expected_returns.index
    n = len(tickers)
    cov = covariance.loc[tickers, tickers].to_numpy()
    mu = expected_returns.to_numpy()

    def negative_sharpe(weights: np.ndarray) -> float:
        port_ret = weights @ mu
        port_vol = np.sqrt(max(weights @ cov @ weights, 1e-12))
        return -(port_ret - cfg.risk_free_rate) / port_vol

    equal_w = np.full(n, 1.0 / n)
    result = minimize(
        negative_sharpe,
        equal_w,
        method="SLSQP",
        bounds=[(0.0, cfg.max_weight)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Max Sharpe optimization failed: {result.message}")

    projected = project_to_capped_simplex(result.x, max_weight=cfg.max_weight)
    validate_projected_weights(projected, cfg.max_weight, "Max Sharpe")
    return pd.Series(projected, index=tickers)


def inverse_volatility(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    config: StrategyConfig | None = None,
) -> pd.Series:
    """Inverse-volatility allocation (w_i proportional to 1/sigma_i)."""
    del expected_returns
    cfg = config or StrategyConfig()
    tickers = covariance.index
    cov = covariance.loc[tickers, tickers].to_numpy()
    volatilities = np.sqrt(np.diag(cov))

    raw_weights = 1.0 / np.maximum(volatilities, 1e-8)
    normalized = raw_weights / raw_weights.sum()
    projected = project_to_capped_simplex(normalized, max_weight=cfg.max_weight)
    validate_projected_weights(projected, cfg.max_weight, "Inverse Volatility")
    return pd.Series(projected, index=tickers)


def minimum_variance(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    config: StrategyConfig | None = None,
) -> pd.Series:
    """Long-only Minimum Variance portfolio minimizing total portfolio risk."""
    del expected_returns
    cfg = config or StrategyConfig()
    tickers = covariance.index
    n = len(tickers)
    cov = covariance.loc[tickers, tickers].to_numpy()

    def portfolio_variance(weights: np.ndarray) -> float:
        return float(weights @ cov @ weights)

    equal_w = np.full(n, 1.0 / n)
    result = minimize(
        portfolio_variance,
        equal_w,
        method="SLSQP",
        bounds=[(0.0, cfg.max_weight)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Minimum Variance optimization failed: {result.message}")

    projected = project_to_capped_simplex(result.x, max_weight=cfg.max_weight)
    validate_projected_weights(projected, cfg.max_weight, "Minimum Variance")
    return pd.Series(projected, index=tickers)


def risk_parity(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    config: StrategyConfig | None = None,
) -> pd.Series:
    """Equal Risk Contribution (Risk Parity) allocation."""
    del expected_returns
    cfg = config or StrategyConfig()
    tickers = covariance.index
    n = len(tickers)
    cov = covariance.loc[tickers, tickers].to_numpy()

    def objective(weights: np.ndarray) -> float:
        port_var = weights @ cov @ weights
        port_vol = np.sqrt(max(port_var, 1e-12))
        marginal_contrib = cov @ weights
        risk_contrib = weights * marginal_contrib / port_vol
        target = port_vol / n
        return float(np.sum((risk_contrib - target) ** 2))

    equal_w = np.full(n, 1.0 / n)
    result = minimize(
        objective,
        equal_w,
        method="SLSQP",
        bounds=[(0.0, cfg.max_weight)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Risk Parity optimization failed: {result.message}")

    projected = project_to_capped_simplex(result.x, max_weight=cfg.max_weight)
    validate_projected_weights(projected, cfg.max_weight, "Risk Parity")
    return pd.Series(projected, index=tickers)


def maximum_diversification(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    config: StrategyConfig | None = None,
) -> pd.Series:
    """Maximum Diversification portfolio maximizing the Diversification Ratio."""
    del expected_returns
    cfg = config or StrategyConfig()
    tickers = covariance.index
    n = len(tickers)
    cov = covariance.loc[tickers, tickers].to_numpy()
    vol = np.sqrt(np.diag(cov))

    def negative_dr(weights: np.ndarray) -> float:
        port_vol = np.sqrt(max(weights @ cov @ weights, 1e-12))
        weighted_vol = float(weights @ vol)
        return -weighted_vol / port_vol

    equal_w = np.full(n, 1.0 / n)
    result = minimize(
        negative_dr,
        equal_w,
        method="SLSQP",
        bounds=[(0.0, cfg.max_weight)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Maximum Diversification optimization failed: {result.message}")

    projected = project_to_capped_simplex(result.x, max_weight=cfg.max_weight)
    validate_projected_weights(projected, cfg.max_weight, "Maximum Diversification")
    return pd.Series(projected, index=tickers)


# ============================================================================
# STRATEGY REGISTRY (ORIGINAL 6 STRATEGIES)
# ============================================================================

STRATEGIES: dict[str, StrategyCallable] = {
    "Equal Weight": equal_weight,
    "Max Sharpe": max_sharpe,
    "Inverse Volatility": inverse_volatility,
    "Minimum Variance": minimum_variance,
    "Risk Parity": risk_parity,
    "Maximum Diversification": maximum_diversification,
}
