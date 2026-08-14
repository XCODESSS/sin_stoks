"""Unit tests for portfolio allocation strategies and constraint helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_strategies import (
    STRATEGIES,
    StrategyConfig,
    calculate_portfolio_volatility,
    project_to_capped_simplex,
    validate_projected_weights,
)


@pytest.fixture
def synthetic_market():
    """Create a deterministic synthetic return and covariance fixture for 10 assets."""
    np.random.seed(42)
    n_assets = 10
    tickers = [f"ASSET_{i}" for i in range(n_assets)]

    # Expected returns between 5% and 25%
    mu = pd.Series(np.linspace(0.05, 0.25, n_assets), index=tickers)

    # Positive-definite covariance matrix
    a = np.random.randn(n_assets, n_assets) * 0.05
    cov_mat = a @ a.T + np.diag(np.linspace(0.02, 0.08, n_assets))
    cov = pd.DataFrame(cov_mat, index=tickers, columns=tickers)

    return mu, cov


@pytest.mark.parametrize("strat_name", list(STRATEGIES.keys()))
def test_all_strategies_weight_sum_and_bounds(strat_name, synthetic_market):
    """Every registered strategy must produce weights that sum to 1 and obey the cap."""
    mu, cov = synthetic_market
    cap = 0.25
    cfg = StrategyConfig(max_weight=cap, risk_free_rate=0.04)

    strat_fn = STRATEGIES[strat_name]
    weights = strat_fn(mu, cov, cfg)

    assert isinstance(weights, pd.Series)
    assert len(weights) == len(mu)
    assert np.isclose(weights.sum(), 1.0, atol=1e-5), f"{strat_name} sum={weights.sum()}"
    assert (weights >= -1e-6).all(), f"{strat_name} has negative weights"
    assert (weights <= cap + 1e-5).all(), f"{strat_name} violated cap={cap}"


@pytest.mark.parametrize("cap", [0.10, 0.15, 0.20, 0.25])
def test_capped_simplex_projection(cap):
    """Capped simplex projection must strictly project raw weights onto [0, cap] summing to 1."""
    raw = np.array([0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005])
    projected = project_to_capped_simplex(raw, max_weight=cap)

    assert np.isclose(projected.sum(), 1.0, atol=1e-6)
    assert (projected >= -1e-8).all()
    assert (projected <= cap + 1e-8).all()


def test_infeasible_cap_raises():
    """An infeasible cap (n * cap < 1.0) must raise ValueError."""
    raw = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="Infeasible cap"):
        project_to_capped_simplex(raw, max_weight=0.40)


def test_equal_weight_strategy_respects_registered_cap_failure(synthetic_market):
    """Equal Weight must fail fast when the registered cap makes a feasible allocation impossible."""
    mu, cov = synthetic_market
    with pytest.raises(ValueError, match="Infeasible cap"):
        STRATEGIES["Equal Weight"](mu, cov, StrategyConfig(max_weight=0.05, risk_free_rate=0.04))


def test_portfolio_volatility_calculation(synthetic_market):
    """Portfolio volatility must be positive and match analytical matrix product."""
    mu, cov = synthetic_market
    w = pd.Series(1.0 / len(mu), index=mu.index)
    vol = calculate_portfolio_volatility(w, cov)

    expected = np.sqrt(w.to_numpy() @ cov.to_numpy() @ w.to_numpy())
    assert np.isclose(vol, expected)
    assert vol > 0.0


def test_validate_projected_weights_catches_invalid():
    """Validator must reject non-summing or negative weights."""
    with pytest.raises(ValueError, match=r"do not sum to 1\.0"):
        validate_projected_weights(np.array([0.5, 0.3]), max_weight=0.5, strategy_name="Test")

    with pytest.raises(ValueError, match="violated position cap"):
        validate_projected_weights(np.array([0.6, 0.4]), max_weight=0.5, strategy_name="Test")

    with pytest.raises(ValueError, match="negative weights"):
        validate_projected_weights(np.array([1.2, -0.2]), max_weight=1.5, strategy_name="Test")
