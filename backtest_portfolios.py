"""Build and sanity-check the monthly-return covariance matrix.

Reads data/monthly_returns.csv, restricts to the 2017-2025 window (Snap has
no 2016 history), and produces a covariance matrix for Max Sharpe.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

DATA_DIR = Path("data")
COVARIANCE_START = "2017-04-01"  # Snap has no 2016 history
MAX_WEIGHT = 0.25
RISK_FREE_RATE = 0.04  # annual, USD — approx short-term T-bill proxy; change here if you want to source it live later


def load_monthly_returns() -> pd.DataFrame:
    returns = pd.read_csv(DATA_DIR / "monthly_returns.csv", index_col=0, parse_dates=True)
    return returns.loc[(returns.index >= COVARIANCE_START) & (returns.index <= "2025-12-31")]


def sanity_check(covariance: pd.DataFrame, returns: pd.DataFrame) -> None:
    eigenvalues = np.linalg.eigvalsh(covariance.to_numpy())
    smallest_eigenvalue = eigenvalues.min()
    if smallest_eigenvalue <= 0:
        raise ValueError(
            f"Covariance matrix is not positive definite: smallest eigenvalue = {smallest_eigenvalue:.6f}"
        )

    condition_number = eigenvalues.max() / eigenvalues.min()
    print(f"Condition Number: {condition_number:.2f}")
    print(f"Observations used: {len(returns)} months ({returns.index.min().date()} to {returns.index.max().date()})")
    print(f"Assets: {covariance.shape[0]}")
    print(f"Smallest eigenvalue: {smallest_eigenvalue:.6f}  (must be > 0 for a valid covariance matrix)")

    correlation = returns.corr()
    correlation.to_csv(DATA_DIR / "correlation_matrix.csv")
    print(f"Saved -> {DATA_DIR / 'correlation_matrix.csv'}")
    pairs = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool)).stack()
    print(f"\nAverage correlation : {pairs.mean():.3f}")
    print(f"Median correlation  : {pairs.median():.3f}")
    print(f"Maximum correlation : {pairs.max():.3f}")
    print(f"Minimum correlation : {pairs.min():.3f}")
    print("\nHighest correlated pairs:")
    print(pairs.sort_values(ascending=False).head(5))
    print("\nLowest (most diversifying) pairs:")
    print(pairs.sort_values().head(5))
    annualized_vol = np.diag(covariance) ** 0.5

    volatility = pd.Series(
        annualized_vol,
        index=covariance.index,
        name="Annualized Volatility",
    )
    volatility.to_csv(DATA_DIR / "asset_volatility.csv")
    print("\nMost volatile (annualized):")
    print(pd.Series(annualized_vol, index=covariance.index).sort_values(ascending=False).head(5))

def compute_expected_returns(monthly_returns: pd.DataFrame) -> pd.Series:
    """Annualized arithmetic expected return per asset from monthly simple returns."""
    mean_monthly_simple_return = monthly_returns.mean()
    return mean_monthly_simple_return * 12


def _project_to_capped_simplex(weights: np.ndarray, max_weight: float, tol: float = 1e-12) -> np.ndarray:
    """Project weights onto {w | w>=0, w<=max_weight, sum(w)=1}."""
    lo = weights.min() - max_weight
    hi = weights.max()
    for _ in range(200):
        mid = (lo + hi) / 2
        projected = np.clip(weights - mid, 0.0, max_weight)
        total = projected.sum()
        if abs(total - 1.0) <= tol:
            return projected
        if total > 1.0:
            lo = mid
        else:
            hi = mid

    return np.clip(weights - (lo + hi) / 2, 0.0, max_weight)


def max_sharpe_weights(
    expected_returns: pd.Series, covariance: pd.DataFrame, max_weight: float = MAX_WEIGHT
) -> pd.Series:
    """Long-only Max Sharpe weights, capped at max_weight per asset."""
    tickers = expected_returns.index
    n = len(tickers)
    if n * max_weight < 1:
        raise ValueError(f"Infeasible cap: n * max_weight = {n * max_weight:.4f} < 1")

    cov = covariance.loc[tickers, tickers].to_numpy()
    mu = expected_returns.to_numpy()

    def negative_sharpe(weights: np.ndarray) -> float:
        portfolio_return = weights @ mu
        portfolio_vol = np.sqrt(max(weights @ cov @ weights, 1e-12))
        return -(portfolio_return - RISK_FREE_RATE) / portfolio_vol

    equal_weight = np.full(n, 1 / n)
    result = minimize(
        negative_sharpe,
        equal_weight,
        method="SLSQP",
        bounds=[(0, max_weight)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Max Sharpe optimization failed: {result.message}")

    projected = _project_to_capped_simplex(result.x, max_weight=max_weight)
    if (
        (projected < -1e-8).any()
        or (projected > max_weight + 1e-8).any()
        or not np.isclose(projected.sum(), 1.0, atol=1e-8)
    ):
        raise RuntimeError("Projected Max Sharpe weights are infeasible under capped-simplex constraints")

    return pd.Series(np.clip(projected, 0.0, max_weight), index=tickers)

def main() -> None:
    monthly_returns = load_monthly_returns()
    expected_months = pd.date_range(start=COVARIANCE_START, end="2025-12-31", freq="ME")
    duplicate_months = monthly_returns.index[monthly_returns.index.duplicated()].unique()
    unexpected_months = monthly_returns.index.difference(expected_months)
    missing_months = expected_months.difference(monthly_returns.index.unique())
    if not duplicate_months.empty or not unexpected_months.empty or not missing_months.empty:
        details: list[str] = []
        if not duplicate_months.empty:
            details.append(f"Duplicate month-end rows: {list(duplicate_months.strftime('%Y-%m'))}")
        if not unexpected_months.empty:
            details.append(f"Unexpected month-end rows: {list(unexpected_months.strftime('%Y-%m'))}")
        if not missing_months.empty:
            details.append(f"Missing month-end rows: {list(missing_months.strftime('%Y-%m'))}")
        raise ValueError(
            "Covariance matrix needs complete data for all assets in this window. "
            + "; ".join(details)
        )

    monthly_returns = monthly_returns.reindex(expected_months)
    expected_count = len(expected_months)

    coverage = monthly_returns.notna().sum()
    incomplete = coverage[coverage < expected_count]
    if not incomplete.empty:
        for ticker in incomplete.index:
            missing_dates = monthly_returns.index[monthly_returns[ticker].isna()]
            print(f"{ticker} missing: {list(missing_dates.strftime('%Y-%m'))}")
        raise ValueError(
            f"Covariance matrix needs complete data for all assets in this window. "
            f"Missing months: {list(expected_months[monthly_returns.isna().all(axis=1)].strftime('%Y-%m'))}; "
            f"Missing assets: {dict(incomplete)}"
        )
    print("\nCoverage (%):")
    coverage_pct = coverage / expected_count * 100
    print(coverage_pct.sort_values())
    
    monthly_simple_returns = np.expm1(monthly_returns)
    covariance = monthly_simple_returns.cov() * 12  # Annualize from monthly simple returns
    sanity_check(covariance, monthly_returns)

    covariance.to_csv(DATA_DIR / "covariance_matrix.csv")
    print(f"\nSaved -> {DATA_DIR / 'covariance_matrix.csv'}")
    expected_returns = compute_expected_returns(monthly_simple_returns)

    expected_returns.to_csv(
        DATA_DIR / "expected_returns.csv",
        header=["Expected Annual Return"]
    )

    print(f"Saved -> {DATA_DIR / 'expected_returns.csv'}")

    weights = max_sharpe_weights(expected_returns, covariance)

    print(f"\nMax Sharpe weights (capped at {MAX_WEIGHT:.0%} per stock):")
    print(weights.sort_values(ascending=False).round(4))

    at_cap = weights[weights >= MAX_WEIGHT - 1e-6]
    if not at_cap.empty:
        print(f"\nHit the {MAX_WEIGHT:.0%} cap: {list(at_cap.index)}")

    weights.to_csv(DATA_DIR / "max_sharpe_weights.csv", header=["Weight"])
    print(f"Saved -> {DATA_DIR / 'max_sharpe_weights.csv'}")

if __name__ == "__main__":
    main()