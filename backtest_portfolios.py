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
MONTHS_PER_YEAR = 12
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

    std = np.sqrt(np.diag(covariance.to_numpy()))
    correlation_matrix = (covariance.to_numpy() / np.outer(std, std)).copy()
    np.fill_diagonal(correlation_matrix, 1.0)
    correlation = pd.DataFrame(
        correlation_matrix,
        index=covariance.index,
        columns=covariance.columns,
    )
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
    # temporary diagnostic — compare log-return vol vs simple-return vol
    # on the exact same underlying data, in the same run
    log_vol_check = returns.std() * np.sqrt(12)
    simple_vol_check = np.expm1(returns).std() * np.sqrt(12)
    comparison = pd.DataFrame({
        "log_return_vol": log_vol_check,
        "simple_return_vol": simple_vol_check,
        "ratio": simple_vol_check / log_vol_check,
    }).sort_values("log_return_vol", ascending=False)
    print("\nLog vs simple volatility, same data, same run:")
    print(comparison.head(10))
    volatility.to_csv(DATA_DIR / "asset_volatility.csv")
    print("\nMost volatile (annualized):")
    print(pd.Series(annualized_vol, index=covariance.index).sort_values(ascending=False).head(5))

def compute_expected_returns(simple_returns: pd.DataFrame, periods_per_year: int = MONTHS_PER_YEAR) -> pd.Series:
    """Annualized arithmetic expected return per asset from simple returns."""
    mean_period_simple_return = simple_returns.mean()
    return mean_period_simple_return * periods_per_year

def load_weekly_returns() -> pd.DataFrame:
    """Weekly log-return history, restricted to the covariance window and fully populated.

    Drops any week where at least one ticker is missing, rather than
    reindexing to a synthetic weekly calendar — weekly bars can drift by a
    day or two across tickers in ways month-end bars don't.
    """
    returns = pd.read_csv(DATA_DIR / "weekly_returns.csv", index_col=0, parse_dates=True)
    returns = returns.loc[(returns.index >= COVARIANCE_START) & (returns.index <= "2025-12-31")]

    complete_returns = returns.dropna(how="any")
    dropped_row_count = len(returns) - len(complete_returns)
    if dropped_row_count:
        print(f"load_weekly_returns: dropped {dropped_row_count} weeks with at least one missing ticker")

    return complete_returns


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


def _validate_projected_weights(
    weights: np.ndarray, max_weight: float, strategy_name: str, tol: float = 1e-8
) -> None:
    if (
        not np.isfinite(weights).all()
        or (weights < -tol).any()
        or (weights > max_weight + tol).any()
        or not np.isclose(weights.sum(), 1.0, atol=tol, rtol=0.0)
    ):
        raise RuntimeError(
            f"Projected {strategy_name} weights are infeasible under capped-simplex constraints"
        )

def portfolio_volatility(weights: pd.Series, covariance: pd.DataFrame) -> float:
    """Annualized portfolio volatility."""
    w = weights.to_numpy()
    cov = covariance.to_numpy()
    return np.sqrt(w @ cov @ w)

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
    _validate_projected_weights(projected, max_weight, "Max Sharpe")

    return pd.Series(np.clip(projected, 0.0, max_weight), index=tickers)

#=============================================================================
# INVERSE VOLATILITY
#=============================================================================
def _asset_volatilities(covariance: pd.DataFrame) -> pd.Series:
    """Individual annualized volatility per asset - sqrt of the covariance matrix diagonal."""
    return pd.Series(np.sqrt(np.diag(covariance)), index=covariance.index)

def inverse_vol_weights(covariance: pd.DataFrame, max_weight: float = MAX_WEIGHT) -> pd.Series:
    """Weight each asset inversely to its volatility. No optimizerm no correlation used."""
    volatilities = _asset_volatilities(covariance)
    if len(volatilities) * max_weight < 1:
        raise ValueError(f"Infeasible cap: n * max_weight = {len(volatilities) * max_weight:.4f} < 1")
    raw_weights = 1 / volatilities
    normalized = raw_weights / raw_weights.sum()
    capped = _project_to_capped_simplex(normalized.to_numpy(), max_weight=max_weight)
    _validate_projected_weights(capped, max_weight, "inverse-volatility")
    return pd.Series(capped, index=covariance.index)
#=============================================================================
# MINIMUM VARIANCE
#=============================================================================
def min_variance_weights(covariance: pd.DataFrame, max_weight: float = MAX_WEIGHT) -> pd.Series:
    """Long-only minimum-variance weights, capped at max_weight per asset."""
    tickers = covariance.index
    n = len(tickers)
    cov = covariance.to_numpy()

    def portfolio_variance(weights: np.ndarray) -> float:
        return weights @ cov @ weights

    equal_weight = np.full(n, 1 / n)
    result = minimize(
        portfolio_variance,
        equal_weight,
        method="SLSQP",
        bounds=[(0, max_weight)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Minimum Variance optimization failed: {result.message}")

    projected = _project_to_capped_simplex(result.x, max_weight=max_weight)
    _validate_projected_weights(projected, max_weight, "Minimum Variance")
    return pd.Series(projected, index=tickers)

#=============================================================================
# RISK PARITY
#=============================================================================
def risk_parity_weights(covariance: pd.DataFrame, max_weight: float = MAX_WEIGHT) -> pd.Series:
    """Long-only Risk Parity portfolio, capped at max_weight per asset."""

    tickers = covariance.index
    n = len(tickers)
    cov = covariance.to_numpy()

    def objective(weights: np.ndarray) -> float:
        portfolio_var = weights @ cov @ weights
        portfolio_vol = np.sqrt(max(portfolio_var, 1e-12))

        marginal = cov @ weights
        risk_contrib = weights * marginal / portfolio_vol

        target = portfolio_vol / n

        return np.sum((risk_contrib - target) ** 2)

    equal_weight = np.full(n, 1 / n)

    result = minimize(
        objective,
        equal_weight,
        method="SLSQP",
        bounds=[(0, max_weight)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    if not result.success:
        raise RuntimeError(f"Risk Parity optimization failed: {result.message}")

    projected = _project_to_capped_simplex(result.x, max_weight=max_weight)
    _validate_projected_weights(projected, max_weight, "Risk Parity")

    return pd.Series(projected, index=tickers)

#=============================================================================
# MAXIMUM DIVERSIFICATION
#=============================================================================
def max_diversification_weights(
    covariance: pd.DataFrame,
    max_weight: float = MAX_WEIGHT,
) -> pd.Series:
    """Long-only Maximum Diversification portfolio, capped at max_weight per asset."""

    tickers = covariance.index
    n = len(tickers)

    cov = covariance.to_numpy()
    vol = np.sqrt(np.diag(cov))

    def negative_diversification_ratio(weights: np.ndarray) -> float:
        portfolio_vol = np.sqrt(max(weights @ cov @ weights, 1e-12))
        diversification_ratio = (weights @ vol) / portfolio_vol
        return -diversification_ratio

    equal_weight = np.full(n, 1 / n)

    result = minimize(
        negative_diversification_ratio,
        equal_weight,
        method="SLSQP",
        bounds=[(0, max_weight)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    if not result.success:
        raise RuntimeError(
            f"Maximum Diversification optimization failed: {result.message}"
        )

    projected = _project_to_capped_simplex(
        result.x,
        max_weight=max_weight,
    )
    _validate_projected_weights(projected, max_weight, "Maximum Diversification")

    return pd.Series(projected, index=tickers)

CAP_TOLERANCE = 1e-6


def validate_month_index(actual_months: pd.DatetimeIndex, expected_months: pd.DatetimeIndex) -> None:
    """Fail fast if the monthly-return index has duplicate, unexpected, or missing month-ends."""
    duplicate_months = actual_months[actual_months.duplicated()].unique()
    unexpected_months = actual_months.difference(expected_months)
    missing_months = expected_months.difference(actual_months.unique())

    if duplicate_months.empty and unexpected_months.empty and missing_months.empty:
        return

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


def validate_full_asset_coverage(monthly_returns: pd.DataFrame) -> None:
    """Fail fast if any ticker is missing observations within the (already-reindexed) window."""
    expected_count = len(monthly_returns.index)
    coverage = monthly_returns.notna().sum()
    incomplete = coverage[coverage < expected_count]

    if incomplete.empty:
        return

    for ticker in incomplete.index:
        missing_dates = monthly_returns.index[monthly_returns[ticker].isna()]
        print(f"{ticker} missing: {list(missing_dates.strftime('%Y-%m'))}")

    fully_missing_months = monthly_returns.index[monthly_returns.isna().all(axis=1)]
    raise ValueError(
        "Covariance matrix needs complete data for all assets in this window. "
        f"Missing months: {list(fully_missing_months.strftime('%Y-%m'))}; "
        f"Missing assets: {dict(incomplete)}"
    )


def print_coverage_report(monthly_returns: pd.DataFrame) -> None:
    expected_count = len(monthly_returns.index)
    coverage_pct = monthly_returns.notna().sum() / expected_count * 100
    print("\nCoverage (%):")
    print(coverage_pct.sort_values())


def load_covariance_window_returns() -> pd.DataFrame:
    """Load monthly returns, validate the window is complete, and reindex to it."""
    monthly_returns = load_monthly_returns()
    expected_months = pd.date_range(start=COVARIANCE_START, end="2025-12-31", freq="ME")

    validate_month_index(monthly_returns.index, expected_months)
    monthly_returns = monthly_returns.reindex(expected_months)
    validate_full_asset_coverage(monthly_returns)

    print_coverage_report(monthly_returns)
    return monthly_returns


def build_covariance_and_expected_returns(monthly_returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    monthly_simple_returns = np.expm1(monthly_returns)
    covariance = monthly_simple_returns.cov() * MONTHS_PER_YEAR  # annualized covariance
    expected_returns = compute_expected_returns(monthly_simple_returns)
    return covariance, expected_returns


def save_to_csv(data: pd.DataFrame | pd.Series, path: Path, header: list[str] | None = None) -> None:
    data.to_csv(path, header=header) if header is not None else data.to_csv(path)
    print(f"Saved -> {path}")


def print_strategy_weights(strategy_name: str, weights: pd.Series, covariance: pd.DataFrame) -> None:
    """Print a strategy's weights (sorted, rounded) and its portfolio volatility."""
    print(f"\n{strategy_name} weights:")
    print(weights.sort_values(ascending=False).round(4))
    print(f"{strategy_name} portfolio volatility: {portfolio_volatility(weights, covariance):.2%}")


def print_cap_hits(strategy_name: str, weights: pd.Series, max_weight: float) -> None:
    at_cap = weights[weights >= max_weight - CAP_TOLERANCE]
    if at_cap.empty:
        return
    print(f"\n{strategy_name} hit the {max_weight:.0%} cap: {list(at_cap.index)}")


def run_strategy(
    strategy_name: str,
    weights: pd.Series,
    covariance: pd.DataFrame,
    save_path: Path,
    max_weight: float | None = None,
) -> None:
    """Print a strategy's results and persist its weights — the shared tail end of every strategy."""
    print_strategy_weights(strategy_name, weights, covariance)
    if max_weight is not None:
        print_cap_hits(strategy_name, weights, max_weight)
    save_to_csv(weights, save_path, header=["Weight"])


def main() -> None:
    monthly_returns = load_covariance_window_returns()
    covariance, expected_returns = build_covariance_and_expected_returns(monthly_returns)
    sanity_check(covariance, monthly_returns)

    save_to_csv(covariance, DATA_DIR / "covariance_matrix.csv")
    save_to_csv(expected_returns, DATA_DIR / "expected_returns.csv", header=["Expected Annual Return"])

    max_sharpe = max_sharpe_weights(expected_returns, covariance)
    run_strategy("Max Sharpe", max_sharpe, covariance, DATA_DIR / "max_sharpe_weights.csv", max_weight=MAX_WEIGHT)

    inverse_vol = inverse_vol_weights(covariance)
    run_strategy("Inverse Volatility", inverse_vol, covariance, DATA_DIR / "inverse_vol_weights.csv", max_weight=MAX_WEIGHT)

    min_var = min_variance_weights(covariance)
    run_strategy("Minimum Variance", min_var, covariance, DATA_DIR / "min_variance_weights.csv", max_weight=MAX_WEIGHT)

    risk_parity = risk_parity_weights(covariance)
    run_strategy("Risk Parity", risk_parity, covariance, DATA_DIR / "risk_parity_weights.csv", max_weight=MAX_WEIGHT)

    max_diversification = max_diversification_weights(covariance)
    run_strategy("Maximum Diversification", max_diversification, covariance, DATA_DIR / "max_diversification_weights.csv", max_weight=MAX_WEIGHT)

if __name__ == "__main__":
    main()
