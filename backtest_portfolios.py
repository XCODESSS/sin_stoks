"""Build and sanity-check the monthly-return covariance matrix.

Reads data/monthly_returns.csv, restricts to the 2017-2025 window (Snap has
no 2016 history), and produces a covariance matrix for Max Sharpe.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
COVARIANCE_START = "2017-04-01"  # Snap has no 2016 history


def load_monthly_returns() -> pd.DataFrame:
    returns = pd.read_csv(DATA_DIR / "monthly_returns.csv", index_col=0, parse_dates=True)
    return returns.loc[returns.index >= COVARIANCE_START]


def sanity_check(covariance: pd.DataFrame, returns: pd.DataFrame) -> None:
    eigenvalues = np.linalg.eigvalsh(covariance.to_numpy())
    condition_number = eigenvalues.max() / eigenvalues.min()
    print(f"Condition Number: {condition_number:.2f}")
    print(f"Observations used: {len(returns)} months ({returns.index.min().date()} to {returns.index.max().date()})")
    print(f"Assets: {covariance.shape[0]}")
    print(f"Smallest eigenvalue: {eigenvalues.min():.6f}  (must be > 0 for a valid covariance matrix)")

    correlation = returns.corr()
    pairs = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool)).stack()
    print(f"\nAverage correlation : {pairs.mean():.3f}")
    print(f"Median correlation  : {pairs.median():.3f}")
    print(f"Maximum correlation : {pairs.max():.3f}")
    print(f"Minimum correlation : {pairs.min():.3f}")
    print("\nHighest correlated pairs:")
    print(pairs.sort_values(ascending=False).head(5))
    print("\nLowest (most diversifying) pairs:")
    print(pairs.sort_values().head(5))

    annualized_vol = (np.diag(covariance) * 12) ** 0.5
    print("\nMost volatile (annualized):")
    print(pd.Series(annualized_vol, index=covariance.index).sort_values(ascending=False).head(5))


def main() -> None:
    monthly_returns = load_monthly_returns()

    coverage = monthly_returns.notna().sum()
    incomplete = coverage[coverage < len(monthly_returns)]
    if not incomplete.empty:
        for ticker in incomplete.index:
            missing_dates = monthly_returns.index[monthly_returns[ticker].isna()]
            print(f"{ticker} missing: {list(missing_dates.strftime('%Y-%m'))}")
        raise ValueError(f"Covariance matrix needs complete data for all assets in this window. Missing: {dict(incomplete)}")
    covariance = monthly_returns.cov()
    sanity_check(covariance, monthly_returns)

    covariance.to_csv(DATA_DIR / "covariance_matrix.csv")
    print(f"\nSaved -> {DATA_DIR / 'covariance_matrix.csv'}")


if __name__ == "__main__":
    main()