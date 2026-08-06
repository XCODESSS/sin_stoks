"""Download calendar-year returns for the behavioral-sector stock universe.

Outputs:
  data/annual_returns.csv   – Calendar-year returns (%) for 30 portfolio tickers + SPY
  data/total_returns.csv    – Cumulative total return per ticker
  data/monthly_returns.csv  – Monthly log-returns for covariance estimation
  data/coverage_report.csv  – Per-year stock-count validation
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ──────────────────────────────────────────────────────────────────────────────
# Universe: 30 portfolio tickers across 6 sectors + 1 benchmark
# ──────────────────────────────────────────────────────────────────────────────
COMPANIES = {
    # Alcohol
    "DEO": ("Diageo", "Alcohol"),
    "BUD": ("Anheuser-Busch InBev", "Alcohol"),
    "STZ": ("Constellation Brands", "Alcohol"),
    "BF-B": ("Brown-Forman", "Alcohol"),
    "PRNDY": ("Pernod Ricard", "Alcohol"),
    # Energy Drinks
    "MNST": ("Monster Beverage", "Energy Drinks"),
    "CELH": ("Celsius Holdings", "Energy Drinks"),
    "KDP": ("Keurig Dr Pepper", "Energy Drinks"),
    "PEP": ("PepsiCo", "Energy Drinks"),
    "KO": ("Coca-Cola", "Energy Drinks"),
    # Social Media
    "META": ("Meta Platforms", "Social Media"),
    "RDDT": ("Reddit", "Social Media"),
    "SNAP": ("Snap Inc.", "Social Media"),
    "TCEHY": ("Tencent", "Social Media"),
    "MSFT": ("Microsoft", "Social Media"),
    # Tobacco & Nicotine
    "PM": ("Philip Morris", "Tobacco & Nicotine"),
    "BTI": ("British American Tobacco", "Tobacco & Nicotine"),
    "MO": ("Altria", "Tobacco & Nicotine"),
    "UVV": ("Universal Corporation", "Tobacco & Nicotine"),
    "TPB": ("Turning Point Brands", "Tobacco & Nicotine"),
    # Gaming
    "NTDOY": ("Nintendo", "Gaming"),
    "EA": ("Electronic Arts", "Gaming"),
    "TTWO": ("Take-Two Interactive", "Gaming"),
    "CCOEY": ("Capcom", "Gaming"),
    "NCBDY": ("Bandai Namco", "Gaming"),
    # Quick Service Restaurants
    "MCD": ("McDonald's", "Quick Service Restaurants"),
    "CMG": ("Chipotle Mexican Grill", "Quick Service Restaurants"),
    "YUM": ("Yum! Brands", "Quick Service Restaurants"),
    "DPZ": ("Domino's Pizza", "Quick Service Restaurants"),
    "QSR": ("Restaurant Brands International", "Quick Service Restaurants"),
    # Benchmark
    "SPY": ("S&P 500 ETF", "Benchmark"),
}

PORTFOLIO_TICKERS = [t for t, (_, s) in COMPANIES.items() if s != "Benchmark"]
BENCHMARK_TICKERS = {t for t, (_, s) in COMPANIES.items() if s == "Benchmark"}
ALL_TICKERS = list(COMPANIES.keys())

START_DATE = "2020-01-01"
END_DATE = "2026-01-01"
YEARS = list(range(2020, 2026))

DATA_DIR = Path("data")

# ──────────────────────────────────────────────────────────────────────────────
# Helper: extract a single ticker's close series from a batch download
# ──────────────────────────────────────────────────────────────────────────────


def extract_close_prices(close_data: pd.DataFrame | pd.Series, ticker: str) -> pd.Series:
    """Return a 1-D adjusted close series for *ticker* from a batch download."""
    if isinstance(close_data, pd.DataFrame):
        if ticker not in close_data.columns:
            return pd.Series(dtype=float)
        close = close_data[ticker]
    else:
        close = close_data
    return close.dropna()


# ──────────────────────────────────────────────────────────────────────────────
# Annual returns
# ──────────────────────────────────────────────────────────────────────────────


def calculate_annual_returns(close: pd.Series) -> pd.Series:
    """Calendar-year returns (%) from an adjusted-close series.

    For the first year the stock appears, the return is measured from its
    first available trading day to the last trading day of that year.
    """
    year_end_prices = close.groupby(close.index.year).last()
    annual_returns = year_end_prices.pct_change()

    first_year = int(close.index.year.min())
    annual_returns.loc[first_year] = (
        year_end_prices.loc[first_year] / close.loc[close.index.year == first_year].iloc[0] - 1
    )

    total_return = (year_end_prices.iloc[-1] / close.iloc[0]) - 1
    print(f"  Total return {close.index[0].date()} → {close.index[-1].date()}: {total_return:.2%}")
    return annual_returns.mul(100)


def calculate_total_returns(
    annual_returns: pd.DataFrame, benchmark_tickers: set[str] = frozenset()
) -> tuple[pd.Series, float]:
    """Per-company and combined (equal-weight) total returns from calendar-year data.

    Benchmark tickers appear in per-company output but are excluded from the
    combined portfolio average.
    """
    per_company_total_return = annual_returns.apply(
        lambda row: (1 + row.dropna().div(100)).prod() - 1 if row.notna().any() else np.nan,
        axis=1,
    )

    portfolio_only = annual_returns.drop(index=list(benchmark_tickers), errors="ignore")
    combined_yearly_returns = portfolio_only.mean(axis=0, skipna=True).div(100)
    combined_total_return = (1 + combined_yearly_returns).prod() - 1

    return per_company_total_return, float(combined_total_return)


# ──────────────────────────────────────────────────────────────────────────────
# Coverage validation
# ──────────────────────────────────────────────────────────────────────────────


def validate_coverage(
    annual_returns: pd.DataFrame,
    benchmark_tickers: set[str],
    expected_count: int | None = None,
) -> pd.DataFrame:
    """Print and return a per-year coverage report for the portfolio universe.

    Shows how many stocks have data each year and lists any that are missing.
    """
    portfolio_returns = annual_returns.drop(index=list(benchmark_tickers), errors="ignore")

    if expected_count is None:
        expected_count = len(portfolio_returns)

    rows = []
    for year in YEARS:
        if year not in portfolio_returns.columns:
            continue
        present = portfolio_returns[year].dropna().index.tolist()
        missing = [t for t in portfolio_returns.index if t not in present]
        actual = len(present)
        rows.append(
            {
                "Year": year,
                "Expected": expected_count,
                "Actual": actual,
                "Missing": ", ".join(missing) if missing else "—",
            }
        )

    report = pd.DataFrame(rows)

    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║               PER-YEAR COVERAGE REPORT                     ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    for _, r in report.iterrows():
        status = "✓" if r["Actual"] == r["Expected"] else "⚠"
        print(
            f"║  {status}  {r['Year']}  |  {r['Actual']:>2}/{r['Expected']:>2}  |  Missing: {r['Missing']:<20s} ║"
        )
    print("╚══════════════════════════════════════════════════════════════╝")

    if any_gap := (report["Actual"] < report["Expected"]).any():
        warnings.warn(
            "Coverage gap detected — some years have fewer stocks than expected. "
            "See coverage_report.csv for details.",
            stacklevel=2,
        )
    else:
        print("  All years have full coverage.")

    return report


# ──────────────────────────────────────────────────────────────────────────────
# Monthly returns for covariance estimation
# ──────────────────────────────────────────────────────────────────────────────


def download_monthly_returns() -> pd.DataFrame:
    """Download monthly close prices and compute simple monthly returns.

    Returns a DataFrame of shape (months, tickers) with monthly percentage
    returns suitable for covariance matrix estimation.
    """
    print(f"\nDownloading monthly prices for {len(PORTFOLIO_TICKERS)} portfolio tickers...")
    data = yf.download(
        PORTFOLIO_TICKERS,
        start=START_DATE,
        end=END_DATE,
        interval="1mo",
        auto_adjust=True,
        progress=True,
        group_by="column",
    )

    if data.empty:
        raise RuntimeError("Monthly download returned no data.")

    close = data["Close"]

    # Simple monthly returns (not log — keeps consistency with annual calc)
    monthly_returns = close.pct_change().iloc[1:]  # drop first NaN row

    # Flatten any MultiIndex columns
    if isinstance(monthly_returns.columns, pd.MultiIndex):
        monthly_returns.columns = monthly_returns.columns.get_level_values(-1)

    count = monthly_returns.notna().sum()
    print(f"\n  Monthly return observations per ticker (min={count.min()}, max={count.max()}):")
    short = count[count < 60]
    if not short.empty:
        print(f"  Short-history tickers: {dict(short)}")

    return monthly_returns


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    # ── 1. Download daily adjusted close prices ──────────────────────────
    print(f"Downloading {len(ALL_TICKERS)} tickers in one batch...")
    data = yf.download(
        ALL_TICKERS,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=True,
        group_by="column",
    )

    if data.empty:
        raise RuntimeError("Batch download returned no data — check connection or tickers.")

    close_data = data["Close"]

    # ── 2. Compute calendar-year returns ─────────────────────────────────
    returns_by_ticker: dict[str, pd.Series] = {}
    for ticker in COMPANIES:
        close = extract_close_prices(close_data, ticker)
        if close.empty:
            print(f"  ✗ No data for {ticker}")
            continue
        try:
            returns_by_ticker[ticker] = calculate_annual_returns(close)
        except Exception as error:
            print(f"  ✗ {ticker}: {error}")

    annual_returns = pd.DataFrame(index=list(COMPANIES.keys()), columns=YEARS, dtype=float)
    for ticker, returns in returns_by_ticker.items():
        annual_returns.loc[ticker, returns.index.intersection(YEARS)] = returns.loc[
            returns.index.intersection(YEARS)
        ].values

    # ── 3. Coverage validation ───────────────────────────────────────────
    coverage_report = validate_coverage(
        annual_returns,
        BENCHMARK_TICKERS,
        expected_count=len(PORTFOLIO_TICKERS),
    )

    # ── 4. Total returns ─────────────────────────────────────────────────
    per_company_total_returns, combined_total_return = calculate_total_returns(
        annual_returns, BENCHMARK_TICKERS
    )

    # Compute CAGR
    n_years = len(YEARS)
    cagr = (1 + combined_total_return) ** (1 / n_years) - 1

    print(f"\n  Equal-weight cumulative return: {combined_total_return:.2%}")
    print(f"  Equal-weight CAGR ({n_years} yr):     {cagr:.2%}")

    # ── 5. Build output DataFrames ───────────────────────────────────────
    annual_returns.index.name = "Ticker"
    annual_returns.insert(0, "Company", [COMPANIES[t][0] for t in annual_returns.index])
    annual_returns.insert(1, "Sector", [COMPANIES[t][1] for t in annual_returns.index])
    annual_returns = annual_returns.round(2)

    total_returns = pd.DataFrame(
        {
            "Ticker": list(COMPANIES.keys()),
            "Company": [COMPANIES[t][0] for t in COMPANIES],
            "Sector": [COMPANIES[t][1] for t in COMPANIES],
            "Total Return (%)": per_company_total_returns.reindex(COMPANIES.keys()).mul(100).round(2).values,
        }
    )
    total_returns = pd.concat(
        [
            total_returns,
            pd.DataFrame(
                [
                    {
                        "Ticker": "PORTFOLIO",
                        "Company": "Equal-Weight Portfolio",
                        "Sector": "Portfolio",
                        "Total Return (%)": round(combined_total_return * 100, 2),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    # ── 6. Save annual + total returns ───────────────────────────────────
    annual_returns.to_csv(DATA_DIR / "annual_returns.csv")
    total_returns.to_csv(DATA_DIR / "total_returns.csv", index=False)
    coverage_report.to_csv(DATA_DIR / "coverage_report.csv", index=False)

    print(f"\n  Saved → {DATA_DIR / 'annual_returns.csv'}")
    print(f"  Saved → {DATA_DIR / 'total_returns.csv'}")
    print(f"  Saved → {DATA_DIR / 'coverage_report.csv'}")

    # ── 7. Download & save monthly returns ───────────────────────────────
    monthly_returns = download_monthly_returns()
    monthly_returns.to_csv(DATA_DIR / "monthly_returns.csv")
    print(f"  Saved → {DATA_DIR / 'monthly_returns.csv'}")

    _extracted_from_main_94("ANNUAL RETURNS (%)")
    print(annual_returns.to_string())

    _extracted_from_main_94("TOTAL RETURNS (%)")
    print(total_returns.to_string(index=False))

    print(f"\n  Combined Total Return: {combined_total_return:.2%}")
    print(f"  CAGR:                  {cagr:.2%}")
    print("=" * 70)


# TODO Rename this here and in `main`
def _extracted_from_main_94(arg0):
    # ── 8. Summary tables ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(arg0)
    print("=" * 70)


if __name__ == "__main__":
    main()
