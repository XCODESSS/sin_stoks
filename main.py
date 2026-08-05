"""Download calendar-year returns for the behavioral-sector stock universe."""

from __future__ import annotations

import pandas as pd
import numpy as np
import yfinance as yf


COMPANIES = {
    "DEO": ("Diageo", "Alcohol"),
    "BUD": ("Anheuser-Busch InBev", "Alcohol"),
    "STZ": ("Constellation Brands", "Alcohol"),
    "BF-B": ("Brown-Forman", "Alcohol"),
    "PRMBF": ("Pernod Ricard", "Alcohol"),
    "MNST": ("Monster Beverage", "Energy Drinks"),
    "CELH": ("Celsius Holdings", "Energy Drinks"),
    "KDP": ("Keurig Dr Pepper", "Energy Drinks"),
    "PEP": ("PepsiCo", "Energy Drinks"),
    "KO": ("Coca-Cola", "Energy Drinks"),
    "META": ("Meta Platforms", "Social Media"),
    "PINS": ("Pinterest", "Social Media"),
    "SNAP": ("Snap Inc.", "Social Media"),
    "RDDT": ("Reddit", "Social Media"),
    "BIDU": ("Baidu", "Social Media"),
    "PM": ("Philip Morris", "Tobacco & Nicotine"),
    "BTI": ("British American Tobacco", "Tobacco & Nicotine"),
    "MO": ("Altria", "Tobacco & Nicotine"),
    "UVV": ("Universal Corporation", "Tobacco & Nicotine"),
    "TPB": ("Turning Point Brands", "Tobacco & Nicotine"),
    "MTCH": ("Match Group", "Dating Apps"),
    "BMBL": ("Bumble", "Dating Apps"),
    "LOV": ("Spark Networks", "Dating Apps"),
    "PSM.DE": ("ProSiebenSat.1 Media", "Dating Apps"),
    "MTY": ("MTY Food Group", "Dating Apps"),
}

# The user-facing universe is retained as supplied. These replacements use the
# current exchange symbols for companies whose original symbols are unavailable.
SOURCE_TICKERS = {"PRMBF": "RI.PA", "MTY": "MTY.TO"}
START_DATE = "2020-01-01"
END_DATE = "2026-01-01"
YEARS = list(range(2020, 2026))


def extract_close_prices(data: pd.DataFrame, source_ticker: str) -> pd.Series:
    """Return a one-dimensional adjusted closing-price series."""
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close[source_ticker] if source_ticker in close.columns else close.iloc[:, 0]
    return close.dropna()


def calculate_annual_returns(close: pd.Series) -> pd.Series:
    """Calculate calendar-year returns, using the first available 2020 price."""
    year_end_prices = close.groupby(close.index.year).last()
    annual_returns = year_end_prices.pct_change()
    first_year = int(close.index.year.min())
    annual_returns.loc[first_year] = (
        year_end_prices.loc[first_year] / close.loc[close.index.year == first_year].iloc[0] - 1
    )
    total_return = (year_end_prices.iloc[-1] / close.iloc[0]) - 1
    print(f"Total return from {close.index[0].date()} to {close.index[-1].date()}: {total_return:.2%}")
    return annual_returns.mul(100)


def calculate_total_returns(annual_returns: pd.DataFrame) -> tuple[pd.Series, float]:
    """Calculate per-company and combined total returns from calendar-year returns."""
    per_company_total_return = annual_returns.apply(
        lambda row: (1 + row.dropna().div(100)).prod() - 1 if row.notna().any() else np.nan,
        axis=1,
    )
    combined_yearly_returns = annual_returns.mean(axis=0, skipna=True).div(100)
    combined_total_return = (1 + combined_yearly_returns).prod() - 1
    combined_portfolio_return = (1 + combined_yearly_returns).prod() - 1
    return combined_portfolio_return, combined_total_return
    return per_company_total_return, float(combined_total_return)


def main() -> None:
    returns_by_ticker: dict[str, pd.Series] = {}

    for ticker in COMPANIES:
        source_ticker = SOURCE_TICKERS.get(ticker, ticker)
        print(f"Processing {ticker} ({source_ticker})...")
        try:
            data = yf.download(
                source_ticker,
                start=START_DATE,
                end=END_DATE,
                auto_adjust=True,
                progress=False,
            )
            if data.empty:
                print(f"No data available for {ticker}.")
                continue
            returns_by_ticker[ticker] = calculate_annual_returns(
                extract_close_prices(data, source_ticker)
            )
        except Exception as error:
            print(f"{ticker}: {error}")

    annual_returns = pd.DataFrame(index=COMPANIES.keys(), columns=YEARS, dtype=float)
    for ticker, returns in returns_by_ticker.items():
        annual_returns.loc[ticker, returns.index.intersection(YEARS)] = returns.loc[
            returns.index.intersection(YEARS)
        ].values

    per_company_total_returns, combined_total_return = calculate_total_returns(annual_returns)

    annual_returns.index.name = "Ticker"
    annual_returns.insert(0, "Company", [COMPANIES[ticker][0] for ticker in annual_returns.index])
    annual_returns.insert(1, "Sector", [COMPANIES[ticker][1] for ticker in annual_returns.index])
    annual_returns = annual_returns.round(2)

    total_returns = pd.DataFrame(
        {
            "Ticker": list(COMPANIES.keys()),
            "Company": [COMPANIES[ticker][0] for ticker in COMPANIES],
            "Sector": [COMPANIES[ticker][1] for ticker in COMPANIES],
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
                        "Company": "All Stocks Combined",
                        "Sector": "Portfolio",
                        "Total Return (%)": round(combined_total_return * 100, 2),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    print("\nAnnual Returns (%)")
    print(annual_returns)
    print("\nTotal Returns (%)")
    print(total_returns)
    print(f"\nCombined Total Return: {combined_total_return:.2%}")
    annual_returns.to_csv("annual_returns_25_companies.csv")
    total_returns.to_csv("total_returns_25_companies.csv", index=False)
    print("\nSaved to annual_returns_25_companies.csv")
    print("Saved to total_returns_25_companies.csv")


if __name__ == "__main__":
    main()
