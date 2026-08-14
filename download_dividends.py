"""Download explicit cash dividend histories and unadjusted closing price series.

Outputs:
  data/dividends.csv          - Ex-date, Ticker, Cash Dividend Amount ($/share)
  data/unadjusted_prices.csv  - Daily unadjusted close prices for cost basis calculation
  data/annual_dividends.csv   - Annual total dividend ($/share) per ticker per year
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from config import DATA_DIR, END_DATE, START_DATE
from universe import ALL_TICKERS


def download_dividend_and_price_history() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch unadjusted prices and corporate actions (dividends) for all universe tickers."""
    tickers = sorted(set([*ALL_TICKERS, "SPY"]))
    print(f"Downloading corporate actions & unadjusted prices for {len(tickers)} tickers...")

    # Download batch data with actions=True to get Dividends
    data = yf.download(
        tickers,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        actions=True,
        progress=True,
        group_by="column",
    )

    if data.empty:
        raise RuntimeError("Batch download returned no data.")

    if "Dividends" not in data:
        raise RuntimeError("Dividend data was not returned by Yahoo Finance.")

    # Extract Close prices (unadjusted)
    close_prices = data["Close"]
    if isinstance(close_prices.columns, pd.MultiIndex):
        close_prices.columns = close_prices.columns.get_level_values(-1)

    # Extract Dividends
    dividends = data["Dividends"]
    if isinstance(dividends.columns, pd.MultiIndex):
        dividends.columns = dividends.columns.get_level_values(-1)

    # Clean dividend dataframe: melt into long format (Date, Ticker, Dividend)
    div_records = []
    for ticker in dividends.columns:
        series = dividends[ticker].dropna()
        series = series[series > 0]
        for date, amount in series.items():
            div_records.append(
                {
                    "Date": pd.to_datetime(date).strftime("%Y-%m-%d"),
                    "Year": pd.to_datetime(date).year,
                    "Ticker": ticker,
                    "Dividend": float(amount),
                }
            )

    div_df = pd.DataFrame(div_records)
    if not div_df.empty:
        div_df = div_df.sort_values(by=["Date", "Ticker"]).reset_index(drop=True)
        print("\nDividend coverage:")
        print(div_df.groupby("Ticker").size().sort_index())

    return close_prices, div_df


def compute_annual_dividends_per_share(div_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate total cash dividend paid per share for each ticker per calendar year."""
    if div_df.empty:
        return pd.DataFrame()

    annual_div = div_df.groupby(["Ticker", "Year"])["Dividend"].sum().unstack(level="Year").fillna(0.0)
    return annual_div


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    close_prices, div_df = download_dividend_and_price_history()
    annual_div = compute_annual_dividends_per_share(div_df)

    # Save to CSV
    close_prices.to_csv(DATA_DIR / "unadjusted_prices.csv")
    print(f"Saved -> {DATA_DIR / 'unadjusted_prices.csv'}")

    div_df.to_csv(DATA_DIR / "dividends.csv", index=False)
    print(f"Saved {len(div_df)} dividend events -> {DATA_DIR / 'dividends.csv'}")

    annual_div.to_csv(DATA_DIR / "annual_dividends.csv")
    print(f"Saved annual dividends per share -> {DATA_DIR / 'annual_dividends.csv'}")


if __name__ == "__main__":
    main()
