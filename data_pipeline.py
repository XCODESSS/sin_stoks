"""Market data acquisition, cleaning, outlier filtering, and artifact persistence."""

from __future__ import annotations

import hashlib
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    DATA_DIR,
    END_DATE,
    KDP_OUTLIER_THRESHOLD,
    START_DATE,
    YEARS,
)
from universe import (
    ALL_TICKERS,
    BENCHMARK_TICKER,
    BENCHMARK_TICKERS,
    COMPANIES,
    PORTFOLIO_TICKERS,
)


def extract_close_prices(close_data: pd.DataFrame | pd.Series, ticker: str) -> pd.Series:
    """Return a 1-D adjusted close series for ticker from a batch download."""
    if isinstance(close_data, pd.DataFrame):
        if ticker not in close_data.columns:
            return pd.Series(dtype=float)
        close_series = close_data[ticker]
    else:
        close_series = close_data
    return close_series.dropna()


def calculate_annual_returns(close_series: pd.Series) -> pd.Series:
    """Calendar-year returns (%) from an adjusted-close series."""
    year_end_prices = close_series.groupby(close_series.index.year).last()
    annual_returns = year_end_prices.pct_change()

    first_year = int(close_series.index.year.min())
    first_price = close_series.loc[close_series.index.year == first_year].iloc[0]
    annual_returns.loc[first_year] = (year_end_prices.loc[first_year] / first_price) - 1.0

    return annual_returns.mul(100.0)


def calculate_total_returns(
    annual_returns: pd.DataFrame, benchmark_tickers: set[str] = frozenset()
) -> tuple[pd.Series, float]:
    """Per-company and combined (equal-weight) total returns from calendar-year data."""
    per_company_total_return = annual_returns.apply(
        lambda row: (1.0 + row.dropna().div(100.0)).prod() - 1.0 if row.notna().any() else np.nan,
        axis=1,
    )

    portfolio_only = annual_returns.drop(index=list(benchmark_tickers), errors="ignore")
    combined_yearly_returns = portfolio_only.mean(axis=0, skipna=True).div(100.0)
    combined_total_return = float((1.0 + combined_yearly_returns).prod() - 1.0)

    return per_company_total_return, combined_total_return


def validate_coverage(
    annual_returns: pd.DataFrame,
    benchmark_tickers: set[str],
    expected_count: int | None = None,
) -> pd.DataFrame:
    """Validate and print a per-year coverage report for the portfolio universe."""
    portfolio_returns = annual_returns.drop(index=list(benchmark_tickers), errors="ignore")
    expected_num = expected_count or len(portfolio_returns)

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
                "Expected": expected_num,
                "Actual": actual,
                "Missing": ", ".join(missing) if missing else "—",
            }
        )

    report = pd.DataFrame(rows)
    if (report["Actual"] < report["Expected"]).any():
        warnings.warn(
            "Coverage gap detected — some years have fewer stocks than expected. "
            "See data/coverage_report.csv for details.",
            stacklevel=2,
        )
    return report


def download_monthly_returns() -> pd.DataFrame:
    """Download daily prices, resample to month-end, and compute monthly log returns."""
    print(f"\nDownloading daily prices for {len(PORTFOLIO_TICKERS)} portfolio tickers...")
    data = yf.download(
        PORTFOLIO_TICKERS,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if data.empty:
        raise RuntimeError("Monthly download returned no data.")

    close_data = data.get("Close", data)
    if isinstance(close_data.columns, pd.MultiIndex):
        close_data.columns = close_data.columns.get_level_values(-1)

    monthly_close = close_data.resample("ME").last()
    monthly_returns = np.log(monthly_close / monthly_close.shift(1)).iloc[1:]
    return monthly_returns


def download_weekly_returns() -> pd.DataFrame:
    """Download individual weekly prices, resample to W-FRI, and apply KDP outlier filter."""
    print(f"\nDownloading weekly prices for {len(PORTFOLIO_TICKERS)} portfolio tickers (individually)...")
    close_series: dict[str, pd.Series] = {}
    missing_tickers: list[str] = []

    def download_close_series(ticker: str, max_attempts: int = 3) -> pd.Series | None:
        for attempt in range(1, max_attempts + 1):
            try:
                solo = yf.download(
                    ticker,
                    start=START_DATE,
                    end=END_DATE,
                    interval="1wk",
                    auto_adjust=True,
                    progress=False,
                )
                if not solo.empty:
                    series = solo["Close"].squeeze()
                    if isinstance(series, pd.DataFrame):
                        series = series.iloc[:, 0]
                    return series.dropna()
            except Exception:
                pass

            if attempt < max_attempts:
                time.sleep(2 ** (attempt - 1))

        return None

    for ticker in PORTFOLIO_TICKERS:
        series = download_close_series(ticker)
        if series is None or series.empty:
            missing_tickers.append(ticker)
            continue

        close_series[ticker] = series

    if missing_tickers:
        raise RuntimeError(
            f"Weekly download returned no data for required tickers: {', '.join(missing_tickers)}"
        )

    # Normalize each series to W-FRI period end
    normalized_close: dict[str, pd.Series] = {}
    for ticker, series in close_series.items():
        w_series = series.copy()
        w_series.index = pd.to_datetime(w_series.index).to_period("W-FRI")
        normalized_close[ticker] = w_series.groupby(level=0).last()

    combined_close = pd.DataFrame(normalized_close)
    retained_start = combined_close.index.min()
    retained_end = combined_close.index.max()
    removed_weeks = len(combined_close)
    combined_close = combined_close.dropna(how="any")
    removed_weeks -= len(combined_close)
    if not combined_close.empty:
        retained_start_dt = retained_start.to_timestamp(how="end").normalize()
        retained_end_dt = retained_end.to_timestamp(how="end").normalize()
        print(
            f"Retained intersected weekly range: {retained_start_dt.date()} to {retained_end_dt.date()} "
            f"({removed_weeks} weeks removed from normalized range {retained_start_dt.date()} to {retained_end_dt.date()})."
        )
    weekly_returns = np.log(combined_close / combined_close.shift(1)).iloc[1:]
    weekly_returns.index = weekly_returns.index.to_timestamp(how="end").normalize()

    # Apply documented KDP outlier threshold policy to filter merger artifacts
    outlier_mask = weekly_returns.abs() > KDP_OUTLIER_THRESHOLD
    stacked_returns = weekly_returns.stack(dropna=False)
    outliers = stacked_returns[outlier_mask.stack()]
    if not outliers.empty:
        print(
            f"\n  Filtered {len(outliers)} outlier weekly returns (|log return| > {KDP_OUTLIER_THRESHOLD}):"
        )
        for (date, tkr), val in outliers.items():
            print(f"    {tkr} {date.date()}: {val:.3f}")
        weekly_returns = weekly_returns.mask(outlier_mask, 0.0)

    return weekly_returns


def download_spy_weekly_returns() -> pd.Series:
    """Download weekly log returns for the SPY benchmark."""
    data = yf.download(
        BENCHMARK_TICKER,
        start=START_DATE,
        end=END_DATE,
        interval="1wk",
        auto_adjust=True,
        progress=False,
    )
    if data.empty:
        raise RuntimeError("Weekly download returned no data for SPY benchmark.")

    close_data = data["Close"].squeeze()
    if isinstance(close_data, pd.DataFrame):
        close_data = close_data.iloc[:, 0]

    close_data.index = pd.to_datetime(close_data.index).to_period("W-FRI")
    weekly_close = close_data.groupby(level=0).last()
    spy_returns = np.log(weekly_close / weekly_close.shift(1)).iloc[1:]
    spy_returns.index = spy_returns.index.to_timestamp(how="end").normalize()
    spy_returns.name = BENCHMARK_TICKER
    return spy_returns


def write_csv_outputs_atomically(
    outputs: dict[Path, tuple[pd.DataFrame | pd.Series, dict[str, object]]],
) -> None:
    """Write output DataFrames atomically via temporary files."""
    temp_paths: dict[Path, Path] = {}
    try:
        for target, (frame, csv_kwargs) in outputs.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            temp_path = target.with_name(f"{target.name}.tmp")
            if temp_path.exists():
                temp_path.unlink()
            frame.to_csv(temp_path, **csv_kwargs)
            temp_paths[target] = temp_path

        for target, temp_path in temp_paths.items():
            temp_path.replace(target)
            print(f"Saved -> {target}")
    finally:
        for temp_path in temp_paths.values():
            if temp_path.exists():
                temp_path.unlink()


def generate_data_manifest(data_files: list[Path]) -> None:
    """Generate SHA256 checksum manifest for data reproducibility."""
    manifest: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "start_date": START_DATE,
        "end_date": END_DATE,
        "universe_count": len(PORTFOLIO_TICKERS),
        "portfolio_tickers": PORTFOLIO_TICKERS,
        "benchmark_ticker": BENCHMARK_TICKER,
        "kdp_outlier_threshold": KDP_OUTLIER_THRESHOLD,
        "files": {},
    }

    for path in data_files:
        if path.exists():
            content = path.read_bytes()
            manifest["files"][path.name] = {
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }

    manifest_path = DATA_DIR / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved -> {manifest_path}")


def refresh_market_data() -> None:
    """Execute the full market data pipeline and refresh all artifacts."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading daily prices for {len(ALL_TICKERS)} tickers...")
    data = yf.download(
        ALL_TICKERS,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if data.empty:
        raise RuntimeError("Batch download returned no data.")

    close_data = data.get("Close", data)
    if isinstance(close_data.columns, pd.MultiIndex):
        close_data.columns = close_data.columns.get_level_values(-1)

    # 1. Annual returns
    returns_by_ticker: dict[str, pd.Series] = {}
    for ticker in COMPANIES:
        close_series = extract_close_prices(close_data, ticker)
        if not close_series.empty:
            returns_by_ticker[ticker] = calculate_annual_returns(close_series)

    annual_returns = pd.DataFrame(index=list(COMPANIES.keys()), columns=YEARS, dtype=float)
    for ticker, rets in returns_by_ticker.items():
        valid_years = rets.index.intersection(YEARS)
        annual_returns.loc[ticker, valid_years] = rets.loc[valid_years].values

    # 2. Coverage & total returns
    coverage_report = validate_coverage(annual_returns, BENCHMARK_TICKERS)
    per_company_tot, comb_tot = calculate_total_returns(annual_returns, BENCHMARK_TICKERS)

    n_years = len(YEARS)
    comb_cagr = (1.0 + comb_tot) ** (1.0 / n_years) - 1.0

    annual_returns.index.name = "Ticker"
    annual_returns.insert(0, "Company", [COMPANIES[t][0] for t in annual_returns.index])
    annual_returns.insert(1, "Sector", [COMPANIES[t][1] for t in annual_returns.index])
    annual_returns = annual_returns.round(2)

    total_returns = pd.DataFrame(
        {
            "Ticker": list(COMPANIES.keys()),
            "Company": [COMPANIES[t][0] for t in COMPANIES],
            "Sector": [COMPANIES[t][1] for t in COMPANIES],
            "Total Return (%)": per_company_tot.reindex(COMPANIES.keys()).mul(100.0).round(2).values,
        }
    )
    years_available = annual_returns[YEARS].notna().sum(axis=1)
    total_returns["CAGR (%)"] = total_returns.apply(
        lambda row: (
            (1.0 + row["Total Return (%)"] / 100.0) ** (1.0 / years_available.get(row["Ticker"], n_years))
            - 1.0
        )
        * 100.0
        if pd.notna(row["Total Return (%)"])
        else np.nan,
        axis=1,
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
                        "Total Return (%)": round(comb_tot * 100.0, 2),
                        "CAGR (%)": round(comb_cagr * 100.0, 2),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    # 3. Monthly & Weekly returns
    monthly_returns = download_monthly_returns()
    weekly_returns = download_weekly_returns()
    spy_weekly_returns = download_spy_weekly_returns()

    # 4. Atomic writes
    output_map: dict[Path, tuple[pd.DataFrame | pd.Series, dict[str, object]]] = {
        DATA_DIR / "annual_returns.csv": (annual_returns, {}),
        DATA_DIR / "total_returns.csv": (total_returns, {"index": False}),
        DATA_DIR / "coverage_report.csv": (coverage_report, {"index": False}),
        DATA_DIR / "monthly_returns.csv": (monthly_returns, {}),
        DATA_DIR / "weekly_returns.csv": (weekly_returns, {}),
        DATA_DIR / "spy_weekly_returns.csv": (spy_weekly_returns.to_frame(), {}),
    }
    write_csv_outputs_atomically(output_map)

    generate_data_manifest(list(output_map.keys()))
    print(f"\nEqual-weight cumulative return: {comb_tot:.2%}")
    print(f"Equal-weight CAGR ({n_years} yr):     {comb_cagr:.2%}")
