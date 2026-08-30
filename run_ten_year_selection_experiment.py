"""Run an explicitly exploratory 2016-2025 dynamic-universe selection test.

This module does not alter the frozen 2020-2025 experiment.  Earlier years use
only securities with 104 complete pre-rebalance weekly returns and an exact,
point-in-time fundamental record for that rebalance.  Coverage below the frozen
80% gate is recorded rather than hidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from backtest_engine import (
    apply_weights,
    build_covariance,
    build_expected_returns,
    calculate_drifted_weights,
)
from config import (
    DEFAULT_TRANSACTION_COST_BPS,
    KDP_OUTLIER_THRESHOLD,
    SELECTION_LOOKBACK_WEEKS,
    SELECTION_MIN_COVERAGE,
    STARTING_VALUE,
)
from data_pipeline import write_csv_outputs_atomically
from fundamental_data import load_fundamentals
from portfolio_strategies import STRATEGIES, StrategyConfig
from reporting.metrics import (
    calculate_cagr,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_total_return,
    calculate_volatility,
)
from selection_features import SelectionFeatures, build_selection_features
from selection_strategies import SelectionResult, select_density, select_partitioned
from universe import BENCHMARK_TICKER, PORTFOLIO_TICKERS

TEN_YEAR_MARKET_START = "2011-01-01"
TEN_YEAR_MARKET_END = "2026-01-01"
TEN_YEAR_REBALANCE_YEARS = tuple(range(2016, 2026))
TEN_YEAR_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "selection_experiment_10y"
TEN_YEAR_FUNDAMENTALS_PATH = TEN_YEAR_OUTPUT_DIR / "feasibility_fundamentals.csv"

SELECTORS = {
    "Partitioning Selection": select_partitioned,
    "Density Selection": select_density,
}


def _extract_close(downloaded: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if downloaded.empty:
        raise RuntimeError("Yahoo returned no market data")
    if isinstance(downloaded.columns, pd.MultiIndex):
        if "Close" in downloaded.columns.get_level_values(0):
            close = downloaded.xs("Close", axis=1, level=0)
        elif "Close" in downloaded.columns.get_level_values(-1):
            close = downloaded.xs("Close", axis=1, level=-1)
        else:
            raise RuntimeError("Yahoo response has no Close field")
    elif "Close" in downloaded.columns and len(tickers) == 1:
        close = downloaded[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        raise RuntimeError("Unexpected Yahoo market-data schema")
    close.columns = close.columns.astype(str)
    missing = sorted(set(tickers).difference(close.columns))
    if missing:
        raise RuntimeError(f"Yahoo returned no Close history for: {missing}")
    return close.loc[:, tickers].sort_index()


def download_dynamic_weekly_returns(
    output_dir: Path = TEN_YEAR_OUTPUT_DIR,
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Download adjusted prices while preserving pre-listing missingness."""
    asset_path = output_dir / "weekly_returns_dynamic.csv"
    spy_path = output_dir / "spy_weekly_returns_dynamic.csv"
    if asset_path.exists() and spy_path.exists() and not refresh:
        asset_returns = pd.read_csv(asset_path, index_col=0, parse_dates=True)
        spy_frame = pd.read_csv(spy_path, index_col=0, parse_dates=True)
        return asset_returns, spy_frame[BENCHMARK_TICKER]

    tickers = list(PORTFOLIO_TICKERS)
    downloaded = yf.download(
        tickers,
        start=TEN_YEAR_MARKET_START,
        end=TEN_YEAR_MARKET_END,
        interval="1d",
        auto_adjust=True,
        actions=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    close = _extract_close(downloaded, tickers)
    weekly_close = close.resample("W-FRI").last()
    asset_returns = np.log(weekly_close / weekly_close.shift(1)).iloc[1:]
    outliers = asset_returns.abs() > KDP_OUTLIER_THRESHOLD
    asset_returns = asset_returns.mask(outliers, 0.0)
    asset_returns.index.name = "Date"

    spy_download = yf.download(
        BENCHMARK_TICKER,
        start=TEN_YEAR_MARKET_START,
        end=TEN_YEAR_MARKET_END,
        interval="1d",
        auto_adjust=True,
        actions=False,
        progress=False,
    )
    spy_close = _extract_close(spy_download, [BENCHMARK_TICKER])[BENCHMARK_TICKER]
    spy_weekly_close = spy_close.resample("W-FRI").last()
    spy_returns = np.log(spy_weekly_close / spy_weekly_close.shift(1)).iloc[1:]
    spy_returns.name = BENCHMARK_TICKER
    spy_returns.index.name = "Date"

    write_csv_outputs_atomically(
        {
            asset_path: (asset_returns, {}),
            spy_path: (spy_returns.to_frame(), {}),
        }
    )
    return asset_returns, spy_returns


def price_eligible_tickers(
    returns: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    lookback_weeks: int = SELECTION_LOOKBACK_WEEKS,
) -> tuple[pd.DataFrame, list[str]]:
    """Return the exact trailing window and names with complete observations."""
    trailing = returns.loc[returns.index < rebalance_date].tail(lookback_weeks)
    if len(trailing) != lookback_weeks:
        raise ValueError(
            f"{rebalance_date.date()} has {len(trailing)} prior weeks; expected {lookback_weeks}"
        )
    eligible = sorted(trailing.columns[trailing.notna().all(axis=0)].astype(str))
    return trailing, eligible


def exact_fundamental_snapshot(
    fundamentals: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    requested_tickers: list[str],
) -> pd.DataFrame:
    """Use only records constructed specifically for this rebalance date."""
    if "rebalance_date" not in fundamentals:
        raise ValueError("Ten-year fundamentals require a rebalance_date column")
    snapshot = fundamentals.loc[
        (fundamentals["rebalance_date"] == rebalance_date)
        & fundamentals["ticker"].isin(requested_tickers)
        & (fundamentals["available_date"] < rebalance_date)
    ].copy()
    if snapshot["ticker"].duplicated().any():
        raise ValueError(f"Duplicate exact snapshot rows at {rebalance_date.date()}")
    snapshot = snapshot.set_index("ticker").sort_index()
    if len(snapshot) < 12:
        raise ValueError(
            f"Exact fundamental snapshot at {rebalance_date.date()} has {len(snapshot)} assets; needs 12"
        )
    return snapshot


def _equal_weights(tickers: list[str]) -> pd.Series:
    if not tickers:
        raise ValueError("Cannot equal-weight an empty universe")
    return pd.Series(1.0 / len(tickers), index=tickers, dtype=float)


def _dynamic_turnover(old_weights: pd.Series, new_weights: pd.Series) -> float:
    union = old_weights.index.union(new_weights.index)
    old_aligned = old_weights.reindex(union, fill_value=0.0)
    new_aligned = new_weights.reindex(union, fill_value=0.0)
    return float((new_aligned - old_aligned).abs().sum() / 2.0)


def _audit_rows(
    rebalance_date: pd.Timestamp,
    strategy: str,
    snapshot: pd.DataFrame,
    inputs: SelectionFeatures,
    selection: SelectionResult,
) -> list[dict[str, object]]:
    selected = set(selection.selected_tickers)
    rows: list[dict[str, object]] = []
    for ticker in inputs.features.index:
        row = inputs.features.loc[ticker]
        rows.append(
            {
                "rebalance_date": rebalance_date,
                "strategy": strategy,
                "ticker": ticker,
                "selected": ticker in selected,
                "cluster_label": int(selection.labels.loc[ticker]),
                "trailing_pe": snapshot.loc[ticker, "trailing_pe"],
                "market_cap": snapshot.loc[ticker, "market_cap"],
                "available_date": snapshot.loc[ticker, "available_date"],
                "value_rank": row["value_rank"],
                "size_rank": row["size_rank"],
                "sharpe_rank": row["sharpe_rank"],
                "base_score": inputs.base_score.loc[ticker],
                "adjusted_score": selection.adjusted_scores.loc[ticker],
            }
        )
    return rows


def run_dynamic_experiment(
    returns: pd.DataFrame,
    spy_log_returns: pd.Series,
    fundamentals: pd.DataFrame,
    exclusions: tuple[str, ...] = (),
) -> dict[str, pd.DataFrame]:
    """Execute the exploratory ten-year selection experiment."""
    normalized_exclusions = {ticker.upper() for ticker in exclusions}
    experiment_returns = returns.drop(columns=list(normalized_exclusions), errors="ignore")
    return_chunks: dict[str, list[pd.Series]] = {
        **{name: [] for name in STRATEGIES},
        "Eligible Universe Equal Weight": [],
        **{name: [] for name in SELECTORS},
    }
    previous_drifted: dict[str, pd.Series] = {}
    weight_rows: dict[tuple[pd.Timestamp, str], pd.Series] = {}
    turnover_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    cost_fraction = DEFAULT_TRANSACTION_COST_BPS / 10_000.0

    for year in TEN_YEAR_REBALANCE_YEARS:
        rebalance_date = pd.Timestamp(year=year, month=1, day=1)
        trailing, price_eligible = price_eligible_tickers(experiment_returns, rebalance_date)
        snapshot = exact_fundamental_snapshot(fundamentals, rebalance_date, price_eligible)
        fundamental_eligible = list(snapshot.index)
        allocation_train = experiment_returns.loc[
            experiment_returns.index < rebalance_date,
            price_eligible,
        ].dropna(how="any")
        if len(allocation_train) < SELECTION_LOOKBACK_WEEKS:
            raise ValueError(
                f"Allocation window at {rebalance_date.date()} has only "
                f"{len(allocation_train)} complete weeks"
            )
        expected_returns = build_expected_returns(allocation_train)
        covariance = build_covariance(allocation_train)
        strategy_config = StrategyConfig()
        coverage = len(snapshot) / len(PORTFOLIO_TICKERS)
        coverage_rows.append(
            {
                "rebalance_date": rebalance_date,
                "price_eligible_assets": len(price_eligible),
                "fundamental_eligible_assets": len(snapshot),
                "allocation_estimation_weeks": len(allocation_train),
                "coverage_vs_frozen_universe": coverage,
                "frozen_80pct_gate_passed": coverage >= SELECTION_MIN_COVERAGE,
            }
        )

        inputs = build_selection_features(trailing.loc[:, fundamental_eligible], snapshot)
        strategy_weights: dict[str, pd.Series] = {
            name: strategy(expected_returns, covariance, strategy_config)
            for name, strategy in STRATEGIES.items()
        }
        strategy_weights["Eligible Universe Equal Weight"] = _equal_weights(fundamental_eligible)
        for strategy, selector in SELECTORS.items():
            selection = selector(inputs)
            strategy_weights[strategy] = _equal_weights(selection.selected_tickers)
            audit_rows.extend(_audit_rows(rebalance_date, strategy, snapshot, inputs, selection))

        holding = experiment_returns.loc[
            (experiment_returns.index >= rebalance_date)
            & (experiment_returns.index < rebalance_date + pd.DateOffset(years=1))
        ]
        if holding.empty:
            raise ValueError(f"No holding-period returns for {rebalance_date.date()}")

        for strategy, weights in strategy_weights.items():
            required = holding.loc[:, weights.index]
            if required.isna().any().any():
                missing = sorted(required.columns[required.isna().any()].astype(str))
                raise ValueError(
                    f"{strategy} has missing holding returns in {year}: {missing}"
                )
            turnover = (
                _dynamic_turnover(previous_drifted[strategy], weights)
                if strategy in previous_drifted
                else 1.0
            )
            period_returns = apply_weights(weights, holding)
            period_returns.iloc[0] -= turnover * cost_fraction
            return_chunks[strategy].append(period_returns)
            previous_drifted[strategy] = calculate_drifted_weights(weights, holding)
            weight_rows[(rebalance_date, strategy)] = weights
            turnover_rows.append(
                {
                    "Rebalance Date": rebalance_date,
                    "Strategy": strategy,
                    "Turnover": turnover,
                    "Cost": turnover * cost_fraction,
                }
            )

    period_returns = pd.DataFrame(
        {strategy: pd.concat(chunks) for strategy, chunks in return_chunks.items()}
    )
    spy = np.expm1(spy_log_returns.reindex(period_returns.index))
    if spy.isna().any():
        raise ValueError("SPY is missing one or more ten-year holding dates")
    period_returns[BENCHMARK_TICKER] = spy
    values = STARTING_VALUE * (1.0 + period_returns).cumprod()
    start = pd.Timestamp("2016-01-01")
    values = pd.concat([pd.DataFrame(STARTING_VALUE, index=[start], columns=values.columns), values])

    weights = pd.DataFrame(weight_rows).T.fillna(0.0)
    weights.index = weights.index.set_names(["Rebalance Date", "Strategy"])
    return {
        "returns": period_returns,
        "values": values,
        "weights": weights,
        "turnover": pd.DataFrame(turnover_rows),
        "coverage": pd.DataFrame(coverage_rows),
        "audit": pd.DataFrame(audit_rows),
    }


def build_summary(artifacts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    returns = artifacts["returns"]
    values = artifacts["values"]
    turnover = artifacts["turnover"]
    rows: list[dict[str, object]] = []
    for strategy in returns.columns:
        recurring = turnover.loc[turnover["Strategy"] == strategy, "Turnover"]
        rows.append(
            {
                "Strategy": strategy,
                "Total Return": calculate_total_return(values[strategy]),
                "CAGR": calculate_cagr(values[strategy]),
                "Volatility": calculate_volatility(returns[strategy]),
                "Sharpe Ratio": calculate_sharpe_ratio(returns[strategy]),
                "Maximum Drawdown": calculate_max_drawdown(values[strategy]),
                "Average Recurring Turnover": (
                    float(recurring.iloc[1:].mean()) if len(recurring) > 1 else 0.0
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("CAGR", ascending=False).reset_index(drop=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def persist_run(
    artifacts: dict[str, pd.DataFrame],
    output_dir: Path,
    fundamentals_path: Path,
    exclusions: tuple[str, ...],
) -> None:
    run_name = "ex_celh" if exclusions == ("CELH",) else "full"
    run_dir = output_dir / run_name
    summary = build_summary(artifacts)
    annual = (1.0 + artifacts["returns"]).groupby(artifacts["returns"].index.year).prod() - 1.0
    write_csv_outputs_atomically(
        {
            run_dir / "walk_forward_returns.csv": (artifacts["returns"], {}),
            run_dir / "walk_forward_values.csv": (artifacts["values"], {}),
            run_dir / "walk_forward_weights.csv": (artifacts["weights"], {}),
            run_dir / "turnover.csv": (artifacts["turnover"], {"index": False}),
            run_dir / "coverage.csv": (artifacts["coverage"], {"index": False}),
            run_dir / "selection_audit.csv": (artifacts["audit"], {"index": False}),
            run_dir / "summary.csv": (summary, {"index": False}),
            run_dir / "annual_returns.csv": (annual, {}),
        }
    )
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "exploratory-invalid-under-frozen-coverage-gate",
        "market_start": TEN_YEAR_MARKET_START,
        "market_end": TEN_YEAR_MARKET_END,
        "rebalance_years": list(TEN_YEAR_REBALANCE_YEARS),
        "lookback_weeks": SELECTION_LOOKBACK_WEEKS,
        "exclusions": list(exclusions),
        "fundamentals_path": str(fundamentals_path.resolve()),
        "fundamentals_sha256": _sha256(fundamentals_path),
        "minimum_coverage": float(artifacts["coverage"]["coverage_vs_frozen_universe"].min()),
        "all_frozen_coverage_gates_passed": bool(
            artifacts["coverage"]["frozen_80pct_gate_passed"].all()
        ),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the exploratory 2016-2025 selection test")
    parser.add_argument("--fundamentals", type=Path, default=TEN_YEAR_FUNDAMENTALS_PATH)
    parser.add_argument("--output-dir", type=Path, default=TEN_YEAR_OUTPUT_DIR)
    parser.add_argument("--refresh-market-data", action="store_true")
    parser.add_argument("--with-celh-robustness", action="store_true")
    args = parser.parse_args()

    returns, spy = download_dynamic_weekly_returns(args.output_dir, args.refresh_market_data)
    fundamentals = load_fundamentals(args.fundamentals)
    full = run_dynamic_experiment(returns, spy, fundamentals)
    persist_run(full, args.output_dir, args.fundamentals, ())
    print(build_summary(full).to_string(index=False))
    print("\nCoverage:\n" + full["coverage"].to_string(index=False))

    if args.with_celh_robustness:
        ex_celh = run_dynamic_experiment(returns, spy, fundamentals, exclusions=("CELH",))
        persist_run(ex_celh, args.output_dir, args.fundamentals, ("CELH",))


if __name__ == "__main__":
    main()
