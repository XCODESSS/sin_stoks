"""Orchestrator CLI: loads market returns, executes registered strategies, and persists artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtest_engine import BacktestConfig, BacktestResult, run_walk_forward_backtest
from config import (
    COVARIANCE_END,
    COVARIANCE_START,
    DATA_DIR,
    DEFAULT_MAX_WEIGHT,
    PORTFOLIO_OUTPUT_DIR,
    REBALANCE_YEARS,
)
from data_pipeline import write_csv_outputs_atomically
from portfolio_strategies import STRATEGIES


def load_returns() -> pd.DataFrame:
    """Load weekly log returns of portfolio assets, handling interval indices."""
    csv_path = DATA_DIR / "weekly_returns.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}. Run 'python main.py' to download market data first.")
    returns = pd.read_csv(csv_path, index_col=0)
    raw_index = returns.index.astype(str)
    if raw_index.str.contains("/").any():
        returns.index = pd.to_datetime(raw_index.str.split("/").str[-1])
    else:
        returns.index = pd.to_datetime(raw_index)

    returns = returns.loc[(returns.index >= COVARIANCE_START) & (returns.index <= COVARIANCE_END)]
    return returns.fillna(0.0)


def load_spy_returns() -> pd.Series:
    """Load weekly log returns of SPY benchmark."""
    csv_path = DATA_DIR / "spy_weekly_returns.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}. Run 'python main.py' to download benchmark data first.")
    returns = pd.read_csv(csv_path, index_col=0)
    returns.index = pd.to_datetime(returns.index)
    return returns["SPY"].dropna()


def save_backtest_artifacts(result: BacktestResult, output_dir: Path) -> None:
    """Persist walk-forward outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_outputs: dict[Path, tuple[pd.DataFrame | pd.Series, dict[str, object]]] = {
        output_dir / "walk_forward_returns.csv": (result.period_returns, {}),
        output_dir / "walk_forward_values.csv": (result.portfolio_values, {}),
        output_dir / "walk_forward_weights.csv": (result.weights, {}),
    }
    write_csv_outputs_atomically(csv_outputs)


def run_orchestrator(
    cap: float = DEFAULT_MAX_WEIGHT,
    output_dir: Path = PORTFOLIO_OUTPUT_DIR,
) -> BacktestResult:
    """Run walk-forward optimization with given parameters and save results."""
    returns = load_returns()
    spy_returns = load_spy_returns()

    config = BacktestConfig(
        max_weight=cap,
        rebalance_years=list(REBALANCE_YEARS),
    )

    result = run_walk_forward_backtest(returns, spy_returns, STRATEGIES, config)
    save_backtest_artifacts(result, output_dir)

    print("\n" + "=" * 70)
    print(f"  BACKTEST RUN SUMMARY (Cap: {cap:.0%})")
    print("=" * 70)
    for col in result.portfolio_values.columns:
        init_val = config.starting_value
        final_val = result.portfolio_values[col].iloc[-1]
        tot_ret = (final_val / init_val) - 1.0
        start_date = pd.Timestamp(result.portfolio_values.index[0])
        end_date = pd.Timestamp(result.portfolio_values.index[-1])
        elapsed_years = (end_date - start_date).days / 365.25
        cagr = (final_val / init_val) ** (1.0 / elapsed_years) - 1.0 if elapsed_years > 0 else float("nan")
        print(f"  {col:<26}: ${final_val:>10,.2f}  ({tot_ret:>+7.2%} | Net CAGR: {cagr:>6.2%})")
    print("=" * 70)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward portfolio optimization orchestrator.")
    parser.add_argument(
        "--cap",
        type=float,
        default=DEFAULT_MAX_WEIGHT,
        help="Maximum position weight per asset (default: 0.25)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PORTFOLIO_OUTPUT_DIR,
        help="Output directory for generated artifacts",
    )

    args = parser.parse_args()
    run_orchestrator(
        cap=args.cap,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
