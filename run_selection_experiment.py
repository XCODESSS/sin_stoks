"""Run isolated point-in-time stock-selection experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtest_engine import (
    BacktestConfig,
    BacktestResult,
    adapt_legacy_strategy,
    run_contextual_walk_forward_backtest,
)
from config import (
    DATA_DIR,
    DEFAULT_MAX_WEIGHT,
    FUNDAMENTALS_PATH,
    REBALANCE_YEARS,
    SELECTION_CLUSTER_COUNT,
    SELECTION_CORRELATION_WEIGHT,
    SELECTION_DIVERSIFICATION_PENALTY,
    SELECTION_FEATURE_WEIGHT,
    SELECTION_LOOKBACK_WEEKS,
    SELECTION_MIN_CLUSTER_SIZE,
    SELECTION_MIN_COVERAGE,
    SELECTION_MIN_SAMPLES,
    SELECTION_OUTPUT_DIR,
    SELECTION_TARGET_COUNT,
)
from data_pipeline import write_csv_outputs_atomically
from fundamental_data import load_fundamentals
from portfolio_strategies import equal_weight
from run_backtest import load_returns, load_spy_returns
from selection_strategies import SelectionStrategy, select_density, select_partitioned


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run_directory(output_dir: Path, exclusions: tuple[str, ...]) -> Path:
    return output_dir / ("ex_celh" if exclusions == ("CELH",) else "full")


def _write_json_atomically(payload: dict[str, object], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _build_metadata(
    fundamentals_path: Path,
    exclusions: tuple[str, ...],
    returns: pd.DataFrame,
    spy_returns: pd.Series,
) -> dict[str, object]:
    input_paths = {
        "asset_returns": DATA_DIR / "weekly_returns.csv",
        "spy_returns": DATA_DIR / "spy_weekly_returns.csv",
        "fundamentals": fundamentals_path,
    }
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "exclusions": list(exclusions),
        "parameters": {
            "target_count": SELECTION_TARGET_COUNT,
            "cluster_count": SELECTION_CLUSTER_COUNT,
            "lookback_weeks": SELECTION_LOOKBACK_WEEKS,
            "min_cluster_size": SELECTION_MIN_CLUSTER_SIZE,
            "min_samples": SELECTION_MIN_SAMPLES,
            "feature_weight": SELECTION_FEATURE_WEIGHT,
            "correlation_weight": SELECTION_CORRELATION_WEIGHT,
            "diversification_penalty": SELECTION_DIVERSIFICATION_PENALTY,
            "min_coverage": SELECTION_MIN_COVERAGE,
            "max_weight": DEFAULT_MAX_WEIGHT,
        },
        "inputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for name, path in input_paths.items()
        },
        "asset_return_rows": len(returns),
        "asset_count": len(returns.columns),
        "asset_return_start": returns.index.min().isoformat(),
        "asset_return_end": returns.index.max().isoformat(),
        "spy_return_rows": len(spy_returns),
        "spy_return_start": spy_returns.index.min().isoformat(),
        "spy_return_end": spy_returns.index.max().isoformat(),
    }


def run_experiment(
    fundamentals_path: Path = FUNDAMENTALS_PATH,
    output_dir: Path = SELECTION_OUTPUT_DIR,
    exclusions: tuple[str, ...] = (),
) -> BacktestResult:
    """Run equal-weight baseline plus both preregistered selectors."""
    normalized_exclusions = tuple(sorted(ticker.upper() for ticker in exclusions))
    unsupported = set(normalized_exclusions).difference({"CELH"})
    if unsupported:
        raise ValueError(f"Only the preregistered CELH exclusion is allowed: {sorted(unsupported)}")

    returns = load_returns()
    spy_returns = load_spy_returns()
    fundamentals = load_fundamentals(fundamentals_path)
    retained_tickers = [ticker for ticker in returns.columns if ticker not in normalized_exclusions]
    if len(retained_tickers) < SELECTION_TARGET_COUNT:
        raise ValueError("Exclusions leave fewer assets than the frozen selection target")
    experiment_returns = returns.loc[:, retained_tickers].copy()

    partitioning = SelectionStrategy(
        "Partitioning Selection",
        fundamentals,
        select_partitioned,
    )
    density = SelectionStrategy(
        "Density Selection",
        fundamentals,
        select_density,
    )
    strategies = {
        "Equal Weight": adapt_legacy_strategy(equal_weight),
        "Partitioning Selection": partitioning,
        "Density Selection": density,
    }
    config = BacktestConfig(
        max_weight=DEFAULT_MAX_WEIGHT,
        rebalance_years=list(REBALANCE_YEARS),
    )
    result = run_contextual_walk_forward_backtest(
        experiment_returns,
        spy_returns,
        strategies,
        config,
    )

    run_directory = _run_directory(output_dir, normalized_exclusions)
    audit = pd.concat([partitioning.audit_frame(), density.audit_frame()], ignore_index=True)
    csv_outputs: dict[Path, tuple[pd.DataFrame | pd.Series, dict[str, object]]] = {
        run_directory / "walk_forward_returns.csv": (result.period_returns, {}),
        run_directory / "walk_forward_values.csv": (result.portfolio_values, {}),
        run_directory / "walk_forward_weights.csv": (result.weights, {}),
        run_directory / "turnover.csv": (result.turnover, {"index": False}),
        run_directory / "selection_audit.csv": (audit, {"index": False}),
    }
    write_csv_outputs_atomically(csv_outputs)
    metadata = _build_metadata(
        fundamentals_path,
        normalized_exclusions,
        experiment_returns,
        spy_returns,
    )
    _write_json_atomically(metadata, run_directory / "run_metadata.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen stock-selection experiments.")
    parser.add_argument("--fundamentals", type=Path, default=FUNDAMENTALS_PATH)
    parser.add_argument("--output-dir", type=Path, default=SELECTION_OUTPUT_DIR)
    parser.add_argument("--with-celh-robustness", action="store_true")
    args = parser.parse_args()

    run_experiment(args.fundamentals, args.output_dir)
    if args.with_celh_robustness:
        run_experiment(args.fundamentals, args.output_dir, exclusions=("CELH",))


if __name__ == "__main__":
    main()
