"""Evaluate stock-selection experiments and preregistered promotion gates."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PORTFOLIO_OUTPUT_DIR, SELECTION_MIN_COVERAGE, SELECTION_OUTPUT_DIR, STARTING_VALUE
from data_pipeline import write_csv_outputs_atomically
from reporting.metrics import calculate_information_ratio
from reporting.tables import build_summary_table
from run_backtest import load_returns

plt.switch_backend("Agg")

CANDIDATE_STRATEGIES = ("Partitioning Selection", "Density Selection")
BENCHMARK_STRATEGIES = ("Equal Weight", "SPY")


@dataclass(frozen=True)
class RunArtifacts:
    """Persisted outputs for one full or robustness run."""

    returns: pd.DataFrame
    values: pd.DataFrame
    weights: pd.DataFrame
    turnover: pd.DataFrame
    audit: pd.DataFrame
    metadata: dict[str, object]


def _read_run(run_directory: Path) -> RunArtifacts:
    returns = pd.read_csv(run_directory / "walk_forward_returns.csv", index_col=0, parse_dates=True)
    values = pd.read_csv(run_directory / "walk_forward_values.csv", index_col=0, parse_dates=True)
    weights = pd.read_csv(run_directory / "walk_forward_weights.csv", index_col=[0, 1])
    rebalance_dates = pd.to_datetime(weights.index.get_level_values(0))
    strategies = weights.index.get_level_values(1)
    weights.index = pd.MultiIndex.from_arrays(
        [rebalance_dates, strategies], names=["Rebalance Date", "Strategy"]
    )
    turnover = pd.read_csv(run_directory / "turnover.csv", parse_dates=["Rebalance Date"])
    audit = pd.read_csv(
        run_directory / "selection_audit.csv",
        parse_dates=["rebalance_date", "available_date"],
    )
    metadata = json.loads((run_directory / "run_metadata.json").read_text(encoding="utf-8"))
    return RunArtifacts(returns, values, weights, turnover, audit, metadata)


def build_annual_comparison(returns: pd.DataFrame) -> pd.DataFrame:
    """Compound weekly returns by calendar year and add benchmark differences."""
    annual = (1.0 + returns).groupby(returns.index.year).prod() - 1.0
    for strategy in returns.columns:
        if strategy in BENCHMARK_STRATEGIES:
            continue
        if "Equal Weight" in annual:
            annual[f"{strategy} vs Equal Weight"] = annual[strategy] - annual["Equal Weight"]
        if "SPY" in annual:
            annual[f"{strategy} vs SPY"] = annual[strategy] - annual["SPY"]
    annual.index.name = "Year"
    return annual


def _summary_with_turnover(artifacts: RunArtifacts, asset_returns: pd.DataFrame) -> pd.DataFrame:
    summary = build_summary_table(
        artifacts.returns,
        artifacts.values,
        artifacts.weights,
        asset_returns.loc[:, artifacts.weights.columns],
    )
    recurring_turnover = (
        artifacts.turnover.sort_values("Rebalance Date")
        .groupby("Strategy", sort=False)["Turnover"]
        .apply(lambda values: float(values.iloc[1:].mean()) if len(values) > 1 else 0.0)
    )
    summary["Turnover"] = summary["Strategy"].map(recurring_turnover).fillna(
        summary["Turnover"]
    )
    return summary


def _selection_stability(audit: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for strategy in CANDIDATE_STRATEGIES:
        strategy_audit = audit.loc[audit["strategy"] == strategy]
        selected_by_date = {
            date: set(group.loc[group["selected"].astype(bool), "ticker"])
            for date, group in strategy_audit.groupby("rebalance_date")
        }
        ordered_dates = sorted(selected_by_date)
        for position, date in enumerate(ordered_dates):
            selected = selected_by_date[date]
            previous = selected_by_date[ordered_dates[position - 1]] if position else set()
            union = selected | previous
            jaccard = len(selected & previous) / len(union) if union else np.nan
            strategy_weights = weights.loc[(date, strategy)]
            effective_holdings = 1.0 / float(np.square(strategy_weights).sum())
            records.append(
                {
                    "strategy": strategy,
                    "rebalance_date": date,
                    "selected_count": len(selected),
                    "jaccard_vs_previous": jaccard,
                    "effective_holdings": effective_holdings,
                }
            )
    return pd.DataFrame(records)


def _coverage_by_rebalance(artifacts: RunArtifacts) -> pd.DataFrame:
    expected_assets = int(artifacts.metadata["asset_count"])
    coverage = (
        artifacts.audit.groupby(["strategy", "rebalance_date"])["ticker"]
        .nunique()
        .div(expected_assets)
        .rename("fundamental_coverage")
        .reset_index()
    )
    return coverage


def _recurring_turnover(artifacts: RunArtifacts, strategy: str) -> float:
    values = artifacts.turnover.loc[
        artifacts.turnover["Strategy"] == strategy
    ].sort_values("Rebalance Date")["Turnover"]
    return float(values.iloc[1:].mean()) if len(values) > 1 else 0.0


def evaluate_promotion_gates(
    full: RunArtifacts,
    ex_celh: RunArtifacts,
    quality_checks_passed: bool,
) -> dict[str, object]:
    """Evaluate each candidate against the immutable research gates."""
    asset_returns = load_returns()
    full_summary = _summary_with_turnover(full, asset_returns).set_index("Strategy")
    ex_summary = _summary_with_turnover(
        ex_celh,
        asset_returns.drop(columns=["CELH"], errors="ignore"),
    ).set_index("Strategy")
    annual = build_annual_comparison(full.returns)
    coverage = _coverage_by_rebalance(full)

    decisions: dict[str, object] = {}
    for strategy in CANDIDATE_STRATEGIES:
        candidate_cagr = float(full_summary.loc[strategy, "CAGR"])
        equal_weight_cagr = float(full_summary.loc["Equal Weight", "CAGR"])
        spy_cagr = float(full_summary.loc["SPY", "CAGR"])
        annual_wins = int((annual[strategy] > annual["Equal Weight"]).sum())
        information_ratio = calculate_information_ratio(
            full.returns[strategy], full.returns["Equal Weight"]
        )
        candidate_drawdown = float(full_summary.loc[strategy, "Maximum Drawdown"])
        equal_weight_drawdown = float(full_summary.loc["Equal Weight", "Maximum Drawdown"])
        turnover = _recurring_turnover(full, strategy)
        ex_candidate_cagr = float(ex_summary.loc[strategy, "CAGR"])
        ex_equal_weight_cagr = float(ex_summary.loc["Equal Weight", "CAGR"])
        strategy_coverage = coverage.loc[coverage["strategy"] == strategy, "fundamental_coverage"]
        minimum_coverage = float(strategy_coverage.min())

        gates = {
            "cagr_exceeds_equal_weight_and_spy": candidate_cagr > max(equal_weight_cagr, spy_cagr),
            "beats_equal_weight_in_four_of_six_years": annual_wins >= 4,
            "positive_information_ratio_vs_equal_weight": information_ratio > 0.0,
            "drawdown_within_three_percentage_points": candidate_drawdown
            >= equal_weight_drawdown - 0.03,
            "recurring_turnover_at_most_sixty_percent": turnover <= 0.60,
            "ex_celh_cagr_exceeds_equal_weight": ex_candidate_cagr > ex_equal_weight_cagr,
            "coverage_at_least_eighty_percent": minimum_coverage >= SELECTION_MIN_COVERAGE,
            "quality_checks_passed": quality_checks_passed,
        }
        decisions[strategy] = {
            "research_promising": all(gates.values()),
            "classification": "research-promising" if all(gates.values()) else "feasible-but-not-promising",
            "gates": gates,
            "supporting_values": {
                "candidate_cagr": candidate_cagr,
                "equal_weight_cagr": equal_weight_cagr,
                "spy_cagr": spy_cagr,
                "annual_wins_vs_equal_weight": annual_wins,
                "information_ratio_vs_equal_weight": information_ratio,
                "candidate_max_drawdown": candidate_drawdown,
                "equal_weight_max_drawdown": equal_weight_drawdown,
                "average_recurring_turnover": turnover,
                "ex_celh_candidate_cagr": ex_candidate_cagr,
                "ex_celh_equal_weight_cagr": ex_equal_weight_cagr,
                "minimum_fundamental_coverage": minimum_coverage,
            },
        }
    return decisions


def _combine_strategy_returns(existing_returns: pd.DataFrame, selection_returns: pd.DataFrame) -> pd.DataFrame:
    if not existing_returns.index.equals(selection_returns.index):
        raise ValueError("Existing and selection strategy return dates must match exactly")
    shared = [column for column in ("Equal Weight", "SPY") if column in existing_returns]
    for column in shared:
        if not np.allclose(existing_returns[column], selection_returns[column], atol=1e-12, rtol=0.0):
            raise ValueError(f"Existing and selection {column} returns do not match")
    new_columns = [column for column in selection_returns if column not in existing_returns]
    return pd.concat([existing_returns, selection_returns[new_columns]], axis=1)


def build_all_strategy_summary(
    selection: RunArtifacts,
    existing_output_dir: Path = PORTFOLIO_OUTPUT_DIR,
) -> pd.DataFrame:
    """Build one comparable metric table for all existing and new strategies."""
    existing_returns = pd.read_csv(
        existing_output_dir / "walk_forward_returns.csv", index_col=0, parse_dates=True
    )
    combined_returns = _combine_strategy_returns(existing_returns, selection.returns)
    combined_values = STARTING_VALUE * (1.0 + combined_returns).cumprod()
    start_date = pd.Timestamp(combined_returns.index.min().year, 1, 1)
    if start_date not in combined_values.index:
        combined_values = pd.concat(
            [
                pd.DataFrame(STARTING_VALUE, index=[start_date], columns=combined_values.columns),
                combined_values,
            ]
        )

    existing_weights = pd.read_csv(
        existing_output_dir / "walk_forward_weights.csv", index_col=[0, 1]
    )
    existing_weights.index = pd.MultiIndex.from_arrays(
        [
            pd.to_datetime(existing_weights.index.get_level_values(0)),
            existing_weights.index.get_level_values(1),
        ],
        names=["Rebalance Date", "Strategy"],
    )
    new_weight_rows = selection.weights.loc[
        selection.weights.index.get_level_values("Strategy").isin(CANDIDATE_STRATEGIES)
    ]
    combined_weights = pd.concat([existing_weights, new_weight_rows]).fillna(0.0)
    return build_summary_table(
        combined_returns,
        combined_values,
        combined_weights,
        load_returns(),
    )


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def generate_diagnostic_graphs(
    artifacts: RunArtifacts,
    annual: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create concise performance, risk, and selection diagnostic graphs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_values = artifacts.values / artifacts.values.iloc[0]
    figure, axis = plt.subplots(figsize=(11, 6))
    normalized_values.plot(ax=axis)
    axis.set(title="Selection Experiment Equity Curves", ylabel="Growth of $1", xlabel="Date")
    axis.grid(alpha=0.25)
    _save_figure(figure, output_dir / "selection_equity_curves.png")

    drawdowns = artifacts.values.div(artifacts.values.cummax()).sub(1.0)
    figure, axis = plt.subplots(figsize=(11, 6))
    drawdowns.plot(ax=axis)
    axis.set(title="Selection Experiment Drawdowns", ylabel="Drawdown", xlabel="Date")
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    axis.grid(alpha=0.25)
    _save_figure(figure, output_dir / "selection_drawdowns.png")

    annual_columns = [*CANDIDATE_STRATEGIES, *BENCHMARK_STRATEGIES]
    figure, axis = plt.subplots(figsize=(12, 6))
    annual.loc[:, annual_columns].plot(kind="bar", ax=axis)
    axis.set(title="Calendar-Year Strategy Returns", ylabel="Return", xlabel="Year")
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, output_dir / "annual_strategy_returns.png")

    selected = artifacts.audit.loc[artifacts.audit["selected"].astype(bool)]
    frequency = selected.groupby(["ticker", "strategy"]).size().unstack(fill_value=0)
    frequency = frequency.sort_values(list(CANDIDATE_STRATEGIES), ascending=False)
    figure, axis = plt.subplots(figsize=(12, 7))
    frequency.plot(kind="bar", ax=axis)
    axis.set(title="Selection Frequency by Ticker", ylabel="Rebalances Selected", xlabel="Ticker")
    axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, output_dir / "selection_frequency.png")


def _markdown_table(frame: pd.DataFrame, include_index: bool = False) -> str:
    display = frame.reset_index() if include_index else frame.copy()
    headers = [str(column) for column in display.columns]

    def format_value(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    rows = [
        "| " + " | ".join(format_value(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _build_markdown_report(
    decisions: dict[str, object],
    summary: pd.DataFrame,
    annual: pd.DataFrame,
) -> str:
    outcomes = ", ".join(
        f"{strategy}: {decision['classification']}"
        for strategy, decision in decisions.items()
    )
    return "\n".join(
        [
            "# Stock-Selection Experiment Report",
            "",
            f"**Preregistered gate outcome:** {outcomes}.",
            "",
            "This is a historical walk-forward result over a retrospective 30-stock universe, not an "
            "investment recommendation or a strict constituent-level out-of-sample test.",
            "",
            "## Full-Universe Metrics",
            "",
            _markdown_table(summary),
            "",
            "## Calendar-Year Returns",
            "",
            _markdown_table(annual, include_index=True),
            "",
            "Only six annual holding periods are available, so no reliable statistical significance "
            "or p-value claim is made. Full and ex-CELH results, coverage, turnover, and integrity gates "
            "are recorded in `decision.json` and the companion CSV files.",
        ]
    )


def generate_report(
    input_dir: Path = SELECTION_OUTPUT_DIR,
    quality_checks_passed: bool = False,
) -> dict[str, object]:
    """Generate comparison tables, diagnostics, gate decisions, and Markdown report."""
    full = _read_run(input_dir / "full")
    ex_celh = _read_run(input_dir / "ex_celh")
    asset_returns = load_returns()
    summary = _summary_with_turnover(full, asset_returns)
    annual = build_annual_comparison(full.returns)
    stability = _selection_stability(full.audit, full.weights)
    coverage = _coverage_by_rebalance(full)
    decisions = evaluate_promotion_gates(full, ex_celh, quality_checks_passed)
    all_strategies = build_all_strategy_summary(full)
    generate_diagnostic_graphs(full, annual, input_dir)

    csv_outputs: dict[Path, tuple[pd.DataFrame | pd.Series, dict[str, object]]] = {
        input_dir / "comparison_summary.csv": (summary, {"index": False}),
        input_dir / "annual_excess_returns.csv": (annual, {}),
        input_dir / "selection_stability.csv": (stability, {"index": False}),
        input_dir / "fundamental_coverage.csv": (coverage, {"index": False}),
        input_dir / "all_strategy_summary.csv": (all_strategies, {"index": False}),
    }
    write_csv_outputs_atomically(csv_outputs)
    decision_path = input_dir / "decision.json"
    decision_path.write_text(json.dumps(decisions, indent=2, sort_keys=True), encoding="utf-8")
    report = _build_markdown_report(decisions, summary, annual)
    (input_dir / "experiment_report.md").write_text(report, encoding="utf-8")
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser(description="Report frozen stock-selection experiment results.")
    parser.add_argument("--input-dir", type=Path, default=SELECTION_OUTPUT_DIR)
    parser.add_argument(
        "--quality-checks-passed",
        action="store_true",
        help="Mark the integrity gate true only after all frozen quality commands pass.",
    )
    args = parser.parse_args()
    generate_report(args.input_dir, quality_checks_passed=args.quality_checks_passed)


if __name__ == "__main__":
    main()
