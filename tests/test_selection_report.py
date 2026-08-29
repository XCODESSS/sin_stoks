from __future__ import annotations

import numpy as np
import pandas as pd

from report_selection_experiment import (
    CANDIDATE_STRATEGIES,
    ELIGIBLE_BASELINE,
    RunArtifacts,
    build_all_strategy_summary,
    build_annual_comparison,
    evaluate_promotion_gates,
    generate_benchmark_comparison_graphs,
    generate_diagnostic_graphs,
    generate_ex_celh_sensitivity_graph,
)


def test_annual_returns_compound_weekly_returns():
    returns = pd.DataFrame(
        {"Candidate": [0.10, -0.10], "Equal Weight": [0.05, 0.05]},
        index=pd.to_datetime(["2020-01-03", "2020-01-10"]),
    )

    annual = build_annual_comparison(returns)

    assert np.isclose(annual.loc[2020, "Candidate"], 1.10 * 0.90 - 1.0)
    assert np.isclose(
        annual.loc[2020, "Candidate vs Equal Weight"],
        (1.10 * 0.90) - (1.05 * 1.05),
    )


def make_gate_artifacts(candidate_return: float, equal_weight_return: float = 0.05) -> RunArtifacts:
    dates = pd.to_datetime([f"{year}-01-03" for year in range(2020, 2026)])
    returns = pd.DataFrame(
        {
            "Equal Weight": equal_weight_return,
            ELIGIBLE_BASELINE: equal_weight_return,
            "Partitioning Selection": candidate_return,
            "Density Selection": candidate_return - 0.005,
            "SPY": 0.04,
        },
        index=dates,
    )
    values = 10_000.0 * (1.0 + returns).cumprod()
    values = pd.concat(
        [pd.DataFrame(10_000.0, index=[pd.Timestamp("2020-01-01")], columns=returns.columns), values]
    )
    weight_index = pd.MultiIndex.from_product(
        [dates, ["Equal Weight", ELIGIBLE_BASELINE, *CANDIDATE_STRATEGIES]],
        names=["Rebalance Date", "Strategy"],
    )
    weights = pd.DataFrame(0.5, index=weight_index, columns=["A", "B"])
    turnover = pd.DataFrame(
        [
            {
                "Rebalance Date": date,
                "Strategy": strategy,
                "Turnover": 1.0 if position == 0 else 0.10,
                "Cost": 0.001 if position == 0 else 0.0001,
            }
            for strategy in ["Equal Weight", ELIGIBLE_BASELINE, *CANDIDATE_STRATEGIES]
            for position, date in enumerate(dates)
        ]
    )
    audit = pd.DataFrame(
        [
            {
                "rebalance_date": date,
                "available_date": date - pd.Timedelta(days=30),
                "strategy": strategy,
                "ticker": ticker,
                "selected": True,
                "cluster_label": 0 if ticker == "A" else 1,
                "value_rank": 0.25 if ticker == "A" else 0.75,
                "size_rank": 0.40 if ticker == "A" else 0.80,
                "sharpe_rank": 0.30 if ticker == "A" else 0.70,
            }
            for date in dates
            for strategy in CANDIDATE_STRATEGIES
            for ticker in ["A", "B"]
        ]
    )
    return RunArtifacts(
        returns=returns,
        values=values,
        weights=weights,
        turnover=turnover,
        audit=audit,
        metadata={"asset_count": 2},
    )


def test_all_strategy_summary_combines_existing_and_new_metrics(tmp_path, monkeypatch):
    selection = make_gate_artifacts(candidate_return=0.08)
    existing_returns = selection.returns[["Equal Weight", "SPY"]].copy()
    existing_returns["Existing Strategy"] = 0.06
    existing_returns.to_csv(tmp_path / "walk_forward_returns.csv")
    existing_weights = selection.weights.loc[
        selection.weights.index.get_level_values("Strategy") == "Equal Weight"
    ].copy()
    existing_strategy_weights = existing_weights.copy()
    existing_strategy_weights.index = pd.MultiIndex.from_arrays(
        [
            existing_strategy_weights.index.get_level_values("Rebalance Date"),
            ["Existing Strategy"] * len(existing_strategy_weights),
        ],
        names=["Rebalance Date", "Strategy"],
    )
    pd.concat([existing_weights, existing_strategy_weights]).to_csv(tmp_path / "walk_forward_weights.csv")
    asset_returns = pd.DataFrame(0.01, index=selection.returns.index, columns=["A", "B"])
    monkeypatch.setattr("report_selection_experiment.load_returns", lambda: asset_returns)

    summary = build_all_strategy_summary(selection, existing_output_dir=tmp_path)

    assert set(summary["Strategy"]) == {
        "Equal Weight",
        "Existing Strategy",
        ELIGIBLE_BASELINE,
        "Partitioning Selection",
        "Density Selection",
        "SPY",
    }
    assert {"CAGR", "Volatility", "Sharpe Ratio", "Maximum Drawdown", "Turnover"}.issubset(summary.columns)


def test_diagnostic_graphs_are_written(tmp_path):
    artifacts = make_gate_artifacts(candidate_return=0.08)
    annual = build_annual_comparison(artifacts.returns)

    generate_diagnostic_graphs(artifacts, annual, tmp_path)

    assert (tmp_path / "selection_equity_curves.png").exists()
    assert (tmp_path / "selection_drawdowns.png").exists()
    assert (tmp_path / "annual_strategy_returns.png").exists()
    assert (tmp_path / "selection_frequency.png").exists()
    assert (tmp_path / "partitioning_selection_clusters.png").exists()
    assert (tmp_path / "density_selection_clusters.png").exists()


def test_requested_benchmark_comparison_graphs_are_written(tmp_path):
    artifacts = make_gate_artifacts(candidate_return=0.08)
    existing_output = tmp_path / "existing"
    graph_output = tmp_path / "graphs"
    existing_output.mkdir()
    existing_returns = artifacts.returns[["Equal Weight", "SPY"]].copy()
    existing_returns["Max Sharpe"] = 0.07
    existing_returns["Maximum Diversification"] = 0.06
    existing_returns.to_csv(existing_output / "walk_forward_returns.csv")

    generate_benchmark_comparison_graphs(artifacts, graph_output, existing_output)

    assert (graph_output / "strategy_benchmark_equity_curves.png").exists()
    assert (graph_output / "strategy_benchmark_drawdowns.png").exists()
    assert (graph_output / "strategy_benchmark_annual_returns.png").exists()


def test_ex_celh_sensitivity_graph_is_written(tmp_path):
    full = make_gate_artifacts(candidate_return=0.08)
    ex_celh = make_gate_artifacts(candidate_return=0.02)

    generate_ex_celh_sensitivity_graph(full, ex_celh, tmp_path)

    assert (tmp_path / "ex_celh_sensitivity.png").exists()


def test_candidate_fails_when_ex_celh_advantage_reverses(monkeypatch):
    asset_dates = pd.to_datetime([f"{year}-01-03" for year in range(2020, 2026)])
    asset_returns = pd.DataFrame(0.01, index=asset_dates, columns=["A", "B"])
    monkeypatch.setattr("report_selection_experiment.load_returns", lambda: asset_returns)
    full = make_gate_artifacts(candidate_return=0.08)
    ex_celh = make_gate_artifacts(candidate_return=0.02)

    decisions = evaluate_promotion_gates(full, ex_celh, quality_checks_passed=True)

    partitioning = decisions["Partitioning Selection"]
    assert partitioning["gates"]["ex_celh_cagr_exceeds_full_and_eligible"] is False
    assert partitioning["research_promising"] is False
