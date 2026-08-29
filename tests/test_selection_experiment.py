from __future__ import annotations

import json

import numpy as np
import pandas as pd

from run_selection_experiment import run_experiment


def make_experiment_inputs():
    tickers = ["CELH", *[f"T{position:02d}" for position in range(29)]]
    dates = pd.date_range("2017-01-06", "2025-12-26", freq="W-FRI")
    phases = np.linspace(0.0, 2.0 * np.pi, len(tickers), endpoint=False)
    simple_returns = np.column_stack(
        [
            0.002 + 0.012 * np.sin(np.linspace(0.0, 20.0, len(dates)) + phase) + 0.0001 * position
            for position, phase in enumerate(phases)
        ]
    )
    returns = pd.DataFrame(np.log1p(simple_returns), index=dates, columns=tickers)
    spy_returns = pd.Series(
        np.log1p(0.0015 + 0.006 * np.cos(np.linspace(0.0, 16.0, len(dates)))),
        index=dates,
        name="SPY",
    )
    return returns, spy_returns


def write_fundamentals(path, tickers):
    rows = []
    for year in range(2019, 2025):
        for position, ticker in enumerate(tickers):
            rows.append(
                {
                    "ticker": ticker,
                    "observation_date": f"{year}-09-30",
                    "available_date": f"{year}-11-15",
                    "trailing_pe": 10.0 + position + 0.1 * (year - 2019),
                    "market_cap": 1e9 * (position + 1) * (1.0 + 0.02 * (year - 2019)),
                    "earnings_positive": True,
                    "source": "Synthetic test fixture",
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def patch_inputs(monkeypatch):
    returns, spy_returns = make_experiment_inputs()
    monkeypatch.setattr("run_selection_experiment.load_returns", lambda: returns.copy())
    monkeypatch.setattr("run_selection_experiment.load_spy_returns", lambda: spy_returns.copy())
    return returns


def test_experiment_writes_only_isolated_outputs(tmp_path, monkeypatch):
    returns = patch_inputs(monkeypatch)
    fundamentals_path = tmp_path / "fundamentals.csv"
    write_fundamentals(fundamentals_path, list(returns.columns))
    output_dir = tmp_path / "selection_experiment"

    result = run_experiment(fundamentals_path=fundamentals_path, output_dir=output_dir)

    assert set(result.period_returns.columns) == {
        "Equal Weight",
        "Eligible Universe Equal Weight",
        "Partitioning Selection",
        "Density Selection",
        "SPY",
    }
    assert (output_dir / "full/walk_forward_returns.csv").exists()
    assert (output_dir / "full/selection_audit.csv").exists()
    assert (output_dir / "full/turnover.csv").exists()
    metadata_path = output_dir / "full/run_metadata.json"
    assert metadata_path.exists()
    assert json.loads(metadata_path.read_text())["parameters"]["target_count"] == 12
    assert not (tmp_path / "portfolio_backtest").exists()


def test_ex_celh_run_excludes_celh_before_selection(tmp_path, monkeypatch):
    returns = patch_inputs(monkeypatch)
    fundamentals_path = tmp_path / "fundamentals.csv"
    write_fundamentals(fundamentals_path, list(returns.columns))
    output_dir = tmp_path / "selection_experiment"

    result = run_experiment(
        fundamentals_path=fundamentals_path,
        output_dir=output_dir,
        exclusions=("CELH",),
    )

    assert "CELH" not in result.weights.columns
    selector_weights = result.weights.loc[
        result.weights.index.get_level_values("Strategy").isin(
            ["Partitioning Selection", "Density Selection"]
        )
    ]
    assert (selector_weights.gt(0).sum(axis=1) == 12).all()
    audit = pd.read_csv(output_dir / "ex_celh/selection_audit.csv")
    assert "CELH" not in set(audit["ticker"])
