from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from free_fundamental_builder import FreeFundamentalBuild
from prepare_free_fundamentals import PROVENANCE_COLUMNS, prepare_free_fundamentals


def successful_build() -> FreeFundamentalBuild:
    rows = []
    for position in range(24):
        row = {column: "synthetic" for column in PROVENANCE_COLUMNS}
        row.update(
            {
                "ticker": f"T{position:02d}",
                "rebalance_date": pd.Timestamp("2020-01-01"),
                "observation_date": pd.Timestamp("2019-12-30"),
                "available_date": pd.Timestamp("2019-12-30"),
                "trailing_pe": 10.0 + position,
                "market_cap": 1_000_000.0 * (position + 1),
                "earnings_positive": True,
                "source": "Synthetic SEC and Yahoo",
                "cik": position + 1,
                "price_date": pd.Timestamp("2019-12-30"),
                "spot_fx_to_usd": 1.0,
                "earnings_start": pd.Timestamp("2018-10-01"),
                "earnings_end": pd.Timestamp("2019-09-30"),
                "earnings_available_date": pd.Timestamp("2019-11-01"),
                "shares_date": pd.Timestamp("2019-10-31"),
                "shares_available_date": pd.Timestamp("2019-11-05"),
                "shares_component_count": 1,
                "filed_shares": 100.0,
                "trailing_earnings_usd": 100_000.0,
            }
        )
        rows.append(row)
    coverage = pd.DataFrame(
        [
            {
                "rebalance_date": pd.Timestamp("2020-01-01"),
                "expected_assets": 30,
                "automated_candidates": 25,
                "eligible_assets": 24,
                "coverage": 0.8,
                "minimum_required_assets": 24,
                "coverage_passed": True,
                "missing_tickers": "M0, M1, M2, M3, M4, T24",
                "failed_tickers": "T24",
            }
        ]
    )
    errors = pd.DataFrame(
        [{"rebalance_date": "2020-01-01", "ticker": "T24", "stage": "ValueError", "error": "missing"}]
    )
    return FreeFundamentalBuild(pd.DataFrame(rows), coverage, errors)


def test_preparation_writes_only_source_outputs(tmp_path, monkeypatch):
    build = successful_build()
    monkeypatch.setattr(
        "prepare_free_fundamentals.build_source_clients", lambda **kwargs: (object(), object())
    )
    monkeypatch.setattr("prepare_free_fundamentals.build_free_fundamentals", lambda sec, market: build)
    monkeypatch.setattr(
        "prepare_free_fundamentals.build_source_manifest",
        lambda *args, **kwargs: {"methodology": "sec-yfinance-v1"},
    )
    fundamentals_path = tmp_path / "data/fundamentals_point_in_time.csv"
    output_dir = tmp_path / "outputs/selection_experiment"

    result = prepare_free_fundamentals(
        fundamentals_path=fundamentals_path,
        output_dir=output_dir,
        cache_dir=tmp_path / "data/source_cache",
        sec_user_agent="sin_stoks test test@example.com",
    )

    assert result.coverage["coverage_passed"].all()
    assert fundamentals_path.exists()
    loaded = pd.read_csv(fundamentals_path)
    assert "earnings_accessions" in loaded.columns
    assert (output_dir / "source_coverage.csv").exists()
    assert (output_dir / "source_errors.csv").exists()
    assert json.loads((output_dir / "source_manifest.json").read_text())["methodology"] == "sec-yfinance-v1"
    assert not (output_dir / "full").exists()
    assert not (tmp_path / "outputs/portfolio_backtest").exists()


def test_preparation_rejects_missing_contact_before_source_clients(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "prepare_free_fundamentals.build_source_clients",
        lambda **kwargs: pytest.fail("source clients must not run"),
    )
    with pytest.raises(ValueError, match="explicitly set"):
        prepare_free_fundamentals(
            fundamentals_path=tmp_path / "fundamentals.csv",
            output_dir=tmp_path / "outputs",
            cache_dir=tmp_path / "cache",
            sec_user_agent="",
        )


def test_preparation_module_cannot_import_backtest_runner():
    source = Path("prepare_free_fundamentals.py").read_text(encoding="utf-8")
    assert "run_selection_experiment" not in source
    assert "backtest_engine" not in source


def test_generated_source_paths_are_gitignored():
    paths = [
        "data/source_cache/CIK0000000001.json",
        "data/fundamentals_point_in_time.csv",
        "outputs/selection_experiment/source_manifest.json",
    ]
    completed = subprocess.run(
        ["git", "check-ignore", *paths],
        check=True,
        capture_output=True,
        text=True,
    )
    assert set(completed.stdout.splitlines()) == set(paths)
