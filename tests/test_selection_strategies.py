from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest_engine import RebalanceContext
from portfolio_strategies import StrategyConfig
from selection_features import SelectionFeatures
from selection_strategies import (
    SelectionConfig,
    SelectionStrategy,
    pam_labels,
    select_density,
    select_partitioned,
)


def block_distance_frame() -> pd.DataFrame:
    tickers = list("ABCDEF")
    values = np.full((6, 6), 0.9)
    np.fill_diagonal(values, 0.0)
    for group in (range(3), range(3, 6)):
        for left in group:
            for right in group:
                if left != right:
                    values[left, right] = 0.1
    return pd.DataFrame(values, index=tickers, columns=tickers)


def make_asset_features(n_assets: int = 30) -> SelectionFeatures:
    tickers = [f"T{position:02d}" for position in range(n_assets)]
    positions = np.arange(n_assets, dtype=float)
    distance_values = np.abs(positions[:, None] - positions[None, :]) / max(n_assets - 1, 1)
    correlation_values = 1.0 - 2.0 * distance_values**2
    np.fill_diagonal(correlation_values, 1.0)
    features = pd.DataFrame(
        {
            "value_rank": (positions + 1.0) / n_assets,
            "size_rank": (positions[::-1] + 1.0) / n_assets,
            "sharpe_rank": np.roll(positions + 1.0, 3) / n_assets,
            "trailing_sharpe": positions / 10.0,
        },
        index=tickers,
    )
    base_score = (0.5 * features["value_rank"] + 0.5 * features["sharpe_rank"]).rename(
        "base_score"
    )
    return SelectionFeatures(
        features=features,
        base_score=base_score,
        correlation=pd.DataFrame(correlation_values, index=tickers, columns=tickers),
        distance=pd.DataFrame(distance_values, index=tickers, columns=tickers),
    )


def test_pam_finds_two_obvious_partitions():
    labels = pam_labels(block_distance_frame(), n_clusters=2)

    assert labels.nunique() == 2
    assert labels["A"] == labels["B"] == labels["C"]
    assert labels["D"] == labels["E"] == labels["F"]
    assert labels["A"] != labels["D"]


def test_partitioned_selector_returns_exactly_twelve_unique_tickers():
    result = select_partitioned(make_asset_features(), SelectionConfig())

    assert len(result.selected_tickers) == 12
    assert len(set(result.selected_tickers)) == 12
    assert result.labels.nunique() == 6


def test_pam_is_order_invariant():
    distance = block_distance_frame()
    reordered = distance.loc[list(reversed(distance.index)), list(reversed(distance.columns))]

    pd.testing.assert_series_equal(pam_labels(distance, 2), pam_labels(reordered, 2))


def test_pam_breaks_equal_ties_lexicographically():
    distance = pd.DataFrame(
        np.ones((4, 4)) - np.eye(4),
        index=list("DCBA"),
        columns=list("DCBA"),
    )

    labels = pam_labels(distance, 2)

    assert labels.index.tolist() == list("ABCD")
    assert labels["A"] == 0


@pytest.mark.parametrize(
    ("transform", "message"),
    [
        (lambda frame: frame.assign(G=0.5), "same assets"),
        (lambda frame: frame.mask(np.eye(len(frame), dtype=bool), 0.2), "diagonal"),
    ],
)
def test_pam_rejects_invalid_distance(transform, message):
    with pytest.raises(ValueError, match=message):
        pam_labels(transform(block_distance_frame()), 2)


def test_pam_rejects_more_clusters_than_assets():
    with pytest.raises(ValueError, match="cannot exceed"):
        pam_labels(block_distance_frame(), 7)


def test_partitioned_selector_handles_single_asset_clusters():
    inputs = make_asset_features(12)
    result = select_partitioned(
        inputs,
        SelectionConfig(target_count=12, partition_count=12),
    )

    assert len(result.selected_tickers) == 12
    assert result.labels.nunique() == 12


def make_density_features() -> SelectionFeatures:
    coordinates = np.array([0.00, 0.01, 0.02, 0.03, 0.40, 0.41, 0.42, 0.43, 1.00, 3.00])
    tickers = [f"D{position:02d}" for position in range(len(coordinates))]
    distance_values = np.abs(coordinates[:, None] - coordinates[None, :]) / coordinates.max()
    features = pd.DataFrame(
        {
            "value_rank": np.linspace(0.1, 1.0, len(tickers)),
            "size_rank": np.linspace(1.0, 0.1, len(tickers)),
            "sharpe_rank": np.linspace(0.2, 0.9, len(tickers)),
            "trailing_sharpe": np.linspace(-0.5, 1.5, len(tickers)),
        },
        index=tickers,
    )
    base_score = (0.5 * features["value_rank"] + 0.5 * features["sharpe_rank"]).rename(
        "base_score"
    )
    correlation = 1.0 - 2.0 * distance_values**2
    np.fill_diagonal(correlation, 1.0)
    return SelectionFeatures(
        features=features,
        base_score=base_score,
        correlation=pd.DataFrame(correlation, index=tickers, columns=tickers),
        distance=pd.DataFrame(distance_values, index=tickers, columns=tickers),
    )


def test_density_selector_uses_precomputed_distance_and_keeps_noise_eligible():
    result = select_density(
        make_density_features(),
        SelectionConfig(target_count=8, partition_count=6, min_cluster_size=3, min_samples=3),
    )

    assert len(result.selected_tickers) == 8
    assert len(set(result.selected_tickers)) == 8
    assert -1 in set(result.labels)
    noise = set(result.labels.index[result.labels == -1])
    assert noise.intersection(result.selected_tickers)


def test_density_selector_is_deterministic_and_does_not_mutate_distance():
    inputs = make_density_features()
    original_distance = inputs.distance.copy()
    config = SelectionConfig(target_count=8, partition_count=6)

    first = select_density(inputs, config)
    second = select_density(inputs, config)

    assert first.selected_tickers == second.selected_tickers
    pd.testing.assert_series_equal(first.labels, second.labels)
    pd.testing.assert_frame_equal(inputs.distance, original_distance)


def make_rebalance_context(n_assets: int = 30) -> RebalanceContext:
    tickers = [f"T{position:02d}" for position in range(n_assets)]
    dates = pd.date_range("2018-09-14", periods=120, freq="W-FRI")
    phases = np.linspace(0.0, np.pi, n_assets)
    simple_returns = np.column_stack(
        [0.002 + 0.01 * np.sin(np.linspace(0.0, 8.0, len(dates)) + phase) for phase in phases]
    )
    training_returns = pd.DataFrame(np.log1p(simple_returns), index=dates, columns=tickers)
    covariance = pd.DataFrame(np.eye(n_assets) * 0.04, index=tickers, columns=tickers)
    expected_returns = pd.Series(np.linspace(0.05, 0.20, n_assets), index=tickers)
    return RebalanceContext(
        rebalance_date=pd.Timestamp("2021-01-01"),
        training_returns=training_returns,
        expected_returns=expected_returns,
        covariance=covariance,
    )


def make_point_in_time_fundamentals(n_assets: int = 30) -> pd.DataFrame:
    tickers = [f"T{position:02d}" for position in range(n_assets)]
    return pd.DataFrame(
        {
            "ticker": tickers,
            "observation_date": pd.Timestamp("2020-09-30"),
            "available_date": pd.Timestamp("2020-11-15"),
            "trailing_pe": np.linspace(10.0, 35.0, n_assets),
            "market_cap": np.geomspace(1e9, 1e12, n_assets),
            "earnings_positive": True,
            "source": "Synthetic",
        }
    )


def test_selection_strategy_returns_full_universe_capped_weights_and_audit():
    context = make_rebalance_context()
    strategy = SelectionStrategy(
        name="Partitioning Selection",
        fundamentals=make_point_in_time_fundamentals(),
        selector=select_partitioned,
        selection_config=SelectionConfig(),
    )

    weights = strategy(context, StrategyConfig(max_weight=0.25, risk_free_rate=0.04))

    assert list(weights.index) == list(context.covariance.index)
    assert np.isclose(weights.sum(), 1.0)
    assert (weights > 0).sum() == 12
    assert np.allclose(weights[weights > 0], 1.0 / 12.0)
    assert weights.max() <= 0.25
    audit = strategy.audit_frame()
    assert set(audit["strategy"]) == {"Partitioning Selection"}
    assert audit["selected"].sum() == 12
    assert (audit["available_date"] < audit["rebalance_date"]).all()


def test_future_fundamental_record_does_not_change_earlier_weights():
    context = make_rebalance_context()
    fundamentals = make_point_in_time_fundamentals()
    future_record = fundamentals.iloc[[0]].copy()
    future_record["observation_date"] = pd.Timestamp("2021-03-31")
    future_record["available_date"] = pd.Timestamp("2021-05-01")
    future_record["trailing_pe"] = 1.0
    with_future = pd.concat([fundamentals, future_record], ignore_index=True)
    config = StrategyConfig(max_weight=0.25, risk_free_rate=0.04)

    first = SelectionStrategy("Partitioning Selection", fundamentals, select_partitioned)(context, config)
    second = SelectionStrategy("Partitioning Selection", with_future, select_partitioned)(context, config)

    pd.testing.assert_series_equal(first, second)


def test_density_selector_handles_all_noise(monkeypatch):
    inputs = make_asset_features(12)

    def all_noise(self, values):
        return np.full(len(values), -1)

    monkeypatch.setattr("selection_strategies.HDBSCAN.fit_predict", all_noise)
    result = select_density(inputs, SelectionConfig(target_count=8, partition_count=6))

    assert len(result.selected_tickers) == 8
    assert set(result.labels) == {-1}
