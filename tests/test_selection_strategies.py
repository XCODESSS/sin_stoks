from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from selection_features import SelectionFeatures
from selection_strategies import SelectionConfig, pam_labels, select_density, select_partitioned


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


def test_density_selector_handles_all_noise(monkeypatch):
    inputs = make_asset_features(12)

    def all_noise(self, values):
        return np.full(len(values), -1)

    monkeypatch.setattr("selection_strategies.HDBSCAN.fit_predict", all_noise)
    result = select_density(inputs, SelectionConfig(target_count=8, partition_count=6))

    assert len(result.selected_tickers) == 8
    assert set(result.labels) == {-1}
