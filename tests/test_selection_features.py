import numpy as np
import pandas as pd
import pytest

from config import (
    FUNDAMENTALS_PATH,
    SELECTION_CLUSTER_COUNT,
    SELECTION_CORRELATION_WEIGHT,
    SELECTION_DIVERSIFICATION_PENALTY,
    SELECTION_FEATURE_WEIGHT,
    SELECTION_LOOKBACK_WEEKS,
    SELECTION_MIN_CLUSTER_SIZE,
    SELECTION_MIN_SAMPLES,
    SELECTION_OUTPUT_DIR,
    SELECTION_TARGET_COUNT,
)
from selection_features import build_selection_features


def make_fundamental_snapshot(tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trailing_pe": np.linspace(10.0, 30.0, len(tickers)),
            "market_cap": np.geomspace(1e9, 1e11, len(tickers)),
            "earnings_positive": [True] * len(tickers),
        },
        index=tickers,
    )


def make_training_returns(tickers: list[str], periods: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2017-01-06", periods=periods, freq="W-FRI")
    values = {
        ticker: np.log1p(np.linspace(-0.01 + position * 0.001, 0.02 + position * 0.002, periods))
        for position, ticker in enumerate(tickers)
    }
    return pd.DataFrame(values, index=dates)


def test_selection_experiment_constants_are_preregistered():
    assert FUNDAMENTALS_PATH.name == "fundamentals_point_in_time.csv"
    assert SELECTION_OUTPUT_DIR.name == "selection_experiment"
    assert SELECTION_TARGET_COUNT == 12
    assert SELECTION_CLUSTER_COUNT == 6
    assert SELECTION_LOOKBACK_WEEKS == 104
    assert SELECTION_MIN_CLUSTER_SIZE == 3
    assert SELECTION_MIN_SAMPLES == 3
    assert SELECTION_FEATURE_WEIGHT == 0.50
    assert SELECTION_CORRELATION_WEIGHT == 0.50
    assert SELECTION_DIVERSIFICATION_PENALTY == 0.25


def test_feature_builder_uses_only_last_104_training_weeks():
    training = make_training_returns(["A", "B"])
    changed_old_history = training.copy()
    changed_old_history.iloc[:16] = np.log1p(0.50)
    fundamentals = make_fundamental_snapshot(["A", "B"])

    first = build_selection_features(training, fundamentals, lookback_weeks=104)
    second = build_selection_features(changed_old_history, fundamentals, lookback_weeks=104)

    pd.testing.assert_series_equal(first.base_score, second.base_score)
    pd.testing.assert_frame_equal(first.distance, second.distance)


def test_feature_ranks_follow_frozen_value_size_and_sharpe_rules():
    tickers = ["A", "B", "C", "D"]
    training = make_training_returns(tickers, periods=104)
    fundamentals = make_fundamental_snapshot(tickers)
    fundamentals.loc["D", ["earnings_positive", "trailing_pe"]] = [False, np.nan]

    selection = build_selection_features(training, fundamentals)

    assert selection.features.loc["A", "value_rank"] > selection.features.loc["B", "value_rank"]
    assert selection.features.loc["D", "value_rank"] == 0.0
    assert selection.features.loc["D", "size_rank"] > selection.features.loc["A", "size_rank"]
    assert selection.features.loc["D", "sharpe_rank"] > selection.features.loc["A", "sharpe_rank"]
    expected_score = 0.5 * selection.features["value_rank"] + 0.5 * selection.features["sharpe_rank"]
    pd.testing.assert_series_equal(selection.base_score, expected_score.rename("base_score"))


def test_distance_is_finite_symmetric_bounded_and_zero_diagonal():
    tickers = ["A", "B", "C", "D"]
    selection = build_selection_features(make_training_returns(tickers), make_fundamental_snapshot(tickers))

    assert np.isfinite(selection.distance.to_numpy()).all()
    assert np.allclose(selection.distance, selection.distance.T)
    assert np.allclose(np.diag(selection.distance), 0.0)
    assert selection.distance.to_numpy().min() >= 0.0
    assert selection.distance.to_numpy().max() <= 1.0
    assert list(selection.distance.index) == sorted(tickers)


def test_feature_builder_rejects_short_history():
    with pytest.raises(ValueError, match="104 observations"):
        build_selection_features(
            make_training_returns(["A", "B"], periods=103),
            make_fundamental_snapshot(["A", "B"]),
        )


def test_feature_builder_rejects_zero_volatility():
    training = make_training_returns(["A", "B"], periods=104)
    training["A"] = np.log1p(0.01)

    with pytest.raises(ValueError, match="volatility"):
        build_selection_features(training, make_fundamental_snapshot(["A", "B"]))
