"""Leakage-safe features and distances for stock-selection strategies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from config import (
    RISK_FREE_RATE,
    SELECTION_CORRELATION_WEIGHT,
    SELECTION_FEATURE_WEIGHT,
    SELECTION_LOOKBACK_WEEKS,
    WEEKS_PER_YEAR,
)


@dataclass(frozen=True)
class SelectionFeatures:
    """Cross-sectional features, scores, correlation, and mixed distances."""

    features: pd.DataFrame
    base_score: pd.Series
    correlation: pd.DataFrame
    distance: pd.DataFrame


def _validate_inputs(
    training_log_returns: pd.DataFrame,
    fundamentals: pd.DataFrame,
    lookback_weeks: int,
) -> list[str]:
    if lookback_weeks < 2:
        raise ValueError("lookback_weeks must be at least 2")
    if len(training_log_returns) < lookback_weeks:
        raise ValueError(
            f"Selection features require {lookback_weeks} observations; "
            f"received {len(training_log_returns)}"
        )
    if not fundamentals.index.is_unique:
        raise ValueError("Fundamental snapshot index must contain unique tickers")

    tickers = sorted(fundamentals.index.intersection(training_log_returns.columns).astype(str))
    if len(tickers) != len(fundamentals):
        missing = sorted(set(fundamentals.index.astype(str)).difference(tickers))
        raise ValueError(f"Training returns are missing fundamental tickers: {missing}")
    if len(tickers) < 2:
        raise ValueError("Selection features require at least two assets")
    return tickers


def _build_ranked_features(
    simple_returns: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    annualized_return = simple_returns.mean() * WEEKS_PER_YEAR
    annualized_volatility = simple_returns.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR)
    volatility_values = annualized_volatility.to_numpy(dtype=float)
    if not np.isfinite(volatility_values).all() or np.isclose(volatility_values, 0.0, atol=1e-12).any():
        raise ValueError("Trailing return volatility must be finite and positive for every asset")

    trailing_sharpe = (annualized_return - RISK_FREE_RATE) / annualized_volatility
    if not np.isfinite(trailing_sharpe.to_numpy(dtype=float)).all():
        raise ValueError("Trailing Sharpe must be finite for every asset")

    profitable = fundamentals["earnings_positive"].astype(bool)
    value_rank = pd.Series(0.0, index=fundamentals.index, dtype=float)
    profitable_pe = fundamentals.loc[profitable, "trailing_pe"]
    value_rank.loc[profitable] = profitable_pe.rank(ascending=False, method="average", pct=True)

    market_cap = fundamentals["market_cap"].astype(float)
    if not np.isfinite(market_cap.to_numpy()).all() or (market_cap <= 0).any():
        raise ValueError("market_cap must be finite and positive")
    size_rank = np.log(market_cap).rank(ascending=True, method="average", pct=True)
    sharpe_rank = trailing_sharpe.rank(ascending=True, method="average", pct=True)

    return pd.DataFrame(
        {
            "value_rank": value_rank,
            "size_rank": size_rank,
            "sharpe_rank": sharpe_rank,
            "trailing_sharpe": trailing_sharpe,
        },
        index=fundamentals.index,
    )


def _build_correlation(simple_returns: pd.DataFrame) -> pd.DataFrame:
    covariance = LedoitWolf().fit(simple_returns.to_numpy()).covariance_
    standard_deviation = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(standard_deviation, standard_deviation)
    correlation = np.clip(correlation, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return pd.DataFrame(correlation, index=simple_returns.columns, columns=simple_returns.columns)


def _build_mixed_distance(features: pd.DataFrame, correlation: pd.DataFrame) -> pd.DataFrame:
    correlation_distance = np.sqrt(0.5 * (1.0 - correlation.to_numpy()))
    ranked_values = features[["value_rank", "size_rank", "sharpe_rank"]].to_numpy()
    feature_delta = ranked_values[:, np.newaxis, :] - ranked_values[np.newaxis, :, :]
    feature_distance = np.linalg.norm(feature_delta, axis=2) / np.sqrt(3.0)
    mixed_distance = (
        SELECTION_CORRELATION_WEIGHT * correlation_distance
        + SELECTION_FEATURE_WEIGHT * feature_distance
    )
    mixed_distance = np.clip(mixed_distance, 0.0, 1.0)
    np.fill_diagonal(mixed_distance, 0.0)
    return pd.DataFrame(mixed_distance, index=features.index, columns=features.index)


def build_selection_features(
    training_log_returns: pd.DataFrame,
    fundamentals: pd.DataFrame,
    lookback_weeks: int = SELECTION_LOOKBACK_WEEKS,
) -> SelectionFeatures:
    """Create leakage-safe cross-sectional features and a mixed distance matrix."""
    tickers = _validate_inputs(training_log_returns, fundamentals, lookback_weeks)
    aligned_fundamentals = fundamentals.loc[tickers].copy()
    trailing_log_returns = training_log_returns.loc[:, tickers].tail(lookback_weeks)
    if trailing_log_returns.isna().any().any():
        raise ValueError("Trailing training returns must not contain missing values")

    simple_returns = np.expm1(trailing_log_returns)
    if not np.isfinite(simple_returns.to_numpy()).all():
        raise ValueError("Trailing training returns must be finite")

    features = _build_ranked_features(simple_returns, aligned_fundamentals)
    base_score = (
        SELECTION_FEATURE_WEIGHT * features["value_rank"]
        + SELECTION_FEATURE_WEIGHT * features["sharpe_rank"]
    ).rename("base_score")
    correlation = _build_correlation(simple_returns)
    distance = _build_mixed_distance(features, correlation)

    return SelectionFeatures(
        features=features,
        base_score=base_score,
        correlation=correlation,
        distance=distance,
    )
