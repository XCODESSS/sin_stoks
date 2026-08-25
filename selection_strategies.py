"""Deterministic clustering-based stock-selection strategies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN

from backtest_engine import RebalanceContext
from config import (
    SELECTION_CLUSTER_COUNT,
    SELECTION_DIVERSIFICATION_PENALTY,
    SELECTION_MIN_CLUSTER_SIZE,
    SELECTION_MIN_SAMPLES,
    SELECTION_TARGET_COUNT,
)
from fundamental_data import fundamentals_as_of
from portfolio_strategies import StrategyConfig, equal_weight, validate_projected_weights
from selection_features import SelectionFeatures, build_selection_features

_PAM_MAX_ITERATIONS = 100
_COST_TOLERANCE = 1e-12


@dataclass(frozen=True)
class SelectionConfig:
    """Frozen clustering and selection parameters."""

    target_count: int = SELECTION_TARGET_COUNT
    partition_count: int = SELECTION_CLUSTER_COUNT
    min_cluster_size: int = SELECTION_MIN_CLUSTER_SIZE
    min_samples: int = SELECTION_MIN_SAMPLES
    diversification_penalty: float = SELECTION_DIVERSIFICATION_PENALTY


@dataclass(frozen=True)
class SelectionResult:
    """Selected tickers with cluster labels and auditable scores."""

    selected_tickers: list[str]
    labels: pd.Series
    adjusted_scores: pd.Series


SelectorCallable = Callable[[SelectionFeatures, SelectionConfig], SelectionResult]


class SelectionStrategy:
    """Contextual equal-weight selector with point-in-time audit records."""

    def __init__(
        self,
        name: str,
        fundamentals: pd.DataFrame,
        selector: SelectorCallable,
        selection_config: SelectionConfig | None = None,
    ) -> None:
        self.name = name
        self._fundamentals = fundamentals.copy()
        self._selector = selector
        self._selection_config = selection_config or SelectionConfig()
        self._audit_rows: list[dict[str, object]] = []

    def __call__(self, context: RebalanceContext, config: StrategyConfig) -> pd.Series:
        snapshot = fundamentals_as_of(
            self._fundamentals,
            context.rebalance_date,
            context.covariance.index,
        )
        eligible_tickers = list(snapshot.index)
        selection_inputs = build_selection_features(
            context.training_returns.loc[:, eligible_tickers],
            snapshot,
        )
        selection = self._selector(selection_inputs, self._selection_config)
        selected_tickers = selection.selected_tickers
        selected_weights = equal_weight(
            context.expected_returns.loc[selected_tickers],
            context.covariance.loc[selected_tickers, selected_tickers],
            config,
        )
        weights = selected_weights.reindex(context.covariance.index, fill_value=0.0)
        validate_projected_weights(weights.to_numpy(), config.max_weight, self.name)
        self._append_audit(context.rebalance_date, snapshot, selection_inputs, selection)
        return weights

    def _append_audit(
        self,
        rebalance_date: pd.Timestamp,
        snapshot: pd.DataFrame,
        inputs: SelectionFeatures,
        selection: SelectionResult,
    ) -> None:
        selected = set(selection.selected_tickers)
        for ticker in inputs.features.index:
            feature_row = inputs.features.loc[ticker]
            self._audit_rows.append(
                {
                    "rebalance_date": rebalance_date,
                    "strategy": self.name,
                    "ticker": ticker,
                    "selected": ticker in selected,
                    "cluster_label": int(selection.labels.loc[ticker]),
                    "trailing_pe": snapshot.loc[ticker, "trailing_pe"],
                    "market_cap": snapshot.loc[ticker, "market_cap"],
                    "available_date": snapshot.loc[ticker, "available_date"],
                    "value_rank": feature_row["value_rank"],
                    "size_rank": feature_row["size_rank"],
                    "sharpe_rank": feature_row["sharpe_rank"],
                    "base_score": inputs.base_score.loc[ticker],
                    "adjusted_score": selection.adjusted_scores.loc[ticker],
                }
            )

    def audit_frame(self) -> pd.DataFrame:
        """Return a defensive, deterministically ordered audit copy."""
        audit = pd.DataFrame(self._audit_rows)
        if audit.empty:
            return audit
        return audit.sort_values(["rebalance_date", "strategy", "ticker"]).reset_index(drop=True).copy()


def _validated_distance(distance: pd.DataFrame) -> pd.DataFrame:
    if distance.empty or not distance.index.is_unique or not distance.columns.is_unique:
        raise ValueError("Distance matrix must have unique assets")
    if set(distance.index) != set(distance.columns):
        raise ValueError("Distance matrix rows and columns must contain the same assets")

    ordered = sorted(distance.index.astype(str))
    matrix = distance.loc[ordered, ordered].astype(float)
    values = matrix.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Distance matrix must be finite")
    if not np.allclose(values, values.T, atol=_COST_TOLERANCE, rtol=0.0):
        raise ValueError("Distance matrix must be symmetric")
    if not np.allclose(np.diag(values), 0.0, atol=_COST_TOLERANCE, rtol=0.0):
        raise ValueError("Distance matrix diagonal must be zero")
    if (values < -_COST_TOLERANCE).any():
        raise ValueError("Distance matrix must be non-negative")
    return matrix


def _assignment_cost(distance: pd.DataFrame, medoids: list[str]) -> float:
    return float(distance.loc[:, medoids].min(axis=1).sum())


def _build_medoids(distance: pd.DataFrame, n_clusters: int) -> list[str]:
    total_distance = distance.sum(axis=1)
    first_cost = total_distance.min()
    first_medoid = min(total_distance.index[np.isclose(total_distance, first_cost, atol=_COST_TOLERANCE)])
    medoids = [first_medoid]

    while len(medoids) < n_clusters:
        current_cost = _assignment_cost(distance, medoids)
        candidates = sorted(set(distance.index).difference(medoids))
        reductions = {
            candidate: current_cost - _assignment_cost(distance, [*medoids, candidate])
            for candidate in candidates
        }
        best_reduction = max(reductions.values())
        best_candidate = min(
            candidate
            for candidate, reduction in reductions.items()
            if np.isclose(reduction, best_reduction, atol=_COST_TOLERANCE)
        )
        medoids.append(best_candidate)
    return medoids


def _optimize_medoids(distance: pd.DataFrame, initial_medoids: list[str]) -> list[str]:
    medoids = list(initial_medoids)
    for _ in range(_PAM_MAX_ITERATIONS):
        current_cost = _assignment_cost(distance, medoids)
        swaps: list[tuple[float, str, str, list[str]]] = []
        non_medoids = sorted(set(distance.index).difference(medoids))
        for old_medoid in sorted(medoids):
            for new_medoid in non_medoids:
                proposed = [new_medoid if ticker == old_medoid else ticker for ticker in medoids]
                reduction = current_cost - _assignment_cost(distance, proposed)
                swaps.append((reduction, old_medoid, new_medoid, proposed))

        if not swaps:
            break
        best_reduction = max(swap[0] for swap in swaps)
        if best_reduction <= _COST_TOLERANCE:
            break
        best_swap = min(
            (swap for swap in swaps if np.isclose(swap[0], best_reduction, atol=_COST_TOLERANCE)),
            key=lambda swap: (swap[1], swap[2]),
        )
        medoids = best_swap[3]
    return sorted(medoids)


def pam_labels(distance: pd.DataFrame, n_clusters: int) -> pd.Series:
    """Partition a precomputed distance matrix using deterministic PAM BUILD/SWAP."""
    matrix = _validated_distance(distance)
    if n_clusters < 1:
        raise ValueError("n_clusters must be positive")
    if n_clusters > len(matrix):
        raise ValueError("n_clusters cannot exceed the number of assets")

    medoids = _optimize_medoids(matrix, _build_medoids(matrix, n_clusters))
    nearest_medoid = matrix.loc[:, medoids].idxmin(axis=1)
    label_by_medoid = {medoid: label for label, medoid in enumerate(medoids)}
    return nearest_medoid.map(label_by_medoid).astype(int).rename("cluster")


def _validate_selection_inputs(inputs: SelectionFeatures, config: SelectionConfig) -> None:
    n_assets = len(inputs.distance)
    if config.target_count < 1 or config.target_count > n_assets:
        raise ValueError("target_count must be positive and cannot exceed the number of assets")
    if config.partition_count < 1 or config.partition_count > n_assets:
        raise ValueError("partition_count must be positive and cannot exceed the number of assets")
    if config.min_cluster_size < 1 or config.min_samples < 1:
        raise ValueError("Cluster size parameters must be positive")
    if config.diversification_penalty < 0:
        raise ValueError("diversification_penalty must be non-negative")
    _validated_distance(inputs.distance)


def _rank_candidates(scores: pd.Series, candidates: list[str]) -> list[str]:
    return sorted(candidates, key=lambda ticker: (-float(scores.loc[ticker]), ticker))


def _fill_diversified(
    selected: list[str],
    candidates: list[str],
    inputs: SelectionFeatures,
    config: SelectionConfig,
    adjusted_scores: pd.Series,
) -> list[str]:
    remaining = set(candidates).difference(selected)
    if not selected and remaining:
        seed = _rank_candidates(inputs.base_score, list(remaining))[0]
        selected.append(seed)
        remaining.remove(seed)

    while len(selected) < config.target_count and remaining:
        candidate_scores = {
            ticker: float(inputs.base_score.loc[ticker])
            - config.diversification_penalty
            * float(inputs.correlation.loc[ticker, selected].mean())
            for ticker in remaining
        }
        best_score = max(candidate_scores.values())
        chosen = min(
            ticker
            for ticker, score in candidate_scores.items()
            if np.isclose(score, best_score, atol=_COST_TOLERANCE)
        )
        adjusted_scores.loc[chosen] = candidate_scores[chosen]
        selected.append(chosen)
        remaining.remove(chosen)

    if len(selected) != config.target_count:
        raise ValueError(f"Selector produced {len(selected)} assets; expected {config.target_count}")
    return selected


def select_partitioned(
    inputs: SelectionFeatures,
    config: SelectionConfig | None = None,
) -> SelectionResult:
    """Select high-scoring representatives from deterministic PAM partitions."""
    config = config or SelectionConfig()
    _validate_selection_inputs(inputs, config)
    labels = pam_labels(inputs.distance, config.partition_count)
    selected: list[str] = []
    adjusted_scores = inputs.base_score.copy().rename("adjusted_score")

    for label in sorted(labels.unique()):
        members = list(labels.index[labels == label])
        for ticker in _rank_candidates(inputs.base_score, members)[:2]:
            if len(selected) == config.target_count:
                break
            selected.append(ticker)

    selected = _fill_diversified(
        selected,
        list(inputs.distance.index),
        inputs,
        config,
        adjusted_scores,
    )
    return SelectionResult(selected_tickers=selected, labels=labels, adjusted_scores=adjusted_scores)


def select_density(
    inputs: SelectionFeatures,
    config: SelectionConfig | None = None,
) -> SelectionResult:
    """Select cluster representatives and diversified HDBSCAN candidates."""
    config = config or SelectionConfig()
    _validate_selection_inputs(inputs, config)
    distance = _validated_distance(inputs.distance)
    model = HDBSCAN(
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        metric="precomputed",
        cluster_selection_method="eom",
        allow_single_cluster=True,
        store_centers=None,
    )
    labels = pd.Series(model.fit_predict(distance.to_numpy()), index=distance.index, name="cluster")
    adjusted_scores = inputs.base_score.copy().rename("adjusted_score")

    representatives: list[str] = []
    for label in sorted(value for value in labels.unique() if value != -1):
        members = list(labels.index[labels == label])
        representatives.append(_rank_candidates(inputs.base_score, members)[0])
    selected = _rank_candidates(inputs.base_score, representatives)[: config.target_count]
    selected = _fill_diversified(
        selected,
        list(distance.index),
        inputs,
        config,
        adjusted_scores,
    )
    return SelectionResult(selected_tickers=selected, labels=labels, adjusted_scores=adjusted_scores)
