# Stock-Selection Experiment Specification

**Frozen:** 2026-08-21

**Status:** Preregistered; implementation and data-source approval pending

## Purpose

Compare two point-in-time stock selectors within the existing historical walk-forward backtest:

1. deterministic Partitioning Around Medoids (PAM); and
2. density-based HDBSCAN.

Each selector chooses exactly 12 of the existing 30 equities and assigns equal weights. The experiment evaluates whether selection adds value relative to Equal Weight and SPY without changing the existing allocation strategies or retrospective universe.

Changing any frozen parameter, formula, target count, gate, or robustness rule requires a new dated plan. Results from a changed specification are exploratory and cannot be presented as preregistered results.

## Pre-Run Source Amendment

The dated SEC+yfinance source plan adds a fair `Eligible Universe Equal Weight` baseline because automated point-in-time coverage may exclude foreign or OTC names. It also requires structured SEC/Yahoo provenance, explicit share-class reconciliation, unadjusted prices for market-cap reconstruction, and an explicitly supplied SEC User-Agent. This amendment was frozen before any historical selection run.

## Research Boundaries

- Run implementation in `D:\sin_stoks-strategy-lab` on `codex/partition-density-strategies`, based on committed `main`.
- Preserve the authoritative retrospective universe of 30 equities.
- Preserve annual January rebalances for 2020-2025, weekly returns, a 4% annual risk-free rate, a 25% maximum position, and 10 bps one-way drift-aware transaction costs.
- Treat the work as a historical walk-forward backtest, not a strict constituent-level out-of-sample test or investment recommendation.
- Keep the existing `run_walk_forward_backtest(...)` public signature and all existing strategy behavior backward compatible.
- Write experiment artifacts only under `outputs/selection_experiment/`. Do not overwrite `outputs/portfolio_backtest/` or `outputs/report/`.
- Do not tune feature weights, cluster parameters, cluster count, target count, or other hyperparameters against returns.
- Do not add risk-parity or maximum-Sharpe allocation to the primary comparison.
- Do not merge into a dirty main worktree. Merge and worktree cleanup each require explicit approval.

## Point-in-Time Fundamental Data Contract

The runtime input is `data/fundamentals_point_in_time.csv`, with core columns:

- `ticker`
- `observation_date`
- `available_date`
- `trailing_pe`
- `market_cap`
- `earnings_positive`
- `source`

Free-source records must also preserve structured provenance for the rebalance date, CIK, SEC filing/accession and fact periods, Yahoo price symbol/date/currency, historical FX rates, filed shares, share reconciliation, TTM earnings, and calculation method.

Only records with `available_date < rebalance_date` are eligible. An observation date or fiscal-period date alone is not evidence that information was available. Current P/E or market-cap values must not be substituted for historical values.

The experiment must not download external data or run on historical data until the source and licensing terms are presented and explicitly approved. It must not silently impute missing fundamentals. Every rebalance requires at least 80% universe coverage and at least 12 eligible securities.

## Frozen Parameters

| Parameter | Value |
|---|---:|
| Selected equities | 12 |
| PAM partitions | 6 |
| Trailing return window | 104 weeks |
| HDBSCAN `min_cluster_size` | 3 |
| HDBSCAN `min_samples` | 3 |
| Feature-distance weight | 0.50 |
| Correlation-distance weight | 0.50 |
| Diversification penalty | 0.25 |
| Minimum fundamental coverage | 0.80 |
| Annualization periods | 52 |
| Risk-free rate | 0.04 |
| Maximum position | 0.25 |
| Transaction cost | 10 bps |

## Shared Feature Pipeline

At each rebalance, align the validated fundamental snapshot with training returns and sort tickers lexicographically. Use only the final 104 rows of pre-rebalance weekly log returns. Convert log returns to arithmetic returns with `expm1`.

Calculate annualized arithmetic return and sample volatility using 52 periods per year. Define trailing Sharpe as:

```text
trailing_sharpe = (annualized_return - 0.04) / annualized_volatility
```

Reject zero or non-finite volatility.

Cross-sectional features are:

```text
value_rank  = descending percentile rank of trailing_pe among profitable firms;
              non-positive earners receive 0.0
size_rank   = percentile rank of log(market_cap)
sharpe_rank = percentile rank of trailing 104-week Sharpe
base_score  = 0.50 * value_rank + 0.50 * sharpe_rank
```

Size contributes to distance but does not directly increase `base_score`.

Fit Ledoit-Wolf covariance on the same 104-week training window and convert it to correlation. Clip correlations to `[-1, 1]`. Define distances as:

```text
correlation_distance(i,j) = sqrt(0.5 * (1 - correlation(i,j)))
feature_distance(i,j)     = Euclidean([value_rank,size_rank,sharpe_rank]_i,
                                      [value_rank,size_rank,sharpe_rank]_j) / sqrt(3)
mixed_distance(i,j)       = 0.50 * correlation_distance(i,j)
                            + 0.50 * feature_distance(i,j)
```

The mixed-distance matrix must be finite and symmetric, remain within `[0, 1]`, and have an exact zero diagonal.

## Deterministic PAM Selector

Use true PAM BUILD/SWAP optimization on the mixed-distance matrix:

1. Sort rows and columns by ticker.
2. Choose the first medoid with the lowest total distance.
3. Add BUILD medoids by the greatest reduction in nearest-medoid cost.
4. Evaluate every medoid/non-medoid swap and apply the one with the greatest strict cost reduction.
5. Stop when no strict improvement remains or after 100 iterations.
6. Break equal-cost choices lexicographically.
7. Assign each ticker to its nearest medoid and order integer labels by medoid ticker.

Run PAM with six partitions. Select up to the two highest-`base_score` members in each partition, breaking ties lexicographically. If this yields fewer than 12 names, repeatedly select the remaining candidate maximizing:

```text
adjusted_score(candidate) = base_score(candidate)
                            - 0.25 * mean_correlation(candidate, already_selected)
```

If no ticker is selected, seed with the highest-`base_score` ticker. The final set must contain exactly 12 unique tickers.

## HDBSCAN Selector

Fit scikit-learn HDBSCAN only on the shared precomputed mixed-distance matrix using:

```python
HDBSCAN(
    min_cluster_size=3,
    min_samples=3,
    metric="precomputed",
    cluster_selection_method="eom",
    allow_single_cluster=True,
    store_centers=None,
)
```

Do not search epsilon or tune clustering parameters against returns.

Select the highest-`base_score` member of every non-noise cluster. Noise observations (`label == -1`) remain eligible singleton candidates. Fill open positions with the same diversification-adjusted greedy rule used by PAM. If every observation is noise, seed with the highest-`base_score` ticker and fill greedily. The final set must contain exactly 12 unique tickers with lexicographic tie-breaking.

## Portfolio Construction and Comparators

PAM and HDBSCAN selections receive equal weights. The primary comparison includes:

- PAM-selected Equal Weight;
- HDBSCAN-selected Equal Weight;
- Eligible Universe Equal Weight, rebuilt from the exact valid fundamental snapshot at each rebalance;
- full-universe Equal Weight; and
- SPY.

The engine must preserve point-in-time estimation, annual January rebalancing, buy-and-hold weight drift, one-way turnover, and a single transaction-cost deduction after each rebalance.

## Preregistered CELH Robustness Run

Repeat the complete experiment after excluding CELH. This is the only preregistered ticker-exclusion robustness run. It does not authorize excluding any other influential observation. Compare each selector with the corresponding ex-CELH Equal Weight portfolio.

## Outputs and Diagnostics

Persist isolated full-universe and ex-CELH inputs, weekly net returns, target weights, turnover, selection records, cluster labels, feature snapshots, coverage records, run metadata, and input hashes under `outputs/selection_experiment/`.

Reporting must include:

- standard net performance metrics;
- compounded calendar-year returns and annual excess returns;
- information ratio versus Equal Weight;
- recurring one-way turnover excluding initial cash;
- ticker selection frequency;
- year-to-year Jaccard similarity;
- effective holdings, `1 / sum(weights ** 2)`;
- fundamental coverage by rebalance; and
- full-universe versus ex-CELH metric deltas.

Do not calculate or imply reliable p-values from six annual observations.

## Promotion Gates

Evaluate PAM and HDBSCAN independently. A selector is **research-promising** only if every condition passes:

1. Full-universe net CAGR exceeds full-universe Equal Weight, Eligible Universe Equal Weight, and SPY.
2. Calendar-year return exceeds Eligible Universe Equal Weight in at least four of six years.
3. Information ratio versus Eligible Universe Equal Weight is positive.
4. Maximum drawdown is no more than 3 percentage points worse than Eligible Universe Equal Weight.
5. Average recurring one-way turnover is at most 60%.
6. Ex-CELH net CAGR exceeds both corresponding ex-CELH full-universe and eligible-universe Equal Weight CAGRs.
7. Fundamental coverage is at least 80% at every rebalance.
8. All integrity, test, lint, formatting, and diff checks pass.

Record each gate and its supporting value in `decision.json`. A failed research gate does not invalidate the run; classify the selector as feasible-but-not-promising or data-infeasible as appropriate.

Any report must lead with the gate outcome and disclose the retrospective universe, limited six-year evaluation, missing fundamental coverage, and CELH sensitivity. Even a passing result may only be described as a **“promising historical walk-forward result.”**
