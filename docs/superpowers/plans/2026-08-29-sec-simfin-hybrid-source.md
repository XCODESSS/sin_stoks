# SEC + SimFin Hybrid Point-in-Time Source Amendment

**Status:** Frozen before any PAM/HDBSCAN historical return run.

## Goal

Maximize defensible coverage of the frozen 30-stock universe without current fundamentals or silent imputation. Retain SEC filing chronology for earnings and use SimFin only for narrowly defined market-reference gaps.

## Frozen Source Precedence

1. Earnings remain SEC Company Facts observations filed strictly before rebalance.
2. Standard SEC instant share tags remain first choice.
3. SNAP may fall back to `us-gaap:SharesOutstanding`.
4. META may fall back to filed `us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding`, labeled as an approximate market-cap denominator.
5. SimFin may provide EA's missing unadjusted close and STZ's missing historical shares. SimFin values are never used for earnings.
6. Known annual/fiscal filing patterns use issuer-specific maximum share ages: BF-B 280 days, BUD 400 days, BTI 400 days, QSR 280 days; all others remain 200 days.
7. Every fallback is retained in structured provenance (`price_source`, `shares_source`, fact tag, method, dates).
8. SimFin API credentials come only from `SIMFIN_API_KEY`, are never printed, cached, hashed into manifests, or committed.
9. The five preregistered manual-only names remain missing unless a later plan adds issuer-report extraction.

## Decision Rule

- First rerun the six-date source coverage probe.
- Use the strict experiment when all dates have at least 24/30 names.
- If strict coverage still fails, the user has authorized an explicitly labeled exploratory reduced-universe run using the eligible 18–19 names. It must retain the eligible-universe equal-weight comparator and must not be described as the preregistered full-universe experiment.
- No selector parameter, selected count, or promotion threshold may be tuned after returns are observed.

## Required Verification

- Unit tests for SEC tag fallbacks, issuer-specific staleness, SimFin cache/chronology, and hybrid provenance.
- No source observation on or after rebalance.
- At least 12 eligible names at every rebalance.
- Full tests, Ruff, and `git diff --check` before historical execution.
