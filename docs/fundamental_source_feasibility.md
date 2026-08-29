# Point-in-Time Fundamental Source Feasibility Gate

**Assessment date:** 2026-08-25

The stock-selection experiment requires historical trailing P/E, historical market capitalization, and a defensible `available_date` strictly before each January rebalance from 2020 through 2025. Coverage must include all 30 retrospective-universe securities, including ADR and OTC names, with reproducible access and licensing suitable for this project.

No external dataset has been downloaded and no vendor-derived values have been added to the repository.

## Candidate comparison

| Candidate | Historical fundamentals | Defensible availability timestamp | Historical market cap | ADR/OTC coverage | Reproducible access | Licensing | Gate assessment |
|---|---|---|---|---|---|---|---|
| Nasdaq Data Link Sharadar SF1 + SEP | Point-in-time fundamental dimensions and prices are designed for historical research | SF1 exposes filing/date-key fields that may support an availability rule | Can be derived from approved point-in-time shares/prices or an applicable Sharadar field | Must be verified ticker-by-ticker, especially PRNDY, TCEHY, NTDOY, CCOEY, and UBSFY | Paid API/export | Commercial subscription; redistribution restrictions apply | **Best pilot candidate, not yet approved or coverage-verified** |
| S&P Compustat/Capital IQ | Institutional point-in-time financial and market datasets | Filing/publication metadata is available in appropriate licensed products | Available | Broad global/ADR coverage; exact OTC symbols require mapping | Licensed API/export | Commercial and generally non-redistributable | **Likely technically feasible if the user supplies licensed access/export** |
| FactSet | Institutional point-in-time fundamentals and market data | Point-in-time packages provide availability controls | Available | Broad global coverage; exact universe mapping must be verified | Licensed API/export | Commercial and generally non-redistributable | **Likely technically feasible if the user supplies licensed access/export** |
| SimFin | Historical statements and publish dates are available for covered companies | Publish dates can support chronology for covered filings | Historical shares/prices may permit derivation | Coverage of all foreign ADR/OTC names is not established | API/bulk export | Plan-dependent terms and attribution requirements | **Unconfirmed; requires a 30-ticker coverage proof** |
| SEC EDGAR Company Facts + market prices | Filing facts are historical | SEC acceptance timestamps are defensible | Not supplied directly; must be reconstructed from contemporaneous shares and prices | Incomplete for several foreign issuers/OTC instruments and accounting mappings | Public API | Public-data terms and SEC fair-access policy | **Does not satisfy the complete contract by itself** |
| Financial Modeling Prep historical ratios/market cap | Historical ratio and market-cap endpoints exist | Ratio dates do not by themselves prove when every value became public | Available through historical endpoints | Coverage must be verified | API | Plan-dependent commercial terms | **Fails until an explicit point-in-time availability field and full coverage are proven** |
| Current `yfinance` company information | Current snapshot only | No historical availability timestamp | Current value only | Mixed | API-like access | Upstream terms apply | **Rejected by the frozen plan** |

## Required approval package

Before a historical run, the approved input or extraction procedure must provide:

1. a ticker mapping for all 30 securities;
2. source and licence/redistribution terms;
3. records covering the 2019-2024 information vintages needed for 2020-2025 rebalances;
4. `observation_date` and a defensible `available_date` for every record;
5. historical trailing P/E and historical market capitalization without current-value substitution;
6. at least 80% universe coverage and at least 12 eligible securities at every rebalance; and
7. a reproducible export or API procedure whose raw restricted data need not be committed.

## Current decision

The user approved an automated **SEC EDGAR Company Facts + yfinance unadjusted primary-listing price/FX coverage pilot**. The implementation now reconstructs TTM earnings and reconciled filed shares using only facts filed before each rebalance, retains structured provenance, and compares selectors with an eligible-universe baseline to expose source-coverage bias.

The live source probe has not run because SEC fair-access requests require an explicitly supplied contact-bearing `SEC_USER_AGENT`; the implementation never derives or transmits the Git email automatically. Historical selector metrics remain `N/A` until the source probe proves at least 24 of 30 eligible names at every rebalance and the user separately approves the backtest command.

If the automated pilot fails coverage or share/currency reconciliation, the experiment remains data-infeasible. Current values, revised future filings, ambiguous share aggregation, and silent manual fills remain prohibited.

## Automated SEC/Yahoo coverage result

The approved source probe completed without invoking a backtest. Methodology: `sec-yfinance-v1`. Source manifest SHA-256: `a87ac6afa3f9eeb2439f00f8ee6bd7c0960b3426fa7fb1dea1f483633358a1ed`.

| Rebalance | Eligible | Coverage | 80% gate |
|---|---:|---:|:---:|
| 2020-01-01 | 18 | 60.00% | Fail |
| 2021-01-01 | 18 | 60.00% | Fail |
| 2022-01-01 | 18 | 60.00% | Fail |
| 2023-01-01 | 18 | 60.00% | Fail |
| 2024-01-01 | 19 | 63.33% | Fail |
| 2025-01-01 | 18 | 60.00% | Fail |

Repeated automated failures were:

- BF-B, BUD, and BTI: filed-share observations exceeded the frozen staleness limit;
- EA: Yahoo no longer returned the required unadjusted historical close series;
- META and early SNAP years: SEC Company Facts lacked eligible point-in-time cover shares under the frozen tags;
- STZ: SEC Company Facts lacked eligible shares under both frozen standard tags;
- QSR in 2025: filed shares exceeded the staleness limit; and
- PRNDY, TCEHY, NTDOY, CCOEY, and UBSFY: preregistered manual-only instruments.

**Decision: data-infeasible.** The 24-of-30 gate failed in every year, so PAM/HDBSCAN historical returns and real-data graphs were not generated. Changing staleness limits, using current Yahoo shares, substituting adjusted prices, or adding post-result manual values would violate the frozen source contract. Any broader issuer-report/manual-source effort requires a separately dated plan.

## Financial Modeling Prep access pilot

A user-supplied FMP key was tested transiently and was not written to source code, artifacts, or Git. Authentication succeeded on the profile endpoint, but the current subscription returned HTTP 402 for historical statements or market capitalization for BF-B, BUD, BTI, EA, STZ, QSR, PRNDY, TCEHY, NTDOY, CCOEY, and UBSFY. META and SNAP exposed only 64 recent market-cap observations (2026-05-29 through 2026-08-28), not the required 2019-2024 history; their historical income statements were also restricted. Legacy v3 endpoints returned HTTP 403.

Therefore, the current FMP subscription cannot satisfy the point-in-time contract. A plan that unlocks historical statements and historical market capitalization, including the international/OTC symbols, would need a new coverage pilot before use.

## SimFin access pilot

A user-supplied SimFin key was tested transiently and was not written to source code, artifacts, or Git. Authentication succeeded. The price endpoint returned unadjusted closes and `Common Shares Outstanding` from 2019-08-29 through 2025-01-02 for BUD, EA, META, SNAP, STZ, QSR, and TCEHY; BTI had prices throughout but shares only from late 2021. BF-B/BF.B, PRNDY, NTDOY, CCOEY, and UBSFY had no company records.

Income statements with publication dates were available for EA, META, SNAP, STZ, and QSR, but every returned row was marked `Restated`; these values are not accepted as point-in-time earnings without proof that the published-date/value pairing preserves the originally reported observation. BUD and BTI statement coverage was annual and insufficient in early rebalances.

SimFin is therefore promising as a **market-reference fallback**, especially for EA's missing Yahoo prices and STZ's historical shares, but it cannot replace SEC filing chronology wholesale. A separately frozen hybrid-source amendment could retain SEC earnings, add explicit SEC share-tag fallbacks for META/SNAP, use documented annual-share staleness rules for foreign filers, and use SimFin only where its price/share history is auditable.

## Frozen hybrid-source result

The user approved the dated SEC + SimFin hybrid amendment before any selector returns were run. SEC remains the earnings source for every ticker. SimFin supplies only EA's unavailable raw close and STZ's historical shares; META uses a labeled SEC diluted weighted-average-share fallback, SNAP uses the SEC `SharesOutstanding` tag, and issuer-specific share-age limits handle documented annual/fiscal filing patterns.

| Rebalance | Eligible | Coverage | 80% gate |
|---|---:|---:|:---:|
| 2020-01-01 | 25 | 83.33% | Pass |
| 2021-01-01 | 25 | 83.33% | Pass |
| 2022-01-01 | 25 | 83.33% | Pass |
| 2023-01-01 | 25 | 83.33% | Pass |
| 2024-01-01 | 25 | 83.33% | Pass |
| 2025-01-01 | 25 | 83.33% | Pass |

The validated candidate contains 150 records, every `available_date` is strictly before its rebalance, and no extraction errors remain among the 25 automated candidates. CCOEY, NTDOY, PRNDY, TCEHY, and UBSFY remain missing without imputation. Methodology: `sec-yahoo-simfin-v2`; source manifest SHA-256: `8205e04d3e5426d16efc05aee4de546ec1f8caedbf6146b76a8acd677a18ece3`.

**Decision: source-feasible with disclosed approximations.** The strict 24-of-30 gate now passes in every year. META's denominator is diluted weighted-average shares rather than period-end shares, and SimFin's STZ share series lacks SEC acceptance timestamps; these limitations must remain visible in the final report and sensitivity interpretation.
