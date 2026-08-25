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

The implementation and synthetic validation may proceed, but the historical experiment is **data-infeasible pending explicit source approval and access**. The preferred next step is either:

- approve a ticker-coverage pilot using licensed Sharadar SF1 + SEP access; or
- provide an approved point-in-time export from FactSet, Compustat/Capital IQ, or another institutional source matching `data/fundamentals_point_in_time.csv`.

Until then, new-strategy historical metrics must remain `N/A`; substituting current ratios or revised statements would invalidate the preregistered experiment.
