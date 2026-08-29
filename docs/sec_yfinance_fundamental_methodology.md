# SEC EDGAR + Yahoo Point-in-Time Fundamental Methodology

## Scope

This free-source pilot attempts to cover 25 of the frozen 30 equities. PRNDY, TCEHY, NTDOY, CCOEY, and UBSFY are manual-only and remain missing; no value is imputed. The pilot must cover at least 24 of 30 names at every 2020–2025 rebalance before a selection backtest can be proposed.

## Frozen Rules

1. A SEC fact is eligible only when its `filed` date is strictly before the rebalance.
2. Price is the final primary-listing unadjusted close strictly before rebalance. Adjusted closes are not multiplied by filed shares.
3. TTM earnings are latest annual earnings plus current interim YTD earnings minus the same filing's prior-year YTD comparison. Annual-only fallback is allowed only when no post-annual interim exists.
4. Market cap is scaled local close × reconciled SEC-filed ordinary shares × historical spot USD FX.
5. P/E is USD market cap / USD TTM earnings. Non-positive earnings receive `earnings_positive=False` and no P/E.
6. Foreign-currency earnings use average daily FX over the earnings observation period; market cap uses spot FX at the price date. This translation is an approximation and is disclosed.
7. `available_date` is the latest of the SEC earnings filing date, SEC shares filing date, and Yahoo price date. Every component must be before rebalance.
8. SEC Company Facts omits XBRL dimensions needed to identify share classes. Duplicate values are removed. A reported total is accepted only when it approximately equals distinct class components; otherwise exactly the configured number of class values is required and summed. Ambiguous groups fail.
9. Share observations more than 200 days old at the price date fail.
10. Structured CIK, accession, fact-period, price, FX, share-reconciliation, and calculation provenance is retained in the generated CSV.
11. SEC requests require an explicitly supplied contact-bearing User-Agent and at least 0.12 seconds between requests. Git identity is never transmitted automatically.
12. SEC and Yahoo caches and generated data remain local and uncommitted. Yahoo-derived observations are used for local research and are not redistributed.

Any source mapping, tag, unit, share-class rule, or formula change requires a new dated plan and makes affected results exploratory.
