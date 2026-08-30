# Exploratory Ten-Year Selection-Test Amendment

**Date:** 2026-08-30

This amendment authorizes a separate exploratory 2016–2025 run with market
history beginning in 2011. It does not replace or modify the frozen 2020–2025
experiment or its artifacts.

At each rebalance, a security is eligible only when it has:

- 104 complete weekly returns strictly before the rebalance;
- a point-in-time fundamental record constructed specifically for that
  rebalance date; and
- complete returns during the subsequent holding period if selected.

The original target count, feature definitions, PAM/HDBSCAN parameters,
equal-weight construction, transaction cost, and CELH exclusion are retained.
The price-eligible and fundamental-eligible universes are recorded separately.

The automated fundamental coverage gate fails before 2020: available coverage
is 16/30 in 2016, 17/30 in 2017, 18/30 in 2018, and 21/30 in 2019, versus the
frozen minimum of 24/30. Therefore all ten-year results must be labelled
`exploratory-invalid-under-frozen-coverage-gate`. They cannot confirm, promote,
or supersede the preregistered six-year result.
