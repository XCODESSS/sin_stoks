# Sin Stoks Weekly Project Report

**Reporting date:** 14 August 2026  
**Scope:** Current `D:\sin_stoks` workspace, regenerated local walk-forward outputs, and comparison with the report supplied from last week.

## Executive Summary

- The current pipeline runs successfully through the local walk-forward backtest and reporting steps. Python compilation passed for `main.py`, `backtest_portfolios.py`, `run_backtest.py`, `report.py`, and `download_dividends.py`.
- The active, verified implementation covers **30 portfolio stocks plus SPY**, not the **100-stock / 11-sector** target described in the uncommitted `PROJECT_CONTEXT.md`. Current performance claims must therefore be framed as results for the 30-stock universe.
- Over the 2020-2025 weekly walk-forward period, **Max Sharpe** produces the highest final value ($24,452.77 from $10,000; +144.5%), while **Equal Weight** has the strongest risk-adjusted profile (Sharpe 0.75, volatility 15.8%, maximum drawdown -25.2%). Both exceed SPY's $22,906.36 final value (+129.1%).
- A material methodology change remains uncommitted: missing weekly and monthly observations are filled with 0.0 rather than excluded. This preserves all rows but can create artificial zero returns, so the results should be treated as provisional until missing observations are classified and handled intentionally.

## This Week's Progress

### Implemented and committed work

The repository history since 7 August shows the following material progress:

1. Improved the price-download and return-calculation pipeline, including expected-return and covariance checks.
2. Added Risk Parity and Maximum Diversification allocation strategies alongside Equal Weight, Max Sharpe, Inverse Volatility, and Minimum Variance.
3. Added an expanding-window walk-forward evaluator for annual rebalances from 2020 through 2025.
4. Added portfolio-performance reporting, drawdown/equity-curve graphics, dividend downloads, and dividend-return reporting.
5. Added ignore rules and refreshed project documentation.

### Current uncommitted work

- `backtest_portfolios.py` fills missing monthly and weekly returns with `0.0` and retains all 456 post-2017 weekly observations. This affects covariance inputs and the walk-forward holdings.
- `PROJECT_CONTEXT.md` is untracked. It documents a future 100-stock, 11-sector design but does not match the checked-in 30-stock implementation.

## Current Data and Backtest Status

| Item | Current status |
|---|---|
| Active universe | 30 portfolio equities across six sectors, plus SPY benchmark |
| Annual returns | 31 rows across 2016-2025 |
| Annual coverage | 29 of 30 portfolio stocks in 2016 (SNAP unavailable); full 30-stock coverage from 2017-2025 |
| Monthly returns | 119 months from February 2016 to December 2025 |
| Weekly returns | 460 weeks from 4-10 March 2017 through 20-26 December 2025 |
| Walk-forward period | 313 weekly observations from 3 January 2020 to 26 December 2025 |
| Rebalances | 1 January each year from 2020 through 2025 |
| Position constraints | Long-only; all saved strategy weights sum to 100%; maximum observed weight is 25% |

### Data-quality observations

- `SNAP` has 14 missing monthly observations before its trading history begins, and `TPB` has four missing monthly observations.
- `KDP` has two missing weekly observations within the weekly data range.
- The raw portfolio weekly-return file contains missing values, but the current backtest output has no missing portfolio or SPY returns because `backtest_portfolios.py` zero-fills them.
- SPY and portfolio weekly source files use different raw row counts, but the final 313-row out-of-sample series is complete and aligned.

## Refreshed Walk-Forward Performance

All figures below use the current `outputs/report/summary.csv` and start from a $10,000 notional. Returns are based on the refreshed weekly walk-forward series; they are not an INR portfolio simulation.

| Strategy | Final Value | Total Return | CAGR | Volatility | Sharpe | Sortino | Maximum Drawdown | Calmar |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Max Sharpe | $24,452.77 | 144.5% | 16.1% | 21.6% | 0.61 | 0.54 | -33.3% | 0.48 |
| Equal Weight | $23,972.72 | 139.7% | 15.7% | 15.8% | 0.75 | 0.68 | -25.2% | 0.62 |
| Maximum Diversification | $23,299.15 | 133.0% | 15.2% | 16.2% | 0.70 | 0.66 | -25.5% | 0.60 |
| SPY | $22,906.36 | 129.1% | 14.9% | 18.5% | 0.62 | 0.54 | -28.6% | 0.52 |
| Risk Parity | $20,913.44 | 109.1% | 13.1% | 15.2% | 0.62 | 0.55 | -25.6% | 0.51 |
| Inverse Volatility | $20,106.56 | 101.1% | 12.4% | 15.3% | 0.57 | 0.51 | -26.1% | 0.47 |
| Minimum Variance | $15,293.66 | 52.9% | 7.4% | 15.5% | 0.28 | 0.25 | -27.4% | 0.27 |

### Interpretation

- **Highest absolute return:** Max Sharpe exceeds SPY by 15.4 percentage points of total return, but with the highest volatility and deepest drawdown among the active portfolios.
- **Best return-risk balance:** Equal Weight has the highest Sharpe ratio, lowest maximum drawdown, and highest Calmar ratio. Its result is more balanced than Max Sharpe despite a slightly lower final value.
- **Diversification result:** Maximum Diversification delivers a risk-adjusted profile close to Equal Weight and beats SPY in final value, CAGR, volatility, Sharpe, and drawdown.
- **Defensive strategies:** Risk Parity and Inverse Volatility reduce volatility relative to SPY, but their lower return leaves them behind the leading diversified strategies.
- **Minimum Variance:** The lowest-variance objective does not win on realized volatility and significantly lags the other allocations in return, indicating that the covariance objective was too conservative for this sample.

## Dividend Reporting Reconciliation

Dividend data and charts are present, but the cash-dividend table is not fully reconciled to the refreshed walk-forward summary.

| Strategy | Refreshed Walk-Forward Final Value | Dividend Table DRIP Final Value | Difference |
|---|---:|---:|---:|
| Max Sharpe | $24,452.77 | $24,247.12 | $205.65 |
| Maximum Diversification | $23,299.15 | $23,223.26 | $75.89 |
| Risk Parity | $20,913.44 | $20,882.32 | $31.12 |
| Minimum Variance | $15,293.66 | $15,250.90 | $42.76 |

The dividend table should not be used for final total-return claims until it is regenerated from the same post-change walk-forward values and the reconciliation difference is explained. Small differences may be rounding, but the larger differences above require a deterministic tie-out.

## Validation Performed

- Python compilation passed for all five core pipeline/reporting scripts.
- Walk-forward backtest completed and regenerated portfolio weights, returns, and values.
- Reporting pipeline completed and regenerated the performance summary and chart artifacts.
- All 36 saved annual strategy weight rows sum to 100%.
- No saved weights are negative; no saved weight exceeds the 25% cap.
- The refreshed 313-row out-of-sample return matrix is complete for all six strategies and SPY.

## Risks and Recommended Next Steps

1. **Resolve the universe contract first.** Decide whether the project is formally the checked-in 30-stock / six-sector universe or the proposed 100-stock / 11-sector universe. Update code, datasets, README, and `PROJECT_CONTEXT.md` to one authoritative definition before expanding performance claims.
2. **Replace blanket zero filling with an explicit missing-data policy.** Treat pre-IPO history, delistings, and interior download gaps separately. In particular, investigate the two `KDP` weekly gaps before accepting the current walk-forward statistics.
3. **Reconcile dividend and DRIP outputs.** Regenerate `total_returns_with_dividends.csv` from the same current walk-forward values and add a numerical tie-out check against `summary.csv`.
4. **Add SciPy to `pyproject.toml`.** `backtest_portfolios.py` imports `scipy.optimize.minimize`, but SciPy is not declared as a project dependency.
5. **Commit or discard the current method changes intentionally.** The zero-fill behavior changes outputs and should be reviewed as a methodological decision, not left as an undocumented working-tree modification.

## Overall Assessment

The project has progressed from a static allocator into a usable walk-forward portfolio-analysis pipeline with six active allocation strategies, benchmark comparison, dividend artifacts, and reproducible local reporting. The strongest current conclusion is narrow: in the active 30-stock universe, Equal Weight provides the best realized return-risk balance and Max Sharpe the highest terminal wealth over 2020-2025. The broader behavioral-finance thesis and any 100-stock claim remain unvalidated until the universe definition, missing-data treatment, and dividend tie-out are resolved.
