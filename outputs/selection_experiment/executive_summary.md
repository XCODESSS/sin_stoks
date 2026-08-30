# Executive Summary — PAM and HDBSCAN Stock-Selection Experiment

## Headline

The experiment found a strong full-sample result for the HDBSCAN-based Density Selection strategy, but the preregistered robustness tests show that the result is not yet reliable enough for promotion. Density Selection produced a 21.82% net CAGR and 0.949 Sharpe ratio, leading all tested strategies, while PAM-based Partitioning Selection produced a 16.60% CAGR and 0.759 Sharpe ratio. Neither selector passed every frozen promotion gate; both are classified **feasible-but-not-promising**.

## Full-sample results

| Strategy | Final $10,000 | CAGR | Volatility | Sharpe | Maximum drawdown | Recurring turnover |
|---|---:|---:|---:|---:|---:|---:|
| Density Selection | $32,591 | 21.82% | 18.26% | 0.949 | -25.40% | 43.82% |
| Eligible Universe Equal Weight | $27,264 | 18.24% | 16.23% | 0.864 | -27.25% | 13.54% |
| Partitioning Selection | $25,066 | 16.60% | 16.74% | 0.759 | -25.44% | 44.06% |
| Max Sharpe | $24,401 | 16.07% | 21.58% | 0.611 | -33.33% | 22.83% |
| Equal Weight | $23,933 | 15.70% | 15.79% | 0.746 | -25.22% | 13.49% |
| Maximum Diversification | $23,253 | 15.14% | 16.19% | 0.702 | -25.49% | 19.94% |
| SPY | $22,906 | 14.85% | 18.52% | 0.623 | -28.64% | 0.00% |

## Positive findings

1. **Density Selection led every benchmark in the full run.** Its 21.82% CAGR exceeded Eligible Universe Equal Weight by 3.58 percentage points annually and SPY by 6.97 points.
2. **Density improved risk-adjusted performance.** Its 0.949 Sharpe was the highest tested, its volatility was slightly below SPY, and its -25.40% drawdown was better than both Eligible Universe Equal Weight and SPY.
3. **Density had positive active performance.** Its information ratio versus the eligible-universe baseline was approximately 0.47, and its reported CAPM alpha versus SPY was approximately 9.15% annually.
4. **Partitioning showed greater year-to-year consistency.** It beat Eligible Universe Equal Weight in five of six calendar years, despite losing over the full compounded period.
5. **Partitioning retained a small ex-CELH advantage over eligible equal weight.** Its ex-CELH CAGR was 12.60%, versus 12.40% for Eligible Universe Equal Weight and 10.90% for full Equal Weight.
6. **Risk controls worked.** Both selectors held exactly 12 stocks at each rebalance, were equally weighted at approximately 8.33% per selected stock, and never approached the 25% position cap.
7. **CELH was selected dynamically rather than hard-coded.** Partitioning omitted CELH at the January 2020 rebalance, then selected it in 2021 and 2022 after its trailing Sharpe rank strengthened. Density selected it in 2020 and 2021, omitted it in 2022 and 2023, selected it again in 2024, and omitted it in 2025. This demonstrates that neither strategy simply held the eventual winner continuously.
8. **Turnover remained within the frozen limit.** Recurring one-way turnover was about 44% for both selectors, below the 60% gate.
9. **Data coverage passed.** Point-in-time fundamentals were available for 25/30 stocks, or 83.33%, at every rebalance, exceeding the frozen 80% gate.
10. **Chronology and reproducibility checks passed.** Every source availability date preceded its rebalance, 108 tests passed after adding the ex-CELH graph, Ruff and integrity checks passed, and ten output CSV hashes matched on rerun.
11. **The comparison included a fair eligibility baseline.** Eligible Universe Equal Weight separates stock-selection effects from the effect of excluding names without usable point-in-time fundamentals.

## Negative findings and limitations

1. **Neither strategy passed all promotion gates.** The correct formal conclusion is feasible-but-not-promising, not validated or production-ready.
2. **Density was highly sensitive to CELH.** Excluding CELH reduced Density CAGR from 21.82% to 12.38%. That was slightly below Eligible Universe Equal Weight at 12.40% and below SPY at 14.85%.
3. **Partitioning also weakened ex-CELH.** Its CAGR fell from 16.60% to 12.60%, below SPY's 14.85%, although it retained a small advantage over the eligible baseline.
4. **Density lacked annual consistency.** It beat Eligible Universe Equal Weight in only three of six years. Its exceptional 103.07% return in 2020 contributed heavily to the compounded result.
5. **Partitioning's compounded performance lagged the fair baseline.** Its 16.60% CAGR was below Eligible Universe Equal Weight's 18.24%, and its information ratio versus that baseline was negative.
6. **The CELH effect was not caused by an excessive portfolio weight.** CELH was held at only 8.33%, selected in three of six Density portfolios and two of six Partitioning portfolios. The sensitivity came from CELH's extreme return path, demonstrating outlier dependence despite equal weighting.
7. **The evaluation contains only six annual holding periods.** This is too small for reliable statistical significance, p-values, or strong claims about persistence across market regimes.
8. **The conclusion is conditional on the chosen thematic sample.** The universe was selected to represent companies associated with coping and everyday life, without using subsequent returns; CELH's inclusion therefore is not evidence of return-based cherry-picking. The sample is not an exhaustive historical reconstruction of every company that could fit the theme, so the result establishes performance for this basket rather than a universal category premium.
9. **Five stocks remain absent.** CCOEY, NTDOY, PRNDY, TCEHY, and UBSFY lacked acceptable point-in-time fundamentals. Results therefore do not represent all 30 intended names.
10. **Some source observations are approximations.** META uses filed diluted weighted-average shares rather than period-end shares. STZ uses SimFin historical shares without an SEC acceptance timestamp. Issuer-specific share-age extensions were required for several annual or foreign filers.
11. **Trading costs are simplified.** The run applies 10 basis points of transaction costs but does not model taxes, bid-ask variation, market impact, execution delay, or account constraints.
12. **Turnover is materially higher than passive baselines.** Approximately 41–44% recurring turnover ex-CELH/full is acceptable under the frozen gate but remains much higher than Equal Weight and Eligible Universe Equal Weight.
13. **Selection stability is moderate rather than high.** Year-to-year Jaccard similarity generally ranged from roughly 0.33 to 0.60, showing meaningful annual membership changes.
14. **No live or untouched out-of-sample period exists.** The result remains a historical walk-forward simulation and should not be interpreted as live evidence.
15. **The experiment tests selection, not optimized weighting.** Every selected stock received approximately 8.33%; the findings concern which stocks were selected, not whether PAM or HDBSCAN can produce superior continuous portfolio weights.

## Formal conclusion

The experiment provides positive historical evidence that the chosen coping-company basket could generate high returns over this period. CELH was selected thematically rather than because of its later performance, so its success is a legitimate realized outcome—not look-ahead bias. Nevertheless, the density-clustering result is economically large but outlier-sensitive: excluding CELH materially changes the conclusion. PAM appears less spectacular but somewhat more stable across calendar years and marginally more robust ex-CELH. The next defensible step is a newly frozen validation program using a broader point-in-time thematic sample, longer history, alternative rebalance dates, explicit outlier/placebo tests, and a genuinely untouched holdout period.

## Graph

The CELH robustness graph is saved as `outputs/selection_experiment/ex_celh_sensitivity.png`. It shows ex-CELH equity curves and full-versus-ex-CELH CAGR bars.
