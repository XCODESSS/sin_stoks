# Quantitative Research Thesis: Vice & Habitual Stocks vs. the Market (SPY)

## 1. Executive Summary & Core Research Question

> **Primary Research Question:**  
> _Did a historical walk-forward portfolio of 30 companies catering to habitual, recurring human behaviors ("vice and habit stocks") outperform the S&P 500 (`SPY`) between 2020 and 2025, and is any outperformance robust or primarily attributable to estimation error and single-stock concentration?_

### The Verdict

**The portfolio outperformed SPY in historical returns net of the modeled 10 bps transaction cost, but the evidence does not support a robust claim of persistent outperformance.**

Each annual portfolio is formed on January 1 from 2020 through 2025 using only information available before that rebalance date for return and covariance estimation. However, the 30-stock universe itself is a retrospective curated list rather than a point-in-time 2020 constituent set, so this study should be read as a historical backtest with walk-forward estimation rather than a strict out-of-sample constituent test.

The Equal-Weight and Maximum Sharpe portfolios achieved CAGRs of **15.70%** and **16.07%**, respectively, compared with **14.85% for SPY**. However, this advantage is highly sensitive to the inclusion of **Celsius Holdings (`CELH`)**, which returned approximately **+2,788.8%** over the sample.

When `CELH` is excluded, every strategy listed in the ex-CELH table underperformed SPY. Ex-CELH CAGRs ranged from **7.30% to 13.48%**, compared with SPY's **14.85%**.

The evidence therefore supports a more nuanced conclusion:

> **Vice and habitual-consumption stocks exhibited attractive defensive and diversification characteristics during 2020–2025, but the sample does not demonstrate that the thematic universe reliably generated superior market returns. The observed outperformance was highly dependent on a single exceptional stock.**

---

# 2. Historical Walk-Forward Performance Summary (2020–2025)

The table presents the walk-forward performance across six annual rebalances in the historical backtest using a **$10,000 initial investment**, annual rebalancing, a **25% maximum position constraint**, and **10 bps transaction-cost friction**.

| Strategy                     |    Final Value | Total Return |   Net CAGR | Volatility |     Sharpe |    Sortino | Max Drawdown |     Calmar |   Beta |    Alpha\* | Tracking Error | Info Ratio\* | Drifted Turnover |
| ---------------------------- | -------------: | -----------: | ---------: | ---------: | ---------: | ---------: | -----------: | ---------: | -----: | ---------: | -------------: | -----------: | ---------------: |
| **Max Sharpe**               | **$24,400.54** |  **144.01%** | **16.07%** |     21.58% |     0.6114 |     0.5407 |      -33.33% |     0.4823 | 0.8347 |     +3.57% |         15.37% |      +0.1079 |           22.83% |
| **Equal Weight**             | **$23,932.92** |  **139.33%** | **15.70%** |     15.79% | **0.7464** | **0.6814** |  **-25.22%** | **0.6223** | 0.7005 |     +3.70% |         10.56% |      +0.0231 |           13.49% |
| **Maximum Diversification**  |     $23,252.84 |      132.53% |     15.14% |     16.19% |     0.7018 |     0.6574 |      -25.49% |     0.5941 | 0.6245 | **+4.16%** |         13.29% |      -0.0132 |           19.94% |
| **SPY Benchmark**            |     $22,906.36 |      129.06% |     14.85% |     18.52% |     0.6229 |     0.5430 |      -28.64% |     0.5185 | 1.0000 |      0.00% |          0.00% |          N/A |            0.00% |
| **Risk Parity**              |     $20,880.27 |      108.80% |     13.09% | **15.24%** |     0.6184 |     0.5520 |      -25.58% |     0.5118 | 0.6816 |     +1.56% |         10.38% |      -0.2034 |           11.95% |
| **Inverse Volatility**       |     $20,075.98 |      100.76% |     12.35% |     15.33% |     0.5731 |     0.5133 |      -26.07% |     0.4737 | 0.7010 |     +0.70% |          9.86% |      -0.2793 |           10.66% |
| **Hierarchical Risk Parity** |     $17,633.72 |       76.34% |      9.94% |     15.41% |     0.4309 |     0.3835 |      -27.19% |     0.3657 | 0.6995 |     -1.43% |         10.02% |      -0.4891 |           15.09% |
| **Minimum Variance**         |     $15,266.69 |       52.67% |      7.33% |     15.54% |     0.2741 |     0.2492 |      -27.43% |     0.2671 | 0.6350 |     -3.07% |         12.20% |      -0.5969 |           15.34% |

\* **Important:** Alpha and Information Ratio should only be interpreted according to the exact regression/active-return methodology used in the backtest. They should not automatically be described as Jensen's alpha or as `active return / tracking error` unless those are the formulas actually implemented.

---

# 3. Return and Risk Decomposition

## A. Lower Market Exposure and Defensive Characteristics

The vice portfolios generally exhibited lower market sensitivity than SPY.

Portfolio betas ranged from approximately **0.62 to 0.83**, compared with SPY's beta of 1.00. This indicates that the universe had substantially less systematic equity-market exposure during the sample.

The Equal-Weight portfolio also experienced lower realized volatility:

- **Equal Weight:** 15.79%
- **SPY:** 18.52%

Its maximum drawdown was also smaller:

- **Equal Weight:** -25.22%
- **SPY:** -28.64%

These results provide evidence that the universe had **defensive characteristics during this particular sample period**.

However, six years is not sufficient to establish that these properties are structural. They should therefore be described as **sample evidence**, rather than a confirmed permanent characteristic of vice stocks.

---

## B. Equal Weight vs. Optimization

One of the strongest findings is the performance of the simple **1/N Equal-Weight portfolio**.

Equal Weight achieved:

- **15.70% CAGR**
- **0.7464 Sharpe**
- **0.6814 Sortino**
- **-25.22% maximum drawdown**
- **15.79% volatility**

It therefore produced the **highest Sharpe and Sortino ratios among the tested strategies**, despite using no optimization whatsoever.

The intuition behind mean-variance optimization is that estimated expected returns and covariances can be used to find an efficient portfolio. In practice, however, expected-return estimates are noisy.

The Max Sharpe strategy achieved a higher CAGR (**16.07%**) than Equal Weight, but its:

- Sharpe ratio was lower (**0.6114 vs. 0.7464**),
- volatility was higher (**21.58% vs. 15.79%**),
- maximum drawdown was substantially larger (**-33.33% vs. -25.22%**),
- drift-aware turnover was higher (**22.83% vs. 13.49%**).

This is consistent with the **estimation-error problem in mean-variance optimization**: small errors in expected returns and covariance estimates can produce materially different portfolio weights.

The result should not be interpreted as proof that optimization is inherently inferior. Rather:

> **Within this universe and sample period, Equal Weight was more robust to estimation error than the Max Sharpe allocation.**

---

## C. Risk-Based Strategies

Risk Parity and Inverse Volatility successfully reduced portfolio volatility:

- **Risk Parity:** 15.24%
- **Inverse Volatility:** 15.33%
- **Equal Weight:** 15.79%

However, lower volatility did not translate into superior returns.

Their CAGRs were:

- Risk Parity: **13.09%**
- Inverse Volatility: **12.35%**
- Equal Weight: **15.70%**

This demonstrates an important distinction:

> **Risk reduction and return maximization are different objectives.**

The available results support the claim that these strategies reduced volatility, but the stronger claim that they specifically underweighted high-growth sectors such as Social Media and Energy Drinks should only be made after presenting their actual portfolio weights and attribution analysis.

---

# 4. The CELH Concentration Effect

This is the most important robustness test in the entire study.

Celsius Holdings (`CELH`) generated approximately **+2,788.8%** over the period.

The impact of removing CELH is dramatic:

| Strategy                    | Full-Universe CAGR | Ex-CELH CAGR |   Change | Ex-CELH vs. SPY |
| --------------------------- | -----------------: | -----------: | -------: | --------------: |
| **Max Sharpe**              |         **16.07%** |   **13.48%** | -2.59 pp |    **-1.37 pp** |
| **Equal Weight**            |         **15.70%** |   **10.88%** | -4.82 pp |    **-3.97 pp** |
| **Maximum Diversification** |         **15.14%** |   **10.23%** | -4.91 pp |    **-4.62 pp** |
| **Risk Parity**             |         **13.09%** |   **10.11%** | -2.98 pp |    **-4.74 pp** |
| **Minimum Variance**        |          **7.33%** |    **7.30%** | -0.03 pp |    **-7.55 pp** |

The result is unambiguous:

> **Every tested strategy loses its advantage over SPY when CELH is removed.**

Equal Weight provides the clearest example. Its CAGR falls from **15.70% to 10.88%**, converting a +0.85 percentage-point advantage over SPY into a **-3.97 percentage-point annual shortfall**.

Therefore, the original hypothesis of broad-based superior performance is not supported.

The full-universe result is better interpreted as:

> **A diversified thematic portfolio happened to contain an extreme winner whose performance materially altered the portfolio's aggregate return.**

That is an important result rather than a failure of the research. It demonstrates why robustness testing is necessary.

---

# 5. Sector Correlation and Diversification

The six behavioral sectors exhibit meaningful differences in their return behavior.

| Sector                 |  Alcohol | Energy Drinks | Social Media |  Tobacco |   Gaming |      QSR |
| ---------------------- | -------: | ------------: | -----------: | -------: | -------: | -------: |
| **Alcohol**            | **1.00** |          0.58 |         0.32 |     0.41 |     0.31 |     0.61 |
| **Energy Drinks**      |     0.58 |      **1.00** |         0.29 |     0.33 |     0.34 |     0.54 |
| **Social Media**       |     0.32 |          0.29 |     **1.00** | **0.21** |     0.45 |     0.38 |
| **Tobacco & Nicotine** |     0.41 |          0.33 |     **0.21** | **1.00** | **0.19** |     0.29 |
| **Gaming**             |     0.31 |          0.34 |         0.45 | **0.19** | **1.00** |     0.36 |
| **QSR**                |     0.61 |          0.54 |         0.38 |     0.29 |     0.36 | **1.00** |

There are several genuinely low-correlation relationships:

- Tobacco × Gaming: **0.19**
- Tobacco × Social Media: **0.21**
- Energy Drinks × Social Media: **0.29**
- Tobacco × QSR: **0.29**

However, the sectors are **not uniformly low-correlated**.

The strongest relationships are:

- Alcohol × QSR: **0.61**
- Alcohol × Energy Drinks: **0.58**
- Energy Drinks × QSR: **0.54**

The average off-diagonal correlation is approximately **0.37**, so describing the sectors as having "low pairwise correlations" or claiming that the average correlation is below 0.35 would be inaccurate.

A more defensible conclusion is:

> **The six behavioral sectors exhibit moderate overall correlation with several low-correlation pairings, providing meaningful but incomplete diversification benefits.**

---

# 6. Strategy Correlation

The strategy-return correlation matrix tells a different story.

Most strategies have extremely high correlations with one another:

- Equal Weight × Risk Parity: **0.990**
- Inverse Volatility × Risk Parity: **0.992**
- Equal Weight × Inverse Volatility: **0.983**
- Risk Parity × HRP: **0.979**
- Minimum Variance × HRP: **0.959**

Max Sharpe is somewhat more distinct, with correlations generally between **0.645 and 0.805**.

This is important because it means:

> **The different portfolio optimizers do not create fundamentally different return streams.**

They are alternative implementations applied to essentially the same underlying universe.

Consequently, comparing their performance is useful for evaluating **portfolio-construction methodology**, and the correlation structure suggests that combining all seven strategies into an equal-weighted meta-blend would likely not provide substantial diversification because their returns are highly correlated.

---

# 7. Total Return & Dividend Reinvestment (DRIP)

The backtest uses Yahoo Finance's `auto_adjust=True` market data stream as the single source of truth for total returns, which directly compounds all corporate cash dividends and stock splits back into adjusted closing prices.

- **High-Yield Sectors**: Habitual consumer sectors (particularly Tobacco with PM, MO, and BTI historically paying 6–9% yields, and Alcohol) generate a substantial component of long-term total return from cash dividend compounding.
- **Tax Friction Considerations**: While `auto_adjust=True` correctly tracks gross pre-tax total return under automatic DRIP, in a taxable account, annual dividend distributions would face ordinary or qualified income taxation drag, moderating net realized wealth accumulation relative to non-dividend-paying growth equities. The backtest is appropriately interpreted as a **pre-tax total-return analysis**.

---

# 8. Robustness, Biases and Methodological Limitations

An institutional-quality interpretation requires substantial caution.

## 1. Survivorship Bias

The universe was constructed using companies that survived and remained identifiable leaders within their respective categories.

This creates survivorship bias.

Companies that failed, were acquired, delisted, or experienced severe permanent impairment may be absent.

Therefore, the results potentially overstate the historical attractiveness of the thematic universe.

---

## 2. Universe Selection and Look-Ahead Bias

The inclusion of CELH is particularly important.

CELH was a relatively small and speculative company earlier in the sample, yet became one of the strongest performers in the entire universe.

If the universe was constructed using information that was only obvious **after** the investment period began, this introduces look-ahead bias.

Therefore, the CELH result should not be interpreted as evidence that an investor in 2020 could have confidently identified CELH as the eventual winner.

A stronger future test would define the universe using only information available at the **initial portfolio formation date**, with subsequent additions and deletions handled systematically.

---

## 3. Limited Six-Year Sample

The 2020–2025 period is relatively short and contains unusual market regimes:

1. COVID-19 shock and recovery
2. 2022 inflation and aggressive monetary tightening
3. 2023–2025 technology and AI-driven market leadership

A six-year period cannot establish that the observed relationships are persistent across economic cycles.

The findings should therefore be treated as **historical evidence**, not proof of a permanent "vice factor."

---

## 4. Small Universe

The portfolio contains only **30 companies across six behavioral categories**.

With equal weighting, each company represents approximately **3.33%** of the portfolio, while each sector contains approximately one-sixth of the universe.

This creates meaningful idiosyncratic exposure.

CELH demonstrates exactly why this matters: one extreme winner can materially change aggregate portfolio performance.

---

## 5. Optimization and Concentration

Mean-variance optimization is highly sensitive to estimated expected returns and covariance matrices.

The 25% position cap reduces extreme concentration but also means the Max Sharpe portfolio is no longer the unconstrained theoretical maximum-Sharpe portfolio.

Therefore, the correct interpretation is:

> **The 25%-capped Max Sharpe strategy represents a constrained implementation of mean-variance optimization, not the unconstrained theoretical optimum.**

This distinction should be stated explicitly.

---

## 6. Transaction Costs and Turnover

Max Sharpe produced approximately **22.83% recurring annual turnover**, compared with **13.49%** for Equal Weight. Equal Weight still trades at annual rebalances because its holdings drift away from equal weights during each holding period.

Turnover is reported as **one-way turnover**, defined as $\tau_t = \frac{1}{2}\sum_i |w_{i,t}^{target} - w_{i,t}^{drifted}|$. The recurring-turnover averages exclude the initial investment from cash. In accordance with the project plan, the modeled per-rebalance cost is $c\tau_t$, with $c = 10\,\text{bps} = 0.001$.

If the gross first-period return after a rebalance is $g_{t,1}$, the engine records $r^{net}_{t,1} = g_{t,1} - c\tau_t$. The portfolio values and CAGRs in this report are computed from those net returns, so no second transaction-cost adjustment is applied during reporting.

However, this may underestimate real-world implementation costs for smaller or less liquid companies.

Market impact, bid-ask spreads, taxes and liquidity constraints could therefore make the optimized strategies less attractive in practice.

---

## 7. Total Return and Dividend Treatment

The backtest uses adjusted prices and therefore incorporates dividends into total returns.

Tobacco companies in particular derive a substantial portion of their historical investor return from dividends.

This means comparing only price appreciation would materially understate their economic return.

The results should therefore be understood as **total-return results**, not price-return results.

For taxable investors, dividend taxation could reduce realized after-tax wealth relative to the backtested results.

---

## 8. Benchmark Selection

SPY is a useful broad-market benchmark, but it is not a perfectly matched control.

SPY is market-cap weighted and heavily influenced by mega-cap technology, whereas the vice universe contains consumer, entertainment, gaming and tobacco businesses with different factor exposures.

Therefore, underperformance versus SPY does **not** necessarily mean that the behavioral-stock selection process was poor.

A stronger study should additionally compare against:

- **SPY** — broad U.S. equity market
- **XLP** — Consumer Staples
- A custom sector-weighted benchmark
- Potentially factor benchmarks such as value, momentum and quality

This would help separate:

**behavioral-theme exposure → sector exposure → factor exposure → genuine stock-selection effect.**

---

# 9. What the Results Actually Establish

The results support three conclusions with different levels of confidence.

### Finding 1 — Strongest Evidence

**The full universe outperformed SPY over 2020–2025.**

Equal Weight:

> **15.70% CAGR vs. 14.85% SPY**

Max Sharpe:

> **16.07% CAGR vs. 14.85% SPY**

This is an observed historical result.

---

### Finding 2 — Strong Evidence Within the Sample

**The vice universe displayed lower volatility and, in the case of Equal Weight, smaller drawdowns than SPY.**

Equal Weight:

> 15.79% volatility vs. 18.52% for SPY

> -25.22% maximum drawdown vs. -28.64% for SPY

This supports the hypothesis that the basket behaved relatively defensively during the period examined.

---

### Finding 3 — Critical Robustness Result

**The return outperformance is not robust to the removal of CELH.**

Every strategy listed in the ex-CELH table underperformed SPY after removing CELH.

Equal Weight:

> **10.88% ex-CELH CAGR vs. 14.85% SPY**

Max Sharpe:

> **13.48% ex-CELH CAGR vs. 14.85% SPY**

This substantially weakens the argument that the behavioral-stock universe possesses a persistent return premium.

---

# 10. Final Research Conclusion

The original hypothesis should therefore be **partially rejected and partially supported**.

### Hypothesis A

> **"Companies built around recurring human behaviors produce superior market returns."**

**Not supported.**

Although the full universe outperformed SPY, that result disappears in the ex-CELH table once the extreme CELH outlier is removed.

The ex-CELH Equal-Weight portfolio produced only **10.88% CAGR**, compared with **14.85% for SPY**.

---

### Hypothesis B

> **"Companies built around recurring human behaviors exhibit defensive and diversification characteristics."**

**Supported, but with important qualifications.**

The Equal-Weight portfolio exhibited:

- lower volatility than SPY,
- lower maximum drawdown,
- lower market beta,
- competitive Sharpe and Sortino ratios,
- and meaningful diversification across behavioral sectors.

However, the sector correlations are **moderate rather than uniformly low**, with an average off-diagonal correlation of approximately **0.37**.

---

### Hypothesis C

> **"Simple portfolio construction may be more robust than return-optimized allocation."**

**Supported within this experiment.**

Equal Weight produced the highest Sharpe ratio (**0.7464**) and Sortino ratio (**0.6814**) despite requiring no parameter estimation or optimization.

The Max Sharpe portfolio achieved the highest CAGR but at substantially higher volatility, drawdown and turnover.

This provides evidence that **optimization did not improve risk-adjusted performance in this particular sample**.

---

# Final Verdict

> **The research does not demonstrate a persistent "vice stock premium." Instead, it demonstrates that a portfolio of companies serving recurring consumer behaviors can provide defensive characteristics and meaningful diversification, but its apparent market outperformance during 2020–2025 was heavily dependent on a single exceptional winner, Celsius Holdings.**

The most defensible interpretation is therefore:

**Vice stocks provided resilience, diversification and competitive risk-adjusted performance in this sample — but not robust evidence of superior long-term returns.**

The strongest result of the study is arguably **not that vice stocks beat SPY**, but that **the apparent outperformance disappears under a simple and economically meaningful robustness test**.

That makes the project substantially more credible as quantitative research: the analysis does not merely report the attractive headline CAGR; it investigates **why the result occurred and whether the conclusion survives reasonable perturbations of the data.**
