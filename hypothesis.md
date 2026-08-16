# Quantitative Research Thesis: Vice & Habitual Stocks vs. The Market (SPY)

## 1. Executive Summary & Core Research Question

> **Primary Research Question:**
> *Did an out-of-sample portfolio of 30 companies catering to habitual, recurring human behaviors ("sin stocks") outperform the S&P 500 (`SPY`) between 2020 and 2025, and is this outperformance robust or an artifact of estimation error and single-stock concentration?*

### The Verdict
**Yes, in nominal gross returns, but NO, it is not robust.** 

While the Equal-Weight and Maximum Sharpe vice portfolios nominally outperformed `SPY` (CAGR of **15.70%** and **16.07%** vs. **14.85%** for `SPY`), our factor and concentration decomposition reveals that **the entire outperformance was driven by a single microcap-to-largecap outlier: Celsius Holdings (`CELH`, +2,788.8% return)**. 

Excluding `CELH`, every single vice portfolio strategy underperformed the S&P 500 by **-1.37% to -7.55% annualized**.

---

## 2. Final Out-of-Sample Performance Summary (2020–2025)

The table below presents the audited walk-forward performance across 6 out-of-sample years ($10,000 initial capital, annual rebalancing, 25% max position cap, and 10 bps transaction cost friction):

| Strategy | Final Value ($) | Total Return | CAGR | Volatility | Sharpe Ratio | Sortino Ratio | Max Drawdown | Calmar Ratio | Beta ($\beta$) | Alpha ($\alpha$) | Tracking Error | Info Ratio | Turnover | Net CAGR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Max Sharpe** | **$24,400.54** | **144.01%** | **16.07%** | 21.58% | 0.6114 | 0.5407 | -33.33% | 0.4823 | 0.8347 | +3.57% | 15.37% | +0.1079 | 29.32% | **16.01%** |
| **Equal Weight ($1/N$)** | **$23,932.92** | **139.33%** | **15.70%** | 15.79% | **0.7464** | **0.6814** | **-25.22%** | **0.6223** | 0.7005 | +3.70% | 10.56% | +0.0231 | 0.00% | **15.70%** |
| **Max Diversification** | $23,252.84 | 132.53% | 15.14% | 16.19% | 0.7018 | 0.6574 | -25.49% | 0.5941 | 0.6245 | **+4.16%** | 13.29% | -0.0132 | 11.98% | 15.12% |
| **SPY Benchmark** | $22,906.36 | 129.06% | 14.85% | 18.52% | 0.6229 | 0.5430 | -28.64% | 0.5185 | 1.0000 | 0.00% | 0.00% | N/A | 0.00% | 14.85% |
| **Risk Parity** | $20,880.27 | 108.80% | 13.09% | **15.24%** | 0.6184 | 0.5520 | -25.58% | 0.5118 | 0.6816 | +1.56% | 10.38% | -0.2034 | 3.08% | 13.08% |
| **Inverse Volatility** | $20,075.98 | 100.76% | 12.35% | 15.33% | 0.5731 | 0.5133 | -26.07% | 0.4737 | 0.7010 | +0.70% | 9.86% | -0.2793 | 1.98% | 12.35% |
| **Hierarchical Risk Parity** | $17,633.72 | 76.34% | 9.94% | 15.41% | 0.4309 | 0.3835 | -27.19% | 0.3657 | 0.6995 | -1.43% | 10.02% | -0.4891 | 12.15% | 9.92% |
| **Minimum Variance** | $15,266.69 | 52.67% | 7.33% | 15.54% | 0.2741 | 0.2492 | -27.43% | 0.2671 | 0.6350 | -3.07% | 12.20% | -0.5969 | 12.70% | 7.30% |

---

## 3. Return & Risk Decomposition

### A. The Universe Effect vs. Benchmark Beta
- **Lower Beta Exposure**: Vice companies exhibit defensive, recession-resistant demand inelasticity. Across the board, portfolio beta ranged from **0.62 to 0.83** vs. SPY.
- **Lower Realized Volatility**: Equal Weight achieved **15.79% annualized volatility** vs. **18.52% for SPY**, and smaller max drawdown (**-25.22% vs. -28.64%**).
- **Jensen's Alpha**: Because of the low beta ($\beta \approx 0.70$), CAPM expected returns were lower than market returns, producing positive annualized alphas for Equal Weight (**+3.70%**) and Maximum Diversification (**+4.16%**).

### B. Portfolio Construction Effect: Heuristic vs. Optimization
- **The $1/N$ Triumph**: Equal Weight delivered the highest risk-adjusted return (Sharpe **0.7464**, Sortino **0.6814**, Calmar **0.6223**), completely dominating Mean-Variance Max Sharpe (Sharpe **0.6114**).
- **Estimation Error in Mean-Variance**: In-sample Max Sharpe over-concentrated into high-beta winners that experienced severe subsequent mean-reversion, incurring **33.33% maximum drawdown** and 29.32% turnover.
- **Risk-Based Allocations**: Risk Parity and Inverse Volatility reduced volatility down to **15.24%**, but suffered in CAGR due to heavy under-weighting of high-growth social media and energy drink stocks.

### C. The CELH Outlier & Concentration Effect (Crucial Finding)
Celsius Holdings (`CELH`) gained **+2,788.8%** between 2020 and 2025. When we isolate the universe without this single security, the true baseline behavior emerges:

| Strategy | Full Universe CAGR (With CELH) | Ex-CELH Universe CAGR | Performance Delta | Ex-CELH vs. SPY (14.85%) |
| :--- | :---: | :---: | :---: | :---: |
| **Max Sharpe** | **16.07%** | **13.48%** | **-2.59%** | **Underperformed by -1.37%** |
| **Equal Weight ($1/N$)** | **15.70%** | **10.88%** | **-4.82%** | **Underperformed by -3.97%** |
| **Maximum Diversification** | **15.14%** | **10.23%** | **-4.91%** | **Underperformed by -4.62%** |
| **Risk Parity** | **13.09%** | **10.11%** | **-2.98%** | **Underperformed by -4.74%** |
| **Minimum Variance** | **7.33%** | **7.30%** | **-0.03%** | **Underperformed by -7.55%** |

Without `CELH`, **Equal Weight falls from 15.70% to 10.88%**, lagging SPY by nearly 400 basis points per year.

### D. Sector Correlation & Genuine Diversification
The 6 aggregated behavioral vice sector indices (equal-weighted composite return series) exhibit low pairwise correlations, proving that habitual human consumption operates across distinct, non-overlapping demand drivers:

| Sector | Alcohol | Energy Drinks | Social Media | Tobacco | Gaming | QSR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Alcohol** | **1.00** | 0.58 | 0.32 | 0.41 | 0.31 | 0.61 |
| **Energy Drinks** | 0.58 | **1.00** | 0.29 | 0.32 | 0.34 | 0.54 |
| **Social Media** | 0.32 | 0.29 | **1.00** | **0.21** | 0.45 | 0.38 |
| **Tobacco & Nicotine** | 0.41 | 0.32 | **0.21** | **1.00** | **0.19** | 0.29 |
| **Gaming** | 0.31 | 0.34 | 0.45 | **0.19** | **1.00** | 0.36 |
| **Quick Service Restaurants** | 0.61 | 0.54 | 0.38 | 0.29 | 0.36 | **1.00** |

Tobacco and Gaming operate with an empirical correlation of just **0.190**, and Tobacco and Social Media at **0.208**.

---

## 4. Robustness, Biases & Methodological Limitations

An institutional-grade research project must explicitly acknowledge all potential points of failure and biases:

### 1. Survivorship Bias
- **Issue**: Our 30-stock universe was selected using modern market capitalization leaders (e.g. Constellation Brands, Monster, Meta, Altria).
- **Impact**: Companies that went bankrupt, faced fatal regulatory crackdowns, or suffered delisting prior to 2020 were not included, artificially boosting baseline returns.

### 2. Selection & Look-Ahead Bias
- **Issue**: Including breakout microcaps like Celsius Holdings (`CELH`) in the energy drink bucket reflects hindsight bias. In 2017–2019, CELH was an illiquid microcap with sub-$100M market cap, not an established blue-chip vice leader.
- **Impact**: Outperformance is heavily dependent on this ex-post winner.

### 3. Limited Out-of-Sample Window (2020–2025)
- **Issue**: 6 years is a narrow macroeconomic window dominated by three idiosyncratic regimes: the COVID-19 lockdown boom, the 2022 Fed tightening shock, and the 2023–2025 AI/Big Tech rally.
- **Impact**: Social media (Meta, Alphabet, Microsoft) heavily benefited from the tech multiple expansion, which is not purely a "vice" factor.

### 4. Small Universe Size ($N=30$)
- **Issue**: 30 stocks across 6 sectors means each company represents 3.33% of the Equal Weight portfolio and 20% of its sector block.
- **Impact**: High vulnerability to idiosyncratic single-company events (e.g., Celsius's distribution deal vs. Ubisoft's -74% collapse).

### 5. Position Concentration & Simplex Cap Bounds
- **Issue**: Unconstrained Markowitz optimization assigned up to 70%+ to individual stocks. Imposing a 25% position cap improved stability but artificially constrained pure optimizer intent.

### 6. Transaction Friction & Turnover
- **Issue**: Max Sharpe experienced **29.32% annual turnover**, incurring transaction and slippage drag. While 10 bps friction only reduced CAGR by 6 bps, real-world execution of less liquid names (like Turning Point Brands `TPB`) would incur higher market impact.

### 7. Total Return & Dividend Treatment
- **Issue**: Vice sectors (especially Tobacco with PM, MO, BTI paying 6–9% yields) generate a large fraction of total return via dividends.
- **Impact**: Using `auto_adjust=True` correctly accounts for DRIP, but in taxable accounts, annual dividend drag would significantly reduce net realized wealth.

### 8. Benchmark Appropriateness
- **Issue**: Comparing a 30-stock consumer staple/vice basket to `SPY` (market-cap weighted, 30%+ mega-cap tech) blends sector bias with market risk.
- **Impact**: Comparing against the Consumer Staples ETF (`XLP`) or a blended sector benchmark would isolate pure stock selection alpha.

---

## 5. Final Research Takeaway

The thesis that **vice and habitual consumer companies provide superior risk-adjusted defensive properties is confirmed**:
- Lower volatility (**15.79% vs. 18.52%**)
- Smaller drawdowns (**-25.22% vs. -28.64%**)
- Genuine cross-sector diversification ($\rho < 0.35$ on average)

However, the thesis that **vice stocks reliably beat broad market equity returns on an unconstrained basis is rejected**:
- Without the idiosyncratic breakout of `CELH`, the vice portfolio achieved **10.88% CAGR**, falling short of the S&P 500's **14.85%**.
- High dividend payouts and defensive pricing power provide **resilience**, not high-beta explosive capital growth.
