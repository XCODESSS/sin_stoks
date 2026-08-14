# Sin Stocks Portfolio — Project Context

## Purpose

`sin_stoks` is a quantitative portfolio research platform evaluating whether a focused set of behavioral vice equities produces distinct risk and return profiles compared to the S&P 500 (`SPY`). It is an educational, reproducible backtesting project designed with clean architectural separation and quantitative rigor.

## Authoritative Universe (30 Stocks × 6 Sectors + SPY)

| Sector | Tickers |
| --- | --- |
| Alcohol | DEO, BUD, STZ, BF-B, PRNDY |
| Energy Drinks | MNST, CELH, KDP, PEP, KO |
| Social Media | META, GOOGL, SNAP, TCEHY, MSFT |
| Tobacco & Nicotine | PM, BTI, MO, UVV, TPB |
| Gaming | NTDOY, EA, TTWO, CCOEY, UBSFY |
| Quick Service Restaurants | MCD, CMG, YUM, DPZ, QSR |
| Benchmark | SPY |

## Architecture & Module Layout

```text
config.py                 # Centralized paths, rebalance dates, cost policies, and parameters
universe.py               # 30-stock taxonomy, sector mappings, and lookup helpers
data_pipeline.py          # Market data downloader, KDP outlier filter, and atomic persistence
portfolio_strategies.py   # 7 allocation strategies (EW, Max Sharpe, Inv Vol, Min Var, RP, Max Div, HRP) & registry
backtest_engine.py        # Walk-forward engine, weight drift, turnover, costs, and result objects
main.py                   # Thin CLI: market data refresh command
run_backtest.py           # Thin CLI orchestrator: loads data, runs engine, saves artifacts
report.py                 # Command entry point: generates summary tables, metrics, and figures
reporting/
├── metrics.py            # CAGR, Sharpe, Sortino, Beta, Alpha, Tracking Error, IR, Captures, Drawdown
├── tables.py             # Performance summaries, sensitivity tables, and sector correlation metrics
└── plots.py              # Equity curves, drawdowns, allocation heatmaps, dendrograms
tests/
├── test_strategies.py    # Weight sum=1, non-negativity, cap compliance, feasibility
├── test_backtest_engine.py # Window isolation, weight drift, turnover math, 2-asset hand calculation
└── test_metrics.py       # Exact mathematical verification of financial metrics vs standard formulas
```

## Methodology & Settings

- **Data Period**: 2016-01-01 through 2026-01-01.
- **Covariance Estimation Window**: 2017-04-01 through 2025-12-31 (Ledoit-Wolf shrinkage on weekly log returns).
- **Rebalance Schedule**: Annual expanding window ($2020 \rightarrow 2025$).
- **Position Bounds**: 25.0% maximum asset weight cap enforced via binary search simplex projection.
- **Transaction Costs**: 10 bps default on one-way turnover deducted from week 0 out-of-sample holding returns.
- **KDP Outlier Filter**: `KDP_OUTLIER_THRESHOLD = 0.50` applied to filter merger-related data artifacts.

## Verified Results (2020–2025 OOS, Net of 10 bps Costs)

- **Max Sharpe**: $24,400.54 (CAGR: 16.03%, Sharpe: 0.611, Max DD: -33.33%, Beta: 0.83, Alpha: +3.57%)
- **Equal Weight**: $23,932.92 (CAGR: 15.66%, Sharpe: 0.746, Max DD: -25.22%, Beta: 0.70, Alpha: +3.70%)
- **Max Diversification**: $23,252.84 (CAGR: 15.10%, Sharpe: 0.702, Max DD: -25.49%, Beta: 0.62, Alpha: +4.16%)
- **SPY Benchmark**: $22,906.36 (CAGR: 14.81%, Sharpe: 0.623, Max DD: -28.64%, Beta: 1.00, Alpha: 0.00%)
- **Risk Parity**: $20,880.27 (CAGR: 13.05%, Sharpe: 0.618, Max DD: -25.58%, Beta: 0.68, Alpha: +1.56%)
- **Inverse Volatility**: $20,075.98 (CAGR: 12.32%, Sharpe: 0.573, Max DD: -26.07%, Beta: 0.70, Alpha: +0.70%)
- **HRP**: $17,633.72 (CAGR: 9.92%, Sharpe: 0.431, Max DD: -27.19%, Beta: 0.70, Alpha: -1.43%)
- **Minimum Variance**: $15,266.69 (CAGR: 7.31%, Sharpe: 0.274, Max DD: -27.43%, Beta: 0.63, Alpha: -3.07%)
- **Intra/Inter Sector Correlation Ratio**: 1.82x (Within: 0.389 vs Across: 0.213)

## Quality Standards

- **Pytest Suite**: 23/23 unit tests passing.
- **Ruff Linting**: 0 errors, 0 warnings.
