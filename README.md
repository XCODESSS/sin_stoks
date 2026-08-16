# Behavioral "Sin Stocks" Portfolio Research

A modular Python backtest comparing seven portfolio-allocation methods across
30 behavioral-consumption equities with the S&P 500 (`SPY`). The project uses
annual walk-forward estimation, Ledoit-Wolf covariance shrinkage, drift-aware
turnover, and explicit transaction costs.

This is historical quantitative research, not investment advice or evidence of
a persistent trading edge. The stock universe was selected retrospectively, so
the results should not be described as a strict point-in-time out-of-sample
constituent test.

## What the project found

Across the 2020-2025 historical walk-forward backtest, net of the modeled 10 bps
transaction cost:

| Strategy | Final value | Net CAGR | Sharpe | Max drawdown | Drifted turnover |
| --- | ---: | ---: | ---: | ---: | ---: |
| Max Sharpe | $24,400.54 | 16.07% | 0.6114 | -33.33% | 22.83% |
| Equal Weight | $23,932.92 | 15.70% | **0.7464** | **-25.22%** | 13.49% |
| Maximum Diversification | $23,252.84 | 15.14% | 0.7018 | -25.49% | 19.94% |
| SPY | $22,906.36 | 14.85% | 0.6229 | -28.64% | 0.00% |
| Risk Parity | $20,880.27 | 13.09% | 0.6184 | -25.58% | 11.95% |
| Inverse Volatility | $20,075.98 | 12.35% | 0.5731 | -26.07% | 10.66% |
| Hierarchical Risk Parity | $17,633.72 | 9.94% | 0.4309 | -27.19% | 15.09% |
| Minimum Variance | $15,266.69 | 7.33% | 0.2741 | -27.43% | 15.34% |

The headline outperformance is not broad-based. Celsius Holdings (`CELH`) was
an exceptional winner, and removing it pushes the tested portfolios below SPY.
That concentration result is treated as a limitation, not hidden as an
inconvenient detail.

The other important finding came from debugging. A merger-related artifact in
Keurig Dr Pepper (`KDP`) produced an implausible weekly return that distorted
the covariance estimate and flattened several risk-based allocations. The data
pipeline now applies a documented outlier rule before backtesting.

The detailed interpretation, formulas, and limitations are in
[hypothesis.md](hypothesis.md).

## Methodology

- 30 equities across Alcohol, Energy Drinks, Social Media, Tobacco & Nicotine,
  Gaming, and Quick Service Restaurants
- `SPY` benchmark
- Weekly adjusted-price returns from 2016 through 2025
- Expanding estimation windows and annual rebalancing from 2020 through 2025
- Ledoit-Wolf covariance shrinkage
- Long-only portfolios with a 25% position cap
- Equal Weight, Max Sharpe, Inverse Volatility, Minimum Variance, Risk Parity,
  Maximum Diversification, and Hierarchical Risk Parity
- Intra-year buy-and-hold weight drift
- One-way turnover calculated from drifted pre-trade weights
- Cost model: `turnover × 10 bps`, deducted once at each rebalance

The initial investment from cash receives turnover of 1.0 for cost accounting.
Reported recurring turnover averages exclude that initial investment.

## Architecture

```text
config.py                 Shared paths and model parameters
universe.py               Universe and sector definitions
data_pipeline.py          Downloading, cleaning, validation, and atomic writes
portfolio_strategies.py   Allocation strategies and strategy registry
backtest_engine.py        Walk-forward windows, covariance, drift, costs, results
run_backtest.py           Thin backtest command
reporting/                Metrics, tables, static plots, interactive dashboard
report.py                 Thin reporting command
tests/                    Deterministic strategy, engine, and metric tests
```

`run_backtest.py` and `report.py` use persisted data. Network access is required
only when refreshing the market data with `main.py`.

## Reproduce locally

Python 3.10 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python main.py
python run_backtest.py --cap 0.25
python report.py
```

If the tracked data artifacts are already present, you can skip `python main.py`
and reproduce the backtest and report without downloading new data.

The report command creates static figures in `outputs/report/` and a local
interactive dashboard at `outputs/report/dashboard.html`.

## Quality checks

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check .
```

Current verified baseline: 22 tests passing and zero Ruff violations.

## Limitations

- Retrospective universe selection introduces look-ahead and survivorship bias.
- CELH materially drives the strongest headline result.
- Annual rebalancing is the only implemented schedule.
- The return and covariance estimates remain sensitive to finite-sample error.
- The 10 bps model excludes market impact, taxes, and security-specific
  liquidity constraints.
- Long-only, unlevered portfolios with a 25% cap are evaluated; conclusions do
  not generalize automatically to other constraints.

See [hypothesis.md](hypothesis.md) for the full research discussion and
[outputs/report/summary.csv](outputs/report/summary.csv) for machine-readable
results.
