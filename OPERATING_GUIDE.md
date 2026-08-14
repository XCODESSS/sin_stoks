# 📖 Repository Operating & Developer Guide

Welcome to the **`sin_stoks`** codebase. This guide explains how the repository is structured, how to run each pipeline step, and exact recipes for modifying the project.

---

## ⚡ Quickstart Commands

```bash
# 1. (Optional) Re-download fresh market data from Yahoo Finance
python main.py

# 2. Run the walk-forward backtest engine across all 6 strategies
python run_backtest.py

# 3. Generate summary metrics table and visualization charts
python report.py

# 4. Run automated test suite
pytest -v

# 5. Check code formatting & linting
ruff check .
```

---

## 🗺️ Repository Map: What Lives Where

```
sin_stoks/
├── config.py                 # ⚙️ Global constants, date ranges, paths, position cap
├── universe.py               # 🏢 30-stock list, 6 behavioral sectors, lookup helpers
├── data_pipeline.py          # 📥 Yahoo Finance downloads, cleaning, outlier filter, atomic CSV saves
├── portfolio_strategies.py   # 🧮 Allocation math (Equal Weight, Max Sharpe, Min Var, etc.) & registry
├── backtest_engine.py        # 🔁 Walk-forward engine, covariance shrinkage, weight drift, BacktestResult
├── main.py                   # 🚀 Thin CLI command to download/refresh market data
├── run_backtest.py           # 🚀 Thin CLI command to run walk-forward backtest
├── report.py                 # 🚀 Thin CLI command to generate summary.csv and PNG charts
├── reporting/                # 📊 Reporting subpackage
│   ├── metrics.py            #    Pure math: CAGR, Volatility, Sharpe, Sortino, Drawdown, Calmar
│   ├── tables.py             #    Table builder for summary.csv
│   └── plots.py              #    Matplotlib charts: equity curves, drawdowns, dividend breakdown
├── tests/                    # 🧪 Offline unit test suite (no network required)
│   ├── test_strategies.py    #    Tests weights sum=1, non-negativity, 25% cap enforcement
│   ├── test_backtest_engine.py #  Tests 2-asset compounding math & window isolation
│   └── test_metrics.py       #    Tests exact mathematical formula outputs
├── data/                     # 💾 Cached market data (weekly/monthly/annual returns)
└── outputs/                  # 📈 Backtest outputs and published reports
    ├── portfolio_backtest/   #    walk_forward_returns.csv, values.csv, weights.csv
    └── report/               #    summary.csv, equity_curves.png, drawdowns.png
```

---

## 🛠️ Common Recipes & How to Modify Things

### 1. How do I change the position cap (e.g. 5% vs 25%)?
- **Option A (Without touching code)**: Pass the CLI argument:
  ```bash
  python run_backtest.py --cap 0.05
  ```
- **Option B (Permanently in config)**: Open [`config.py`](file:///d:/sin_stoks/config.py) and update:
  ```python
  DEFAULT_MAX_WEIGHT = 0.05  # Change from 0.25 to 0.05
  ```

---

### 2. How do I add or modify stocks in the universe?
Open [`universe.py`](file:///d:/sin_stoks/universe.py):
- Update the `COMPANIES` dictionary:
  ```python
  COMPANIES = {
      "DEO": ("Diageo", "Alcohol"),
      # Add or remove tickers here...
  }
  ```
- After changing tickers, run `python main.py` to fetch their market data.

---

### 3. How do I change the backtest years or dates?
Open [`config.py`](file:///d:/sin_stoks/config.py):
- Change `REBALANCE_YEARS` (e.g., `[2021, 2022, 2023, 2024, 2025]`).
- Change `START_DATE`, `END_DATE`, or `COVARIANCE_START`.

---

### 4. How do I add a new portfolio allocation strategy?
1. Open [`portfolio_strategies.py`](file:///d:/sin_stoks/portfolio_strategies.py).
2. Write your strategy function following the standard signature:
   ```python
   def my_new_strategy(
       expected_returns: pd.Series,
       covariance: pd.DataFrame,
       config: StrategyConfig | None = None,
   ) -> pd.Series:
       cfg = config or StrategyConfig()
       tickers = covariance.index
       # Your custom allocation logic here...
       weights = np.full(len(tickers), 1.0 / len(tickers))
       projected = project_to_capped_simplex(weights, max_weight=cfg.max_weight)
       validate_projected_weights(projected, cfg.max_weight, "My New Strategy")
       return pd.Series(projected, index=tickers)
   ```
3. Register it in the `STRATEGIES` dictionary at the bottom of the file:
   ```python
   STRATEGIES = {
       "Equal Weight": equal_weight,
       # ...
       "My New Strategy": my_new_strategy,
   }
   ```
4. Run `python run_backtest.py` and `python report.py`. It will automatically be included in all backtests, outputs, and plots!

---

### 5. How do I add or change a financial metric?
1. Open [`reporting/metrics.py`](file:///d:/sin_stoks/reporting/metrics.py) and add your formula function (e.g., `calculate_omega_ratio(returns)`).
2. Open [`reporting/tables.py`](file:///d:/sin_stoks/reporting/tables.py) and add your metric column to `build_summary_table`.
3. Open [`reporting/__init__.py`](file:///d:/sin_stoks/reporting/__init__.py) to expose the function.

---

### 6. How do I add or modify a chart?
1. Open [`reporting/plots.py`](file:///d:/sin_stoks/reporting/plots.py) and edit or add a plotting function using matplotlib.
2. Call your plotting function inside `main()` in [`report.py`](file:///d:/sin_stoks/report.py).

---

## 📁 Artifacts & Outputs Guide

| Output File | Location | Description |
| :--- | :--- | :--- |
| `walk_forward_returns.csv` | `outputs/portfolio_backtest/` | Weekly net simple returns for all 6 strategies + SPY |
| `walk_forward_values.csv` | `outputs/portfolio_backtest/` | Cumulative portfolio value series starting from $10,000 |
| `walk_forward_weights.csv` | `outputs/portfolio_backtest/` | Target asset weights assigned at each annual rebalance point |
| `summary.csv` | `outputs/report/` | Final performance metrics table (CAGR, Vol, Sharpe, Drawdown, etc.) |
| `equity_curves.png` | `outputs/report/` | High-res comparison plot of portfolio growth over time |
| `drawdowns.png` | `outputs/report/` | Underwater drawdown profile chart |
| `dividend_returns_breakdown.png` | `outputs/report/` | Capital growth vs cash dividends collected breakdown |
