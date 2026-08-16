"""CLI entry point for performance reports, analytical tables, and visualizations."""

from __future__ import annotations

import pandas as pd

from backtest_engine import BacktestConfig, run_walk_forward_backtest
from config import PORTFOLIO_OUTPUT_DIR, REPORT_DIR
from data_pipeline import write_csv_outputs_atomically
from portfolio_strategies import STRATEGIES
from reporting.interactive import generate_interactive_dashboard
from reporting.plots import (
    plot_allocation_heatmap,
    plot_celh_concentration_analysis,
    plot_dividend_contribution,
    plot_drawdowns,
    plot_equity_curves,
    plot_performance_comparison,
    plot_sector_correlation_heatmap,
    plot_strategy_correlation_heatmap,
)
from reporting.tables import (
    build_summary_table,
)
from run_backtest import load_returns, load_spy_returns
from universe import get_tickers_by_sector


def load_backtest_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load generated walk-forward outputs."""
    returns_path = PORTFOLIO_OUTPUT_DIR / "walk_forward_returns.csv"
    values_path = PORTFOLIO_OUTPUT_DIR / "walk_forward_values.csv"
    weights_path = PORTFOLIO_OUTPUT_DIR / "walk_forward_weights.csv"

    if not returns_path.exists() or not values_path.exists():
        raise FileNotFoundError("Backtest outputs not found. Please run 'python run_backtest.py' first.")

    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    values = pd.read_csv(values_path, index_col=0, parse_dates=True)
    weights = pd.read_csv(weights_path, index_col=[0, 1])

    return returns, values, weights


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    returns, values, weights = load_backtest_data()

    # 1. Build comprehensive 14-column summary table
    summary_table = build_summary_table(returns, values, weights=weights, cost_bps=10.0)
    write_csv_outputs_atomically(
        {
            REPORT_DIR / "summary.csv": (summary_table, {"index": False}),
        }
    )

    # 2. Curated 8-Plot Visual Set
    # Plot 1: Portfolio Growth vs SPY
    plot_equity_curves(values, REPORT_DIR / "equity_curves.png")

    # Plot 2: Drawdowns
    plot_drawdowns(values, REPORT_DIR / "drawdowns.png")

    # Plot 3: Performance Comparison (CAGR vs Sharpe)
    plot_performance_comparison(summary_table, REPORT_DIR / "performance_comparison.png")

    # Plot 4: Representative Allocation Heatmaps
    if not weights.empty:
        plot_allocation_heatmap(weights, "Max Sharpe", REPORT_DIR / "allocation_heatmap_max_sharpe.png")
        plot_allocation_heatmap(weights, "Equal Weight", REPORT_DIR / "allocation_heatmap_equal_weight.png")

    # Plot 5: 6x6 Behavioral Vice Sector Correlation Matrix
    try:
        stock_returns = load_returns()
        sector_mapping = get_tickers_by_sector()
        sector_returns = pd.DataFrame(
            {
                sector: stock_returns[tickers].mean(axis=1)
                for sector, tickers in sector_mapping.items()
                if sector != "Benchmark" and len(tickers) > 0
            }
        )
        if not sector_returns.empty:
            sector_corr = sector_returns.corr()
            plot_sector_correlation_heatmap(sector_corr, REPORT_DIR / "sector_correlation_heatmap.png")
            write_csv_outputs_atomically(
                {
                    REPORT_DIR / "sector_correlation.csv": (sector_corr, {}),
                }
            )
    except Exception as e:
        print(f"Could not generate sector correlation: {e}")

    # Plot 6: 7x7 Strategy Return Correlation Matrix
    strategy_returns = returns.drop(columns=["SPY"], errors="ignore")
    if not strategy_returns.empty:
        plot_strategy_correlation_heatmap(strategy_returns, REPORT_DIR / "strategy_correlation_heatmap.png")

    # Plot 7: Dividend Contribution Breakdown
    plot_dividend_contribution(REPORT_DIR / "dividend_contribution.png")

    # Plot 8: CELH Concentration Analysis (With vs Without CELH)
    try:
        stock_returns = load_returns()
        spy_returns = load_spy_returns()
        res_no_celh = run_walk_forward_backtest(
            stock_returns.drop(columns=["CELH"], errors="ignore"),
            spy_returns,
            STRATEGIES,
            BacktestConfig(),
        )
        plot_celh_concentration_analysis(
            values,
            res_no_celh.portfolio_values,
            REPORT_DIR / "celh_concentration_analysis.png",
        )
    except Exception as e:
        print(f"Could not run CELH concentration analysis: {e}")

    # 3. Interactive dashboard (HTML)
    generate_interactive_dashboard(
        portfolio_values=values,
        summary=summary_table,
        weights=weights,
        save_path=REPORT_DIR / "dashboard.html",
    )

    print("\n" + "=" * 135)
    print("  FINAL PORTFOLIO PERFORMANCE SUMMARY (2020-2025 OUT-OF-SAMPLE)")
    print("=" * 135)
    display_cols = [
        "Strategy",
        "Final Value ($)",
        "Total Return",
        "CAGR",
        "Volatility",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Maximum Drawdown",
        "Calmar Ratio",
        "Beta",
        "Alpha",
        "Tracking Error",
        "Information Ratio",
        "Turnover",
        "Net CAGR",
    ]
    formatted = summary_table[display_cols].copy()
    formatted["Total Return"] = formatted["Total Return"].map(lambda x: f"{x:.2%}")
    formatted["CAGR"] = formatted["CAGR"].map(lambda x: f"{x:.2%}")
    formatted["Volatility"] = formatted["Volatility"].map(lambda x: f"{x:.2%}")
    formatted["Maximum Drawdown"] = formatted["Maximum Drawdown"].map(lambda x: f"{x:.2%}")
    formatted["Sharpe Ratio"] = formatted["Sharpe Ratio"].map(lambda x: f"{x:.4f}")
    formatted["Sortino Ratio"] = formatted["Sortino Ratio"].map(lambda x: f"{x:.4f}")
    formatted["Calmar Ratio"] = formatted["Calmar Ratio"].map(lambda x: f"{x:.4f}")
    formatted["Beta"] = formatted["Beta"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    formatted["Alpha"] = formatted["Alpha"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    formatted["Tracking Error"] = formatted["Tracking Error"].map(
        lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
    )
    formatted["Information Ratio"] = formatted["Information Ratio"].map(
        lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
    )
    formatted["Turnover"] = formatted["Turnover"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    formatted["Net CAGR"] = formatted["Net CAGR"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")

    print(formatted.to_string(index=False))
    print("=" * 135)


if __name__ == "__main__":
    main()
