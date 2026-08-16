"""CLI entry point for performance reports, analytical tables, and visualizations."""

from __future__ import annotations

import pandas as pd

from config import PORTFOLIO_OUTPUT_DIR, REPORT_DIR
from data_pipeline import write_csv_outputs_atomically
from reporting.interactive import generate_interactive_dashboard
from reporting.plots import (
    plot_allocation_heatmap,
    plot_drawdowns,
    plot_equity_curves,
    plot_sector_correlation_heatmap,
    plot_strategy_correlation_heatmap,
)
from reporting.tables import (
    build_summary_table,
)
from run_backtest import load_returns
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

    # 1. Build summary table
    summary_table = build_summary_table(returns, values)
    write_csv_outputs_atomically(
        {
            REPORT_DIR / "summary.csv": (summary_table, {"index": False}),
        }
    )

    # 2. Static visualizations (PNG)
    plot_equity_curves(values, REPORT_DIR / "equity_curves.png")
    plot_drawdowns(values, REPORT_DIR / "drawdowns.png")

    # 3. Strategy Allocation Heatmaps across all 7 strategies
    if not weights.empty:
        all_strategies = [
            "Equal Weight",
            "Max Sharpe",
            "Maximum Diversification",
            "Risk Parity",
            "Inverse Volatility",
            "Hierarchical Risk Parity",
            "Minimum Variance",
        ]
        for strat in all_strategies:
            filename = f"allocation_heatmap_{strat.lower().replace(' ', '_')}.png"
            plot_allocation_heatmap(weights, strat, REPORT_DIR / filename)

    # 4. 6x6 Behavioral Sector Correlation Matrix (from underlying 30-stock returns)
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
    except FileNotFoundError:
        pass

    # 5. 7x7 Strategy Return Correlation Matrix
    strategy_returns = returns.drop(columns=["SPY"], errors="ignore")
    if not strategy_returns.empty:
        plot_strategy_correlation_heatmap(strategy_returns, REPORT_DIR / "strategy_correlation_heatmap.png")

    # 6. Interactive dashboard (HTML)
    generate_interactive_dashboard(
        portfolio_values=values,
        summary=summary_table,
        weights=weights,
        save_path=REPORT_DIR / "dashboard.html",
    )

    print("\n" + "=" * 115)
    print("  PORTFOLIO PERFORMANCE SUMMARY")
    print("=" * 115)
    display_cols = [
        "Strategy",
        "Initial Value ($)",
        "Final Value ($)",
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
    ]
    formatted = summary_table[display_cols].copy()
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
    print(formatted.to_string(index=False))
    print("=" * 115)


if __name__ == "__main__":
    main()
