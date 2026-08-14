"""CLI entry point for performance reports, analytical tables, and visualizations."""

from __future__ import annotations

import pandas as pd

from config import PORTFOLIO_OUTPUT_DIR, REPORT_DIR
from data_pipeline import write_csv_outputs_atomically
from reporting.interactive import generate_interactive_dashboard
from reporting.plots import (
    plot_dividend_returns_breakdown,
    plot_drawdowns,
    plot_equity_curves,
)
from reporting.tables import (
    build_summary_table,
)


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
    plot_dividend_returns_breakdown(
        REPORT_DIR / "total_returns_with_dividends.csv",
        REPORT_DIR / "dividend_returns_breakdown.png",
    )

    # 3. Interactive dashboard (HTML)
    generate_interactive_dashboard(
        portfolio_values=values,
        summary=summary_table,
        weights=weights,
        save_path=REPORT_DIR / "dashboard.html",
    )

    print("\n" + "=" * 80)
    print("  PORTFOLIO PERFORMANCE SUMMARY")
    print("=" * 80)
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
    ]
    formatted = summary_table[display_cols].copy()
    formatted["CAGR"] = formatted["CAGR"].map(lambda x: f"{x:.2%}")
    formatted["Volatility"] = formatted["Volatility"].map(lambda x: f"{x:.2%}")
    formatted["Maximum Drawdown"] = formatted["Maximum Drawdown"].map(lambda x: f"{x:.2%}")
    formatted["Sharpe Ratio"] = formatted["Sharpe Ratio"].map(lambda x: f"{x:.4f}")
    formatted["Sortino Ratio"] = formatted["Sortino Ratio"].map(lambda x: f"{x:.4f}")
    formatted["Calmar Ratio"] = formatted["Calmar Ratio"].map(lambda x: f"{x:.4f}")
    print(formatted.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
