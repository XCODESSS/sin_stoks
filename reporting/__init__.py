"""Reporting and visualization subpackage for sin_stoks."""

from __future__ import annotations

from reporting.interactive import generate_interactive_dashboard
from reporting.metrics import (
    calculate_cagr,
    calculate_calmar_ratio,
    calculate_drawdown,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_total_return,
    calculate_volatility,
)
from reporting.plots import (
    plot_dividend_returns_breakdown,
    plot_drawdowns,
    plot_equity_curves,
)
from reporting.tables import (
    build_summary_table,
)

__all__ = [
    "build_summary_table",
    "calculate_cagr",
    "calculate_calmar_ratio",
    "calculate_drawdown",
    "calculate_max_drawdown",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_total_return",
    "calculate_volatility",
    "generate_interactive_dashboard",
    "plot_dividend_returns_breakdown",
    "plot_drawdowns",
    "plot_equity_curves",
]
