"""Reporting and visualization subpackage for sin_stoks."""

from __future__ import annotations

from reporting.interactive import generate_interactive_dashboard
from reporting.metrics import (
    calculate_alpha,
    calculate_beta,
    calculate_cagr,
    calculate_calmar_ratio,
    calculate_drawdown,
    calculate_information_ratio,
    calculate_max_drawdown,
    calculate_net_cagr,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_total_return,
    calculate_tracking_error,
    calculate_turnover,
    calculate_volatility,
)
from reporting.plots import (
    plot_allocation_heatmap,
    plot_celh_concentration_analysis,
    plot_drawdowns,
    plot_equity_curves,
    plot_performance_comparison,
    plot_sector_correlation_heatmap,
    plot_strategy_correlation_heatmap,
)
from reporting.tables import (
    build_summary_table,
)

__all__ = [
    "build_summary_table",
    "calculate_alpha",
    "calculate_beta",
    "calculate_cagr",
    "calculate_calmar_ratio",
    "calculate_drawdown",
    "calculate_information_ratio",
    "calculate_max_drawdown",
    "calculate_net_cagr",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_total_return",
    "calculate_tracking_error",
    "calculate_turnover",
    "calculate_volatility",
    "generate_interactive_dashboard",
    "plot_allocation_heatmap",
    "plot_celh_concentration_analysis",
    "plot_drawdowns",
    "plot_equity_curves",
    "plot_performance_comparison",
    "plot_sector_correlation_heatmap",
    "plot_strategy_correlation_heatmap",
]
