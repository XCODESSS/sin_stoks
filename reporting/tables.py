"""Summary tables and analytical reporting structures."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import RISK_FREE_RATE
from reporting.metrics import (
    calculate_alpha,
    calculate_beta,
    calculate_cagr,
    calculate_calmar_ratio,
    calculate_information_ratio,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_total_return,
    calculate_tracking_error,
    calculate_volatility,
)


def build_summary_table(
    returns: pd.DataFrame,
    portfolio_values: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
    benchmark_column: str = "SPY",
) -> pd.DataFrame:
    """Compute comprehensive risk and return performance metrics for all strategies."""
    rows: list[dict[str, object]] = []
    benchmark_returns = (
        returns[benchmark_column].dropna() if benchmark_column in returns.columns else pd.Series(dtype=float)
    )

    for strategy in returns.columns:
        s_returns = returns[strategy].dropna()
        s_values = (
            portfolio_values[strategy].dropna()
            if strategy in portfolio_values.columns
            else pd.Series(dtype=float)
        )

        if s_returns.empty or s_values.empty:
            continue

        init_val = round(float(s_values.iloc[0]), 2)
        final_val = round(float(s_values.iloc[-1]), 2)
        ret_amt = round(final_val - init_val, 2)

        has_benchmark = not benchmark_returns.empty and len(benchmark_returns) > 1

        rows.append(
            {
                "Strategy": strategy,
                "Initial Value ($)": init_val,
                "Final Value ($)": final_val,
                "Return Amount ($)": ret_amt,
                "Total Return": calculate_total_return(s_values),
                "CAGR": calculate_cagr(s_values),
                "Volatility": calculate_volatility(s_returns),
                "Sharpe Ratio": calculate_sharpe_ratio(s_returns, risk_free_rate),
                "Sortino Ratio": calculate_sortino_ratio(s_returns, risk_free_rate),
                "Maximum Drawdown": calculate_max_drawdown(s_values),
                "Calmar Ratio": calculate_calmar_ratio(s_values),
                "Beta": calculate_beta(s_returns, benchmark_returns) if has_benchmark else np.nan,
                "Alpha": (
                    calculate_alpha(s_returns, benchmark_returns, risk_free_rate) if has_benchmark else np.nan
                ),
                "Tracking Error": (
                    calculate_tracking_error(s_returns, benchmark_returns) if has_benchmark else np.nan
                ),
                "Information Ratio": (
                    calculate_information_ratio(s_returns, benchmark_returns) if has_benchmark else np.nan
                ),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(by="CAGR", ascending=False).reset_index(drop=True)
