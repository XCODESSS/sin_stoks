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
    calculate_turnover,
    calculate_volatility,
)


def build_summary_table(
    returns: pd.DataFrame,
    portfolio_values: pd.DataFrame,
    weights: pd.DataFrame | None = None,
    asset_log_returns: pd.DataFrame | None = None,
    risk_free_rate: float = RISK_FREE_RATE,
    benchmark_column: str = "SPY",
) -> pd.DataFrame:
    """Compute comprehensive risk, return, and friction metrics for all strategies."""
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
        cagr_val = calculate_cagr(s_values)

        has_benchmark = not benchmark_returns.empty and len(benchmark_returns) > 1

        # Drift-aware recurring turnover. Portfolio values are already net of
        # the transaction costs applied by the backtest engine.
        if (
            weights is not None
            and not weights.empty
            and asset_log_returns is not None
            and strategy != benchmark_column
        ):
            turnover_val = calculate_turnover(weights, strategy, asset_log_returns)
        else:
            turnover_val = 0.0 if strategy == benchmark_column else np.nan

        rows.append(
            {
                "Strategy": strategy,
                "Initial Value ($)": init_val,
                "Final Value ($)": final_val,
                "Return Amount ($)": ret_amt,
                "Total Return": calculate_total_return(s_values),
                "CAGR": cagr_val,
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
                "Turnover": turnover_val,
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(by="CAGR", ascending=False).reset_index(drop=True)
