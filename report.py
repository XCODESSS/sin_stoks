"""Performance report for walk-forward portfolio backtests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path("outputs/portfolio_backtest")
REPORT_DIR = Path("outputs/report")

STARTING_VALUE = 10_000
RISK_FREE_RATE = 0.04
PERIODS_PER_YEAR = 52


# ============================================================================
# LOAD DATA
# ============================================================================

def load_returns() -> pd.DataFrame:
    """Load portfolio period returns."""
    return pd.read_csv(
        OUTPUT_DIR / "walk_forward_returns.csv",
        index_col=0,
        parse_dates=True,
    )


def load_portfolio_values() -> pd.DataFrame:
    """Load cumulative portfolio values."""
    return pd.read_csv(
        OUTPUT_DIR / "walk_forward_values.csv",
        index_col=0,
        parse_dates=True,
    )


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

def calculate_total_return(values: pd.Series) -> float:
    """Total portfolio return."""
    return values.iloc[-1] / values.iloc[0] - 1


def calculate_cagr(values: pd.Series) -> float:
    """Compound Annual Growth Rate."""
    years = (values.index[-1] - values.index[0]).days / 365.25
    return (values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1


def calculate_volatility(returns: pd.Series) -> float:
    """Annualized volatility."""
    return returns.std() * np.sqrt(PERIODS_PER_YEAR)


def _per_period_risk_free_target(
    risk_free_rate: float,
    returns_are_log: bool = False,
) -> float:
    """Convert an annual risk-free rate into the same per-period return domain."""
    if returns_are_log:
        return np.log1p(risk_free_rate) / PERIODS_PER_YEAR
    return (1 + risk_free_rate) ** (1 / PERIODS_PER_YEAR) - 1


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
) -> float:
    """Annualized Sharpe ratio."""
    annual_return = returns.mean() * PERIODS_PER_YEAR
    annual_volatility = calculate_volatility(returns)

    if annual_volatility == 0:
        return np.nan

    return (annual_return - risk_free_rate) / annual_volatility


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
    returns_are_log: bool = False,
) -> float:
    """Annualized Sortino ratio."""
    target_return = _per_period_risk_free_target(risk_free_rate, returns_are_log)
    downside = target_return - returns[returns < target_return]
    downside_deviation = np.sqrt(np.mean(np.square(downside))) * np.sqrt(PERIODS_PER_YEAR)

    if downside.empty or downside_deviation == 0:
        return np.nan

    annual_return = returns.mean() * PERIODS_PER_YEAR

    return (annual_return - risk_free_rate) / downside_deviation


def calculate_drawdown(values: pd.Series) -> pd.Series:
    """Drawdown series."""
    running_max = values.cummax()
    return values / running_max - 1


def calculate_max_drawdown(values: pd.Series) -> float:
    """Maximum drawdown."""
    return calculate_drawdown(values).min()


def calculate_calmar_ratio(values: pd.Series) -> float:
    """Calmar ratio."""
    max_drawdown = abs(calculate_max_drawdown(values))

    if max_drawdown == 0:
        return np.nan

    return calculate_cagr(values) / max_drawdown
# ============================================================================
# SUMMARY TABLE
# ============================================================================

def build_summary_table(
    returns: pd.DataFrame,
    portfolio_values: pd.DataFrame,
) -> pd.DataFrame:
    """Compute performance metrics for every strategy."""

    rows = []

    for strategy in returns.columns:

        strategy_returns = returns[strategy].dropna()
        strategy_values = portfolio_values[strategy].dropna()

        if strategy_returns.empty:
            rows.append(
                {
                    "Strategy": strategy,
                    "Status": "skipped: no return observations",
                    "Initial Value ($)": np.nan,
                    "Final Value ($)": np.nan,
                    "Return Amount ($)": np.nan,
                    "Total Return": np.nan,
                    "CAGR": np.nan,
                    "Volatility": np.nan,
                    "Sharpe Ratio": np.nan,
                    "Sortino Ratio": np.nan,
                    "Maximum Drawdown": np.nan,
                    "Calmar Ratio": np.nan,
                }
            )
            continue

        if strategy_values.index.nunique() < 2:
            rows.append(
                {
                    "Strategy": strategy,
                    "Status": "skipped: fewer than two value timestamps",
                    "Initial Value ($)": np.nan,
                    "Final Value ($)": np.nan,
                    "Return Amount ($)": np.nan,
                    "Total Return": np.nan,
                    "CAGR": np.nan,
                    "Volatility": np.nan,
                    "Sharpe Ratio": np.nan,
                    "Sortino Ratio": np.nan,
                    "Maximum Drawdown": np.nan,
                    "Calmar Ratio": np.nan,
                }
            )
            continue

        init_val = round(float(strategy_values.iloc[0]), 2)
        final_val = round(float(strategy_values.iloc[-1]), 2)
        return_amt = round(final_val - init_val, 2)

        rows.append(
            {
                "Strategy": strategy,
                "Initial Value ($)": init_val,
                "Final Value ($)": final_val,
                "Return Amount ($)": return_amt,
                "Total Return": calculate_total_return(strategy_values),
                "CAGR": calculate_cagr(strategy_values),
                "Volatility": calculate_volatility(strategy_returns),
                "Sharpe Ratio": calculate_sharpe_ratio(strategy_returns),
                "Sortino Ratio": calculate_sortino_ratio(strategy_returns),
                "Maximum Drawdown": calculate_max_drawdown(strategy_values),
                "Calmar Ratio": calculate_calmar_ratio(strategy_values),
            }
        )

    summary = pd.DataFrame(rows)

    return summary.sort_values(
        by="CAGR",
        ascending=False,
    ).reset_index(drop=True)


# ============================================================================
# PLOTS
# ============================================================================

def plot_equity_curves(
    portfolio_values: pd.DataFrame,
) -> None:
    """Plot cumulative portfolio values."""

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))

    for strategy in portfolio_values.columns:
        plt.plot(
            portfolio_values.index,
            portfolio_values[strategy],
            label=strategy,
        )

    plt.title("Walk-Forward Portfolio Value")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.tight_layout()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    plt.savefig(REPORT_DIR / "equity_curves.png")
    plt.close()


def plot_drawdowns(
    portfolio_values: pd.DataFrame,
) -> None:
    """Plot drawdown curves."""

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))

    for strategy in portfolio_values.columns:

        drawdown = calculate_drawdown(
            portfolio_values[strategy]
        )

        plt.plot(
            drawdown.index,
            drawdown,
            label=strategy,
        )

    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    plt.savefig(REPORT_DIR / "drawdowns.png")
    plt.close()
# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_summary(summary: pd.DataFrame) -> None:
    """Save the summary table."""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        REPORT_DIR / "summary.csv",
        index=False,
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    returns = load_returns()

    portfolio_values = load_portfolio_values()

    summary = build_summary_table(
        returns,
        portfolio_values,
    )

    save_summary(summary)

    plot_equity_curves(portfolio_values)

    plot_drawdowns(portfolio_values)

    print("\nPortfolio Performance Summary")
    print(summary.round(4))


if __name__ == "__main__":
    main()