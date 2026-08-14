"""Visualization graphics for walk-forward portfolio reporting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

STRATEGY_COLORS: dict[str, str] = {
    "Max Sharpe": "#e74c3c",
    "Equal Weight": "#2980b9",
    "Maximum Diversification": "#8e44ad",
    "Risk Parity": "#27ae60",
    "Inverse Volatility": "#e67e22",
    "Minimum Variance": "#16a085",
    "Hierarchical Risk Parity": "#d35400",
    "SPY": "#34495e",
}


def _apply_theme() -> None:
    """Configure clean aesthetics for plots."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.labelweight": "bold",
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
        }
    )


def plot_equity_curves(portfolio_values: pd.DataFrame, save_path: Path) -> None:
    """Plot cumulative walk-forward equity curves vs SPY benchmark."""
    _apply_theme()
    _fig, ax = plt.subplots(figsize=(12, 7))

    for strategy in portfolio_values.columns:
        color = STRATEGY_COLORS.get(strategy, "#7f8c8d")
        lw = 2.5 if strategy in ("Max Sharpe", "Equal Weight", "SPY") else 1.8
        ls = "--" if strategy == "SPY" else "-"
        ax.plot(
            portfolio_values.index,
            portfolio_values[strategy],
            label=strategy,
            color=color,
            linewidth=lw,
            linestyle=ls,
            alpha=0.9,
        )

        final_val = portfolio_values[strategy].iloc[-1]
        ax.annotate(
            f" ${final_val:,.0f}",
            xy=(portfolio_values.index[-1], final_val),
            xytext=(6, -2),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
            color=color,
        )

    ax.set_title("Walk-Forward Out-of-Sample Portfolio Growth ($10,000 Initial Capital)", pad=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9, fontsize=9.5)
    ax.set_xlim(portfolio_values.index[0], portfolio_values.index[-1] + pd.DateOffset(months=4))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved -> {save_path}")


def plot_drawdowns(portfolio_values: pd.DataFrame, save_path: Path) -> None:
    """Plot cumulative underwater drawdown curves across all strategies."""
    _apply_theme()
    _fig, ax = plt.subplots(figsize=(12, 6))

    for strategy in portfolio_values.columns:
        drawdown = (portfolio_values[strategy] / portfolio_values[strategy].cummax()) - 1.0
        color = STRATEGY_COLORS.get(strategy, "#7f8c8d")
        lw = 2.0 if strategy in ("Max Sharpe", "Equal Weight", "SPY") else 1.4
        ls = "--" if strategy == "SPY" else "-"
        ax.plot(
            drawdown.index,
            drawdown.mul(100.0),
            label=strategy,
            color=color,
            linewidth=lw,
            linestyle=ls,
            alpha=0.85,
        )

    ax.set_title("Underwater Portfolio Drawdown Profile (%)", pad=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.axhline(0, color="#7f8c8d", linewidth=0.8)
    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9, fontsize=9.5)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved -> {save_path}")
