"""Visualization graphics for walk-forward portfolio reporting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


def plot_dividend_returns_breakdown(div_summary_path: Path, save_path: Path) -> None:
    """Plot stacked bar chart of Capital Growth + Cash Dividends Earned."""
    if not div_summary_path.exists():
        return

    df = pd.read_csv(div_summary_path)
    _apply_theme()
    _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10), sharex=True)

    strategies = df["Strategy"]
    x = np.arange(len(strategies))
    bar_width = 0.55

    capital_vals = df["Capital Value ($)"]
    div_vals = df["Cash Dividends Earned ($)"]
    total_vals = df["Total Portfolio Value ($)"]
    returns_pct = df["Total Return (Cash Div %)"]

    ax1.bar(x, capital_vals, bar_width, label="Capital Value ($)", color="#2b5c8f", edgecolor="none")
    ax1.bar(
        x,
        div_vals,
        bar_width,
        bottom=capital_vals,
        label="Cash Dividends Earned ($)",
        color="#27ae60",
        edgecolor="none",
    )

    ax1.axhline(10000, color="#e74c3c", linestyle="--", linewidth=1.5, label="Initial Capital ($10,000)")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.set_title("Walk-Forward Total Portfolio Value Breakdown (2020-2025)", pad=12)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    ax1.set_ylim(0, max(total_vals) * 1.18)

    for i in range(len(strategies)):
        tot = total_vals.iloc[i]
        ret = returns_pct.iloc[i]
        ax1.text(
            x[i],
            tot + 400,
            f"${tot:,.2f}\n({ret:+.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    bars_div = ax2.bar(x, div_vals, bar_width, color="#2e7d32", alpha=0.85)
    ax2.set_ylabel("Cash Dividends ($)")
    ax2.set_title("Total Cash Dividends Collected (2020-2025)", pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, rotation=15, ha="right", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, max(div_vals) * 1.25)

    for bar in bars_div:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 30,
            f"${h:,.2f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved -> {save_path}")
