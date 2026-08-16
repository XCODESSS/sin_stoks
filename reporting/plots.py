"""Visualization graphics for walk-forward portfolio reporting."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from universe import get_sector

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


def plot_allocation_heatmap(
    weights: pd.DataFrame, strategy_name: str, save_path: Path, max_weight: float = 0.25
) -> None:
    """Heatmap of one strategy's weights across rebalance years with cell text and sector labels."""
    strategy_weights = weights.xs(strategy_name, level="Strategy")  # index=Rebalance Date, columns=tickers
    tickers_sorted = sorted(strategy_weights.columns, key=lambda t: (get_sector(t), t))
    grid = strategy_weights[tickers_sorted]
    data = grid.to_numpy()

    _apply_theme()
    fig, ax = plt.subplots(figsize=(15, 6))

    # Colormap with white background for zero/near-zero allocations
    cmap = plt.cm.YlGnBu.copy()
    cmap.set_under("#f8f9fa")

    v_max = max(max_weight, float(data.max())) if data.max() > 0 else max_weight
    im = ax.imshow(
        data,
        aspect="auto",
        cmap=cmap,
        vmin=0.005,
        vmax=v_max,
        interpolation="nearest",
    )

    ax.set_xticks(range(len(tickers_sorted)))
    ax.set_xticklabels(tickers_sorted, rotation=90, fontsize=8.5, fontweight="bold")
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels([pd.Timestamp(d).year for d in grid.index], fontsize=10, fontweight="bold")

    # Annotate percentage numbers inside cells
    for r in range(len(grid.index)):
        for c in range(len(tickers_sorted)):
            val = data[r, c]
            if val >= 0.005:  # Display text if >= 0.5%
                text_color = "white" if val > (v_max * 0.55) else "#1a252f"
                text_str = f"{val:.1%}" if val < 0.05 else f"{val * 100:.1f}%"
                ax.text(
                    c,
                    r,
                    text_str,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=6.8,
                    fontweight="bold",
                )

    # Sector grouping and labels above the plot
    sectors_in_order = [get_sector(t) for t in tickers_sorted]
    sector_starts: dict[str, list[int]] = {}
    for i, s in enumerate(sectors_in_order):
        sector_starts.setdefault(s, []).append(i)

    for sector, indices in sector_starts.items():
        start_idx = indices[0]
        end_idx = indices[-1]
        if start_idx > 0:
            ax.axvline(start_idx - 0.5, color="#2c3e50", linewidth=1.5, linestyle="-")

        mid_idx = (start_idx + end_idx) / 2.0
        ax.text(
            mid_idx,
            -0.75,
            sector,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#2c3e50",
        )

    ax.set_title(
        f"{strategy_name} — Asset Allocation by Rebalance Year", pad=28, fontsize=13, fontweight="bold"
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Target Portfolio Weight", fontsize=10, fontweight="bold")
    cbar.formatter = mticker.PercentFormatter(1.0, decimals=0)
    cbar.update_ticks()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved -> {save_path}")


def plot_sector_correlation_heatmap(sector_correlation: pd.DataFrame, save_path: Path) -> None:
    """6x6 Behavioral Vice Sector correlation heatmap with in-cell numerical annotations."""
    sectors = list(sector_correlation.index)
    data = sector_correlation.to_numpy()

    _apply_theme()
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    im = ax.imshow(data, cmap="YlGnBu", vmin=0.15, vmax=1.0, interpolation="nearest")

    ax.set_xticks(range(len(sectors)))
    ax.set_xticklabels(sectors, rotation=25, ha="right", fontsize=9.5, fontweight="bold")
    ax.set_yticks(range(len(sectors)))
    ax.set_yticklabels(sectors, fontsize=9.5, fontweight="bold")

    # Annotate correlation numbers inside cells
    for r in range(len(sectors)):
        for c in range(len(sectors)):
            val = data[r, c]
            text_color = "white" if val > 0.65 else "#1a252f"
            ax.text(
                c,
                r,
                f"{val:.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10.5,
                fontweight="bold",
            )

    ax.set_title("Behavioral Vice Sector Correlation Matrix (6x6)", pad=16, fontsize=13, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Correlation Coefficient", fontsize=10, fontweight="bold")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved -> {save_path}")


def plot_strategy_correlation_heatmap(strategy_returns: pd.DataFrame, save_path: Path) -> None:
    """7x7 Strategy Return correlation heatmap with in-cell values."""
    corr = strategy_returns.corr()
    strategies = list(corr.index)
    data = corr.to_numpy()

    _apply_theme()
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(data, cmap="Blues", vmin=0.50, vmax=1.0, interpolation="nearest")

    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=28, ha="right", fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies, fontsize=9, fontweight="bold")

    # Annotate correlation values inside cells
    for r in range(len(strategies)):
        for c in range(len(strategies)):
            val = data[r, c]
            text_color = "white" if val > 0.82 else "#1a252f"
            ax.text(
                c,
                r,
                f"{val:.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9.5,
                fontweight="bold",
            )

    ax.set_title(
        "Strategy Return Correlation Matrix - Weekly Log Returns (7x7)",
        pad=16,
        fontsize=13,
        fontweight="bold",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Correlation Coefficient", fontsize=10, fontweight="bold")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved -> {save_path}")
