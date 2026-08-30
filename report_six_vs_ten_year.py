"""Build six-year versus ten-year strategy comparison tables and figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from data_pipeline import write_csv_outputs_atomically

ROOT = Path(__file__).resolve().parent
SIX_YEAR_SELECTION_DIR = ROOT / "outputs" / "selection_experiment"
SIX_YEAR_PORTFOLIO_DIR = ROOT / "outputs" / "portfolio_backtest"
TEN_YEAR_DIR = ROOT / "outputs" / "selection_experiment_10y"
COMPARISON_DIR = TEN_YEAR_DIR / "comparison"

DISPLAY_ORDER = (
    "Partitioning Selection",
    "Density Selection",
    "Eligible Universe Equal Weight",
    "Max Sharpe",
    "Equal Weight",
    "Maximum Diversification",
    "Risk Parity",
    "Inverse Volatility",
    "Hierarchical Risk Parity",
    "Minimum Variance",
    "SPY",
)

DISPLAY_NAMES = {
    "Partitioning Selection": "PAM Partitioning",
    "Density Selection": "HDBSCAN Density",
    "Eligible Universe Equal Weight": "Eligible-universe EW",
    "Maximum Diversification": "Maximum Diversification",
    "Hierarchical Risk Parity": "Hierarchical Risk Parity",
}

COLORS = {
    "Partitioning Selection": "#D97706",
    "Density Selection": "#DB2777",
    "Eligible Universe Equal Weight": "#6B7C2F",
    "Max Sharpe": "#1D4ED8",
    "Equal Weight": "#2563EB",
    "Maximum Diversification": "#3B82F6",
    "Risk Parity": "#60A5FA",
    "Inverse Volatility": "#93C5FD",
    "Hierarchical Risk Parity": "#64748B",
    "Minimum Variance": "#94A3B8",
    "SPY": "#111827",
}


def _load_returns(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index.name = "Date"
    return frame


def load_six_year_returns() -> pd.DataFrame:
    allocation = _load_returns(SIX_YEAR_PORTFOLIO_DIR / "walk_forward_returns.csv")
    selection = _load_returns(SIX_YEAR_SELECTION_DIR / "full" / "walk_forward_returns.csv")
    if not allocation.index.equals(selection.index):
        raise ValueError("Six-year allocation and selection dates do not match")
    for shared in ("Equal Weight", "SPY"):
        if not np.allclose(allocation[shared], selection[shared], atol=1e-12, rtol=0.0):
            raise ValueError(f"Six-year {shared} returns disagree between artifacts")
    added = [column for column in selection if column not in allocation]
    return pd.concat([allocation, selection[added]], axis=1).loc[:, DISPLAY_ORDER]


def load_ten_year_returns() -> pd.DataFrame:
    returns = _load_returns(TEN_YEAR_DIR / "full" / "walk_forward_returns.csv")
    missing = set(DISPLAY_ORDER).difference(returns.columns)
    if missing:
        raise ValueError(f"Ten-year artifacts are missing strategies: {sorted(missing)}")
    return returns.loc[:, DISPLAY_ORDER]


def build_comparison_table() -> pd.DataFrame:
    six = pd.read_csv(SIX_YEAR_SELECTION_DIR / "all_strategy_summary.csv").set_index("Strategy")
    ten = pd.read_csv(TEN_YEAR_DIR / "full" / "summary.csv").set_index("Strategy")
    missing_six = set(DISPLAY_ORDER).difference(six.index)
    missing_ten = set(DISPLAY_ORDER).difference(ten.index)
    if missing_six or missing_ten:
        raise ValueError(
            f"Missing strategies: six-year={sorted(missing_six)}, ten-year={sorted(missing_ten)}"
        )

    comparison = pd.DataFrame(index=DISPLAY_ORDER)
    comparison.index.name = "Strategy"
    for metric in ("CAGR", "Volatility", "Sharpe Ratio", "Maximum Drawdown"):
        comparison[f"6Y {metric}"] = six.loc[list(DISPLAY_ORDER), metric].astype(float)
        comparison[f"10Y {metric}"] = ten.loc[list(DISPLAY_ORDER), metric].astype(float)
    comparison["CAGR Change"] = comparison["10Y CAGR"] - comparison["6Y CAGR"]
    comparison["Sharpe Change"] = (
        comparison["10Y Sharpe Ratio"] - comparison["6Y Sharpe Ratio"]
    )
    comparison["6Y Final Value ($)"] = six.loc[list(DISPLAY_ORDER), "Final Value ($)"].astype(float)
    comparison["10Y Final Value ($)"] = 10_000.0 * (
        1.0 + ten.loc[list(DISPLAY_ORDER), "Total Return"].astype(float)
    )
    comparison["6Y Turnover"] = six.loc[list(DISPLAY_ORDER), "Turnover"].astype(float)
    comparison["10Y Turnover"] = ten.loc[
        list(DISPLAY_ORDER), "Average Recurring Turnover"
    ].astype(float)
    return comparison.sort_values("10Y CAGR", ascending=False)


def _growth(returns: pd.DataFrame, start: str) -> pd.DataFrame:
    growth = 10_000.0 * (1.0 + returns).cumprod()
    start_row = pd.DataFrame(10_000.0, index=[pd.Timestamp(start)], columns=growth.columns)
    return pd.concat([start_row, growth])


def plot_equity_curves(six_returns: pd.DataFrame, ten_returns: pd.DataFrame, target: Path) -> None:
    six = _growth(six_returns, "2020-01-01")
    ten = _growth(ten_returns, "2016-01-01")
    figure, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)

    for axis, values, title in (
        (axes[0], six, "Six-year test: 2020-2025"),
        (axes[1], ten, "Ten-year test: 2016-2025"),
    ):
        for strategy in DISPLAY_ORDER:
            emphasized = strategy in {"Partitioning Selection", "Density Selection", "SPY"}
            axis.plot(
                values.index,
                values[strategy],
                color=COLORS[strategy],
                linewidth=2.8 if emphasized else 1.45,
                linestyle="--" if strategy == "SPY" else "-",
                alpha=1.0 if emphasized else 0.80,
                label=DISPLAY_NAMES.get(strategy, strategy),
            )
        axis.set_title(title, fontsize=14, fontweight="bold", loc="left")
        axis.set_yscale("log", base=2)
        axis.grid(True, which="major", color="#D1D5DB", linewidth=0.8, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlabel("Holding date")

    axes[0].set_ylabel("Portfolio value from $10,000 (log₂ scale)")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value/1000:.0f}k"))
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    figure.suptitle("All Portfolio Strategies: Six-Year and Ten-Year Equity Curves", fontsize=18)
    figure.text(
        0.5,
        0.925,
        "Same logarithmic value scale; net of modeled 10 bps turnover costs. Selectors are emphasized.",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    figure.tight_layout(rect=(0, 0.12, 1, 0.90))
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_metric_comparison(comparison: pd.DataFrame, target: Path) -> None:
    order = list(comparison.index[::-1])
    labels = [DISPLAY_NAMES.get(strategy, strategy) for strategy in order]
    y = np.arange(len(order))
    height = 0.34
    figure, axes = plt.subplots(1, 3, figsize=(20, 9), sharey=True)
    specifications = (
        ("CAGR", "Annualized return", lambda value: f"{value:.1%}"),
        ("Sharpe Ratio", "Sharpe ratio", lambda value: f"{value:.2f}"),
        ("Maximum Drawdown", "Maximum drawdown magnitude", lambda value: f"{value:.1%}"),
    )

    for axis, (metric, title, formatter) in zip(axes, specifications, strict=True):
        six = comparison.loc[order, f"6Y {metric}"].to_numpy(dtype=float)
        ten = comparison.loc[order, f"10Y {metric}"].to_numpy(dtype=float)
        if metric == "Maximum Drawdown":
            six = np.abs(six)
            ten = np.abs(ten)
        axis.barh(
            y - height / 2,
            six,
            height,
            facecolor="white",
            edgecolor="#2563EB",
            linewidth=1.5,
            label="6 years",
        )
        axis.barh(
            y + height / 2,
            ten,
            height,
            color="#D97706",
            edgecolor="#92400E",
            linewidth=0.8,
            label="10 years",
        )
        max_value = max(float(np.max(six)), float(np.max(ten)))
        axis.set_xlim(0.0, max_value * 1.28 if max_value else 1.0)
        for position, value in zip(y - height / 2, six, strict=True):
            axis.text(value + max_value * 0.018, position, formatter(value), va="center", fontsize=8)
        for position, value in zip(y + height / 2, ten, strict=True):
            axis.text(value + max_value * 0.018, position, formatter(value), va="center", fontsize=8)
        axis.set_title(title, fontsize=13, fontweight="bold")
        axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0].set_yticks(y, labels)
    axes[1].legend(loc="lower right", frameon=False)
    figure.suptitle("Strategy Metrics: Six-Year vs Ten-Year Tests", fontsize=18)
    figure.text(
        0.5,
        0.935,
        "Rows ordered by ten-year CAGR. Drawdown is shown as loss magnitude; smaller is better.",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_table(comparison: pd.DataFrame, target: Path) -> None:
    columns = [
        "Strategy",
        "6Y CAGR",
        "10Y CAGR",
        "6Y Sharpe",
        "10Y Sharpe",
        "6Y Max DD",
        "10Y Max DD",
    ]
    rows: list[list[str]] = []
    strategies = list(comparison.index)
    for strategy in strategies:
        row = comparison.loc[strategy]
        rows.append(
            [
                DISPLAY_NAMES.get(strategy, strategy),
                f"{row['6Y CAGR']:.2%}",
                f"{row['10Y CAGR']:.2%}",
                f"{row['6Y Sharpe Ratio']:.3f}",
                f"{row['10Y Sharpe Ratio']:.3f}",
                f"{row['6Y Maximum Drawdown']:.2%}",
                f"{row['10Y Maximum Drawdown']:.2%}",
            ]
        )

    figure, axis = plt.subplots(figsize=(16, 7.8))
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="right",
        colLoc="right",
        loc="center",
        colWidths=[0.29, 0.115, 0.115, 0.115, 0.115, 0.12, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.65)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        cell.set_linewidth(0.6)
        if row_index == 0:
            cell.set_facecolor("#1F2937")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            strategy = strategies[row_index - 1]
            if strategy == "Partitioning Selection":
                cell.set_facecolor("#FFF7ED")
            elif strategy == "Density Selection":
                cell.set_facecolor("#FDF2F8")
            elif strategy == "SPY":
                cell.set_facecolor("#F3F4F6")
            else:
                cell.set_facecolor("white" if row_index % 2 else "#F9FAFB")
        if column_index == 0:
            cell.get_text().set_ha("left")

    figure.suptitle("All Strategies: Six-Year and Ten-Year Results", fontsize=18, y=0.96)
    figure.text(
        0.5,
        0.06,
        "Ten-year selectors use the available-data universe; 2016-2019 remain below the original 80% fundamental-coverage gate.",
        ha="center",
        fontsize=9.5,
        color="#4B5563",
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def generate_comparison() -> pd.DataFrame:
    comparison = build_comparison_table()
    output_csv = COMPARISON_DIR / "strategy_comparison_6y_vs_10y.csv"
    write_csv_outputs_atomically({output_csv: (comparison, {})})
    plot_equity_curves(
        load_six_year_returns(),
        load_ten_year_returns(),
        COMPARISON_DIR / "equity_curves_6y_vs_10y.png",
    )
    plot_metric_comparison(
        comparison,
        COMPARISON_DIR / "metrics_6y_vs_10y.png",
    )
    plot_table(
        comparison,
        COMPARISON_DIR / "strategy_table_6y_vs_10y.png",
    )
    return comparison


if __name__ == "__main__":
    print(generate_comparison().to_string())
