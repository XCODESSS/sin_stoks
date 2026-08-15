"""Interactive Plotly HTML dashboard for walk-forward portfolio reporting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

STRATEGY_COLORS: dict[str, str] = {
    "Max Sharpe": "#e74c3c",
    "Equal Weight": "#2980b9",
    "Maximum Diversification": "#8e44ad",
    "Risk Parity": "#27ae60",
    "Inverse Volatility": "#e67e22",
    "Minimum Variance": "#16a085",
    "Hierarchical Risk Parity": "#d35400",
    "SPY": "#5d6d7e",
}


def _format_pct(val: float) -> str:
    return f"{val:.2%}"


def _format_money(val: float) -> str:
    return f"${val:,.2f}"


def _format_ratio(val: float) -> str:
    return f"{val:.4f}"


def build_interactive_equity_curves(portfolio_values: pd.DataFrame) -> go.Figure:
    """Build interactive equity curve chart with hover tooltips."""
    fig = go.Figure()

    for strategy in portfolio_values.columns:
        vals = portfolio_values[strategy]
        color = STRATEGY_COLORS.get(strategy, "#95a5a6")
        dash = "dash" if strategy == "SPY" else "solid"
        width = 3.0 if strategy in ("Max Sharpe", "Equal Weight", "SPY") else 2.0

        fig.add_trace(
            go.Scatter(
                x=vals.index,
                y=vals.values,
                name=strategy,
                mode="lines",
                line=dict(color=color, width=width, dash=dash),
                hovertemplate=(
                    f"<b>{strategy}</b><br>Date: %{{x|%b %d, %Y}}<br>Value: $%{{y:,.2f}}<br><extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text="Walk-Forward Out-of-Sample Portfolio Growth",
            font=dict(size=20, family="Inter, sans-serif"),
            x=0.5,
        ),
        xaxis=dict(title="Date", showgrid=True, gridcolor="rgba(200,200,200,0.3)"),
        yaxis=dict(
            title="Portfolio Value ($)",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
            tickprefix="$",
            tickformat=",",
        ),
        template="plotly_dark",
        plot_bgcolor="rgba(20,20,30,1)",
        paper_bgcolor="rgba(15,15,25,1)",
        font=dict(family="Inter, sans-serif", color="#e0e0e0"),
        legend=dict(
            bgcolor="rgba(30,30,45,0.9)",
            bordercolor="rgba(100,100,120,0.4)",
            borderwidth=1,
            font=dict(size=12),
        ),
        hovermode="x unified",
        height=550,
        margin=dict(l=70, r=30, t=70, b=50),
    )
    return fig


def build_interactive_drawdowns(portfolio_values: pd.DataFrame) -> go.Figure:
    """Build interactive drawdown chart with hover tooltips."""
    fig = go.Figure()

    for strategy in portfolio_values.columns:
        vals = portfolio_values[strategy]
        dd = (vals / vals.cummax()) - 1.0
        color = STRATEGY_COLORS.get(strategy, "#95a5a6")
        dash = "dash" if strategy == "SPY" else "solid"
        width = 2.5 if strategy in ("Max Sharpe", "Equal Weight", "SPY") else 1.8

        fig.add_trace(
            go.Scatter(
                x=dd.index,
                y=dd.values * 100,
                name=strategy,
                mode="lines",
                line=dict(color=color, width=width, dash=dash),
                fill="tozeroy",
                fillcolor=color.replace(")", ",0.05)").replace("rgb", "rgba")
                if color.startswith("rgb")
                else f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.05)",
                hovertemplate=(
                    f"<b>{strategy}</b><br>Date: %{{x|%b %d, %Y}}<br>Drawdown: %{{y:.2f}}%<br><extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text="Underwater Portfolio Drawdown Profile",
            font=dict(size=20, family="Inter, sans-serif"),
            x=0.5,
        ),
        xaxis=dict(title="Date", showgrid=True, gridcolor="rgba(200,200,200,0.3)"),
        yaxis=dict(
            title="Drawdown (%)",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
            ticksuffix="%",
        ),
        template="plotly_dark",
        plot_bgcolor="rgba(20,20,30,1)",
        paper_bgcolor="rgba(15,15,25,1)",
        font=dict(family="Inter, sans-serif", color="#e0e0e0"),
        legend=dict(
            bgcolor="rgba(30,30,45,0.9)",
            bordercolor="rgba(100,100,120,0.4)",
            borderwidth=1,
            font=dict(size=12),
        ),
        hovermode="x unified",
        height=500,
        margin=dict(l=70, r=30, t=70, b=50),
    )
    fig.add_hline(y=0, line_color="rgba(150,150,150,0.5)", line_width=1)
    return fig


def build_interactive_weights(weights: pd.DataFrame) -> go.Figure:
    """Build a grouped bar chart showing strategy asset allocations at each rebalance."""
    if weights.empty:
        return go.Figure()

    fig = go.Figure()
    tickers = weights.columns.tolist()
    palette = [
        "#e74c3c",
        "#2980b9",
        "#27ae60",
        "#e67e22",
        "#8e44ad",
        "#16a085",
        "#d35400",
        "#c0392b",
        "#2c3e50",
        "#f39c12",
        "#1abc9c",
        "#9b59b6",
        "#34495e",
        "#e91e63",
        "#3f51b5",
        "#009688",
        "#ff5722",
        "#607d8b",
        "#795548",
        "#cddc39",
        "#00bcd4",
        "#ff9800",
        "#4caf50",
        "#673ab7",
        "#f44336",
        "#03a9f4",
        "#8bc34a",
        "#ffc107",
        "#9c27b0",
        "#00e676",
    ]

    strategies = weights.index.get_level_values("Strategy").unique()
    rebalance_dates = weights.index.get_level_values("Rebalance Date").unique()

    for idx, ticker in enumerate(tickers):
        color = palette[idx % len(palette)]
        for strat in strategies:
            vals = []
            labels = []
            for rd in rebalance_dates:
                if (rd, strat) in weights.index:
                    vals.append(weights.loc[(rd, strat), ticker] * 100)
                    labels.append(f"{str(rd)[:10]} — {strat}")
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=vals,
                    name=ticker,
                    marker_color=color,
                    legendgroup=ticker,
                    showlegend=(strat == strategies[0] and rebalance_dates[0] == rebalance_dates[0]),
                    hovertemplate=(f"<b>{ticker}</b><br>%{{x}}<br>Weight: %{{y:.2f}}%<br><extra></extra>"),
                )
            )

    fig.update_layout(
        barmode="stack",
        title=dict(
            text="Asset Allocation Weights by Strategy & Rebalance Date",
            font=dict(size=20, family="Inter, sans-serif"),
            x=0.5,
        ),
        xaxis=dict(title="", tickangle=-30, tickfont=dict(size=9)),
        yaxis=dict(title="Weight (%)", ticksuffix="%"),
        template="plotly_dark",
        plot_bgcolor="rgba(20,20,30,1)",
        paper_bgcolor="rgba(15,15,25,1)",
        font=dict(family="Inter, sans-serif", color="#e0e0e0"),
        legend=dict(
            bgcolor="rgba(30,30,45,0.9)",
            bordercolor="rgba(100,100,120,0.4)",
            borderwidth=1,
            font=dict(size=10),
            itemsizing="constant",
        ),
        height=550,
        margin=dict(l=60, r=30, t=70, b=100),
    )
    return fig


def build_summary_html_table(summary: pd.DataFrame) -> str:
    """Build a styled HTML summary table from the summary DataFrame."""
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        .summary-container {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0f0f19 0%, #1a1a2e 100%);
            border-radius: 16px;
            padding: 32px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.5);
        }
        .summary-title {
            color: #e0e0e0;
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 20px;
            text-align: center;
            letter-spacing: 0.5px;
        }
        .summary-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            border-radius: 12px;
            overflow: hidden;
        }
        .summary-table thead th {
            background: linear-gradient(180deg, #2a2a4a 0%, #1e1e38 100%);
            color: #b0b8c8;
            font-weight: 600;
            font-size: 12px;
            padding: 14px 16px;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            text-align: right;
            border-bottom: 2px solid rgba(100,100,150,0.3);
            white-space: nowrap;
        }
        .summary-table thead th:first-child { text-align: left; }
        .summary-table tbody tr {
            transition: background 0.2s ease;
        }
        .summary-table tbody tr:hover {
            background: rgba(80,80,130,0.15) !important;
        }
        .summary-table tbody tr:nth-child(odd)  { background: rgba(25,25,45,0.6); }
        .summary-table tbody tr:nth-child(even) { background: rgba(30,30,50,0.4); }
        .summary-table tbody td {
            padding: 12px 16px;
            font-size: 13px;
            color: #d0d0d0;
            text-align: right;
            border-bottom: 1px solid rgba(80,80,100,0.2);
            white-space: nowrap;
        }
        .summary-table tbody td:first-child {
            text-align: left;
            font-weight: 600;
            color: #ffffff;
        }
        .positive { color: #27ae60; font-weight: 600; }
        .negative { color: #e74c3c; font-weight: 600; }
        .highlight-best { color: #f1c40f; font-weight: 700; }
        .strategy-dot {
            display: inline-block;
            width: 10px; height: 10px;
            border-radius: 50%;
            margin-right: 8px;
            vertical-align: middle;
        }
        .spy-row td { font-style: italic; opacity: 0.85; }
    </style>
    """

    # Find best values for highlighting
    non_spy = summary[summary["Strategy"] != "SPY"]
    best_cagr = non_spy["CAGR"].max() if not non_spy.empty else None
    best_sharpe = non_spy["Sharpe Ratio"].max() if not non_spy.empty else None
    best_dd = non_spy["Maximum Drawdown"].max() if not non_spy.empty else None  # least negative
    best_calmar = non_spy["Calmar Ratio"].max() if not non_spy.empty else None

    rows_html = ""
    for _, row in summary.iterrows():
        strat = row["Strategy"]
        color = STRATEGY_COLORS.get(strat, "#95a5a6")
        is_spy = strat == "SPY"
        row_class = ' class="spy-row"' if is_spy else ""

        dot = f'<span class="strategy-dot" style="background:{color};"></span>'

        # Format cells with conditional classes
        cagr_cls = ' class="highlight-best"' if row["CAGR"] == best_cagr and not is_spy else ""
        sharpe_cls = ' class="highlight-best"' if row["Sharpe Ratio"] == best_sharpe and not is_spy else ""
        dd_cls = ' class="highlight-best"' if row["Maximum Drawdown"] == best_dd and not is_spy else ""
        calmar_cls = ' class="highlight-best"' if row["Calmar Ratio"] == best_calmar and not is_spy else ""

        tot_ret_cls = ' class="positive"' if row["Total Return"] > 0 else ' class="negative"'

        beta_str = f"{row['Beta']:.4f}" if "Beta" in row and pd.notna(row["Beta"]) else "&mdash;"
        alpha_str = f"{row['Alpha']:.2%}" if "Alpha" in row and pd.notna(row["Alpha"]) else "&mdash;"
        te_str = (
            f"{row['Tracking Error']:.2%}"
            if "Tracking Error" in row and pd.notna(row["Tracking Error"])
            else "&mdash;"
        )
        ir_str = (
            f"{row['Information Ratio']:.4f}"
            if "Information Ratio" in row and pd.notna(row["Information Ratio"])
            else "&mdash;"
        )

        rows_html += f"""<tr{row_class}>
            <td>{dot}{strat}</td>
            <td>{_format_money(row["Initial Value ($)"])}</td>
            <td>{_format_money(row["Final Value ($)"])}</td>
            <td{tot_ret_cls}>{_format_pct(row["Total Return"])}</td>
            <td{cagr_cls}>{_format_pct(row["CAGR"])}</td>
            <td>{_format_pct(row["Volatility"])}</td>
            <td{sharpe_cls}>{_format_ratio(row["Sharpe Ratio"])}</td>
            <td>{_format_ratio(row["Sortino Ratio"])}</td>
            <td{dd_cls}>{_format_pct(row["Maximum Drawdown"])}</td>
            <td{calmar_cls}>{_format_ratio(row["Calmar Ratio"])}</td>
            <td>{beta_str}</td>
            <td>{alpha_str}</td>
            <td>{te_str}</td>
            <td>{ir_str}</td>
        </tr>"""

    table_html = f"""
    {css}
    <div class="summary-container">
        <div class="summary-title">📊 Portfolio Performance Summary (2020-2025 Walk-Forward OOS)</div>
        <table class="summary-table">
            <thead>
                <tr>
                    <th>Strategy</th>
                    <th>Initial ($)</th>
                    <th>Final ($)</th>
                    <th>Total Return</th>
                    <th>CAGR</th>
                    <th>Volatility</th>
                    <th>Sharpe</th>
                    <th>Sortino</th>
                    <th>Max DD</th>
                    <th>Calmar</th>
                    <th>Beta</th>
                    <th>Alpha</th>
                    <th>Tracking Err</th>
                    <th>Info Ratio</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """
    return table_html


def generate_interactive_dashboard(
    portfolio_values: pd.DataFrame,
    summary: pd.DataFrame,
    weights: pd.DataFrame,
    save_path: Path,
) -> None:
    """Generate a single-page interactive HTML dashboard with all charts and summary table."""
    equity_fig = build_interactive_equity_curves(portfolio_values)
    drawdown_fig = build_interactive_drawdowns(portfolio_values)
    weights_fig = build_interactive_weights(weights)
    summary_table = build_summary_html_table(summary)

    equity_html = equity_fig.to_html(full_html=False, include_plotlyjs=False)
    drawdown_html = drawdown_fig.to_html(full_html=False, include_plotlyjs=False)
    weights_html = weights_fig.to_html(full_html=False, include_plotlyjs=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sin Stocks — Interactive Portfolio Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', sans-serif;
            background: linear-gradient(160deg, #0a0a14 0%, #111128 50%, #0d0d1a 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 24px;
        }}
        .dashboard-header {{
            text-align: center;
            padding: 40px 20px 30px;
        }}
        .dashboard-header h1 {{
            font-size: 32px;
            font-weight: 700;
            background: linear-gradient(135deg, #e74c3c, #f39c12, #2980b9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
        }}
        .dashboard-header p {{
            color: #8890a0;
            font-size: 14px;
        }}
        .section {{
            max-width: 1300px;
            margin: 0 auto 32px;
            background: rgba(20,20,35,0.7);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 8px 40px rgba(0,0,0,0.3);
            border: 1px solid rgba(60,60,90,0.3);
            backdrop-filter: blur(10px);
        }}
        .section-title {{
            font-size: 16px;
            font-weight: 600;
            color: #a0a8b8;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 16px;
            padding-left: 4px;
        }}
        .footer {{
            text-align: center;
            color: #4a4a6a;
            font-size: 12px;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="dashboard-header">
        <h1>Sin Stocks Portfolio Dashboard</h1>
        <p>Walk-Forward Out-of-Sample Performance — 7 Strategies vs SPY Benchmark</p>
    </div>

    <div class="section">
        <div class="section-title">📋 Performance Summary</div>
        {summary_table}
    </div>

    <div class="section">
        <div class="section-title">📈 Equity Curves</div>
        {equity_html}
    </div>

    <div class="section">
        <div class="section-title">📉 Drawdown Profile</div>
        {drawdown_html}
    </div>

    <div class="section">
        <div class="section-title">⚖️ Asset Allocation Weights</div>
        {weights_html}
    </div>

    <div class="footer">
        Sin Stocks Portfolio Research — Generated with Plotly
    </div>
</body>
</html>"""

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(html, encoding="utf-8")
    print(f"Saved -> {save_path}")
