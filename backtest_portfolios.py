"""Reproducible annual portfolio backtest for the behavioral-sector universe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize


INITIAL_CAPITAL = 10_000_000
RISK_FREE_RATE = 0.04
START_BACKTEST_YEAR = 2022
MAX_WEIGHT = 0.30
OUTPUT_DIR = Path("outputs/portfolio_backtest")
INPUT_FILE = Path("annual_returns_25_companies.csv")

STRATEGY_NAMES = {
    "equal_weight": "Equal Weight",
    "mean_variance": "Mean-Variance Optimization",
    "max_sharpe": "Maximum Sharpe Ratio",
    "min_variance": "Minimum Variance",
    "risk_parity": "Risk Parity",
    "kelly": "Kelly Criterion",
    "core_satellite": "Core-Satellite",
    "factor": "Factor-Based",
    "black_litterman": "Black-Litterman",
    "behavioral": "Behavioral Finance (Custom)",
}

SECTOR_VIEWS = {
    "Alcohol": 0.020,
    "Energy Drinks": 0.030,
    "Social Media": 0.040,
    "Tobacco & Nicotine": 0.025,
    "Dating Apps": 0.010,
}

CORE_SATELLITE_TARGETS = {
    "Alcohol": 0.375,
    "Tobacco & Nicotine": 0.375,
    "Energy Drinks": 0.100,
    "Social Media": 0.100,
    "Dating Apps": 0.050,
}

BEHAVIORAL_TARGETS = {
    "Alcohol": 0.250,
    "Energy Drinks": 0.180,
    "Social Media": 0.170,
    "Tobacco & Nicotine": 0.300,
    "Dating Apps": 0.100,
}

METHOD_NOTES = {
    "Equal Weight": "Holds every eligible asset equally, so its result reflects the broad universe rather than a return or risk forecast.",
    "Mean-Variance Optimization": "Maximizes estimated return while keeping forecast volatility no higher than equal weight; this can favor the strongest historical return estimates.",
    "Maximum Sharpe Ratio": "Allocates toward the highest estimated excess return per unit of covariance-adjusted risk, subject to long-only diversification limits.",
    "Minimum Variance": "Prioritizes the lowest covariance-based volatility, which usually favors lower-volatility and diversifying holdings over high-growth names.",
    "Risk Parity": "Balances estimated marginal risk contributions, so volatile stocks receive less capital and no single risk source should dominate.",
    "Kelly Criterion": "Uses a covariance-aware log-growth approximation; the long-only cap limits leverage-like concentration caused by noisy return estimates.",
    "Core-Satellite": "Reserves 75% for Alcohol and Tobacco & Nicotine, then distributes 25% across higher-growth sectors; outcome depends on whether the stable core protects the satellite drawdowns.",
    "Factor-Based": "Uses only information available in annual returns: trailing momentum, positive-return consistency, and lower volatility; it is a return-only proxy, not a fundamental quality model.",
    "Black-Litterman": "Blends an equal-weight market proxy with stated sector views, reducing the tendency of unconstrained mean-variance weights to overreact to short histories.",
    "Behavioral Finance (Custom)": "Overweights recurring-demand categories, particularly Tobacco & Nicotine and Alcohol, while retaining growth exposure; it tests the behavioral-demand thesis directly.",
}


def normalize(weights: np.ndarray) -> np.ndarray:
    """Return non-negative weights summing to one."""
    weights = np.clip(np.asarray(weights, dtype=float), 0, None)
    total = weights.sum()
    return weights / total if total > 0 else np.full(len(weights), 1 / len(weights))


def capped_normalize(scores: np.ndarray, cap: float = MAX_WEIGHT) -> np.ndarray:
    """Normalize non-negative scores while enforcing a per-asset cap."""
    weights = normalize(scores)
    for _ in range(len(weights) + 1):
        over_cap = weights > cap + 1e-10
        if not over_cap.any():
            return weights
        remaining = 1 - cap * over_cap.sum()
        if remaining <= 0:
            return np.full(len(weights), 1 / len(weights))
        weights[over_cap] = cap
        weights[~over_cap] = normalize(weights[~over_cap]) * remaining
    return normalize(weights)


def regularized_covariance(history: pd.DataFrame) -> np.ndarray:
    """Return a positive-definite covariance matrix for short annual histories."""
    covariance = history.cov().to_numpy()
    diagonal_mean = float(np.mean(np.diag(covariance)))
    ridge = max(diagonal_mean * 0.05, 1e-6)
    return covariance + np.eye(covariance.shape[0]) * ridge


def solve_weights(
    objective: Callable[[np.ndarray], float],
    asset_count: int,
    constraints: tuple[dict, ...] = (),
) -> np.ndarray:
    """Solve a long-only, fully invested allocation with a stable fallback."""
    equal_weight = np.full(asset_count, 1 / asset_count)
    result = minimize(
        objective,
        equal_weight,
        method="SLSQP",
        bounds=[(0, MAX_WEIGHT)] * asset_count,
        constraints=[{"type": "eq", "fun": lambda weights: weights.sum() - 1}, *constraints],
        options={"ftol": 1e-12, "maxiter": 1_000},
    )
    return normalize(result.x) if result.success else equal_weight


def inverse_volatility_weights(history: pd.DataFrame) -> np.ndarray:
    volatility = history.std(ddof=1).to_numpy()
    return capped_normalize(1 / np.maximum(volatility, 1e-4))


def sector_target_weights(
    history: pd.DataFrame, sectors: pd.Series, targets: dict[str, float]
) -> np.ndarray:
    """Allocate fixed sector targets, redistributing unavailable sector weight."""
    available_targets = {
        sector: target
        for sector, target in targets.items()
        if (sectors == sector).any()
    }
    target_total = sum(available_targets.values())
    weights = np.zeros(len(history.columns))
    for sector, target in available_targets.items():
        mask = (sectors == sector).to_numpy()
        sector_history = history.loc[:, mask]
        weights[mask] = inverse_volatility_weights(sector_history) * target / target_total
    return normalize(weights)


def factor_weights(history: pd.DataFrame) -> np.ndarray:
    annual_returns = history.to_numpy()
    momentum = np.prod(1 + annual_returns, axis=0) ** (1 / len(history)) - 1
    positive_frequency = (annual_returns > 0).mean(axis=0)
    inverse_volatility = 1 / np.maximum(annual_returns.std(axis=0, ddof=1), 1e-4)

    def percentile(values: np.ndarray) -> np.ndarray:
        return pd.Series(values).rank(pct=True).to_numpy()

    composite = (
        0.50 * percentile(momentum)
        + 0.30 * percentile(inverse_volatility)
        + 0.20 * percentile(positive_frequency)
    )
    return capped_normalize(composite)


def black_litterman_weights(history: pd.DataFrame, sectors: pd.Series) -> np.ndarray:
    covariance = regularized_covariance(history)
    asset_count = len(history.columns)
    market_weights = np.full(asset_count, 1 / asset_count)
    risk_aversion = 2.5
    tau = 0.05
    equilibrium_returns = risk_aversion * covariance @ market_weights
    views = np.array([SECTOR_VIEWS[sector] for sector in sectors])
    omega = np.diag(np.maximum(np.diag(tau * covariance), 1e-6))
    posterior_precision = np.linalg.inv(tau * covariance) + np.linalg.inv(omega)
    posterior_returns = np.linalg.solve(
        posterior_precision,
        np.linalg.inv(tau * covariance) @ equilibrium_returns
        + np.linalg.inv(omega) @ (equilibrium_returns + views),
    )
    return maximum_sharpe_weights(posterior_returns, covariance)


def maximum_sharpe_weights(expected_returns: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    excess_returns = expected_returns - RISK_FREE_RATE

    def objective(weights: np.ndarray) -> float:
        volatility = np.sqrt(max(weights @ covariance @ weights, 1e-12))
        return -(weights @ excess_returns) / volatility

    return solve_weights(objective, len(expected_returns))


def calculate_strategy_weights(history: pd.DataFrame, sectors: pd.Series) -> dict[str, np.ndarray]:
    expected_returns = history.mean().to_numpy()
    covariance = regularized_covariance(history)
    asset_count = len(expected_returns)
    equal_weight = np.full(asset_count, 1 / asset_count)
    equal_variance = float(equal_weight @ covariance @ equal_weight)

    mean_variance = solve_weights(
        lambda weights: -(weights @ expected_returns),
        asset_count,
        constraints=(
            {
                "type": "ineq",
                "fun": lambda weights: equal_variance - weights @ covariance @ weights,
            },
        ),
    )
    minimum_variance = solve_weights(
        lambda weights: weights @ covariance @ weights,
        asset_count,
    )
    risk_parity = solve_weights(
        lambda weights: np.square(
            weights * (covariance @ weights) / max(weights @ covariance @ weights, 1e-12)
            - 1 / asset_count
        ).sum(),
        asset_count,
    )
    kelly = solve_weights(
        lambda weights: -(
            weights @ (expected_returns - RISK_FREE_RATE)
            - 0.5 * weights @ covariance @ weights
        ),
        asset_count,
    )

    return {
        "equal_weight": equal_weight,
        "mean_variance": mean_variance,
        "max_sharpe": maximum_sharpe_weights(expected_returns, covariance),
        "min_variance": minimum_variance,
        "risk_parity": risk_parity,
        "kelly": kelly,
        "core_satellite": sector_target_weights(history, sectors, CORE_SATELLITE_TARGETS),
        "factor": factor_weights(history),
        "black_litterman": black_litterman_weights(history, sectors),
        "behavioral": sector_target_weights(history, sectors, BEHAVIORAL_TARGETS),
    }


def performance_statistics(returns: pd.Series) -> dict[str, object]:
    values = (1 + returns).cumprod() * INITIAL_CAPITAL
    values = pd.concat([pd.Series({returns.index.min() - 1: INITIAL_CAPITAL}), values])
    years = len(returns)
    total_return = values.iloc[-1] / INITIAL_CAPITAL - 1
    cagr = (values.iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1
    annualized_volatility = returns.std(ddof=1)
    sharpe = (returns.mean() - RISK_FREE_RATE) / annualized_volatility
    downside_deviation = np.sqrt(np.mean(np.minimum(returns - RISK_FREE_RATE, 0) ** 2))
    sortino = (returns.mean() - RISK_FREE_RATE) / downside_deviation if downside_deviation else np.nan
    drawdowns = values / values.cummax() - 1
    maximum_drawdown = drawdowns.min()
    calmar = cagr / abs(maximum_drawdown) if maximum_drawdown < 0 else np.nan
    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Maximum Drawdown": maximum_drawdown,
        "Calmar Ratio": calmar,
        "Best Year": int(returns.idxmax()),
        "Best Year Return": returns.max(),
        "Worst Year": int(returns.idxmin()),
        "Worst Year Return": returns.min(),
        "Final Portfolio Value": values.iloc[-1],
    }


def build_report(
    summary: pd.DataFrame,
    annual_returns: pd.DataFrame,
    final_allocations: pd.DataFrame,
    coverage: pd.DataFrame,
) -> str:
    lines = [
        "# Portfolio Backtest Report",
        "",
        "## Decision Summary",
        "",
        f"The backtest starts with ₹{INITIAL_CAPITAL:,.0f} and uses annual rebalancing from 2022 through 2025. The ranking averages CAGR, Sharpe Ratio, lower volatility, and shallower maximum drawdown; Total Return and Final Portfolio Value are presented but excluded from the composite because they duplicate CAGR over a fixed horizon.",
        "",
        "## Strategy Comparison",
        "",
        summary.to_markdown(index=False, floatfmt=".2f"),
        "",
        "## Method Notes",
        "",
    ]
    for strategy, note in METHOD_NOTES.items():
        strategy_allocations = final_allocations.loc[
            final_allocations["Strategy"] == strategy
        ].sort_values("Weight", ascending=False)
        top_holding = strategy_allocations.iloc[0]
        lines.extend(
            [
                f"### {strategy}",
                f"{note} In the final rebalance it assigned its largest allocation to {top_holding['Ticker']} ({top_holding['Weight']:.1%}); the reported outcome compounds its four annual out-of-sample returns.",
                "",
            ]
        )
    lines.extend(
        [
            "## Data and Method Limitations",
            "",
            "- The historical source contains annual returns only, so all covariance, factor, and drawdown estimates are based on a short, low-frequency sample.",
            "- We use a 4.0% annual USD risk-free-rate assumption because the source securities are primarily USD-listed; it affects Sharpe, Sortino, Kelly, and maximum-Sharpe weights.",
            "- Stock availability is dynamic and requires at least two prior annual returns plus a current-year return. Spark Networks (LOV) has no downloadable history, while Reddit (RDDT) lacks enough pre-2025 history to enter the test; both receive 0% in the displayed final allocation.",
            "- Pernod Ricard and MTY Food Group use current source tickers RI.PA and MTY.TO respectively. Results exclude fees, taxes, trading frictions, FX conversion, and market-cap weights.",
            "",
            "## Coverage",
            "",
            coverage.to_markdown(index=False),
            "",
            "## Annual Portfolio Returns",
            "",
            annual_returns.to_markdown(index=False, floatfmt=".2%"),
        ]
    )
    return "\n".join(lines)


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    source = pd.read_csv(INPUT_FILE).set_index("Ticker")
    metadata = source[["Company", "Sector"]].copy()
    yearly_returns = source.drop(columns=["Company", "Sector"]).T
    yearly_returns.index = yearly_returns.index.astype(int)
    yearly_returns = yearly_returns.apply(pd.to_numeric, errors="coerce").div(100)
    backtest_years = [year for year in yearly_returns.index if year >= START_BACKTEST_YEAR]
    strategy_returns = pd.DataFrame(index=backtest_years, columns=STRATEGY_NAMES.values(), dtype=float)
    weight_records: list[dict[str, object]] = []

    for year in backtest_years:
        history = yearly_returns.loc[yearly_returns.index < year]
        eligible = history.columns[
            (history.notna().sum() >= 2) & yearly_returns.loc[year].notna()
        ]
        eligible_history = history.loc[:, eligible].apply(
            lambda column: column.fillna(column.mean()), axis=0
        )
        eligible_sectors = metadata.loc[eligible_history.columns, "Sector"]
        weights_by_strategy = calculate_strategy_weights(eligible_history, eligible_sectors)

        for strategy_key, strategy_weights in weights_by_strategy.items():
            strategy_name = STRATEGY_NAMES[strategy_key]
            full_weights = pd.Series(0.0, index=metadata.index)
            full_weights.loc[eligible_history.columns] = strategy_weights
            portfolio_return = float(full_weights @ yearly_returns.loc[year].fillna(0))
            strategy_returns.loc[year, strategy_name] = portfolio_return
            for ticker, weight in full_weights.items():
                weight_records.append(
                    {
                        "Year": year,
                        "Strategy": strategy_name,
                        "Ticker": ticker,
                        "Company": metadata.loc[ticker, "Company"],
                        "Sector": metadata.loc[ticker, "Sector"],
                        "Weight": weight,
                        "Target Allocation (₹)": weight * INITIAL_CAPITAL,
                        "Eligible": ticker in eligible_history.columns,
                    }
                )

    weight_schedule = pd.DataFrame(weight_records)
    statistics = pd.DataFrame(
        {strategy: performance_statistics(strategy_returns[strategy]) for strategy in strategy_returns}
    ).T
    statistics.index.name = "Strategy"
    summary = statistics.reset_index()
    rank_columns = ["CAGR", "Sharpe Ratio", "Annualized Volatility", "Maximum Drawdown"]
    summary["CAGR Rank"] = summary["CAGR"].rank(ascending=False, method="min")
    summary["Sharpe Rank"] = summary["Sharpe Ratio"].rank(ascending=False, method="min")
    summary["Volatility Rank"] = summary["Annualized Volatility"].rank(ascending=True, method="min")
    summary["Drawdown Rank"] = summary["Maximum Drawdown"].abs().rank(ascending=True, method="min")
    summary["Average Risk-Return Rank"] = summary[[f"{column} Rank" for column in ["CAGR", "Sharpe", "Volatility", "Drawdown"]]].mean(axis=1)
    summary["Overall Rank"] = summary["Average Risk-Return Rank"].rank(ascending=True, method="min").astype(int)
    summary = summary.sort_values(["Overall Rank", "CAGR"], ascending=[True, False]).reset_index(drop=True)

    final_year = max(backtest_years)
    final_allocations = weight_schedule.loc[weight_schedule["Year"] == final_year].copy()
    annual_values = (1 + strategy_returns).cumprod().mul(INITIAL_CAPITAL)
    annual_values.index.name = "Year"
    annual_returns_for_export = strategy_returns.copy()
    annual_returns_for_export.index.name = "Year"
    coverage = pd.DataFrame(
        {
            "Ticker": metadata.index,
            "Company": metadata["Company"].values,
            "Sector": metadata["Sector"].values,
            "Annual Returns Available": yearly_returns.notna().sum().values,
            "Used in Final Backtest Year": metadata.index.isin(
                final_allocations.loc[final_allocations["Eligible"], "Ticker"]
            ),
        }
    )
    methodology = pd.DataFrame(
        [
            ("Initial capital", f"₹{INITIAL_CAPITAL:,.0f}"),
            ("Backtest years", f"{min(backtest_years)}-{max(backtest_years)}"),
            ("Rebalancing", "Annual; weights use only prior annual returns"),
            ("Risk-free rate", f"{RISK_FREE_RATE:.1%} annual USD assumption"),
            ("Constraints", f"Long-only, fully invested, {MAX_WEIGHT:.0%} maximum single-stock weight"),
            ("Covariance treatment", "Sample covariance plus 5% diagonal ridge"),
            ("Core-Satellite", "75% Alcohol/Tobacco & Nicotine; 25% growth sectors"),
            ("Black-Litterman prior", "Equal-weight market proxy; sector views from 1.0% to 4.0%"),
            ("Factor model", "50% momentum, 30% inverse volatility, 20% positive-return frequency"),
        ],
        columns=["Assumption", "Value"],
    )
    checks = pd.DataFrame(
        [
            ("Each strategy has four annual returns", strategy_returns.notna().all().all()),
            ("Every annual weight set sums to 100%", np.allclose(weight_schedule.groupby(["Year", "Strategy"])["Weight"].sum(), 1.0)),
            ("Every annual target allocation sums to ₹1 Crore", np.allclose(weight_schedule.groupby(["Year", "Strategy"])["Target Allocation (₹)"].sum(), INITIAL_CAPITAL)),
            ("No negative weights", (weight_schedule["Weight"] >= -1e-10).all()),
            ("No weight exceeds cap", (weight_schedule["Weight"] <= MAX_WEIGHT + 1e-10).all()),
        ],
        columns=["Check", "Pass"],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "strategy_summary.csv", index=False)
    annual_returns_for_export.reset_index().to_csv(OUTPUT_DIR / "annual_portfolio_returns.csv", index=False)
    annual_values.reset_index().to_csv(OUTPUT_DIR / "annual_portfolio_values.csv", index=False)
    final_allocations.to_csv(OUTPUT_DIR / "final_allocations.csv", index=False)
    weight_schedule.to_csv(OUTPUT_DIR / "annual_weight_schedule.csv", index=False)
    coverage.to_csv(OUTPUT_DIR / "coverage.csv", index=False)
    checks.to_csv(OUTPUT_DIR / "checks.csv", index=False)

    report = build_report(summary, annual_returns_for_export.reset_index(), final_allocations, coverage)
    (OUTPUT_DIR / "portfolio_backtest_report.md").write_text(report, encoding="utf-8")

    workbook_data = {
        "summary": summary.replace({np.nan: None}).to_dict(orient="records"),
        "annualReturns": annual_returns_for_export.reset_index().replace({np.nan: None}).to_dict(orient="records"),
        "annualValues": annual_values.reset_index().replace({np.nan: None}).to_dict(orient="records"),
        "finalAllocations": final_allocations.replace({np.nan: None}).to_dict(orient="records"),
        "weightSchedule": weight_schedule.replace({np.nan: None}).to_dict(orient="records"),
        "sourceReturns": source.reset_index().replace({np.nan: None}).to_dict(orient="records"),
        "methodology": methodology.to_dict(orient="records"),
        "coverage": coverage.to_dict(orient="records"),
        "checks": checks.to_dict(orient="records"),
        "initialCapital": INITIAL_CAPITAL,
        "riskFreeRate": RISK_FREE_RATE,
    }
    (OUTPUT_DIR / "workbook_data.json").write_text(json.dumps(workbook_data, indent=2, default=str), encoding="utf-8")

    print(summary[["Strategy", "Overall Rank", "CAGR", "Sharpe Ratio", "Annualized Volatility", "Maximum Drawdown"]].to_string(index=False))
    print(f"\nOutputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
