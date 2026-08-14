"""Configuration parameters and paths for sin_stoks."""

from __future__ import annotations

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
PORTFOLIO_OUTPUT_DIR = OUTPUT_DIR / "portfolio_backtest"
REPORT_DIR = OUTPUT_DIR / "report"

# Data download & date boundaries
START_DATE = "2016-01-01"
END_DATE = "2026-01-01"
COVARIANCE_START = "2017-04-01"
YEARS = list(range(2016, 2026))

# Walk-forward parameters
REBALANCE_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
WEEKS_PER_YEAR = 52
MONTHS_PER_YEAR = 12
MIN_ESTIMATION_WEEKS = 104
STARTING_VALUE = 10_000.0  # USD initial notional

# Financial and optimization constants
RISK_FREE_RATE = 0.04  # 4% annual risk-free rate
DEFAULT_MAX_WEIGHT = 0.25  # 25% single-asset position cap
DEFAULT_TRANSACTION_COST_BPS = 10.0  # 10 bps default (0.0010 per 1.00 turnover)

# Data cleaning policies
# Weekly returns with |log return| > 0.50 are filtered as corporate merger / data artifacts (e.g. KDP July 2018)
KDP_OUTLIER_THRESHOLD = 0.50
