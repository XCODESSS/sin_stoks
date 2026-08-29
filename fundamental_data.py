"""Point-in-time fundamental data loading and snapshot validation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from config import SELECTION_MIN_COVERAGE, SELECTION_TARGET_COUNT

REQUIRED_COLUMNS = {
    "ticker",
    "observation_date",
    "available_date",
    "trailing_pe",
    "market_cap",
    "earnings_positive",
    "source",
}
PROVENANCE_DATE_COLUMNS = {
    "rebalance_date",
    "price_date",
    "earnings_start",
    "earnings_end",
    "earnings_available_date",
    "shares_date",
    "shares_available_date",
}
PROVENANCE_POSITIVE_COLUMNS = {
    "cik",
    "spot_fx_to_usd",
    "shares_component_count",
    "filed_shares",
}


def _parse_earnings_positive(values: pd.Series) -> pd.Series:
    normalized = values.map(lambda value: str(value).strip().lower())
    mapping = {"true": True, "false": False}
    invalid = ~normalized.isin(mapping)
    if invalid.any():
        invalid_values = sorted(normalized[invalid].unique())
        raise ValueError(f"Invalid earnings_positive values: {invalid_values}")
    return normalized.map(mapping).astype(bool)


def _validate_fundamentals(fundamentals: pd.DataFrame) -> None:
    duplicate_keys = fundamentals.duplicated(["ticker", "available_date"], keep=False)
    if duplicate_keys.any():
        raise ValueError("Duplicate (ticker, available_date) keys are not allowed")

    if (fundamentals["observation_date"] > fundamentals["available_date"]).any():
        raise ValueError("observation_date must not be after available_date")

    market_caps = fundamentals["market_cap"].to_numpy(dtype=float)
    if not np.isfinite(market_caps).all() or (market_caps <= 0).any():
        raise ValueError("market_cap must be finite and positive")

    if fundamentals["source"].isna().any() or fundamentals["source"].eq("").any():
        raise ValueError("source must not be blank")

    profitable = fundamentals["earnings_positive"]
    profitable_pe = fundamentals.loc[profitable, "trailing_pe"].to_numpy(dtype=float)
    if not np.isfinite(profitable_pe).all() or (profitable_pe <= 0).any():
        raise ValueError("Profitable observations require a finite positive trailing_pe")

    non_missing_pe = fundamentals["trailing_pe"].dropna().to_numpy(dtype=float)
    if not np.isfinite(non_missing_pe).all():
        raise ValueError("trailing_pe must be finite when present")

    if (
        "rebalance_date" in fundamentals
        and (fundamentals["available_date"] >= fundamentals["rebalance_date"]).any()
    ):
        raise ValueError("available_date must be strictly before rebalance_date")

    for column in ("price_date", "earnings_available_date", "shares_available_date"):
        if column in fundamentals and (fundamentals[column] > fundamentals["available_date"]).any():
            raise ValueError(f"{column} must not be after available_date")

    for column in PROVENANCE_POSITIVE_COLUMNS.intersection(fundamentals.columns):
        values = fundamentals[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"{column} must be finite and positive")

    if "trailing_earnings_usd" in fundamentals:
        values = fundamentals["trailing_earnings_usd"].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("trailing_earnings_usd must be finite")


def load_fundamentals(path: Path) -> pd.DataFrame:
    """Load and validate point-in-time fundamental observations."""
    if not path.exists():
        raise FileNotFoundError(f"Missing point-in-time fundamental file: {path}")

    fundamentals = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS.difference(fundamentals.columns)
    if missing_columns:
        raise ValueError(f"Missing required fundamental columns: {sorted(missing_columns)}")

    fundamentals = fundamentals.copy()
    fundamentals["ticker"] = fundamentals["ticker"].astype("string").str.strip().str.upper()
    if fundamentals["ticker"].isna().any() or fundamentals["ticker"].eq("").any():
        raise ValueError("ticker must not be blank")

    fundamentals["source"] = fundamentals["source"].astype("string").str.strip()
    fundamentals["observation_date"] = pd.to_datetime(fundamentals["observation_date"], errors="raise")
    fundamentals["available_date"] = pd.to_datetime(fundamentals["available_date"], errors="raise")
    for column in PROVENANCE_DATE_COLUMNS.intersection(fundamentals.columns):
        fundamentals[column] = pd.to_datetime(fundamentals[column], errors="raise")
    fundamentals["market_cap"] = pd.to_numeric(fundamentals["market_cap"], errors="raise")
    fundamentals["trailing_pe"] = pd.to_numeric(fundamentals["trailing_pe"], errors="raise")
    for column in PROVENANCE_POSITIVE_COLUMNS.intersection(fundamentals.columns):
        fundamentals[column] = pd.to_numeric(fundamentals[column], errors="raise")
    if "trailing_earnings_usd" in fundamentals:
        fundamentals["trailing_earnings_usd"] = pd.to_numeric(
            fundamentals["trailing_earnings_usd"], errors="raise"
        )
    fundamentals["earnings_positive"] = _parse_earnings_positive(fundamentals["earnings_positive"])

    _validate_fundamentals(fundamentals)
    return fundamentals.sort_values(["ticker", "available_date"]).reset_index(drop=True)


def fundamentals_as_of(
    data: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    tickers: Sequence[str],
    min_coverage: float = SELECTION_MIN_COVERAGE,
    min_assets: int = SELECTION_TARGET_COUNT,
) -> pd.DataFrame:
    """Return each ticker's latest eligible record known before rebalance."""
    requested_tickers = sorted({str(ticker).strip().upper() for ticker in tickers})
    if not requested_tickers:
        raise ValueError("At least one requested ticker is required")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage must be between 0 and 1")
    if min_assets < 1:
        raise ValueError("min_assets must be positive")

    rebalance_timestamp = pd.Timestamp(rebalance_date)
    eligible = data.loc[
        data["ticker"].isin(requested_tickers) & (data["available_date"] < rebalance_timestamp)
    ].copy()
    eligible = eligible.sort_values(["ticker", "available_date"])
    snapshot = eligible.groupby("ticker", sort=True, as_index=False).tail(1).set_index("ticker")
    snapshot = snapshot.sort_index()

    coverage = len(snapshot) / len(requested_tickers)
    if coverage < min_coverage:
        raise ValueError(
            f"Insufficient fundamental coverage at {rebalance_timestamp.date()}: "
            f"{coverage:.1%} is below {min_coverage:.1%}"
        )
    if len(snapshot) < min_assets:
        raise ValueError(
            f"Fundamental snapshot has {len(snapshot)} assets; at least {min_assets} are required"
        )
    return snapshot
