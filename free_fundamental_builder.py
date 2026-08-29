"""Assemble auditable SEC/Yahoo point-in-time fundamental observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from fundamental_sources import MANUAL_ONLY_TICKERS, SEC_ISSUERS, SecIssuerConfig
from market_reference_data import MarketReferenceData
from sec_companyfacts import (
    EarningsObservation,
    SecCompanyFactsClient,
    SharesObservation,
    select_filed_shares,
    select_ttm_earnings,
)

_EXPECTED_UNIVERSE_SIZE = 30
_MINIMUM_ELIGIBLE_ASSETS = 24
_MAX_SHARE_AGE_DAYS = 200


class MarketDataProtocol(Protocol):
    def close_before(self, symbol: str, cutoff: pd.Timestamp, scale: float = 1.0): ...

    def spot_usd_rate(self, currency: str, date: pd.Timestamp) -> float: ...

    def average_usd_rate(self, currency: str, start: pd.Timestamp, end: pd.Timestamp) -> float: ...


@dataclass(frozen=True)
class FreeFundamentalBuild:
    """Candidate fundamentals plus complete source coverage/error audits."""

    fundamentals: pd.DataFrame
    coverage: pd.DataFrame
    errors: pd.DataFrame


def build_fundamental_record(
    issuer: SecIssuerConfig,
    rebalance_date: pd.Timestamp,
    earnings: EarningsObservation,
    shares: SharesObservation,
    market_data: MarketDataProtocol,
) -> dict[str, object]:
    """Combine one issuer's point-in-time SEC and Yahoo observations."""
    rebalance_date = pd.Timestamp(rebalance_date)
    price = market_data.close_before(issuer.price_symbol, rebalance_date, issuer.price_scale)
    spot_fx = market_data.spot_usd_rate(issuer.price_currency, price.date)
    earnings_fx = market_data.average_usd_rate(
        issuer.earnings_unit,
        earnings.start,
        earnings.end,
    )
    share_age_days = (price.date - shares.observation_date).days
    if share_age_days < 0 or share_age_days > _MAX_SHARE_AGE_DAYS:
        raise ValueError(
            f"Filed shares for {issuer.ticker} are {share_age_days} days from the price date; "
            f"allowed range is 0-{_MAX_SHARE_AGE_DAYS} days"
        )

    market_cap_usd = price.value * shares.shares * spot_fx
    trailing_earnings_usd = earnings.value * earnings_fx
    earnings_positive = trailing_earnings_usd > 0
    trailing_pe = market_cap_usd / trailing_earnings_usd if earnings_positive else np.nan
    available_date = max(price.date, earnings.available_date, shares.available_date)

    component_dates = (
        price.date,
        earnings.available_date,
        shares.available_date,
        earnings.end,
        shares.observation_date,
    )
    if any(pd.Timestamp(date) >= rebalance_date for date in component_dates):
        raise ValueError(f"Every source date for {issuer.ticker} must be strictly before rebalance")
    if not np.isfinite(market_cap_usd) or market_cap_usd <= 0:
        raise ValueError(f"market_cap for {issuer.ticker} must be finite and positive")
    if not np.isfinite(trailing_earnings_usd):
        raise ValueError(f"trailing earnings for {issuer.ticker} must be finite")
    if earnings_positive and (not np.isfinite(trailing_pe) or trailing_pe <= 0):
        raise ValueError(f"trailing P/E for {issuer.ticker} must be finite and positive")

    return {
        "ticker": issuer.ticker,
        "rebalance_date": rebalance_date,
        "observation_date": price.date,
        "available_date": available_date,
        "trailing_pe": trailing_pe,
        "market_cap": market_cap_usd,
        "earnings_positive": bool(earnings_positive),
        "source": "SEC EDGAR Company Facts + Yahoo Finance unadjusted primary-listing close",
        "cik": issuer.cik,
        "price_symbol": issuer.price_symbol,
        "price_date": price.date,
        "price_currency": issuer.price_currency,
        "spot_fx_to_usd": spot_fx,
        "earnings_start": earnings.start,
        "earnings_end": earnings.end,
        "earnings_available_date": earnings.available_date,
        "earnings_tag": earnings.tag,
        "earnings_accessions": "|".join(earnings.accessions),
        "earnings_method": earnings.method,
        "shares_date": shares.observation_date,
        "shares_available_date": shares.available_date,
        "shares_tag": shares.tag,
        "shares_accession": shares.accession,
        "shares_aggregation": shares.aggregation,
        "shares_component_count": shares.component_count,
        "filed_shares": shares.shares,
        "trailing_earnings_usd": trailing_earnings_usd,
    }


def _build_one_record(
    sec_client: SecCompanyFactsClient,
    market_data: MarketReferenceData,
    issuer: SecIssuerConfig,
    rebalance_date: pd.Timestamp,
) -> dict[str, object]:
    companyfacts = sec_client.get_companyfacts(issuer.cik)
    earnings = select_ttm_earnings(companyfacts, issuer, rebalance_date)
    shares = select_filed_shares(companyfacts, issuer, rebalance_date)
    return build_fundamental_record(issuer, rebalance_date, earnings, shares, market_data)


def build_free_fundamentals(
    sec_client: SecCompanyFactsClient,
    market_data: MarketReferenceData,
    rebalance_years: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024, 2025),
) -> FreeFundamentalBuild:
    """Build all automated issuer/rebalance records without imputation."""
    fundamental_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []

    for year in rebalance_years:
        rebalance_date = pd.Timestamp(year=year, month=1, day=1)
        successful_tickers: set[str] = set()
        failed_tickers: set[str] = set()
        for ticker, issuer in sorted(SEC_ISSUERS.items()):
            try:
                record = _build_one_record(sec_client, market_data, issuer, rebalance_date)
            except (KeyError, TypeError, ValueError) as error:
                failed_tickers.add(ticker)
                error_rows.append(
                    {
                        "rebalance_date": rebalance_date,
                        "ticker": ticker,
                        "stage": error.__class__.__name__,
                        "error": str(error),
                    }
                )
                continue
            fundamental_rows.append(record)
            successful_tickers.add(ticker)

        eligible_assets = len(successful_tickers)
        missing = MANUAL_ONLY_TICKERS | failed_tickers
        coverage_rows.append(
            {
                "rebalance_date": rebalance_date,
                "expected_assets": _EXPECTED_UNIVERSE_SIZE,
                "automated_candidates": len(SEC_ISSUERS),
                "eligible_assets": eligible_assets,
                "coverage": eligible_assets / _EXPECTED_UNIVERSE_SIZE,
                "minimum_required_assets": _MINIMUM_ELIGIBLE_ASSETS,
                "coverage_passed": eligible_assets >= _MINIMUM_ELIGIBLE_ASSETS,
                "missing_tickers": ", ".join(sorted(missing)),
                "failed_tickers": ", ".join(sorted(failed_tickers)),
            }
        )

    fundamentals = pd.DataFrame(fundamental_rows)
    if not fundamentals.empty:
        fundamentals = fundamentals.sort_values(["ticker", "available_date"]).reset_index(drop=True)
    coverage = pd.DataFrame(coverage_rows)
    errors = pd.DataFrame(
        error_rows,
        columns=["rebalance_date", "ticker", "stage", "error"],
    )
    return FreeFundamentalBuild(fundamentals=fundamentals, coverage=coverage, errors=errors)
