from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from free_fundamental_builder import build_free_fundamentals, build_fundamental_record
from fundamental_sources import SecIssuerConfig
from market_reference_data import PriceObservation
from sec_companyfacts import EarningsObservation, SharesObservation


class FakeMarket:
    def __init__(self, price_date: str = "2019-12-30") -> None:
        self.price = PriceObservation("TEST.L", pd.Timestamp(price_date), 2000.0)

    def close_before(self, symbol, cutoff, scale=1.0):
        assert self.price.date < cutoff
        return PriceObservation(symbol, self.price.date, self.price.value * scale)

    def spot_usd_rate(self, currency, date):
        return 1.30 if currency == "GBP" else 1.0

    def average_usd_rate(self, currency, start, end):
        return 1.25 if currency == "GBP" else 1.0


class FakeSecClient:
    def get_companyfacts(self, cik):
        return {"cik": cik}


def make_issuer() -> SecIssuerConfig:
    return SecIssuerConfig("TEST", 1, "TEST.L", "GBP", 0.01, "ifrs-full", ("ProfitLoss",), "GBP")


def make_inputs():
    earnings = EarningsObservation(
        pd.Timestamp("2018-10-01"),
        pd.Timestamp("2019-09-30"),
        pd.Timestamp("2019-11-01"),
        100.0,
        "GBP",
        "ProfitLoss",
        "annual_plus_ytd_less_prior_ytd",
        ("annual", "interim"),
    )
    shares = SharesObservation(
        pd.Timestamp("2019-10-31"),
        pd.Timestamp("2019-11-05"),
        10.0,
        "EntityCommonStockSharesOutstanding",
        "shares",
        "single_distinct_value",
        1,
    )
    return earnings, shares


def test_builder_calculates_market_cap_pe_and_provenance():
    earnings, shares = make_inputs()
    record = build_fundamental_record(
        make_issuer(), pd.Timestamp("2020-01-01"), earnings, shares, FakeMarket()
    )

    assert record["market_cap"] == 260.0
    assert record["trailing_earnings_usd"] == 125.0
    assert record["trailing_pe"] == 260.0 / 125.0
    assert record["available_date"] == pd.Timestamp("2019-12-30")
    assert record["cik"] == 1
    assert record["earnings_accessions"] == "annual|interim"
    assert record["price_source"] == "Yahoo Finance unadjusted close"
    assert record["shares_source"] == "SEC EDGAR Company Facts"


def test_builder_keeps_negative_earnings_without_pe():
    earnings, shares = make_inputs()
    earnings = dataclasses.replace(earnings, value=-100.0)
    record = build_fundamental_record(
        make_issuer(), pd.Timestamp("2020-01-01"), earnings, shares, FakeMarket()
    )
    assert record["earnings_positive"] is False
    assert np.isnan(record["trailing_pe"])


def test_builder_honors_frozen_issuer_specific_share_age():
    earnings, shares = make_inputs()
    shares = dataclasses.replace(shares, observation_date=pd.Timestamp("2019-01-01"))
    issuer = dataclasses.replace(make_issuer(), max_share_age_days=400)

    record = build_fundamental_record(
        issuer,
        pd.Timestamp("2020-01-01"),
        earnings,
        shares,
        FakeMarket(),
    )

    assert record["market_cap"] > 0


def test_builder_rejects_future_or_stale_inputs():
    earnings, shares = make_inputs()
    future = dataclasses.replace(earnings, available_date=pd.Timestamp("2020-02-01"))
    with pytest.raises(ValueError, match="strictly before rebalance"):
        build_fundamental_record(make_issuer(), pd.Timestamp("2020-01-01"), future, shares, FakeMarket())

    stale = dataclasses.replace(shares, observation_date=pd.Timestamp("2019-01-01"))
    with pytest.raises(ValueError, match="allowed range"):
        build_fundamental_record(make_issuer(), pd.Timestamp("2020-01-01"), earnings, stale, FakeMarket())


@pytest.mark.parametrize(("eligible", "passed"), [(24, True), (23, False)])
def test_coverage_uses_frozen_thirty_asset_denominator(monkeypatch, eligible, passed):
    issuers = {
        f"T{position:02d}": SecIssuerConfig(
            f"T{position:02d}",
            position + 1,
            f"T{position:02d}",
            "USD",
            1.0,
            "us-gaap",
            ("NetIncomeLoss",),
            "USD",
        )
        for position in range(25)
    }
    monkeypatch.setattr("free_fundamental_builder.SEC_ISSUERS", issuers)
    monkeypatch.setattr(
        "free_fundamental_builder.MANUAL_ONLY_TICKERS",
        frozenset({"M0", "M1", "M2", "M3", "M4"}),
    )

    def fake_build(sec_client, market_data, issuer, rebalance_date, simfin_data):
        if issuer.cik > eligible:
            raise ValueError("synthetic missing fact")
        return {
            "ticker": issuer.ticker,
            "available_date": pd.Timestamp("2019-12-30"),
        }

    monkeypatch.setattr("free_fundamental_builder._build_one_record", fake_build)
    build = build_free_fundamentals(FakeSecClient(), object(), rebalance_years=(2020,))
    row = build.coverage.iloc[0]

    assert row["coverage"] == eligible / 30.0
    assert bool(row["coverage_passed"]) is passed
    assert row["minimum_required_assets"] == 24
    assert len(build.fundamentals) == eligible
    assert len(build.errors) == 25 - eligible
