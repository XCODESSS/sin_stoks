from __future__ import annotations

import pytest

from fundamental_sources import (
    AUTOMATED_SEC_TICKERS,
    MANUAL_ONLY_TICKERS,
    SEC_ISSUERS,
    SecIssuerConfig,
)
from universe import PORTFOLIO_TICKERS


def test_sec_registry_partitions_the_frozen_universe():
    assert frozenset(SEC_ISSUERS) == AUTOMATED_SEC_TICKERS
    assert frozenset({"PRNDY", "TCEHY", "NTDOY", "CCOEY", "UBSFY"}) == MANUAL_ONLY_TICKERS
    assert frozenset(PORTFOLIO_TICKERS) == AUTOMATED_SEC_TICKERS | MANUAL_ONLY_TICKERS
    assert AUTOMATED_SEC_TICKERS.isdisjoint(MANUAL_ONLY_TICKERS)
    assert len(AUTOMATED_SEC_TICKERS) == 25


def test_foreign_sec_issuers_use_primary_listings_and_frozen_units():
    assert SEC_ISSUERS["DEO"].price_symbol == "DGE.L"
    assert SEC_ISSUERS["DEO"].price_currency == "GBP"
    assert SEC_ISSUERS["DEO"].price_scale == 0.01
    assert SEC_ISSUERS["BUD"].price_symbol == "ABI.BR"
    assert SEC_ISSUERS["BUD"].earnings_unit == "USD"
    assert SEC_ISSUERS["BTI"].price_symbol == "BATS.L"
    assert SEC_ISSUERS["QSR"].price_symbol == "QSR.TO"


def test_multiclass_registry_is_explicit():
    assert SEC_ISSUERS["BF-B"].expected_share_classes == 2
    assert SEC_ISSUERS["META"].expected_share_classes == 2
    assert SEC_ISSUERS["GOOGL"].expected_share_classes == 3
    assert SEC_ISSUERS["SNAP"].expected_share_classes == 3
    assert {issuer.share_aggregation for issuer in SEC_ISSUERS.values()} == {"reconcile"}


def test_issuer_config_rejects_unsafe_share_aggregation():
    with pytest.raises(ValueError, match="reconcile"):
        SecIssuerConfig("X", 1, "X", "USD", 1.0, "us-gaap", ("NetIncomeLoss",), "USD", "sum")
