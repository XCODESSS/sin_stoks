from __future__ import annotations

import json

import pandas as pd
import pytest

from simfin_reference_data import SimfinReferenceData


def payload(ticker: str = "TEST") -> bytes:
    return json.dumps(
        [
            {
                "ticker": ticker,
                "columns": ["Date", "Common Shares Outstanding", "Last Closing Price"],
                "data": [
                    ["2019-12-30", 100.0, 20.0],
                    ["2020-01-01", 110.0, 22.0],
                ],
            }
        ]
    ).encode()


def test_simfin_reference_uses_strict_dates_and_cache(tmp_path):
    calls = []

    def download(url, api_key):
        calls.append((url, api_key))
        return payload()

    first = SimfinReferenceData("secret", tmp_path, downloader=download)
    price = first.close_before("TEST", pd.Timestamp("2020-01-01"))
    shares = first.shares_before("TEST", pd.Timestamp("2020-01-01"))
    second = SimfinReferenceData(
        "secret",
        tmp_path,
        downloader=lambda *_: pytest.fail("cache should prevent a request"),
    )

    assert price.date == pd.Timestamp("2019-12-30")
    assert price.value == 20.0
    assert price.source == "SimFin unadjusted close"
    assert shares.shares == 100.0
    assert shares.source == "SimFin historical shares"
    assert second.close_before("TEST", pd.Timestamp("2020-01-01")).value == 20.0
    assert len(calls) == 1


def test_simfin_reference_rejects_missing_key_before_network(tmp_path):
    with pytest.raises(ValueError, match="SIMFIN_API_KEY"):
        SimfinReferenceData("", tmp_path)


def test_simfin_reference_rejects_missing_ticker(tmp_path):
    reference = SimfinReferenceData(
        "secret",
        tmp_path,
        downloader=lambda *_: payload("OTHER"),
    )
    with pytest.raises(ValueError, match="exactly one"):
        reference.close_before("TEST", pd.Timestamp("2020-01-01"))
