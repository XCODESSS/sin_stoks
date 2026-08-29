from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from market_reference_data import MarketReferenceData


def series(values: dict[str, float]) -> pd.Series:
    return pd.Series(list(values.values()), index=pd.to_datetime(list(values)), dtype=float)


def downloaded_frame(symbols: list[str]) -> pd.DataFrame:
    dates = pd.to_datetime(["2019-12-27", "2019-12-30"])
    columns = pd.MultiIndex.from_product([["Close"], symbols])
    values = np.tile(np.array([[10.0], [11.0]]), (1, len(symbols)))
    return pd.DataFrame(values, index=dates, columns=columns)


def test_close_before_uses_last_observation_strictly_before_cutoff():
    market = MarketReferenceData({"DGE.L": series({"2019-12-30": 3100.0, "2020-01-01": 3200.0})})
    price = market.close_before("DGE.L", pd.Timestamp("2020-01-01"), scale=0.01)
    assert price.date == pd.Timestamp("2019-12-30")
    assert price.value == 31.0


def test_currency_conversion_uses_spot_and_period_average():
    market = MarketReferenceData({"GBPUSD=X": series({"2019-06-30": 1.20, "2019-12-30": 1.30})})
    assert market.spot_usd_rate("GBP", pd.Timestamp("2019-12-31")) == 1.30
    assert market.average_usd_rate("GBP", pd.Timestamp("2019-06-01"), pd.Timestamp("2019-12-31")) == 1.25
    assert market.spot_usd_rate("USD", pd.Timestamp("2019-12-31")) == 1.0


def test_market_reference_rejects_duplicates_and_nonpositive_values():
    duplicate = pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-01", "2020-01-01"]))
    with pytest.raises(ValueError, match="duplicate"):
        MarketReferenceData({"X": duplicate})
    with pytest.raises(ValueError, match="finite positive"):
        MarketReferenceData({"X": series({"2020-01-01": 0.0})})


def test_from_yfinance_writes_and_reuses_complete_caches(tmp_path, monkeypatch):
    calls = []

    def fake_download(symbols, **kwargs):
        calls.append((tuple(symbols), kwargs))
        return downloaded_frame(list(symbols))

    monkeypatch.setattr("market_reference_data.yf.download", fake_download)
    symbols = ("TEST", "QSR.TO")
    start = pd.Timestamp("2017-01-01")
    end = pd.Timestamp("2026-01-02")
    MarketReferenceData.from_yfinance(symbols, start, end, tmp_path, refresh=True)
    MarketReferenceData.from_yfinance(symbols, start, end, tmp_path)

    assert len(calls) == 1
    assert (tmp_path / "TEST.csv").exists()
    assert (tmp_path / "QSR.TO.csv").exists()
    assert len(list(tmp_path.glob("*.csv"))) == 5


def test_from_yfinance_rejects_missing_symbol(tmp_path, monkeypatch):
    def missing_qsr(symbols, **kwargs):
        return downloaded_frame([symbol for symbol in symbols if symbol != "QSR.TO"])

    monkeypatch.setattr("market_reference_data.yf.download", missing_qsr)
    with pytest.raises(ValueError, match="QSR.TO"):
        MarketReferenceData.from_yfinance(
            ("TEST", "QSR.TO"),
            pd.Timestamp("2017-01-01"),
            pd.Timestamp("2026-01-02"),
            tmp_path,
            refresh=True,
        )


def test_from_yfinance_rejects_partial_cache_range(tmp_path, monkeypatch):
    symbols = ("TEST", "CADUSD=X", "EURUSD=X", "GBPUSD=X")
    for symbol in symbols:
        safe = symbol.replace("=", "_eq_")
        pd.DataFrame({"date": ["2019-12-30"], "close": [10.0]}).to_csv(tmp_path / f"{safe}.csv", index=False)
    (tmp_path / "cache_coverage.json").write_text(
        json.dumps(
            {
                "symbols": list(symbols),
                "requested_start": "2019-01-01",
                "requested_end": "2020-01-01",
            }
        )
    )

    monkeypatch.setattr(
        "market_reference_data.yf.download",
        lambda *args, **kwargs: pytest.fail("partial cache should fail before network"),
    )
    with pytest.raises(ValueError, match="does not cover"):
        MarketReferenceData.from_yfinance(
            ("TEST",),
            pd.Timestamp("2017-01-01"),
            pd.Timestamp("2026-01-02"),
            tmp_path,
        )
