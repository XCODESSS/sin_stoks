"""Cached SimFin market-reference fallbacks for narrowly approved source gaps."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from market_reference_data import PriceObservation
from sec_companyfacts import SharesObservation

SIMFIN_PRICES_URL = "https://backend.simfin.com/api/v3/companies/prices/compact"
SIMFIN_START = "2019-01-01"
SIMFIN_END = "2025-01-02"
Downloader = Callable[[str, str], bytes]


def _download(url: str, api_key: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"api-key {api_key}", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


class SimfinReferenceData:
    """Load cached SimFin unadjusted closes and historical share observations."""

    def __init__(
        self,
        api_key: str,
        cache_dir: Path,
        downloader: Downloader = _download,
        refresh: bool = False,
    ) -> None:
        if not api_key.strip():
            raise ValueError("SIMFIN_API_KEY must be explicitly set")
        self._api_key = api_key
        self._cache_dir = cache_dir
        self._downloader = downloader
        self._refresh = refresh
        self._frames: dict[str, pd.DataFrame] = {}

    def close_before(self, ticker: str, cutoff: pd.Timestamp) -> PriceObservation:
        """Return the final SimFin raw close strictly before cutoff."""
        frame = self._frame(ticker)
        eligible = frame.loc[(frame.index < pd.Timestamp(cutoff)) & frame["Last Closing Price"].notna()]
        if eligible.empty:
            raise ValueError(f"No SimFin close for {ticker} before {pd.Timestamp(cutoff).date()}")
        date = pd.Timestamp(eligible.index[-1])
        return PriceObservation(
            symbol=ticker,
            date=date,
            value=float(eligible.iloc[-1]["Last Closing Price"]),
            source="SimFin unadjusted close",
        )

    def shares_before(self, ticker: str, cutoff: pd.Timestamp) -> SharesObservation:
        """Return the final SimFin historical common-share observation before cutoff."""
        frame = self._frame(ticker)
        eligible = frame.loc[
            (frame.index < pd.Timestamp(cutoff)) & frame["Common Shares Outstanding"].notna()
        ]
        if eligible.empty:
            raise ValueError(f"No SimFin common shares for {ticker} before {pd.Timestamp(cutoff).date()}")
        date = pd.Timestamp(eligible.index[-1])
        shares = float(eligible.iloc[-1]["Common Shares Outstanding"])
        if not np.isfinite(shares) or shares <= 0:
            raise ValueError(f"SimFin common shares for {ticker} must be finite and positive")
        return SharesObservation(
            observation_date=date,
            available_date=date,
            shares=shares,
            tag="SimFinCommonSharesOutstanding",
            accession="simfin-price-series",
            aggregation="simfin_historical_observation",
            component_count=1,
            source="SimFin historical shares",
        )

    def _frame(self, ticker: str) -> pd.DataFrame:
        normalized = ticker.strip().upper()
        if normalized not in self._frames:
            payload = self._load_payload(normalized)
            self._frames[normalized] = _normalize_prices_payload(payload, normalized)
        return self._frames[normalized]

    def _load_payload(self, ticker: str) -> list[dict[str, object]]:
        cache_path = self._cache_dir / f"{ticker}.json"
        if cache_path.exists() and not self._refresh:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        query = urllib.parse.urlencode({"ticker": ticker, "start": SIMFIN_START, "end": SIMFIN_END})
        payload = self._downloader(f"{SIMFIN_PRICES_URL}?{query}", self._api_key)
        parsed = json.loads(payload)
        if not isinstance(parsed, list) or not parsed:
            raise ValueError(f"SimFin returned no market history for {ticker}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".json.tmp")
        try:
            temporary.write_bytes(payload)
            temporary.replace(cache_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return parsed


def _normalize_prices_payload(payload: list[dict[str, object]], ticker: str) -> pd.DataFrame:
    matching = [record for record in payload if str(record.get("ticker", "")).upper() == ticker]
    if len(matching) != 1:
        raise ValueError(f"SimFin response must contain exactly one company for {ticker}")
    record = matching[0]
    columns = record.get("columns")
    rows = record.get("data")
    if not isinstance(columns, list) or not isinstance(rows, list):
        raise ValueError(f"Invalid SimFin market response for {ticker}")
    required = {"Date", "Common Shares Outstanding", "Last Closing Price"}
    if not required.issubset(columns):
        raise ValueError(f"SimFin market response is missing columns for {ticker}")
    frame = pd.DataFrame(rows, columns=columns)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    if frame["Date"].duplicated().any():
        raise ValueError(f"SimFin market response has duplicate dates for {ticker}")
    frame = frame.set_index("Date").sort_index()
    for column in ("Common Shares Outstanding", "Last Closing Price"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        values = frame[column].dropna().to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"SimFin {column} values must be finite and positive for {ticker}")
    return frame
