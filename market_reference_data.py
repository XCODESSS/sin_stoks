"""Cached Yahoo historical closes and point-in-time USD FX lookups."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

FX_SYMBOLS = {
    "CAD": "CADUSD=X",
    "EUR": "EURUSD=X",
    "GBP": "GBPUSD=X",
}
_CACHE_MANIFEST = "cache_coverage.json"


@dataclass(frozen=True)
class PriceObservation:
    """A scaled unadjusted close and its observation date."""

    symbol: str
    date: pd.Timestamp
    value: float


class MarketReferenceData:
    """Validated immutable views over primary-listing and FX close histories."""

    def __init__(self, histories: dict[str, pd.Series]) -> None:
        if not histories:
            raise ValueError("At least one market history is required")
        self._histories = {
            symbol: self._validated_history(symbol, history) for symbol, history in histories.items()
        }

    @staticmethod
    def _validated_history(symbol: str, history: pd.Series) -> pd.Series:
        copied = history.copy()
        copied.index = pd.to_datetime(copied.index).tz_localize(None).normalize()
        if copied.index.has_duplicates:
            raise ValueError(f"{symbol} history contains duplicate dates")
        copied = copied.sort_index().astype(float)
        values = copied.to_numpy()
        if copied.empty or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"{symbol} history must contain finite positive closes")
        return copied

    def close_before(self, symbol: str, cutoff: pd.Timestamp, scale: float = 1.0) -> PriceObservation:
        """Return the final scaled close strictly before cutoff."""
        if scale <= 0:
            raise ValueError("Price scale must be positive")
        history = self._history(symbol)
        eligible = history.loc[history.index < pd.Timestamp(cutoff)]
        if eligible.empty:
            raise ValueError(f"No {symbol} close strictly before {pd.Timestamp(cutoff).date()}")
        date = pd.Timestamp(eligible.index[-1])
        return PriceObservation(symbol=symbol, date=date, value=float(eligible.iloc[-1] * scale))

    def spot_usd_rate(self, currency: str, date: pd.Timestamp) -> float:
        """Return the final currency/USD observation on or before date."""
        currency = currency.upper()
        if currency == "USD":
            return 1.0
        symbol = self._fx_symbol(currency)
        history = self._history(symbol)
        eligible = history.loc[history.index <= pd.Timestamp(date)]
        if eligible.empty:
            raise ValueError(f"No {currency}/USD rate on or before {pd.Timestamp(date).date()}")
        return float(eligible.iloc[-1])

    def average_usd_rate(
        self,
        currency: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> float:
        """Return arithmetic mean currency/USD closes over an inclusive period."""
        currency = currency.upper()
        if currency == "USD":
            return 1.0
        if pd.Timestamp(start) > pd.Timestamp(end):
            raise ValueError("FX period start must not be after end")
        symbol = self._fx_symbol(currency)
        history = self._history(symbol)
        eligible = history.loc[(history.index >= pd.Timestamp(start)) & (history.index <= pd.Timestamp(end))]
        if eligible.empty:
            raise ValueError(
                f"No {currency}/USD rates from {pd.Timestamp(start).date()} to {pd.Timestamp(end).date()}"
            )
        return float(eligible.mean())

    def _history(self, symbol: str) -> pd.Series:
        try:
            return self._histories[symbol]
        except KeyError as error:
            raise ValueError(f"Missing market history for {symbol}") from error

    @staticmethod
    def _fx_symbol(currency: str) -> str:
        try:
            return FX_SYMBOLS[currency]
        except KeyError as error:
            raise ValueError(f"Unsupported FX currency: {currency}") from error

    @classmethod
    def from_yfinance(
        cls,
        symbols: tuple[str, ...],
        start: pd.Timestamp,
        end: pd.Timestamp,
        cache_dir: Path,
        refresh: bool = False,
        allow_missing: bool = False,
    ) -> MarketReferenceData:
        """Load complete caches or download unadjusted closes in one Yahoo request."""
        start = pd.Timestamp(start).normalize()
        end = pd.Timestamp(end).normalize()
        if start >= end:
            raise ValueError("Yahoo history start must be before end")
        requested_symbols = tuple(dict.fromkeys((*symbols, *FX_SYMBOLS.values())))
        cache_paths = {symbol: cache_dir / f"{_safe_filename(symbol)}.csv" for symbol in requested_symbols}

        if not refresh and _coverage_path(cache_dir).exists():
            available_symbols = _validate_cache_coverage(
                cache_dir,
                requested_symbols,
                start,
                end,
            )
            missing_symbols = sorted(set(requested_symbols).difference(available_symbols))
            if missing_symbols and not allow_missing:
                raise ValueError(f"Yahoo cache has no Close history for: {missing_symbols}")
            return cls(
                {
                    symbol: _read_cached_history(cache_paths[symbol])
                    for symbol in requested_symbols
                    if symbol in available_symbols
                }
            )

        downloaded = yf.download(
            list(requested_symbols),
            start=start,
            end=end,
            auto_adjust=False,
            actions=False,
            progress=False,
        )
        histories = _extract_close_histories(downloaded, requested_symbols)
        missing_symbols = sorted(set(requested_symbols).difference(histories))
        if missing_symbols and not allow_missing:
            raise ValueError(f"Yahoo returned no Close history for: {missing_symbols}")
        for symbol, history in histories.items():
            _write_history_atomically(history, cache_paths[symbol])
        _write_cache_coverage(cache_dir, requested_symbols, tuple(histories), start, end)
        return cls(histories)


def _safe_filename(symbol: str) -> str:
    substituted = symbol.replace("=", "_eq_").replace("/", "_slash_")
    return re.sub(r"[^A-Za-z0-9.\-]", "_", substituted)


def _read_cached_history(path: Path) -> pd.Series:
    cached = pd.read_csv(path, parse_dates=["date"])
    if list(cached.columns) != ["date", "close"]:
        raise ValueError(f"Invalid Yahoo cache schema: {path}")
    return pd.Series(cached["close"].to_numpy(), index=cached["date"], name=path.stem, dtype=float)


def _write_history_atomically(history: pd.Series, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".csv.tmp")
    frame = pd.DataFrame({"date": pd.to_datetime(history.index), "close": history.to_numpy()})
    try:
        frame.to_csv(temporary, index=False, date_format="%Y-%m-%d")
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _extract_close_histories(
    downloaded: pd.DataFrame,
    requested_symbols: tuple[str, ...],
) -> dict[str, pd.Series]:
    if downloaded.empty:
        return {}
    if isinstance(downloaded.columns, pd.MultiIndex):
        if "Close" in downloaded.columns.get_level_values(0):
            close = downloaded.xs("Close", axis=1, level=0)
        elif "Close" in downloaded.columns.get_level_values(-1):
            close = downloaded.xs("Close", axis=1, level=-1)
        else:
            return {}
    elif "Close" in downloaded.columns and len(requested_symbols) == 1:
        close = downloaded[["Close"]].rename(columns={"Close": requested_symbols[0]})
    else:
        return {}

    histories: dict[str, pd.Series] = {}
    for symbol in requested_symbols:
        if symbol not in close.columns:
            continue
        history = close[symbol].dropna().astype(float)
        if not history.empty:
            histories[symbol] = history
    return histories


def _coverage_path(cache_dir: Path) -> Path:
    return cache_dir / _CACHE_MANIFEST


def _write_cache_coverage(
    cache_dir: Path,
    symbols: tuple[str, ...],
    available_symbols: tuple[str, ...],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = _coverage_path(cache_dir)
    temporary = target.with_suffix(".json.tmp")
    payload = {
        "symbols": list(symbols),
        "available_symbols": list(available_symbols),
        "requested_start": start.date().isoformat(),
        "requested_end": end.date().isoformat(),
    }
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_cache_coverage(
    cache_dir: Path,
    symbols: tuple[str, ...],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> set[str]:
    path = _coverage_path(cache_dir)
    if not path.exists():
        raise ValueError("Yahoo cache coverage metadata is missing; rerun with refresh=True")
    payload = json.loads(path.read_text(encoding="utf-8"))
    cached_symbols = set(payload.get("symbols", []))
    cached_start = pd.Timestamp(payload.get("requested_start"))
    cached_end = pd.Timestamp(payload.get("requested_end"))
    if not set(symbols).issubset(cached_symbols) or cached_start > start or cached_end < end:
        raise ValueError(
            "Yahoo cache does not cover the requested symbols/date range; rerun with refresh=True"
        )
    available_symbols = set(payload.get("available_symbols", payload.get("symbols", [])))
    missing_cache_files = [
        symbol for symbol in available_symbols if not (cache_dir / f"{_safe_filename(symbol)}.csv").exists()
    ]
    if missing_cache_files:
        raise ValueError(f"Yahoo cache files are missing for: {sorted(missing_cache_files)}")
    return available_symbols
