"""Polite SEC Company Facts access and point-in-time fact reconstruction."""

from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fundamental_sources import SecIssuerConfig

SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
ANNUAL_FORMS = frozenset({"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"})
INTERIM_FORMS = frozenset({"10-Q", "10-Q/A", "6-K", "6-K/A"})
SHARE_FORMS = ANNUAL_FORMS | INTERIM_FORMS | {"8-K", "8-K/A"}


@dataclass(frozen=True)
class FactRecord:
    """One normalized SEC Company Facts unit record."""

    start: pd.Timestamp | None
    end: pd.Timestamp
    filed: pd.Timestamp
    form: str
    accession: str
    fiscal_year: int | None
    fiscal_period: str | None
    frame: str | None
    value: float
    unit: str


@dataclass(frozen=True)
class EarningsObservation:
    """Point-in-time trailing earnings and its SEC provenance."""

    start: pd.Timestamp
    end: pd.Timestamp
    available_date: pd.Timestamp
    value: float
    unit: str
    tag: str
    method: str
    accessions: tuple[str, ...]


@dataclass(frozen=True)
class SharesObservation:
    """Point-in-time filed shares and reconciliation provenance."""

    observation_date: pd.Timestamp
    available_date: pd.Timestamp
    shares: float
    tag: str
    accession: str
    aggregation: str
    component_count: int
    source: str = "SEC EDGAR Company Facts"


Downloader = Callable[[str, str], bytes]


def _download(url: str, user_agent: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": user_agent, "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


class SecCompanyFactsClient:
    """Fair-access SEC client with one immutable JSON cache per CIK."""

    def __init__(
        self,
        user_agent: str,
        cache_dir: Path,
        downloader: Downloader = _download,
        minimum_interval_seconds: float = 0.12,
        refresh: bool = False,
    ) -> None:
        if "@" not in user_agent:
            raise ValueError("SEC_USER_AGENT must include a contact email")
        if minimum_interval_seconds < 0:
            raise ValueError("minimum_interval_seconds must be non-negative")
        self._user_agent = user_agent
        self._cache_dir = cache_dir
        self._downloader = downloader
        self._minimum_interval_seconds = minimum_interval_seconds
        self._refresh = refresh
        self._last_request_time: float | None = None

    def get_companyfacts(self, cik: int, refresh: bool = False) -> dict[str, object]:
        """Return one CIK payload, preferring a validated local cache."""
        if cik <= 0:
            raise ValueError("CIK must be positive")
        cache_path = self._cache_dir / f"CIK{cik:010d}.json"
        if cache_path.exists() and not (refresh or self._refresh):
            return self._read_cached_payload(cache_path, cik)

        self._wait_for_fair_access()
        payload = self._downloader(SEC_COMPANYFACTS_URL.format(cik=cik), self._user_agent)
        parsed = self._parse_payload(payload, cik)
        self._write_cache_atomically(cache_path, payload)
        self._last_request_time = time.monotonic()
        return parsed

    def _wait_for_fair_access(self) -> None:
        if self._last_request_time is None:
            return
        elapsed = time.monotonic() - self._last_request_time
        remaining = self._minimum_interval_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    @staticmethod
    def _parse_payload(payload: bytes, cik: int) -> dict[str, object]:
        parsed = json.loads(payload)
        if not isinstance(parsed, dict) or int(parsed.get("cik", -1)) != cik:
            raise ValueError(f"SEC response CIK mismatch for {cik}")
        return parsed

    def _read_cached_payload(self, cache_path: Path, cik: int) -> dict[str, object]:
        return self._parse_payload(cache_path.read_bytes(), cik)

    @staticmethod
    def _write_cache_atomically(cache_path: Path, payload: bytes) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".json.tmp")
        try:
            temporary.write_bytes(payload)
            temporary.replace(cache_path)
        finally:
            if temporary.exists():
                temporary.unlink()


def normalize_fact_records(
    companyfacts: dict[str, object],
    namespace: str,
    tag: str,
    unit: str,
) -> list[FactRecord]:
    """Normalize one SEC concept/unit without mutating the source payload."""
    try:
        raw_records = companyfacts["facts"][namespace][tag]["units"][unit]  # type: ignore[index]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Missing SEC fact {namespace}.{tag} in unit {unit}") from error
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError(f"Missing SEC fact {namespace}.{tag} in unit {unit}")

    records: list[FactRecord] = []
    for raw_record in raw_records:
        if not isinstance(raw_record, dict):
            raise ValueError(f"Invalid SEC fact record for {namespace}.{tag}")
        value = float(raw_record["val"])
        if not np.isfinite(value):
            raise ValueError(f"SEC fact {namespace}.{tag} values must be finite")
        start_value = raw_record.get("start")
        start = pd.to_datetime(start_value, errors="raise") if start_value else None
        records.append(
            FactRecord(
                start=start,
                end=pd.to_datetime(raw_record["end"], errors="raise"),
                filed=pd.to_datetime(raw_record["filed"], errors="raise"),
                form=str(raw_record.get("form", "")),
                accession=str(raw_record.get("accn", "")),
                fiscal_year=(int(raw_record["fy"]) if raw_record.get("fy") is not None else None),
                fiscal_period=(str(raw_record["fp"]) if raw_record.get("fp") else None),
                frame=(str(raw_record["frame"]) if raw_record.get("frame") else None),
                value=value,
                unit=unit,
            )
        )
    return sorted(records, key=lambda record: (record.end, record.filed, record.accession))


def _duration_days(record: FactRecord) -> int:
    if record.start is None:
        raise ValueError("Duration fact is missing a start date")
    return (record.end - record.start).days + 1


def _select_earnings_for_tag(
    companyfacts: dict[str, object],
    issuer: SecIssuerConfig,
    tag: str,
    cutoff: pd.Timestamp,
) -> EarningsObservation:
    records = normalize_fact_records(
        companyfacts,
        issuer.earnings_namespace,
        tag,
        issuer.earnings_unit,
    )
    eligible = [
        record
        for record in records
        if record.start is not None and record.filed < cutoff and record.end < cutoff
    ]
    annual = [
        record for record in eligible if record.form in ANNUAL_FORMS and 300 <= _duration_days(record) <= 430
    ]
    if not annual:
        raise ValueError("no eligible annual fact")
    annual_record = max(annual, key=lambda record: (record.end, record.filed, record.accession))

    interim = [
        record
        for record in eligible
        if record.form in INTERIM_FORMS
        and record.end > annual_record.end
        and 60 <= _duration_days(record) <= 300
    ]
    if not interim:
        return EarningsObservation(
            start=annual_record.start,
            end=annual_record.end,
            available_date=annual_record.filed,
            value=annual_record.value,
            unit=issuer.earnings_unit,
            tag=tag,
            method="annual_fallback",
            accessions=(annual_record.accession,),
        )

    current_ytd = max(
        interim,
        key=lambda record: (record.end, record.filed, _duration_days(record), record.accession),
    )
    comparisons = [
        record
        for record in eligible
        if record.accession == current_ytd.accession
        and record is not current_ytd
        and 330 <= (current_ytd.end - record.end).days <= 400
        and abs(_duration_days(current_ytd) - _duration_days(record)) <= 14
    ]
    if not comparisons:
        raise ValueError("interim fact lacks a same-filing prior-year comparison")
    prior_ytd = max(comparisons, key=lambda record: (record.end, _duration_days(record)))
    value = annual_record.value + current_ytd.value - prior_ytd.value
    return EarningsObservation(
        start=prior_ytd.end + pd.Timedelta(days=1),
        end=current_ytd.end,
        available_date=max(annual_record.filed, current_ytd.filed, prior_ytd.filed),
        value=value,
        unit=issuer.earnings_unit,
        tag=tag,
        method="annual_plus_ytd_less_prior_ytd",
        accessions=(annual_record.accession, current_ytd.accession),
    )


def select_ttm_earnings(
    companyfacts: dict[str, object],
    issuer: SecIssuerConfig,
    cutoff: pd.Timestamp,
) -> EarningsObservation:
    """Select leakage-safe TTM earnings using configured tags in priority order."""
    cutoff = pd.Timestamp(cutoff)
    attempted: list[str] = []
    for tag in issuer.earnings_tags:
        try:
            return _select_earnings_for_tag(companyfacts, issuer, tag, cutoff)
        except ValueError as error:
            attempted.append(f"{tag}: {error}")
    reasons = "; ".join(attempted)
    raise ValueError(f"No TTM earnings for {issuer.ticker} before {cutoff.date()}: {reasons}")


def _reconcile_share_values(values: list[float], expected_share_classes: int) -> tuple[float, str, int]:
    distinct = sorted(set(values))
    if not distinct or any(not np.isfinite(value) or value <= 0 for value in distinct):
        raise ValueError("Selected share facts must be finite and positive")
    if len(distinct) == 1:
        return distinct[0], "single_distinct_value", 1

    largest = distinct[-1]
    components = distinct[:-1]
    if np.isclose(largest, sum(components), rtol=0.005, atol=1.0):
        return largest, "reported_total_matches_components", len(components)
    if len(distinct) == expected_share_classes:
        return float(sum(distinct)), "sum_expected_classes", len(distinct)
    raise ValueError(
        f"ambiguous share facts: {len(distinct)} distinct values for "
        f"{expected_share_classes} expected classes"
    )


def _select_shares_for_tag(
    companyfacts: dict[str, object],
    issuer: SecIssuerConfig,
    namespace: str,
    tag: str,
    cutoff: pd.Timestamp,
) -> SharesObservation:
    records = normalize_fact_records(companyfacts, namespace, tag, "shares")
    eligible = [
        record
        for record in records
        if record.filed < cutoff and record.end < cutoff and record.form in SHARE_FORMS
    ]
    if not eligible:
        raise ValueError("no eligible filed shares")
    latest_key = max((record.end, record.filed, record.accession) for record in eligible)
    latest_group = [
        record for record in eligible if (record.end, record.filed, record.accession) == latest_key
    ]
    shares, aggregation, component_count = _reconcile_share_values(
        [record.value for record in latest_group],
        issuer.expected_share_classes,
    )
    selected = latest_group[0]
    return SharesObservation(
        observation_date=selected.end,
        available_date=selected.filed,
        shares=shares,
        tag=tag,
        accession=selected.accession,
        aggregation=aggregation,
        component_count=component_count,
    )


def _select_duration_shares_for_tag(
    companyfacts: dict[str, object],
    issuer: SecIssuerConfig,
    namespace: str,
    tag: str,
    cutoff: pd.Timestamp,
) -> SharesObservation:
    records = normalize_fact_records(companyfacts, namespace, tag, "shares")
    eligible = [
        record
        for record in records
        if record.start is not None
        and record.filed < cutoff
        and record.end < cutoff
        and record.form in ANNUAL_FORMS | INTERIM_FORMS
        and record.value > 0
    ]
    if not eligible:
        raise ValueError("no eligible filed duration shares")
    latest_end = max(record.end for record in eligible)
    latest_end_records = [record for record in eligible if record.end == latest_end]
    latest_filing_key = max((record.filed, record.accession) for record in latest_end_records)
    filing_records = [
        record for record in latest_end_records if (record.filed, record.accession) == latest_filing_key
    ]
    selected = max(filing_records, key=_duration_days)
    return SharesObservation(
        observation_date=selected.end,
        available_date=selected.filed,
        shares=selected.value,
        tag=tag,
        accession=selected.accession,
        aggregation="weighted_average_diluted_fallback",
        component_count=1,
    )


def select_filed_shares(
    companyfacts: dict[str, object],
    issuer: SecIssuerConfig,
    cutoff: pd.Timestamp,
) -> SharesObservation:
    """Select and reconcile the latest eligible SEC common-share facts."""
    cutoff = pd.Timestamp(cutoff)
    attempted: list[str] = []
    instant_tags = (
        ("dei", "EntityCommonStockSharesOutstanding"),
        ("us-gaap", "CommonStockSharesOutstanding"),
        *issuer.additional_instant_share_tags,
    )
    for namespace, tag in instant_tags:
        try:
            return _select_shares_for_tag(companyfacts, issuer, namespace, tag, cutoff)
        except ValueError as error:
            attempted.append(f"{namespace}.{tag}: {error}")
    for namespace, tag in issuer.duration_share_tags:
        try:
            return _select_duration_shares_for_tag(
                companyfacts,
                issuer,
                namespace,
                tag,
                cutoff,
            )
        except ValueError as error:
            attempted.append(f"{namespace}.{tag}: {error}")
    reasons = "; ".join(attempted)
    raise ValueError(f"No filed shares for {issuer.ticker} before {cutoff.date()}: {reasons}")
