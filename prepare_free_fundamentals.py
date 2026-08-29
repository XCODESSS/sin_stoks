"""Prepare SEC/Yahoo source artifacts without invoking a backtest."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import DATA_DIR, FUNDAMENTALS_PATH, REBALANCE_YEARS, SELECTION_OUTPUT_DIR
from data_pipeline import write_csv_outputs_atomically
from free_fundamental_builder import FreeFundamentalBuild, build_free_fundamentals
from fundamental_data import load_fundamentals
from fundamental_sources import SEC_ISSUERS
from market_reference_data import FX_SYMBOLS, MarketReferenceData
from sec_companyfacts import SEC_COMPANYFACTS_URL, SecCompanyFactsClient

SOURCE_CACHE_DIR = DATA_DIR / "source_cache"
MARKET_START = pd.Timestamp("2017-01-01")
MARKET_END = pd.Timestamp("2026-01-02")
METHODOLOGY_VERSION = "sec-yfinance-v1"
PROVENANCE_COLUMNS = [
    "ticker",
    "rebalance_date",
    "observation_date",
    "available_date",
    "trailing_pe",
    "market_cap",
    "earnings_positive",
    "source",
    "cik",
    "price_symbol",
    "price_date",
    "price_currency",
    "spot_fx_to_usd",
    "earnings_start",
    "earnings_end",
    "earnings_available_date",
    "earnings_tag",
    "earnings_accessions",
    "earnings_method",
    "shares_date",
    "shares_available_date",
    "shares_tag",
    "shares_accession",
    "shares_aggregation",
    "shares_component_count",
    "filed_shares",
    "trailing_earnings_usd",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_source_clients(
    cache_dir: Path,
    sec_user_agent: str,
    refresh: bool = False,
) -> tuple[SecCompanyFactsClient, MarketReferenceData]:
    """Create configured source clients, downloading only at this boundary."""
    sec_client = SecCompanyFactsClient(
        user_agent=sec_user_agent,
        cache_dir=cache_dir / "sec",
        refresh=refresh,
    )
    price_symbols = tuple(sorted({issuer.price_symbol for issuer in SEC_ISSUERS.values()}))
    market_data = MarketReferenceData.from_yfinance(
        price_symbols,
        MARKET_START,
        MARKET_END,
        cache_dir / "yahoo",
        refresh=refresh,
    )
    return sec_client, market_data


def _cache_manifest(cache_dir: Path) -> list[dict[str, object]]:
    return [
        {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(cache_dir.rglob("*"))
        if path.is_file()
    ]


def build_source_manifest(
    build: FreeFundamentalBuild,
    fundamentals_path: Path,
    cache_dir: Path,
    sec_user_agent: str,
) -> dict[str, object]:
    """Describe source choices and hash local inputs without exposing contact details."""
    user_agent_hash = hashlib.sha256(sec_user_agent.encode("utf-8")).hexdigest()
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "methodology": METHODOLOGY_VERSION,
        "rebalance_years": list(REBALANCE_YEARS),
        "sec": {
            "companyfacts_url": SEC_COMPANYFACTS_URL,
            "user_agent_sha256": user_agent_hash,
        },
        "yahoo": {
            "start": MARKET_START.date().isoformat(),
            "end": MARKET_END.date().isoformat(),
            "price_symbols": sorted({issuer.price_symbol for issuer in SEC_ISSUERS.values()}),
            "fx_symbols": sorted(FX_SYMBOLS.values()),
            "auto_adjust": False,
        },
        "issuer_configs": {
            ticker: dataclasses.asdict(issuer) for ticker, issuer in sorted(SEC_ISSUERS.items())
        },
        "rows": {
            "fundamentals": len(build.fundamentals),
            "coverage": len(build.coverage),
            "errors": len(build.errors),
        },
        "coverage": build.coverage[["rebalance_date", "eligible_assets", "coverage_passed"]]
        .assign(rebalance_date=lambda frame: frame["rebalance_date"].astype(str))
        .to_dict(orient="records"),
        "fundamentals": {
            "path": str(fundamentals_path.resolve()),
            "sha256": _sha256(fundamentals_path),
        },
        "cache_files": _cache_manifest(cache_dir),
    }


def write_json_atomically(payload: dict[str, object], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_validated_fundamentals(build: FreeFundamentalBuild, target: Path) -> None:
    if build.fundamentals.empty:
        raise ValueError("SEC/Yahoo source build produced no fundamental records")
    missing = set(PROVENANCE_COLUMNS).difference(build.fundamentals.columns)
    if missing:
        raise ValueError(f"Source build is missing provenance columns: {sorted(missing)}")
    candidate = build.fundamentals.loc[:, PROVENANCE_COLUMNS].sort_values(["ticker", "available_date"])
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    try:
        candidate.to_csv(temporary, index=False)
        load_fundamentals(temporary)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare_free_fundamentals(
    fundamentals_path: Path = FUNDAMENTALS_PATH,
    output_dir: Path = SELECTION_OUTPUT_DIR,
    cache_dir: Path = SOURCE_CACHE_DIR,
    sec_user_agent: str = "",
    refresh: bool = False,
) -> FreeFundamentalBuild:
    """Build, validate, and atomically persist source artifacts only."""
    if "@" not in sec_user_agent:
        raise ValueError("SEC_USER_AGENT must be explicitly set and include a contact email")
    sec_client, market_data = build_source_clients(
        cache_dir=cache_dir,
        sec_user_agent=sec_user_agent,
        refresh=refresh,
    )
    build = build_free_fundamentals(sec_client, market_data)
    _write_validated_fundamentals(build, fundamentals_path)
    write_csv_outputs_atomically(
        {
            output_dir / "source_coverage.csv": (build.coverage, {"index": False}),
            output_dir / "source_errors.csv": (build.errors, {"index": False}),
        }
    )
    manifest = build_source_manifest(build, fundamentals_path, cache_dir, sec_user_agent)
    write_json_atomically(manifest, output_dir / "source_manifest.json")
    return build


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SEC/Yahoo point-in-time fundamentals.")
    parser.add_argument("--fundamentals", type=Path, default=FUNDAMENTALS_PATH)
    parser.add_argument("--output-dir", type=Path, default=SELECTION_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=SOURCE_CACHE_DIR)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    sec_user_agent = os.environ.get("SEC_USER_AGENT", "")
    build = prepare_free_fundamentals(
        fundamentals_path=args.fundamentals,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        sec_user_agent=sec_user_agent,
        refresh=args.refresh,
    )
    print(build.coverage.to_string(index=False))
    if not build.coverage["coverage_passed"].all():
        print("Source gate failed: do not run the historical selection experiment.")


if __name__ == "__main__":
    main()
