# SEC EDGAR and yfinance Point-in-Time Fundamentals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a no-subscription, auditable point-in-time fundamental dataset from SEC EDGAR filings and historical yfinance market data, prove whether it satisfies the frozen 80% coverage gate, and only then request approval to run the PAM and HDBSCAN experiment.

**Architecture:** A frozen issuer registry identifies the 25 SEC-covered companies and the local primary listing used when SEC shares represent ordinary shares rather than ADRs. Focused SEC and market-data modules cache raw public responses, reconstruct trailing-twelve-month earnings and filed shares using only records available before each rebalance, and assemble the existing `fundamentals_point_in_time.csv` contract. A separate preparation CLI writes the candidate input, coverage report, and source manifest without invoking a backtest; historical execution remains a later approval gate.

**Tech Stack:** Python 3.10+, pandas 2.2+, NumPy 2+, yfinance 0.2+, Python standard-library `urllib`, pytest 8+, Ruff 0.5+, SEC Company Facts API, Yahoo historical prices and FX rates, Git worktrees, PowerShell.

**Spec:** Implements the source-selection extension to `docs/selection_strategy_spec.md`, governed by `docs/fundamental_source_feasibility.md` and the user's 2026-08-25 approval to try SEC EDGAR plus yfinance before considering manual issuer-report collection.

## Global Constraints

- Work only in `D:\sin_stoks-strategy-lab` on branch `codex/partition-density-strategies`; do not modify or merge `main`.
- Apply the globally installed `clean-code` skill: explicit domain names and types, pure transformations, deterministic tests, no hidden mutation, no magic research parameters, and no undeclared optional dependencies.
- Preserve the frozen 30-stock retrospective universe, annual January rebalances for 2020-2025, weekly returns, 4% risk-free rate, 25% maximum position, 10 bps drift-aware one-way costs, 104-week feature window, 12 selections, and all preregistered promotion gates.
- Use only records with `available_date < rebalance_date`; SEC `filed` dates and market observation dates must both pass the strict inequality.
- Do not use `Ticker.info`, current trailing P/E, current market capitalization, current shares, or a financial value filed after the evaluated rebalance.
- Define trailing earnings as the latest point-in-time TTM value: latest annual income plus latest post-annual year-to-date income minus its same-filing prior-year comparison; use the annual value only when no post-annual interim filing exists.
- Use unadjusted contemporaneous primary-listing closes with SEC-filed ordinary shares; convert market capitalization and earnings to USD with historical yfinance FX observations.
- Require at least 24 of 30 tickers (80%) and at least 12 eligible securities at every rebalance. Do not silently lower either threshold or impute a missing company.
- Treat PRNDY, TCEHY, NTDOY, CCOEY, and UBSFY as explicitly out of scope for this automated pass. Manual issuer-report work requires a separate plan after the automated experiment is reviewed.
- Cache raw SEC, Yahoo price, and FX responses locally, hash them in a manifest, and never commit caches, generated point-in-time fundamentals, credentials, or licence-restricted data.
- Identify SEC requests with an explicitly user-supplied `SEC_USER_AGENT` containing a contact email and respect a minimum 0.12-second interval between network requests. Never derive or transmit `git config user.email` automatically.
- Do not run the historical selection experiment until the generated source/coverage table has been shown to the user and the user has explicitly approved the exact run command.
- Use TDD, focused commits, `python -m pytest`, `python -m ruff check .`, and `git diff --check` at each review gate.

---

## File Responsibility Map

| File | Action | Single responsibility |
|---|---|---|
| `docs/sec_yfinance_fundamental_methodology.md` | Create | Frozen free-source formulas, ticker scope, limitations, and source gate |
| `fundamental_sources.py` | Create | Immutable SEC issuer registry and primary-listing/currency metadata |
| `sec_companyfacts.py` | Create | SEC HTTP/cache client and normalized point-in-time fact extraction |
| `market_reference_data.py` | Create | Cached historical primary-listing prices and currency conversion rates |
| `free_fundamental_builder.py` | Create | Combine SEC earnings/shares and market observations into validated records |
| `prepare_free_fundamentals.py` | Create | CLI that writes candidate fundamentals, coverage, and hash manifest only |
| `.gitignore` | Modify | Exclude source caches and generated candidate fundamentals/artifacts |
| `docs/fundamental_source_feasibility.md` | Modify | Record SEC+yfinance selection and actual automated coverage outcome |
| `tests/test_fundamental_sources.py` | Create | Registry completeness and immutable source choices |
| `tests/test_sec_companyfacts.py` | Create | Cache, chronology, TTM, fact-tag, and share-aggregation behavior |
| `tests/test_market_reference_data.py` | Create | Strict historical price/FX lookup and cache behavior |
| `tests/test_free_fundamental_builder.py` | Create | Formula, chronology, coverage, validation, and no-imputation behavior |
| `tests/test_prepare_free_fundamentals.py` | Create | CLI/output isolation, atomic writes, and manifest hashes |
| `data/source_cache/**` | Generate, do not commit | Raw SEC JSON and Yahoo CSV cache |
| `data/fundamentals_point_in_time.csv` | Generate, do not commit | Validated candidate experiment input |
| `outputs/selection_experiment/source_coverage.csv` | Generate, do not commit | Per-rebalance eligible/missing/error audit |
| `outputs/selection_experiment/source_manifest.json` | Generate, do not commit | Source configuration, request metadata, and SHA-256 hashes |

---

### Task 1: Freeze the Free-Source Methodology and Issuer Registry

**Files:**
- Create: `docs/sec_yfinance_fundamental_methodology.md`
- Create: `fundamental_sources.py`
- Create: `tests/test_fundamental_sources.py`

**Interfaces:**
- Consumes: `PORTFOLIO_TICKERS` from `universe.py` and the frozen 80% coverage rule.
- Produces: `SecIssuerConfig`, `SEC_ISSUERS`, `AUTOMATED_SEC_TICKERS`, and `MANUAL_ONLY_TICKERS` for every later source task.

- [ ] **Step 1: Write the failing registry test**

Create `tests/test_fundamental_sources.py`:

```python
from fundamental_sources import AUTOMATED_SEC_TICKERS, MANUAL_ONLY_TICKERS, SEC_ISSUERS
from universe import PORTFOLIO_TICKERS


def test_sec_registry_partitions_the_frozen_universe():
    assert AUTOMATED_SEC_TICKERS == frozenset(SEC_ISSUERS)
    assert MANUAL_ONLY_TICKERS == frozenset({"PRNDY", "TCEHY", "NTDOY", "CCOEY", "UBSFY"})
    assert AUTOMATED_SEC_TICKERS | MANUAL_ONLY_TICKERS == frozenset(PORTFOLIO_TICKERS)
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
```

- [ ] **Step 2: Run the registry test and confirm the expected failure**

Run:

```powershell
cd D:\sin_stoks-strategy-lab
python -m pytest tests\test_fundamental_sources.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'fundamental_sources'`.

- [ ] **Step 3: Implement the immutable source registry**

Create `fundamental_sources.py` with this interface and registry:

```python
from __future__ import annotations

from dataclasses import dataclass


US_GAAP_INCOME_TAGS = (
    "NetIncomeLossAvailableToCommonStockholdersBasic",
    "NetIncomeLoss",
    "ProfitLoss",
)
IFRS_INCOME_TAGS = (
    "ProfitLossAttributableToOwnersOfParent",
    "ProfitLossAttributableToOrdinaryEquityHoldersOfParentEntity",
    "ProfitLoss",
)


@dataclass(frozen=True)
class SecIssuerConfig:
    ticker: str
    cik: int
    price_symbol: str
    price_currency: str
    price_scale: float
    earnings_namespace: str
    earnings_tags: tuple[str, ...]
    earnings_unit: str
    share_aggregation: str = "reconcile"
    expected_share_classes: int = 1


def _us(ticker: str, cik: int, expected_share_classes: int = 1) -> SecIssuerConfig:
    return SecIssuerConfig(
        ticker=ticker,
        cik=cik,
        price_symbol=ticker,
        price_currency="USD",
        price_scale=1.0,
        earnings_namespace="us-gaap",
        earnings_tags=US_GAAP_INCOME_TAGS,
        earnings_unit="USD",
        share_aggregation="reconcile",
        expected_share_classes=expected_share_classes,
    )


SEC_ISSUERS = {
    "DEO": SecIssuerConfig("DEO", 835403, "DGE.L", "GBP", 0.01, "ifrs-full", IFRS_INCOME_TAGS, "GBP"),
    "BUD": SecIssuerConfig("BUD", 1668717, "ABI.BR", "EUR", 1.0, "ifrs-full", IFRS_INCOME_TAGS, "USD"),
    "STZ": _us("STZ", 16918),
    "BF-B": _us("BF-B", 14693, expected_share_classes=2),
    "MNST": _us("MNST", 865752),
    "CELH": _us("CELH", 1341766),
    "KDP": _us("KDP", 1418135),
    "PEP": _us("PEP", 77476),
    "KO": _us("KO", 21344),
    "META": _us("META", 1326801, expected_share_classes=2),
    "GOOGL": _us("GOOGL", 1652044, expected_share_classes=3),
    "SNAP": _us("SNAP", 1564408, expected_share_classes=3),
    "MSFT": _us("MSFT", 789019),
    "PM": _us("PM", 1413329),
    "BTI": SecIssuerConfig("BTI", 1303523, "BATS.L", "GBP", 0.01, "ifrs-full", IFRS_INCOME_TAGS, "GBP"),
    "MO": _us("MO", 764180),
    "UVV": _us("UVV", 102037),
    "TPB": _us("TPB", 1290677),
    "EA": _us("EA", 712515),
    "TTWO": _us("TTWO", 946581),
    "MCD": _us("MCD", 63908),
    "CMG": _us("CMG", 1058090),
    "YUM": _us("YUM", 1041061),
    "DPZ": _us("DPZ", 1286681),
    "QSR": SecIssuerConfig("QSR", 1618756, "QSR.TO", "CAD", 1.0, "us-gaap", US_GAAP_INCOME_TAGS, "USD"),
}
AUTOMATED_SEC_TICKERS = frozenset(SEC_ISSUERS)
MANUAL_ONLY_TICKERS = frozenset({"PRNDY", "TCEHY", "NTDOY", "CCOEY", "UBSFY"})
```

Format the long registry lines with Ruff rather than changing values.

- [ ] **Step 4: Freeze the methodology document**

Create `docs/sec_yfinance_fundamental_methodology.md` containing all of these explicit rules:

1. SEC filing `filed` is the fundamental availability date and must be strictly before rebalance.
2. Price date is the final primary-listing trading date strictly before rebalance.
3. TTM earnings equals annual earnings plus current YTD minus same-filing prior YTD; annual fallback is permitted only when no post-annual interim exists.
4. Market cap equals scaled unadjusted local close times SEC-filed ordinary shares times spot USD conversion.
5. P/E equals USD market cap divided by USD TTM earnings; non-positive earnings produce `earnings_positive=False` and missing P/E.
6. Earnings use average daily FX over their observation period; market cap uses spot FX on the price date.
7. `available_date` is the maximum of earnings filing date, shares filing date, and price date.
8. Company Facts omits the XBRL dimensions needed to identify share classes reliably. Share values are deduplicated within an accession/end group and reconciled: use the sole value; use the largest value only when it approximately equals the sum of the remaining distinct class values; otherwise require exactly the registry's expected class count and sum those distinct values. Ambiguous groups fail rather than being guessed.
9. The five `MANUAL_ONLY_TICKERS` remain missing, never imputed.
10. Any tag/unit/mapping change requires a new dated plan and marks affected output exploratory.

- [ ] **Step 5: Run focused checks and commit**

Run:

```powershell
python -m pytest tests\test_fundamental_sources.py -v
python -m ruff check fundamental_sources.py tests\test_fundamental_sources.py
python -m ruff format fundamental_sources.py tests\test_fundamental_sources.py
python -m pytest tests\test_fundamental_sources.py -q
git diff --check
```

Expected: all registry tests pass, Ruff is clean, and diff check prints nothing.

Commit:

```powershell
git add fundamental_sources.py docs\sec_yfinance_fundamental_methodology.md tests\test_fundamental_sources.py
git commit -m "docs: freeze free fundamental source methodology"
```

---

### Task 2: Build the Polite, Cached SEC Company Facts Client

**Files:**
- Create: `sec_companyfacts.py`
- Create: `tests/test_sec_companyfacts.py`

**Interfaces:**
- Consumes: SEC CIK values from `SecIssuerConfig` and a caller-supplied `SEC_USER_AGENT`.
- Produces: `SecCompanyFactsClient.get_companyfacts(cik: int, refresh: bool = False) -> dict[str, object]`, `FactRecord`, and `normalize_fact_records(...)`.

- [ ] **Step 1: Write failing cache and normalization tests**

Create `tests/test_sec_companyfacts.py` with an injectable byte downloader:

```python
import json
from pathlib import Path

import pandas as pd

from sec_companyfacts import SecCompanyFactsClient, normalize_fact_records


def test_companyfacts_client_uses_cache_after_first_request(tmp_path):
    calls = []

    def download(url: str, user_agent: str) -> bytes:
        calls.append((url, user_agent))
        return json.dumps({"cik": 1234, "facts": {}}).encode()

    client = SecCompanyFactsClient(
        user_agent="sin_stoks test test@example.com",
        cache_dir=tmp_path,
        downloader=download,
        minimum_interval_seconds=0.0,
    )
    first = client.get_companyfacts(1234)
    second = client.get_companyfacts(1234)

    assert first == second
    assert len(calls) == 1
    assert (tmp_path / "CIK0000001234.json").exists()


def test_normalize_fact_records_preserves_filing_chronology():
    companyfacts = make_companyfacts_fixture()
    records = normalize_fact_records(companyfacts, "us-gaap", "NetIncomeLoss", "USD")

    assert records[0].start == pd.Timestamp("2018-01-01")
    assert records[0].end == pd.Timestamp("2018-12-31")
    assert records[0].filed == pd.Timestamp("2019-02-15")
    assert records[0].form == "10-K"
    assert records[0].value == 100.0
```

Define the fixture in the same test file:

```python
def make_companyfacts_fixture(value: float = 100.0) -> dict[str, object]:
    return {
        "cik": 1234,
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            {
                                "start": "2018-01-01",
                                "end": "2018-12-31",
                                "val": value,
                                "accn": "0001-19-001",
                                "fy": 2018,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2019-02-15",
                                "frame": "CY2018",
                            }
                        ]
                    }
                }
            }
        },
    }
```

- [ ] **Step 2: Verify focused failure**

Run:

```powershell
python -m pytest tests\test_sec_companyfacts.py -k "cache or normalize" -v
```

Expected: collection fails because `sec_companyfacts` does not exist.

- [ ] **Step 3: Implement normalized records and cache client**

Create `sec_companyfacts.py` with:

```python
from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"


@dataclass(frozen=True)
class FactRecord:
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


Downloader = Callable[[str, str], bytes]


def _download(url: str, user_agent: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": user_agent, "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


class SecCompanyFactsClient:
    def __init__(
        self,
        user_agent: str,
        cache_dir: Path,
        downloader: Downloader = _download,
        minimum_interval_seconds: float = 0.12,
    ) -> None:
        if "@" not in user_agent:
            raise ValueError("SEC_USER_AGENT must include a contact email")
        self._user_agent = user_agent
        self._cache_dir = cache_dir
        self._downloader = downloader
        self._minimum_interval_seconds = minimum_interval_seconds
        self._last_request_time = 0.0

    def get_companyfacts(self, cik: int, refresh: bool = False) -> dict[str, object]:
        cache_path = self._cache_dir / f"CIK{cik:010d}.json"
        if cache_path.exists() and not refresh:
            return json.loads(cache_path.read_text(encoding="utf-8"))

        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._minimum_interval_seconds:
            time.sleep(self._minimum_interval_seconds - elapsed)
        payload = self._downloader(SEC_COMPANYFACTS_URL.format(cik=cik), self._user_agent)
        parsed = json.loads(payload)
        if int(parsed.get("cik", -1)) != cik:
            raise ValueError(f"SEC response CIK mismatch for {cik}")

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".json.tmp")
        temporary.write_bytes(payload)
        temporary.replace(cache_path)
        self._last_request_time = time.monotonic()
        return parsed
```

Implement `normalize_fact_records(companyfacts, namespace, tag, unit)` as a pure function that raises a descriptive `ValueError` for a missing namespace/tag/unit, parses dates with `errors="raise"`, converts `val` to finite float, preserves absent `start` as `None`, sorts by `(end, filed, accession)`, and does not mutate its input.

- [ ] **Step 4: Add exact failure tests**

Add tests asserting rejection of:

```python
with pytest.raises(ValueError, match="contact email"):
    SecCompanyFactsClient("anonymous-client", tmp_path)

with pytest.raises(ValueError, match="Missing SEC fact"):
    normalize_fact_records({"facts": {}}, "us-gaap", "NetIncomeLoss", "USD")

with pytest.raises(ValueError, match="finite"):
    normalize_fact_records(make_companyfacts_fixture(value=float("inf")), "us-gaap", "NetIncomeLoss", "USD")
```

- [ ] **Step 5: Run checks and commit**

Run:

```powershell
python -m pytest tests\test_sec_companyfacts.py -v
python -m ruff check sec_companyfacts.py tests\test_sec_companyfacts.py
python -m ruff format sec_companyfacts.py tests\test_sec_companyfacts.py
python -m pytest tests\test_sec_companyfacts.py -q
git diff --check
```

Commit:

```powershell
git add sec_companyfacts.py tests\test_sec_companyfacts.py
git commit -m "feat: cache and normalize SEC company facts"
```

---

### Task 3: Reconstruct Point-in-Time TTM Earnings

**Files:**
- Modify: `sec_companyfacts.py`
- Modify: `tests/test_sec_companyfacts.py`

**Interfaces:**
- Consumes: normalized `FactRecord` values and `SecIssuerConfig.earnings_tags`.
- Produces: `EarningsObservation` and `select_ttm_earnings(companyfacts: dict[str, object], issuer: SecIssuerConfig, cutoff: pd.Timestamp) -> EarningsObservation`.

- [ ] **Step 1: Write the failing TTM formula test**

Add this deterministic case to `tests/test_sec_companyfacts.py`:

```python
def test_ttm_earnings_uses_annual_plus_current_ytd_minus_same_filing_prior_ytd():
    facts = make_ttm_companyfacts(
        annual_value=100.0,
        annual_end="2018-12-31",
        annual_filed="2019-02-15",
        current_ytd_value=30.0,
        current_start="2019-01-01",
        current_end="2019-09-30",
        prior_ytd_value=20.0,
        prior_start="2018-01-01",
        prior_end="2018-09-30",
        interim_filed="2019-11-01",
        interim_accession="0001-19-003",
    )

    observation = select_ttm_earnings(facts, make_us_issuer(), pd.Timestamp("2020-01-01"))

    assert observation.value == 110.0
    assert observation.start == pd.Timestamp("2018-10-01")
    assert observation.end == pd.Timestamp("2019-09-30")
    assert observation.available_date == pd.Timestamp("2019-11-01")
    assert observation.method == "annual_plus_ytd_less_prior_ytd"
```

Define the helper with both current and prior YTD facts under the same accession and filing date so the implementation cannot mix revisions:

```python
def make_ttm_companyfacts(
    annual_value: float,
    annual_end: str,
    annual_filed: str,
    current_ytd_value: float,
    current_start: str,
    current_end: str,
    prior_ytd_value: float,
    prior_start: str,
    prior_end: str,
    interim_filed: str,
    interim_accession: str,
) -> dict[str, object]:
    records = [
        {
            "start": "2018-01-01",
            "end": annual_end,
            "val": annual_value,
            "accn": "0001-19-001",
            "fy": 2018,
            "fp": "FY",
            "form": "10-K",
            "filed": annual_filed,
            "frame": "CY2018",
        },
        {
            "start": current_start,
            "end": current_end,
            "val": current_ytd_value,
            "accn": interim_accession,
            "fy": 2019,
            "fp": "Q3",
            "form": "10-Q",
            "filed": interim_filed,
            "frame": None,
        },
        {
            "start": prior_start,
            "end": prior_end,
            "val": prior_ytd_value,
            "accn": interim_accession,
            "fy": 2019,
            "fp": "Q3",
            "form": "10-Q",
            "filed": interim_filed,
            "frame": None,
        },
    ]
    return {
        "cik": 1,
        "facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": records}}}},
    }


def make_us_issuer(expected_share_classes: int = 1) -> SecIssuerConfig:
    return SecIssuerConfig(
        ticker="TEST",
        cik=1,
        price_symbol="TEST",
        price_currency="USD",
        price_scale=1.0,
        earnings_namespace="us-gaap",
        earnings_tags=("NetIncomeLoss",),
        earnings_unit="USD",
        share_aggregation="reconcile",
        expected_share_classes=expected_share_classes,
    )
```

- [ ] **Step 2: Run and confirm missing symbol failure**

Run:

```powershell
python -m pytest tests\test_sec_companyfacts.py::test_ttm_earnings_uses_annual_plus_current_ytd_minus_same_filing_prior_ytd -v
```

Expected: import or name failure for `select_ttm_earnings`.

- [ ] **Step 3: Implement exact earnings selection**

Add:

```python
ANNUAL_FORMS = frozenset({"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"})
INTERIM_FORMS = frozenset({"10-Q", "10-Q/A", "6-K"})


@dataclass(frozen=True)
class EarningsObservation:
    start: pd.Timestamp
    end: pd.Timestamp
    available_date: pd.Timestamp
    value: float
    unit: str
    tag: str
    method: str
    accessions: tuple[str, ...]
```

For each configured tag in order:

1. normalize the configured namespace/tag/unit;
2. filter records to `filed < cutoff`, `end < cutoff`, and non-null `start`;
3. classify annual duration as 300-430 days and interim duration as 60-300 days;
4. select the annual record with maximum `(end, filed, accession)` from annual forms;
5. find interim records ending after the annual end, then choose the maximum `(end, filed, duration)` so a YTD fact wins over a single-quarter fact with the same end;
6. find the prior-year comparison in the same interim accession with duration difference at most 14 days and end 330-400 days before the current YTD end;
7. return `annual + current_ytd - prior_ytd` when the pair exists;
8. return the annual record unchanged only when no post-annual interim record exists;
9. reject an interim record lacking its same-filing prior comparison rather than silently falling back;
10. move to the next tag only when the current tag cannot produce a complete observation.

If no configured tag works, raise `ValueError` containing ticker, cutoff date, and each attempted tag's reason.

- [ ] **Step 4: Add chronology and fallback tests**

Define these fixture transforms above the tests:

```python
def annual_only_facts() -> dict[str, object]:
    facts = make_ttm_companyfacts(
        100.0, "2018-12-31", "2019-02-15", 30.0, "2019-01-01", "2019-09-30",
        20.0, "2018-01-01", "2018-09-30", "2019-11-01", "0001-19-003"
    )
    facts["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"] = facts["facts"][
        "us-gaap"
    ]["NetIncomeLoss"]["units"]["USD"][:1]
    return facts


def facts_with_future_revision() -> dict[str, object]:
    facts = make_ttm_companyfacts(
        100.0, "2018-12-31", "2019-02-15", 30.0, "2019-01-01", "2019-09-30",
        20.0, "2018-01-01", "2018-09-30", "2019-11-01", "0001-19-003"
    )
    future = dict(facts["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"][1])
    future.update({"val": 999.0, "filed": "2020-02-01", "accn": "0001-20-001"})
    facts["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"].append(future)
    return facts


def unmatched_interim_facts() -> dict[str, object]:
    facts = make_ttm_companyfacts(
        100.0, "2018-12-31", "2019-02-15", 30.0, "2019-01-01", "2019-09-30",
        20.0, "2018-01-01", "2018-09-30", "2019-11-01", "0001-19-003"
    )
    facts["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"].pop()
    return facts
```

Add concrete tests proving:

```python
def test_ttm_earnings_ignores_future_filing():
    issuer = make_us_issuer()
    before = select_ttm_earnings(facts_with_future_revision(), issuer, pd.Timestamp("2020-01-01"))
    assert before.value == 110.0
    assert before.available_date < pd.Timestamp("2020-01-01")


def test_ttm_earnings_uses_annual_when_no_later_interim_exists():
    observation = select_ttm_earnings(
        annual_only_facts(), make_us_issuer(), pd.Timestamp("2019-04-01")
    )
    assert observation.value == 100.0
    assert observation.method == "annual_fallback"


def test_ttm_earnings_rejects_unmatched_interim_comparison():
    with pytest.raises(ValueError, match="same-filing prior-year comparison"):
        select_ttm_earnings(
            unmatched_interim_facts(), make_us_issuer(), pd.Timestamp("2020-01-01")
        )
```

- [ ] **Step 5: Run checks and commit**

Run:

```powershell
python -m pytest tests\test_sec_companyfacts.py -k earnings -v
python -m pytest tests\test_sec_companyfacts.py -q
python -m ruff check sec_companyfacts.py tests\test_sec_companyfacts.py
git diff --check
```

Commit:

```powershell
git add sec_companyfacts.py tests\test_sec_companyfacts.py
git commit -m "feat: reconstruct point-in-time TTM earnings"
```

---

### Task 4: Select and Reconcile Filed Shares Outstanding

**Files:**
- Modify: `sec_companyfacts.py`
- Modify: `tests/test_sec_companyfacts.py`

**Interfaces:**
- Consumes: `dei.EntityCommonStockSharesOutstanding`, fallback `us-gaap.CommonStockSharesOutstanding`, cutoff date, and `SecIssuerConfig.share_aggregation`.
- Produces: `SharesObservation` and `select_filed_shares(companyfacts: dict[str, object], issuer: SecIssuerConfig, cutoff: pd.Timestamp) -> SharesObservation`.

- [ ] **Step 1: Write failing share-reconciliation tests**

Add this exact fixture and the tests:

```python
def make_share_facts(
    values: list[float],
    end: str,
    filed: str,
) -> dict[str, object]:
    records = [
        {
            "end": end,
            "val": value,
            "accn": "0001-19-004",
            "fy": 2019,
            "fp": "Q3",
            "form": "10-Q",
            "filed": filed,
            "frame": None,
        }
        for value in values
    ]
    return {
        "cik": 1,
        "facts": {
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "units": {"shares": records}
                }
            }
        },
    }


def test_single_class_shares_deduplicate_repeated_values():
    observation = select_filed_shares(
        make_share_facts(values=[100.0, 100.0], end="2019-10-31", filed="2019-11-05"),
        make_us_issuer(expected_share_classes=1),
        pd.Timestamp("2020-01-01"),
    )
    assert observation.shares == 100.0
    assert observation.available_date == pd.Timestamp("2019-11-05")


def test_multiclass_shares_use_reported_total_when_it_matches_components():
    observation = select_filed_shares(
        make_share_facts(values=[60.0, 40.0, 100.0], end="2019-10-31", filed="2019-11-05"),
        make_us_issuer(expected_share_classes=2),
        pd.Timestamp("2020-01-01"),
    )
    assert observation.shares == 100.0


def test_multiclass_shares_reject_ambiguous_values():
    with pytest.raises(ValueError, match="ambiguous share facts"):
        select_filed_shares(
            make_share_facts(values=[60.0, 40.0, 30.0], end="2019-10-31", filed="2019-11-05"),
            make_us_issuer(expected_share_classes=2),
            pd.Timestamp("2020-01-01"),
        )
```

- [ ] **Step 2: Verify focused failure**

Run:

```powershell
python -m pytest tests\test_sec_companyfacts.py -k shares -v
```

Expected: failure because `select_filed_shares` is absent.

- [ ] **Step 3: Implement strict share selection**

Add:

```python
@dataclass(frozen=True)
class SharesObservation:
    observation_date: pd.Timestamp
    available_date: pd.Timestamp
    shares: float
    tag: str
    accession: str
    aggregation: str
    component_count: int
```

Implementation rules:

1. try `dei.EntityCommonStockSharesOutstanding` in `shares`, then `us-gaap.CommonStockSharesOutstanding`;
2. filter to `filed < cutoff`, `end < cutoff`, finite positive values, and forms in `ANNUAL_FORMS | INTERIM_FORMS | {"8-K"}`;
3. select the latest group by `(end, filed, accession)`;
4. deduplicate exactly repeated values within the group before reconciliation;
5. use the sole distinct value for a single-class issuer;
6. for a multi-class issuer, use the largest value when it approximately equals the sum of the remaining distinct values, otherwise sum only when the number of distinct values exactly matches `expected_share_classes`;
7. reject every ambiguous group rather than guessing, and preserve the reconciliation method and component count in the observation;
8. reject a selected observation older than 200 days relative to the market-price date in the record builder;
9. raise with ticker and cutoff when neither tag produces a valid observation.

- [ ] **Step 4: Add future, staleness, and invalid-value tests**

Add:

```python
def test_filed_shares_ignore_future_filing():
    facts = make_share_facts([100.0], "2019-10-31", "2019-11-05")
    future = dict(
        facts["facts"]["dei"]["EntityCommonStockSharesOutstanding"]["units"]["shares"][0]
    )
    future.update({"val": 500.0, "filed": "2020-02-01", "accn": "0001-20-001"})
    facts["facts"]["dei"]["EntityCommonStockSharesOutstanding"]["units"]["shares"].append(
        future
    )
    observation = select_filed_shares(facts, make_us_issuer(), pd.Timestamp("2020-01-01"))
    assert observation.shares == 100.0


def test_filed_shares_reject_stale_observation():
    facts = make_share_facts([100.0], "2019-01-01", "2019-02-01")
    with pytest.raises(ValueError, match="stale"):
        select_filed_shares(facts, make_us_issuer(), pd.Timestamp("2020-08-01"))


@pytest.mark.parametrize("invalid_value", [0.0, float("nan")])
def test_filed_shares_reject_invalid_values(invalid_value):
    facts = make_share_facts([invalid_value], "2019-10-31", "2019-11-05")
    with pytest.raises(ValueError, match="finite and positive"):
        select_filed_shares(facts, make_us_issuer(), pd.Timestamp("2020-01-01"))
```

- [ ] **Step 5: Run checks and commit**

Run:

```powershell
python -m pytest tests\test_sec_companyfacts.py -k shares -v
python -m pytest tests\test_sec_companyfacts.py -q
python -m ruff check sec_companyfacts.py tests\test_sec_companyfacts.py
git diff --check
```

Commit:

```powershell
git add sec_companyfacts.py tests\test_sec_companyfacts.py
git commit -m "feat: select filed historical shares"
```

---

### Task 5: Build Cached Historical Price and FX Lookups

**Files:**
- Create: `market_reference_data.py`
- Create: `tests/test_market_reference_data.py`

**Interfaces:**
- Consumes: primary-listing symbols/currencies from `SEC_ISSUERS` and yfinance daily history.
- Produces: `PriceObservation`, `MarketReferenceData.from_yfinance(...)`, `close_before(symbol, cutoff, scale)`, `spot_usd_rate(currency, date)`, and `average_usd_rate(currency, start, end)`.

- [ ] **Step 1: Write failing strict-date and FX tests**

Create tests using these exact helpers and injected series rather than network calls:

```python
def series(values: dict[str, float]) -> pd.Series:
    return pd.Series(values, dtype=float).set_axis(pd.to_datetime(list(values)))


def make_market_reference(histories: dict[str, pd.Series]) -> MarketReferenceData:
    return MarketReferenceData(histories)


def test_close_before_uses_last_observation_strictly_before_cutoff():
    market = make_market_reference(
        {"DGE.L": series({"2019-12-30": 3100.0, "2020-01-01": 3200.0})}
    )
    price = market.close_before("DGE.L", pd.Timestamp("2020-01-01"), scale=0.01)
    assert price.date == pd.Timestamp("2019-12-30")
    assert price.value == 31.0


def test_currency_conversion_uses_spot_for_market_cap_and_period_average_for_earnings():
    market = make_market_reference(
        {"GBPUSD=X": series({"2019-06-30": 1.20, "2019-12-30": 1.30})}
    )
    assert market.spot_usd_rate("GBP", pd.Timestamp("2019-12-31")) == 1.30
    assert market.average_usd_rate(
        "GBP", pd.Timestamp("2019-06-01"), pd.Timestamp("2019-12-31")
    ) == 1.25
    assert market.spot_usd_rate("USD", pd.Timestamp("2019-12-31")) == 1.0
```

- [ ] **Step 2: Verify module-absent failure**

Run:

```powershell
python -m pytest tests\test_market_reference_data.py -v
```

Expected: collection fails for missing `market_reference_data`.

- [ ] **Step 3: Implement pure historical lookups**

Create:

```python
@dataclass(frozen=True)
class PriceObservation:
    symbol: str
    date: pd.Timestamp
    value: float


FX_SYMBOLS = {
    "CAD": "CADUSD=X",
    "EUR": "EURUSD=X",
    "GBP": "GBPUSD=X",
}
```

`MarketReferenceData` must defensively copy sorted `pd.Series` histories; reject duplicate dates, non-finite/non-positive values, unknown currencies, missing observations, and any lookup whose selected date is not strictly before the requested cutoff. `average_usd_rate` must use observations between inclusive `start` and `end`, require at least one value, and return 1.0 for USD.

- [ ] **Step 4: Implement cached yfinance acquisition**

Add:

```python
@classmethod
def from_yfinance(
    cls,
    symbols: tuple[str, ...],
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path,
    refresh: bool = False,
) -> "MarketReferenceData":
    requested_symbols = tuple(dict.fromkeys((*symbols, *FX_SYMBOLS.values())))
    cache_paths = {symbol: cache_dir / f"{_safe_filename(symbol)}.csv" for symbol in requested_symbols}
    if not refresh and all(path.exists() for path in cache_paths.values()):
        histories = {symbol: _read_cached_history(path) for symbol, path in cache_paths.items()}
        return cls(histories)

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
    if missing_symbols:
        raise ValueError(f"Yahoo returned no Close history for: {missing_symbols}")
    for symbol, history in histories.items():
        _write_history_atomically(history, cache_paths[symbol])
    return cls(histories)
```

Implement `_safe_filename(symbol: str) -> str`, `_read_cached_history(path: Path) -> pd.Series`, `_extract_close_histories(downloaded: pd.DataFrame, requested_symbols: tuple[str, ...]) -> dict[str, pd.Series]`, and `_write_history_atomically(history: pd.Series, target: Path) -> None` in the same module. `_safe_filename` replaces `=` with `_eq_`, `/` with `_slash_`, and every non-alphanumeric/dot/dash character with `_`; cached CSVs use columns `date,close` and ISO dates.

The method must:

1. add `CADUSD=X`, `EURUSD=X`, and `GBPUSD=X` to the requested primary-listing symbols;
2. use one `yf.download(..., auto_adjust=False, actions=False, progress=False)` request;
3. extract `Close` deterministically from single- or multi-index responses;
4. write one atomic CSV per symbol under `data/source_cache/yahoo/`;
5. reuse caches only after verifying that every cache covers the requested start/end bounds; partial or stale caches must fail or be refreshed explicitly;
6. request dates from `2017-01-01` through `2026-01-02`, covering all filing periods and price cutoffs;
7. preserve exact missing symbols in cache metadata so the coverage builder can record ticker-level failures; strict callers still raise, while the source-probe caller continues without imputation.

- [ ] **Step 5: Test the cache without network**

Add:

```python
def downloaded_frame(symbols: list[str]) -> pd.DataFrame:
    dates = pd.to_datetime(["2019-12-27", "2019-12-30"])
    columns = pd.MultiIndex.from_product([["Close"], symbols])
    values = np.tile(np.array([[10.0], [11.0]]), (1, len(symbols)))
    return pd.DataFrame(values, index=dates, columns=columns)


def test_from_yfinance_writes_and_reuses_symbol_caches(tmp_path, monkeypatch):
    calls = []

    def fake_download(symbols, **kwargs):
        calls.append((tuple(symbols), kwargs))
        return downloaded_frame(list(symbols))

    monkeypatch.setattr("market_reference_data.yf.download", fake_download)
    symbols = ("TEST", "QSR.TO")
    MarketReferenceData.from_yfinance(
        symbols, pd.Timestamp("2017-01-01"), pd.Timestamp("2026-01-02"), tmp_path, refresh=True
    )
    MarketReferenceData.from_yfinance(
        symbols, pd.Timestamp("2017-01-01"), pd.Timestamp("2026-01-02"), tmp_path
    )

    assert len(calls) == 1
    assert (tmp_path / "TEST.csv").exists()
    assert (tmp_path / "QSR.TO.csv").exists()
    assert len(list(tmp_path.glob("*.csv"))) == 5


def test_from_yfinance_rejects_missing_symbol(tmp_path, monkeypatch):
    def missing_qsr(symbols, **kwargs):
        retained = [symbol for symbol in symbols if symbol != "QSR.TO"]
        return downloaded_frame(retained)

    monkeypatch.setattr("market_reference_data.yf.download", missing_qsr)
    with pytest.raises(ValueError, match="QSR.TO"):
        MarketReferenceData.from_yfinance(
            ("TEST", "QSR.TO"),
            pd.Timestamp("2017-01-01"),
            pd.Timestamp("2026-01-02"),
            tmp_path,
            refresh=True,
        )
```

- [ ] **Step 6: Run checks and commit**

Run:

```powershell
python -m pytest tests\test_market_reference_data.py -v
python -m ruff check market_reference_data.py tests\test_market_reference_data.py
python -m ruff format market_reference_data.py tests\test_market_reference_data.py
python -m pytest tests\test_market_reference_data.py -q
git diff --check
```

Commit:

```powershell
git add market_reference_data.py tests\test_market_reference_data.py
git commit -m "feat: cache historical prices and FX rates"
```

---

### Task 6: Assemble and Validate Point-in-Time Fundamental Records

**Files:**
- Create: `free_fundamental_builder.py`
- Create: `tests/test_free_fundamental_builder.py`

**Interfaces:**
- Consumes: `SecCompanyFactsClient`, `MarketReferenceData`, `SEC_ISSUERS`, `select_ttm_earnings`, `select_filed_shares`, and `load_fundamentals`.
- Produces: `build_fundamental_record(...) -> dict[str, object]` and `build_free_fundamentals(...) -> FreeFundamentalBuild`.

- [ ] **Step 1: Write the failing formula test**

Create the fake market and test:

```python
class FakeMarket:
    def __init__(self, price: float, price_date: str, spot_fx: float, average_fx: float) -> None:
        self._price = PriceObservation("TEST.L", pd.Timestamp(price_date), price)
        self._spot_fx = spot_fx
        self._average_fx = average_fx

    def close_before(self, symbol: str, cutoff: pd.Timestamp, scale: float) -> PriceObservation:
        assert symbol == "TEST.L"
        assert self._price.date < cutoff
        return PriceObservation(symbol, self._price.date, self._price.value * scale)

    def spot_usd_rate(self, currency: str, date: pd.Timestamp) -> float:
        assert currency == "GBP"
        assert date == self._price.date
        return self._spot_fx

    def average_usd_rate(
        self, currency: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> float:
        assert currency == "GBP"
        assert start < end
        return self._average_fx


def make_market(price: float, price_date: str, spot_fx: float, average_fx: float) -> FakeMarket:
    return FakeMarket(price, price_date, spot_fx, average_fx)


def test_builder_calculates_usd_market_cap_and_pe_from_point_in_time_inputs():
    issuer = SecIssuerConfig(
        ticker="TEST",
        cik=1,
        price_symbol="TEST.L",
        price_currency="GBP",
        price_scale=0.01,
        earnings_namespace="ifrs-full",
        earnings_tags=("ProfitLoss",),
        earnings_unit="GBP",
    )
    earnings = EarningsObservation(
        start=pd.Timestamp("2018-10-01"),
        end=pd.Timestamp("2019-09-30"),
        available_date=pd.Timestamp("2019-11-01"),
        value=100.0,
        unit="GBP",
        tag="ProfitLoss",
        method="annual_plus_ytd_less_prior_ytd",
        accessions=("annual", "interim"),
    )
    shares = SharesObservation(
        observation_date=pd.Timestamp("2019-10-31"),
        available_date=pd.Timestamp("2019-11-05"),
        shares=10.0,
        tag="EntityCommonStockSharesOutstanding",
        accession="shares",
        aggregation="max",
        component_count=1,
    )
    market = make_market(price=2000.0, price_date="2019-12-30", spot_fx=1.30, average_fx=1.25)

    record = build_fundamental_record(
        issuer, pd.Timestamp("2020-01-01"), earnings, shares, market
    )

    assert record["market_cap"] == 260.0
    assert record["trailing_pe"] == 260.0 / 125.0
    assert record["available_date"] == pd.Timestamp("2019-12-30")
    assert record["available_date"] < pd.Timestamp("2020-01-01")
```

- [ ] **Step 2: Verify focused failure**

Run:

```powershell
python -m pytest tests\test_free_fundamental_builder.py::test_builder_calculates_usd_market_cap_and_pe_from_point_in_time_inputs -v
```

Expected: collection fails because `free_fundamental_builder` does not exist.

- [ ] **Step 3: Implement record formulas**

Create:

```python
@dataclass(frozen=True)
class FreeFundamentalBuild:
    fundamentals: pd.DataFrame
    coverage: pd.DataFrame
    errors: pd.DataFrame
```

`build_fundamental_record` must calculate:

```text
scaled_local_price = primary_listing_close * price_scale
market_cap_usd = scaled_local_price * filed_shares * spot_fx_to_usd
trailing_earnings_usd = ttm_earnings * average_period_fx_to_usd
trailing_pe = market_cap_usd / trailing_earnings_usd  when earnings > 0
```

Set `earnings_positive` from `trailing_earnings_usd > 0`; set `trailing_pe=np.nan` otherwise. Set `observation_date` to price date and `available_date` to `max(price_date, earnings.available_date, shares.available_date)`. Raise if either date is not strictly before rebalance, market cap is non-finite/non-positive, positive earnings yields non-finite/non-positive P/E, or the selected share observation is more than 200 days old at the price date. Persist structured provenance columns (`rebalance_date`, `cik`, `price_symbol`, `price_date`, `price_currency`, `spot_fx_to_usd`, `earnings_start`, `earnings_end`, `earnings_available_date`, `earnings_tag`, `earnings_accessions`, `earnings_method`, `shares_date`, `shares_available_date`, `shares_tag`, `shares_accession`, `shares_aggregation`, `shares_component_count`, `filed_shares`, and `trailing_earnings_usd`) in addition to a concise source label. The production loader must preserve and validate these columns rather than discarding them.

- [ ] **Step 4: Implement the six-rebalance build**

Implement:

```python
def build_free_fundamentals(
    sec_client: SecCompanyFactsClient,
    market_data: MarketReferenceData,
    rebalance_years: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024, 2025),
) -> FreeFundamentalBuild:
    fundamental_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    for year in rebalance_years:
        rebalance_date = pd.Timestamp(year=year, month=1, day=1)
        successful_tickers: set[str] = set()
        failed_tickers: set[str] = set()
        for ticker, issuer in sorted(SEC_ISSUERS.items()):
            try:
                companyfacts = sec_client.get_companyfacts(issuer.cik)
                earnings = select_ttm_earnings(companyfacts, issuer, rebalance_date)
                shares = select_filed_shares(companyfacts, issuer, rebalance_date)
                record = build_fundamental_record(
                    issuer, rebalance_date, earnings, shares, market_data
                )
                record["rebalance_date"] = rebalance_date
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
        coverage_rows.append(
            {
                "rebalance_date": rebalance_date,
                "expected_assets": 30,
                "automated_candidates": len(SEC_ISSUERS),
                "eligible_assets": eligible_assets,
                "coverage": eligible_assets / 30.0,
                "minimum_required_assets": 24,
                "coverage_passed": eligible_assets >= 24,
                "missing_tickers": ", ".join(sorted(MANUAL_ONLY_TICKERS)),
                "failed_tickers": ", ".join(sorted(failed_tickers)),
            }
        )
    return FreeFundamentalBuild(
        fundamentals=pd.DataFrame(fundamental_rows),
        coverage=pd.DataFrame(coverage_rows),
        errors=pd.DataFrame(error_rows, columns=["rebalance_date", "ticker", "stage", "error"]),
    )
```

For each year and each of the 25 configured issuers, fetch/cache company facts, calculate earnings/shares, and append either one record or one error containing `rebalance_date`, `ticker`, `stage`, and exact exception text. Never append a synthetic or imputed record after an error. Add the five manual-only tickers to coverage as `not_automated` rather than errors.

Coverage columns must be:

```text
rebalance_date, expected_assets, automated_candidates, eligible_assets,
coverage, minimum_required_assets, coverage_passed, missing_tickers, failed_tickers
```

Use `expected_assets=30`, `automated_candidates=25`, `minimum_required_assets=24`, and `coverage_passed=(eligible_assets >= 24)`. Sort output by available date and ticker, then call the existing `load_fundamentals` through a temporary CSV in the orchestration layer so the exact production contract validates the candidate.

- [ ] **Step 5: Add leakage, negative-earnings, and coverage tests**

Add these helpers and tests:

```python
class FakeSecClient:
    def get_companyfacts(self, cik: int) -> dict[str, object]:
        return {"cik": cik}


def build_with_eligible_count(monkeypatch, eligible_count: int) -> FreeFundamentalBuild:
    issuers = {
        f"T{position:02d}": SecIssuerConfig(
            ticker=f"T{position:02d}",
            cik=position + 1,
            price_symbol=f"T{position:02d}",
            price_currency="USD",
            price_scale=1.0,
            earnings_namespace="us-gaap",
            earnings_tags=("NetIncomeLoss",),
            earnings_unit="USD",
        )
        for position in range(25)
    }
    monkeypatch.setattr("free_fundamental_builder.SEC_ISSUERS", issuers)
    monkeypatch.setattr(
        "free_fundamental_builder.MANUAL_ONLY_TICKERS",
        frozenset({"M0", "M1", "M2", "M3", "M4"}),
    )

    def fake_earnings(companyfacts, issuer, cutoff):
        if issuer.cik > eligible_count:
            raise ValueError("synthetic missing earnings")
        return EarningsObservation(
            pd.Timestamp("2018-10-01"),
            pd.Timestamp("2019-09-30"),
            pd.Timestamp("2019-11-01"),
            100.0,
            "USD",
            "NetIncomeLoss",
            "annual_plus_ytd_less_prior_ytd",
            ("annual", "interim"),
        )

    def fake_shares(companyfacts, issuer, cutoff):
        return SharesObservation(
            pd.Timestamp("2019-10-31"),
            pd.Timestamp("2019-11-05"),
            10.0,
            "EntityCommonStockSharesOutstanding",
            "shares",
            issuer.share_aggregation,
            1,
        )

    def fake_record(issuer, rebalance_date, earnings, shares, market_data):
        return {
            "ticker": issuer.ticker,
            "observation_date": pd.Timestamp("2019-12-30"),
            "available_date": pd.Timestamp("2019-12-30"),
            "trailing_pe": 2.0,
            "market_cap": 200.0,
            "earnings_positive": True,
            "source": "Synthetic SEC CIK and Yahoo",
        }

    monkeypatch.setattr("free_fundamental_builder.select_ttm_earnings", fake_earnings)
    monkeypatch.setattr("free_fundamental_builder.select_filed_shares", fake_shares)
    monkeypatch.setattr("free_fundamental_builder.build_fundamental_record", fake_record)
    return build_free_fundamentals(
        FakeSecClient(), object(), rebalance_years=(2020,)
    )


def test_builder_rejects_future_available_fundamental():
    issuer = make_test_issuer()
    earnings, shares, market = make_builder_inputs()
    earnings = dataclasses.replace(earnings, available_date=pd.Timestamp("2020-02-01"))
    with pytest.raises(ValueError, match="strictly before rebalance"):
        build_fundamental_record(issuer, pd.Timestamp("2020-01-01"), earnings, shares, market)


def test_builder_keeps_negative_earnings_without_pe():
    issuer = make_test_issuer()
    earnings, shares, market = make_builder_inputs()
    earnings = dataclasses.replace(earnings, value=-100.0)
    record = build_fundamental_record(
        issuer, pd.Timestamp("2020-01-01"), earnings, shares, market
    )
    assert record["market_cap"] > 0
    assert record["earnings_positive"] is False
    assert np.isnan(record["trailing_pe"])


@pytest.mark.parametrize(("eligible", "passed"), [(24, True), (23, False)])
def test_coverage_uses_frozen_thirty_asset_denominator(monkeypatch, eligible, passed):
    build = build_with_eligible_count(monkeypatch, eligible)
    row = build.coverage.iloc[0]
    assert row["coverage"] == eligible / 30.0
    assert row["coverage_passed"] == passed
    assert row["minimum_required_assets"] == 24
    assert set(build.fundamentals["ticker"]).isdisjoint({"M0", "M1", "M2", "M3", "M4"})
    assert (build.fundamentals["available_date"] < build.fundamentals["rebalance_date"]).all()
```

Place these helpers above the tests and import `dataclasses`, NumPy, pandas, and pytest at the top of the module:

```python
def make_test_issuer() -> SecIssuerConfig:
    return SecIssuerConfig(
        ticker="TEST",
        cik=1,
        price_symbol="TEST.L",
        price_currency="GBP",
        price_scale=0.01,
        earnings_namespace="ifrs-full",
        earnings_tags=("ProfitLoss",),
        earnings_unit="GBP",
    )


def make_builder_inputs() -> tuple[EarningsObservation, SharesObservation, FakeMarket]:
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
        "max",
        1,
    )
    market = make_market(2000.0, "2019-12-30", 1.30, 1.25)
    return earnings, shares, market
```

- [ ] **Step 6: Run checks and commit**

Run:

```powershell
python -m pytest tests\test_free_fundamental_builder.py -v
python -m pytest tests\test_fundamental_data.py tests\test_free_fundamental_builder.py -q
python -m ruff check free_fundamental_builder.py tests\test_free_fundamental_builder.py
git diff --check
```

Commit:

```powershell
git add free_fundamental_builder.py tests\test_free_fundamental_builder.py
git commit -m "feat: assemble free point-in-time fundamentals"
```

---

### Task 7: Add an Output-Isolated Preparation CLI and Manifest

**Files:**
- Create: `prepare_free_fundamentals.py`
- Create: `tests/test_prepare_free_fundamentals.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: source clients and `build_free_fundamentals`.
- Produces: candidate `data/fundamentals_point_in_time.csv`, `outputs/selection_experiment/source_coverage.csv`, `source_errors.csv`, and `source_manifest.json`; it never imports or calls a backtest runner.

- [ ] **Step 1: Write the failing orchestration test**

Create a test with this complete synthetic build:

```python
def successful_build() -> FreeFundamentalBuild:
    fundamentals = pd.DataFrame(
        [
            {
                "ticker": f"T{position:02d}",
                "rebalance_date": pd.Timestamp("2020-01-01"),
                "observation_date": pd.Timestamp("2019-12-30"),
                "available_date": pd.Timestamp("2019-12-30"),
                "trailing_pe": 10.0 + position,
                "market_cap": 1_000_000.0 * (position + 1),
                "earnings_positive": True,
                "source": "Synthetic SEC CIK and Yahoo",
            }
            for position in range(24)
        ]
    )
    coverage = pd.DataFrame(
        [
            {
                "rebalance_date": pd.Timestamp("2020-01-01"),
                "expected_assets": 30,
                "automated_candidates": 25,
                "eligible_assets": 24,
                "coverage": 0.80,
                "minimum_required_assets": 24,
                "coverage_passed": True,
                "missing_tickers": "M0, M1, M2, M3, M4",
                "failed_tickers": "T24",
            }
        ]
    )
    errors = pd.DataFrame(
        [{"rebalance_date": "2020-01-01", "ticker": "T24", "stage": "ValueError", "error": "missing"}]
    )
    return FreeFundamentalBuild(fundamentals, coverage, errors)


def test_preparation_writes_only_source_outputs(tmp_path, monkeypatch):
    build = successful_build()
    monkeypatch.setattr(
        "prepare_free_fundamentals.build_source_clients", lambda **kwargs: (object(), object())
    )
    monkeypatch.setattr(
        "prepare_free_fundamentals.build_free_fundamentals", lambda sec, market: build
    )
    monkeypatch.setattr(
        "prepare_free_fundamentals.build_source_manifest",
        lambda result, fundamentals_path, cache_dir: {"methodology": "sec-yfinance-v1"},
    )
    result = prepare_free_fundamentals(
        fundamentals_path=tmp_path / "data/fundamentals_point_in_time.csv",
        output_dir=tmp_path / "outputs/selection_experiment",
        cache_dir=tmp_path / "data/source_cache",
        sec_user_agent="sin_stoks test test@example.com",
    )

    assert result.coverage["coverage_passed"].all()
    assert (tmp_path / "data/fundamentals_point_in_time.csv").exists()
    assert (tmp_path / "outputs/selection_experiment/source_coverage.csv").exists()
    assert (tmp_path / "outputs/selection_experiment/source_errors.csv").exists()
    assert (tmp_path / "outputs/selection_experiment/source_manifest.json").exists()
    assert not (tmp_path / "outputs/selection_experiment/full").exists()
    assert not (tmp_path / "outputs/portfolio_backtest").exists()
```

- [ ] **Step 2: Verify module-absent failure**

Run:

```powershell
python -m pytest tests\test_prepare_free_fundamentals.py -v
```

Expected: collection fails because `prepare_free_fundamentals` does not exist.

- [ ] **Step 3: Implement atomic source output**

Implement:

```python
def prepare_free_fundamentals(
    fundamentals_path: Path,
    output_dir: Path,
    cache_dir: Path,
    sec_user_agent: str,
    refresh: bool = False,
) -> FreeFundamentalBuild:
    sec_client, market_data = build_source_clients(
        cache_dir=cache_dir,
        sec_user_agent=sec_user_agent,
        refresh=refresh,
    )
    build = build_free_fundamentals(sec_client, market_data)
    final_columns = [
        "ticker", "rebalance_date", "observation_date", "available_date",
        "trailing_pe", "market_cap", "earnings_positive", "source", "cik",
        "price_symbol", "price_date", "price_currency", "spot_fx_to_usd",
        "earnings_start", "earnings_end", "earnings_available_date", "earnings_tag",
        "earnings_accessions", "earnings_method", "shares_date",
        "shares_available_date", "shares_tag", "shares_accession",
        "shares_aggregation", "shares_component_count", "filed_shares",
        "trailing_earnings_usd",
    ]
    candidate = build.fundamentals.loc[:, final_columns].sort_values(
        ["ticker", "available_date"]
    )
    temporary_fundamentals = fundamentals_path.with_suffix(".csv.tmp")
    temporary_fundamentals.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(temporary_fundamentals, index=False)
    load_fundamentals(temporary_fundamentals)
    temporary_fundamentals.replace(fundamentals_path)

    write_csv_outputs_atomically(
        {
            output_dir / "source_coverage.csv": (build.coverage, {"index": False}),
            output_dir / "source_errors.csv": (build.errors, {"index": False}),
        }
    )
    manifest = build_source_manifest(build, fundamentals_path, cache_dir)
    write_json_atomically(manifest, output_dir / "source_manifest.json")
    return build
```

Implement `build_source_clients(cache_dir: Path, sec_user_agent: str, refresh: bool) -> tuple[SecCompanyFactsClient, MarketReferenceData]`, `build_source_manifest(...) -> dict[str, object]`, and `write_json_atomically(payload: dict[str, object], target: Path) -> None` in the same CLI module with the exact source dates and hashing rules below.

Use `write_csv_outputs_atomically` for all audit CSVs. Validate the final candidate by writing to a sibling `.tmp`, calling `load_fundamentals` on it, and replacing the target only after validation passes. Write JSON to `source_manifest.json.tmp` and replace atomically.

The manifest must include UTC creation time, Git commit, SEC API URL, SEC user-agent project name with the email redacted to its SHA-256 hash, Yahoo symbols, FX symbols, issuer configs, methodology version `sec-yfinance-v1`, rebalance years, row counts, coverage values, cache file paths, byte sizes, and SHA-256 hashes.

- [ ] **Step 4: Implement CLI flags without a backtest import**

Support exactly:

```text
--fundamentals PATH
--output-dir PATH
--cache-dir PATH
--refresh
```

Read the user agent only from `SEC_USER_AGENT`; fail before network access when absent or lacking `@`. A module-level import scan test must assert `run_selection_experiment` and `backtest_engine` are absent from `prepare_free_fundamentals.py`.

- [ ] **Step 5: Protect generated inputs and caches**

Append to `.gitignore`:

```gitignore
# Point-in-time source caches and locally generated experiment inputs
data/source_cache/
data/fundamentals_point_in_time.csv
outputs/selection_experiment/
```

Add:

```python
def test_generated_source_paths_are_gitignored():
    paths = [
        "data/source_cache/CIK0000000001.json",
        "data/fundamentals_point_in_time.csv",
        "outputs/selection_experiment/source_manifest.json",
    ]
    completed = subprocess.run(
        ["git", "check-ignore", *paths],
        check=True,
        capture_output=True,
        text=True,
    )
    assert set(completed.stdout.splitlines()) == set(paths)
```

- [ ] **Step 6: Run checks and commit**

Run:

```powershell
python -m pytest tests\test_prepare_free_fundamentals.py -v
python -m pytest -q
python -m ruff check .
git diff --check
```

Expected: all tests pass, no established output changes, Ruff is clean, and diff check is empty.

Commit:

```powershell
git add .gitignore prepare_free_fundamentals.py tests\test_prepare_free_fundamentals.py
git commit -m "feat: prepare free fundamental source artifacts"
```

---

### Task 8: Execute the Automated Source Probe and Stop at the Coverage Gate

**Files:**
- Generate, do not commit: `data/source_cache/**`
- Generate, do not commit: `data/fundamentals_point_in_time.csv`
- Generate, do not commit: `outputs/selection_experiment/source_coverage.csv`
- Generate, do not commit: `outputs/selection_experiment/source_errors.csv`
- Generate, do not commit: `outputs/selection_experiment/source_manifest.json`
- Modify: `docs/fundamental_source_feasibility.md`

**Interfaces:**
- Consumes: completed preparation CLI and approved SEC/yfinance source choice.
- Produces: a concrete six-rebalance coverage decision; does not produce portfolio returns.

- [ ] **Step 1: Run all checks before network acquisition**

Run:

```powershell
cd D:\sin_stoks-strategy-lab
python -m pytest -q
python -m ruff check .
git diff --check
if (-not $env:SEC_USER_AGENT -or $env:SEC_USER_AGENT -notmatch '@') { throw 'Set SEC_USER_AGENT explicitly, for example: sin_stoks research your-email@example.com' }
```

Expected: tests and lint pass, diff check prints nothing, and `SEC_USER_AGENT` contains the configured email.

- [ ] **Step 2: Run the frozen source preparation command**

Run:

```powershell
python prepare_free_fundamentals.py `
  --fundamentals data\fundamentals_point_in_time.csv `
  --output-dir outputs\selection_experiment `
  --cache-dir data\source_cache
```

Expected: exactly 25 SEC CIK files plus required Yahoo/FX caches are created or reused, no backtest directory is created, and the command exits successfully even when coverage fails so the error report remains inspectable.

- [ ] **Step 3: Print and independently verify coverage**

Run:

```powershell
python -c "import pandas as pd; x=pd.read_csv('outputs/selection_experiment/source_coverage.csv'); print(x.to_string(index=False)); assert set(x['eligible_assets'].astype(int)) <= set(range(31)); assert len(x)==6"
python -c "from config import FUNDAMENTALS_PATH; from fundamental_data import load_fundamentals; x=load_fundamentals(FUNDAMENTALS_PATH); print(x.groupby('ticker').size().sort_index().to_string()); print(x['available_date'].min(), x['available_date'].max())"
```

Expected: six rows for 2020-2025 and a validated candidate dataset. The source gate passes only if every row has `eligible_assets >= 24` and `coverage >= 0.80`.

- [ ] **Step 4: Audit chronology and source provenance**

Run:

```powershell
python -c "import pandas as pd; f=pd.read_csv('data/fundamentals_point_in_time.csv',parse_dates=['available_date']); c=pd.read_csv('outputs/selection_experiment/source_coverage.csv',parse_dates=['rebalance_date']); assert f['source'].str.contains('SEC CIK').all(); assert f['source'].str.contains('Yahoo').all(); print(c[['rebalance_date','eligible_assets','coverage','missing_tickers','failed_tickers']].to_string(index=False))"
```

Print the complete error set:

```powershell
python -c "import pandas as pd; x=pd.read_csv('outputs/selection_experiment/source_errors.csv'); print('no extraction errors' if x.empty else x.to_string(index=False))"
```

Reject any ticker whose share aggregation, currency, or fact tag cannot be reconciled. Do not convert an extraction failure into a record.

- [ ] **Step 5: Apply the source decision matrix**

- If any rebalance has fewer than 24 eligible tickers, classify the automated source as `data-infeasible`, retain the code and audit files locally, and do not run a historical experiment.
- If all rebalances have at least 24 eligible tickers but a share/currency reconciliation is unresolved, classify it as `data-infeasible` and do not run.
- If all six rebalances pass coverage and reconciliation, classify the candidate as `source-feasible` and proceed only to the user approval request in Step 7.

- [ ] **Step 6: Record actual source evidence and commit documentation only**

Append the exact coverage table, successful ticker set, failure reasons, methodology version, manifest hash, Yahoo/SEC limitations, and `source-feasible` or `data-infeasible` classification to `docs/fundamental_source_feasibility.md`.

Run:

```powershell
python -m ruff check .
git diff --check
git status --short
git add docs\fundamental_source_feasibility.md
git commit -m "docs: record SEC yfinance source coverage"
```

Expected: generated data and caches remain ignored; only the documentation file is committed.

- [ ] **Step 7: Present the coverage gate and wait for explicit historical-run approval**

Present to the user:

- the six-row coverage table;
- every included and excluded ticker;
- every calculation or reconciliation warning;
- the source-manifest SHA-256;
- the exact command below;
- confirmation that no parameter search or manual fill will run.

Proposed command, not executed in this step:

```powershell
python run_selection_experiment.py --fundamentals data\fundamentals_point_in_time.csv --output-dir outputs\selection_experiment --with-celh-robustness
```

Stop and wait for explicit approval.

---

### Task 9: Run and Audit the Historical Experiment After Approval

**Files:**
- Consume local: `data/fundamentals_point_in_time.csv`
- Generate, do not commit: `outputs/selection_experiment/full/**`
- Generate, do not commit: `outputs/selection_experiment/ex_celh/**`
- Generate, do not commit: `outputs/selection_experiment/*.csv`, `*.json`, `*.md`, and `*.png`
- Modify: `docs/selection_strategy_spec.md` only to append run identifiers and hashes

**Interfaces:**
- Consumes: a source-feasible candidate and the user's explicit approval from Task 8.
- Produces: full/ex-CELH returns, the combined existing/new metric table, diagnostics, graphs, and immutable gate decisions.

- [ ] **Step 1: Re-run quality and input checks**

Run:

```powershell
python -m pytest -q
python -m ruff check .
git diff --check
python -c "from config import FUNDAMENTALS_PATH; from fundamental_data import load_fundamentals; x=load_fundamentals(FUNDAMENTALS_PATH); assert x['ticker'].nunique() >= 24; print(x.groupby('ticker').size().describe())"
```

Expected: all checks pass and at least 24 unique tickers validate.

- [ ] **Step 2: Execute the approved frozen full and CELH runs**

Run:

```powershell
python run_selection_experiment.py --fundamentals data\fundamentals_point_in_time.csv --output-dir outputs\selection_experiment --with-celh-robustness
```

Expected: `full` and `ex_celh` artifacts contain full-universe Equal Weight, Eligible Universe Equal Weight, Partitioning Selection, Density Selection, and SPY without modifying `outputs/portfolio_backtest` or `outputs/report`. The eligible baseline must use exactly the names with valid point-in-time source records at each rebalance.

- [ ] **Step 3: Run exact integrity assertions**

Run:

```powershell
python -c "import pandas as pd; w=pd.read_csv('outputs/selection_experiment/full/walk_forward_weights.csv',index_col=[0,1]); s=w.loc[w.index.get_level_values(1).isin(['Partitioning Selection','Density Selection'])]; assert (s.gt(0).sum(axis=1)==12).all(); assert (s.max(axis=1)<=0.25+1e-12).all(); a=pd.read_csv('outputs/selection_experiment/full/selection_audit.csv',parse_dates=['rebalance_date','available_date']); assert (a['available_date']<a['rebalance_date']).all(); assert set(a.groupby(['rebalance_date','strategy'])['selected'].sum())=={12}; print('integrity checks passed')"
```

Run the remaining assertions:

```powershell
python -c "import numpy as np,pandas as pd; t=pd.read_csv('outputs/selection_experiment/full/turnover.csv'); assert np.allclose(t['Cost'],t['Turnover']*10/10_000); w=pd.read_csv('outputs/selection_experiment/full/walk_forward_weights.csv',index_col=[0,1]); assert set(pd.to_datetime(w.index.get_level_values(0)).year)==set(range(2020,2026)); r=pd.read_csv('outputs/selection_experiment/full/walk_forward_returns.csv',index_col=0); assert not r['SPY'].isna().any(); ew=pd.read_csv('outputs/selection_experiment/ex_celh/walk_forward_weights.csv',index_col=[0,1]); ea=pd.read_csv('outputs/selection_experiment/ex_celh/selection_audit.csv'); assert 'CELH' not in ew.columns and 'CELH' not in set(ea['ticker']); print('cost, year, SPY, and ex-CELH checks passed')"
```

- [ ] **Step 4: Generate metrics and requested graphs**

Run:

```powershell
python report_selection_experiment.py --input-dir outputs\selection_experiment --quality-checks-passed
```

Expected output includes:

- `all_strategy_summary.csv` containing all seven existing strategies, PAM, HDBSCAN, and SPY;
- `comparison_summary.csv`, `annual_excess_returns.csv`, `decision.json`, and `experiment_report.md`;
- `selection_equity_curves.png`, `selection_drawdowns.png`, `annual_strategy_returns.png`, and `selection_frequency.png`;
- `partitioning_selection_clusters.png` and `density_selection_clusters.png`, with annual cluster labels and selected stocks highlighted; and
- `strategy_benchmark_equity_curves.png`, `strategy_benchmark_drawdowns.png`, and `strategy_benchmark_annual_returns.png`, comparing PAM and HDBSCAN with SPY, Max Sharpe, Equal Weight, and Maximum Diversification.

- [ ] **Step 5: Verify reproducibility**

Save hashes for all deterministic full/ex-CELH CSV artifacts:

```powershell
python -c "from pathlib import Path; import hashlib,json; files=sorted(Path('outputs/selection_experiment').glob('full/*.csv'))+sorted(Path('outputs/selection_experiment').glob('ex_celh/*.csv')); Path('outputs/selection_experiment/first_run_hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},sort_keys=True,indent=2),encoding='utf-8')"
```

Rerun Steps 2 and 4, then compare:

```powershell
python -c "from pathlib import Path; import hashlib,json; expected=json.loads(Path('outputs/selection_experiment/first_run_hashes.json').read_text()); actual={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in expected}; assert actual==expected, {'expected':expected,'actual':actual}; print('reproducibility hashes match')"
```

Ignore UTC generation timestamps and report-generation timestamps because neither is part of the compared CSV set.

- [ ] **Step 6: Append immutable run identifiers**

Append source manifest hash, full/ex-CELH metadata hashes, Git commit, exact commands, and decision classifications to `docs/selection_strategy_spec.md` without changing methods or gates.

Commit only the spec update:

```powershell
git add docs\selection_strategy_spec.md
git commit -m "docs: record free-source selection experiment run"
```

---

### Task 10: Present Results and Decide Whether Manual Completion Merits a Separate Plan

**Files:**
- Read: `outputs/selection_experiment/all_strategy_summary.csv`
- Read: `outputs/selection_experiment/decision.json`
- Read: `outputs/selection_experiment/experiment_report.md`
- Preserve: all branch code and frozen specifications

**Interfaces:**
- Consumes: reproducible historical output and preregistered decisions.
- Produces: a user-facing result review and either a stopped negative experiment or approval to plan manual completion for the five uncovered tickers.

- [ ] **Step 1: Review the bounded branch diff**

Run:

```powershell
git diff --stat main...codex/partition-density-strategies
git diff --check main...codex/partition-density-strategies
git log --oneline main..codex/partition-density-strategies
git status --short
```

Expected: only planned code, tests, source specifications, and permitted run-hash documentation appear; worktree status is empty.

- [ ] **Step 2: Present the combined metric table and graphs**

Show every row from `all_strategy_summary.csv` with final value, total return, CAGR, volatility, Sharpe, Sortino, maximum drawdown, Calmar, beta, alpha, tracking error, information ratio, and recurring turnover. Label PAM and HDBSCAN as historical walk-forward results and include full/ex-CELH gate outcomes.

Link or identify all generated graph paths, including both annual cluster maps and the three requested benchmark-comparison figures. Explain cluster membership, selected representatives, concentration, annual consistency, drawdowns, and turnover without claiming statistical significance from six years.

- [ ] **Step 3: Apply the manual-completion decision rule**

- Decide whether manual completion is worthwhile from source quality, coverage, effort, and the value of removing eligibility bias—not from whether the observed strategy return is attractive.
- If manual collection is justified, use a separate dated plan for PRNDY, TCEHY, NTDOY, CCOEY, and UBSFY and report the completed-universe run as a new experiment, not as a repair of an unfavorable result.
- Do not change the current model parameters, source records, gates, or automated ticker set after seeing results.

- [ ] **Step 4: Keep the branch unmerged until explicit approval**

Do not merge, push, remove the worktree, update main-branch `README.md`, or delete generated local evidence. Ask separately for promotion/merge approval only after the result and source limitations are reviewed.

---

## Final Execution Checklist

- [ ] The source methodology, issuer registry, units, local listings, and share aggregation are frozen before acquisition.
- [ ] SEC filing and Yahoo observation dates are strictly before each rebalance.
- [ ] TTM earnings never mix a current YTD value with a prior comparison from a different filing accession.
- [ ] Market cap and earnings are converted to USD with historical, not current, FX.
- [ ] The five manual-only tickers remain missing and no value is imputed.
- [ ] Every rebalance independently has at least 24 eligible tickers before a historical run is proposed.
- [ ] Source caches, generated fundamentals, contact details, and raw data remain uncommitted.
- [ ] The preparation CLI cannot invoke the backtest.
- [ ] Historical execution receives explicit approval after the actual coverage table is presented.
- [ ] Full and ex-CELH artifacts are reproducible and established output directories remain unchanged.
- [ ] All tests, Ruff, and diff checks pass.
- [ ] Manual completion, merge, and worktree cleanup each require separate approval.
