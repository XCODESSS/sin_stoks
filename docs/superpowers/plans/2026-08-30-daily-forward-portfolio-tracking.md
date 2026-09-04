# Daily Forward Portfolio Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an auditable Windows-based daily paper-portfolio tracker for Partitioning Selection and Density Selection, with fair baselines, immutable rebalance records, fail-closed market-data handling, and a clearly separated untouched-forward score.

**Architecture:** Keep selection and rebalancing separate from daily valuation. A freeze command creates an immutable rebalance epoch from point-in-time fundamentals and the prior 104 completed weekly returns; a daily command then fetches adjusted closes, values unchanged holdings, appends a hash-chained ledger, and regenerates a local report. Windows Task Scheduler invokes only the idempotent daily command; it never changes strategy parameters, universe membership, or target weights.

**Tech Stack:** Python 3.10+, pandas, NumPy, scikit-learn, yfinance, matplotlib, pytest, Ruff, PowerShell, Windows Task Scheduler.

**Spec:** No standalone live-tracking specification exists yet. This plan implements the user's 2026-08-30 request, preserves `docs/selection_strategy_spec.md`, and creates `docs/live_forward_tracking_spec.md` in Task 1 before implementation code is added.

## Global Constraints

- Work in a new branch and isolated worktree based on `codex/partition-density-strategies`; do not merge without explicit user approval.
- Preserve the user's existing modification to `docs/stock_selection_strategies_simple_explanation.md`; do not discard, overwrite, or reformat unrelated content.
- Do not modify or regenerate `outputs/portfolio_backtest/`, `outputs/report/`, or `outputs/selection_experiment/`.
- Freeze and track these five series: `Partitioning Selection`, `Density Selection`, `Eligible Universe Equal Weight`, `Equal Weight`, and `SPY`.
- Preserve the frozen selector parameters: 12 selected equities, six PAM partitions, 104 completed weeks, HDBSCAN `min_cluster_size=3`, `min_samples=3`, 50/50 feature/correlation distance, 0.25 diversification penalty, 4% risk-free rate, 25% maximum position, and 10 bps one-way transaction costs.
- Daily tracking marks holdings to market; it must not rerun PAM, HDBSCAN, fundamentals, or weights between rebalance epochs.
- Use adjusted daily closes (`auto_adjust=True`) for total-return valuation. Continue using unadjusted point-in-time prices only inside the approved market-cap reconstruction pipeline.
- Use SPY's latest returned adjusted-close date as the authoritative completed market session. A weekend or exchange holiday with no new SPY bar is a successful no-op.
- Never forward-fill, backward-fill, or zero-fill SPY or a held asset. Missing or non-finite data for any held ticker stops the entire session before the ledger changes.
- Preserve delisted, acquired, or renamed holdings economically. A missing ticker requires a separately reviewed corporate-action correction record; it must not be silently replaced or dropped.
- Every fundamental observation used for selection must satisfy `available_date < selection_as_of`. Every return used for selection must be dated before `selection_as_of`.
- A rebalance freeze must retain the Git commit, universe hash, configuration, input hashes, selected tickers, weights, feature/cluster audit, coverage, creation timestamp, and effective-session rule.
- Rebalance directories are immutable. If a correction is necessary, create a superseding rebalance ID that points to the prior ID; never overwrite the original.
- Keep immediate launch observations in `mode=shadow`. Only a rebalance frozen before its first executable market session can use `mode=scored`; reports must never combine shadow and scored performance.
- The formal annual score starts at the first January rebalance frozen before entry. A mid-year launch may be tracked immediately as a shadow launch epoch but cannot count as one of the annual validation years.
- Default notional is USD 10,000 per strategy. The notional is configurable only when creating the first epoch and is immutable afterward.
- Charge initial turnover of 1.0 to every investable portfolio and recurring one-way turnover from drifted pre-trade weights. Preserve SPY's existing zero-cost benchmark convention and label it explicitly.
- Store all timestamps in UTC and display both UTC and `Asia/Calcutta` in status/report output.
- Do not store `SEC_USER_AGENT`, `SIMFIN_API_KEY`, passwords, or Task Scheduler credentials in source, manifests, command output, or logs.
- Scheduler registration is an external machine change and requires explicit approval immediately before `Register-ScheduledTask` is executed.
- The scheduled task defaults to an interactive user token so network access works; document that it runs only while that Windows user is logged in.
- Generated live data and reports remain local and ignored by Git. Only code, tests, protocol documentation, and PowerShell scripts are committed.

---

## File Map

| Path | Change | Responsibility |
| --- | --- | --- |
| `docs/live_forward_tracking_spec.md` | Create | Frozen operating contract, evidence modes, accounting, data rules, and recovery procedure. |
| `README.md` | Modify | Link the live tracker and describe the random stock-selection procedure accurately. |
| `hypothesis.md` | Modify | Replace the unsupported retrospective-stock-selection claim while retaining conditional survivorship caveats. |
| `docs/selection_strategy_spec.md` | Modify | Link the live-forward protocol without altering historical results or frozen parameters. |
| `.gitignore` | Modify | Ignore local live price snapshots, ledgers, reports, locks, and logs. |
| `config.py` | Modify | Add live-tracking paths and immutable defaults without changing historical constants. |
| `live_tracking.py` | Create | Manifest/weight schemas, validation, canonical hashing, atomic JSON/CSV helpers. |
| `live_price_data.py` | Create | Adjusted-close download normalization, completed-session validation, weekly-return construction, immutable raw snapshots. |
| `free_fundamental_builder.py` | Modify | Accept explicit future selection dates while preserving existing 2020–2025 defaults. |
| `prepare_free_fundamentals.py` | Modify | Accept explicit selection dates and market end dates; isolate live source outputs from historical outputs. |
| `freeze_live_portfolio.py` | Create | Build a single point-in-time selector context and write an immutable rebalance epoch. |
| `live_portfolio_accounting.py` | Create | Drift, turnover, cost, epoch valuation, ledger rows, drawdown, and hash-chain calculations. |
| `track_live_portfolio.py` | Create | Idempotent daily orchestration, lock handling, status output, and exit codes. |
| `report_live_portfolio.py` | Create | Daily summary CSV, Markdown report, equity/drawdown graph, and health assessment. |
| `scripts/run_daily_tracking.ps1` | Create | Stable scheduled-task entrypoint using the repository virtual environment. |
| `scripts/install_daily_tracking_task.ps1` | Create | Validate prerequisites and optionally register the Windows task after approval. |
| `tests/test_live_tracking.py` | Create | Manifest, weight, immutable-write, and hash validation. |
| `tests/test_live_price_data.py` | Create | Yahoo normalization, session, missing-data, and weekly-cutoff tests. |
| `tests/test_freeze_live_portfolio.py` | Create | No-look-ahead, selector reuse, five-series weights, and no-overwrite tests. |
| `tests/test_live_portfolio_accounting.py` | Create | Valuation, drift, turnover, cost, drawdown, and hash-chain tests. |
| `tests/test_track_live_portfolio.py` | Create | Idempotency, no-op, lock, atomic failure, and recovery tests. |
| `tests/test_report_live_portfolio.py` | Create | Preliminary labels, mode separation, staleness, and report output tests. |
| `tests/test_scheduler_scripts.py` | Create | PowerShell parseability and required safe task settings. |

---

### Task 1: Freeze the Live-Forward Contract and Correct Universe Wording

**Files:**
- Create: `docs/live_forward_tracking_spec.md`
- Modify: `README.md:8-11,107-112`
- Modify: `hypothesis.md:8-16,223-251`
- Modify: `docs/selection_strategy_spec.md:20-28,195-208`

**Interfaces:**
- Consumes: The existing frozen selection contract in `docs/selection_strategy_spec.md` and the user's clarification that categories were chosen deliberately while stocks were randomly selected without reference to subsequent returns.
- Produces: A normative protocol used by every later task; no runtime interface.

- [ ] **Step 1: Write the live protocol before code**

Create `docs/live_forward_tracking_spec.md` with these exact normative sections:

```markdown
# Live Forward Tracking Specification

## Universe provenance
The six thematic categories were chosen deliberately. Individual stocks were randomly selected from the category candidate pools without using subsequent returns. The frozen 30-stock list is held constant for this validation; removals, acquisitions, and delistings are economic events, not permission to revise history.

## Evidence modes
- `shadow`: operational validation, including any mid-year launch epoch; excluded from formal annual gate counts.
- `scored`: created before its first executable market session and never backfilled; eligible for forward-validation reporting.

## Rebalance rule
PAM and HDBSCAN run only at a frozen epoch. The standard scored schedule uses January 1 as `selection_as_of` and enters at the first completed US market session afterward. Daily runs only value holdings.

## Comparators
Track Partitioning Selection, Density Selection, Eligible Universe Equal Weight, Equal Weight, and SPY on aligned sessions.

## Data failure rule
Missing SPY or held-asset adjusted closes stop the complete daily update. No forward-fill, backward-fill, zero-fill, ticker substitution, or silent exclusion is permitted.
```

Continue the file with the remaining Global Constraints: point-in-time cutoff, adjusted-close valuation, 10-bps cost, rebalance immutability, corporate-action correction records, UTC/IST timestamps, credential handling, artifact paths, and scheduler behavior.

- [ ] **Step 2: Correct the unsupported retrospective-stock wording**

Replace claims that stocks were selected retrospectively with this bounded language:

```markdown
The thematic categories were defined deliberately, while individual stocks were randomly sampled without reference to subsequent returns. This removes return-based stock cherry-picking. The strength of any survivorship claim still depends on how each category candidate pool was assembled and is therefore reported as a conditional limitation unless a dated candidate-pool record is available.
```

Do not change the recorded historical metrics, CELH sensitivity, six-period limitation, missing five-ticker coverage, or investment-advice disclaimer.

- [ ] **Step 3: Link the historical specification to the forward protocol**

Add a short `Live-forward continuation` paragraph to `docs/selection_strategy_spec.md` stating that historical promotion gates remain unchanged and live results reside only under `outputs/live_tracking/`.

- [ ] **Step 4: Verify the documentation is internally consistent**

Run:

```powershell
rg -n -i "retrospective universe|selected retrospectively|survivorship bias|randomly selected|live forward" README.md hypothesis.md docs/selection_strategy_spec.md docs/live_forward_tracking_spec.md
```

Expected: no unconditional claim that the individual stocks were retrospectively selected; random selection and the conditional candidate-pool caveat appear consistently.

- [ ] **Step 5: Commit the protocol separately**

```powershell
git add README.md hypothesis.md docs/selection_strategy_spec.md docs/live_forward_tracking_spec.md
git commit -m "docs: freeze daily forward tracking protocol"
```

Expected: one documentation-only commit; the pre-existing user edit in `docs/stock_selection_strategies_simple_explanation.md` is not staged unless the user has already committed it separately.

---

### Task 2: Add Live Paths, Immutable Schemas, and Hash Validation

**Files:**
- Modify: `config.py:5-20,24-48`
- Modify: `.gitignore`
- Create: `live_tracking.py`
- Create: `tests/test_live_tracking.py`

**Interfaces:**
- Consumes: `BASE_DIR`, `STARTING_VALUE`, `DEFAULT_TRANSACTION_COST_BPS`, `PORTFOLIO_TICKERS`, and `BENCHMARK_TICKER`.
- Produces: `TrackingConfig`, `TrackingPaths`, `RebalanceManifest`, `load_manifest(path: Path) -> RebalanceManifest`, `validate_target_weights(frame: pd.DataFrame) -> None`, `sha256_file(path: Path) -> str`, `canonical_record_hash(previous_hash: str, record: Mapping[str, object]) -> str`, `write_json_atomically(payload: Mapping[str, object], path: Path, immutable: bool = False) -> None`, and `write_csv_atomically(frame: pd.DataFrame, path: Path, immutable: bool = False) -> None`.

- [ ] **Step 1: Write failing schema tests**

Create `tests/test_live_tracking.py` with focused tests including:

```python
def test_target_weights_require_exact_strategy_sums():
    weights = pd.DataFrame(
        {
            "strategy": ["Density Selection", "Density Selection"],
            "ticker": ["A", "B"],
            "target_weight": [0.6, 0.3],
        }
    )
    with pytest.raises(ValueError, match="sum to 1"):
        validate_target_weights(weights)


def test_canonical_hash_changes_when_value_changes():
    first = canonical_record_hash("0" * 64, {"session_date": "2027-01-04", "net_value": 10000.0})
    changed = canonical_record_hash("0" * 64, {"session_date": "2027-01-04", "net_value": 10001.0})
    assert first != changed


def test_atomic_json_refuses_to_overwrite_immutable_manifest(tmp_path):
    target = tmp_path / "manifest.json"
    write_json_atomically({"rebalance_id": "2027-01-01-scored"}, target, immutable=True)
    with pytest.raises(FileExistsError):
        write_json_atomically({"rebalance_id": "changed"}, target, immutable=True)
```

- [ ] **Step 2: Run the focused tests and confirm the missing module failure**

Run:

```powershell
python -m pytest tests/test_live_tracking.py -q
```

Expected: collection fails because `live_tracking` does not exist.

- [ ] **Step 3: Add live configuration constants**

Add these paths and defaults without modifying historical date constants:

```python
LIVE_DATA_DIR = DATA_DIR / "live_tracking"
LIVE_OUTPUT_DIR = OUTPUT_DIR / "live_tracking"
LIVE_REBALANCE_DIR = LIVE_DATA_DIR / "rebalances"
LIVE_PRICE_SNAPSHOT_DIR = LIVE_DATA_DIR / "price_snapshots"
LIVE_LEDGER_PATH = LIVE_DATA_DIR / "daily_ledger.csv"
LIVE_STATUS_PATH = LIVE_OUTPUT_DIR / "latest_status.json"
LIVE_STARTING_VALUE = STARTING_VALUE
LIVE_TIMEZONE = "Asia/Calcutta"
LIVE_TRACKED_STRATEGIES = (
    "Partitioning Selection",
    "Density Selection",
    "Eligible Universe Equal Weight",
    "Equal Weight",
    "SPY",
)
```

Add `/data/live_tracking/`, `/outputs/live_tracking/`, and `/logs/live_tracking/` to `.gitignore`.

- [ ] **Step 4: Implement the immutable schema module**

Use frozen dataclasses and explicit validation:

```python
@dataclass(frozen=True)
class TrackingConfig:
    starting_value: float = LIVE_STARTING_VALUE
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS
    timezone: str = LIVE_TIMEZONE


@dataclass(frozen=True)
class TrackingPaths:
    data_dir: Path = LIVE_DATA_DIR
    output_dir: Path = LIVE_OUTPUT_DIR
    rebalance_dir: Path = LIVE_REBALANCE_DIR
    price_snapshot_dir: Path = LIVE_PRICE_SNAPSHOT_DIR
    ledger_path: Path = LIVE_LEDGER_PATH
    status_path: Path = LIVE_STATUS_PATH


@dataclass(frozen=True)
class RebalanceManifest:
    schema_version: int
    rebalance_id: str
    mode: Literal["shadow", "scored"]
    selection_as_of: str
    freeze_timestamp_utc: str
    effective_after: str
    git_commit: str
    predecessor_rebalance_id: str | None
    input_hashes: dict[str, str]
```

`validate_target_weights` must require the columns `strategy`, `ticker`, and `target_weight`; finite non-negative values; exactly the configured strategies; weight sum 1.0 within `1e-10`; exactly 12 positive positions for each selector; and no weight above 0.25. `SPY` must contain one 100% position.

Canonical hashes must serialize `{"previous_record_hash": previous_hash, **record}` with `json.dumps(record_with_previous_hash, sort_keys=True, separators=(",", ":"), allow_nan=False)` and SHA-256.

- [ ] **Step 5: Run tests and lint**

```powershell
python -m pytest tests/test_live_tracking.py -q
python -m ruff check config.py live_tracking.py tests/test_live_tracking.py
```

Expected: all focused tests pass and Ruff reports no violations.

- [ ] **Step 6: Commit the schema boundary**

```powershell
git add .gitignore config.py live_tracking.py tests/test_live_tracking.py
git commit -m "feat: add immutable live tracking schemas"
```

---

### Task 3: Add Fail-Closed Daily Adjusted-Close Ingestion

**Files:**
- Create: `live_price_data.py`
- Create: `tests/test_live_price_data.py`

**Interfaces:**
- Consumes: a ticker sequence, inclusive start date, inclusive as-of date, and a `yfinance.download`-compatible callable.
- Produces: `AdjustedCloseSource` protocol, `YahooAdjustedCloseSource`, `download_adjusted_closes(tickers: Sequence[str], start: pd.Timestamp, as_of: pd.Timestamp) -> pd.DataFrame`, `authoritative_session(prices: pd.DataFrame) -> pd.Timestamp`, `require_complete_session(prices: pd.DataFrame, session: pd.Timestamp, tickers: Sequence[str]) -> pd.Series`, `completed_weekly_log_returns(prices: pd.DataFrame, selection_as_of: pd.Timestamp, weeks: int = 104) -> pd.DataFrame`, and `write_price_snapshot(prices: pd.DataFrame, session: pd.Timestamp, directory: Path) -> Path`.

- [ ] **Step 1: Write failing price-contract tests**

Include deterministic fixtures for single-level and MultiIndex Yahoo frames:

```python
def test_spy_defines_authoritative_session():
    prices = pd.DataFrame(
        {"A": [10.0, 11.0], "SPY": [500.0, 501.0]},
        index=pd.to_datetime(["2027-01-04", "2027-01-05"]),
    )
    assert authoritative_session(prices) == pd.Timestamp("2027-01-05")


def test_missing_held_asset_fails_instead_of_filling():
    prices = pd.DataFrame(
        {"A": [10.0, np.nan], "SPY": [500.0, 501.0]},
        index=pd.to_datetime(["2027-01-04", "2027-01-05"]),
    )
    with pytest.raises(ValueError, match="A"):
        require_complete_session(prices, pd.Timestamp("2027-01-05"), ["A", "SPY"])


def test_weekly_features_exclude_selection_date_and_incomplete_week():
    prices = make_daily_prices_ending("2027-01-01", periods=800)
    weekly = completed_weekly_log_returns(prices, pd.Timestamp("2027-01-01"), weeks=104)
    assert len(weekly) == 104
    assert weekly.index.max() < pd.Timestamp("2027-01-01")
```

- [ ] **Step 2: Verify the focused tests fail before implementation**

```powershell
python -m pytest tests/test_live_price_data.py -q
```

Expected: import failure for `live_price_data`.

- [ ] **Step 3: Implement deterministic Yahoo normalization**

Define the injectable source interface and production implementation:

```python
class AdjustedCloseSource(Protocol):
    def fetch(
        self,
        tickers: Sequence[str],
        start: pd.Timestamp,
        as_of: pd.Timestamp,
    ) -> pd.DataFrame: ...


class YahooAdjustedCloseSource:
    def fetch(
        self,
        tickers: Sequence[str],
        start: pd.Timestamp,
        as_of: pd.Timestamp,
    ) -> pd.DataFrame:
        return download_adjusted_closes(tickers, start, as_of)
```

The download boundary must use:

```python
downloaded = yf.download(
    sorted(set(tickers)),
    start=start.date().isoformat(),
    end=(as_of + pd.Timedelta(days=2)).date().isoformat(),
    interval="1d",
    auto_adjust=True,
    actions=False,
    progress=False,
    group_by="column",
    threads=False,
)
```

Normalize `Close` into lexicographically sorted ticker columns, normalize the index to timezone-naive dates, reject duplicate dates, and reject non-finite or non-positive values only when a ticker is required on the authoritative session. Do not call `fillna`.

- [ ] **Step 4: Implement immutable raw snapshots**

Write `data/live_tracking/price_snapshots/YYYY-MM-DD.csv` using atomic replacement only when the target does not exist. If it exists, compare SHA-256: identical content is an idempotent success; different content raises `ValueError("Price snapshot revision requires a correction record")`.

- [ ] **Step 5: Run focused tests and lint**

```powershell
python -m pytest tests/test_live_price_data.py -q
python -m ruff check live_price_data.py tests/test_live_price_data.py
```

Expected: all tests pass; no path writes occur outside pytest temporary directories.

- [ ] **Step 6: Commit the market-data boundary**

```powershell
git add live_price_data.py tests/test_live_price_data.py
git commit -m "feat: add fail-closed daily price ingestion"
```

---

### Task 4: Parameterize Future Fundamentals and Freeze One Rebalance Epoch

**Files:**
- Modify: `free_fundamental_builder.py:162-214`
- Modify: `prepare_free_fundamentals.py:26-28,79-230,233-252`
- Create: `freeze_live_portfolio.py`
- Modify: `tests/test_free_fundamental_builder.py`
- Modify: `tests/test_prepare_free_fundamentals.py`
- Create: `tests/test_freeze_live_portfolio.py`

**Interfaces:**
- Consumes: `load_fundamentals`, `fundamentals_as_of`, `build_selection_features`, `select_partitioned`, `select_density`, adjusted daily prices, explicit `selection_as_of`, and immutable live paths.
- Produces: `normalize_selection_dates(selection_dates: Sequence[pd.Timestamp]) -> tuple[pd.Timestamp, ...]`, the extended `prepare_free_fundamentals` signature shown in Step 3, `build_single_rebalance(selection_as_of: pd.Timestamp, daily_prices: pd.DataFrame, fundamentals: pd.DataFrame, mode: Literal["shadow", "scored"], freeze_timestamp: pd.Timestamp, effective_after: pd.Timestamp, predecessor_rebalance_id: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]`, and CLI `freeze_live_portfolio.py`.

- [ ] **Step 1: Add failing future-date builder tests**

Extend existing tests to prove historical defaults remain identical and explicit dates work:

```python
def test_builder_accepts_explicit_selection_dates(fake_clients):
    build = build_free_fundamentals(
        *fake_clients,
        selection_dates=(pd.Timestamp("2027-01-01"),),
    )
    assert set(pd.to_datetime(build.coverage["rebalance_date"])) == {pd.Timestamp("2027-01-01")}


def test_builder_rejects_duplicate_or_unsorted_selection_dates(fake_clients):
    with pytest.raises(ValueError, match="strictly increasing"):
        build_free_fundamentals(
            *fake_clients,
            selection_dates=(pd.Timestamp("2027-01-01"), pd.Timestamp("2027-01-01")),
        )
```

- [ ] **Step 2: Parameterize without changing historical defaults**

Change the builder signature to:

```python
def build_free_fundamentals(
    sec_client: SecCompanyFactsClient,
    market_data: MarketReferenceData,
    simfin_data: SimfinReferenceData | None = None,
    rebalance_years: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024, 2025),
    selection_dates: tuple[pd.Timestamp, ...] | None = None,
) -> FreeFundamentalBuild:
```

When `selection_dates is None`, derive January 1 dates from `rebalance_years` exactly as today. Otherwise require normalized, unique, strictly increasing dates and iterate those dates. Existing historical tests and hashes must remain unaffected unless the historical pipeline is explicitly rerun, which this implementation must not do.

- [ ] **Step 3: Isolate future source preparation**

Add repeatable CLI arguments to `prepare_free_fundamentals.py`:

```python
parser.add_argument("--selection-date", action="append", type=pd.Timestamp)
parser.add_argument("--market-end", type=pd.Timestamp)
```

Extend the internal signatures without changing their defaults:

```python
def build_source_clients(
    cache_dir: Path,
    sec_user_agent: str,
    simfin_api_key: str,
    refresh: bool = False,
    market_start: pd.Timestamp = MARKET_START,
    market_end: pd.Timestamp = MARKET_END,
) -> tuple[SecCompanyFactsClient, MarketReferenceData, SimfinReferenceData]:


def prepare_free_fundamentals(
    fundamentals_path: Path = FUNDAMENTALS_PATH,
    output_dir: Path = SELECTION_OUTPUT_DIR,
    cache_dir: Path = SOURCE_CACHE_DIR,
    sec_user_agent: str = "",
    simfin_api_key: str = "",
    refresh: bool = False,
    selection_dates: tuple[pd.Timestamp, ...] | None = None,
    market_end: pd.Timestamp = MARKET_END,
) -> FreeFundamentalBuild:
```

Pass dates through the source manifest as `selection_dates`, calculate Yahoo's exclusive end as `max(selection_dates) + 2 days` unless `--market-end` is later, and require live invocations to provide paths under `data/live_tracking/source/` and `outputs/live_tracking/source/`. Historical defaults must continue targeting the current historical locations.

- [ ] **Step 4: Write failing single-rebalance tests**

Create `tests/test_freeze_live_portfolio.py` with:

```python
def test_freeze_builds_five_weight_sets_without_future_rows(tmp_path):
    weights, audit, manifest = build_single_rebalance(
        selection_as_of=pd.Timestamp("2027-01-01"),
        daily_prices=make_prices_ending("2026-12-31"),
        fundamentals=make_fundamentals(available_before="2027-01-01"),
        mode="scored",
        freeze_timestamp=pd.Timestamp("2027-01-01T00:00:00Z"),
    )
    assert set(weights["strategy"]) == set(LIVE_TRACKED_STRATEGIES)
    assert weights.query("strategy == 'Partitioning Selection' and target_weight > 0").shape[0] == 12
    assert weights.query("strategy == 'Density Selection' and target_weight > 0").shape[0] == 12
    assert pd.to_datetime(audit["available_date"]).max() < pd.Timestamp("2027-01-01")
    assert manifest["mode"] == "scored"
```

Also test that a future-dated fundamental row is ignored, fewer than 104 completed weeks fail, any incomplete eligible return series fails, a scored mid-year date fails, and an existing rebalance directory cannot be overwritten.

- [ ] **Step 5: Implement the one-date freeze command**

Construct `RebalanceContext` directly rather than invoking the multi-year backtest:

```python
weekly_log_returns = completed_weekly_log_returns(daily_prices, selection_as_of, weeks=104)
snapshot = fundamentals_as_of(fundamentals, selection_as_of, weekly_log_returns.columns)
eligible = list(snapshot.index)
inputs = build_selection_features(weekly_log_returns.loc[:, eligible], snapshot)
partitioning = select_partitioned(inputs)
density = select_density(inputs)
```

Build equal weights for both 12-name selections, all eligible names, all 30 frozen names, and SPY. Store zero weights explicitly for non-held universe names so future turnover remains auditable. Write, in this order, to a temporary rebalance directory: `target_weights.csv`, `selection_audit.csv`, `coverage.csv`, `input_prices.csv`, and `manifest.json`; rename the directory to its final `rebalance_id` only after every hash and validation passes.

The CLI must support:

```text
python freeze_live_portfolio.py --selection-as-of 2027-01-01 --mode scored --fundamentals <path> --prices <path> --effective-after 2027-01-01
```

For an immediate operational launch, require `--mode shadow`; label it as a non-scored launch stub in the manifest.

- [ ] **Step 6: Run all affected tests**

```powershell
python -m pytest tests/test_free_fundamental_builder.py tests/test_prepare_free_fundamentals.py tests/test_freeze_live_portfolio.py -q
python -m ruff check free_fundamental_builder.py prepare_free_fundamentals.py freeze_live_portfolio.py tests/test_freeze_live_portfolio.py
```

Expected: existing historical behavior remains green; new future-date and freeze tests pass.

- [ ] **Step 7: Commit the freeze boundary**

```powershell
git add free_fundamental_builder.py prepare_free_fundamentals.py freeze_live_portfolio.py tests/test_free_fundamental_builder.py tests/test_prepare_free_fundamentals.py tests/test_freeze_live_portfolio.py
git commit -m "feat: freeze auditable live rebalance epochs"
```

---

### Task 5: Implement Daily Valuation, Drift, Costs, and the Hash-Chained Ledger

**Files:**
- Create: `live_portfolio_accounting.py`
- Create: `tests/test_live_portfolio_accounting.py`

**Interfaces:**
- Consumes: validated target weights, entry adjusted closes, current adjusted closes, prior epoch value, predecessor drifted weights, and transaction-cost basis points.
- Produces: `EpochActivation`, `one_way_turnover(old: pd.Series, new: pd.Series) -> float`, `activate_epoch(pretrade_value: float, old_weights: pd.Series, target_weights: pd.Series, transaction_cost_bps: float, benchmark: bool = False) -> EpochActivation`, `value_epoch(invested_value: float, target_weights: pd.Series, entry_prices: pd.Series, current_prices: pd.Series) -> float`, `build_session_rows(session_date: pd.Timestamp, mode: Literal["shadow", "scored"], rebalance_id: str, activations: Mapping[str, EpochActivation], weights: pd.DataFrame, entry_prices: pd.Series, current_prices: pd.Series, previous_ledger: pd.DataFrame, price_snapshot_sha256: str, created_utc: pd.Timestamp) -> pd.DataFrame`, `validate_ledger(frame: pd.DataFrame) -> None`, and `append_hashed_rows(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame`.

- [ ] **Step 1: Write failing accounting tests with hand-calculated values**

```python
def test_two_asset_epoch_values_total_return_without_rebalancing():
    weights = pd.Series({"A": 0.5, "B": 0.5})
    entry = pd.Series({"A": 100.0, "B": 200.0})
    current = pd.Series({"A": 110.0, "B": 180.0})
    assert value_epoch(9990.0, weights, entry, current) == pytest.approx(9990.0)


def test_initial_cost_is_deducted_once():
    activation = activate_epoch(
        pretrade_value=10_000.0,
        old_weights=pd.Series(dtype=float),
        target_weights=pd.Series({"A": 0.5, "B": 0.5}),
        transaction_cost_bps=10.0,
    )
    assert activation.turnover == pytest.approx(1.0)
    assert activation.cost_dollars == pytest.approx(10.0)
    assert activation.invested_value == pytest.approx(9990.0)


def test_hash_chain_detects_edited_history():
    ledger = append_hashed_rows(empty_ledger(), make_rows("2027-01-04"))
    ledger.loc[0, "net_value"] += 1.0
    with pytest.raises(ValueError, match="hash chain"):
        validate_ledger(ledger)
```

Add tests for recurring drift-aware turnover, zero SPY cost, drawdown from the prior peak, duplicate `(session_date, strategy, mode)` rejection, and disjoint shadow/scored chains.

- [ ] **Step 2: Confirm focused tests fail**

```powershell
python -m pytest tests/test_live_portfolio_accounting.py -q
```

Expected: import failure for `live_portfolio_accounting`.

- [ ] **Step 3: Implement economically explicit epoch accounting**

Use adjusted-close relatives:

```python
relative_growth = current_prices / entry_prices
gross_multiplier = float((target_weights * relative_growth).sum())
net_value = invested_value * gross_multiplier
drifted_weights = target_weights * relative_growth / gross_multiplier
```

Define the activation result explicitly:

```python
@dataclass(frozen=True)
class EpochActivation:
    turnover: float
    cost_dollars: float
    invested_value: float
```

At activation, cost is `pretrade_value * turnover * bps / 10_000`. The new epoch's `invested_value` is `pretrade_value - cost`. For SPY, set turnover and cost to zero to preserve the historical benchmark convention.

The tidy ledger must contain:

```text
session_date, strategy, mode, rebalance_id, gross_value, net_value,
daily_return, cumulative_return, drawdown, turnover, cost_dollars,
price_snapshot_sha256, previous_record_hash, record_hash, created_utc
```

- [ ] **Step 4: Make ledger updates atomic and deterministic**

Sort new rows by `session_date` then the fixed `LIVE_TRACKED_STRATEGIES` order. Validate the complete pre-existing chain before appending. Write the complete candidate ledger to a temporary CSV, reload and revalidate it, then replace `daily_ledger.csv`. Any exception must leave the prior ledger byte-for-byte unchanged.

- [ ] **Step 5: Run tests and lint**

```powershell
python -m pytest tests/test_live_portfolio_accounting.py -q
python -m ruff check live_portfolio_accounting.py tests/test_live_portfolio_accounting.py
```

Expected: all accounting and integrity tests pass.

- [ ] **Step 6: Commit the accounting core**

```powershell
git add live_portfolio_accounting.py tests/test_live_portfolio_accounting.py
git commit -m "feat: add drift-aware live portfolio ledger"
```

---

### Task 6: Build the Idempotent Daily Runner and Recovery Status

**Files:**
- Create: `track_live_portfolio.py`
- Create: `tests/test_track_live_portfolio.py`

**Interfaces:**
- Consumes: frozen rebalance epochs, `download_adjusted_closes`, immutable price snapshots, accounting functions, and local paths.
- Produces: `DataIncompleteError`, `TrackingRunResult`, `run_daily_tracking(as_of: pd.Timestamp, price_source: AdjustedCloseSource, paths: TrackingPaths) -> TrackingRunResult`, CLI exit codes, `latest_status.json`, and append-only `run_log.jsonl`.

- [ ] **Step 1: Write failing orchestration tests**

Cover all state transitions:

```python
def test_second_run_for_same_session_is_a_noop(tmp_path, frozen_epoch, fake_price_source):
    first = run_daily_tracking(pd.Timestamp("2027-01-05"), fake_price_source, paths(tmp_path))
    before = (tmp_path / "daily_ledger.csv").read_bytes()
    second = run_daily_tracking(pd.Timestamp("2027-01-05"), fake_price_source, paths(tmp_path))
    assert first.sessions_appended == 1
    assert second.sessions_appended == 0
    assert (tmp_path / "daily_ledger.csv").read_bytes() == before


def test_missing_held_price_does_not_partially_append(tmp_path, frozen_epoch, missing_price_source):
    before = existing_ledger_bytes(tmp_path)
    with pytest.raises(DataIncompleteError):
        run_daily_tracking(pd.Timestamp("2027-01-05"), missing_price_source, paths(tmp_path))
    assert existing_ledger_bytes(tmp_path) == before
```

Also test weekend no-op, missing SPY, stale lock rejection, active lock rejection, corrupt ledger rejection, manifest-hash mismatch, more than one new session processed sequentially, and status timestamps in UTC/IST.

- [ ] **Step 2: Implement an exclusive process lock**

Define the result and expected data error before the orchestration function:

```python
class DataIncompleteError(RuntimeError):
    """A completed session lacks a required benchmark or held-asset price."""


@dataclass(frozen=True)
class TrackingRunResult:
    authoritative_session: pd.Timestamp | None
    sessions_appended: int
    ledger_sha256: str | None
```

Create `data/live_tracking/tracker.lock` with `os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)`. Store PID and UTC timestamp. Always remove a lock created by the current process in `finally`. Never automatically remove an existing lock; report its contents and require an operator to verify no tracker process is active before manual removal.

- [ ] **Step 3: Implement daily state transitions**

The runner must:

1. acquire the lock;
2. load and validate every active epoch and the ledger chain;
3. download from the earliest needed entry/ledger date through `as_of`;
4. identify SPY's authoritative completed session;
5. return success with `sessions_appended=0` if there is no new session;
6. validate every held ticker for every new session before writing anything;
7. write or verify immutable price snapshots;
8. activate pending epochs on the first completed session after `effective_after`;
9. value all five aligned series sequentially;
10. atomically append the ledger;
11. write status and run-log records; and
12. release the lock.

Use exit code `0` for updated or no-op, `2` for incomplete market data, `3` for manifest/ledger integrity failure, and `4` for lock contention.

- [ ] **Step 4: Write status without masking failures**

`latest_status.json` must include `state`, `exit_code`, `started_utc`, `finished_utc`, `finished_ist`, `authoritative_session`, `sessions_appended`, `active_rebalance_ids`, `ledger_sha256`, and sanitized `error`. Do not include environment values or request headers.

- [ ] **Step 5: Run focused and cross-module tests**

```powershell
python -m pytest tests/test_live_tracking.py tests/test_live_price_data.py tests/test_live_portfolio_accounting.py tests/test_track_live_portfolio.py -q
python -m ruff check track_live_portfolio.py tests/test_track_live_portfolio.py
```

Expected: all tests pass; failure tests prove the existing ledger remains unchanged.

- [ ] **Step 6: Commit the daily runner**

```powershell
git add track_live_portfolio.py tests/test_track_live_portfolio.py
git commit -m "feat: add idempotent daily portfolio tracker"
```

---

### Task 7: Generate a Daily Human-Readable Scorecard

**Files:**
- Create: `report_live_portfolio.py`
- Create: `tests/test_report_live_portfolio.py`
- Modify: `track_live_portfolio.py`

**Interfaces:**
- Consumes: validated daily ledger, latest status, and frozen manifest metadata.
- Produces: `LiveReportOutputs`, `outputs/live_tracking/latest_summary.csv`, `latest_report.md`, `equity_and_drawdown.png`, and `generate_live_report(ledger: pd.DataFrame, output_dir: Path) -> LiveReportOutputs`.

- [ ] **Step 1: Write failing reporting tests**

```python
def test_report_never_combines_shadow_and_scored_rows(tmp_path):
    outputs = generate_live_report(make_mixed_mode_ledger(), tmp_path)
    report = outputs.report_path.read_text(encoding="utf-8")
    assert "Shadow Operations" in report
    assert "Scored Forward Record" in report
    assert "Combined CAGR" not in report


def test_short_history_is_labeled_preliminary(tmp_path):
    outputs = generate_live_report(make_ledger(sessions=20), tmp_path)
    summary = pd.read_csv(outputs.summary_path)
    assert summary["annualized_metrics_status"].eq("insufficient_history_lt_63_sessions").all()
```

Also test current NAV, total return, excess versus SPY, maximum drawdown, last successful session, staleness warning, and deterministic output ordering.

- [ ] **Step 2: Implement conservative metrics**

Define report paths as a frozen result:

```python
@dataclass(frozen=True)
class LiveReportOutputs:
    summary_path: Path
    report_path: Path
    graph_path: Path
```

Always report current net value, cumulative return, SPY cumulative return, excess cumulative return, drawdown, number of sessions, data-through date, and tracking state. Report annualized volatility, Sharpe, and CAGR only after at least 63 aligned sessions; label 63–251 sessions `preliminary` and 252+ sessions `established_sample`, without implying statistical validation.

- [ ] **Step 3: Build the Markdown and graph outputs**

The Markdown report must lead with status and data freshness, then show separate shadow/scored tables, last rebalance metadata, current holdings, turnover/costs, integrity hashes, and any blocked corporate action. The graph must use separate panels for normalized equity and drawdown and visually distinguish shadow from scored data.

- [ ] **Step 4: Integrate reporting after a successful/no-op run**

Call `generate_live_report` only after the ledger validates. A report-rendering failure must set status to `report_failed` and return a non-zero integrity exit without changing the ledger.

- [ ] **Step 5: Run tests and lint**

```powershell
python -m pytest tests/test_report_live_portfolio.py tests/test_track_live_portfolio.py -q
python -m ruff check report_live_portfolio.py track_live_portfolio.py tests/test_report_live_portfolio.py
```

Expected: report artifacts are created under temporary test paths; mode separation and short-history labels pass.

- [ ] **Step 6: Commit reporting**

```powershell
git add report_live_portfolio.py track_live_portfolio.py tests/test_report_live_portfolio.py tests/test_track_live_portfolio.py
git commit -m "feat: report daily forward portfolio status"
```

---

### Task 8: Add Approval-Gated Windows Scheduling, Full Verification, and Handoff

**Files:**
- Create: `scripts/run_daily_tracking.ps1`
- Create: `scripts/install_daily_tracking_task.ps1`
- Create: `tests/test_scheduler_scripts.py`
- Modify: `README.md`
- Modify: `docs/live_forward_tracking_spec.md`

**Interfaces:**
- Consumes: `.venv\Scripts\python.exe`, `track_live_portfolio.py`, local task time 06:30, and Windows ScheduledTasks cmdlets.
- Produces: scheduled task `SinStoksDailyForwardTracker`, timestamped local logs, installation dry-run output, and operator runbook.

- [ ] **Step 1: Write scheduler-script contract tests**

Create tests that read the scripts and require these tokens:

```python
def test_installer_requires_explicit_register_switch():
    script = Path("scripts/install_daily_tracking_task.ps1").read_text(encoding="utf-8")
    assert "[switch]$Register" in script
    assert "if (-not $Register)" in script
    assert "Register-ScheduledTask" in script
    assert "SinStoksDailyForwardTracker" in script


def test_runner_uses_repo_virtual_environment_and_propagates_exit_code():
    script = Path("scripts/run_daily_tracking.ps1").read_text(encoding="utf-8")
    assert ".venv\\Scripts\\python.exe" in script
    assert "track_live_portfolio.py" in script
    assert "exit $LASTEXITCODE" in script
```

- [ ] **Step 2: Implement the stable runner script**

`scripts/run_daily_tracking.ps1` must derive the repository root from `$PSScriptRoot`, verify the virtual-environment interpreter, create `logs/live_tracking`, invoke the tracker with `--as-of` equal to the current local date, redirect stdout/stderr to a timestamped log, and propagate the Python exit code. It must not print environment variables.

- [ ] **Step 3: Implement a non-mutating installer dry run**

The installer parameters must be:

```powershell
param(
    [switch]$Register,
    [string]$TaskName = "SinStoksDailyForwardTracker",
    [string]$DailyTime = "06:30"
)
```

Without `-Register`, print the resolved Python path, runner path, task name, trigger, principal mode, restart policy, and the exact fact that no task was created. With `-Register`, use:

```powershell
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arguments
$trigger = New-ScheduledTaskTrigger -Daily -At $DailyTime
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 20) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 15)
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description "Daily fail-closed tracking for frozen PAM and HDBSCAN paper portfolios"
```

Before registering, fail if a task with the same name already exists. Do not overwrite or unregister an existing task automatically.

- [ ] **Step 4: Validate PowerShell syntax without registering anything**

Run:

```powershell
powershell -NoProfile -Command "[scriptblock]::Create((Get-Content -Raw 'scripts/run_daily_tracking.ps1')) | Out-Null; [scriptblock]::Create((Get-Content -Raw 'scripts/install_daily_tracking_task.ps1')) | Out-Null"
python -m pytest tests/test_scheduler_scripts.py -q
powershell -NoProfile -File scripts/install_daily_tracking_task.ps1
```

Expected: scripts parse, tests pass, dry run prints `No scheduled task was created`, and `Get-ScheduledTask -TaskName SinStoksDailyForwardTracker -ErrorAction SilentlyContinue` returns nothing.

- [ ] **Step 5: Run the complete offline verification suite**

```powershell
python -m pytest -q
python -m ruff check .
git diff --check
git status --short
```

Expected: all tests pass, Ruff passes, no whitespace errors, generated live directories remain ignored, and only intended source/documentation changes are present.

- [ ] **Step 6: Perform a synthetic end-to-end acceptance run**

Use pytest fixtures only; do not download market data or create a financial experiment:

```powershell
python -m pytest tests/test_freeze_live_portfolio.py tests/test_track_live_portfolio.py tests/test_report_live_portfolio.py -q
```

Expected: a temporary scored epoch activates, two sessions append exactly once, a repeated run is a no-op, and the report shows five aligned strategies with a valid hash chain.

- [ ] **Step 7: Commit scheduling and documentation**

```powershell
git add scripts/run_daily_tracking.ps1 scripts/install_daily_tracking_task.ps1 tests/test_scheduler_scripts.py README.md docs/live_forward_tracking_spec.md
git commit -m "feat: schedule daily forward portfolio tracking"
```

- [ ] **Step 8: Stop for explicit launch approval**

Present these four separately approved actions and their consequences:

1. Refresh live market/fundamental data using user-supplied credentials.
2. Freeze an immediate `shadow` launch epoch or wait for the next January `scored` epoch.
3. Run the first real daily valuation and inspect holdings, hashes, costs, and report.
4. Register the Windows scheduled task with `-Register`.

Do not execute any of them merely because implementation tests passed.

- [ ] **Step 9: After approval, register and verify the task**

Run only after the user explicitly approves task registration:

```powershell
powershell -NoProfile -File scripts/install_daily_tracking_task.ps1 -Register
Get-ScheduledTask -TaskName SinStoksDailyForwardTracker | Format-List TaskName,State,Description
Start-ScheduledTask -TaskName SinStoksDailyForwardTracker
```

Then inspect `outputs/live_tracking/latest_status.json`, the newest log, ledger hash, and report. Expected: task exists, a market holiday/weekend gives a clean no-op, and a completed new session appends exactly five aligned ledger rows.

---

## Final Review and Merge Gate

Before proposing a merge, verify all of the following:

- The individual-stock selection description says random rather than retrospective and retains only evidence-supported candidate-pool caveats.
- Historical selection outputs and hashes are unchanged.
- Shadow and scored data cannot be combined by the report or metric code.
- A scored epoch cannot be backfilled after its first executable session.
- All five strategies share identical session dates.
- Missing SPY or held-asset prices produce no ledger mutation.
- Daily reruns are byte-for-byte idempotent.
- Rebalance epochs and raw price snapshots cannot be overwritten.
- The ledger hash chain detects edits.
- Transaction costs are charged exactly once at activation/rebalance and never subtracted again in reporting.
- The scheduled task was not registered without immediate explicit approval.
- `python -m pytest -q`, `python -m ruff check .`, and `git diff --check` pass.
- The branch remains isolated until the user reviews the first real holdings, live protocol, and scheduler behavior.
