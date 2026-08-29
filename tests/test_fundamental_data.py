from pathlib import Path

import pandas as pd
import pytest

from fundamental_data import fundamentals_as_of, load_fundamentals

FIXTURE = Path("tests/fixtures/fundamentals_point_in_time.csv")


def test_snapshot_uses_latest_record_strictly_before_rebalance():
    fundamentals = load_fundamentals(FIXTURE)

    snapshot = fundamentals_as_of(
        fundamentals,
        pd.Timestamp("2021-01-01"),
        ["A", "B", "C", "D"],
        min_coverage=0.75,
        min_assets=4,
    )

    assert (snapshot["available_date"] < pd.Timestamp("2021-01-01")).all()
    assert snapshot.index.is_unique
    assert list(snapshot.index) == sorted(snapshot.index)
    assert snapshot.loc["A", "trailing_pe"] == 14.0


def test_snapshot_rejects_insufficient_coverage():
    fundamentals = load_fundamentals(FIXTURE)

    with pytest.raises(ValueError, match="fundamental coverage"):
        fundamentals_as_of(
            fundamentals,
            pd.Timestamp("2018-01-01"),
            ["A", "B", "C", "D", "E", "F"],
            min_coverage=0.80,
            min_assets=1,
        )


@pytest.mark.parametrize(
    ("column", "replacement", "message"),
    [
        ("market_cap", 0, "market_cap"),
        ("source", "", "source"),
        ("trailing_pe", -1, "positive trailing_pe"),
    ],
)
def test_loader_rejects_invalid_required_values(tmp_path, column, replacement, message):
    frame = pd.read_csv(FIXTURE)
    frame.loc[0, column] = replacement
    path = tmp_path / "invalid.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match=message):
        load_fundamentals(path)


def test_loader_rejects_duplicate_keys(tmp_path):
    frame = pd.read_csv(FIXTURE)
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    path = tmp_path / "duplicate.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Duplicate"):
        load_fundamentals(path)


def test_loader_rejects_observation_after_availability(tmp_path):
    frame = pd.read_csv(FIXTURE)
    frame.loc[0, "observation_date"] = "2020-01-01"
    frame.loc[0, "available_date"] = "2019-01-01"
    path = tmp_path / "reversed_dates.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="observation_date"):
        load_fundamentals(path)


def test_loader_rejects_invalid_earnings_flag(tmp_path):
    frame = pd.read_csv(FIXTURE)
    frame["earnings_positive"] = frame["earnings_positive"].astype(object)
    frame.loc[0, "earnings_positive"] = "unknown"
    path = tmp_path / "invalid_flag.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="earnings_positive"):
        load_fundamentals(path)


def test_negative_earnings_may_have_missing_pe():
    fundamentals = load_fundamentals(FIXTURE)
    negative_earnings = fundamentals.loc[~fundamentals["earnings_positive"]]

    assert negative_earnings["trailing_pe"].isna().any()


def test_loader_preserves_and_parses_source_provenance(tmp_path):
    frame = pd.read_csv(FIXTURE).iloc[[0]].copy()
    frame["rebalance_date"] = "2020-01-01"
    frame["available_date"] = "2019-12-30"
    frame["price_date"] = "2019-12-30"
    frame["earnings_available_date"] = "2019-11-01"
    frame["shares_available_date"] = "2019-11-05"
    frame["cik"] = 1234
    frame["spot_fx_to_usd"] = 1.0
    frame["shares_component_count"] = 1
    frame["filed_shares"] = 100.0
    frame["trailing_earnings_usd"] = 10.0
    path = tmp_path / "provenance.csv"
    frame.to_csv(path, index=False)

    loaded = load_fundamentals(path)

    assert loaded.loc[0, "cik"] == 1234
    assert loaded.loc[0, "price_date"] == pd.Timestamp("2019-12-30")
    assert loaded.loc[0, "trailing_earnings_usd"] == 10.0


def test_loader_rejects_provenance_available_on_rebalance(tmp_path):
    frame = pd.read_csv(FIXTURE).iloc[[0]].copy()
    frame["rebalance_date"] = frame["available_date"]
    path = tmp_path / "leaking_provenance.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="strictly before rebalance"):
        load_fundamentals(path)
