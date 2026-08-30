from __future__ import annotations

import pandas as pd

from run_ten_year_selection_experiment import (
    _dynamic_turnover,
    exact_fundamental_snapshot,
    price_eligible_tickers,
)


def test_price_eligibility_requires_complete_trailing_window():
    dates = pd.date_range("2013-01-04", periods=104, freq="W-FRI")
    returns = pd.DataFrame({"A": 0.01, "B": 0.02}, index=dates)
    returns.loc[dates[0], "B"] = None

    trailing, eligible = price_eligible_tickers(returns, pd.Timestamp("2016-01-01"))

    assert len(trailing) == 104
    assert eligible == ["A"]


def test_exact_snapshot_does_not_reuse_stale_prior_year_record():
    fundamentals = pd.DataFrame(
        {
            "ticker": [f"T{i:02d}" for i in range(13)],
            "rebalance_date": [pd.Timestamp("2016-01-01")] * 12 + [pd.Timestamp("2015-01-01")],
            "available_date": [pd.Timestamp("2015-12-01")] * 12 + [pd.Timestamp("2014-12-01")],
        }
    )

    snapshot = exact_fundamental_snapshot(
        fundamentals,
        pd.Timestamp("2016-01-01"),
        list(fundamentals["ticker"]),
    )

    assert len(snapshot) == 12
    assert "T12" not in snapshot.index


def test_dynamic_turnover_counts_assets_that_leave_the_universe():
    old = pd.Series({"A": 0.5, "B": 0.5})
    new = pd.Series({"B": 0.5, "C": 0.5})

    assert _dynamic_turnover(old, new) == 0.5
