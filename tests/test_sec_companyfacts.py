from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fundamental_sources import SecIssuerConfig
from sec_companyfacts import (
    SecCompanyFactsClient,
    normalize_fact_records,
    select_filed_shares,
    select_ttm_earnings,
)


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
        expected_share_classes=expected_share_classes,
    )


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


def make_ttm_companyfacts(include_comparison: bool = True) -> dict[str, object]:
    records = [
        {
            "start": "2018-01-01",
            "end": "2018-12-31",
            "val": 100.0,
            "accn": "annual",
            "fy": 2018,
            "fp": "FY",
            "form": "10-K",
            "filed": "2019-02-15",
            "frame": "CY2018",
        },
        {
            "start": "2019-01-01",
            "end": "2019-09-30",
            "val": 30.0,
            "accn": "interim",
            "fy": 2019,
            "fp": "Q3",
            "form": "10-Q",
            "filed": "2019-11-01",
            "frame": None,
        },
    ]
    if include_comparison:
        records.append(
            {
                "start": "2018-01-01",
                "end": "2018-09-30",
                "val": 20.0,
                "accn": "interim",
                "fy": 2019,
                "fp": "Q3",
                "form": "10-Q",
                "filed": "2019-11-01",
                "frame": None,
            }
        )
    return {"cik": 1, "facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": records}}}}}


def make_share_facts(values: list[float], filed: str = "2019-11-05") -> dict[str, object]:
    records = [
        {
            "end": "2019-10-31",
            "val": value,
            "accn": "shares",
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
        "facts": {"dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": records}}}},
    }


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
    assert client.get_companyfacts(1234) == client.get_companyfacts(1234)
    assert len(calls) == 1
    assert (tmp_path / "CIK0000001234.json").exists()


def test_companyfacts_client_requires_explicit_contact(tmp_path):
    with pytest.raises(ValueError, match="contact email"):
        SecCompanyFactsClient("anonymous", tmp_path)


def test_normalize_fact_records_preserves_filing_chronology():
    records = normalize_fact_records(make_companyfacts_fixture(), "us-gaap", "NetIncomeLoss", "USD")
    assert records[0].start == pd.Timestamp("2018-01-01")
    assert records[0].end == pd.Timestamp("2018-12-31")
    assert records[0].filed == pd.Timestamp("2019-02-15")
    assert records[0].value == 100.0


def test_normalize_fact_records_rejects_missing_or_nonfinite():
    with pytest.raises(ValueError, match="Missing SEC fact"):
        normalize_fact_records({"facts": {}}, "us-gaap", "NetIncomeLoss", "USD")
    with pytest.raises(ValueError, match="finite"):
        normalize_fact_records(make_companyfacts_fixture(np.inf), "us-gaap", "NetIncomeLoss", "USD")


def test_ttm_earnings_uses_annual_plus_same_filing_ytd_comparison():
    observation = select_ttm_earnings(make_ttm_companyfacts(), make_us_issuer(), pd.Timestamp("2020-01-01"))
    assert observation.value == 110.0
    assert observation.start == pd.Timestamp("2018-10-01")
    assert observation.end == pd.Timestamp("2019-09-30")
    assert observation.available_date == pd.Timestamp("2019-11-01")
    assert observation.method == "annual_plus_ytd_less_prior_ytd"


def test_ttm_earnings_ignores_future_filing():
    facts = make_ttm_companyfacts()
    future = dict(facts["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"][1])
    future.update({"val": 999.0, "filed": "2020-02-01", "accn": "future"})
    facts["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"].append(future)
    observation = select_ttm_earnings(facts, make_us_issuer(), pd.Timestamp("2020-01-01"))
    assert observation.value == 110.0


def test_ttm_earnings_uses_annual_only_without_post_annual_interim():
    facts = make_ttm_companyfacts()
    facts["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"] = facts["facts"]["us-gaap"]["NetIncomeLoss"][
        "units"
    ]["USD"][:1]
    observation = select_ttm_earnings(facts, make_us_issuer(), pd.Timestamp("2020-01-01"))
    assert observation.value == 100.0
    assert observation.method == "annual_fallback"


def test_ttm_earnings_rejects_unmatched_interim_comparison():
    with pytest.raises(ValueError, match="same-filing prior-year comparison"):
        select_ttm_earnings(
            make_ttm_companyfacts(include_comparison=False),
            make_us_issuer(),
            pd.Timestamp("2020-01-01"),
        )


def test_single_class_shares_deduplicate_repeated_values():
    observation = select_filed_shares(
        make_share_facts([100.0, 100.0]), make_us_issuer(), pd.Timestamp("2020-01-01")
    )
    assert observation.shares == 100.0
    assert observation.component_count == 1


def test_multiclass_shares_use_total_matching_components():
    observation = select_filed_shares(
        make_share_facts([60.0, 40.0, 100.0]),
        make_us_issuer(expected_share_classes=2),
        pd.Timestamp("2020-01-01"),
    )
    assert observation.shares == 100.0
    assert observation.aggregation == "reported_total_matches_components"


def test_multiclass_shares_sum_exact_expected_components():
    observation = select_filed_shares(
        make_share_facts([60.0, 40.0]),
        make_us_issuer(expected_share_classes=2),
        pd.Timestamp("2020-01-01"),
    )
    assert observation.shares == 100.0
    assert observation.aggregation == "sum_expected_classes"


def test_multiclass_shares_reject_ambiguous_values():
    with pytest.raises(ValueError, match="ambiguous share facts"):
        select_filed_shares(
            make_share_facts([60.0, 40.0, 30.0]),
            make_us_issuer(expected_share_classes=2),
            pd.Timestamp("2020-01-01"),
        )


def test_filed_shares_ignore_future_filing():
    facts = make_share_facts([100.0])
    future = dict(facts["facts"]["dei"]["EntityCommonStockSharesOutstanding"]["units"]["shares"][0])
    future.update({"val": 500.0, "filed": "2020-02-01", "accn": "future"})
    facts["facts"]["dei"]["EntityCommonStockSharesOutstanding"]["units"]["shares"].append(future)
    observation = select_filed_shares(facts, make_us_issuer(), pd.Timestamp("2020-01-01"))
    assert observation.shares == 100.0


@pytest.mark.parametrize("invalid_value", [0.0, -1.0])
def test_filed_shares_reject_invalid_values(invalid_value):
    with pytest.raises(ValueError, match="finite and positive"):
        select_filed_shares(make_share_facts([invalid_value]), make_us_issuer(), pd.Timestamp("2020-01-01"))
