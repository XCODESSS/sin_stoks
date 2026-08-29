"""Frozen issuer metadata for the SEC and Yahoo fundamental source pilot."""

from __future__ import annotations

from dataclasses import dataclass, replace

from universe import PORTFOLIO_TICKERS

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
    """Immutable source and unit choices for one SEC-covered instrument."""

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
    max_share_age_days: int = 200
    additional_instant_share_tags: tuple[tuple[str, str], ...] = ()
    duration_share_tags: tuple[tuple[str, str], ...] = ()
    simfin_price_fallback: bool = False
    simfin_shares_fallback: bool = False

    def __post_init__(self) -> None:
        if self.cik <= 0:
            raise ValueError("CIK must be positive")
        if self.price_scale <= 0:
            raise ValueError("price_scale must be positive")
        if self.share_aggregation != "reconcile":
            raise ValueError("share_aggregation must be 'reconcile'")
        if self.expected_share_classes < 1:
            raise ValueError("expected_share_classes must be positive")
        if self.max_share_age_days < 1:
            raise ValueError("max_share_age_days must be positive")


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
        expected_share_classes=expected_share_classes,
    )


SEC_ISSUERS: dict[str, SecIssuerConfig] = {
    "DEO": SecIssuerConfig("DEO", 835403, "DGE.L", "GBP", 0.01, "ifrs-full", IFRS_INCOME_TAGS, "GBP"),
    "BUD": SecIssuerConfig(
        "BUD",
        1668717,
        "ABI.BR",
        "EUR",
        1.0,
        "ifrs-full",
        IFRS_INCOME_TAGS,
        "USD",
        max_share_age_days=400,
    ),
    "STZ": replace(_us("STZ", 16918), simfin_shares_fallback=True),
    "BF-B": replace(
        _us("BF-B", 14693, expected_share_classes=2),
        max_share_age_days=280,
    ),
    "MNST": _us("MNST", 865752),
    "CELH": _us("CELH", 1341766),
    "KDP": _us("KDP", 1418135),
    "PEP": _us("PEP", 77476),
    "KO": _us("KO", 21344),
    "META": replace(
        _us("META", 1326801, expected_share_classes=2),
        duration_share_tags=(("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"),),
    ),
    "GOOGL": _us("GOOGL", 1652044, expected_share_classes=3),
    "SNAP": replace(
        _us("SNAP", 1564408, expected_share_classes=3),
        additional_instant_share_tags=(("us-gaap", "SharesOutstanding"),),
    ),
    "MSFT": _us("MSFT", 789019),
    "PM": _us("PM", 1413329),
    "BTI": SecIssuerConfig(
        "BTI",
        1303523,
        "BATS.L",
        "GBP",
        0.01,
        "ifrs-full",
        IFRS_INCOME_TAGS,
        "GBP",
        max_share_age_days=400,
    ),
    "MO": _us("MO", 764180),
    "UVV": _us("UVV", 102037),
    "TPB": _us("TPB", 1290677),
    "EA": replace(_us("EA", 712515), simfin_price_fallback=True),
    "TTWO": _us("TTWO", 946581),
    "MCD": _us("MCD", 63908),
    "CMG": _us("CMG", 1058090),
    "YUM": _us("YUM", 1041061),
    "DPZ": _us("DPZ", 1286681),
    "QSR": SecIssuerConfig(
        "QSR",
        1618756,
        "QSR.TO",
        "CAD",
        1.0,
        "us-gaap",
        US_GAAP_INCOME_TAGS,
        "USD",
        max_share_age_days=280,
    ),
}
AUTOMATED_SEC_TICKERS = frozenset(SEC_ISSUERS)
MANUAL_ONLY_TICKERS = frozenset({"PRNDY", "TCEHY", "NTDOY", "CCOEY", "UBSFY"})

if frozenset(PORTFOLIO_TICKERS) != AUTOMATED_SEC_TICKERS | MANUAL_ONLY_TICKERS:
    raise RuntimeError("SEC source registry does not partition the frozen portfolio universe")
