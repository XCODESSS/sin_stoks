"""Audited 30-stock behavioral vice universe taxonomy and helpers."""

from __future__ import annotations

COMPANIES: dict[str, tuple[str, str]] = {
    # Alcohol (5)
    "DEO": ("Diageo", "Alcohol"),
    "BUD": ("Anheuser-Busch InBev", "Alcohol"),
    "STZ": ("Constellation Brands", "Alcohol"),
    "BF-B": ("Brown-Forman", "Alcohol"),
    "PRNDY": ("Pernod Ricard", "Alcohol"),
    # Energy Drinks (5)
    "MNST": ("Monster Beverage", "Energy Drinks"),
    "CELH": ("Celsius Holdings", "Energy Drinks"),
    "KDP": ("Keurig Dr Pepper", "Energy Drinks"),
    "PEP": ("PepsiCo", "Energy Drinks"),
    "KO": ("Coca-Cola", "Energy Drinks"),
    # Social Media (5)
    "META": ("Meta Platforms", "Social Media"),
    "GOOGL": ("Alphabet", "Social Media"),
    "SNAP": ("Snap Inc.", "Social Media"),
    "TCEHY": ("Tencent", "Social Media"),
    "MSFT": ("Microsoft", "Social Media"),
    # Tobacco & Nicotine (5)
    "PM": ("Philip Morris", "Tobacco & Nicotine"),
    "BTI": ("British American Tobacco", "Tobacco & Nicotine"),
    "MO": ("Altria", "Tobacco & Nicotine"),
    "UVV": ("Universal Corporation", "Tobacco & Nicotine"),
    "TPB": ("Turning Point Brands", "Tobacco & Nicotine"),
    # Gaming (5)
    "NTDOY": ("Nintendo", "Gaming"),
    "EA": ("Electronic Arts", "Gaming"),
    "TTWO": ("Take-Two Interactive", "Gaming"),
    "CCOEY": ("Capcom", "Gaming"),
    "UBSFY": ("Ubisoft", "Gaming"),
    # Quick Service Restaurants (5)
    "MCD": ("McDonald's", "Quick Service Restaurants"),
    "CMG": ("Chipotle Mexican Grill", "Quick Service Restaurants"),
    "YUM": ("Yum! Brands", "Quick Service Restaurants"),
    "DPZ": ("Domino's Pizza", "Quick Service Restaurants"),
    "QSR": ("Restaurant Brands International", "Quick Service Restaurants"),
    # Benchmark (1)
    "SPY": ("S&P 500 ETF", "Benchmark"),
}

PORTFOLIO_TICKERS: list[str] = [t for t, (_, s) in COMPANIES.items() if s != "Benchmark"]
BENCHMARK_TICKER: str = "SPY"
BENCHMARK_TICKERS: set[str] = {BENCHMARK_TICKER}
ALL_TICKERS: list[str] = list(COMPANIES.keys())

SECTORS: list[str] = [
    "Alcohol",
    "Energy Drinks",
    "Social Media",
    "Tobacco & Nicotine",
    "Gaming",
    "Quick Service Restaurants",
]


def get_company_name(ticker: str) -> str:
    """Return the human-readable company name for a given ticker."""
    return COMPANIES.get(ticker, (ticker, "Unknown"))[0]


def get_sector(ticker: str) -> str:
    """Return the behavioral sector name for a given ticker."""
    return COMPANIES.get(ticker, ("Unknown", "Unknown"))[1]


def get_portfolio_tickers() -> list[str]:
    """Return the list of 30 portfolio tickers."""
    return list(PORTFOLIO_TICKERS)


def get_tickers_by_sector() -> dict[str, list[str]]:
    """Return a mapping of sector name -> list of tickers in that sector."""
    by_sector: dict[str, list[str]] = {s: [] for s in SECTORS}
    for ticker in PORTFOLIO_TICKERS:
        sector = get_sector(ticker)
        if sector in by_sector:
            by_sector[sector].append(ticker)
    return by_sector


def get_sector_mapping() -> dict[str, str]:
    """Return a mapping of ticker -> sector."""
    return {t: get_sector(t) for t in PORTFOLIO_TICKERS}
