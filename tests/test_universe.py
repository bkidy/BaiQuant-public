import pandas as pd

from baiquant.data.bundle import MarketDataBundle
from baiquant.universe.filters import UniverseConfig, build_universe


def test_universe_filters_st_recent_paused_and_illiquid_names() -> None:
    prices = pd.DataFrame(
        [
            ["2026-01-10", "A", 10, 11, 9, 10.5, 1000, 10_500, 0, 0, 0],
            ["2026-01-10", "B", 10, 11, 9, 10.5, 1000, 10_500, 0, 0, 0],
            ["2026-01-10", "C", 10, 11, 9, 10.5, 1000, 10_500, 0, 0, 0],
            ["2026-01-10", "D", 10, 11, 9, 10.5, 1000, 999, 0, 0, 0],
            ["2026-01-10", "E", 10, 11, 9, 10.5, 1000, 10_500, 1, 0, 0],
        ],
        columns=[
            "date",
            "code",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "paused",
            "limit_up",
            "limit_down",
        ],
    )
    prices["date"] = pd.to_datetime(prices["date"])
    stocks = pd.DataFrame(
        [
            ["A", "Alpha", "Semi", "2020-01-01", 0],
            ["B", "Beta", "Semi", "2020-01-01", 1],
            ["C", "Gamma", "Semi", "2025-12-01", 0],
            ["D", "Delta", "Semi", "2020-01-01", 0],
            ["E", "Epsilon", "Semi", "2020-01-01", 0],
        ],
        columns=["code", "name", "industry", "list_date", "is_st"],
    )
    stocks["list_date"] = pd.to_datetime(stocks["list_date"])
    bundle = MarketDataBundle(prices=prices, fundamentals=pd.DataFrame(), stocks=stocks)

    result = build_universe(
        bundle,
        as_of="2026-01-10",
        config=UniverseConfig(min_listed_days=180, min_amount=1_000),
    )

    assert result["code"].tolist() == ["A"]


def test_universe_filters_bj_and_stocks_outside_price_band() -> None:
    prices = pd.DataFrame(
        [
            ["2026-01-10", "KEEP.SZ", 10, 11, 9, 20.0, 1000, 100_000_000, 0, 0, 0],
            ["2026-01-10", "HIGH.SZ", 10, 11, 9, 120.0, 1000, 100_000_000, 0, 0, 0],
            ["2026-01-10", "LOW.SZ", 10, 11, 9, 2.0, 1000, 100_000_000, 0, 0, 0],
            ["2026-01-10", "920001.BJ", 10, 11, 9, 20.0, 1000, 100_000_000, 0, 0, 0],
        ],
        columns=[
            "date",
            "code",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "paused",
            "limit_up",
            "limit_down",
        ],
    )
    prices["date"] = pd.to_datetime(prices["date"])
    stocks = pd.DataFrame(
        [
            ["KEEP.SZ", "Keep", "A", "2020-01-01", 0],
            ["HIGH.SZ", "High", "A", "2020-01-01", 0],
            ["LOW.SZ", "Low", "A", "2020-01-01", 0],
            ["920001.BJ", "BJ", "A", "2020-01-01", 0],
        ],
        columns=["code", "name", "industry", "list_date", "is_st"],
    )
    stocks["list_date"] = pd.to_datetime(stocks["list_date"])

    universe = build_universe(
        MarketDataBundle(prices=prices, fundamentals=pd.DataFrame(), stocks=stocks),
        "2026-01-10",
        UniverseConfig(min_price=3, max_price=80, exclude_bj=True),
    )

    assert universe["code"].tolist() == ["KEEP.SZ"]


def test_universe_can_exclude_star_market_and_chinext() -> None:
    prices = pd.DataFrame(
        [
            ["2026-01-10", "600001.SH", 10, 11, 9, 20.0, 1000, 100_000_000, 0, 0, 0],
            ["2026-01-10", "688001.SH", 10, 11, 9, 20.0, 1000, 100_000_000, 0, 0, 0],
            ["2026-01-10", "300001.SZ", 10, 11, 9, 20.0, 1000, 100_000_000, 0, 0, 0],
        ],
        columns=[
            "date",
            "code",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "paused",
            "limit_up",
            "limit_down",
        ],
    )
    prices["date"] = pd.to_datetime(prices["date"])

    universe = build_universe(
        MarketDataBundle(prices=prices, stocks=pd.DataFrame()),
        "2026-01-10",
        UniverseConfig(exclude_star=True, exclude_chinext=True),
    )

    assert universe["code"].tolist() == ["600001.SH"]


def test_universe_requires_minimum_price_history_rows() -> None:
    prices = pd.DataFrame(
        [
            ["2026-01-08", "ENOUGH.SZ", 10, 11, 9, 10.0, 1000, 100_000_000, 0, 0, 0],
            ["2026-01-09", "ENOUGH.SZ", 10, 11, 9, 10.0, 1000, 100_000_000, 0, 0, 0],
            ["2026-01-10", "ENOUGH.SZ", 10, 11, 9, 10.0, 1000, 100_000_000, 0, 0, 0],
            ["2026-01-10", "SHORT.SZ", 10, 11, 9, 10.0, 1000, 100_000_000, 0, 0, 0],
        ],
        columns=[
            "date",
            "code",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "paused",
            "limit_up",
            "limit_down",
        ],
    )
    prices["date"] = pd.to_datetime(prices["date"])

    universe = build_universe(
        MarketDataBundle(prices=prices, stocks=pd.DataFrame()),
        "2026-01-10",
        UniverseConfig(min_history_days=3),
    )

    assert universe["code"].tolist() == ["ENOUGH.SZ"]
