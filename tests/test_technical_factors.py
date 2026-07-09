import pandas as pd

from baiquant.data.bundle import MarketDataBundle
from baiquant.factors.registry import compute_factor


def _price_bundle(closes_by_code: dict[str, list[float]]) -> MarketDataBundle:
    rows = []
    for code, closes in closes_by_code.items():
        for index, close in enumerate(closes):
            date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=index)
            rows.append([date, code, close, close, close, close, 1_000, 1_000_000, 0, 0, 0])
    prices = pd.DataFrame(
        rows,
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
    return MarketDataBundle(prices=prices)


def test_rsi_factor_ranks_persistent_gains_above_persistent_losses() -> None:
    bundle = _price_bundle(
        {
            "UP": [float(value) for value in range(1, 17)],
            "DOWN": [float(value) for value in range(16, 0, -1)],
        }
    )

    factor = compute_factor(bundle, "rsi14", pd.Timestamp("2026-01-16"))

    assert factor["UP"] == 100.0
    assert factor["DOWN"] == 0.0


def test_close_position_factor_measures_latest_close_within_recent_range() -> None:
    bundle = _price_bundle(
        {
            "HIGH": [10.0, 12.0, 14.0],
            "MID": [10.0, 14.0, 12.0],
        }
    )

    factor = compute_factor(bundle, "close_position_3d", pd.Timestamp("2026-01-03"))

    assert factor["HIGH"] == 1.0
    assert factor["MID"] == 0.5


def test_macd_momentum_factor_is_positive_for_rising_trend_and_negative_for_falling_trend() -> None:
    bundle = _price_bundle(
        {
            "UP": [float(value) for value in range(1, 41)],
            "DOWN": [float(value) for value in range(40, 0, -1)],
        }
    )

    factor = compute_factor(bundle, "macd_momentum", pd.Timestamp("2026-02-09"))

    assert factor["UP"] > 0
    assert factor["DOWN"] < 0
