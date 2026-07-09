from __future__ import annotations

import re

import pandas as pd

from baiquant.data.bundle import MarketDataBundle
from baiquant.factors.events import buyback_event
from baiquant.factors.fundamental import growth, quality, value
from baiquant.factors.money_flow import big_order, money_flow
from baiquant.factors.technical import (
    close_position,
    macd_momentum,
    momentum,
    rsi,
    trend_pullback,
    volatility,
    volume_score,
    week_52_high,
)


def compute_factor(bundle: MarketDataBundle, name: str, as_of: pd.Timestamp) -> pd.Series:
    momentum_match = re.fullmatch(r"momentum_(\d+)d", name)
    if momentum_match:
        return momentum(bundle.prices, as_of, int(momentum_match.group(1))).rename(name)

    volatility_match = re.fullmatch(r"volatility_(\d+)d", name)
    if volatility_match:
        return volatility(bundle.prices, as_of, int(volatility_match.group(1))).rename(name)

    rsi_match = re.fullmatch(r"rsi(\d+)", name)
    if rsi_match:
        return rsi(bundle.prices, as_of, int(rsi_match.group(1))).rename(name)

    close_position_match = re.fullmatch(r"close_position_(\d+)d", name)
    if close_position_match:
        return close_position(bundle.prices, as_of, int(close_position_match.group(1))).rename(name)

    if name == "volume_score":
        return volume_score(bundle.prices, as_of).rename(name)
    if name == "week_52_high":
        return week_52_high(bundle.prices, as_of).rename(name)
    if name == "trend_pullback":
        return trend_pullback(bundle.prices, as_of).rename(name)
    if name == "macd_momentum":
        return macd_momentum(bundle.prices, as_of).rename(name)
    if name == "quality":
        return quality(bundle.fundamentals, as_of).rename(name)
    if name == "growth":
        return growth(bundle.fundamentals, as_of).rename(name)
    if name == "value":
        return value(bundle.fundamentals, as_of).rename(name)
    if name == "buyback_event":
        return buyback_event(bundle.events, as_of).rename(name)
    if name == "money_flow":
        return money_flow(bundle.money_flow, as_of).rename(name)
    if name == "big_order":
        return big_order(bundle.money_flow, as_of).rename(name)

    raise KeyError(f"Unknown factor: {name}")


def compute_factor_frame(
    bundle: MarketDataBundle,
    universe: pd.DataFrame,
    factor_names: list[str],
    as_of: str | pd.Timestamp,
) -> pd.DataFrame:
    frame = universe[["code"]].drop_duplicates().copy()
    cutoff = pd.Timestamp(as_of)
    for name in factor_names:
        series = compute_factor(bundle, name, cutoff)
        factor_frame = series.rename(name).reset_index()
        factor_frame.columns = ["code", name]
        frame = frame.merge(factor_frame, on="code", how="left")
    return frame
