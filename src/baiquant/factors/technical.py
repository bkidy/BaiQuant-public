from __future__ import annotations

import pandas as pd


def _price_window(prices: pd.DataFrame, as_of: pd.Timestamp, lookback: int) -> pd.DataFrame:
    eligible = prices.loc[prices["date"] <= as_of].copy()
    if eligible.empty:
        return eligible
    return eligible.sort_values(["code", "date"]).groupby("code", as_index=False).tail(lookback + 1)


def momentum(prices: pd.DataFrame, as_of: pd.Timestamp, window: int) -> pd.Series:
    frame = _price_window(prices, as_of, window)
    values: dict[str, float] = {}
    for code, group in frame.groupby("code"):
        group = group.sort_values("date")
        if len(group) <= window:
            values[code] = float("nan")
            continue
        start = group["close"].iloc[-window - 1]
        end = group["close"].iloc[-1]
        values[code] = end / start - 1 if start else float("nan")
    return pd.Series(values, name=f"momentum_{window}d")


def volatility(prices: pd.DataFrame, as_of: pd.Timestamp, window: int) -> pd.Series:
    frame = _price_window(prices, as_of, window)
    values: dict[str, float] = {}
    for code, group in frame.groupby("code"):
        returns = group.sort_values("date")["close"].pct_change().dropna().tail(window)
        values[code] = returns.std(ddof=0) if len(returns) else float("nan")
    return pd.Series(values, name=f"volatility_{window}d")


def rsi(prices: pd.DataFrame, as_of: pd.Timestamp, window: int = 14) -> pd.Series:
    frame = _price_window(prices, as_of, window)
    values: dict[str, float] = {}
    for code, group in frame.groupby("code"):
        close = group.sort_values("date")["close"]
        changes = close.diff().dropna().tail(window)
        if len(changes) < window:
            values[code] = float("nan")
            continue
        average_gain = changes.clip(lower=0).mean()
        average_loss = -changes.clip(upper=0).mean()
        if average_loss == 0:
            values[code] = 100.0 if average_gain > 0 else 50.0
            continue
        relative_strength = average_gain / average_loss
        values[code] = 100 - 100 / (1 + relative_strength)
    return pd.Series(values, name=f"rsi{window}")


def close_position(prices: pd.DataFrame, as_of: pd.Timestamp, window: int = 20) -> pd.Series:
    frame = _price_window(prices, as_of, window)
    values: dict[str, float] = {}
    for code, group in frame.groupby("code"):
        close = group.sort_values("date")["close"].tail(window)
        if len(close) < window:
            values[code] = float("nan")
            continue
        low = close.min()
        high = close.max()
        if high == low:
            values[code] = 0.5
            continue
        values[code] = (close.iloc[-1] - low) / (high - low)
    return pd.Series(values, name=f"close_position_{window}d")


def macd_momentum(prices: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    frame = _price_window(prices, as_of, 60)
    values: dict[str, float] = {}
    for code, group in frame.groupby("code"):
        close = group.sort_values("date")["close"]
        if len(close) < 35:
            values[code] = float("nan")
            continue
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=9, adjust=False).mean()
        values[code] = macd.iloc[-1] - signal.iloc[-1]
    return pd.Series(values, name="macd_momentum")


def volume_score(prices: pd.DataFrame, as_of: pd.Timestamp, window: int = 20) -> pd.Series:
    frame = _price_window(prices, as_of, window)
    values: dict[str, float] = {}
    for code, group in frame.groupby("code"):
        group = group.sort_values("date")
        if len(group) < 2:
            values[code] = float("nan")
            continue
        baseline = group["volume"].iloc[:-1].tail(window).mean()
        latest = group["volume"].iloc[-1]
        values[code] = latest / baseline if baseline else float("nan")
    return pd.Series(values, name="volume_score")


def week_52_high(prices: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    frame = _price_window(prices, as_of, 252)
    values: dict[str, float] = {}
    for code, group in frame.groupby("code"):
        close = group.sort_values("date")["close"]
        high = close.max()
        values[code] = close.iloc[-1] / high if high else float("nan")
    return pd.Series(values, name="week_52_high")


def trend_pullback(prices: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    frame = _price_window(prices, as_of, 60)
    values: dict[str, float] = {}
    for code, group in frame.groupby("code"):
        close = group.sort_values("date")["close"]
        if len(close) < 20:
            values[code] = float("nan")
            continue
        ma20 = close.tail(20).mean()
        ma60 = close.mean()
        pullback = abs(close.iloc[-1] / ma20 - 1) if ma20 else float("nan")
        values[code] = (ma20 / ma60 - 1) - pullback if ma60 else float("nan")
    return pd.Series(values, name="trend_pullback")
