from __future__ import annotations

import pandas as pd


def buyback_event(events: pd.DataFrame, as_of: pd.Timestamp, lookback_days: int = 30) -> pd.Series:
    if events.empty:
        return pd.Series(dtype=float, name="buyback_event")
    start = as_of - pd.Timedelta(days=lookback_days)
    recent = events.loc[
        (events["date"] >= start)
        & (events["date"] <= as_of)
        & (events["event_type"].str.lower() == "buyback")
    ]
    if recent.empty:
        return pd.Series(dtype=float, name="buyback_event")
    scores = recent.assign(score=1.0).groupby("code")["score"].max()
    scores.name = "buyback_event"
    return scores
