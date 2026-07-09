from __future__ import annotations

import numpy as np
import pandas as pd


def latest_fundamentals(fundamentals: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    if fundamentals.empty:
        return fundamentals.copy()
    dated = fundamentals.loc[fundamentals["date"] <= as_of].copy()
    if dated.empty:
        return dated
    value_columns = ["pe_ttm", "pb", "roe", "revenue_yoy", "profit_yoy"]
    present_value_columns = [column for column in value_columns if column in dated.columns]
    if present_value_columns:
        informative = dated[present_value_columns].notna().any(axis=1)
        if informative.any():
            dated = dated.loc[informative].copy()
    return dated.sort_values(["code", "date"]).groupby("code", as_index=False).tail(1)


def quality(fundamentals: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    latest = latest_fundamentals(fundamentals, as_of)
    if latest.empty:
        return pd.Series(dtype=float, name="quality")
    value = latest["roe"].fillna(0) + 0.5 * latest["profit_yoy"].fillna(0)
    return pd.Series(value.to_numpy(), index=latest["code"], name="quality")


def growth(fundamentals: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    latest = latest_fundamentals(fundamentals, as_of)
    if latest.empty:
        return pd.Series(dtype=float, name="growth")
    value = 0.5 * latest["revenue_yoy"].fillna(0) + 0.5 * latest["profit_yoy"].fillna(0)
    return pd.Series(value.to_numpy(), index=latest["code"], name="growth")


def value(fundamentals: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    latest = latest_fundamentals(fundamentals, as_of)
    if latest.empty:
        return pd.Series(dtype=float, name="value")
    pe = latest["pe_ttm"].replace(0, np.nan)
    pb = latest["pb"].replace(0, np.nan)
    raw = 1 / pe + 1 / pb
    return pd.Series(raw.to_numpy(), index=latest["code"], name="value")
