from __future__ import annotations

import pandas as pd


def latest_money_flow(money_flow: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    if money_flow.empty:
        return money_flow.copy()
    dated = money_flow.loc[pd.to_datetime(money_flow["date"]) <= as_of].copy()
    if dated.empty:
        return dated
    return dated.sort_values(["code", "date"]).groupby("code", as_index=False).tail(1)


def money_flow(money_flow_frame: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    latest = latest_money_flow(money_flow_frame, as_of)
    if latest.empty or "main_net_inflow_pct" not in latest.columns:
        return pd.Series(dtype=float, name="money_flow")
    values = pd.to_numeric(latest["main_net_inflow_pct"], errors="coerce")
    return pd.Series(values.to_numpy(), index=latest["code"], name="money_flow")


def big_order(money_flow_frame: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    latest = latest_money_flow(money_flow_frame, as_of)
    if latest.empty:
        return pd.Series(dtype=float, name="big_order")
    large = pd.to_numeric(latest.get("large_net_inflow_pct", 0), errors="coerce").fillna(0)
    super_large = pd.to_numeric(latest.get("super_large_net_inflow_pct", 0), errors="coerce").fillna(0)
    return pd.Series((large + super_large).to_numpy(), index=latest["code"], name="big_order")
