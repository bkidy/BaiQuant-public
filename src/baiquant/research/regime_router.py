from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from baiquant.strategy.live20k import build_market_regime


@dataclass(frozen=True, slots=True)
class RegimePolicy:
    strategy: str
    max_signals: int


DEFAULT_ROUTER_POLICIES: dict[str, RegimePolicy] = {
    "bull": RegimePolicy(strategy="turbo", max_signals=4),
    "broad_rebound": RegimePolicy(strategy="hybrid", max_signals=4),
    "structural": RegimePolicy(strategy="hybrid", max_signals=4),
    "weak_range": RegimePolicy(strategy="hybrid", max_signals=1),
    "bear_weak": RegimePolicy(strategy="cash", max_signals=0),
    "extreme_risk": RegimePolicy(strategy="cash", max_signals=0),
    "unknown": RegimePolicy(strategy="cash", max_signals=0),
}


def build_regime_frame(prices: pd.DataFrame) -> pd.DataFrame:
    regime = build_market_regime(prices)
    if regime.empty:
        return pd.DataFrame(columns=_regime_columns())
    regime = regime.copy()
    regime["market_ma120"] = regime["market_equity"].rolling(120, min_periods=120).mean()
    regime["dist_ma60"] = regime["market_equity"] / regime["market_ma60"] - 1
    return classify_regime_frame(regime)


def classify_regime_frame(regime: pd.DataFrame) -> pd.DataFrame:
    if regime.empty:
        return pd.DataFrame(columns=_regime_columns())
    frame = regime.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if "market_ma120" not in frame.columns:
        frame["market_ma120"] = np.nan
    if "dist_ma60" not in frame.columns and {"market_equity", "market_ma60"}.issubset(frame.columns):
        frame["dist_ma60"] = frame["market_equity"] / frame["market_ma60"] - 1
    frame["regime"] = frame.apply(_classify_regime_row, axis=1)
    return frame.reset_index(drop=True)


def build_regime_router_signals(
    signal_frames: Mapping[str, pd.DataFrame],
    regimes: pd.DataFrame,
    policies: Mapping[str, RegimePolicy] | None = None,
) -> pd.DataFrame:
    policies = dict(DEFAULT_ROUTER_POLICIES if policies is None else policies)
    if regimes.empty:
        return _empty_router_signals()
    regime_frame = regimes.copy()
    regime_frame["date"] = pd.to_datetime(regime_frame["date"])
    if "regime" not in regime_frame.columns:
        regime_frame = classify_regime_frame(regime_frame)
    regime_by_date = regime_frame.set_index("date")["regime"].to_dict()

    rows = []
    for date in sorted(regime_by_date):
        regime = str(regime_by_date[date])
        policy = policies.get(regime, policies.get("unknown", RegimePolicy("cash", 0)))
        if policy.strategy == "cash" or policy.max_signals <= 0:
            continue
        source = signal_frames.get(policy.strategy)
        if source is None or source.empty:
            continue
        source_frame = source.copy()
        source_frame["date"] = pd.to_datetime(source_frame["date"])
        day = source_frame.loc[source_frame["date"] == date].copy()
        if day.empty:
            continue
        day = _sort_signal_day(day).head(policy.max_signals).copy()
        day["route_strategy"] = policy.strategy
        day["regime"] = regime
        day["_signal_order"] = range(1, len(day) + 1)
        rows.append(day)
    if not rows:
        return _empty_router_signals()
    routed = pd.concat(rows, ignore_index=True)
    if "_signal_order" not in routed.columns:
        routed["_signal_order"] = routed.groupby("date").cumcount() + 1
    return routed.reset_index(drop=True)


def _classify_regime_row(row: pd.Series) -> str:
    breadth = row.get("breadth_ma20", np.nan)
    if pd.isna(breadth):
        return "unknown"
    breadth_value = float(breadth)
    if breadth_value < 0.15:
        return "extreme_risk"
    if breadth_value < 0.25:
        return "bear_weak"
    if breadth_value < 0.35:
        return "weak_range"
    if breadth_value < 0.55:
        return "structural"

    market_equity = row.get("market_equity", np.nan)
    market_ma60 = row.get("market_ma60", np.nan)
    market_ma120 = row.get("market_ma120", np.nan)
    trend60 = pd.notna(market_equity) and pd.notna(market_ma60) and float(market_equity) > float(market_ma60)
    trend120 = pd.notna(market_equity) and pd.notna(market_ma120) and float(market_equity) > float(market_ma120)
    return "bull" if trend60 and trend120 else "broad_rebound"


def _sort_signal_day(day: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [column for column in ["score_rank", "raw_rank", "score", "code"] if column in day.columns]
    if not sort_columns:
        return day.sort_values("code") if "code" in day.columns else day
    ascending = [True if column != "score" else False for column in sort_columns]
    return day.sort_values(sort_columns, ascending=ascending)


def _empty_router_signals() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "code", "route_strategy", "regime", "_signal_order"])


def _regime_columns() -> list[str]:
    return [
        "date",
        "breadth_ma20",
        "market_ret",
        "market_equity",
        "market_ma20",
        "market_ma60",
        "market_ma120",
        "dist_ma60",
        "regime",
    ]
