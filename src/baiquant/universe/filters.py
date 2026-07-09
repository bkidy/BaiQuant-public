from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from baiquant.data.bundle import MarketDataBundle


@dataclass(slots=True)
class UniverseConfig:
    """Rules that define whether a stock is tradable for a signal date."""

    min_listed_days: int = 180
    min_history_days: int = 0
    min_amount: float = 0.0
    exclude_st: bool = True
    exclude_paused: bool = True
    exclude_limit_up: bool = False
    exclude_limit_down: bool = True
    exclude_bj: bool = False
    exclude_star: bool = False
    exclude_chinext: bool = False
    min_price: float = 0.0
    max_price: float = 0.0


def latest_by_code(frame: pd.DataFrame, as_of: str | pd.Timestamp) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    cutoff = pd.Timestamp(as_of)
    dated = frame.loc[frame["date"] <= cutoff].copy()
    if dated.empty:
        return dated
    return dated.sort_values(["code", "date"]).groupby("code", as_index=False).tail(1)


def build_universe(
    bundle: MarketDataBundle,
    as_of: str | pd.Timestamp,
    config: UniverseConfig | None = None,
) -> pd.DataFrame:
    config = config or UniverseConfig()
    cutoff = pd.Timestamp(as_of)
    latest_prices = latest_by_code(bundle.prices, cutoff)
    if latest_prices.empty:
        return latest_prices

    universe = latest_prices.copy()
    if not bundle.stocks.empty:
        universe = universe.merge(bundle.stocks, on="code", how="left")

    if config.exclude_st and "is_st" in universe.columns:
        universe = universe.loc[universe["is_st"].fillna(0).astype(int) == 0]

    if "list_date" in universe.columns and config.min_listed_days > 0:
        listed_days = (cutoff - pd.to_datetime(universe["list_date"])).dt.days
        universe = universe.loc[listed_days >= config.min_listed_days]

    if config.min_history_days > 0:
        historical = bundle.prices.loc[pd.to_datetime(bundle.prices["date"]) <= cutoff]
        history_counts = historical.groupby("code").size().rename("history_days")
        universe = universe.merge(history_counts, on="code", how="left")
        universe = universe.loc[universe["history_days"].fillna(0) >= config.min_history_days]

    if config.exclude_paused and "paused" in universe.columns:
        universe = universe.loc[universe["paused"].fillna(0).astype(int) == 0]

    if config.exclude_limit_up and "limit_up" in universe.columns:
        universe = universe.loc[universe["limit_up"].fillna(0).astype(int) == 0]

    if config.exclude_limit_down and "limit_down" in universe.columns:
        universe = universe.loc[universe["limit_down"].fillna(0).astype(int) == 0]

    if config.exclude_bj:
        universe = universe.loc[~universe["code"].astype(str).str.endswith(".BJ")]

    if config.exclude_star:
        universe = universe.loc[~universe["code"].astype(str).str.startswith("688")]

    if config.exclude_chinext:
        universe = universe.loc[
            ~universe["code"].astype(str).str.startswith(("300", "301"))
        ]

    if "close" in universe.columns and config.min_price > 0:
        universe = universe.loc[universe["close"].fillna(0) >= config.min_price]

    if "close" in universe.columns and config.max_price > 0:
        universe = universe.loc[universe["close"].fillna(float("inf")) <= config.max_price]

    if "amount" in universe.columns and config.min_amount > 0:
        universe = universe.loc[universe["amount"].fillna(0) >= config.min_amount]

    return universe.sort_values("code").reset_index(drop=True)
