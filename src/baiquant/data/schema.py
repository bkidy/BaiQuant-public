from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


PRICE_COLUMN_ORDER = [
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
]
PRICE_COLUMNS = set(PRICE_COLUMN_ORDER)

FUNDAMENTAL_COLUMN_ORDER = [
    "date",
    "code",
    "pe_ttm",
    "pb",
    "roe",
    "revenue_yoy",
    "profit_yoy",
]
FUNDAMENTAL_COLUMNS = set(FUNDAMENTAL_COLUMN_ORDER)

STOCK_COLUMN_ORDER = ["code", "name", "industry", "list_date", "is_st"]
STOCK_COLUMNS = set(STOCK_COLUMN_ORDER)

EVENT_COLUMN_ORDER = ["date", "code", "event_type", "sentiment"]
EVENT_COLUMNS = set(EVENT_COLUMN_ORDER)

MONEY_FLOW_COLUMN_ORDER = [
    "date",
    "code",
    "main_net_inflow",
    "small_net_inflow",
    "medium_net_inflow",
    "large_net_inflow",
    "super_large_net_inflow",
    "main_net_inflow_pct",
    "small_net_inflow_pct",
    "medium_net_inflow_pct",
    "large_net_inflow_pct",
    "super_large_net_inflow_pct",
    "close",
    "pct_change",
]
MONEY_FLOW_COLUMNS = set(MONEY_FLOW_COLUMN_ORDER)


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source: str | Path) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing required columns: {', '.join(missing)}")


def parse_date_column(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if column in frame.columns:
        frame = frame.copy()
        frame[column] = pd.to_datetime(frame[column])
    return frame


def sort_by_date_code(frame: pd.DataFrame) -> pd.DataFrame:
    if {"date", "code"}.issubset(frame.columns):
        return frame.sort_values(["date", "code"]).reset_index(drop=True)
    return frame.reset_index(drop=True)
