from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from baiquant.data.bundle import MarketDataBundle
from baiquant.data.schema import (
    EVENT_COLUMNS,
    FUNDAMENTAL_COLUMNS,
    MONEY_FLOW_COLUMNS,
    PRICE_COLUMNS,
    STOCK_COLUMNS,
    parse_date_column,
    require_columns,
    sort_by_date_code,
)


@dataclass(slots=True)
class CsvDataProvider:
    """Load the framework's normalized CSV tables from a directory."""

    root: str | Path

    def load(self) -> MarketDataBundle:
        root = Path(self.root)
        prices = self._read_required(root / "prices.csv", PRICE_COLUMNS)
        prices = sort_by_date_code(parse_date_column(prices, "date"))

        fundamentals = self._read_optional(root / "fundamentals.csv", FUNDAMENTAL_COLUMNS)
        fundamentals = sort_by_date_code(parse_date_column(fundamentals, "date"))

        stocks = self._read_optional(root / "stocks.csv", STOCK_COLUMNS)
        stocks = parse_date_column(stocks, "list_date").reset_index(drop=True)

        events = self._read_optional(root / "events.csv", EVENT_COLUMNS)
        events = sort_by_date_code(parse_date_column(events, "date"))

        money_flow = self._read_optional(root / "money_flow.csv", MONEY_FLOW_COLUMNS)
        money_flow = sort_by_date_code(parse_date_column(money_flow, "date"))

        return MarketDataBundle(
            prices=prices,
            fundamentals=fundamentals,
            stocks=stocks,
            events=events,
            money_flow=money_flow,
        )

    @staticmethod
    def _read_required(path: Path, columns: set[str]) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Required data file not found: {path}")
        frame = pd.read_csv(path)
        require_columns(frame, columns, path)
        return frame

    @staticmethod
    def _read_optional(path: Path, columns: set[str]) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame(columns=sorted(columns))
        frame = pd.read_csv(path)
        require_columns(frame, columns, path)
        return frame
