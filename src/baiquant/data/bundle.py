from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame()


@dataclass(slots=True)
class MarketDataBundle:
    """Container for the data tables used by a research run."""

    prices: pd.DataFrame
    fundamentals: pd.DataFrame = field(default_factory=empty_frame)
    stocks: pd.DataFrame = field(default_factory=empty_frame)
    events: pd.DataFrame = field(default_factory=empty_frame)
    money_flow: pd.DataFrame = field(default_factory=empty_frame)
