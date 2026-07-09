from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from baiquant.data.bundle import MarketDataBundle
from baiquant.factors.registry import compute_factor_frame
from baiquant.portfolio import build_equal_weight_portfolio, build_lot_sized_portfolio
from baiquant.scoring import FactorSpec, score_factor_frame
from baiquant.universe.filters import UniverseConfig, build_universe


@dataclass(slots=True)
class PipelineConfig:
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    factors: list[FactorSpec] = field(default_factory=list)
    max_positions: int = 20
    min_factor_hits: int = 1
    capital: float | None = None
    lot_size: int = 100
    cash_buffer_pct: float = 0.0
    max_position_pct: float = 1.0
    max_industry_positions: int = 0
    rank_start: int = 1


@dataclass(slots=True)
class SelectionResult:
    as_of: pd.Timestamp
    universe: pd.DataFrame
    factors: pd.DataFrame
    scored: pd.DataFrame
    selected: pd.DataFrame


def run_selection(
    bundle: MarketDataBundle,
    as_of: str | pd.Timestamp,
    config: PipelineConfig,
) -> SelectionResult:
    cutoff = pd.Timestamp(as_of)
    universe = build_universe(bundle, cutoff, config.universe)
    factor_names = [factor.name for factor in config.factors if factor.enabled]
    factors = compute_factor_frame(bundle, universe, factor_names, cutoff)
    enriched = factors.merge(
        universe[[column for column in ["code", "name", "industry", "close", "amount"] if column in universe.columns]],
        on="code",
        how="left",
    )
    scored = score_factor_frame(enriched, config.factors)
    if config.capital is not None:
        selected = build_lot_sized_portfolio(
            scored,
            capital=config.capital,
            max_positions=config.max_positions,
            min_factor_hits=config.min_factor_hits,
            lot_size=config.lot_size,
            cash_buffer_pct=config.cash_buffer_pct,
            max_position_pct=config.max_position_pct,
            max_industry_positions=config.max_industry_positions,
            rank_start=config.rank_start,
        )
    else:
        selected = build_equal_weight_portfolio(
            scored,
            max_positions=config.max_positions,
            min_factor_hits=config.min_factor_hits,
            rank_start=config.rank_start,
        )
    return SelectionResult(
        as_of=cutoff,
        universe=universe,
        factors=factors,
        scored=scored,
        selected=selected,
    )
