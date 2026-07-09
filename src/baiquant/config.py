from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from baiquant.pipeline import PipelineConfig
from baiquant.scoring import FactorSpec
from baiquant.universe.filters import UniverseConfig


@dataclass(slots=True)
class DataConfig:
    kind: str
    path: Path


def load_pipeline_config(path: str | Path) -> tuple[DataConfig, PipelineConfig]:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    data_raw = raw.get("data", {})
    kind = str(data_raw.get("kind", "csv")).lower()
    data_path = Path(data_raw.get("path", data_raw.get("root", "data")))
    if not data_path.is_absolute():
        data_path = (config_path.parent / data_path).resolve()

    universe_raw = raw.get("universe", {})
    universe = UniverseConfig(
        min_listed_days=int(universe_raw.get("min_listed_days", 180)),
        min_history_days=int(universe_raw.get("min_history_days", 0)),
        min_amount=float(universe_raw.get("min_amount", 0)),
        exclude_st=bool(universe_raw.get("exclude_st", True)),
        exclude_paused=bool(universe_raw.get("exclude_paused", True)),
        exclude_limit_up=bool(universe_raw.get("exclude_limit_up", False)),
        exclude_limit_down=bool(universe_raw.get("exclude_limit_down", True)),
        exclude_bj=bool(universe_raw.get("exclude_bj", False)),
        exclude_star=bool(universe_raw.get("exclude_star", False)),
        exclude_chinext=bool(universe_raw.get("exclude_chinext", False)),
        min_price=float(universe_raw.get("min_price", 0)),
        max_price=float(universe_raw.get("max_price", 0)),
    )

    factors = [
        FactorSpec(
            name=str(item["name"]),
            weight=float(item.get("weight", 1.0)),
            direction=int(item.get("direction", 1)),
            enabled=bool(item.get("enabled", True)),
        )
        for item in raw.get("factors", [])
    ]

    portfolio_raw = raw.get("portfolio", {})
    pipeline = PipelineConfig(
        universe=universe,
        factors=factors,
        max_positions=int(portfolio_raw.get("max_positions", 20)),
        min_factor_hits=int(portfolio_raw.get("min_factor_hits", 1)),
        capital=float(portfolio_raw["capital"]) if "capital" in portfolio_raw else None,
        lot_size=int(portfolio_raw.get("lot_size", 100)),
        cash_buffer_pct=float(portfolio_raw.get("cash_buffer_pct", 0)),
        max_position_pct=float(portfolio_raw.get("max_position_pct", 1)),
        max_industry_positions=int(portfolio_raw.get("max_industry_positions", 0)),
        rank_start=int(portfolio_raw.get("rank_start", 1)),
    )
    return DataConfig(kind=kind, path=data_path), pipeline
