import pandas as pd

from baiquant.data.bundle import MarketDataBundle
from baiquant.pipeline import PipelineConfig, run_selection
from baiquant.scoring import FactorSpec
from baiquant.universe.filters import UniverseConfig


def _selection_bundle() -> MarketDataBundle:
    rows = []
    for i, date in enumerate(pd.date_range("2026-01-01", periods=40, freq="D"), start=1):
        rows.append([date, "FAST", i * 1.0, i * 1.0, i * 1.0, i * 1.0, 2_000, 2_000_000, 0, 0, 0])
        rows.append([date, "SLOW", 50.0 - i * 0.5, 50.0 - i * 0.5, 50.0 - i * 0.5, 50.0 - i * 0.5, 1_000, 1_000_000, 0, 0, 0])
    prices = pd.DataFrame(
        rows,
        columns=[
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
        ],
    )
    fundamentals = pd.DataFrame(
        [
            ["2026-02-09", "FAST", 18, 2.0, 0.20, 0.35, 0.50],
            ["2026-02-09", "SLOW", 40, 6.0, 0.05, -0.10, -0.20],
        ],
        columns=["date", "code", "pe_ttm", "pb", "roe", "revenue_yoy", "profit_yoy"],
    )
    fundamentals["date"] = pd.to_datetime(fundamentals["date"])
    stocks = pd.DataFrame(
        [
            ["FAST", "Fast Co", "Electronics", "2020-01-01", 0],
            ["SLOW", "Slow Co", "Retail", "2020-01-01", 0],
        ],
        columns=["code", "name", "industry", "list_date", "is_st"],
    )
    stocks["list_date"] = pd.to_datetime(stocks["list_date"])
    return MarketDataBundle(prices=prices, fundamentals=fundamentals, stocks=stocks)


def test_run_selection_scores_factors_and_builds_equal_weight_portfolio() -> None:
    config = PipelineConfig(
        universe=UniverseConfig(min_listed_days=180, min_amount=10_000),
        factors=[
            FactorSpec(name="momentum_20d", weight=1.0, direction=1),
            FactorSpec(name="quality", weight=1.0, direction=1),
            FactorSpec(name="value", weight=0.5, direction=1),
        ],
        max_positions=1,
        min_factor_hits=2,
    )

    result = run_selection(_selection_bundle(), as_of="2026-02-09", config=config)

    assert result.selected["code"].tolist() == ["FAST"]
    row = result.selected.iloc[0]
    assert row["rank"] == 1
    assert row["weight"] == 1.0
    assert row["hits"] >= 2
    assert row["score"] > 0


def test_run_selection_can_build_lot_sized_portfolio_for_small_account() -> None:
    config = PipelineConfig(
        universe=UniverseConfig(min_listed_days=180, min_amount=10_000, max_price=80),
        factors=[
            FactorSpec(name="momentum_20d", weight=1.0, direction=1),
            FactorSpec(name="quality", weight=1.0, direction=1),
        ],
        max_positions=3,
        min_factor_hits=1,
        capital=50_000,
        lot_size=100,
        cash_buffer_pct=0.05,
        max_position_pct=0.25,
        max_industry_positions=1,
    )

    result = run_selection(_selection_bundle(), as_of="2026-02-09", config=config)

    assert "shares" in result.selected.columns
    assert result.selected["shares"].mod(100).eq(0).all()
    assert result.selected["position_value"].sum() <= 50_000 * 0.95
