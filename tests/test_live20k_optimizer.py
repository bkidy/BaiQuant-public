import pandas as pd

from baiquant.backtest import MultiPositionTrendConfig
from baiquant.research.live20k_optimizer import (
    ExecutionVariant,
    default_live20k_execution_variants,
    evaluate_execution_variants,
)


def test_evaluate_execution_variants_ranks_by_recent_and_ytd_return() -> None:
    prices = pd.DataFrame(
        [
            ["2026-01-01", "A", 10.0, 10.0, 0, 0, 0],
            ["2026-01-02", "A", 10.0, 10.0, 0, 0, 0],
            ["2026-01-03", "A", 12.0, 12.0, 0, 0, 0],
            ["2026-01-04", "A", 13.0, 13.0, 0, 0, 0],
        ],
        columns=["date", "code", "open", "close", "paused", "limit_up", "limit_down"],
    )
    signals = pd.DataFrame([["2026-01-01", "A"]], columns=["date", "code"])

    leaderboard = evaluate_execution_variants(
        prices,
        signals,
        variants=[
            ExecutionVariant("blocked", MultiPositionTrendConfig(initial_cash=10_000, max_positions=0)),
            ExecutionVariant(
                "active",
                MultiPositionTrendConfig(
                    initial_cash=10_000,
                    max_positions=1,
                    fee_bps=0,
                    slippage_bps=0,
                    ma_window=20,
                ),
            ),
        ],
        start="2026-01-01",
        end="2026-01-04",
        recent_start="2026-01-02",
    )

    assert leaderboard["name"].tolist() == ["active", "blocked"]
    assert leaderboard.loc[0, "ytd_return"] > 0
    assert leaderboard.loc[0, "recent_return"] > 0
    assert leaderboard.loc[0, "filled_trades"] == 1
    assert {"ma_window", "max_positions", "score"}.issubset(leaderboard.columns)


def test_default_execution_variants_match_their_names() -> None:
    variants = {item.name: item.config for item in default_live20k_execution_variants(MultiPositionTrendConfig())}

    assert variants["ma8_take30_no_trail_pos3"].ma_window == 8
    assert variants["ma8_take30_no_trail_pos3"].max_positions == 3
    assert variants["ma8_take30_no_trail_pos3"].take_profit_pct == 0.30
    assert variants["ma8_no_take_no_trail_pos3"].take_profit_pct == 0.0
    assert variants["ma20_take30_no_trail_pos3"].ma_window == 20
    assert variants["ma20_take30_no_trail_pos3"].max_positions == 3
