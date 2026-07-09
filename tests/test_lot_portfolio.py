import pandas as pd

from baiquant.portfolio import build_lot_sized_portfolio


def test_lot_sized_portfolio_respects_cash_lot_and_industry_limits() -> None:
    scored = pd.DataFrame(
        [
            ["A", "Tech", 50.0, 9.0, 5],
            ["B", "Tech", 30.0, 8.0, 5],
            ["C", "Health", 140.0, 7.0, 5],
            ["D", "Energy", 20.0, 6.0, 5],
        ],
        columns=["code", "industry", "close", "score", "hits"],
    )

    selected = build_lot_sized_portfolio(
        scored,
        capital=50_000,
        max_positions=3,
        min_factor_hits=2,
        lot_size=100,
        cash_buffer_pct=0.05,
        max_position_pct=0.25,
        max_industry_positions=1,
    )

    assert selected["code"].tolist() == ["A", "D"]
    assert selected.loc[selected["code"] == "A", "shares"].iloc[0] == 200
    assert selected.loc[selected["code"] == "A", "position_value"].iloc[0] == 10_000
    assert selected["position_value"].sum() <= 50_000 * 0.95
    assert selected["weight"].sum() <= 0.95
    assert selected["industry"].tolist() == ["Tech", "Energy"]


def test_lot_sized_portfolio_can_start_from_later_candidate_rank() -> None:
    scored = pd.DataFrame(
        [
            ["A", "Tech", 10.0, 10.0, 5],
            ["B", "Tech", 10.0, 9.0, 5],
            ["C", "Health", 10.0, 8.0, 5],
            ["D", "Energy", 10.0, 7.0, 5],
        ],
        columns=["code", "industry", "close", "score", "hits"],
    )

    selected = build_lot_sized_portfolio(
        scored,
        capital=50_000,
        max_positions=2,
        min_factor_hits=1,
        lot_size=100,
        rank_start=3,
    )

    assert selected["code"].tolist() == ["C", "D"]
    assert selected["rank"].tolist() == [1, 2]
