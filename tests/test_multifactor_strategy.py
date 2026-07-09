import pandas as pd

from baiquant.strategy.multifactor import MultifactorPlanConfig, build_multifactor_daily_plan


def test_multifactor_daily_plan_sells_stop_loss_and_reuses_planned_sell_cash() -> None:
    prices = pd.DataFrame(
        [
            {"date": "2026-07-01", "code": "AAA", "close": 9.5},
            {"date": "2026-07-01", "code": "BBB", "close": 20.0},
        ]
    )
    holdings = pd.DataFrame(
        [{"code": "AAA", "name": "Alpha", "shares": 1000, "average_cost": 10.0, "entry_date": "2026-06-01"}]
    )
    candidates = pd.DataFrame(
        [{"code": "BBB", "name": "Beta", "industry": "医药", "close": 20.0, "candidate_rank": 1, "multi_factor_score": 3.2}]
    )

    plan = build_multifactor_daily_plan(
        candidates,
        prices,
        as_of="2026-07-01",
        holdings=holdings,
        cash=0,
        config=MultifactorPlanConfig(max_positions=1, stop_loss_pct=0.04),
    )

    assert plan[["action", "code", "reason", "shares"]].to_dict("records") == [
        {"action": "sell_next_open", "code": "AAA", "reason": "single_stop", "shares": 1000},
        {"action": "buy_next_open", "code": "BBB", "reason": "entry", "shares": 400},
    ]
    assert plan.loc[plan["code"] == "BBB", "cash_budget"].iloc[0] == 8000.0


def test_multifactor_daily_plan_holds_existing_position_and_blocks_extra_buy() -> None:
    prices = pd.DataFrame(
        [
            {"date": "2026-07-01", "code": "AAA", "close": 10.3},
            {"date": "2026-07-01", "code": "BBB", "close": 20.0},
        ]
    )
    holdings = pd.DataFrame(
        [{"code": "AAA", "name": "Alpha", "shares": 1000, "average_cost": 10.0, "entry_date": "2026-07-01"}]
    )
    candidates = pd.DataFrame(
        [{"code": "BBB", "name": "Beta", "industry": "医药", "close": 20.0, "candidate_rank": 1, "multi_factor_score": 3.2}]
    )

    plan = build_multifactor_daily_plan(
        candidates,
        prices,
        as_of="2026-07-01",
        holdings=holdings,
        cash=50_000,
        config=MultifactorPlanConfig(max_positions=1),
    )

    assert plan["action"].tolist() == ["hold"]
    assert plan.loc[0, "code"] == "AAA"
    assert plan.loc[0, "reason"] == "existing_position"


def test_multifactor_daily_plan_marks_missing_holding_price_for_manual_review() -> None:
    prices = pd.DataFrame([{"date": "2026-07-01", "code": "BBB", "close": 20.0}])
    holdings = pd.DataFrame([{"code": "ETF001", "name": "ETF", "shares": 100, "average_cost": 1.0}])
    candidates = pd.DataFrame(
        [{"code": "BBB", "name": "Beta", "industry": "医药", "close": 20.0, "candidate_rank": 1, "multi_factor_score": 3.2}]
    )

    plan = build_multifactor_daily_plan(
        candidates,
        prices,
        as_of="2026-07-01",
        holdings=holdings,
        cash=5_000,
        config=MultifactorPlanConfig(max_positions=1),
    )

    assert plan[["action", "code", "reason"]].to_dict("records") == [
        {"action": "manual_review", "code": "ETF001", "reason": "missing_price"},
        {"action": "buy_next_open", "code": "BBB", "reason": "entry"},
    ]
