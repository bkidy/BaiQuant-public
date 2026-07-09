import pandas as pd
import pytest

from baiquant.data.bundle import MarketDataBundle
from baiquant.research.multifactor import (
    build_multifactor_frame,
    derive_validated_multifactor_weights,
    evaluate_multifactor_factors,
    generate_multifactor_factor_history,
    generate_multifactor_signals,
    score_multifactor_frame,
    select_multifactor_candidates,
)


def _sample_bundle() -> MarketDataBundle:
    rows = []
    for index, date in enumerate(pd.date_range("2026-01-01", periods=8, freq="D"), start=0):
        rows.extend(
            [
                {
                    "date": date,
                    "code": "AAA",
                    "open": 10 + index,
                    "high": 10.5 + index,
                    "low": 9.5 + index,
                    "close": 10 + index,
                    "volume": 1_000 + index * 100,
                    "amount": 10_000 + index * 1_000,
                    "paused": 0,
                    "limit_up": 0,
                    "limit_down": 0,
                },
                {
                    "date": date,
                    "code": "BBB",
                    "open": 20 - index,
                    "high": 20.5 - index,
                    "low": 19.5 - index,
                    "close": 20 - index,
                    "volume": 2_000 - index * 50,
                    "amount": 20_000 - index * 500,
                    "paused": 0,
                    "limit_up": 0,
                    "limit_down": 0,
                },
                {
                    "date": date,
                    "code": "CCC",
                    "open": 8 + index * 0.2,
                    "high": 8.4 + index * 0.2,
                    "low": 7.8 + index * 0.2,
                    "close": 8 + index * 0.2,
                    "volume": 500 + index * 10,
                    "amount": 4_000 + index * 100,
                    "paused": 0,
                    "limit_up": 0,
                    "limit_down": 0,
                },
            ]
        )
    prices = pd.DataFrame(rows)
    stocks = pd.DataFrame(
        [
            {"code": "AAA", "name": "Alpha", "industry": "半导体", "list_date": "2020-01-01", "is_st": 0},
            {"code": "BBB", "name": "Beta", "industry": "医药", "list_date": "2020-01-01", "is_st": 0},
            {"code": "CCC", "name": "Gamma", "industry": "医药", "list_date": "2020-01-01", "is_st": 0},
        ]
    )
    money_flow = pd.DataFrame(
        [
            {
                "date": "2026-01-08",
                "code": "AAA",
                "main_net_inflow": 1_000,
                "main_net_inflow_pct": 5.0,
                "large_net_inflow_pct": 2.0,
                "super_large_net_inflow_pct": 1.0,
            },
            {
                "date": "2026-01-08",
                "code": "BBB",
                "main_net_inflow": -500,
                "main_net_inflow_pct": -2.0,
                "large_net_inflow_pct": -1.0,
                "super_large_net_inflow_pct": -1.0,
            },
        ]
    )
    fundamentals = pd.DataFrame(
        [
            {"date": "2026-01-08", "code": "AAA", "pe_ttm": 20.0, "pb": 2.0, "roe": 0.12, "revenue_yoy": 0.2, "profit_yoy": 0.3},
            {"date": "2026-01-08", "code": "BBB", "pe_ttm": 50.0, "pb": 4.0, "roe": 0.05, "revenue_yoy": -0.1, "profit_yoy": -0.2},
        ]
    )
    return MarketDataBundle(prices=prices, stocks=stocks, money_flow=money_flow, fundamentals=fundamentals)


def test_build_multifactor_frame_exposes_explainable_factor_columns() -> None:
    factors = build_multifactor_frame(_sample_bundle(), "2026-01-08")

    row = factors.set_index("code").loc["AAA"]
    assert row["name"] == "Alpha"
    assert row["industry"] == "半导体"
    assert row["momentum_3d"] == pytest.approx(3 / 14)
    assert row["momentum_5d"] == pytest.approx(5 / 12)
    assert row["money_flow_pct"] == 5.0
    assert row["big_order_pct"] == 3.0
    assert row["roe"] == pytest.approx(0.12)
    assert row["profit_yoy"] == pytest.approx(0.3)
    assert row["pe_ttm"] == pytest.approx(20.0)
    assert row["pb"] == pytest.approx(2.0)
    assert row["industry_momentum_3d"] > 0
    assert 0 <= row["close_position_20d"] <= 1


def test_evaluate_multifactor_factors_summarizes_ic_and_top_slices() -> None:
    factors = pd.DataFrame(
        [
            ["2026-01-01", "AAA", 3.0, 1.0],
            ["2026-01-01", "BBB", 2.0, 2.0],
            ["2026-01-01", "CCC", 1.0, 3.0],
            ["2026-01-02", "AAA", 1.0, 3.0],
            ["2026-01-02", "BBB", 2.0, 2.0],
            ["2026-01-02", "CCC", 3.0, 1.0],
        ],
        columns=["date", "code", "good_factor", "bad_factor"],
    )
    forward = pd.DataFrame(
        [
            ["2026-01-01", "AAA", 0.30],
            ["2026-01-01", "BBB", 0.10],
            ["2026-01-01", "CCC", -0.10],
            ["2026-01-02", "AAA", -0.10],
            ["2026-01-02", "BBB", 0.10],
            ["2026-01-02", "CCC", 0.30],
        ],
        columns=["date", "code", "forward_return"],
    )

    report = evaluate_multifactor_factors(
        factors,
        forward,
        factor_names=["good_factor", "bad_factor"],
        slices=[("top_1", 1, 1)],
    )

    by_factor = report.set_index("factor")
    assert by_factor.loc["good_factor", "mean_rank_ic"] == pytest.approx(1.0)
    assert by_factor.loc["good_factor", "valid_factor_rows"] == 6
    assert by_factor.loc["good_factor", "factor_coverage_rate"] == pytest.approx(1.0)
    assert by_factor.loc["good_factor", "top_1_mean_forward_return"] == pytest.approx(0.30)
    assert by_factor.loc["bad_factor", "mean_rank_ic"] == pytest.approx(-1.0)
    assert by_factor.loc["bad_factor", "top_1_mean_forward_return"] == pytest.approx(-0.10)


def test_score_multifactor_frame_ranks_candidates_with_factor_contributions() -> None:
    factors = pd.DataFrame(
        [
            {
                "date": "2026-01-08",
                "code": "AAA",
                "name": "Alpha",
                "industry": "半导体",
                "momentum_5d": 0.20,
                "money_flow_pct": 5.0,
                "volatility_20d": 0.01,
                "pe_ttm": 20.0,
            },
            {
                "date": "2026-01-08",
                "code": "BBB",
                "name": "Beta",
                "industry": "医药",
                "momentum_5d": -0.05,
                "money_flow_pct": -2.0,
                "volatility_20d": 0.08,
                "pe_ttm": 50.0,
            },
            {
                "date": "2026-01-08",
                "code": "CCC",
                "name": "Gamma",
                "industry": "医药",
                "momentum_5d": 0.03,
                "money_flow_pct": 1.0,
                "volatility_20d": 0.03,
                "pe_ttm": 25.0,
            },
        ]
    )

    scored = score_multifactor_frame(
        factors,
        factor_weights={
            "momentum_5d": 1.0,
            "money_flow_pct": 1.0,
            "volatility_20d": -0.5,
            "pe_ttm": -0.3,
        },
        top_n=2,
    )

    assert scored["code"].tolist() == ["AAA", "CCC"]
    assert scored.loc[0, "score_rank"] == 1
    assert scored.loc[0, "factor_hits"] >= scored.loc[1, "factor_hits"]
    assert "momentum_5d" in scored.loc[0, "positive_factors"]
    assert "money_flow_pct" in scored.loc[0, "positive_factors"]


def test_derive_validated_multifactor_weights_uses_coverage_and_diagnostics() -> None:
    diagnostics = pd.DataFrame(
        [
            {
                "factor": "momentum_20d",
                "factor_coverage_rate": 1.0,
                "mean_rank_ic": 0.05,
                "ic_observations": 6,
                "top_1_5_mean_forward_return": 0.02,
            },
            {
                "factor": "money_flow_pct",
                "factor_coverage_rate": 0.95,
                "mean_rank_ic": -0.02,
                "ic_observations": 6,
                "top_1_5_mean_forward_return": 0.12,
            },
            {
                "factor": "pe_ttm",
                "factor_coverage_rate": 0.90,
                "mean_rank_ic": 0.03,
                "ic_observations": 6,
                "top_1_5_mean_forward_return": 0.06,
            },
            {
                "factor": "roe",
                "factor_coverage_rate": 0.0,
                "mean_rank_ic": pd.NA,
                "ic_observations": 0,
                "top_1_5_mean_forward_return": pd.NA,
            },
        ]
    )

    weights = derive_validated_multifactor_weights(
        diagnostics,
        {
            "momentum_20d": 1.0,
            "money_flow_pct": 0.7,
            "pe_ttm": -0.25,
            "roe": 0.35,
        },
        min_coverage_rate=0.8,
        min_ic_observations=3,
    )

    assert weights["momentum_20d"] == pytest.approx(1.0)
    assert weights["money_flow_pct"] == pytest.approx(0.7)
    assert weights["pe_ttm"] == pytest.approx(-0.25)
    assert weights["roe"] == 0.0


def test_select_multifactor_candidates_limits_industry_crowding_and_lot_cost() -> None:
    scored = pd.DataFrame(
        [
            {"score_rank": 1, "code": "AAA", "industry": "半导体", "close": 80.0, "multi_factor_score": 9.0},
            {"score_rank": 2, "code": "BBB", "industry": "半导体", "close": 90.0, "multi_factor_score": 8.5},
            {"score_rank": 3, "code": "CCC", "industry": "半导体", "close": 70.0, "multi_factor_score": 8.0},
            {"score_rank": 4, "code": "DDD", "industry": "医药", "close": 260.0, "multi_factor_score": 7.5},
            {"score_rank": 5, "code": "EEE", "industry": "医药", "close": 35.0, "multi_factor_score": 7.0},
            {"score_rank": 6, "code": "FFF", "industry": "化工", "close": 20.0, "multi_factor_score": 6.5},
        ]
    )

    selected = select_multifactor_candidates(
        scored,
        top_n=4,
        max_per_industry=2,
        max_lot_cost=25_000,
    )

    assert selected["code"].tolist() == ["AAA", "BBB", "EEE", "FFF"]
    assert selected["candidate_rank"].tolist() == [1, 2, 3, 4]
    assert selected["industry"].value_counts().to_dict()["半导体"] == 2
    assert "DDD" not in selected["code"].tolist()


def test_generate_multifactor_signals_uses_fixed_rebalance_dates() -> None:
    signals = generate_multifactor_signals(
        _sample_bundle(),
        start="2026-01-06",
        end="2026-01-08",
        factor_weights={"momentum_5d": 1.0, "money_flow_pct": 0.5},
        top_n=1,
        signal_every_n_days=2,
        max_per_industry=1,
        max_lot_cost=25_000,
    )

    assert signals["date"].tolist() == [pd.Timestamp("2026-01-06"), pd.Timestamp("2026-01-08")]
    assert signals["candidate_rank"].tolist() == [1, 1]
    assert signals["position_scale"].tolist() == [1.0, 1.0]
    assert {"code", "name", "industry", "multi_factor_score"}.issubset(signals.columns)


def test_generate_multifactor_signals_can_filter_by_market_regime() -> None:
    regimes = pd.DataFrame(
        [
            {"date": "2026-01-06", "regime": "bear_weak"},
            {"date": "2026-01-08", "regime": "structural"},
        ]
    )

    signals = generate_multifactor_signals(
        _sample_bundle(),
        start="2026-01-06",
        end="2026-01-08",
        factor_weights={"momentum_5d": 1.0, "money_flow_pct": 0.5},
        top_n=1,
        signal_every_n_days=2,
        max_per_industry=1,
        max_lot_cost=25_000,
        regimes=regimes,
        allowed_regimes=("structural",),
    )

    assert signals["date"].tolist() == [pd.Timestamp("2026-01-08")]
    assert signals["regime"].tolist() == ["structural"]


def test_generate_multifactor_factor_history_samples_signal_dates() -> None:
    factors = generate_multifactor_factor_history(
        _sample_bundle(),
        start="2026-01-06",
        end="2026-01-08",
        signal_every_n_days=2,
        factor_names=["momentum_5d", "money_flow_pct"],
    )

    assert factors["date"].drop_duplicates().tolist() == [pd.Timestamp("2026-01-06"), pd.Timestamp("2026-01-08")]
    assert {"date", "code", "name", "industry", "close", "momentum_5d", "money_flow_pct"}.issubset(factors.columns)
    assert "momentum_20d" not in factors.columns
