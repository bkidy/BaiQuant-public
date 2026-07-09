import pandas as pd

from baiquant.research.regime_router import (
    RegimePolicy,
    build_regime_router_signals,
    classify_regime_frame,
)


def test_classify_regime_frame_splits_bull_structural_and_risk_states() -> None:
    regime = pd.DataFrame(
        [
            {
                "date": "2026-01-02",
                "breadth_ma20": 0.62,
                "market_equity": 1.20,
                "market_ma60": 1.10,
                "market_ma120": 1.05,
            },
            {
                "date": "2026-01-03",
                "breadth_ma20": 0.45,
                "market_equity": 1.08,
                "market_ma60": 1.10,
                "market_ma120": 1.02,
            },
            {
                "date": "2026-01-04",
                "breadth_ma20": 0.30,
                "market_equity": 1.00,
                "market_ma60": 1.10,
                "market_ma120": 1.05,
            },
            {
                "date": "2026-01-05",
                "breadth_ma20": 0.20,
                "market_equity": 0.92,
                "market_ma60": 1.00,
                "market_ma120": 1.03,
            },
            {
                "date": "2026-01-06",
                "breadth_ma20": 0.12,
                "market_equity": 0.88,
                "market_ma60": 0.98,
                "market_ma120": 1.02,
            },
        ]
    )

    classified = classify_regime_frame(regime)

    assert classified["regime"].tolist() == [
        "bull",
        "structural",
        "weak_range",
        "bear_weak",
        "extreme_risk",
    ]


def test_build_regime_router_signals_uses_policy_strategy_and_rank_limit() -> None:
    hybrid = pd.DataFrame(
        [
            {"date": "2026-01-02", "code": "H1", "score_rank": 1, "score": 3.0},
            {"date": "2026-01-02", "code": "H2", "score_rank": 2, "score": 2.0},
            {"date": "2026-01-03", "code": "H3", "score_rank": 1, "score": 1.5},
            {"date": "2026-01-04", "code": "H4", "score_rank": 1, "score": 1.0},
        ]
    )
    turbo = pd.DataFrame(
        [
            {"date": "2026-01-02", "code": "T1", "score_rank": 1, "score": 4.0},
            {"date": "2026-01-02", "code": "T2", "score_rank": 2, "score": 3.5},
            {"date": "2026-01-03", "code": "T3", "score_rank": 1, "score": 2.5},
        ]
    )
    regimes = pd.DataFrame(
        [
            {"date": "2026-01-02", "regime": "bull"},
            {"date": "2026-01-03", "regime": "structural"},
            {"date": "2026-01-04", "regime": "bear_weak"},
        ]
    )
    policies = {
        "bull": RegimePolicy(strategy="turbo", max_signals=1),
        "structural": RegimePolicy(strategy="hybrid", max_signals=2),
        "bear_weak": RegimePolicy(strategy="cash", max_signals=0),
    }

    routed = build_regime_router_signals(
        {"hybrid": hybrid, "turbo": turbo},
        regimes,
        policies,
    )

    assert routed[["date", "code", "route_strategy", "regime", "_signal_order"]].to_dict("records") == [
        {
            "date": pd.Timestamp("2026-01-02"),
            "code": "T1",
            "route_strategy": "turbo",
            "regime": "bull",
            "_signal_order": 1,
        },
        {
            "date": pd.Timestamp("2026-01-03"),
            "code": "H3",
            "route_strategy": "hybrid",
            "regime": "structural",
            "_signal_order": 1,
        },
    ]
