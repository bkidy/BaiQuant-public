import json
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from threading import Thread

import pandas as pd
import pytest

from baiquant.data.bundle import MarketDataBundle
from baiquant.desk import (
    DESK_MANUAL_20D_PRESET,
    DESK_MULTIFACTOR_PRESET,
    DESK_TURBO_SPRINT_PRESET,
    DESK_HTML,
    DeskDefaults,
    _normalize_desk_preset,
    _make_handler,
    _desk_plan_columns,
    _desk_watchlist_columns,
    _build_multifactor_strategy_snapshot,
    _quick_plan_from_watchlist,
    build_desk_account_snapshot,
    build_desk_positions,
    load_desk_holdings,
    save_desk_holdings,
)
from baiquant.strategy.live20k import (
    live100k_hotspot_manual_fixed_execution_config,
    live100k_hotspot_manual_fixed_signal_config,
)


def test_desk_holdings_roundtrip_keeps_manual_position_fields(tmp_path) -> None:
    path = tmp_path / "holdings.csv"
    rows = [
        {
            "code": "600520.SH",
            "name": "三佳科技",
            "shares": 500,
            "average_cost": 31.2,
            "entry_date": "2026-05-26",
        }
    ]

    save_desk_holdings(path, rows)
    holdings = load_desk_holdings(path)

    assert holdings.to_dict("records") == [
        {
            "code": "600520.SH",
            "name": "三佳科技",
            "shares": 500,
            "average_cost": 31.2,
            "high_close": None,
            "entry_shares": 500,
            "added": False,
            "entry_date": "2026-05-26",
        }
    ]


def test_desk_holdings_missing_file_returns_empty_frame(tmp_path) -> None:
    holdings = load_desk_holdings(tmp_path / "missing.csv")

    assert holdings.empty
    assert holdings.columns.tolist() == [
        "code",
        "name",
        "shares",
        "average_cost",
        "high_close",
        "entry_shares",
        "added",
        "entry_date",
    ]


def test_desk_strategy_tables_include_technical_overlay_columns() -> None:
    overlay_columns = {"tech_score", "tech_grade", "trade_advice", "position_scale", "risk_flags"}
    overlay_labels = {"技术分", "档位", "买法", "仓位建议", "风险提示"}

    assert overlay_columns.issubset(_desk_plan_columns())
    assert overlay_columns.issubset(_desk_watchlist_columns())
    for column in overlay_columns:
        assert column in DESK_HTML
    for label in overlay_labels:
        assert label in DESK_HTML
    assert "20天稳打版,短线冲刺版" in DESK_HTML


def test_desk_accepts_public_english_preset_aliases() -> None:
    assert _normalize_desk_preset("steady-20d") == DESK_MANUAL_20D_PRESET
    assert _normalize_desk_preset("turbo-sprint") == DESK_TURBO_SPRINT_PRESET
    assert _normalize_desk_preset("manual-multifactor") == DESK_MULTIFACTOR_PRESET


def test_desk_html_exposes_trade_recording_form() -> None:
    assert "成交记录" in DESK_HTML
    assert "成交流水" in DESK_HTML
    assert "tradeTable" in DESK_HTML
    assert "renderTrades" in DESK_HTML
    assert "recordTrade" in DESK_HTML
    assert "/api/trades" in DESK_HTML


def test_desk_trade_api_lists_recent_trades(tmp_path) -> None:
    trade_log_path = tmp_path / "trade_log.csv"
    pd.DataFrame(
        [
            {"date": "2026-07-01", "action": "buy", "code": "601636.SH", "name": "示例A", "shares": 100, "price": 9.70},
            {"date": "2026-07-02", "action": "sell", "code": "601611.SH", "name": "示例B", "shares": 300, "price": 11.20},
        ]
    ).to_csv(trade_log_path, index=False)
    defaults = DeskDefaults(trade_log_path=str(trade_log_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(defaults))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        connection.request("GET", "/api/trades?limit=1")
        response = connection.getresponse()
        data = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert response.status == 200
    assert data["trade_log_path"] == str(trade_log_path)
    assert [trade["code"] for trade in data["trades"]] == ["601611.SH"]


def test_desk_trade_api_records_fill_and_updates_holdings(tmp_path) -> None:
    holdings_path = tmp_path / "holdings.csv"
    trade_log_path = tmp_path / "trade_log.csv"
    pd.DataFrame([{"code": "601611.SH", "name": "示例B", "shares": 800, "average_cost": 13.678}]).to_csv(
        holdings_path,
        index=False,
    )
    defaults = DeskDefaults(holdings_path=str(holdings_path), trade_log_path=str(trade_log_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(defaults))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        payload = json.dumps(
            {
                "date": "2026-07-02",
                "action": "sell",
                "code": "601611.SH",
                "name": "示例B",
                "shares": 300,
                "price": 11.20,
                "fees": 2,
                "source": "desk_test",
            },
            ensure_ascii=False,
        ).encode("utf-8")
        connection.request("POST", "/api/trades", body=payload, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        data = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert response.status == 200
    assert data["trade"]["action"] == "sell"
    assert data["trade"]["holdings_shares_after"] == 500
    assert data["holdings"][0]["shares"] == 500
    assert pd.read_csv(holdings_path)["shares"].tolist() == [500]
    assert pd.read_csv(trade_log_path)["source"].tolist() == ["desk_test"]


def test_desk_multifactor_snapshot_exposes_plan_and_watchlist(monkeypatch) -> None:
    bundle = MarketDataBundle(prices=pd.DataFrame([{"date": "2026-07-01", "code": "A", "close": 9.5}]))
    holdings = pd.DataFrame([{"code": "A", "shares": 100, "average_cost": 10.0}])

    def build_factors(bundle, as_of):  # noqa: ANN001
        return pd.DataFrame([{"date": as_of, "code": "B", "close": 20.0}])

    def score_factors(factors, factor_weights, top_n=None):  # noqa: ANN001
        return pd.DataFrame(
            [{"score_rank": 1, "code": "B", "name": "Beta", "industry": "医药", "close": 20.0, "multi_factor_score": 3.2}]
        )

    def select_candidates(scored, top_n, max_per_industry=None, max_price=None, max_lot_cost=None):  # noqa: ANN001
        return scored.assign(candidate_rank=1)

    def build_plan(candidates, prices, as_of, holdings=None, cash=None, config=None):  # noqa: ANN001
        assert candidates["code"].tolist() == ["B"]
        return pd.DataFrame(
            [
                {
                    "date": as_of,
                    "action": "sell_next_open",
                    "code": "A",
                    "name": "Alpha",
                    "reason": "single_stop",
                    "shares": 100,
                    "reference_price": 9.5,
                    "cash_budget": 0.0,
                }
            ]
        )

    def build_regime(prices):  # noqa: ANN001
        return pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-07-01"),
                    "regime": "structural",
                    "breadth_ma20": 0.45,
                    "dist_ma60": -0.02,
                }
            ]
        )

    monkeypatch.setattr("baiquant.desk.build_multifactor_frame", build_factors)
    monkeypatch.setattr("baiquant.desk.score_multifactor_frame", score_factors)
    monkeypatch.setattr("baiquant.desk.select_multifactor_candidates", select_candidates)
    monkeypatch.setattr("baiquant.desk.build_multifactor_daily_plan", build_plan)
    monkeypatch.setattr("baiquant.desk.build_regime_frame", build_regime)

    snapshot = _build_multifactor_strategy_snapshot(
        bundle=bundle,
        as_of_date=pd.Timestamp("2026-07-01"),
        holdings=holdings,
        cash=5000,
        watchlist_limit=5,
    )

    assert snapshot["preset"] == DESK_MULTIFACTOR_PRESET
    assert snapshot["entry_gate"] is True
    assert snapshot["market_gate"] is True
    assert snapshot["regime"]["regime"] == "structural"
    assert snapshot["regime"]["breadth_ma20"] == 0.45
    assert snapshot["watchlist"][0]["code"] == "B"
    assert snapshot["watchlist"][0]["candidate_action"] == "buy_candidate"
    assert snapshot["plan"][0]["action"] == "sell_next_open"
    assert snapshot["orders"][0]["reason"] == "single_stop"


def test_desk_multifactor_snapshot_blocks_new_buys_when_regime_is_not_allowed(monkeypatch) -> None:
    bundle = MarketDataBundle(prices=pd.DataFrame([{"date": "2026-07-01", "code": "A", "close": 9.5}]))
    holdings = pd.DataFrame([{"code": "A", "shares": 100, "average_cost": 10.0}])
    seen = {}

    def build_factors(bundle, as_of):  # noqa: ANN001
        return pd.DataFrame([{"date": as_of, "code": "B", "close": 20.0}])

    def score_factors(factors, factor_weights, top_n=None):  # noqa: ANN001
        return pd.DataFrame(
            [{"score_rank": 1, "code": "B", "name": "Beta", "industry": "医药", "close": 20.0, "multi_factor_score": 3.2}]
        )

    def select_candidates(scored, top_n, max_per_industry=None, max_price=None, max_lot_cost=None):  # noqa: ANN001
        return scored.assign(candidate_rank=1)

    def build_regime(prices):  # noqa: ANN001
        return pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-07-01"),
                    "regime": "bear_weak",
                    "breadth_ma20": 0.20,
                    "dist_ma60": -0.10,
                }
            ]
        )

    def build_plan(candidates, prices, as_of, holdings=None, cash=None, config=None):  # noqa: ANN001
        seen["candidate_rows"] = len(candidates)
        return pd.DataFrame(
            [
                {
                    "date": as_of,
                    "action": "hold",
                    "code": "A",
                    "reason": "existing_position",
                    "shares": 100,
                }
            ]
        )

    monkeypatch.setattr("baiquant.desk.build_multifactor_frame", build_factors)
    monkeypatch.setattr("baiquant.desk.score_multifactor_frame", score_factors)
    monkeypatch.setattr("baiquant.desk.select_multifactor_candidates", select_candidates)
    monkeypatch.setattr("baiquant.desk.build_regime_frame", build_regime)
    monkeypatch.setattr("baiquant.desk.build_multifactor_daily_plan", build_plan)

    snapshot = _build_multifactor_strategy_snapshot(
        bundle=bundle,
        as_of_date=pd.Timestamp("2026-07-01"),
        holdings=holdings,
        cash=5000,
        watchlist_limit=5,
    )

    assert snapshot["entry_gate"] is False
    assert snapshot["market_gate"] is False
    assert snapshot["regime"]["regime"] == "bear_weak"
    assert snapshot["watchlist"][0]["candidate_action"] == "watch_only"
    assert seen["candidate_rows"] == 0


def test_desk_positions_are_marked_to_market_from_selected_date() -> None:
    bundle = MarketDataBundle(
        prices=pd.DataFrame(
            [
                {"date": "2026-05-26", "code": "600520.SH", "close": 31.0},
                {"date": "2026-05-27", "code": "600520.SH", "close": 34.0},
                {"date": "2026-05-28", "code": "600520.SH", "close": 33.4},
            ]
        ),
        stocks=pd.DataFrame(
            [
                {
                    "code": "600520.SH",
                    "name": "三佳科技",
                    "industry": "半导体",
                    "list_date": "2000-01-01",
                    "is_st": 0,
                }
            ]
        ),
    )
    holdings = pd.DataFrame(
        [
            {
                "code": "600520.SH",
                "shares": 200,
                "average_cost": 31.2,
                "entry_date": "2026-05-26",
            }
        ]
    )

    positions = build_desk_positions(bundle, "2026-05-28", holdings)
    account = build_desk_account_snapshot(positions, cash=5000)

    row = positions.iloc[0]
    assert row["name"] == "三佳科技"
    assert row["current_price"] == 33.4
    assert row["high_close"] == 34.0
    assert row["market_value"] == 6680.0
    assert row["cost_value"] == 6240.0
    assert row["unrealized_pnl"] == 440.0
    assert row["unrealized_return"] == pytest.approx(0.0705128205)
    assert row["drawdown_from_high"] == pytest.approx(-0.0176470588)
    assert row["holding_days"] == 3
    assert row["stop_signal"] == "hold"
    assert account["equity"] == 11680.0
    assert account["exposure"] == pytest.approx(0.5719178082)


def test_desk_positions_accept_broker_six_digit_codes() -> None:
    bundle = MarketDataBundle(
        prices=pd.DataFrame(
            [
                {"date": "2026-05-28", "code": "688520.SH", "close": 39.46},
            ]
        ),
        stocks=pd.DataFrame(
            [
                {
                    "code": "688520.SH",
                    "name": "示例C",
                    "industry": "医疗保健",
                    "list_date": "2020-01-01",
                    "is_st": 0,
                }
            ]
        ),
    )
    holdings = pd.DataFrame(
        [
            {
                "code": "688520",
                "shares": 200,
                "average_cost": 41.125,
            }
        ]
    )

    positions = build_desk_positions(bundle, "2026-05-28", holdings)

    row = positions.iloc[0]
    assert row["code"] == "688520.SH"
    assert row["current_price"] == 39.46
    assert row["market_value"] == 7892.0
    assert row["unrealized_pnl"] == -333.0


def test_quick_plan_uses_holdings_cost_for_default_portfolio_stop() -> None:
    bundle = MarketDataBundle(
        prices=pd.DataFrame(
            [
                {"date": "2026-06-01", "code": "A", "close": 96.0},
                {"date": "2026-06-01", "code": "B", "close": 82.0},
            ]
        )
    )
    holdings = pd.DataFrame(
        [
            {"code": "A", "name": "Alpha", "shares": 100, "average_cost": 100.0},
            {"code": "B", "name": "Beta", "shares": 100, "average_cost": 100.0},
        ]
    )
    regime = pd.Series({"market_gate": False, "breadth_ma20": 0.2, "dist_ma60": -0.02})

    plan = _quick_plan_from_watchlist(
        bundle=bundle,
        as_of_date=pd.Timestamp("2026-06-01"),
        holdings=holdings,
        watchlist=pd.DataFrame(),
        signal_config=live100k_hotspot_manual_fixed_signal_config(),
        execution_config=live100k_hotspot_manual_fixed_execution_config(),
        regime=regime,
        cash=None,
        equity_peak=None,
    )

    sells = plan.loc[plan["action"] == "sell_next_open"]
    assert sells["code"].tolist() == ["A", "B"]
    assert sells["reason"].tolist() == ["portfolio_stop", "portfolio_stop"]
