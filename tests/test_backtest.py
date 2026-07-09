import pytest
import pandas as pd

from baiquant.backtest import AShareExecutionConfig, run_a_share_execution_backtest
from baiquant.backtest import BacktestConfig, run_rebalance_backtest
from baiquant.backtest import MultiPositionTrendConfig, run_multi_position_trend_backtest
from baiquant.backtest import TrendPyramidConfig, run_trend_pyramid_backtest


def test_rebalance_backtest_uses_next_period_returns_and_fees() -> None:
    prices = pd.DataFrame(
        [
            ["2026-01-01", "A", 10],
            ["2026-01-02", "A", 11],
            ["2026-01-03", "A", 12],
            ["2026-01-01", "B", 10],
            ["2026-01-02", "B", 9],
            ["2026-01-03", "B", 8],
        ],
        columns=["date", "code", "close"],
    )
    prices["date"] = pd.to_datetime(prices["date"])
    weights = pd.DataFrame(
        [
            ["2026-01-01", "A", 1.0],
            ["2026-01-02", "A", 1.0],
        ],
        columns=["date", "code", "weight"],
    )
    weights["date"] = pd.to_datetime(weights["date"])

    result = run_rebalance_backtest(
        prices,
        weights,
        config=BacktestConfig(fee_bps=10),
    )

    assert result.equity_curve["equity"].iloc[-1] > 1.18
    assert result.metrics["total_return"] > 0.18
    assert result.daily_returns["portfolio_return"].notna().all()


def test_a_share_execution_backtest_does_not_buy_limit_up_stock() -> None:
    prices = pd.DataFrame(
        [
            ["2026-01-01", "A", 10.0, 10.0, 0, 0, 0],
            ["2026-01-02", "A", 10.0, 10.0, 0, 1, 0],
        ],
        columns=["date", "code", "open", "close", "paused", "limit_up", "limit_down"],
    )
    targets = pd.DataFrame(
        [["2026-01-01", "A", 100]],
        columns=["date", "code", "shares"],
    )

    result = run_a_share_execution_backtest(
        prices,
        targets,
        config=AShareExecutionConfig(initial_cash=10_000, fee_bps=0, slippage_bps=0),
    )

    assert result.positions.empty
    assert result.trades.loc[0, "status"] == "failed_limit_up"
    assert result.equity_curve["equity"].iloc[-1] == 10_000


def test_a_share_execution_backtest_does_not_sell_limit_down_stock() -> None:
    prices = pd.DataFrame(
        [
            ["2026-01-01", "A", 10.0, 10.0, 0, 0, 0],
            ["2026-01-02", "A", 10.0, 10.0, 0, 0, 0],
            ["2026-01-03", "A", 10.0, 8.0, 0, 0, 1],
        ],
        columns=["date", "code", "open", "close", "paused", "limit_up", "limit_down"],
    )
    targets = pd.DataFrame(
        [
            ["2026-01-01", "A", 100],
            ["2026-01-02", "A", 0],
        ],
        columns=["date", "code", "shares"],
    )

    result = run_a_share_execution_backtest(
        prices,
        targets,
        config=AShareExecutionConfig(initial_cash=10_000, fee_bps=0, slippage_bps=0),
    )

    assert result.positions.set_index("code").loc["A", "shares"] == 100
    assert result.trades["status"].tolist() == ["filled", "failed_limit_down"]
    assert result.equity_curve["equity"].iloc[-1] == 9800


def test_a_share_execution_backtest_blocks_new_buys_after_drawdown_stop() -> None:
    prices = pd.DataFrame(
        [
            ["2026-01-01", "A", 10.0, 10.0, 0, 0, 0],
            ["2026-01-02", "A", 10.0, 8.0, 0, 0, 0],
            ["2026-01-03", "B", 10.0, 10.0, 0, 0, 0],
            ["2026-01-04", "B", 10.0, 10.0, 0, 0, 0],
        ],
        columns=["date", "code", "open", "close", "paused", "limit_up", "limit_down"],
    )
    targets = pd.DataFrame(
        [
            ["2026-01-01", "A", 100],
            ["2026-01-02", "B", 100],
        ],
        columns=["date", "code", "shares"],
    )

    result = run_a_share_execution_backtest(
        prices,
        targets,
        config=AShareExecutionConfig(
            initial_cash=1_000,
            fee_bps=0,
            slippage_bps=0,
            stop_drawdown_pct=0.10,
            stop_cooldown_days=10,
        ),
    )

    assert "risk_stop" in result.daily_returns["risk_state"].tolist()
    assert result.trades.loc[result.trades["code"] == "B", "status"].tolist() == ["blocked_risk_stop"]


def test_a_share_execution_backtest_can_liquidate_after_drawdown_stop() -> None:
    prices = pd.DataFrame(
        [
            ["2026-01-01", "A", 10.0, 10.0, 0, 0, 0],
            ["2026-01-02", "A", 10.0, 8.0, 0, 0, 0],
            ["2026-01-03", "A", 8.0, 8.0, 0, 0, 0],
        ],
        columns=["date", "code", "open", "close", "paused", "limit_up", "limit_down"],
    )
    targets = pd.DataFrame(
        [["2026-01-01", "A", 100]],
        columns=["date", "code", "shares"],
    )

    result = run_a_share_execution_backtest(
        prices,
        targets,
        config=AShareExecutionConfig(
            initial_cash=1_000,
            fee_bps=0,
            slippage_bps=0,
            stop_drawdown_pct=0.10,
            stop_cooldown_days=10,
            liquidate_on_stop=True,
        ),
    )

    assert result.trades["side"].tolist() == ["buy", "sell"]
    assert result.trades["status"].tolist() == ["filled", "filled"]
    assert result.positions.empty


def test_a_share_execution_backtest_rounds_cash_limited_buys_to_lots() -> None:
    prices = pd.DataFrame(
        [
            ["2026-01-01", "A", 10.0, 10.0, 0, 0, 0],
            ["2026-01-02", "A", 10.0, 10.0, 0, 0, 0],
        ],
        columns=["date", "code", "open", "close", "paused", "limit_up", "limit_down"],
    )
    targets = pd.DataFrame(
        [["2026-01-01", "A", 200]],
        columns=["date", "code", "shares"],
    )

    result = run_a_share_execution_backtest(
        prices,
        targets,
        config=AShareExecutionConfig(initial_cash=1_500, fee_bps=0, slippage_bps=0, lot_size=100),
    )

    assert result.trades.loc[0, "filled_shares"] == 100
    assert result.positions.set_index("code").loc["A", "shares"] == 100


def test_a_share_execution_backtest_carries_last_close_for_missing_position_prices() -> None:
    prices = pd.DataFrame(
        [
            ["2026-01-01", "A", 10.0, 10.0, 0, 0, 0],
            ["2026-01-02", "A", 10.0, 10.0, 0, 0, 0],
            ["2026-01-03", "B", 20.0, 20.0, 0, 0, 0],
        ],
        columns=["date", "code", "open", "close", "paused", "limit_up", "limit_down"],
    )
    targets = pd.DataFrame(
        [["2026-01-01", "A", 100]],
        columns=["date", "code", "shares"],
    )

    result = run_a_share_execution_backtest(
        prices,
        targets,
        config=AShareExecutionConfig(initial_cash=1_000, fee_bps=0, slippage_bps=0),
    )

    assert result.equity_curve.loc[result.equity_curve["date"] == pd.Timestamp("2026-01-03"), "equity"].iloc[0] == 1_000


def test_trend_pyramid_backtest_buys_two_thirds_then_adds_half_on_five_pct_gain() -> None:
    prices = pd.DataFrame(
        [
            ["2026-01-01", "A", 10.0, 10.0, 0, 0, 0],
            ["2026-01-02", "A", 10.0, 10.6, 0, 0, 0],
            ["2026-01-03", "A", 10.8, 11.0, 0, 0, 0],
        ],
        columns=["date", "code", "open", "close", "paused", "limit_up", "limit_down"],
    )
    signals = pd.DataFrame([["2026-01-01", "A"]], columns=["date", "code"])

    result = run_trend_pyramid_backtest(
        prices,
        signals,
        TrendPyramidConfig(
            initial_cash=20_000,
            fee_bps=0,
            slippage_bps=0,
            lot_size=100,
            initial_position_pct=2 / 3,
            add_trigger_pct=0.05,
            add_position_multiple=0.5,
        ),
    )

    assert result.trades[["date", "code", "side", "filled_shares", "reason", "status"]].to_dict("records") == [
        {
            "date": pd.Timestamp("2026-01-02"),
            "code": "A",
            "side": "buy",
            "filled_shares": 1300,
            "reason": "entry",
            "status": "filled",
        },
        {
            "date": pd.Timestamp("2026-01-03"),
            "code": "A",
            "side": "buy",
            "filled_shares": 600,
            "reason": "profit_add",
            "status": "filled",
        },
    ]
    assert result.positions.set_index("code").loc["A", "shares"] == 1900


def test_trend_pyramid_backtest_exits_next_open_after_six_pct_stop_loss() -> None:
    prices = pd.DataFrame(
        [
            ["2026-01-01", "A", 10.0, 10.0, 0, 0, 0],
            ["2026-01-02", "A", 10.0, 9.3, 0, 0, 0],
            ["2026-01-03", "A", 9.2, 9.2, 0, 0, 0],
        ],
        columns=["date", "code", "open", "close", "paused", "limit_up", "limit_down"],
    )
    signals = pd.DataFrame([["2026-01-01", "A"]], columns=["date", "code"])

    result = run_trend_pyramid_backtest(
        prices,
        signals,
        TrendPyramidConfig(initial_cash=20_000, fee_bps=0, slippage_bps=0, lot_size=100),
    )

    sell = result.trades.loc[result.trades["side"] == "sell"].iloc[0]
    assert sell["date"] == pd.Timestamp("2026-01-03")
    assert sell["filled_shares"] == 1300
    assert sell["reason"] == "stop_loss"
    assert result.positions.empty


def test_trend_pyramid_backtest_exits_next_open_after_close_breaks_ma5() -> None:
    prices = pd.DataFrame(
        [
            ["2026-01-01", "A", 10.0, 10.0, 0, 0, 0],
            ["2026-01-02", "A", 10.0, 10.2, 0, 0, 0],
            ["2026-01-03", "A", 10.2, 10.4, 0, 0, 0],
            ["2026-01-04", "A", 10.4, 10.6, 0, 0, 0],
            ["2026-01-05", "A", 10.6, 10.8, 0, 0, 0],
            ["2026-01-06", "A", 10.8, 10.1, 0, 0, 0],
            ["2026-01-07", "A", 10.0, 10.0, 0, 0, 0],
        ],
        columns=["date", "code", "open", "close", "paused", "limit_up", "limit_down"],
    )
    signals = pd.DataFrame([["2026-01-01", "A"]], columns=["date", "code"])

    result = run_trend_pyramid_backtest(
        prices,
        signals,
        TrendPyramidConfig(
            initial_cash=20_000,
            fee_bps=0,
            slippage_bps=0,
            lot_size=100,
            stop_loss_pct=0.20,
            add_trigger_pct=1.0,
            ma_window=5,
        ),
    )

    sell = result.trades.loc[result.trades["side"] == "sell"].iloc[0]
    assert sell["date"] == pd.Timestamp("2026-01-07")
    assert sell["reason"] == "ma_break"
    assert result.positions.empty


def test_trend_pyramid_backtest_exits_next_open_after_five_pct_trailing_drawdown() -> None:
    prices = pd.DataFrame(
        [
            ["2026-01-01", "A", 10.0, 10.0, 0, 0, 0],
            ["2026-01-02", "A", 10.0, 10.8, 0, 0, 0],
            ["2026-01-03", "A", 10.8, 11.0, 0, 0, 0],
            ["2026-01-04", "A", 10.4, 10.4, 0, 0, 0],
            ["2026-01-05", "A", 10.3, 10.3, 0, 0, 0],
        ],
        columns=["date", "code", "open", "close", "paused", "limit_up", "limit_down"],
    )
    signals = pd.DataFrame([["2026-01-01", "A"]], columns=["date", "code"])

    result = run_trend_pyramid_backtest(
        prices,
        signals,
        TrendPyramidConfig(
            initial_cash=20_000,
            fee_bps=0,
            slippage_bps=0,
            lot_size=100,
            stop_loss_pct=0.06,
            add_trigger_pct=0.05,
            add_position_multiple=0.5,
            ma_window=0,
            trailing_drawdown_pct=0.05,
        ),
    )

    sell = result.trades.loc[result.trades["side"] == "sell"].iloc[0]
    assert sell["date"] == pd.Timestamp("2026-01-05")
    assert sell["filled_shares"] == 1900
    assert sell["reason"] == "trailing_drawdown"
    assert result.positions.empty


def test_multi_position_trend_backtest_buys_multiple_signals_to_fill_account() -> None:
    prices = pd.DataFrame(
        [
            ["2026-01-01", "A", 10.0, 10.0, 0, 0, 0],
            ["2026-01-01", "B", 20.0, 20.0, 0, 0, 0],
            ["2026-01-02", "A", 10.0, 10.0, 0, 0, 0],
            ["2026-01-02", "B", 20.0, 20.0, 0, 0, 0],
        ],
        columns=["date", "code", "open", "close", "paused", "limit_up", "limit_down"],
    )
    signals = pd.DataFrame(
        [["2026-01-01", "A"], ["2026-01-01", "B"]],
        columns=["date", "code"],
    )

    result = run_multi_position_trend_backtest(
        prices,
        signals,
        MultiPositionTrendConfig(initial_cash=20_000, max_positions=2, fee_bps=0, slippage_bps=0),
    )

    assert result.trades[["date", "code", "side", "filled_shares", "reason", "status"]].to_dict("records") == [
        {
            "date": pd.Timestamp("2026-01-02"),
            "code": "A",
            "side": "buy",
            "filled_shares": 1000,
            "reason": "entry",
            "status": "filled",
        },
        {
            "date": pd.Timestamp("2026-01-02"),
            "code": "B",
            "side": "buy",
            "filled_shares": 500,
            "reason": "entry",
            "status": "filled",
        },
    ]
    assert result.daily_returns["cash"].iloc[-1] == 0
    assert result.positions.set_index("code")["shares"].to_dict() == {"A": 1000, "B": 500}


def test_multi_position_trend_backtest_honors_signal_position_scale() -> None:
    prices = pd.DataFrame(
        [
            ["2026-01-01", "A", 10.0, 10.0, 0, 0, 0],
            ["2026-01-01", "B", 20.0, 20.0, 0, 0, 0],
            ["2026-01-02", "A", 10.0, 10.0, 0, 0, 0],
            ["2026-01-02", "B", 20.0, 20.0, 0, 0, 0],
        ],
        columns=["date", "code", "open", "close", "paused", "limit_up", "limit_down"],
    )
    signals = pd.DataFrame(
        [
            ["2026-01-01", "A", 1.0],
            ["2026-01-01", "B", 0.5],
        ],
        columns=["date", "code", "position_scale"],
    )

    result = run_multi_position_trend_backtest(
        prices,
        signals,
        MultiPositionTrendConfig(
            initial_cash=20_000,
            max_positions=2,
            fee_bps=0,
            slippage_bps=0,
            ma_window=0,
            use_position_scale=True,
        ),
    )

    buys = result.trades.loc[(result.trades["side"] == "buy") & (result.trades["status"] == "filled")]
    assert buys.set_index("code")["filled_shares"].to_dict() == {"A": 1000, "B": 200}


def test_multi_position_trend_backtest_ignores_signal_position_scale_by_default() -> None:
    prices = pd.DataFrame(
        [
            ["2026-01-01", "A", 10.0, 10.0, 0, 0, 0],
            ["2026-01-01", "B", 20.0, 20.0, 0, 0, 0],
            ["2026-01-02", "A", 10.0, 10.0, 0, 0, 0],
            ["2026-01-02", "B", 20.0, 20.0, 0, 0, 0],
        ],
        columns=["date", "code", "open", "close", "paused", "limit_up", "limit_down"],
    )
    signals = pd.DataFrame(
        [
            ["2026-01-01", "A", 1.0],
            ["2026-01-01", "B", 0.5],
        ],
        columns=["date", "code", "position_scale"],
    )

    result = run_multi_position_trend_backtest(
        prices,
        signals,
        MultiPositionTrendConfig(initial_cash=20_000, max_positions=2, fee_bps=0, slippage_bps=0, ma_window=0),
    )

    buys = result.trades.loc[(result.trades["side"] == "buy") & (result.trades["status"] == "filled")]
    assert buys.set_index("code")["filled_shares"].to_dict() == {"A": 1000, "B": 500}


def test_multi_position_trend_backtest_stops_one_position_without_selling_others() -> None:
    prices = pd.DataFrame(
        [
            ["2026-01-01", "A", 10.0, 10.0, 0, 0, 0],
            ["2026-01-01", "B", 20.0, 20.0, 0, 0, 0],
            ["2026-01-02", "A", 10.0, 9.3, 0, 0, 0],
            ["2026-01-02", "B", 20.0, 20.0, 0, 0, 0],
            ["2026-01-03", "A", 9.2, 9.2, 0, 0, 0],
            ["2026-01-03", "B", 20.0, 20.0, 0, 0, 0],
        ],
        columns=["date", "code", "open", "close", "paused", "limit_up", "limit_down"],
    )
    signals = pd.DataFrame(
        [["2026-01-01", "A"], ["2026-01-01", "B"]],
        columns=["date", "code"],
    )

    result = run_multi_position_trend_backtest(
        prices,
        signals,
        MultiPositionTrendConfig(initial_cash=20_000, max_positions=2, fee_bps=0, slippage_bps=0),
    )

    sell = result.trades.loc[result.trades["side"] == "sell"].iloc[0]
    assert sell["date"] == pd.Timestamp("2026-01-03")
    assert sell["code"] == "A"
    assert sell["reason"] == "stop_loss"
    assert result.positions.set_index("code")["shares"].to_dict() == {"B": 500}


def test_multi_position_trend_backtest_adds_half_after_five_pct_gain() -> None:
    prices = pd.DataFrame(
        [
            ["2026-01-01", "A", 10.0, 10.0, 0, 0, 0],
            ["2026-01-02", "A", 10.0, 10.6, 0, 0, 0],
            ["2026-01-03", "A", 10.8, 11.0, 0, 0, 0],
        ],
        columns=["date", "code", "open", "close", "paused", "limit_up", "limit_down"],
    )
    signals = pd.DataFrame([["2026-01-01", "A"]], columns=["date", "code"])

    result = run_multi_position_trend_backtest(
        prices,
        signals,
        MultiPositionTrendConfig(
            initial_cash=20_000,
            max_positions=2,
            fee_bps=0,
            slippage_bps=0,
            lot_size=100,
            ma_window=0,
            add_trigger_pct=0.05,
            add_position_multiple=0.5,
        ),
    )

    assert result.trades[["date", "code", "side", "filled_shares", "reason", "status"]].to_dict("records") == [
        {
            "date": pd.Timestamp("2026-01-02"),
            "code": "A",
            "side": "buy",
            "filled_shares": 1000,
            "reason": "entry",
            "status": "filled",
        },
        {
            "date": pd.Timestamp("2026-01-03"),
            "code": "A",
            "side": "buy",
            "filled_shares": 500,
            "reason": "profit_add",
            "status": "filled",
        },
    ]
    assert result.positions.set_index("code").loc["A", "shares"] == 1500


def test_multi_position_trend_backtest_exits_after_max_holding_days() -> None:
    prices = pd.DataFrame(
        [
            ["2026-01-01", "A", 10.0, 10.0, 0, 0, 0],
            ["2026-01-02", "A", 10.0, 10.1, 0, 0, 0],
            ["2026-01-03", "A", 10.2, 10.2, 0, 0, 0],
            ["2026-01-04", "A", 10.3, 10.3, 0, 0, 0],
        ],
        columns=["date", "code", "open", "close", "paused", "limit_up", "limit_down"],
    )
    signals = pd.DataFrame([["2026-01-01", "A"]], columns=["date", "code"])

    result = run_multi_position_trend_backtest(
        prices,
        signals,
        MultiPositionTrendConfig(
            initial_cash=20_000,
            max_positions=1,
            fee_bps=0,
            slippage_bps=0,
            lot_size=100,
            ma_window=0,
            max_holding_days=2,
        ),
    )

    assert result.trades[["date", "code", "side", "filled_shares", "reason", "status"]].to_dict("records") == [
        {
            "date": pd.Timestamp("2026-01-02"),
            "code": "A",
            "side": "buy",
            "filled_shares": 2000,
            "reason": "entry",
            "status": "filled",
        },
        {
            "date": pd.Timestamp("2026-01-04"),
            "code": "A",
            "side": "sell",
            "filled_shares": 2000,
            "reason": "max_holding_days",
            "status": "filled",
        },
    ]
    assert result.positions.empty


def test_multi_position_trend_backtest_refills_slot_after_ma5_break() -> None:
    rows = []
    closes = {
        "A": [10.0, 10.2, 10.4, 10.6, 10.8, 10.1, 10.0],
        "B": [20.0, 20.1, 20.2, 20.3, 20.4, 20.5, 20.6],
        "C": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
    }
    for day_index in range(7):
        date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=day_index)
        for code, code_closes in closes.items():
            close = code_closes[day_index]
            rows.append([date, code, close, close, 0, 0, 0])
    prices = pd.DataFrame(rows, columns=["date", "code", "open", "close", "paused", "limit_up", "limit_down"])
    signals = pd.DataFrame(
        [
            ["2026-01-01", "A"],
            ["2026-01-01", "B"],
            ["2026-01-06", "C"],
        ],
        columns=["date", "code"],
    )

    result = run_multi_position_trend_backtest(
        prices,
        signals,
        MultiPositionTrendConfig(initial_cash=20_000, max_positions=2, fee_bps=0, slippage_bps=0),
    )

    day_7_trades = result.trades.loc[result.trades["date"] == pd.Timestamp("2026-01-07")]
    assert day_7_trades[["code", "side", "reason", "status"]].to_dict("records") == [
        {"code": "A", "side": "sell", "reason": "ma_break", "status": "filled"},
        {"code": "C", "side": "buy", "reason": "entry", "status": "filled"},
    ]
    assert set(result.positions["code"]) == {"B", "C"}


def test_multi_position_trend_backtest_exits_after_activated_trailing_stop() -> None:
    prices = pd.DataFrame(
        [
            ["2026-01-01", "A", 10.0, 10.0, 0, 0, 0],
            ["2026-01-02", "A", 10.0, 10.0, 0, 0, 0],
            ["2026-01-03", "A", 11.5, 11.5, 0, 0, 0],
            ["2026-01-04", "A", 10.7, 10.7, 0, 0, 0],
            ["2026-01-05", "A", 10.6, 10.6, 0, 0, 0],
        ],
        columns=["date", "code", "open", "close", "paused", "limit_up", "limit_down"],
    )
    signals = pd.DataFrame([["2026-01-01", "A"]], columns=["date", "code"])

    result = run_multi_position_trend_backtest(
        prices,
        signals,
        MultiPositionTrendConfig(
            initial_cash=20_000,
            max_positions=1,
            fee_bps=0,
            slippage_bps=0,
            ma_window=20,
            stop_loss_pct=0.50,
            trailing_stop_activation_pct=0.10,
            trailing_stop_pct=0.05,
        ),
    )

    sell = result.trades.loc[result.trades["side"] == "sell"].iloc[0]
    assert sell["date"] == pd.Timestamp("2026-01-05")
    assert sell["reason"] == "trailing_stop"
    assert result.positions.empty


def test_multi_position_trend_backtest_exits_after_take_profit() -> None:
    prices = pd.DataFrame(
        [
            ["2026-01-01", "A", 10.0, 10.0, 0, 0, 0],
            ["2026-01-02", "A", 10.0, 10.0, 0, 0, 0],
            ["2026-01-03", "A", 12.5, 12.5, 0, 0, 0],
            ["2026-01-04", "A", 12.3, 12.3, 0, 0, 0],
        ],
        columns=["date", "code", "open", "close", "paused", "limit_up", "limit_down"],
    )
    signals = pd.DataFrame([["2026-01-01", "A"]], columns=["date", "code"])

    result = run_multi_position_trend_backtest(
        prices,
        signals,
        MultiPositionTrendConfig(
            initial_cash=20_000,
            max_positions=1,
            fee_bps=0,
            slippage_bps=0,
            ma_window=20,
            stop_loss_pct=0.50,
            take_profit_pct=0.20,
        ),
    )

    sell = result.trades.loc[result.trades["side"] == "sell"].iloc[0]
    assert sell["date"] == pd.Timestamp("2026-01-04")
    assert sell["reason"] == "take_profit"
    assert result.positions.empty


def test_multi_position_trend_backtest_blocks_new_entries_after_portfolio_stop() -> None:
    prices = pd.DataFrame(
        [
            ["2026-01-01", "A", 10.0, 10.0, 0, 0, 0],
            ["2026-01-01", "B", 10.0, 10.0, 0, 0, 0],
            ["2026-01-02", "A", 10.0, 8.8, 0, 0, 0],
            ["2026-01-02", "B", 10.0, 10.0, 0, 0, 0],
            ["2026-01-03", "A", 8.8, 8.8, 0, 0, 0],
            ["2026-01-03", "B", 10.0, 10.0, 0, 0, 0],
        ],
        columns=["date", "code", "open", "close", "paused", "limit_up", "limit_down"],
    )
    signals = pd.DataFrame(
        [["2026-01-01", "A"], ["2026-01-02", "B"]],
        columns=["date", "code"],
    )

    result = run_multi_position_trend_backtest(
        prices,
        signals,
        MultiPositionTrendConfig(
            initial_cash=10_000,
            max_positions=1,
            fee_bps=0,
            slippage_bps=0,
            stop_loss_pct=0.50,
            portfolio_stop_drawdown_pct=0.10,
            portfolio_stop_cooldown_days=2,
        ),
    )

    blocked = result.trades.loc[result.trades["status"] == "blocked_portfolio_stop"].iloc[0]
    assert blocked["date"] == pd.Timestamp("2026-01-03")
    assert blocked["code"] == "B"
    assert "risk_stop" in result.daily_returns["risk_state"].tolist()
    assert result.positions.set_index("code")["shares"].to_dict() == {"A": 1000}


def test_multi_position_trend_backtest_can_liquidate_on_portfolio_stop() -> None:
    prices = pd.DataFrame(
        [
            ["2026-01-01", "A", 10.0, 10.0, 0, 0, 0],
            ["2026-01-02", "A", 10.0, 8.8, 0, 0, 0],
            ["2026-01-03", "A", 8.7, 8.7, 0, 0, 0],
        ],
        columns=["date", "code", "open", "close", "paused", "limit_up", "limit_down"],
    )
    signals = pd.DataFrame([["2026-01-01", "A"]], columns=["date", "code"])

    result = run_multi_position_trend_backtest(
        prices,
        signals,
        MultiPositionTrendConfig(
            initial_cash=10_000,
            max_positions=1,
            fee_bps=0,
            slippage_bps=0,
            stop_loss_pct=0.50,
            portfolio_stop_drawdown_pct=0.10,
            portfolio_stop_cooldown_days=2,
            liquidate_on_portfolio_stop=True,
        ),
    )

    assert result.trades["side"].tolist() == ["buy", "sell"]
    assert result.trades.loc[result.trades["side"] == "sell", "reason"].iloc[0] == "portfolio_stop"
    assert result.positions.empty


def test_multi_position_trend_backtest_can_reset_peak_after_portfolio_stop_liquidation() -> None:
    rows = []
    prices_by_code = {
        "A": [(10.0, 10.0), (10.0, 8.8), (8.8, 8.8), (8.8, 8.8), (8.8, 8.8)],
        "C": [(8.8, 8.8), (8.8, 8.8), (8.8, 8.8), (8.8, 8.8), (8.8, 8.8)],
    }
    for day_index in range(5):
        date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=day_index)
        for code, code_prices in prices_by_code.items():
            open_price, close_price = code_prices[day_index]
            rows.append([date, code, open_price, close_price, 0, 0, 0])
    prices = pd.DataFrame(rows, columns=["date", "code", "open", "close", "paused", "limit_up", "limit_down"])
    signals = pd.DataFrame(
        [
            ["2026-01-01", "A"],
            ["2026-01-03", "C"],
        ],
        columns=["date", "code"],
    )

    result = run_multi_position_trend_backtest(
        prices,
        signals,
        MultiPositionTrendConfig(
            initial_cash=10_000,
            max_positions=1,
            fee_bps=0,
            slippage_bps=0,
            stop_loss_pct=0.50,
            portfolio_stop_drawdown_pct=0.10,
            portfolio_stop_cooldown_days=1,
            liquidate_on_portfolio_stop=True,
            reset_peak_on_portfolio_stop=True,
        ),
    )

    assert result.trades[["date", "code", "side", "reason", "status"]].to_dict("records") == [
        {
            "date": pd.Timestamp("2026-01-02"),
            "code": "A",
            "side": "buy",
            "reason": "entry",
            "status": "filled",
        },
        {
            "date": pd.Timestamp("2026-01-03"),
            "code": "A",
            "side": "sell",
            "reason": "portfolio_stop",
            "status": "filled",
        },
        {
            "date": pd.Timestamp("2026-01-04"),
            "code": "C",
            "side": "buy",
            "reason": "entry",
            "status": "filled",
        },
    ]
    assert result.positions.set_index("code")["shares"].to_dict() == {"C": 900}


def test_multi_position_trend_backtest_can_require_profit_before_resetting_peak() -> None:
    rows = []
    prices_by_code = {
        "A": [(10.0, 10.0), (10.0, 8.8), (8.8, 8.8), (8.8, 8.8), (8.8, 8.8), (8.8, 8.8)],
        "C": [(8.8, 8.8), (8.8, 8.8), (8.8, 8.8), (8.8, 8.8), (8.8, 8.8), (8.8, 8.8)],
    }
    for day_index in range(6):
        date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=day_index)
        for code, code_prices in prices_by_code.items():
            open_price, close_price = code_prices[day_index]
            rows.append([date, code, open_price, close_price, 0, 0, 0])
    prices = pd.DataFrame(rows, columns=["date", "code", "open", "close", "paused", "limit_up", "limit_down"])
    signals = pd.DataFrame(
        [
            ["2026-01-01", "A"],
            ["2026-01-03", "C"],
        ],
        columns=["date", "code"],
    )

    result = run_multi_position_trend_backtest(
        prices,
        signals,
        MultiPositionTrendConfig(
            initial_cash=10_000,
            max_positions=1,
            fee_bps=0,
            slippage_bps=0,
            stop_loss_pct=0.50,
            portfolio_stop_drawdown_pct=0.10,
            portfolio_stop_cooldown_days=1,
            liquidate_on_portfolio_stop=True,
            reset_peak_on_portfolio_stop=True,
            reset_peak_min_profit_pct=0.20,
        ),
    )

    c_trades = result.trades.loc[result.trades["code"] == "C"]
    assert c_trades[["side", "reason", "status"]].to_dict("records") == [
        {"side": "buy", "reason": "entry", "status": "filled"},
        {"side": "sell", "reason": "portfolio_stop", "status": "filled"},
    ]
    assert result.positions.empty
