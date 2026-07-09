import pandas as pd
import pytest

from baiquant.live_ledger import TRADE_LOG_COLUMNS, load_trade_log, record_live_trade


def test_load_trade_log_returns_recent_trades_first_and_handles_missing_columns(tmp_path) -> None:
    path = tmp_path / "trade_log.csv"
    pd.DataFrame(
        [
            {"date": "2026-07-01", "time": "09:35", "action": "buy", "code": "A", "name": "Alpha", "shares": 100, "price": 10.0},
            {"date": "2026-07-02", "time": "09:36", "action": "sell", "code": "B", "name": "Beta", "shares": 200, "price": 11.0},
            {"date": "2026-07-03", "time": "09:37", "action": "buy", "code": "C", "name": "Gamma", "shares": 300, "price": 12.0},
        ]
    ).to_csv(path, index=False)

    trades = load_trade_log(path, limit=2)

    assert trades["code"].tolist() == ["C", "B"]
    assert all(column in trades.columns for column in TRADE_LOG_COLUMNS)
    assert trades.iloc[0]["source"] is None
    missing = load_trade_log(tmp_path / "missing.csv")
    assert missing.empty
    assert missing.columns.tolist() == TRADE_LOG_COLUMNS


def test_record_live_buy_updates_average_cost_and_trade_log(tmp_path) -> None:
    holdings_path = tmp_path / "holdings.csv"
    trade_log_path = tmp_path / "trade_log.csv"
    pd.DataFrame(
        [
            {
                "code": "601636.SH",
                "name": "示例A",
                "shares": 300,
                "average_cost": 9.91,
                "high_close": 9.93,
                "entry_shares": 300,
                "added": False,
                "entry_date": "2026-07-01",
            }
        ]
    ).to_csv(holdings_path, index=False)

    holdings, trade = record_live_trade(
        holdings_path=holdings_path,
        trade_log_path=trade_log_path,
        trade_date="2026-07-02",
        action="buy",
        code="601636",
        name="示例A",
        shares=100,
        price=9.70,
        fees=1.0,
        source="manual_test",
        note="add after pullback",
    )

    row = holdings.iloc[0]
    assert row["code"] == "601636.SH"
    assert row["shares"] == 400
    assert row["entry_shares"] == 300
    assert row["added"] is True
    assert row["average_cost"] == pytest.approx(((300 * 9.91) + (100 * 9.70) + 1.0) / 400)
    log = pd.read_csv(trade_log_path)
    assert log[["date", "action", "code", "name", "shares", "price", "source", "note"]].to_dict("records") == [
        {
            "date": "2026-07-02",
            "action": "buy",
            "code": "601636.SH",
            "name": "示例A",
            "shares": 100,
            "price": 9.70,
            "source": "manual_test",
            "note": "add after pullback",
        }
    ]
    assert trade["holdings_shares_after"] == 400


def test_record_live_sell_reduces_holding_and_logs_realized_pnl(tmp_path) -> None:
    holdings_path = tmp_path / "holdings.csv"
    trade_log_path = tmp_path / "trade_log.csv"
    pd.DataFrame(
        [
            {
                "code": "601611.SH",
                "name": "示例B",
                "shares": 800,
                "average_cost": 13.678,
                "high_close": 11.03,
                "entry_shares": 800,
                "added": False,
                "entry_date": "2026-07-01",
            }
        ]
    ).to_csv(holdings_path, index=False)

    holdings, trade = record_live_trade(
        holdings_path=holdings_path,
        trade_log_path=trade_log_path,
        trade_date="2026-07-02",
        action="sell",
        code="601611.SH",
        name="示例B",
        shares=300,
        price=11.20,
        fees=2.0,
        source="manual_test",
    )

    row = holdings.iloc[0]
    assert row["shares"] == 500
    assert row["average_cost"] == pytest.approx(13.678)
    assert trade["realized_pnl"] == pytest.approx((11.20 - 13.678) * 300 - 2.0)
    assert trade["holdings_shares_after"] == 500
    log = pd.read_csv(trade_log_path)
    assert log.iloc[0]["action"] == "sell"
    assert log.iloc[0]["holdings_shares_after"] == 500


def test_record_live_sell_removes_position_when_fully_sold(tmp_path) -> None:
    holdings_path = tmp_path / "holdings.csv"
    trade_log_path = tmp_path / "trade_log.csv"
    pd.DataFrame([{"code": "688520.SH", "name": "示例C", "shares": 400, "average_cost": 38.619}]).to_csv(
        holdings_path,
        index=False,
    )

    holdings, trade = record_live_trade(
        holdings_path=holdings_path,
        trade_log_path=trade_log_path,
        trade_date="2026-07-02",
        action="sell",
        code="688520.SH",
        name="示例C",
        shares=400,
        price=38.00,
    )

    assert holdings.empty
    assert trade["holdings_shares_after"] == 0


def test_record_live_sell_rejects_oversell(tmp_path) -> None:
    holdings_path = tmp_path / "holdings.csv"
    trade_log_path = tmp_path / "trade_log.csv"
    pd.DataFrame([{"code": "688520.SH", "name": "示例C", "shares": 100, "average_cost": 38.619}]).to_csv(
        holdings_path,
        index=False,
    )

    with pytest.raises(ValueError, match="cannot sell"):
        record_live_trade(
            holdings_path=holdings_path,
            trade_log_path=trade_log_path,
            trade_date="2026-07-02",
            action="sell",
            code="688520.SH",
            name="示例C",
            shares=200,
            price=38.00,
        )
