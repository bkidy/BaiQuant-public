from pathlib import Path
import sqlite3

import pandas as pd
import pytest

from baiquant.data.bundle import MarketDataBundle
from baiquant.data.efinance_provider import EFinanceMoneyFlowConfig, ingest_efinance_money_flow
from baiquant.data.sqlite_provider import SqliteDataProvider, write_market_data_to_sqlite


class FakeEFinanceStock:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get_history_bill(self, stock_code: str) -> pd.DataFrame:
        self.calls.append(stock_code)
        return pd.DataFrame(
            [
                {
                    "股票名称": "平安银行",
                    "股票代码": stock_code,
                    "日期": "2026-01-02",
                    "主力净流入": 1000,
                    "小单净流入": -100,
                    "中单净流入": 50,
                    "大单净流入": 400,
                    "超大单净流入": 600,
                    "主力净流入占比": 2.5,
                    "小单流入净占比": -0.2,
                    "中单流入净占比": 0.1,
                    "大单流入净占比": 1.0,
                    "超大单流入净占比": 1.5,
                    "收盘价": 10.5,
                    "涨跌幅": 1.2,
                },
                {
                    "股票名称": "平安银行",
                    "股票代码": stock_code,
                    "日期": "2026-01-03",
                    "主力净流入": 2000,
                    "小单净流入": -200,
                    "中单净流入": 100,
                    "大单净流入": 700,
                    "超大单净流入": 1300,
                    "主力净流入占比": 4.0,
                    "小单流入净占比": -0.4,
                    "中单流入净占比": 0.2,
                    "大单流入净占比": 1.4,
                    "超大单流入净占比": 2.6,
                    "收盘价": 10.8,
                    "涨跌幅": 2.9,
                },
            ]
        )


class FakeEFinance:
    def __init__(self) -> None:
        self.stock = FakeEFinanceStock()


class FakeEFinanceStockWithFailure(FakeEFinanceStock):
    def get_history_bill(self, stock_code: str) -> pd.DataFrame:
        if stock_code == "600000":
            raise TimeoutError("stuck request")
        return super().get_history_bill(stock_code)


class FakeEFinanceWithFailure:
    def __init__(self) -> None:
        self.stock = FakeEFinanceStockWithFailure()


def test_ingest_efinance_money_flow_appends_normalized_sqlite_table(tmp_path: Path) -> None:
    db_path = tmp_path / "market.db"
    write_market_data_to_sqlite(
        db_path,
        MarketDataBundle(
            prices=pd.DataFrame(
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
                ]
            ),
            stocks=pd.DataFrame(
                [["000001.SZ", "平安银行", "银行", "1991-04-03", 0]],
                columns=["code", "name", "industry", "list_date", "is_st"],
            ),
        ),
    )
    fake_ef = FakeEFinance()

    summary = ingest_efinance_money_flow(
        EFinanceMoneyFlowConfig(
            output_path=db_path,
            symbols=["000001.SZ"],
            start_date="20260103",
            end_date="20260103",
        ),
        ef=fake_ef,
    )

    bundle = SqliteDataProvider(db_path).load()

    assert summary == {"money_flow": 1, "failures": 0, "requested_symbols": 1}
    assert fake_ef.stock.calls == ["000001"]
    assert bundle.money_flow[["date", "code", "main_net_inflow", "main_net_inflow_pct"]].to_dict(
        "records"
    ) == [
        {
            "date": pd.Timestamp("2026-01-03"),
            "code": "000001.SZ",
            "main_net_inflow": 2000,
            "main_net_inflow_pct": 4.0,
        }
    ]


def test_ingest_efinance_money_flow_only_requires_stock_catalog(tmp_path: Path) -> None:
    db_path = tmp_path / "market.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE stocks (code TEXT, name TEXT, industry TEXT, list_date TEXT, is_st INTEGER)"
        )
        connection.execute(
            "INSERT INTO stocks VALUES ('000001.SZ', '平安银行', '银行', '1991-04-03', 0)"
        )

    summary = ingest_efinance_money_flow(
        EFinanceMoneyFlowConfig(
            output_path=db_path,
            symbols=["000001.SZ"],
            start_date="20260103",
            end_date="20260103",
        ),
        ef=FakeEFinance(),
    )

    bundle = SqliteDataProvider(db_path).load()
    assert summary["money_flow"] == 1
    assert bundle.money_flow.loc[0, "code"] == "000001.SZ"


def test_ingest_efinance_money_flow_flushes_successful_rows_before_fail_fast_error(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "market.db"
    write_market_data_to_sqlite(
        db_path,
        MarketDataBundle(
            prices=pd.DataFrame(
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
                ]
            ),
            stocks=pd.DataFrame(
                [
                    ["000001.SZ", "平安银行", "银行", "1991-04-03", 0],
                    ["600000.SH", "浦发银行", "银行", "1999-11-10", 0],
                ],
                columns=["code", "name", "industry", "list_date", "is_st"],
            ),
        ),
    )

    with pytest.raises(TimeoutError):
        ingest_efinance_money_flow(
            EFinanceMoneyFlowConfig(
                output_path=db_path,
                symbols=["000001.SZ", "600000.SH"],
                start_date="20260103",
                end_date="20260103",
                continue_on_error=False,
            ),
            ef=FakeEFinanceWithFailure(),
        )

    bundle = SqliteDataProvider(db_path).load()
    assert bundle.money_flow["code"].tolist() == ["000001.SZ"]
