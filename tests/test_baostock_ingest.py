from pathlib import Path

import pandas as pd

from baiquant.data.baostock_provider import BaoStockIngestConfig, ingest_baostock
from baiquant.data.bundle import MarketDataBundle
from baiquant.data.sqlite_provider import SqliteDataProvider, write_market_data_to_sqlite


class FakeResult:
    def __init__(self, fields: list[str], rows: list[list[str]]) -> None:
        self.fields = fields
        self.rows = rows
        self.error_code = "0"
        self.error_msg = "success"
        self._index = -1

    def next(self) -> bool:
        self._index += 1
        return self._index < len(self.rows)

    def get_row_data(self) -> list[str]:
        return self.rows[self._index]


class FakeLoginResult:
    error_code = "0"
    error_msg = "success"


class FakeBaoStock:
    def __init__(self) -> None:
        self.profit_calls: list[tuple[str, int, int]] = []

    def login(self) -> FakeLoginResult:
        return FakeLoginResult()

    def logout(self) -> None:
        return None

    def query_stock_basic(self, code: str | None = None) -> FakeResult:
        rows = [
            ["sz.000001", "平安银行", "1991-04-03", "", "1", "1"],
            ["sh.600000", "浦发银行", "1999-11-10", "", "1", "1"],
        ]
        if code:
            rows = [row for row in rows if row[0] == code]
        return FakeResult(["code", "code_name", "ipoDate", "outDate", "type", "status"], rows)

    def query_stock_industry(self, code: str | None = None) -> FakeResult:
        rows = [
            ["2026-05-25", "sz.000001", "平安银行", "J66货币金融服务", "证监会行业分类"],
            ["2026-05-25", "sh.600000", "浦发银行", "J66货币金融服务", "证监会行业分类"],
        ]
        if code:
            rows = [row for row in rows if row[1] == code]
        return FakeResult(
            ["updateDate", "code", "code_name", "industry", "industryClassification"],
            rows,
        )

    def query_profit_data(self, code: str, year: int, quarter: int) -> FakeResult:
        self.profit_calls.append((code, year, quarter))
        return FakeResult(
            [
                "code",
                "pubDate",
                "statDate",
                "roeAvg",
                "npMargin",
                "gpMargin",
                "netProfit",
                "epsTTM",
                "MBRevenue",
                "totalShare",
                "liqaShare",
            ],
            [[code, "2026-03-31", "2025-12-31", "0.064", "0.28", "", "100", "1.2", "200", "10", "9"]],
        )

    def query_growth_data(self, code: str, year: int, quarter: int) -> FakeResult:
        return FakeResult(
            ["code", "pubDate", "statDate", "YOYEquity", "YOYAsset", "YOYNI", "YOYEPSBasic", "YOYPNI"],
            [[code, "2026-03-31", "2025-12-31", "0.1", "0.2", "0.35", "0.3", "0.34"]],
        )


class FakeBaoStockWithBrokenBasics(FakeBaoStock):
    def query_stock_basic(self, code: str | None = None) -> FakeResult:
        return FakeResult([], [])


def test_baostock_ingest_updates_sqlite_stocks_and_fundamentals(tmp_path: Path) -> None:
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
                [["000001.SZ", "Old Name", "", "1900-01-01", 0]],
                columns=["code", "name", "industry", "list_date", "is_st"],
            ),
            fundamentals=pd.DataFrame(
                columns=["date", "code", "pe_ttm", "pb", "roe", "revenue_yoy", "profit_yoy"]
            ),
        ),
    )

    summary = ingest_baostock(
        BaoStockIngestConfig(
            output_path=db_path,
            symbols=["000001.SZ"],
            year=2025,
            quarter=4,
        ),
        bs=FakeBaoStock(),
    )
    bundle = SqliteDataProvider(db_path).load()

    assert summary == {"stocks": 1, "fundamentals": 1, "failures": 0}
    assert bundle.stocks.loc[0, "name"] == "平安银行"
    assert bundle.stocks.loc[0, "industry"] == "J66货币金融服务"
    assert bundle.stocks.loc[0, "list_date"] == pd.Timestamp("1991-04-03")
    assert bundle.fundamentals.loc[0, "date"] == pd.Timestamp("2026-03-31")
    assert bundle.fundamentals.loc[0, "roe"] == 0.064
    assert bundle.fundamentals.loc[0, "profit_yoy"] == 0.35


def test_baostock_ingest_reuses_existing_stocks_when_basic_query_breaks(tmp_path: Path) -> None:
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
            fundamentals=pd.DataFrame(
                columns=["date", "code", "pe_ttm", "pb", "roe", "revenue_yoy", "profit_yoy"]
            ),
        ),
    )

    summary = ingest_baostock(
        BaoStockIngestConfig(
            output_path=db_path,
            symbols=["000001.SZ"],
            year=2025,
            quarter=4,
        ),
        bs=FakeBaoStockWithBrokenBasics(),
    )

    bundle = SqliteDataProvider(db_path).load()
    assert summary["stocks"] == 1
    assert summary["fundamentals"] == 1
    assert bundle.stocks.loc[0, "industry"] == "银行"
