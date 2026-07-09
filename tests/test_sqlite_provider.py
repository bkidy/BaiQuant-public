from pathlib import Path
import sqlite3

import pandas as pd

from baiquant.config import load_pipeline_config
from baiquant.data.bundle import MarketDataBundle
from baiquant.data.sqlite_provider import (
    SqliteDataProvider,
    append_market_data_to_sqlite,
    upsert_market_data_to_sqlite,
    write_market_data_to_sqlite,
)


def test_sqlite_provider_round_trips_market_data_with_dates(tmp_path: Path) -> None:
    bundle = MarketDataBundle(
        prices=pd.DataFrame(
            [
                ["2026-01-03", "000001.SZ", 10, 11, 9, 10.5, 1000, 10_500, 0, 0, 0],
                ["2026-01-02", "000001.SZ", 9, 10, 8, 9.5, 900, 8_550, 0, 0, 0],
            ],
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
            ],
        ),
        fundamentals=pd.DataFrame(
            [["2026-01-03", "000001.SZ", 12, 1.5, 0.12, 0.2, 0.3]],
            columns=["date", "code", "pe_ttm", "pb", "roe", "revenue_yoy", "profit_yoy"],
        ),
        stocks=pd.DataFrame(
            [["000001.SZ", "PingAn", "Bank", "1991-04-03", 0]],
            columns=["code", "name", "industry", "list_date", "is_st"],
        ),
        events=pd.DataFrame(
            [["2026-01-03", "000001.SZ", "buyback", "positive"]],
            columns=["date", "code", "event_type", "sentiment"],
        ),
    )
    db_path = tmp_path / "market.db"

    write_market_data_to_sqlite(db_path, bundle)
    loaded = SqliteDataProvider(db_path).load()

    assert list(loaded.prices["date"]) == [
        pd.Timestamp("2026-01-02"),
        pd.Timestamp("2026-01-03"),
    ]
    assert loaded.fundamentals.loc[0, "roe"] == 0.12
    assert loaded.stocks.loc[0, "list_date"] == pd.Timestamp("1991-04-03")
    assert loaded.events.loc[0, "event_type"] == "buyback"


def test_sqlite_provider_load_window_reads_recent_dated_tables_and_all_stocks(tmp_path: Path) -> None:
    columns = [
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
    money_flow_columns = [
        "date",
        "code",
        "main_net_inflow",
        "small_net_inflow",
        "medium_net_inflow",
        "large_net_inflow",
        "super_large_net_inflow",
        "main_net_inflow_pct",
        "small_net_inflow_pct",
        "medium_net_inflow_pct",
        "large_net_inflow_pct",
        "super_large_net_inflow_pct",
        "close",
        "pct_change",
    ]
    db_path = tmp_path / "market.db"
    write_market_data_to_sqlite(
        db_path,
        MarketDataBundle(
            prices=pd.DataFrame(
                [
                    ["2026-01-01", "000001.SZ", 9, 10, 8, 9.5, 900, 8_550, 0, 0, 0],
                    ["2026-01-10", "000001.SZ", 10, 11, 9, 10.5, 1000, 10_500, 0, 0, 0],
                ],
                columns=columns,
            ),
            money_flow=pd.DataFrame(
                [
                    ["2026-01-01", "000001.SZ", 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 9.5, 0],
                    ["2026-01-10", "000001.SZ", 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 10.5, 0],
                ],
                columns=money_flow_columns,
            ),
            stocks=pd.DataFrame(
                [
                    ["000001.SZ", "PingAn", "Bank", "1991-04-03", 0],
                    ["000002.SZ", "Vanke", "RealEstate", "1991-01-29", 0],
                ],
                columns=["code", "name", "industry", "list_date", "is_st"],
            ),
        ),
    )

    loaded = SqliteDataProvider(db_path).load_window(as_of="2026-01-10", lookback_days=3)

    assert loaded.prices["date"].tolist() == [pd.Timestamp("2026-01-10")]
    assert loaded.money_flow["date"].tolist() == [pd.Timestamp("2026-01-10")]
    assert loaded.stocks["code"].tolist() == ["000001.SZ", "000002.SZ"]


def test_sqlite_provider_load_window_can_use_shorter_side_table_windows(tmp_path: Path) -> None:
    db_path = tmp_path / "market.db"
    write_market_data_to_sqlite(
        db_path,
        MarketDataBundle(
            prices=pd.DataFrame(
                [
                    ["2026-01-01", "000001.SZ", 9, 10, 8, 9.5, 900, 8_550, 0, 0, 0],
                    ["2026-01-10", "000001.SZ", 10, 11, 9, 10.5, 1000, 10_500, 0, 0, 0],
                ],
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
                ],
            ),
            fundamentals=pd.DataFrame(
                [
                    ["2026-01-01", "000001.SZ", 12, 1.5, 0.12, 0.2, 0.3],
                    ["2026-01-10", "000001.SZ", 13, 1.6, 0.13, 0.3, 0.4],
                ],
                columns=["date", "code", "pe_ttm", "pb", "roe", "revenue_yoy", "profit_yoy"],
            ),
            money_flow=pd.DataFrame(
                [
                    ["2026-01-01", "000001.SZ", 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 9.5, 0],
                    ["2026-01-10", "000001.SZ", 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 10.5, 0],
                ],
                columns=[
                    "date",
                    "code",
                    "main_net_inflow",
                    "small_net_inflow",
                    "medium_net_inflow",
                    "large_net_inflow",
                    "super_large_net_inflow",
                    "main_net_inflow_pct",
                    "small_net_inflow_pct",
                    "medium_net_inflow_pct",
                    "large_net_inflow_pct",
                    "super_large_net_inflow_pct",
                    "close",
                    "pct_change",
                ],
            ),
            events=pd.DataFrame(
                [
                    ["2026-01-01", "000001.SZ", "buyback", "positive"],
                    ["2026-01-10", "000001.SZ", "buyback", "positive"],
                ],
                columns=["date", "code", "event_type", "sentiment"],
            ),
        ),
    )

    loaded = SqliteDataProvider(db_path).load_window(
        as_of="2026-01-10",
        lookback_days=20,
        fundamentals_lookback_days=3,
        events_lookback_days=3,
        money_flow_lookback_days=3,
    )

    assert loaded.prices["date"].tolist() == [
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-01-10"),
    ]
    assert loaded.fundamentals["date"].tolist() == [pd.Timestamp("2026-01-10")]
    assert loaded.events["date"].tolist() == [pd.Timestamp("2026-01-10")]
    assert loaded.money_flow["date"].tolist() == [pd.Timestamp("2026-01-10")]


def test_pipeline_config_can_select_sqlite_data_source(tmp_path: Path) -> None:
    config_path = tmp_path / "sqlite.toml"
    config_path.write_text(
        "\n".join(
            [
                "[data]",
                'kind = "sqlite"',
                'path = "./market.db"',
                "",
                "[universe]",
                "exclude_bj = true",
                "max_price = 80",
                "min_history_days = 120",
                "",
                "[portfolio]",
                "max_positions = 3",
                "capital = 50000",
                "lot_size = 100",
                "cash_buffer_pct = 0.05",
                "max_position_pct = 0.25",
                "max_industry_positions = 1",
                "rank_start = 6",
                "",
                "[[factors]]",
                'name = "momentum_20d"',
            ]
        )
    )

    data_config, pipeline_config = load_pipeline_config(config_path)

    assert data_config.kind == "sqlite"
    assert data_config.path == tmp_path / "market.db"
    assert pipeline_config.max_positions == 3
    assert pipeline_config.universe.exclude_bj is True
    assert pipeline_config.universe.max_price == 80
    assert pipeline_config.universe.min_history_days == 120
    assert pipeline_config.capital == 50_000
    assert pipeline_config.lot_size == 100
    assert pipeline_config.cash_buffer_pct == 0.05
    assert pipeline_config.max_position_pct == 0.25
    assert pipeline_config.max_industry_positions == 1
    assert pipeline_config.rank_start == 6


def test_append_market_data_to_sqlite_merges_by_date_and_code(tmp_path: Path) -> None:
    db_path = tmp_path / "market.db"
    first = MarketDataBundle(
        prices=pd.DataFrame(
            [["2026-01-02", "000001.SZ", 9, 10, 8, 9.5, 900, 8_550, 0, 0, 0]],
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
            ],
        ),
        stocks=pd.DataFrame(
            [["000001.SZ", "PingAn", "Bank", "1991-04-03", 0]],
            columns=["code", "name", "industry", "list_date", "is_st"],
        ),
    )
    second = MarketDataBundle(
        prices=pd.DataFrame(
            [
                ["2026-01-02", "000001.SZ", 10, 11, 9, 10.5, 1000, 10_500, 0, 0, 0],
                ["2026-01-03", "000001.SZ", 11, 12, 10, 11.5, 1100, 12_650, 0, 0, 0],
            ],
            columns=first.prices.columns,
        ),
        stocks=pd.DataFrame(
            [["000001.SZ", "PingAn Bank", "Bank", "1991-04-03", 0]],
            columns=["code", "name", "industry", "list_date", "is_st"],
        ),
    )

    append_market_data_to_sqlite(db_path, first)
    append_market_data_to_sqlite(db_path, second)
    loaded = SqliteDataProvider(db_path).load()

    assert loaded.prices[["date", "code", "close"]].to_dict("records") == [
        {"date": pd.Timestamp("2026-01-02"), "code": "000001.SZ", "close": 10.5},
        {"date": pd.Timestamp("2026-01-03"), "code": "000001.SZ", "close": 11.5},
    ]
    assert loaded.stocks.loc[0, "name"] == "PingAn Bank"


def test_upsert_market_data_to_sqlite_replaces_keys_without_full_table_merge(tmp_path: Path) -> None:
    db_path = tmp_path / "market.db"
    columns = [
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
    write_market_data_to_sqlite(
        db_path,
        MarketDataBundle(
            prices=pd.DataFrame(
                [["2026-01-02", "000001.SZ", 9, 10, 8, 9.5, 900, 8_550, 0, 0, 0]],
                columns=columns,
            )
        ),
    )

    upsert_market_data_to_sqlite(
        db_path,
        MarketDataBundle(
            prices=pd.DataFrame(
                [
                    ["2026-01-02", "000001.SZ", 9, 10, 8, 10.0, 1000, 10_000, 0, 0, 0],
                    ["2026-01-03", "000001.SZ", 10, 11, 9, 10.5, 1100, 11_550, 0, 0, 0],
                ],
                columns=columns,
            )
        ),
    )

    prices = SqliteDataProvider(db_path).load().prices

    assert len(prices) == 2
    assert prices.loc[prices["date"] == pd.Timestamp("2026-01-02"), "close"].iloc[0] == 10.0


def test_sqlite_writer_creates_upsert_key_indexes(tmp_path: Path) -> None:
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
                [["000001.SZ", "PingAn", "Bank", "1991-04-03", 0]],
                columns=["code", "name", "industry", "list_date", "is_st"],
            )
        ),
    )

    with sqlite3.connect(db_path) as connection:
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }

    assert "idx_prices_date_code" in indexes
    assert "idx_fundamentals_date_code" in indexes
    assert "idx_stocks_code" in indexes
    assert "idx_events_date_code_type" in indexes
    assert "idx_money_flow_date_code" in indexes
