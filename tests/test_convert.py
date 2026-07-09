from pathlib import Path
import sqlite3

from baiquant.data.converters import convert_csv_to_sqlite
from baiquant.data.sqlite_provider import SqliteDataProvider


def test_convert_csv_to_sqlite_creates_queryable_database(tmp_path: Path) -> None:
    csv_root = tmp_path / "csv"
    csv_root.mkdir()
    (csv_root / "prices.csv").write_text(
        "\n".join(
            [
                "date,code,open,high,low,close,volume,amount,paused,limit_up,limit_down",
                "2026-01-02,000001.SZ,9,10,8,9.5,900,8550,0,0,0",
            ]
        )
    )
    (csv_root / "stocks.csv").write_text(
        "\n".join(
            [
                "code,name,industry,list_date,is_st",
                "000001.SZ,PingAn,Bank,1991-04-03,0",
            ]
        )
    )

    db_path = tmp_path / "market.db"
    summary = convert_csv_to_sqlite(csv_root, db_path)

    bundle = SqliteDataProvider(db_path).load()
    assert summary == {"prices": 1, "fundamentals": 0, "stocks": 1, "events": 0, "money_flow": 0}
    assert bundle.prices.loc[0, "code"] == "000001.SZ"
    assert bundle.stocks.loc[0, "name"] == "PingAn"
    assert bundle.money_flow.empty


def test_convert_csv_to_sqlite_replaces_existing_database(tmp_path: Path) -> None:
    csv_root = tmp_path / "csv"
    csv_root.mkdir()
    (csv_root / "prices.csv").write_text(
        "\n".join(
            [
                "date,code,open,high,low,close,volume,amount,paused,limit_up,limit_down",
                "2026-01-02,000001.SZ,9,10,8,9.5,900,8550,0,0,0",
            ]
        )
    )
    db_path = tmp_path / "market.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE ingest_errors (message TEXT)")
        connection.execute("INSERT INTO ingest_errors VALUES ('stale')")

    convert_csv_to_sqlite(csv_root, db_path)

    with sqlite3.connect(db_path) as connection:
        stale = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'ingest_errors'"
        ).fetchone()
    assert stale is None
