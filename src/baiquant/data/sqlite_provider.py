from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from baiquant.data.bundle import MarketDataBundle
from baiquant.data.schema import (
    EVENT_COLUMN_ORDER,
    FUNDAMENTAL_COLUMN_ORDER,
    MONEY_FLOW_COLUMN_ORDER,
    PRICE_COLUMN_ORDER,
    STOCK_COLUMN_ORDER,
    parse_date_column,
    require_columns,
    sort_by_date_code,
)


TABLE_COLUMNS = {
    "prices": PRICE_COLUMN_ORDER,
    "fundamentals": FUNDAMENTAL_COLUMN_ORDER,
    "stocks": STOCK_COLUMN_ORDER,
    "events": EVENT_COLUMN_ORDER,
    "money_flow": MONEY_FLOW_COLUMN_ORDER,
}


@dataclass(slots=True)
class SqliteDataProvider:
    """Load normalized BaiQuant tables from a SQLite database."""

    path: str | Path

    def load(self) -> MarketDataBundle:
        db_path = Path(self.path)
        if not db_path.exists():
            raise FileNotFoundError(f"SQLite database not found: {db_path}")

        with closing(sqlite3.connect(db_path)) as connection:
            prices = _read_table(connection, "prices", required=True)
            fundamentals = _read_table(connection, "fundamentals", required=False)
            stocks = _read_table(connection, "stocks", required=False)
            events = _read_table(connection, "events", required=False)
            money_flow = _read_table(connection, "money_flow", required=False)

        prices = sort_by_date_code(parse_date_column(prices, "date"))
        fundamentals = sort_by_date_code(parse_date_column(fundamentals, "date"))
        stocks = parse_date_column(stocks, "list_date").reset_index(drop=True)
        events = sort_by_date_code(parse_date_column(events, "date"))
        money_flow = sort_by_date_code(parse_date_column(money_flow, "date"))

        return MarketDataBundle(
            prices=prices,
            fundamentals=fundamentals,
            stocks=stocks,
            events=events,
            money_flow=money_flow,
        )

    def load_window(
        self,
        as_of: str | pd.Timestamp,
        lookback_days: int = 540,
        fundamentals_lookback_days: int | None = None,
        events_lookback_days: int | None = None,
        money_flow_lookback_days: int | None = None,
    ) -> MarketDataBundle:
        db_path = Path(self.path)
        if not db_path.exists():
            raise FileNotFoundError(f"SQLite database not found: {db_path}")

        as_of_date = pd.Timestamp(as_of).normalize()
        end_key = as_of_date.date().isoformat()
        price_start_key = _window_start_key(as_of_date, lookback_days)
        fundamentals_start_key = _window_start_key(
            as_of_date,
            fundamentals_lookback_days if fundamentals_lookback_days is not None else lookback_days,
        )
        events_start_key = _window_start_key(
            as_of_date,
            events_lookback_days if events_lookback_days is not None else lookback_days,
        )
        money_flow_start_key = _window_start_key(
            as_of_date,
            money_flow_lookback_days if money_flow_lookback_days is not None else lookback_days,
        )

        with closing(sqlite3.connect(db_path)) as connection:
            prices = _read_dated_table_window(connection, "prices", True, price_start_key, end_key)
            fundamentals = _read_dated_table_window(
                connection,
                "fundamentals",
                False,
                fundamentals_start_key,
                end_key,
            )
            stocks = _read_table(connection, "stocks", required=False)
            events = _read_dated_table_window(connection, "events", False, events_start_key, end_key)
            money_flow = _read_dated_table_window(
                connection,
                "money_flow",
                False,
                money_flow_start_key,
                end_key,
            )

        prices = sort_by_date_code(parse_date_column(prices, "date"))
        fundamentals = sort_by_date_code(parse_date_column(fundamentals, "date"))
        stocks = parse_date_column(stocks, "list_date").reset_index(drop=True)
        events = sort_by_date_code(parse_date_column(events, "date"))
        money_flow = sort_by_date_code(parse_date_column(money_flow, "date"))

        return MarketDataBundle(
            prices=prices,
            fundamentals=fundamentals,
            stocks=stocks,
            events=events,
            money_flow=money_flow,
        )


def write_market_data_to_sqlite(path: str | Path, bundle: MarketDataBundle) -> None:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as connection:
        with connection:
            _write_table(connection, "prices", bundle.prices)
            _write_table(connection, "fundamentals", bundle.fundamentals)
            _write_table(connection, "stocks", bundle.stocks)
            _write_table(connection, "events", bundle.events)
            _write_table(connection, "money_flow", bundle.money_flow)
            _create_indexes(connection)


def append_market_data_to_sqlite(path: str | Path, bundle: MarketDataBundle) -> None:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as connection:
        with connection:
            _merge_table(connection, "prices", bundle.prices, keys=["date", "code"])
            _merge_table(connection, "fundamentals", bundle.fundamentals, keys=["date", "code"])
            _merge_table(connection, "stocks", bundle.stocks, keys=["code"])
            _merge_table(connection, "events", bundle.events, keys=["date", "code", "event_type"])
            _merge_table(connection, "money_flow", bundle.money_flow, keys=["date", "code"])
            _create_indexes(connection)


def upsert_market_data_to_sqlite(path: str | Path, bundle: MarketDataBundle) -> None:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as connection:
        with connection:
            _upsert_table(connection, "prices", bundle.prices, keys=["date", "code"])
            _upsert_table(connection, "fundamentals", bundle.fundamentals, keys=["date", "code"])
            _upsert_table(connection, "stocks", bundle.stocks, keys=["code"])
            _upsert_table(connection, "events", bundle.events, keys=["date", "code", "event_type"])
            _upsert_table(connection, "money_flow", bundle.money_flow, keys=["date", "code"])
            _create_indexes(connection)


def _read_table(connection: sqlite3.Connection, table: str, required: bool) -> pd.DataFrame:
    columns = TABLE_COLUMNS[table]
    if not _table_exists(connection, table):
        if required:
            raise FileNotFoundError(f"Required SQLite table not found: {table}")
        return pd.DataFrame(columns=columns)
    frame = pd.read_sql_query(f'SELECT * FROM "{table}"', connection)
    require_columns(frame, columns, table)
    return frame[columns]


def _read_dated_table_window(
    connection: sqlite3.Connection,
    table: str,
    required: bool,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    columns = TABLE_COLUMNS[table]
    if not _table_exists(connection, table):
        if required:
            raise FileNotFoundError(f"Required SQLite table not found: {table}")
        return pd.DataFrame(columns=columns)
    frame = pd.read_sql_query(
        f'SELECT * FROM "{table}" WHERE "date" >= ? AND "date" <= ?',
        connection,
        params=(start_date, end_date),
    )
    require_columns(frame, columns, table)
    return frame[columns]


def _window_start_key(as_of_date: pd.Timestamp, lookback_days: int) -> str:
    return (as_of_date - pd.Timedelta(days=lookback_days)).date().isoformat()


def _write_table(connection: sqlite3.Connection, table: str, frame: pd.DataFrame) -> None:
    prepared = _prepare_table_frame(table, frame)
    prepared.to_sql(table, connection, if_exists="replace", index=False)


def _merge_table(
    connection: sqlite3.Connection,
    table: str,
    frame: pd.DataFrame,
    keys: list[str],
) -> None:
    columns = TABLE_COLUMNS[table]
    incoming = _prepare_table_frame(table, frame)
    if incoming.empty:
        if not _table_exists(connection, table):
            incoming.to_sql(table, connection, if_exists="replace", index=False)
        return
    if _table_exists(connection, table):
        existing = _read_table(connection, table, required=False)
        combined = pd.concat([existing, incoming], ignore_index=True)
    else:
        combined = incoming
    if not combined.empty:
        combined = combined.drop_duplicates(subset=keys, keep="last")
        combined = combined.sort_values(keys).reset_index(drop=True)
    else:
        combined = pd.DataFrame(columns=columns)
    combined.to_sql(table, connection, if_exists="replace", index=False)


def _upsert_table(
    connection: sqlite3.Connection,
    table: str,
    frame: pd.DataFrame,
    keys: list[str],
) -> None:
    incoming = _prepare_table_frame(table, frame)
    if incoming.empty:
        if not _table_exists(connection, table):
            incoming.to_sql(table, connection, if_exists="replace", index=False)
        return
    if not _table_exists(connection, table):
        incoming.to_sql(table, connection, if_exists="replace", index=False)
        return
    if _table_row_count(connection, table) == 0:
        incoming.to_sql(table, connection, if_exists="replace", index=False)
        return

    temp_table = f"_incoming_{table}"
    incoming.to_sql(temp_table, connection, if_exists="replace", index=False)
    _create_key_index(connection, temp_table, keys, f"idx_{temp_table}_{'_'.join(keys)}")
    _create_key_index(connection, table, keys, f"idx_{table}_{'_'.join(keys)}")
    key_clause = " AND ".join(f't."{key}" = i."{key}"' for key in keys)
    columns = TABLE_COLUMNS[table]
    column_list = ", ".join(f'"{column}"' for column in columns)
    connection.execute(
        f'DELETE FROM "{table}" '
        f'WHERE rowid IN (SELECT t.rowid FROM "{table}" AS t JOIN "{temp_table}" AS i ON {key_clause})'
    )
    connection.execute(
        f'INSERT INTO "{table}" ({column_list}) SELECT {column_list} FROM "{temp_table}"'
    )
    connection.execute(f'DROP TABLE "{temp_table}"')


def _prepare_table_frame(table: str, frame: pd.DataFrame) -> pd.DataFrame:
    columns = TABLE_COLUMNS[table]
    prepared = frame.copy()
    if prepared.empty and not set(columns).issubset(prepared.columns):
        prepared = pd.DataFrame(columns=columns)
    require_columns(prepared, columns, table)
    prepared = prepared[columns].copy()
    for column in ["date", "list_date"]:
        if column in prepared.columns:
            prepared[column] = pd.to_datetime(prepared[column], errors="coerce").dt.date.astype("string")
    return prepared


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    cursor = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    )
    return cursor.fetchone() is not None


def _table_row_count(connection: sqlite3.Connection, table: str) -> int:
    cursor = connection.execute(f'SELECT COUNT(*) FROM "{table}"')
    return int(cursor.fetchone()[0])


def _create_indexes(connection: sqlite3.Connection) -> None:
    connection.execute('CREATE INDEX IF NOT EXISTS idx_prices_date_code ON prices ("date", "code")')
    connection.execute('CREATE INDEX IF NOT EXISTS idx_fundamentals_date_code ON fundamentals ("date", "code")')
    connection.execute('CREATE INDEX IF NOT EXISTS idx_stocks_code ON stocks ("code")')
    connection.execute('CREATE INDEX IF NOT EXISTS idx_events_date_code ON events ("date", "code")')
    connection.execute('CREATE INDEX IF NOT EXISTS idx_events_date_code_type ON events ("date", "code", "event_type")')
    connection.execute('CREATE INDEX IF NOT EXISTS idx_money_flow_date_code ON money_flow ("date", "code")')


def _create_key_index(
    connection: sqlite3.Connection,
    table: str,
    keys: list[str],
    name: str,
) -> None:
    key_columns = ", ".join(f'"{key}"' for key in keys)
    connection.execute(f'CREATE INDEX IF NOT EXISTS "{name}" ON "{table}" ({key_columns})')
