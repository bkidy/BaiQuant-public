from __future__ import annotations

import sqlite3
import time
import signal
import threading
from contextlib import contextmanager
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from baiquant.data.bundle import MarketDataBundle
from baiquant.data.schema import MONEY_FLOW_COLUMN_ORDER, STOCK_COLUMN_ORDER, require_columns
from baiquant.data.symbols import strip_exchange, with_exchange
from baiquant.data.sqlite_provider import append_market_data_to_sqlite


@dataclass(slots=True)
class EFinanceMoneyFlowConfig:
    output_path: str | Path
    symbols: list[str] | None = None
    start_date: str | None = None
    end_date: str | None = None
    limit: int | None = None
    offset: int = 0
    sleep_seconds: float = 0.0
    retries: int = 1
    timeout: float | None = 10.0
    flush_every: int = 50
    progress_every: int = 0
    continue_on_error: bool = True


def ingest_efinance_money_flow(
    config: EFinanceMoneyFlowConfig,
    ef: Any | None = None,
) -> dict[str, int]:
    ef_module = ef or _load_efinance()
    stock_client = getattr(ef_module, "stock", ef_module)
    stock_codes = _load_stock_codes_from_sqlite(config.output_path)
    symbols = _select_symbols(stock_codes, config.symbols, config.limit, config.offset)

    pending_frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    total_rows = 0
    for index, code in enumerate(symbols):
        try:
            history = _fetch_history_bill_with_retries(stock_client, code, config)
            money_flow = normalize_history_bill_to_money_flow(history, code)
            money_flow = _filter_dates(money_flow, config.start_date, config.end_date)
        except Exception as exc:
            failures.append({"code": code, "error_type": type(exc).__name__, "message": str(exc)})
            if not config.continue_on_error:
                if pending_frames:
                    _append_money_flow(config.output_path, pending_frames)
                    pending_frames.clear()
                _write_errors(config.output_path, failures)
                raise
            if config.progress_every > 0 and (index + 1) % config.progress_every == 0:
                print(
                    "efinance progress: "
                    f"processed={index + 1}/{len(symbols)}, rows={total_rows}, failures={len(failures)}",
                    flush=True,
                )
            if config.sleep_seconds > 0 and index < len(symbols) - 1:
                time.sleep(config.sleep_seconds)
            continue
        if not money_flow.empty:
            pending_frames.append(money_flow)
            total_rows += len(money_flow)
        if pending_frames and len(pending_frames) >= max(config.flush_every, 1):
            _append_money_flow(config.output_path, pending_frames)
            pending_frames.clear()
        if config.progress_every > 0 and (index + 1) % config.progress_every == 0:
            print(
                "efinance progress: "
                f"processed={index + 1}/{len(symbols)}, rows={total_rows}, failures={len(failures)}",
                flush=True,
            )
        if config.sleep_seconds > 0 and index < len(symbols) - 1:
            time.sleep(config.sleep_seconds)

    if pending_frames:
        _append_money_flow(config.output_path, pending_frames)
    _write_errors(config.output_path, failures)

    if config.progress_every > 0:
        print(
            "efinance progress: "
            f"processed={len(symbols)}/{len(symbols)}, rows={total_rows}, failures={len(failures)}",
            flush=True,
        )

    return {
        "money_flow": total_rows,
        "failures": len(failures),
        "requested_symbols": len(symbols),
    }


def _append_money_flow(output_path: str | Path, frames: list[pd.DataFrame]) -> None:
    combined = pd.concat(frames, ignore_index=True) if frames else _empty_money_flow()
    bundle = MarketDataBundle(
        prices=pd.DataFrame(),
        fundamentals=pd.DataFrame(),
        stocks=pd.DataFrame(),
        events=pd.DataFrame(),
        money_flow=combined,
    )
    append_market_data_to_sqlite(output_path, bundle)


def _write_errors(output_path: str | Path, failures: list[dict[str, str]]) -> None:
    errors = pd.DataFrame(failures, columns=["code", "error_type", "message"])
    with closing(sqlite3.connect(output_path)) as connection:
        with connection:
            errors.to_sql("efinance_money_flow_errors", connection, if_exists="replace", index=False)


def normalize_history_bill_to_money_flow(history: pd.DataFrame, code: str) -> pd.DataFrame:
    if history.empty:
        return _empty_money_flow()
    normalized = pd.DataFrame(
        {
            "date": pd.to_datetime(history["日期"], errors="coerce").dt.date.astype(str),
            "code": history.get("股票代码", code).astype(str).map(with_exchange),
            "main_net_inflow": pd.to_numeric(history["主力净流入"], errors="coerce"),
            "small_net_inflow": pd.to_numeric(history["小单净流入"], errors="coerce"),
            "medium_net_inflow": pd.to_numeric(history["中单净流入"], errors="coerce"),
            "large_net_inflow": pd.to_numeric(history["大单净流入"], errors="coerce"),
            "super_large_net_inflow": pd.to_numeric(history["超大单净流入"], errors="coerce"),
            "main_net_inflow_pct": pd.to_numeric(history["主力净流入占比"], errors="coerce"),
            "small_net_inflow_pct": pd.to_numeric(history["小单流入净占比"], errors="coerce"),
            "medium_net_inflow_pct": pd.to_numeric(history["中单流入净占比"], errors="coerce"),
            "large_net_inflow_pct": pd.to_numeric(history["大单流入净占比"], errors="coerce"),
            "super_large_net_inflow_pct": pd.to_numeric(history["超大单流入净占比"], errors="coerce"),
            "close": pd.to_numeric(history["收盘价"], errors="coerce"),
            "pct_change": pd.to_numeric(history["涨跌幅"], errors="coerce"),
        }
    )
    return normalized.dropna(subset=["date", "code"]).sort_values(["date", "code"]).reset_index(drop=True)


def _fetch_history_bill_with_retries(
    stock_client: Any,
    code: str,
    config: EFinanceMoneyFlowConfig,
) -> pd.DataFrame:
    raw_code = strip_exchange(code)
    last_error: Exception | None = None
    for attempt in range(config.retries + 1):
        try:
            with _timeout_after(config.timeout, f"efinance get_history_bill timed out for {code}"):
                return stock_client.get_history_bill(raw_code)
        except Exception as exc:
            last_error = exc
            if attempt < config.retries and config.sleep_seconds > 0:
                time.sleep(config.sleep_seconds)
    if last_error is None:
        raise RuntimeError(f"efinance money flow fetch failed for {code}")
    raise last_error


def _filter_dates(frame: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    if frame.empty:
        return frame
    filtered = frame.copy()
    date_values = pd.to_datetime(filtered["date"])
    if start_date:
        date_values_start = pd.Timestamp(start_date)
        filtered = filtered.loc[date_values >= date_values_start].copy()
        date_values = pd.to_datetime(filtered["date"])
    if end_date:
        filtered = filtered.loc[date_values <= pd.Timestamp(end_date)].copy()
    return filtered.reset_index(drop=True)


def _select_symbols(
    all_codes: list[str],
    requested: list[str] | None,
    limit: int | None,
    offset: int,
) -> list[str]:
    symbols = [with_exchange(code) for code in requested] if requested else list(all_codes)
    deduped = list(dict.fromkeys(symbols))
    if offset < 0:
        raise ValueError("offset must be >= 0")
    sliced = deduped[offset:]
    return sliced[:limit] if limit is not None else sliced


def _load_stock_codes_from_sqlite(path: str | Path) -> list[str]:
    db_path = Path(path)
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")
    with closing(sqlite3.connect(db_path)) as connection:
        if not _table_exists(connection, "stocks"):
            raise FileNotFoundError("Required SQLite table not found: stocks")
        stocks = pd.read_sql_query('SELECT * FROM "stocks"', connection)
    require_columns(stocks, STOCK_COLUMN_ORDER, "stocks")
    return stocks["code"].astype(str).tolist()


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    cursor = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    )
    return cursor.fetchone() is not None


@contextmanager
def _timeout_after(seconds: float | None, message: str):
    if not seconds or threading.current_thread() is not threading.main_thread():
        yield
        return
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)

    def _raise_timeout(signum, frame):  # noqa: ANN001
        raise TimeoutError(message)

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def _empty_money_flow() -> pd.DataFrame:
    return pd.DataFrame(columns=MONEY_FLOW_COLUMN_ORDER)


def _load_efinance() -> Any:
    try:
        import efinance as ef
    except ImportError as exc:
        raise RuntimeError(
            "efinance is not installed. Run `.venv/bin/python -m pip install -e '.[data]'` "
            "or `.venv/bin/python -m pip install efinance`."
        ) from exc
    return ef
