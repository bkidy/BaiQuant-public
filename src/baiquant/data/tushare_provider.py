from __future__ import annotations

import os
import sqlite3
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from baiquant.data.bundle import MarketDataBundle
from baiquant.data.schema import (
    EVENT_COLUMN_ORDER,
    FUNDAMENTAL_COLUMN_ORDER,
    MONEY_FLOW_COLUMN_ORDER,
    PRICE_COLUMN_ORDER,
    STOCK_COLUMN_ORDER,
)
from baiquant.data.sqlite_provider import upsert_market_data_to_sqlite, write_market_data_to_sqlite


@dataclass(slots=True)
class TushareIngestConfig:
    output_path: str | Path
    start_date: str
    end_date: str
    token: str | None = None
    token_path: str | Path = ".secrets/tushare_token"
    adjust: str = "qfq"
    write_mode: str = "replace"
    list_statuses: list[str] = field(default_factory=lambda: ["L", "D", "P"])
    include_prices: bool = True
    include_daily_basic: bool = True
    include_money_flow: bool = True
    include_limits: bool = True
    include_suspends: bool = True
    flush_every: int = 20
    progress_every: int = 20
    resume: bool = False
    sleep_seconds: float = 0.0
    workers: int = 1
    rate_limit_per_minute: int = 0
    timeout: int = 30
    retries: int = 0
    retry_sleep_seconds: float = 1.0
    continue_on_error: bool = True


def load_tushare_token(
    token: str | None = None,
    token_path: str | Path = ".secrets/tushare_token",
) -> str:
    if token and token.strip():
        return token.strip()
    for env_name in ("TUSHARE_TOKEN", "TS_TOKEN"):
        env_token = os.environ.get(env_name, "").strip()
        if env_token:
            return env_token
    path = Path(token_path)
    if path.exists():
        file_token = path.read_text().strip()
        if file_token:
            return file_token
    raise RuntimeError(
        "Tushare token not found. Set TUSHARE_TOKEN or save it to .secrets/tushare_token."
    )


def ingest_tushare(
    config: TushareIngestConfig,
    pro: Any | None = None,
) -> dict[str, int]:
    token = None if pro is not None else load_tushare_token(config.token, config.token_path)
    pro_client = pro or _load_tushare_pro(token or "", timeout=config.timeout)
    worker_client = _build_tushare_worker_client_getter(
        pro_client,
        token,
        config.timeout,
        pro is not None,
    )
    rate_limiter = _TushareRateLimiter(config.rate_limit_per_minute)
    trade_dates = _fetch_trade_dates(pro_client, config.start_date, config.end_date)
    failures: list[dict[str, str]] = []

    stocks = normalize_stock_basic_to_stocks(_fetch_stock_basic(pro_client, config.list_statuses))
    _initialize_tushare_database(config, stocks)
    loaded_dates = _loaded_resume_dates(config) if config.resume else set()
    latest_adj_factor = (
        _latest_adj_factor_by_code(pro_client, trade_dates[-1])
        if config.include_prices and config.adjust == "qfq" and trade_dates
        else None
    )
    last_close_by_code = _load_last_closes(config.output_path) if config.include_prices else {}
    pending = _empty_pending_frames()
    fetched_dates = 0
    flushed_dates = 0
    skipped_dates = 0
    total_prices = 0
    total_fundamentals = 0
    total_money_flow = 0
    processed_dates = 0
    fetched_trade_dates: list[str] = []

    for trade_date in trade_dates:
        if trade_date in loaded_dates:
            skipped_dates += 1
            processed_dates += 1
            _print_progress(config, processed_dates, len(trade_dates), fetched_dates, skipped_dates, failures)
            continue
        fetched_trade_dates.append(trade_date)

    def flush_pending() -> None:
        nonlocal pending, total_prices, total_fundamentals, total_money_flow
        counts = _flush_tushare_pending_frames(
            config.output_path,
            pending,
            config.adjust,
            latest_adj_factor,
            last_close_by_code,
        )
        total_prices += counts["prices"]
        total_fundamentals += counts["fundamentals"]
        total_money_flow += counts["money_flow"]
        pending = _empty_pending_frames()

    if max(config.workers, 1) == 1 or len(fetched_trade_dates) <= 1:
        for trade_date in fetched_trade_dates:
            try:
                fetched_pending = _fetch_tushare_trade_date_with_client(
                    config, worker_client, rate_limiter, trade_date
                )
                _extend_pending_frames(pending, fetched_pending)
                fetched_dates += 1
                flushed_dates += 1
                if flushed_dates % max(config.flush_every, 1) == 0:
                    flush_pending()
            except Exception as exc:
                failures.append(
                    {"date": trade_date, "error_type": type(exc).__name__, "message": str(exc)}
                )
                if not config.continue_on_error:
                    flush_pending()
                    _write_errors(config.output_path, failures)
                    raise
            processed_dates += 1
            _print_progress(config, processed_dates, len(trade_dates), fetched_dates, skipped_dates, failures)
    else:
        completed_frames: dict[str, dict[str, list[pd.DataFrame]]] = {}
        failed_dates: set[str] = set()
        next_flush_index = 0
        max_workers = max(config.workers, 1)
        trade_date_iter = iter(fetched_trade_dates)
        in_flight: dict[Future[dict[str, list[pd.DataFrame]]], str] = {}
        abort = False

        def submit_next_fetch(executor: ThreadPoolExecutor) -> None:
            try:
                trade_date = next(trade_date_iter)
            except StopIteration:
                return
            future = executor.submit(
                _fetch_tushare_trade_date_with_client,
                config,
                worker_client,
                rate_limiter,
                trade_date,
            )
            in_flight[future] = trade_date

        executor = ThreadPoolExecutor(max_workers=max_workers)
        try:
            for _ in range(max_workers):
                submit_next_fetch(executor)
            while in_flight:
                done, _ = wait(list(in_flight), return_when=FIRST_COMPLETED)
                for future in done:
                    trade_date = in_flight.pop(future)
                    try:
                        completed_frames[trade_date] = future.result()
                        fetched_dates += 1
                    except Exception as exc:
                        failures.append(
                            {"date": trade_date, "error_type": type(exc).__name__, "message": str(exc)}
                        )
                        failed_dates.add(trade_date)
                        if not config.continue_on_error:
                            abort = True
                            for pending_future in in_flight:
                                pending_future.cancel()
                            flush_pending()
                            _write_errors(config.output_path, failures)
                            raise
                    processed_dates += 1
                    while next_flush_index < len(fetched_trade_dates):
                        next_trade_date = fetched_trade_dates[next_flush_index]
                        if next_trade_date in completed_frames:
                            _extend_pending_frames(pending, completed_frames.pop(next_trade_date))
                            flushed_dates += 1
                            next_flush_index += 1
                            if flushed_dates % max(config.flush_every, 1) == 0:
                                flush_pending()
                        elif next_trade_date in failed_dates:
                            failed_dates.remove(next_trade_date)
                            next_flush_index += 1
                        else:
                            break
                    _print_progress(config, processed_dates, len(trade_dates), fetched_dates, skipped_dates, failures)
                    submit_next_fetch(executor)
        finally:
            executor.shutdown(wait=not abort, cancel_futures=abort)

    flush_pending()
    _write_errors(config.output_path, failures)

    return {
        "stocks": len(stocks),
        "prices": total_prices,
        "fundamentals": total_fundamentals,
        "money_flow": total_money_flow,
        "events": 0,
        "failures": len(failures),
        "trade_dates": len(trade_dates),
        "fetched_dates": fetched_dates,
        "skipped_dates": skipped_dates,
    }


def _fetch_tushare_trade_date(
    config: TushareIngestConfig,
    pro_client: Any,
    rate_limiter: "_TushareRateLimiter",
    trade_date: str,
) -> dict[str, list[pd.DataFrame]]:
    pending = _empty_pending_frames()
    if config.include_prices:
        pending["daily"].append(_call_tushare_api(config, rate_limiter, pro_client.daily, trade_date=trade_date))
    if config.include_prices and config.adjust != "none":
        pending["adj"].append(_call_tushare_api(config, rate_limiter, pro_client.adj_factor, trade_date=trade_date))
    if config.include_daily_basic:
        pending["daily_basic"].append(
            _call_tushare_api(
                config,
                rate_limiter,
                pro_client.daily_basic,
                trade_date=trade_date,
                fields="ts_code,trade_date,pe_ttm,pb",
            )
        )
    if config.include_prices and config.include_limits:
        pending["limits"].append(_call_tushare_api(config, rate_limiter, pro_client.stk_limit, trade_date=trade_date))
    if config.include_prices and config.include_suspends:
        pending["suspends"].append(
            _call_tushare_api(config, rate_limiter, pro_client.suspend_d, trade_date=trade_date, suspend_type="S")
        )
    if config.include_money_flow:
        pending["money_flow"].append(_call_tushare_api(config, rate_limiter, pro_client.moneyflow, trade_date=trade_date))
    if config.sleep_seconds > 0:
        time.sleep(config.sleep_seconds)
    return pending


def _fetch_tushare_trade_date_with_client(
    config: TushareIngestConfig,
    client_getter,
    rate_limiter: "_TushareRateLimiter",
    trade_date: str,
) -> dict[str, list[pd.DataFrame]]:
    return _fetch_tushare_trade_date(config, client_getter(), rate_limiter, trade_date)


def _call_tushare_api(config: TushareIngestConfig, rate_limiter: "_TushareRateLimiter", method, **kwargs):
    last_error: Exception | None = None
    attempts = max(config.retries, 0) + 1
    for attempt in range(attempts):
        try:
            rate_limiter.wait()
            return method(**kwargs)
        except Exception as exc:
            last_error = exc
            if attempt >= attempts - 1:
                break
            if config.retry_sleep_seconds > 0:
                time.sleep(config.retry_sleep_seconds * (attempt + 1))
    assert last_error is not None
    raise last_error


class _TushareRateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self.interval_seconds = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self.lock = threading.Lock()
        self.next_allowed_at = 0.0

    def wait(self) -> None:
        if self.interval_seconds <= 0:
            return
        with self.lock:
            now = time.monotonic()
            sleep_seconds = max(0.0, self.next_allowed_at - now)
            self.next_allowed_at = max(now, self.next_allowed_at) + self.interval_seconds
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)


def _extend_pending_frames(
    target: dict[str, list[pd.DataFrame]],
    incoming: dict[str, list[pd.DataFrame]],
) -> None:
    for name, frames in incoming.items():
        target[name].extend(frames)


def _build_tushare_worker_client_getter(
    pro_client: Any,
    token: str | None,
    timeout: int,
    use_shared_client: bool,
):
    if use_shared_client:
        return lambda: pro_client
    local = threading.local()

    def get_client() -> Any:
        client = getattr(local, "client", None)
        if client is None:
            client = _load_tushare_pro(token or "", timeout=timeout)
            local.client = client
        return client

    return get_client


def normalize_stock_basic_to_stocks(stock_basic: pd.DataFrame) -> pd.DataFrame:
    if stock_basic.empty:
        return pd.DataFrame(columns=["code", "name", "industry", "list_date", "is_st"])
    stocks = pd.DataFrame(
        {
            "code": stock_basic["ts_code"].astype(str),
            "name": stock_basic["name"].astype(str),
            "industry": stock_basic.get("industry", "").fillna("").astype(str),
            "list_date": pd.to_datetime(stock_basic["list_date"], errors="coerce").dt.date.astype(str),
        }
    )
    stocks["is_st"] = stocks["name"].str.contains("ST", case=False, na=False).astype(int)
    return stocks.drop_duplicates(subset=["code"], keep="first").sort_values("code").reset_index(drop=True)

def normalize_stock_basic_to_stocks(stock_basic: pd.DataFrame) -> pd.DataFrame:
    if stock_basic.empty:
        return pd.DataFrame(columns=["code", "name", "industry", "list_date", "is_st"])
    stocks = pd.DataFrame(
        {
            "code": stock_basic["ts_code"].astype(str),
            "name": stock_basic["name"].astype(str),
            "industry": stock_basic.get("industry", "").fillna("").astype(str),
            "list_date": pd.to_datetime(stock_basic["list_date"], errors="coerce").dt.date.astype(str),
        }
    )
    stocks["is_st"] = stocks["name"].str.contains("ST", case=False, na=False).astype(int)
    return stocks.drop_duplicates(subset=["code"], keep="first").sort_values("code").reset_index(drop=True)


def normalize_daily_to_prices(
    daily: pd.DataFrame,
    adj_factor: pd.DataFrame | None = None,
    limits: pd.DataFrame | None = None,
    suspends: pd.DataFrame | None = None,
    adjust: str = "qfq",
    latest_adj_factor: pd.Series | None = None,
    last_close_by_code: dict[str, float] | None = None,
) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(
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
        )
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame["code"] = frame["ts_code"].astype(str)
    frame["raw_close"] = pd.to_numeric(frame["close"], errors="coerce")
    for column in ["open", "high", "low", "close"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if adjust == "qfq" and adj_factor is not None and not adj_factor.empty:
        factors = adj_factor.copy()
        factors["date"] = pd.to_datetime(factors["trade_date"], errors="coerce")
        factors["code"] = factors["ts_code"].astype(str)
        factors["adj_factor"] = pd.to_numeric(factors["adj_factor"], errors="coerce")
        if latest_adj_factor is None:
            latest_factor = (
                factors.dropna(subset=["adj_factor"])
                .sort_values(["code", "date"])
                .groupby("code")["adj_factor"]
                .last()
                .rename("latest_adj_factor")
            )
        else:
            latest_factor = latest_adj_factor.rename("latest_adj_factor")
            latest_factor.index.name = "code"
        frame = frame.merge(factors[["date", "code", "adj_factor"]], on=["date", "code"], how="left")
        frame = frame.merge(latest_factor, on="code", how="left")
        ratio = frame["adj_factor"] / frame["latest_adj_factor"]
        ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        for column in ["open", "high", "low", "close"]:
            frame[column] = frame[column] * ratio
    elif adjust != "none":
        raise ValueError(f"Unsupported Tushare adjust mode: {adjust}")

    frame["volume"] = pd.to_numeric(frame.get("vol"), errors="coerce").fillna(0) * 100
    frame["amount"] = pd.to_numeric(frame.get("amount"), errors="coerce").fillna(0) * 1000
    frame["paused"] = 0
    frame["limit_up"] = 0
    frame["limit_down"] = 0
    if limits is not None and not limits.empty:
        limit_frame = limits.copy()
        limit_frame["date"] = pd.to_datetime(limit_frame["trade_date"], errors="coerce")
        limit_frame["code"] = limit_frame["ts_code"].astype(str)
        limit_frame["up_limit"] = pd.to_numeric(limit_frame["up_limit"], errors="coerce")
        limit_frame["down_limit"] = pd.to_numeric(limit_frame["down_limit"], errors="coerce")
        frame = frame.merge(
            limit_frame[["date", "code", "up_limit", "down_limit"]],
            on=["date", "code"],
            how="left",
        )
        frame["limit_up"] = _at_price_limit(frame["raw_close"], frame["up_limit"]).astype(int)
        frame["limit_down"] = _at_price_limit(frame["raw_close"], frame["down_limit"]).astype(int)

    prices = frame[
        ["date", "code", "open", "high", "low", "close", "volume", "amount", "paused", "limit_up", "limit_down"]
    ].copy()
    prices = _append_suspended_price_rows(prices, suspends, last_close_by_code)
    prices = prices.dropna(subset=["date", "code"]).sort_values(["date", "code"]).reset_index(drop=True)
    if last_close_by_code is not None:
        for row in prices.itertuples(index=False):
            if pd.notna(row.close) and float(row.close) > 0:
                last_close_by_code[str(row.code)] = float(row.close)
    return prices


def normalize_daily_basic_to_fundamentals(daily_basic: pd.DataFrame) -> pd.DataFrame:
    if daily_basic.empty:
        return pd.DataFrame(columns=["date", "code", "pe_ttm", "pb", "roe", "revenue_yoy", "profit_yoy"])
    fundamentals = pd.DataFrame(
        {
            "date": pd.to_datetime(daily_basic["trade_date"], errors="coerce"),
            "code": daily_basic["ts_code"].astype(str),
            "pe_ttm": pd.to_numeric(daily_basic.get("pe_ttm"), errors="coerce"),
            "pb": pd.to_numeric(daily_basic.get("pb"), errors="coerce"),
            "roe": pd.NA,
            "revenue_yoy": pd.NA,
            "profit_yoy": pd.NA,
        }
    )
    return fundamentals.dropna(subset=["date", "code"]).sort_values(["date", "code"]).reset_index(drop=True)


def normalize_moneyflow_to_money_flow(moneyflow: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    if moneyflow.empty:
        return pd.DataFrame(columns=MONEY_FLOW_COLUMN_ORDER)
    frame = moneyflow.copy()
    frame["date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame["code"] = frame["ts_code"].astype(str)
    for column in [
        "buy_sm_amount",
        "sell_sm_amount",
        "buy_md_amount",
        "sell_md_amount",
        "buy_lg_amount",
        "sell_lg_amount",
        "buy_elg_amount",
        "sell_elg_amount",
    ]:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce").fillna(0)

    small = (frame["buy_sm_amount"] - frame["sell_sm_amount"]) * 10_000
    medium = (frame["buy_md_amount"] - frame["sell_md_amount"]) * 10_000
    large = (frame["buy_lg_amount"] - frame["sell_lg_amount"]) * 10_000
    super_large = (frame["buy_elg_amount"] - frame["sell_elg_amount"]) * 10_000
    main = large + super_large

    if prices.empty:
        price_context = pd.DataFrame(columns=["date", "code", "close", "amount"])
    else:
        price_context = prices[["date", "code", "close", "amount"]].copy()
    price_context["date"] = pd.to_datetime(price_context["date"], errors="coerce")
    merged = pd.DataFrame(
        {
            "date": frame["date"],
            "code": frame["code"],
            "main_net_inflow": main,
            "small_net_inflow": small,
            "medium_net_inflow": medium,
            "large_net_inflow": large,
            "super_large_net_inflow": super_large,
        }
    ).merge(price_context, on=["date", "code"], how="left")
    amount = pd.to_numeric(merged["amount"], errors="coerce")
    amount = amount.where(amount != 0)
    for column in [
        "main_net_inflow",
        "small_net_inflow",
        "medium_net_inflow",
        "large_net_inflow",
        "super_large_net_inflow",
    ]:
        merged[f"{column}_pct"] = merged[column] / amount * 100
    merged["pct_change"] = pd.NA
    merged = merged.rename(
        columns={
            "main_net_inflow_pct": "main_net_inflow_pct",
            "small_net_inflow_pct": "small_net_inflow_pct",
            "medium_net_inflow_pct": "medium_net_inflow_pct",
            "large_net_inflow_pct": "large_net_inflow_pct",
            "super_large_net_inflow_pct": "super_large_net_inflow_pct",
        }
    )
    return merged[MONEY_FLOW_COLUMN_ORDER].dropna(subset=["date", "code"]).sort_values(["date", "code"]).reset_index(drop=True)


def _fetch_stock_basic(pro_client: Any, list_statuses: list[str]) -> pd.DataFrame:
    frames = []
    fields = "ts_code,name,industry,list_date"
    for status in list_statuses:
        frames.append(pro_client.stock_basic(exchange="", list_status=status, fields=fields))
    return _concat(frames)


def _fetch_trade_dates(pro_client: Any, start_date: str, end_date: str) -> list[str]:
    calendar = pro_client.trade_cal(exchange="", start_date=start_date, end_date=end_date, is_open="1")
    if calendar.empty:
        return []
    open_days = calendar.loc[pd.to_numeric(calendar.get("is_open", 1), errors="coerce").fillna(1) == 1]
    return sorted(open_days["cal_date"].astype(str).tolist())


def _initialize_tushare_database(config: TushareIngestConfig, stocks: pd.DataFrame) -> None:
    bundle = MarketDataBundle(
        prices=pd.DataFrame(columns=PRICE_COLUMN_ORDER),
        fundamentals=pd.DataFrame(columns=FUNDAMENTAL_COLUMN_ORDER),
        stocks=stocks,
        events=pd.DataFrame(columns=EVENT_COLUMN_ORDER),
        money_flow=pd.DataFrame(columns=MONEY_FLOW_COLUMN_ORDER),
    )
    if config.write_mode == "replace":
        write_market_data_to_sqlite(config.output_path, bundle)
    elif config.write_mode == "append":
        upsert_market_data_to_sqlite(config.output_path, bundle)
    else:
        raise ValueError(f"Unsupported write mode: {config.write_mode}")


def _empty_pending_frames() -> dict[str, list[pd.DataFrame]]:
    return {
        "daily": [],
        "adj": [],
        "daily_basic": [],
        "limits": [],
        "suspends": [],
        "money_flow": [],
    }


def _flush_tushare_pending_frames(
    output_path: str | Path,
    pending: dict[str, list[pd.DataFrame]],
    adjust: str,
    latest_adj_factor: pd.Series | None,
    last_close_by_code: dict[str, float],
) -> dict[str, int]:
    daily = _concat(pending["daily"])
    if daily.empty:
        prices = pd.DataFrame(columns=PRICE_COLUMN_ORDER)
    else:
        prices = normalize_daily_to_prices(
            daily,
            adj_factor=_concat(pending["adj"]),
            limits=_concat(pending["limits"]),
            suspends=_concat(pending["suspends"]),
            adjust=adjust,
            latest_adj_factor=latest_adj_factor,
            last_close_by_code=last_close_by_code,
        )
    fundamentals = normalize_daily_basic_to_fundamentals(_concat(pending["daily_basic"]))
    raw_money_flow = _concat(pending["money_flow"])
    money_flow_prices = prices
    if not raw_money_flow.empty and prices.empty:
        money_flow_prices = _load_price_context(output_path, _trade_dates_from_frame(raw_money_flow))
    money_flow = normalize_moneyflow_to_money_flow(raw_money_flow, money_flow_prices)
    if prices.empty and fundamentals.empty and money_flow.empty:
        return {"prices": 0, "fundamentals": 0, "money_flow": 0}
    upsert_market_data_to_sqlite(
        output_path,
        MarketDataBundle(
            prices=prices,
            fundamentals=fundamentals,
            stocks=pd.DataFrame(columns=STOCK_COLUMN_ORDER),
            events=pd.DataFrame(columns=EVENT_COLUMN_ORDER),
            money_flow=money_flow,
        ),
    )
    return {
        "prices": len(prices),
        "fundamentals": len(fundamentals),
        "money_flow": len(money_flow),
    }


def _latest_adj_factor_by_code(pro_client: Any, trade_date: str) -> pd.Series:
    factors = pro_client.adj_factor(trade_date=trade_date)
    if factors.empty:
        empty = pd.Series(dtype=float, name="adj_factor")
        empty.index.name = "code"
        return empty
    frame = factors.copy()
    frame["code"] = frame["ts_code"].astype(str)
    frame["adj_factor"] = pd.to_numeric(frame["adj_factor"], errors="coerce")
    return frame.dropna(subset=["adj_factor"]).set_index("code")["adj_factor"]


def _loaded_resume_dates(config: TushareIngestConfig) -> set[str]:
    if config.include_prices:
        return _loaded_table_dates(config.output_path, "prices")
    if config.include_money_flow:
        return _loaded_table_dates(config.output_path, "money_flow")
    if config.include_daily_basic:
        return _loaded_table_dates(config.output_path, "fundamentals")
    return set()


def _loaded_price_dates(output_path: str | Path) -> set[str]:
    return _loaded_table_dates(output_path, "prices")


def _loaded_table_dates(output_path: str | Path, table: str) -> set[str]:
    path = Path(output_path)
    if not path.exists():
        return set()
    with closing(sqlite3.connect(path)) as connection:
        if not _table_exists(connection, table):
            return set()
        frame = pd.read_sql_query(f'SELECT DISTINCT "date" FROM "{table}"', connection)
    if frame.empty:
        return set()
    return set(pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y%m%d").dropna())


def _load_last_closes(output_path: str | Path) -> dict[str, float]:
    path = Path(output_path)
    if not path.exists():
        return {}
    query = """
        SELECT p.code, p.close
        FROM prices p
        JOIN (
            SELECT code, MAX(date) AS date
            FROM prices
            GROUP BY code
        ) latest ON latest.code = p.code AND latest.date = p.date
    """
    with closing(sqlite3.connect(path)) as connection:
        if not _table_exists(connection, "prices"):
            return {}
        frame = pd.read_sql_query(query, connection)
    return {
        str(row.code): float(row.close)
        for row in frame.itertuples(index=False)
        if pd.notna(row.close) and float(row.close) > 0
    }


def _load_price_context(output_path: str | Path, trade_dates: list[str]) -> pd.DataFrame:
    if not trade_dates:
        return pd.DataFrame(columns=["date", "code", "close", "amount"])
    path = Path(output_path)
    if not path.exists():
        return pd.DataFrame(columns=["date", "code", "close", "amount"])
    normalized_dates = [
        value
        for value in pd.to_datetime(pd.Series(trade_dates), errors="coerce").dt.date.astype("string").dropna()
    ]
    if not normalized_dates:
        return pd.DataFrame(columns=["date", "code", "close", "amount"])
    placeholders = ", ".join("?" for _ in normalized_dates)
    query = f'SELECT "date", "code", "close", "amount" FROM prices WHERE "date" IN ({placeholders})'
    with closing(sqlite3.connect(path)) as connection:
        if not _table_exists(connection, "prices"):
            return pd.DataFrame(columns=["date", "code", "close", "amount"])
        return pd.read_sql_query(query, connection, params=normalized_dates)


def _trade_dates_from_frame(frame: pd.DataFrame) -> list[str]:
    if "trade_date" not in frame.columns:
        return []
    return sorted(set(frame["trade_date"].astype(str).tolist()))


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    cursor = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    )
    return cursor.fetchone() is not None


def _print_progress(
    config: TushareIngestConfig,
    current: int,
    total: int,
    fetched_dates: int,
    skipped_dates: int,
    failures: list[dict[str, str]],
) -> None:
    if config.progress_every <= 0:
        return
    if current % config.progress_every != 0 and current != total:
        return
    print(
        "tushare progress: "
        f"processed={current}/{total}, fetched_dates={fetched_dates}, "
        f"skipped_dates={skipped_dates}, failures={len(failures)}",
        flush=True,
    )


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _at_price_limit(close: pd.Series, limit_price: pd.Series) -> pd.Series:
    close_num = pd.to_numeric(close, errors="coerce")
    limit_num = pd.to_numeric(limit_price, errors="coerce")
    tolerance = np.maximum(0.01, limit_num.abs() * 0.001)
    return (close_num - limit_num).abs() <= tolerance


def _append_suspended_price_rows(
    prices: pd.DataFrame,
    suspends: pd.DataFrame | None,
    last_close_by_code: dict[str, float] | None = None,
) -> pd.DataFrame:
    if suspends is None or suspends.empty or prices.empty:
        return prices
    existing_keys = set(zip(prices["date"], prices["code"]))
    sorted_prices = prices.sort_values(["code", "date"])
    rows = []
    suspend_frame = suspends.copy()
    suspend_frame["date"] = pd.to_datetime(suspend_frame["trade_date"], errors="coerce")
    suspend_frame["code"] = suspend_frame["ts_code"].astype(str)
    for row in suspend_frame.itertuples(index=False):
        key = (row.date, row.code)
        if key in existing_keys:
            continue
        history = sorted_prices.loc[
            (sorted_prices["code"] == row.code) & (sorted_prices["date"] < row.date)
        ]
        if history.empty:
            close = (last_close_by_code or {}).get(str(row.code))
            if close is None:
                continue
        else:
            previous = history.iloc[-1]
            close = float(previous["close"])
        rows.append(
            {
                "date": row.date,
                "code": row.code,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 0.0,
                "amount": 0.0,
                "paused": 1,
                "limit_up": 0,
                "limit_down": 0,
            }
        )
    if not rows:
        return prices
    return pd.concat([prices, pd.DataFrame(rows)], ignore_index=True)


def _write_errors(output_path: str | Path, failures: list[dict[str, str]]) -> None:
    errors = pd.DataFrame(failures, columns=["date", "error_type", "message"])
    with closing(sqlite3.connect(output_path)) as connection:
        with connection:
            errors.to_sql("tushare_ingest_errors", connection, if_exists="replace", index=False)


def _load_tushare_pro(token: str, timeout: int = 30) -> Any:
    try:
        import tushare as ts
    except ImportError as exc:
        raise RuntimeError(
            "Tushare is not installed. Run `.venv/bin/python -m pip install -e '.[data]'` "
            "or `.venv/bin/python -m pip install tushare`."
        ) from exc
    return ts.pro_api(token, timeout=timeout)
