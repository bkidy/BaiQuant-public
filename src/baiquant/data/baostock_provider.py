from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from baiquant.data.bundle import MarketDataBundle
from baiquant.data.sqlite_provider import SqliteDataProvider, append_market_data_to_sqlite


@dataclass(slots=True)
class BaoStockIngestConfig:
    output_path: str | Path
    symbols: list[str] | None = None
    year: int = 2025
    quarter: int = 4
    limit: int | None = None
    offset: int = 0
    sleep_seconds: float = 0.0
    continue_on_error: bool = True


def ingest_baostock(config: BaoStockIngestConfig, bs: Any | None = None) -> dict[str, int]:
    bs_module = bs or _load_baostock()
    login = bs_module.login()
    if getattr(login, "error_code", "0") != "0":
        raise RuntimeError(f"BaoStock login failed: {getattr(login, 'error_msg', '')}")

    try:
        existing = SqliteDataProvider(config.output_path).load()
        symbols = _select_symbols(existing.stocks["code"].tolist(), config.symbols, config.limit, config.offset)
        stocks = _fetch_stocks(bs_module, symbols, existing.stocks)
        fundamentals, failures = _fetch_fundamentals(bs_module, symbols, config)
        bundle = MarketDataBundle(
            prices=pd.DataFrame(columns=existing.prices.columns),
            fundamentals=fundamentals,
            stocks=stocks,
            events=pd.DataFrame(columns=existing.events.columns),
        )
        append_market_data_to_sqlite(config.output_path, bundle)
    finally:
        bs_module.logout()

    return {
        "stocks": len(stocks),
        "fundamentals": len(fundamentals),
        "failures": len(failures),
    }


def _fetch_stocks(bs_module: Any, symbols: list[str], fallback_stocks: pd.DataFrame) -> pd.DataFrame:
    basics = _result_to_frame(bs_module.query_stock_basic())
    industries = _result_to_frame(bs_module.query_stock_industry())
    if basics.empty or "type" not in basics.columns:
        return fallback_stocks.loc[fallback_stocks["code"].isin(symbols)].copy().reset_index(drop=True)

    basics = basics.loc[basics["type"].astype(str) == "1"].copy()
    basics["code"] = basics["code"].map(from_baostock_code)
    basics = basics.loc[basics["code"].isin(symbols)]
    basics["name"] = basics["code_name"].astype(str)
    basics["list_date"] = basics["ipoDate"]
    basics["is_st"] = basics["name"].str.contains("ST", case=False, na=False).astype(int)

    if not industries.empty:
        industries = industries.copy()
        industries["code"] = industries["code"].map(from_baostock_code)
        industry = industries[["code", "industry"]]
        basics = basics.merge(industry, on="code", how="left")
    else:
        basics["industry"] = ""

    stocks = basics[["code", "name", "industry", "list_date", "is_st"]].copy()
    stocks["industry"] = stocks["industry"].fillna("")
    return stocks.sort_values("code").reset_index(drop=True)


def _fetch_fundamentals(
    bs_module: Any,
    symbols: list[str],
    config: BaoStockIngestConfig,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, code in enumerate(symbols):
        bs_code = to_baostock_code(code)
        try:
            profit = _result_to_frame(bs_module.query_profit_data(bs_code, config.year, config.quarter))
            growth = _result_to_frame(bs_module.query_growth_data(bs_code, config.year, config.quarter))
            if profit.empty:
                continue
            row = profit.iloc[0].to_dict()
            growth_row = growth.iloc[0].to_dict() if not growth.empty else {}
            rows.append(
                {
                    "date": row.get("pubDate") or row.get("statDate"),
                    "code": from_baostock_code(str(row["code"])),
                    "pe_ttm": pd.NA,
                    "pb": pd.NA,
                    "roe": _to_number(row.get("roeAvg")),
                    "revenue_yoy": pd.NA,
                    "profit_yoy": _to_number(growth_row.get("YOYNI")),
                }
            )
        except Exception as exc:
            failures.append({"code": code, "error_type": type(exc).__name__, "message": str(exc)})
            if not config.continue_on_error:
                raise
        if config.sleep_seconds > 0 and index < len(symbols) - 1:
            time.sleep(config.sleep_seconds)

    fundamentals = pd.DataFrame(
        rows,
        columns=["date", "code", "pe_ttm", "pb", "roe", "revenue_yoy", "profit_yoy"],
    )
    return fundamentals, failures


def _result_to_frame(result: Any) -> pd.DataFrame:
    rows: list[list[str]] = []
    while getattr(result, "error_code", "0") == "0" and result.next():
        rows.append(result.get_row_data())
    return pd.DataFrame(rows, columns=result.fields)


def _select_symbols(
    all_codes: list[str],
    requested: list[str] | None,
    limit: int | None,
    offset: int,
) -> list[str]:
    symbols = requested if requested else all_codes
    deduped = list(dict.fromkeys(symbols))
    if offset < 0:
        raise ValueError("offset must be >= 0")
    sliced = deduped[offset:]
    return sliced[:limit] if limit is not None else sliced


def to_baostock_code(code: str) -> str:
    raw, suffix = code.split(".")
    return f"{suffix.lower()}.{raw}"


def from_baostock_code(code: str) -> str:
    suffix, raw = code.split(".")
    return f"{raw}.{suffix.upper()}"


def _to_number(value: Any) -> float | pd.NA:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return pd.NA
    return float(numeric)


def _load_baostock() -> Any:
    try:
        import baostock as bs
    except ImportError as exc:
        raise RuntimeError(
            "BaoStock is not installed. Run `.venv/bin/python -m pip install -e '.[data]'` "
            "or `.venv/bin/python -m pip install baostock`."
        ) from exc
    return bs
