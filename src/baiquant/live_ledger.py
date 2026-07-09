from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from baiquant.desk import HOLDING_COLUMNS, load_desk_holdings, save_desk_holdings


TRADE_LOG_COLUMNS = [
    "date",
    "time",
    "action",
    "code",
    "name",
    "shares",
    "price",
    "amount",
    "fees",
    "average_cost_before",
    "realized_pnl",
    "holdings_shares_after",
    "source",
    "note",
]


def record_live_trade(
    holdings_path: str | Path,
    trade_log_path: str | Path,
    trade_date: str,
    action: str,
    code: str,
    name: str = "",
    shares: int = 0,
    price: float = 0.0,
    fees: float = 0.0,
    trade_time: str = "",
    source: str = "manual",
    note: str = "",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    action = str(action).strip().lower()
    if action not in {"buy", "sell"}:
        raise ValueError("action must be buy or sell")
    shares = int(shares)
    if shares <= 0:
        raise ValueError("shares must be positive")
    price = float(price)
    if price <= 0:
        raise ValueError("price must be positive")
    fees = float(fees or 0.0)

    holdings = load_desk_holdings(holdings_path)
    code = _normalize_trade_code(code)
    if action == "buy":
        updated, trade = _record_buy(holdings, trade_date, trade_time, code, name, shares, price, fees, source, note)
    else:
        updated, trade = _record_sell(holdings, trade_date, trade_time, code, name, shares, price, fees, source, note)
    saved = save_desk_holdings(holdings_path, updated.to_dict("records"))
    _append_trade_log(trade_log_path, trade)
    return saved, trade


def load_trade_log(path: str | Path, limit: int | None = 50) -> pd.DataFrame:
    input_path = Path(path)
    if not input_path.exists():
        return pd.DataFrame(columns=TRADE_LOG_COLUMNS)
    frame = pd.read_csv(input_path)
    for column in TRADE_LOG_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    extra_columns = [column for column in frame.columns if column not in TRADE_LOG_COLUMNS]
    frame = frame[[*TRADE_LOG_COLUMNS, *extra_columns]]
    if limit is not None and int(limit) > 0:
        frame = frame.tail(int(limit))
    frame = frame.iloc[::-1].reset_index(drop=True)
    return frame.astype(object).where(pd.notna(frame), None)


def _record_buy(
    holdings: pd.DataFrame,
    trade_date: str,
    trade_time: str,
    code: str,
    name: str,
    shares: int,
    price: float,
    fees: float,
    source: str,
    note: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = holdings.copy()
    amount = shares * price
    existing = frame.index[frame["code"].astype(str) == code].tolist() if not frame.empty else []
    average_cost_before = None
    if existing:
        idx = existing[0]
        old_shares = int(frame.loc[idx, "shares"])
        old_cost = frame.loc[idx, "average_cost"]
        average_cost_before = float(old_cost) if pd.notna(old_cost) else price
        new_shares = old_shares + shares
        frame.loc[idx, "shares"] = new_shares
        frame.loc[idx, "average_cost"] = ((old_shares * average_cost_before) + amount + fees) / new_shares
        if name:
            frame.loc[idx, "name"] = name
        frame.loc[idx, "added"] = True
        shares_after = new_shares
    else:
        shares_after = shares
        row = {column: None for column in HOLDING_COLUMNS}
        row.update(
            {
                "code": code,
                "name": name,
                "shares": shares,
                "average_cost": (amount + fees) / shares,
                "high_close": price,
                "entry_shares": shares,
                "added": False,
                "entry_date": trade_date,
            }
        )
        frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    trade = _trade_record(
        trade_date,
        trade_time,
        "buy",
        code,
        name,
        shares,
        price,
        amount,
        fees,
        average_cost_before,
        realized_pnl=0.0,
        shares_after=shares_after,
        source=source,
        note=note,
    )
    return frame, trade


def _record_sell(
    holdings: pd.DataFrame,
    trade_date: str,
    trade_time: str,
    code: str,
    name: str,
    shares: int,
    price: float,
    fees: float,
    source: str,
    note: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = holdings.copy()
    existing = frame.index[frame["code"].astype(str) == code].tolist() if not frame.empty else []
    if not existing:
        raise ValueError(f"cannot sell {code}: not in holdings")
    idx = existing[0]
    old_shares = int(frame.loc[idx, "shares"])
    if shares > old_shares:
        raise ValueError(f"cannot sell {shares} shares of {code}: only {old_shares} held")
    average_cost = frame.loc[idx, "average_cost"]
    average_cost_before = float(average_cost) if pd.notna(average_cost) else price
    amount = shares * price
    realized_pnl = (price - average_cost_before) * shares - fees
    shares_after = old_shares - shares
    if shares_after:
        frame.loc[idx, "shares"] = shares_after
    else:
        frame = frame.drop(index=idx).reset_index(drop=True)
    trade = _trade_record(
        trade_date,
        trade_time,
        "sell",
        code,
        name or str(holdings.loc[idx, "name"] or ""),
        shares,
        price,
        amount,
        fees,
        average_cost_before,
        realized_pnl=realized_pnl,
        shares_after=shares_after,
        source=source,
        note=note,
    )
    return frame, trade


def _trade_record(
    trade_date: str,
    trade_time: str,
    action: str,
    code: str,
    name: str,
    shares: int,
    price: float,
    amount: float,
    fees: float,
    average_cost_before: float | None,
    realized_pnl: float,
    shares_after: int,
    source: str,
    note: str,
) -> dict[str, Any]:
    return {
        "date": str(trade_date),
        "time": str(trade_time or ""),
        "action": action,
        "code": code,
        "name": str(name or ""),
        "shares": int(shares),
        "price": float(price),
        "amount": float(amount),
        "fees": float(fees),
        "average_cost_before": average_cost_before,
        "realized_pnl": float(realized_pnl),
        "holdings_shares_after": int(shares_after),
        "source": str(source or "manual"),
        "note": str(note or ""),
    }


def _append_trade_log(path: str | Path, trade: dict[str, Any]) -> pd.DataFrame:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        log = pd.read_csv(output_path)
    else:
        log = pd.DataFrame(columns=TRADE_LOG_COLUMNS)
    for column in TRADE_LOG_COLUMNS:
        if column not in log.columns:
            log[column] = None
    row = pd.DataFrame([{column: trade.get(column) for column in TRADE_LOG_COLUMNS}])
    merged = pd.concat([log, row], ignore_index=True)
    extra_columns = [column for column in merged.columns if column not in TRADE_LOG_COLUMNS]
    merged = merged[[*TRADE_LOG_COLUMNS, *extra_columns]]
    merged.to_csv(output_path, index=False)
    return merged


def _normalize_trade_code(code: str) -> str:
    from baiquant.desk import _normalize_stock_code

    normalized = _normalize_stock_code(code)
    if not normalized:
        raise ValueError("code is required")
    return normalized
