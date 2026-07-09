from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from baiquant.universe.filters import latest_by_code


@dataclass(slots=True)
class MultifactorPlanConfig:
    initial_cash: float = 50_000.0
    max_positions: int = 1
    lot_size: int = 100
    stop_loss_pct: float = 0.04
    take_profit_pct: float = 0.25
    trailing_stop_activation_pct: float = 0.10
    trailing_stop_pct: float = 0.06
    max_holding_days: int = 15
    cash_buffer_pct: float = 0.0


def build_multifactor_daily_plan(
    candidates: pd.DataFrame,
    prices: pd.DataFrame,
    as_of: str | pd.Timestamp,
    holdings: pd.DataFrame | None = None,
    cash: float | None = None,
    config: MultifactorPlanConfig | None = None,
) -> pd.DataFrame:
    config = config or MultifactorPlanConfig()
    plan_date = pd.Timestamp(as_of).normalize()
    holding_frame = _normalize_holdings(holdings)
    latest = _latest_price_index(prices, plan_date)
    rows: list[dict[str, object]] = []

    for _, holding in holding_frame.iterrows():
        rows.append(_holding_plan_row(holding, latest, plan_date, config))

    exiting_codes = {str(row["code"]) for row in rows if row["action"] == "sell_next_open"}
    held_codes = {
        str(row["code"])
        for row in rows
        if row["action"] == "hold"
    }
    active_positions = len(held_codes)
    available_cash = float(config.initial_cash if cash is None else cash)
    available_cash += _planned_sell_value(rows)
    available_cash *= 1 - config.cash_buffer_pct
    empty_slots = max(0, int(config.max_positions) - active_positions)

    if empty_slots > 0:
        rows.extend(
            _entry_plan_rows(
                candidates=candidates,
                plan_date=plan_date,
                held_codes=held_codes,
                exiting_codes=exiting_codes,
                available_cash=available_cash,
                empty_slots=empty_slots,
                config=config,
            )
        )

    return pd.DataFrame(rows, columns=_PLAN_COLUMNS)


def _latest_price_index(prices: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    latest = latest_by_code(frame, as_of)
    if latest.empty:
        return latest
    return latest.set_index("code", drop=False)


def _holding_plan_row(
    holding: pd.Series,
    latest: pd.DataFrame,
    plan_date: pd.Timestamp,
    config: MultifactorPlanConfig,
) -> dict[str, object]:
    code = str(holding["code"])
    shares = int(holding["shares"])
    if latest.empty or code not in latest.index:
        return _plan_row(
            date=plan_date,
            action="manual_review",
            code=code,
            name=str(holding.get("name", "")),
            reason="missing_price",
            shares=shares,
        )

    price_row = latest.loc[code]
    close_value = pd.to_numeric(pd.Series([price_row.get("close", np.nan)]), errors="coerce").iloc[0]
    if pd.isna(close_value) or float(close_value) <= 0:
        return _plan_row(
            date=plan_date,
            action="manual_review",
            code=code,
            name=str(holding.get("name", price_row.get("name", ""))),
            reason="missing_price",
            shares=shares,
        )

    close = float(close_value)
    average_cost = _numeric_or_default(holding.get("average_cost"), close)
    high_close = max(_numeric_or_default(holding.get("high_close"), close), close)
    reason = _exit_reason(
        close=close,
        average_cost=average_cost,
        high_close=high_close,
        entry_date=holding.get("entry_date"),
        plan_date=plan_date,
        config=config,
    )
    action = "sell_next_open" if reason else "hold"
    return _plan_row(
        date=plan_date,
        action=action,
        code=code,
        name=str(holding.get("name", price_row.get("name", ""))),
        industry=str(price_row.get("industry", "")),
        reason=reason or "existing_position",
        shares=shares,
        reference_price=close,
        average_cost=average_cost,
        current_return=close / average_cost - 1 if average_cost else np.nan,
        cash_budget=0.0,
    )


def _exit_reason(
    close: float,
    average_cost: float,
    high_close: float,
    entry_date: object,
    plan_date: pd.Timestamp,
    config: MultifactorPlanConfig,
) -> str:
    if average_cost > 0 and close <= average_cost * (1 - config.stop_loss_pct):
        return "single_stop"
    if config.take_profit_pct > 0 and average_cost > 0 and close >= average_cost * (1 + config.take_profit_pct):
        return "take_profit"
    if _trailing_stop_triggered(close, average_cost, high_close, config):
        return "trailing_stop"
    if _max_holding_days_reached(entry_date, plan_date, config.max_holding_days):
        return "max_holding_days"
    return ""


def _entry_plan_rows(
    candidates: pd.DataFrame,
    plan_date: pd.Timestamp,
    held_codes: set[str],
    exiting_codes: set[str],
    available_cash: float,
    empty_slots: int,
    config: MultifactorPlanConfig,
) -> list[dict[str, object]]:
    if candidates.empty:
        return []

    frame = candidates.copy()
    if "candidate_rank" in frame.columns:
        frame = frame.sort_values(["candidate_rank", "code"], ascending=[True, True])
    elif "score_rank" in frame.columns:
        frame = frame.sort_values(["score_rank", "code"], ascending=[True, True])
    else:
        frame = frame.sort_values(["multi_factor_score", "code"], ascending=[False, True])

    rows: list[dict[str, object]] = []
    remaining_slots = empty_slots
    cash_left = float(available_cash)
    for _, candidate in frame.iterrows():
        if remaining_slots <= 0:
            break
        code = str(candidate["code"])
        if code in held_codes or code in exiting_codes:
            continue
        price = pd.to_numeric(pd.Series([candidate.get("close", np.nan)]), errors="coerce").iloc[0]
        if pd.isna(price) or float(price) <= 0:
            continue
        cash_budget = cash_left / remaining_slots
        shares = _lot_sized_shares(cash_budget, float(price), config.lot_size)
        if shares < config.lot_size:
            continue
        position_value = shares * float(price)
        rows.append(
            _plan_row(
                date=plan_date,
                action="buy_next_open",
                code=code,
                name=str(candidate.get("name", "")),
                industry=str(candidate.get("industry", "")),
                reason="entry",
                shares=shares,
                reference_price=float(price),
                score_rank=candidate.get("score_rank", np.nan),
                candidate_rank=candidate.get("candidate_rank", np.nan),
                multi_factor_score=candidate.get("multi_factor_score", np.nan),
                cash_budget=position_value,
            )
        )
        cash_left -= position_value
        remaining_slots -= 1
    return rows


def _normalize_holdings(holdings: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["code", "name", "shares", "average_cost", "high_close", "entry_date"]
    if holdings is None or holdings.empty:
        return pd.DataFrame(columns=columns)
    frame = holdings.copy()
    if "code" not in frame.columns or "shares" not in frame.columns:
        raise ValueError("holdings must include code and shares columns")
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    frame["code"] = frame["code"].astype(str)
    frame["shares"] = pd.to_numeric(frame["shares"], errors="coerce").fillna(0).astype(int)
    frame["average_cost"] = pd.to_numeric(frame["average_cost"], errors="coerce")
    frame["high_close"] = pd.to_numeric(frame["high_close"], errors="coerce")
    frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce")
    return frame.loc[frame["shares"] > 0, columns].reset_index(drop=True)


def _numeric_or_default(value: object, default: float) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return float(default)
    return float(number)


def _trailing_stop_triggered(
    close: float,
    average_cost: float,
    high_close: float,
    config: MultifactorPlanConfig,
) -> bool:
    if config.trailing_stop_activation_pct <= 0 or config.trailing_stop_pct <= 0 or average_cost <= 0:
        return False
    activated = high_close >= average_cost * (1 + config.trailing_stop_activation_pct)
    protected = close <= high_close * (1 - config.trailing_stop_pct)
    return bool(activated and protected)


def _max_holding_days_reached(entry_date: object, plan_date: pd.Timestamp, max_holding_days: int) -> bool:
    if max_holding_days <= 0:
        return False
    entry = pd.Timestamp(entry_date) if pd.notna(entry_date) else pd.NaT
    if pd.isna(entry):
        return False
    return int((plan_date.normalize() - entry.normalize()).days) >= max_holding_days


def _planned_sell_value(rows: list[dict[str, object]]) -> float:
    total = 0.0
    for row in rows:
        if row.get("action") != "sell_next_open":
            continue
        shares = int(row.get("shares", 0) or 0)
        price = pd.to_numeric(pd.Series([row.get("reference_price")]), errors="coerce").iloc[0]
        if pd.notna(price):
            total += shares * float(price)
    return total


def _lot_sized_shares(cash_budget: float, price: float, lot_size: int) -> int:
    if cash_budget <= 0 or price <= 0 or lot_size <= 0:
        return 0
    return math.floor(cash_budget / (price * lot_size)) * lot_size


def _plan_row(
    date: pd.Timestamp,
    action: str,
    code: str,
    name: str = "",
    industry: str = "",
    reason: str = "",
    shares: int = 0,
    reference_price: float = np.nan,
    average_cost: float = np.nan,
    current_return: float = np.nan,
    score_rank: object = np.nan,
    candidate_rank: object = np.nan,
    multi_factor_score: object = np.nan,
    cash_budget: float = 0.0,
) -> dict[str, object]:
    return {
        "date": date,
        "action": action,
        "code": code,
        "name": name,
        "industry": industry,
        "reason": reason,
        "shares": int(shares),
        "reference_price": reference_price,
        "average_cost": average_cost,
        "current_return": current_return,
        "score_rank": score_rank,
        "candidate_rank": candidate_rank,
        "multi_factor_score": multi_factor_score,
        "cash_budget": cash_budget,
    }


_PLAN_COLUMNS = [
    "date",
    "action",
    "code",
    "name",
    "industry",
    "reason",
    "shares",
    "reference_price",
    "average_cost",
    "current_return",
    "score_rank",
    "candidate_rank",
    "multi_factor_score",
    "cash_budget",
]
