from __future__ import annotations

import math

import pandas as pd


def build_equal_weight_portfolio(
    scored: pd.DataFrame,
    max_positions: int,
    min_factor_hits: int = 1,
    rank_start: int = 1,
) -> pd.DataFrame:
    if rank_start < 1:
        raise ValueError("rank_start must be >= 1")
    candidates = scored.loc[scored["hits"] >= min_factor_hits].copy()
    candidates = candidates.sort_values(["score", "hits", "code"], ascending=[False, False, True])
    candidates = candidates.iloc[rank_start - 1 :]
    selected = candidates.head(max_positions).copy()
    if selected.empty:
        selected["rank"] = pd.Series(dtype=int)
        selected["weight"] = pd.Series(dtype=float)
        return selected
    selected["rank"] = range(1, len(selected) + 1)
    selected["weight"] = 1.0 / len(selected)
    columns = ["code", "rank", "weight", "score", "hits"]
    passthrough = [column for column in selected.columns if column not in columns]
    return selected[columns + passthrough].reset_index(drop=True)


def build_lot_sized_portfolio(
    scored: pd.DataFrame,
    capital: float,
    max_positions: int,
    min_factor_hits: int = 1,
    lot_size: int = 100,
    cash_buffer_pct: float = 0.0,
    max_position_pct: float = 1.0,
    max_industry_positions: int = 0,
    rank_start: int = 1,
) -> pd.DataFrame:
    if rank_start < 1:
        raise ValueError("rank_start must be >= 1")
    candidates = scored.loc[scored["hits"] >= min_factor_hits].copy()
    candidates = candidates.sort_values(["score", "hits", "code"], ascending=[False, False, True])
    candidates = candidates.iloc[rank_start - 1 :]

    available_cash = capital * (1 - cash_buffer_pct)
    per_position_budget = available_cash / max_positions
    max_position_value = capital * max_position_pct
    industry_counts: dict[str, int] = {}
    rows: list[pd.Series] = []

    for _, candidate in candidates.iterrows():
        if len(rows) >= max_positions:
            break
        price = pd.to_numeric(candidate.get("close"), errors="coerce")
        if pd.isna(price) or price <= 0:
            continue

        industry = str(candidate.get("industry", ""))
        if max_industry_positions > 0 and industry:
            if industry_counts.get(industry, 0) >= max_industry_positions:
                continue

        position_budget = min(per_position_budget, max_position_value, available_cash)
        shares = math.floor(position_budget / (float(price) * lot_size)) * lot_size
        if shares < lot_size:
            continue
        position_value = shares * float(price)
        row = candidate.copy()
        row["shares"] = int(shares)
        row["position_value"] = position_value
        row["weight"] = position_value / capital
        rows.append(row)
        available_cash -= position_value
        if industry:
            industry_counts[industry] = industry_counts.get(industry, 0) + 1

    if not rows:
        empty = candidates.head(0).copy()
        empty["rank"] = pd.Series(dtype=int)
        empty["shares"] = pd.Series(dtype=int)
        empty["position_value"] = pd.Series(dtype=float)
        empty["weight"] = pd.Series(dtype=float)
        return empty

    selected = pd.DataFrame(rows).reset_index(drop=True)
    selected["rank"] = range(1, len(selected) + 1)
    columns = ["code", "rank", "weight", "shares", "position_value", "score", "hits"]
    passthrough = [column for column in selected.columns if column not in columns]
    return selected[columns + passthrough].reset_index(drop=True)
