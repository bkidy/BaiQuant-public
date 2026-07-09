from __future__ import annotations

import pandas as pd


def classify_market_regimes(
    regime: pd.DataFrame,
    weak_breadth_threshold: float = 0.45,
    broad_breadth_threshold: float = 0.60,
    overheat_dist_ma60: float = 0.18,
) -> pd.DataFrame:
    frame = regime.copy()
    if frame.empty:
        return pd.DataFrame(columns=list(frame.columns) + ["dist_ma60", "market_regime", "regime_action"])
    required = {"date", "market_equity", "market_ma60", "breadth_ma20"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"regime missing required columns: {sorted(missing)}")

    frame["date"] = pd.to_datetime(frame["date"])
    frame["dist_ma60"] = (
        pd.to_numeric(frame["dist_ma60"], errors="coerce")
        if "dist_ma60" in frame.columns
        else frame["market_equity"] / frame["market_ma60"] - 1
    )
    trend_up = frame["market_equity"] > frame["market_ma60"]
    breadth = pd.to_numeric(frame["breadth_ma20"], errors="coerce")
    dist = pd.to_numeric(frame["dist_ma60"], errors="coerce")

    frame["market_regime"] = "downtrend"
    frame.loc[trend_up & (breadth < weak_breadth_threshold), "market_regime"] = "weak_breadth"
    frame.loc[
        trend_up & (breadth >= weak_breadth_threshold) & (breadth < broad_breadth_threshold),
        "market_regime",
    ] = "narrow_uptrend"
    frame.loc[trend_up & (breadth >= broad_breadth_threshold), "market_regime"] = "broad_uptrend"
    frame.loc[trend_up & (dist > overheat_dist_ma60), "market_regime"] = "overheated_uptrend"
    frame["regime_action"] = frame["market_regime"].map(
        {
            "broad_uptrend": "trade_top_6_10",
            "narrow_uptrend": "watch_light",
            "weak_breadth": "watch_only",
            "overheated_uptrend": "avoid_new_buys",
            "downtrend": "cash",
        }
    )
    return frame


def compute_forward_returns(
    prices: pd.DataFrame,
    signal_dates: list[pd.Timestamp],
    holding_days: int,
) -> pd.DataFrame:
    price_frame = prices.copy()
    price_frame["date"] = pd.to_datetime(price_frame["date"])
    close = price_frame.pivot(index="date", columns="code", values="close").sort_index()
    rows: list[dict[str, float | str | pd.Timestamp]] = []

    for signal_date in signal_dates:
        date = pd.Timestamp(signal_date)
        if date not in close.index:
            continue
        start_index = close.index.get_loc(date)
        end_index = start_index + holding_days
        if end_index >= len(close.index):
            continue
        start_prices = close.iloc[start_index]
        end_prices = close.iloc[end_index]
        returns = end_prices / start_prices - 1
        for code, value in returns.dropna().items():
            rows.append({"date": date, "code": code, "forward_return": float(value)})
    return pd.DataFrame(rows, columns=["date", "code", "forward_return"])


def factor_rank_ic(
    factors: pd.DataFrame,
    forward_returns: pd.DataFrame,
    factor_name: str,
) -> pd.DataFrame:
    merged = _merge_factor_forward(factors, forward_returns, factor_name)
    rows: list[dict[str, float | int | pd.Timestamp]] = []
    for date, group in merged.groupby("date", sort=True):
        valid = group[[factor_name, "forward_return"]].dropna()
        if len(valid) < 2:
            continue
        if valid[factor_name].nunique() < 2 or valid["forward_return"].nunique() < 2:
            continue
        rank_ic = valid[factor_name].rank().corr(valid["forward_return"].rank())
        if pd.isna(rank_ic):
            continue
        rows.append({"date": date, "rank_ic": float(rank_ic), "count": int(len(valid))})
    return pd.DataFrame(rows, columns=["date", "rank_ic", "count"])


def top_slice_returns(
    factors: pd.DataFrame,
    forward_returns: pd.DataFrame,
    factor_name: str,
    slices: list[tuple[str, int, int]] | None = None,
) -> pd.DataFrame:
    slices = slices or [("top_1_5", 1, 5), ("top_6_10", 6, 10)]
    merged = _merge_factor_forward(factors, forward_returns, factor_name)
    rows: list[dict[str, float | int | str | pd.Timestamp]] = []
    for date, group in merged.groupby("date", sort=True):
        ranked = group.dropna(subset=[factor_name, "forward_return"]).sort_values(
            [factor_name, "code"],
            ascending=[False, True],
        )
        ranked = ranked.reset_index(drop=True)
        ranked["rank"] = ranked.index + 1
        for label, start, end in slices:
            window = ranked.loc[(ranked["rank"] >= start) & (ranked["rank"] <= end)]
            if window.empty:
                continue
            rows.append(
                {
                    "date": date,
                    "slice": label,
                    "start_rank": start,
                    "end_rank": end,
                    "mean_forward_return": float(window["forward_return"].mean()),
                    "count": int(len(window)),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["date", "slice", "start_rank", "end_rank", "mean_forward_return", "count"],
    )


def evaluate_ranked_signal_quality(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    horizons: list[int] | tuple[int, ...] = (5, 10, 20),
    stable_drawdown_floor: float = -0.08,
    min_gain_retention: float = 0.5,
) -> pd.DataFrame:
    signal_frame = signals.copy()
    price_frame = prices.copy()
    if signal_frame.empty or price_frame.empty:
        return _empty_signal_quality_frame()
    required_signal_columns = {"date", "code", "score_rank"}
    missing_signal_columns = required_signal_columns - set(signal_frame.columns)
    if missing_signal_columns:
        raise ValueError(f"signals missing required columns: {sorted(missing_signal_columns)}")
    if not {"date", "code", "close"}.issubset(price_frame.columns):
        raise ValueError("prices must include date, code, close columns")

    signal_frame["date"] = pd.to_datetime(signal_frame["date"])
    signal_frame["code"] = signal_frame["code"].astype(str)
    signal_frame["score_rank"] = pd.to_numeric(signal_frame["score_rank"], errors="coerce")
    if "raw_rank" in signal_frame.columns:
        signal_frame["raw_rank"] = pd.to_numeric(signal_frame["raw_rank"], errors="coerce")
    else:
        signal_frame["raw_rank"] = signal_frame["score_rank"]
    signal_frame = signal_frame.dropna(subset=["date", "code", "score_rank"]).sort_values(["date", "score_rank", "code"])
    price_frame["date"] = pd.to_datetime(price_frame["date"])
    price_frame["code"] = price_frame["code"].astype(str)
    close = price_frame.pivot(index="date", columns="code", values="close").sort_index()
    rows: list[dict[str, bool | float | int | str | pd.Timestamp]] = []

    for _, signal in signal_frame.iterrows():
        signal_date = pd.Timestamp(signal["date"])
        code = str(signal["code"])
        if signal_date not in close.index or code not in close.columns:
            continue
        start_index = close.index.get_loc(signal_date)
        if not isinstance(start_index, int):
            continue
        start_close = close.iloc[start_index][code]
        if pd.isna(start_close) or float(start_close) <= 0:
            continue
        for horizon in horizons:
            end_index = start_index + int(horizon)
            if end_index >= len(close.index):
                continue
            end_close = close.iloc[end_index][code]
            if pd.isna(end_close):
                continue
            raw_rank = int(signal["raw_rank"]) if pd.notna(signal["raw_rank"]) else int(signal["score_rank"])
            window = close.iloc[start_index : end_index + 1][code].dropna()
            if window.empty:
                continue
            forward_return = float(end_close) / float(start_close) - 1
            path_returns = window / float(start_close) - 1
            max_gain = float(path_returns.max())
            max_drawdown = float(path_returns.min())
            raw_gain_retention = float(forward_return / max_gain) if max_gain > 0 else 0.0
            gain_retention = min(max(raw_gain_retention, 0.0), 1.0)
            stable_gain = (
                forward_return > 0
                and max_drawdown >= stable_drawdown_floor
                and gain_retention >= min_gain_retention
            )
            rows.append(
                {
                    "date": signal_date,
                    "code": code,
                    "raw_rank": raw_rank,
                    "score_rank": int(signal["score_rank"]),
                    "rank_bucket": _rank_bucket(raw_rank),
                    "horizon_days": int(horizon),
                    "start_close": float(start_close),
                    "end_close": float(end_close),
                    "forward_return": forward_return,
                    "max_gain": max_gain,
                    "max_drawdown": max_drawdown,
                    "gain_retention": gain_retention,
                    "stable_gain": bool(stable_gain),
                }
            )
    return pd.DataFrame(rows, columns=_signal_quality_columns())


def summarize_signal_quality(details: pd.DataFrame) -> pd.DataFrame:
    if details.empty:
        return _empty_signal_quality_summary_frame()
    frame = details.copy()
    frame["horizon_days"] = pd.to_numeric(frame["horizon_days"], errors="coerce")
    frame["forward_return"] = pd.to_numeric(frame["forward_return"], errors="coerce")
    frame["max_gain"] = pd.to_numeric(frame["max_gain"], errors="coerce")
    frame["max_drawdown"] = pd.to_numeric(frame["max_drawdown"], errors="coerce")
    frame["gain_retention"] = pd.to_numeric(frame["gain_retention"], errors="coerce")
    frame["stable_gain"] = frame["stable_gain"].astype(bool)
    rows: list[dict[str, float | int | str]] = []
    for (horizon, bucket), group in frame.groupby(["horizon_days", "rank_bucket"], sort=False):
        valid = group.dropna(subset=["forward_return"])
        if valid.empty:
            continue
        rows.append(
            {
                "horizon_days": int(horizon),
                "rank_bucket": str(bucket),
                "count": int(len(valid)),
                "positive_rate": float((valid["forward_return"] > 0).mean()),
                "stable_rate": float(valid["stable_gain"].mean()),
                "mean_forward_return": float(valid["forward_return"].mean()),
                "median_forward_return": float(valid["forward_return"].median()),
                "mean_max_gain": float(valid["max_gain"].mean()),
                "mean_max_drawdown": float(valid["max_drawdown"].mean()),
                "mean_gain_retention": float(valid["gain_retention"].mean()),
            }
        )
    if not rows:
        return _empty_signal_quality_summary_frame()
    summary = pd.DataFrame(rows, columns=_signal_quality_summary_columns())
    summary["_bucket_order"] = summary["rank_bucket"].map(_rank_bucket_order).fillna(99)
    summary = summary.sort_values(["horizon_days", "_bucket_order", "rank_bucket"]).drop(columns="_bucket_order")
    return summary.reset_index(drop=True)


def _merge_factor_forward(
    factors: pd.DataFrame,
    forward_returns: pd.DataFrame,
    factor_name: str,
) -> pd.DataFrame:
    factor_frame = factors[["date", "code", factor_name]].copy()
    forward_frame = forward_returns[["date", "code", "forward_return"]].copy()
    factor_frame["date"] = pd.to_datetime(factor_frame["date"])
    forward_frame["date"] = pd.to_datetime(forward_frame["date"])
    return factor_frame.merge(forward_frame, on=["date", "code"], how="inner")


def _rank_bucket(rank: int) -> str:
    if rank <= 5:
        return "top_1_5"
    if rank <= 10:
        return "top_6_10"
    if rank <= 20:
        return "top_11_20"
    return "top_21_plus"


def _rank_bucket_order(bucket: str) -> int:
    return {
        "top_1_5": 1,
        "top_6_10": 2,
        "top_11_20": 3,
        "top_21_plus": 4,
    }.get(bucket, 99)


def _signal_quality_columns() -> list[str]:
    return [
        "date",
        "code",
        "raw_rank",
        "score_rank",
        "rank_bucket",
        "horizon_days",
        "start_close",
        "end_close",
        "forward_return",
        "max_gain",
        "max_drawdown",
        "gain_retention",
        "stable_gain",
    ]


def _empty_signal_quality_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_signal_quality_columns())


def _signal_quality_summary_columns() -> list[str]:
    return [
        "horizon_days",
        "rank_bucket",
        "count",
        "positive_rate",
        "stable_rate",
        "mean_forward_return",
        "median_forward_return",
        "mean_max_gain",
        "mean_max_drawdown",
        "mean_gain_retention",
    ]


def _empty_signal_quality_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_signal_quality_summary_columns())
