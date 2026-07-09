from __future__ import annotations

import numpy as np
import pandas as pd

from baiquant.data.bundle import MarketDataBundle
from baiquant.research.factor_diagnostics import factor_rank_ic, top_slice_returns
from baiquant.scoring import robust_zscore
from baiquant.universe.filters import UniverseConfig, build_universe


DEFAULT_FACTOR_NAMES = [
    "momentum_3d",
    "momentum_5d",
    "momentum_10d",
    "momentum_20d",
    "momentum_60d",
    "reversal_5d",
    "volatility_20d",
    "volume_ratio_20d",
    "close_position_20d",
    "ma5_distance",
    "ma20_distance",
    "money_flow_pct",
    "big_order_pct",
    "pe_ttm",
    "pb",
    "roe",
    "revenue_yoy",
    "profit_yoy",
    "industry_momentum_3d",
]
DEFAULT_MULTIFACTOR_WEIGHTS = {
    "momentum_3d": 0.50,
    "momentum_5d": 0.70,
    "momentum_10d": 0.80,
    "momentum_20d": 1.00,
    "momentum_60d": 0.35,
    "reversal_5d": 0.90,
    "volatility_20d": -0.45,
    "volume_ratio_20d": 0.40,
    "close_position_20d": 0.35,
    "ma5_distance": 0.25,
    "ma20_distance": 0.45,
    "money_flow_pct": 0.70,
    "big_order_pct": 0.50,
    "pe_ttm": -0.25,
    "pb": -0.15,
    "roe": 0.35,
    "revenue_yoy": 0.25,
    "profit_yoy": 0.35,
    "industry_momentum_3d": 0.60,
}


def build_multifactor_frame(
    bundle: MarketDataBundle,
    as_of: str | pd.Timestamp,
    universe_config: UniverseConfig | None = None,
) -> pd.DataFrame:
    """Build one cross-sectional, point-in-time factor frame for an A-share signal date."""
    as_of_date = pd.Timestamp(as_of).normalize()
    universe = build_universe(bundle, as_of_date, universe_config or UniverseConfig(min_history_days=6))
    if universe.empty:
        return _empty_multifactor_frame()

    history = bundle.prices.copy()
    history["date"] = pd.to_datetime(history["date"])
    history = history.loc[history["date"] <= as_of_date].sort_values(["code", "date"]).reset_index(drop=True)
    universe_codes = set(universe["code"].astype(str))
    history = history.loc[history["code"].astype(str).isin(universe_codes)].copy()

    factor_frame = universe[
        [column for column in ["code", "name", "industry", "close", "amount"] if column in universe.columns]
    ].copy()
    factor_frame["date"] = as_of_date

    price_factors = _price_factor_frame(history)
    factor_frame = factor_frame.merge(price_factors, on="code", how="left")
    factor_frame = factor_frame.merge(_latest_money_flow_factors(bundle.money_flow, as_of_date), on="code", how="left")
    factor_frame = factor_frame.merge(_latest_fundamental_factors(bundle.fundamentals, as_of_date), on="code", how="left")
    factor_frame = _add_industry_factors(factor_frame)

    ordered = ["date", "code", "name", "industry", "close", "amount", *DEFAULT_FACTOR_NAMES]
    for column in ordered:
        if column not in factor_frame.columns:
            factor_frame[column] = np.nan
    return factor_frame[ordered].sort_values(["date", "code"]).reset_index(drop=True)


def evaluate_multifactor_factors(
    factors: pd.DataFrame,
    forward_returns: pd.DataFrame,
    factor_names: list[str] | tuple[str, ...] | None = None,
    slices: list[tuple[str, int, int]] | None = None,
) -> pd.DataFrame:
    """Summarize rank IC and top-slice returns for a set of candidate factors."""
    factor_names = list(factor_names or _numeric_factor_columns(factors))
    slices = slices or [("top_1_5", 1, 5), ("top_6_10", 6, 10), ("top_11_20", 11, 20)]
    rows: list[dict[str, float | int | str]] = []
    for factor in factor_names:
        if factor not in factors.columns:
            continue
        valid_factor_rows = int(pd.to_numeric(factors[factor], errors="coerce").notna().sum())
        total_factor_rows = int(len(factors))
        ic = factor_rank_ic(factors, forward_returns, factor)
        slice_returns = top_slice_returns(factors, forward_returns, factor, slices=slices)
        row: dict[str, float | int | str] = {
            "factor": factor,
            "valid_factor_rows": valid_factor_rows,
            "factor_coverage_rate": float(valid_factor_rows / total_factor_rows) if total_factor_rows else np.nan,
            "mean_rank_ic": float(ic["rank_ic"].mean()) if not ic.empty else np.nan,
            "median_rank_ic": float(ic["rank_ic"].median()) if not ic.empty else np.nan,
            "positive_ic_rate": float((ic["rank_ic"] > 0).mean()) if not ic.empty else np.nan,
            "ic_observations": int(len(ic)),
        }
        for label, _, _ in slices:
            values = slice_returns.loc[slice_returns["slice"] == label, "mean_forward_return"]
            row[f"{label}_mean_forward_return"] = float(values.mean()) if not values.empty else np.nan
        rows.append(row)

    columns = [
        "factor",
        "valid_factor_rows",
        "factor_coverage_rate",
        "mean_rank_ic",
        "median_rank_ic",
        "positive_ic_rate",
        "ic_observations",
        *[f"{label}_mean_forward_return" for label, _, _ in slices],
    ]
    return pd.DataFrame(rows, columns=columns).sort_values("mean_rank_ic", ascending=False).reset_index(drop=True)


def score_multifactor_frame(
    factors: pd.DataFrame,
    factor_weights: dict[str, float],
    top_n: int | None = None,
) -> pd.DataFrame:
    """Score a factor frame with signed weights and keep explainable contribution columns."""
    frame = factors.copy()
    frame["multi_factor_score"] = 0.0
    contribution_columns: list[str] = []
    for factor, weight in factor_weights.items():
        if factor not in frame.columns:
            frame[f"{factor}_contribution"] = 0.0
            contribution_columns.append(f"{factor}_contribution")
            continue
        contribution = robust_zscore(frame[factor]) * float(weight)
        contribution = contribution.fillna(0.0)
        column = f"{factor}_contribution"
        frame[column] = contribution
        contribution_columns.append(column)
        frame["multi_factor_score"] = frame["multi_factor_score"] + contribution

    frame["factor_hits"] = (frame[contribution_columns] > 0).sum(axis=1) if contribution_columns else 0
    frame["positive_factors"] = frame.apply(
        lambda row: "|".join(
            factor
            for factor in factor_weights
            if f"{factor}_contribution" in row.index and pd.notna(row[f"{factor}_contribution"]) and row[f"{factor}_contribution"] > 0
        ),
        axis=1,
    )
    frame = frame.sort_values(["multi_factor_score", "factor_hits", "code"], ascending=[False, False, True]).reset_index(drop=True)
    frame["score_rank"] = frame.index + 1
    if top_n is not None:
        frame = frame.head(int(top_n)).copy()
    return frame.reset_index(drop=True)


def derive_validated_multifactor_weights(
    diagnostics: pd.DataFrame,
    base_weights: dict[str, float],
    min_coverage_rate: float = 0.8,
    min_ic_observations: int = 3,
) -> dict[str, float]:
    """Deactivate weights only when diagnostics show insufficient usable data."""
    if diagnostics.empty:
        return dict(base_weights)

    frame = diagnostics.copy()
    frame["factor"] = frame["factor"].astype(str)
    by_factor = frame.drop_duplicates("factor", keep="first").set_index("factor")
    validated: dict[str, float] = {}
    for factor, weight in base_weights.items():
        weight_value = float(weight)
        if factor not in by_factor.index or weight_value == 0:
            validated[factor] = 0.0
            continue

        row = by_factor.loc[factor]
        coverage = _diagnostic_number(row.get("factor_coverage_rate"))
        observations = _diagnostic_number(row.get("ic_observations"))
        if coverage < min_coverage_rate or observations < min_ic_observations:
            validated[factor] = 0.0
            continue

        validated[factor] = weight_value
    return validated


def select_multifactor_candidates(
    scored: pd.DataFrame,
    top_n: int = 20,
    max_per_industry: int | None = 3,
    max_price: float | None = None,
    max_lot_cost: float | None = None,
) -> pd.DataFrame:
    """Apply practical portfolio constraints to a scored multifactor ranking."""
    if scored.empty:
        return scored.copy()

    frame = scored.copy()
    if "close" in frame.columns:
        close = pd.to_numeric(frame["close"], errors="coerce")
        if max_price is not None:
            frame = frame.loc[close <= float(max_price)].copy()
            close = pd.to_numeric(frame["close"], errors="coerce")
        if max_lot_cost is not None:
            frame = frame.loc[(close * 100) <= float(max_lot_cost)].copy()
    if frame.empty:
        return frame.assign(candidate_rank=pd.Series(dtype=int)).reset_index(drop=True)

    if "score_rank" in frame.columns:
        frame = frame.sort_values(["score_rank", "code"], ascending=[True, True])
    else:
        frame = frame.sort_values(["multi_factor_score", "code"], ascending=[False, True])

    selected_rows = []
    industry_counts: dict[str, int] = {}
    for _, row in frame.iterrows():
        industry = str(row.get("industry", ""))
        if max_per_industry is not None and int(max_per_industry) > 0:
            if industry_counts.get(industry, 0) >= int(max_per_industry):
                continue
        selected_rows.append(row)
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
        if len(selected_rows) >= int(top_n):
            break

    if not selected_rows:
        return frame.head(0).assign(candidate_rank=pd.Series(dtype=int)).reset_index(drop=True)
    result = pd.DataFrame(selected_rows).reset_index(drop=True)
    result["candidate_rank"] = result.index + 1
    return result


def generate_multifactor_signals(
    bundle: MarketDataBundle,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    factor_weights: dict[str, float],
    top_n: int = 20,
    signal_every_n_days: int = 5,
    max_per_industry: int | None = 3,
    max_price: float | None = None,
    max_lot_cost: float | None = 25_000,
    universe_config: UniverseConfig | None = None,
    regimes: pd.DataFrame | None = None,
    allowed_regimes: tuple[str, ...] | list[str] | set[str] | None = None,
) -> pd.DataFrame:
    """Generate historical multifactor signal rows for execution backtests."""
    if bundle.prices.empty:
        return _empty_signal_frame()

    price_frame = bundle.prices.copy()
    price_frame["date"] = pd.to_datetime(price_frame["date"])
    start_date = pd.Timestamp(start).normalize()
    end_date = pd.Timestamp(end).normalize()
    trading_dates = (
        price_frame.loc[(price_frame["date"] >= start_date) & (price_frame["date"] <= end_date), "date"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if not trading_dates:
        return _empty_signal_frame()

    step = max(1, int(signal_every_n_days))
    regime_by_date = _regime_by_date(regimes)
    allowed = set(allowed_regimes or [])
    frames: list[pd.DataFrame] = []
    for as_of_date in trading_dates[::step]:
        regime = regime_by_date.get(pd.Timestamp(as_of_date).normalize(), "")
        if allowed and regime not in allowed:
            continue
        factors = build_multifactor_frame(bundle, as_of_date, universe_config=universe_config)
        scored = score_multifactor_frame(factors, factor_weights, top_n=None)
        selected = select_multifactor_candidates(
            scored,
            top_n=top_n,
            max_per_industry=max_per_industry,
            max_price=max_price,
            max_lot_cost=max_lot_cost,
        )
        if selected.empty:
            continue
        selected = selected.copy()
        selected["date"] = as_of_date
        if regime:
            selected["regime"] = regime
        selected["position_scale"] = 1.0
        frames.append(selected)

    if not frames:
        return _empty_signal_frame()
    return pd.concat(frames, ignore_index=True, sort=False)


def generate_multifactor_factor_history(
    bundle: MarketDataBundle,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    signal_every_n_days: int = 5,
    factor_names: list[str] | tuple[str, ...] | None = None,
    universe_config: UniverseConfig | None = None,
) -> pd.DataFrame:
    """Build point-in-time factor snapshots on fixed signal dates for diagnostics."""
    if bundle.prices.empty:
        return _empty_multifactor_frame()

    price_frame = bundle.prices.copy()
    price_frame["date"] = pd.to_datetime(price_frame["date"])
    start_date = pd.Timestamp(start).normalize()
    end_date = pd.Timestamp(end).normalize()
    trading_dates = (
        price_frame.loc[(price_frame["date"] >= start_date) & (price_frame["date"] <= end_date), "date"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if not trading_dates:
        return _empty_multifactor_frame()

    selected_columns = ["date", "code", "name", "industry", "close"]
    selected_columns.extend(list(factor_names or DEFAULT_FACTOR_NAMES))
    frames: list[pd.DataFrame] = []
    for as_of_date in trading_dates[:: max(1, int(signal_every_n_days))]:
        factors = build_multifactor_frame(bundle, as_of_date, universe_config=universe_config)
        if factors.empty:
            continue
        available = [column for column in selected_columns if column in factors.columns]
        frames.append(factors[available].copy())

    if not frames:
        return pd.DataFrame(columns=selected_columns)
    return pd.concat(frames, ignore_index=True, sort=False)


def _price_factor_frame(history: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for code, group in history.groupby("code", sort=True):
        group = group.sort_values("date")
        close = pd.to_numeric(group["close"], errors="coerce")
        volume = pd.to_numeric(group.get("volume", pd.Series(index=group.index, dtype=float)), errors="coerce")
        high = pd.to_numeric(group.get("high", close), errors="coerce")
        low = pd.to_numeric(group.get("low", close), errors="coerce")
        latest_close = close.iloc[-1]
        row: dict[str, float | str] = {"code": str(code)}
        for window in (3, 5, 10, 20, 60):
            row[f"momentum_{window}d"] = _momentum(close, window)
        row["reversal_5d"] = -float(row["momentum_5d"]) if pd.notna(row["momentum_5d"]) else np.nan
        row["volatility_20d"] = float(close.pct_change().tail(20).std(ddof=0))
        row["volume_ratio_20d"] = _latest_to_average(volume, 20)
        row["close_position_20d"] = _close_position(close, 20)
        row["ma5_distance"] = _ma_distance(close, 5, latest_close)
        row["ma20_distance"] = _ma_distance(close, 20, latest_close)
        rows.append(row)
    return pd.DataFrame(rows)


def _latest_money_flow_factors(money_flow: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    columns = ["code", "money_flow_pct", "big_order_pct"]
    if money_flow.empty:
        return pd.DataFrame(columns=columns)
    frame = money_flow.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    latest = frame.loc[frame["date"] <= as_of].sort_values(["code", "date"]).groupby("code", as_index=False).tail(1)
    if latest.empty:
        return pd.DataFrame(columns=columns)
    large = pd.to_numeric(latest.get("large_net_inflow_pct", 0), errors="coerce").fillna(0)
    super_large = pd.to_numeric(latest.get("super_large_net_inflow_pct", 0), errors="coerce").fillna(0)
    return pd.DataFrame(
        {
            "code": latest["code"].astype(str).to_numpy(),
            "money_flow_pct": pd.to_numeric(latest.get("main_net_inflow_pct", np.nan), errors="coerce").to_numpy(),
            "big_order_pct": (large + super_large).to_numpy(),
        }
    )


def _latest_fundamental_factors(fundamentals: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    columns = ["code", "pe_ttm", "pb", "roe", "revenue_yoy", "profit_yoy"]
    if fundamentals.empty:
        return pd.DataFrame(columns=columns)
    frame = fundamentals.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    latest = frame.loc[frame["date"] <= as_of].sort_values(["code", "date"]).groupby("code", as_index=False).tail(1)
    if latest.empty:
        return pd.DataFrame(columns=columns)
    result = latest[["code"]].copy()
    for column in columns[1:]:
        result[column] = pd.to_numeric(latest.get(column, np.nan), errors="coerce")
    return result[columns]


def _add_industry_factors(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "industry" not in result.columns or "momentum_3d" not in result.columns:
        result["industry_momentum_3d"] = np.nan
        return result
    result["industry_momentum_3d"] = result.groupby("industry")["momentum_3d"].transform("mean")
    return result


def _momentum(close: pd.Series, window: int) -> float:
    if len(close) <= window:
        return np.nan
    start = close.iloc[-window - 1]
    end = close.iloc[-1]
    return float(end / start - 1) if pd.notna(start) and start else np.nan


def _latest_to_average(values: pd.Series, window: int) -> float:
    if values.empty:
        return np.nan
    baseline = values.iloc[:-1].tail(window).mean()
    latest = values.iloc[-1]
    return float(latest / baseline) if pd.notna(baseline) and baseline else np.nan


def _close_position(close: pd.Series, window: int) -> float:
    sample = close.tail(window).dropna()
    if sample.empty:
        return np.nan
    low = sample.min()
    high = sample.max()
    return float((sample.iloc[-1] - low) / (high - low)) if high != low else 0.5


def _ma_distance(close: pd.Series, window: int, latest_close: float) -> float:
    ma = close.tail(window).mean()
    return float(latest_close / ma - 1) if pd.notna(ma) and ma else np.nan


def _atr_pct(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> float:
    previous_close = close.shift(1)
    true_range = pd.concat(
        [(high - low).abs(), (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = true_range.tail(window).mean()
    latest_close = close.iloc[-1]
    return float(atr / latest_close) if pd.notna(atr) and latest_close else np.nan


def _numeric_factor_columns(factors: pd.DataFrame) -> list[str]:
    excluded = {"date", "code", "name", "industry"}
    return [
        column
        for column in factors.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(factors[column])
    ]


def _regime_by_date(regimes: pd.DataFrame | None) -> dict[pd.Timestamp, str]:
    if regimes is None or regimes.empty or "date" not in regimes.columns or "regime" not in regimes.columns:
        return {}
    frame = regimes[["date", "regime"]].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["regime"] = frame["regime"].fillna("").astype(str)
    return frame.drop_duplicates("date", keep="last").set_index("date")["regime"].to_dict()


def _diagnostic_number(value: object, default: float = 0.0) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return float(default)
    return float(number)


def _empty_signal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "candidate_rank",
            "score_rank",
            "code",
            "name",
            "industry",
            "close",
            "multi_factor_score",
            "factor_hits",
            "positive_factors",
            "position_scale",
        ]
    )


def _empty_multifactor_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "code", "name", "industry", "close", "amount", *DEFAULT_FACTOR_NAMES])
