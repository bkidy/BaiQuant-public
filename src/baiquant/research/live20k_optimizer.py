from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from baiquant.backtest import MultiPositionTrendConfig, run_multi_position_trend_backtest


@dataclass(frozen=True, slots=True)
class ExecutionVariant:
    name: str
    config: MultiPositionTrendConfig


def evaluate_execution_variants(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    variants: list[ExecutionVariant],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    recent_start: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    price_frame = prices.copy()
    signal_frame = signals.copy()
    if price_frame.empty or signal_frame.empty:
        return pd.DataFrame(columns=_leaderboard_columns())

    price_frame["date"] = pd.to_datetime(price_frame["date"])
    signal_frame["date"] = pd.to_datetime(signal_frame["date"])
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)
    recent_start_date = pd.Timestamp(recent_start) if recent_start is not None else start_date

    signal_frame = signal_frame.loc[
        (signal_frame["date"] >= start_date) & (signal_frame["date"] <= end_date)
    ].copy()
    if signal_frame.empty:
        return pd.DataFrame(columns=_leaderboard_columns())

    signal_codes = set(signal_frame["code"].astype(str))
    if signal_codes and "code" in price_frame.columns:
        price_frame = price_frame.loc[price_frame["code"].astype(str).isin(signal_codes)].copy()

    rows = []
    for variant in variants:
        result = run_multi_position_trend_backtest(price_frame, signal_frame, variant.config)
        daily = result.daily_returns.copy()
        if daily.empty:
            rows.append(_empty_variant_row(variant))
            continue
        daily["date"] = pd.to_datetime(daily["date"])
        full = daily.loc[(daily["date"] >= start_date) & (daily["date"] <= end_date)].copy()
        recent = daily.loc[(daily["date"] >= recent_start_date) & (daily["date"] <= end_date)].copy()
        full_return, full_mdd, full_exposure = _period_stats(full)
        recent_return, recent_mdd, recent_exposure = _period_stats(recent)
        trades = result.trades.copy()
        filled_trades = int(trades["status"].eq("filled").sum()) if not trades.empty else 0
        rows.append(
            {
                "name": variant.name,
                "score": _score_variant(full_return, full_mdd, recent_return, recent_mdd),
                "ytd_return": full_return,
                "ytd_mdd": full_mdd,
                "ytd_exposure": full_exposure,
                "recent_return": recent_return,
                "recent_mdd": recent_mdd,
                "recent_exposure": recent_exposure,
                "filled_trades": filled_trades,
                "max_positions": variant.config.max_positions,
                "ma_window": variant.config.ma_window,
                "stop_loss_pct": variant.config.stop_loss_pct,
                "take_profit_pct": variant.config.take_profit_pct,
                "portfolio_stop_drawdown_pct": variant.config.portfolio_stop_drawdown_pct,
                "trailing_stop_activation_pct": variant.config.trailing_stop_activation_pct,
                "trailing_stop_pct": variant.config.trailing_stop_pct,
            }
        )
    leaderboard = pd.DataFrame(rows, columns=_leaderboard_columns())
    return leaderboard.sort_values(
        ["score", "recent_return", "ytd_return"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def default_live20k_execution_variants(base: MultiPositionTrendConfig) -> list[ExecutionVariant]:
    take30_base = _replace_config(base, take_profit_pct=0.30)
    return [
        ExecutionVariant("ma10_take30_trail12_8_pos2", take30_base),
        ExecutionVariant("ma10_take25_trail12_8_pos2", _replace_config(take30_base, take_profit_pct=0.25)),
        ExecutionVariant("ma10_take35_trail12_8_pos2", _replace_config(take30_base, take_profit_pct=0.35)),
        ExecutionVariant("ma10_no_take_trail12_8_pos2", _replace_config(base, take_profit_pct=0.0)),
        ExecutionVariant("ma10_take30_no_trail_pos2", _replace_config(take30_base, trailing_stop_activation_pct=0.0, trailing_stop_pct=0.0)),
        ExecutionVariant("ma8_take30_no_trail_pos2", _replace_config(take30_base, ma_window=8, trailing_stop_activation_pct=0.0, trailing_stop_pct=0.0)),
        ExecutionVariant("ma20_take30_no_trail_pos2", _replace_config(take30_base, ma_window=20, trailing_stop_activation_pct=0.0, trailing_stop_pct=0.0)),
        ExecutionVariant("ma10_take30_trail12_8_pos3", _replace_config(take30_base, max_positions=3)),
        ExecutionVariant("ma8_take30_no_trail_pos3", _replace_config(take30_base, max_positions=3, ma_window=8, trailing_stop_activation_pct=0.0, trailing_stop_pct=0.0)),
        ExecutionVariant("ma8_no_take_no_trail_pos3", _replace_config(base, max_positions=3, ma_window=8, take_profit_pct=0.0, trailing_stop_activation_pct=0.0, trailing_stop_pct=0.0)),
        ExecutionVariant("ma20_take30_no_trail_pos3", _replace_config(take30_base, max_positions=3, ma_window=20, trailing_stop_activation_pct=0.0, trailing_stop_pct=0.0)),
        ExecutionVariant("ma10_take30_no_portfolio_stop", _replace_config(take30_base, portfolio_stop_drawdown_pct=0.0)),
        ExecutionVariant("ma10_take30_wide_portfolio_stop", _replace_config(take30_base, portfolio_stop_drawdown_pct=0.06)),
    ]


def _replace_config(config: MultiPositionTrendConfig, **updates: object) -> MultiPositionTrendConfig:
    values = {field: getattr(config, field) for field in config.__dataclass_fields__}
    values.update(updates)
    return MultiPositionTrendConfig(**values)


def _period_stats(frame: pd.DataFrame) -> tuple[float, float, float]:
    if frame.empty:
        return 0.0, 0.0, 0.0
    equity = pd.to_numeric(frame["equity"], errors="coerce")
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1) if float(equity.iloc[0]) else 0.0
    max_drawdown = float((equity / equity.cummax() - 1).min())
    exposure = float((pd.to_numeric(frame.get("position_value", 0), errors="coerce").fillna(0) > 0).mean())
    return total_return, max_drawdown, exposure


def _score_variant(
    full_return: float,
    full_mdd: float,
    recent_return: float,
    recent_mdd: float,
) -> float:
    return full_return + 1.5 * recent_return + 0.3 * full_mdd + 0.5 * recent_mdd


def _empty_variant_row(variant: ExecutionVariant) -> dict[str, object]:
    return {
        "name": variant.name,
        "score": 0.0,
        "ytd_return": 0.0,
        "ytd_mdd": 0.0,
        "ytd_exposure": 0.0,
        "recent_return": 0.0,
        "recent_mdd": 0.0,
        "recent_exposure": 0.0,
        "filled_trades": 0,
        "max_positions": variant.config.max_positions,
        "ma_window": variant.config.ma_window,
        "stop_loss_pct": variant.config.stop_loss_pct,
        "take_profit_pct": variant.config.take_profit_pct,
        "portfolio_stop_drawdown_pct": variant.config.portfolio_stop_drawdown_pct,
        "trailing_stop_activation_pct": variant.config.trailing_stop_activation_pct,
        "trailing_stop_pct": variant.config.trailing_stop_pct,
    }


def _leaderboard_columns() -> list[str]:
    return [
        "name",
        "score",
        "ytd_return",
        "ytd_mdd",
        "ytd_exposure",
        "recent_return",
        "recent_mdd",
        "recent_exposure",
        "filled_trades",
        "max_positions",
        "ma_window",
        "stop_loss_pct",
        "take_profit_pct",
        "portfolio_stop_drawdown_pct",
        "trailing_stop_activation_pct",
        "trailing_stop_pct",
    ]
