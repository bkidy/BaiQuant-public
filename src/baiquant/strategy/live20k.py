from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

import numpy as np
import pandas as pd

from baiquant.backtest import MultiPositionTrendConfig
from baiquant.data.bundle import MarketDataBundle
from baiquant.scoring import FactorSpec, robust_zscore


@dataclass(slots=True)
class Live20KSignalConfig:
    min_listed_days: int = 180
    min_history_days: int = 60
    min_amount: float = 30_000_000
    min_price: float = 3.0
    max_price: float = 120.0
    exclude_bj: bool = True
    exclude_star: bool = False
    exclude_chinext: bool = False
    min_factor_hits: int = 3
    rank_start: int = 6
    signal_limit: int = 5
    market_breadth_min: float = 0.60
    market_dist_ma60_max: float = 0.18
    apply_market_gate: bool = True
    market_breadth_floor: float | None = None
    industry_allowlist: tuple[str, ...] = ()
    dynamic_hotspot: bool = False
    hotspot_top_n: int = 3
    hotspot_window: int = 20
    hotspot_min_stock_count: int = 20
    hotspot_use_money_flow: bool = False
    hotspot_prefer_early_strength: bool = False
    hotspot_exclude_retreat: bool = False
    hotspot_retreat_momentum_3d: float = -0.03
    hotspot_retreat_breadth_delta_5d: float = -0.10
    min_money_flow_3d: float | None = None
    min_big_order_3d: float | None = None
    min_close_position_20d: float | None = None
    max_close_position_20d: float | None = None
    min_momentum_20d: float | None = None
    max_momentum_20d: float | None = None
    min_momentum_5d: float | None = None
    max_momentum_5d: float | None = None
    min_amount_ratio_5d: float | None = None
    min_trend_pullback: float | None = None
    max_close_vs_20d_high: float | None = None
    factor_specs: list[FactorSpec] = field(
        default_factory=lambda: [
            FactorSpec("trend_pullback", 1.2, 1),
            FactorSpec("volatility_20d", 1.2, -1),
            FactorSpec("week_52_high", 0.6, 1),
            FactorSpec("momentum_20d", 0.4, 1),
            FactorSpec("volume_score", 0.4, 1),
        ]
    )


@dataclass(slots=True)
class Live20KPaperFillResult:
    holdings: pd.DataFrame
    fills: pd.DataFrame
    cash: float
    equity: float
    equity_peak: float


@dataclass(slots=True)
class Live20KPaperStepResult:
    plan: pd.DataFrame
    fills: pd.DataFrame
    holdings: pd.DataFrame
    state: pd.DataFrame
    report: pd.DataFrame
    filled_previous_plan: bool


@dataclass(slots=True)
class Live20KPaperReplayResult:
    plans: pd.DataFrame
    fills: pd.DataFrame
    holdings: pd.DataFrame
    states: pd.DataFrame
    report: pd.DataFrame


def live20k_execution_config() -> MultiPositionTrendConfig:
    return MultiPositionTrendConfig(
        initial_cash=20_000,
        max_positions=2,
        fee_bps=10,
        slippage_bps=5,
        lot_size=100,
        stop_loss_pct=0.06,
        take_profit_pct=0.30,
        ma_window=10,
        trailing_stop_activation_pct=0.12,
        trailing_stop_pct=0.08,
        portfolio_stop_drawdown_pct=0.04,
        portfolio_stop_cooldown_days=5,
        liquidate_on_portfolio_stop=True,
        reset_peak_on_portfolio_stop=True,
        reset_peak_min_profit_pct=0.10,
    )


def live100k_hotspot_turbo_execution_config() -> MultiPositionTrendConfig:
    return MultiPositionTrendConfig(
        initial_cash=100_000,
        max_positions=3,
        cash_buffer_pct=0.1,
        fee_bps=10,
        slippage_bps=5,
        lot_size=100,
        stop_loss_pct=0.06,
        take_profit_pct=0.0,
        add_trigger_pct=0.03,
        add_position_multiple=1.0,
        ma_window=0,
        max_holding_days=15,
        trailing_stop_activation_pct=0.05,
        trailing_stop_pct=0.05,
        portfolio_stop_drawdown_pct=0.10,
        portfolio_stop_cooldown_days=10,
        liquidate_on_portfolio_stop=True,
        reset_peak_on_portfolio_stop=True,
        reset_peak_min_profit_pct=0.0,
    )


def live100k_hotspot_manual_fixed_execution_config() -> MultiPositionTrendConfig:
    return replace(
        live100k_hotspot_turbo_execution_config(),
        initial_cash=50_000,
        add_trigger_pct=0.0,
        add_position_multiple=0.0,
        max_holding_days=20,
        trailing_stop_activation_pct=0.0,
        trailing_stop_pct=0.0,
    )


def live100k_hotspot_turbo_signal_config() -> Live20KSignalConfig:
    return Live20KSignalConfig(
        min_amount=80_000_000,
        max_price=120.0,
        min_factor_hits=4,
        rank_start=1,
        signal_limit=4,
        exclude_star=False,
        exclude_chinext=False,
        apply_market_gate=False,
        market_breadth_floor=0.25,
        dynamic_hotspot=True,
        hotspot_top_n=8,
        hotspot_min_stock_count=10,
        hotspot_use_money_flow=True,
        hotspot_prefer_early_strength=True,
        hotspot_exclude_retreat=True,
        min_money_flow_3d=2,
        min_big_order_3d=2,
        min_close_position_20d=0.80,
        min_momentum_20d=0.15,
        min_momentum_5d=0.10,
        min_amount_ratio_5d=1.0,
        factor_specs=[
            FactorSpec("hot_score", 1.4, 1),
            FactorSpec("momentum_5d", 1.4, 1),
            FactorSpec("money_flow_3d", 1.1, 1),
            FactorSpec("big_order_3d", 1.0, 1),
            FactorSpec("momentum_20d", 0.9, 1),
            FactorSpec("amount_ratio_5d", 0.9, 1),
            FactorSpec("week_52_high", 0.7, 1),
            FactorSpec("close_position_20d", 0.6, 1),
            FactorSpec("volatility_20d", 0.3, -1),
        ],
    )


def live100k_hotspot_manual_fixed_signal_config() -> Live20KSignalConfig:
    return live100k_hotspot_turbo_signal_config()


def slice_market_data_for_signal(
    bundle: MarketDataBundle,
    as_of: str | pd.Timestamp,
    lookback_days: int = 540,
) -> MarketDataBundle:
    as_of_date = pd.Timestamp(as_of)
    start_date = as_of_date - pd.Timedelta(days=lookback_days)
    return MarketDataBundle(
        prices=_slice_dated_frame(bundle.prices, "date", start_date, as_of_date),
        fundamentals=_slice_dated_frame(bundle.fundamentals, "date", start_date, as_of_date),
        stocks=bundle.stocks.copy(),
        events=_slice_dated_frame(bundle.events, "date", start_date, as_of_date),
        money_flow=_slice_dated_frame(bundle.money_flow, "date", start_date, as_of_date),
    )


def _slice_dated_frame(
    frame: pd.DataFrame,
    column: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return frame.copy()
    sliced = frame.copy()
    dates = pd.to_datetime(sliced[column])
    mask = (dates >= start_date) & (dates <= end_date)
    return sliced.loc[mask].reset_index(drop=True)


def assert_live20k_data_fresh(bundle: MarketDataBundle, as_of: str | pd.Timestamp) -> None:
    if bundle.prices.empty or "date" not in bundle.prices.columns:
        raise ValueError("stale market data: no prices available")
    as_of_date = pd.Timestamp(as_of).normalize()
    latest_price_date = pd.to_datetime(bundle.prices["date"]).max().normalize()
    if as_of_date > latest_price_date:
        raise ValueError(
            "stale market data: "
            f"requested_as_of={as_of_date.date()} "
            f"latest_price_date={latest_price_date.date()}"
        )


def build_market_regime(prices: pd.DataFrame) -> pd.DataFrame:
    price_frame = prices.copy()
    price_frame["date"] = pd.to_datetime(price_frame["date"])
    close = price_frame.pivot(index="date", columns="code", values="close").sort_index()
    market_ret = close.pct_change().mean(axis=1, skipna=True).fillna(0.0)
    market_equity = (1 + market_ret).cumprod()
    market_ma20 = market_equity.rolling(20, min_periods=20).mean()
    market_ma60 = market_equity.rolling(60, min_periods=60).mean()
    ma20 = close.rolling(20, min_periods=20).mean()
    breadth_ma20 = (close > ma20).sum(axis=1) / close.notna().sum(axis=1)
    return pd.DataFrame(
        {
            "date": close.index,
            "breadth_ma20": breadth_ma20.to_numpy(),
            "market_ret": market_ret.to_numpy(),
            "market_equity": market_equity.to_numpy(),
            "market_ma20": market_ma20.to_numpy(),
            "market_ma60": market_ma60.to_numpy(),
        }
    ).reset_index(drop=True)


def build_hot_industries(
    bundle: MarketDataBundle,
    window: int = 20,
    top_n: int = 3,
    min_stock_count: int = 20,
    use_money_flow: bool = False,
    prefer_early_strength: bool = False,
    exclude_retreat: bool = False,
    retreat_momentum_3d: float = -0.03,
    retreat_breadth_delta_5d: float = -0.10,
) -> pd.DataFrame:
    prices = bundle.prices.copy()
    if prices.empty or bundle.stocks.empty:
        return _empty_hot_industry_frame()
    prices["date"] = pd.to_datetime(prices["date"])
    prices["code"] = prices["code"].astype(str)
    stocks = bundle.stocks[["code", "industry"]].copy()
    stocks["code"] = stocks["code"].astype(str)
    stocks["industry"] = stocks["industry"].fillna("").astype(str)
    frame = prices.merge(stocks, on="code", how="left")
    if use_money_flow:
        money_flow_features = _money_flow_feature_frame(bundle.money_flow)
        if not money_flow_features.empty:
            frame = frame.merge(money_flow_features, on=["date", "code"], how="left")
    for column in ["money_flow_3d", "big_order_3d", "main_inflow_3d"]:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["industry"] = frame["industry"].fillna("").astype(str)
    frame = frame.loc[frame["industry"] != ""].sort_values(["code", "date"]).reset_index(drop=True)
    if frame.empty:
        return _empty_hot_industry_frame()

    grouped = frame.groupby("code", sort=False)
    close = grouped["close"]
    volume = grouped["volume"]
    frame["momentum"] = close.pct_change(window)
    frame["momentum_3d"] = close.pct_change(3)
    frame["momentum_5d"] = close.pct_change(5)
    ma = close.transform(lambda series: series.rolling(window, min_periods=window).mean())
    frame["above_ma"] = frame["close"] > ma
    frame["volume_ratio"] = frame["volume"] / volume.transform(
        lambda series: series.shift(1).rolling(window, min_periods=window).mean()
    )
    frame["limit_up_flag"] = frame.get("limit_up", 0)

    industry = (
        frame.groupby(["date", "industry"], sort=True)
        .agg(
            industry_momentum_20d=("momentum", "mean"),
            industry_momentum_3d=("momentum_3d", "mean"),
            industry_momentum_5d=("momentum_5d", "mean"),
            industry_breadth_ma20=("above_ma", "mean"),
            industry_volume_ratio=("volume_ratio", "median"),
            industry_limit_up_rate=("limit_up_flag", "mean"),
            industry_money_flow_3d=("money_flow_3d", "median"),
            industry_big_order_3d=("big_order_3d", "median"),
            industry_main_inflow_3d=("main_inflow_3d", "sum"),
            stock_count=("code", "nunique"),
        )
        .reset_index()
    )
    industry = industry.dropna(subset=["industry_momentum_20d", "industry_breadth_ma20"])
    industry = industry.loc[industry["stock_count"] >= min_stock_count].copy()
    if industry.empty:
        return _empty_hot_industry_frame()
    industry = industry.sort_values(["industry", "date"]).reset_index(drop=True)
    industry["industry_breadth_delta_5d"] = industry.groupby("industry", sort=False)["industry_breadth_ma20"].diff(5)
    industry["industry_retreat"] = (
        (industry["industry_momentum_3d"].fillna(0.0) <= retreat_momentum_3d)
        & (industry["industry_breadth_delta_5d"].fillna(0.0) <= retreat_breadth_delta_5d)
    )
    industry["industry_volume_ratio"] = pd.to_numeric(industry["industry_volume_ratio"], errors="coerce").fillna(1.0)
    for column in ["industry_money_flow_3d", "industry_big_order_3d", "industry_main_inflow_3d"]:
        industry[column] = pd.to_numeric(industry[column], errors="coerce").fillna(0.0)
    industry = score_hot_industry_frame(
        industry,
        use_money_flow=use_money_flow,
        prefer_early_strength=prefer_early_strength,
    )
    if exclude_retreat:
        industry = industry.loc[~industry["industry_retreat"]].copy()
        if industry.empty:
            return _empty_hot_industry_frame()
    industry = industry.sort_values(["date", "hot_score", "industry"], ascending=[True, False, True])
    industry["hot_rank"] = industry.groupby("date").cumcount() + 1
    industry = industry.loc[industry["hot_rank"] <= top_n].copy()
    return industry[_hot_industry_columns()].reset_index(drop=True)


def score_hot_industry_frame(
    industry: pd.DataFrame,
    *,
    use_money_flow: bool,
    prefer_early_strength: bool,
) -> pd.DataFrame:
    scored = industry.copy()
    zscore = _hot_industry_zscore(scored)
    if prefer_early_strength:
        scored["hot_score"] = (
            0.50 * zscore("industry_momentum_20d")
            + 0.90 * zscore("industry_momentum_3d")
            + 0.60 * zscore("industry_momentum_5d")
            + 0.60 * zscore("industry_breadth_ma20")
            + 0.80 * zscore("industry_breadth_delta_5d")
            + 0.80 * zscore("industry_limit_up_rate")
            + 0.20 * zscore("industry_volume_ratio")
        )
        if use_money_flow:
            scored["hot_score"] += (
                0.70 * zscore("industry_money_flow_3d")
                + 0.50 * zscore("industry_big_order_3d")
            )
        return scored

    scored["hot_score"] = (
        zscore("industry_momentum_20d")
        + 0.6 * zscore("industry_breadth_ma20")
        + 0.4 * zscore("industry_volume_ratio")
        + 0.8 * zscore("industry_limit_up_rate")
    )
    if use_money_flow:
        scored["hot_score"] = (
            zscore("industry_momentum_20d")
            + 0.7 * zscore("industry_breadth_ma20")
            + 0.7 * zscore("industry_money_flow_3d")
            + 0.5 * zscore("industry_big_order_3d")
            + 0.8 * zscore("industry_limit_up_rate")
            + 0.2 * zscore("industry_volume_ratio")
        )
    return scored


def _hot_industry_zscore(industry: pd.DataFrame):
    grouped = industry.groupby("date", group_keys=False) if "date" in industry.columns else None

    def zscore(column: str) -> pd.Series:
        if column not in industry.columns:
            return pd.Series(0.0, index=industry.index)
        values = pd.to_numeric(industry[column], errors="coerce").fillna(0.0)
        if grouped is None:
            return robust_zscore(values).reindex(industry.index).fillna(0.0)
        return grouped[column].apply(robust_zscore).reindex(industry.index).fillna(0.0)

    return zscore


def build_live20k_daily_plan(
    bundle: MarketDataBundle,
    as_of: str | pd.Timestamp,
    holdings: pd.DataFrame | None = None,
    cash: float | None = None,
    equity_peak: float | None = None,
    signal_config: Live20KSignalConfig | None = None,
    execution_config: MultiPositionTrendConfig | None = None,
) -> pd.DataFrame:
    signal_config = signal_config or Live20KSignalConfig()
    execution_config = execution_config or live20k_execution_config()
    as_of_date = pd.Timestamp(as_of)
    assert_live20k_data_fresh(bundle, as_of_date)
    cash_value = float(execution_config.initial_cash if cash is None else cash)
    holding_frame = _normalize_holdings(holdings)

    prices = bundle.prices.copy()
    if prices.empty:
        return _empty_plan_frame()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.loc[prices["date"] <= as_of_date].sort_values(["date", "code"]).reset_index(drop=True)
    if prices.empty:
        return _empty_plan_frame()

    plan_date = prices["date"].max()
    trading_dates = list(prices["date"].drop_duplicates().sort_values())
    latest = prices.loc[prices["date"] == plan_date].set_index("code")
    ma = _latest_moving_average(prices, execution_config.ma_window, plan_date)
    regime_row = _latest_regime_row(prices, plan_date, signal_config)
    market_gate = bool(regime_row["market_gate"])

    equity = cash_value + _holdings_market_value(holding_frame, latest)
    peak = float(equity if equity_peak is None else equity_peak)
    drawdown = equity / peak - 1 if peak else 0.0
    portfolio_stop = (
        execution_config.portfolio_stop_drawdown_pct > 0
        and drawdown <= -execution_config.portfolio_stop_drawdown_pct
    )

    rows: list[dict[str, object]] = []
    rows.extend(
        _build_exit_plan_rows(
            holding_frame=holding_frame,
            latest=latest,
            ma=ma,
            plan_date=plan_date,
            trading_dates=trading_dates,
            market_gate=market_gate,
            regime_row=regime_row,
            equity=equity,
            drawdown=drawdown,
            execution_config=execution_config,
            portfolio_stop=portfolio_stop,
        )
    )

    exiting_codes = {str(row["code"]) for row in rows if row["action"] == "sell_next_open"}
    held_codes = set(holding_frame["code"].astype(str)) - exiting_codes
    entry_gate = market_gate or not signal_config.apply_market_gate
    if entry_gate and not portfolio_stop:
        add_rows = _build_add_plan_rows(
            holding_frame=holding_frame,
            latest=latest,
            plan_date=plan_date,
            exiting_codes=exiting_codes,
            cash_value=cash_value + _planned_sell_value(rows),
            market_gate=market_gate,
            regime_row=regime_row,
            equity=equity,
            drawdown=drawdown,
            execution_config=execution_config,
        )
        rows.extend(add_rows)
    if entry_gate and not portfolio_stop:
        entry_cash = cash_value + _planned_sell_value(rows) - _planned_buy_value(rows)
        rows.extend(
            _build_entry_plan_rows(
                bundle=bundle,
                plan_date=plan_date,
                held_codes=held_codes,
                latest=latest,
                cash_value=entry_cash,
                signal_config=signal_config,
                execution_config=execution_config,
                regime_row=regime_row,
                equity=equity,
                drawdown=drawdown,
            )
        )

    if not rows:
        wait_reason = "portfolio_stop"
        if not portfolio_stop:
            wait_reason = "market_gate_off" if not entry_gate else "no_signal"
        rows.append(
            _plan_row(
                date=plan_date,
                action="wait",
                code="",
                name="",
                reason=wait_reason,
                shares=0,
                reference_price=np.nan,
                score_rank=np.nan,
                cash_budget=0.0,
                market_gate=market_gate,
                regime_row=regime_row,
                equity=equity,
                drawdown=drawdown,
            )
        )
    return pd.DataFrame(rows, columns=_plan_columns())


def apply_live20k_paper_fills(
    bundle: MarketDataBundle,
    plan: pd.DataFrame,
    holdings: pd.DataFrame | None = None,
    cash: float | None = None,
    equity_peak: float | None = None,
    execution_date: str | pd.Timestamp | None = None,
    execution_config: MultiPositionTrendConfig | None = None,
) -> Live20KPaperFillResult:
    execution_config = execution_config or live20k_execution_config()
    cash_value = float(execution_config.initial_cash if cash is None else cash)
    holding_frame = _normalize_holdings(holdings)
    positions = {str(row["code"]): int(row["shares"]) for _, row in holding_frame.iterrows()}
    average_costs = {
        str(row["code"]): float(row["average_cost"])
        for _, row in holding_frame.iterrows()
        if pd.notna(row["average_cost"])
    }
    entry_shares = {
        str(row["code"]): int(row["entry_shares"])
        for _, row in holding_frame.iterrows()
        if pd.notna(row["entry_shares"]) and int(row["entry_shares"]) > 0
    }
    entry_dates = {
        str(row["code"]): pd.Timestamp(row["entry_date"])
        for _, row in holding_frame.iterrows()
        if "entry_date" in row.index and pd.notna(row["entry_date"])
    }
    added_codes = {str(row["code"]) for _, row in holding_frame.iterrows() if bool(row["added"])}
    high_closes = {
        str(row["code"]): float(row["high_close"])
        for _, row in holding_frame.iterrows()
        if "high_close" in row.index and pd.notna(row["high_close"])
    }

    price_frame = bundle.prices.copy()
    if price_frame.empty:
        return Live20KPaperFillResult(holding_frame, _empty_fill_frame(), cash_value, cash_value, cash_value)
    price_frame["date"] = pd.to_datetime(price_frame["date"])
    price_frame = price_frame.sort_values(["date", "code"]).reset_index(drop=True)
    trading_dates = list(price_frame["date"].drop_duplicates().sort_values())
    execution_ts = _resolve_execution_date(plan, trading_dates, execution_date)
    market = price_frame.loc[price_frame["date"] == execution_ts].set_index("code")

    fee_rate = execution_config.fee_bps / 10_000
    slippage_rate = execution_config.slippage_bps / 10_000
    fill_rows: list[dict[str, object]] = []
    order_frame = plan.loc[plan.get("action", pd.Series(dtype=object)).isin(["sell_next_open", "buy_next_open"])].copy()
    order_frame["_order"] = range(len(order_frame))
    sells = order_frame.loc[order_frame["action"] == "sell_next_open"].sort_values("_order")
    buys = order_frame.loc[order_frame["action"] == "buy_next_open"].sort_values("_order")

    for _, order in sells.iterrows():
        code = str(order["code"])
        requested = min(int(order.get("shares", 0) or 0), int(positions.get(code, 0)))
        cash_value, fill = _paper_sell(
            date=execution_ts,
            code=code,
            requested=requested,
            reason=str(order.get("reason", "")),
            market=market,
            positions=positions,
            cash=cash_value,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
        )
        if code not in positions:
            average_costs.pop(code, None)
            high_closes.pop(code, None)
            entry_shares.pop(code, None)
            entry_dates.pop(code, None)
            added_codes.discard(code)
        fill_rows.append(fill)

    for _, order in buys.iterrows():
        code = str(order["code"])
        requested = int(order.get("shares", 0) or 0)
        reason = str(order.get("reason", ""))
        had_position = code in positions
        cash_value, fill, average_cost = _paper_buy(
            date=execution_ts,
            code=code,
            requested=requested,
            reason=reason,
            market=market,
            positions=positions,
            cash=cash_value,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            lot_size=execution_config.lot_size,
            previous_average_cost=average_costs.get(code, 0.0),
        )
        if int(fill["filled_shares"]) > 0:
            average_costs[code] = average_cost
            high_closes[code] = max(high_closes.get(code, average_cost), average_cost)
            if reason == "profit_add":
                added_codes.add(code)
                entry_shares.setdefault(code, max(int(positions.get(code, 0)) - int(fill["filled_shares"]), 0))
            elif not had_position:
                entry_shares[code] = int(fill["filled_shares"])
                entry_dates[code] = execution_ts
        fill_rows.append(fill)

    high_closes = _update_high_closes_from_market(positions, high_closes, market)
    updated_holdings = _positions_to_holdings(positions, average_costs, high_closes, entry_shares, added_codes, entry_dates)
    equity = cash_value + _holdings_market_value(updated_holdings, market)
    fills = pd.DataFrame(fill_rows, columns=_fill_columns())
    peak = _paper_equity_peak_after_fills(equity, equity_peak, fills, updated_holdings, execution_config)
    return Live20KPaperFillResult(
        holdings=updated_holdings,
        fills=fills,
        cash=round(cash_value, 6),
        equity=round(equity, 6),
        equity_peak=round(peak, 6),
    )


def summarize_live20k_paper_run(
    plans: pd.DataFrame,
    fills: pd.DataFrame | None = None,
    states: pd.DataFrame | None = None,
    min_days: int = 20,
    max_allowed_drawdown: float | None = None,
    min_order_days: int = 0,
    min_total_return: float | None = None,
) -> pd.DataFrame:
    execution_config = live20k_execution_config()
    drawdown_limit = (
        execution_config.portfolio_stop_drawdown_pct
        if max_allowed_drawdown is None
        else max_allowed_drawdown
    )
    plan_frame = plans.copy()
    fill_frame = fills.copy() if fills is not None else _empty_fill_frame()
    state_frame = states.copy() if states is not None else pd.DataFrame()

    if not plan_frame.empty and "date" in plan_frame.columns:
        plan_frame["date"] = pd.to_datetime(plan_frame["date"])
    if not fill_frame.empty and "date" in fill_frame.columns:
        fill_frame["date"] = pd.to_datetime(fill_frame["date"])
    if not state_frame.empty and "date" in state_frame.columns:
        state_frame["date"] = pd.to_datetime(state_frame["date"])

    paper_days = int(plan_frame["date"].nunique()) if "date" in plan_frame.columns else 0
    order_days = _count_order_days(plan_frame)
    planned_orders = _count_orders(plan_frame)
    filled_orders = _count_filled_orders(fill_frame)
    failed_fills = _count_failed_fills(fill_frame)
    blocking_failed_fills = _count_blocking_failed_fills(fill_frame)
    rule_violations = _count_rule_violations(plan_frame)
    start_equity, latest_equity, max_drawdown = _paper_state_metrics(state_frame, plan_frame)

    blocking: list[str] = []
    if paper_days < min_days:
        blocking.append(f"paper_days<{min_days}")
    if order_days < min_order_days:
        blocking.append(f"order_days<{min_order_days}")
    if blocking_failed_fills > 0:
        blocking.append("failed_fills")
    if rule_violations > 0:
        blocking.append("rule_violations")
    if max_drawdown < -drawdown_limit:
        blocking.append("drawdown")

    total_return = latest_equity / start_equity - 1 if start_equity else 0.0
    if min_total_return is not None and total_return < min_total_return:
        blocking.append(f"total_return<{_pct(min_total_return)}")

    ready = len(blocking) == 0
    return pd.DataFrame(
        [
            {
                "paper_days": paper_days,
                "order_days": order_days,
                "planned_orders": planned_orders,
                "filled_orders": filled_orders,
                "failed_fills": failed_fills,
                "blocking_failed_fills": blocking_failed_fills,
                "rule_violations": rule_violations,
                "start_equity": start_equity,
                "latest_equity": latest_equity,
                "total_return": round(total_return, 6),
                "max_drawdown": round(max_drawdown, 6),
                "ready_for_live": bool(ready),
                "blocking_reason": ",".join(blocking),
            }
        ]
    ).astype({"ready_for_live": object})


def export_live20k_orders(plan: pd.DataFrame, report: pd.DataFrame) -> pd.DataFrame:
    _require_ready_report(report)
    if plan.empty or "action" not in plan.columns:
        return _empty_order_frame()
    orders = plan.loc[plan["action"].isin(["buy_next_open", "sell_next_open"])].copy()
    if orders.empty:
        return _empty_order_frame()
    order_rows = []
    for _, order in orders.iterrows():
        action = str(order["action"])
        order_rows.append(
            {
                "plan_date": pd.Timestamp(order["date"]) if "date" in order.index else pd.NaT,
                "side": "buy" if action == "buy_next_open" else "sell",
                "code": str(order.get("code", "")),
                "name": str(order.get("name", "")),
                "shares": int(order.get("shares", 0) or 0),
                "reference_price": pd.to_numeric(order.get("reference_price", np.nan), errors="coerce"),
                "reason": str(order.get("reason", "")),
                "score_rank": pd.to_numeric(order.get("score_rank", np.nan), errors="coerce"),
                "source_action": action,
            }
        )
    return pd.DataFrame(order_rows, columns=_order_columns())


def run_live20k_paper_step(
    bundle: MarketDataBundle,
    as_of: str | pd.Timestamp,
    holdings: pd.DataFrame | None = None,
    state: pd.DataFrame | None = None,
    previous_plan: pd.DataFrame | None = None,
    existing_plans: pd.DataFrame | None = None,
    existing_fills: pd.DataFrame | None = None,
    existing_states: pd.DataFrame | None = None,
    min_days: int = 20,
    min_order_days: int = 0,
    min_total_return: float | None = None,
    signal_config: Live20KSignalConfig | None = None,
    execution_config: MultiPositionTrendConfig | None = None,
) -> Live20KPaperStepResult:
    execution_config = execution_config or live20k_execution_config()
    as_of_date = pd.Timestamp(as_of)
    current_holdings = _normalize_holdings(holdings)
    cash, equity_peak = _latest_cash_and_peak(state, execution_config)
    filled_previous_plan = False
    fills = _empty_fill_frame()

    if previous_plan is not None and not previous_plan.empty:
        fill_result = apply_live20k_paper_fills(
            bundle,
            previous_plan,
            holdings=current_holdings,
            cash=cash,
            equity_peak=equity_peak,
            execution_date=as_of_date,
            execution_config=execution_config,
        )
        current_holdings = fill_result.holdings
        cash = fill_result.cash
        equity_peak = fill_result.equity_peak
        fills = fill_result.fills
        filled_previous_plan = True

    plan = build_live20k_daily_plan(
        bundle,
        as_of=as_of_date,
        holdings=current_holdings,
        cash=cash,
        equity_peak=equity_peak,
        signal_config=signal_config,
        execution_config=execution_config,
    )
    current_holdings = _mark_holdings_to_latest_close(bundle.prices, current_holdings, as_of_date)
    equity = _plan_equity(plan, cash)
    equity_peak = max(equity_peak, equity)
    state_row = pd.DataFrame(
        [
            {
                "date": as_of_date,
                "cash": cash,
                "equity": equity,
                "equity_peak": equity_peak,
            }
        ]
    )

    report_plans = _concat_frames(existing_plans, plan)
    report_fills = _concat_frames(existing_fills, fills)
    report_states = _concat_frames(existing_states, state_row)
    report = summarize_live20k_paper_run(
        report_plans,
        report_fills,
        report_states,
        min_days=min_days,
        min_order_days=min_order_days,
        min_total_return=min_total_return,
    )
    return Live20KPaperStepResult(
        plan=plan,
        fills=fills,
        holdings=current_holdings,
        state=state_row,
        report=report,
        filled_previous_plan=filled_previous_plan,
    )


def run_live20k_paper_replay(
    bundle: MarketDataBundle,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    min_days: int = 20,
    min_order_days: int = 0,
    min_total_return: float | None = None,
    signal_config: Live20KSignalConfig | None = None,
    execution_config: MultiPositionTrendConfig | None = None,
) -> Live20KPaperReplayResult:
    trading_dates = _trading_dates_in_range(bundle.prices, start, end)
    holdings = _normalize_holdings(None)
    state = pd.DataFrame()
    previous_plan = None
    plans = pd.DataFrame()
    fills = pd.DataFrame()
    states = pd.DataFrame()
    report = summarize_live20k_paper_run(
        plans,
        fills,
        states,
        min_days=min_days,
        min_order_days=min_order_days,
        min_total_return=min_total_return,
    )

    for trade_date in trading_dates:
        step = run_live20k_paper_step(
            bundle,
            as_of=trade_date,
            holdings=holdings,
            state=state,
            previous_plan=previous_plan,
            existing_plans=plans,
            existing_fills=fills,
            existing_states=states,
            min_days=min_days,
            min_order_days=min_order_days,
            min_total_return=min_total_return,
            signal_config=signal_config,
            execution_config=execution_config,
        )
        holdings = step.holdings
        state = step.state
        previous_plan = step.plan
        plans = _concat_frames(plans, step.plan)
        fills = _concat_frames(fills, step.fills)
        states = _concat_frames(states, step.state)
        report = step.report

    return Live20KPaperReplayResult(
        plans=plans,
        fills=fills if not fills.empty else _empty_fill_frame(),
        holdings=holdings,
        states=states,
        report=report,
    )


def generate_live20k_signals(
    bundle: MarketDataBundle,
    config: Live20KSignalConfig | None = None,
) -> pd.DataFrame:
    config = config or Live20KSignalConfig()
    features = _build_feature_frame(bundle, config)
    if features.empty:
        return _empty_signal_frame()

    regime = build_market_regime(bundle.prices)
    regime["dist_ma60"] = regime["market_equity"] / regime["market_ma60"] - 1
    regime["market_gate"] = (
        (regime["market_equity"] > regime["market_ma60"])
        & (regime["breadth_ma20"] >= config.market_breadth_min)
        & (regime["dist_ma60"] <= config.market_dist_ma60_max)
    )
    features = features.merge(
        regime[["date", "breadth_ma20", "market_equity", "market_ma60", "dist_ma60", "market_gate"]],
        on="date",
        how="left",
    )
    if config.apply_market_gate:
        features = features.loc[features["market_gate"].fillna(False)].copy()
    if config.market_breadth_floor is not None:
        features = features.loc[features["breadth_ma20"].fillna(0.0) >= config.market_breadth_floor].copy()
    if features.empty:
        return _empty_signal_frame()

    scored = _score_live20k_frame(features, config.factor_specs)
    scored = _apply_technical_overlay(scored)
    scored = scored.loc[scored["hits"] >= config.min_factor_hits].copy()
    if scored.empty:
        return _empty_signal_frame()
    scored = scored.sort_values(["date", "score", "hits", "code"], ascending=[True, False, False, True])
    scored["raw_rank"] = scored.groupby("date").cumcount() + 1
    scored = scored.loc[scored["raw_rank"] >= config.rank_start].copy()
    scored["score_rank"] = scored.groupby("date").cumcount() + 1
    scored = scored.loc[scored["score_rank"] <= config.signal_limit].copy()
    columns = [
        "date",
        "code",
        "raw_rank",
        "score_rank",
        "score",
        "hits",
        "name",
        "industry",
        "close",
        "tech_score",
        "tech_grade",
        "trade_advice",
        "position_scale",
        "risk_flags",
    ]
    return scored[columns].reset_index(drop=True)


def build_live20k_watchlist(
    bundle: MarketDataBundle,
    as_of: str | pd.Timestamp,
    signal_config: Live20KSignalConfig | None = None,
    limit: int = 10,
) -> pd.DataFrame:
    config = signal_config or Live20KSignalConfig()
    watch_config = replace(
        config,
        apply_market_gate=False,
        market_breadth_floor=None,
        rank_start=1,
        signal_limit=max(limit, config.signal_limit),
    )
    signals = generate_live20k_signals(bundle, watch_config)
    if signals.empty:
        return _empty_watchlist_frame()

    as_of_date = pd.Timestamp(as_of)
    signals["date"] = pd.to_datetime(signals["date"])
    regime = build_market_regime(bundle.prices)
    regime["dist_ma60"] = regime["market_equity"] / regime["market_ma60"] - 1
    regime["market_gate"] = (
        (regime["market_equity"] > regime["market_ma60"])
        & (regime["breadth_ma20"] >= config.market_breadth_min)
        & (regime["dist_ma60"] <= config.market_dist_ma60_max)
    )
    regime_row = regime.loc[regime["date"] <= as_of_date].tail(1)
    if regime_row.empty:
        return _empty_watchlist_frame()
    row = regime_row.iloc[0]
    watch_date = pd.Timestamp(row["date"])
    latest = signals.loc[signals["date"] == watch_date].sort_values("score_rank").head(limit).copy()
    if latest.empty:
        return _empty_watchlist_frame()
    market_gate = bool(row["market_gate"])
    entry_gate = live20k_entry_gate_open(config, row)
    latest["market_gate"] = market_gate
    latest["breadth_ma20"] = float(row["breadth_ma20"])
    latest["dist_ma60"] = float(row["dist_ma60"])
    latest["candidate_action"] = "buy_candidate" if entry_gate else "watch_only"
    return latest[_watchlist_columns()].reset_index(drop=True)


def live20k_entry_gate_open(config: Live20KSignalConfig, row: pd.Series) -> bool:
    market_gate = bool(row.get("market_gate", False))
    market_gate_ok = market_gate or not config.apply_market_gate
    breadth_floor = config.market_breadth_floor
    if breadth_floor is None:
        return market_gate_ok
    breadth = row.get("breadth_ma20", np.nan)
    breadth_floor_ok = pd.notna(breadth) and float(breadth) >= breadth_floor
    return bool(market_gate_ok and breadth_floor_ok)


def _build_feature_frame(bundle: MarketDataBundle, config: Live20KSignalConfig) -> pd.DataFrame:
    prices = bundle.prices.copy()
    if prices.empty:
        return pd.DataFrame()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["code", "date"]).reset_index(drop=True)

    grouped = prices.groupby("code", sort=False)
    prices["history_days"] = grouped.cumcount() + 1
    close = grouped["close"]
    volume = grouped["volume"]
    amount = grouped["amount"]
    returns = close.pct_change()
    prices["momentum_3d"] = close.pct_change(3)
    prices["momentum_5d"] = close.pct_change(5)
    prices["momentum_10d"] = close.pct_change(10)
    prices["momentum_20d"] = close.pct_change(20)
    prices["amount_ratio_5d"] = prices["amount"] / amount.transform(
        lambda series: series.shift(1).rolling(20, min_periods=20).mean()
    )
    prices["volume_score"] = prices["volume"] / volume.transform(lambda series: series.shift(1).rolling(20, min_periods=20).mean())
    prices["week_52_high"] = prices["close"] / close.transform(lambda series: series.rolling(252, min_periods=1).max())
    ma5 = close.transform(lambda series: series.rolling(5, min_periods=5).mean())
    ma10 = close.transform(lambda series: series.rolling(10, min_periods=10).mean())
    ma20 = close.transform(lambda series: series.rolling(20, min_periods=20).mean())
    ma60 = close.transform(lambda series: series.rolling(60, min_periods=60).mean())
    prices["ma5"] = ma5
    prices["ma10"] = ma10
    prices["ma20"] = ma20
    prices["ma60"] = ma60
    prices["close_vs_ma5"] = prices["close"] / ma5 - 1
    prices["close_vs_ma10"] = prices["close"] / ma10 - 1
    prices["close_vs_ma20"] = prices["close"] / ma20 - 1
    prices["trend_pullback"] = (ma20 / ma60 - 1) - (prices["close"] / ma20 - 1).abs()
    prices["volatility_20d"] = returns.groupby(prices["code"]).transform(lambda series: series.rolling(20, min_periods=20).std(ddof=0))
    prices["rsi14"] = _rolling_rsi(prices)
    prices["macd_momentum"] = _macd_momentum(prices)
    prices["macd_slope_3d"] = prices.groupby("code", sort=False)["macd_momentum"].diff(3)
    prices["close_position_20d"] = _close_position(prices, 20)
    prices["close_vs_20d_high"] = prices["close"] / close.transform(
        lambda series: series.rolling(20, min_periods=20).max()
    ) - 1
    _add_technical_overlay_features(prices)
    money_flow_features = _money_flow_feature_frame(bundle.money_flow)
    if not money_flow_features.empty:
        prices = prices.merge(money_flow_features, on=["date", "code"], how="left")
    for column in ["money_flow_3d", "big_order_3d", "main_inflow_3d"]:
        if column not in prices.columns:
            prices[column] = 0.0
        prices[column] = pd.to_numeric(prices[column], errors="coerce").fillna(0.0)

    universe = prices.copy()
    if not bundle.stocks.empty:
        stocks = bundle.stocks.copy()
        stocks["list_date"] = pd.to_datetime(stocks["list_date"], errors="coerce")
        universe = universe.merge(stocks, on="code", how="left")
    else:
        universe["name"] = universe["code"]
        universe["industry"] = ""
        universe["is_st"] = 0
        universe["list_date"] = pd.NaT

    listed_days = (universe["date"] - pd.to_datetime(universe["list_date"], errors="coerce")).dt.days
    mask = pd.Series(True, index=universe.index)
    mask &= universe["history_days"] >= config.min_history_days
    mask &= listed_days.fillna(config.min_listed_days) >= config.min_listed_days
    mask &= universe.get("is_st", 0).fillna(0).astype(int) == 0
    mask &= universe.get("paused", 0).fillna(0).astype(int) == 0
    mask &= universe.get("limit_up", 0).fillna(0).astype(int) == 0
    mask &= universe.get("limit_down", 0).fillna(0).astype(int) == 0
    mask &= universe["close"].fillna(0) >= config.min_price
    if config.max_price > 0:
        mask &= universe["close"].fillna(float("inf")) <= config.max_price
    mask &= universe["amount"].fillna(0) >= config.min_amount
    codes = universe["code"].astype(str)
    if config.exclude_bj:
        mask &= ~codes.str.endswith(".BJ")
    if config.exclude_star:
        mask &= ~codes.str.startswith("688")
    if config.exclude_chinext:
        mask &= ~codes.str.startswith(("300", "301"))
    if config.industry_allowlist:
        industries = universe.get("industry", pd.Series("", index=universe.index)).fillna("").astype(str)
        mask &= industries.isin(config.industry_allowlist)
    if config.min_money_flow_3d is not None:
        mask &= universe["money_flow_3d"] > config.min_money_flow_3d
    if config.min_big_order_3d is not None:
        mask &= universe["big_order_3d"] > config.min_big_order_3d
    if config.min_close_position_20d is not None:
        mask &= universe["close_position_20d"] >= config.min_close_position_20d
    if config.max_close_position_20d is not None:
        mask &= universe["close_position_20d"] <= config.max_close_position_20d
    if config.min_momentum_20d is not None:
        mask &= universe["momentum_20d"] >= config.min_momentum_20d
    if config.max_momentum_20d is not None:
        mask &= universe["momentum_20d"] <= config.max_momentum_20d
    if config.min_momentum_5d is not None:
        mask &= universe["momentum_5d"] >= config.min_momentum_5d
    if config.max_momentum_5d is not None:
        mask &= universe["momentum_5d"] <= config.max_momentum_5d
    if config.min_amount_ratio_5d is not None:
        mask &= universe["amount_ratio_5d"] >= config.min_amount_ratio_5d
    if config.min_trend_pullback is not None:
        mask &= universe["trend_pullback"] >= config.min_trend_pullback
    if config.max_close_vs_20d_high is not None:
        mask &= universe["close_vs_20d_high"] <= config.max_close_vs_20d_high
    if config.dynamic_hotspot:
        hot = build_hot_industries(
            bundle,
            window=config.hotspot_window,
            top_n=config.hotspot_top_n,
            min_stock_count=config.hotspot_min_stock_count,
            use_money_flow=config.hotspot_use_money_flow,
            prefer_early_strength=config.hotspot_prefer_early_strength,
            exclude_retreat=config.hotspot_exclude_retreat,
            retreat_momentum_3d=config.hotspot_retreat_momentum_3d,
            retreat_breadth_delta_5d=config.hotspot_retreat_breadth_delta_5d,
        )
        if hot.empty:
            return pd.DataFrame()
        universe = universe.merge(
            hot[["date", "industry", "hot_rank", "hot_score"]],
            on=["date", "industry"],
            how="left",
        )
        mask &= universe["hot_rank"].notna()
    return universe.loc[mask].reset_index(drop=True)


def _money_flow_feature_frame(money_flow: pd.DataFrame) -> pd.DataFrame:
    columns = ["date", "code", "money_flow_3d", "big_order_3d", "main_inflow_3d"]
    if money_flow.empty or "date" not in money_flow.columns or "code" not in money_flow.columns:
        return pd.DataFrame(columns=columns)
    frame = money_flow.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["code"] = frame["code"].astype(str)
    frame["main_net_inflow_pct"] = pd.to_numeric(frame.get("main_net_inflow_pct", 0.0), errors="coerce").fillna(0.0)
    large = pd.to_numeric(frame.get("large_net_inflow_pct", 0.0), errors="coerce").fillna(0.0)
    super_large = pd.to_numeric(frame.get("super_large_net_inflow_pct", 0.0), errors="coerce").fillna(0.0)
    frame["big_order_pct"] = large + super_large
    frame["main_net_inflow"] = pd.to_numeric(frame.get("main_net_inflow", 0.0), errors="coerce").fillna(0.0)
    frame = frame.sort_values(["code", "date"]).reset_index(drop=True)
    grouped = frame.groupby("code", sort=False)
    frame["money_flow_3d"] = grouped["main_net_inflow_pct"].transform(lambda series: series.rolling(3, min_periods=1).sum())
    frame["big_order_3d"] = grouped["big_order_pct"].transform(lambda series: series.rolling(3, min_periods=1).sum())
    frame["main_inflow_3d"] = grouped["main_net_inflow"].transform(lambda series: series.rolling(3, min_periods=1).sum())
    return frame[columns].reset_index(drop=True)


def _add_technical_overlay_features(prices: pd.DataFrame) -> None:
    grouped = prices.groupby("code", sort=False)
    previous_close = grouped["close"].shift(1)
    true_range = pd.concat(
        [
            prices["high"] - prices["low"],
            (prices["high"] - previous_close).abs(),
            (prices["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    prices["atr14_pct"] = true_range.groupby(prices["code"], sort=False).transform(
        lambda series: series.rolling(14, min_periods=14).mean()
    ) / prices["close"].replace(0, np.nan)

    ma20 = prices["ma20"]
    std20 = grouped["close"].transform(lambda series: series.rolling(20, min_periods=20).std(ddof=0))
    boll_upper = ma20 + 2 * std20
    boll_lower = ma20 - 2 * std20
    boll_span = (boll_upper - boll_lower).replace(0, np.nan)
    prices["boll_position"] = (prices["close"] - boll_lower) / boll_span
    prices["boll_width"] = boll_span / ma20.replace(0, np.nan)

    low9 = grouped["low"].transform(lambda series: series.rolling(9, min_periods=9).min())
    high9 = grouped["high"].transform(lambda series: series.rolling(9, min_periods=9).max())
    rsv = ((prices["close"] - low9) / (high9 - low9).replace(0, np.nan) * 100).fillna(50.0)
    prices["kdj_k"] = rsv.groupby(prices["code"], group_keys=False, sort=False).apply(
        lambda series: series.ewm(alpha=1 / 3, adjust=False).mean()
    ).reindex(prices.index)
    prices["kdj_d"] = prices.groupby("code", group_keys=False, sort=False)["kdj_k"].apply(
        lambda series: series.ewm(alpha=1 / 3, adjust=False).mean()
    ).reindex(prices.index)
    prices["kdj_j"] = 3 * prices["kdj_k"] - 2 * prices["kdj_d"]

    close_diff = grouped["close"].diff()
    obv_step = np.sign(close_diff.fillna(0.0)) * prices["volume"].fillna(0.0)
    obv = obv_step.groupby(prices["code"], sort=False).cumsum()
    volume_base = grouped["volume"].transform(lambda series: series.rolling(5, min_periods=1).sum()).replace(0, np.nan)
    prices["obv_slope_5d"] = obv.groupby(prices["code"], sort=False).diff(5) / volume_base

    candle_range = (prices["high"] - prices["low"]).replace(0, np.nan)
    body = (prices["close"] - prices["open"]).abs()
    upper = prices["high"] - pd.concat([prices["open"], prices["close"]], axis=1).max(axis=1)
    lower = pd.concat([prices["open"], prices["close"]], axis=1).min(axis=1) - prices["low"]
    prices["body_ratio"] = body / candle_range
    prices["upper_shadow_ratio"] = upper.clip(lower=0) / candle_range
    prices["lower_shadow_ratio"] = lower.clip(lower=0) / candle_range
    prices["candle_close_position"] = (prices["close"] - prices["low"]) / candle_range


def _score_live20k_frame(frame: pd.DataFrame, specs: list[FactorSpec]) -> pd.DataFrame:
    scored = frame.copy()
    scored["score"] = 0.0
    scored["hits"] = 0
    for spec in [item for item in specs if item.enabled]:
        if spec.name not in scored.columns:
            continue
        factor_score = (
            scored.groupby("date", group_keys=False)[spec.name]
            .apply(robust_zscore)
            .reindex(scored.index)
            .fillna(0.0)
            * spec.direction
            * spec.weight
        )
        scored[f"{spec.name}_score"] = factor_score
        scored["score"] += factor_score
        scored["hits"] += (factor_score > 0).astype(int)
    return scored


def _apply_technical_overlay(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    scored = frame.copy()
    numeric_defaults = {
        "close_vs_ma10": 0.0,
        "close_vs_ma20": 0.0,
        "rsi14": 50.0,
        "macd_momentum": 0.0,
        "macd_slope_3d": 0.0,
        "close_position_20d": 0.5,
        "atr14_pct": 0.0,
        "boll_position": 0.5,
        "obv_slope_5d": 0.0,
        "upper_shadow_ratio": 0.0,
        "candle_close_position": 0.5,
        "money_flow_3d": 0.0,
        "big_order_3d": 0.0,
        "momentum_5d": 0.0,
    }
    for column, default in numeric_defaults.items():
        if column not in scored.columns:
            scored[column] = default
        scored[column] = pd.to_numeric(scored[column], errors="coerce").fillna(default)

    tech_score = pd.Series(50.0, index=scored.index)
    tech_score += np.where(scored["close_vs_ma10"] >= 0, 8.0, -10.0)
    tech_score += np.where(scored["close_vs_ma20"] >= 0, 8.0, -10.0)
    tech_score += np.where(scored["macd_momentum"] > 0, 7.0, -5.0)
    tech_score += np.where(scored["macd_slope_3d"] > 0, 4.0, -3.0)
    tech_score += np.where(scored["money_flow_3d"] > 0, 6.0, -8.0)
    tech_score += np.where(scored["big_order_3d"] > 0, 5.0, -7.0)
    tech_score += np.where(scored["obv_slope_5d"] >= 0, 4.0, -5.0)
    tech_score += np.where(scored["close_position_20d"].between(0.65, 0.97), 5.0, -3.0)
    tech_score += np.where(scored["rsi14"].between(45, 82), 4.0, -4.0)
    tech_score -= np.where(scored["upper_shadow_ratio"] >= 0.35, 8.0, 0.0)
    tech_score -= np.where(scored["atr14_pct"] >= 0.08, 8.0, 0.0)
    scored["tech_score"] = tech_score.clip(0, 100).round(2)

    grades: list[str] = []
    advice: list[str] = []
    scales: list[float] = []
    flags_text: list[str] = []
    for _, row in scored.iterrows():
        flags = _technical_risk_flags(row)
        grade, trade_advice, position_scale = _overlay_grade(float(row["tech_score"]), flags)
        grades.append(grade)
        advice.append(trade_advice)
        scales.append(position_scale)
        flags_text.append("|".join(flags))
    scored["tech_grade"] = grades
    scored["trade_advice"] = advice
    scored["position_scale"] = scales
    scored["risk_flags"] = flags_text
    return scored


def _technical_risk_flags(row: pd.Series) -> list[str]:
    flags: list[str] = []
    if float(row.get("close_position_20d", 0.5)) >= 0.96 and float(row.get("rsi14", 50.0)) >= 82:
        flags.append("高位过热")
    if float(row.get("upper_shadow_ratio", 0.0)) >= 0.35 and float(row.get("candle_close_position", 0.5)) <= 0.65:
        flags.append("长上影")
    if float(row.get("atr14_pct", 0.0)) >= 0.08:
        flags.append("波动过高")
    if float(row.get("momentum_5d", 0.0)) > 0 and (
        float(row.get("obv_slope_5d", 0.0)) < 0 or float(row.get("money_flow_3d", 0.0)) < 0
    ):
        flags.append("量价背离")
    if float(row.get("close_vs_ma10", 0.0)) < -0.02 or float(row.get("close_vs_ma20", 0.0)) < -0.02:
        flags.append("趋势破位")
    if float(row.get("money_flow_3d", 0.0)) < 0 or float(row.get("big_order_3d", 0.0)) < 0:
        flags.append("资金转弱")
    if bool(row.get("industry_retreat", False)):
        flags.append("行业退潮")
    return flags


def _overlay_grade(score: float, flags: list[str]) -> tuple[str, str, float]:
    severe = {"趋势破位", "行业退潮"}
    severe_count = sum(1 for flag in flags if flag in severe)
    if severe_count > 0 or score < 45:
        return "D", "观察不买", 0.0
    if len(flags) >= 2 or score < 62:
        return "C", "半仓买", 0.5
    if score >= 78 and not flags:
        return "A", "正常买", 1.0
    return "B", "正常买", 1.0


def _normalize_holdings(holdings: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["code", "shares", "average_cost", "high_close", "entry_shares", "added", "entry_date"]
    if holdings is None or holdings.empty:
        return pd.DataFrame(columns=columns)
    frame = holdings.copy()
    if "code" not in frame.columns or "shares" not in frame.columns:
        raise ValueError("holdings must include code and shares columns")
    frame["code"] = frame["code"].astype(str)
    frame["shares"] = pd.to_numeric(frame["shares"], errors="coerce").fillna(0).astype(int)
    if "average_cost" not in frame.columns:
        frame["average_cost"] = np.nan
    frame["average_cost"] = pd.to_numeric(frame["average_cost"], errors="coerce")
    if "high_close" not in frame.columns:
        frame["high_close"] = np.nan
    frame["high_close"] = pd.to_numeric(frame["high_close"], errors="coerce")
    if "entry_shares" not in frame.columns:
        frame["entry_shares"] = frame["shares"]
    frame["entry_shares"] = pd.to_numeric(frame["entry_shares"], errors="coerce").fillna(frame["shares"]).astype(int)
    frame.loc[frame["entry_shares"] <= 0, "entry_shares"] = frame.loc[frame["entry_shares"] <= 0, "shares"]
    if "added" not in frame.columns:
        frame["added"] = False
    frame["added"] = frame["added"].apply(_coerce_bool)
    if "entry_date" not in frame.columns:
        frame["entry_date"] = pd.NaT
    frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce")
    return frame.loc[frame["shares"] > 0, columns].reset_index(drop=True)


def _latest_cash_and_peak(
    state: pd.DataFrame | None,
    config: MultiPositionTrendConfig | None = None,
) -> tuple[float, float]:
    config = config or live20k_execution_config()
    if state is None or state.empty:
        return float(config.initial_cash), float(config.initial_cash)
    frame = state.copy()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.sort_values("date")
    latest = frame.iloc[-1]
    cash = pd.to_numeric(latest.get("cash", config.initial_cash), errors="coerce")
    equity_peak = pd.to_numeric(latest.get("equity_peak", config.initial_cash), errors="coerce")
    cash_value = float(config.initial_cash if pd.isna(cash) else cash)
    peak_value = float(config.initial_cash if pd.isna(equity_peak) else equity_peak)
    return cash_value, peak_value


def _paper_equity_peak_after_fills(
    equity: float,
    previous_peak: float | None,
    fills: pd.DataFrame,
    holdings: pd.DataFrame,
    config: MultiPositionTrendConfig,
) -> float:
    old_peak = float(config.initial_cash if previous_peak is None else previous_peak)
    if _filled_portfolio_stop_liquidated(fills, holdings):
        reset_threshold = config.initial_cash * (1 + config.reset_peak_min_profit_pct)
        should_reset_peak = (
            config.reset_peak_on_portfolio_stop
            and (
                config.reset_peak_min_profit_pct <= 0
                or equity >= reset_threshold
            )
        )
        if should_reset_peak:
            return float(equity)
    return max(old_peak, float(equity))


def _filled_portfolio_stop_liquidated(fills: pd.DataFrame, holdings: pd.DataFrame) -> bool:
    if not holdings.empty or fills.empty:
        return False
    if not {"side", "reason", "status", "filled_shares"}.issubset(fills.columns):
        return False
    filled_shares = pd.to_numeric(fills["filled_shares"], errors="coerce").fillna(0)
    portfolio_sells = (
        (fills["side"].astype(str) == "sell")
        & (fills["reason"].astype(str) == "portfolio_stop")
        & (fills["status"].astype(str) == "filled")
        & (filled_shares > 0)
    )
    return bool(portfolio_sells.any())


def _plan_equity(plan: pd.DataFrame, fallback_cash: float) -> float:
    if not plan.empty and "equity" in plan.columns:
        equity = pd.to_numeric(plan["equity"].iloc[0], errors="coerce")
        if pd.notna(equity):
            return round(float(equity), 6)
    return round(float(fallback_cash), 6)


def _planned_sell_value(rows: list[dict[str, object]]) -> float:
    value = 0.0
    for row in rows:
        if row.get("action") != "sell_next_open":
            continue
        shares = pd.to_numeric(row.get("shares", 0), errors="coerce")
        reference_price = pd.to_numeric(row.get("reference_price", 0.0), errors="coerce")
        if pd.isna(shares) or pd.isna(reference_price):
            continue
        value += float(shares) * float(reference_price)
    return value


def _planned_buy_value(rows: list[dict[str, object]]) -> float:
    value = 0.0
    for row in rows:
        if row.get("action") != "buy_next_open":
            continue
        shares = pd.to_numeric(row.get("shares", 0), errors="coerce")
        reference_price = pd.to_numeric(row.get("reference_price", 0.0), errors="coerce")
        if pd.isna(shares) or pd.isna(reference_price):
            continue
        value += float(shares) * float(reference_price)
    return value


def _concat_frames(first: pd.DataFrame | None, second: pd.DataFrame | None) -> pd.DataFrame:
    frames = [frame for frame in [first, second] if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _trading_dates_in_range(
    prices: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> list[pd.Timestamp]:
    if prices.empty or "date" not in prices.columns:
        return []
    start_date = pd.Timestamp(start).normalize()
    end_date = pd.Timestamp(end).normalize()
    if end_date < start_date:
        raise ValueError(f"end date {end_date.date()} is before start date {start_date.date()}")
    dates = pd.to_datetime(prices["date"], errors="coerce").dropna().dt.normalize().drop_duplicates()
    dates = dates.loc[(dates >= start_date) & (dates <= end_date)].sort_values()
    return list(dates)


def _count_order_days(plans: pd.DataFrame) -> int:
    if plans.empty or "date" not in plans.columns or "action" not in plans.columns:
        return 0
    orders = plans.loc[plans["action"].isin(["buy_next_open", "sell_next_open"])]
    return int(orders["date"].nunique())


def _count_orders(plans: pd.DataFrame) -> int:
    if plans.empty or "action" not in plans.columns:
        return 0
    return int(plans["action"].isin(["buy_next_open", "sell_next_open"]).sum())


def _count_filled_orders(fills: pd.DataFrame) -> int:
    if fills.empty or "status" not in fills.columns:
        return 0
    return int((fills["status"] == "filled").sum())


def _count_failed_fills(fills: pd.DataFrame) -> int:
    if fills.empty or "status" not in fills.columns:
        return 0
    return int((fills["status"].astype(str) != "filled").sum())


def _count_blocking_failed_fills(fills: pd.DataFrame) -> int:
    if fills.empty or "status" not in fills.columns:
        return 0
    failed = fills.loc[fills["status"].astype(str) != "filled"].copy()
    if failed.empty:
        return 0
    if "side" not in failed.columns:
        return int(len(failed))
    side = failed["side"].astype(str).str.lower()
    status = failed["status"].astype(str)
    non_blocking_buy = (side == "buy") & status.isin(["failed_limit_up", "failed_not_tradable"])
    return int((~non_blocking_buy).sum())


def _count_rule_violations(plans: pd.DataFrame) -> int:
    if plans.empty or "action" not in plans.columns or "market_gate" not in plans.columns:
        return 0
    buy_when_gate_off = (
        (plans["action"] == "buy_next_open")
        & ~plans["market_gate"].fillna(False).astype(bool)
    )
    return int(buy_when_gate_off.sum())


def _require_ready_report(report: pd.DataFrame) -> None:
    if report.empty or "ready_for_live" not in report.columns:
        raise ValueError("live order export requires a paper report")
    latest = report.iloc[-1]
    ready = _coerce_bool(latest["ready_for_live"])
    if not ready:
        reason = str(latest.get("blocking_reason", "not_ready"))
        if not reason:
            reason = "not_ready"
        raise ValueError(f"paper report is not ready for live orders: {reason}")


def _coerce_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _pct(value: float) -> str:
    return f"{value:.2%}"


def _order_columns() -> list[str]:
    return [
        "plan_date",
        "side",
        "code",
        "name",
        "shares",
        "reference_price",
        "reason",
        "score_rank",
        "source_action",
    ]


def _empty_order_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_order_columns())


def _paper_state_metrics(states: pd.DataFrame, plans: pd.DataFrame) -> tuple[float, float, float]:
    equity_series = pd.Series(dtype=float)
    if not states.empty and "equity" in states.columns:
        equity_series = pd.to_numeric(states["equity"], errors="coerce").dropna().astype(float)
    elif not plans.empty and "equity" in plans.columns:
        equity_series = pd.to_numeric(plans["equity"], errors="coerce").dropna().astype(float)
    if equity_series.empty:
        initial = float(live20k_execution_config().initial_cash)
        return initial, initial, 0.0
    start_equity = float(equity_series.iloc[0])
    latest_equity = float(equity_series.iloc[-1])
    running_peak = equity_series.cummax()
    drawdown = equity_series / running_peak - 1
    return start_equity, latest_equity, float(drawdown.min())


def _resolve_execution_date(
    plan: pd.DataFrame,
    trading_dates: list[pd.Timestamp],
    execution_date: str | pd.Timestamp | None,
) -> pd.Timestamp:
    if execution_date is not None:
        date = pd.Timestamp(execution_date)
        if date not in trading_dates:
            raise ValueError(f"No market data for execution date {date.date()}")
        return date
    if plan.empty or "date" not in plan.columns:
        raise ValueError("plan must include a date column when execution_date is omitted")
    plan_date = pd.to_datetime(plan["date"]).max()
    eligible_dates = [date for date in trading_dates if date > plan_date]
    if not eligible_dates:
        raise ValueError(f"No trading date available after plan date {plan_date.date()}")
    return eligible_dates[0]


def _paper_buy(
    date: pd.Timestamp,
    code: str,
    requested: int,
    reason: str,
    market: pd.DataFrame,
    positions: dict[str, int],
    cash: float,
    fee_rate: float,
    slippage_rate: float,
    lot_size: int,
    previous_average_cost: float,
) -> tuple[float, dict[str, object], float]:
    price_row = market.loc[code] if code in market.index else None
    if price_row is None or _paper_not_tradable(price_row):
        return cash, _fill_row(date, code, "buy", requested, 0, 0.0, 0.0, 0.0, "failed_not_tradable", reason), previous_average_cost
    if int(price_row.get("limit_up", 0) or 0) == 1:
        return cash, _fill_row(date, code, "buy", requested, 0, 0.0, 0.0, 0.0, "failed_limit_up", reason), previous_average_cost
    price = _paper_execution_price(price_row, "buy", slippage_rate)
    budget_shares = int(cash // (price * (1 + fee_rate)))
    if lot_size > 1:
        budget_shares = budget_shares // lot_size * lot_size
    filled = min(requested, budget_shares)
    if lot_size > 1:
        filled = filled // lot_size * lot_size
    if filled <= 0:
        return cash, _fill_row(date, code, "buy", requested, 0, price, 0.0, 0.0, "failed_cash", reason), previous_average_cost
    existing_shares = int(positions.get(code, 0))
    cost = filled * price
    fee = cost * fee_rate
    cash -= cost + fee
    positions[code] = existing_shares + filled
    average_cost = ((existing_shares * previous_average_cost) + cost + fee) / positions[code]
    return cash, _fill_row(date, code, "buy", requested, filled, price, -cost, fee, "filled", reason), average_cost


def _paper_sell(
    date: pd.Timestamp,
    code: str,
    requested: int,
    reason: str,
    market: pd.DataFrame,
    positions: dict[str, int],
    cash: float,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[float, dict[str, object]]:
    price_row = market.loc[code] if code in market.index else None
    if requested <= 0:
        return cash, _fill_row(date, code, "sell", requested, 0, 0.0, 0.0, 0.0, "failed_no_position", reason)
    if price_row is None or _paper_not_tradable(price_row):
        return cash, _fill_row(date, code, "sell", requested, 0, 0.0, 0.0, 0.0, "failed_not_tradable", reason)
    if int(price_row.get("limit_down", 0) or 0) == 1:
        return cash, _fill_row(date, code, "sell", requested, 0, 0.0, 0.0, 0.0, "failed_limit_down", reason)
    price = _paper_execution_price(price_row, "sell", slippage_rate)
    proceeds = requested * price
    fee = proceeds * fee_rate
    cash += proceeds - fee
    remaining = int(positions.get(code, 0)) - requested
    if remaining > 0:
        positions[code] = remaining
    else:
        positions.pop(code, None)
    return cash, _fill_row(date, code, "sell", requested, requested, price, proceeds, fee, "filled", reason)


def _positions_to_holdings(
    positions: dict[str, int],
    average_costs: dict[str, float],
    high_closes: dict[str, float] | None = None,
    entry_shares: dict[str, int] | None = None,
    added_codes: set[str] | None = None,
    entry_dates: dict[str, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    high_closes = high_closes or {}
    entry_shares = entry_shares or {}
    added_codes = added_codes or set()
    entry_dates = entry_dates or {}
    rows = [
        {
            "code": code,
            "shares": shares,
            "average_cost": average_costs.get(code, np.nan),
            "high_close": high_closes.get(code, np.nan),
            "entry_shares": entry_shares.get(code, shares),
            "added": code in added_codes,
            "entry_date": entry_dates.get(code, pd.NaT),
        }
        for code, shares in sorted(positions.items())
        if shares > 0
    ]
    return pd.DataFrame(
        rows,
        columns=["code", "shares", "average_cost", "high_close", "entry_shares", "added", "entry_date"],
    )


def _update_high_closes_from_market(
    positions: dict[str, int],
    high_closes: dict[str, float],
    market: pd.DataFrame,
) -> dict[str, float]:
    updated = dict(high_closes)
    for code in positions:
        if code not in market.index:
            continue
        close_value = market.loc[code].get("close", np.nan)
        if pd.isna(close_value):
            continue
        updated[code] = max(float(updated.get(code, close_value)), float(close_value))
    return updated


def _mark_holdings_to_latest_close(
    prices: pd.DataFrame,
    holdings: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    frame = _normalize_holdings(holdings)
    if frame.empty or prices.empty:
        return frame
    price_frame = prices.copy()
    price_frame["date"] = pd.to_datetime(price_frame["date"])
    latest_date = price_frame.loc[price_frame["date"] <= as_of, "date"].max()
    if pd.isna(latest_date):
        return frame
    latest = price_frame.loc[price_frame["date"] == latest_date].set_index("code")
    for index, holding in frame.iterrows():
        code = str(holding["code"])
        if code not in latest.index:
            continue
        close_value = latest.loc[code].get("close", np.nan)
        if pd.isna(close_value):
            continue
        current_high = holding.get("high_close", np.nan)
        base_high = close_value if pd.isna(current_high) else current_high
        frame.loc[index, "high_close"] = max(float(base_high), float(close_value))
    return frame


def _paper_not_tradable(price_row: pd.Series) -> bool:
    return bool(int(price_row.get("paused", 0) or 0) == 1 or pd.isna(price_row.get("open")))


def _paper_trailing_stop_triggered(
    close_value: float,
    average_cost: float,
    high_close: float,
    config: MultiPositionTrendConfig,
) -> bool:
    if config.trailing_stop_activation_pct <= 0 or config.trailing_stop_pct <= 0:
        return False
    activated = high_close >= average_cost * (1 + config.trailing_stop_activation_pct)
    protected = close_value <= high_close * (1 - config.trailing_stop_pct)
    return bool(activated and protected)


def _paper_execution_price(price_row: pd.Series, side: str, slippage_rate: float) -> float:
    price = float(price_row["open"])
    if side == "buy":
        return price * (1 + slippage_rate)
    return price * (1 - slippage_rate)


def _fill_row(
    date: pd.Timestamp,
    code: str,
    side: str,
    requested_shares: int,
    filled_shares: int,
    price: float,
    cash_flow: float,
    fee: float,
    status: str,
    reason: str,
) -> dict[str, object]:
    return {
        "date": date,
        "code": code,
        "side": side,
        "requested_shares": int(requested_shares),
        "filled_shares": int(filled_shares),
        "price": round(float(price), 6),
        "cash_flow": round(float(cash_flow), 6),
        "fee": round(float(fee), 6),
        "status": status,
        "reason": reason,
    }


def _fill_columns() -> list[str]:
    return [
        "date",
        "code",
        "side",
        "requested_shares",
        "filled_shares",
        "price",
        "cash_flow",
        "fee",
        "status",
        "reason",
    ]


def _empty_fill_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_fill_columns())


def _latest_moving_average(prices: pd.DataFrame, window: int, plan_date: pd.Timestamp) -> pd.Series:
    if window <= 0:
        return pd.Series(dtype=float)
    frame = prices.copy()
    frame["_ma"] = frame.groupby("code", sort=False)["close"].transform(
        lambda series: series.rolling(window, min_periods=window).mean()
    )
    return frame.loc[frame["date"] == plan_date].set_index("code")["_ma"]


def _latest_regime_row(
    prices: pd.DataFrame,
    plan_date: pd.Timestamp,
    config: Live20KSignalConfig,
) -> pd.Series:
    regime = build_market_regime(prices)
    regime["dist_ma60"] = regime["market_equity"] / regime["market_ma60"] - 1
    regime["market_gate"] = (
        (regime["market_equity"] > regime["market_ma60"])
        & (regime["breadth_ma20"] >= config.market_breadth_min)
        & (regime["dist_ma60"] <= config.market_dist_ma60_max)
    )
    latest = regime.loc[regime["date"] <= plan_date].tail(1)
    if latest.empty:
        raise ValueError(f"No market regime available on or before {plan_date.date()}")
    return latest.iloc[0]


def _holdings_market_value(holdings: pd.DataFrame, latest: pd.DataFrame) -> float:
    value = 0.0
    for _, holding in holdings.iterrows():
        code = str(holding["code"])
        if code in latest.index and pd.notna(latest.loc[code].get("close")):
            value += int(holding["shares"]) * float(latest.loc[code]["close"])
    return value


def _build_exit_plan_rows(
    holding_frame: pd.DataFrame,
    latest: pd.DataFrame,
    ma: pd.Series,
    plan_date: pd.Timestamp,
    trading_dates: list[pd.Timestamp],
    market_gate: bool,
    regime_row: pd.Series,
    equity: float,
    drawdown: float,
    execution_config: MultiPositionTrendConfig,
    portfolio_stop: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for _, holding in holding_frame.iterrows():
        code = str(holding["code"])
        if code not in latest.index:
            continue
        close_value = latest.loc[code].get("close", np.nan)
        if pd.isna(close_value):
            continue
        close_float = float(close_value)
        average_cost = holding.get("average_cost", np.nan)
        if pd.isna(average_cost):
            average_cost = close_float
        high_close = holding.get("high_close", np.nan)
        if pd.isna(high_close):
            high_close = close_float
        high_close = max(float(high_close), close_float)
        reason = ""
        if portfolio_stop:
            reason = "portfolio_stop"
        elif close_float <= float(average_cost) * (1 - execution_config.stop_loss_pct):
            reason = "stop_loss"
        elif (
            execution_config.take_profit_pct > 0
            and close_float >= float(average_cost) * (1 + execution_config.take_profit_pct)
        ):
            reason = "take_profit"
        elif _paper_trailing_stop_triggered(close_float, float(average_cost), high_close, execution_config):
            reason = "trailing_stop"
        elif code in ma.index and pd.notna(ma.loc[code]) and close_float < float(ma.loc[code]):
            reason = "ma_break"
        elif _holding_reaches_max_holding_days(holding, plan_date, trading_dates, execution_config):
            reason = "max_holding_days"
        if reason:
            rows.append(
                _plan_row(
                    date=plan_date,
                    action="sell_next_open",
                    code=code,
                    name=str(latest.loc[code].get("name", "")),
                    reason=reason,
                    shares=int(holding["shares"]),
                    reference_price=close_float,
                    score_rank=np.nan,
                    cash_budget=0.0,
                    market_gate=market_gate,
                    regime_row=regime_row,
                    equity=equity,
                    drawdown=drawdown,
                )
            )
    return rows


def _build_add_plan_rows(
    holding_frame: pd.DataFrame,
    latest: pd.DataFrame,
    plan_date: pd.Timestamp,
    exiting_codes: set[str],
    cash_value: float,
    market_gate: bool,
    regime_row: pd.Series,
    equity: float,
    drawdown: float,
    execution_config: MultiPositionTrendConfig,
) -> list[dict[str, object]]:
    if execution_config.add_trigger_pct <= 0 or execution_config.add_position_multiple <= 0:
        return []
    rows: list[dict[str, object]] = []
    available_cash = cash_value * (1 - execution_config.cash_buffer_pct)
    for _, holding in holding_frame.iterrows():
        code = str(holding["code"])
        if code in exiting_codes or code not in latest.index or bool(holding.get("added", False)):
            continue
        close_value = latest.loc[code].get("close", np.nan)
        if pd.isna(close_value) or float(close_value) <= 0:
            continue
        average_cost = holding.get("average_cost", np.nan)
        if pd.isna(average_cost) or float(average_cost) <= 0:
            continue
        close_float = float(close_value)
        if close_float < float(average_cost) * (1 + execution_config.add_trigger_pct):
            continue
        entry_shares = int(holding.get("entry_shares", holding["shares"]) or holding["shares"])
        target_shares = _lot_sized_shares(
            entry_shares * execution_config.add_position_multiple,
            1.0,
            execution_config.lot_size,
        )
        budget_shares = _lot_sized_shares(available_cash, close_float, execution_config.lot_size)
        shares = min(target_shares, budget_shares)
        if shares < execution_config.lot_size:
            continue
        cash_budget = shares * close_float
        rows.append(
            _plan_row(
                date=plan_date,
                action="buy_next_open",
                code=code,
                name=str(latest.loc[code].get("name", "")),
                reason="profit_add",
                shares=shares,
                reference_price=close_float,
                score_rank=np.nan,
                cash_budget=cash_budget,
                market_gate=market_gate,
                regime_row=regime_row,
                equity=equity,
                drawdown=drawdown,
            )
        )
        available_cash -= cash_budget
    return rows


def _holding_reaches_max_holding_days(
    holding: pd.Series,
    plan_date: pd.Timestamp,
    trading_dates: list[pd.Timestamp],
    config: MultiPositionTrendConfig,
) -> bool:
    if config.max_holding_days <= 0:
        return False
    entry_date = pd.to_datetime(holding.get("entry_date", pd.NaT), errors="coerce")
    if pd.isna(entry_date):
        return False
    normalized_entry = pd.Timestamp(entry_date).normalize()
    normalized_plan = pd.Timestamp(plan_date).normalize()
    held_dates = [
        date
        for date in trading_dates
        if normalized_entry <= pd.Timestamp(date).normalize() <= normalized_plan
    ]
    return len(held_dates) >= config.max_holding_days


def _build_entry_plan_rows(
    bundle: MarketDataBundle,
    plan_date: pd.Timestamp,
    held_codes: set[str],
    latest: pd.DataFrame,
    cash_value: float,
    signal_config: Live20KSignalConfig,
    execution_config: MultiPositionTrendConfig,
    regime_row: pd.Series,
    equity: float,
    drawdown: float,
) -> list[dict[str, object]]:
    open_slots = max(execution_config.max_positions - len(held_codes), 0)
    if open_slots == 0:
        return []

    signals = generate_live20k_signals(bundle, signal_config)
    day_signals = signals.loc[signals["date"] == plan_date].sort_values("score_rank")
    rows: list[dict[str, object]] = []
    available_cash = cash_value * (1 - execution_config.cash_buffer_pct)
    for _, signal in day_signals.iterrows():
        if len(rows) >= open_slots:
            break
        code = str(signal["code"])
        if code in held_codes or code not in latest.index:
            continue
        close_value = latest.loc[code].get("close", signal.get("close", np.nan))
        if pd.isna(close_value) or float(close_value) <= 0:
            continue
        remaining_slots = open_slots - len(rows)
        cash_budget = available_cash / remaining_slots if remaining_slots else 0.0
        shares = _lot_sized_shares(cash_budget, float(close_value), execution_config.lot_size)
        if shares < execution_config.lot_size:
            continue
        position_value = shares * float(close_value)
        rows.append(
            _plan_row(
                date=plan_date,
                action="buy_next_open",
                code=code,
                name=str(signal.get("name", "")),
                reason="entry",
                shares=shares,
                reference_price=float(close_value),
                score_rank=float(signal["score_rank"]),
                cash_budget=cash_budget,
                market_gate=True,
                regime_row=regime_row,
                equity=equity,
                drawdown=drawdown,
                tech_score=signal.get("tech_score", np.nan),
                tech_grade=signal.get("tech_grade", ""),
                trade_advice=signal.get("trade_advice", ""),
                position_scale=signal.get("position_scale", np.nan),
                risk_flags=signal.get("risk_flags", ""),
            )
        )
        available_cash -= position_value
    return rows


def _lot_sized_shares(cash_budget: float, price: float, lot_size: int) -> int:
    if cash_budget <= 0 or price <= 0 or lot_size <= 0:
        return 0
    return math.floor(cash_budget / (price * lot_size)) * lot_size


def _plan_row(
    date: pd.Timestamp,
    action: str,
    code: str,
    name: str,
    reason: str,
    shares: int,
    reference_price: float,
    score_rank: float,
    cash_budget: float,
    market_gate: bool,
    regime_row: pd.Series,
    equity: float,
    drawdown: float,
    tech_score: object = np.nan,
    tech_grade: object = "",
    trade_advice: object = "",
    position_scale: object = np.nan,
    risk_flags: object = "",
) -> dict[str, object]:
    return {
        "date": date,
        "action": action,
        "code": code,
        "name": name,
        "reason": reason,
        "shares": shares,
        "reference_price": reference_price,
        "score_rank": score_rank,
        "cash_budget": cash_budget,
        "market_gate": market_gate,
        "breadth_ma20": float(regime_row["breadth_ma20"]),
        "dist_ma60": float(regime_row["dist_ma60"]),
        "equity": equity,
        "drawdown": drawdown,
        "tech_score": tech_score,
        "tech_grade": tech_grade,
        "trade_advice": trade_advice,
        "position_scale": position_scale,
        "risk_flags": risk_flags,
    }


def _plan_columns() -> list[str]:
    return [
        "date",
        "action",
        "code",
        "name",
        "reason",
        "shares",
        "reference_price",
        "score_rank",
        "cash_budget",
        "market_gate",
        "breadth_ma20",
        "dist_ma60",
        "equity",
        "drawdown",
        "tech_score",
        "tech_grade",
        "trade_advice",
        "position_scale",
        "risk_flags",
    ]


def _empty_plan_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_plan_columns())


def _hot_industry_columns() -> list[str]:
    return [
        "date",
        "industry",
        "hot_rank",
        "hot_score",
        "industry_momentum_20d",
        "industry_momentum_3d",
        "industry_momentum_5d",
        "industry_breadth_ma20",
        "industry_breadth_delta_5d",
        "industry_retreat",
        "industry_volume_ratio",
        "industry_limit_up_rate",
        "industry_money_flow_3d",
        "industry_big_order_3d",
        "industry_main_inflow_3d",
        "stock_count",
    ]


def _empty_hot_industry_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_hot_industry_columns())


def _rolling_rsi(prices: pd.DataFrame, window: int = 14) -> pd.Series:
    changes = prices.groupby("code", sort=False)["close"].diff()
    gains = changes.clip(lower=0)
    losses = -changes.clip(upper=0)
    average_gain = gains.groupby(prices["code"], sort=False).transform(lambda series: series.rolling(window, min_periods=window).mean())
    average_loss = losses.groupby(prices["code"], sort=False).transform(lambda series: series.rolling(window, min_periods=window).mean())
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + relative_strength)
    rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100.0)
    rsi = rsi.mask((average_loss == 0) & (average_gain == 0), 50.0)
    return rsi


def _macd_momentum(prices: pd.DataFrame) -> pd.Series:
    def calculate(close: pd.Series) -> pd.Series:
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd - signal

    return prices.groupby("code", group_keys=False, sort=False)["close"].apply(calculate).reindex(prices.index)


def _close_position(prices: pd.DataFrame, window: int) -> pd.Series:
    rolling_low = prices.groupby("code", sort=False)["close"].transform(lambda series: series.rolling(window, min_periods=window).min())
    rolling_high = prices.groupby("code", sort=False)["close"].transform(lambda series: series.rolling(window, min_periods=window).max())
    span = rolling_high - rolling_low
    position = (prices["close"] - rolling_low) / span.replace(0, np.nan)
    return position.fillna(0.5).where(span.notna(), np.nan)


def _empty_signal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "code",
            "raw_rank",
            "score_rank",
            "score",
            "hits",
            "name",
            "industry",
            "close",
            "tech_score",
            "tech_grade",
            "trade_advice",
            "position_scale",
            "risk_flags",
        ]
    )


def _watchlist_columns() -> list[str]:
    return [
        "date",
        "candidate_action",
        "code",
        "name",
        "industry",
        "close",
        "raw_rank",
        "score_rank",
        "score",
        "hits",
        "tech_score",
        "tech_grade",
        "trade_advice",
        "position_scale",
        "risk_flags",
        "market_gate",
        "breadth_ma20",
        "dist_ma60",
    ]


def _empty_watchlist_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_watchlist_columns())
