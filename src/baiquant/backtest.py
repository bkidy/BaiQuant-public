from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class BacktestConfig:
    fee_bps: float = 10.0
    annual_trading_days: int = 252


@dataclass(slots=True)
class BacktestResult:
    daily_returns: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: dict[str, float]


@dataclass(slots=True)
class AShareExecutionConfig:
    initial_cash: float = 50_000.0
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    lot_size: int = 100
    annual_trading_days: int = 252
    stop_drawdown_pct: float = 0.0
    stop_cooldown_days: int = 0
    liquidate_on_stop: bool = False


@dataclass(slots=True)
class AShareExecutionResult:
    daily_returns: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: dict[str, float]
    trades: pd.DataFrame
    positions: pd.DataFrame


@dataclass(slots=True)
class TrendPyramidConfig:
    initial_cash: float = 20_000.0
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    lot_size: int = 100
    annual_trading_days: int = 252
    initial_position_pct: float = 2 / 3
    add_trigger_pct: float = 0.05
    add_position_multiple: float = 0.5
    stop_loss_pct: float = 0.06
    ma_window: int = 5
    trailing_drawdown_pct: float = 0.0


@dataclass(slots=True)
class MultiPositionTrendConfig:
    initial_cash: float = 20_000.0
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    lot_size: int = 100
    annual_trading_days: int = 252
    max_positions: int = 3
    cash_buffer_pct: float = 0.0
    stop_loss_pct: float = 0.06
    take_profit_pct: float = 0.0
    add_trigger_pct: float = 0.0
    add_position_multiple: float = 0.5
    ma_window: int = 5
    trailing_stop_activation_pct: float = 0.0
    trailing_stop_pct: float = 0.0
    max_holding_days: int = 0
    portfolio_stop_drawdown_pct: float = 0.0
    portfolio_stop_cooldown_days: int = 0
    liquidate_on_portfolio_stop: bool = False
    reset_peak_on_portfolio_stop: bool = False
    reset_peak_min_profit_pct: float = 0.0
    use_position_scale: bool = False


def run_rebalance_backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    config = config or BacktestConfig()
    price_frame = prices.copy()
    weight_frame = weights.copy()
    price_frame["date"] = pd.to_datetime(price_frame["date"])
    weight_frame["date"] = pd.to_datetime(weight_frame["date"])

    close = price_frame.pivot(index="date", columns="code", values="close").sort_index()
    asset_returns = close.pct_change().shift(-1)
    rebalance_dates = [date for date in sorted(weight_frame["date"].unique()) if date in close.index]

    rows: list[dict[str, float | pd.Timestamp]] = []
    previous_weights = pd.Series(0.0, index=close.columns)
    fee_rate = config.fee_bps / 10_000

    for rebalance_date in rebalance_dates:
        loc = close.index.get_loc(rebalance_date)
        if loc >= len(close.index) - 1:
            continue
        next_date = close.index[loc + 1]
        day_weights = (
            weight_frame.loc[weight_frame["date"] == rebalance_date]
            .set_index("code")["weight"]
            .reindex(close.columns)
            .fillna(0.0)
        )
        turnover = float((day_weights - previous_weights).abs().sum())
        gross = float((asset_returns.loc[rebalance_date].fillna(0.0) * day_weights).sum())
        fee = turnover * fee_rate
        rows.append(
            {
                "date": next_date,
                "gross_return": gross,
                "fee_return": fee,
                "portfolio_return": gross - fee,
                "turnover": turnover,
            }
        )
        previous_weights = day_weights

    daily_returns = pd.DataFrame(rows)
    if daily_returns.empty:
        equity_curve = pd.DataFrame(columns=["date", "equity"])
        return BacktestResult(daily_returns, equity_curve, _metrics(pd.Series(dtype=float), config))

    equity = (1 + daily_returns["portfolio_return"]).cumprod()
    equity_curve = pd.DataFrame({"date": daily_returns["date"], "equity": equity})
    metrics = _metrics(daily_returns["portfolio_return"], config)
    return BacktestResult(daily_returns=daily_returns, equity_curve=equity_curve, metrics=metrics)


def run_a_share_execution_backtest(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    config: AShareExecutionConfig | None = None,
) -> AShareExecutionResult:
    config = config or AShareExecutionConfig()
    price_frame = prices.copy()
    target_frame = targets.copy()
    price_frame["date"] = pd.to_datetime(price_frame["date"])
    target_frame["date"] = pd.to_datetime(target_frame["date"])

    price_frame = price_frame.sort_values(["date", "code"]).reset_index(drop=True)
    trading_dates = list(price_frame["date"].drop_duplicates().sort_values())
    price_by_date = {
        date: group.set_index("code")
        for date, group in price_frame.groupby("date", sort=True)
    }
    close = price_frame.pivot(index="date", columns="code", values="close").sort_index()
    targets_by_execution_date = _targets_by_execution_date(target_frame, trading_dates)

    cash = float(config.initial_cash)
    positions: dict[str, int] = {}
    last_buy_date: dict[str, pd.Timestamp] = {}
    previous_equity = float(config.initial_cash)
    daily_rows: list[dict[str, float | pd.Timestamp]] = []
    trade_rows: list[dict[str, float | int | str | pd.Timestamp]] = []
    fee_rate = config.fee_bps / 10_000
    slippage_rate = config.slippage_bps / 10_000
    peak_equity = float(config.initial_cash)
    risk_stop_until_index = -1
    liquidation_pending = False
    last_close_by_code: dict[str, float] = {}

    for date_index, date in enumerate(trading_dates):
        market = price_by_date[date]
        _update_last_closes(market, last_close_by_code)
        target_shares = targets_by_execution_date.get(date)
        risk_state = "risk_stop" if date_index <= risk_stop_until_index or liquidation_pending else "active"
        if config.liquidate_on_stop and liquidation_pending:
            target_shares = pd.Series({code: 0 for code in positions}, dtype=int)
        if target_shares is not None:
            if risk_state == "risk_stop":
                target_shares = _block_new_buys_during_risk_stop(
                    date=date,
                    target_shares=target_shares,
                    current_positions=positions,
                    trade_rows=trade_rows,
                )
            trade_cash, trades = _execute_target_shares(
                date=date,
                market=market,
                current_positions=positions,
                target_shares=target_shares,
                cash=cash,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
                lot_size=config.lot_size,
                last_buy_date=last_buy_date,
            )
            cash = trade_cash
            trade_rows.extend(trades)
            if config.liquidate_on_stop and liquidation_pending and not positions:
                liquidation_pending = False

        equity = cash + _position_market_value(positions, close.loc[date], last_close_by_code)
        daily_return = equity / previous_equity - 1 if previous_equity else 0.0
        peak_equity = max(peak_equity, equity)
        drawdown = equity / peak_equity - 1 if peak_equity else 0.0
        if (
            config.stop_drawdown_pct > 0
            and drawdown <= -config.stop_drawdown_pct
            and date_index > risk_stop_until_index
        ):
            risk_stop_until_index = date_index + max(config.stop_cooldown_days, 0)
            risk_state = "risk_stop"
            if config.liquidate_on_stop and positions:
                liquidation_pending = True
        daily_rows.append(
            {
                "date": date,
                "cash": cash,
                "equity": equity,
                "portfolio_return": daily_return,
                "position_value": equity - cash,
                "drawdown": drawdown,
                "risk_state": risk_state,
            }
        )
        previous_equity = equity

    daily_returns = pd.DataFrame(daily_rows)
    if daily_returns.empty:
        equity_curve = pd.DataFrame(columns=["date", "equity"])
        metrics = _metrics(pd.Series(dtype=float), BacktestConfig(config.fee_bps, config.annual_trading_days))
    else:
        equity_curve = daily_returns[["date", "equity"]].copy()
        metrics = _metrics(
            daily_returns["portfolio_return"],
            BacktestConfig(config.fee_bps, config.annual_trading_days),
        )

    trades = pd.DataFrame(
        trade_rows,
        columns=["date", "code", "side", "requested_shares", "filled_shares", "price", "cash_flow", "fee", "status"],
    )
    final_positions = pd.DataFrame(
        [
            {"code": code, "shares": shares}
            for code, shares in sorted(positions.items())
            if shares != 0
        ],
        columns=["code", "shares"],
    )
    return AShareExecutionResult(
        daily_returns=daily_returns,
        equity_curve=equity_curve,
        metrics=metrics,
        trades=trades,
        positions=final_positions,
    )


def run_multi_position_trend_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    config: MultiPositionTrendConfig | None = None,
) -> AShareExecutionResult:
    config = config or MultiPositionTrendConfig()
    price_frame = prices.copy()
    signal_frame = signals.copy()
    price_frame["date"] = pd.to_datetime(price_frame["date"])
    signal_frame["date"] = pd.to_datetime(signal_frame["date"])
    signal_frame["_signal_order"] = range(len(signal_frame))

    price_frame = price_frame.sort_values(["date", "code"]).reset_index(drop=True)
    if config.ma_window > 0:
        price_frame["_ma"] = price_frame.groupby("code", sort=False)["close"].transform(
            lambda series: series.rolling(config.ma_window, min_periods=config.ma_window).mean()
        )
    else:
        price_frame["_ma"] = np.nan
    signal_frame = signal_frame.sort_values(["date", "_signal_order"]).reset_index(drop=True)
    trading_dates = list(price_frame["date"].drop_duplicates().sort_values())
    price_by_date = {
        date: group.set_index("code")
        for date, group in price_frame.groupby("date", sort=True)
    }
    signals_by_execution_date = _multi_signals_by_execution_date(signal_frame, trading_dates)

    cash = float(config.initial_cash)
    positions: dict[str, int] = {}
    average_costs: dict[str, float] = {}
    high_closes: dict[str, float] = {}
    entry_shares: dict[str, int] = {}
    entry_date_indices: dict[str, int] = {}
    added_codes: set[str] = set()
    pending_exits: dict[str, str] = {}
    pending_adds: set[str] = set()
    previous_equity = float(config.initial_cash)
    peak_equity = float(config.initial_cash)
    last_close_by_code: dict[str, float] = {}
    daily_rows: list[dict[str, float | int | str | pd.Timestamp]] = []
    trade_rows: list[dict[str, float | int | str | pd.Timestamp]] = []
    fee_rate = config.fee_bps / 10_000
    slippage_rate = config.slippage_bps / 10_000
    risk_stop_until_index = -1
    liquidation_pending = False

    for date_index, date in enumerate(trading_dates):
        market = price_by_date[date]
        _update_last_closes(market, last_close_by_code)
        risk_state = "risk_stop" if date_index <= risk_stop_until_index or liquidation_pending else "active"

        if config.liquidate_on_portfolio_stop and liquidation_pending:
            for code in positions:
                pending_exits.setdefault(code, "portfolio_stop")

        for code, reason in list(pending_exits.items()):
            if code not in positions:
                del pending_exits[code]
                average_costs.pop(code, None)
                high_closes.pop(code, None)
                entry_shares.pop(code, None)
                entry_date_indices.pop(code, None)
                added_codes.discard(code)
                pending_adds.discard(code)
                continue
            cash, exit_trades = _execute_trend_exit(
                date=date,
                code=code,
                reason=reason,
                market=market,
                positions=positions,
                cash=cash,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
            )
            trade_rows.extend(exit_trades)
            if code not in positions:
                del pending_exits[code]
                average_costs.pop(code, None)
                high_closes.pop(code, None)
                entry_shares.pop(code, None)
                entry_date_indices.pop(code, None)
                added_codes.discard(code)
                pending_adds.discard(code)
        if config.liquidate_on_portfolio_stop and liquidation_pending and not positions:
            liquidation_pending = False
            post_liquidation_equity = cash + _position_market_value(positions, market["close"], last_close_by_code)
            reset_threshold = config.initial_cash * (1 + config.reset_peak_min_profit_pct)
            should_reset_peak = (
                config.reset_peak_on_portfolio_stop
                and (
                    config.reset_peak_min_profit_pct <= 0
                    or post_liquidation_equity >= reset_threshold
                )
            )
            if should_reset_peak:
                peak_equity = post_liquidation_equity
            risk_state = "risk_stop" if date_index <= risk_stop_until_index else "active"

        if risk_state == "active":
            for code in sorted(pending_adds):
                if code not in positions or code in pending_exits:
                    pending_adds.discard(code)
                    continue
                cash, add_trades, average_cost, added_now = _execute_trend_add(
                    date=date,
                    code=code,
                    market=market,
                    positions=positions,
                    cash=cash,
                    fee_rate=fee_rate,
                    slippage_rate=slippage_rate,
                    lot_size=config.lot_size,
                    entry_shares=entry_shares.get(code, int(positions.get(code, 0))),
                    add_position_multiple=config.add_position_multiple,
                    average_cost=average_costs.get(code, 0.0),
                )
                trade_rows.extend(add_trades)
                if added_now:
                    average_costs[code] = average_cost
                    added_codes.add(code)
                    pending_adds.discard(code)

        execution_signals = signals_by_execution_date.get(date, [])
        if risk_state == "risk_stop":
            for signal in execution_signals:
                code = str(signal["code"])
                if code not in positions and code not in pending_exits:
                    trade_rows.append(
                        _trend_trade_row(
                            date,
                            code,
                            "buy",
                            0,
                            0,
                            0.0,
                            0.0,
                            0.0,
                            "blocked_portfolio_stop",
                            "portfolio_stop",
                        )
                    )
        elif config.max_positions > 0 and len(positions) < config.max_positions:
            for signal in execution_signals:
                code = str(signal["code"])
                if len(positions) >= config.max_positions:
                    break
                if code in positions or code in pending_exits:
                    continue
                remaining_slots = config.max_positions - len(positions)
                deployable_cash = cash * (1 - config.cash_buffer_pct)
                per_position_cash = deployable_cash / remaining_slots if remaining_slots else 0.0
                position_scale = float(signal.get("position_scale", 1.0)) if config.use_position_scale else 1.0
                if not np.isfinite(position_scale):
                    position_scale = 1.0
                position_scale = min(max(position_scale, 0.0), 1.0)
                if position_scale <= 0:
                    trade_rows.append(
                        _trend_trade_row(
                            date,
                            code,
                            "buy",
                            0,
                            0,
                            0.0,
                            0.0,
                            0.0,
                            "skipped_position_scale",
                            "entry",
                        )
                    )
                    continue
                per_position_cash *= position_scale
                cash, entry_trades, average_cost = _execute_trend_buy(
                    date=date,
                    code=code,
                    reason="entry",
                    requested_cash=per_position_cash,
                    requested_shares=None,
                    market=market,
                    positions=positions,
                    cash=cash,
                    fee_rate=fee_rate,
                    slippage_rate=slippage_rate,
                    lot_size=config.lot_size,
                    previous_average_cost=0.0,
                )
                trade_rows.extend(entry_trades)
                filled = int(entry_trades[-1]["filled_shares"]) if entry_trades else 0
                if filled > 0:
                    average_costs[code] = average_cost
                    high_closes[code] = max(high_closes.get(code, average_cost), average_cost)
                    entry_shares[code] = filled
                    entry_date_indices[code] = date_index

        for code in list(positions):
            if code in pending_exits:
                continue
            if code not in market.index:
                continue
            close_price = market.loc[code].get("close", np.nan)
            if pd.isna(close_price):
                continue
            close_value = float(close_price)
            average_cost = average_costs.get(code, close_value)
            high_closes[code] = max(high_closes.get(code, average_cost), close_value)
            high_close = high_closes[code]
            ma_price = market.loc[code].get("_ma", np.nan)
            held_days = date_index - entry_date_indices.get(code, date_index) + 1
            if close_value <= average_cost * (1 - config.stop_loss_pct):
                pending_exits[code] = "stop_loss"
            elif config.take_profit_pct > 0 and close_value >= average_cost * (1 + config.take_profit_pct):
                pending_exits[code] = "take_profit"
            elif _trailing_stop_triggered(close_value, average_cost, high_close, config):
                pending_exits[code] = "trailing_stop"
            elif pd.notna(ma_price) and close_value < float(ma_price):
                pending_exits[code] = "ma_break"
            elif config.max_holding_days > 0 and held_days >= config.max_holding_days:
                pending_exits[code] = "max_holding_days"
            elif (
                config.add_trigger_pct > 0
                and code not in added_codes
                and close_value >= average_cost * (1 + config.add_trigger_pct)
            ):
                pending_adds.add(code)

        equity = cash + _position_market_value(positions, market["close"], last_close_by_code)
        daily_return = equity / previous_equity - 1 if previous_equity else 0.0
        peak_equity = max(peak_equity, equity)
        drawdown = equity / peak_equity - 1 if peak_equity else 0.0
        if (
            config.portfolio_stop_drawdown_pct > 0
            and drawdown <= -config.portfolio_stop_drawdown_pct
            and date_index > risk_stop_until_index
        ):
            risk_stop_until_index = date_index + max(config.portfolio_stop_cooldown_days, 0)
            risk_state = "risk_stop"
            if config.liquidate_on_portfolio_stop and positions:
                liquidation_pending = True
        daily_rows.append(
            {
                "date": date,
                "cash": cash,
                "equity": equity,
                "portfolio_return": daily_return,
                "position_value": equity - cash,
                "drawdown": drawdown,
                "held_count": len(positions),
                "held_codes": ",".join(sorted(positions)),
                "risk_state": risk_state,
            }
        )
        previous_equity = equity

    daily_returns = pd.DataFrame(daily_rows)
    if daily_returns.empty:
        equity_curve = pd.DataFrame(columns=["date", "equity"])
        metrics = _metrics(pd.Series(dtype=float), BacktestConfig(config.fee_bps, config.annual_trading_days))
    else:
        equity_curve = daily_returns[["date", "equity"]].copy()
        metrics = _metrics(
            daily_returns["portfolio_return"],
            BacktestConfig(config.fee_bps, config.annual_trading_days),
        )

    trades = pd.DataFrame(
        trade_rows,
        columns=[
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
        ],
    )
    final_positions = pd.DataFrame(
        [
            {
                "code": code,
                "shares": shares,
                "average_cost": average_costs.get(code, np.nan),
                "high_close": high_closes.get(code, np.nan),
            }
            for code, shares in sorted(positions.items())
            if shares != 0
        ],
        columns=["code", "shares", "average_cost", "high_close"],
    )
    return AShareExecutionResult(
        daily_returns=daily_returns,
        equity_curve=equity_curve,
        metrics=metrics,
        trades=trades,
        positions=final_positions,
    )


def _multi_signals_by_execution_date(
    signals: pd.DataFrame,
    trading_dates: list[pd.Timestamp],
) -> dict[pd.Timestamp, list[dict[str, object]]]:
    execution_signals: dict[pd.Timestamp, list[dict[str, object]]] = {}
    if signals.empty:
        return execution_signals
    for signal_date, group in signals.groupby("date", sort=True):
        execution_index = bisect_right(trading_dates, signal_date)
        if execution_index >= len(trading_dates):
            continue
        execution_date = trading_dates[execution_index]
        seen: set[str] = set()
        records: list[dict[str, object]] = []
        for _, row in group.sort_values("_signal_order").iterrows():
            code = str(row["code"])
            if code in seen:
                continue
            seen.add(code)
            scale = row.get("position_scale", 1.0)
            scale_value = pd.to_numeric(pd.Series([scale]), errors="coerce").iloc[0]
            if pd.isna(scale_value):
                scale_value = 1.0
            records.append({"code": code, "position_scale": float(scale_value)})
        execution_signals.setdefault(execution_date, []).extend(records)
    return execution_signals


def _trailing_stop_triggered(
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


def run_trend_pyramid_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    config: TrendPyramidConfig | None = None,
) -> AShareExecutionResult:
    config = config or TrendPyramidConfig()
    price_frame = prices.copy()
    signal_frame = signals.copy()
    price_frame["date"] = pd.to_datetime(price_frame["date"])
    signal_frame["date"] = pd.to_datetime(signal_frame["date"])

    price_frame = price_frame.sort_values(["date", "code"]).reset_index(drop=True)
    signal_frame = signal_frame.sort_values(["date", "code"]).reset_index(drop=True)
    trading_dates = list(price_frame["date"].drop_duplicates().sort_values())
    price_by_date = {
        date: group.set_index("code")
        for date, group in price_frame.groupby("date", sort=True)
    }
    close = price_frame.pivot(index="date", columns="code", values="close").sort_index()
    ma = (
        close.rolling(config.ma_window, min_periods=config.ma_window).mean()
        if config.ma_window > 0
        else pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    )
    signal_by_execution_date = _signals_by_execution_date(signal_frame, trading_dates)

    cash = float(config.initial_cash)
    positions: dict[str, int] = {}
    position_code: str | None = None
    entry_shares = 0
    added = False
    average_cost = 0.0
    high_close = 0.0
    pending_exit_reason: str | None = None
    pending_add = False
    previous_equity = float(config.initial_cash)
    peak_equity = float(config.initial_cash)
    last_close_by_code: dict[str, float] = {}
    daily_rows: list[dict[str, float | int | str | pd.Timestamp | None]] = []
    trade_rows: list[dict[str, float | int | str | pd.Timestamp]] = []
    fee_rate = config.fee_bps / 10_000
    slippage_rate = config.slippage_bps / 10_000

    for date in trading_dates:
        market = price_by_date[date]
        _update_last_closes(market, last_close_by_code)

        if position_code and pending_exit_reason:
            cash, exit_trades = _execute_trend_exit(
                date=date,
                code=position_code,
                reason=pending_exit_reason,
                market=market,
                positions=positions,
                cash=cash,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
            )
            trade_rows.extend(exit_trades)
            if not positions:
                position_code = None
                entry_shares = 0
                added = False
                average_cost = 0.0
                high_close = 0.0
                pending_exit_reason = None
                pending_add = False

        if position_code and pending_add and not pending_exit_reason:
            cash, add_trades, average_cost, added_now = _execute_trend_add(
                date=date,
                code=position_code,
                market=market,
                positions=positions,
                cash=cash,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
                lot_size=config.lot_size,
                entry_shares=entry_shares,
                add_position_multiple=config.add_position_multiple,
                average_cost=average_cost,
            )
            trade_rows.extend(add_trades)
            if added_now:
                added = True
                pending_add = False

        if not position_code:
            entry_code = signal_by_execution_date.get(date)
            if entry_code:
                cash, entry_trades, average_cost, filled = _execute_trend_entry(
                    date=date,
                    code=entry_code,
                    market=market,
                    positions=positions,
                    cash=cash,
                    fee_rate=fee_rate,
                    slippage_rate=slippage_rate,
                    lot_size=config.lot_size,
                    initial_position_pct=config.initial_position_pct,
                )
                trade_rows.extend(entry_trades)
                if filled > 0:
                    position_code = entry_code
                    entry_shares = filled
                    added = False
                    pending_add = False
                    pending_exit_reason = None
                    high_close = 0.0

        if position_code and not pending_exit_reason:
            close_price = close.loc[date].get(position_code, np.nan)
            ma_price = ma.loc[date].get(position_code, np.nan) if date in ma.index else np.nan
            if pd.notna(close_price):
                close_value = float(close_price)
                high_close = max(high_close, close_value)
                trailing_active = high_close >= average_cost * (1 + config.add_trigger_pct)
                trailing_stop = (
                    config.trailing_drawdown_pct > 0
                    and trailing_active
                    and close_value <= high_close * (1 - config.trailing_drawdown_pct)
                )
                if close_value <= average_cost * (1 - config.stop_loss_pct):
                    pending_exit_reason = "stop_loss"
                elif trailing_stop:
                    pending_exit_reason = "trailing_drawdown"
                elif pd.notna(ma_price) and close_value < float(ma_price):
                    pending_exit_reason = "ma_break"
                elif not added and close_value >= average_cost * (1 + config.add_trigger_pct):
                    pending_add = True

        equity = cash + _position_market_value(positions, close.loc[date], last_close_by_code)
        daily_return = equity / previous_equity - 1 if previous_equity else 0.0
        peak_equity = max(peak_equity, equity)
        drawdown = equity / peak_equity - 1 if peak_equity else 0.0
        held_shares = int(next(iter(positions.values()))) if positions else 0
        daily_rows.append(
            {
                "date": date,
                "cash": cash,
                "equity": equity,
                "portfolio_return": daily_return,
                "position_value": equity - cash,
                "drawdown": drawdown,
                "held_code": position_code,
                "shares": held_shares,
            }
        )
        previous_equity = equity

    daily_returns = pd.DataFrame(daily_rows)
    if daily_returns.empty:
        equity_curve = pd.DataFrame(columns=["date", "equity"])
        metrics = _metrics(pd.Series(dtype=float), BacktestConfig(config.fee_bps, config.annual_trading_days))
    else:
        equity_curve = daily_returns[["date", "equity"]].copy()
        metrics = _metrics(
            daily_returns["portfolio_return"],
            BacktestConfig(config.fee_bps, config.annual_trading_days),
        )

    trades = pd.DataFrame(
        trade_rows,
        columns=[
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
        ],
    )
    final_positions = pd.DataFrame(
        [
            {"code": code, "shares": shares}
            for code, shares in sorted(positions.items())
            if shares != 0
        ],
        columns=["code", "shares"],
    )
    return AShareExecutionResult(
        daily_returns=daily_returns,
        equity_curve=equity_curve,
        metrics=metrics,
        trades=trades,
        positions=final_positions,
    )


def _signals_by_execution_date(
    signals: pd.DataFrame,
    trading_dates: list[pd.Timestamp],
) -> dict[pd.Timestamp, str]:
    execution_signals: dict[pd.Timestamp, str] = {}
    for signal_date, group in signals.groupby("date", sort=True):
        eligible_dates = [date for date in trading_dates if date > signal_date]
        if not eligible_dates:
            continue
        execution_date = eligible_dates[0]
        if execution_date not in execution_signals:
            execution_signals[execution_date] = str(group.iloc[0]["code"])
    return execution_signals


def _execute_trend_entry(
    date: pd.Timestamp,
    code: str,
    market: pd.DataFrame,
    positions: dict[str, int],
    cash: float,
    fee_rate: float,
    slippage_rate: float,
    lot_size: int,
    initial_position_pct: float,
) -> tuple[float, list[dict[str, float | int | str | pd.Timestamp]], float, int]:
    budget = cash * initial_position_pct
    cash, trades, average_cost = _execute_trend_buy(
        date=date,
        code=code,
        reason="entry",
        requested_cash=budget,
        requested_shares=None,
        market=market,
        positions=positions,
        cash=cash,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        lot_size=lot_size,
        previous_average_cost=0.0,
    )
    filled = int(trades[-1]["filled_shares"]) if trades else 0
    return cash, trades, average_cost, filled


def _execute_trend_add(
    date: pd.Timestamp,
    code: str,
    market: pd.DataFrame,
    positions: dict[str, int],
    cash: float,
    fee_rate: float,
    slippage_rate: float,
    lot_size: int,
    entry_shares: int,
    add_position_multiple: float,
    average_cost: float,
) -> tuple[float, list[dict[str, float | int | str | pd.Timestamp]], float, bool]:
    requested_shares = int(entry_shares * add_position_multiple)
    if lot_size > 1:
        requested_shares = requested_shares // lot_size * lot_size
    cash, trades, average_cost = _execute_trend_buy(
        date=date,
        code=code,
        reason="profit_add",
        requested_cash=cash,
        requested_shares=requested_shares,
        market=market,
        positions=positions,
        cash=cash,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        lot_size=lot_size,
        previous_average_cost=average_cost,
    )
    filled = int(trades[-1]["filled_shares"]) if trades else 0
    return cash, trades, average_cost, filled > 0


def _execute_trend_buy(
    date: pd.Timestamp,
    code: str,
    reason: str,
    requested_cash: float,
    requested_shares: int | None,
    market: pd.DataFrame,
    positions: dict[str, int],
    cash: float,
    fee_rate: float,
    slippage_rate: float,
    lot_size: int,
    previous_average_cost: float,
) -> tuple[float, list[dict[str, float | int | str | pd.Timestamp]], float]:
    price_row = market.loc[code] if code in market.index else None
    if price_row is None or _is_not_tradable(price_row):
        requested = requested_shares if requested_shares is not None else 0
        return cash, [_trend_trade_row(date, code, "buy", requested, 0, 0.0, 0.0, 0.0, "failed_not_tradable", reason)], previous_average_cost
    if int(price_row.get("limit_up", 0) or 0) == 1:
        requested = requested_shares if requested_shares is not None else 0
        return cash, [_trend_trade_row(date, code, "buy", requested, 0, 0.0, 0.0, 0.0, "failed_limit_up", reason)], previous_average_cost

    price = _execution_price(price_row, "buy", slippage_rate)
    budget_shares = int(min(cash, requested_cash) // (price * (1 + fee_rate)))
    if lot_size > 1:
        budget_shares = budget_shares // lot_size * lot_size
    requested = requested_shares if requested_shares is not None else budget_shares
    filled = min(requested, budget_shares)
    if lot_size > 1:
        filled = filled // lot_size * lot_size
    if filled <= 0:
        return cash, [_trend_trade_row(date, code, "buy", requested, 0, price, 0.0, 0.0, "failed_cash", reason)], previous_average_cost

    existing_shares = int(positions.get(code, 0))
    cost = filled * price
    fee = cost * fee_rate
    cash -= cost + fee
    positions[code] = existing_shares + filled
    average_cost = ((existing_shares * previous_average_cost) + cost + fee) / positions[code]
    return cash, [_trend_trade_row(date, code, "buy", requested, filled, price, -cost, fee, "filled", reason)], average_cost


def _execute_trend_exit(
    date: pd.Timestamp,
    code: str,
    reason: str,
    market: pd.DataFrame,
    positions: dict[str, int],
    cash: float,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[float, list[dict[str, float | int | str | pd.Timestamp]]]:
    requested = int(positions.get(code, 0))
    price_row = market.loc[code] if code in market.index else None
    if price_row is None or _is_not_tradable(price_row):
        return cash, [_trend_trade_row(date, code, "sell", requested, 0, 0.0, 0.0, 0.0, "failed_not_tradable", reason)]
    if int(price_row.get("limit_down", 0) or 0) == 1:
        return cash, [_trend_trade_row(date, code, "sell", requested, 0, 0.0, 0.0, 0.0, "failed_limit_down", reason)]

    price = _execution_price(price_row, "sell", slippage_rate)
    proceeds = requested * price
    fee = proceeds * fee_rate
    cash += proceeds - fee
    positions[code] = 0
    del positions[code]
    return cash, [_trend_trade_row(date, code, "sell", requested, requested, price, proceeds, fee, "filled", reason)]


def _trend_trade_row(
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
) -> dict[str, float | int | str | pd.Timestamp]:
    row = _trade_row(date, code, side, requested_shares, filled_shares, price, cash_flow, fee, status)
    row["reason"] = reason
    return row


def _block_new_buys_during_risk_stop(
    date: pd.Timestamp,
    target_shares: pd.Series,
    current_positions: dict[str, int],
    trade_rows: list[dict[str, float | int | str | pd.Timestamp]],
) -> pd.Series:
    adjusted = target_shares.copy()
    for code, target in target_shares.items():
        current = int(current_positions.get(code, 0))
        if int(target) > current:
            adjusted.loc[code] = current
            trade_rows.append(
                _trade_row(
                    date,
                    str(code),
                    "buy",
                    int(target) - current,
                    0,
                    0.0,
                    0.0,
                    0.0,
                    "blocked_risk_stop",
                )
            )
    return adjusted


def _targets_by_execution_date(
    targets: pd.DataFrame,
    trading_dates: list[pd.Timestamp],
) -> dict[pd.Timestamp, pd.Series]:
    execution_targets: dict[pd.Timestamp, pd.Series] = {}
    trading_index = {date: index for index, date in enumerate(trading_dates)}
    for signal_date, group in targets.groupby("date", sort=True):
        eligible_dates = [date for date in trading_dates if date > signal_date]
        if not eligible_dates:
            continue
        execution_date = eligible_dates[0]
        execution_targets[execution_date] = (
            group.set_index("code")["shares"].astype(int).groupby(level=0).last()
        )
    return execution_targets


def _execute_target_shares(
    date: pd.Timestamp,
    market: pd.DataFrame,
    current_positions: dict[str, int],
    target_shares: pd.Series,
    cash: float,
    fee_rate: float,
    slippage_rate: float,
    lot_size: int,
    last_buy_date: dict[str, pd.Timestamp],
) -> tuple[float, list[dict[str, float | int | str | pd.Timestamp]]]:
    trades: list[dict[str, float | int | str | pd.Timestamp]] = []
    codes = sorted(set(current_positions) | set(target_shares.index))
    desired = {code: int(target_shares.get(code, 0)) for code in codes}

    for code in codes:
        current = int(current_positions.get(code, 0))
        target = desired[code]
        if target >= current:
            continue
        requested = current - target
        price_row = market.loc[code] if code in market.index else None
        if price_row is None or _is_not_tradable(price_row):
            trades.append(_trade_row(date, code, "sell", requested, 0, 0.0, 0.0, 0.0, "failed_not_tradable"))
            continue
        if int(price_row.get("limit_down", 0) or 0) == 1:
            trades.append(_trade_row(date, code, "sell", requested, 0, 0.0, 0.0, 0.0, "failed_limit_down"))
            continue
        if last_buy_date.get(code) == date:
            trades.append(_trade_row(date, code, "sell", requested, 0, 0.0, 0.0, 0.0, "failed_t_plus_1"))
            continue
        price = _execution_price(price_row, "sell", slippage_rate)
        proceeds = requested * price
        fee = proceeds * fee_rate
        cash += proceeds - fee
        current_positions[code] = current - requested
        trades.append(_trade_row(date, code, "sell", requested, requested, price, proceeds, fee, "filled"))

    for code in codes:
        current = int(current_positions.get(code, 0))
        target = desired[code]
        if target <= current:
            continue
        requested = target - current
        price_row = market.loc[code] if code in market.index else None
        if price_row is None or _is_not_tradable(price_row):
            trades.append(_trade_row(date, code, "buy", requested, 0, 0.0, 0.0, 0.0, "failed_not_tradable"))
            continue
        if int(price_row.get("limit_up", 0) or 0) == 1:
            trades.append(_trade_row(date, code, "buy", requested, 0, 0.0, 0.0, 0.0, "failed_limit_up"))
            continue
        price = _execution_price(price_row, "buy", slippage_rate)
        affordable = int(cash // (price * (1 + fee_rate)))
        if lot_size > 1:
            affordable = affordable // lot_size * lot_size
        filled = min(requested, affordable)
        if lot_size > 1:
            filled = filled // lot_size * lot_size
        if filled <= 0:
            trades.append(_trade_row(date, code, "buy", requested, 0, price, 0.0, 0.0, "failed_cash"))
            continue
        cost = filled * price
        fee = cost * fee_rate
        cash -= cost + fee
        current_positions[code] = current + filled
        last_buy_date[code] = date
        trades.append(_trade_row(date, code, "buy", requested, filled, price, -cost, fee, "filled"))

    for code in list(current_positions):
        if current_positions[code] == 0:
            del current_positions[code]
    return cash, trades


def _trade_row(
    date: pd.Timestamp,
    code: str,
    side: str,
    requested_shares: int,
    filled_shares: int,
    price: float,
    cash_flow: float,
    fee: float,
    status: str,
) -> dict[str, float | int | str | pd.Timestamp]:
    return {
        "date": date,
        "code": code,
        "side": side,
        "requested_shares": requested_shares,
        "filled_shares": filled_shares,
        "price": price,
        "cash_flow": cash_flow,
        "fee": fee,
        "status": status,
    }


def _is_not_tradable(price_row: pd.Series) -> bool:
    return int(price_row.get("paused", 0) or 0) == 1


def _execution_price(price_row: pd.Series, side: str, slippage_rate: float) -> float:
    raw_price = price_row.get("open", np.nan)
    if pd.isna(raw_price) or float(raw_price) <= 0:
        raw_price = price_row.get("close", np.nan)
    price = float(raw_price)
    if side == "buy":
        return price * (1 + slippage_rate)
    return price * (1 - slippage_rate)


def _update_last_closes(market: pd.DataFrame, last_close_by_code: dict[str, float]) -> None:
    for code, row in market.iterrows():
        close_price = row.get("close", np.nan)
        if pd.notna(close_price) and float(close_price) > 0:
            last_close_by_code[str(code)] = float(close_price)


def _position_market_value(
    positions: dict[str, int],
    close_row: pd.Series,
    last_close_by_code: dict[str, float],
) -> float:
    value = 0.0
    for code, shares in positions.items():
        price = close_row.get(code, np.nan)
        if pd.isna(price):
            price = last_close_by_code.get(code, np.nan)
        if pd.notna(price):
            value += shares * float(price)
    return value


def _metrics(returns: pd.Series, config: BacktestConfig) -> dict[str, float]:
    if returns.empty:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }
    equity = (1 + returns).cumprod()
    total_return = float(equity.iloc[-1] - 1)
    periods = len(returns)
    annualized = float(equity.iloc[-1] ** (config.annual_trading_days / periods) - 1)
    volatility = float(returns.std(ddof=0) * np.sqrt(config.annual_trading_days))
    sharpe = float(returns.mean() / returns.std(ddof=0) * np.sqrt(config.annual_trading_days)) if returns.std(ddof=0) else 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return {
        "total_return": total_return,
        "annualized_return": annualized,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
    }
