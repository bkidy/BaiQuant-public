from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sqlite3
from typing import Any
from urllib.parse import parse_qs, urlparse
import webbrowser

import numpy as np
import pandas as pd

from baiquant.data.sqlite_provider import SqliteDataProvider
from baiquant.research.multifactor import (
    DEFAULT_MULTIFACTOR_WEIGHTS,
    build_multifactor_frame,
    score_multifactor_frame,
    select_multifactor_candidates,
)
from baiquant.research.regime_router import build_regime_frame
from baiquant.scoring import robust_zscore
from baiquant.strategy.live20k import (
    Live20KSignalConfig,
    _add_technical_overlay_features,
    _apply_technical_overlay,
    _macd_momentum,
    _rolling_rsi,
    build_live20k_daily_plan,
    build_live20k_watchlist,
    build_market_regime,
    score_hot_industry_frame,
    live20k_entry_gate_open,
    live100k_hotspot_manual_fixed_execution_config,
    live100k_hotspot_manual_fixed_signal_config,
    live100k_hotspot_turbo_execution_config,
    live100k_hotspot_turbo_signal_config,
)
from baiquant.strategy.multifactor import MultifactorPlanConfig, build_multifactor_daily_plan


DEFAULT_DB_PATH = "data/tushare/baiquant.db"
DEFAULT_HOLDINGS_PATH = "data/live/holdings.csv"
DEFAULT_TRADE_LOG_PATH = "data/live/trade_log.csv"
DESK_MANUAL_20D_PRESET = "20天稳打版"
DESK_TURBO_SPRINT_PRESET = "短线冲刺版"
DESK_MULTIFACTOR_PRESET = "多因子手动版"
DESK_MULTIFACTOR_ALLOWED_REGIMES = ("bull", "broad_rebound", "structural", "weak_range")
DESK_PRESET_ALIASES = {
    "steady-20d": DESK_MANUAL_20D_PRESET,
    "manual-20d": DESK_MANUAL_20D_PRESET,
    "fixed-20d": DESK_MANUAL_20D_PRESET,
    "turbo-sprint": DESK_TURBO_SPRINT_PRESET,
    "turbo": DESK_TURBO_SPRINT_PRESET,
    "manual-multifactor": DESK_MULTIFACTOR_PRESET,
    "multifactor-manual": DESK_MULTIFACTOR_PRESET,
    "手动20天版": DESK_MANUAL_20D_PRESET,
    "20天加强版": DESK_MANUAL_20D_PRESET,
    "收盘后20天主策略": DESK_MANUAL_20D_PRESET,
    "Turbo激进版": DESK_TURBO_SPRINT_PRESET,
    "短线Turbo冲刺版": DESK_TURBO_SPRINT_PRESET,
    "Turbo": DESK_TURBO_SPRINT_PRESET,
    "多因子": DESK_MULTIFACTOR_PRESET,
    "多因子版": DESK_MULTIFACTOR_PRESET,
}
DEFAULT_DESK_PRESETS = (DESK_MANUAL_20D_PRESET, DESK_TURBO_SPRINT_PRESET, DESK_MULTIFACTOR_PRESET)
HOLDING_COLUMNS = ["code", "name", "shares", "average_cost", "high_close", "entry_shares", "added", "entry_date"]
POSITION_COLUMNS = [
    *HOLDING_COLUMNS,
    "as_of",
    "industry",
    "current_price",
    "market_value",
    "cost_value",
    "unrealized_pnl",
    "unrealized_return",
    "drawdown_from_high",
    "holding_days",
    "stop_signal",
]


@dataclass(slots=True)
class DeskDefaults:
    db_path: str = DEFAULT_DB_PATH
    holdings_path: str = DEFAULT_HOLDINGS_PATH
    trade_log_path: str = DEFAULT_TRADE_LOG_PATH
    host: str = "127.0.0.1"
    port: int = 8765


def run_desk(
    host: str = "127.0.0.1",
    port: int = 8765,
    db_path: str = DEFAULT_DB_PATH,
    holdings_path: str = DEFAULT_HOLDINGS_PATH,
    trade_log_path: str = DEFAULT_TRADE_LOG_PATH,
    open_browser: bool = True,
) -> None:
    defaults = DeskDefaults(db_path=db_path, holdings_path=holdings_path, trade_log_path=trade_log_path, host=host, port=port)
    server = ThreadingHTTPServer((host, port), _make_handler(defaults))
    url = f"http://{host}:{port}"
    print(f"baiquant desk listening on {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbaiquant desk stopped")
    finally:
        server.server_close()


def load_desk_holdings(path: str | Path) -> pd.DataFrame:
    holdings_path = Path(path)
    if not holdings_path.exists():
        return pd.DataFrame(columns=HOLDING_COLUMNS)
    frame = pd.read_csv(holdings_path)
    return _normalize_desk_holdings(frame)


def save_desk_holdings(path: str | Path, rows: list[dict[str, Any]]) -> pd.DataFrame:
    holdings_path = Path(path)
    holdings_path.parent.mkdir(parents=True, exist_ok=True)
    frame = _normalize_desk_holdings(pd.DataFrame(rows))
    csv_frame = frame.copy()
    csv_frame = csv_frame.where(pd.notna(csv_frame), "")
    csv_frame.to_csv(holdings_path, index=False)
    return frame


def build_desk_positions(
    bundle: Any,
    as_of: str | pd.Timestamp,
    holdings: pd.DataFrame,
    execution_config: Any | None = None,
) -> pd.DataFrame:
    holding_frame = _normalize_desk_holdings(holdings)
    if holding_frame.empty:
        return pd.DataFrame(columns=POSITION_COLUMNS)

    prices = bundle.prices.copy()
    if prices.empty:
        return _positions_without_prices(holding_frame)
    prices["date"] = pd.to_datetime(prices["date"])
    as_of_date = pd.Timestamp(as_of)
    prices = prices.loc[prices["date"] <= as_of_date].sort_values(["date", "code"]).reset_index(drop=True)
    if prices.empty:
        return _positions_without_prices(holding_frame)

    latest_date = prices["date"].max()
    latest = prices.loc[prices["date"] == latest_date].set_index("code")
    stock_info = _desk_stock_info(bundle.stocks)
    trading_dates = list(prices["date"].drop_duplicates().sort_values())
    config = execution_config or live100k_hotspot_manual_fixed_execution_config()

    rows = []
    for _, holding in holding_frame.iterrows():
        code = str(holding["code"])
        info = stock_info.get(code, {})
        current_price = latest.loc[code].get("close", np.nan) if code in latest.index else np.nan
        shares = int(holding["shares"])
        average_cost = holding.get("average_cost")
        cost_value = shares * float(average_cost) if pd.notna(average_cost) and float(average_cost) > 0 else np.nan
        market_value = shares * float(current_price) if pd.notna(current_price) else np.nan
        unrealized_pnl = market_value - cost_value if pd.notna(cost_value) else np.nan
        unrealized_return = unrealized_pnl / cost_value if pd.notna(cost_value) and cost_value else np.nan
        high_close = _desk_position_high_close(prices, code, latest_date, holding.get("entry_date"), holding.get("high_close"), current_price)
        drawdown_from_high = float(current_price) / high_close - 1 if pd.notna(current_price) and high_close else np.nan
        holding_days = _desk_holding_days(holding.get("entry_date"), latest_date, trading_dates)
        rows.append(
            {
                **{column: holding.get(column) for column in HOLDING_COLUMNS},
                "name": holding.get("name") or info.get("name", ""),
                "high_close": high_close,
                "as_of": latest_date.date().isoformat(),
                "industry": info.get("industry", ""),
                "current_price": float(current_price) if pd.notna(current_price) else np.nan,
                "market_value": float(market_value),
                "cost_value": float(cost_value) if pd.notna(cost_value) else np.nan,
                "unrealized_pnl": float(unrealized_pnl) if pd.notna(unrealized_pnl) else np.nan,
                "unrealized_return": float(unrealized_return) if pd.notna(unrealized_return) else np.nan,
                "drawdown_from_high": float(drawdown_from_high) if pd.notna(drawdown_from_high) else np.nan,
                "holding_days": holding_days,
                "stop_signal": _desk_stop_signal(
                    current_price=current_price,
                    average_cost=average_cost,
                    unrealized_return=unrealized_return,
                    drawdown_from_high=drawdown_from_high,
                    holding_days=holding_days,
                    config=config,
                ),
            }
        )
    return pd.DataFrame(rows, columns=POSITION_COLUMNS)


def build_desk_account_snapshot(positions: pd.DataFrame, cash: float | None = None) -> dict[str, Any]:
    cash_value = 0.0 if cash is None else float(cash)
    market_value = float(pd.to_numeric(positions.get("market_value", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    cost_value = float(pd.to_numeric(positions.get("cost_value", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    unrealized_pnl_series = pd.to_numeric(positions.get("unrealized_pnl", pd.Series(dtype=float)), errors="coerce")
    unrealized_pnl = float(unrealized_pnl_series.fillna(0).sum())
    known_cost = pd.to_numeric(
        positions.loc[unrealized_pnl_series.notna(), "cost_value"] if "cost_value" in positions else pd.Series(dtype=float),
        errors="coerce",
    ).fillna(0).sum()
    equity = cash_value + market_value
    return {
        "cash": cash_value,
        "market_value": market_value,
        "cost_value": cost_value,
        "equity": equity,
        "unrealized_pnl": unrealized_pnl,
        "unrealized_return": unrealized_pnl / known_cost if known_cost else None,
        "exposure": market_value / equity if equity else 0.0,
        "positions": int(len(positions)),
    }


def latest_market_date(db_path: str | Path) -> str | None:
    path = Path(db_path)
    if not path.exists():
        return None
    with sqlite3.connect(path) as connection:
        row = connection.execute('SELECT MAX("date") FROM "prices"').fetchone()
    if not row or row[0] is None:
        return None
    return str(row[0])


def build_desk_payload(
    db_path: str | Path = DEFAULT_DB_PATH,
    as_of: str | None = None,
    holdings_path: str | Path = DEFAULT_HOLDINGS_PATH,
    trade_log_path: str | Path = DEFAULT_TRADE_LOG_PATH,
    presets: tuple[str, ...] = DEFAULT_DESK_PRESETS,
    cash: float | None = None,
    equity_peak: float | None = None,
    watchlist_limit: int = 10,
    trade_log_limit: int = 50,
    plan_mode: str = "quick",
) -> dict[str, Any]:
    latest_date = latest_market_date(db_path)
    if as_of is None or not str(as_of).strip():
        as_of = latest_date
    if as_of is None:
        raise FileNotFoundError(f"No prices found in SQLite database: {db_path}")

    as_of_date = pd.Timestamp(as_of)
    bundle = SqliteDataProvider(db_path).load_window(
        as_of=as_of_date,
        lookback_days=420,
        fundamentals_lookback_days=10,
        events_lookback_days=10,
        money_flow_lookback_days=10,
    )
    holdings = load_desk_holdings(holdings_path)
    _, primary_execution_config = _preset_configs(presets[0] if presets else DEFAULT_DESK_PRESETS[0])
    positions = build_desk_positions(bundle, as_of_date, holdings, execution_config=primary_execution_config)
    account = build_desk_account_snapshot(positions, cash=cash)
    with ThreadPoolExecutor(max_workers=max(1, min(len(presets), 2))) as executor:
        snapshots = list(
            executor.map(
                lambda preset: _build_strategy_snapshot(
                    preset=preset,
                    bundle=bundle,
                    as_of_date=as_of_date,
                    holdings=holdings,
                    cash=cash,
                    equity_peak=equity_peak,
                    watchlist_limit=watchlist_limit,
                    plan_mode=plan_mode,
                )
                if _normalize_desk_preset(preset) != DESK_MULTIFACTOR_PRESET
                else _build_multifactor_strategy_snapshot(
                    bundle=bundle,
                    as_of_date=as_of_date,
                    holdings=holdings,
                    cash=cash,
                    watchlist_limit=watchlist_limit,
                ),
                presets,
            )
        )
    return {
        "db_path": str(db_path),
        "holdings_path": str(holdings_path),
        "latest_date": latest_date,
        "as_of": pd.Timestamp(as_of_date).date().isoformat(),
        "cash": cash,
        "equity_peak": equity_peak,
        "watchlist_limit": int(watchlist_limit),
        "plan_mode": plan_mode,
        "holdings": _frame_records(holdings),
        "positions": _frame_records(positions),
        "account": _json_ready_record(account),
        "trades": _frame_records(_load_desk_trade_log(trade_log_path, trade_log_limit)),
        "strategies": snapshots,
        "presets": list(DEFAULT_DESK_PRESETS),
    }


def build_review_payload(
    db_path: str | Path = DEFAULT_DB_PATH,
    as_of: str | None = None,
    preset: str = DESK_MANUAL_20D_PRESET,
    limit: int = 10,
    horizons: tuple[int, ...] = (1, 3, 5),
) -> dict[str, Any]:
    latest_date = latest_market_date(db_path)
    if as_of is None or not str(as_of).strip():
        as_of = latest_date
    if as_of is None:
        raise FileNotFoundError(f"No prices found in SQLite database: {db_path}")

    latest_as_of = pd.Timestamp(latest_date or as_of)
    bundle = SqliteDataProvider(db_path).load_window(
        as_of=latest_as_of,
        lookback_days=900,
        fundamentals_lookback_days=10,
        events_lookback_days=10,
        money_flow_lookback_days=900,
    )
    preset = _normalize_desk_preset(preset)
    signal_config, _ = _preset_configs(preset)
    watchlist = build_live20k_watchlist(bundle, as_of, signal_config=signal_config, limit=limit)
    review = _forward_review(bundle.prices, watchlist, pd.Timestamp(as_of), horizons)
    return {
        "db_path": str(db_path),
        "as_of": pd.Timestamp(as_of).date().isoformat(),
        "preset": preset,
        "horizons": list(horizons),
        "review": _frame_records(review),
    }


def _make_handler(defaults: DeskDefaults) -> type[BaseHTTPRequestHandler]:
    class DeskHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self._send_html(DESK_HTML)
                elif parsed.path == "/favicon.ico":
                    self.send_response(204)
                    self.end_headers()
                elif parsed.path == "/api/defaults":
                    self._send_json(
                        {
                            "db_path": defaults.db_path,
                            "holdings_path": defaults.holdings_path,
                            "trade_log_path": defaults.trade_log_path,
                            "latest_date": latest_market_date(defaults.db_path),
                            "presets": list(DEFAULT_DESK_PRESETS),
                        }
                    )
                elif parsed.path == "/api/dashboard":
                    query = parse_qs(parsed.query)
                    self._send_json(
                        build_desk_payload(
                            db_path=_query_value(query, "db", defaults.db_path),
                            as_of=_query_value(query, "as_of", ""),
                            holdings_path=_query_value(query, "holdings", defaults.holdings_path),
                            trade_log_path=_query_value(query, "trade_log", defaults.trade_log_path),
                            presets=_query_presets(query),
                            cash=_query_float(query, "cash"),
                            equity_peak=_query_float(query, "equity_peak"),
                            watchlist_limit=int(_query_value(query, "limit", "10")),
                            plan_mode=_query_value(query, "plan_mode", "quick"),
                        )
                    )
                elif parsed.path == "/api/trades":
                    query = parse_qs(parsed.query)
                    trade_log_path = _query_value(query, "trade_log", defaults.trade_log_path)
                    trades = _load_desk_trade_log(trade_log_path, int(_query_value(query, "limit", "50")))
                    self._send_json({"trade_log_path": trade_log_path, "trades": _frame_records(trades)})
                elif parsed.path == "/api/review":
                    query = parse_qs(parsed.query)
                    self._send_json(
                        build_review_payload(
                            db_path=_query_value(query, "db", defaults.db_path),
                            as_of=_query_value(query, "as_of", ""),
                            preset=_query_value(query, "preset", DESK_MANUAL_20D_PRESET),
                            limit=int(_query_value(query, "limit", "10")),
                        )
                    )
                else:
                    self.send_error(404, "Not found")
            except Exception as error:  # noqa: BLE001
                self._send_json({"error": str(error)}, status=400)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path not in {"/api/holdings", "/api/trades"}:
                self.send_error(404, "Not found")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8")
                payload = json.loads(body or "{}")
                if parsed.path == "/api/holdings":
                    holdings_path = payload.get("path") or defaults.holdings_path
                    rows = payload.get("holdings") or []
                    holdings = save_desk_holdings(holdings_path, rows)
                    self._send_json({"holdings_path": holdings_path, "holdings": _frame_records(holdings)})
                else:
                    holdings, trade = _record_desk_trade(payload, defaults)
                    self._send_json(
                        {
                            "holdings_path": payload.get("holdings_path") or defaults.holdings_path,
                            "trade_log_path": payload.get("trade_log_path") or defaults.trade_log_path,
                            "trade": _json_ready_record(trade),
                            "holdings": _frame_records(holdings),
                        }
                    )
            except Exception as error:  # noqa: BLE001
                self._send_json({"error": str(error)}, status=400)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send_html(self, html: str) -> None:
            data = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return DeskHandler


def _record_desk_trade(payload: dict[str, Any], defaults: DeskDefaults) -> tuple[pd.DataFrame, dict[str, Any]]:
    from baiquant.live_ledger import record_live_trade

    return record_live_trade(
        holdings_path=payload.get("holdings_path") or payload.get("path") or defaults.holdings_path,
        trade_log_path=payload.get("trade_log_path") or defaults.trade_log_path,
        trade_date=payload.get("date") or payload.get("trade_date") or "",
        trade_time=payload.get("time") or payload.get("trade_time") or "",
        action=payload.get("action") or "",
        code=payload.get("code") or "",
        name=payload.get("name") or "",
        shares=int(payload.get("shares") or 0),
        price=float(payload.get("price") or 0),
        fees=float(payload.get("fees") or 0),
        source=payload.get("source") or "desk",
        note=payload.get("note") or "",
    )


def _load_desk_trade_log(path: str | Path, limit: int = 50) -> pd.DataFrame:
    from baiquant.live_ledger import load_trade_log

    return load_trade_log(path, limit=limit)


def _build_strategy_snapshot(
    preset: str,
    bundle: Any,
    as_of_date: pd.Timestamp,
    holdings: pd.DataFrame,
    cash: float | None,
    equity_peak: float | None,
    watchlist_limit: int,
    plan_mode: str,
) -> dict[str, Any]:
    signal_config, execution_config = _preset_configs(preset)
    regime = _strategy_regime(bundle, as_of_date, signal_config)
    watchlist = _build_fast_watchlist(bundle, as_of_date, signal_config, regime, watchlist_limit)
    if plan_mode == "precise":
        plan = build_live20k_daily_plan(
            bundle,
            as_of=as_of_date,
            holdings=holdings,
            cash=cash,
            equity_peak=equity_peak,
            signal_config=signal_config,
            execution_config=execution_config,
        )
    else:
        plan = _quick_plan_from_watchlist(
            bundle=bundle,
            as_of_date=as_of_date,
            holdings=holdings,
            watchlist=watchlist,
            signal_config=signal_config,
            execution_config=execution_config,
            regime=regime,
            cash=cash,
            equity_peak=equity_peak,
        )
    return {
        "preset": preset,
        "initial_cash": execution_config.initial_cash,
        "entry_gate": live20k_entry_gate_open(signal_config, regime),
        "market_gate": bool(regime["market_gate"]),
        "breadth_floor": _breadth_floor_state(signal_config, regime),
        "regime": _json_ready_record(regime.to_dict()),
        "plan": _frame_records(plan),
        "orders": _frame_records(plan.loc[plan["action"] != "wait"]) if not plan.empty else [],
        "watchlist": _frame_records(watchlist),
    }


def _build_multifactor_strategy_snapshot(
    bundle: Any,
    as_of_date: pd.Timestamp,
    holdings: pd.DataFrame,
    cash: float | None,
    watchlist_limit: int,
) -> dict[str, Any]:
    regime = _multifactor_regime(bundle, as_of_date)
    regime_name = str(regime.get("regime", "unknown"))
    entry_gate = regime_name in DESK_MULTIFACTOR_ALLOWED_REGIMES
    factors = build_multifactor_frame(bundle, as_of_date)
    scored = score_multifactor_frame(factors, DEFAULT_MULTIFACTOR_WEIGHTS, top_n=None)
    candidates = select_multifactor_candidates(
        scored,
        top_n=watchlist_limit,
        max_per_industry=3,
        max_lot_cost=25_000,
    )
    watchlist = candidates.copy()
    if not watchlist.empty:
        watchlist["candidate_action"] = "buy_candidate" if entry_gate else "watch_only"
        watchlist["market_gate"] = entry_gate
        watchlist["regime"] = regime_name
        watchlist["breadth_ma20"] = regime.get("breadth_ma20", np.nan)
        watchlist["dist_ma60"] = regime.get("dist_ma60", np.nan)
    eligible_candidates = candidates if entry_gate else candidates.head(0)
    config = MultifactorPlanConfig()
    plan = build_multifactor_daily_plan(
        eligible_candidates,
        bundle.prices,
        as_of=as_of_date,
        holdings=holdings,
        cash=cash,
        config=config,
    )
    return {
        "preset": DESK_MULTIFACTOR_PRESET,
        "initial_cash": config.initial_cash,
        "entry_gate": entry_gate,
        "market_gate": entry_gate,
        "breadth_floor": {"enabled": True, "open": entry_gate, "floor": None},
        "regime": _json_ready_record(regime.to_dict()),
        "plan": _frame_records(plan),
        "orders": _frame_records(plan.loc[plan["action"].isin(["buy_next_open", "sell_next_open"])]) if not plan.empty else [],
        "watchlist": _frame_records(watchlist),
    }


def _build_fast_watchlist(
    bundle: Any,
    as_of_date: pd.Timestamp,
    config: Live20KSignalConfig,
    regime: pd.Series,
    limit: int,
) -> pd.DataFrame:
    features = _build_latest_feature_frame(bundle, as_of_date, config)
    if features.empty:
        return _empty_desk_watchlist()
    scored = _score_latest_frame(features, config)
    scored = _apply_technical_overlay(scored)
    scored = scored.loc[scored["hits"] >= config.min_factor_hits].copy()
    if scored.empty:
        return _empty_desk_watchlist()
    scored = scored.sort_values(["score", "hits", "code"], ascending=[False, False, True]).reset_index(drop=True)
    scored["raw_rank"] = scored.index + 1
    scored = scored.loc[scored["raw_rank"] >= 1].copy()
    scored["score_rank"] = range(1, len(scored) + 1)
    latest = scored.head(max(limit, config.signal_limit)).copy()
    latest = latest.head(limit)
    entry_gate = live20k_entry_gate_open(config, regime)
    latest["market_gate"] = bool(regime["market_gate"])
    latest["breadth_ma20"] = float(regime["breadth_ma20"])
    latest["dist_ma60"] = float(regime["dist_ma60"])
    latest["candidate_action"] = "buy_candidate" if entry_gate else "watch_only"
    return latest.reindex(columns=_desk_watchlist_columns()).reset_index(drop=True)


def _build_latest_feature_frame(bundle: Any, as_of_date: pd.Timestamp, config: Live20KSignalConfig) -> pd.DataFrame:
    prices = bundle.prices.copy()
    if prices.empty:
        return pd.DataFrame()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.loc[prices["date"] <= as_of_date].sort_values(["code", "date"]).reset_index(drop=True)
    if prices.empty:
        return pd.DataFrame()

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
    prices["above_ma20"] = prices["close"] > ma20
    prices["trend_pullback"] = (ma20 / ma60 - 1) - (prices["close"] / ma20 - 1).abs()
    prices["volatility_20d"] = returns.groupby(prices["code"]).transform(lambda series: series.rolling(20, min_periods=20).std(ddof=0))
    prices["rsi14"] = _rolling_rsi(prices)
    prices["macd_momentum"] = _macd_momentum(prices)
    prices["macd_slope_3d"] = prices.groupby("code", sort=False)["macd_momentum"].diff(3)
    prices["close_position_20d"] = _desk_close_position(prices, 20)
    prices["close_vs_20d_high"] = prices["close"] / close.transform(
        lambda series: series.rolling(20, min_periods=20).max()
    ) - 1
    _add_technical_overlay_features(prices)

    latest_date = prices["date"].max()
    latest = prices.loc[prices["date"] == latest_date].copy()
    latest = latest.merge(_latest_money_flow_features(bundle.money_flow, latest_date), on="code", how="left")
    for column in ["money_flow_3d", "big_order_3d", "main_inflow_3d"]:
        if column not in latest.columns:
            latest[column] = 0.0
        latest[column] = pd.to_numeric(latest[column], errors="coerce").fillna(0.0)

    if not bundle.stocks.empty:
        stocks = bundle.stocks.copy()
        stocks["list_date"] = pd.to_datetime(stocks["list_date"], errors="coerce")
        latest = latest.merge(stocks, on="code", how="left")
    else:
        latest["name"] = latest["code"]
        latest["industry"] = ""
        latest["is_st"] = 0
        latest["list_date"] = pd.NaT

    listed_days = (latest["date"] - pd.to_datetime(latest["list_date"], errors="coerce")).dt.days
    mask = pd.Series(True, index=latest.index)
    mask &= latest["history_days"] >= config.min_history_days
    mask &= listed_days.fillna(config.min_listed_days) >= config.min_listed_days
    mask &= latest.get("is_st", 0).fillna(0).astype(int) == 0
    mask &= latest.get("paused", 0).fillna(0).astype(int) == 0
    mask &= latest.get("limit_up", 0).fillna(0).astype(int) == 0
    mask &= latest.get("limit_down", 0).fillna(0).astype(int) == 0
    mask &= latest["close"].fillna(0) >= config.min_price
    if config.max_price > 0:
        mask &= latest["close"].fillna(float("inf")) <= config.max_price
    mask &= latest["amount"].fillna(0) >= config.min_amount
    codes = latest["code"].astype(str)
    if config.exclude_bj:
        mask &= ~codes.str.endswith(".BJ")
    if config.exclude_star:
        mask &= ~codes.str.startswith("688")
    if config.exclude_chinext:
        mask &= ~codes.str.startswith(("300", "301"))
    if config.industry_allowlist:
        industries = latest.get("industry", pd.Series("", index=latest.index)).fillna("").astype(str)
        mask &= industries.isin(config.industry_allowlist)
    if config.min_money_flow_3d is not None:
        mask &= latest["money_flow_3d"] > config.min_money_flow_3d
    if config.min_big_order_3d is not None:
        mask &= latest["big_order_3d"] > config.min_big_order_3d
    if config.min_close_position_20d is not None:
        mask &= latest["close_position_20d"] >= config.min_close_position_20d
    if config.max_close_position_20d is not None:
        mask &= latest["close_position_20d"] <= config.max_close_position_20d
    if config.min_momentum_20d is not None:
        mask &= latest["momentum_20d"] >= config.min_momentum_20d
    if config.max_momentum_20d is not None:
        mask &= latest["momentum_20d"] <= config.max_momentum_20d
    if config.min_momentum_5d is not None:
        mask &= latest["momentum_5d"] >= config.min_momentum_5d
    if config.max_momentum_5d is not None:
        mask &= latest["momentum_5d"] <= config.max_momentum_5d
    if config.min_amount_ratio_5d is not None:
        mask &= latest["amount_ratio_5d"] >= config.min_amount_ratio_5d
    if config.min_trend_pullback is not None:
        mask &= latest["trend_pullback"] >= config.min_trend_pullback
    if config.max_close_vs_20d_high is not None:
        mask &= latest["close_vs_20d_high"] <= config.max_close_vs_20d_high

    latest = latest.loc[mask].copy()
    if latest.empty:
        return pd.DataFrame()
    if config.dynamic_hotspot:
        hot = _build_latest_hot_industries(prices, bundle.stocks, bundle.money_flow, latest_date, config)
        if hot.empty:
            return pd.DataFrame()
        latest = latest.merge(hot[["industry", "hot_rank", "hot_score"]], on="industry", how="left")
        latest = latest.loc[latest["hot_rank"].notna()].copy()
    return latest.reset_index(drop=True)


def _build_latest_hot_industries(
    prices: pd.DataFrame,
    stocks: pd.DataFrame,
    money_flow: pd.DataFrame,
    latest_date: pd.Timestamp,
    config: Live20KSignalConfig,
) -> pd.DataFrame:
    if prices.empty or stocks.empty:
        return pd.DataFrame()
    stock_frame = stocks[["code", "industry"]].copy()
    stock_frame["code"] = stock_frame["code"].astype(str)
    stock_frame["industry"] = stock_frame["industry"].fillna("").astype(str)
    frame = prices.merge(stock_frame, on="code", how="left")
    frame["industry"] = frame["industry"].fillna("").astype(str)
    frame = frame.loc[frame["industry"] != ""].copy()
    if frame.empty:
        return pd.DataFrame()

    breadth = (
        frame.groupby(["date", "industry"], sort=True)
        .agg(industry_breadth_ma20=("above_ma20", "mean"))
        .reset_index()
        .sort_values(["industry", "date"])
    )
    breadth["industry_breadth_delta_5d"] = breadth.groupby("industry", sort=False)["industry_breadth_ma20"].diff(5)
    latest_breadth = breadth.loc[breadth["date"] == pd.Timestamp(latest_date), ["industry", "industry_breadth_delta_5d"]]

    latest = frame.loc[frame["date"] == pd.Timestamp(latest_date)].copy()
    latest = latest.merge(_latest_money_flow_features(money_flow, latest_date), on="code", how="left")
    for column in ["money_flow_3d", "big_order_3d", "main_inflow_3d"]:
        latest[column] = pd.to_numeric(latest.get(column, 0.0), errors="coerce").fillna(0.0)
    if latest.empty:
        return pd.DataFrame()

    industry = (
        latest.groupby("industry", sort=True)
        .agg(
            industry_momentum_20d=("momentum_20d", "mean"),
            industry_momentum_3d=("momentum_3d", "mean"),
            industry_momentum_5d=("momentum_5d", "mean"),
            industry_breadth_ma20=("above_ma20", "mean"),
            industry_volume_ratio=("volume_score", "median"),
            industry_limit_up_rate=("limit_up", "mean"),
            industry_money_flow_3d=("money_flow_3d", "median"),
            industry_big_order_3d=("big_order_3d", "median"),
            industry_main_inflow_3d=("main_inflow_3d", "sum"),
            stock_count=("code", "nunique"),
        )
        .reset_index()
    )
    industry = industry.dropna(subset=["industry_momentum_20d", "industry_breadth_ma20"])
    industry = industry.loc[industry["stock_count"] >= config.hotspot_min_stock_count].copy()
    if industry.empty:
        return pd.DataFrame()
    industry = industry.merge(latest_breadth, on="industry", how="left")
    industry["industry_breadth_delta_5d"] = pd.to_numeric(
        industry["industry_breadth_delta_5d"], errors="coerce"
    ).fillna(0.0)
    industry["industry_retreat"] = (
        (industry["industry_momentum_3d"].fillna(0.0) <= config.hotspot_retreat_momentum_3d)
        & (industry["industry_breadth_delta_5d"] <= config.hotspot_retreat_breadth_delta_5d)
    )
    industry["industry_volume_ratio"] = pd.to_numeric(industry["industry_volume_ratio"], errors="coerce").fillna(1.0)
    for column in ["industry_money_flow_3d", "industry_big_order_3d", "industry_main_inflow_3d"]:
        industry[column] = pd.to_numeric(industry[column], errors="coerce").fillna(0.0)
    industry = score_hot_industry_frame(
        industry,
        use_money_flow=config.hotspot_use_money_flow,
        prefer_early_strength=config.hotspot_prefer_early_strength,
    )
    if config.hotspot_exclude_retreat:
        industry = industry.loc[~industry["industry_retreat"]].copy()
        if industry.empty:
            return pd.DataFrame()
    industry = industry.sort_values(["hot_score", "industry"], ascending=[False, True]).reset_index(drop=True)
    industry["hot_rank"] = industry.index + 1
    return industry.loc[industry["hot_rank"] <= config.hotspot_top_n].reset_index(drop=True)


def _latest_money_flow_features(money_flow: pd.DataFrame, latest_date: pd.Timestamp) -> pd.DataFrame:
    columns = ["code", "money_flow_3d", "big_order_3d", "main_inflow_3d"]
    if money_flow.empty or "date" not in money_flow.columns or "code" not in money_flow.columns:
        return pd.DataFrame(columns=columns)
    frame = money_flow.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.loc[frame["date"] <= latest_date].sort_values(["code", "date"]).reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["main_net_inflow_pct"] = pd.to_numeric(frame.get("main_net_inflow_pct", 0.0), errors="coerce").fillna(0.0)
    large = pd.to_numeric(frame.get("large_net_inflow_pct", 0.0), errors="coerce").fillna(0.0)
    super_large = pd.to_numeric(frame.get("super_large_net_inflow_pct", 0.0), errors="coerce").fillna(0.0)
    frame["big_order_pct"] = large + super_large
    frame["main_net_inflow"] = pd.to_numeric(frame.get("main_net_inflow", 0.0), errors="coerce").fillna(0.0)
    latest = (
        frame.groupby("code", sort=False)
        .tail(3)
        .groupby("code", as_index=False)
        .agg(
            money_flow_3d=("main_net_inflow_pct", "sum"),
            big_order_3d=("big_order_pct", "sum"),
            main_inflow_3d=("main_net_inflow", "sum"),
        )
    )
    return latest[columns]


def _score_latest_frame(frame: pd.DataFrame, config: Live20KSignalConfig) -> pd.DataFrame:
    scored = frame.copy()
    scored["score"] = 0.0
    scored["hits"] = 0
    for spec in [item for item in config.factor_specs if item.enabled]:
        if spec.name not in scored.columns:
            continue
        factor_score = robust_zscore(scored[spec.name]).fillna(0.0) * spec.direction * spec.weight
        scored[f"{spec.name}_score"] = factor_score
        scored["score"] += factor_score
        scored["hits"] += (factor_score > 0).astype(int)
    return scored


def _desk_close_position(prices: pd.DataFrame, window: int) -> pd.Series:
    grouped = prices.groupby("code", sort=False)["close"]
    rolling_low = grouped.transform(lambda series: series.rolling(window, min_periods=window).min())
    rolling_high = grouped.transform(lambda series: series.rolling(window, min_periods=window).max())
    span = rolling_high - rolling_low
    position = (prices["close"] - rolling_low) / span.replace(0, np.nan)
    return position.fillna(0.5).where(span.notna(), np.nan)


def _desk_watchlist_columns() -> list[str]:
    return [
        "date",
        "candidate_action",
        "code",
        "name",
        "industry",
        "close",
        "candidate_rank",
        "raw_rank",
        "score_rank",
        "multi_factor_score",
        "factor_hits",
        "positive_factors",
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


def _empty_desk_watchlist() -> pd.DataFrame:
    return pd.DataFrame(columns=_desk_watchlist_columns())


def _quick_plan_from_watchlist(
    bundle: Any,
    as_of_date: pd.Timestamp,
    holdings: pd.DataFrame,
    watchlist: pd.DataFrame,
    signal_config: Live20KSignalConfig,
    execution_config: Any,
    regime: pd.Series,
    cash: float | None,
    equity_peak: float | None,
) -> pd.DataFrame:
    prices = bundle.prices.copy()
    if prices.empty:
        return _empty_desk_plan()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.loc[prices["date"] <= as_of_date].sort_values(["date", "code"]).reset_index(drop=True)
    if prices.empty:
        return _empty_desk_plan()

    plan_date = prices["date"].max()
    latest = prices.loc[prices["date"] == plan_date].set_index("code")
    cash_value = 0.0 if cash is None else float(cash)
    equity = cash_value + _desk_holdings_market_value(holdings, latest)
    cost_value = _desk_holdings_cost_value(holdings)
    peak = float(max(cost_value, equity) if equity_peak is None else equity_peak)
    drawdown = equity / peak - 1 if peak else 0.0
    portfolio_stop = (
        execution_config.portfolio_stop_drawdown_pct > 0
        and drawdown <= -execution_config.portfolio_stop_drawdown_pct
    )

    rows: list[dict[str, Any]] = []
    rows.extend(
        _quick_exit_rows(
            plan_date=plan_date,
            prices=prices,
            latest=latest,
            holdings=holdings,
            execution_config=execution_config,
            regime=regime,
            equity=equity,
            drawdown=drawdown,
            portfolio_stop=portfolio_stop,
        )
    )
    exiting_codes = {str(row["code"]) for row in rows if row["action"] == "sell_next_open"}
    held_codes = set(holdings["code"].astype(str)) - exiting_codes if not holdings.empty else set()
    entry_gate = live20k_entry_gate_open(signal_config, regime)
    if entry_gate and not portfolio_stop:
        rows.extend(
            _quick_entry_rows(
                plan_date=plan_date,
                latest=latest,
                watchlist=watchlist,
                held_codes=held_codes,
                cash_value=cash_value + _desk_planned_sell_value(rows) - _desk_planned_buy_value(rows),
                execution_config=execution_config,
                regime=regime,
                equity=equity,
                drawdown=drawdown,
                tech_score=signal.get("tech_score", np.nan),
                tech_grade=signal.get("tech_grade", ""),
                trade_advice=signal.get("trade_advice", ""),
                position_scale=signal.get("position_scale", np.nan),
                risk_flags=signal.get("risk_flags", ""),
            )
        )

    if not rows:
        reason = "portfolio_stop" if portfolio_stop else "market_gate_off" if not entry_gate else "no_signal"
        rows.append(
            _desk_plan_row(
                date=plan_date,
                action="wait",
                code="",
                name="",
                reason=reason,
                shares=0,
                reference_price=np.nan,
                score_rank=np.nan,
                cash_budget=0.0,
                regime=regime,
                equity=equity,
                drawdown=drawdown,
            )
        )
    return pd.DataFrame(rows, columns=_desk_plan_columns())


def _quick_exit_rows(
    plan_date: pd.Timestamp,
    prices: pd.DataFrame,
    latest: pd.DataFrame,
    holdings: pd.DataFrame,
    execution_config: Any,
    regime: pd.Series,
    equity: float,
    drawdown: float,
    portfolio_stop: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if holdings.empty:
        return rows
    trading_dates = list(prices["date"].drop_duplicates().sort_values())
    for _, holding in holdings.iterrows():
        code = str(holding["code"])
        if code not in latest.index:
            continue
        close = float(latest.loc[code].get("close", np.nan))
        if np.isnan(close) or close <= 0:
            continue
        shares = int(holding["shares"])
        average_cost = holding.get("average_cost")
        reason = ""
        if portfolio_stop:
            reason = "portfolio_stop"
        elif pd.notna(average_cost) and float(average_cost) > 0:
            gain = close / float(average_cost) - 1
            if execution_config.stop_loss_pct > 0 and gain <= -execution_config.stop_loss_pct:
                reason = "stop_loss"
            else:
                high_close = holding.get("high_close")
                high_value = close if pd.isna(high_close) else max(close, float(high_close))
                trailing_drawdown = close / high_value - 1 if high_value else 0.0
                if (
                    execution_config.trailing_stop_activation_pct > 0
                    and gain >= execution_config.trailing_stop_activation_pct
                    and trailing_drawdown <= -execution_config.trailing_stop_pct
                ):
                    reason = "trailing_stop"
        if not reason and execution_config.max_holding_days > 0 and holding.get("entry_date"):
            if _desk_holding_days_reached(holding.get("entry_date"), plan_date, trading_dates, execution_config.max_holding_days):
                reason = "max_holding_days"
        if reason:
            rows.append(
                _desk_plan_row(
                    date=plan_date,
                    action="sell_next_open",
                    code=code,
                    name=str(holding.get("name", "")),
                    reason=reason,
                    shares=shares,
                    reference_price=close,
                    score_rank=np.nan,
                    cash_budget=0.0,
                    regime=regime,
                    equity=equity,
                    drawdown=drawdown,
                )
            )
    return rows


def _quick_entry_rows(
    plan_date: pd.Timestamp,
    latest: pd.DataFrame,
    watchlist: pd.DataFrame,
    held_codes: set[str],
    cash_value: float,
    execution_config: Any,
    regime: pd.Series,
    equity: float,
    drawdown: float,
) -> list[dict[str, Any]]:
    open_slots = max(execution_config.max_positions - len(held_codes), 0)
    if open_slots == 0 or watchlist.empty:
        return []
    rows: list[dict[str, Any]] = []
    available_cash = cash_value * (1 - execution_config.cash_buffer_pct)
    for _, signal in watchlist.sort_values("score_rank").iterrows():
        if len(rows) >= open_slots:
            break
        code = str(signal["code"])
        if code in held_codes or code not in latest.index:
            continue
        close = latest.loc[code].get("close", signal.get("close", np.nan))
        if pd.isna(close) or float(close) <= 0:
            continue
        remaining_slots = open_slots - len(rows)
        cash_budget = available_cash / remaining_slots if remaining_slots else 0.0
        shares = _desk_lot_sized_shares(cash_budget, float(close), execution_config.lot_size)
        if shares < execution_config.lot_size:
            continue
        rows.append(
            _desk_plan_row(
                date=plan_date,
                action="buy_next_open",
                code=code,
                name=str(signal.get("name", "")),
                reason="entry_quick",
                shares=shares,
                reference_price=float(close),
                score_rank=float(signal.get("score_rank", np.nan)),
                cash_budget=cash_budget,
                regime=regime,
                equity=equity,
                drawdown=drawdown,
            )
        )
        available_cash -= shares * float(close)
    return rows


def _normalize_desk_holdings(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=HOLDING_COLUMNS)
    normalized = frame.copy()
    for column in HOLDING_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None
    normalized["code"] = normalized["code"].fillna("").astype(str).map(_normalize_stock_code)
    normalized["name"] = normalized["name"].fillna("").astype(str)
    normalized["shares"] = pd.to_numeric(normalized["shares"], errors="coerce").fillna(0).astype(int)
    normalized["average_cost"] = pd.to_numeric(normalized["average_cost"], errors="coerce")
    normalized["high_close"] = pd.to_numeric(normalized["high_close"], errors="coerce")
    normalized["entry_shares"] = pd.to_numeric(normalized["entry_shares"], errors="coerce").fillna(normalized["shares"]).astype(int)
    normalized.loc[normalized["entry_shares"] <= 0, "entry_shares"] = normalized.loc[normalized["entry_shares"] <= 0, "shares"]
    normalized["added"] = normalized["added"].apply(_coerce_bool)
    normalized["entry_date"] = normalized["entry_date"].fillna("").astype(str)
    normalized = normalized.loc[(normalized["code"] != "") & (normalized["shares"] > 0), HOLDING_COLUMNS].reset_index(drop=True)
    return normalized.astype(object).where(pd.notna(normalized), None)


def _normalize_stock_code(value: Any) -> str:
    code = str(value).strip().upper()
    if not code or code == "NAN":
        return ""
    if "." in code:
        left, right = code.split(".", 1)
        return f"{left.zfill(6)}.{right}"
    digits = "".join(ch for ch in code if ch.isdigit())
    if len(digits) != 6:
        return code
    if digits.startswith(("600", "601", "603", "605", "688", "689")):
        return f"{digits}.SH"
    if digits.startswith(("000", "001", "002", "003", "300", "301")):
        return f"{digits}.SZ"
    if digits.startswith(("430", "830", "831", "832", "833", "834", "835", "836", "837", "838", "839", "870", "871", "872", "873", "920")):
        return f"{digits}.BJ"
    return digits


def _positions_without_prices(holdings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, holding in holdings.iterrows():
        rows.append(
            {
                **{column: holding.get(column) for column in HOLDING_COLUMNS},
                "as_of": "",
                "industry": "",
                "current_price": np.nan,
                "market_value": 0.0,
                "cost_value": np.nan,
                "unrealized_pnl": np.nan,
                "unrealized_return": np.nan,
                "drawdown_from_high": np.nan,
                "holding_days": 0,
                "stop_signal": "missing_price",
            }
        )
    return pd.DataFrame(rows, columns=POSITION_COLUMNS)


def _desk_stock_info(stocks: pd.DataFrame) -> dict[str, dict[str, str]]:
    if stocks.empty or "code" not in stocks.columns:
        return {}
    frame = stocks.copy()
    frame["code"] = frame["code"].astype(str)
    if "name" not in frame.columns:
        frame["name"] = ""
    if "industry" not in frame.columns:
        frame["industry"] = ""
    frame["name"] = frame["name"].fillna("").astype(str)
    frame["industry"] = frame["industry"].fillna("").astype(str)
    return frame.set_index("code")[["name", "industry"]].to_dict("index")


def _desk_position_high_close(
    prices: pd.DataFrame,
    code: str,
    latest_date: pd.Timestamp,
    entry_date: Any,
    stored_high: Any,
    current_price: Any,
) -> float:
    candidates = []
    for value in [stored_high, current_price]:
        if pd.notna(value):
            candidates.append(float(value))
    entry = pd.to_datetime(entry_date, errors="coerce")
    history = prices.loc[prices["code"].astype(str) == code]
    if pd.notna(entry):
        history = history.loc[(history["date"] >= entry.normalize()) & (history["date"] <= latest_date)]
    else:
        history = history.loc[history["date"] <= latest_date]
    if not history.empty:
        max_close = pd.to_numeric(history["close"], errors="coerce").max()
        if pd.notna(max_close):
            candidates.append(float(max_close))
    return max(candidates) if candidates else np.nan


def _desk_holding_days(entry_date: Any, plan_date: pd.Timestamp, trading_dates: list[pd.Timestamp]) -> int:
    entry = pd.to_datetime(entry_date, errors="coerce")
    if pd.isna(entry):
        return 0
    normalized_entry = entry.normalize()
    normalized_plan = pd.Timestamp(plan_date).normalize()
    return len(
        [
            date
            for date in trading_dates
            if normalized_entry <= pd.Timestamp(date).normalize() <= normalized_plan
        ]
    )


def _desk_stop_signal(
    current_price: Any,
    average_cost: Any,
    unrealized_return: Any,
    drawdown_from_high: Any,
    holding_days: int,
    config: Any,
) -> str:
    if pd.isna(current_price):
        return "missing_price"
    if pd.notna(average_cost) and float(average_cost) > 0 and pd.notna(unrealized_return):
        if config.stop_loss_pct > 0 and float(unrealized_return) <= -config.stop_loss_pct:
            return "stop_loss"
        if (
            config.trailing_stop_activation_pct > 0
            and config.trailing_stop_pct > 0
            and float(unrealized_return) >= config.trailing_stop_activation_pct
            and pd.notna(drawdown_from_high)
            and float(drawdown_from_high) <= -config.trailing_stop_pct
        ):
            return "trailing_stop"
    if config.max_holding_days > 0 and holding_days >= config.max_holding_days:
        return "max_holding_days"
    return "hold"


def _desk_holdings_market_value(holdings: pd.DataFrame, latest: pd.DataFrame) -> float:
    if holdings.empty or latest.empty:
        return 0.0
    total = 0.0
    for _, holding in holdings.iterrows():
        code = str(holding["code"])
        if code not in latest.index:
            continue
        close = latest.loc[code].get("close", np.nan)
        if pd.notna(close):
            total += int(holding["shares"]) * float(close)
    return float(total)


def _desk_holdings_cost_value(holdings: pd.DataFrame) -> float:
    if holdings.empty:
        return 0.0
    total = 0.0
    for _, holding in holdings.iterrows():
        average_cost = holding.get("average_cost")
        if pd.notna(average_cost) and float(average_cost) > 0:
            total += int(holding["shares"]) * float(average_cost)
    return float(total)


def _desk_planned_sell_value(rows: list[dict[str, Any]]) -> float:
    return sum(
        float(row.get("shares", 0) or 0) * float(row.get("reference_price", 0) or 0)
        for row in rows
        if row.get("action") == "sell_next_open"
    )


def _desk_planned_buy_value(rows: list[dict[str, Any]]) -> float:
    return sum(
        float(row.get("shares", 0) or 0) * float(row.get("reference_price", 0) or 0)
        for row in rows
        if row.get("action") == "buy_next_open"
    )


def _desk_holding_days_reached(entry_date: Any, plan_date: pd.Timestamp, trading_dates: list[pd.Timestamp], max_days: int) -> bool:
    entry = pd.to_datetime(entry_date, errors="coerce")
    if pd.isna(entry):
        return False
    normalized_entry = entry.normalize()
    normalized_plan = pd.Timestamp(plan_date).normalize()
    held_dates = [
        date
        for date in trading_dates
        if normalized_entry <= pd.Timestamp(date).normalize() <= normalized_plan
    ]
    return len(held_dates) >= max_days


def _desk_lot_sized_shares(cash_budget: float, price: float, lot_size: int) -> int:
    if price <= 0 or cash_budget <= 0:
        return 0
    shares = int(cash_budget // price)
    return (shares // lot_size) * lot_size


def _desk_plan_row(
    date: pd.Timestamp,
    action: str,
    code: str,
    name: str,
    reason: str,
    shares: int,
    reference_price: float,
    score_rank: float,
    cash_budget: float,
    regime: pd.Series,
    equity: float,
    drawdown: float,
    tech_score: Any = np.nan,
    tech_grade: Any = "",
    trade_advice: Any = "",
    position_scale: Any = np.nan,
    risk_flags: Any = "",
) -> dict[str, Any]:
    return {
        "date": pd.Timestamp(date).date().isoformat(),
        "action": action,
        "code": code,
        "name": name,
        "reason": reason,
        "shares": int(shares),
        "reference_price": reference_price,
        "score_rank": score_rank,
        "cash_budget": cash_budget,
        "market_gate": bool(regime.get("market_gate", False)),
        "breadth_ma20": float(regime.get("breadth_ma20", np.nan)),
        "dist_ma60": float(regime.get("dist_ma60", np.nan)),
        "equity": equity,
        "drawdown": drawdown,
        "tech_score": tech_score,
        "tech_grade": tech_grade,
        "trade_advice": trade_advice,
        "position_scale": position_scale,
        "risk_flags": risk_flags,
    }


def _desk_plan_columns() -> list[str]:
    return [
        "date",
        "action",
        "code",
        "name",
        "reason",
        "shares",
        "reference_price",
        "average_cost",
        "current_return",
        "candidate_rank",
        "score_rank",
        "multi_factor_score",
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


def _empty_desk_plan() -> pd.DataFrame:
    return pd.DataFrame(columns=_desk_plan_columns())


def _strategy_regime(bundle: Any, as_of: pd.Timestamp, config: Live20KSignalConfig) -> pd.Series:
    regime = build_market_regime(bundle.prices)
    regime["dist_ma60"] = regime["market_equity"] / regime["market_ma60"] - 1
    regime["market_gate"] = (
        (regime["market_equity"] > regime["market_ma60"])
        & (regime["breadth_ma20"] >= config.market_breadth_min)
        & (regime["dist_ma60"] <= config.market_dist_ma60_max)
    )
    row = regime.loc[regime["date"] <= as_of].tail(1)
    if row.empty:
        raise ValueError(f"No market regime available on or before {as_of.date()}")
    return row.iloc[0]


def _multifactor_regime(bundle: Any, as_of: pd.Timestamp) -> pd.Series:
    regimes = build_regime_frame(bundle.prices)
    if regimes.empty:
        return pd.Series({"date": as_of, "regime": "unknown", "breadth_ma20": np.nan, "dist_ma60": np.nan})
    frame = regimes.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    row = frame.loc[frame["date"] <= as_of].tail(1)
    if row.empty:
        return pd.Series({"date": as_of, "regime": "unknown", "breadth_ma20": np.nan, "dist_ma60": np.nan})
    return row.iloc[0]


def _preset_configs(preset: str):
    preset = _normalize_desk_preset(preset)
    if preset == DESK_MANUAL_20D_PRESET:
        return live100k_hotspot_manual_fixed_signal_config(), live100k_hotspot_manual_fixed_execution_config()
    if preset == DESK_TURBO_SPRINT_PRESET:
        return live100k_hotspot_turbo_signal_config(), live100k_hotspot_turbo_execution_config()
    return live100k_hotspot_manual_fixed_signal_config(), live100k_hotspot_manual_fixed_execution_config()


def _normalize_desk_preset(preset: str | None) -> str:
    name = (preset or DESK_MANUAL_20D_PRESET).strip()
    return DESK_PRESET_ALIASES.get(name, name)


def _breadth_floor_state(config: Live20KSignalConfig, row: pd.Series) -> dict[str, Any]:
    if config.market_breadth_floor is None:
        return {"enabled": False, "open": True, "floor": None}
    breadth = row.get("breadth_ma20", np.nan)
    return {
        "enabled": True,
        "open": bool(pd.notna(breadth) and float(breadth) >= config.market_breadth_floor),
        "floor": float(config.market_breadth_floor),
    }


def _forward_review(prices: pd.DataFrame, watchlist: pd.DataFrame, as_of: pd.Timestamp, horizons: tuple[int, ...]) -> pd.DataFrame:
    if watchlist.empty or prices.empty:
        return pd.DataFrame(columns=["code", "name", "industry", "close", *[f"ret_{day}d" for day in horizons]])
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    trading_dates = list(frame["date"].drop_duplicates().sort_values())
    eligible_dates = [date for date in trading_dates if date <= as_of]
    if not eligible_dates:
        return pd.DataFrame()
    base_date = eligible_dates[-1]
    date_index = trading_dates.index(base_date)
    close_by_date_code = frame.set_index(["date", "code"])["close"]
    rows = []
    for _, item in watchlist.iterrows():
        code = str(item["code"])
        base_close = close_by_date_code.get((base_date, code), np.nan)
        row = {
            "code": code,
            "name": item.get("name", ""),
            "industry": item.get("industry", ""),
            "close": item.get("close", np.nan),
        }
        for horizon in horizons:
            target_index = date_index + horizon
            target_close = np.nan
            if target_index < len(trading_dates):
                target_close = close_by_date_code.get((trading_dates[target_index], code), np.nan)
            row[f"ret_{horizon}d"] = float(target_close / base_close - 1) if pd.notna(base_close) and pd.notna(target_close) and base_close else None
        rows.append(row)
    return pd.DataFrame(rows)


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    clean = frame.copy()
    for column in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[column]):
            clean[column] = clean[column].dt.strftime("%Y-%m-%d")
    clean = clean.astype(object).where(pd.notna(clean), None)
    return [_json_ready_record(row) for row in clean.to_dict("records")]


def _json_ready_record(row: dict[str, Any]) -> dict[str, Any]:
    ready: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, pd.Timestamp):
            ready[key] = value.date().isoformat()
        elif isinstance(value, np.generic):
            ready[key] = value.item()
        elif pd.isna(value) if not isinstance(value, (list, dict, tuple, bool)) else False:
            ready[key] = None
        else:
            ready[key] = value
    return ready


def _query_value(query: dict[str, list[str]], name: str, default: str) -> str:
    values = query.get(name)
    if not values:
        return default
    return values[0]


def _query_float(query: dict[str, list[str]], name: str) -> float | None:
    value = _query_value(query, name, "")
    if not value.strip():
        return None
    return float(value)


def _query_presets(query: dict[str, list[str]]) -> tuple[str, ...]:
    value = _query_value(query, "presets", ",".join(DEFAULT_DESK_PRESETS))
    presets = tuple(_normalize_desk_preset(item.strip()) for item in value.split(",") if item.strip())
    return presets or DEFAULT_DESK_PRESETS


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


DESK_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BaiQuant Desk</title>
  <style>
    :root {
      --bg: #f4f5f2;
      --panel: #ffffff;
      --ink: #1f2520;
      --muted: #69726b;
      --line: #d9ded8;
      --green: #087f4f;
      --red: #b43d3d;
      --amber: #9b6a13;
      --cyan: #0b7285;
      --shadow: 0 12px 30px rgba(31, 37, 32, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, "PingFang SC", "Microsoft YaHei", sans-serif;
      letter-spacing: 0;
    }
    header {
      display: grid;
      grid-template-columns: 220px 1fr auto;
      align-items: center;
      gap: 18px;
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: #fbfcf8;
      position: sticky;
      top: 0;
      z-index: 5;
    }
    h1 { margin: 0; font-size: 22px; font-weight: 800; }
    .sub { color: var(--muted); font-size: 12px; margin-top: 2px; }
    .controls {
      display: grid;
      grid-template-columns: minmax(220px, 2fr) 140px 110px 110px 90px 110px minmax(180px, 1.5fr);
      gap: 10px;
      align-items: end;
    }
    label { display: grid; gap: 4px; font-size: 11px; color: var(--muted); font-weight: 700; }
    input, select {
      width: 100%;
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 6px 9px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      font-size: 13px;
    }
    button {
      height: 34px;
      border: 1px solid #1f2520;
      border-radius: 6px;
      background: #1f2520;
      color: white;
      padding: 0 14px;
      font-weight: 800;
      cursor: pointer;
    }
    button.secondary { background: #fff; color: var(--ink); border-color: var(--line); }
    main { padding: 18px 24px 28px; display: grid; gap: 16px; }
    .grid { display: grid; grid-template-columns: 1.1fr 1fr; gap: 16px; align-items: start; }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .section-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: #fbfcf8;
    }
    h2 { margin: 0; font-size: 14px; }
    .meta { color: var(--muted); font-size: 12px; }
    .body { padding: 12px 14px; }
    .strategy-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    .strategy { border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
    .strategy-title { padding: 10px 12px; font-weight: 800; display: flex; justify-content: space-between; align-items: center; }
    .pills { display: flex; gap: 6px; flex-wrap: wrap; }
    .pill { font-size: 11px; padding: 3px 7px; border-radius: 999px; border: 1px solid var(--line); color: var(--muted); }
    .pill.on { color: var(--green); border-color: rgba(8,127,79,.35); background: rgba(8,127,79,.08); }
    .pill.off { color: var(--red); border-color: rgba(180,61,61,.35); background: rgba(180,61,61,.08); }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { padding: 8px 9px; border-bottom: 1px solid #edf0ec; text-align: left; white-space: nowrap; }
    th { color: var(--muted); font-size: 11px; font-weight: 800; background: #fbfcf8; }
    tr:last-child td { border-bottom: 0; }
    .num { text-align: right; font-variant-numeric: tabular-nums; }
    .buy { color: var(--green); font-weight: 800; }
    .watch { color: var(--amber); font-weight: 800; }
    .sell { color: var(--red); font-weight: 800; }
    .scroll { overflow: auto; max-height: 410px; }
    .holdings-editor { display: grid; gap: 8px; }
    .holding-row { display: grid; grid-template-columns: 1.1fr 1fr .7fr .8fr .8fr 34px; gap: 8px; }
    .trade-row { display: grid; grid-template-columns: 110px 90px 1fr 1fr 90px 90px 80px 1fr 90px; gap: 8px; align-items: end; }
    .account-summary { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }
    .metric { border: 1px solid var(--line); border-radius: 8px; padding: 9px 10px; background: #fbfcf8; }
    .metric span { display: block; color: var(--muted); font-size: 11px; font-weight: 800; }
    .metric strong { display: block; margin-top: 3px; font-size: 16px; font-variant-numeric: tabular-nums; }
    .editor-label { color: var(--muted); font-size: 11px; font-weight: 800; margin-top: 4px; }
    .positive { color: var(--green); }
    .negative { color: var(--red); }
    .signal { font-weight: 800; }
    .signal.hold { color: var(--green); }
    .signal.stop_loss, .signal.trailing_stop, .signal.max_holding_days { color: var(--red); }
    .signal.missing_price { color: var(--amber); }
    .status { color: var(--muted); min-height: 18px; font-size: 12px; }
    .error { color: var(--red); }
    .review-controls { display: grid; grid-template-columns: 150px 150px 90px auto; gap: 10px; align-items: end; }
    @media (max-width: 1120px) {
      header { grid-template-columns: 1fr; }
      .controls, .grid, .strategy-grid, .trade-row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>BaiQuant Desk</h1>
      <div class="sub" id="latestLine">loading</div>
    </div>
    <div class="controls">
      <label>SQLite<input id="dbPath"></label>
      <label>日期<input id="asOf" type="date"></label>
      <label>现金<input id="cash" type="number" step="1000"></label>
      <label>权益峰值<input id="equityPeak" type="number" step="1000"></label>
      <label>观测数<input id="limit" type="number" min="1" max="50" value="10"></label>
      <label>计划<select id="planMode"><option value="quick">快速</option><option value="precise">精确</option></select></label>
      <label>持仓文件<input id="holdingsPath"></label>
    </div>
    <button id="refreshBtn">刷新</button>
  </header>
  <main>
    <section>
      <div class="section-head">
        <h2>策略状态</h2>
        <div class="meta" id="statusLine"></div>
      </div>
      <div class="body strategy-grid" id="strategies"></div>
    </section>
    <div class="grid">
      <section>
        <div class="section-head">
          <h2>持仓同步</h2>
          <div>
            <button class="secondary" id="addHoldingBtn">新增</button>
            <button id="saveHoldingsBtn">保存</button>
          </div>
        </div>
        <div class="body holdings-editor">
          <div class="account-summary" id="accountSummary"></div>
          <div class="scroll"><table id="positionTable"></table></div>
          <div class="editor-label">成交记录</div>
          <div class="trade-row">
            <label>日期<input id="tradeDate" type="date"></label>
            <label>方向<select id="tradeAction"><option value="buy">买入</option><option value="sell">卖出</option></select></label>
            <label>代码<input id="tradeCode" placeholder="601636"></label>
            <label>名称<input id="tradeName" placeholder="示例A"></label>
            <label>股数<input id="tradeShares" type="number" min="1" step="1"></label>
            <label>价格<input id="tradePrice" type="number" min="0" step="0.001"></label>
            <label>费用<input id="tradeFees" type="number" min="0" step="0.01"></label>
            <label>备注<input id="tradeNote"></label>
            <button id="recordTradeBtn">记录</button>
          </div>
          <div class="status" id="tradeStatus"></div>
          <div class="editor-label">手动同步</div>
          <div id="holdingRows"></div>
          <div class="status" id="holdingStatus"></div>
        </div>
      </section>
      <section>
        <div class="section-head"><h2>操作计划</h2><div class="meta">按当前持仓计算</div></div>
        <div class="scroll"><table id="planTable"></table></div>
      </section>
    </div>
    <section>
      <div class="section-head"><h2>成交流水</h2><div class="meta">最近成交</div></div>
      <div class="scroll"><table id="tradeTable"></table></div>
    </section>
    <section>
      <div class="section-head">
        <h2>快速复盘</h2>
        <div class="review-controls">
          <label>策略<select id="reviewPreset"></select></label>
          <label>日期<input id="reviewDate" type="date"></label>
          <label>数量<input id="reviewLimit" type="number" min="1" max="50" value="10"></label>
          <button id="reviewBtn">复盘</button>
        </div>
      </div>
      <div class="scroll"><table id="reviewTable"></table></div>
    </section>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    let state = { holdings: [], strategies: [] };

    function fmtPct(v) {
      if (v === null || v === undefined || Number.isNaN(Number(v))) return "";
      return `${(Number(v) * 100).toFixed(2)}%`;
    }
    function fmtNum(v, digits = 2) {
      if (v === null || v === undefined || Number.isNaN(Number(v))) return "";
      return Number(v).toFixed(digits);
    }
    function query() {
      const params = new URLSearchParams();
      params.set("db", $("dbPath").value);
      params.set("as_of", $("asOf").value);
      params.set("holdings", $("holdingsPath").value);
      params.set("cash", $("cash").value);
      params.set("equity_peak", $("equityPeak").value);
      params.set("limit", $("limit").value || "10");
      params.set("plan_mode", $("planMode").value || "quick");
      params.set("presets", "20天稳打版,短线冲刺版,多因子手动版");
      return params;
    }
    async function fetchJson(url, options) {
      const res = await fetch(url, options);
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || res.statusText);
      return data;
    }
    async function loadDefaults() {
      const data = await fetchJson("/api/defaults");
      $("dbPath").value = data.db_path;
      $("holdingsPath").value = data.holdings_path;
      $("asOf").value = data.latest_date || "";
      $("reviewDate").value = data.latest_date || "";
      $("tradeDate").value = data.latest_date || "";
      $("reviewPreset").innerHTML = data.presets.map(p => `<option>${p}</option>`).join("");
      $("latestLine").textContent = `latest ${data.latest_date || "no data"}`;
      await refresh();
    }
    async function refresh() {
      $("statusLine").textContent = "loading";
      $("statusLine").className = "meta";
      try {
        const data = await fetchJson(`/api/dashboard?${query().toString()}`);
        state = data;
        $("latestLine").textContent = `latest ${data.latest_date || "no data"} · as-of ${data.as_of}`;
        $("statusLine").textContent = `${data.db_path} · ${data.plan_mode}`;
        renderStrategies(data.strategies);
        renderHoldings(data.holdings, data.positions, data.account);
        renderPlan(data.strategies);
        renderTrades(data.trades || []);
      } catch (err) {
        $("statusLine").textContent = err.message;
        $("statusLine").className = "meta error";
      }
    }
    function renderStrategies(strategies) {
      $("strategies").innerHTML = strategies.map(strategy => {
        const floor = strategy.breadth_floor.enabled
          ? `<span class="pill ${strategy.breadth_floor.open ? "on" : "off"}">floor ${strategy.breadth_floor.open ? "ON" : "OFF"}</span>`
          : `<span class="pill">floor NA</span>`;
        const regime = strategy.regime && strategy.regime.regime
          ? `<span class="pill">regime ${strategy.regime.regime}</span>`
          : "";
        return `<div class="strategy">
          <div class="strategy-title">
            <span>${strategy.preset}</span>
            <span class="pills">
              <span class="pill ${strategy.entry_gate ? "on" : "off"}">entry ${strategy.entry_gate ? "ON" : "OFF"}</span>
              <span class="pill ${strategy.market_gate ? "on" : "off"}">market ${strategy.market_gate ? "ON" : "OFF"}</span>
              ${floor}
              ${regime}
            </span>
          </div>
          ${table(strategy.watchlist, ["candidate_action","code","name","industry","close","candidate_rank","score_rank","multi_factor_score","factor_hits","positive_factors","tech_score","tech_grade","trade_advice","position_scale","risk_flags","score","hits"], true)}
        </div>`;
      }).join("");
    }
    function renderHoldings(rows, positions = state.positions || [], account = state.account || null) {
      state.holdings = rows.length ? rows : [];
      state.positions = positions || [];
      state.account = account;
      renderAccount(account);
      renderPositions(positions);
      $("holdingRows").innerHTML = state.holdings.map(holdingRow).join("");
    }
    function renderAccount(account) {
      if (!account) {
        $("accountSummary").innerHTML = "";
        return;
      }
      const metrics = [
        ["现金", fmtNum(account.cash)],
        ["持仓市值", fmtNum(account.market_value)],
        ["权益", fmtNum(account.equity)],
        ["浮盈", fmtNum(account.unrealized_pnl), account.unrealized_pnl],
        ["浮盈率", fmtPct(account.unrealized_return), account.unrealized_pnl],
        ["仓位", fmtPct(account.exposure)],
      ];
      $("accountSummary").innerHTML = metrics.map(([label, value, signed]) => {
        const cls = signed > 0 ? "positive" : signed < 0 ? "negative" : "";
        return `<div class="metric"><span>${label}</span><strong class="${cls}">${value || ""}</strong></div>`;
      }).join("");
    }
    function renderPositions(rows) {
      $("positionTable").innerHTML = tableMarkup(rows || [], [
        "code","name","industry","shares","average_cost","current_price","market_value",
        "unrealized_pnl","unrealized_return","drawdown_from_high","holding_days","stop_signal"
      ]);
    }
    function holdingRow(row = {}, index = 0) {
      return `<div class="holding-row" data-index="${index}">
        <input value="${row.code || ""}" placeholder="代码">
        <input value="${row.name || ""}" placeholder="名称">
        <input type="number" value="${row.shares || ""}" placeholder="股数">
        <input type="number" step="0.001" value="${row.average_cost || ""}" placeholder="成本">
        <input value="${row.entry_date || ""}" placeholder="买入日">
        <button class="secondary" onclick="removeHolding(${index})">×</button>
      </div>`;
    }
    function currentHoldings() {
      return [...document.querySelectorAll(".holding-row")].map(row => {
        const inputs = row.querySelectorAll("input");
        return {
          code: inputs[0].value.trim(),
          name: inputs[1].value.trim(),
          shares: Number(inputs[2].value || 0),
          average_cost: Number(inputs[3].value || 0),
          entry_date: inputs[4].value.trim(),
        };
      }).filter(row => row.code && row.shares > 0);
    }
    window.removeHolding = (index) => {
      state.holdings.splice(index, 1);
      renderHoldings(state.holdings);
    };
    function renderPlan(strategies) {
      const rows = strategies.flatMap(s => (s.plan || []).map(row => ({ strategy: s.preset, ...row })));
      $("planTable").innerHTML = tableMarkup(rows, [
        "strategy","action","code","name","reason","shares","reference_price",
        "average_cost","current_return","candidate_rank","multi_factor_score",
        "trade_advice","position_scale","risk_flags","cash_budget","drawdown"
      ]);
    }
    function renderTrades(rows) {
      $("tradeTable").innerHTML = tableMarkup(rows || [], [
        "date","time","action","code","name","shares","price","amount","fees","realized_pnl","holdings_shares_after","source","note"
      ]);
    }
    function table(rows, columns, compact = false) {
      return `<div class="scroll">${tableMarkup(rows, columns, compact)}</div>`;
    }
    function tableMarkup(rows, columns) {
      if (!rows || !rows.length) return `<table><tbody><tr><td class="meta">empty</td></tr></tbody></table>`;
      return `<table><thead><tr>${columns.map(c => `<th>${columnLabel(c)}</th>`).join("")}</tr></thead><tbody>${rows.map(row =>
        `<tr>${columns.map(col => cell(row, col)).join("")}</tr>`
      ).join("")}</tbody></table>`;
    }
    function columnLabel(col) {
      const labels = {
        strategy: "策略",
        candidate_action: "动作",
        action: "操作",
        date: "日期",
        time: "时间",
        code: "代码",
        name: "名称",
        industry: "行业",
        close: "收盘价",
        score_rank: "排名",
        candidate_rank: "候选排名",
        raw_rank: "原始排名",
        multi_factor_score: "多因子分",
        factor_hits: "命中因子",
        positive_factors: "正向因子",
        tech_score: "技术分",
        tech_grade: "档位",
        trade_advice: "买法",
        position_scale: "仓位建议",
        risk_flags: "风险提示",
        score: "原始分",
        hits: "命中因子",
        reason: "原因",
        shares: "股数",
        reference_price: "参考价",
        price: "成交价",
        amount: "成交额",
        fees: "费用",
        realized_pnl: "已实现盈亏",
        holdings_shares_after: "成交后持仓",
        cash_budget: "预算",
        drawdown: "回撤",
        average_cost: "成本",
        current_return: "浮盈率",
        current_price: "现价",
        market_value: "市值",
        unrealized_pnl: "浮盈",
        unrealized_return: "浮盈率",
        drawdown_from_high: "高点回撤",
        holding_days: "持有天数",
        stop_signal: "持仓信号",
        ret_1d: "1日后",
        ret_3d: "3日后",
        ret_5d: "5日后",
      };
      return labels[col] || col;
    }
    function cell(row, col) {
      const value = row[col];
      let text = value ?? "";
      let cls = "";
      const valueLabels = {
        action: {
          buy_next_open: "次日买入",
          sell_next_open: "次日卖出",
          wait: "观望",
          hold: "继续持有",
          manual_review: "手动复核",
        },
        candidate_action: {
          buy_candidate: "买入候选",
          watch: "观察",
        },
        reason: {
          entry: "入场",
          portfolio_stop: "账户止损",
          single_stop: "个股止损",
          take_profit: "止盈",
          trailing_stop: "回撤止盈",
          max_holding_days: "到期轮动",
          existing_position: "已有持仓",
          missing_price: "缺少价格",
        },
        stop_signal: {
          hold: "持有",
          stop_loss: "止损",
          trailing_stop: "回撤止盈",
          max_holding_days: "到期轮动",
          missing_price: "缺价格",
        },
      };
      if (valueLabels[col] && valueLabels[col][value]) text = valueLabels[col][value];
      if (col.includes("price") || col.includes("budget") || col === "score" || col === "multi_factor_score" || col === "close") { text = fmtNum(value); cls = "num"; }
      if (["amount","fees","realized_pnl"].includes(col)) { text = fmtNum(value); cls = "num"; }
      if (["average_cost","high_close","current_price","market_value","cost_value","unrealized_pnl"].includes(col)) { text = fmtNum(value); cls = "num"; }
      if (["current_return","unrealized_return","drawdown_from_high","exposure"].includes(col) || col.startsWith("ret_")) { text = fmtPct(value); cls = "num"; }
      if (["holding_days","shares","holdings_shares_after","candidate_rank","score_rank","raw_rank","factor_hits","hits"].includes(col)) { text = value ?? ""; cls = "num"; }
      if (col === "tech_score" || col === "position_scale") { text = fmtNum(value); cls = "num"; }
      if (col === "candidate_action") cls = value === "buy_candidate" ? "buy" : "watch";
      if (col === "action") cls = String(value).startsWith("sell") ? "sell" : String(value).startsWith("buy") ? "buy" : "watch";
      if (col === "unrealized_pnl" || col === "unrealized_return") cls += Number(value) > 0 ? " positive" : Number(value) < 0 ? " negative" : "";
      if (col === "stop_signal") cls = `signal ${String(value)}`;
      return `<td class="${cls}">${text}</td>`;
    }
    async function saveHoldings() {
      $("holdingStatus").textContent = "saving";
      try {
        const payload = { path: $("holdingsPath").value, holdings: currentHoldings() };
        const data = await fetchJson("/api/holdings", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        $("holdingStatus").textContent = `saved ${data.holdings.length}`;
        renderHoldings(data.holdings, [], null);
        await refresh();
      } catch (err) {
        $("holdingStatus").textContent = err.message;
        $("holdingStatus").className = "status error";
      }
    }
    async function recordTrade() {
      $("tradeStatus").className = "status";
      $("tradeStatus").textContent = "recording";
      try {
        const payload = {
          holdings_path: $("holdingsPath").value,
          date: $("tradeDate").value || $("asOf").value,
          action: $("tradeAction").value,
          code: $("tradeCode").value.trim(),
          name: $("tradeName").value.trim(),
          shares: Number($("tradeShares").value || 0),
          price: Number($("tradePrice").value || 0),
          fees: Number($("tradeFees").value || 0),
          source: "desk",
          note: $("tradeNote").value.trim(),
        };
        const data = await fetchJson("/api/trades", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        $("tradeStatus").textContent = `${data.trade.action} ${data.trade.code} ${data.trade.shares} @ ${fmtNum(data.trade.price)} recorded`;
        renderHoldings(data.holdings, [], null);
        await refresh();
      } catch (err) {
        $("tradeStatus").textContent = err.message;
        $("tradeStatus").className = "status error";
      }
    }
    async function review() {
      const params = new URLSearchParams();
      params.set("db", $("dbPath").value);
      params.set("as_of", $("reviewDate").value || $("asOf").value);
      params.set("preset", $("reviewPreset").value);
      params.set("limit", $("reviewLimit").value || "10");
      const data = await fetchJson(`/api/review?${params.toString()}`);
      $("reviewTable").innerHTML = tableMarkup(data.review, ["code","name","industry","close","ret_1d","ret_3d","ret_5d"]);
    }
    $("refreshBtn").addEventListener("click", refresh);
    $("saveHoldingsBtn").addEventListener("click", saveHoldings);
    $("recordTradeBtn").addEventListener("click", recordTrade);
    $("addHoldingBtn").addEventListener("click", () => {
      state.holdings.push({ code: "", name: "", shares: "", average_cost: "", entry_date: "" });
      renderHoldings(state.holdings);
    });
    $("reviewBtn").addEventListener("click", () => review().catch(err => {
      $("reviewTable").innerHTML = `<table><tbody><tr><td class="error">${err.message}</td></tr></tbody></table>`;
    }));
    loadDefaults().catch(err => {
      $("statusLine").textContent = err.message;
      $("statusLine").className = "meta error";
    });
  </script>
</body>
</html>"""
