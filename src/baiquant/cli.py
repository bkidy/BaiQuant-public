from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path

import pandas as pd

from baiquant.backtest import MultiPositionTrendConfig, run_multi_position_trend_backtest
from baiquant.config import load_pipeline_config
from baiquant.data.bundle import MarketDataBundle
from baiquant.data.baostock_provider import BaoStockIngestConfig, ingest_baostock
from baiquant.data.converters import convert_csv_to_sqlite
from baiquant.data.csv_provider import CsvDataProvider
from baiquant.data.efinance_provider import EFinanceMoneyFlowConfig, ingest_efinance_money_flow
from baiquant.data.sqlite_provider import SqliteDataProvider
from baiquant.data.tushare_provider import TushareIngestConfig, ingest_tushare
from baiquant.desk import DESK_MULTIFACTOR_PRESET, build_desk_payload
from baiquant.pipeline import run_selection
from baiquant.research.factor_diagnostics import (
    classify_market_regimes,
    compute_forward_returns,
    evaluate_ranked_signal_quality,
    summarize_signal_quality,
)
from baiquant.research.multifactor import (
    DEFAULT_MULTIFACTOR_WEIGHTS,
    build_multifactor_frame,
    derive_validated_multifactor_weights,
    evaluate_multifactor_factors,
    generate_multifactor_factor_history,
    generate_multifactor_signals,
    score_multifactor_frame,
    select_multifactor_candidates,
)
from baiquant.research.regime_router import build_regime_frame
from baiquant.research.live20k_optimizer import (
    default_live20k_execution_variants,
    evaluate_execution_variants,
)
from baiquant.live_ledger import record_live_trade
from baiquant.strategy.live20k import (
    Live20KSignalConfig,
    apply_live20k_paper_fills,
    assert_live20k_data_fresh,
    build_live20k_daily_plan,
    build_live20k_watchlist,
    build_market_regime,
    export_live20k_orders,
    generate_live20k_signals,
    live20k_entry_gate_open,
    live100k_hotspot_manual_fixed_execution_config,
    live100k_hotspot_manual_fixed_signal_config,
    live100k_hotspot_turbo_execution_config,
    live100k_hotspot_turbo_signal_config,
    live20k_execution_config,
    run_live20k_paper_replay,
    run_live20k_paper_step,
    summarize_live20k_paper_run,
)
from baiquant.strategy.multifactor import MultifactorPlanConfig, build_multifactor_daily_plan


LIVE20K_SIGNAL_PRICE_LOOKBACK_DAYS = 420
LIVE20K_SIGNAL_SIDE_LOOKBACK_DAYS = 10
MULTIFACTOR_SIGNAL_LOOKBACK_DAYS = 520
MULTIFACTOR_EVENT_LOOKBACK_DAYS = 30
MULTIFACTOR_DEFAULT_WEIGHTS = DEFAULT_MULTIFACTOR_WEIGHTS
LIVE20K_MANUAL_20D_PRESET = "20天稳打版"
LIVE20K_TURBO_SPRINT_PRESET = "短线冲刺版"
LIVE20K_PRESET_ALIASES = {
    "steady-20d": LIVE20K_MANUAL_20D_PRESET,
    "manual-20d": LIVE20K_MANUAL_20D_PRESET,
    "fixed-20d": LIVE20K_MANUAL_20D_PRESET,
    "turbo-sprint": LIVE20K_TURBO_SPRINT_PRESET,
    "turbo": LIVE20K_TURBO_SPRINT_PRESET,
    "手动20天版": LIVE20K_MANUAL_20D_PRESET,
    "20天加强版": LIVE20K_MANUAL_20D_PRESET,
    "收盘后20天主策略": LIVE20K_MANUAL_20D_PRESET,
    "Turbo激进版": LIVE20K_TURBO_SPRINT_PRESET,
    "短线Turbo冲刺版": LIVE20K_TURBO_SPRINT_PRESET,
    "Turbo": LIVE20K_TURBO_SPRINT_PRESET,
}
LIVE20K_PRESETS = [
    LIVE20K_MANUAL_20D_PRESET,
    LIVE20K_TURBO_SPRINT_PRESET,
]
LIVE20K_2026_PRESETS = [
    LIVE20K_MANUAL_20D_PRESET,
    LIVE20K_TURBO_SPRINT_PRESET,
]
LIVE20K_DISPLAY_LABELS = {
    "date": "日期",
    "strategy": "策略",
    "candidate_action": "动作",
    "action": "操作",
    "code": "代码",
    "name": "名称",
    "industry": "行业",
    "reason": "原因",
    "shares": "股数",
    "reference_price": "参考价",
    "close": "收盘价",
    "raw_rank": "原始排名",
    "score_rank": "排名",
    "score": "原始分",
    "hits": "命中因子",
    "tech_score": "技术分",
    "tech_grade": "档位",
    "trade_advice": "买法",
    "position_scale": "仓位建议",
    "risk_flags": "风险提示",
    "cash_budget": "预算",
    "market_gate": "大盘开关",
    "breadth_ma20": "大盘宽度",
    "dist_ma60": "离60日线",
}
LIVE20K_DISPLAY_VALUES = {
    "action": {
        "buy_next_open": "次日买入",
        "sell_next_open": "次日卖出",
        "wait": "观望",
    },
    "candidate_action": {
        "buy_candidate": "买入候选",
        "watch": "观察",
    },
    "reason": {
        "entry": "入场",
        "portfolio_stop": "账户止损",
        "single_stop": "个股止损",
        "trailing_stop": "回撤止盈",
        "max_holding_days": "到期轮动",
    },
}
MULTIFACTOR_DISPLAY_LABELS = {
    "action": "操作",
    "candidate_rank": "候选排名",
    "score_rank": "排名",
    "date": "日期",
    "code": "代码",
    "name": "名称",
    "industry": "行业",
    "reason": "原因",
    "shares": "股数",
    "reference_price": "参考价",
    "average_cost": "成本",
    "current_return": "浮盈亏",
    "close": "收盘价",
    "multi_factor_score": "多因子分",
    "factor_hits": "命中因子",
    "positive_factors": "正向因子",
    "cash_budget": "预算",
    "current_price": "现价",
    "market_value": "市值",
    "unrealized_return": "浮盈亏",
    "drawdown_from_high": "高点回撤",
    "stop_signal": "持仓信号",
    "momentum_5d": "5日动量",
    "momentum_20d": "20日动量",
    "reversal_5d": "5日反转",
    "money_flow_pct": "主力净流入%",
    "big_order_pct": "大单净流入%",
    "industry_momentum_3d": "行业3日动量",
}
MULTIFACTOR_DISPLAY_VALUES = {
    "action": {
        "buy_next_open": "次日买入",
        "sell_next_open": "次日卖出",
        "hold": "继续持有",
        "manual_review": "手动复核",
    },
    "reason": {
        "entry": "入场",
        "single_stop": "个股止损",
        "take_profit": "止盈",
        "trailing_stop": "回撤止盈",
        "max_holding_days": "到期轮动",
        "existing_position": "已有持仓",
        "missing_price": "缺少价格",
    },
}
MULTIFACTOR_DIAGNOSTIC_DISPLAY_LABELS = {
    "factor": "因子",
    "valid_factor_rows": "有效行数",
    "factor_coverage_rate": "覆盖率",
    "mean_rank_ic": "平均RankIC",
    "median_rank_ic": "中位RankIC",
    "positive_ic_rate": "IC胜率",
    "ic_observations": "样本期数",
    "top_1_5_mean_forward_return": "Top1-5均收益",
    "top_6_10_mean_forward_return": "Top6-10均收益",
    "top_11_20_mean_forward_return": "Top11-20均收益",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="baiquant")
    subparsers = parser.add_subparsers(dest="command", required=True)

    desk_parser = subparsers.add_parser("desk", help="Launch the local BaiQuant visual operations desk")
    desk_parser.add_argument("--db", default="data/tushare/baiquant.db", help="Default SQLite database path shown in the WebUI")
    desk_parser.add_argument("--holdings", default="data/live/holdings.csv", help="Default holdings CSV path shown in the WebUI")
    desk_parser.add_argument("--trade-log", default="data/live/trade_log.csv", help="Default live trade log CSV path")
    desk_parser.add_argument("--host", default="127.0.0.1", help="Local bind host")
    desk_parser.add_argument("--port", type=int, default=8765, help="Local WebUI port")
    desk_parser.add_argument("--no-open", action="store_true", help="Do not open the browser automatically")

    daily_plan_parser = subparsers.add_parser(
        "daily-plan",
        help="Generate a one-command live manual trading checklist",
    )
    daily_plan_parser.add_argument("--db", default="data/tushare/baiquant.db", help="SQLite database path")
    daily_plan_parser.add_argument("--as-of", default="", help="Signal date; defaults to latest market date")
    daily_plan_parser.add_argument("--holdings", default="data/live/holdings.csv", help="Current holdings CSV")
    daily_plan_parser.add_argument("--cash", type=float, default=0.0, help="Available cash")
    daily_plan_parser.add_argument("--equity-peak", type=float, help="Optional account equity peak for risk checks")
    daily_plan_parser.add_argument("--limit", type=int, default=8, help="Watchlist size")
    daily_plan_parser.add_argument("--output", default="data/live/daily_plan.md", help="Markdown report output path")
    daily_plan_parser.add_argument("--plan-csv", default="data/live/daily_plan.csv", help="Operation plan CSV output path")
    daily_plan_parser.add_argument(
        "--watchlist-csv",
        default="data/live/daily_watchlist.csv",
        help="Candidate watchlist CSV output path",
    )

    record_trade_parser = subparsers.add_parser(
        "record-trade",
        help="Record a real manual fill and update live holdings/trade log",
    )
    record_trade_parser.add_argument("--holdings", default="data/live/holdings.csv", help="Live holdings CSV")
    record_trade_parser.add_argument("--trade-log", default="data/live/trade_log.csv", help="Trade log CSV")
    record_trade_parser.add_argument("--date", required=True, help="Trade date, e.g. 2026-07-02")
    record_trade_parser.add_argument("--time", default="", help="Optional trade time, e.g. 09:35:48")
    record_trade_parser.add_argument("--action", choices=["buy", "sell"], required=True)
    record_trade_parser.add_argument("--code", required=True)
    record_trade_parser.add_argument("--name", default="")
    record_trade_parser.add_argument("--shares", type=int, required=True)
    record_trade_parser.add_argument("--price", type=float, required=True)
    record_trade_parser.add_argument("--fees", type=float, default=0.0)
    record_trade_parser.add_argument("--source", default="manual")
    record_trade_parser.add_argument("--note", default="")

    select_parser = subparsers.add_parser("select", help="Run one multi-factor selection date")
    select_parser.add_argument("--config", required=True, help="Path to a TOML pipeline config")
    select_parser.add_argument("--as-of", required=True, help="Signal date, e.g. 2026-05-25")
    select_parser.add_argument("--columns", default="code,name,industry,rank,weight,score,hits")

    multifactor_parser = subparsers.add_parser(
        "multifactor-select",
        help="Run the explainable multifactor candidate selector",
    )
    multifactor_parser.add_argument("--db", required=True, help="SQLite database path")
    multifactor_parser.add_argument("--as-of", required=True, help="Signal date, e.g. 2026-07-01")
    multifactor_parser.add_argument(
        "--lookback-days",
        type=int,
        default=MULTIFACTOR_SIGNAL_LOOKBACK_DAYS,
        help="Price, money-flow, and fundamentals lookback window",
    )
    multifactor_parser.add_argument(
        "--events-lookback-days",
        type=int,
        default=MULTIFACTOR_EVENT_LOOKBACK_DAYS,
        help="Event table lookback window",
    )
    multifactor_parser.add_argument("--top", type=int, default=20, help="Maximum candidates to print")
    multifactor_parser.add_argument(
        "--max-per-industry",
        type=int,
        default=3,
        help="Maximum candidates kept from the same industry; use 0 to disable",
    )
    multifactor_parser.add_argument("--max-price", type=float, help="Optional maximum stock price")
    multifactor_parser.add_argument(
        "--max-lot-cost",
        type=float,
        default=25_000,
        help="Maximum one-lot cost based on close * 100",
    )
    multifactor_parser.add_argument("--output", help="Optional path to write the full scored candidate CSV")
    multifactor_parser.add_argument(
        "--columns",
        default="candidate_rank,score_rank,code,name,industry,close,multi_factor_score,factor_hits,positive_factors",
    )
    _add_multifactor_weight_args(multifactor_parser)

    multifactor_plan_parser = subparsers.add_parser(
        "multifactor-plan",
        help="Build a live manual operation plan from multifactor candidates and current holdings",
    )
    multifactor_plan_parser.add_argument("--db", required=True, help="SQLite database path")
    multifactor_plan_parser.add_argument("--as-of", required=True, help="Signal date, e.g. 2026-07-01")
    multifactor_plan_parser.add_argument("--holdings-csv", default="data/live/holdings.csv", help="Current holdings CSV")
    multifactor_plan_parser.add_argument("--cash", type=float, default=0.0, help="Available cash for new buys")
    multifactor_plan_parser.add_argument("--lookback-days", type=int, default=MULTIFACTOR_SIGNAL_LOOKBACK_DAYS)
    multifactor_plan_parser.add_argument("--events-lookback-days", type=int, default=MULTIFACTOR_EVENT_LOOKBACK_DAYS)
    multifactor_plan_parser.add_argument("--top", type=int, default=8, help="Candidate pool size before plan sizing")
    multifactor_plan_parser.add_argument("--max-per-industry", type=int, default=3)
    multifactor_plan_parser.add_argument("--max-price", type=float)
    multifactor_plan_parser.add_argument("--max-lot-cost", type=float, default=25_000)
    multifactor_plan_parser.add_argument("--max-positions", type=int, default=1)
    multifactor_plan_parser.add_argument("--stop-loss-pct", type=float, default=0.04)
    multifactor_plan_parser.add_argument("--take-profit-pct", type=float, default=0.25)
    multifactor_plan_parser.add_argument("--trailing-stop-activation-pct", type=float, default=0.10)
    multifactor_plan_parser.add_argument("--trailing-stop-pct", type=float, default=0.06)
    multifactor_plan_parser.add_argument("--max-holding-days", type=int, default=15)
    multifactor_plan_parser.add_argument("--output", help="Optional path to write the daily operation plan CSV")
    multifactor_plan_parser.add_argument("--candidates-output", help="Optional path to write the scored candidate CSV")
    multifactor_plan_parser.add_argument(
        "--columns",
        default=(
            "action,code,name,industry,reason,shares,reference_price,average_cost,"
            "current_return,candidate_rank,score_rank,multi_factor_score,cash_budget"
        ),
    )
    _add_multifactor_weight_args(multifactor_plan_parser)
    _add_multifactor_regime_args(multifactor_plan_parser)

    multifactor_diagnostics_parser = subparsers.add_parser(
        "multifactor-diagnostics",
        help="Evaluate multifactor factor IC and forward-return slices over a historical window",
    )
    multifactor_diagnostics_parser.add_argument("--db", required=True, help="SQLite database path")
    multifactor_diagnostics_parser.add_argument("--start", required=True, help="First signal date, e.g. 2026-01-01")
    multifactor_diagnostics_parser.add_argument("--end", required=True, help="Last signal date, e.g. 2026-07-01")
    multifactor_diagnostics_parser.add_argument("--lookback-days", type=int, default=MULTIFACTOR_SIGNAL_LOOKBACK_DAYS)
    multifactor_diagnostics_parser.add_argument("--events-lookback-days", type=int, default=MULTIFACTOR_EVENT_LOOKBACK_DAYS)
    multifactor_diagnostics_parser.add_argument("--holding-days", type=int, default=5)
    multifactor_diagnostics_parser.add_argument("--signal-every-n-days", type=int, default=5)
    multifactor_diagnostics_parser.add_argument(
        "--factors",
        default=",".join(MULTIFACTOR_DEFAULT_WEIGHTS.keys()),
        help="Comma-separated factor names to evaluate",
    )
    multifactor_diagnostics_parser.add_argument("--output", help="Optional path to write the factor diagnostic CSV")
    multifactor_diagnostics_parser.add_argument("--factor-output", help="Optional path to write the sampled factor history CSV")
    multifactor_diagnostics_parser.add_argument(
        "--columns",
        default=(
            "factor,valid_factor_rows,factor_coverage_rate,mean_rank_ic,median_rank_ic,positive_ic_rate,ic_observations,"
            "top_1_5_mean_forward_return,top_6_10_mean_forward_return,top_11_20_mean_forward_return"
        ),
    )

    multifactor_backtest_parser = subparsers.add_parser(
        "multifactor-backtest",
        help="Backtest the explainable multifactor selector with A-share execution rules",
    )
    multifactor_backtest_parser.add_argument("--db", required=True, help="SQLite database path")
    multifactor_backtest_parser.add_argument("--start", required=True, help="First signal date, e.g. 2026-01-01")
    multifactor_backtest_parser.add_argument("--end", required=True, help="Last signal date, e.g. 2026-07-01")
    multifactor_backtest_parser.add_argument(
        "--lookback-days",
        type=int,
        default=MULTIFACTOR_SIGNAL_LOOKBACK_DAYS,
        help="Pre-start factor history window",
    )
    multifactor_backtest_parser.add_argument(
        "--events-lookback-days",
        type=int,
        default=MULTIFACTOR_EVENT_LOOKBACK_DAYS,
        help="Pre-start event history window",
    )
    multifactor_backtest_parser.add_argument("--cash", type=float, default=50_000, help="Initial capital")
    multifactor_backtest_parser.add_argument("--top", type=int, default=8, help="Signal candidates per rebalance date")
    multifactor_backtest_parser.add_argument(
        "--signal-every-n-days",
        type=int,
        default=20,
        help="Generate a fresh multifactor signal every N trading days",
    )
    multifactor_backtest_parser.add_argument("--max-positions", type=int, default=1, help="Maximum simultaneous holdings")
    multifactor_backtest_parser.add_argument("--max-per-industry", type=int, default=3, help="Maximum candidates from one industry")
    multifactor_backtest_parser.add_argument("--max-price", type=float, help="Optional maximum stock price")
    multifactor_backtest_parser.add_argument("--max-lot-cost", type=float, default=25_000, help="Maximum close * 100 one-lot cost")
    multifactor_backtest_parser.add_argument("--stop-loss-pct", type=float, default=0.04)
    multifactor_backtest_parser.add_argument("--take-profit-pct", type=float, default=0.25)
    multifactor_backtest_parser.add_argument("--add-trigger-pct", type=float, default=0.0)
    multifactor_backtest_parser.add_argument("--add-position-multiple", type=float, default=0.5)
    multifactor_backtest_parser.add_argument("--ma-window", type=int, default=8)
    multifactor_backtest_parser.add_argument("--trailing-stop-activation-pct", type=float, default=0.10)
    multifactor_backtest_parser.add_argument("--trailing-stop-pct", type=float, default=0.06)
    multifactor_backtest_parser.add_argument("--max-holding-days", type=int, default=15)
    multifactor_backtest_parser.add_argument("--portfolio-stop-drawdown-pct", type=float, default=0.04)
    multifactor_backtest_parser.add_argument("--portfolio-stop-cooldown-days", type=int, default=8)
    multifactor_backtest_parser.add_argument(
        "--liquidate-on-portfolio-stop",
        dest="liquidate_on_portfolio_stop",
        action="store_true",
        default=True,
    )
    multifactor_backtest_parser.add_argument(
        "--keep-positions-on-portfolio-stop",
        dest="liquidate_on_portfolio_stop",
        action="store_false",
    )
    multifactor_backtest_parser.add_argument("--summary-output", help="Optional path to write summary CSV")
    multifactor_backtest_parser.add_argument("--signals-output", help="Optional path to write generated signals CSV")
    multifactor_backtest_parser.add_argument("--trades-output", help="Optional path to write trades CSV")
    multifactor_backtest_parser.add_argument("--equity-output", help="Optional path to write equity curve CSV")
    _add_multifactor_weight_args(multifactor_backtest_parser)
    _add_multifactor_regime_args(multifactor_backtest_parser)

    multifactor_validate_parser = subparsers.add_parser(
        "multifactor-validate-periods",
        help="Run the multifactor backtest over multiple named periods with one consistent configuration",
    )
    multifactor_validate_parser.add_argument("--db", required=True, help="SQLite database path")
    multifactor_validate_parser.add_argument(
        "--periods",
        required=True,
        help="Comma-separated label:start:end specs, e.g. 2024:2024-01-01:2024-12-31",
    )
    multifactor_validate_parser.add_argument("--lookback-days", type=int, default=MULTIFACTOR_SIGNAL_LOOKBACK_DAYS)
    multifactor_validate_parser.add_argument("--events-lookback-days", type=int, default=MULTIFACTOR_EVENT_LOOKBACK_DAYS)
    multifactor_validate_parser.add_argument("--cash", type=float, default=50_000)
    multifactor_validate_parser.add_argument("--top", type=int, default=8)
    multifactor_validate_parser.add_argument("--signal-every-n-days", type=int, default=20)
    multifactor_validate_parser.add_argument("--max-positions", type=int, default=1)
    multifactor_validate_parser.add_argument("--max-per-industry", type=int, default=3)
    multifactor_validate_parser.add_argument("--max-price", type=float)
    multifactor_validate_parser.add_argument("--max-lot-cost", type=float, default=25_000)
    multifactor_validate_parser.add_argument("--stop-loss-pct", type=float, default=0.04)
    multifactor_validate_parser.add_argument("--take-profit-pct", type=float, default=0.25)
    multifactor_validate_parser.add_argument("--add-trigger-pct", type=float, default=0.0)
    multifactor_validate_parser.add_argument("--add-position-multiple", type=float, default=0.5)
    multifactor_validate_parser.add_argument("--ma-window", type=int, default=8)
    multifactor_validate_parser.add_argument("--trailing-stop-activation-pct", type=float, default=0.10)
    multifactor_validate_parser.add_argument("--trailing-stop-pct", type=float, default=0.06)
    multifactor_validate_parser.add_argument("--max-holding-days", type=int, default=15)
    multifactor_validate_parser.add_argument("--portfolio-stop-drawdown-pct", type=float, default=0.04)
    multifactor_validate_parser.add_argument("--portfolio-stop-cooldown-days", type=int, default=8)
    multifactor_validate_parser.add_argument(
        "--liquidate-on-portfolio-stop",
        dest="liquidate_on_portfolio_stop",
        action="store_true",
        default=True,
    )
    multifactor_validate_parser.add_argument(
        "--keep-positions-on-portfolio-stop",
        dest="liquidate_on_portfolio_stop",
        action="store_false",
    )
    multifactor_validate_parser.add_argument("--summary-output", help="Optional path to write period summary CSV")
    _add_multifactor_weight_args(multifactor_validate_parser)
    _add_multifactor_regime_args(multifactor_validate_parser)

    live20k_parser = subparsers.add_parser("live20k", help="Run the 20K small-account live signal gate")
    live20k_parser.add_argument("--db", required=True, help="SQLite database path")
    live20k_parser.add_argument("--as-of", required=True, help="Signal date, e.g. 2026-05-25")
    live20k_parser.add_argument("--columns", default="action,code,name,reason,shares,reference_price,score_rank,cash_budget")
    live20k_parser.add_argument("--holdings-csv", help="Optional current holdings CSV with code,shares,average_cost")
    live20k_parser.add_argument("--cash", type=float, default=20_000, help="Current available cash for plan sizing")
    live20k_parser.add_argument("--equity-peak", type=float, help="Current paper/live high-water equity for portfolio stop")
    live20k_parser.add_argument("--plan-output", help="Optional path to write the daily plan CSV")

    live20k_2026_parser = subparsers.add_parser(
        "live20k-2026",
        help="Run the steady-20d or turbo-sprint preset",
    )
    live20k_2026_parser.add_argument("--db", required=True, help="SQLite database path")
    live20k_2026_parser.add_argument("--as-of", required=True, help="Signal date, e.g. 2026-05-25")
    live20k_2026_parser.add_argument(
        "--columns",
        default=(
            "action,code,name,industry,reason,shares,reference_price,raw_rank,score_rank,"
            "tech_score,tech_grade,trade_advice,position_scale,risk_flags,cash_budget"
        ),
    )
    live20k_2026_parser.add_argument("--holdings-csv", help="Optional current holdings CSV with code,shares,average_cost")
    live20k_2026_parser.add_argument("--cash", type=float, help="Current available cash for plan sizing; defaults to preset initial cash")
    live20k_2026_parser.add_argument("--equity-peak", type=float, help="Current paper/live high-water equity for portfolio stop")
    live20k_2026_parser.add_argument("--plan-output", help="Optional path to write the daily plan CSV")
    live20k_2026_parser.add_argument(
        "--preset",
        default=LIVE20K_MANUAL_20D_PRESET,
        help="2026 execution preset for the daily plan",
    )
    live20k_2026_parser.add_argument("--watchlist", action="store_true", help="Print turbo candidates even when the entry gate blocks buys")
    live20k_2026_parser.add_argument("--watchlist-limit", type=int, default=10, help="Maximum watchlist candidates to print")
    live20k_2026_parser.add_argument("--watchlist-output", help="Optional path to write the watchlist CSV")

    live20k_fill_parser = subparsers.add_parser("live20k-fill", help="Apply a 20K paper plan to next-open fills")
    live20k_fill_parser.add_argument("--db", required=True, help="SQLite database path")
    live20k_fill_parser.add_argument("--plan-csv", required=True, help="Daily plan CSV written by live20k")
    live20k_fill_parser.add_argument("--execution-date", help="Optional execution date; defaults to next trading date")
    live20k_fill_parser.add_argument("--holdings-csv", help="Optional current holdings CSV with code,shares,average_cost")
    live20k_fill_parser.add_argument("--cash", type=float, default=20_000, help="Cash before applying fills")
    live20k_fill_parser.add_argument("--equity-peak", type=float, help="High-water equity before applying fills")
    live20k_fill_parser.add_argument("--holdings-output", help="Path to write updated holdings CSV")
    live20k_fill_parser.add_argument("--fills-output", help="Path to write fill log CSV")
    live20k_fill_parser.add_argument("--state-output", help="Path to write one-row cash/equity state CSV")

    live20k_report_parser = subparsers.add_parser("live20k-report", help="Summarize the 20K paper-run promotion gate")
    live20k_report_parser.add_argument("--plans-glob", required=True, help="Glob for daily plan CSV files")
    live20k_report_parser.add_argument("--fills-glob", help="Glob for daily fills CSV files")
    live20k_report_parser.add_argument("--states-glob", help="Glob for paper state CSV files")
    live20k_report_parser.add_argument("--min-days", type=int, default=20, help="Required paper-run trading days")
    live20k_report_parser.add_argument("--min-order-days", type=int, default=3, help="Required paper-run order days")
    live20k_report_parser.add_argument("--min-total-return", type=float, default=0.0, help="Required paper-run total return")
    live20k_report_parser.add_argument("--output", help="Optional path to write the report CSV")

    live20k_orders_parser = subparsers.add_parser("live20k-orders", help="Export live order CSV after paper gate passes")
    live20k_orders_parser.add_argument("--plan-csv", required=True, help="Daily plan CSV written by live20k")
    live20k_orders_parser.add_argument("--report-csv", required=True, help="Paper-run report CSV written by live20k-report")
    live20k_orders_parser.add_argument("--output", required=True, help="Path to write broker-review order CSV")

    live20k_step_parser = subparsers.add_parser("live20k-step", help="Run one full 20K paper-ledger step")
    live20k_step_parser.add_argument("--db", required=True, help="SQLite database path")
    live20k_step_parser.add_argument("--as-of", required=True, help="Trade date to process, e.g. 2026-05-25")
    live20k_step_parser.add_argument("--paper-dir", default="data/paper", help="Directory for paper ledger CSV files")
    live20k_step_parser.add_argument("--min-days", type=int, default=20, help="Required paper-run trading days")
    live20k_step_parser.add_argument("--min-order-days", type=int, default=3, help="Required paper-run order days")
    live20k_step_parser.add_argument("--min-total-return", type=float, default=0.0, help="Required paper-run total return")
    live20k_step_parser.add_argument(
        "--preset",
        default=LIVE20K_MANUAL_20D_PRESET,
        help="Signal preset for the generated daily plan",
    )

    live20k_replay_parser = subparsers.add_parser(
        "live20k-replay",
        help="Replay the 20K paper-ledger gate over a historical date range",
    )
    live20k_replay_parser.add_argument("--db", required=True, help="SQLite database path")
    live20k_replay_parser.add_argument("--start", required=True, help="First replay date, e.g. 2026-04-27")
    live20k_replay_parser.add_argument("--end", required=True, help="Last replay date, e.g. 2026-05-26")
    live20k_replay_parser.add_argument("--paper-dir", default="data/paper_replay", help="Directory for replay CSV files")
    live20k_replay_parser.add_argument("--min-days", type=int, default=20, help="Required replay trading days")
    live20k_replay_parser.add_argument("--min-order-days", type=int, default=3, help="Required replay order days")
    live20k_replay_parser.add_argument("--min-total-return", type=float, default=0.0, help="Required replay total return")
    live20k_replay_parser.add_argument(
        "--preset",
        default=LIVE20K_MANUAL_20D_PRESET,
        help="Signal preset for the generated daily plans",
    )

    live20k_optimize_parser = subparsers.add_parser(
        "live20k-optimize",
        help="Rank 20K execution variants for higher-return strategy tuning",
    )
    live20k_optimize_parser.add_argument("--db", required=True, help="SQLite database path")
    live20k_optimize_parser.add_argument("--start", required=True, help="First optimization date")
    live20k_optimize_parser.add_argument("--end", required=True, help="Last optimization date")
    live20k_optimize_parser.add_argument("--recent-start", help="First recent-window date; defaults to start")
    live20k_optimize_parser.add_argument("--output", help="Optional path to write the leaderboard CSV")
    live20k_optimize_parser.add_argument(
        "--columns",
        default=(
            "name,score,ytd_return,ytd_mdd,recent_return,recent_mdd,"
            "recent_exposure,filled_trades,max_positions,ma_window,"
            "take_profit_pct,portfolio_stop_drawdown_pct,"
            "trailing_stop_activation_pct,trailing_stop_pct"
        ),
    )

    live20k_quality_parser = subparsers.add_parser(
        "live20k-quality",
        help="Measure post-signal gain stability by rank bucket",
    )
    live20k_quality_parser.add_argument("--db", required=True, help="SQLite database path")
    live20k_quality_parser.add_argument("--start", help="First signal date to evaluate")
    live20k_quality_parser.add_argument("--end", help="Last signal date to evaluate")
    live20k_quality_parser.add_argument("--horizons", default="5,10,20", help="Comma-separated holding days")
    live20k_quality_parser.add_argument("--rank-start", type=int, default=1, help="First raw rank to include")
    live20k_quality_parser.add_argument("--signal-limit", type=int, default=20, help="Signals per day to evaluate")
    live20k_quality_parser.add_argument("--stable-drawdown-floor", type=float, default=-0.08)
    live20k_quality_parser.add_argument("--min-gain-retention", type=float, default=0.5)
    live20k_quality_parser.add_argument("--apply-market-gate", action="store_true", help="Evaluate only market-gate-on dates")
    live20k_quality_parser.add_argument("--detail-output", help="Optional path to write per-signal quality CSV")
    live20k_quality_parser.add_argument("--summary-output", help="Optional path to write rank-bucket summary CSV")
    live20k_quality_parser.add_argument(
        "--columns",
        default=(
            "horizon_days,rank_bucket,count,positive_rate,stable_rate,"
            "mean_forward_return,median_forward_return,mean_max_drawdown,mean_gain_retention"
        ),
    )

    live20k_context_parser = subparsers.add_parser(
        "live20k-context",
        help="Diagnose the current market regime and matching signal quality",
    )
    live20k_context_parser.add_argument("--db", required=True, help="SQLite database path")
    live20k_context_parser.add_argument("--as-of", required=True, help="Context date, e.g. 2026-05-25")
    live20k_context_parser.add_argument("--lookback-days", type=int, default=252, help="Same-regime dates to evaluate")
    live20k_context_parser.add_argument("--horizons", default="5,10,20", help="Comma-separated holding days")
    live20k_context_parser.add_argument("--signal-limit", type=int, default=20, help="Raw-ranked signals per day")
    live20k_context_parser.add_argument("--candidate-rank-start", type=int, default=6)
    live20k_context_parser.add_argument("--candidate-rank-end", type=int, default=10)
    live20k_context_parser.add_argument("--summary-output", help="Optional path to write same-regime summary CSV")
    live20k_context_parser.add_argument("--detail-output", help="Optional path to write same-regime detail CSV")
    live20k_context_parser.add_argument("--candidates-output", help="Optional path to write current candidates CSV")

    ingest_parser = subparsers.add_parser("ingest", help="Fetch and normalize market data")
    ingest_subparsers = ingest_parser.add_subparsers(dest="source", required=True)

    baostock_parser = ingest_subparsers.add_parser("baostock", help="Enrich SQLite data from BaoStock")
    baostock_parser.add_argument("--output", required=True, help="SQLite database path")
    baostock_parser.add_argument("--symbols", help="Comma-separated symbols, e.g. 000001.SZ,600000.SH")
    baostock_parser.add_argument("--year", type=int, required=True, help="Financial report year")
    baostock_parser.add_argument("--quarter", type=int, required=True, choices=[1, 2, 3, 4])
    baostock_parser.add_argument("--limit", type=int, help="Limit symbol count for batch runs")
    baostock_parser.add_argument("--offset", type=int, default=0, help="Skip this many symbols before applying --limit")
    baostock_parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between symbols")
    baostock_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first symbol failure instead of writing partial data",
    )

    efinance_parser = ingest_subparsers.add_parser("efinance", help="Fetch money-flow data from efinance")
    efinance_parser.add_argument("--output", required=True, help="SQLite database path")
    efinance_parser.add_argument("--symbols", help="Comma-separated symbols, e.g. 000001.SZ,600000.SH")
    efinance_parser.add_argument("--start", help="Optional start date as YYYYMMDD or YYYY-MM-DD")
    efinance_parser.add_argument("--end", help="Optional end date as YYYYMMDD or YYYY-MM-DD")
    efinance_parser.add_argument("--limit", type=int, help="Limit symbol count for batch runs")
    efinance_parser.add_argument("--offset", type=int, default=0, help="Skip this many symbols before applying --limit")
    efinance_parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between symbols")
    efinance_parser.add_argument("--retries", type=int, default=1, help="Retries per symbol")
    efinance_parser.add_argument("--timeout", type=float, default=10.0, help="Timeout seconds per symbol")
    efinance_parser.add_argument("--flush-every", type=int, default=50, help="Persist every N successful symbols")
    efinance_parser.add_argument("--progress-every", type=int, default=50, help="Print progress every N symbols")
    efinance_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first symbol failure instead of writing partial data",
    )

    tushare_parser = ingest_subparsers.add_parser("tushare", help="Fetch A-share data from Tushare")
    tushare_parser.add_argument("--output", required=True, help="SQLite database path")
    tushare_parser.add_argument("--start", required=True, help="Start date as YYYYMMDD")
    tushare_parser.add_argument("--end", required=True, help="End date as YYYYMMDD")
    tushare_parser.add_argument("--token-path", default=".secrets/tushare_token", help="Path to local Tushare token")
    tushare_parser.add_argument("--adjust", default="qfq", choices=["none", "qfq"])
    tushare_parser.add_argument("--write-mode", default="replace", choices=["replace", "append"])
    tushare_parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between trade dates")
    tushare_parser.add_argument("--flush-every", type=int, default=20, help="Persist every N fetched trade dates")
    tushare_parser.add_argument("--progress-every", type=int, default=20, help="Print progress every N trade dates")
    tushare_parser.add_argument("--workers", type=int, default=1, help="Parallel trade-date fetch workers")
    tushare_parser.add_argument(
        "--rate-limit-per-minute",
        type=int,
        default=0,
        help="Global Tushare API request cap across workers; 0 disables the cap",
    )
    tushare_parser.add_argument("--timeout", type=int, default=30, help="Tushare HTTP timeout seconds")
    tushare_parser.add_argument("--retries", type=int, default=0, help="Retries per Tushare API request")
    tushare_parser.add_argument("--retry-sleep", type=float, default=1.0, help="Base seconds to sleep between retries")
    tushare_parser.add_argument("--resume", action="store_true", help="Skip trade dates already present in the target table")
    tushare_mode = tushare_parser.add_mutually_exclusive_group()
    tushare_mode.add_argument("--only-prices", action="store_true", help="Fetch daily prices only; skip fundamentals, limits, suspends, and moneyflow")
    tushare_mode.add_argument("--only-core", action="store_true", help="Fetch prices and fundamentals only; skip moneyflow")
    tushare_mode.add_argument(
        "--only-money-flow",
        action="store_true",
        help="Fetch only moneyflow and enrich it from existing prices",
    )
    tushare_parser.add_argument("--skip-daily-basic", action="store_true", help="Do not fetch daily_basic")
    tushare_parser.add_argument("--skip-money-flow", action="store_true", help="Do not fetch moneyflow")
    tushare_parser.add_argument("--skip-limits", action="store_true", help="Do not fetch stk_limit")
    tushare_parser.add_argument("--skip-suspends", action="store_true", help="Do not fetch suspend_d")
    tushare_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first trade-date failure instead of writing partial data",
    )

    convert_parser = subparsers.add_parser("convert", help="Convert between local storage formats")
    convert_subparsers = convert_parser.add_subparsers(dest="conversion", required=True)
    csv_sqlite_parser = convert_subparsers.add_parser("csv-to-sqlite", help="Convert normalized CSV tables to SQLite")
    csv_sqlite_parser.add_argument("--input", required=True, help="Input directory containing normalized CSV files")
    csv_sqlite_parser.add_argument("--output", required=True, help="Output SQLite database path")
    return parser


def _add_multifactor_weight_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--weights-profile",
        default="default",
        choices=["default", "validated"],
        help="Use default hand-tuned weights or diagnostics-validated weights",
    )
    parser.add_argument("--diagnostics-csv", help="Factor diagnostics CSV for --weights-profile validated")
    parser.add_argument("--min-factor-coverage", type=float, default=0.8)
    parser.add_argument("--min-factor-observations", type=int, default=3)


def _add_multifactor_regime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--regime-gate", action="store_true", help="Only generate multifactor signals in allowed regimes")
    parser.add_argument(
        "--allowed-regimes",
        default="bull,broad_rebound,structural,weak_range",
        help="Comma-separated regimes allowed when --regime-gate is enabled",
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "desk":
        from baiquant.desk import run_desk

        run_desk(
            host=args.host,
            port=args.port,
            db_path=args.db,
            holdings_path=args.holdings,
            trade_log_path=args.trade_log,
            open_browser=not args.no_open,
        )
    elif args.command == "daily-plan":
        payload = build_desk_payload(
            db_path=args.db,
            as_of=args.as_of or None,
            holdings_path=args.holdings,
            presets=(DESK_MULTIFACTOR_PRESET,),
            cash=args.cash,
            equity_peak=args.equity_peak,
            watchlist_limit=args.limit,
            plan_mode="quick",
        )
        strategy = _daily_plan_strategy(payload)
        plan = pd.DataFrame(strategy.get("plan", []))
        watchlist = pd.DataFrame(strategy.get("watchlist", []))
        if args.plan_csv:
            _write_csv(plan, args.plan_csv)
        if args.watchlist_csv:
            _write_csv(watchlist, args.watchlist_csv)
        report = _format_daily_plan_markdown(payload, strategy)
        if args.output:
            _write_text(report, args.output)
        print(report)
    elif args.command == "record-trade":
        _, trade = record_live_trade(
            holdings_path=args.holdings,
            trade_log_path=args.trade_log,
            trade_date=args.date,
            trade_time=args.time,
            action=args.action,
            code=args.code,
            name=args.name,
            shares=args.shares,
            price=args.price,
            fees=args.fees,
            source=args.source,
            note=args.note,
        )
        print(
            f"recorded {trade['action']} {trade['code']} {trade['shares']} @ {trade['price']:.2f}; "
            f"after={trade['holdings_shares_after']}; realized_pnl={trade['realized_pnl']:.2f}"
        )
    elif args.command == "select":
        data_config, config = load_pipeline_config(args.config)
        if data_config.kind == "sqlite":
            bundle = SqliteDataProvider(data_config.path).load()
        elif data_config.kind == "csv":
            bundle = CsvDataProvider(data_config.path).load()
        else:
            raise ValueError(f"Unsupported data kind: {data_config.kind}")
        result = run_selection(bundle, as_of=args.as_of, config=config)
        columns = [column.strip() for column in args.columns.split(",") if column.strip()]
        available = [column for column in columns if column in result.selected.columns]
        print(result.selected[available].to_string(index=False))
    elif args.command == "multifactor-select":
        as_of = pd.Timestamp(args.as_of)
        bundle = SqliteDataProvider(args.db).load_window(
            as_of=as_of,
            lookback_days=args.lookback_days,
            fundamentals_lookback_days=args.lookback_days,
            events_lookback_days=args.events_lookback_days,
            money_flow_lookback_days=args.lookback_days,
        )
        weights = _resolve_multifactor_weights(args)
        factors = build_multifactor_frame(bundle, as_of)
        scored = score_multifactor_frame(factors, weights, top_n=None)
        candidates = select_multifactor_candidates(
            scored,
            top_n=args.top,
            max_per_industry=args.max_per_industry,
            max_price=args.max_price,
            max_lot_cost=args.max_lot_cost,
        )
        if args.output:
            _write_csv(candidates, args.output)
        if candidates.empty:
            print("No multifactor candidates.")
        else:
            columns = [column.strip() for column in args.columns.split(",") if column.strip()]
            available = [column for column in columns if column in candidates.columns]
            print(_format_multifactor_display(candidates[available]))
    elif args.command == "multifactor-plan":
        as_of = pd.Timestamp(args.as_of)
        bundle = SqliteDataProvider(args.db).load_window(
            as_of=as_of,
            lookback_days=args.lookback_days,
            fundamentals_lookback_days=args.lookback_days,
            events_lookback_days=args.events_lookback_days,
            money_flow_lookback_days=args.lookback_days,
        )
        weights = _resolve_multifactor_weights(args)
        factors = build_multifactor_frame(bundle, as_of)
        scored = score_multifactor_frame(factors, weights, top_n=None)
        candidates = select_multifactor_candidates(
            scored,
            top_n=args.top,
            max_per_industry=args.max_per_industry,
            max_price=args.max_price,
            max_lot_cost=args.max_lot_cost,
        )
        if not _multifactor_regime_allows_date(bundle, args, as_of):
            candidates = candidates.head(0).copy()
        holdings = pd.read_csv(args.holdings_csv) if args.holdings_csv and Path(args.holdings_csv).exists() else pd.DataFrame()
        plan = build_multifactor_daily_plan(
            candidates,
            bundle.prices,
            as_of=as_of,
            holdings=holdings,
            cash=args.cash,
            config=MultifactorPlanConfig(
                max_positions=args.max_positions,
                stop_loss_pct=args.stop_loss_pct,
                take_profit_pct=args.take_profit_pct,
                trailing_stop_activation_pct=args.trailing_stop_activation_pct,
                trailing_stop_pct=args.trailing_stop_pct,
                max_holding_days=args.max_holding_days,
            ),
        )
        if args.candidates_output:
            _write_csv(candidates, args.candidates_output)
        if args.output:
            _write_csv(plan, args.output)
        if plan.empty:
            print("No multifactor plan actions.")
        else:
            columns = [column.strip() for column in args.columns.split(",") if column.strip()]
            available = [column for column in columns if column in plan.columns]
            print(_format_multifactor_display(plan[available]))
    elif args.command == "multifactor-diagnostics":
        start = pd.Timestamp(args.start)
        end = pd.Timestamp(args.end)
        span_days = max(0, int((end.normalize() - start.normalize()).days))
        factor_names = [factor.strip() for factor in args.factors.split(",") if factor.strip()]
        bundle = SqliteDataProvider(args.db).load_window(
            as_of=end,
            lookback_days=args.lookback_days + span_days,
            fundamentals_lookback_days=args.lookback_days + span_days,
            events_lookback_days=args.events_lookback_days + span_days,
            money_flow_lookback_days=args.lookback_days + span_days,
        )
        factors = generate_multifactor_factor_history(
            bundle,
            start=start,
            end=end,
            signal_every_n_days=args.signal_every_n_days,
            factor_names=factor_names,
        )
        signal_dates = pd.to_datetime(factors["date"]).drop_duplicates().sort_values().tolist() if not factors.empty else []
        forward_returns = compute_forward_returns(bundle.prices, signal_dates, args.holding_days)
        diagnostics = evaluate_multifactor_factors(
            factors,
            forward_returns,
            factor_names=factor_names,
        )
        if args.factor_output:
            _write_csv(factors, args.factor_output)
        if args.output:
            _write_csv(diagnostics, args.output)
        if diagnostics.empty:
            print("No multifactor diagnostics.")
        else:
            rank_ic = diagnostics["mean_rank_ic"].dropna()
            mean_rank_ic = float(rank_ic.mean()) if not rank_ic.empty else 0.0
            print(
                "multifactor_diagnostics "
                f"start={start.date().isoformat()} end={end.date().isoformat()} "
                f"factor_rows={len(factors)} factors={len(diagnostics)} "
                f"rank_ic_mean={mean_rank_ic:.4f}"
            )
            columns = [column.strip() for column in args.columns.split(",") if column.strip()]
            available = [column for column in columns if column in diagnostics.columns]
            print(_format_multifactor_diagnostics_display(diagnostics[available]))
    elif args.command == "multifactor-backtest":
        start = pd.Timestamp(args.start)
        end = pd.Timestamp(args.end)
        span_days = max(0, int((end.normalize() - start.normalize()).days))
        bundle = SqliteDataProvider(args.db).load_window(
            as_of=end,
            lookback_days=args.lookback_days + span_days,
            fundamentals_lookback_days=args.lookback_days + span_days,
            events_lookback_days=args.events_lookback_days + span_days,
            money_flow_lookback_days=args.lookback_days + span_days,
        )
        weights = _resolve_multifactor_weights(args)
        regimes = _resolve_multifactor_regimes(bundle, args)
        result, signals, summary = _run_multifactor_backtest_period(bundle, args, start, end, weights, regimes=regimes)
        if args.summary_output:
            _write_csv(summary, args.summary_output)
        if args.signals_output:
            _write_csv(signals, args.signals_output)
        if args.trades_output:
            _write_csv(result.trades, args.trades_output)
        if args.equity_output:
            _write_csv(result.equity_curve, args.equity_output)
        row = summary.iloc[0]
        print(
            "multifactor_backtest "
            f"start={start.date()} "
            f"end={end.date()} "
            f"signals={int(row['signals'])} "
            f"filled_trades={int(row['filled_trades'])} "
            f"total_return={float(row['total_return']):.2%} "
            f"max_drawdown={float(row['max_drawdown']):.2%} "
            f"sharpe={float(row['sharpe']):.2f}"
        )
    elif args.command == "multifactor-validate-periods":
        periods = _parse_period_specs(args.periods)
        first_start = min(start for _, start, _ in periods)
        last_end = max(end for _, _, end in periods)
        span_days = max(0, int((last_end.normalize() - first_start.normalize()).days))
        bundle = SqliteDataProvider(args.db).load_window(
            as_of=last_end,
            lookback_days=args.lookback_days + span_days,
            fundamentals_lookback_days=args.lookback_days + span_days,
            events_lookback_days=args.events_lookback_days + span_days,
            money_flow_lookback_days=args.lookback_days + span_days,
        )
        weights = _resolve_multifactor_weights(args)
        regimes = _resolve_multifactor_regimes(bundle, args)
        summaries: list[pd.DataFrame] = []
        for label, start, end in periods:
            _, _, summary = _run_multifactor_backtest_period(bundle, args, start, end, weights, period=label, regimes=regimes)
            summaries.append(summary)
        result_summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
        if args.summary_output:
            _write_csv(result_summary, args.summary_output)
        print(
            "multifactor_validate_periods "
            f"periods={len(result_summary)} "
            f"summary_output={args.summary_output or ''}"
        )
        if not result_summary.empty:
            columns = [
                "period",
                "start",
                "end",
                "total_return",
                "max_drawdown",
                "sharpe",
                "filled_trades",
                "signals",
            ]
            print(result_summary[[column for column in columns if column in result_summary.columns]].to_string(index=False))
    elif args.command in {"live20k", "live20k-2026"}:
        as_of = pd.Timestamp(args.as_of)
        bundle = _load_live20k_signal_bundle(args.db, as_of)
        try:
            assert_live20k_data_fresh(bundle, as_of)
        except ValueError as error:
            raise SystemExit(f"{args.command} refused: {error}") from None
        signal_bundle = bundle
        if args.command == "live20k-2026":
            preset_name, config, execution_config = _cli_preset_configs(args.preset)
        else:
            config = Live20KSignalConfig()
            execution_config = None
            preset_name = args.command
        regime = build_market_regime(signal_bundle.prices)
        regime["dist_ma60"] = regime["market_equity"] / regime["market_ma60"] - 1
        regime["market_gate"] = (
            (regime["market_equity"] > regime["market_ma60"])
            & (regime["breadth_ma20"] >= config.market_breadth_min)
            & (regime["dist_ma60"] <= config.market_dist_ma60_max)
        )
        latest_regime = regime.loc[regime["date"] <= as_of].tail(1)
        if latest_regime.empty:
            raise ValueError(f"No market data available on or before {args.as_of}")
        row = latest_regime.iloc[0]
        gate = bool(row["market_gate"])
        entry_gate = live20k_entry_gate_open(config, row)
        print(
            "状态 "
            f"入场={'开' if entry_gate else '关'} "
            f"大盘={'开' if gate else '关'} "
            f"宽度底线={_format_breadth_floor_display(config, row)} "
            f"策略={preset_name} "
            f"日期={pd.Timestamp(row['date']).date()} "
            f"大盘宽度={row['breadth_ma20']:.2%} "
            f"离60日线={row['dist_ma60']:.2%}"
        )
        holdings = pd.read_csv(args.holdings_csv) if args.holdings_csv else None
        plan = build_live20k_daily_plan(
            signal_bundle,
            as_of=as_of,
            holdings=holdings,
            cash=args.cash,
            equity_peak=args.equity_peak,
            signal_config=config,
            execution_config=execution_config,
        )
        if args.plan_output:
            output_path = Path(args.plan_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plan.to_csv(output_path, index=False)
        orders = plan.loc[plan["action"] != "wait"].copy()
        if orders.empty:
            print("No new positions.")
        else:
            columns = [column.strip() for column in args.columns.split(",") if column.strip()]
            available = [column for column in columns if column in orders.columns]
            print(_format_live20k_display(orders[available]))
        if args.command == "live20k-2026" and args.watchlist:
            watchlist = build_live20k_watchlist(
                signal_bundle,
                as_of=as_of,
                signal_config=config,
                limit=args.watchlist_limit,
            )
            if args.watchlist_output:
                _write_csv(watchlist, args.watchlist_output)
            if watchlist.empty:
                print("No watchlist candidates.")
            else:
                print("watchlist")
                print(_format_live20k_display(watchlist))
    elif args.command == "live20k-fill":
        bundle = SqliteDataProvider(args.db).load()
        plan = pd.read_csv(args.plan_csv)
        holdings = pd.read_csv(args.holdings_csv) if args.holdings_csv else None
        result = apply_live20k_paper_fills(
            bundle,
            plan=plan,
            holdings=holdings,
            cash=args.cash,
            equity_peak=args.equity_peak,
            execution_date=args.execution_date,
        )
        if args.holdings_output:
            _write_csv(result.holdings, args.holdings_output)
        if args.fills_output:
            _write_csv(result.fills, args.fills_output)
        state = pd.DataFrame(
            [
                {
                    "date": _paper_state_date(args.execution_date, result.fills, plan),
                    "cash": result.cash,
                    "equity": result.equity,
                    "equity_peak": result.equity_peak,
                }
            ]
        )
        if args.state_output:
            _write_csv(state, args.state_output)
        print(
            "paper_state "
            f"cash={result.cash:.2f} "
            f"equity={result.equity:.2f} "
            f"equity_peak={result.equity_peak:.2f} "
            f"fills={len(result.fills)} "
            f"holdings={len(result.holdings)}"
        )
    elif args.command == "live20k-report":
        plans = _read_csv_glob(args.plans_glob)
        fills = _read_csv_glob(args.fills_glob) if args.fills_glob else pd.DataFrame()
        states = _read_csv_glob(args.states_glob) if args.states_glob else pd.DataFrame()
        report = summarize_live20k_paper_run(
            plans,
            fills,
            states,
            min_days=args.min_days,
            min_order_days=args.min_order_days,
            min_total_return=args.min_total_return,
        )
        if args.output:
            _write_csv(report, args.output)
        print(report.to_string(index=False))
    elif args.command == "live20k-orders":
        plan = pd.read_csv(args.plan_csv)
        report = pd.read_csv(args.report_csv)
        try:
            orders = export_live20k_orders(plan, report)
        except ValueError as error:
            raise SystemExit(f"live20k-orders refused: {error}") from None
        _write_csv(orders, args.output)
        print(f"live_orders orders={len(orders)} output={args.output}")
    elif args.command == "live20k-step":
        paper_dir = Path(args.paper_dir)
        as_of = pd.Timestamp(args.as_of)
        bundle = _load_live20k_signal_bundle(args.db, as_of)
        try:
            assert_live20k_data_fresh(bundle, as_of)
        except ValueError as error:
            raise SystemExit(f"live20k-step refused: {error}") from None
        signal_bundle = bundle
        tag = as_of.strftime("%Y%m%d")
        previous_plan_path = _latest_previous_plan_path(paper_dir, as_of)
        fills_path = paper_dir / f"live20k_{tag}_fills.csv"
        previous_plan = None
        if previous_plan_path is not None and not fills_path.exists():
            previous_plan = pd.read_csv(previous_plan_path)
        holdings_path = paper_dir / "live20k_holdings.csv"
        state_path = paper_dir / "live20k_state.csv"
        holdings = pd.read_csv(holdings_path) if holdings_path.exists() else None
        state = pd.read_csv(state_path) if state_path.exists() else None
        existing_plans = _read_csv_glob(str(paper_dir / "live20k_*_plan.csv"))
        existing_fills = _read_csv_glob(str(paper_dir / "live20k_*_fills.csv"))
        existing_states = _read_csv_glob(str(paper_dir / "live20k_*_state.csv"))
        preset_name, signal_config, execution_config = _cli_preset_configs(args.preset)
        result = run_live20k_paper_step(
            signal_bundle,
            as_of=as_of,
            holdings=holdings,
            state=state,
            previous_plan=previous_plan,
            existing_plans=_drop_date(existing_plans, as_of),
            existing_fills=_drop_date(existing_fills, as_of),
            existing_states=_drop_date(existing_states, as_of),
            min_days=args.min_days,
            min_order_days=args.min_order_days,
            min_total_return=args.min_total_return,
            signal_config=signal_config,
            execution_config=execution_config,
        )
        _write_csv(result.plan, str(paper_dir / f"live20k_{tag}_plan.csv"))
        if result.filled_previous_plan or not fills_path.exists():
            _write_csv(result.fills, str(fills_path))
        _write_csv(result.holdings, str(holdings_path))
        _write_csv(result.state, str(state_path))
        _write_csv(result.state, str(paper_dir / f"live20k_{tag}_state.csv"))
        _write_csv(result.report, str(paper_dir / "live20k_report.csv"))
        report_row = result.report.iloc[0]
        print(
            "paper_step "
            f"date={as_of.date()} "
            f"preset={preset_name} "
            f"filled_previous_plan={result.filled_previous_plan} "
            f"paper_days={report_row['paper_days']} "
            f"ready_for_live={report_row['ready_for_live']} "
            f"blocking_reason={report_row['blocking_reason']}"
        )
    elif args.command == "live20k-replay":
        start = pd.Timestamp(args.start)
        end = pd.Timestamp(args.end)
        bundle = _load_live20k_replay_bundle(args.db, start, end)
        try:
            assert_live20k_data_fresh(bundle, end)
        except ValueError as error:
            raise SystemExit(f"live20k-replay refused: {error}") from None
        _, signal_config, execution_config = _cli_preset_configs(args.preset)
        result = run_live20k_paper_replay(
            bundle,
            start=start,
            end=end,
            min_days=args.min_days,
            min_order_days=args.min_order_days,
            min_total_return=args.min_total_return,
            signal_config=signal_config,
            execution_config=execution_config,
        )
        paper_dir = Path(args.paper_dir)
        _write_csv(result.plans, str(paper_dir / "live20k_replay_plans.csv"))
        _write_csv(result.fills, str(paper_dir / "live20k_replay_fills.csv"))
        _write_csv(result.holdings, str(paper_dir / "live20k_holdings.csv"))
        _write_csv(result.states, str(paper_dir / "live20k_replay_states.csv"))
        _write_csv(result.report, str(paper_dir / "live20k_report.csv"))
        report_row = result.report.iloc[0]
        print(
            "paper_replay "
            f"start={start.date()} "
            f"end={end.date()} "
            f"preset={args.preset} "
            f"paper_days={report_row['paper_days']} "
            f"order_days={report_row['order_days']} "
            f"total_return={report_row['total_return']:.2%} "
            f"max_drawdown={report_row['max_drawdown']:.2%} "
            f"ready_for_live={report_row['ready_for_live']} "
            f"blocking_reason={report_row['blocking_reason']}"
        )
    elif args.command == "live20k-optimize":
        start = pd.Timestamp(args.start)
        end = pd.Timestamp(args.end)
        recent_start = pd.Timestamp(args.recent_start) if args.recent_start else start
        bundle = _load_live20k_replay_bundle(args.db, start, end)
        try:
            assert_live20k_data_fresh(bundle, end)
        except ValueError as error:
            raise SystemExit(f"live20k-optimize refused: {error}") from None
        signals = generate_live20k_signals(bundle, live100k_hotspot_turbo_signal_config())
        leaderboard = evaluate_execution_variants(
            bundle.prices,
            signals,
            default_live20k_execution_variants(live100k_hotspot_manual_fixed_execution_config()),
            start=start,
            end=end,
            recent_start=recent_start,
        )
        if args.output:
            _write_csv(leaderboard, args.output)
        if leaderboard.empty:
            print("No optimization variants produced results.")
        else:
            columns = [column.strip() for column in args.columns.split(",") if column.strip()]
            available = [column for column in columns if column in leaderboard.columns]
            print(leaderboard[available].to_string(index=False))
    elif args.command == "live20k-quality":
        bundle = SqliteDataProvider(args.db).load()
        config = Live20KSignalConfig(
            rank_start=args.rank_start,
            signal_limit=args.signal_limit,
            apply_market_gate=args.apply_market_gate,
        )
        signals = generate_live20k_signals(bundle, config=config)
        if not signals.empty:
            signals["date"] = pd.to_datetime(signals["date"])
            if args.start:
                signals = signals.loc[signals["date"] >= pd.Timestamp(args.start)].copy()
            if args.end:
                signals = signals.loc[signals["date"] <= pd.Timestamp(args.end)].copy()
        details = evaluate_ranked_signal_quality(
            signals,
            bundle.prices,
            horizons=_parse_int_csv(args.horizons),
            stable_drawdown_floor=args.stable_drawdown_floor,
            min_gain_retention=args.min_gain_retention,
        )
        summary = summarize_signal_quality(details)
        if args.detail_output:
            _write_csv(details, args.detail_output)
        if args.summary_output:
            _write_csv(summary, args.summary_output)
        if summary.empty:
            print("No evaluable signals.")
        else:
            columns = [column.strip() for column in args.columns.split(",") if column.strip()]
            available = [column for column in columns if column in summary.columns]
            print(summary[available].to_string(index=False))
    elif args.command == "live20k-context":
        bundle = SqliteDataProvider(args.db).load()
        as_of = pd.Timestamp(args.as_of)
        try:
            assert_live20k_data_fresh(bundle, as_of)
        except ValueError as error:
            raise SystemExit(f"live20k-context refused: {error}") from None

        regime = classify_market_regimes(build_market_regime(bundle.prices))
        current_rows = regime.loc[regime["date"] <= as_of].tail(1)
        if current_rows.empty:
            raise ValueError(f"No market data available on or before {args.as_of}")
        current = current_rows.iloc[0]
        current_date = pd.Timestamp(current["date"])
        current_regime = str(current["market_regime"])
        current_action = str(current["regime_action"])

        same_regime_dates = (
            regime.loc[(regime["date"] <= current_date) & (regime["market_regime"] == current_regime), "date"]
            .tail(args.lookback_days)
            .tolist()
        )
        signals = generate_live20k_signals(
            bundle,
            config=Live20KSignalConfig(rank_start=1, signal_limit=args.signal_limit, apply_market_gate=False),
        )
        if not signals.empty:
            signals["date"] = pd.to_datetime(signals["date"])
        quality_signals = signals.loc[signals["date"].isin(same_regime_dates)].copy()
        details = evaluate_ranked_signal_quality(
            quality_signals,
            bundle.prices,
            horizons=_parse_int_csv(args.horizons),
        )
        summary = summarize_signal_quality(details)
        candidates = _current_context_candidates(
            signals,
            current_date=current_date,
            rank_start=args.candidate_rank_start,
            rank_end=args.candidate_rank_end,
            market_regime=current_regime,
            regime_action=current_action,
        )

        if args.detail_output:
            _write_csv(details, args.detail_output)
        if args.summary_output:
            _write_csv(summary, args.summary_output)
        if args.candidates_output:
            _write_csv(candidates, args.candidates_output)

        print(
            "market_context "
            f"date={current_date.date()} "
            f"regime={current_regime} "
            f"action={current_action} "
            f"breadth_ma20={float(current['breadth_ma20']):.2%} "
            f"dist_ma60={float(current['dist_ma60']):.2%} "
            f"same_regime_dates={len(same_regime_dates)}"
        )
        if summary.empty:
            print("No same-regime signal quality available.")
        else:
            print(
                summary[
                    [
                        "horizon_days",
                        "rank_bucket",
                        "count",
                        "positive_rate",
                        "stable_rate",
                        "mean_forward_return",
                        "median_forward_return",
                        "mean_max_drawdown",
                    ]
                ].to_string(index=False)
            )
        if candidates.empty:
            print("No current candidates.")
        else:
            print(
                candidates[
                    [
                        "raw_rank",
                        "code",
                        "name",
                        "industry",
                        "close",
                        "score",
                        "hits",
                        "candidate_action",
                    ]
                ].to_string(index=False)
            )
    elif args.command == "ingest" and args.source == "baostock":
        symbols = [item.strip() for item in args.symbols.split(",")] if args.symbols else None
        summary = ingest_baostock(
            BaoStockIngestConfig(
                output_path=args.output,
                symbols=symbols,
                year=args.year,
                quarter=args.quarter,
                limit=args.limit,
                offset=args.offset,
                sleep_seconds=args.sleep,
                continue_on_error=not args.fail_fast,
            )
        )
        print(
            "BaoStock ingest complete: "
            f"stocks={summary['stocks']}, fundamentals={summary['fundamentals']}, "
            f"failures={summary['failures']}"
        )
    elif args.command == "ingest" and args.source == "efinance":
        symbols = [item.strip() for item in args.symbols.split(",")] if args.symbols else None
        summary = ingest_efinance_money_flow(
            EFinanceMoneyFlowConfig(
                output_path=args.output,
                symbols=symbols,
                start_date=args.start,
                end_date=args.end,
                limit=args.limit,
                offset=args.offset,
                sleep_seconds=args.sleep,
                retries=args.retries,
                timeout=args.timeout,
                flush_every=args.flush_every,
                progress_every=args.progress_every,
                continue_on_error=not args.fail_fast,
            )
        )
        print(
            "efinance money-flow ingest complete: "
            f"money_flow={summary['money_flow']}, failures={summary['failures']}, "
            f"requested_symbols={summary['requested_symbols']}"
        )
    elif args.command == "ingest" and args.source == "tushare":
        summary = ingest_tushare(
            TushareIngestConfig(
                output_path=args.output,
                start_date=args.start,
                end_date=args.end,
                token_path=args.token_path,
                adjust=args.adjust,
                write_mode=args.write_mode,
                include_prices=not args.only_money_flow,
                include_daily_basic=not args.skip_daily_basic
                and not args.only_money_flow
                and not args.only_prices,
                include_money_flow=not args.skip_money_flow and not args.only_core and not args.only_prices,
                include_limits=not args.skip_limits and not args.only_money_flow and not args.only_prices,
                include_suspends=not args.skip_suspends and not args.only_money_flow and not args.only_prices,
                flush_every=args.flush_every,
                progress_every=args.progress_every,
                resume=args.resume,
                sleep_seconds=args.sleep,
                workers=args.workers,
                rate_limit_per_minute=args.rate_limit_per_minute,
                timeout=args.timeout,
                retries=args.retries,
                retry_sleep_seconds=args.retry_sleep,
                continue_on_error=not args.fail_fast,
            )
        )
        print(
            "Tushare ingest complete: "
            f"stocks={summary['stocks']}, prices={summary['prices']}, "
            f"fundamentals={summary['fundamentals']}, money_flow={summary['money_flow']}, "
            f"events={summary['events']}, failures={summary['failures']}, "
            f"trade_dates={summary['trade_dates']}, fetched_dates={summary['fetched_dates']}, "
            f"skipped_dates={summary['skipped_dates']}"
        )
    elif args.command == "convert" and args.conversion == "csv-to-sqlite":
        summary = convert_csv_to_sqlite(args.input, args.output)
        print(
            "CSV to SQLite complete: "
            f"prices={summary['prices']}, fundamentals={summary['fundamentals']}, "
            f"stocks={summary['stocks']}, events={summary['events']}, "
            f"money_flow={summary['money_flow']}"
        )


def _write_csv(frame: pd.DataFrame, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def _write_text(text: str, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def _daily_plan_strategy(payload: dict) -> dict:
    strategies = payload.get("strategies") or []
    for strategy in strategies:
        if strategy.get("preset") == DESK_MULTIFACTOR_PRESET:
            return strategy
    if strategies:
        return strategies[0]
    raise SystemExit("daily-plan requires at least one strategy snapshot")


def _format_daily_plan_markdown(payload: dict, strategy: dict) -> str:
    as_of = str(payload.get("as_of") or payload.get("latest_date") or "")
    regime = strategy.get("regime") or {}
    regime_name = regime.get("regime", "unknown")
    breadth = _markdown_percent(regime.get("breadth_ma20"))
    dist_ma60 = _markdown_percent(regime.get("dist_ma60"))
    entry_gate = "开" if strategy.get("entry_gate") else "关"
    market_gate = "开" if strategy.get("market_gate") else "关"
    lines = [
        f"# BaiQuant 每日实盘清单 - {as_of}",
        "",
        "## 市场状态",
        f"- 策略：{strategy.get('preset', DESK_MULTIFACTOR_PRESET)}",
        f"- Regime：{regime_name}",
        f"- 大盘宽度：{breadth}",
        f"- 距离 60 日线：{dist_ma60}",
        f"- 新开仓门：{entry_gate}；市场门：{market_gate}",
        "",
        "## 账户快照",
        _markdown_account_snapshot(payload.get("account") or {}),
        "",
        "## 持仓风控",
        _markdown_table(
            payload.get("positions") or [],
            [
                "code",
                "name",
                "shares",
                "current_price",
                "market_value",
                "unrealized_return",
                "drawdown_from_high",
                "stop_signal",
            ],
        ),
        "",
        "## 次日操作",
        _markdown_table(
            strategy.get("plan") or [],
            ["action", "code", "name", "reason", "shares", "reference_price", "current_return", "cash_budget"],
        ),
        "",
        "## 观察池",
        _markdown_table(
            strategy.get("watchlist") or [],
            ["candidate_action", "code", "name", "industry", "close", "multi_factor_score", "candidate_rank"],
        ),
        "",
        "## 执行纪律",
        "- 只买计划里明确写成 `次日买入` 的票；观察池不是买单。",
        "- `新开仓门=关` 时只处理旧仓，不新增股票。",
        "- `手动复核` 通常代表价格库缺失或证券类型特殊，先用券商行情确认。",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _markdown_account_snapshot(account: dict) -> str:
    if not account:
        return "无"
    return "\n".join(
        [
            f"- 账户权益：{_markdown_number(account.get('equity'))}",
            f"- 可用现金：{_markdown_number(account.get('cash'))}",
            f"- 持仓市值：{_markdown_number(account.get('market_value'))}",
            f"- 持仓浮盈：{_markdown_number(account.get('unrealized_pnl'))}",
            f"- 持仓收益率：{_markdown_percent(account.get('unrealized_return'))}",
            f"- 仓位暴露：{_markdown_percent(account.get('exposure'))}",
            f"- 持仓数量：{account.get('positions', '')}",
        ]
    )


def _markdown_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "无"
    header = [MULTIFACTOR_DISPLAY_LABELS.get(column, LIVE20K_DISPLAY_LABELS.get(column, column)) for column in columns]
    table = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        table.append("| " + " | ".join(_markdown_cell(row.get(column), column) for column in columns) + " |")
    return "\n".join(table)


def _markdown_cell(value, column: str) -> str:  # noqa: ANN001
    if value is None or pd.isna(value):
        return ""
    replacements = {}
    replacements.update(MULTIFACTOR_DISPLAY_VALUES.get(column, {}))
    if column == "candidate_action":
        replacements.update({"buy_candidate": "买入候选", "watch": "观察", "watch_only": "只观察"})
    if value in replacements:
        return str(replacements[value])
    if column in {"reference_price", "close", "cash_budget", "multi_factor_score", "current_price", "market_value"}:
        return _markdown_number(value)
    if column in {"current_return", "unrealized_return", "drawdown_from_high"}:
        return _markdown_percent(value)
    if column == "stop_signal":
        return {
            "hold": "持有",
            "stop_loss": "止损",
            "trailing_stop": "回撤止盈",
            "max_holding_days": "到期轮动",
            "missing_price": "缺少价格",
        }.get(str(value), str(value))
    return str(value)


def _markdown_number(value) -> str:  # noqa: ANN001
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return ""
    return f"{float(number):.2f}"


def _markdown_percent(value) -> str:  # noqa: ANN001
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return ""
    return f"{float(number) * 100:.2f}%"


def _resolve_multifactor_weights(args) -> dict[str, float]:  # noqa: ANN001
    if getattr(args, "weights_profile", "default") == "default":
        return dict(MULTIFACTOR_DEFAULT_WEIGHTS)
    diagnostics_path = getattr(args, "diagnostics_csv", None)
    if not diagnostics_path:
        raise SystemExit("--diagnostics-csv is required when --weights-profile validated")
    path = Path(diagnostics_path)
    if not path.exists():
        raise SystemExit(f"diagnostics CSV not found: {diagnostics_path}")
    diagnostics = pd.read_csv(path)
    return derive_validated_multifactor_weights(
        diagnostics,
        MULTIFACTOR_DEFAULT_WEIGHTS,
        min_coverage_rate=float(getattr(args, "min_factor_coverage", 0.8)),
        min_ic_observations=int(getattr(args, "min_factor_observations", 3)),
    )


def _allowed_regimes(args) -> tuple[str, ...]:  # noqa: ANN001
    return tuple(item.strip() for item in str(getattr(args, "allowed_regimes", "")).split(",") if item.strip())


def _resolve_multifactor_regimes(bundle: MarketDataBundle, args) -> pd.DataFrame | None:  # noqa: ANN001
    if not getattr(args, "regime_gate", False):
        return None
    return build_regime_frame(bundle.prices)


def _multifactor_regime_allows_date(bundle: MarketDataBundle, args, as_of: pd.Timestamp) -> bool:  # noqa: ANN001
    regimes = _resolve_multifactor_regimes(bundle, args)
    if regimes is None or regimes.empty:
        return True
    frame = regimes.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    day = frame.loc[frame["date"] == pd.Timestamp(as_of).normalize()]
    if day.empty:
        return False
    regime = str(day.iloc[-1].get("regime", ""))
    return regime in set(_allowed_regimes(args))


def _parse_period_specs(value: str) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    periods: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for raw_spec in [item.strip() for item in str(value).split(",") if item.strip()]:
        parts = raw_spec.split(":")
        if len(parts) != 3:
            raise SystemExit(f"invalid period spec: {raw_spec}; expected label:start:end")
        label, start, end = parts
        start_date = pd.Timestamp(start)
        end_date = pd.Timestamp(end)
        if end_date < start_date:
            raise SystemExit(f"invalid period spec: {raw_spec}; end before start")
        periods.append((label, start_date, end_date))
    if not periods:
        raise SystemExit("--periods must include at least one label:start:end spec")
    return periods


def _multifactor_execution_config(args) -> MultiPositionTrendConfig:  # noqa: ANN001
    return MultiPositionTrendConfig(
        initial_cash=args.cash,
        max_positions=args.max_positions,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
        add_trigger_pct=args.add_trigger_pct,
        add_position_multiple=args.add_position_multiple,
        ma_window=args.ma_window,
        trailing_stop_activation_pct=args.trailing_stop_activation_pct,
        trailing_stop_pct=args.trailing_stop_pct,
        max_holding_days=args.max_holding_days,
        portfolio_stop_drawdown_pct=args.portfolio_stop_drawdown_pct,
        portfolio_stop_cooldown_days=args.portfolio_stop_cooldown_days,
        liquidate_on_portfolio_stop=args.liquidate_on_portfolio_stop,
        use_position_scale=True,
    )


def _run_multifactor_backtest_period(
    bundle: MarketDataBundle,
    args,  # noqa: ANN001
    start: pd.Timestamp,
    end: pd.Timestamp,
    weights: dict[str, float],
    period: str | None = None,
    regimes: pd.DataFrame | None = None,
):
    signals = generate_multifactor_signals(
        bundle,
        start=start,
        end=end,
        factor_weights=weights,
        top_n=args.top,
        signal_every_n_days=args.signal_every_n_days,
        max_per_industry=args.max_per_industry,
        max_price=args.max_price,
        max_lot_cost=args.max_lot_cost,
        regimes=regimes,
        allowed_regimes=_allowed_regimes(args) if regimes is not None else None,
    )
    execution_config = _multifactor_execution_config(args)
    execution_prices = _filter_prices_to_signal_codes(bundle.prices, signals)
    result = run_multi_position_trend_backtest(execution_prices, signals, execution_config)
    summary = _multifactor_backtest_summary(
        result,
        signals=signals,
        start=start,
        end=end,
        cash=args.cash,
        top_n=args.top,
        max_positions=args.max_positions,
        signal_every_n_days=args.signal_every_n_days,
    )
    if period is not None:
        summary.insert(0, "period", period)
    return result, signals, summary


def _format_live20k_display(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in ("market_gate", "entry_gate"):
        if column in display.columns:
            display[column] = display[column].map({True: "开", False: "关"}).fillna(display[column])
    for column, replacements in LIVE20K_DISPLAY_VALUES.items():
        if column in display.columns:
            display[column] = display[column].replace(replacements)
    return display.rename(columns=LIVE20K_DISPLAY_LABELS).to_string(index=False)


def _format_multifactor_display(frame: pd.DataFrame) -> str:
    display = frame.copy()
    if "positive_factors" in display.columns:
        display["positive_factors"] = display["positive_factors"].astype(str).str.replace("|", "、", regex=False)
    for column in ("reference_price", "average_cost", "close", "cash_budget", "multi_factor_score"):
        if column in display.columns:
            display[column] = pd.to_numeric(display[column], errors="coerce").map(
                lambda value: "" if pd.isna(value) else f"{float(value):.2f}"
            )
    for column in (
        "current_return",
        "momentum_5d",
        "momentum_20d",
        "reversal_5d",
        "money_flow_pct",
        "big_order_pct",
        "industry_momentum_3d",
    ):
        if column in display.columns:
            display[column] = pd.to_numeric(display[column], errors="coerce").map(
                lambda value: "" if pd.isna(value) else f"{float(value) * 100:.2f}%"
            )
    for column, replacements in MULTIFACTOR_DISPLAY_VALUES.items():
        if column in display.columns:
            display[column] = display[column].replace(replacements)
    display = display.fillna("")
    return display.rename(columns=MULTIFACTOR_DISPLAY_LABELS).to_string(index=False)


def _format_multifactor_diagnostics_display(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in ("mean_rank_ic", "median_rank_ic"):
        if column in display.columns:
            display[column] = pd.to_numeric(display[column], errors="coerce").map(
                lambda value: "" if pd.isna(value) else f"{float(value):.4f}"
            )
    pct_columns = [column for column in display.columns if column.endswith("_mean_forward_return")]
    for column in ("positive_ic_rate", "factor_coverage_rate"):
        if column in display.columns:
            pct_columns.append(column)
    for column in pct_columns:
        display[column] = pd.to_numeric(display[column], errors="coerce").map(
            lambda value: "" if pd.isna(value) else f"{float(value) * 100:.2f}%"
        )
    display = display.fillna("")
    return display.rename(columns=MULTIFACTOR_DIAGNOSTIC_DISPLAY_LABELS).to_string(index=False)


def _multifactor_backtest_summary(
    result,
    signals: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cash: float,
    top_n: int,
    max_positions: int,
    signal_every_n_days: int,
) -> pd.DataFrame:
    trades = result.trades.copy()
    filled_trades = int(trades["status"].eq("filled").sum()) if not trades.empty and "status" in trades.columns else 0
    return pd.DataFrame(
        [
            {
                "start": start.date().isoformat(),
                "end": end.date().isoformat(),
                "initial_cash": cash,
                "top_n": top_n,
                "max_positions": max_positions,
                "signal_every_n_days": signal_every_n_days,
                "signal_days": int(pd.to_datetime(signals["date"]).nunique()) if not signals.empty else 0,
                "signals": int(len(signals)),
                "filled_trades": filled_trades,
                "total_return": result.metrics.get("total_return", 0.0),
                "annualized_return": result.metrics.get("annualized_return", 0.0),
                "volatility": result.metrics.get("volatility", 0.0),
                "sharpe": result.metrics.get("sharpe", 0.0),
                "max_drawdown": result.metrics.get("max_drawdown", 0.0),
            }
        ]
    )


def _filter_prices_to_signal_codes(prices: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    if prices.empty or signals.empty or "code" not in prices.columns or "code" not in signals.columns:
        return prices
    signal_codes = set(signals["code"].astype(str))
    return prices.loc[prices["code"].astype(str).isin(signal_codes)].copy()


def _load_live20k_signal_bundle(db_path: str, as_of: pd.Timestamp) -> MarketDataBundle:
    return SqliteDataProvider(db_path).load_window(
        as_of=as_of,
        lookback_days=LIVE20K_SIGNAL_PRICE_LOOKBACK_DAYS,
        fundamentals_lookback_days=LIVE20K_SIGNAL_SIDE_LOOKBACK_DAYS,
        events_lookback_days=LIVE20K_SIGNAL_SIDE_LOOKBACK_DAYS,
        money_flow_lookback_days=LIVE20K_SIGNAL_SIDE_LOOKBACK_DAYS,
    )


def _load_live20k_replay_bundle(
    db_path: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> MarketDataBundle:
    span_days = max(0, int((end.normalize() - start.normalize()).days))
    return SqliteDataProvider(db_path).load_window(
        as_of=end,
        lookback_days=LIVE20K_SIGNAL_PRICE_LOOKBACK_DAYS + span_days,
        fundamentals_lookback_days=LIVE20K_SIGNAL_SIDE_LOOKBACK_DAYS + span_days,
        events_lookback_days=LIVE20K_SIGNAL_SIDE_LOOKBACK_DAYS + span_days,
        money_flow_lookback_days=LIVE20K_SIGNAL_SIDE_LOOKBACK_DAYS + span_days,
    )


def _normalize_live20k_preset(preset: str | None) -> str:
    name = (preset or LIVE20K_MANUAL_20D_PRESET).strip()
    return LIVE20K_PRESET_ALIASES.get(name, name)


def _cli_preset_configs(preset: str | None):
    normalized = _normalize_live20k_preset(preset)
    signal_config = _signal_config_for_preset(normalized)
    execution_config = _execution_config_for_preset(normalized)
    if signal_config is None or execution_config is None:
        available = ", ".join(LIVE20K_PRESETS)
        raise SystemExit(f"unknown preset: {preset}. available presets: {available}")
    return normalized, signal_config, execution_config


def _signal_config_for_preset(preset: str) -> Live20KSignalConfig | None:
    preset = _normalize_live20k_preset(preset)
    if preset == LIVE20K_MANUAL_20D_PRESET:
        return live100k_hotspot_manual_fixed_signal_config()
    if preset == LIVE20K_TURBO_SPRINT_PRESET:
        return live100k_hotspot_turbo_signal_config()
    return None


def _execution_config_for_preset(preset: str):
    preset = _normalize_live20k_preset(preset)
    if preset == LIVE20K_MANUAL_20D_PRESET:
        return live100k_hotspot_manual_fixed_execution_config()
    if preset == LIVE20K_TURBO_SPRINT_PRESET:
        return live100k_hotspot_turbo_execution_config()
    return None


def _format_breadth_floor(config: Live20KSignalConfig, row: pd.Series) -> str:
    if config.market_breadth_floor is None:
        return "NA"
    breadth = row.get("breadth_ma20", pd.NA)
    floor_ok = pd.notna(breadth) and float(breadth) >= config.market_breadth_floor
    return f"{'ON' if floor_ok else 'OFF'}({config.market_breadth_floor:.2%})"


def _format_breadth_floor_display(config: Live20KSignalConfig, row: pd.Series) -> str:
    if config.market_breadth_floor is None:
        return "无"
    breadth = row.get("breadth_ma20", pd.NA)
    floor_ok = pd.notna(breadth) and float(breadth) >= config.market_breadth_floor
    return f"{'开' if floor_ok else '关'}({config.market_breadth_floor:.2%})"


def _read_csv_glob(pattern: str) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in sorted(glob(pattern))]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _paper_state_date(execution_date: str | None, fills: pd.DataFrame, plan: pd.DataFrame) -> str:
    if execution_date:
        return str(pd.Timestamp(execution_date).date())
    if not fills.empty and "date" in fills.columns:
        return str(pd.to_datetime(fills["date"]).max().date())
    if "date" in plan.columns and not plan.empty:
        return str(pd.to_datetime(plan["date"]).max().date())
    return ""


def _latest_previous_plan_path(paper_dir: Path, as_of: pd.Timestamp) -> Path | None:
    paths = []
    for path in sorted(paper_dir.glob("live20k_*_plan.csv")):
        try:
            tag = path.name.removeprefix("live20k_").removesuffix("_plan.csv")
            date = pd.Timestamp(tag)
        except ValueError:
            continue
        if date < as_of:
            paths.append((date, path))
    if not paths:
        return None
    return sorted(paths, key=lambda item: item[0])[-1][1]


def _drop_date(frame: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    if frame.empty or "date" not in frame.columns:
        return frame
    dates = pd.to_datetime(frame["date"], errors="coerce")
    return frame.loc[dates != date].reset_index(drop=True)


def _parse_int_csv(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one integer")
    return [int(item) for item in items]


def _current_context_candidates(
    signals: pd.DataFrame,
    current_date: pd.Timestamp,
    rank_start: int,
    rank_end: int,
    market_regime: str,
    regime_action: str,
) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()
    frame = signals.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if "raw_rank" not in frame.columns:
        frame["raw_rank"] = frame["score_rank"]
    frame["raw_rank"] = pd.to_numeric(frame["raw_rank"], errors="coerce")
    candidates = frame.loc[
        (frame["date"] == current_date)
        & (frame["raw_rank"] >= rank_start)
        & (frame["raw_rank"] <= rank_end)
    ].copy()
    if candidates.empty:
        return candidates
    candidates["market_regime"] = market_regime
    candidates["candidate_action"] = "buy_candidate" if regime_action == "trade_top_6_10" else regime_action
    return candidates.sort_values(["raw_rank", "code"]).reset_index(drop=True)
