import pandas as pd
import pytest

from baiquant.data.bundle import MarketDataBundle
from baiquant.cli import (
    LIVE20K_MANUAL_20D_PRESET,
    LIVE20K_TURBO_SPRINT_PRESET,
    _normalize_live20k_preset,
    build_parser,
    main,
)


def test_cli_parses_live20k_command() -> None:
    args = build_parser().parse_args(
        [
            "live20k",
            "--db",
            "data/tushare/baiquant.db",
            "--as-of",
            "2026-05-25",
        ]
    )

    assert args.command == "live20k"
    assert args.db == "data/tushare/baiquant.db"
    assert args.as_of == "2026-05-25"


def test_cli_parses_desk_command_with_quick_defaults() -> None:
    args = build_parser().parse_args(["desk"])

    assert args.command == "desk"
    assert args.db == "data/tushare/baiquant.db"
    assert args.holdings == "data/live/holdings.csv"
    assert args.host == "127.0.0.1"
    assert args.port == 8765


def test_cli_parses_daily_plan_with_live_defaults() -> None:
    args = build_parser().parse_args(["daily-plan"])

    assert args.command == "daily-plan"
    assert args.db == "data/tushare/baiquant.db"
    assert args.holdings == "data/live/holdings.csv"
    assert args.as_of == ""
    assert args.cash == 0.0
    assert args.limit == 8
    assert args.output == "data/live/daily_plan.md"
    assert args.plan_csv == "data/live/daily_plan.csv"
    assert args.watchlist_csv == "data/live/daily_watchlist.csv"


def test_cli_parses_record_trade_with_live_defaults() -> None:
    args = build_parser().parse_args(
        [
            "record-trade",
            "--date",
            "2026-07-02",
            "--action",
            "buy",
            "--code",
            "601636",
            "--name",
            "示例A",
            "--shares",
            "100",
            "--price",
            "9.70",
        ]
    )

    assert args.command == "record-trade"
    assert args.holdings == "data/live/holdings.csv"
    assert args.trade_log == "data/live/trade_log.csv"
    assert args.action == "buy"
    assert args.code == "601636"
    assert args.shares == 100
    assert args.price == 9.70
    assert args.fees == 0.0
    assert args.source == "manual"


def test_record_trade_command_updates_live_files(tmp_path, monkeypatch, capsys) -> None:
    holdings_path = tmp_path / "holdings.csv"
    trade_log_path = tmp_path / "trade_log.csv"
    pd.DataFrame([{"code": "601611.SH", "name": "示例B", "shares": 800, "average_cost": 13.678}]).to_csv(
        holdings_path,
        index=False,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "baiquant",
            "record-trade",
            "--date",
            "2026-07-02",
            "--action",
            "sell",
            "--code",
            "601611.SH",
            "--name",
            "示例B",
            "--shares",
            "300",
            "--price",
            "11.20",
            "--fees",
            "2",
            "--holdings",
            str(holdings_path),
            "--trade-log",
            str(trade_log_path),
            "--source",
            "manual_test",
        ],
    )

    main()

    output = capsys.readouterr().out
    assert "recorded sell 601611.SH 300 @ 11.20" in output
    assert "after=500" in output
    assert "realized_pnl=-745.40" in output
    assert pd.read_csv(holdings_path)["shares"].tolist() == [500]
    log = pd.read_csv(trade_log_path)
    assert log[["action", "code", "shares", "holdings_shares_after"]].to_dict("records") == [
        {"action": "sell", "code": "601611.SH", "shares": 300, "holdings_shares_after": 500}
    ]


def test_daily_plan_command_writes_markdown_and_csvs(tmp_path, monkeypatch, capsys) -> None:
    output_path = tmp_path / "daily.md"
    plan_path = tmp_path / "plan.csv"
    watchlist_path = tmp_path / "watchlist.csv"
    seen = {}

    def build_payload(**kwargs):  # noqa: ANN001
        seen.update(kwargs)
        return {
            "db_path": kwargs["db_path"],
            "holdings_path": kwargs["holdings_path"],
            "latest_date": "2026-07-01",
            "as_of": "2026-07-01",
            "cash": kwargs["cash"],
            "account": {
                "equity": 50_000,
                "cash": 5_000,
                "market_value": 45_000,
                "unrealized_pnl": -1_200,
                "unrealized_return": -0.026,
                "exposure": 0.90,
                "positions": 2,
            },
            "positions": [
                {
                    "code": "601611.SH",
                    "name": "示例B",
                    "shares": 800,
                    "current_price": 11.28,
                    "market_value": 9024,
                    "unrealized_return": -0.1753,
                    "drawdown_from_high": -0.21,
                    "stop_signal": "stop_loss",
                },
                {
                    "code": "688520.SH",
                    "name": "示例C",
                    "shares": 400,
                    "current_price": 37.60,
                    "market_value": 15040,
                    "unrealized_return": -0.0264,
                    "drawdown_from_high": -0.04,
                    "stop_signal": "hold",
                },
            ],
            "strategies": [
                {
                    "preset": "多因子手动版",
                    "entry_gate": True,
                    "market_gate": True,
                    "regime": {"regime": "weak_range", "breadth_ma20": 0.3216, "dist_ma60": -0.0524},
                    "plan": [
                        {
                            "action": "sell_next_open",
                            "code": "601611.SH",
                            "name": "示例B",
                            "reason": "single_stop",
                            "shares": 800,
                            "reference_price": 11.28,
                        }
                    ],
                    "watchlist": [
                        {
                            "candidate_action": "buy_candidate",
                            "code": "688689.SH",
                            "name": "银河微电",
                            "industry": "半导体",
                            "close": 77.5,
                            "multi_factor_score": 15.24,
                            "candidate_rank": 1,
                        }
                    ],
                }
            ],
        }

    monkeypatch.setattr("baiquant.cli.build_desk_payload", build_payload, raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "baiquant",
            "daily-plan",
            "--as-of",
            "2026-07-01",
            "--cash",
            "50000",
            "--output",
            str(output_path),
            "--plan-csv",
            str(plan_path),
            "--watchlist-csv",
            str(watchlist_path),
        ],
    )

    main()

    assert seen["db_path"] == "data/tushare/baiquant.db"
    assert seen["as_of"] == "2026-07-01"
    assert seen["holdings_path"] == "data/live/holdings.csv"
    assert seen["cash"] == 50_000
    assert seen["watchlist_limit"] == 8
    report = output_path.read_text()
    assert "# BaiQuant 每日实盘清单 - 2026-07-01" in report
    assert "## 账户快照" in report
    assert "账户权益：50000.00" in report
    assert "仓位暴露：90.00%" in report
    assert "## 持仓风控" in report
    assert "止损" in report
    assert "示例C" in report
    assert "weak_range" in report
    assert "大盘宽度：32.16%" in report
    assert "次日卖出" in report
    assert "示例B" in report
    assert "买入候选" in report
    assert "银河微电" in report
    assert "只买计划里明确写成 `次日买入` 的票" in report
    assert "BaiQuant 每日实盘清单" in capsys.readouterr().out
    assert pd.read_csv(plan_path)["code"].tolist() == ["601611.SH"]
    assert pd.read_csv(watchlist_path)["code"].tolist() == ["688689.SH"]


def test_cli_parses_live20k_2026_command() -> None:
    args = build_parser().parse_args(
        [
            "live20k-2026",
            "--db",
            "data/tushare/baiquant.db",
            "--as-of",
            "2026-05-25",
            "--cash",
            "15000",
            "--plan-output",
            "data/paper/live20k_2026_20260525_plan.csv",
            "--watchlist",
            "--watchlist-limit",
            "8",
            "--watchlist-output",
            "data/research/watchlist.csv",
            "--preset",
            "20天稳打版",
        ]
    )

    assert args.command == "live20k-2026"
    assert args.db == "data/tushare/baiquant.db"
    assert args.as_of == "2026-05-25"
    assert args.cash == 15000
    assert args.plan_output == "data/paper/live20k_2026_20260525_plan.csv"
    assert args.watchlist is True
    assert args.watchlist_limit == 8
    assert args.watchlist_output == "data/research/watchlist.csv"
    assert args.preset == "20天稳打版"
    for column in ["tech_score", "tech_grade", "trade_advice", "position_scale", "risk_flags"]:
        assert column in args.columns.split(",")


def test_cli_parses_live20k_2026_manual_fixed_preset() -> None:
    args = build_parser().parse_args(
        [
            "live20k-2026",
            "--db",
            "data/tushare/baiquant.db",
            "--as-of",
            "2026-05-25",
            "--preset",
            "20天稳打版",
        ]
    )

    assert args.command == "live20k-2026"
    assert args.preset == "20天稳打版"
    assert args.cash is None


def test_cli_accepts_public_english_live20k_preset_aliases() -> None:
    assert _normalize_live20k_preset("steady-20d") == LIVE20K_MANUAL_20D_PRESET
    assert _normalize_live20k_preset("manual-20d") == LIVE20K_MANUAL_20D_PRESET
    assert _normalize_live20k_preset("turbo-sprint") == LIVE20K_TURBO_SPRINT_PRESET


def test_cli_parses_live20k_2026_turbo_preset() -> None:
    args = build_parser().parse_args(
        [
            "live20k-2026",
            "--db",
            "data/tushare/baiquant.db",
            "--as-of",
            "2026-05-25",
            "--preset",
            "短线冲刺版",
        ]
    )

    assert args.command == "live20k-2026"
    assert args.preset == "短线冲刺版"
    assert args.cash is None


def test_cli_parses_live20k_paper_plan_options() -> None:
    args = build_parser().parse_args(
        [
            "live20k",
            "--db",
            "data/tushare/baiquant.db",
            "--as-of",
            "2026-05-25",
            "--holdings-csv",
            "data/paper/live20k_holdings.csv",
            "--cash",
            "18000",
            "--equity-peak",
            "22000",
            "--plan-output",
            "data/paper/live20k_20260525_plan.csv",
        ]
    )

    assert args.holdings_csv == "data/paper/live20k_holdings.csv"
    assert args.cash == 18000
    assert args.equity_peak == 22000
    assert args.plan_output == "data/paper/live20k_20260525_plan.csv"


def test_cli_parses_live20k_fill_command() -> None:
    args = build_parser().parse_args(
        [
            "live20k-fill",
            "--db",
            "data/tushare/baiquant.db",
            "--plan-csv",
            "data/paper/live20k_20260525_plan.csv",
            "--execution-date",
            "2026-05-26",
            "--holdings-csv",
            "data/paper/live20k_holdings.csv",
            "--cash",
            "18000",
            "--equity-peak",
            "22000",
            "--holdings-output",
            "data/paper/live20k_holdings_next.csv",
            "--fills-output",
            "data/paper/live20k_20260526_fills.csv",
            "--state-output",
            "data/paper/live20k_state.csv",
        ]
    )

    assert args.command == "live20k-fill"
    assert args.plan_csv == "data/paper/live20k_20260525_plan.csv"
    assert args.execution_date == "2026-05-26"
    assert args.holdings_output == "data/paper/live20k_holdings_next.csv"
    assert args.fills_output == "data/paper/live20k_20260526_fills.csv"
    assert args.state_output == "data/paper/live20k_state.csv"


def test_cli_parses_live20k_report_command() -> None:
    args = build_parser().parse_args(
        [
            "live20k-report",
            "--plans-glob",
            "data/paper/live20k_*_plan.csv",
            "--fills-glob",
            "data/paper/live20k_*_fills.csv",
            "--states-glob",
            "data/paper/live20k_*_state.csv",
            "--min-days",
            "20",
            "--min-order-days",
            "3",
            "--min-total-return",
            "0",
            "--output",
            "data/paper/live20k_report.csv",
        ]
    )

    assert args.command == "live20k-report"
    assert args.plans_glob == "data/paper/live20k_*_plan.csv"
    assert args.fills_glob == "data/paper/live20k_*_fills.csv"
    assert args.states_glob == "data/paper/live20k_*_state.csv"
    assert args.min_days == 20
    assert args.min_order_days == 3
    assert args.min_total_return == 0
    assert args.output == "data/paper/live20k_report.csv"


def test_cli_parses_live20k_orders_command() -> None:
    args = build_parser().parse_args(
        [
            "live20k-orders",
            "--plan-csv",
            "data/paper/live20k_20260525_plan.csv",
            "--report-csv",
            "data/paper/live20k_report.csv",
            "--output",
            "data/live/live20k_orders.csv",
        ]
    )

    assert args.command == "live20k-orders"
    assert args.plan_csv == "data/paper/live20k_20260525_plan.csv"
    assert args.report_csv == "data/paper/live20k_report.csv"
    assert args.output == "data/live/live20k_orders.csv"


def test_cli_parses_live20k_step_command() -> None:
    args = build_parser().parse_args(
        [
            "live20k-step",
            "--db",
            "data/tushare/baiquant.db",
            "--as-of",
            "2026-05-25",
            "--paper-dir",
            "data/paper",
            "--min-days",
            "20",
            "--preset",
            "20天稳打版",
            "--min-order-days",
            "3",
            "--min-total-return",
            "0",
        ]
    )

    assert args.command == "live20k-step"
    assert args.db == "data/tushare/baiquant.db"
    assert args.as_of == "2026-05-25"
    assert args.paper_dir == "data/paper"
    assert args.min_days == 20
    assert args.preset == "20天稳打版"
    assert args.min_order_days == 3
    assert args.min_total_return == 0


def test_cli_parses_live20k_step_manual_fixed_preset() -> None:
    args = build_parser().parse_args(
        [
            "live20k-step",
            "--db",
            "data/tushare/baiquant.db",
            "--as-of",
            "2026-05-25",
            "--preset",
            "20天稳打版",
        ]
    )

    assert args.command == "live20k-step"
    assert args.preset == "20天稳打版"


def test_cli_parses_live20k_step_turbo_preset() -> None:
    args = build_parser().parse_args(
        [
            "live20k-step",
            "--db",
            "data/tushare/baiquant.db",
            "--as-of",
            "2026-05-25",
            "--preset",
            "短线冲刺版",
        ]
    )

    assert args.command == "live20k-step"
    assert args.preset == "短线冲刺版"


def test_cli_parses_live20k_replay_command() -> None:
    args = build_parser().parse_args(
        [
            "live20k-replay",
            "--db",
            "data/tushare/baiquant.db",
            "--start",
            "2026-04-27",
            "--end",
            "2026-05-26",
            "--paper-dir",
            "data/paper_replay",
            "--preset",
            "20天稳打版",
            "--min-days",
            "20",
            "--min-order-days",
            "3",
            "--min-total-return",
            "0",
        ]
    )

    assert args.command == "live20k-replay"
    assert args.db == "data/tushare/baiquant.db"
    assert args.start == "2026-04-27"
    assert args.end == "2026-05-26"
    assert args.paper_dir == "data/paper_replay"
    assert args.preset == "20天稳打版"
    assert args.min_days == 20
    assert args.min_order_days == 3
    assert args.min_total_return == 0


def test_cli_parses_live20k_quality_command() -> None:
    args = build_parser().parse_args(
        [
            "live20k-quality",
            "--db",
            "data/tushare/baiquant.db",
            "--start",
            "2024-01-01",
            "--end",
            "2026-05-25",
            "--horizons",
            "5,10,20",
            "--signal-limit",
            "20",
            "--detail-output",
            "data/research/live20k_quality_detail.csv",
            "--summary-output",
            "data/research/live20k_quality_summary.csv",
        ]
    )

    assert args.command == "live20k-quality"
    assert args.db == "data/tushare/baiquant.db"
    assert args.start == "2024-01-01"
    assert args.end == "2026-05-25"
    assert args.horizons == "5,10,20"
    assert args.signal_limit == 20
    assert args.detail_output == "data/research/live20k_quality_detail.csv"
    assert args.summary_output == "data/research/live20k_quality_summary.csv"


def test_cli_parses_live20k_optimize_command() -> None:
    args = build_parser().parse_args(
        [
            "live20k-optimize",
            "--db",
            "data/tushare/baiquant.db",
            "--start",
            "2026-01-01",
            "--end",
            "2026-05-26",
            "--recent-start",
            "2026-04-24",
            "--output",
            "data/research/live20k_execution_leaderboard.csv",
        ]
    )

    assert args.command == "live20k-optimize"
    assert args.db == "data/tushare/baiquant.db"
    assert args.start == "2026-01-01"
    assert args.end == "2026-05-26"
    assert args.recent_start == "2026-04-24"
    assert args.output == "data/research/live20k_execution_leaderboard.csv"


def test_cli_parses_live20k_context_command() -> None:
    args = build_parser().parse_args(
        [
            "live20k-context",
            "--db",
            "data/tushare/baiquant.db",
            "--as-of",
            "2026-05-25",
            "--lookback-days",
            "252",
            "--horizons",
            "5,10,20",
            "--summary-output",
            "data/research/live20k_context_summary.csv",
            "--candidates-output",
            "data/research/live20k_context_candidates.csv",
        ]
    )

    assert args.command == "live20k-context"
    assert args.db == "data/tushare/baiquant.db"
    assert args.as_of == "2026-05-25"
    assert args.lookback_days == 252
    assert args.summary_output == "data/research/live20k_context_summary.csv"
    assert args.candidates_output == "data/research/live20k_context_candidates.csv"


def test_cli_parses_multifactor_select_command() -> None:
    args = build_parser().parse_args(
        [
            "multifactor-select",
            "--db",
            "data/tushare/baiquant.db",
            "--as-of",
            "2026-07-01",
            "--lookback-days",
            "520",
            "--top",
            "15",
            "--output",
            "data/research/multifactor_candidates.csv",
        ]
    )

    assert args.command == "multifactor-select"
    assert args.db == "data/tushare/baiquant.db"
    assert args.as_of == "2026-07-01"
    assert args.lookback_days == 520
    assert args.top == 15
    assert args.output == "data/research/multifactor_candidates.csv"
    assert args.max_per_industry == 3
    assert args.max_lot_cost == 25000
    for column in ["score_rank", "multi_factor_score", "factor_hits", "positive_factors"]:
        assert column in args.columns.split(",")


def test_cli_parses_multifactor_backtest_command() -> None:
    args = build_parser().parse_args(
        [
            "multifactor-backtest",
            "--db",
            "data/tushare/baiquant.db",
            "--start",
            "2026-01-01",
            "--end",
            "2026-07-01",
            "--cash",
            "50000",
            "--top",
            "8",
            "--max-positions",
            "2",
            "--signal-every-n-days",
            "5",
            "--summary-output",
            "data/research/multifactor_backtest_summary.csv",
        ]
    )

    assert args.command == "multifactor-backtest"
    assert args.db == "data/tushare/baiquant.db"
    assert args.start == "2026-01-01"
    assert args.end == "2026-07-01"
    assert args.cash == 50_000
    assert args.top == 8
    assert args.max_positions == 2
    assert args.signal_every_n_days == 5
    assert args.summary_output == "data/research/multifactor_backtest_summary.csv"


def test_cli_multifactor_backtest_defaults_to_safer_manual_execution() -> None:
    args = build_parser().parse_args(
        [
            "multifactor-backtest",
            "--db",
            "data/tushare/baiquant.db",
            "--start",
            "2026-01-01",
            "--end",
            "2026-07-01",
        ]
    )

    assert args.cash == 50_000
    assert args.signal_every_n_days == 20
    assert args.max_positions == 1
    assert args.stop_loss_pct == pytest.approx(0.04)
    assert args.take_profit_pct == pytest.approx(0.25)
    assert args.add_trigger_pct == pytest.approx(0.0)
    assert args.ma_window == 8
    assert args.trailing_stop_activation_pct == pytest.approx(0.10)
    assert args.trailing_stop_pct == pytest.approx(0.06)
    assert args.max_holding_days == 15
    assert args.portfolio_stop_drawdown_pct == pytest.approx(0.04)
    assert args.portfolio_stop_cooldown_days == 8
    assert args.liquidate_on_portfolio_stop is True


def test_cli_parses_multifactor_plan_command() -> None:
    args = build_parser().parse_args(
        [
            "multifactor-plan",
            "--db",
            "data/tushare/baiquant.db",
            "--as-of",
            "2026-07-01",
            "--holdings-csv",
            "data/live/holdings.csv",
            "--cash",
            "50000",
            "--output",
            "data/live/multifactor_plan.csv",
        ]
    )

    assert args.command == "multifactor-plan"
    assert args.db == "data/tushare/baiquant.db"
    assert args.as_of == "2026-07-01"
    assert args.holdings_csv == "data/live/holdings.csv"
    assert args.cash == 50_000
    assert args.output == "data/live/multifactor_plan.csv"
    assert args.max_positions == 1
    assert args.stop_loss_pct == pytest.approx(0.04)
    assert args.take_profit_pct == pytest.approx(0.25)


def test_cli_parses_multifactor_diagnostics_command() -> None:
    args = build_parser().parse_args(
        [
            "multifactor-diagnostics",
            "--db",
            "data/tushare/baiquant.db",
            "--start",
            "2026-01-01",
            "--end",
            "2026-07-01",
            "--holding-days",
            "5",
            "--signal-every-n-days",
            "10",
            "--output",
            "data/research/multifactor_factor_diagnostics.csv",
            "--factor-output",
            "data/research/multifactor_factor_history.csv",
        ]
    )

    assert args.command == "multifactor-diagnostics"
    assert args.db == "data/tushare/baiquant.db"
    assert args.start == "2026-01-01"
    assert args.end == "2026-07-01"
    assert args.holding_days == 5
    assert args.signal_every_n_days == 10
    assert args.output == "data/research/multifactor_factor_diagnostics.csv"
    assert args.factor_output == "data/research/multifactor_factor_history.csv"


def test_cli_parses_multifactor_weight_profile_options() -> None:
    args = build_parser().parse_args(
        [
            "multifactor-backtest",
            "--db",
            "data/tushare/baiquant.db",
            "--start",
            "2026-01-01",
            "--end",
            "2026-07-01",
            "--weights-profile",
            "validated",
            "--diagnostics-csv",
            "data/research/multifactor_factor_diagnostics_2026.csv",
            "--min-factor-coverage",
            "0.8",
            "--min-factor-observations",
            "3",
        ]
    )

    assert args.weights_profile == "validated"
    assert args.diagnostics_csv == "data/research/multifactor_factor_diagnostics_2026.csv"
    assert args.min_factor_coverage == pytest.approx(0.8)
    assert args.min_factor_observations == 3


def test_cli_parses_multifactor_validate_periods_command() -> None:
    args = build_parser().parse_args(
        [
            "multifactor-validate-periods",
            "--db",
            "data/tushare/baiquant.db",
            "--periods",
            "2024:2024-01-01:2024-12-31,2025:2025-01-01:2025-12-31",
            "--summary-output",
            "data/research/multifactor_period_validation.csv",
            "--weights-profile",
            "validated",
            "--diagnostics-csv",
            "data/research/multifactor_factor_diagnostics_2026.csv",
        ]
    )

    assert args.command == "multifactor-validate-periods"
    assert args.periods == "2024:2024-01-01:2024-12-31,2025:2025-01-01:2025-12-31"
    assert args.summary_output == "data/research/multifactor_period_validation.csv"
    assert args.weights_profile == "validated"


def test_cli_parses_multifactor_regime_gate_options() -> None:
    args = build_parser().parse_args(
        [
            "multifactor-validate-periods",
            "--db",
            "data/tushare/baiquant.db",
            "--periods",
            "2026YTD:2026-01-01:2026-07-01",
            "--regime-gate",
            "--allowed-regimes",
            "bull,broad_rebound,structural",
        ]
    )

    assert args.regime_gate is True
    assert args.allowed_regimes == "bull,broad_rebound,structural"


def test_cli_multifactor_regime_gate_defaults_allow_weak_range() -> None:
    args = build_parser().parse_args(
        [
            "multifactor-validate-periods",
            "--db",
            "data/tushare/baiquant.db",
            "--periods",
            "2026YTD:2026-01-01:2026-07-01",
            "--regime-gate",
        ]
    )

    assert args.allowed_regimes == "bull,broad_rebound,structural,weak_range"


def test_multifactor_select_command_scores_candidates_and_writes_csv(tmp_path, monkeypatch, capsys) -> None:
    output_path = tmp_path / "multifactor_candidates.csv"
    calls = {}

    def load_window(
        self,  # noqa: ANN001
        as_of,
        lookback_days=540,
        fundamentals_lookback_days=None,
        events_lookback_days=None,
        money_flow_lookback_days=None,
    ):
        calls["as_of"] = pd.Timestamp(as_of)
        calls["lookback_days"] = lookback_days
        calls["fundamentals_lookback_days"] = fundamentals_lookback_days
        calls["events_lookback_days"] = events_lookback_days
        calls["money_flow_lookback_days"] = money_flow_lookback_days
        return MarketDataBundle(prices=pd.DataFrame([{"date": pd.Timestamp(as_of), "code": "000001.SZ"}]))

    def build_factors(bundle, as_of):  # noqa: ANN001
        calls["factor_as_of"] = pd.Timestamp(as_of)
        assert not bundle.prices.empty
        return pd.DataFrame(
            [
                {
                    "date": pd.Timestamp(as_of),
                    "code": "000001.SZ",
                    "name": "平安银行",
                    "industry": "银行",
                    "close": 10.2,
                    "momentum_5d": 0.08,
                    "money_flow_pct": 3.0,
                }
            ]
        )

    def score_factors(factors, factor_weights, top_n=None):  # noqa: ANN001
        calls["score_top_n"] = top_n
        assert "momentum_5d" in factor_weights
        assert not factors.empty
        return pd.DataFrame(
            [
                {
                    "score_rank": 1,
                    "code": "000001.SZ",
                    "name": "平安银行",
                    "industry": "银行",
                    "close": 10.2,
                    "multi_factor_score": 2.345,
                    "factor_hits": 3,
                    "positive_factors": "momentum_5d|money_flow_pct",
                }
            ]
        )

    def select_candidates(scored, top_n, max_per_industry=None, max_price=None, max_lot_cost=None):  # noqa: ANN001
        calls["selection_top_n"] = top_n
        calls["max_per_industry"] = max_per_industry
        calls["max_price"] = max_price
        calls["max_lot_cost"] = max_lot_cost
        return scored.head(top_n).assign(candidate_rank=range(1, min(top_n, len(scored)) + 1))

    monkeypatch.setattr("baiquant.cli.SqliteDataProvider.load_window", load_window)
    monkeypatch.setattr("baiquant.cli.build_multifactor_frame", build_factors, raising=False)
    monkeypatch.setattr("baiquant.cli.score_multifactor_frame", score_factors, raising=False)
    monkeypatch.setattr("baiquant.cli.select_multifactor_candidates", select_candidates, raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "baiquant",
            "multifactor-select",
            "--db",
            "data/tushare/baiquant.db",
            "--as-of",
            "2026-07-01",
            "--lookback-days",
            "520",
            "--top",
            "1",
            "--output",
            str(output_path),
        ],
    )

    main()

    assert calls == {
        "as_of": pd.Timestamp("2026-07-01"),
        "lookback_days": 520,
        "fundamentals_lookback_days": 520,
        "events_lookback_days": 30,
        "money_flow_lookback_days": 520,
        "factor_as_of": pd.Timestamp("2026-07-01"),
        "score_top_n": None,
        "selection_top_n": 1,
        "max_per_industry": 3,
        "max_price": None,
        "max_lot_cost": 25000,
    }
    output = capsys.readouterr().out
    assert "多因子分" in output
    assert "正向因子" in output
    saved = pd.read_csv(output_path)
    assert saved["code"].tolist() == ["000001.SZ"]


def test_multifactor_select_command_uses_validated_diagnostic_weights(tmp_path, monkeypatch) -> None:
    diagnostics_path = tmp_path / "diagnostics.csv"
    pd.DataFrame(
        [
            {
                "factor": "momentum_20d",
                "factor_coverage_rate": 1.0,
                "mean_rank_ic": 0.05,
                "ic_observations": 6,
                "top_1_5_mean_forward_return": 0.02,
            },
            {
                "factor": "money_flow_pct",
                "factor_coverage_rate": 0.95,
                "mean_rank_ic": -0.02,
                "ic_observations": 6,
                "top_1_5_mean_forward_return": 0.12,
            },
            {
                "factor": "roe",
                "factor_coverage_rate": 0.0,
                "mean_rank_ic": pd.NA,
                "ic_observations": 0,
                "top_1_5_mean_forward_return": pd.NA,
            },
        ]
    ).to_csv(diagnostics_path, index=False)
    seen = {}

    def load_window(self, as_of, **kwargs):  # noqa: ANN001
        return MarketDataBundle(prices=pd.DataFrame([{"date": pd.Timestamp(as_of), "code": "000001.SZ"}]))

    def build_factors(bundle, as_of):  # noqa: ANN001
        return pd.DataFrame([{"date": pd.Timestamp(as_of), "code": "000001.SZ", "close": 10.0}])

    def score_factors(factors, factor_weights, top_n=None):  # noqa: ANN001
        seen["weights"] = factor_weights
        return pd.DataFrame(
            [{"score_rank": 1, "code": "000001.SZ", "name": "平安银行", "industry": "银行", "close": 10.0, "multi_factor_score": 1.0}]
        )

    def select_candidates(scored, top_n, max_per_industry=None, max_price=None, max_lot_cost=None):  # noqa: ANN001
        return scored.assign(candidate_rank=1)

    monkeypatch.setattr("baiquant.cli.SqliteDataProvider.load_window", load_window)
    monkeypatch.setattr("baiquant.cli.build_multifactor_frame", build_factors, raising=False)
    monkeypatch.setattr("baiquant.cli.score_multifactor_frame", score_factors, raising=False)
    monkeypatch.setattr("baiquant.cli.select_multifactor_candidates", select_candidates, raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "baiquant",
            "multifactor-select",
            "--db",
            "data/tushare/baiquant.db",
            "--as-of",
            "2026-07-01",
            "--weights-profile",
            "validated",
            "--diagnostics-csv",
            str(diagnostics_path),
        ],
    )

    main()

    assert seen["weights"]["momentum_20d"] == pytest.approx(1.0)
    assert seen["weights"]["money_flow_pct"] == pytest.approx(0.7)
    assert seen["weights"]["roe"] == 0.0


def test_multifactor_plan_command_builds_action_plan_from_holdings(tmp_path, monkeypatch, capsys) -> None:
    holdings_path = tmp_path / "holdings.csv"
    output_path = tmp_path / "plan.csv"
    pd.DataFrame([{"code": "000001.SZ", "shares": 100, "average_cost": 10.0}]).to_csv(holdings_path, index=False)
    calls = {}

    def load_window(
        self,  # noqa: ANN001
        as_of,
        lookback_days=540,
        fundamentals_lookback_days=None,
        events_lookback_days=None,
        money_flow_lookback_days=None,
    ):
        calls["as_of"] = pd.Timestamp(as_of)
        calls["lookback_days"] = lookback_days
        calls["fundamentals_lookback_days"] = fundamentals_lookback_days
        calls["events_lookback_days"] = events_lookback_days
        calls["money_flow_lookback_days"] = money_flow_lookback_days
        return MarketDataBundle(prices=pd.DataFrame([{"date": pd.Timestamp(as_of), "code": "000001.SZ", "close": 9.5}]))

    def build_factors(bundle, as_of):  # noqa: ANN001
        return pd.DataFrame([{"date": pd.Timestamp(as_of), "code": "000002.SZ", "close": 20.0}])

    def score_factors(factors, factor_weights, top_n=None):  # noqa: ANN001
        return pd.DataFrame(
            [{"score_rank": 1, "code": "000002.SZ", "name": "万科A", "industry": "地产", "close": 20.0, "multi_factor_score": 2.0}]
        )

    def select_candidates(scored, top_n, max_per_industry=None, max_price=None, max_lot_cost=None):  # noqa: ANN001
        return scored.assign(candidate_rank=1)

    def build_plan(candidates, prices, as_of, holdings=None, cash=None, config=None):  # noqa: ANN001
        calls["plan_as_of"] = pd.Timestamp(as_of)
        calls["cash"] = cash
        calls["max_positions"] = config.max_positions
        assert candidates["code"].tolist() == ["000002.SZ"]
        assert holdings["code"].tolist() == ["000001.SZ"]
        return pd.DataFrame(
            [
                {
                    "date": pd.Timestamp(as_of),
                    "action": "sell_next_open",
                    "code": "000001.SZ",
                    "name": "平安银行",
                    "reason": "single_stop",
                    "shares": 100,
                    "reference_price": 9.5,
                    "cash_budget": 0,
                }
            ]
        )

    monkeypatch.setattr("baiquant.cli.SqliteDataProvider.load_window", load_window)
    monkeypatch.setattr("baiquant.cli.build_multifactor_frame", build_factors, raising=False)
    monkeypatch.setattr("baiquant.cli.score_multifactor_frame", score_factors, raising=False)
    monkeypatch.setattr("baiquant.cli.select_multifactor_candidates", select_candidates, raising=False)
    monkeypatch.setattr("baiquant.cli.build_multifactor_daily_plan", build_plan, raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "baiquant",
            "multifactor-plan",
            "--db",
            "data/tushare/baiquant.db",
            "--as-of",
            "2026-07-01",
            "--holdings-csv",
            str(holdings_path),
            "--cash",
            "50000",
            "--output",
            str(output_path),
        ],
    )

    main()

    assert calls == {
        "as_of": pd.Timestamp("2026-07-01"),
        "lookback_days": 520,
        "fundamentals_lookback_days": 520,
        "events_lookback_days": 30,
        "money_flow_lookback_days": 520,
        "plan_as_of": pd.Timestamp("2026-07-01"),
        "cash": 50_000,
        "max_positions": 1,
    }
    output = capsys.readouterr().out
    assert "操作" in output
    assert "次日卖出" in output
    saved = pd.read_csv(output_path)
    assert saved["action"].tolist() == ["sell_next_open"]


def test_multifactor_plan_command_blocks_new_buys_when_regime_gate_is_off(tmp_path, monkeypatch) -> None:
    holdings_path = tmp_path / "holdings.csv"
    pd.DataFrame([{"code": "000001.SZ", "shares": 100, "average_cost": 10.0}]).to_csv(holdings_path, index=False)
    seen = {}

    def load_window(self, as_of, **kwargs):  # noqa: ANN001
        return MarketDataBundle(prices=pd.DataFrame([{"date": pd.Timestamp(as_of), "code": "000001.SZ", "close": 9.5}]))

    def build_factors(bundle, as_of):  # noqa: ANN001
        return pd.DataFrame([{"date": pd.Timestamp(as_of), "code": "000002.SZ", "close": 20.0}])

    def score_factors(factors, factor_weights, top_n=None):  # noqa: ANN001
        return pd.DataFrame(
            [{"score_rank": 1, "code": "000002.SZ", "name": "万科A", "industry": "地产", "close": 20.0, "multi_factor_score": 2.0}]
        )

    def select_candidates(scored, top_n, max_per_industry=None, max_price=None, max_lot_cost=None):  # noqa: ANN001
        return scored.assign(candidate_rank=1)

    def build_regime(bundle_prices):  # noqa: ANN001
        return pd.DataFrame([{"date": pd.Timestamp("2026-07-01"), "regime": "bear_weak"}])

    def build_plan(candidates, prices, as_of, holdings=None, cash=None, config=None):  # noqa: ANN001
        seen["candidate_rows"] = len(candidates)
        seen["holding_rows"] = len(holdings)
        return pd.DataFrame(
            [
                {
                    "date": pd.Timestamp(as_of),
                    "action": "hold",
                    "code": "000001.SZ",
                    "reason": "existing_position",
                    "shares": 100,
                }
            ]
        )

    monkeypatch.setattr("baiquant.cli.SqliteDataProvider.load_window", load_window)
    monkeypatch.setattr("baiquant.cli.build_multifactor_frame", build_factors, raising=False)
    monkeypatch.setattr("baiquant.cli.score_multifactor_frame", score_factors, raising=False)
    monkeypatch.setattr("baiquant.cli.select_multifactor_candidates", select_candidates, raising=False)
    monkeypatch.setattr("baiquant.cli.build_regime_frame", build_regime, raising=False)
    monkeypatch.setattr("baiquant.cli.build_multifactor_daily_plan", build_plan, raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "baiquant",
            "multifactor-plan",
            "--db",
            "data/tushare/baiquant.db",
            "--as-of",
            "2026-07-01",
            "--holdings-csv",
            str(holdings_path),
            "--regime-gate",
            "--allowed-regimes",
            "structural",
        ],
    )

    main()

    assert seen == {"candidate_rows": 0, "holding_rows": 1}


def test_multifactor_diagnostics_command_evaluates_factor_history(tmp_path, monkeypatch, capsys) -> None:
    output_path = tmp_path / "diagnostics.csv"
    factor_path = tmp_path / "factor_history.csv"
    calls = {}

    def load_window(
        self,  # noqa: ANN001
        as_of,
        lookback_days=540,
        fundamentals_lookback_days=None,
        events_lookback_days=None,
        money_flow_lookback_days=None,
    ):
        calls["as_of"] = pd.Timestamp(as_of)
        calls["lookback_days"] = lookback_days
        calls["fundamentals_lookback_days"] = fundamentals_lookback_days
        calls["events_lookback_days"] = events_lookback_days
        calls["money_flow_lookback_days"] = money_flow_lookback_days
        return MarketDataBundle(
            prices=pd.DataFrame(
                [
                    {"date": "2026-01-01", "code": "AAA", "close": 10.0},
                    {"date": "2026-01-02", "code": "AAA", "close": 11.0},
                    {"date": "2026-01-01", "code": "BBB", "close": 10.0},
                    {"date": "2026-01-02", "code": "BBB", "close": 9.0},
                ]
            )
        )

    def generate_history(bundle, start, end, signal_every_n_days=5, factor_names=None):  # noqa: ANN001
        calls["history_start"] = pd.Timestamp(start)
        calls["history_end"] = pd.Timestamp(end)
        calls["history_step"] = signal_every_n_days
        assert factor_names == ["momentum_5d", "money_flow_pct"]
        assert not bundle.prices.empty
        return pd.DataFrame(
            [
                {"date": pd.Timestamp("2026-01-01"), "code": "AAA", "momentum_5d": 2.0, "money_flow_pct": 1.0},
                {"date": pd.Timestamp("2026-01-01"), "code": "BBB", "momentum_5d": 1.0, "money_flow_pct": 2.0},
            ]
        )

    def forward_returns(prices, signal_dates, holding_days):  # noqa: ANN001
        calls["signal_dates"] = signal_dates
        calls["holding_days"] = holding_days
        return pd.DataFrame(
            [
                {"date": pd.Timestamp("2026-01-01"), "code": "AAA", "forward_return": 0.10},
                {"date": pd.Timestamp("2026-01-01"), "code": "BBB", "forward_return": -0.05},
            ]
        )

    monkeypatch.setattr("baiquant.cli.SqliteDataProvider.load_window", load_window)
    monkeypatch.setattr("baiquant.cli.generate_multifactor_factor_history", generate_history, raising=False)
    monkeypatch.setattr("baiquant.cli.compute_forward_returns", forward_returns, raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "baiquant",
            "multifactor-diagnostics",
            "--db",
            "data/tushare/baiquant.db",
            "--start",
            "2026-01-01",
            "--end",
            "2026-01-02",
            "--holding-days",
            "1",
            "--signal-every-n-days",
            "1",
            "--factors",
            "momentum_5d,money_flow_pct",
            "--output",
            str(output_path),
            "--factor-output",
            str(factor_path),
        ],
    )

    main()

    assert calls["as_of"] == pd.Timestamp("2026-01-02")
    assert calls["lookback_days"] == 521
    assert calls["history_start"] == pd.Timestamp("2026-01-01")
    assert calls["history_end"] == pd.Timestamp("2026-01-02")
    assert calls["history_step"] == 1
    assert calls["signal_dates"] == [pd.Timestamp("2026-01-01")]
    assert calls["holding_days"] == 1
    output = capsys.readouterr().out
    assert "multifactor_diagnostics" in output
    assert "rank_ic" in output
    diagnostics = pd.read_csv(output_path)
    assert diagnostics["factor"].tolist() == ["momentum_5d", "money_flow_pct"]
    assert factor_path.exists()


def test_multifactor_backtest_command_generates_signals_and_writes_outputs(tmp_path, monkeypatch, capsys) -> None:
    summary_path = tmp_path / "summary.csv"
    signals_path = tmp_path / "signals.csv"
    trades_path = tmp_path / "trades.csv"
    equity_path = tmp_path / "equity.csv"
    calls = {}

    def load_window(
        self,  # noqa: ANN001
        as_of,
        lookback_days=540,
        fundamentals_lookback_days=None,
        events_lookback_days=None,
        money_flow_lookback_days=None,
    ):
        calls["as_of"] = pd.Timestamp(as_of)
        calls["lookback_days"] = lookback_days
        calls["fundamentals_lookback_days"] = fundamentals_lookback_days
        calls["events_lookback_days"] = events_lookback_days
        calls["money_flow_lookback_days"] = money_flow_lookback_days
        return MarketDataBundle(
            prices=pd.DataFrame(
                [
                    {"date": "2026-01-01", "code": "000001.SZ", "open": 10.0, "close": 10.0},
                    {"date": "2026-01-02", "code": "000001.SZ", "open": 10.0, "close": 10.5},
                    {"date": "2026-01-02", "code": "000002.SZ", "open": 20.0, "close": 20.5},
                ]
            )
        )

    def generate_signals(bundle, **kwargs):  # noqa: ANN001
        calls["signal_kwargs"] = kwargs
        assert not bundle.prices.empty
        return pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-01-01"),
                    "code": "000001.SZ",
                    "candidate_rank": 1,
                    "position_scale": 1.0,
                }
            ]
        )

    def run_backtest(prices, signals, config):  # noqa: ANN001
        calls["backtest_initial_cash"] = config.initial_cash
        calls["backtest_max_positions"] = config.max_positions
        assert not prices.empty
        assert set(prices["code"].astype(str)) == {"000001.SZ"}
        assert signals["code"].tolist() == ["000001.SZ"]

        class Result:
            metrics = {"total_return": 0.12, "max_drawdown": -0.03, "sharpe": 1.5}
            daily_returns = pd.DataFrame(
                [{"date": pd.Timestamp("2026-01-02"), "equity": 56_000, "position_value": 30_000}]
            )
            equity_curve = pd.DataFrame([{"date": pd.Timestamp("2026-01-02"), "equity": 56_000}])
            trades = pd.DataFrame([{"date": pd.Timestamp("2026-01-02"), "code": "000001.SZ", "status": "filled"}])
            positions = pd.DataFrame([{"code": "000001.SZ", "shares": 100}])

        return Result()

    monkeypatch.setattr("baiquant.cli.SqliteDataProvider.load_window", load_window)
    monkeypatch.setattr("baiquant.cli.generate_multifactor_signals", generate_signals, raising=False)
    monkeypatch.setattr("baiquant.cli.run_multi_position_trend_backtest", run_backtest, raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "baiquant",
            "multifactor-backtest",
            "--db",
            "data/tushare/baiquant.db",
            "--start",
            "2026-01-01",
            "--end",
            "2026-01-02",
            "--cash",
            "50000",
            "--top",
            "5",
            "--max-positions",
            "2",
            "--summary-output",
            str(summary_path),
            "--signals-output",
            str(signals_path),
            "--trades-output",
            str(trades_path),
            "--equity-output",
            str(equity_path),
        ],
    )

    main()

    assert calls["as_of"] == pd.Timestamp("2026-01-02")
    assert calls["lookback_days"] == 521
    assert calls["fundamentals_lookback_days"] == 521
    assert calls["money_flow_lookback_days"] == 521
    assert calls["signal_kwargs"]["start"] == pd.Timestamp("2026-01-01")
    assert calls["signal_kwargs"]["top_n"] == 5
    assert calls["backtest_initial_cash"] == 50_000
    assert calls["backtest_max_positions"] == 2
    output = capsys.readouterr().out
    assert "multifactor_backtest" in output
    assert "total_return=12.00%" in output
    assert pd.read_csv(summary_path)["total_return"].iloc[0] == pytest.approx(0.12)
    assert pd.read_csv(signals_path)["code"].tolist() == ["000001.SZ"]
    assert trades_path.exists()
    assert equity_path.exists()


def test_multifactor_validate_periods_command_writes_period_summary(tmp_path, monkeypatch, capsys) -> None:
    summary_path = tmp_path / "period_summary.csv"
    calls = {"signals": [], "backtests": 0}

    def load_window(
        self,  # noqa: ANN001
        as_of,
        lookback_days=540,
        fundamentals_lookback_days=None,
        events_lookback_days=None,
        money_flow_lookback_days=None,
    ):
        calls["as_of"] = pd.Timestamp(as_of)
        calls["lookback_days"] = lookback_days
        calls["fundamentals_lookback_days"] = fundamentals_lookback_days
        calls["money_flow_lookback_days"] = money_flow_lookback_days
        return MarketDataBundle(
            prices=pd.DataFrame(
                [
                    {"date": "2024-01-01", "code": "000001.SZ", "open": 10.0, "close": 10.0},
                    {"date": "2024-01-02", "code": "000001.SZ", "open": 10.0, "close": 10.5},
                    {"date": "2025-01-01", "code": "000002.SZ", "open": 20.0, "close": 20.0},
                    {"date": "2025-01-02", "code": "000002.SZ", "open": 20.0, "close": 19.0},
                ]
            )
        )

    def generate_signals(bundle, **kwargs):  # noqa: ANN001
        calls["signals"].append(kwargs)
        if kwargs.get("regimes") is not None:
            assert "regime" in kwargs["regimes"].columns
            assert kwargs["allowed_regimes"] == ("structural",)
        code = "000001.SZ" if pd.Timestamp(kwargs["start"]).year == 2024 else "000002.SZ"
        return pd.DataFrame(
            [
                {
                    "date": pd.Timestamp(kwargs["start"]),
                    "code": code,
                    "candidate_rank": 1,
                    "position_scale": 1.0,
                }
            ]
        )

    def run_backtest(prices, signals, config):  # noqa: ANN001
        calls["backtests"] += 1
        return_value = 0.10 if signals["code"].iloc[0] == "000001.SZ" else -0.05

        class Result:
            metrics = {
                "total_return": return_value,
                "annualized_return": return_value,
                "volatility": 0.2,
                "sharpe": 1.0 if return_value > 0 else -0.5,
                "max_drawdown": -0.03 if return_value > 0 else -0.08,
            }
            daily_returns = pd.DataFrame()
            equity_curve = pd.DataFrame([{"date": signals["date"].iloc[0], "equity": 50_000 * (1 + return_value)}])
            trades = pd.DataFrame([{"date": signals["date"].iloc[0], "code": signals["code"].iloc[0], "status": "filled"}])
            positions = pd.DataFrame()

        return Result()

    monkeypatch.setattr("baiquant.cli.SqliteDataProvider.load_window", load_window)
    monkeypatch.setattr("baiquant.cli.generate_multifactor_signals", generate_signals, raising=False)
    monkeypatch.setattr("baiquant.cli.run_multi_position_trend_backtest", run_backtest, raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "baiquant",
            "multifactor-validate-periods",
            "--db",
            "data/tushare/baiquant.db",
            "--periods",
            "2024:2024-01-01:2024-01-02,2025:2025-01-01:2025-01-02",
            "--summary-output",
            str(summary_path),
            "--regime-gate",
            "--allowed-regimes",
            "structural",
        ],
    )

    main()

    assert calls["as_of"] == pd.Timestamp("2025-01-02")
    assert calls["lookback_days"] == 887
    assert [call["start"] for call in calls["signals"]] == [pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")]
    assert calls["backtests"] == 2
    output = capsys.readouterr().out
    assert "multifactor_validate_periods periods=2" in output
    summary = pd.read_csv(summary_path)
    assert summary["period"].astype(str).tolist() == ["2024", "2025"]
    assert summary["total_return"].tolist() == pytest.approx([0.10, -0.05])
    assert summary["filled_trades"].tolist() == [1, 1]


def test_live20k_orders_command_exits_cleanly_when_report_is_not_ready(tmp_path, monkeypatch) -> None:
    plan_path = tmp_path / "plan.csv"
    report_path = tmp_path / "report.csv"
    output_path = tmp_path / "orders.csv"
    pd.DataFrame(
        [{"date": "2026-01-01", "action": "buy_next_open", "code": "A", "shares": 100}]
    ).to_csv(plan_path, index=False)
    pd.DataFrame([{"ready_for_live": False, "blocking_reason": "paper_days<20"}]).to_csv(report_path, index=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "baiquant",
            "live20k-orders",
            "--plan-csv",
            str(plan_path),
            "--report-csv",
            str(report_path),
            "--output",
            str(output_path),
        ],
    )

    with pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == "live20k-orders refused: paper report is not ready for live orders: paper_days<20"
    assert not output_path.exists()


def test_live20k_2026_command_uses_sqlite_window_load(monkeypatch, capsys) -> None:
    calls = {}

    def fail_full_load(self):  # noqa: ANN001
        raise AssertionError("live20k-2026 should not load the full SQLite database")

    def load_window(
        self,  # noqa: ANN001
        as_of,
        lookback_days=540,
        fundamentals_lookback_days=None,
        events_lookback_days=None,
        money_flow_lookback_days=None,
    ):
        calls["as_of"] = pd.Timestamp(as_of)
        calls["lookback_days"] = lookback_days
        calls["fundamentals_lookback_days"] = fundamentals_lookback_days
        calls["events_lookback_days"] = events_lookback_days
        calls["money_flow_lookback_days"] = money_flow_lookback_days
        return MarketDataBundle(
            prices=pd.DataFrame(
                [
                    {
                        "date": pd.Timestamp("2026-05-25"),
                        "code": "000001.SZ",
                        "open": 10.0,
                        "high": 10.5,
                        "low": 9.8,
                        "close": 10.2,
                        "volume": 1000,
                        "amount": 10_200,
                        "paused": 0,
                        "limit_up": 0,
                        "limit_down": 0,
                    }
                ]
            )
        )

    def build_regime(prices):  # noqa: ANN001
        return pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-05-25"),
                    "breadth_ma20": 0.7,
                    "market_equity": 1.0,
                    "market_ma60": 1.0,
                }
            ]
        )

    def build_plan(*args, **kwargs):  # noqa: ANN002, ANN003
        return pd.DataFrame([{"action": "wait"}])

    monkeypatch.setattr("baiquant.cli.SqliteDataProvider.load", fail_full_load)
    monkeypatch.setattr("baiquant.cli.SqliteDataProvider.load_window", load_window)
    monkeypatch.setattr("baiquant.cli.build_market_regime", build_regime)
    monkeypatch.setattr("baiquant.cli.build_live20k_daily_plan", build_plan)
    monkeypatch.setattr(
        "sys.argv",
        [
            "baiquant",
            "live20k-2026",
            "--db",
            "data/tushare/baiquant.db",
            "--as-of",
            "2026-05-25",
        ],
    )

    main()

    assert calls == {
        "as_of": pd.Timestamp("2026-05-25"),
        "lookback_days": 420,
        "fundamentals_lookback_days": 10,
        "events_lookback_days": 10,
        "money_flow_lookback_days": 10,
    }
    assert "No new positions." in capsys.readouterr().out


def test_live20k_2026_prints_plain_chinese_headers(monkeypatch, capsys) -> None:
    def load_window(
        self,  # noqa: ANN001
        as_of,
        lookback_days=540,
        fundamentals_lookback_days=None,
        events_lookback_days=None,
        money_flow_lookback_days=None,
    ):
        return MarketDataBundle(
            prices=pd.DataFrame(
                [
                    {
                        "date": pd.Timestamp(as_of),
                        "code": "000001.SZ",
                        "open": 10.0,
                        "high": 10.5,
                        "low": 9.8,
                        "close": 10.2,
                        "volume": 1000,
                        "amount": 10_200,
                        "paused": 0,
                        "limit_up": 0,
                        "limit_down": 0,
                    }
                ]
            )
        )

    def build_regime(prices):  # noqa: ANN001
        return pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-05-28"),
                    "breadth_ma20": 0.7,
                    "market_equity": 1.0,
                    "market_ma60": 1.0,
                }
            ]
        )

    def build_plan(*args, **kwargs):  # noqa: ANN002, ANN003
        return pd.DataFrame(
            [
                {
                    "action": "buy_next_open",
                    "code": "000001.SZ",
                    "name": "平安银行",
                    "reason": "entry",
                    "shares": 100,
                    "reference_price": 10.2,
                    "score_rank": 1,
                    "tech_score": 88,
                    "tech_grade": "A",
                    "trade_advice": "正常买",
                    "position_scale": 1.0,
                    "risk_flags": "",
                    "cash_budget": 10000,
                }
            ]
        )

    def build_watchlist(*args, **kwargs):  # noqa: ANN002, ANN003
        return pd.DataFrame(
            [
                {
                    "date": "2026-05-28",
                    "candidate_action": "buy_candidate",
                    "code": "000001.SZ",
                    "name": "平安银行",
                    "industry": "银行",
                    "close": 10.2,
                    "score_rank": 1,
                    "tech_score": 88,
                    "tech_grade": "A",
                    "trade_advice": "正常买",
                    "position_scale": 1.0,
                    "risk_flags": "",
                }
            ]
        )

    monkeypatch.setattr("baiquant.cli.SqliteDataProvider.load_window", load_window)
    monkeypatch.setattr("baiquant.cli.build_market_regime", build_regime)
    monkeypatch.setattr("baiquant.cli.build_live20k_daily_plan", build_plan)
    monkeypatch.setattr("baiquant.cli.build_live20k_watchlist", build_watchlist)
    monkeypatch.setattr(
        "sys.argv",
        [
            "baiquant",
            "live20k-2026",
            "--db",
            "data/tushare/baiquant.db",
            "--as-of",
            "2026-05-28",
            "--watchlist",
        ],
    )

    main()

    output = capsys.readouterr().out
    assert "技术分" in output
    assert "档位" in output
    assert "买法" in output
    assert "仓位建议" in output
    assert "风险提示" in output
    assert "tech_score" not in output


def test_live20k_2026_legacy_manual_name_prints_canonical_preset(monkeypatch, capsys) -> None:
    def load_window(
        self,  # noqa: ANN001
        as_of,
        lookback_days=540,
        fundamentals_lookback_days=None,
        events_lookback_days=None,
        money_flow_lookback_days=None,
    ):
        return MarketDataBundle(
            prices=pd.DataFrame(
                [
                    {
                        "date": pd.Timestamp(as_of),
                        "code": "000001.SZ",
                        "open": 10.0,
                        "high": 10.5,
                        "low": 9.8,
                        "close": 10.2,
                        "volume": 1000,
                        "amount": 10_200,
                        "paused": 0,
                        "limit_up": 0,
                        "limit_down": 0,
                    }
                ]
            )
        )

    def build_regime(prices):  # noqa: ANN001
        return pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-05-28"),
                    "breadth_ma20": 0.24,
                    "market_equity": 0.99,
                    "market_ma60": 1.0,
                }
            ]
        )

    def build_plan(*args, **kwargs):  # noqa: ANN002, ANN003
        return pd.DataFrame([{"action": "wait"}])

    monkeypatch.setattr("baiquant.cli.SqliteDataProvider.load_window", load_window)
    monkeypatch.setattr("baiquant.cli.build_market_regime", build_regime)
    monkeypatch.setattr("baiquant.cli.build_live20k_daily_plan", build_plan)
    monkeypatch.setattr(
        "sys.argv",
        [
            "baiquant",
            "live20k-2026",
            "--db",
            "data/tushare/baiquant.db",
            "--as-of",
            "2026-05-28",
            "--preset",
            "收盘后20天主策略",
        ],
    )

    main()

    output = capsys.readouterr().out
    assert "入场=关" in output
    assert "宽度底线=关(25.00%)" in output
    assert "策略=20天稳打版" in output
    assert "No new positions." in output
