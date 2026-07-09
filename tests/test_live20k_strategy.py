from dataclasses import replace

import pandas as pd
import pytest

from baiquant.data.bundle import MarketDataBundle
from baiquant.scoring import FactorSpec
from baiquant.strategy.live20k import (
    Live20KSignalConfig,
    apply_live20k_paper_fills,
    assert_live20k_data_fresh,
    build_live20k_daily_plan,
    build_hot_industries,
    build_live20k_watchlist,
    export_live20k_orders,
    live100k_hotspot_manual_fixed_execution_config,
    live100k_hotspot_manual_fixed_signal_config,
    live100k_hotspot_turbo_execution_config,
    live100k_hotspot_turbo_signal_config,
    live20k_execution_config,
    build_market_regime,
    generate_live20k_signals,
    run_live20k_paper_replay,
    run_live20k_paper_step,
    slice_market_data_for_signal,
    summarize_live20k_paper_run,
)


def _strategy_bundle() -> MarketDataBundle:
    rows = []
    dates = pd.date_range("2026-01-01", periods=70, freq="D")
    for index, date in enumerate(dates, start=1):
        rows.append([date, "A", 10 + index * 0.1, 10 + index * 0.1, 10 + index * 0.1, 10 + index * 0.1, 2_000, 2_000_000, 0, 0, 0])
        rows.append([date, "B", 20 - index * 0.05, 20 - index * 0.05, 20 - index * 0.05, 20 - index * 0.05, 3_000, 3_000_000, 0, 0, 0])
        rows.append([date, "C", 8 + index * 0.2, 8 + index * 0.2, 8 + index * 0.2, 8 + index * 0.2, 4_000, 4_000_000, 0, 0, 0])
    prices = pd.DataFrame(
        rows,
        columns=["date", "code", "open", "high", "low", "close", "volume", "amount", "paused", "limit_up", "limit_down"],
    )
    stocks = pd.DataFrame(
        [
            ["A", "Alpha", "Tech", "2020-01-01", 0],
            ["B", "Beta", "Bank", "2020-01-01", 0],
            ["C", "Gamma", "Energy", "2020-01-01", 0],
        ],
        columns=["code", "name", "industry", "list_date", "is_st"],
    )
    stocks["list_date"] = pd.to_datetime(stocks["list_date"])
    return MarketDataBundle(prices=prices, stocks=stocks)


def _hotspot_bundle() -> MarketDataBundle:
    rows = []
    dates = pd.date_range("2026-01-01", periods=70, freq="D")
    for index, date in enumerate(dates, start=1):
        rows.append([date, "T1", 10 + index * 0.25, 10 + index * 0.25, 10 + index * 0.25, 10 + index * 0.25, 5_000 + index * 60, 8_000_000, 0, int(index > 65), 0])
        rows.append([date, "T2", 12 + index * 0.20, 12 + index * 0.20, 12 + index * 0.20, 12 + index * 0.20, 4_800 + index * 50, 7_000_000, 0, 0, 0])
        rows.append([date, "B1", 20 - index * 0.04, 20 - index * 0.04, 20 - index * 0.04, 20 - index * 0.04, 3_000, 6_000_000, 0, 0, 0])
        rows.append([date, "B2", 18 - index * 0.03, 18 - index * 0.03, 18 - index * 0.03, 18 - index * 0.03, 3_100, 6_000_000, 0, 0, 0])
    prices = pd.DataFrame(
        rows,
        columns=["date", "code", "open", "high", "low", "close", "volume", "amount", "paused", "limit_up", "limit_down"],
    )
    stocks = pd.DataFrame(
        [
            ["T1", "TechOne", "Tech", "2020-01-01", 0],
            ["T2", "TechTwo", "Tech", "2020-01-01", 0],
            ["B1", "BankOne", "Bank", "2020-01-01", 0],
            ["B2", "BankTwo", "Bank", "2020-01-01", 0],
        ],
        columns=["code", "name", "industry", "list_date", "is_st"],
    )
    stocks["list_date"] = pd.to_datetime(stocks["list_date"])
    return MarketDataBundle(prices=prices, stocks=stocks)


def _money_flow_hotspot_bundle() -> MarketDataBundle:
    bundle = _hotspot_bundle()
    prices = bundle.prices.copy()
    for offset, date in enumerate(pd.to_datetime(prices["date"].drop_duplicates()), start=1):
        for code, base, step in [("B1", 9.0, 0.24), ("B2", 11.0, 0.22)]:
            value = base + offset * step
            mask = (pd.to_datetime(prices["date"]) == date) & (prices["code"] == code)
            prices.loc[mask, ["open", "high", "low", "close"]] = value
    rows = []
    for date in pd.to_datetime(prices["date"].drop_duplicates()):
        for code in ["T1", "T2"]:
            rows.append([date, code, -1_000_000, 0, 0, -500_000, -500_000, -5.0, 0, 0, -2.5, -2.5, 10.0, 0.0])
        for code in ["B1", "B2"]:
            rows.append([date, code, 2_000_000, 0, 0, 1_000_000, 1_000_000, 8.0, 0, 0, 4.0, 4.0, 20.0, 0.0])
    money_flow = pd.DataFrame(
        rows,
        columns=[
            "date",
            "code",
            "main_net_inflow",
            "small_net_inflow",
            "medium_net_inflow",
            "large_net_inflow",
            "super_large_net_inflow",
            "main_net_inflow_pct",
            "small_net_inflow_pct",
            "medium_net_inflow_pct",
            "large_net_inflow_pct",
            "super_large_net_inflow_pct",
            "close",
            "pct_change",
        ],
    )
    return MarketDataBundle(prices=prices, stocks=bundle.stocks, money_flow=money_flow)


def _rotating_hotspot_bundle() -> MarketDataBundle:
    rows = []
    dates = pd.date_range("2026-01-01", periods=70, freq="D")
    for index, date in enumerate(dates, start=1):
        if index <= 66:
            old_price = 10 + index * 0.40
        else:
            old_price = {67: 35.0, 68: 34.0, 69: 33.0, 70: 32.5}[index]
        new_price = 12 + index * 0.05
        if index >= 67:
            new_price += (index - 66) * 0.60
        for code in ["O1", "O2"]:
            rows.append([date, code, old_price, old_price, old_price, old_price, 6_000, 8_000_000, 0, 0, 0])
        for code in ["N1", "N2"]:
            rows.append([date, code, new_price, new_price, new_price, new_price, 5_000, 8_000_000, 0, 0, 0])
    prices = pd.DataFrame(
        rows,
        columns=["date", "code", "open", "high", "low", "close", "volume", "amount", "paused", "limit_up", "limit_down"],
    )
    stocks = pd.DataFrame(
        [
            ["O1", "OldOne", "OldTech", "2020-01-01", 0],
            ["O2", "OldTwo", "OldTech", "2020-01-01", 0],
            ["N1", "NewOne", "NewEnergy", "2020-01-01", 0],
            ["N2", "NewTwo", "NewEnergy", "2020-01-01", 0],
        ],
        columns=["code", "name", "industry", "list_date", "is_st"],
    )
    stocks["list_date"] = pd.to_datetime(stocks["list_date"])
    return MarketDataBundle(prices=prices, stocks=stocks)


def test_market_regime_uses_equal_weight_market_proxy_and_breadth() -> None:
    regime = build_market_regime(_strategy_bundle().prices)

    latest = regime.iloc[-1]

    assert latest["market_equity"] > latest["market_ma60"]
    assert latest["breadth_ma20"] > 0


def test_slice_market_data_for_signal_keeps_recent_market_tables_and_all_stocks() -> None:
    bundle = _money_flow_hotspot_bundle()

    sliced = slice_market_data_for_signal(bundle, as_of="2026-03-11", lookback_days=10)

    assert sliced.prices["date"].min() == pd.Timestamp("2026-03-01")
    assert sliced.prices["date"].max() == pd.Timestamp("2026-03-11")
    assert sliced.money_flow["date"].min() == pd.Timestamp("2026-03-01")
    assert sliced.money_flow["date"].max() == pd.Timestamp("2026-03-11")
    assert len(sliced.stocks) == len(bundle.stocks)


def test_live20k_signals_are_blocked_when_market_gate_is_off() -> None:
    bundle = _strategy_bundle()

    signals = generate_live20k_signals(
        bundle,
        Live20KSignalConfig(
            factor_specs=[FactorSpec("momentum_20d", 1.0, 1)],
            min_amount=0,
            min_factor_hits=0,
            market_breadth_min=1.1,
            signal_limit=2,
        ),
    )

    assert signals.empty


def test_live20k_signals_rank_candidates_when_market_gate_is_on() -> None:
    bundle = _strategy_bundle()

    signals = generate_live20k_signals(
        bundle,
        Live20KSignalConfig(
            factor_specs=[FactorSpec("momentum_20d", 1.0, 1)],
            min_amount=0,
            min_factor_hits=0,
            rank_start=1,
            market_breadth_min=0.0,
            market_dist_ma60_max=10.0,
            signal_limit=2,
        ),
    )

    latest = signals.loc[signals["date"] == signals["date"].max()]
    assert latest["score_rank"].tolist() == [1, 2]
    assert latest["code"].tolist() == ["C", "A"]


def test_live20k_signals_filter_short_momentum_and_amount_ratio() -> None:
    bundle = _strategy_bundle()
    prices = bundle.prices.copy()
    latest_date = prices["date"].max()
    mask = (prices["date"] == latest_date) & (prices["code"] == "A")
    prices.loc[mask, ["open", "high", "low", "close"]] = 25.0
    prices.loc[mask, "amount"] = 20_000_000
    bundle = MarketDataBundle(prices=prices, stocks=bundle.stocks)

    signals = generate_live20k_signals(
        bundle,
        Live20KSignalConfig(
            factor_specs=[FactorSpec("momentum_5d", 1.0, 1)],
            min_amount=0,
            min_factor_hits=0,
            rank_start=1,
            market_breadth_min=0.0,
            market_dist_ma60_max=10.0,
            signal_limit=5,
            min_momentum_5d=0.05,
            min_amount_ratio_5d=2.0,
        ),
    )

    latest = signals.loc[signals["date"] == latest_date]
    assert latest["code"].tolist() == ["A"]


def test_live20k_signals_filter_overheated_entries_with_max_bounds() -> None:
    bundle = _strategy_bundle()
    prices = bundle.prices.copy()
    latest_date = prices["date"].max()
    mask = (prices["date"] == latest_date) & (prices["code"] == "A")
    prices.loc[mask, ["open", "high", "low", "close"]] = 16.0
    bundle = MarketDataBundle(prices=prices, stocks=bundle.stocks)

    signals = generate_live20k_signals(
        bundle,
        Live20KSignalConfig(
            factor_specs=[FactorSpec("momentum_20d", 1.0, 1)],
            min_amount=0,
            min_factor_hits=0,
            rank_start=1,
            signal_limit=5,
            apply_market_gate=False,
            min_momentum_20d=0.06,
            max_momentum_20d=0.20,
            min_momentum_5d=-0.10,
            max_momentum_5d=0.14,
            max_close_vs_20d_high=-0.035,
        ),
    )

    latest = signals.loc[signals["date"] == latest_date]
    assert latest["code"].tolist() == ["A"]


def test_live20k_signals_can_filter_to_mainline_industries() -> None:
    bundle = _strategy_bundle()

    signals = generate_live20k_signals(
        bundle,
        Live20KSignalConfig(
            factor_specs=[FactorSpec("momentum_20d", 1.0, 1)],
            min_amount=0,
            min_factor_hits=0,
            rank_start=1,
            signal_limit=3,
            market_breadth_min=0.0,
            market_dist_ma60_max=10.0,
            industry_allowlist=("Tech",),
        ),
    )

    assert set(signals["industry"]) == {"Tech"}
    assert set(signals["code"]) == {"A"}


def test_build_hot_industries_ranks_recent_industry_strength() -> None:
    hot = build_hot_industries(_hotspot_bundle(), top_n=1, min_stock_count=2)

    latest = hot.loc[hot["date"] == hot["date"].max()]

    assert latest["industry"].tolist() == ["Tech"]
    assert latest.iloc[0]["hot_rank"] == 1
    assert latest.iloc[0]["industry_momentum_20d"] > 0


def test_build_hot_industries_can_filter_tiny_industries() -> None:
    hot = build_hot_industries(_hotspot_bundle(), top_n=1, min_stock_count=3)

    assert hot.empty


def test_build_hot_industries_can_rank_with_money_flow_confirmation() -> None:
    hot = build_hot_industries(
        _money_flow_hotspot_bundle(),
        top_n=1,
        min_stock_count=2,
        use_money_flow=True,
    )

    latest = hot.loc[hot["date"] == hot["date"].max()]

    assert latest["industry"].tolist() == ["Bank"]
    assert "industry_money_flow_3d" in latest.columns
    assert latest.iloc[0]["industry_money_flow_3d"] > 0


def test_build_hot_industries_can_exclude_retreating_old_leaders() -> None:
    hot = build_hot_industries(
        _rotating_hotspot_bundle(),
        top_n=1,
        min_stock_count=2,
        prefer_early_strength=True,
        exclude_retreat=True,
    )

    latest = hot.loc[hot["date"] == hot["date"].max()]

    assert latest["industry"].tolist() == ["NewEnergy"]
    assert latest.iloc[0]["industry_momentum_3d"] > 0
    assert not bool(latest.iloc[0]["industry_retreat"])


def test_live20k_signals_can_filter_to_dynamic_hot_industries() -> None:
    bundle = _hotspot_bundle()

    signals = generate_live20k_signals(
        bundle,
        Live20KSignalConfig(
            factor_specs=[FactorSpec("momentum_20d", 1.0, 1)],
            min_amount=0,
            min_factor_hits=1,
            rank_start=1,
            signal_limit=5,
            market_breadth_min=0.0,
            market_dist_ma60_max=10.0,
            dynamic_hotspot=True,
            hotspot_top_n=1,
            hotspot_min_stock_count=2,
        ),
    )

    latest = signals.loc[signals["date"] == signals["date"].max()]
    assert set(latest["industry"]) == {"Tech"}


def test_live20k_signals_skip_retreating_dynamic_hot_industries() -> None:
    signals = generate_live20k_signals(
        _rotating_hotspot_bundle(),
        Live20KSignalConfig(
            factor_specs=[FactorSpec("momentum_20d", 1.0, 1)],
            min_amount=0,
            min_factor_hits=0,
            rank_start=1,
            signal_limit=5,
            apply_market_gate=False,
            dynamic_hotspot=True,
            hotspot_top_n=1,
            hotspot_min_stock_count=2,
            hotspot_prefer_early_strength=True,
            hotspot_exclude_retreat=True,
        ),
    )

    latest = signals.loc[signals["date"] == signals["date"].max()]
    assert set(latest["industry"]) == {"NewEnergy"}


def test_live20k_signals_can_require_soft_market_breadth_floor() -> None:
    bundle = _hotspot_bundle()

    signals = generate_live20k_signals(
        bundle,
        Live20KSignalConfig(
            factor_specs=[FactorSpec("momentum_20d", 1.0, 1)],
            min_amount=0,
            min_factor_hits=0,
            rank_start=1,
            signal_limit=5,
            apply_market_gate=False,
            market_breadth_floor=1.1,
            dynamic_hotspot=True,
            hotspot_top_n=1,
            hotspot_min_stock_count=2,
        ),
    )

    assert signals.empty


def test_live20k_signals_can_require_positive_money_flow() -> None:
    bundle = _money_flow_hotspot_bundle()

    signals = generate_live20k_signals(
        bundle,
        Live20KSignalConfig(
            factor_specs=[FactorSpec("money_flow_3d", 1.0, 1)],
            min_amount=0,
            min_factor_hits=0,
            rank_start=1,
            signal_limit=5,
            market_breadth_min=0.0,
            market_dist_ma60_max=10.0,
            min_money_flow_3d=0.0,
            min_big_order_3d=0.0,
        ),
    )

    latest = signals.loc[signals["date"] == signals["date"].max()]
    assert set(latest["industry"]) == {"Bank"}
    assert set(latest["code"]) == {"B1", "B2"}


def test_live20k_watchlist_keeps_candidates_when_market_gate_is_off() -> None:
    bundle = _money_flow_hotspot_bundle()

    watchlist = build_live20k_watchlist(
        bundle,
        as_of="2026-03-11",
        signal_config=Live20KSignalConfig(
            factor_specs=[FactorSpec("money_flow_3d", 1.0, 1)],
            min_amount=0,
            min_factor_hits=0,
            rank_start=1,
            signal_limit=5,
            market_breadth_min=1.1,
            market_dist_ma60_max=10.0,
            min_money_flow_3d=0.0,
            min_big_order_3d=0.0,
        ),
    )

    assert set(watchlist["code"]) == {"B1", "B2"}
    assert watchlist["market_gate"].tolist() == [False, False]
    assert watchlist["candidate_action"].tolist() == ["watch_only", "watch_only"]


def test_live20k_watchlist_marks_candidates_buyable_when_market_gate_disabled() -> None:
    bundle = _money_flow_hotspot_bundle()

    watchlist = build_live20k_watchlist(
        bundle,
        as_of="2026-03-11",
        signal_config=Live20KSignalConfig(
            factor_specs=[FactorSpec("money_flow_3d", 1.0, 1)],
            min_amount=0,
            min_factor_hits=0,
            rank_start=1,
            signal_limit=5,
            market_breadth_min=1.1,
            market_dist_ma60_max=10.0,
            min_money_flow_3d=0.0,
            min_big_order_3d=0.0,
            apply_market_gate=False,
        ),
    )

    assert set(watchlist["code"]) == {"B1", "B2"}
    assert watchlist["market_gate"].tolist() == [False, False]
    assert watchlist["candidate_action"].tolist() == ["buy_candidate", "buy_candidate"]


def test_live20k_watchlist_includes_technical_overlay_fields() -> None:
    watchlist = build_live20k_watchlist(
        _money_flow_hotspot_bundle(),
        as_of="2026-03-11",
        signal_config=Live20KSignalConfig(
            min_amount=1_000_000,
            rank_start=1,
            signal_limit=4,
            apply_market_gate=False,
            min_factor_hits=0,
            factor_specs=[FactorSpec("momentum_20d", 1.0, 1)],
        ),
    )

    assert not watchlist.empty
    assert {"tech_score", "tech_grade", "trade_advice", "position_scale", "risk_flags"}.issubset(watchlist.columns)


def test_live20k_watchlist_uses_current_date_when_soft_breadth_floor_blocks_buys() -> None:
    bundle = _money_flow_hotspot_bundle()

    watchlist = build_live20k_watchlist(
        bundle,
        as_of="2026-03-11",
        signal_config=Live20KSignalConfig(
            factor_specs=[FactorSpec("money_flow_3d", 1.0, 1)],
            min_amount=0,
            min_factor_hits=0,
            rank_start=1,
            signal_limit=5,
            market_breadth_min=0.0,
            market_dist_ma60_max=10.0,
            min_money_flow_3d=0.0,
            min_big_order_3d=0.0,
            apply_market_gate=False,
            market_breadth_floor=1.1,
        ),
    )

    assert not watchlist.empty
    assert watchlist["date"].eq(pd.Timestamp("2026-03-11")).all()
    assert set(watchlist["code"]) == {"B1", "B2"}
    assert watchlist["candidate_action"].tolist() == ["watch_only", "watch_only"]


def test_live20k_signals_keep_raw_rank_after_skipping_front_ranks() -> None:
    bundle = _strategy_bundle()

    signals = generate_live20k_signals(
        bundle,
        Live20KSignalConfig(
            factor_specs=[FactorSpec("momentum_20d", 1.0, 1)],
            min_amount=0,
            min_factor_hits=1,
            rank_start=2,
            market_breadth_min=0.0,
            market_dist_ma60_max=10.0,
            signal_limit=1,
        ),
    )

    latest = signals.loc[signals["date"] == signals["date"].max()]
    assert latest["raw_rank"].tolist() == [2]
    assert latest["score_rank"].tolist() == [1]


def test_live20k_execution_config_matches_current_research_candidate() -> None:
    config = live20k_execution_config()

    assert config.initial_cash == 20_000
    assert config.max_positions == 2
    assert config.stop_loss_pct == 0.06
    assert config.take_profit_pct == 0.30
    assert config.ma_window == 10
    assert config.trailing_stop_activation_pct == 0.12
    assert config.trailing_stop_pct == 0.08
    assert config.portfolio_stop_drawdown_pct == 0.04
    assert config.portfolio_stop_cooldown_days == 5
    assert config.liquidate_on_portfolio_stop is True
    assert config.reset_peak_on_portfolio_stop is True
    assert config.reset_peak_min_profit_pct == 0.10


def test_live20k_default_selector_uses_raw_top_6_to_10_bucket() -> None:
    config = Live20KSignalConfig()

    assert config.rank_start == 6
    assert config.signal_limit == 5


def test_live100k_hotspot_turbo_signal_config_matches_current_best_candidate() -> None:
    config = live100k_hotspot_turbo_signal_config()

    assert config.apply_market_gate is False
    assert config.market_breadth_floor == 0.25
    assert config.dynamic_hotspot is True
    assert config.hotspot_prefer_early_strength is True
    assert config.hotspot_exclude_retreat is True
    assert config.hotspot_top_n == 8
    assert config.hotspot_min_stock_count == 10
    assert config.exclude_star is False
    assert config.exclude_chinext is False
    assert config.rank_start == 1
    assert config.signal_limit == 4
    assert config.min_momentum_20d == 0.15
    assert config.min_momentum_5d == 0.10
    assert config.min_close_position_20d == 0.80
    assert config.min_money_flow_3d == 2
    assert config.min_big_order_3d == 2
    assert {spec.name for spec in config.factor_specs}.issuperset({"hot_score", "momentum_5d", "amount_ratio_5d"})


def test_live100k_hotspot_turbo_execution_config_targets_high_return_variant() -> None:
    config = live100k_hotspot_turbo_execution_config()

    assert config.initial_cash == 100_000
    assert config.max_positions == 3
    assert config.cash_buffer_pct == 0.1
    assert config.stop_loss_pct == 0.06
    assert config.take_profit_pct == 0.0
    assert config.add_trigger_pct == 0.03
    assert config.add_position_multiple == 1.0
    assert config.ma_window == 0
    assert config.max_holding_days == 15
    assert config.trailing_stop_activation_pct == 0.05
    assert config.trailing_stop_pct == 0.05
    assert config.portfolio_stop_drawdown_pct == 0.10
    assert config.portfolio_stop_cooldown_days == 10
    assert config.liquidate_on_portfolio_stop is True
    assert config.reset_peak_on_portfolio_stop is True


def test_live100k_hotspot_manual_fixed_signal_config_reuses_turbo_selection() -> None:
    config = live100k_hotspot_manual_fixed_signal_config()

    assert config.apply_market_gate is False
    assert config.market_breadth_floor == 0.25
    assert config.dynamic_hotspot is True
    assert config.hotspot_prefer_early_strength is True
    assert config.hotspot_exclude_retreat is True
    assert config.hotspot_top_n == 8
    assert config.signal_limit == 4
    assert config.min_factor_hits == 4
    assert config.min_money_flow_3d == 2
    assert config.min_big_order_3d == 2
    assert config.min_momentum_20d == 0.15
    assert config.min_momentum_5d == 0.10
    assert config.min_close_position_20d == 0.80
    assert {spec.name for spec in config.factor_specs}.issuperset(
        {"hot_score", "money_flow_3d", "amount_ratio_5d", "momentum_5d", "close_position_20d"}
    )


def test_live100k_hotspot_manual_fixed_execution_config_matches_manual_20_day_variant() -> None:
    config = live100k_hotspot_manual_fixed_execution_config()

    assert config.initial_cash == 50_000
    assert config.max_positions == 3
    assert config.cash_buffer_pct == 0.1
    assert config.stop_loss_pct == 0.06
    assert config.add_trigger_pct == 0.0
    assert config.add_position_multiple == 0.0
    assert config.max_holding_days == 20
    assert config.trailing_stop_activation_pct == 0.0
    assert config.trailing_stop_pct == 0.0
    assert config.portfolio_stop_drawdown_pct == 0.10
    assert config.portfolio_stop_cooldown_days == 10


def test_live20k_signals_include_technical_overlay_fields() -> None:
    config = Live20KSignalConfig(
        min_amount=1_000_000,
        rank_start=1,
        signal_limit=4,
        apply_market_gate=False,
        min_factor_hits=0,
        factor_specs=[FactorSpec("momentum_20d", 1.0, 1)],
    )
    signals = generate_live20k_signals(
        _money_flow_hotspot_bundle(),
        config,
    )
    latest = signals.loc[signals["date"] == pd.Timestamp("2026-03-11")]

    assert not latest.empty
    assert {"tech_score", "tech_grade", "trade_advice", "position_scale", "risk_flags"}.issubset(signals.columns)
    assert latest["tech_score"].between(0, 100).all()
    assert set(latest["tech_grade"]).issubset({"A", "B", "C", "D"})
    assert set(latest["trade_advice"]).issubset({"正常买", "半仓买", "观察不买", "风险跳过"})
    assert latest["position_scale"].between(0, 1).all()


def test_live20k_technical_overlay_flags_long_upper_shadow_and_weak_flow() -> None:
    bundle = _money_flow_hotspot_bundle()
    prices = bundle.prices.copy()
    target = (prices["date"] == pd.Timestamp("2026-03-11")) & (prices["code"] == "B1")
    prices.loc[target, "high"] = prices.loc[target, "close"] * 1.18
    prices.loc[target, "open"] = prices.loc[target, "close"] * 0.98
    money_flow = bundle.money_flow.copy()
    flow_target = (money_flow["date"] >= pd.Timestamp("2026-03-09")) & (money_flow["code"] == "B1")
    money_flow.loc[flow_target, "main_net_inflow_pct"] = -5.0
    money_flow.loc[flow_target, "large_net_inflow_pct"] = -3.0
    money_flow.loc[flow_target, "super_large_net_inflow_pct"] = -2.0
    config = Live20KSignalConfig(
        min_amount=1_000_000,
        rank_start=1,
        signal_limit=4,
        apply_market_gate=False,
        min_factor_hits=0,
        factor_specs=[FactorSpec("momentum_20d", 1.0, 1)],
    )

    signals = generate_live20k_signals(
        MarketDataBundle(prices=prices, stocks=bundle.stocks, money_flow=money_flow),
        config,
    )
    row = signals.loc[(signals["date"] == pd.Timestamp("2026-03-11")) & (signals["code"] == "B1")].iloc[0]

    assert "长上影" in row["risk_flags"]
    assert "资金转弱" in row["risk_flags"]
    assert row["trade_advice"] in {"半仓买", "观察不买", "风险跳过"}
    assert row["position_scale"] <= 0.5


def test_live20k_data_freshness_allows_latest_or_historical_dates() -> None:
    bundle = _strategy_bundle()

    assert_live20k_data_fresh(bundle, "2026-03-11")
    assert_live20k_data_fresh(bundle, "2026-03-10")


def test_live20k_data_freshness_rejects_future_requested_dates() -> None:
    bundle = _strategy_bundle()

    with pytest.raises(ValueError, match="latest_price_date=2026-03-11"):
        assert_live20k_data_fresh(bundle, "2026-03-12")


def test_live20k_daily_plan_blocks_new_buys_when_market_gate_is_off() -> None:
    bundle = _strategy_bundle()

    plan = build_live20k_daily_plan(
        bundle,
        as_of="2026-03-11",
        signal_config=Live20KSignalConfig(
            factor_specs=[FactorSpec("momentum_20d", 1.0, 1)],
            min_amount=0,
            min_factor_hits=1,
            market_breadth_min=1.1,
            signal_limit=2,
        ),
        execution_config=replace(live20k_execution_config(), take_profit_pct=0.0),
    )

    assert plan["action"].tolist() == ["wait"]
    assert plan["reason"].tolist() == ["market_gate_off"]
    assert plan["market_gate"].tolist() == [False]


def test_live20k_daily_plan_uses_cash_override_as_peak_baseline() -> None:
    bundle = _strategy_bundle()

    plan = build_live20k_daily_plan(
        bundle,
        as_of="2026-03-11",
        cash=50_000,
        signal_config=Live20KSignalConfig(
            factor_specs=[FactorSpec("momentum_20d", 1.0, 1)],
            min_listed_days=0,
            min_history_days=1,
            min_amount=0,
            min_factor_hits=0,
            rank_start=1,
            signal_limit=1,
            apply_market_gate=False,
        ),
        execution_config=replace(live100k_hotspot_manual_fixed_execution_config(), max_positions=1),
    )

    assert "portfolio_stop" not in set(plan["reason"].astype(str))
    assert float(plan.iloc[0]["drawdown"]) == 0.0


def test_live20k_daily_plan_can_ignore_market_gate_for_turbo_preset() -> None:
    bundle = _strategy_bundle()

    plan = build_live20k_daily_plan(
        bundle,
        as_of="2026-03-11",
        signal_config=Live20KSignalConfig(
            factor_specs=[FactorSpec("momentum_20d", 1.0, 1)],
            min_amount=0,
            min_factor_hits=1,
            rank_start=1,
            market_breadth_min=1.1,
            signal_limit=1,
            apply_market_gate=False,
        ),
        execution_config=replace(live20k_execution_config(), take_profit_pct=0.0),
    )

    assert plan["action"].tolist() == ["buy_next_open"]
    assert plan["reason"].tolist() == ["entry"]


def test_live20k_daily_plan_creates_lot_sized_buy_orders_when_gate_is_on() -> None:
    bundle = _strategy_bundle()

    plan = build_live20k_daily_plan(
        bundle,
        as_of="2026-03-11",
        signal_config=Live20KSignalConfig(
            factor_specs=[FactorSpec("momentum_20d", 1.0, 1)],
            min_amount=0,
            min_factor_hits=1,
            rank_start=1,
            market_breadth_min=0.0,
            market_dist_ma60_max=10.0,
            signal_limit=2,
        ),
    )

    assert plan["action"].tolist() == ["buy_next_open", "buy_next_open"]
    assert plan["code"].tolist() == ["C", "A"]
    assert plan["shares"].tolist() == [400, 600]
    assert plan["reason"].tolist() == ["entry", "entry"]


def test_live20k_daily_plan_includes_technical_overlay_fields() -> None:
    config = Live20KSignalConfig(
        min_amount=1_000_000,
        rank_start=1,
        signal_limit=4,
        apply_market_gate=False,
        min_factor_hits=0,
        factor_specs=[FactorSpec("momentum_20d", 1.0, 1)],
    )
    plan = build_live20k_daily_plan(
        _money_flow_hotspot_bundle(),
        as_of="2026-03-11",
        cash=50_000,
        signal_config=config,
        execution_config=live100k_hotspot_manual_fixed_execution_config(),
    )
    buys = plan.loc[plan["action"] == "buy_next_open"]

    assert not buys.empty
    assert {"tech_score", "tech_grade", "trade_advice", "position_scale", "risk_flags"}.issubset(plan.columns)
    assert buys["tech_score"].notna().all()
    assert buys["trade_advice"].notna().all()


def test_live20k_daily_plan_supports_zero_ma_window_execution() -> None:
    bundle = _strategy_bundle()

    plan = build_live20k_daily_plan(
        bundle,
        as_of="2026-03-11",
        signal_config=Live20KSignalConfig(
            factor_specs=[FactorSpec("momentum_20d", 1.0, 1)],
            min_amount=0,
            min_factor_hits=1,
            rank_start=1,
            market_breadth_min=0.0,
            market_dist_ma60_max=10.0,
            signal_limit=1,
        ),
        execution_config=replace(live20k_execution_config(), ma_window=0, take_profit_pct=0.0),
    )

    assert plan["action"].tolist() == ["buy_next_open"]


def test_live20k_daily_plan_adds_half_after_profit_trigger() -> None:
    bundle = _strategy_bundle()
    holdings = pd.DataFrame(
        [
            {
                "code": "A",
                "shares": 1000,
                "average_cost": 16.0,
                "high_close": 17.0,
                "entry_shares": 1000,
                "added": False,
            }
        ]
    )

    plan = build_live20k_daily_plan(
        bundle,
        as_of="2026-03-11",
        holdings=holdings,
        cash=10_000,
        signal_config=Live20KSignalConfig(
            factor_specs=[FactorSpec("momentum_20d", 1.0, 1)],
            min_amount=0,
            min_factor_hits=1,
            rank_start=1,
            market_breadth_min=0.0,
            market_dist_ma60_max=10.0,
            signal_limit=1,
        ),
        execution_config=replace(
            live20k_execution_config(),
            max_positions=1,
            take_profit_pct=0.0,
            add_trigger_pct=0.05,
            add_position_multiple=0.5,
            ma_window=0,
            trailing_stop_activation_pct=0.0,
            trailing_stop_pct=0.0,
        ),
    )

    assert plan["action"].tolist() == ["buy_next_open"]
    assert plan["code"].tolist() == ["A"]
    assert plan["reason"].tolist() == ["profit_add"]
    assert plan["shares"].tolist() == [500]


def test_live20k_daily_plan_exits_after_max_holding_days() -> None:
    bundle = _strategy_bundle()
    holdings = pd.DataFrame(
        [
            {
                "code": "A",
                "shares": 1000,
                "average_cost": 15.0,
                "high_close": 17.0,
                "entry_date": "2026-03-10",
            }
        ]
    )

    plan = build_live20k_daily_plan(
        bundle,
        as_of="2026-03-11",
        holdings=holdings,
        signal_config=Live20KSignalConfig(
            factor_specs=[FactorSpec("momentum_20d", 1.0, 1)],
            min_amount=0,
            min_factor_hits=1,
            rank_start=1,
            market_breadth_min=0.0,
            market_dist_ma60_max=10.0,
            signal_limit=1,
        ),
        execution_config=replace(
            live20k_execution_config(),
            max_holding_days=2,
            take_profit_pct=0.0,
            ma_window=0,
            trailing_stop_activation_pct=0.0,
            trailing_stop_pct=0.0,
        ),
    )

    sell_plan = plan.loc[plan["action"] == "sell_next_open"]
    assert sell_plan["code"].tolist() == ["A"]
    assert sell_plan["reason"].tolist() == ["max_holding_days"]


def test_live20k_daily_plan_keeps_exit_orders_when_market_gate_is_off() -> None:
    bundle = _strategy_bundle()
    holdings = pd.DataFrame([{"code": "A", "shares": 300, "average_cost": 20.0}])

    plan = build_live20k_daily_plan(
        bundle,
        as_of="2026-03-11",
        holdings=holdings,
        signal_config=Live20KSignalConfig(
            factor_specs=[FactorSpec("momentum_20d", 1.0, 1)],
            min_amount=0,
            min_factor_hits=1,
            market_breadth_min=1.1,
            signal_limit=2,
        ),
        execution_config=replace(live20k_execution_config(), take_profit_pct=0.0),
    )

    assert plan["action"].tolist() == ["sell_next_open"]
    assert plan["code"].tolist() == ["A"]
    assert plan["reason"].tolist() == ["stop_loss"]
    assert plan["market_gate"].tolist() == [False]


def test_live20k_daily_plan_uses_planned_sell_value_for_next_entries() -> None:
    bundle = _strategy_bundle()
    holdings = pd.DataFrame([{"code": "A", "shares": 300, "average_cost": 20.0}])

    plan = build_live20k_daily_plan(
        bundle,
        as_of="2026-03-11",
        holdings=holdings,
        cash=0,
        signal_config=Live20KSignalConfig(
            factor_specs=[FactorSpec("momentum_20d", 1.0, 1)],
            min_amount=0,
            min_factor_hits=1,
            rank_start=1,
            market_breadth_min=0.0,
            signal_limit=2,
        ),
        equity_peak=5100,
        execution_config=replace(live20k_execution_config(), max_positions=1),
    )

    assert plan["action"].tolist() == ["sell_next_open", "buy_next_open"]
    assert plan["code"].tolist()[0] == "A"
    assert plan["reason"].tolist()[0] == "stop_loss"
    assert plan["shares"].tolist()[1] >= 100


def test_live20k_daily_plan_exits_after_activated_trailing_stop() -> None:
    bundle = _strategy_bundle()
    latest_close = bundle.prices.loc[bundle.prices["code"] == "A", "close"].iloc[-1]
    holdings = pd.DataFrame(
        [{"code": "A", "shares": 100, "average_cost": 10.0, "high_close": latest_close * 1.15}]
    )

    plan = build_live20k_daily_plan(
        bundle,
        as_of="2026-03-11",
        holdings=holdings,
        signal_config=Live20KSignalConfig(
            factor_specs=[FactorSpec("momentum_20d", 1.0, 1)],
            min_amount=0,
            min_factor_hits=1,
            market_breadth_min=1.1,
            signal_limit=2,
        ),
        execution_config=replace(live20k_execution_config(), take_profit_pct=0.0),
    )

    assert plan["action"].tolist() == ["sell_next_open"]
    assert plan["code"].tolist() == ["A"]
    assert plan["reason"].tolist() == ["trailing_stop"]


def test_live20k_daily_plan_exits_after_take_profit() -> None:
    bundle = _strategy_bundle()
    holdings = pd.DataFrame([{"code": "A", "shares": 100, "average_cost": 13.0}])

    plan = build_live20k_daily_plan(
        bundle,
        as_of="2026-03-11",
        holdings=holdings,
        equity_peak=1700,
        signal_config=Live20KSignalConfig(
            factor_specs=[FactorSpec("momentum_20d", 1.0, 1)],
            min_amount=0,
            min_factor_hits=1,
            market_breadth_min=1.1,
            signal_limit=2,
        ),
        execution_config=replace(live20k_execution_config(), take_profit_pct=0.20),
    )

    assert plan["action"].tolist() == ["sell_next_open"]
    assert plan["code"].tolist() == ["A"]
    assert plan["reason"].tolist() == ["take_profit"]


def test_apply_live20k_paper_fills_buys_next_open_and_updates_state() -> None:
    bundle = _strategy_bundle()
    plan = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-01"),
                "action": "buy_next_open",
                "code": "A",
                "name": "Alpha",
                "reason": "entry",
                "shares": 100,
            }
        ]
    )

    result = apply_live20k_paper_fills(bundle, plan, cash=20_000, equity_peak=20_000)

    assert result.fills["status"].tolist() == ["filled"]
    assert result.fills["date"].tolist() == [pd.Timestamp("2026-01-02")]
    assert result.fills["side"].tolist() == ["buy"]
    assert result.holdings["code"].tolist() == ["A"]
    assert result.holdings["shares"].tolist() == [100]
    assert result.cash == 18_978.46949
    assert result.equity == 19_998.46949
    assert result.equity_peak == 20_000
    assert result.holdings["entry_date"].tolist() == [pd.Timestamp("2026-01-02")]


def test_apply_live20k_paper_fills_marks_profit_add_done() -> None:
    bundle = _strategy_bundle()
    holdings = pd.DataFrame(
        [
            {
                "code": "A",
                "shares": 1000,
                "average_cost": 16.0,
                "high_close": 17.0,
                "entry_shares": 1000,
                "added": False,
                "entry_date": "2026-01-01",
            }
        ]
    )
    plan = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-01"),
                "action": "buy_next_open",
                "code": "A",
                "name": "Alpha",
                "reason": "profit_add",
                "shares": 500,
            }
        ]
    )

    result = apply_live20k_paper_fills(bundle, plan, holdings=holdings, cash=10_000, equity_peak=27_000)

    assert result.fills["status"].tolist() == ["filled"]
    assert result.holdings["shares"].tolist() == [1500]
    assert result.holdings["entry_shares"].tolist() == [1000]
    assert result.holdings["added"].tolist() == [True]
    assert result.holdings["entry_date"].tolist() == [pd.Timestamp("2026-01-01")]


def test_apply_live20k_paper_fills_sells_next_open_and_removes_holding() -> None:
    bundle = _strategy_bundle()
    holdings = pd.DataFrame([{"code": "A", "shares": 100, "average_cost": 10.0}])
    plan = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-01"),
                "action": "sell_next_open",
                "code": "A",
                "name": "Alpha",
                "reason": "stop_loss",
                "shares": 100,
            }
        ]
    )

    result = apply_live20k_paper_fills(bundle, plan, holdings=holdings, cash=1_000, equity_peak=2_000)

    assert result.fills["status"].tolist() == ["filled"]
    assert result.fills["side"].tolist() == ["sell"]
    assert result.holdings.empty
    assert result.cash == 2_018.47051
    assert result.equity == 2_018.47051
    assert result.equity_peak == 2_018.47051


def test_apply_live20k_paper_fills_resets_peak_after_profitable_portfolio_stop() -> None:
    bundle = _strategy_bundle()
    holdings = pd.DataFrame([{"code": "A", "shares": 100, "average_cost": 10.0}])
    plan = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-01"),
                "action": "sell_next_open",
                "code": "A",
                "name": "Alpha",
                "reason": "portfolio_stop",
                "shares": 100,
            }
        ]
    )

    result = apply_live20k_paper_fills(
        bundle,
        plan,
        holdings=holdings,
        cash=1_000,
        equity_peak=2_500,
        execution_config=replace(
            live20k_execution_config(),
            initial_cash=1_000,
            reset_peak_on_portfolio_stop=True,
            reset_peak_min_profit_pct=0.10,
        ),
    )

    assert result.holdings.empty
    assert result.equity == 2_018.47051
    assert result.equity_peak == 2_018.47051


def test_summarize_live20k_paper_run_marks_clean_20_day_run_ready() -> None:
    dates = pd.date_range("2026-01-01", periods=20, freq="B")
    plans = pd.DataFrame(
        {
            "date": dates,
            "action": ["wait"] * 20,
            "market_gate": [False] * 20,
        }
    )
    fills = pd.DataFrame(columns=["date", "status", "side"])
    states = pd.DataFrame(
        {
            "date": dates,
            "cash": [20_000] * 20,
            "equity": [20_000 + index * 10 for index in range(20)],
            "equity_peak": [20_000 + index * 10 for index in range(20)],
        }
    )

    report = summarize_live20k_paper_run(plans, fills, states, min_days=20)

    assert report.loc[0, "paper_days"] == 20
    assert report.loc[0, "failed_fills"] == 0
    assert report.loc[0, "rule_violations"] == 0
    assert report.loc[0, "ready_for_live"] is True
    assert report.loc[0, "blocking_reason"] == ""


def test_summarize_live20k_paper_run_blocks_short_or_dirty_runs() -> None:
    dates = pd.date_range("2026-01-01", periods=3, freq="B")
    plans = pd.DataFrame(
        {
            "date": dates,
            "action": ["buy_next_open", "wait", "wait"],
            "market_gate": [False, False, False],
        }
    )
    fills = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-02"),
                "status": "failed_limit_down",
                "side": "sell",
            }
        ]
    )
    states = pd.DataFrame(
        {
            "date": dates,
            "cash": [20_000, 20_000, 20_000],
            "equity": [20_000, 19_000, 18_500],
            "equity_peak": [20_000, 20_000, 20_000],
        }
    )

    report = summarize_live20k_paper_run(plans, fills, states, min_days=20)

    assert report.loc[0, "ready_for_live"] is False
    assert report.loc[0, "paper_days"] == 3
    assert report.loc[0, "failed_fills"] == 1
    assert report.loc[0, "rule_violations"] == 1
    assert report.loc[0, "max_drawdown"] == -0.075
    assert "paper_days<20" in report.loc[0, "blocking_reason"]
    assert "failed_fills" in report.loc[0, "blocking_reason"]
    assert "rule_violations" in report.loc[0, "blocking_reason"]
    assert "drawdown" in report.loc[0, "blocking_reason"]


def test_summarize_live20k_paper_run_does_not_block_failed_limit_up_buys() -> None:
    dates = pd.date_range("2026-01-01", periods=20, freq="B")
    plans = pd.DataFrame(
        {
            "date": dates,
            "action": ["wait"] * len(dates),
            "market_gate": [True] * len(dates),
        }
    )
    fills = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-02"),
                "status": "failed_limit_up",
                "side": "buy",
            }
        ]
    )
    states = pd.DataFrame(
        {
            "date": dates,
            "cash": [20_000] * len(dates),
            "equity": [20_000] * len(dates),
            "equity_peak": [20_000] * len(dates),
        }
    )

    report = summarize_live20k_paper_run(plans, fills, states, min_days=20)

    assert report.loc[0, "failed_fills"] == 1
    assert report.loc[0, "blocking_failed_fills"] == 0
    assert report.loc[0, "ready_for_live"] is True
    assert "failed_fills" not in report.loc[0, "blocking_reason"]


def test_summarize_live20k_paper_run_requires_order_days_when_requested() -> None:
    dates = pd.date_range("2026-01-01", periods=20, freq="B")
    plans = pd.DataFrame(
        {
            "date": dates,
            "action": ["wait"] * 20,
            "market_gate": [False] * 20,
        }
    )
    states = pd.DataFrame(
        {
            "date": dates,
            "cash": [20_000] * 20,
            "equity": [20_000] * 20,
            "equity_peak": [20_000] * 20,
        }
    )

    report = summarize_live20k_paper_run(plans, states=states, min_days=20, min_order_days=3)

    assert report.loc[0, "ready_for_live"] is False
    assert report.loc[0, "order_days"] == 0
    assert "order_days<3" in report.loc[0, "blocking_reason"]


def test_summarize_live20k_paper_run_blocks_negative_total_return_when_requested() -> None:
    dates = pd.date_range("2026-01-01", periods=20, freq="B")
    plans = pd.DataFrame(
        {
            "date": dates,
            "action": ["buy_next_open"] + ["wait"] * 19,
            "market_gate": [True] * 20,
        }
    )
    states = pd.DataFrame(
        {
            "date": dates,
            "cash": [20_000] * 20,
            "equity": [20_000] + [19_900] * 19,
            "equity_peak": [20_000] * 20,
        }
    )

    report = summarize_live20k_paper_run(
        plans,
        states=states,
        min_days=20,
        min_order_days=1,
        min_total_return=0.0,
    )

    assert report.loc[0, "ready_for_live"] is False
    assert report.loc[0, "total_return"] == -0.005
    assert "total_return<0.00%" in report.loc[0, "blocking_reason"]


def test_export_live20k_orders_refuses_when_paper_report_is_not_ready() -> None:
    plan = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-01"),
                "action": "buy_next_open",
                "code": "A",
                "name": "Alpha",
                "reason": "entry",
                "shares": 100,
            }
        ]
    )
    report = pd.DataFrame([{"ready_for_live": False, "blocking_reason": "paper_days<20"}])

    with pytest.raises(ValueError, match="paper_days<20"):
        export_live20k_orders(plan, report)


def test_export_live20k_orders_writes_only_trade_actions_when_ready() -> None:
    plan = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-01"),
                "action": "wait",
                "code": "",
                "name": "",
                "reason": "market_gate_off",
                "shares": 0,
            },
            {
                "date": pd.Timestamp("2026-01-02"),
                "action": "buy_next_open",
                "code": "A",
                "name": "Alpha",
                "reason": "entry",
                "shares": 100,
                "reference_price": 10.0,
                "score_rank": 1,
            },
            {
                "date": pd.Timestamp("2026-01-02"),
                "action": "sell_next_open",
                "code": "B",
                "name": "Beta",
                "reason": "stop_loss",
                "shares": 200,
                "reference_price": 20.0,
                "score_rank": pd.NA,
            },
        ]
    )
    report = pd.DataFrame([{"ready_for_live": True, "blocking_reason": ""}])

    orders = export_live20k_orders(plan, report)

    assert orders["side"].tolist() == ["buy", "sell"]
    assert orders["code"].tolist() == ["A", "B"]
    assert orders["shares"].tolist() == [100, 200]
    assert orders["source_action"].tolist() == ["buy_next_open", "sell_next_open"]


def test_run_live20k_paper_step_generates_first_day_plan_without_fill() -> None:
    bundle = _strategy_bundle()

    result = run_live20k_paper_step(bundle, as_of="2026-01-02")

    assert result.filled_previous_plan is False
    assert result.fills.empty
    assert result.holdings.empty
    assert result.state.loc[0, "date"] == pd.Timestamp("2026-01-02")
    assert result.state.loc[0, "cash"] == 20_000
    assert result.state.loc[0, "equity"] == 20_000
    assert result.report.loc[0, "paper_days"] == 1
    assert result.report.loc[0, "ready_for_live"] is False


def test_run_live20k_paper_step_accepts_signal_config() -> None:
    bundle = _strategy_bundle()

    result = run_live20k_paper_step(
        bundle,
        as_of="2026-03-11",
        signal_config=Live20KSignalConfig(
            factor_specs=[FactorSpec("momentum_20d", 1.0, 1)],
            min_amount=0,
            min_factor_hits=1,
            market_breadth_min=1.1,
        ),
    )

    assert result.plan["action"].tolist() == ["wait"]
    assert result.plan["reason"].tolist() == ["market_gate_off"]


def test_run_live20k_paper_step_fills_previous_plan_before_new_plan() -> None:
    bundle = _strategy_bundle()
    previous_plan = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-01"),
                "action": "buy_next_open",
                "code": "A",
                "name": "Alpha",
                "reason": "entry",
                "shares": 100,
            }
        ]
    )
    previous_state = pd.DataFrame([{"date": "2026-01-01", "cash": 20_000, "equity": 20_000, "equity_peak": 20_000}])

    result = run_live20k_paper_step(
        bundle,
        as_of="2026-01-02",
        previous_plan=previous_plan,
        state=previous_state,
    )

    assert result.filled_previous_plan is True
    assert result.fills["status"].tolist() == ["filled"]
    assert result.holdings["code"].tolist() == ["A"]
    assert result.holdings["shares"].tolist() == [100]
    assert result.state.loc[0, "cash"] == 18_978.46949
    assert result.state.loc[0, "equity"] == 19_998.46949
    assert result.state.loc[0, "equity_peak"] == 20_000
    assert result.plan.loc[0, "date"] == pd.Timestamp("2026-01-02")


def test_run_live20k_paper_replay_processes_each_trading_day_in_range() -> None:
    bundle = _strategy_bundle()

    result = run_live20k_paper_replay(
        bundle,
        start="2026-01-02",
        end="2026-01-04",
        signal_config=Live20KSignalConfig(
            factor_specs=[FactorSpec("momentum_20d", 1.0, 1)],
            min_amount=0,
            min_factor_hits=1,
            market_breadth_min=1.1,
        ),
        min_days=3,
    )

    assert result.plans["date"].drop_duplicates().tolist() == [
        pd.Timestamp("2026-01-02"),
        pd.Timestamp("2026-01-03"),
        pd.Timestamp("2026-01-04"),
    ]
    assert result.states["date"].tolist() == [
        pd.Timestamp("2026-01-02"),
        pd.Timestamp("2026-01-03"),
        pd.Timestamp("2026-01-04"),
    ]
    assert result.report.loc[0, "paper_days"] == 3
    assert result.report.loc[0, "ready_for_live"] is True
