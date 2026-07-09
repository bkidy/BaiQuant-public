import pandas as pd
import pytest

from baiquant.research.factor_diagnostics import (
    classify_market_regimes,
    compute_forward_returns,
    evaluate_ranked_signal_quality,
    factor_rank_ic,
    summarize_signal_quality,
    top_slice_returns,
)


def test_compute_forward_returns_uses_trading_day_offset() -> None:
    prices = pd.DataFrame(
        [
            ["2026-01-01", "A", 10.0],
            ["2026-01-02", "A", 11.0],
            ["2026-01-03", "A", 12.0],
            ["2026-01-01", "B", 10.0],
            ["2026-01-02", "B", 9.0],
            ["2026-01-03", "B", 8.0],
        ],
        columns=["date", "code", "close"],
    )

    returns = compute_forward_returns(prices, [pd.Timestamp("2026-01-01")], holding_days=2)

    by_code = returns.set_index("code")["forward_return"].to_dict()
    assert by_code["A"] == pytest.approx(0.2)
    assert by_code["B"] == pytest.approx(-0.2)


def test_classify_market_regimes_maps_current_context_to_action() -> None:
    regime = pd.DataFrame(
        [
            ["2026-01-01", 1.10, 1.00, 0.65],
            ["2026-01-02", 1.05, 1.00, 0.35],
            ["2026-01-03", 1.25, 1.00, 0.70],
            ["2026-01-04", 0.95, 1.00, 0.30],
        ],
        columns=["date", "market_equity", "market_ma60", "breadth_ma20"],
    )

    labelled = classify_market_regimes(regime)

    by_date = labelled.set_index("date")
    assert by_date.loc[pd.Timestamp("2026-01-01"), "market_regime"] == "broad_uptrend"
    assert by_date.loc[pd.Timestamp("2026-01-01"), "regime_action"] == "trade_top_6_10"
    assert by_date.loc[pd.Timestamp("2026-01-02"), "market_regime"] == "weak_breadth"
    assert by_date.loc[pd.Timestamp("2026-01-02"), "regime_action"] == "watch_only"
    assert by_date.loc[pd.Timestamp("2026-01-03"), "market_regime"] == "overheated_uptrend"
    assert by_date.loc[pd.Timestamp("2026-01-04"), "market_regime"] == "downtrend"


def test_factor_rank_ic_is_positive_when_factor_orders_forward_returns() -> None:
    factors = pd.DataFrame(
        [
            ["2026-01-01", "A", 3.0],
            ["2026-01-01", "B", 2.0],
            ["2026-01-01", "C", 1.0],
        ],
        columns=["date", "code", "momentum"],
    )
    forward = pd.DataFrame(
        [
            ["2026-01-01", "A", 0.3],
            ["2026-01-01", "B", 0.1],
            ["2026-01-01", "C", -0.1],
        ],
        columns=["date", "code", "forward_return"],
    )
    factors["date"] = pd.to_datetime(factors["date"])
    forward["date"] = pd.to_datetime(forward["date"])

    report = factor_rank_ic(factors, forward, "momentum")

    assert report.loc[0, "rank_ic"] == 1.0
    assert report.loc[0, "count"] == 3


def test_factor_rank_ic_skips_constant_cross_sections() -> None:
    factors = pd.DataFrame(
        [
            ["2026-01-01", "A", 1.0],
            ["2026-01-01", "B", 1.0],
            ["2026-01-01", "C", 1.0],
        ],
        columns=["date", "code", "momentum"],
    )
    forward = pd.DataFrame(
        [
            ["2026-01-01", "A", 0.3],
            ["2026-01-01", "B", 0.1],
            ["2026-01-01", "C", -0.1],
        ],
        columns=["date", "code", "forward_return"],
    )
    factors["date"] = pd.to_datetime(factors["date"])
    forward["date"] = pd.to_datetime(forward["date"])

    report = factor_rank_ic(factors, forward, "momentum")

    assert report.empty


def test_top_slice_returns_can_compare_top_1_5_and_top_6_10() -> None:
    rows = []
    returns = []
    for index in range(1, 11):
        rows.append(["2026-01-01", f"S{index}", 11 - index])
        returns.append(["2026-01-01", f"S{index}", index / 100])
    factors = pd.DataFrame(rows, columns=["date", "code", "score"])
    forward = pd.DataFrame(returns, columns=["date", "code", "forward_return"])
    factors["date"] = pd.to_datetime(factors["date"])
    forward["date"] = pd.to_datetime(forward["date"])

    slices = top_slice_returns(
        factors,
        forward,
        "score",
        slices=[("top_1_5", 1, 5), ("top_6_10", 6, 10)],
    )

    by_slice = slices.set_index("slice")["mean_forward_return"].to_dict()
    assert by_slice["top_1_5"] == pytest.approx(0.03)
    assert by_slice["top_6_10"] == pytest.approx(0.08)


def test_evaluate_ranked_signal_quality_measures_retained_gain_and_drawdown() -> None:
    prices = pd.DataFrame(
        [
            ["2026-01-01", "A", 10.0],
            ["2026-01-02", "A", 12.0],
            ["2026-01-03", "A", 11.0],
            ["2026-01-01", "B", 10.0],
            ["2026-01-02", "B", 11.0],
            ["2026-01-03", "B", 9.5],
        ],
        columns=["date", "code", "close"],
    )
    signals = pd.DataFrame(
        [
            ["2026-01-01", "A", 1, 2.0],
            ["2026-01-01", "B", 6, 1.5],
        ],
        columns=["date", "code", "score_rank", "score"],
    )

    details = evaluate_ranked_signal_quality(
        signals,
        prices,
        horizons=[2],
        stable_drawdown_floor=-0.08,
    )

    by_code = details.set_index("code")
    assert by_code.loc["A", "rank_bucket"] == "top_1_5"
    assert by_code.loc["A", "forward_return"] == pytest.approx(0.1)
    assert by_code.loc["A", "max_gain"] == pytest.approx(0.2)
    assert by_code.loc["A", "max_drawdown"] == pytest.approx(0.0)
    assert by_code.loc["A", "gain_retention"] == pytest.approx(0.5)
    assert bool(by_code.loc["A", "stable_gain"])

    assert by_code.loc["B", "rank_bucket"] == "top_6_10"
    assert by_code.loc["B", "forward_return"] == pytest.approx(-0.05)
    assert by_code.loc["B", "max_gain"] == pytest.approx(0.1)
    assert by_code.loc["B", "max_drawdown"] == pytest.approx(-0.05)
    assert by_code.loc["B", "gain_retention"] == pytest.approx(0.0)
    assert not bool(by_code.loc["B", "stable_gain"])


def test_evaluate_ranked_signal_quality_buckets_by_raw_rank_when_available() -> None:
    prices = pd.DataFrame(
        [
            ["2026-01-01", "A", 10.0],
            ["2026-01-02", "A", 11.0],
        ],
        columns=["date", "code", "close"],
    )
    signals = pd.DataFrame(
        [["2026-01-01", "A", 6, 1, 2.0]],
        columns=["date", "code", "raw_rank", "score_rank", "score"],
    )

    details = evaluate_ranked_signal_quality(signals, prices, horizons=[1])

    assert details.loc[0, "raw_rank"] == 6
    assert details.loc[0, "score_rank"] == 1
    assert details.loc[0, "rank_bucket"] == "top_6_10"


def test_summarize_signal_quality_groups_by_horizon_and_rank_bucket() -> None:
    details = pd.DataFrame(
        [
            ["2026-01-01", "A", 1, "top_1_5", 5, 0.10, 0.20, 0.00, 0.50, True],
            ["2026-01-01", "B", 2, "top_1_5", 5, -0.05, 0.02, -0.10, -2.50, False],
        ],
        columns=[
            "date",
            "code",
            "score_rank",
            "rank_bucket",
            "horizon_days",
            "forward_return",
            "max_gain",
            "max_drawdown",
            "gain_retention",
            "stable_gain",
        ],
    )

    summary = summarize_signal_quality(details)

    row = summary.iloc[0]
    assert row["horizon_days"] == 5
    assert row["rank_bucket"] == "top_1_5"
    assert row["count"] == 2
    assert row["positive_rate"] == pytest.approx(0.5)
    assert row["stable_rate"] == pytest.approx(0.5)
    assert row["mean_forward_return"] == pytest.approx(0.025)
