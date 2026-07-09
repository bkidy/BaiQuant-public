import pandas as pd

from baiquant.live_readiness import LiveReadinessThresholds, evaluate_live_readiness


def test_live_readiness_fails_when_any_window_loses_money_or_drawdown_is_too_deep() -> None:
    summary = pd.DataFrame(
        [
            {"window": "2025FULL", "total_return": -0.16, "sharpe": -0.02, "max_drawdown": -0.29},
            {"window": "2026YTD", "total_return": 0.26, "sharpe": 2.3, "max_drawdown": -0.10},
        ]
    )

    report = evaluate_live_readiness(summary, LiveReadinessThresholds(min_windows=2))

    assert report.status == "FAIL"
    assert "2025FULL total_return -16.00% < 0.00%" in report.reasons
    assert "2025FULL max_drawdown -29.00% < -20.00%" in report.reasons


def test_live_readiness_passes_when_all_windows_clear_required_thresholds() -> None:
    summary = pd.DataFrame(
        [
            {"window": "2025H2", "total_return": 0.06, "sharpe": 0.9, "max_drawdown": -0.10},
            {"window": "2026YTD", "total_return": 0.08, "sharpe": 1.1, "max_drawdown": -0.12},
            {"window": "paper", "total_return": 0.02, "sharpe": 0.8, "max_drawdown": -0.03},
        ]
    )

    report = evaluate_live_readiness(summary, LiveReadinessThresholds(min_windows=3, min_sharpe=0.5))

    assert report.status == "PASS"
    assert report.reasons == []
