from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class LiveReadinessThresholds:
    min_windows: int = 3
    min_total_return: float = 0.0
    max_drawdown_floor: float = -0.20
    min_sharpe: float = 0.8


@dataclass(slots=True)
class LiveReadinessReport:
    status: str
    reasons: list[str]


def evaluate_live_readiness(
    summary: pd.DataFrame,
    thresholds: LiveReadinessThresholds | None = None,
) -> LiveReadinessReport:
    thresholds = thresholds or LiveReadinessThresholds()
    reasons: list[str] = []

    if len(summary) < thresholds.min_windows:
        reasons.append(f"windows {len(summary)} < required {thresholds.min_windows}")

    for row in summary.itertuples(index=False):
        window = str(getattr(row, "window"))
        total_return = float(getattr(row, "total_return"))
        sharpe = float(getattr(row, "sharpe"))
        max_drawdown = float(getattr(row, "max_drawdown"))
        if total_return < thresholds.min_total_return:
            reasons.append(
                f"{window} total_return {_pct(total_return)} < {_pct(thresholds.min_total_return)}"
            )
        if max_drawdown < thresholds.max_drawdown_floor:
            reasons.append(
                f"{window} max_drawdown {_pct(max_drawdown)} < {_pct(thresholds.max_drawdown_floor)}"
            )
        if sharpe < thresholds.min_sharpe:
            reasons.append(f"{window} sharpe {sharpe:.2f} < {thresholds.min_sharpe:.2f}")

    return LiveReadinessReport(status="PASS" if not reasons else "FAIL", reasons=reasons)


def _pct(value: float) -> str:
    return f"{value:.2%}"
