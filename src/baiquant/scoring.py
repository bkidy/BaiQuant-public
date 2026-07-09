from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class FactorSpec:
    """Configuration for one factor in the cross-sectional score."""

    name: str
    weight: float = 1.0
    direction: int = 1
    enabled: bool = True


def robust_zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() < 2:
        return pd.Series(np.zeros(len(numeric)), index=numeric.index, dtype=float)
    clipped = numeric.clip(numeric.quantile(0.01), numeric.quantile(0.99))
    mean = clipped.mean()
    std = clipped.std(ddof=0)
    if not std or np.isnan(std):
        return pd.Series(np.zeros(len(numeric)), index=numeric.index, dtype=float)
    return (clipped - mean) / std


def score_factor_frame(factors: pd.DataFrame, specs: list[FactorSpec]) -> pd.DataFrame:
    scored = factors.copy()
    scored["score"] = 0.0
    scored["hits"] = 0

    for spec in [item for item in specs if item.enabled]:
        if spec.name not in scored.columns:
            scored[f"{spec.name}_raw"] = np.nan
            scored[f"{spec.name}_score"] = 0.0
            continue
        raw = scored[spec.name]
        factor_score = robust_zscore(raw) * spec.direction * spec.weight
        factor_score = factor_score.fillna(0.0)
        scored[f"{spec.name}_raw"] = raw
        scored[f"{spec.name}_score"] = factor_score
        scored["score"] = scored["score"] + factor_score
        scored["hits"] = scored["hits"] + (factor_score > 0).astype(int)

    return scored.sort_values(["score", "hits", "code"], ascending=[False, False, True]).reset_index(drop=True)
