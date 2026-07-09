import pandas as pd

from baiquant.factors.fundamental import quality


def test_quality_uses_latest_informative_fundamental_row() -> None:
    fundamentals = pd.DataFrame(
        [
            ["2026-03-31", "000001.SZ", pd.NA, pd.NA, 0.08, pd.NA, -0.04],
            ["2026-05-25", "000001.SZ", pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
        ],
        columns=["date", "code", "pe_ttm", "pb", "roe", "revenue_yoy", "profit_yoy"],
    )
    fundamentals["date"] = pd.to_datetime(fundamentals["date"])

    result = quality(fundamentals, pd.Timestamp("2026-05-25"))

    assert result["000001.SZ"] == 0.08 + 0.5 * -0.04
