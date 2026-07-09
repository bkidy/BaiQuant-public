import pandas as pd

from baiquant.data.bundle import MarketDataBundle
from baiquant.factors.registry import compute_factor


def test_money_flow_factors_use_latest_row_on_or_before_signal_date() -> None:
    money_flow = pd.DataFrame(
        [
            ["2026-01-02", "FAST", 1_000_000, 2.0, 1.0, 2.5],
            ["2026-01-03", "FAST", 2_000_000, 5.0, 2.0, 3.5],
            ["2026-01-03", "SLOW", -1_000_000, -3.0, -1.0, 0.5],
            ["2026-01-04", "SLOW", 9_000_000, 9.0, 8.0, 7.0],
        ],
        columns=[
            "date",
            "code",
            "main_net_inflow",
            "main_net_inflow_pct",
            "large_net_inflow_pct",
            "super_large_net_inflow_pct",
        ],
    )
    money_flow["date"] = pd.to_datetime(money_flow["date"])
    bundle = MarketDataBundle(prices=pd.DataFrame(), money_flow=money_flow)

    main_flow = compute_factor(bundle, "money_flow", pd.Timestamp("2026-01-03"))
    big_order = compute_factor(bundle, "big_order", pd.Timestamp("2026-01-03"))

    assert main_flow.to_dict() == {"FAST": 5.0, "SLOW": -3.0}
    assert big_order.to_dict() == {"FAST": 5.5, "SLOW": -0.5}
