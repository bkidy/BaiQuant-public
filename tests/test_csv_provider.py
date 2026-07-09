from pathlib import Path

import pandas as pd

from baiquant.data.csv_provider import CsvDataProvider


def test_csv_provider_loads_required_tables_with_parsed_dates(tmp_path: Path) -> None:
    (tmp_path / "prices.csv").write_text(
        "\n".join(
            [
                "date,code,open,high,low,close,volume,amount,paused,limit_up,limit_down",
                "2026-01-03,000001.SZ,10,11,9,10.5,1000,10500,0,0,0",
                "2026-01-02,000001.SZ,9,10,8,9.5,900,8550,0,0,0",
            ]
        )
    )
    (tmp_path / "fundamentals.csv").write_text(
        "\n".join(
            [
                "date,code,pe_ttm,pb,roe,revenue_yoy,profit_yoy",
                "2026-01-03,000001.SZ,12,1.5,0.12,0.2,0.3",
            ]
        )
    )
    (tmp_path / "stocks.csv").write_text(
        "\n".join(
            [
                "code,name,industry,list_date,is_st",
                "000001.SZ,PingAn,Bank,1991-04-03,0",
            ]
        )
    )
    (tmp_path / "events.csv").write_text(
        "\n".join(
            [
                "date,code,event_type,sentiment",
                "2026-01-03,000001.SZ,buyback,positive",
            ]
        )
    )

    bundle = CsvDataProvider(tmp_path).load()

    assert list(bundle.prices["date"]) == [
        pd.Timestamp("2026-01-02"),
        pd.Timestamp("2026-01-03"),
    ]
    assert bundle.fundamentals.loc[0, "roe"] == 0.12
    assert bundle.stocks.loc[0, "list_date"] == pd.Timestamp("1991-04-03")
    assert bundle.events.loc[0, "event_type"] == "buyback"
