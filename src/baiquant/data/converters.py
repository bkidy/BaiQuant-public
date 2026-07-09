from __future__ import annotations

from pathlib import Path

from baiquant.data.csv_provider import CsvDataProvider
from baiquant.data.sqlite_provider import write_market_data_to_sqlite


def convert_csv_to_sqlite(csv_root: str | Path, sqlite_path: str | Path) -> dict[str, int]:
    bundle = CsvDataProvider(csv_root).load()
    db_path = Path(sqlite_path)
    if db_path.exists():
        db_path.unlink()
    write_market_data_to_sqlite(db_path, bundle)
    return {
        "prices": len(bundle.prices),
        "fundamentals": len(bundle.fundamentals),
        "stocks": len(bundle.stocks),
        "events": len(bundle.events),
        "money_flow": len(bundle.money_flow),
    }
