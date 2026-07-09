from pathlib import Path

import pytest

from baiquant.cli import build_parser
from baiquant.config import load_pipeline_config


def test_cli_no_longer_exposes_akshare_ingest() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "ingest",
                "akshare",
                "--output",
                "unused.db",
                "--start",
                "20260101",
                "--end",
                "20260525",
            ]
        )


def test_project_configs_use_tushare_sqlite_database() -> None:
    root = Path(__file__).resolve().parents[1]

    live_data, _ = load_pipeline_config(root / "configs/live_50k_conservative.toml")
    default_data, _ = load_pipeline_config(root / "configs/tushare_sqlite.toml")

    assert live_data.kind == "sqlite"
    assert default_data.kind == "sqlite"
    assert live_data.path == root / "data/tushare/baiquant.db"
    assert default_data.path == root / "data/tushare/baiquant.db"
