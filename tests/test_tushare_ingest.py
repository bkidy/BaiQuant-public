from pathlib import Path
import threading
import time

import pandas as pd
import pytest

from baiquant.data.sqlite_provider import SqliteDataProvider
from baiquant.data.tushare_provider import (
    TushareIngestConfig,
    ingest_tushare,
    load_tushare_token,
)


class FakeTusharePro:
    def __init__(self) -> None:
        self.stock_basic_calls: list[str] = []
        self.daily_calls: list[str] = []
        self.adj_factor_calls: list[str] = []
        self.daily_basic_calls: list[str] = []
        self.stk_limit_calls: list[str] = []
        self.suspend_d_calls: list[str] = []
        self.moneyflow_calls: list[str] = []

    def stock_basic(self, exchange: str, list_status: str, fields: str) -> pd.DataFrame:
        self.stock_basic_calls.append(list_status)
        if list_status == "L":
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "name": "平安银行",
                        "industry": "银行",
                        "list_date": "19910403",
                    },
                    {
                        "ts_code": "600000.SH",
                        "name": "浦发银行",
                        "industry": "银行",
                        "list_date": "19991110",
                    },
                ]
            )
        if list_status == "D":
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000003.SZ",
                        "name": "退市测试",
                        "industry": "综合",
                        "list_date": "19910101",
                    }
                ]
            )
        return pd.DataFrame(columns=["ts_code", "name", "industry", "list_date"])

    def trade_cal(self, exchange: str, start_date: str, end_date: str, is_open: str) -> pd.DataFrame:
        frame = pd.DataFrame(
            [
                {"cal_date": "20260102", "is_open": 1},
                {"cal_date": "20260105", "is_open": 1},
            ]
        )
        return frame.loc[
            (frame["cal_date"] >= start_date) & (frame["cal_date"] <= end_date)
        ].reset_index(drop=True)

    def daily(self, trade_date: str) -> pd.DataFrame:
        self.daily_calls.append(trade_date)
        if trade_date == "20260102":
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": trade_date,
                        "open": 10.0,
                        "high": 10.5,
                        "low": 9.8,
                        "close": 10.2,
                        "pct_chg": 2.0,
                        "vol": 100,
                        "amount": 1000,
                    },
                    {
                        "ts_code": "600000.SH",
                        "trade_date": trade_date,
                        "open": 8.0,
                        "high": 8.1,
                        "low": 7.9,
                        "close": 8.0,
                        "pct_chg": 0.0,
                        "vol": 200,
                        "amount": 1600,
                    },
                ]
            )
        return pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": trade_date,
                    "open": 10.3,
                    "high": 11.22,
                    "low": 10.2,
                    "close": 11.22,
                    "pct_chg": 10.0,
                    "vol": 110,
                    "amount": 1234,
                }
            ]
        )

    def adj_factor(self, trade_date: str) -> pd.DataFrame:
        self.adj_factor_calls.append(trade_date)
        return pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "trade_date": trade_date, "adj_factor": 1.0},
                {"ts_code": "600000.SH", "trade_date": trade_date, "adj_factor": 1.0},
            ]
        )

    def daily_basic(self, trade_date: str, fields: str) -> pd.DataFrame:
        self.daily_basic_calls.append(trade_date)
        return pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "trade_date": trade_date, "pe_ttm": 7.5, "pb": 0.8},
                {"ts_code": "600000.SH", "trade_date": trade_date, "pe_ttm": 6.5, "pb": 0.6},
            ]
        )

    def stk_limit(self, trade_date: str) -> pd.DataFrame:
        self.stk_limit_calls.append(trade_date)
        return pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "trade_date": trade_date, "up_limit": 11.22, "down_limit": 9.18},
                {"ts_code": "600000.SH", "trade_date": trade_date, "up_limit": 8.8, "down_limit": 7.2},
            ]
        )

    def suspend_d(self, trade_date: str, suspend_type: str = "S") -> pd.DataFrame:
        self.suspend_d_calls.append(trade_date)
        if trade_date != "20260105":
            return pd.DataFrame(columns=["ts_code", "trade_date", "suspend_type"])
        return pd.DataFrame(
            [{"ts_code": "600000.SH", "trade_date": trade_date, "suspend_type": "S"}]
        )

    def moneyflow(self, trade_date: str) -> pd.DataFrame:
        self.moneyflow_calls.append(trade_date)
        return pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": trade_date,
                    "buy_sm_amount": 10,
                    "sell_sm_amount": 12,
                    "buy_md_amount": 20,
                    "sell_md_amount": 15,
                    "buy_lg_amount": 50,
                    "sell_lg_amount": 30,
                    "buy_elg_amount": 100,
                    "sell_elg_amount": 40,
                    "net_mf_amount": 73,
                }
            ]
        )


class FakeTushareProWithFailure(FakeTusharePro):
    def daily(self, trade_date: str) -> pd.DataFrame:
        if trade_date == "20260105":
            raise TimeoutError("daily timeout")
        return super().daily(trade_date)


class FakeTushareProWithFastFirstFailure(FakeTusharePro):
    def daily(self, trade_date: str) -> pd.DataFrame:
        if trade_date == "20260102":
            self.daily_calls.append(trade_date)
            raise TimeoutError("first date timeout")
        time.sleep(0.5)
        return super().daily(trade_date)


class FakeTushareProWithTransientDailyBasicFailure(FakeTusharePro):
    def __init__(self) -> None:
        super().__init__()
        self.daily_basic_failures = 0

    def daily_basic(self, trade_date: str, fields: str) -> pd.DataFrame:
        if trade_date == "20260102" and self.daily_basic_failures == 0:
            self.daily_basic_failures += 1
            raise TimeoutError("daily_basic transient timeout")
        return super().daily_basic(trade_date, fields)


class FakeTushareProWithEmptyLatestAdjFactor(FakeTusharePro):
    def adj_factor(self, trade_date: str) -> pd.DataFrame:
        self.adj_factor_calls.append(trade_date)
        if trade_date == "20260105" and self.adj_factor_calls == ["20260105"]:
            return pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"])
        return pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "trade_date": trade_date, "adj_factor": 1.0},
                {"ts_code": "600000.SH", "trade_date": trade_date, "adj_factor": 1.0},
            ]
        )


class FakeSlowTusharePro(FakeTusharePro):
    def __init__(self, delay_seconds: float) -> None:
        super().__init__()
        self.delay_seconds = delay_seconds
        self.active_daily_calls = 0
        self.max_active_daily_calls = 0
        self.lock = threading.Lock()

    def daily(self, trade_date: str) -> pd.DataFrame:
        with self.lock:
            self.active_daily_calls += 1
            self.max_active_daily_calls = max(self.max_active_daily_calls, self.active_daily_calls)
        try:
            time.sleep(self.delay_seconds)
            return super().daily(trade_date)
        finally:
            with self.lock:
                self.active_daily_calls -= 1


def test_load_tushare_token_reads_secret_file(tmp_path: Path) -> None:
    token_path = tmp_path / "tushare_token"
    token_path.write_text("abc123\n")

    assert load_tushare_token(token_path=token_path) == "abc123"


def test_load_tushare_token_raises_without_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.delenv("TS_TOKEN", raising=False)

    with pytest.raises(RuntimeError):
        load_tushare_token(token_path=tmp_path / "missing")


def test_ingest_tushare_writes_normalized_sqlite_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "market.db"
    fake = FakeTusharePro()

    summary = ingest_tushare(
        TushareIngestConfig(
            output_path=db_path,
            start_date="20260102",
            end_date="20260105",
            adjust="none",
            token="fake-token",
        ),
        pro=fake,
    )

    bundle = SqliteDataProvider(db_path).load()
    prices = bundle.prices.set_index(["date", "code"])
    money_flow = bundle.money_flow.set_index(["date", "code"])

    assert fake.stock_basic_calls == ["L", "D", "P"]
    assert summary["stocks"] == 3
    assert summary["prices"] == 4
    assert summary["fundamentals"] == 4
    assert summary["money_flow"] == 2
    assert bundle.stocks["code"].tolist() == ["000001.SZ", "000003.SZ", "600000.SH"]
    assert prices.loc[(pd.Timestamp("2026-01-02"), "000001.SZ"), "amount"] == 1_000_000
    assert prices.loc[(pd.Timestamp("2026-01-05"), "000001.SZ"), "limit_up"] == 1
    assert prices.loc[(pd.Timestamp("2026-01-05"), "600000.SH"), "paused"] == 1
    assert prices.loc[(pd.Timestamp("2026-01-05"), "600000.SH"), "close"] == 8.0
    assert bundle.fundamentals.loc[bundle.fundamentals["code"] == "000001.SZ", "pe_ttm"].iloc[0] == 7.5
    assert money_flow.loc[(pd.Timestamp("2026-01-02"), "000001.SZ"), "large_net_inflow"] == 200_000
    assert money_flow.loc[(pd.Timestamp("2026-01-02"), "000001.SZ"), "super_large_net_inflow"] == 600_000
    assert money_flow.loc[(pd.Timestamp("2026-01-02"), "000001.SZ"), "main_net_inflow"] == 800_000


def test_ingest_tushare_flushes_successful_batches_before_fail_fast_error(tmp_path: Path) -> None:
    db_path = tmp_path / "market.db"

    with pytest.raises(TimeoutError):
        ingest_tushare(
            TushareIngestConfig(
                output_path=db_path,
                start_date="20260102",
                end_date="20260105",
                adjust="none",
                token="fake-token",
                flush_every=1,
                continue_on_error=False,
            ),
            pro=FakeTushareProWithFailure(),
        )

    bundle = SqliteDataProvider(db_path).load()

    assert bundle.prices["date"].dt.strftime("%Y%m%d").unique().tolist() == ["20260102"]
    assert len(bundle.prices) == 2


def test_ingest_tushare_parallel_fail_fast_does_not_drain_future_queue(tmp_path: Path) -> None:
    db_path = tmp_path / "market.db"

    started = time.perf_counter()
    with pytest.raises(TimeoutError):
        ingest_tushare(
            TushareIngestConfig(
                output_path=db_path,
                start_date="20260102",
                end_date="20260105",
                adjust="none",
                token="fake-token",
                flush_every=1,
                workers=2,
                continue_on_error=False,
            ),
            pro=FakeTushareProWithFastFirstFailure(),
        )
    elapsed = time.perf_counter() - started

    assert elapsed < 0.3


def test_ingest_tushare_retries_transient_endpoint_failures(tmp_path: Path) -> None:
    db_path = tmp_path / "market.db"
    fake = FakeTushareProWithTransientDailyBasicFailure()

    summary = ingest_tushare(
        TushareIngestConfig(
            output_path=db_path,
            start_date="20260102",
            end_date="20260102",
            adjust="none",
            token="fake-token",
            retries=1,
            retry_sleep_seconds=0,
        ),
        pro=fake,
    )

    assert fake.daily_basic_failures == 1
    assert fake.daily_basic_calls == ["20260102"]
    assert summary["failures"] == 0
    assert summary["fundamentals"] == 2


def test_ingest_tushare_handles_empty_latest_adj_factor(tmp_path: Path) -> None:
    db_path = tmp_path / "market.db"
    fake = FakeTushareProWithEmptyLatestAdjFactor()

    summary = ingest_tushare(
        TushareIngestConfig(
            output_path=db_path,
            start_date="20260102",
            end_date="20260105",
            adjust="qfq",
            token="fake-token",
            flush_every=1,
        ),
        pro=fake,
    )

    bundle = SqliteDataProvider(db_path).load()

    assert summary["prices"] == 4
    assert bundle.prices["date"].dt.strftime("%Y%m%d").unique().tolist() == ["20260102", "20260105"]


def test_ingest_tushare_resume_skips_dates_already_in_prices(tmp_path: Path) -> None:
    db_path = tmp_path / "market.db"
    ingest_tushare(
        TushareIngestConfig(
            output_path=db_path,
            start_date="20260102",
            end_date="20260102",
            adjust="none",
            token="fake-token",
            flush_every=1,
        ),
        pro=FakeTusharePro(),
    )
    fake = FakeTusharePro()

    summary = ingest_tushare(
        TushareIngestConfig(
            output_path=db_path,
            start_date="20260102",
            end_date="20260105",
            adjust="none",
            token="fake-token",
            write_mode="append",
            flush_every=1,
            resume=True,
        ),
        pro=fake,
    )

    bundle = SqliteDataProvider(db_path).load()

    assert fake.daily_calls == ["20260105"]
    assert summary["skipped_dates"] == 1
    assert bundle.prices["date"].dt.strftime("%Y%m%d").unique().tolist() == ["20260102", "20260105"]


def test_ingest_tushare_can_skip_moneyflow_for_core_downloads(tmp_path: Path) -> None:
    db_path = tmp_path / "market.db"
    fake = FakeTusharePro()

    summary = ingest_tushare(
        TushareIngestConfig(
            output_path=db_path,
            start_date="20260102",
            end_date="20260105",
            adjust="none",
            token="fake-token",
            include_money_flow=False,
        ),
        pro=fake,
    )

    bundle = SqliteDataProvider(db_path).load()

    assert fake.daily_calls == ["20260102", "20260105"]
    assert fake.moneyflow_calls == []
    assert summary["prices"] == 4
    assert summary["fundamentals"] == 4
    assert summary["money_flow"] == 0
    assert bundle.money_flow.empty


def test_ingest_tushare_can_fetch_prices_only(tmp_path: Path) -> None:
    db_path = tmp_path / "market.db"
    fake = FakeTusharePro()

    summary = ingest_tushare(
        TushareIngestConfig(
            output_path=db_path,
            start_date="20260102",
            end_date="20260105",
            adjust="none",
            token="fake-token",
            include_daily_basic=False,
            include_money_flow=False,
            include_limits=False,
            include_suspends=False,
        ),
        pro=fake,
    )

    bundle = SqliteDataProvider(db_path).load()

    assert fake.daily_calls == ["20260102", "20260105"]
    assert fake.daily_basic_calls == []
    assert fake.stk_limit_calls == []
    assert fake.suspend_d_calls == []
    assert fake.moneyflow_calls == []
    assert summary["prices"] == 3
    assert summary["fundamentals"] == 0
    assert summary["money_flow"] == 0
    assert bundle.fundamentals.empty
    assert bundle.money_flow.empty


def test_ingest_tushare_fetches_trade_dates_with_parallel_workers(tmp_path: Path) -> None:
    db_path = tmp_path / "market.db"
    fake = FakeSlowTusharePro(delay_seconds=0.05)

    summary = ingest_tushare(
        TushareIngestConfig(
            output_path=db_path,
            start_date="20260102",
            end_date="20260105",
            adjust="none",
            token="fake-token",
            include_daily_basic=False,
            include_money_flow=False,
            include_limits=False,
            include_suspends=False,
            flush_every=2,
            workers=2,
        ),
        pro=fake,
    )

    bundle = SqliteDataProvider(db_path).load()

    assert sorted(fake.daily_calls) == ["20260102", "20260105"]
    assert fake.max_active_daily_calls == 2
    assert summary["fetched_dates"] == 2
    assert summary["prices"] == 3
    assert bundle.prices["date"].dt.strftime("%Y%m%d").unique().tolist() == ["20260102", "20260105"]


def test_ingest_tushare_can_backfill_only_moneyflow_from_existing_prices(tmp_path: Path) -> None:
    db_path = tmp_path / "market.db"
    ingest_tushare(
        TushareIngestConfig(
            output_path=db_path,
            start_date="20260102",
            end_date="20260102",
            adjust="none",
            token="fake-token",
            include_money_flow=False,
            flush_every=1,
        ),
        pro=FakeTusharePro(),
    )
    fake = FakeTusharePro()

    summary = ingest_tushare(
        TushareIngestConfig(
            output_path=db_path,
            start_date="20260102",
            end_date="20260102",
            adjust="none",
            token="fake-token",
            write_mode="append",
            include_prices=False,
            include_daily_basic=False,
            include_limits=False,
            include_suspends=False,
            include_money_flow=True,
            flush_every=1,
        ),
        pro=fake,
    )

    bundle = SqliteDataProvider(db_path).load()
    money_flow = bundle.money_flow.set_index(["date", "code"])

    assert fake.daily_calls == []
    assert fake.daily_basic_calls == []
    assert fake.stk_limit_calls == []
    assert fake.suspend_d_calls == []
    assert fake.moneyflow_calls == ["20260102"]
    assert summary["prices"] == 0
    assert summary["fundamentals"] == 0
    assert summary["money_flow"] == 1
    assert money_flow.loc[(pd.Timestamp("2026-01-02"), "000001.SZ"), "close"] == 10.2
    assert money_flow.loc[(pd.Timestamp("2026-01-02"), "000001.SZ"), "main_net_inflow_pct"] == 80.0


def test_cli_exposes_tushare_ingest() -> None:
    from baiquant.cli import build_parser

    parser = build_parser()

    args = parser.parse_args(
        [
            "ingest",
            "tushare",
            "--output",
            "data/tushare/baiquant.db",
            "--start",
            "20260102",
            "--end",
            "20260105",
            "--adjust",
            "none",
            "--flush-every",
            "1",
            "--progress-every",
            "1",
            "--workers",
            "3",
            "--rate-limit-per-minute",
            "180",
            "--timeout",
            "15",
            "--retries",
            "2",
            "--retry-sleep",
            "0.5",
            "--resume",
            "--only-core",
        ]
    )

    assert args.command == "ingest"
    assert args.source == "tushare"
    assert args.adjust == "none"
    assert args.flush_every == 1
    assert args.progress_every == 1
    assert args.workers == 3
    assert args.rate_limit_per_minute == 180
    assert args.timeout == 15
    assert args.retries == 2
    assert args.retry_sleep == 0.5
    assert args.resume is True
    assert args.only_core is True


def test_cli_exposes_tushare_moneyflow_only_mode() -> None:
    from baiquant.cli import build_parser

    parser = build_parser()

    args = parser.parse_args(
        [
            "ingest",
            "tushare",
            "--output",
            "data/tushare/baiquant.db",
            "--start",
            "20260102",
            "--end",
            "20260105",
            "--only-money-flow",
        ]
    )

    assert args.only_money_flow is True


def test_cli_exposes_tushare_prices_only_mode() -> None:
    from baiquant.cli import build_parser

    parser = build_parser()

    args = parser.parse_args(
        [
            "ingest",
            "tushare",
            "--output",
            "data/tushare/baiquant.db",
            "--start",
            "20260102",
            "--end",
            "20260105",
            "--only-prices",
        ]
    )

    assert args.only_prices is True
