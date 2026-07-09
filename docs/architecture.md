# BaiQuant Architecture

## Goal

Build a transparent daily A-share stock selection system that can start with
free/local CSV data and later swap in professional data providers.

## Flow

```text
CsvDataProvider
or SqliteDataProvider
-> MarketDataBundle
-> build_universe(as_of)
-> compute_factor_frame(as_of)
-> score_factor_frame()
-> build_equal_weight_portfolio()
-> optional run_rebalance_backtest()
```

CSV data can be converted to SQLite without touching the research pipeline:

```text
CsvDataProvider -> MarketDataBundle -> write_market_data_to_sqlite()
```

## Boundaries

- `data/` owns loading and schema validation only.
- `data/tushare_provider.py` owns external Tushare calls and normalizes them to
  the storage contract.
- `data/baostock_provider.py` enriches the SQLite contract with profile and
  financial fields from BaoStock.
- `data/efinance_provider.py` enriches the SQLite contract with recent
  historical money-flow fields from efinance.
- `data/sqlite_provider.py` owns SQLite persistence for the normalized tables.
- `data/converters.py` owns local storage migration helpers.
- `universe/` decides whether a stock is eligible on a signal date.
- `factors/` computes raw factor values and does not know about weights.
- `scoring.py` normalizes raw factor values and applies weights/directions.
- `portfolio.py` turns ranked scores into target weights.
- `backtest.py` evaluates dated target weights and A-share execution plans.
- `strategy/live20k.py` owns the current RMB 20,000 strategy preset, signal
  generation, market gate, one-command paper step, daily paper/live-before-live
  plan, and paper-fill ledger helper/report/order export gate.

## Next Extensions

- Add JQData/RiceQuant adapters that output the same `MarketDataBundle` if a
  second professional data source is needed.
- Add concept/sector heat and news-event adapters.
- Store larger normalized tables as Parquet or DuckDB if SQLite becomes too
  slow for broad historical scans.
- Add point-in-time financial disclosure dates before using financial factors
  in serious historical backtests.
- Add broker-specific adapters only after the gated order CSV has been reviewed
  in a paper-to-live rehearsal.
- Add IC, rank-bucket, turnover, and industry-neutral analysis reports.
