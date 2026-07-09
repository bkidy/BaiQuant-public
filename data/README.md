# Local Data Directory

This directory is intentionally kept out of git except for this README.

Do not commit real market databases, downloaded provider data, broker
screenshots, live holdings, trade logs, portfolio snapshots, generated plans,
or backtest artifacts that contain licensed data or personal trading records.

Typical local-only paths:

- `data/tushare/baiquant.db`
- `data/live/holdings.csv`
- `data/live/trade_log.csv`
- `data/live/portfolio_snapshots.csv`
- `data/live/daily_plan.md`
- `data/research/*.csv`
- `data/backtests/*.csv`

Use the tiny fixtures under `examples/data/` for tests and open-source demos.
