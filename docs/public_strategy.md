# Public Strategy Scope

BaiQuant's public release focuses on one documented strategy:

- `steady-20d`

This is an after-close manual A-share stock-selection workflow. It combines a
hotspot/Turbo selector with execution guardrails such as market breadth checks,
lot-size sizing, single-name stop loss, account drawdown protection, and
next-trading-day planning.

Other modules in the repository, including multi-factor diagnostics, data
ingestion adapters, paper ledgers, and technical overlays, are supporting
research infrastructure. They are useful for validation and experimentation,
but they are not presented as independent live-trading strategies in the public
documentation.

No generated live ledger, broker screenshot, private watchlist, Tushare
database, or provider-downloaded market data is part of the public release.
