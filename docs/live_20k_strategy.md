# BaiQuant Manual Trading Strategy

This is research infrastructure, not investment advice.

## Public Presets

The hotspot/Turbo selector exposes two public preset aliases:

- `steady-20d`: the default manual strategy. It uses the Turbo stock-picking
  core, keeps at most three positions, uses the 6% single-name stop, 10%
  account drawdown stop, and rotates stale positions after 20 trading days.
  This is the main after-close manual trading preset.
- `turbo-sprint`: the more aggressive short-term version. It keeps the same
  Turbo signal core, rotates faster, and can add after early profit. Use it only
  when you explicitly want higher turnover and higher drawdown tolerance.

The explainable multi-factor selector is available separately through the
`manual-multifactor` alias in the Web UI and the `daily-plan` command. It
focuses on a smaller manual operation checklist: current holdings first, then
new buys only when the market-regime gate allows them.

Older local Chinese preset names are still accepted for backward compatibility,
but public docs should use the English aliases above.

## Daily Command

```bash
.venv/bin/baiquant live20k-2026 \
  --db data/tushare/baiquant.db \
  --as-of 2026-06-30 \
  --cash 50000 \
  --preset steady-20d \
  --watchlist \
  --watchlist-limit 10
```

## One-Command Multi-Factor Checklist

```bash
.venv/bin/baiquant daily-plan \
  --cash 50000
```

Defaults:

- Reads `data/tushare/baiquant.db`.
- Reads live holdings from `data/live/holdings.csv`.
- Uses the latest available trading date unless `--as-of` is provided.
- Writes `data/live/daily_plan.md`, `data/live/daily_plan.csv`, and
  `data/live/daily_watchlist.csv`.

Read the generated Markdown report in this order:

1. Market state: only add new stocks when the new-entry gate is open.
2. Account snapshot: cash, equity, exposure, and total unrealized PnL.
3. Holding risk: every current holding, marked price, return, drawdown, and stop
   signal.
4. Next-day actions: executable old-position and new-position instructions.
5. Watchlist: ranked candidates only; it is not a blind buy list.
6. Execution discipline: guardrails for manual execution.

## Record Real Fills

After you actually buy or sell in the broker app, record the fill:

```bash
.venv/bin/baiquant record-trade \
  --date 2026-07-02 \
  --action buy \
  --code 600000.SH \
  --name ExampleStock \
  --shares 100 \
  --price 10.00
```

Defaults:

- Updates `data/live/holdings.csv`.
- Appends to `data/live/trade_log.csv`.
- Normalizes six-digit A-share codes such as `601636` to `601636.SH`.
- For buys, recalculates weighted average cost.
- For sells, reduces/removes the position and records realized PnL.

The Web UI has a fill-entry form in the holdings section. It uses the same
ledger path and rules as `record-trade`. The recent-fills table reads from
`data/live/trade_log.csv`, newest first, so the daily review can compare planned
actions with what was actually executed.

## Web UI

```bash
.venv/bin/baiquant desk
```

The Web UI accepts these public preset aliases through query/API parameters:

- `steady-20d`
- `turbo-sprint`
- `manual-multifactor`

## Main Rules

- Signals are generated after close from local Tushare SQLite data.
- Orders are planned for the next trading day's open.
- A-share constraints are respected in the research pipeline: lot size, paused
  stocks, limit-up buys, limit-down sells, and T+1-style next-open execution
  assumptions.
- The default strategy is designed for manual review. The watchlist is not a
  blind buy list; it is a ranked pool for next-day confirmation.

## Technical Overlay

The main strategy adds an advisory technical overlay to both the daily plan and
watchlist.

Output fields:

- `tech_score`: 0-100 technical setup score.
- `tech_grade`: A/B/C/D setup grade.
- `trade_advice`: buy/half-size/watch/skip style guidance in the local UI.
- `position_scale`: suggested size multiplier.
- `risk_flags`: compact local risk labels for overextension, long upper shadow,
  weakening money flow, trend breaks, and industry cooling.

Default behavior:

- `position_scale` is advisory only for `steady-20d`.
- The backtest engine supports opt-in scaled sizing through
  `use_position_scale=True`.
- The default strategy keeps baseline sizing because the opt-in scaled overlay
  weakened the 2026 sample without improving drawdown.

## Latest Local Evidence

The strongest verified manual baseline from the current local research set is
the fixed 20-trading-day version:

- 2026 YTD backtest capital: RMB 50,000.
- Total return: +164.48%.
- Max drawdown: -19.92%.
- Worst week: -16.31%.
- Exposure: 71.55%.

Overlay validation through 2026-06-30:

- Default advisory overlay: +164.48%, max drawdown -19.92%, worst week -16.31%.
- Opt-in scaled overlay: +131.06%, max drawdown -19.92%, worst week -16.31%.
- Conclusion: keep the overlay visible for manual risk review, but do not let it
  reduce default position sizing.

That result proves the local 2026 sample was favorable to this rule set. It is
not an expected return, and it does not prove future live performance. Continue
to treat drawdown control and manual execution discipline as part of the
strategy.
