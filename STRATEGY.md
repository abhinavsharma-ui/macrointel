# Current Strategy

Current public snapshot: 2026-05-19

MacroIntel is currently organized around a US fixed-return paper strategy plus a separate NSE paper strategy. The goal is not to predict every stock. The goal is to find a small number of liquid, tradable setups that pass model probability, hard market structure gates, and an LLM risk review.

## US Fixed-Return Lane

The US lane is the active Alpaca-integrated path.

1. Feature refresh
   - Reads and updates per-symbol feature parquet files.
   - Uses Alpaca for missing or refreshed symbol data where the diagnostic or intraday tools need it.
   - Keeps large feature files outside git.

2. Hard blocks
   - Drops ETFs and leveraged ETFs.
   - Drops manual blocklist symbols.
   - Enforces minimum price and liquidity.
   - Requires enough feature history for production treatment.

3. ML scoring
   - Scores each symbol with the fixed-return model.
   - Uses `SIG_THRESHOLD` as the probability gate.
   - Keeps `SIG_TOP_N` as daily portfolio capacity for normal nightly production.
   - The single-symbol diagnostic ignores daily top-N as a blocker because it is answering a different question: whether this symbol itself passes the system.

4. Pre-LLM gates
   - Sector and regime checks.
   - Earnings feature checks.
   - Earnings calendar proximity checks.
   - Volatility and momentum context.

5. LLM risk filter
   - Reviews only model-qualified candidates in normal production.
   - Uses tool evidence for short interest, options IV, price momentum, sector performance, news risk, setup score, and catalyst events.
   - Can return `proceed`, `reduce_half`, or `skip`.
   - Uses key rotation so rate-limited keys are retired during the run.

6. Signal output
   - Writes the final signal JSON and CSV under `project/reports/`.
   - Writes scored universe context separately so diagnostics can explain why names did or did not pass.

7. Paper execution and exits
   - Reads the daily signal file.
   - Adds open positions up to configured paper capacity.
   - Tracks profit target, stop loss, and hold period.
   - Uses Alpaca paper order bridging when configured.

8. Intraday mark-to-market
   - Reads open positions.
   - Pulls latest Alpaca bars for open names and current signals.
   - Updates current price, P&L, and dashboard state.
   - Writes shadow signal performance without placing extra intraday orders.

## Default Trade Shape

The public defaults are controlled through environment variables:

```text
SIG_PROFIT_TARGET_PCT=10
SIG_STOP_LOSS_PCT=3
SIG_HOLD_DAYS=8
SIG_MIN_PRICE=5
SIG_MIN_ADV_DOLLAR=5000000
```

Position sizing is based on each signal's `position_pct`. Dashboard P&L uses:

```text
return = live_price / entry_price - 1
pnl = initial_capital * position_pct * return
```

If a real quantity exists, execution state may also use quantity-based P&L.

## Stock Diagnostic

The dashboard search panel is built for operator research and client-facing diagnostics.

It answers:

- Is this symbol in the system already?
- If not, can it be fetched from Alpaca?
- Does it have enough usable feature history?
- What probability does the production model assign?
- Which gates pass or fail?
- Would the LLM proceed, reduce size, or skip?
- Is a failure a hard block or just a research-only rejection?

Force LLM mode intentionally lets the LLM review a name even when ML or soft gates fail. It does not convert a failed stock into a tradable production signal. It is for explanation and research.

## NSE Lane

The NSE pipeline is separate from the US fixed-return lane.

- Separate dashboard panels.
- Separate paper positions and P&L.
- Separate signal source.
- Separate regime/gate state.
- No Alpaca execution path.

Do not remove NSE files as old crypto/US experiments. The active NSE paper monitor is part of the current system.

## Risk Rules

Current public hard rules:

- Paper-first operation.
- No secrets in git.
- ETF and leveraged ETF blocks before signal selection.
- Minimum liquidity before trading.
- Stop loss and target monitored from paper execution state.
- LLM cannot approve a hard-blocked ETF/blocklist symbol.
- Diagnostic Force LLM is research-only when normal gates fail.

## What Is Legacy

The repo still contains older experiments and research scripts. Treat them as historical unless they are called by the active entry points:

```text
project/dashboard_ultra.py
project/scripts/fixed_return_daily_signals.py
project/scripts/fixed_return_paper_execute.py
project/scripts/intraday_mark_to_market.py
project/scripts/single_symbol_diagnostic.py
project/scripts/daily_data_refresh.py
project/scripts/intraday_universe_refresh.py
```

Before deleting anything under `project/scripts/`, trace whether the dashboard, cron, or NSE lane imports it.
