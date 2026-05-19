# MacroIntel Institutional

Current public snapshot: 2026-05-19

MacroIntel Institutional is a paper-first trading research and monitoring system for US equities, with a separate NSE paper lane. The active system combines daily feature refreshes, a fixed-return ML selector, an LLM risk filter, Alpaca-backed live marks, stop/target monitoring, and an operator dashboard.

This repository intentionally excludes private runtime state: API keys, broker account data, VM addresses, live logs, generated reports, model binaries, and large market-data parquet files.

## What Runs Now

- US fixed-return paper trading lane
  - Scores a broad US equity feature universe.
  - Blocks ETFs and manually banned symbols before selection.
  - Applies price, liquidity, data quality, sector, earnings, and regime gates.
  - Scores ML probability against the runtime threshold.
  - Sends final candidates through an LLM portfolio/risk filter.
  - Writes daily signals for paper execution and dashboard display.

- Alpaca integration
  - Uses Alpaca paper credentials for live quote marks and paper order bridging.
  - Updates open-position mark-to-market and dashboard prices from Alpaca data.
  - Paper execution writes local state and can submit paper orders through the bridge when configured.

- Intraday monitoring
  - Refreshes open-position P&L from Alpaca.
  - Monitors profit targets and stop losses through the paper execution path.
  - Maintains shadow intraday scores for current signals without creating new orders.

- Stock Diagnostic
  - Lets an operator type a symbol into the dashboard and run the same ML/gate/LLM logic on that one name.
  - Uses local feature data when available.
  - Refreshes from Alpaca when a symbol is missing or stale.
  - Has a Force LLM research mode so a rejected stock can still be evaluated by the LLM without pretending it is tradable.
  - Ignores daily top-N capacity as a blocking rule. The diagnostic answers whether the symbol itself passes the system, not whether the daily portfolio had a spare slot.

- NSE paper lane
  - Separate NSE dashboard, signals, positions, regime/gate state, and paper P&L tracker.
  - Not part of the US Alpaca lane.
  - Do not delete or merge it into the US fixed-return scripts.

## Main Entry Points

```text
project/dashboard_ultra.py
    Flask dashboard on port 5055 by default.

project/scripts/fixed_return_daily_signals.py
    Daily US signal generator: ML threshold, hard gates, LLM filter, signal files.

project/scripts/fixed_return_paper_execute.py
    Paper execution and stop/target handling for US fixed-return signals.

project/scripts/intraday_mark_to_market.py
    Alpaca mark-to-market refresh for open US positions and shadow signal marks.

project/scripts/single_symbol_diagnostic.py
    One-symbol diagnostic used by the dashboard search panel and CLI.

project/scripts/daily_data_refresh.py
project/scripts/intraday_universe_refresh.py
    Feature refresh and symbol-level Alpaca data updates.
```

## Quick Start

From a Linux VM or local shell with Python 3.10+:

```bash
cd macro_intelligence_complete
python3 -m venv venv
source venv/bin/activate
pip install -r project/requirements.txt
cp project/.env.example project/.env
```

Edit `project/.env` with private values. Do not commit it.

Start the dashboard in the background:

```bash
./start.sh
```

Stop it:

```bash
./stop.sh
```

Manual dashboard start:

```bash
cd project
source ../venv/bin/activate
nohup python dashboard_ultra.py > /tmp/dash.log 2>&1 &
```

Open:

```text
http://<host>:5055
```

## Common Commands

Run the US daily signal generator:

```bash
cd project
source ../venv/bin/activate
python scripts/fixed_return_daily_signals.py
```

Execute generated US paper signals and evaluate exits:

```bash
cd project
source ../venv/bin/activate
python scripts/fixed_return_paper_execute.py
```

Refresh Alpaca intraday marks:

```bash
cd project
source ../venv/bin/activate
python scripts/intraday_mark_to_market.py --force
```

Run a single-symbol diagnostic:

```bash
cd project
source ../venv/bin/activate
python scripts/single_symbol_diagnostic.py ASBP --force-llm
```

## Dashboard APIs

The active dashboard exposes lightweight operator endpoints:

```text
GET  /api/snapshot
GET  /api/pnl
GET  /api/nse
GET  /api/operator
GET  /api/live_prices
POST /api/run_signals
POST /api/symbol_diagnostic
GET  /api/symbol_diagnostic_examples
```

## Environment

Runtime configuration lives in `project/.env`. Public examples belong in `project/.env.example`.

Important private values:

```text
ALPACA_API_KEY=<paper key>
ALPACA_SECRET_KEY=<paper secret>
ALPACA_PAPER=true
GROQ_API_KEY=<comma-separated OpenRouter/Groq-compatible keys>
```

Important non-secret controls:

```text
DASHBOARD_ULTRA_PORT=5055
SIG_THRESHOLD=0.61
SIG_TOP_N=30
SIG_MIN_PRICE=5
SIG_MIN_ADV_DOLLAR=5000000
SIG_PROFIT_TARGET_PCT=10
SIG_STOP_LOSS_PCT=3
SIG_HOLD_DAYS=8
LLM_RESERVE_LAST_KEY_FOR_SEARCH=1
LLM_SEARCH_USE_ALL_KEYS=1
```

The code supports key rotation for the LLM filter. Exhausted or rate-limited keys are retired in the current run so the next request starts from the next usable key instead of retrying a dead key forever.

## Generated Files

These are runtime outputs and are intentionally ignored by git:

```text
project/reports/
project/logs/
project/data/features/
project/data/prices_full/
*.parquet
*.joblib
*.pkl
*.log
project/.env
```

If you need to reproduce a live result, regenerate those artifacts from the private runtime environment. Do not commit broker state, API keys, VM addresses, or raw account logs.

## Current Safety Posture

- Paper trading is the intended default.
- Alpaca should be configured with paper credentials unless live trading has been explicitly approved and separately audited.
- ETFs and leveraged ETFs are blocked through repo data files before signal selection.
- Stop-loss and profit-target logic belongs to the paper execution and MTM path.
- The dashboard is an operator surface, not a replacement for broker-side controls.

See `STRATEGY.md`, `DEPLOY.md`, and `INTEGRATION_GUIDE.md` for the current operating model.
