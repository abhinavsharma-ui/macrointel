# Handoff: Stop-Loss, Cron, Dashboard Amount Fix

VM: `abhinavsharma1359@34.61.149.100`
Project: `/home/abhinavsharma1359/macro_intelligence_complete/project`
Venv: `source ../venv/bin/activate` from the project directory
Dashboard: `python dashboard_ultra.py` on port `5055`

## What Was Broken

- `scripts/fixed_return_paper_execute.py` had the stop-loss branches commented out with `[SL REMOVED]`, so AGPU and CNSP could sit below stop without closing.
- The executor checked signal freshness before exits, so a stale/missing signal file could block risk exits.
- The paper execute cron was scheduled at `20:45 UTC`, but the 2026-05-18 signal job finished at `21:21 UTC`; execution fired too early and exited as stale.
- `scripts/sl_monitor.py` was a separate Yahoo-only implementation with no current log evidence, creating a second divergent stop-loss path.

## What Changed

- Restored stop-loss/profit-target/time-exit handling in `scripts/fixed_return_paper_execute.py`.
- Exits now stay active even when signals are stale or missing; stale signals only block new entries.
- Added duplicate trade-row protection.
- Added safe Alpaca buy/sell wrappers so an Alpaca bridge issue cannot abort the local paper ledger update.
- Prevented same-run re-entry into symbols closed that run.
- Switched executor date checks to `America/New_York`.
- Updated `/home/abhinavsharma1359/macro_intelligence_complete/scripts/run_us_intraday_mtm.sh` to:
  - refresh open-position live universe data first,
  - run intraday MTM,
  - then run the canonical executor.
- Updated `dashboard_ultra.py` live polling so if the open-position symbol set changes, the US dashboard section reloads immediately instead of leaving closed symbols in an already-open browser tab.
- Replaced `scripts/sl_monitor.py` with a compatibility wrapper that delegates to `fixed_return_paper_execute.py`.
- Updated `scripts/alpaca_bridge.py` to load the project `.env` before checking Alpaca keys.
- Moved paper-execute cron to `21:35 UTC`, after the daily signal cron.

## Backups

- `scripts/fixed_return_paper_execute.py.bak_restore_sl_20260518_2155`
- `reports/fixed_return_open_positions.json.bak_restore_sl_20260518_2155`
- `reports/fixed_return_paper_trades.csv.bak_restore_sl_20260518_2155`
- `scripts/sl_monitor.py.bak_delegate_executor_20260518`
- `scripts/alpaca_bridge.py.bak_dotenv_20260518`
- `dashboard_ultra.py.bak_live_symbol_reload_20260518`

## Current Verified State

Run:

```bash
cd /home/abhinavsharma1359/macro_intelligence_complete/project
source ../venv/bin/activate
python3 scripts/fixed_return_paper_execute.py --dry-run
```

Verified output after fixes:

```text
OPEN POSITIONS BEFORE 3
CLOSED TODAY 0
NEW POSITIONS OPENED 0
OPEN POSITIONS AFTER 3
RUNNING CLOSED PNL CONTRIBUTION PCT 0.0000
DRY RUN: no files written
```

AGPU and CNSP were closed in the paper ledger:

```text
CNSP stop_loss entry=5.01 exit=4.8597 pnl=-16.87 closed_at=2026-05-18T21:56:49.951190+00:00
AGPU stop_loss entry=5.60 exit=5.4320 pnl=-8.44 closed_at=2026-05-18T21:56:49.978518+00:00
```

Current open symbols:

```text
ACVA, IPGP, RPD
```

Dashboard API verification:

```text
portfolio_value=100794.07
cash=100636.73
closed_pnl=636.73
closed_trades=11
open_positions_count=3
unrealized_net=157.34
```

Open positions from `/api/snapshot`:

```text
IPGP current=106.45 unrealized_pnl=84.20 return=3.742%
RPD  current=6.73   unrealized_pnl=45.14 return=8.026%
ACVA current=6.01   unrealized_pnl=28.00 return=4.978%
```

`/api/live_prices` returns 3 positions: `ACVA, IPGP, RPD`.

The served dashboard HTML now includes `function liveSymbolsChanged`, and the dashboard process was restarted after the patch.

## ACVA/RPD Signal Provenance

ACVA and RPD did come from the latest daily signal run:

```text
reports/fixed_return_daily_signals.json mtime UTC: 2026-05-18T21:21:21.784138+00:00
signal_date: 2026-05-18
generated_at: 2026-05-18T21:21:21.779450+00:00
ACVA probability=0.651844 entry=6.05 PT=6.655 SL=5.8685
RPD  probability=0.581711 entry=6.695 PT=7.3645 SL=6.4942
```

The issue was not that signal cron failed; it ran late relative to execution. The execute cron has now been moved from `20:45 UTC` to `21:35 UTC`.

## Current Cron

Relevant final line:

```cron
35 21 * * 1-5 /bin/bash -lc 'cd /home/abhinavsharma1359/macro_intelligence_complete/project && source ../venv/bin/activate && python scripts/fixed_return_paper_execute.py >> logs/paper_execute.log 2>&1'
```

The intraday runner still runs every 30 minutes during the US market window and now invokes the canonical executor too.

## Alpaca Bridge Note

When the manual stop-loss close was run, Alpaca bridge printed:

```text
ALPACA BRIDGE: keys not set, skipping sell CNSP
ALPACA BRIDGE: keys not set, skipping sell AGPU
```

So the local paper ledger and dashboard are fixed. If real Alpaca paper orders must be submitted too, make sure the environment visible to `alpaca_bridge.py` has `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, and `ALPACA_PAPER=true`.

After that manual run, `scripts/alpaca_bridge.py` was patched to load `.env`; verification now reports:

```text
ALPACA_ENABLED_AFTER_DOTENV True
```

This does not retroactively submit the already-closed CNSP/AGPU sells. It makes future executor runs able to see the configured Alpaca paper credentials.
