# Handoff: dashboard_ultra live polling and PnL fix

Date: 2026-05-18
VM: `abhinavsharma1359@34.61.149.100`
Project: `/home/abhinavsharma1359/macro_intelligence_complete`
Dashboard file: `/home/abhinavsharma1359/macro_intelligence_complete/project/dashboard_ultra.py`
Dashboard URL: `http://34.61.149.100:5055`

## What was broken

- `/api/live_prices` was returning PnL about 100x too large.
- Root cause: `open_positions()` returns `position_pct` as display percent, e.g. `0.5625` means `0.5625%`, but `/api/live_prices` treated it as a capital fraction.
- The injected browser JS existed twice at the bottom of the HTML string, creating duplicate polling timers and making debugging noisy.
- The Open Positions header did not actually include the visible `LIVE` badge / `RUN SIGNALS` button in the live VM file.

## What was changed

In `project/dashboard_ultra.py`:

- Lines around `1153-1176`: `/api/live_prices` now converts display percent back to capital fraction:
  - `display_pos_pct = num(p.get("position_pct", 0))`
  - `pos_fraction = display_pos_pct / 100.0`
  - `pnl = round((live_px - entry) * qty if qty else INITIAL * pos_fraction * ret, 2)`
- Lines around `1281`: Open Positions header now has:
  - `livePriceBadge`
  - `runSignalsBtn`
- Lines around `1463`: the normal 30s dashboard `load()` calls `await liveRefresh()` after re-rendering so snapshot refreshes do not leave stale values on screen.
- Lines around `1469-1541`: duplicate injected JS was replaced with one guarded polling block:
  - `signedMoney`
  - `setLiveTone`
  - `applyLivePosition`
  - `liveRefresh`
  - `startLivePolling`
  - `runSignals`
  - one `document.addEventListener('DOMContentLoaded', startLivePolling)`

## Current server state

- Backup made before overwrite: `/home/abhinavsharma1359/macro_intelligence_complete/project/dashboard_ultra.py.bak_`
- Dashboard restarted and listening on port `5055`.
- Current process when checked: `python dashboard_ultra.py`, pid `20763`.
- `/tmp/dash.log` shows Flask serving and recent `GET /api/live_prices` requests with HTTP 200.

## Validation already run

Compiled on VM:

```bash
cd /home/abhinavsharma1359/macro_intelligence_complete/project
python3 -m py_compile dashboard_ultra.py
```

Checked duplicate JS counts in served HTML:

```text
livePriceBadge 2
runSignalsBtn 2
liveRefresh 1
startLivePolling 1
runSignals 1
DOMContentLoaded 1
```

Checked `/api/live_prices` formula from localhost on VM:

```text
IPGP api_pnl 18.09 expected 18.09 pct 0.804 position_fraction 0.0225
RPD  api_pnl 38.37 expected 38.37 pct 6.822 position_fraction 0.005625
CNSP api_pnl -26.38 expected -26.38 pct -4.691 position_fraction 0.005625
ACVA api_pnl 36.35 expected 36.35 pct 6.463 position_fraction 0.005625
AGPU api_pnl -39.17 expected -39.17 pct -13.929 position_fraction 0.002812
```

This proves the old `$3,566` / `$3,831` style bad PnL is fixed; numbers are now position-size adjusted.

## Safe restart command

Use this from the VM:

```bash
cd /home/abhinavsharma1359/macro_intelligence_complete/project
source ../venv/bin/activate
pkill -f '^python dashboard_ultra.py$' || true
nohup python dashboard_ultra.py > /tmp/dash.log 2>&1 < /dev/null &
sleep 2
ss -ltnp | grep 5055
tail -30 /tmp/dash.log
```

Avoid broad `pkill -f dashboard_ultra.py` inside a long SSH command; it can match the SSH shell command itself.

## Next Claude quick check

Open the dashboard, hard refresh, then in console:

```javascript
startLivePolling
liveRefresh()
fetch('/api/live_prices').then(r=>r.json()).then(console.log)
```

Expected:

- `startLivePolling` exists.
- `liveRefresh()` resolves without console errors.
- Open Positions `Current`, `P&L`, and `Return` cells update from `/api/live_prices`.
- PnL should be tens of dollars for these small positions, not thousands.
