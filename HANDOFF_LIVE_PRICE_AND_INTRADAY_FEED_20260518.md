# Handoff: live dashboard prices and intraday universe feed

Date: 2026-05-18
VM: `abhinavsharma1359@34.61.149.100`
Project: `/home/abhinavsharma1359/macro_intelligence_complete`
Dashboard: `project/dashboard_ultra.py`
Dashboard URL: `http://34.61.149.100:5055`

## Dashboard fix completed

The live dashboard had two separate stale paths:

- `/api/live_prices` was live, but only the Open Positions table used it.
- `/api/snapshot` still rendered stale `reports/fixed_return_open_positions.json` values into both Open Positions and Unrealized P&L Attribution every 30 seconds.
- The old helper used Alpaca latest bars, so prices only moved when a new minute bar printed.

Fixed in `project/dashboard_ultra.py`:

- `alpaca_live_prices()` now uses Alpaca latest trade first, quote midpoint only when trade is stale and quote spread is sane, and bar only as fallback.
- New `live_mark_positions()` applies the same live mark to position rows.
- `/api/snapshot` now calls `portfolio(use_live=True)`, so the snapshot path and attribution table no longer repaint stale current prices.
- `/api/live_prices` now returns `source`, `timestamp`, `stale_seconds`, and `quote_spread_pct`.
- Browser JS now updates both:
  - Open Positions table `#pos`
  - Unrealized P&L Attribution table `#usAttribution`
- Live Summary and `usAttrBadge` update from the same live payload.
- Money formatting now renders negative dollars as `-$43.79` instead of `$-43.79`.

Current dashboard process:

```bash
pgrep -af 'python dashboard_ultra.py'
# 27808 python dashboard_ultra.py
```

Restart command:

```bash
cd /home/abhinavsharma1359/macro_intelligence_complete/project
source ../venv/bin/activate
pkill -f '^python dashboard_ultra.py$' || true
setsid -f python dashboard_ultra.py > /tmp/dash.log 2>&1 < /dev/null
```

## Dashboard verification already done

Backend:

```bash
curl -sS http://127.0.0.1:5055/api/live_prices | python3 -m json.tool
curl -sS http://127.0.0.1:5055/api/snapshot -o /tmp/snapshot_live.json
```

Observed live API after fix:

```text
IPGP 103.49 pnl 19.30 source trade
RPD  6.765 pnl 48.30 source trade
CNSP 4.62  pnl -43.79 source trade
ACVA 6.065 pnl 33.41 source trade
AGPU 4.835 pnl -38.41 source quote_mid
```

Observed snapshot after fix:

```text
portfolio:
IPGP 103.51 19.73 0.877 trade
RPD  6.765 48.30 8.587 trade
CNSP 4.62 -43.79 -7.784 trade
ACVA 6.065 33.41 5.939 trade
AGPU 4.835 -38.41 -13.661 quote_mid

attribution:
CNSP 4.62 -43.79 -7.784
AGPU 4.835 -38.41 -13.661
RPD  6.765 48.30 8.587
ACVA 6.065 33.41 5.939
IPGP 103.51 19.73 0.877
```

Browser DOM verification:

- Main table and attribution table matched the live values.
- LIVE badge updated and tooltip showed source/timestamp per symbol:

```text
IPGP: trade @ 2026-05-18 16:26:23.876832+00:00
RPD: trade @ 2026-05-18 16:25:43.791940+00:00
CNSP: trade @ 2026-05-18 16:19:38.794936+00:00
ACVA: trade @ 2026-05-18 16:25:46.391493+00:00
AGPU: quote_mid @ 2026-05-18 16:22:53.811266+00:00
```

Important note: illiquid symbols can still look "stuck" when Alpaca has no newer trade and quote spread is too wide. The UI now exposes that via `source`, `timestamp`, and staleness instead of silently pretending.

## Intraday universe feed added

New file:

```text
project/scripts/intraday_universe_refresh.py
```

Purpose:

- Refresh `data/features/*.parquet` with Alpaca intraday marks.
- Uses latest trade first, quote-mid fallback only when spread is sane.
- Appends or updates today's row.
- Recomputes core rolling features used by `fixed_return_daily_signals.py`.
- Updates filesystem mtime so the existing signal script freshness gate sees the file as fresh honestly.
- Writes report:

```text
reports/fixed_return_intraday_universe_refresh.json
```

Smoke tests already run:

```bash
cd /home/abhinavsharma1359/macro_intelligence_complete/project
source ../venv/bin/activate
python3 -m py_compile scripts/intraday_universe_refresh.py
python scripts/intraday_universe_refresh.py --symbols IPGP,RPD,ACVA,AGPU,CNSP --dry-run --force
python scripts/intraday_universe_refresh.py --symbols IPGP,RPD,ACVA,AGPU,CNSP --force
```

Real write result for the 5 open symbols:

```text
quotes_ok=5 files_updated=5 files_failed=0
ACVA close 6.05 source trade
AGPU close 4.835 source quote_mid
CNSP close 4.62 source trade
IPGP close 103.51 source trade
RPD close 6.695 source trade
```

Verified parquet state:

```text
IPGP last_date 2026-05-18 close 103.51 source trade
RPD  last_date 2026-05-18 close 6.695 source trade
ACVA last_date 2026-05-18 close 6.05 source trade
AGPU last_date 2026-05-18 close 4.835 source quote_mid
CNSP last_date 2026-05-18 close 4.62 source trade
```

## Full-universe command

Use this for a real full refresh during market hours:

```bash
cd /home/abhinavsharma1359/macro_intelligence_complete/project
source ../venv/bin/activate
python scripts/intraday_universe_refresh.py --batch-size 200
```

Use this for a bounded smoke:

```bash
python scripts/intraday_universe_refresh.py --limit 50 --dry-run --force
python scripts/intraday_universe_refresh.py --limit 50 --force
```

After a refresh, run signals with LLM off for fast operational testing:

```bash
LLM_FILTER_ENABLED=0 python scripts/fixed_return_daily_signals.py --dry-run
```

The attempted `--limit 20 --dry-run` signal run took longer than the timeout and was killed. Do not leave it running in the background.

## Remaining work

- Schedule `intraday_universe_refresh.py` every 30 minutes during US market hours if desired.
- Run a full-universe refresh once during market hours and then rerun `fixed_return_daily_signals.py`.
- Consider disabling or caching yfinance-heavy earnings/news checks during intraday signal runs, because they can make a small dry-run hang for minutes.
