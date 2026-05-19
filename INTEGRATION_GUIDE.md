# Integration Guide

Current public snapshot: 2026-05-19

This guide explains how the current pieces fit together without exposing private runtime details.

## External Services

### Alpaca

Used for:

- Latest US quote/bar marks.
- Missing-symbol refresh for the stock diagnostic.
- Paper order bridge for US fixed-return entries and exits.

Environment:

```text
ALPACA_API_KEY=<paper key>
ALPACA_SECRET_KEY=<paper secret>
ALPACA_PAPER=true
```

Keep `ALPACA_PAPER=true` unless live trading has been separately audited and approved.

### LLM Provider

The current LLM path uses an OpenRouter/Groq-compatible key pool stored as a comma-separated `GROQ_API_KEY` value.

```text
GROQ_API_KEY=<key1>,<key2>,<key3>
```

Behavior:

- Normal nightly LLM filtering rotates through the available keys.
- Rate-limited or exhausted keys are retired for the current run.
- Search/diagnostic can reserve or reverse key usage so ad hoc research does not burn the same first key the nightly job starts with.

Useful controls:

```text
LLM_RESERVE_LAST_KEY_FOR_SEARCH=1
LLM_SEARCH_USE_ALL_KEYS=1
```

## Data Flow

```text
Feature parquet files
    -> fixed_return_daily_signals.py
    -> ML score and hard gates
    -> LLM risk filter
    -> reports/fixed_return_daily_signals.json
    -> fixed_return_paper_execute.py
    -> reports/fixed_return_open_positions.json
    -> intraday_mark_to_market.py
    -> dashboard_ultra.py
```

The dashboard reads report files and lightweight APIs. It should not do heavy model work except for explicit single-symbol diagnostic requests.

## Stock Diagnostic Flow

```text
Dashboard symbol input
    -> /api/symbol_diagnostic
    -> single_symbol_diagnostic.py
    -> local feature lookup
    -> Alpaca refresh if missing/stale
    -> model probability
    -> gates
    -> optional LLM or Force LLM
    -> diagnostic JSON rendered on dashboard
```

Diagnostic output is intentionally more verbose than the daily signal table. It should show:

- Final verdict.
- Probability and threshold.
- Data coverage.
- Liquidity.
- Sector.
- Regime.
- Earnings state.
- LLM decision and reason.
- Feature snapshot.
- Failed gates.

`top_n` is daily capacity context only. It should not block the single-stock research answer.

## ETF and Blocklist Controls

Current block files:

```text
project/data/blocklist.txt
project/data/etf_blocklist.txt
project/data/leveraged_etf_blacklist.txt
```

These are hard blocks. If a symbol is listed here, the normal system and diagnostic should reject it before LLM approval.

## Report Files

Runtime report files are ignored by git because they contain live state:

```text
project/reports/fixed_return_daily_signals.json
project/reports/fixed_return_daily_scores.json
project/reports/fixed_return_open_positions.json
project/reports/fixed_return_paper_trades.csv
project/reports/intraday_quotes.json
project/reports/intraday_mtm_history.json
project/reports/symbol_diagnostics.json
```

For public examples, use synthetic snippets or explain the schema. Do not commit real account state.

## Troubleshooting

Dashboard does not load:

```bash
pgrep -af dashboard_ultra.py
tail -80 /tmp/dash.log
curl -s http://127.0.0.1:5055/api/snapshot | head
```

Live prices do not update:

```bash
cd project
source ../venv/bin/activate
python scripts/intraday_mark_to_market.py --force
curl -s http://127.0.0.1:5055/api/live_prices | python -m json.tool | head -80
```

Signals look too small:

```bash
cd project
source ../venv/bin/activate
python - <<'PY'
import json
from pathlib import Path
d = json.loads(Path("reports/fixed_return_daily_signals.json").read_text())
print("threshold:", d.get("threshold"))
print("top_n:", d.get("top_n"))
print("scored:", d.get("scored_count"))
print("signals:", len(d.get("signals", [])))
print("unknown:", d.get("llm_unknown_symbols"))
print("skipped:", d.get("llm_skipped_symbols"))
PY
```

LLM appears stuck on a dead key:

```bash
grep -E "LLM key|retired|active_keys|429|exhausted" project/logs/*.log logs/*.log 2>/dev/null | tail -80
```

Single-symbol diagnostic from CLI:

```bash
cd project
source ../venv/bin/activate
python scripts/single_symbol_diagnostic.py DIS --force-llm
```
