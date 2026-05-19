# Current Deliverables

Current public snapshot: 2026-05-19

This file replaces the old one-off WebSocket/options deliverable list. The current deliverable is the operating MacroIntel system.

## Operator Dashboard

- `project/dashboard_ultra.py`
- US paper trading dashboard.
- NSE paper dashboard.
- Alpaca live price refresh.
- Open positions, P&L attribution, volatility regime, and daily signals.
- Manual `RUN SIGNALS` button.
- Stock Diagnostic panel with optional LLM and Force LLM research mode.

## US Fixed-Return System

- `project/scripts/fixed_return_daily_signals.py`
- ML probability scoring.
- ETF/blocklist enforcement.
- Price, liquidity, data, sector, earnings, and regime gates.
- LLM risk filter with tool evidence.
- Multi-key rotation and in-run retirement of exhausted keys.
- Signal JSON/CSV outputs.

## Paper Execution

- `project/scripts/fixed_return_paper_execute.py`
- Reads daily signals.
- Opens paper positions.
- Evaluates stop loss, profit target, blocked symbols, and hold-period exits.
- Uses Alpaca paper bridge when configured.

## Intraday Mark-to-Market

- `project/scripts/intraday_mark_to_market.py`
- Alpaca latest-bar marks.
- Open-position P&L refresh.
- Intraday quote report.
- Shadow signal marks without extra orders.

## Stock Diagnostic

- `project/scripts/single_symbol_diagnostic.py`
- Dashboard route: `POST /api/symbol_diagnostic`.
- Checks system availability, Alpaca availability, feature freshness, ML probability, gates, and LLM reasoning.
- Force LLM lets the operator see research reasoning even when normal production gates fail.
- Daily top-N capacity is displayed as context, not used as a blocker.

## NSE Paper Lane

- Rendered in `project/dashboard_ultra.py`.
- Separate signal, position, regime, and P&L panels.
- Separate from US Alpaca execution.
- Must be preserved when cleaning old scripts.

## Public Documentation

- `README.md` - current overview and commands.
- `STRATEGY.md` - current strategy and gates.
- `DEPLOY.md` - deployment and cron runbook.
- `INTEGRATION_GUIDE.md` - integrations and troubleshooting.
- `go_live_trading_strategy.md` - live-readiness plan.

## Not Committed

These are runtime/private artifacts:

```text
project/.env
project/reports/
project/logs/
project/data/features/
project/data/prices_full/
*.parquet
*.joblib
*.pkl
*.log
```

The public repo should describe the system and include source code, but not expose credentials, broker state, generated reports, VM addresses, or account-specific logs.
