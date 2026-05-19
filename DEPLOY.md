# Deployment Runbook

Current public snapshot: 2026-05-19

This runbook describes the current paper-first deployment shape. It avoids private VM addresses, account names, and secrets.

## 1. Prepare Environment

```bash
cd macro_intelligence_complete
python3 -m venv venv
source venv/bin/activate
pip install -r project/requirements.txt
cp project/.env.example project/.env
```

Edit `project/.env` locally on the server. Do not commit it.

Required private values:

```text
ALPACA_API_KEY=<paper key>
ALPACA_SECRET_KEY=<paper secret>
ALPACA_PAPER=true
GROQ_API_KEY=<comma-separated LLM keys>
```

Useful runtime controls:

```text
DASHBOARD_ULTRA_PORT=5055
SIG_THRESHOLD=0.61
SIG_TOP_N=30
LLM_RESERVE_LAST_KEY_FOR_SEARCH=1
LLM_SEARCH_USE_ALL_KEYS=1
```

## 2. Start Dashboard

Preferred:

```bash
./start.sh
```

Manual:

```bash
cd project
source ../venv/bin/activate
nohup python dashboard_ultra.py > /tmp/dash.log 2>&1 &
```

Verify:

```bash
pgrep -af dashboard_ultra.py
tail -50 /tmp/dash.log
curl -s http://127.0.0.1:5055/api/snapshot | python -m json.tool | head -40
```

Stop:

```bash
./stop.sh
```

## 3. Daily US Signal Run

```bash
cd project
source ../venv/bin/activate
python scripts/fixed_return_daily_signals.py
```

Expected outputs:

```text
project/reports/fixed_return_daily_signals.json
project/reports/fixed_return_daily_signals.csv
project/reports/fixed_return_daily_scores.json
```

Inspect the run:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("reports/fixed_return_daily_signals.json")
d = json.loads(p.read_text())
print("signal_date:", d.get("signal_date"))
print("threshold:", d.get("threshold"))
print("scored:", d.get("scored_count"))
print("signals:", len(d.get("signals", [])))
for s in d.get("signals", []):
    print(s.get("rank"), s.get("symbol"), s.get("probability"), s.get("llm_decision"))
PY
```

## 4. Paper Execute

```bash
cd project
source ../venv/bin/activate
python scripts/fixed_return_paper_execute.py
```

This updates paper open positions and evaluates exits. If the Alpaca bridge is configured, it can submit paper orders.

## 5. Intraday MTM

```bash
cd project
source ../venv/bin/activate
python scripts/intraday_mark_to_market.py --force
```

This refreshes open-position prices and P&L from Alpaca. Without `--force`, it skips outside US market hours.

## 6. Stock Diagnostic

CLI:

```bash
cd project
source ../venv/bin/activate
python scripts/single_symbol_diagnostic.py INTC --force-llm
```

Dashboard:

```text
Open the dashboard -> Stock Diagnostic -> enter symbol -> Run Check.
```

Use Force LLM when you want explanation for a rejected stock. It is research-only if production gates fail.

## 7. Cron Shape

Use cron or a scheduler to run the three production jobs separately. Example only:

```cron
# Refresh daily features before signal generation.
30 16 * * 1-5 cd /path/to/macro_intelligence_complete/project && ../venv/bin/python scripts/daily_data_refresh.py >> logs/daily_data_refresh.log 2>&1

# Generate US fixed-return candidates.
00 17 * * 1-5 cd /path/to/macro_intelligence_complete/project && ../venv/bin/python scripts/fixed_return_daily_signals.py >> logs/daily_signals.log 2>&1

# Paper execute after signals are written.
10 17 * * 1-5 cd /path/to/macro_intelligence_complete/project && ../venv/bin/python scripts/fixed_return_paper_execute.py >> logs/paper_execute.log 2>&1

# Intraday mark-to-market while the US market is open.
*/30 9-16 * * 1-5 cd /path/to/macro_intelligence_complete/project && ../venv/bin/python scripts/intraday_mark_to_market.py >> logs/intraday_mtm.log 2>&1
```

Adjust times to the server timezone and market calendar.

## 8. Public Repo Hygiene

Never commit:

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

Before pushing:

```bash
git status --short
git diff --check
git grep -n "OPENROUTER_KEY_PREFIX\\|PRIVATE_KEY_PLACEHOLDER\\|REAL_VM_ADDRESS" -- . || true
```
