#!/usr/bin/env python3
"""LLM Alpha Test v8 — per-call gap + 429 retry"""
import os, sys, time, csv
from pathlib import Path

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL        = "deepseek/deepseek-v4-flash:free"
MODELS = [
    "deepseek/deepseek-v4-flash:free",
    "google/gemma-4-26b-a4b-it:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
]
MIN_INTERVAL = 5
MIN_CALL_GAP = 3.0

PROJECT_ROOT = Path(__file__).parent.parent
PROD_SCRIPT  = PROJECT_ROOT / "scripts" / "fixed_return_daily_signals.py"
TRADES_CSV   = PROJECT_ROOT / "reports" / "backtest_trades.csv"
RESULTS_CSV  = PROJECT_ROOT / "reports" / "llm_alpha_test_results.csv"

if not GROQ_API_KEY:
    raise SystemExit("GROQ_API_KEY is required")

os.environ["GROQ_API_KEY"]       = GROQ_API_KEY
os.environ["LLM_FILTER_ENABLED"] = "1"
os.environ["LLM_FILTER_MODEL"]   = MODEL

import requests as _req_mod
import requests.adapters as _req_adapters

_orig_send   = _req_adapters.HTTPAdapter.send
_last_call_t = [0.0]

def _send_with_retry(self, request, *a, **kw):
    if "groq.com" in (request.url or ""):
        gap = MIN_CALL_GAP - (time.time() - _last_call_t[0])
        if gap > 0:
            time.sleep(gap)
    for attempt in range(6):
        resp = _orig_send(self, request, *a, **kw)
        if "groq.com" in (request.url or ""):
            _last_call_t[0] = time.time()
        if resp.status_code == 429:
            wait = min(90, int(resp.headers.get("Retry-After", 60)))
            print(f"  [429] waiting {wait}s (attempt {attempt+1}/6)", flush=True)
            time.sleep(wait + 2)
            _last_call_t[0] = time.time()
            continue
        return resp
    return resp

_req_adapters.HTTPAdapter.send = _send_with_retry
print("✓ HTTPAdapter.send patched — per-call gap enforced at transport level")

print(f"Loading {PROD_SCRIPT} ...")
src = PROD_SCRIPT.read_text()
for guard in ('if __name__ == "__main__":', "if __name__ == '__main__':"):
    src = src.replace(guard, 'if False:')
_ns = {"__name__": "__exec__", "__file__": str(PROD_SCRIPT)}
try:
    exec(compile(src, str(PROD_SCRIPT), 'exec'), _ns)
except SystemExit:
    pass
except Exception as e:
    print(f"WARNING during exec: {e}")

call_llm_filter    = _ns.get('call_llm_filter')
market_context_pct = _ns.get('market_context_pct')
if call_llm_filter is None:
    print("ERROR: call_llm_filter not found"); sys.exit(1)
print(f"✓ call_llm_filter loaded  |  model={MODEL}")

import pandas as pd

df_trades = pd.read_csv(TRADES_CSV)
df_trades['date'] = pd.to_datetime(df_trades['date'])
df_test = df_trades[
    (df_trades['date'] >= pd.Timestamp('2026-03-16')) &
    (df_trades['date'] <= pd.Timestamp('2026-04-06'))
].copy().reset_index(drop=True).head(114)
if len(df_test) == 0:
    df_test = df_trades.tail(225).copy().reset_index(drop=True)

print(f"Trades: {len(df_test)}  ({str(df_test['date'].min())[:10]} -> {str(df_test['date'].max())[:10]})")
print(f"Est time: ~{len(df_test)*MIN_INTERVAL//60} min")

done_keys = set()
if RESULTS_CSV.exists():
    df_done = pd.read_csv(RESULTS_CSV)
    done_keys = set(zip(df_done['date'].astype(str).str[:10], df_done['symbol']))
    print(f"Resuming: {len(done_keys)} done")

FIELDS = ['idx','date','symbol','probability',
          'raw_return_pct','clamped_return_pct','hit_target','hit_stop',
          'llm_decision','llm_confidence','llm_reasoning','error']

write_header = not RESULTS_CSV.exists()
fout   = open(RESULTS_CSV, 'a', newline='')
writer = csv.DictWriter(fout, fieldnames=FIELDS)
if write_header:
    writer.writeheader(); fout.flush()

total = len(df_test)
for i, row in df_test.iterrows():
    sym      = str(row['symbol'])
    date_str = str(row['date'])[:10]
    key      = (date_str, sym)
    if key in done_keys:
        continue

    seq = len(done_keys) + 1
    t0  = time.time()
    print(f"\n[{seq}/{total}] {sym}  {date_str}  prob={row['probability']:.3f}  actual={row['clamped_return_pct']:+.2f}%")

    os.environ["LLM_FILTER_MODEL"] = MODELS[(seq+1) % len(MODELS)]
    candidate = {'symbol': sym, 'probability': float(row['probability']), 'date': date_str}
    skip_cols = {'symbol','date','entry_price','exit_price','raw_return_pct',
                 'clamped_return_pct','hit_target','hit_stop','idx','index'}
    for c in row.index:
        if c in skip_cols: continue
        try:
            v = float(row[c])
            if v == v: candidate[c] = v
        except: pass

    spy_pct, qqq_pct = 0.0, 0.0
    if market_context_pct:
        try: spy_pct, qqq_pct = market_context_pct(date_str)
        except: pass

    out = {
        'idx': i, 'date': date_str, 'symbol': sym,
        'probability': row['probability'],
        'raw_return_pct': row['raw_return_pct'],
        'clamped_return_pct': row['clamped_return_pct'],
        'hit_target': row['hit_target'], 'hit_stop': row['hit_stop'],
        'llm_decision': 'error', 'llm_confidence': '',
        'llm_reasoning': '', 'error': '',
    }

    try:
        decisions = call_llm_filter([candidate], spy_pct, qqq_pct, num_signals=1)
        d = decisions.get(sym) or decisions.get(sym.upper()) or {}
        if isinstance(d, dict):
            out['llm_decision']   = str(d.get('decision',   'proceed')).lower()
            out['llm_confidence'] = str(d.get('reason',     ''))
            out['llm_reasoning']  = str(d.get('scratchpad', ''))[:300]
        elif isinstance(d, str):
            out['llm_decision'] = d.strip().lower()
        else:
            out['llm_decision'] = 'proceed'
        if not decisions:
            out['llm_decision'] = 'error'
            out['error'] = 'empty decisions — likely 429'
        print(f"  -> LLM: {out['llm_decision'].upper()}")
    except Exception as e:
        out['error'] = str(e)[:250]
        print(f"  X Error: {e}")

    writer.writerow(out); fout.flush()
    done_keys.add(key)

    if seq < total:
        elapsed = time.time() - t0
        wait = max(0, MIN_INTERVAL - elapsed)
        print(f"  took {elapsed:.1f}s  sleeping {wait:.1f}s")
        if wait > 0: time.sleep(wait)

fout.close()

print("\n" + "="*60)
print("LLM ALPHA TEST — RESULTS")
print("="*60)
df_r  = pd.read_csv(RESULTS_CSV)
df_ok = df_r[df_r['error'].fillna('') == ''].copy()
df_ok['win'] = df_ok['clamped_return_pct'] > 0
proceed = df_ok[df_ok['llm_decision'].isin(['proceed','reduce_half'])]
skipped = df_ok[df_ok['llm_decision'] == 'skip']
all_ret = df_ok['clamped_return_pct'].mean()
print(f"\nTotal: {len(df_r)}  clean: {len(df_ok)}  errors: {len(df_r)-len(df_ok)}")
print(f"Baseline  win={df_ok['win'].mean():.1%}  mean={all_ret:+.3f}%")
print(f"\nPROCEED/REDUCE n={len(proceed)}")
if len(proceed): print(f"  win={proceed['win'].mean():.1%}  mean={proceed['clamped_return_pct'].mean():+.3f}%")
print(f"\nSKIP n={len(skipped)}")
if len(skipped): print(f"  win={skipped['win'].mean():.1%}  mean={skipped['clamped_return_pct'].mean():+.3f}%")
if len(proceed) and len(df_ok):
    print(f"\nLLM lift: {proceed['clamped_return_pct'].mean() - all_ret:+.3f}%/trade vs baseline")
