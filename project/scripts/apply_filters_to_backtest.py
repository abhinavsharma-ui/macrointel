#!/usr/bin/env python3
"""
Apply LLM-approximation skip/reduce rules to the existing backtest_trades.csv.
Loads feature values from parquets for each trade date to get rule inputs.
No ADV filter — preserves the original 85,473 trade universe.
"""
import numpy as np, pandas as pd, joblib
from pathlib import Path

PROJECT      = Path('.')
FEATURES_DIR = PROJECT / 'data/features'
df = pd.read_csv('reports/backtest_trades.csv')
df['date'] = pd.to_datetime(df['date'])
print(f"Loaded {len(df)} trades from backtest_trades.csv")

# Load SPY for daily market context
spy_df = None
for name in ['SPY','SPY_US']:
    p = FEATURES_DIR / f'{name}.parquet'
    if p.exists():
        spy_df = pd.read_parquet(p)
        if 'date' in spy_df.columns:
            spy_df = spy_df.set_index('date')
        spy_df.index = pd.to_datetime(spy_df.index)
        spy_df['spy_r1'] = spy_df['close'].pct_change() * 100
        break
print(f"SPY loaded: {spy_df is not None}")

# Load all parquets into a lookup: {symbol: df}
print("Loading parquets for feature lookup...")
parquet_data = {}
for p in sorted(FEATURES_DIR.glob('*.parquet')):
    if any(p.stem.endswith(x) for x in ('.NS','.BO')): continue
    try:
        sym = p.stem.replace('_US','').replace('.US','')
        pf = pd.read_parquet(p)
        if 'date' in pf.columns:
            pf = pf.set_index('date')
        pf.index = pd.to_datetime(pf.index)
        parquet_data[sym] = pf
    except: pass
print(f"Loaded {len(parquet_data)} symbol parquets")

def get_features(sym, date):
    df_sym = parquet_data.get(sym)
    if df_sym is None: return {}
    try:
        row = df_sym.loc[date]
        if isinstance(row, pd.DataFrame): row = row.iloc[0]
        return row.to_dict()
    except:
        # Try nearest date
        try:
            idx = df_sym.index.searchsorted(date)
            if idx < len(df_sym):
                return df_sym.iloc[idx].to_dict()
        except: pass
    return {}

def get_spy_r1(date):
    if spy_df is None: return 0.0
    try:
        row = spy_df.loc[date, 'spy_r1']
        return float(row) if not np.isnan(float(row)) else 0.0
    except:
        try:
            idx = spy_df.index.searchsorted(date)
            if idx < len(spy_df):
                return float(spy_df.iloc[idx]['spy_r1'])
        except: pass
    return 0.0

def llm_decision(feats, spy_r1):
    r1   = float(feats.get('return_1d',  0) or 0)
    r5   = float(feats.get('return_5d',  0) or 0)
    evt  = float(feats.get('event_day_extreme', 0) or 0)
    off  = float(feats.get('official_event_signal', 0) or 0)
    vstr = float(feats.get('vol_regime_stressed', 0) or 0)
    rsi  = float(feats.get('rsi_14', 50) or 50)

    # SKIP
    if r1 < -8: return 'skip'
    if evt == 1 and r1 < -10: return 'skip'
    if r5 > 15 and rsi > 75: return 'skip'
    if spy_r1 < -1.5 and r1 < spy_r1: return 'skip'

    # REDUCE
    if spy_r1 < -1.0: return 'reduce_half'
    if off == 1: return 'reduce_half'
    if r1 > 7: return 'reduce_half'
    if vstr == 1: return 'reduce_half'

    return 'proceed'

# Apply rules to each trade
print("Applying LLM rules to all trades...")
decisions, size_mults = [], []
dates = df['date'].unique()
spy_cache = {d: get_spy_r1(d) for d in dates}

for i, row in df.iterrows():
    feats   = get_features(row['symbol'], row['date'])
    spy_r1  = spy_cache.get(row['date'], 0.0)
    dec     = llm_decision(feats, spy_r1)
    mult    = 0.0 if dec=='skip' else (0.5 if dec=='reduce_half' else 1.0)
    decisions.append(dec)
    size_mults.append(mult)
    if (i+1) % 10000 == 0:
        print(f"  {i+1:6d}/{len(df)} done...", flush=True)

df['decision']   = decisions
df['size_mult']  = size_mults
df['sized_return_pct'] = df['clamped_return_pct'] * df['size_mult']
df.to_csv('reports/backtest_filtered_trades.csv', index=False)
print(f"\nSaved {len(df)} rows to reports/backtest_filtered_trades.csv")

# Print comparison
def show(label, sub, col):
    r = sub[col]
    pf = r[r>0].sum()/abs(r[r<0].sum()) if r[r<0].sum()!=0 else 999
    sh = r.mean()/r.std() if r.std()>0 else 0
    print(f"\n  {label}")
    print(f"    n={len(sub):6d}  win={r.gt(0).mean():.1%}  mean={r.mean():.3f}%  PF={pf:.2f}  Sharpe={sh:.3f}")

print("\n" + "="*60)
print("  RESULTS: RAW vs FILTERED")
print("="*60)
show('RAW model (no filter)',        df, 'clamped_return_pct')
show('After skip filter',            df[df['decision']!='skip'], 'clamped_return_pct')
show('Proceed only',                 df[df['decision']=='proceed'], 'clamped_return_pct')
show('Sized (skip=0, reduce=0.5x)', df, 'sized_return_pct')

print(f"\n  Skipped : {(df.decision=='skip').sum():6d} ({(df.decision=='skip').mean():.1%})")
print(f"  Reduced : {(df.decision=='reduce_half').sum():6d} ({(df.decision=='reduce_half').mean():.1%})")
print(f"  Proceed : {(df.decision=='proceed').sum():6d} ({(df.decision=='proceed').mean():.1%})")
print("="*60)
