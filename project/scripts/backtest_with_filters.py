#!/usr/bin/env python3
"""
Replay the backtest with deterministic approximations of the LLM's skip/reduce rules.
Uses features already in the parquets — no API calls needed.
Compares: raw model vs model+filters vs model+filters+reduce_half sizing.
"""
import joblib, numpy as np, pandas as pd
from pathlib import Path

PROJECT      = Path('.')
FEATURES_DIR = PROJECT / 'data/features'
MODEL_PATH   = PROJECT / 'models/checkpoints/fixed_return_h10_model.joblib'
THRESHOLD    = 0.55
HOLD_DAYS    = 12
PROFIT_TARGET = 0.08
STOP_LOSS    = -0.02   # updated to 2%
TOP_N        = 15

obj     = joblib.load(MODEL_PATH)
model   = obj.get('model') if isinstance(obj, dict) else obj
features = list(
    (obj.get('features') or obj.get('feature_cols') or []) if isinstance(obj, dict)
    else (getattr(model, 'feature_names_in_', None) or [])
)

def recompute_rolling(df):
    if df.empty or 'close' not in df.columns:
        return df
    c = df['close'].astype(float)
    h = df['high'].astype(float) if 'high' in df.columns else c
    l = df['low'].astype(float) if 'low' in df.columns else c
    df = df.copy()
    df['return_1d']   = c.pct_change(1) * 100
    df['return_5d']   = c.pct_change(5) * 100
    df['return_20d']  = c.pct_change(20) * 100
    df['returns_1d']  = c.pct_change(1)
    df['sma_20']  = c.rolling(20, min_periods=10).mean()
    df['sma_50']  = c.rolling(50, min_periods=25).mean()
    df['sma_200'] = c.rolling(200, min_periods=50).mean()
    df['close_vs_sma_50']  = (c - df['sma_50'])  / df['sma_50'].replace(0,  np.nan) * 100
    df['close_vs_sma_200'] = (c - df['sma_200']) / df['sma_200'].replace(0, np.nan) * 100
    delta = c.diff(); gain = delta.clip(lower=0); loss = (-delta).clip(lower=0)
    df['rsi_14'] = 100 - 100/(1 + gain.ewm(com=13,adjust=False).mean()/loss.ewm(com=13,adjust=False).mean().replace(0,np.nan))
    df['rsi_21'] = 100 - 100/(1 + gain.ewm(com=20,adjust=False).mean()/loss.ewm(com=20,adjust=False).mean().replace(0,np.nan))
    prev_c = c.shift(1)
    tr = pd.concat([h-l,(h-prev_c).abs(),(l-prev_c).abs()],axis=1).max(axis=1)
    df['atr_14'] = tr.ewm(com=13,adjust=False).mean()
    df['atr_pct'] = df['atr_14'] / c.replace(0,np.nan) * 100
    bb_mid = c.rolling(20,min_periods=10).mean(); bb_std = c.rolling(20,min_periods=10).std()
    bb_range = (4*bb_std).replace(0,np.nan)
    df['bb_width']    = bb_range / bb_mid.replace(0,np.nan)
    df['bb_position'] = (c-(bb_mid-2*bb_std)) / bb_range
    ema12=c.ewm(span=12,adjust=False).mean(); ema26=c.ewm(span=26,adjust=False).mean()
    df['macd']=ema12-ema26; df['macd_signal']=df['macd'].ewm(span=9,adjust=False).mean()
    df['macd_hist']=df['macd']-df['macd_signal']
    df['momentum_20d']=c.pct_change(20)*100; df['momentum_60d']=c.pct_change(60)*100
    daily_ret=c.pct_change()
    df['realized_vol_21d']=daily_ret.rolling(21,min_periods=10).std()*np.sqrt(252)
    vs=daily_ret.rolling(21,min_periods=10).std(); vl=daily_ret.rolling(63,min_periods=30).std().replace(0,np.nan)
    df['vol_regime_ratio']=vs/vl
    df['vol_regime_stressed']=(df['vol_regime_ratio']>1.5).astype(float)
    df['zscore_vs_60d']=(c-c.rolling(60,min_periods=20).mean())/c.rolling(60,min_periods=20).std().replace(0,np.nan)
    df['price_acceleration']=df['return_1d']-df['return_1d'].shift(1)
    return df

# ── LLM rule approximations ──────────────────────────────────────────────────
def llm_decision(row, spy_return_1d=0.0):
    """
    Returns: 'skip', 'reduce_half', or 'proceed'
    Mirrors the LLM system prompt rules using features available in parquets.
    """
    r1   = float(row.get('return_1d',  0) or 0)
    r5   = float(row.get('return_5d',  0) or 0)
    evt  = float(row.get('event_day_extreme', 0) or 0)
    off  = float(row.get('official_event_signal', 0) or 0)
    vstr = float(row.get('vol_regime_stressed', 0) or 0)
    rsi  = float(row.get('rsi_14', 50) or 50)

    # ── SKIP rules ──────────────────────────────────────────────────────────
    # Extreme down day — model scoring a broken chart
    if r1 < -8:
        return 'skip'
    # Post-earnings collapse (event_day_extreme fired + large drop)
    if evt == 1 and r1 < -10:
        return 'skip'
    # Chasing: up >15% in 5 days with RSI overbought
    if r5 > 15 and rsi > 75:
        return 'skip'
    # Severe market risk-off (SPY proxy from spy_return_1d)
    if spy_return_1d < -1.5 and r1 < spy_return_1d:   # no relative strength
        return 'skip'

    # ── REDUCE_HALF rules ────────────────────────────────────────────────────
    # Mild risk-off
    if spy_return_1d < -1.0:
        return 'reduce_half'
    # Earnings imminent (official event signal on or near)
    if off == 1:
        return 'reduce_half'
    # Gapped up hard today — entry risk
    if r1 > 7:
        return 'reduce_half'
    # Elevated vol regime — fragile market
    if vstr == 1:
        return 'reduce_half'

    return 'proceed'

# ── Load SPY for market context ───────────────────────────────────────────────
spy_df = None
for name in ['SPY', 'SPY_US']:
    p = FEATURES_DIR / f'{name}.parquet'
    if p.exists():
        spy_df = pd.read_parquet(p)
        spy_df = recompute_rolling(spy_df)
        spy_df = spy_df.set_index('date') if 'date' in spy_df.columns else spy_df
        break
print(f"SPY loaded: {spy_df is not None}")

# ── Load all US parquets ──────────────────────────────────────────────────────
print("Loading parquets...")
all_data = {}
for p in sorted(FEATURES_DIR.glob('*.parquet')):
    if any(p.stem.endswith(x) for x in ('.NS','.BO')): continue
    try:
        df = pd.read_parquet(p)
        df = recompute_rolling(df)
        sym = p.stem.replace('_US','').replace('.US','')
        if 'close' in df.columns and len(df) > 60:
            all_data[sym] = df
    except: pass
print(f"Loaded {len(all_data)} symbols")

# ── Walk dates ────────────────────────────────────────────────────────────────
all_dates = sorted(set(
    idx for df in all_data.values()
    for idx in df.index
))

trades = []
for i, entry_date in enumerate(all_dates[:-HOLD_DAYS-5]):
    # SPY return on this date
    spy_r1 = 0.0
    if spy_df is not None:
        try:
            spy_r1 = float(spy_df.loc[entry_date, 'return_1d'])
            if np.isnan(spy_r1): spy_r1 = 0.0
        except: pass

    day_rows = []
    for sym, df in all_data.items():
        try:
            if entry_date not in df.index: continue
            idx = df.index.get_loc(entry_date)
            if idx < 50: continue
            row = df.iloc[idx]
            if float(row.get('close',0) or 0) < 5: continue
            if float(row.get('adv20_dollar_vol',0) or 0) < 5_000_000: continue
            feat_row = {c: float(row.get(c,0) or 0) for c in features}
            day_rows.append((sym, feat_row, df, idx, row))
        except: pass

    if not day_rows: continue

    syms_l, feat_rows, dfs_l, idxs_l, raw_rows = zip(*day_rows)
    X = pd.DataFrame(list(feat_rows))[features].replace([np.inf,-np.inf],np.nan).fillna(0).astype('float32')
    probas = model.predict_proba(X)[:,1]

    top_idx = np.where(probas >= THRESHOLD)[0]
    top_idx = top_idx[np.argsort(probas[top_idx])[::-1]][:TOP_N]

    for ti in top_idx:
        sym      = syms_l[ti]
        df       = dfs_l[ti]
        idx      = idxs_l[ti]
        prob     = probas[ti]
        row      = raw_rows[ti]
        decision = llm_decision(row, spy_r1)
        size_mult = 0.0 if decision == 'skip' else (0.5 if decision == 'reduce_half' else 1.0)
        try:
            entry_price = float(df['close'].iloc[idx])
            exit_idx    = min(idx + HOLD_DAYS, len(df)-1)
            exit_price  = float(df['close'].iloc[exit_idx])
            raw_ret     = (exit_price - entry_price) / entry_price
            clamped_ret = max(STOP_LOSS, min(PROFIT_TARGET, raw_ret))
            trades.append({
                'date':              str(entry_date)[:10],
                'symbol':            sym,
                'probability':       round(prob,4),
                'decision':          decision,
                'size_mult':         size_mult,
                'raw_return_pct':    round(raw_ret*100,3),
                'clamped_return_pct': round(clamped_ret*100,3),
                'sized_return_pct':  round(clamped_ret*size_mult*100,3),
            })
        except: pass

    if i % 500 == 0:
        print(f"  {str(entry_date)[:10]}  trades so far: {len(trades)}")

df_t = pd.DataFrame(trades)
df_t.to_csv('reports/backtest_filtered_trades.csv', index=False)

# ── Results comparison ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  RESULTS COMPARISON")
print("="*60)

raw  = df_t
filt = df_t[df_t['decision'] != 'skip']
proc = df_t[df_t['decision'] == 'proceed']

for label, sub, col in [
    ('RAW model (no filter)',        raw,  'clamped_return_pct'),
    ('After skip filter',            filt, 'clamped_return_pct'),
    ('Proceed only (LLM confident)', proc, 'clamped_return_pct'),
    ('Sized (skip=0, reduce=0.5x)',  raw,  'sized_return_pct'),
]:
    r = sub[col]
    pf = r[r>0].sum()/abs(r[r<0].sum()) if r[r<0].sum()!=0 else 999
    sh = r.mean()/r.std() if r.std()>0 else 0
    print(f"\n  {label}")
    print(f"    n={len(sub):6d}  win={r.gt(0).mean():.1%}  mean={r.mean():.3f}%  PF={pf:.2f}  Sharpe={sh:.3f}")

print(f"\n  Skipped: {(df_t.decision=='skip').sum()} trades ({(df_t.decision=='skip').mean():.1%})")
print(f"  Reduced: {(df_t.decision=='reduce_half').sum()} trades ({(df_t.decision=='reduce_half').mean():.1%})")
print(f"  Proceed: {(df_t.decision=='proceed').sum()} trades ({(df_t.decision=='proceed').mean():.1%})")
print("="*60)
