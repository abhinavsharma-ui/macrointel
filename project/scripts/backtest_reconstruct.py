import joblib, numpy as np, pandas as pd
from pathlib import Path

PROJECT=Path('.'); HOLD_DAYS=10; THRESHOLD=0.55; PROFIT_TARGET=0.08; STOP_LOSS=-0.03; TOP_N=15; MIN_PRICE=5.0; MIN_ADV=5_000_000

FEATURES_DIR=(PROJECT/'data/features_26yr_liquid' if (PROJECT/'data/features_26yr_liquid').exists() else PROJECT/'data/features')
MODEL_PATHS=[PROJECT/'models/checkpoints/fixed_return_h10_model.joblib',PROJECT/'models/fixed_return_h10_model.joblib']

model_path=next((p for p in MODEL_PATHS if p.exists()),None)
if model_path is None: raise SystemExit("model not found")
obj=joblib.load(model_path); model=obj.get('model') if isinstance(obj,dict) else obj
features=list((obj.get('features') or obj.get('feature_cols') or []) if isinstance(obj,dict) else (getattr(model,'feature_names_in_',None) or []))
print(f"Model: {model_path.name}  Features: {len(features)}  DataDir: {FEATURES_DIR}")

def load_parquet(p):
    try: df=pd.read_parquet(p)
    except: return None
    if df.empty or 'close' not in df.columns: return None
    date_vals=pd.to_datetime(df['date'],errors='coerce') if 'date' in df.columns else pd.to_datetime(df.index,errors='coerce')
    df=df.reset_index(drop=True); df=df.loc[:,~df.columns.duplicated()].copy()
    if 'date' in df.columns: df=df.drop(columns=['date'])
    df['date']=date_vals.values
    if 'adv20_dollar_vol' not in df.columns and {'close','volume'}.issubset(df.columns):
        df['adv20_dollar_vol']=(df['close']*df['volume']).rolling(20,min_periods=5).mean()
    df=df.dropna(subset=['date','close']).sort_values('date').reset_index(drop=True)
    return df if len(df)>60 else None

print(f"Loading from {FEATURES_DIR}...")
all_data={}
for p in sorted(FEATURES_DIR.glob('*.parquet')):
    sym=p.stem.replace('_US','').replace('.US','')
    if sym.endswith(('.NS','.BO')): continue
    df=load_parquet(p)
    if df is not None: all_data[sym]=df
print(f"Loaded {len(all_data)} symbols")

all_dates=sorted({d for df in all_data.values() for d in df['date'] if pd.notna(d)})
print(f"Date range: {str(all_dates[0])[:10]} to {str(all_dates[-1])[:10]}  ({len(all_dates)/252:.1f} years)")

trades=[]
for i,entry_date in enumerate(all_dates[:-(HOLD_DAYS+5)]):
    day_rows=[]
    for sym,df in all_data.items():
        mask=df['date']==entry_date
        if not mask.any(): continue
        idx=int(df.index[mask][0])
        if idx<50 or idx+HOLD_DAYS+1>=len(df): continue
        row=df.iloc[idx]
        px=float(row.get('close',0) or 0); adv=float(row.get('adv20_dollar_vol',0) or 0)
        if px<MIN_PRICE or (adv>0 and adv<MIN_ADV): continue
        day_rows.append((sym,{c:float(row.get(c,0) or 0) for c in features},df,idx))
    if not day_rows: continue
    syms_l,feat_r,dfs_l,idxs=zip(*day_rows)
    X=pd.DataFrame(list(feat_r))[features].replace([np.inf,-np.inf],np.nan).fillna(0).astype('float32')
    probas=model.predict_proba(X)[:,1]
    above=np.where(probas>=THRESHOLD)[0]; top=above[np.argsort(probas[above])[::-1]][:TOP_N]
    for ti in top:
        sym=syms_l[ti]; df=dfs_l[ti]; idx=idxs[ti]; prob=float(probas[ti])
        ecol='open' if 'open' in df.columns else 'close'
        epx=float(df[ecol].iloc[idx+1]); xpx=float(df['close'].iloc[min(idx+HOLD_DAYS+1,len(df)-1)])
        if epx<=0: continue
        raw=(xpx-epx)/epx; clamped=max(STOP_LOSS,min(PROFIT_TARGET,raw))
        trades.append({'date':str(entry_date)[:10],'symbol':sym,'probability':round(prob,4),
            'entry_price':round(epx,4),'exit_price':round(xpx,4),
            'raw_return_pct':round(raw*100,3),'clamped_return_pct':round(clamped*100,3),
            'hit_target':raw>=PROFIT_TARGET,'hit_stop':raw<=STOP_LOSS})
    if i%100==0: print(f"  [{str(entry_date)[:10]}] trades: {len(trades)}",flush=True)

Path('reports').mkdir(exist_ok=True)
df_t=pd.DataFrame(trades); df_t.to_csv('reports/backtest_trades.csv',index=False)
n=len(df_t); mr=df_t['clamped_return_pct'].mean(); sd=df_t['clamped_return_pct'].std()
wins=df_t.loc[df_t['clamped_return_pct']>0,'clamped_return_pct'].sum()
losses=df_t.loc[df_t['clamped_return_pct']<0,'clamped_return_pct'].abs().sum()
print(f"\n=== BACKTEST COMPLETE ===")
print(f"Total trades : {n:,}"); print(f"Win rate     : {df_t['clamped_return_pct'].gt(0).mean():.1%}")
print(f"Mean return  : {mr:.3f}%"); print(f"Profit factor: {wins/losses if losses else 999:.2f}")
print(f"Sharpe/trade : {mr/sd if sd>0 else 0:.3f}")
df_t['year']=pd.to_datetime(df_t['date']).dt.year
for _,r in df_t.groupby('year').agg(n=('clamped_return_pct','count'),wr=('clamped_return_pct',lambda x:(x>0).mean()),m=('clamped_return_pct','mean')).reset_index().iterrows():
    print(f"  {int(r['year'])}: {int(r['n']):>5} trades  WR={r['wr']:.1%}  mean={r['m']:.3f}%")
print(f"Saved → reports/backtest_trades.csv")
