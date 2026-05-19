import json, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT=Path(os.getenv("FIXED_FEATURE_ROOT","data/features_26yr_liquid"))
OUT=Path(os.getenv("FIXED_OPT_OUT","reports/ohlcv_fixed_return_optimizer_filtered.json"))
SEED=42
INIT=100000.0
HOLD=int(os.getenv("FIXED_HOLD_DAYS","10"))
COST=float(os.getenv("FIXED_TOTAL_COST_PCT","0.45"))
MIN_ADV=float(os.getenv("FIXED_MIN_ADV20_DOLLAR_VOL","5000000"))
MIN_ENTRY_PRICE=float(os.getenv("FIXED_MIN_ENTRY_PRICE","5"))
MAX_ABS_RET=float(os.getenv("FIXED_MAX_ABS_HONEST_RET_PCT","100"))
MAX_DAILY_JUMP=float(os.getenv("FIXED_MAX_DAILY_JUMP_PCT","60"))
MAX_ENTRY_GAP=float(os.getenv("FIXED_MAX_ENTRY_GAP_PCT","40"))
MAX_GROSS=float(os.getenv("FIXED_MAX_GROSS_EXPOSURE_PCT","1.0"))
POS_GRID=[float(x) for x in os.getenv("FIXED_POSITION_PCTS","0.005,0.01,0.015,0.02").split(",")]
MAX_NEW_GRID=[int(x) for x in os.getenv("FIXED_MAX_NEW_GRID","1,2,3,5").split(",")]
PROB_GRID=[float(x) for x in os.getenv("FIXED_PROB_GRID","0.55,0.58,0.60,0.62,0.65,0.68,0.70").split(",")]
FOLDS=4

def log(*x): print("FIXED_FILTERED",*x,flush=True)

def parse_date(raw):
    for c in ["date","timestamp","datetime","time","Date"]:
        if c in raw.columns:
            s=pd.Series(pd.to_datetime(raw[c],errors="coerce",utc=True))
            if s.notna().sum()>0 and s.dt.year.median()>1980:
                return s.dt.tz_localize(None).dt.normalize()
    s=pd.Series(pd.to_datetime(raw.index,errors="coerce",utc=True))
    if s.notna().sum()>0 and s.dt.year.median()>1980:
        return s.dt.tz_localize(None).dt.normalize()
    raise ValueError("no usable date/index")

def feats(df):
    banned={"date","symbol","year","open","high","low","close","volume","entry_open","exit_close","exit_date","honest_ret","adv20_dollar_vol","entry_gap_pct","max_path_jump_pct"}
    bad=("gross_ret","exit_timestamp","hold_days","future","next","target","label","edge_pct","drawdown_pct")
    out=[]
    for c in df.columns:
        low=str(c).lower()
        if c in banned or any(x in low for x in bad):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    preferred=["return_20d","return_60d","momentum_composite","vol_regime_ratio","atr_pct","price_acceleration","close_vs_sma_50","close_vs_sma_200","rsi_14"]
    return [c for c in preferred if c in out]+[c for c in out if c not in preferred]

def load():
    files=sorted(ROOT.glob("*.parquet"))
    if not files: raise SystemExit(f"no parquet files under {ROOT}")
    parts=[]
    for i,p in enumerate(files,1):
        try:
            raw=pd.read_parquet(p)
            if not {"open","close","volume"}.issubset(raw.columns): continue
            x=raw.copy()
            x["date"]=parse_date(raw).to_numpy()
            x["symbol"]=p.stem
            for c in ["open","high","low","close","volume"]:
                if c in x.columns: x[c]=pd.to_numeric(x[c],errors="coerce")
            x=x.dropna(subset=["date","open","close","volume"]).sort_values("date").reset_index(drop=True)
            x=x[(x["open"]>0)&(x["close"]>0)&(x["volume"]>0)]
            x["adv20_dollar_vol"]=(x["close"]*x["volume"]).rolling(20,min_periods=5).mean()
            parts.append(x)
        except Exception as e:
            log("load_error",p.name,repr(e))
        if i%100==0 or i==len(files): log("loaded",i,"ok",len(parts))
    df=pd.concat(parts,ignore_index=True).sort_values(["symbol","date"]).reset_index(drop=True)
    before=len(df)
    df=df[df["adv20_dollar_vol"].fillna(0)>=MIN_ADV].copy()
    log("liquidity",before,"after",len(df),"symbols",df.symbol.nunique())

    g=df.groupby("symbol",sort=False)
    df["entry_open"]=g["open"].shift(-1)
    df["exit_close"]=g["close"].shift(-(HOLD+1))
    df["exit_date"]=g["date"].shift(-(HOLD+1))
    df["honest_ret"]=((df["exit_close"]/df["entry_open"])-1.0)*100.0
    df["entry_gap_pct"]=((df["entry_open"]/df["close"])-1.0).abs()*100.0

    daily_jump=g["close"].pct_change().abs()*100.0
    df["max_path_jump_pct"]=daily_jump.groupby(df["symbol"],sort=False).transform(lambda s: s.shift(-1).rolling(HOLD+1,min_periods=1).max().shift(-HOLD))

    before=len(df)
    df=df.replace([np.inf,-np.inf],np.nan).dropna(subset=["entry_open","exit_close","exit_date","honest_ret","entry_gap_pct","max_path_jump_pct"])
    df=df[(df["entry_open"]>=MIN_ENTRY_PRICE)&(df["exit_close"]>=MIN_ENTRY_PRICE)]
    df=df[df["honest_ret"].abs()<=MAX_ABS_RET]
    df=df[df["entry_gap_pct"]<=MAX_ENTRY_GAP]
    df=df[df["max_path_jump_pct"]<=MAX_DAILY_JUMP]
    log("sanity_filter",before,"after",len(df),"dropped",before-len(df))
    log("honest_ret_quantiles",json.dumps({str(k):float(v) for k,v in df["honest_ret"].quantile([0,.001,.01,.05,.5,.95,.99,.999,1]).round(4).to_dict().items()}))
    return df.sort_values(["date","symbol"]).reset_index(drop=True)

def splits(df):
    dates=np.array(sorted(df.date.drop_duplicates()))
    chunks=np.array_split(dates[int(len(dates)*0.35):],FOLDS)
    out=[]
    for i,ch in enumerate(chunks,1):
        ts=np.searchsorted(dates,ch[0]); te=np.searchsorted(dates,ch[-1],side="right")
        ve=max(0,ts-HOLD-1); vs=max(0,ve-min(756,max(252,len(ch)//2)))
        tr=max(0,vs-HOLD-1)
        if tr<504: continue
        out.append((i,set(dates[:tr]),set(dates[vs:ve]),set(dates[ts:te]),str(pd.Timestamp(ch[0]).date()),str(pd.Timestamp(ch[-1]).date())))
    return out

def simulate(t,pos,max_new):
    if t.empty:
        return dict(trades=0,executed_trades=0,skipped_capacity=0,final_equity=INIT,return_pct=0,max_drawdown_pct=0,avg_trade_net_return_pct=0,win_rate_pct=0)
    t=t.sort_values(["date","prob"],ascending=[True,False]).groupby("date",group_keys=False).head(max_new).copy()
    active=[]; eq=INIT; peak=INIT; dd=0; rets=[]; exe=0; skip=0
    for d,day in t.groupby("date",sort=True):
        keep=[]
        for exit_d,pnl,notional in active:
            if exit_d<=d: eq+=pnl
            else: keep.append((exit_d,pnl,notional))
        active=keep; peak=max(peak,eq); dd=max(dd,(peak-eq)/peak*100 if peak else 0)
        exposure=sum(x[2] for x in active)/max(eq,1)
        for _,r in day.iterrows():
            if exposure+pos>MAX_GROSS+1e-12: skip+=1; continue
            net=float(r.honest_ret)-COST
            notional=eq*pos; pnl=notional*net/100
            active.append((r.exit_date,pnl,notional)); exposure+=pos
            rets.append(net); exe+=1
    for _,pnl,_ in active: eq+=pnl
    arr=np.array(rets)
    return dict(trades=int(len(t)),executed_trades=int(exe),skipped_capacity=int(skip),final_equity=round(eq,2),return_pct=round((eq/INIT-1)*100,4),max_drawdown_pct=round(dd,4),avg_trade_net_return_pct=round(float(arr.mean()) if len(arr) else 0,4),win_rate_pct=round(float((arr>0).mean()*100) if len(arr) else 0,2))

def choose(val):
    best=None
    for th in PROB_GRID:
        base=val[val.prob>=th].copy()
        if len(base)<30: continue
        for mx in MAX_NEW_GRID:
            for pos in POS_GRID:
                r=simulate(base,pos,mx)
                obj=r["return_pct"]-2*max(0,r["max_drawdown_pct"]-8)
                if r["executed_trades"]<80: obj-=30
                cand=(obj,th,mx,pos,r)
                if best is None or cand[0]>best[0]: best=cand
    return best

def main():
    df=load(); fs=feats(df); log("rows",len(df),"symbols",df.symbol.nunique(),"features",len(fs))
    folds=[]; trades=[]
    for fold,trd,vd,ted,ts,te in splits(df):
        train=df[df.date.isin(trd)].copy(); val=df[df.date.isin(vd)].copy(); test=df[df.date.isin(ted)].copy()
        log("fold_start",fold,ts,te,len(train),len(val),len(test))
        y=(train.honest_ret>COST).astype(int)
        if y.nunique()<2: continue
        if len(train)>200000:
            train=train.sample(200000,random_state=SEED+fold); y=(train.honest_ret>COST).astype(int)
        model=HistGradientBoostingClassifier(max_iter=120,learning_rate=0.05,max_leaf_nodes=31,l2_regularization=0.1,random_state=SEED+fold)
        model.fit(train[fs].replace([np.inf,-np.inf],np.nan).fillna(0).astype("float32"),y)
        val=val.copy(); test=test.copy()
        val["prob"]=model.predict_proba(val[fs].replace([np.inf,-np.inf],np.nan).fillna(0).astype("float32"))[:,1]
        test["prob"]=model.predict_proba(test[fs].replace([np.inf,-np.inf],np.nan).fillna(0).astype("float32"))[:,1]
        best=choose(val)
        if not best: log("fold_no_policy",fold); continue
        obj,th,mx,pos,vr=best
        tt=test[test.prob>=th].copy()
        tr=simulate(tt,pos,mx)
        log("fold_done",json.dumps({"fold":fold,"policy":{"threshold":th,"max_new":mx,"position_pct":pos},"validation":vr,"test":tr})[:2000])
        tt["position_pct"]=pos; tt["max_new"]=mx
        trades.append(tt); folds.append({"fold":fold,"test_start":ts,"test_end":te,"policy":{"threshold":th,"max_new":mx,"position_pct":pos},"validation":vr,"test":tr})
    alltr=pd.concat(trades,ignore_index=True) if trades else pd.DataFrame()
    if len(alltr):
        pos=float(alltr.position_pct.mode().iloc[0]); mx=int(alltr.max_new.mode().iloc[0])
        summary=simulate(alltr,pos,mx)
        odd=simulate(alltr[alltr.date.dt.year%2==1],pos,mx)
        rb=alltr.assign(net=alltr.honest_ret-COST).sort_values("net",ascending=False).iloc[50:]
        remove_best_50=simulate(rb,pos,mx)
    else:
        summary=odd=remove_best_50={}
    out={"summary":summary,"folds":folds,"stress":{"odd_years_only":odd,"remove_best_50":remove_best_50},"contract":"raw 26yr OHLCV; no gross_ret; next open to HOLD+1 close; min price, max return, max gap, max daily jump filters; validation policy only"}
    OUT.write_text(json.dumps(out,indent=2,default=str))
    print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
