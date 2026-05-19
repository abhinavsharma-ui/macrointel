import json, os, warnings

from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.ensemble import HistGradientBoostingClassifier

ROOT=Path(os.getenv("SIM_ROOT","data/features_26yr_liquid"))
OUT=Path(os.getenv("SIM_OUT","reports/fixed_return_h10_walkforward_sim.json"))
TRADES=Path(os.getenv("SIM_TRADES","reports/fixed_return_h10_walkforward_trades.csv"))
RUNTIME_STATE=Path(os.getenv("SIM_RUNTIME_STATE","data/runtime_state.json"))

HOLD=int(os.getenv("SIM_HOLD_DAYS","10"))
PROFIT_TARGET_PCT=float(os.getenv("SIM_PROFIT_TARGET_PCT","0") or 0)
COST=float(os.getenv("SIM_TOTAL_COST_PCT","0.45"))
MIN_ADV=float(os.getenv("SIM_MIN_ADV20_DOLLAR_VOL","5000000"))
MIN_PRICE=float(os.getenv("SIM_MIN_ENTRY_PRICE","5"))
MAX_ABS_RET=float(os.getenv("SIM_MAX_ABS_HONEST_RET_PCT","100"))
TH=float(os.getenv("SIM_THRESHOLD","0.55"))
STOP_LOSS_PCT = float(os.getenv("SIM_STOP_LOSS_PCT", "0"))
TOP_N=int(os.getenv("SIM_TOP_N","5"))
POS=float(os.getenv("SIM_POSITION_PCT","0.002"))
MAX_OPEN=int(os.getenv("SIM_MAX_OPEN_POSITIONS","50"))
ATR_MIN=float(os.getenv("SIM_ATR_MIN","0.0"))
SPYVOL_MIN=float(os.getenv("SIM_SPYVOL_MIN","0.0"))
MAX_TRAIN=int(os.getenv("SIM_MAX_TRAIN_ROWS","350000"))
FOLDS=int(os.getenv("SIM_FOLDS","4"))
INIT=float(os.getenv("SIM_INITIAL_CAPITAL","100000"))
USE_RUNTIME_UNIVERSE=os.getenv("SIM_USE_RUNTIME_UNIVERSE","1").lower() in {"1","true","yes","on"}
VOL_SCALING=os.getenv("SIM_VOL_SCALING","0").lower() in {"1","true","yes","on"}
KELLY_ENABLED=os.getenv("SIM_KELLY_ENABLED","0").strip().lower() in {"1","true","yes","on"}
KELLY_CAP=float(os.getenv("SIM_KELLY_CAP","2.0"))
KELLY_FLOOR=float(os.getenv("SIM_KELLY_FLOOR","0.5"))
KELLY_MEDIAN_PROB=float(os.getenv("SIM_KELLY_MEDIAN_PROB","0.70"))
# ── SPY Regime Filter ───────────────────────────────────────────────
REGIME_FILTER   = os.getenv("SIM_REGIME_FILTER","0").lower() in {"1","true","yes","on"}
REGIME_SMA_DAYS = int(os.getenv("SIM_REGIME_SPY_SMA_DAYS","20"))
REGIME_RET_DAYS = int(os.getenv("SIM_REGIME_SPY_RET_DAYS","10"))
REGIME_MIN_RET  = float(os.getenv("SIM_REGIME_SPY_MIN_RET","-0.03"))



def kelly_multiplier(prob):
    """Conviction-weighted sizing multiplier. Returns 1.0 when disabled."""
    if not KELLY_ENABLED:
        return 1.0
    try:
        p = float(prob)
    except Exception:
        return 1.0
    edge = max(0.001, p - 0.5)
    median_edge = max(0.001, KELLY_MEDIAN_PROB - 0.5)
    mult = (edge / median_edge) ** 0.5
    return max(KELLY_FLOOR, min(KELLY_CAP, mult))

EXPAND_RUNTIME_UNIVERSE=os.getenv("SIM_EXPAND_RUNTIME_UNIVERSE","0").lower() in {"1","true","yes","on"}
EXPAND_ADV_CUT=float(os.getenv("SIM_RUNTIME_UNIVERSE_ADV_CUT","0.30") or 0.30)
EXPANDED_CAP=int(os.getenv("SIM_EXPANDED_UNIVERSE_CAP","400") or 400)
EFFECTIVE_MIN_ADV=MIN_ADV*(1.0-EXPAND_ADV_CUT) if EXPAND_RUNTIME_UNIVERSE else MIN_ADV

def log(*x): print("SIM", *x, flush=True)

def norm_sym(p):
    return p.stem.replace("_US","").replace(".US","").upper()

def read_price_file(p, need_symbol=True):
    df=pd.read_parquet(p)
    if df.empty or not {"open","close","volume"}.issubset(df.columns):
        return None
    idx=pd.to_datetime(df.index, errors="coerce")
    df=df.loc[:,~df.columns.duplicated()].copy()
    df.index=pd.RangeIndex(len(df))
    if "date" in df.columns:
        dc=pd.to_datetime(df["date"], errors="coerce")
        df["date"]=dc if dc.notna().sum() >= idx.notna().sum() else idx
    else:
        df["date"]=idx
    df=df.dropna(subset=["date","open","close"]).sort_values("date").reset_index(drop=True)
    if need_symbol:
        df["symbol"]=norm_sym(p)
    if "adv20_dollar_vol" not in df.columns:
        df["adv20_dollar_vol"]=(df["close"]*df["volume"]).rolling(20,min_periods=5).mean()
    return df

def expanded_symbols_from_root():
    rows=[]
    for p in sorted(ROOT.glob("*.parquet")):
        sym=norm_sym(p)
        if sym.endswith((".NS",".BO",".NSE",".BSE")):
            continue
        try:
            x=read_price_file(p)
            if x is None or x.empty: continue
            r=x.iloc[-1]
            adv=float(r.get("adv20_dollar_vol",0) or 0)
            px=float(r.get("close",0) or 0)
            if adv>=EFFECTIVE_MIN_ADV and px>=MIN_PRICE:
                rows.append((adv,sym))
        except Exception:
            continue
    rows=sorted(rows, reverse=True)
    if EXPANDED_CAP>0:
        rows=rows[:EXPANDED_CAP]
    return {s for _,s in rows}

def runtime_symbols():
    base=set()
    if USE_RUNTIME_UNIVERSE and RUNTIME_STATE.exists():
        try:
            j=json.load(open(RUNTIME_STATE))
            base={str(k).upper() for k in (j.get("signal_store") or {}).keys()}
        except Exception:
            base=set()
    if EXPAND_RUNTIME_UNIVERSE:
        expanded=expanded_symbols_from_root()
        log("expanded_universe",len(expanded),"effective_min_adv",round(EFFECTIVE_MIN_ADV,2),"cap",EXPANDED_CAP)
        return (base | expanded) if base else expanded
    return base or None

ALLOWED=runtime_symbols()

def load_spy_vol_map():
    candidates=[ROOT/"SPY.parquet", ROOT/"SPY_US.parquet", ROOT/"SPY.US.parquet"]
    p=next((x for x in candidates if x.exists()), None)
    if p is None:
        return {}
    x=read_price_file(p, need_symbol=False)
    if x is None or x.empty:
        return {}
    ret=x["close"].pct_change()
    vol=ret.rolling(20,min_periods=10).std()*np.sqrt(252)*100.0
    d=pd.to_datetime(x["date"]).dt.normalize()
    return dict(zip(d, vol.fillna(20.0)))

SPY_VOL=load_spy_vol_map()

def load_spy_regime_map(df_all):
    if not REGIME_FILTER:
        return {}
    spy = df_all[df_all["symbol"]=="SPY"].copy()
    if spy.empty:
        spy = df_all[df_all["symbol"]=="SPY_US"].copy()
    if spy.empty:
        print("SIM REGIME: SPY not found, filter disabled"); return {}
    spy = spy.sort_values("date").reset_index(drop=True)
    spy["sma"]    = spy["close"].rolling(REGIME_SMA_DAYS, min_periods=max(5,REGIME_SMA_DAYS//2)).mean()
    spy["ret_nd"] = spy["close"].pct_change(REGIME_RET_DAYS)
    spy["ok"]     = (spy["close"] >= spy["sma"]) & (spy["ret_nd"] >= REGIME_MIN_RET)
    result = dict(zip(spy["date"], spy["ok"].fillna(False)))
    ok = sum(1 for v in result.values() if v)
    bad = len(result)-ok
    print(f"SIM REGIME sma={REGIME_SMA_DAYS}d ret={REGIME_RET_DAYS}d min={REGIME_MIN_RET} ok={ok} blocked={bad}")
    return result

def load_one(p):
    sym=norm_sym(p)
    if ALLOWED is not None and sym not in ALLOWED:
        return None
    x=read_price_file(p)
    if x is None:
        return None
    x=x[x["adv20_dollar_vol"].fillna(0)>=EFFECTIVE_MIN_ADV].copy()
    x=x[x["close"].fillna(0)>=MIN_PRICE].copy()
    if x.empty:
        return None

    x["entry_open"]=x["open"].shift(-1)
    x["default_exit_close"]=x["close"].shift(-(HOLD+1))
    x["model_ret"]=((x["default_exit_close"]/x["entry_open"])-1)*100

    x["effective_exit_close"]=x["default_exit_close"]
    x["exit_offset"]=HOLD+1
    x["pt_hit"]=False
    x["pt_day"]=np.nan
    x["exit_reason"]="hold"

    if PROFIT_TARGET_PCT>0:
        high_col="high" if "high" in x.columns else "close"
        hit_so_far=pd.Series(False,index=x.index)
        sl_hit_so_far = pd.Series(False, index=x.index)
        for k in range(1,HOLD+2):
            future_high=x[high_col].shift(-k)
            future_close=x["close"].shift(-k)
            hit=((future_high/x["entry_open"]-1)*100 >= PROFIT_TARGET_PCT).fillna(False)
            first=hit & ~hit_so_far
            if first.any():
                x.loc[first,"effective_exit_close"]=future_close.loc[first]
                x.loc[first,"exit_offset"]=k
                x.loc[first,"pt_hit"]=True
                x.loc[first,"pt_day"]=k
                x.loc[first,"exit_reason"]="profit_target"
            hit_so_far = hit_so_far | hit

            if STOP_LOSS_PCT > 0 and "low" in x.columns:
                future_low = x["low"].shift(-k)
                sl_hit = ((future_low / x["entry_open"]) - 1) * 100 <= -STOP_LOSS_PCT
                sl_hit = sl_hit.fillna(False)
                first_sl = sl_hit & ~hit_so_far & ~sl_hit_so_far
                if first_sl.any():
                    sl_price = x.loc[first_sl, "entry_open"] * (1 - STOP_LOSS_PCT / 100)
                    x.loc[first_sl, "effective_exit_close"] = sl_price
                    x.loc[first_sl, "exit_offset"] = k
                    x.loc[first_sl, "exit_reason"] = "stop_loss"
                sl_hit_so_far = sl_hit_so_far | sl_hit
                hit_so_far = hit_so_far | sl_hit_so_far
    x["honest_ret"]=((x["effective_exit_close"]/x["entry_open"])-1)*100
    x["spy_realized_vol_pct"]=pd.to_datetime(x["date"]).dt.normalize().map(SPY_VOL).fillna(20.0)
    x=x.replace([np.inf,-np.inf],np.nan).dropna(subset=["model_ret","honest_ret"])
    x=x[x["model_ret"].abs()<=MAX_ABS_RET]
    x=x[x["honest_ret"].abs()<=MAX_ABS_RET]
    return x

def feature_cols(df):
    banned={
        "date","symbol","year","open","high","low","close","volume","adj_close",
        "entry_open","default_exit_close","effective_exit_close","exit_close",
        "model_ret","honest_ret","adv20_dollar_vol","pt_hit","pt_day",
        "exit_reason","exit_offset","spy_realized_vol_pct"
    }
    out=[]
    for c in df.columns:
        cl=c.lower()
        if c in banned or cl.startswith("gross_ret_") or cl.startswith("exit_timestamp") or cl.startswith("future") or cl.startswith("next") or "target" in cl or "label" in cl:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    pref=["return_20d","return_60d","momentum_composite","vol_regime_ratio","atr_pct","price_acceleration","close_vs_sma_50","close_vs_sma_200","rsi_14"]
    return [c for c in pref if c in out]+[c for c in out if c not in pref]

def load_all():
    rows=[]
    files=sorted(ROOT.glob("*.parquet"))
    for i,p in enumerate(files,1):
        x=load_one(p)
        if x is not None and len(x):
            rows.append(x)
        if i%100==0:
            log("loaded",i,"files","rows",sum(len(r) for r in rows))
    df=pd.concat(rows,ignore_index=True)
    df["date"]=pd.to_datetime(df["date"])
    return df.sort_values(["date","symbol"]).reset_index(drop=True)

def vol_mult(v):
    if not VOL_SCALING:
        return 1.0
    try: v=float(v)
    except Exception: v=20.0
    if v < 15: return 0.75
    if v < 25: return 1.0
    if v < 40: return 1.25
    return 1.5

def simulate(pred, calendar_dates, regime_map=None):
    pred=pred[pred["prob"]>=TH].copy()
    if pred.empty:
        return {}, pd.DataFrame()
    pred["row_id"]=pred["symbol"].astype(str)+"|"+pred["date"].astype(str)
    by_date={d:g.sort_values("prob",ascending=False) for d,g in pred.groupby("date")}
    eq=INIT; peak=INIT; dd=0.0; active=[]; trades=[]; skipped=0
    for i,d in enumerate(calendar_dates):
        keep=[]; active_syms=set()
        for a in active:
            if a["exit_i"]<=i:
                eq+=a["pnl"]
                trades.append({**a, "exit_date":str(d), "status":"closed"})
            else:
                keep.append(a); active_syms.add(a["symbol"])
        active=keep
        peak=max(peak,eq); dd=max(dd,(peak-eq)/peak*100 if peak else 0)
        day=by_date.get(d)
        if day is None: continue
        new=0
        if regime_map and not regime_map.get(d, True): continue
        for _,r in day.head(TOP_N*3).iterrows():
            if new>=TOP_N: break
            sym=str(r.symbol)
            if sym in active_syms or len(active)>=MAX_OPEN: continue
            if ATR_MIN > 0 and float(r.get("atr_pct", 1.0)) < ATR_MIN: skipped+=1; continue
            if SPYVOL_MIN > 0 and float(r.get("spy_realized_vol_pct", 99.0)) < SPYVOL_MIN: skipped+=1; continue
            m=vol_mult(r.get("spy_realized_vol_pct",20.0))
            k=kelly_multiplier(r.prob)
            eff_pos=POS*m*k
            net=float(r.honest_ret)-COST
            _exit_r=str(r.exit_reason)
            notional=eq*eff_pos
            pnl=notional*net/100
            off=int(r.get("exit_offset",HOLD+1) or HOLD+1)
            active.append({
                "row_id":r.row_id, "symbol":sym, "signal_date":str(d),
                "prob":float(r.prob), "model_ret":float(r.model_ret),
                "honest_ret":float(r.honest_ret), "net_ret":net,
                "notional":notional, "pnl":pnl, "entry_price":float(r.entry_open),
                "exit_price":float(r.effective_exit_close), "exit_i":i+off,
                "exit_offset":off, "exit_reason":_exit_r,
                "pt_hit":bool(r.pt_hit), "pt_day":None if pd.isna(r.pt_day) else int(r.pt_day),
                "spy_realized_vol_pct":float(r.spy_realized_vol_pct),
                "vol_multiplier":float(m), "kelly_multiplier":float(k), "position_pct_effective":float(eff_pos),
            })
            active_syms.add(sym); new+=1
    final_date=str(calendar_dates[-1]) if len(calendar_dates) else ""
    for a in active:
        eq+=a["pnl"]
        trades.append({**a, "exit_date":final_date, "status":"closed_final"})
    t=pd.DataFrame(trades)
    arr=t["net_ret"].to_numpy() if len(t) else np.array([])
    pt_count=int((t["exit_reason"]=="profit_target").sum()) if len(t) else 0
    return {
        "trades":int(len(pred)),
        "executed_trades":int(len(t)),
        "skipped_capacity_or_duplicate":int(skipped),
        "profit_target_exits":pt_count,
        "profit_target_exit_pct":round(pt_count/max(len(t),1)*100,2),
        "final_equity":round(float(eq),2),
        "return_pct":round((eq/INIT-1)*100,4),
        "max_drawdown_pct":round(float(dd),4),
        "avg_trade_net_return_pct":round(float(arr.mean()) if len(arr) else 0,4),
        "win_rate_pct":round(float((arr>0).mean()*100) if len(arr) else 0,2),
        "avg_hold_days":round(float(t["exit_offset"].mean()) if len(t) else 0,2),
        "avg_effective_position_pct":round(float(t["position_pct_effective"].mean()) if len(t) else POS,5),
    }, t

df=load_all()
fs=feature_cols(df)
regime_map=load_spy_regime_map(df)
dates=pd.Index(sorted(df["date"].drop_duplicates()))
chunks=np.array_split(dates[int(len(dates)*0.35):], FOLDS)
log("dataset",len(df),"symbols",df.symbol.nunique(),"features",len(fs),"runtime_universe",ALLOWED is not None,"vol_scaling",VOL_SCALING,"pt",PROFIT_TARGET_PCT,"hold",HOLD)

all_preds=[]; folds=[]
for fold,ch in enumerate(chunks,1):
    test_start=pd.Timestamp(ch[0]); test_end=pd.Timestamp(ch[-1])
    start_i=dates.get_loc(test_start)
    train_end_i=max(0,start_i-HOLD-1)
    if train_end_i<504: continue
    train_dates=set(dates[:train_end_i])
    test_dates=pd.Index(ch)
    train=df[df.date.isin(train_dates)].copy()
    test=df[df.date.isin(set(test_dates))].copy()
    if len(train)>MAX_TRAIN:
        train=train.sample(MAX_TRAIN,random_state=42+fold)
    y=(train["model_ret"]>COST).astype(int)
    if y.nunique()<2: continue
    model=HistGradientBoostingClassifier(max_iter=180,learning_rate=0.045,max_leaf_nodes=31,l2_regularization=0.1,random_state=42+fold)
    model.fit(train[fs].replace([np.inf,-np.inf],np.nan).fillna(0).astype("float32"),y)
    test=test.copy()
    test["prob"]=model.predict_proba(test[fs].replace([np.inf,-np.inf],np.nan).fillna(0).astype("float32"))[:,1]
    res,tr=simulate(test, test_dates, regime_map=regime_map)
    log("fold_done",fold,str(test_start.date()),str(test_end.date()),json.dumps(res))
    keep=["date","symbol","model_ret","honest_ret","entry_open","effective_exit_close","prob","exit_offset","exit_reason","pt_hit","pt_day","spy_realized_vol_pct","atr_pct"]
    all_preds.append(test[keep])
    folds.append({"fold":fold,"test_start":str(test_start.date()),"test_end":str(test_end.date()),"train_rows":len(train),"test_rows":len(test),"positive_rate":round(float(y.mean()),4),"test":res})

pred=pd.concat(all_preds,ignore_index=True) if all_preds else pd.DataFrame()
calendar=pd.Index(sorted(pred["date"].drop_duplicates())) if len(pred) else pd.Index([])
summary,trades=simulate(pred,calendar,regime_map=regime_map)
odd,_=simulate(pred[pred.date.dt.year%2==1],pd.Index(sorted(pred[pred.date.dt.year%2==1]["date"].drop_duplicates())),regime_map=regime_map) if len(pred) else ({},pd.DataFrame())
if len(trades):
    remove_ids=set(trades.sort_values("net_ret",ascending=False).head(50)["row_id"])
    rb=pred[~(pred["symbol"].astype(str)+"|"+pred["date"].astype(str)).isin(remove_ids)]
    remove_best_50,_=simulate(rb,pd.Index(sorted(rb["date"].drop_duplicates())),regime_map=regime_map)
else:
    remove_best_50={}
out={
    "summary":summary,
    "folds":folds,
    "stress":{"odd_years_only":odd,"remove_best_50":remove_best_50},
    "settings":{
        "root":str(ROOT),"runtime_universe":ALLOWED is not None,"expanded_runtime_universe":EXPAND_RUNTIME_UNIVERSE,
        "symbols":int(df.symbol.nunique()),"threshold":TH,"top_n":TOP_N,"position_pct":POS,
        "hold_days":HOLD,"profit_target_pct":PROFIT_TARGET_PCT,"vol_scaling":VOL_SCALING,
        "cost_pct":COST,"max_open":MAX_OPEN,"min_adv":EFFECTIVE_MIN_ADV
    },
    "contract":"walk-forward only; train dates end before test by HOLD+1; target next open to HOLD+1 close; optional profit target uses future intraperiod high/close only inside test trade simulation; no gross_ret"
}
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps(out,indent=2,default=str))
if len(trades):
    TRADES.parent.mkdir(parents=True,exist_ok=True)
    trades.to_csv(TRADES,index=False)
print(json.dumps(out["summary"],indent=2))
print("wrote",OUT,TRADES)
