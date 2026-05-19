import json, os, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings("ignore")

from sklearn.ensemble import HistGradientBoostingClassifier
from core.paper_trading import VirtualBroker, Order, OrderSide

TRAIN_ROOT=Path(os.getenv("FIXED_TRAIN_ROOT","data/features_26yr_liquid"))
LIVE_ROOT=Path(os.getenv("FIXED_LIVE_ROOT","data/features"))
MODEL_PATH=Path(os.getenv("FIXED_MODEL_PATH","models/checkpoints/fixed_return_h10_model.joblib"))
OUT=Path(os.getenv("FIXED_LIVE_OUT","reports/fixed_return_live_paper.json"))

HOLD=int(os.getenv("FIXED_HOLD_DAYS","10"))
COST=float(os.getenv("FIXED_TOTAL_COST_PCT","0.45"))
MIN_ADV=float(os.getenv("FIXED_MIN_ADV20_DOLLAR_VOL","5000000"))
MIN_PRICE=float(os.getenv("FIXED_MIN_ENTRY_PRICE","5"))
MAX_ABS_RET=float(os.getenv("FIXED_MAX_ABS_HONEST_RET_PCT","100"))
MAX_TRAIN=int(os.getenv("FIXED_MAX_TRAIN_ROWS","350000"))
TH=float(os.getenv("FIXED_THRESHOLD","0.55"))
TOP_N=int(os.getenv("FIXED_TOP_N","5"))
POS=float(os.getenv("FIXED_POSITION_PCT","0.002"))
RETRAIN=os.getenv("FIXED_RETRAIN","0").lower() in {"1","true","yes","on"}


STATE_PATH=Path(os.getenv("FIXED_RUNTIME_STATE","data/runtime_state.json"))
US_ONLY=os.getenv("FIXED_US_ONLY","1").lower() in {"1","true","yes","on"}

def allowed_live_symbols():
    if not STATE_PATH.exists():
        return None
    try:
        j=json.load(open(STATE_PATH))
        keys={str(k).upper() for k in (j.get("signal_store") or {}).keys()}
        return keys or None
    except Exception:
        return None

ALLOWED_SYMBOLS=allowed_live_symbols()

def is_allowed_symbol(sym):
    u=str(sym or "").upper()
    if US_ONLY and (u.endswith(".NS") or u.endswith(".BO") or u.endswith(".NSE") or u.endswith(".BSE")):
        return False
    if ALLOWED_SYMBOLS is not None:
        return u in ALLOWED_SYMBOLS
    return True

def log(*x): print("FIXED_LIVE", *x, flush=True)

def load_one(p):
    df=pd.read_parquet(p)
    if df.empty or not {"open","close","volume"}.issubset(df.columns):
        return None
    df=df.loc[:,~df.columns.duplicated()].copy()
    idx=pd.to_datetime(df.index, errors="coerce")
    if idx.notna().sum() >= max(3, int(len(df)*0.8)):
        df["date"]=idx
    elif "date" in df.columns:
        df["date"]=pd.to_datetime(df["date"], errors="coerce")
    else:
        return None
    # Avoid pandas ambiguity when parquet index is also named "date".
    df.index = pd.RangeIndex(len(df))
    df=df.dropna(subset=["date","open","close"]).sort_values("date").reset_index(drop=True)
    df["symbol"]=p.stem.replace("_US","").replace(".US","")
    if "adv20_dollar_vol" not in df.columns:
        df["adv20_dollar_vol"]=(df["close"]*df["volume"]).rolling(20,min_periods=5).mean()
    return df

def feature_cols(df):
    banned={"date","symbol","year","open","high","low","close","volume","adj_close","entry_open","exit_close","honest_ret","adv20_dollar_vol"}
    out=[]
    for c in df.columns:
        cl=c.lower()
        if c in banned or cl.startswith("gross_ret_") or cl.startswith("exit_timestamp") or cl.startswith("future") or cl.startswith("next") or "target" in cl or "label" in cl:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    preferred=["return_20d","return_60d","momentum_composite","vol_regime_ratio","atr_pct","price_acceleration","close_vs_sma_50","close_vs_sma_200","rsi_14"]
    return [c for c in preferred if c in out]+[c for c in out if c not in preferred]

def build_train():
    rows=[]
    files=sorted(TRAIN_ROOT.glob("*.parquet"))
    for i,p in enumerate(files,1):
        x=load_one(p)
        if x is None:
            continue
        x=x[x["adv20_dollar_vol"].fillna(0)>=MIN_ADV].copy()
        x=x[x["close"].fillna(0)>=MIN_PRICE].copy()
        x["entry_open"]=x["open"].shift(-1)
        x["exit_close"]=x["close"].shift(-(HOLD+1))
        x["honest_ret"]=((x["exit_close"]/x["entry_open"])-1)*100
        x=x.replace([np.inf,-np.inf],np.nan).dropna(subset=["honest_ret"])
        x=x[x["honest_ret"].abs()<=MAX_ABS_RET]
        if len(x): rows.append(x)
        if i%100==0: log("loaded",i,"files")
    df=pd.concat(rows,ignore_index=True)
    fs=feature_cols(df)
    if len(df)>MAX_TRAIN:
        df=df.sample(MAX_TRAIN,random_state=42)
    y=(df["honest_ret"]>COST).astype(int)
    model=HistGradientBoostingClassifier(max_iter=180,learning_rate=0.045,max_leaf_nodes=31,l2_regularization=0.1,random_state=42)
    model.fit(df[fs].replace([np.inf,-np.inf],np.nan).fillna(0).astype("float32"), y)
    MODEL_PATH.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({"model":model,"features":fs,"train_rows":len(df),"positive_rate":float(y.mean())}, MODEL_PATH)
    log("trained",len(df),"features",len(fs),"positive_rate",round(float(y.mean()),4))
    return model,fs

def load_or_train():
    if MODEL_PATH.exists() and not RETRAIN:
        o=joblib.load(MODEL_PATH)
        log("loaded_model",MODEL_PATH,"features",len(o["features"]))
        return o["model"],o["features"]
    return build_train()

def latest_rows(fs):
    rows=[]
    for p in sorted(LIVE_ROOT.glob("*.parquet")):
        x=load_one(p)
        if x is None or x.empty: continue
        r=x.iloc[-1].copy()
        sym=str(r.get("symbol") or p.stem.replace("_US","").replace(".US",""))
        if not is_allowed_symbol(sym):
            continue
        price=float(r.get("close",0) or 0)
        adv=float(r.get("adv20_dollar_vol",0) or 0)
        if price<MIN_PRICE or adv<MIN_ADV: continue
        row={c:float(r.get(c,0) or 0) for c in fs}
        row.update({"symbol":sym,"price":price,"adv20_dollar_vol":adv})
        rows.append(row)
    return pd.DataFrame(rows)

model,fs=load_or_train()
live=latest_rows(fs)
if live.empty:
    raise SystemExit("no live rows passed filters")
live["prob"]=model.predict_proba(live[fs].replace([np.inf,-np.inf],np.nan).fillna(0).astype("float32"))[:,1]
live=live.sort_values("prob",ascending=False)
picks=live[live["prob"]>=TH].head(TOP_N).copy()
fallback=False
if picks.empty and os.getenv("FIXED_TOP_FALLBACK","1").lower() in {"1","true","yes","on"}:
    picks=live.head(TOP_N).copy()
    fallback=True

broker=VirtualBroker(initial_capital=float(os.getenv("PAPER_CAPITAL","100000")),max_position_pct=0.10,session_guardrails_enabled=False)
equity=float(getattr(broker,"portfolio_value",broker.cash))
orders=[]
for _,r in picks.iterrows():
    sym=str(r.symbol)
    if broker.has_open_position(symbol=sym): continue
    qty=max(1,int((equity*POS)/float(r.price)))
    order=Order(symbol=sym,side=OrderSide.BUY,quantity=qty,position_key=f"{sym}::fixed_return",signal_source="fixed_return_h10_model",metadata={
        "lane":"fixed_return","market":"US","take_probability":float(r.prob),"threshold":TH,
        "threshold_fallback":fallback,"strategy":"next_open_to_h10_close","cost_pct":COST,
        "tick_age_seconds":0,"signal_age_seconds":0,"spread_pct":0.05
    })
    res=broker.submit_order(order,current_price=float(r.price))
    orders.append({"symbol":sym,"price":float(r.price),"prob":round(float(r.prob),4),"qty":qty,"result":res})
getattr(broker,"_persist_state",lambda:None)()
out={"model":str(MODEL_PATH),"train_root":str(TRAIN_ROOT),"live_root":str(LIVE_ROOT),"live_candidates":len(live),"threshold":TH,"fallback_used":fallback,"orders":orders}
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps(out,indent=2,default=str))
print(json.dumps(out,indent=2,default=str))
