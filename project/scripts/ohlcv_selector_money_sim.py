import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score

ROOT = Path("data/features_26yr_liquid")
OUT = Path("reports/ohlcv_selector_money_sim.json")
FEATURES = ["return_20d","return_60d","momentum_composite","vol_regime_ratio","atr_pct","price_acceleration","close_vs_sma_50","close_vs_sma_200","rsi_14"]

INITIAL_CAPITAL = 100000.0
HORIZON = 10
MAX_NEW_PER_DAY = 5
POSITION_PCT = 0.01
ROUNDTRIP_COST_PCT = 0.35
STOP_LOSS_PCT = 2.5
TAKE_PROFIT_PCT = 3.0

def load_one(p):
    df = pd.read_parquet(p)
    if df.empty or "close" not in df.columns:
        return None
    df = df.copy().sort_index()
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce") if "timestamp" in df.columns else pd.to_datetime(df.index, utc=True, errors="coerce")
    df["timestamp"] = ts
    df = df.dropna(subset=["timestamp", "close"]).reset_index(drop=True)
    if len(df) < 300:
        return None

    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df.get("high", close), errors="coerce")
    low = pd.to_numeric(df.get("low", close), errors="coerce")

    fut_close = close.shift(-HORIZON)
    fwd_ret = (fut_close / close - 1.0) * 100.0

    # Path-aware approximation: stop first if future low breaches stop, take profit if future high reaches target.
    path_ret = []
    for i in range(len(df)):
        if i + HORIZON >= len(df) or not np.isfinite(close.iloc[i]) or close.iloc[i] <= 0:
            path_ret.append(np.nan)
            continue
        entry = float(close.iloc[i])
        ret = float(fwd_ret.iloc[i]) if np.isfinite(fwd_ret.iloc[i]) else np.nan
        for j in range(i + 1, i + 1 + HORIZON):
            lo = float(low.iloc[j]) if np.isfinite(low.iloc[j]) else entry
            hi = float(high.iloc[j]) if np.isfinite(high.iloc[j]) else entry
            if (lo / entry - 1.0) * 100.0 <= -STOP_LOSS_PCT:
                ret = -STOP_LOSS_PCT
                break
            if (hi / entry - 1.0) * 100.0 >= TAKE_PROFIT_PCT:
                ret = TAKE_PROFIT_PCT
                break
        path_ret.append(ret)

    out = pd.DataFrame({
        "symbol": p.stem,
        "timestamp": df["timestamp"],
        "future_return_pct": path_ret,
        "edge_pct": ((high.shift(-1).rolling(HORIZON, min_periods=2).max().shift(-(HORIZON-1)) / close) - 1.0) * 100.0,
        "drawdown_pct": (1.0 - (low.shift(-1).rolling(HORIZON, min_periods=2).min().shift(-(HORIZON-1)) / close)) * 100.0,
    })
    for f in FEATURES:
        if f in df.columns:
            out[f] = pd.to_numeric(df[f], errors="coerce")
        elif f == "return_20d" and "momentum_20d" in df.columns:
            out[f] = pd.to_numeric(df["momentum_20d"], errors="coerce")
        elif f == "return_60d" and "momentum_60d" in df.columns:
            out[f] = pd.to_numeric(df["momentum_60d"], errors="coerce")
        else:
            out[f] = 0.0
    return out.dropna(subset=["future_return_pct", "edge_pct", "drawdown_pct"] + FEATURES)

print("LOADING_FILES", len(list(ROOT.glob("*.parquet"))), flush=True)
parts=[]
for i,p in enumerate(sorted(ROOT.glob("*.parquet")),1):
    r=load_one(p)
    if r is not None and not r.empty:
        parts.append(r)
    if i%100==0:
        print("loaded", i, "ok", len(parts), flush=True)

df=pd.concat(parts, ignore_index=True)
df=df.replace([np.inf,-np.inf], np.nan).dropna(subset=FEATURES+["future_return_pct","edge_pct","drawdown_pct"])
df["timestamp"]=pd.to_datetime(df["timestamp"], utc=True)
df["date"]=df["timestamp"].dt.normalize()
df["take_label"]=((df["edge_pct"]>=0.90)&(df["drawdown_pct"]<=2.50)&((df["edge_pct"]/df["drawdown_pct"].clip(lower=0.35))>=0.30)).astype(int)

dates=sorted(df["date"].unique())
folds=4
span=len(dates)//(folds+1)
all_trades=[]
fold_rows=[]

for i in range(folds):
    train_end=dates[span*(i+1)]
    test_end=dates[span*(i+2)] if i<folds-1 else dates[-1]
    train_all=df[df["date"]<=train_end].copy()
    test=df[(df["date"]>train_end)&(df["date"]<=test_end)].copy()

    inner_dates=sorted(train_all["date"].unique())
    val_start=inner_dates[int(len(inner_dates)*0.80)]
    train=train_all[train_all["date"]<val_start].copy()
    val=train_all[train_all["date"]>=val_start].copy()

    model=HistGradientBoostingClassifier(max_iter=180, learning_rate=0.045, max_leaf_nodes=31, l2_regularization=0.05, random_state=900+i)
    model.fit(train[FEATURES].fillna(0), train["take_label"].astype(int))

    val_prob=model.predict_proba(val[FEATURES].fillna(0))[:,1]
    yv=val["take_label"].astype(int).to_numpy()
    best=None
    for th in np.linspace(0.50,0.90,21):
        pred=val_prob>=th
        count=int(pred.sum())
        cov=float(pred.mean()*100)
        if count<250 or cov<0.15 or cov>5.0:
            continue
        prec=float(precision_score(yv,pred,zero_division=0))
        if prec<0.55:
            continue
        avg_ret=float((val.loc[pred,"future_return_pct"]-ROUNDTRIP_COST_PCT).mean())
        score=100*prec + 10*avg_ret + min(cov,3)
        rec=(score,th,prec,cov,avg_ret,count)
        if best is None or rec>best:
            best=rec
    if best is None:
        fold_rows.append({"fold":i+1,"skipped":True,"reason":"no_threshold"})
        continue

    th=float(best[1])
    final=HistGradientBoostingClassifier(max_iter=180, learning_rate=0.045, max_leaf_nodes=31, l2_regularization=0.05, random_state=950+i)
    final.fit(train_all[FEATURES].fillna(0), train_all["take_label"].astype(int))
    test["prob"]=final.predict_proba(test[FEATURES].fillna(0))[:,1]
    selected=test[test["prob"]>=th].sort_values(["date","prob"], ascending=[True,False]).groupby("date").head(MAX_NEW_PER_DAY).copy()
    selected["net_return_pct"]=selected["future_return_pct"]-ROUNDTRIP_COST_PCT
    selected["fold"]=i+1
    all_trades.append(selected[["fold","date","symbol","prob","future_return_pct","net_return_pct"]])
    fold_rows.append({
        "fold":i+1,
        "threshold":round(th,4),
        "trades":int(len(selected)),
        "avg_net_return_pct":round(float(selected["net_return_pct"].mean()) if len(selected) else 0.0,4),
        "win_rate_pct":round(float((selected["net_return_pct"]>0).mean()*100) if len(selected) else 0.0,2),
        "test_start":str(test["date"].min()),
        "test_end":str(test["date"].max()),
    })
    print("FOLD", fold_rows[-1], flush=True)

trades=pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
equity=INITIAL_CAPITAL
peak=equity
max_dd=0.0
equity_curve=[]
for _,t in trades.sort_values("date").iterrows():
    pnl=equity*POSITION_PCT*(float(t["net_return_pct"])/100.0)
    equity+=pnl
    peak=max(peak,equity)
    max_dd=max(max_dd,(peak-equity)/peak*100.0)
    equity_curve.append({"date":str(t["date"]), "equity":round(equity,2), "symbol":t["symbol"], "net_return_pct":round(float(t["net_return_pct"]),4)})

summary={
    "initial_capital":INITIAL_CAPITAL,
    "final_equity":round(equity,2),
    "total_return_pct":round((equity/INITIAL_CAPITAL-1)*100,4),
    "max_drawdown_pct":round(max_dd,4),
    "trades":int(len(trades)),
    "avg_trade_net_return_pct":round(float(trades["net_return_pct"].mean()) if len(trades) else 0.0,4),
    "win_rate_pct":round(float((trades["net_return_pct"]>0).mean()*100) if len(trades) else 0.0,2),
    "position_pct":POSITION_PCT,
    "max_new_per_day":MAX_NEW_PER_DAY,
    "roundtrip_cost_pct":ROUNDTRIP_COST_PCT,
}
payload={"summary":summary,"folds":fold_rows,"last_50_trades":trades.tail(50).to_dict("records") if len(trades) else [],"equity_tail":equity_curve[-50:]}
OUT.write_text(json.dumps(payload,indent=2,default=str))
print("SIM_DONE")
print(json.dumps(summary,indent=2))
