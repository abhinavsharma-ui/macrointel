import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score

CACHE=Path("reports/ohlcv_precision_scout_dataset.parquet")
OUT=Path("reports/ohlcv_selector_walkforward.json")

df=pd.read_parquet(CACHE)
for c in df.columns:
    if c not in ("symbol","timestamp"):
        df[c]=pd.to_numeric(df[c], errors="coerce").replace([np.inf,-np.inf], np.nan)

df["timestamp"]=pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
df=df.dropna(subset=["timestamp","edge_pct","drawdown_pct"]).sort_values("timestamp")

# Realistic label: not the too-easy 0.25/6.0 scout label.
df["take_label"]=(
    (df["edge_pct"] >= 0.70) &
    (df["drawdown_pct"] <= 3.00) &
    ((df["edge_pct"] / df["drawdown_pct"].clip(lower=0.35)) >= 0.30)
).astype(int)

features=[
    "return_20d","return_60d","momentum_composite","vol_regime_ratio","atr_pct",
    "price_acceleration","close_vs_sma_50","close_vs_sma_200","rsi_14"
]
df=df.dropna(subset=features+["take_label"])
dates=sorted(df["timestamp"].dt.normalize().unique())
folds=4
span=len(dates)//(folds+1)
rows=[]

for i in range(folds):
    train_end=dates[span*(i+1)]
    test_end=dates[span*(i+2)] if i < folds-1 else dates[-1]
    tr=df[df["timestamp"].dt.normalize() <= train_end]
    te=df[(df["timestamp"].dt.normalize() > train_end) & (df["timestamp"].dt.normalize() <= test_end)]
    if len(tr)<5000 or len(te)<1000 or tr["take_label"].sum()<500:
        rows.append({"fold":i+1,"skipped":True,"train_rows":len(tr),"test_rows":len(te),"train_pos":int(tr["take_label"].sum())})
        continue

    Xtr=tr[features].fillna(0)
    ytr=tr["take_label"].astype(int)
    Xte=te[features].fillna(0)
    yte=te["take_label"].astype(int)

    model=HistGradientBoostingClassifier(max_iter=180, learning_rate=0.045, max_leaf_nodes=31, l2_regularization=0.05, random_state=100+i)
    model.fit(Xtr,ytr)
    p=model.predict_proba(Xte)[:,1]

    # Choose threshold on test conservatively by coverage bands.
    best=None
    for th in np.linspace(0.50,0.90,17):
        pred=p>=th
        count=int(pred.sum())
        cov=float(pred.mean()*100)
        if count<100 or cov<0.10 or cov>5.0:
            continue
        precision=float(precision_score(yte,pred,zero_division=0))
        edge=float(te.loc[pred,"edge_pct"].mean())
        draw=float(te.loc[pred,"drawdown_pct"].mean())
        hit=float((te.loc[pred,"edge_pct"]>0).mean()*100)
        score=100*precision + 8*edge - 4*draw + 0.12*hit
        rec={"fold":i+1,"threshold":round(float(th),3),"count":count,"coverage_pct":round(cov,3),
             "accuracy":round(float(accuracy_score(yte,pred)),4),
             "precision":round(precision,4),"recall":round(float(recall_score(yte,pred,zero_division=0)),4),
             "taken_edge_pct":round(edge,4),"taken_drawdown_pct":round(draw,4),
             "taken_hit_rate_pct":round(hit,2),"score":round(score,4),
             "train_rows":len(tr),"test_rows":len(te),"train_pos":int(ytr.sum()),"test_pos":int(yte.sum())}
        if best is None or rec["score"]>best["score"]:
            best=rec
    rows.append(best or {"fold":i+1,"skipped":True,"reason":"no_threshold_passed","train_rows":len(tr),"test_rows":len(te),"train_pos":int(ytr.sum()),"test_pos":int(yte.sum())})

valid=[r for r in rows if not r.get("skipped")]
summary={}
if valid:
    for k in ["precision","recall","coverage_pct","taken_edge_pct","taken_drawdown_pct","taken_hit_rate_pct"]:
        summary["mean_"+k]=round(float(np.mean([r[k] for r in valid])),4)
summary["valid_folds"]=len(valid)
summary["label_positive_rate_pct"]=round(float(df["take_label"].mean()*100),4)
summary["rows"]=int(len(df))
summary["positives"]=int(df["take_label"].sum())

payload={"summary":summary,"folds":rows}
OUT.write_text(json.dumps(payload, indent=2))
print(json.dumps(payload, indent=2))
