import json
from pathlib import Path
import pandas as pd
import numpy as np

CACHE = Path("reports/ohlcv_precision_scout_dataset.parquet")
TOP = Path("reports/ohlcv_precision_scout_top.json")
ENV = Path("reports/ohlcv_precision_scout_recommended.env")
FAIL = Path("reports/ohlcv_precision_scout_failed.json")

def fail(msg, extra=None):
    payload={"ok":False,"reason":msg,"extra":extra or {}}
    FAIL.write_text(json.dumps(payload, indent=2, default=str))
    print("FAILED_SAFE", json.dumps(payload, indent=2, default=str))
    raise SystemExit(2)

if not CACHE.exists():
    fail("CACHE_MISSING")

print("LOADING_CACHE", CACHE, CACHE.stat().st_size, flush=True)
df = pd.read_parquet(CACHE)
print("DATASET_READY", len(df), "symbols", df["symbol"].nunique(), flush=True)

need=["edge_pct","drawdown_pct","return_20d","return_60d","momentum_composite","vol_regime_ratio","atr_pct","price_acceleration","close_vs_sma_50","close_vs_sma_200","rsi_14"]
for c in need:
    if c not in df.columns:
        fail("MISSING_COLUMN", {"column": c})
    df[c]=pd.to_numeric(df[c], errors="coerce").replace([np.inf,-np.inf], np.nan)

df=df.dropna(subset=need)
print("CLEAN_ROWS", len(df), flush=True)

base = {
    "rows": int(len(df)),
    "symbols": int(df["symbol"].nunique()),
    "edge_mean": float(df["edge_pct"].mean()),
    "drawdown_mean": float(df["drawdown_pct"].mean()),
}
print("BASE", json.dumps(base, indent=2), flush=True)

# Precompute reusable boolean masks. This is much faster than recomputing giant masks blindly.
labels=[]
for min_edge in [0.25,0.35,0.45,0.55,0.70,0.90]:
    for max_dd in [2.0,2.5,3.0,3.5,4.5,6.0]:
        labels.append((min_edge,max_dd,(df["edge_pct"]>=min_edge)&(df["drawdown_pct"]<=max_dd)))

filters=[]
for r20 in [-0.02,0.0,0.02,0.04,0.06,0.08]:
    m20=df["return_20d"]>=r20
    for r60 in [-0.04,0.0,0.04,0.08,0.12,0.16]:
        m2060=m20 & (df["return_60d"]>=r60)
        for max_vol in [1.0,1.4,1.8,2.4,3.2]:
            mv=m2060 & (df["vol_regime_ratio"]<=max_vol)
            for max_atr in [0.015,0.025,0.04,0.06,0.09]:
                ma=mv & (df["atr_pct"]<=max_atr)
                for accel in [-999,-0.02,0.0,0.02,0.05]:
                    m=ma & (df["price_acceleration"]>=accel) & (df["close_vs_sma_50"]>-0.05) & (df["close_vs_sma_200"]>-0.10) & (df["rsi_14"].between(35,78))
                    n=int(m.sum())
                    if n>=750:
                        filters.append((r20,r60,max_vol,max_atr,accel,m,n))
print("FILTERS", len(filters), flush=True)

results=[]
for i,(r20,r60,max_vol,max_atr,accel,m,n) in enumerate(filters,1):
    edge_m=float(df.loc[m,"edge_pct"].mean())
    draw_m=float(df.loc[m,"drawdown_pct"].mean())
    hit=float((df.loc[m,"edge_pct"]>0).mean()*100)
    cov=float(n/len(df)*100)
    for min_edge,max_dd,y in labels:
        precision=float(y[m].mean())
        ratio=float(edge_m/max(draw_m,0.35))
        score=100*precision + 18*ratio + 7*edge_m - 3.5*draw_m + 0.12*hit + min(cov,5)
        results.append({
            "score": round(score,4), "count": n, "coverage_pct": round(cov,4),
            "precision": round(precision,4), "avg_edge_pct": round(edge_m,4),
            "avg_drawdown_pct": round(draw_m,4), "hit_rate_pct": round(hit,2),
            "edge_draw_ratio": round(ratio,4),
            "label_min_edge_pct": min_edge, "label_max_drawdown_pct": max_dd,
            "min_return_20d": r20, "min_return_60d": r60,
            "max_vol_regime_ratio": max_vol, "max_atr_pct": max_atr,
            "min_price_acceleration": accel,
        })
    if i % 100 == 0:
        print("GRID_PROGRESS", i, "/", len(filters), "results", len(results), flush=True)

if not results:
    fail("NO_GRID_RESULTS", base)

results.sort(key=lambda x:(x["precision"],x["edge_draw_ratio"],x["avg_edge_pct"],-x["avg_drawdown_pct"],x["count"]), reverse=True)
TOP.write_text(json.dumps({"base":base,"top":results[:100]}, indent=2))
best=results[0]
print("BEST", json.dumps(best, indent=2), flush=True)

env={
"XGB_RETRAIN_FEATURE_DIR":"data/features_26yr_liquid",
"INSTITUTIONAL_RETRAIN_MARKET":"us",
"SKIP_XGB_RETRAIN":"1",
"META_PRICE_ONLY_RANKING":"1",
"META_MODEL_BUILD_WORKERS":"24",
"META_MODEL_WALKFORWARD_FOLDS":"4",
"META_MODEL_MIN_TRAIN_DAYS":"1800",
"META_MODEL_TAKE_THRESHOLD":"0.78",
"META_MODEL_MIN_PRECISION_FLOOR":str(max(0.42,min(0.65,best["precision"]-0.02))),
"META_MODEL_MIN_EDGE_DRAW_RATIO_FLOOR":str(max(0.30,min(0.90,best["edge_draw_ratio"]*0.75))),
"META_MODEL_MIN_COVERAGE_PCT":"0.20",
"META_MODEL_MAX_COVERAGE_PCT":str(max(1.0,min(8.0,best["coverage_pct"]*1.8))),
"META_MODEL_LABEL_MIN_EDGE_PCT":str(best["label_min_edge_pct"]),
"META_MODEL_LABEL_MAX_DRAWDOWN_PCT":str(best["label_max_drawdown_pct"]),
"META_MODEL_LABEL_MIN_EDGE_RATIO":"0.25",
"META_MODEL_ROUNDTRIP_COST_PCT":"0.25",
"META_MODEL_CROSS_SECTIONAL_LABEL_PCT":"0.08",
"META_MODEL_CROSS_SECTIONAL_MIN_CANDIDATES":"8",
}
ENV.write_text("\n".join(f"{k}={v}" for k,v in env.items())+"\n")
print("WROTE_ENV", ENV, flush=True)
print("DONE_OK", flush=True)
