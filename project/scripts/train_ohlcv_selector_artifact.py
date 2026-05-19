import json, os, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score

CACHE = Path("reports/ohlcv_precision_scout_dataset.parquet")
ARTIFACT = Path("models/checkpoints/ohlcv_selector.pkl")
REPORT = Path("reports/ohlcv_selector_artifact_report.json")
FEATURES = ["return_20d","return_60d","momentum_composite","vol_regime_ratio","atr_pct","price_acceleration","close_vs_sma_50","close_vs_sma_200","rsi_14"]

EDGE = float(os.getenv("OHLCV_SELECTOR_LABEL_EDGE", "0.90"))
DD = float(os.getenv("OHLCV_SELECTOR_LABEL_MAX_DRAWDOWN", "2.50"))
RATIO = float(os.getenv("OHLCV_SELECTOR_LABEL_MIN_RATIO", "0.30"))
MIN_PREC = float(os.getenv("OHLCV_SELECTOR_MIN_VALIDATION_PRECISION", "0.58"))

print("LOADING", CACHE, flush=True)
df = pd.read_parquet(CACHE)
for c in df.columns:
    if c not in ("symbol", "timestamp"):
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
df = df.dropna(subset=["timestamp","edge_pct","drawdown_pct"] + FEATURES).sort_values("timestamp")
df["take_label"] = ((df["edge_pct"] >= EDGE) & (df["drawdown_pct"] <= DD) & ((df["edge_pct"] / df["drawdown_pct"].clip(lower=0.35)) >= RATIO)).astype(int)

dates = sorted(df["timestamp"].dt.normalize().unique())
val_start = dates[int(len(dates) * 0.85)]
train = df[df["timestamp"].dt.normalize() < val_start]
val = df[df["timestamp"].dt.normalize() >= val_start]
print("DATASET", len(df), "positives", int(df["take_label"].sum()), "train", len(train), "val", len(val), flush=True)

model = HistGradientBoostingClassifier(max_iter=220, learning_rate=0.045, max_leaf_nodes=31, l2_regularization=0.05, random_state=777)
model.fit(train[FEATURES].fillna(0), train["take_label"].astype(int))
prob = model.predict_proba(val[FEATURES].fillna(0))[:, 1]
y = val["take_label"].astype(int).to_numpy()

best = None
for th in np.linspace(0.50, 0.92, 22):
    pred = prob >= th
    count = int(pred.sum())
    cov = float(pred.mean() * 100)
    if count < 500 or cov < 0.20 or cov > 5.0:
        continue
    prec = float(precision_score(y, pred, zero_division=0))
    if prec < MIN_PREC:
        continue
    edge = float(val.loc[pred, "edge_pct"].mean())
    draw = float(val.loc[pred, "drawdown_pct"].mean())
    hit = float((val.loc[pred, "edge_pct"] > 0).mean() * 100)
    rec = {"threshold": round(float(th), 4), "count": count, "coverage_pct": round(cov,4), "precision": round(prec,4), "recall": round(float(recall_score(y,pred,zero_division=0)),4), "accuracy": round(float(accuracy_score(y,pred)),4), "taken_edge_pct": round(edge,4), "taken_drawdown_pct": round(draw,4), "taken_hit_rate_pct": round(hit,2)}
    rec["objective"] = round(100*prec + 8*edge - 4*draw + 0.12*hit + min(cov,3.0), 4)
    if best is None or rec["objective"] > best["objective"]:
        best = rec

if best is None:
    raise SystemExit("NO_VALID_THRESHOLD")

threshold = max(0.62, float(best["threshold"]))
print("BEST_VALIDATION", json.dumps(best, indent=2), flush=True)
print("FINAL_THRESHOLD", threshold, flush=True)

final_model = HistGradientBoostingClassifier(max_iter=260, learning_rate=0.045, max_leaf_nodes=31, l2_regularization=0.05, random_state=778)
final_model.fit(df[FEATURES].fillna(0), df["take_label"].astype(int))

payload = {"model": final_model, "features": FEATURES, "threshold": threshold, "created_at": pd.Timestamp.utcnow().isoformat(), "label": {"edge_pct": EDGE, "max_drawdown_pct": DD, "min_edge_draw_ratio": RATIO}, "validation": best, "source_cache": str(CACHE)}
ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
with ARTIFACT.open("wb") as f:
    pickle.dump(payload, f)
REPORT.write_text(json.dumps({k:v for k,v in payload.items() if k != "model"}, indent=2, default=str))
print("ARTIFACT_DONE", ARTIFACT, REPORT, flush=True)
