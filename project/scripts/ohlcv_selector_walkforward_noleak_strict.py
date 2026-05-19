import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score

CACHE = Path("reports/ohlcv_precision_scout_dataset.parquet")
OUT = Path("reports/ohlcv_selector_walkforward_noleak.json")

FEATURES = [
    "return_20d","return_60d","momentum_composite","vol_regime_ratio","atr_pct",
    "price_acceleration","close_vs_sma_50","close_vs_sma_200","rsi_14"
]

def score_block(df, pred, y):
    count = int(pred.sum())
    coverage = float(pred.mean() * 100.0)
    precision = float(precision_score(y, pred, zero_division=0))
    recall = float(recall_score(y, pred, zero_division=0))
    accuracy = float(accuracy_score(y, pred))
    edge = float(df.loc[pred, "edge_pct"].mean()) if count else 0.0
    draw = float(df.loc[pred, "drawdown_pct"].mean()) if count else 0.0
    hit = float((df.loc[pred, "edge_pct"] > 0).mean() * 100.0) if count else 0.0
    return {
        "count": count,
        "coverage_pct": round(coverage, 4),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "taken_edge_pct": round(edge, 4),
        "taken_drawdown_pct": round(draw, 4),
        "taken_hit_rate_pct": round(hit, 2),
    }

df = pd.read_parquet(CACHE)
for c in df.columns:
    if c not in ("symbol", "timestamp"):
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
df = df.dropna(subset=["timestamp", "edge_pct", "drawdown_pct"] + FEATURES).sort_values("timestamp")

df["take_label"] = (
    (df["edge_pct"] >= 0.90) &
    (df["drawdown_pct"] <= 2.50) &
    ((df["edge_pct"] / df["drawdown_pct"].clip(lower=0.35)) >= 0.30)
).astype(int)

dates = sorted(df["timestamp"].dt.normalize().unique())
folds = 4
span = len(dates) // (folds + 1)
rows = []

print("NOLEAK_DATASET", len(df), "symbols", df["symbol"].nunique(), "positives", int(df["take_label"].sum()), "positive_rate_pct", round(float(df["take_label"].mean()*100), 4), flush=True)

for i in range(folds):
    train_end = dates[span * (i + 1)]
    test_end = dates[span * (i + 2)] if i < folds - 1 else dates[-1]

    train_all = df[df["timestamp"].dt.normalize() <= train_end].copy()
    test = df[(df["timestamp"].dt.normalize() > train_end) & (df["timestamp"].dt.normalize() <= test_end)].copy()

    inner_dates = sorted(train_all["timestamp"].dt.normalize().unique())
    if len(inner_dates) < 200:
        rows.append({"fold": i+1, "skipped": True, "reason": "not_enough_inner_dates"})
        continue

    val_start = inner_dates[int(len(inner_dates) * 0.80)]
    train_inner = train_all[train_all["timestamp"].dt.normalize() < val_start].copy()
    val = train_all[train_all["timestamp"].dt.normalize() >= val_start].copy()

    if len(train_inner) < 5000 or len(val) < 1000 or len(test) < 1000 or train_inner["take_label"].sum() < 500:
        rows.append({
            "fold": i+1, "skipped": True, "reason": "insufficient_rows_or_positives",
            "train_inner_rows": len(train_inner), "val_rows": len(val), "test_rows": len(test),
            "train_inner_pos": int(train_inner["take_label"].sum()), "val_pos": int(val["take_label"].sum()), "test_pos": int(test["take_label"].sum())
        })
        continue

    model_inner = HistGradientBoostingClassifier(
        max_iter=180, learning_rate=0.045, max_leaf_nodes=31,
        l2_regularization=0.05, random_state=200+i
    )
    model_inner.fit(train_inner[FEATURES].fillna(0), train_inner["take_label"].astype(int))

    val_prob = model_inner.predict_proba(val[FEATURES].fillna(0))[:, 1]
    y_val = val["take_label"].astype(int).to_numpy()

    best = None
    for th in np.linspace(0.50, 0.92, 22):
        pred = val_prob >= th
        m = score_block(val, pred, y_val)
        if m["count"] < 250 or m["coverage_pct"] < 0.15 or m["coverage_pct"] > 5.0:
            continue
        if m["precision"] < 0.52:
            continue
        objective = (
            100 * m["precision"]
            + 8 * m["taken_edge_pct"]
            - 4 * m["taken_drawdown_pct"]
            + 0.12 * m["taken_hit_rate_pct"]
            + min(m["coverage_pct"], 3.0)
        )
        candidate = {"threshold": round(float(th), 4), "objective": round(float(objective), 4), **m}
        if best is None or candidate["objective"] > best["objective"]:
            best = candidate

    if best is None:
        rows.append({
            "fold": i+1, "skipped": True, "reason": "no_validation_threshold_passed",
            "train_inner_rows": len(train_inner), "val_rows": len(val), "test_rows": len(test),
            "train_inner_pos": int(train_inner["take_label"].sum()), "val_pos": int(val["take_label"].sum()), "test_pos": int(test["take_label"].sum())
        })
        continue

    # Refit on all past data only. Threshold remains chosen only from past validation, never from test.
    model_final = HistGradientBoostingClassifier(
        max_iter=180, learning_rate=0.045, max_leaf_nodes=31,
        l2_regularization=0.05, random_state=300+i
    )
    model_final.fit(train_all[FEATURES].fillna(0), train_all["take_label"].astype(int))

    test_prob = model_final.predict_proba(test[FEATURES].fillna(0))[:, 1]
    y_test = test["take_label"].astype(int).to_numpy()
    test_pred = test_prob >= best["threshold"]
    test_metrics = score_block(test, test_pred, y_test)

    rec = {
        "fold": i+1,
        "threshold_chosen_on_validation": best["threshold"],
        "validation": best,
        "test": test_metrics,
        "train_rows": len(train_all),
        "train_pos": int(train_all["take_label"].sum()),
        "val_rows": len(val),
        "val_pos": int(val["take_label"].sum()),
        "test_rows": len(test),
        "test_pos": int(test["take_label"].sum()),
    }
    rows.append(rec)
    print("FOLD_DONE", json.dumps(rec), flush=True)

valid = [r for r in rows if not r.get("skipped")]
summary = {
    "valid_folds": len(valid),
    "rows": int(len(df)),
    "positives": int(df["take_label"].sum()),
    "label_positive_rate_pct": round(float(df["take_label"].mean() * 100.0), 4),
}
if valid:
    for k in ["precision","recall","coverage_pct","taken_edge_pct","taken_drawdown_pct","taken_hit_rate_pct"]:
        summary["mean_" + k] = round(float(np.mean([r["test"][k] for r in valid])), 4)

payload = {"summary": summary, "folds": rows}
OUT.write_text(json.dumps(payload, indent=2))
print("NOLEAK_DONE")
print(json.dumps(payload, indent=2))
