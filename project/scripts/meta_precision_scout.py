from pathlib import Path
import os, json
import numpy as np
import pandas as pd

os.environ.setdefault("XGB_RETRAIN_FEATURE_DIR", "data/features_26yr_liquid")
os.environ.setdefault("INSTITUTIONAL_RETRAIN_MARKET", "us")
os.environ.setdefault("META_MODEL_BUILD_WORKERS", "24")
os.environ.setdefault("META_PRICE_ONLY_RANKING", "1")
os.environ.setdefault("META_MODEL_LABEL_MIN_EDGE_PCT", "0.35")
os.environ.setdefault("META_MODEL_LABEL_MIN_EDGE_RATIO", "0.20")
os.environ.setdefault("META_MODEL_LABEL_MAX_DRAWDOWN_PCT", "3.50")
os.environ.setdefault("META_MODEL_ROUNDTRIP_COST_PCT", "0.25")
os.environ.setdefault("META_MODEL_CROSS_SECTIONAL_LABEL_PCT", "0.04")
os.environ.setdefault("META_MODEL_CROSS_SECTIONAL_MIN_CANDIDATES", "15")

from scripts.fast_meta_retrain_24 import load_features, make_trainer, fast_build_training_dataset
from models import institutional_retraining as ir

ir.MetaModelTrainer.build_training_dataset = fast_build_training_dataset
trainer = make_trainer()
matrices = load_features()
df = trainer.build_training_dataset(matrices)

Path("reports").mkdir(exist_ok=True)
base = {
    "rows": int(len(df)),
    "positive_rate": float(df["take_label"].mean()),
    "edge": float(df["edge_pct"].mean()),
    "drawdown": float(df["drawdown_pct"].mean()),
}
print("BASE", json.dumps(base, indent=2))

def col(name, default=0.0):
    return pd.to_numeric(df[name], errors="coerce").fillna(default) if name in df.columns else pd.Series(default, index=df.index)

rank = col("cross_sectional_score_rank_pct", 0.5)
mom = col("cross_sectional_momentum_rank_pct", 0.5)
conv = col("conviction_score", 0.0)
volr = col("vol_regime_ratio", 1.0)
atr = col("atr_pct", 0.02)
edge = col("edge_pct", 0.0)
draw = col("drawdown_pct", 0.0)
y = df["take_label"].astype(int)
is_long = df["direction_label"].astype(str).eq("long")

results = []
for side in ["all", "long"]:
    side_mask = pd.Series(True, index=df.index) if side == "all" else is_long
    for top in [0.01,0.02,0.03,0.04,0.05,0.07,0.10]:
        for min_conv in [0.0,0.5,0.75,1.0,1.25,1.5,2.0]:
            for min_mom in [0.45,0.55,0.65,0.75]:
                for max_volr in [1.05,1.25,1.50,2.00,99.0]:
                    for max_atr in [0.025,0.035,0.050,99.0]:
                        m = side_mask & (rank >= 1-top) & (conv >= min_conv) & (mom >= min_mom) & (volr <= max_volr) & (atr <= max_atr)
                        n = int(m.sum())
                        if n < 1000:
                            continue
                        prec = float(y[m].mean())
                        avg_edge = float(edge[m].mean())
                        avg_draw = float(draw[m].mean())
                        hit = float((edge[m] > 0).mean() * 100)
                        coverage = float(n / len(df) * 100)
                        edr = float(avg_edge / max(avg_draw, 0.35))
                        score = 100*prec + 12*avg_edge - 5*avg_draw + 0.20*hit + 1.2*min(coverage,4) + 10*edr
                        results.append({
                            "score": round(score,4), "side": side, "count": n, "coverage_pct": round(coverage,3),
                            "precision": round(prec,4), "edge": round(avg_edge,4), "drawdown": round(avg_draw,4),
                            "hit_rate": round(hit,2), "edge_draw_ratio": round(edr,4),
                            "top_pct": top, "min_conviction": min_conv, "min_momentum_rank": min_mom,
                            "max_vol_regime_ratio": max_volr, "max_atr_pct": max_atr,
                        })

results = sorted(results, key=lambda r: (r["precision"], r["edge_draw_ratio"], r["edge"], -r["drawdown"]), reverse=True)
Path("reports/meta_precision_scout_top.json").write_text(json.dumps({"base": base, "top": results[:50]}, indent=2))
top = results[0]
print("BEST", json.dumps(top, indent=2))
env = {
    "XGB_RETRAIN_FEATURE_DIR": "data/features_26yr_liquid",
    "INSTITUTIONAL_RETRAIN_MARKET": "us",
    "SKIP_XGB_RETRAIN": "1",
    "META_PRICE_ONLY_RANKING": "1",
    "META_MODEL_BUILD_WORKERS": "24",
    "META_MODEL_WALKFORWARD_FOLDS": "3",
    "META_MODEL_MIN_TRAIN_DAYS": "1800",
    "META_MODEL_TAKE_THRESHOLD": "0.78",
    "META_MODEL_MIN_PRECISION_FLOOR": str(max(0.45, min(0.60, top["precision"] - 0.02))),
    "META_MODEL_MIN_EDGE_DRAW_RATIO_FLOOR": str(max(0.35, min(0.80, top["edge_draw_ratio"] * 0.75))),
    "META_MODEL_MIN_COVERAGE_PCT": "0.20",
    "META_MODEL_MAX_COVERAGE_PCT": str(max(1.0, min(6.0, top["coverage_pct"] * 1.5))),
    "META_MODEL_LABEL_MIN_EDGE_PCT": "0.45",
    "META_MODEL_LABEL_MIN_EDGE_RATIO": "0.30",
    "META_MODEL_LABEL_MAX_DRAWDOWN_PCT": str(max(2.0, min(3.2, top["drawdown"] * 1.10))),
    "META_MODEL_ROUNDTRIP_COST_PCT": "0.25",
    "META_MODEL_CROSS_SECTIONAL_LABEL_PCT": str(top["top_pct"]),
    "META_MODEL_CROSS_SECTIONAL_MIN_CANDIDATES": "15",
}
Path("reports/meta_precision_scout_recommended.env").write_text("\n".join(f"{k}={v}" for k,v in env.items()) + "\n")
print("WROTE reports/meta_precision_scout_recommended.env")
