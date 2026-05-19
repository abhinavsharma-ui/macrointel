import json, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("SCOUT_FEATURE_DIR", "data/features_26yr_liquid"))
OUT = Path("reports")
OUT.mkdir(exist_ok=True)

HORIZON = int(os.environ.get("SCOUT_HORIZON_DAYS", "10"))
WORKERS = int(os.environ.get("SCOUT_WORKERS", "24"))
MAX_FILES = int(os.environ.get("SCOUT_MAX_FILES", "0") or 0)

MIN_EDGE_GRID = [0.35, 0.45, 0.55, 0.70, 0.90]
MAX_DD_GRID = [2.0, 2.5, 3.0, 3.5, 4.5]
RET20_GRID = [0.02, 0.04, 0.06, 0.08]
RET60_GRID = [0.04, 0.08, 0.12, 0.16]
VOL_GRID = [1.0, 1.4, 1.8, 2.4]
ATR_GRID = [0.015, 0.025, 0.04, 0.06]
PRICE_ACCEL_GRID = [-999, 0.0, 0.02, 0.05]

def num(s, default=0.0):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)

def load_one(p):
    try:
        df = pd.read_parquet(p)
        if df.empty or "close" not in df.columns:
            return None
        df = df.copy()
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        else:
            ts = pd.to_datetime(df.index, errors="coerce", utc=True)
        df["timestamp"] = ts
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        if len(df) < 300:
            return None

        close = num(df["close"])
        high = num(df.get("high", close))
        low = num(df.get("low", close))

        future_high = high.shift(-1).rolling(HORIZON, min_periods=2).max().shift(-(HORIZON-1))
        future_low = low.shift(-1).rolling(HORIZON, min_periods=2).min().shift(-(HORIZON-1))
        edge = (future_high / close - 1.0) * 100.0
        draw = (1.0 - future_low / close) * 100.0

        out = pd.DataFrame({
            "symbol": p.stem,
            "timestamp": df["timestamp"],
            "edge_pct": edge,
            "drawdown_pct": draw,
            "return_20d": num(df.get("return_20d", 0)),
            "return_60d": num(df.get("return_60d", 0)),
            "momentum_20d": num(df.get("momentum_20d", 0)),
            "momentum_60d": num(df.get("momentum_60d", 0)),
            "momentum_composite": num(df.get("momentum_composite", 0)),
            "vol_regime_ratio": num(df.get("vol_regime_ratio", 1), 1),
            "atr_pct": num(df.get("atr_pct", 0)),
            "price_acceleration": num(df.get("price_acceleration", 0)),
            "close_vs_sma_50": num(df.get("close_vs_sma_50", 0)),
            "close_vs_sma_200": num(df.get("close_vs_sma_200", 0)),
            "rsi_14": num(df.get("rsi_14", 50), 50),
        })
        out = out.dropna(subset=["edge_pct", "drawdown_pct"])
        return out
    except Exception as e:
        return None

files = sorted(ROOT.glob("*.parquet"))
if MAX_FILES:
    files = files[:MAX_FILES]
print(f"SCOUT root={ROOT} files={len(files)} workers={WORKERS} horizon={HORIZON}", flush=True)

parts = []
done = 0
with ProcessPoolExecutor(max_workers=WORKERS) as ex:
    futs = [ex.submit(load_one, p) for p in files]
    for fut in as_completed(futs):
        done += 1
        r = fut.result()
        if r is not None and not r.empty:
            parts.append(r)
        if done % 50 == 0 or done == len(futs):
            rows = sum(len(x) for x in parts)
            print(f"load {done}/{len(futs)} ok={len(parts)} rows={rows}", flush=True)

df = pd.concat(parts, ignore_index=True)
print("DATASET", len(df), "symbols", df["symbol"].nunique(), "dates", df["timestamp"].dt.date.nunique(), flush=True)

# Cross-sectional ranks per day. These are the important quality levers.
df["date"] = df["timestamp"].dt.date
for col, asc in [
    ("return_20d", False),
    ("return_60d", False),
    ("momentum_composite", False),
    ("close_vs_sma_50", False),
    ("close_vs_sma_200", False),
    ("vol_regime_ratio", True),
    ("atr_pct", True),
]:
    df[col + "_rank"] = df.groupby("date")[col].rank(pct=True, ascending=asc)

results = []
base_rows = len(df)
for min_edge in MIN_EDGE_GRID:
    for max_dd in MAX_DD_GRID:
        y = (df["edge_pct"] >= min_edge) & (df["drawdown_pct"] <= max_dd)
        for r20 in RET20_GRID:
            for r60 in RET60_GRID:
                for max_vol in VOL_GRID:
                    for max_atr in ATR_GRID:
                        for accel in PRICE_ACCEL_GRID:
                            m = (
                                (df["return_20d"] >= r20) &
                                (df["return_60d"] >= r60) &
                                (df["vol_regime_ratio"] <= max_vol) &
                                (df["atr_pct"] <= max_atr) &
                                (df["price_acceleration"] >= accel) &
                                (df["close_vs_sma_50"] > 0) &
                                (df["close_vs_sma_200"] > -0.03) &
                                (df["rsi_14"].between(42, 72))
                            )
                            n = int(m.sum())
                            if n < 500:
                                continue
                            precision = float(y[m].mean())
                            edge = float(df.loc[m, "edge_pct"].mean())
                            draw = float(df.loc[m, "drawdown_pct"].mean())
                            hit = float((df.loc[m, "edge_pct"] > 0).mean() * 100)
                            coverage = float(n / base_rows * 100)
                            ratio = edge / max(draw, 0.35)
                            score = 100*precision + 16*ratio + 8*edge - 4*draw + 0.15*hit
                            results.append({
                                "score": round(score, 4),
                                "count": n,
                                "coverage_pct": round(coverage, 4),
                                "precision": round(precision, 4),
                                "avg_edge_pct": round(edge, 4),
                                "avg_drawdown_pct": round(draw, 4),
                                "hit_rate_pct": round(hit, 2),
                                "edge_draw_ratio": round(ratio, 4),
                                "label_min_edge_pct": min_edge,
                                "label_max_drawdown_pct": max_dd,
                                "min_return_20d": r20,
                                "min_return_60d": r60,
                                "max_vol_regime_ratio": max_vol,
                                "max_atr_pct": max_atr,
                                "min_price_acceleration": accel,
                            })

results.sort(key=lambda x: (x["precision"], x["edge_draw_ratio"], x["avg_edge_pct"], -x["avg_drawdown_pct"], x["count"]), reverse=True)
Path("reports/ohlcv_precision_scout_top.json").write_text(json.dumps({"rows": len(df), "top": results[:100]}, indent=2))
print("TOP")
print(json.dumps(results[:20], indent=2))

if results:
    best = results[0]
    env = {
        "XGB_RETRAIN_FEATURE_DIR": "data/features_26yr_liquid",
        "INSTITUTIONAL_RETRAIN_MARKET": "us",
        "SKIP_XGB_RETRAIN": "1",
        "META_PRICE_ONLY_RANKING": "1",
        "META_MODEL_BUILD_WORKERS": "24",
        "META_MODEL_WALKFORWARD_FOLDS": "4",
        "META_MODEL_MIN_TRAIN_DAYS": "1800",
        "META_MODEL_TAKE_THRESHOLD": "0.78",
        "META_MODEL_MIN_PRECISION_FLOOR": str(max(0.45, min(0.65, best["precision"] - 0.02))),
        "META_MODEL_MIN_EDGE_DRAW_RATIO_FLOOR": str(max(0.35, min(0.90, best["edge_draw_ratio"] * 0.75))),
        "META_MODEL_MIN_COVERAGE_PCT": "0.20",
        "META_MODEL_MAX_COVERAGE_PCT": str(max(1.0, min(8.0, best["coverage_pct"] * 1.8))),
        "META_MODEL_LABEL_MIN_EDGE_PCT": str(best["label_min_edge_pct"]),
        "META_MODEL_LABEL_MAX_DRAWDOWN_PCT": str(best["label_max_drawdown_pct"]),
        "META_MODEL_LABEL_MIN_EDGE_RATIO": "0.30",
        "META_MODEL_ROUNDTRIP_COST_PCT": "0.25",
        "META_MODEL_CROSS_SECTIONAL_LABEL_PCT": "0.08",
        "META_MODEL_CROSS_SECTIONAL_MIN_CANDIDATES": "8",
    }
    Path("reports/ohlcv_precision_scout_recommended.env").write_text("\n".join(f"{k}={v}" for k,v in env.items()) + "\n")
    print("WROTE reports/ohlcv_precision_scout_recommended.env")
else:
    print("NO_RESULTS")
    raise SystemExit(2)
