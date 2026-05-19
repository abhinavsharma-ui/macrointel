import json, os, traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("SCOUT_FEATURE_DIR", "data/features_26yr_liquid"))
OUT = Path("reports")
OUT.mkdir(exist_ok=True)
CACHE = OUT / "ohlcv_precision_scout_dataset.parquet"
TOP_JSON = OUT / "ohlcv_precision_scout_top.json"
ENV_OUT = OUT / "ohlcv_precision_scout_recommended.env"
FAIL_JSON = OUT / "ohlcv_precision_scout_failed.json"

HORIZON = int(os.environ.get("SCOUT_HORIZON_DAYS", "10"))
WORKERS = int(os.environ.get("SCOUT_WORKERS", "16"))
MIN_ROWS_PER_SYMBOL = int(os.environ.get("SCOUT_MIN_ROWS_PER_SYMBOL", "120"))

def write_fail(reason, extra=None):
    payload = {"ok": False, "reason": reason, "extra": extra or {}}
    FAIL_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print("FAILED_SAFE", json.dumps(payload, indent=2, default=str), flush=True)

def as_num(x, default=0.0):
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)
    return pd.Series(default)

def parse_ts(df):
    for c in ["timestamp", "Date", "date", "datetime"]:
        if c in df.columns:
            ts = pd.to_datetime(df[c], errors="coerce", utc=True)
            if ts.notna().sum() > 0:
                return ts
    ts = pd.to_datetime(df.index, errors="coerce", utc=True)
    return pd.Series(ts, index=df.index)

def future_window_max(s, h):
    vals = s.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    for i in range(len(vals) - h):
        win = vals[i+1:i+1+h]
        if len(win):
            out[i] = np.nanmax(win)
    return pd.Series(out, index=s.index)

def future_window_min(s, h):
    vals = s.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    for i in range(len(vals) - h):
        win = vals[i+1:i+1+h]
        if len(win):
            out[i] = np.nanmin(win)
    return pd.Series(out, index=s.index)

def load_one(path_str):
    p = Path(path_str)
    try:
        df = pd.read_parquet(p)
        if df is None or df.empty:
            return ("ERR", p.name, "empty")
        lower = {str(c).lower(): c for c in df.columns}
        if "close" not in lower:
            return ("ERR", p.name, f"missing_close columns={list(df.columns)[:20]}")
        close_col = lower["close"]
        high_col = lower.get("high", close_col)
        low_col = lower.get("low", close_col)

        ts = parse_ts(df)
        close = as_num(df[close_col])
        high = as_num(df[high_col])
        low = as_num(df[low_col])

        work = pd.DataFrame({
            "timestamp": ts.to_numpy(),
            "close": close.to_numpy(),
            "high": high.to_numpy(),
            "low": low.to_numpy(),
            "return_20d": as_num(df[lower["return_20d"]]).to_numpy() if "return_20d" in lower else 0.0,
            "return_60d": as_num(df[lower["return_60d"]]).to_numpy() if "return_60d" in lower else 0.0,
            "momentum_composite": as_num(df[lower["momentum_composite"]]).to_numpy() if "momentum_composite" in lower else 0.0,
            "vol_regime_ratio": as_num(df[lower["vol_regime_ratio"]], 1.0).to_numpy() if "vol_regime_ratio" in lower else 1.0,
            "atr_pct": as_num(df[lower["atr_pct"]]).to_numpy() if "atr_pct" in lower else 0.0,
            "price_acceleration": as_num(df[lower["price_acceleration"]]).to_numpy() if "price_acceleration" in lower else 0.0,
            "close_vs_sma_50": as_num(df[lower["close_vs_sma_50"]]).to_numpy() if "close_vs_sma_50" in lower else 0.0,
            "close_vs_sma_200": as_num(df[lower["close_vs_sma_200"]]).to_numpy() if "close_vs_sma_200" in lower else 0.0,
            "rsi_14": as_num(df[lower["rsi_14"]], 50.0).to_numpy() if "rsi_14" in lower else 50.0,
        })
        work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce", utc=True)
        work = work.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)
        work = work[work["close"] > 0]
        if len(work) < MIN_ROWS_PER_SYMBOL:
            return ("ERR", p.name, f"too_short:{len(work)}")

        fh = future_window_max(work["high"], HORIZON)
        fl = future_window_min(work["low"], HORIZON)
        work["symbol"] = p.stem
        work["edge_pct"] = (fh / work["close"] - 1.0) * 100.0
        work["drawdown_pct"] = (1.0 - fl / work["close"]) * 100.0
        work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["edge_pct", "drawdown_pct"])
        if work.empty:
            return ("ERR", p.name, "empty_after_labels")
        return ("OK", p.name, work[[
            "symbol","timestamp","edge_pct","drawdown_pct","return_20d","return_60d",
            "momentum_composite","vol_regime_ratio","atr_pct","price_acceleration",
            "close_vs_sma_50","close_vs_sma_200","rsi_14"
        ]])
    except Exception as e:
        return ("ERR", p.name, repr(e))

def build_or_load():
    if CACHE.exists() and CACHE.stat().st_size > 50_000_000:
        print(f"LOADING_CACHE {CACHE} size={CACHE.stat().st_size}", flush=True)
        df = pd.read_parquet(CACHE)
        print(f"DATASET_READY rows={len(df)} symbols={df['symbol'].nunique()}", flush=True)
        return df

    files = [str(p) for p in sorted(ROOT.glob("*.parquet"))]
    print(f"BUILDING_DATASET root={ROOT} files={len(files)} workers={WORKERS} horizon={HORIZON}", flush=True)
    if not files:
        write_fail("NO_PARQUET_FILES", {"root": str(ROOT)})
        raise SystemExit(2)

    parts, errors = [], []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(load_one, f) for f in files]
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                status, name, payload = fut.result()
            except Exception as e:
                status, name, payload = "ERR", "future", repr(e)
            if status == "OK":
                parts.append(payload)
            elif len(errors) < 50:
                errors.append((name, payload))
            if done % 50 == 0 or done == len(futs):
                rows = sum(len(x) for x in parts)
                print(f"load {done}/{len(futs)} ok={len(parts)} rows={rows} errors={done-len(parts)}", flush=True)

    if not parts:
        write_fail("NO_USABLE_FRAMES", {"errors": errors[:30]})
        raise SystemExit(2)

    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(CACHE, index=False)
    print(f"SAVED_CACHE {CACHE} rows={len(df)} symbols={df['symbol'].nunique()} size={CACHE.stat().st_size}", flush=True)
    return df

def run_grid(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"], utc=True).dt.date
    base = {
        "rows": int(len(df)),
        "symbols": int(df["symbol"].nunique()),
        "dates": int(df["date"].nunique()),
        "edge_mean": float(df["edge_pct"].mean()),
        "drawdown_mean": float(df["drawdown_pct"].mean()),
    }
    print("BASE", json.dumps(base, indent=2), flush=True)

    results = []
    min_edge_grid = [0.25, 0.35, 0.45, 0.55, 0.70, 0.90]
    max_dd_grid = [2.0, 2.5, 3.0, 3.5, 4.5, 6.0]
    r20_grid = [-0.02, 0.0, 0.02, 0.04, 0.06]
    r60_grid = [-0.04, 0.0, 0.04, 0.08, 0.12]
    vol_grid = [1.0, 1.4, 1.8, 2.4, 3.2]
    atr_grid = [0.015, 0.025, 0.04, 0.06, 0.09]
    accel_grid = [-999, -0.02, 0.0, 0.02, 0.05]

    nbase = len(df)
    for min_edge in min_edge_grid:
        for max_dd in max_dd_grid:
            y = (df["edge_pct"] >= min_edge) & (df["drawdown_pct"] <= max_dd)
            for r20 in r20_grid:
                for r60 in r60_grid:
                    for max_vol in vol_grid:
                        for max_atr in atr_grid:
                            for accel in accel_grid:
                                m = (
                                    (df["return_20d"] >= r20) &
                                    (df["return_60d"] >= r60) &
                                    (df["vol_regime_ratio"] <= max_vol) &
                                    (df["atr_pct"] <= max_atr) &
                                    (df["price_acceleration"] >= accel) &
                                    (df["close_vs_sma_50"] > -0.05) &
                                    (df["close_vs_sma_200"] > -0.10) &
                                    (df["rsi_14"].between(35, 78))
                                )
                                n = int(m.sum())
                                if n < 750:
                                    continue
                                precision = float(y[m].mean())
                                edge = float(df.loc[m, "edge_pct"].mean())
                                draw = float(df.loc[m, "drawdown_pct"].mean())
                                hit = float((df.loc[m, "edge_pct"] > 0).mean() * 100)
                                cov = float(n / nbase * 100)
                                ratio = float(edge / max(draw, 0.35))
                                score = 100*precision + 18*ratio + 7*edge - 3.5*draw + 0.12*hit + min(cov, 5)
                                results.append({
                                    "score": round(score,4), "count": n, "coverage_pct": round(cov,4),
                                    "precision": round(precision,4), "avg_edge_pct": round(edge,4),
                                    "avg_drawdown_pct": round(draw,4), "hit_rate_pct": round(hit,2),
                                    "edge_draw_ratio": round(ratio,4),
                                    "label_min_edge_pct": min_edge, "label_max_drawdown_pct": max_dd,
                                    "min_return_20d": r20, "min_return_60d": r60,
                                    "max_vol_regime_ratio": max_vol, "max_atr_pct": max_atr,
                                    "min_price_acceleration": accel,
                                })

    results.sort(key=lambda x: (x["precision"], x["edge_draw_ratio"], x["avg_edge_pct"], -x["avg_drawdown_pct"], x["count"]), reverse=True)
    TOP_JSON.write_text(json.dumps({"base": base, "top": results[:100]}, indent=2))
    print(f"GRID_RESULTS count={len(results)} wrote={TOP_JSON}", flush=True)

    if not results:
        write_fail("NO_GRID_RESULTS", {"base": base})
        raise SystemExit(2)

    best = results[0]
    print("BEST", json.dumps(best, indent=2), flush=True)

    env = {
        "XGB_RETRAIN_FEATURE_DIR": "data/features_26yr_liquid",
        "INSTITUTIONAL_RETRAIN_MARKET": "us",
        "SKIP_XGB_RETRAIN": "1",
        "META_PRICE_ONLY_RANKING": "1",
        "META_MODEL_BUILD_WORKERS": "24",
        "META_MODEL_WALKFORWARD_FOLDS": "4",
        "META_MODEL_MIN_TRAIN_DAYS": "1800",
        "META_MODEL_TAKE_THRESHOLD": "0.78",
        "META_MODEL_MIN_PRECISION_FLOOR": str(max(0.42, min(0.65, best["precision"] - 0.02))),
        "META_MODEL_MIN_EDGE_DRAW_RATIO_FLOOR": str(max(0.30, min(0.90, best["edge_draw_ratio"] * 0.75))),
        "META_MODEL_MIN_COVERAGE_PCT": "0.20",
        "META_MODEL_MAX_COVERAGE_PCT": str(max(1.0, min(8.0, best["coverage_pct"] * 1.8))),
        "META_MODEL_LABEL_MIN_EDGE_PCT": str(best["label_min_edge_pct"]),
        "META_MODEL_LABEL_MAX_DRAWDOWN_PCT": str(best["label_max_drawdown_pct"]),
        "META_MODEL_LABEL_MIN_EDGE_RATIO": "0.25",
        "META_MODEL_ROUNDTRIP_COST_PCT": "0.25",
        "META_MODEL_CROSS_SECTIONAL_LABEL_PCT": "0.08",
        "META_MODEL_CROSS_SECTIONAL_MIN_CANDIDATES": "8",
    }
    ENV_OUT.write_text("\n".join(f"{k}={v}" for k,v in env.items()) + "\n")
    print(f"WROTE_ENV {ENV_OUT}", flush=True)
    print("DONE_OK", flush=True)

try:
    df = build_or_load()
    run_grid(df)
except SystemExit:
    raise
except Exception as e:
    write_fail("UNHANDLED_TOP_LEVEL", {"error": repr(e), "traceback": traceback.format_exc()})
    raise SystemExit(2)
