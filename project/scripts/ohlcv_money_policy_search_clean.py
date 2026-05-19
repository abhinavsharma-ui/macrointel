import json, math, os, re
from pathlib import Path
import numpy as np
import pandas as pd

CACHE = Path("reports/ohlcv_money_dataset.parquet")
OUT = Path("reports/ohlcv_money_policy_search_clean.json")
ENVOUT = Path("reports/ohlcv_money_policy_recommended.env")
LOG_PREFIX = "CLEAN_MONEY"

POSITION_PCT = float(os.getenv("MONEY_POSITION_PCT", "0.01"))
FOLDS = int(os.getenv("MONEY_FOLDS", "4"))
MIN_VAL_TRADES = int(os.getenv("MONEY_MIN_VAL_TRADES", "200"))
THRESHOLDS = [round(x, 2) for x in np.arange(0.52, 0.76, 0.02)]
MAX_NEW_PER_DAY = [1, 2, 3, 5]

def log(*x):
    print(LOG_PREFIX, *x, flush=True)

def finite(x):
    return np.isfinite(np.asarray(x, dtype=float))

def pick_cols(df):
    prob = next((c for c in ["probability", "selector_probability", "score"] if c in df.columns), None)
    if prob is None:
        raise SystemExit(f"No probability column found. columns={list(df.columns)[:80]}")
    date = next((c for c in ["date", "timestamp", "Date"] if c in df.columns), None)
    if date is None:
        raise SystemExit(f"No date column found. columns={list(df.columns)[:80]}")
    symbol = "symbol" if "symbol" in df.columns else None
    ret_cols = [c for c in df.columns if re.match(r"^ret_sl[0-9]", str(c))]
    if not ret_cols:
        raise SystemExit(f"No ret_sl*_tp* columns found. columns={list(df.columns)[:120]}")
    return date, symbol, prob, ret_cols

def selected_rows(df, prob_col, ret_col, threshold, max_new):
    d = df[[ "date", "symbol", prob_col, ret_col ]].copy()
    d = d[np.isfinite(d[prob_col].astype(float)) & np.isfinite(d[ret_col].astype(float))]
    d = d[d[prob_col] >= threshold]
    if d.empty:
        return d
    d = d.sort_values(["date", prob_col], ascending=[True, False])
    return d.groupby("date", sort=False).head(max_new).copy()

def equity_metrics(trades):
    if trades.empty:
        return None
    rets = trades["net_ret"].astype(float).to_numpy()
    if len(rets) == 0 or not np.isfinite(rets).all():
        return None
    equity = 100000.0
    peak = equity
    max_dd = 0.0
    for r in rets:
        equity *= 1.0 + POSITION_PCT * (r / 100.0)
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak * 100.0)
    return {
        "trades": int(len(rets)),
        "final_equity": round(equity, 2),
        "return_pct": round((equity / 100000.0 - 1.0) * 100.0, 4),
        "max_drawdown_pct": round(max_dd, 4),
        "avg_trade_net_return_pct": round(float(np.mean(rets)), 4),
        "win_rate_pct": round(float((rets > 0).mean() * 100.0), 2),
    }

def objective(m):
    if not m:
        return -1e9
    if m["trades"] < MIN_VAL_TRADES:
        return -1e9
    if m["return_pct"] <= 0 or m["avg_trade_net_return_pct"] <= 0 or m["win_rate_pct"] < 50:
        return -1e9
    return (
        m["return_pct"] * 4.0
        + m["avg_trade_net_return_pct"] * 40.0
        + m["win_rate_pct"] * 0.10
        - m["max_drawdown_pct"] * 2.5
        + min(m["trades"], 3000) / 3000.0
    )

log("loading", CACHE)
df = pd.read_parquet(CACHE)

if not any(c in df.columns for c in ["probability", "selector_probability", "score"]):
    import pickle
    try:
        import joblib
        artifact = joblib.load("models/checkpoints/ohlcv_selector.pkl")
    except Exception:
        artifact = pickle.loads(Path("models/checkpoints/ohlcv_selector.pkl").read_bytes())

    if isinstance(artifact, dict):
        model = artifact.get("model") or artifact.get("classifier") or artifact.get("estimator")
        features = artifact.get("features") or artifact.get("feature_cols") or artifact.get("feature_names")
    else:
        model = getattr(artifact, "model", None) or getattr(artifact, "classifier", None) or artifact
        features = getattr(artifact, "features", None) or getattr(artifact, "feature_cols", None) or getattr(artifact, "feature_names", None)

    if model is None:
        raise SystemExit("Selector artifact has no model")
    if not features:
        features = ["return_20d", "return_60d", "momentum_composite", "vol_regime_ratio", "atr_pct", "price_acceleration", "close_vs_sma_50", "close_vs_sma_200", "rsi_14"]

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise SystemExit(f"Money dataset missing selector features: {missing}")

    log("scoring_selector", "rows", len(df), "features", features)
    X = df[features].copy()
    for f in features:
        X[f] = pd.to_numeric(X[f], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    probs = np.zeros(len(X), dtype=float)
    chunk = 250000
    for start in range(0, len(X), chunk):
        end = min(start + chunk, len(X))
        part = X.iloc[start:end]
        if hasattr(model, "predict_proba"):
            raw = model.predict_proba(part)
            probs[start:end] = raw[:, 1] if getattr(raw, "ndim", 1) == 2 else raw
        else:
            raw = model.predict(part)
            probs[start:end] = np.asarray(raw, dtype=float)
        if end % 1000000 == 0 or end == len(X):
            log("score_progress", end, "/", len(X))

    df["selector_probability"] = probs
    log("scored_selector", "min", round(float(np.nanmin(probs)), 4), "max", round(float(np.nanmax(probs)), 4), "mean", round(float(np.nanmean(probs)), 4))

date_col, symbol_col, prob_col, ret_cols = pick_cols(df)

df = df.rename(columns={date_col: "date"})
if symbol_col is None:
    df["symbol"] = ""
else:
    df = df.rename(columns={symbol_col: "symbol"})

df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
df[prob_col] = pd.to_numeric(df[prob_col], errors="coerce")
df = df[df["date"].notna() & np.isfinite(df[prob_col].astype(float))].copy()
df = df.sort_values("date")

for c in ret_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

dates = np.array(sorted(df["date"].dropna().unique()))
test_span = max(80, len(dates) // (FOLDS + 2))
folds = []
all_test_trades = []

log("dataset", "rows", len(df), "dates", len(dates), "ret_cols", ret_cols)

for fold in range(1, FOLDS + 1):
    test_end_i = len(dates) - (FOLDS - fold) * test_span
    test_start_i = test_end_i - test_span
    train_end_i = test_start_i
    if train_end_i < test_span:
        folds.append({"fold": fold, "skipped": True, "reason": "too_little_history"})
        continue

    train_dates = dates[:train_end_i]
    val_start_i = max(0, int(len(train_dates) * 0.82))
    val_dates = train_dates[val_start_i:]
    test_dates = dates[test_start_i:test_end_i]

    val = df[df["date"].isin(val_dates)]
    test = df[df["date"].isin(test_dates)]

    best = None
    for ret_col in ret_cols:
        base_val = val[["date", "symbol", prob_col, ret_col]].dropna()
        if base_val.empty:
            continue
        for th in THRESHOLDS:
            for max_new in MAX_NEW_PER_DAY:
                sel = selected_rows(base_val, prob_col, ret_col, th, max_new)
                if sel.empty:
                    continue
                sel["net_ret"] = sel[ret_col].astype(float)
                vm = equity_metrics(sel)
                obj = objective(vm)
                if obj <= -1e8:
                    continue
                cand = {"ret_col": ret_col, "threshold": th, "max_new_per_day": max_new, "validation": vm, "objective": round(obj, 4)}
                if best is None or cand["objective"] > best["objective"]:
                    best = cand

    if best is None:
        folds.append({"fold": fold, "skipped": True, "reason": "no_profitable_validation_policy"})
        log("fold_skip", fold)
        continue

    test_sel = selected_rows(test, prob_col, best["ret_col"], best["threshold"], best["max_new_per_day"])
    test_sel["net_ret"] = test_sel[best["ret_col"]].astype(float)
    tm = equity_metrics(test_sel)
    rec = {"fold": fold, "policy": best, "test": tm}
    if tm is None:
        rec["skipped"] = True
        rec["reason"] = "no_clean_test_trades"
    else:
        all_test_trades.append(test_sel[["date", "symbol", "net_ret"]].assign(fold=fold))
    folds.append(rec)
    log("fold_done", json.dumps(rec, default=str))

if all_test_trades:
    all_trades = pd.concat(all_test_trades, ignore_index=True).sort_values("date")
    overall = equity_metrics(all_trades)
else:
    all_trades = pd.DataFrame()
    overall = None

valid = [f for f in folds if f.get("test") and not f.get("skipped")]
result = {
    "summary": {
        "valid_folds": len(valid),
        "folds": FOLDS,
        "position_pct": POSITION_PCT,
        **(overall or {})
    },
    "folds": folds,
}
OUT.write_text(json.dumps(result, indent=2, default=str) + "\n")
log("DONE", json.dumps(result["summary"], indent=2))

if overall and len(valid) >= 3 and overall["return_pct"] > 0 and overall["max_drawdown_pct"] < 8:
    best_policies = [f["policy"] for f in valid]
    thresholds = [p["threshold"] for p in best_policies]
    max_new = [p["max_new_per_day"] for p in best_policies]
    env = {
        "OHLCV_SELECTOR_MONITOR_ONLY": "1",
        "OHLCV_SELECTOR_THRESHOLD": str(round(float(np.median(thresholds)), 2)),
        "AUTO_TRADE_MAX_NEW_PER_CYCLE": str(int(np.median(max_new))),
        "NORMAL_LANE_MAX_NEW_PER_CYCLE": str(int(np.median(max_new))),
    }
    ENVOUT.write_text("\n".join(f"{k}={v}" for k, v in env.items()) + "\n")
    log("WROTE_ENV", ENVOUT)
else:
    log("NO_DEPLOYABLE_POLICY", "kept monitor-only")
