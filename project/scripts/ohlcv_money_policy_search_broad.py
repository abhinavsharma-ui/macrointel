import json, math, os, pickle, re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

RAW_CACHE = Path("reports/ohlcv_money_dataset.parquet")
SCORED_CACHE = Path("reports/ohlcv_money_dataset_scored.parquet")
OUT = Path("reports/ohlcv_money_policy_search_broad.json")
ENVOUT = Path("reports/ohlcv_money_policy_broad_recommended.env")
LOG_PREFIX = "BROAD_MONEY"

POSITION_PCT = float(os.getenv("MONEY_POSITION_PCT", "0.01"))
FOLDS = int(os.getenv("MONEY_FOLDS", "4"))
MIN_VAL_TRADES = int(os.getenv("MONEY_MIN_VAL_TRADES", "200"))
MIN_VALID_FOLDS = int(os.getenv("MONEY_MIN_VALID_FOLDS", "3"))
MAX_DD_LIMIT = float(os.getenv("MONEY_MAX_DD_PCT", "8.0"))
DEFAULT_FEATURES = ["return_20d","return_60d","momentum_composite","vol_regime_ratio","atr_pct","price_acceleration","close_vs_sma_50","close_vs_sma_200","rsi_14"]

def log(*x): print(LOG_PREFIX, *x, flush=True)
def parse_float_list(name, default): return [float(x.strip()) for x in os.getenv(name, default).split(",") if x.strip()]
def parse_int_list(name, default): return [int(x.strip()) for x in os.getenv(name, default).split(",") if x.strip()]

THRESHOLDS = sorted(set(round(x, 3) for x in parse_float_list("MONEY_THRESHOLDS", ",".join(str(round(x,2)) for x in np.arange(0.46,0.741,0.01)))))
MAX_NEW_PER_DAY = parse_int_list("MONEY_MAX_NEW_PER_DAY", "1,2,3,5,8,10,12,15,20,30,40,60,80")

def load_selector_artifact(path):
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with path.open("rb") as f:
            return pickle.load(f)

def score_selector(df):
    if "selector_probability" in df.columns:
        return df
    artifact = load_selector_artifact(Path("models/checkpoints/ohlcv_selector.pkl"))
    if isinstance(artifact, dict):
        model = artifact.get("model") or artifact.get("classifier") or artifact.get("estimator")
        features = artifact.get("features") or artifact.get("feature_cols") or artifact.get("feature_names")
    else:
        model = getattr(artifact, "model", None) or getattr(artifact, "classifier", None) or artifact
        features = getattr(artifact, "features", None) or getattr(artifact, "feature_cols", None) or getattr(artifact, "feature_names", None)
    features = list(features or DEFAULT_FEATURES)
    missing = [c for c in features if c not in df.columns]
    if model is None or missing:
        raise SystemExit(f"bad selector artifact/model missing={missing}")
    log("scoring_selector", "rows", len(df), "features", features)
    X = df[features].copy()
    for c in features:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    probs = np.zeros(len(X), dtype=float)
    for start in range(0, len(X), 250000):
        end = min(start + 250000, len(X))
        raw = np.asarray(model.predict_proba(X.iloc[start:end]))
        probs[start:end] = raw[:, 1] if raw.ndim == 2 else raw.astype(float)
        if end % 1000000 == 0 or end == len(X):
            log("score_progress", end, "/", len(X))
    df = df.copy()
    df["selector_probability"] = probs
    return df

def normalize_frame(df) -> Tuple[pd.DataFrame, List[str]]:
    date_col = next((c for c in ["date","timestamp","Date"] if c in df.columns), None)
    if date_col is None:
        raise SystemExit("no date/timestamp column")
    ret_cols = [c for c in df.columns if re.match(r"^ret_sl[0-9]", str(c))]
    if not ret_cols:
        raise SystemExit("no ret_sl*_tp* columns")
    keep = [date_col, "selector_probability"] + ret_cols + (["symbol"] if "symbol" in df.columns else [])
    df = df[keep].copy()
    if date_col != "date":
        df = df.rename(columns={date_col: "date"})
    if "symbol" not in df.columns:
        df["symbol"] = ""
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["selector_probability"] = pd.to_numeric(df["selector_probability"], errors="coerce")
    df = df[df["date"].notna() & np.isfinite(df["selector_probability"].astype(float))].copy()
    for c in ret_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(["date","selector_probability"], ascending=[True,False]).reset_index(drop=True)
    df["prob_rank"] = df.groupby("date", sort=False).cumcount().astype("int32") + 1
    return df, ret_cols

def load_dataset():
    if SCORED_CACHE.exists():
        log("loading_scored_cache", SCORED_CACHE)
        return normalize_frame(pd.read_parquet(SCORED_CACHE))
    log("loading_raw_cache", RAW_CACHE)
    df = score_selector(pd.read_parquet(RAW_CACHE))
    df, ret_cols = normalize_frame(df)
    df[["date","symbol","selector_probability","prob_rank"] + ret_cols].to_parquet(SCORED_CACHE, index=False)
    log("wrote_scored_cache", SCORED_CACHE, "rows", len(df))
    return df, ret_cols

def make_slice(frame, ret_cols):
    return {
        "prob": frame["selector_probability"].to_numpy(float, copy=True),
        "rank": frame["prob_rank"].to_numpy(np.int32, copy=True),
        "date": frame["date"].to_numpy(copy=True),
        "symbol": frame["symbol"].astype(str).to_numpy(copy=True),
        "rets": {c: frame[c].to_numpy(float, copy=True) for c in ret_cols},
    }

def selected_returns(slc, ret_col, threshold, max_new):
    ret = slc["rets"][ret_col]
    mask = (slc["prob"] >= threshold) & (slc["rank"] <= max_new) & np.isfinite(ret)
    return ret[mask]

def equity_metrics_arr(rets, position_pct):
    rets = np.asarray(rets, dtype=float)
    if rets.size == 0 or not np.isfinite(rets).all():
        return None
    factors = 1.0 + position_pct * (rets / 100.0)
    if not np.isfinite(factors).all() or np.any(factors <= 0):
        return None
    curve = 100000.0 * np.cumprod(factors)
    peaks = np.maximum.accumulate(curve)
    dd = (peaks - curve) / peaks * 100.0
    return {
        "trades": int(rets.size),
        "final_equity": round(float(curve[-1]), 2),
        "return_pct": round(float((curve[-1] / 100000.0 - 1.0) * 100.0), 4),
        "max_drawdown_pct": round(float(np.max(dd)), 4),
        "avg_trade_net_return_pct": round(float(np.mean(rets)), 4),
        "win_rate_pct": round(float(np.mean(rets > 0) * 100.0), 2),
    }

def objective(m):
    if not m or m["trades"] < MIN_VAL_TRADES:
        return -1e12
    if m["return_pct"] <= 0 or m["avg_trade_net_return_pct"] <= 0 or m["win_rate_pct"] < 50 or m["max_drawdown_pct"] > MAX_DD_LIMIT:
        return -1e12
    return m["return_pct"]*4 + m["avg_trade_net_return_pct"]*35 + m["win_rate_pct"]*0.10 - m["max_drawdown_pct"]*3 + min(m["trades"],10000)/10000

def deployable(s, valid):
    if not s or valid < MIN_VALID_FOLDS:
        return False
    return s["return_pct"] > 0 and s["avg_trade_net_return_pct"] > 0 and s["win_rate_pct"] >= 50 and s["max_drawdown_pct"] < MAX_DD_LIMIT

def main():
    df, ret_cols = load_dataset()
    dates = np.array(sorted(pd.Series(df["date"].dropna().unique()).to_numpy()))
    test_span = max(80, len(dates)//(FOLDS+2))
    log("dataset", "rows", len(df), "dates", len(dates), "position_pct", POSITION_PCT)

    folds = []
    for fold in range(1, FOLDS+1):
        test_end_i = len(dates) - (FOLDS-fold)*test_span
        test_start_i = test_end_i - test_span
        train_end_i = test_start_i
        train_dates = dates[:train_end_i]
        val_dates = train_dates[max(0, int(len(train_dates)*0.82)):]
        test_dates = dates[test_start_i:test_end_i]
        folds.append({
            "fold": fold,
            "val": make_slice(df[df["date"].isin(val_dates)], ret_cols),
            "test": make_slice(df[df["date"].isin(test_dates)], ret_cols),
        })

    ranked = []
    for ret_col in ret_cols:
        for th in THRESHOLDS:
            for max_new in MAX_NEW_PER_DAY:
                vals, scores, valid = [], [], 0
                for fs in folds:
                    m = equity_metrics_arr(selected_returns(fs["val"], ret_col, th, max_new), POSITION_PCT)
                    obj = objective(m)
                    vals.append(m)
                    if obj > -1e11:
                        valid += 1
                        scores.append(obj)
                if valid >= MIN_VALID_FOLDS:
                    ranked.append({
                        "policy": {"ret_col": ret_col, "threshold": th, "max_new_per_day": max_new},
                        "valid_validation_folds": valid,
                        "validation_score": round(float(np.mean(scores)), 4),
                        "validation": vals,
                    })

    ranked.sort(key=lambda x: (x["valid_validation_folds"], x["validation_score"]), reverse=True)
    policy = ranked[0]["policy"]
    test_folds, all_rets = [], []
    for fs in folds:
        r = selected_returns(fs["test"], policy["ret_col"], policy["threshold"], policy["max_new_per_day"])
        all_rets.append(r)
        test_folds.append({"test": equity_metrics_arr(r, POSITION_PCT)})

    summary = equity_metrics_arr(np.concatenate(all_rets), POSITION_PCT)
    result = {
        "stable": {
            "policy": policy,
            "summary": {"valid_folds": sum(1 for x in test_folds if x["test"]), "folds": FOLDS, "position_pct": POSITION_PCT, **summary},
            "test_folds": test_folds,
            "validation": ranked[0],
            "deployable": deployable(summary, sum(1 for x in test_folds if x["test"])),
        }
    }
    OUT.write_text(json.dumps(result, indent=2, default=str) + "\n")
    log("STABLE_DONE", json.dumps(result["stable"]["summary"], indent=2))
    if result["stable"]["deployable"]:
        env = {
            "OHLCV_SELECTOR_MONITOR_ONLY": "1",
            "OHLCV_SELECTOR_THRESHOLD": str(policy["threshold"]),
            "AUTO_TRADE_MAX_NEW_PER_CYCLE": str(policy["max_new_per_day"]),
            "NORMAL_LANE_MAX_NEW_PER_CYCLE": str(policy["max_new_per_day"]),
            "NORMAL_LANE_BASE_POSITION_PCT": str(POSITION_PCT),
            "NORMAL_LANE_MAX_POSITION_PCT": str(POSITION_PCT),
            "OHLCV_MONEY_RET_COL": policy["ret_col"],
        }
        ENVOUT.write_text("\n".join(f"{k}={v}" for k,v in env.items()) + "\n")
        log("WROTE_ENV", ENVOUT)

if __name__ == "__main__":
    main()
