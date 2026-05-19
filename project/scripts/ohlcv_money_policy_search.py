import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path("data/features_26yr_liquid")
CACHE = Path("reports/ohlcv_money_dataset.parquet")
OUT = Path("reports/ohlcv_money_policy_search.json")

FEATURES = [
    "return_20d","return_60d","momentum_composite","vol_regime_ratio","atr_pct",
    "price_acceleration","close_vs_sma_50","close_vs_sma_200","rsi_14"
]

HORIZON = 10
POSITION_PCT = 0.01
ROUNDTRIP_COST_PCT = 0.35
POLICIES = [(1.5, 2.0), (1.5, 3.0), (2.0, 3.0), (2.0, 4.0), (2.5, 3.0), (2.5, 4.0), (3.0, 4.0), (3.0, 5.0)]
THRESHOLDS = np.round(np.arange(0.56, 0.86, 0.02), 4)
MAX_PER_DAY_GRID = [1, 2, 3]
MIN_VAL_TRADES = 80

def policy_col(sl, tp):
    return f"ret_sl{str(sl).replace('.','p')}_tp{str(tp).replace('.','p')}"

def parse_ts(df):
    if "timestamp" in df.columns:
        return pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return pd.to_datetime(df.index, utc=True, errors="coerce")

def num(s, default=0.0):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)

def load_one(p):
    try:
        df = pd.read_parquet(p)
        if df.empty or "close" not in df.columns:
            return None
        df = df.copy().sort_index()
        ts = parse_ts(df)
        close = num(df["close"])
        high = num(df.get("high", close))
        low = num(df.get("low", close))
        volume = num(df.get("volume", 0))

        work = pd.DataFrame({
            "symbol": p.stem,
            "timestamp": ts,
            "close": close,
            "volume": volume,
        })
        for f in FEATURES:
            if f in df.columns:
                work[f] = num(df[f])
            elif f == "return_20d" and "momentum_20d" in df.columns:
                work[f] = num(df["momentum_20d"])
            elif f == "return_60d" and "momentum_60d" in df.columns:
                work[f] = num(df["momentum_60d"])
            else:
                work[f] = 0.0

        n = len(df)
        if n <= HORIZON + 50:
            return None

        c = close.to_numpy(dtype=float)
        h = high.to_numpy(dtype=float)
        l = low.to_numpy(dtype=float)

        fut_high = np.column_stack([np.r_[h[k:], np.full(k, np.nan)] for k in range(1, HORIZON + 1)])
        fut_low = np.column_stack([np.r_[l[k:], np.full(k, np.nan)] for k in range(1, HORIZON + 1)])
        fut_close = np.r_[c[HORIZON:], np.full(HORIZON, np.nan)]

        work["edge_pct"] = (np.nanmax(fut_high, axis=1) / c - 1.0) * 100.0
        work["drawdown_pct"] = (1.0 - np.nanmin(fut_low, axis=1) / c) * 100.0

        for sl, tp in POLICIES:
            tp_hit = fut_high >= (c[:, None] * (1.0 + tp / 100.0))
            sl_hit = fut_low <= (c[:, None] * (1.0 - sl / 100.0))
            tp_any = tp_hit.any(axis=1)
            sl_any = sl_hit.any(axis=1)
            tp_first = np.where(tp_any, tp_hit.argmax(axis=1), 999)
            sl_first = np.where(sl_any, sl_hit.argmax(axis=1), 999)
            raw = (fut_close / c - 1.0) * 100.0
            raw = np.where(tp_any & (tp_first <= sl_first), tp, raw)
            raw = np.where(sl_any & (sl_first < tp_first), -sl, raw)
            work[policy_col(sl, tp)] = raw - ROUNDTRIP_COST_PCT

        work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
        work = work.replace([np.inf, -np.inf], np.nan)
        return work.dropna(subset=["timestamp", "close", "edge_pct", "drawdown_pct"] + FEATURES)
    except Exception as e:
        print("LOAD_ERROR", p.name, repr(e), flush=True)
        return None

def build_or_load():
    if CACHE.exists() and CACHE.stat().st_size > 50_000_000:
        print("LOADING_CACHE", CACHE, CACHE.stat().st_size, flush=True)
        return pd.read_parquet(CACHE)

    parts = []
    files = sorted(ROOT.glob("*.parquet"))
    print("BUILDING_MONEY_DATASET", len(files), flush=True)
    for i, p in enumerate(files, 1):
        r = load_one(p)
        if r is not None and not r.empty:
            parts.append(r)
        if i % 100 == 0 or i == len(files):
            print("load", i, "ok", len(parts), "rows", sum(len(x) for x in parts), flush=True)

    if not parts:
        raise SystemExit("NO_DATA")
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(CACHE, index=False)
    print("SAVED_CACHE", CACHE, CACHE.stat().st_size, len(df), flush=True)
    return df

def simulate_trades(trades):
    equity = 100000.0
    peak = equity
    max_dd = 0.0
    for r in trades:
        equity += equity * POSITION_PCT * (float(r) / 100.0)
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak * 100.0)
    return {
        "final_equity": round(equity, 2),
        "return_pct": round((equity / 100000.0 - 1.0) * 100.0, 4),
        "max_drawdown_pct": round(max_dd, 4),
    }

def select_daily(df, prob, threshold, max_per_day, ret_col):
    tmp = df[["date", "symbol", ret_col]].copy()
    tmp["prob"] = prob
    tmp = tmp[tmp["prob"] >= threshold]
    if tmp.empty:
        return tmp
    return tmp.sort_values(["date", "prob"], ascending=[True, False]).groupby("date").head(max_per_day)

df = build_or_load()
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
df["date"] = df["timestamp"].dt.normalize()
df = df.dropna(subset=["timestamp", "date", "edge_pct", "drawdown_pct"] + FEATURES)
df["take_label"] = (
    (df["edge_pct"] >= 0.90)
    & (df["drawdown_pct"] <= 2.50)
    & ((df["edge_pct"] / df["drawdown_pct"].clip(lower=0.35)) >= 0.30)
).astype(int)

print("DATASET", len(df), "symbols", df["symbol"].nunique(), "positives", int(df["take_label"].sum()), flush=True)

dates = sorted(df["date"].unique())
folds = 4
span = len(dates) // (folds + 1)
fold_rows = []
all_test_trades = []

for i in range(folds):
    train_end = dates[span * (i + 1)]
    test_end = dates[span * (i + 2)] if i < folds - 1 else dates[-1]

    train_all = df[df["date"] <= train_end].copy()
    test = df[(df["date"] > train_end) & (df["date"] <= test_end)].copy()
    inner_dates = sorted(train_all["date"].unique())
    val_start = inner_dates[int(len(inner_dates) * 0.80)]
    train = train_all[train_all["date"] < val_start].copy()
    val = train_all[train_all["date"] >= val_start].copy()

    model = HistGradientBoostingClassifier(max_iter=180, learning_rate=0.045, max_leaf_nodes=31, l2_regularization=0.05, random_state=700+i)
    model.fit(train[FEATURES].fillna(0), train["take_label"].astype(int))
    val_prob = model.predict_proba(val[FEATURES].fillna(0))[:, 1]

    best = None
    for sl, tp in POLICIES:
        ret_col = policy_col(sl, tp)
        for th in THRESHOLDS:
            for max_day in MAX_PER_DAY_GRID:
                selected = select_daily(val, val_prob, th, max_day, ret_col)
                n = len(selected)
                if n < MIN_VAL_TRADES:
                    continue
                returns = selected[ret_col].astype(float).to_numpy()
                sim = simulate_trades(returns)
                avg = float(np.mean(returns))
                win = float(np.mean(returns > 0) * 100.0)
                profit_factor = float(returns[returns > 0].sum() / abs(returns[returns < 0].sum())) if (returns < 0).any() else 99.0
                if avg <= 0 or win < 52:
                    continue
                objective = sim["return_pct"] - 0.45 * sim["max_drawdown_pct"] + 0.08 * win + 0.35 * min(profit_factor, 3.0)
                rec = {
                    "objective": round(objective, 4),
                    "threshold": float(th),
                    "max_new_per_day": int(max_day),
                    "stop_loss_pct": sl,
                    "take_profit_pct": tp,
                    "ret_col": ret_col,
                    "val_trades": int(n),
                    "val_avg_net_return_pct": round(avg, 4),
                    "val_win_rate_pct": round(win, 2),
                    "val_profit_factor": round(profit_factor, 4),
                    **sim,
                }
                if best is None or rec["objective"] > best["objective"]:
                    best = rec

    if best is None:
        fold_rows.append({"fold": i + 1, "skipped": True, "reason": "no_profitable_validation_policy"})
        print("FOLD_SKIP", fold_rows[-1], flush=True)
        continue

    final = HistGradientBoostingClassifier(max_iter=180, learning_rate=0.045, max_leaf_nodes=31, l2_regularization=0.05, random_state=800+i)
    final.fit(train_all[FEATURES].fillna(0), train_all["take_label"].astype(int))
    test_prob = final.predict_proba(test[FEATURES].fillna(0))[:, 1]
    selected_test = select_daily(test, test_prob, best["threshold"], best["max_new_per_day"], best["ret_col"])
    returns = selected_test[best["ret_col"]].astype(float).to_numpy()
    sim = simulate_trades(returns)
    avg = float(np.mean(returns)) if len(returns) else 0.0
    win = float(np.mean(returns > 0) * 100.0) if len(returns) else 0.0
    rec = {
        "fold": i + 1,
        "selected_policy": best,
        "test_trades": int(len(selected_test)),
        "test_avg_net_return_pct": round(avg, 4),
        "test_win_rate_pct": round(win, 2),
        "test": sim,
        "test_start": str(test["date"].min()),
        "test_end": str(test["date"].max()),
    }
    fold_rows.append(rec)
    selected_test = selected_test.copy()
    selected_test["fold"] = i + 1
    all_test_trades.append(selected_test[["fold", "date", "symbol", "prob", best["ret_col"]]].rename(columns={best["ret_col"]: "net_return_pct"}))
    print("FOLD_DONE", json.dumps(rec, default=str), flush=True)

trades = pd.concat(all_test_trades, ignore_index=True) if all_test_trades else pd.DataFrame(columns=["fold","date","symbol","prob","net_return_pct"])
combined = simulate_trades(trades["net_return_pct"].astype(float).to_numpy()) if len(trades) else {"final_equity": 100000.0, "return_pct": 0.0, "max_drawdown_pct": 0.0}
summary = {
    "trades": int(len(trades)),
    "avg_trade_net_return_pct": round(float(trades["net_return_pct"].mean()) if len(trades) else 0.0, 4),
    "win_rate_pct": round(float((trades["net_return_pct"] > 0).mean() * 100.0) if len(trades) else 0.0, 2),
    "position_pct": POSITION_PCT,
    "roundtrip_cost_pct": ROUNDTRIP_COST_PCT,
    **combined,
}
payload = {"summary": summary, "folds": fold_rows, "last_50_trades": trades.tail(50).to_dict("records")}
OUT.write_text(json.dumps(payload, indent=2, default=str))
print("MONEY_POLICY_DONE")
print(json.dumps(summary, indent=2))
