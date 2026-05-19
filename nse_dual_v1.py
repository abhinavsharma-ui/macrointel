"""NSE Dual Regime Model v1"""
import json, os, warnings, pickle
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression

ROOT      = Path("data/prices_full")
OUT       = Path("reports/nse_dual_v1.json")
TRADES    = Path("reports/nse_dual_v1_trades.csv")
MODEL_DIR = Path("data/models/checkpoints")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MIN_ADV, MAX_ADV, MIN_PRICE = 200000, 5000000, 5
HOLD, MAX_ABS, MAX_TRAIN    = 60, 50, 500000
COST, VIX_THRESHOLD         = 0.22, 16.0

FOLDS = [
    ("2008-01-01","2013-01-01","2015-01-01"),
    ("2008-01-01","2017-01-01","2019-01-01"),
    ("2008-01-01","2021-01-01","2023-01-01"),
    ("2008-01-01","2024-01-01","2026-01-01"),
]

def log(*a): print("NSE_DUAL", *a, flush=True)

def load_vix_map():
    vix = pd.read_parquet(ROOT / "INDIAVIX.parquet").reset_index()
    vix.columns = [c.lower() for c in vix.columns]
    vix["date"] = pd.to_datetime(vix["date"]).dt.normalize()
    vix = vix.sort_values("date")
    vix["vix_ma20"]  = vix["close"].rolling(20, min_periods=5).mean()
    vix["vix_ma60"]  = vix["close"].rolling(60, min_periods=20).mean()
    vix["vix_ma10"]  = vix["close"].rolling(10, min_periods=3).mean()
    vix["vix_spike"] = (vix["close"] / vix["vix_ma20"]).clip(0, 5)
    vix["vix_trend"] = (vix["vix_ma10"] / vix["vix_ma60"]).clip(0, 5)
    result = {}
    for _, r in vix.iterrows():
        result[r["date"]] = {
            "vix_level":   r["close"],
            "vix_change":  r["close"] - vix.loc[max(0,_-1),"close"] if _ > 0 else 0.0,
            "vix_regime":  int(r["close"] > VIX_THRESHOLD),
            "vix_vs_ma20": r["close"] / r["vix_ma20"] if r["vix_ma20"] > 0 else 1.0,
            "vix_vs_ma60": r["close"] / r["vix_ma60"] if r["vix_ma60"] > 0 else 1.0,
            "vix_spike":   r["vix_spike"],
            "vix_trend":   r["vix_trend"],
            "vix_pct_5d":  vix["close"].pct_change(5).iloc[_] * 100 if not pd.isna(vix["close"].pct_change(5).iloc[_]) else 0.0,
            "vix_pct_20d": vix["close"].pct_change(20).iloc[_] * 100 if not pd.isna(vix["close"].pct_change(20).iloc[_]) else 0.0,
        }
    log(f"VIX loaded {len(result)} dates")
    return result

def load_nifty_maps():
    n = pd.read_parquet(ROOT / "NIFTY50.parquet").reset_index()
    n.columns = [c.lower() for c in n.columns]
    n["date"] = pd.to_datetime(n["date"]).dt.normalize()
    n = n.sort_values("date").reset_index(drop=True)
    prices = n["close"]
    fwd_ret, regime = {}, {}
    for i in range(len(n)):
        d = n.loc[i, "date"]
        j = min(i + HOLD + 1, len(n) - 1)
        fwd_ret[d] = (prices.iloc[j] / prices.iloc[i] - 1) * 100
        sma50  = prices.iloc[max(0,i-50):i+1].mean()
        sma200 = prices.iloc[max(0,i-200):i+1].mean()
        ret20  = (prices.iloc[i] / prices.iloc[max(0,i-20)] - 1) * 100
        ret60  = (prices.iloc[i] / prices.iloc[max(0,i-60)] - 1) * 100
        ret120 = (prices.iloc[i] / prices.iloc[max(0,i-120)] - 1) * 100
        regime[d] = {
            "nifty_ret_5d":    (prices.iloc[i] / prices.iloc[max(0,i-5)]  - 1) * 100,
            "nifty_ret_20d":   ret20,
            "nifty_ret_60d":   ret60,
            "nifty_ret_120d":  ret120,
            "nifty_vs_sma50":  (prices.iloc[i] / sma50  - 1) * 100 if sma50  > 0 else 0.0,
            "nifty_vs_sma200": (prices.iloc[i] / sma200 - 1) * 100 if sma200 > 0 else 0.0,
            "nifty_trend":     int(sma50 > sma200),
        }
    log(f"Nifty maps loaded {len(fwd_ret)} dates")
    return fwd_ret, regime

VIX_MAP = load_vix_map()
NIFTY_RET_MAP, NIFTY_REGIME_MAP = load_nifty_maps()

def read_price_file(p):
    try:
        df = pd.read_parquet(p)
        df.columns = [c.lower() for c in df.columns]
        if "date" not in df.columns:
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        needed = ["date","open","high","low","close","volume"]
        if not all(c in df.columns for c in needed):
            return None
        df = df[needed].dropna(subset=["close","open"])
        df["symbol"] = p.stem
        df["adv20_dollar_vol"] = df["close"] * df["volume"].rolling(20, min_periods=5).mean()
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return None

def add_features(x):
    c = x["close"]
    h = x["high"]
    l = x["low"]
    v = x["volume"]

    for d in [3,5,10,20,40,60,90,120,180,252]:
        x[f"return_{d}d"] = c.pct_change(d) * 100

    sma10  = c.rolling(10,  min_periods=5).mean()
    sma20  = c.rolling(20,  min_periods=10).mean()
    sma50  = c.rolling(50,  min_periods=20).mean()
    sma100 = c.rolling(100, min_periods=40).mean()
    sma150 = c.rolling(150, min_periods=60).mean()
    sma200 = c.rolling(200, min_periods=60).mean()

    x["close_vs_sma_10"]  = (c / sma10.replace(0,1)  - 1) * 100
    x["close_vs_sma_20"]  = (c / sma20.replace(0,1)  - 1) * 100
    x["close_vs_sma_50"]  = (c / sma50.replace(0,1)  - 1) * 100
    x["close_vs_sma_100"] = (c / sma100.replace(0,1) - 1) * 100
    x["close_vs_sma_200"] = (c / sma200.replace(0,1) - 1) * 100
    x["sma_10_vs_20"]     = (sma10  / sma20.replace(0,1)  - 1) * 100
    x["sma_20_vs_50"]     = (sma20  / sma50.replace(0,1)  - 1) * 100
    x["sma_50_vs_100"]    = (sma50  / sma100.replace(0,1) - 1) * 100
    x["sma_50_vs_150"]    = (sma50  / sma150.replace(0,1) - 1) * 100
    x["sma_50_vs_200"]    = (sma50  / sma200.replace(0,1) - 1) * 100
    x["sma_100_vs_200"]   = (sma100 / sma200.replace(0,1) - 1) * 100

    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr14  = tr.rolling(14, min_periods=5).mean()
    atr60  = tr.rolling(60, min_periods=20).mean()
    x["atr_pct"]         = atr14 / c.replace(0,1) * 100
    x["atr_expansion"]   = atr14 / atr60.replace(0,1)
    x["atr_compression"] = (atr14 < atr60 * 0.8).astype(int)

    high52 = c.rolling(252, min_periods=60).max()
    low52  = c.rolling(252, min_periods=60).min()
    x["pct_from_52w_high"] = (c / high52.replace(0,1) - 1) * 100
    x["range_52w_pct"]     = (high52 - low52) / high52.replace(0,1) * 100
    x["near_52w_high"]     = (c >= high52 * 0.95).astype(int)
    x["new_52w_high"]      = (c >= high52).astype(int)

    don10_h = h.rolling(10).max()
    don10_l = l.rolling(10).min()
    don20_h = h.rolling(20).max()
    don20_l = l.rolling(20).min()
    don60_h = h.rolling(60).max()
    don60_l = l.rolling(60).min()
    x["donchian_10_pct"] = (c - don10_l) / (don10_h - don10_l).replace(0,1)
    x["donchian_20_pct"] = (c - don20_l) / (don20_h - don20_l).replace(0,1)
    x["donchian_60_pct"] = (c - don60_l) / (don60_h - don60_l).replace(0,1)

    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14, min_periods=5).mean()
    loss  = (-delta.clip(upper=0)).rolling(14, min_periods=5).mean()
    rs    = gain / loss.replace(0, 1e-9)
    x["rsi_14"]   = 100 - 100 / (1 + rs)
    x["rsi_trend"] = x["rsi_14"] - x["rsi_14"].rolling(10).mean()

    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9).mean()
    x["macd_signal"] = macd - sig

    up_dm   = (h.diff().clip(lower=0))
    dn_dm   = (-l.diff().clip(upper=0))
    atr14s  = tr.rolling(14, min_periods=5).sum()
    di_plus  = 100 * up_dm.rolling(14, min_periods=5).sum() / atr14s.replace(0,1)
    di_minus = 100 * dn_dm.rolling(14, min_periods=5).sum() / atr14s.replace(0,1)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0,1)
    x["adx"]      = dx.rolling(14, min_periods=5).mean()
    x["di_spread"] = di_plus - di_minus
    x["di_ratio"]  = di_plus / di_minus.replace(0,1)

    vol20  = v.rolling(20, min_periods=5).mean()
    vol60  = v.rolling(60, min_periods=20).mean()
    x["vol_20d"]         = v / vol20.replace(0,1)
    x["vol_60d"]         = v / vol60.replace(0,1)
    x["vol_regime_ratio"] = vol20 / vol60.replace(0,1)
    x["volume_trend"]    = (vol20 > vol60).astype(int)
    x["vol_compression"] = (v < vol20 * 0.5).astype(int)

    up_days   = (c.diff() > 0).astype(float)
    up_vol    = (v * up_days).rolling(20, min_periods=5).sum()
    total_vol = v.rolling(20, min_periods=5).sum()
    x["up_vol_ratio"] = up_vol / total_vol.replace(0,1)

    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    x["obv_trend"]    = (obv > obv.rolling(20).mean()).astype(int)
    x["obv_momentum"] = obv.pct_change(20) * 100

    ret_daily = c.pct_change()
    for w in [60, 120, 252]:
        roll_mean = ret_daily.rolling(w, min_periods=w//2).mean()
        roll_std  = ret_daily.rolling(w, min_periods=w//2).std()
        x[f"momentum_sharpe_{w}d"] = roll_mean / roll_std.replace(0,1e-9) * np.sqrt(252)
        x[f"return_zscore_{w}d"]   = (ret_daily - roll_mean) / roll_std.replace(0,1e-9)

    x["vol_quality"] = x["momentum_sharpe_60d"] * (1 - x["atr_pct"] / 100)

    cons = (c.diff() > 0).astype(int)
    x["momentum_consistency"] = cons.rolling(60, min_periods=20).mean()
    x["momentum_accel"]       = x["return_20d"] - x["return_60d"]

    x["days_above_sma200"] = (c > sma200).astype(int).rolling(60, min_periods=20).sum()
    x["days_above_sma50"]  = (c > sma50).astype(int).rolling(20, min_periods=5).sum()

    gap = (x["open"] / c.shift() - 1) * 100
    x["gap_up_freq"]     = (gap > 1).astype(int).rolling(60, min_periods=20).mean()
    x["gap_follow_thru"] = ((gap > 0.5) & (c > x["open"])).astype(int).rolling(20).mean()

    x["high_quality"]  = ((x["momentum_sharpe_120d"] > 0.5) & (x["adx"] > 20) & (c > sma200)).astype(int)
    x["extension_score"] = x["close_vs_sma_20"] + x["close_vs_sma_50"]
    x["overextension"]   = (x["extension_score"] > 15).astype(int)

    x["momentum_quality"] = (
        (x["return_120d"] > 0).astype(int) +
        (x["return_60d"]  > 0).astype(int) +
        (x["return_20d"]  > 0).astype(int) +
        (x["adx"] > 20).astype(int) +
        (c > sma200).astype(int)
    )
    x["trend_composite"] = (
        x["close_vs_sma_50"] * 0.3 +
        x["return_20d"] * 0.2 +
        x["adx"] * 0.3 +
        x["donchian_60_pct"] * 10 * 0.2
    )
    return x

def add_vix_features(x):
    dates = pd.to_datetime(x["date"]).dt.normalize()
    for key, default in [
        ("vix_level",20.0),("vix_change",0.0),("vix_regime",0),
        ("vix_vs_ma20",1.0),("vix_vs_ma60",1.0),("vix_spike",1.0),
        ("vix_trend",1.0),("vix_pct_5d",0.0),("vix_pct_20d",0.0),
    ]:
        x[key] = dates.map(lambda d, k=key, df=default: VIX_MAP.get(d, {}).get(k, df))
    return x

def add_nifty_regime_features(x):
    dates = pd.to_datetime(x["date"]).dt.normalize()
    for key, default in [
        ("nifty_ret_5d",0.0),("nifty_ret_20d",0.0),("nifty_ret_60d",0.0),
        ("nifty_ret_120d",0.0),("nifty_vs_sma50",0.0),("nifty_vs_sma200",0.0),
        ("nifty_trend",1),
    ]:
        x[key] = dates.map(lambda d, k=key, df=default: NIFTY_REGIME_MAP.get(d, {}).get(k, df))
    return x

def load_one(p):
    if p.stem in ("INDIAVIX","NIFTY50","SPY","SPY_US"):
        return None
    if not p.stem.endswith(".NS"):
        return None
    x = read_price_file(p)
    if x is None:
        return None
    adv = x["adv20_dollar_vol"].fillna(0)
    x = x[(adv >= MIN_ADV) & (adv < MAX_ADV)].copy()
    x = x[x["close"].fillna(0) >= MIN_PRICE].copy()
    if len(x) < 100:
        return None
    x["entry_open"]    = x["open"].shift(-1)
    x["exit_close"]    = x["close"].shift(-(HOLD + 1))
    x["model_ret"]     = (x["exit_close"] / x["entry_open"].replace(0,1) - 1) * 100
    dates_norm         = pd.to_datetime(x["date"]).dt.normalize()
    x["nifty_fwd_ret"] = dates_norm.map(lambda d: NIFTY_RET_MAP.get(d, 0.0))
    x["rel_ret"]       = x["model_ret"] - x["nifty_fwd_ret"]
    x["honest_ret"]    = x["model_ret"]
    x = add_features(x)
    x = add_vix_features(x)
    x = add_nifty_regime_features(x)
    x = x[x["model_ret"].abs() <= MAX_ABS].copy()
    return x.dropna(subset=["rel_ret","entry_open"]).reset_index(drop=True)

def load_all():
    files = sorted(Path(ROOT).glob("*.parquet"))
    chunks, n = [], 0
    for p in files:
        df = load_one(p)
        if df is not None and len(df) > 0:
            chunks.append(df)
        n += 1
        if n % 200 == 0:
            loaded = sum(len(c) for c in chunks)
            log(f"loaded {n} files rows {loaded}")
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

BANNED = {
    "date","symbol","year","open","high","low","close","volume",
    "entry_open","exit_close","model_ret","honest_ret",
    "adv20_dollar_vol","nifty_fwd_ret","rel_ret",
}

HIGHVIX_PREFERRED = [
    "trend_composite","momentum_quality","momentum_consistency",
    "momentum_sharpe_60d","momentum_sharpe_120d","momentum_accel",
    "return_120d","return_252d","return_60d","return_40d","return_20d",
    "adx","di_spread","di_ratio",
    "donchian_60_pct","donchian_20_pct","donchian_10_pct",
    "atr_expansion","atr_pct","atr_compression",
    "pct_from_52w_high","near_52w_high","new_52w_high","range_52w_pct",
    "days_above_sma200","days_above_sma50",
    "volume_trend","vol_quality","up_vol_ratio","obv_trend","obv_momentum",
    "vix_level","vix_regime","vix_spike","vix_trend","vix_vs_ma20","vix_pct_5d",
    "return_zscore_120d","return_zscore_60d",
    "nifty_ret_20d","nifty_ret_60d","nifty_trend",
]

LOWVIX_PREFERRED = [
    "momentum_sharpe_120d","momentum_sharpe_252d","momentum_sharpe_60d",
    "momentum_quality","momentum_consistency","return_zscore_120d","return_zscore_60d",
    "return_120d","return_252d","return_180d","return_60d",
    "close_vs_sma_50","close_vs_sma_100","close_vs_sma_200",
    "sma_50_vs_200","sma_50_vs_150","sma_100_vs_200",
    "high_quality","gap_follow_thru","gap_up_freq",
    "vol_quality","up_vol_ratio","vol_compression","obv_momentum",
    "pct_from_52w_high","range_52w_pct","near_52w_high","days_above_sma200",
    "rsi_14","rsi_trend","macd_signal",
    "extension_score","overextension",
    "vol_20d","vol_60d","vol_regime_ratio",
    "nifty_vs_sma50","nifty_vs_sma200","nifty_ret_60d","nifty_ret_120d","nifty_trend",
    "vix_level","vix_vs_ma60","vix_trend",
]

def feature_cols(df, preferred):
    all_cols = [c for c in df.columns
                if c not in BANNED
                and pd.api.types.is_numeric_dtype(df[c])]
    pref = [c for c in preferred if c in all_cols]
    rest = [c for c in all_cols if c not in pref]
    return pref + rest

def train_ensemble(X_tr, y_tr, X_val, y_val, fold, regime):
    models = []
    for seed in [0, 100, 200]:
        if regime == "high":
            m = XGBClassifier(
                n_estimators=600, max_depth=5, learning_rate=0.02,
                min_child_weight=20, gamma=1.0, subsample=0.8,
                colsample_bytree=0.7, scale_pos_weight=1.0,
                early_stopping_rounds=40, eval_metric="logloss",
                random_state=seed, n_jobs=-1, verbosity=0,
            )
        else:
            m = XGBClassifier(
                n_estimators=600, max_depth=4, learning_rate=0.02,
                min_child_weight=30, gamma=2.0, subsample=0.8,
                colsample_bytree=0.7, scale_pos_weight=1.5,
                early_stopping_rounds=40, eval_metric="logloss",
                random_state=seed, n_jobs=-1, verbosity=0,
            )
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        models.append(m)
    return models

def ensemble_predict(models, X):
    probs = np.mean([m.predict_proba(X)[:,1] for m in models], axis=0)
    return probs

def calibrate(probs_val, y_val, probs_test):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs_val, y_val)
    return iso.predict(probs_test), iso

def evaluate_fold(test_r, probs, fs, fold, regime, train_end, test_end):
    t = test_r.copy()
    t["prob"] = probs
    t["label"] = (t["rel_ret"] > 2.0).astype(int)
    t = t.sort_values(["date","prob"], ascending=[True,False])

    # simulate: top signal per day, max 15 open
    trades = []
    open_pos = []
    for date, grp in t.groupby("date"):
        # close positions past hold period
        open_pos = [(d, sym, ep, ex) for d, sym, ep, ex in open_pos
                    if (pd.Timestamp(date) - pd.Timestamp(d)).days < HOLD]
        slots = 15 - len(open_pos)
        if slots <= 0:
            continue
        candidates = grp[grp["prob"] >= 0.52].head(slots)
        for _, row in candidates.iterrows():
            net = row["model_ret"] - COST
            rel = row["rel_ret"] - COST
            trades.append({
                "date": date, "symbol": row["symbol"],
                "prob": row["prob"], "model_ret": row["model_ret"],
                "net_ret": net, "rel_net_ret": rel,
                "label": row["label"], "fold": fold, "regime": regime,
            })
            open_pos.append((date, row["symbol"], row["prob"], row["model_ret"]))

    if not trades:
        return {"trades":0,"win_rate_pct":0,"rel_win_rate_pct":0,
                "avg_net_ret_pct":0,"avg_rel_ret_pct":0,"sharpe":0,"equity":100000}, []

    tr = pd.DataFrame(trades)
    wins_abs = (tr["net_ret"] > 0).mean() * 100
    wins_rel = (tr["rel_net_ret"] > 0).mean() * 100
    avg_net  = tr["net_ret"].mean()
    avg_rel  = tr["rel_net_ret"].mean()
    equity   = 100000 + tr["net_ret"].sum() * 100
    sharpe   = (tr["net_ret"].mean() / tr["net_ret"].std() * np.sqrt(252)
                if tr["net_ret"].std() > 0 else 0.0)

    res = {
        "fold": fold, "regime": regime,
        "train_end": train_end, "test_end": test_end,
        "trades": len(tr),
        "win_rate_pct":     round(wins_abs, 2),
        "rel_win_rate_pct": round(wins_rel, 2),
        "avg_net_ret_pct":  round(avg_net,  4),
        "avg_rel_ret_pct":  round(avg_rel,  4),
        "sharpe":           round(sharpe,   3),
        "equity":           round(equity,   2),
    }
    log(f"FOLD {fold} {regime}: trades={len(tr)} rel_wr={wins_rel:.1f}% avg_rel={avg_rel:.3f}% sharpe={sharpe:.3f}")
    return res, tr.to_dict("records")

log("loading data...")
df = load_all()
log(f"dataset {len(df)} rows {df['symbol'].nunique()} symbols")

fs_high = feature_cols(df, HIGHVIX_PREFERRED)
fs_low  = feature_cols(df, LOWVIX_PREFERRED)
log(f"High-VIX features: {len(fs_high)}  Low-VIX features: {len(fs_low)}")

all_trades, results = [], []

for fold, (train_start, train_end, test_end) in enumerate(FOLDS, 1):
    log(f"FOLD {fold}: train<{train_end} test {train_end}->{test_end}")
    train_all = df[(df["date"] >= train_start) & (df["date"] < train_end)].copy()
    test_all  = df[(df["date"] >= train_end)   & (df["date"] < test_end)].copy()
    if len(train_all) < 500 or len(test_all) < 100:
        log(f"FOLD {fold}: insufficient data, skipping")
        continue

    train_high = train_all[train_all["vix_level"] >  VIX_THRESHOLD].copy()
    train_low  = train_all[train_all["vix_level"] <= VIX_THRESHOLD].copy()
    test_high  = test_all[test_all["vix_level"]  >  VIX_THRESHOLD].copy()
    test_low   = test_all[test_all["vix_level"]  <= VIX_THRESHOLD].copy()
    log(f"  train high={len(train_high)} low={len(train_low)} | test high={len(test_high)} low={len(test_low)}")

    for regime, train_r, test_r, fs in [
        ("high", train_high, test_high, fs_high),
        ("low",  train_low,  test_low,  fs_low),
    ]:
        if len(train_r) < 200 or len(test_r) < 50:
            log(f"  {regime}: insufficient data, skipping")
            continue

        if len(train_r) > MAX_TRAIN:
            train_r = train_r.sample(MAX_TRAIN, random_state=42+fold)

        y = (train_r["rel_ret"] > 2.0).astype(int)
        if y.nunique() < 2:
            log(f"  {regime}: single class, skipping")
            continue
        log(f"  {regime}: {len(train_r)} rows label=1:{y.mean():.1%}")

        cutoff  = train_r["date"].quantile(0.80)
        val_mask = train_r["date"] >= cutoff
        X_tr  = train_r[~val_mask][fs].replace([np.inf,-np.inf],np.nan).fillna(0).astype("float32")
        y_tr  = y[~val_mask]
        X_val = train_r[val_mask][fs].replace([np.inf,-np.inf],np.nan).fillna(0).astype("float32")
        y_val = y[val_mask]

        if y_tr.nunique() < 2 or y_val.nunique() < 2:
            log(f"  {regime}: single class after split, skipping")
            continue

        models = train_ensemble(X_tr, y_tr, X_val, y_val, fold, regime)

        probs_val  = ensemble_predict(models, X_val)
        X_test     = test_r[fs].replace([np.inf,-np.inf],np.nan).fillna(0).astype("float32")
        probs_test_raw = ensemble_predict(models, X_test)
        probs_test, iso = calibrate(probs_val, y_val, probs_test_raw)

        ckpt = MODEL_DIR / f"nse_dual_{regime}_fold{fold}.pkl"
        with open(ckpt, "wb") as f:
            pickle.dump({"models": models, "calibrator": iso, "features": fs}, f)
        log(f"  saved {ckpt}")

        res, trades = evaluate_fold(test_r, probs_test, fs, fold, regime, train_end, test_end)
        results.append(res)
        all_trades.extend(trades)

log("saving results...")
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    json.dump(results, f, indent=2, default=str)

if all_trades:
    pd.DataFrame(all_trades).to_csv(TRADES, index=False)
    log(f"trades saved: {len(all_trades)} rows -> {TRADES}")

log("NSE_DUAL DONE")
for r in results:
    log(f"  fold={r['fold']} regime={r['regime']} trades={r['trades']} "
        f"rel_wr={r['rel_win_rate_pct']}% avg_rel={r['avg_rel_ret_pct']}% sharpe={r['sharpe']}")
