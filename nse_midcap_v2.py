"""
NSE Mid/Small Cap - Trend Following Model v2
Universe : *.NS.parquet, ADV $200k-$5M
Strategy : Trend following - ride structural breakouts
"""
import json, os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
import pickle

ROOT      = Path(os.getenv("SIM_ROOT",      "data/prices_full"))
OUT       = Path(os.getenv("SIM_OUT",       "reports/nse_midcap_v2.json"))
TRADES    = Path(os.getenv("SIM_TRADES",    "reports/nse_midcap_v2_trades.csv"))
MODEL_DIR = Path(os.getenv("SIM_MODEL_DIR", "data/models/checkpoints"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MIN_ADV   = float(os.getenv("SIM_MIN_ADV",     "200000"))
MAX_ADV   = float(os.getenv("SIM_MAX_ADV",     "5000000"))
MIN_PRICE = float(os.getenv("SIM_MIN_PRICE",   "5"))
HOLD      = int(os.getenv("SIM_HOLD",          "60"))
MAX_ABS   = float(os.getenv("SIM_MAX_ABS",     "50"))
MAX_TRAIN = int(os.getenv("SIM_MAX_TRAIN",     "600000"))
COST      = 0.22
MAX_OPEN  = 15

FOLDS = [
    ("2008-01-01", "2013-01-01", "2015-01-01"),
    ("2008-01-01", "2017-01-01", "2019-01-01"),
    ("2008-01-01", "2021-01-01", "2023-01-01"),
    ("2008-01-01", "2024-01-01", "2026-01-01"),
]

def log(*a): print("NSE_MC", *a, flush=True)

def load_vix_map():
    try:
        vix = pd.read_parquet(ROOT / "INDIAVIX.parquet").reset_index()
        vix.columns = [c.lower() for c in vix.columns]
        vix["date"] = pd.to_datetime(vix["date"]).dt.normalize()
        vix = vix.sort_values("date")
        vix["vix_change"] = vix["close"].pct_change() * 100
        vix["vix_ma20"]   = vix["close"].rolling(20, min_periods=5).mean()
        vix["vix_regime"] = (vix["close"] > 20).astype(int)
        result = {}
        for _, row in vix.iterrows():
            result[row["date"]] = {
                "vix_level":  row["close"],
                "vix_change": row["vix_change"],
                "vix_regime": row["vix_regime"],
                "vix_vs_ma":  row["close"] / row["vix_ma20"] if row["vix_ma20"] > 0 else 1.0,
            }
        log(f"VIX loaded {len(result)} dates")
        return result
    except Exception as e:
        log(f"VIX load failed: {e}")
        return {}

VIX_MAP = load_vix_map()
def load_nifty_ret_map():
    try:
        n = pd.read_parquet(ROOT / "NIFTY50.parquet").reset_index()
        n.columns = [c.lower() for c in n.columns]
        n["date"] = pd.to_datetime(n["date"]).dt.normalize()
        n = n.sort_values("date").set_index("date")["close"]
        result = {}
        for i in range(len(n)):
            d = n.index[i]
            future_idx = i + 61
            if future_idx < len(n):
                nifty_ret = (n.iloc[future_idx] / n.iloc[i] - 1) * 100
            else:
                nifty_ret = 0.0
            result[d] = nifty_ret
        log(f"Nifty ret map loaded {len(result)} dates")
        return result
    except Exception as e:
        log(f"Nifty ret map failed: {e}")
        return {}

NIFTY_RET_MAP = load_nifty_ret_map()


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

def add_vix_features(x):
    dates = pd.to_datetime(x["date"]).dt.normalize()
    x["vix_level"]  = dates.map(lambda d: VIX_MAP.get(d, {}).get("vix_level",  20.0))
    x["vix_change"] = dates.map(lambda d: VIX_MAP.get(d, {}).get("vix_change",  0.0))
    x["vix_regime"] = dates.map(lambda d: VIX_MAP.get(d, {}).get("vix_regime",  0))
    x["vix_vs_ma"]  = dates.map(lambda d: VIX_MAP.get(d, {}).get("vix_vs_ma",   1.0))
    return x

def add_features(x):
    c    = x["close"]
    h    = x["high"]
    l    = x["low"]
    v    = x["volume"]
    ret1 = c.pct_change()

    # Momentum
    x["return_5d"]   = c.pct_change(5)   * 100
    x["return_10d"]  = c.pct_change(10)  * 100
    x["return_20d"]  = c.pct_change(20)  * 100
    x["return_60d"]  = c.pct_change(60)  * 100
    x["return_120d"] = c.pct_change(120) * 100
    x["return_252d"] = c.pct_change(252) * 100

    # Moving averages
    sma_20  = c.rolling(20,  min_periods=10).mean()
    sma_50  = c.rolling(50,  min_periods=20).mean()
    sma_100 = c.rolling(100, min_periods=40).mean()
    sma_200 = c.rolling(200, min_periods=60).mean()
    x["close_vs_sma_50"]  = (c / sma_50.replace(0,1)  - 1) * 100
    x["close_vs_sma_200"] = (c / sma_200.replace(0,1) - 1) * 100
    x["sma_20_vs_50"]     = (sma_20 / sma_50.replace(0,1)  - 1) * 100
    x["sma_50_vs_200"]    = (sma_50 / sma_200.replace(0,1) - 1) * 100
    x["sma_50_vs_100"]    = (sma_50 / sma_100.replace(0,1) - 1) * 100

    # Donchian channels
    high_20 = h.rolling(20, min_periods=10).max()
    low_20  = l.rolling(20, min_periods=10).min()
    high_60 = h.rolling(60, min_periods=30).max()
    low_60  = l.rolling(60, min_periods=30).min()
    x["donchian_20_pct"] = (c - low_20) / (high_20 - low_20).replace(0, 1)
    x["donchian_60_pct"] = (c - low_60) / (high_60 - low_60).replace(0, 1)

    # 52-week
    high_52w = h.rolling(252, min_periods=60).max()
    low_52w  = l.rolling(252, min_periods=60).min()
    x["pct_from_52w_high"] = (c / high_52w.replace(0,1) - 1) * 100
    x["range_52w_pct"]     = (c - low_52w) / (high_52w - low_52w).replace(0, 1)
    x["near_52w_high"]     = (x["pct_from_52w_high"] >= -5).astype(int)

    # ATR
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr_14 = tr.rolling(14, min_periods=5).mean()
    atr_50 = tr.rolling(50, min_periods=20).mean()
    x["atr_pct"]       = atr_14 / c.replace(0,1) * 100
    x["atr_expansion"] = atr_14 / atr_50.replace(0, 1)

    # ADX
    up_move  = h - h.shift(1)
    dn_move  = l.shift(1) - l
    plus_dm  = up_move.where((up_move > dn_move) & (up_move > 0), 0)
    minus_dm = dn_move.where((dn_move > up_move) & (dn_move > 0), 0)
    atr_sm   = tr.ewm(span=14, min_periods=7).mean().replace(0, 1)
    plus_di  = 100 * plus_dm.ewm(span=14, min_periods=7).mean()  / atr_sm
    minus_di = 100 * minus_dm.ewm(span=14, min_periods=7).mean() / atr_sm
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1)
    x["adx"]       = dx.ewm(span=14, min_periods=7).mean()
    x["di_spread"] = plus_di - minus_di

    # Volatility
    x["vol_20d"]          = ret1.rolling(20, min_periods=10).std() * np.sqrt(252) * 100
    x["vol_60d"]          = ret1.rolling(60, min_periods=30).std() * np.sqrt(252) * 100
    x["vol_regime_ratio"] = x["vol_20d"] / x["vol_20d"].rolling(60, min_periods=20).mean().replace(0, 1)

    # Volume
    vol_sma_20 = v.rolling(20, min_periods=10).mean().replace(0, 1)
    vol_sma_60 = v.rolling(60, min_periods=30).mean().replace(0, 1)
    x["volume_ratio_1d"] = v / vol_sma_20
    x["volume_ratio_5d"] = v.rolling(5, min_periods=3).mean() / vol_sma_20
    x["volume_trend"]    = vol_sma_20 / vol_sma_60

    # OBV
    obv = (np.sign(ret1) * v).cumsum()
    x["obv_trend"] = (obv / obv.rolling(20, min_periods=10).mean().replace(0, 1) - 1) * 100

    # RSI
    delta = c.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta.clip(upper=0))
    x["rsi_14"] = 100 - 100 / (1 + gain.rolling(14, min_periods=7).mean() / loss.rolling(14, min_periods=7).mean().replace(0, 1e-9))

    # MACD
    ema_12 = c.ewm(span=12, min_periods=8).mean()
    ema_26 = c.ewm(span=26, min_periods=15).mean()
    macd_line   = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, min_periods=5).mean()
    x["macd"]      = macd_line   / c.replace(0,1) * 100
    x["macd_hist"] = (macd_line - signal_line) / c.replace(0,1) * 100

    # Trend composites
    x["trend_composite"]  = (
        x["return_120d"] * 0.35 +
        x["return_252d"] * 0.35 +
        x["return_60d"]  * 0.20 +
        x["donchian_60_pct"] * 10 * 0.10
    )
    x["momentum_quality"] = x["return_120d"] - x["vol_60d"]

    # Momentum consistency - are all timeframes aligned?
    x["momentum_consistency"] = (
        (x["return_20d"]  > 0).astype(int) +
        (x["return_60d"]  > 0).astype(int) +
        (x["return_120d"] > 0).astype(int) +
        (x["return_252d"] > 0).astype(int)
    )

    # Trend age - consecutive days above 200d SMA
    above_sma200 = (c > sma_200).astype(int)
    x["days_above_sma200"] = above_sma200.groupby(
        (above_sma200 != above_sma200.shift()).cumsum()
    ).cumcount() * above_sma200

    # Volume quality - up days vs down days volume
    up_vol   = v.where(ret1 > 0, 0).rolling(10, min_periods=5).mean()
    down_vol = v.where(ret1 < 0, 0).rolling(10, min_periods=5).mean()
    x["up_vol_ratio"] = up_vol   / vol_sma_20
    x["vol_quality"]  = up_vol   / down_vol.replace(0, 1)

    # Momentum acceleration
    x["momentum_accel"] = x["return_60d"] - x["return_120d"]

    # Return z-score
    vol60 = ret1.rolling(60, min_periods=20).std() * 100
    x["return_zscore_120d"] = x["return_120d"] / (vol60 * np.sqrt(120)).replace(0, 1)

    x = add_vix_features(x)
    return x

def load_one(p):
    if p.stem in ("INDIAVIX", "NIFTY50", "SPY", "SPY_US"):
        return None
    if not p.stem.endswith(".NS"):
        return None
    x = read_price_file(p)
    if x is None:
        return None
    adv = x["adv20_dollar_vol"].fillna(0)
    x = x[(adv >= MIN_ADV) & (adv < MAX_ADV)].copy()
    x = x[x["close"].fillna(0) >= MIN_PRICE].copy()
    if x.empty:
        return None

    x["entry_open"]  = x["open"].shift(-1)
    x["exit_close"]  = x["close"].shift(-(HOLD + 1))
    x["model_ret"]   = ((x["exit_close"] / x["entry_open"].replace(0,1)) - 1) * 100
    # Relative return: stock minus Nifty over same 60-day window
    dates_norm = pd.to_datetime(x["date"]).dt.normalize()
    x["nifty_fwd_ret"] = dates_norm.map(lambda d: NIFTY_RET_MAP.get(d, 0.0))
    x["rel_ret"] = x["model_ret"] - x["nifty_fwd_ret"]
    x["honest_ret"]  = x["model_ret"]
    x["exit_offset"] = HOLD + 1

    x = add_features(x)
    x = x.replace([np.inf, -np.inf], np.nan).dropna(subset=["model_ret","honest_ret"])
    x = x[(x["model_ret"].abs() <= MAX_ABS) & (x["honest_ret"].abs() <= MAX_ABS)]
    return x if not x.empty else None

def feature_cols(df):
    banned = {
        "date","symbol","year","open","high","low","close","volume",
        "entry_open","exit_close","model_ret","honest_ret",
        "adv20_dollar_vol","exit_offset","nifty_fwd_ret","rel_ret"
    }
    pref = ["trend_composite","momentum_quality","momentum_consistency",
            "return_120d","return_252d","return_60d","adx","donchian_60_pct",
            "atr_expansion","volume_trend","vol_quality","di_spread","obv_trend",
            "pct_from_52w_high","near_52w_high","days_above_sma200",
            "momentum_accel","return_zscore_120d",
            "vix_level","vix_regime","vix_vs_ma","vix_change"]
    out = [c for c in df.columns
           if c.lower() not in banned
           and pd.api.types.is_numeric_dtype(df[c])]
    return [c for c in pref if c in out] + [c for c in out if c not in pref]

def load_all():
    rows  = []
    files = sorted(ROOT.glob("*.NS.parquet"))
    for i, p in enumerate(files, 1):
        x = load_one(p)
        if x is not None and len(x) > 0:
            rows.append(x)
        if i % 100 == 0:
            log(f"loaded {i} files rows {sum(len(r) for r in rows)}")
    if not rows:
        raise ValueError("No data - check ADV range")
    df = pd.concat(rows, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date","symbol"]).reset_index(drop=True)

def simulate(test, test_dates):
    results = []
    for dt in sorted(test_dates):
        day = test[test["date"] == dt].copy()
        day = day[day["prob"] >= 0.55].sort_values("prob", ascending=False).head(MAX_OPEN)
        for _, row in day.iterrows():
            net = float(row["honest_ret"]) - COST
            net = max(min(net, MAX_ABS), -MAX_ABS)
            results.append({
                "date":       dt,
                "symbol":     row["symbol"],
                "prob":       row["prob"],
                "honest_ret": row["honest_ret"],
                "net_ret":    net,
            })

    if not results:
        return {"trades": 0, "win_rate_pct": 0, "avg_net_ret_pct": 0,
                "final_equity": 100000, "return_pct": 0}, pd.DataFrame()

    tr       = pd.DataFrame(results)
    wins     = (tr["net_ret"] > 0).mean() * 100
    avg_ret  = tr["net_ret"].mean()
    final_eq = 100000 * (1 + tr["net_ret"].sum() / 100 * 0.01)

    res = {
        "trades":          len(tr),
        "win_rate_pct":    round(wins, 2),
        "avg_net_ret_pct": round(avg_ret, 4),
        "final_equity":    round(final_eq, 2),
        "return_pct":      round((final_eq / 100000 - 1) * 100, 4),
    }
    return res, tr

log("loading data...")
df = load_all()
fs = feature_cols(df)
log(f"dataset {len(df)} symbols {df['symbol'].nunique()} features {len(fs)}")

all_preds, all_trades, folds = [], [], []

for fold, (train_start, train_end, test_end) in enumerate(FOLDS, 1):
    train = df[(df["date"] >= train_start) & (df["date"] < train_end)].copy()
    test  = df[(df["date"] >= train_end)   & (df["date"] < test_end)].copy()
    if len(train) < 500 or len(test) < 100:
        continue
    if len(train) > MAX_TRAIN:
        train = train.sample(MAX_TRAIN, random_state=42 + fold)

    y = (train["rel_ret"] > 2.0).astype(int)  # must beat Nifty by 2%
    if y.nunique() < 2:
        continue

    model = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.04, max_leaf_nodes=31,
        l2_regularization=0.1, random_state=42 + fold
    )
    model.fit(train[fs].replace([np.inf,-np.inf],np.nan).fillna(0).astype("float32"), y)

    pkl = MODEL_DIR / f"xgb_nse_midcap_v2_fold{fold}.pkl"
    with open(pkl, "wb") as f:
        pickle.dump({"model": model, "features": fs, "fold": fold, "type": "nse_midcap_v2"}, f)
    log(f"saved {pkl}")

    test = test.copy()
    test["prob"] = model.predict_proba(
        test[fs].replace([np.inf,-np.inf],np.nan).fillna(0).astype("float32")
    )[:, 1]

    test_dates = pd.Index(sorted(test["date"].drop_duplicates()))
    res, tr = simulate(test, test_dates)
    log(f"fold_done {fold} {train_end} {test_end}", json.dumps(res))

    all_preds.append(test)
    if len(tr):
        all_trades.append(tr)
    folds.append({"fold": fold, "train_end": train_end, "test_end": test_end, **res})

trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
summary = {
    "model":        "nse_midcap_trend_following_v2",
    "cost_pct":     COST,
    "hold_days":    HOLD,
    "features":     len(fs),
    "feature_list": fs,
    "folds":        folds,
    "avg_win_rate": round(np.mean([f["win_rate_pct"] for f in folds]), 2) if folds else 0,
    "avg_return":   round(np.mean([f["return_pct"]   for f in folds]), 2) if folds else 0,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"summary": summary, "folds": folds}, indent=2, default=str))
if len(trades):
    trades.to_csv(TRADES, index=False)

log("DONE")
print(json.dumps(summary, indent=2))
