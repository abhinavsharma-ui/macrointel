import json, os, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

DATA = Path("reports/ohlcv_pure_audit_dataset.parquet")
OUT = Path(os.getenv("HONEST_OPT_OUT", "reports/ohlcv_honest_policy_optimizer.json"))
SEED = int(os.getenv("HONEST_SEED", "42"))
INIT_EQUITY = float(os.getenv("HONEST_INIT_EQUITY", "100000"))
POSITION_PCTS = [float(x) for x in os.getenv("HONEST_POSITION_PCTS", "0.01,0.02,0.03,0.04").split(",")]
MAX_NEW_GRID = [int(x) for x in os.getenv("HONEST_MAX_NEW_GRID", "1,2,3,5,8,12,20,30").split(",")]
PROB_GRID = [float(x) for x in os.getenv("HONEST_PROB_GRID", "0.55,0.58,0.60,0.62,0.65,0.68,0.70,0.72,0.75").split(",")]
RULE_Q_GRID = [float(x) for x in os.getenv("HONEST_RULE_Q_GRID", "0.90,0.93,0.95,0.97,0.98").split(",")]
MIN_ADV = float(os.getenv("HONEST_MIN_ADV20_DOLLAR_VOL", "5000000"))
COST_PCT = float(os.getenv("HONEST_TOTAL_COST_PCT", "0.45"))
MAX_GROSS = float(os.getenv("HONEST_MAX_GROSS_EXPOSURE_PCT", "1.0"))
HORIZON_DAYS = int(os.getenv("HONEST_HORIZON_DAYS", "10"))
MAX_TRAIN_ROWS = int(os.getenv("HONEST_MAX_TRAIN_ROWS", "200000"))
MAX_RET_COLS = int(os.getenv("HONEST_MAX_RET_COLS", "3"))
FOLDS = int(os.getenv("HONEST_FOLDS", "4"))

ETF_CORE = set("""
SPY QQQ DIA IWM VTI VOO IVV RSP VONE VONG VONV IUSV IUSG ESGU DGRW AVUS BBUS
SCHB SCHG SCHV XLK XLF XLY XLI XLC XLV XLE XLP XLU XLB IYW IWF IWD VUG VTV
""".split())

def log(*x):
    print("HONEST_OPT", *x, flush=True)

def load_data():
    if not DATA.exists():
        raise SystemExit(f"missing {DATA}; run pure audit cache first")
    df = pd.read_parquet(DATA)

    if "date" not in df.columns:
        for c in ["__index_level_0__", "index", "level_0", "timestamp", "datetime"]:
            if c in df.columns:
                df = df.rename(columns={c: "date"})
                break

    if "date" not in df.columns:
        idx_name = df.index.name
        if idx_name is not None and "date" in str(idx_name).lower():
            df = df.reset_index().rename(columns={idx_name: "date"})
        else:
            df = df.reset_index()
            for c in df.columns:
                if "date" in str(c).lower() or str(c).lower() in ["index", "level_0"]:
                    df = df.rename(columns={c: "date"})
                    break

    if "date" not in df.columns:
        raise SystemExit("could not find date column after reset_index; columns=" + repr(list(df.columns)[:40]))

    df["date"] = pd.to_datetime(df["date"])
    if "year" not in df.columns:
        df["year"] = df["date"].dt.year
    if "adv20_dollar_vol" in df.columns:
        before = len(df)
        df = df[df["adv20_dollar_vol"].fillna(0) >= MIN_ADV].copy()
        log("liquidity_filter", before, "after", len(df), "symbols", df["symbol"].nunique())
    return df.sort_values(["date","symbol"]).reset_index(drop=True)


def ret_cols(df):
    cols = []
    for c in df.columns:
        if c.startswith("gross_ret_") or re.match(r"^ret_sl.*tp", c):
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
    preferred = [c for c in cols if "sl3p0" in c and "tp5p0" in c]
    return preferred + [c for c in cols if c not in preferred]

def feature_cols(df):
    bad = ("ret_", "gross_ret_", "future", "label", "target", "take_", "entry", "exit")
    banned = {"date","symbol","year","open","high","low","close","volume","regime_allowed"}
    out = []
    for c in df.columns:
        if c in banned or any(c.startswith(x) for x in bad):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    must = ["return_20d","return_60d","momentum_composite","vol_regime_ratio","atr_pct",
            "price_acceleration","close_vs_sma_50","close_vs_sma_200","rsi_14"]
    return [c for c in must if c in out] + [c for c in out if c not in must]

def add_rule_score(df):
    pieces = []
    weights = {
        "return_60d": 0.34, "return_20d": 0.24, "momentum_composite": 0.18,
        "close_vs_sma_200": 0.14, "close_vs_sma_50": 0.08, "atr_pct": -0.08,
        "vol_regime_ratio": -0.04,
    }
    s = pd.Series(0.0, index=df.index)
    for c,w in weights.items():
        if c in df.columns:
            r = df.groupby("date")[c].rank(pct=True)
            s = s + w * r.fillna(0.5)
    df = df.copy()
    df["rule_score"] = s
    return df

def market_regimes(df):
    by = {}
    for sym in ["SPY","QQQ"]:
        x = df[df["symbol"].eq(sym)].drop_duplicates("date").set_index("date")
        if len(x):
            by[sym] = x
    modes = ["none"]
    if "SPY" in by:
        modes += ["spy_sma200", "spy_sma200_ret20"]
    if "QQQ" in by:
        modes += ["qqq_sma200", "qqq_sma200_ret20"]
    return by, modes

def regime_mask(x, mode, markets):
    if mode == "none":
        return np.ones(len(x), dtype=bool)
    sym = "SPY" if mode.startswith("spy") else "QQQ"
    m = markets.get(sym)
    if m is None:
        return np.zeros(len(x), dtype=bool)
    ok = pd.Series(True, index=m.index)
    if "sma200" in mode:
        ok &= m.get("close_vs_sma_200", pd.Series(0, index=m.index)).fillna(0) > 0
    if "ret20" in mode:
        ok &= m.get("return_20d", pd.Series(0, index=m.index)).fillna(0) > 0
    return x["date"].map(ok).fillna(False).to_numpy(dtype=bool)

def universe_mask(x, mode):
    if mode == "all":
        return np.ones(len(x), dtype=bool)
    if mode == "etf_core":
        return x["symbol"].isin(ETF_CORE).to_numpy(dtype=bool)
    if mode == "top_liquid":
        if "adv20_dollar_vol" not in x.columns:
            return np.ones(len(x), dtype=bool)
        cut = x["adv20_dollar_vol"].quantile(0.75)
        return (x["adv20_dollar_vol"] >= cut).to_numpy(dtype=bool)
    return np.ones(len(x), dtype=bool)

def splits(df):
    dates = np.array(sorted(df["date"].drop_duplicates()))
    first_test = int(len(dates) * 0.35)
    chunks = np.array_split(dates[first_test:], FOLDS)
    out = []
    for i,ch in enumerate(chunks, 1):
        if len(ch) < 50:
            continue
        test_start = np.searchsorted(dates, ch[0])
        test_end = np.searchsorted(dates, ch[-1], side="right")
        val_len = max(252, min(756, len(ch)//2))
        val_end = max(0, test_start - HORIZON_DAYS)
        val_start = max(0, val_end - val_len)
        train_end = max(0, val_start - HORIZON_DAYS)
        if train_end < 504:
            continue
        out.append({
            "fold": i,
            "train_dates": set(dates[:train_end]),
            "val_dates": set(dates[val_start:val_end]),
            "test_dates": set(dates[test_start:test_end]),
            "test_start": str(ch[0].date()),
            "test_end": str(ch[-1].date()),
        })
    return out

def simulate(trades, ret_col, pos_pct, max_new):
    if trades.empty:
        return {"trades":0,"executed_trades":0,"skipped_capacity":0,"final_equity":INIT_EQUITY,
                "return_pct":0.0,"max_drawdown_pct":0.0,"avg_trade_net_return_pct":0.0,"win_rate_pct":0.0}
    t = trades.sort_values(["date","score"], ascending=[True,False]).copy()
    t = t.groupby("date", group_keys=False).head(max_new).copy()
    all_dates = np.array(sorted(t["date"].drop_duplicates()))
    date_pos = {d:i for i,d in enumerate(all_dates)}
    active, equity, peak, maxdd = [], INIT_EQUITY, INIT_EQUITY, 0.0
    execs, skips, net_rets = 0, 0, []
    for d, day in t.groupby("date", sort=True):
        idx = date_pos[d]
        still = []
        for exit_idx, pnl, notional in active:
            if exit_idx <= idx:
                equity += pnl
            else:
                still.append((exit_idx, pnl, notional))
        active = still
        peak = max(peak, equity)
        if peak > 0:
            maxdd = max(maxdd, (peak - equity) / peak * 100)
        exposure = sum(a[2] for a in active) / max(equity, 1)
        for _, r in day.iterrows():
            if exposure + pos_pct > MAX_GROSS + 1e-12:
                skips += 1
                continue
            net = float(r[ret_col]) - COST_PCT
            notional = equity * pos_pct
            pnl = notional * net / 100.0
            active.append((idx + HORIZON_DAYS, pnl, notional))
            exposure += pos_pct
            execs += 1
            net_rets.append(net)
    for _, pnl, _ in active:
        equity += pnl
    peak = max(peak, equity)
    if peak > 0:
        maxdd = max(maxdd, (peak - equity) / peak * 100)
    arr = np.array(net_rets, dtype=float)
    return {
        "trades": int(len(t)), "executed_trades": int(execs), "skipped_capacity": int(skips),
        "final_equity": round(float(equity),2),
        "return_pct": round((float(equity)/INIT_EQUITY - 1)*100,4),
        "max_drawdown_pct": round(float(maxdd),4),
        "avg_trade_net_return_pct": round(float(arr.mean()) if len(arr) else 0,4),
        "win_rate_pct": round(float((arr > 0).mean()*100) if len(arr) else 0,2),
    }

def choose_policy(scored, ret_col, markets):
    best = None
    universes = ["all","etf_core","top_liquid"]
    regimes = market_regimes(scored)[1]
    for engine in ["ml","rule"]:
        score_col = "prob" if engine == "ml" else "rule_score"
        grid = PROB_GRID if engine == "ml" else RULE_Q_GRID
        for universe in universes:
            um = universe_mask(scored, universe)
            for regime in regimes:
                rm = regime_mask(scored, regime, markets)
                base = scored[um & rm & scored[ret_col].notna()].copy()
                if len(base) < 50:
                    continue
                for th in grid:
                    if engine == "ml":
                        cand = base[base[score_col] >= th].copy()
                    else:
                        cand = base[base[score_col] >= base[score_col].quantile(th)].copy()
                    if len(cand) < 20:
                        continue
                    cand["score"] = cand[score_col]
                    for max_new in MAX_NEW_GRID:
                        for pos in POSITION_PCTS:
                            res = simulate(cand, ret_col, pos, max_new)
                            obj = res["return_pct"] - 1.5 * max(0, res["max_drawdown_pct"] - 8) + 0.05 * res["avg_trade_net_return_pct"]
                            if res["executed_trades"] < 80:
                                obj -= 25
                            row = {"engine":engine,"ret_col":ret_col,"threshold":th,"universe":universe,
                                   "regime":regime,"max_new_per_day":max_new,"position_pct":pos,
                                   "validation":res,"objective":round(float(obj),4)}
                            if best is None or row["objective"] > best["objective"]:
                                best = row
    return best

def apply_policy(scored, policy, markets):
    score_col = "prob" if policy["engine"] == "ml" else "rule_score"
    x = scored[
        universe_mask(scored, policy["universe"]) &
        regime_mask(scored, policy["regime"], markets) &
        scored[policy["ret_col"]].notna()
    ].copy()
    if policy["engine"] == "ml":
        x = x[x[score_col] >= policy["threshold"]].copy()
    else:
        x = x[x[score_col] >= x[score_col].quantile(policy["threshold"])].copy()
    x["score"] = x[score_col]
    return x

def train_score(train, val, test, feats, ret_col, fold):
    y = (train[ret_col].fillna(-999) > 0).astype(int)
    if y.nunique() < 2:
        return None, None, None
    rng = np.random.default_rng(SEED + fold)
    idx = np.arange(len(train))
    if len(idx) > MAX_TRAIN_ROWS:
        idx = rng.choice(idx, size=MAX_TRAIN_ROWS, replace=False)
    X = train.iloc[idx][feats].replace([np.inf,-np.inf], np.nan).fillna(0).astype("float32")
    yy = y.iloc[idx]
    model = HistGradientBoostingClassifier(
        max_iter=140, learning_rate=0.055, max_leaf_nodes=31,
        l2_regularization=0.08, random_state=SEED + fold
    )
    model.fit(X, yy)
    def score(x):
        z = x.copy()
        z["prob"] = model.predict_proba(z[feats].replace([np.inf,-np.inf], np.nan).fillna(0).astype("float32"))[:,1]
        return z
    return score(val), score(test), model

def combine(results):
    trades = pd.concat([r["test_trades"] for r in results if len(r["test_trades"])], ignore_index=True) if results else pd.DataFrame()
    if trades.empty:
        return {}
    first = results[0]["policy"]
    ret_col = first["ret_col"]
    pos = first["position_pct"]
    mx = first["max_new_per_day"]
    return simulate(trades, ret_col, pos, mx)


def simulate_variable(trades):
    if trades.empty:
        return {"trades":0,"executed_trades":0,"skipped_capacity":0,"final_equity":INIT_EQUITY,
                "return_pct":0.0,"max_drawdown_pct":0.0,"avg_trade_net_return_pct":0.0,"win_rate_pct":0.0}
    t = trades.sort_values(["date","score"], ascending=[True,False]).copy()
    all_dates = list(sorted(t["date"].drop_duplicates()))
    date_pos = {d:i for i,d in enumerate(all_dates)}
    active, equity, peak, maxdd = [], INIT_EQUITY, INIT_EQUITY, 0.0
    execs, skips, net_rets = 0, 0, []
    for d, day in t.groupby("date", sort=True):
        idx = date_pos[d]
        still = []
        for exit_idx, pnl, notional in active:
            if exit_idx <= idx:
                equity += pnl
            else:
                still.append((exit_idx, pnl, notional))
        active = still
        peak = max(peak, equity)
        if peak > 0:
            maxdd = max(maxdd, (peak - equity) / peak * 100)
        day_max = int(day["max_new"].iloc[0]) if "max_new" in day.columns else 1
        day = day.head(day_max)
        exposure = sum(a[2] for a in active) / max(equity, 1)
        for _, r in day.iterrows():
            pos_pct = float(r.get("position_pct", 0.01))
            if exposure + pos_pct > MAX_GROSS + 1e-12:
                skips += 1
                continue
            net = float(r["chosen_ret"]) - COST_PCT
            notional = equity * pos_pct
            pnl = notional * net / 100.0
            active.append((idx + HORIZON_DAYS, pnl, notional))
            exposure += pos_pct
            execs += 1
            net_rets.append(net)
    for _, pnl, _ in active:
        equity += pnl
    arr = __import__("numpy").array(net_rets, dtype=float)
    return {
        "trades": int(len(t)), "executed_trades": int(execs), "skipped_capacity": int(skips),
        "final_equity": round(float(equity),2),
        "return_pct": round((float(equity)/INIT_EQUITY - 1)*100,4),
        "max_drawdown_pct": round(float(maxdd),4),
        "avg_trade_net_return_pct": round(float(arr.mean()) if len(arr) else 0,4),
        "win_rate_pct": round(float((arr > 0).mean()*100) if len(arr) else 0,2),
    }


def main():
    df = add_rule_score(load_data())
    rcols = ret_cols(df)[:MAX_RET_COLS]
    feats = feature_cols(df)
    markets, _ = market_regimes(df)
    log("rows", len(df), "symbols", df.symbol.nunique(), "ret_cols", rcols, "features", len(feats))
    fold_rows = []
    all_test_trades = []
    for sp in splits(df):
        fold = sp["fold"]
        train = df[df["date"].isin(sp["train_dates"])].copy()
        val = df[df["date"].isin(sp["val_dates"])].copy()
        test = df[df["date"].isin(sp["test_dates"])].copy()
        best_fold = None
        log("fold_start", fold, sp["test_start"], sp["test_end"], "train", len(train), "val", len(val), "test", len(test))
        for rc in rcols:
            log("fold_retcol_start", fold, rc)
            scored_val, scored_test, model = train_score(train, val, test, feats, rc, fold)
            if scored_val is None:
                continue
            pol = choose_policy(scored_val, rc, markets)
            if pol:
                log("fold_retcol_policy", fold, rc, json.dumps({"objective": pol.get("objective"), "engine": pol.get("engine"), "threshold": pol.get("threshold"), "universe": pol.get("universe"), "regime": pol.get("regime"), "max_new": pol.get("max_new_per_day"), "pos": pol.get("position_pct")}))
            if not pol:
                continue
            test_trades = apply_policy(scored_test, pol, markets)
            test_res = simulate(test_trades, rc, pol["position_pct"], pol["max_new_per_day"])
            row = {"fold":fold,"test_start":sp["test_start"],"test_end":sp["test_end"],
                   "policy":pol,"test":test_res}
            select_score = float(pol.get("objective", -1e9))
            if best_fold is None or select_score > best_fold["_score"]:
                row["_score"] = select_score
                tt = test_trades.copy()
                tt["chosen_ret"] = tt[rc].astype(float)
                tt["ret_col_used"] = rc
                tt["position_pct"] = pol["position_pct"]
                tt["max_new"] = pol["max_new_per_day"]
                row["test_trades"] = tt
                best_fold = row
        if best_fold:
            log("fold_done", json.dumps({k:v for k,v in best_fold.items() if k not in ["test_trades","_score"]})[:2500])
            fold_rows.append({k:v for k,v in best_fold.items() if k not in ["test_trades","_score"]})
            all_test_trades.append(best_fold["test_trades"])
        else:
            log("fold_no_policy", fold)
    trades = pd.concat(all_test_trades, ignore_index=True) if all_test_trades else pd.DataFrame()
    if not trades.empty:
        trades["score"] = trades["score"].astype(float)
        baseline = simulate_variable(trades)
        odd = simulate_variable(trades[trades["date"].dt.year % 2 == 1])
        rb = trades.assign(net_for_rank=trades["chosen_ret"] - COST_PCT).sort_values("net_for_rank", ascending=False).iloc[50:].copy()
        remove_best_50 = simulate_variable(rb)
    else:
        baseline = odd = remove_best_50 = {}
    out = {"summary":baseline,"folds":fold_rows,"stress":{"odd_years_only":odd,"remove_best_50":remove_best_50}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    log("wrote", OUT)
    print(json.dumps(out["summary"], indent=2))

if __name__ == "__main__":
    main()
