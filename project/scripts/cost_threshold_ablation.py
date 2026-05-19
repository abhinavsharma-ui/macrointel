#!/usr/bin/env python3
"""cost_threshold_ablation.py — Walk-forward ablation over COST thresholds."""
from __future__ import annotations
import json, os, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT)

ROOT          = Path(os.getenv("SIM_ROOT", "data/features_26yr_liquid"))
RUNTIME_STATE = Path(os.getenv("SIM_RUNTIME_STATE", "data/runtime_state.json"))
HOLD          = int(os.getenv("SIM_HOLD_DAYS", "10"))
MIN_ADV       = float(os.getenv("SIM_MIN_ADV20_DOLLAR_VOL", "5000000"))
MIN_PRICE     = float(os.getenv("SIM_MIN_ENTRY_PRICE", "5"))
MAX_ABS_RET   = float(os.getenv("SIM_MAX_ABS_HONEST_RET_PCT", "100"))
MAX_TRAIN     = int(os.getenv("SIM_MAX_TRAIN_ROWS", "350000"))
SIG_THRESHOLD = float(os.getenv("SIG_THRESHOLD", "0.60"))
TOP_N         = int(os.getenv("SIG_TOP_N", "10"))
POSITION_PCT  = float(os.getenv("SIG_POSITION_PCT", "0.03"))
PT_PCT        = float(os.getenv("SIG_PROFIT_TARGET_PCT", "8.0"))   # live system exit cap
ROUNDTRIP     = 0.45   # 0.35 commission + 0.10 slippage
COST_GRID     = [float(x) for x in os.getenv("ABLATION_COST_GRID","0.45,1.00,1.50").split(",")]
OUT_JSON      = Path("reports/cost_threshold_ablation.json")

FOLDS = [(2013,2014,2018),(2018,2019,2021),(2021,2022,2023),(2023,2024,2099)]
MODEL_PARAMS  = dict(max_iter=180,learning_rate=0.045,max_leaf_nodes=31,l2_regularization=0.1,random_state=42)

def log(*a): print("ABLATION",*a,flush=True)
def norm_sym(p): return p.stem.replace("_US","").replace(".US","").upper()

def read_price_file(p):
    df = pd.read_parquet(p)
    if df.empty or not {"open","close","volume"}.issubset(df.columns): return None
    idx = pd.to_datetime(df.index,errors="coerce")
    df  = df.loc[:,~df.columns.duplicated()].copy()
    df.index = pd.RangeIndex(len(df))
    if "date" in df.columns:
        dc = pd.to_datetime(df["date"],errors="coerce")
        df["date"] = dc if dc.notna().sum()>=idx.notna().sum() else idx
    else:
        df["date"] = idx
    df = df.dropna(subset=["date","open","close"]).sort_values("date").reset_index(drop=True)
    df["symbol"] = norm_sym(p)
    if "adv20_dollar_vol" not in df.columns:
        df["adv20_dollar_vol"] = (df["close"]*df["volume"]).rolling(20,min_periods=5).mean()
    return df

def feature_cols(df):
    banned = {"date","symbol","year","open","high","low","close","volume","adj_close",
              "entry_open","default_exit_close","effective_exit_close","exit_close",
              "model_ret","honest_ret","adv20_dollar_vol","pt_hit","pt_day",
              "exit_reason","exit_offset","spy_realized_vol_pct"}
    out = []
    for c in df.columns:
        cl = c.lower()
        if (c in banned or cl.startswith("gross_ret_") or cl.startswith("exit_timestamp")
                or cl.startswith("future") or cl.startswith("next")
                or "target" in cl or "label" in cl): continue
        if pd.api.types.is_numeric_dtype(df[c]): out.append(c)
    pref = ["return_20d","return_60d","momentum_composite","vol_regime_ratio","atr_pct",
            "price_acceleration","close_vs_sma_50","close_vs_sma_200","rsi_14"]
    return [c for c in pref if c in out]+[c for c in out if c not in pref]

def compute_metrics(trades: pd.DataFrame, position_pct: float, hold: int) -> dict:
    """Daily portfolio sim — concurrent positions, each sized at position_pct."""
    if trades.empty:
        return {"n_trades":0,"win_rate":0,"avg_ret":0,"total_ret":0,"max_dd":0,"calmar":0,"cagr":0}
    t = trades.sort_values("date").copy()
    t["exit_date"] = t["date"] + pd.tseries.offsets.BDay(hold)
    all_days = pd.bdate_range(t["date"].min(), t["exit_date"].max())
    equity = 1.0; peak = 1.0; max_dd = 0.0

    for day in all_days:
        open_trades = t[(t["date"]<=day)&(t["exit_date"]>day)]
        if not open_trades.empty:
            # spread each trade's capped net return evenly over hold days
            daily = (open_trades["net_ret"] / hold / 100 * position_pct).sum()
            equity *= (1 + daily)
        if equity > peak: peak = equity
        dd = (peak-equity)/peak*100
        if dd > max_dd: max_dd = dd

    total_ret = (equity-1)*100
    years = max((all_days[-1]-all_days[0]).days/365.25, 0.1)
    cagr  = ((equity)**(1/years)-1)*100
    n = len(t); wins = (t["net_ret"]>0).sum()
    return {"n_trades":n,"win_rate":round(float(wins/n*100),2),
            "avg_ret":round(float(t["net_ret"].mean()),4),
            "total_ret":round(total_ret,2),"max_dd":round(max_dd,2),
            "cagr":round(cagr,2),"calmar":round(cagr/max_dd,3) if max_dd>0 else 0}

# ── load ──────────────────────────────────────────────────────────────────────
log("loading universe...")
if RUNTIME_STATE.exists():
    rs = json.load(open(RUNTIME_STATE))
    ALLOWED = {str(k).upper() for k in (rs.get("signal_store") or {}).keys()} or None
else:
    ALLOWED = None
log("allowed:", "all" if ALLOWED is None else len(ALLOWED))

rows = []
files = sorted(ROOT.glob("*.parquet"))
for i,p in enumerate(files,1):
    sym = norm_sym(p)
    if ALLOWED is not None and sym not in ALLOWED: continue
    raw = read_price_file(p)
    if raw is None: continue
    x = raw[raw["adv20_dollar_vol"].fillna(0)>=MIN_ADV].copy()
    x = x[x["close"].fillna(0)>=MIN_PRICE].copy()
    if x.empty: continue
    x["entry_open"]         = x["open"].shift(-1)
    x["default_exit_close"] = x["close"].shift(-(HOLD+1))
    x["model_ret"]          = ((x["default_exit_close"]/x["entry_open"])-1)*100
    x = x.replace([np.inf,-np.inf],np.nan).dropna(subset=["model_ret"])
    x = x[x["model_ret"].abs()<=MAX_ABS_RET]
    if not x.empty: rows.append(x)
    if i%100==0: log(f"  {i}/{len(files)} files, {sum(len(r) for r in rows):,} rows")

if not rows: log("no data"); sys.exit(2)
DF = pd.concat(rows,ignore_index=True)
DF["date"] = pd.to_datetime(DF["date"])
DF = DF.sort_values(["date","symbol"]).reset_index(drop=True)
today   = pd.Timestamp.today().normalize()
embargo = today - pd.tseries.offsets.BDay(HOLD+1)
DF      = DF[DF["date"]<=embargo].reset_index(drop=True)
FS      = feature_cols(DF)
log(f"dataset: {len(DF):,} rows, {DF['symbol'].nunique()} syms, {len(FS)} features")

# ── ablation ──────────────────────────────────────────────────────────────────
results = {}
for cost in COST_GRID:
    log(f"\n{'='*55}\nCOST = {cost}%  (PT cap = {PT_PCT}%)\n{'='*55}")
    all_trades = []
    for fold_i,(train_end_yr,test_start_yr,test_end_yr) in enumerate(FOLDS):
        tr = DF[DF["date"].dt.year<=train_end_yr].copy()
        te = DF[(DF["date"].dt.year>=test_start_yr)&(DF["date"].dt.year<=test_end_yr)].copy()
        if len(tr)<1000 or len(te)<100: continue
        if len(tr)>MAX_TRAIN: tr=tr.sample(MAX_TRAIN,random_state=42).reset_index(drop=True)
        y = (tr["model_ret"]>cost).astype(int)
        if y.sum()<50: log(f"  fold {fold_i}: too few positives, skip"); continue
        log(f"  fold {fold_i}: train={len(tr):,} pos={float(y.mean()):.3f}  test={len(te):,}  {test_start_yr}–{min(test_end_yr,2026)}")
        X_tr = tr[FS].replace([np.inf,-np.inf],np.nan).fillna(0).astype("float32")
        mdl  = HistGradientBoostingClassifier(**MODEL_PARAMS)
        mdl.fit(X_tr,y)
        X_te = te[FS].replace([np.inf,-np.inf],np.nan).fillna(0).astype("float32")
        te   = te.copy(); te["prob"] = mdl.predict_proba(X_te)[:,1]
        above = te[te["prob"]>=SIG_THRESHOLD].copy()
        if above.empty: log(f"  fold {fold_i}: 0 signals above threshold"); continue
        above["rank"] = above.groupby("date")["prob"].rank(method="first",ascending=False)
        sigs = above[above["rank"]<=TOP_N].copy()
        # *** KEY FIX: cap upside at profit target, same as live system ***
        sigs["net_ret"] = sigs["model_ret"].clip(upper=PT_PCT) - ROUNDTRIP
        log(f"  fold {fold_i}: {len(sigs):,} signals  WR={(sigs['net_ret']>0).mean()*100:.1f}%  avg={sigs['net_ret'].mean():.3f}%")
        all_trades.append(sigs[["date","symbol","prob","model_ret","net_ret"]])

    if all_trades:
        combined = pd.concat(all_trades,ignore_index=True)
        m = compute_metrics(combined,POSITION_PCT,HOLD)
        results[cost] = m
        log(f"  RESULT: {m['n_trades']:,} trades | WR={m['win_rate']}% | avg={m['avg_ret']:.3f}% | "
            f"TotalRet={m['total_ret']:.1f}% | CAGR={m['cagr']:.1f}% | MaxDD={m['max_dd']:.1f}% | Calmar={m['calmar']}")
    else:
        results[cost] = {"n_trades":0}

log(f"\n{'='*65}")
log(f"{'COST':>6}  {'Trades':>7}  {'WinRate':>8}  {'AvgRet':>7}  {'TotalRet':>9}  {'CAGR':>6}  {'MaxDD':>6}  {'Calmar':>7}")
log(f"{'-'*65}")
for cost,m in sorted(results.items()):
    if m.get("n_trades",0)==0:
        log(f"{cost:>6.2f}%  no signals")
    else:
        log(f"{cost:>6.2f}%  {m['n_trades']:>7,}  {m['win_rate']:>7.1f}%  {m['avg_ret']:>7.3f}%  "
            f"{m['total_ret']:>8.1f}%  {m['cagr']:>5.1f}%  {m['max_dd']:>5.1f}%  {m['calmar']:>7.3f}")
log("="*65)

OUT_JSON.parent.mkdir(parents=True,exist_ok=True)
OUT_JSON.write_text(json.dumps({str(k):v for k,v in results.items()},indent=2,default=str))
log("saved",str(OUT_JSON))
log("DONE")
