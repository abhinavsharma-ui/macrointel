#!/usr/bin/env python3
"""
Feature Winner Analysis
=======================
Finds which input features at entry time best separate WINNERS from
TIME-EXIT LOSERS within the high-confidence (0.83-0.86) probability cluster.

Run from:  ~/macro_intelligence_complete/project/
Usage:     python scripts/feature_winner_analysis.py

Outputs:
  reports/feature_winner_analysis.json   -- machine-readable results
  reports/feature_winner_analysis.txt    -- human-readable summary
"""

import os, sys, json, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

TRADES_CSV      = Path("reports/fixed_return_paper_trades.csv")
SIM_RESULTS_DIR = Path("reports")
DATA_DIR        = Path("data/processed")
SPY_PARQUET     = Path("data/processed/SPY.parquet")

PROB_LOW  = float(os.getenv("ANALYSIS_PROB_LOW",  "0.60"))
PROB_HIGH = float(os.getenv("ANALYSIS_PROB_HIGH", "1.00"))
LOOKBACK_DAYS = int(os.getenv("ANALYSIS_LOOKBACK_DAYS", "20"))
MIN_TRADES = int(os.getenv("ANALYSIS_MIN_TRADES", "30"))

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def load_trades() -> pd.DataFrame:
    frames = []
    if TRADES_CSV.exists():
        df = pd.read_csv(TRADES_CSV, parse_dates=True)
        df.columns = df.columns.str.lower().str.strip()
        rename = {
            "ret_pct": "net_return_pct", "return_pct": "net_return_pct",
            "return": "net_return_pct", "net_ret": "net_return_pct",
            "prob": "probability", "pred_prob": "probability",
            "signal_prob": "probability", "entry": "entry_date",
            "exit": "exit_date", "ticker": "symbol",
        }
        df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)
        for col in ["entry_date", "exit_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        frames.append(("paper", df))
        log(f"Paper trades: {len(df)} rows, cols={list(df.columns)}")

    wf_files = sorted(SIM_RESULTS_DIR.glob("wf_*.json"))
    for fp in wf_files:
        try:
            blob = json.loads(fp.read_text())
            trades_list = blob.get("trades", [])
            if not trades_list:
                continue
            df = pd.DataFrame(trades_list)
            df.columns = df.columns.str.lower().str.strip()
            rename2 = {
                "ret": "net_return_pct", "return": "net_return_pct",
                "prob": "probability", "pred": "probability",
                "entry": "entry_date", "exit": "exit_date",
            }
            df.rename(columns={k: v for k, v in rename2.items() if k in df.columns}, inplace=True)
            for col in ["entry_date", "exit_date"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            frames.append((fp.stem, df))
            log(f"  {fp.name}: {len(df)} trades")
        except Exception as e:
            log(f"  skip {fp.name}: {e}")

    if not frames:
        log("ERROR: No trade data found.")
        sys.exit(1)

    frames.sort(key=lambda x: len(x[1]), reverse=True)
    source, df = frames[0]
    log(f"Using '{source}' as primary trade source ({len(df)} trades)")

    needed = {"symbol", "entry_date", "exit_date", "net_return_pct", "probability"}
    missing = needed - set(df.columns)
    if missing:
        log(f"ERROR: Missing columns {missing}. Available: {list(df.columns)}")
        sys.exit(1)

    df = df.dropna(subset=list(needed))
    return df

def label_trades(df: pd.DataFrame) -> pd.DataFrame:
    PT_NET_THRESHOLD = float(os.getenv("ANALYSIS_PT_NET_THRESHOLD", "6.0"))
    if "exit_reason" in df.columns:
        reason = df["exit_reason"].astype(str).str.lower()
        df["winner"] = reason.str.contains("profit|target|pt|win", na=False).astype(int)
        log(f"Labels from exit_reason: {df['winner'].sum()} winners / {(df['winner']==0).sum()} losers")
    else:
        df["winner"] = (df["net_return_pct"] >= PT_NET_THRESHOLD).astype(int)
        log(f"Labels inferred from return >= {PT_NET_THRESHOLD}%: "
            f"{df['winner'].sum()} winners / {(df['winner']==0).sum()} losers")
    return df

def filter_cluster(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df["probability"] >= PROB_LOW) & (df["probability"] <= PROB_HIGH)
    cluster = df[mask].copy()
    log(f"Cluster [{PROB_LOW}, {PROB_HIGH}]: {len(cluster)} trades "
        f"({cluster['winner'].mean()*100:.1f}% win rate)")
    if len(cluster) < MIN_TRADES:
        log(f"ERROR: Only {len(cluster)} trades in cluster -- need {MIN_TRADES}.")
        sys.exit(1)
    return cluster

def load_price(symbol: str):
    for candidate in [
        DATA_DIR / f"{symbol}.parquet",
        DATA_DIR / f"{symbol}_US.parquet",
        DATA_DIR / f"{symbol.lower()}.parquet",
    ]:
        if candidate.exists():
            df = pd.read_parquet(candidate)
            df.columns = df.columns.str.lower()
            if "date" not in df.columns and df.index.name == "date":
                df = df.reset_index()
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            return df
    return None

def build_entry_features(symbol, entry_date, price_cache, spy_df):
    if symbol not in price_cache:
        price_cache[symbol] = load_price(symbol)
    df = price_cache[symbol]
    if df is None:
        return None

    rows = df[df["date"] <= entry_date]
    if len(rows) < LOOKBACK_DAYS + 5:
        return None
    hist  = rows.tail(LOOKBACK_DAYS + 5)
    close = hist["close"].values
    vol   = hist["volume"].values if "volume" in hist.columns else None

    feats = {}

    for n in [5, 10, 20]:
        feats[f"ret_{n}d"] = (close[-1] / close[-1-n] - 1) * 100 if len(close) > n else float("nan")

    rets = np.diff(np.log(close[-21:]))
    feats["vol_20d"] = float(np.std(rets) * np.sqrt(252) * 100) if len(rets) >= 10 else float("nan")

    if len(close) >= 16:
        deltas = np.diff(close[-15:])
        gains  = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_g  = gains.mean() + 1e-9
        avg_l  = losses.mean() + 1e-9
        feats["rsi_14"] = 100 - 100 / (1 + avg_g / avg_l)
    else:
        feats["rsi_14"] = float("nan")

    window_252 = hist.tail(252)["close"].values if len(hist) >= 252 else close
    feats["dist_52w_high"] = (close[-1] / window_252.max() - 1) * 100
    feats["dist_52w_low"]  = (close[-1] / window_252.min() - 1) * 100

    for n in [20, 50]:
        w = close[-n:] if len(close) >= n else close
        feats[f"dist_sma{n}"] = (close[-1] / w.mean() - 1) * 100

    if vol is not None and len(vol) >= 5:
        avg_vol = vol[-20:].mean() + 1
        feats["vol_ratio"] = vol[-1] / avg_vol
    else:
        feats["vol_ratio"] = float("nan")

    if all(c in hist.columns for c in ["high", "low"]):
        atr_vals = []
        h = hist["high"].values[-15:]
        l = hist["low"].values[-15:]
        c2 = hist["close"].values[-15:]
        for i in range(1, len(h)):
            atr_vals.append(max(h[i]-l[i], abs(h[i]-c2[i-1]), abs(l[i]-c2[i-1])))
        feats["atr_14d_pct"] = (np.mean(atr_vals) / close[-1] * 100) if atr_vals else float("nan")
    else:
        feats["atr_14d_pct"] = float("nan")

    if spy_df is not None:
        spy_rows = spy_df[spy_df["date"] <= entry_date]
        if len(spy_rows) >= 21:
            sc = spy_rows["close"].values
            feats["spy_ret_5d"]      = (sc[-1] / sc[-6]  - 1) * 100 if len(sc) >= 6  else float("nan")
            feats["spy_ret_20d"]     = (sc[-1] / sc[-21] - 1) * 100 if len(sc) >= 21 else float("nan")
            feats["spy_above_sma20"] = float(sc[-1] >= sc[-20:].mean())
            sr = np.diff(np.log(sc[-21:]))
            feats["spy_vol_20d"]     = float(np.std(sr) * np.sqrt(252) * 100) if len(sr) >= 10 else float("nan")
        else:
            for k in ["spy_ret_5d", "spy_ret_20d", "spy_above_sma20", "spy_vol_20d"]:
                feats[k] = float("nan")
    else:
        for k in ["spy_ret_5d", "spy_ret_20d", "spy_above_sma20", "spy_vol_20d"]:
            feats[k] = float("nan")

    feats["entry_month"] = float(entry_date.month)
    feats["entry_dow"]   = float(entry_date.dayofweek)
    return feats

def build_feature_matrix(cluster):
    log("Loading price data and engineering features...")
    spy_df = None
    if SPY_PARQUET.exists():
        spy_df = pd.read_parquet(SPY_PARQUET)
        spy_df.columns = spy_df.columns.str.lower()
        if "date" not in spy_df.columns and spy_df.index.name == "date":
            spy_df = spy_df.reset_index()
        spy_df["date"] = pd.to_datetime(spy_df["date"])
        spy_df = spy_df.sort_values("date").reset_index(drop=True)
        log(f"SPY loaded: {len(spy_df)} rows")

    price_cache, rows, skipped = {}, [], 0
    for _, trade in cluster.iterrows():
        feats = build_entry_features(trade["symbol"], trade["entry_date"], price_cache, spy_df)
        if feats is None:
            skipped += 1
            continue
        feats["probability"]    = trade["probability"]
        feats["hold_days"]      = (trade["exit_date"] - trade["entry_date"]).days \
                                  if pd.notna(trade.get("exit_date")) else float("nan")
        feats["net_return_pct"] = trade["net_return_pct"]
        feats["winner"]         = trade["winner"]
        rows.append(feats)

    log(f"Features built: {len(rows)} trades ({skipped} skipped)")
    df = pd.DataFrame(rows)
    y  = df.pop("winner").astype(int)
    df.drop(columns=["net_return_pct"], inplace=True, errors="ignore")

    null_frac = df.isnull().mean()
    drop_cols = null_frac[null_frac > 0.4].index.tolist()
    if drop_cols:
        log(f"Dropping high-NaN columns: {drop_cols}")
        df.drop(columns=drop_cols, inplace=True)

    df.fillna(df.median(numeric_only=True), inplace=True)
    log(f"Feature matrix: {df.shape[0]} rows x {df.shape[1]} features")
    return df, y

def run_analysis(X, y):
    from scipy import stats as scipy_stats
    results = {}
    feature_names = list(X.columns)
    n_pos, n_neg = int(y.sum()), int((y==0).sum())
    log(f"Analysis: {n_pos} winners, {n_neg} losers ({n_pos/(n_pos+n_neg)*100:.1f}% win rate)")

    log("Running univariate statistics...")
    uni_stats = []
    for col in feature_names:
        winners = X.loc[y==1, col].dropna()
        losers  = X.loc[y==0, col].dropna()
        if len(winners) < 3 or len(losers) < 3:
            continue
        stat, pval = scipy_stats.mannwhitneyu(winners, losers, alternative="two-sided")
        effect = (winners.mean() - losers.mean()) / (X[col].std() + 1e-9)
        uni_stats.append({
            "feature": col,
            "winner_mean": round(float(winners.mean()), 4),
            "loser_mean":  round(float(losers.mean()),  4),
            "diff":        round(float(winners.mean() - losers.mean()), 4),
            "effect_size": round(float(effect), 4),
            "mw_pvalue":   round(float(pval), 6),
            "significant": pval < 0.05,
        })
    uni_stats.sort(key=lambda x: abs(x["effect_size"]), reverse=True)
    results["univariate"] = uni_stats

    log("Fitting GradientBoostingClassifier...")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    gbt = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                     subsample=0.8, min_samples_leaf=5, random_state=42)
    cv = StratifiedKFold(n_splits=min(5, max(2, n_pos//5)), shuffle=True, random_state=42)
    try:
        cv_aucs = cross_val_score(gbt, Xs, y, cv=cv, scoring="roc_auc")
        log(f"  GBT CV AUC: {cv_aucs.mean():.3f} +/- {cv_aucs.std():.3f}")
    except Exception as e:
        log(f"  CV failed: {e}")
        cv_aucs = np.array([0.5])

    gbt.fit(Xs, y)
    importances = gbt.feature_importances_

    log("  Computing permutation importance...")
    try:
        perm = permutation_importance(gbt, Xs, y, n_repeats=30, random_state=42, scoring="roc_auc")
        perm_means, perm_stds = perm.importances_mean, perm.importances_std
    except Exception as e:
        log(f"  Permutation importance failed: {e}")
        perm_means, perm_stds = importances, np.zeros_like(importances)

    gbt_results = []
    for i, feat in enumerate(feature_names):
        gbt_results.append({
            "feature":         feat,
            "gbt_importance":  round(float(importances[i]), 5),
            "perm_importance": round(float(perm_means[i]),  5),
            "perm_std":        round(float(perm_stds[i]),   5),
        })
    gbt_results.sort(key=lambda x: x["perm_importance"], reverse=True)
    results["gbt_importance"] = gbt_results
    results["cv_auc_mean"]    = round(float(cv_aucs.mean()), 4)
    results["cv_auc_std"]     = round(float(cv_aucs.std()),  4)

    log("Fitting logistic regression...")
    lr = LogisticRegression(max_iter=1000, C=0.5, random_state=42)
    lr.fit(Xs, y)
    lr_results = [{"feature": feat, "coef": round(float(lr.coef_[0][i]), 5)}
                  for i, feat in enumerate(feature_names)]
    lr_results.sort(key=lambda x: abs(x["coef"]), reverse=True)
    results["logistic_coef"] = lr_results

    log("Building quartile tables...")
    top_features = [r["feature"] for r in gbt_results[:8]]
    quartile_tables = {}
    for feat in top_features:
        try:
            q = pd.qcut(X[feat], q=4, duplicates="drop")
            tbl = y.groupby(q).agg(["mean","count"])
            tbl.columns = ["win_rate","n"]
            tbl["win_rate"] = tbl["win_rate"].round(3)
            quartile_tables[feat] = tbl.reset_index().to_dict(orient="records")
        except Exception:
            pass
    results["quartile_tables"] = quartile_tables
    return results

def format_report(results, n_trades, win_rate):
    lines = []
    HR = "=" * 72
    hr = "-" * 72
    lines += [HR, "  FEATURE WINNER ANALYSIS -- US Fixed-Return System",
              f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", HR, "",
              f"Cluster:   probability >= {PROB_LOW}",
              f"Trades:    {n_trades}   Win rate: {win_rate*100:.1f}%",
              f"CV AUC:    {results['cv_auc_mean']:.3f} +/- {results['cv_auc_std']:.3f}", ""]

    auc = results["cv_auc_mean"]
    if auc < 0.52:
        interp = "NO signal above chance. Losses in this cluster are random at entry time."
    elif auc < 0.57:
        interp = "Weak signal. Marginal separation -- not strong enough to act on alone."
    elif auc < 0.65:
        interp = "Moderate signal. Worth building a secondary feature gate."
    else:
        interp = "Strong signal. Features clearly separate winners from losers."
    lines += [f"Interpretation: {interp}", ""]

    lines += [HR, "  GBT PERMUTATION IMPORTANCE", hr,
              f"  {'Feature':<22}  {'Perm Imp':>10}  {'+-':>8}  {'GBT Imp':>10}", hr]
    for r in results["gbt_importance"][:15]:
        lines.append(f"  {r['feature']:<22}  {r['perm_importance']:>10.5f}  "
                     f"{r['perm_std']:>8.5f}  {r['gbt_importance']:>10.5f}")
    lines.append("")

    lines += [HR, "  UNIVARIATE: WINNER vs LOSER (Mann-Whitney p<0.05)", hr,
              f"  {'Feature':<22}  {'Winner':>9}  {'Loser':>9}  {'Diff':>8}  {'Effect':>8}  {'p-val':>10}", hr]
    sig = [r for r in results["univariate"] if r["significant"]]
    for r in sig[:20]:
        lines.append(f"  {r['feature']:<22}  {r['winner_mean']:>9.3f}  {r['loser_mean']:>9.3f}  "
                     f"{r['diff']:>8.3f}  {r['effect_size']:>8.3f}  {r['mw_pvalue']:>10.5f}")
    if not sig:
        lines.append("  No features reach p<0.05.")
    lines.append("")

    lines += [HR, "  WIN RATE BY QUARTILE -- TOP FEATURES", hr]
    for feat, tbl in list(results["quartile_tables"].items())[:6]:
        lines.append(f"\n  {feat}")
        for row in tbl:
            bar = "X" * int(row["win_rate"] * 20)
            lines.append(f"    {str(row[feat]):<28}  WR={row['win_rate']*100:5.1f}%  n={row['n']:>4}  {bar}")

    lines += ["", HR, "  LOGISTIC REGRESSION COEFFICIENTS", hr,
              f"  {'Feature':<22}  {'Coef':>10}  Direction", hr]
    for r in results["logistic_coef"][:15]:
        direction = "higher = more likely WINNER" if r["coef"] > 0 else "higher = more likely LOSER"
        lines.append(f"  {r['feature']:<22}  {r['coef']:>10.5f}  {direction}")

    lines += ["", HR, "  RECOMMENDED NEXT STEPS", hr]
    if auc >= 0.57:
        top_feats = [r["feature"] for r in results["gbt_importance"][:3]
                     if r["perm_importance"] > 0.001]
        lines.append("  Top features to investigate as entry filters:")
        for f in top_feats:
            if f in results["quartile_tables"]:
                tbl = results["quartile_tables"][f]
                best_q = max(tbl, key=lambda r: r["win_rate"])
                lines.append(f"    {f}: best quartile WR={best_q['win_rate']*100:.1f}%")
        lines += ["", "  Next: run feature_gate_ablation.py with top feature"]
    else:
        lines += ["  AUC < 0.57 -- no actionable feature gate identified.",
                  "  Consider: sector clustering or correlation filter (LBRDK/CHTR pairs)."]

    lines += ["", HR]
    return "\n".join(lines)

def main():
    log("=== Feature Winner Analysis Starting ===")
    try:
        from scipy import stats
    except ImportError:
        log("Installing scipy...")
        os.system("pip install scipy --break-system-packages -q")

    trades  = load_trades()
    trades  = label_trades(trades)
    cluster = filter_cluster(trades)
    n_trades  = len(cluster)
    win_rate  = cluster["winner"].mean()

    X, y = build_feature_matrix(cluster)
    if len(X) < MIN_TRADES:
        log(f"ERROR: Only {len(X)} trades after price matching. Check DATA_DIR: {DATA_DIR.resolve()}")
        sys.exit(1)

    results = run_analysis(X, y)

    out_json = SIM_RESULTS_DIR / "feature_winner_analysis.json"
    out_json.write_text(json.dumps({
        "meta": {"generated": datetime.now().isoformat(), "n_trades": n_trades,
                 "win_rate": round(win_rate,4), "prob_low": PROB_LOW,
                 "prob_high": PROB_HIGH, "cv_auc": results["cv_auc_mean"]},
        **results,
    }, indent=2))
    log(f"JSON saved -> {out_json}")

    report  = format_report(results, n_trades, win_rate)
    out_txt = SIM_RESULTS_DIR / "feature_winner_analysis.txt"
    out_txt.write_text(report)
    log(f"Report saved -> {out_txt}")
    print("\n" + report)

if __name__ == "__main__":
    main()
