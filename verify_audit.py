"""
verify_audit.py
===============
Re-runs the feature audit on whatever is currently in
project/data/features/*.parquet and diffs against the recorded findings.

Run from the repo root:
    python verify_audit.py

What it checks
--------------
1. Dead (constant) features in the current parquet store
2. Test accuracy of a baseline HistGradientBoosting model
3. Permutation importance — top, bottom, harmful (negative)
4. Diff vs the recorded audit so you can see what changed
"""
from __future__ import annotations
import os, glob, time, warnings, sys
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(HERE, "project", "data", "features")
AUDIT_CSV    = os.path.join(HERE, "feature_audit_perm_importance.csv")

# --- Recorded findings (from the original audit) ---
RECORDED_DEAD = {
    "compound_score","weighted_compound_score","media_sentiment","official_sentiment",
    "filing_sentiment","filing_change_score","filing_fresh_language_score","new_risk_factors",
    "earnings_tone_signal","earnings_call_count","article_count","media_article_count",
    "official_article_count","filing_article_count","press_release_count","official_event_hit",
    "filing_event_hit","source_quality_score","sentiment_zscore","sentiment_velocity",
    "earnings_tone_velocity","weighted_sentiment_zscore","news_volume_spike",
    "source_quality_signal","media_sentiment_signal","travel_activity_level",
    "travel_activity_change",
}
RECORDED_HARMFUL = {
    "williams_r","vol_zscore","candle_body","bb_pct","hist_vol_10","rsi_9",
    "close_reversal_signal","vol_ratio",
}

def step(msg):
    print(f"\n=== {msg} ===", flush=True)

def main():
    paths = sorted(glob.glob(os.path.join(FEATURES_DIR, "*.parquet")))
    if not paths:
        sys.exit(f"No parquet files in {FEATURES_DIR}")

    step(f"Step 1: loading {min(60, len(paths))} of {len(paths)} tickers")
    rng = np.random.RandomState(0)
    sample = list(rng.choice(paths, size=min(60, len(paths)), replace=False))
    dfs = []
    for p in sample:
        try:
            d = pd.read_parquet(p).sort_index()
            d["__fwd_ret_5"] = d["close"].pct_change(5).shift(-5)
            dfs.append(d)
        except Exception as e:
            print(f"  skip {p}: {e}")
    df = pd.concat(dfs).sort_index()
    print(f"  combined shape: {df.shape}")

    # ----- Step 2: dead features -----
    step("Step 2: dead (constant) features")
    nunique = df.nunique()
    current_dead = set(nunique[nunique <= 1].index)
    print(f"  current dead count: {len(current_dead)}")
    fixed   = RECORDED_DEAD - current_dead
    still   = RECORDED_DEAD & current_dead
    new_dead = current_dead - RECORDED_DEAD
    if fixed:
        print(f"  FIXED since audit ({len(fixed)}): {sorted(fixed)}")
    if still:
        print(f"  STILL DEAD ({len(still)}): {sorted(still)}")
    if new_dead:
        print(f"  NEW DEAD ({len(new_dead)}): {sorted(new_dead)}")

    # ----- Step 3: train baseline model -----
    step("Step 3: train baseline + permutation importance")
    drop = current_dead | {"open","high","low","close","volume","obv","atr_14",
                          "chaikin_osc","di_plus","di_minus","adx_14","macd","macd_signal",
                          "macd_hist","price_vs_ema9","price_vs_ema21","price_vs_ema50",
                          "price_vs_ema200"}
    feat_cols = [c for c in df.columns if c not in drop and not c.startswith("__")]
    print(f"  candidate features: {len(feat_cols)}")

    df2 = df.dropna(subset=["__fwd_ret_5"]).copy().sort_index()
    y = df2["__fwd_ret_5"].values
    q_low, q_high = np.quantile(y, [0.33, 0.67])
    df2["__y"] = np.where(y < q_low, 0, np.where(y > q_high, 2, 1))

    split = int(len(df2) * 0.7)
    X_tr = df2.iloc[:split][feat_cols].fillna(0).replace([np.inf, -np.inf], 0).astype("float32")
    y_tr = df2.iloc[:split]["__y"].values
    X_te = df2.iloc[split:][feat_cols].fillna(0).replace([np.inf, -np.inf], 0).astype("float32")
    y_te = df2.iloc[split:]["__y"].values

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.inspection import permutation_importance

    model = HistGradientBoostingClassifier(
        max_iter=80, max_depth=5, learning_rate=0.08,
        l2_regularization=1.0, min_samples_leaf=50, random_state=42,
    )
    t0 = time.time()
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    print(f"  fit: {time.time()-t0:.1f}s | test_acc={acc:.4f} (chance=0.333)")

    n = min(1500, len(X_te))
    idx = np.random.RandomState(0).choice(len(X_te), n, replace=False)
    t0 = time.time()
    perm = permutation_importance(
        model, X_te.iloc[idx], y_te[idx],
        n_repeats=2, random_state=42, n_jobs=2, scoring="accuracy",
    )
    print(f"  perm: {time.time()-t0:.1f}s")

    out = pd.DataFrame({
        "feature": feat_cols,
        "perm_importance_mean": perm.importances_mean,
        "perm_importance_std":  perm.importances_std,
    }).sort_values("perm_importance_mean", ascending=False)
    out_path = os.path.join(HERE, "verify_perm_importance.csv")
    out.to_csv(out_path, index=False)
    print(f"  saved → {out_path}")

    print("\n  TOP 15 BY PERM IMPORTANCE:")
    print(out.head(15).to_string(index=False))
    print("\n  BOTTOM 10:")
    print(out.tail(10).to_string(index=False))

    # ----- Step 4: harmful feature diff -----
    step("Step 4: harmful features diff vs audit")
    current_harmful = set(out[out["perm_importance_mean"] < 0]["feature"])
    fixed_h = RECORDED_HARMFUL - current_harmful
    still_h = RECORDED_HARMFUL & current_harmful
    new_h   = current_harmful - RECORDED_HARMFUL
    if fixed_h:
        print(f"  NO LONGER HARMFUL: {sorted(fixed_h)}")
    if still_h:
        print(f"  STILL HARMFUL: {sorted(still_h)}")
    if new_h:
        print(f"  NEW HARMFUL: {sorted(new_h)}")

    # ----- Step 5: side-by-side -----
    if os.path.exists(AUDIT_CSV):
        step("Step 5: side-by-side with recorded audit")
        prev = pd.read_csv(AUDIT_CSV)
        merged = out.merge(prev, on="feature", how="outer", suffixes=("_now", "_audit"))
        merged["delta"] = merged["perm_importance_mean_now"] - merged["perm_importance_mean_audit"]
        big_moves = merged.dropna(subset=["delta"]).reindex(
            merged["delta"].abs().sort_values(ascending=False).index
        ).head(15)
        print(big_moves[["feature","perm_importance_mean_audit","perm_importance_mean_now","delta"]].to_string(index=False))

    print("\nDONE.")

if __name__ == "__main__":
    main()
