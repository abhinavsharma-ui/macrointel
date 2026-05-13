"""
compare_features.py
===================
A/B test: same data, same model, same split — but with and without
`add_new_features` applied. Prints test accuracy for each and the lift.

Run from repo root (after activating your venv):
    python compare_features.py
"""
from __future__ import annotations
import os, glob, time, warnings, sys
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "project"))

FEATURES_DIR = os.path.join(HERE, "project", "data", "features")

# --- load the data once, exactly like verify_audit.py does ---
paths = sorted(glob.glob(os.path.join(FEATURES_DIR, "*.parquet")))
rng = np.random.RandomState(0)
sample = list(rng.choice(paths, size=min(60, len(paths)), replace=False))
print(f"Loading {len(sample)} of {len(paths)} tickers...")

dfs_raw, dfs_new = [], []
try:
    from pipeline.new_features import add_new_features
    HAVE_NEW = True
except Exception as e:
    print(f"FATAL: cannot import add_new_features — {e}")
    sys.exit(1)

t0 = time.time()
for p in sample:
    try:
        d = pd.read_parquet(p).sort_index()
        d["__fwd_ret_5"] = d["close"].pct_change(5).shift(-5)
        dfs_raw.append(d.copy())
        # apply the new feature pipeline per-ticker
        d_new = add_new_features(d)
        d_new["__fwd_ret_5"] = d["close"].pct_change(5).shift(-5)  # restore target
        dfs_new.append(d_new)
    except Exception as e:
        print(f"  skip {p}: {e}")
print(f"  load + transform: {time.time()-t0:.1f}s")

df_raw = pd.concat(dfs_raw).sort_index()
df_new = pd.concat(dfs_new).sort_index()
print(f"  raw shape: {df_raw.shape}  | new shape: {df_new.shape}")


def train_and_score(df, label):
    nunique = df.nunique()
    dead = set(nunique[nunique <= 1].index)
    drop = dead | {"open","high","low","close","volume","obv","atr_14",
                   "chaikin_osc","di_plus","di_minus","adx_14","macd","macd_signal",
                   "macd_hist","price_vs_ema9","price_vs_ema21","price_vs_ema50",
                   "price_vs_ema200"}
    feat_cols = [c for c in df.columns if c not in drop and not c.startswith("__")]

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
    model = HistGradientBoostingClassifier(
        max_iter=80, max_depth=5, learning_rate=0.08,
        l2_regularization=1.0, min_samples_leaf=50, random_state=42,
    )
    t = time.time()
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    print(f"  [{label}] features={len(feat_cols):3d}  fit={time.time()-t:.1f}s  test_acc={acc:.4f}")
    return acc, len(feat_cols)


print("\n=== A/B comparison ===")
acc_raw, n_raw = train_and_score(df_raw, "BEFORE (102 cols, no cleanup)")
acc_new, n_new = train_and_score(df_new, "AFTER  (cleaned + new features)")

lift = acc_new - acc_raw
print(f"\nresult: {acc_raw:.4f} → {acc_new:.4f}  (lift = {lift:+.4f}, {lift/acc_raw*100:+.2f}%)")
print(f"feature count: {n_raw} → {n_new}")
if lift > 0:
    print("Pipeline change helps — wire it into the main training loop.")
else:
    print("No lift on this sample. Try with more tickers / longer history before deciding.")
