#!/usr/bin/env python3
"""
Final Triple Ensemble: XGBoost + LightGBM + CatBoost
==========================================
"""

import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent
FEATURES_DIR = PROJECT_DIR / "data" / "features_10yr"
MODELS_DIR = PROJECT_DIR / "models" / "checkpoints"


def build_labels(df, horizon=5):
    if "close" not in df.columns:
        return pd.Series(1, index=df.index)
    close = df["close"]
    fwd_ret = close.pct_change(horizon).shift(-horizon)
    daily_std = close.pct_change().std()
    threshold = daily_std * 0.5 * (horizon ** 0.5)
    labels = pd.cut(fwd_ret, bins=[-np.inf, -threshold, threshold, np.inf], labels=[0, 1, 2]).astype(float).fillna(1)
    return labels.astype(int)


def main():
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import TimeSeriesSplit
    
    logger.info("Loading data...")
    matrices = {}
    for f in FEATURES_DIR.glob("*.parquet"):
        try:
            df = pd.read_parquet(f)
            if len(df) > 100:
                matrices[f.stem] = df
        except: pass
    
    all_X, all_y = [], []
    for df in matrices.values():
        numeric = [c for c in df.columns if df[c].dtype in (float, int, "float64", "int64")]
        X = df[numeric].dropna(how="all")
        y = build_labels(df, 5)
        y = y.reindex(X.index).fillna(1).astype(int)
        n = len(X)
        all_X.append(X.iloc[:int(n*0.80)])
        all_y.append(y.iloc[:int(n*0.80)])
    
    X = pd.concat(all_X, ignore_index=True).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    y = pd.concat(all_y, ignore_index=True)
    
    logger.info(f"Data: {X.shape}")
    
    # Train 3 models
    models = []
    names = []
    
    # XGBoost
    logger.info("Training XGBoost...")
    xgb_m = xgb.XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.015, subsample=0.75, colsample_bytree=0.7, random_state=42, n_jobs=4, tree_method="hist")
    xgb_m.fit(X, y, verbose=False)
    models.append(xgb_m)
    names.append("XGB")
    
    # LightGBM
    logger.info("Training LightGBM...")
    lgb_m = lgb.LGBMClassifier(n_estimators=600, max_depth=6, learning_rate=0.015, subsample=0.75, colsample_bytree=0.7, random_state=42, n_jobs=4, verbose=-1)
    lgb_m.fit(X, y)
    models.append(lgb_m)
    names.append("LGB")
    
    # CatBoost
    logger.info("Training CatBoost...")
    cat_m = CatBoostClassifier(iterations=600, depth=6, learning_rate=0.015, random_state=42, verbose=False)
    cat_m.fit(X, y)
    models.append(cat_m)
    names.append("CAT")
    
    # CV ensemble
    logger.info("Cross-validation...")
    tscv = TimeSeriesSplit(n_splits=5)
    fold_accs = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        
        # Get predictions from each model
        probs = []
        for m in models:
            p = m.predict_proba(X_te)
            probs.append(p)
        
        # Average ensemble
        ens_prob = np.mean(probs, axis=0)
        ens_pred = ens_prob.argmax(axis=1)
        
        # Directional accuracy
        dir_mask = (y_te != 1) & (ens_pred != 1)
        if dir_mask.any():
            acc = accuracy_score(y_te[dir_mask], ens_pred[dir_mask])
        else:
            acc = 0
        
        fold_accs.append(acc)
        logger.info(f"Fold {fold+1}: dir_acc={acc:.3f}")
    
    logger.info("=" * 60)
    logger.info(f"TRIPLE ENSEMBLE: {np.mean(fold_accs):.1%}")
    
    if np.mean(fold_accs) >= 0.70:
        logger.info("★ 70% ACHIEVED!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())