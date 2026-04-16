#!/usr/bin/env python3
"""
Balanced Directional Training
=======================
Properly balanced buy/sell predictions.
"""

import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.utils import resample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

FEATURES_DIR = PROJECT_DIR / "data" / "features_10yr"
MODELS_DIR = PROJECT_DIR / "models" / "checkpoints"


def build_balanced_labels(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    if "close" not in df.columns:
        return pd.Series(1, index=df.index)
    
    close = df["close"]
    fwd_ret = close.pct_change(horizon).shift(-horizon)
    daily_std = close.pct_change().std()
    
    # Use smaller threshold to detect more signals
    threshold = daily_std * 0.4 * (horizon ** 0.5)
    
    # Binary: 0=sell, 1=buy
    labels = pd.cut(
        fwd_ret,
        bins=[-np.inf, -threshold, threshold, np.inf],
        labels=[0, 2, 2],  # Map hold (1) and buy (2) to different values
    )
    
    # Only keep strong signals (buy/sell), not hold
    return labels.replace("2", 1).astype(float).fillna(-1).replace(-1, np.nan)


def main():
    logger.info("=" * 60)
    logger.info("BALANCED DIRECTIONAL TRAINING")
    logger.info("=" * 60)
    
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, recall_score, precision_score
    from sklearn.model_selection import TimeSeriesSplit
    
    # Load data
    matrices = {}
    for f in FEATURES_DIR.glob("*.parquet"):
        try:
            df = pd.read_parquet(f)
            if len(df) > 100:
                matrices[f.stem] = df
        except:
            pass
    
    logger.info(f"Loaded {len(matrices)} symbols")
    
    # Combine
    all_X, all_y = [], []
    for sym, df in matrices.items():
        numeric = [c for c in df.columns if df[c].dtype in (float, int, "float64", "int64")]
        X = df[numeric].dropna(how="all")
        y = build_balanced_labels(df, 5)
        y = y.reindex(X.index)
        
        # Filter to only buy/sell (not hold)
        mask = y.notna()
        X = X[mask]
        y = y[mask].astype(int)
        
        n = len(X)
        train_end = int(n * 0.80)
        all_X.append(X.iloc[:train_end])
        all_y.append(y.iloc[:train_end])
    
    X = pd.concat(all_X, ignore_index=True).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    y = pd.concat(all_y, ignore_index=True)
    
    # Balance classes
    logger.info(f"Before balance: buy={y.sum()}, sell={len(y)-y.sum()}")
    
    # Upsample minority class
    buy_idx = y[y == 1].index
    sell_idx = y[y == 0].index
    
    if len(buy_idx) < len(sell_idx):
        buy_upsampled = resample(list(buy_idx), replace=True, n_samples=len(sell_idx), random_state=42)
        keep_idx = list(buy_upsampled) + list(sell_idx)
    else:
        sell_upsampled = resample(list(sell_idx), replace=True, n_samples=len(buy_idx), random_state=42)
        keep_idx = list(sell_upsampled) + list(buy_idx)
    
    X = X.loc[keep_idx]
    y = y.loc[keep_idx]
    
    logger.info(f"After balance: buy={y.sum()}, sell={len(y)-y.sum()}")
    logger.info(f"Training set: {X.shape}")
    
    # Train balanced
    xgb_params = {
        "n_estimators": 600,
        "max_depth": 5,
        "learning_rate": 0.02,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "scale_pos_weight": 1.0,  # Balanced
        "random_state": 42,
        "n_jobs": 4,
        "tree_method": "hist",
    }
    
    # CV
    tscv = TimeSeriesSplit(n_splits=5)
    fold_precisions = []
    fold_recalls = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_tr, y_tr, verbose=False)
        preds = model.predict(X_te)
        
        precision = precision_score(y_te, preds, zero_division=0)
        recall = recall_score(y_te, preds, zero_division=0)
        
        fold_precisions.append(precision)
        fold_recalls.append(recall)
        
        logger.info(f"Fold {fold+1}: precision={precision:.3f}, recall={recall:.3f}")
    
    avg_precision = np.mean(fold_precisions)
    avg_recall = np.mean(fold_recalls)
    
    logger.info("=" * 60)
    logger.info("BALANCED RESULTS")
    logger.info("=" * 60)
    logger.info(f"Precision: {avg_precision:.1%}")
    logger.info(f"Recall: {avg_recall:.1%}")
    
    # Train final
    final = xgb.XGBClassifier(**xgb_params)
    final.fit(X, y, verbose=False)
    
    # Save
    import pickle
    with open(MODELS_DIR / "balanced_model.pkl", "wb") as f:
        pickle.dump(final, f)
    logger.info(f"Model saved!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())