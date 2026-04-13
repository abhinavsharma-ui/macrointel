#!/usr/bin/env python3
"""
Binary Directional Training
====================
Only predict buy/sell (skip hold) for better precision.
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
sys.path.insert(0, str(PROJECT_DIR))

FEATURES_DIR = PROJECT_DIR / "data" / "features_10yr"
MODELS_DIR = PROJECT_DIR / "models" / "checkpoints"


def build_binary_labels(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Binary: 0=sell, 1=buy (skip hold)"""
    if "close" not in df.columns:
        return pd.Series(1, index=df.index)
    
    close = df["close"]
    fwd_ret = close.pct_change(horizon).shift(-horizon)
    daily_std = close.pct_change().std()
    threshold = daily_std * 0.6 * (horizon ** 0.5)
    
    # Binary: upward > threshold = buy (1), downward < -threshold = sell (0)
    labels = (fwd_ret > threshold).astype(int)
    labels = labels.fillna(1)
    
    return labels


def main():
    logger.info("=" * 60)
    logger.info("BINARY DIRECTIONAL TRAINING")
    logger.info("=" * 60)
    
    import xgboost as xgb
    from sklearn.metrics import accuracy_score
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
        y = build_binary_labels(df, 5)
        y = y.reindex(X.index).fillna(1).astype(int)
        
        n = len(X)
        train_end = int(n * 0.80)
        all_X.append(X.iloc[:train_end])
        all_y.append(y.iloc[:train_end])
    
    X = pd.concat(all_X, ignore_index=True).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    y = pd.concat(all_y, ignore_index=True)
    
    logger.info(f"Training set: {X.shape}")
    
    # Check class balance
    logger.info(f"Class distribution: buy={y.sum()}, sell={len(y)-y.sum()}")
    
    # Train with optimized params for binary
    xgb_params = {
        "n_estimators": 800,
        "max_depth": 5,
        "learning_rate": 0.015,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "scale_pos_weight": 1.0,
        "random_state": 42,
        "n_jobs": 4,
        "tree_method": "hist",
    }
    
    # CV
    tscv = TimeSeriesSplit(n_splits=5)
    fold_accs = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_tr, y_tr, verbose=False)
        preds = model.predict(X_te)
        
        acc = accuracy_score(y_te, preds)
        fold_accs.append(acc)
        
        buy_acc = accuracy_score(y_te[y_te==1], preds[y_te==1])
        sell_acc = accuracy_score(y_te[y_te==0], preds[y_te==0])
        
        logger.info(f"Fold {fold+1}: acc={acc:.3f}, buy_acc={buy_acc:.3f}, sell_acc={sell_acc:.3f}")
    
    avg_acc = np.mean(fold_accs)
    
    logger.info("=" * 60)
    logger.info("BINARY RESULTS")
    logger.info("=" * 60)
    logger.info(f"Accuracy: {avg_acc:.1%}")
    
    if avg_acc >= 0.70:
        logger.info("★ 70% TARGET ACHIEVED!")
    
    # Train final model
    logger.info("Training final model...")
    final = xgb.XGBClassifier(**xgb_params)
    final.fit(X, y, verbose=False)
    
    # Save
    final.save_model(str(MODELS_DIR / "binary_model.json"))
    logger.info(f"Model saved!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())