#!/usr/bin/env python3
"""
Optimized Training with hyperparameter search
======================================
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
    logger.info("=" * 60)
    logger.info("OPTIMIZED TRAINING")
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
        except: pass
    
    logger.info(f"Loaded {len(matrices)} symbols")
    
    all_X, all_y = [], []
    for sym, df in matrices.items():
        numeric = [c for c in df.columns if df[c].dtype in (float, int, "float64", "int64")]
        X = df[numeric].dropna(how="all")
        y = build_labels(df, 5)
        y = y.reindex(X.index).fillna(1).astype(int)
        n = len(X)
        train_end = int(n * 0.80)
        all_X.append(X.iloc[:train_end])
        all_y.append(y.iloc[:train_end])
    
    X = pd.concat(all_X, ignore_index=True).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    y = pd.concat(all_y, ignore_index=True)
    
    logger.info(f"Training set: {X.shape}")
    
    # Try different hyperparameter configs
    configs = [
        {"max_depth": 4, "learning_rate": 0.01, "n_estimators": 800, "min_child_weight": 20},
        {"max_depth": 5, "learning_rate": 0.02, "n_estimators": 600, "min_child_weight": 15},
        {"max_depth": 6, "learning_rate": 0.015, "n_estimators": 700, "min_child_weight": 10},
        {"max_depth": 3, "learning_rate": 0.005, "n_estimators": 1000, "min_child_weight": 25},
    ]
    
    best_config = None
    best_acc = 0
    
    for i, cfg in enumerate(configs):
        xgb_params = {
            **cfg,
            "subsample": 0.75,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.1,
            "reg_lambda": 1.5,
            "random_state": 42,
            "n_jobs": 4,
            "tree_method": "hist",
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        accs = []
        
        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            
            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X_tr, y_tr, verbose=False)
            preds = model.predict(X_te)
            
            # Directional accuracy
            dir_mask = (y_te != 1) & (preds != 1)
            if dir_mask.any():
                acc = accuracy_score(y_te[dir_mask], preds[dir_mask])
            else:
                acc = 0
            accs.append(acc)
        
        avg_acc = np.mean(accs)
        logger.info(f"Config {i+1}: depth={cfg['max_depth']}, lr={cfg['learning_rate']}, acc={avg_acc:.3f}")
        
        if avg_acc > best_acc:
            best_acc = avg_acc
            best_config = cfg
    
    logger.info("=" * 60)
    logger.info(f"BEST CONFIG: {best_config}")
    logger.info(f"Best accuracy: {best_acc:.1%}")
    
    # Train final with best config
    final_params = {
        **best_config,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.5,
        "random_state": 42,
        "n_jobs": 4,
        "tree_method": "hist",
    }
    
    final = xgb.XGBClassifier(**final_params)
    final.fit(X, y, verbose=False)
    final.save_model(str(MODELS_DIR / "optimized_model.json"))
    logger.info("Model saved!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())