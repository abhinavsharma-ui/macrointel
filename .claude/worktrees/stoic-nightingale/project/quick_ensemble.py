#!/usr/bin/env python3
"""
Quick Ensemble Training
===================
Fast XGBoost + LightGBM ensemble for precision boost.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List
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


def build_labels(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    if "close" not in df.columns:
        return pd.Series(1, index=df.index)
    
    close = df["close"]
    fwd_ret = close.pct_change(horizon).shift(-horizon)
    daily_std = close.pct_change().std()
    threshold = daily_std * 0.5 * (horizon ** 0.5)
    
    labels = pd.cut(
        fwd_ret,
        bins=[-np.inf, -threshold, threshold, np.inf],
        labels=[0, 1, 2],
    ).astype(float).fillna(1)
    
    return labels.astype(int)


def main():
    logger.info("=" * 60)
    logger.info("ENSEMBLE TRAINING (XGBoost + LightGBM)")
    logger.info("=" * 60)
    
    try:
        import xgboost as xgb
        import lightgbm as lgb
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.model_selection import TimeSeriesSplit
        
        logger.info("Loading features...")
        
        matrices = {}
        for f in FEATURES_DIR.glob("*.parquet"):
            try:
                df = pd.read_parquet(f)
                if len(df) > 100:
                    matrices[f.stem] = df
            except:
                pass
        
        logger.info(f"Loaded {len(matrices)} symbols")
        
        # Combine all data
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
        
        X_combined = pd.concat(all_X, ignore_index=True).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        y_combined = pd.concat(all_y, ignore_index=True)
        
        logger.info(f"Training set: {X_combined.shape}")
        
        # XGBoost params
        xgb_params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.02,
            "subsample": 0.75,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.1,
            "reg_lambda": 1.5,
            "min_child_weight": 10,
            "random_state": 42,
            "n_jobs": 4,
            "tree_method": "hist",
        }
        
        # LightGBM params
        lgb_params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.02,
            "subsample": 0.75,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.1,
            "reg_lambda": 1.5,
            "min_child_samples": 10,
            "random_state": 42,
            "n_jobs": 4,
            "verbose": -1,
        }
        
        # Train XGBoost
        logger.info("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(**xgb_params)
        
        n = len(X_combined)
        n_train = int(n * 0.85)
        
        xgb_model.fit(
            X_combined.iloc[:n_train], 
            y_combined.iloc[:n_train],
            verbose=False,
        )
        
        # Train LightGBM
        logger.info("Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(
            X_combined.iloc[:n_train], 
            y_combined.iloc[:n_train],
        )
        
        # Ensemble CV
        logger.info("Cross-validation...")
        
        tscv = TimeSeriesSplit(n_splits=5)
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_combined)):
            X_tr, X_te = X_combined.iloc[train_idx], X_combined.iloc[test_idx]
            y_tr, y_te = y_combined.iloc[train_idx], y_combined.iloc[test_idx]
            
            # XGB predictions
            xgb_m = xgb.XGBClassifier(**xgb_params)
            xgb_m.fit(X_tr, y_tr, verbose=False)
            xgb_probs = xgb_m.predict_proba(X_te)
            
            # LGB predictions
            lgb_m = lgb.LGBMClassifier(**lgb_params)
            lgb_m.fit(X_tr, y_tr)
            lgb_probs = lgb_m.predict_proba(X_te)
            
            # Ensemble (average)
            ens_probs = (xgb_probs + lgb_probs) / 2
            ens_preds = ens_probs.argmax(axis=1)
            
            # Directional accuracy (buy/sell only)
            dir_mask = (y_te != 1) & (ens_preds != 1)
            if dir_mask.any():
                dir_acc = accuracy_score(y_te[dir_mask], ens_preds[dir_mask])
            else:
                dir_acc = 0
            
            fold_results.append(dir_acc)
            logger.info(f"Fold {fold+1}: dir_acc={dir_acc:.3f}")
        
        avg_dir_acc = np.mean(fold_results)
        
        logger.info("=" * 60)
        logger.info("ENSEMBLE RESULTS")
        logger.info("=" * 60)
        logger.info(f"Directional Accuracy: {avg_dir_acc:.1%}")
        
        if avg_dir_acc >= 0.70:
            logger.info("★ 70% TARGET ACHIEVED!")
        
        # Save ensemble
        import pickle
        ensemble = {"xgb": xgb_model, "lgb": lgb_model}
        with open(MODELS_DIR / "ensemble_model.pkl", "wb") as f:
            pickle.dump(ensemble, f)
        
        logger.info(f"Models saved to {MODELS_DIR}")
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())