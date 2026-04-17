#!/usr/bin/env python3
"""
Production Model - Triple Ensemble
================================
Ready for live trading with real money.
Saves complete model + metadata for production use.
"""

import logging
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent
FEATURES_DIR = PROJECT_DIR / "data" / "features_10yr"
MODELS_DIR = PROJECT_DIR / "models" / "production"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


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
    import pickle
    
    logger.info("=" * 60)
    logger.info("SAVING PRODUCTION MODEL")
    logger.info("=" * 60)
    
    # Load data
    matrices = {}
    for f in FEATURES_DIR.glob("*.parquet"):
        try:
            df = pd.read_parquet(f)
            if len(df) > 100:
                matrices[f.stem] = df
        except: pass
    
    all_X, all_y = [], []
    holdout_X, holdout_y = [], []
    for df in matrices.values():
        numeric = [c for c in df.columns if df[c].dtype in (float, int, "float64", "int64")]
        X = df[numeric].dropna(how="all")
        y = build_labels(df, 5)
        y = y.reindex(X.index).fillna(1).astype(int)
        n = len(X)
        split = int(n * 0.80)
        all_X.append(X.iloc[:split])
        all_y.append(y.iloc[:split])
        if n - split > 0:
            holdout_X.append(X.iloc[split:])
            holdout_y.append(y.iloc[split:])

    X = pd.concat(all_X, ignore_index=True).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    y = pd.concat(all_y, ignore_index=True)

    X_holdout = (
        pd.concat(holdout_X, ignore_index=True).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        if holdout_X else pd.DataFrame(columns=X.columns)
    )
    y_holdout = pd.concat(holdout_y, ignore_index=True) if holdout_y else pd.Series(dtype=int)

    feature_names = list(X.columns)
    
    logger.info(f"Training on {X.shape[0]} samples, {len(feature_names)} features")
    
    # Train 3 models
    models = {}
    
    # XGBoost
    logger.info("Training XGBoost...")
    xgb_m = xgb.XGBClassifier(
        n_estimators=700, max_depth=6, learning_rate=0.015,
        subsample=0.75, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.5,
        random_state=42, n_jobs=4, tree_method="hist"
    )
    xgb_m.fit(X, y, verbose=False)
    models["xgboost"] = xgb_m
    
    # LightGBM
    logger.info("Training LightGBM...")
    lgb_m = lgb.LGBMClassifier(
        n_estimators=700, max_depth=6, learning_rate=0.015,
        subsample=0.75, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.5,
        random_state=42, n_jobs=4, verbose=-1
    )
    lgb_m.fit(X, y)
    models["lightgbm"] = lgb_m
    
    # CatBoost
    logger.info("Training CatBoost...")
    cat_m = CatBoostClassifier(
        iterations=700, depth=6, learning_rate=0.015,
        random_state=42, verbose=False
    )
    cat_m.fit(X, y)
    models["catboost"] = cat_m
    
    # Save models
    with open(MODELS_DIR / "ensemble_models.pkl", "wb") as f:
        pickle.dump(models, f)

    # Evaluate ensemble on held-out tail split (20% per symbol, unseen in training)
    eval_metrics = {
        "buy_precision": None,
        "buy_recall": None,
        "buy_support": 0,
        "accuracy": None,
        "n_eval_samples": 0,
    }
    target_precision = float(0.70)
    actual_precision = None
    if not X_holdout.empty and len(y_holdout) > 0:
        try:
            from sklearn.metrics import precision_score, recall_score, accuracy_score

            # Ensemble = probability average across the 3 models for class 2 (buy)
            p_xgb = xgb_m.predict_proba(X_holdout)
            p_lgb = lgb_m.predict_proba(X_holdout)
            p_cat = cat_m.predict_proba(X_holdout)
            classes = list(xgb_m.classes_)
            buy_idx = classes.index(2) if 2 in classes else (len(classes) - 1)

            def _col(prob_matrix, idx):
                if prob_matrix.shape[1] <= idx:
                    return np.zeros(prob_matrix.shape[0])
                return prob_matrix[:, idx]

            ens_buy = (_col(p_xgb, buy_idx) + _col(p_lgb, buy_idx) + _col(p_cat, buy_idx)) / 3.0
            pred_buy = (ens_buy >= 0.5).astype(int)
            y_buy_true = (y_holdout.values == 2).astype(int)

            # Majority-vote class for accuracy
            full_ensemble = (p_xgb + p_lgb + p_cat) / 3.0
            pred_class = np.argmax(full_ensemble, axis=1)

            actual_precision = float(
                precision_score(y_buy_true, pred_buy, zero_division=0.0)
            )
            eval_metrics = {
                "buy_precision": round(actual_precision, 4),
                "buy_recall": round(float(recall_score(y_buy_true, pred_buy, zero_division=0.0)), 4),
                "buy_support": int(y_buy_true.sum()),
                "accuracy": round(float(accuracy_score(y_holdout.values, pred_class)), 4),
                "n_eval_samples": int(len(y_holdout)),
            }
            logger.info(
                f"Holdout eval: buy_precision={eval_metrics['buy_precision']} "
                f"recall={eval_metrics['buy_recall']} support={eval_metrics['buy_support']} "
                f"n={eval_metrics['n_eval_samples']}"
            )
        except Exception as exc:
            logger.warning(f"Holdout evaluation failed: {exc}")
            actual_precision = None
    else:
        logger.warning("No holdout split available — skipping evaluation; actual_precision=None")

    # Save metadata
    metadata = {
        "version": "1.0.0",
        "created_at": datetime.utcnow().isoformat(),
        "target_precision": target_precision,
        "actual_precision": actual_precision,
        "evaluation": eval_metrics,
        "n_samples": int(X.shape[0]),
        "n_features": len(feature_names),
        "n_symbols": len(matrices),
        "features": feature_names,
        "model_types": ["xgboost", "lightgbm", "catboost"],
        "horizon": 5,
        "ensemble_method": "probability_average",
    }

    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save feature list for inference
    pd.Series(feature_names).to_csv(MODELS_DIR / "features.csv", index=False)

    logger.info("=" * 60)
    logger.info("PRODUCTION MODEL SAVED")
    logger.info("=" * 60)
    logger.info(f"Location: {MODELS_DIR}")
    if actual_precision is not None:
        meets_target = "meets" if actual_precision >= target_precision else "below"
        logger.info(
            f"Buy-class precision (holdout): {actual_precision*100:.1f}% "
            f"({meets_target} {target_precision*100:.0f}% target)"
        )
    else:
        logger.info("Precision: not evaluated (no holdout data)")
    logger.info(f"Models: XGBoost + LightGBM + CatBoost")

    return 0


if __name__ == "__main__":
    sys.exit(main())