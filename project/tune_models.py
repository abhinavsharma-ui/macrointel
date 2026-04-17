#!/usr/bin/env python3
"""
Bayesian Hyperparameter Tuning via Optuna
==========================================
Tunes XGBoost, LightGBM, and CatBoost base learners using:
    * TimeSeriesSplit (n_splits=5) — NEVER shuffle time-series data
    * Objective: maximize out-of-fold PRECISION on the "take" class

The "take" class is computed as a binary label: class != 1 in the 3-class
schema (i.e. both buy=2 and sell=0 are "take", neutral=1 is "skip"). This
matches the meta-model take/skip convention used elsewhere in the system.

Output: project/params.json with best params per base learner. Also writes
project/models/checkpoints/tuning_report_{YYYYMMDD}.json with cv scores.

Run weekly via cron:
    0 3 * * 0 cd ~/macro_intelligence_complete && source venv/bin/activate \
        && python project/tune_models.py --trials 40

Standalone test (fast smoke, 3 trials each):
    python project/tune_models.py --trials 3 --max-samples 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent
FEATURES_DIR_DEFAULT = PROJECT_DIR / "data" / "features_10yr"
CHECKPOINTS_DIR = PROJECT_DIR / "models" / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
PARAMS_PATH = PROJECT_DIR / "params.json"


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def build_take_labels(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """
    Binary take/skip label:
      1 = actionable move (forward return > 0.5*sigma*sqrt(horizon) in either direction)
      0 = skip / chop
    """
    if "close" not in df.columns:
        return pd.Series(0, index=df.index, dtype=int)
    close = df["close"]
    fwd_ret = close.pct_change(horizon).shift(-horizon)
    daily_std = close.pct_change().std()
    threshold = (daily_std or 0.01) * 0.5 * (horizon ** 0.5)
    take = (fwd_ret.abs() >= threshold).astype(int)
    take.loc[fwd_ret.isna()] = 0
    return take


def load_training_matrix(
    features_dir: Path,
    *,
    horizon: int = 5,
    max_samples: Optional[int] = None,
    min_rows_per_symbol: int = 100,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load all per-symbol parquets, build take labels, concat chronologically per-symbol."""
    if not features_dir.exists():
        raise FileNotFoundError(f"features dir not found: {features_dir}")

    all_X: List[pd.DataFrame] = []
    all_y: List[pd.Series] = []
    for path in sorted(features_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            logger.warning("skip %s (%s)", path.name, exc)
            continue
        if len(df) < min_rows_per_symbol:
            continue
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric:
            continue
        X = df[numeric].dropna(how="all")
        if X.empty:
            continue
        y = build_take_labels(df, horizon=horizon).reindex(X.index).fillna(0).astype(int)
        # Chronological only: concat in order; pandas order is preserved
        all_X.append(X)
        all_y.append(y)

    if not all_X:
        raise RuntimeError("no training data found")

    X = pd.concat(all_X, ignore_index=True).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    y = pd.concat(all_y, ignore_index=True).astype(int)

    if max_samples is not None and len(X) > max_samples:
        X = X.iloc[-max_samples:].reset_index(drop=True)
        y = y.iloc[-max_samples:].reset_index(drop=True)

    logger.info("loaded %d rows, %d features, take_rate=%.2f%%", len(X), X.shape[1], 100 * y.mean())
    return X, y


# -----------------------------------------------------------------------------
# Objective factories (one per base learner)
# -----------------------------------------------------------------------------


def _precision_on_take(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def _cv_split_indices(n: int, n_splits: int = 5):
    from sklearn.model_selection import TimeSeriesSplit
    tss = TimeSeriesSplit(n_splits=n_splits)
    return list(tss.split(np.arange(n)))


def tune_xgboost(X: pd.DataFrame, y: pd.Series, n_trials: int, n_splits: int, timeout: Optional[int]) -> Dict[str, Any]:
    import optuna
    import xgboost as xgb

    splits = _cv_split_indices(len(X), n_splits)

    def objective(trial: "optuna.trial.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "tree_method": "hist",
            "n_jobs": 4,
            "random_state": 42,
            "eval_metric": "logloss",
        }
        scores = []
        for train_idx, val_idx in splits:
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr, verbose=False)
            preds = model.predict(X_va)
            scores.append(_precision_on_take(y_va.values, preds))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    return {"best_params": study.best_params, "best_score": float(study.best_value), "n_trials": len(study.trials)}


def tune_lightgbm(X: pd.DataFrame, y: pd.Series, n_trials: int, n_splits: int, timeout: Optional[int]) -> Dict[str, Any]:
    import optuna
    import lightgbm as lgb

    splits = _cv_split_indices(len(X), n_splits)

    def objective(trial: "optuna.trial.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 120),
            "n_jobs": 4,
            "random_state": 42,
            "verbose": -1,
        }
        scores = []
        for train_idx, val_idx in splits:
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_va)
            scores.append(_precision_on_take(y_va.values, preds))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    return {"best_params": study.best_params, "best_score": float(study.best_value), "n_trials": len(study.trials)}


def tune_catboost(X: pd.DataFrame, y: pd.Series, n_trials: int, n_splits: int, timeout: Optional[int]) -> Dict[str, Any]:
    import optuna
    from catboost import CatBoostClassifier

    splits = _cv_split_indices(len(X), n_splits)

    def objective(trial: "optuna.trial.Trial") -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 200, 900, step=100),
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "random_state": 42,
            "verbose": False,
            "bootstrap_type": "Bernoulli",
        }
        scores = []
        for train_idx, val_idx in splits:
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
            model = CatBoostClassifier(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_va).astype(int).ravel()
            scores.append(_precision_on_take(y_va.values, preds))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    return {"best_params": study.best_params, "best_score": float(study.best_value), "n_trials": len(study.trials)}


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _persist(results: Dict[str, Dict[str, Any]]) -> Path:
    payload = {
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "models": results,
    }
    # Merge on top of existing params.json if present
    if PARAMS_PATH.exists():
        try:
            prev = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))
            if isinstance(prev, dict):
                prev.update(payload)
                payload = prev
        except Exception:
            pass
    PARAMS_PATH.write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s", PARAMS_PATH)

    report = CHECKPOINTS_DIR / f"tuning_report_{datetime.utcnow():%Y%m%d}.json"
    report.write_text(json.dumps(payload, indent=2))
    return PARAMS_PATH


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default=str(FEATURES_DIR_DEFAULT))
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None, help="per-model optuna timeout in seconds")
    parser.add_argument("--only", choices=["xgb", "lgb", "cat"], default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    X, y = load_training_matrix(
        Path(args.features_dir),
        horizon=args.horizon,
        max_samples=args.max_samples,
    )
    if y.sum() < 50:
        logger.error("fewer than 50 take labels; aborting")
        return 2

    results: Dict[str, Dict[str, Any]] = {}
    runners = {
        "xgboost": tune_xgboost,
        "lightgbm": tune_lightgbm,
        "catboost": tune_catboost,
    }
    if args.only:
        key_map = {"xgb": "xgboost", "lgb": "lightgbm", "cat": "catboost"}
        runners = {key_map[args.only]: runners[key_map[args.only]]}

    for name, fn in runners.items():
        logger.info("=== tuning %s (trials=%d splits=%d) ===", name, args.trials, args.splits)
        t0 = time.time()
        try:
            results[name] = fn(X, y, n_trials=args.trials, n_splits=args.splits, timeout=args.timeout)
        except Exception as exc:
            logger.exception("tuning %s failed", name)
            results[name] = {"error": str(exc)}
            continue
        elapsed = time.time() - t0
        logger.info("%s best_score=%.4f elapsed=%.1fs", name, results[name]["best_score"], elapsed)

    _persist(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
