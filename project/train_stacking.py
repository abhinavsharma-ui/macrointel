#!/usr/bin/env python3
"""
Stacking Ensemble Trainer — 3 Base Learners + Meta-Model
==========================================================
Builds a production stacking ensemble:

  Base learners (level-0):    XGBoost, LightGBM, CatBoost
  Meta-model  (level-1):      RandomForest (default) or Ridge

Training protocol (STRICTLY chronological, no shuffle):
  1. Chronological split: first 80% => train/val for stacking, last 20% => HOLDOUT
  2. Within the 80%: TimeSeriesSplit(n_splits=5) for OOF predictions
  3. Train meta-model on the OOF base predictions (with feature_completeness_ratio
     appended as an input feature so the meta can self-calibrate on data quality)
  4. Retrain each base learner on the full 80% train pool
  5. Evaluate on the 20% HOLDOUT — precision is measured honestly; NEVER hardcoded

Artifacts written to project/models/checkpoints/:
  stacking_xgb.joblib
  stacking_lgb.joblib
  stacking_cat.joblib
  stacking_meta.joblib
  stacking_meta.json            - feature order, meta type, target precision, holdout metrics
  stacking_holdout_report.json  - detailed per-model holdout metrics

The feature list is persisted alongside each base model so run.py can load
partial subsets (partial-fallback requirement).

Usage:
    python project/train_stacking.py                    # full train
    python project/train_stacking.py --meta ridge       # swap meta model
    python project/train_stacking.py --max-samples 20000 --folds 3  # quick smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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

TARGET_PRECISION = float(os.getenv("STACKING_TARGET_PRECISION", "0.70"))
HOLDOUT_FRACTION = float(os.getenv("STACKING_HOLDOUT_FRACTION", "0.20"))


# -----------------------------------------------------------------------------
# Label construction (binary take/skip, matches tune_models.py)
# -----------------------------------------------------------------------------


def build_take_labels(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
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
    horizon: int,
    max_samples: Optional[int],
    min_rows_per_symbol: int = 100,
) -> Tuple[pd.DataFrame, pd.Series]:
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
        all_X.append(X)
        all_y.append(y)
    if not all_X:
        raise RuntimeError("no training data found")

    X = pd.concat(all_X, ignore_index=True).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    y = pd.concat(all_y, ignore_index=True).astype(int)
    if max_samples and len(X) > max_samples:
        X = X.iloc[-max_samples:].reset_index(drop=True)
        y = y.iloc[-max_samples:].reset_index(drop=True)

    logger.info("training matrix: rows=%d features=%d take_rate=%.2f%%",
                len(X), X.shape[1], 100 * y.mean())
    return X, y


# -----------------------------------------------------------------------------
# Feature completeness helper
# -----------------------------------------------------------------------------


def compute_feature_completeness(df: pd.DataFrame) -> pd.Series:
    """
    Per-row fraction of non-null, finite features. Appended to the meta input
    so the meta model can self-calibrate on data quality.
    """
    arr = df.to_numpy(dtype=np.float64, copy=False)
    finite = np.isfinite(arr) & (arr != 0.0)
    # zeros-as-missing is conservative; ffill above means genuine zeros survive,
    # but for our purpose "all features filled with neutrals" should still look
    # incomplete.
    ratio = finite.sum(axis=1) / max(arr.shape[1], 1)
    return pd.Series(np.clip(ratio, 0.0, 1.0), index=df.index, name="feature_completeness_ratio")


# -----------------------------------------------------------------------------
# Base learners (factories so we can re-instantiate cleanly per fold)
# -----------------------------------------------------------------------------


def _xgb_factory(params: Optional[Dict[str, Any]] = None):
    import xgboost as xgb
    base = dict(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.02,
        subsample=0.75,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.5,
        tree_method="hist",
        n_jobs=4,
        random_state=42,
        eval_metric="logloss",
    )
    if params:
        base.update(params)
    return xgb.XGBClassifier(**base)


def _lgb_factory(params: Optional[Dict[str, Any]] = None):
    import lightgbm as lgb
    base = dict(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.02,
        subsample=0.75,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.5,
        num_leaves=63,
        n_jobs=4,
        random_state=42,
        verbose=-1,
    )
    if params:
        base.update(params)
    return lgb.LGBMClassifier(**base)


def _cat_factory(params: Optional[Dict[str, Any]] = None):
    from catboost import CatBoostClassifier
    base = dict(
        iterations=600,
        depth=6,
        learning_rate=0.02,
        l2_leaf_reg=3.0,
        random_state=42,
        verbose=False,
    )
    if params:
        base.update(params)
    return CatBoostClassifier(**base)


# -----------------------------------------------------------------------------
# Stacking training
# -----------------------------------------------------------------------------


def _predict_proba_take(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Binary take-class probability from any of the three learners."""
    try:
        probs = model.predict_proba(X)
    except Exception:
        # Very rare, but fall back to decision_function scaled
        raw = model.predict(X)
        return np.asarray(raw, dtype=np.float64).ravel()
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim == 1:
        return probs
    # Binary classifiers: take class == 1
    # Locate the positive column via .classes_ if available
    classes = getattr(model, "classes_", None)
    if classes is not None:
        try:
            idx = int(np.where(np.asarray(classes) == 1)[0][0])
            return probs[:, idx]
        except Exception:
            pass
    if probs.shape[1] >= 2:
        return probs[:, 1]
    return probs[:, 0]


def _build_meta_model(kind: str):
    kind = (kind or "rf").lower()
    if kind == "ridge":
        from sklearn.linear_model import RidgeClassifier
        return RidgeClassifier(alpha=1.0), "ridge"
    from sklearn.ensemble import RandomForestClassifier
    return (
        RandomForestClassifier(
            n_estimators=400,
            max_depth=7,
            min_samples_leaf=10,
            n_jobs=4,
            random_state=42,
        ),
        "random_forest",
    )


def _meta_predict(meta: Any, X_meta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (preds, take_proba). Works for both RF and Ridge (Ridge has no proba)."""
    preds = meta.predict(X_meta)
    if hasattr(meta, "predict_proba"):
        proba = meta.predict_proba(X_meta)
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            classes = getattr(meta, "classes_", None)
            idx = 1
            if classes is not None:
                try:
                    idx = int(np.where(np.asarray(classes) == 1)[0][0])
                except Exception:
                    pass
            return preds, proba[:, idx]
        return preds, proba.ravel()
    # Ridge fallback: map decision_function through sigmoid
    try:
        dfunc = meta.decision_function(X_meta)
        sig = 1.0 / (1.0 + np.exp(-dfunc))
        return preds, sig
    except Exception:
        return preds, preds.astype(np.float64)


def train_stacking(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    meta_kind: str = "rf",
    n_folds: int = 5,
    params_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Returns a dict ready to be joblib-dumped into individual files.
    Keys: xgb_model, lgb_model, cat_model, meta_model, features, holdout_metrics, ...
    """
    from sklearn.model_selection import TimeSeriesSplit

    # Load tuned params if available
    best_params = {"xgboost": None, "lightgbm": None, "catboost": None}
    if params_file and params_file.exists():
        try:
            payload = json.loads(params_file.read_text(encoding="utf-8"))
            models_block = payload.get("models") or payload
            for key in best_params:
                if isinstance(models_block.get(key), dict):
                    best_params[key] = models_block[key].get("best_params") or None
            logger.info("loaded tuned params from %s", params_file)
        except Exception as exc:
            logger.warning("failed to load %s: %s", params_file, exc)

    # Chronological split: first 80% train, last 20% holdout
    n = len(X)
    split = int(n * (1.0 - HOLDOUT_FRACTION))
    if split < 200 or (n - split) < 50:
        raise RuntimeError(f"insufficient data: n={n}, split={split}")

    X_train, X_hold = X.iloc[:split], X.iloc[split:]
    y_train, y_hold = y.iloc[:split], y.iloc[split:]
    feature_names = list(X.columns)

    logger.info("train=%d holdout=%d (holdout spans the most recent %.0f%% of data)",
                len(X_train), len(X_hold), HOLDOUT_FRACTION * 100)

    # Build OOF base predictions on the 80%
    tss = TimeSeriesSplit(n_splits=n_folds)
    oof_xgb = np.full(len(X_train), np.nan, dtype=np.float64)
    oof_lgb = np.full(len(X_train), np.nan, dtype=np.float64)
    oof_cat = np.full(len(X_train), np.nan, dtype=np.float64)

    for fold_idx, (tr_idx, va_idx) in enumerate(tss.split(np.arange(len(X_train)))):
        t0 = time.time()
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_va = X_train.iloc[va_idx]

        xgb_m = _xgb_factory(best_params["xgboost"])
        lgb_m = _lgb_factory(best_params["lightgbm"])
        cat_m = _cat_factory(best_params["catboost"])

        xgb_m.fit(X_tr, y_tr, verbose=False)
        lgb_m.fit(X_tr, y_tr)
        cat_m.fit(X_tr, y_tr)

        oof_xgb[va_idx] = _predict_proba_take(xgb_m, X_va)
        oof_lgb[va_idx] = _predict_proba_take(lgb_m, X_va)
        oof_cat[va_idx] = _predict_proba_take(cat_m, X_va)

        logger.info("fold %d: train=%d val=%d elapsed=%.1fs",
                    fold_idx + 1, len(tr_idx), len(va_idx), time.time() - t0)

    # Fill any gaps (first fold's train rows have no OOF preds; use per-fold mean)
    for arr in (oof_xgb, oof_lgb, oof_cat):
        mask = np.isnan(arr)
        if mask.any():
            arr[mask] = np.nanmean(arr) if not np.all(mask) else 0.5

    # Meta-training features: stacked probas + feature_completeness_ratio on training set
    fcr_train = compute_feature_completeness(X_train).to_numpy(dtype=np.float64)
    meta_X_train = np.column_stack([oof_xgb, oof_lgb, oof_cat, fcr_train])
    meta_feature_names = ["xgb_take_p", "lgb_take_p", "cat_take_p", "feature_completeness_ratio"]

    meta, meta_kind_resolved = _build_meta_model(meta_kind)
    logger.info("training meta model: %s", meta_kind_resolved)
    meta.fit(meta_X_train, y_train.values)

    # Retrain each base model on the FULL 80% train pool
    logger.info("retraining base models on full train pool (%d rows)", len(X_train))
    xgb_final = _xgb_factory(best_params["xgboost"])
    lgb_final = _lgb_factory(best_params["lightgbm"])
    cat_final = _cat_factory(best_params["catboost"])
    xgb_final.fit(X_train, y_train, verbose=False)
    lgb_final.fit(X_train, y_train)
    cat_final.fit(X_train, y_train)

    # Honest holdout evaluation on the last 20%
    hold_xgb = _predict_proba_take(xgb_final, X_hold)
    hold_lgb = _predict_proba_take(lgb_final, X_hold)
    hold_cat = _predict_proba_take(cat_final, X_hold)
    fcr_hold = compute_feature_completeness(X_hold).to_numpy(dtype=np.float64)
    meta_X_hold = np.column_stack([hold_xgb, hold_lgb, hold_cat, fcr_hold])

    meta_preds, meta_proba = _meta_predict(meta, meta_X_hold)
    y_true = y_hold.values.astype(int)

    def _precision(preds: np.ndarray) -> float:
        preds = preds.astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        return tp / (tp + fp) if (tp + fp) else 0.0

    def _recall(preds: np.ndarray) -> float:
        preds = preds.astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fn = int(((preds == 1) & (y_true == 0)).sum())
        pos = int((y_true == 1).sum())
        return tp / pos if pos else 0.0

    # Per-base precision: use 0.5 threshold on take-proba
    base_metrics = {
        "xgboost":  {"precision": _precision((hold_xgb >= 0.5).astype(int)), "n_take": int((hold_xgb >= 0.5).sum())},
        "lightgbm": {"precision": _precision((hold_lgb >= 0.5).astype(int)), "n_take": int((hold_lgb >= 0.5).sum())},
        "catboost": {"precision": _precision((hold_cat >= 0.5).astype(int)), "n_take": int((hold_cat >= 0.5).sum())},
    }

    actual_precision = _precision(meta_preds)
    recall_meta = _recall(meta_preds)
    n_take_meta = int((meta_preds == 1).sum())

    # MANDATORY log format (per spec)
    msg = (
        f"Precision: {actual_precision*100:.1f}% "
        f"(target: {TARGET_PRECISION*100:.0f}%, met: {actual_precision >= TARGET_PRECISION})"
    )
    logger.info(msg)
    logger.info("  holdout rows=%d take_rate=%.2f%% n_take_meta=%d recall=%.2f%%",
                len(X_hold), 100 * y_hold.mean(), n_take_meta, recall_meta * 100)
    logger.info("  base precisions: xgb=%.3f lgb=%.3f cat=%.3f",
                base_metrics["xgboost"]["precision"],
                base_metrics["lightgbm"]["precision"],
                base_metrics["catboost"]["precision"])

    holdout_report = {
        "target_precision": TARGET_PRECISION,
        "actual_precision": float(actual_precision),
        "precision_target_met": bool(actual_precision >= TARGET_PRECISION),
        "recall_on_take": float(recall_meta),
        "n_take_predictions": n_take_meta,
        "n_holdout_rows": len(X_hold),
        "holdout_take_rate": float(y_hold.mean()),
        "base_metrics": base_metrics,
        "meta_kind": meta_kind_resolved,
        "n_features": len(feature_names),
        "n_folds": n_folds,
        "trained_at": datetime.utcnow().isoformat() + "Z",
    }

    return {
        "xgb_model": xgb_final,
        "lgb_model": lgb_final,
        "cat_model": cat_final,
        "meta_model": meta,
        "features": feature_names,
        "meta_features": meta_feature_names,
        "meta_kind": meta_kind_resolved,
        "holdout_report": holdout_report,
    }


# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------


def _save_artifacts(artifacts: Dict[str, Any], out_dir: Path) -> None:
    import joblib
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {"model": artifacts["xgb_model"], "features": artifacts["features"]},
        out_dir / "stacking_xgb.joblib",
    )
    joblib.dump(
        {"model": artifacts["lgb_model"], "features": artifacts["features"]},
        out_dir / "stacking_lgb.joblib",
    )
    joblib.dump(
        {"model": artifacts["cat_model"], "features": artifacts["features"]},
        out_dir / "stacking_cat.joblib",
    )
    joblib.dump(
        {
            "model": artifacts["meta_model"],
            "meta_features": artifacts["meta_features"],
            "kind": artifacts["meta_kind"],
        },
        out_dir / "stacking_meta.joblib",
    )

    meta_json = {
        "features": artifacts["features"],
        "meta_features": artifacts["meta_features"],
        "meta_kind": artifacts["meta_kind"],
        "holdout_report": artifacts["holdout_report"],
        "target_precision": TARGET_PRECISION,
    }
    (out_dir / "stacking_meta.json").write_text(json.dumps(meta_json, indent=2))
    (out_dir / "stacking_holdout_report.json").write_text(json.dumps(artifacts["holdout_report"], indent=2))
    logger.info("saved artifacts to %s", out_dir)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default=str(FEATURES_DIR_DEFAULT))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--meta", choices=["rf", "ridge"], default="rf")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--params-file", default=str(PROJECT_DIR / "params.json"))
    parser.add_argument("--out-dir", default=str(CHECKPOINTS_DIR))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    X, y = load_training_matrix(
        Path(args.features_dir), horizon=args.horizon, max_samples=args.max_samples
    )
    if y.sum() < 50:
        logger.error("fewer than 50 take labels; aborting")
        return 2

    artifacts = train_stacking(
        X, y,
        meta_kind=args.meta,
        n_folds=args.folds,
        params_file=Path(args.params_file) if args.params_file else None,
    )
    _save_artifacts(artifacts, Path(args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
