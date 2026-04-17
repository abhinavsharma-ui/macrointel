#!/usr/bin/env python3
"""
Stacking Inference Engine (Production)
========================================
Loads the 3 base learners + meta-model trained by train_stacking.py and
produces a single classification + probability at inference time. Designed
to be a drop-in replacement for the existing single-XGBoost prediction path
inside run.py::_inference_loop.

Partial-model fallback policy (per spec):
  * All 3 base learners available     -> use full stack, run meta
  * 2 base learners available         -> average those two probs, skip meta,
                                          return majority vote w/ mean proba
  * 1 base learner available          -> use it directly
  * 0 base learners available         -> caller should fall back to its prior
                                          XGBoost model (returns None)

Loads are mtime-aware: models hot-reload when files change on disk.

Public API:
    engine = StackingInferenceEngine(checkpoints_dir)
    engine.load()                       # idempotent, hot-reloadable
    result = engine.predict(row_or_df)  # -> PredictionResult

PredictionResult fields:
    pred_class : int  (1=take, 0=skip)
    pred_proba : np.ndarray shape (2,)  [skip_p, take_p]
    take_proba : float
    models_used : list[str]
    meta_used : bool
    fallback_reason : str | ""
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_BASE_FILES = {
    "xgb": "stacking_xgb.joblib",
    "lgb": "stacking_lgb.joblib",
    "cat": "stacking_cat.joblib",
}
_META_FILE = "stacking_meta.joblib"


@dataclass
class PredictionResult:
    pred_class: int
    pred_proba: np.ndarray  # shape (2,)
    take_proba: float
    models_used: List[str] = field(default_factory=list)
    meta_used: bool = False
    fallback_reason: str = ""


class StackingInferenceEngine:
    # When no model files exist, re-check the filesystem at most every N seconds.
    _MISSING_RECHECK_INTERVAL = 60.0

    def __init__(self, checkpoints_dir: Optional[Path] = None) -> None:
        self.checkpoints_dir = Path(
            checkpoints_dir or Path(__file__).resolve().parent.parent / "models" / "checkpoints"
        )
        self._lock = threading.Lock()
        self._base: Dict[str, Dict[str, Any]] = {}  # model_key -> {"model":..., "features":[...]}
        self._meta: Optional[Dict[str, Any]] = None
        self._mtimes: Dict[str, float] = {}
        self._available = False
        self._last_missing_check: float = 0.0   # throttle re-checks when unavailable
        self._logged_missing: set = set()        # only log each missing file once

    # ---------------------------------------------------------------- loading

    def _file_mtime(self, name: str) -> float:
        path = self.checkpoints_dir / name
        return path.stat().st_mtime if path.exists() else 0.0

    def _needs_reload(self) -> bool:
        for key, fname in _BASE_FILES.items():
            if self._file_mtime(fname) != self._mtimes.get(fname, 0.0):
                return True
        if self._file_mtime(_META_FILE) != self._mtimes.get(_META_FILE, 0.0):
            return True
        return False

    def load(self, *, force: bool = False) -> bool:
        """Returns True if at least one base model is available."""
        if not force and not self._available:
            # Throttle: don't hammer the filesystem every inference call.
            now = time.time()
            if now - self._last_missing_check < self._MISSING_RECHECK_INTERVAL:
                return False
            self._last_missing_check = now
        if not force and self._available and not self._needs_reload():
            return True

        try:
            import joblib
        except ImportError:
            logger.error("joblib not installed; stacking inference disabled")
            return False

        with self._lock:
            new_base: Dict[str, Dict[str, Any]] = {}
            for key, fname in _BASE_FILES.items():
                path = self.checkpoints_dir / fname
                if not path.exists():
                    if fname not in self._logged_missing:
                        logger.info(
                            "stacking base %s missing: %s — will retry in %ds; "
                            "run train_stacking.py to generate models",
                            key, path, int(self._MISSING_RECHECK_INTERVAL),
                        )
                        self._logged_missing.add(fname)
                    continue
                # File appeared — remove from missing set so we log again if it
                # later disappears.
                self._logged_missing.discard(fname)
                try:
                    payload = joblib.load(path)
                    model = payload.get("model") if isinstance(payload, dict) else payload
                    features = payload.get("features") if isinstance(payload, dict) else None
                    if model is None:
                        raise ValueError("model key missing")
                    new_base[key] = {"model": model, "features": features}
                    self._mtimes[fname] = path.stat().st_mtime
                    logger.info("loaded stacking base: %s (%d features)",
                                key, len(features) if features else -1)
                except Exception as exc:
                    logger.warning("failed to load %s: %s", path, exc)

            meta_path = self.checkpoints_dir / _META_FILE
            new_meta = None
            if meta_path.exists():
                try:
                    payload = joblib.load(meta_path)
                    new_meta = {
                        "model": payload.get("model") if isinstance(payload, dict) else payload,
                        "meta_features": payload.get("meta_features") if isinstance(payload, dict) else None,
                        "kind": payload.get("kind") if isinstance(payload, dict) else None,
                    }
                    self._mtimes[_META_FILE] = meta_path.stat().st_mtime
                    logger.info("loaded stacking meta (%s)", new_meta.get("kind"))
                except Exception as exc:
                    logger.warning("failed to load meta model: %s", exc)

            self._base = new_base
            self._meta = new_meta
            self._available = len(new_base) > 0
            return self._available

    # ------------------------------------------------------ feature alignment

    @staticmethod
    def _align_features(X: Union[pd.Series, pd.DataFrame], features: Optional[Sequence[str]]) -> Optional[pd.DataFrame]:
        if X is None:
            return None
        if isinstance(X, pd.Series):
            X = X.to_frame().T
        elif not isinstance(X, pd.DataFrame):
            try:
                X = pd.DataFrame(X)
            except Exception:
                return None
        if features:
            # Ensure the exact column order & fill missing w/ 0.0 (neutral)
            X = X.reindex(columns=list(features))
            X = X.apply(pd.to_numeric, errors="coerce")
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return X

    @staticmethod
    def _take_proba(model: Any, X: pd.DataFrame) -> Optional[float]:
        try:
            probs = model.predict_proba(X)
        except Exception as exc:
            logger.debug("predict_proba failed: %s", exc)
            return None
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim == 1:
            return float(probs[0])
        if probs.shape[1] < 2:
            return float(probs[0, 0])
        classes = getattr(model, "classes_", None)
        if classes is not None:
            try:
                idx = int(np.where(np.asarray(classes) == 1)[0][0])
                return float(probs[0, idx])
            except Exception:
                pass
        return float(probs[0, 1])

    # --------------------------------------------------------------- predict

    def predict(
        self,
        features: Union[pd.Series, pd.DataFrame],
        *,
        feature_completeness_ratio: Optional[float] = None,
    ) -> Optional[PredictionResult]:
        """
        Returns a PredictionResult or None if no base models are available
        (caller should fall back to their prior XGBoost single-model path).
        """
        self.load()
        if not self._available:
            return None

        base_probs: Dict[str, float] = {}
        models_used: List[str] = []

        for key, info in self._base.items():
            X = self._align_features(features, info.get("features"))
            if X is None or X.empty:
                continue
            p = self._take_proba(info["model"], X)
            if p is None:
                continue
            base_probs[key] = float(p)
            models_used.append(key)

        if not base_probs:
            return None

        n_base = len(base_probs)

        # Partial fallback: if we don't have all 3, skip meta and use simple
        # average of whatever we do have.
        if n_base < 3 or self._meta is None:
            avg = float(np.mean(list(base_probs.values())))
            pred_class = 1 if avg >= 0.5 else 0
            proba = np.array([1 - avg, avg], dtype=np.float64)
            return PredictionResult(
                pred_class=pred_class,
                pred_proba=proba,
                take_proba=avg,
                models_used=models_used,
                meta_used=False,
                fallback_reason=f"partial_models_{n_base}_of_3",
            )

        # Full stack: build meta input and run meta
        meta_features_order = self._meta.get("meta_features") or [
            "xgb_take_p", "lgb_take_p", "cat_take_p", "feature_completeness_ratio"
        ]
        mf_row: List[float] = []
        for name in meta_features_order:
            if name == "xgb_take_p":
                mf_row.append(base_probs.get("xgb", 0.5))
            elif name == "lgb_take_p":
                mf_row.append(base_probs.get("lgb", 0.5))
            elif name == "cat_take_p":
                mf_row.append(base_probs.get("cat", 0.5))
            elif name == "feature_completeness_ratio":
                mf_row.append(float(feature_completeness_ratio if feature_completeness_ratio is not None else 1.0))
            else:
                mf_row.append(0.0)

        X_meta = np.array([mf_row], dtype=np.float64)
        meta = self._meta["model"]

        try:
            pred = int(np.asarray(meta.predict(X_meta)).ravel()[0])
        except Exception as exc:
            logger.warning("meta predict failed: %s", exc)
            avg = float(np.mean(list(base_probs.values())))
            return PredictionResult(
                pred_class=1 if avg >= 0.5 else 0,
                pred_proba=np.array([1 - avg, avg]),
                take_proba=avg,
                models_used=models_used,
                meta_used=False,
                fallback_reason="meta_predict_failed",
            )

        # Probabilities from meta (handle RidgeClassifier missing predict_proba)
        proba_arr: np.ndarray
        take_p: float
        if hasattr(meta, "predict_proba"):
            try:
                pp = np.asarray(meta.predict_proba(X_meta), dtype=np.float64).ravel()
                if pp.size >= 2:
                    classes = getattr(meta, "classes_", None)
                    if classes is not None:
                        idx = 1
                        try:
                            idx = int(np.where(np.asarray(classes) == 1)[0][0])
                        except Exception:
                            pass
                        take_p = float(pp[idx])
                    else:
                        take_p = float(pp[1])
                else:
                    take_p = float(pp[0])
                proba_arr = np.array([1 - take_p, take_p], dtype=np.float64)
            except Exception:
                take_p = float(pred)
                proba_arr = np.array([1 - take_p, take_p], dtype=np.float64)
        else:
            try:
                dfunc = float(np.asarray(meta.decision_function(X_meta)).ravel()[0])
                take_p = 1.0 / (1.0 + np.exp(-dfunc))
                proba_arr = np.array([1 - take_p, take_p], dtype=np.float64)
            except Exception:
                take_p = float(pred)
                proba_arr = np.array([1 - take_p, take_p], dtype=np.float64)

        return PredictionResult(
            pred_class=pred,
            pred_proba=proba_arr,
            take_proba=take_p,
            models_used=models_used,
            meta_used=True,
            fallback_reason="",
        )


_ENGINE: Optional[StackingInferenceEngine] = None
_ENGINE_LOCK = threading.Lock()


def get_engine() -> StackingInferenceEngine:
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = StackingInferenceEngine()
            _ENGINE.load()
        return _ENGINE


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    engine = get_engine()
    print("stacking engine available:", engine._available)
    print("base models loaded:", list(engine._base.keys()))
    print("meta loaded:", engine._meta is not None)
