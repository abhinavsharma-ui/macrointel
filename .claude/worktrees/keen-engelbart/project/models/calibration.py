"""
Probability calibration helpers.

These utilities keep calibration artifacts lightweight and serializable so
they can be reused by both retraining jobs and the live runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression


def _clip_probabilities(values) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)


def _logit(values) -> np.ndarray:
    clipped = _clip_probabilities(values)
    return np.log(clipped / (1.0 - clipped))


@dataclass
class BinaryPlattCalibrator:
    """Classic sigmoid calibration on top of model probabilities."""

    use_logit: bool = True
    model: Optional[LogisticRegression] = None
    fitted: bool = False

    def fit(self, raw_probabilities, labels) -> "BinaryPlattCalibrator":
        y = np.asarray(labels, dtype=int)
        if y.size == 0 or np.unique(y).size < 2:
            self.model = None
            self.fitted = False
            return self

        raw = _clip_probabilities(raw_probabilities)
        X = _logit(raw).reshape(-1, 1) if self.use_logit else raw.reshape(-1, 1)

        classifier = LogisticRegression(solver="lbfgs", max_iter=200)
        classifier.fit(X, y)
        self.model = classifier
        self.fitted = True
        return self

    def transform(self, raw_probabilities) -> np.ndarray:
        raw = _clip_probabilities(raw_probabilities)
        if not self.fitted or self.model is None:
            return raw
        X = _logit(raw).reshape(-1, 1) if self.use_logit else raw.reshape(-1, 1)
        return self.model.predict_proba(X)[:, 1]


@dataclass
class MultiClassPlattCalibrator:
    """One-vs-rest Platt scaling with row renormalization."""

    calibrators: Dict[int, BinaryPlattCalibrator] = field(default_factory=dict)
    n_classes: int = 0
    fitted: bool = False

    def fit(self, raw_probabilities, labels) -> "MultiClassPlattCalibrator":
        probs = np.asarray(raw_probabilities, dtype=float)
        y = np.asarray(labels, dtype=int)
        if probs.ndim != 2 or probs.shape[0] == 0:
            self.calibrators = {}
            self.n_classes = 0
            self.fitted = False
            return self

        self.n_classes = probs.shape[1]
        self.calibrators = {}
        for class_index in range(self.n_classes):
            calibrator = BinaryPlattCalibrator()
            calibrator.fit(probs[:, class_index], (y == class_index).astype(int))
            self.calibrators[class_index] = calibrator
        self.fitted = True
        return self

    def transform(self, raw_probabilities) -> np.ndarray:
        probs = np.asarray(raw_probabilities, dtype=float)
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
        probs = _clip_probabilities(probs)
        if not self.fitted or not self.calibrators:
            row_sums = probs.sum(axis=1, keepdims=True)
            return probs / np.where(row_sums <= 0, 1.0, row_sums)

        calibrated = np.zeros_like(probs, dtype=float)
        for class_index in range(probs.shape[1]):
            calibrator = self.calibrators.get(class_index)
            if calibrator is None:
                calibrated[:, class_index] = probs[:, class_index]
            else:
                calibrated[:, class_index] = calibrator.transform(probs[:, class_index])

        row_sums = calibrated.sum(axis=1, keepdims=True)
        fallback = np.where(row_sums <= 0, probs, calibrated)
        row_sums = fallback.sum(axis=1, keepdims=True)
        return fallback / np.where(row_sums <= 0, 1.0, row_sums)
