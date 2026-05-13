#!/usr/bin/env python3
"""
Trade Gate — stacking inference + confidence scoring (single entry-point)
===========================================================================
Called from run.py right before an order is submitted. Wraps:

  1. Stacking inference (3 base + meta) via core.stacking_inference
  2. Trade confidence scoring + size multiplier via core.confidence_score
  3. Feature completeness computation (row-level)

Always returns a GateDecision. NEVER raises. If stacking is unavailable,
the caller's existing base-model probability (e.g. legacy XGBoost) is used
as the meta-probability input, preserving the "always execute when possible"
philosophy.

Usage (in run.py)::

    from core.trade_gate import evaluate_trade_gate, GateDecision

    decision: GateDecision = evaluate_trade_gate(
        symbol="RELIANCE",
        lane="day",                  # "day" | "normal" | "crypto"
        features_row=latest_row,     # pandas Series or 1-row DataFrame
        legacy_base_proba=xgb_confidence,   # fallback if stacking unavail.
        residual_used_fallback=res.used_fallback,
        finbert_neutral=(finbert_score == 0.0),
        options_unavailable=(opts.get("options_data_available", 0) == 0),
        stale_data=stale_flag,
        sector_unavailable=(lane == "day" and not has_sector_rs),
        net_vanna=opts.get("net_vanna", 0.0),
        iv_falling=iv_falling_flag,
        distance_to_pin=opts.get("distance_to_pin", 1.0),
        expected_features=expected_feature_list,
    )
    if decision.should_execute:
        final_qty = int(intended_qty * decision.size_multiplier)
        ...
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .confidence_score import (
    ConfidenceInputs,
    ConfidenceResult,
    compute_feature_completeness,
    trade_confidence_score,
)
from .stacking_inference import PredictionResult, get_engine

logger = logging.getLogger(__name__)


@dataclass
class GateDecision:
    should_execute: bool
    size_multiplier: float
    final_score: float
    base_probability: float
    prediction: Optional[PredictionResult]
    confidence: ConfidenceResult
    feature_completeness_ratio: float
    reasons: List[str] = field(default_factory=list)
    stacking_used: bool = False
    meta_used: bool = False
    models_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_execute": self.should_execute,
            "size_multiplier": self.size_multiplier,
            "final_score": self.final_score,
            "base_probability": self.base_probability,
            "feature_completeness_ratio": self.feature_completeness_ratio,
            "stacking_used": self.stacking_used,
            "meta_used": self.meta_used,
            "models_used": self.models_used,
            "stacking_fallback_reason": (
                self.prediction.fallback_reason
                if self.prediction is not None and self.prediction.fallback_reason
                else ("stacking_unavailable" if not self.stacking_used else "")
            ),
            "stacking_prediction_class": self.prediction.pred_class if self.prediction is not None else None,
            "penalties": self.confidence.penalties_applied,
            "bonuses": self.confidence.bonuses_applied,
            "reasons": self.reasons,
            "reason": self.confidence.reason,
        }


def _as_series(row: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    if isinstance(row, pd.DataFrame):
        if row.empty:
            return pd.Series(dtype=float)
        return row.iloc[-1]
    if isinstance(row, pd.Series):
        return row
    try:
        return pd.Series(row)
    except Exception:
        return pd.Series(dtype=float)


def evaluate_trade_gate(
    *,
    symbol: str,
    lane: str,
    features_row: Union[pd.Series, pd.DataFrame],
    legacy_base_proba: Optional[float] = None,
    expected_features: Optional[List[str]] = None,
    residual_used_fallback: bool = False,
    finbert_neutral: bool = False,
    options_unavailable: bool = False,
    stale_data: bool = False,
    sector_unavailable: bool = False,
    net_vanna: float = 0.0,
    iv_falling: bool = False,
    distance_to_pin: float = 1.0,
) -> GateDecision:
    """
    Compute full trade-gate decision. Never raises.
    """
    reasons: List[str] = []

    row = _as_series(features_row)
    if row.empty:
        logger.debug("trade_gate: empty features row for %s", symbol)

    # Feature completeness
    if expected_features:
        # Use a dict view so compute_feature_completeness() can inspect values
        available = {k: row.get(k) for k in expected_features}
        completeness = compute_feature_completeness(available, expected_features=expected_features)
    else:
        available = row.to_dict() if not row.empty else {}
        completeness = compute_feature_completeness(available)

    # Stacking inference
    stack_pred: Optional[PredictionResult] = None
    base_proba = float(legacy_base_proba or 0.0)
    stacking_used = False
    meta_used = False
    models_used: List[str] = []

    try:
        engine = get_engine()
        stack_pred = engine.predict(row, feature_completeness_ratio=completeness)
    except Exception as exc:
        logger.warning("stacking inference raised for %s: %s", symbol, exc)
        stack_pred = None

    if stack_pred is not None:
        stacking_used = True
        meta_used = stack_pred.meta_used
        models_used = list(stack_pred.models_used)
        base_proba = float(stack_pred.take_proba)
        reasons.append(
            f"stacking meta_used={meta_used} base_models={len(models_used)} take_p={base_proba:.3f}"
        )
        if stack_pred.fallback_reason:
            reasons.append(f"stacking_fallback={stack_pred.fallback_reason}")
    else:
        reasons.append(f"stacking_unavailable; using legacy_base={base_proba:.3f}")

    # Confidence score
    conf_inputs = ConfidenceInputs(
        meta_probability=base_proba,
        residual_momentum_fallback=residual_used_fallback,
        finbert_neutral=finbert_neutral,
        options_unavailable=options_unavailable,
        stale_data=stale_data,
        feature_completeness_ratio=completeness,
        sector_unavailable=sector_unavailable,
        net_vanna=float(net_vanna or 0.0),
        iv_falling=bool(iv_falling),
        distance_to_pin=float(distance_to_pin or 1.0),
        symbol=symbol,
        lane=lane,
    )
    try:
        confidence = trade_confidence_score(conf_inputs)
    except Exception as exc:
        logger.warning("confidence scoring failed for %s: %s", symbol, exc)
        # Build a conservative "skip" confidence result
        confidence = ConfidenceResult(
            final_score=0.0,
            base_score=base_proba,
            penalties_applied={},
            bonuses_applied={},
            size_multiplier=0.0,
            should_execute=False,
            reason=f"confidence_error: {exc}",
            symbol=symbol,
            lane=lane,
        )

    return GateDecision(
        should_execute=confidence.should_execute,
        size_multiplier=confidence.size_multiplier,
        final_score=confidence.final_score,
        base_probability=base_proba,
        prediction=stack_pred,
        confidence=confidence,
        feature_completeness_ratio=completeness,
        reasons=reasons,
        stacking_used=stacking_used,
        meta_used=meta_used,
        models_used=models_used,
    )
