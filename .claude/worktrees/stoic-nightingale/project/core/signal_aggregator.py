"""
Signal Aggregator - Combines Multiple Models
============================================
Aggregates predictions from:
1. Stacking Ensemble (LightGBM + CatBoost + XGBoost)
2. XGBoost directional model
3. Meta model (take/skip decision)

Uses weighted averaging with agreement filtering.
Optionally uses AdaptiveEnsembleRouter for dynamic, regime-aware weights.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "checkpoints"


class SignalAggregator:
    """
    Combines multiple model predictions for robust signals.

    Key features:
    - Weighted ensemble of 3+ models
    - Agreement filtering (require consensus)
    - Confidence-based weighting
    - Regime-aware model selection
    - Dynamic weight adaptation via AdaptiveEnsembleRouter
    """

    MODEL_WEIGHTS = {
        "stacking": 0.45,
        "xgboost": 0.35,
        "meta": 0.20,
    }

    def __init__(self, min_agreement: float = 0.66, router_state_path: str = "data/router_state.json"):
        self.min_agreement = min_agreement
        self.router_state_path = router_state_path
        self._models_loaded = False

        # Load adaptive router — fall back to None on any failure
        self.router = None
        try:
            from models.adaptive_router import AdaptiveEnsembleRouter
            self.router = AdaptiveEnsembleRouter.load(router_state_path)
            logger.info("SignalAggregator: AdaptiveEnsembleRouter loaded")
        except Exception as e:
            logger.warning(f"SignalAggregator: Router unavailable, using static weights: {e}")

        self._load_models()

    def _load_models(self):
        """Load available models."""
        self._stacking_model = None
        self._xgb_model = None
        self._meta_model = None

        stacking_path = MODELS_DIR / "stacking_ensemble.pkl"
        if stacking_path.exists():
            try:
                import pickle
                with open(stacking_path, "rb") as f:
                    self._stacking_model = pickle.load(f)
                logger.info("SignalAggregator: Stacking model loaded")
            except Exception as e:
                logger.warning(f"SignalAggregator: Failed to load stacking: {e}")

        self._models_loaded = True

    def _get_weights(self, regime: str, prediction_keys: List[str]) -> Dict[str, float]:
        """
        Get model weights — dynamic from router if available, else static.

        The router may track different model names (xgboost/lstm/transformer)
        than the aggregator's prediction keys (stacking/xgboost/meta).
        For any prediction key that the router doesn't cover, the static
        weight is used.  If the router fails entirely, all static weights
        are returned.
        """
        if self.router is not None:
            try:
                router_weights = self.router.get_weights(regime)
            except Exception as e:
                logger.debug(f"Router get_weights failed, using static: {e}")
                router_weights = {}
        else:
            router_weights = {}

        weights = {}
        for key in prediction_keys:
            if key in router_weights:
                weights[key] = router_weights[key]
            else:
                weights[key] = self.MODEL_WEIGHTS.get(key, 1.0 / max(len(prediction_keys), 1))

        # Normalize to sum=1
        total = sum(weights.values())
        if total > 1e-12:
            weights = {k: v / total for k, v in weights.items()}
        else:
            equal = 1.0 / max(len(prediction_keys), 1)
            weights = {k: equal for k in prediction_keys}

        return weights

    def predict(self, features: pd.DataFrame, regime: str = "normal", trade_id: Optional[str] = None) -> Dict:
        """
        Generate aggregated signal from multiple models.

        Args:
            features: Input feature DataFrame.
            regime: Current market regime (calm/normal/stressed/crisis).
            trade_id: If provided, registers each model's prediction with
                      the adaptive router for later outcome tracking.

        Returns:
            {
                "direction": "buy/sell/hold",
                "confidence": 0.75,
                "agreeing_models": ["stacking", "xgboost"],
                "take_probability": 0.68,
                "risk_level": "low",
            }
        """
        if not self._models_loaded:
            return self._default_signal()

        predictions = {}

        if self._stacking_model is not None:
            try:
                stack_result = self._stacking_model.predict(features)
                predictions["stacking"] = {
                    "direction": stack_result.get("directions", ["hold"])[0] if stack_result.get("directions") else "hold",
                    "confidence": stack_result.get("confidence", [0.33])[0] if stack_result.get("confidence") else 0.33,
                    "probs": stack_result.get("probabilities", {}),
                }
            except Exception as e:
                logger.debug(f"Stacking prediction error: {e}")

        if not predictions:
            return self._default_signal()

        # Get dynamic weights from router (falls back to static)
        weights = self._get_weights(regime, list(predictions.keys()))

        # Weighted average confidence
        avg_confidence = sum(
            weights.get(k, 0) * p["confidence"] for k, p in predictions.items()
        )

        # Weighted direction voting
        direction_scores: Dict[str, float] = {}
        for model_name, p in predictions.items():
            d = p["direction"]
            w = weights.get(model_name, 0)
            direction_scores[d] = direction_scores.get(d, 0) + w

        agreeing = max(direction_scores, key=direction_scores.get)
        agreement_rate = direction_scores[agreeing]  # already normalized to sum=1

        if agreement_rate >= self.min_agreement:
            final_direction = agreeing
        else:
            final_direction = "hold"

        # Register predictions with adaptive router for outcome tracking
        if trade_id is not None and self.router is not None:
            try:
                self.router.register_predictions(
                    trade_id=trade_id,
                    regime=regime,
                    model_predictions={k: v["direction"] for k, v in predictions.items()},
                )
            except Exception as e:
                logger.debug(f"Router register_predictions failed: {e}")

        return {
            "direction": final_direction,
            "confidence": round(avg_confidence * agreement_rate, 3),
            "agreeing_models": [k for k, v in predictions.items() if v["direction"] == final_direction],
            "all_predictions": {k: v["direction"] for k, v in predictions.items()},
            "model_weights": weights,
            "take_probability": min(0.95, avg_confidence * (1 + agreement_rate * 0.3)),
            "risk_level": "low" if agreement_rate >= 0.8 else "medium" if agreement_rate >= 0.6 else "high",
        }

    def record_outcome(self, trade_result: dict) -> None:
        """
        Forward a closed-trade result to the adaptive router so it can
        update per-model accuracy and reweight.

        Args:
            trade_result: dict with trade_id/order_id and either
                          actual_direction or realized_pnl + side.
        """
        if self.router is None:
            return
        try:
            self.router.record_outcome(trade_result)
            self.router.save(self.router_state_path)
        except Exception as e:
            logger.warning(f"SignalAggregator: router record_outcome failed: {e}")

    def _default_signal(self) -> Dict:
        """Return default signal when models unavailable."""
        return {
            "direction": "hold",
            "confidence": 0.33,
            "agreeing_models": [],
            "all_predictions": {},
            "model_weights": {},
            "take_probability": 0.33,
            "risk_level": "high",
        }

    def predict_batch(self, features_dict: Dict[str, pd.DataFrame], regime: str = "normal") -> Dict[str, Dict]:
        """Predict for multiple symbols."""
        results = {}
        for symbol, features in features_dict.items():
            if features is not None and len(features) > 0:
                results[symbol] = self.predict(features, regime=regime)
            else:
                results[symbol] = self._default_signal()
        return results


def get_signal_aggregator(router_state_path: str = "data/router_state.json") -> SignalAggregator:
    """Factory function."""
    return SignalAggregator(router_state_path=router_state_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    aggregator = get_signal_aggregator()
    print("Signal Aggregator initialized")
    print(f"  Models loaded: {aggregator._models_loaded}")
    print(f"  Router active: {aggregator.router is not None}")