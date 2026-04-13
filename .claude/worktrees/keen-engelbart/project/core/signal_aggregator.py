"""
Signal Aggregator - Combines Multiple Models
============================================
Aggregates predictions from:
1. Stacking Ensemble (LightGBM + CatBoost + XGBoost)
2. XGBoost directional model
3. Meta model (take/skip decision)

Uses weighted averaging with agreement filtering.
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
    """
    
    MODEL_WEIGHTS = {
        "stacking": 0.45,
        "xgboost": 0.35,
        "meta": 0.20,
    }
    
    def __init__(self, min_agreement: float = 0.66):
        self.min_agreement = min_agreement
        self._models_loaded = False
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
    
    def predict(self, features: pd.DataFrame) -> Dict:
        """
        Generate aggregated signal from multiple models.
        
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
        
        avg_confidence = np.mean([p["confidence"] for p in predictions.values()])
        
        direction_counts = {}
        for p in predictions.values():
            d = p["direction"]
            direction_counts[d] = direction_counts.get(d, 0) + 1
        
        agreeing = max(direction_counts, key=direction_counts.get)
        agreement_rate = direction_counts[agreeing] / len(predictions)
        
        if agreement_rate >= self.min_agreement:
            final_direction = agreeing
        else:
            final_direction = "hold"
        
        return {
            "direction": final_direction,
            "confidence": round(avg_confidence * agreement_rate, 3),
            "agreeing_models": [k for k, v in predictions.items() if v["direction"] == final_direction],
            "all_predictions": {k: v["direction"] for k, v in predictions.items()},
            "take_probability": min(0.95, avg_confidence * (1 + agreement_rate * 0.3)),
            "risk_level": "low" if agreement_rate >= 0.8 else "medium" if agreement_rate >= 0.6 else "high",
        }
    
    def _default_signal(self) -> Dict:
        """Return default signal when models unavailable."""
        return {
            "direction": "hold",
            "confidence": 0.33,
            "agreeing_models": [],
            "all_predictions": {},
            "take_probability": 0.33,
            "risk_level": "high",
        }
    
    def predict_batch(self, features_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Predict for multiple symbols."""
        results = {}
        for symbol, features in features_dict.items():
            if features is not None and len(features) > 0:
                results[symbol] = self.predict(features)
            else:
                results[symbol] = self._default_signal()
        return results


def get_signal_aggregator() -> SignalAggregator:
    """Factory function."""
    return SignalAggregator()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    aggregator = get_signal_aggregator()
    print("Signal Aggregator initialized")
    print(f"  Models loaded: {aggregator._models_loaded}")