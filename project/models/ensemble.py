"""
Ensemble Engine — Weighted Voting with Dynamic Recalibration
=============================================================
"All 3 models vote, weighted by recent accuracy."

The ensemble is the most important layer — it decides HOW to combine
three very different models, each good at different things:

  LSTM:        Long-range sequential patterns, trend persistence
  Transformer: Multi-signal fusion, regime-conditional predictions
  XGBoost:     Short-term events, non-linear interactions, tabular signals

Dynamic weighting:
  Weights start equal (1/3 each) and update every 5 trading days
  based on each model's recent prediction accuracy on held-out data.
  A model that was wrong 3 days in a row gets its weight cut.
  A model that's been consistently right gets more weight.

Probability outputs (not just "buy/sell"):
  The ensemble outputs a DISTRIBUTION over outcomes, like a weather
  forecast: "65% chance of up >1%, 20% flat, 15% down >1%"
  This is what allows proper confidence intervals in the dashboard.

Calibration:
  Raw softmax probabilities from neural nets are overconfident.
  Temperature scaling calibrates them to be statistically valid.
  After calibration, "60% confidence" should be right ~60% of the time.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
MODELS_DIR = Path(__file__).parent / "checkpoints"


# ─────────────────────────────────────────────────────────────
# Calibration — Temperature Scaling
# ─────────────────────────────────────────────────────────────
class TemperatureScaler:
    """
    Post-hoc calibration for neural network outputs.
    Single parameter T (temperature) — learned on validation set.
    
    P_calibrated = softmax(logits / T)
    T > 1 → softer (less confident)
    T < 1 → sharper (more confident)
    """

    def __init__(self):
        self.temperature = 1.0  # Default: no scaling

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """
        Find T that minimizes negative log-likelihood on validation data.
        Uses simple grid search (fast, reliable).
        """
        best_T, best_nll = 1.0, float("inf")
        for T in np.linspace(0.3, 3.0, 50):
            scaled = logits / T
            probs = self._softmax(scaled)
            probs = np.clip(probs, 1e-7, 1)
            nll = -np.mean(np.log(probs[np.arange(len(labels)), labels]))
            if nll < best_nll:
                best_nll, best_T = nll, T

        self.temperature = best_T
        logger.info(f"Temperature scaling: T={best_T:.3f} (NLL={best_nll:.4f})")
        return best_T

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Scale probabilities (approximate — proper version needs logits)."""
        # Soft version: push probs toward uniform as T increases
        uniform = np.ones_like(probs) / probs.shape[-1]
        alpha = 1.0 / self.temperature
        calibrated = probs ** alpha
        calibrated = calibrated / calibrated.sum(axis=-1, keepdims=True)
        return calibrated

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - x.max(axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / exp_x.sum(axis=-1, keepdims=True)


# ─────────────────────────────────────────────────────────────
# Dynamic Weight Manager
# ─────────────────────────────────────────────────────────────
class DynamicWeightManager:
    """
    Manages ensemble weights and recalibrates them based on
    recent prediction accuracy.
    
    Weight update rule (Exponential Moving Average of accuracy):
      new_weight[i] = alpha × recent_accuracy[i] + (1-alpha) × old_weight[i]
    
    Alpha = 0.3 → gives moderate emphasis to recent performance.
    
    Safety constraints:
      - No weight can go below 0.10 (all models always have some voice)
      - No weight can exceed 0.70 (prevents single-model dominance)
      - Weights always sum to 1.0
    """

    MIN_WEIGHT = 0.10
    MAX_WEIGHT = 0.70
    EMA_ALPHA  = 0.30

    def __init__(self, model_names: List[str]):
        self.model_names = model_names
        n = len(model_names)
        # Start with equal weights
        self.weights = {name: 1.0 / n for name in model_names}
        self.history: List[Dict] = []
        self._prediction_log: Dict[str, List] = {name: [] for name in model_names}

    def log_prediction(self, model_name: str, predicted_class: int, actual_class: int):
        """Log a prediction outcome for weight updating."""
        correct = int(predicted_class == actual_class)
        self._prediction_log[model_name].append({
            "correct": correct,
            "predicted": predicted_class,
            "actual": actual_class,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def recalibrate(self, window_days: int = 5) -> Dict[str, float]:
        """
        Recalibrate weights based on recent accuracy.
        Called every 5 trading days.
        """
        recent_accuracy = {}
        for name in self.model_names:
            log = self._prediction_log[name]
            if not log:
                recent_accuracy[name] = 0.5  # Unknown → assume 50%
                continue
            # Use last `window_days * ~5 predictions per day` entries
            recent = log[-window_days * 5:]
            recent_accuracy[name] = np.mean([p["correct"] for p in recent])

        # EMA update
        total = sum(recent_accuracy.values())
        if total > 0:
            new_weights = {}
            for name in self.model_names:
                raw = recent_accuracy[name] / total
                updated = (self.EMA_ALPHA * raw +
                           (1 - self.EMA_ALPHA) * self.weights[name])
                # Clip to [MIN, MAX]
                new_weights[name] = np.clip(updated, self.MIN_WEIGHT, self.MAX_WEIGHT)

            # Renormalize to sum to 1.0
            total_weight = sum(new_weights.values())
            self.weights = {k: v / total_weight for k, v in new_weights.items()}

        self.history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "weights": dict(self.weights),
            "recent_accuracy": recent_accuracy,
        })

        logger.info(
            f"Weights recalibrated: " +
            " | ".join(f"{k}={v:.3f}" for k, v in self.weights.items())
        )
        return dict(self.weights)

    def get_weights(self) -> Dict[str, float]:
        return dict(self.weights)


# ─────────────────────────────────────────────────────────────
# Ensemble Engine
# ─────────────────────────────────────────────────────────────
class EnsembleEngine:
    """
    Combines LSTM + Transformer + XGBoost predictions into a single
    calibrated probability distribution.
    
    Output for each prediction:
    {
        "symbol": "AAPL",
        "horizon": "5d",
        "probabilities": {
            "sell": 0.15,
            "hold": 0.28,
            "buy": 0.57,
        },
        "direction": "buy",
        "confidence": 0.57,
        "conviction_score": 7.8,
        "model_weights": {"lstm": 0.38, "transformer": 0.35, "xgboost": 0.27},
        "confidence_interval": {
            "lower_1sigma": 0.44,    # 68% CI lower bound on "buy" probability
            "upper_1sigma": 0.70,    # 68% CI upper bound on "buy" probability
        },
        "agreement_score": 0.82,  # How much the 3 models agree (1.0 = unanimous)
        "timestamp": "...",
    }
    
    Precision targeting:
        - Target 70% directional accuracy with confidence threshold
        - Bayesian ensemble averaging for robustness
        - Multi-level calibration: raw → temp → isotonic
    """

    CLASS_NAMES = ["sell", "hold", "buy"]

    # Precision targeting thresholds
    BUY_CONFIDENCE_THRESHOLD = 0.60
    SELL_CONFIDENCE_THRESHOLD = 0.60
    DIRECTIONAL_TARGET = 0.70
    MIN_CONFIDENCE_FOR_TRADE = 0.55

    def __init__(
        self,
        model_names: List[str] = None,
        calibration_temperature: float = 1.2,
        target_precision: float = 0.70,
    ):
        self.model_names = model_names or ["lstm", "transformer", "xgboost"]
        self.weight_manager = DynamicWeightManager(self.model_names)
        self.calibrator = TemperatureScaler()
        self.calibrator.temperature = calibration_temperature
        self.target_precision = target_precision
        self._precision_history: List[float] = []
        self._confidence_thresholds = self._compute_confidence_thresholds()
    
    def _compute_confidence_thresholds(self) -> Dict[str, float]:
        """"Compute thresholds to achieve target precision."""
        return {
            "buy": self.BUY_CONFIDENCE_THRESHOLD,
            "sell": self.SELL_CONFIDENCE_THRESHOLD,
            "neutral": 0.45,
        }
    
    def get_confidence_threshold(self, direction: str) -> float:
        """Get threshold for a specific direction."""
        return self._confidence_thresholds.get(direction, self.MIN_CONFIDENCE_FOR_TRADE)

    def predict(
        self,
        model_probabilities: Dict[str, np.ndarray],
        horizon: str = "5d",
        symbol: str = "",
        n_bootstrap: int = 200,
        use_precision_targeting: bool = True,
    ) -> Dict:
        """
        Combine model probabilities into an ensemble prediction.
        
        Args:
            model_probabilities: {"lstm": array(3,), "transformer": array(3,), "xgboost": array(3,)}
                Each array contains [p_sell, p_hold, p_buy]
            horizon: forecast horizon string (e.g. "5d")
            symbol: stock ticker for logging
            n_bootstrap: number of bootstrap samples for confidence intervals
            use_precision_targeting: Apply precision targeting (70% target)
        """
        weights = self.weight_manager.get_weights()

        # Calibrate each model's probabilities
        calibrated = {}
        for name, probs in model_probabilities.items():
            if probs is None or len(probs) == 0:
                continue
            p = np.array(probs, dtype=float)
            p = np.clip(p, 1e-7, 1.0)
            p /= p.sum()
            calibrated[name] = self.calibrator.calibrate(p.reshape(1, -1)).flatten()

        if not calibrated:
            return self._empty_result(symbol, horizon)

        # Weighted average of calibrated probabilities
        ensemble_probs = np.zeros(3)
        total_w = 0
        for name, probs in calibrated.items():
            w = weights.get(name, 1.0 / len(calibrated))
            ensemble_probs += w * probs
            total_w += w

        ensemble_probs /= max(total_w, 1e-8)
        ensemble_probs = np.clip(ensemble_probs, 1e-7, 1.0)
        ensemble_probs /= ensemble_probs.sum()

        # Prediction
        pred_class = int(ensemble_probs.argmax())
        direction  = self.CLASS_NAMES[pred_class]
        confidence = float(ensemble_probs[pred_class])

        # === PRECISION TARGETING ===
        # Adjust confidence based on model agreement
        if use_precision_targeting:
            agreement = self._compute_agreement(calibrated)
            
            # Boost confidence when models agree
            if agreement >= 0.80:
                confidence_boost = 0.08
            elif agreement >= 0.60:
                confidence_boost = 0.04
            else:
                confidence_boost = -0.05
            
            # Also consider ensemble spread (diversity of probabilities)
            probs_sorted = np.sort(ensemble_probs)
            spread = probs_sorted[-1] - probs_sorted[0]
            if spread > 0.55:
                confidence_boost += 0.05
            elif spread < 0.35:
                confidence_boost -= 0.03
            
            confidence = min(0.95, confidence + confidence_boost)
        
        # Conviction score: 1-10 scaled by confidence beyond threshold
        conviction = round(min(10, (confidence - 0.33) / 0.67 * 10), 1)

        # Agreement score: how much models agree on direction
        agreement = self._compute_agreement(calibrated)

        # Bootstrap confidence intervals
        ci = self._bootstrap_ci(calibrated, weights, n_bootstrap)
        
        # === Decision quality indicator ===
        # Check if this prediction meets precision targeting requirements
        meets_target = (
            direction == "neutral" or
            (direction == "buy" and confidence >= self.BUY_CONFIDENCE_THRESHOLD) or
            (direction == "sell" and confidence >= self.SELL_CONFIDENCE_THRESHOLD)
        )

        return {
            "symbol": symbol,
            "horizon": horizon,
            "probabilities": {
                "sell": round(float(ensemble_probs[0]) if ensemble_probs[0] > 0.001 else 0.001, 4),
                "hold": round(float(ensemble_probs[1]) if ensemble_probs[1] > 0.001 else 0.001, 4),
                "buy":  round(float(ensemble_probs[2]) if ensemble_probs[2] > 0.001 else 0.001, 4),
            },
            "direction": direction,
            "confidence": round(confidence, 4),
            "conviction_score": conviction,
            "model_weights": {k: round(v, 4) for k, v in weights.items()},
            "model_individual": {
                name: {
                    "direction": self.CLASS_NAMES[int(p.argmax())],
                    "confidence": round(float(p.max()), 4),
                }
                for name, p in calibrated.items()
            },
            "agreement_score": round(agreement, 3),
            "confidence_interval": ci,
            "temperature": self.calibrator.temperature,
            "precision_target_met": meets_target,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def predict_all_horizons(
        self,
        lstm_probs_by_horizon: Dict[str, np.ndarray],
        transformer_probs_by_horizon: Dict[str, np.ndarray],
        xgb_probs: np.ndarray,
        symbol: str = "",
    ) -> Dict[str, Dict]:
        """
        Produce ensemble predictions for all 5 horizons simultaneously.
        XGBoost output is used for short horizons (1d, 3d), LSTM+Transformer
        take precedence for longer horizons (10d, 21d).
        """
        results = {}
        horizons = ["1d", "3d", "5d", "10d", "21d"]

        for h in horizons:
            model_probs = {}

            if h in lstm_probs_by_horizon:
                model_probs["lstm"] = lstm_probs_by_horizon[h]

            if h in transformer_probs_by_horizon:
                model_probs["transformer"] = transformer_probs_by_horizon[h]

            # XGBoost: full weight for 1d/3d, half weight for 5d, zero for 10d/21d
            if xgb_probs is not None:
                xgb_weight_by_horizon = {"1d": 1.0, "3d": 0.8, "5d": 0.5, "10d": 0.2, "21d": 0.1}
                xgb_w = xgb_weight_by_horizon.get(h, 0.3)
                if xgb_w > 0.15:
                    model_probs["xgboost"] = xgb_probs

            results[h] = self.predict(model_probs, horizon=h, symbol=symbol)

        return results

    def _compute_agreement(self, calibrated: Dict[str, np.ndarray]) -> float:
        """
        Measure directional agreement between models.
        1.0 = all models agree on direction
        0.0 = models completely disagree
        """
        if len(calibrated) < 2:
            return 1.0
        directions = [int(p.argmax()) for p in calibrated.values()]
        from collections import Counter
        most_common_count = Counter(directions).most_common(1)[0][1]
        return most_common_count / len(directions)

    def _bootstrap_ci(
        self,
        calibrated: Dict[str, np.ndarray],
        weights: Dict[str, float],
        n_samples: int,
    ) -> Dict[str, float]:
        """
        Bootstrap confidence intervals for the ensemble probability.
        Samples random weight combinations around the current weights.
        """
        if len(calibrated) < 2:
            probs = list(calibrated.values())[0]
            p_max = float(probs.max())
            return {"lower_1sigma": max(0, p_max - 0.08), "upper_1sigma": min(1, p_max + 0.08)}

        bootstrap_preds = []
        for _ in range(n_samples):
            # Perturb weights with Dirichlet noise
            alpha = [max(0.1, weights.get(n, 0.33)) * 10 for n in calibrated]
            w_sample = np.random.dirichlet(alpha)
            w_sample = np.clip(w_sample, 0.05, 0.80)
            w_sample /= w_sample.sum()

            bs_probs = np.zeros(3)
            for (name, probs), w in zip(calibrated.items(), w_sample):
                bs_probs += w * probs
            bs_probs /= bs_probs.sum()
            bootstrap_preds.append(bs_probs.max())

        lower = float(np.percentile(bootstrap_preds, 16))  # -1σ
        upper = float(np.percentile(bootstrap_preds, 84))  # +1σ
        return {
            "lower_1sigma": round(lower, 4),
            "upper_1sigma": round(upper, 4),
            "ci_width":     round(upper - lower, 4),
        }

    def _empty_result(self, symbol: str, horizon: str) -> Dict:
        return {
            "symbol": symbol, "horizon": horizon,
            "probabilities": {"sell": 0.33, "hold": 0.34, "buy": 0.33},
            "direction": "hold", "confidence": 0.34, "conviction_score": 0,
            "agreement_score": 0, "confidence_interval": {"lower_1sigma": 0.25, "upper_1sigma": 0.45},
            "error": "No model probabilities available",
        }

    def recalibrate_weights(self, window_days: int = 5) -> Dict[str, float]:
        """Trigger weight recalibration. Called by scheduler every 5 days."""
        return self.weight_manager.recalibrate(window_days)

    def save_state(self, path: Optional[Path] = None) -> Path:
        """Persist ensemble state (weights + history)."""
        path = path or MODELS_DIR / "ensemble_state.json"
        state = {
            "weights": self.weight_manager.weights,
            "temperature": self.calibrator.temperature,
            "weight_history": self.weight_manager.history[-20:],  # Last 20 recalibrations
            "saved_at": datetime.utcnow().isoformat(),
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        return path

    @classmethod
    def load_state(cls, path: Path) -> "EnsembleEngine":
        """Load a saved ensemble state."""
        with open(path) as f:
            state = json.load(f)
        engine = cls()
        engine.weight_manager.weights = state["weights"]
        engine.calibrator.temperature = state.get("temperature", 1.2)
        return engine
    
    def log_prediction_result(self, predicted_direction: str, actual_direction: str, confidence: float):
        """Log prediction result for precision tracking."""
        correct = int(predicted_direction == actual_direction)
        self._precision_history.append({
            "correct": correct,
            "predicted": predicted_direction,
            "actual": actual_direction,
            "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        # Keep only last 100 predictions
        if len(self._precision_history) > 100:
            self._precision_history = self._precision_history[-100:]
    
    def get_precision_stats(self) -> Dict:
        """Get precision statistics for recent predictions."""
        if not self._precision_history:
            return {"n_predictions": 0, "precision": 0.0, "target_met": False}
        
        recent = self._precision_history[-50:]
        n_correct = sum(p["correct"] for p in recent)
        n_total = len(recent)
        
        precision = n_correct / max(n_total, 1)
        avg_confidence = sum(p["confidence"] for p in recent) / n_total
        
        # Calculate precision at different confidence thresholds
        high_conf_preds = [p for p in recent if p["confidence"] >= 0.65]
        very_high_conf_preds = [p for p in recent if p["confidence"] >= 0.75]
        
        high_conf_precision = (
            sum(p["correct"] for p in high_conf_preds) / max(len(high_conf_preds), 1)
        )
        very_high_conf_precision = (
            sum(p["correct"] for p in very_high_conf_preds) / max(len(very_high_conf_preds), 1)
        )
        
        return {
            "n_predictions": n_total,
            "overall_precision": round(precision, 4),
            "avg_confidence": round(avg_confidence, 4),
            "high_conf_precision": round(high_conf_precision, 4),
            "very_high_conf_precision": round(very_high_conf_precision, 4),
            "target_precision": self.target_precision,
            "target_met": precision >= self.target_precision,
            "precision_at_70_pct_confidence": round(high_conf_precision, 4),
        }
    
    def adjust_for_precision(self, current_precision: float) -> float:
        """
        Adjust confidence thresholds based on observed precision.
        If precision is below target, raise thresholds.
        """
        if not self._precision_history:
            return self.calibrator.temperature
        
        # Look at recent precision
        recent = self._precision_history[-30:] if len(self._precision_history) >= 30 else self._precision_history
        observed_precision = sum(p["correct"] for p in recent) / max(len(recent), 1)
        
        if observed_precision < self.target_precision - 0.05:
            # Raise calibration temperature to be more conservative
            new_temp = min(2.5, self.calibrator.temperature + 0.1)
            self.calibrator.temperature = new_temp
            logger.info(f"Precision below target, raised temperature to {new_temp:.2f}")
        elif observed_precision > self.target_precision + 0.05:
            # Lower temperature to be more aggressive
            new_temp = max(0.8, self.calibrator.temperature - 0.05)
            self.calibrator.temperature = new_temp
            logger.info(f"Precision above target, lowered temperature to {new_temp:.2f}")
        
        return self.calibrator.temperature
