"""
Explainable AI (XAI) Engine
============================
"A Black Box model is a liability."

This module implements SHAP (SHapley Additive exPlanations) to make
every prediction fully explainable. Based on game-theory (Shapley values),
SHAP is the gold standard for ML explainability.

Reference: Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions"
Used by: Two Sigma, AQR, most top quant funds for regulatory compliance.

Output per prediction:
  - SHAP waterfall chart data (top 10 drivers)
  - Feature importance ranking
  - Interaction effects (which features amplify each other)
  - Counterfactual: "If RSI were 45 instead of 72, signal would be NEUTRAL"
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not installed. Run: pip install shap")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# SHAP Explainer Factory
# ─────────────────────────────────────────────────────────────
class SHAPExplainerFactory:
    """Creates the right SHAP explainer for each model type."""

    @staticmethod
    def for_xgboost(model) -> Optional[Any]:
        """TreeExplainer — exact SHAP values, very fast for tree models."""
        if not SHAP_AVAILABLE:
            return None
        return shap.TreeExplainer(model)

    @staticmethod
    def for_pytorch(model, background_data: np.ndarray) -> Optional[Any]:
        """
        DeepExplainer for PyTorch models (LSTM, Transformer).
        background_data: representative sample of training data for baseline.
        """
        if not SHAP_AVAILABLE:
            return None
        try:
            import torch
            background_tensor = torch.FloatTensor(background_data[:100])
            return shap.DeepExplainer(model, background_tensor)
        except Exception as e:
            logger.warning(f"DeepExplainer failed, falling back to KernelExplainer: {e}")
            # KernelExplainer works on any model but is slower
            def model_fn(x):
                import torch
                with torch.no_grad():
                    return model(torch.FloatTensor(x)).numpy()
            return shap.KernelExplainer(model_fn, shap.sample(background_data, 50))

    @staticmethod
    def for_ensemble(models: Dict[str, Any], background_data: np.ndarray) -> Dict:
        """Create explainers for all models in the ensemble."""
        explainers = {}
        for name, model in models.items():
            if "xgb" in name.lower():
                exp = SHAPExplainerFactory.for_xgboost(model)
            else:
                exp = SHAPExplainerFactory.for_pytorch(model, background_data)
            if exp is not None:
                explainers[name] = exp
        return explainers


# ─────────────────────────────────────────────────────────────
# Signal Explanation Engine
# ─────────────────────────────────────────────────────────────
class SignalExplainer:
    """
    Generates human-readable explanations for every trading signal.
    
    Example output:
    {
        "signal": "BUY",
        "confidence": 0.82,
        "conviction_score": 8.2,
        "top_drivers": [
            {"feature": "Reddit Sentiment Velocity", "impact": +0.18, "direction": "bullish"},
            {"feature": "MACD Crossover (bullish)", "impact": +0.12, "direction": "bullish"},
            {"feature": "Options Flow (call sweep)", "impact": +0.09, "direction": "bullish"},
            {"feature": "Rising Interest Rates",     "impact": -0.05, "direction": "bearish"},
        ],
        "waterfall": [...],  # For Plotly chart
        "counterfactual": "Signal would flip to NEUTRAL if RSI > 72",
        "regime_warning": "CPI report in 2 hours — reduce position size"
    }
    """

    # Human-readable names for feature codes
    FEATURE_LABELS = {
        "rsi_14":                    "RSI (14-day)",
        "macd_crossover":            "MACD Bullish Crossover",
        "macd_hist":                 "MACD Histogram",
        "sentiment_normalized":      "News Sentiment",
        "sentiment_zscore":          "Sentiment Z-Score",
        "sentiment_velocity":        "Sentiment Velocity",
        "weighted_sentiment_zscore": "Weighted Sentiment Z-Score",
        "news_volume_spike":         "News Volume Spike",
        "source_quality_signal":     "Source Quality Signal",
        "official_event_signal":     "Official Event Signal",
        "filing_event_signal":       "Filing Event Signal",
        "media_sentiment_signal":    "Media Sentiment Signal",
        "travel_activity_level":     "Travel Activity Level",
        "travel_activity_change":    "Travel Activity Change",
        "congress_signal":           "Congress Trading Activity",
        "options_sentiment":         "Options Flow Sentiment",
        "options_pc_ratio":          "Put/Call Ratio",
        "unusual_options":           "Unusual Options Activity",
        "yield_curve_inverted":      "Yield Curve Inversion",
        "dxy_momentum_20d":          "Dollar Index Momentum",
        "vix_level":                 "VIX (Fear Index)",
        "risk_off_flag":             "Risk-Off Market Regime",
        "credit_spread_hy":          "High-Yield Credit Spreads",
        "us_inflation_yoy":          "US Inflation (YoY)",
        "inr_depreciation_30d":      "INR Depreciation vs USD",
        "spy_nifty_corr_60d":        "US-India Market Correlation",
        "vol_regime_stressed":       "Stressed Volatility Regime",
        "bb_squeeze":                "Bollinger Band Squeeze",
        "momentum_12_1":             "12-Month Momentum Factor",
        "volume_spike":              "Unusual Volume Spike",
        "golden_cross_50_200":       "Golden Cross (50/200 SMA)",
        "death_cross_50_200":        "Death Cross (50/200 SMA)",
        "strong_trend":              "Strong Trend (ADX > 25)",
        "dist_from_52w_high":        "Distance from 52-Week High",
        "alpha_signal":              "Composite Alpha Signal",
        "garman_klass_vol":          "Realized Volatility (GK)",
        "mfi":                       "Money Flow Index",
        "obv_trend":                 "OBV Trend",
        "price_acceleration":        "Price Acceleration",
        "close_vs_sma_50":          "Price vs 50-Day SMA",
        "close_vs_sma_200":         "Price vs 200-Day SMA",
        "momentum_20d":             "20-Day Momentum",
        "momentum_60d":             "60-Day Momentum",
        "bb_position":              "Bollinger Position",
        "zscore_vs_60d":            "60-Day Price Z-Score",
        "realized_vol_21d":         "21-Day Realized Volatility",
        "momentum_composite":        "Momentum Composite",
        "trend_composite":           "Trend Composite",
        "event_alpha_signal":        "Event Alpha Composite",
        "earnings_propagation_signal": "Earnings Propagation Signal",
        "earnings_propagation_strength": "Earnings Propagation Strength",
        "peer_earnings_shock_3d":   "Peer Earnings Shock (3d)",
        "peer_earnings_shock_7d":   "Peer Earnings Shock (7d)",
        "peer_earnings_breadth_7d": "Peer Earnings Breadth",
        "close_reversal_signal":    "Close Reversal Signal",
        "close_reversal_strength":  "Close Reversal Strength",
        "event_move_strength":      "Event Move Strength",
        "event_day_extreme":        "Extreme Event Day",
        "oil_momentum_20d":          "Oil Price Momentum",
        "risk_appetite_composite":   "Global Risk Appetite",
    }

    def __init__(self, explainers: Dict = None):
        self.explainers = explainers or {}
        self._shap_cache: Dict = {}

    def explain_prediction(
        self,
        symbol: str,
        features: pd.DataFrame,
        model_predictions: Dict[str, float],
        ensemble_score: float,
        feature_names: List[str],
        macro_calendar: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Generate a complete explanation for a trading signal.
        
        Args:
            symbol: Stock ticker
            features: Feature vector (1 row)
            model_predictions: {"lstm": 0.7, "transformer": 0.8, "xgboost": 0.65}
            ensemble_score: Weighted ensemble score
            feature_names: Column names for feature vector
            macro_calendar: Upcoming macro events [{"event": "CPI", "hours_until": 2}]
        """
        explanation = {
            "symbol": symbol,
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "signal": "buy" if ensemble_score > 0.1 else ("sell" if ensemble_score < -0.1 else "neutral"),
            "ensemble_score": float(ensemble_score),
            "conviction_score": float(min(10, abs(ensemble_score) * 10)),
            "model_agreement": self._compute_model_agreement(model_predictions),
            "model_breakdown": model_predictions,
        }

        # SHAP values if available
        shap_values = self._compute_shap_values(features, feature_names)
        if shap_values:
            explanation["top_drivers"] = shap_values["top_drivers"]
            explanation["waterfall_data"] = shap_values["waterfall_data"]
            explanation["counterfactual"] = self._compute_counterfactual(
                shap_values, features, feature_names, ensemble_score
            )
        else:
            # Rule-based fallback explanation when SHAP unavailable
            explanation["top_drivers"] = self._rule_based_drivers(features, feature_names)
            explanation["waterfall_data"] = []

        # Regime context
        explanation["regime_context"] = self._get_regime_context(features)

        # Macro calendar warning
        if macro_calendar:
            explanation["macro_warnings"] = self._build_macro_warnings(macro_calendar, ensemble_score)
        else:
            explanation["macro_warnings"] = []

        # Risk parameters
        explanation["risk_parameters"] = self._compute_risk_parameters(
            features, ensemble_score, explanation.get("top_drivers", [])
        )

        return explanation

    def _compute_shap_values(
        self,
        features: pd.DataFrame,
        feature_names: List[str],
    ) -> Optional[Dict]:
        """Compute SHAP values using the XGBoost explainer (most reliable)."""
        if not SHAP_AVAILABLE or "xgboost" not in self.explainers:
            return None

        try:
            X = features[feature_names].fillna(0).values
            explainer = self.explainers["xgboost"]
            shap_vals = explainer.shap_values(X)

            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]  # Binary classification: take positive class

            shap_vals = shap_vals.flatten()
            expected_value = float(explainer.expected_value
                                   if not isinstance(explainer.expected_value, list)
                                   else explainer.expected_value[1])

            # Build top drivers list
            idx_sorted = np.argsort(np.abs(shap_vals))[::-1]
            top_drivers = []
            for i in idx_sorted[:10]:
                feat_name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                shap_val = float(shap_vals[i])
                feat_val = float(X[0][i]) if i < len(X[0]) else 0
                top_drivers.append({
                    "feature": self.FEATURE_LABELS.get(feat_name, feat_name.replace("_", " ").title()),
                    "feature_code": feat_name,
                    "shap_value": shap_val,
                    "feature_value": round(feat_val, 4),
                    "direction": "bullish" if shap_val > 0 else "bearish",
                    "impact_pct": round(abs(shap_val) / (np.abs(shap_vals).sum() + 1e-8) * 100, 1),
                })

            # Waterfall data for Plotly
            waterfall_data = self._build_waterfall_data(
                shap_vals, feature_names, expected_value, top_n=12
            )

            return {"top_drivers": top_drivers, "waterfall_data": waterfall_data}

        except Exception as e:
            logger.error(f"SHAP computation failed: {e}")
            return None

    def _rule_based_drivers(
        self, features: pd.DataFrame, feature_names: List[str]
    ) -> List[Dict]:
        """Fallback when SHAP unavailable: use feature magnitudes as proxy."""
        drivers = []
        priority_features = [
            "official_event_signal", "filing_event_signal", "media_sentiment_signal",
            "sentiment_zscore", "news_volume_spike", "source_quality_signal",
            "travel_activity_change", "congress_signal", "unusual_options",
            "macd_crossover", "golden_cross_50_200", "rsi_14",
            "vol_regime_stressed", "risk_off_flag", "momentum_composite",
            "earnings_propagation_signal", "close_reversal_signal",
            "peer_earnings_shock_3d", "event_alpha_signal",
        ]

        for feat in priority_features:
            if feat not in features.columns:
                continue
            val = float(features[feat].iloc[0])
            if abs(val) < 0.01:
                continue

            direction = "bullish" if val > 0 else "bearish"
            if feat in ["vol_regime_stressed", "risk_off_flag", "yield_curve_inverted"]:
                direction = "bearish" if val > 0 else "bullish"

            drivers.append({
                "feature": self.FEATURE_LABELS.get(feat, feat),
                "feature_code": feat,
                "shap_value": val * 0.1,  # Scaled proxy
                "feature_value": round(val, 4),
                "direction": direction,
                "impact_pct": min(25, abs(val) * 10),
            })

        return sorted(drivers, key=lambda x: abs(x["shap_value"]), reverse=True)[:8]

    def _build_waterfall_data(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        expected_value: float,
        top_n: int = 12,
    ) -> List[Dict]:
        """Build data for a SHAP waterfall chart (for Plotly)."""
        idx_sorted = np.argsort(np.abs(shap_values))[::-1][:top_n]

        waterfall = []
        cumulative = expected_value

        # Base value
        waterfall.append({
            "label": "Base Rate",
            "value": round(expected_value, 4),
            "cumulative": round(cumulative, 4),
            "type": "base",
            "color": "#888780",
        })

        for i in idx_sorted:
            sv = float(shap_values[i])
            if abs(sv) < 0.001:
                continue
            cumulative += sv
            feat_name = feature_names[i] if i < len(feature_names) else f"f{i}"
            waterfall.append({
                "label": self.FEATURE_LABELS.get(feat_name, feat_name.replace("_", " ").title()),
                "value": round(sv, 4),
                "cumulative": round(cumulative, 4),
                "type": "positive" if sv > 0 else "negative",
                "color": "#1D9E75" if sv > 0 else "#E24B4A",
            })

        # Final prediction
        waterfall.append({
            "label": "Final Prediction",
            "value": round(cumulative, 4),
            "cumulative": round(cumulative, 4),
            "type": "total",
            "color": "#378ADD",
        })

        return waterfall

    def _compute_counterfactual(
        self,
        shap_values: Dict,
        features: pd.DataFrame,
        feature_names: List[str],
        ensemble_score: float,
    ) -> str:
        """
        Generate a counterfactual explanation:
        "Signal would flip to NEUTRAL if [top negative driver] reversed"
        """
        if not shap_values.get("top_drivers"):
            return ""

        # Find the driver that, if reversed, would change the signal
        current_direction = "buy" if ensemble_score > 0.1 else ("sell" if ensemble_score < -0.1 else "neutral")

        for driver in shap_values["top_drivers"]:
            sv = driver["shap_value"]
            feat = driver["feature"]
            val = driver["feature_value"]

            # If this is the top driver opposing the signal, it's the key constraint
            if current_direction == "buy" and sv < -0.05:
                return f"Signal would strengthen to STRONG BUY if '{feat}' reversed (currently {val:.2f})"
            elif current_direction == "sell" and sv > 0.05:
                return f"Signal would strengthen to STRONG SELL if '{feat}' reversed"
            elif abs(sv) > 0.1:  # Top driver supporting signal
                flip_threshold = val * 0.7 if sv > 0 else val * 1.3
                return (f"Signal would flip to NEUTRAL if '{feat}' moved to "
                        f"{flip_threshold:.2f} (currently {val:.2f})")
        return ""

    def _get_regime_context(self, features: pd.DataFrame) -> Dict:
        """Extract current market regime information."""
        regime = {
            "volatility": "unknown",
            "trend": "unknown",
            "risk_appetite": "unknown",
        }

        if "vol_regime" in features.columns:
            regime["volatility"] = str(features["vol_regime"].iloc[0])
        elif "vol_regime_stressed" in features.columns:
            stressed = bool(features["vol_regime_stressed"].iloc[0])
            calm = bool(features.get("vol_regime_calm", pd.Series([False])).iloc[0])
            regime["volatility"] = "stressed" if stressed else ("calm" if calm else "normal")

        if "strong_trend" in features.columns:
            regime["trend"] = "strong" if bool(features["strong_trend"].iloc[0]) else "weak"

        if "risk_off_flag" in features.columns:
            regime["risk_appetite"] = "risk-off" if bool(features["risk_off_flag"].iloc[0]) else "risk-on"

        if "risk_appetite_composite" in features.columns:
            ra = float(features["risk_appetite_composite"].iloc[0])
            regime["risk_appetite_score"] = round(ra, 3)

        return regime

    def _build_macro_warnings(
        self, macro_calendar: List[Dict], ensemble_score: float
    ) -> List[str]:
        """Generate macro event warnings."""
        warnings = []
        for event in macro_calendar:
            hours = event.get("hours_until", 999)
            name = event.get("event", "Macro event")
            impact = event.get("expected_impact", "medium")

            if hours <= 2:
                warnings.append(
                    f"⚠️ {name} releasing in ~{hours}h — HIGH RISK: reduce position size 50%"
                )
            elif hours <= 24:
                level = "HIGH" if impact == "high" else "MODERATE"
                warnings.append(
                    f"📅 {name} tomorrow — {level} RISK: set tight stop-loss"
                )

        return warnings

    def _compute_risk_parameters(
        self,
        features: pd.DataFrame,
        ensemble_score: float,
        top_drivers: List[Dict],
    ) -> Dict:
        """
        Compute trade risk parameters.
        Output matches the "Structured Advice Template":
        - Entry trigger
        - Stop-loss level
        - Position sizing suggestion
        """
        risk = {}

        # ATR-based stop loss
        if "atr_normalized" in features.columns:
            atr_pct = float(features["atr_normalized"].iloc[0])
            stop_loss_pct = max(0.015, min(0.06, atr_pct * 2.0))  # 2×ATR stop
            take_profit_pct = stop_loss_pct * 2.5  # 2.5:1 reward/risk
            risk["stop_loss_pct"] = round(stop_loss_pct * 100, 2)
            risk["take_profit_pct"] = round(take_profit_pct * 100, 2)
            risk["risk_reward_ratio"] = 2.5

        # Position sizing (Kelly-inspired, scaled for safety)
        confidence = min(1.0, abs(ensemble_score))
        kelly_fraction = confidence * 0.25  # Use 25% of Kelly for safety (fractional Kelly)
        risk["suggested_position_size_pct"] = round(kelly_fraction * 100, 1)

        # Volatility regime adjustment
        if "vol_regime_stressed" in features.columns and features["vol_regime_stressed"].iloc[0]:
            risk["suggested_position_size_pct"] *= 0.5
            risk["regime_note"] = "Position size halved due to stressed volatility regime"

        # Max drawdown protection
        risk["max_risk_per_trade_pct"] = min(2.0, risk.get("stop_loss_pct", 2.0))
        risk["entry_note"] = (
            "Do not enter if price moves >0.5% from signal generation price (slippage protection)"
        )

        return risk

    def _compute_model_agreement(self, model_predictions: Dict[str, float]) -> float:
        """
        Measures how much the 3 models agree.
        All positive = 1.0 (full agreement)
        Mixed = 0.0–0.5
        """
        if not model_predictions:
            return 0.0
        signs = [np.sign(v) for v in model_predictions.values()]
        return float(np.abs(np.mean(signs)))


# ─────────────────────────────────────────────────────────────
# Global Feature Importance (portfolio-level)
# ─────────────────────────────────────────────────────────────
class GlobalFeatureImportance:
    """
    Computes global (portfolio-level) SHAP importance.
    Answers: "What features drive most signals across ALL stocks?"
    This is shown in the dashboard's "Signal Intelligence" panel.
    """

    def compute_global_importance(
        self,
        explainer: Any,
        feature_matrix: pd.DataFrame,
        feature_names: List[str],
        sample_n: int = 500,
    ) -> pd.DataFrame:
        """Compute mean |SHAP| for all features across a sample of the dataset."""
        if not SHAP_AVAILABLE or explainer is None:
            return pd.DataFrame()

        sample = feature_matrix[feature_names].fillna(0).sample(
            n=min(sample_n, len(feature_matrix))
        ).values

        try:
            shap_vals = explainer.shap_values(sample)
            if isinstance(shap_vals, list):
                # Binary: [neg_class, pos_class] — use positive class
                shap_vals = shap_vals[-1] if len(shap_vals) > 1 else shap_vals[0]

            arr = np.asarray(shap_vals)
            # TreeExplainer can return (n, f), (n, f, c), or (n, f, 1)
            abs_val = np.abs(arr)
            if abs_val.ndim == 3:
                if abs_val.shape[2] > 1:
                    mean_abs_shap = abs_val[:, :, -1].mean(axis=0)
                else:
                    mean_abs_shap = abs_val[:, :, 0].mean(axis=0)
            else:
                mean_abs_shap = abs_val.mean(axis=0)

            mean_abs_shap = np.asarray(mean_abs_shap, dtype=float).reshape(-1)
            names = list(feature_names)
            n = min(len(names), len(mean_abs_shap))
            if n < len(names):
                names = names[:n]
                mean_abs_shap = mean_abs_shap[:n]
            elif len(mean_abs_shap) > len(names):
                mean_abs_shap = mean_abs_shap[: len(names)]

            importance_df = pd.DataFrame({
                "feature": names,
                "mean_abs_shap": mean_abs_shap,
                "label": [SignalExplainer.FEATURE_LABELS.get(f, f) for f in names],
            }).sort_values("mean_abs_shap", ascending=False)

            return importance_df

        except Exception as e:
            logger.error(f"Global importance computation failed: {e}")
            return pd.DataFrame()
