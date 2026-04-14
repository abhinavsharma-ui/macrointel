"""
Signal Generator
=================
Produces institutional-grade trading signals with:
  - XGBoost model probability (trained on real data)
  - Entry price, stop loss, take profit levels
  - Position sizing (Kelly criterion, risk-adjusted)
  - Conviction score (1-10)
  - Human-readable reasoning

Output per signal:
    symbol, signal (buy/sell/neutral), confidence,
    entry_price, stop_loss, take_profit,
    conviction_score, position_size_pct,
    reasoning, top_drivers, risk_reward_ratio
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
SIGNALS = []
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Risk calculator
# ─────────────────────────────────────────────────────────────

def _atr_stops(
    close: float,
    atr: float,
    signal: str,
    atr_stop_mult: float = 2.0,
    atr_tp_mult: float   = 3.5,
) -> Dict:
    """Compute ATR-based stop loss and take profit levels."""
    if signal == "buy":
        stop_loss   = round(close - atr_stop_mult * atr, 2)
        take_profit = round(close + atr_tp_mult  * atr, 2)
        stop_pct    = round((close - stop_loss)   / close * 100, 2)
        tp_pct      = round((take_profit - close) / close * 100, 2)
    elif signal == "sell":
        stop_loss   = round(close + atr_stop_mult * atr, 2)
        take_profit = round(close - atr_tp_mult  * atr, 2)
        stop_pct    = round((stop_loss - close)   / close * 100, 2)
        tp_pct      = round((close - take_profit) / close * 100, 2)
    else:
        stop_loss = take_profit = close
        stop_pct  = tp_pct = 0.0

    rr = round(tp_pct / max(stop_pct, 0.01), 2)

    return {
        "stop_loss":          stop_loss,
        "take_profit":        take_profit,
        "stop_loss_pct":      stop_pct,
        "take_profit_pct":    tp_pct,
        "risk_reward_ratio":  rr,
    }


def _kelly_position_size(
    win_prob: float,
    rr_ratio: float,
    max_position_pct: float = 10.0,
) -> float:
    """
    Kelly criterion position size.
    f = (p * b - q) / b
    where p = win_prob, q = 1-p, b = risk:reward ratio
    Using half-Kelly for safety.
    """
    if rr_ratio <= 0 or win_prob <= 0:
        return 1.0
    q      = 1 - win_prob
    kelly  = (win_prob * rr_ratio - q) / rr_ratio
    half_k = kelly / 2   # Half Kelly is safer for live trading
    pct    = max(0, min(half_k * 100, max_position_pct))
    return round(pct, 1)


# ─────────────────────────────────────────────────────────────
# Signal builder
# ─────────────────────────────────────────────────────────────

def _build_reasoning(symbol: str, signal: str, features: pd.Series, model_used: str) -> str:
    """Generate human-readable explanation of the signal."""
    reasons = []

    rsi = features.get("rsi_14", 50)
    if signal == "buy"  and rsi < 40: reasons.append(f"RSI oversold ({rsi:.0f})")
    if signal == "sell" and rsi > 65: reasons.append(f"RSI overbought ({rsi:.0f})")

    macd_h = features.get("macd_hist", 0)
    if signal == "buy"  and macd_h > 0: reasons.append("MACD bullish crossover")
    if signal == "sell" and macd_h < 0: reasons.append("MACD bearish crossover")

    golden = features.get("golden_cross", 0)
    if golden == 1: reasons.append("Above 200-day MA (uptrend)")
    if golden == 0: reasons.append("Below 200-day MA (downtrend)")

    bb_pct = features.get("bb_pct", 0.5)
    if signal == "buy"  and bb_pct < 0.2: reasons.append("Near Bollinger lower band")
    if signal == "sell" and bb_pct > 0.8: reasons.append("Near Bollinger upper band")

    vol_z = features.get("vol_zscore", 0)
    if abs(vol_z) > 1.5: reasons.append(f"High volume spike (z={vol_z:.1f})")

    sent = features.get("compound_score", 0)
    if sent > 0.1:  reasons.append(f"Positive news sentiment ({sent:.2f})")
    if sent < -0.1: reasons.append(f"Negative news sentiment ({sent:.2f})")

    roc = features.get("roc_10", 0)
    if abs(roc) > 0.03: reasons.append(f"Strong 10-day momentum ({roc*100:.1f}%)")

    if not reasons:
        reasons.append(f"Composite alpha signal ({model_used})")

    return " · ".join(reasons[:4])


def _get_top_drivers(features: pd.Series, signal: str, n: int = 8) -> List[Dict]:
    """
    Return top feature contributors as waterfall chart data.
    Simple importance proxy: feature value × direction weight.
    (Real SHAP values computed separately by explainability module.)
    """
    direction = 1 if signal == "buy" else -1

    importance_weights = {
        "rsi_14":            0.18,
        "macd_hist":         0.15,
        "momentum_composite":0.14,
        "alpha_signal":      0.13,
        "bb_pct":            0.10,
        "roc_10":            0.09,
        "vol_ratio_20":      0.07,
        "obv_slope":         0.06,
        "compound_score":    0.05,
        "golden_cross":      0.04,
        "atr_pct":           0.03,
        "52w_high_ratio":    0.03,
        "vol_regime":        0.02,
        "ema_slope_9":       0.02,
    }

    drivers = []
    for feat, weight in importance_weights.items():
        val = features.get(feat, 0)
        if pd.isna(val):
            val = 0
        shap_val = round(float(val) * weight * direction, 4)
        drivers.append({
            "feature":    feat,
            "label":      feat.replace("_", " ").title(),
            "value":      round(float(val), 4),
            "shap_value": shap_val,
        })

    drivers.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    return drivers[:n]


# ─────────────────────────────────────────────────────────────
# Main signal generator
# ─────────────────────────────────────────────────────────────

class SignalGenerator:
    """
    Generates full trading signals for all symbols in the universe.

    Usage:
        gen = SignalGenerator()
        gen.load_models()
        signals = gen.generate_all(feature_matrices, price_data)
    """

    def __init__(self):
        from pipeline.model_trainer import ModelPredictor
        self.predictor = ModelPredictor()
        self._models_loaded = False

    def load_models(self) -> bool:
        self._models_loaded = self.predictor.load()
        return self._models_loaded

    def generate_all(
        self,
        feature_matrices: Dict[str, pd.DataFrame],
        price_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, Dict]:
        """
        Generate signals for all symbols.
        Returns {symbol: signal_dict}
        """
        signals = {}
        global SIGNALS
        SIGNALS.clear()

        for symbol, features in feature_matrices.items():
            if features.empty:
                continue
            try:
                sig = self.generate_one(symbol, features, price_data.get(symbol))
                if sig:
                    signals[symbol] = sig
                    SIGNALS.append(sig)
            except Exception as e:
                logger.error(f"Signal error for {symbol}: {e}")

        buys  = sum(1 for s in signals.values() if s.get("signal") == "buy")
        sells = sum(1 for s in signals.values() if s.get("signal") == "sell")
        logger.info(f"Signals generated: {len(signals)} total ({buys} buy, {sells} sell)")
        return signals

    def generate_one(
        self,
        symbol: str,
        features: pd.DataFrame,
        price_df: Optional[pd.DataFrame],
    ) -> Optional[Dict]:
        """Generate a full signal dict for one symbol."""
        if features.empty:
            return None

        latest   = features.iloc[-1]
        pred     = self.predictor.predict(symbol, features)
        signal   = pred["signal"]
        conf     = pred["confidence"]

        # Current price
        close = float(price_df["close"].iloc[-1]) if price_df is not None and not price_df.empty else 0.0

        # ATR for stop/target calculation
        atr_pct = float(latest.get("atr_pct", 0.02))
        atr     = close * atr_pct if close > 0 else 1.0

        # Stop loss & take profit
        risk_params = _atr_stops(close, atr, signal)

        # Position size via Kelly
        pos_size = _kelly_position_size(
            win_prob=conf,
            rr_ratio=risk_params["risk_reward_ratio"],
        ) if signal != "neutral" else 0.0

        # Conviction score (1-10) — weighted blend so the score spreads across the
        # full range. Capping the momentum term prevents it from saturating the
        # score whenever momentum_composite happens to be extreme.
        base_conviction = conf * 7.0
        momentum_bonus  = min(abs(float(latest.get("momentum_composite", 0))) * 1.3, 2.0)
        conviction      = round(min(base_conviction + momentum_bonus, 10), 1)

        # Market regime
        trend   = "bull" if float(latest.get("golden_cross", 0)) == 1 else "bear"
        vol_reg = "stressed" if float(latest.get("vol_regime", 0)) == 1 else "calm"
        regime  = {"trend": trend, "volatility": vol_reg}

        # Human reasoning
        reasoning = _build_reasoning(symbol, signal, latest, pred["model_used"])

        # Top SHAP-like drivers
        drivers = _get_top_drivers(latest, signal)

        # Model breakdown
        model_breakdown = {
            "xgboost": round(pred["buy_prob"] - 0.5, 3),
            "momentum": round(float(latest.get("momentum_composite", 0)) * 0.5, 3),
            "sentiment": round(float(latest.get("compound_score", 0)) * 0.3, 3),
        }

        # Macro warnings
        warnings = []
        if float(latest.get("hist_vol_30", 0)) > 0.4:
            warnings.append("⚠ High historical volatility — reduce position size")
        if vol_reg == "stressed" and signal == "buy":
            warnings.append("⚠ Volatile market regime — tighten stop loss")
        if float(latest.get("52w_high_ratio", 1)) > 0.98 and signal == "buy":
            warnings.append("⚠ Near 52-week high — watch for resistance")
        if float(latest.get("52w_low_ratio", 1)) < 1.05 and signal == "sell":
            warnings.append("⚠ Near 52-week low — watch for support bounce")

        return {
            "symbol":          symbol,
            "signal":          signal,
            "confidence":      round(conf, 4),
            "conviction_score":conviction,
            "entry_price":     round(close, 2),
            "timestamp":       datetime.now(timezone.utc).isoformat(),
            "model_used":      pred["model_used"],
            "reasoning":       reasoning,
            "top_drivers":     drivers,
            "waterfall_data":  drivers,
            "model_breakdown": model_breakdown,
            "risk_parameters": {
                **risk_params,
                "suggested_position_size_pct": pos_size,
            },
            "regime_context":  regime,
            "macro_warnings":  warnings,
            "counterfactual":  _counterfactual(symbol, signal, latest),
        }


def _counterfactual(symbol: str, signal: str, features: pd.Series) -> str:
    """Generate a 'what would flip this signal' explanation."""
    rsi = features.get("rsi_14", 50)
    if signal == "buy":
        return f"Signal would flip to NEUTRAL if RSI rises above 60 (currently {rsi:.0f}) or MACD turns negative."
    if signal == "sell":
        return f"Signal would flip to NEUTRAL if RSI drops below 40 (currently {rsi:.0f}) or price reclaims 21-day MA."
    return f"Signal would strengthen to BUY if RSI drops below 35 and MACD crosses positive."
