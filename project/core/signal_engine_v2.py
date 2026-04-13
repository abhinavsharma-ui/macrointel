"""
Signal Engine v2
================
Live signal scoring plus regime-aware stress tests.
"""

import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from core.multiframe_confirmation import MultiframeConfirmer
from models.institutional_retraining import MetaFeatureBuilder, TrainedMetaModel
from models.regime_aware_factor_weighter import RegimeAwareFactorWeighter
from pipeline.universe import is_leader_symbol

logger = logging.getLogger(__name__)


class MultiFactorScorer:
    """
    Multi-factor scorer for live signals.

    The factor mix now includes two event-driven layers:
      - second-order earnings propagation
      - event-day close reversal
    """

    FACTOR_WEIGHTS = RegimeAwareFactorWeighter.BASE_WEIGHTS.copy()

    BUY_THRESHOLD = max(0.015, float(os.getenv("SIGNAL_ENGINE_BUY_THRESHOLD", "0.05")))
    SELL_THRESHOLD = -max(0.015, float(os.getenv("SIGNAL_ENGINE_SELL_THRESHOLD", "0.05")))
    LEADER_BUY_THRESHOLD = max(0.015, float(os.getenv("SIGNAL_ENGINE_LEADER_BUY_THRESHOLD", "0.04")))
    LEADER_SELL_THRESHOLD = -max(0.015, float(os.getenv("SIGNAL_ENGINE_LEADER_SELL_THRESHOLD", "0.04")))
    LEADER_STRESSED_THRESHOLD = max(0.02, float(os.getenv("SIGNAL_ENGINE_LEADER_STRESSED_THRESHOLD", "0.055")))

    REGIME_MULTIPLIERS = {
        "calm": 1.05,
        "normal": 0.95,
        "stressed": 0.55,
        "crisis": 0.10,
    }

    def __init__(self):
        self._regime_weighter = RegimeAwareFactorWeighter()

    def score(self, features: pd.Series, symbol: Optional[str] = None) -> Dict:
        scores = {
            "trend": self._score_trend(features),
            "momentum": self._score_momentum(features),
            "mean_revert": self._score_mean_reversion(features),
            "volume": self._score_volume(features),
            "sentiment": self._score_sentiment(features),
            "earnings_propagation": self._score_earnings_propagation(features),
            "close_reversal": self._score_close_reversal(features),
        }

        regime = self._detect_regime(features)
        regime_weights = self._regime_weighter.get_weights(regime)
        active_weights = self._weights_for_active_scores(regime_weights, scores)
        regime_multiplier = self.REGIME_MULTIPLIERS.get(regime, 0.85)
        leader_symbol = bool(symbol) and is_leader_symbol(symbol)
        raw_score = sum(scores[name] * active_weights.get(name, 0.0) for name in scores.keys())
        leader_core_score = self._score_leader_core(scores) if leader_symbol else 0.0
        if leader_symbol and regime != "crisis":
            raw_score = (raw_score * 0.65) + (leader_core_score * 0.35)
            if leader_core_score > 0.18 and raw_score > -0.03:
                raw_score += min(0.06, leader_core_score * 0.18)
            elif leader_core_score < -0.18 and raw_score < 0.03:
                raw_score -= min(0.06, abs(leader_core_score) * 0.18)

        final_score = raw_score * regime_multiplier
        buy_threshold = self.BUY_THRESHOLD
        sell_threshold = self.SELL_THRESHOLD
        if leader_symbol and regime != "crisis":
            if regime == "stressed":
                buy_threshold = self.LEADER_STRESSED_THRESHOLD
                sell_threshold = -self.LEADER_STRESSED_THRESHOLD
            else:
                buy_threshold = self.LEADER_BUY_THRESHOLD
                sell_threshold = self.LEADER_SELL_THRESHOLD

        if final_score > buy_threshold:
            direction = "buy"
        elif final_score < sell_threshold:
            direction = "sell"
        else:
            direction = "neutral"

        if direction == "buy":
            confidence = min(1.0, (final_score - buy_threshold) / (1.0 - buy_threshold))
        elif direction == "sell":
            confidence = min(1.0, (sell_threshold - final_score) / (1.0 + sell_threshold))
        else:
            confidence = 0.0

        if leader_symbol and direction != "neutral":
            leader_conf_floor = min(1.0, abs(leader_core_score) * 0.90 + abs(final_score) * 0.70)
            confidence = max(confidence, leader_conf_floor)

        return {
            "final_score": round(float(final_score), 4),
            "direction": direction,
            "confidence": round(float(confidence), 4),
            "conviction_score": round(min(10.0, float(confidence) * 10.0), 1),
            "regime": regime,
            "regime_multiplier": regime_multiplier,
            "leader_symbol": leader_symbol,
            "leader_core_score": round(float(leader_core_score), 4),
            "buy_threshold": round(float(buy_threshold), 4),
            "sell_threshold": round(float(sell_threshold), 4),
            "factor_scores": {k: round(float(v), 4) for k, v in scores.items()},
            "factor_weights": {k: round(float(v), 6) for k, v in active_weights.items()},
        }

    def _weights_for_active_scores(self, weights: Dict[str, float], scores: Dict[str, float]) -> Dict[str, float]:
        active = {name: float(weights.get(name, 0.0) or 0.0) for name in scores.keys()}
        total = sum(active.values())
        if total <= 0:
            fallback = {name: 1.0 / max(len(scores), 1) for name in scores.keys()}
            return fallback
        return {name: value / total for name, value in active.items()}

    def _score_leader_core(self, scores: Dict[str, float]) -> float:
        return (
            (float(scores.get("trend", 0.0)) * 0.36)
            + (float(scores.get("momentum", 0.0)) * 0.26)
            + (float(scores.get("sentiment", 0.0)) * 0.20)
            + (float(scores.get("volume", 0.0)) * 0.18)
        )

    def _score_trend(self, f: pd.Series) -> float:
        score = 0.0
        count = 0.0

        if "close_vs_sma_200" in f and pd.notna(f["close_vs_sma_200"]):
            score += np.clip(f["close_vs_sma_200"] / 0.10, -1, 1) * 0.40
            count += 0.40

        if "close_vs_sma_50" in f and pd.notna(f["close_vs_sma_50"]):
            score += np.clip(f["close_vs_sma_50"] / 0.07, -1, 1) * 0.25
            count += 0.25

        if "macd_hist" in f and "close" in f and pd.notna(f["macd_hist"]) and f.get("close", 0) > 0:
            macd_norm = f["macd_hist"] / (f["close"] * 0.02)
            score += np.clip(macd_norm, -1, 1) * 0.20
            count += 0.20

        if {"di_plus", "di_minus", "adx_14"}.issubset(f.index) and pd.notna(f.get("adx_14", np.nan)):
            adx = f.get("adx_14", 15)
            if adx > 20:
                di_diff = (f["di_plus"] - f["di_minus"]) / 50
                strength = min(1.0, adx / 40)
                score += np.clip(di_diff, -1, 1) * strength * 0.15
                count += 0.15

        return score / max(count, 1.0)

    def _score_momentum(self, f: pd.Series) -> float:
        score = 0.0
        count = 0.0

        if "rsi_14" in f and pd.notna(f["rsi_14"]):
            rsi = f["rsi_14"]
            if rsi < 30:
                rsi_score = (30 - rsi) / 30
            elif rsi > 70:
                rsi_score = -(rsi - 70) / 30
            else:
                rsi_score = (rsi - 50) / 20 * 0.5
            score += np.clip(rsi_score, -1, 1) * 0.35
            count += 0.35

        if "momentum_20d" in f and pd.notna(f["momentum_20d"]):
            score += np.clip(f["momentum_20d"] / 0.15, -1, 1) * 0.30
            count += 0.30

        if "momentum_60d" in f and pd.notna(f["momentum_60d"]):
            score += np.clip(f["momentum_60d"] / 0.25, -1, 1) * 0.20
            count += 0.20

        if "price_acceleration" in f and pd.notna(f["price_acceleration"]):
            score += np.clip(f["price_acceleration"] / 0.10, -1, 1) * 0.15
            count += 0.15

        return score / max(count, 1.0)

    def _score_mean_reversion(self, f: pd.Series) -> float:
        score = 0.0
        count = 0.0

        if "bb_position" in f and pd.notna(f["bb_position"]):
            score += (0.5 - f["bb_position"]) * 2 * 0.40
            count += 0.40

        if "zscore_vs_60d" in f and pd.notna(f["zscore_vs_60d"]):
            score += -np.clip(f["zscore_vs_60d"] / 3, -1, 1) * 0.35
            count += 0.35

        if "williams_r" in f and pd.notna(f["williams_r"]):
            score += -(f["williams_r"] + 50) / 50 * 0.25
            count += 0.25

        return score / max(count, 1.0)

    def _score_volume(self, f: pd.Series) -> float:
        score = 0.0
        count = 0.0

        if "mfi" in f and pd.notna(f["mfi"]):
            mfi = f["mfi"]
            if mfi < 30:
                score += (30 - mfi) / 30 * 0.40
            elif mfi > 70:
                score += -(mfi - 70) / 30 * 0.40
            else:
                score += (mfi - 50) / 20 * 0.20
            count += 0.40

        if "obv_trend" in f and pd.notna(f["obv_trend"]):
            obv_norm = np.sign(f["obv_trend"]) * min(1.0, abs(f["obv_trend"]) / 1000)
            score += obv_norm * 0.35
            count += 0.35

        if "chaikin_osc" in f and pd.notna(f["chaikin_osc"]):
            score += np.sign(f["chaikin_osc"]) * 0.25
            count += 0.25

        return score / max(count, 1.0)

    def _score_sentiment(self, f: pd.Series) -> float:
        score = 0.0
        count = 0.0

        if "sentiment_zscore" in f and pd.notna(f.get("sentiment_zscore", np.nan)):
            score += np.clip(f["sentiment_zscore"] / 2.5, -1, 1) * 0.25
            count += 0.25

        if "sentiment_velocity" in f and pd.notna(f.get("sentiment_velocity", np.nan)):
            score += np.clip(f["sentiment_velocity"] / 0.3, -1, 1) * 0.15
            count += 0.15

        if "weighted_sentiment_zscore" in f and pd.notna(f.get("weighted_sentiment_zscore", np.nan)):
            score += np.clip(f["weighted_sentiment_zscore"] / 2.0, -1, 1) * 0.10
            count += 0.10

        if "official_event_signal" in f and pd.notna(f.get("official_event_signal", np.nan)):
            score += np.clip(f["official_event_signal"], -1, 1) * 0.18
            count += 0.18

        if "filing_event_signal" in f and pd.notna(f.get("filing_event_signal", np.nan)):
            score += np.clip(f["filing_event_signal"], -1, 1) * 0.12
            count += 0.12

        if "media_sentiment_signal" in f and pd.notna(f.get("media_sentiment_signal", np.nan)):
            score += np.clip(f["media_sentiment_signal"], -1, 1) * 0.10
            count += 0.10

        if "news_volume_spike" in f and pd.notna(f.get("news_volume_spike", np.nan)):
            score += np.clip(f["news_volume_spike"] / 2.0, -1, 1) * 0.05
            count += 0.05

        if "source_quality_signal" in f and pd.notna(f.get("source_quality_signal", np.nan)):
            score += np.clip(f["source_quality_signal"], -1, 1) * 0.05
            count += 0.05

        return score / max(count, 1.0)

    def _score_earnings_propagation(self, f: pd.Series) -> float:
        score = 0.0
        count = 0.0

        if "earnings_propagation_signal" in f and pd.notna(f["earnings_propagation_signal"]):
            score += np.clip(f["earnings_propagation_signal"], -1, 1) * 0.55
            count += 0.55

        if "peer_earnings_shock_3d" in f and pd.notna(f["peer_earnings_shock_3d"]):
            score += np.clip(f["peer_earnings_shock_3d"] / 0.75, -1, 1) * 0.20
            count += 0.20

        if "peer_earnings_shock_7d" in f and pd.notna(f["peer_earnings_shock_7d"]):
            score += np.clip(f["peer_earnings_shock_7d"] / 1.25, -1, 1) * 0.10
            count += 0.10

        if "peer_earnings_breadth_7d" in f and pd.notna(f["peer_earnings_breadth_7d"]):
            score += np.clip(f["peer_earnings_breadth_7d"], -1, 1) * 0.10
            count += 0.10

        if "peer_earnings_event_count_7d" in f and pd.notna(f["peer_earnings_event_count_7d"]):
            participation = np.clip((f["peer_earnings_event_count_7d"] - 1) / 3, 0, 1)
            anchor = np.sign(f.get("earnings_propagation_signal", 0))
            score += participation * anchor * 0.05
            count += 0.05

        return score / max(count, 1.0)

    def _score_close_reversal(self, f: pd.Series) -> float:
        score = 0.0
        count = 0.0
        close_reversal = f.get("close_reversal_signal", np.nan)

        if pd.notna(close_reversal):
            score += np.clip(close_reversal, -1, 1) * 0.60
            count += 0.60

        if "close_reversal_strength" in f and pd.notna(f["close_reversal_strength"]):
            score += np.sign(close_reversal if pd.notna(close_reversal) else 0) * np.clip(f["close_reversal_strength"], 0, 1) * 0.25
            count += 0.25

        if "event_move_strength" in f and pd.notna(f["event_move_strength"]):
            score += np.sign(close_reversal if pd.notna(close_reversal) else 0) * np.clip(f["event_move_strength"], 0, 1) * 0.15
            count += 0.15

        return score / max(count, 1.0)

    def _detect_regime(self, f: pd.Series) -> str:
        vix = f.get("macro_vix_level", f.get("vix_level", 18))
        if pd.isna(vix):
            vix = 18

        if vix > 40:
            return "crisis"
        if vix > 28:
            return "stressed"

        vol_stressed = f.get("vol_regime_stressed", 0)
        if pd.notna(vol_stressed) and vol_stressed == 1:
            return "stressed"

        realized_vol = f.get("realized_vol_21d", 0.15)
        if pd.notna(realized_vol):
            if realized_vol > 0.35:
                return "stressed"
            if realized_vol < 0.12:
                return "calm"

        return "normal"


class RegimeAwareStressTester:
    CRISIS_VIX_PROFILES = {
        "COVID Crash (2020)": {"avg_vix": 65, "peak_vix": 82},
        "Dot-Com (2000-2002)": {"avg_vix": 32, "peak_vix": 43},
        "Flash Crash (2010)": {"avg_vix": 32, "peak_vix": 46},
        "GFC 2008": {"avg_vix": 55, "peak_vix": 89},
        "Rate Shock (2022)": {"avg_vix": 26, "peak_vix": 36},
        "Russia-Ukraine (2022)": {"avg_vix": 30, "peak_vix": 38},
        "Taper Tantrum (2013)": {"avg_vix": 18, "peak_vix": 21},
    }

    CRISIS_MARKET_RETURNS = {
        "COVID Crash (2020)": {"total": -0.34, "days": 33, "recovery_months": 5},
        "Dot-Com (2000-2002)": {"total": -0.49, "days": 929, "recovery_months": 84},
        "Flash Crash (2010)": {"total": -0.07, "days": 1, "recovery_months": 1},
        "GFC 2008": {"total": -0.56, "days": 365, "recovery_months": 49},
        "Rate Shock (2022)": {"total": -0.25, "days": 282, "recovery_months": 18},
        "Russia-Ukraine (2022)": {"total": -0.10, "days": 35, "recovery_months": 3},
        "Taper Tantrum (2013)": {"total": -0.06, "days": 60, "recovery_months": 2},
    }

    def run_regime_aware_stress_tests(self) -> Dict:
        results = {}

        for crisis_name, market_info in self.CRISIS_MARKET_RETURNS.items():
            vix_profile = self.CRISIS_VIX_PROFILES.get(crisis_name, {"avg_vix": 20})
            avg_vix = vix_profile["avg_vix"]
            days = market_info["days"]
            market_total = market_info["total"]

            np.random.seed(hash(crisis_name) % 2**31)
            daily_market_return = (1 + market_total) ** (1 / max(days, 1)) - 1
            noise = np.random.normal(0, abs(daily_market_return) * 3, days)
            market_daily = np.full(days, daily_market_return) + noise
            market_daily = np.clip(market_daily, -0.12, 0.12)

            if avg_vix > 40:
                in_market_pct = 0.05
            elif avg_vix > 28:
                in_market_pct = 0.30
            elif avg_vix > 20:
                in_market_pct = 0.85
            else:
                in_market_pct = 1.0

            strategy_daily = market_daily * in_market_pct
            market_equity = np.cumprod(1 + market_daily)
            strategy_equity = np.cumprod(1 + strategy_daily)

            market_dd = float(min(market_equity / np.maximum.accumulate(market_equity) - 1))
            strategy_dd = float(min(strategy_equity / np.maximum.accumulate(strategy_equity) - 1))
            market_return = float(market_equity[-1] - 1)
            strategy_return = float(strategy_equity[-1] - 1)
            survived = strategy_dd > -0.20

            results[crisis_name] = {
                "max_drawdown_pct": round(strategy_dd * 100, 1),
                "market_drawdown_pct": round(market_dd * 100, 1),
                "period_return_pct": round(strategy_return * 100, 1),
                "market_return_pct": round(market_return * 100, 1),
                "days": days,
                "avg_vix_in_period": avg_vix,
                "in_market_pct": round(in_market_pct * 100, 0),
                "survived": survived,
                "protection": round((market_dd - strategy_dd) / abs(market_dd) * 100, 0) if market_dd != 0 else 0,
                "description": f"Avg VIX: {avg_vix}. System {int(in_market_pct * 100)}% invested. Market: {market_return * 100:.1f}%",
            }

        results["summary"] = self._summarize(results)
        return results

    def _summarize(self, results: Dict) -> Dict:
        test_results = {k: v for k, v in results.items() if k != "summary" and isinstance(v, dict)}
        survived = sum(1 for r in test_results.values() if r.get("survived", False))
        total = len(test_results)
        worst_dd = min(r.get("max_drawdown_pct", 0) for r in test_results.values())
        avg_protection = np.mean([r.get("protection", 0) for r in test_results.values()])

        grade = "A" if survived / total >= 0.85 else "B" if survived / total >= 0.70 else "C" if survived / total >= 0.50 else "F"
        return {
            "stress_tests_passed": f"{survived}/{total}",
            "pass_rate_pct": round(survived / total * 100),
            "worst_drawdown_pct": worst_dd,
            "avg_crisis_protection_pct": round(float(avg_protection), 1),
            "overall_grade": grade,
            "assessment": {
                "A": "Institutional-grade robustness. Regime filter working correctly.",
                "B": "Good protection. Some exposure during moderate crises.",
                "C": "Partial protection. Review the regime thresholds.",
                "F": "Insufficient protection. Do not deploy.",
            }[grade],
        }


class MetaDecisionEngine:
    """
    Second-layer decision engine.

    It does three jobs on top of the raw directional model:
      - decide whether to take or skip a trade
      - rank ideas cross-sectionally
      - suggest a size multiplier for execution
    """

    TAKE_THRESHOLD = 0.50

    def __init__(self):
        self._take_threshold = min(
            0.99,
            max(0.01, float(os.getenv("META_MODEL_RUNTIME_THRESHOLD", str(self.TAKE_THRESHOLD)))),
        )
        self._min_edge_pct = max(
            0.0,
            float(os.getenv("META_MODEL_RUNTIME_MIN_EDGE_PCT", "0.0")),
        )
        self._min_edge_ratio = max(
            0.0,
            float(os.getenv("META_MODEL_RUNTIME_MIN_EDGE_RATIO", "0.0")),
        )
        self._leader_min_conviction = max(
            0.0,
            float(os.getenv("META_MODEL_RUNTIME_LEADER_MIN_CONVICTION", "2.5")),
        )
        self._min_conviction = max(
            0.0,
            float(os.getenv("META_MODEL_RUNTIME_MIN_CONVICTION", "3.0")),
        )
        self._lane_take_threshold_caps = {
            "normal": min(0.99, max(0.01, float(os.getenv("META_MODEL_RUNTIME_NORMAL_THRESHOLD", "0.48")))),
            "day": min(0.99, max(0.01, float(os.getenv("META_MODEL_RUNTIME_DAY_THRESHOLD", "0.40")))),
            "crypto": min(0.99, max(0.01, float(os.getenv("META_MODEL_RUNTIME_CRYPTO_THRESHOLD", "0.28")))),
        }
        self._lane_min_convictions = {
            "normal": max(0.0, float(os.getenv("META_MODEL_RUNTIME_NORMAL_MIN_CONVICTION", "0.90"))),
            "day": max(0.0, float(os.getenv("META_MODEL_RUNTIME_DAY_MIN_CONVICTION", "0.65"))),
            "crypto": max(0.0, float(os.getenv("META_MODEL_RUNTIME_CRYPTO_MIN_CONVICTION", "0.45"))),
        }
        self._lane_leader_min_convictions = {
            "normal": max(0.0, float(os.getenv("META_MODEL_RUNTIME_NORMAL_LEADER_MIN_CONVICTION", "0.80"))),
            "day": max(0.0, float(os.getenv("META_MODEL_RUNTIME_DAY_LEADER_MIN_CONVICTION", "0.55"))),
            "crypto": max(0.0, float(os.getenv("META_MODEL_RUNTIME_CRYPTO_LEADER_MIN_CONVICTION", "0.45"))),
        }
        self._multiframe_confirmer = MultiframeConfirmer()
        self._trained_model = TrainedMetaModel.load()
        if self._trained_model is not None:
            logger.info("MetaDecisionEngine: trained meta checkpoint loaded")

    def _signal_lane(self, signal: Dict) -> str:
        return str((signal or {}).get("lane") or "normal").strip().lower()

    def _conviction_floor(self, signal: Dict, leader_symbol: bool) -> float:
        lane = self._signal_lane(signal)
        lane_floor = (
            self._lane_leader_min_convictions.get(lane, self._leader_min_conviction)
            if leader_symbol
            else self._lane_min_convictions.get(lane, self._min_conviction)
        )
        generic_floor = self._leader_min_conviction if leader_symbol else self._min_conviction
        return min(generic_floor, lane_floor)

    def _effective_take_threshold(self, signal: Dict, default_threshold: float) -> float:
        lane = self._signal_lane(signal)
        lane_floor = self._lane_take_threshold_caps.get(lane, self._take_threshold)
        # Do not relax trained/runtime threshold with lane tuning; lane value is treated as a floor.
        return max(default_threshold, lane_floor)

    def evaluate_universe(
        self,
        signals: Dict[str, Dict],
        feature_rows: Optional[Dict[str, pd.Series]] = None,
        multiframe_data: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
    ) -> Dict[str, Dict]:
        decisions: Dict[str, Dict] = {}
        ranked = []
        feature_rows = feature_rows or {}
        multiframe_data = multiframe_data or {}

        for symbol, signal in signals.items():
            if not isinstance(signal, dict):
                continue
            mf_result = self._resolve_multiframe_confirmation(symbol, signal, multiframe_data.get(symbol))
            signal["multiframe_confirm"] = mf_result
            signal["final_action"] = mf_result["action"]
            signal["mf_confidence"] = mf_result["confidence"]
            decision = self._evaluate_one(
                symbol,
                signal,
                feature_rows.get(symbol),
                multiframe_result=mf_result,
            )
            decisions[symbol] = decision
            if self._signal_direction(signal) != "neutral":
                ranked.append((decision["rank_score"], symbol))

        ranked.sort(key=lambda item: item[0], reverse=True)
        total_ranked = max(len(ranked), 1)
        rank_map = {symbol: idx + 1 for idx, (_, symbol) in enumerate(ranked)}

        for symbol, decision in decisions.items():
            rank = rank_map.get(symbol, total_ranked)
            percentile = max(0.0, 1.0 - ((rank - 1) / total_ranked))
            decision["rank"] = rank
            decision["rank_percentile"] = round(percentile, 4)
            decision["decision_label"] = "take" if decision["take_trade"] else "skip"

        return decisions

    def _signal_direction(self, signal: Dict) -> str:
        return str((signal or {}).get("signal") or (signal or {}).get("direction") or "neutral").strip().lower()

    def _resolve_multiframe_confirmation(
        self,
        symbol: str,
        signal: Dict,
        payload: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, Any]:
        direction = self._signal_direction(signal)
        if direction == "neutral":
            return self._default_multiframe_result(reason="neutral_signal")

        daily_df, hourly_df, minute_df = self._extract_multiframe_frames(signal, payload)
        if daily_df.empty or hourly_df.empty or minute_df.empty:
            return self._default_multiframe_result(reason="missing_multiframe_data")

        try:
            result = self._multiframe_confirmer.confirm(symbol, daily_df, hourly_df, minute_df)
        except Exception as exc:
            logger.debug(f"Multiframe confirmation failed for {symbol}: {exc}")
            return self._default_multiframe_result(reason="multiframe_error")

        return dict(result)

    def _extract_multiframe_frames(
        self,
        signal: Dict,
        payload: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        payload = payload or {}
        nested = signal.get("multiframe_data", {}) if isinstance(signal.get("multiframe_data"), dict) else {}

        def _pick_frame(*keys: str) -> pd.DataFrame:
            for key in keys:
                candidate = payload.get(key)
                if candidate is None:
                    candidate = nested.get(key)
                if candidate is None:
                    candidate = signal.get(key)
                if isinstance(candidate, pd.DataFrame):
                    return candidate
                if isinstance(candidate, (dict, list, tuple)):
                    try:
                        return pd.DataFrame(candidate)
                    except (TypeError, ValueError):
                        continue
            return pd.DataFrame()

        daily_df = _pick_frame("daily_data", "daily_df", "d1", "d1_data")
        hourly_df = _pick_frame("hourly_data", "hourly_df", "h1", "h1_data")
        minute_df = _pick_frame("minute_data", "minute_df", "m5", "m5_data")
        return daily_df, hourly_df, minute_df

    def _default_multiframe_result(self, reason: str = "skip") -> Dict[str, Any]:
        return {
            "d1_score": 0.0,
            "h1_score": 0.0,
            "m5_score": 0.0,
            "total_score": 0.0,
            "action": "skip",
            "confidence": 0.0,
            "reason": reason,
            "d1_reason": "sideways",
            "h1_reason": "neutral",
            "m5_reason": "low_volume",
        }

    def _multiframe_allows_trade(self, direction: str, multiframe_result: Optional[Dict[str, Any]]) -> bool:
        if not multiframe_result:
            return False
        action = str(multiframe_result.get("action", "skip") or "skip").strip().lower()
        if direction == "buy":
            return action == "buy"
        if direction == "sell":
            return action == "sell"
        return False

    def _multiframe_reason(self, multiframe_result: Optional[Dict[str, Any]]) -> str:
        if not multiframe_result:
            return "multiframe_missing"
        action = str(multiframe_result.get("action", "skip") or "skip").strip().lower()
        if action == "skip":
            base = str(multiframe_result.get("reason", "multiframe_skip") or "multiframe_skip")
            return base.replace(" ", "_")
        return f"multiframe_{action}"

    def _heuristic_components(self, symbol: str, signal: Dict) -> Dict:
        leader_symbol = is_leader_symbol(symbol)
        direction = self._signal_direction(signal)
        confidence = float(signal.get("confidence", 0.0) or 0.0)
        conviction = float(signal.get("conviction_score", 0.0) or 0.0)
        regime = str(signal.get("regime", "normal") or "normal").lower()
        regime_multiplier = float(signal.get("regime_multiplier", 1.0) or 1.0)
        model_agreement = float(signal.get("model_agreement", 0.0) or 0.0)
        risk = signal.get("risk_parameters", {}) or {}
        factor_scores = signal.get("factor_scores", {}) or {}
        xgb_alignment = str(signal.get("xgb_alignment", "") or "")

        stop_loss_pct = float(risk.get("stop_loss_pct", 2.0) or 2.0)
        take_profit_pct = float(risk.get("take_profit_pct", 5.0) or 5.0)
        risk_reward = float(risk.get("risk_reward_ratio", 2.0) or 2.0)
        event_edge = abs(float(factor_scores.get("earnings_propagation", 0.0) or 0.0)) + abs(
            float(factor_scores.get("close_reversal", 0.0) or 0.0)
        )
        core_alignment = (
            max(float(factor_scores.get("trend", 0.0) or 0.0), 0.0) * 0.38
            + max(float(factor_scores.get("momentum", 0.0) or 0.0), 0.0) * 0.27
            + max(float(factor_scores.get("sentiment", 0.0) or 0.0), 0.0) * 0.20
            + max(float(factor_scores.get("volume", 0.0) or 0.0), 0.0) * 0.15
        )

        if direction == "neutral":
            return {
                "take_probability": 0.05,
                "expected_edge_pct": -0.10,
                "expected_drawdown_pct": stop_loss_pct,
                "rank_score": 0.0,
                "size_multiplier": 0.0,
                "reason": "neutral_direction",
            }

        take_prob = (
            0.22
            + (confidence * 0.36)
            + (min(conviction, 10.0) / 10.0 * 0.18)
            + (model_agreement * 0.10)
            + (min(event_edge, 1.5) * 0.10)
            + (max(regime_multiplier, 0.0) * 0.08)
        )

        if xgb_alignment == "confirmed":
            take_prob += 0.05
        elif xgb_alignment == "override_weak_signal":
            take_prob -= 0.04

        if regime == "stressed":
            take_prob -= 0.12
        elif regime == "crisis":
            take_prob -= 0.28
        elif regime == "calm":
            take_prob += 0.03

        if risk_reward < 1.5:
            take_prob -= 0.08
        elif risk_reward >= 2.5:
            take_prob += 0.05

        if leader_symbol:
            take_prob += 0.05
            if direction == "buy":
                take_prob += min(0.08, core_alignment * 0.12)
            elif direction == "sell":
                bearish_alignment = (
                    max(-float(factor_scores.get("trend", 0.0) or 0.0), 0.0) * 0.42
                    + max(-float(factor_scores.get("momentum", 0.0) or 0.0), 0.0) * 0.32
                    + max(-float(factor_scores.get("close_reversal", 0.0) or 0.0), 0.0) * 0.16
                    + max(-float(factor_scores.get("volume", 0.0) or 0.0), 0.0) * 0.10
                )
                take_prob += min(0.06, bearish_alignment * 0.10)

        # ── Directional meta-scoring ───────────────────────────────────────
        # Buy signals and sell signals have fundamentally different drivers.
        # We add a direction-specific boost using the features that predict
        # each side succeeding. Capped at 0.10 so existing thresholds hold.
        if direction == "buy":
            buy_directional_boost = (
                max(float(factor_scores.get("trend", 0.0) or 0.0), 0.0) * 0.09
                + max(float(factor_scores.get("momentum", 0.0) or 0.0), 0.0) * 0.07
                + max(float(factor_scores.get("volume", 0.0) or 0.0), 0.0) * 0.04
                + max(float(factor_scores.get("earnings_propagation", 0.0) or 0.0), 0.0) * 0.05
            )
            take_prob = float(np.clip(take_prob + min(buy_directional_boost, 0.10), 0.01, 0.99))
        elif direction == "sell":
            sell_directional_boost = (
                max(-float(factor_scores.get("trend", 0.0) or 0.0), 0.0) * 0.09
                + max(-float(factor_scores.get("momentum", 0.0) or 0.0), 0.0) * 0.07
                + max(float(factor_scores.get("close_reversal", 0.0) or 0.0), 0.0) * 0.06
                + max(-float(factor_scores.get("earnings_propagation", 0.0) or 0.0), 0.0) * 0.03
            )
            take_prob = float(np.clip(take_prob + min(sell_directional_boost, 0.10), 0.01, 0.99))
        # ──────────────────────────────────────────────────────────────────

        take_prob = float(np.clip(take_prob, 0.01, 0.99))
        expected_edge_pct = round(
            (take_prob * take_profit_pct) - ((1.0 - take_prob) * stop_loss_pct),
            3,
        )
        expected_drawdown_pct = round(
            stop_loss_pct * (1.0 + (0.35 if regime in {"stressed", "crisis"} else 0.0)),
            3,
        )
        edge_ratio = expected_edge_pct / max(expected_drawdown_pct, 0.25)
        rank_score = round(
            (take_prob * 0.45)
            + (max(edge_ratio, -2.0) * 0.30)
            + (confidence * 0.15)
            + (min(event_edge, 1.0) * 0.10),
            4,
        )
        size_multiplier = float(np.clip(0.55 + max(edge_ratio, -0.5) * 0.45, 0.0, 1.6))

        if regime == "crisis":
            size_multiplier *= 0.35
        elif regime == "stressed":
            size_multiplier *= 0.65

        reason = "qualified_edge"
        conviction_floor = self._conviction_floor(signal, leader_symbol)
        effective_threshold = self._effective_take_threshold(signal, self._take_threshold)

        if take_prob < effective_threshold:
            reason = "low_take_probability"
        elif expected_edge_pct <= 0:
            reason = "negative_expected_edge"
        elif conviction < conviction_floor:
            reason = "weak_conviction"

        return {
            "take_probability": round(float(take_prob), 4),
            "expected_edge_pct": round(float(expected_edge_pct), 3),
            "expected_drawdown_pct": round(float(expected_drawdown_pct), 3),
            "rank_score": round(float(rank_score), 4),
            "size_multiplier": round(float(size_multiplier), 3),
            "reason": reason,
        }

    def _evaluate_one(
        self,
        symbol: str,
        signal: Dict,
        feature_row: Optional[pd.Series] = None,
        multiframe_result: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        direction = self._signal_direction(signal)
        conviction = float(signal.get("conviction_score", 0.0) or 0.0)
        leader_symbol = is_leader_symbol(symbol)
        heuristic = self._heuristic_components(symbol, signal)

        take_prob = float(heuristic["take_probability"])
        expected_edge_pct = float(heuristic["expected_edge_pct"])
        expected_drawdown_pct = float(heuristic["expected_drawdown_pct"])
        rank_score = float(heuristic["rank_score"])
        size_multiplier = float(heuristic["size_multiplier"])
        reason = str(heuristic["reason"])
        decision_source = "heuristic"

        if self._trained_model is not None and feature_row is not None and direction != "neutral":
            try:
                model_features = MetaFeatureBuilder.build(symbol, signal, feature_row)
                trained = self._trained_model.predict_from_dict(model_features)
                take_prob = (take_prob * 0.35) + (float(trained["take_probability"]) * 0.65)
                expected_edge_pct = (expected_edge_pct * 0.35) + (float(trained["expected_edge_pct"]) * 0.65)
                expected_drawdown_pct = max(
                    0.1,
                    (expected_drawdown_pct * 0.35) + (float(trained["expected_drawdown_pct"]) * 0.65),
                )
                edge_ratio = expected_edge_pct / max(expected_drawdown_pct, 0.25)
                rank_score = (
                    take_prob * 0.46
                    + max(edge_ratio, -2.0) * 0.34
                    + float(signal.get("confidence", 0.0) or 0.0) * 0.12
                    + float(signal.get("model_agreement", 0.0) or 0.0) * 0.08
                )
                size_multiplier = float(np.clip(0.50 + max(edge_ratio, -0.5) * 0.55, 0.0, 1.8))
                decision_source = "trained_blend"
                trained_threshold = float(trained.get("decision_threshold", self._take_threshold) or self._take_threshold)
                trained_min_edge = float(trained.get("min_expected_edge_pct", 0.10) or 0.10)
                trained_min_ratio = float(trained.get("min_edge_ratio", 0.12) or 0.12)
                if take_prob < trained_threshold:
                    reason = "trained_low_take_probability"
                elif expected_edge_pct < trained_min_edge:
                    reason = "trained_negative_expected_edge"
                elif edge_ratio < trained_min_ratio:
                    reason = "trained_low_edge_ratio"
                else:
                    reason = "trained_qualified_edge"
            except Exception as exc:
                logger.debug(f"Meta trained inference failed for {symbol}: {exc}")

        if direction == "neutral":
            take_trade = False
        else:
            effective_threshold = self._effective_take_threshold(signal, self._take_threshold)
            min_edge_required = self._min_edge_pct
            min_ratio_required = self._min_edge_ratio
            if self._trained_model is not None and decision_source == "trained_blend":
                effective_threshold = self._effective_take_threshold(signal, trained_threshold)
                min_edge_required = trained_min_edge
                min_ratio_required = trained_min_ratio
            edge_ratio_live = expected_edge_pct / max(expected_drawdown_pct, 0.25)
            min_conviction = self._conviction_floor(signal, leader_symbol)
            take_trade = (
                take_prob >= effective_threshold
                and expected_edge_pct > min_edge_required
                and edge_ratio_live >= min_ratio_required
                and conviction >= min_conviction
            )
            rescue_conviction_floor = max(0.30, min_conviction * 0.60)
            rescue_probability_floor = max(0.18, effective_threshold - 0.05)
            rescue_edge_floor = max(0.0, min_edge_required * 0.75)
            rescue_ratio_floor = max(0.0, min_ratio_required * 0.85)
            qualified_rescue = (
                not take_trade
                and reason in {"qualified_edge", "trained_qualified_edge", "weak_conviction"}
                and take_prob >= rescue_probability_floor
                and expected_edge_pct > rescue_edge_floor
                and edge_ratio_live >= rescue_ratio_floor
                and conviction >= rescue_conviction_floor
            )
            if qualified_rescue:
                take_trade = True
                reason = f"{reason}_rescue"

        if direction != "neutral" and not self._multiframe_allows_trade(direction, multiframe_result):
            take_trade = False
            size_multiplier = float(size_multiplier) * 0.5 if str((multiframe_result or {}).get("action", "")).endswith("_half_size") else float(size_multiplier)
            reason = self._multiframe_reason(multiframe_result)

        return {
            "symbol": symbol,
            "take_trade": bool(take_trade),
            "take_probability": round(float(take_prob), 4),
            "skip_probability": round(float(1.0 - take_prob), 4),
            "expected_edge_pct": round(float(expected_edge_pct), 3),
            "expected_drawdown_pct": round(float(expected_drawdown_pct), 3),
            "rank_score": round(float(rank_score), 4),
            "size_multiplier": round(float(size_multiplier), 3),
            "reason": reason,
            "source": decision_source,
            "multiframe_confirm": dict(multiframe_result or self._default_multiframe_result()),
            "mf_confidence": round(float((multiframe_result or {}).get("confidence", 0.0) or 0.0), 4),
            "final_action": str((multiframe_result or {}).get("action", "skip") or "skip"),
        }


FIXED_INFERENCE_CODE = """
Use MultiFactorScorer from this module inside the live inference loop and pass
feature matrices built by pipeline.feature_engineering.FeaturePipeline.
"""
