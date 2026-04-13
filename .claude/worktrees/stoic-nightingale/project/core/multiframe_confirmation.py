"""
Multi-Timeframe Signal Confirmation
====================================
D1 (daily) → trend direction
H1 (hourly) → entry timing
M5 (5-min) → execution pressure

A trade only triggers if all three timeframes align.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TimeframeSignal:
    """Signal from a single timeframe."""
    timeframe: str          # "D1", "H1", "M5"
    direction: str          # "bullish", "bearish", "neutral"
    strength: float         # 0.0 – 1.0
    indicators: Dict[str, float] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class MultiframeVerdict:
    """Final verdict after checking all timeframes."""
    aligned: bool
    direction: str          # "buy", "sell", "neutral"
    alignment_score: float  # 0.0 – 1.0
    confidence_boost: float # multiplier applied to original confidence
    signals: Dict[str, TimeframeSignal] = field(default_factory=dict)
    reason: str = ""


# ---------------------------------------------------------------------------
# Timeframe analysers
# ---------------------------------------------------------------------------

class _D1Analyser:
    """Daily trend — SMA crossovers, ADX, MACD."""

    def analyse(self, daily: pd.DataFrame) -> TimeframeSignal:
        if daily is None or len(daily) < 50:
            return TimeframeSignal("D1", "neutral", 0.0, reason_missing("D1", "insufficient data"))

        close = daily["close"].values
        indicators: Dict[str, float] = {}
        votes: List[float] = []

        # SMA 20/50 cross
        sma20 = _sma(close, 20)
        sma50 = _sma(close, 50)
        if sma20 is not None and sma50 is not None:
            cross = (sma20 - sma50) / max(sma50, 1e-9)
            indicators["sma20_vs_sma50"] = round(cross, 5)
            votes.append(np.clip(cross / 0.03, -1, 1))

        # Price vs SMA 200
        if len(close) >= 200:
            sma200 = _sma(close, 200)
            if sma200 is not None and sma200 > 0:
                pct_above = (close[-1] - sma200) / sma200
                indicators["close_vs_sma200"] = round(pct_above, 5)
                votes.append(np.clip(pct_above / 0.10, -1, 1))

        # MACD histogram direction (12,26,9)
        macd_line = _ema(close, 12) - _ema(close, 26) if len(close) >= 26 else None
        if macd_line is not None:
            signal_line = _ema_from_val(close[-9:], 9, seed=macd_line) if len(close) >= 35 else macd_line
            macd_hist = macd_line - signal_line
            indicators["macd_hist"] = round(macd_hist, 5)
            votes.append(np.clip(macd_hist / (close[-1] * 0.01 + 1e-9), -1, 1))

        # ADX trend strength
        if len(daily) >= 28 and "high" in daily.columns and "low" in daily.columns:
            adx, di_plus, di_minus = _adx(daily, 14)
            if adx is not None:
                indicators["adx"] = round(adx, 2)
                indicators["di_plus"] = round(di_plus, 2)
                indicators["di_minus"] = round(di_minus, 2)
                if adx > 20:
                    di_signal = np.clip((di_plus - di_minus) / 40, -1, 1) * min(1.0, adx / 35)
                    votes.append(di_signal)

        if not votes:
            return TimeframeSignal("D1", "neutral", 0.0, indicators)

        avg = float(np.mean(votes))
        strength = min(1.0, abs(avg))
        direction = "bullish" if avg > 0.10 else "bearish" if avg < -0.10 else "neutral"
        return TimeframeSignal("D1", direction, strength, indicators, time.time())


class _H1Analyser:
    """Hourly entry timing — RSI, Bollinger, VWAP."""

    def analyse(self, hourly: pd.DataFrame) -> TimeframeSignal:
        if hourly is None or len(hourly) < 20:
            return TimeframeSignal("H1", "neutral", 0.0, reason_missing("H1", "insufficient data"))

        close = hourly["close"].values
        indicators: Dict[str, float] = {}
        votes: List[float] = []

        # RSI 14
        rsi = _rsi(close, 14)
        if rsi is not None:
            indicators["rsi_14"] = round(rsi, 2)
            if rsi < 30:
                votes.append((30 - rsi) / 30)
            elif rsi > 70:
                votes.append(-(rsi - 70) / 30)
            else:
                votes.append((rsi - 50) / 40)

        # Bollinger band position
        if len(close) >= 20:
            sma = _sma(close, 20)
            std = float(np.std(close[-20:]))
            if sma is not None and std > 0:
                upper = sma + 2 * std
                lower = sma - 2 * std
                bb_pos = (close[-1] - lower) / max(upper - lower, 1e-9)
                indicators["bb_position"] = round(bb_pos, 4)
                # Near lower band = bullish, near upper = bearish
                votes.append(np.clip((0.5 - bb_pos) * 2, -1, 1))

        # VWAP deviation (if volume present)
        if "volume" in hourly.columns and len(hourly) >= 10:
            vol = hourly["volume"].values[-20:]
            px = close[-20:]
            total_vp = float(np.sum(px * vol))
            total_vol = float(np.sum(vol))
            if total_vol > 0:
                vwap = total_vp / total_vol
                dev = (close[-1] - vwap) / max(vwap, 1e-9)
                indicators["vwap_deviation"] = round(dev, 5)
                votes.append(np.clip(dev / 0.02, -1, 1))

        # Stochastic %K/%D (14,3,3)
        if len(hourly) >= 17 and "high" in hourly.columns and "low" in hourly.columns:
            stoch_k, stoch_d = _stochastic(hourly, 14, 3)
            if stoch_k is not None:
                indicators["stoch_k"] = round(stoch_k, 2)
                indicators["stoch_d"] = round(stoch_d, 2)
                if stoch_k < 20:
                    votes.append((20 - stoch_k) / 20 * 0.8)
                elif stoch_k > 80:
                    votes.append(-(stoch_k - 80) / 20 * 0.8)
                else:
                    votes.append((stoch_k - 50) / 60 * 0.4)

        if not votes:
            return TimeframeSignal("H1", "neutral", 0.0, indicators)

        avg = float(np.mean(votes))
        strength = min(1.0, abs(avg))
        direction = "bullish" if avg > 0.08 else "bearish" if avg < -0.08 else "neutral"
        return TimeframeSignal("H1", direction, strength, indicators, time.time())


class _M5Analyser:
    """5-minute execution pressure — volume burst, spread, micro-momentum."""

    def analyse(self, m5: pd.DataFrame) -> TimeframeSignal:
        if m5 is None or len(m5) < 12:
            return TimeframeSignal("M5", "neutral", 0.0, reason_missing("M5", "insufficient data"))

        close = m5["close"].values
        indicators: Dict[str, float] = {}
        votes: List[float] = []

        # Micro-momentum: last 6 bars
        recent = close[-6:]
        if len(recent) >= 2:
            micro_ret = (recent[-1] - recent[0]) / max(abs(recent[0]), 1e-9)
            indicators["micro_momentum_6bar"] = round(micro_ret, 5)
            votes.append(np.clip(micro_ret / 0.005, -1, 1))

        # Volume burst vs 60-bar average
        if "volume" in m5.columns and len(m5) >= 60:
            vol = m5["volume"].values
            avg_vol = float(np.mean(vol[-60:]))
            recent_vol = float(np.mean(vol[-6:]))
            if avg_vol > 0:
                vol_ratio = recent_vol / avg_vol
                indicators["volume_burst_ratio"] = round(vol_ratio, 3)
                # High volume confirms direction
                if vol_ratio > 1.5:
                    micro_dir = 1.0 if close[-1] >= close[-6] else -1.0
                    votes.append(micro_dir * min(1.0, (vol_ratio - 1) / 2))

        # Price acceleration (second derivative)
        if len(close) >= 12:
            ret_recent = (close[-1] - close[-6]) / max(abs(close[-6]), 1e-9)
            ret_prior = (close[-6] - close[-12]) / max(abs(close[-12]), 1e-9)
            accel = ret_recent - ret_prior
            indicators["price_acceleration"] = round(accel, 6)
            votes.append(np.clip(accel / 0.003, -1, 1))

        # Tick pressure: count up vs down closes in last 12 bars
        if len(close) >= 12:
            diffs = np.diff(close[-12:])
            up = int(np.sum(diffs > 0))
            down = int(np.sum(diffs < 0))
            total = up + down
            if total > 0:
                tick_bias = (up - down) / total
                indicators["tick_pressure"] = round(tick_bias, 3)
                votes.append(tick_bias * 0.7)

        if not votes:
            return TimeframeSignal("M5", "neutral", 0.0, indicators)

        avg = float(np.mean(votes))
        strength = min(1.0, abs(avg))
        direction = "bullish" if avg > 0.05 else "bearish" if avg < -0.05 else "neutral"
        return TimeframeSignal("M5", direction, strength, indicators, time.time())


# ---------------------------------------------------------------------------
# Main confirmer
# ---------------------------------------------------------------------------

TIMEFRAME_WEIGHTS = {"D1": 0.50, "H1": 0.30, "M5": 0.20}

# Minimum alignment score to consider confirmed
ALIGNMENT_THRESHOLD = float(os.getenv("MULTIFRAME_ALIGNMENT_THRESHOLD", "0.40"))


class MultiframeConfirmer:
    """
    Check D1 + H1 + M5 alignment before allowing a trade.

    Usage::

        confirmer = MultiframeConfirmer()
        verdict = confirmer.confirm(
            daily_df, hourly_df, m5_df,
            proposed_direction="buy",
        )
        if verdict.aligned:
            # execute trade with verdict.confidence_boost applied
    """

    def __init__(self):
        self._d1 = _D1Analyser()
        self._h1 = _H1Analyser()
        self._m5 = _M5Analyser()

    def confirm(
        self,
        daily: Optional[pd.DataFrame] = None,
        hourly: Optional[pd.DataFrame] = None,
        m5: Optional[pd.DataFrame] = None,
        proposed_direction: str = "buy",
    ) -> MultiframeVerdict:
        """
        Check multi-timeframe alignment against a proposed direction.

        Parameters
        ----------
        daily : recent daily OHLCV (>= 50 rows ideal)
        hourly : recent hourly OHLCV (>= 20 rows ideal)
        m5 : recent 5-min OHLCV (>= 12 rows ideal)
        proposed_direction : "buy" or "sell"

        Returns
        -------
        MultiframeVerdict with .aligned, .direction, .alignment_score, .confidence_boost
        """
        target = "bullish" if proposed_direction == "buy" else "bearish"

        sig_d1 = self._d1.analyse(daily)
        sig_h1 = self._h1.analyse(hourly)
        sig_m5 = self._m5.analyse(m5)

        signals = {"D1": sig_d1, "H1": sig_h1, "M5": sig_m5}

        # Weighted alignment score: +1 if agrees, -1 if opposes, 0 if neutral
        alignment = 0.0
        for tf, sig in signals.items():
            w = TIMEFRAME_WEIGHTS[tf]
            if sig.direction == target:
                alignment += w * sig.strength
            elif sig.direction == "neutral":
                alignment += 0.0
            else:
                alignment -= w * sig.strength

        alignment_score = max(0.0, min(1.0, (alignment + 1) / 2))  # map [-1,1] → [0,1]

        # D1 must not oppose — non-negotiable gate
        d1_opposes = (sig_d1.direction != "neutral" and sig_d1.direction != target)
        if d1_opposes:
            return MultiframeVerdict(
                aligned=False,
                direction="neutral",
                alignment_score=alignment_score,
                confidence_boost=0.0,
                signals=signals,
                reason=f"D1 opposes ({sig_d1.direction} vs required {target})",
            )

        # At least 2 of 3 must agree (non-neutral in the right direction)
        agree_count = sum(1 for s in signals.values() if s.direction == target)
        if agree_count < 2:
            return MultiframeVerdict(
                aligned=False,
                direction="neutral",
                alignment_score=alignment_score,
                confidence_boost=0.0,
                signals=signals,
                reason=f"Only {agree_count}/3 timeframes agree on {target}",
            )

        # Alignment threshold gate
        if alignment_score < ALIGNMENT_THRESHOLD:
            return MultiframeVerdict(
                aligned=False,
                direction="neutral",
                alignment_score=alignment_score,
                confidence_boost=0.0,
                signals=signals,
                reason=f"Alignment score {alignment_score:.2f} < threshold {ALIGNMENT_THRESHOLD}",
            )

        # Confidence boost: all 3 agree → 1.15x, 2 agree → 1.05x
        if agree_count == 3:
            boost = 1.10 + 0.10 * alignment_score  # 1.10 – 1.20
        else:
            boost = 1.00 + 0.08 * alignment_score  # 1.00 – 1.08

        direction = "buy" if target == "bullish" else "sell"
        return MultiframeVerdict(
            aligned=True,
            direction=direction,
            alignment_score=round(alignment_score, 4),
            confidence_boost=round(boost, 4),
            signals=signals,
            reason=f"{agree_count}/3 timeframes aligned, score={alignment_score:.2f}",
        )

    def confirm_from_features(
        self,
        features: pd.Series,
        proposed_direction: str = "buy",
    ) -> MultiframeVerdict:
        """
        Lightweight confirmation using pre-computed features when
        raw OHLCV is not available. Extracts what it can from the
        feature Series and runs partial analysis.
        """
        # Build pseudo-signals from available feature columns
        signals: Dict[str, TimeframeSignal] = {}
        target = "bullish" if proposed_direction == "buy" else "bearish"

        # D1 from daily features
        d1_votes: List[float] = []
        d1_ind: Dict[str, float] = {}
        for col, scale in [("close_vs_sma_200", 0.10), ("close_vs_sma_50", 0.07)]:
            val = features.get(col)
            if val is not None and pd.notna(val):
                d1_ind[col] = float(val)
                d1_votes.append(np.clip(float(val) / scale, -1, 1))
        macd = features.get("macd_hist")
        px = features.get("close", 1)
        if macd is not None and pd.notna(macd) and px and px > 0:
            d1_ind["macd_hist"] = float(macd)
            d1_votes.append(np.clip(float(macd) / (float(px) * 0.01), -1, 1))

        if d1_votes:
            avg = float(np.mean(d1_votes))
            d1_dir = "bullish" if avg > 0.10 else "bearish" if avg < -0.10 else "neutral"
            signals["D1"] = TimeframeSignal("D1", d1_dir, min(1.0, abs(avg)), d1_ind)
        else:
            signals["D1"] = TimeframeSignal("D1", "neutral", 0.0, {})

        # H1 from hourly-ish features
        h1_votes: List[float] = []
        h1_ind: Dict[str, float] = {}
        rsi = features.get("rsi_14")
        if rsi is not None and pd.notna(rsi):
            h1_ind["rsi_14"] = float(rsi)
            if rsi < 30:
                h1_votes.append((30 - rsi) / 30)
            elif rsi > 70:
                h1_votes.append(-(rsi - 70) / 30)
            else:
                h1_votes.append((rsi - 50) / 40)
        bb = features.get("bb_position")
        if bb is not None and pd.notna(bb):
            h1_ind["bb_position"] = float(bb)
            h1_votes.append(np.clip((0.5 - float(bb)) * 2, -1, 1))

        if h1_votes:
            avg = float(np.mean(h1_votes))
            h1_dir = "bullish" if avg > 0.08 else "bearish" if avg < -0.08 else "neutral"
            signals["H1"] = TimeframeSignal("H1", h1_dir, min(1.0, abs(avg)), h1_ind)
        else:
            signals["H1"] = TimeframeSignal("H1", "neutral", 0.0, {})

        # M5 from momentum features (best approximation)
        m5_votes: List[float] = []
        m5_ind: Dict[str, float] = {}
        mom = features.get("momentum_20d")
        if mom is not None and pd.notna(mom):
            m5_ind["momentum_proxy"] = float(mom)
            m5_votes.append(np.clip(float(mom) / 0.05, -1, 1))
        accel = features.get("price_acceleration")
        if accel is not None and pd.notna(accel):
            m5_ind["acceleration_proxy"] = float(accel)
            m5_votes.append(np.clip(float(accel) / 0.03, -1, 1))

        if m5_votes:
            avg = float(np.mean(m5_votes))
            m5_dir = "bullish" if avg > 0.05 else "bearish" if avg < -0.05 else "neutral"
            signals["M5"] = TimeframeSignal("M5", m5_dir, min(1.0, abs(avg)), m5_ind)
        else:
            signals["M5"] = TimeframeSignal("M5", "neutral", 0.0, {})

        # Run same alignment logic
        alignment = 0.0
        for tf, sig in signals.items():
            w = TIMEFRAME_WEIGHTS[tf]
            if sig.direction == target:
                alignment += w * sig.strength
            elif sig.direction == "neutral":
                pass
            else:
                alignment -= w * sig.strength

        alignment_score = max(0.0, min(1.0, (alignment + 1) / 2))

        d1_opposes = signals["D1"].direction not in ("neutral", target)
        agree_count = sum(1 for s in signals.values() if s.direction == target)

        if d1_opposes:
            return MultiframeVerdict(False, "neutral", alignment_score, 0.0, signals,
                                     f"D1 opposes ({signals['D1'].direction} vs {target})")
        if agree_count < 2:
            return MultiframeVerdict(False, "neutral", alignment_score, 0.0, signals,
                                     f"Only {agree_count}/3 agree on {target}")
        if alignment_score < ALIGNMENT_THRESHOLD:
            return MultiframeVerdict(False, "neutral", alignment_score, 0.0, signals,
                                     f"Alignment {alignment_score:.2f} < {ALIGNMENT_THRESHOLD}")

        boost = (1.10 + 0.10 * alignment_score) if agree_count == 3 else (1.00 + 0.08 * alignment_score)
        direction = "buy" if target == "bullish" else "sell"
        return MultiframeVerdict(True, direction, round(alignment_score, 4),
                                 round(boost, 4), signals,
                                 f"{agree_count}/3 aligned (feature-based), score={alignment_score:.2f}")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def reason_missing(tf: str, msg: str) -> Dict[str, float]:
    """Return an indicators dict noting why analysis was skipped."""
    return {"_skip_reason": hash(msg) % 1000}  # lightweight marker


def _sma(values: np.ndarray, period: int) -> Optional[float]:
    if len(values) < period:
        return None
    return float(np.mean(values[-period:]))


def _ema(values: np.ndarray, period: int) -> Optional[float]:
    if len(values) < period:
        return None
    alpha = 2.0 / (period + 1)
    ema = float(values[-period])
    for v in values[-period + 1:]:
        ema = alpha * float(v) + (1 - alpha) * ema
    return ema


def _ema_from_val(values: np.ndarray, period: int, seed: float) -> float:
    alpha = 2.0 / (period + 1)
    ema = seed
    for v in values:
        ema = alpha * float(v) + (1 - alpha) * ema
    return ema


def _rsi(close: np.ndarray, period: int = 14) -> Optional[float]:
    if len(close) < period + 1:
        return None
    deltas = np.diff(close[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _adx(df: pd.DataFrame, period: int = 14) -> Tuple[Optional[float], float, float]:
    if len(df) < period * 2:
        return None, 0.0, 0.0
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1])))
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]),
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]),
                        np.maximum(low[:-1] - low[1:], 0), 0)

    atr = float(np.mean(tr[-period:]))
    if atr == 0:
        return None, 0.0, 0.0
    di_plus = float(np.mean(plus_dm[-period:])) / atr * 100
    di_minus = float(np.mean(minus_dm[-period:])) / atr * 100
    di_sum = di_plus + di_minus
    if di_sum == 0:
        return 0.0, di_plus, di_minus
    dx = abs(di_plus - di_minus) / di_sum * 100
    return dx, di_plus, di_minus


def _stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[Optional[float], Optional[float]]:
    if len(df) < k_period + d_period:
        return None, None
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    k_values = []
    for i in range(d_period):
        idx = -(d_period - i)
        if idx == 0:
            h = high[-k_period:]
            l = low[-k_period:]
            c = close[-1]
        else:
            h = high[-(k_period - idx):idx] if idx != 0 else high[-k_period:]
            l = low[-(k_period - idx):idx] if idx != 0 else low[-k_period:]
            c = close[idx]
        hh = float(np.max(h))
        ll = float(np.min(l))
        if hh == ll:
            k_values.append(50.0)
        else:
            k_values.append((c - ll) / (hh - ll) * 100)

    stoch_k = k_values[-1]
    stoch_d = float(np.mean(k_values))
    return stoch_k, stoch_d
