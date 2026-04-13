"""
Multi-timeframe confirmation helpers for signal gating.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class MultiframeConfirmer:
    """
    Confirms signals across D1 + H1 + M5 timeframes.
    Score: -3 to +3 (sum of individual timeframe scores)
    """

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}

    def confirm(
        self,
        symbol: str,
        daily_data: pd.DataFrame,
        hourly_data: pd.DataFrame,
        minute_data: pd.DataFrame,
    ) -> Dict[str, float]:
        daily_df = self._prepare_frame(daily_data)
        hourly_df = self._prepare_frame(hourly_data)
        minute_df = self._prepare_frame(minute_data)

        cache_key = self._cache_key(symbol, daily_df, hourly_df, minute_df)
        cached = self._cache.get(symbol)
        if cached and cached.get("cache_key") == cache_key:
            return dict(cached["result"])

        d1_score = self._score_daily(daily_df)
        h1_score = self._score_hourly(hourly_df)
        m5_score = self._score_fivemin(minute_df)
        total = d1_score + h1_score + m5_score
        bullish_alignment = all(score > 0 for score in (d1_score, h1_score, m5_score))
        bearish_alignment = all(score < 0 for score in (d1_score, h1_score, m5_score))

        if bullish_alignment:
            action = "buy"
            confidence = min(1.0, total / 3.0)
        elif bearish_alignment:
            action = "sell"
            confidence = min(1.0, abs(total) / 3.0)
        elif total > 0:
            action = "buy_half_size"
            confidence = 0.5
        elif total < 0:
            action = "sell_half_size"
            confidence = 0.5
        else:
            action = "skip"
            confidence = 0.0

        result = {
            "d1_score": float(d1_score),
            "h1_score": float(h1_score),
            "m5_score": float(m5_score),
            "total_score": float(total),
            "action": action,
            "confidence": float(confidence),
            "reason": self._build_reason(d1_score, h1_score, m5_score, action),
            "d1_reason": self._reason_daily(d1_score),
            "h1_reason": self._reason_hourly(h1_score),
            "m5_reason": self._reason_fivemin(m5_score),
        }
        self._cache[symbol] = {"cache_key": cache_key, "result": dict(result)}
        return result

    def _score_daily(self, df: pd.DataFrame) -> float:
        """
        Trend direction: D1 score is +1 (uptrend), 0 (sideways), -1 (downtrend)
        """
        if df.empty or "close" not in df.columns or len(df) < 50:
            return 0.0

        close_series = pd.to_numeric(df["close"], errors="coerce").dropna()
        if len(close_series) < 50:
            return 0.0

        close = float(close_series.iloc[-1])
        ema20 = float(close_series.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(close_series.ewm(span=50, adjust=False).mean().iloc[-1])

        if close > ema20 > ema50:
            return 1.0
        if close < ema20 < ema50:
            return -1.0
        return 0.0

    def _score_hourly(self, df: pd.DataFrame) -> float:
        """
        Entry timing: H1 score based on pullback recovery or exhaustion.
        """
        if df.empty or "close" not in df.columns or len(df) < 20:
            return 0.0

        close_series = pd.to_numeric(df["close"], errors="coerce").dropna()
        if len(close_series) < 20:
            return 0.0

        rsi = self._compute_rsi(close_series, period=14).dropna()
        if len(rsi) < 6:
            return 0.0

        recent_rsi = rsi.iloc[-6:]
        rsi_now = float(recent_rsi.iloc[-1])
        rsi_min_6h = float(recent_rsi.min())
        rsi_max_6h = float(recent_rsi.max())

        if rsi_min_6h < 40 and rsi_now > rsi_min_6h:
            return 1.0
        if rsi_max_6h > 60 and rsi_now < rsi_max_6h:
            return -1.0
        return 0.0

    def _score_fivemin(self, df: pd.DataFrame) -> float:
        """
        Execution pressure: M5 score based on volume confirmation.
        """
        if df.empty or "close" not in df.columns or "volume" not in df.columns or len(df) < 5:
            return 0.0

        volume = pd.to_numeric(df["volume"], errors="coerce").dropna()
        close = pd.to_numeric(df["close"], errors="coerce").dropna()
        if len(volume) < 5 or len(close) < 2:
            return 0.0

        vol_last_5 = float(volume.iloc[-5:].mean())
        vol_20 = float(volume.iloc[-20:].mean()) if len(volume) >= 20 else float(volume.mean())
        if vol_20 <= 0:
            return 0.0

        close_change = float(close.iloc[-1] - close.iloc[-2])
        if vol_last_5 > (vol_20 * 1.5) and close_change > 0:
            return 1.0
        if vol_last_5 > (vol_20 * 1.5) and close_change < 0:
            return -1.0
        return 0.0

    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.where(avg_loss != 0, 100.0)
        rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
        return rsi

    def _reason_daily(self, score: float) -> str:
        if score > 0.5:
            return "uptrend"
        if score < -0.5:
            return "downtrend"
        return "sideways"

    def _reason_hourly(self, score: float) -> str:
        if score > 0.5:
            return "pullback_recovery"
        if score < -0.5:
            return "exhaustion"
        return "neutral"

    def _reason_fivemin(self, score: float) -> str:
        if score > 0.5:
            return "buying_pressure"
        if score < -0.5:
            return "selling_pressure"
        return "low_volume"

    def _build_reason(self, d1_score: float, h1_score: float, m5_score: float, action: str) -> str:
        return (
            f"{action}:"
            f"d1={self._reason_daily(d1_score)},"
            f"h1={self._reason_hourly(h1_score)},"
            f"m5={self._reason_fivemin(m5_score)}"
        )

    def _prepare_frame(self, data: Any) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            frame = data.copy()
        elif isinstance(data, (list, tuple)):
            try:
                frame = pd.DataFrame(data)
            except (TypeError, ValueError):
                return pd.DataFrame()
        elif isinstance(data, dict):
            try:
                frame = pd.DataFrame(data)
            except (TypeError, ValueError):
                return pd.DataFrame()
        else:
            return pd.DataFrame()

        if frame.empty:
            return pd.DataFrame()

        renamed = {column: str(column).strip().lower() for column in frame.columns}
        frame = frame.rename(columns=renamed)
        return frame.sort_index()

    def _cache_key(
        self,
        symbol: str,
        daily_df: pd.DataFrame,
        hourly_df: pd.DataFrame,
        minute_df: pd.DataFrame,
    ) -> str:
        def _frame_token(frame: pd.DataFrame) -> str:
            if frame.empty:
                return "empty"
            last_index = frame.index[-1]
            last_close = frame["close"].iloc[-1] if "close" in frame.columns else "na"
            return f"{len(frame)}:{last_index}:{last_close}"

        return "|".join(
            [
                str(symbol),
                _frame_token(daily_df),
                _frame_token(hourly_df),
                _frame_token(minute_df),
            ]
        )
