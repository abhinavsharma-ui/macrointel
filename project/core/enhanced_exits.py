"""
Enhanced exit management for live and paper positions.

The manager is intentionally tolerant of multiple position shapes so it can
work with the lightweight state used across the current trading modules.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Optional


class EnhancedExitManager:
    """
    Manages exits across 7 different conditions.
    """

    STOP_LOSS_ATR_MULTIPLIER = 2.0
    TAKE_PROFIT_ATR_MULTIPLIER = 3.5
    TRAILING_STOP_ATR_MULTIPLIER = 1.5
    TRAILING_ACTIVATION_PCT = 0.02
    DAY_TIME_EXIT_AFTER = timedelta(minutes=90)
    DAY_TIME_EXIT_MIN_GAIN_PCT = 0.005
    SWING_TIME_EXIT_AFTER = timedelta(days=5)
    SWING_TIME_EXIT_MIN_GAIN_PCT = 0.01
    SIGNAL_DECAY_RATIO = 0.40
    CORRELATION_EXIT_THRESHOLD = 0.80
    FAST_LANES = {"day", "intraday", "crypto", "scalp", "scalper"}

    def evaluate_exit(
        self,
        position: Dict,
        current_price: float,
        current_regime: str,
        current_signal_strength: float,
    ) -> Dict:
        """
        Returns:
        {
            "should_exit": bool,
            "exit_reason": str,
            "priority": int (1=highest, 7=lowest),
            "price": float (exit price if market order, or target)
        }
        """
        pos = dict(position or {})
        price = self._resolve_price(pos, current_price)
        pos["current_price"] = price
        pos["_side"] = self._normalize_side(pos)

        checks = [
            self._check_stop_loss(pos, price),
            self._check_take_profit(pos, price),
            self._check_trailing_stop(pos, price),
            self._check_time_exit(pos),
            self._check_regime_exit(pos, current_regime),
            self._check_signal_decay(pos, current_signal_strength),
            self._check_correlation_exit(pos),
        ]

        for check in sorted(checks, key=lambda item: item["priority"]):
            if check["should_exit"]:
                return check

        return {
            "should_exit": False,
            "exit_reason": "none",
            "priority": 0,
            "price": price,
        }

    def _check_stop_loss(self, pos: Dict, price: float) -> Dict:
        """Highest priority: hard stop loss at 2x ATR from entry."""
        entry_price = self._entry_price(pos)
        side = pos["_side"]
        atr = self._safe_float(self._value(pos, "atr_at_entry", "atr"), 0.0)
        explicit_stop = self._value(pos, "stop_loss_price", "stop_loss", "stop")

        if explicit_stop is not None:
            stop_loss = self._safe_float(explicit_stop, entry_price)
        elif atr > 0 and entry_price > 0:
            stop_loss = (
                entry_price - (self.STOP_LOSS_ATR_MULTIPLIER * atr)
                if side == "long"
                else entry_price + (self.STOP_LOSS_ATR_MULTIPLIER * atr)
            )
        else:
            return self._no_exit(1)

        if (side == "long" and price <= stop_loss) or (side == "short" and price >= stop_loss):
            return self._exit_result("stop_loss", 1, stop_loss)
        return self._no_exit(1)

    def _check_take_profit(self, pos: Dict, price: float) -> Dict:
        """Take profit at 3.5x ATR from entry."""
        entry_price = self._entry_price(pos)
        side = pos["_side"]
        atr = self._safe_float(self._value(pos, "atr_at_entry", "atr"), 0.0)
        explicit_target = self._value(pos, "take_profit_price", "take_profit", "target_price")

        if explicit_target is not None:
            take_profit = self._safe_float(explicit_target, entry_price)
        elif atr > 0 and entry_price > 0:
            take_profit = (
                entry_price + (self.TAKE_PROFIT_ATR_MULTIPLIER * atr)
                if side == "long"
                else entry_price - (self.TAKE_PROFIT_ATR_MULTIPLIER * atr)
            )
        else:
            return self._no_exit(2)

        if (side == "long" and price >= take_profit) or (side == "short" and price <= take_profit):
            return self._exit_result("take_profit", 2, take_profit)
        return self._no_exit(2)

    def _check_trailing_stop(self, pos: Dict, price: float) -> Dict:
        """After +2% gain, trail at 1.5x ATR."""
        entry_price = self._entry_price(pos)
        side = pos["_side"]
        unrealized_pnl_pct = self._unrealized_pnl_pct(side, entry_price, price)

        if unrealized_pnl_pct < self.TRAILING_ACTIVATION_PCT:
            return self._no_exit(3)

        atr = self._safe_float(self._value(pos, "atr_at_entry", "atr"), 0.0)
        if atr <= 0 or entry_price <= 0:
            return self._no_exit(3)

        if side == "long":
            anchor_price = self._safe_float(
                self._value(pos, "peak_price", "highest_price", "high_watermark"),
                max(entry_price, price),
            )
            trail_stop = max(anchor_price - (self.TRAILING_STOP_ATR_MULTIPLIER * atr), entry_price)
            if price <= trail_stop:
                return self._exit_result("trailing_stop", 3, trail_stop)
        else:
            anchor_price = self._safe_float(
                self._value(pos, "trough_price", "lowest_price", "low_watermark"),
                min(entry_price, price),
            )
            trail_stop = min(anchor_price + (self.TRAILING_STOP_ATR_MULTIPLIER * atr), entry_price)
            if price >= trail_stop:
                return self._exit_result("trailing_stop", 3, trail_stop)

        return self._no_exit(3)

    def _check_time_exit(self, pos: Dict) -> Dict:
        """
        Day/fast trades: exit if >90 min without +0.5% gain.
        Swing trades: exit if >5 days without +1% gain.
        """
        entry_time = self._parse_datetime(self._value(pos, "entry_time", "opened_at"))
        if entry_time is None:
            return self._no_exit(4)

        now = self._now_like(entry_time)
        elapsed = now - entry_time
        price = self._resolve_price(pos, pos.get("current_price"))
        entry_price = self._entry_price(pos)
        side = pos["_side"]
        unrealized_pnl_pct = self._unrealized_pnl_pct(side, entry_price, price)
        lane = self._normalize_lane(pos)

        if lane in self.FAST_LANES:
            if elapsed >= self.DAY_TIME_EXIT_AFTER and unrealized_pnl_pct < self.DAY_TIME_EXIT_MIN_GAIN_PCT:
                return self._exit_result("time_exit_day", 4, price)
        elif elapsed >= self.SWING_TIME_EXIT_AFTER and unrealized_pnl_pct < self.SWING_TIME_EXIT_MIN_GAIN_PCT:
            return self._exit_result("time_exit_swing", 4, price)

        return self._no_exit(4)

    def _check_regime_exit(self, pos: Dict, regime: str) -> Dict:
        """Exit if regime shifts from calm/normal to stressed/crisis."""
        entry_regime = str(self._value(pos, "entry_regime", "regime", default="") or "").strip().lower()
        current_regime = str(regime or self._value(pos, "current_regime", default="") or "").strip().lower()

        if entry_regime in {"calm", "normal"} and current_regime in {"stressed", "crisis"}:
            return self._exit_result(
                f"regime_shift_{entry_regime}_to_{current_regime}",
                5,
                self._resolve_price(pos, pos.get("current_price")),
            )

        return self._no_exit(5)

    def _check_signal_decay(self, pos: Dict, current_strength: float) -> Dict:
        """Exit if signal strength drops below 40% of entry strength."""
        entry_strength = abs(
            self._safe_float(
                self._value(pos, "signal_strength_at_entry", "entry_signal_strength", "signal_strength"),
                0.0,
            )
        )
        live_strength = abs(
            self._safe_float(
                current_strength,
                self._value(pos, "current_signal_strength", "signal_strength"),
            )
        )

        if entry_strength <= 0:
            return self._no_exit(6)

        if live_strength < (entry_strength * self.SIGNAL_DECAY_RATIO):
            return self._exit_result(
                "signal_decay",
                6,
                self._resolve_price(pos, pos.get("current_price")),
            )

        return self._no_exit(6)

    def _check_correlation_exit(self, pos: Dict) -> Dict:
        """Exit if portfolio correlation spikes above 0.80."""
        portfolio_correlation = abs(
            self._safe_float(
                self._value(pos, "portfolio_correlation", "correlation", "pair_correlation"),
                0.0,
            )
        )

        if portfolio_correlation > self.CORRELATION_EXIT_THRESHOLD:
            return self._exit_result(
                "correlation_spike",
                7,
                self._resolve_price(pos, pos.get("current_price")),
            )

        return self._no_exit(7)

    def _entry_price(self, pos: Dict) -> float:
        return self._safe_float(
            self._value(pos, "entry_price", "avg_cost", "filled_price"),
            0.0,
        )

    def _normalize_side(self, pos: Dict) -> str:
        raw_side = str(
            self._value(pos, "side", "position_side", "direction", "action", default="") or ""
        ).strip().lower()
        if raw_side in {"sell", "short"}:
            return "short"
        if raw_side in {"buy", "long"}:
            return "long"

        quantity = self._safe_float(pos.get("quantity"), 0.0)
        return "short" if quantity < 0 else "long"

    def _normalize_lane(self, pos: Dict) -> str:
        lane = str(self._value(pos, "lane", default="") or "").strip().lower()
        if lane:
            return lane

        position_key = str(self._value(pos, "position_key", default="") or "")
        if "::" in position_key:
            return position_key.rsplit("::", 1)[-1].strip().lower()

        return "normal"

    def _resolve_price(self, pos: Dict, explicit_price: Optional[float]) -> float:
        return self._safe_float(
            explicit_price,
            self._value(pos, "current_price", "market_price", "last_price", "reference_price"),
        )

    def _unrealized_pnl_pct(self, side: str, entry_price: float, current_price: float) -> float:
        if entry_price <= 0:
            return 0.0
        if side == "short":
            return (entry_price - current_price) / entry_price
        return (current_price - entry_price) / entry_price

    def _value(self, pos: Dict, *keys: str, default: Any = None) -> Any:
        metadata = pos.get("metadata", {}) if isinstance(pos.get("metadata"), dict) else {}
        for key in keys:
            if key in pos and pos[key] is not None:
                return pos[key]
            if key in metadata and metadata[key] is not None:
                return metadata[key]
        return default

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        fallback = 0.0 if default in (None, "") else default
        try:
            if value is None or value == "":
                return float(fallback)
            return float(value)
        except (TypeError, ValueError):
            return float(fallback)

    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value
        if not value:
            return None

        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned.endswith("Z"):
                cleaned = cleaned[:-1] + "+00:00"
            try:
                return datetime.fromisoformat(cleaned)
            except ValueError:
                return None

        return None

    def _now_like(self, reference_time: datetime) -> datetime:
        if reference_time.tzinfo is not None and reference_time.utcoffset() is not None:
            return datetime.now(reference_time.tzinfo)
        return datetime.now()

    def _exit_result(self, reason: str, priority: int, price: float) -> Dict[str, Any]:
        return {
            "should_exit": True,
            "exit_reason": reason,
            "priority": priority,
            "price": float(price),
        }

    def _no_exit(self, priority: int) -> Dict[str, Any]:
        return {
            "should_exit": False,
            "exit_reason": "none",
            "priority": priority,
            "price": 0.0,
        }
