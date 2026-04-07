"""
Virtual Paper Trading Broker
==============================
"Paper Trade First. Never put real money into a new model."

This module simulates a complete broker with:
  - Virtual account with realistic fills
  - Slippage and commission simulation
  - Position tracking and P&L
  - Performance logging for signal validation
  - Risk management (position limits, stop-losses, max drawdown kill switch)

Use this for 30+ days before any live deployment.
"""

import json
import hashlib
import logging
import os
from datetime import datetime, timezone
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass
class Order:
    symbol: str
    side: OrderSide
    quantity: float
    position_key: Optional[str] = None
    order_type: str = "market"  # "market" | "limit"
    limit_price: Optional[float] = None
    signal_source: str = ""     # Which model generated this signal
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    commission: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    order_id: str = ""
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.order_id:
            self.order_id = f"VPT-{self.symbol}-{int(self.created_at.timestamp())}"


@dataclass
class Position:
    symbol: str
    quantity: float            # Positive = long, negative = short
    avg_cost: float
    position_key: str = ""
    market: str = "US"
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    def __post_init__(self):
        if not self.position_key:
            self.position_key = self.symbol

    @property
    def cost_basis(self) -> float:
        return abs(self.quantity) * self.avg_cost

    def update_market_price(self, current_price: float):
        self.unrealized_pnl = (current_price - self.avg_cost) * self.quantity


class VirtualBroker:
    """
    Complete paper trading simulator.
    
    Designed to be a drop-in replacement for a real broker API.
    Same interface — just doesn't use real money.
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        max_position_pct: float = 0.10,   # Max 10% per position
        max_drawdown_pct: float = 0.20,   # Kill switch at 20% portfolio drawdown
        max_daily_loss_pct: float = 0.02,
        max_consecutive_losses: int = 3,
        market: str = "US",
        session_guardrails_enabled: bool = True,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.max_position_pct = max_position_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_consecutive_losses = max(1, int(max_consecutive_losses))
        self.market = market
        self.session_guardrails_enabled = bool(session_guardrails_enabled)

        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trade_log: List[Dict] = []
        self._kill_switch_activated = False
        self._kill_switch_reason = ""
        self._kill_switch_day = ""
        self._peak_portfolio_value = initial_capital
        self._state_path = Path(os.getenv("PAPER_BROKER_STATE_PATH", "data/paper_broker_state.json"))
        self._realism_enabled = os.getenv("PAPER_EXECUTION_REALISM_ENABLED", "1").strip().lower() not in {"0", "false", "off"}
        self._partial_fill_enabled = os.getenv("PAPER_PARTIAL_FILL_ENABLED", "1").strip().lower() not in {"0", "false", "off"}
        self._partial_fill_entry_only = os.getenv("PAPER_PARTIAL_FILL_ENTRY_ONLY", "1").strip().lower() not in {"0", "false", "off"}
        self._max_spread_pct = max(0.0, float(os.getenv("PAPER_MAX_SPREAD_PCT", "0.45")))
        self._max_tick_age_seconds = max(0.0, float(os.getenv("PAPER_MAX_TICK_AGE_SECONDS", "8.0")))
        self._max_signal_age_seconds = max(0.0, float(os.getenv("PAPER_MAX_SIGNAL_AGE_SECONDS", "120.0")))
        self._base_rejection_probability = min(0.60, max(0.0, float(os.getenv("PAPER_BASE_REJECTION_PROBABILITY", "0.015"))))
        self._partial_fill_min_ratio = min(1.0, max(0.05, float(os.getenv("PAPER_PARTIAL_FILL_MIN_RATIO", "0.35"))))
        self._simulated_latency_ms = max(0.0, float(os.getenv("PAPER_SIMULATED_LATENCY_MS", "220")))
        self._simulated_latency_jitter_ms = max(0.0, float(os.getenv("PAPER_SIMULATED_LATENCY_JITTER_MS", "180")))

        from core.backtesting import TransactionCostModel
        self._cost_model = TransactionCostModel()

        self._load_state()
        logger.info(f"Virtual broker initialized: ${initial_capital:,.0f} capital")

    @staticmethod
    def _order_position_key(order: Order) -> str:
        if order.position_key:
            return str(order.position_key)
        metadata = order.metadata or {}
        for field in ("position_key", "signal_key"):
            value = metadata.get(field)
            if value:
                return str(value)
        return str(order.symbol)

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            if value in (None, ""):
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    # ── Order execution ───────────────────────────────────────

    def submit_order(self, order: Order, current_price: float) -> Dict:
        """Submit and simulate-fill a market order."""
        self._maybe_reset_session_kill_switch()
        if self._kill_switch_activated:
            reason = self._kill_switch_reason or "portfolio protection"
            return {"status": "rejected", "reason": f"Kill switch active — {reason}"}

        # Risk checks
        rejection = self._risk_check(order, current_price)
        if rejection:
            order.status = OrderStatus.REJECTED
            logger.warning(f"Order rejected: {order.symbol} — {rejection}")
            return {"status": "rejected", "reason": rejection}

        requested_quantity = float(order.quantity)
        fill_outcome = self._simulate_fill_outcome(order, current_price)
        if fill_outcome.get("rejected"):
            order.status = OrderStatus.REJECTED
            reason = str(fill_outcome.get("reason") or "market_conditions")
            logger.warning(f"Order rejected: {order.symbol} â€” {reason}")
            return {"status": "rejected", "reason": reason}

        # Simulate fill with slippage and imperfect liquidity.
        fill_price = float(fill_outcome["fill_price"])
        order.quantity = float(fill_outcome["filled_quantity"])
        order.metadata = {
            **dict(order.metadata or {}),
            "requested_quantity": requested_quantity,
            "fill_ratio": float(fill_outcome.get("fill_ratio", 1.0) or 1.0),
            "simulated_latency_ms": float(fill_outcome.get("simulated_latency_ms", 0.0) or 0.0),
            "partial_fill": bool(fill_outcome.get("partial_fill", False)),
        }
        execution_market = str((order.metadata or {}).get("market") or self.market or "US")
        trade_value = abs(order.quantity) * fill_price
        commission = self._cost_model.compute_cost(
            trade_value=trade_value,
            market=execution_market,
            side=order.side.value,
            shares=abs(order.quantity),
            order_size_usd=trade_value,
        ) * trade_value

        # Execute the fill
        order.filled_price = fill_price
        order.commission = commission
        order.filled_at = datetime.now(timezone.utc)
        order.status = OrderStatus.FILLED

        realized_pnl = self._update_position(order)
        self._update_cash(order, fill_price, commission)
        self.orders.append(order)

        reference_price = self._safe_float((order.metadata or {}).get("execution_reference_price"), current_price)
        trade_record = {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "position_key": self._order_position_key(order),
            "side": order.side,
            "quantity": order.quantity,
            "requested_quantity": round(requested_quantity, 8),
            "fill_ratio": round(float(order.metadata.get("fill_ratio", 1.0) or 1.0), 6),
            "partial_fill": bool(order.metadata.get("partial_fill", False)),
            "fill_price": round(fill_price, 4),
            "commission": round(commission, 4),
            "slippage_pct": round(((fill_price / max(reference_price, 1e-9)) - 1) * 100, 4),
            "reference_price": round(reference_price, 4),
            "simulated_latency_ms": round(float(order.metadata.get("simulated_latency_ms", 0.0) or 0.0), 2),
            "market": execution_market,
            "cash_after": round(self.cash, 2),
            "portfolio_value": round(self.portfolio_value, 2),
            "signal_source": order.signal_source,
            "realized_pnl": round(realized_pnl, 4),
            "filled_at": order.filled_at.isoformat(),
            "metadata": dict(order.metadata or {}),
        }
        self.trade_log.append(trade_record)
        self._refresh_session_risk_controls()
        self._persist_state()

        logger.info(
            f"FILL: {order.side.upper()} {order.quantity}/{requested_quantity}x {order.symbol} "
            f"@ ${fill_price:.2f} (slippage: {trade_record['slippage_pct']:.3f}%)"
        )

        return {"status": "filled", "trade": trade_record}

    def update_prices(self, prices: Dict[str, float]) -> Dict:
        """
        Update all position prices and compute P&L.
        Call this on every price tick.
        """
        self._maybe_reset_session_kill_switch()
        total_unrealized = 0
        for position_key, pos in self.positions.items():
            latest_price = None
            if position_key in prices:
                latest_price = prices[position_key]
            elif pos.symbol in prices:
                latest_price = prices[pos.symbol]
            if latest_price is not None:
                pos.update_market_price(latest_price)
                total_unrealized += pos.unrealized_pnl

        # Kill switch check
        pv = self.portfolio_value
        self._peak_portfolio_value = max(self._peak_portfolio_value, pv)
        current_drawdown = (pv - self._peak_portfolio_value) / self._peak_portfolio_value

        if current_drawdown <= -self.max_drawdown_pct and not self._kill_switch_activated:
            self._kill_switch_activated = True
            self._kill_switch_reason = "max drawdown breached"
            self._kill_switch_day = datetime.now(timezone.utc).date().isoformat()
            logger.critical(
                f"⚠️  KILL SWITCH ACTIVATED: Portfolio drawdown {current_drawdown:.1%} "
                f"exceeds limit {-self.max_drawdown_pct:.1%}"
            )

        self._refresh_session_risk_controls()

        return {
            "portfolio_value": round(pv, 2),
            "cash": round(self.cash, 2),
            "unrealized_pnl": round(total_unrealized, 2),
            "drawdown_pct": round(current_drawdown * 100, 2),
            "kill_switch": self._kill_switch_activated,
            "kill_switch_reason": self._kill_switch_reason,
        }

    def close_position(self, symbol: str, current_price: float) -> Dict:
        """Close an existing position."""
        position_key = None
        if symbol in self.positions:
            position_key = symbol
        else:
            matches = [
                key
                for key, position in self.positions.items()
                if position.symbol == symbol and position.quantity != 0
            ]
            if len(matches) == 1:
                position_key = matches[0]
            elif len(matches) > 1:
                return {"status": "ambiguous_position", "matches": matches}
        if not position_key or position_key not in self.positions:
            return {"status": "no_position"}

        pos = self.positions[position_key]
        side = OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY
        order = Order(
            symbol=pos.symbol,
            side=side,
            quantity=abs(pos.quantity),
            position_key=position_key,
            signal_source="manual_close",
        )
        return self.submit_order(order, current_price)

    def has_open_position(self, *, symbol: Optional[str] = None, position_key: Optional[str] = None) -> bool:
        if position_key is not None:
            position = self.positions.get(position_key)
            return bool(position and position.quantity != 0)
        if symbol is not None:
            return any(position.quantity != 0 and position.symbol == symbol for position in self.positions.values())
        return False

    # ── Portfolio reporting ───────────────────────────────────

    @property
    def portfolio_value(self) -> float:
        """Total portfolio value including all positions at current prices."""
        position_value = sum(
            pos.quantity * (pos.avg_cost + pos.unrealized_pnl / max(abs(pos.quantity), 1))
            for pos in self.positions.values()
        )
        return self.cash + position_value

    @property
    def total_return_pct(self) -> float:
        return (self.portfolio_value / self.initial_capital - 1) * 100

    def get_summary(self) -> Dict:
        """Complete portfolio summary."""
        self._maybe_reset_session_kill_switch()
        self._refresh_session_risk_controls()
        realized_pnl = sum(t.get("realized_pnl", 0) for t in self.trade_log)
        winning_trades = sum(1 for t in self.trade_log if t.get("realized_pnl", 0) > 0)
        total_trades = len([t for t in self.trade_log if t.get("realized_pnl") is not None])
        daily_realized_pnl, daily_loss_pct, consecutive_losses = self._session_risk_snapshot()

        return {
            "initial_capital": self.initial_capital,
            "current_portfolio_value": round(self.portfolio_value, 2),
            "cash": round(self.cash, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "realized_pnl": round(realized_pnl, 2),
            "daily_realized_pnl": round(daily_realized_pnl, 2),
            "daily_loss_pct": round(daily_loss_pct * 100, 2),
            "consecutive_losses": consecutive_losses,
            "total_commissions_paid": round(sum(t.get("commission", 0) for t in self.trade_log), 2),
            "open_positions": sum(1 for pos in self.positions.values() if getattr(pos, "quantity", 0) != 0),
            "total_trades": len(self.orders),
            "win_rate_pct": round(winning_trades / max(total_trades, 1) * 100, 1),
            "kill_switch_active": self._kill_switch_activated,
            "kill_switch_reason": self._kill_switch_reason,
            "session_guardrails_enabled": self.session_guardrails_enabled,
            "execution_mode": "paper",
            "execution_realism": {
                "enabled": self._realism_enabled,
                "partial_fills": self._partial_fill_enabled,
                "partial_fill_entry_only": self._partial_fill_entry_only,
                "max_spread_pct": self._max_spread_pct,
                "max_tick_age_seconds": self._max_tick_age_seconds,
                "max_signal_age_seconds": self._max_signal_age_seconds,
                "simulated_latency_ms": self._simulated_latency_ms,
                "simulated_latency_jitter_ms": self._simulated_latency_jitter_ms,
            },
            "positions": {
                position_key: {
                    "symbol": pos.symbol,
                    "position_key": position_key,
                    "quantity": pos.quantity,
                    "avg_cost": round(pos.avg_cost, 4),
                    "unrealized_pnl": round(pos.unrealized_pnl, 2),
                    "market": pos.market,
                }
                for position_key, pos in self.positions.items()
            },
        }

    def get_trade_dataframe(self) -> pd.DataFrame:
        """Return trade history as a DataFrame for analysis."""
        if not self.trade_log:
            return pd.DataFrame()
        return pd.DataFrame(self.trade_log)

    def _session_risk_snapshot(self) -> tuple[float, float, int]:
        today = datetime.now(timezone.utc).date()
        todays_closed_trades: List[float] = []
        for trade in self.trade_log:
            filled_at_raw = trade.get("filled_at")
            if not filled_at_raw:
                continue
            try:
                filled_at = datetime.fromisoformat(str(filled_at_raw))
            except Exception:
                continue
            if filled_at.astimezone(timezone.utc).date() != today:
                continue
            realized_pnl = float(trade.get("realized_pnl", 0.0) or 0.0)
            if abs(realized_pnl) < 1e-9:
                continue
            todays_closed_trades.append(realized_pnl)

        daily_realized_pnl = float(sum(todays_closed_trades))
        daily_loss_pct = max(0.0, -daily_realized_pnl / max(self.initial_capital, 1.0))
        consecutive_losses = 0
        for pnl in reversed(todays_closed_trades):
            if pnl < 0:
                consecutive_losses += 1
            else:
                break
        return daily_realized_pnl, daily_loss_pct, consecutive_losses

    def _maybe_reset_session_kill_switch(self):
        if not self._kill_switch_activated:
            return
        if self._kill_switch_reason not in {"daily loss limit", "consecutive losses"}:
            return
        if not self.session_guardrails_enabled:
            self._kill_switch_activated = False
            self._kill_switch_reason = ""
            self._kill_switch_day = ""
            return
        today_key = datetime.now(timezone.utc).date().isoformat()
        if self._kill_switch_day != today_key:
            self._kill_switch_activated = False
            self._kill_switch_reason = ""
            self._kill_switch_day = ""

    def _refresh_session_risk_controls(self):
        if self._kill_switch_activated and self._kill_switch_reason == "max drawdown breached":
            return
        if not self.session_guardrails_enabled:
            if self._kill_switch_reason in {"daily loss limit", "consecutive losses"}:
                self._kill_switch_activated = False
                self._kill_switch_reason = ""
                self._kill_switch_day = ""
            return
        _, daily_loss_pct, consecutive_losses = self._session_risk_snapshot()
        today_key = datetime.now(timezone.utc).date().isoformat()
        if daily_loss_pct >= self.max_daily_loss_pct:
            self._kill_switch_activated = True
            self._kill_switch_reason = "daily loss limit"
            self._kill_switch_day = today_key
        elif consecutive_losses >= self.max_consecutive_losses:
            self._kill_switch_activated = True
            self._kill_switch_reason = "consecutive losses"
            self._kill_switch_day = today_key
        elif (
            self._kill_switch_reason in {"daily loss limit", "consecutive losses"}
            and self._kill_switch_day == today_key
            and daily_loss_pct < self.max_daily_loss_pct
            and consecutive_losses < self.max_consecutive_losses
        ):
            self._kill_switch_activated = False
            self._kill_switch_reason = ""
            self._kill_switch_day = ""

    # ── Internal helpers ──────────────────────────────────────

    @staticmethod
    def _order_noise(order: Order, salt: str) -> float:
        digest = hashlib.sha256(f"{order.order_id}|{order.symbol}|{salt}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") / float(2**64 - 1)

    @staticmethod
    def _is_crypto_order(order: Order) -> bool:
        symbol = str(order.symbol or "").upper()
        metadata = dict(order.metadata or {})
        market = str(metadata.get("market") or "").upper()
        return market in {"CRYPTO", "BINANCE", "BINANCEUS", "CRYPTOUSD"} or symbol.endswith(("USDT", "USDC", "BUSD"))

    def _simulate_fill_outcome(self, order: Order, market_price: float) -> Dict:
        metadata = dict(order.metadata or {})
        spread_pct = max(0.0, self._safe_float(metadata.get("spread_pct"), 0.0))
        tick_age_seconds = max(0.0, self._safe_float(metadata.get("tick_age_seconds"), 0.0))
        signal_age_seconds = max(0.0, self._safe_float(metadata.get("signal_age_seconds"), 0.0))
        trade_value = abs(order.quantity) * max(market_price, 1e-9)
        position_value_cap = max(self.portfolio_value * self.max_position_pct, 1.0)
        size_pressure = min(1.0, trade_value / position_value_cap)

        if self._realism_enabled:
            if self._max_spread_pct > 0 and spread_pct > self._max_spread_pct:
                return {"rejected": True, "reason": f"spread too wide ({spread_pct:.3f}%)"}
            if self._max_tick_age_seconds > 0 and tick_age_seconds > self._max_tick_age_seconds:
                return {"rejected": True, "reason": f"stale tick ({tick_age_seconds:.1f}s old)"}
            if self._max_signal_age_seconds > 0 and signal_age_seconds > self._max_signal_age_seconds:
                return {"rejected": True, "reason": f"stale signal ({signal_age_seconds:.1f}s old)"}

            rejection_probability = self._base_rejection_probability
            rejection_probability += min(0.18, size_pressure * 0.18)
            if self._max_spread_pct > 0:
                rejection_probability += min(0.18, (spread_pct / max(self._max_spread_pct, 1e-9)) * 0.10)
            if self._max_tick_age_seconds > 0:
                rejection_probability += min(0.16, (tick_age_seconds / max(self._max_tick_age_seconds, 1e-9)) * 0.08)
            if self._order_noise(order, "reject") < rejection_probability:
                return {"rejected": True, "reason": "liquidity miss / queue loss"}

        latency_ms = self._simulated_latency_ms + (self._order_noise(order, "latency") * self._simulated_latency_jitter_ms)
        fill_price = self._simulate_fill_price(order, market_price, latency_ms=latency_ms)
        fill_ratio = 1.0

        eligible_for_partial = self._partial_fill_enabled and (
            not self._partial_fill_entry_only or order.side == OrderSide.BUY
        )
        if eligible_for_partial:
            partial_probability = min(0.80, 0.08 + (size_pressure * 0.55) + min(0.20, spread_pct * 0.20))
            if self._order_noise(order, "partial_trigger") < partial_probability:
                ratio_draw = self._order_noise(order, "partial_ratio")
                pressure_penalty = min(0.55, size_pressure * 0.45 + min(0.15, spread_pct * 0.08))
                fill_ratio = max(self._partial_fill_min_ratio, 1.0 - pressure_penalty - (ratio_draw * 0.25))

        requested_quantity = abs(float(order.quantity))
        filled_quantity = requested_quantity * fill_ratio
        if self._is_crypto_order(order):
            filled_quantity = round(max(0.0001, filled_quantity), 6)
        else:
            filled_quantity = float(max(1, int(np.floor(filled_quantity))))
        filled_quantity = min(requested_quantity, filled_quantity)
        fill_ratio = filled_quantity / max(requested_quantity, 1e-9)
        return {
            "rejected": False,
            "fill_price": fill_price,
            "filled_quantity": filled_quantity,
            "fill_ratio": fill_ratio,
            "partial_fill": filled_quantity < requested_quantity,
            "simulated_latency_ms": latency_ms,
        }

    def _simulate_fill_price(self, order: Order, market_price: float, *, latency_ms: float = 0.0) -> float:
        """Simulate realistic fill with slippage."""
        from core.backtesting import TransactionCostModel
        model = TransactionCostModel()
        metadata = dict(order.metadata or {})
        best_bid = self._safe_float(metadata.get("best_bid"), 0.0)
        best_ask = self._safe_float(metadata.get("best_ask"), 0.0)
        spread_pct = max(0.0, self._safe_float(metadata.get("spread_pct"), 0.0))
        trade_value = abs(order.quantity) * max(market_price, 1e-9)
        size_impact = min(1.8, (abs(order.quantity) / 2500.0) + (trade_value / 150_000.0))

        if best_bid > 0 and best_ask > 0 and best_ask >= best_bid:
            spread = max(best_ask - best_bid, market_price * model.base_slippage_pct)
            impact = max(spread * 0.35, market_price * model.base_slippage_pct) * (1.0 + size_impact)
            fill_price = (best_ask + impact) if order.side == OrderSide.BUY else max(1e-9, best_bid - impact)
        else:
            impact = max(
                market_price * model.base_slippage_pct,
                market_price * (spread_pct / 100.0) * 0.20,
            ) * (1.0 + size_impact)
            fill_price = (market_price + impact) if order.side == OrderSide.BUY else max(1e-9, market_price - impact)

        if latency_ms > 0:
            latency_impact = market_price * model.base_slippage_pct * min(3.0, latency_ms / 150.0)
            fill_price = (fill_price + latency_impact) if order.side == OrderSide.BUY else max(1e-9, fill_price - latency_impact)

        if order.limit_price is not None and order.order_type == "limit":
            limit_price = float(order.limit_price)
            if order.side == OrderSide.BUY:
                fill_price = min(fill_price, limit_price)
            else:
                fill_price = max(fill_price, limit_price)
        return max(fill_price, 1e-9)

    def _risk_check(self, order: Order, current_price: float) -> Optional[str]:
        """Pre-trade risk checks. Return error message if rejected."""
        order_value = abs(order.quantity) * current_price

        # Insufficient cash
        if order.side == OrderSide.BUY and order_value > self.cash:
            return f"Insufficient cash: need ${order_value:,.0f}, have ${self.cash:,.0f}"

        # Position size limit
        max_position_value = self.portfolio_value * self.max_position_pct
        if order_value > max_position_value:
            return (f"Position too large: ${order_value:,.0f} exceeds "
                    f"${max_position_value:,.0f} limit ({self.max_position_pct:.0%})")

        return None

    def _update_position(self, order: Order) -> float:
        """Update position tracking after a fill."""
        position_key = self._order_position_key(order)
        symbol = order.symbol
        qty = order.quantity if order.side == OrderSide.BUY else -order.quantity
        realized_pnl = 0.0
        execution_market = str((order.metadata or {}).get("market") or self.market or "US")

        if position_key in self.positions:
            pos = self.positions[position_key]
            new_qty = pos.quantity + qty

            if abs(new_qty) < 1e-9:
                realized = abs(pos.quantity) * (order.filled_price - pos.avg_cost) * np.sign(pos.quantity)
                pos.realized_pnl += realized
                realized_pnl = realized
                del self.positions[position_key]
            elif (pos.quantity > 0) == (qty > 0):
                # Adding to existing position — compute new average cost
                total_cost = pos.quantity * pos.avg_cost + qty * order.filled_price
                pos.quantity = new_qty
                pos.avg_cost = total_cost / new_qty
            else:
                # Partial close — realize P&L on closed portion
                closed_qty = min(abs(pos.quantity), abs(qty))
                realized = closed_qty * (order.filled_price - pos.avg_cost) * np.sign(pos.quantity)
                pos.realized_pnl += realized
                realized_pnl = realized
                if abs(qty) > abs(pos.quantity):
                    # Flip position
                    self.positions[position_key] = Position(
                        symbol=symbol,
                        position_key=position_key,
                        quantity=new_qty,
                        avg_cost=order.filled_price,
                        market=execution_market,
                    )
                else:
                    pos.quantity = new_qty
        else:
            self.positions[position_key] = Position(
                symbol=symbol,
                position_key=position_key,
                quantity=qty,
                avg_cost=order.filled_price,
                market=execution_market,
            )
        return realized_pnl

    def _update_cash(self, order: Order, fill_price: float, commission: float):
        """Update cash balance after trade."""
        trade_value = abs(order.quantity) * fill_price
        if order.side == OrderSide.BUY:
            self.cash -= (trade_value + commission)
        else:
            self.cash += (trade_value - commission)

    def _serialize_state(self) -> Dict:
        return {
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "max_position_pct": self.max_position_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "max_consecutive_losses": self.max_consecutive_losses,
            "market": self.market,
            "session_guardrails_enabled": self.session_guardrails_enabled,
            "kill_switch_active": self._kill_switch_activated,
            "kill_switch_reason": self._kill_switch_reason,
            "kill_switch_day": self._kill_switch_day,
            "peak_portfolio_value": self._peak_portfolio_value,
            "positions": {
                position_key: {
                    "symbol": pos.symbol,
                    "position_key": position_key,
                    "quantity": pos.quantity,
                    "avg_cost": pos.avg_cost,
                    "market": pos.market,
                    "opened_at": pos.opened_at.isoformat(),
                    "unrealized_pnl": pos.unrealized_pnl,
                    "realized_pnl": pos.realized_pnl,
                }
                for position_key, pos in self.positions.items()
            },
            "orders": [
                {
                    "symbol": order.symbol,
                    "position_key": order.position_key,
                    "side": order.side.value,
                    "quantity": order.quantity,
                    "order_type": order.order_type,
                    "limit_price": order.limit_price,
                    "signal_source": order.signal_source,
                    "created_at": order.created_at.isoformat(),
                    "filled_at": order.filled_at.isoformat() if order.filled_at else None,
                    "filled_price": order.filled_price,
                    "commission": order.commission,
                    "status": order.status.value,
                    "order_id": order.order_id,
                    "metadata": dict(order.metadata or {}),
                }
                for order in self.orders
            ],
            "trade_log": self.trade_log,
        }

    def _persist_state(self):
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps(self._serialize_state(), indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning(f"Paper broker state save failed: {exc}")

    def _load_state(self):
        if not self._state_path.exists():
            return
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
            self.initial_capital = float(payload.get("initial_capital", self.initial_capital))
            self.cash = float(payload.get("cash", self.cash))
            self.max_position_pct = float(payload.get("max_position_pct", self.max_position_pct))
            self.max_drawdown_pct = float(payload.get("max_drawdown_pct", self.max_drawdown_pct))
            self.max_daily_loss_pct = float(payload.get("max_daily_loss_pct", self.max_daily_loss_pct))
            self.max_consecutive_losses = int(payload.get("max_consecutive_losses", self.max_consecutive_losses))
            self.market = payload.get("market", self.market)
            self.session_guardrails_enabled = bool(
                payload.get("session_guardrails_enabled", self.session_guardrails_enabled)
            )
            self._kill_switch_activated = bool(payload.get("kill_switch_active", self._kill_switch_activated))
            self._kill_switch_reason = str(payload.get("kill_switch_reason", self._kill_switch_reason) or "")
            self._kill_switch_day = str(payload.get("kill_switch_day", self._kill_switch_day) or "")
            self._peak_portfolio_value = float(payload.get("peak_portfolio_value", self._peak_portfolio_value))

            self.positions = {}
            for position_key, raw in (payload.get("positions") or {}).items():
                opened_at_raw = raw.get("opened_at")
                opened_at = datetime.fromisoformat(opened_at_raw) if opened_at_raw else datetime.now(timezone.utc)
                resolved_position_key = str(raw.get("position_key", position_key) or position_key)
                self.positions[resolved_position_key] = Position(
                    symbol=raw.get("symbol", position_key),
                    position_key=resolved_position_key,
                    quantity=float(raw.get("quantity", 0.0)),
                    avg_cost=float(raw.get("avg_cost", 0.0)),
                    market=raw.get("market", self.market),
                    opened_at=opened_at,
                    unrealized_pnl=float(raw.get("unrealized_pnl", 0.0)),
                    realized_pnl=float(raw.get("realized_pnl", 0.0)),
                )

            self.orders = []
            for raw in payload.get("orders") or []:
                order = Order(
                    symbol=raw.get("symbol", ""),
                    side=OrderSide(raw.get("side", "buy")),
                    quantity=float(raw.get("quantity", 0.0)),
                    position_key=raw.get("position_key"),
                    order_type=raw.get("order_type", "market"),
                    limit_price=raw.get("limit_price"),
                    signal_source=raw.get("signal_source", ""),
                    created_at=datetime.fromisoformat(raw["created_at"]) if raw.get("created_at") else datetime.now(timezone.utc),
                    filled_at=datetime.fromisoformat(raw["filled_at"]) if raw.get("filled_at") else None,
                    filled_price=raw.get("filled_price"),
                    commission=float(raw.get("commission", 0.0)),
                    status=OrderStatus(raw.get("status", "pending")),
                    order_id=raw.get("order_id", ""),
                    metadata=dict(raw.get("metadata") or {}),
                )
                self.orders.append(order)

            self.trade_log = list(payload.get("trade_log") or [])
            self._maybe_reset_session_kill_switch()
            self._refresh_session_risk_controls()
            if self.positions or self.trade_log:
                logger.info(
                    f"Paper broker state restored: {len(self.positions)} positions | {len(self.trade_log)} trade records"
                )
        except Exception as exc:
            logger.warning(f"Paper broker state load failed: {exc}")

