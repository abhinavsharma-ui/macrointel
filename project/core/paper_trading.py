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
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from zoneinfo import ZoneInfo
    _EASTERN_TZ = ZoneInfo("America/New_York")
except Exception:
    _EASTERN_TZ = timezone(timedelta(hours=-5))


def _eastern_date_str() -> str:
    return datetime.now(_EASTERN_TZ).strftime("%Y-%m-%d")


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
        max_consecutive_losses: int = 8,
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
        self._start_time = datetime.now(timezone.utc)
        self._consecutive_loss_warmup = timedelta(minutes=10)
        self._kill_switch_cooldown = timedelta(hours=2)
        self._consecutive_losses = 0
        self._kill_switch_activated = False
        self._kill_switch_reason = ""
        self._kill_switch_day = ""
        self._kill_switch_activated_at: Optional[datetime] = None
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
        self._headline_win_rate_window = max(5, int(os.getenv("PAPER_WIN_RATE_WINDOW_TRADES", "50")))
        self._headline_win_rate_min_trades = max(1, int(os.getenv("PAPER_WIN_RATE_MIN_TRADES", "5")))

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

    def _is_consecutive_loss_warmup_active(self) -> bool:
        return datetime.now(timezone.utc) - self._start_time < self._consecutive_loss_warmup

    def _set_kill_switch(
        self,
        *,
        activated: bool,
        reason: str = "",
        day: str = "",
        activated_at: Optional[datetime] = None,
    ) -> None:
        self._kill_switch_activated = bool(activated)
        self._kill_switch_reason = str(reason or "")
        self._kill_switch_day = str(day or "")
        self._kill_switch_activated_at = activated_at if self._kill_switch_activated else None

    def _reset_consecutive_loss_guardrail(self, reset_reason: str) -> None:
        was_active = self._kill_switch_activated and self._kill_switch_reason == "consecutive losses"
        self._consecutive_losses = 0
        if was_active:
            logger.warning(reset_reason)
            self._set_kill_switch(activated=False)

    def _record_closed_trade_result(self, realized_pnl: float, closing_fill: bool) -> None:
        if not closing_fill:
            return
        if realized_pnl < -1e-9:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

    def _persist_if_state_changed(self, previous_state: tuple) -> None:
        current_state = (
            self._kill_switch_activated,
            self._kill_switch_reason,
            self._kill_switch_day,
            self._kill_switch_activated_at.isoformat() if self._kill_switch_activated_at else None,
            self._consecutive_losses,
        )
        if current_state != previous_state:
            self._persist_state()

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

        # Simulate fill with live-quote slippage and imperfect liquidity.
        fill_price = float(fill_outcome["fill_price"])
        order.quantity = float(fill_outcome["filled_quantity"])
        execution_market = str((order.metadata or {}).get("market") or self.market or "US")
        execution_mode = str(fill_outcome.get("execution_mode") or self._execution_mode(order))
        fee_pct_override = (order.metadata or {}).get("fee_pct_override")
        if fee_pct_override is None and execution_market.upper() in {"CRYPTO", "BINANCE", "BINANCEUS", "CRYPTOUSD", "BYBIT"}:
            fee_pct_override = (order.metadata or {}).get(
                "maker_fee_pct" if execution_mode == "maker" else "taker_fee_pct"
            )
        order.metadata = {
            **dict(order.metadata or {}),
            "requested_quantity": requested_quantity,
            "fill_ratio": float(fill_outcome.get("fill_ratio", 1.0) or 1.0),
            "simulated_latency_ms": float(fill_outcome.get("simulated_latency_ms", 0.0) or 0.0),
            "partial_fill": bool(fill_outcome.get("partial_fill", False)),
            "execution_mode": execution_mode,
            "execution_reference_price": float(fill_outcome.get("reference_price", current_price) or current_price),
            "quoted_spread_bps": float(fill_outcome.get("quoted_spread_bps", 0.0) or 0.0),
            "liquidity_ratio": float(fill_outcome.get("liquidity_ratio", 0.0) or 0.0),
            "queue_pressure": float(fill_outcome.get("queue_pressure", 0.0) or 0.0),
            "adverse_selection": float(fill_outcome.get("adverse_selection", 0.0) or 0.0),
        }
        trade_value = abs(order.quantity) * fill_price
        commission = self._cost_model.compute_explicit_fee(
            trade_value=trade_value,
            market=execution_market,
            side=order.side.value,
            shares=abs(order.quantity),
            execution_mode=execution_mode,
            fee_pct_override=fee_pct_override,
        )

        # Execute the fill
        order.filled_price = fill_price
        order.commission = commission
        order.filled_at = datetime.now(timezone.utc)
        order.status = OrderStatus.FILLED

        closing_fill = self._is_closing_fill(order)
        realized_pnl = self._update_position(order)
        self._record_closed_trade_result(realized_pnl, closing_fill)
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
        self._append_daily_trade_csv(trade_record)

        logger.info(
            f"FILL: {order.side.upper()} {order.quantity}/{requested_quantity}x {order.symbol} "
            f"@ ${fill_price:.2f} (slippage: {trade_record['slippage_pct']:.3f}%)"
        )

        return {"status": "filled", "trade": trade_record}

    def _append_daily_trade_csv(self, trade_record: Dict) -> None:
        """Append each fill to logs/paper_trades_YYYY-MM-DD.csv for audit trail.

        Columns: timestamp, symbol, side, qty, fill_price, pnl_running
        pnl_running = portfolio_value - initial_capital (running total P&L)
        """
        try:
            log_dir = Path(os.getenv("PAPER_TRADE_LOG_DIR", "logs"))
            if not log_dir.is_absolute():
                # Resolve relative to the project/ directory (parent of core/).
                log_dir = Path(__file__).resolve().parent.parent / log_dir
            log_dir.mkdir(parents=True, exist_ok=True)
            date_str = _eastern_date_str()
            csv_path = log_dir / f"paper_trades_{date_str}.csv"
            new_file = not csv_path.exists()
            pnl_running = float(trade_record.get("portfolio_value", self.initial_capital)) - float(self.initial_capital)
            with open(csv_path, "a", encoding="utf-8", newline="") as fh:
                if new_file:
                    fh.write("timestamp,symbol,side,qty,fill_price,pnl_running\n")
                side_val = trade_record.get("side", "")
                if hasattr(side_val, "value"):
                    side_val = side_val.value
                filled_at = trade_record.get("filled_at", "")
                if isinstance(filled_at, datetime):
                    ts_dt = filled_at if filled_at.tzinfo else filled_at.replace(tzinfo=timezone.utc)
                    ts_str = ts_dt.astimezone(_EASTERN_TZ).isoformat()
                elif isinstance(filled_at, str) and filled_at:
                    try:
                        ts_dt = datetime.fromisoformat(filled_at)
                        if ts_dt.tzinfo is None:
                            ts_dt = ts_dt.replace(tzinfo=timezone.utc)
                        ts_str = ts_dt.astimezone(_EASTERN_TZ).isoformat()
                    except ValueError:
                        ts_str = filled_at
                else:
                    ts_str = datetime.now(_EASTERN_TZ).isoformat()
                fh.write(
                    f"{ts_str},"
                    f"{trade_record.get('symbol', '')},"
                    f"{side_val},"
                    f"{trade_record.get('quantity', 0)},"
                    f"{trade_record.get('fill_price', 0)},"
                    f"{round(pnl_running, 4)}\n"
                )
        except Exception as exc:
            logger.debug(f"paper_trades CSV append failed: {exc}")

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
            self._set_kill_switch(
                activated=True,
                reason="max drawdown breached",
                day=datetime.now(timezone.utc).date().isoformat(),
                activated_at=datetime.now(timezone.utc),
            )
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
        closed_trade_log = self._closed_trade_log()
        lifetime_win_rate = self._win_rate_snapshot(closed_trade_log, "lifetime")
        recent_win_rate = self._win_rate_snapshot(
            closed_trade_log[-self._headline_win_rate_window :],
            "recent",
            sample_size_override=min(len(closed_trade_log), self._headline_win_rate_window),
        )
        session_win_rate = self._win_rate_snapshot(self._session_closed_trade_log(closed_trade_log), "session")
        headline_win_rate = lifetime_win_rate
        if recent_win_rate["closed_trades"] >= self._headline_win_rate_min_trades:
            headline_win_rate = recent_win_rate
        elif session_win_rate["closed_trades"] >= self._headline_win_rate_min_trades:
            headline_win_rate = session_win_rate
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
            "closed_trades": lifetime_win_rate["closed_trades"],
            "winning_closed_trades": lifetime_win_rate["wins"],
            "losing_closed_trades": lifetime_win_rate["losses"],
            "breakeven_closed_trades": lifetime_win_rate["breakeven"],
            "lifetime_win_rate_pct": lifetime_win_rate["win_rate_pct"],
            "recent_win_rate_pct": recent_win_rate["win_rate_pct"],
            "recent_closed_trades": recent_win_rate["closed_trades"],
            "session_win_rate_pct": session_win_rate["win_rate_pct"],
            "session_closed_trades": session_win_rate["closed_trades"],
            "win_rate_pct": headline_win_rate["win_rate_pct"],
            "win_rate_basis": headline_win_rate["basis"],
            "win_rate_label": headline_win_rate["label"],
            "win_rate_sample_size": headline_win_rate["closed_trades"],
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

    def _closed_trade_log(self) -> List[Dict]:
        closed: List[Dict] = []
        for trade in self.trade_log:
            if trade.get("realized_pnl") is None or not trade.get("filled_at"):
                continue
            closed.append(trade)
        return closed

    def _session_closed_trade_log(self, closed_trade_log: Optional[List[Dict]] = None) -> List[Dict]:
        today = datetime.now(timezone.utc).date()
        rows = closed_trade_log if closed_trade_log is not None else self._closed_trade_log()
        session_rows: List[Dict] = []
        for trade in rows:
            try:
                filled_at = datetime.fromisoformat(str(trade.get("filled_at")))
            except Exception:
                continue
            if filled_at.astimezone(timezone.utc).date() == today:
                session_rows.append(trade)
        return session_rows

    @staticmethod
    def _win_rate_snapshot(rows: List[Dict], basis: str, sample_size_override: Optional[int] = None) -> Dict:
        wins = 0
        losses = 0
        breakeven = 0
        for trade in rows:
            realized = float(trade.get("realized_pnl", 0.0) or 0.0)
            if realized > 1e-9:
                wins += 1
            elif realized < -1e-9:
                losses += 1
            else:
                breakeven += 1
        closed_trades = int(sample_size_override if sample_size_override is not None else len(rows))
        labels = {
            "recent": f"last {closed_trades} closed paper trades",
            "session": f"today • {closed_trades} closed paper trades",
            "lifetime": f"lifetime • {closed_trades} closed paper trades",
        }
        # Exclude breakeven trades from the win-rate denominator so that
        # a large number of flat/zero-PnL closes (e.g. 63 breakevens) does
        # not dilute the rate toward 0%. Win rate = wins / (wins + losses).
        decisive_trades = wins + losses
        win_rate_pct = round((wins / max(decisive_trades, 1)) * 100.0, 1) if decisive_trades else 0.0
        return {
            "basis": basis,
            "label": labels.get(basis, f"{closed_trades} closed paper trades"),
            "closed_trades": closed_trades,
            "wins": wins,
            "losses": losses,
            "breakeven": breakeven,
            "win_rate_pct": win_rate_pct,
        }

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
        return daily_realized_pnl, daily_loss_pct, int(self._consecutive_losses)

    def _maybe_reset_session_kill_switch(self):
        previous_state = (
            self._kill_switch_activated,
            self._kill_switch_reason,
            self._kill_switch_day,
            self._kill_switch_activated_at.isoformat() if self._kill_switch_activated_at else None,
            self._consecutive_losses,
        )
        if not self._kill_switch_activated:
            return
        if self._kill_switch_reason not in {"daily loss limit", "consecutive losses"}:
            return
        if not self.session_guardrails_enabled:
            self._set_kill_switch(activated=False)
            self._consecutive_losses = 0
            self._persist_if_state_changed(previous_state)
            return
        now = datetime.now(timezone.utc)
        if self._kill_switch_reason == "consecutive losses":
            activated_at = self._kill_switch_activated_at or now
            if now - activated_at >= self._kill_switch_cooldown:
                self._reset_consecutive_loss_guardrail(
                    "Consecutive-loss kill switch auto-reset after 2 hour cooldown"
                )
                self._persist_if_state_changed(previous_state)
                return
            if self._is_consecutive_loss_warmup_active():
                self._set_kill_switch(activated=False)
                self._persist_if_state_changed(previous_state)
                return
        today_key = now.date().isoformat()
        if self._kill_switch_reason == "daily loss limit" and self._kill_switch_day != today_key:
            self._set_kill_switch(activated=False)
        self._persist_if_state_changed(previous_state)

    def _refresh_session_risk_controls(self):
        previous_state = (
            self._kill_switch_activated,
            self._kill_switch_reason,
            self._kill_switch_day,
            self._kill_switch_activated_at.isoformat() if self._kill_switch_activated_at else None,
            self._consecutive_losses,
        )
        if self._kill_switch_activated and self._kill_switch_reason == "max drawdown breached":
            return
        if not self.session_guardrails_enabled:
            if self._kill_switch_reason in {"daily loss limit", "consecutive losses"}:
                self._set_kill_switch(activated=False)
                self._consecutive_losses = 0
            self._persist_if_state_changed(previous_state)
            return
        _, daily_loss_pct, consecutive_losses = self._session_risk_snapshot()
        today_key = datetime.now(timezone.utc).date().isoformat()
        if daily_loss_pct >= self.max_daily_loss_pct:
            if not (self._kill_switch_activated and self._kill_switch_reason == "daily loss limit"):
                self._set_kill_switch(
                    activated=True,
                    reason="daily loss limit",
                    day=today_key,
                    activated_at=datetime.now(timezone.utc),
                )
        elif consecutive_losses >= self.max_consecutive_losses:
            if self._is_consecutive_loss_warmup_active():
                if self._kill_switch_reason == "consecutive losses":
                    self._set_kill_switch(activated=False)
            elif not (self._kill_switch_activated and self._kill_switch_reason == "consecutive losses"):
                self._set_kill_switch(
                    activated=True,
                    reason="consecutive losses",
                    day=today_key,
                    activated_at=datetime.now(timezone.utc),
                )
        elif (
            self._kill_switch_reason in {"daily loss limit", "consecutive losses"}
            and (
                (
                    self._kill_switch_reason == "daily loss limit"
                    and self._kill_switch_day == today_key
                    and daily_loss_pct < self.max_daily_loss_pct
                )
                or (
                    self._kill_switch_reason == "consecutive losses"
                    and consecutive_losses < self.max_consecutive_losses
                )
            )
        ):
            self._set_kill_switch(activated=False)
        self._persist_if_state_changed(previous_state)

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

    def _execution_mode(self, order: Order) -> str:
        metadata = dict(order.metadata or {})
        mode = str(metadata.get("execution_mode") or order.order_type or "taker").strip().lower()
        if mode in {"market", "taker"}:
            return "taker"
        if mode in {"limit", "maker"}:
            return "maker"
        return "taker"

    def _market_microstructure(self, order: Order, market_price: float) -> Dict[str, float]:
        metadata = dict(order.metadata or {})
        best_bid = self._safe_float(metadata.get("best_bid"), 0.0)
        best_ask = self._safe_float(metadata.get("best_ask"), 0.0)
        spread_pct = max(0.0, self._safe_float(metadata.get("spread_pct"), 0.0))
        if best_bid <= 0 or best_ask <= 0 or best_ask < best_bid:
            half_spread = max(market_price * (spread_pct / 100.0) / 2.0, market_price * 0.00005)
            best_bid = max(1e-9, market_price - half_spread)
            best_ask = max(best_bid, market_price + half_spread)
        mid_price = (best_bid + best_ask) / 2.0 if best_bid > 0 and best_ask > 0 else max(market_price, 1e-9)
        quoted_spread = max(best_ask - best_bid, mid_price * max(spread_pct / 100.0, 1e-6))

        best_bid_qty = max(0.0, self._safe_float(metadata.get("best_bid_qty"), 0.0))
        best_ask_qty = max(0.0, self._safe_float(metadata.get("best_ask_qty"), 0.0))
        bid_depth_qty = max(0.0, self._safe_float(metadata.get("bid_depth_qty"), best_bid_qty))
        ask_depth_qty = max(0.0, self._safe_float(metadata.get("ask_depth_qty"), best_ask_qty))
        tick_age_seconds = max(0.0, self._safe_float(metadata.get("tick_age_seconds"), 0.0))
        depth_age_seconds = max(0.0, self._safe_float(metadata.get("depth_age_seconds"), tick_age_seconds))
        signal_age_seconds = max(0.0, self._safe_float(metadata.get("signal_age_seconds"), 0.0))
        book_pressure = self._safe_float(metadata.get("book_pressure"), 0.0)
        book_imbalance = self._safe_float(metadata.get("book_imbalance"), book_pressure)
        top_book_imbalance = self._safe_float(metadata.get("top_book_imbalance"), book_imbalance)
        order_flow_imbalance = self._safe_float(metadata.get("order_flow_imbalance"), 0.0)
        microprice_bias = self._safe_float(metadata.get("microprice_bias"), 0.0)
        spread_velocity = self._safe_float(metadata.get("spread_velocity"), 0.0)
        relative_volume = max(0.0, self._safe_float(metadata.get("relative_volume"), 1.0))
        flicker_filter_retention = min(1.0, max(0.0, self._safe_float(metadata.get("flicker_filter_retention"), 1.0)))

        direction = 1.0 if order.side == OrderSide.BUY else -1.0
        requested_qty = abs(float(order.quantity))
        side_top_qty = best_ask_qty if order.side == OrderSide.BUY else best_bid_qty
        side_depth_qty = ask_depth_qty if order.side == OrderSide.BUY else bid_depth_qty
        accessible_depth_qty = max(side_top_qty, side_depth_qty * 0.45, 1e-9)
        liquidity_ratio = requested_qty / accessible_depth_qty if accessible_depth_qty > 0 else 0.0
        top_level_ratio = requested_qty / max(side_top_qty, 1e-9) if side_top_qty > 0 else liquidity_ratio

        age_penalty = 0.0
        if self._max_tick_age_seconds > 0:
            age_penalty += min(1.0, tick_age_seconds / max(self._max_tick_age_seconds, 1e-9)) * 0.45
            age_penalty += min(1.0, depth_age_seconds / max(self._max_tick_age_seconds * 2.0, 1e-9)) * 0.20
        if self._max_signal_age_seconds > 0:
            age_penalty += min(1.0, signal_age_seconds / max(self._max_signal_age_seconds, 1e-9)) * 0.35

        directional_pressure = max(0.0, direction * book_pressure)
        directional_top = max(0.0, direction * top_book_imbalance)
        directional_flow = max(0.0, direction * order_flow_imbalance)
        directional_micro = max(0.0, direction * microprice_bias)
        adverse_selection = min(
            1.75,
            (directional_pressure * 0.45)
            + (directional_top * 0.20)
            + (directional_flow * 0.15)
            + (directional_micro * 0.10)
            + (max(0.0, spread_velocity) * 18.0),
        )
        queue_pressure = min(
            2.5,
            max(0.0, top_level_ratio - 0.65) * 0.55
            + max(0.0, liquidity_ratio - 0.85) * 0.45
            + (1.0 - flicker_filter_retention) * 0.35
            + max(0.0, 1.0 - min(relative_volume, 1.0)) * 0.15
            + adverse_selection * 0.10,
        )
        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid_price,
            "quoted_spread": quoted_spread,
            "quoted_spread_bps": ((quoted_spread / max(mid_price, 1e-9)) * 10_000.0) if mid_price > 0 else 0.0,
            "spread_pct": spread_pct,
            "best_bid_qty": best_bid_qty,
            "best_ask_qty": best_ask_qty,
            "bid_depth_qty": bid_depth_qty,
            "ask_depth_qty": ask_depth_qty,
            "tick_age_seconds": tick_age_seconds,
            "depth_age_seconds": depth_age_seconds,
            "signal_age_seconds": signal_age_seconds,
            "book_pressure": book_pressure,
            "book_imbalance": book_imbalance,
            "top_book_imbalance": top_book_imbalance,
            "order_flow_imbalance": order_flow_imbalance,
            "microprice_bias": microprice_bias,
            "spread_velocity": spread_velocity,
            "relative_volume": relative_volume,
            "flicker_filter_retention": flicker_filter_retention,
            "liquidity_ratio": liquidity_ratio,
            "top_level_ratio": top_level_ratio,
            "age_penalty": age_penalty,
            "adverse_selection": adverse_selection,
            "queue_pressure": queue_pressure,
            "execution_mode": self._execution_mode(order),
        }

    def _simulate_fill_outcome(self, order: Order, market_price: float) -> Dict:
        market_state = self._market_microstructure(order, market_price)
        spread_pct = max(0.0, float(market_state.get("spread_pct", 0.0) or 0.0))
        tick_age_seconds = max(0.0, float(market_state.get("tick_age_seconds", 0.0) or 0.0))
        signal_age_seconds = max(0.0, float(market_state.get("signal_age_seconds", 0.0) or 0.0))
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
            rejection_probability += min(0.20, size_pressure * 0.12)
            if self._max_spread_pct > 0:
                rejection_probability += min(0.18, (spread_pct / max(self._max_spread_pct, 1e-9)) * 0.06)
            if self._max_tick_age_seconds > 0:
                rejection_probability += min(0.16, (tick_age_seconds / max(self._max_tick_age_seconds, 1e-9)) * 0.05)
            rejection_probability += min(0.20, float(market_state.get("liquidity_ratio", 0.0) or 0.0) * 0.12)
            rejection_probability += min(0.18, float(market_state.get("queue_pressure", 0.0) or 0.0) * 0.14)
            rejection_probability += min(0.16, float(market_state.get("adverse_selection", 0.0) or 0.0) * 0.12)
            rejection_probability += min(0.12, float(market_state.get("age_penalty", 0.0) or 0.0) * 0.15)
            rejection_probability += min(
                0.12,
                max(0.0, 1.0 - float(market_state.get("flicker_filter_retention", 1.0) or 1.0)) * 0.20,
            )
            if market_state.get("execution_mode") == "maker":
                rejection_probability += min(0.10, float(market_state.get("queue_pressure", 0.0) or 0.0) * 0.08)
            if self._order_noise(order, "reject") < rejection_probability:
                return {"rejected": True, "reason": "liquidity miss / queue loss"}

        latency_ms = self._simulated_latency_ms + (self._order_noise(order, "latency") * self._simulated_latency_jitter_ms)
        fill_price = self._simulate_fill_price(order, market_price, latency_ms=latency_ms, market_state=market_state)
        fill_ratio = 1.0

        eligible_for_partial = self._partial_fill_enabled and (
            not self._partial_fill_entry_only or order.side == OrderSide.BUY
        )
        if eligible_for_partial:
            partial_probability = min(
                0.95,
                0.05
                + (size_pressure * 0.18)
                + min(0.30, float(market_state.get("liquidity_ratio", 0.0) or 0.0) * 0.45)
                + min(0.22, float(market_state.get("queue_pressure", 0.0) or 0.0) * 0.18)
                + min(0.18, float(market_state.get("adverse_selection", 0.0) or 0.0) * 0.14)
                + min(0.08, float(market_state.get("age_penalty", 0.0) or 0.0) * 0.08),
            )
            if market_state.get("execution_mode") == "maker":
                partial_probability = min(0.98, partial_probability + 0.12)
            if self._order_noise(order, "partial_trigger") < partial_probability:
                ratio_draw = self._order_noise(order, "partial_ratio")
                pressure_penalty = min(
                    0.90,
                    size_pressure * 0.10
                    + min(0.35, float(market_state.get("liquidity_ratio", 0.0) or 0.0) * 0.40)
                    + min(0.30, float(market_state.get("queue_pressure", 0.0) or 0.0) * 0.25)
                    + min(0.20, float(market_state.get("adverse_selection", 0.0) or 0.0) * 0.15)
                    + min(0.10, float(market_state.get("age_penalty", 0.0) or 0.0) * 0.10),
                )
                if market_state.get("execution_mode") == "maker":
                    pressure_penalty += 0.05
                fill_ratio = max(self._partial_fill_min_ratio, 1.0 - pressure_penalty - (ratio_draw * 0.15))

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
            "execution_mode": str(market_state.get("execution_mode") or "taker"),
            "reference_price": float(market_state.get("mid_price", market_price) or market_price),
            "quoted_spread_bps": float(market_state.get("quoted_spread_bps", 0.0) or 0.0),
            "liquidity_ratio": float(market_state.get("liquidity_ratio", 0.0) or 0.0),
            "queue_pressure": float(market_state.get("queue_pressure", 0.0) or 0.0),
            "adverse_selection": float(market_state.get("adverse_selection", 0.0) or 0.0),
        }

    def _simulate_fill_price(
        self,
        order: Order,
        market_price: float,
        *,
        latency_ms: float = 0.0,
        market_state: Optional[Dict[str, float]] = None,
    ) -> float:
        """Simulate realistic fill with slippage."""
        from core.backtesting import TransactionCostModel
        model = TransactionCostModel()
        market_state = dict(market_state or self._market_microstructure(order, market_price))
        best_bid = self._safe_float(market_state.get("best_bid"), 0.0)
        best_ask = self._safe_float(market_state.get("best_ask"), 0.0)
        mid_price = max(self._safe_float(market_state.get("mid_price"), market_price), 1e-9)
        quoted_spread = max(self._safe_float(market_state.get("quoted_spread"), 0.0), mid_price * 1e-6)
        spread_pct = max(0.0, self._safe_float(market_state.get("spread_pct"), 0.0))
        trade_value = abs(order.quantity) * max(market_price, 1e-9)
        size_impact = min(1.8, (abs(order.quantity) / 2500.0) + (trade_value / 150_000.0))
        liquidity_ratio = max(0.0, self._safe_float(market_state.get("liquidity_ratio"), 0.0))
        queue_pressure = max(0.0, self._safe_float(market_state.get("queue_pressure"), 0.0))
        adverse_selection = max(0.0, self._safe_float(market_state.get("adverse_selection"), 0.0))
        spread_velocity = self._safe_float(market_state.get("spread_velocity"), 0.0)
        execution_mode = str(market_state.get("execution_mode") or "taker")

        spread = max(quoted_spread, market_price * max(model.base_slippage_pct, spread_pct / 100.0 * 0.20))
        latency_factor = min(4.0, latency_ms / 150.0) if latency_ms > 0 else 0.0
        if execution_mode == "maker":
            inside_offset = spread * (0.12 + min(0.22, queue_pressure * 0.10))
            passive_price = (best_bid + inside_offset) if order.side == OrderSide.BUY else max(1e-9, best_ask - inside_offset)
            maker_drift = spread * (
                0.04
                + min(0.10, adverse_selection * 0.04)
                + min(0.06, max(0.0, spread_velocity) * 4.0)
            )
            fill_price = passive_price + maker_drift if order.side == OrderSide.BUY else max(1e-9, passive_price - maker_drift)
        elif best_bid > 0 and best_ask > 0 and best_ask >= best_bid:
            walk_component = spread * (
                0.55
                + min(0.45, liquidity_ratio * 0.30)
                + min(0.25, queue_pressure * 0.20)
                + min(0.22, adverse_selection * 0.14)
            )
            impact = max(spread * 0.35, market_price * model.base_slippage_pct) * (1.0 + size_impact) + walk_component
            fill_price = (best_ask + impact) if order.side == OrderSide.BUY else max(1e-9, best_bid - impact)
        else:
            impact = max(
                market_price * model.base_slippage_pct,
                market_price * (spread_pct / 100.0) * 0.20,
            ) * (1.0 + size_impact + liquidity_ratio * 0.20 + adverse_selection * 0.10)
            fill_price = (market_price + impact) if order.side == OrderSide.BUY else max(1e-9, market_price - impact)

        if latency_ms > 0:
            latency_impact = market_price * model.base_slippage_pct * (
                latency_factor
                + min(0.50, max(0.0, spread_velocity) * 6.0)
                + min(0.35, adverse_selection * 0.18)
            )
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

    def _is_closing_fill(self, order: Order) -> bool:
        position_key = self._order_position_key(order)
        pos = self.positions.get(position_key)
        if pos is None or abs(getattr(pos, "quantity", 0.0) or 0.0) < 1e-9:
            return False
        qty = order.quantity if order.side == OrderSide.BUY else -order.quantity
        return (pos.quantity > 0) != (qty > 0)

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
            "consecutive_losses": self._consecutive_losses,
            "market": self.market,
            "session_guardrails_enabled": self.session_guardrails_enabled,
            "kill_switch_active": self._kill_switch_activated,
            "kill_switch_activated": self._kill_switch_activated,
            "kill_switch_reason": self._kill_switch_reason,
            "kill_switch_day": self._kill_switch_day,
            "kill_switch_activated_at": self._kill_switch_activated_at.isoformat() if self._kill_switch_activated_at else None,
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
            self._consecutive_losses = int(
                payload.get(
                    "consecutive_losses",
                    self._rebuild_consecutive_losses_from_trade_log(payload.get("trade_log") or []),
                )
            )
            self.market = payload.get("market", self.market)
            self.session_guardrails_enabled = bool(
                payload.get("session_guardrails_enabled", self.session_guardrails_enabled)
            )
            self._kill_switch_activated = bool(
                payload.get("kill_switch_activated", payload.get("kill_switch_active", self._kill_switch_activated))
            )
            self._kill_switch_reason = str(payload.get("kill_switch_reason", self._kill_switch_reason) or "")
            self._kill_switch_day = str(payload.get("kill_switch_day", self._kill_switch_day) or "")
            activated_at_raw = payload.get("kill_switch_activated_at")
            self._kill_switch_activated_at = (
                datetime.fromisoformat(activated_at_raw) if activated_at_raw else None
            )
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

    @staticmethod
    def _rebuild_consecutive_losses_from_trade_log(trade_log: List[Dict]) -> int:
        consecutive_losses = 0
        for trade in reversed(list(trade_log or [])):
            if not trade.get("filled_at"):
                continue
            realized_pnl = float(trade.get("realized_pnl", 0.0) or 0.0)
            if realized_pnl < -1e-9:
                consecutive_losses += 1
                continue
            if abs(realized_pnl) >= 1e-9:
                break
        return consecutive_losses
