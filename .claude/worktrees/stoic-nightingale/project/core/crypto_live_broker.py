"""
KuCoin Live Crypto Broker — Production Execution Adapter
=========================================================
Drop-in replacement for CryptoPaperBroker when CRYPTO_EXECUTION_MODE=live.

Same interface: execute_buy, execute_sell, update_prices, check_stops,
get_summary, can_trade.  The live loop swaps brokers without code changes.

Safety layers:
  - LIVE_SIZE_MULTIPLIER (default 0.25 → 25% of recommended size)
  - MAX_SINGLE_ORDER_USDT (default $50)
  - MAX_OPEN_POSITIONS (default 3)
  - Daily loss circuit breaker (default $15)
  - Sandbox mode by default (KUCOIN_SANDBOX=true)
  - LIMIT orders only — never falls back to MARKET
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import math
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from core.paper_trading_crypto import (
    CryptoPortfolio,
    CryptoPosition,
    CryptoTrade,
)

logger = logging.getLogger(__name__)


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    raw = _env(name)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


# ─────────────────────────────────────────────────────────────
# KuCoin REST client (auth + order management)
# ─────────────────────────────────────────────────────────────

class _KuCoinClient:
    """Low-level authenticated KuCoin REST v1/v2 client."""

    PRODUCTION_URL = "https://api.kucoin.com"
    SANDBOX_URL = "https://openapi-sandbox.kucoin.com"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: str,
        sandbox: bool = True,
        timeout: int = 10,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.base_url = self.SANDBOX_URL if sandbox else self.PRODUCTION_URL
        self.timeout = timeout
        self._session = requests.Session()

    # ── auth ─────────────────────────────────────────────────

    def _sign_request(
        self, method: str, endpoint: str, body: str = ""
    ) -> Dict[str, str]:
        """
        KuCoin v2 signature:
          str_to_sign = timestamp + method + endpoint + body
          signature   = base64(hmac-sha256(secret, str_to_sign))
          passphrase  = base64(hmac-sha256(secret, raw_passphrase))
        """
        timestamp = str(int(time.time() * 1000))
        str_to_sign = timestamp + method.upper() + endpoint + body

        sig = hmac.new(
            self.api_secret.encode("utf-8"),
            str_to_sign.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        signature = base64.b64encode(sig).decode("utf-8")

        pp_sig = hmac.new(
            self.api_secret.encode("utf-8"),
            self.passphrase.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        signed_passphrase = base64.b64encode(pp_sig).decode("utf-8")

        return {
            "KC-API-KEY": self.api_key,
            "KC-API-SIGN": signature,
            "KC-API-TIMESTAMP": timestamp,
            "KC-API-PASSPHRASE": signed_passphrase,
            "KC-API-KEY-VERSION": "2",
            "Content-Type": "application/json",
        }

    # ── HTTP helpers ─────────────────────────────────────────

    def _request(
        self, method: str, endpoint: str, body: Optional[dict] = None
    ) -> dict:
        body_str = json.dumps(body, separators=(",", ":")) if body else ""
        headers = self._sign_request(method, endpoint, body_str)

        url = self.base_url + endpoint
        resp = self._session.request(
            method,
            url,
            headers=headers,
            data=body_str if body_str else None,
            timeout=self.timeout,
        )

        try:
            data = resp.json()
        except ValueError:
            raise RuntimeError(
                f"KuCoin non-JSON response ({resp.status_code}): {resp.text[:300]}"
            )

        code = str(data.get("code", ""))
        if code != "200000":
            raise RuntimeError(
                f"KuCoin API error {code}: {data.get('msg', resp.text[:300])}"
            )
        return data.get("data", {})

    def get(self, endpoint: str) -> dict:
        return self._request("GET", endpoint)

    def post(self, endpoint: str, body: dict) -> dict:
        return self._request("POST", endpoint, body)

    def delete(self, endpoint: str) -> dict:
        return self._request("DELETE", endpoint)

    # ── order operations ─────────────────────────────────────

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        price: float,
        size: float,
        client_oid: str = "",
    ) -> dict:
        """POST /api/v1/orders — limit order."""
        payload: Dict[str, Any] = {
            "clientOid": client_oid or f"mi-{symbol}-{int(time.time()*1000)}",
            "side": side.lower(),
            "symbol": symbol,
            "type": "limit",
            "price": f"{price:.8f}".rstrip("0").rstrip("."),
            "size": f"{size:.8f}".rstrip("0").rstrip("."),
            "timeInForce": "GTC",
        }
        return self.post("/api/v1/orders", payload)

    def cancel_order(self, order_id: str) -> dict:
        """DELETE /api/v1/orders/{orderId}"""
        return self.delete(f"/api/v1/orders/{order_id}")

    def get_order(self, order_id: str) -> dict:
        """GET /api/v1/orders/{orderId}"""
        return self.get(f"/api/v1/orders/{order_id}")

    def get_accounts(self, currency: str = "", account_type: str = "trade") -> list:
        """GET /api/v1/accounts"""
        params = f"?type={account_type}"
        if currency:
            params += f"&currency={currency}"
        data = self.get(f"/api/v1/accounts{params}")
        return data if isinstance(data, list) else []

    def get_ticker(self, symbol: str) -> dict:
        """GET /api/v1/market/orderbook/level1 (public, but we sign anyway)."""
        return self.get(f"/api/v1/market/orderbook/level1?symbol={symbol}")


# ─────────────────────────────────────────────────────────────
# KuCoin Live Broker — same interface as CryptoPaperBroker
# ─────────────────────────────────────────────────────────────

class KuCoinLiveBroker:
    """
    Production crypto broker for KuCoin.

    Interface matches CryptoPaperBroker: execute_buy, execute_sell,
    update_prices, check_stops, get_summary, can_trade, reset.
    """

    def __init__(
        self,
        initial_balance: float = 0.0,
        base_currency: str = "USDT",
    ):
        sandbox = _env_flag("KUCOIN_SANDBOX", True)
        self._client = _KuCoinClient(
            api_key=_env("KUCOIN_API_KEY"),
            api_secret=_env("KUCOIN_API_SECRET"),
            passphrase=_env("KUCOIN_API_PASSPHRASE"),
            sandbox=sandbox,
            timeout=_env_int("KUCOIN_TIMEOUT_SECONDS", 10),
        )

        self.base_currency = base_currency
        self._sandbox = sandbox
        self._size_multiplier = _env_float("LIVE_SIZE_MULTIPLIER", 0.25)
        self._max_order_usdt = _env_float("MAX_SINGLE_ORDER_USDT", 50.0)
        self._max_positions = _env_int("CRYPTO_MAX_LIVE_POSITIONS", 3)
        self._max_daily_loss = _env_float("CRYPTO_MAX_DAILY_LOSS_USDT", 15.0)
        self._limit_wait_seconds = _env_float("KUCOIN_LIMIT_WAIT_SECONDS", 8.0)

        # Portfolio bookkeeping (mirrors CryptoPaperBroker)
        self.portfolio = CryptoPortfolio(
            usdt_balance=initial_balance,
            starting_balance=initial_balance,
        )

        # Circuit breaker state
        self._daily_realized_pnl: float = 0.0
        self._daily_reset_date: str = ""
        self.halted: bool = False

        self._lock = threading.Lock()
        self._order_log: List[Dict] = []

        mode = "SANDBOX" if sandbox else "PRODUCTION"
        logger.info(
            f"KuCoinLiveBroker initialized [{mode}] "
            f"size_mult={self._size_multiplier} "
            f"max_order=${self._max_order_usdt} "
            f"max_pos={self._max_positions} "
            f"max_daily_loss=${self._max_daily_loss}"
        )

    # ── circuit breaker ──────────────────────────────────────

    def _check_daily_reset(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._daily_reset_date != today:
            self._daily_reset_date = today
            self._daily_realized_pnl = 0.0
            self.halted = False
            logger.info("KuCoinLiveBroker: daily circuit breaker reset")

    def _record_pnl(self, pnl: float) -> None:
        self._daily_realized_pnl += pnl
        if self._daily_realized_pnl <= -self._max_daily_loss:
            self.halted = True
            logger.warning(
                f"CIRCUIT BREAKER TRIPPED: daily loss ${self._daily_realized_pnl:.2f} "
                f"exceeds limit ${self._max_daily_loss:.2f} — halting new orders"
            )

    # ── position guards ──────────────────────────────────────

    def _reject(self, reason: str, symbol: str, side: str, qty: float) -> None:
        logger.warning(
            f"ORDER REJECTED: {reason} | "
            f"symbol={symbol} side={side} qty={qty:.6f}"
        )

    def _guard_checks(
        self, symbol: str, side: str, quantity: float, price: float
    ) -> Optional[str]:
        """Return rejection reason or None if all guards pass."""
        self._check_daily_reset()

        # Circuit breaker: allow exits when halted
        is_exit = side == "sell" and self._find_position(symbol) is not None
        if self.halted and not is_exit:
            return (
                f"circuit breaker active (daily loss ${self._daily_realized_pnl:.2f})"
            )

        if side == "buy":
            order_usdt = quantity * price
            if order_usdt > self._max_order_usdt:
                return (
                    f"order value ${order_usdt:.2f} exceeds "
                    f"MAX_SINGLE_ORDER_USDT ${self._max_order_usdt:.2f}"
                )
            if len(self.portfolio.positions) >= self._max_positions:
                return (
                    f"open positions ({len(self.portfolio.positions)}) "
                    f"at limit ({self._max_positions})"
                )
        return None

    # ── size scaling ─────────────────────────────────────────

    def _scale_quantity(self, recommended_qty: float, min_qty: float = 0.0) -> float:
        scaled = recommended_qty * self._size_multiplier
        return max(min_qty, math.floor(scaled * 1e8) / 1e8)  # floor to 8dp

    # ── order execution (limit with retry) ───────────────────

    def _execute_limit_with_retry(
        self,
        symbol: str,
        side: str,
        quantity: float,
        bid: float,
        ask: float,
    ) -> Optional[Dict]:
        """
        1. Place limit 25% inside the spread.
        2. Wait up to _limit_wait_seconds for fill.
        3. If unfilled → cancel → retry at mid.
        4. If still unfilled → cancel → return None (skip).
        Never falls back to market.
        """
        spread = ask - bid
        if side == "buy":
            price_1 = ask - spread * 0.25
            price_2 = (bid + ask) / 2
        else:
            price_1 = bid + spread * 0.25
            price_2 = (bid + ask) / 2

        for attempt, limit_price in enumerate([price_1, price_2], start=1):
            limit_price = round(limit_price, 8)
            oid = f"mi-{symbol}-{side[0]}-{int(time.time()*1000)}"

            logger.info(
                f"KuCoin LIMIT attempt {attempt}/2: "
                f"{side.upper()} {quantity:.8f} {symbol} @ ${limit_price:.8f} "
                f"(bid={bid:.8f} ask={ask:.8f})"
            )

            try:
                result = self._client.place_limit_order(
                    symbol=symbol,
                    side=side,
                    price=limit_price,
                    size=quantity,
                    client_oid=oid,
                )
            except Exception as exc:
                logger.error(f"KuCoin order placement failed: {exc}")
                return None

            order_id = result.get("orderId", "")
            if not order_id:
                logger.error(f"KuCoin returned no orderId: {result}")
                return None

            # Poll for fill
            filled = self._wait_for_fill(order_id)
            if filled:
                fill_price = float(filled.get("dealFunds", 0)) / max(
                    float(filled.get("dealSize", 0)), 1e-12
                )
                fill_size = float(filled.get("dealSize", 0))
                logger.info(
                    f"KuCoin FILLED: {side.upper()} {fill_size:.8f} {symbol} "
                    f"@ avg ${fill_price:.8f} (orderId={order_id})"
                )
                return {
                    "order_id": order_id,
                    "fill_price": fill_price if fill_price > 0 else limit_price,
                    "fill_size": fill_size if fill_size > 0 else quantity,
                    "status": "filled",
                }

            # Not filled → cancel
            try:
                self._client.cancel_order(order_id)
                logger.info(
                    f"KuCoin cancelled unfilled order {order_id} (attempt {attempt})"
                )
            except Exception as exc:
                logger.warning(f"KuCoin cancel failed for {order_id}: {exc}")

        logger.warning(
            f"KuCoin SKIP: {side.upper()} {quantity:.8f} {symbol} — "
            f"not filled after 2 limit attempts"
        )
        return None

    def _wait_for_fill(self, order_id: str) -> Optional[Dict]:
        """Poll order status until filled or timeout."""
        deadline = time.monotonic() + self._limit_wait_seconds
        while time.monotonic() < deadline:
            try:
                order = self._client.get_order(order_id)
            except Exception:
                time.sleep(0.5)
                continue

            if order.get("isActive") is False:
                deal_size = float(order.get("dealSize", 0))
                if deal_size > 0:
                    return order
                return None  # cancelled externally or zero fill
            time.sleep(0.5)
        return None

    # ── bid/ask helper ───────────────────────────────────────

    def _get_bbo(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get best bid/ask from KuCoin."""
        try:
            ticker = self._client.get_ticker(symbol)
            bid = float(ticker.get("bestBid", 0))
            ask = float(ticker.get("bestAsk", 0))
            if bid > 0 and ask > 0:
                return {"bid": bid, "ask": ask}
        except Exception as exc:
            logger.error(f"KuCoin ticker fetch failed for {symbol}: {exc}")
        return None

    # ── CryptoPaperBroker interface ──────────────────────────

    def _find_position(self, symbol: str) -> Optional[CryptoPosition]:
        for pos in self.portfolio.positions:
            if pos.symbol == symbol:
                return pos
        return None

    def can_trade(
        self, symbol: str, direction: str, quantity: float, price: float
    ) -> bool:
        reason = self._guard_checks(symbol, direction, quantity, price)
        if reason:
            return False
        if direction == "buy":
            return self.portfolio.usdt_balance >= quantity * price
        pos = self._find_position(symbol)
        return pos is not None and pos.quantity >= quantity

    def execute_buy(
        self,
        symbol: str,
        quantity: float,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Optional[CryptoTrade]:
        with self._lock:
            scaled_qty = self._scale_quantity(quantity)
            if scaled_qty <= 0:
                self._reject("scaled quantity is zero", symbol, "buy", quantity)
                return None

            reason = self._guard_checks(symbol, "buy", scaled_qty, price)
            if reason:
                self._reject(reason, symbol, "buy", scaled_qty)
                return None

            bbo = self._get_bbo(symbol)
            if bbo is None:
                self._reject("cannot fetch bid/ask", symbol, "buy", scaled_qty)
                return None

            fill = self._execute_limit_with_retry(
                symbol, "buy", scaled_qty, bbo["bid"], bbo["ask"]
            )
            if fill is None:
                return None

            fill_price = fill["fill_price"]
            fill_size = fill["fill_size"]
            cost = fill_price * fill_size
            self.portfolio.usdt_balance -= cost

            position = CryptoPosition(
                symbol=symbol,
                direction="buy",
                quantity=fill_size,
                entry_price=fill_price,
                entry_time=datetime.now(timezone.utc),
                stop_loss=stop_loss or (fill_price * 0.95),
                take_profit=take_profit or (fill_price * 1.10),
                current_price=fill_price,
            )
            self.portfolio.positions.append(position)

            self._log_order(
                symbol, "buy", fill_size, fill_price, "filled", fill["order_id"]
            )

            return CryptoTrade(
                symbol=symbol,
                direction="buy",
                quantity=fill_size,
                price=fill_price,
                timestamp=position.entry_time,
            )

    def execute_sell(
        self,
        symbol: str,
        quantity: float,
        price: float,
        reason: str = "signal",
    ) -> Optional[CryptoTrade]:
        with self._lock:
            position = self._find_position(symbol)
            if not position or position.direction != "buy":
                self._reject("no long position to sell", symbol, "sell", quantity)
                return None

            sell_qty = min(quantity, position.quantity)
            sell_qty = self._scale_quantity(sell_qty) if reason == "signal" else sell_qty

            if sell_qty <= 0:
                self._reject("sell quantity is zero after scaling", symbol, "sell", quantity)
                return None

            guard_reason = self._guard_checks(symbol, "sell", sell_qty, price)
            if guard_reason:
                self._reject(guard_reason, symbol, "sell", sell_qty)
                return None

            bbo = self._get_bbo(symbol)
            if bbo is None:
                self._reject("cannot fetch bid/ask", symbol, "sell", sell_qty)
                return None

            fill = self._execute_limit_with_retry(
                symbol, "sell", sell_qty, bbo["bid"], bbo["ask"]
            )
            if fill is None:
                return None

            fill_price = fill["fill_price"]
            fill_size = fill["fill_size"]
            proceeds = fill_price * fill_size
            pnl = proceeds - (position.entry_price * fill_size)

            self.portfolio.usdt_balance += proceeds
            self._record_pnl(pnl)

            holding_hours = (
                datetime.now(timezone.utc) - position.entry_time
            ).total_seconds() / 3600

            trade = CryptoTrade(
                symbol=symbol,
                direction="sell",
                quantity=fill_size,
                price=fill_price,
                timestamp=datetime.now(timezone.utc),
                pnl=pnl,
                holding_hours=holding_hours,
                exit_reason=reason,
            )

            position.quantity -= fill_size
            if position.quantity <= 1e-12:
                self.portfolio.positions.remove(position)

            self.portfolio.closed_trades.append(trade)
            self._log_order(
                symbol, "sell", fill_size, fill_price, "filled", fill["order_id"]
            )

            return trade

    def update_prices(self, prices: Dict[str, float]) -> None:
        with self._lock:
            for pos in self.portfolio.positions:
                if pos.symbol in prices:
                    pos.current_price = prices[pos.symbol]
                    pos.unrealized_pnl = (
                        pos.current_price - pos.entry_price
                    ) * pos.quantity

    def check_stops(self, prices: Dict[str, float]) -> List[Dict]:
        self.update_prices(prices)
        triggers: List[Dict] = []

        for pos in list(self.portfolio.positions):
            if pos.direction != "buy":
                continue
            current = pos.current_price
            if current <= pos.stop_loss:
                self.execute_sell(pos.symbol, pos.quantity, current, "stop_loss")
                triggers.append({"symbol": pos.symbol, "reason": "sl", "price": current})
            elif current >= pos.take_profit:
                self.execute_sell(pos.symbol, pos.quantity, current, "take_profit")
                triggers.append({"symbol": pos.symbol, "reason": "tp", "price": current})

        return triggers

    def get_summary(self) -> Dict:
        import numpy as np

        with self._lock:
            trades = self.portfolio.closed_trades
            winners = [t for t in trades if t.pnl > 0]
            losers = [t for t in trades if t.pnl <= 0]

            win_rate = len(winners) / len(trades) if trades else 0
            avg_win = float(np.mean([t.pnl for t in winners])) if winners else 0
            avg_loss = float(np.mean([t.pnl for t in losers])) if losers else 0

            return {
                "balance": round(self.portfolio.usdt_balance, 2),
                "total_value": round(self.portfolio.total_value, 2),
                "total_pnl": round(self.portfolio.total_pnl, 2),
                "open_positions": len(self.portfolio.positions),
                "total_trades": len(trades),
                "win_rate": round(win_rate * 100, 2),
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "profit_factor": (
                    round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else 0
                ),
                "daily_pnl": round(self._daily_realized_pnl, 2),
                "halted": self.halted,
                "sandbox": self._sandbox,
            }

    def get_balances(self) -> Dict:
        """
        Return USDT balance and all open positions with current value.
        Fetches live account data from KuCoin.
        """
        with self._lock:
            result: Dict[str, Any] = {"usdt": 0.0, "positions": []}
            try:
                accounts = self._client.get_accounts("USDT", "trade")
                for acc in accounts:
                    if acc.get("currency") == "USDT":
                        result["usdt"] = float(acc.get("available", 0))
                        break
            except Exception as exc:
                logger.error(f"Failed to fetch USDT balance: {exc}")
                result["usdt"] = self.portfolio.usdt_balance

            for pos in self.portfolio.positions:
                result["positions"].append({
                    "symbol": pos.symbol,
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "value_usdt": pos.current_price * pos.quantity,
                })
            return result

    def reset(self) -> None:
        with self._lock:
            self.portfolio = CryptoPortfolio(
                usdt_balance=self.portfolio.starting_balance,
                starting_balance=self.portfolio.starting_balance,
            )
            self._daily_realized_pnl = 0.0
            self.halted = False
            logger.info("KuCoinLiveBroker: reset")

    # ── logging ──────────────────────────────────────────────

    def _log_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        result: str,
        order_id: str = "",
    ) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "result": result,
            "order_id": order_id,
        }
        self._order_log.append(entry)
        logger.info(
            f"LIVE ORDER: {side.upper()} {qty:.8f} {symbol} "
            f"@ ${price:.8f} → {result} (id={order_id})"
        )


# ─────────────────────────────────────────────────────────────
# Sandbox Validator
# ─────────────────────────────────────────────────────────────

class SandboxValidator:
    """
    Pre-flight check: places a tiny test order on KuCoin sandbox
    and immediately cancels it.  Returns (success, message).
    """

    TEST_SYMBOL = "BTC-USDT"
    TEST_SIZE = 0.00001
    TEST_PRICE = 10.0  # absurdly low limit — will never fill

    def validate(self) -> tuple:
        """
        Returns (True, "message") on success, (False, "message") on failure.
        """
        client = _KuCoinClient(
            api_key=_env("KUCOIN_API_KEY"),
            api_secret=_env("KUCOIN_API_SECRET"),
            passphrase=_env("KUCOIN_API_PASSPHRASE"),
            sandbox=True,
            timeout=15,
        )

        # 1. Check connectivity + auth
        try:
            accounts = client.get_accounts("USDT", "trade")
            logger.info(f"SandboxValidator: auth OK, {len(accounts)} trade accounts")
        except Exception as exc:
            return (False, f"Auth failed: {exc}")

        # 2. Place test limit order
        try:
            oid = f"sandbox-test-{int(time.time()*1000)}"
            result = client.place_limit_order(
                symbol=self.TEST_SYMBOL,
                side="buy",
                price=self.TEST_PRICE,
                size=self.TEST_SIZE,
                client_oid=oid,
            )
            order_id = result.get("orderId", "")
            if not order_id:
                return (False, f"Order placed but no orderId returned: {result}")
            logger.info(f"SandboxValidator: test order placed: {order_id}")
        except Exception as exc:
            return (False, f"Test order failed: {exc}")

        # 3. Cancel it
        try:
            client.cancel_order(order_id)
            logger.info(f"SandboxValidator: test order cancelled: {order_id}")
        except Exception as exc:
            return (False, f"Test order cancel failed: {exc}")

        return (True, f"Sandbox validation passed (order {order_id} placed and cancelled)")


# ─────────────────────────────────────────────────────────────
# Factory (mirrors get_crypto_paper_broker)
# ─────────────────────────────────────────────────────────────

def get_crypto_live_broker(initial_balance: float = 0.0) -> KuCoinLiveBroker:
    """Factory function matching the paper broker convention."""
    return KuCoinLiveBroker(initial_balance=initial_balance)
