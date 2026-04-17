"""
Broker abstractions and shadow-routing helpers.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import threading
import time
from gzip import decompress
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

logger = logging.getLogger(__name__)


class BaseBroker(ABC):
    """Minimal broker contract shared by paper and future live adapters."""

    positions: Dict[str, Any]
    orders: list
    trade_log: list

    @property
    @abstractmethod
    def cash(self) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def portfolio_value(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def submit_order(self, order: Any, current_price: float) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def update_prices(self, prices: Dict[str, float]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def has_open_position(self, *, symbol: Optional[str] = None, position_key: Optional[str] = None) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_summary(self) -> Dict[str, Any]:
        raise NotImplementedError


class UpstoxExecutionBroker:
    """
    Lightweight execution adapter for NSE orders.

    This is intentionally conservative:
    - supports only NSE equity instruments with configured instrument tokens
    - can run in dry-run mode for shadow validation
    - does not act as the portfolio source of truth for the strategy engine
    """

    PLACE_ORDER_URL = "https://api-hft.upstox.com/v2/order/place"
    NSE_INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
    NSE_SYMBOL_ALIASES = {
        "AARTI.NS": ("AARTIIND",),
        "DRLALPATH.NS": ("LALPATHLAB",),
        "ICICILOMBARD.NS": ("ICICIGI",),
        "ICICIPRULIFE.NS": ("ICICIPRULI",),
        "IPCA.NS": ("IPCALAB",),
        "LARSEN.NS": ("LT",),
        "LAXMIMACH.NS": ("LMW",),
        "LTIM.NS": ("LTM",),
        "VEDANT.NS": ("VEDL",),
        "ZOMATO.NS": ("ETERNAL",),
    }

    def __init__(
        self,
        access_token: str,
        *,
        instrument_map_path: Optional[str] = None,
        enabled: bool = False,
        dry_run: bool = True,
        timeout_seconds: int = 8,
        tracked_symbols: Optional[Iterable[str]] = None,
    ):
        self.access_token = (access_token or "").strip()
        self.enabled = bool(enabled and self.access_token)
        self.dry_run = bool(dry_run or not self.enabled)
        self.timeout_seconds = max(2, int(timeout_seconds))
        self.instrument_map_path = self._resolve_instrument_map_path(instrument_map_path)
        self.tracked_symbols = {
            str(symbol).strip().upper()
            for symbol in (tracked_symbols or [])
            if str(symbol).strip().upper().endswith(".NS")
        }
        self.last_sync_status: Dict[str, Any] = {
            "configured": False,
            "tracked_symbols": len(self.tracked_symbols),
            "resolved_symbols": 0,
            "missing_symbols": len(self.tracked_symbols),
        }
        self.instrument_map = self._load_instrument_map(self.instrument_map_path)
        self._ensure_instrument_coverage()

    @property
    def configured(self) -> bool:
        return bool(self.access_token)

    @staticmethod
    def _resolve_instrument_map_path(instrument_map_path: Optional[str]) -> Path:
        path_text = str(
            instrument_map_path
            or os.getenv("UPSTOX_INSTRUMENT_MAP_PATH", "data/upstox_instruments.json")
        ).strip()
        if not path_text:
            return Path.cwd() / "data" / "upstox_instruments.json"
        path = Path(path_text)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path

    def _load_instrument_map(self, instrument_map_path: Path) -> Dict[str, str]:
        if not instrument_map_path.exists():
            return {}
        try:
            payload = json.loads(instrument_map_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"Upstox instrument map load failed: {exc}")
            return {}
        mapping: Dict[str, str] = {}
        if isinstance(payload, dict):
            for key, value in payload.items():
                if key and value:
                    mapping[str(key).strip().upper()] = str(value).strip()
        return mapping

    def _covered_symbols(self) -> set[str]:
        return {symbol for symbol in self.tracked_symbols if symbol in self.instrument_map}

    @staticmethod
    def _normalize_nse_symbol(symbol: str) -> str:
        text = str(symbol or "").strip().upper()
        return text[:-3] if text.endswith(".NS") else text

    def _coverage_missing(self) -> set[str]:
        return {symbol for symbol in self.tracked_symbols if symbol not in self.instrument_map}

    def _ensure_instrument_coverage(self):
        if not self.tracked_symbols:
            self.last_sync_status = {
                "configured": True,
                "source": str(self.instrument_map_path),
                "tracked_symbols": 0,
                "resolved_symbols": len(self.instrument_map),
                "missing_symbols": 0,
            }
            return
        missing = self._coverage_missing()
        stale_seconds = max(0, int(os.getenv("UPSTOX_INSTRUMENT_SYNC_MAX_AGE_SECONDS", "86400")))
        should_sync = bool(missing)
        if self.instrument_map_path.exists() and stale_seconds > 0:
            age_seconds = (datetime.now(timezone.utc).timestamp() - self.instrument_map_path.stat().st_mtime)
            should_sync = should_sync or age_seconds > stale_seconds
        if should_sync:
            try:
                synced = self.sync_instrument_map()
                if synced:
                    self.instrument_map = synced
            except Exception as exc:
                logger.warning(f"Upstox instrument sync failed: {exc}")
        missing = self._coverage_missing()
        self.last_sync_status = {
            "configured": True,
            "source": str(self.instrument_map_path),
            "tracked_symbols": len(self.tracked_symbols),
            "resolved_symbols": len(self._covered_symbols()),
            "missing_symbols": len(missing),
            "missing_preview": sorted(list(missing))[:12],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def sync_instrument_map(self) -> Dict[str, str]:
        response = requests.get(self.NSE_INSTRUMENTS_URL, timeout=max(10, self.timeout_seconds * 3))
        response.raise_for_status()
        raw = response.content
        payload = json.loads(decompress(raw).decode("utf-8"))
        if not isinstance(payload, list):
            raise RuntimeError("unexpected Upstox instruments payload")

        tracked_base = {self._normalize_nse_symbol(symbol): symbol for symbol in self.tracked_symbols}
        alias_base: Dict[str, str] = {}
        for symbol, aliases in self.NSE_SYMBOL_ALIASES.items():
            if symbol not in self.tracked_symbols:
                continue
            for alias in aliases:
                alias_base[self._normalize_nse_symbol(alias)] = symbol
        mapping = dict(self.instrument_map)
        for row in payload:
            if not isinstance(row, dict):
                continue
            if str(row.get("segment") or "").upper() != "NSE_EQ":
                continue
            trading_symbol = self._normalize_nse_symbol(row.get("trading_symbol"))
            target_symbol = tracked_base.get(trading_symbol) or alias_base.get(trading_symbol)
            if not target_symbol:
                continue
            instrument_key = str(row.get("instrument_key") or "").strip()
            if not instrument_key:
                continue
            mapping[target_symbol] = instrument_key

        for symbol, aliases in self.NSE_SYMBOL_ALIASES.items():
            if symbol not in self.tracked_symbols or symbol in mapping:
                continue
            for alias in aliases:
                alias_symbol = f"{self._normalize_nse_symbol(alias)}.NS"
                if alias_symbol in mapping:
                    mapping[symbol] = mapping[alias_symbol]
                    break

        self.instrument_map_path.parent.mkdir(parents=True, exist_ok=True)
        self.instrument_map_path.write_text(json.dumps(dict(sorted(mapping.items())), indent=2), encoding="utf-8")
        logger.info(
            f"Upstox instrument sync complete: {len(mapping)} mapped | "
            f"{len([s for s in self.tracked_symbols if s not in mapping])} missing"
        )
        return mapping

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.access_token}",
        }

    def _resolve_instrument_token(self, order: Any) -> str:
        metadata = dict(getattr(order, "metadata", {}) or {})
        direct = str(metadata.get("instrument_token") or "").strip()
        if direct:
            return direct
        symbol = str(getattr(order, "symbol", "") or "").strip().upper()
        if symbol in self.instrument_map:
            return self.instrument_map[symbol]
        return ""

    @staticmethod
    def _side_value(order: Any) -> str:
        side = getattr(order, "side", None)
        side_value = getattr(side, "value", side)
        return str(side_value or "buy").upper()

    def _build_payload(self, order: Any) -> Dict[str, Any]:
        metadata = dict(getattr(order, "metadata", {}) or {})
        symbol = str(getattr(order, "symbol", "") or "")
        market = str(metadata.get("market") or "").upper()
        if not symbol.endswith(".NS") and market not in {"IN", "NSE"}:
            raise ValueError(f"unsupported market for Upstox execution: {symbol or market or 'unknown'}")

        instrument_token = self._resolve_instrument_token(order)
        if not instrument_token:
            raise ValueError(f"missing instrument token for {symbol}")

        requested_quantity = float(getattr(order, "quantity", 0.0) or 0.0)
        quantity = int(requested_quantity)
        if quantity <= 0 or abs(quantity - requested_quantity) > 1e-9:
            raise ValueError(f"Upstox requires whole-share NSE quantity for {symbol}, got {requested_quantity}")

        payload = {
            "quantity": quantity,
            "product": str(os.getenv("UPSTOX_ORDER_PRODUCT", "D")).strip().upper() or "D",
            "validity": str(os.getenv("UPSTOX_ORDER_VALIDITY", "DAY")).strip().upper() or "DAY",
            "price": 0 if getattr(order, "order_type", "market") == "market" else float(getattr(order, "limit_price", 0.0) or 0.0),
            "tag": str(metadata.get("position_key") or metadata.get("signal_key") or symbol)[:40],
            "instrument_token": instrument_token,
            "order_type": str(getattr(order, "order_type", "market")).strip().upper(),
            "transaction_type": self._side_value(order),
            "disclosed_quantity": 0,
            "trigger_price": float(metadata.get("trigger_price", 0.0) or 0.0),
            "is_amo": False,
        }
        return payload

    def submit_order(self, order: Any, current_price: float) -> Dict[str, Any]:
        if not self.configured:
            return {"status": "not_configured", "broker": "upstox", "reason": "UPSTOX_ACCESS_TOKEN missing"}
        try:
            payload = self._build_payload(order)
        except Exception as exc:
            return {"status": "unsupported", "broker": "upstox", "reason": str(exc)}

        if self.dry_run:
            return {
                "status": "shadow_ready",
                "broker": "upstox",
                "payload": payload,
                "reference_price": float(current_price),
                "instrument_sync": dict(self.last_sync_status),
            }

        response = requests.post(
            self.PLACE_ORDER_URL,
            headers=self._headers(),
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        if body.get("status") != "success":
            raise RuntimeError(body.get("message") or "Upstox order rejected")
        data = body.get("data") or {}
        return {
            "status": "submitted",
            "broker": "upstox",
            "order_id": data.get("order_id"),
            "payload": payload,
            "reference_price": float(current_price),
            "instrument_sync": dict(self.last_sync_status),
        }


class UpstoxV3ExecutionBroker(UpstoxExecutionBroker):
    """
    V3 execution adapter: adds margin tiering, auto-slicing, market
    protection, SEBI Algo-ID and the 15:20 IST square-off gate on top of the
    V2 instrument-map plumbing inherited from UpstoxExecutionBroker.

    Interface-compatible with UpstoxExecutionBroker.submit_order so run.py's
    _execute_lane_entries path routes without modification.
    """

    def __init__(
        self,
        access_token: str,
        *,
        analytics_token: str = "",
        algo_name: str = "",
        instrument_map_path: Optional[str] = None,
        enabled: bool = False,
        dry_run: bool = True,
        timeout_seconds: int = 8,
        tracked_symbols: Optional[Iterable[str]] = None,
        default_product: str = "I",
    ):
        super().__init__(
            access_token=access_token,
            instrument_map_path=instrument_map_path,
            enabled=enabled,
            dry_run=dry_run,
            timeout_seconds=timeout_seconds,
            tracked_symbols=tracked_symbols,
        )
        # Lazy import to avoid circular references when run.py loads brokerages first.
        try:
            from upstox_v3_client import UpstoxTokens, UpstoxV3Client
        except Exception:
            from project.upstox_v3_client import UpstoxTokens, UpstoxV3Client  # type: ignore[no-redef]
        self._v3_client = UpstoxV3Client(
            tokens=UpstoxTokens(
                execution_token=self.access_token,
                analytics_token=(analytics_token or "").strip()
                or os.getenv("UPSTOX_ANALYTICS_TOKEN", "").strip(),
            ),
            algo_name=algo_name or os.getenv("UPSTOX_ALGO_NAME", "").strip(),
            dry_run=self.dry_run,
            enabled=self.enabled and not self.dry_run,
        )
        self.default_product = str(default_product or "I").strip().upper() or "I"
        mode = "DRY-RUN" if self.dry_run else "LIVE"
        logger.info(
            f"UpstoxV3ExecutionBroker [{mode}] algo={self._v3_client.algo_name or 'ALGO_UNSET'} "
            f"tracked={len(self.tracked_symbols)} resolved={len(self.instrument_map)}"
        )

    @staticmethod
    def _run_async(coro):
        """Run an async coroutine from a sync context (thread-safe)."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule on the running loop and wait.
                import concurrent.futures

                future = asyncio.run_coroutine_threadsafe(coro, loop)
                return future.result(timeout=30)
        except RuntimeError:
            pass
        return asyncio.run(coro)

    def submit_order(self, order: Any, current_price: float) -> Dict[str, Any]:
        if not self.configured:
            return {"status": "not_configured", "broker": "upstox_v3", "reason": "UPSTOX_ACCESS_TOKEN missing"}

        # Resolve instrument_key via inherited instrument_map
        try:
            instrument_key = self._resolve_instrument_token(order)
            if not instrument_key:
                return {
                    "status": "unsupported",
                    "broker": "upstox_v3",
                    "reason": f"missing instrument token for {getattr(order, 'symbol', '?')}",
                }
            symbol = str(getattr(order, "symbol", "") or "").strip().upper()
            market = str((getattr(order, "metadata", {}) or {}).get("market") or "").upper()
            if not symbol.endswith(".NS") and market not in {"IN", "NSE"}:
                return {
                    "status": "unsupported",
                    "broker": "upstox_v3",
                    "reason": f"non-NSE symbol {symbol}",
                }
        except Exception as exc:
            return {"status": "unsupported", "broker": "upstox_v3", "reason": str(exc)}

        try:
            requested_quantity = float(getattr(order, "quantity", 0.0) or 0.0)
            quantity = int(requested_quantity)
            if quantity <= 0:
                return {"status": "unsupported", "broker": "upstox_v3", "reason": "non-positive qty"}

            metadata = dict(getattr(order, "metadata", {}) or {})
            order_type = str(getattr(order, "order_type", "market") or "market").upper()
            limit_price = float(getattr(order, "limit_price", 0.0) or 0.0)
            side_raw = getattr(order, "side", None)
            side_value = str(getattr(side_raw, "value", side_raw) or "buy").upper()

            try:
                from upstox_v3_client import OrderRequest
            except Exception:
                from project.upstox_v3_client import OrderRequest  # type: ignore[no-redef]

            req = OrderRequest(
                symbol=symbol,
                instrument_key=instrument_key,
                side=side_value,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                product=str(metadata.get("upstox_product") or self.default_product),
                tag=str(metadata.get("position_key") or metadata.get("signal_key") or symbol)[:40],
                lot_size=int(metadata.get("lot_size") or 1),
                freeze_qty=int(metadata.get("freeze_qty") or 0) or None,
            )
            result = self._run_async(
                self._v3_client.submit_order(req, reference_price=float(current_price or 0.0))
            )
        except Exception as exc:
            logger.warning(f"UpstoxV3 submit failed: {exc}")
            return {"status": "error", "broker": "upstox_v3", "reason": str(exc)}

        return {
            "status": str(result.status),
            "broker": "upstox_v3",
            "reason": str(result.reason or ""),
            "margin_tier": str(result.margin_tier or ""),
            "size_multiplier": float(result.size_multiplier or 1.0),
            "algo_id_used": str(result.algo_id_used or ""),
            "market_protection_applied": bool(result.market_protection_applied),
            "child_orders": list(result.child_orders or []),
            "reference_price": float(current_price or 0.0),
            "instrument_sync": dict(self.last_sync_status),
        }

    def square_off_all_intraday(self) -> List[Dict[str, Any]]:
        """Wrapper around V3 client square-off for callers in sync context."""
        try:
            results = self._run_async(self._v3_client.square_off_all_intraday())
        except Exception as exc:
            logger.warning(f"UpstoxV3 square-off failed: {exc}")
            return []
        return [
            {
                "status": r.status,
                "reason": r.reason,
                "margin_tier": r.margin_tier,
                "market_protection_applied": r.market_protection_applied,
                "child_orders": r.child_orders,
            }
            for r in (results or [])
        ]


class ZerodhaExecutionBroker:
    """
    Execution adapter for NSE orders via Zerodha Kite Connect API v3.

    Safety layers:
    - MAX_STOCK_ORDER_INR hard cap per order
    - MAX_STOCK_LIVE_POSITIONS hard cap on open positions
    - STOCK_MAX_DAILY_LOSS_INR circuit breaker
    - LIVE_SIZE_MULTIPLIER scales all quantities
    - dry_run mode by default
    - LIMIT orders only (25% inside spread), never market fallback
    """

    BASE_URL = "https://api.kite.trade"
    LIMIT_WAIT_SECONDS = 8.0

    def __init__(
        self,
        *,
        api_key: str = "",
        access_token: str = "",
        enabled: bool = False,
        dry_run: bool = True,
        product_type: str = "MIS",
        timeout_seconds: int = 8,
    ):
        self.api_key = (api_key or "").strip()
        self.access_token = (access_token or "").strip()
        self.enabled = bool(enabled and self.api_key and self.access_token)
        self.dry_run = bool(dry_run or not self.enabled)
        self.product_type = product_type.strip().upper() if product_type else "MIS"
        if self.product_type not in {"MIS", "CNC"}:
            self.product_type = "MIS"
        self.timeout_seconds = max(2, int(timeout_seconds))

        self._size_multiplier = self._env_float("LIVE_SIZE_MULTIPLIER", 0.10)
        self._max_order_inr = self._env_float("MAX_STOCK_ORDER_INR", 2000.0)
        self._max_positions = self._env_int("MAX_STOCK_LIVE_POSITIONS", 3)
        self._max_daily_loss = self._env_float("STOCK_MAX_DAILY_LOSS_INR", 500.0)

        self._lock = threading.Lock()
        self._daily_realized_pnl: float = 0.0
        self._daily_reset_date: str = ""
        self._halted: bool = False
        self._open_position_count: int = 0
        self._order_log: List[Dict[str, Any]] = []

        self.last_sync_status: Dict[str, Any] = {"configured": self.configured}

        mode = "DRY-RUN" if self.dry_run else "LIVE"
        logger.info(
            f"ZerodhaExecutionBroker [{mode}] product={self.product_type} "
            f"size_mult={self._size_multiplier} max_order=₹{self._max_order_inr} "
            f"max_pos={self._max_positions} max_daily_loss=₹{self._max_daily_loss}"
        )

    # ── helpers ──────────────────────────────────────────────

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)).strip() or default)
        except ValueError:
            return default

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)).strip() or default)
        except ValueError:
            return default

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.access_token)

    @staticmethod
    def _translate_symbol(symbol: str) -> str:
        """Convert 'RELIANCE.NS' → 'RELIANCE' (NSE trading symbol)."""
        text = str(symbol or "").strip().upper()
        if text.endswith(".NS"):
            return text[:-3]
        return text

    def _headers(self) -> Dict[str, str]:
        return {
            "X-Kite-Version": "3",
            "Authorization": f"token {self.api_key}:{self.access_token}",
            "Content-Type": "application/x-www-form-urlencoded",
        }

    def _scale_quantity(self, recommended_qty: float) -> int:
        scaled = recommended_qty * self._size_multiplier
        return max(1, int(math.floor(scaled)))

    # ── circuit breaker ──────────────────────────────────────

    def _check_daily_reset(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._daily_reset_date != today:
            self._daily_reset_date = today
            self._daily_realized_pnl = 0.0
            self._halted = False

    def _record_pnl(self, pnl: float) -> None:
        self._daily_realized_pnl += pnl
        if self._daily_realized_pnl <= -self._max_daily_loss:
            self._halted = True
            logger.warning(
                f"ZERODHA CIRCUIT BREAKER: daily loss ₹{self._daily_realized_pnl:.2f} "
                f"exceeds limit ₹{self._max_daily_loss:.2f}"
            )

    # ── guard checks ─────────────────────────────────────────

    def _guard_checks(self, symbol: str, side: str, quantity: int, price: float) -> Optional[str]:
        self._check_daily_reset()
        is_exit = side.upper() == "SELL"
        if self._halted and not is_exit:
            return f"circuit breaker active (daily loss ₹{self._daily_realized_pnl:.2f})"
        if not is_exit:
            order_value = quantity * price
            if order_value > self._max_order_inr:
                return f"order value ₹{order_value:.2f} exceeds MAX_STOCK_ORDER_INR ₹{self._max_order_inr:.2f}"
            if self._open_position_count >= self._max_positions:
                return f"open positions ({self._open_position_count}) at limit ({self._max_positions})"
        return None

    # ── order execution ──────────────────────────────────────

    def _place_kite_order(
        self,
        trading_symbol: str,
        side: str,
        quantity: int,
        order_type: str,
        price: float = 0.0,
    ) -> Dict[str, Any]:
        """Place order via Kite Connect API."""
        payload = {
            "tradingsymbol": trading_symbol,
            "exchange": "NSE",
            "transaction_type": side.upper(),
            "order_type": order_type.upper(),
            "quantity": int(quantity),
            "product": self.product_type,
            "validity": "DAY",
        }
        if order_type.upper() == "LIMIT" and price > 0:
            payload["price"] = round(price, 2)

        resp = requests.post(
            f"{self.BASE_URL}/orders/regular",
            headers=self._headers(),
            data=payload,
            timeout=self.timeout_seconds,
        )
        body = resp.json()
        if body.get("status") != "success":
            raise RuntimeError(body.get("message") or f"Kite order rejected ({resp.status_code})")
        return body.get("data", {})

    def _get_order_status(self, order_id: str) -> Dict[str, Any]:
        resp = requests.get(
            f"{self.BASE_URL}/orders/{order_id}",
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        body = resp.json()
        if body.get("status") != "success":
            raise RuntimeError(body.get("message") or "Kite order status fetch failed")
        # Kite returns list of order updates, last is current
        updates = body.get("data", [])
        return updates[-1] if updates else {}

    def _cancel_kite_order(self, order_id: str) -> None:
        requests.delete(
            f"{self.BASE_URL}/orders/regular/{order_id}",
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )

    def _wait_for_fill(self, order_id: str) -> Optional[Dict[str, Any]]:
        deadline = time.monotonic() + self.LIMIT_WAIT_SECONDS
        while time.monotonic() < deadline:
            try:
                status = self._get_order_status(order_id)
                kite_status = str(status.get("status", "")).upper()
                if kite_status == "COMPLETE":
                    return status
                if kite_status in {"CANCELLED", "REJECTED"}:
                    return None
            except Exception:
                pass
            time.sleep(0.5)
        return None

    def _execute_limit_with_retry(
        self,
        trading_symbol: str,
        side: str,
        quantity: int,
        bid: float,
        ask: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Limit order 25% inside spread → wait → cancel → retry at mid → skip.
        Never falls back to market.
        """
        spread = ask - bid
        if side.upper() == "BUY":
            price_1 = ask - spread * 0.25
            price_2 = (bid + ask) / 2.0
        else:
            price_1 = bid + spread * 0.25
            price_2 = (bid + ask) / 2.0

        for attempt, limit_price in enumerate([price_1, price_2], start=1):
            limit_price = round(max(limit_price, 0.05), 2)  # Kite min tick ₹0.05

            logger.info(
                f"Zerodha LIMIT attempt {attempt}/2: {side.upper()} {quantity} "
                f"{trading_symbol} @ ₹{limit_price:.2f} (bid={bid:.2f} ask={ask:.2f})"
            )

            if self.dry_run:
                return {
                    "order_id": f"dry-{trading_symbol}-{int(time.time()*1000)}",
                    "average_price": limit_price,
                    "filled_quantity": quantity,
                    "status": "shadow_ready",
                }

            try:
                result = self._place_kite_order(trading_symbol, side, quantity, "LIMIT", limit_price)
            except Exception as exc:
                logger.error(f"Zerodha order placement failed: {exc}")
                return None

            order_id = result.get("order_id", "")
            if not order_id:
                logger.error(f"Zerodha returned no order_id: {result}")
                return None

            filled = self._wait_for_fill(order_id)
            if filled:
                fill_price = float(filled.get("average_price", limit_price))
                fill_qty = int(filled.get("filled_quantity", quantity))
                logger.info(
                    f"Zerodha FILLED: {side.upper()} {fill_qty} {trading_symbol} "
                    f"@ ₹{fill_price:.2f} (orderId={order_id})"
                )
                return {
                    "order_id": order_id,
                    "average_price": fill_price,
                    "filled_quantity": fill_qty,
                    "status": "filled",
                }

            # Cancel unfilled
            try:
                self._cancel_kite_order(order_id)
                logger.info(f"Zerodha cancelled unfilled order {order_id} (attempt {attempt})")
            except Exception as exc:
                logger.warning(f"Zerodha cancel failed for {order_id}: {exc}")

        logger.warning(
            f"Zerodha SKIP: {side.upper()} {quantity} {trading_symbol} — "
            f"not filled after 2 limit attempts"
        )
        return None

    def _log_order(self, symbol: str, side: str, qty: int, price: float, result: str, order_id: str = "") -> None:
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
        if len(self._order_log) > 5000:
            del self._order_log[:len(self._order_log) - 5000]
        logger.info(
            f"ZERODHA ORDER: {side.upper()} {qty} {symbol} "
            f"@ ₹{price:.2f} → {result} (id={order_id})"
        )

    # ── public interface (matches UpstoxExecutionBroker) ─────

    def submit_order(self, order: Any, current_price: float) -> Dict[str, Any]:
        """Submit order via Zerodha Kite. Same interface as UpstoxExecutionBroker."""
        symbol_raw = str(getattr(order, "symbol", "") or "").strip().upper()
        metadata = dict(getattr(order, "metadata", {}) or {})
        market = str(metadata.get("market") or "").upper()

        # Only handle Indian equity
        if not symbol_raw.endswith(".NS") and market not in {"IN", "NSE"}:
            return {"status": "unsupported", "broker": "zerodha", "reason": f"not an NSE symbol: {symbol_raw}"}
        if not self.configured:
            return {"status": "not_configured", "broker": "zerodha", "reason": "ZERODHA_API_KEY/ACCESS_TOKEN missing"}

        trading_symbol = self._translate_symbol(symbol_raw)
        side_obj = getattr(order, "side", None)
        side = str(getattr(side_obj, "value", side_obj) or "buy").strip().upper()
        raw_qty = float(getattr(order, "quantity", 0) or 0)
        quantity = self._scale_quantity(raw_qty) if side == "BUY" else max(1, int(raw_qty))

        with self._lock:
            reason = self._guard_checks(symbol_raw, side, quantity, current_price)
            if reason:
                logger.warning(f"ZERODHA REJECTED: {reason} | {side} {quantity} {trading_symbol}")
                self._log_order(trading_symbol, side, quantity, current_price, f"rejected: {reason}")
                return {"status": "rejected", "broker": "zerodha", "reason": reason}

            # Use current_price as bid/ask proxy when no L1 data
            bid = float(metadata.get("best_bid", current_price * 0.999))
            ask = float(metadata.get("best_ask", current_price * 1.001))

            fill = self._execute_limit_with_retry(trading_symbol, side, quantity, bid, ask)
            if fill is None:
                self._log_order(trading_symbol, side, quantity, current_price, "skipped")
                return {"status": "skipped", "broker": "zerodha", "reason": "limit order not filled after 2 attempts"}

            fill_price = float(fill.get("average_price", current_price))
            fill_qty = int(fill.get("filled_quantity", quantity))
            order_id = str(fill.get("order_id", ""))
            status = fill.get("status", "filled")

            if side == "BUY":
                self._open_position_count += 1
            elif side == "SELL":
                self._open_position_count = max(0, self._open_position_count - 1)
                # Estimate PnL from metadata if available
                entry_price = float(metadata.get("entry_price", 0) or 0)
                if entry_price > 0:
                    pnl = (fill_price - entry_price) * fill_qty
                    self._record_pnl(pnl)

            self._log_order(trading_symbol, side, fill_qty, fill_price, status, order_id)

            return {
                "status": "submitted" if status != "shadow_ready" else "shadow_ready",
                "broker": "zerodha",
                "order_id": order_id,
                "fill_price": fill_price,
                "fill_quantity": fill_qty,
                "reference_price": float(current_price),
            }


class ShadowBroker:
    """Primary paper broker with optional secondary broker validation/routing."""

    def __init__(
        self,
        primary: BaseBroker,
        *,
        secondary: Optional[Any] = None,
        reconciliation_path: Optional[str] = None,
    ):
        self.primary = primary
        self.secondary = secondary
        self.positions = primary.positions
        self.orders = primary.orders
        self.trade_log = primary.trade_log
        self.shadow_trade_log = []
        path_text = str(
            reconciliation_path
            or os.getenv("SHADOW_RECONCILIATION_LOG_PATH", "data/shadow_execution_log.jsonl")
        ).strip()
        self._reconciliation_path = Path(path_text)

    def __getattr__(self, item: str):
        return getattr(self.primary, item)

    @property
    def cash(self) -> float:
        return float(getattr(self.primary, "cash", 0.0) or 0.0)

    @property
    def portfolio_value(self) -> float:
        return float(getattr(self.primary, "portfolio_value", 0.0) or 0.0)

    def _clone_order(self, order: Any) -> Any:
        if is_dataclass(order):
            return replace(order, metadata=dict(getattr(order, "metadata", {}) or {}))
        return asdict(order)

    def _persist_shadow_event(self, event: Dict[str, Any]):
        try:
            self._reconciliation_path.parent.mkdir(parents=True, exist_ok=True)
            with self._reconciliation_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, default=str) + "\n")
        except Exception as exc:
            logger.warning(f"Shadow reconciliation save failed: {exc}")

    def _shadow_submit(self, order: Any, current_price: float) -> Dict[str, Any]:
        if self.secondary is None:
            return {"status": "not_configured", "reason": "no_secondary_broker"}
        try:
            cloned = self._clone_order(order)
            return self.secondary.submit_order(cloned, current_price)
        except Exception as exc:
            return {"status": "error", "reason": str(exc)}

    def submit_order(self, order: Any, current_price: float) -> Dict[str, Any]:
        shadow_order = self._clone_order(order)
        primary_result = self.primary.submit_order(order, current_price)
        shadow_result = self._shadow_submit(shadow_order, current_price)
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": getattr(shadow_order, "symbol", ""),
            "position_key": getattr(shadow_order, "position_key", None),
            "side": getattr(getattr(shadow_order, "side", None), "value", getattr(shadow_order, "side", "")),
            "requested_quantity": float((getattr(shadow_order, "metadata", {}) or {}).get("requested_quantity", getattr(shadow_order, "quantity", 0.0)) or 0.0),
            "paper_result": primary_result,
            "shadow_result": shadow_result,
        }
        self.shadow_trade_log.append(event)
        if len(self.shadow_trade_log) > 1000:
            del self.shadow_trade_log[: len(self.shadow_trade_log) - 1000]
        self._persist_shadow_event(event)
        merged = dict(primary_result)
        merged["shadow"] = shadow_result
        return merged

    def update_prices(self, prices: Dict[str, float]) -> Dict[str, Any]:
        return self.primary.update_prices(prices)

    def has_open_position(self, *, symbol: Optional[str] = None, position_key: Optional[str] = None) -> bool:
        return self.primary.has_open_position(symbol=symbol, position_key=position_key)

    def get_summary(self) -> Dict[str, Any]:
        summary = dict(self.primary.get_summary() or {})
        summary["execution_mode"] = "shadow"
        summary["shadow_router"] = {
            "configured": self.secondary is not None,
            "broker": getattr(self.secondary, "__class__", type("obj", (), {})).__name__ if self.secondary is not None else None,
            "events_logged": len(self.shadow_trade_log),
            "reconciliation_path": str(self._reconciliation_path),
            "instrument_sync": dict(getattr(self.secondary, "last_sync_status", {}) or {}),
        }
        return summary
