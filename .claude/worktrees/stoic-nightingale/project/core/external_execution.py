"""
Optional live and shadow execution adapters for external venues.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

from core.brokerages import UpstoxExecutionBroker

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _format_quantity(value: float, decimals: int = 8) -> str:
    text = f"{float(value):.{decimals}f}"
    return text.rstrip("0").rstrip(".") or "0"


def _load_json_mapping(path_value: Optional[str]) -> Dict[str, str]:
    path_text = str(path_value or "").strip()
    if not path_text:
        return {}
    path = Path(path_text)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Execution symbol map load failed from {path}: {exc}")
        return {}
    mapping: Dict[str, str] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key and value:
                mapping[str(key).strip().upper()] = str(value).strip()
    return mapping


def _order_symbol(order: Any) -> str:
    return str(getattr(order, "symbol", "") or "").strip().upper()


def _order_market(order: Any) -> str:
    metadata = dict(getattr(order, "metadata", {}) or {})
    market = str(metadata.get("market") or "").strip().upper()
    if market:
        return market
    symbol = _order_symbol(order)
    if symbol.endswith(".NS") or symbol.startswith("^NSE"):
        return "IN"
    if symbol.endswith(("USDT", "USDC", "BUSD")):
        return "CRYPTO"
    return "US"


def _order_side(order: Any) -> str:
    side = getattr(order, "side", None)
    return str(getattr(side, "value", side) or "buy").strip().lower()


def _order_type(order: Any) -> str:
    return str(getattr(order, "order_type", "market") or "market").strip().lower()


def _success_status(status: str) -> bool:
    return str(status or "").lower() in {"submitted", "shadow_ready", "accepted"}


class MetaTrader5ExecutionBroker:
    def __init__(
        self,
        *,
        login: str = "",
        password: str = "",
        server: str = "",
        path: str = "",
        symbol_map_path: Optional[str] = None,
        enabled: bool = False,
        dry_run: bool = True,
        timeout_seconds: int = 10,
        deviation_points: int = 20,
    ):
        self.login = str(login or "").strip()
        self.password = str(password or "").strip()
        self.server = str(server or "").strip()
        self.path = str(path or "").strip()
        self.enabled = bool(enabled)
        self.dry_run = bool(dry_run or not self.enabled)
        self.timeout_seconds = max(2, int(timeout_seconds))
        self.deviation_points = max(1, int(deviation_points))
        self.symbol_map = _load_json_mapping(symbol_map_path or os.getenv("MT5_SYMBOL_MAP_PATH", "data/mt5_symbols.json"))
        self._lock = threading.Lock()
        self._initialized = False
        self._last_error = ""
        try:
            import MetaTrader5 as mt5  # type: ignore

            self._mt5 = mt5
        except Exception as exc:
            self._mt5 = None
            self._last_error = str(exc)

    @property
    def configured(self) -> bool:
        return self._mt5 is not None

    def _resolve_symbol(self, order: Any) -> str:
        symbol = _order_symbol(order)
        metadata = dict(getattr(order, "metadata", {}) or {})
        direct = str(metadata.get("mt5_symbol") or "").strip()
        if direct:
            return direct
        if symbol in self.symbol_map:
            return self.symbol_map[symbol]
        if symbol.endswith(".NS") and symbol[:-3] in self.symbol_map:
            return self.symbol_map[symbol[:-3]]
        return symbol[:-3] if symbol.endswith(".NS") else symbol

    def _initialize(self) -> bool:
        if self._mt5 is None:
            return False
        with self._lock:
            if self._initialized:
                return True
            init_kwargs: Dict[str, Any] = {"timeout": self.timeout_seconds * 1000}
            if self.path:
                init_kwargs["path"] = self.path
            if self.login:
                init_kwargs["login"] = int(self.login)
            if self.password:
                init_kwargs["password"] = self.password
            if self.server:
                init_kwargs["server"] = self.server
            initialized = bool(self._mt5.initialize(**init_kwargs))
            if not initialized:
                self._last_error = str(self._mt5.last_error())
                return False
            self._initialized = True
            return True

    def submit_order(self, order: Any, current_price: float) -> Dict[str, Any]:
        if not self.configured:
            return {"status": "not_configured", "broker": "mt5", "reason": self._last_error or "MetaTrader5 package unavailable"}
        if not self._initialize():
            return {"status": "error", "broker": "mt5", "reason": self._last_error or "MT5 initialize failed"}

        symbol = self._resolve_symbol(order)
        if not symbol:
            return {"status": "unsupported", "broker": "mt5", "reason": "missing MT5 symbol mapping"}

        mt5 = self._mt5
        if not mt5.symbol_select(symbol, True):
            return {"status": "unsupported", "broker": "mt5", "reason": f"symbol not available: {symbol}"}

        tick = mt5.symbol_info_tick(symbol)
        symbol_info = mt5.symbol_info(symbol)
        side = _order_side(order)
        order_type = _order_type(order)
        quantity = max(_safe_float(getattr(order, "quantity", 0.0), 0.0), 0.0)
        if quantity <= 0:
            return {"status": "unsupported", "broker": "mt5", "reason": f"invalid quantity for {symbol}"}

        price = _safe_float(getattr(order, "limit_price", None), 0.0)
        if order_type == "market":
            price = _safe_float(getattr(tick, "ask" if side == "buy" else "bid", 0.0), current_price)
        if price <= 0:
            price = max(_safe_float(current_price, 0.0), 0.0)
        if price <= 0:
            return {"status": "unsupported", "broker": "mt5", "reason": f"missing executable price for {symbol}"}

        payload = {
            "action": mt5.TRADE_ACTION_DEAL if order_type == "market" else mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": quantity,
            "type": (
                mt5.ORDER_TYPE_BUY if side == "buy" and order_type == "market"
                else mt5.ORDER_TYPE_SELL if side == "sell" and order_type == "market"
                else mt5.ORDER_TYPE_BUY_LIMIT if side == "buy"
                else mt5.ORDER_TYPE_SELL_LIMIT
            ),
            "price": price,
            "deviation": self.deviation_points,
            "magic": int(os.getenv("MT5_MAGIC_NUMBER", "20260408")),
            "comment": str((getattr(order, "metadata", {}) or {}).get("signal_key") or _order_symbol(order))[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": getattr(symbol_info, "filling_mode", mt5.ORDER_FILLING_RETURN),
        }
        if getattr(order, "limit_price", None) is not None and order_type == "limit":
            payload["price"] = _safe_float(getattr(order, "limit_price", None), price)

        if self.dry_run:
            return {
                "status": "shadow_ready",
                "broker": "mt5",
                "payload": payload,
                "resolved_symbol": symbol,
                "reference_price": float(current_price),
            }

        result = mt5.order_send(payload)
        if result is None:
            return {"status": "error", "broker": "mt5", "reason": str(mt5.last_error()), "payload": payload}
        result_dict = result._asdict() if hasattr(result, "_asdict") else {"result": str(result)}
        retcode = int(result_dict.get("retcode", 0) or 0)
        if retcode not in {
            getattr(mt5, "TRADE_RETCODE_DONE", 10009),
            getattr(mt5, "TRADE_RETCODE_PLACED", 10008),
        }:
            return {
                "status": "rejected",
                "broker": "mt5",
                "reason": str(result_dict.get("comment") or result_dict.get("retcode") or "MT5 order rejected"),
                "payload": payload,
                "response": result_dict,
            }
        return {
            "status": "submitted",
            "broker": "mt5",
            "order_id": result_dict.get("order") or result_dict.get("deal"),
            "payload": payload,
            "response": result_dict,
            "resolved_symbol": symbol,
            "reference_price": float(current_price),
        }


class BinanceSpotExecutionBroker:
    def __init__(
        self,
        *,
        api_key: str = "",
        api_secret: str = "",
        enabled: bool = False,
        dry_run: bool = True,
        base_url: str = "https://testnet.binance.vision",
        timeout_seconds: int = 8,
        recv_window_ms: int = 5000,
    ):
        self.api_key = str(api_key or "").strip()
        self.api_secret = str(api_secret or "").strip()
        self.enabled = bool(enabled and self.api_key and self.api_secret)
        self.dry_run = bool(dry_run or not self.enabled)
        self.base_url = str(base_url or "https://testnet.binance.vision").rstrip("/")
        self.timeout_seconds = max(2, int(timeout_seconds))
        self.recv_window_ms = max(1000, int(recv_window_ms))

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.api_secret)

    def submit_order(self, order: Any, current_price: float) -> Dict[str, Any]:
        symbol = _order_symbol(order)
        market = _order_market(order)
        if market != "CRYPTO" or not symbol.endswith(("USDT", "USDC", "BUSD")):
            return {"status": "unsupported", "broker": "binance", "reason": f"unsupported symbol {symbol}"}
        if not self.configured:
            return {"status": "not_configured", "broker": "binance", "reason": "BINANCE_API_KEY/SECRET missing"}

        side = _order_side(order).upper()
        order_type = "LIMIT" if _order_type(order) == "limit" else "MARKET"
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": _format_quantity(_safe_float(getattr(order, "quantity", 0.0), 0.0)),
            "recvWindow": self.recv_window_ms,
            "timestamp": int(time.time() * 1000),
            "newClientOrderId": str((getattr(order, "metadata", {}) or {}).get("signal_key") or f"shadow-{symbol}-{int(time.time())}")[:32],
        }
        if order_type == "LIMIT":
            payload["price"] = _format_quantity(_safe_float(getattr(order, "limit_price", None), current_price))
            payload["timeInForce"] = "GTC"

        ordered_items = [(key, payload[key]) for key in sorted(payload.keys())]
        query = "&".join(f"{key}={value}" for key, value in ordered_items)
        signature = hmac.new(self.api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()
        signed_payload = ordered_items + [("signature", signature)]

        if self.dry_run:
            return {
                "status": "shadow_ready",
                "broker": "binance",
                "payload": signed_payload,
                "reference_price": float(current_price),
            }

        try:
            response = requests.post(
                f"{self.base_url}/api/v3/order",
                params=signed_payload,
                headers={"X-MBX-APIKEY": self.api_key},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
            return {
                "status": "submitted",
                "broker": "binance",
                "order_id": body.get("orderId"),
                "payload": signed_payload,
                "response": body,
                "reference_price": float(current_price),
            }
        except Exception as exc:
            return {"status": "error", "broker": "binance", "reason": str(exc), "payload": signed_payload}


class BybitSpotExecutionBroker:
    def __init__(
        self,
        *,
        api_key: str = "",
        api_secret: str = "",
        enabled: bool = False,
        dry_run: bool = True,
        base_url: str = "https://api-testnet.bybit.com",
        timeout_seconds: int = 8,
        recv_window_ms: int = 5000,
    ):
        self.api_key = str(api_key or "").strip()
        self.api_secret = str(api_secret or "").strip()
        self.enabled = bool(enabled and self.api_key and self.api_secret)
        self.dry_run = bool(dry_run or not self.enabled)
        self.base_url = str(base_url or "https://api-testnet.bybit.com").rstrip("/")
        self.timeout_seconds = max(2, int(timeout_seconds))
        self.recv_window_ms = max(1000, int(recv_window_ms))

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.api_secret)

    def submit_order(self, order: Any, current_price: float) -> Dict[str, Any]:
        symbol = _order_symbol(order)
        market = _order_market(order)
        if market != "CRYPTO" or not symbol.endswith(("USDT", "USDC", "BUSD")):
            return {"status": "unsupported", "broker": "bybit", "reason": f"unsupported symbol {symbol}"}
        if not self.configured:
            return {"status": "not_configured", "broker": "bybit", "reason": "BYBIT_API_KEY/SECRET missing"}

        payload: Dict[str, Any] = {
            "category": "spot",
            "symbol": symbol,
            "side": "Buy" if _order_side(order) == "buy" else "Sell",
            "orderType": "Limit" if _order_type(order) == "limit" else "Market",
            "qty": _format_quantity(_safe_float(getattr(order, "quantity", 0.0), 0.0)),
            "orderLinkId": str((getattr(order, "metadata", {}) or {}).get("signal_key") or f"shadow-{symbol}-{int(time.time())}")[:36],
        }
        if payload["orderType"] == "Limit":
            payload["price"] = _format_quantity(_safe_float(getattr(order, "limit_price", None), current_price))
            payload["timeInForce"] = "GTC"

        body = json.dumps(payload, separators=(",", ":"))
        timestamp = str(int(time.time() * 1000))
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            f"{timestamp}{self.api_key}{self.recv_window_ms}{body}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": str(self.recv_window_ms),
            "X-BAPI-SIGN": signature,
            "Content-Type": "application/json",
        }

        if self.dry_run:
            return {
                "status": "shadow_ready",
                "broker": "bybit",
                "payload": payload,
                "reference_price": float(current_price),
            }

        try:
            response = requests.post(
                f"{self.base_url}/v5/order/create",
                headers=headers,
                data=body,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            result = response.json()
            if int(result.get("retCode", -1) or -1) != 0:
                return {
                    "status": "rejected",
                    "broker": "bybit",
                    "reason": str(result.get("retMsg") or "Bybit order rejected"),
                    "payload": payload,
                    "response": result,
                }
            data = result.get("result") or {}
            return {
                "status": "submitted",
                "broker": "bybit",
                "order_id": data.get("orderId"),
                "payload": payload,
                "response": result,
                "reference_price": float(current_price),
            }
        except Exception as exc:
            return {"status": "error", "broker": "bybit", "reason": str(exc), "payload": payload}


class FallbackExecutionBroker:
    def __init__(self, brokers: Iterable[Any], *, name: str = "multi_router"):
        self.brokers = [broker for broker in brokers if broker is not None]
        self.name = name
        self.last_sync_status = {
            getattr(broker, "__class__", type("obj", (), {})).__name__: getattr(broker, "configured", False)
            for broker in self.brokers
        }

    @property
    def configured(self) -> bool:
        return any(bool(getattr(broker, "configured", False)) for broker in self.brokers)

    def submit_order(self, order: Any, current_price: float) -> Dict[str, Any]:
        attempts: List[Dict[str, Any]] = []
        last_result: Dict[str, Any] = {"status": "not_configured", "broker": self.name, "reason": "no_execution_router"}
        for broker in self.brokers:
            try:
                result = dict(broker.submit_order(order, current_price) or {})
            except Exception as exc:
                result = {"status": "error", "broker": broker.__class__.__name__.lower(), "reason": str(exc)}
            attempts.append(
                {
                    "broker": result.get("broker", broker.__class__.__name__.lower()),
                    "status": result.get("status"),
                    "reason": result.get("reason", ""),
                }
            )
            last_result = result
            if _success_status(str(result.get("status", ""))):
                result["attempts"] = attempts
                result["router"] = self.name
                return result

        last_result = dict(last_result)
        last_result["attempts"] = attempts
        last_result["router"] = self.name
        return last_result


def build_shadow_router(*, api_keys: Optional[Dict[str, str]] = None, tracked_symbols: Optional[Iterable[str]] = None) -> Optional[Any]:
    api_keys = dict(api_keys or {})
    tracked_symbols = list(dict.fromkeys(str(symbol).strip().upper() for symbol in (tracked_symbols or []) if str(symbol).strip()))
    router_order = [
        name.strip().lower()
        for name in os.getenv("SHADOW_BROKER_ORDER", "upstox,mt5,binance,bybit").replace(";", ",").split(",")
        if name.strip()
    ]
    shared_dry_run = _env_flag("SHADOW_ROUTER_DRY_RUN", True)
    routers: List[Any] = []

    for router_name in router_order:
        if router_name == "upstox":
            routers.append(
                UpstoxExecutionBroker(
                    access_token=api_keys.get("upstox", os.getenv("UPSTOX_ACCESS_TOKEN", "")),
                    instrument_map_path=os.getenv("UPSTOX_INSTRUMENT_MAP_PATH", "data/upstox_instruments.json"),
                    enabled=_env_flag("UPSTOX_ORDER_ENABLED", False),
                    dry_run=_env_flag("UPSTOX_ORDER_DRY_RUN", shared_dry_run),
                    timeout_seconds=int(os.getenv("UPSTOX_ORDER_TIMEOUT_SECONDS", "8")),
                    tracked_symbols=tracked_symbols,
                )
            )
        elif router_name == "mt5":
            routers.append(
                MetaTrader5ExecutionBroker(
                    login=os.getenv("MT5_LOGIN", ""),
                    password=os.getenv("MT5_PASSWORD", ""),
                    server=os.getenv("MT5_SERVER", ""),
                    path=os.getenv("MT5_PATH", ""),
                    symbol_map_path=os.getenv("MT5_SYMBOL_MAP_PATH", "data/mt5_symbols.json"),
                    enabled=_env_flag("MT5_ENABLED", False),
                    dry_run=_env_flag("MT5_ORDER_DRY_RUN", shared_dry_run),
                    timeout_seconds=int(os.getenv("MT5_ORDER_TIMEOUT_SECONDS", "10")),
                    deviation_points=int(os.getenv("MT5_ORDER_DEVIATION_POINTS", "20")),
                )
            )
        elif router_name == "binance":
            routers.append(
                BinanceSpotExecutionBroker(
                    api_key=os.getenv("BINANCE_API_KEY", ""),
                    api_secret=os.getenv("BINANCE_API_SECRET", ""),
                    enabled=_env_flag("BINANCE_ENABLED", False),
                    dry_run=_env_flag("BINANCE_ORDER_DRY_RUN", shared_dry_run),
                    base_url=os.getenv("BINANCE_REST_BASE_URL", "https://testnet.binance.vision"),
                    timeout_seconds=int(os.getenv("BINANCE_ORDER_TIMEOUT_SECONDS", "8")),
                    recv_window_ms=int(os.getenv("BINANCE_RECV_WINDOW_MS", "5000")),
                )
            )
        elif router_name == "bybit":
            routers.append(
                BybitSpotExecutionBroker(
                    api_key=os.getenv("BYBIT_API_KEY", ""),
                    api_secret=os.getenv("BYBIT_API_SECRET", ""),
                    enabled=_env_flag("BYBIT_ENABLED", False),
                    dry_run=_env_flag("BYBIT_ORDER_DRY_RUN", shared_dry_run),
                    base_url=os.getenv("BYBIT_REST_BASE_URL", "https://api-testnet.bybit.com"),
                    timeout_seconds=int(os.getenv("BYBIT_ORDER_TIMEOUT_SECONDS", "8")),
                    recv_window_ms=int(os.getenv("BYBIT_RECV_WINDOW_MS", "5000")),
                )
            )

    active = [router for router in routers if bool(getattr(router, "configured", False))]
    if not active:
        return None
    if len(active) == 1:
        return active[0]
    return FallbackExecutionBroker(active, name="shadow_router")


# ─────────────────────────────────────────────────────────────
# Crypto broker factory — paper vs live via CRYPTO_EXECUTION_MODE
# ─────────────────────────────────────────────────────────────

def get_crypto_broker(initial_balance: float = 5000.0):
    """
    Return the correct crypto broker based on CRYPTO_EXECUTION_MODE env var.

    - "paper" (default) → CryptoPaperBroker  (unchanged existing behaviour)
    - "live"            → KuCoinLiveBroker   (real exchange execution)

    Usage in the live loop:
        from core.external_execution import get_crypto_broker
        broker = get_crypto_broker()
    """
    mode = os.getenv("CRYPTO_EXECUTION_MODE", "paper").strip().lower()

    if mode == "live":
        from core.crypto_live_broker import KuCoinLiveBroker, SandboxValidator

        # Guard: validate sandbox connectivity before first live use
        if _env_flag("KUCOIN_SANDBOX", True):
            ok, msg = SandboxValidator().validate()
            if not ok:
                logger.error(f"KuCoin sandbox validation failed: {msg}")
                logger.error("Falling back to paper broker for safety")
                from core.paper_trading_crypto import CryptoPaperBroker
                return CryptoPaperBroker(initial_balance=initial_balance)
            logger.info(f"KuCoin sandbox validation: {msg}")

        logger.info("Crypto execution mode: LIVE (KuCoin)")
        return KuCoinLiveBroker(initial_balance=initial_balance)

    # Default: paper mode — existing behaviour unchanged
    from core.paper_trading_crypto import CryptoPaperBroker
    logger.info("Crypto execution mode: PAPER")
    return CryptoPaperBroker(initial_balance=initial_balance)
