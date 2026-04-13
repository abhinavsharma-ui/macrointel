"""
Broker abstractions and shadow-routing helpers.
"""

from __future__ import annotations

import json
import logging
import os
from gzip import decompress
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

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
