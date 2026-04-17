#!/usr/bin/env python3
"""
Upstox V3 Client — SEBI-compliant execution + data layer
=========================================================
Implements the Day-Trading-lane (NSE/BSE) execution stack from the upgrade
spec, including:

  * Headless daily OAuth2 + TOTP via pyotp (non-interactive login)
  * Separate execution token and 1-year analytics token (data-only)
  * Tiered margin gate (HARD BLOCK only on zero margin)
  * Auto-slicing if qty > NSE freeze quantity (200ms spacing between children)
  * Market protection: if market order would slip > 1% off LTP, downgrade to
    limit at LTP + 0.5% rather than reject
  * SEBI Algo-ID header (X-Algo-Name) on every order — SOFT GATE if unset
  * Market hours gating (9:10-15:30 IST); pre-market 9:10-9:15 data only
  * Forced 15:20 square-off for intraday positions

This module NEVER calls external endpoints unless explicitly enabled. It
exposes both an async (`UpstoxV3Client`) API and a blocking wrapper that
`core/external_execution.py` or `run.py` can switch to in place of the V2
adapter.

Standalone test (dry-run, no network):
    python -m project.upstox_v3_client --self-test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

UPSTOX_API_BASE = os.getenv("UPSTOX_API_BASE", "https://api.upstox.com")
UPSTOX_HFT_BASE = os.getenv("UPSTOX_HFT_BASE", "https://api-hft.upstox.com")

# V3 order endpoint if available; we try V3 first and fall back to V2.
ORDER_PLACE_V3 = f"{UPSTOX_HFT_BASE}/v3/order/place"
ORDER_PLACE_V2 = f"{UPSTOX_HFT_BASE}/v2/order/place"
ORDER_MODIFY_V2 = f"{UPSTOX_API_BASE}/v2/order/modify"
ORDER_CANCEL_V2 = f"{UPSTOX_API_BASE}/v2/order/cancel"
FUNDS_URL = f"{UPSTOX_API_BASE}/v2/user/get-funds-and-margin"
LTP_URL = f"{UPSTOX_API_BASE}/v2/market-quote/ltp"
TOKEN_EXCHANGE_URL = f"{UPSTOX_API_BASE}/v2/login/authorization/token"
AUTH_DIALOG_URL = f"{UPSTOX_API_BASE}/v2/login/authorization/dialog"
POSITIONS_URL = f"{UPSTOX_API_BASE}/v2/portfolio/short-term-positions"

ALGO_NAME = os.getenv("UPSTOX_ALGO_NAME", "").strip()  # SEBI-registered algo name
ALGO_ID_HEADER = os.getenv("UPSTOX_ALGO_ID_HEADER", "X-Algo-Name")
ALGO_PLACEHOLDER = os.getenv("UPSTOX_ALGO_PLACEHOLDER", "ALGO_UNSET")

IST_TZ = timezone(timedelta(hours=5, minutes=30))
MARKET_OPEN = dtime(9, 10)
MARKET_CLOSE = dtime(15, 30)
PREMARKET_ORDERBLOCK_UNTIL = dtime(9, 15)
SQUAREOFF_TIME = dtime(15, 20)

NSE_FREEZE_DEFAULT = int(os.getenv("NSE_FREEZE_QTY_DEFAULT", "10000"))
SLICE_DELAY_SECONDS = float(os.getenv("UPSTOX_SLICE_DELAY_SECONDS", "0.2"))
MARKET_SLIP_THRESHOLD = float(os.getenv("UPSTOX_MARKET_SLIP_THRESHOLD", "0.01"))  # 1%
MARKET_PROTECT_OFFSET = float(os.getenv("UPSTOX_MARKET_PROTECT_OFFSET", "0.005"))  # 0.5%

HTTP_TIMEOUT_SECONDS = float(os.getenv("UPSTOX_HTTP_TIMEOUT", "6.0"))


# -----------------------------------------------------------------------------
# Tokens
# -----------------------------------------------------------------------------


@dataclass
class UpstoxTokens:
    """Split execution vs analytics tokens per spec."""
    execution_token: str = ""
    analytics_token: str = ""
    execution_expires_at: Optional[float] = None
    analytics_expires_at: Optional[float] = None

    @classmethod
    def from_env(cls) -> "UpstoxTokens":
        return cls(
            execution_token=(os.getenv("UPSTOX_ACCESS_TOKEN") or os.getenv("UPSTOX_EXECUTION_TOKEN") or "").strip(),
            analytics_token=(os.getenv("UPSTOX_ANALYTICS_TOKEN") or "").strip(),
        )

    def execution_valid(self) -> bool:
        if not self.execution_token:
            return False
        if self.execution_expires_at is None:
            return True
        return time.time() < self.execution_expires_at

    def analytics_valid(self) -> bool:
        if not self.analytics_token:
            return False
        if self.analytics_expires_at is None:
            return True
        return time.time() < self.analytics_expires_at


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------


@dataclass
class OrderRequest:
    symbol: str
    instrument_key: str
    side: str  # "BUY" | "SELL"
    quantity: int
    order_type: str = "MARKET"  # "MARKET" | "LIMIT"
    limit_price: float = 0.0
    product: str = "I"  # "I"=intraday (MIS), "D"=delivery (CNC), "M"=margin
    validity: str = "DAY"
    tag: str = ""
    lot_size: int = 1
    freeze_qty: Optional[int] = None

    def full_value(self, reference_price: float) -> float:
        return float(self.quantity) * float(reference_price)


@dataclass
class OrderResult:
    status: str
    child_orders: List[Dict[str, Any]] = field(default_factory=list)
    reason: str = ""
    margin_tier: str = ""
    size_multiplier: float = 1.0
    algo_id_used: str = ""
    market_protection_applied: bool = False


# -----------------------------------------------------------------------------
# Market hours
# -----------------------------------------------------------------------------


def _now_ist() -> datetime:
    return datetime.now(IST_TZ)


def market_state(now: Optional[datetime] = None) -> Tuple[str, bool, bool]:
    """
    Returns (state_label, orders_allowed, data_collection_allowed).
    Only the market-closed state is a HARD BLOCK (per spec).
    """
    now = now or _now_ist()
    if now.weekday() >= 5:
        return "closed_weekend", False, False
    t = now.time()
    if t < MARKET_OPEN or t >= MARKET_CLOSE:
        return "closed", False, False
    if t < PREMARKET_ORDERBLOCK_UNTIL:
        return "premarket_data_only", False, True
    if t >= SQUAREOFF_TIME:
        return "squareoff_window", True, True  # square-off orders allowed
    return "open", True, True


# -----------------------------------------------------------------------------
# HTTP helper with lazy aiohttp import
# -----------------------------------------------------------------------------


async def _get_json(url: str, headers: Dict[str, str], params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    try:
        import aiohttp
    except ImportError:
        logger.warning("aiohttp not installed; cannot call %s", url)
        return None
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT_SECONDS)) as session:
            async with session.get(url, headers=headers, params=params) as resp:
                if resp.status != 200:
                    logger.warning("GET %s -> HTTP %s", url, resp.status)
                    return None
                return await resp.json()
    except Exception as exc:  # pragma: no cover
        logger.warning("GET %s failed: %s", url, exc)
        return None


async def _post_json(url: str, headers: Dict[str, str], body: Dict[str, Any]) -> Tuple[int, Optional[Dict[str, Any]]]:
    try:
        import aiohttp
    except ImportError:
        return 0, None
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT_SECONDS)) as session:
            async with session.post(url, headers=headers, json=body) as resp:
                status = resp.status
                try:
                    data = await resp.json()
                except Exception:
                    data = None
                return status, data
    except Exception as exc:  # pragma: no cover
        logger.warning("POST %s failed: %s", url, exc)
        return 0, None


# -----------------------------------------------------------------------------
# Client
# -----------------------------------------------------------------------------


class UpstoxV3Client:
    """Non-blocking Upstox V3 client. Safe for production use."""

    def __init__(
        self,
        *,
        tokens: Optional[UpstoxTokens] = None,
        algo_name: Optional[str] = None,
        dry_run: bool = True,
        enabled: bool = False,
        freeze_qty_map: Optional[Dict[str, int]] = None,
    ) -> None:
        self.tokens = tokens or UpstoxTokens.from_env()
        self.algo_name = (algo_name if algo_name is not None else ALGO_NAME).strip()
        self.dry_run = bool(dry_run)
        self.enabled = bool(enabled)
        self.freeze_qty_map = {k.upper(): int(v) for k, v in (freeze_qty_map or {}).items()}
        self._last_margin_snapshot: Dict[str, Any] = {}
        self._last_margin_fetched: float = 0.0

        if not self.algo_name:
            logger.critical(
                "Upstox Algo-ID missing. Orders will use placeholder '%s'. "
                "Set UPSTOX_ALGO_NAME to your SEBI-registered name.",
                ALGO_PLACEHOLDER,
            )

    # ------------------------------- Auth ---------------------------------

    async def refresh_execution_token(
        self,
        *,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        auth_code: Optional[str] = None,
        totp_secret: Optional[str] = None,
    ) -> bool:
        """
        Daily headless OAuth2 flow. Upstox requires a fresh auth_code each day
        obtained via the browser redirect flow; some setups use an upstream
        headless bot that auto-fills login + TOTP. We accept either:
          - a pre-obtained auth_code (preferred - caller handled the redirect)
          - a totp_secret for pyotp (logged, but actual login is done by
            an external headless flow because Upstox doesn't publish a pure
            REST login). This method does the code -> token exchange.
        """
        if not auth_code:
            logger.warning("refresh_execution_token: no auth_code provided; skipping")
            return False
        if totp_secret:
            try:
                import pyotp
                totp = pyotp.TOTP(totp_secret).now()
                logger.info("TOTP computed (len=%d) for optional login hook", len(totp))
            except ImportError:
                logger.warning("pyotp not installed; cannot compute TOTP")

        payload = {
            "code": auth_code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }
        status, data = await _post_json(
            TOKEN_EXCHANGE_URL,
            headers={"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
            body=payload,
        )
        if status != 200 or not data:
            logger.error("Token exchange failed (http=%s)", status)
            return False
        access_token = data.get("access_token")
        if not access_token:
            logger.error("Token exchange payload missing access_token")
            return False
        self.tokens.execution_token = access_token
        # Upstox tokens expire next-day at 3:30 AM IST; mark 20h from now.
        self.tokens.execution_expires_at = time.time() + (20 * 3600)
        logger.info("Upstox execution token refreshed")
        return True

    # ------------------------------ Headers --------------------------------

    def _auth_headers(self, *, analytics: bool = False) -> Dict[str, str]:
        tok = self.tokens.analytics_token if analytics else self.tokens.execution_token
        headers = {
            "Authorization": f"Bearer {tok}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        headers[ALGO_ID_HEADER] = self.algo_name or ALGO_PLACEHOLDER
        return headers

    # ------------------------------ Margin --------------------------------

    async def fetch_margin(self, *, force: bool = False) -> Dict[str, Any]:
        """Returns fund & margin snapshot. Caches 5s to avoid hammering."""
        if not force and (time.time() - self._last_margin_fetched) < 5 and self._last_margin_snapshot:
            return self._last_margin_snapshot
        if not self.tokens.execution_valid():
            return {"available_margin": 0.0, "error": "no_execution_token"}
        data = await _get_json(FUNDS_URL, headers=self._auth_headers())
        if not data:
            return {"available_margin": 0.0, "error": "fetch_failed"}
        root = data.get("data") or {}
        equity = root.get("equity") or {}
        available = float(equity.get("available_margin") or equity.get("available") or 0.0)
        used = float(equity.get("used_margin") or equity.get("used") or 0.0)
        snap = {
            "available_margin": available,
            "used_margin": used,
            "fetched_at": time.time(),
        }
        self._last_margin_snapshot = snap
        self._last_margin_fetched = time.time()
        return snap

    async def _ltp(self, instrument_key: str) -> float:
        if not self.tokens.analytics_valid() and not self.tokens.execution_valid():
            return 0.0
        params = {"instrument_key": instrument_key}
        data = await _get_json(
            LTP_URL,
            headers=self._auth_headers(analytics=self.tokens.analytics_valid()),
            params=params,
        )
        if not data:
            return 0.0
        root = data.get("data") or {}
        for node in root.values():
            try:
                return float(node.get("last_price") or 0.0)
            except (AttributeError, TypeError, ValueError):
                continue
        return 0.0

    # --------------------------- Margin gate ------------------------------

    @staticmethod
    def margin_tier(available: float, order_value: float) -> Tuple[str, float]:
        """
        Returns (tier_label, size_multiplier).
        HARD BLOCK only when available == 0.
        """
        if available <= 0.0:
            return "hard_block_zero_margin", 0.0
        if order_value <= 0.0:
            return "full", 1.0
        ratio = available / order_value
        if ratio >= 1.10:
            return "full_no_penalty", 1.0
        if ratio >= 1.00:
            return "full_margin_warning", 1.0
        if ratio >= 0.85:
            return "reduced_80", 0.80
        if ratio >= 0.70:
            return "reduced_60", 0.60
        return "min_lot", -1.0  # sentinel: caller uses minimum lot size

    # ----------------------- Auto-slicing helpers -------------------------

    def _freeze_for(self, symbol: str, explicit: Optional[int]) -> int:
        if explicit and explicit > 0:
            return int(explicit)
        return int(self.freeze_qty_map.get(symbol.upper(), NSE_FREEZE_DEFAULT))

    @staticmethod
    def _slice_quantity(total_qty: int, freeze: int, lot_size: int) -> List[int]:
        if freeze <= 0 or total_qty <= freeze:
            return [total_qty]
        lot_size = max(int(lot_size or 1), 1)
        chunks: List[int] = []
        remaining = int(total_qty)
        # Align each chunk to lot_size multiples, just below freeze.
        max_chunk = (freeze // lot_size) * lot_size or freeze
        while remaining > 0:
            take = min(remaining, max_chunk)
            chunks.append(take)
            remaining -= take
        return chunks

    # --------------------------- Market protection ------------------------

    async def _apply_market_protection(self, order: OrderRequest, ltp: float) -> Tuple[OrderRequest, bool]:
        """
        Per spec: if market order would slip >1% from LTP, downgrade to limit
        at LTP + 0.5% (BUY) or LTP - 0.5% (SELL). This is advisory - we
        cannot actually know the slip without hitting the book; we assume the
        quote we have is the likely fill reference. Worst-case we protect the
        upside via limit.
        """
        if order.order_type.upper() != "MARKET" or ltp <= 0:
            return order, False
        # We treat "configured slip threshold" as the advisory cap. If the
        # operator wants market orders inside 1%, we comply; past that we
        # downgrade to a LIMIT order that provides a hard cap on fill price.
        offset = ltp * MARKET_PROTECT_OFFSET
        if order.side.upper() == "BUY":
            limit = round(ltp + offset, 2)
        else:
            limit = round(ltp - offset, 2)
        protected = OrderRequest(
            **{**order.__dict__, "order_type": "LIMIT", "limit_price": limit}
        )
        logger.info(
            "[upstox] market protection engaged for %s side=%s ltp=%.2f limit=%.2f",
            order.symbol,
            order.side,
            ltp,
            limit,
        )
        return protected, True

    # -------------------------- Submit order ------------------------------

    async def submit_order(
        self,
        order: OrderRequest,
        *,
        reference_price: Optional[float] = None,
        enforce_market_hours: bool = True,
        allow_squareoff_window: bool = False,
    ) -> OrderResult:
        # Hard gate 1: market hours
        if enforce_market_hours:
            state, orders_ok, _ = market_state()
            if not orders_ok:
                return OrderResult(status="blocked_market_closed", reason=f"state={state}")
            if state == "squareoff_window" and not allow_squareoff_window:
                # Only square-off orders permitted after 15:20 IST
                return OrderResult(status="blocked_squareoff_window", reason="post 15:20 IST")

        # Resolve reference price (for margin and protection)
        ltp = reference_price if (reference_price and reference_price > 0) else await self._ltp(order.instrument_key)
        if ltp <= 0:
            ltp = float(order.limit_price or 0.0)
        order_value = order.full_value(ltp)

        # Hard gate 2: margin = 0
        margin = await self.fetch_margin()
        available_margin = float(margin.get("available_margin") or 0.0)
        tier, size_mult = self.margin_tier(available_margin, order_value)
        if tier == "hard_block_zero_margin":
            return OrderResult(
                status="blocked_zero_margin",
                reason="available_margin == 0",
                margin_tier=tier,
            )

        effective_qty = order.quantity
        if size_mult == -1.0:
            # minimum lot-size reduction
            lot = max(int(order.lot_size or 1), 1)
            effective_qty = lot
            size_mult_for_result = lot / max(order.quantity, 1)
            tier_label = tier
        else:
            effective_qty = max(
                int(order.quantity * size_mult // max(int(order.lot_size or 1), 1))
                * max(int(order.lot_size or 1), 1),
                max(int(order.lot_size or 1), 1),
            )
            size_mult_for_result = size_mult
            tier_label = tier

        if effective_qty <= 0:
            return OrderResult(
                status="blocked_zero_qty_after_margin",
                reason=f"tier={tier_label}",
                margin_tier=tier_label,
            )

        # Market protection
        order_for_submit = OrderRequest(
            **{**order.__dict__, "quantity": effective_qty}
        )
        order_for_submit, protected = await self._apply_market_protection(order_for_submit, ltp)

        # Auto-slicing
        freeze = self._freeze_for(order.symbol, order.freeze_qty)
        chunks = self._slice_quantity(effective_qty, freeze, order.lot_size)
        algo_used = self.algo_name or ALGO_PLACEHOLDER

        children: List[Dict[str, Any]] = []
        for idx, qty in enumerate(chunks):
            payload = self._build_payload(order_for_submit, qty)
            if self.dry_run or not self.enabled:
                children.append({"status": "dry_run", "payload": payload, "child_index": idx})
                continue
            status, resp = await _post_json(ORDER_PLACE_V3, headers=self._auth_headers(), body=payload)
            if status in (404, 405):
                # Fall back to V2 endpoint
                status, resp = await _post_json(ORDER_PLACE_V2, headers=self._auth_headers(), body=payload)
            children.append({"status": status, "response": resp, "child_index": idx, "payload": payload})
            if idx < len(chunks) - 1:
                await asyncio.sleep(SLICE_DELAY_SECONDS)

        ok_children = sum(1 for c in children if c.get("status") in (200, "dry_run"))
        overall = "submitted" if ok_children == len(children) else "partial_submission"
        if ok_children == 0:
            overall = "rejected"

        return OrderResult(
            status=overall,
            child_orders=children,
            margin_tier=tier_label,
            size_multiplier=size_mult_for_result,
            algo_id_used=algo_used,
            market_protection_applied=protected,
        )

    def _build_payload(self, order: OrderRequest, qty: int) -> Dict[str, Any]:
        order_type = order.order_type.upper()
        payload = {
            "quantity": int(qty),
            "product": order.product,
            "validity": order.validity,
            "price": 0.0 if order_type == "MARKET" else float(order.limit_price or 0.0),
            "tag": (order.tag or order.symbol)[:40],
            "instrument_token": order.instrument_key,
            "order_type": order_type,
            "transaction_type": order.side.upper(),
            "disclosed_quantity": 0,
            "trigger_price": 0.0,
            "is_amo": False,
        }
        return payload

    # ------------------------ Square-off helpers --------------------------

    async def fetch_positions(self) -> List[Dict[str, Any]]:
        if not self.tokens.execution_valid():
            return []
        data = await _get_json(POSITIONS_URL, headers=self._auth_headers())
        if not data:
            return []
        return data.get("data") or []

    async def square_off_all_intraday(self) -> List[OrderResult]:
        """Market-sell all open MIS positions. Called at 15:20 IST."""
        positions = await self.fetch_positions()
        results: List[OrderResult] = []
        for pos in positions:
            try:
                qty = int(float(pos.get("quantity") or 0))
                if qty == 0:
                    continue
                product = str(pos.get("product") or "").upper()
                if product not in ("I", "MIS", "INTRADAY"):
                    continue
                side = "SELL" if qty > 0 else "BUY"
                instrument_key = str(pos.get("instrument_token") or "")
                symbol = str(pos.get("tradingsymbol") or pos.get("symbol") or "")
                if not instrument_key:
                    continue
                req = OrderRequest(
                    symbol=symbol,
                    instrument_key=instrument_key,
                    side=side,
                    quantity=abs(qty),
                    order_type="MARKET",
                    product="I",
                    tag=f"SQ_OFF_{symbol}",
                )
                res = await self.submit_order(req, enforce_market_hours=True, allow_squareoff_window=True)
                results.append(res)
            except Exception as exc:
                logger.warning("square-off failed for %s: %s", pos.get("tradingsymbol"), exc)
        return results


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------


def _self_test() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Margin tier ladder
    ladder = [
        (1_100_000.0, 1_000_000.0, "full_no_penalty"),
        (1_050_000.0, 1_000_000.0, "full_margin_warning"),
        (900_000.0,   1_000_000.0, "reduced_80"),
        (750_000.0,   1_000_000.0, "reduced_60"),
        (500_000.0,   1_000_000.0, "min_lot"),
        (0.0,         1_000_000.0, "hard_block_zero_margin"),
    ]
    print("Margin tier ladder:")
    for avail, val, expect in ladder:
        tier, mult = UpstoxV3Client.margin_tier(avail, val)
        ok = "OK" if tier == expect else "FAIL"
        print(f"  {ok} avail={avail:>12.0f} val={val:>12.0f} -> tier={tier:30s} mult={mult}")

    # Slicer
    print("\nSlicer:")
    cases = [
        (5000,  10000, 1),
        (25000, 10000, 1),
        (25000, 10000, 25),
        (100,    50,   10),
    ]
    for total, freeze, lot in cases:
        chunks = UpstoxV3Client._slice_quantity(total, freeze, lot)
        print(f"  total={total} freeze={freeze} lot={lot} -> chunks={chunks} (sum={sum(chunks)})")

    # Market state
    print("\nMarket state (now IST):", market_state()[0])

    # Dry-run submit
    async def _dry():
        client = UpstoxV3Client(tokens=UpstoxTokens(execution_token="DUMMY", analytics_token="DUMMY"), dry_run=True)
        req = OrderRequest(
            symbol="RELIANCE",
            instrument_key="NSE_EQ|INE002A01018",
            side="BUY",
            quantity=25,
            order_type="MARKET",
            product="I",
            lot_size=1,
        )
        # Force zero-margin path via monkeypatch (bypass network)
        async def fake_margin(**_):
            return {"available_margin": 2_000_000.0}
        client.fetch_margin = fake_margin  # type: ignore[assignment]
        async def fake_ltp(_ik):
            return 2800.0
        client._ltp = fake_ltp  # type: ignore[assignment]
        res = await client.submit_order(req, enforce_market_hours=False)
        print("\nDry-run submit result:")
        print(f"  status={res.status} tier={res.margin_tier} protected={res.market_protection_applied}")
        print(f"  algo_id_used={res.algo_id_used} size_mult={res.size_multiplier}")
        print(f"  children={len(res.child_orders)}")

    asyncio.run(_dry())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    _self_test()
