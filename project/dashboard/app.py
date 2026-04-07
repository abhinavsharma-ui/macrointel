"""
Flask-SocketIO Dashboard - Fully Wired
========================================
This is the production version. Zero demo mode.

Every API route returns real data:
  /api/signals      â†’ live signals from signal_engine_v2 (MultiFactorScorer)
  /api/portfolio    â†’ live paper broker P&L and positions
  /api/stress-test  â†’ regime-aware stress test results
  /api/prices       â†’ latest tick per symbol from price buffer
  /api/latency      â†’ WebSocket latency stats
  /api/positions    â†’ open positions with current P&L
  /api/diagnostics  â†’ signal quality report (detects 38%-win-rate problems)

SocketIO push (1 Hz):
  prices_update     â†’ latest price per active symbol
  new_signal        â†’ fired immediately when a new signal is generated
  portfolio_update  â†’ P&L, cash, drawdown after every price tick
  latency_update    â†’ latency stats every 10s

Standalone launch:
  python dashboard/app.py
  Boots the FULL system: data pipeline + signal engine + paper broker + dashboard.
  Not demo mode. Requires API keys in .env or environment.
"""

import csv
import json
import logging
import os
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit

logger = logging.getLogger(__name__)
_startup_time = time.time()
_dashboard_signal_limit = max(0, int(os.getenv("DASHBOARD_SIGNAL_LIMIT", "500")))
_dashboard_price_limit = max(0, int(os.getenv("DASHBOARD_PRICE_LIMIT", "500")))
_dashboard_positions_limit = max(1, int(os.getenv("DASHBOARD_POSITIONS_LIMIT", "500")))
_dashboard_push_interval = max(1.0, float(os.getenv("DASHBOARD_PUSH_INTERVAL_SECONDS", "2.0")))
_dashboard_execution_trace_limit = max(25, int(os.getenv("DASHBOARD_EXECUTION_TRACE_LIMIT", "300")))
_dashboard_target_open_positions = max(
    1,
    int(os.getenv("DAY_TRADE_MAX_OPEN_POSITIONS", os.getenv("AUTO_TRADE_MAX_OPEN_POSITIONS", "12"))),
)
_live_portfolio_path = Path(os.getenv("LIVE_PORTFOLIO_PATH", str(ROOT / "data" / "live_portfolio.json")))
_live_portfolio_csv_path = Path(os.getenv("LIVE_PORTFOLIO_CSV_PATH", str(ROOT / "data" / "live_portfolio.csv")))


class ManualPortfolioClient:
    """Brokerless portfolio loader for users without API/KYC access."""

    def __init__(self, source_path: Path, csv_path: Path):
        self.source_path = source_path
        self.csv_path = csv_path

    def _empty_snapshot(self) -> Dict[str, Any]:
        preferred = self.csv_path.name if self.csv_path else self.source_path.name
        return {
            "configured": False,
            "broker": "Manual import",
            "summary": {
                "portfolio_value": None,
                "cash": None,
                "holdings_value": None,
                "positions_value": None,
                "open_positions": 0,
                "day_pnl": None,
            },
            "positions": [],
            "total_positions": 0,
            "note": f"Add your broker export to {preferred} to show your live brokerage portfolio",
        }

    def _placeholder_snapshot(self, note: str) -> Dict[str, Any]:
        payload = self._empty_snapshot()
        payload["note"] = note
        return payload

    @staticmethod
    def _pick(row: Dict[str, Any], *names: str) -> Any:
        lowered = {str(k).strip().lower(): v for k, v in row.items()}
        for name in names:
            value = lowered.get(name.lower())
            if value not in (None, ""):
                return value
        return None

    @staticmethod
    def _num(value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        text = str(value).strip().replace(",", "").replace("$", "")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _load_json(self) -> Dict[str, Any]:
        payload = json.loads(self.source_path.read_text(encoding="utf-8"))
        note = str(payload.get("note") or "").strip().lower()
        summary = dict(payload.get("summary") or {})
        if (
            not (payload.get("positions") or [])
            and (
                "replace this template" in note
                or (
                    str(payload.get("broker") or "").strip().lower() == "manual import"
                    and all(
                        self._num(summary.get(field)) in (None, 0.0)
                        for field in ("portfolio_value", "cash", "holdings_value", "positions_value", "day_pnl")
                    )
                )
            )
        ):
            return self._placeholder_snapshot(
                f"Ignoring bundled template {self.source_path.name}. Replace it with a real broker export to show live brokerage data."
            )
        positions = list(payload.get("positions") or [])
        positions.sort(key=lambda p: abs(p.get("unrealized_pnl") or 0), reverse=True)
        summary = dict(payload.get("summary") or {})
        if "open_positions" not in summary:
            summary["open_positions"] = len(positions)
        return {
            "configured": True,
            "broker": payload.get("broker") or "Manual import",
            "summary": {
                "portfolio_value": summary.get("portfolio_value"),
                "cash": summary.get("cash"),
                "holdings_value": summary.get("holdings_value"),
                "positions_value": summary.get("positions_value"),
                "open_positions": summary.get("open_positions"),
                "day_pnl": summary.get("day_pnl"),
            },
            "positions": positions[:_dashboard_positions_limit],
            "total_positions": len(positions),
            "updated_at": payload.get("updated_at"),
            "note": payload.get("note") or f"Loaded from {self.source_path.name}",
        }

    def _load_csv(self) -> Dict[str, Any]:
        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return self._empty_snapshot()
        brokers = {
            str(self._pick(row, "broker", "account", "platform") or "").strip().lower()
            for row in rows
            if any(str(value or "").strip() for value in row.values())
        }
        symbols = {
            str(self._pick(row, "symbol", "ticker", "tradingsymbol", "trading_symbol", "instrument") or "").strip().upper()
            for row in rows
            if any(str(value or "").strip() for value in row.values())
        }
        if brokers and brokers <= {"samplebroker"}:
            return self._placeholder_snapshot(
                f"Ignoring bundled sample file {self.csv_path.name}. Replace it with your real broker export to show live brokerage data."
            )
        if symbols and symbols <= {"AAPL", "MSFT"} and brokers <= {"", "samplebroker"}:
            return self._placeholder_snapshot(
                f"Ignoring placeholder holdings in {self.csv_path.name}. Replace them with your real broker export to show live brokerage data."
            )

        positions: List[Dict[str, Any]] = []
        cash = None
        broker = "Manual CSV import"

        for row in rows:
            symbol = self._pick(row, "symbol", "ticker", "tradingsymbol", "trading_symbol", "instrument")
            quantity = self._num(self._pick(row, "quantity", "qty", "shares"))
            if not symbol or quantity in (None, 0):
                continue

            avg_cost = self._num(self._pick(row, "avg_cost", "average_price", "avg price", "cost", "cost_price")) or 0.0
            current_price = self._num(self._pick(row, "current_price", "ltp", "last_price", "price", "market_price"))
            unrealized_pnl = self._num(self._pick(row, "unrealized_pnl", "pnl", "profit_loss", "profit/loss"))
            source = str(self._pick(row, "source", "type", "segment") or "holding")
            row_cash = self._num(self._pick(row, "cash", "available_cash", "available_margin"))
            row_broker = self._pick(row, "broker", "account", "platform")

            if row_cash is not None and cash is None:
                cash = row_cash
            if row_broker:
                broker = str(row_broker)
            if unrealized_pnl is None and current_price is not None:
                unrealized_pnl = round((current_price - avg_cost) * quantity, 2)

            positions.append({
                "symbol": str(symbol).strip(),
                "quantity": int(quantity) if float(quantity).is_integer() else quantity,
                "avg_cost": round(avg_cost, 4),
                "current_price": current_price,
                "unrealized_pnl": unrealized_pnl or 0.0,
                "source": source,
                "value": round((current_price or avg_cost) * quantity, 2),
            })

        positions.sort(key=lambda p: abs(p.get("unrealized_pnl") or 0), reverse=True)
        holdings_value = round(sum((p.get("value") or 0) for p in positions), 2)
        day_pnl = round(sum((p.get("unrealized_pnl") or 0) for p in positions), 2)
        portfolio_value = round(holdings_value + (cash or 0), 2)

        return {
            "configured": True,
            "broker": broker,
            "summary": {
                "portfolio_value": portfolio_value,
                "cash": round(cash, 2) if cash is not None else None,
                "holdings_value": holdings_value,
                "positions_value": 0.0,
                "open_positions": len(positions),
                "day_pnl": day_pnl,
            },
            "positions": positions[:_dashboard_positions_limit],
            "total_positions": len(positions),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "note": f"Loaded from {self.csv_path.name}",
        }

    def get_snapshot(self) -> Dict[str, Any]:
        if self.csv_path.exists():
            return self._load_csv()
        if self.source_path.exists():
            return self._load_json()
        return self._empty_snapshot()


class UpstoxPortfolioClient:
    """Read-only live portfolio adapter for Upstox."""

    HOLDINGS_URL = "https://api.upstox.com/v2/portfolio/long-term-holdings"
    POSITIONS_URL = "https://api.upstox.com/v2/portfolio/short-term-positions"
    FUNDS_URL = "https://api.upstox.com/v2/user/get-funds-and-margin"

    def __init__(self, access_token: str, timeout_seconds: int = 8, cache_ttl_seconds: int = 10):
        self.access_token = (access_token or "").strip()
        self.timeout_seconds = timeout_seconds
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache_expires_at = 0.0
        self._cache_payload: Dict[str, Any] = self._not_configured()

    def _headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.access_token}",
        }

    def _not_configured(self) -> Dict[str, Any]:
        return {
            "configured": False,
            "broker": "Upstox",
            "summary": {
                "portfolio_value": None,
                "cash": None,
                "holdings_value": None,
                "positions_value": None,
                "open_positions": 0,
                "day_pnl": None,
            },
            "positions": [],
            "total_positions": 0,
            "note": "UPSTOX_ACCESS_TOKEN not configured",
        }

    def _request(self, url: str) -> Any:
        response = requests.get(url, headers=self._headers(), timeout=self.timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "success":
            raise RuntimeError(payload.get("message") or f"Unexpected Upstox response from {url}")
        return payload.get("data")

    def _fetch_snapshot(self) -> Dict[str, Any]:
        if not self.access_token:
            return self._not_configured()

        holdings = self._request(self.HOLDINGS_URL) or []
        positions = self._request(self.POSITIONS_URL) or []
        funds = self._request(self.FUNDS_URL) or {}

        live_positions: List[Dict[str, Any]] = []

        for item in holdings:
            quantity = item.get("quantity") or 0
            live_positions.append({
                "symbol": item.get("trading_symbol") or item.get("tradingsymbol") or item.get("instrument_token") or "UNKNOWN",
                "quantity": quantity,
                "avg_cost": item.get("average_price") or 0,
                "current_price": item.get("last_price"),
                "unrealized_pnl": item.get("pnl") or 0,
                "source": "holding",
                "value": (item.get("last_price") or 0) * quantity,
            })

        for item in positions:
            quantity = item.get("quantity") or 0
            live_positions.append({
                "symbol": item.get("trading_symbol") or item.get("tradingsymbol") or item.get("instrument_token") or "UNKNOWN",
                "quantity": quantity,
                "avg_cost": item.get("average_price") or item.get("buy_price") or 0,
                "current_price": item.get("last_price"),
                "unrealized_pnl": item.get("pnl") if item.get("pnl") is not None else item.get("unrealised") or 0,
                "source": "position",
                "value": item.get("value") if item.get("value") is not None else (item.get("last_price") or 0) * quantity,
            })

        live_positions.sort(key=lambda p: abs(p.get("unrealized_pnl") or 0), reverse=True)

        holdings_value = sum((p.get("value") or 0) for p in live_positions if p.get("source") == "holding")
        positions_value = sum((p.get("value") or 0) for p in live_positions if p.get("source") == "position")
        day_pnl = sum((p.get("unrealized_pnl") or 0) for p in live_positions)

        equity_funds = funds.get("equity") if isinstance(funds, dict) else {}
        cash = (equity_funds or {}).get("available_margin")
        portfolio_value = holdings_value + positions_value + (cash or 0)

        return {
            "configured": True,
            "broker": "Upstox",
            "summary": {
                "portfolio_value": round(portfolio_value, 2),
                "cash": round(cash, 2) if cash is not None else None,
                "holdings_value": round(holdings_value, 2),
                "positions_value": round(positions_value, 2),
                "open_positions": len(live_positions),
                "day_pnl": round(day_pnl, 2),
            },
            "positions": live_positions[:_dashboard_positions_limit],
            "total_positions": len(live_positions),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def get_snapshot(self) -> Dict[str, Any]:
        now = time.time()
        if now < self._cache_expires_at:
            return self._cache_payload
        try:
            self._cache_payload = self._fetch_snapshot()
        except Exception as exc:
            logger.warning(f"Upstox portfolio fetch failed: {exc}")
            self._cache_payload = {
                "configured": bool(self.access_token),
                "broker": "Upstox",
                "summary": {
                    "portfolio_value": None,
                    "cash": None,
                    "holdings_value": None,
                    "positions_value": None,
                    "open_positions": 0,
                    "day_pnl": None,
                },
                "positions": [],
                "total_positions": 0,
                "error": str(exc),
            }
        self._cache_expires_at = now + self.cache_ttl_seconds
        return self._cache_payload


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# App factory
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def create_app(
    price_buffer=None,
    paper_broker=None,
    signal_store: Optional[Dict] = None,
    stress_results: Optional[Dict] = None,
    execution_trace: Optional[List[Dict]] = None,
    execution_backtest: Optional[Dict] = None,
    execution_reconciliation: Optional[List[Dict]] = None,
    portfolio_overlay: Optional[Dict] = None,
    meta_model_status: Optional[Dict] = None,
    learning_status: Optional[Dict] = None,
    latency_monitor=None,
    security_suite=None,
) -> tuple:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "macro-intel-prod-2024")
    CORS(app)
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading",
                        logger=False, engineio_logger=False)

    _signal_store   = signal_store   if signal_store   is not None else {}
    _stress_results = stress_results if stress_results is not None else {}
    _execution_trace = execution_trace if execution_trace is not None else []
    _execution_backtest = execution_backtest if execution_backtest is not None else {}
    _execution_reconciliation = execution_reconciliation if execution_reconciliation is not None else []
    _portfolio_overlay = portfolio_overlay if portfolio_overlay is not None else {}
    _meta_model_status = meta_model_status if meta_model_status is not None else {}
    _learning_status = learning_status if learning_status is not None else {}
    _client_count   = [0]
    live_broker = UpstoxPortfolioClient(os.environ.get("UPSTOX_ACCESS_TOKEN", ""))
    manual_broker = ManualPortfolioClient(_live_portfolio_path, _live_portfolio_csv_path)

    def _limit_values(values, limit: int):
        values = list(values)
        return values if limit <= 0 else values[:limit]

    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value in (None, ""):
                return default
            return float(value)
        except Exception:
            return default

    def _to_datetime(value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value))
        except Exception:
            return None

    def _infer_lane(symbol: str, payload: Optional[Dict] = None, metadata: Optional[Dict] = None) -> str:
        payload = payload or {}
        metadata = metadata or {}
        for source in (metadata, payload):
            lane = source.get("lane")
            if lane:
                return str(lane)
            signal_key = source.get("signal_key") or source.get("position_key")
            if signal_key and "::" in str(signal_key):
                return str(signal_key).rsplit("::", 1)[-1].lower()
        if "::" in str(symbol):
            return str(symbol).rsplit("::", 1)[-1].lower()
        signal_style = str(payload.get("signal_style", "") or metadata.get("signal_style", "")).lower()
        if signal_style == "crypto_depth_intraday" or str(symbol).upper().endswith(("USDT", "USDC", "BUSD")):
            return "crypto"
        if signal_style == "day_trade_intraday":
            return "day"
        return "normal"

    def _lane_label(lane: str) -> str:
        return {
            "normal": "Normal Trading",
            "day": "Day Trading",
            "crypto": "Crypto Scalper",
        }.get(str(lane or "").lower(), "Normal Trading")

    _dashboard_meta_engine = [None]

    def _get_dashboard_meta_engine():
        if _dashboard_meta_engine[0] is None:
            from core.signal_engine_v2 import MetaDecisionEngine

            _dashboard_meta_engine[0] = MetaDecisionEngine()
        return _dashboard_meta_engine[0]

    def _latest_price_payload(symbol: str) -> Dict[str, Any]:
        if not price_buffer:
            return {}
        tick = price_buffer.latest(symbol)
        if not tick:
            return {}
        recent = price_buffer.recent_ticks(symbol, n=60)
        prev_price = recent[0].price if len(recent) > 1 else tick.price
        change_pct = round((tick.price / prev_price - 1) * 100, 3) if prev_price else 0.0
        return {
            "last_price": round(float(tick.price), 4),
            "price_change_pct": change_pct,
            "price_direction": "up" if change_pct > 0 else "down" if change_pct < 0 else "flat",
            "price_timestamp": tick.timestamp.isoformat(),
            "market": getattr(tick, "market", None),
            "volume": getattr(tick, "volume", None),
        }

    def _enrich_signal_row(row: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(row or {})
        symbol = str(payload.get("symbol") or "")
        if not symbol:
            return payload

        lane = _infer_lane(symbol, payload=payload)
        payload.setdefault("lane", lane)
        payload.setdefault("lane_label", _lane_label(lane))
        payload.setdefault("signal_key", payload.get("signal_key") or f"{symbol}::{lane}")
        payload.setdefault("scenario", payload.get("lane_label"))

        meta = payload.get("meta_decision")
        if not (isinstance(meta, dict) and "take_trade" in meta):
            try:
                evaluated = _get_dashboard_meta_engine().evaluate_universe({symbol: payload}).get(symbol, {})
            except Exception as exc:
                logger.debug(f"Dashboard meta hydration failed for {symbol}: {exc}")
                evaluated = {}
            if evaluated:
                payload["meta_decision"] = evaluated
                payload["trade_eligible"] = evaluated.get("take_trade", False)
                payload["take_probability"] = evaluated.get("take_probability", 0.0)
                payload["skip_probability"] = evaluated.get("skip_probability", 1.0)
                payload["expected_edge_pct"] = evaluated.get("expected_edge_pct", 0.0)
                payload["expected_drawdown_pct"] = evaluated.get("expected_drawdown_pct", 0.0)
                payload["rank_score"] = evaluated.get("rank_score", 0.0)
                payload["rank_percentile"] = evaluated.get("rank_percentile", 0.0)
                payload["size_multiplier"] = evaluated.get("size_multiplier", 1.0)
                payload["meta_source"] = evaluated.get("source", "heuristic")

        allocations = (_portfolio_overlay.get("allocations", {}) or {}) if isinstance(_portfolio_overlay, dict) else {}
        allocation = allocations.get(symbol, {})
        if allocation and not isinstance(payload.get("portfolio_construction"), dict):
            payload["portfolio_construction"] = allocation
            payload["portfolio_overlay_summary"] = (_portfolio_overlay.get("summary", {}) or {}) if isinstance(_portfolio_overlay, dict) else {}
            payload["target_weight"] = allocation.get("target_weight", 0.0)
            payload["target_position_pct"] = allocation.get("target_position_pct", 0.0)
            payload["residual_alpha_score"] = allocation.get("residual_alpha_score", 0.0)
            payload["beta_exposure"] = allocation.get("beta", 0.0)
            payload["portfolio_score"] = allocation.get("portfolio_score", 0.0)

        price_payload = _latest_price_payload(symbol)
        if price_payload:
            payload.update({k: v for k, v in price_payload.items() if payload.get(k) in (None, "")})

        return payload

    def _reprice_portfolio_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(snapshot or {})
        positions = []
        repriced = 0
        for raw in list(payload.get("positions") or []):
            row = dict(raw or {})
            symbol = str(row.get("symbol") or "")
            quote = _latest_price_payload(symbol) if symbol else {}
            price = quote.get("last_price")
            quantity = _safe_float(row.get("quantity"), 0.0)
            avg_cost = _safe_float(row.get("avg_cost"), price or 0.0)
            if price is not None:
                row["current_price"] = price
                row["value"] = round(price * quantity, 2)
                row["unrealized_pnl"] = round((price - avg_cost) * quantity, 2)
                row["price_change_pct"] = quote.get("price_change_pct", 0.0)
                repriced += 1
            positions.append(row)

        summary = dict(payload.get("summary") or {})
        holdings_value = round(sum(_safe_float(row.get("value"), 0.0) for row in positions), 2)
        day_pnl = round(sum(_safe_float(row.get("unrealized_pnl"), 0.0) for row in positions), 2)
        cash = summary.get("cash")
        if holdings_value or cash is not None:
            summary["holdings_value"] = holdings_value
            summary["day_pnl"] = day_pnl
            summary["open_positions"] = len(positions)
            summary["portfolio_value"] = round(holdings_value + (_safe_float(cash, 0.0) if cash is not None else 0.0), 2)
        payload["summary"] = summary
        payload["positions"] = positions[:_dashboard_positions_limit]
        payload["total_positions"] = len(positions)
        if repriced:
            note = str(payload.get("note") or "").strip()
            payload["note"] = f"{note} • repriced from live market feed".strip(" •")
            payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        return payload

    def _reprice_paper_broker_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(summary or {})
        repriced_positions: Dict[str, Dict[str, Any]] = {}
        holdings_value = 0.0
        total_unrealized = 0.0
        repriced = 0

        for position_key, raw in dict(payload.get("positions") or {}).items():
            row = dict(raw or {})
            symbol = str(row.get("symbol") or position_key)
            quantity = _safe_float(row.get("quantity"), 0.0)
            avg_cost = _safe_float(row.get("avg_cost"), 0.0)
            quote = _latest_price_payload(symbol) if symbol else {}
            current_price = quote.get("last_price")
            if current_price is None:
                if quantity:
                    current_price = avg_cost + (_safe_float(row.get("unrealized_pnl"), 0.0) / quantity)
                else:
                    current_price = avg_cost
            else:
                repriced += 1
                row["price_change_pct"] = quote.get("price_change_pct", 0.0)
                row["price_direction"] = quote.get("price_direction", "flat")
                row["price_timestamp"] = quote.get("price_timestamp")
            row["current_price"] = round(_safe_float(current_price, avg_cost), 4)
            row["unrealized_pnl"] = round((row["current_price"] - avg_cost) * quantity, 2)
            row["value"] = round(row["current_price"] * quantity, 2)
            repriced_positions[str(position_key)] = row
            holdings_value += row["value"]
            total_unrealized += row["unrealized_pnl"]

        cash = _safe_float(payload.get("cash"), 0.0)
        payload["positions"] = repriced_positions
        payload["open_positions"] = sum(1 for row in repriced_positions.values() if _safe_float(row.get("quantity"), 0.0) != 0)
        payload["holdings_value"] = round(holdings_value, 2)
        payload["unrealized_pnl"] = round(total_unrealized, 2)
        payload["day_pnl"] = round(total_unrealized, 2)
        payload["current_portfolio_value"] = round(cash + holdings_value, 2)
        payload["portfolio_value"] = payload["current_portfolio_value"]
        if repriced:
            payload["last_repriced_at"] = datetime.now(timezone.utc).isoformat()
        payload["return_periods"] = _build_return_periods(payload)
        return payload

    def _build_return_periods(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        payload = dict(summary or {})
        now = datetime.now(timezone.utc)
        current_value = _safe_float(
            payload.get("current_portfolio_value", payload.get("portfolio_value", payload.get("initial_capital", 0.0))),
            0.0,
        )
        initial_capital = _safe_float(payload.get("initial_capital"), current_value)
        rows: List[Dict[str, Any]] = []
        if paper_broker is not None:
            for trade in list(getattr(paper_broker, "trade_log", []) or []):
                filled_at = _to_datetime(trade.get("filled_at"))
                portfolio_value = _safe_float(trade.get("portfolio_value"), None)
                if filled_at is None or portfolio_value is None:
                    continue
                rows.append({"timestamp": filled_at, "portfolio_value": portfolio_value})
        rows.sort(key=lambda item: item["timestamp"])

        def _baseline_for_cutoff(cutoff: datetime) -> tuple[float, Optional[datetime], str]:
            baseline_value = initial_capital
            baseline_time = None
            basis = "initial capital"
            for row in rows:
                if row["timestamp"] <= cutoff:
                    baseline_value = row["portfolio_value"]
                    baseline_time = row["timestamp"]
                    basis = "portfolio snapshot"
                else:
                    break
            return baseline_value, baseline_time, basis

        periods = [
            ("1D", 1),
            ("7D", 7),
            ("30D", 30),
            ("90D", 90),
        ]
        output: List[Dict[str, Any]] = []
        for label, days in periods:
            cutoff = now - timedelta(days=days)
            baseline_value, baseline_time, basis = _baseline_for_cutoff(cutoff)
            actual_days = days if baseline_time is not None else max(
                0,
                int(round((now - rows[0]["timestamp"]).total_seconds() / 86400.0)) if rows else 0,
            )
            ret_pct = ((current_value / max(baseline_value, 1e-9)) - 1.0) * 100.0 if baseline_value > 0 else 0.0
            output.append(
                {
                    "label": label,
                    "days": days,
                    "actual_days": actual_days,
                    "return_pct": round(ret_pct, 2),
                    "current_value": round(current_value, 2),
                    "baseline_value": round(baseline_value, 2),
                    "baseline_at": baseline_time.isoformat() if baseline_time else None,
                    "basis": basis,
                }
            )

        inception_days = max(
            0,
            int(round((now - rows[0]["timestamp"]).total_seconds() / 86400.0)) if rows else 0,
        )
        inception_return = ((current_value / max(initial_capital, 1e-9)) - 1.0) * 100.0 if initial_capital > 0 else 0.0
        output.append(
            {
                "label": "Since Start",
                "days": inception_days,
                "actual_days": inception_days,
                "return_pct": round(inception_return, 2),
                "current_value": round(current_value, 2),
                "baseline_value": round(initial_capital, 2),
                "baseline_at": None,
                "basis": "initial capital",
            }
        )
        return output

    def _flatten_signal_store() -> List[Dict]:
        rows: List[Dict] = []
        for raw in _signal_store.values():
            if not isinstance(raw, dict):
                continue
            symbol = str(raw.get("symbol") or "")
            if not symbol:
                continue
            actual = _enrich_signal_row(dict(raw))
            rows.append(actual)
            normal_lane = raw.get("normal_lane_signal")
            if isinstance(normal_lane, dict):
                variant = _enrich_signal_row(dict(normal_lane))
                variant["symbol"] = symbol
                variant["lane"] = "normal"
                variant["lane_label"] = _lane_label("normal")
                variant["signal_key"] = variant.get("signal_key") or f"{symbol}::normal"
                variant["scenario"] = variant.get("lane_label")
                rows.append(variant)
        return rows

    def _today_trade_rows() -> List[Dict]:
        if paper_broker is None:
            return []
        today = datetime.now(timezone.utc).date()
        rows = []
        for trade in list(getattr(paper_broker, "trade_log", []) or []):
            filled_at = _to_datetime(trade.get("filled_at"))
            if not filled_at or filled_at.astimezone(timezone.utc).date() != today:
                continue
            rows.append(trade)
        return rows

    def _build_daily_report() -> Dict[str, Any]:
        signal_rows = _flatten_signal_store()
        trades_today = _today_trade_rows()
        current_positions = getattr(paper_broker, "positions", {}) if paper_broker is not None else {}
        lane_positions = defaultdict(int)
        for position_key, position in current_positions.items():
            if getattr(position, "quantity", 0) <= 0:
                continue
            symbol = str(getattr(position, "symbol", position_key) or position_key)
            signal = _signal_store.get(symbol, {}) if isinstance(_signal_store.get(symbol), dict) else {}
            lane_positions[_infer_lane(symbol, payload=signal, metadata={"position_key": position_key})] += 1

        signal_counts = defaultdict(int)
        trade_ready_counts = defaultdict(int)
        for row in signal_rows:
            lane = _infer_lane(row.get("symbol"), payload=row)
            signal_counts[lane] += 1
            if row.get("trade_eligible") or (row.get("meta_decision", {}) or {}).get("take_trade"):
                trade_ready_counts[lane] += 1

        lane_trade_rows = defaultdict(list)
        for trade in trades_today:
            metadata = trade.get("metadata") or {}
            lane = _infer_lane(trade.get("symbol"), metadata=metadata)
            lane_trade_rows[lane].append(trade)

        lanes = []
        total_day_pnl = 0.0
        total_closed = 0
        total_fills = 0
        for lane in ("normal", "day", "crypto"):
            rows = lane_trade_rows.get(lane, [])
            fills = len(rows)
            closed_rows = [r for r in rows if abs(_safe_float(r.get("realized_pnl"), 0.0)) > 1e-9]
            wins = [r for r in closed_rows if _safe_float(r.get("realized_pnl"), 0.0) > 0]
            losses = [r for r in closed_rows if _safe_float(r.get("realized_pnl"), 0.0) < 0]
            day_pnl = round(sum(_safe_float(r.get("realized_pnl"), 0.0) for r in closed_rows), 2)
            total_day_pnl += day_pnl
            total_closed += len(closed_rows)
            total_fills += fills
            gross_wins = sum(_safe_float(r.get("realized_pnl"), 0.0) for r in wins)
            gross_losses = abs(sum(_safe_float(r.get("realized_pnl"), 0.0) for r in losses))
            profit_factor = round(gross_wins / gross_losses, 3) if gross_losses > 0 else (999.0 if gross_wins > 0 else 0.0)
            win_rate_pct = round((len(wins) / max(len(closed_rows), 1)) * 100.0, 1) if closed_rows else 0.0
            avg_pnl = round(day_pnl / max(len(closed_rows), 1), 2) if closed_rows else 0.0

            setup_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0, "gross_wins": 0.0, "gross_losses": 0.0})
            reason_stats = defaultdict(lambda: {"count": 0, "pnl": 0.0})
            time_stats = defaultdict(lambda: {"count": 0, "pnl": 0.0})
            for trade in rows:
                metadata = trade.get("metadata") or {}
                realized = _safe_float(trade.get("realized_pnl"), 0.0)
                setup_id = str(metadata.get("setup_id") or metadata.get("signal_style") or "unclassified")
                exit_reason = str(metadata.get("exit_reason") or metadata.get("action_reason") or trade.get("signal_source") or "unknown")
                time_bucket = str(metadata.get("time_bucket") or "unknown")
                setup_stats[setup_id]["trades"] += 1
                setup_stats[setup_id]["pnl"] += realized
                if realized > 0:
                    setup_stats[setup_id]["wins"] += 1
                    setup_stats[setup_id]["gross_wins"] += realized
                elif realized < 0:
                    setup_stats[setup_id]["gross_losses"] += abs(realized)
                reason_stats[exit_reason]["count"] += 1
                reason_stats[exit_reason]["pnl"] += realized
                time_stats[time_bucket]["count"] += 1
                time_stats[time_bucket]["pnl"] += realized

            setup_rows = []
            for name, stat in setup_stats.items():
                gross_losses_setup = stat["gross_losses"]
                pf = round(stat["gross_wins"] / gross_losses_setup, 3) if gross_losses_setup > 0 else (999.0 if stat["gross_wins"] > 0 else 0.0)
                setup_rows.append(
                    {
                        "setup_id": name,
                        "trades": stat["trades"],
                        "win_rate_pct": round((stat["wins"] / max(stat["trades"], 1)) * 100.0, 1),
                        "pnl": round(stat["pnl"], 2),
                        "profit_factor": pf,
                    }
                )
            setup_rows.sort(key=lambda row: (row["pnl"], row["profit_factor"], row["trades"]), reverse=True)

            reason_rows = [{"reason": key, "count": value["count"], "pnl": round(value["pnl"], 2)} for key, value in reason_stats.items()]
            reason_rows.sort(key=lambda row: (row["count"], row["pnl"]), reverse=True)
            time_rows = [{"bucket": key, "count": value["count"], "pnl": round(value["pnl"], 2)} for key, value in time_stats.items()]
            time_rows.sort(key=lambda row: row["bucket"])

            status = "paper"
            if len(closed_rows) >= 30 and profit_factor >= 1.2 and day_pnl >= 0:
                status = "live_candidate"
            elif len(closed_rows) >= 15 and profit_factor < 1.0:
                status = "review"

            lanes.append(
                {
                    "lane": lane,
                    "lane_label": _lane_label(lane),
                    "signals_live": signal_counts.get(lane, 0),
                    "trade_ready": trade_ready_counts.get(lane, 0),
                    "open_positions": lane_positions.get(lane, 0),
                    "fills_today": fills,
                    "closed_trades_today": len(closed_rows),
                    "day_pnl": day_pnl,
                    "win_rate_pct": win_rate_pct,
                    "profit_factor": profit_factor,
                    "avg_closed_pnl": avg_pnl,
                    "status": status,
                    "setup_stats": setup_rows[:6],
                    "exit_reasons": reason_rows[:6],
                    "time_buckets": time_rows[:6],
                }
            )

        overall = {
            "report_date": datetime.now(timezone.utc).date().isoformat(),
            "total_day_pnl": round(total_day_pnl, 2),
            "fills_today": total_fills,
            "closed_trades_today": total_closed,
            "open_positions": sum(lane_positions.values()),
            "kill_switch_active": bool(getattr(paper_broker, "_kill_switch_activated", False)) if paper_broker is not None else False,
            "kill_switch_reason": getattr(paper_broker, "_kill_switch_reason", "") if paper_broker is not None else "",
        }
        return {"overall": overall, "lanes": lanes}

    # â”€â”€ REST â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @app.route("/")
    def index():
        return render_template_string(DASHBOARD_HTML)

    @app.route("/api/signals")
    def get_signals():
        lane_filter = (request.args.get("lane") or "").strip().lower()
        sigs = _flatten_signal_store()
        if lane_filter in {"normal", "day", "crypto"}:
            sigs = [s for s in sigs if str(s.get("lane", "")).lower() == lane_filter]
        sigs.sort(key=lambda s: (
            0 if s.get("signal") == "buy" else 1 if s.get("signal") == "sell" else 2,
            -(s.get("confidence") or 0),
        ))
        return jsonify({
            "signals":    sigs,
            "count":      len(sigs),
            "buy_count":  sum(1 for s in sigs if s.get("signal") == "buy"),
            "sell_count": sum(1 for s in sigs if s.get("signal") == "sell"),
            "hold_count": sum(1 for s in sigs if s.get("signal") == "neutral"),
            "lane_counts": {
                lane: sum(1 for s in sigs if str(s.get("lane", "")).lower() == lane)
                for lane in ("normal", "day", "crypto")
            },
            "timestamp":  datetime.now(timezone.utc).isoformat(),
        })

    @app.route("/api/signals/<symbol>")
    def get_signal(symbol):
        sig = _signal_store.get(symbol.upper())
        if not sig:
            return jsonify({"error": f"No signal for {symbol}"}), 404
        return jsonify(_enrich_signal_row(dict(sig)))

    @app.route("/api/prices")
    def get_prices():
        if price_buffer is None:
            return jsonify({"prices": {}, "note": "price_buffer not connected"})
        prices = {}
        for sym in price_buffer.active_symbols():
            tick = price_buffer.latest(sym)
            if not tick:
                continue
            recent = price_buffer.recent_ticks(sym, n=60)
            prev = recent[0].price if len(recent) > 1 else tick.price
            chg = round((tick.price / prev - 1) * 100, 3) if prev else 0
            prices[sym] = {
                "price":      round(tick.price, 4),
                "volume":     tick.volume,
                "market":     tick.market,
                "change_pct": chg,
                "direction":  "up" if chg > 0 else "down" if chg < 0 else "flat",
                "timestamp":  tick.timestamp.isoformat(),
            }
        return jsonify({"prices": prices, "active_count": len(prices),
                        "timestamp": datetime.now(timezone.utc).isoformat()})

    @app.route("/api/portfolio")
    def get_portfolio():
        if paper_broker is None:
            return jsonify({"initial_capital": 100000, "current_portfolio_value": 100000,
                            "total_return_pct": 0, "cash": 100000, "open_positions": 0,
                            "total_trades": 0, "win_rate_pct": 0, "kill_switch_active": False,
                            "positions": {}, "note": "paper_broker not connected"})
        summary = _reprice_paper_broker_summary(paper_broker.get_summary())
        summary["return_periods"] = _build_return_periods(summary)
        return jsonify(summary)

    @app.route("/api/positions")
    def get_positions():
        if paper_broker is None:
            return jsonify({"positions": []})
        summary = _reprice_paper_broker_summary(paper_broker.get_summary())
        positions = []
        for position_key, pos in (summary.get("positions") or {}).items():
            entry = dict(pos)
            symbol = str(entry.get("symbol") or position_key)
            entry["symbol"] = symbol
            entry["position_key"] = entry.get("position_key") or position_key
            entry["lane"] = _infer_lane(symbol, metadata={"position_key": entry["position_key"]})
            if price_buffer:
                tick = price_buffer.latest(symbol)
                if tick:
                    entry["current_price"] = round(tick.price, 4)
                    entry["unrealized_pnl"] = round(
                        (tick.price - pos.get("avg_cost", tick.price)) * pos.get("quantity", 0), 2
                    )
            positions.append(entry)
        positions.sort(key=lambda p: abs(p.get("unrealized_pnl", 0)), reverse=True)
        total_count = len(positions)
        limited_positions = positions[:_dashboard_positions_limit]
        return jsonify({
            "positions": limited_positions,
            "count": len(limited_positions),
            "total_count": total_count,
            "display_limit": _dashboard_positions_limit,
            "target_open_positions": _dashboard_target_open_positions,
            "open_positions": summary.get("open_positions", total_count),
            "cash": summary.get("cash"),
            "portfolio_value": summary.get("current_portfolio_value", summary.get("portfolio_value")),
        })

    @app.route("/api/live-portfolio")
    def get_live_portfolio():
        snapshot = live_broker.get_snapshot()
        if snapshot.get("configured"):
            return jsonify(_reprice_portfolio_snapshot(snapshot))
        return jsonify(_reprice_portfolio_snapshot(manual_broker.get_snapshot()))

    @app.route("/api/stress-test")
    def get_stress_test():
        if not _stress_results:
            return jsonify({"_note": "Computing... check back in ~30s"})
        return jsonify(_stress_results)

    @app.route("/api/latency")
    def get_latency():
        if latency_monitor is None:
            return jsonify({"note": "latency_monitor not connected"})
        return jsonify(latency_monitor.stats)

    @app.route("/api/diagnostics")
    def get_diagnostics():
        if not _signal_store:
            return jsonify({"error": "No signals yet",
                            "hint": "Wait ~5 min for first inference cycle"})
        sigs = list(_signal_store.values())
        directions = [s.get("signal", "neutral") for s in sigs]
        confidences = [s.get("confidence", 0) for s in sigs]
        regimes = [s.get("regime", "unknown") for s in sigs]
        factor_scores = [s.get("factor_scores", {}) for s in sigs if s.get("factor_scores")]

        diag = {
            "total_signals":   len(sigs),
            "distribution":    {"buy": directions.count("buy"), "sell": directions.count("sell"),
                                "neutral": directions.count("neutral")},
            "buy_pct":         round(directions.count("buy") / max(len(directions), 1) * 100, 1),
            "sell_pct":        round(directions.count("sell") / max(len(directions), 1) * 100, 1),
            "avg_confidence":  round(sum(confidences) / max(len(confidences), 1), 3),
            "regime_counts":   {r: regimes.count(r) for r in set(regimes)},
        }

        if factor_scores:
            import numpy as np
            factor_names = sorted({name for fs in factor_scores for name in fs.keys()})
            for factor in factor_names:
                vals = [fs.get(factor, 0) for fs in factor_scores]
                diag[f"avg_{factor}_score"] = round(float(np.mean(vals)), 4)
            diag["factor_names"] = factor_names

        issues = []
        if diag["sell_pct"] > diag["buy_pct"] * 3:
            issues.append("SELL BIAS: sell signals 3x buys. Check RSI scoring in signal_engine_v2.py")
        if diag["buy_pct"] + diag["sell_pct"] < 10:
            issues.append("TOO FEW SIGNALS: <10% actionable. Data may be stale.")
        if diag["avg_confidence"] < 0.15:
            issues.append("LOW CONFIDENCE: scores near zero. Check feature normalization.")
        if not issues:
            issues.append("OK: Signal distribution looks healthy.")
        diag["issues"] = issues
        diag["checked_at"] = datetime.now(timezone.utc).isoformat()
        return jsonify(diag)

    @app.route("/api/execution-trace")
    def get_execution_trace():
        rows = list(_execution_trace)
        rows.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
        limited = rows[:_dashboard_execution_trace_limit]
        return jsonify(
            {
                "events": limited,
                "count": len(limited),
                "total_count": len(rows),
                "display_limit": _dashboard_execution_trace_limit,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    @app.route("/api/execution-backtest")
    def get_execution_backtest():
        if not _execution_backtest:
            return jsonify({"note": "Execution backtest not ready yet. Wait for inference cycle."})
        return jsonify(_execution_backtest)

    @app.route("/api/execution-divergence")
    def get_execution_divergence():
        rows = list(_execution_reconciliation or [])
        rows.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
        limited = rows[:_dashboard_execution_trace_limit]
        total = len(rows)
        matched = 0
        routeable = 0
        unsupported = 0
        missing_token = 0
        shadow_errors = 0
        fill_ratio_sum = 0.0
        fill_ratio_count = 0
        partial_fills = 0
        latency_sum = 0.0
        latency_count = 0
        shadow_router = {}
        if paper_broker is not None:
            try:
                shadow_router = dict((paper_broker.get_summary() or {}).get("shadow_router") or {})
            except Exception:
                shadow_router = {}
        for row in rows:
            broker_status = str(row.get("broker_status") or "")
            shadow_status = str(row.get("shadow_status") or "")
            if broker_status == "filled":
                matched += 1
            if shadow_status in {"shadow_ready", "submitted"}:
                routeable += 1
            if shadow_status == "unsupported":
                unsupported += 1
            reason = str(row.get("shadow_reason") or "").lower()
            if "missing instrument token" in reason:
                missing_token += 1
            if shadow_status == "error":
                shadow_errors += 1
            fill_ratio = _safe_float(row.get("fill_ratio"), None)
            if fill_ratio is not None:
                fill_ratio_sum += fill_ratio
                fill_ratio_count += 1
            if bool(row.get("partial_fill")):
                partial_fills += 1
            latency_ms = _safe_float(row.get("simulated_latency_ms"), None)
            if latency_ms is not None:
                latency_sum += latency_ms
                latency_count += 1
        summary = {
            "total_events": total,
            "paper_fills": matched,
            "shadow_routeable": routeable,
            "shadow_routeable_pct": round((routeable / total) * 100, 2) if total else None,
            "shadow_unsupported": unsupported,
            "missing_instrument_tokens": missing_token,
            "shadow_errors": shadow_errors,
            "partial_fill_count": partial_fills,
            "avg_fill_ratio_pct": round((fill_ratio_sum / fill_ratio_count) * 100, 2) if fill_ratio_count else None,
            "avg_simulated_latency_ms": round(latency_sum / latency_count, 2) if latency_count else None,
        }
        return jsonify(
            {
                "summary": summary,
                "shadow_router": shadow_router,
                "events": limited,
                "count": len(limited),
                "total_count": total,
                "display_limit": _dashboard_execution_trace_limit,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    @app.route("/api/portfolio-overlay")
    def get_portfolio_overlay():
        if not _portfolio_overlay:
            return jsonify({"note": "Portfolio overlay not ready yet. Wait for inference cycle."})
        return jsonify(_portfolio_overlay)

    @app.route("/api/meta-model")
    def get_meta_model():
        if not _meta_model_status:
            return jsonify({"note": "Meta model status not ready yet."})
        return jsonify(_meta_model_status)

    @app.route("/api/learning-status")
    def get_learning_status():
        if not _learning_status:
            return jsonify({"note": "Learning status not ready yet."})
        return jsonify(_learning_status)

    @app.route("/api/daily-report")
    def get_daily_report():
        return jsonify(_build_daily_report())

    @app.route("/api/health")
    def health():
        buf = price_buffer.stats if price_buffer else {}
        broker_summary = paper_broker.get_summary() if paper_broker is not None else {}
        return jsonify({
            "status":           "ok",
            "tick_count":       buf.get("total_ticks", 0),
            "active_symbols":   len(price_buffer.active_symbols()) if price_buffer else 0,
            "signal_count":     len(_signal_store),
            "broker_connected": paper_broker is not None,
            "execution_mode":   broker_summary.get("execution_mode", "paper"),
            "shadow_router":    broker_summary.get("shadow_router", {}),
            "stress_computed":  bool(_stress_results),
            "uptime_seconds":   round(time.time() - _startup_time),
            "timestamp":        datetime.now(timezone.utc).isoformat(),
        })

    # â”€â”€ SocketIO events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @socketio.on("connect")
    def handle_connect():
        _client_count[0] += 1
        logger.debug(f"Client connected [{_client_count[0]}]")
        if _signal_store:
            emit("signals_update", {"signals": _limit_values(_flatten_signal_store(), _dashboard_signal_limit)})
        if paper_broker:
            emit("portfolio_update", _reprice_paper_broker_summary(paper_broker.get_summary()))
        if _stress_results:
            emit("stress_update", _stress_results)

    @socketio.on("disconnect")
    def handle_disconnect():
        _client_count[0] = max(0, _client_count[0] - 1)

    @socketio.on("subscribe_symbol")
    def handle_subscribe(data):
        sym = (data.get("symbol") or "").upper()
        if sym and price_buffer:
            tick = price_buffer.latest(sym)
            if tick:
                emit("price_update", {"symbol": sym, "price": tick.price,
                                      "market": tick.market,
                                      "timestamp": tick.timestamp.isoformat()})
        sig = _signal_store.get(sym)
        if sig:
            emit("signal_detail", sig)

    # â”€â”€ Push thread â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    _last_ids = set()
    _last_lat_push = [0.0]
    _last_prices_snapshot: Dict[str, tuple] = {}
    _last_signal_snapshot = {"fingerprint": ""}

    def _push_loop():
        while True:
            try:
                if _client_count[0] == 0:
                    time.sleep(0.5)
                    continue

                # Prices
                if price_buffer:
                    prices = {}
                    for sym in _limit_values(price_buffer.active_symbols(), _dashboard_price_limit):
                        tick = price_buffer.latest(sym)
                        if not tick:
                            continue
                        recent = price_buffer.recent_ticks(sym, n=10)
                        prev = recent[0].price if len(recent) > 1 else tick.price
                        chg = round((tick.price / prev - 1) * 100, 3) if prev else 0
                        snapshot = (round(tick.price, 4), chg, tick.market)
                        if _last_prices_snapshot.get(sym) == snapshot:
                            continue
                        _last_prices_snapshot[sym] = snapshot
                        prices[sym] = {"price": snapshot[0], "change_pct": chg,
                                       "market": tick.market,
                                       "direction": "up" if chg > 0 else "down" if chg < 0 else "flat"}
                    if prices:
                        socketio.emit("prices_update", {"prices": prices})

                # Signals
                cur_ids = set(_signal_store.keys())
                new_ids = cur_ids - _last_ids
                for sid in new_ids:
                    sig = _signal_store.get(sid)
                    if sig:
                        socketio.emit("new_signal", sig)
                _last_ids.update(new_ids)
                if cur_ids:
                    limited_signals = _limit_values(_flatten_signal_store(), _dashboard_signal_limit)
                    fingerprint = "|".join(
                        f"{sig.get('signal_key', sig.get('symbol'))}:{sig.get('signal')}:{round(float(sig.get('confidence', 0.0)), 4)}:{round(float(sig.get('conviction_score', 0.0)), 2)}:{round(float(sig.get('rank_score', 0.0)), 4)}:{sig.get('lane')}:{sig.get('trade_eligible')}"
                        for sig in limited_signals
                    )
                    if fingerprint != _last_signal_snapshot["fingerprint"]:
                        socketio.emit("signals_update", {"signals": limited_signals})
                        _last_signal_snapshot["fingerprint"] = fingerprint

                # Portfolio
                if paper_broker:
                    socketio.emit("portfolio_update", _reprice_paper_broker_summary(paper_broker.get_summary()))

                # Latency
                now = time.time()
                if latency_monitor and now - _last_lat_push[0] > 10:
                    socketio.emit("latency_update", latency_monitor.stats)
                    _last_lat_push[0] = now

            except Exception as exc:
                logger.debug(f"Push loop: {exc}")

            time.sleep(_dashboard_push_interval)

    threading.Thread(target=_push_loop, daemon=True, name="push").start()
    return app, socketio


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Dashboard HTML
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MacroIntel Live</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.6.0/socket.io.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=Syne:wght@700;800&display=swap" rel="stylesheet">
<style>
:root{--bg:#06101f;--bg2:#0a1529;--bg3:#0f1d36;--bg4:#152748;--panel:#0d1830;--panel2:#13213d;--glass:rgba(5,12,24,.84);--glass-strong:rgba(10,17,32,.94);--border:rgba(148,163,184,.18);--border2:rgba(56,189,248,.28);--text:#e5f0ff;--muted:#8ea7cb;--faint:#22375b;--green:#22c55e;--red:#fb7185;--amber:#f59e0b;--blue:#38bdf8;--purple:#a78bfa;--shadow:rgba(2,6,23,.56);--shadow-soft:rgba(2,6,23,.32);--accent-glow:rgba(56,189,248,.16);--mono:'JetBrains Mono',monospace;--sans:'Syne',sans-serif;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:
  radial-gradient(circle at top left, rgba(56,189,248,.18), transparent 28%),
  radial-gradient(circle at 78% 0, rgba(167,139,250,.16), transparent 24%),
  radial-gradient(circle at 100% 40%, rgba(34,197,94,.10), transparent 22%),
  linear-gradient(180deg, #040814 0%, #09111f 40%, #0c1730 100%);
color:var(--text);font-family:var(--mono);font-size:13px;min-height:100vh;}
body.theme-light{--bg:#edf3fb;--bg2:#f9fbff;--bg3:#ffffff;--bg4:#d8e4f3;--panel:#ffffff;--panel2:#f8fbff;--glass:rgba(249,251,255,.88);--glass-strong:rgba(255,255,255,.96);--border:rgba(15,23,42,.09);--border2:rgba(37,99,235,.16);--text:#0f172a;--muted:#5c6c84;--faint:#bed0e5;--green:#059669;--red:#dc2626;--amber:#d97706;--blue:#2563eb;--purple:#7c3aed;--shadow:rgba(15,23,42,.16);--shadow-soft:rgba(15,23,42,.06);--accent-glow:rgba(37,99,235,.12);}
body.theme-light{background:
  radial-gradient(circle at top left, rgba(37,99,235,.12), transparent 30%),
  radial-gradient(circle at 80% 0, rgba(5,150,105,.10), transparent 24%),
  linear-gradient(180deg, #f6f9fe 0%, #eef4fb 42%, #e8eff8 100%);}
.topbar{display:flex;align-items:center;justify-content:space-between;padding:12px 20px;background:var(--glass);backdrop-filter:blur(18px);border-bottom:1px solid var(--border);min-height:58px;position:sticky;top:0;z-index:25;box-shadow:0 12px 30px var(--shadow-soft);}
.menuBtn{border:1px solid var(--border2);background:linear-gradient(135deg, rgba(56,189,248,.12), rgba(167,139,250,.08));color:var(--blue);padding:8px 12px;border-radius:999px;font-family:var(--mono);font-size:10px;cursor:pointer;text-transform:uppercase;letter-spacing:.08em;box-shadow:inset 0 0 0 1px rgba(255,255,255,.02);}
.themeBtn{min-width:68px;}
.menuDrawer{position:fixed;top:68px;left:20px;width:220px;background:var(--glass-strong);border:1px solid var(--border);border-radius:16px;box-shadow:0 24px 60px var(--shadow);padding:10px;display:none;flex-direction:column;gap:8px;z-index:40;}
.menuDrawer.open{display:flex;}
.menuLabel{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.14em;padding:4px 6px;}
.menuLink{border:none;background:transparent;color:var(--muted);padding:11px 12px;border-radius:12px;text-align:left;font-family:var(--mono);font-size:11px;cursor:pointer;}
.menuLink.active{background:linear-gradient(135deg, rgba(56,189,248,.16), rgba(167,139,250,.12));color:var(--text);border:1px solid rgba(56,189,248,.18);}
.logo{font-family:var(--sans);font-size:17px;font-weight:800;letter-spacing:-.5px;}
.logo em{color:var(--blue);font-style:normal;}
.dot{width:7px;height:7px;border-radius:50%;background:var(--red);animation:blink 2s infinite;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
.tr{display:flex;gap:20px;font-size:11px;}.tr span{color:var(--muted);}.tr strong{color:var(--text);}
.sbar{display:flex;gap:16px;padding:7px 20px;background:var(--glass);border-bottom:1px solid var(--border);font-size:10px;color:var(--muted);}
.layout{display:grid;grid-template-columns:276px 1fr;height:calc(100vh - 80px);}
.layout.no-sidebar{grid-template-columns:1fr;}
.layout.no-sidebar .sidebar{display:none;}
.sidebar{background:var(--glass);border-right:1px solid var(--border);overflow-y:auto;backdrop-filter:blur(12px);}
.sbh{padding:10px 14px;font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;}
.sbh button{font-size:9px;background:none;border:1px solid var(--border2);color:var(--muted);padding:2px 8px;border-radius:3px;cursor:pointer;font-family:var(--mono);}
.ssh{padding:9px 14px 6px;font-size:8px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;border-top:1px solid var(--border);}
.ssh:first-child{border-top:none;}
.si{padding:10px 14px;border-bottom:1px solid var(--border);cursor:pointer;transition:background .1s;}
.si:hover{background:var(--bg3);}
.si.active{background:rgba(56,189,248,.10);border-left:2px solid var(--blue);padding-left:12px;}
.sr{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;}
.ssym{font-weight:600;font-size:12px;}
.badge{font-size:9px;padding:1px 6px;border-radius:3px;font-weight:600;text-transform:uppercase;}
.b-buy{background:rgba(34,197,94,.14);color:var(--green);}
.b-sell{background:rgba(239,68,68,.14);color:var(--red);}
.b-neutral{background:rgba(90,100,120,.2);color:var(--muted);}
.smeta{font-size:10px;color:var(--muted);display:flex;gap:8px;}
.ct{height:2px;background:var(--faint);border-radius:1px;margin-top:6px;}
.cf{height:100%;border-radius:1px;transition:width .4s;}
.content{overflow-y:auto;padding:18px;display:flex;flex-direction:column;gap:14px;}
.page{display:none;flex-direction:column;gap:14px;}
.page.active{display:flex;}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;}
.met{background:linear-gradient(180deg, var(--panel2), var(--panel));border:1px solid var(--border);border-radius:12px;padding:12px 14px;box-shadow:0 12px 32px var(--shadow-soft);}
.ml{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;margin-bottom:6px;}
.mv{font-size:20px;font-weight:700;font-family:var(--sans);}
.ms{font-size:10px;color:var(--muted);margin-top:2px;}
.card{background:linear-gradient(180deg, var(--panel2), var(--panel));border:1px solid var(--border);border-radius:16px;padding:16px;box-shadow:0 18px 40px var(--shadow-soft);}
.ct2{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;margin-bottom:12px;}
.dg{display:grid;grid-template-columns:1fr 1fr;gap:14px;}
.heroGrid{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(320px,.7fr);gap:14px;}
.quoteGrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;}
.quoteCard{background:linear-gradient(180deg, rgba(56,189,248,.08), rgba(167,139,250,.06));border:1px solid rgba(56,189,248,.16);border-radius:14px;padding:12px;cursor:pointer;transition:transform .15s ease, box-shadow .15s ease, border-color .15s ease;box-shadow:inset 0 0 0 1px rgba(255,255,255,.02);}
.quoteCard:hover{transform:translateY(-2px);box-shadow:0 14px 30px var(--accent-glow);border-color:rgba(56,189,248,.28);}
.quoteCard.active{border-color:rgba(37,99,235,.42);box-shadow:0 16px 30px rgba(37,99,235,.12), inset 0 0 0 1px rgba(37,99,235,.16);}
.quoteTop{display:flex;justify-content:space-between;gap:8px;align-items:center;margin-bottom:8px;}
.quoteSym{font-family:var(--sans);font-weight:700;font-size:15px;}
.quoteLane{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;}
.quotePrice{font-size:22px;font-family:var(--sans);font-weight:700;}
.quoteMeta{font-size:10px;color:var(--muted);display:flex;gap:10px;flex-wrap:wrap;}
.tvHost{min-height:420px;background:linear-gradient(180deg, var(--panel2), var(--panel));border:1px solid var(--border);border-radius:16px;overflow:hidden;}
.tvEmpty{display:flex;align-items:center;justify-content:center;min-height:420px;color:var(--muted);font-size:11px;}
.wr{display:flex;align-items:center;gap:7px;margin-bottom:4px;}
.wl{width:150px;font-size:10px;color:var(--muted);text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.wt{flex:1;height:16px;background:var(--bg4);border-radius:2px;position:relative;overflow:hidden;}
.wp{position:absolute;left:0;top:0;height:100%;background:rgba(34,197,94,.2);color:var(--green);font-size:9px;display:flex;align-items:center;padding-left:3px;}
.wn{position:absolute;right:0;top:0;height:100%;background:rgba(239,68,68,.18);color:var(--red);font-size:9px;display:flex;align-items:center;justify-content:flex-end;padding-right:3px;}
.fr{display:flex;align-items:center;gap:5px;margin-bottom:3px;}
.fn{width:74px;font-size:10px;color:var(--muted);}
.fb{flex:1;height:5px;background:var(--faint);border-radius:3px;overflow:hidden;}
.ff{height:100%;border-radius:3px;transition:width .4s;}
.fv{width:44px;text-align:right;font-size:10px;}
.fw{width:22px;font-size:9px;color:var(--faint);}
.rg{display:grid;grid-template-columns:1fr 1fr;gap:5px;margin-top:7px;}
.ri{background:var(--bg4);border-radius:4px;padding:7px 9px;}
.rl{font-size:9px;color:var(--muted);margin-bottom:2px;}
.rv{font-size:11px;font-weight:500;}
.rp{display:inline-block;font-size:10px;padding:2px 7px;border-radius:4px;border:1px solid var(--border2);margin:2px;}
.cfb{background:rgba(167,139,250,.07);border:1px solid rgba(167,139,250,.18);border-radius:5px;padding:9px 11px;font-size:10px;color:var(--purple);line-height:1.6;margin-top:9px;}
.wb{background:rgba(245,158,11,.07);border:1px solid rgba(245,158,11,.2);border-radius:5px;padding:7px 11px;font-size:10px;color:var(--amber);margin-top:5px;}
.sg{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;}
.sc{background:var(--bg4);border-radius:5px;padding:9px 11px;border-left:2px solid var(--faint);}
.sc.pass{border-left-color:var(--green);}
.sc.fail{border-left-color:var(--red);}
.sn{font-size:9px;color:var(--muted);margin-bottom:4px;}
.sd{font-size:13px;font-weight:600;}
.sm{font-size:9px;color:var(--muted);margin-top:2px;}
.ptw{max-height:360px;overflow:auto;border-radius:5px;}
.pt{width:100%;border-collapse:collapse;font-size:11px;}
.pt th{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.4px;padding:5px 7px;text-align:left;border-bottom:1px solid var(--border);}
.pt td{padding:6px 7px;border-bottom:1px solid var(--border);}
.pt tr:last-child td{border-bottom:none;}
.pt tr:hover td{background:var(--bg3);}
.pill{display:inline-flex;align-items:center;gap:6px;padding:2px 8px;border:1px solid var(--border2);border-radius:999px;font-size:9px;color:var(--muted);}
.cw{position:relative;height:150px;}
::-webkit-scrollbar{width:3px;height:3px;}
::-webkit-scrollbar-thumb{background:var(--faint);border-radius:2px;}
.up{color:var(--green);}.dn{color:var(--red);}.fl{color:var(--muted);}
.empty{padding:18px;color:var(--muted);font-size:10px;text-align:center;}
.small{font-size:9px;color:var(--muted);}
.ok{color:var(--green);}
.warn{color:var(--amber);}
.bad{color:var(--red);}
.tabs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:4px;}
.tab{background:var(--bg4);border:1px solid var(--border);color:var(--muted);padding:8px 12px;border-radius:999px;font-size:10px;cursor:pointer;font-family:var(--mono);transition:all .15s ease;}
.tab.active{background:rgba(56,189,248,.14);border-color:rgba(56,189,248,.28);color:var(--text);}
.lanehead{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:10px;flex-wrap:wrap;}
.lanegrid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;}
.laneCard{background:var(--bg4);border:1px solid var(--border);border-radius:6px;padding:11px;}
.laneCard.active{border-color:rgba(56,189,248,.32);box-shadow:inset 0 0 0 1px rgba(56,189,248,.12);}
.laneCard.clickable{cursor:pointer;transition:transform .16s ease,border-color .16s ease,box-shadow .16s ease;}
.laneCard.clickable:hover{transform:translateY(-1px);border-color:rgba(56,189,248,.34);box-shadow:0 10px 24px rgba(2,8,23,.18);}
.lanel{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;margin-bottom:6px;}
.lanev{font-size:18px;font-family:var(--sans);font-weight:700;}
.laneSub{font-size:9px;color:var(--muted);margin-top:4px;line-height:1.5;}
.laneStatRow{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-top:8px;padding-top:8px;border-top:1px solid var(--border);}
.laneMini{font-size:9px;color:var(--muted);}
.lanePill{display:inline-flex;align-items:center;gap:6px;padding:3px 8px;border-radius:999px;border:1px solid var(--border);background:rgba(148,163,184,.08);font-size:9px;color:var(--text);}
.tvHost iframe{display:block;}
.returnGrid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;}
.returnCard{background:var(--bg4);border:1px solid var(--border);border-radius:6px;padding:11px;}
.returnLabel{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;margin-bottom:5px;}
.returnValue{font-size:18px;font-family:var(--sans);font-weight:700;}
.returnMeta{font-size:9px;color:var(--muted);margin-top:4px;line-height:1.5;}
.hint{font-size:9px;color:var(--muted);}
@media (max-width: 1024px){.lanegrid{grid-template-columns:1fr;}}
@media (max-width: 1024px){.returnGrid{grid-template-columns:1fr 1fr;}}
@media (max-width: 640px){.returnGrid{grid-template-columns:1fr;}}
@media (max-width: 1100px){.heroGrid{grid-template-columns:1fr;}.layout{grid-template-columns:1fr;height:auto;}.sidebar{display:none;}.layout.show-sidebar .sidebar{display:block;max-height:45vh;}.layout.show-sidebar .content{display:none;}.tr{gap:12px;flex-wrap:wrap;justify-content:flex-end;}}
</style>
</head>
<body>
<div class="topbar">
  <div style="display:flex;align-items:center;gap:10px">
    <button class="menuBtn" type="button" onclick="toggleMenu()">menu</button>
    <button class="menuBtn themeBtn" id="themeToggle" type="button" onclick="toggleTheme()">light</button>
    <div class="dot" id="cdot"></div>
    <div class="logo">Macro<em>Intel</em></div>
    <div style="font-size:10px;color:var(--muted)">live trading intelligence</div>
  </div>
  <div class="tr">
    <span>Symbols <strong id="hs">-</strong></span>
    <span>Signals <strong id="hsi">-</strong></span>
    <span>Portfolio <strong id="hp" class="up">-</strong></span>
    <span>Last tick <strong id="ht">-</strong></span>
    <span>Uptime <strong id="hu">-</strong></span>
  </div>
</div>
<div class="menuDrawer" id="menuDrawer">
  <div class="menuLabel">Trading Pages</div>
  <button class="menuLink active" id="menu-overview" type="button" onclick="setPage('overview')">Overview</button>
  <button class="menuLink" id="menu-signals" type="button" onclick="setPage('signals')">Signal Board</button>
  <button class="menuLink" id="menu-portfolio" type="button" onclick="setPage('portfolio')">Portfolio</button>
  <button class="menuLink" id="menu-execution" type="button" onclick="setPage('execution')">Execution</button>
</div>
<div class="sbar">
  <span>WS avg: <strong id="sl" style="color:var(--muted)">-</strong> ms</span>
  <span>p95: <strong id="sp">-</strong> ms</span>
  <span id="scl" style="color:var(--red)">Disconnected</span>
  <span style="margin-left:auto">
    <button onclick="fetch('/api/diagnostics').then(r=>r.json()).then(d=>alert(JSON.stringify(d.issues||d,null,2)))"
      style="background:none;border:1px solid var(--border2);color:var(--muted);padding:1px 7px;border-radius:3px;cursor:pointer;font-family:var(--mono);font-size:9px">
      run diagnostics
    </button>
  </span>
</div>
<div class="layout">
  <div class="sidebar">
    <div class="sbh"><span>Live Signals</span><button type="button" id="srtbtn" onclick="cycleSortMode()">sort: conviction</button></div>
    <div style="padding:0 12px 8px 12px;display:flex;flex-direction:column;gap:6px">
      <input
        id="ssearch"
        type="text"
        placeholder="Search all stocks..."
        oninput="renderSidebar()"
        style="width:100%;background:#0d121b;border:1px solid var(--border2);color:var(--text);padding:7px 9px;border-radius:5px;font-family:var(--mono);font-size:10px;outline:none"
      />
      <div id="smeta" style="font-size:9px;color:var(--muted)"></div>
    </div>
    <div id="slist"><div class="empty">Connecting...</div></div>
  </div>
  <div class="content">
    <div class="page active" id="page-overview">
      <div class="metrics">
        <div class="met"><div class="ml">Portfolio value</div><div class="mv up" id="mpv">$100,000</div><div class="ms" id="mret">+0.00%</div></div>
        <div class="met"><div class="ml">Directional signals</div><div class="mv" style="color:var(--blue)" id="mns">0</div><div class="ms" id="msp">0 buy • 0 sell • 0 trade-ready</div></div>
        <div class="met"><div class="ml">Win rate (paper)</div><div class="mv" id="mwr">-</div><div class="ms">rolling paper fills</div></div>
        <div class="met"><div class="ml">Stress grade</div><div class="mv" style="color:var(--amber)" id="msg">-</div><div class="ms" id="msgs">black swan score</div></div>
        <div class="met"><div class="ml">Meta ML</div><div class="mv" id="mmeta">-</div><div class="ms" id="mmetas">heuristic fallback</div></div>
        <div class="met"><div class="ml">Model learning</div><div class="mv" id="learnv">-</div><div class="ms" id="learns">feature store not ready</div></div>
      </div>
      <div class="card" style="margin-bottom:12px">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
          <div class="ct2" style="margin-bottom:0">Trading Domains</div>
          <div class="pill" id="laneSummaryPill">cross-lane live view</div>
        </div>
        <div class="lanegrid" id="laneOverviewGrid">
          <div class="laneCard"><div class="lanel">Normal Trading</div><div class="lanev">-</div><div class="laneSub">Waiting for signals</div></div>
          <div class="laneCard"><div class="lanel">Day Trading</div><div class="lanev">-</div><div class="laneSub">Waiting for signals</div></div>
          <div class="laneCard"><div class="lanel">Crypto Scalper</div><div class="lanev">-</div><div class="laneSub">Waiting for signals</div></div>
        </div>
      </div>
      <div class="card" style="margin-bottom:12px">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
          <div class="ct2" style="margin-bottom:0">Returns You Can Read</div>
          <div class="pill" id="retpill">paper performance by lookback</div>
        </div>
        <div class="returnGrid" id="retgrid">
          <div class="returnCard"><div class="returnLabel">1D</div><div class="returnValue">-</div><div class="returnMeta">Waiting for portfolio history</div></div>
          <div class="returnCard"><div class="returnLabel">7D</div><div class="returnValue">-</div><div class="returnMeta">Waiting for portfolio history</div></div>
          <div class="returnCard"><div class="returnLabel">30D</div><div class="returnValue">-</div><div class="returnMeta">Waiting for portfolio history</div></div>
          <div class="returnCard"><div class="returnLabel">90D</div><div class="returnValue">-</div><div class="returnMeta">Waiting for portfolio history</div></div>
          <div class="returnCard"><div class="returnLabel">Since Start</div><div class="returnValue">-</div><div class="returnMeta">Waiting for portfolio history</div></div>
        </div>
      </div>
      <div class="heroGrid">
        <div class="card">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
            <div class="ct2" style="margin-bottom:0">Market board</div>
            <div class="pill" id="qmeta">watching leaders</div>
          </div>
          <div class="quoteGrid" id="qgrid"><div class="empty" style="grid-column:1/-1">Loading market board...</div></div>
        </div>
        <div class="card">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
            <div class="ct2" style="margin-bottom:0">Selected symbol chart</div>
            <div class="pill" id="tvsym">Select a signal</div>
          </div>
          <div id="tvchart" class="tvHost"><div class="tvEmpty">Select a signal to load a TradingView chart</div></div>
        </div>
      </div>
    </div>

    <div class="page" id="page-signals">
      <div class="lanehead">
        <div class="tabs">
          <button class="tab active" id="tab-all" onclick="setLane('all')">All Lanes</button>
          <button class="tab" id="tab-normal" onclick="setLane('normal')">Normal Trading</button>
          <button class="tab" id="tab-day" onclick="setLane('day')">Day Trading</button>
          <button class="tab" id="tab-crypto" onclick="setLane('crypto')">Crypto Scalper</button>
          <button class="tab" id="tab-reports" onclick="setLane('reports')">Daily Reports</button>
        </div>
        <div class="hint" id="lanehint">Cross-lane live view</div>
      </div>
      <div class="card">
        <div class="ct2">Signal intelligence - <span id="dsym" style="color:var(--blue)">select a signal</span> <span id="dtag" style="color:var(--muted)"></span> <span id="dprice" style="color:var(--green)"></span></div>
        <div class="dg">
          <div>
            <div style="font-size:9px;color:var(--muted);margin-bottom:8px">Top model drivers</div>
            <div id="dwf"><div class="empty" style="padding:10px">No signal selected</div></div>
          </div>
          <div style="display:flex;flex-direction:column;gap:10px">
            <div>
              <div style="font-size:9px;color:var(--muted);margin-bottom:6px">Conviction score</div>
              <div style="display:flex;align-items:baseline;gap:6px">
                <span id="dconv" style="font-size:28px;font-weight:700;font-family:var(--sans)">-</span>
                <span style="font-size:10px;color:var(--muted)">/ 10</span>
                <span id="dconf" style="font-size:10px;color:var(--muted)"></span>
              </div>
            </div>
            <div>
              <div style="font-size:9px;color:var(--muted);margin-bottom:6px">Factor breakdown</div>
              <div id="dfact"></div>
            </div>
            <div>
              <div style="font-size:9px;color:var(--muted);margin-bottom:5px">Risk parameters</div>
              <div class="rg" id="drisk"></div>
              <div id="dnote" style="font-size:9px;color:var(--muted);margin-top:5px"></div>
            </div>
            <div>
              <div style="font-size:9px;color:var(--muted);margin-bottom:5px">Market regime</div>
              <div id="dreg"></div>
            </div>
          </div>
        </div>
        <div id="dcf" class="cfb" style="display:none"></div>
        <div id="dwarn"></div>
      </div>
    </div>

    <div class="page" id="page-portfolio">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
        <div class="card">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
            <div class="ct2" style="margin-bottom:0">Open positions</div>
            <div class="pill" id="ppc">0 shown / 0 total</div>
          </div>
          <div id="ppmeta" class="small" style="margin-bottom:10px">Waiting for paper broker state...</div>
          <div class="ptw">
            <table class="pt"><thead><tr><th>Symbol</th><th>Qty</th><th>Cost</th><th>Price</th><th>P&amp;L</th></tr></thead>
            <tbody id="ptb"><tr><td colspan="5" style="color:var(--muted);padding:14px">No open positions</td></tr></tbody></table>
          </div>
        </div>
        <div class="card"><div class="ct2">Paper equity curve</div><div class="cw"><canvas id="eqc"></canvas></div></div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
        <div class="card">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
            <div class="ct2" style="margin-bottom:0">Live brokerage portfolio</div>
            <div class="pill" id="lbc">Upstox</div>
          </div>
          <div class="metrics" style="grid-template-columns:repeat(4,1fr)">
            <div class="met"><div class="ml">Live value</div><div class="mv" id="lbv">-</div><div class="ms" id="lbn">Waiting for broker</div></div>
            <div class="met"><div class="ml">Cash</div><div class="mv" id="lbcash">-</div><div class="ms">Available margin</div></div>
            <div class="met"><div class="ml">Holdings</div><div class="mv" id="lbhold">-</div><div class="ms">Long-term</div></div>
            <div class="met"><div class="ml">Open positions</div><div class="mv" id="lbpos">0</div><div class="ms" id="lbpnl">Day P&amp;L -</div></div>
          </div>
        </div>
        <div class="card">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
            <div class="ct2" style="margin-bottom:0">Live brokerage positions</div>
            <div class="pill" id="lbpc">0 shown / 0 total</div>
          </div>
          <div class="ptw">
            <table class="pt"><thead><tr><th>Symbol</th><th>Type</th><th>Qty</th><th>Price</th><th>P&amp;L</th></tr></thead>
            <tbody id="lptb"><tr><td colspan="5" style="color:var(--muted);padding:14px">Live brokerage not configured</td></tr></tbody></table>
          </div>
        </div>
      </div>
    </div>

    <div class="page" id="page-execution">
      <div class="card">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
          <div class="ct2" style="margin-bottom:0">Daily trading report</div>
          <div class="pill" id="rdate">today</div>
        </div>
        <div class="lanegrid" id="rgrid">
          <div class="laneCard"><div class="lanel">Normal Trading</div><div class="lanev">-</div><div class="laneSub">Waiting for report</div></div>
          <div class="laneCard"><div class="lanel">Day Trading</div><div class="lanev">-</div><div class="laneSub">Waiting for report</div></div>
          <div class="laneCard"><div class="lanel">Crypto Scalper</div><div class="lanev">-</div><div class="laneSub">Waiting for report</div></div>
        </div>
        <div style="display:grid;grid-template-columns:1.2fr .9fr .9fr;gap:12px;margin-top:12px">
          <div class="card" style="padding:10px;background:var(--bg3)">
            <div class="ct2" style="margin-bottom:8px">Setup scoreboard</div>
            <div class="ptw" style="max-height:220px">
              <table class="pt">
                <thead><tr><th>Setup</th><th>Trades</th><th>Win</th><th>PF</th><th>P&amp;L</th></tr></thead>
                <tbody id="rsetup"><tr><td colspan="5" style="color:var(--muted);padding:12px">No setup data yet</td></tr></tbody>
              </table>
            </div>
          </div>
          <div class="card" style="padding:10px;background:var(--bg3)">
            <div class="ct2" style="margin-bottom:8px">Exit reasons</div>
            <div class="ptw" style="max-height:220px">
              <table class="pt">
                <thead><tr><th>Reason</th><th>Count</th><th>P&amp;L</th></tr></thead>
                <tbody id="rreasons"><tr><td colspan="3" style="color:var(--muted);padding:12px">No exit data yet</td></tr></tbody>
              </table>
            </div>
          </div>
          <div class="card" style="padding:10px;background:var(--bg3)">
            <div class="ct2" style="margin-bottom:8px">Time buckets</div>
            <div class="ptw" style="max-height:220px">
              <table class="pt">
                <thead><tr><th>Bucket</th><th>Count</th><th>P&amp;L</th></tr></thead>
                <tbody id="rtimes"><tr><td colspan="3" style="color:var(--muted);padding:12px">No time data yet</td></tr></tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
      <div class="card">
        <div class="ct2">Black swan stress tests - regime-aware</div>
        <div class="sg" id="stg"><div class="empty" style="grid-column:1/-1">Computing...</div></div>
        <div id="stsum" style="font-size:9px;color:var(--muted);margin-top:8px"></div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
        <div class="card">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
            <div class="ct2" style="margin-bottom:0">Execution overlay backtest</div>
            <div class="pill" id="exlb">waiting</div>
          </div>
          <div class="rg">
            <div class="ri"><div class="rl">Overlay return</div><div class="rv" id="exor">-</div></div>
            <div class="ri"><div class="rl">Baseline return</div><div class="rv" id="exbr">-</div></div>
            <div class="ri"><div class="rl">Overlay sharpe</div><div class="rv" id="exos">-</div></div>
            <div class="ri"><div class="rl">Uplift sharpe</div><div class="rv" id="exus">-</div></div>
            <div class="ri"><div class="rl">Overlay max DD</div><div class="rv" id="exod">-</div></div>
            <div class="ri"><div class="rl">Uplift return</div><div class="rv" id="exur">-</div></div>
          </div>
          <div id="exnote" class="small" style="margin-top:8px">Quick rolling check for execution rules.</div>
        </div>
        <div class="card">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
            <div class="ct2" style="margin-bottom:0">Recent execution decisions</div>
            <div class="pill" id="exct">0 shown / 0 total</div>
          </div>
          <div class="ptw" style="max-height:260px">
            <table class="pt">
              <thead><tr><th>Time</th><th>Symbol</th><th>Action</th><th>Status</th><th>Reason</th></tr></thead>
              <tbody id="extb"><tr><td colspan="5" style="color:var(--muted);padding:12px">No execution events yet</td></tr></tbody>
            </table>
          </div>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
        <div class="card">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
            <div class="ct2" style="margin-bottom:0">Shadow routing divergence</div>
            <div class="pill" id="sdmode">paper only</div>
          </div>
          <div class="rg">
            <div class="ri"><div class="rl">Shadow routeable</div><div class="rv" id="sdroute">-</div></div>
            <div class="ri"><div class="rl">Missing NSE tokens</div><div class="rv" id="sdmiss">-</div></div>
            <div class="ri"><div class="rl">Avg fill ratio</div><div class="rv" id="sdfill">-</div></div>
            <div class="ri"><div class="rl">Avg paper latency</div><div class="rv" id="sdlat">-</div></div>
            <div class="ri"><div class="rl">Partial fills</div><div class="rv" id="sdpart">-</div></div>
            <div class="ri"><div class="rl">Shadow errors</div><div class="rv" id="sderr">-</div></div>
          </div>
          <div id="sdnote" class="small" style="margin-top:8px">Shadow routing not active yet.</div>
        </div>
        <div class="card">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
            <div class="ct2" style="margin-bottom:0">Recent shadow checks</div>
            <div class="pill" id="sdct">0 shown / 0 total</div>
          </div>
          <div class="ptw" style="max-height:260px">
            <table class="pt">
              <thead><tr><th>Time</th><th>Symbol</th><th>Paper</th><th>Shadow</th><th>Detail</th></tr></thead>
              <tbody id="sdtb"><tr><td colspan="5" style="color:var(--muted);padding:12px">No shadow checks yet</td></tr></tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
<script>
const SOCKET_AVAILABLE = typeof window.io === 'function';
const CHART_AVAILABLE = typeof window.Chart !== 'undefined';
const io_sock = SOCKET_AVAILABLE ? io({transports:['websocket','polling']}) : { on(){}, emit(){} };
const MARKET_BOARD_LIMIT = 24;
const LEADER_SYMBOLS = ['AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA','AMD','NFLX','QCOM','AVGO','JPM','WMT','XOM','SPY','QQQ','DIA','IWM','RELIANCE.NS','TCS.NS','INFY.NS'];
const TV_AMEX_SYMBOLS = new Set(['SPY','DIA','IWM','GLD','SLV','TLT','HYG','VTI','XLF','XLE','XLK','XLY','XLI','XLV','XLP','XLB','XLU']);
const TV_NYSE_SYMBOLS = new Set(['JPM','BAC','WMT','DIS','KO','JNJ','XOM','CVX','UNH','HD','MCD','NKE','BA','CAT','GS','V','MA','PG','IBM','GE','F','GM','T','VZ','PFE','MRK','ABBV','CRM','ORCL','UBER','SNOW']);
const TV_INDEX_SYMBOLS = {
  '^VIX': 'TVC:VIX',
  '^GSPC': 'SP:SPX',
  '^SPX': 'SP:SPX',
  '^DJI': 'DJ:DJI',
  '^IXIC': 'NASDAQ:IXIC',
  '^NDX': 'NASDAQ:NDX',
  '^NSEI': 'NSE:NIFTY',
  '^NSEBANK': 'NSE:NIFTYBANK',
};
const SORT_MODES = ['conviction', 'rank', 'symbol'];
const SORT_LABELS = {
  conviction: 'sort: conviction',
  rank: 'sort: rank',
  symbol: 'sort: symbol'
};
const LANE_LABELS = {
  all: 'All Lanes',
  normal: 'Normal Trading',
  day: 'Day Trading',
  crypto: 'Crypto Scalper',
  reports: 'Daily Reports'
};
const SIGNAL_LANES = ['normal', 'day', 'crypto'];
let signals = {}, sel = null, eq = [100000], chart = null, sd = 'conviction', activeLane = 'all', activePage = 'overview', reportData = null, livePrices = {};
let healthSnapshot = {};
let tvState = { src: '', symbol: '', theme: '', interval: '' };
let chartSelection = { symbol: '', signalKey: '', lane: 'normal', market: '', source: 'auto', laneLabel: '' };

function setConnectionState(label, color){
  const dot = document.getElementById('cdot');
  const text = document.getElementById('scl');
  if(dot) dot.style.background = color;
  if(text){
    text.textContent = label;
    text.style.color = color;
  }
}

function themePalette(){
  return document.body.classList.contains('theme-light')
    ? {
        axis:'#5c6c84',
        grid:'rgba(15,23,42,0.06)',
        area:'rgba(37,99,235,0.06)',
        lineUp:'#059669',
        lineDown:'#dc2626',
      }
    : {
        axis:'#8ea7cb',
        grid:'rgba(148,163,184,0.10)',
        area:'rgba(56,189,248,0.12)',
        lineUp:'#22c55e',
        lineDown:'#fb7185',
      };
}

function applyTheme(theme, persist=true){
  const nextTheme = theme === 'light' ? 'light' : 'dark';
  document.body.classList.toggle('theme-light', nextTheme === 'light');
  const toggle = document.getElementById('themeToggle');
  if(toggle) toggle.textContent = nextTheme === 'light' ? 'dark' : 'light';
  if(persist){
    try{ localStorage.setItem('macrointel-theme', nextTheme); }catch(_err){}
  }
  updateChart();
  syncChartSelection(false);
}

function initTheme(){
  let storedTheme = 'dark';
  try{
    storedTheme = localStorage.getItem('macrointel-theme') || 'dark';
  }catch(_err){}
  applyTheme(storedTheme, false);
}

function toggleTheme(){
  applyTheme(document.body.classList.contains('theme-light') ? 'dark' : 'light');
}

function openLaneView(lane){
  activeLane = SIGNAL_LANES.includes(lane) ? lane : 'all';
  updateLaneTabs();
  setPage('signals');
}

function formatMoney(value, digits=2){
  const num = Number(value);
  if(!Number.isFinite(num)) return '-';
  return '$' + num.toLocaleString(undefined, {maximumFractionDigits: digits, minimumFractionDigits: digits === 0 ? 0 : Math.min(digits, 2)});
}

function formatSignedPct(value){
  const num = Number(value);
  if(!Number.isFinite(num)) return '-';
  return `${num >= 0 ? '+' : ''}${num.toFixed(2)}%`;
}

function fmtQty(value){
  const num = Number(value);
  if(!Number.isFinite(num)) return String(value ?? '-');
  if(Number.isInteger(num)) return num.toLocaleString();
  const digits = Math.abs(num) >= 1 ? 3 : 6;
  return num.toLocaleString(undefined, {maximumFractionDigits: digits}).replace(/\.?0+$/, '');
}

function toggleMenu(forceOpen){
  const drawer = document.getElementById('menuDrawer');
  if(!drawer) return;
  const nextState = typeof forceOpen === 'boolean' ? forceOpen : !drawer.classList.contains('open');
  drawer.classList.toggle('open', nextState);
}

function setPage(page){
  const nextPage = ['overview', 'signals', 'portfolio', 'execution'].includes(page) ? page : 'overview';
  activePage = nextPage;
  document.querySelectorAll('.page').forEach(el => el.classList.toggle('active', el.id === `page-${nextPage}`));
  ['overview', 'signals', 'portfolio', 'execution'].forEach(name => {
    const link = document.getElementById(`menu-${name}`);
    if(link) link.classList.toggle('active', name === nextPage);
  });
  const layout = document.querySelector('.layout');
  if(layout){
    layout.classList.toggle('no-sidebar', nextPage !== 'signals');
    if(nextPage !== 'signals'){
      layout.classList.remove('show-sidebar');
    }
  }
  if(nextPage === 'overview'){
    renderMarketBoard();
    syncChartSelection(true);
  }else if(nextPage === 'signals'){
    renderSidebar();
    if(sel && signals[sel]) renderDetail(signals[sel]);
  }else if(nextPage === 'portfolio'){
    refreshPortfolioSummary();
    refreshPos();
    refreshLivePortfolio();
  }else if(nextPage === 'execution'){
    refreshExecutionTrace();
    refreshExecutionBacktest();
    refreshExecutionDivergence();
    refreshDailyReport();
  }
  toggleMenu(false);
}

function selectedBySymbol(){
  const grouped = {};
  Object.values(signals).forEach(signal => {
    if(!signal || !signal.symbol) return;
    grouped[signal.symbol] = grouped[signal.symbol] || [];
    grouped[signal.symbol].push(signal);
  });
  return Object.values(grouped).map(rows => preferredSignal(rows)).filter(Boolean).sort(compareSignals);
}

function marketHintForSymbol(symbol){
  const clean = String(symbol || '').trim().toUpperCase();
  return String((livePrices[clean] || {}).market || '');
}

function mergeSignalWithLivePrice(signal){
  if(!signal || !signal.symbol) return null;
  const payload = livePrices[String(signal.symbol).toUpperCase()] || {};
  const merged = {...signal};
  if(payload.price != null){
    merged.last_price = payload.price;
    merged.current_price = payload.price;
  }
  if(payload.change_pct != null){
    merged.price_change_pct = payload.change_pct;
  }
  if(payload.market && !merged.market){
    merged.market = payload.market;
  }
  return merged;
}

function buildLiveFallbackSignal(symbol, overrides={}){
  const clean = String(symbol || '').trim().toUpperCase();
  if(!clean) return null;
  const payload = livePrices[clean] || {};
  const payloadChange = Number(payload.change_pct);
  const overrideChange = Number(overrides.price_change_pct);
  return {
    symbol: clean,
    lane: overrides.lane || 'normal',
    lane_label: overrides.lane_label || overrides.laneLabel || 'Live price',
    signal: overrides.signal || 'neutral',
    last_price: overrides.last_price ?? overrides.current_price ?? payload.price,
    current_price: overrides.current_price ?? overrides.last_price ?? payload.price,
    price_change_pct: Number.isFinite(overrideChange) ? overrideChange : (Number.isFinite(payloadChange) ? payloadChange : null),
    take_probability: Number(overrides.take_probability || 0),
    conviction_score: Number(overrides.conviction_score || 0),
    signal_key: overrides.signal_key || `${clean}::price`,
    market: overrides.market || payload.market || '',
    trade_eligible: !!overrides.trade_eligible,
  };
}

function buildPriceLeaders(limit=MARKET_BOARD_LIMIT){
  const rows = [];
  const seen = new Set();
  LEADER_SYMBOLS.forEach(symbol => {
    const payload = livePrices[symbol];
    if(!payload || seen.has(symbol) || rows.length >= limit) return;
    rows.push(buildLiveFallbackSignal(symbol, {market: payload.market}));
    seen.add(symbol);
  });
  Object.keys(livePrices)
    .sort((a, b) => {
      const delta = Math.abs(Number(livePrices[b]?.change_pct || 0)) - Math.abs(Number(livePrices[a]?.change_pct || 0));
      return delta || a.localeCompare(b);
    })
    .forEach(symbol => {
      if(seen.has(symbol) || rows.length >= limit) return;
      rows.push(buildLiveFallbackSignal(symbol, {market: livePrices[symbol]?.market}));
      seen.add(symbol);
    });
  return rows.filter(Boolean).slice(0, limit);
}

function buildMarketBoardRows(limit=MARKET_BOARD_LIMIT){
  const rows = [];
  const seen = new Set();
  selectedBySymbol().map(mergeSignalWithLivePrice).filter(Boolean).forEach(signal => {
    if(rows.length >= limit || seen.has(signal.symbol)) return;
    rows.push(signal);
    seen.add(signal.symbol);
  });
  buildPriceLeaders(limit).forEach(signal => {
    if(rows.length >= limit || seen.has(signal.symbol)) return;
    rows.push(signal);
    seen.add(signal.symbol);
  });
  return rows;
}

function tradingViewSymbol(symbol, marketHint=''){
  const clean = String(symbol || '').trim().toUpperCase();
  if(!clean) return '';
  if(clean.includes(':')) return clean;
  if(TV_INDEX_SYMBOLS[clean]) return TV_INDEX_SYMBOLS[clean];
  if(clean.endsWith('.NS')) return `NSE:${clean.replace('.NS', '')}`;
  if(clean.endsWith('.BO')) return `BSE:${clean.replace('.BO', '')}`;
  if(clean.endsWith('USDT') || clean.endsWith('USDC') || clean.endsWith('BUSD')) return `BINANCE:${clean}`;
  const market = String(marketHint || '').trim().toUpperCase();
  if(market.includes('NSE')) return `NSE:${clean}`;
  if(market.includes('BSE')) return `BSE:${clean}`;
  if(market.includes('BINANCE') || market.includes('CRYPTO')) return `BINANCE:${clean}`;
  if(market.includes('NYSE')) return `NYSE:${clean}`;
  if(market.includes('AMEX') || market.includes('ARCA') || TV_AMEX_SYMBOLS.has(clean)) return `AMEX:${clean}`;
  if(market.includes('NASDAQ')) return `NASDAQ:${clean}`;
  if(TV_NYSE_SYMBOLS.has(clean)) return `NYSE:${clean}`;
  if(TV_AMEX_SYMBOLS.has(clean)) return `AMEX:${clean}`;
  return clean;
}

function currentChartSignal(){
  if(chartSelection.signalKey && signals[chartSelection.signalKey]){
    return mergeSignalWithLivePrice(signals[chartSelection.signalKey]);
  }
  if(chartSelection.symbol){
    const liveMatch = preferredSignal(Object.values(signals).filter(s => s.symbol === chartSelection.symbol));
    if(liveMatch){
      chartSelection.signalKey = liveMatch.signal_key || '';
      chartSelection.lane = liveMatch.lane || chartSelection.lane;
      chartSelection.laneLabel = liveMatch.lane_label || chartSelection.laneLabel;
      if(liveMatch.market) chartSelection.market = liveMatch.market;
      return mergeSignalWithLivePrice(liveMatch);
    }
    return buildLiveFallbackSignal(chartSelection.symbol, {
      lane: chartSelection.lane || 'normal',
      lane_label: chartSelection.laneLabel || 'Live price',
      market: chartSelection.market || marketHintForSymbol(chartSelection.symbol),
      signal_key: chartSelection.signalKey || `${chartSelection.symbol}::price`,
    });
  }
  return null;
}

function defaultChartSignal(){
  const preferred = selectedBySymbol().map(mergeSignalWithLivePrice).filter(Boolean)[0];
  if(preferred) return preferred;
  return buildPriceLeaders(1)[0] || null;
}

function setChartSelection(signal, source='manual'){
  const merged = mergeSignalWithLivePrice(signal) || buildLiveFallbackSignal(signal && signal.symbol, signal || {});
  if(!merged || !merged.symbol) return null;
  chartSelection = {
    symbol: merged.symbol,
    signalKey: signals[merged.signal_key] ? merged.signal_key : '',
    lane: merged.lane || 'normal',
    market: merged.market || marketHintForSymbol(merged.symbol),
    source,
    laneLabel: merged.lane_label || merged.laneLabel || '',
  };
  return merged;
}

function syncChartSelection(forceFallback=false){
  let signal = currentChartSignal();
  if(!signal && forceFallback){
    const fallback = defaultChartSignal();
    if(fallback) signal = setChartSelection(fallback, 'auto');
  }
  renderTradingViewChart(signal);
}

function isBoardCardActive(signal){
  if(!signal) return false;
  return (!!chartSelection.symbol && signal.symbol === chartSelection.symbol)
    || (!!chartSelection.signalKey && !!signal.signal_key && signal.signal_key === chartSelection.signalKey);
}

function focusBoardSymbol(symbol, signalKey=''){
  const matchedSignal = signalKey && signals[signalKey]
    ? signals[signalKey]
    : preferredSignal(Object.values(signals).filter(s => s.symbol === symbol));
  const signal = matchedSignal || buildLiveFallbackSignal(symbol);
  const selected = setChartSelection(signal, 'board');
  if(matchedSignal){
    sel = matchedSignal.signal_key;
    renderSidebar();
    if(SOCKET_AVAILABLE) io_sock.emit('subscribe_symbol', {symbol: matchedSignal.symbol});
  }
  renderMarketBoard();
  renderTradingViewChart(selected);
}

function openSignal(signalKey){
  setPage('signals');
  selectSig(signalKey);
}

function renderTradingViewChart(signal){
  const host = document.getElementById('tvchart');
  const badge = document.getElementById('tvsym');
  if(!host || !badge) return;
  if(!signal || !signal.symbol){
    badge.textContent = 'Select a signal';
    host.innerHTML = '<div class="tvEmpty">Select a signal to load a TradingView chart</div>';
    tvState = { src: '', symbol: '', theme: '', interval: '' };
    return;
  }
  const symbol = tradingViewSymbol(signal.symbol, signal.market || marketHintForSymbol(signal.symbol));
  if(!symbol){
    badge.textContent = String(signal.symbol || '').toUpperCase();
    host.innerHTML = `<div class="tvEmpty">TradingView symbol mapping is not ready for ${badge.textContent}</div>`;
    tvState = { src: '', symbol: '', theme: '', interval: '' };
    return;
  }
  const interval = signal.lane === 'day' || signal.lane === 'crypto' ? '15' : 'D';
  const tvTheme = document.body.classList.contains('theme-light') ? 'light' : 'dark';
  const toolbarBg = tvTheme === 'light' ? '%23f8fafc' : '%23091322';
  const src = `https://s.tradingview.com/widgetembed/?frameElementId=tvchart_frame&symbol=${encodeURIComponent(symbol)}&interval=${encodeURIComponent(interval)}&hidesidetoolbar=1&symboledit=1&saveimage=0&toolbarbg=${toolbarBg}&theme=${tvTheme}&style=1&timezone=Etc%2FUTC&withdateranges=1&hideideas=1`;
  badge.textContent = symbol;
  if(tvState.src === src){
    return;
  }
  const frame = host.querySelector('iframe');
  if(frame){
    frame.src = src;
    frame.title = `TradingView ${symbol}`;
  }else{
    host.innerHTML = `<iframe title="TradingView ${symbol}" src="${src}" style="width:100%;height:420px;border:0" loading="lazy" referrerpolicy="origin"></iframe>`;
  }
  tvState = { src, symbol, theme: tvTheme, interval };
}

function updateSelectedPrice(signal){
  const priceEl = document.getElementById('dprice');
  if(!priceEl) return;
  const price = signal && (signal.last_price ?? signal.current_price);
  const change = Number(signal && signal.price_change_pct);
  if(price == null || !Number.isFinite(Number(price))){
    priceEl.textContent = 'waiting for live price';
    priceEl.style.color = 'var(--muted)';
    return;
  }
  priceEl.textContent = `${formatMoney(price)}${Number.isFinite(change) ? ` • ${formatSignedPct(change)}` : ''}`;
  priceEl.style.color = Number.isFinite(change)
    ? (change > 0 ? 'var(--green)' : change < 0 ? 'var(--red)' : 'var(--muted)')
    : 'var(--text)';
}

function renderMarketBoard(){
  const grid = document.getElementById('qgrid');
  const meta = document.getElementById('qmeta');
  if(!grid) return;
  const leaders = selectedBySymbol().slice(0, MARKET_BOARD_LIMIT);
  const rows = buildMarketBoardRows(MARKET_BOARD_LIMIT);
  if(meta){
    const ready = leaders.filter(isTradeReady).length;
    meta.textContent = leaders.length
      ? `${ready} trade-ready • ${rows.length} shown`
      : rows.length
        ? `${rows.length} live leaders • click to load chart`
        : 'watching leaders';
  }
  if(!rows.length){
    grid.innerHTML = '<div class="empty" style="grid-column:1/-1">Waiting for live prices and signal stream...</div>';
    return;
  }
  grid.innerHTML = rows.map(signal => {
    const price = signal.last_price ?? signal.current_price;
    const change = Number(signal.price_change_pct);
    const changeColor = Number.isFinite(change) ? (change > 0 ? 'var(--green)' : change < 0 ? 'var(--red)' : 'var(--muted)') : 'var(--muted)';
    const take = Math.round(Number(signal.take_probability || (signal.meta_decision || {}).take_probability || 0) * 100);
    const signalKey = signals[signal.signal_key] ? signal.signal_key : '';
    const active = isBoardCardActive(signal);
    return `<div class="quoteCard${active ? ' active' : ''}" onclick='focusBoardSymbol(${JSON.stringify(signal.symbol)}, ${JSON.stringify(signalKey)})'>
      <div class="quoteTop">
        <div>
          <div class="quoteSym">${signal.symbol}</div>
          <div class="quoteLane">${signal.lane_label || LANE_LABELS[signal.lane] || signal.lane || 'Signal'}</div>
        </div>
        <div class="badge ${isTradeReady(signal) ? `b-${signal.signal}` : 'b-neutral'}">${isTradeReady(signal) ? (signal.signal || 'watch').toUpperCase() : (signalKey ? 'WATCH' : 'LIVE')}</div>
      </div>
      <div class="quotePrice">${price != null && Number.isFinite(Number(price)) ? formatMoney(price) : '-'}</div>
      <div class="quoteMeta">
        <span style="color:${changeColor}">${Number.isFinite(change) ? formatSignedPct(change) : 'price pending'}</span>
        <span>${signalKey ? `conv ${(signal.conviction_score || 0).toFixed(1)}/10` : 'live tape'}</span>
        <span>${signalKey ? `take ${take}%` : 'awaiting signal'}</span>
      </div>
    </div>`;
  }).join('');
}

function laneReportMap(){
  const items = Array.isArray(reportData && reportData.lanes) ? reportData.lanes : [];
  const map = {};
  items.forEach(item => {
    if(item && item.lane){
      map[item.lane] = item;
    }
  });
  return map;
}

function renderLaneOverview(){
  const grid = document.getElementById('laneOverviewGrid');
  const pill = document.getElementById('laneSummaryPill');
  if(!grid || !pill) return;
  const reportMap = laneReportMap();
  const cards = SIGNAL_LANES.map(lane => {
    const laneSignals = Object.values(signals).filter(signal => signal.lane === lane);
    const actionable = laneSignals.filter(isTradeReady);
    const buys = actionable.filter(signal => signal.signal === 'buy').length;
    const sells = actionable.filter(signal => signal.signal === 'sell').length;
    const watch = Math.max(0, laneSignals.length - actionable.length);
    const laneReport = reportMap[lane] || {};
    const pnl = Number(laneReport.day_pnl || 0);
    const status = laneReport.status || (actionable.length ? 'active' : laneSignals.length ? 'warming' : 'standby');
    const active = activeLane === lane || (activeLane === 'all' && lane === 'normal');
    const accent = pnl > 0 ? 'var(--green)' : pnl < 0 ? 'var(--red)' : 'var(--text)';
    return `<div class="laneCard clickable${active ? ' active' : ''}" onclick="openLaneView('${lane}')">
      <div class="lanel">${LANE_LABELS[lane]}</div>
      <div class="lanev" style="color:${actionable.length ? 'var(--blue)' : accent}">${actionable.length}</div>
      <div class="laneSub">${actionable.length} trade-ready • ${laneSignals.length} total signals<br>${buys} buy • ${sells} sell • ${watch} watch</div>
      <div class="laneStatRow">
        <span class="lanePill">${status}</span>
        <span class="laneMini">${laneReport.open_positions || 0} open • day P&amp;L ${pnl >= 0 ? '+' : '-'}$${Math.abs(pnl).toFixed(2)}</span>
      </div>
    </div>`;
  });
  const totalActionable = Object.values(signals).filter(isTradeReady).length;
  pill.textContent = `${totalActionable} trade-ready across 3 domains`;
  grid.innerHTML = cards.join('');
}

function normalizeSignal(s){
  if(!s || !s.symbol) return null;
  const dir = s.signal || s.direction || 'neutral';
  const lane = s.lane || (String(s.signal_style || '').toLowerCase() === 'crypto_depth_intraday' ? 'crypto' : String(s.signal_style || '').toLowerCase() === 'day_trade_intraday' ? 'day' : 'normal');
  const signalKey = s.signal_key || `${s.symbol}::${lane}`;
  return {...s, signal: dir === 'hold' ? 'neutral' : dir, lane, lane_label: s.lane_label || LANE_LABELS[lane] || lane, signal_key: signalKey};
}

function isTradeReady(s){
  return !!(s && (s.trade_eligible || (s.meta_decision && s.meta_decision.take_trade)));
}

function rankedSortValue(a, b){
  return ((isTradeReady(b) ? 1 : 0) - (isTradeReady(a) ? 1 : 0)
    || (a.signal === 'buy' ? 0 : a.signal === 'sell' ? 1 : 2) - (b.signal === 'buy' ? 0 : b.signal === 'sell' ? 1 : 2)
    || ((b.rank_score || 0) - (a.rank_score || 0))
    || ((b.take_probability || 0) - (a.take_probability || 0))
    || ((b.conviction_score || 0) - (a.conviction_score || 0))
    || ((b.confidence || 0) - (a.confidence || 0))
    || a.symbol.localeCompare(b.symbol));
}

function convictionSortValue(a, b){
  return ((isTradeReady(b) ? 1 : 0) - (isTradeReady(a) ? 1 : 0)
    || ((b.conviction_score || 0) - (a.conviction_score || 0))
    || ((b.confidence || 0) - (a.confidence || 0))
    || ((b.take_probability || 0) - (a.take_probability || 0))
    || ((b.rank_score || 0) - (a.rank_score || 0))
    || (a.signal === 'buy' ? 0 : a.signal === 'sell' ? 1 : 2) - (b.signal === 'buy' ? 0 : b.signal === 'sell' ? 1 : 2)
    || a.symbol.localeCompare(b.symbol));
}

function compareSignals(a, b){
  if(sd === 'symbol') return a.symbol.localeCompare(b.symbol);
  if(sd === 'rank') return rankedSortValue(a, b);
  return convictionSortValue(a, b);
}

function updateSortButton(){
  const btn = document.getElementById('srtbtn');
  if(btn) btn.textContent = SORT_LABELS[sd] || 'sort';
}

function cycleSortMode(){
  const idx = SORT_MODES.indexOf(sd);
  sd = SORT_MODES[(idx + 1) % SORT_MODES.length];
  updateSortButton();
  renderSidebar();
}

function updateLaneTabs(){
  ['all','normal','day','crypto','reports'].forEach(lane => {
    const el = document.getElementById('tab-' + lane);
    if(el) el.classList.toggle('active', lane === activeLane);
  });
  const hint = document.getElementById('lanehint');
  if(hint) hint.textContent = activeLane === 'reports'
    ? 'Daily report mode'
    : `${LANE_LABELS[activeLane] || 'All Lanes'} signal view`;
}

function setLane(lane){
  activeLane = lane || 'all';
  updateLaneTabs();
  renderSidebar();
  renderLaneOverview();
  renderDailyReport(reportData);
  if(sel && signals[sel]) renderDetail(signals[sel]);
}

function preferredSignal(rows){
  if(!rows.length) return null;
  const sorted = [...rows].sort(compareSignals);
  return sorted.find(isTradeReady)
    || sorted.find(s => s.signal === 'buy')
    || sorted.find(s => s.signal === 'sell')
    || sorted[0];
}

function applySignals(rows, replace=false){
  const clean = (rows || []).map(normalizeSignal).filter(Boolean);
  if(replace) signals = {};
  clean.forEach(s => { signals[s.signal_key] = s; });
  const arr = Object.values(signals);
  renderSidebar();
  renderMarketBoard();
  renderLaneOverview();
  updateMetrics();
  if((!sel || !signals[sel]) && arr.length){
    const preferred = preferredSignal(arr);
    sel = preferred ? preferred.signal_key : null;
  }
  if(!chartSelection.symbol){
    const preferred = preferredSignal(arr);
    if(preferred) setChartSelection(preferred, 'auto');
  }else if(chartSelection.signalKey && !signals[chartSelection.signalKey]){
    const replacement = preferredSignal(arr.filter(s => s.symbol === chartSelection.symbol));
    if(replacement){
      setChartSelection(replacement, chartSelection.source || 'auto');
    }else{
      chartSelection.signalKey = '';
    }
  }
  if(sel && signals[sel]) renderDetail(signals[sel]);
  syncChartSelection(!chartSelection.symbol);
}

function applyPortfolioSummary(d){
  const val = Number(d.current_portfolio_value ?? d.portfolio_value ?? 100000);
  const ret = Number(d.total_return_pct ?? 0);
  const mpv = document.getElementById('mpv');
  const mret = document.getElementById('mret');
  const hp = document.getElementById('hp');
  const mwr = document.getElementById('mwr');
  if(mpv){
    mpv.textContent = '$' + Math.round(val).toLocaleString();
    mpv.style.color = ret >= 0 ? 'var(--green)' : 'var(--red)';
  }
  if(mret){
    mret.textContent = (ret >= 0 ? '+' : '') + ret.toFixed(2) + '%';
    mret.style.color = ret >= 0 ? 'var(--green)' : 'var(--red)';
  }
  if(hp) hp.textContent = '$' + Math.round(val).toLocaleString();
  if(mwr) mwr.textContent = d.win_rate_pct != null ? Number(d.win_rate_pct).toFixed(1) + '%' : '-';
  if(!eq.length || eq[eq.length - 1] !== val){
    eq.push(val);
    if(eq.length > 200) eq.shift();
    updateChart();
  }
  renderReturnPeriods(d.return_periods || []);
}

function refreshPricesSnapshot(){
  fetch('/api/prices').then(r => r.json()).then(d => {
    livePrices = d.prices || {};
    renderMarketBoard();
    if(!chartSelection.symbol){
      const preferred = defaultChartSignal();
      if(preferred) setChartSelection(preferred, 'auto');
    }
    syncChartSelection(!chartSelection.symbol);
  }).catch(() => {});
}

function renderReturnPeriods(periods){
  const grid = document.getElementById('retgrid');
  const pill = document.getElementById('retpill');
  if(!grid || !pill) return;
  if(!Array.isArray(periods) || !periods.length){
    grid.innerHTML = '<div class="returnCard"><div class="returnLabel">Returns</div><div class="returnValue">-</div><div class="returnMeta">Waiting for portfolio history</div></div>';
    pill.textContent = 'paper performance by lookback';
    return;
  }
  pill.textContent = `${periods.length} windows from paper history`;
  grid.innerHTML = periods.map(item => {
    const days = Number(item.actual_days ?? item.days ?? 0);
    const ret = Number(item.return_pct ?? 0);
    const good = ret >= 0;
    const color = good ? 'var(--green)' : 'var(--red)';
    const label = item.label || `${days}D`;
    const baseline = item.baseline_value != null ? fmtMoney(item.baseline_value) : '-';
    return `<div class="returnCard">
      <div class="returnLabel">${label}</div>
      <div class="returnValue" style="color:${color}">${ret >= 0 ? '+' : ''}${ret.toFixed(2)}%</div>
      <div class="returnMeta">from the last ${days} day${days === 1 ? '' : 's'}</div>
      <div class="returnMeta">base ${baseline}</div>
    </div>`;
  }).join('');
}

function renderSignalCard(s){
  const dir = s.signal || 'neutral';
  const conf = Math.round((s.confidence || 0) * 100);
  const conv = (s.conviction_score || 0).toFixed(1);
  const actionable = isTradeReady(s);
  const meta = s.meta_decision || {};
  const regimeColor = s.regime === 'stressed' ? 'var(--amber)' : s.regime === 'crisis' ? 'var(--red)' : 'var(--muted)';
  const laneLabel = s.lane_label || LANE_LABELS[s.lane] || s.lane || '';
  const badgeLabel = actionable ? dir.toUpperCase() : (dir !== 'neutral' ? `BIAS ${dir.toUpperCase()}` : 'WATCH');
  const badgeClass = actionable ? `b-${dir}` : (dir !== 'neutral' ? `b-${dir}` : 'b-neutral');
  const gateLabel = actionable
    ? `live ${Math.round(((s.take_probability ?? meta.take_probability ?? 0)) * 100)}%`
    : `skip ${Math.round(((s.skip_probability ?? meta.skip_probability ?? 0)) * 100)}%`;
  const barColor = actionable
    ? (dir === 'buy' ? 'var(--green)' : dir === 'sell' ? 'var(--red)' : 'var(--faint)')
    : 'var(--amber)';
  return `<div class="si${s.signal_key === sel ? ' active' : ''}" onclick="selectSig('${s.signal_key}')">
    <div class="sr"><span class="ssym">${s.symbol}</span><span class="badge ${badgeClass}">${badgeLabel}</span></div>
    <div class="sr"><span class="smeta"><span>${laneLabel}</span><span>conv ${conv}/10</span><span>conf ${conf}%</span><span>${gateLabel}</span><span style="color:${regimeColor}">${s.regime || ''}</span></span>
    <span data-price-symbol="${s.symbol}" class="fl"></span></div>
    <div class="ct"><div class="cf" style="width:${conf}%;background:${barColor}"></div></div>
  </div>`;
}

function renderSection(title, rows, emptyMessage=''){
  if(!rows.length && !emptyMessage) return '';
  const body = rows.length ? rows.map(renderSignalCard).join('') : `<div class="empty">${emptyMessage}</div>`;
  return `<div class="ssh">${title}</div>${body}`;
}

if(SOCKET_AVAILABLE){
  io_sock.on('connect', () => setConnectionState('Connected', 'var(--green)'));
  io_sock.on('disconnect', () => setConnectionState('Disconnected', 'var(--red)'));
  io_sock.on('signals_update', d => applySignals(d.signals || [], true));
  io_sock.on('new_signal', s => applySignals([s]));
  io_sock.on('prices_update', d => {
    const ht = document.getElementById('ht');
    const hs = document.getElementById('hs');
    if(ht) ht.textContent = new Date().toLocaleTimeString();
    if(hs) hs.textContent = Object.keys(d.prices || {}).length;
    Object.entries(d.prices || {}).forEach(([sym, p]) => {
      Object.values(signals).forEach(signal => {
        if(signal.symbol === sym){
          signal.last_price = p.price;
          signal.current_price = p.price;
          signal.price_change_pct = p.change_pct;
          signal.price_direction = p.direction;
        }
      });
      document.querySelectorAll(`[data-price-symbol="${sym}"]`).forEach(el => {
        el.className = p.direction === 'up' ? 'up' : p.direction === 'down' ? 'dn' : 'fl';
        el.textContent = p.price != null ? '$' + Number(p.price).toLocaleString(undefined, {maximumFractionDigits:2}) : '';
      });
    });
    renderMarketBoard();
    if(sel && signals[sel]) updateSelectedPrice(signals[sel]);
    if(!chartSelection.symbol){
      const preferred = defaultChartSignal();
      if(preferred) setChartSelection(preferred, 'auto');
    }
    syncChartSelection(!chartSelection.symbol);
  });
  io_sock.on('portfolio_update', d => {
    applyPortfolioSummary(d || {});
    if(d.kill_switch_active){
      let b = document.getElementById('ksb');
      if(!b){
        b = document.createElement('div');
        b.id = 'ksb';
        b.style.cssText = 'background:rgba(239,68,68,.13);border:1px solid var(--red);border-radius:5px;padding:7px 12px;font-size:10px;color:var(--red);margin-bottom:10px';
        document.querySelector('.content').prepend(b);
      }
      const reason = d.kill_switch_reason ? String(d.kill_switch_reason) : 'drawdown limit hit';
      b.textContent = `KILL SWITCH ACTIVE - ${reason}. No new orders.`;
    }
  });
  io_sock.on('stress_update', renderStress);
  io_sock.on('latency_update', d => {
    const l = document.getElementById('sl');
    const p95 = document.getElementById('sp');
    if(l){
      l.textContent = d.avg_latency_ms != null ? d.avg_latency_ms.toFixed(0) : '-';
      l.style.color = d.avg_latency_ms < 500 ? 'var(--green)' : 'var(--amber)';
    }
    if(p95) p95.textContent = d.p95_ms != null ? d.p95_ms.toFixed(0) : '-';
  });
}else{
  setConnectionState('Polling', 'var(--amber)');
}

function renderSidebar(){
  const el = document.getElementById('slist');
  const countEl = document.getElementById('hsi');
  const metaEl = document.getElementById('smeta');
  const searchInput = document.getElementById('ssearch');
  const query = (searchInput && searchInput.value ? searchInput.value : '').trim().toLowerCase();
  let arr = Object.values(signals);
  const laneFilter = activeLane === 'reports' ? 'all' : activeLane;
  if(!arr.length){
    const activeSymbols = Number(healthSnapshot.active_symbols || 0);
    const tickCount = Number(healthSnapshot.tick_count || 0);
    const warming = activeSymbols > 0 || tickCount > 0;
    if(countEl) countEl.textContent = warming ? 'warming' : '0';
    if(metaEl) metaEl.textContent = warming
      ? `Signal engine warming up • ${activeSymbols} symbols live • ${tickCount.toLocaleString()} ticks`
      : 'Waiting for signal stream...';
    el.innerHTML = `<div class="empty">${warming ? 'Signal engine is warming up from live data...' : 'No signals yet...'}</div>`;
    return;
  }
  arr.sort(compareSignals);
  if(laneFilter !== 'all'){
    arr = arr.filter(s => s.lane === laneFilter);
  }
  const filtered = query
    ? arr.filter(s => {
        const meta = s.meta_decision || {};
        const blob = [
          s.symbol,
          s.signal,
          s.regime,
          s.lane_label,
          s.setup_id,
          isTradeReady(s) ? 'trade ready' : 'watch',
          meta.reason,
        ].join(' ').toLowerCase();
        return blob.includes(query);
      })
    : arr;
  const laneMap = {
    normal: filtered.filter(s => s.lane === 'normal'),
    day: filtered.filter(s => s.lane === 'day'),
    crypto: filtered.filter(s => s.lane === 'crypto'),
  };
  const actionable = filtered.filter(isTradeReady);
  const broaderWatch = filtered.filter(s => !isTradeReady(s));
  if(query){
    el.innerHTML = renderSection(`Search results (${filtered.length})`, filtered, 'No stocks matched your search');
    if(metaEl) metaEl.textContent = `Showing ${filtered.length} of ${arr.length} stocks`;
  }else if(laneFilter !== 'all'){
    el.innerHTML = [
      renderSection(`Trade Ready (${actionable.length})`, actionable, `No ${LANE_LABELS[laneFilter]} ideas are trade-ready right now`),
      renderSection(`Watchlist (${broaderWatch.length})`, broaderWatch, `No ${LANE_LABELS[laneFilter]} watchlist signals yet`),
    ].filter(Boolean).join('');
    if(metaEl) metaEl.textContent = `${arr.length} ${LANE_LABELS[laneFilter].toLowerCase()} signals - ${actionable.length} trade-ready`;
  }else{
    el.innerHTML = [
      renderSection(`Normal Trading (${laneMap.normal.length})`, laneMap.normal, 'No normal-trading signals yet'),
      renderSection(`Day Trading (${laneMap.day.length})`, laneMap.day, 'No day-trading signals yet'),
      renderSection(`Crypto Scalper (${laneMap.crypto.length})`, laneMap.crypto, 'No crypto scalper signals yet'),
    ].filter(Boolean).join('');
    if(metaEl) metaEl.textContent = `${arr.length} total lane signals - ${actionable.length} trade-ready - ${broaderWatch.length} on watch`;
  }
  if(countEl) countEl.textContent = String(actionable.length);
}

function updateMetrics(){
  const arr = Object.values(signals);
  const filtered = activeLane === 'all' || activeLane === 'reports' ? arr : arr.filter(s => s.lane === activeLane);
  const actionable = filtered.filter(isTradeReady);
  const buys = actionable.filter(s => s.signal === 'buy').length;
  const sells = actionable.filter(s => s.signal === 'sell').length;
  const watch = filtered.length - actionable.length;
  const mns = document.getElementById('mns');
  const msp = document.getElementById('msp');
  const warming = !arr.length && Number(healthSnapshot.active_symbols || 0) > 0;
  if(mns) mns.textContent = warming ? 'WARMING' : String(actionable.length);
  if(msp) msp.textContent = warming
    ? `signal engine warming up from ${Number(healthSnapshot.active_symbols || 0)} live symbols`
    : `${buys} live buy - ${sells} live sell - ${watch} watch${activeLane !== 'all' && activeLane !== 'reports' ? ` - ${LANE_LABELS[activeLane]}` : ''}`;
}

function selectSig(sym){
  sel = sym;
  renderSidebar();
  const signal = signals[sym];
  const layout = document.querySelector('.layout');
  if(layout && window.innerWidth <= 1100){
    layout.classList.remove('show-sidebar');
  }
  if(SOCKET_AVAILABLE && signal) io_sock.emit('subscribe_symbol', {symbol: signal.symbol});
  if(signal){
    setChartSelection(signal, 'signal');
    renderDetail(signal);
  }
}

const FLABELS = {
  trend:'Trend',
  momentum:'Momentum',
  mean_revert:'Mean Rev',
  volume:'Volume',
  sentiment:'Sentiment',
  earnings_propagation:'Earn Prop',
  close_reversal:'Close Reversal'
};

const FWEIGHTS = {
  trend:0.22,
  momentum:0.18,
  mean_revert:0.14,
  volume:0.10,
  sentiment:0.08,
  earnings_propagation:0.16,
  close_reversal:0.12
};

function renderDetail(s){
  const dsym = document.getElementById('dsym');
  const dconv = document.getElementById('dconv');
  const dconf = document.getElementById('dconf');
  const dtag = document.getElementById('dtag');
  if(dsym) dsym.textContent = s.symbol;
  if(dtag) dtag.textContent = `• ${s.lane_label || LANE_LABELS[s.lane] || ''} • ${s.setup_id || s.signal_style || 'signal'}`;
  updateSelectedPrice(s);
  const conv = s.conviction_score || 0;
  if(dconv){
    dconv.textContent = conv.toFixed(1);
    dconv.style.color = s.signal === 'buy' ? 'var(--green)' : s.signal === 'sell' ? 'var(--red)' : 'var(--muted)';
  }
  if(dconf) dconf.textContent = 'conf ' + Math.round((s.confidence || 0) * 100) + '%';
  const drivers = (Array.isArray(s.waterfall_data) && s.waterfall_data.length) ? s.waterfall_data : (s.top_drivers || []);
  renderWF(drivers);

  const fac = s.factor_scores || {};
  const weights = s.factor_weights || FWEIGHTS;
  const dfact = document.getElementById('dfact');
  if(dfact){
    dfact.innerHTML = Object.entries(fac).map(([k, v]) => {
      const pct = Math.min(100, Math.abs(v) * 100).toFixed(0);
      const col = v > 0.05 ? 'var(--green)' : v < -0.05 ? 'var(--red)' : 'var(--faint)';
      return `<div class="fr"><span class="fn">${FLABELS[k] || k}</span><div class="fb"><div class="ff" style="width:${pct}%;background:${col}"></div></div>
        <span class="fv" style="color:${col}">${v > 0 ? '+' : ''}${Number(v).toFixed(3)}</span><span class="fw">${Math.round((weights[k] || 0) * 100)}%w</span></div>`;
    }).join('') + (s.regime_multiplier != null ? `<div style="font-size:9px;color:var(--muted);margin-top:4px;padding-top:4px;border-top:1px solid var(--border)">
      Regime: <span style="color:${s.regime === 'stressed' ? 'var(--amber)' : s.regime === 'crisis' ? 'var(--red)' : 'var(--muted)'}">${s.regime || 'normal'}</span>
      - signals at ${Math.round((s.regime_multiplier || 1) * 100)}%</div>` : '');
  }

  const risk = s.risk_parameters || {};
  const drisk = document.getElementById('drisk');
  if(drisk){
    drisk.innerHTML = [
      ['Stop loss', risk.stop_loss_pct ? risk.stop_loss_pct + '%' : '-'],
      ['Take profit', risk.take_profit_pct ? risk.take_profit_pct + '%' : '-'],
      ['Trail stop', risk.trailing_stop_pct ? risk.trailing_stop_pct + '%' : '-'],
      ['Position size', risk.suggested_position_size_pct ? risk.suggested_position_size_pct + '%' : '-'],
      ['R:R', risk.risk_reward_ratio ? risk.risk_reward_ratio + ':1' : '-'],
    ].map(([l, v]) => `<div class="ri"><div class="rl">${l}</div><div class="rv">${v}</div></div>`).join('');
  }

  const meta = s.meta_decision || {};
  const fit = s.portfolio_fit || {};
  const alloc = s.portfolio_construction || {};
  const intraday = s.intraday_overlay || {};
  const metaNote = meta.decision_label ? `meta ${String(meta.decision_label).toUpperCase()} - take ${Math.round((meta.take_probability || 0) * 100)}% - edge ${meta.expected_edge_pct != null ? meta.expected_edge_pct + '%' : '-'} - rank ${meta.rank || '-'}` : '';
  const fitNote = (fit.sample_size || 0) > 0 ? `portfolio fit avg corr ${Math.round((fit.avg_abs_corr || 0) * 100)}% - max ${Math.round((fit.max_abs_corr || 0) * 100)}%` : '';
  const allocNote = alloc.target_position_pct ? `allocator ${Math.round((alloc.target_position_pct || 0) * 100)}% capital - beta ${Number(alloc.beta || 0).toFixed(2)} - residual ${Number(alloc.residual_alpha_score || 0).toFixed(3)}` : '';
  const intradayNote = intraday.setup
    ? `intraday ${String(intraday.setup).replaceAll('_', ' ')} - ${String(intraday.regime || 'neutral').replaceAll('_', ' ')} - OFI ${Number(intraday.order_flow_imbalance || 0).toFixed(2)} - score ${Number(intraday.score || 0).toFixed(2)}${intraday.trade_window_active === false ? ' - outside prime window' : ''}`
    : (intraday.note || '');
  const dnote = document.getElementById('dnote');
  if(dnote) dnote.textContent = [risk.entry_note || '', intradayNote, metaNote, fitNote, allocNote].filter(Boolean).join(' | ');

  const reg = s.regime_context || {};
  const dreg = document.getElementById('dreg');
  if(dreg){
    dreg.innerHTML = Object.entries(reg).filter(([k]) => !k.includes('score')).map(([k,v]) => {
      const bg = (v === 'stressed' || v === 'risk-off') ? 'rgba(239,68,68,.1);color:var(--red)' : (v === 'calm' || v === 'risk-on') ? 'rgba(34,197,94,.1);color:var(--green)' : 'var(--border);color:var(--muted)';
      return `<span class="rp" style="background:${bg}">${k}: ${v}</span>`;
    }).join('');
  }

  const cf = s.counterfactual;
  const cfe = document.getElementById('dcf');
  if(cfe){
    if(cf){
      cfe.textContent = cf;
      cfe.style.display = 'block';
    }else{
      cfe.style.display = 'none';
    }
  }
  const dwarn = document.getElementById('dwarn');
  if(dwarn) dwarn.innerHTML = (s.macro_warnings || []).map(w => `<div class="wb">${w}</div>`).join('');
}

function renderWF(drivers){
  const el = document.getElementById('dwf');
  if(!el) return;
  if(!drivers || !drivers.length){
    el.innerHTML = '<div style="color:var(--muted);font-size:10px;padding:8px">Feature drivers are not available for this signal yet.</div>';
    return;
  }
  const mx = Math.max(...drivers.map(d => Math.abs(d.shap_value || d.value || 0)), 0.001);
  el.innerHTML = drivers.slice(0,10).map(d => {
    const v = d.shap_value || d.value || 0;
    const pct = (Math.abs(v) / mx * 100).toFixed(0);
    const isP = v > 0;
    const lb = d.label || d.feature || '-';
    return `<div class="wr"><div class="wl" title="${lb}">${lb}</div><div class="wt">
      ${isP ? `<div class="wp" style="width:${pct}%">${v > 0 ? '+' : ''}${v.toFixed(3)}</div>` : `<div class="wn" style="width:${pct}%">${v.toFixed(3)}</div>`}
    </div></div>`;
  }).join('');
}

function renderStress(data){
  const grid = document.getElementById('stg');
  const sum = document.getElementById('stsum');
  if(!grid || !sum) return;
  if(!data || data._note){
    grid.innerHTML = '<div class="empty" style="grid-column:1/-1">Computing...</div>';
    sum.textContent = '';
    return;
  }
  const entries = Object.entries(data).filter(([k,v]) => k !== 'summary' && v && typeof v === 'object');
  if(!entries.length){
    grid.innerHTML = '<div class="empty" style="grid-column:1/-1">No stress data yet</div>';
    sum.textContent = '';
    return;
  }
  grid.innerHTML = entries.map(([name, r]) => {
    const ret = Number(r.period_return_pct || 0);
    const mdd = Number(r.market_drawdown_pct || 0);
    const dd = Number(r.max_drawdown_pct || 0);
    const prot = Number(r.protection || 0);
    const dc = dd > -20 ? 'var(--green)' : 'var(--red)';
    return `<div class="sc ${r.survived ? 'pass' : 'fail'}">
      <div class="sn">${name}</div>
      <div class="sd" style="color:${dc}">${dd.toFixed(1)}% DD</div>
      <div class="sm">vs market ${mdd.toFixed(1)}%</div>
      ${prot > 0 ? `<div class="sm" style="color:var(--green)">+${prot.toFixed(0)}% protected</div>` : ''}
      <div class="sm" style="color:${ret >= 0 ? 'var(--green)' : 'var(--red)'}">${ret >= 0 ? '+' : ''}${ret.toFixed(1)}% - ${r.survived ? 'pass' : 'fail'}</div>
    </div>`;
  }).join('');
  const summary = data.summary || {};
  const msg = document.getElementById('msg');
  const msgs = document.getElementById('msgs');
  if(msg){
    msg.textContent = summary.overall_grade || '-';
    msg.style.color = (summary.overall_grade === 'A' || summary.overall_grade === 'B') ? 'var(--green)' : summary.overall_grade === 'C' ? 'var(--amber)' : 'var(--red)';
  }
  if(msgs) msgs.textContent = (summary.pass_rate_pct || 0) + '% passed - ' + (summary.avg_crisis_protection_pct || 0) + '% avg protection';
  sum.textContent = summary.assessment || '';
}

function renderDailyReport(data){
  reportData = data || reportData;
  const grid = document.getElementById('rgrid');
  const date = document.getElementById('rdate');
  const setupBody = document.getElementById('rsetup');
  const reasonBody = document.getElementById('rreasons');
  const timeBody = document.getElementById('rtimes');
  if(!grid || !date || !setupBody || !reasonBody || !timeBody || !reportData) return;
  const overall = reportData.overall || {};
  const lanes = reportData.lanes || [];
  date.textContent = overall.report_date || 'today';
  grid.innerHTML = lanes.map(lane => {
    const isActive = activeLane === lane.lane || (activeLane === 'all' && lane.lane === 'day') || (activeLane === 'reports' && lane.lane === 'day');
    return `<div class="laneCard${isActive ? ' active' : ''}">
      <div class="lanel">${lane.lane_label}</div>
      <div class="lanev" style="color:${(lane.day_pnl || 0) >= 0 ? 'var(--green)' : 'var(--red)'}">${lane.day_pnl >= 0 ? '+' : ''}$${Math.abs(lane.day_pnl || 0).toFixed(2)}</div>
      <div class="laneSub">
        ${lane.trade_ready || 0} trade-ready • ${lane.open_positions || 0} open positions<br>
        ${lane.closed_trades_today || 0} closed • ${lane.win_rate_pct || 0}% win • PF ${lane.profit_factor || 0}<br>
        status ${lane.status || 'paper'}
      </div>
    </div>`;
  }).join('');
  renderLaneOverview();

  const targetLane = activeLane === 'all' || activeLane === 'reports'
    ? (lanes.find(l => l.lane === 'day') || lanes[0])
    : (lanes.find(l => l.lane === activeLane) || lanes[0]);

  if(!targetLane){
    setupBody.innerHTML = '<tr><td colspan="5" style="color:var(--muted);padding:12px">No report data yet</td></tr>';
    reasonBody.innerHTML = '<tr><td colspan="3" style="color:var(--muted);padding:12px">No reason data yet</td></tr>';
    timeBody.innerHTML = '<tr><td colspan="3" style="color:var(--muted);padding:12px">No time-bucket data yet</td></tr>';
    return;
  }

  setupBody.innerHTML = (targetLane.setup_stats || []).length
    ? targetLane.setup_stats.map(row => `<tr><td>${row.setup_id}</td><td>${row.trades}</td><td>${row.win_rate_pct}%</td><td>${row.profit_factor}</td><td style="color:${row.pnl >= 0 ? 'var(--green)' : 'var(--red)'}">${row.pnl >= 0 ? '+' : ''}$${Math.abs(row.pnl).toFixed(2)}</td></tr>`).join('')
    : '<tr><td colspan="5" style="color:var(--muted);padding:12px">No setup data yet</td></tr>';
  reasonBody.innerHTML = (targetLane.exit_reasons || []).length
    ? targetLane.exit_reasons.map(row => `<tr><td>${row.reason}</td><td>${row.count}</td><td style="color:${row.pnl >= 0 ? 'var(--green)' : 'var(--red)'}">${row.pnl >= 0 ? '+' : ''}$${Math.abs(row.pnl).toFixed(2)}</td></tr>`).join('')
    : '<tr><td colspan="3" style="color:var(--muted);padding:12px">No exit data yet</td></tr>';
  timeBody.innerHTML = (targetLane.time_buckets || []).length
    ? targetLane.time_buckets.map(row => `<tr><td>${row.bucket}</td><td>${row.count}</td><td style="color:${row.pnl >= 0 ? 'var(--green)' : 'var(--red)'}">${row.pnl >= 0 ? '+' : ''}$${Math.abs(row.pnl).toFixed(2)}</td></tr>`).join('')
    : '<tr><td colspan="3" style="color:var(--muted);padding:12px">No time-bucket data yet</td></tr>';
}

function initChart(){
  if(!CHART_AVAILABLE) return;
  const canvas = document.getElementById('eqc');
  if(!canvas) return;
  const ctx = canvas.getContext('2d');
  const palette = themePalette();
  chart = new Chart(ctx, {
    type:'line',
    data:{labels:eq.map((_,i)=>i), datasets:[{data:eq, borderColor:palette.lineUp, borderWidth:1.5, fill:true, backgroundColor:palette.area, pointRadius:0, tension:.3}]},
    options:{responsive:true, maintainAspectRatio:false, plugins:{legend:{display:false}}, scales:{x:{display:false}, y:{ticks:{color:palette.axis, font:{size:9, family:'JetBrains Mono'}, callback:v=>'$'+Math.round(v).toLocaleString()}, grid:{color:palette.grid}}}}
  });
}

function updateChart(){
  if(!chart) return;
  const palette = themePalette();
  chart.data.labels = eq.map((_,i)=>i);
  chart.data.datasets[0].data = [...eq];
  chart.data.datasets[0].borderColor = eq[eq.length - 1] >= (eq[0] || eq[eq.length - 1]) ? palette.lineUp : palette.lineDown;
  chart.data.datasets[0].backgroundColor = palette.area;
  if(chart.options && chart.options.scales && chart.options.scales.y){
    chart.options.scales.y.ticks.color = palette.axis;
    chart.options.scales.y.grid.color = palette.grid;
  }
  chart.update('none');
}

function refreshPos(){
  fetch('/api/positions').then(r => r.json()).then(d => {
    const tb = document.getElementById('ptb');
    const ppc = document.getElementById('ppc');
    const ppm = document.getElementById('ppmeta');
    const shown = (d.positions || []).length;
    const total = d.total_count || 0;
    const open = d.open_positions != null ? Number(d.open_positions) : total;
    const target = d.target_open_positions != null ? Number(d.target_open_positions) : null;
    if(ppc){
      ppc.textContent = `${shown} shown / ${total} total`;
    }
    if(ppm){
      const cash = d.cash != null ? '$' + Number(d.cash).toLocaleString(undefined, {maximumFractionDigits:0}) : '-';
      ppm.textContent = target != null
        ? `${open} live / target ${target} positions • cash ${cash}`
        : `${open} live positions • cash ${cash}`;
    }
    if(!tb) return;
    if(!(d.positions || []).length){
      tb.innerHTML = '<tr><td colspan="5" style="color:var(--muted);padding:12px">No open positions</td></tr>';
      return;
    }
    tb.innerHTML = d.positions.map(p => {
      const pnl = p.unrealized_pnl || 0;
      const lane = p.lane ? ` <span style="color:var(--muted);font-size:9px">(${p.lane})</span>` : '';
      return `<tr><td style="font-weight:500">${p.symbol}${lane}</td><td>${fmtQty(p.quantity)}</td><td>$${Number(p.avg_cost || 0).toFixed(2)}</td><td>${p.current_price != null ? '$' + Number(p.current_price).toFixed(2) : '-'}</td><td style="color:${pnl >= 0 ? 'var(--green)' : 'var(--red)'}">${pnl >= 0 ? '+' : ''}$${Math.abs(pnl).toFixed(0)}</td></tr>`;
    }).join('');
  }).catch(() => {});
}

function fmtMoney(v){
  return v == null ? '-' : '$' + Number(v).toLocaleString(undefined, {maximumFractionDigits:2});
}

function refreshLivePortfolio(){
  fetch('/api/live-portfolio').then(r => r.json()).then(d => {
    const s = d.summary || {};
    const note = d.error || d.note || 'Connected';
    const lbc = document.getElementById('lbc');
    const lbv = document.getElementById('lbv');
    const lbcash = document.getElementById('lbcash');
    const lbhold = document.getElementById('lbhold');
    const lbpos = document.getElementById('lbpos');
    const lbpnl = document.getElementById('lbpnl');
    const lbn = document.getElementById('lbn');
    if(lbc) lbc.textContent = d.broker || 'Broker';
    if(lbv) lbv.textContent = fmtMoney(s.portfolio_value);
    if(lbcash) lbcash.textContent = fmtMoney(s.cash);
    if(lbhold) lbhold.textContent = fmtMoney(s.holdings_value);
    if(lbpos) lbpos.textContent = String(s.open_positions || 0);
    if(lbpnl) lbpnl.textContent = `Day P&L ${fmtMoney(s.day_pnl)}`;
    if(lbn) lbn.textContent = note;

    const tb = document.getElementById('lptb');
    const lbpc = document.getElementById('lbpc');
    if(lbpc){
      const shown = (d.positions || []).length;
      const total = d.total_positions || 0;
      lbpc.textContent = `${shown} shown / ${total} total`;
    }
    if(!tb) return;
    if(!(d.positions || []).length){
      tb.innerHTML = `<tr><td colspan="5" style="color:var(--muted);padding:12px">${note}</td></tr>`;
      return;
    }
    tb.innerHTML = d.positions.map(p => {
      const pnl = p.unrealized_pnl || 0;
      return `<tr><td style="font-weight:500">${p.symbol}</td><td>${p.source || 'live'}</td><td>${fmtQty(p.quantity || 0)}</td><td>${p.current_price != null ? '$' + Number(p.current_price).toFixed(2) : '-'}</td><td style="color:${pnl >= 0 ? 'var(--green)' : 'var(--red)'}">${pnl >= 0 ? '+' : ''}$${Math.abs(pnl).toFixed(0)}</td></tr>`;
    }).join('');
  }).catch(() => {});
}

function refreshExecutionTrace(){
  const counter = document.getElementById('exct');
  const tb = document.getElementById('extb');
  if(!counter || !tb) return;
  fetch('/api/execution-trace').then(r => r.json()).then(d => {
    const rows = d.events || [];
    counter.textContent = `${rows.length} shown / ${d.total_count || 0} total`;
    if(!rows.length){
      tb.innerHTML = '<tr><td colspan="5" style="color:var(--muted);padding:12px">No execution events yet</td></tr>';
      return;
    }
    tb.innerHTML = rows.map(ev => {
      const ts = ev.timestamp ? new Date(ev.timestamp).toLocaleTimeString() : '-';
      const st = (ev.status || '').toLowerCase();
      const sc = st === 'filled' ? 'var(--green)' : st === 'skipped' ? 'var(--amber)' : st === 'rejected' ? 'var(--red)' : 'var(--muted)';
      return `<tr>
        <td>${ts}</td>
        <td style="font-weight:500">${ev.symbol || '-'}</td>
        <td>${ev.action || '-'}</td>
        <td style="color:${sc}">${ev.status || '-'}</td>
        <td>${ev.reason || '-'}</td>
      </tr>`;
    }).join('');
  }).catch(() => {});
}

function refreshExecutionDivergence(){
  const counter = document.getElementById('sdct');
  const tb = document.getElementById('sdtb');
  const mode = document.getElementById('sdmode');
  const note = document.getElementById('sdnote');
  const route = document.getElementById('sdroute');
  const miss = document.getElementById('sdmiss');
  const fill = document.getElementById('sdfill');
  const lat = document.getElementById('sdlat');
  const part = document.getElementById('sdpart');
  const err = document.getElementById('sderr');
  if(!counter || !tb || !mode || !note || !route || !miss || !fill || !lat || !part || !err) return;
  fetch('/api/execution-divergence').then(r => r.json()).then(d => {
    const summary = d.summary || {};
    const sync = (d.shadow_router || {}).instrument_sync || {};
    const routePct = Number(summary.shadow_routeable_pct);
    mode.textContent = (d.shadow_router || {}).broker ? 'shadow live-checks' : 'paper only';
    route.textContent = Number.isFinite(routePct) ? `${routePct.toFixed(1)}%` : '-';
    miss.textContent = summary.missing_instrument_tokens ?? '-';
    fill.textContent = Number.isFinite(Number(summary.avg_fill_ratio_pct)) ? `${Number(summary.avg_fill_ratio_pct).toFixed(1)}%` : '-';
    lat.textContent = Number.isFinite(Number(summary.avg_simulated_latency_ms)) ? `${Number(summary.avg_simulated_latency_ms).toFixed(0)} ms` : '-';
    part.textContent = summary.partial_fill_count ?? '-';
    err.textContent = summary.shadow_errors ?? '-';
    const updatedAt = d.timestamp ? new Date(d.timestamp).toLocaleTimeString() : '-';
    note.textContent = `Updated ${updatedAt} | NSE tokens ${sync.resolved_symbols ?? 0}/${sync.tracked_symbols ?? 0} covered | missing ${(sync.missing_symbols ?? 0)}`;
    const rows = d.events || [];
    counter.textContent = `${rows.length} shown / ${d.total_count || 0} total`;
    if(!rows.length){
      tb.innerHTML = '<tr><td colspan="5" style="color:var(--muted);padding:12px">No shadow checks yet</td></tr>';
      return;
    }
    tb.innerHTML = rows.map(ev => {
      const ts = ev.timestamp ? new Date(ev.timestamp).toLocaleTimeString() : '-';
      const broker = ev.broker_status || '-';
      const shadow = ev.shadow_status || '-';
      const detail = ev.shadow_reason || (ev.partial_fill ? 'partial fill' : (ev.fill_ratio != null ? `fill ${Math.round(Number(ev.fill_ratio) * 100)}%` : '-'));
      const brokerColor = broker === 'filled' ? 'var(--green)' : broker === 'rejected' ? 'var(--red)' : 'var(--amber)';
      const shadowColor = shadow === 'shadow_ready' || shadow === 'submitted' ? 'var(--green)' : shadow === 'unsupported' || shadow === 'error' ? 'var(--red)' : 'var(--amber)';
      return `<tr>
        <td>${ts}</td>
        <td style="font-weight:500">${ev.symbol || '-'}</td>
        <td style="color:${brokerColor}">${broker}</td>
        <td style="color:${shadowColor}">${shadow}</td>
        <td>${detail}</td>
      </tr>`;
    }).join('');
  }).catch(() => {});
}

updateSortButton();
document.addEventListener('click', event => {
  const drawer = document.getElementById('menuDrawer');
  const trigger = document.querySelector('.menuBtn');
  if(!drawer || !drawer.classList.contains('open')) return;
  if(drawer.contains(event.target) || (trigger && trigger.contains(event.target))) return;
  toggleMenu(false);
});
document.addEventListener('keydown', event => {
  if(event.key === 'Escape') toggleMenu(false);
});

function refreshExecutionBacktest(){
  const exnote = document.getElementById('exnote');
  if(!exnote) return;
  const n = (v, d=2) => { const f = Number(v); return Number.isFinite(f) ? f.toFixed(d) : null; };
  fetch('/api/execution-backtest').then(r => r.json()).then(d => {
    if(d.note){
      exnote.textContent = d.note;
      return;
    }
    const ov = (d.overlay || {}).metrics || {};
    const bl = (d.baseline || {}).metrics || {};
    const up = d.uplift || {};
    const exor = document.getElementById('exor');
    const exbr = document.getElementById('exbr');
    const exos = document.getElementById('exos');
    const exus = document.getElementById('exus');
    const exod = document.getElementById('exod');
    const exur = document.getElementById('exur');
    const exlb = document.getElementById('exlb');
    if(!exor || !exbr || !exos || !exus || !exod || !exur || !exlb) return;
    const ovRet = n(ov.total_return_pct, 2);
    const blRet = n(bl.total_return_pct, 2);
    const ovSharpe = n(ov.sharpe_ratio, 3);
    const upSharpe = n(up.sharpe_ratio, 3);
    const ovDd = n(ov.max_drawdown_pct, 2);
    const upRet = n(up.total_return_pct, 2);
    exor.textContent = ovRet != null ? `${ovRet}%` : '-';
    exbr.textContent = blRet != null ? `${blRet}%` : '-';
    exos.textContent = ovSharpe != null ? ovSharpe : '-';
    exus.textContent = upSharpe != null ? `${Number(up.sharpe_ratio) > 0 ? '+' : ''}${upSharpe}` : '-';
    exod.textContent = ovDd != null ? `${ovDd}%` : '-';
    exur.textContent = upRet != null ? `${Number(up.total_return_pct) > 0 ? '+' : ''}${upRet}%` : '-';
    exlb.textContent = `${d.lookback_days || '-'}d window`;
    const ts = d.timestamp ? new Date(d.timestamp).toLocaleTimeString() : '-';
    exnote.textContent = `Updated ${ts} | overlay turnover ${((d.overlay || {}).avg_daily_turnover ?? '-')} | baseline turnover ${((d.baseline || {}).avg_daily_turnover ?? '-')}`;
  }).catch(() => {});
}

function refreshMetaModel(){
  const label = document.getElementById('mmeta');
  const sub = document.getElementById('mmetas');
  if(!label || !sub) return;
  fetch('/api/meta-model').then(r => r.json()).then(d => {
    if(d.note){
      label.textContent = '-';
      label.style.color = 'var(--muted)';
      sub.textContent = d.note;
      return;
    }
    const active = !!d.active;
    label.textContent = active ? 'ACTIVE' : 'FALLBACK';
    label.style.color = active ? 'var(--green)' : 'var(--amber)';
    const edge = d.mean_taken_edge_pct != null ? `${Number(d.mean_taken_edge_pct).toFixed(2)}% edge` : 'edge -';
    const hit = d.mean_taken_hit_rate_pct != null ? `${Number(d.mean_taken_hit_rate_pct).toFixed(1)}% hit` : 'hit -';
    const prec = d.mean_precision != null ? `${Number(d.mean_precision).toFixed(2)} precision` : 'precision -';
    sub.textContent = `${active ? 'trained meta' : 'heuristic fallback'} - ${edge} - ${hit} - ${prec}`;
  }).catch(() => {
    label.textContent = '-';
    label.style.color = 'var(--muted)';
    sub.textContent = 'meta status unavailable';
  });
}

function refreshLearningStatus(){
  const label = document.getElementById('learnv');
  const sub = document.getElementById('learns');
  if(!label || !sub) return;
  fetch('/api/learning-status').then(r => r.json()).then(d => {
    if(d.note){
      label.textContent = '-';
      label.style.color = 'var(--muted)';
      sub.textContent = d.note;
      return;
    }
    const active = d.enabled != null ? !!d.enabled : !!d.learning_enabled;
    const inProgress = !!d.retrain_in_progress;
    label.textContent = inProgress ? 'TRAINING' : (active ? 'ACTIVE' : 'OFF');
    label.style.color = inProgress ? 'var(--amber)' : (active ? 'var(--green)' : 'var(--muted)');
    const files = d.feature_store_files != null ? `${d.feature_store_files} feature files` : 'feature store -';
    const metaEdge = d.meta_edge_pct != null ? `${Number(d.meta_edge_pct).toFixed(2)}% edge` : 'edge -';
    sub.textContent = `${files} - ${metaEdge} - refresh ${Math.round((d.learning_refresh_seconds || 0) / 3600)}h`;
  }).catch(() => {
    label.textContent = '-';
    label.style.color = 'var(--muted)';
    sub.textContent = 'learning status unavailable';
  });
}

function refreshDailyReport(){
  fetch('/api/daily-report').then(r => r.json()).then(renderDailyReport).catch(() => {});
}

function refreshHealth(){
  fetch('/api/health').then(r => r.json()).then(d => {
    healthSnapshot = d || {};
    const s = d.uptime_seconds || 0;
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sc = s % 60;
    const hs = document.getElementById('hs');
    const hsi = document.getElementById('hsi');
    const ht = document.getElementById('ht');
    const hu = document.getElementById('hu');
    if(hs) hs.textContent = d.active_symbols != null ? d.active_symbols : '-';
    if(hsi && !Object.keys(signals).length){
      hsi.textContent = d.signal_count ? d.signal_count : ((d.active_symbols || d.tick_count) ? 'warming' : '0');
    }
    if(ht && d.timestamp) ht.textContent = new Date(d.timestamp).toLocaleTimeString();
    if(hu) hu.textContent = h > 0 ? `${h}h ${m}m` : m > 0 ? `${m}m ${sc}s` : `${sc}s`;
    updateMetrics();
    renderSidebar();
    renderMarketBoard();
  }).catch(() => {});
}

function refreshSignals(){
  fetch('/api/signals').then(r => r.json()).then(d => applySignals(d.signals || [], true)).catch(() => {});
}

function refreshPortfolioSummary(){
  fetch('/api/portfolio').then(r => r.json()).then(d => applyPortfolioSummary(d || {})).catch(() => {});
}

initTheme();
initChart();
updateLaneTabs();
renderLaneOverview();
setPage(activePage);
refreshSignals();
refreshPricesSnapshot();
refreshPortfolioSummary();
fetch('/api/stress-test').then(r => r.json()).then(renderStress).catch(() => {});
refreshPos();
refreshLivePortfolio();
refreshExecutionTrace();
refreshExecutionBacktest();
refreshExecutionDivergence();
refreshMetaModel();
refreshLearningStatus();
refreshDailyReport();
refreshHealth();
setInterval(refreshSignals, 10000);
setInterval(refreshPricesSnapshot, 5000);
setInterval(refreshPortfolioSummary, 10000);
setInterval(refreshPos, 5000);
setInterval(refreshLivePortfolio, 15000);
setInterval(refreshExecutionTrace, 5000);
setInterval(refreshExecutionBacktest, 30000);
setInterval(refreshExecutionDivergence, 7000);
setInterval(refreshMetaModel, 30000);
setInterval(refreshLearningStatus, 30000);
setInterval(refreshDailyReport, 15000);
setInterval(refreshHealth, 10000);
setInterval(() => fetch('/api/stress-test').then(r => r.json()).then(renderStress).catch(() => {}), 30000);
</script>
</body>
</html>
"""


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Standalone launcher
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
    )
    for lib in ["urllib3", "yfinance", "peewee", "transformers", "engineio", "socketio"]:
        logging.getLogger(lib).setLevel(logging.WARNING)

    # Load .env
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
        logger.info(f"Loaded .env from {env_path}")

    port = int(os.environ.get("PORT", 5050))

    print(
        "\n"
        "====================================================\n"
        "  MacroIntel - Full System\n"
        f"  Dashboard:   http://localhost:{port}\n"
        f"  Diagnostics: http://localhost:{port}/api/diagnostics\n"
        "====================================================\n"
    )

    try:
        from run import MacroIntelligenceSystem
        system = MacroIntelligenceSystem()
        system.start(run_dashboard=True, run_realtime=True,
                     run_backtest=True, dashboard_port=port)
    except ImportError as e:
        logger.warning(f"run.py not importable ({e}), starting dashboard-only mode...")

        from core.realtime_engine import PRICE_BUFFER, PollingFallback
        from core.paper_trading import VirtualBroker
        from core.signal_engine_v2 import RegimeAwareStressTester, MultiFactorScorer
        from core.explainability import SignalExplainer

        symbols = ["AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","JPM","SPY",
                   "QQQ","GLD","TLT","RELIANCE.NS","TCS.NS","INFY.NS","^NSEI","^VIX"]
        PollingFallback(symbols=symbols, interval_seconds=15, buffer=PRICE_BUFFER).start()
        broker  = VirtualBroker(initial_capital=100_000)
        sig_store: Dict = {}
        stress: Dict    = {}

        def _stress():
            r = RegimeAwareStressTester().run_regime_aware_stress_tests()
            stress.update(r)
        threading.Thread(target=_stress, daemon=True).start()

        def _infer():
            scorer   = MultiFactorScorer()
            explainer = SignalExplainer()
            while True:
                try:
                    from pipeline.earnings_collector import EarningsEventPipeline
                    from pipeline.price_collector import PriceDataPipeline
                    from pipeline.feature_engineering import FeaturePipeline
                    pd_res = PriceDataPipeline().run_incremental_update()
                    earn_res = EarningsEventPipeline().run(
                        symbols=list(pd_res.get("price_daily_recent", {}).keys()),
                        save=False,
                    )
                    fms    = FeaturePipeline().build_feature_matrix(
                        price_data=pd_res.get("price_daily_recent", {}),
                        earnings_data=earn_res.get("earnings_events"),
                    )
                    for sym, feats in fms.items():
                        if feats.empty: continue
                        row    = feats.iloc[-1]
                        scored = scorer.score(row)
                        sig    = explainer.explain_prediction(
                            symbol=sym, features=feats.iloc[[-1]],
                            model_predictions=scored["factor_scores"],
                            ensemble_score=scored["final_score"],
                            feature_names=[c for c in feats.columns if feats[c].dtype != object],
                        )
                        sig.update({"signal": scored["direction"], "confidence": scored["confidence"],
                                    "conviction_score": scored["conviction_score"],
                                    "regime": scored["regime"],
                                    "regime_multiplier": scored["regime_multiplier"],
                                    "factor_scores": scored["factor_scores"],
                                    "factor_weights": scored["factor_weights"]})
                        sig_store[sym] = sig
                    b = sum(1 for s in sig_store.values() if s.get("signal")=="buy")
                    s = sum(1 for s in sig_store.values() if s.get("signal")=="sell")
                    logger.info(f"Signals: {b} BUY | {s} SELL | {len(sig_store)-b-s} HOLD")
                except Exception as ex:
                    logger.error(f"Inference: {ex}")
                time.sleep(300)

        threading.Thread(target=_infer, daemon=True).start()
        app, socketio = create_app(price_buffer=PRICE_BUFFER, paper_broker=broker,
                                   signal_store=sig_store, stress_results=stress)
        socketio.run(app, host="0.0.0.0", port=port, debug=False, use_reloader=False)


