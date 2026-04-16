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
import math
import os
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.example", override=False)
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
_dashboard_trade_log_limit = max(25, int(os.getenv("DASHBOARD_TRADE_LOG_LIMIT", "250")))
_dashboard_win_rate_window = max(5, int(os.getenv("PAPER_WIN_RATE_WINDOW_TRADES", "50")))
_dashboard_win_rate_min_trades = max(1, int(os.getenv("PAPER_WIN_RATE_MIN_TRADES", "5")))
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
    alpha_quality: Optional[Dict] = None,
    execution_reconciliation: Optional[List[Dict]] = None,
    portfolio_overlay: Optional[Dict] = None,
    meta_model_status: Optional[Dict] = None,
    learning_status: Optional[Dict] = None,
    system_health: Optional[Dict] = None,
    runtime_controller=None,
    latency_monitor=None,
    security_suite=None,
) -> tuple:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "macro-intel-prod-2024")
    CORS(app)
    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode="threading",
        logger=False,
        engineio_logger=False,
        ping_timeout=60,
        ping_interval=25,
        max_http_buffer_size=10_000_000,
    )

    _signal_store   = signal_store   if signal_store   is not None else {}
    _stress_results = stress_results if stress_results is not None else {}
    _execution_trace = execution_trace if execution_trace is not None else []
    _execution_backtest = execution_backtest if execution_backtest is not None else {}
    _alpha_quality = alpha_quality if alpha_quality is not None else {}
    _execution_reconciliation = execution_reconciliation if execution_reconciliation is not None else []
    _portfolio_overlay = portfolio_overlay if portfolio_overlay is not None else {}
    _meta_model_status = meta_model_status if meta_model_status is not None else {}
    _learning_status = learning_status if learning_status is not None else {}
    _system_health = system_health if system_health is not None else {}
    _client_count   = [0]
    _dual_lane_variants_enabled = os.getenv(
        "DASHBOARD_DUAL_LANE_VARIANTS_ENABLED",
        os.getenv("DUAL_LANE_VARIANTS_ENABLED", "0"),
    ).strip().lower() in {"1", "true", "yes", "on"}
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

    def _json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): _json_safe(val) for key, val in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(item) for item in value]
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, (str, int, bool)) or value is None:
            return value
        if hasattr(value, "isoformat") and callable(getattr(value, "isoformat")):
            try:
                return value.isoformat()
            except Exception:
                pass
        if hasattr(value, "item") and callable(getattr(value, "item")):
            try:
                return _json_safe(value.item())
            except Exception:
                pass
        try:
            if value != value:
                return None
        except Exception:
            pass
        return str(value)

    def _snapshot_mapping(value: Any) -> Dict[str, Any]:
        if not isinstance(value, dict):
            try:
                return dict(value or {})
            except Exception:
                return {}
        # _signal_store can mutate while dashboard threads read it; retry with stable item snapshots.
        for _ in range(3):
            try:
                return value.copy()
            except RuntimeError:
                try:
                    return {k: v for k, v in list(value.items())}
                except RuntimeError:
                    continue
            except Exception:
                try:
                    return dict(value)
                except Exception:
                    return {}
        return {}

    def _signal_store_snapshot() -> Dict[str, Dict[str, Any]]:
        snapshot = _snapshot_mapping(_signal_store)
        return {
            str(symbol): dict(payload)
            for symbol, payload in snapshot.items()
            if isinstance(payload, dict)
        }

    def _to_datetime(value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        if not value:
            return None
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except Exception:
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)

    _trace_fallback_timestamp = datetime.fromtimestamp(0, tz=timezone.utc)

    def _normalize_trace_row(row: Any) -> Dict[str, Any]:
        if not isinstance(row, dict):
            return {}
        normalized = dict(row)
        parsed = _to_datetime(normalized.get("timestamp"))
        normalized["timestamp"] = (parsed or _trace_fallback_timestamp).astimezone(timezone.utc).isoformat()
        return normalized

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

    def _apply_meta_decision(payload: Dict[str, Any], evaluated: Dict[str, Any]) -> None:
        if not evaluated:
            return
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

    def _enrich_signal_row(
        row: Dict[str, Any],
        batch_meta: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
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
        if isinstance(meta, dict) and "take_trade" in meta:
            pass
        elif batch_meta is not None and symbol in batch_meta:
            _apply_meta_decision(payload, batch_meta[symbol])
        elif not (isinstance(meta, dict) and "take_trade" in meta):
            try:
                evaluated = _get_dashboard_meta_engine().evaluate_universe({symbol: payload}).get(symbol, {})
            except Exception as exc:
                logger.debug(f"Dashboard meta hydration failed for {symbol}: {exc}")
                evaluated = {}
            if evaluated:
                _apply_meta_decision(payload, evaluated)

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

    def _build_symbol_chart(symbol: str, window_seconds: int, max_points: int) -> Dict[str, Any]:
        symbol = str(symbol or "").strip().upper()
        latest = _latest_price_payload(symbol)
        points: List[Dict[str, Any]] = []

        if price_buffer and symbol:
            ticks = price_buffer.recent_ticks(symbol, n=max(max_points * 6, 240))
            if ticks:
                cutoff = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)
                recent = [tick for tick in ticks if tick.timestamp >= cutoff]
                if not recent:
                    recent = ticks[-max_points:]
                if len(recent) > max_points:
                    step = max(1, (len(recent) + max_points - 1) // max_points)
                    sampled = recent[::step]
                    if sampled[-1] is not recent[-1]:
                        sampled.append(recent[-1])
                    recent = sampled[-max_points:]
                label_fmt = "%H:%M:%S" if window_seconds <= 900 else "%H:%M" if window_seconds <= 86400 else "%m-%d %H:%M"
                points = [
                    {
                        "time": tick.timestamp.isoformat(),
                        "label": tick.timestamp.astimezone(timezone.utc).strftime(label_fmt),
                        "price": round(float(tick.price), 4),
                        "volume": int(getattr(tick, "volume", 0) or 0),
                    }
                    for tick in recent
                ]

        if not points and latest.get("last_price") is not None:
            now = datetime.now(timezone.utc)
            points = [{
                "time": now.isoformat(),
                "label": now.strftime("%H:%M:%S"),
                "price": round(float(latest["last_price"]), 4),
                "volume": 0,
            }]

        return {
            "symbol": symbol,
            "market": latest.get("market"),
            "price": latest.get("last_price"),
            "change_pct": latest.get("price_change_pct"),
            "points": points,
            "point_count": len(points),
            "window_seconds": window_seconds,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

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

    def _dashboard_portfolio_summary() -> Dict[str, Any]:
        if paper_broker is None:
            return {
                "initial_capital": 100000,
                "current_portfolio_value": 100000,
                "total_return_pct": 0,
                "cash": 100000,
                "open_positions": 0,
                "total_trades": 0,
                "win_rate_pct": 0,
                "kill_switch_active": False,
                "positions": {},
                "note": "paper_broker not connected",
            }
        summary = _reprice_paper_broker_summary(paper_broker.get_summary())
        trade_summary = (_build_trade_log().get("summary") or {})
        if trade_summary:
            summary["closed_trades"] = trade_summary.get("closed_trades", summary.get("closed_trades", 0))
            summary["winning_closed_trades"] = trade_summary.get("winning_trades", summary.get("winning_closed_trades", 0))
            summary["losing_closed_trades"] = trade_summary.get("losing_trades", summary.get("losing_closed_trades", 0))
            summary["breakeven_closed_trades"] = trade_summary.get("breakeven_trades", summary.get("breakeven_closed_trades", 0))
            summary["lifetime_win_rate_pct"] = trade_summary.get("lifetime_win_rate_pct", summary.get("lifetime_win_rate_pct", 0.0))
            summary["recent_win_rate_pct"] = trade_summary.get("recent_win_rate_pct", summary.get("recent_win_rate_pct", 0.0))
            summary["recent_closed_trades"] = trade_summary.get("recent_closed_trades", summary.get("recent_closed_trades", 0))
            summary["session_win_rate_pct"] = trade_summary.get("session_win_rate_pct", summary.get("session_win_rate_pct", 0.0))
            summary["session_closed_trades"] = trade_summary.get("session_closed_trades", summary.get("session_closed_trades", 0))
            summary["win_rate_pct"] = trade_summary.get("win_rate_pct", summary.get("win_rate_pct", 0.0))
            summary["win_rate_basis"] = trade_summary.get("win_rate_basis", summary.get("win_rate_basis", "lifetime"))
            summary["win_rate_label"] = trade_summary.get("win_rate_label", summary.get("win_rate_label", "closed paper trades"))
            summary["win_rate_sample_size"] = trade_summary.get("win_rate_sample_size", summary.get("win_rate_sample_size", 0))
            summary["trade_log_total_count"] = trade_summary.get("closed_trades", 0) + trade_summary.get("open_trades", 0)
            summary["trade_lane_breakdown"] = trade_summary.get("lane_breakdown", {})
        summary["return_periods"] = _build_return_periods(summary)
        return summary

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

    def _flatten_signal_store(signal_snapshot: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Dict]:
        rows: List[Dict] = []
        seen_keys: set[str] = set()
        snapshot = signal_snapshot if signal_snapshot is not None else _signal_store_snapshot()
        batch_meta: Dict[str, Dict[str, Any]] = {}
        try:
            if snapshot:
                batch_meta = _get_dashboard_meta_engine().evaluate_universe(dict(snapshot))
        except Exception as exc:
            logger.debug(f"Dashboard batch meta evaluation failed: {exc}")

        for raw in snapshot.values():
            symbol = str(raw.get("symbol") or "")
            if not symbol:
                continue
            actual = _enrich_signal_row(dict(raw), batch_meta=batch_meta)
            actual_key = str(actual.get("signal_key") or f"{symbol}::{actual.get('lane', 'normal')}")
            if actual_key not in seen_keys:
                rows.append(actual)
                seen_keys.add(actual_key)
            if not _dual_lane_variants_enabled:
                continue
            normal_lane = raw.get("normal_lane_signal")
            if isinstance(normal_lane, dict):
                variant_payload = dict(normal_lane)
                variant_payload["symbol"] = symbol
                variant_payload["lane"] = "normal"
                variant_payload.setdefault("signal_key", f"{symbol}::normal")
                variant = _enrich_signal_row(variant_payload)
                variant["lane"] = "normal"
                variant["lane_label"] = _lane_label("normal")
                variant["signal_key"] = variant.get("signal_key") or f"{symbol}::normal"
                variant["scenario"] = variant.get("lane_label")
                variant_key = str(variant.get("signal_key"))
                if variant_key not in seen_keys:
                    rows.append(variant)
                    seen_keys.add(variant_key)
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
        signal_snapshot = _signal_store_snapshot()
        signal_rows = _flatten_signal_store(signal_snapshot)
        trades_today = _today_trade_rows()
        current_positions = _snapshot_mapping(getattr(paper_broker, "positions", {})) if paper_broker is not None else {}
        lane_positions = defaultdict(int)
        lane_open_unrealized = defaultdict(float)
        for position_key, position in current_positions.items():
            quantity = _safe_float(getattr(position, "quantity", None), _safe_float((position or {}).get("quantity"), 0.0) if isinstance(position, dict) else 0.0)
            if quantity <= 0:
                continue
            symbol = str(
                getattr(position, "symbol", None)
                or ((position or {}).get("symbol") if isinstance(position, dict) else None)
                or position_key
            )
            signal = signal_snapshot.get(symbol, {})
            lane = _infer_lane(symbol, payload=signal, metadata={"position_key": position_key})
            lane_positions[lane] += 1
            unrealized_pnl = _safe_float(
                getattr(position, "unrealized_pnl", None),
                _safe_float((position or {}).get("unrealized_pnl"), 0.0) if isinstance(position, dict) else 0.0,
            )
            lane_open_unrealized[lane] += unrealized_pnl

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
        total_open_unrealized_pnl = 0.0
        total_closed = 0
        total_fills = 0
        for lane in ("normal", "day", "crypto"):
            rows = lane_trade_rows.get(lane, [])
            fills = len(rows)
            closed_rows = [r for r in rows if abs(_safe_float(r.get("realized_pnl"), 0.0)) > 1e-9]
            wins = [r for r in closed_rows if _safe_float(r.get("realized_pnl"), 0.0) > 0]
            losses = [r for r in closed_rows if _safe_float(r.get("realized_pnl"), 0.0) < 0]
            day_pnl = round(sum(_safe_float(r.get("realized_pnl"), 0.0) for r in closed_rows), 2)
            open_unrealized_pnl = round(lane_open_unrealized.get(lane, 0.0), 2)
            total_day_pnl += day_pnl
            total_open_unrealized_pnl += open_unrealized_pnl
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
                    "open_unrealized_pnl": open_unrealized_pnl,
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
            "total_open_unrealized_pnl": round(total_open_unrealized_pnl, 2),
            "fills_today": total_fills,
            "closed_trades_today": total_closed,
            "open_positions": sum(lane_positions.values()),
            "kill_switch_active": bool(getattr(paper_broker, "_kill_switch_activated", False)) if paper_broker is not None else False,
            "kill_switch_reason": getattr(paper_broker, "_kill_switch_reason", "") if paper_broker is not None else "",
        }
        return {"overall": overall, "lanes": lanes}

    def _resolved_lane(symbol: str, position_key: Optional[str] = None, payload: Optional[Dict] = None, metadata: Optional[Dict] = None) -> str:
        key = str(position_key or "").strip()
        if "::" in key:
            return key.rsplit("::", 1)[-1].lower()
        return _infer_lane(symbol, payload=payload, metadata=metadata)

    def _is_crypto_record(
        symbol: str,
        *,
        position_key: Optional[str] = None,
        payload: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        market: Any = None,
    ) -> bool:
        lane = _resolved_lane(symbol, position_key=position_key, payload=payload, metadata=metadata)
        if lane == "crypto":
            return True
        market_text = " ".join(
            part
            for part in (
                str(market or "").strip(),
                str((metadata or {}).get("market") or "").strip(),
                str((payload or {}).get("market") or "").strip(),
            )
            if part
        ).lower()
        if any(token in market_text for token in ("crypto", "binance", "bybit")):
            return True
        clean_symbol = str(symbol or "").strip().upper()
        return clean_symbol.endswith(("USDT", "USDC", "BUSD", "FDUSD", "TUSD", "USDE"))

    def _build_trade_log(lane_filter: Optional[str] = None, display_limit: Optional[int] = _dashboard_trade_log_limit) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        empty_payload = {
            "summary": {
                "fill_events": 0,
                "closed_trades": 0,
                "open_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "breakeven_trades": 0,
                "win_rate_pct": 0.0,
                "lifetime_win_rate_pct": 0.0,
                "recent_win_rate_pct": 0.0,
                "recent_closed_trades": 0,
                "session_win_rate_pct": 0.0,
                "session_closed_trades": 0,
                "gross_closed_pnl": 0.0,
                "net_closed_pnl": 0.0,
                "gross_open_pnl": 0.0,
                "net_open_pnl": 0.0,
                "gross_unrealized_pnl": 0.0,
                "net_unrealized_pnl": 0.0,
                "total_commission": 0.0,
                "avg_hold_minutes": None,
                "best_trade_net_pnl": None,
                "best_trade_symbol": "",
                "worst_trade_net_pnl": None,
                "worst_trade_symbol": "",
                "last_trade_at": None,
                "win_rate_label": "waiting for closed paper trades",
                "win_rate_basis": "lifetime",
                "win_rate_sample_size": 0,
                "lane_breakdown": {},
            },
            "trades": [],
            "count": 0,
            "total_count": 0,
            "display_limit": display_limit,
            "timestamp": now.isoformat(),
        }
        if paper_broker is None:
            empty_payload["note"] = "paper_broker not connected"
            return empty_payload

        summary = _reprice_paper_broker_summary(paper_broker.get_summary())
        open_positions = dict(summary.get("positions") or {})
        trade_log = sorted(
            [row for row in list(getattr(paper_broker, "trade_log", []) or []) if isinstance(row, dict)],
            key=lambda row: _to_datetime(row.get("filled_at")) or _trace_fallback_timestamp,
        )
        if not trade_log and not open_positions:
            return empty_payload

        def _new_cycle(
            *,
            symbol: str,
            position_key: str,
            lane: str,
            market: str,
            opening_side: str,
            opened_at: datetime,
            metadata: Dict[str, Any],
            signal_source: str,
        ) -> Dict[str, Any]:
            return {
                "symbol": symbol,
                "position_key": position_key,
                "lane": lane,
                "lane_label": _lane_label(lane),
                "market": market,
                "opening_side": opening_side,
                "entry_time": opened_at.isoformat(),
                "exit_time": None,
                "last_fill_time": opened_at.isoformat(),
                "entry_qty": 0.0,
                "exit_qty": 0.0,
                "open_qty": 0.0,
                "entry_notional": 0.0,
                "exit_notional": 0.0,
                "entry_commission": 0.0,
                "exit_commission": 0.0,
                "gross_realized_pnl": 0.0,
                "entry_fill_count": 0,
                "exit_fill_count": 0,
                "fill_count": 0,
                "partial_fill_count": 0,
                "slippage_pct_sum": 0.0,
                "slippage_pct_count": 0,
                "fill_ratio_sum": 0.0,
                "fill_ratio_count": 0,
                "entry_reason": str(metadata.get("entry_reason") or signal_source or ""),
                "exit_reason": str(metadata.get("exit_reason") or metadata.get("action_reason") or ""),
                "setup_id": str(metadata.get("setup_id") or metadata.get("signal_style") or signal_source or ""),
                "signal_style": str(metadata.get("signal_style") or ""),
                "time_bucket": str(metadata.get("entry_time_bucket") or metadata.get("time_bucket") or ""),
                "entry_regime": str(metadata.get("entry_regime") or metadata.get("regime") or ""),
                "execution_mode": str(metadata.get("execution_mode") or ""),
                "signal_source": str(signal_source or ""),
                "entry_take_probability": _safe_float(metadata.get("entry_take_probability"), None),
                "entry_expected_edge_pct": _safe_float(metadata.get("entry_expected_edge_pct"), None),
                "entry_expected_drawdown_pct": _safe_float(metadata.get("entry_expected_drawdown_pct"), None),
            }

        def _finalize_cycle(cycle: Dict[str, Any]) -> Dict[str, Any]:
            position_key = str(cycle.get("position_key") or "")
            position = dict(open_positions.get(position_key) or {})
            qty_open = abs(_safe_float(position.get("quantity"), cycle.get("open_qty", 0.0)))
            entry_qty = _safe_float(cycle.get("entry_qty"), 0.0)
            exit_qty = _safe_float(cycle.get("exit_qty"), 0.0)
            entry_notional = _safe_float(cycle.get("entry_notional"), 0.0)
            exit_notional = _safe_float(cycle.get("exit_notional"), 0.0)
            avg_entry_price = (entry_notional / entry_qty) if entry_qty > 0 else None
            avg_exit_price = (exit_notional / exit_qty) if exit_qty > 0 else None
            current_price = _safe_float(position.get("current_price"), None)
            if current_price is None and avg_entry_price is not None and qty_open > 0:
                current_price = avg_entry_price + (_safe_float(position.get("unrealized_pnl"), 0.0) / max(qty_open, 1e-9))

            if cycle.get("opening_side") == "sell":
                gross_unrealized = (avg_entry_price - current_price) * qty_open if current_price is not None and avg_entry_price is not None else 0.0
                avg_buy_price = avg_exit_price
                avg_sell_price = avg_entry_price
            else:
                gross_unrealized = _safe_float(position.get("unrealized_pnl"), 0.0)
                avg_buy_price = avg_entry_price
                avg_sell_price = avg_exit_price

            gross_realized = _safe_float(cycle.get("gross_realized_pnl"), 0.0)
            total_commission = _safe_float(cycle.get("entry_commission"), 0.0) + _safe_float(cycle.get("exit_commission"), 0.0)
            gross_pnl = gross_realized + gross_unrealized
            net_pnl = gross_pnl - total_commission
            entry_time = _to_datetime(cycle.get("entry_time")) or _trace_fallback_timestamp
            exit_time = _to_datetime(cycle.get("exit_time"))
            last_fill_time = _to_datetime(cycle.get("last_fill_time")) or entry_time
            status = "open" if qty_open > 1e-9 else "closed"
            hold_end = exit_time if exit_time is not None else now
            hold_minutes = max(0.0, round((hold_end - entry_time).total_seconds() / 60.0, 1))
            avg_slippage_pct = (
                _safe_float(cycle.get("slippage_pct_sum"), 0.0) / max(int(cycle.get("slippage_pct_count") or 0), 1)
                if int(cycle.get("slippage_pct_count") or 0) > 0
                else None
            )
            avg_fill_ratio_pct = (
                (_safe_float(cycle.get("fill_ratio_sum"), 0.0) / max(int(cycle.get("fill_ratio_count") or 0), 1)) * 100.0
                if int(cycle.get("fill_ratio_count") or 0) > 0
                else None
            )
            display_qty = exit_qty if status == "closed" and exit_qty > 0 else qty_open

            return {
                "symbol": cycle.get("symbol"),
                "position_key": position_key,
                "lane": cycle.get("lane"),
                "lane_label": cycle.get("lane_label"),
                "market": cycle.get("market"),
                "status": status,
                "position_direction": "short" if cycle.get("opening_side") == "sell" else "long",
                "entry_time": entry_time.isoformat(),
                "exit_time": exit_time.isoformat() if exit_time is not None else None,
                "last_fill_time": last_fill_time.isoformat(),
                "hold_minutes": hold_minutes,
                "entry_qty": round(entry_qty, 8),
                "exit_qty": round(exit_qty, 8),
                "open_qty": round(qty_open, 8),
                "quantity": round(display_qty, 8),
                "avg_entry_price": round(avg_entry_price, 8) if avg_entry_price is not None else None,
                "avg_exit_price": round(avg_exit_price, 8) if avg_exit_price is not None else None,
                "avg_buy_price": round(avg_buy_price, 8) if avg_buy_price is not None else None,
                "avg_sell_price": round(avg_sell_price, 8) if avg_sell_price is not None else None,
                "current_price": round(current_price, 8) if current_price is not None else None,
                "gross_realized_pnl": round(gross_realized, 2),
                "gross_unrealized_pnl": round(gross_unrealized, 2),
                "gross_pnl": round(gross_pnl, 2),
                "net_pnl": round(net_pnl, 2),
                "net_return_pct": round((net_pnl / entry_notional) * 100.0, 2) if entry_notional > 0 else None,
                "gross_return_pct": round((gross_pnl / entry_notional) * 100.0, 2) if entry_notional > 0 else None,
                "commission": round(total_commission, 2),
                "entry_commission": round(_safe_float(cycle.get("entry_commission"), 0.0), 2),
                "exit_commission": round(_safe_float(cycle.get("exit_commission"), 0.0), 2),
                "entry_notional": round(entry_notional, 2),
                "exit_notional": round(exit_notional, 2),
                "fill_count": int(cycle.get("fill_count") or 0),
                "entry_fill_count": int(cycle.get("entry_fill_count") or 0),
                "exit_fill_count": int(cycle.get("exit_fill_count") or 0),
                "partial_fill_count": int(cycle.get("partial_fill_count") or 0),
                "avg_slippage_pct": round(avg_slippage_pct, 4) if avg_slippage_pct is not None else None,
                "avg_fill_ratio_pct": round(avg_fill_ratio_pct, 2) if avg_fill_ratio_pct is not None else None,
                "entry_reason": cycle.get("entry_reason"),
                "exit_reason": cycle.get("exit_reason"),
                "setup_id": cycle.get("setup_id"),
                "signal_style": cycle.get("signal_style"),
                "time_bucket": cycle.get("time_bucket"),
                "entry_regime": cycle.get("entry_regime"),
                "execution_mode": cycle.get("execution_mode"),
                "signal_source": cycle.get("signal_source"),
                "entry_take_probability": cycle.get("entry_take_probability"),
                "entry_expected_edge_pct": cycle.get("entry_expected_edge_pct"),
                "entry_expected_drawdown_pct": cycle.get("entry_expected_drawdown_pct"),
            }

        active_cycles: Dict[str, Dict[str, Any]] = {}
        rows: List[Dict[str, Any]] = []
        crypto_fill_events = 0

        for trade in trade_log:
            symbol = str(trade.get("symbol") or "").strip().upper()
            metadata = dict(trade.get("metadata") or {})
            position_key = str(trade.get("position_key") or metadata.get("position_key") or symbol)
            lane = _resolved_lane(symbol, position_key=position_key, payload=trade, metadata=metadata)
            market = str(metadata.get("market") or trade.get("market") or "")
            if lane_filter == "crypto" and not _is_crypto_record(
                symbol,
                position_key=position_key,
                payload=trade,
                metadata=metadata,
                market=market,
            ):
                continue
            if lane_filter in {"normal", "day"} and lane != lane_filter:
                continue

            side = str(trade.get("side") or "").strip().lower()
            qty = abs(_safe_float(trade.get("quantity"), 0.0))
            if side not in {"buy", "sell"} or qty <= 0:
                continue

            crypto_fill_events += 1
            filled_at = _to_datetime(trade.get("filled_at")) or _trace_fallback_timestamp
            fill_price = _safe_float(trade.get("fill_price"), 0.0)
            commission = _safe_float(trade.get("commission"), 0.0)
            realized_pnl = _safe_float(trade.get("realized_pnl"), 0.0)
            slippage_pct = _safe_float(trade.get("slippage_pct"), None)
            fill_ratio = _safe_float(trade.get("fill_ratio"), None)
            signal_source = str(trade.get("signal_source") or "")

            cycle = active_cycles.get(position_key)
            if cycle is None:
                cycle = _new_cycle(
                    symbol=symbol,
                    position_key=position_key,
                    lane=lane,
                    market=market,
                    opening_side=side,
                    opened_at=filled_at,
                    metadata=metadata,
                    signal_source=signal_source,
                )
                active_cycles[position_key] = cycle

            cycle["last_fill_time"] = filled_at.isoformat()
            cycle["fill_count"] += 1
            if slippage_pct is not None:
                cycle["slippage_pct_sum"] += slippage_pct
                cycle["slippage_pct_count"] += 1
            if fill_ratio is not None:
                cycle["fill_ratio_sum"] += fill_ratio
                cycle["fill_ratio_count"] += 1
            if bool(trade.get("partial_fill") or metadata.get("partial_fill")):
                cycle["partial_fill_count"] += 1

            if not cycle.get("entry_reason"):
                cycle["entry_reason"] = str(metadata.get("entry_reason") or signal_source or "")
            if not cycle.get("setup_id"):
                cycle["setup_id"] = str(metadata.get("setup_id") or metadata.get("signal_style") or signal_source or "")
            if not cycle.get("signal_style"):
                cycle["signal_style"] = str(metadata.get("signal_style") or "")
            if not cycle.get("time_bucket"):
                cycle["time_bucket"] = str(metadata.get("entry_time_bucket") or metadata.get("time_bucket") or "")
            if not cycle.get("entry_regime"):
                cycle["entry_regime"] = str(metadata.get("entry_regime") or metadata.get("regime") or "")
            if not cycle.get("execution_mode"):
                cycle["execution_mode"] = str(metadata.get("execution_mode") or "")

            if side == cycle.get("opening_side"):
                cycle["entry_qty"] += qty
                cycle["open_qty"] += qty
                cycle["entry_notional"] += qty * fill_price
                cycle["entry_commission"] += commission
                cycle["entry_fill_count"] += 1
            else:
                open_qty_before = _safe_float(cycle.get("open_qty"), 0.0)
                closed_qty = min(open_qty_before, qty) if open_qty_before > 0 else qty
                commission_ratio = (closed_qty / qty) if qty > 0 else 1.0
                cycle["exit_qty"] += closed_qty
                cycle["open_qty"] = max(0.0, open_qty_before - closed_qty)
                cycle["exit_notional"] += closed_qty * fill_price
                cycle["exit_commission"] += commission * commission_ratio
                cycle["gross_realized_pnl"] += realized_pnl
                cycle["exit_fill_count"] += 1
                cycle["exit_time"] = filled_at.isoformat()
                cycle["exit_reason"] = str(metadata.get("exit_reason") or metadata.get("action_reason") or cycle.get("exit_reason") or "")

                remainder_qty = max(0.0, qty - closed_qty)
                if cycle["open_qty"] <= 1e-9:
                    rows.append(_finalize_cycle(cycle))
                    active_cycles.pop(position_key, None)
                    if remainder_qty > 1e-9:
                        next_cycle = _new_cycle(
                            symbol=symbol,
                            position_key=position_key,
                            lane=lane,
                            market=market,
                            opening_side=side,
                            opened_at=filled_at,
                            metadata=metadata,
                            signal_source=signal_source,
                        )
                        next_cycle["entry_qty"] += remainder_qty
                        next_cycle["open_qty"] += remainder_qty
                        next_cycle["entry_notional"] += remainder_qty * fill_price
                        next_cycle["entry_commission"] += commission * (1.0 - commission_ratio)
                        next_cycle["entry_fill_count"] += 1
                        next_cycle["fill_count"] += 1
                        if slippage_pct is not None:
                            next_cycle["slippage_pct_sum"] += slippage_pct
                            next_cycle["slippage_pct_count"] += 1
                        if fill_ratio is not None:
                            next_cycle["fill_ratio_sum"] += fill_ratio
                            next_cycle["fill_ratio_count"] += 1
                        if bool(trade.get("partial_fill") or metadata.get("partial_fill")):
                            next_cycle["partial_fill_count"] += 1
                        active_cycles[position_key] = next_cycle

        for cycle in active_cycles.values():
            rows.append(_finalize_cycle(cycle))

        rows.sort(key=lambda row: _to_datetime(row.get("last_fill_time")) or _trace_fallback_timestamp, reverse=True)
        limited_rows = rows[:display_limit] if display_limit else list(rows)

        closed_rows = [row for row in rows if row.get("status") == "closed"]
        open_rows = [row for row in rows if row.get("status") == "open"]
        winners = [row for row in closed_rows if _safe_float(row.get("net_pnl"), 0.0) > 0]
        losers = [row for row in closed_rows if _safe_float(row.get("net_pnl"), 0.0) < 0]
        breakeven = [
            row
            for row in closed_rows
            if abs(_safe_float(row.get("net_pnl"), 0.0)) <= 1e-9
        ]
        holds = [row.get("hold_minutes") for row in closed_rows if row.get("hold_minutes") is not None]
        best_trade = max(closed_rows, key=lambda row: _safe_float(row.get("net_pnl"), float("-inf")), default=None)
        worst_trade = min(closed_rows, key=lambda row: _safe_float(row.get("net_pnl"), float("inf")), default=None)

        def _cycle_win_rate_snapshot(source_rows: List[Dict[str, Any]], basis: str) -> Dict[str, Any]:
            wins = sum(1 for row in source_rows if _safe_float(row.get("net_pnl"), 0.0) > 0)
            losses = sum(1 for row in source_rows if _safe_float(row.get("net_pnl"), 0.0) < 0)
            flat = sum(1 for row in source_rows if abs(_safe_float(row.get("net_pnl"), 0.0)) <= 1e-9)
            closed_count = len(source_rows)
            labels = {
                "recent": f"last {closed_count} closed paper trades",
                "session": f"today | {closed_count} closed paper trades",
                "lifetime": f"lifetime | {closed_count} closed paper trades",
            }
            return {
                "basis": basis,
                "label": labels.get(basis, f"{closed_count} closed paper trades"),
                "closed_trades": closed_count,
                "wins": wins,
                "losses": losses,
                "breakeven": flat,
                "win_rate_pct": round((wins / max(closed_count, 1)) * 100.0, 1) if closed_count else 0.0,
            }

        recent_closed_rows = closed_rows[:_dashboard_win_rate_window]
        session_closed_rows = []
        for row in closed_rows:
            close_ts = _to_datetime(row.get("exit_time") or row.get("last_fill_time"))
            if close_ts and close_ts.astimezone(timezone.utc).date() == now.date():
                session_closed_rows.append(row)

        lifetime_win_rate = _cycle_win_rate_snapshot(closed_rows, "lifetime")
        recent_win_rate = _cycle_win_rate_snapshot(recent_closed_rows, "recent")
        session_win_rate = _cycle_win_rate_snapshot(session_closed_rows, "session")
        headline_win_rate = lifetime_win_rate
        if recent_win_rate["closed_trades"] >= _dashboard_win_rate_min_trades:
            headline_win_rate = recent_win_rate
        elif session_win_rate["closed_trades"] >= _dashboard_win_rate_min_trades:
            headline_win_rate = session_win_rate

        lane_breakdown = {}
        for lane_name in ("normal", "day", "crypto"):
            lane_rows = [row for row in rows if str(row.get("lane") or "").lower() == lane_name]
            lane_closed = [row for row in lane_rows if row.get("status") == "closed"]
            lane_open = [row for row in lane_rows if row.get("status") == "open"]
            lane_winners = [row for row in lane_closed if _safe_float(row.get("net_pnl"), 0.0) > 0]
            lane_breakdown[lane_name] = {
                "lane": lane_name,
                "lane_label": _lane_label(lane_name),
                "trade_count": len(lane_rows),
                "closed_trades": len(lane_closed),
                "open_trades": len(lane_open),
                "win_rate_pct": round((len(lane_winners) / max(len(lane_closed), 1)) * 100.0, 1) if lane_closed else 0.0,
                "net_pnl": round(sum(_safe_float(row.get("net_pnl"), 0.0) for row in lane_rows), 2),
            }

        return {
            "summary": {
                "fill_events": crypto_fill_events,
                "closed_trades": len(closed_rows),
                "open_trades": len(open_rows),
                "winning_trades": len(winners),
                "losing_trades": len(losers),
                "breakeven_trades": len(breakeven),
                "win_rate_pct": headline_win_rate["win_rate_pct"],
                "lifetime_win_rate_pct": lifetime_win_rate["win_rate_pct"],
                "recent_win_rate_pct": recent_win_rate["win_rate_pct"],
                "recent_closed_trades": recent_win_rate["closed_trades"],
                "session_win_rate_pct": session_win_rate["win_rate_pct"],
                "session_closed_trades": session_win_rate["closed_trades"],
                "gross_closed_pnl": round(sum(_safe_float(row.get("gross_pnl"), 0.0) for row in closed_rows), 2),
                "net_closed_pnl": round(sum(_safe_float(row.get("net_pnl"), 0.0) for row in closed_rows), 2),
                "gross_open_pnl": round(sum(_safe_float(row.get("gross_pnl"), 0.0) for row in open_rows), 2),
                "net_open_pnl": round(sum(_safe_float(row.get("net_pnl"), 0.0) for row in open_rows), 2),
                "gross_unrealized_pnl": round(sum(_safe_float(row.get("gross_unrealized_pnl"), 0.0) for row in open_rows), 2),
                "net_unrealized_pnl": round(
                    sum(_safe_float(row.get("gross_unrealized_pnl"), 0.0) for row in open_rows)
                    - sum(_safe_float(row.get("commission"), 0.0) for row in open_rows),
                    2,
                ),
                "total_commission": round(sum(_safe_float(row.get("commission"), 0.0) for row in rows), 2),
                "avg_hold_minutes": round(sum(float(value) for value in holds) / len(holds), 1) if holds else None,
                "best_trade_net_pnl": round(_safe_float((best_trade or {}).get("net_pnl"), 0.0), 2) if best_trade else None,
                "best_trade_symbol": str((best_trade or {}).get("symbol") or ""),
                "worst_trade_net_pnl": round(_safe_float((worst_trade or {}).get("net_pnl"), 0.0), 2) if worst_trade else None,
                "worst_trade_symbol": str((worst_trade or {}).get("symbol") or ""),
                "last_trade_at": rows[0].get("last_fill_time") if rows else None,
                "win_rate_label": headline_win_rate["label"],
                "win_rate_basis": headline_win_rate["basis"],
                "win_rate_sample_size": headline_win_rate["closed_trades"],
                "lane_breakdown": lane_breakdown,
            },
            "trades": limited_rows,
            "count": len(limited_rows),
            "total_count": len(rows),
            "display_limit": display_limit,
            "timestamp": now.isoformat(),
        }

    def _build_crypto_trade_log() -> Dict[str, Any]:
        return _build_trade_log("crypto", display_limit=_dashboard_trade_log_limit)

    def _load_event_intel() -> Dict[str, Any]:
        try:
            from pipeline.event_intel import load_latest_event_intelligence

            payload = load_latest_event_intelligence(ROOT / "data" / "event_intel")
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    # â”€â”€ REST â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @app.route("/")
    def index():
        return render_template_string(DASHBOARD_HTML)

    @app.route("/api/signals")
    def get_signals():
        lane_filter = (request.args.get("lane") or "").strip().lower()
        sigs = _flatten_signal_store(_signal_store_snapshot())
        if lane_filter in {"normal", "day", "crypto"}:
            sigs = [s for s in sigs if str(s.get("lane", "")).lower() == lane_filter]
        sigs.sort(key=lambda s: (
            0 if s.get("signal") == "buy" else 1 if s.get("signal") == "sell" else 2,
            -_safe_float(s.get("confidence"), 0.0),
        ))
        return jsonify(_json_safe({
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
        }))

    @app.route("/api/signals/<symbol>")
    def get_signal(symbol):
        sig = _signal_store_snapshot().get(symbol.upper())
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

    @app.route("/api/chart/<symbol>")
    def get_symbol_chart(symbol):
        window_seconds = max(60, min(7 * 24 * 3600, int(request.args.get("window", "1800") or 1800)))
        max_points = max(24, min(480, int(request.args.get("points", "240") or 240)))
        return jsonify(_build_symbol_chart(symbol, window_seconds, max_points))

    @app.route("/api/portfolio")
    def get_portfolio():
        return jsonify(_dashboard_portfolio_summary())

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

    @app.route("/api/crypto-trade-log")
    def get_crypto_trade_log():
        return jsonify(_json_safe(_build_crypto_trade_log()))

    @app.route("/api/trades")
    def get_trades():
        return jsonify(_json_safe(_build_trade_log(display_limit=None)))

    @app.route("/api/lane-control/<lane>/stop", methods=["POST"])
    def stop_lane_control(lane):
        lane_name = str(lane or "").strip().lower()
        if lane_name not in {"normal", "day", "crypto"}:
            return jsonify({"error": f"Unsupported lane '{lane}'"}), 400
        if runtime_controller is None or not hasattr(runtime_controller, "manual_stop_lane"):
            return jsonify({"error": "Lane control unavailable"}), 503
        reason = ""
        payload = request.get_json(silent=True)
        if isinstance(payload, dict):
            reason = str(payload.get("reason") or "").strip()
        try:
            controls = runtime_controller.manual_stop_lane(lane_name, source="dashboard", reason=reason or "manual_stop")
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500
        return jsonify({"ok": True, "lane": lane_name, "manual_lane_controls": _json_safe(controls)})

    @app.route("/api/lane-control/<lane>/resume", methods=["POST"])
    def resume_lane_control(lane):
        lane_name = str(lane or "").strip().lower()
        if lane_name not in {"normal", "day", "crypto"}:
            return jsonify({"error": f"Unsupported lane '{lane}'"}), 400
        if runtime_controller is None or not hasattr(runtime_controller, "manual_resume_lane"):
            return jsonify({"error": "Lane control unavailable"}), 503
        try:
            controls = runtime_controller.manual_resume_lane(lane_name, source="dashboard")
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500
        return jsonify({"ok": True, "lane": lane_name, "manual_lane_controls": _json_safe(controls)})

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
        snapshot = _signal_store_snapshot()
        if not snapshot:
            return jsonify({"error": "No signals yet",
                            "hint": "Wait ~5 min for first inference cycle"})
        sigs = list(snapshot.values())
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
        rows = [_normalize_trace_row(row) for row in list(_execution_trace) if isinstance(row, dict)]
        rows.sort(key=lambda r: _to_datetime(r.get("timestamp")) or _trace_fallback_timestamp, reverse=True)
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

    @app.route("/api/alpha-quality")
    def get_alpha_quality():
        if not _alpha_quality:
            return jsonify({"note": "Alpha quality report not ready yet. Wait for execution backtest refresh."})
        return jsonify(_alpha_quality)

    @app.route("/api/execution-divergence")
    def get_execution_divergence():
        rows = [_normalize_trace_row(row) for row in list(_execution_reconciliation or []) if isinstance(row, dict)]
        rows.sort(key=lambda r: _to_datetime(r.get("timestamp")) or _trace_fallback_timestamp, reverse=True)
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

    @app.route("/api/trade_debug")
    def trade_debug():
        """Show why each trade-ready signal is or isn't being executed."""
        signal_store = _signal_store_snapshot()
        runtime_controls = (((_system_health.get("runtime", {}) if isinstance(_system_health, dict) else {}) or {}).get("manual_lane_controls", {}) or {})
        rows = []
        for symbol, signal in list(signal_store.items()):
            if not isinstance(signal, dict):
                continue
            lane = str(signal.get("lane") or "normal").lower()
            meta = signal.get("meta_decision") or {}
            construction = signal.get("portfolio_construction") or {}
            entry_readiness = signal.get("entry_readiness") or {}
            governor = signal.get("governor_decision") or {}
            execution_filter = signal.get("execution_filter") or {}
            manual_lane_control = signal.get("manual_lane_control") or runtime_controls.get(lane) or {}
            intraday = signal.get("intraday_overlay") or {}
            best_bid = float(intraday.get("best_bid", 0.0) or 0.0)
            best_ask = float(intraday.get("best_ask", 0.0) or 0.0)
            execution_price_hint = round((best_bid + best_ask) / 2.0, 8) if best_bid > 0 and best_ask > 0 else None
            rows.append(
                {
                    "symbol": symbol,
                    "lane": lane,
                    "signal_direction": signal.get("signal"),
                    "conviction": round(float(signal.get("conviction_score", 0.0) or 0.0), 3),
                    "trade_eligible": signal.get("trade_eligible", False),
                    "take_trade": meta.get("take_trade", False),
                    "take_probability": round(float(meta.get("take_probability", 0.0) or 0.0), 4),
                    "meta_reason": meta.get("reason", ""),
                    "meta_source": meta.get("source", ""),
                    "expected_edge_pct": round(float(meta.get("expected_edge_pct", 0.0) or 0.0), 3),
                    "portfolio_target_pct": round(float(construction.get("target_position_pct", 0.0) or 0.0), 4),
                    "entry_readiness": entry_readiness.get("reason", ""),
                    "entry_allowed": entry_readiness.get("allow", True),
                    "meta_override": signal.get("meta_override", ""),
                    "governor_allow": governor.get("allow", True),
                    "governor_reason": governor.get("reason", ""),
                    "manual_lane_enabled": manual_lane_control.get("enabled", True),
                    "manual_lane_stopped_at": manual_lane_control.get("stopped_at"),
                    "execution_filter_allow": execution_filter.get("allow", True),
                    "execution_filter_reason": execution_filter.get("reason", ""),
                    "quote_only_fallback": bool(intraday.get("quote_only_fallback", False)),
                    "tick_age_seconds": round(float(intraday.get("tick_age_seconds", 0.0) or 0.0), 3),
                    "depth_age_seconds": round(float(intraday.get("depth_age_seconds", 0.0) or 0.0), 3) if intraday.get("depth_age_seconds") is not None else None,
                    "execution_price_hint": execution_price_hint,
                    "execution_reference_price": signal.get("execution_reference_price"),
                }
            )

        trade_ready = [r for r in rows if r["trade_eligible"] or r["take_trade"]]
        buy_signals = [r for r in rows if r["signal_direction"] == "buy"]
        blocked_by_meta = [r for r in buy_signals if not r["take_trade"]]
        blocked_by_readiness = [r for r in buy_signals if not r["entry_allowed"]]
        blocked_by_governor = [r for r in buy_signals if not r["governor_allow"]]
        blocked_by_manual_lane = [r for r in buy_signals if not r["manual_lane_enabled"]]
        blocked_by_execution_filter = [r for r in buy_signals if not r["execution_filter_allow"]]
        zero_weight = [r for r in buy_signals if r["portfolio_target_pct"] <= 0]
        missing_execution_price = [r for r in buy_signals if not r["execution_reference_price"] and not r["execution_price_hint"]]

        return jsonify(
            {
                "summary": {
                    "total_signals": len(rows),
                    "buy_signals": len(buy_signals),
                    "trade_ready": len(trade_ready),
                    "blocked_by_meta_skip": len(blocked_by_meta),
                    "blocked_by_entry_readiness": len(blocked_by_readiness),
                    "blocked_by_governor": len(blocked_by_governor),
                    "blocked_by_manual_lane": len(blocked_by_manual_lane),
                    "blocked_by_execution_filter": len(blocked_by_execution_filter),
                    "blocked_by_zero_weight": len(zero_weight),
                    "missing_execution_price": len(missing_execution_price),
                },
                "buy_signals_sample": sorted(buy_signals, key=lambda r: r["conviction"], reverse=True)[:20],
            }
        )

    @app.route("/api/learning-status")
    def get_learning_status():
        if not _learning_status:
            return jsonify({"note": "Learning status not ready yet."})
        return jsonify(_learning_status)

    @app.route("/api/daily-report")
    def get_daily_report():
        return jsonify(_build_daily_report())

    @app.route("/api/event-intel")
    def get_event_intel():
        payload = _load_event_intel()
        if payload:
            return jsonify(payload)
        return jsonify(
            {
                "summary": {"headline_count": 0, "symbols_with_news": 0, "symbols_covered": 0, "official_symbols": 0},
                "bullets": ["Event intelligence is warming up. A fresh sentiment refresh will create the first report."],
                "top_headlines": [],
                "symbol_heat": [],
                "top_bullish": [],
                "top_bearish": [],
            }
        )

    @app.route("/api/health")
    def health():
        buf = price_buffer.stats if price_buffer else {}
        broker_summary = paper_broker.get_summary() if paper_broker is not None else {}
        signal_snapshot = _signal_store_snapshot()
        return jsonify({
            "status":           "ok",
            "tick_count":       buf.get("total_ticks", 0),
            "active_symbols":   len(price_buffer.active_symbols()) if price_buffer else 0,
            "signal_count":     len(signal_snapshot),
            "broker_connected": paper_broker is not None,
            "execution_mode":   broker_summary.get("execution_mode", "paper"),
            "shadow_router":    broker_summary.get("shadow_router", {}),
            "stress_computed":  bool(_stress_results),
            "trade_readiness":  (_system_health.get("status") if isinstance(_system_health, dict) else "unknown"),
            "new_entries_enabled": bool((_system_health.get("new_entries_enabled") if isinstance(_system_health, dict) else False)),
            "blocking_reasons": ((_system_health.get("blocking_reasons", []) if isinstance(_system_health, dict) else []) or []),
            "data_sources":     (_system_health.get("data_sources", {}) if isinstance(_system_health, dict) else {}),
            "lane_open_counts": (((_system_health.get("runtime", {}) if isinstance(_system_health, dict) else {}) or {}).get("lane_open_counts", {})),
            "lane_targets":     (((_system_health.get("runtime", {}) if isinstance(_system_health, dict) else {}) or {}).get("lane_targets", {})),
            "manual_lane_controls": (((_system_health.get("runtime", {}) if isinstance(_system_health, dict) else {}) or {}).get("manual_lane_controls", {})),
            "universe_sync":    (((_system_health.get("runtime", {}) if isinstance(_system_health, dict) else {}) or {}).get("universe_sync", {})),
            "pipeline_selection": (((_system_health.get("runtime", {}) if isinstance(_system_health, dict) else {}) or {}).get("pipeline_selection", {})),
            "data_pipeline":   (((_system_health.get("runtime", {}) if isinstance(_system_health, dict) else {}) or {}).get("data_pipeline", {})),
            "uptime_seconds":   round(time.time() - _startup_time),
            "timestamp":        datetime.now(timezone.utc).isoformat(),
        })

    # â”€â”€ SocketIO events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @socketio.on("connect")
    def handle_connect():
        _client_count[0] += 1
        logger.debug(f"Client connected [{_client_count[0]}]")
        signal_snapshot = _signal_store_snapshot()
        emit(
            "signals_update",
            {"signals": _limit_values(_flatten_signal_store(signal_snapshot), _dashboard_signal_limit)},
        )
        if paper_broker:
            emit("portfolio_update", _dashboard_portfolio_summary())
        if _stress_results:
            emit("stress_update", _stress_results)
        try:
            emit("daily_report_update", _build_daily_report())
        except Exception as exc:
            logger.debug("daily_report on connect: %s", exc)

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
        sig = _signal_store_snapshot().get(sym)
        if sig:
            emit("signal_detail", sig)

    # â”€â”€ Push thread â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    _last_ids = set()
    _last_lat_push = [0.0]
    _last_report_push = [0.0]
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
                signal_snapshot = _signal_store_snapshot()
                cur_ids = set(signal_snapshot.keys())
                new_ids = cur_ids - _last_ids
                for sid in new_ids:
                    sig = signal_snapshot.get(sid)
                    if sig:
                        socketio.emit("new_signal", sig)
                _last_ids.update(new_ids)
                if cur_ids:
                    limited_signals = _limit_values(_flatten_signal_store(signal_snapshot), _dashboard_signal_limit)
                    fingerprint = "|".join(
                        f"{sig.get('signal_key', sig.get('symbol'))}:{sig.get('signal')}:{round(float(sig.get('confidence', 0.0)), 4)}:{round(float(sig.get('conviction_score', 0.0)), 2)}:{round(float(sig.get('rank_score', 0.0)), 4)}:{sig.get('lane')}:{sig.get('trade_eligible')}"
                        for sig in limited_signals
                    )
                    if fingerprint != _last_signal_snapshot["fingerprint"]:
                        socketio.emit("signals_update", {"signals": limited_signals})
                        _last_signal_snapshot["fingerprint"] = fingerprint

                # Portfolio
                if paper_broker:
                    socketio.emit("portfolio_update", _dashboard_portfolio_summary())

                # Daily report — emit every 15 s so trade counts stay live
                now = time.time()
                if now - _last_report_push[0] > 15:
                    try:
                        socketio.emit("daily_report_update", _build_daily_report())
                    except Exception:
                        pass
                    _last_report_push[0] = now

                # Latency
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
<script src="/socket.io/socket.io.js"></script>
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
.dot{width:7px;height:7px;border-radius:50%;background:var(--amber);animation:blink 2s infinite;}
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
.tvHost{min-height:420px;background:linear-gradient(180deg, var(--panel2), var(--panel));border:1px solid var(--border);border-radius:16px;overflow:hidden;display:flex;flex-direction:column;}
.tvEmpty{display:flex;align-items:center;justify-content:center;min-height:420px;color:var(--muted);font-size:11px;padding:16px;text-align:center;}
.chartToolbar{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap;margin-bottom:12px;}
.chartActions{display:flex;align-items:center;gap:8px;flex-wrap:wrap;}
.chartTab{font-size:10px;padding:5px 9px;border-radius:999px;border:1px solid var(--border2);background:transparent;color:var(--muted);cursor:pointer;font-family:var(--mono);}
.chartTab.active{color:var(--text);border-color:rgba(56,189,248,.38);background:rgba(56,189,248,.12);}
.chartLink{font-size:10px;padding:5px 10px;border-radius:999px;border:1px solid rgba(56,189,248,.24);color:var(--blue);text-decoration:none;}
.chartLink:hover{border-color:rgba(56,189,248,.42);background:rgba(56,189,248,.08);}
.chartMeta{font-size:10px;color:var(--muted);}
.chartCanvasWrap{position:relative;flex:1;min-height:420px;padding:12px;}
.chartCanvasWrap canvas{width:100% !important;height:100% !important;}
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
.laneActionBtn{background:rgba(56,189,248,.10);border:1px solid rgba(56,189,248,.25);color:var(--text);padding:4px 9px;border-radius:999px;font-size:9px;font-family:var(--mono);cursor:pointer;transition:all .15s ease;}
.laneActionBtn:hover{border-color:rgba(56,189,248,.45);background:rgba(56,189,248,.18);}
.laneActionBtn.stop{background:rgba(239,68,68,.10);border-color:rgba(239,68,68,.25);}
.laneActionBtn.stop:hover{background:rgba(239,68,68,.18);border-color:rgba(239,68,68,.45);}
.laneActionBtn.resume{background:rgba(34,197,94,.10);border-color:rgba(34,197,94,.25);}
.laneActionBtn.resume:hover{background:rgba(34,197,94,.18);border-color:rgba(34,197,94,.45);}
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
  <button class="menuLink" id="menu-trades" type="button" onclick="setPage('trades')">Trades</button>
  <button class="menuLink" id="menu-execution" type="button" onclick="setPage('execution')">Execution</button>
</div>
<div class="sbar">
  <span>WS avg: <strong id="sl" style="color:var(--muted)">-</strong> ms</span>
  <span>p95: <strong id="sp">-</strong> ms</span>
  <span id="scl" style="color:var(--amber)">Connecting...</span>
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
        <div class="met"><div class="ml">Win rate (paper)</div><div class="mv" id="mwr">-</div><div class="ms" id="mwrsub">recent closed paper trades</div></div>
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
          <div class="chartToolbar">
            <div class="ct2" style="margin-bottom:0">Selected symbol chart</div>
            <div class="chartActions">
              <button class="chartTab active" data-window="5m" onclick="setSymbolChartWindow('5m')">5m</button>
              <button class="chartTab" data-window="30m" onclick="setSymbolChartWindow('30m')">30m</button>
              <button class="chartTab" data-window="2h" onclick="setSymbolChartWindow('2h')">2h</button>
              <button class="chartTab" data-window="1d" onclick="setSymbolChartWindow('1d')">1d</button>
              <a id="tvopen" class="chartLink" href="#" target="_blank" rel="noopener noreferrer">open external</a>
            </div>
          </div>
          <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap;margin-bottom:12px">
            <div class="pill" id="tvsym">Select a signal</div>
            <div class="chartMeta" id="tvmeta">Market-board clicks update this live chart instantly.</div>
          </div>
          <div id="tvchart" class="tvHost"><div class="tvEmpty">Select a market-board symbol to load the live chart.</div></div>
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
      <div class="card" style="margin-top:12px">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;gap:10px;flex-wrap:wrap">
          <div class="ct2" style="margin-bottom:0">Crypto trade ledger</div>
          <div class="pill" id="ctlc">0 shown / 0 total</div>
        </div>
        <div class="metrics" style="grid-template-columns:repeat(6,1fr);margin-bottom:10px">
          <div class="met"><div class="ml">Closed net P&amp;L</div><div class="mv" id="ctlnet">-</div><div class="ms" id="ctlnetmeta">Waiting for crypto trades</div></div>
          <div class="met"><div class="ml">Open net P&amp;L</div><div class="mv" id="ctlopen">-</div><div class="ms" id="ctlopenmeta">Open crypto positions</div></div>
          <div class="met"><div class="ml">Win rate</div><div class="mv" id="ctlwin">-</div><div class="ms" id="ctlwinmeta">Closed crypto trades</div></div>
          <div class="met"><div class="ml">Fees paid</div><div class="mv" id="ctlfees">-</div><div class="ms" id="ctlfeesmeta">Entry + exit commissions</div></div>
          <div class="met"><div class="ml">Best trade</div><div class="mv" id="ctlbest">-</div><div class="ms" id="ctlbestmeta">Largest closed win</div></div>
          <div class="met"><div class="ml">Worst trade</div><div class="mv" id="ctlworst">-</div><div class="ms" id="ctlworstmeta">Largest closed loss</div></div>
        </div>
        <div class="small" id="ctlmeta" style="margin-bottom:10px">Rebuilding crypto trade history from real paper fills...</div>
        <div class="ptw" style="max-height:420px">
          <table class="pt">
            <thead><tr><th>Time</th><th>Symbol</th><th>Status</th><th>Qty</th><th>Buy</th><th>Sell / Now</th><th>Realized</th><th>Unrealized</th><th>Net</th><th>Fees</th><th>Hold</th><th>Reason</th></tr></thead>
            <tbody id="ctltb"><tr><td colspan="12" style="color:var(--muted);padding:14px">No crypto trades yet</td></tr></tbody>
          </table>
        </div>
      </div>
    </div>

    <div class="page" id="page-trades">
      <div class="card">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;gap:10px;flex-wrap:wrap">
          <div class="ct2" style="margin-bottom:0">Executed trade ledger</div>
          <div class="pill" id="trdc">0 shown / 0 total</div>
        </div>
        <div class="metrics" style="grid-template-columns:repeat(6,1fr);margin-bottom:10px">
          <div class="met"><div class="ml">Headline win rate</div><div class="mv" id="trdwin">-</div><div class="ms" id="trdwinmeta">Waiting for closed trades</div></div>
          <div class="met"><div class="ml">Closed net P&amp;L</div><div class="mv" id="trdnet">-</div><div class="ms" id="trdnetmeta">Closed trade result</div></div>
          <div class="met"><div class="ml">Open net P&amp;L</div><div class="mv" id="trdopen">-</div><div class="ms" id="trdopenmeta">Open trade exposure</div></div>
          <div class="met"><div class="ml">Fees paid</div><div class="mv" id="trdfees">-</div><div class="ms" id="trdfeesmeta">Total commissions</div></div>
          <div class="met"><div class="ml">Best trade</div><div class="mv" id="trdbest">-</div><div class="ms" id="trdbestmeta">Largest closed winner</div></div>
          <div class="met"><div class="ml">Worst trade</div><div class="mv" id="trdworst">-</div><div class="ms" id="trdworstmeta">Largest closed loser</div></div>
        </div>
        <div class="lanegrid" id="trdlanes" style="margin-bottom:10px">
          <div class="laneCard"><div class="lanel">Normal Trading</div><div class="lanev">-</div><div class="laneSub">Waiting for trades</div></div>
          <div class="laneCard"><div class="lanel">Day Trading</div><div class="lanev">-</div><div class="laneSub">Waiting for trades</div></div>
          <div class="laneCard"><div class="lanel">Crypto Scalper</div><div class="lanev">-</div><div class="laneSub">Waiting for trades</div></div>
        </div>
        <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:10px">
          <select id="trdlane" onchange="renderTradeLedger(tradeLedgerCache)" style="background:#0d121b;border:1px solid var(--border2);color:var(--text);padding:7px 9px;border-radius:5px;font-family:var(--mono);font-size:10px">
            <option value="all">all lanes</option>
            <option value="normal">normal</option>
            <option value="day">day</option>
            <option value="crypto">crypto</option>
          </select>
          <select id="trdstatus" onchange="renderTradeLedger(tradeLedgerCache)" style="background:#0d121b;border:1px solid var(--border2);color:var(--text);padding:7px 9px;border-radius:5px;font-family:var(--mono);font-size:10px">
            <option value="all">all statuses</option>
            <option value="closed">closed</option>
            <option value="open">open</option>
          </select>
          <input
            id="trdsearch"
            type="text"
            placeholder="Search symbol, setup, reason, position key..."
            oninput="renderTradeLedger(tradeLedgerCache)"
            style="flex:1 1 260px;background:#0d121b;border:1px solid var(--border2);color:var(--text);padding:7px 9px;border-radius:5px;font-family:var(--mono);font-size:10px;outline:none"
          />
        </div>
        <div class="small" id="trdmeta" style="margin-bottom:10px">Rebuilding executed trade history...</div>
        <div class="ptw" style="max-height:560px">
          <table class="pt">
            <thead><tr><th>Last fill</th><th>Symbol</th><th>Lane</th><th>Status</th><th>Qty</th><th>Buy</th><th>Sell / Now</th><th>Realized</th><th>Unrealized</th><th>Net</th><th>Fees</th><th>Hold</th><th>Reason</th></tr></thead>
            <tbody id="trdtb"><tr><td colspan="13" style="color:var(--muted);padding:14px">No executed trades yet</td></tr></tbody>
          </table>
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
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
          <div class="ct2" style="margin-bottom:0">Event intelligence</div>
          <div class="pill" id="evstamp">warming</div>
        </div>
        <div id="evbullets" class="small">Waiting for the first event-news digest...</div>
        <div style="display:grid;grid-template-columns:1.3fr .9fr;gap:12px;margin-top:12px">
          <div class="card" style="padding:10px;background:var(--bg3)">
            <div class="ct2" style="margin-bottom:8px">Top headlines</div>
            <div class="ptw" style="max-height:260px">
              <table class="pt">
                <thead><tr><th>Symbol</th><th>Headline</th><th>Catalyst</th><th>Score</th></tr></thead>
                <tbody id="evheadlines"><tr><td colspan="4" style="color:var(--muted);padding:12px">No headlines yet</td></tr></tbody>
              </table>
            </div>
          </div>
          <div class="card" style="padding:10px;background:var(--bg3)">
            <div class="ct2" style="margin-bottom:8px">Symbol heat</div>
            <div class="ptw" style="max-height:260px">
              <table class="pt">
                <thead><tr><th>Symbol</th><th>Sentiment</th><th>News</th><th>Catalyst</th><th>Lane</th></tr></thead>
                <tbody id="evsymbols"><tr><td colspan="5" style="color:var(--muted);padding:12px">No symbol heat yet</td></tr></tbody>
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
const _ioOrigin = (typeof window !== 'undefined' && window.location && window.location.origin)
  ? window.location.origin
  : undefined;
const _ioOpts = {
  path: '/socket.io',
  transports: ['polling', 'websocket'],
  upgrade: true,
  reconnection: true,
  reconnectionAttempts: 100,
  reconnectionDelay: 1000,
  reconnectionDelayMax: 8000,
  timeout: 20000,
};
const io_sock = SOCKET_AVAILABLE
  ? (_ioOrigin ? io(_ioOrigin, _ioOpts) : io(_ioOpts))
  : { on(){}, emit(){} };
const MARKET_BOARD_LIMIT = 36;
const LEADER_SYMBOLS = ['AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA','AMD','NFLX','QCOM','AVGO','JPM','WMT','XOM','SPY','QQQ','DIA','IWM','RELIANCE.NS','TCS.NS','INFY.NS'];
const TV_AMEX_SYMBOLS = new Set(['SPY','DIA','IWM','GLD','SLV','TLT','HYG','VTI','XLF','XLE','XLK','XLY','XLI','XLV','XLP','XLB','XLU']);
const TV_NASDAQ_SYMBOLS = new Set(['AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA','AMD','NFLX','QCOM','AVGO','INTC','ORCL','ADBE','CSCO','PEP','COST','TMUS','TXN','AMAT','ADP','ISRG','BKNG','LRCX','MELI','MU','PANW','SNPS','CDNS','KLAC','ASML','SHOP','CRWD','DDOG','MDB','TEAM','ZS']);
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
let signals = {}, sel = null, eq = [100000], chart = null, sd = 'conviction', activeLane = 'all', activePage = 'overview', reportData = null, eventIntel = null, livePrices = {};
let tradeLedgerCache = null;
let healthSnapshot = {};
let metaStatusFailures = 0;
let learningStatusFailures = 0;
const SYMBOL_CHART_WINDOWS = { '5m': 300, '30m': 1800, '2h': 7200, '1d': 86400 };
let tvState = { symbol: '', window: '30m', fetchedAt: 0 };
let chartSelection = { symbol: '', signalKey: '', lane: 'normal', market: '', source: 'auto', laneLabel: '' };
let selectedSymbolChart = null;
let symbolChartRequestId = 0;
let symbolChartCache = { symbol: '', window: '', fetchedAt: 0, payload: null };

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
  const nextPage = ['overview', 'signals', 'portfolio', 'trades', 'execution'].includes(page) ? page : 'overview';
  activePage = nextPage;
  document.querySelectorAll('.page').forEach(el => el.classList.toggle('active', el.id === `page-${nextPage}`));
  ['overview', 'signals', 'portfolio', 'trades', 'execution'].forEach(name => {
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
    refreshCryptoTradeLog();
  }else if(nextPage === 'trades'){
    refreshTradeLedger();
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
  if(TV_NASDAQ_SYMBOLS.has(clean)) return `NASDAQ:${clean}`;
  if(TV_NYSE_SYMBOLS.has(clean)) return `NYSE:${clean}`;
  if(TV_AMEX_SYMBOLS.has(clean)) return `AMEX:${clean}`;
  return clean;
}

function selectedChartWindowSeconds(){
  return SYMBOL_CHART_WINDOWS[tvState.window] || SYMBOL_CHART_WINDOWS['30m'];
}

function setSymbolChartWindow(windowKey){
  tvState.window = SYMBOL_CHART_WINDOWS[windowKey] ? windowKey : '30m';
  document.querySelectorAll('.chartTab').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.window === tvState.window);
  });
  renderTradingViewChart(currentChartSignal() || defaultChartSignal(), true);
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

async function renderTradingViewChart(signal, forceRefresh=false){
  const host = document.getElementById('tvchart');
  const badge = document.getElementById('tvsym');
  const meta = document.getElementById('tvmeta');
  const openLink = document.getElementById('tvopen');
  if(!host || !badge) return;
  if(!signal || !signal.symbol){
    badge.textContent = 'Select a signal';
    if(meta) meta.textContent = 'Market-board clicks update this live chart instantly.';
    if(openLink){
      openLink.href = '#';
      openLink.textContent = 'open external';
    }
    host.innerHTML = '<div class="tvEmpty">Select a market-board symbol to load the live chart.</div>';
    if(selectedSymbolChart){
      selectedSymbolChart.destroy();
      selectedSymbolChart = null;
    }
    symbolChartCache = { symbol: '', window: '', fetchedAt: 0, payload: null };
    return;
  }

  const rawSymbol = String(signal.symbol || '').toUpperCase();
  const mappedSymbol = tradingViewSymbol(rawSymbol, signal.market || marketHintForSymbol(rawSymbol));
  badge.textContent = rawSymbol;
  if(openLink){
    openLink.href = mappedSymbol
      ? `https://www.tradingview.com/chart/?symbol=${encodeURIComponent(mappedSymbol)}`
      : `https://finance.yahoo.com/quote/${encodeURIComponent(rawSymbol)}`;
    openLink.textContent = mappedSymbol ? 'open in TradingView' : 'open external';
  }

  const requestId = ++symbolChartRequestId;
  const now = Date.now();
  const windowKey = tvState.window;
  let payload = null;
  if(
    !forceRefresh
    && symbolChartCache.symbol === rawSymbol
    && symbolChartCache.window === windowKey
    && (now - symbolChartCache.fetchedAt) < 10000
  ){
    payload = symbolChartCache.payload;
  }else{
    try{
      const response = await fetch(`/api/chart/${encodeURIComponent(rawSymbol)}?window=${selectedChartWindowSeconds()}&points=240`);
      payload = await response.json();
      symbolChartCache = { symbol: rawSymbol, window: windowKey, fetchedAt: now, payload };
    }catch(_err){
      payload = symbolChartCache.symbol === rawSymbol ? symbolChartCache.payload : null;
    }
  }
  if(requestId !== symbolChartRequestId) return;

  const points = Array.isArray(payload && payload.points) ? payload.points : [];
  const livePrice = Number(signal.last_price ?? signal.current_price ?? payload?.price);
  const changePct = Number(signal.price_change_pct ?? payload?.change_pct);
  const palette = themePalette();
  const lineColor = Number.isFinite(changePct) ? (changePct >= 0 ? '#22c55e' : '#fb7185') : '#38bdf8';

  if(meta){
    const windowLabel = Object.entries(SYMBOL_CHART_WINDOWS).find(([, seconds]) => seconds === selectedChartWindowSeconds())?.[0] || windowKey;
    const pointLabel = points.length ? `${points.length} live points` : 'waiting for live points';
    meta.textContent = `${windowLabel} • ${pointLabel}${Number.isFinite(changePct) ? ` • ${formatSignedPct(changePct)}` : ''}`;
  }

  if(!points.length && !Number.isFinite(livePrice)){
    host.innerHTML = `<div class="tvEmpty">Waiting for live chart data for ${rawSymbol}...</div>`;
    if(selectedSymbolChart){
      selectedSymbolChart.destroy();
      selectedSymbolChart = null;
    }
    tvState = { symbol: rawSymbol, window: windowKey, fetchedAt: now };
    return;
  }

  if(!host.querySelector('#symbolChartCanvas')){
    host.innerHTML = '<div class="chartCanvasWrap"><canvas id="symbolChartCanvas"></canvas></div>';
  }
  const canvas = host.querySelector('#symbolChartCanvas');
  if(!canvas || !CHART_AVAILABLE){
    host.innerHTML = '<div class="tvEmpty">Chart rendering unavailable in this browser.</div>';
    return;
  }

  const labels = points.length ? points.map(point => point.label || '') : ['now'];
  const values = points.length ? points.map(point => Number(point.price || 0)) : [livePrice];

  if(!selectedSymbolChart){
    selectedSymbolChart = new Chart(canvas.getContext('2d'), {
      type:'line',
      data:{
        labels,
        datasets:[{
          label: rawSymbol,
          data: values,
          borderColor: lineColor,
          backgroundColor: palette.area,
          fill: true,
          pointRadius: 0,
          tension: 0.22,
          borderWidth: 2,
        }]
      },
      options:{
        responsive:true,
        maintainAspectRatio:false,
        animation:false,
        plugins:{legend:{display:false}, tooltip:{mode:'index', intersect:false}},
        scales:{
          x:{ticks:{color: palette.axis, maxTicksLimit: 8}, grid:{color: palette.grid}},
          y:{ticks:{color: palette.axis}, grid:{color: palette.grid}}
        }
      }
    });
  }else{
    selectedSymbolChart.data.labels = labels;
    selectedSymbolChart.data.datasets[0].data = values;
    selectedSymbolChart.data.datasets[0].label = rawSymbol;
    selectedSymbolChart.data.datasets[0].borderColor = lineColor;
    selectedSymbolChart.data.datasets[0].backgroundColor = palette.area;
    selectedSymbolChart.options.scales.x.ticks.color = palette.axis;
    selectedSymbolChart.options.scales.x.grid.color = palette.grid;
    selectedSymbolChart.options.scales.y.ticks.color = palette.axis;
    selectedSymbolChart.options.scales.y.grid.color = palette.grid;
    selectedSymbolChart.update('none');
  }
  tvState = { symbol: rawSymbol, window: windowKey, fetchedAt: now };
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

function manualLaneControl(lane){
  const controls = healthSnapshot.manual_lane_controls || {};
  const control = controls[lane] || {};
  return {
    enabled: control.enabled !== false,
    stopped_at: control.stopped_at || '',
    reason: control.reason || '',
  };
}

function laneControlStatusLabel(lane, fallbackStatus){
  const control = manualLaneControl(lane);
  if(!control.enabled) return 'manually stopped';
  return fallbackStatus;
}

function setLaneControl(lane, enabled, event){
  if(event){
    event.preventDefault();
    event.stopPropagation();
  }
  const action = enabled ? 'resume' : 'stop';
  fetch(`/api/lane-control/${encodeURIComponent(lane)}/${action}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({reason: enabled ? '' : 'manual_stop'}),
  }).then(r => r.json()).then(() => {
    refreshHealth();
    refreshSignals();
    refreshDailyReport();
  }).catch(() => {});
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
    const openPnl = Number(laneReport.open_unrealized_pnl || 0);
    const control = manualLaneControl(lane);
    const status = laneControlStatusLabel(lane, laneReport.status || (actionable.length ? 'active' : laneSignals.length ? 'warming' : 'standby'));
    const active = activeLane === lane || (activeLane === 'all' && lane === 'normal');
    const accent = pnl > 0 ? 'var(--green)' : pnl < 0 ? 'var(--red)' : 'var(--text)';
    const buttonLabel = control.enabled ? 'Stop' : 'Resume';
    const buttonClass = control.enabled ? 'stop' : 'resume';
    const detailText = control.enabled
      ? `${laneReport.open_positions || 0} open | ${laneReport.closed_trades_today || 0} closed | realized ${pnl >= 0 ? '+' : '-'}$${Math.abs(pnl).toFixed(2)} | open ${openPnl >= 0 ? '+' : '-'}$${Math.abs(openPnl).toFixed(2)}`
      : `stopped ${control.stopped_at ? new Date(control.stopped_at).toLocaleTimeString() : 'now'} | ${laneReport.open_positions || 0} open`;
    return `<div class="laneCard clickable${active ? ' active' : ''}" onclick="openLaneView('${lane}')">
      <div class="lanel">${LANE_LABELS[lane]}</div>
      <div class="lanev" style="color:${laneSignals.length ? 'var(--blue)' : accent}">${laneSignals.length}</div>
      <div class="laneSub">${actionable.length} trade-ready | ${laneSignals.length} total signals<br>${buys} buy | ${sells} sell | ${watch} watch</div>
      <div class="laneStatRow">
        <span class="lanePill">${status}</span>
        <span class="laneMini">${detailText}</span>
      </div>
      <div class="laneStatRow" style="margin-top:6px;padding-top:6px">
        <span class="laneMini">${control.enabled ? 'new entries allowed' : 'new entries blocked'}</span>
        <button type="button" class="laneActionBtn ${buttonClass}" onclick="setLaneControl('${lane}', ${control.enabled ? 'false' : 'true'}, event)">${buttonLabel}</button>
      </div>
    </div>`;
  });
  const totalActionable = Object.values(signals).filter(isTradeReady).length;
  const totalSignals = Object.values(signals).length;
  const srv = Number(healthSnapshot.signal_count || 0);
  const livePx = Number(healthSnapshot.active_symbols || 0);
  if(totalSignals){
    pill.textContent = `${totalSignals} live signals • ${totalActionable} trade-ready across 3 domains`;
  }else if(srv > 0){
    pill.textContent = `Server reports ${srv} signals (UI empty) — hard refresh`;
  }else if(livePx > 0){
    pill.textContent = `No signals yet • ${livePx} symbols on price feed`;
  }else{
    pill.textContent = `0 live signals • ${totalActionable} trade-ready across 3 domains`;
  }
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
  return !!(s && !s.warmup_only && (s.trade_eligible || (s.meta_decision && s.meta_decision.take_trade)));
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
  const mwrsub = document.getElementById('mwrsub');
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
  if(mwrsub){
    const sample = Number(d.win_rate_sample_size ?? 0);
    const label = d.win_rate_label || 'recent closed paper trades';
    mwrsub.textContent = sample > 0 ? label : 'waiting for closed paper trades';
  }
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
  // Fallback: if socket hasn't connected within 8s, show Polling mode (REST still updates)
  const _connTimeout = setTimeout(() => {
    const txt = document.getElementById('scl');
    if(txt && (txt.textContent === 'Connecting...' || txt.textContent === 'Reconnecting...')){
      setConnectionState('Polling', 'var(--amber)');
    }
  }, 8000);
  io_sock.on('connect', () => { clearTimeout(_connTimeout); setConnectionState('Connected', 'var(--green)'); });
  io_sock.on('connect_error', () => setConnectionState('Reconnecting...', 'var(--amber)'));
  io_sock.on('disconnect', () => setConnectionState('Reconnecting...', 'var(--amber)'));
  io_sock.on('reconnect', () => { clearTimeout(_connTimeout); setConnectionState('Connected', 'var(--green)'); });
  io_sock.on('reconnect_failed', () => setConnectionState('Disconnected', 'var(--red)'));
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
  io_sock.on('daily_report_update', renderDailyReport);
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
    const serverSignalCount = Number(healthSnapshot.signal_count || 0);
    if(serverSignalCount > 0){
      if(countEl) countEl.textContent = String(serverSignalCount);
      if(metaEl) metaEl.textContent = `Server reports ${serverSignalCount} signals — browser list empty (hard refresh or check /api/signals)`;
      el.innerHTML = `<div class="empty">The API has signals but this page did not load them. Hard refresh (Ctrl+Shift+R), or DevTools → Network → failed /api/signals. Mixed http/https or an old tab can cause this.</div>`;
      return;
    }
    const warming = activeSymbols > 0 || tickCount > 0;
    if(countEl) countEl.textContent = warming ? 'warming' : '0';
    if(metaEl) metaEl.textContent = warming
      ? `Live prices (${activeSymbols} symbols • ${tickCount.toLocaleString()} ticks) — no signals in memory yet; inference/bootstrap may still be running`
      : 'Waiting for signal stream...';
    el.innerHTML = `<div class="empty">${warming ? 'Price feed is live, but the scored signal store is still empty. If this persists for many minutes, check the VM: curl -s 127.0.0.1:5050/api/health | head -c 400 and logs/system.out for inference or blocking_reasons.' : 'No signals yet...'}</div>`;
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
    if(countEl) countEl.textContent = String(filtered.length);
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
  const serverSignalCount = Number(healthSnapshot.signal_count || 0);
  const liveSyms = Number(healthSnapshot.active_symbols || 0);
  const warming = !arr.length && liveSyms > 0 && serverSignalCount === 0;
  const desync = !arr.length && serverSignalCount > 0;
  if(mns) mns.textContent = desync ? 'SYNC' : (warming ? 'WARMING' : String(filtered.length));
  if(msp) msp.textContent = desync
    ? `server has ${serverSignalCount} signals — reload page or verify /api/signals`
    : (warming
      ? `prices live for ${liveSyms} symbols; signal_count=0 — scoring not populated (see logs)`
      : `${actionable.length} trade-ready - ${buys} live buy - ${sells} live sell - ${watch} watch${activeLane !== 'all' && activeLane !== 'reports' ? ` - ${LANE_LABELS[activeLane]}` : ''}`);
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
    const openPnl = Number(lane.open_unrealized_pnl || 0);
    return `<div class="laneCard${isActive ? ' active' : ''}">
      <div class="lanel">${lane.lane_label}</div>
      <div class="lanev" style="color:${(lane.day_pnl || 0) >= 0 ? 'var(--green)' : 'var(--red)'}">${lane.day_pnl >= 0 ? '+' : ''}$${Math.abs(lane.day_pnl || 0).toFixed(2)}</div>
      <div class="laneSub">
        ${lane.trade_ready || 0} trade-ready • ${lane.open_positions || 0} open positions<br>
        ${lane.fills_today || 0} fills today • ${lane.closed_trades_today || 0} closed • ${lane.win_rate_pct || 0}% win • PF ${lane.profit_factor || 0}<br>
        open unrealized ${openPnl >= 0 ? '+' : '-'}$${Math.abs(openPnl).toFixed(2)} • realized day P&amp;L above<br>
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

function renderEventIntel(data){
  eventIntel = data || eventIntel;
  const stamp = document.getElementById('evstamp');
  const bullets = document.getElementById('evbullets');
  const headlinesBody = document.getElementById('evheadlines');
  const symbolsBody = document.getElementById('evsymbols');
  if(!stamp || !bullets || !headlinesBody || !symbolsBody || !eventIntel) return;
  const summary = eventIntel.summary || {};
  const generatedAt = eventIntel.generated_at ? new Date(eventIntel.generated_at).toLocaleTimeString() : 'warming';
  stamp.textContent = `${summary.symbols_with_news || 0}/${summary.symbols_covered || 0} symbols • ${generatedAt}`;
  bullets.innerHTML = (eventIntel.bullets || []).length
    ? `<ul style="margin:0;padding-left:18px">${eventIntel.bullets.map(item => `<li style="margin:0 0 6px 0">${item}</li>`).join('')}</ul>`
    : 'Event intelligence is warming up.';
  const headlineRows = eventIntel.top_headlines || [];
  headlinesBody.innerHTML = headlineRows.length
    ? headlineRows.map(item => {
        const score = Number(item.headline_score || 0);
        const catalyst = item.primary_catalyst ? String(item.primary_catalyst).replaceAll('_',' ') : 'general';
        return `<tr>
          <td style="font-weight:600">${item.symbol || '-'}</td>
          <td><a href="${item.url || '#'}" target="_blank" rel="noopener noreferrer" style="color:var(--text);text-decoration:none">${item.title || '-'}</a></td>
          <td style="color:var(--muted)">${catalyst}</td>
          <td style="color:${score >= 4 ? 'var(--green)' : score >= 2 ? 'var(--amber)' : 'var(--muted)'}">${score.toFixed(2)}</td>
        </tr>`;
      }).join('')
    : '<tr><td colspan="4" style="color:var(--muted);padding:12px">No headlines yet</td></tr>';
  const symbolRows = eventIntel.symbol_heat || [];
  symbolsBody.innerHTML = symbolRows.length
    ? symbolRows.map(item => {
        const sentiment = Number(item.mean_weighted_sentiment || 0);
        const sentimentColor = sentiment > 0 ? 'var(--green)' : sentiment < 0 ? 'var(--red)' : 'var(--muted)';
        const catalyst = item.primary_catalyst || (Array.isArray(item.top_catalysts) && item.top_catalysts.length ? item.top_catalysts[0] : '-');
        return `<tr>
          <td style="font-weight:600">${item.symbol || '-'}</td>
          <td style="color:${sentimentColor}">${sentiment >= 0 ? '+' : ''}${sentiment.toFixed(2)}</td>
          <td>${item.headline_count || 0}</td>
          <td>${catalyst}</td>
          <td>${item.lane_label || '-'}</td>
        </tr>`;
      }).join('')
    : '<tr><td colspan="5" style="color:var(--muted);padding:12px">No symbol heat yet</td></tr>';
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

function fmtSignedMoney(v){
  const num = Number(v);
  if(!Number.isFinite(num)) return '-';
  return `${num >= 0 ? '+' : '-'}$${Math.abs(num).toLocaleString(undefined, {maximumFractionDigits:2, minimumFractionDigits:2})}`;
}

function fmtDateTime(v){
  if(!v) return '-';
  const d = new Date(v);
  return Number.isNaN(d.getTime()) ? '-' : d.toLocaleString();
}

function fmtHoldMinutes(v){
  const num = Number(v);
  if(!Number.isFinite(num)) return '-';
  if(num < 60) return `${num.toFixed(1)}m`;
  const hours = num / 60;
  if(hours < 24) return `${hours.toFixed(1)}h`;
  return `${(hours / 24).toFixed(1)}d`;
}

function escapeHtml(value){
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function tradeLedgerFilters(){
  const lane = document.getElementById('trdlane')?.value || 'all';
  const status = document.getElementById('trdstatus')?.value || 'all';
  const query = (document.getElementById('trdsearch')?.value || '').trim().toLowerCase();
  return {lane, status, query};
}

function renderTradeLedger(data){
  tradeLedgerCache = data || tradeLedgerCache;
  const payload = tradeLedgerCache || {};
  const summary = payload.summary || {};
  const rows = Array.isArray(payload.trades) ? payload.trades : [];
  const counter = document.getElementById('trdc');
  const meta = document.getElementById('trdmeta');
  const body = document.getElementById('trdtb');
  const win = document.getElementById('trdwin');
  const winMeta = document.getElementById('trdwinmeta');
  const net = document.getElementById('trdnet');
  const netMeta = document.getElementById('trdnetmeta');
  const open = document.getElementById('trdopen');
  const openMeta = document.getElementById('trdopenmeta');
  const fees = document.getElementById('trdfees');
  const feesMeta = document.getElementById('trdfeesmeta');
  const best = document.getElementById('trdbest');
  const bestMeta = document.getElementById('trdbestmeta');
  const worst = document.getElementById('trdworst');
  const worstMeta = document.getElementById('trdworstmeta');
  const lanes = document.getElementById('trdlanes');
  if(!counter || !meta || !body || !win || !winMeta || !net || !netMeta || !open || !openMeta || !fees || !feesMeta || !best || !bestMeta || !worst || !worstMeta || !lanes) return;

  const breakdown = summary.lane_breakdown || {};
  const filters = tradeLedgerFilters();
  const filteredRows = rows.filter(row => {
    const laneMatch = filters.lane === 'all' || String(row.lane || '').toLowerCase() === filters.lane;
    const statusMatch = filters.status === 'all' || String(row.status || '').toLowerCase() === filters.status;
    if(!laneMatch || !statusMatch) return false;
    if(!filters.query) return true;
    const haystack = [
      row.symbol,
      row.position_key,
      row.lane_label,
      row.setup_id,
      row.signal_source,
      row.entry_reason,
      row.exit_reason,
      row.time_bucket,
      row.execution_mode,
    ].join(' ').toLowerCase();
    return haystack.includes(filters.query);
  });

  const closedNet = Number(summary.net_closed_pnl || 0);
  const openNet = Number(summary.net_open_pnl || 0);
  const feesPaid = Number(summary.total_commission || 0);
  const bestValue = Number(summary.best_trade_net_pnl);
  const worstValue = Number(summary.worst_trade_net_pnl);

  counter.textContent = `${filteredRows.length} shown / ${Number(payload.total_count || rows.length)} total`;
  win.textContent = `${Number(summary.win_rate_pct || 0).toFixed(1)}%`;
  win.style.color = Number(summary.win_rate_pct || 0) >= 50 ? 'var(--green)' : 'var(--amber)';
  winMeta.textContent = summary.win_rate_label || 'waiting for closed paper trades';

  net.textContent = fmtSignedMoney(closedNet);
  net.style.color = closedNet >= 0 ? 'var(--green)' : 'var(--red)';
  netMeta.textContent = `${summary.closed_trades || 0} closed | ${summary.winning_trades || 0} winners`;

  open.textContent = fmtSignedMoney(openNet);
  open.style.color = openNet >= 0 ? 'var(--green)' : 'var(--red)';
  openMeta.textContent = `${summary.open_trades || 0} open | unrealized ${fmtSignedMoney(summary.gross_unrealized_pnl || 0)}`;

  fees.textContent = fmtMoney(feesPaid);
  fees.style.color = feesPaid > 0 ? 'var(--amber)' : 'var(--muted)';
  feesMeta.textContent = summary.avg_hold_minutes != null ? `Avg hold ${fmtHoldMinutes(summary.avg_hold_minutes)}` : 'Hold time builds after exits';

  best.textContent = summary.best_trade_symbol ? `${summary.best_trade_symbol} ${fmtSignedMoney(bestValue)}` : '-';
  best.style.color = Number.isFinite(bestValue) && bestValue >= 0 ? 'var(--green)' : 'var(--muted)';
  bestMeta.textContent = summary.best_trade_symbol ? 'Largest closed winner' : 'Waiting for a closed win';

  worst.textContent = summary.worst_trade_symbol ? `${summary.worst_trade_symbol} ${fmtSignedMoney(worstValue)}` : '-';
  worst.style.color = Number.isFinite(worstValue) && worstValue < 0 ? 'var(--red)' : 'var(--muted)';
  worstMeta.textContent = summary.worst_trade_symbol ? 'Largest closed loser' : 'Waiting for a closed loss';

  lanes.innerHTML = ['normal', 'day', 'crypto'].map(laneName => {
    const lane = breakdown[laneName] || {};
    const laneNet = Number(lane.net_pnl || 0);
    return `<div class="laneCard">
      <div class="lanel">${escapeHtml(lane.lane_label || LANE_LABELS[laneName] || laneName)}</div>
      <div class="lanev" style="color:${laneNet >= 0 ? 'var(--green)' : 'var(--red)'}">${fmtSignedMoney(laneNet)}</div>
      <div class="laneSub">${lane.trade_count || 0} trades | ${lane.closed_trades || 0} closed | ${lane.open_trades || 0} open<br>${Number(lane.win_rate_pct || 0).toFixed(1)}% win</div>
    </div>`;
  }).join('');

  const lastTradeLabel = summary.last_trade_at ? `Last fill ${fmtDateTime(summary.last_trade_at)}` : 'No fills yet';
  meta.textContent = `${lastTradeLabel} | ${summary.fill_events || 0} fill events | ${summary.closed_trades || 0} closed | ${summary.open_trades || 0} open`;

  if(!filteredRows.length){
    body.innerHTML = '<tr><td colspan="13" style="color:var(--muted);padding:14px">No trades match the current filters</td></tr>';
    return;
  }

  body.innerHTML = filteredRows.map(row => {
    const isOpen = String(row.status || '').toLowerCase() === 'open';
    const netPnl = Number(row.net_pnl || 0);
    const realized = Number(row.gross_realized_pnl || 0);
    const unrealized = Number(row.gross_unrealized_pnl || 0);
    const statusColor = isOpen ? 'var(--blue)' : (netPnl >= 0 ? 'var(--green)' : 'var(--red)');
    const reason = row.exit_reason || row.entry_reason || row.signal_source || '-';
    const symbolMeta = [row.setup_id, row.time_bucket, row.execution_mode, row.position_key].filter(Boolean).join(' | ');
    const laneMeta = [row.lane_label || row.lane || '-', row.position_direction || '-'].join(' | ');
    const edgeBits = [];
    if(row.entry_expected_edge_pct != null) edgeBits.push(`edge ${Number(row.entry_expected_edge_pct).toFixed(2)}%`);
    if(row.entry_take_probability != null) edgeBits.push(`take ${Math.round(Number(row.entry_take_probability) * 100)}%`);
    if(row.avg_fill_ratio_pct != null) edgeBits.push(`fill ${Number(row.avg_fill_ratio_pct).toFixed(1)}%`);
    return `<tr>
      <td><div>${escapeHtml(fmtDateTime(row.last_fill_time || row.entry_time))}</div><div style="color:var(--muted);font-size:9px">${row.exit_time ? `out ${escapeHtml(fmtDateTime(row.exit_time))}` : 'still open'}</div></td>
      <td><div style="font-weight:600">${escapeHtml(row.symbol || '-')}</div><div style="color:var(--muted);font-size:9px">${escapeHtml(symbolMeta || '-')}</div></td>
      <td><div>${escapeHtml(laneMeta)}</div><div style="color:var(--muted);font-size:9px">${escapeHtml(row.market || '-')}</div></td>
      <td><span style="color:${statusColor};font-weight:600">${isOpen ? 'OPEN' : 'CLOSED'}</span><div style="color:var(--muted);font-size:9px">${row.fill_count || 0} fills${row.partial_fill_count ? ` | ${row.partial_fill_count} partial` : ''}</div></td>
      <td>${fmtQty(row.quantity)}</td>
      <td>${row.avg_buy_price != null ? fmtMoney(row.avg_buy_price) : '-'}</td>
      <td>${(isOpen ? row.current_price : row.avg_sell_price) != null ? fmtMoney(isOpen ? row.current_price : row.avg_sell_price) : '-'}</td>
      <td style="color:${realized >= 0 ? 'var(--green)' : 'var(--red)'}">${fmtSignedMoney(realized)}</td>
      <td style="color:${unrealized >= 0 ? 'var(--green)' : 'var(--red)'}">${fmtSignedMoney(unrealized)}</td>
      <td style="color:${netPnl >= 0 ? 'var(--green)' : 'var(--red)'}">${fmtSignedMoney(netPnl)}<div style="color:var(--muted);font-size:9px">${row.net_return_pct != null ? `${Number(row.net_return_pct).toFixed(2)}%` : ''}</div></td>
      <td>${fmtMoney(row.commission || 0)}</td>
      <td>${fmtHoldMinutes(row.hold_minutes)}</td>
      <td><div>${escapeHtml(reason)}</div><div style="color:var(--muted);font-size:9px">${escapeHtml(edgeBits.join(' | '))}</div></td>
    </tr>`;
  }).join('');
}

function refreshTradeLedger(){
  fetch('/api/trades').then(r => r.json()).then(d => renderTradeLedger(d)).catch(() => {});
}

function refreshCryptoTradeLog(){
  const counter = document.getElementById('ctlc');
  const meta = document.getElementById('ctlmeta');
  const body = document.getElementById('ctltb');
  const net = document.getElementById('ctlnet');
  const netMeta = document.getElementById('ctlnetmeta');
  const open = document.getElementById('ctlopen');
  const openMeta = document.getElementById('ctlopenmeta');
  const win = document.getElementById('ctlwin');
  const winMeta = document.getElementById('ctlwinmeta');
  const fees = document.getElementById('ctlfees');
  const feesMeta = document.getElementById('ctlfeesmeta');
  const best = document.getElementById('ctlbest');
  const bestMeta = document.getElementById('ctlbestmeta');
  const worst = document.getElementById('ctlworst');
  const worstMeta = document.getElementById('ctlworstmeta');
  if(!counter || !meta || !body || !net || !netMeta || !open || !openMeta || !win || !winMeta || !fees || !feesMeta || !best || !bestMeta || !worst || !worstMeta) return;

  fetch('/api/crypto-trade-log').then(r => r.json()).then(d => {
    const summary = d.summary || {};
    const rows = d.trades || [];
    const closedNet = Number(summary.net_closed_pnl || 0);
    const openNet = Number(summary.net_open_pnl || 0);
    const feesPaid = Number(summary.total_commission || 0);
    const bestValue = Number(summary.best_trade_net_pnl);
    const worstValue = Number(summary.worst_trade_net_pnl);

    counter.textContent = `${rows.length} shown / ${d.total_count || 0} total`;
    net.textContent = fmtSignedMoney(closedNet);
    net.style.color = closedNet >= 0 ? 'var(--green)' : 'var(--red)';
    netMeta.textContent = `${summary.closed_trades || 0} closed • ${summary.fill_events || 0} fill events`;
    open.textContent = fmtSignedMoney(openNet);
    open.style.color = openNet >= 0 ? 'var(--green)' : 'var(--red)';
    openMeta.textContent = `${summary.open_trades || 0} open • unrealized ${fmtSignedMoney(summary.gross_unrealized_pnl || 0)}`;
    win.textContent = `${Number(summary.win_rate_pct || 0).toFixed(1)}%`;
    win.style.color = Number(summary.win_rate_pct || 0) >= 50 ? 'var(--green)' : 'var(--amber)';
    winMeta.textContent = `${summary.winning_trades || 0} winners • ${summary.losing_trades || 0} losers`;
    fees.textContent = fmtMoney(feesPaid);
    fees.style.color = feesPaid > 0 ? 'var(--amber)' : 'var(--muted)';
    feesMeta.textContent = summary.avg_hold_minutes != null ? `Avg hold ${fmtHoldMinutes(summary.avg_hold_minutes)}` : 'Hold time builds after exits';
    best.textContent = summary.best_trade_symbol ? `${summary.best_trade_symbol} ${fmtSignedMoney(bestValue)}` : '-';
    best.style.color = Number.isFinite(bestValue) && bestValue >= 0 ? 'var(--green)' : 'var(--muted)';
    bestMeta.textContent = summary.best_trade_symbol ? 'Largest closed winner' : 'Waiting for a closed win';
    worst.textContent = summary.worst_trade_symbol ? `${summary.worst_trade_symbol} ${fmtSignedMoney(worstValue)}` : '-';
    worst.style.color = Number.isFinite(worstValue) && worstValue < 0 ? 'var(--red)' : 'var(--muted)';
    worstMeta.textContent = summary.worst_trade_symbol ? 'Largest closed loser' : 'Waiting for a closed loss';

    const lastTradeLabel = summary.last_trade_at ? `Last fill ${fmtDateTime(summary.last_trade_at)}` : 'No crypto fills yet';
    meta.textContent = `${lastTradeLabel} • closed gross ${fmtSignedMoney(summary.gross_closed_pnl || 0)} • open gross ${fmtSignedMoney(summary.gross_open_pnl || 0)}`;

    if(!rows.length){
      body.innerHTML = '<tr><td colspan="12" style="color:var(--muted);padding:14px">No crypto trades logged yet</td></tr>';
      return;
    }

    body.innerHTML = rows.map(row => {
      const isOpen = row.status === 'open';
      const netPnl = Number(row.net_pnl || 0);
      const realized = Number(row.gross_realized_pnl || 0);
      const unrealized = Number(row.gross_unrealized_pnl || 0);
      const statusColor = isOpen ? 'var(--blue)' : (netPnl >= 0 ? 'var(--green)' : 'var(--red)');
      const reason = row.exit_reason || row.entry_reason || row.signal_source || '-';
      const symbolMeta = [row.setup_id, row.time_bucket, row.execution_mode].filter(Boolean).join(' • ');
      const edgeBits = [];
      if(row.entry_expected_edge_pct != null) edgeBits.push(`edge ${Number(row.entry_expected_edge_pct).toFixed(2)}%`);
      if(row.entry_take_probability != null) edgeBits.push(`take ${Math.round(Number(row.entry_take_probability) * 100)}%`);
      return `<tr>
        <td><div>${fmtDateTime(row.entry_time)}</div><div style="color:var(--muted);font-size:9px">${row.exit_time ? `out ${fmtDateTime(row.exit_time)}` : 'still open'}</div></td>
        <td><div style="font-weight:600">${row.symbol || '-'}</div><div style="color:var(--muted);font-size:9px">${symbolMeta || (row.position_key || '')}</div></td>
        <td><span style="color:${statusColor};font-weight:600">${isOpen ? 'OPEN' : 'CLOSED'}</span><div style="color:var(--muted);font-size:9px">${row.fill_count || 0} fills${row.partial_fill_count ? ` • ${row.partial_fill_count} partial` : ''}</div></td>
        <td>${fmtQty(row.quantity)}</td>
        <td>${row.avg_buy_price != null ? fmtMoney(row.avg_buy_price) : '-'}</td>
        <td>${(isOpen ? row.current_price : row.avg_sell_price) != null ? fmtMoney(isOpen ? row.current_price : row.avg_sell_price) : '-'}</td>
        <td style="color:${realized >= 0 ? 'var(--green)' : 'var(--red)'}">${fmtSignedMoney(realized)}</td>
        <td style="color:${unrealized >= 0 ? 'var(--green)' : 'var(--red)'}">${fmtSignedMoney(unrealized)}</td>
        <td style="color:${netPnl >= 0 ? 'var(--green)' : 'var(--red)'}">${fmtSignedMoney(netPnl)}<div style="color:var(--muted);font-size:9px">${row.net_return_pct != null ? `${Number(row.net_return_pct).toFixed(2)}%` : ''}</div></td>
        <td>${fmtMoney(row.commission || 0)}</td>
        <td>${fmtHoldMinutes(row.hold_minutes)}</td>
        <td><div>${reason}</div><div style="color:var(--muted);font-size:9px">${edgeBits.join(' • ')}</div></td>
      </tr>`;
    }).join('');
  }).catch(() => {});
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
    metaStatusFailures = 0;
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
    metaStatusFailures += 1;
    if(metaStatusFailures < 4){
      if(sub.textContent && !sub.textContent.includes('reconnecting')){
        sub.textContent = `${sub.textContent} • reconnecting`;
      }else if(!sub.textContent){
        sub.textContent = 'meta reconnecting';
      }
      return;
    }
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
    learningStatusFailures = 0;
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
    learningStatusFailures += 1;
    if(learningStatusFailures < 4){
      if(sub.textContent && !sub.textContent.includes('reconnecting')){
        sub.textContent = `${sub.textContent} • reconnecting`;
      }else if(!sub.textContent){
        sub.textContent = 'learning reconnecting';
      }
      return;
    }
    label.textContent = '-';
    label.style.color = 'var(--muted)';
    sub.textContent = 'learning status unavailable';
  });
}

function refreshDailyReport(){
  fetch('/api/daily-report').then(r => r.json()).then(renderDailyReport).catch(() => {});
}

function refreshEventIntel(){
  fetch('/api/event-intel').then(r => r.json()).then(renderEventIntel).catch(() => {});
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
    // Server has signals but browser store empty (often /api/signals timed out while CPU-heavy); retry fetch.
    const sc = Number(d.signal_count || 0);
    if(sc > 0 && !Object.keys(signals).length){
      refreshSignals();
      setTimeout(refreshSignals, 3000);
    }
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
refreshCryptoTradeLog();
refreshTradeLedger();
refreshExecutionTrace();
refreshExecutionBacktest();
refreshExecutionDivergence();
refreshMetaModel();
refreshLearningStatus();
refreshDailyReport();
refreshEventIntel();
refreshHealth();
setInterval(refreshSignals, 10000);
setInterval(refreshPricesSnapshot, 5000);
setInterval(refreshPortfolioSummary, 10000);
setInterval(refreshPos, 5000);
setInterval(refreshLivePortfolio, 15000);
setInterval(refreshCryptoTradeLog, 10000);
setInterval(refreshTradeLedger, 10000);
setInterval(refreshExecutionTrace, 5000);
setInterval(refreshExecutionBacktest, 30000);
setInterval(refreshExecutionDivergence, 7000);
setInterval(refreshMetaModel, 30000);
setInterval(refreshLearningStatus, 30000);
setInterval(refreshDailyReport, 15000);
setInterval(refreshEventIntel, 30000);
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
        socketio.run(
            app,
            host="0.0.0.0",
            port=port,
            debug=False,
            use_reloader=False,
            allow_unsafe_werkzeug=os.getenv("ALLOW_UNSAFE_WERKZEUG", "1").strip().lower()
            in {"1", "true", "yes", "on"},
        )
