"""
Real-Time WebSocket Data Engine
=================================
"Professional systems don't use polled APIs for real-time work. They use WebSockets."

This module replaces the polling approach with event-driven WebSocket connections:
  - Finnhub: Sub-second US stock tick data (NYSE, NASDAQ)
  - Upstox/Dhan: Real-time NSE/BSE tick data
  - Latency target: price â†’ alert on phone in < 500ms

Architecture:
  WebSocket threads â†’ Shared price buffer â†’ Signal engine â†’ SocketIO â†’ Dashboard
  (real-time)          (thread-safe)         (async)        (push)     (browser)
"""

import json
import time
import os
import queue
import random
import logging
import threading
from datetime import datetime, timezone
from collections import deque, defaultdict
from pathlib import Path
from typing import Optional, Dict, Callable, List, Set
from dataclasses import dataclass, field

import websocket  # websocket-client
import requests
import pandas as pd

logger = logging.getLogger(__name__)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Shared Real-Time Price Buffer
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@dataclass
class Tick:
    """A single price tick from any market."""
    symbol: str
    price: float
    volume: int
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    market: str = "US"   # "US" or "IN"
    source: str = ""

    @property
    def spread(self) -> Optional[float]:
        if self.bid and self.ask:
            return self.ask - self.bid
        return None


class RealTimePriceBuffer:
    """
    Thread-safe circular buffer for real-time price data.
    Shared between all WebSocket threads and the signal engine.
    """

    def __init__(self, buffer_size: int = 1000, *, dispatch_queue_size: int = 5000):
        self._buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=buffer_size))
        self._latest: Dict[str, Tick] = {}
        self._lock = threading.RLock()
        self._subscribers: List[Callable] = []
        self._stats: Dict[str, int] = defaultdict(int)
        self._dispatch_queue: "queue.Queue[Optional[Tick]]" = queue.Queue(maxsize=max(1, int(dispatch_queue_size)))
        self._dispatcher = threading.Thread(target=self._dispatch_loop, daemon=True, name="realtime-tick-dispatcher")
        self._dispatcher.start()

    def _dispatch_loop(self):
        while True:
            tick = self._dispatch_queue.get()
            if tick is None:
                return
            with self._lock:
                callbacks = list(self._subscribers)
            for callback in callbacks:
                try:
                    callback(tick)
                except Exception as exc:
                    logger.debug(f"Subscriber callback error: {exc}")

    def push(self, tick: Tick):
        """Push a new tick into the buffer. Thread-safe."""
        with self._lock:
            self._buffer[tick.symbol].append(tick)
            self._latest[tick.symbol] = tick
            self._stats["total_ticks"] += 1

        # Notify subscribers (async to avoid blocking WebSocket threads)
        if self._subscribers:
            try:
                self._dispatch_queue.put_nowait(tick)
            except queue.Full:
                with self._lock:
                    self._stats["dispatch_dropped_ticks"] += 1

    def latest(self, symbol: str) -> Optional[Tick]:
        """Get the most recent tick for a symbol."""
        with self._lock:
            return self._latest.get(symbol)

    def recent_ticks(self, symbol: str, n: int = 50) -> List[Tick]:
        """Get the N most recent ticks for a symbol."""
        with self._lock:
            buf = self._buffer.get(symbol, deque())
            return list(buf)[-n:]

    def get_ohlcv(self, symbol: str, seconds: int = 60) -> Optional[Dict]:
        """Compute a 1-minute OHLCV bar from tick data in real time."""
        ticks = self.recent_ticks(symbol, n=500)
        if not ticks:
            return None

        cutoff = datetime.now(timezone.utc).timestamp() - seconds
        recent = [t for t in ticks if t.timestamp.timestamp() > cutoff]
        if not recent:
            return None

        prices = [t.price for t in recent]
        return {
            "symbol": symbol,
            "open": prices[0],
            "high": max(prices),
            "low": min(prices),
            "close": prices[-1],
            "volume": sum(t.volume for t in recent),
            "tick_count": len(recent),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def subscribe(self, callback: Callable[[Tick], None]):
        """Subscribe to all incoming ticks."""
        with self._lock:
            self._subscribers.append(callback)

    def active_symbols(self) -> Set[str]:
        with self._lock:
            return set(self._latest.keys())

    @property
    def stats(self) -> Dict:
        with self._lock:
            return dict(self._stats)


class PublicDepthBuffer:
    """Thread-safe top-of-book / depth snapshot store."""

    def __init__(self):
        self._books: Dict[str, Dict] = {}
        self._level_stats: Dict[str, Dict[str, Dict[str, Dict]]] = defaultdict(lambda: {"bids": {}, "asks": {}})
        self._spread_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=64))
        self._lock = threading.RLock()

    @staticmethod
    def _level_key(price: float) -> str:
        return f"{float(price):.8f}"

    @staticmethod
    def _normalize_level(level) -> Optional[Dict[str, float]]:
        if isinstance(level, (list, tuple)) and len(level) >= 2:
            try:
                return {"price": float(level[0] or 0.0), "quantity": float(level[1] or 0.0)}
            except Exception:
                return None
        if isinstance(level, dict):
            try:
                return {
                    "price": float(level.get("price", level.get("p", 0.0)) or 0.0),
                    "quantity": float(level.get("quantity", level.get("qty", level.get("q", 0.0))) or 0.0),
                }
            except Exception:
                return None
        return None

    def _update_level_stats(self, symbol: str, side: str, levels, ts: datetime):
        side_stats = self._level_stats[symbol][side]
        for raw_level in levels or []:
            normalized = self._normalize_level(raw_level)
            if not normalized:
                continue
            price = normalized["price"]
            quantity = normalized["quantity"]
            if price <= 0 or quantity <= 0:
                continue
            key = self._level_key(price)
            state = side_stats.get(key)
            if state is None:
                side_stats[key] = {
                    "price": price,
                    "quantity": quantity,
                    "first_seen_ts": ts,
                    "last_seen_ts": ts,
                    "updates": 1,
                    "inter_update_delay_ms": 9999.0,
                }
                continue

            changed = abs(quantity - float(state.get("quantity", 0.0) or 0.0)) > 1e-12
            delay_ms = max(0.0, (ts - state["last_seen_ts"]).total_seconds() * 1000.0)
            state["last_seen_ts"] = ts
            state["inter_update_delay_ms"] = delay_ms if changed else float(state.get("inter_update_delay_ms", 9999.0) or 9999.0)
            if changed:
                state["updates"] = int(state.get("updates", 0) or 0) + 1
            state["quantity"] = quantity
            side_stats[key] = state

    def update(self, symbol: str, payload: Dict):
        with self._lock:
            current = dict(self._books.get(symbol, {}))
            current.update(payload)
            current["symbol"] = symbol
            now = datetime.now(timezone.utc)
            current["updated_at"] = now.isoformat()
            if "bids" in payload:
                self._update_level_stats(symbol, "bids", payload.get("bids", []), now)
            if "asks" in payload:
                self._update_level_stats(symbol, "asks", payload.get("asks", []), now)
            best_bid = float(current.get("best_bid", 0.0) or 0.0)
            best_ask = float(current.get("best_ask", 0.0) or 0.0)
            if best_bid > 0 and best_ask > 0 and best_ask >= best_bid:
                self._spread_history[symbol].append(
                    {
                        "timestamp": now,
                        "spread": best_ask - best_bid,
                        "mid": (best_bid + best_ask) / 2.0,
                    }
                )
            self._books[symbol] = current

    def latest(self, symbol: str) -> Dict:
        with self._lock:
            return dict(self._books.get(symbol, {}))

    def filtered_snapshot(
        self,
        symbol: str,
        *,
        min_lifetime_ms: float = 200.0,
        max_updates_per_second: float = 5.0,
        min_inter_update_delay_ms: float = 20.0,
    ) -> Dict:
        with self._lock:
            book = dict(self._books.get(symbol, {}))
            if not book:
                return {}
            side_stats = self._level_stats.get(symbol, {"bids": {}, "asks": {}})
            spread_history = list(self._spread_history.get(symbol, []))

        updated_at_raw = book.get("updated_at")
        try:
            updated_at = datetime.fromisoformat(str(updated_at_raw))
        except Exception:
            updated_at = datetime.now(timezone.utc)

        def _filter_side(side_name: str):
            filtered_levels = []
            metrics = []
            for raw_level in book.get(side_name, []) or []:
                normalized = self._normalize_level(raw_level)
                if not normalized:
                    continue
                price = normalized["price"]
                quantity = normalized["quantity"]
                if price <= 0 or quantity <= 0:
                    continue
                state = (side_stats.get(side_name) or {}).get(self._level_key(price))
                if state is None:
                    continue
                visible_seconds = max((updated_at - state["first_seen_ts"]).total_seconds(), 1e-3)
                lifetime_ms = max(0.0, (updated_at - state["first_seen_ts"]).total_seconds() * 1000.0)
                updates_per_second = float(state.get("updates", 1) or 1) / visible_seconds
                inter_delay_ms = float(state.get("inter_update_delay_ms", 9999.0) or 9999.0)
                keep = (
                    lifetime_ms >= min_lifetime_ms
                    and updates_per_second <= max_updates_per_second
                    and inter_delay_ms >= min_inter_update_delay_ms
                )
                metrics.append(
                    {
                        "price": price,
                        "quantity": quantity,
                        "lifetime_ms": round(lifetime_ms, 2),
                        "updates_per_second": round(updates_per_second, 3),
                        "inter_update_delay_ms": round(inter_delay_ms, 2),
                        "kept": keep,
                    }
                )
                if keep:
                    filtered_levels.append(raw_level)
            return filtered_levels, metrics

        filtered_bids, bid_metrics = _filter_side("bids")
        filtered_asks, ask_metrics = _filter_side("asks")

        def _notional(levels) -> float:
            total = 0.0
            for raw_level in levels[:5]:
                normalized = self._normalize_level(raw_level)
                if not normalized:
                    continue
                total += normalized["price"] * normalized["quantity"]
            return total

        bid_notional = _notional(filtered_bids)
        ask_notional = _notional(filtered_asks)
        denominator = bid_notional + ask_notional
        filtered_imbalance = ((bid_notional - ask_notional) / denominator) if denominator > 0 else 0.0

        spread_velocity = 0.0
        if len(spread_history) >= 2:
            first = spread_history[max(0, len(spread_history) - 10)]
            last = spread_history[-1]
            delta_seconds = max((last["timestamp"] - first["timestamp"]).total_seconds(), 1e-3)
            mid_price = max(float(last.get("mid", 0.0) or 0.0), 1e-6)
            spread_velocity = ((float(last.get("spread", 0.0) or 0.0) - float(first.get("spread", 0.0) or 0.0)) / mid_price) / delta_seconds

        book["filtered_bids"] = filtered_bids
        book["filtered_asks"] = filtered_asks
        book["filtered_bid_metrics"] = bid_metrics
        book["filtered_ask_metrics"] = ask_metrics
        book["filtered_book_imbalance"] = round(float(filtered_imbalance), 6)
        book["spread_velocity"] = round(float(spread_velocity), 8)
        book["flicker_filter_retention"] = round(
            (len(filtered_bids) + len(filtered_asks)) / max(len(book.get("bids", [])) + len(book.get("asks", [])), 1),
            4,
        )
        return book

    def active_symbols(self) -> Set[str]:
        with self._lock:
            return set(self._books.keys())


# Global shared buffer (singleton pattern)
PRICE_BUFFER = RealTimePriceBuffer()
DEPTH_BUFFER = PublicDepthBuffer()


class BinancePublicDepthWebSocket:
    """
    Public no-KYC crypto market data.

    Streams:
    - trade
    - bookTicker
    - depth20@100ms
    """

    WS_BASE = "wss://stream.binance.com:9443/stream?streams="
    TESTNET_WS_BASE = "wss://stream.testnet.binance.vision/stream?streams="

    def __init__(
        self,
        symbols: List[str],
        buffer: RealTimePriceBuffer = None,
        depth_buffer: PublicDepthBuffer = None,
        *,
        testnet: bool = False,
    ):
        self.symbols = [str(symbol).strip().lower() for symbol in symbols if str(symbol).strip()]
        self.buffer = buffer or PRICE_BUFFER
        self.depth_buffer = depth_buffer or DEPTH_BUFFER
        self.testnet = bool(testnet)
        self.source = "binance_testnet" if self.testnet else "binance"
        self.ws: Optional[websocket.WebSocketApp] = None
        self._running = False
        self._message_queue: "queue.Queue[Optional[str]]" = queue.Queue(
            maxsize=max(1, int(os.getenv("BINANCE_WS_MESSAGE_QUEUE_SIZE", "20000")))
        )
        self._worker_thread: Optional[threading.Thread] = None
        self._watchdog_thread: Optional[threading.Thread] = None
        self._reconnect_base_delay_seconds = float(os.getenv("BINANCE_WS_RECONNECT_BASE_DELAY_SECONDS", "3"))
        self._reconnect_max_delay_seconds = float(os.getenv("BINANCE_WS_RECONNECT_MAX_DELAY_SECONDS", "60"))
        self._ping_interval_seconds = float(os.getenv("BINANCE_WS_PING_INTERVAL_SECONDS", "30"))
        self._ping_timeout_seconds = float(os.getenv("BINANCE_WS_PING_TIMEOUT_SECONDS", "20"))
        self._depth_update_ms = int(os.getenv("BINANCE_WS_DEPTH_UPDATE_MS", "100"))
        self._stale_seconds = float(os.getenv("BINANCE_WS_STALE_SECONDS", "90"))
        self._last_message_ts = 0.0
        self._dropped_messages = 0
        self._last_drop_log_ts = 0.0

    def _build_url(self) -> str:
        depth_update_ms = self._depth_update_ms if self._depth_update_ms in (100, 1000) else 1000
        streams: List[str] = []
        for symbol in self.symbols:
            streams.extend(
                [f"{symbol}@trade", f"{symbol}@bookTicker", f"{symbol}@depth20@{depth_update_ms}ms"]
            )
        base = self.TESTNET_WS_BASE if self.testnet else self.WS_BASE
        return base + "/".join(streams)

    def start(self, daemon: bool = True):
        self._running = True
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._worker_thread = threading.Thread(target=self._process_messages, daemon=True, name="binance-ws-worker")
            self._worker_thread.start()
        if self._watchdog_thread is None or not self._watchdog_thread.is_alive():
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop, daemon=True, name="binance-ws-watchdog"
            )
            self._watchdog_thread.start()
        thread = threading.Thread(target=self._run_forever, daemon=daemon)
        thread.start()
        logger.info(f"Binance public depth WebSocket started for {len(self.symbols)} crypto symbols")

    def stop(self):
        self._running = False
        if self.ws:
            self.ws.close()
        try:
            self._message_queue.put_nowait(None)
        except Exception:
            pass

    def _enqueue_message(self, message: str):
        try:
            self._message_queue.put_nowait(message)
            return
        except queue.Full:
            pass

        # Drop the oldest message and try again (depth updates are high-frequency).
        try:
            _ = self._message_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._message_queue.put_nowait(message)
        except queue.Full:
            self._dropped_messages += 1

    def _process_messages(self):
        while self._running:
            try:
                message = self._message_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if message is None:
                return
            self._handle_message(message)

            # Rate-limit queue-drop logging to avoid log spam.
            now = time.time()
            if self._dropped_messages and now - self._last_drop_log_ts > 60:
                dropped = self._dropped_messages
                self._dropped_messages = 0
                self._last_drop_log_ts = now
                logger.warning(f"Binance public depth WebSocket worker dropped {dropped} messages (queue full)")

    def _watchdog_loop(self):
        while self._running:
            time.sleep(5)
            if not self._running or self._stale_seconds <= 0:
                continue
            last_ts = float(self._last_message_ts or 0.0)
            if last_ts <= 0:
                continue
            if time.time() - last_ts > float(self._stale_seconds):
                logger.warning(
                    f"Binance public depth WebSocket stale for >{int(self._stale_seconds)}s; forcing reconnect"
                )
                try:
                    if self.ws:
                        self.ws.close()
                except Exception:
                    pass

    def _run_forever(self):
        delay = max(0.5, float(self._reconnect_base_delay_seconds))
        max_delay = max(delay, float(self._reconnect_max_delay_seconds))
        ping_interval = self._ping_interval_seconds if self._ping_interval_seconds > 0 else None
        ping_timeout = self._ping_timeout_seconds if self._ping_timeout_seconds > 0 else None
        while self._running:
            start_ts = time.time()
            try:
                self.ws = websocket.WebSocketApp(
                    self._build_url(),
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self.ws.run_forever(ping_interval=ping_interval, ping_timeout=ping_timeout)
            except Exception as exc:
                logger.warning(f"Binance public depth WebSocket error: {exc}")
            if self._running:
                runtime = time.time() - start_ts
                # If the connection flaps quickly, back off exponentially with jitter.
                if runtime < 15:
                    delay = min(max_delay, delay * 2.0)
                else:
                    delay = max(0.5, float(self._reconnect_base_delay_seconds))
                time.sleep(min(max_delay, delay) + random.random())

    def _on_open(self, ws):
        if self._depth_update_ms not in (100, 1000):
            logger.warning(
                f"BINANCE_WS_DEPTH_UPDATE_MS={self._depth_update_ms} unsupported; using 1000ms instead"
            )
        self._last_message_ts = time.time()
        logger.info("Binance public depth WebSocket connected")

    def _on_message(self, ws, message):
        if not message:
            return
        self._last_message_ts = time.time()
        self._enqueue_message(message)

    def _handle_message(self, message: str):
        try:
            payload = json.loads(message)
        except Exception:
            return

        stream = str(payload.get("stream", ""))
        data = payload.get("data", {})
        if not stream or not isinstance(data, dict):
            return

        symbol = stream.split("@", 1)[0].upper()

        if "@trade" in stream:
            trade_ts = datetime.fromtimestamp(float(data.get("T", time.time() * 1000)) / 1000.0, tz=timezone.utc)
            try:
                volume = float(data.get("q", 0.0) or 0.0)
            except Exception:
                volume = 0.0
            price = float(data.get("p", 0.0) or 0.0)
            volume_proxy = int(max(1.0, price * volume)) if price > 0 and volume > 0 else 0
            tick = Tick(
                symbol=symbol,
                price=price,
                volume=volume_proxy,
                timestamp=trade_ts,
                market="CRYPTO",
                source=self.source,
            )
            if tick.price > 0:
                self.buffer.push(tick)
            return

        if "@bookTicker" in stream:
            best_bid = float(data.get("b", 0.0) or 0.0)
            best_ask = float(data.get("a", 0.0) or 0.0)
            self.depth_buffer.update(
                symbol,
                {
                    "best_bid": best_bid,
                    "best_bid_qty": float(data.get("B", 0.0) or 0.0),
                    "best_ask": best_ask,
                    "best_ask_qty": float(data.get("A", 0.0) or 0.0),
                    "book_ticker_update_id": data.get("u"),
                    "source": self.source,
                },
            )
            mid = (best_bid + best_ask) / 2.0 if best_bid > 0 and best_ask > 0 else 0.0
            if mid > 0:
                self.buffer.push(
                    Tick(
                        symbol=symbol,
                        price=mid,
                        volume=0,
                        timestamp=datetime.now(timezone.utc),
                        bid=best_bid,
                        ask=best_ask,
                        market="CRYPTO",
                        source=self.source,
                    )
                )
            return

        if "@depth20" in stream:
            bids = data.get("bids", []) or []
            asks = data.get("asks", []) or []
            self.depth_buffer.update(
                symbol,
                {
                    "bids": bids,
                    "asks": asks,
                    "last_depth_update_id": data.get("lastUpdateId"),
                    "source": self.source,
                },
            )

    def _on_error(self, ws, error):
        logger.warning(f"Binance public depth WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        logger.info(f"Binance public depth WebSocket closed: {close_status_code} {close_msg or ''}".strip())


class BybitPublicDepthWebSocket:
    """
    Public spot market-data stream for Bybit.

    Runs alongside Binance so the shared buffers always have a fresh crypto feed.
    """

    WS_BASE = "wss://stream.bybit.com/v5/public/spot"
    TESTNET_WS_BASE = "wss://stream-testnet.bybit.com/v5/public/spot"

    def __init__(
        self,
        symbols: List[str],
        buffer: RealTimePriceBuffer = None,
        depth_buffer: PublicDepthBuffer = None,
        *,
        testnet: bool = False,
        depth_levels: int = 50,
    ):
        self.symbols = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
        self.buffer = buffer or PRICE_BUFFER
        self.depth_buffer = depth_buffer or DEPTH_BUFFER
        self.testnet = bool(testnet)
        self.depth_levels = max(1, int(depth_levels))
        self.source = "bybit_testnet" if self.testnet else "bybit"
        self.ws: Optional[websocket.WebSocketApp] = None
        self._running = False
        self._books: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: {"bids": {}, "asks": {}})
        self._ping_thread: Optional[threading.Thread] = None
        self._topics_per_subscribe = max(1, min(10, int(os.getenv("BYBIT_WS_TOPICS_PER_SUBSCRIBE", "10") or 10)))
        # Library RFC ping/pong uses a short default timeout and drops under CPU load (e.g. retrain + run.py).
        # Bybit v5 answers app-level {"op":"ping"} in _ping_loop — prefer that unless explicitly enabled.
        self._bybit_lib_ping = os.getenv("BYBIT_WS_LIBRARY_PING", "0").strip().lower() in {"1", "true", "yes", "on"}
        self._bybit_ping_interval = max(0.0, float(os.getenv("BYBIT_WS_PING_INTERVAL_SECONDS", "45") or 45))
        self._bybit_ping_timeout = max(0.0, float(os.getenv("BYBIT_WS_PING_TIMEOUT_SECONDS", "90") or 90))
        self._bybit_app_ping_seconds = max(10.0, float(os.getenv("BYBIT_WS_APP_PING_SECONDS", "20") or 20))
        self._bybit_subscribe_pause_seconds = max(
            0.05,
            float(os.getenv("BYBIT_WS_SUBSCRIBE_PAUSE_SECONDS", "0.15") or 0.15),
        )
        self._reconnect_base_delay_seconds = float(os.getenv("BYBIT_WS_RECONNECT_BASE_DELAY_SECONDS", "3"))
        self._reconnect_max_delay_seconds = float(os.getenv("BYBIT_WS_RECONNECT_MAX_DELAY_SECONDS", "60"))
        self._connect_jitter_seconds = max(0.0, float(os.getenv("BYBIT_WS_CONNECT_JITTER_SECONDS", "1.5") or 1.5))
        self._rate_limit_backoff_seconds = max(
            5.0,
            float(os.getenv("BYBIT_WS_RATE_LIMIT_BACKOFF_SECONDS", "20") or 20),
        )
        # Increased from 30s → 90s: 30s was too aggressive and caused ~10k false reconnects
        # when the server sends no data (quiet market periods are normal, not stale)
        self._stale_seconds = float(os.getenv("BYBIT_WS_STALE_SECONDS", "90"))
        self._last_message_ts = 0.0
        self._watchdog_thread: Optional[threading.Thread] = None
        self._next_connect_not_before = 0.0

    def _ws_url(self) -> str:
        return self.TESTNET_WS_BASE if self.testnet else self.WS_BASE

    def _subscribe_topics(self) -> List[str]:
        topics: List[str] = []
        for symbol in self.symbols:
            topics.append(f"orderbook.{self.depth_levels}.{symbol}")
            topics.append(f"publicTrade.{symbol}")
        return topics

    def start(self, daemon: bool = True):
        self._running = True
        thread = threading.Thread(target=self._run_forever, daemon=daemon)
        thread.start()
        if self._watchdog_thread is None or not self._watchdog_thread.is_alive():
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop, daemon=True, name="bybit-ws-watchdog"
            )
            self._watchdog_thread.start()
        logger.info(f"Bybit public depth WebSocket started for {len(self.symbols)} crypto symbols")

    def stop(self):
        self._running = False
        if self.ws:
            self.ws.close()

    def _watchdog_loop(self):
        # Wait for the connection to establish before starting staleness checks
        time.sleep(15)
        while self._running:
            time.sleep(10)
            if not self._running or self._stale_seconds <= 0:
                continue
            last_ts = float(self._last_message_ts or 0.0)
            if last_ts <= 0:
                # Never received a message yet — don't force reconnect during startup
                continue
            stale_for = time.time() - last_ts
            if stale_for > float(self._stale_seconds):
                logger.warning(
                    f"Bybit WebSocket stale for {stale_for:.0f}s (>{int(self._stale_seconds)}s threshold); forcing reconnect"
                )
                try:
                    if self.ws:
                        self.ws.close()
                except Exception:
                    pass

    def _run_forever(self):
        delay = max(0.5, float(self._reconnect_base_delay_seconds))
        max_delay = max(delay, float(self._reconnect_max_delay_seconds))
        if self._connect_jitter_seconds > 0:
            time.sleep(random.uniform(0.0, self._connect_jitter_seconds))
        while self._running:
            wait_for_slot = self._next_connect_not_before - time.time()
            if wait_for_slot > 0:
                time.sleep(wait_for_slot)
            start_ts = time.time()
            try:
                self.ws = websocket.WebSocketApp(
                    self._ws_url(),
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                # Spec: Bybit must use ping_interval=0 explicitly (not None) —
                # app-level {"op":"ping"} in _ping_loop handles heartbeat.
                if self._bybit_lib_ping and self._bybit_ping_interval > 0:
                    self.ws.run_forever(
                        ping_interval=self._bybit_ping_interval,
                        ping_timeout=self._bybit_ping_timeout if self._bybit_ping_timeout > 0 else None,
                    )
                else:
                    # Keep websocket-client's "disabled" ping behavior explicit.
                    # Passing None here can leave the primary crypto feed in a bad
                    # state under this app's threading/load profile.
                    self.ws.run_forever(ping_interval=0, ping_timeout=None)
            except Exception as exc:
                logger.warning(f"Bybit public depth WebSocket error: {exc}")
            if self._running:
                runtime = time.time() - start_ts
                if runtime < 15:
                    delay = min(max_delay, delay * 2.0)
                else:
                    delay = max(0.5, float(self._reconnect_base_delay_seconds))
                sleep_for = min(max_delay, delay) + random.random()
                wait_for_slot = self._next_connect_not_before - time.time()
                if wait_for_slot > 0:
                    sleep_for = max(sleep_for, wait_for_slot)
                time.sleep(sleep_for)

    def _ping_loop(self):
        """Send Bybit app-level pings to keep the connection alive.

        The loop captures the ws reference at start so it exits cleanly when the
        connection is replaced during a reconnect (the new _on_open starts a fresh
        ping thread for the new ws instance).
        """
        ws_ref = self.ws
        while self._running and ws_ref is not None and ws_ref is self.ws:
            try:
                if ws_ref.sock and ws_ref.sock.connected:
                    ws_ref.send(json.dumps({"op": "ping"}))
            except Exception:
                return  # ws is gone, exit — _on_open will spawn a new ping thread
            time.sleep(self._bybit_app_ping_seconds)

    def _on_open(self, ws):
        # Reset reconnect backoff on successful connect
        self._reconnect_delay = 3.0
        self._next_connect_not_before = 0.0
        self._last_message_ts = time.time()
        self._ping_thread = threading.Thread(target=self._ping_loop, daemon=True)
        self._ping_thread.start()
        topics = self._subscribe_topics()
        for idx in range(0, len(topics), self._topics_per_subscribe):
            chunk = topics[idx : idx + self._topics_per_subscribe]
            ws.send(json.dumps({"op": "subscribe", "args": chunk}))
            # Avoid rate-limit bursts on large topic sets.
            time.sleep(self._bybit_subscribe_pause_seconds)
        logger.info(
            f"Bybit public depth WebSocket connected | topics={len(topics)} | "
            f"chunk_size={self._topics_per_subscribe}"
        )

    @staticmethod
    def _normalize_level(level) -> Optional[Dict[str, float]]:
        if isinstance(level, (list, tuple)) and len(level) >= 2:
            try:
                return {"price": float(level[0] or 0.0), "quantity": float(level[1] or 0.0)}
            except Exception:
                return None
        if isinstance(level, dict):
            try:
                return {
                    "price": float(level.get("price", 0.0) or 0.0),
                    "quantity": float(level.get("size", level.get("qty", 0.0)) or 0.0),
                }
            except Exception:
                return None
        return None

    def _book_lists(self, symbol: str) -> Dict[str, List[List[float]]]:
        book = self._books[symbol]
        bids = sorted(book["bids"].values(), key=lambda item: item["price"], reverse=True)
        asks = sorted(book["asks"].values(), key=lambda item: item["price"])
        return {
            "bids": [[item["price"], item["quantity"]] for item in bids[: self.depth_levels]],
            "asks": [[item["price"], item["quantity"]] for item in asks[: self.depth_levels]],
        }

    def _apply_orderbook(self, symbol: str, payload: Dict):
        book = self._books[symbol]
        event_type = str(payload.get("type", "") or "").lower()
        data = payload.get("data", {}) or {}
        if event_type == "snapshot":
            book["bids"] = {}
            book["asks"] = {}
        for side_name, target in (("b", "bids"), ("a", "asks")):
            for raw_level in data.get(side_name, []) or []:
                normalized = self._normalize_level(raw_level)
                if not normalized:
                    continue
                price = normalized["price"]
                quantity = normalized["quantity"]
                key = f"{price:.8f}"
                if quantity <= 0:
                    book[target].pop(key, None)
                else:
                    book[target][key] = {"price": price, "quantity": quantity}

        lists = self._book_lists(symbol)
        best_bid = lists["bids"][0][0] if lists["bids"] else 0.0
        best_ask = lists["asks"][0][0] if lists["asks"] else 0.0
        best_bid_qty = lists["bids"][0][1] if lists["bids"] else 0.0
        best_ask_qty = lists["asks"][0][1] if lists["asks"] else 0.0
        self.depth_buffer.update(
            symbol,
            {
                "bids": lists["bids"],
                "asks": lists["asks"],
                "best_bid": best_bid,
                "best_ask": best_ask,
                "best_bid_qty": best_bid_qty,
                "best_ask_qty": best_ask_qty,
                "last_depth_update_id": data.get("u"),
                "source": self.source,
            },
        )
        if best_bid > 0 and best_ask > 0:
            self.buffer.push(
                Tick(
                    symbol=symbol,
                    price=(best_bid + best_ask) / 2.0,
                    volume=0,
                    timestamp=datetime.now(timezone.utc),
                    bid=best_bid,
                    ask=best_ask,
                    market="CRYPTO",
                    source=self.source,
                )
            )

    def _apply_trade(self, symbol: str, payload: Dict):
        for item in payload.get("data", []) or []:
            price = float(item.get("p", 0.0) or 0.0)
            size = float(item.get("v", 0.0) or 0.0)
            trade_ts = datetime.fromtimestamp(float(item.get("T", time.time() * 1000)) / 1000.0, tz=timezone.utc)
            lists = self._book_lists(symbol)
            best_bid = lists["bids"][0][0] if lists["bids"] else None
            best_ask = lists["asks"][0][0] if lists["asks"] else None
            if price <= 0:
                continue
            self.buffer.push(
                Tick(
                    symbol=symbol,
                    price=price,
                    volume=int(max(0.0, size * price)),
                    timestamp=trade_ts,
                    bid=best_bid,
                    ask=best_ask,
                    market="CRYPTO",
                    source=self.source,
                )
            )

    def _on_message(self, ws, message):
        self._last_message_ts = time.time()
        try:
            payload = json.loads(message)
        except Exception:
            return

        if payload.get("op") in {"pong", "ping"} or str(payload.get("ret_msg", "")).lower() == "pong":
            return
        if payload.get("op") == "subscribe":
            success = bool(payload.get("success", False))
            if not success:
                logger.warning(f"Bybit subscribe failed: {payload}")
            return

        topic = str(payload.get("topic", "") or "")
        if not topic:
            return
        symbol = topic.split(".")[-1].upper()
        if topic.startswith("orderbook."):
            self._apply_orderbook(symbol, payload)
        elif topic.startswith("publicTrade."):
            self._apply_trade(symbol, payload)

    def _on_error(self, ws, error):
        message = str(error or "")
        if "429" in message or "too many requests" in message.lower():
            backoff_seconds = self._rate_limit_backoff_seconds + random.uniform(0.0, max(1.0, self._rate_limit_backoff_seconds * 0.25))
            self._next_connect_not_before = max(self._next_connect_not_before, time.time() + backoff_seconds)
        logger.warning(f"Bybit public depth WebSocket error: {error}")

    def _on_close(self, ws, *args):
        logger.info("Bybit public depth WebSocket closed")


class BybitTickerPollingFeed:
    """
    Public Bybit spot ticker poller.

    This is a reliability fallback for crypto signals when websocket depth is
    connected but not delivering enough live top-of-book updates to populate
    the signal engine.
    """

    API_URL = "https://api.bybit.com/v5/market/tickers"
    TESTNET_API_URL = "https://api-testnet.bybit.com/v5/market/tickers"

    def __init__(
        self,
        symbols: List[str],
        buffer: RealTimePriceBuffer = None,
        depth_buffer: PublicDepthBuffer = None,
        *,
        testnet: bool = False,
        interval_seconds: float = 3.0,
        timeout_seconds: float = 8.0,
    ):
        self.symbols = {str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()}
        self.buffer = buffer or PRICE_BUFFER
        self.depth_buffer = depth_buffer or DEPTH_BUFFER
        self.testnet = bool(testnet)
        self.interval_seconds = max(1.0, float(interval_seconds))
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.source = "bybit_rest_testnet" if self.testnet else "bybit_rest"
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._warned_empty = False

    def _url(self) -> str:
        return self.TESTNET_API_URL if self.testnet else self.API_URL

    def start(self, daemon: bool = True):
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=daemon, name="bybit-ticker-poll")
        self._thread.start()
        logger.info(f"Bybit ticker polling feed started for {len(self.symbols)} crypto symbols")

    def stop(self):
        self._running = False

    def _poll_loop(self):
        while self._running:
            started = time.time()
            try:
                response = requests.get(
                    self._url(),
                    params={"category": "spot"},
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
                entries = (((payload or {}).get("result") or {}).get("list") or [])
                pushed = 0
                for item in entries:
                    symbol = str(item.get("symbol") or "").upper()
                    if symbol not in self.symbols:
                        continue
                    last_price = float(item.get("lastPrice", 0.0) or 0.0)
                    bid = float(item.get("bid1Price", 0.0) or 0.0)
                    ask = float(item.get("ask1Price", 0.0) or 0.0)
                    bid_qty = float(item.get("bid1Size", 0.0) or 0.0)
                    ask_qty = float(item.get("ask1Size", 0.0) or 0.0)
                    if last_price <= 0 and bid <= 0 and ask <= 0:
                        continue
                    if last_price <= 0 and bid > 0 and ask > 0:
                        last_price = (bid + ask) / 2.0
                    if bid > 0 and ask > 0:
                        self.depth_buffer.update(
                            symbol,
                            {
                                "best_bid": bid,
                                "best_ask": ask,
                                "best_bid_qty": bid_qty,
                                "best_ask_qty": ask_qty,
                                "source": self.source,
                            },
                        )
                    self.buffer.push(
                        Tick(
                            symbol=symbol,
                            price=last_price,
                            volume=int(float(item.get("volume24h", 0.0) or 0.0)),
                            timestamp=datetime.now(timezone.utc),
                            bid=bid if bid > 0 else None,
                            ask=ask if ask > 0 else None,
                            market="CRYPTO",
                            source=self.source,
                        )
                    )
                    pushed += 1
                if pushed == 0 and not self._warned_empty:
                    logger.warning("Bybit ticker polling feed returned no matching symbols")
                    self._warned_empty = True
                elif pushed > 0:
                    self._warned_empty = False
            except Exception as exc:
                logger.warning(f"Bybit ticker polling feed error: {exc}")
            elapsed = time.time() - started
            time.sleep(max(0.0, self.interval_seconds - elapsed))


# ─────────────────────────────────────────────────────────────────────────────
# CCXT Pro WebSocket (Crypto — any exchange, no rate limits)
# ─────────────────────────────────────────────────────────────────────────────
class CCXTPublicWebSocket:
    """
    Public crypto market data via CCXT Pro (unified WebSocket library).

    Connects directly to 100+ exchanges (Binance, OKX, KuCoin, Gate.io, MEXC,
    Bybit, etc.) with no middleman — no Tardis, no per-exchange boilerplate,
    completely free public WebSocket access.

    This is a drop-in addition alongside BybitPublicDepthWebSocket.
    Prefer this feed because:
      - One library handles 100+ exchanges (failover = change CCXT_EXCHANGE)
      - CCXT Pro manages reconnections and pings internally
      - No custom ping/pong dance needed

    Requirements:
        pip install "ccxt[pro]"

    Symbols: accepts BTCUSDT format — auto-converts to BTC/USDT internally.

    Key env vars:
        CCXT_EXCHANGE          Exchange to stream from (default: binance)
        CRYPTO_FEED_ORDER      e.g. "ccxt,binance" to prefer this feed
    """

    # Ordered longest-first so "USDT" isn't confused with "USD"
    _QUOTE_CURRENCIES = ["USDT", "USDC", "BUSD", "TUSD", "FDUSD", "DAI", "USD", "BTC", "ETH", "BNB"]

    def __init__(
        self,
        symbols: List[str],
        buffer: RealTimePriceBuffer = None,
        depth_buffer: PublicDepthBuffer = None,
        *,
        exchange_id: str = "binance",
        watch_depth: bool = True,
        watch_trades: bool = True,
        reconnect_delay: float = 5.0,
    ):
        self.symbols = [str(s).strip().upper() for s in symbols if str(s).strip()]
        self.buffer = buffer or PRICE_BUFFER
        self.depth_buffer = depth_buffer or DEPTH_BUFFER
        self.exchange_id = (
            exchange_id or os.getenv("CCXT_EXCHANGE", "binance")
        ).lower().strip()
        self.watch_depth = bool(watch_depth)
        self.watch_trades = bool(watch_trades)
        self.source = f"ccxt_{self.exchange_id}"
        self._reconnect_delay = max(1.0, float(reconnect_delay))
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop = None  # asyncio.AbstractEventLoop, set in worker thread

    # ── symbol helpers ───────────────────────────────────────────────────────

    @classmethod
    def _to_ccxt_symbol(cls, symbol: str) -> str:
        """BTCUSDT → BTC/USDT"""
        s = symbol.upper()
        for quote in cls._QUOTE_CURRENCIES:
            if s.endswith(quote) and len(s) > len(quote):
                return f"{s[:-len(quote)]}/{quote}"
        return s

    @staticmethod
    def _from_ccxt_symbol(ccxt_symbol: str) -> str:
        """BTC/USDT → BTCUSDT"""
        return ccxt_symbol.replace("/", "").upper()

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self, daemon: bool = True):
        self._running = True
        self._thread = threading.Thread(
            target=self._run_event_loop,
            daemon=daemon,
            name=f"ccxt-{self.exchange_id}-ws",
        )
        self._thread.start()
        logger.info(
            f"CCXT {self.exchange_id} WebSocket started for {len(self.symbols)} crypto symbols"
        )

    def stop(self):
        self._running = False
        loop = self._loop
        if loop and not loop.is_closed():
            loop.call_soon_threadsafe(loop.stop)

    # ── event loop (runs in background thread) ───────────────────────────────

    def _run_event_loop(self):
        import asyncio as _asyncio
        self._loop = _asyncio.new_event_loop()
        _asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_main())
        except Exception as exc:
            logger.warning(f"CCXT {self.exchange_id} event loop exited: {exc}")
        finally:
            try:
                self._loop.close()
            except Exception:
                pass

    async def _async_main(self):
        """Outer retry loop — reconnects the whole exchange on any fatal error."""
        import asyncio as _asyncio
        while self._running:
            exchange = None
            try:
                try:
                    import ccxt.pro as ccxtpro  # type: ignore
                except ImportError:
                    logger.error(
                        "ccxt[pro] is not installed. "
                        "Run: pip install \'ccxt[pro]\' --break-system-packages"
                    )
                    return

                exchange_class = getattr(ccxtpro, self.exchange_id, None)
                if exchange_class is None:
                    logger.error(
                        f"CCXT exchange \'{self.exchange_id}\' not found. "
                        "Try: binance, bybit, okx, kucoin, gate, mexc, coinbase"
                    )
                    return

                exchange = exchange_class({"newUpdates": True})
                await exchange.load_markets()

                # Filter symbols to those actually listed on this exchange
                ccxt_symbols = []
                for raw in self.symbols:
                    csym = self._to_ccxt_symbol(raw)
                    if csym in exchange.markets:
                        ccxt_symbols.append(csym)
                    else:
                        logger.debug(
                            f"CCXT {self.exchange_id}: {raw} → {csym} not listed, skipping"
                        )

                if not ccxt_symbols:
                    logger.warning(
                        f"CCXT {self.exchange_id}: none of the {len(self.symbols)} symbols "
                        f"found on this exchange — retrying in {self._reconnect_delay}s"
                    )
                    await _asyncio.sleep(self._reconnect_delay)
                    continue

                logger.info(
                    f"CCXT {self.exchange_id}: streaming "
                    f"{len(ccxt_symbols)}/{len(self.symbols)} symbols"
                )

                tasks = []
                for csym in ccxt_symbols:
                    if self.watch_trades:
                        tasks.append(self._watch_trades_loop(exchange, csym))
                    if self.watch_depth:
                        tasks.append(self._watch_orderbook_loop(exchange, csym))

                if tasks:
                    await _asyncio.gather(*tasks, return_exceptions=True)

            except Exception as exc:
                logger.warning(f"CCXT {self.exchange_id} stream error: {exc}")
            finally:
                if exchange is not None:
                    try:
                        await exchange.close()
                    except Exception:
                        pass

            if self._running:
                await _asyncio.sleep(self._reconnect_delay + random.random())

    async def _watch_trades_loop(self, exchange, ccxt_symbol: str):
        """Per-symbol trade watcher — never propagates exceptions."""
        import asyncio as _asyncio
        raw_symbol = self._from_ccxt_symbol(ccxt_symbol)
        while self._running:
            try:
                trades = await exchange.watch_trades(ccxt_symbol)
                for trade in trades:
                    price = float(trade.get("price") or 0.0)
                    if price <= 0:
                        continue
                    amount = float(trade.get("amount") or 0.0)
                    ts_ms = trade.get("timestamp") or int(time.time() * 1000)
                    self.buffer.push(
                        Tick(
                            symbol=raw_symbol,
                            price=price,
                            volume=int(max(0.0, price * amount)),
                            timestamp=datetime.fromtimestamp(
                                ts_ms / 1000.0, tz=timezone.utc
                            ),
                            market="CRYPTO",
                            source=self.source,
                        )
                    )
            except _asyncio.CancelledError:
                return
            except Exception as exc:
                logger.debug(
                    f"CCXT {self.exchange_id} trade [{ccxt_symbol}]: {exc}"
                )
                await _asyncio.sleep(2.0)

    async def _watch_orderbook_loop(self, exchange, ccxt_symbol: str):
        """Per-symbol orderbook watcher — never propagates exceptions."""
        import asyncio as _asyncio
        raw_symbol = self._from_ccxt_symbol(ccxt_symbol)
        while self._running:
            try:
                ob = await exchange.watch_order_book(ccxt_symbol, limit=20)
                bids = [[float(b[0]), float(b[1])] for b in (ob.get("bids") or [])[:20]]
                asks = [[float(a[0]), float(a[1])] for a in (ob.get("asks") or [])[:20]]
                best_bid = bids[0][0] if bids else 0.0
                best_ask = asks[0][0] if asks else 0.0
                self.depth_buffer.update(
                    raw_symbol,
                    {
                        "bids": bids,
                        "asks": asks,
                        "best_bid": best_bid,
                        "best_ask": best_ask,
                        "best_bid_qty": bids[0][1] if bids else 0.0,
                        "best_ask_qty": asks[0][1] if asks else 0.0,
                        "source": self.source,
                    },
                )
                if best_bid > 0 and best_ask > 0:
                    self.buffer.push(
                        Tick(
                            symbol=raw_symbol,
                            price=(best_bid + best_ask) / 2.0,
                            volume=0,
                            timestamp=datetime.now(timezone.utc),
                            bid=best_bid,
                            ask=best_ask,
                            market="CRYPTO",
                            source=self.source,
                        )
                    )
            except _asyncio.CancelledError:
                return
            except Exception as exc:
                logger.debug(
                    f"CCXT {self.exchange_id} orderbook [{ccxt_symbol}]: {exc}"
                )
                await _asyncio.sleep(2.0)




# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Finnhub WebSocket (US Markets)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class FinnhubWebSocket:
    """
    Real-time US stock data via Finnhub WebSocket.
    Free tier: 50 symbols, sub-second data.
    
    Sign up: https://finnhub.io (free API key)
    
    Message format:
    {"type":"trade","data":[{"p":150.25,"s":"AAPL","t":1640000000000,"v":100}]}
    """

    WS_URL = "wss://ws.finnhub.io?token={api_key}"

    def __init__(
        self,
        api_key: str,
        symbols: List[str],
        buffer: RealTimePriceBuffer = None,
        on_signal_callback: Optional[Callable] = None,
    ):
        self.api_key = api_key
        self.symbols = symbols[:50]  # Free tier limit
        self.buffer = buffer or PRICE_BUFFER
        self.on_signal = on_signal_callback
        self.ws: Optional[websocket.WebSocketApp] = None
        self._running = False
        self._latency_tracker: deque = deque(maxlen=100)
        self._reconnect_base_delay_seconds = float(os.getenv("FINNHUB_WS_RECONNECT_BASE_DELAY_SECONDS", "3"))
        self._reconnect_max_delay_seconds = float(os.getenv("FINNHUB_WS_RECONNECT_MAX_DELAY_SECONDS", "60"))
        self._ping_interval_seconds = float(os.getenv("FINNHUB_WS_PING_INTERVAL_SECONDS", "20"))
        self._ping_timeout_seconds = float(os.getenv("FINNHUB_WS_PING_TIMEOUT_SECONDS", "10"))
        self._stale_seconds = float(os.getenv("FINNHUB_WS_STALE_SECONDS", "30"))
        self._last_message_ts = 0.0
        self._watchdog_thread: Optional[threading.Thread] = None

    def start(self, daemon: bool = True):
        """Start the WebSocket connection in a background thread."""
        self._running = True
        thread = threading.Thread(target=self._run_forever, daemon=daemon)
        thread.start()
        if self._watchdog_thread is None or not self._watchdog_thread.is_alive():
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop, daemon=True, name="finnhub-ws-watchdog"
            )
            self._watchdog_thread.start()
        logger.info(f"Finnhub WebSocket started for {len(self.symbols)} symbols")

    def stop(self):
        self._running = False
        if self.ws:
            self.ws.close()

    def _watchdog_loop(self):
        while self._running:
            time.sleep(5)
            if not self._running or self._stale_seconds <= 0:
                continue
            last_ts = float(self._last_message_ts or 0.0)
            if last_ts <= 0:
                continue
            if time.time() - last_ts > float(self._stale_seconds):
                logger.warning(
                    f"Finnhub WebSocket stale for >{int(self._stale_seconds)}s; forcing reconnect"
                )
                try:
                    if self.ws:
                        self.ws.close()
                except Exception:
                    pass

    def _run_forever(self):
        """Keep reconnecting if the WebSocket drops."""
        delay = max(0.5, float(self._reconnect_base_delay_seconds))
        max_delay = max(delay, float(self._reconnect_max_delay_seconds))
        ping_interval = self._ping_interval_seconds if self._ping_interval_seconds > 0 else None
        ping_timeout = self._ping_timeout_seconds if self._ping_timeout_seconds > 0 else None
        while self._running:
            start_ts = time.time()
            try:
                url = self.WS_URL.format(api_key=self.api_key)
                self.ws = websocket.WebSocketApp(
                    url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self.ws.run_forever(ping_interval=ping_interval, ping_timeout=ping_timeout)
            except Exception as e:
                logger.error(f"Finnhub WebSocket error: {e}")

            if self._running:
                runtime = time.time() - start_ts
                if runtime < 15:
                    delay = min(max_delay, delay * 2.0)
                else:
                    delay = max(0.5, float(self._reconnect_base_delay_seconds))
                sleep_for = min(max_delay, delay) + random.random()
                logger.info(f"Finnhub reconnect in {sleep_for:.1f}s (backoff)...")
                time.sleep(sleep_for)

    def _on_open(self, ws):
        """Subscribe to all symbols on connection."""
        self._last_message_ts = time.time()
        logger.info("Finnhub WebSocket connected")
        for symbol in self.symbols:
            ws.send(json.dumps({"type": "subscribe", "symbol": symbol}))
        logger.info(f"Subscribed to {len(self.symbols)} symbols")

    def _on_message(self, ws, message):
        """Parse incoming tick data."""
        self._last_message_ts = time.time()
        recv_time = datetime.now(timezone.utc)
        try:
            data = json.loads(message)
            if data.get("type") != "trade":
                return

            for trade in data.get("data", []):
                exchange_time_ms = trade.get("t", recv_time.timestamp() * 1000)
                tick_time = datetime.fromtimestamp(exchange_time_ms / 1000, tz=timezone.utc)

                # Latency measurement
                latency_ms = (recv_time.timestamp() - exchange_time_ms / 1000) * 1000
                self._latency_tracker.append(latency_ms)

                tick = Tick(
                    symbol=trade["s"],
                    price=float(trade["p"]),
                    volume=int(trade.get("v", 0)),
                    timestamp=tick_time,
                    market="US",
                )
                self.buffer.push(tick)

        except Exception as e:
            logger.debug(f"Message parse error: {e}")

    def _on_error(self, ws, error):
        logger.warning(f"Finnhub WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        logger.info(f"Finnhub WebSocket closed: {close_status_code}")

    @property
    def avg_latency_ms(self) -> float:
        if not self._latency_tracker:
            return 0.0
        return sum(self._latency_tracker) / len(self._latency_tracker)

# ─────────────────────────────────────────────────────────────────────────────
# Polygon.io IEX WebSocket (US Stocks — free real-time feed)
# ─────────────────────────────────────────────────────────────────────────────
class PolygonIEXWebSocket:
    """
    Real-time US stock tick data via Polygon.io's free IEX WebSocket.

    Why better than Finnhub:
      - No 50-symbol limit — subscribe to the entire US equity universe
      - IEX is a real exchange (~3-5% of total US volume)
      - Price tracks the overall market almost perfectly for signal generation
      - Free API key — no credit card required

    Sign up: https://polygon.io (free Starter plan)
    Set POLYGON_API_KEY in your .env file.

    Protocol:
        1. Connect → receive {"ev":"status","status":"connected"}
        2. Send    → {"action":"auth","params":"<key>"}
        3. Receive → {"ev":"status","status":"auth_success"}
        4. Send    → {"action":"subscribe","params":"T.AAPL,T.MSFT,..."}
        5. Receive → [{"ev":"T","sym":"AAPL","p":150.25,"s":100,"t":1640000000000}]

    Env vars:
        POLYGON_API_KEY              Required — your free Polygon API key
        POLYGON_SUBSCRIBE_QUOTES     Set "1" to also subscribe to Q.* bid/ask quotes
    """

    WS_URL = "wss://socket.polygon.io/stocks"

    def __init__(
        self,
        api_key: str,
        symbols: List[str],
        buffer: RealTimePriceBuffer = None,
        *,
        subscribe_quotes: bool = False,
    ):
        self.api_key = api_key
        self.symbols = [str(s).strip().upper() for s in symbols if str(s).strip()]
        self.buffer = buffer or PRICE_BUFFER
        self.subscribe_quotes = bool(subscribe_quotes) or os.getenv(
            "POLYGON_SUBSCRIBE_QUOTES", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.ws: Optional[websocket.WebSocketApp] = None
        self._running = False
        self._authenticated = False
        self._reconnect_delay = 5.0
        self._latency_tracker: deque = deque(maxlen=100)

    def start(self, daemon: bool = True):
        self._running = True
        thread = threading.Thread(
            target=self._run_forever, daemon=daemon, name="polygon-iex-ws"
        )
        thread.start()
        logger.info(
            f"Polygon IEX WebSocket started for {len(self.symbols)} US symbols"
        )

    def stop(self):
        self._running = False
        if self.ws:
            self.ws.close()

    def _run_forever(self):
        while self._running:
            start_ts = time.time()
            try:
                self._authenticated = False
                self.ws = websocket.WebSocketApp(
                    self.WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self.ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as exc:
                logger.warning(f"Polygon IEX WebSocket error: {exc}")
            if self._running:
                runtime = time.time() - start_ts
                # Exponential backoff if connection is flapping
                if runtime < 15:
                    self._reconnect_delay = min(60.0, self._reconnect_delay * 1.5)
                else:
                    self._reconnect_delay = 5.0
                logger.info(
                    f"Polygon IEX reconnecting in {self._reconnect_delay:.1f}s..."
                )
                time.sleep(self._reconnect_delay + random.uniform(0, 1))

    def _on_open(self, ws):
        # Polygon sends the "connected" status event first; auth happens there
        logger.info("Polygon IEX WebSocket connection established — awaiting server hello")

    def _on_message(self, ws, message):
        recv_time = datetime.now(timezone.utc)
        try:
            events = json.loads(message)
            if not isinstance(events, list):
                return

            for event in events:
                ev = event.get("ev", "")

                # ── Connection / auth handshake ──────────────────────────────
                if ev == "status":
                    status = str(event.get("status", ""))
                    msg = str(event.get("message", ""))

                    if status == "connected":
                        ws.send(json.dumps({"action": "auth", "params": self.api_key}))

                    elif status == "auth_success":
                        self._authenticated = True
                        self._reconnect_delay = 5.0  # reset backoff on good auth
                        # Build subscription string: T.AAPL,T.MSFT,...
                        subs = [f"T.{sym}" for sym in self.symbols]
                        if self.subscribe_quotes:
                            subs += [f"Q.{sym}" for sym in self.symbols]
                        ws.send(
                            json.dumps(
                                {"action": "subscribe", "params": ",".join(subs)}
                            )
                        )
                        logger.info(
                            f"Polygon IEX: authenticated — subscribed to "
                            f"{len(self.symbols)} symbols"
                            + (" + quotes" if self.subscribe_quotes else "")
                        )

                    elif status == "auth_failed":
                        logger.error(
                            f"Polygon IEX auth failed: {msg}. "
                            "Check POLYGON_API_KEY in your .env"
                        )
                        ws.close()

                    elif status == "max_connections":
                        logger.warning(
                            "Polygon IEX: max connections reached — "
                            "close another session or upgrade plan"
                        )
                        ws.close()
                    continue

                if not self._authenticated:
                    continue

                # ── Trade event ──────────────────────────────────────────────
                if ev == "T":
                    price = float(event.get("p", 0.0) or 0.0)
                    if price <= 0:
                        continue
                    ts_ms = event.get("t") or int(recv_time.timestamp() * 1000)
                    tick_time = datetime.fromtimestamp(
                        ts_ms / 1000.0, tz=timezone.utc
                    )
                    latency_ms = (recv_time.timestamp() - ts_ms / 1000.0) * 1000
                    self._latency_tracker.append(latency_ms)
                    symbol = str(event.get("sym", "") or "")
                    if not symbol:
                        continue
                    self.buffer.push(
                        Tick(
                            symbol=symbol,
                            price=price,
                            volume=int(event.get("s", 0) or 0),  # trade size in shares
                            timestamp=tick_time,
                            market="US",
                            source="polygon_iex",
                        )
                    )

                # ── Quote event (bid/ask) ────────────────────────────────────
                elif ev == "Q":
                    bid = float(event.get("bp", 0.0) or 0.0)
                    ask = float(event.get("ap", 0.0) or 0.0)
                    symbol = str(event.get("sym", "") or "")
                    if not symbol or bid <= 0 or ask <= 0:
                        continue
                    self.buffer.push(
                        Tick(
                            symbol=symbol,
                            price=(bid + ask) / 2.0,
                            volume=0,
                            timestamp=recv_time,
                            bid=bid,
                            ask=ask,
                            market="US",
                            source="polygon_iex",
                        )
                    )

        except Exception as exc:
            logger.debug(f"Polygon IEX message parse error: {exc}")

    def _on_error(self, ws, error):
        logger.warning(f"Polygon IEX WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        logger.info(
            f"Polygon IEX WebSocket closed: {close_status_code} {close_msg or ''}".strip()
        )

    @property
    def avg_latency_ms(self) -> float:
        if not self._latency_tracker:
            return 0.0
        return sum(self._latency_tracker) / len(self._latency_tracker)




# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Upstox WebSocket (India NSE/BSE)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class UpstoxWebSocket:
    """
    Real-time NSE/BSE data via Upstox WebSocket API.
    Free for Upstox account holders.
    
    Sign up: https://upstox.com/developer (free)
    Requires: Upstox trading account + API key
    
    The Upstox WebSocket uses a different binary protocol (protobuf).
    This implementation handles both JSON and protobuf modes.
    """

    WS_URL = "wss://api.upstox.com/v2/feed/market-data-feed"

    # NSE instrument token format: NSE_EQ|{isin}
    # We convert our standard NSE symbols to Upstox format
    SYMBOL_MAP = {
        "RELIANCE.NS": "NSE_EQ|INE002A01018",
        "TCS.NS": "NSE_EQ|INE467B01029",
        "INFY.NS": "NSE_EQ|INE009A01021",
        "HDFCBANK.NS": "NSE_EQ|INE040A01034",
        "ICICIBANK.NS": "NSE_EQ|INE090A01021",
        "SBIN.NS": "NSE_EQ|INE062A01020",
        "WIPRO.NS": "NSE_EQ|INE075A01022",
        "AXISBANK.NS": "NSE_EQ|INE238A01034",
        "^NSEI": "NSE_INDEX|Nifty 50",
        "^NSEBANK": "NSE_INDEX|Nifty Bank",
    }

    def __init__(
        self,
        access_token: str,
        symbols: List[str],
        buffer: RealTimePriceBuffer = None,
    ):
        self.access_token = access_token
        self.nse_symbols = [s for s in symbols if s.endswith(".NS") or s.startswith("^")]
        self.buffer = buffer or PRICE_BUFFER
        self.ws: Optional[websocket.WebSocketApp] = None
        self._running = False
        self._reconnect_base_delay_seconds = float(os.getenv("UPSTOX_WS_RECONNECT_BASE_DELAY_SECONDS", "3"))
        self._reconnect_max_delay_seconds = float(os.getenv("UPSTOX_WS_RECONNECT_MAX_DELAY_SECONDS", "60"))
        self._ping_interval_seconds = float(os.getenv("UPSTOX_WS_PING_INTERVAL_SECONDS", "20"))
        self._ping_timeout_seconds = float(os.getenv("UPSTOX_WS_PING_TIMEOUT_SECONDS", "10"))
        self._stale_seconds = float(os.getenv("UPSTOX_WS_STALE_SECONDS", "30"))
        self._last_message_ts = 0.0
        self._auth_failed = False
        self._watchdog_thread: Optional[threading.Thread] = None

    def start(self, daemon: bool = True):
        self._running = True
        thread = threading.Thread(target=self._run_forever, daemon=daemon)
        thread.start()
        if self._watchdog_thread is None or not self._watchdog_thread.is_alive():
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop, daemon=True, name="upstox-ws-watchdog"
            )
            self._watchdog_thread.start()
        logger.info(f"Upstox WebSocket started for {len(self.nse_symbols)} NSE symbols")

    def stop(self):
        self._running = False
        if self.ws:
            self.ws.close()

    def _watchdog_loop(self):
        while self._running:
            time.sleep(5)
            if not self._running or self._stale_seconds <= 0:
                continue
            last_ts = float(self._last_message_ts or 0.0)
            if last_ts <= 0:
                continue
            if time.time() - last_ts > float(self._stale_seconds):
                logger.warning(
                    f"Upstox WebSocket stale for >{int(self._stale_seconds)}s; forcing reconnect"
                )
                try:
                    if self.ws:
                        self.ws.close()
                except Exception:
                    pass

    def _run_forever(self):
        if not self.access_token or not str(self.access_token).strip():
            logger.warning("Upstox WebSocket: no access token set — skipping connection")
            return

        delay = max(0.5, float(self._reconnect_base_delay_seconds))
        max_delay = max(delay, float(self._reconnect_max_delay_seconds))
        ping_interval = self._ping_interval_seconds if self._ping_interval_seconds > 0 else None
        ping_timeout = self._ping_timeout_seconds if self._ping_timeout_seconds > 0 else None
        while self._running:
            start_ts = time.time()
            try:
                self.ws = websocket.WebSocketApp(
                    self.WS_URL,
                    header={"Authorization": f"Bearer {self.access_token}"},
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self.ws.run_forever(ping_interval=ping_interval, ping_timeout=ping_timeout)
            except Exception as e:
                logger.error(f"Upstox WebSocket error: {e}")

            # Stop retrying on auth failures — token is invalid or missing
            if getattr(self, "_auth_failed", False):
                logger.error(
                    "Upstox WebSocket: auth failed (401) — stopping reconnect loop. "
                    "Set a valid UPSTOX_ACCESS_TOKEN or leave it unset to disable."
                )
                self._running = False
                return

            if self._running:
                runtime = time.time() - start_ts
                if runtime < 15:
                    delay = min(max_delay, delay * 2.0)
                else:
                    delay = max(0.5, float(self._reconnect_base_delay_seconds))
                sleep_for = min(max_delay, delay) + random.random()
                logger.info(f"Upstox reconnect in {sleep_for:.1f}s (backoff)...")
                time.sleep(sleep_for)

    def _on_open(self, ws):
        self._auth_failed = False
        self._last_message_ts = time.time()
        # Subscribe to instruments
        instrument_keys = [
            self.SYMBOL_MAP.get(s) for s in self.nse_symbols
            if s in self.SYMBOL_MAP
        ]
        if instrument_keys:
            ws.send(json.dumps({
                "guid": "upstox-feed",
                "method": "sub",
                "data": {
                    "mode": "full",
                    "instrumentKeys": [k for k in instrument_keys if k],
                }
            }))
        logger.info("Upstox WebSocket connected and subscribed")

    def _on_message(self, ws, message):
        """Parse Upstox feed message (JSON mode)."""
        self._last_message_ts = time.time()
        try:
            data = json.loads(message) if isinstance(message, str) else {}
            # Upstox v2 JSON message format
            feeds = data.get("feeds", {})
            for instrument_key, feed_data in feeds.items():
                # Map back from instrument key to our symbol
                symbol = next(
                    (sym for sym, key in self.SYMBOL_MAP.items() if key == instrument_key),
                    instrument_key
                )
                ltp_data = feed_data.get("ff", {}).get("marketFF", {})
                ltpc = ltp_data.get("ltpc", {})
                ltp = ltpc.get("ltp")

                if ltp:
                    tick = Tick(
                        symbol=symbol,
                        price=float(ltp),
                        volume=int(ltp_data.get("tv", 0)),
                        timestamp=datetime.now(timezone.utc),
                        bid=ltpc.get("cp"),
                        market="IN",
                    )
                    self.buffer.push(tick)
        except Exception as e:
            logger.debug(f"Upstox parse error: {e}")

    def _on_error(self, ws, error):
        error_str = str(error)
        if "401" in error_str or "Unauthorized" in error_str.lower():
            self._auth_failed = True
            logger.error(
                "Upstox WebSocket: 401 Unauthorized — invalid or expired token. "
                "Unset UPSTOX_ACCESS_TOKEN to disable Upstox feed."
            )
        else:
            logger.warning(f"Upstox WebSocket error: {error}")

    def _on_close(self, ws, *args):
        logger.info("Upstox WebSocket closed")


# ─────────────────────────────────────────────────────────────────────────────
# NSEPython Polling Feed (India NSE — free, no account required)
# ─────────────────────────────────────────────────────────────────────────────
class NSEPythonPollingFeed:
    """
    Free NSE equity feed via the nsepython library.

    nsepython scrapes NSE's own public quote endpoints, so no API key, OAuth
    token, or KYC is required. This is the zero-cost alternative to
    UpstoxWebSocket for users who don't have an Upstox trading account.

    Poll interval defaults to 3s (env NSE_POLL_INTERVAL_SECONDS).
    Index symbols (prefixed with '^') are skipped — nse_eq is equities-only.
    """

    def __init__(
        self,
        symbols: List[str],
        buffer: RealTimePriceBuffer = None,
        *,
        interval_seconds: float = 3.0,
        batch_delay_seconds: float = 0.1,
    ):
        cleaned: List[str] = []
        for raw in symbols or []:
            sym = str(raw).strip()
            if not sym or sym.startswith("^"):
                continue
            cleaned.append(sym)
        self.symbols = cleaned
        self.buffer = buffer or PRICE_BUFFER
        self.interval_seconds = max(1.0, float(interval_seconds))
        self.batch_delay_seconds = max(0.0, float(batch_delay_seconds))
        self.source = "nsepython"
        self._running = False
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def _strip_ns(symbol: str) -> str:
        return symbol[:-3] if symbol.endswith(".NS") else symbol

    def start(self, daemon: bool = True):
        if not self.symbols:
            logger.info("NSEPython polling feed has no equity symbols to poll; not starting")
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=daemon, name="nsepython-poll")
        self._thread.start()
        logger.info(
            f"NSEPython polling feed started for {len(self.symbols)} NSE equity symbols "
            f"(interval={self.interval_seconds}s)"
        )

    def stop(self):
        self._running = False

    def _poll_loop(self):
        try:
            from nsepython import nse_eq
        except Exception as exc:
            logger.warning(
                f"nsepython not available ({exc}); NSE free feed disabled. "
                "Install with: pip install nsepython"
            )
            self._running = False
            return

        while self._running:
            started = time.time()
            pushed = 0
            for symbol in self.symbols:
                if not self._running:
                    break
                nse_code = self._strip_ns(symbol)
                try:
                    quote = nse_eq(nse_code)
                except Exception as exc:
                    logger.debug(f"nsepython fetch error for {symbol}: {exc}")
                    if self.batch_delay_seconds:
                        time.sleep(self.batch_delay_seconds)
                    continue

                try:
                    price_info = (quote or {}).get("priceInfo") or {}
                    last_price = float(price_info.get("lastPrice") or 0.0)
                    if last_price <= 0:
                        if self.batch_delay_seconds:
                            time.sleep(self.batch_delay_seconds)
                        continue

                    intraday = price_info.get("intraDayHighLow") or {}
                    open_price = price_info.get("open")
                    high_price = intraday.get("max") if intraday else price_info.get("intraDayHighLow", {}).get("max")
                    low_price = intraday.get("min") if intraday else price_info.get("intraDayHighLow", {}).get("min")

                    preopen = (quote or {}).get("preOpenMarket") or {}
                    security_wise = (quote or {}).get("securityWiseDP") or {}
                    volume_raw = (
                        price_info.get("totalTradedVolume")
                        or security_wise.get("quantityTraded")
                        or preopen.get("totalTradedVolume")
                        or 0
                    )
                    try:
                        volume = int(float(volume_raw or 0))
                    except (TypeError, ValueError):
                        volume = 0

                    tick = Tick(
                        symbol=symbol,
                        price=last_price,
                        volume=volume,
                        timestamp=datetime.now(timezone.utc),
                        market="IN",
                        source=self.source,
                    )
                    # Attach intraday OHLC as ad-hoc attributes for downstream consumers
                    # that introspect them; Tick dataclass itself is untouched.
                    if open_price is not None:
                        try:
                            setattr(tick, "open", float(open_price))
                        except (TypeError, ValueError):
                            pass
                    if high_price is not None:
                        try:
                            setattr(tick, "high", float(high_price))
                        except (TypeError, ValueError):
                            pass
                    if low_price is not None:
                        try:
                            setattr(tick, "low", float(low_price))
                        except (TypeError, ValueError):
                            pass

                    self.buffer.push(tick)
                    pushed += 1
                except Exception as exc:
                    logger.debug(f"nsepython parse error for {symbol}: {exc}")

                if self.batch_delay_seconds:
                    time.sleep(self.batch_delay_seconds)

            if pushed == 0:
                logger.debug("NSEPython polling feed produced no ticks this cycle")

            elapsed = time.time() - started
            time.sleep(max(0.0, self.interval_seconds - elapsed))


class MetaTrader5PollingFeed:
    """
    Lightweight MT5 tick bridge.

    MT5 does not expose a normal external WebSocket, so we poll symbol_info_tick
    and push the freshest prices into the shared live buffer.
    """

    def __init__(
        self,
        symbols: List[str],
        buffer: RealTimePriceBuffer = None,
        *,
        poll_interval_seconds: float = 1.0,
        symbol_map_path: str = "data/mt5_symbols.json",
        path: str = "",
        login: str = "",
        password: str = "",
        server: str = "",
    ):
        self.symbols = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
        self.buffer = buffer or PRICE_BUFFER
        self.poll_interval_seconds = max(0.2, float(poll_interval_seconds))
        self.symbol_map = self._load_symbol_map(symbol_map_path)
        self.path = str(path or "").strip()
        self.login = str(login or "").strip()
        self.password = str(password or "").strip()
        self.server = str(server or "").strip()
        self._running = False
        self._initialized = False
        try:
            import MetaTrader5 as mt5  # type: ignore

            self._mt5 = mt5
        except Exception as exc:
            self._mt5 = None
            logger.warning(f"MT5 realtime feed unavailable: {exc}")

    @staticmethod
    def _load_symbol_map(path_value: str) -> Dict[str, str]:
        path = Path(path_value)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        mapping: Dict[str, str] = {}
        if isinstance(payload, dict):
            for key, value in payload.items():
                if key and value:
                    mapping[str(key).strip().upper()] = str(value).strip()
        return mapping

    def _resolve_symbol(self, symbol: str) -> str:
        if symbol in self.symbol_map:
            return self.symbol_map[symbol]
        return symbol[:-3] if symbol.endswith(".NS") else symbol

    def _initialize(self) -> bool:
        if self._mt5 is None:
            return False
        if self._initialized:
            return True
        kwargs: Dict[str, object] = {"timeout": 5000}
        if self.path:
            kwargs["path"] = self.path
        if self.login:
            kwargs["login"] = int(self.login)
        if self.password:
            kwargs["password"] = self.password
        if self.server:
            kwargs["server"] = self.server
        self._initialized = bool(self._mt5.initialize(**kwargs))
        if not self._initialized:
            logger.warning(f"MT5 realtime initialize failed: {self._mt5.last_error()}")
        return self._initialized

    def start(self):
        if self._mt5 is None:
            return
        self._running = True
        threading.Thread(target=self._poll_loop, daemon=True).start()
        logger.info(f"MT5 polling feed started for {len(self.symbols)} symbols")

    def stop(self):
        self._running = False

    def _poll_loop(self):
        if not self._initialize():
            return
        while self._running:
            loop_started = time.time()
            for symbol in self.symbols:
                resolved_symbol = self._resolve_symbol(symbol)
                try:
                    if not self._mt5.symbol_select(resolved_symbol, True):
                        continue
                    tick = self._mt5.symbol_info_tick(resolved_symbol)
                    if tick is None:
                        continue
                    bid = float(getattr(tick, "bid", 0.0) or 0.0)
                    ask = float(getattr(tick, "ask", 0.0) or 0.0)
                    last = float(getattr(tick, "last", 0.0) or 0.0)
                    price = last or (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
                    if price <= 0:
                        continue
                    timestamp_seconds = float(getattr(tick, "time", time.time()) or time.time())
                    self.buffer.push(
                        Tick(
                            symbol=symbol,
                            price=price,
                            volume=int(float(getattr(tick, "volume_real", getattr(tick, "volume", 0.0)) or 0.0)),
                            timestamp=datetime.fromtimestamp(timestamp_seconds, tz=timezone.utc),
                            bid=bid if bid > 0 else None,
                            ask=ask if ask > 0 else None,
                            market="CRYPTO" if symbol.endswith(("USDT", "USDC", "BUSD")) else "IN" if symbol.endswith(".NS") else "US",
                            source="mt5",
                        )
                    )
                except Exception:
                    continue
            elapsed = time.time() - loop_started
            time.sleep(max(0.0, self.poll_interval_seconds - elapsed))


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Fallback: Polling for when WebSocket keys aren't available
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class PollingFallback:
    """
    REST polling fallback. Not as fast as WebSocket but works without extra API keys.
    Polls Yahoo Finance every N seconds to simulate real-time data.
    Latency: ~1-3 seconds (vs <500ms for WebSocket).
    """

    def __init__(
        self,
        symbols: List[str],
        interval_seconds: int = 5,
        buffer: RealTimePriceBuffer = None,
        batch_size: int = 20,
    ):
        self.symbols = list(dict.fromkeys(symbols or []))
        self.interval = interval_seconds
        self.buffer = buffer or PRICE_BUFFER
        self.batch_size = max(1, batch_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._batch_index = 0
        self._batches: List[List[str]] = [
            self.symbols[i : i + self.batch_size]
            for i in range(0, len(self.symbols), self.batch_size)
        ]

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info(f"Polling fallback started for {len(self.symbols)} symbols ({self.interval}s interval)")

    def stop(self):
        self._running = False

    def _poll_loop(self):
        import yfinance as yf
        while self._running:
            start_time = time.time()
            try:
                if not self.symbols:
                    time.sleep(self.interval)
                    continue

                if not self._batches:
                    time.sleep(self.interval)
                    continue

                batch = self._batches[self._batch_index % len(self._batches)]
                self._batch_index += 1
                tickers = yf.download(
                    " ".join(batch),
                    period="1d",
                    interval="1m",
                    progress=False,
                    auto_adjust=True,
                    threads=False,
                )

                if not tickers.empty and "Close" in tickers.columns:
                    close_row = tickers["Close"].iloc[-1]
                    vol_row = tickers.get("Volume", pd.Series()).iloc[-1] if hasattr(tickers, "__getitem__") else {}

                    for symbol in batch:
                        try:
                            price = float(close_row.get(symbol, close_row) if hasattr(close_row, 'get') else close_row)
                            if price > 0:
                                tick = Tick(
                                    symbol=symbol,
                                    price=price,
                                    volume=int(vol_row.get(symbol, 0) if hasattr(vol_row, "get") else 0),
                                    timestamp=datetime.now(timezone.utc),
                                    market="IN" if symbol.endswith(".NS") else "US",
                                    source="yfinance_1m",
                                )
                                self.buffer.push(tick)
                        except Exception:
                            pass

            except Exception as e:
                logger.debug(f"Polling error: {e}")

            elapsed = time.time() - start_time
            sleep_time = max(0, self.interval - elapsed)
            time.sleep(sleep_time)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Latency Monitor
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class LatencyMonitor:
    """
    Measures end-to-end latency across the live market-data pipeline.
    Professionals target < 500ms.
    """

    def __init__(self):
        self._measurements: deque = deque(maxlen=1000)
        self._checkpoints: Dict[str, float] = {}

    def mark(self, event_id: str, checkpoint: str):
        """Mark a timestamp for an event at a specific checkpoint."""
        self._checkpoints[f"{event_id}:{checkpoint}"] = time.time()

    def measure(self, event_id: str, from_checkpoint: str, to_checkpoint: str) -> Optional[float]:
        """Measure elapsed time between two checkpoints in milliseconds."""
        t1 = self._checkpoints.get(f"{event_id}:{from_checkpoint}")
        t2 = self._checkpoints.get(f"{event_id}:{to_checkpoint}")
        if t1 and t2:
            ms = (t2 - t1) * 1000
            self._measurements.append(ms)
            return ms
        return None

    @property
    def stats(self) -> Dict:
        if not self._measurements:
            return {}
        measurements = list(self._measurements)
        return {
            "avg_latency_ms": round(sum(measurements) / len(measurements), 1),
            "p50_ms": round(sorted(measurements)[len(measurements) // 2], 1),
            "p95_ms": round(sorted(measurements)[int(len(measurements) * 0.95)], 1),
            "p99_ms": round(sorted(measurements)[int(len(measurements) * 0.99)], 1),
            "target_met_pct": round(sum(1 for m in measurements if m < 500) / len(measurements) * 100, 1),
            "samples": len(measurements),
        }


LATENCY = LatencyMonitor()
