"""
Public no-KYC market-depth starter.

Uses Binance public websocket streams:
- <symbol>@depth20@100ms
- <symbol>@bookTicker

This is the fastest path to real order-book style data without broker KYC.
It is crypto, not US/NSE equities.
"""

from __future__ import annotations

import json
import os
import ssl
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List

import websocket


RAW_SYMBOLS = os.getenv("CRYPTO_DEPTH_SYMBOLS", "btcusdt,ethusdt,solusdt")
SYMBOLS = [part.strip().lower() for part in RAW_SYMBOLS.split(",") if part.strip()]
WS_BASE = "wss://stream.binance.com:9443/stream?streams="
PRINT_INTERVAL_SECONDS = float(os.getenv("CRYPTO_DEPTH_PRINT_INTERVAL_SECONDS", "1.0"))

BOOKS: Dict[str, Dict] = defaultdict(dict)
LAST_PRINT_TS = 0.0


def _stream_name(symbol: str) -> List[str]:
    return [f"{symbol}@depth20@100ms", f"{symbol}@bookTicker"]


def _build_ws_url(symbols: List[str]) -> str:
    streams: List[str] = []
    for symbol in symbols:
        streams.extend(_stream_name(symbol))
    return WS_BASE + "/".join(streams)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _top_n_notional(levels: List[List[str]], n: int = 5) -> float:
    total = 0.0
    for price, qty in levels[:n]:
        total += _safe_float(price) * _safe_float(qty)
    return total


def _book_imbalance(book: Dict, n: int = 5) -> float:
    bids = book.get("bids") or []
    asks = book.get("asks") or []
    bid_notional = _top_n_notional(bids, n=n)
    ask_notional = _top_n_notional(asks, n=n)
    denom = bid_notional + ask_notional
    if denom <= 0:
        return 0.0
    return (bid_notional - ask_notional) / denom


def _fmt_side(levels: List[List[str]]) -> str:
    if not levels:
        return "-"
    price, qty = levels[0]
    return f"{_safe_float(price):,.2f} x {_safe_float(qty):,.4f}"


def _print_snapshot(force: bool = False) -> None:
    global LAST_PRINT_TS
    now = time.time()
    if not force and (now - LAST_PRINT_TS) < PRINT_INTERVAL_SECONDS:
        return
    LAST_PRINT_TS = now

    print("=" * 88)
    print(datetime.now(timezone.utc).isoformat(), "UTC")
    for symbol in SYMBOLS:
        book = BOOKS.get(symbol, {})
        bids = book.get("bids") or []
        asks = book.get("asks") or []
        best_bid = _safe_float(book.get("best_bid") or (bids[0][0] if bids else 0.0))
        best_ask = _safe_float(book.get("best_ask") or (asks[0][0] if asks else 0.0))
        spread = best_ask - best_bid if best_bid > 0 and best_ask > 0 else 0.0
        imbalance_5 = _book_imbalance(book, n=5)
        imbalance_10 = _book_imbalance(book, n=10)
        print(
            f"{symbol.upper():<10} "
            f"bid { _fmt_side(bids):<24} "
            f"ask { _fmt_side(asks):<24} "
            f"spread {spread:,.4f} "
            f"imb5 {imbalance_5:+.3f} "
            f"imb10 {imbalance_10:+.3f}"
        )


def on_open(ws) -> None:
    print("Connected to Binance public market-data websocket")
    print("Streaming:", ", ".join(symbol.upper() for symbol in SYMBOLS))


def on_message(ws, message: str) -> None:
    try:
        payload = json.loads(message)
    except Exception:
        return

    stream = payload.get("stream", "")
    data = payload.get("data", {})
    if not stream or not isinstance(data, dict):
        return

    symbol = stream.split("@", 1)[0]
    book = BOOKS[symbol]

    if "@depth20" in stream:
        book["bids"] = data.get("bids", [])
        book["asks"] = data.get("asks", [])
        book["last_depth_update_id"] = data.get("lastUpdateId")
        book["depth_ts"] = datetime.now(timezone.utc).isoformat()
    elif "@bookTicker" in stream:
        book["best_bid"] = data.get("b")
        book["best_bid_qty"] = data.get("B")
        book["best_ask"] = data.get("a")
        book["best_ask_qty"] = data.get("A")
        book["book_ticker_update_id"] = data.get("u")
        book["ticker_ts"] = datetime.now(timezone.utc).isoformat()

    _print_snapshot()


def on_error(ws, error) -> None:
    print("WebSocket error:", error)


def on_close(ws, status_code, close_msg) -> None:
    print("WebSocket closed:", status_code, close_msg)


def main() -> None:
    if not SYMBOLS:
        raise SystemExit("Set CRYPTO_DEPTH_SYMBOLS to at least one symbol, e.g. btcusdt,ethusdt")

    url = _build_ws_url(SYMBOLS)
    ws = websocket.WebSocketApp(
        url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE}, ping_interval=20, ping_timeout=10)


if __name__ == "__main__":
    main()
