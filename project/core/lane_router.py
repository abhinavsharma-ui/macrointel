from __future__ import annotations

import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integration.defi_logging import get_logger
from integration.defi_types import FillEvent, JSONValue

LOGGER = get_logger("lane_router")

SignalPayload = dict[str, JSONValue]
SignalListener = Callable[[SignalPayload], None]

_LOCK = threading.RLock()
_SIGNAL_LISTENERS: list[SignalListener] = []
_SIGNAL_BUFFER: deque[SignalPayload] = deque(maxlen=10_000)
_FILL_BUFFER: deque[FillEvent] = deque(maxlen=10_000)


def register_signal_listener(listener: SignalListener) -> None:
    with _LOCK:
        if listener not in _SIGNAL_LISTENERS:
            _SIGNAL_LISTENERS.append(listener)


def unregister_signal_listener(listener: SignalListener) -> None:
    with _LOCK:
        if listener in _SIGNAL_LISTENERS:
            _SIGNAL_LISTENERS.remove(listener)


def publish_signal(signal: dict[str, object]) -> None:
    normalized = normalize_signal_payload(signal)
    if "ts" not in normalized:
        normalized["ts"] = time.time()
    with _LOCK:
        _SIGNAL_BUFFER.append(normalized)
        listeners = list(_SIGNAL_LISTENERS)
    for listener in listeners:
        try:
            listener(dict(normalized))
        except Exception as exc:
            LOGGER.warning("signal_listener_failed", error=str(exc))


def replay_signals(since_ts: float = 0.0) -> list[SignalPayload]:
    with _LOCK:
        return [
            dict(signal)
            for signal in _SIGNAL_BUFFER
            if float(signal.get("ts", 0.0) or 0.0) >= float(since_ts)
        ]


def record_fill(fill_event: FillEvent) -> None:
    with _LOCK:
        _FILL_BUFFER.append(fill_event)
    LOGGER.info(
        "fill_recorded",
        fill_id=fill_event.fill_id,
        status=fill_event.status,
        filled_usd=fill_event.filled_usd,
    )


def recent_fills(limit: int = 100) -> list[FillEvent]:
    with _LOCK:
        items = list(_FILL_BUFFER)[-max(1, int(limit)) :]
    return items


def normalize_signal_payload(payload: dict[str, object]) -> SignalPayload:
    return {
        str(key): _normalize_value(value)
        for key, value in payload.items()
    }


def _normalize_value(value: object) -> JSONValue:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {
            str(key): _normalize_value(nested_value)
            for key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    return str(value)
