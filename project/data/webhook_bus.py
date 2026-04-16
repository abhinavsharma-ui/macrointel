from __future__ import annotations

import asyncio
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Awaitable, Callable

import httpx
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integration.defi_logging import get_logger
from integration.defi_types import JSONValue, WebhookEvent

LOGGER = get_logger("webhook_bus")

WebhookHandler = Callable[[WebhookEvent], Awaitable[None]]


class _WebhookEventModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    event_type: str
    payload: dict[str, object]
    ts: float


class WebhookBus:
    def __init__(self, port: int = 9000) -> None:
        self.port = int(port)
        self.app = FastAPI()
        self._handlers: dict[str, list[WebhookHandler]] = {}
        self._buffer: deque[WebhookEvent] = deque(maxlen=10_000)
        self._pending_tasks: set[asyncio.Task[None]] = set()
        self._dispatch_semaphore = asyncio.Semaphore(8)
        self._lock = threading.RLock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: uvicorn.Server | None = None

        @self.app.post("/event")
        async def receive_event(event_model: _WebhookEventModel) -> dict[str, str]:
            event = WebhookEvent(
                source=event_model.source,
                event_type=event_model.event_type,
                payload=_coerce_json_object(event_model.payload),
                ts=float(event_model.ts),
            )
            await self.publish_local(event)
            return {"status": "accepted"}

    def on(self, event_type: str) -> Callable[[WebhookHandler], WebhookHandler]:
        def decorator(async_fn: WebhookHandler) -> WebhookHandler:
            with self._lock:
                self._handlers.setdefault(event_type, []).append(async_fn)
            return async_fn

        return decorator

    async def publish_local(self, event: WebhookEvent) -> None:
        self._loop = asyncio.get_running_loop()
        with self._lock:
            self._buffer.append(event)
        origin_ts = self._origin_timestamp(event)
        dispatch_latency_ms = max(0.0, (time.time() - origin_ts) * 1000.0)
        if dispatch_latency_ms > 100.0:
            LOGGER.warning(
                "webhook_latency_exceeded",
                event_type=event.event_type,
                latency_ms=round(dispatch_latency_ms, 3),
                source=event.source,
            )
        await self._schedule_handlers(event)

    def publish_from_thread(self, event: WebhookEvent) -> None:
        loop = self._loop
        if loop is None:
            LOGGER.warning("webhook_loop_unavailable", event_type=event.event_type, source=event.source)
            return
        asyncio.run_coroutine_threadsafe(self.publish_local(event), loop)

    async def replay(self, event_type: str, since_ts: float) -> int:
        with self._lock:
            replay_events = [
                event
                for event in list(self._buffer)
                if event.event_type == event_type and event.ts >= float(since_ts)
            ]
        for event in replay_events:
            await self._schedule_handlers(event)
        return len(replay_events)

    async def serve(self, host: str = "0.0.0.0") -> None:
        self._loop = asyncio.get_running_loop()
        config = uvicorn.Config(
            self.app,
            host=host,
            port=self.port,
            log_level="warning",
            access_log=False,
            lifespan="off",
        )
        self._server = uvicorn.Server(config)
        await self._server.serve()

    async def shutdown(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._pending_tasks:
            await asyncio.gather(*tuple(self._pending_tasks), return_exceptions=True)

    async def _schedule_handlers(self, event: WebhookEvent) -> None:
        with self._lock:
            handlers = list(self._handlers.get(event.event_type, []))
        for handler in handlers:
            task = asyncio.create_task(self._run_handler(handler, event))
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)

    async def _run_handler(self, handler: WebhookHandler, event: WebhookEvent) -> None:
        async with self._dispatch_semaphore:
            try:
                await handler(event)
            except Exception as exc:
                LOGGER.error(
                    "webhook_handler_failed",
                    event_type=event.event_type,
                    source=event.source,
                    error=str(exc),
                )

    def _origin_timestamp(self, event: WebhookEvent) -> float:
        payload_ts = event.payload.get("ts")
        if isinstance(payload_ts, (int, float)):
            return float(payload_ts)
        if isinstance(payload_ts, str):
            try:
                return float(payload_ts)
            except ValueError:
                return float(event.ts)
        return float(event.ts)


def _coerce_json_value(value: object) -> JSONValue:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {
            str(key): _coerce_json_value(nested_value)
            for key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [_coerce_json_value(item) for item in value]
    return str(value)


def _coerce_json_object(payload: dict[str, object]) -> dict[str, JSONValue]:
    return {
        str(key): _coerce_json_value(value)
        for key, value in payload.items()
    }


async def _smoke_test() -> int:
    bus = WebhookBus(port=9001)
    handled: list[str] = []
    gate = asyncio.Event()

    @bus.on("lane_router.signal")
    async def _handler(event: WebhookEvent) -> None:
        handled.append(str(event.payload.get("asset") or "unknown"))
        if len(handled) >= 2:
            gate.set()

    transport = httpx.ASGITransport(app=bus.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        now_ts = time.time()
        response = await client.post(
            "/event",
            json={
                "source": "smoke",
                "event_type": "lane_router.signal",
                "payload": {"asset": "BTC", "ts": now_ts},
                "ts": now_ts,
            },
        )
        if response.status_code != 200:
            return 1
        await asyncio.sleep(0.05)
        replayed = await bus.replay("lane_router.signal", now_ts - 1.0)
        await asyncio.wait_for(gate.wait(), timeout=2.0)

    print("metric\tvalue")
    print(f"http_status\t{response.status_code}")
    print(f"handled_count\t{len(handled)}")
    print(f"replayed_count\t{replayed}")
    return 0 if len(handled) >= 2 and replayed == 1 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_smoke_test()))
