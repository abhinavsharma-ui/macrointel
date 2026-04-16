from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import OrderedDict, deque
from pathlib import Path
from typing import AsyncGenerator, Iterable, Protocol

import websockets
from websockets.exceptions import ConnectionClosed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integration.defi_logging import get_logger
from integration.defi_types import AddressSnapshot, BlockEvent, JSONValue

LOGGER = get_logger("block_intel")


class _SmokeWebSocket(Protocol):
    async def recv(self) -> str | bytes:
        ...

    async def send(self, message: str) -> None:
        ...


def _normalize_address(address: str) -> str:
    return str(address or "").strip().lower()


class _BlockEventCache:
    def __init__(self, maxsize: int = 4096) -> None:
        self._maxsize = max(1, int(maxsize))
        self._entries: OrderedDict[tuple[str, int], BlockEvent] = OrderedDict()

    def add(self, event: BlockEvent) -> bool:
        key = (event.address, event.block)
        if key in self._entries:
            self._entries.move_to_end(key)
            return False
        self._entries[key] = event
        self._entries.move_to_end(key)
        if len(self._entries) > self._maxsize:
            self._entries.popitem(last=False)
        return True


class BlockIntelFeed:
    def __init__(self, feed_url: str, api_key: str) -> None:
        self.feed_url = str(feed_url or "").strip()
        self.api_key = str(api_key or "").strip()
        self._cache = _BlockEventCache(maxsize=4096)
        self._events_24h: dict[str, deque[BlockEvent]] = {}
        self._snapshots: dict[str, AddressSnapshot] = {}
        self._whale_threshold_usd = max(1.0, float(os.getenv("BLOCK_INTEL_WHALE_USD", "100000.0")))

    async def subscribe(self, addresses: list[str]) -> AsyncGenerator[BlockEvent, None]:
        subscribed = {_normalize_address(address) for address in addresses if str(address).strip()}
        if not subscribed:
            return

        backoff_seconds = 0.5
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        subscribe_message = {
            "op": "subscribe",
            "addresses": sorted(subscribed),
            "api_key": self.api_key,
        }

        while True:
            try:
                async with websockets.connect(
                    self.feed_url,
                    additional_headers=headers,
                    ping_interval=20.0,
                    ping_timeout=20.0,
                    close_timeout=2.0,
                ) as websocket:
                    await websocket.send(json.dumps(subscribe_message))
                    backoff_seconds = 0.5
                    async for raw_message in websocket:
                        for event in self._decode_events(raw_message, subscribed):
                            if not self._cache.add(event):
                                continue
                            self._remember_event(event)
                            LOGGER.info(
                                "block_event",
                                block=event.block,
                                addr=event.address,
                                delta_usd=round(event.delta_usd, 6),
                            )
                            yield event
            except asyncio.CancelledError:
                raise
            except (ConnectionClosed, OSError, ValueError, json.JSONDecodeError) as exc:
                LOGGER.warning(
                    "block_feed_reconnect",
                    error=str(exc),
                    backoff_s=round(backoff_seconds, 2),
                )
                await asyncio.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2.0, 30.0)

    def latest_snapshot(self, address: str) -> AddressSnapshot | None:
        return self._snapshots.get(_normalize_address(address))

    def _decode_events(self, raw_message: str | bytes, subscribed: set[str]) -> list[BlockEvent]:
        payload = json.loads(raw_message.decode("utf-8") if isinstance(raw_message, bytes) else raw_message)
        if not isinstance(payload, dict):
            return []
        if payload.get("confirmed") is False:
            return []

        block_number = int(payload.get("block") or payload.get("block_number") or 0)
        if block_number <= 0:
            return []

        raw_events = payload.get("events")
        if isinstance(raw_events, list):
            candidates = [event for event in raw_events if isinstance(event, dict)]
        else:
            candidates = [payload]

        parsed_events: list[BlockEvent] = []
        for item in candidates:
            event = self._to_block_event(block_number, item, subscribed)
            if event is not None:
                parsed_events.append(event)
        return parsed_events

    def _to_block_event(
        self,
        block_number: int,
        payload: dict[str, JSONValue],
        subscribed: set[str],
    ) -> BlockEvent | None:
        normalized_address = _normalize_address(str(payload.get("address") or payload.get("addr") or ""))
        if not normalized_address or normalized_address not in subscribed:
            return None

        raw_direction = str(payload.get("direction") or "in").strip().lower()
        direction = "out" if raw_direction == "out" else "in"
        delta_raw = payload.get("delta_usd") or payload.get("deltaUsd") or payload.get("usd_delta") or 0.0
        delta_value = float(delta_raw) if isinstance(delta_raw, (int, float, str)) else 0.0
        signed_delta = abs(delta_value) if direction == "in" else -abs(delta_value)
        event_ts_raw = payload.get("ts") or payload.get("timestamp") or time.time()
        event_ts = float(event_ts_raw) if isinstance(event_ts_raw, (int, float, str)) else time.time()

        return BlockEvent(
            block=block_number,
            address=normalized_address,
            delta_usd=signed_delta,
            direction=direction,
            ts=event_ts,
        )

    def _remember_event(self, event: BlockEvent) -> None:
        history = self._events_24h.setdefault(event.address, deque())
        history.append(event)
        cutoff = event.ts - 86_400.0
        while history and history[0].ts < cutoff:
            history.popleft()

        net_flow = sum(item.delta_usd for item in history)
        tx_count = len(history)
        whale_flag = any(abs(item.delta_usd) >= self._whale_threshold_usd for item in history)
        self._snapshots[event.address] = AddressSnapshot(
            address=event.address,
            net_flow_24h=round(net_flow, 6),
            tx_count_24h=tx_count,
            whale_flag=whale_flag,
            ts=event.ts,
        )


async def _smoke_test() -> int:
    address = "0xfeed000000000000000000000000000000000001"
    event_messages = [
        {
            "confirmed": True,
            "block": 101,
            "events": [
                {"address": address, "delta_usd": "250.5", "direction": "in", "ts": time.time()},
                {"address": "0xdead000000000000000000000000000000000000", "delta_usd": 999.0, "direction": "in", "ts": time.time()},
            ],
        },
        {
            "confirmed": True,
            "block": 102,
            "events": [
                {"address": address, "delta_usd": 20.0, "direction": "out", "ts": time.time()},
            ],
        },
    ]

    async def handler(websocket: _SmokeWebSocket) -> None:
        await websocket.recv()
        for message in event_messages:
            await websocket.send(json.dumps(message))
        await asyncio.sleep(0.1)

    server = await websockets.serve(handler, "127.0.0.1", 0)
    try:
        port = int(server.sockets[0].getsockname()[1])
        feed = BlockIntelFeed(feed_url=f"ws://127.0.0.1:{port}", api_key="stub")
        collected: list[BlockEvent] = []
        async for block_event in feed.subscribe([address]):
            collected.append(block_event)
            if len(collected) == 2:
                break
        snapshot = feed.latest_snapshot(address)
    finally:
        server.close()
        await server.wait_closed()

    print("metric\tvalue")
    print(f"events_collected\t{len(collected)}")
    print(f"last_block\t{collected[-1].block if collected else 0}")
    print(f"net_flow_24h\t{snapshot.net_flow_24h if snapshot is not None else 0.0}")
    print(f"tx_count_24h\t{snapshot.tx_count_24h if snapshot is not None else 0}")
    return 0 if len(collected) == 2 and snapshot is not None else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_smoke_test()))
