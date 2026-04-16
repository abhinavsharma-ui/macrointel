from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import httpx
import pandas as pd
from web3 import Web3

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integration.defi_logging import get_logger
from integration.defi_types import TokenomicsSignal

LOGGER = get_logger("buyback_tracker")
SUPPORTED_PROTOCOLS = {"lista_dao", "dfinity_mission70"}


class BuybackTracker:
    def __init__(self, protocols: list[str]) -> None:
        normalized = [str(protocol).strip().lower() for protocol in protocols if str(protocol).strip()]
        invalid = [protocol for protocol in normalized if protocol not in SUPPORTED_PROTOCOLS]
        if invalid:
            raise ValueError(f"unsupported protocols: {', '.join(invalid)}")
        self.protocols = normalized
        self.signal_df = pd.DataFrame(
            columns=[
                "protocol",
                "buyback_usd_7d",
                "revenue_usd_7d",
                "ratio",
                "trend",
                "size_scalar",
                "ts",
            ]
        )
        self._history: dict[str, deque[tuple[float, float]]] = {
            protocol: deque(maxlen=32)
            for protocol in self.protocols
        }
        self._state: dict[str, dict[str, float | str]] = {
            protocol: {"buyback_usd_7d": 0.0, "revenue_usd_7d": 0.0, "ratio": 0.0, "trend": "flat", "size_scalar": 1.0, "ts": 0.0}
            for protocol in self.protocols
        }
        self._transport: httpx.AsyncBaseTransport | None = None

    async def poll(self, interval_secs: int = 30) -> None:
        sleep_seconds = max(1, int(interval_secs))
        while True:
            await self._poll_once()
            await asyncio.sleep(sleep_seconds)

    def signal(self, protocol: str) -> TokenomicsSignal:
        key = str(protocol).strip().lower()
        state = self._state.get(key)
        if state is None:
            raise ValueError(f"unknown protocol: {protocol}")
        return TokenomicsSignal(
            protocol=key,
            ratio=float(state["ratio"]),
            trend=str(state["trend"]),
            size_scalar=float(state["size_scalar"]),
        )

    async def _poll_once(self) -> None:
        results = await asyncio.gather(
            *(self._fetch_protocol(protocol) for protocol in self.protocols),
            return_exceptions=True,
        )
        rows: list[dict[str, object]] = []
        now_ts = time.time()
        for protocol, result in zip(self.protocols, results, strict=True):
            if isinstance(result, Exception):
                LOGGER.warning("tokenomics_fetch_failed", protocol=protocol, error=str(result))
                continue
            buyback_usd_7d, revenue_usd_7d = result
            ratio = buyback_usd_7d / revenue_usd_7d if revenue_usd_7d > 0 else 0.0
            history = self._history[protocol]
            history.append((now_ts, ratio))
            trend = self._trend(protocol)
            size_scalar = self._size_scalar(ratio)
            self._state[protocol] = {
                "buyback_usd_7d": buyback_usd_7d,
                "revenue_usd_7d": revenue_usd_7d,
                "ratio": ratio,
                "trend": trend,
                "size_scalar": size_scalar,
                "ts": now_ts,
            }
            rows.append(
                {
                    "protocol": protocol,
                    "buyback_usd_7d": round(buyback_usd_7d, 6),
                    "revenue_usd_7d": round(revenue_usd_7d, 6),
                    "ratio": round(ratio, 6),
                    "trend": trend,
                    "size_scalar": size_scalar,
                    "ts": now_ts,
                }
            )

        if rows:
            self.signal_df = pd.DataFrame(rows)

    async def _fetch_protocol(self, protocol: str) -> tuple[float, float]:
        if protocol == "lista_dao":
            return await asyncio.to_thread(self._fetch_lista_dao_sync)
        return await self._fetch_dfinity_m70()

    def _fetch_lista_dao_sync(self) -> tuple[float, float]:
        fallback_buyback = float(os.getenv("LISTA_DAO_BUYBACK_USD_7D", "0"))
        fallback_revenue = float(os.getenv("LISTA_DAO_REVENUE_USD_7D", "0"))
        provider_url = str(os.getenv("LISTA_DAO_WEB3_PROVIDER", "")).strip()
        contract_address = str(os.getenv("LISTA_LIQUID_STAKING_ADDRESS", "")).strip()
        abi_path = str(os.getenv("LISTA_LIQUID_STAKING_ABI_PATH", "")).strip()
        if not provider_url or not contract_address or not abi_path:
            return fallback_buyback, fallback_revenue

        try:
            abi = json.loads(Path(abi_path).read_text(encoding="utf-8"))
            web3 = Web3(Web3.HTTPProvider(provider_url, request_kwargs={"timeout": 5}))
            contract = web3.eth.contract(address=Web3.to_checksum_address(contract_address), abi=abi)
            latest_block = web3.eth.block_number
            lookback_blocks = max(1, int(os.getenv("LISTA_DAO_LOOKBACK_BLOCKS", "50000")))
            from_block = max(0, latest_block - lookback_blocks)
            buyback_event_name = str(os.getenv("LISTA_DAO_BUYBACK_EVENT", "Buyback")).strip()
            revenue_event_name = str(os.getenv("LISTA_DAO_REVENUE_EVENT", "RevenueAccrued")).strip()
            decimals = max(0, int(os.getenv("LISTA_DAO_EVENT_DECIMALS", "18")))

            buyback_logs = getattr(contract.events, buyback_event_name)().get_logs(from_block=from_block, to_block=latest_block)
            revenue_logs = getattr(contract.events, revenue_event_name)().get_logs(from_block=from_block, to_block=latest_block)
            divisor = float(10**decimals)
            buyback_total = sum(self._extract_log_amount(log) for log in buyback_logs) / divisor
            revenue_total = sum(self._extract_log_amount(log) for log in revenue_logs) / divisor
            return buyback_total, revenue_total
        except Exception as exc:
            LOGGER.warning("lista_fetch_fallback", error=str(exc))
            return fallback_buyback, fallback_revenue

    async def _fetch_dfinity_m70(self) -> tuple[float, float]:
        async with httpx.AsyncClient(timeout=5.0, transport=self._transport) as client:
            response = await client.get("https://ic-api.internetcomputer.org/api/v3/metrics")
            response.raise_for_status()
            payload = response.json()
        buyback_usd = self._first_number(
            payload,
            ("buyback_usd_7d",),
            ("metrics", "buyback_usd_7d"),
            ("mission70", "buyback_usd_7d"),
            ("buyback7d",),
        )
        revenue_usd = self._first_number(
            payload,
            ("revenue_usd_7d",),
            ("metrics", "revenue_usd_7d"),
            ("mission70", "revenue_usd_7d"),
            ("revenue7d",),
        )
        return buyback_usd, revenue_usd

    def _trend(self, protocol: str) -> str:
        history = self._history[protocol]
        if len(history) < 2:
            return "flat"
        previous = history[-2][1]
        current = history[-1][1]
        if current > previous + 0.02:
            return "up"
        if current < previous - 0.02:
            return "down"
        return "flat"

    def _size_scalar(self, ratio: float) -> float:
        if ratio > 1.2:
            return 1.15
        if ratio < 0.8:
            return 0.85
        return 1.0

    def _extract_log_amount(self, log: object) -> float:
        if hasattr(log, "args"):
            args = getattr(log, "args")
            if isinstance(args, dict):
                for key in ("amount", "value", "buybackAmount", "revenueAmount"):
                    raw = args.get(key)
                    if isinstance(raw, (int, float)):
                        return float(raw)
        return 0.0

    def _first_number(self, payload: object, *paths: tuple[object, ...]) -> float:
        for path in paths:
            current: object = payload
            for part in path:
                if isinstance(part, int):
                    if not isinstance(current, list) or part >= len(current):
                        current = None
                        break
                    current = current[part]
                else:
                    if not isinstance(current, dict):
                        current = None
                        break
                    current = current.get(str(part))
            if isinstance(current, (int, float)):
                return float(current)
            if isinstance(current, str):
                try:
                    return float(current)
                except ValueError:
                    continue
        return 0.0


async def _smoke_test() -> int:
    os.environ["LISTA_DAO_BUYBACK_USD_7D"] = "1500000"
    os.environ["LISTA_DAO_REVENUE_USD_7D"] = "1200000"

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "metrics": {
                    "buyback_usd_7d": 500_000.0,
                    "revenue_usd_7d": 700_000.0,
                }
            },
        )

    tracker = BuybackTracker(["lista_dao", "dfinity_mission70"])
    tracker._transport = httpx.MockTransport(mock_handler)
    await tracker._poll_once()
    lista_signal = tracker.signal("lista_dao")
    dfinity_signal = tracker.signal("dfinity_mission70")

    print("metric\tvalue")
    print(f"rows\t{len(tracker.signal_df)}")
    print(f"lista_ratio\t{lista_signal.ratio}")
    print(f"dfinity_scalar\t{dfinity_signal.size_scalar}")
    return 0 if len(tracker.signal_df) == 2 and lista_signal.trend == "flat" and dfinity_signal.size_scalar == 0.85 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_smoke_test()))
