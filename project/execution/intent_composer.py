from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import httpx
from eth_abi import encode
from eth_utils import keccak

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integration.defi_logging import get_logger
from integration.defi_types import ChainConfig, CrossChainOrder, OrderLeg, OutputEstimate, TradeSignal

LOGGER = get_logger("intent_composer")

GASLESS_CROSS_CHAIN_ORDER_TYPE_STRING = (
    "GaslessCrossChainOrder("
    "address originSettler,"
    "address user,"
    "uint256 nonce,"
    "uint256 originChainId,"
    "uint32 openDeadline,"
    "uint32 fillDeadline,"
    "bytes32 orderDataType,"
    "bytes orderData)"
)


class IntentComposer:
    def __init__(self, chain_configs: dict[str, ChainConfig]) -> None:
        self.chain_configs = dict(chain_configs)
        self._nonce_seed = int(time.time() * 1_000)
        self._transport: httpx.AsyncBaseTransport | None = None

    def compose(self, signal: TradeSignal) -> CrossChainOrder:
        source = self._get_chain_config(signal.src_chain)
        destination = self._get_chain_config(signal.dst_chain)
        now_ts = int(time.time())
        nonce = self._next_nonce()
        fill_deadline = now_ts + max(30, int(signal.deadline_secs))
        initiate_deadline = now_ts + min(30, max(5, int(signal.deadline_secs // 6) or 5))
        exclusivity_deadline = min(fill_deadline, now_ts + max(15, int(signal.deadline_secs // 3) or 15))

        source_amount = self._usd_to_token_units(signal.size_usd, source.token_decimals)
        destination_amount = self._usd_to_token_units(signal.size_usd, destination.token_decimals)
        slippage_factor = max(0.0, min(float(signal.max_slippage_bps) / 10_000.0, 0.95))
        min_received_amount = max(1, int(destination_amount * (1.0 - slippage_factor)))

        implementation_payload = encode(
            [
                "address",
                "uint256",
                "uint256",
                "address",
                "address",
                "uint32",
                "bytes",
            ],
            [
                destination.output_token,
                destination_amount,
                destination.chain_id,
                destination.recipient,
                destination.exclusive_relayer,
                exclusivity_deadline,
                self._message_bytes(destination.message_hex),
            ],
        )

        order_data_type = f"0x{keccak(text=GASLESS_CROSS_CHAIN_ORDER_TYPE_STRING).hex()}"
        gasless_order = encode(
            [
                "address",
                "address",
                "uint256",
                "uint256",
                "uint32",
                "uint32",
                "bytes32",
                "bytes",
            ],
            [
                source.origin_settler,
                source.user_address,
                nonce,
                source.chain_id,
                initiate_deadline,
                fill_deadline,
                bytes.fromhex(order_data_type[2:]),
                implementation_payload,
            ],
        )

        max_spent = [
            OrderLeg(
                token=source.input_token,
                amount=source_amount,
                recipient=source.origin_settler,
                chain_id=source.chain_id,
            )
        ]
        min_received = [
            OrderLeg(
                token=destination.output_token,
                amount=min_received_amount,
                recipient=destination.recipient,
                chain_id=destination.chain_id,
            )
        ]

        return CrossChainOrder(
            orderDataType=order_data_type,
            orderData=gasless_order,
            fillDeadline=fill_deadline,
            exclusivityDeadline=exclusivity_deadline,
            exclusiveRelayer=destination.exclusive_relayer,
            nonce=nonce,
            originChainId=source.chain_id,
            initiateDeadline=initiate_deadline,
            maxSpent=max_spent,
            minReceived=min_received,
        )

    async def estimate_output(self, order: CrossChainOrder) -> OutputEstimate:
        async with self._build_client() as client:
            results = await asyncio.gather(
                self._quote_eco(client, order),
                self._quote_across(client, order),
                self._quote_debridge(client, order),
                return_exceptions=True,
            )

        estimates: list[OutputEstimate] = []
        for result in results:
            if isinstance(result, OutputEstimate):
                estimates.append(result)
                continue
            LOGGER.warning("solver_quote_failed", error=str(result))

        if not estimates:
            fallback = self._fallback_estimate(order)
            return OutputEstimate(
                best_solver="fallback",
                net_out_usd=fallback,
                fee_bps=50,
                fill_time_est_ms=120_000,
            )

        ranked = sorted(
            estimates,
            key=lambda item: (item.net_out_usd, -item.fee_bps, -item.fill_time_est_ms),
            reverse=True,
        )
        return ranked[0]

    def _build_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=5.0, transport=self._transport)

    def _get_chain_config(self, chain_name: str) -> ChainConfig:
        try:
            return self.chain_configs[str(chain_name)]
        except KeyError as exc:
            raise ValueError(f"missing chain config: {chain_name}") from exc

    def _next_nonce(self) -> int:
        self._nonce_seed += 1
        return self._nonce_seed

    def _usd_to_token_units(self, size_usd: float, decimals: int) -> int:
        return max(1, int(round(max(0.0, float(size_usd)) * (10 ** max(0, decimals)))))

    def _message_bytes(self, message_hex: str) -> bytes:
        message = str(message_hex or "0x").strip().lower()
        if message.startswith("0x"):
            message = message[2:]
        if not message:
            return b""
        return bytes.fromhex(message)

    def _fallback_estimate(self, order: CrossChainOrder) -> float:
        if not order.minReceived:
            return 0.0
        leg = order.minReceived[0]
        decimals = self._infer_decimals(leg.amount)
        return leg.amount / float(10 ** decimals)

    def _infer_decimals(self, amount: int) -> int:
        return 6 if amount >= 10**6 else 0

    async def _quote_eco(self, client: httpx.AsyncClient, order: CrossChainOrder) -> OutputEstimate:
        response = await self._request_json(
            client,
            "GET",
            "https://api.eco.com/v1/quote",
            params=self._solver_params(order),
        )
        payload = response.json()
        net_out = self._first_number(
            payload,
            ("netOutUsd",),
            ("outputUsd",),
            ("routes", 0, "outputUsd"),
            ("routes", 0, "amountOutUsd"),
        )
        return OutputEstimate(
            best_solver="eco",
            net_out_usd=net_out if net_out > 0 else self._fallback_estimate(order),
            fee_bps=self._first_int(payload, ("feeBps",), ("routes", 0, "feeBps"), default=18),
            fill_time_est_ms=self._first_int(payload, ("fillTimeMs",), ("routes", 0, "fillTimeMs"), default=45_000),
        )

    async def _quote_across(self, client: httpx.AsyncClient, order: CrossChainOrder) -> OutputEstimate:
        response = await self._request_json(
            client,
            "GET",
            "https://app.across.to/api/suggested-fees",
            params=self._solver_params(order),
        )
        payload = response.json()
        fee_bps = self._first_int(
            payload,
            ("feeBps",),
            ("totalRelayFee", "pct"),
            ("fees", "totalRelayFee", "pct"),
            default=14,
        )
        net_out = self._first_number(
            payload,
            ("netOutUsd",),
            ("outputAmountUsd",),
            ("estimatedFillUsd",),
        )
        return OutputEstimate(
            best_solver="across",
            net_out_usd=net_out if net_out > 0 else max(self._fallback_estimate(order) * (1.0 - fee_bps / 10_000.0), 0.0),
            fee_bps=fee_bps,
            fill_time_est_ms=self._first_int(payload, ("fillTimeMs",), ("estimatedFillTimeSec",), default=30_000),
        )

    async def _quote_debridge(self, client: httpx.AsyncClient, order: CrossChainOrder) -> OutputEstimate:
        response = await self._request_json(
            client,
            "POST",
            "https://api.dln.trade/v1.0/dln/order/quote",
            json=self._solver_payload(order),
        )
        payload = response.json()
        net_out = self._first_number(
            payload,
            ("estimation", "dstChainTokenOut", "amountUsd"),
            ("estimation", "amountOutUsd"),
            ("netOutUsd",),
            ("quote", "netOutUsd"),
        )
        fee_bps = self._first_int(payload, ("estimation", "feeBps"), ("feeBps",), default=22)
        return OutputEstimate(
            best_solver="debridge",
            net_out_usd=net_out if net_out > 0 else max(self._fallback_estimate(order) * (1.0 - fee_bps / 10_000.0), 0.0),
            fee_bps=fee_bps,
            fill_time_est_ms=self._first_int(payload, ("fillTimeMs",), ("estimation", "fillTimeMs"), default=55_000),
        )

    async def _request_json(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        *,
        params: dict[str, str] | None = None,
        json: dict[str, object] | None = None,
    ) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = await client.request(method, url, params=params, json=json)
                response.raise_for_status()
                return response
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt == 2:
                    break
                await asyncio.sleep(0.1 * (attempt + 1))
        raise RuntimeError(f"quote request failed for {url}: {last_error}")

    def _solver_params(self, order: CrossChainOrder) -> dict[str, str]:
        max_leg = order.maxSpent[0]
        min_leg = order.minReceived[0]
        return {
            "originChainId": str(order.originChainId),
            "destinationChainId": str(min_leg.chain_id),
            "inputToken": max_leg.token,
            "outputToken": min_leg.token,
            "amount": str(max_leg.amount),
            "recipient": min_leg.recipient,
            "nonce": str(order.nonce),
        }

    def _solver_payload(self, order: CrossChainOrder) -> dict[str, object]:
        params = self._solver_params(order)
        return {
            "srcChainId": int(params["originChainId"]),
            "dstChainId": int(params["destinationChainId"]),
            "srcChainTokenIn": params["inputToken"],
            "dstChainTokenOut": params["outputToken"],
            "srcChainTokenInAmount": params["amount"],
            "dstChainTokenOutRecipient": params["recipient"],
            "affiliateFeePercent": 0,
        }

    def _first_number(self, payload: object, *paths: tuple[object, ...]) -> float:
        for path in paths:
            candidate = self._extract_path(payload, path)
            if isinstance(candidate, (int, float)):
                return float(candidate)
            if isinstance(candidate, str):
                try:
                    return float(candidate)
                except ValueError:
                    continue
        return 0.0

    def _first_int(self, payload: object, *paths: tuple[object, ...], default: int) -> int:
        number = self._first_number(payload, *paths)
        if number <= 0:
            return default
        if number < 1.0:
            return max(1, int(round(number * 10_000)))
        if number > 1_000 and "Sec" in "".join(str(part) for path in paths for part in path):
            return int(round(number * 1_000))
        return int(round(number))

    def _extract_path(self, payload: object, path: tuple[object, ...]) -> object | None:
        current: object = payload
        for part in path:
            if isinstance(part, int):
                if not isinstance(current, list) or part >= len(current):
                    return None
                current = current[part]
                continue
            if not isinstance(current, dict):
                return None
            current = current.get(str(part))
        return current


async def _smoke_test() -> int:
    chain_configs = {
        "ethereum": ChainConfig(
            name="ethereum",
            chain_id=1,
            origin_settler="0x1111111111111111111111111111111111111111",
            user_address="0x2222222222222222222222222222222222222222",
            input_token="0xA0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
            output_token="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
            recipient="0x3333333333333333333333333333333333333333",
        ),
        "base": ChainConfig(
            name="base",
            chain_id=8453,
            origin_settler="0x4444444444444444444444444444444444444444",
            user_address="0x2222222222222222222222222222222222222222",
            input_token="0x4200000000000000000000000000000000000006",
            output_token="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
            recipient="0x3333333333333333333333333333333333333333",
        ),
    }

    def mock_handler(request: httpx.Request) -> httpx.Response:
        if "eco.com" in str(request.url):
            return httpx.Response(200, json={"routes": [{"outputUsd": 1_015.0, "feeBps": 18, "fillTimeMs": 42_000}]})
        if "across.to" in str(request.url):
            return httpx.Response(200, json={"outputAmountUsd": 1_012.0, "feeBps": 12, "estimatedFillTimeSec": 28})
        return httpx.Response(200, json={"estimation": {"amountOutUsd": 1_021.0, "feeBps": 25, "fillTimeMs": 38_000}})

    composer = IntentComposer(chain_configs=chain_configs)
    composer._transport = httpx.MockTransport(mock_handler)
    signal = TradeSignal(
        asset="ETH",
        side="buy",
        size_usd=1_000.0,
        src_chain="ethereum",
        dst_chain="base",
        max_slippage_bps=50,
        deadline_secs=600,
    )
    order = composer.compose(signal)
    estimate = await composer.estimate_output(order)

    print("metric\tvalue")
    print(f"nonce\t{order.nonce}")
    print(f"order_data_len\t{len(order.orderData)}")
    print(f"best_solver\t{estimate.best_solver}")
    print(f"net_out_usd\t{estimate.net_out_usd}")
    return 0 if estimate.best_solver == "debridge" and len(order.orderData) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_smoke_test()))
