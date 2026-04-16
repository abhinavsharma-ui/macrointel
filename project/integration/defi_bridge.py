from __future__ import annotations

import argparse
import asyncio
import os
import sys
import threading
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auth.session_key import SessionKey
from core import lane_router
from data.block_intel import BlockIntelFeed
from data.webhook_bus import WebhookBus
from execution.intent_composer import IntentComposer
from execution.solver_mesh import SolverMesh
from integration.defi_logging import get_logger
from integration.defi_types import AuthScope, ChainConfig, FillEvent, OutputEstimate, TokenomicsSignal, TradeSignal, WebhookEvent
from tokenomics.buyback_tracker import BuybackTracker

LOGGER = get_logger("defi_bridge")

_RUNTIME_LOCK = threading.RLock()
_RUNTIME_BRIDGE: DeFiBridge | None = None
_RUNTIME_THREAD: threading.Thread | None = None


class DeFiBridge:
    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = bool(dry_run)
        if self.dry_run:
            os.environ.setdefault(
                "DELEGATE_PRIVATE_KEY",
                "0x59c6995e998f97a5a0044976f7d5f9bc6ab4fe71f5b3c3d4fe26c7c0e59101cf",
            )
            os.environ.setdefault("SOLVER_MESH_DRY_RUN", "1")

        self.chain_configs = self._build_chain_configs()
        self.block_intel_feed = BlockIntelFeed(
            feed_url=str(os.getenv("BLOCK_INTEL_FEED_URL", "ws://127.0.0.1:0")).strip(),
            api_key=str(os.getenv("BLOCK_INTEL_API_KEY", "stub")).strip(),
        )
        self.webhook_bus = WebhookBus(port=int(os.getenv("DEFI_WEBHOOK_PORT", "9000")))
        self.intent_composer = IntentComposer(self.chain_configs)
        self.session_key = SessionKey(
            delegate_address=str(os.getenv("EIP7702_DELEGATE_ADDRESS", "0x2222222222222222222222222222222222222222")),
            authority_scope=self._build_auth_scope(),
        )
        self.solver_mesh = SolverMesh(
            web3_provider=str(os.getenv("WEB3_PROVIDER_URI", "")).strip(),
            session_key=self.session_key,
        )
        self.solver_mesh.attach_intent_composer(self.intent_composer)
        protocol_list = [
            item.strip()
            for item in os.getenv("DEFI_TOKENOMICS_PROTOCOLS", "lista_dao,dfinity_mission70").split(",")
            if item.strip()
        ]
        self.buyback_tracker = BuybackTracker(protocol_list)
        self._terminal_events: deque[FillEvent] = deque(maxlen=10_000)
        self._terminal_gate = asyncio.Event()
        self._listener_registered = False
        self._started = False
        self._register_handlers()
        if self.dry_run:
            self._install_dry_run_transports()

        try:
            self.session_key.generate()
        except Exception as exc:
            LOGGER.warning("session_key_generate_failed", error=str(exc))

    async def start(self) -> None:
        self._started = True
        await self.buyback_tracker._poll_once()
        self._register_lane_router_listener()
        await self._replay_buffered_signals()
        await asyncio.gather(
            self.buyback_tracker.poll(interval_secs=int(os.getenv("DEFI_BUYBACK_INTERVAL_SECS", "30"))),
            self.webhook_bus.serve(host=str(os.getenv("DEFI_WEBHOOK_HOST", "0.0.0.0"))),
            self._block_intel_loop(),
        )

    async def run_dry_run(self) -> int:
        self._started = True
        await self.buyback_tracker._poll_once()
        synthetic_signal = {
            "symbol": "ICPUSDT",
            "asset": "ICPUSDT",
            "signal": "buy",
            "lane": "crypto",
            "trade_eligible": True,
            "size_multiplier": 1.1,
            "target_notional_usd": 750.0,
            "ts": time.time(),
        }
        await self._execute_signal_flow(lane_router.normalize_signal_payload(synthetic_signal))
        health = self.health_snapshot()
        print("metric\tvalue")
        print(f"fill_rate_1h\t{health['fill_rate_1h']}")
        print(f"avg_fill_ms\t{health['avg_fill_ms']}")
        print(f"session_key_expires_in_s\t{health['session_key_expires_in_s']}")
        print(f"active_solvers\t{','.join(health['active_solvers'])}")
        return 0

    def health_snapshot(self) -> dict[str, object]:
        one_hour_ago = time.time() - 3_600.0
        recent = [event for event in self._terminal_events if event.ts >= one_hour_ago]
        filled = [event for event in recent if event.status == "filled"]
        fill_rate = round(len(filled) / len(recent), 4) if recent else 0.0
        durations = [event.duration_ms for event in filled if event.duration_ms is not None]
        avg_fill_ms = int(round(sum(durations) / len(durations))) if durations else 0
        expires_in = max(0, int(self.session_key.authority_scope.expires_at - time.time()))
        return {
            "fill_rate_1h": fill_rate,
            "avg_fill_ms": avg_fill_ms,
            "session_key_expires_in_s": expires_in,
            "active_solvers": ["eco", "across", "debridge"],
        }

    def _register_handlers(self) -> None:
        @self.webhook_bus.on("lane_router.signal")
        async def _handle_signal(event: WebhookEvent) -> None:
            await self._execute_signal_flow(event.payload)

    def _register_lane_router_listener(self) -> None:
        if self._listener_registered:
            return
        lane_router.register_signal_listener(self._forward_lane_signal)
        self._listener_registered = True

    def _forward_lane_signal(self, payload: dict[str, object]) -> None:
        event_ts = payload.get("ts", time.time())
        ts_value = float(event_ts) if isinstance(event_ts, (int, float, str)) else time.time()
        event = WebhookEvent(
            source="lane_router",
            event_type="lane_router.signal",
            payload=lane_router.normalize_signal_payload(payload),
            ts=ts_value,
        )
        self.webhook_bus.publish_from_thread(event)

    async def _replay_buffered_signals(self) -> None:
        replay_window = time.time() - max(1, int(os.getenv("DEFI_SIGNAL_REPLAY_SECS", "300")))
        for payload in lane_router.replay_signals(since_ts=replay_window):
            await self.webhook_bus.publish_local(
                WebhookEvent(
                    source="lane_router",
                    event_type="lane_router.signal",
                    payload=payload,
                    ts=float(payload.get("ts", time.time()) or time.time()),
                )
            )

    async def _block_intel_loop(self) -> None:
        addresses = [
            item.strip()
            for item in os.getenv("DEFI_WATCH_ADDRESSES", "").split(",")
            if item.strip()
        ]
        if not addresses or ":0" in self.block_intel_feed.feed_url:
            while True:
                await asyncio.sleep(60)
        async for event in self.block_intel_feed.subscribe(addresses):
            LOGGER.info("block_intel_event", block=event.block, address=event.address, delta_usd=event.delta_usd)

    def _tokenomics_signal_for(self, asset: str) -> TokenomicsSignal:
        asset_upper = str(asset or "").upper()
        preferred = "dfinity_mission70" if "ICP" in asset_upper and "dfinity_mission70" in self.buyback_tracker.protocols else self.buyback_tracker.protocols[0]
        return self.buyback_tracker.signal(preferred)

    def _to_trade_signal(self, payload: dict[str, object]) -> TradeSignal | None:
        lane = str(payload.get("lane") or "").strip().lower()
        direction = str(payload.get("signal") or payload.get("side") or "").strip().lower()
        if lane != "crypto" or direction not in {"buy", "sell"}:
            return None
        if not bool(payload.get("trade_eligible", True)):
            return None

        asset = str(payload.get("asset") or payload.get("symbol") or "").strip().upper()
        if not asset:
            return None
        side = "buy" if direction == "buy" else "sell"
        size_multiplier = self._safe_float(payload.get("size_multiplier"), default=1.0)
        base_size_usd = self._safe_float(payload.get("target_notional_usd"), default=float(os.getenv("DEFI_DEFAULT_SIZE_USD", "500")))
        return TradeSignal(
            asset=asset,
            side=side,
            size_usd=max(25.0, base_size_usd * max(0.1, size_multiplier)),
            src_chain=str(payload.get("src_chain") or os.getenv("DEFI_DEFAULT_SRC_CHAIN", "ethereum")),
            dst_chain=str(payload.get("dst_chain") or os.getenv("DEFI_DEFAULT_DST_CHAIN", "base")),
            max_slippage_bps=int(self._safe_float(payload.get("max_slippage_bps"), default=float(os.getenv("DEFI_MAX_SLIPPAGE_BPS", "50")))),
            deadline_secs=int(self._safe_float(payload.get("deadline_secs"), default=float(os.getenv("DEFI_DEADLINE_SECS", "600")))),
        )

    def _build_chain_configs(self) -> dict[str, ChainConfig]:
        user_address = str(os.getenv("DEFI_USER_ADDRESS", "0x3333333333333333333333333333333333333333"))
        recipient = str(os.getenv("DEFI_RECIPIENT_ADDRESS", user_address))
        return {
            "ethereum": ChainConfig(
                name="ethereum",
                chain_id=int(os.getenv("ETHEREUM_CHAIN_ID", "1")),
                origin_settler=str(os.getenv("ETHEREUM_ORIGIN_SETTLER", "0x1111111111111111111111111111111111111111")),
                user_address=user_address,
                input_token=str(os.getenv("ETHEREUM_INPUT_TOKEN", "0xA0b86991c6218b36c1d19d4a2e9eb0ce3606eb48")),
                output_token=str(os.getenv("ETHEREUM_OUTPUT_TOKEN", "0xA0b86991c6218b36c1d19d4a2e9eb0ce3606eb48")),
                recipient=recipient,
                exclusive_relayer=str(os.getenv("ETHEREUM_EXCLUSIVE_RELAYER", "0x0000000000000000000000000000000000000000")),
            ),
            "base": ChainConfig(
                name="base",
                chain_id=int(os.getenv("BASE_CHAIN_ID", "8453")),
                origin_settler=str(os.getenv("BASE_ORIGIN_SETTLER", "0x4444444444444444444444444444444444444444")),
                user_address=user_address,
                input_token=str(os.getenv("BASE_INPUT_TOKEN", "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913")),
                output_token=str(os.getenv("BASE_OUTPUT_TOKEN", "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913")),
                recipient=recipient,
                exclusive_relayer=str(os.getenv("BASE_EXCLUSIVE_RELAYER", "0x0000000000000000000000000000000000000000")),
            ),
        }

    def _build_auth_scope(self) -> AuthScope:
        allowed_contracts = {
            config.recipient
            for config in self._build_chain_configs().values()
        }
        for env_name in ("SOLVER_ECO_CONTRACT", "SOLVER_ACROSS_CONTRACT", "SOLVER_DEBRIDGE_CONTRACT"):
            value = str(os.getenv(env_name, "")).strip()
            if value:
                allowed_contracts.add(value)
        return AuthScope(
            allowed_contracts=sorted(allowed_contracts),
            max_value_wei=int(os.getenv("DEFI_MAX_VALUE_WEI", str(10**18))),
            expires_at=int(time.time()) + int(os.getenv("DEFI_SESSION_EXPIRES_SECS", "3600")),
        )

    def _safe_float(self, value: object, default: float) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return default
        return default

    async def _execute_signal_flow(self, payload: dict[str, object]) -> FillEvent | None:
        trade_signal = self._to_trade_signal(payload)
        if trade_signal is None:
            LOGGER.info(
                "signal_skipped",
                lane=str(payload.get("lane") or ""),
                signal=str(payload.get("signal") or payload.get("side") or ""),
                symbol=str(payload.get("symbol") or payload.get("asset") or ""),
                trade_eligible=bool(payload.get("trade_eligible", True)),
            )
            return None
        tokenomics_signal = self._tokenomics_signal_for(trade_signal.asset)
        adjusted_signal = replace(
            trade_signal,
            size_usd=round(trade_signal.size_usd * tokenomics_signal.size_scalar, 6),
        )
        LOGGER.info(
            "signal_accepted",
            asset=adjusted_signal.asset,
            side=adjusted_signal.side,
            size_usd=adjusted_signal.size_usd,
            src_chain=adjusted_signal.src_chain,
            dst_chain=adjusted_signal.dst_chain,
            tokenomics_ratio=round(tokenomics_signal.ratio, 6),
            tokenomics_scalar=tokenomics_signal.size_scalar,
        )
        order = self.intent_composer.compose(adjusted_signal)
        estimate = await self.intent_composer.estimate_output(order)
        LOGGER.info(
            "solver_selected",
            asset=adjusted_signal.asset,
            solver=estimate.best_solver,
            net_out_usd=estimate.net_out_usd,
            fee_bps=estimate.fee_bps,
            fill_time_est_ms=estimate.fill_time_est_ms,
        )
        fill_id = await asyncio.to_thread(self.solver_mesh.submit, order, estimate.best_solver)
        self.solver_mesh.track_signal(fill_id, adjusted_signal)
        async for fill_event in self.solver_mesh.watch_fill(fill_id):
            if fill_event.status in {"filled", "expired", "slippage_exceeded"}:
                self._terminal_events.append(fill_event)
                self._terminal_gate.set()
                LOGGER.info(
                    "fill_terminal",
                    fill_id=fill_event.fill_id,
                    status=fill_event.status,
                    filled_usd=fill_event.filled_usd,
                    solver=fill_event.solver,
                )
                return fill_event
        return None

    def _install_dry_run_transports(self) -> None:
        os.environ.setdefault("LISTA_DAO_BUYBACK_USD_7D", "1500000")
        os.environ.setdefault("LISTA_DAO_REVENUE_USD_7D", "1200000")

        def _tokenomics_handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"metrics": {"buyback_usd_7d": 500_000.0, "revenue_usd_7d": 700_000.0}},
            )

        def _quote_handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "eco.com" in url:
                return httpx.Response(200, json={"routes": [{"outputUsd": 765.0, "feeBps": 18, "fillTimeMs": 42_000}]})
            if "across.to" in url:
                return httpx.Response(200, json={"outputAmountUsd": 762.0, "feeBps": 12, "estimatedFillTimeSec": 28})
            return httpx.Response(200, json={"estimation": {"amountOutUsd": 771.0, "feeBps": 25, "fillTimeMs": 38_000}})

        self.buyback_tracker._transport = httpx.MockTransport(_tokenomics_handler)
        self.intent_composer._transport = httpx.MockTransport(_quote_handler)


def maybe_bootstrap_defi_bridge() -> None:
    enabled = os.getenv("DEFI_BRIDGE_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return
    global _RUNTIME_BRIDGE, _RUNTIME_THREAD
    with _RUNTIME_LOCK:
        if _RUNTIME_BRIDGE is not None and _RUNTIME_THREAD is not None and _RUNTIME_THREAD.is_alive():
            return
        _RUNTIME_BRIDGE = DeFiBridge(dry_run=False)

        def _runner() -> None:
            asyncio.run(_RUNTIME_BRIDGE.start())

        _RUNTIME_THREAD = threading.Thread(target=_runner, daemon=True, name="defi-bridge")
        _RUNTIME_THREAD.start()


def get_registered_health_payload() -> dict[str, object]:
    with _RUNTIME_LOCK:
        bridge = _RUNTIME_BRIDGE
    if bridge is None:
        return {
            "fill_rate_1h": 0.0,
            "avg_fill_ms": 0,
            "session_key_expires_in_s": 0,
            "active_solvers": ["eco", "across", "debridge"],
        }
    return bridge.health_snapshot()


async def _main_async(dry_run: bool) -> int:
    bridge = DeFiBridge(dry_run=dry_run)
    if dry_run:
        return await bridge.run_dry_run()
    await bridge.start()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Macro Intelligence DeFi bridge")
    parser.add_argument("--dry-run", action="store_true", help="run a local stub flow and exit")
    args = parser.parse_args()
    return asyncio.run(_main_async(dry_run=args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
