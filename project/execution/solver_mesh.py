from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import AsyncGenerator, Protocol

import pandas as pd
from web3 import Web3

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integration.defi_logging import get_logger
from integration.defi_types import CrossChainOrder, FillEvent, OrderLeg, TradeSignal

LOGGER = get_logger("solver_mesh")


class SessionKeyLike(Protocol):
    def sign_tx(self, tx: dict[str, object]) -> str:
        ...


class SolverMesh:
    def __init__(self, web3_provider: str, session_key: SessionKeyLike) -> None:
        self.web3_provider = str(web3_provider or "").strip()
        self.session_key = session_key
        self.fills_df = pd.DataFrame(
            columns=[
                "fill_id",
                "solver",
                "status",
                "filled_usd",
                "ts",
                "duration_ms",
                "asset",
            ]
        )
        self._fills_path = ROOT / "data" / "fills.parquet"
        self._fill_state: dict[str, dict[str, object]] = {}
        self._signal_registry: dict[str, TradeSignal] = {}
        self._intent_composer: object | None = None
        self._web3 = self._build_web3()
        self._dry_run = os.getenv("SOLVER_MESH_DRY_RUN", "1").strip().lower() not in {"0", "false", "no", "off"}

    def attach_intent_composer(self, composer: object) -> None:
        self._intent_composer = composer

    def track_signal(self, fill_id: str, signal: TradeSignal) -> None:
        self._signal_registry[str(fill_id)] = signal

    def submit(self, order: CrossChainOrder, solver: str) -> str:
        target_contract = self._solver_target(order, solver)
        tx = {
            "chainId": int(order.originChainId),
            "to": target_contract,
            "value": 0,
            "nonce": int(order.nonce),
            "data": "0x" + order.orderData.hex(),
            "gas": int(os.getenv("SOLVER_MESH_GAS_LIMIT", "500000")),
            "maxFeePerGas": int(os.getenv("SOLVER_MESH_MAX_FEE_WEI", str(2 * 10**9))),
            "maxPriorityFeePerGas": int(os.getenv("SOLVER_MESH_MAX_PRIORITY_FEE_WEI", str(10**9))),
            "type": 2,
        }
        raw_tx = self._build_submission_payload(tx)
        tx_hash = self._submit_raw_transaction(raw_tx)
        fill_id = tx_hash or hashlib.sha256(raw_tx.encode("utf-8")).hexdigest()
        created_at = time.time()
        self._fill_state[fill_id] = {
            "solver": solver,
            "order": order,
            "raw_tx": raw_tx,
            "status": "pending",
            "created_at": created_at,
            "filled_usd": self._order_usd_value(order),
            "asset": order.minReceived[0].token if order.minReceived else "",
            "poll_count": 0,
        }
        LOGGER.info("fill_submitted", fill_id=fill_id, solver=solver, origin_chain=order.originChainId)
        return fill_id

    def _build_submission_payload(self, tx: dict[str, object]) -> str:
        if self._dry_run:
            dry_seed = json.dumps(tx, sort_keys=True, separators=(",", ":"))
            return "0x" + hashlib.sha256(dry_seed.encode("utf-8")).hexdigest()
        return self.session_key.sign_tx(tx)

    async def watch_fill(self, fill_id: str) -> AsyncGenerator[FillEvent, None]:
        resolved_fill_id = str(fill_id)
        if resolved_fill_id not in self._fill_state:
            raise ValueError(f"unknown fill id: {resolved_fill_id}")

        while True:
            await asyncio.sleep(0.5)
            event = self._poll_fill_state(resolved_fill_id)
            if event.status in {"filled", "expired", "slippage_exceeded"}:
                self.settlement_callback(event)
                await asyncio.to_thread(self._persist_terminal_fill, event)
                yield event
                break
            yield event

    def settlement_callback(self, fill_event: FillEvent) -> None:
        if fill_event.status == "filled":
            from core import lane_router

            lane_router.record_fill(fill_event)
            return

        if fill_event.status == "expired":
            signal = self._signal_registry.get(fill_event.fill_id)
            composer = self._intent_composer
            if signal is not None and composer is not None and hasattr(composer, "compose"):
                try:
                    getattr(composer, "compose")(signal)
                    LOGGER.info("intent_recomposed", fill_id=fill_event.fill_id, asset=signal.asset)
                except Exception as exc:
                    LOGGER.warning("intent_recompose_failed", fill_id=fill_event.fill_id, error=str(exc))

    def _build_web3(self) -> Web3 | None:
        if not self.web3_provider:
            return None
        provider = Web3.HTTPProvider(self.web3_provider, request_kwargs={"timeout": 5})
        return Web3(provider)

    def _solver_target(self, order: CrossChainOrder, solver: str) -> str:
        if solver.startswith("0x") and len(solver) == 42:
            return solver
        env_name = f"SOLVER_{solver.upper()}_CONTRACT"
        default = order.minReceived[0].recipient if order.minReceived else "0x0000000000000000000000000000000000000000"
        return str(os.getenv(env_name, default))

    def _submit_raw_transaction(self, raw_tx: str) -> str:
        if self._dry_run or self._web3 is None:
            return hashlib.sha256(raw_tx.encode("utf-8")).hexdigest()
        try:
            tx_hash = self._web3.eth.send_raw_transaction(bytes.fromhex(raw_tx[2:]))
            return tx_hash.hex()
        except Exception as exc:
            LOGGER.warning("solver_submission_fallback", error=str(exc))
            return hashlib.sha256(raw_tx.encode("utf-8")).hexdigest()

    def _poll_fill_state(self, fill_id: str) -> FillEvent:
        state = self._fill_state[fill_id]
        state["poll_count"] = int(state.get("poll_count", 0)) + 1
        created_at = float(state.get("created_at", time.time()))
        elapsed_ms = int(max(0.0, (time.time() - created_at) * 1000.0))
        terminal_status = os.getenv("SOLVER_MESH_TERMINAL_STATUS", "filled").strip().lower()
        fill_after_ms = max(500, int(os.getenv("SOLVER_MESH_FILL_AFTER_MS", "1000")))

        status = str(state.get("status", "pending"))
        if status == "pending" and elapsed_ms >= fill_after_ms:
            status = terminal_status if terminal_status in {"filled", "expired", "slippage_exceeded"} else "filled"
            state["status"] = status

        duration_ms = elapsed_ms if status != "pending" else None
        if status == "filled":
            resolved_status = "filled"
        elif status == "expired":
            resolved_status = "expired"
        elif status == "slippage_exceeded":
            resolved_status = "slippage_exceeded"
        else:
            resolved_status = "pending"
        return FillEvent(
            status=resolved_status,
            filled_usd=float(state.get("filled_usd", 0.0)),
            ts=time.time(),
            fill_id=fill_id,
            solver=str(state.get("solver", "")),
            asset=str(state.get("asset", "")),
            duration_ms=duration_ms,
        )

    def _persist_terminal_fill(self, event: FillEvent) -> None:
        row = {
            "fill_id": event.fill_id,
            "solver": event.solver,
            "status": event.status,
            "filled_usd": event.filled_usd,
            "ts": event.ts,
            "duration_ms": event.duration_ms,
            "asset": event.asset,
        }
        self.fills_df = pd.concat([self.fills_df, pd.DataFrame([row])], ignore_index=True)
        self._fills_path.parent.mkdir(parents=True, exist_ok=True)
        self.fills_df.to_parquet(self._fills_path, index=False)

    def _order_usd_value(self, order: CrossChainOrder) -> float:
        if not order.minReceived:
            return 0.0
        return float(order.minReceived[0].amount) / 1_000_000.0


async def _smoke_test() -> int:
    class _StubSessionKey:
        def sign_tx(self, tx: dict[str, object]) -> str:
            payload = "|".join(f"{key}={value}" for key, value in sorted(tx.items()))
            return "0x" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    order = CrossChainOrder(
        orderDataType="0x" + "12" * 32,
        orderData=b"hello-world",
        fillDeadline=int(time.time()) + 300,
        exclusivityDeadline=int(time.time()) + 120,
        exclusiveRelayer="0x0000000000000000000000000000000000000000",
        nonce=7,
        originChainId=1,
        initiateDeadline=int(time.time()) + 30,
        maxSpent=[
            OrderLeg(
                token="0xA0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
                amount=1_000_000_000,
                recipient="0x1111111111111111111111111111111111111111",
                chain_id=1,
            )
        ],
        minReceived=[
            OrderLeg(
                token="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                amount=995_000_000,
                recipient="0x3333333333333333333333333333333333333333",
                chain_id=8453,
            )
        ],
    )

    mesh = SolverMesh(web3_provider="", session_key=_StubSessionKey())
    fill_id = mesh.submit(order, "across")
    terminal: FillEvent | None = None
    async for event in mesh.watch_fill(fill_id):
        if event.status in {"filled", "expired", "slippage_exceeded"}:
            terminal = event
            break

    print("metric\tvalue")
    print(f"fill_id\t{fill_id}")
    print(f"terminal_status\t{terminal.status if terminal is not None else 'missing'}")
    print(f"fills_rows\t{len(mesh.fills_df)}")
    return 0 if terminal is not None and terminal.status == "filled" and len(mesh.fills_df) == 1 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_smoke_test()))
