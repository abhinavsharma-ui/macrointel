from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | dict[str, "JSONValue"] | list["JSONValue"]

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"


@dataclass(slots=True, frozen=True)
class BlockEvent:
    block: int
    address: str
    delta_usd: float
    direction: Literal["in", "out"]
    ts: float


@dataclass(slots=True, frozen=True)
class AddressSnapshot:
    address: str
    net_flow_24h: float
    tx_count_24h: int
    whale_flag: bool
    ts: float


@dataclass(slots=True, frozen=True)
class WebhookEvent:
    source: str
    event_type: str
    payload: dict[str, JSONValue]
    ts: float


@dataclass(slots=True, frozen=True)
class OrderLeg:
    token: str
    amount: int
    recipient: str
    chain_id: int


@dataclass(slots=True, frozen=True)
class ChainConfig:
    name: str
    chain_id: int
    origin_settler: str
    user_address: str
    input_token: str
    output_token: str
    recipient: str
    token_decimals: int = 6
    exclusive_relayer: str = ZERO_ADDRESS
    message_hex: str = "0x"
    solver_contracts: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class TradeSignal:
    asset: str
    side: Literal["buy", "sell"]
    size_usd: float
    src_chain: str
    dst_chain: str
    max_slippage_bps: int
    deadline_secs: int


@dataclass(slots=True, frozen=True)
class CrossChainOrder:
    orderDataType: str
    orderData: bytes
    fillDeadline: int
    exclusivityDeadline: int
    exclusiveRelayer: str
    nonce: int
    originChainId: int
    initiateDeadline: int
    maxSpent: list[OrderLeg]
    minReceived: list[OrderLeg]


@dataclass(slots=True, frozen=True)
class OutputEstimate:
    best_solver: str
    net_out_usd: float
    fee_bps: int
    fill_time_est_ms: int


@dataclass(slots=True, frozen=True)
class FillEvent:
    status: Literal["pending", "filled", "expired", "slippage_exceeded"]
    filled_usd: float
    ts: float
    fill_id: str = ""
    solver: str = ""
    asset: str = ""
    duration_ms: int | None = None


@dataclass(slots=True, frozen=True)
class AuthScope:
    allowed_contracts: list[str]
    max_value_wei: int
    expires_at: int


@dataclass(slots=True, frozen=True)
class TokenomicsSignal:
    protocol: str
    ratio: float
    trend: Literal["up", "flat", "down"]
    size_scalar: float
