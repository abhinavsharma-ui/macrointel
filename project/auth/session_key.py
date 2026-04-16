from __future__ import annotations

import ctypes
import os
import sys
import time
from pathlib import Path

import rlp
from eth_account import Account
from eth_keys import keys
from eth_utils import keccak, to_checksum_address

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integration.defi_logging import get_logger
from integration.defi_types import AuthScope

LOGGER = get_logger("session_key")
EIP7702_MAGIC = b"\x05"


class SessionExpiredError(RuntimeError):
    """Raised when a session key is used after its expiry."""


class ScopeViolationError(RuntimeError):
    """Raised when a transaction exceeds the session scope."""


class SessionKey:
    def __init__(self, delegate_address: str, authority_scope: AuthScope) -> None:
        self.delegate_address = to_checksum_address(delegate_address)
        self.authority_scope = authority_scope
        self.session_address: str | None = None
        self.signed_auth_tuple: bytes | None = None
        self._session_key_buffer: ctypes.Array[ctypes.c_char] | None = None

    def generate(self) -> tuple[str, bytes]:
        delegate_private_key = self._load_delegate_private_key()
        ephemeral_key = keys.PrivateKey(os.urandom(32))
        self._session_key_buffer = ctypes.create_string_buffer(ephemeral_key.to_bytes(), 32)
        self.session_address = to_checksum_address(ephemeral_key.public_key.to_address())

        chain_id = int(os.getenv("EIP7702_CHAIN_ID", "0"))
        nonce = int(os.getenv("EIP7702_NONCE", "0"))
        message = EIP7702_MAGIC + rlp.encode(
            [
                chain_id,
                bytes.fromhex(self.delegate_address[2:]),
                nonce,
            ]
        )
        message_hash = keccak(message)
        signature = delegate_private_key.sign_msg_hash(message_hash)
        self.signed_auth_tuple = rlp.encode(
            [
                chain_id,
                bytes.fromhex(self.delegate_address[2:]),
                nonce,
                int(signature.v),
                int(signature.r),
                int(signature.s),
            ]
        )
        return self.session_address, self.signed_auth_tuple

    def sign_tx(self, tx: dict[str, object]) -> str:
        self._validate_scope(tx)
        private_key_bytes = self._active_private_key_bytes()
        signed = Account.from_key(private_key_bytes).sign_transaction(tx)
        return "0x" + signed.raw_transaction.hex()

    def revoke(self) -> None:
        if self._session_key_buffer is None:
            return
        ctypes.memset(ctypes.addressof(self._session_key_buffer), 0, len(self._session_key_buffer.raw))
        self.signed_auth_tuple = None
        LOGGER.info("session_key_revoked", revoked_at=int(time.time()), session_address=self.session_address or "")

    def __repr__(self) -> str:
        return (
            "SessionKey("
            f"delegate_address='{self.delegate_address}', "
            f"session_address='{self.session_address or ''}', "
            f"expires_at={self.authority_scope.expires_at})"
        )

    __str__ = __repr__

    def _load_delegate_private_key(self) -> keys.PrivateKey:
        raw_key = str(os.getenv("DELEGATE_PRIVATE_KEY", "")).strip()
        if not raw_key:
            raise RuntimeError("DELEGATE_PRIVATE_KEY is required for session key generation")
        key_hex = raw_key[2:] if raw_key.startswith("0x") else raw_key
        if len(key_hex) != 64:
            raise RuntimeError("DELEGATE_PRIVATE_KEY must be a 32-byte hex string")
        return keys.PrivateKey(bytes.fromhex(key_hex))

    def _validate_scope(self, tx: dict[str, object]) -> None:
        if time.time() >= float(self.authority_scope.expires_at):
            raise SessionExpiredError("session key expired")
        to_address = str(tx.get("to") or "")
        normalized_allowed = {
            to_checksum_address(address)
            for address in self.authority_scope.allowed_contracts
        }
        if to_checksum_address(to_address) not in normalized_allowed:
            raise ScopeViolationError("transaction target is outside the session scope")
        raw_value = tx.get("value", 0)
        tx_value = int(raw_value) if isinstance(raw_value, (int, float, str)) else 0
        if tx_value > int(self.authority_scope.max_value_wei):
            raise ScopeViolationError("transaction value exceeds the session scope")

    def _active_private_key_bytes(self) -> bytes:
        buffer = self._session_key_buffer
        if buffer is None:
            raise SessionExpiredError("session key not generated")
        material = bytes(buffer.raw[:32])
        if material == b"\x00" * 32:
            raise SessionExpiredError("session key revoked")
        return material


def _smoke_test() -> int:
    os.environ.setdefault(
        "DELEGATE_PRIVATE_KEY",
        "0x59c6995e998f97a5a0044976f7d5f9bc6ab4fe71f5b3c3d4fe26c7c0e59101cf",
    )
    os.environ.setdefault("EIP7702_CHAIN_ID", "1")
    os.environ.setdefault("EIP7702_NONCE", "0")

    allowed_contract = "0x1111111111111111111111111111111111111111"
    session_key = SessionKey(
        delegate_address="0x2222222222222222222222222222222222222222",
        authority_scope=AuthScope(
            allowed_contracts=[allowed_contract],
            max_value_wei=10**18,
            expires_at=int(time.time()) + 300,
        ),
    )
    session_address, signed_auth_tuple = session_key.generate()
    raw_tx = session_key.sign_tx(
        {
            "chainId": 1,
            "to": allowed_contract,
            "value": 0,
            "nonce": 0,
            "gas": 21_000,
            "maxFeePerGas": 2_000_000_000,
            "maxPriorityFeePerGas": 1_000_000_000,
            "data": "0x",
            "type": 2,
        }
    )
    session_key.revoke()

    print("metric\tvalue")
    print(f"session_address\t{session_address}")
    print(f"auth_tuple_bytes\t{len(signed_auth_tuple)}")
    print(f"raw_tx_prefix\t{raw_tx[:10]}")
    return 0 if raw_tx.startswith("0x") and len(signed_auth_tuple) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(_smoke_test())
