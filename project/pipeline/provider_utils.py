import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


def parse_api_keys(*env_names: str) -> List[str]:
    keys: List[str] = []
    for env_name in env_names:
        raw = os.getenv(env_name, "")
        for part in raw.replace(";", ",").split(","):
            key = part.strip()
            if key and key not in keys:
                keys.append(key)
    return keys


def stable_rotate(items: List[Any], seed_text: str) -> List[Any]:
    if not items:
        return []
    start = sum(ord(ch) for ch in seed_text) % len(items)
    return items[start:] + items[:start]


def looks_rate_limited(message: str) -> bool:
    lowered = (message or "").lower()
    needles = (
        "rate limit",
        "too many requests",
        "call frequency",
        "quota",
        "limit reached",
        "max requests",
        "premium",
        "exceeded",
        "429",
        "credits",
        "throttl",
    )
    return any(needle in lowered for needle in needles)


class APIKeyPool:
    def __init__(self, keys: List[str], default_cooldown: float = 300.0):
        self._keys = keys
        self._default_cooldown = default_cooldown
        self._cooldowns: Dict[str, float] = {key: 0.0 for key in keys}
        self._cursor = 0
        self._lock = threading.Lock()

    def acquire(self) -> Optional[str]:
        if not self._keys:
            return None

        now = time.time()
        with self._lock:
            for offset in range(len(self._keys)):
                idx = (self._cursor + offset) % len(self._keys)
                key = self._keys[idx]
                if self._cooldowns.get(key, 0.0) <= now:
                    self._cursor = (idx + 1) % len(self._keys)
                    return key
        return None

    def cool_down(self, key: str, cooldown_seconds: Optional[float] = None):
        if not key:
            return
        cooldown = cooldown_seconds if cooldown_seconds is not None else self._default_cooldown
        with self._lock:
            self._cooldowns[key] = max(self._cooldowns.get(key, 0.0), time.time() + cooldown)

    @property
    def configured(self) -> bool:
        return bool(self._keys)


@dataclass
class FetchOutcome:
    provider: str
    data: Any = None
    status: str = "ok"
    error: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.data is not None

    @property
    def rate_limited(self) -> bool:
        return self.status == "rate_limited"
