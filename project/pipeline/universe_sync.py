"""
Official universe synchronization for large-breadth trading modes.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
NASDAQ_OTHER_URL = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"
NSE_EQUITY_LIST_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
BINANCE_EXCHANGE_INFO_URL = "https://api.binance.com/api/v3/exchangeInfo"
BYBIT_SPOT_INSTRUMENTS_URL = "https://api.bybit.com/v5/market/instruments-info"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json,text/csv,text/plain,*/*",
}
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9.\-]+$")
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR", "3L", "3S", "5L", "5S")
US_EXCLUDE_NAME_PARTS = (
    "warrant",
    "rights",
    "rights ",
    " units",
    " unit ",
    " preferred",
    "depositary shares",
    "contingent value rights",
    "blank check",
    "acquisition corp",
    "acquisition corporation",
    "acquisition company",
    "beneficial interest",
    "trust units",
    "notes",
    "debenture",
    "bond",
)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _safe_int_env(name: str, default: int) -> int:
    try:
        return max(0, int(float(os.getenv(name, str(default)) or default)))
    except Exception:
        return max(0, int(default))


def _resolve_path(path_value: str, default_value: str) -> Path:
    path_text = str(path_value or default_value).strip() or default_value
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _dedupe(values: List[str]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for value in values:
        symbol = str(value or "").strip().upper()
        if not symbol or symbol in seen:
            continue
        ordered.append(symbol)
        seen.add(symbol)
    return ordered


def _slice_symbols(values: List[str], limit: int) -> List[str]:
    if limit <= 0:
        return list(values)
    return list(values[:limit])


def _manifest_path() -> Path:
    return _resolve_path(
        os.getenv("OFFICIAL_UNIVERSE_MANIFEST_PATH", ""),
        "data/universe_sync_manifest.json",
    )


def _read_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _manifest_is_fresh(payload: Dict[str, Any], stale_hours: int, outputs: List[Path]) -> bool:
    if not payload:
        return False
    generated_at = str(payload.get("generated_at") or "").strip()
    if not generated_at:
        return False
    try:
        generated_ts = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except Exception:
        return False
    age = datetime.now(timezone.utc) - generated_ts.astimezone(timezone.utc)
    if age > timedelta(hours=max(1, stale_hours)):
        return False
    return all(path.exists() for path in outputs)


def _write_symbol_file(path: Path, symbols: List[str], header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {header}", f"# generated_at={datetime.now(timezone.utc).isoformat()}"]
    lines.extend(symbols)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fetch_text(url: str, *, timeout: int = 30) -> str:
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.text


def _fetch_json(url: str, *, params: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Dict[str, Any]:
    response = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def _keep_us_symbol(symbol: str, security_name: str, test_issue: str) -> bool:
    text = str(symbol or "").strip().upper()
    if not text or text.startswith("FILE CREATION TIME") or test_issue.strip().upper() == "Y":
        return False
    if not SYMBOL_PATTERN.match(text):
        return False
    lowered = str(security_name or "").strip().lower()
    if any(marker in lowered for marker in US_EXCLUDE_NAME_PARTS):
        return False
    return True


def _fetch_us_symbols() -> List[str]:
    symbols: List[str] = []

    listed_rows = csv.DictReader(io.StringIO(_fetch_text(NASDAQ_LISTED_URL)), delimiter="|")
    for row in listed_rows:
        symbol = str(row.get("Symbol") or "").strip().upper()
        security_name = str(row.get("Security Name") or "").strip()
        test_issue = str(row.get("Test Issue") or "").strip()
        if _keep_us_symbol(symbol, security_name, test_issue):
            symbols.append(symbol)

    other_rows = csv.DictReader(io.StringIO(_fetch_text(NASDAQ_OTHER_URL)), delimiter="|")
    for row in other_rows:
        symbol = str(row.get("ACT Symbol") or row.get("NASDAQ Symbol") or "").strip().upper()
        security_name = str(row.get("Security Name") or "").strip()
        test_issue = str(row.get("Test Issue") or "").strip()
        if _keep_us_symbol(symbol, security_name, test_issue):
            symbols.append(symbol)

    return _dedupe(symbols)


def _fetch_nse_symbols() -> List[str]:
    payload = _fetch_text(NSE_EQUITY_LIST_URL)
    reader = csv.DictReader(io.StringIO(payload))
    symbols: List[str] = []
    for row in reader:
        series = str(row.get(" SERIES") or row.get("SERIES") or "").strip().upper()
        symbol = str(row.get("SYMBOL") or "").strip().upper()
        if series != "EQ" or not symbol or not re.match(r"^[A-Z0-9\-]+$", symbol):
            continue
        symbols.append(f"{symbol}.NS")
    return _dedupe(symbols)


def _is_supported_crypto_symbol(symbol: str, base_asset: str, quote_asset: str) -> bool:
    text = str(symbol or "").strip().upper()
    base = str(base_asset or "").strip().upper()
    quote = str(quote_asset or "").strip().upper()
    if not text or not base or quote not in {"USDT", "USDC", "BUSD"}:
        return False
    if not text.endswith(quote):
        return False
    if not re.match(r"^[A-Z0-9]+$", text):
        return False
    if any(base.endswith(suffix) for suffix in LEVERAGED_SUFFIXES):
        return False
    return True


def _fetch_binance_symbols() -> List[str]:
    payload = _fetch_json(BINANCE_EXCHANGE_INFO_URL)
    rows = payload.get("symbols") or []
    symbols: List[str] = []
    for row in rows:
        if str(row.get("status") or "").upper() != "TRADING":
            continue
        if not bool(row.get("isSpotTradingAllowed", True)):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        base_asset = str(row.get("baseAsset") or "").strip().upper()
        quote_asset = str(row.get("quoteAsset") or "").strip().upper()
        if _is_supported_crypto_symbol(symbol, base_asset, quote_asset):
            symbols.append(symbol)
    return _dedupe(symbols)


def _fetch_bybit_symbols() -> List[str]:
    cursor = ""
    collected: List[str] = []
    while True:
        params: Dict[str, Any] = {"category": "spot", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        payload = _fetch_json(BYBIT_SPOT_INSTRUMENTS_URL, params=params)
        result = payload.get("result") or {}
        rows = result.get("list") or []
        for row in rows:
            if str(row.get("status") or "").lower() != "trading":
                continue
            symbol = str(row.get("symbol") or "").strip().upper()
            base_coin = str(row.get("baseCoin") or "").strip().upper()
            quote_coin = str(row.get("quoteCoin") or "").strip().upper()
            if _is_supported_crypto_symbol(symbol, base_coin, quote_coin):
                collected.append(symbol)
        cursor = str(result.get("nextPageCursor") or "").strip()
        if not cursor or not rows:
            break
    return _dedupe(collected)


def sync_official_universes(force: bool = False) -> Dict[str, Any]:
    if not _env_flag("OFFICIAL_UNIVERSE_AUTO_SYNC", True):
        return {"status": "disabled", "reason": "OFFICIAL_UNIVERSE_AUTO_SYNC=0"}

    us_path = _resolve_path(
        os.getenv("OFFICIAL_US_UNIVERSE_FILE", os.getenv("US_UNIVERSE_EXTRA_FILE", "")),
        "data/universe_us_official.txt",
    )
    nse_path = _resolve_path(
        os.getenv("OFFICIAL_NSE_UNIVERSE_FILE", os.getenv("NSE_UNIVERSE_EXTRA_FILE", "")),
        "data/universe_nse_official.txt",
    )
    crypto_path = _resolve_path(
        os.getenv("OFFICIAL_CRYPTO_SYMBOLS_FILE", os.getenv("CRYPTO_DEPTH_SYMBOLS_FILE", "")),
        "data/crypto_symbols_official.txt",
    )
    binance_path = _resolve_path(
        os.getenv("OFFICIAL_BINANCE_CRYPTO_SYMBOLS_FILE", os.getenv("BINANCE_CRYPTO_SYMBOLS_FILE", "")),
        "data/crypto_symbols_binance_official.txt",
    )
    bybit_path = _resolve_path(
        os.getenv("OFFICIAL_BYBIT_CRYPTO_SYMBOLS_FILE", os.getenv("BYBIT_CRYPTO_SYMBOLS_FILE", "")),
        "data/crypto_symbols_bybit_official.txt",
    )
    manifest_path = _manifest_path()
    outputs = [us_path, nse_path, crypto_path, binance_path, bybit_path]

    stale_hours = max(1, _safe_int_env("OFFICIAL_UNIVERSE_STALE_HOURS", 24))
    cached = _read_manifest(manifest_path)
    if not force and _manifest_is_fresh(cached, stale_hours, outputs):
        cached["status"] = "fresh"
        return cached

    us_limit = _safe_int_env("OFFICIAL_US_MAX_SYMBOLS", 6000)
    nse_limit = _safe_int_env("OFFICIAL_NSE_MAX_SYMBOLS", 3000)
    crypto_limit = _safe_int_env("OFFICIAL_CRYPTO_MAX_SYMBOLS", 800)
    exchange_crypto_limit = _safe_int_env("OFFICIAL_CRYPTO_EXCHANGE_MAX_SYMBOLS", max(crypto_limit, 800))

    us_symbols = _slice_symbols(_fetch_us_symbols(), us_limit)
    nse_symbols = _slice_symbols(_fetch_nse_symbols(), nse_limit)
    binance_symbols = _slice_symbols(_fetch_binance_symbols(), exchange_crypto_limit)
    bybit_symbols = _slice_symbols(_fetch_bybit_symbols(), exchange_crypto_limit)

    binance_set = set(binance_symbols)
    bybit_set = set(bybit_symbols)
    shared_crypto = [symbol for symbol in binance_symbols if symbol in bybit_set]
    combined_crypto = shared_crypto + [symbol for symbol in binance_symbols if symbol not in bybit_set] + [
        symbol for symbol in bybit_symbols if symbol not in binance_set
    ]
    combined_crypto = _slice_symbols(_dedupe(combined_crypto), crypto_limit)

    _write_symbol_file(us_path, us_symbols, "Official US symbol universe")
    _write_symbol_file(nse_path, nse_symbols, "Official NSE EQ symbol universe")
    _write_symbol_file(crypto_path, combined_crypto, "Official crypto spot universe (shared-first)")
    _write_symbol_file(binance_path, binance_symbols, "Official Binance spot crypto universe")
    _write_symbol_file(bybit_path, bybit_symbols, "Official Bybit spot crypto universe")

    payload = {
        "status": "ok",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "us_symbols": len(us_symbols),
        "nse_symbols": len(nse_symbols),
        "crypto_symbols": len(combined_crypto),
        "binance_crypto_symbols": len(binance_symbols),
        "bybit_crypto_symbols": len(bybit_symbols),
        "shared_crypto_symbols": len(shared_crypto),
        "files": {
            "us": str(us_path),
            "nse": str(nse_path),
            "crypto": str(crypto_path),
            "binance_crypto": str(binance_path),
            "bybit_crypto": str(bybit_path),
        },
        "sources": {
            "us": [NASDAQ_LISTED_URL, NASDAQ_OTHER_URL],
            "nse": [NSE_EQUITY_LIST_URL],
            "crypto": [BINANCE_EXCHANGE_INFO_URL, BYBIT_SPOT_INSTRUMENTS_URL],
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    print(json.dumps(sync_official_universes(force=True), indent=2))
