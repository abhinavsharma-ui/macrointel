"""
Price Data Pipeline
===================
Rotates across multiple providers for daily OHLCV:
  - yfinance (free baseline)
  - Finnhub candles
  - Alpha Vantage daily adjusted
  - Twelve Data time series
"""

import logging
import os
import time
import warnings
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional
import re
from pathlib import Path

import pandas as pd
import requests

from pipeline.provider_utils import APIKeyPool, FetchOutcome, looks_rate_limited, parse_api_keys, stable_rotate
from pipeline.universe import get_universe

logger = logging.getLogger(__name__)

ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"
FINNHUB_BASE = "https://finnhub.io/api/v1/stock/candle"
TWELVE_DATA_BASE = "https://api.twelvedata.com/time_series"

DEFAULT_UNIVERSE_MODE = os.getenv("UNIVERSE_MODE", "full").strip().lower() or "full"
if DEFAULT_UNIVERSE_MODE not in {"core", "full", "us", "nse"}:
    DEFAULT_UNIVERSE_MODE = "full"

PRICE_PROVIDER_ORDER = [
    name.strip().lower()
    for name in os.getenv("PRICE_PROVIDER_ORDER", "yfinance,finnhub,alpha_vantage,twelve_data").split(",")
    if name.strip()
]
PRICE_MAX_PROVIDER_ATTEMPTS = max(1, int(os.getenv("PRICE_MAX_PROVIDER_ATTEMPTS", "2")))
PRICE_FETCH_FULL_HISTORY = os.getenv("PRICE_FETCH_FULL_HISTORY", "1").strip().lower() not in {"0", "false", "no"}
PRICE_SYMBOL_REFRESH_SECONDS = max(60, int(os.getenv("PRICE_SYMBOL_REFRESH_SECONDS", "3600")))
PRICE_KEEP_FULL_HISTORY = os.getenv("PRICE_KEEP_FULL_HISTORY", "0").strip().lower() in {"1", "true", "yes"}
PRICE_STORE_FULL_HISTORY = os.getenv("PRICE_STORE_FULL_HISTORY", "0").strip().lower() in {"1", "true", "yes"}
PRICE_STORE_FULL_DIR = Path(os.getenv("PRICE_STORE_FULL_DIR", "data/prices_full"))
PRICE_RECENT_DAYS = max(120, int(os.getenv("PRICE_RECENT_DAYS", "420")))
PRICE_FULL_FETCH_ONCE = os.getenv("PRICE_FULL_FETCH_ONCE", "1").strip().lower() not in {"0", "false", "no"}
PRICE_NO_DATA_LOG_LIMIT = max(0, int(os.getenv("PRICE_NO_DATA_LOG_LIMIT", "6")))
PRICE_LOG_SUCCESS_EVERY = max(0, int(os.getenv("PRICE_LOG_SUCCESS_EVERY", "25")))
YFINANCE_SILENCE = os.getenv("YFINANCE_SILENCE", "1").strip().lower() not in {"0", "false", "no"}

_SHARED_PRICE_CACHE: Dict[str, pd.DataFrame] = {}
_SHARED_RECENT_CACHE: Dict[str, pd.DataFrame] = {}
_SHARED_PROVIDER_CACHE: Dict[str, str] = {}
_SHARED_FETCHED_AT: Dict[str, float] = {}
_SHARED_FULL_FETCHED: Dict[str, bool] = {}
_YFINANCE_LOGGING_CONFIGURED = False


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _normalize_price_frame(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None

    frame = df.copy()
    index = pd.to_datetime(frame.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    frame.index = index
    frame.index.name = "date"
    for col in ("open", "high", "low", "close", "volume"):
        if col not in frame.columns:
            return None
    if "adj_close" not in frame.columns:
        frame["adj_close"] = frame["close"]
    return frame.sort_index()


def _configure_yfinance_logging():
    global _YFINANCE_LOGGING_CONFIGURED
    if _YFINANCE_LOGGING_CONFIGURED or not YFINANCE_SILENCE:
        return
    for noisy_logger in ("yfinance", "peewee", "urllib3", "curl_cffi"):
        target = logging.getLogger(noisy_logger)
        target.setLevel(logging.CRITICAL)
        target.propagate = False
    _YFINANCE_LOGGING_CONFIGURED = True


def _safe_symbol_filename(symbol: str) -> str:
    raw = (symbol or "").strip()
    if not raw:
        return "unknown"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
    cleaned = cleaned.strip("._-") or "unknown"
    return cleaned.upper()


class AlphaVantageFetcher:
    name = "alpha_vantage"

    def __init__(self):
        self.keys = APIKeyPool(
            parse_api_keys("ALPHA_VANTAGE_API_KEYS", "ALPHA_VANTAGE_API_KEY"),
            default_cooldown=float(os.getenv("ALPHA_VANTAGE_COOLDOWN_SECONDS", "300")),
        )
        self._min_call_interval = float(os.getenv("ALPHA_VANTAGE_MIN_CALL_INTERVAL", "12.5"))
        self._last_call_by_key: Dict[str, float] = {}

    def supports(self, symbol: str) -> bool:
        if symbol.startswith("^"):
            return False
        if symbol.endswith(".NS"):
            return os.getenv("ALPHA_VANTAGE_ENABLE_NSE", "0").strip().lower() in {"1", "true", "yes"}
        return True

    def _throttle(self, api_key: str):
        elapsed = time.time() - self._last_call_by_key.get(api_key, 0.0)
        if elapsed < self._min_call_interval:
            time.sleep(self._min_call_interval - elapsed)
        self._last_call_by_key[api_key] = time.time()

    def fetch_daily(self, symbol: str, full: bool = False) -> FetchOutcome:
        api_key = self.keys.acquire()
        if not api_key:
            return FetchOutcome(self.name, status="unavailable", error="No Alpha Vantage key available")

        self._throttle(api_key)
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "full" if full else "compact",
            "apikey": api_key,
        }

        try:
            resp = requests.get(ALPHA_VANTAGE_BASE, params=params, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
            series = payload.get("Time Series (Daily)")
            if not series:
                error = payload.get("Note") or payload.get("Information") or payload.get("Error Message") or "No daily data returned"
                if looks_rate_limited(error):
                    self.keys.cool_down(api_key)
                    return FetchOutcome(self.name, status="rate_limited", error=error)
                return FetchOutcome(self.name, status="empty", error=error)

            records = []
            for dt_str, values in series.items():
                records.append({
                    "date": pd.Timestamp(dt_str),
                    "open": _as_float(values.get("1. open")),
                    "high": _as_float(values.get("2. high")),
                    "low": _as_float(values.get("3. low")),
                    "close": _as_float(values.get("4. close")),
                    "adj_close": _as_float(values.get("5. adjusted close")),
                    "volume": _as_int(values.get("6. volume")),
                })

            frame = _normalize_price_frame(pd.DataFrame(records).set_index("date"))
            if frame is None:
                return FetchOutcome(self.name, status="empty", error="Alpha Vantage returned malformed data")
            return FetchOutcome(self.name, data=frame)
        except Exception as exc:
            error = str(exc)
            if looks_rate_limited(error):
                self.keys.cool_down(api_key)
                return FetchOutcome(self.name, status="rate_limited", error=error)
            return FetchOutcome(self.name, status="error", error=error)


class FinnhubPriceFetcher:
    name = "finnhub"

    def __init__(self):
        self.keys = APIKeyPool(
            parse_api_keys("FINNHUB_API_KEYS", "FINNHUB_API_KEY"),
            default_cooldown=float(os.getenv("FINNHUB_COOLDOWN_SECONDS", "120")),
        )

    def supports(self, symbol: str) -> bool:
        return not symbol.endswith(".NS") and not symbol.startswith("^")

    def fetch_daily(self, symbol: str, full: bool = False) -> FetchOutcome:
        api_key = self.keys.acquire()
        if not api_key:
            return FetchOutcome(self.name, status="unavailable", error="No Finnhub key available")

        start_date = date.today() - timedelta(days=365 * 3 if full else 365)
        end_date = date.today()
        params = {
            "symbol": symbol,
            "resolution": "D",
            "from": int(datetime.combine(start_date, datetime.min.time()).timestamp()),
            "to": int(datetime.combine(end_date, datetime.min.time()).timestamp()),
            "token": api_key,
        }

        try:
            resp = requests.get(FINNHUB_BASE, params=params, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("s") != "ok":
                error = payload.get("error") or payload.get("s") or "No candle data returned"
                if looks_rate_limited(error):
                    self.keys.cool_down(api_key)
                    return FetchOutcome(self.name, status="rate_limited", error=error)
                return FetchOutcome(self.name, status="empty", error=error)

            frame = pd.DataFrame({
                "date": pd.to_datetime(payload.get("t", []), unit="s"),
                "open": payload.get("o", []),
                "high": payload.get("h", []),
                "low": payload.get("l", []),
                "close": payload.get("c", []),
                "volume": payload.get("v", []),
            }).set_index("date")
            frame["adj_close"] = frame["close"]
            frame = _normalize_price_frame(frame)
            if frame is None:
                return FetchOutcome(self.name, status="empty", error="Finnhub returned malformed data")
            return FetchOutcome(self.name, data=frame)
        except Exception as exc:
            error = str(exc)
            if looks_rate_limited(error):
                self.keys.cool_down(api_key)
                return FetchOutcome(self.name, status="rate_limited", error=error)
            return FetchOutcome(self.name, status="error", error=error)


class TwelveDataFetcher:
    name = "twelve_data"

    def __init__(self):
        self.keys = APIKeyPool(
            parse_api_keys("TWELVE_DATA_API_KEYS", "TWELVE_DATA_API_KEY"),
            default_cooldown=float(os.getenv("TWELVE_DATA_COOLDOWN_SECONDS", "300")),
        )

    def supports(self, symbol: str) -> bool:
        if symbol.startswith("^"):
            return False
        if symbol.endswith(".NS"):
            return os.getenv("TWELVE_DATA_ENABLE_NSE", "0").strip().lower() in {"1", "true", "yes"}
        return True

    def fetch_daily(self, symbol: str, full: bool = False) -> FetchOutcome:
        api_key = self.keys.acquire()
        if not api_key:
            return FetchOutcome(self.name, status="unavailable", error="No Twelve Data key available")

        params = {
            "symbol": symbol,
            "interval": "1day",
            "outputsize": "5000" if full else "365",
            "format": "JSON",
            "apikey": api_key,
        }

        try:
            resp = requests.get(TWELVE_DATA_BASE, params=params, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
            values = payload.get("values")
            if not values:
                error = payload.get("message") or payload.get("status") or "No Twelve Data values returned"
                if looks_rate_limited(error):
                    self.keys.cool_down(api_key)
                    return FetchOutcome(self.name, status="rate_limited", error=error)
                return FetchOutcome(self.name, status="empty", error=error)

            frame = pd.DataFrame([
                {
                    "date": pd.Timestamp(row["datetime"]),
                    "open": _as_float(row.get("open")),
                    "high": _as_float(row.get("high")),
                    "low": _as_float(row.get("low")),
                    "close": _as_float(row.get("close")),
                    "adj_close": _as_float(row.get("close")),
                    "volume": _as_int(row.get("volume")),
                }
                for row in values
            ]).set_index("date")
            frame = _normalize_price_frame(frame)
            if frame is None:
                return FetchOutcome(self.name, status="empty", error="Twelve Data returned malformed data")
            return FetchOutcome(self.name, data=frame)
        except Exception as exc:
            error = str(exc)
            if looks_rate_limited(error):
                self.keys.cool_down(api_key)
                return FetchOutcome(self.name, status="rate_limited", error=error)
            return FetchOutcome(self.name, status="error", error=error)


class YFinanceFetcher:
    name = "yfinance"

    def supports(self, symbol: str) -> bool:
        return True

    def fetch_daily(self, symbol: str, full: bool = False) -> FetchOutcome:
        try:
            _configure_yfinance_logging()
            with warnings.catch_warnings():
                if YFINANCE_SILENCE:
                    warnings.simplefilter("ignore")
                import yfinance as yf

                start = date.today() - timedelta(days=365 * 3 if full else 365)
                end = date.today()
                ticker = yf.Ticker(symbol)
                raw = ticker.history(start=str(start), end=str(end), auto_adjust=True)
            if raw.empty:
                return FetchOutcome(self.name, status="empty", error="yfinance returned no history")

            frame = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            frame.columns = ["open", "high", "low", "close", "volume"]
            frame["adj_close"] = frame["close"]
            frame = _normalize_price_frame(frame)
            if frame is None:
                return FetchOutcome(self.name, status="empty", error="yfinance returned malformed data")
            return FetchOutcome(self.name, data=frame)
        except Exception as exc:
            error = str(exc)
            error_lower = error.lower()
            if any(
                marker in error_lower
                for marker in (
                    "possibly delisted",
                    "no data found",
                    "no price data found",
                    "symbol may be delisted",
                    "no timezone found",
                )
            ):
                return FetchOutcome(self.name, status="empty", error="yfinance returned no history")
            return FetchOutcome(self.name, status="error", error=error)


class MultiSourcePriceFetcher:
    def __init__(self):
        registry = {
            "alpha_vantage": AlphaVantageFetcher(),
            "finnhub": FinnhubPriceFetcher(),
            "twelve_data": TwelveDataFetcher(),
            "yfinance": YFinanceFetcher(),
        }
        ordered_names = [name for name in PRICE_PROVIDER_ORDER if name in registry]
        if "yfinance" not in ordered_names:
            ordered_names.append("yfinance")
        self.providers = [registry[name] for name in ordered_names]
        self._preferred_provider_by_symbol: Dict[str, str] = {}

    def fetch_daily(self, symbol: str, full: bool = False) -> FetchOutcome:
        primaries = [provider for provider in self.providers if provider.name == "yfinance"]
        backups = [provider for provider in self.providers if provider.name != "yfinance"]
        provider_order = primaries + stable_rotate(backups, symbol)
        preferred = self._preferred_provider_by_symbol.get(symbol)
        if preferred and preferred != "yfinance":
            provider_order = primaries + [p for p in stable_rotate(backups, symbol) if p.name == preferred] + [
                p for p in stable_rotate(backups, symbol) if p.name != preferred
            ]

        errors: List[str] = []
        attempts = 0
        for provider in provider_order:
            if not provider.supports(symbol):
                continue
            attempts += 1
            if attempts > PRICE_MAX_PROVIDER_ATTEMPTS:
                break
            outcome = provider.fetch_daily(symbol, full=full)
            if outcome.ok:
                self._preferred_provider_by_symbol[symbol] = provider.name
                return outcome
            if outcome.error:
                errors.append(f"{provider.name}: {outcome.error}")

        return FetchOutcome("multi_source", status="error", error=" | ".join(errors[-4:]) or "No provider returned data")


class PriceDataPipeline:
    def __init__(self, symbols: Optional[List[str]] = None):
        self.symbols = symbols or get_universe(DEFAULT_UNIVERSE_MODE)
        self.fetcher = MultiSourcePriceFetcher()
        self._cache = _SHARED_PRICE_CACHE
        self._recent_cache = _SHARED_RECENT_CACHE
        self._provider_by_symbol = _SHARED_PROVIDER_CACHE
        self._fetched_at = _SHARED_FETCHED_AT
        self._full_fetched = _SHARED_FULL_FETCHED
        self._full_history = PRICE_FETCH_FULL_HISTORY
        self._keep_full_history = PRICE_KEEP_FULL_HISTORY
        self._refresh_seconds = PRICE_SYMBOL_REFRESH_SECONDS

    def _build_recent_slice(self, df: pd.DataFrame) -> pd.DataFrame:
        cutoff = pd.Timestamp(date.today() - timedelta(days=PRICE_RECENT_DAYS))
        return df[df.index >= cutoff].copy()

    def _fetch_symbol(self, symbol: str) -> FetchOutcome:
        already_persisted_full = False
        if PRICE_STORE_FULL_HISTORY and PRICE_FULL_FETCH_ONCE:
            try:
                full_path = PRICE_STORE_FULL_DIR / f"{_safe_symbol_filename(symbol)}.parquet"
                already_persisted_full = full_path.exists()
            except Exception:
                already_persisted_full = False

        full = self._full_history and (not PRICE_FULL_FETCH_ONCE or (not self._full_fetched.get(symbol, False) and not already_persisted_full))
        outcome = self.fetcher.fetch_daily(symbol, full=full)
        if outcome.ok and full:
            self._full_fetched[symbol] = True
        return outcome

    def _get_cached_symbol(self, symbol: str) -> Optional[FetchOutcome]:
        cached = self._cache.get(symbol)
        fetched_at = self._fetched_at.get(symbol, 0.0)
        if cached is None or cached.empty:
            return None
        if (time.time() - fetched_at) > self._refresh_seconds:
            return None
        provider = self._provider_by_symbol.get(symbol, "cache")
        return FetchOutcome(provider=provider, data=cached)

    def run_incremental_update(self) -> Dict:
        logger.info(f"PriceDataPipeline: fetching {len(self.symbols)} symbols using {', '.join(PRICE_PROVIDER_ORDER)}")
        errors = []
        no_data_symbols = []
        missing_column_symbols = []
        price_daily_recent = {}
        price_daily_full = {}
        provider_by_symbol = {}
        reused = 0
        fetched = 0

        for i, symbol in enumerate(self.symbols, start=1):
            try:
                reused_this_symbol = False
                outcome = self._get_cached_symbol(symbol)
                if outcome is not None:
                    reused += 1
                    reused_this_symbol = True
                else:
                    outcome = self._fetch_symbol(symbol)
                    if outcome.ok:
                        fetched += 1
                df = outcome.data if outcome.ok else None
                if df is None or df.empty:
                    no_data_symbols.append(symbol)
                    errors.append(symbol)
                    continue

                required = {"open", "high", "low", "close", "volume"}
                if not required.issubset(df.columns):
                    errors.append(symbol)
                    missing_column_symbols.append(symbol)
                    continue

                df = df.dropna(subset=["close", "volume"])
                recent_df = self._build_recent_slice(df)
                price_daily_full[symbol] = df if self._keep_full_history else recent_df
                price_daily_recent[symbol] = recent_df

                # Keep memory bounded unless explicitly requested.
                self._cache[symbol] = df if self._keep_full_history else recent_df
                self._recent_cache[symbol] = recent_df
                self._fetched_at[symbol] = time.time()
                provider_by_symbol[symbol] = outcome.provider
                self._provider_by_symbol[symbol] = outcome.provider

                # Persist full history to disk without keeping it in RAM.
                # This lets you run large universes on modest instances (e.g., 4 vCPU / 16 GB).
                if PRICE_STORE_FULL_HISTORY and (len(df) > len(recent_df)):
                    try:
                        PRICE_STORE_FULL_DIR.mkdir(parents=True, exist_ok=True)
                        full_path = PRICE_STORE_FULL_DIR / f"{_safe_symbol_filename(symbol)}.parquet"
                        df.sort_index().to_parquet(full_path)
                    except Exception:
                        pass

                source_note = "cache" if reused_this_symbol else outcome.provider
                if PRICE_LOG_SUCCESS_EVERY and (i % PRICE_LOG_SUCCESS_EVERY == 0 or i == len(self.symbols)):
                    logger.info(
                        f"PriceDataPipeline progress: {i}/{len(self.symbols)} scanned | "
                        f"{len(price_daily_recent)} ok | {len(errors)} skipped | latest {symbol} via {source_note}"
                    )
                else:
                    logger.debug(f"[{i}/{len(self.symbols)}] {symbol}: {len(recent_df)} recent rows OK via {source_note}")
            except Exception as exc:
                logger.error(f"Pipeline error for {symbol}: {exc}")
                errors.append(symbol)

        result = {
            "price_daily_recent": price_daily_recent,
            "price_daily_full": price_daily_full,
            "fetch_errors": errors,
            "symbols_ok": len(price_daily_recent),
            "price_provider_by_symbol": provider_by_symbol,
            "run_timestamp": datetime.utcnow().isoformat(),
        }
        if no_data_symbols:
            preview = ", ".join(no_data_symbols[:PRICE_NO_DATA_LOG_LIMIT])
            suffix = " ..." if len(no_data_symbols) > PRICE_NO_DATA_LOG_LIMIT and preview else ""
            logger.info(
                f"PriceDataPipeline skipped {len(no_data_symbols)} symbols with no data"
                + (f": {preview}{suffix}" if preview else "")
            )
        if missing_column_symbols:
            preview = ", ".join(missing_column_symbols[:PRICE_NO_DATA_LOG_LIMIT])
            suffix = " ..." if len(missing_column_symbols) > PRICE_NO_DATA_LOG_LIMIT and preview else ""
            logger.info(
                f"PriceDataPipeline skipped {len(missing_column_symbols)} symbols with malformed OHLCV"
                + (f": {preview}{suffix}" if preview else "")
            )
        logger.info(
            f"PriceDataPipeline complete: {result['symbols_ok']}/{len(self.symbols)} symbols OK, "
            f"{len(errors)} errors, {fetched} fetched, {reused} reused"
        )
        return result

    def get_latest_close(self, symbol: str) -> Optional[float]:
        df = self._cache.get(symbol)
        if df is not None and not df.empty:
            return float(df["close"].iloc[-1])
        return None
