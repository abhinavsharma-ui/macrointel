"""
Earnings Event Pipeline
=======================
Collects recent earnings events with quota-aware provider fallback.

Design goals:
  - Prefer batch endpoints first so we do not burn one API call per symbol
  - Fall back per-symbol only for uncovered names
  - Persist known earnings into the PIT database when available
"""

import logging
import os
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests

from pipeline.provider_utils import APIKeyPool, FetchOutcome, looks_rate_limited, parse_api_keys, to_eodhd_symbol

logger = logging.getLogger(__name__)

FINNHUB_EARNINGS_BASE = "https://finnhub.io/api/v1/calendar/earnings"
ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"
EODHD_EARNINGS_BASE = "https://eodhd.com/api/calendar/earnings"

EARNINGS_PROVIDER_ORDER = [
    name.strip().lower()
    for name in os.getenv("EARNINGS_PROVIDER_ORDER", "finnhub,alpha_vantage,eodhd").split(",")
    if name.strip()
]
EARNINGS_LOOKBACK_DAYS = max(7, int(os.getenv("EARNINGS_LOOKBACK_DAYS", "45")))
EARNINGS_FORWARD_DAYS = max(0, int(os.getenv("EARNINGS_FORWARD_DAYS", "7")))
EARNINGS_MAX_PROVIDER_ATTEMPTS = max(1, int(os.getenv("EARNINGS_MAX_PROVIDER_ATTEMPTS", "3")))
EARNINGS_ALPHA_MAX_SYMBOLS = max(0, int(os.getenv("EARNINGS_ALPHA_MAX_SYMBOLS", "25")))
EARNINGS_EODHD_MAX_SYMBOLS = max(0, int(os.getenv("EODHD_EARNINGS_MAX_SYMBOLS", "80")))


def _to_float(value) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_event_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "fiscal_date",
                "reported_date",
                "reported_eps",
                "estimated_eps",
                "surprise_pct",
                "revenue",
                "pe_ratio",
                "source",
            ]
        )

    result = frame.copy()
    result["symbol"] = result["symbol"].astype(str).str.upper().str.strip()
    result["reported_date"] = pd.to_datetime(result["reported_date"], errors="coerce").dt.normalize()
    result["fiscal_date"] = pd.to_datetime(result.get("fiscal_date"), errors="coerce").dt.normalize()
    result["fiscal_date"] = result["fiscal_date"].fillna(result["reported_date"])
    result["source"] = result.get("source", "unknown").fillna("unknown")

    for col in ("reported_eps", "estimated_eps", "surprise_pct", "revenue", "pe_ratio"):
        result[col] = pd.to_numeric(result.get(col), errors="coerce")

    result = result.dropna(subset=["symbol", "reported_date"])
    result = result.drop_duplicates(subset=["symbol", "reported_date", "source"], keep="last")
    return result.sort_values(["reported_date", "symbol"]).reset_index(drop=True)


class FinnhubEarningsCalendarFetcher:
    name = "finnhub"

    def __init__(self):
        self.keys = APIKeyPool(
            parse_api_keys("FINNHUB_API_KEYS", "FINNHUB_API_KEY"),
            default_cooldown=float(os.getenv("FINNHUB_EARNINGS_COOLDOWN_SECONDS", "300")),
        )

    def supports(self, symbols: List[str]) -> bool:
        return any(not symbol.endswith(".NS") and not symbol.startswith("^") for symbol in symbols)

    def fetch(self, symbols: List[str], from_date: date, to_date: date) -> FetchOutcome:
        api_key = self.keys.acquire()
        if not api_key:
            return FetchOutcome(self.name, status="unavailable", error="No Finnhub key available")

        tracked = {symbol for symbol in symbols if not symbol.endswith(".NS") and not symbol.startswith("^")}
        if not tracked:
            return FetchOutcome(self.name, data=_normalize_event_frame(pd.DataFrame()))

        params = {
            "from": str(from_date),
            "to": str(to_date),
            "token": api_key,
        }

        try:
            resp = requests.get(FINNHUB_EARNINGS_BASE, params=params, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
            items = payload.get("earningsCalendar") or payload.get("earnings_calendar") or []
            if not items:
                error = payload.get("error") or "Finnhub returned no earnings events"
                if looks_rate_limited(error):
                    self.keys.cool_down(api_key)
                    return FetchOutcome(self.name, status="rate_limited", error=error)
                return FetchOutcome(self.name, status="empty", error=error)

            records = []
            for item in items:
                symbol = str(item.get("symbol", "")).upper().strip()
                if symbol not in tracked:
                    continue

                reported_date = item.get("date") or item.get("reportDate")
                est_eps = _to_float(item.get("epsEstimate"))
                act_eps = _to_float(item.get("epsActual"))
                surprise_pct = _to_float(item.get("surprise"))
                if surprise_pct is None and est_eps not in (None, 0) and act_eps is not None:
                    surprise_pct = ((act_eps - est_eps) / abs(est_eps)) * 100.0

                revenue_actual = _to_float(item.get("revenueActual"))
                revenue_estimate = _to_float(item.get("revenueEstimate"))
                fiscal_date = item.get("period") or reported_date

                records.append(
                    {
                        "symbol": symbol,
                        "fiscal_date": fiscal_date,
                        "reported_date": reported_date,
                        "reported_eps": act_eps,
                        "estimated_eps": est_eps,
                        "surprise_pct": surprise_pct,
                        "revenue": revenue_actual if revenue_actual is not None else revenue_estimate,
                        "pe_ratio": None,
                        "source": self.name,
                    }
                )

            frame = _normalize_event_frame(pd.DataFrame(records))
            return FetchOutcome(self.name, data=frame)
        except Exception as exc:
            error = str(exc)
            if looks_rate_limited(error):
                self.keys.cool_down(api_key)
                return FetchOutcome(self.name, status="rate_limited", error=error)
            return FetchOutcome(self.name, status="error", error=error)


class AlphaVantageEarningsFetcher:
    name = "alpha_vantage"

    def __init__(self):
        self.keys = APIKeyPool(
            parse_api_keys("ALPHA_VANTAGE_API_KEYS", "ALPHA_VANTAGE_API_KEY"),
            default_cooldown=float(os.getenv("ALPHA_VANTAGE_EARNINGS_COOLDOWN_SECONDS", "300")),
        )
        self._min_call_interval = float(os.getenv("ALPHA_VANTAGE_EARNINGS_MIN_CALL_INTERVAL", "12.5"))
        self._last_call_by_key: Dict[str, float] = {}

    def supports(self, symbol: str) -> bool:
        if symbol.endswith(".NS") or symbol.startswith("^"):
            return False
        return True

    def _throttle(self, api_key: str):
        elapsed = time.time() - self._last_call_by_key.get(api_key, 0.0)
        if elapsed < self._min_call_interval:
            time.sleep(self._min_call_interval - elapsed)
        self._last_call_by_key[api_key] = time.time()

    def fetch_symbol(self, symbol: str, from_date: date, to_date: date) -> FetchOutcome:
        api_key = self.keys.acquire()
        if not api_key:
            return FetchOutcome(self.name, status="unavailable", error="No Alpha Vantage key available")

        self._throttle(api_key)
        params = {
            "function": "EARNINGS",
            "symbol": symbol,
            "apikey": api_key,
        }

        try:
            resp = requests.get(ALPHA_VANTAGE_BASE, params=params, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
            rows = payload.get("quarterlyEarnings", [])
            if not rows:
                error = payload.get("Note") or payload.get("Information") or payload.get("Error Message") or "Alpha Vantage returned no earnings rows"
                if looks_rate_limited(error):
                    self.keys.cool_down(api_key)
                    return FetchOutcome(self.name, status="rate_limited", error=error)
                return FetchOutcome(self.name, status="empty", error=error)

            records = []
            for row in rows:
                reported_date = pd.to_datetime(row.get("reportedDate"), errors="coerce")
                if pd.isna(reported_date):
                    continue
                if reported_date.date() < from_date or reported_date.date() > to_date:
                    continue

                records.append(
                    {
                        "symbol": symbol,
                        "fiscal_date": row.get("fiscalDateEnding") or row.get("reportedDate"),
                        "reported_date": row.get("reportedDate"),
                        "reported_eps": _to_float(row.get("reportedEPS")),
                        "estimated_eps": _to_float(row.get("estimatedEPS")),
                        "surprise_pct": _to_float(row.get("surprisePercentage")),
                        "revenue": None,
                        "pe_ratio": None,
                        "source": self.name,
                    }
                )

            frame = _normalize_event_frame(pd.DataFrame(records))
            return FetchOutcome(self.name, data=frame)
        except Exception as exc:
            error = str(exc)
            if looks_rate_limited(error):
                self.keys.cool_down(api_key)
                return FetchOutcome(self.name, status="rate_limited", error=error)
            return FetchOutcome(self.name, status="error", error=error)


class EODHDEarningsFetcher:
    name = "eodhd"

    def __init__(self):
        self.keys = APIKeyPool(
            parse_api_keys("EODHD_API_KEYS", "EODHD_API_KEY"),
            default_cooldown=float(os.getenv("EODHD_EARNINGS_COOLDOWN_SECONDS", "120")),
        )
        self._min_call_interval = float(os.getenv("EODHD_EARNINGS_MIN_CALL_INTERVAL", "0.35"))
        self._last_call_by_key: Dict[str, float] = {}

    def supports(self, symbols: List[str]) -> bool:
        return any(to_eodhd_symbol(symbol) for symbol in symbols)

    def _throttle(self, api_key: str):
        elapsed = time.time() - self._last_call_by_key.get(api_key, 0.0)
        if elapsed < self._min_call_interval:
            time.sleep(self._min_call_interval - elapsed)
        self._last_call_by_key[api_key] = time.time()

    def fetch(self, symbols: List[str], from_date: date, to_date: date) -> FetchOutcome:
        api_key = self.keys.acquire()
        if not api_key:
            return FetchOutcome(self.name, status="unavailable", error="No EODHD key available")

        symbol_map = {
            converted.upper(): original
            for original in symbols
            for converted in [to_eodhd_symbol(original)]
            if converted
        }
        converted_symbols = list(symbol_map.keys())
        if not converted_symbols:
            return FetchOutcome(self.name, data=_normalize_event_frame(pd.DataFrame()))
        if EARNINGS_EODHD_MAX_SYMBOLS > 0:
            converted_symbols = converted_symbols[:EARNINGS_EODHD_MAX_SYMBOLS]

        self._throttle(api_key)
        params = {
            "api_token": api_key,
            "fmt": "json",
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
            "symbols": ",".join(converted_symbols),
        }

        try:
            resp = requests.get(EODHD_EARNINGS_BASE, params=params, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
            if isinstance(payload, dict):
                error = payload.get("message") or payload.get("error") or "EODHD returned no earnings rows"
                if looks_rate_limited(error):
                    self.keys.cool_down(api_key)
                    return FetchOutcome(self.name, status="rate_limited", error=error)
                return FetchOutcome(self.name, status="empty", error=error)

            raw_rows = []
            if isinstance(payload, list):
                for item in payload:
                    if isinstance(item, list):
                        raw_rows.extend(item)
                    elif isinstance(item, dict):
                        raw_rows.append(item)

            records = []
            for row in raw_rows:
                if not isinstance(row, dict):
                    continue
                code = str(row.get("code") or row.get("symbol") or "").upper().strip()
                original_symbol = symbol_map.get(code)
                if not original_symbol:
                    continue
                reported_date = row.get("report_date") or row.get("date") or row.get("reportDate")
                if not reported_date:
                    continue
                records.append(
                    {
                        "symbol": original_symbol,
                        "fiscal_date": row.get("date") or row.get("fiscalDateEnding") or reported_date,
                        "reported_date": reported_date,
                        "reported_eps": _to_float(row.get("epsActual", row.get("eps_actual"))),
                        "estimated_eps": _to_float(row.get("epsEstimate", row.get("eps_estimate"))),
                        "surprise_pct": _to_float(row.get("surprisePercent", row.get("surprise_percent"))),
                        "revenue": _to_float(row.get("revenueActual", row.get("revenue_actual", row.get("revenue")))),
                        "pe_ratio": _to_float(row.get("pe", row.get("pe_ratio"))),
                        "source": self.name,
                    }
                )

            frame = _normalize_event_frame(pd.DataFrame(records))
            if frame.empty:
                return FetchOutcome(self.name, status="empty", error="EODHD returned no matching earnings events")
            return FetchOutcome(self.name, data=frame)
        except Exception as exc:
            error = str(exc)
            if looks_rate_limited(error):
                self.keys.cool_down(api_key)
                return FetchOutcome(self.name, status="rate_limited", error=error)
            return FetchOutcome(self.name, status="error", error=error)


class MultiSourceEarningsFetcher:
    def __init__(self):
        registry = {
            "finnhub": FinnhubEarningsCalendarFetcher(),
            "alpha_vantage": AlphaVantageEarningsFetcher(),
            "eodhd": EODHDEarningsFetcher(),
        }
        ordered_names = [name for name in EARNINGS_PROVIDER_ORDER if name in registry]
        if "finnhub" not in ordered_names:
            ordered_names.append("finnhub")
        if "alpha_vantage" not in ordered_names:
            ordered_names.append("alpha_vantage")
        if "eodhd" not in ordered_names:
            ordered_names.append("eodhd")
        self.providers = [registry[name] for name in ordered_names]

    def fetch(self, symbols: List[str], from_date: date, to_date: date) -> FetchOutcome:
        remaining = list(dict.fromkeys(symbols))
        provider_by_symbol: Dict[str, str] = {}
        frames: List[pd.DataFrame] = []
        errors: List[str] = []
        attempts = 0

        for provider in self.providers:
            if attempts >= EARNINGS_MAX_PROVIDER_ATTEMPTS or not remaining:
                break

            if provider.name == "finnhub":
                if not provider.supports(remaining):
                    continue
                outcome = provider.fetch(remaining, from_date, to_date)
                attempts += 1
                if outcome.ok and outcome.data is not None and not outcome.data.empty:
                    frame = outcome.data
                    frames.append(frame)
                    covered = set(frame["symbol"].unique())
                    for symbol in covered:
                        provider_by_symbol[symbol] = provider.name
                    remaining = [symbol for symbol in remaining if symbol not in covered]
                elif outcome.error:
                    errors.append(f"{provider.name}: {outcome.error}")
                continue

            if provider.name == "eodhd":
                if not provider.supports(remaining):
                    continue
                outcome = provider.fetch(remaining, from_date, to_date)
                attempts += 1
                if outcome.ok and outcome.data is not None and not outcome.data.empty:
                    frame = outcome.data
                    frames.append(frame)
                    covered = set(frame["symbol"].unique())
                    for symbol in covered:
                        provider_by_symbol[symbol] = provider.name
                    remaining = [symbol for symbol in remaining if symbol not in covered]
                elif outcome.error:
                    errors.append(f"{provider.name}: {outcome.error}")
                continue

            fetched = 0
            next_remaining = []
            for symbol in remaining:
                if not provider.supports(symbol):
                    next_remaining.append(symbol)
                    continue
                if provider.name == "alpha_vantage" and fetched >= EARNINGS_ALPHA_MAX_SYMBOLS:
                    next_remaining.append(symbol)
                    continue
                outcome = provider.fetch_symbol(symbol, from_date, to_date)
                fetched += 1
                if outcome.ok and outcome.data is not None and not outcome.data.empty:
                    frames.append(outcome.data)
                    provider_by_symbol[symbol] = provider.name
                else:
                    next_remaining.append(symbol)
                    if outcome.error:
                        errors.append(f"{provider.name}:{symbol}: {outcome.error}")
            attempts += 1
            remaining = next_remaining

        merged = _normalize_event_frame(pd.concat(frames, ignore_index=True)) if frames else _normalize_event_frame(pd.DataFrame())
        if merged.empty:
            return FetchOutcome(
                provider=",".join(EARNINGS_PROVIDER_ORDER),
                data=merged,
                status="empty",
                error=" | ".join(errors[:6]),
                meta={"provider_by_symbol": provider_by_symbol, "missing_symbols": remaining},
            )

        return FetchOutcome(
            provider=",".join(sorted(set(provider_by_symbol.values()))),
            data=merged,
            meta={"provider_by_symbol": provider_by_symbol, "missing_symbols": remaining, "errors": errors},
        )


class EarningsEventPipeline:
    def __init__(self, pit_db=None):
        self.fetcher = MultiSourceEarningsFetcher()
        self.pit_db = pit_db

    def run(self, symbols: List[str], save: bool = True) -> Dict:
        logger.info(
            f"EarningsEventPipeline: {len(symbols)} symbols, last {EARNINGS_LOOKBACK_DAYS}d, next {EARNINGS_FORWARD_DAYS}d"
        )
        from_date = date.today() - timedelta(days=EARNINGS_LOOKBACK_DAYS)
        to_date = date.today() + timedelta(days=EARNINGS_FORWARD_DAYS)

        outcome = self.fetcher.fetch(symbols, from_date, to_date)
        frame = outcome.data if outcome.data is not None else _normalize_event_frame(pd.DataFrame())
        past_events = frame[frame["reported_date"].dt.date <= date.today()].copy() if not frame.empty else frame.copy()
        upcoming_events = frame[frame["reported_date"].dt.date > date.today()].copy() if not frame.empty else frame.copy()

        inserted = 0
        if save and self.pit_db is not None and not past_events.empty:
            try:
                inserted = int(self.pit_db.insert_fundamentals(past_events))
            except Exception as exc:
                logger.warning(f"Earnings PIT insert failed: {exc}")

        latest_by_symbol = {}
        if not past_events.empty:
            latest = past_events.sort_values("reported_date").groupby("symbol").tail(1)
            latest_by_symbol = latest.set_index("symbol").to_dict("index")

        upcoming_by_symbol = {}
        if not upcoming_events.empty:
            upcoming = upcoming_events.sort_values("reported_date").groupby("symbol").head(1)
            upcoming_by_symbol = upcoming.set_index("symbol").to_dict("index")

        logger.info(
            f"EarningsEventPipeline complete: {past_events['symbol'].nunique() if not past_events.empty else 0} symbols with recent events"
        )
        return {
            "earnings_events": past_events,
            "upcoming_earnings_events": upcoming_events,
            "latest_earnings_by_symbol": latest_by_symbol,
            "upcoming_earnings_by_symbol": upcoming_by_symbol,
            "earnings_provider_by_symbol": outcome.meta.get("provider_by_symbol", {}),
            "pit_rows_inserted": inserted,
            "run_timestamp": datetime.utcnow().isoformat(),
            "fetch_error": outcome.error,
        }
