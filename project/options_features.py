#!/usr/bin/env python3
"""
Options Microstructure Features — GEX, Vanna, Charm, Pinning
=============================================================
Computes dealer-hedging microstructure features that drive intraday flow:

  net_gex            - Net Dollar Gamma Exposure across all strikes
  total_call_gex     - Call-side gamma exposure
  total_put_gex      - Put-side gamma exposure
  gex_regime         - "dampening" / "expanding" / "neutral"
  put_call_ratio     - Put OI / Call OI
  net_vanna          - Aggregate dealer dVega/dSpot (IV shock -> delta shock)
  net_charm          - Aggregate dealer dDelta/dTime (time decay of delta)
  charm_direction    - "call_decay" / "put_decay" / "neutral"
  pin_strike         - Strike with max |GEX| (magnet strike)
  distance_to_pin    - (spot - pin_strike) / spot

SOFT GATE: If options chain unavailable, all features default to safe
neutrals (0.0 / "neutral") and caller should penalize confidence score
by 0.08. This module NEVER raises — it logs and returns neutral dict.

Data source priority:
  1. Upstox options chain (India lane) — via analytics token
  2. Finnhub options endpoint (US lane)
  3. yfinance fallback (both lanes)

Cache: features recomputed every 5 minutes per symbol; reads served
from in-memory cache between refreshes to keep inference loop cheap.

Standalone test:
    python options_features.py --symbol AAPL
    python options_features.py --symbol RELIANCE --india
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants & cache
# -----------------------------------------------------------------------------

DEFAULT_CONTRACT_SIZE_US = 100
DEFAULT_CONTRACT_SIZE_NSE = 1  # overridden per-symbol via lot_size param
CACHE_TTL_SECONDS = int(os.getenv("OPTIONS_FEATURES_CACHE_TTL", "300"))
HTTP_TIMEOUT_SECONDS = float(os.getenv("OPTIONS_FEATURES_HTTP_TIMEOUT", "5.0"))

# GEX regime thresholds are expressed relative to |net_gex| in raw dollar units.
# Very small |net_gex| == neutral dealer positioning.
GEX_NEUTRAL_BAND = float(os.getenv("OPTIONS_FEATURES_GEX_NEUTRAL_BAND", "5e6"))

NEUTRAL_FEATURES: Dict[str, Any] = {
    "net_gex": 0.0,
    "total_call_gex": 0.0,
    "total_put_gex": 0.0,
    "gex_regime": "neutral",
    "put_call_ratio": 1.0,
    "net_vanna": 0.0,
    "net_charm": 0.0,
    "charm_direction": "neutral",
    "pin_strike": 0.0,
    "distance_to_pin": 0.0,
    "options_data_available": 0,
}


@dataclass
class _CacheEntry:
    features: Dict[str, Any]
    computed_at: float
    prev_delta_by_strike: Dict[Tuple[str, float], float] = field(default_factory=dict)


_CACHE: Dict[str, _CacheEntry] = {}
_CACHE_LOCK = threading.Lock()


# -----------------------------------------------------------------------------
# Black-Scholes Greeks (finite-difference Vanna/Charm per spec)
# -----------------------------------------------------------------------------


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _bs_d1(S: float, K: float, tau: float, sigma: float, r: float = 0.05, q: float = 0.0) -> float:
    if S <= 0 or K <= 0 or tau <= 0 or sigma <= 0:
        return 0.0
    return (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * tau) / (sigma * math.sqrt(tau))


def _bs_delta(S: float, K: float, tau: float, sigma: float, is_call: bool, r: float = 0.05, q: float = 0.0) -> float:
    if S <= 0 or K <= 0 or tau <= 0 or sigma <= 0:
        return 0.0
    d1 = _bs_d1(S, K, tau, sigma, r, q)
    if is_call:
        return math.exp(-q * tau) * _norm_cdf(d1)
    return math.exp(-q * tau) * (_norm_cdf(d1) - 1.0)


def _bs_gamma(S: float, K: float, tau: float, sigma: float, r: float = 0.05, q: float = 0.0) -> float:
    if S <= 0 or K <= 0 or tau <= 0 or sigma <= 0:
        return 0.0
    d1 = _bs_d1(S, K, tau, sigma, r, q)
    return math.exp(-q * tau) * _norm_pdf(d1) / (S * sigma * math.sqrt(tau))


def _finite_diff_vanna(S: float, K: float, tau: float, sigma: float, is_call: bool) -> float:
    """Vanna ~= (Delta(sigma+0.01) - Delta(sigma-0.01)) / 0.02 per spec."""
    sigma_up = sigma + 0.01
    sigma_down = max(sigma - 0.01, 1e-4)
    d_up = _bs_delta(S, K, tau, sigma_up, is_call)
    d_dn = _bs_delta(S, K, tau, sigma_down, is_call)
    return (d_up - d_dn) / (sigma_up - sigma_down)


def _finite_diff_charm(
    S: float,
    K: float,
    tau: float,
    sigma: float,
    is_call: bool,
    prev_delta: Optional[float],
) -> float:
    """
    Charm ~= -(Delta(t) - Delta(t-1min)) / (1/525600) per spec.
    If no prior delta available, fall back to closed-form charm:
      Charm_call = q*e^(-q*tau)*N(d1) - e^(-q*tau)*n(d1)*[2(r-q)tau - d2*sigma*sqrt(tau)]/(2*tau*sigma*sqrt(tau))
    """
    current_delta = _bs_delta(S, K, tau, sigma, is_call)
    if prev_delta is not None:
        dt_years = 1.0 / 525600.0  # 1 minute in years
        return -(current_delta - prev_delta) / dt_years

    # Closed-form fallback (per-year charm)
    if S <= 0 or K <= 0 or tau <= 0 or sigma <= 0:
        return 0.0
    r = 0.05
    q = 0.0
    d1 = _bs_d1(S, K, tau, sigma, r, q)
    d2 = d1 - sigma * math.sqrt(tau)
    pdf = _norm_pdf(d1)
    term1 = q * math.exp(-q * tau) * (_norm_cdf(d1) if is_call else _norm_cdf(d1) - 1.0)
    term2 = math.exp(-q * tau) * pdf * (2.0 * (r - q) * tau - d2 * sigma * math.sqrt(tau)) / (2.0 * tau * sigma * math.sqrt(tau))
    return term1 - term2


# -----------------------------------------------------------------------------
# Options chain data source loaders
# -----------------------------------------------------------------------------


def _years_to_expiry(expiry_str: str, now: Optional[datetime] = None) -> float:
    now = now or datetime.now(timezone.utc)
    try:
        expiry = datetime.strptime(expiry_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return 0.0
    seconds = (expiry - now).total_seconds()
    if seconds <= 0:
        return 0.0
    return seconds / (365.25 * 24 * 3600)


def _fetch_yfinance_chain(symbol: str) -> Tuple[float, List[Dict[str, Any]]]:
    """Returns (spot, [contracts]). Each contract: strike, is_call, oi, iv, tau."""
    try:
        import yfinance as yf
    except ImportError:
        logger.debug("yfinance not installed; skipping yf fallback")
        return 0.0, []

    try:
        tkr = yf.Ticker(symbol)
        info = tkr.fast_info
        spot = float(info.get("last_price") or info.get("lastPrice") or 0.0)
        if spot <= 0:
            hist = tkr.history(period="1d", interval="1m")
            if not hist.empty:
                spot = float(hist["Close"].iloc[-1])
        expiries = tkr.options or []
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("yfinance fetch failed for %s: %s", symbol, exc)
        return 0.0, []

    contracts: List[Dict[str, Any]] = []
    # Limit to nearest 3 expiries to keep latency bounded
    for expiry in expiries[:3]:
        try:
            chain = tkr.option_chain(expiry)
        except Exception as exc:  # pragma: no cover
            logger.debug("option_chain failed for %s %s: %s", symbol, expiry, exc)
            continue
        tau = _years_to_expiry(expiry)
        if tau <= 0:
            continue
        for df, is_call in ((chain.calls, True), (chain.puts, False)):
            if df is None or df.empty:
                continue
            for _, row in df.iterrows():
                try:
                    strike = float(row.get("strike") or 0.0)
                    oi = float(row.get("openInterest") or 0.0)
                    iv = float(row.get("impliedVolatility") or 0.0)
                except (TypeError, ValueError):
                    continue
                if strike <= 0 or oi <= 0:
                    continue
                iv = max(min(iv, 5.0), 0.01)  # clamp absurd IVs
                contracts.append({
                    "strike": strike,
                    "is_call": is_call,
                    "oi": oi,
                    "iv": iv,
                    "tau": tau,
                    "expiry": expiry,
                })
    return spot, contracts


async def _fetch_upstox_chain(symbol: str, access_token: str, instrument_key: str) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Upstox V3 options chain endpoint. Uses analytics token for data-only access.
    Endpoint: https://api.upstox.com/v2/option/chain
    """
    try:
        import aiohttp
    except ImportError:
        logger.debug("aiohttp not installed; skipping upstox fetch")
        return 0.0, []

    if not access_token or not instrument_key:
        return 0.0, []

    url = "https://api.upstox.com/v2/option/chain"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
    }
    # next monthly expiry, best-effort
    next_expiry = (date.today() + timedelta(days=30)).isoformat()
    params = {"instrument_key": instrument_key, "expiry_date": next_expiry}

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT_SECONDS)) as session:
            async with session.get(url, headers=headers, params=params) as resp:
                if resp.status != 200:
                    logger.warning("Upstox option chain HTTP %s for %s", resp.status, symbol)
                    return 0.0, []
                payload = await resp.json()
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("Upstox option chain fetch failed for %s: %s", symbol, exc)
        return 0.0, []

    data = payload.get("data") or []
    if not data:
        return 0.0, []

    spot = 0.0
    contracts: List[Dict[str, Any]] = []
    for node in data:
        try:
            spot = float(node.get("underlying_spot_price") or spot or 0.0)
            strike = float(node.get("strike_price") or 0.0)
            expiry = node.get("expiry") or next_expiry
            tau = _years_to_expiry(expiry)
            if tau <= 0 or strike <= 0:
                continue
            for side_key, is_call in (("call_options", True), ("put_options", False)):
                side = node.get(side_key) or {}
                mkt = side.get("market_data") or {}
                og = side.get("option_greeks") or {}
                oi = float(mkt.get("oi") or 0.0)
                iv = float(og.get("iv") or mkt.get("iv") or 0.0)
                if oi <= 0:
                    continue
                iv = max(min(iv / 100.0 if iv > 5 else iv, 5.0), 0.01)
                contracts.append({
                    "strike": strike,
                    "is_call": is_call,
                    "oi": oi,
                    "iv": iv,
                    "tau": tau,
                    "expiry": expiry,
                })
        except (TypeError, ValueError):
            continue
    return spot, contracts


async def _fetch_finnhub_chain(symbol: str, api_key: str) -> Tuple[float, List[Dict[str, Any]]]:
    try:
        import aiohttp
    except ImportError:
        return 0.0, []
    if not api_key:
        return 0.0, []

    url = "https://finnhub.io/api/v1/stock/option-chain"
    params = {"symbol": symbol, "token": api_key}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT_SECONDS)) as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return 0.0, []
                payload = await resp.json()
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("Finnhub option chain failed for %s: %s", symbol, exc)
        return 0.0, []

    data = payload.get("data") or []
    if not data:
        return 0.0, []

    # Finnhub does not provide spot in options endpoint; fetch quote separately
    spot = 0.0
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT_SECONDS)) as session:
            async with session.get(
                "https://finnhub.io/api/v1/quote",
                params={"symbol": symbol, "token": api_key},
            ) as resp:
                if resp.status == 200:
                    q = await resp.json()
                    spot = float(q.get("c") or 0.0)
    except Exception:
        pass

    contracts: List[Dict[str, Any]] = []
    for expiry_bucket in data[:3]:
        expiry = expiry_bucket.get("expirationDate") or ""
        tau = _years_to_expiry(expiry)
        if tau <= 0:
            continue
        options = expiry_bucket.get("options") or {}
        for side_key, is_call in (("CALL", True), ("PUT", False)):
            for row in options.get(side_key) or []:
                try:
                    strike = float(row.get("strike") or 0.0)
                    oi = float(row.get("openInterest") or 0.0)
                    iv = float(row.get("impliedVolatility") or 0.0)
                except (TypeError, ValueError):
                    continue
                if strike <= 0 or oi <= 0:
                    continue
                iv = max(min(iv, 5.0), 0.01)
                contracts.append({
                    "strike": strike,
                    "is_call": is_call,
                    "oi": oi,
                    "iv": iv,
                    "tau": tau,
                    "expiry": expiry,
                })
    return spot, contracts


# -----------------------------------------------------------------------------
# Core computation
# -----------------------------------------------------------------------------


def _compute_features_from_chain(
    symbol: str,
    spot: float,
    contracts: List[Dict[str, Any]],
    contract_size: int,
    prev_delta_by_strike: Dict[Tuple[str, float], float],
) -> Tuple[Dict[str, Any], Dict[Tuple[str, float], float]]:
    """Pure function: chain -> features. Also returns next prev_delta map."""
    if spot <= 0 or not contracts:
        return dict(NEUTRAL_FEATURES), {}

    total_call_gex = 0.0
    total_put_gex = 0.0
    total_call_oi = 0.0
    total_put_oi = 0.0
    net_vanna = 0.0
    net_charm = 0.0
    charm_call = 0.0
    charm_put = 0.0
    gex_per_strike: Dict[float, float] = {}
    next_prev: Dict[Tuple[str, float], float] = {}

    spot_sq = spot * spot

    for c in contracts:
        K = float(c["strike"])
        oi = float(c["oi"])
        iv = float(c["iv"])
        tau = float(c["tau"])
        is_call = bool(c["is_call"])
        expiry = str(c.get("expiry") or "")

        gamma = _bs_gamma(spot, K, tau, iv)
        gex = gamma * contract_size * oi * spot_sq * 0.01
        # Dealer convention: dealers are typically short calls, long puts from
        # retail buying. GEX sign convention below follows the spec literally
        # (total_call_gex - total_put_gex), the regime interpretation treats
        # positive net as "dampening" (long-gamma) regardless.
        if is_call:
            total_call_gex += gex
            total_call_oi += oi
        else:
            total_put_gex += gex
            total_put_oi += oi
        gex_per_strike[K] = gex_per_strike.get(K, 0.0) + abs(gex)

        vanna_k = _finite_diff_vanna(spot, K, tau, iv, is_call)
        net_vanna += vanna_k * oi

        key = (expiry, K) if expiry else ("", K)
        current_delta = _bs_delta(spot, K, tau, iv, is_call)
        prev = prev_delta_by_strike.get(key)
        charm_k = _finite_diff_charm(spot, K, tau, iv, is_call, prev)
        next_prev[key] = current_delta
        contribution = charm_k * oi
        net_charm += contribution
        if is_call:
            charm_call += contribution
        else:
            charm_put += contribution

    net_gex = total_call_gex - total_put_gex

    if abs(net_gex) < GEX_NEUTRAL_BAND:
        gex_regime = "neutral"
    elif net_gex > 0:
        gex_regime = "dampening"
    else:
        gex_regime = "expanding"

    put_call_ratio = (total_put_oi / total_call_oi) if total_call_oi > 0 else 1.0

    if charm_call > 0 and charm_call > abs(charm_put):
        charm_direction = "call_decay"
    elif charm_put < 0 and abs(charm_put) > charm_call:
        charm_direction = "put_decay"
    else:
        charm_direction = "neutral"

    if gex_per_strike:
        pin_strike = max(gex_per_strike.items(), key=lambda kv: kv[1])[0]
    else:
        pin_strike = 0.0
    distance_to_pin = (spot - pin_strike) / spot if (spot > 0 and pin_strike > 0) else 0.0

    features = {
        "net_gex": float(net_gex),
        "total_call_gex": float(total_call_gex),
        "total_put_gex": float(total_put_gex),
        "gex_regime": gex_regime,
        "put_call_ratio": float(put_call_ratio),
        "net_vanna": float(net_vanna),
        "net_charm": float(net_charm),
        "charm_direction": charm_direction,
        "pin_strike": float(pin_strike),
        "distance_to_pin": float(distance_to_pin),
        "options_data_available": 1,
    }
    return features, next_prev


async def _fetch_polygon_chain(symbol: str, api_key: str) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Polygon.io options snapshot — free tier (5 req/min), delayed 15min.
    Endpoint: GET /v3/snapshot/options/{underlyingAsset}
    Returns up to 250 options per call; paginated via next_url.
    """
    try:
        import aiohttp
    except ImportError:
        logger.debug("aiohttp not installed; skipping Polygon options fetch")
        return 0.0, []

    if not api_key or not symbol:
        return 0.0, []

    # Polygon uses plain ticker (no .NS suffix)
    ticker = symbol.upper().replace(".NS", "").replace(".BO", "")
    url = f"https://api.polygon.io/v3/snapshot/options/{ticker}"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"limit": "250", "expired": "false"}

    spot = 0.0
    contracts: List[Dict[str, Any]] = []

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT_SECONDS)) as session:
            # First page
            async with session.get(url, headers=headers, params=params) as resp:
                if resp.status == 429:
                    logger.warning("Polygon options rate limited for %s", symbol)
                    return 0.0, []
                if resp.status != 200:
                    logger.debug("Polygon options HTTP %s for %s", resp.status, symbol)
                    return 0.0, []
                payload = await resp.json()

            results = payload.get("results") or []
            next_url = payload.get("next_url")

            # Pull spot from underlying snapshot
            try:
                und_url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
                async with session.get(und_url, headers=headers) as uresp:
                    if uresp.status == 200:
                        udata = await uresp.json()
                        spot = float(
                            (udata.get("ticker") or {}).get("day", {}).get("c")
                            or (udata.get("ticker") or {}).get("lastTrade", {}).get("p")
                            or 0.0
                        )
            except Exception:
                pass

            # Optionally fetch a second page (up to 500 contracts total)
            if next_url and len(results) < 500:
                try:
                    async with session.get(next_url, headers=headers) as p2:
                        if p2.status == 200:
                            p2data = await p2.json()
                            results.extend(p2data.get("results") or [])
                except Exception:
                    pass

        for item in results:
            try:
                details = item.get("details") or {}
                greeks = item.get("greeks") or {}
                day = item.get("day") or {}
                oi = float(item.get("open_interest") or 0.0)
                iv = float(item.get("implied_volatility") or 0.0)
                strike = float(details.get("strike_price") or 0.0)
                exp_str = details.get("expiration_date") or ""
                contract_type = (details.get("contract_type") or "").lower()
                is_call = contract_type == "call"
                tau = _years_to_expiry(exp_str)
                if oi <= 0 or strike <= 0 or tau <= 0:
                    continue
                iv = max(min(float(iv), 5.0), 0.01)
                contracts.append({
                    "strike": strike,
                    "is_call": is_call,
                    "oi": oi,
                    "iv": iv,
                    "tau": tau,
                    "expiry": exp_str,
                })
            except (TypeError, ValueError, KeyError):
                continue

    except Exception as exc:
        logger.warning("Polygon options fetch failed for %s: %s", symbol, exc)
        return 0.0, []

    return spot, contracts


async def compute_options_features(
    symbol: str,
    *,
    is_india: bool = False,
    lot_size: Optional[int] = None,
    upstox_token: Optional[str] = None,
    upstox_instrument_key: Optional[str] = None,
    finnhub_api_key: Optional[str] = None,
    polygon_api_key: Optional[str] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Main entry point. Returns feature dict; NEVER raises.
    Caller should check `options_data_available` and apply 0.08 confidence
    penalty if zero.
    """
    cache_key = f"{symbol}|{'IN' if is_india else 'US'}"
    now = time.time()

    if use_cache:
        with _CACHE_LOCK:
            entry = _CACHE.get(cache_key)
            if entry and (now - entry.computed_at) < CACHE_TTL_SECONDS:
                return dict(entry.features)

    contract_size = lot_size if (is_india and lot_size) else (
        DEFAULT_CONTRACT_SIZE_NSE if is_india else DEFAULT_CONTRACT_SIZE_US
    )

    spot = 0.0
    contracts: List[Dict[str, Any]] = []

    # Resolve Polygon key from arg or env
    _polygon_key = polygon_api_key or os.getenv("POLYGON_API_KEY") or (os.getenv("POLYGON_API_KEYS") or "").split(",")[0].strip()

    # Data source priority:
    #   India  : Upstox → yfinance
    #   US     : Finnhub → Polygon → yfinance
    if is_india and upstox_token and upstox_instrument_key:
        try:
            spot, contracts = await _fetch_upstox_chain(symbol, upstox_token, upstox_instrument_key)
        except Exception as exc:  # pragma: no cover
            logger.warning("Upstox fetch exception for %s: %s", symbol, exc)

    if not contracts and not is_india and finnhub_api_key:
        try:
            spot, contracts = await _fetch_finnhub_chain(symbol, finnhub_api_key)
        except Exception as exc:  # pragma: no cover
            logger.warning("Finnhub fetch exception for %s: %s", symbol, exc)

    # Polygon: richer chain data than yfinance, still free (delayed 15min)
    if not contracts and not is_india and _polygon_key:
        try:
            spot, contracts = await _fetch_polygon_chain(symbol, _polygon_key)
        except Exception as exc:  # pragma: no cover
            logger.warning("Polygon fetch exception for %s: %s", symbol, exc)

    if not contracts:
        # yfinance last-resort fallback (offloaded to thread since yfinance is blocking)
        try:
            spot, contracts = await asyncio.get_event_loop().run_in_executor(
                None, _fetch_yfinance_chain, symbol
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("yfinance fallback failed for %s: %s", symbol, exc)

    if not contracts or spot <= 0:
        logger.info("No options chain available for %s; returning neutral features", symbol)
        neutral = dict(NEUTRAL_FEATURES)
        if use_cache:
            with _CACHE_LOCK:
                _CACHE[cache_key] = _CacheEntry(features=neutral, computed_at=now)
        return neutral

    with _CACHE_LOCK:
        prev_entry = _CACHE.get(cache_key)
        prev_delta_map = dict(prev_entry.prev_delta_by_strike) if prev_entry else {}

    features, next_prev = _compute_features_from_chain(
        symbol, spot, contracts, contract_size, prev_delta_map
    )

    if use_cache:
        with _CACHE_LOCK:
            _CACHE[cache_key] = _CacheEntry(
                features=features,
                computed_at=now,
                prev_delta_by_strike=next_prev,
            )

    return features


def compute_options_features_sync(symbol: str, **kwargs: Any) -> Dict[str, Any]:
    """Blocking wrapper around compute_options_features for sync call sites."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Should not happen from sync code path, but guard anyway.
            return dict(NEUTRAL_FEATURES)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(compute_options_features(symbol, **kwargs))


# -----------------------------------------------------------------------------
# __main__ — standalone test harness
# -----------------------------------------------------------------------------


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Options microstructure feature test harness")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--india", action="store_true", help="Use NSE / Upstox path")
    parser.add_argument("--lot-size", type=int, default=None)
    parser.add_argument("--upstox-token", default=os.getenv("UPSTOX_ANALYTICS_TOKEN") or os.getenv("UPSTOX_ACCESS_TOKEN"))
    parser.add_argument("--upstox-instrument-key", default=None)
    parser.add_argument("--finnhub-key", default=os.getenv("FINNHUB_API_KEY") or (os.getenv("FINNHUB_API_KEYS") or "").split(",")[0])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    features = compute_options_features_sync(
        args.symbol,
        is_india=args.india,
        lot_size=args.lot_size,
        upstox_token=args.upstox_token or None,
        upstox_instrument_key=args.upstox_instrument_key,
        finnhub_api_key=args.finnhub_key or None,
    )
    print(f"\nOptions features for {args.symbol}:")
    for k, v in features.items():
        print(f"  {k:22s} = {v}")


if __name__ == "__main__":
    _cli()
