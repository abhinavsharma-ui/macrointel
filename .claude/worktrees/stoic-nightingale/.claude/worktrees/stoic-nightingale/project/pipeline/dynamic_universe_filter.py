"""
Dynamic Universe Filter — Daily Hot-List Generator
====================================================
Narrows the full 2600-symbol universe to ~200 actionable names
based on liquidity, anomaly scoring, and sector momentum.

Runs once daily (market open) or on demand.  Caches to
``data/hot_list.json`` so repeated calls within the cache window
(default 4 hours) are instant.

Usage:
    from pipeline.dynamic_universe_filter import DynamicUniverseFilter

    duf = DynamicUniverseFilter()
    hot = duf.get_hot_list()            # cached if fresh
    hot = duf.refresh(all_symbols)      # force full recompute
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip() or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip() or default)
    except ValueError:
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _market_code(symbol: str) -> str:
    s = symbol.upper()
    if s.endswith(".NS"):
        return "NSE"
    if s.endswith(("USDT", "USDC", "BUSD")):
        return "CRYPTO"
    return "US"


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

class _Config:
    """All tunables in one place, env-overridable."""

    def __init__(self):
        self.enabled = _env_flag("DYNAMIC_UNIVERSE_ENABLED", True)
        self.top_k = _env_int("DYNAMIC_UNIVERSE_TOP_K", 200)
        self.min_volume_usd = _env_float("DYNAMIC_UNIVERSE_MIN_VOLUME_USD", 5_000_000)
        self.min_volume_inr = _env_float("DYNAMIC_UNIVERSE_MIN_VOLUME_INR", 100_000_000)
        self.min_volume_crypto_usd = _env_float("DYNAMIC_UNIVERSE_MIN_VOLUME_CRYPTO_USD", 50_000_000)
        self.min_price = _env_float("DYNAMIC_UNIVERSE_MIN_PRICE", 2.0)
        self.min_history_days = 60
        self.earnings_blackout_days = 2
        self.cache_hours = _env_float("DYNAMIC_UNIVERSE_CACHE_HOURS", 4.0)
        self.hot_list_path = DATA_DIR / "hot_list.json"

        # Tier-2 scoring weights
        self.w_vol_anomaly = 0.30
        self.w_options_anomaly = 0.20
        self.w_sector_momentum = 0.25
        self.w_tech_proximity = 0.15
        self.w_sentiment = 0.10


# ─────────────────────────────────────────────────────────────
# Dynamic Universe Filter
# ─────────────────────────────────────────────────────────────

class DynamicUniverseFilter:
    """
    Two-tier daily filter: hard liquidity gates → anomaly ranking.

    Thread-safe: stateless scoring + atomic JSON writes.
    """

    def __init__(self):
        self.cfg = _Config()
        self._last_refresh_ts: float = 0.0
        self._cached_hot_list: List[Dict[str, Any]] = []

    # ── public API ───────────────────────────────────────────

    def get_hot_list(self) -> List[Dict[str, Any]]:
        """
        Return cached hot list if fresh (< cache_hours old),
        otherwise load from disk.  Returns list of dicts:
        ``[{"symbol": "AAPL", "score": 1.82, "rank": 1, ...}, ...]``
        """
        if not self.cfg.enabled:
            return []

        # In-memory cache
        age_hours = (time.time() - self._last_refresh_ts) / 3600
        if self._cached_hot_list and age_hours < self.cfg.cache_hours:
            return self._cached_hot_list

        # Disk cache
        loaded = self._load_from_disk()
        if loaded is not None:
            self._cached_hot_list = loaded
            self._last_refresh_ts = time.time()
            return loaded

        return []

    def get_hot_symbols(self) -> List[str]:
        """Convenience: just the symbol list."""
        return [entry["symbol"] for entry in self.get_hot_list()]

    def refresh(
        self,
        all_symbols: List[str],
        price_data: Optional[Dict[str, pd.DataFrame]] = None,
        earnings_data: Optional[Dict] = None,
        sentiment_data: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Full recompute: Tier 1 → Tier 2 → top_k → persist.

        Parameters
        ----------
        all_symbols : list
            Full universe (2600 symbols).
        price_data : dict
            ``{symbol: DataFrame}`` with OHLCV columns.
            If None, attempts lazy load via PriceDataPipeline.
        earnings_data : dict
            Output of ``EarningsEventPipeline.run()``.
        sentiment_data
            Sentiment DataFrame or dict from SentimentPipeline.

        Returns
        -------
        Hot list: list of dicts sorted by score descending.
        """
        if not self.cfg.enabled:
            logger.info("DynamicUniverseFilter: disabled via DYNAMIC_UNIVERSE_ENABLED=0")
            return []

        t0 = time.time()
        logger.info(f"DynamicUniverseFilter: starting refresh on {len(all_symbols)} symbols")

        # Lazy-load price data if not provided
        if price_data is None:
            price_data = self._fetch_price_data(all_symbols)

        # Tier 1: hard filters
        survivors = self._tier1_filter(all_symbols, price_data, earnings_data)
        logger.info(f"DynamicUniverseFilter: Tier 1 passed {len(survivors)}/{len(all_symbols)}")

        if not survivors:
            return []

        # Tier 2: anomaly scoring
        scored = self._tier2_score(survivors, price_data, sentiment_data)

        # Sort and take top K
        scored.sort(key=lambda x: x["score"], reverse=True)
        hot_list = scored[: self.cfg.top_k]

        # Assign ranks
        for i, entry in enumerate(hot_list, start=1):
            entry["rank"] = i

        # Persist
        self._persist(hot_list)
        self._cached_hot_list = hot_list
        self._last_refresh_ts = time.time()

        elapsed = time.time() - t0
        logger.info(
            f"DynamicUniverseFilter: refresh done in {elapsed:.1f}s — "
            f"{len(hot_list)} symbols selected (top score={hot_list[0]['score']:.3f})"
        )
        return hot_list

    # ── Tier 1: hard liquidity / quality gates ───────────────

    def _tier1_filter(
        self,
        symbols: List[str],
        price_data: Dict[str, pd.DataFrame],
        earnings_data: Optional[Dict],
    ) -> List[str]:
        """Must pass ALL criteria to survive."""
        blackout_symbols = self._get_earnings_blackout(earnings_data)
        survivors = []

        for symbol in symbols:
            df = price_data.get(symbol)
            if df is None or df.empty:
                continue

            # Minimum history
            if len(df) < self.cfg.min_history_days:
                continue

            # Minimum price (use last close)
            close_col = self._close_col(df)
            if close_col is None:
                continue
            last_close = float(df[close_col].iloc[-1])
            if last_close < self.cfg.min_price:
                continue

            # Volume gate (market-dependent)
            vol_col = self._volume_col(df)
            if vol_col is None:
                continue
            avg_volume_value = self._avg_dollar_volume(df, close_col, vol_col, window=20)
            market = _market_code(symbol)
            min_vol = {
                "US": self.cfg.min_volume_usd,
                "NSE": self.cfg.min_volume_inr,
                "CRYPTO": self.cfg.min_volume_crypto_usd,
            }.get(market, self.cfg.min_volume_usd)

            if avg_volume_value < min_vol:
                continue

            # Earnings blackout
            if symbol in blackout_symbols:
                continue

            survivors.append(symbol)

        return survivors

    # ── Tier 2: anomaly scoring ──────────────────────────────

    def _tier2_score(
        self,
        symbols: List[str],
        price_data: Dict[str, pd.DataFrame],
        sentiment_data: Optional[Any],
    ) -> List[Dict[str, Any]]:
        """Score each symbol on 5 anomaly dimensions."""
        # Pre-compute sector momentum once for all
        sector_mom = self._compute_sector_momentum(symbols, price_data)
        sentiment_map = self._build_sentiment_map(sentiment_data)

        scored = []
        for symbol in symbols:
            df = price_data.get(symbol)
            if df is None or df.empty:
                continue

            close_col = self._close_col(df)
            vol_col = self._volume_col(df)
            if close_col is None or vol_col is None:
                continue

            # 1. Unusual volume: (today_vol / 20d_avg) - 1, capped ±2
            vol_anomaly = self._volume_anomaly(df, vol_col)

            # 2. Options flow anomaly (proxy: put/call volume if available)
            options_anomaly = self._options_anomaly(df)

            # 3. Sector momentum rank
            from pipeline.universe import get_sector
            sector = get_sector(symbol)
            sect_score = sector_mom.get(sector, 0.0)

            # 4. Technical proximity: distance to 52w high/low
            tech_prox = self._technical_proximity(df, close_col)

            # 5. Sentiment: 24h change
            sent_score = sentiment_map.get(symbol, 0.0)

            composite = (
                self.cfg.w_vol_anomaly * vol_anomaly
                + self.cfg.w_options_anomaly * options_anomaly
                + self.cfg.w_sector_momentum * sect_score
                + self.cfg.w_tech_proximity * tech_prox
                + self.cfg.w_sentiment * sent_score
            )

            scored.append({
                "symbol": symbol,
                "score": round(float(composite), 4),
                "market": _market_code(symbol),
                "sector": sector,
                "vol_anomaly": round(float(vol_anomaly), 4),
                "options_anomaly": round(float(options_anomaly), 4),
                "sector_momentum": round(float(sect_score), 4),
                "tech_proximity": round(float(tech_prox), 4),
                "sentiment": round(float(sent_score), 4),
            })

        return scored

    # ── scoring components ───────────────────────────────────

    def _volume_anomaly(self, df: pd.DataFrame, vol_col: str) -> float:
        """(today_volume / 20d_avg_volume) - 1, capped at ±2.0."""
        vol = df[vol_col].values
        if len(vol) < 21:
            return 0.0
        avg_20 = float(np.mean(vol[-21:-1]))
        if avg_20 <= 0:
            return 0.0
        ratio = (float(vol[-1]) / avg_20) - 1.0
        return float(np.clip(ratio, -2.0, 2.0))

    def _options_anomaly(self, df: pd.DataFrame) -> float:
        """
        Options flow anomaly.  If put_volume / call_volume columns exist,
        compute deviation from 20-day median.  Otherwise return 0.
        """
        for put_col, call_col in [
            ("put_volume", "call_volume"),
            ("Put_Volume", "Call_Volume"),
        ]:
            if put_col in df.columns and call_col in df.columns:
                put = df[put_col].values
                call = df[call_col].values
                if len(put) < 21 or len(call) < 21:
                    return 0.0
                ratios = np.where(call > 0, put / call, 1.0)
                median_20 = float(np.median(ratios[-21:-1]))
                if median_20 <= 0:
                    return 0.0
                current = float(ratios[-1])
                anomaly = (current / median_20) - 1.0
                return float(np.clip(anomaly, -2.0, 2.0))
        return 0.0

    def _compute_sector_momentum(
        self,
        symbols: List[str],
        price_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        """
        Sector rotation score: mean 3-day return per sector,
        normalized to [-1, 1] range.
        """
        from pipeline.universe import get_sector

        sector_returns: Dict[str, List[float]] = {}
        for symbol in symbols:
            df = price_data.get(symbol)
            if df is None or df.empty:
                continue
            close_col = self._close_col(df)
            if close_col is None or len(df) < 4:
                continue
            closes = df[close_col].values
            ret_3d = (float(closes[-1]) / float(closes[-4])) - 1.0 if closes[-4] != 0 else 0.0
            sector = get_sector(symbol)
            sector_returns.setdefault(sector, []).append(ret_3d)

        sector_scores: Dict[str, float] = {}
        all_means = []
        for sector, rets in sector_returns.items():
            m = float(np.mean(rets))
            sector_scores[sector] = m
            all_means.append(m)

        # Normalize to [-1, 1]
        if all_means:
            max_abs = max(abs(v) for v in all_means) or 1e-8
            sector_scores = {k: v / max_abs for k, v in sector_scores.items()}

        return sector_scores

    def _technical_proximity(self, df: pd.DataFrame, close_col: str) -> float:
        """
        Distance to 52-week high/low.
        Returns score in [0, 1]:
          - Near 52w high → higher score (momentum)
          - Near 52w low → lower score
        Uses last 252 trading days if available.
        """
        closes = df[close_col].values
        window = min(252, len(closes))
        if window < 20:
            return 0.5
        recent = closes[-window:]
        high_52 = float(np.max(recent))
        low_52 = float(np.min(recent))
        current = float(closes[-1])
        rng = high_52 - low_52
        if rng <= 0:
            return 0.5
        return float((current - low_52) / rng)

    def _build_sentiment_map(self, sentiment_data: Optional[Any]) -> Dict[str, float]:
        """
        Extract per-symbol sentiment score.
        Handles DataFrame (multi-indexed or flat) or dict formats.
        Returns {symbol: normalized_score} in [-1, 1].
        """
        if sentiment_data is None:
            return {}

        scores: Dict[str, float] = {}

        if isinstance(sentiment_data, pd.DataFrame) and not sentiment_data.empty:
            # Typical: multi-indexed (symbol, date) or has 'symbol' column
            score_col = None
            for candidate in ["compound_score", "weighted_compound_score", "sentiment_score"]:
                if candidate in sentiment_data.columns:
                    score_col = candidate
                    break
            if score_col is None:
                return {}

            if isinstance(sentiment_data.index, pd.MultiIndex) and "symbol" in sentiment_data.index.names:
                for symbol in sentiment_data.index.get_level_values("symbol").unique():
                    try:
                        vals = sentiment_data.loc[symbol][score_col].values
                        if len(vals) >= 2:
                            # 24h change: last - second-to-last
                            scores[symbol] = float(np.clip(vals[-1] - vals[-2], -1.0, 1.0))
                        elif len(vals) == 1:
                            scores[symbol] = float(np.clip(vals[0], -1.0, 1.0))
                    except Exception:
                        pass
            elif "symbol" in sentiment_data.columns:
                for symbol, group in sentiment_data.groupby("symbol"):
                    vals = group[score_col].values
                    if len(vals) >= 2:
                        scores[str(symbol)] = float(np.clip(vals[-1] - vals[-2], -1.0, 1.0))
                    elif len(vals) == 1:
                        scores[str(symbol)] = float(np.clip(vals[0], -1.0, 1.0))

        elif isinstance(sentiment_data, dict):
            for symbol, val in sentiment_data.items():
                if isinstance(val, (int, float)):
                    scores[str(symbol)] = float(np.clip(val, -1.0, 1.0))
                elif isinstance(val, dict):
                    for key in ["compound_score", "sentiment_score", "score"]:
                        if key in val:
                            scores[str(symbol)] = float(np.clip(float(val[key]), -1.0, 1.0))
                            break

        return scores

    # ── earnings blackout ────────────────────────────────────

    def _get_earnings_blackout(self, earnings_data: Optional[Dict]) -> set:
        """Return set of symbols within ±blackout_days of an earnings date."""
        if not earnings_data:
            return set()

        blackout = set()
        today = datetime.now(timezone.utc).date()
        window = timedelta(days=self.cfg.earnings_blackout_days)

        for key in ["upcoming_earnings_by_symbol", "latest_earnings_by_symbol"]:
            by_symbol = earnings_data.get(key, {})
            if not isinstance(by_symbol, dict):
                continue
            for symbol, info in by_symbol.items():
                if not isinstance(info, dict):
                    continue
                for date_key in ["reported_date", "date", "earnings_date"]:
                    date_str = info.get(date_key)
                    if not date_str:
                        continue
                    try:
                        edate = pd.Timestamp(date_str).date()
                        if abs((edate - today).days) <= self.cfg.earnings_blackout_days:
                            blackout.add(str(symbol).strip().upper())
                    except Exception:
                        pass

        return blackout

    # ── DataFrame column helpers ─────────────────────────────

    @staticmethod
    def _close_col(df: pd.DataFrame) -> Optional[str]:
        for c in ["Close", "close", "adj_close", "Adj Close"]:
            if c in df.columns:
                return c
        return None

    @staticmethod
    def _volume_col(df: pd.DataFrame) -> Optional[str]:
        for c in ["Volume", "volume", "vol"]:
            if c in df.columns:
                return c
        return None

    @staticmethod
    def _avg_dollar_volume(df: pd.DataFrame, close_col: str, vol_col: str, window: int = 20) -> float:
        """Average daily dollar (or rupee/USDT) volume over last `window` days."""
        if len(df) < window:
            return 0.0
        closes = df[close_col].values[-window:]
        volumes = df[vol_col].values[-window:]
        dollar_volumes = closes * volumes
        return float(np.mean(dollar_volumes))

    # ── lazy price fetch ─────────────────────────────────────

    def _fetch_price_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Best-effort lazy load via PriceDataPipeline."""
        try:
            from pipeline.price_collector import PriceDataPipeline
            return PriceDataPipeline(symbols=symbols).run_incremental_update()
        except Exception as exc:
            logger.warning(f"DynamicUniverseFilter: price fetch failed: {exc}")
            return {}

    # ── persistence ──────────────────────────────────────────

    def _persist(self, hot_list: List[Dict[str, Any]]) -> None:
        """Save hot list to data/hot_list.json."""
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "count": len(hot_list),
            "top_k": self.cfg.top_k,
            "symbols": hot_list,
        }
        try:
            self.cfg.hot_list_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cfg.hot_list_path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2)
            tmp.replace(self.cfg.hot_list_path)
        except OSError as exc:
            logger.warning(f"DynamicUniverseFilter: persist failed: {exc}")

    def _load_from_disk(self) -> Optional[List[Dict[str, Any]]]:
        """Load cached hot list if within cache window."""
        if not self.cfg.hot_list_path.exists():
            return None
        try:
            with open(self.cfg.hot_list_path, "r") as f:
                payload = json.load(f)
            generated_at = payload.get("generated_at", "")
            ts = pd.Timestamp(generated_at)
            age_hours = (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 3600
            if age_hours > self.cfg.cache_hours:
                logger.info(f"DynamicUniverseFilter: disk cache stale ({age_hours:.1f}h old)")
                return None
            symbols = payload.get("symbols", [])
            logger.info(f"DynamicUniverseFilter: loaded {len(symbols)} symbols from cache ({age_hours:.1f}h old)")
            return symbols
        except Exception as exc:
            logger.warning(f"DynamicUniverseFilter: cache load failed: {exc}")
            return None


# ─────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────

def get_dynamic_universe_filter() -> DynamicUniverseFilter:
    """Factory function."""
    return DynamicUniverseFilter()
