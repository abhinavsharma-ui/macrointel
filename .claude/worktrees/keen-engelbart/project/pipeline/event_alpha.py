"""
Event Alpha Features
====================
Builds two event-driven feature families:
  1. Second-order earnings propagation across sector peers
  2. Event-day close reversal / exhaustion proxies from OHLCV bars

The earnings propagation layer prefers real earnings events. When those are
missing, it can fall back to a small number of extreme gap-volume proxy events
so the engine still has event context instead of going fully blind.
"""

import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from pipeline.universe import get_sector

logger = logging.getLogger(__name__)

PROP_FAST_WINDOW_DAYS = max(1, int(os.getenv("EARNINGS_PROPAGATION_FAST_WINDOW_DAYS", "3")))
PROP_WINDOW_DAYS = max(PROP_FAST_WINDOW_DAYS, int(os.getenv("EARNINGS_PROPAGATION_WINDOW_DAYS", "7")))
PROXY_ENABLED = os.getenv("EVENT_PROXY_FALLBACK_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
PROXY_GAP_THRESHOLD = float(os.getenv("EVENT_PROXY_GAP_THRESHOLD", "0.04"))
PROXY_VOLUME_Z_THRESHOLD = float(os.getenv("EVENT_PROXY_VOLUME_Z_THRESHOLD", "2.0"))
PROXY_MAX_EVENTS_PER_SYMBOL = max(1, int(os.getenv("EVENT_PROXY_MAX_EVENTS_PER_SYMBOL", "2")))

EVENT_FEATURE_COLUMNS = [
    "peer_earnings_shock_3d",
    "peer_earnings_shock_7d",
    "peer_earnings_breadth_7d",
    "peer_earnings_event_count_7d",
    "peer_earnings_negative_ratio_7d",
    "earnings_propagation_signal",
    "earnings_propagation_strength",
    "close_reversal_signal",
    "close_reversal_strength",
    "event_move_strength",
    "event_day_extreme",
]


def _market(symbol: str) -> str:
    return "nse" if symbol.endswith(".NS") else "us"


def _normalize_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(df.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    return index.normalize()


def _align_event_date(index: pd.DatetimeIndex, raw_date) -> Optional[pd.Timestamp]:
    if raw_date is None or index.empty:
        return None
    event_date = pd.Timestamp(raw_date).normalize()
    candidates = index[index >= event_date]
    if len(candidates) == 0:
        return None
    return pd.Timestamp(candidates[0]).normalize()


def _clip_unit(values) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), -1.0, 1.0)


def _compute_close_reversal_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    prev_close = close.shift(1).replace(0, np.nan)
    body = (close - open_) / open_.replace(0, np.nan)
    gap = (open_ - prev_close) / prev_close
    range_pct = (high - low) / prev_close
    upper_wick = (high - np.maximum(close, open_)) / prev_close
    lower_wick = (np.minimum(close, open_) - low) / prev_close
    vol_ma20 = volume.rolling(20).mean()
    vol_std20 = volume.rolling(20).std().replace(0, np.nan)
    vol_z = ((volume - vol_ma20) / vol_std20).clip(-4, 4).fillna(0)

    bullish_exhaustion = (
        np.clip((-gap) / 0.05, 0, 1) * 0.35
        + np.clip((-body) / 0.05, 0, 1) * 0.20
        + np.clip(lower_wick / 0.03, 0, 1) * 0.30
        + np.clip(vol_z / 3.0, 0, 1) * 0.15
    )
    bearish_exhaustion = (
        np.clip(gap / 0.05, 0, 1) * 0.35
        + np.clip(body / 0.05, 0, 1) * 0.20
        + np.clip(upper_wick / 0.03, 0, 1) * 0.30
        + np.clip(vol_z / 3.0, 0, 1) * 0.15
    )

    close_reversal = (bullish_exhaustion - bearish_exhaustion).clip(-1, 1)
    event_move_strength = (
        np.clip(gap.abs() / 0.05, 0, 1) * 0.40
        + np.clip(body.abs() / 0.06, 0, 1) * 0.30
        + np.clip(range_pct.abs() / 0.08, 0, 1) * 0.15
        + np.clip(vol_z / 3.0, 0, 1) * 0.15
    ).clip(0, 1)
    close_reversal_strength = np.maximum(bullish_exhaustion, bearish_exhaustion).clip(0, 1)

    return pd.DataFrame(
        {
            "close_reversal_signal": close_reversal.fillna(0.0),
            "close_reversal_strength": close_reversal_strength.fillna(0.0),
            "event_move_strength": event_move_strength.fillna(0.0),
            "event_day_extreme": ((event_move_strength > 0.55) & (close_reversal_strength > 0.45)).astype(float),
        },
        index=pd.DatetimeIndex(df.index),
    )


def compute_close_reversal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Public wrapper for the close-reversal / exhaustion feature block.

    This is used by the live inference loop to refresh only the latest bar's
    reversal metrics without recomputing the full event propagation layer.
    """
    return _compute_close_reversal_features(df)


def _build_sector_correlations(price_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    sector_members: Dict[str, List[str]] = defaultdict(list)
    for symbol in price_data:
        sector_members[get_sector(symbol)].append(symbol)

    correlations: Dict[str, pd.DataFrame] = {}
    for sector, symbols in sector_members.items():
        if len(symbols) < 2:
            continue
        returns = {}
        for symbol in symbols:
            df = price_data.get(symbol)
            if df is None or df.empty:
                continue
            returns[symbol] = df["close"].astype(float).pct_change()
        if len(returns) < 2:
            continue
        frame = pd.DataFrame(returns).tail(120)
        correlations[sector] = frame.corr(min_periods=20).fillna(0.0)
    return correlations


def _prepare_actual_events(price_data: Dict[str, pd.DataFrame], earnings_events: Optional[pd.DataFrame]) -> List[Dict]:
    if earnings_events is None or earnings_events.empty:
        return []

    events: List[Dict] = []
    for _, row in earnings_events.iterrows():
        symbol = str(row.get("symbol", "")).upper().strip()
        df = price_data.get(symbol)
        if df is None or df.empty:
            continue

        trade_index = _normalize_index(df)
        event_day = _align_event_date(trade_index, row.get("reported_date"))
        if event_day is None:
            continue

        raw_index = pd.DatetimeIndex(pd.to_datetime(df.index))
        if raw_index.tz is not None:
            raw_index = raw_index.tz_localize(None)
        position = raw_index.get_indexer([event_day], method="nearest")[0]
        if position <= 0:
            continue

        prev_close = float(df["close"].iloc[position - 1])
        day_close = float(df["close"].iloc[position])
        if prev_close <= 0:
            continue

        reaction_1d = (day_close / prev_close) - 1.0
        surprise_pct = pd.to_numeric(row.get("surprise_pct"), errors="coerce")
        surprise_component = 0.0 if pd.isna(surprise_pct) else float(np.clip(surprise_pct / 20.0, -1, 1))
        reaction_component = float(np.clip(reaction_1d / 0.08, -1, 1))
        shock = 0.55 * surprise_component + 0.45 * reaction_component
        if abs(shock) < 0.05:
            continue

        events.append(
            {
                "symbol": symbol,
                "event_day": event_day,
                "sector": get_sector(symbol),
                "market": _market(symbol),
                "shock": float(np.clip(shock, -1, 1)),
                "reliability": 1.0,
                "source": "earnings",
            }
        )
    return events


def _prepare_proxy_events(price_data: Dict[str, pd.DataFrame]) -> List[Dict]:
    if not PROXY_ENABLED:
        return []

    latest_seen = max((pd.Timestamp(df.index[-1]).normalize() for df in price_data.values() if df is not None and not df.empty), default=None)
    if latest_seen is None:
        return []
    cutoff = latest_seen - pd.Timedelta(days=PROP_WINDOW_DAYS + 7)

    events: List[Dict] = []
    for symbol, df in price_data.items():
        if df is None or df.empty or len(df) < 25:
            continue
        close = df["close"].astype(float)
        open_ = df["open"].astype(float)
        volume = df["volume"].astype(float)
        prev_close = close.shift(1).replace(0, np.nan)
        gap = (open_ - prev_close) / prev_close
        body = (close - open_) / open_.replace(0, np.nan)
        vol_ma20 = volume.rolling(20).mean()
        vol_std20 = volume.rolling(20).std().replace(0, np.nan)
        vol_z = ((volume - vol_ma20) / vol_std20).fillna(0)

        candidates = df.index[
            (gap.abs() >= PROXY_GAP_THRESHOLD)
            & (vol_z >= PROXY_VOLUME_Z_THRESHOLD)
            & ((gap.abs() + body.abs()) >= 0.05)
        ]
        recent_candidates = [pd.Timestamp(dt).normalize() for dt in candidates if pd.Timestamp(dt).normalize() >= cutoff]
        for event_day in recent_candidates[-PROXY_MAX_EVENTS_PER_SYMBOL:]:
            idx = pd.Timestamp(event_day)
            shock = float(np.clip(((gap.loc[idx] * 0.65) + (body.loc[idx] * 0.35)) / 0.07, -1, 1))
            if abs(shock) < 0.25:
                continue
            events.append(
                {
                    "symbol": symbol,
                    "event_day": idx,
                    "sector": get_sector(symbol),
                    "market": _market(symbol),
                    "shock": shock,
                    "reliability": 0.35,
                    "source": "price_shock_proxy",
                }
            )
    return events


def _empty_feature_frame(index) -> pd.DataFrame:
    frame = pd.DataFrame(index=pd.DatetimeIndex(index))
    for col in EVENT_FEATURE_COLUMNS:
        frame[col] = 0.0
    return frame


def build_event_feature_matrices(
    price_data: Dict[str, pd.DataFrame],
    earnings_events: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    if not price_data:
        return {}

    correlations = _build_sector_correlations(price_data)
    actual_events = _prepare_actual_events(price_data, earnings_events)
    proxy_events = _prepare_proxy_events(price_data) if not actual_events else []
    all_events = actual_events + proxy_events

    grouped_events: Dict[tuple, List[Dict]] = defaultdict(list)
    for event in all_events:
        grouped_events[(event["market"], event["sector"])].append(event)

    result: Dict[str, pd.DataFrame] = {}
    for symbol, df in price_data.items():
        if df is None or df.empty:
            continue

        idx = pd.DatetimeIndex(pd.to_datetime(df.index))
        if idx.tz is not None:
            idx = idx.tz_localize(None)

        features = _empty_feature_frame(idx)
        reversal = _compute_close_reversal_features(df)
        features.update(reversal.reindex(features.index).fillna(0.0))

        sector = get_sector(symbol)
        market = _market(symbol)
        peer_events = grouped_events.get((market, sector), [])
        if not peer_events:
            result[symbol] = features
            continue

        corr_matrix = correlations.get(sector)
        normalized_idx = idx.normalize()
        shock_3d = np.zeros(len(features), dtype=float)
        shock_7d = np.zeros(len(features), dtype=float)
        breadth_7d = np.zeros(len(features), dtype=float)
        count_7d = np.zeros(len(features), dtype=float)
        negative_7d = np.zeros(len(features), dtype=float)

        for event in peer_events:
            if event["symbol"] == symbol:
                continue

            relation = 0.45
            if corr_matrix is not None and symbol in corr_matrix.index and event["symbol"] in corr_matrix.columns:
                relation = float(np.clip(corr_matrix.loc[symbol, event["symbol"]], 0.0, 1.0))
            base_weight = (0.35 + 0.65 * relation) * float(event.get("reliability", 1.0))
            event_day = pd.Timestamp(event["event_day"]).normalize()
            delta_days = (normalized_idx - event_day).days
            valid = (delta_days >= 0) & (delta_days <= PROP_WINDOW_DAYS)
            if not np.any(valid):
                continue

            decay = np.exp(-delta_days[valid] / 3.0)
            contribution = float(event["shock"]) * base_weight * decay
            shock_7d[valid] += contribution
            breadth_7d[valid] += np.sign(event["shock"]) * base_weight * decay
            count_7d[valid] += 1.0
            if event["shock"] < 0:
                negative_7d[valid] += 1.0

            fast = valid & (delta_days <= PROP_FAST_WINDOW_DAYS)
            if np.any(fast):
                fast_decay = np.exp(-delta_days[fast] / 2.0)
                shock_3d[fast] += float(event["shock"]) * base_weight * fast_decay

        safe_count = np.maximum(count_7d, 1.0)
        breadth_norm = breadth_7d / safe_count
        negative_ratio = negative_7d / safe_count
        propagation_signal = np.clip((0.55 * shock_3d) + (0.30 * shock_7d) + (0.15 * breadth_norm), -1.0, 1.0)
        propagation_strength = np.clip(np.abs(propagation_signal), 0.0, 1.0)

        features["peer_earnings_shock_3d"] = np.clip(shock_3d, -2.0, 2.0)
        features["peer_earnings_shock_7d"] = np.clip(shock_7d, -3.0, 3.0)
        features["peer_earnings_breadth_7d"] = np.clip(breadth_norm, -1.0, 1.0)
        features["peer_earnings_event_count_7d"] = count_7d
        features["peer_earnings_negative_ratio_7d"] = np.clip(negative_ratio, 0.0, 1.0)
        features["earnings_propagation_signal"] = propagation_signal
        features["earnings_propagation_strength"] = propagation_strength
        result[symbol] = features.fillna(0.0)

    logger.debug(
        f"Event alpha features built for {len(result)} symbols using {len(actual_events)} real earnings events and {len(proxy_events)} proxy events"
    )
    return result
