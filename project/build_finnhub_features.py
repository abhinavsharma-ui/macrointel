#!/usr/bin/env python3
"""
Finnhub Feature Builder
=======================
Fetches earnings surprises, analyst recommendations, news sentiment,
and basic fundamentals from Finnhub API and joins them onto the
existing features_10yr parquet files.

These are leading/fundamental signals that significantly lift precision
beyond what technical indicators alone can achieve.

Usage:
    python build_finnhub_features.py              # all symbols in features_10yr/
    python build_finnhub_features.py --symbols AAPL MSFT
    python build_finnhub_features.py --workers 4

Requirements:
    pip install requests pandas pyarrow tqdm
"""

import argparse
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent
FEATURES_DIR = PROJECT_DIR / "data" / "features_10yr"
OUTPUT_DIR = PROJECT_DIR / "data" / "features_10yr"  # enrich in-place

# Load API key from .env
def _load_env() -> dict:
    env = {}
    env_path = PROJECT_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env

_ENV = _load_env()
FINNHUB_KEY = _ENV.get("FINNHUB_API_KEYS", "").split(",")[0].strip()
BASE_URL = "https://finnhub.io/api/v1"
RATE_LIMIT_DELAY = 0.5  # 2 req/sec — free tier allows 60/min, being conservative
RATE_LIMIT_SLEEP_ON_429 = 90  # Sleep longer on rate limit (was 60s, increased to 90s)


def _normalize_daily_index(values) -> pd.DatetimeIndex:
    idx = pd.to_datetime(values, errors="coerce")
    idx = pd.DatetimeIndex(idx)
    # Convert to UTC first if timezone-aware
    if idx.tz is not None:
        idx = idx.tz_convert("UTC")
    # ALWAYS strip timezone to avoid datetime64[ms, TZ] dtype issues
    idx = idx.tz_localize(None)
    # Normalize to midnight
    idx = idx.normalize()
    # Standardize resolution to nanoseconds (fixes pandas 2.x dtype mismatches)
    try:
        idx = idx.as_unit("ns")
    except AttributeError:
        idx = pd.DatetimeIndex(idx.astype("datetime64[ns]"))
    return idx


# ── API helpers ──────────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict, retries: int = 3) -> Optional[dict]:
    params["token"] = FINNHUB_KEY
    for attempt in range(retries):
        try:
            r = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=10)
            if r.status_code == 429:
                logger.warning(f"Rate limited (429) — sleeping {RATE_LIMIT_SLEEP_ON_429}s")
                time.sleep(RATE_LIMIT_SLEEP_ON_429)
                continue
            if r.status_code == 401:
                logger.error("API key invalid (401) — check FINNHUB_API_KEYS in .env")
                return None
            if r.status_code == 403:
                logger.error("API key forbidden (403) — quota may be exceeded")
                return None
            if r.status_code != 200:
                logger.debug(f"API returned {r.status_code} for {endpoint}")
                return None
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                logger.debug(f"Request failed {endpoint}: {e}")
            time.sleep(1)
    return None


# ── Feature extractors ───────────────────────────────────────────────────────

def fetch_earnings_surprises(symbol: str) -> pd.DataFrame:
    """
    Returns a df indexed by date with:
      - earnings_surprise_pct : (actual - estimate) / |estimate|
      - earnings_beat          : 1 if beat, -1 if miss, 0 if in-line
      - earnings_yoy_growth    : year-over-year EPS growth %
    """
    data = _get("stock/earnings", {"symbol": symbol, "limit": 40})
    time.sleep(RATE_LIMIT_DELAY)
    if not data:
        return pd.DataFrame()

    rows = []
    for i, q in enumerate(data):
        try:
            actual = float(q.get("actual") or 0)
            estimate = float(q.get("estimate") or 0)
            date_str = q.get("period") or q.get("date")
            if not date_str:
                continue
            dt = pd.Timestamp(date_str)
            if dt.tzinfo is not None:
                dt = dt.tz_convert("UTC").tz_localize(None)
            dt = dt.normalize()
            surprise_pct = (actual - estimate) / (abs(estimate) + 1e-9)
            beat = 1 if surprise_pct > 0.01 else (-1 if surprise_pct < -0.01 else 0)
            # YoY: compare to same quarter 1yr ago (4 quarters back)
            yoy = np.nan
            if i + 4 < len(data):
                prev_actual = float(data[i + 4].get("actual") or 0)
                if abs(prev_actual) > 1e-6:
                    yoy = (actual - prev_actual) / abs(prev_actual)
            rows.append({
                "date": dt,
                "earnings_surprise_pct": np.clip(surprise_pct, -3.0, 3.0),
                "earnings_beat": beat,
                "earnings_yoy_growth": np.clip(yoy, -5.0, 5.0) if not np.isnan(yoy) else np.nan,
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("date").sort_index()
    # Forward-fill so every trading day gets the latest earnings signal
    return df


def fetch_analyst_recommendations(symbol: str) -> pd.DataFrame:
    """
    Returns monthly analyst recommendation counts as features:
      - analyst_strong_buy_pct, analyst_buy_pct, analyst_hold_pct, analyst_sell_pct
      - analyst_consensus_score : weighted avg (5=strong buy, 1=strong sell)
    """
    data = _get("stock/recommendation", {"symbol": symbol})
    time.sleep(RATE_LIMIT_DELAY)
    if not data:
        return pd.DataFrame()

    rows = []
    for rec in data:
        try:
            dt = pd.Timestamp(rec.get("period"))
            if dt.tzinfo is not None:
                dt = dt.tz_convert("UTC").tz_localize(None)
            dt = dt.normalize()
            sb = int(rec.get("strongBuy") or 0)
            b = int(rec.get("buy") or 0)
            h = int(rec.get("hold") or 0)
            s = int(rec.get("sell") or 0)
            ss = int(rec.get("strongSell") or 0)
            total = sb + b + h + s + ss
            if total == 0:
                continue
            score = (5 * sb + 4 * b + 3 * h + 2 * s + 1 * ss) / total
            rows.append({
                "date": dt,
                "analyst_strong_buy_pct": sb / total,
                "analyst_buy_pct": (sb + b) / total,
                "analyst_hold_pct": h / total,
                "analyst_sell_pct": (s + ss) / total,
                "analyst_consensus_score": score,
                "analyst_total_coverage": total,
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("date").sort_index()


def fetch_news_sentiment(symbol: str, days_back: int = 365 * 10) -> pd.DataFrame:
    """
    Returns daily news sentiment (average sentiment score per day):
      - news_sentiment_avg   : mean sentiment [-1, 1]
      - news_sentiment_pos   : fraction of positive articles
      - news_count           : number of articles that day
    Finnhub free tier: last ~1yr of news only. We'll use what's available.
    """
    to_date = datetime.now()
    from_date = to_date - timedelta(days=min(days_back, 365))  # free tier limit
    data = _get("company-news", {
        "symbol": symbol,
        "from": from_date.strftime("%Y-%m-%d"),
        "to": to_date.strftime("%Y-%m-%d"),
    })
    time.sleep(RATE_LIMIT_DELAY)
    if not data:
        return pd.DataFrame()

    rows = []
    for article in data:
        try:
            ts = article.get("datetime")
            if not ts:
                continue
            dt = pd.to_datetime(ts, unit="s", utc=True).tz_convert("UTC").tz_localize(None).normalize()
            sentiment = float(article.get("sentiment", 0) or 0)
            rows.append({"date": dt, "sentiment": sentiment})
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    daily = df.groupby("date").agg(
        news_sentiment_avg=("sentiment", "mean"),
        news_sentiment_pos=("sentiment", lambda x: (x > 0.1).mean()),
        news_count=("sentiment", "count"),
    )
    return daily.sort_index()


def fetch_basic_financials(symbol: str) -> dict:
    """
    Returns scalar fundamental metrics:
      pe_ttm, pb, ev_ebitda, roe, debt_equity, current_ratio, revenue_growth_ttm
    """
    data = _get("stock/metric", {"symbol": symbol, "metric": "all"})
    time.sleep(RATE_LIMIT_DELAY)
    if not data or "metric" not in data:
        return {}

    m = data["metric"]
    return {
        "fundamental_pe_ttm": m.get("peTTM"),
        "fundamental_pb": m.get("pbAnnual"),
        "fundamental_ev_ebitda": m.get("evEbitdaTTM"),
        "fundamental_roe": m.get("roeTTM"),
        "fundamental_debt_equity": m.get("totalDebt/totalEquityAnnual"),
        "fundamental_revenue_growth": m.get("revenueGrowthTTMYoy"),
        "fundamental_eps_growth": m.get("epsGrowthTTMYoy"),
        "fundamental_52w_high_pct": m.get("52WeekPriceReturnDaily"),
    }


# ── Main enrichment ──────────────────────────────────────────────────────────

def enrich_symbol(symbol: str) -> bool:
    """Load existing feature parquet, add Finnhub features, save back."""
    parquet_path = FEATURES_DIR / f"{symbol}.parquet"
    if not parquet_path.exists():
        logger.debug(f"No feature file for {symbol}, skipping")
        return False

    try:
        df = pd.read_parquet(parquet_path)
        df.index = _normalize_daily_index(df.index)
        df.index.name = "date"
        # Ensure index dtype is consistent (pandas 2.x parquet may return datetime64[us])
        if hasattr(df.index, "as_unit"):
            df.index = df.index.as_unit("ns")
    except Exception as e:
        logger.warning(f"Failed to read {symbol}: {e}")
        return False

    # Skip if already enriched with Finnhub features
    if "analyst_consensus_score" in df.columns and "earnings_beat" in df.columns:
        return True  # Already done, no API calls needed

    # No-data cache — symbols Finnhub has no data for, never hit them again
    NO_DATA_CACHE = FEATURES_DIR.parent / ".finnhub_no_data_cache.txt"
    if NO_DATA_CACHE.exists():
        if symbol in set(NO_DATA_CACHE.read_text().splitlines()):
            return False

    # Pre-check: 1 cheap call to see if Finnhub covers this symbol at all
    profile = _get("stock/profile2", {"symbol": symbol})
    time.sleep(RATE_LIMIT_DELAY)
    if not profile or not profile.get("name"):
        with open(NO_DATA_CACHE, "a") as cf:
            cf.write(symbol + "\n")
        return False

    original_cols = set(df.columns)
    features_added = []

    # ── Earnings surprises ──
    earnings_df = fetch_earnings_surprises(symbol)
    if not earnings_df.empty:
        # Reindex to trading days and forward-fill (signal persists until next quarter)
        earnings_df.index = _normalize_daily_index(earnings_df.index)
        earnings_df = earnings_df.reindex(df.index).ffill()
        for col in earnings_df.columns:
            df[col] = earnings_df[col]
        features_added.append("earnings")
    else:
        logger.debug(f"{symbol}: earnings API returned empty")

    # ── Analyst recommendations ──
    rec_df = fetch_analyst_recommendations(symbol)
    if not rec_df.empty:
        rec_df.index = _normalize_daily_index(rec_df.index)
        rec_df = rec_df.reindex(df.index).ffill()
        for col in rec_df.columns:
            df[col] = rec_df[col]
        features_added.append("analyst_recs")
    else:
        logger.debug(f"{symbol}: analyst API returned empty")

    # ── News sentiment ──
    news_df = fetch_news_sentiment(symbol)
    if not news_df.empty:
        news_df.index = _normalize_daily_index(news_df.index)
        news_df = news_df.reindex(df.index).ffill()
        for col in news_df.columns:
            df[col] = news_df[col]
        features_added.append("news_sentiment")
    else:
        logger.debug(f"{symbol}: news API returned empty")

    # ── Scalar fundamentals (broadcast as constant columns) ──
    fundamentals = fetch_basic_financials(symbol)
    fundamental_count = 0
    for k, v in fundamentals.items():
        if v is not None:
            try:
                df[k] = float(v)
                fundamental_count += 1
            except (TypeError, ValueError):
                pass
    if fundamental_count > 0:
        features_added.append(f"fundamentals({fundamental_count})")
    else:
        logger.debug(f"{symbol}: fundamentals API returned empty")

    new_cols = set(df.columns) - original_cols
    if not new_cols:
        NO_DATA_CACHE = FEATURES_DIR.parent / ".finnhub_no_data_cache.txt"
        with open(NO_DATA_CACHE, "a") as cf:
            cf.write(symbol + "\n")
        return False

    # Save enriched parquet
    df.to_parquet(parquet_path)
    logger.info(f"{symbol}: added {len(new_cols)} Finnhub features — {sorted(new_cols)}")
    return True


def _validate_api_key() -> bool:
    """Quick validation that API key is working."""
    test = _get("quote", {"symbol": "AAPL"})
    if test is None:
        logger.error("API key validation failed — API returned empty/error. Check key is valid.")
        return False
    if "data" not in test and "c" not in test:  # quote endpoint returns 'c' for last price
        logger.error("API key validation failed — unexpected response format")
        return False
    logger.info("API key validated successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description="Enrich feature files with Finnhub data")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=2,  # keep low for rate limits
                        help="Parallel workers (keep ≤2 for Finnhub free tier)")
    args = parser.parse_args()

    if not FINNHUB_KEY:
        logger.error("FINNHUB_API_KEYS not set in project/.env")
        return 1

    logger.info(f"API key: {FINNHUB_KEY[:8]}...")
    if not _validate_api_key():
        return 1

    time.sleep(2)  # Wait after validation before main run

    if args.symbols:
        symbols = args.symbols
    else:
        symbols = [p.stem for p in sorted(FEATURES_DIR.glob("*.parquet"))]

    logger.info(f"Enriching {len(symbols)} symbols with Finnhub features")

    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(enrich_symbol, s): s for s in symbols}
        for i, fut in enumerate(as_completed(futures), 1):
            sym = futures[fut]
            try:
                ok = fut.result()
                if ok:
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                logger.warning(f"{sym} failed: {e}")
                failed += 1
            if i % 20 == 0:
                logger.info(f"Progress: {i}/{len(symbols)} — {success} enriched, {failed} skipped")

    logger.info(f"Done — {success} enriched, {failed} skipped/failed")
    logger.info("Run retrain now: python retrain_institutional_models.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
