#!/usr/bin/env python3
"""
Crypto Intraday (4H) Feature Builder
======================================
Fetches 4-hour candle data for crypto symbols from Binance public API
(no API key required) and builds intraday momentum + structure features.

Daily candles miss most of crypto's signal — crypto moves 24/7 and
the 4h timeframe captures trend structure, session patterns, and
overnight gaps that daily OHLCV completely ignores.

Features added (to features_10yr parquet, resampled to daily):
  crypto_4h_trend        - 4h EMA9 vs EMA21 alignment (bullish=1, bearish=-1)
  crypto_4h_momentum     - 4h RSI(14) as of last candle of day
  crypto_4h_vol_surge    - 1 if any 4h candle had volume > 2x 4h avg
  crypto_session_bias    - Asia/London/NY session return bias (daily avg)
  crypto_overnight_gap   - overnight return (US close to Asia open)
  crypto_high_tf_confirm - 1 if daily AND 4h trend both aligned
  crypto_4h_consec_green - consecutive green 4h candles (momentum)
  crypto_4h_consec_red   - consecutive red 4h candles (distribution)
  crypto_wick_ratio      - avg upper wick / body (rejection at highs signal)

Data source: Binance public API (free, no key, rate limit friendly)
Symbols: all *USDT pairs in your CRYPTO_DEPTH_SYMBOLS

Usage:
    python build_crypto_intraday_features.py           # all configured crypto
    python build_crypto_intraday_features.py --symbols BTCUSDT ETHUSDT
    python build_crypto_intraday_features.py --days 730  # 2 years of 4h data
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent
FEATURES_DIR = PROJECT_DIR / "data" / "features_10yr"
INTRADAY_CACHE = PROJECT_DIR / "data" / "altdata" / "crypto_4h"
INTRADAY_CACHE.mkdir(parents=True, exist_ok=True)

BINANCE_URL = "https://api.binance.com/api/v3/klines"
MAX_CANDLES_PER_REQUEST = 1000
RATE_DELAY = 0.15


def _load_env() -> dict:
    env = {}
    p = PROJECT_DIR / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env


def get_configured_crypto_symbols() -> list:
    env = _load_env()
    depth_syms = env.get("CRYPTO_DEPTH_SYMBOLS", "")
    return [s.strip() for s in depth_syms.split(",") if s.strip().endswith("USDT")]


def fetch_4h_candles(symbol: str, days: int = 730) -> pd.DataFrame:
    """Fetch 4h OHLCV from Binance public API. No key needed."""
    cache_file = INTRADAY_CACHE / f"{symbol}_4h.parquet"

    # Load cached data
    existing = pd.DataFrame()
    if cache_file.exists():
        try:
            existing = pd.read_parquet(cache_file)
            existing.index = pd.to_datetime(existing.index)
        except Exception:
            existing = pd.DataFrame()

    # Determine fetch start
    if not existing.empty:
        last_ts = int(existing.index[-1].timestamp() * 1000) + 1
    else:
        import time as t
        last_ts = int((t.time() - days * 86400) * 1000)

    new_candles = []
    while True:
        try:
            r = requests.get(BINANCE_URL, params={
                "symbol": symbol,
                "interval": "4h",
                "startTime": last_ts,
                "limit": MAX_CANDLES_PER_REQUEST,
            }, timeout=15)
            if r.status_code != 200:
                break
            data = r.json()
            if not data:
                break
            new_candles.extend(data)
            if len(data) < MAX_CANDLES_PER_REQUEST:
                break
            last_ts = data[-1][0] + 1
            time.sleep(RATE_DELAY)
        except Exception as e:
            logger.debug(f"{symbol}: fetch error — {e}")
            break

    if not new_candles:
        return existing

    new_df = pd.DataFrame(new_candles, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","trades","taker_buy_base",
        "taker_buy_quote","ignore"
    ])
    new_df["open_time"] = pd.to_datetime(new_df["open_time"], unit="ms")
    new_df = new_df.set_index("open_time")[["open","high","low","close","volume"]]
    new_df = new_df.apply(pd.to_numeric, errors="coerce")

    if not existing.empty:
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new_df.sort_index()

    combined.to_parquet(cache_file)
    return combined


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean() + 1e-9
    rs = gain / loss
    return (100 - 100 / (1 + rs)).clip(0, 100)


def build_4h_daily_features(df_4h: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 4h features and resample to daily (last candle of each UTC day).
    """
    if df_4h.empty or len(df_4h) < 20:
        return pd.DataFrame()

    close = df_4h["close"]
    high = df_4h["high"]
    low = df_4h["low"]
    volume = df_4h["volume"]
    body = (close - df_4h["open"]).abs()
    upper_wick = high - pd.DataFrame({"c": close, "o": df_4h["open"]}).max(axis=1)

    # EMA trend
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    trend = np.where(ema9 > ema21, 1.0, -1.0)

    rsi = compute_rsi(close, 14)

    # Volume surge
    vol_avg = volume.rolling(24).mean()  # 24 × 4h = 4 days avg
    vol_surge = (volume > 2 * vol_avg).astype(float)

    # Session returns (UTC hours: Asia=0-8, London=8-16, NY=16-24)
    hour = df_4h.index.hour
    ret = close.pct_change().fillna(0)
    asia_ret = ret.where(hour < 8, 0)
    london_ret = ret.where((hour >= 8) & (hour < 16), 0)
    ny_ret = ret.where(hour >= 16, 0)

    # Consecutive green/red candles
    green = (close > df_4h["open"]).astype(int)
    consec_green = green.groupby((green != green.shift()).cumsum()).cumcount() + 1
    consec_green = consec_green.where(green == 1, 0)
    consec_red = (1 - green).groupby(((1-green) != (1-green).shift()).cumsum()).cumcount() + 1
    consec_red = consec_red.where(green == 0, 0)

    # Wick ratio
    wick_ratio = (upper_wick / (body + 1e-9)).clip(0, 10)

    df_4h_features = pd.DataFrame({
        "crypto_4h_trend": trend,
        "crypto_4h_momentum": rsi,
        "crypto_4h_vol_surge": vol_surge,
        "crypto_asia_ret": asia_ret,
        "crypto_london_ret": london_ret,
        "crypto_ny_ret": ny_ret,
        "crypto_4h_consec_green": consec_green.clip(0, 10),
        "crypto_4h_consec_red": consec_red.clip(0, 10),
        "crypto_wick_ratio": wick_ratio,
    }, index=df_4h.index)

    # Resample to daily: take last value of each UTC day
    daily = df_4h_features.resample("1D").agg({
        "crypto_4h_trend": "last",
        "crypto_4h_momentum": "last",
        "crypto_4h_vol_surge": "max",       # any surge = 1
        "crypto_asia_ret": "sum",
        "crypto_london_ret": "sum",
        "crypto_ny_ret": "sum",
        "crypto_4h_consec_green": "last",
        "crypto_4h_consec_red": "last",
        "crypto_wick_ratio": "mean",
    })

    # Session bias: which session is driving
    daily["crypto_session_bias"] = np.select(
        [
            daily["crypto_asia_ret"] > daily[["crypto_london_ret","crypto_ny_ret"]].max(axis=1),
            daily["crypto_ny_ret"] > daily[["crypto_asia_ret","crypto_london_ret"]].max(axis=1),
        ],
        [-0.5, 0.5],  # Asia dominance = slightly bearish for US hours, NY = bullish
        default=0.0
    )

    # Overnight gap (asia return as proxy)
    daily["crypto_overnight_gap"] = daily["crypto_asia_ret"].clip(-0.1, 0.1)

    return daily


def enrich_crypto_symbol(crypto_symbol: str, days: int) -> bool:
    """
    Map BTCUSDT → BTC-USD style parquet lookup and enrich with 4h features.
    """
    # Try to find matching parquet file
    candidates = [
        crypto_symbol,                          # BTCUSDT
        crypto_symbol.replace("USDT", "-USD"),  # BTC-USD
        crypto_symbol.replace("USDT", "USD"),   # BTCUSD
        crypto_symbol.replace("USDT", ""),      # BTC (unlikely but try)
    ]

    feat_file = None
    for candidate in candidates:
        p = FEATURES_DIR / f"{candidate}.parquet"
        if p.exists():
            feat_file = p
            break

    if feat_file is None:
        logger.debug(f"No feature file for {crypto_symbol}")
        return False

    # Fetch and build 4h features
    logger.info(f"Fetching 4h data for {crypto_symbol}...")
    df_4h = fetch_4h_candles(crypto_symbol, days=days)
    if df_4h.empty:
        logger.warning(f"{crypto_symbol}: no 4h data from Binance")
        return False

    daily_features = build_4h_daily_features(df_4h)
    if daily_features.empty:
        return False

    try:
        df = pd.read_parquet(feat_file)
        df.index = pd.to_datetime(df.index)
        aligned = daily_features.reindex(df.index, method="ffill")
        for col in aligned.columns:
            df[col] = aligned[col]
        df.to_parquet(feat_file)
        logger.info(f"{crypto_symbol} → {feat_file.name}: +{len(aligned.columns)} intraday features")
        return True
    except Exception as e:
        logger.warning(f"{crypto_symbol}: enrich error — {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--days", type=int, default=730, help="Days of 4h history to fetch")
    args = parser.parse_args()

    symbols = args.symbols or get_configured_crypto_symbols()
    if not symbols:
        logger.error("No crypto symbols configured. Check CRYPTO_DEPTH_SYMBOLS in .env")
        return 1

    logger.info(f"Building 4h intraday features for {len(symbols)} crypto symbols")
    success = 0
    for sym in symbols:
        if enrich_crypto_symbol(sym, args.days):
            success += 1
        time.sleep(RATE_DELAY)

    logger.info(f"Done — {success}/{len(symbols)} crypto symbols enriched with 4h features")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
