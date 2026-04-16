#!/usr/bin/env python3
"""
Build 10-Year Daily Crypto Base Features
==========================================
Fetches 10 years of daily OHLCV from Binance public API (free, no key needed)
for each symbol in CRYPTO_DEPTH_SYMBOLS and builds the same technical indicator
features as build_10yr_features.py does for equities.

Creates BTCUSDT.parquet, ETHUSDT.parquet, etc. in data/features_10yr/ so
the crypto intraday script can then layer 4h features on top.

Usage:
    python build_crypto_base_features.py                        # all symbols from .env
    python build_crypto_base_features.py --symbols BTCUSDT ETHUSDT
    python build_crypto_base_features.py --days 3650            # 10yr (default)
    python build_crypto_base_features.py --skip-existing        # skip already built

After this, run:
    python build_crypto_intraday_features.py
"""

import argparse
import logging
import time
from pathlib import Path

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
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

BINANCE_URL = "https://api.binance.com/api/v3/klines"
MAX_CANDLES_PER_REQUEST = 1000
RATE_DELAY = 0.2


# ── Config ────────────────────────────────────────────────────────────────────

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


def get_crypto_symbols() -> list:
    env = _load_env()
    depth_syms = env.get("CRYPTO_DEPTH_SYMBOLS", "")
    return [s.strip() for s in depth_syms.split(",") if s.strip().endswith("USDT")]


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_daily_candles(symbol: str, days: int = 3650) -> pd.DataFrame:
    """Fetch daily OHLCV from Binance public API. No key required."""
    import time as t
    start_ms = int((t.time() - days * 86400) * 1000)

    all_candles = []
    current_start = start_ms

    while True:
        try:
            r = requests.get(BINANCE_URL, params={
                "symbol": symbol,
                "interval": "1d",
                "startTime": current_start,
                "limit": MAX_CANDLES_PER_REQUEST,
            }, timeout=15)
            if r.status_code != 200:
                logger.warning(f"{symbol}: Binance returned {r.status_code}")
                break
            data = r.json()
            if not data:
                break
            all_candles.extend(data)
            if len(data) < MAX_CANDLES_PER_REQUEST:
                break
            current_start = data[-1][0] + 1
            time.sleep(RATE_DELAY)
        except Exception as e:
            logger.warning(f"{symbol}: fetch error — {e}")
            break

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("open_time")[["open", "high", "low", "close", "volume"]]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


# ── Technical indicators (matches build_10yr_features.py) ─────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return (100 - (100 / (1 + rs))).clip(0, 100)


def compute_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_sig, macd - macd_sig


def compute_bb(series: pd.Series, period: int = 20):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + std * 2
    lower = sma - std * 2
    return (series - lower) / (upper - lower + 1e-9), (upper - lower) / (sma + 1e-9)


def compute_atr(high, low, close, period: int = 14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def compute_stochastic(high, low, close, period: int = 14):
    ll = low.rolling(window=period).min()
    hh = high.rolling(window=period).max()
    k = 100 * (close - ll) / (hh - ll + 1e-9)
    return k, k.rolling(3).mean()


def compute_adx(high, low, close, period: int = 14):
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    atr = compute_atr(high, low, close, period)
    plus_di = 100 * plus_dm.ewm(span=period).mean() / (atr + 1e-9)
    minus_di = 100 * minus_dm.ewm(span=period).mean() / (atr + 1e-9)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    return dx.ewm(span=period).mean(), plus_di, minus_di


def compute_obv(close, volume):
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv, obv.diff()


def compute_mfi(high, low, close, volume, period: int = 14):
    tp = (high + low + close) / 3
    mf = tp * volume
    pos = mf.where(tp.diff() > 0, 0).rolling(period).sum()
    neg = mf.where(tp.diff() < 0, 0).rolling(period).sum()
    return (100 - 100 / (1 + pos / (neg + 1e-9))).clip(0, 100)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build full technical indicator set from daily OHLCV."""
    f = pd.DataFrame(index=df.index)
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    open_ = df["open"]

    # Raw OHLCV
    f["open"] = open_
    f["high"] = high
    f["low"] = low
    f["close"] = close
    f["volume"] = volume

    # Returns
    for p in [1, 3, 5, 10, 20, 60]:
        f[f"return_{p}d"] = close.pct_change(p)

    # RSI
    f["rsi_7"] = compute_rsi(close, 7)
    f["rsi_14"] = compute_rsi(close, 14)

    # MACD
    macd, macd_sig, macd_hist = compute_macd(close)
    f["macd"] = macd
    f["macd_signal"] = macd_sig
    f["macd_hist"] = macd_hist

    # Bollinger Bands
    f["bb_position"], f["bb_width"] = compute_bb(close)

    # ATR
    atr = compute_atr(high, low, close, 14)
    f["atr_14"] = atr
    f["atr_pct"] = atr / (close + 1e-9)

    # Stochastic
    f["stoch_k"], f["stoch_d"] = compute_stochastic(high, low, close)

    # Momentum
    f["momentum_20d"] = close.pct_change(20)
    f["momentum_60d"] = close.pct_change(60)
    f["price_acceleration"] = close.pct_change(20).pct_change(5)

    # SMA distance
    for sma in [9, 21, 50, 200]:
        sma_val = close.rolling(sma).mean()
        f[f"close_vs_sma_{sma}"] = (close - sma_val) / (sma_val + 1e-9)

    # ADX
    f["adx_14"], f["di_plus"], f["di_minus"] = compute_adx(high, low, close)

    # OBV
    f["obv"], f["obv_trend"] = compute_obv(close, volume)

    # MFI
    f["mfi"] = compute_mfi(high, low, close, volume)

    # Volatility
    ret = close.pct_change()
    f["hist_vol_10"] = ret.rolling(10).std()
    f["hist_vol_30"] = ret.rolling(30).std()
    f["realized_vol_21d"] = ret.rolling(21).std()

    # Williams %R
    hh14 = high.rolling(14).max()
    ll14 = low.rolling(14).min()
    f["williams_r"] = -100 * (hh14 - close) / (hh14 - ll14 + 1e-9)

    # ROC
    for p in [5, 10, 20]:
        f[f"roc_{p}"] = close.pct_change(p)

    # 52-week range
    f["52w_high_ratio"] = close / (close.rolling(252).max() + 1e-9)
    f["52w_low_ratio"] = close / (close.rolling(252).min() + 1e-9)

    # Crypto-specific: volume patterns (no market hours — 24/7)
    f["vol_vs_30d_avg"] = volume / (volume.rolling(30).mean() + 1e-9)
    f["vol_vs_7d_avg"] = volume / (volume.rolling(7).mean() + 1e-9)

    # Candle structure
    f["hl_ratio"] = (high - low) / (close + 1e-9)
    f["hl_ratio_5d_avg"] = f["hl_ratio"].rolling(5).mean()
    f["body_ratio"] = (close - open_).abs() / (high - low + 1e-9)
    f["direction"] = np.sign(close - open_)

    f = f.ffill().fillna(0)
    return f


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build 10yr base features for crypto symbols")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Override symbols (default: CRYPTO_DEPTH_SYMBOLS from .env)")
    parser.add_argument("--days", type=int, default=3650,
                        help="Days of history to fetch (default 3650 = ~10yr)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip symbols that already have a feature file")
    args = parser.parse_args()

    symbols = args.symbols or get_crypto_symbols()
    if not symbols:
        logger.error("No crypto symbols found. Set CRYPTO_DEPTH_SYMBOLS in project/.env")
        return 1

    logger.info(f"Building base features for {len(symbols)} crypto symbols ({args.days} days)")

    success = failed = skipped = 0

    for sym in symbols:
        out_path = FEATURES_DIR / f"{sym}.parquet"

        if args.skip_existing and out_path.exists():
            logger.debug(f"{sym}: already exists, skipping")
            skipped += 1
            continue

        try:
            df = fetch_daily_candles(sym, days=args.days)
            if df.empty:
                logger.warning(f"{sym}: no data returned from Binance")
                failed += 1
                continue
            if len(df) < 100:
                logger.warning(f"{sym}: only {len(df)} candles, skipping")
                failed += 1
                continue

            features = build_features(df)
            features.to_parquet(out_path, compression="snappy")
            logger.info(f"{sym}: {len(df)} days → {len(features.columns)} features saved")
            success += 1

        except Exception as e:
            logger.warning(f"{sym}: failed — {e}")
            failed += 1

        time.sleep(RATE_DELAY)

    logger.info(f"Done — {success} built, {skipped} skipped, {failed} failed")
    if success > 0:
        logger.info("Next step: python project/build_crypto_intraday_features.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
