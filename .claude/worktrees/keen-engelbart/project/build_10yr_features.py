#!/usr/bin/env python3
"""
Build Features from 10-Year Price Data
===================================
Creates technical indicator features from raw OHLCV data.

Usage:
    python build_10yr_features.py
"""

import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent
PRICES_DIR = PROJECT_DIR / "data" / "prices_10yr"
FEATURES_DIR = PROJECT_DIR / "data" / "features_10yr"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD indicator."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    
    return macd, macd_signal, macd_hist


def compute_bb(series: pd.Series, period: int = 20):
    """Bollinger Bands."""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    
    upper = sma + (std * 2)
    lower = sma - (std * 2)
    
    position = (series - lower) / (upper - lower)
    width = (upper - lower) / sma
    
    return position, width


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr


def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """Stochastic oscillator."""
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=3).mean()
    
    return k, d


def compute_momentum(series: pd.Series, period: int = 20):
    """Momentum indicator."""
    return series.pct_change(period)


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """Average Directional Index."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr = compute_atr(high, low, close, period)
    
    plus_di = 100 * (plus_dm.ewm(span=period).mean() / tr)
    minus_di = 100 * (minus_dm.ewm(span=period).mean() / tr)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(span=period).mean()
    
    return adx, plus_di, minus_di


def compute_obv(close: pd.Series, volume: pd.Series):
    """On-Balance Volume."""
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    obv_trend = obv.diff()
    
    return obv, obv_trend


def compute_mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14):
    """Money Flow Index."""
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    positive_flow = money_flow.where(typical_price.diff() > 0, 0)
    negative_flow = money_flow.where(typical_price.diff() < 0, 0)
    
    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()
    
    mfi = 100 - (100 / (1 + positive_mf / negative_mf))
    return mfi


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all features from OHLCV data."""
    features = pd.DataFrame(index=df.index)
    
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    
    # Basic OHLCV
    features["open"] = df["open"]
    features["high"] = high
    features["low"] = low
    features["close"] = close
    features["volume"] = volume
    
    # Returns
    features["returns_1d"] = close.pct_change()
    
    # RSI
    features["rsi_14"] = compute_rsi(close, 14)
    features["rsi_9"] = compute_rsi(close, 9)
    features["rsi_21"] = compute_rsi(close, 21)
    
    # MACD
    macd, macd_signal, macd_hist = compute_macd(close)
    features["macd"] = macd
    features["macd_signal"] = macd_signal
    features["macd_hist"] = macd_hist
    
    # Bollinger Bands
    bb_position, bb_width = compute_bb(close)
    features["bb_position"] = bb_position
    features["bb_width"] = bb_width
    
    # ATR
    features["atr_14"] = compute_atr(high, low, close, 14)
    features["atr_pct"] = features["atr_14"] / close
    
    # Stochastic
    stoch_k, stoch_d = compute_stochastic(high, low, close)
    features["stoch_k"] = stoch_k
    features["stoch_d"] = stoch_d
    
    # Momentum
    features["momentum_20d"] = compute_momentum(close, 20)
    features["momentum_60d"] = compute_momentum(close, 60)
    features["price_acceleration"] = compute_momentum(features["momentum_20d"], 5)
    
    # Moving averages
    for sma in [9, 21, 50, 200]:
        features[f"close_vs_sma_{sma}"] = (close - close.rolling(sma).mean()) / close.rolling(sma).mean()
    
    # ADX
    adx, di_plus, di_minus = compute_adx(high, low, close)
    features["adx_14"] = adx
    features["di_plus"] = di_plus
    features["di_minus"] = di_minus
    
    # OBV
    obv, obv_trend = compute_obv(close, volume)
    features["obv"] = obv
    features["obv_trend"] = obv_trend
    
    # MFI
    features["mfi"] = compute_mfi(high, low, close, volume)
    
    # Volatility
    features["hist_vol_10"] = close.pct_change().rolling(10).std()
    features["hist_vol_30"] = close.pct_change().rolling(30).std()
    features["realized_vol_21d"] = close.pct_change().rolling(21).std()
    
    # Williams %R
    features["williams_r"] = -100 * (high.rolling(14).max() - close) / (high.rolling(14).max() - low.rolling(14).min())
    
    # ROC
    for p in [5, 10, 20]:
        features[f"roc_{p}"] = close.pct_change(p)
    
    # Price relative to ranges
    features["52w_high_ratio"] = close / close.rolling(252).max()
    features["52w_low_ratio"] = close / close.rolling(252).min()
    
    # Fill NaN
    features = features.ffill().fillna(0)
    
    return features


def main():
    logger.info("Building features from 10-year price data...")
    
    # Load prices
    prices = {}
    for f in PRICES_DIR.glob("*.parquet"):
        try:
            df = pd.read_parquet(f)
            sym = f.stem
            prices[sym] = df
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")
    
    logger.info(f"Loaded {len(prices)} price files")
    
    # Build features
    features_count = 0
    for sym, df in prices.items():
        if df.empty or len(df) < 500:
            continue
        
        try:
            feats = build_features(df)
            if feats.empty:
                continue
            
            # Save
            out_path = FEATURES_DIR / f"{sym}.parquet"
            feats.to_parquet(out_path, compression="snappy")
            features_count += 1
            
            if features_count % 10 == 0:
                logger.info(f"Processed {features_count} symbols...")
                
        except Exception as e:
            logger.warning(f"Failed to build features for {sym}: {e}")
    
    logger.info(f"Complete! Built features for {features_count} symbols")
    return 0


if __name__ == "__main__":
    sys.exit(main())