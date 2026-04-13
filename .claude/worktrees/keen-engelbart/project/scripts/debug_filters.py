"""
Debug - Show all signals at each filter level
"""

import yfinance as yf
import pandas as pd
import numpy as np

SYMBOLS = [
    "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "TCS.NS", "SBIN.NS",
    "BAJFINANCE.NS", "TITAN.NS", "SUNPHARMA.NS", "MARUTI.NS", "M&M.NS",
    "KOTAKBANK.NS", "ICICIBANK.NS", "ADANIPORTS.NS", "HINDUNILVR.NS",
    "CIPLA.NS", "NESTLEIND.NS", "ULTRACEMCO.NS", "NTPC.NS", "POWERGRID.NS",
    "IRCTC.NS", "VEDL.NS", "TATASTEEL.NS", "COALINDIA.NS", "ONGC.NS",
    "ITC.NS", "HCLTECH.NS", "WIPRO.NS", "DIVISLAB.NS"
]


def calculate_adx(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    plus_dm = high.diff().where(high.diff() > -low.diff(), 0)
    minus_dm = (-low.diff()).where(-low.diff() > high.diff(), 0)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = (plus_dm.rolling(period).mean() / atr) * 100
    minus_di = (minus_dm.rolling(period).mean() / atr) * 100
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(period).mean().iloc[-1] if not pd.isna(dx.rolling(period).mean().iloc[-1]) else 20


def get_market():
    try:
        nifty = yf.Ticker("^NSEI").history(period="20d")
        sma20 = nifty["Close"].rolling(20).mean().iloc[-1]
        return nifty["Close"].iloc[-1] > sma20 if not pd.isna(sma20) else True
    except:
        return True


print("Market:", "BULLISH" if get_market() else "BEARISH")
print("\nAnalyzing stocks:")
print("=" * 100)
print(f"{'Symbol':<15} {'Price':>8} {'SMA20':>8} {'RSI':>6} {'Vol':>6} {'ADX':>6} {'Mom3d':>7} {'Score':>6}")
print("-" * 100)

for sym in SYMBOLS:
    try:
        df = yf.Ticker(sym).history(period="60d")
        df = df.dropna(subset=['Close'])
        if len(df) < 30:
            continue
        
        close = df["Close"]
        volume = df["Volume"]
        price = close.iloc[-1]
        
        sma20 = close.rolling(20).mean().iloc[-1]
        
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = (100 - (100 / (1 + gain / loss))).iloc[-1]
        
        vol_ma = volume.rolling(20).mean().iloc[-1]
        vol_ratio = volume.iloc[-1] / vol_ma if vol_ma > 0 else 1
        
        adx = calculate_adx(df)
        
        mom3d = (price / close.shift(3).iloc[-1] - 1) * 100
        
        # Score
        s = 0
        if price > sma20: s += 1
        if vol_ratio > 1.2: s += 1
        if adx > 15: s += 1
        if mom3d > 1: s += 1
        if rsi < 70: s += 1
        
        if s >= 3:
            print(f"{sym:<15} {price:>8.0f} {sma20:>8.0f} {rsi:>6.1f} {vol_ratio:>6.2f} {adx:>6.1f} {mom3d:>7.1f} {s:>6}")
    except:
        pass