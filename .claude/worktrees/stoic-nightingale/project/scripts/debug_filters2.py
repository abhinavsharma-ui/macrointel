"""
Debug - Show stock status at each filter
"""

import yfinance as yf
import pandas as pd
import numpy as np

SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "KOTAKBANK.NS", "BAJFINANCE.NS", "HINDUNILVR.NS", "ITC.NS",
    "TITAN.NS", "SUNPHARMA.NS", "MARUTI.NS", "M&M.NS", "NESTLEIND.NS",
    "ULTRACEMCO.NS", "ADANIPORTS.NS", "NTPC.NS", "POWERGRID.NS", "ONGC.NS"
]

def calc_adx(df, p=14):
    try:
        h, l, c = df["High"], df["Low"], df["Close"]
        pdm = h.diff().clip(lower=0)
        mdm = (-l.diff()).clip(lower=0)
        tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(p).mean()
        pdi = (pdm.rolling(p).mean() / atr) * 100
        mdi = (mdm.rolling(p).mean() / atr) * 100
        dx = (pdi - mdi).abs() / (pdi + mdi) * 100
        return dx.rolling(p).mean().iloc[-1] if len(dx) >= p else 20
    except:
        return 20

def get_market():
    try:
        n = yf.Ticker("^NSEI").history(period="15d")
        return n["Close"].iloc[-1] > n["Close"].iloc[-5]
    except:
        return True

print(f"Market: {'BULLISH' if get_market() else 'BEARISH'}\n")

for sym in SYMBOLS:
    try:
        df = yf.Ticker(sym).history(period="60d").dropna(subset=['Close'])
        if len(df) < 30: 
            print(f"{sym}: Not enough data")
            continue
        
        close = df["Close"]
        vol = df["Volume"]
        price = close.iloc[-1]
        
        # RSI
        d = close.diff()
        g = d.where(d > 0, 0).rolling(14).mean()
        l = (-d.where(d < 0, 0)).rolling(14).mean()
        rsi = (100 - (100 / (1 + g / l))).iloc[-1]
        
        sma20 = close.rolling(20).mean().iloc[-1]
        vratio = vol.iloc[-1] / vol.rolling(20).mean().iloc[-1] if len(vol) >= 20 else 1
        adx = calc_adx(df)
        mom3d = (price / close.shift(3).iloc[-1] - 1) * 100
        
        # Filters
        f1 = price > sma20
        f2 = mom3d >= 0.5
        f3 = rsi <= 75
        f4 = adx >= 10
        
        if f1 and f2 and f3 and f4:
            print(f"{sym}: OK - Price:{price:.0f} SMA20:{sma20:.0f} RSI:{rsi:.0f} "
                  f"Vol:{vratio:.1f}x ADX:{adx:.0f} Mom:{mom3d:.1f}%")
        else:
            flags = ""
            if not f1: flags += "P "
            if not f2: flags += "M "
            if not f3: flags += "R "
            if not f4: flags += "A "
            print(f"{sym}: FAIL - {flags} (price>{sma20}, mom>0.5, rsi<75, adx>10)")
            
    except Exception as e:
        print(f"{sym}: Error - {e}")