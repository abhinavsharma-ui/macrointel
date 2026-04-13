"""
Debug - Print RSI from historical data
"""

import yfinance as yf
import pandas as pd

symbols = [
    "AJANTPHARM.NS", "DRREDDY.NS", "JUBLFOOD.NS",
    "IRCTC.NS", "LALPATHLAB.NS", "CLEAN.NS", "RBA.NS",
    "SUNPHARMA.NS", "CYIENT.NS", "MUTHOOTFIN.NS",
    "MANAPPURAM.NS", "INDUSINDBK.NS", "AUBANK.NS",
    "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "TCS.NS", "SBIN.NS"
]

print("Current RSI from 200-day historical data:")
print("=" * 70)

for sym in symbols:
    try:
        ticker = yf.Ticker(sym)
        df = ticker.history(period="200d", interval="1d")
        
        if len(df) < 30:
            print(f"{sym}: Not enough data")
            continue
            
        close = df["Close"]
        volume = df["Volume"]
        price = close.iloc[-1]
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_current = rsi.iloc[-1]
        
        # Volume
        vol_ma20 = volume.rolling(20).mean().iloc[-1]
        vol_ratio = volume.iloc[-1] / vol_ma20 if vol_ma20 > 0 else 1
        
        signal = "<<< BUY" if rsi_current < 35 and vol_ratio > 1.0 else ""
        print(f"{sym}: RSI={rsi_current:.1f}, Vol={vol_ratio:.2f}x, Price={price:.0f} {signal}")
    except Exception as e:
        print(f"{sym}: Error - {e}")