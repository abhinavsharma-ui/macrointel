"""
Quick RSI Check - What's the current RSI for each stock?
"""

import os
import yfinance as yf

symbols = ["RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "TCS.NS", "SBIN.NS", "BAJFINANCE.NS"]

print("Current RSI values:")
print("=" * 50)

for sym in symbols:
    try:
        ticker = yf.Ticker(sym)
        df = ticker.history(period="30d", interval="1d")
        
        close = df["Close"]
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_current = rsi.iloc[-1]
        
        # Volume
        vol_ma20 = df["Volume"].rolling(20).mean().iloc[-1]
        vol_ratio = df["Volume"].iloc[-1] / vol_ma20 if vol_ma20 > 0 else 1
        
        print(f"{sym}: RSI={rsi_current:.1f}, Vol={vol_ratio:.2f}x, Price=Rs{close.iloc[-1]:.0f}")
    except Exception as e:
        print(f"{sym}: Error - {e}")