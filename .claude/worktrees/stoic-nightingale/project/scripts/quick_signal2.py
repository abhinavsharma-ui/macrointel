"""
Quick Signal Check - Using last valid price
"""

import os
import yfinance as yf
import pandas as pd

os.environ["FINNHUB_API_KEY"] = "d78vt3hr01qp0fl6mfjgd78vt3hr01qp0fl6mfk0"

symbols = [
    "AJANTPHARM.NS", "DRREDDY.NS", "JUBLFOOD.NS",
    "IRCTC.NS", "LALPATHLAB.NS", "CLEAN.NS", "RBA.NS",
    "SUNPHARMA.NS", "CYIENT.NS", "MUTHOOTFIN.NS",
    "MANAPPURAM.NS", "INDUSINDBK.NS", "AUBANK.NS",
    "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "TCS.NS", "SBIN.NS"
]

print("Checking for BUY signals (RSI < 38):")
print("=" * 60)

signals_found = 0

for sym in symbols:
    try:
        ticker = yf.Ticker(sym)
        df = ticker.history(period="200d", interval="1d")
        
        if df is None or len(df) < 30:
            continue
        
        # Drop rows with NaN close
        df = df.dropna(subset=['Close'])
        
        if len(df) < 30:
            continue
        
        close = df["Close"]
        volume = df["Volume"]
        price = close.iloc[-1]  # Use last valid price
        
        if pd.isna(price) or price <= 0:
            continue
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_current = rsi.iloc[-1]
        
        if pd.isna(rsi_current):
            continue
        
        # Volume
        vol_ma20 = volume.rolling(20).mean().iloc[-1]
        if pd.isna(vol_ma20) or vol_ma20 <= 0:
            vol_ma20 = volume.mean()
        volume_ratio = volume.iloc[-1] / vol_ma20 if vol_ma20 > 0 else 1
        
        # Signal check
        if rsi_current < 38:
            signals_found += 1
            stop_loss = round(price * 0.94, 2)
            take_profit = round(price * 1.18, 2)
            print(f"*** BUY: {sym} ***")
            print(f"    Price: Rs{price:.0f}")
            print(f"    RSI: {rsi_current:.1f}")
            print(f"    Volume: {volume_ratio:.2f}x")
            print(f"    Stop: Rs{stop_loss}, Target: Rs{take_profit}")
            print()
    
    except Exception as e:
        pass

print(f"\nTotal BUY signals found: {signals_found}")