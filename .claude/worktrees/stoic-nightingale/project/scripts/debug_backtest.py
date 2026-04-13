"""
Debug - Show why backtest is generating 0 samples
"""

import yfinance as yf
import pandas as pd
import numpy as np

SYMBOLS = ["TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "RELIANCE.NS", "INFY.NS"]

print("Debugging backtest data collection...\n")

for sym in SYMBOLS:
    try:
        df = yf.Ticker(sym).history(period="2y", interval="1d")
        df = df.dropna(subset=['Close'])
        print(f"\n{sym}: {len(df)} days")
        
        if len(df) < 300:
            print(f"  Skipping - not enough data")
            continue
        
        # Check a few days
        for i in [60, 100, 200]:
            if i >= len(df) - 20:
                continue
            
            close = df.iloc[:i]["Close"]
            price = close.iloc[-1]
            sma20 = close.rolling(20).mean().iloc[-1]
            
            d = close.diff()
            g = d.where(d > 0, 0).rolling(14).mean()
            l = (-d.where(d < 0, 0)).rolling(14).mean()
            rsi = (100 - (100 / (1 + g / l))).iloc[-1]
            
            mom3d = (price / close.shift(3).iloc[-1] - 1) * 100
            
            # Check each condition
            cond1 = price > sma20
            cond2 = mom3d > 0.5
            cond3 = rsi < 70
            
            all_ok = cond1 and cond2 and cond3
            
            print(f"  Day {i}: price>{sma20:.0f}({cond1}), mom>{mom3d:.1f}%({cond2}), rsi<{rsi:.0f}({cond3}) = {all_ok}")
    
    except Exception as e:
        print(f"{sym}: Error - {e}")