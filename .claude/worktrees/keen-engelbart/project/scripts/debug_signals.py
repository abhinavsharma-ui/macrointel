"""
Debug - Check all stocks for signals
"""

import yfinance as yf
import pandas as pd

SYMBOLS = [
    "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "TCS.NS", "SBIN.NS",
    "BAJFINANCE.NS", "TITAN.NS", "SUNPHARMA.NS", "MARUTI.NS", "M&M.NS",
    "KOTAKBANK.NS", "ICICIBANK.NS", "ADANIPORTS.NS", "HINDUNILVR.NS",
    "CIPLA.NS", "NESTLEIND.NS", "ULTRACEMCO.NS", "NTPC.NS", "POWERGRID.NS"
]

print("Stock Analysis:")
print("=" * 80)

for sym in SYMBOLS:
    try:
        ticker = yf.Ticker(sym)
        df = ticker.history(period="100d", interval="1d")
        df = df.dropna(subset=['Close'])
        
        if len(df) < 50:
            print(f"{sym}: Not enough data")
            continue
        
        close = df["Close"]
        price = close.iloc[-1]
        
        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]
        
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_current = rsi.iloc[-1]
        
        returns_5d = (price / close.shift(5).iloc[-1] - 1) * 100
        
        # Check conditions
        cond1 = price > sma20 if not pd.isna(sma20) else False
        cond2 = sma20 > sma50 if not pd.isna(sma50) else False
        cond3 = returns_5d > 2
        cond4 = rsi_current < 65
        
        all_cond = cond1 and cond2 and cond3 and cond4
        
        flag = " <<< SIGNAL" if all_cond else ""
        print(f"{sym}: P={price:.0f} SMA20={sma20:.0f} SMA50={sma50:.0f} "
              f"Mom={returns_5d:.1f}% RSI={rsi_current:.0f} [{cond1} {cond2} {cond3} {cond4}]{flag}")
    except Exception as e:
        print(f"{sym}: Error")