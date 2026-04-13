"""
Debug - Check yfinance data structure
"""

import yfinance as yf

sym = "RELIANCE.NS"
ticker = yf.Ticker(sym)
df = ticker.history(period="5d", interval="1d")

print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nLast 3 rows:")
print(df.tail(3))
print(f"\nClose price type: {type(df['Close'].iloc[-1])}")
print(f"Close price value: {df['Close'].iloc[-1]}")