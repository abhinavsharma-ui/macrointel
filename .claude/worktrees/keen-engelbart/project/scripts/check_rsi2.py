"""
Quick RSI Check - Updated list
"""

import yfinance as yf

symbols = [
    "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "TCS.NS", "SBIN.NS", "BAJFINANCE.NS",
    "TITAN.NS", "VEDL.NS", "JINDALSTEL.NS", "TATASTEEL.NS",
    "CIPLA.NS", "MARUTI.NS", "EICHERMOT.NS", "M&M.NS"
]

print("Current RSI values (RSI < 35 highlighted):")
print("=" * 60)

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
        
        flag = " <<< SIGNAL" if rsi_current < 35 else ""
        print(f"{sym}: RSI={rsi_current:.1f}{flag}")
    except Exception as e:
        print(f"{sym}: Error")