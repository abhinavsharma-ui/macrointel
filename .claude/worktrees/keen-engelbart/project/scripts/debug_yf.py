"""
Debug - Print RSI from live_paper_v7 historical data
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

print("Checking historical data from yfinance:")
print("=" * 60)

for sym in symbols:
    try:
        ticker = yf.Ticker(sym)
        df = ticker.history(period="200d", interval="1d")
        print(f"{sym}: {len(df)} days of data")
    except Exception as e:
        print(f"{sym}: Error - {e}")