"""
Quick RSI Check - More volatile small/mid caps
"""

import yfinance as yf

symbols = [
    # More volatile mid/small cap
    "IRCTC.NS", "LALPATHLAB.NS", "FLUORO.NS", "CLEAN.NS", "RBA.NS",
    "LTI.NS", "MPHASIS.NS", "PERSISTENT.NS", "COFORGE.NS", "CYIENT.NS",
    "AJANTPHARM.NS", "SUNPHARMA.NS", "DRREDDY.NS", "CADILAHC.NS",
    "JUBLFOOD.NS", "DELHIVERY.NS", "RAZIL.NS", "MUTHOOTFIN.NS",
    "MANAPPURAM.NS", "BHARATFIN.NS", "INDUSINDBK.NS", "AUBANK.NS"
]

print("Current RSI values (RSI < 40 = potential signal):")
print("=" * 60)

for sym in symbols:
    try:
        ticker = yf.Ticker(sym)
        df = ticker.history(period="30d", interval="1d")
        
        if len(df) < 20:
            print(f"{sym}: No data")
            continue
            
        close = df["Close"]
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_current = rsi.iloc[-1]
        
        flag = " <<< BUY SIGNAL" if rsi_current < 38 else ""
        print(f"{sym}: RSI={rsi_current:.1f}{flag}")
    except Exception as e:
        print(f"{sym}: Error")