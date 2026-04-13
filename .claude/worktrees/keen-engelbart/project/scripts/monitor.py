"""
Paper Trading Monitor
=====================
Monitor open position ADANIPORTS.NS with real-time updates.
"""

import time
import yfinance as yf
import requests

API_KEY = "d78vt3hr01qp0fl6mfjgd78vt3hr01qp0fl6mfk0"

POSITION = {
    "symbol": "ADANIPORTS.NS",
    "shares": 3,
    "entry": 1447,
    "stop": 1309.01,
    "target": 1592.14
}

def get_live_price(symbol):
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={API_KEY}"
        resp = requests.get(url, timeout=3).json()
        if "c" in resp and resp["c"] > 0:
            return resp["c"]
    except:
        pass
    return None


print("=" * 70)
print("PAPER TRADING MONITOR")
print("=" * 70)
print(f"Position: {POSITION['shares']} shares of {POSITION['symbol']}")
print(f"Entry: Rs{POSITION['entry']}, Stop: Rs{POSITION['stop']}, Target: Rs{POSITION['target']}")
print("\nMonitoring for 3 minutes (6 checks)...\n")

for i in range(6):
    live = get_live_price(POSITION["symbol"])
    yf_price = yf.Ticker(POSITION["symbol"]).history(period="1d")["Close"].iloc[-1]
    price = live if live else yf_price
    
    pnl = (price - POSITION["entry"]) * POSITION["shares"]
    pnl_pct = (price - POSITION["entry"]) / POSITION["entry"] * 100
    
    status = "HOLDING"
    if price <= POSITION["stop"]:
        status = "STOPPED OUT"
    elif price >= POSITION["target"]:
        status = "TARGET HIT"
    
    print(f"Check {i+1}: Price Rs{price:.0f} | PnL: Rs{pnl:.0f} ({pnl_pct:+.1f}%) | {status}")
    
    if status != "HOLDING":
        print(f"\n*** TRADE {status} ***")
        break
    
    time.sleep(30)

if i == 5:
    print("\nMonitor session ended. Position still open.")