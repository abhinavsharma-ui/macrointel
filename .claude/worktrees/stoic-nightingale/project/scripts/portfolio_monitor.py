"""
Portfolio Monitor - Track all positions
"""

import yfinance as yf
import requests

API_KEY = "d78vt3hr01qp0fl6mfjgd78vt3hr01qp0fl6mfk0"

# Current positions
POSITIONS = [
    {"symbol": "ICICIBANK.NS", "shares": 2, "entry": 1281, "stop": 1206.88, "target": 1383.8},
    {"symbol": "ASIANPAINT.NS", "shares": 1, "entry": 2270, "stop": 2127.77, "target": 2451.17},
    # Previous position
    {"symbol": "ADANIPORTS.NS", "shares": 3, "entry": 1447, "stop": 1309.01, "target": 1592.14}
]

def get_price(sym):
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol={sym}&token={API_KEY}"
        r = requests.get(url, timeout=3).json()
        if r.get("c", 0) > 0: return r["c"]
    except: pass
    return None

print("=" * 70)
print("PORTFOLIO MONITOR")
print("=" * 70)

total_pnl = 0
cash = 127.80  # Remaining cash from last trade

for i, p in enumerate(POSITIONS, 1):
    live = get_price(p["symbol"])
    yf_price = yf.Ticker(p["symbol"]).history(period="1d")["Close"].iloc[-1]
    price = live if live else yf_price
    
    pnl = (price - p["entry"]) * p["shares"]
    pnl_pct = (price - p["entry"]) / p["entry"] * 100
    
    status = "HOLDING"
    if price <= p["stop"]:
        status = "STOPPED"
        pnl = (p["stop"] - p["entry"]) * p["shares"]
    elif price >= p["target"]:
        status = "TARGET!"
        pnl = (p["target"] - p["entry"]) * p["shares"]
    
    total_pnl += pnl
    
    print(f"\n{i}. {p['symbol']}")
    print(f"   Entry: Rs{p['entry']}, Current: Rs{price:.0f}")
    print(f"   P&L: Rs{pnl:.0f} ({pnl_pct:+.1f}%)")
    print(f"   Stop: Rs{p['stop']}, Target: Rs{p['target']}")
    print(f"   Status: {status}")

invested = sum(p["entry"] * p["shares"] for p in POSITIONS)
print(f"\n{'='*70}")
print("SUMMARY")
print("=" * 70)
print(f"Cash: Rs{cash:.2f}")
print(f"Invested: Rs{invested:.2f}")
print(f"Total P&L: Rs{total_pnl:.2f}")
print(f"Total Value: Rs{cash + invested + total_pnl:.2f}")