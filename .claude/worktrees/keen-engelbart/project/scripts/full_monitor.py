"""
Portfolio Monitor - All positions
"""

import yfinance as yf
import requests
import json
from pathlib import Path
from datetime import datetime

API_KEY = "d78vt3hr01qp0fl6mfjgd78vt3hr01qp0fl6mfk0"

def get_price(sym):
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol={sym}&token={API_KEY}"
        r = requests.get(url, timeout=3).json()
        if r.get("c", 0) > 0: return r["c"]
    except: pass
    return None

# All current positions
POSITIONS = [
    {"symbol": "ICICIBANK.NS", "shares": 2, "entry": 1281, "stop": 1207, "target": 1384},
    {"symbol": "ASIANPAINT.NS", "shares": 1, "entry": 2270, "stop": 2128, "target": 2451},
    {"symbol": "ADANIPORTS.NS", "shares": 3, "entry": 1447, "stop": 1309, "target": 1592},
    # New trades
    {"symbol": "UPL.NS", "shares": 1, "entry": 642, "stop": 597, "target": 693},
    {"symbol": "MOTHERSON.NS", "shares": 6, "entry": 117, "stop": 108, "target": 126},
    {"symbol": "TATASTEEL.NS", "shares": 3, "entry": 205, "stop": 190, "target": 221},
    {"symbol": "BPCL.NS", "shares": 4, "entry": 297, "stop": 283, "target": 321},
    {"symbol": "HINDPETRO.NS", "shares": 2, "entry": 358, "stop": 334, "target": 387},
    {"symbol": "CANBK.NS", "shares": 6, "entry": 138, "stop": 129, "target": 149}
]

print("=" * 70)
print("COMPLETE PORTFOLIO MONITOR")
print("=" * 70)

total_pnl = 0
total_value = 248.14  # Cash from last run

for i, p in enumerate(POSITIONS, 1):
    live = get_price(p["symbol"])
    try:
        yf_price = yf.Ticker(p["symbol"]).history(period="1d")["Close"].iloc[-1]
    except:
        yf_price = p["entry"]
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
    total_value += price * p["shares"]
    
    print(f"\n{i}. {p['symbol']}")
    print(f"   Entry: Rs{p['entry']}, Now: Rs{price:.0f}")
    print(f"   P&L: Rs{pnl:.0f} ({pnl_pct:+.1f}%)")
    print(f"   Stop: Rs{p['stop']}, Target: Rs{p['target']}")
    print(f"   Status: {status}")

invested = sum(p["entry"] * p["shares"] for p in POSITIONS)

print(f"\n{'='*70}")
print("SUMMARY")
print("=" * 70)
print(f"Cash: Rs248.14")
print(f"Invested: Rs{invested:.2f}")
print(f"Total P&L: Rs{total_pnl:.2f}")
print(f"Total Value: Rs{total_value:.2f}")
print(f"Return: {(total_value - 5000) / 5000 * 100:.1f}%")

# Count winners/losers
print(f"\nPositions: {len(POSITIONS)}")
wins = sum(1 for p in POSITIONS if (get_price(p["symbol"]) or p["entry"]) > p["entry"])
print(f"Winners: {wins}, Losers: {len(POSITIONS) - wins}")

# ============================================
# SAVE FOR DASHBOARD
# ============================================
portfolio_data = {
    "broker": "Paper Trading",
    "summary": {
        "portfolio_value": round(total_value, 2),
        "cash": round(248.14, 2),
        "holdings_value": round(invested, 2),
        "open_positions": len(POSITIONS),
        "total_return_pct": round(((total_value - 5000) / 5000) * 100, 2),
        "day_pnl": round(total_pnl, 2)
    },
    "positions": [
        {
            "symbol": p["symbol"],
            "quantity": p["shares"],
            "avg_cost": p["entry"],
            "current_price": p["entry"],
            "unrealized_pnl": round(((get_price(p["symbol"]) or p["entry"]) - p["entry"]) * p["shares"], 2),
            "stop_loss": p["stop"],
            "target": p["target"],
            "price_change_pct": round(((get_price(p["symbol"]) or p["entry"]) - p["entry"]) / p["entry"] * 100, 2)
        }
        for p in POSITIONS
    ],
    "updated_at": datetime.now().isoformat()
}

# Save to data folder
data_path = Path(__file__).parent.parent / "data"
(data_path / "live_portfolio.json").write_text(json.dumps(portfolio_data, indent=2))
print(f"\n[Dashboard] Saved portfolio to live_portfolio.json")