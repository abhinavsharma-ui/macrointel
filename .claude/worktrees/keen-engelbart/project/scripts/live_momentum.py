"""
Live Momentum Trading - Higher Risk
====================================
Uses momentum signals with increased position sizing.
"""

import os
import logging
import yfinance as yf
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

API_KEY = "d78vt3hr01qp0fl6mfjgd78vt3hr01qp0fl6mfk0"

# More stocks to scan
SYMBOLS = [
    "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "TCS.NS", "SBIN.NS",
    "BAJFINANCE.NS", "TITAN.NS", "SUNPHARMA.NS", "MARUTI.NS", "M&M.NS",
    "KOTAKBANK.NS", "ICICIBANK.NS", "ADANIPORTS.NS", "HINDUNILVR.NS",
    "CIPLA.NS", "NESTLEIND.NS", "ULTRACEMCO.NS", "NTPC.NS", "POWERGRID.NS"
]


def get_live_price(symbol: str) -> float:
    """Get live price from Finnhub."""
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={API_KEY}"
        resp = requests.get(url, timeout=3).json()
        if "c" in resp and resp["c"] > 0:
            return resp["c"]
    except:
        pass
    return None


def get_momentum_signals() -> dict:
    """Scan for momentum buy signals."""
    signals = {}
    
    for sym in SYMBOLS:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period="100d", interval="1d")
            df = df.dropna(subset=['Close'])
            
            if len(df) < 50:
                continue
            
            close = df["Close"]
            volume = df["Volume"]
            price = close.iloc[-1]
            
            # Moving averages
            sma20 = close.rolling(20).mean().iloc[-1]
            sma50 = close.rolling(50).mean().iloc[-1]
            
            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_current = rsi.iloc[-1]
            
            # Momentum
            returns_5d = (price / close.shift(5).iloc[-1] - 1) if not pd.isna(close.shift(5).iloc[-1]) else 0
            
            # Volume
            vol_ma20 = volume.rolling(20).mean().iloc[-1]
            vol_ratio = volume.iloc[-1] / vol_ma20 if vol_ma20 > 0 else 1
            
            # SIGNAL: Price above 20-day MA + 20-day > 50-day + positive momentum + not overbought
            if (price > sma20 and not pd.isna(sma50) and sma20 > sma50 and 
                returns_5d > 0.02 and rsi_current < 65):
                
                # Stop below 20-day MA, target 10%
                stop_loss = round(sma20 * 0.95, 2)
                risk = price - stop_loss
                
                if risk > 0:
                    # Increased risk: Rs500 (10%)
                    size = int(500 / risk)
                    
                    signals[sym] = {
                        "price": price,
                        "rsi": rsi_current,
                        "momentum": returns_5d * 100,
                        "stop_loss": stop_loss,
                        "target": round(price * 1.10, 2),
                        "size": size,
                        "risk": risk,
                        "reason": f"Above SMA20, Mom:{returns_5d*100:.1f}%"
                    }
        
        except:
            pass
    
    return signals


# ============== MAIN ==============

print("=" * 70)
print("LIVE MOMENTUM TRADING")
print("=" * 70)
print("\nScanning for momentum signals...\n")

signals = get_momentum_signals()

print(f"Found {len(signals)} signals:\n")

# Sort by momentum
sorted_signals = sorted(signals.items(), key=lambda x: x[1]['momentum'], reverse=True)

broker_cash = 5000
positions = []
executed = 0

for sym, data in sorted_signals:
    if executed >= 2:  # Max 2 positions
        break
    
    # Check if we have enough cash
    cost = data['price'] * data['size'] + 20  # brokerage
    if broker_cash >= cost:
        broker_cash -= cost
        positions.append({
            "symbol": sym,
            "price": data['price'],
            "size": data['size'],
            "stop": data['stop_loss'],
            "target": data['target']
        })
        executed += 1
        print(f"*** EXECUTED: {sym} ***")
        print(f"  Price: Rs{data['price']:.0f}")
        print(f"  Size: {data['size']} shares")
        print(f"  Stop: Rs{data['stop_loss']}, Target: Rs{data['target']}")
        print(f"  Reason: {data['reason']}")
        print()

print(f"\n=== PORTFOLIO ===")
print(f"Cash: Rs{broker_cash:.2f}")
print(f"Positions: {len(positions)}")
total_value = broker_cash + sum(p['price'] * p['size'] for p in positions)
print(f"Total Value: Rs{total_value:.2f}")

if positions:
    print("\nOpen Positions:")
    for p in positions:
        print(f"  {p['symbol']}: {p['size']} @ Rs{p['price']:.0f}")