"""
Live Momentum Trading - Relaxed Conditions
=============================================
Finding only 1 stock met all conditions. Let's relax to find more trades.
"""

import os
import logging
import yfinance as yf
import pandas as pd
import requests
import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

API_KEY = "d78vt3hr01qp0fl6mfjgd78vt3hr01qp0fl6mfk0"

SYMBOLS = [
    "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "TCS.NS", "SBIN.NS",
    "BAJFINANCE.NS", "TITAN.NS", "SUNPHARMA.NS", "MARUTI.NS", "M&M.NS",
    "KOTAKBANK.NS", "ICICIBANK.NS", "ADANIPORTS.NS", "HINDUNILVR.NS",
    "CIPLA.NS", "NESTLEIND.NS", "ULTRACEMCO.NS", "NTPC.NS", "POWERGRID.NS",
    "IRCTC.NS", "TATAMOTORS.NS", "ADANIENT.NS", "VEDL.NS", "TATASTEEL.NS",
    "COALINDIA.NS", "ONGC.NS", "ITC.NS", "HCLTECH.NS", "WIPRO.NS"
]


def get_live_price(symbol: str) -> float:
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={API_KEY}"
        resp = requests.get(url, timeout=3).json()
        if "c" in resp and resp["c"] > 0:
            return resp["c"]
    except:
        pass
    return None


def get_signals() -> list:
    """Relaxed momentum signals - just price above 20-day MA + positive momentum"""
    signals = []
    
    for sym in SYMBOLS:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period="60d", interval="1d")
            df = df.dropna(subset=['Close'])
            
            if len(df) < 30:
                continue
            
            close = df["Close"]
            price = close.iloc[-1]
            
            sma20 = close.rolling(20).mean().iloc[-1]
            if pd.isna(sma20):
                continue
            
            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_current = rsi.iloc[-1]
            
            # Momentum
            returns_3d = (price / close.shift(3).iloc[-1] - 1) * 100 if len(close) >= 4 else 0
            
            # RELAXED: Just price above SMA20 + positive momentum + not overbought
            if price > sma20 and returns_3d > 1 and rsi_current < 70:
                stop_loss = round(sma20 * 0.95, 2)
                risk = price - stop_loss
                
                if risk > 0:
                    # Rs500 risk (10%)
                    size = int(500 / risk)
                    
                    if size > 0:
                        live_price = get_live_price(sym)
                        exec_price = live_price if live_price else price
                        
                        signals.append({
                            "symbol": sym,
                            "price": exec_price,
                            "rsi": rsi_current,
                            "momentum": returns_3d,
                            "sma20": sma20,
                            "stop_loss": stop_loss,
                            "target": round(exec_price * 1.10, 2),
                            "size": size,
                            "risk": risk
                        })
        
        except:
            pass
    
    # Sort by momentum
    signals.sort(key=lambda x: x['momentum'], reverse=True)
    return signals


# ============== MAIN ==============

print("=" * 70)
print("LIVE MOMENTUM TRADING - RELAXED")
print("=" * 70)
print("\nScanning for momentum signals...\n")

signals = get_signals()

print(f"Found {len(signals)} signals:\n")

broker_cash = 5000
positions = []
executed = 0

for sig in signals:
    if executed >= 2:
        break
    
    cost = sig['price'] * sig['size'] + 20
    if broker_cash >= cost:
        broker_cash -= cost
        positions.append(sig)
        executed += 1
        print(f"*** BOUGHT {sig['symbol']} ***")
        print(f"  Price: Rs{sig['price']:.0f}, Size: {sig['size']}")
        print(f"  Stop: Rs{sig['stop_loss']}, Target: Rs{sig['target']}")
        print(f"  RSI: {sig['rsi']:.0f}, Momentum: {sig['momentum']:.1f}%")
        print()

print(f"\n=== PORTFOLIO ===")
print(f"Cash: Rs{broker_cash:.2f}")
print(f"Positions: {len(positions)}")

if positions:
    pos_value = sum(p['price'] * p['size'] for p in positions)
    total = broker_cash + pos_value
    print(f"Position Value: Rs{pos_value:.2f}")
    print(f"Total Value: Rs{total:.2f}")
    print(f"\nOpen Positions:")
    for p in positions:
        print(f"  {p['symbol']}: {p['size']} shares @ Rs{p['price']:.0f}")
        print(f"    Stop: Rs{p['stop_loss']}, Target: Rs{p['target']}")