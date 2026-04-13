"""
Fixed Live Momentum Trading
"""

import os
import logging
import yfinance as yf
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

API_KEY = "d78vt3hr01qp0fl6mfjgd78vt3hr01qp0fl6mfk0"

SYMBOLS = [
    "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "TCS.NS", "SBIN.NS",
    "BAJFINANCE.NS", "TITAN.NS", "SUNPHARMA.NS", "MARUTI.NS", "M&M.NS",
    "KOTAKBANK.NS", "ICICIBANK.NS", "ADANIPORTS.NS", "HINDUNILVR.NS",
    "CIPLA.NS", "NESTLEIND.NS", "ULTRACEMCO.NS", "NTPC.NS", "POWERGRID.NS",
    "IRCTC.NS", "VEDL.NS", "TATASTEEL.NS", "COALINDIA.NS", "ONGC.NS",
    "ITC.NS", "HCLTECH.NS", "WIPRO.NS"
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
            if pd.isna(sma20) or pd.isna(price):
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
            
            # Signal: price above SMA20 + positive momentum + not overbought
            if price > sma20 and returns_3d > 1 and rsi_current < 70:
                stop_loss = round(sma20 * 0.95, 2)
                risk = price - stop_loss
                
                if risk > 0 and risk < price * 0.15:  # Reasonable risk
                    size = int(500 / risk)
                    
                    if size > 0:
                        signals.append({
                            "symbol": sym,
                            "price": price,
                            "rsi": rsi_current,
                            "momentum": returns_3d,
                            "sma20": sma20,
                            "stop_loss": stop_loss,
                            "target": round(price * 1.10, 2),
                            "size": size,
                            "risk": risk
                        })
        
        except:
            pass
    
    signals.sort(key=lambda x: x['momentum'], reverse=True)
    return signals


# ============== MAIN ==============

print("=" * 70)
print("LIVE MOMENTUM TRADING")
print("=" * 70)
print("\nScanning for momentum signals...\n")

signals = get_signals()

print(f"Found {len(signals)} signals\n")

broker_cash = 5000
positions = []
executed = 0

for sig in signals:
    if executed >= 2:
        break
    
    cost = sig['price'] * sig['size'] + 20
    if broker_cash >= cost and sig['size'] > 0:
        broker_cash -= cost
        positions.append(sig)
        executed += 1
        print(f"*** BOUGHT {sig['symbol']} ***")
        print(f"  Price: Rs{sig['price']:.0f}, Size: {sig['size']} shares")
        print(f"  Stop: Rs{sig['stop_loss']}, Target: Rs{sig['target']}")
        print(f"  RSI: {sig['rsi']:.0f}, Momentum: {sig['momentum']:.1f}%")
        print()

if executed == 0:
    print("No trades executed.\n")

print(f"=== PORTFOLIO ===")
print(f"Cash: Rs{broker_cash:.2f}")
print(f"Positions: {len(positions)}")

if positions:
    pos_value = sum(p['price'] * p['size'] for p in positions)
    print(f"Position Value: Rs{pos_value:.2f}")
    print(f"Total Value: Rs{broker_cash + pos_value:.2f}")
    print(f"\nOpen Positions:")
    for p in positions:
        print(f"  {p['symbol']}: {p['size']} @ Rs{p['price']:.0f} | Stop:Rs{p['stop_loss']} Target:Rs{p['target']}")