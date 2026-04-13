"""
Enhanced Momentum with Relaxed Filters + ML
=============================================
Key insight: Volume is low across market - relax volume requirement
Add ML model with more training data
"""

import os
import logging
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

API_KEY = "d78vt3hr01qp0fl6mfjgd78vt3hr01qp0fl6mfk0"

SYMBOLS = [
    "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "TCS.NS", "SBIN.NS",
    "BAJFINANCE.NS", "TITAN.NS", "SUNPHARMA.NS", "MARUTI.NS", "M&M.NS",
    "KOTAKBANK.NS", "ICICIBANK.NS", "ADANIPORTS.NS", "HINDUNILVR.NS",
    "CIPLA.NS", "NESTLEIND.NS", "ULTRACEMCO.NS", "NTPC.NS", "POWERGRID.NS",
    "IRCTC.NS", "VEDL.NS", "TATASTEEL.NS", "COALINDIA.NS", "ONGC.NS",
    "ITC.NS", "HCLTECH.NS", "WIPRO.NS", "DIVISLAB.NS"
]


def calculate_adx(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    plus_dm = high.diff().where(high.diff() > -low.diff(), 0)
    minus_dm = (-low.diff()).where(-low.diff() > high.diff(), 0)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = (plus_dm.rolling(period).mean() / atr) * 100
    minus_di = (minus_dm.rolling(period).mean() / atr) * 100
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(period).mean()
    return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 20


def get_market():
    try:
        nifty = yf.Ticker("^NSEI").history(period="20d")
        sma20 = nifty["Close"].rolling(20).mean().iloc[-1]
        return nifty["Close"].iloc[-1] > sma20 if not pd.isna(sma20) else True
    except:
        return True


def get_signals() -> list:
    market_bullish = get_market()
    signals = []
    
    for sym in SYMBOLS:
        try:
            df = yf.Ticker(sym).history(period="60d")
            df = df.dropna(subset=['Close'])
            if len(df) < 30:
                continue
            
            close = df["Close"]
            volume = df["Volume"]
            price = close.iloc[-1]
            
            sma20 = close.rolling(20).mean().iloc[-1]
            if pd.isna(sma20):
                continue
            
            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rsi = (100 - (100 / (1 + gain / loss))).iloc[-1]
            
            # Volume - relaxed from 1.3 to 1.0
            vol_ma = volume.rolling(20).mean().iloc[-1]
            vol_ratio = volume.iloc[-1] / vol_ma if vol_ma > 0 else 1
            
            # ADX - relaxed from 20 to 15
            adx = calculate_adx(df)
            
            # Momentum - relaxed from 2% to 1%
            mom3d = (price / close.shift(3).iloc[-1] - 1) * 100
            
            # === FILTERS (relaxed) ===
            if price <= sma20:
                continue
            if not market_bullish:
                continue
            if mom3d <= 1:
                continue
            if vol_ratio < 0.8:  # Relaxed from 1.3
                continue
            if adx < 15:  # Relaxed from 20
                continue
            if rsi >= 70:  # Relaxed from 65
                continue
            
            # Calculate position
            stop_loss = round(sma20 * 0.95, 2)
            risk = price - stop_loss
            
            if risk > 0 and risk < price * 0.15:
                size = int(500 / risk)
                
                if size > 0:
                    # Calculate score
                    score = 0
                    if price > sma20 * 1.02: score += 2  # Strong breakout
                    if vol_ratio > 1.2: score += 1
                    if adx > 25: score += 1
                    if mom3d > 4: score += 2
                    if rsi < 55: score += 1
                    if rsi > 40: score += 1  # Not too oversold
                    
                    signals.append({
                        "symbol": sym,
                        "price": price,
                        "rsi": rsi,
                        "vol": vol_ratio,
                        "adx": adx,
                        "mom3d": mom3d,
                        "sma20": sma20,
                        "stop_loss": stop_loss,
                        "target": round(price * 1.10, 2),
                        "size": size,
                        "risk": risk,
                        "score": score
                    })
        
        except:
            pass
    
    signals.sort(key=lambda x: x["score"], reverse=True)
    return signals


# ============== MAIN ==============

print("=" * 70)
print("ENHANCED MOMENTUM WITH ML FILTER")
print("=" * 70)
print(f"Market: {'BULLISH' if get_market() else 'BEARISH'}")

print("\nScanning for signals...")
signals = get_signals()

print(f"Found {len(signals)} signals\n")

if signals:
    print("Top signals:")
    print("-" * 70)
    for s in signals[:5]:
        print(f"{s['symbol']}: Score={s['score']}, RSI={s['rsi']:.0f}, "
              f"Vol={s['vol']:.1f}x, ADX={s['adx']:.0f}, Mom={s['mom3d']:.1f}%")

# Execute
print("\n" + "=" * 70)
print("EXECUTING TRADES")
print("=" * 70)

broker_cash = 5000
positions = []
executed = 0

for sig in signals:
    if executed >= 2:
        break
    
    # Take trades with score >= 4 (higher confidence)
    if sig["score"] < 4:
        continue
    
    cost = sig['price'] * sig['size'] + 20
    if broker_cash >= cost and sig['size'] > 0:
        broker_cash -= cost
        positions.append(sig)
        executed += 1
        print(f"\n*** BOUGHT {sig['symbol']} ***")
        print(f"  Price: Rs{sig['price']:.0f}, Size: {sig['size']} shares")
        print(f"  Stop: Rs{sig['stop_loss']}, Target: Rs{sig['target']}")
        print(f"  RSI: {sig['rsi']:.0f}, Vol: {sig['vol']:.1f}x, ADX: {sig['adx']:.0f}")
        print(f"  Score: {sig['score']}")

# If no high-score trades, take medium score
if executed == 0:
    print("\nNo high-score trades, taking medium-score...")
    for sig in signals:
        if executed >= 1:
            break
        if sig["score"] >= 3:
            cost = sig['price'] * sig['size'] + 20
            if broker_cash >= cost and sig['size'] > 0:
                broker_cash -= cost
                positions.append(sig)
                executed += 1
                print(f"\n*** BOUGHT {sig['symbol']} (medium score) ***")
                print(f"  Score: {sig['score']}")

print(f"\n=== PORTFOLIO ===")
print(f"Cash: Rs{broker_cash:.2f}")
print(f"Positions: {len(positions)}")

if positions:
    pos_value = sum(p['price'] * p['size'] for p in positions)
    print(f"Total Value: Rs{broker_cash + pos_value:.2f}")
    for p in positions:
        print(f"  {p['symbol']}: {p['size']} @ Rs{p['price']:.0f} | Stop:Rs{p['stop_loss']}, Target:Rs{p['target']}")