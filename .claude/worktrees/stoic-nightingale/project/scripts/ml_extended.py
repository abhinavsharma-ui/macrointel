"""
Enhanced ML Model with More Data + More Stocks
===============================================
1. Train ML model on more stocks and longer history
2. Add more NSE stocks to scan
3. Better feature engineering
"""

import os
import logging
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from datetime import timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

API_KEY = "d78vt3hr01qp0fl6mfjgd78vt3hr01qp0fl6mfk0"

# Extended stock list - more stocks to scan
SYMBOLS = [
    # Large Cap
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "KOTAKBANK.NS", "BAJFINANCE.NS", "HINDUNILVR.NS", "ITC.NS",
    "TITAN.NS", "SUNPHARMA.NS", "MARUTI.NS", "M&M.NS", "NESTLEIND.NS",
    "ULTRACEMCO.NS", "ADANIPORTS.NS", "NTPC.NS", "POWERGRID.NS", "ONGC.NS",
    # Mid Cap
    "VEDL.NS", "TATASTEEL.NS", "COALINDIA.NS", "CIPLA.NS", "HCLTECH.NS",
    "WIPRO.NS", "IRCTC.NS", "DIVISLAB.NS", "BHARTIARTL.NS", "KERNAL",
    "TATAMOTORS.NS", "L&T.NS", "TECHM.NS", "INDUSINDBK.NS", "AXISBANK.NS",
    # More mid/small
    "UPL.NS", "BAYERCROP.NS", "GODREJPROP.NS", "PIDILITIND.NS", "CONCOR.NS",
    "IOC.NS", "BPCL.NS", "HINDPETRO.NS", "GAIL.NS", "CHENNPETRO.NS",
    "MOTHERSON.NS", "GRASIM.NS", "LTIM.NS", "APOLLOTYRE.NS", "BANDHANBNK.NS"
]


def calculate_adx(df, period=14):
    try:
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
    except:
        return 20


def get_market():
    try:
        nifty = yf.Ticker("^NSEI").history(period="30d")
        sma20 = nifty["Close"].rolling(20).mean().iloc[-1]
        return nifty["Close"].iloc[-1] > sma20 if not pd.isna(sma20) else True
    except:
        return True


def get_nifty_momentum():
    """Get Nifty 50 momentum for market filter"""
    try:
        nifty = yf.Ticker("^NSEI").history(period="20d")
        price = nifty["Close"].iloc[-1]
        past = nifty["Close"].iloc[-10]
        return (price / past - 1) * 100
    except:
        return 0


def extract_features(df: pd.DataFrame) -> dict:
    """Extract comprehensive features for ML"""
    if df is None or len(df) < 50:
        return None
    
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    price = close.iloc[-1]
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = (100 - (100 / (1 + rs))).iloc[-1]
    rsi_ma = rsi.rolling(5).mean().iloc[-1]  # RSI trend
    
    # Moving averages
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else sma50
    
    # Price position
    price_vs_sma20 = (price - sma20) / sma20 * 100 if sma20 else 0
    price_vs_sma50 = (price - sma50) / sma50 * 100 if sma50 else 0
    above_sma200 = 1 if (not pd.isna(sma200) and price > sma200) else 0
    
    # Volume
    vol_ma20 = volume.rolling(20).mean().iloc[-1]
    vol_ratio = volume.iloc[-1] / vol_ma20 if vol_ma20 > 0 else 1
    vol_trend = (volume.iloc[-1] - volume.iloc[-5]) / volume.iloc[-5] if volume.iloc[-5] > 0 else 0
    
    # ADX
    adx = calculate_adx(df)
    
    # Momentum
    returns = {}
    for d in [1, 3, 5, 10, 20]:
        if len(close) >= d + 1:
            returns[d] = (price / close.shift(d).iloc[-1] - 1) * 100
        else:
            returns[d] = 0
    
    # Volatility
    atr = (high - low).rolling(14).mean().iloc[-1]
    atr_pct = atr / price * 100 if price > 0 else 2
    
    return {
        "price": price,
        "rsi": rsi,
        "rsi_trend": rsi_ma - rsi if not pd.isna(rsi_ma) else 0,
        "sma20": sma20,
        "sma50": sma50,
        "price_vs_sma20": price_vs_sma20,
        "price_vs_sma50": price_vs_sma50,
        "above_sma200": above_sma200,
        "volume_ratio": vol_ratio,
        "volume_trend": vol_trend,
        "adx": adx,
        "momentum_1d": returns[1],
        "momentum_3d": returns[3],
        "momentum_5d": returns[5],
        "momentum_10d": returns[10],
        "momentum_20d": returns[20],
        "atr_pct": atr_pct
    }


def train_ml_model():
    """Train ML model with extensive historical data"""
    print("Training ML model on extended dataset...")
    
    X_data = []
    y_data = []
    
    # Train on more stocks
    train_symbols = SYMBOLS[:30]  # Use 30 stocks for training
    
    for sym in train_symbols:
        try:
            df = yf.Ticker(sym).history(period="2y", interval="1d")
            df = df.dropna(subset=['Close'])
            
            if len(df) < 150:
                continue
            
            # Generate samples
            for i in range(60, len(df) - 20):
                subset = df.iloc[:i]
                features = extract_features(subset)
                
                if features is None:
                    continue
                
                # Check if signal conditions met
                price = features["price"]
                sma20 = features["sma20"]
                mom = features["momentum_3d"]
                vol = features["volume_ratio"]
                adx = features["adx"]
                rsi = features["rsi"]
                
                signal = (price > sma20 and mom > 1.5 and vol > 0.8 and 
                         adx > 15 and rsi < 70)
                
                if signal:
                    # Check outcome (10% target hit in 20 days)
                    future = df.iloc[i:i+20]
                    if len(future) >= 10:
                        entry_price = df.iloc[i]["Close"]
                        high_20d = future["High"].max()
                        low_20d = future["Low"].min()
                        stop = sma20 * 0.95
                        
                        # Win: target hit before stop
                        if high_20d >= entry_price * 1.10 and high_20d > low_20d:
                            label = 1
                        elif low_20d <= stop:
                            label = 0
                        else:
                            continue  # Unclear outcome
                        
                        X_data.append([
                            features["rsi"],
                            features["rsi_trend"],
                            features["volume_ratio"],
                            features["volume_trend"],
                            features["adx"],
                            features["momentum_1d"],
                            features["momentum_3d"],
                            features["momentum_5d"],
                            features["price_vs_sma20"],
                            features["price_vs_sma50"],
                            features["above_sma200"],
                            features["atr_pct"]
                        ])
                        y_data.append(label)
        
        except Exception as e:
            pass
    
    if len(X_data) < 50:
        print(f"Not enough training data ({len(X_data)} samples), using fallback scoring")
        return None, None
    
    X = np.array(X_data)
    y = np.array(y_data)
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train
    model = GradientBoostingClassifier(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_scaled, y)
    
    # Evaluate
    train_acc = model.score(X_scaled, y)
    win_rate = y.mean()
    
    print(f"Training samples: {len(X_data)}")
    print(f"Training accuracy: {train_acc:.1%}")
    print(f"Historical win rate: {win_rate:.1%}")
    
    return model, scaler


def predict_success(model, scaler, features: dict) -> float:
    """Predict trade success probability"""
    if model is None:
        # Fallback scoring
        score = 0
        if features["rsi"] < 55: score += 1
        if features["volume_ratio"] > 1.2: score += 1
        if features["adx"] > 25: score += 1
        if features["momentum_3d"] > 4: score += 1
        if features["price_vs_sma20"] > 3: score += 1
        if features["above_sma200"]: score += 1
        return score / 6
    
    X = np.array([[
        features["rsi"],
        features["rsi_trend"],
        features["volume_ratio"],
        features["volume_trend"],
        features["adx"],
        features["momentum_1d"],
        features["momentum_3d"],
        features["momentum_5d"],
        features["price_vs_sma20"],
        features["price_vs_sma50"],
        features["above_sma200"],
        features["atr_pct"]
    ]])
    
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0][1]
    return prob


def get_signals(model, scaler) -> list:
    """Get enhanced signals with ML scoring"""
    market_bullish = get_market()
    nifty_mom = get_nifty_momentum()
    signals = []
    
    for sym in SYMBOLS:
        try:
            df = yf.Ticker(sym).history(period="60d")
            df = df.dropna(subset=['Close'])
            if len(df) < 30:
                continue
            
            features = extract_features(df)
            if features is None:
                continue
            
            price = features["price"]
            sma20 = features["sma20"]
            
            # === FILTERS ===
            if price <= sma20:
                continue
            if not market_bullish:
                continue
            if features["momentum_3d"] <= 1:
                continue
            if features["volume_ratio"] < 0.7:
                continue
            if features["adx"] < 12:
                continue
            if features["rsi"] >= 72:
                continue
            
            # Calculate position
            stop_loss = round(sma20 * 0.95, 2)
            risk = price - stop_loss
            
            if risk > 0 and risk < price * 0.15:
                size = int(500 / risk)
                
                if size > 0:
                    # ML prediction
                    ml_prob = predict_success(model, scaler, features)
                    
                    signals.append({
                        "symbol": sym,
                        "features": features,
                        "price": price,
                        "sma20": sma20,
                        "stop_loss": stop_loss,
                        "target": round(price * 1.10, 2),
                        "size": size,
                        "ml_prob": ml_prob,
                        "score": ml_prob
                    })
        
        except:
            pass
    
    signals.sort(key=lambda x: x["score"], reverse=True)
    return signals


# ============== MAIN ==============

print("=" * 70)
print("ENHANCED ML TRADING - EXTENDED STOCK LIST")
print("=" * 70)

# Market analysis
market_bullish = get_market()
nifty_mom = get_nifty_momentum()
print(f"\nMarket: {'BULLISH' if market_bullish else 'BEARISH'}")
print(f"Nifty Momentum (10d): {nifty_mom:.1f}%")
print(f"Stocks to scan: {len(SYMBOLS)}")

# Train ML
model, scaler = train_ml_model()

# Get signals
print("\nScanning for signals...")
signals = get_signals(model, scaler)

print(f"\nFound {len(signals)} signals")

# Top signals
if signals:
    print("\nTop 10 signals:")
    print("-" * 85)
    print(f"{'Symbol':<15} {'ML Prob':>8} {'RSI':>6} {'Vol':>6} {'ADX':>6} {'Mom3d':>7} {'Score':>6}")
    print("-" * 85)
    for s in signals[:10]:
        f = s["features"]
        print(f"{s['symbol']:<15} {s['ml_prob']:>7.0%} {f['rsi']:>6.1f} {f['volume_ratio']:>6.2f} "
              f"{f['adx']:>6.1f} {f['momentum_3d']:>7.1f} {s['score']:>6.2f}")

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
    
    # ML probability threshold
    if sig["ml_prob"] < 0.35:
        continue
    
    cost = sig['price'] * sig['size'] + 20
    if broker_cash >= cost and sig['size'] > 0:
        broker_cash -= cost
        positions.append(sig)
        executed += 1
        f = sig["features"]
        print(f"\n*** BOUGHT {sig['symbol']} ***")
        print(f"  Price: Rs{sig['price']:.0f}, Size: {sig['size']} shares")
        print(f"  Stop: Rs{sig['stop_loss']}, Target: Rs{sig['target']}")
        print(f"  RSI: {f['rsi']:.0f}, Vol: {f['volume_ratio']:.2f}x, ADX: {f['adx']:.0f}")
        print(f"  ML Probability: {sig['ml_prob']:.0%}")

# If no trades, relax
if executed == 0:
    print("\nNo high-probability trades. Taking best available...")
    for sig in signals[:1]:
        cost = sig['price'] * sig['size'] + 20
        if broker_cash >= cost and sig['size'] > 0:
            broker_cash -= cost
            positions.append(sig)
            executed += 1
            print(f"\n*** BOUGHT {sig['symbol']} *** (relaxed)")
            print(f"  ML Probability: {sig['ml_prob']:.0%}")

print(f"\n=== PORTFOLIO ===")
print(f"Cash: Rs{broker_cash:.2f}")
print(f"Positions: {len(positions)}")

if positions:
    pos_value = sum(p['price'] * p['size'] for p in positions)
    print(f"Total Value: Rs{broker_cash + pos_value:.2f}")
    print(f"\nOpen Positions:")
    for p in positions:
        print(f"  {p['symbol']}: {p['size']} @ Rs{p['price']:.0f}")
        print(f"    Stop:Rs{p['stop_loss']}, Target:Rs{p['target']}, ML:{p['ml_prob']:.0%}")