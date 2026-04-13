"""
Enhanced Momentum Strategy with ML Filter
==========================================
Adds:
1. Volume confirmation (>1.3x)
2. ADX trend strength (>20)
3. Market direction filter (Nifty trend)
4. ML model to predict trade success
"""

import os
import logging
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

API_KEY = "d78vt3hr01qp0fl6mfjgd78vt3hr01qp0fl6mfk0"

# More stocks to scan
SYMBOLS = [
    "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "TCS.NS", "SBIN.NS",
    "BAJFINANCE.NS", "TITAN.NS", "SUNPHARMA.NS", "MARUTI.NS", "M&M.NS",
    "KOTAKBANK.NS", "ICICIBANK.NS", "ADANIPORTS.NS", "HINDUNILVR.NS",
    "CIPLA.NS", "NESTLEIND.NS", "ULTRACEMCO.NS", "NTPC.NS", "POWERGRID.NS",
    "IRCTC.NS", "VEDL.NS", "TATASTEEL.NS", "COALINDIA.NS", "ONGC.NS",
    "ITC.NS", "HCLTECH.NS", "WIPRO.NS", "TATA consumer.NS", "DIVISLAB.NS"
]


def calculate_adx(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate ADX (trend strength)"""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
    
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


def get_market_direction() -> bool:
    """Get Nifty market direction (True = bullish)"""
    try:
        nifty = yf.Ticker("^NSEI").history(period="20d", interval="1d")
        if len(nifty) > 10:
            nifty_sma20 = nifty["Close"].rolling(20).mean().iloc[-1]
            nifty_price = nifty["Close"].iloc[-1]
            return nifty_price > nifty_sma20 if not pd.isna(nifty_sma20) else True
    except:
        pass
    return True


def extract_features(df: pd.DataFrame) -> dict:
    """Extract all features for ML model"""
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
    rsi = 100 - (100 / (1 + rs))
    rsi_current = rsi.iloc[-1]
    
    # Moving averages
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    
    # Volume
    vol_ma20 = volume.rolling(20).mean().iloc[-1]
    volume_ratio = volume.iloc[-1] / vol_ma20 if vol_ma20 > 0 else 1
    
    # ADX
    adx = calculate_adx(df)
    
    # Momentum
    returns_3d = (price / close.shift(3).iloc[-1] - 1) * 100 if len(close) >= 4 else 0
    returns_5d = (price / close.shift(5).iloc[-1] - 1) * 100 if len(close) >= 6 else 0
    
    # Price position
    price_vs_sma20 = (price - sma20) / sma20 * 100 if sma20 else 0
    
    return {
        "price": price,
        "rsi": rsi_current,
        "sma20": sma20,
        "sma50": sma50,
        "volume_ratio": volume_ratio,
        "adx": adx,
        "momentum_3d": returns_3d,
        "momentum_5d": returns_5d,
        "price_vs_sma20": price_vs_sma20
    }


def get_enhanced_signals(market_bullish: bool = True) -> list:
    """Enhanced signals with all confirmations"""
    signals = []
    
    for sym in SYMBOLS:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period="60d", interval="1d")
            df = df.dropna(subset=['Close'])
            
            if len(df) < 30:
                continue
            
            features = extract_features(df)
            if features is None:
                continue
            
            price = features["price"]
            sma20 = features["sma20"]
            
            # === ENHANCED FILTER ===
            
            # 1. Price above 20-day MA
            if price <= sma20:
                continue
            
            # 2. Market direction filter
            if not market_bullish:
                continue
            
            # 3. Positive momentum (>2% in 3 days)
            if features["momentum_3d"] <= 2:
                continue
            
            # 4. Volume confirmation (>1.3x)
            if features["volume_ratio"] < 1.3:
                continue
            
            # 5. ADX trend strength (>20)
            if features["adx"] < 20:
                continue
            
            # 6. RSI not overbought (<65)
            if features["rsi"] >= 65:
                continue
            
            # Calculate stop and target
            stop_loss = round(sma20 * 0.95, 2)
            risk = price - stop_loss
            
            if risk > 0 and risk < price * 0.15:
                size = int(500 / risk)
                
                if size > 0:
                    signals.append({
                        "symbol": sym,
                        "features": features,
                        "price": price,
                        "stop_loss": stop_loss,
                        "target": round(price * 1.10, 2),
                        "size": size,
                        "risk": risk,
                        "score": 0  # Will be filled by ML
                    })
        
        except:
            pass
    
    return signals


def train_ml_model():
    """Train ML model on historical momentum trades"""
    print("Training ML model...")
    
    # Collect historical data
    X_data = []
    y_data = []
    
    for sym in SYMBOLS[:15]:  # Use 15 symbols for training
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period="1y", interval="1d")
            df = df.dropna(subset=['Close'])
            
            if len(df) < 100:
                continue
            
            # Generate features for each day
            for i in range(50, len(df) - 10):
                subset = df.iloc[:i]
                features = extract_features(subset)
                
                if features is None:
                    continue
                
                # Check if signal would have been generated
                close = subset["Close"]
                price = close.iloc[-1]
                sma20 = features["sma20"]
                momentum = features["momentum_3d"]
                volume = features["volume_ratio"]
                adx = features["adx"]
                rsi = features["rsi"]
                
                # Signal conditions
                has_signal = (price > sma20 and momentum > 2 and 
                             volume > 1.3 and adx > 20 and rsi < 65)
                
                if has_signal:
                    # Check if trade would have been profitable (10% target hit before 5% stop)
                    future = df.iloc[i:i+10]
                    if len(future) > 5:
                        future_high = future["High"].max()
                        future_low = future["Low"].min()
                        
                        # Target hit
                        if future_high >= price * 1.10:
                            label = 1  # Win
                        # Stop hit
                        elif future_low <= sma20 * 0.95:
                            label = 0  # Loss
                        else:
                            continue  # No clear outcome
                        
                        # Feature vector
                        X_data.append([
                            features["rsi"],
                            features["volume_ratio"],
                            features["adx"],
                            features["momentum_3d"],
                            features["momentum_5d"],
                            features["price_vs_sma20"]
                        ])
                        y_data.append(label)
        
        except:
            pass
    
    if len(X_data) < 20:
        print("Not enough training data, using rule-based scoring")
        return None
    
    X = np.array(X_data)
    y = np.array(y_data)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X_scaled, y)
    
    # Evaluate
    train_acc = model.score(X_scaled, y)
    print(f"Training accuracy: {train_acc:.1%}")
    
    return model, scaler


def predict_with_ml(model, scaler, features: dict) -> float:
    """Predict trade success probability"""
    if model is None:
        # Fallback to rule-based score
        score = 0
        if features["rsi"] < 50: score += 1
        if features["volume_ratio"] > 1.5: score += 1
        if features["adx"] > 25: score += 1
        if features["momentum_3d"] > 5: score += 1
        return score / 4
    
    X = np.array([[
        features["rsi"],
        features["volume_ratio"],
        features["adx"],
        features["momentum_3d"],
        features["momentum_5d"],
        features["price_vs_sma20"]
    ]])
    
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0][1]  # Probability of success
    
    return prob


# ============== MAIN ==============

print("=" * 70)
print("ENHANCED MOMENTUM TRADING WITH ML")
print("=" * 70)

# Get market direction
print("\nChecking market direction...")
market_bullish = get_market_direction()
print(f"Market: {'BULLISH' if market_bullish else 'BEARISH'}")

# Train ML model
print("\nTraining ML model...")
ml_result = train_ml_model()
model = ml_result[0] if ml_result else None
scaler = ml_result[1] if ml_result else None

# Get enhanced signals
print("\nScanning for enhanced momentum signals...")
signals = get_enhanced_signals(market_bullish)

# Score with ML
for sig in signals:
    sig["ml_prob"] = predict_with_ml(model, scaler, sig["features"])
    sig["score"] = sig["ml_prob"]

# Sort by ML probability
signals.sort(key=lambda x: x["score"], reverse=True)

print(f"\nFound {len(signals)} enhanced signals:")
print("-" * 60)
for sig in signals[:5]:
    f = sig["features"]
    print(f"{sig['symbol']}: ML={sig['ml_prob']:.0%}, RSI={f['rsi']:.0f}, "
          f"Vol={f['volume_ratio']:.1f}x, ADX={f['adx']:.0f}, Mom={f['momentum_3d']:.1f}%")

# Execute trades
print("\n" + "=" * 70)
print("EXECUTING TRADES")
print("=" * 70)

broker_cash = 5000
positions = []
executed = 0

for sig in signals:
    if executed >= 2:
        break
    
    # Only take trades with ML probability > 0.4
    if sig["ml_prob"] < 0.4:
        continue
    
    cost = sig['price'] * sig['size'] + 20
    if broker_cash >= cost and sig['size'] > 0:
        broker_cash -= cost
        positions.append(sig)
        executed += 1
        print(f"\n*** BOUGHT {sig['symbol']} ***")
        print(f"  Price: Rs{sig['price']:.0f}, Size: {sig['size']} shares")
        print(f"  Stop: Rs{sig['stop_loss']}, Target: Rs{sig['target']}")
        f = sig['features']
        print(f"  RSI: {f['rsi']:.0f}, Vol: {f['volume_ratio']:.1f}x, ADX: {f['adx']:.0f}")
        print(f"  ML Probability: {sig['ml_prob']:.0%}")

if executed == 0:
    print("\nNo trades met ML probability threshold (>40%)")
    print("Reducing threshold to find trades...")
    
    for sig in signals:
        if executed >= 1:
            break
        cost = sig['price'] * sig['size'] + 20
        if broker_cash >= cost and sig['size'] > 0:
            broker_cash -= cost
            positions.append(sig)
            executed += 1
            print(f"\n*** BOUGHT {sig['symbol']} (lower prob) ***")
            print(f"  ML Probability: {sig['ml_prob']:.0%}")

print(f"\n=== PORTFOLIO ===")
print(f"Cash: Rs{broker_cash:.2f}")
print(f"Positions: {len(positions)}")

if positions:
    pos_value = sum(p['price'] * p['size'] for p in positions)
    print(f"Position Value: Rs{pos_value:.2f}")
    print(f"Total Value: Rs{broker_cash + pos_value:.2f}")
    print(f"\nOpen Positions:")
    for p in positions:
        f = p['features']
        print(f"  {p['symbol']}: {p['size']} @ Rs{p['price']:.0f}")
        print(f"    Stop:Rs{p['stop_loss']}, Target:Rs{p['target']}, ML:{p['ml_prob']:.0%}")