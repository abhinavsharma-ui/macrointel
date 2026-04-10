"""
Comprehensive ML Training - Fixed
===================================
Relaxed conditions to generate more training data
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# Reliable stock list
SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "KOTAKBANK.NS", "BAJFINANCE.NS", "HINDUNILVR.NS", "ITC.NS",
    "TITAN.NS", "SUNPHARMA.NS", "MARUTI.NS", "M&M.NS", "NESTLEIND.NS",
    "ULTRACEMCO.NS", "ADANIPORTS.NS", "NTPC.NS", "POWERGRID.NS", "ONGC.NS",
    "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "INDUSINDBK.NS", "AXISBANK.NS",
    "BHARTIARTL.NS", "ASIANPAINT.NS", "TATACONSUM.NS", "DABUR.NS", "GRASIM.NS",
    "HAVELLS.NS", "CIPLA.NS", "DIVISLAB.NS", "IRCTC.NS", "VEDL.NS", "TATASTEEL.NS",
    "COALINDIA.NS", "BPCL.NS", "IOC.NS", "HINDPETRO.NS", "GAIL.NS", "CONCOR.NS",
    "UPL.NS", "MOTHERSON.NS", "AUBANK.NS", "BANDHANBNK.NS", "CANBK.NS",
    "JSWSTEEL.NS", "JINDALSTEL.NS", "JUBLFOOD.NS", "LUPIN.NS", "MUTHOOTFIN.NS"
]

print("=" * 70)
print("COMPREHENSIVE BACKTEST & ML TRAINING (FIXED)")
print("=" * 70)

# ============================================
# PART 1: BACKTEST - Relaxed conditions
# ============================================
print("\n[1] Running 2-year backtest with relaxed conditions...")

X_train = []
y_train = []
backtest_trades = []

for sym in SYMBOLS[:40]:
    try:
        df = yf.Ticker(sym).history(period="2y", interval="1d")
        df = df.dropna(subset=['Close'])
        if len(df) < 300:
            continue
        
        for i in range(60, len(df) - 20):
            try:
                subset = df.iloc[:i]
                close = subset["Close"]
                high = subset["High"]
                low = subset["Low"]
                vol = subset["Volume"]
                price = close.iloc[-1]
                
                # RSI
                d = close.diff()
                g = d.where(d > 0, 0).rolling(14).mean()
                l = (-d.where(d < 0, 0)).rolling(14).mean()
                rsi = (100 - (100 / (1 + g / l))).iloc[-1]
                
                sma20 = close.rolling(20).mean().iloc[-1]
                vratio = vol.iloc[-1] / vol.rolling(20).mean().iloc[-1]
                
                # Any positive momentum (relaxed from 0.5%)
                mom3d = (price / close.shift(3).iloc[-1] - 1) * 100
                
                # Signal check - RELAXED
                if price > sma20 and mom3d > 0 and rsi < 75:
                    entry = df.iloc[i]["Close"]
                    stop = sma20 * 0.95
                    future = df.iloc[i:i+20]
                    
                    if len(future) >= 10:
                        high_max = future["High"].max()
                        low_min = future["Low"].min()
                        
                        # Win: 8% target hit
                        if high_max >= entry * 1.08:
                            label = 1
                        elif low_min <= stop:
                            label = 0
                        else:
                            # Check if closed at profit or loss after 20 days
                            exit_price = future.iloc[-1]["Close"]
                            if exit_price > entry * 1.03:
                                label = 1
                            elif exit_price < entry * 0.97:
                                label = 0
                            else:
                                continue  # No clear outcome
                        
                        # Add training sample
                        X_train.append([rsi, vratio, mom3d, (price-sma20)/sma20*100])
                        y_train.append(label)
                        
                        backtest_trades.append({"symbol": sym, "win": label == 1})
            except:
                continue
    except:
        pass

print(f"   Generated: {len(X_train)} training samples")
if len(backtest_trades) > 0:
    wins = sum(1 for t in backtest_trades if t["win"])
    print(f"   Historical Win Rate: {wins/len(backtest_trades)*100:.1f}%")

# ============================================
# PART 2: TRAIN ML
# ============================================
print("\n[2] Training ML model...")

if len(X_train) >= 30:
    X = np.array(X_train)
    y = np.array(y_train)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = GradientBoostingClassifier(n_estimators=80, max_depth=4, random_state=42)
    model.fit(X_scaled, y)
    
    acc = model.score(X_scaled, y)
    win_rate = y.mean()
    print(f"   ML: {len(X)} samples, Acc:{acc*100:.1f}%, WinRate:{win_rate*100:.1f}%")
else:
    model, scaler = None, None
    print(f"   Not enough data ({len(X_train)}), using rules")

# ============================================
# PART 3: CURRENT SIGNALS
# ============================================
print("\n[3] Scanning for signals...")

def calc_adx(df, p=14):
    try:
        h, l, c = df["High"], df["Low"], df["Close"]
        pdm = h.diff().clip(lower=0)
        mdm = (-l.diff()).clip(lower=0)
        tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(p).mean()
        pdi = (pdm.rolling(p).mean() / atr) * 100
        mdi = (mdm.rolling(p).mean() / atr) * 100
        dx = (pdi - mdi).abs() / (pdi + mdi) * 100
        return dx.rolling(p).mean().iloc[-1]
    except:
        return 20

def get_market():
    try:
        n = yf.Ticker("^NSEI").history(period="15d")
        return n["Close"].iloc[-1] > n["Close"].iloc[-5]
    except:
        return True

signals = []
market = get_market()

for sym in SYMBOLS:
    try:
        df = yf.Ticker(sym).history(period="60d").dropna(subset=['Close'])
        if len(df) < 30: continue
        
        close = df["Close"]
        vol = df["Volume"]
        price = close.iloc[-1]
        
        d = close.diff()
        g = d.where(d > 0, 0).rolling(14).mean()
        l = (-d.where(d < 0, 0)).rolling(14).mean()
        rsi = (100 - (100 / (1 + g / l))).iloc[-1]
        
        sma20 = close.rolling(20).mean().iloc[-1]
        vratio = vol.iloc[-1] / vol.rolling(20).mean().iloc[-1]
        adx = calc_adx(df)
        mom3d = (price / close.shift(3).iloc[-1] - 1) * 100
        
        # Filters
        if price <= sma20: continue
        if not market: continue
        if mom3d < 0: continue
        if rsi > 75: continue
        
        stop = sma20 * 0.96
        risk = price - stop
        if risk <= 0 or risk > price * 0.10: continue
        
        # Very small for many trades
        size = int(60 / risk)
        
        if size > 0:
            # Score
            score = 0
            if price > sma20: score += 2
            if 40 < rsi < 60: score += 2
            if vratio > 1.0: score += 1
            if adx > 20: score += 1
            if mom3d > 2: score += 2
            
            # ML score
            if model and scaler:
                X = np.array([[rsi, vratio, mom3d, (price-sma20)/sma20*100]])
                ml = model.predict_proba(scaler.transform(X))[0][1]
            else:
                ml = score / 8
            
            final = ml * 0.6 + (score/8) * 0.4
            
            signals.append({
                "symbol": sym, "price": price, "sma20": sma20,
                "rsi": rsi, "vratio": vratio, "adx": adx, "mom3d": mom3d,
                "stop": round(stop, 2), "target": round(price * 1.08, 2),
                "size": size, "score": final, "ml": ml
            })
    except:
        pass

signals.sort(key=lambda x: x["score"], reverse=True)
print(f"   Found: {len(signals)} signals")

# ============================================
# PART 4: EXECUTE TRADES
# ============================================
print("\n[4] Executing trades...")

cash = 5000
positions = []
executed = 0

for s in signals:
    if executed >= 20:
        break
    if s["score"] < 0.15:
        continue
    
    cost = s["price"] * s["size"] + 10
    if cash >= cost and s["size"] > 0:
        cash -= cost
        positions.append(s)
        executed += 1
        print(f"   + {s['symbol']}: {s['size']} @ Rs{s['price']:.0f}")

# ============================================
# SUMMARY
# ============================================
print(f"\n{'='*70}")
print("RESULTS")
print("=" * 70)
print(f"ML Training: {len(X_train)} samples")
if len(X_train) > 0:
    print(f"Historical Win Rate: {win_rate*100:.1f}%")
print(f"Signals: {len(signals)}")
print(f"Trades: {executed}")
print(f"Cash: Rs{cash:.2f}")

if positions:
    invested = sum(p["price"] * p["size"] for p in positions)
    print(f"Invested: Rs{invested:.2f}")
    print(f"\nPositions ({len(positions)}):")
    for i, p in enumerate(positions, 1):
        print(f"   {i}. {p['symbol']}: {p['size']} @ Rs{p['price']:.0f}")