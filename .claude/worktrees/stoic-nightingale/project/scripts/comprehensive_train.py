"""
Comprehensive ML Training + High-Frequency Trading
=====================================================
1. Backtest 2 years of data for 50 stocks (generate 500+ training samples)
2. Train ML model on real historical data
3. Execute many more paper trades
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
    "JSWSTEEL.NS", "JINDALSTEL.NS", "JUBLFOOD.NS", "LUPIN.NS", "MUTHOOTFIN.NS",
    "PETRONET.NS", "PFC.NS", "RBLBANK.NS", "SAIL.NS", "SUNTV.NS", "TVSMOTOR.NS",
    "UBL.NS", "UNIONBANK.NS", "VOLTAS.NS", "WHIRLPOOL.NS", "ZEEL.NS"
]

print("=" * 70)
print("COMPREHENSIVE BACKTEST & ML TRAINING")
print("=" * 70)

# ============================================
# PART 1: BACKTEST 2 YEARS FOR ML TRAINING
# ============================================
print("\n[1] Running 2-year backtest for ML training data...")

X_train = []
y_train = []
backtest_trades = []

for sym in SYMBOLS[:50]:
    try:
        df = yf.Ticker(sym).history(period="2y", interval="1d")
        df = df.dropna(subset=['Close'])
        if len(df) < 300:
            continue
        
        # Process each day
        for i in range(60, len(df) - 20):
            try:
                subset = df.iloc[:i]
                close = subset["Close"]
                high = subset["High"]
                low = subset["Low"]
                vol = subset["Volume"]
                price = close.iloc[-1]
                
                # Features
                d = close.diff()
                g = d.where(d > 0, 0).rolling(14).mean()
                l = (-d.where(d < 0, 0)).rolling(14).mean()
                rsi = (100 - (100 / (1 + g / l))).iloc[-1]
                rsi_5d = rsi.iloc[-5] if len(rsi) >= 5 else rsi.iloc[-1]
                
                sma20 = close.rolling(20).mean().iloc[-1]
                vratio = vol.iloc[-1] / vol.rolling(20).mean().iloc[-1]
                
                # ADX
                h, l, c = high, low, close
                pdm = h.diff().clip(lower=0)
                mdm = (-l.diff()).clip(lower=0)
                tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
                atr = tr.rolling(14).mean()
                pdi = (pdm.rolling(14).mean() / atr) * 100
                mdi = (mdm.rolling(14).mean() / atr) * 100
                dx = (pdi - mdi).abs() / (pdi + mdi) * 100
                adx = dx.rolling(14).mean().iloc[-1]
                
                mom3d = (price / close.shift(3).iloc[-1] - 1) * 100
                
                # Signal check
                if price > sma20 and mom3d > 0.5 and rsi < 70 and adx > 10:
                    # Check outcome
                    entry = df.iloc[i]["Close"]
                    stop = sma20 * 0.95
                    future = df.iloc[i:i+20]
                    
                    if len(future) >= 10:
                        high_max = future["High"].max()
                        low_min = future["Low"].min()
                        
                        # Win: 8% target hit before stop
                        if high_max >= entry * 1.08 and high_max > low_min:
                            label = 1
                            win = True
                        elif low_min <= stop:
                            label = 0
                            win = False
                        else:
                            continue  # Unclear
                        
                        # Add to training
                        X_train.append([rsi, rsi - rsi_5d, vratio, adx, mom3d, 
                                       (price - sma20) / sma20 * 100])
                        y_train.append(label)
                        
                        # Track trade
                        backtest_trades.append({
                            "symbol": sym, "date": str(df.index[i].date()),
                            "price": entry, "win": win
                        })
            except:
                continue
    
    except:
        pass

print(f"   Generated: {len(X_train)} training samples")
if len(backtest_trades) > 0:
    wins = sum(1 for t in backtest_trades if t["win"])
    print(f"   Win Rate: {wins/len(backtest_trades)*100:.1f}%")

# ============================================
# PART 2: TRAIN ML MODEL
# ============================================
print("\n[2] Training ML model...")

if len(X_train) >= 50:
    X = np.array(X_train)
    y = np.array(y_train)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = GradientBoostingClassifier(
        n_estimators=100, 
        max_depth=5, 
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_scaled, y)
    
    acc = model.score(X_scaled, y)
    win_rate = y.mean()
    print(f"   ML Training: {len(X)} samples")
    print(f"   Accuracy: {acc*100:.1f}%")
    print(f"   Historical Win Rate: {win_rate*100:.1f}%")
else:
    print(f"   Not enough data ({len(X_train)}), using rules")
    model = None
    scaler = None

# ============================================
# PART 3: SCAN FOR CURRENT SIGNALS
# ============================================
print("\n[3] Scanning for current signals...")

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
        if mom3d < 0.5: continue
        if rsi > 75: continue
        
        # Position
        stop = sma20 * 0.96
        risk = price - stop
        if risk <= 0 or risk > price * 0.10: continue
        
        # Very small position for many trades
        size = int(80 / risk)  # Rs80 per trade = lots of positions
        
        if size > 0:
            # Score
            score = 0
            if price > sma20: score += 2
            if 40 < rsi < 60: score += 2
            if rsi < 55: score += 1
            if vratio > 1.0: score += 1
            if adx > 20: score += 1
            if mom3d > 3: score += 2
            
            # ML score
            if model and scaler:
                X = np.array([[rsi, 0, vratio, adx, mom3d, (price-sma20)/sma20*100]])
                ml = model.predict_proba(scaler.transform(X))[0][1]
            else:
                ml = score/9
            
            final = ml * 0.6 + (score/9) * 0.4
            
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

# Show top signals
if signals:
    print("\n   Top signals:")
    for s in signals[:10]:
        print(f"   {s['symbol']}: {s['score']:.0%} | RSI:{s['rsi']:.0f} Vol:{s['vratio']:.1f}x ADX:{s['adx']:.0f}")

# ============================================
# PART 4: EXECUTE MANY TRADES
# ============================================
print("\n[4] Executing many trades...")

cash = 5000
positions = []
executed = 0
max_trades = 15  # Many more trades

# Take all signals above 20% score
for s in signals:
    if executed >= max_trades:
        break
    if s["score"] < 0.20:
        continue
    
    cost = s["price"] * s["size"] + 15
    if cash >= cost and s["size"] > 0:
        cash -= cost
        positions.append(s)
        executed += 1
        print(f"   + {s['symbol']}: {s['size']} @ Rs{s['price']:.0f}")

# If still room, take more
if executed < 5 and signals:
    print(f"   Adding more (relaxed)...")
    for s in signals:
        if executed >= max_trades:
            break
        cost = s["price"] * s["size"] + 15
        if cash >= cost and s["size"] > 0:
            cash -= cost
            positions.append(s)
            executed += 1
            print(f"   + {s['symbol']}: {s['size']} @ Rs{s['price']:.0f}")

# ============================================
# SUMMARY
# ============================================
print(f"\n{'='*70}")
print("FINAL RESULTS")
print("=" * 70)
print(f"ML Training Samples: {len(X_train)}")
print(f"Historical Win Rate: {win_rate*100:.1f}%" if len(X_train) >= 50 else "N/A")
print(f"Signals Found: {len(signals)}")
print(f"Trades Executed: {executed}")
print(f"Cash Remaining: Rs{cash:.2f}")

if positions:
    invested = sum(p["price"] * p["size"] for p in positions)
    print(f"Total Invested: Rs{invested:.2f}")
    print(f"\nOpen Positions ({len(positions)}):")
    for i, p in enumerate(positions, 1):
        print(f"   {i}. {p['symbol']}: {p['size']} @ Rs{p['price']:.0f} | "
              f"Stop:Rs{p['stop']:.0f} Target:Rs{p['target']:.0f}")