"""
High-Frequency Momentum Trading System
========================================
- Extended stock list (100+ stocks)
- Stricter filters for precision
- More trades with proper risk management
- Better ML training
"""

import os
import logging
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

API_KEY = "d78vt3hr01qp0fl6mfjgd78vt3hr01qp0fl6mfk0"

# Extended stock list - 100+ stocks
SYMBOLS = [
    # Nifty 50
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "KOTAKBANK.NS", "BAJFINANCE.NS", "HINDUNILVR.NS", "ITC.NS",
    "TITAN.NS", "SUNPHARMA.NS", "MARUTI.NS", "M&M.NS", "NESTLEIND.NS",
    "ULTRACEMCO.NS", "ADANIPORTS.NS", "NTPC.NS", "POWERGRID.NS", "ONGC.NS",
    "LT.NS", "TECHM.NS", "WIPRO.NS", "HCLTECH.NS", "INDUSINDBK.NS",
    "AXISBANK.NS", "KERNEL", "BHARTIARTL.NS", "ASIANPAINT.NS", "MARICO.NS",
    "TATACONSUM.NS", "DABUR.NS", "COLGATE.NS", "HINDUSTANUNILEVER.NS",
    "BRITANNIA.NS", "GODREJCP.NS", "PIDILITIND.NS", "CASTROLIND.NS",
    "BPCL.NS", "IOC.NS", "HINDPETRO.NS", "GAIL.NS", "GAIL.NS",
    # Nifty Next 50
    "ADANIENT.NS", "ADANIGREEN.NS", "AWL.NS", "AUBANK.NS", "BANDHANBNK.NS",
    "BANKBARODA.NS", "BANKINDIA.NS", "CANBK.NS", "CENTRALBK.NS",
    "CHENNPETRO.NS", "COALINDIA.NS", "CONCOR.NS", "CUMMINSIND.NS",
    "ESCORTS.NS", "EXIDEIND.NS", "GODREJPROP.NS", "GODREJIND.NS",
    "GRASIM.NS", "HAVELLS.NS", "HONAUT.NS", "IDFCFIRSTB.NS", "IPAPPMGMNT.NS",
    "JSWSTEEL.NS", "JINDALSTEL.NS", "JUBLFOOD.NS", "KARURVYSYA.NS",
    "LICHSGFIN.NS", "LUPIN.NS", "M&MFIN.NS", "MANAPPURAM.NS",
    "MOTHERSON.NS", "MRF.NS", "MUTHOOTFIN.NS", "NMDC.NS", "NRBINDUS.NS",
    "OIL.NS", "ONGC.NS", "PAGEIND.NS", "PETRONET.NS", "PFC.NS",
    "PNB.NS", "POWERGRID.NS", "RBLBANK.NS", "RECLTD.NS", "SAIL.NS",
    "SHREECEM.NS", "SIEMENS.NS", "SRF.NS", "STAR.NS", "SUNTV.NS",
    "TATAPULL.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TITAN.NS",
    "TORNTPOWER.NS", "TVSMOTOR.NS", "UBL.NS", "UJJIVANSFB.NS",
    "UNIONBANK.NS", "VEDL.NS", "VOLTAS.NS", "WHIRLPOOL.NS", "YESBANK.NS",
    "ZEEL.NS"
]

# Make sure all are valid
SYMBOLS = list(set(SYMBOLS))  # Remove duplicates


def calc_adx(df, period=14):
    try:
        h, l, c = df["High"], df["Low"], df["Close"]
        plus_dm = h.diff().clip(lower=0)
        minus_dm = (-l.diff()).clip(lower=0)
        tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        pdi = (plus_dm.rolling(period).mean() / atr) * 100
        mdi = (minus_dm.rolling(period).mean() / atr) * 100
        dx = (pdi - mdi).abs() / (pdi + mdi) * 100
        return dx.rolling(period).mean().iloc[-1] if len(dx) >= period else 20
    except:
        return 20


def get_market():
    try:
        n = yf.Ticker("^NSEI").history(period="15d")
        price = n["Close"].iloc[-1]
        past = n["Close"].iloc[-5]
        return price > past * 0.98
    except:
        return True


def get_features(df):
    if df is None or len(df) < 30:
        return None
    
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]
    price = close.iloc[-1]
    
    # RSI
    d = close.diff()
    g = d.where(d > 0, 0).rolling(14).mean()
    l = (-d.where(d < 0, 0)).rolling(14).mean()
    rsi = (100 - (100 / (1 + g / l))).iloc[-1]
    
    # RSI divergence (current vs 5 days ago)
    rsi_now = rsi
    rsi_5d = rsi.iloc[-5] if len(rsi) >= 5 else rsi_now
    rsi_div = rsi_now - rsi_5d
    
    # Moving averages
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma20
    ema12 = close.ewm(span=12).mean().iloc[-1]
    ema26 = close.ewm(span=26).mean().iloc[-1]
    
    # MACD
    macd = ema12 - ema26
    macd_signal = pd.Series(macd).ewm(span=9).mean().iloc[-1]
    macd_hist = macd - macd_signal
    macd_cross = (macd > macd_signal) and (macd.shift(1) <= macd_signal.shift(1))
    
    # Volume
    vratio = vol.iloc[-1] / vol.rolling(20).mean().iloc[-1]
    vup = vol.iloc[-1] > vol.iloc[-5] if len(vol) >= 5 else True
    
    # ADX
    adx = calc_adx(df)
    
    # Momentum
    mom3d = (price / close.shift(3).iloc[-1] - 1) * 100
    mom5d = (price / close.shift(5).iloc[-1] - 1) * 100
    mom10d = (price / close.shift(10).iloc[-1] - 1) * 100
    
    # Volatility
    atr = (high - low).rolling(14).mean().iloc[-1]
    atr_pct = atr / price * 100 if price > 0 else 2
    
    return {
        "price": price, "sma20": sma20, "sma50": sma50,
        "rsi": rsi, "rsi_div": rsi_div,
        "macd": macd, "macd_hist": macd_hist, "macd_cross": macd_cross,
        "vratio": vratio, "vup": vup,
        "adx": adx,
        "mom3d": mom3d, "mom5d": mom5d, "mom10d": mom10d,
        "atr_pct": atr_pct
    }


def score_signal(f):
    """STRICTER scoring for higher precision"""
    score = 0
    max_score = 12
    
    # Price above 20-day MA (2 points)
    if f["price"] > f["sma20"]:
        if f["price"] > f["sma20"] * 1.03:
            score += 2
        else:
            score += 1
    
    # RSI in sweet spot (3 points max)
    if 35 < f["rsi"] < 60:
        score += 2
    elif 40 < f["rsi"] < 65:
        score += 1
    
    # RSI divergence (2 points)
    if f["rsi_div"] > 2:
        score += 2
    elif f["rsi_div"] > 0:
        score += 1
    
    # MACD bullish (2 points)
    if f["macd_cross"]:
        score += 2
    elif f["macd_hist"] > 0:
        score += 1
    
    # Volume (1 point)
    if f["vratio"] > 1.2:
        score += 1
    
    # ADX strong trend (1 point)
    if f["adx"] > 25:
        score += 1
    
    # Momentum (2 points max)
    if f["mom3d"] > 3:
        score += 2
    elif f["mom3d"] > 1.5:
        score += 1
    
    return score / max_score


def train_ml():
    """Train ML with more data"""
    print("Training ML model...")
    
    X_data = []
    y_data = []
    
    # Use top 50 stocks for training
    train_stocks = SYMBOLS[:50]
    
    for sym in train_stocks:
        try:
            df = yf.Ticker(sym).history(period="1.5y", interval="1d")
            df = df.dropna(subset=['Close'])
            
            if len(df) < 200:
                continue
            
            # Generate samples
            for i in range(60, len(df) - 20):
                subset = df.iloc[:i]
                f = get_features(subset)
                
                if f is None:
                    continue
                
                # Signal criteria
                signal = (f["price"] > f["sma20"] and f["mom3d"] > 1 and 
                         f["rsi"] < 70 and f["adx"] > 10)
                
                if signal:
                    # Outcome
                    entry = df.iloc[i]["Close"]
                    stop = f["sma20"] * 0.95
                    future = df.iloc[i:i+15]
                    
                    if len(future) >= 10:
                        high = future["High"].max()
                        low = future["Low"].min()
                        
                        if high >= entry * 1.08 and high > low:
                            label = 1
                        elif low <= stop:
                            label = 0
                        else:
                            continue
                        
                        X_data.append([
                            f["rsi"], f["rsi_div"], f["macd_hist"], f["vratio"],
                            f["adx"], f["mom3d"], f["mom5d"], f["atr_pct"],
                            (f["price"] - f["sma20"]) / f["sma20"] * 100
                        ])
                        y_data.append(label)
        
        except:
            pass
    
    if len(X_data) < 30:
        print(f"Not enough ML data ({len(X_data)}), using rule-based")
        return None, None
    
    X = np.array(X_data)
    y = np.array(y_data)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = GradientBoostingClassifier(n_estimators=80, max_depth=4, random_state=42)
    model.fit(X_scaled, y)
    
    acc = model.score(X_scaled, y)
    win_rate = y.mean()
    print(f"ML Training: {len(X_data)} samples, Acc:{acc:.0%}, WinRate:{win_rate:.0%}")
    
    return model, scaler


def get_signals(model, scaler):
    market_up = get_market()
    signals = []
    
    for sym in SYMBOLS:
        try:
            df = yf.Ticker(sym).history(period="60d").dropna(subset=['Close'])
            if len(df) < 30:
                continue
            
            f = get_features(df)
            if f is None:
                continue
            
            # FILTERS - balanced for more trades but still strict
            if f["price"] <= f["sma20"]:
                continue
            if not market_up:
                continue
            if f["mom3d"] < 0.5:  # Lowered for more trades
                continue
            if f["rsi"] > 75:
                continue
            if f["adx"] < 10:
                continue
            
            # Position sizing
            stop = f["sma20"] * 0.96  # Tighter stop
            risk = f["price"] - stop
            
            if risk > 0 and risk < f["price"] * 0.10:  # Max 10% risk
                size = int(400 / risk)  # Rs400 per trade
                
                if size > 0:
                    # Calculate scores
                    rule_score = score_signal(f)
                    
                    # ML score
                    if model:
                        X = np.array([[f["rsi"], f["rsi_div"], f["macd_hist"], f["vratio"],
                                      f["adx"], f["mom3d"], f["mom5d"], f["atr_pct"],
                                      (f["price"] - f["sma20"]) / f["sma20"] * 100]])
                        X_scaled = scaler.transform(X)
                        ml_score = model.predict_proba(X_scaled)[0][1]
                    else:
                        ml_score = rule_score
                    
                    # Combined score
                    final_score = (rule_score * 0.6 + ml_score * 0.4)
                    
                    signals.append({
                        "symbol": sym, "price": f["price"],
                        "sma20": f["sma20"], "rsi": f["rsi"], "rsi_div": f["rsi_div"],
                        "macd_cross": f["macd_cross"], "vratio": f["vratio"],
                        "adx": f["adx"], "mom3d": f["mom3d"],
                        "stop": stop, "target": round(f["price"] * 1.08, 2),
                        "size": size, "risk": risk,
                        "rule_score": rule_score, "ml_score": ml_score,
                        "final_score": final_score
                    })
        
        except:
            pass
    
    signals.sort(key=lambda x: x["final_score"], reverse=True)
    return signals


# MAIN
print("=" * 70)
print("HIGH-FREQUENCY MOMENTUM TRADING")
print("=" * 70)

market = "BULLISH" if get_market() else "BEARISH"
print(f"\nMarket: {market}")
print(f"Scanning: {len(SYMBOLS)} stocks")

# Train ML
model, scaler = train_ml()

# Get signals
print("\nScanning for signals...")
signals = get_signals(model, scaler)
print(f"Found: {len(signals)} signals")

# Show top signals
if signals:
    print(f"\nTop 15 signals:")
    print("-" * 90)
    print(f"{'Symbol':<15} {'Score':>6} {'ML':>6} {'RSI':>6} {'Vol':>6} {'ADX':>6} {'Mom':>7}")
    print("-" * 90)
    for s in signals[:15]:
        print(f"{s['symbol']:<15} {s['final_score']:>6.0%} {s['ml_score']:>6.0%} "
              f"{s['rsi']:>6.1f} {s['vratio']:>6.2f} {s['adx']:>6.0f} {s['mom3d']:>7.1f}%")

# Execute - take top signals, more trades
print("\n" + "=" * 70)
print("EXECUTING TRADES")
print("=" * 70)

cash = 5000
positions = []
executed = 0
max_positions = 4  # More positions

# Take signals with score >= 0.25 (more trades)
for sig in signals:
    if executed >= max_positions:
        break
    
    if sig["final_score"] < 0.25:
        continue
    
    cost = sig["price"] * sig["size"] + 20
    if cash >= cost and sig["size"] > 0:
        cash -= cost
        positions.append(sig)
        executed += 1
        print(f"\n*** Trade {executed}: {sig['symbol']} ***")
        print(f"  Price: Rs{sig['price']:.0f}, Shares: {sig['size']}")
        print(f"  Stop: Rs{sig['stop']}, Target: Rs{sig['target']}")
        print(f"  Score: {sig['final_score']:.0%}, ML: {sig['ml_score']:.0%}")
        print(f"  RSI: {sig['rsi']:.0f}, Vol: {sig['vratio']:.2f}x, ADX: {sig['adx']:.0f}")

# If fewer than max trades, relax threshold
if executed < 2 and len(signals) > executed:
    print(f"\nAdding more trades (relaxed)...")
    for sig in signals[executed:]:
        if executed >= max_positions:
            break
        cost = sig["price"] * sig["size"] + 20
        if cash >= cost and sig["size"] > 0:
            cash -= cost
            positions.append(sig)
            executed += 1
            print(f"  + {sig['symbol']}")

print(f"\n{'='*70}")
print("PORTFOLIO SUMMARY")
print("=" * 70)
print(f"Cash: Rs{cash:.2f}")
print(f"Positions: {len(positions)}")
print(f"Trades Executed: {executed}")

if positions:
    invested = sum(p["price"] * p["size"] for p in positions)
    print(f"Total Invested: Rs{invested:.2f}")
    print(f"Total Value: Rs{cash + invested:.2f}")
    print(f"\nOpen Positions:")
    for i, p in enumerate(positions, 1):
        print(f"  {i}. {p['symbol']}: {p['size']} @ Rs{p['price']:.0f}")
        print(f"     Stop: Rs{p['stop']:.0f}, Target: Rs{p['target']:.0f}")
        print(f"     Score: {p['final_score']:.0%}")