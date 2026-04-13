"""
High-Frequency Momentum - Simplified Working Version
=======================================================
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# Working stocks
SYMBOLS = [
    "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "KOTAKBANK.NS", "BAJFINANCE.NS", "HINDUNILVR.NS", "ITC.NS",
    "TITAN.NS", "SUNPHARMA.NS", "MARUTI.NS", "M&M.NS", "NESTLEIND.NS",
    "ULTRACEMCO.NS", "ADANIPORTS.NS", "NTPC.NS", "POWERGRID.NS",
    "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "INDUSINDBK.NS", "AXISBANK.NS",
    "BHARTIARTL.NS", "ASIANPAINT.NS", "TATACONSUM.NS", "DABUR.NS",
    "GRASIM.NS", "HAVELLS.NS", "CIPLA.NS", "DIVISLAB.NS", "IRCTC.NS",
    "VEDL.NS", "TATASTEEL.NS", "COALINDIA.NS", "BPCL.NS", "IOC.NS",
    "HINDPETRO.NS", "GAIL.NS", "CONCOR.NS", "UPL.NS", "MOTHERSON.NS",
    "AUBANK.NS", "BANDHANBNK.NS", "CANBK.NS", "CENTRALBK.NS",
    "CHENNPETRO.NS", "CUMMINSIND.NS", "ESCORTS.NS", "GODREJPROP.NS",
    "GODREJIND.NS", "HONAUT.NS", "IDFCFIRSTB.NS", "JSWSTEEL.NS",
    "JINDALSTEL.NS", "JUBLFOOD.NS", "KARURVYSYA.NS", "LICHSGFIN.NS",
    "LUPIN.NS", "M&MFIN.NS", "MANAPPURAM.NS", "MUTHOOTFIN.NS",
    "PETRONET.NS", "PFC.NS", "PNB.NS", "RBLBANK.NS", "SAIL.NS",
    "SHREECEM.NS", "SIEMENS.NS", "SUNTV.NS", "TVSMOTOR.NS",
    "UBL.NS", "UNIONBANK.NS", "VOLTAS.NS", "WHIRLPOOL.NS", "ZEEL.NS",
    "APOLLOTYRE.NS", "L&T.NS", "RELAXO.NS", "AMBER.NS"
]


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
        return dx.rolling(p).mean().iloc[-1] if len(dx) >= p else 20
    except:
        return 20


def get_market():
    try:
        n = yf.Ticker("^NSEI").history(period="15d")
        return n["Close"].iloc[-1] > n["Close"].iloc[-5]
    except:
        return True


def analyze_stock(sym):
    """Analyze single stock"""
    try:
        df = yf.Ticker(sym).history(period="60d").dropna(subset=['Close'])
        if df is None or len(df) < 30:
            return None
        
        close = df["Close"]
        vol = df["Volume"]
        price = close.iloc[-1]
        
        # RSI
        d = close.diff()
        g = d.where(d > 0, 0).rolling(14).mean()
        l = (-d.where(d < 0, 0)).rolling(14).mean()
        rsi = (100 - (100 / (1 + g / l))).iloc[-1]
        
        sma20 = close.rolling(20).mean().iloc[-1]
        vratio = vol.iloc[-1] / vol.rolling(20).mean().iloc[-1]
        adx = calc_adx(df)
        mom3d = (price / close.shift(3).iloc[-1] - 1) * 100
        
        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        macd_sig = macd.ewm(span=9).mean()
        macd_x = (macd.iloc[-1] > macd_sig.iloc[-1]) and (macd.iloc[-2] <= macd_sig.iloc[-2])
        
        # Score
        score = 0
        if price > sma20: score += 2
        if 40 < rsi < 60: score += 2
        if rsi < 55: score += 1
        if macd_x: score += 2
        if vratio > 1.0: score += 1
        if adx > 20: score += 1
        if mom3d > 3: score += 2
        elif mom3d > 1: score += 1
        
        score_pct = score / 11
        
        # Check filters
        if price <= sma20: return None
        if mom3d < 0.5: return None
        if rsi > 75: return None
        if adx < 10: return None
        
        stop = sma20 * 0.96
        risk = price - stop
        if risk <= 0 or risk > price * 0.10: return None
        
        size = int(150 / risk)  # Much lower risk for more positions
        if size < 1: return None
        
        return {
            "symbol": sym, "price": price, "sma20": sma20,
            "rsi": rsi, "vratio": vratio, "adx": adx, "mom3d": mom3d, "macd_x": macd_x,
            "stop": round(stop, 2), "target": round(price * 1.08, 2),
            "size": size, "risk": risk, "score": score_pct
        }
    except:
        return None


# MAIN
print("=" * 70)
print("HIGH-FREQUENCY MOMENTUM TRADING")
print("=" * 70)
print(f"Market: {'BULLISH' if get_market() else 'BEARISH'}")
print(f"Scanning: {len(SYMBOLS)} stocks\n")

# Analyze all stocks
signals = []
for sym in SYMBOLS:
    result = analyze_stock(sym)
    if result:
        signals.append(result)

print(f"Found: {len(signals)} signals\n")

# Sort by score
signals.sort(key=lambda x: x["score"], reverse=True)

# Show signals
if signals:
    print("Top 15 signals:")
    print("-" * 80)
    for s in signals[:15]:
        print(f"{s['symbol']:<15} {s['score']:.0%} | RSI:{s['rsi']:>5.0f} "
              f"Vol:{s['vratio']:>4.1f}x ADX:{s['adx']:>3.0f} Mom:{s['mom3d']:>5.1f}%")

# Execute
print("\n" + "=" * 70)
print("EXECUTING TRADES")
print("=" * 70)

cash = 5000
positions = []
executed = 0

for s in signals:
    if executed >= 5:
        break
    
    # Lower threshold for more trades (0.15 instead of 0.18)
    if s["score"] < 0.15:
        continue
    
    cost = s["price"] * s["size"] + 20
    if cash >= cost:
        cash -= cost
        positions.append(s)
        executed += 1
        print(f"\n*** Trade {executed}: {s['symbol']} ***")
        print(f"  Price: Rs{s['price']:.0f}, Shares: {s['size']}")
        print(f"  Stop: Rs{s['stop']}, Target: Rs{s['target']}")
        print(f"  Score: {s['score']:.0%} | RSI:{s['rsi']:.0f} Vol:{s['vratio']:.1f}x "
              f"ADX:{s['adx']:.0f} Mom:{s['mom3d']:.1f}%")

# If no trades, take best available
if executed == 0 and signals:
    print("\nTaking best available signal...")
    s = signals[0]
    cost = s["price"] * s["size"] + 20
    if cash >= cost:
        cash -= cost
        positions.append(s)
        executed = 1
        print(f"  + {s['symbol']}")

print(f"\n{'='*70}")
print("PORTFOLIO")
print("=" * 70)
print(f"Cash: Rs{cash:.2f}")
print(f"Trades: {executed}")
if positions:
    invested = sum(p["price"] * p["size"] for p in positions)
    print(f"Invested: Rs{invested:.2f}")
    print(f"Total: Rs{cash + invested:.2f}")
    for i, p in enumerate(positions, 1):
        print(f"  {i}. {p['symbol']}: {p['size']} @ Rs{p['price']:.0f} | "
              f"Stop:Rs{p['stop']:.0f} Target:Rs{p['target']:.0f}")