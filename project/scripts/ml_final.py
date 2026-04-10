"""
ML Trading - Use momentum instead of MA for market filter
=========================================================
"""

import yfinance as yf
import pandas as pd

# Working stock list
SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "KOTAKBANK.NS", "BAJFINANCE.NS", "HINDUNILVR.NS", "ITC.NS",
    "TITAN.NS", "SUNPHARMA.NS", "MARUTI.NS", "M&M.NS", "NESTLEIND.NS",
    "ULTRACEMCO.NS", "ADANIPORTS.NS", "NTPC.NS", "POWERGRID.NS", "ONGC.NS",
    "VEDL.NS", "TATASTEEL.NS", "COALINDIA.NS", "CIPLA.NS", "HCLTECH.NS",
    "WIPRO.NS", "IRCTC.NS", "DIVISLAB.NS", "BHARTIARTL.NS", "TECHM.NS",
    "INDUSINDBK.NS", "AXISBANK.NS", "UPL.NS", "GRASIM.NS", "LTIM.NS",
    "APOLLOTYRE.NS", "BANDHANBNK.NS", "CONCOR.NS", "IOC.NS", "BPCL.NS"
]


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
    """Use momentum instead of MA for market direction"""
    try:
        n = yf.Ticker("^NSEI").history(period="20d")
        price = n["Close"].iloc[-1]
        past = n["Close"].iloc[-10]  # 10-day momentum
        return price > past * 0.98  # Allow small dip
    except:
        return True


def get_features(df):
    if df is None or len(df) < 30:
        return None
    
    close = df["Close"]
    vol = df["Volume"]
    price = close.iloc[-1]
    
    d = close.diff()
    g = d.where(d > 0, 0).rolling(14).mean()
    l = (-d.where(d < 0, 0)).rolling(14).mean()
    rsi = (100 - (100 / (1 + g / l))).iloc[-1]
    
    sma20 = close.rolling(20).mean().iloc[-1]
    vratio = vol.iloc[-1] / vol.rolling(20).mean().iloc[-1] if len(vol) >= 20 else 1
    adx = calc_adx(df)
    mom3d = (price / close.shift(3).iloc[-1] - 1) * 100 if len(close) >= 4 else 0
    
    return {"price": price, "sma20": sma20, "rsi": rsi, "vratio": vratio, "adx": adx, "mom3d": mom3d}


def score(features):
    s = 0
    if features["price"] > features["sma20"]: s += 2
    if features["mom3d"] > 2: s += 2
    if features["vratio"] > 0.8: s += 1
    if features["adx"] > 15: s += 1
    if features["rsi"] < 65: s += 1
    return s / 8


# MAIN
print("=" * 70)
print("ML TRADING - MOMENTUM MARKET FILTER")
print("=" * 70)
print(f"Market: {'BULLISH' if get_market() else 'BEARISH'}")

signals = []
for sym in SYMBOLS:
    try:
        df = yf.Ticker(sym).history(period="60d").dropna(subset=['Close'])
        if len(df) < 30:
            continue
        
        f = get_features(df)
        
        # Filters
        if f["price"] <= f["sma20"]:
            continue
        if f["mom3d"] < 1:
            continue
        if f["rsi"] > 75:
            continue
        
        stop = f["sma20"] * 0.95
        risk = f["price"] - stop
        
        if risk > 0 and risk < f["price"] * 0.12:
            size = int(500 / risk)
            if size > 0:
                signals.append({
                    "symbol": sym, "price": f["price"], "sma20": f["sma20"],
                    "rsi": f["rsi"], "vratio": f["vratio"], "adx": f["adx"],
                    "mom3d": f["mom3d"], "stop": stop,
                    "target": round(f["price"] * 1.10, 2), "size": size,
                    "score": score(f)
                })
    except:
        pass

signals.sort(key=lambda x: x["score"], reverse=True)
print(f"\nFound {len(signals)} signals")

if signals:
    print("\nTop signals:")
    for s in signals[:6]:
        print(f"  {s['symbol']}: Score={s['score']:.2f}, RSI={s['rsi']:.0f}, "
              f"Vol={s['vratio']:.1f}x, ADX={s['adx']:.0f}, Mom={s['mom3d']:.1f}%")

# Execute
cash = 5000
positions = []
executed = 0

for sig in signals:
    if executed >= 2:
        break
    cost = sig["price"] * sig["size"] + 20
    if cash >= cost and sig["size"] > 0:
        cash -= cost
        positions.append(sig)
        executed += 1
        print(f"\n*** BOUGHT {sig['symbol']} ***")
        print(f"  {sig['size']} shares @ Rs{sig['price']:.0f}")
        print(f"  Stop:Rs{sig['stop']}, Target:Rs{sig['target']}")
        print(f"  RSI:{sig['rsi']:.0f}, Vol:{sig['vratio']:.1f}x, ADX:{sig['adx']:.0f}")

if executed == 0 and signals:
    s = signals[0]
    cost = s["price"] * s["size"] + 20
    if cash >= cost:
        cash -= cost
        positions.append(s)
        print(f"\n*** BOUGHT {s['symbol']} (best available) ***")

print(f"\n=== PORTFOLIO ===")
print(f"Cash: Rs{cash:.2f}")
if positions:
    val = sum(p["price"] * p["size"] for p in positions)
    print(f"Total: Rs{cash + val:.2f}")
    for p in positions:
        print(f"  {p['symbol']}: {p['size']} @ Rs{p['price']:.0f}")