"""
High-Frequency Momentum Trading - Fixed
=========================================
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# Working stocks only - verified symbols
SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "KOTAKBANK.NS", "BAJFINANCE.NS", "HINDUNILVR.NS", "ITC.NS",
    "TITAN.NS", "SUNPHARMA.NS", "MARUTI.NS", "M&M.NS", "NESTLEIND.NS",
    "ULTRACEMCO.NS", "ADANIPORTS.NS", "NTPC.NS", "POWERGRID.NS", "ONGC.NS",
    "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "INDUSINDBK.NS", "AXISBANK.NS",
    "BHARTIARTL.NS", "ASIANPAINT.NS", "MARICO.NS", "TATACONSUM.NS", "DABUR.NS",
    "GRASIM.NS", "HAVELLS.NS", "CIPLA.NS", "DIVISLAB.NS", "IRCTC.NS",
    "VEDL.NS", "TATASTEEL.NS", "COALINDIA.NS", "BPCL.NS", "IOC.NS",
    "HINDPETRO.NS", "GAIL.NS", "CONCOR.NS", "UPL.NS", "MOTHERSON.NS",
    "ADANIENT.NS", "AUBANK.NS", "BANDHANBNK.NS", "BANKBARODA.NS", "CANBK.NS",
    "CENTRALBK.NS", "CHENNPETRO.NS", "CUMMINSIND.NS", "ESCORTS.NS", "EXIDEIND.NS",
    "GODREJPROP.NS", "GODREJIND.NS", "HONAUT.NS", "IDFCFIRSTB.NS", "JSWSTEEL.NS",
    "JINDALSTEL.NS", "JUBLFOOD.NS", "KARURVYSYA.NS", "LICHSGFIN.NS", "LUPIN.NS",
    "M&MFIN.NS", "MANAPPURAM.NS", "MRF.NS", "MUTHOOTFIN.NS", "NMDC.NS",
    "OIL.NS", "PETRONET.NS", "PFC.NS", "PNB.NS", "RBLBANK.NS", "RECLTD.NS",
    "SAIL.NS", "SHREECEM.NS", "SIEMENS.NS", "SRF.NS", "STAR.NS", "SUNTV.NS",
    "TORNTPOWER.NS", "TVSMOTOR.NS", "UBL.NS", "UJJIVANSFB.NS", "UNIONBANK.NS",
    "VOLTAS.NS", "WHIRLPOOL.NS", "ZEEL.NS", "APOLLOTYRE.NS", "BEL.NS",
    "HAL.NS", "L&T.NS", "RELAXO.NS", "AMBER.NS", "BSE.NS", "CDSL.NS"
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
    rsi_5d = rsi.iloc[-5] if len(rsi) >= 5 else rsi.iloc[-1]
    rsi_div = rsi.iloc[-1] - rsi_5d
    
    # MA
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma20
    
    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9).mean()
    macd_hist = macd - macd_sig
    macd_x = (macd.iloc[-1] > macd_sig.iloc[-1]) and (macd.iloc[-2] <= macd_sig.iloc[-2])
    
    # Volume
    vratio = vol.iloc[-1] / vol.rolling(20).mean().iloc[-1] if len(vol) >= 20 else 1
    
    # ADX
    adx = calc_adx(df)
    
    # Momentum
    mom3d = (price / close.shift(3).iloc[-1] - 1) * 100
    
    return {
        "price": price, "sma20": sma20, "sma50": sma50,
        "rsi": rsi, "rsi_div": rsi_div,
        "macd_x": macd_x, "macd_hist": macd_hist,
        "vratio": vratio, "adx": adx, "mom3d": mom3d
    }


def score(f):
    s = 0
    if f["price"] > f["sma20"]: s += 2
    if 40 < f["rsi"] < 60: s += 2
    if f["rsi_div"] > 1: s += 1
    if f["macd_x"]: s += 2
    if f["vratio"] > 1.0: s += 1
    if f["adx"] > 20: s += 1
    if f["mom3d"] > 2: s += 2
    return s / 11


def train_ml():
    print("Training ML...")
    X, y = [], []
    train_stocks = SYMBOLS[:40]
    
    for sym in train_stocks:
        try:
            df = yf.Ticker(sym).history(period="2y", interval="1d")
            df = df.dropna(subset=['Close'])
            if len(df) < 150: continue
            
            for i in range(60, len(df) - 20):
                f = get_features(df.iloc[:i])
                if f is None: continue
                
                if f["price"] > f["sma20"] and f["mom3d"] > 1 and f["rsi"] < 70:
                    entry = df.iloc[i]["Close"]
                    stop = f["sma20"] * 0.95
                    future = df.iloc[i:i+15]
                    
                    if len(future) >= 10:
                        h, l = future["High"].max(), future["Low"].min()
                        if h >= entry * 1.08: label = 1
                        elif l <= stop: label = 0
                        else: continue
                        
                        X.append([f["rsi"], f["rsi_div"], f["macd_hist"], f["vratio"],
                                  f["adx"], f["mom3d"]])
                        y.append(label)
        except: pass
    
    if len(X) < 20:
        print(f"ML data: {len(X)} samples - using rules")
        return None, None
    
    X, y = np.array(X), np.array(y)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    model = GradientBoostingClassifier(n_estimators=60, max_depth=4, random_state=42)
    model.fit(X, y)
    
    print(f"ML: {len(X)} samples, acc:{model.score(X,y):.0%}, win:{y.mean():.0%}")
    return model, scaler


def get_signals(model, scaler):
    mkt = get_market()
    signals = []
    
    for sym in SYMBOLS:
        try:
            df = yf.Ticker(sym).history(period="60d").dropna(subset=['Close'])
            if len(df) < 30: continue
            
            f = get_features(df)
            if not f: continue
            
            # FILTERS - more trades
            if f["price"] <= f["sma20"]: continue
            if not mkt: continue
            if f["mom3d"] < 0.5: continue
            if f["rsi"] > 75: continue
            
            stop = f["sma20"] * 0.96
            risk = f["price"] - stop
            
            if risk > 0 and risk < f["price"] * 0.10:
                size = int(400 / risk)
                if size > 0:
                    rule = score(f)
                    
                    if model:
                        X = np.array([[f["rsi"], f["rsi_div"], f["macd_hist"], f["vratio"],
                                      f["adx"], f["mom3d"]]])
                        ml = model.predict_proba(scaler.transform(X))[0][1]
                    else:
                        ml = rule
                    
                    final = rule * 0.5 + ml * 0.5
                    
                    signals.append({
                        "symbol": sym, "price": f["price"], "sma20": f["sma20"],
                        "rsi": f["rsi"], "vratio": f["vratio"], "adx": f["adx"],
                        "mom3d": f["mom3d"], "macd_x": f["macd_x"],
                        "stop": stop, "target": round(f["price"] * 1.08, 2),
                        "size": size, "rule": rule, "ml": ml, "final": final
                    })
        except: pass
    
    signals.sort(key=lambda x: x["final"], reverse=True)
    return signals


# MAIN
print("=" * 70)
print("HIGH-FREQUENCY MOMENTUM TRADING")
print("=" * 70)
print(f"Market: {'BULLISH' if get_market() else 'BEARISH'}")
print(f"Stocks: {len(SYMBOLS)}")

model, scaler = train_ml()
signals = get_signals(model, scaler)
print(f"\nSignals: {len(signals)}")

if signals:
    print("\nTop signals:")
    for s in signals[:10]:
        print(f"  {s['symbol']}: {s['final']:.0%} | RSI:{s['rsi']:.0f} Vol:{s['vratio']:.1f}x "
              f"ADX:{s['adx']:.0f} Mom:{s['mom3d']:.1f}%")

# Execute more trades
cash = 5000
pos = []
execed = 0

print("\n" + "=" * 70)
print("EXECUTING TRADES")
print("=" * 70)

for s in signals:
    if execed >= 5: break
    if s["final"] < 0.2: continue
    
    cost = s["price"] * s["size"] + 20
    if cash >= cost and s["size"] > 0:
        cash -= cost
        pos.append(s)
        execed += 1
        print(f"\n*** Trade {execed}: {s['symbol']} ***")
        print(f"  {s['size']} @ Rs{s['price']:.0f}")
        print(f"  Stop:Rs{s['stop']:.0f}, Target:Rs{s['target']:.0f}")
        print(f"  Score:{s['final']:.0%} | RSI:{s['rsi']:.0f} Vol:{s['vratio']:.1f}x ADX:{s['adx']:.0f}")

if execed == 0 and signals:
    print("\nTaking best available...")
    for s in signals[:2]:
        cost = s["price"] * s["size"] + 20
        if cash >= cost:
            cash -= cost
            pos.append(s)
            execed += 1
            print(f"  + {s['symbol']}")

print(f"\n=== PORTFOLIO ===")
print(f"Cash: Rs{cash:.2f}")
print(f"Trades: {execed}")
if pos:
    inv = sum(p["price"] * p["size"] for p in pos)
    print(f"Invested: Rs{inv:.2f}")
    print(f"Total: Rs{cash + inv:.2f}")
    for i, p in enumerate(pos, 1):
        print(f"  {i}. {p['symbol']}: {p['size']} @ Rs{p['price']:.0f}")