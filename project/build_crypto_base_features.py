#!/usr/bin/env python3
import argparse, logging, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent
FEATURES_DIR = PROJECT_DIR / "data" / "features_10yr"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
BINANCE_URL = "https://api.binance.com/api/v3/klines"

def _load_env():
    env = {}
    p = PROJECT_DIR / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env

def get_crypto_symbols():
    env = _load_env()
    s = env.get("CRYPTO_DEPTH_SYMBOLS", "")
    return [x.strip() for x in s.split(",") if x.strip().endswith("USDT")]

def fetch_daily_candles(symbol, days=3650):
    import time as t
    start_ms = int((t.time() - days * 86400) * 1000)
    all_candles = []
    while True:
        try:
            r = requests.get(BINANCE_URL, params={"symbol": symbol, "interval": "1d", "startTime": start_ms, "limit": 1000}, timeout=15)
            if r.status_code != 200: break
            data = r.json()
            if not data: break
            all_candles.extend(data)
            if len(data) < 1000: break
            start_ms = data[-1][0] + 1
            time.sleep(0.2)
        except Exception as e:
            logger.warning(f"{symbol}: {e}"); break
    if not all_candles: return pd.DataFrame()
    df = pd.DataFrame(all_candles, columns=["open_time","open","high","low","close","volume","close_time","quote_vol","trades","tbb","tbq","ignore"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("open_time")[["open","high","low","close","volume"]].apply(pd.to_numeric, errors="coerce")
    return df.sort_index()[~df.sort_index().index.duplicated(keep="last")]

def build_features(df):
    f = pd.DataFrame(index=df.index)
    c, h, l, v, o = df["close"], df["high"], df["low"], df["volume"], df["open"]
    f["open"]=o; f["high"]=h; f["low"]=l; f["close"]=c; f["volume"]=v
    for p in [1,3,5,10,20,60]: f[f"return_{p}d"] = c.pct_change(p)
    for p in [7,14]:
        d=c.diff(); g=d.where(d>0,0); ls=-d.where(d<0,0)
        f[f"rsi_{p}"] = (100-(100/(1+g.rolling(p).mean()/(ls.rolling(p).mean()+1e-9)))).clip(0,100)
    ef=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean()
    es=ef.ewm(span=9,adjust=False).mean(); f["macd"]=ef; f["macd_signal"]=es; f["macd_hist"]=ef-es
    sma=c.rolling(20).mean(); std=c.rolling(20).std()
    f["bb_position"]=(c-sma+2*std)/(4*std+1e-9); f["bb_width"]=(4*std)/(sma+1e-9)
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr=tr.rolling(14).mean(); f["atr_14"]=atr; f["atr_pct"]=atr/(c+1e-9)
    ll=l.rolling(14).min(); hh=h.rolling(14).max()
    f["stoch_k"]=100*(c-ll)/(hh-ll+1e-9); f["stoch_d"]=f["stoch_k"].rolling(3).mean()
    f["momentum_20d"]=c.pct_change(20); f["momentum_60d"]=c.pct_change(60)
    for s in [9,21,50,200]: f[f"close_vs_sma_{s}"]=(c-c.rolling(s).mean())/(c.rolling(s).mean()+1e-9)
    pdm=h.diff().clip(lower=0); mdm=(-l.diff()).clip(lower=0)
    pdi=100*pdm.ewm(span=14).mean()/(atr+1e-9); mdi=100*mdm.ewm(span=14).mean()/(atr+1e-9)
    f["adx_14"]=(100*(pdi-mdi).abs()/(pdi+mdi+1e-9)).ewm(span=14).mean(); f["di_plus"]=pdi; f["di_minus"]=mdi
    obv=(np.sign(c.diff())*v).fillna(0).cumsum(); f["obv"]=obv; f["obv_trend"]=obv.diff()
    tp=(h+l+c)/3; mf=tp*v
    f["mfi"]=(100-100/(1+mf.where(tp.diff()>0,0).rolling(14).sum()/(mf.where(tp.diff()<0,0).rolling(14).sum()+1e-9))).clip(0,100)
    ret=c.pct_change(); f["hist_vol_10"]=ret.rolling(10).std(); f["hist_vol_30"]=ret.rolling(30).std(); f["realized_vol_21d"]=ret.rolling(21).std()
    f["williams_r"]=-100*(h.rolling(14).max()-c)/(h.rolling(14).max()-l.rolling(14).min()+1e-9)
    for p in [5,10,20]: f[f"roc_{p}"]=c.pct_change(p)
    f["52w_high_ratio"]=c/(c.rolling(252).max()+1e-9); f["52w_low_ratio"]=c/(c.rolling(252).min()+1e-9)
    f["vol_vs_30d_avg"]=v/(v.rolling(30).mean()+1e-9); f["vol_vs_7d_avg"]=v/(v.rolling(7).mean()+1e-9)
    f["hl_ratio"]=(h-l)/(c+1e-9); f["hl_ratio_5d_avg"]=f["hl_ratio"].rolling(5).mean()
    f["body_ratio"]=(c-o).abs()/(h-l+1e-9); f["direction"]=np.sign(c-o)
    return f.ffill().fillna(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--days", type=int, default=3650)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    symbols = args.symbols or get_crypto_symbols()
    if not symbols: logger.error("No symbols found in CRYPTO_DEPTH_SYMBOLS"); return 1
    logger.info(f"Building base features for {len(symbols)} crypto symbols")
    ok = fail = skip = 0
    for sym in symbols:
        out = FEATURES_DIR / f"{sym}.parquet"
        if args.skip_existing and out.exists(): skip += 1; continue
        try:
            df = fetch_daily_candles(sym, args.days)
            if df.empty or len(df) < 100: logger.warning(f"{sym}: no data"); fail += 1; continue
            build_features(df).to_parquet(out, compression="snappy")
            logger.info(f"{sym}: {len(df)} days saved"); ok += 1
        except Exception as e: logger.warning(f"{sym}: {e}"); fail += 1
        time.sleep(0.2)
    logger.info(f"Done — {ok} built, {skip} skipped, {fail} failed")
    if ok > 0: logger.info("Next: python project/build_crypto_intraday_features.py")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
