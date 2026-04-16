#!/usr/bin/env python3
"""
Build India-Specific Features
================================
Fetches REAL India market data and enriches all _NS symbol parquets with:
  - India VIX regime (real historical data via yfinance ^INDIAVIX)
  - Nifty 50 / BankNifty actual rolling correlation (not random!)
  - Sector rotation signals (real sector indices via yfinance)
  - F&O Put-Call Ratio from NSE option chain API (live, free, no auth)
  - Intraday gap/close patterns

Replaces the fake np.random.uniform() data in IndiaFeatureEngineer.

Usage:
    python build_india_features.py                  # all _NS symbols
    python build_india_features.py --symbols RELIANCE_NS TCS_NS
    python build_india_features.py --skip-existing  # only unenriched symbols
"""

import argparse
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent
FEATURES_DIR = PROJECT_DIR / "data" / "features_10yr"
ALTDATA_DIR  = PROJECT_DIR / "data" / "altdata" / "india"
ALTDATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Symbol mappings ───────────────────────────────────────────────────────────

SECTOR_INDICES = {
    "bank":    "^NSEBANK",
    "it":      "^CNXIT",
    "auto":    "^CNXAUTO",
    "metal":   "^CNXMETAL",
    "fmcg":    "^CNXFMCG",
    "pharma":  "^CNXPHARMA",
    "energy":  "^CNXENERGY",
    "infra":   "^CNXINFRA",
    "realty":  "^CNXREALTY",
}

SECTOR_MAP = {
    # Banks
    "HDFCBANK":"bank","ICICIBANK":"bank","SBIN":"bank","KOTAKBANK":"bank",
    "AXISBANK":"bank","BANDHANBNK":"bank","INDUSINDBK":"bank","IDFCFIRSTB":"bank",
    "FEDERALBNK":"bank","RBLBANK":"bank","AUBANK":"bank","CANARABANK":"bank",
    "BANKBARODA":"bank","UNIONBANK":"bank","PNB":"bank","YESBANK":"bank",
    # IT
    "TCS":"it","INFY":"it","WIPRO":"it","HCLTECH":"it","TECHM":"it",
    "MPHASIS":"it","LTIM":"it","COFORGE":"it","PERSISTENT":"it","OFSS":"it",
    # Energy
    "RELIANCE":"energy","ONGC":"energy","BPCL":"energy","IOC":"energy",
    "GAIL":"energy","COALINDIA":"energy","NTPC":"energy","POWERGRID":"energy",
    # Auto
    "MARUTI":"auto","TATAMOTORS":"auto","M&M":"auto","BAJAJ-AUTO":"auto",
    "HEROMOTOCO":"auto","EICHERMOT":"auto","TVSMOTOR":"auto","ASHOKLEY":"auto",
    # Pharma
    "SUNPHARMA":"pharma","DRREDDY":"pharma","CIPLA":"pharma","DIVISLAB":"pharma",
    "APOLLOHOSP":"pharma","AUROPHARMA":"pharma","BIOCON":"pharma","TORNTPHARM":"pharma",
    # FMCG
    "HINDUNILVR":"fmcg","ITC":"fmcg","NESTLEIND":"fmcg","BRITANNIA":"fmcg",
    "DABUR":"fmcg","MARICO":"fmcg","GODREJCP":"fmcg","COLPAL":"fmcg",
    # Infra
    "ADANIPORTS":"infra","LT":"infra","ADANIENT":"infra","ULTRACEMCO":"infra",
    "GRASIM":"infra","AMBUJACEM":"infra","ACC":"infra",
    # Finance
    "BAJFINANCE":"finance","BAJAJFINSV":"finance","MUTHOOTFIN":"finance",
    "CHOLAFIN":"finance","LICHSGFIN":"finance","SBICARD":"finance",
    # Metal
    "TATASTEEL":"metal","JSWSTEEL":"metal","HINDALCO":"metal","VEDL":"metal",
    "SAIL":"metal","NMDC":"metal","JINDALSTEL":"metal",
    # Realty
    "DLF":"realty","GODREJPROP":"realty","OBEROIRLTY":"realty","PRESTIGE":"realty",
    # Consumer
    "TITAN":"consumer","ASIANPAINT":"consumer","PIDILITIND":"consumer","HAVELLS":"consumer",
}

NIFTY50 = {
    "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","SBIN","BAJFINANCE","ADANIPORTS",
    "KOTAKBANK","HINDUNILVR","BAJAJFINSV","WIPRO","AXISBANK","MARUTI","HCLTECH","LT",
    "TATAMOTORS","SUNPHARMA","TECHM","TATASTEEL","NTPC","POWERGRID","TITAN","ITC",
    "NESTLEIND","ULTRACEMCO","BPCL","INDUSINDBK","ONGC","JSWSTEEL","M&M","COALINDIA",
    "DIVISLAB","DRREDDY","CIPLA","HINDALCO","GRASIM","ASIANPAINT","EICHERMOT",
    "BRITANNIA","APOLLOHOSP","BAJAJ-AUTO","HEROMOTOCO","SBILIFE","HDFCLIFE",
    "ADANIENT","LTIM","TATACONSUM","VEDL","DABUR",
}

BANKNIFTY = {
    "HDFCBANK","ICICIBANK","SBIN","KOTAKBANK","AXISBANK","BANDHANBNK",
    "INDUSINDBK","IDFCFIRSTB","FEDERALBNK","RBLBANK","AUBANK","CANARABANK",
    "BANKBARODA","UNIONBANK","PNB",
}

FNO_STOCKS = {
    "RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK","SBIN","BAJFINANCE","KOTAKBANK",
    "AXISBANK","WIPRO","HCLTECH","TECHM","TITAN","SUNPHARMA","TATASTEEL","JSWSTEEL",
    "TATAMOTORS","M&M","MARUTI","LT","ADANIPORTS","ONGC","BPCL","ITC","HINDUNILVR",
    "BAJAJFINSV","INDUSINDBK","NTPC","POWERGRID","COALINDIA","GRASIM","ULTRACEMCO",
    "DRREDDY","CIPLA","DIVISLAB","HINDALCO","ASIANPAINT","EICHERMOT","BAJAJ-AUTO",
}


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_india_vix(days: int = 3650) -> pd.Series:
    """Fetch India VIX from yfinance ^INDIAVIX."""
    cache = ALTDATA_DIR / "india_vix.parquet"
    cached = pd.Series(dtype=float)
    if cache.exists():
        try:
            cached = pd.read_parquet(cache).squeeze()
            cached.index = pd.to_datetime(cached.index).tz_localize(None).normalize()
        except Exception:
            pass

    try:
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        raw = yf.download("^INDIAVIX", start=start, progress=False, auto_adjust=True)
        if not raw.empty:
            vix = raw["Close"].squeeze()
            vix.index = pd.to_datetime(vix.index).tz_localize(None).normalize()
            vix = vix[~vix.index.duplicated(keep="last")]
            if not cached.empty:
                vix = pd.concat([cached[cached.index < vix.index[0]], vix])
            vix.name = "vix"
            vix.to_frame().to_parquet(cache)
            logger.info(f"India VIX: {len(vix)} days (latest: {vix.iloc[-1]:.1f})")
            return vix
    except Exception as e:
        logger.warning(f"VIX fetch failed: {e}")
    return cached


def fetch_nifty_reference(days: int = 3650) -> pd.DataFrame:
    """Fetch Nifty 50 and BankNifty for correlation calculation."""
    cache = ALTDATA_DIR / "nifty_reference.parquet"
    if cache.exists():
        try:
            df = pd.read_parquet(cache)
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            if (datetime.now() - df.index[-1].to_pydatetime()).days < 2:
                return df
        except Exception:
            pass
    try:
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        raw = yf.download(["^NSEI", "^NSEBANK"], start=start, progress=False, auto_adjust=True)["Close"]
        raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
        raw.columns = ["nifty50", "banknifty"]
        raw = raw.ffill()
        raw.to_parquet(cache)
        logger.info(f"Nifty reference: {len(raw)} days")
        return raw
    except Exception as e:
        logger.warning(f"Nifty reference fetch failed: {e}")
        return pd.DataFrame()


def fetch_sector_indices(days: int = 3650) -> pd.DataFrame:
    """Fetch all sector index daily prices from yfinance."""
    cache = ALTDATA_DIR / "sector_indices.parquet"
    if cache.exists():
        try:
            df = pd.read_parquet(cache)
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            if (datetime.now() - df.index[-1].to_pydatetime()).days < 2:
                logger.info(f"Sector indices: {len(df)} days (cache)")
                return df
        except Exception:
            pass
    try:
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        tickers = list(SECTOR_INDICES.values())
        raw = yf.download(tickers, start=start, progress=False, auto_adjust=True)["Close"]
        raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
        reverse = {v: k for k, v in SECTOR_INDICES.items()}
        raw = raw.rename(columns=reverse).ffill()
        raw.to_parquet(cache)
        logger.info(f"Sector indices: {len(raw)} days, {len(raw.columns)} sectors")
        return raw
    except Exception as e:
        logger.warning(f"Sector fetch failed: {e}")
        return pd.DataFrame()


def fetch_pcr_live() -> dict:
    """Fetch real Put-Call Ratio from NSE option chain API. Free, no auth."""
    result = {"nifty_pcr": 1.0, "banknifty_pcr": 1.0}
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/option-chain",
    })
    try:
        session.get("https://www.nseindia.com", timeout=10)
        time.sleep(1.5)
    except Exception:
        return result

    for sym, key in [("NIFTY", "nifty_pcr"), ("BANKNIFTY", "banknifty_pcr")]:
        try:
            r = session.get(
                f"https://www.nseindia.com/api/option-chain-indices?symbol={sym}",
                timeout=15,
            )
            if r.status_code != 200:
                continue
            records = r.json().get("records", {}).get("data", [])
            pe_oi = sum(rec.get("PE", {}).get("openInterest", 0) for rec in records if "PE" in rec)
            ce_oi = sum(rec.get("CE", {}).get("openInterest", 0) for rec in records if "CE" in rec)
            if ce_oi > 0:
                result[key] = round(pe_oi / ce_oi, 3)
            time.sleep(1)
        except Exception as e:
            logger.debug(f"PCR {sym}: {e}")

    # Cache daily
    pcr_cache = ALTDATA_DIR / "pcr_history.json"
    history = {}
    if pcr_cache.exists():
        try:
            history = json.loads(pcr_cache.read_text())
        except Exception:
            pass
    history[datetime.now().strftime("%Y-%m-%d")] = result
    pcr_cache.write_text(json.dumps(history, indent=2))
    logger.info(f"PCR — Nifty: {result['nifty_pcr']:.2f}, BankNifty: {result['banknifty_pcr']:.2f}")
    return result


# ── Feature builder ───────────────────────────────────────────────────────────

def build_india_features(
    df: pd.DataFrame,
    symbol: str,
    vix_series: pd.Series,
    sector_df: pd.DataFrame,
    nifty_ref: pd.DataFrame,
    pcr: dict,
) -> pd.DataFrame:
    """Build all real India features for a single NS symbol."""
    sym = symbol.replace("_NS", "").replace(".NS", "")
    feat = pd.DataFrame(index=df.index)

    # ── India VIX ──────────────────────────────────────────────────────────
    if not vix_series.empty:
        vix = vix_series.reindex(df.index).ffill().bfill().fillna(18.0)
        feat["india_vix"] = vix
        feat["india_vix_regime"] = pd.cut(
            vix, bins=[0, 12, 18, 25, 35, 9999],
            labels=[3, 2, 1, 0, -1]
        ).astype(float).fillna(1.0)
        roll_mean = vix.rolling(252, min_periods=30).mean()
        roll_std  = vix.rolling(252, min_periods=30).std() + 1e-9
        feat["india_vix_zscore"]   = ((vix - roll_mean) / roll_std).clip(-4, 4)
        feat["india_vix_spike"]    = (vix > vix.rolling(20).mean() * 1.25).astype(int)
        feat["india_vix_5d_chg"]   = vix.pct_change(5)
        feat["high_vol_regime"]    = (vix > 20).astype(int)
        feat["crisis_regime"]      = (vix > 30).astype(int)

    # ── Real Nifty / BankNifty correlation ─────────────────────────────────
    if not nifty_ref.empty and "close" in df.columns:
        stock_ret = df["close"].pct_change()

        nifty_ret = nifty_ref["nifty50"].reindex(df.index).ffill().pct_change()
        feat["nifty50_corr_20d"]  = stock_ret.rolling(20).corr(nifty_ret).clip(-1, 1)
        feat["nifty50_corr_60d"]  = stock_ret.rolling(60).corr(nifty_ret).clip(-1, 1)
        feat["nifty50_beta_60d"]  = (
            stock_ret.rolling(60).cov(nifty_ret) /
            (nifty_ret.rolling(60).var() + 1e-9)
        ).clip(-3, 3)
        feat["is_nifty50"]        = int(sym in NIFTY50)

        if "banknifty" in nifty_ref.columns:
            bank_ret = nifty_ref["banknifty"].reindex(df.index).ffill().pct_change()
            feat["banknifty_corr_60d"] = stock_ret.rolling(60).corr(bank_ret).clip(-1, 1)
            feat["is_banknifty"]       = int(sym in BANKNIFTY)

    # ── Sector features ────────────────────────────────────────────────────
    sector = SECTOR_MAP.get(sym, "other")
    feat["sector_code"] = {
        "bank":7,"it":6,"energy":5,"auto":4,"pharma":3,
        "fmcg":2,"infra":1,"finance":8,"metal":9,"realty":10,"consumer":11,"other":0
    }.get(sector, 0)
    feat["is_bank_stock"]  = int(sector == "bank")
    feat["is_it_stock"]    = int(sector == "it")
    feat["is_fno_stock"]   = int(sym in FNO_STOCKS)
    feat["is_large_cap"]   = int(sym in NIFTY50)

    if not sector_df.empty and sector in sector_df.columns:
        sec = sector_df[sector].reindex(df.index).ffill()
        sec_ret = sec.pct_change()
        feat["sector_mom_5d"]   = sec_ret.rolling(5).sum()
        feat["sector_mom_20d"]  = sec_ret.rolling(20).sum()
        feat["sector_mom_60d"]  = sec_ret.rolling(60).sum()
        feat["sector_vol_20d"]  = sec_ret.rolling(20).std()

        # Stock relative strength vs own sector
        if "close" in df.columns:
            feat["stock_vs_sector_20d"] = df["close"].pct_change(20) - sec_ret.rolling(20).sum()
            feat["stock_vs_sector_5d"]  = df["close"].pct_change(5)  - sec_ret.rolling(5).sum()

        # Broad sector rotation — which sector is leading right now
        sector_cols = [c for c in sector_df.columns if c in SECTOR_INDICES]
        if len(sector_cols) > 2:
            s20 = sector_df[sector_cols].pct_change(20).reindex(df.index).ffill()
            avg = s20.mean(axis=1)
            if sector in s20.columns:
                feat["sector_rel_strength"] = (s20[sector] - avg).clip(-0.5, 0.5)

    # ── Intraday patterns ──────────────────────────────────────────────────
    if "close" in df.columns and "open" in df.columns:
        gap = (df["open"] - df["close"].shift(1)) / (df["close"].shift(1) + 1e-9)
        feat["gap_pct"]        = gap.clip(-0.1, 0.1)
        feat["gap_up"]         = (gap > 0.005).astype(int)
        feat["gap_down"]       = (gap < -0.005).astype(int)
        feat["intraday_return"]= (df["close"] - df["open"]) / (df["open"] + 1e-9)
        # Gap fill: opened up but closed lower than open (or vice versa)
        feat["gap_fill"]       = (
            ((gap > 0.005) & (df["close"] < df["open"])) |
            ((gap < -0.005) & (df["close"] > df["open"]))
        ).astype(int)

    if "high" in df.columns and "low" in df.columns:
        hl = df["high"] - df["low"] + 1e-9
        feat["intraday_range"]  = hl / (df["close"] + 1e-9)
        feat["close_position"]  = (df["close"] - df["low"]) / hl   # 1=closed at high, 0=at low
        feat["upper_wick_ratio"]= (df["high"] - df[["close","open"]].max(axis=1)) / hl
        feat["lower_wick_ratio"]= (df[["close","open"]].min(axis=1) - df["low"]) / hl

    # ── F&O PCR ────────────────────────────────────────────────────────────
    pcr_val = pcr.get("banknifty_pcr" if sector == "bank" else "nifty_pcr", 1.0)
    feat["pcr"]         = pcr_val
    feat["pcr_regime"]  = 2 if pcr_val > 1.3 else (0 if pcr_val < 0.8 else 1)
    feat["pcr_bullish"] = int(pcr_val > 1.2)   # contrarian: high PCR = buy

    return feat.ffill().bfill().fillna(0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols",       nargs="+", default=None)
    parser.add_argument("--days",          type=int,  default=3650)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    symbols = args.symbols or [p.stem for p in sorted(FEATURES_DIR.glob("*_NS.parquet"))]
    if not symbols:
        logger.error("No _NS symbols found in features_10yr/")
        return 1

    logger.info(f"Building India features for {len(symbols)} NS symbols")

    # Fetch shared reference data once
    vix_series = fetch_india_vix(args.days)
    nifty_ref  = fetch_nifty_reference(args.days)
    sector_df  = fetch_sector_indices(args.days)
    pcr        = fetch_pcr_live()

    success = failed = skipped = 0

    for sym in symbols:
        path = FEATURES_DIR / f"{sym}.parquet"
        if not path.exists():
            failed += 1
            continue

        if args.skip_existing:
            try:
                if "india_vix" in pd.read_parquet(path).columns:
                    skipped += 1
                    continue
            except Exception:
                pass

        try:
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            feats = build_india_features(df, sym, vix_series, sector_df, nifty_ref, pcr)
            for col in feats.columns:
                df[col] = feats[col]
            df.to_parquet(path)
            success += 1
            if success % 25 == 0:
                logger.info(f"Progress: {success}/{len(symbols)}")
        except Exception as e:
            logger.warning(f"{sym}: {e}")
            failed += 1

    logger.info(f"Done — {success} enriched, {skipped} skipped, {failed} failed")
    if success > 0:
        logger.info("Next: python project/retrain_institutional_models.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
