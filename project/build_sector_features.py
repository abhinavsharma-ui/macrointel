#!/usr/bin/env python3
"""
Sector Relative Strength Feature Builder
=========================================
Adds sector-context features to every symbol's feature parquet.
How a stock performs RELATIVE to its sector is more predictive than
absolute price movement — it isolates stock-specific alpha from macro noise.

Features added per symbol:
  sector_rs_5d       - stock return vs sector ETF over 5 days (alpha)
  sector_rs_20d      - same over 20 days
  sector_rs_60d      - same over 60 days
  sector_rs_trend    - is relative strength improving? (5d - 20d RS)
  sector_rank_20d    - percentile rank within sector (0=worst, 1=best)
  sector_leader      - 1 if stock is top 20% of sector on 20d RS
  sector_laggard     - 1 if stock is bottom 20% of sector on 20d RS
  sector_etf_rs_spy  - sector ETF vs SPY (sector rotation context)
  sector_momentum    - sector ETF 20d momentum (riding strong sectors)
  market_regime      - SPY trend: 1=bull (above 200d MA), 0=bear

Uses yfinance (free) for all sector ETF data.

Usage:
    python build_sector_features.py            # all symbols
    python build_sector_features.py --symbols AAPL MSFT
    python build_sector_features.py --refresh-etfs  # re-download ETF data
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent
FEATURES_DIR = PROJECT_DIR / "data" / "features_10yr"
ETF_CACHE = PROJECT_DIR / "data" / "altdata" / "sector_etfs.parquet"
ETF_CACHE.parent.mkdir(parents=True, exist_ok=True)

# GICS sector ETFs
SECTOR_ETFS = {
    "XLK":  "Technology",
    "XLF":  "Financials",
    "XLV":  "Healthcare",
    "XLY":  "ConsumerDiscretionary",
    "XLP":  "ConsumerStaples",
    "XLE":  "Energy",
    "XLI":  "Industrials",
    "XLB":  "Materials",
    "XLRE": "RealEstate",
    "XLU":  "Utilities",
    "XLC":  "Communication",
    "SPY":  "Market",
    "QQQ":  "NasdaqTech",
    "IWM":  "SmallCap",
}

# Symbol → sector ETF mapping (extended)
SYMBOL_SECTOR = {
    # Tech
    **{s: "XLK" for s in [
        "AAPL","MSFT","NVDA","AMD","INTC","AVGO","QCOM","TXN","AMAT","MU",
        "LRCX","KLAC","MRVL","ADI","NXPI","SWKS","MCHP","MPWR","ENTG","ONTO",
        "CRM","ADBE","ORCL","SAP","NOW","INTU","SNOW","PLTR","NET","DDOG",
        "GOOGL","GOOG","META","NFLX","UBER","LYFT","SNAP","PINS","TWTR",
        "IBM","CSCO","HPE","HPQ","DELL","NTAP","PSTG","WDC","STX",
    ]},
    # Financials
    **{s: "XLF" for s in [
        "JPM","BAC","WFC","GS","MS","C","BLK","SCHW","AXP","V","MA","COF",
        "USB","PNC","TFC","MTB","KEY","CFG","FITB","HBAN","RF","ZION",
        "ICE","CME","CBOE","NDAQ","SPGI","MCO","IVZ","BEN","TROW",
    ]},
    # Healthcare
    **{s: "XLV" for s in [
        "JNJ","UNH","PFE","MRK","ABT","TMO","DHR","BMY","AMGN","GILD",
        "REGN","VRTX","BIIB","ILMN","IDXX","EW","ZBH","BAX","BDX","BSX",
        "CVS","CI","HUM","MOH","CNC","ANTM","WBA","MCK","CAH","ABC",
        "ISRG","INTUDF","SYK","ZBH","MDT","ELV",
    ]},
    # Consumer Discretionary
    **{s: "XLY" for s in [
        "AMZN","TSLA","HD","MCD","NKE","SBUX","LOW","TJX","BKNG","CMG",
        "YUM","DRI","DKNG","MGM","LVS","WYNN","RCL","CCL","NCLH","MAR",
        "HLT","EXPE","ABNB","F","GM","RIVN","LCID",
    ]},
    # Consumer Staples
    **{s: "XLP" for s in [
        "PG","KO","PEP","WMT","COST","PM","MO","CL","MDLZ","KHC",
        "GIS","K","CPB","CAG","SJM","HRL","MKC","CHD","CLX","EL",
    ]},
    # Energy
    **{s: "XLE" for s in [
        "XOM","CVX","COP","SLB","EOG","PXD","MPC","VLO","PSX","DVN",
        "OXY","HES","APA","BKR","HAL","FTI","NOV","HP",
    ]},
    # Industrials
    **{s: "XLI" for s in [
        "HON","UNP","CAT","DE","BA","RTX","LMT","NOC","GD","LHX",
        "GE","MMM","EMR","ETN","PH","ROK","AME","FTV","XYL","CARR",
        "OTIS","TT","IR","GNRC","PCAR","CMI",
    ]},
    # Communication
    **{s: "XLC" for s in [
        "GOOGL","GOOG","META","NFLX","DIS","CMCSA","T","VZ","TMUS",
        "EA","TTWO","ATVI","RBLX","MTCH","IAC","ZG","TRIP",
    ]},
    # Materials
    **{s: "XLB" for s in [
        "LIN","APD","SHW","FCX","NEM","NUE","STLD","X","CLF","AA",
        "PPG","ECL","IFF","ALB","MP","BALL","IP","PKG","WRK",
    ]},
    # Real Estate
    **{s: "XLRE" for s in [
        "PLD","AMT","CCI","EQIX","PSA","DLR","WELL","SPG","O","VTR",
        "AVB","EQR","ESS","MAA","UDR","CPT","NNN","SRC","STAG","IIPR",
    ]},
    # Utilities
    **{s: "XLU" for s in [
        "NEE","DUK","SO","D","AEP","EXC","XEL","SRE","ED","ETR",
        "FE","EIX","PPL","CNP","AES","NRG","WEC","ES","AWK",
    ]},
}

# Indian NSE → no perfect sector ETFs, use broad index as proxy
NSE_PROXY = "EEM"  # Emerging markets as sector proxy for NSE stocks


def fetch_etf_data(force_refresh: bool = False) -> pd.DataFrame:
    """Download all sector ETF price history via yfinance."""
    if ETF_CACHE.exists() and not force_refresh:
        age_days = (pd.Timestamp.now() - pd.Timestamp(ETF_CACHE.stat().st_mtime, unit="s")).days
        if age_days < 1:
            logger.info("Using cached ETF data")
            return pd.read_parquet(ETF_CACHE)

    logger.info("Downloading sector ETF data (yfinance)...")
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed: pip install yfinance")
        return pd.DataFrame()

    tickers = list(SECTOR_ETFS.keys()) + ["EEM"]
    data = yf.download(tickers, period="15y", auto_adjust=True, progress=False)["Close"]
    data.index = pd.to_datetime(data.index)
    data = data.sort_index().ffill()
    data.to_parquet(ETF_CACHE)
    logger.info(f"ETF data: {len(data)} days × {len(data.columns)} ETFs")
    return data


def compute_rs(stock_returns: pd.Series, etf_returns: pd.Series, window: int) -> pd.Series:
    """Rolling relative strength: stock cumulative return minus ETF cumulative return."""
    stock_cum = (1 + stock_returns).rolling(window).apply(np.prod, raw=True) - 1
    etf_cum = (1 + etf_returns).rolling(window).apply(np.prod, raw=True) - 1
    return (stock_cum - etf_cum).clip(-1, 1)


def enrich_symbol(symbol: str, etf_data: pd.DataFrame) -> bool:
    feat_file = FEATURES_DIR / f"{symbol}.parquet"
    if not feat_file.exists():
        return False

    try:
        df = pd.read_parquet(feat_file)
        df.index = pd.to_datetime(df.index)
    except Exception as e:
        logger.warning(f"{symbol}: read error — {e}")
        return False

    # Determine sector ETF
    base = symbol.replace(".NS", "").replace(".BO", "")
    if ".NS" in symbol or ".BO" in symbol:
        sector_etf = "EEM"
    else:
        sector_etf = SYMBOL_SECTOR.get(symbol, SYMBOL_SECTOR.get(base, "SPY"))

    if sector_etf not in etf_data.columns or "SPY" not in etf_data.columns:
        return False

    # Align ETF data to stock index
    etf_aligned = etf_data.reindex(df.index, method="ffill")

    # Stock returns (use 'returns_1d' if present, else compute from close)
    if "returns_1d" in df.columns:
        stock_ret = df["returns_1d"].fillna(0)
    elif "close" in df.columns:
        stock_ret = df["close"].pct_change().fillna(0)
    else:
        return False

    etf_ret = etf_aligned[sector_etf].pct_change().fillna(0)
    spy_ret = etf_aligned["SPY"].pct_change().fillna(0)
    sector_ret = etf_aligned[sector_etf].pct_change().fillna(0)

    # Relative strength features
    df["sector_rs_5d"] = compute_rs(stock_ret, etf_ret, 5)
    df["sector_rs_20d"] = compute_rs(stock_ret, etf_ret, 20)
    df["sector_rs_60d"] = compute_rs(stock_ret, etf_ret, 60)
    df["sector_rs_trend"] = df["sector_rs_5d"] - df["sector_rs_20d"]

    # Sector rank within its own ETF (proxy: z-score of RS)
    rs_mean = df["sector_rs_20d"].rolling(252).mean()
    rs_std = df["sector_rs_20d"].rolling(252).std() + 1e-9
    rs_zscore = (df["sector_rs_20d"] - rs_mean) / rs_std
    # Convert to percentile-like score [0,1]
    df["sector_rank_20d"] = rs_zscore.apply(
        lambda z: float(1 / (1 + np.exp(-z)))  # sigmoid to [0,1]
    )
    df["sector_leader"] = (df["sector_rank_20d"] > 0.8).astype(float)
    df["sector_laggard"] = (df["sector_rank_20d"] < 0.2).astype(float)

    # Sector ETF vs SPY (sector rotation)
    df["sector_etf_rs_spy"] = compute_rs(sector_ret, spy_ret, 20)

    # Sector momentum
    sector_cum_20 = (1 + sector_ret).rolling(20).apply(np.prod, raw=True) - 1
    df["sector_momentum"] = sector_cum_20.clip(-0.5, 0.5)

    # Market regime: SPY above 200d MA = bull
    spy_price = etf_aligned["SPY"]
    spy_ma200 = spy_price.rolling(200).mean()
    df["market_regime"] = (spy_price > spy_ma200).astype(float)

    df.to_parquet(feat_file)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--refresh-etfs", action="store_true")
    args = parser.parse_args()

    etf_data = fetch_etf_data(force_refresh=args.refresh_etfs)
    if etf_data.empty:
        logger.error("Could not load ETF data")
        return 1

    symbols = args.symbols or [p.stem for p in sorted(FEATURES_DIR.glob("*.parquet"))]
    logger.info(f"Enriching {len(symbols)} symbols with sector RS features")

    success = 0
    for i, sym in enumerate(symbols, 1):
        if enrich_symbol(sym, etf_data):
            success += 1
        if i % 100 == 0:
            logger.info(f"Progress: {i}/{len(symbols)}")

    logger.info(f"Done — {success}/{len(symbols)} symbols enriched with sector features")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
