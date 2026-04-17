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

# Indian NSE → no perfect sector ETFs, use broad index as proxy (legacy default)
NSE_PROXY = "EEM"  # Emerging markets as sector proxy for NSE stocks

# NSE sector indices via yfinance — real sector proxies for Indian stocks.
# Keys are yfinance tickers, values are logical sector labels.
NSE_SECTOR_INDICES = {
    "^NSEI":       "NSE_Market",      # Nifty 50 (market proxy)
    "^CNXIT":      "IT",
    "^NSEBANK":    "Bank",             # Nifty Bank
    "^CNXPHARMA":  "Pharma",
    "^CNXAUTO":    "Auto",
    "^CNXFMCG":    "FMCG",
    "^CNXMETAL":   "Metal",
    "^CNXENERGY":  "Energy",
    "^NSMIDCP":    "NSE_MidCap",      # Nifty Midcap 50 (for regime flag)
}

# NSE symbol -> sector index ticker (subset; unknown symbols fall back to ^NSEI)
NSE_SYMBOL_SECTOR = {
    **{s: "^CNXIT" for s in [
        "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM", "LTM", "PERSISTENT",
        "MPHASIS", "COFORGE", "LT",
    ]},
    **{s: "^NSEBANK" for s in [
        "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK",
        "BANKBARODA", "PNB", "IDFCFIRSTB", "FEDERALBNK", "RBLBANK", "BANDHANBNK",
    ]},
    **{s: "^CNXPHARMA" for s in [
        "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "LUPIN", "BIOCON", "APOLLOHOSP",
        "TORNTPHARM", "ALKEM", "LALPATHLAB", "AUROBINDO", "GLAND",
    ]},
    **{s: "^CNXAUTO" for s in [
        "MARUTI", "TATAMOTORS", "M&M", "HEROMOTOCO", "BAJAJ-AUTO", "EICHERMOT",
        "BAJAJHLDNG", "ASHOKLEY", "TVSMOTOR", "BOSCHLTD", "MOTHERSON", "APOLLOTYRE",
    ]},
    **{s: "^CNXFMCG" for s in [
        "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR", "MARICO", "GODREJCP",
        "COLPAL", "TATACONSUM", "EMAMILTD", "VBL",
    ]},
    **{s: "^CNXMETAL" for s in [
        "TATASTEEL", "JSWSTEEL", "HINDALCO", "COALINDIA", "VEDL", "SAIL", "JINDALSTEL",
        "NATIONALUM", "NMDC",
    ]},
    **{s: "^CNXENERGY" for s in [
        "RELIANCE", "ONGC", "BPCL", "IOC", "GAIL", "POWERGRID", "NTPC", "ADANIGREEN",
        "ADANIPOWER", "TATAPOWER", "NHPC",
    ]},
}

NSE_MARKET_TICKER = "^NSEI"       # Nifty 50
NSE_MIDCAP_TICKER = "^NSMIDCP"    # Nifty Midcap 50
GOLDILOCKS_MIDCAP_PREMIUM = 0.01  # 1% 20-bar outperformance flag per spec


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

    tickers = list(SECTOR_ETFS.keys()) + ["EEM"] + list(NSE_SECTOR_INDICES.keys())
    try:
        raw = yf.download(tickers, period="15y", auto_adjust=True, progress=False, group_by="column")
        if isinstance(raw.columns, pd.MultiIndex) and "Close" in raw.columns.get_level_values(0):
            data = raw["Close"]
        else:
            data = raw
    except Exception as exc:
        logger.error("yfinance download failed: %s", exc)
        return pd.DataFrame()
    data.index = pd.to_datetime(data.index)
    data = data.sort_index().ffill()
    # Drop fully-empty columns (e.g. NSE indices not resolvable in some yf versions)
    data = data.dropna(axis=1, how="all")
    data.to_parquet(ETF_CACHE)
    logger.info(f"ETF data: {len(data)} days × {len(data.columns)} ETFs")
    return data


def compute_rs(stock_returns: pd.Series, etf_returns: pd.Series, window: int) -> pd.Series:
    """Rolling relative strength: stock cumulative return minus ETF cumulative return."""
    stock_cum = (1 + stock_returns).rolling(window).apply(np.prod, raw=True) - 1
    etf_cum = (1 + etf_returns).rolling(window).apply(np.prod, raw=True) - 1
    return (stock_cum - etf_cum).clip(-1, 1)


def _nse_sector_for(symbol: str) -> str:
    base = symbol.replace(".NS", "").replace(".BO", "").upper()
    return NSE_SYMBOL_SECTOR.get(base, NSE_MARKET_TICKER)


def _enrich_nse_symbol(df: pd.DataFrame, symbol: str, etf_data: pd.DataFrame) -> pd.DataFrame:
    """
    NSE-specific enrichment. Writes:
      sector_rs_score_5d / _10d / _20d     : rolling RS vs NSE sector index
      sector_rs_trend                       : 5d - 20d
      sector_leader / sector_laggard        : percentile flags
      sector_etf_rs_market                  : sector vs Nifty 50
      sector_momentum                       : sector 20d cumulative
      market_regime                         : Nifty 50 above 200d MA
      regime_flag                           : "goldilocks" if midcap vs largecap
                                              20-bar cumret premium >1%
    """
    sector_ticker = _nse_sector_for(symbol)
    if sector_ticker not in etf_data.columns or NSE_MARKET_TICKER not in etf_data.columns:
        logger.warning("NSE sector index missing (%s or %s); writing neutral",
                       sector_ticker, NSE_MARKET_TICKER)
        for col in ("sector_rs_score_5d", "sector_rs_score_10d", "sector_rs_score_20d",
                    "sector_rs_trend", "sector_leader", "sector_laggard",
                    "sector_etf_rs_market", "sector_momentum", "market_regime"):
            df[col] = 0.0
        df["regime_flag"] = "neutral"
        return df

    aligned = etf_data.reindex(df.index, method="ffill")
    if "returns_1d" in df.columns:
        stock_ret = df["returns_1d"].fillna(0)
    elif "close" in df.columns:
        stock_ret = df["close"].pct_change().fillna(0)
    else:
        return df

    sector_ret = aligned[sector_ticker].pct_change().fillna(0)
    market_ret = aligned[NSE_MARKET_TICKER].pct_change().fillna(0)

    df["sector_rs_score_5d"] = compute_rs(stock_ret, sector_ret, 5)
    df["sector_rs_score_10d"] = compute_rs(stock_ret, sector_ret, 10)
    df["sector_rs_score_20d"] = compute_rs(stock_ret, sector_ret, 20)
    df["sector_rs_trend"] = df["sector_rs_score_5d"] - df["sector_rs_score_20d"]

    rs_mean = df["sector_rs_score_20d"].rolling(252).mean()
    rs_std = df["sector_rs_score_20d"].rolling(252).std() + 1e-9
    rs_z = (df["sector_rs_score_20d"] - rs_mean) / rs_std
    rank = 1.0 / (1.0 + np.exp(-rs_z))
    df["sector_rank_20d"] = rank
    df["sector_leader"] = (rank > 0.8).astype(float)
    df["sector_laggard"] = (rank < 0.2).astype(float)

    df["sector_etf_rs_market"] = compute_rs(sector_ret, market_ret, 20)
    sector_cum_20 = (1 + sector_ret).rolling(20).apply(np.prod, raw=True) - 1
    df["sector_momentum"] = sector_cum_20.clip(-0.5, 0.5)

    nifty_price = aligned[NSE_MARKET_TICKER]
    ma200 = nifty_price.rolling(200).mean()
    df["market_regime"] = (nifty_price > ma200).astype(float)

    # Goldilocks regime: Nifty Midcap 50 vs Nifty 50 on 20-bar rolling cumret
    if NSE_MIDCAP_TICKER in aligned.columns:
        mid_ret = aligned[NSE_MIDCAP_TICKER].pct_change().fillna(0)
        mid_cum = (1 + mid_ret).rolling(20).apply(np.prod, raw=True) - 1
        large_cum = (1 + market_ret).rolling(20).apply(np.prod, raw=True) - 1
        premium = mid_cum - large_cum
        df["regime_flag"] = np.where(premium > GOLDILOCKS_MIDCAP_PREMIUM, "goldilocks", "normal")
    else:
        df["regime_flag"] = "normal"

    return df


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

    # NSE path takes precedence if the symbol is Indian
    is_nse = ".NS" in symbol or ".BO" in symbol
    if is_nse:
        df = _enrich_nse_symbol(df, symbol, etf_data)
        df.to_parquet(feat_file)
        return True

    # Determine sector ETF (US path)
    base = symbol.replace(".NS", "").replace(".BO", "")
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
