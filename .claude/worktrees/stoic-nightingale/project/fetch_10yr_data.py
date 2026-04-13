#!/usr/bin/env python3
"""
Fetch 10 Years of Historical Data
==============================
Uses yfinance (free) to fetch 10 years of daily OHLCV data
for all symbols in the universe.

Usage:
    python fetch_10yr_data.py --symbols AAPL MSFT GOOGL
    python fetch_10yr_data.py --all
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data" / "prices_10yr"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Try to import yfinance
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    logger.warning("yfinance not installed. Run: pip install yfinance")

# Default universe
DEFAULT_SYMBOLS = [
    # US Large Cap
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "JNPR",
    "V", "UNH", "HD", "MA", "PG", "NV", "DIS", "ADBE", "NFLX", "CRM", "INTC",
    "CMCSA", "VZ", "KO", "PEP", "T", "DIS", "WMT", "MRK", "ABT", "CVX",
    "XOM", "PFE", "TMO", "COST", "AVGO", "LLY", "MCD", "CSCO", "ACN", "ABBV",
    "NKE", "DHR", "TXN", "NEE", "PM", "UNP", "BMY", "RTX", "HON", "LOW",
    "AMGN", "IBM", "QCOM", "SBUX", "CAT", "BA", "GE", "AMD", "GILD", "ISRG",
    # Indian NSE
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "SBIN.NS", "BAJFINANCE.NS",
    "ITC.NS", "KOTAKBANK.NS", "HINDUNILVR.NS", "AXISBANK.NS", "ASIANPAINT.NS",
    "MARUTI.NS", "TITAN.NS", "SUNPHARMA.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS",
]


def fetch_symbol_data(ticker: str, period: str = "10y") -> pd.DataFrame:
    """Fetch 10 years of data for a single symbol."""
    try:
        # Map to yfinance symbol
        yf_ticker = ticker.replace(".NS", "").replace(".BO", "")
        
        stock = yf.Ticker(yf_ticker)
        df = stock.history(period=period, auto_adjust=True)
        
        if df.empty or len(df) < 1000:
            logger.warning(f"{ticker}: Only {len(df)} rows fetched")
            return df
        
        logger.info(f"{ticker}: {len(df)} rows ({df.index.min()} to {df.index.max()})")
        return df
        
    except Exception as e:
        logger.error(f"{ticker}: Error - {e}")
        return pd.DataFrame()


def fetch_batch_parallel(symbols: List[str], max_workers: int = 5) -> Dict[str, pd.DataFrame]:
    """Fetch data in parallel."""
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sym = {
            executor.submit(fetch_symbol_data, sym): sym 
            for sym in symbols
        }
        
        for future in as_completed(future_to_sym):
            sym = future_to_sym[future]
            try:
                df = future.result()
                if not df.empty:
                    results[sym] = df
            except Exception as e:
                logger.error(f"{sym}: Exception - {e}")
    
    return results


def save_to_parquet(data: Dict[str, pd.DataFrame], output_dir: Path):
    """Save data to parquet files."""
    saved = 0
    for sym, df in data.items():
        if df.empty:
            continue
        
        # Clean columns
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low", 
            "Close": "close", "Volume": "volume"
        })
        
        if "Close" in df.columns:
            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume"
            })
        
        # Ensure required columns
        required = ["open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in required):
            continue
        
        # Save
        out_path = output_dir / f"{sym.replace('.NS', '_NS')}.parquet"
        df[required].to_parquet(out_path, compression="snappy")
        saved += 1
    
    logger.info(f"Saved {saved} files to {output_dir}")
    return saved


def main():
    if not YF_AVAILABLE:
        logger.error("yfinance not installed!")
        return 1
    
    parser = argparse.ArgumentParser(description="Fetch 10 years of data")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to fetch")
    parser.add_argument("--all", action="store_true", help="Fetch full universe")
    parser.add_argument("--workers", type=int, default=5, help="Parallel workers")
    args = parser.parse_args()
    
    symbols = args.symbols
    if args.all:
        symbols = DEFAULT_SYMBOLS
    
    if not symbols:
        symbols = DEFAULT_SYMBOLS[:20]  # Default: top 20
    
    logger.info(f"Fetching 10 years for {len(symbols)} symbols...")
    logger.info(f"Using yfinance (free, up to 10 years)")
    
    # Fetch
    data = fetch_batch_parallel(symbols, max_workers=args.workers)
    
    if not data:
        logger.error("No data fetched!")
        return 1
    
    # Save
    saved = save_to_parquet(data, DATA_DIR)
    
    logger.info(f"Complete! Fetched {len(data)} symbols, saved {saved} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())