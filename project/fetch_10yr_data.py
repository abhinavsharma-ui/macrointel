#!/usr/bin/env python3
"""
Fetch 10 Years of Historical Data
=================================
Prefer Finnhub for US symbols, with yfinance as a fallback for unsupported
symbols such as many NSE names.

Usage:
    python fetch_10yr_data.py --symbols AAPL MSFT GOOGL
    python fetch_10yr_data.py --all
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

from pipeline.provider_utils import to_yfinance_symbol

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data" / "prices_10yr"
DATA_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_DIR / ".env")
load_dotenv(PROJECT_DIR / ".env.example", override=False)

FINNHUB_BASE_URL = "https://finnhub.io/api/v1/stock/candle"

try:
    import yfinance as yf

    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    logger.warning("yfinance not installed. Run: pip install yfinance")


DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "JNPR",
    "V", "UNH", "HD", "MA", "PG", "NV", "DIS", "ADBE", "NFLX", "CRM", "INTC",
    "CMCSA", "VZ", "KO", "PEP", "T", "DIS", "WMT", "MRK", "ABT", "CVX",
    "XOM", "PFE", "TMO", "COST", "AVGO", "LLY", "MCD", "CSCO", "ACN", "ABBV",
    "NKE", "DHR", "TXN", "NEE", "PM", "UNP", "BMY", "RTX", "HON", "LOW",
    "AMGN", "IBM", "QCOM", "SBUX", "CAT", "BA", "GE", "AMD", "GILD", "ISRG",
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "SBIN.NS", "BAJFINANCE.NS",
    "ITC.NS", "KOTAKBANK.NS", "HINDUNILVR.NS", "AXISBANK.NS", "ASIANPAINT.NS",
    "MARUTI.NS", "TITAN.NS", "SUNPHARMA.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS",
]


def _finnhub_key() -> str:
    return (os.getenv("FINNHUB_API_KEYS", "") or os.getenv("FINNHUB_API_KEY", "")).split(",")[0].strip()


def _fetch_finnhub_history(ticker: str, days: int = 3650) -> pd.DataFrame:
    api_key = _finnhub_key()
    if not api_key:
        return pd.DataFrame()

    symbol = str(ticker).strip().upper()
    if not symbol or symbol.endswith(".NS") or symbol.startswith("^"):
        return pd.DataFrame()

    start_date = date.today() - timedelta(days=days)
    end_date = date.today()
    params = {
        "symbol": symbol,
        "resolution": "D",
        "from": int(datetime.combine(start_date, datetime.min.time()).timestamp()),
        "to": int(datetime.combine(end_date, datetime.min.time()).timestamp()),
        "token": api_key,
    }

    try:
        response = requests.get(FINNHUB_BASE_URL, params=params, timeout=15)
        response.raise_for_status()
        payload = response.json()
        if payload.get("s") != "ok":
            return pd.DataFrame()

        frame = pd.DataFrame(
            {
                "Open": payload.get("o", []),
                "High": payload.get("h", []),
                "Low": payload.get("l", []),
                "Close": payload.get("c", []),
                "Volume": payload.get("v", []),
            },
            index=pd.to_datetime(payload.get("t", []), unit="s"),
        )
        if frame.empty:
            return pd.DataFrame()
        frame.index.name = None
        return frame.sort_index()
    except Exception as exc:
        logger.debug(f"{ticker}: Finnhub history error - {exc}")
        return pd.DataFrame()


def _fetch_yfinance_history(ticker: str, period: str = "10y") -> pd.DataFrame:
    if not YF_AVAILABLE:
        return pd.DataFrame()

    try:
        yf_ticker = to_yfinance_symbol(ticker) or str(ticker).strip()
        stock = yf.Ticker(yf_ticker)
        return stock.history(period=period, auto_adjust=True)
    except Exception as exc:
        logger.debug(f"{ticker}: yfinance history error - {exc}")
        return pd.DataFrame()


def fetch_symbol_data(ticker: str, period: str = "10y") -> pd.DataFrame:
    """Fetch long history for a single symbol."""
    finnhub_frame = _fetch_finnhub_history(ticker, days=3650 if period == "10y" else 365)
    if not finnhub_frame.empty:
        logger.info(f"{ticker}: fetched {len(finnhub_frame)} rows from Finnhub")
        return finnhub_frame

    yf_frame = _fetch_yfinance_history(ticker, period=period)
    if yf_frame.empty:
        logger.warning(f"{ticker}: no history returned from Finnhub or yfinance")
        return yf_frame

    logger.info(f"{ticker}: fetched {len(yf_frame)} rows from yfinance")
    return yf_frame


def fetch_batch_parallel(symbols: List[str], max_workers: int = 5) -> Dict[str, pd.DataFrame]:
    """Fetch data in parallel."""
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sym = {executor.submit(fetch_symbol_data, sym): sym for sym in symbols}

        for future in as_completed(future_to_sym):
            sym = future_to_sym[future]
            try:
                df = future.result()
                if not df.empty:
                    results[sym] = df
            except Exception as exc:
                logger.error(f"{sym}: Exception - {exc}")

    return results


def save_to_parquet(data: Dict[str, pd.DataFrame], output_dir: Path):
    """Save data to parquet files."""
    saved = 0
    for sym, df in data.items():
        if df.empty:
            continue

        frame = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )

        if "adj_close" not in frame.columns and "close" in frame.columns:
            frame["adj_close"] = frame["close"]

        required = ["open", "high", "low", "close", "volume"]
        if not all(column in frame.columns for column in required):
            logger.warning(f"{sym}: missing required columns after normalization")
            continue

        out_path = output_dir / f"{sym.replace('.NS', '_NS')}.parquet"
        frame[required + (["adj_close"] if "adj_close" in frame.columns else [])].to_parquet(
            out_path,
            compression="snappy",
        )
        saved += 1

    logger.info(f"Saved {saved} files to {output_dir}")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Fetch 10 years of data")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to fetch")
    parser.add_argument("--all", action="store_true", help="Fetch full universe from universe files")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    parser.add_argument("--limit", type=int, default=None, help="Max symbols to fetch (per universe file)")
    args = parser.parse_args()

    symbols = args.symbols
    if args.all:
        universe_files = [
            PROJECT_DIR / "data" / "universe_us_official.txt",
            PROJECT_DIR / "data" / "universe_nse_official.txt",
        ]
        symbols = []
        limits = {"us": args.limit or 500, "nse": args.limit or 200}
        for uf in universe_files:
            if uf.exists():
                tag = "nse" if "nse" in uf.name else "us"
                lines = [line.strip() for line in uf.read_text().splitlines() if line.strip() and not line.startswith("#")]
                symbols.extend(lines[:limits[tag]])
                logger.info(f"Loaded {min(len(lines), limits[tag])} symbols from {uf.name}")
        for sym in DEFAULT_SYMBOLS:
            if sym not in symbols:
                symbols.append(sym)
        symbols = list(dict.fromkeys(symbols))

    if not symbols:
        symbols = DEFAULT_SYMBOLS[:20]

    logger.info(f"Fetching 10 years for {len(symbols)} symbols...")
    logger.info("Finnhub-first collector enabled")

    data = fetch_batch_parallel(symbols, max_workers=args.workers)

    if not data:
        logger.error("No data fetched!")
        return 1

    saved = save_to_parquet(data, DATA_DIR)
    logger.info(f"Complete! Fetched {len(data)} symbols, saved {saved} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
