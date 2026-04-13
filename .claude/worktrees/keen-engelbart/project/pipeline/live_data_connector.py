"""
Real-Time Data Connector
=========================
Connects to live market data:
- Yahoo Finance (free, global)
- Alpha Vantage (free tier)
- Finnhub (free tier, already configured)
"""

import logging
import time
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    YF_OK = True
except ImportError:
    YF_OK = False


class LiveDataConnector:
    """
    Connects to real-time market data.
    Falls back between sources.
    """
    
    def __init__(self, cache_seconds: int = 60):
        self.cache_seconds = cache_seconds
        self._cache: Dict[str, Dict] = {}
        self._cache_times: Dict[str, datetime] = {}
    
    def _is_cache_valid(self, symbol: str) -> bool:
        key = f"quote_{symbol}"
        if key not in self._cache:
            return False
        cache_time = self._cache_times.get(key)
        return cache_time and (datetime.now() - cache_time).total_seconds() < self.cache_seconds
    
    def get_quote_yahoo(self, symbol: str) -> Optional[Dict]:
        """Get quote from Yahoo Finance."""
        if not YF_OK:
            return None
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                "symbol": symbol,
                "price": info.get("currentPrice") or info.get("regularMarketPrice", 0),
                "open": info.get("open", 0),
                "high": info.get("dayHigh", 0),
                "low": info.get("dayLow", 0),
                "volume": info.get("volume", 0),
                "prev_close": info.get("previousClose", 0),
                "change": info.get("regularMarketChange", 0),
                "change_pct": info.get("regularMarketChangePercent", 0),
                "timestamp": datetime.now(),
            }
        except Exception as e:
            logger.debug(f"Yahoo quote error for {symbol}: {e}")
            return None
    
    def get_quote_alpha_vantage(self, symbol: str, api_key: str) -> Optional[Dict]:
        """Get quote from Alpha Vantage."""
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": api_key,
            }
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            
            quote = data.get("Global Quote", {})
            if not quote:
                return None
            
            return {
                "symbol": symbol,
                "price": float(quote.get("05. price", 0)),
                "open": float(quote.get("02. open", 0)),
                "high": float(quote.get("03. high", 0)),
                "low": float(quote.get("04. low", 0)),
                "volume": int(quote.get("06. volume", 0)),
                "prev_close": float(quote.get("08. previous close", 0)),
                "change": float(quote.get("09. change", 0)),
                "change_pct": float(quote.get("10. change percent", "0").replace("%", "")),
                "timestamp": datetime.now(),
            }
        except Exception as e:
            logger.debug(f"Alpha Vantage quote error for {symbol}: {e}")
            return None
    
    def get_quote_finnhub(self, symbol: str, api_key: str) -> Optional[Dict]:
        """Get quote from Finnhub."""
        try:
            url = "https://finnhub.io/api/v1/quote"
            params = {"symbol": symbol, "token": api_key}
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            
            if "c" not in data:
                return None
            
            return {
                "symbol": symbol,
                "price": data.get("c", 0),
                "open": data.get("o", 0),
                "high": data.get("h", 0),
                "low": data.get("l", 0),
                "prev_close": data.get("pc", 0),
                "change": data.get("c", 0) - data.get("pc", 0),
                "change_pct": ((data.get("c", 0) - data.get("pc", 0)) / data.get("pc", 1) * 100) if data.get("pc") else 0,
                "volume": 0,
                "timestamp": datetime.now(),
            }
        except Exception as e:
            logger.debug(f"Finnhub quote error for {symbol}: {e}")
            return None
    
    def get_quote(self, symbol: str, prefer: str = "yahoo") -> Optional[Dict]:
        """
        Get quote with fallback.
        
        Args:
            symbol: Stock/crypto symbol (e.g., "RELIANCE.NS", "BTC-USD")
            prefer: Preferred source ("yahoo", "finnhub", "alphavantage")
        """
        if self._is_cache_valid(symbol):
            return self._cache[f"quote_{symbol}"]
        
        sources = [prefer]
        if prefer == "yahoo":
            sources.extend(["finnhub", "alphavantage"])
        elif prefer == "finnhub":
            sources.extend(["alphavantage", "yahoo"])
        else:
            sources.extend(["finnhub", "yahoo"])
        
        import os
        finnhub_key = os.getenv("FINNHUB_API_KEYS", "")
        alpha_key = os.getenv("ALPHA_VANTAGE_API_KEYS", "")
        
        for source in sources:
            try:
                if source == "yahoo":
                    quote = self.get_quote_yahoo(symbol)
                    if quote:
                        self._cache[f"quote_{symbol}"] = quote
                        self._cache_times[f"quote_{symbol}"] = datetime.now()
                        return quote
                
                elif source == "finnhub" and finnhub_key:
                    quote = self.get_quote_finnhub(symbol, finnhub_key)
                    if quote:
                        self._cache[f"quote_{symbol}"] = quote
                        self._cache_times[f"quote_{symbol}"] = datetime.now()
                        return quote
                
                elif source == "alphavantage" and alpha_key:
                    quote = self.get_quote_alpha_vantage(symbol, alpha_key)
                    if quote:
                        self._cache[f"quote_{symbol}"] = quote
                        self._cache_times[f"quote_{symbol}"] = datetime.now()
                        return quote
            except Exception as e:
                logger.debug(f"Source {source} failed: {e}")
                continue
        
        logger.warning(f"All sources failed for {symbol}")
        return None
    
    def get_quotes_batch(self, symbols: list) -> Dict[str, Dict]:
        """Get quotes for multiple symbols."""
        quotes = {}
        for symbol in symbols:
            quote = self.get_quote(symbol)
            if quote:
                quotes[symbol] = quote
            time.sleep(0.5)
        return quotes
    
    def get_historical(self, symbol: str, period: str = "1mo", interval: str = "1d") -> Optional[pd.DataFrame]:
        """Get historical data."""
        if not YF_OK:
            return None
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            return df
        except Exception as e:
            logger.debug(f"Historical data error for {symbol}: {e}")
            return None


def get_live_data_connector() -> LiveDataConnector:
    """Factory function."""
    return LiveDataConnector()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    connector = get_live_data_connector()
    
    print("Testing live data...")
    
    test_symbols = ["RELIANCE.NS", "INFY.NS", "AAPL", "BTC-USD"]
    
    for symbol in test_symbols:
        quote = connector.get_quote(symbol)
        if quote:
            print(f"  {symbol}: ₹{quote.get('price', 0):.2f} ({quote.get('change_pct', 0):.2f}%)")
        else:
            print(f"  {symbol}: No data")