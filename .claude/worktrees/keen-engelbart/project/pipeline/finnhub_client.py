"""
Finnhub Data Client
===================
Free real-time market data + fundamentals.
Replaces/supplements Yahoo Finance with institutional-grade data.

Free tier: Unlimited API calls for real-time quotes
Documentation: https://finnhub.io/docs/api

Features:
- Real-time stock quotes (US, forex, crypto)
- Company fundamentals (income statement, balance sheet, cash flow)
- Earnings calendar
- News & sentiment
- Technical indicators
- Crypto market data
"""

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import requests

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEYS", os.getenv("FINNHUB_API_KEY", ""))
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"


class FinnhubClient:
    """
    Client for Finnhub API with caching and rate limiting.
    
    Usage:
        client = FinnhubClient()
        quote = client.get_quote("AAPL")
        fundamentals = client.get_company fundamentals("AAPL")
        news = client.get_news("AAPL")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_ttl: int = 300,
        rate_limit_pause: float = 0.5,
    ):
        self.api_key = api_key or FINNHUB_API_KEY
        self.cache_ttl = cache_ttl
        self.rate_limit_pause = rate_limit_pause
        self._cache: Dict[str, Any] = {}
        self._cache_times: Dict[str, datetime] = {}
        
        if not self.api_key or self.api_key == "YOUR_FINNHUB_API_KEY_HERE":
            logger.warning("Finnhub API key not set. Get free key at https://finnhub.io/")
            self._enabled = False
        else:
            self._enabled = True
    
    def _is_cache_valid(self, key: str) -> bool:
        if key not in self._cache:
            return False
        cache_time = self._cache_times.get(key)
        if cache_time is None:
            return False
        return (datetime.utcnow() - cache_time).total_seconds() < self.cache_ttl
    
    def _get_cached(self, key: str) -> Optional[Any]:
        if self._is_cache_valid(key):
            return self._cache.get(key)
        return None
    
    def _set_cache(self, key: str, value: Any):
        self._cache[key] = value
        self._cache_times[key] = datetime.utcnow()
    
    def _request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        if not self._enabled:
            return None
        
        params = params or {}
        params["token"] = self.api_key
        
        cache_key = f"{endpoint}:{str(params)}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        try:
            time.sleep(self.rate_limit_pause)
            response = requests.get(
                f"{FINNHUB_BASE_URL}{endpoint}",
                params=params,
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            self._set_cache(cache_key, data)
            return data
        except Exception as e:
            logger.error(f"Finnhub API error ({endpoint}): {e}")
            return None
    
    def get_quote(self, symbol: str) -> Dict:
        """
        Get real-time quote for a symbol.
        
        Returns:
            {
                "symbol": "AAPL",
                "current_price": 175.50,
                "high_price": 177.20,
                "low_price": 174.80,
                "open_price": 176.00,
                "prev_close": 175.20,
                "timestamp": "2026-04-11T12:00:00"
            }
        """
        cache_key = f"quote:{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        data = self._request("/quote", {"symbol": symbol})
        if not data:
            return {"symbol": symbol, "error": "No data"}
        
        result = {
            "symbol": symbol,
            "current_price": data.get("c", 0),
            "high_price": data.get("h", 0),
            "low_price": data.get("l", 0),
            "open_price": data.get("o", 0),
            "prev_close": data.get("pc", 0),
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._set_cache(cache_key, result)
        return result
    
    def get_company_profile(self, symbol: str) -> Dict:
        """
        Get company profile/metadata.
        
        Returns:
            {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "industry": "Technology",
                "sector": "Consumer Electronics",
                "market_cap": 2800000000000,
                "exchange": "NASDAQ",
                "currency": "USD"
            }
        """
        data = self._request("/stock/profile2", {"symbol": symbol})
        if not data:
            return {"symbol": symbol, "error": "No data"}
        
        return {
            "symbol": symbol,
            "name": data.get("name", ""),
            "industry": data.get("finnhubIndustry", ""),
            "sector": data.get("shareOutstanding", 0) * data.get("marketCapitalization", 0),
            "market_cap": data.get("marketCapitalization", 0),
            "exchange": data.get("exchange", ""),
            "currency": data.get("currency", "USD"),
        }
    
    def get_financials(self, symbol: str, freq: str = "annual") -> Dict:
        """
        Get financial statements.
        
        Args:
            symbol: Stock symbol
            freq: "annual" or "quarterly"
        
        Returns:
            Income statement, balance sheet, cash flow
        """
        data = self._request("/stock/financials", {
            "symbol": symbol,
            "frequency": freq,
        })
        if not data:
            return {"symbol": symbol, "error": "No data"}
        
        return {
            "symbol": symbol,
            "income": data.get("income", []),
            "balance_sheet": data.get("balanceSheet", []),
            "cash_flow": data.get("cashflow", []),
        }
    
    def get_metrics(self, symbol: str) -> Dict:
        """
        Get key metrics and ratios.
        
        Returns:
            {
                "symbol": "AAPL",
                "pe_ratio": 28.5,
                "peg_ratio": 1.8,
                "dividend_yield": 0.52,
                "beta": 1.2,
                "52w_high": 198.50,
                "52w_low": 142.30,
            }
        """
        data = self._request("/stock/metric", {
            "symbol": symbol,
            "metric": "all",
        })
        if not data:
            return {"symbol": symbol, "error": "No data"}
        
        metrics = data.get("metrics", {})
        return {
            "symbol": symbol,
            "pe_ratio": metrics.get("peRatioTTM"),
            "peg_ratio": metrics.get("pegRatioTTM"),
            "dividend_yield": metrics.get("dividendYieldIndicator"),
            "beta": metrics.get("beta"),
            "52w_high": metrics.get("52WeekHigh"),
            "52w_low": metrics.get("52WeekLow"),
            "avg_volume": metrics.get("avg10DayVolume"),
            "avg_vol_30d": metrics.get("avg30DayVolume"),
        }
    
    def get_recommendations(self, symbol: str) -> List[Dict]:
        """
        Get analyst recommendations.
        
        Returns:
            [
                {
                    "period": "2024-Q1",
                    "strong_buy": 15,
                    "buy": 20,
                    "hold": 8,
                    "sell": 2,
                    "strong_sell": 1
                }
            ]
        """
        data = self._request("/stock/recommendation", {"symbol": symbol})
        if not data:
            return []
        
        return data.get("data", [])
    
    def get_earnings_calendar(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> List[Dict]:
        """
        Get earnings calendar.
        
        Args:
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
        """
        if not from_date:
            from_date = datetime.utcnow().strftime("%Y-%m-%d")
        if not to_date:
            to_date = (datetime.utcnow() + timedelta(days=30)).strftime("%Y-%m-%d")
        
        data = self._request("/calendar/earnings", {
            "from": from_date,
            "to": to_date,
        })
        if not data:
            return []
        
        return data.get("earningsCalendar", [])
    
    def get_news(
        self,
        symbol: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        category: str = "general",
    ) -> List[Dict]:
        """
        Get market news.
        
        Args:
            symbol: Optional stock symbol to filter
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            category: "general", "forex", "crypto", "merger"
        """
        params = {"category": category}
        
        if symbol:
            params["symbol"] = symbol
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        
        data = self._request("/news", params)
        if not data:
            return []
        
        return [
            {
                "headline": item.get("headline", ""),
                "summary": item.get("summary", ""),
                "source": item.get("source", ""),
                "url": item.get("url", ""),
                "datetime": item.get("datetime"),
                "symbol": item.get("symbol", []),
            }
            for item in data
        ]
    
    def get_sentiment(self, symbol: str) -> Dict:
        """
        Get social sentiment data (Reddit, Twitter).
        
        Returns:
            {
                "symbol": "AAPL",
                "reddit_mention_count": 150,
                "twitter_mention_count": 320,
                "reddit_sentiment": 0.65,
                "twitter_sentiment": 0.58,
            }
        """
        data = self._request("/stock/social-sentiment", {"symbol": symbol})
        if not data:
            return {"symbol": symbol, "error": "No data"}
        
        reddit = data.get("reddit", [{}])[0] if data.get("reddit") else {}
        twitter = data.get("twitter", [{}])[0] if data.get("twitter") else {}
        
        return {
            "symbol": symbol,
            "reddit_mention_count": reddit.get("mentionVolume", 0),
            "twitter_mention_count": twitter.get("mentionVolume", 0),
            "reddit_sentiment": reddit.get("avgSentiment", 0),
            "twitter_sentiment": twitter.get("avgSentiment", 0),
        }
    
    def get_crypto_quotes(self, symbol: str) -> Dict:
        """
        Get crypto quotes (e.g., "BTC-USD").
        """
        cache_key = f"crypto:{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        data = self._request("/crypto/quote", {"symbol": symbol})
        if not data:
            return {"symbol": symbol, "error": "No data"}
        
        result = {
            "symbol": symbol,
            "current_price": data.get("c", 0),
            "high_price": data.get("h", 0),
            "low_price": data.get("l", 0),
            "volume": data.get("v", 0),
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._set_cache(cache_key, result)
        return result
    
    def get_forex_quotes(self, symbol: str) -> Dict:
        """
        Get forex quotes (e.g., "EUR/USD").
        """
        cache_key = f"forex:{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        data = self._request("/forex/quote", {"symbol": symbol})
        if not data:
            return {"symbol": symbol, "error": "No data"}
        
        result = {
            "symbol": symbol,
            "bid": data.get("b", 0),
            "ask": data.get("a", 0),
            "mid": (data.get("b", 0) + data.get("a", 0)) / 2,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._set_cache(cache_key, result)
        return result
    
    def get_support_resistance(self, symbol: str, resolution: str = "D") -> List[float]:
        """
        Get support/resistance levels.
        """
        data = self._request("/scan/support-resistance", {
            "symbol": symbol,
            "resolution": resolution,
        })
        if not data:
            return []
        
        return data.get("levels", [])
    
    def get_technical_indicator(
        self,
        symbol: str,
        indicator: str = "rsi",
        resolution: str = "D",
        period: int = 14,
    ) -> List[Dict]:
        """
        Get technical indicator data.
        
        Args:
            indicator: "rsi", "macd", "stoch", "adx", "cci"
            resolution: "1", "5", "15", "30", "60", "D", "W", "M"
            period: Indicator period
        """
        to_ts = int(datetime.utcnow().timestamp())
        from_ts = int((datetime.utcnow() - timedelta(days=60)).timestamp())
        
        data = self._request("/indicator", {
            "symbol": symbol,
            "type": indicator,
            "resolution": resolution,
            "from": from_ts,
            "to": to_ts,
            "indicator_params": {"period": period},
        })
        if not data:
            return []
        
        return data.get("values", [])
    
    def get_batch_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get multiple quotes at once (batched for efficiency).
        """
        quotes = {}
        for symbol in symbols:
            try:
                quotes[symbol] = self.get_quote(symbol)
            except Exception as e:
                logger.warning(f"Failed to get quote for {symbol}: {e}")
                quotes[symbol] = {"symbol": symbol, "error": str(e)}
        return quotes


def get_finnhub_client() -> FinnhubClient:
    """Factory function to get Finnhub client."""
    return FinnhubClient()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = FinnhubClient()
    
    if client._enabled:
        print("Testing Finnhub API...")
        
        quote = client.get_quote("AAPL")
        print(f"AAPL Quote: {quote}")
        
        metrics = client.get_metrics("AAPL")
        print(f"AAPL Metrics: {metrics}")
        
        news = client.get_news("AAPL", category="general")
        print(f"AAPL News: {len(news)} articles")
    else:
        print("Finnhub not configured. Add FINNHUB_API_KEYS to .env")