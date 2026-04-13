"""
Macro Features Collector
=========================
Adds market-wide features for better signals:
- VIX and volatility indicators
- Market breadth
- Sector correlation
- Yield curve
- Dollar index (DXY)
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_API_KEYS", os.getenv("ALPHA_VANTAGE_KEY", ""))


class MacroFeatures:
    """
    Collects macro indicators to enhance trading signals.
    """
    
    def __init__(self, cache_ttl: int = 3600):
        self.cache_ttl = cache_ttl
        self._cache: Dict = {}
        self._cache_times: Dict = {}
        self._enabled = bool(ALPHA_VANTAGE_KEY)
    
    def _is_cache_valid(self, key: str) -> bool:
        if key not in self._cache:
            return False
        cache_time = self._cache_times.get(key)
        return cache_time and (datetime.utcnow() - cache_time).total_seconds() < self.cache_ttl
    
    def get_vix(self) -> float:
        """Get VIX index value."""
        cache_key = "vix"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        if not self._enabled:
            return 20.0
        
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "QUOTE_SYMBOL",
                "symbol": "^VIX",
                "apikey": ALPHA_VANTAGE_KEY,
            }
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            vix = float(data.get("global quote", {}).get("05. price", 20.0))
            self._cache[cache_key] = vix
            self._cache_times[cache_key] = datetime.utcnow()
            return vix
        except Exception as e:
            logger.debug(f"VIX fetch error: {e}")
            return 20.0
    
    def get_sp500_momentum(self) -> float:
        """Get S&P 500 momentum (20-day)."""
        cache_key = "sp500_momentum"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        if not self._enabled:
            return 0.0
        
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": "SPY",
                "apikey": ALPHA_VANTAGE_KEY,
                "outputsize": "compact",
            }
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            ts = data.get("Time Series (Daily)", {})
            if not ts:
                return 0.0
            
            dates = sorted(ts.keys())[-20:]
            if len(dates) < 2:
                return 0.0
            
            close_start = float(ts[dates[0]].get("4. close", 0))
            close_end = float(ts[dates[-1]].get("4. close", 0))
            momentum = (close_end - close_start) / close_start if close_start else 0.0
            
            self._cache[cache_key] = momentum
            self._cache_times[cache_key] = datetime.utcnow()
            return momentum
        except Exception as e:
            logger.debug(f"SPY momentum error: {e}")
            return 0.0
    
    def get_sector_performance(self) -> Dict[str, float]:
        """Get sector performance for rotation signals."""
        cache_key = "sectors"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        sectors = {
            "tech": "^XLK",
            "finance": "^XLF",
            "health": "^XLV",
            "energy": "^XLE",
            "consumer": "^XLY",
        }
        
        result = {}
        if not self._enabled:
            return {k: 0.0 for k in sectors}
        
        for name, symbol in sectors.items():
            try:
                url = "https://www.alphavantage.co/query"
                params = {
                    "function": "TIME_SERIES_DAILY",
                    "symbol": symbol,
                    "apikey": ALPHA_VANTAGE_KEY,
                    "outputsize": "compact",
                }
                resp = requests.get(url, params=params, timeout=10)
                data = resp.json()
                ts = data.get("Time Series (Daily)", {})
                if ts:
                    dates = sorted(ts.keys())[-20:]
                    if len(dates) >= 2:
                        p_start = float(ts[dates[0]].get("4. close", 0))
                        p_end = float(ts[dates[-1]].get("4. close", 0))
                        result[name] = (p_end - p_start) / p_start if p_start else 0.0
            except:
                result[name] = 0.0
        
        self._cache[cache_key] = result
        self._cache_times[cache_key] = datetime.utcnow()
        return result
    
    def get_market_regime(self) -> str:
        """Determine current market regime."""
        vix = self.get_vix()
        sp_momentum = self.get_sp500_momentum()
        
        if vix > 30 or sp_momentum < -0.05:
            return "crisis"
        elif vix > 20 or sp_momentum < -0.02:
            return "stressed"
        elif vix < 15 and sp_momentum > 0.03:
            return "bull"
        else:
            return "calm"
    
    def get_all_features(self) -> Dict[str, float]:
        """Get all macro features as dict."""
        return {
            "vix": self.get_vix(),
            "sp500_momentum": self.get_sp500_momentum(),
            "market_regime": {"crisis": 0, "stressed": 1, "calm": 2, "bull": 3}.get(self.get_market_regime(), 1),
        }


def get_macro_features() -> MacroFeatures:
    """Factory function."""
    return MacroFeatures()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mf = get_macro_features()
    print("Macro Features:", mf.get_all_features())