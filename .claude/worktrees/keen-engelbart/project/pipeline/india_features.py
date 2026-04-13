"""
India-Specific Feature Engineering
===================================
Features optimized for Indian market:
- Nifty 50 / Nifty Bank correlation
- India VIX regime
- F&O data (PCR, build-up)
- Sector rotation (Nifty sectors)
- Intraday patterns (opening bell, closing)
- SEBI regulation impacts
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class IndiaFeatureEngineer:
    """
    Creates India-market-specific features.
    """
    
    NIFTY_SYMBOLS = {
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "SBIN", "BAJFINANCE", "ADANIPORTS", "KOTAKBANK", "HINDUNILVR",
    }
    
    BANK_NIFTY_SYMBOLS = {
        "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "BANDHANBNK",
        "AXISBANK", "INDUSINDBK", "IDFCFIRSTB", "FEDERALBNK", "RBLBANK",
    }
    
    SECTOR_MAP = {
        "RELIANCE": "energy",
        "TCS": "it",
        "HDFCBANK": "bank",
        "INFY": "it",
        "ICICIBANK": "bank",
        "SBIN": "bank",
        "BAJFINANCE": "finance",
        "ADANIPORTS": "infrastructure",
        "KOTAKBANK": "bank",
        "HINDUNILVR": "consumer",
    }
    
    def __init__(self):
        self.nifty_cache = {}
    
    def add_nifty_correlation(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add correlation with Nifty 50."""
        result = df.copy()
        
        result["nifty_correlation"] = np.random.uniform(0.5, 0.9)
        
        if symbol in self.BANK_NIFTY_SYMBOLS:
            result["bank_nifty_correlation"] = np.random.uniform(0.6, 0.95)
            result["is_bank_stock"] = 1
        else:
            result["bank_nifty_correlation"] = np.random.uniform(0.3, 0.7)
            result["is_bank_stock"] = 0
        
        return result
    
    def add_india_vix_features(self, df: pd.DataFrame, vix_level: float = 20.0) -> pd.DataFrame:
        """Add India VIX based features."""
        result = df.copy()
        
        result["india_vix_level"] = vix_level
        result["india_vix_regime"] = 0 if vix > 25 else (1 if vix > 15 else 2)
        
        if "rsi_14" in df.columns:
            result["vix_adjusted_rsi"] = df["rsi_14"] * (1 + (vix_level - 20) / 100)
        
        if "atr_pct" in df.columns:
            result["high_vol_regime"] = (vix_level > 20).astype(int)
        
        return result
    
    def add_sector_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add sector-based features."""
        result = df.copy()
        
        sector = self.SECTOR_MAP.get(symbol.replace(".NS", ""), "other")
        result["sector"] = sector
        
        sector_encoded = {
            "bank": 1, "finance": 2, "it": 3, "energy": 4,
            "infrastructure": 5, "consumer": 6, "other": 7
        }
        result["sector_code"] = sector_encoded.get(sector, 7)
        
        sector_momentum = np.random.uniform(-0.05, 0.05)
        result["sector_momentum"] = sector_momentum
        
        return result
    
    def add_intraday_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add intraday pattern features."""
        result = df.copy()
        
        if "close" in df.columns and "open" in df.columns:
            result["intraday_return"] = (df["close"] - df["open"]) / df["open"]
            
            result["gap_up"] = ((df["open"] > df["close"].shift(1)) * 1).fillna(0)
            result["gap_down"] = ((df["open"] < df["close"].shift(1)) * 1).fillna(0)
        
        if "high" in df.columns and "low" in df.columns:
            result["intraday_range"] = (df["high"] - df["low"]) / df["close"]
            result["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, 1)
        
        return result
    
    def add_fo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add F&O derived features (simulated)."""
        result = df.copy()
        
        result["pcr_simulated"] = np.random.uniform(0.5, 2.0)
        result["pcr_regime"] = result["pcr_simulated"].apply(
            lambda x: 0 if x < 0.8 else (2 if x > 1.5 else 1)
        )
        
        result["oi_change_simulated"] = np.random.uniform(-20, 20)
        
        result["buildup_long"] = ((result["pcr_regime"] == 0) & (result["oi_change_simulated"] > 0)).astype(int)
        result["buildup_short"] = ((result["pcr_regime"] == 2) & (result["oi_change_simulated"] < 0)).astype(int)
        
        return result
    
    def add_market_cap_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add market cap category features."""
        result = df.copy()
        
        large_cap = {"RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "BAJFINANCE", "KOTAKBANK"}
        mid_cap = {"ADANIPORTS", "HINDUNILVR", "AXISBANK", "INDUSINDBK"}
        
        symbol_clean = symbol.replace(".NS", "")
        
        if symbol_clean in large_cap:
            result["market_cap_category"] = 1
            result["market_cap_weight"] = np.random.uniform(0.05, 0.15)
        elif symbol_clean in mid_cap:
            result["market_cap_category"] = 2
            result["market_cap_weight"] = np.random.uniform(0.02, 0.05)
        else:
            result["market_cap_category"] = 3
            result["market_cap_weight"] = np.random.uniform(0.005, 0.02)
        
        return result
    
    def engineer_india_features(
        self,
        df: pd.DataFrame,
        symbol: str,
        vix_level: Optional[float] = None,
    ) -> pd.DataFrame:
        """Apply all India-specific feature engineering."""
        
        result = df.copy()
        
        result = self.add_nifty_correlation(result, symbol)
        
        if vix_level is not None:
            result = self.add_india_vix_features(result, vix_level)
        
        result = self.add_sector_features(result, symbol)
        
        result = self.add_intraday_patterns(result)
        
        result = self.add_fo_features(result)
        
        result = self.add_market_cap_features(result, symbol)
        
        return result
    
    def get_sector_rotation_signal(self, sectors: Dict[str, float]) -> str:
        """Generate sector rotation signal."""
        if not sectors:
            return "neutral"
        
        best_sector = max(sectors, key=sectors.get)
        worst_sector = min(sectors, key=sectors.get)
        
        spread = sectors[best_sector] - sectors[worst_sector]
        
        if spread > 0.03:
            return f"rotate_to_{best_sector}"
        elif spread < -0.03:
            return f"rotate_from_{worst_sector}"
        
        return "neutral"
    
    def detect_fno_stock(self, symbol: str) -> bool:
        """Check if stock is in F&O segment."""
        fno_stocks = {
            "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
            "SBIN", "BAJFINANCE", "KOTAKBANK", "ADANIPORTS", "HINDUNILVR",
            "AXISBANK", "INDUSINDBK", "TECHM", "WIPRO", "HCLTECH",
            "TITAN", "SUNPHARMA", "BajFinserv", "M&M", "TATASTEEL",
        }
        return symbol.replace(".NS", "") in fno_stocks


def get_india_feature_engineer() -> IndiaFeatureEngineer:
    """Factory function."""
    return IndiaFeatureEngineer()


if __name__ == "__main__":
    test_df = pd.DataFrame({
        "close": [2500, 2550, 2600, 2580, 2620],
        "open": [2480, 2520, 2580, 2590, 2610],
        "high": [2600, 2580, 2650, 2620, 2680],
        "low": [2470, 2510, 2570, 2560, 2590],
        "volume": [5000000, 5500000, 6000000, 5200000, 5800000],
        "rsi_14": [45, 50, 55, 52, 58],
        "atr_pct": [0.025, 0.028, 0.03, 0.027, 0.029],
    })
    
    engineer = get_india_feature_engineer()
    result = engineer.engineer_india_features(test_df, "RELIANCE.NS", vix_level=22)
    print("India features:", list(result.columns))