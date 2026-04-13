"""
Improved Feature Engineer
=========================
Enhanced feature engineering with:
- Macro features (VIX, market regime)
- Better momentum indicators
- Cross-asset signals
- Improved RSI/MACD combinations
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ImprovedFeatureEngineer:
    """
    Creates enhanced features for better predictions.
    """
    
    def __init__(self, use_market_context: bool = True):
        self.use_market_context = use_market_context
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced features to existing feature matrix."""
        result = df.copy()
        
        result = self._add_momentum_composite(result)
        result = self._add_rsi_divergence(result)
        result = self._add_macd_crossover_signal(result)
        result = self._add_volatility_adjusted_features(result)
        result = self._add_volume_momentum(result)
        
        return result
    
    def _add_momentum_composite(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite momentum score."""
        if "roc_5" in df.columns and "roc_10" in df.columns and "roc_20" in df.columns:
            df["momentum_composite"] = (
                df["roc_5"].fillna(0) * 0.3 +
                df["roc_10"].fillna(0) * 0.4 +
                df["roc_20"].fillna(0) * 0.3
            )
        return df
    
    def _add_rsi_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI divergence signals."""
        if "rsi_14" in df.columns and "close" in df.columns:
            rsi = df["rsi_14"]
            close = df["close"]
            
            rsi_ma5 = rsi.rolling(5).mean()
            rsi_ma10 = rsi.rolling(10).mean()
            
            price_ret = close.pct_change(5)
            rsi_ret = rsi.pct_change(5)
            
            divergence = (rsi_ret > 0) & (price_ret < 0)
            hidden_bullish = (rsi_ret < 0) & (price_ret > 0)
            
            df["rsi_divergence"] = divergence.astype(int) - hidden_bullish.astype(int)
            df["rsi_ma_diff"] = (rsi_ma5 - rsi_ma10) / 100
        
        return df
    
    def _add_macd_crossover_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD crossover detection."""
        if "macd" in df.columns and "macd_signal" in df.columns:
            macd = df["macd"]
            signal = df["macd_signal"]
            
            prev_macd = macd.shift(1)
            prev_signal = signal.shift(1)
            
            bullish_cross = (macd > signal) & (prev_macd <= prev_signal)
            bearish_cross = (macd < signal) & (prev_macd >= prev_signal)
            
            df["macd_bullish_cross"] = bullish_cross.astype(int)
            df["macd_bearish_cross"] = bearish_cross.astype(int)
            
            histogram = df.get("macd_hist", macd - signal)
            df["macd_histogram_trend"] = histogram.diff()
        
        return df
    
    def _add_volatility_adjusted_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-adjusted momentum."""
        if "momentum_composite" in df.columns:
            if "realized_vol_21d" in df.columns:
                vol = df["realized_vol_21d"].fillna(0.02)
                vol = vol.replace(0, 0.02)
                df["vol_adjusted_momentum"] = df["momentum_composite"] / vol
            elif "hist_vol_10" in df.columns:
                vol = df["hist_vol_10"].fillna(0.02).replace(0, 0.02)
                df["vol_adjusted_momentum"] = df["momentum_composite"] / vol
        
        if "atr_pct" in df.columns:
            df["atr_normalized"] = df["atr_pct"] / df["atr_pct"].rolling(20).mean()
        
        return df
    
    def _add_volume_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based signals."""
        if "volume" in df.columns and "obv" in df.columns:
            vol_ma = df["volume"].rolling(20).mean()
            df["volume_ratio"] = df["volume"] / vol_ma
            df["volume_spike"] = (df["volume"] > vol_ma * 1.5).astype(int)
            
            df["obv_slope_5"] = df["obv"].pct_change(5)
            df["obv_momentum"] = np.sign(df["obv_slope_5"])
        
        if "mfi" in df.columns:
            df["mfi_extreme"] = ((df["mfi"] < 20) | (df["mfi"] > 80)).astype(int)
        
        return df
    
    def add_market_context(self, df: pd.DataFrame, macro_features: Dict) -> pd.DataFrame:
        """Add macro market context to features."""
        result = df.copy()
        
        if "vix" in macro_features:
            vix = macro_features["vix"]
            result["vix_level"] = vix / 40
            result["vix_regime"] = 1 if vix > 25 else 0 if vix < 15 else 0.5
        
        if "market_regime" in macro_features:
            result["market_regime"] = macro_features["market_regime"]
        
        if "sp500_momentum" in macro_features:
            result["market_momentum"] = macro_features["sp500_momentum"] * 10
        
        if "rsi_14" in result.columns and "market_regime" in result.columns:
            regime = result["market_regime"]
            rsi = result["rsi_14"]
            
            result["regime_adjusted_rsi"] = np.where(
                regime == 0,
                np.clip(rsi + 10, 0, 100),
                np.where(regime == 2, np.clip(rsi - 5, 0, 100), rsi)
            )
        
        return result


def get_improved_engineer() -> ImprovedFeatureEngineer:
    """Factory function."""
    return ImprovedFeatureEngineer()


if __name__ == "__main__":
    test_df = pd.DataFrame({
        "rsi_14": [30, 50, 70, 40, 60],
        "close": [100, 105, 102, 108, 110],
        "roc_5": [0.02, 0.01, -0.01, 0.03, 0.02],
        "roc_10": [0.03, 0.02, 0.01, 0.04, 0.03],
        "roc_20": [0.05, 0.04, 0.02, 0.06, 0.05],
        "macd": [1.5, 2.0, 1.8, 2.5, 3.0],
        "macd_signal": [1.2, 1.5, 1.6, 1.9, 2.2],
        "volume": [1000000, 1200000, 900000, 1500000, 1400000],
        "obv": [1000000, 2200000, 3100000, 4600000, 6000000],
        "realized_vol_21d": [0.02, 0.025, 0.03, 0.022, 0.018],
    })
    
    engineer = get_improved_engineer()
    result = engineer.engineer_features(test_df)
    print("Enhanced features:", list(result.columns))