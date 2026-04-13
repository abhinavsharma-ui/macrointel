"""
Market-Normalized Feature Scaling
==================================
Different markets (US, India, Crypto) have different:
- Volatility profiles
- Price ranges
- Volume scales
- Typical RSI/Momentum values

This module provides market-aware scaling to normalize features
across different market types for consistent model performance.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

MARKET_TYPES = ["us", "india", "crypto", "forex"]
DEFAULT_MARKET = "us"

VOLATILITY_PROFILES = {
    "crypto": {
        "typical_vol": 0.04,
        "max_vol": 0.15,
        "atr_mult": 2.5,
        "rsi_oversold": 35,
        "rsi_overbought": 65,
        "momentum_threshold": 0.03,
    },
    "forex": {
        "typical_vol": 0.008,
        "max_vol": 0.03,
        "atr_mult": 1.5,
        "rsi_oversold": 35,
        "rsi_overbought": 65,
        "momentum_threshold": 0.005,
    },
    "india": {
        "typical_vol": 0.02,
        "max_vol": 0.08,
        "atr_mult": 2.0,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "momentum_threshold": 0.015,
    },
    "us": {
        "typical_vol": 0.015,
        "max_vol": 0.06,
        "atr_mult": 2.0,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "momentum_threshold": 0.01,
    },
}


class MarketNormalizedScaler(BaseEstimator, TransformerMixin):
    """
    Market-aware feature scaler that adapts scaling parameters
    based on market type (US, India, Crypto, Forex).
    
    For each market type, we:
    1. Use different scaler types (Standard vs Robust vs MinMax)
    2. Apply market-specific clipping thresholds
    3. Normalize momentum/volatility features to market-typical ranges
    4. Apply regime-aware adjustments
    """
    
    def __init__(
        self,
        market_type: str = "us",
        use_robust: bool = True,
        clip_percentiles: Tuple[float, float] = (1, 99),
    ):
        self.market_type = market_type.lower()
        self.use_robust = use_robust
        self.clip_percentiles = clip_percentiles
        
        if self.market_type not in MARKET_TYPES:
            logger.warning(f"Unknown market type: {market_type}, using 'us'")
            self.market_type = "us"
        
        self.vol_profile = VOLATILITY_PROFILES.get(self.market_type, VOLATILITY_PROFILES["us"])
        self._feature_scalers: Dict[str, object] = {}
        self._is_fitted = False
        self._feature_names: List[str] = []
        
    def fit(self, X: pd.DataFrame, y=None) -> "MarketNormalizedScaler":
        """
        Fit scaler on training data.
        
        Args:
            X: Feature DataFrame
            y: Ignored (for sklearn compatibility)
        """
        self._feature_names = list(X.columns)
        
        for col in X.columns:
            col_data = X[col].dropna()
            
            if self._should_use_robust(col):
                scaler = RobustScaler(
                    quantile_range=self.clip_percentiles,
                )
            else:
                scaler = StandardScaler()
            
            scaler.fit(col_data.values.reshape(-1, 1))
            self._feature_scalers[col] = scaler
        
        self._is_fitted = True
        logger.info(f"MarketNormalizedScaler fitted for market: {self.market_type}")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using market-aware scaling.
        """
        if not self._is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        X_scaled = X.copy()
        
        for col in X.columns:
            if col in self._feature_scalers:
                scaler = self._feature_scalers[col]
                
                if col in self._volatility_features():
                    X_scaled[col] = self._scale_volatility_feature(X[col], scaler)
                elif col in self._momentum_features():
                    X_scaled[col] = self._scale_momentum_feature(X[col], scaler)
                elif col in self._price_features():
                    X_scaled[col] = self._scale_price_feature(X[col], scaler)
                elif col in self._volume_features():
                    X_scaled[col] = self._scale_volume_feature(X[col], scaler)
                else:
                    X_scaled[col] = scaler.transform(X[[col]]).flatten()
        
        X_scaled = self._apply_regime_adjustments(X_scaled)
        
        return X_scaled
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
    
    def _should_use_robust(self, col: str) -> bool:
        """Determine if feature should use robust scaling."""
        return col in self._volatility_features() or col in self._momentum_features()
    
    def _volatility_features(self) -> List[str]:
        return [
            "atr", "atr_pct", "hist_vol_10", "hist_vol_30", "realized_vol_21d",
            "bb_width", "vol_ratio", "vol_zscore", "vol_regime",
        ]
    
    def _momentum_features(self) -> List[str]:
        return [
            "rsi_14", "rsi_9", "rsi_21", "momentum_20d", "momentum_60d",
            "roc_5", "roc_10", "roc_20", "macd", "macd_hist",
        ]
    
    def _price_features(self) -> List[str]:
        return [
            "close", "close_vs_sma_50", "close_vs_sma_200", "price_vs_ema9",
            "price_vs_ema21", "price_vs_ema50", "price_vs_ema200",
        ]
    
    def _volume_features(self) -> List[str]:
        return [
            "volume", "obv", "obv_slope", "obv_trend", "mfi", "chaikin_osc",
        ]
    
    def _scale_volatility_feature(self, series: pd.Series, scaler) -> pd.Series:
        """Scale volatility features with market-specific adjustments."""
        scaled = scaler.transform(series.values.reshape(-1, 1)).flatten()
        
        typical = self.vol_profile["typical_vol"]
        max_vol = self.vol_profile["max_vol"]
        
        if "pct" in series.name or "ratio" in series.name:
            scaled = np.clip(scaled, -3, 3)
        else:
            scaled = np.clip(scaled, -2.5, 2.5)
        
        return pd.Series(scaled, index=series.index)
    
    def _scale_momentum_feature(self, series: pd.Series, scaler) -> pd.Series:
        """Scale momentum features with market-specific adjustments."""
        scaled = scaler.transform(series.values.reshape(-1, 1)).flatten()
        
        threshold = self.vol_profile["momentum_threshold"]
        rsi_oversold = self.vol_profile["rsi_oversold"]
        rsi_overbought = self.vol_profile["rsi_overbought"]
        
        if "rsi" in series.name:
            rsi_scaled = (series - 50) / 50
            return pd.Series(np.clip(rsi_scaled, -1, 1), index=series.index)
        
        scaled = np.clip(scaled, -2, 2)
        
        return pd.Series(scaled, index=series.index)
    
    def _scale_price_feature(self, series: pd.Series, scaler) -> pd.Series:
        """Scale price features with market-specific adjustments."""
        scaled = scaler.transform(series.values.reshape(-1, 1)).flatten()
        scaled = np.clip(scaled, -3, 3)
        return pd.Series(scaled, index=series.index)
    
    def _scale_volume_feature(self, series: pd.Series, scaler) -> pd.Series:
        """Scale volume features with log transformation for crypto."""
        if self.market_type == "crypto":
            log_data = np.log1p(np.abs(series))
            scaled = scaler.transform(log_data.values.reshape(-1, 1)).flatten()
        else:
            scaled = scaler.transform(series.values.reshape(-1, 1)).flatten()
        
        scaled = np.clip(scaled, -3, 3)
        return pd.Series(scaled, index=series.index)
    
    def _apply_regime_adjustments(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply regime-aware feature adjustments."""
        X = X.copy()
        
        vol_col = "realized_vol_21d" if "realized_vol_21d" in X.columns else None
        if vol_col and vol_col in self._feature_scalers:
            vol_data = X[vol_col].copy()
            
            high_vol = vol_data > self.vol_profile["max_vol"] * 0.7
            low_vol = vol_data < self.vol_profile["typical_vol"] * 0.5
            
            for col in X.columns:
                if col in self._momentum_features():
                    X.loc[high_vol, col] = X.loc[high_vol, col] * 0.7
                    X.loc[low_vol, col] = X.loc[low_vol, col] * 1.2
        
        return X


def get_market_scaler(market_type: str) -> MarketNormalizedScaler:
    """Factory function to create market-specific scaler."""
    return MarketNormalizedScaler(market_type=market_type)


def scale_features_by_market(
    X: pd.DataFrame,
    market_types: Optional[pd.Series] = None,
    default_market: str = "us",
) -> pd.DataFrame:
    """
    Scale features differently based on market type.
    
    Args:
        X: Feature DataFrame
        market_types: Series mapping index -> market type
        default_market: Fallback market type
    
    Returns:
        Scaled DataFrame
    """
    if market_types is None:
        return MarketNormalizedScaler(market_type=default_market).fit_transform(X)
    
    result = X.copy()
    
    for market in market_types.unique():
        mask = market_types == market
        market_scaler = MarketNormalizedScaler(market_type=market)
        
        if market_scaler._is_fitted:
            result.loc[mask] = market_scaler.transform(X.loc[mask])
        else:
            result.loc[mask] = market_scaler.fit_transform(X.loc[mask])
    
    return result


class MarketAwareFeatureEngineer:
    """
    Creates market-aware derived features.
    
    For example:
    - Crypto: Higher momentum threshold (3% vs 1%)
    - India: Different RSI bounds (30/70)
    - Forex: Lower ATR multiplier (1.5 vs 2.0)
    """
    
    def __init__(self, market_type: str = "us"):
        self.market_type = market_type.lower()
        self.vol_profile = VOLATILITY_PROFILES.get(self.market_type, VOLATILITY_PROFILES["us"])
    
    def create_volatility_adjusted_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volatility-normalized features."""
        result = df.copy()
        
        if "close" in df.columns and "atr_pct" in df.columns:
            vol_ratio = df["atr_pct"] / self.vol_profile["typical_vol"]
            result["vol_adjusted_momentum"] = df.get("momentum_20d", 0) / vol_ratio
            result["vol_adjusted_momentum"] = result["vol_adjusted_momentum"].fillna(0)
        
        if "rsi_14" in df.columns:
            oversold = self.vol_profile["rsi_oversold"]
            overbought = self.vol_profile["rsi_overbought"]
            result["rsi_normalized"] = (df["rsi_14"] - oversold) / (overbought - oversold)
            result["rsi_normalized"] = result["rsi_normalized"].clip(0, 1)
        
        return result
    
    def create_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create regime-conditional features."""
        result = df.copy()
        
        if "realized_vol_21d" in df.columns:
            typical = self.vol_profile["typical_vol"]
            max_vol = self.vol_profile["max_vol"]
            
            result["regime_vol_ratio"] = df["realized_vol_21d"] / typical
            result["regime_stressed"] = (df["realized_vol_21d"] > max_vol * 0.7).astype(int)
            result["regime_calm"] = (df["realized_vol_21d"] < typical * 0.6).astype(int)
        
        return result
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all market-aware transformations."""
        result = self.create_volatility_adjusted_features(df)
        result = self.create_regime_features(result)
        return result


def get_market_aware_engineer(market_type: str) -> MarketAwareFeatureEngineer:
    """Factory function to create market-aware engineer."""
    return MarketAwareFeatureEngineer(market_type=market_type)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    test_data = pd.DataFrame({
        "rsi_14": [25, 35, 50, 65, 75, 85],
        "momentum_20d": [-0.05, -0.02, 0, 0.02, 0.05, 0.10],
        "volume": [1000000, 2000000, 1500000, 3000000, 2500000, 4000000],
        "realized_vol_21d": [0.01, 0.02, 0.03, 0.04, 0.05, 0.08],
    })
    
    print("Testing Market-Normalized Scaling...")
    print(f"\nOriginal data:\n{test_data}")
    
    for market in ["us", "india", "crypto"]:
        scaler = MarketNormalizedScaler(market_type=market)
        scaled = scaler.fit_transform(test_data)
        print(f"\n{market.upper()} scaled:\n{scaled}")
    
    print("\nMarket-Normalized Scaler tests complete.")