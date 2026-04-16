"""
Feature Bridge
==============
Maps the live feature pipeline output to exactly what the trained
XGBoost model expects. Handles name mismatches between training and inference.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from models.xgboost_model import add_regime_interactions

logger = logging.getLogger(__name__)

CHECKPOINTS_DIR = Path(__file__).parent.parent / "models" / "checkpoints"

COLUMN_MAP = {
    "macd_diff":     "macd_hist",
    "stoch":         "stoch_k",
    "returns_1d":    "roc_5",
    "returns_5d":    "roc_10",
    "returns_20d":   "roc_20",
    "volatility_20": "hist_vol_30",
    "bb_upper":      None,
    "bb_lower":      None,
    "sma_20":        None,
    "sma_50":        None,
    "ema_12":        None,
    "ema_26":        None,
    "vwap":          None,
    "open":          None,
    "high":          None,
    "low":           None,
    "close":         None,
    "volume":        None,
}


class FeatureBridge:
    def __init__(self):
        self._feature_cols = None
        self._scaler = None
        self._load()

    def _load(self):
        try:
            cols_path = CHECKPOINTS_DIR / "feature_cols.pkl"
            scaler_path = CHECKPOINTS_DIR / "scaler.pkl"
            if cols_path.exists():
                with open(cols_path, "rb") as f:
                    self._feature_cols = pickle.load(f)
                logger.info(f"Feature bridge: {len(self._feature_cols)} features loaded")
            if scaler_path.exists():
                with open(scaler_path, "rb") as f:
                    self._scaler = pickle.load(f)
                logger.info("Feature bridge: scaler loaded")
        except Exception as e:
            logger.error(f"Feature bridge load error: {e}")

    def prepare_latest(self, features: pd.DataFrame, ohlcv: Optional[pd.DataFrame] = None) -> Optional[np.ndarray]:
        if self._feature_cols is None:
            return None

        row = features.iloc[-1].copy() if len(features) > 0 else None
        if row is None:
            return None

        base = {}
        for col in row.index:
            value = row[col]
            try:
                base[col] = float(value) if pd.notna(value) else 0.0
            except Exception:
                continue

        if "price_above_sma_200" not in base and "close_vs_sma_200" in base:
            base["price_above_sma_200"] = 1.0 if base["close_vs_sma_200"] > 0 else 0.0

        needed_raw = set()
        for col in self._feature_cols:
            needed_raw.add(col)
            mapped = COLUMN_MAP.get(col)
            if mapped:
                needed_raw.add(mapped)

        if ohlcv is not None and not ohlcv.empty:
            for col in needed_raw:
                if col in base:
                    continue
                computed = self._compute_inline(col, ohlcv)
                if computed is not None:
                    base[col] = computed

        prepared = pd.DataFrame([base]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        prepared = add_regime_interactions(prepared)

        result = {}
        for col in self._feature_cols:
            if col in prepared.columns and pd.notna(prepared.iloc[0][col]):
                result[col] = float(prepared.iloc[0][col])
                continue
            mapped = COLUMN_MAP.get(col)
            if mapped and mapped in prepared.columns and pd.notna(prepared.iloc[0][mapped]):
                result[col] = float(prepared.iloc[0][mapped])
                continue
            result[col] = 0.0

        X = np.array([[result[c] for c in self._feature_cols]], dtype=float)
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        if self._scaler is not None:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    X = self._scaler.transform(X)
            except Exception as e:
                logger.warning(f"Scaler transform failed: {e}")

        return X

    def _compute_inline(self, col: str, ohlcv: pd.DataFrame) -> Optional[float]:
        try:
            close = ohlcv["close"]
            high = ohlcv["high"]
            low = ohlcv["low"]
            volume = ohlcv["volume"]

            if col == "open":   return float(ohlcv["open"].iloc[-1])
            if col == "high":   return float(high.iloc[-1])
            if col == "low":    return float(low.iloc[-1])
            if col == "close":  return float(close.iloc[-1])
            if col == "volume": return float(volume.iloc[-1])
            if col == "sma_20": return float(close.rolling(20).mean().iloc[-1])
            if col == "sma_50": return float(close.rolling(50).mean().iloc[-1])
            if col == "ema_12": return float(close.ewm(span=12, adjust=False).mean().iloc[-1])
            if col == "ema_26": return float(close.ewm(span=26, adjust=False).mean().iloc[-1])
            if col == "bb_upper":
                mid = close.rolling(20).mean()
                std = close.rolling(20).std()
                return float((mid + 2 * std).iloc[-1])
            if col == "bb_lower":
                mid = close.rolling(20).mean()
                std = close.rolling(20).std()
                return float((mid - 2 * std).iloc[-1])
            if col == "vwap":
                tp = (high + low + close) / 3
                return float((tp * volume).sum() / volume.replace(0, np.nan).sum())
        except Exception:
            pass
        return None


def diagnose_feature_mismatch():
    bridge = FeatureBridge()
    if bridge._feature_cols is None:
        print("ERROR: feature_cols.pkl not found in models/checkpoints/")
        return

    print(f"Model expects {len(bridge._feature_cols)} features:")
    print(bridge._feature_cols)

    from pipeline.price_collector import PriceDataPipeline
    from pipeline.feature_engineering import FeaturePipeline

    price_data = PriceDataPipeline().run_incremental_update()
    features = FeaturePipeline().build_feature_matrix(
        price_data=price_data.get("price_daily_recent", {}),
        sentiment_data=None
    )

    symbol = "AAPL"
    if symbol in features:
        row = features[symbol].iloc[-1]
        ohlcv = price_data.get("price_daily_recent", {}).get(symbol)
        matched = sum(1 for c in bridge._feature_cols if c in row.index and pd.notna(row[c]))
        mapped = sum(1 for c in bridge._feature_cols
                    if c not in row.index and COLUMN_MAP.get(c) and COLUMN_MAP.get(c) in row.index)
        inline = sum(1 for c in bridge._feature_cols
                    if c not in row.index and not COLUMN_MAP.get(c)
                    and ohlcv is not None and bridge._compute_inline(c, ohlcv) is not None)
        total = len(bridge._feature_cols)
        print(f"\nDirect match: {matched}/{total}")
        print(f"Mapped match: {mapped}/{total}")
        print(f"Inline computed: {inline}/{total}")
        print(f"Coverage: {(matched+mapped+inline)/total*100:.0f}%")
        missing = [c for c in bridge._feature_cols
                  if c not in row.index
                  and not (COLUMN_MAP.get(c) and COLUMN_MAP.get(c) in row.index)
                  and (ohlcv is None or bridge._compute_inline(c, ohlcv) is None)]
        if missing:
            print(f"Still missing (will use 0): {missing}")
        else:
            print("All features covered!")
