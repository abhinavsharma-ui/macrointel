"""
Stacking Ensemble Training Script
===================================
Trains LightGBM + CatBoost + XGBoost stacking ensemble
on existing feature data.

Usage:
    python scripts/train_stacking.py
"""

import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

project_root = Path("C:/Users/Sidhnath/Downloads/macro_intelligence_complete/project")
sys.path.insert(0, str(project_root))

from models.stacking_ensemble import StackingEnsemble, train_stacking_ensemble
from models.market_scaler import get_market_scaler, get_market_aware_engineer
from pipeline.finnhub_client import get_finnhub_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

FEATURES_DIR = project_root / "data" / "features"
MODEL_OUTPUT_DIR = project_root / "models" / "checkpoints"
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "rsi_14", "rsi_9", "rsi_21", "macd", "macd_hist",
    "roc_5", "roc_10", "roc_20", "momentum_20d", "momentum_60d",
    "williams_r", "stoch_k", "stoch_d",
    "price_vs_ema9", "price_vs_ema21", "price_vs_ema50", "price_vs_ema200",
    "close_vs_sma_50", "close_vs_sma_200", "golden_cross",
    "atr_pct", "bb_width", "bb_pct",
    "hist_vol_10", "hist_vol_30", "realized_vol_21d", "vol_regime",
    "vol_ratio", "vol_zscore", "zscore_vs_60d",
    "obv", "obv_slope", "obv_trend", "mfi", "chaikin_osc",
    "di_plus", "di_minus", "adx_14",
    "compound_score", "sentiment_zscore", "sentiment_velocity",
    "earnings_propagation_signal", "close_reversal_signal",
]


def load_feature_data(symbols: Optional[List[str]] = None, max_symbols: int = 50) -> pd.DataFrame:
    """Load and combine feature data from parquet files."""
    logger.info(f"Loading feature data from {FEATURES_DIR}")
    
    parquet_files = list(FEATURES_DIR.glob("*.parquet"))
    
    if symbols:
        parquet_files = [f for f in parquet_files if f.stem in symbols]
    else:
        parquet_files = parquet_files[:max_symbols]
    
    logger.info(f"Found {len(parquet_files)} parquet files")
    
    all_data = []
    for f in parquet_files:
        try:
            df = pd.read_parquet(f)
            if len(df) >= 60:
                df["symbol"] = f.stem
                all_data.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {f.name}: {e}")
    
    if not all_data:
        logger.error("No data loaded!")
        return pd.DataFrame()
    
    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"Loaded {len(combined)} rows, {len(combined['symbol'].unique())} symbols")
    
    return combined


def prepare_training_data(df: pd.DataFrame) -> tuple:
    """Prepare features and labels for training."""
    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    logger.info(f"Using {len(available_cols)} features: {available_cols[:10]}...")
    
    df = df.dropna(subset=["close"])
    
    if len(df) < 100:
        raise ValueError("Insufficient data for training")
    
    X = df[available_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    close = df["close"]
    fwd_ret = close.pct_change(5).shift(-5)
    daily_std = close.pct_change().std()
    
    valid_returns = fwd_ret.dropna()
    threshold_lower = valid_returns.quantile(0.33)
    threshold_upper = valid_returns.quantile(0.67)
    
    labels = pd.cut(
        fwd_ret,
        bins=[-np.inf, threshold_lower, threshold_upper, np.inf],
        labels=[0, 1, 2],
    ).astype(float).fillna(1).astype(int)
    
    labels = labels.fillna(1).astype(int)
    
    valid_mask = labels.index.isin(X.index)
    X_orig = X[valid_mask].reset_index(drop=True)
    labels_orig = labels[valid_mask].reset_index(drop=True)
    
    class_counts = labels_orig.value_counts().sort_index()
    logger.info(f"Original class distribution: {class_counts.to_dict()}")
    
    min_class_size = min(class_counts.min(), 50)
    target_per_class = max(30, min_class_size)
    
    balanced_indices = []
    for cls in [0, 1, 2]:
        cls_indices = labels_orig[labels_orig == cls].index.tolist()
        n_to_select = min(target_per_class, len(cls_indices))
        if n_to_select > 0:
            selected = np.random.choice(cls_indices, n_to_select, replace=False).tolist()
            balanced_indices.extend(selected)
    
    balanced_indices = sorted(balanced_indices)
    X = X_orig.iloc[balanced_indices].reset_index(drop=True)
    labels = labels_orig.iloc[balanced_indices].reset_index(drop=True)
    
    logger.info(f"Balanced class distribution: {labels.value_counts().sort_index().to_dict()}")
    logger.info(f"Training data: {len(X)} samples, {X.shape[1]} features")
    
    return X, labels


def run_training(n_folds: int = 5, max_symbols: int = 100):
    """Run the full training pipeline."""
    logger.info("=" * 60)
    logger.info("STACKING ENSEMBLE TRAINING")
    logger.info("=" * 60)
    
    df = load_feature_data(max_symbols=max_symbols)
    
    if df.empty:
        logger.error("No data loaded. Exiting.")
        return None
    
    X, y = prepare_training_data(df)
    
    logger.info(f"Training data: {len(X)} samples, {X.shape[1]} features")
    logger.info(f"Label distribution: {dict(y.value_counts())}")
    
    ensemble = StackingEnsemble(n_folds=n_folds, agreement_threshold=0.66)
    
    logger.info("Training stacking ensemble...")
    results = ensemble.fit(X, y)
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING RESULTS")
    logger.info("=" * 60)
    
    for model_name, score in results.get("cv_scores", {}).items():
        logger.info(f"  {model_name}: {score:.4f}")
    
    agreement = results.get("agreement_stats", {})
    logger.info(f"\nAgreement Stats:")
    logger.info(f"  Mean Agreement: {agreement.get('mean_agreement', 0):.4f}")
    logger.info(f"  Full Agreement: {agreement.get('full_agreement_pct', 0):.2f}%")
    logger.info(f"  2/3 Agreement: {agreement.get('two_of_three_pct', 0):.2f}%")
    
    return ensemble, results


if __name__ == "__main__":
    logger.info("Starting stacking ensemble training...")
    
    ensemble, results = run_training(n_folds=5, max_symbols=50)
    
    if ensemble:
        logger.info("\n✅ Training complete!")
    else:
        logger.error("\n❌ Training failed!")
        sys.exit(1)