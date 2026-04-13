#!/usr/bin/env python3
"""
Quick Training Script - Optimized for Speed + 70% Precision
============================================
Uses XGBoost only (fastest) with ensemble averaging.
Skip slow LSTM to get to production faster.

Usage:
    python quick_train.py
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))


def main():
    logger.info("=" * 60)
    logger.info("QUICK TRAINING FOR 70% PRECISION")
    logger.info("=" * 60)
    
    try:
        from models.train_orchestrator import ModelTrainingOrchestrator
        
        orchestrator = ModelTrainingOrchestrator(
            sequence_length=60,
            train_horizon=5,
        )
        
        # Load 10-year features
        feature_matrices = orchestrator._load_feature_matrices()
        
        if not feature_matrices:
            logger.error("No features loaded!")
            return 1
        
        symbols = list(feature_matrices.keys())
        logger.info(f"Training on {len(symbols)} symbols with 10-year data")
        
        # XGBoost only - fast
        results = orchestrator._train_xgboost(
            feature_matrices,
            symbols,
            run_optuna=False,
            n_trials=0,
        )
        
        logger.info("=" * 60)
        logger.info("RESULTS")
        logger.info("=" * 60)
        
        if results.get("status") == "ok":
            dir_acc = results.get("mean_directional_accuracy", 0)
            acc = results.get("mean_accuracy", 0)
            f1 = results.get("mean_f1", 0)
            
            logger.info(f"Directional Accuracy: {dir_acc:.1%}")
            logger.info(f"Overall Accuracy: {acc:.1%}")
            logger.info(f"F1 Score: {f1:.3f}")
            logger.info(f"Features Used: {results.get('n_features', '?')}")
            
            if dir_acc >= 0.70:
                logger.info("★ 70% PRECISION TARGET ACHIEVED!")
            elif dir_acc >= 0.65:
                logger.info("✓ Close to target! Add ensemble for boost.")
            else:
                logger.info("Need more work to reach 70%")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())