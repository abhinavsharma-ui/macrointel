#!/usr/bin/env python3
"""
Quick Training Script for 70% Precision Target
==========================================
Run this to train all models targeting 70% directional precision.

Usage:
    python run_training.py                    # Full training
    python run_training.py --xgboost-only     # XGBoost only (faster)
    python run_training.py --optuna            # With hyperparameter tuning
"""

import argparse
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
    parser = argparse.ArgumentParser(description="Train models for 70% precision")
    parser.add_argument("--xgboost-only", action="store_true", help="Train only XGBoost")
    parser.add_argument("--optuna", action="store_true", help="Run Optuna hyperparameter tuning")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to train on")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("TRAINING FOR 70% PRECISION TARGET")
    logger.info("=" * 60)
    
    try:
        from models.train_orchestrator import ModelTrainingOrchestrator
        
        orchestrator = ModelTrainingOrchestrator(
            sequence_length=60,
            train_horizon=5,
        )
        
        if args.xgboost_only:
            logger.info("Running XGBoost-only training...")
            feature_matrices = orchestrator._load_feature_matrices()
            symbols = args.symbols or list(feature_matrices.keys())
            
            results = orchestrator._train_xgboost(
                feature_matrices, 
                symbols[:10],  # Top 10 symbols
                run_optuna=args.optuna,
                n_trials=args.trials,
            )
            
            logger.info(f"XGBoost results: {results}")
            
            # Check if likely to hit target
            if results.get("target_precision_likely"):
                logger.info("✓ Model likely to achieve 70% precision target")
            else:
                logger.warning("⚠ Model may need more training/data for 70% target")
        else:
            logger.info("Running full training pipeline...")
            results = orchestrator.run(
                run_optuna=args.optuna,
                optuna_trials=args.trials,
            )
            
            # Summary
            logger.info("=" * 60)
            logger.info("TRAINING COMPLETE")
            logger.info("=" * 60)
            
            for component, res in results.items():
                status = res.get("status", "?")
                logger.info(f"  {component}: {status}")
            
            # Precision readiness
            if "backtest" in results:
                pf = results["backtest"].get("precision_assessment", {})
                logger.info(f"  Ready for live trading: {pf.get('ready_for_live', False)}")
                logger.info(f"  Target precision: {pf.get('target_precision', 'N/A')}")
            
        logger.info("Training complete!")
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())