"""
Guarded retraining entrypoint for the institutional model stack.

Usage:
    python retrain_institutional_models.py

Meta model quality scales with **years of daily rows** in data/features/*.parquet.
Tune via env (see .env.example): META_MODEL_HORIZON_DAYS, META_MODEL_WALKFORWARD_FOLDS,
META_MODEL_MIN_TRAIN_DAYS, META_MODEL_MIN_FRAME_ROWS, META_MODEL_WARMUP_ROWS,
META_MODEL_RULE_OBJECTIVE=precision|legacy, META_MODEL_THRESHOLD_GRID, etc.
Walk-forward metrics near ~70% precision/hit are **not guaranteed** (markets are noisy);
the trainer now biases toward **precision on take_label** when RULE_OBJECTIVE=precision.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from core.performance import apply_performance_profile

apply_performance_profile()

import pandas as pd

from models.institutional_retraining import InstitutionalTrainingPipeline

FEATURE_DIR = Path(__file__).parent / "data" / "features"
REQUIRED_FEATURE_COLS = {
    "earnings_propagation_signal",
    "close_reversal_signal",
    "event_alpha_signal",
    "alpha_signal",
    "official_event_signal",
    "filing_event_signal",
    "news_volume_spike",
    "weighted_sentiment_zscore",
    "travel_activity_change",
}


def load_feature_store() -> dict:
    feature_matrices = {}
    for path in sorted(FEATURE_DIR.glob("*.parquet")):
        try:
            feature_matrices[path.stem] = pd.read_parquet(path)
        except Exception as exc:
            print(f"skip {path.name}: {exc}")
    return feature_matrices


def main():
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger = logging.getLogger("retrain")
    logger.info("Loading feature store from %s", FEATURE_DIR)
    feature_matrices = load_feature_store()
    if not feature_matrices:
        raise SystemExit("No parquet feature matrices found in data/features")

    coverage = sum(1 for df in feature_matrices.values() if REQUIRED_FEATURE_COLS.issubset(df.columns))
    logger.info("Loaded %s feature matrices, %s with upgraded event/news columns", len(feature_matrices), coverage)
    if coverage == 0:
        raise SystemExit(
            "Aborting retrain: local feature store does not contain the upgraded event/news columns yet. "
            "Rebuild feature history with the latest pipelines first."
        )

    trainer = InstitutionalTrainingPipeline(
        xgb_horizon=max(1, int(os.getenv("XGB_RETRAIN_HORIZON_DAYS", "5"))),
        meta_horizon=max(1, int(os.getenv("META_MODEL_HORIZON_DAYS", "8"))),
        meta_take_threshold=float(os.getenv("META_MODEL_TAKE_THRESHOLD", "0.52")),
        meta_walk_forward_folds=max(3, int(os.getenv("META_MODEL_WALKFORWARD_FOLDS", "6"))),
        meta_min_train_days=max(90, int(os.getenv("META_MODEL_MIN_TRAIN_DAYS", "200"))),
    )
    logger.info("Starting institutional retraining pipeline")
    report = trainer.train_all(
        feature_matrices=feature_matrices,
        run_optuna=os.getenv("RUN_XGB_OPTUNA", "0").strip().lower() in {"1", "true", "yes", "on"},
    )
    logger.info("Retraining complete")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
