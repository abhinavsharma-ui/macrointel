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
FEATURE_DIR_10YR = Path(__file__).parent / "data" / "features_10yr"
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


def _feature_store_sources() -> list[Path]:
    explicit = str(os.getenv("XGB_RETRAIN_FEATURE_DIR", "")).strip()
    if explicit:
        return [Path(explicit)]
    sources: list[Path] = []
    if FEATURE_DIR_10YR.exists():
        sources.append(FEATURE_DIR_10YR)
    if FEATURE_DIR.exists():
        sources.append(FEATURE_DIR)
    return sources


def load_feature_store() -> tuple[dict, dict]:
    feature_matrices = {}
    source_stats: dict[str, dict[str, int]] = {}

    if FEATURE_DIR_10YR in _feature_store_sources():
        file_count = 0
        row_count = 0
        for path in sorted(FEATURE_DIR_10YR.glob("*USDT*.parquet")):
            try:
                df = pd.read_parquet(path)
            except Exception as exc:
                print(f"skip {path.name}: {exc}")
                continue
            file_count += 1
            row_count += len(df)
            feature_matrices.setdefault(path.stem, df)
        source_stats[str(FEATURE_DIR_10YR)] = {
            "files": file_count,
            "rows": row_count,
        }

    if FEATURE_DIR in _feature_store_sources():
        file_count = 0
        row_count = 0
        for path in sorted(FEATURE_DIR.glob("*.parquet")):
            try:
                df = pd.read_parquet(path)
            except Exception as exc:
                print(f"skip {path.name}: {exc}")
                continue
            file_count += 1
            row_count += len(df)
            feature_matrices.setdefault(path.stem, df)
        source_stats[str(FEATURE_DIR)] = {
            "files": file_count,
            "rows": row_count,
        }
    return feature_matrices, source_stats


def main():
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger = logging.getLogger("retrain")
    market_scope = os.getenv("INSTITUTIONAL_RETRAIN_MARKET", "all").strip().lower() or "all"
    logger.info(
        "Retrain env: INSTITUTIONAL_RETRAIN_MARKET=%s SKIP_XGB_RETRAIN=%s META_MODEL_BUILD_WORKERS=%s XGB_RETRAIN_FEATURE_DIR=%s",
        market_scope,
        os.getenv("SKIP_XGB_RETRAIN", "0"),
        os.getenv("META_MODEL_BUILD_WORKERS", ""),
        os.getenv("XGB_RETRAIN_FEATURE_DIR", ""),
    )
    sources = _feature_store_sources()
    logger.info("Loading retrain feature store from %s", ", ".join(str(path) for path in sources) or "<none>")
    feature_matrices, source_stats = load_feature_store()
    if not feature_matrices:
        raise SystemExit("No parquet feature matrices found in retrain feature stores")

    for source, stats in source_stats.items():
        logger.info(
            "Feature source %s: %s parquet files, %s rows",
            source,
            stats.get("files", 0),
            stats.get("rows", 0),
        )

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
        market_scope=market_scope,
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
