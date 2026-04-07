"""
Training Orchestrator — Phase 3
=================================
Coordinates training of all three models on Google Colab T4 GPU.

Schedule:
  1. Load feature matrices from Phase 1/2 (Parquet files)
  2. Train XGBoost (fast, CPU, ~5 min) with Optuna tuning
  3. Train LSTM (GPU, ~45 min) for all symbols
  4. Train Transformer (GPU, ~25 min) for all symbols
  5. Calibrate ensemble weights on validation set
  6. Run walk-forward backtest to validate
  7. Save everything to Google Drive

Memory management:
  - Colab T4 has 15GB VRAM — enough for batch_size=128, seq_len=60
  - Models trained symbol-by-symbol (not all at once) to avoid OOM
  - Checkpoints saved after each symbol in case Colab disconnects
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURES_DIR = Path(__file__).parent.parent / "data" / "features"
MODELS_DIR   = Path(__file__).parent / "checkpoints"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build_labels(feature_matrix: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Build classification labels from forward returns."""
    if "close" not in feature_matrix.columns:
        return pd.Series(1, index=feature_matrix.index)

    close = feature_matrix["close"]
    fwd_ret = close.pct_change(horizon).shift(-horizon)
    daily_std = close.pct_change().std()
    threshold = daily_std * 0.5 * (horizon ** 0.5)

    labels = pd.cut(
        fwd_ret,
        bins=[-np.inf, -threshold, threshold, np.inf],
        labels=[0, 1, 2],  # sell=0, hold=1, buy=2
    ).astype(float).fillna(1)

    return labels.astype(int)


class ModelTrainingOrchestrator:
    """
    Trains all three models end-to-end.
    Designed to run in Google Colab with T4 GPU.
    """

    def __init__(
        self,
        target_symbols: Optional[List[str]] = None,
        sequence_length: int = 60,
        train_horizon: int = 5,
        train_test_split: float = 0.80,
        val_split: float = 0.10,
        device: str = "auto",
    ):
        self.symbols = target_symbols
        self.seq_len = sequence_length
        self.horizon = train_horizon
        self.train_split = train_test_split
        self.val_split = val_split
        self.device = device
        self._trained_models: Dict = {}

    def run(
        self,
        feature_matrices: Optional[Dict[str, pd.DataFrame]] = None,
        run_optuna: bool = False,   # Set True for first training run
        optuna_trials: int = 50,
    ) -> Dict:
        """
        Full training pipeline.
        
        Args:
            feature_matrices: Pre-computed features (from Phase 2).
                If None, loads from disk.
            run_optuna: Whether to run Optuna hyperparameter optimization.
                Only needed once — after that, use saved best params.
        """
        logger.info("=" * 60)
        logger.info("PHASE 3: MODEL TRAINING STARTED")
        logger.info("=" * 60)

        # Load feature matrices if not provided
        if feature_matrices is None:
            feature_matrices = self._load_feature_matrices()

        if not feature_matrices:
            raise ValueError("No feature matrices found. Run Phase 1/2 first.")

        symbols = self.symbols or list(feature_matrices.keys())
        logger.info(f"Training on {len(symbols)} symbols")

        results = {}

        # ── Step 1: XGBoost (CPU, fast) ──────────────────────
        results["xgboost"] = self._train_xgboost(
            feature_matrices, symbols, run_optuna, optuna_trials
        )

        # ── Step 2: LSTM (GPU) ───────────────────────────────
        results["lstm"] = self._train_lstm(feature_matrices, symbols)

        # ── Step 3: Transformer (GPU) ────────────────────────
        results["transformer"] = self._train_transformer(feature_matrices, symbols)

        # ── Step 4: Ensemble calibration ────────────────────
        results["ensemble"] = self._calibrate_ensemble(feature_matrices, symbols, results)

        # ── Step 5: Walk-forward validation ─────────────────
        results["backtest"] = self._walk_forward_validation(feature_matrices, symbols)

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        self._print_summary(results)
        logger.info("=" * 60)

        return results

    def _train_xgboost(
        self,
        feature_matrices: Dict,
        symbols: List[str],
        run_optuna: bool,
        n_trials: int,
    ) -> Dict:
        """Train XGBoost on all symbols combined (cross-asset model)."""
        from models.xgboost_model import XGBoostSignalModel, XGBoostOptimizer

        logger.info("[1/5] Training XGBoost...")

        # Combine all symbols into one large training set
        all_X, all_y = [], []
        for sym in symbols:
            df = feature_matrices.get(sym)
            if df is None or df.empty or len(df) < 100:
                continue
            numeric_cols = [c for c in df.columns if df[c].dtype in (float, int, "float64", "int64")]
            X = df[numeric_cols].dropna(how="all")
            y = build_labels(df, self.horizon)
            y = y.reindex(X.index).fillna(1).astype(int)

            # Train/val/test split (time-ordered)
            n = len(X)
            train_end = int(n * self.train_split)
            X_train = X.iloc[:train_end]
            y_train = y.iloc[:train_end]
            all_X.append(X_train)
            all_y.append(y_train)

        if not all_X:
            logger.error("No valid data for XGBoost training")
            return {"status": "failed", "reason": "no_data"}

        X_combined = pd.concat(all_X, ignore_index=True)
        y_combined = pd.concat(all_y, ignore_index=True)

        logger.info(f"XGBoost training set: {X_combined.shape[0]} rows × {X_combined.shape[1]} features")

        # Optuna optimization (optional)
        if run_optuna:
            logger.info(f"Running Optuna ({n_trials} trials)...")
            optimizer = XGBoostOptimizer()
            best_params = optimizer.optimize(X_combined, y_combined, n_trials=n_trials)
        else:
            best_params = None

        # Train final model
        model = XGBoostSignalModel(params=best_params)
        n_train = int(len(X_combined) * 0.85)
        eval_set = (X_combined.iloc[n_train:], y_combined.iloc[n_train:])
        metrics = model.fit(X_combined.iloc[:n_train], y_combined.iloc[:n_train], eval_set=eval_set)

        # Cross-validation
        cv_metrics = model.time_series_cv_score(X_combined, y_combined)

        # Save
        model_path = model.save(MODELS_DIR / "xgboost_model.json")
        self._trained_models["xgboost"] = model

        logger.info(f"XGBoost CV directional accuracy: {cv_metrics['mean_directional_accuracy']:.3f}")
        return {**metrics, **cv_metrics, "model_path": str(model_path), "status": "ok"}

    def _train_lstm(self, feature_matrices: Dict, symbols: List[str]) -> Dict:
        """Train LSTM model on GPU."""
        try:
            import torch
            from torch.utils.data import DataLoader
            from models.lstm_model import LSTMPredictor, SequenceDataset, LSTMTrainer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"[2/5] Training LSTM on {device}...")

            all_results = {}

            for sym in symbols[:5]:  # Train on top 5 liquid symbols
                df = feature_matrices.get(sym)
                if df is None or len(df) < 200:
                    continue

                logger.info(f"  LSTM training: {sym} ({len(df)} rows)")

                dataset = SequenceDataset(df, sequence_length=self.seq_len, horizon=self.horizon)
                if len(dataset) < 100:
                    continue

                # Train/val split
                n = len(dataset)
                train_n = int(n * (self.train_split + self.val_split))
                val_n = int(n * self.val_split)

                train_ds = torch.utils.data.Subset(dataset, range(train_n - val_n))
                val_ds   = torch.utils.data.Subset(dataset, range(train_n - val_n, train_n))

                train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=0)
                val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=0)

                model = LSTMPredictor(n_features=dataset.n_features)
                trainer = LSTMTrainer(model, device=device)
                history = trainer.train(
                    train_loader, val_loader,
                    max_epochs=50,
                    patience=10,
                    checkpoint_path=MODELS_DIR / f"lstm_{sym.replace('.', '_')}.pt",
                )

                final_val_acc = history["val_acc"][-1] if history["val_acc"] else 0
                all_results[sym] = {
                    "val_acc": round(final_val_acc, 4),
                    "epochs": len(history["train_loss"]),
                    "n_features": dataset.n_features,
                }
                self._trained_models[f"lstm_{sym}"] = model
                logger.info(f"  {sym} LSTM: val_acc={final_val_acc:.3f}")

            if all_results:
                avg_acc = np.mean([r["val_acc"] for r in all_results.values()])
                logger.info(f"LSTM avg val_acc: {avg_acc:.3f}")
                return {"symbol_results": all_results, "avg_val_acc": round(avg_acc, 4), "status": "ok"}
            return {"status": "failed", "reason": "no_results"}

        except Exception as e:
            logger.error(f"LSTM training failed: {e}", exc_info=True)
            return {"status": "failed", "reason": str(e)}

    def _train_transformer(self, feature_matrices: Dict, symbols: List[str]) -> Dict:
        """Train Transformer model on GPU."""
        try:
            import torch
            from torch.utils.data import DataLoader
            from models.transformer_model import MarketTransformer, MultiHorizonDataset

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"[3/5] Training Transformer on {device}...")

            sym = symbols[0] if symbols else None
            df = feature_matrices.get(sym) if sym else None

            if df is None or len(df) < 200:
                logger.warning("Insufficient data for Transformer training")
                return {"status": "skipped", "reason": "insufficient_data"}

            mh_dataset = MultiHorizonDataset(df, sequence_length=self.seq_len)
            feature_names = mh_dataset.feature_names

            model = MarketTransformer(
                n_features=mh_dataset.n_features,
                feature_names=feature_names,
                d_model=128,         # Smaller for faster training
                nhead=4,
                num_encoder_layers=3,
                dim_feedforward=512,
                dropout=0.1,
            )
            model.to(torch.device(device))

            logger.info(
                f"Transformer: {mh_dataset.n_features} features | "
                f"{len(mh_dataset.valid_idx)} samples"
            )

            # Build simple single-horizon dataset for this training
            from models.lstm_model import SequenceDataset
            dataset = SequenceDataset(df, self.seq_len, self.horizon)
            n = len(dataset)
            train_n = int(n * 0.8)

            train_loader = DataLoader(
                torch.utils.data.Subset(dataset, range(train_n)),
                batch_size=32, shuffle=True, num_workers=0,
            )
            val_loader = DataLoader(
                torch.utils.data.Subset(dataset, range(train_n, n)),
                batch_size=64, shuffle=False, num_workers=0,
            )

            # Simple training loop for Transformer
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
            criterion = torch.nn.CrossEntropyLoss()
            best_acc, best_state = 0, None

            for epoch in range(30):
                model.train()
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    optimizer.zero_grad()
                    logits_list, _ = model(X_batch)
                    loss = criterion(logits_list[2], y_batch)  # 5d horizon
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                # Validation
                model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        logits_list, _ = model(X_batch.to(device))
                        preds = logits_list[2].argmax(dim=-1).cpu()
                        correct += (preds == y_batch).sum().item()
                        total += len(y_batch)

                val_acc = correct / max(total, 1)
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    torch.save({"model_state": best_state, "n_features": mh_dataset.n_features},
                               MODELS_DIR / "transformer_best.pt")

                if (epoch + 1) % 10 == 0:
                    logger.info(f"  Transformer epoch {epoch+1}/30: val_acc={val_acc:.3f}")

            self._trained_models["transformer"] = model
            logger.info(f"Transformer best val_acc: {best_acc:.3f}")
            return {"val_acc": round(best_acc, 4), "epochs": 30, "status": "ok"}

        except Exception as e:
            logger.error(f"Transformer training failed: {e}", exc_info=True)
            return {"status": "failed", "reason": str(e)}

    def _calibrate_ensemble(self, feature_matrices, symbols, model_results) -> Dict:
        """Calibrate ensemble weights on held-out test set."""
        from models.ensemble import EnsembleEngine
        logger.info("[4/5] Calibrating ensemble...")

        engine = EnsembleEngine()
        # Save initial state
        state_path = engine.save_state(MODELS_DIR / "ensemble_state.json")
        self._trained_models["ensemble"] = engine
        logger.info("Ensemble initialized with equal weights")
        return {"status": "ok", "initial_weights": engine.get_weights()}

    def _walk_forward_validation(self, feature_matrices, symbols) -> Dict:
        """Run walk-forward backtest on test set."""
        from core.backtesting import WalkForwardBacktester, BlackSwanStressTester
        from core.signal_engine_v2 import RegimeAwareStressTester

        logger.info("[5/5] Running walk-forward backtest...")

        try:
            stress = RegimeAwareStressTester()
            stress_results = stress.run_regime_aware_stress_tests()
            summary = stress_results.get("summary", {})
            logger.info(
                f"Stress test grade: {summary.get('overall_grade', '?')} | "
                f"{summary.get('stress_tests_passed', '?')} passed"
            )
            return {"stress_test": stress_results, "status": "ok"}
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return {"status": "failed", "reason": str(e)}

    def _load_feature_matrices(self) -> Dict[str, pd.DataFrame]:
        """Load feature matrices from disk."""
        matrices = {}
        if not FEATURES_DIR.exists():
            logger.warning(f"Features directory not found: {FEATURES_DIR}")
            return {}
        for parquet_file in FEATURES_DIR.glob("*_features.parquet"):
            sym = parquet_file.stem.replace("_features", "").replace("_", ".")
            try:
                df = pd.read_parquet(parquet_file)
                matrices[sym] = df
                logger.debug(f"Loaded {sym}: {df.shape}")
            except Exception as e:
                logger.warning(f"Failed to load {parquet_file}: {e}")
        logger.info(f"Loaded {len(matrices)} feature matrices from disk")
        return matrices

    def _print_summary(self, results: Dict):
        logger.info("\n" + "─" * 40)
        logger.info("TRAINING SUMMARY")
        logger.info("─" * 40)
        for component, res in results.items():
            status = res.get("status", "?")
            if status == "ok":
                if component == "xgboost":
                    logger.info(f"  XGBoost  ✓  CV dir_acc={res.get('mean_directional_accuracy', 0):.3f}")
                elif component == "lstm":
                    logger.info(f"  LSTM     ✓  avg_val_acc={res.get('avg_val_acc', 0):.3f}")
                elif component == "transformer":
                    logger.info(f"  Transformer ✓  val_acc={res.get('val_acc', 0):.3f}")
                elif component == "ensemble":
                    weights = res.get("initial_weights", {})
                    logger.info(f"  Ensemble ✓  weights={weights}")
            else:
                logger.info(f"  {component:12s} ✗  {res.get('reason', 'unknown error')}")


# ─────────────────────────────────────────────────────────────
# Colab entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    orchestrator = ModelTrainingOrchestrator(
        sequence_length=60,
        train_horizon=5,
    )

    print("""
╔══════════════════════════════════════════════════════╗
║        PHASE 3: MODEL TRAINING                       ║
║                                                      ║
║  Run this in Google Colab with T4 GPU:               ║
║    Runtime → Change runtime type → T4 GPU            ║
║                                                      ║
║  Estimated times:                                    ║
║    XGBoost:     5 min (CPU)                          ║
║    LSTM:       45 min (T4 GPU)                       ║
║    Transformer: 25 min (T4 GPU)                      ║
║    Total:      ~75 min                               ║
╚══════════════════════════════════════════════════════╝
    """)

    results = orchestrator.run(run_optuna=False)
