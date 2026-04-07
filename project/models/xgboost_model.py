"""
XGBoost Short-Term Signal Model — Phase 3
==========================================
XGBoost handles what LSTMs and Transformers struggle with:
  - Non-linear interactions between tabular features
  - Short-term (1-3 day) signals driven by discrete events
  - Feature importance natively (no SHAP needed — it's built in)
  - Handles missing values without imputation
  - Trains in seconds, not minutes

Optimal for:
  - Options flow signals (unusual_options × RSI × momentum)
  - Congress trading signals (congress_signal × sector_momentum)
  - Earnings surprise signals (surprise_pct × historical_reaction)
  - Regime-conditional signals (same RSI value means different things
    in calm vs stressed markets)

Hyperparameter optimization: Optuna with 100 trials, pruning via
Hyperband. Finds the best configuration in ~15 min.
"""

import logging
import json
import inspect
import os
import pickle
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

from models.calibration import MultiClassPlattCalibrator

try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False
    logging.warning("XGBoost not installed. Run: pip install xgboost")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_OK = True
except ImportError:
    OPTUNA_OK = False

logger = logging.getLogger(__name__)
MODELS_DIR = Path(__file__).parent / "checkpoints"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATION_SUFFIX = ".calibration.pkl"


def _default_n_jobs() -> int:
    raw = (os.getenv("XGB_N_JOBS") or os.getenv("MAX_CPU_THREADS") or "").strip()
    if raw:
        try:
            parsed = int(float(raw))
        except Exception:
            parsed = 0
        if parsed > 0:
            return parsed
    cpu_count = os.cpu_count() or 2
    return max(1, min(2, cpu_count))


DEFAULT_N_JOBS = _default_n_jobs()


# ─────────────────────────────────────────────────────────────
# Feature selector — XGBoost-specific
# ─────────────────────────────────────────────────────────────
class FeatureSelector:
    """
    Selects the best features for XGBoost specifically.
    
    XGBoost works best with:
    - Low-correlated features (high correlation = redundant splits)
    - Features with predictive power (non-zero importance after initial fit)
    - Regime interaction features (e.g. RSI × vol_regime)
    
    This runs a quick initial XGB fit and drops features with zero importance.
    """

    def __init__(self, correlation_threshold: float = 0.92):
        self.corr_threshold = correlation_threshold
        self.selected_features: List[str] = []

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Select features using importance + correlation filtering."""
        if not XGB_OK:
            self.selected_features = list(X.columns)
            return X

        # Step 1: Drop near-duplicate features (high correlation)
        X_clean = self._drop_correlated(X)

        # Step 2: Quick XGB fit to get initial importances
        quick_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42,
            eval_metric="mlogloss",
            n_jobs=DEFAULT_N_JOBS,
        )
        quick_model.fit(X_clean.fillna(0), y, verbose=False)
        importances = quick_model.feature_importances_

        # Step 3: Keep features with above-median importance
        median_imp = np.median(importances[importances > 0]) if any(importances > 0) else 0
        keep_mask = importances > max(median_imp * 0.1, 1e-6)
        self.selected_features = [col for col, keep in zip(X_clean.columns, keep_mask) if keep]

        logger.info(
            f"Feature selection: {X.shape[1]} → {len(self.selected_features)} "
            f"(removed {X.shape[1] - len(self.selected_features)} low-importance)"
        )
        return X_clean[self.selected_features]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        available = [f for f in self.selected_features if f in X.columns]
        missing = [f for f in self.selected_features if f not in X.columns]
        if missing:
            logger.warning(f"Missing {len(missing)} features at inference time")
        result = X[available].copy()
        for f in missing:
            result[f] = 0  # Fill missing with 0
        return result[self.selected_features]

    def _drop_correlated(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove one of each pair of highly correlated features."""
        corr_matrix = X.fillna(0).corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [col for col in upper.columns if any(upper[col] > self.corr_threshold)]
        return X.drop(columns=to_drop, errors="ignore")


# ─────────────────────────────────────────────────────────────
# Regime Interaction Features
# ─────────────────────────────────────────────────────────────
def add_regime_interactions(X: pd.DataFrame) -> pd.DataFrame:
    """
    XGBoost can't learn interactions across tree boundaries efficiently.
    Help it by adding explicit interaction terms for key regime combinations.
    
    These features encode domain knowledge:
    "RSI=30 in a calm market = buy, but RSI=30 in a crisis = don't touch"
    """
    X = X.copy()

    # RSI × regime
    if "rsi_14" in X.columns and "vol_regime_stressed" in X.columns:
        X["rsi_x_stressed"] = X["rsi_14"] * X["vol_regime_stressed"]
        X["rsi_x_calm"] = X["rsi_14"] * (1 - X["vol_regime_stressed"])

    # Momentum × trend alignment
    if "momentum_20d" in X.columns and "price_above_sma_200" in X.columns:
        X["momentum_with_trend"] = X["momentum_20d"] * X["price_above_sma_200"]
        X["momentum_against_trend"] = X["momentum_20d"] * (1 - X["price_above_sma_200"])

    # Options flow × momentum
    if "options_sentiment" in X.columns and "momentum_20d" in X.columns:
        X["options_x_momentum"] = X["options_sentiment"] * X["momentum_20d"].clip(-0.2, 0.2)

    # Congress signal × sector momentum
    if "congress_signal" in X.columns and "momentum_60d" in X.columns:
        X["congress_x_momentum"] = X["congress_signal"] * np.sign(X["momentum_60d"])

    # Sentiment velocity × price momentum (both accelerating = strong signal)
    if "sentiment_velocity" in X.columns and "price_acceleration" in X.columns:
        X["sentiment_price_accel"] = X["sentiment_velocity"] * X["price_acceleration"]

    # VIX × credit spreads (double stress = very bearish)
    if "macro_vix_level" in X.columns and "macro_credit_spread_hy" in X.columns:
        X["vix_x_credit_stress"] = (
            X["macro_vix_level"].clip(0, 80) / 80 *
            X["macro_credit_spread_hy"].clip(0, 20) / 20
        )

    return X.fillna(0)


def _fit_xgb_classifier(model, X, y, eval_set=None, early_stopping_rounds: Optional[int] = None, verbose=False):
    """
    XGBoost's sklearn API changed in v3 and no longer accepts
    early_stopping_rounds in fit(). This helper supports both APIs.
    """
    fit_kwargs = {
        "eval_set": eval_set,
        "verbose": verbose,
    }
    fit_sig = inspect.signature(model.fit)
    if "early_stopping_rounds" in fit_sig.parameters:
        fit_kwargs["early_stopping_rounds"] = early_stopping_rounds if eval_set else None
        model.fit(X, y, **fit_kwargs)
        return model

    if eval_set and early_stopping_rounds:
        try:
            model.set_params(early_stopping_rounds=early_stopping_rounds)
        except Exception:
            pass

    model.fit(X, y, **fit_kwargs)
    return model


# ─────────────────────────────────────────────────────────────
# XGBoost Model
# ─────────────────────────────────────────────────────────────
class XGBoostSignalModel:
    """
    XGBoost classifier for short-term (1-3 day) signal generation.
    
    Uses TimeSeriesSplit for cross-validation (never uses future data).
    Hyperparameters optimized via Optuna.
    """

    DEFAULT_PARAMS = {
        "n_estimators":        800,
        "max_depth":           5,
        "learning_rate":       0.02,
        "subsample":           0.8,
        "colsample_bytree":    0.7,
        "colsample_bylevel":   0.7,
        "reg_alpha":           0.1,     # L1 — sparsity
        "reg_lambda":          1.0,     # L2 — smoothness
        "min_child_weight":    10,      # Prevents overfitting on rare signals
        "gamma":               0.1,
        "max_delta_step":      1,
        "tree_method":         "hist",  # Fast histogram method
        "eval_metric":         "mlogloss",
        "random_state":        42,
        "n_jobs":              DEFAULT_N_JOBS,
    }

    def __init__(self, params: Optional[Dict] = None):
        if not XGB_OK:
            raise RuntimeError("Install xgboost: pip install xgboost")
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model: Optional[xgb.XGBClassifier] = None
        self.selector = FeatureSelector()
        self.feature_names_in: List[str] = []
        self.prob_calibrator = MultiClassPlattCalibrator()
        self._is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple] = None,
        early_stopping_rounds: int = 50,
        verbose: int = 0,
    ) -> Dict:
        """
        Train the model.
        
        y must be in {0=sell, 1=hold, 2=buy} encoding.
        Uses TimeSeriesSplit for final evaluation metrics.
        """
        # Add interaction features
        X_aug = add_regime_interactions(X)

        # Feature selection
        X_sel = self.selector.fit_transform(X_aug, y)
        self.feature_names_in = list(X_sel.columns)

        # Build model
        self.model = xgb.XGBClassifier(**self.params)

        # Prepare eval set
        if eval_set:
            X_val, y_val = eval_set
            X_val_aug = add_regime_interactions(X_val)
            X_val_sel = self.selector.transform(X_val_aug)
            fit_eval = [(X_val_sel.fillna(0), y_val)]
        else:
            fit_eval = None

        # Train
        _fit_xgb_classifier(
            self.model,
            X_sel.fillna(0),
            y,
            eval_set=fit_eval,
            early_stopping_rounds=early_stopping_rounds if fit_eval else None,
            verbose=verbose > 0,
        )
        self._is_fitted = True

        calibration_features = X_sel
        calibration_labels = y
        if eval_set:
            calibration_features = X_val_sel
            calibration_labels = y_val
        raw_calibration_probs = self.model.predict_proba(calibration_features.fillna(0))
        self.prob_calibrator = MultiClassPlattCalibrator().fit(raw_calibration_probs, calibration_labels)

        # Compute in-sample metrics (informational only)
        y_pred = self.model.predict(X_sel.fillna(0))
        train_acc = accuracy_score(y, y_pred)
        train_f1  = f1_score(y, y_pred, average="weighted", zero_division=0)

        best_ntree_limit = getattr(self.model, "best_ntree_limit", None)
        if not best_ntree_limit:
            best_iteration = getattr(self.model, "best_iteration", None)
            if best_iteration is not None:
                best_ntree_limit = int(best_iteration) + 1
        if not best_ntree_limit:
            best_ntree_limit = self.params["n_estimators"]

        metrics = {
            "train_accuracy": round(train_acc, 4),
            "train_f1_weighted": round(train_f1, 4),
            "n_estimators_used": best_ntree_limit,
            "n_features": len(self.feature_names_in),
        }
        logger.info(f"XGBoost trained: acc={train_acc:.3f}, f1={train_f1:.3f}, "
                    f"features={len(self.feature_names_in)}")
        return metrics

    def _prepare_runtime_input(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X_aug = add_regime_interactions(X)
            X_sel = self.selector.transform(X_aug).fillna(0)
            return X_sel.to_numpy(dtype=float)

        array = np.asarray(X, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    def predict_proba(self, X) -> np.ndarray:
        """Returns (n_samples, 3) probability array: [p_sell, p_hold, p_buy]"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        prepared = self._prepare_runtime_input(X)
        raw_probs = self.model.predict_proba(prepared)
        return self.prob_calibrator.transform(raw_probs)

    def predict(self, X) -> np.ndarray:
        """Returns class labels: 0=sell, 1=hold, 2=buy"""
        probs = self.predict_proba(X)
        return probs.argmax(axis=1)

    def predict_selected_proba(self, X) -> np.ndarray:
        """Predict probabilities from pre-aligned selected features."""
        return self.predict_proba(X)

    def predict_selected(self, X) -> np.ndarray:
        """Predict classes from pre-aligned selected features."""
        return self.predict(X)

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Return feature importance as a DataFrame."""
        if not self._is_fitted:
            return pd.DataFrame()
        imp = self.model.feature_importances_
        return (
            pd.DataFrame({
                "feature": self.feature_names_in,
                "importance": imp,
            })
            .sort_values("importance", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    def time_series_cv_score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> Dict:
        """
        TimeSeriesSplit cross-validation.
        CRITICAL: split preserves temporal order — no future leakage.
        Each fold: train on past, test on immediate future.
        """
        X_aug = add_regime_interactions(X)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_metrics = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_aug)):
            X_train, X_test = X_aug.iloc[train_idx], X_aug.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Refit selector on this fold's training data
            sel = FeatureSelector()
            X_train_sel = sel.fit_transform(X_train, y_train)
            X_test_sel  = sel.transform(X_test)

            fold_model = xgb.XGBClassifier(**{**self.params, "n_estimators": 300})
            fold_model.fit(X_train_sel.fillna(0), y_train, verbose=False)

            y_pred = fold_model.predict(X_test_sel.fillna(0))
            fold_acc = accuracy_score(y_test, y_pred)
            fold_f1  = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            # Direction accuracy: did we get buy/sell right (ignoring hold)?
            directional_mask = (y_test != 1) & (y_pred != 1)
            dir_acc = (
                accuracy_score(y_test[directional_mask], y_pred[directional_mask])
                if directional_mask.any() else 0
            )

            fold_metrics.append({
                "fold": fold + 1,
                "accuracy": round(fold_acc, 4),
                "f1_weighted": round(fold_f1, 4),
                "directional_accuracy": round(dir_acc, 4),
                "test_samples": len(y_test),
            })
            logger.info(f"Fold {fold+1}: acc={fold_acc:.3f} dir_acc={dir_acc:.3f}")

        summary = pd.DataFrame(fold_metrics)
        return {
            "folds": fold_metrics,
            "mean_accuracy": round(float(summary["accuracy"].mean()), 4),
            "std_accuracy":  round(float(summary["accuracy"].std()), 4),
            "mean_directional_accuracy": round(float(summary["directional_accuracy"].mean()), 4),
            "mean_f1": round(float(summary["f1_weighted"].mean()), 4),
        }

    def save(self, path: Optional[Path] = None) -> Path:
        """Save model + metadata."""
        path = path or MODELS_DIR / "xgboost_model.json"
        self.model.save_model(str(path))
        meta_path = path.with_suffix(".meta.json")
        calibration_path = path.with_suffix(CALIBRATION_SUFFIX)
        with open(meta_path, "w") as f:
            json.dump({
                "feature_names": self.feature_names_in,
                "params": {k: v for k, v in self.params.items() if isinstance(v, (str, int, float, bool))},
                "calibration_path": str(calibration_path),
            }, f)
        with open(calibration_path, "wb") as f:
            pickle.dump(self.prob_calibrator, f)
        logger.info(f"XGBoost saved to {path}")
        return path

    @classmethod
    def load(cls, path: Path) -> "XGBoostSignalModel":
        """Load a saved model."""
        model_obj = cls()
        model_obj.model = xgb.XGBClassifier(n_jobs=DEFAULT_N_JOBS)
        model_obj.model.load_model(str(path))
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            model_obj.feature_names_in = meta.get("feature_names", [])
            model_obj.selector.selected_features = model_obj.feature_names_in
            calibration_path = Path(meta.get("calibration_path") or path.with_suffix(CALIBRATION_SUFFIX))
            if calibration_path.exists():
                try:
                    with open(calibration_path, "rb") as f:
                        model_obj.prob_calibrator = pickle.load(f)
                except Exception:
                    model_obj.prob_calibrator = MultiClassPlattCalibrator()
        model_obj._is_fitted = True
        return model_obj


# ─────────────────────────────────────────────────────────────
# Optuna Hyperparameter Optimization
# ─────────────────────────────────────────────────────────────
class XGBoostOptimizer:
    """
    Finds optimal XGBoost hyperparameters using Optuna Bayesian optimization.
    Runs 100 trials with Hyperband pruning — takes ~15 min on CPU.
    """

    def optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 100,
        n_cv_splits: int = 4,
        timeout_seconds: int = 900,  # 15 minutes max
    ) -> Dict:
        """Run hyperparameter optimization. Returns best params dict."""
        if not OPTUNA_OK:
            logger.warning("Optuna not installed. Using default params.")
            return XGBoostSignalModel.DEFAULT_PARAMS

        X_aug = add_regime_interactions(X)

        def objective(trial: "optuna.Trial") -> float:
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 200, 2000),
                "max_depth":        trial.suggest_int("max_depth", 3, 8),
                "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
                "gamma":            trial.suggest_float("gamma", 0, 0.5),
                "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
                "eval_metric":      "mlogloss",
                "random_state":     42,
                "tree_method":      "hist",
                "n_jobs":           DEFAULT_N_JOBS,
            }

            tscv = TimeSeriesSplit(n_splits=n_cv_splits)
            scores = []
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_aug)):
                X_tr, X_vl = X_aug.iloc[train_idx], X_aug.iloc[val_idx]
                y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]

                model = xgb.XGBClassifier(**params)
                _fit_xgb_classifier(
                    model,
                    X_tr.fillna(0),
                    y_tr,
                    eval_set=[(X_vl.fillna(0), y_vl)],
                    early_stopping_rounds=30,
                    verbose=False,
                )
                y_pred = model.predict(X_vl.fillna(0))
                directional_mask = (y_vl != 1) & (y_pred != 1)
                if directional_mask.any():
                    dir_acc = accuracy_score(y_vl[directional_mask], y_pred[directional_mask])
                else:
                    dir_acc = 0
                scores.append(dir_acc)

                # Optuna pruning
                trial.report(float(np.mean(scores)), step=fold)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            return float(np.mean(scores))

        pruner = optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource=n_cv_splits, reduction_factor=3
        )
        study = optuna.create_study(direction="maximize", pruner=pruner)
        study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds, show_progress_bar=True)

        best = study.best_params
        best.update({
            "eval_metric": "mlogloss",
            "use_label_encoder": False,
            "random_state": 42,
            "tree_method": "hist",
            "n_jobs": DEFAULT_N_JOBS,
        })
        logger.info(f"Optuna complete: best directional_acc={study.best_value:.4f}")
        logger.info(f"Best params: {best}")
        return best
