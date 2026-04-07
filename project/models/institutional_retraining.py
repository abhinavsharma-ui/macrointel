"""
Institutional Retraining Stack
==============================
Adds two production-oriented training paths:
  - event-aware XGBoost retraining for the primary directional model
  - walk-forward meta-model training for take/skip, edge, and drawdown

All artifacts are stored in models/checkpoints so the live runtime can
consume them without extra wiring.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

XGB_REPORT_PATH = CHECKPOINTS_DIR / "xgboost_retrain_report.json"
META_CLASSIFIER_PATH = CHECKPOINTS_DIR / "meta_take_model.joblib"
META_EDGE_PATH = CHECKPOINTS_DIR / "meta_edge_model.joblib"
META_DRAWDOWN_PATH = CHECKPOINTS_DIR / "meta_drawdown_model.joblib"
META_FEATURES_PATH = CHECKPOINTS_DIR / "meta_feature_cols.pkl"
META_REPORT_PATH = CHECKPOINTS_DIR / "meta_walkforward_report.json"


class MetaFeatureBuilder:
    FEATURE_COLUMNS = [
        "is_buy_signal",
        "is_sell_signal",
        "confidence",
        "conviction_score",
        "regime_multiplier",
        "model_agreement",
        "stop_loss_pct",
        "take_profit_pct",
        "risk_reward_ratio",
        "xgb_confirmed",
        "xgb_override_weak_signal",
        "factor_trend",
        "factor_momentum",
        "factor_mean_revert",
        "factor_volume",
        "factor_sentiment",
        "factor_earnings_propagation",
        "factor_close_reversal",
        "close_vs_sma_50",
        "close_vs_sma_200",
        "momentum_20d",
        "momentum_60d",
        "price_acceleration",
        "bb_position",
        "zscore_vs_60d",
        "realized_vol_21d",
        "vol_ratio_20",
        "sentiment_zscore",
        "sentiment_velocity",
        "weighted_sentiment_zscore",
        "news_volume_spike",
        "source_quality_signal",
        "official_event_signal",
        "filing_event_signal",
        "media_sentiment_signal",
        "travel_activity_level",
        "travel_activity_change",
        "earnings_propagation_signal",
        "earnings_propagation_strength",
        "peer_earnings_shock_3d",
        "peer_earnings_shock_7d",
        "peer_earnings_breadth_7d",
        "close_reversal_signal",
        "close_reversal_strength",
        "event_move_strength",
        "event_day_extreme",
        "event_alpha_signal",
        "alpha_signal",
        "is_nse_symbol",
    ]

    FACTOR_MAP = {
        "trend": "factor_trend",
        "momentum": "factor_momentum",
        "mean_revert": "factor_mean_revert",
        "volume": "factor_volume",
        "sentiment": "factor_sentiment",
        "earnings_propagation": "factor_earnings_propagation",
        "close_reversal": "factor_close_reversal",
    }

    ROW_COLUMNS = [
        "close_vs_sma_50",
        "close_vs_sma_200",
        "momentum_20d",
        "momentum_60d",
        "price_acceleration",
        "bb_position",
        "zscore_vs_60d",
        "realized_vol_21d",
        "vol_ratio_20",
        "sentiment_zscore",
        "sentiment_velocity",
        "weighted_sentiment_zscore",
        "news_volume_spike",
        "source_quality_signal",
        "official_event_signal",
        "filing_event_signal",
        "media_sentiment_signal",
        "travel_activity_level",
        "travel_activity_change",
        "earnings_propagation_signal",
        "earnings_propagation_strength",
        "peer_earnings_shock_3d",
        "peer_earnings_shock_7d",
        "peer_earnings_breadth_7d",
        "close_reversal_signal",
        "close_reversal_strength",
        "event_move_strength",
        "event_day_extreme",
        "event_alpha_signal",
        "alpha_signal",
    ]

    @classmethod
    def build(cls, symbol: str, signal: Dict, feature_row: Optional[pd.Series]) -> Dict[str, float]:
        risk = signal.get("risk_parameters", {}) or {}
        factor_scores = signal.get("factor_scores", {}) or {}
        direction = str(signal.get("signal", "neutral") or "neutral").lower()
        xgb_alignment = str(signal.get("xgb_alignment", "") or "")

        payload = {
            "is_buy_signal": 1.0 if direction == "buy" else 0.0,
            "is_sell_signal": 1.0 if direction == "sell" else 0.0,
            "confidence": float(signal.get("confidence", 0.0) or 0.0),
            "conviction_score": float(signal.get("conviction_score", 0.0) or 0.0),
            "regime_multiplier": float(signal.get("regime_multiplier", 1.0) or 1.0),
            "model_agreement": float(signal.get("model_agreement", 0.0) or 0.0),
            "stop_loss_pct": float(risk.get("stop_loss_pct", 2.0) or 2.0),
            "take_profit_pct": float(risk.get("take_profit_pct", 5.0) or 5.0),
            "risk_reward_ratio": float(risk.get("risk_reward_ratio", 2.0) or 2.0),
            "xgb_confirmed": 1.0 if xgb_alignment == "confirmed" else 0.0,
            "xgb_override_weak_signal": 1.0 if xgb_alignment == "override_weak_signal" else 0.0,
            "is_nse_symbol": 1.0 if symbol.endswith(".NS") else 0.0,
        }

        for factor_name, col_name in cls.FACTOR_MAP.items():
            payload[col_name] = float(factor_scores.get(factor_name, 0.0) or 0.0)

        if feature_row is not None:
            for col in cls.ROW_COLUMNS:
                value = feature_row.get(col, 0.0)
                try:
                    payload[col] = float(value) if pd.notna(value) else 0.0
                except Exception:
                    payload[col] = 0.0
        else:
            for col in cls.ROW_COLUMNS:
                payload[col] = 0.0

        for col in cls.FEATURE_COLUMNS:
            payload.setdefault(col, 0.0)
        return payload

    @classmethod
    def to_frame(cls, rows: List[Dict[str, float]]) -> pd.DataFrame:
        frame = pd.DataFrame(rows)
        for col in cls.FEATURE_COLUMNS:
            if col not in frame.columns:
                frame[col] = 0.0
        return frame[cls.FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)


class TrainedMetaModel:
    MIN_MEAN_PRECISION = 0.20
    MIN_MEAN_COVERAGE_PCT = 1.0
    MIN_MEAN_TAKEN_EDGE_PCT = 0.10
    MIN_MEAN_HIT_RATE_PCT = 45.0

    def __init__(
        self,
        classifier,
        edge_model,
        drawdown_model,
        feature_columns: List[str],
        decision_threshold: float = 0.58,
        min_expected_edge_pct: float = 0.10,
        min_edge_ratio: float = 0.12,
    ):
        self.classifier = classifier
        self.edge_model = edge_model
        self.drawdown_model = drawdown_model
        self.feature_columns = feature_columns
        self.decision_threshold = decision_threshold
        self.min_expected_edge_pct = min_expected_edge_pct
        self.min_edge_ratio = min_edge_ratio

    @classmethod
    def is_available(cls) -> bool:
        return (
            META_CLASSIFIER_PATH.exists()
            and META_EDGE_PATH.exists()
            and META_DRAWDOWN_PATH.exists()
            and META_FEATURES_PATH.exists()
            and META_REPORT_PATH.exists()
        )

    @classmethod
    def _report_is_acceptable(cls, report: Dict) -> bool:
        walk_forward = report.get("walk_forward", {}) if isinstance(report, dict) else {}
        summary = walk_forward.get("summary", {}) if isinstance(walk_forward, dict) else {}
        if not summary:
            return False
        if float(summary.get("mean_precision", 0.0) or 0.0) < cls.MIN_MEAN_PRECISION:
            return False
        if float(summary.get("mean_coverage_pct", 0.0) or 0.0) < cls.MIN_MEAN_COVERAGE_PCT:
            return False
        if float(summary.get("mean_taken_edge_pct", 0.0) or 0.0) < cls.MIN_MEAN_TAKEN_EDGE_PCT:
            return False
        if float(summary.get("mean_taken_hit_rate_pct", 0.0) or 0.0) < cls.MIN_MEAN_HIT_RATE_PCT:
            return False
        return True

    @classmethod
    def load(cls) -> Optional["TrainedMetaModel"]:
        if not cls.is_available():
            return None
        try:
            with open(META_REPORT_PATH, "r", encoding="utf-8") as f:
                report = json.load(f)
            if not cls._report_is_acceptable(report):
                summary = report.get("walk_forward", {}).get("summary", {})
                logger.warning(
                    "Meta model checkpoint rejected by validation gate: "
                    f"precision={summary.get('mean_precision')} "
                    f"coverage={summary.get('mean_coverage_pct')} "
                    f"edge={summary.get('mean_taken_edge_pct')} "
                    f"hit_rate={summary.get('mean_taken_hit_rate_pct')}"
                )
                return None
            classifier = joblib.load(META_CLASSIFIER_PATH)
            edge_model = joblib.load(META_EDGE_PATH)
            drawdown_model = joblib.load(META_DRAWDOWN_PATH)
            with open(META_FEATURES_PATH, "rb") as f:
                feature_columns = pickle.load(f)
            deployment_rules = report.get("deployment_rules", {}) if isinstance(report, dict) else {}
            return cls(
                classifier,
                edge_model,
                drawdown_model,
                feature_columns,
                decision_threshold=float(
                    os.getenv(
                        "META_MODEL_RUNTIME_THRESHOLD",
                        str(deployment_rules.get("decision_threshold", 0.58) or 0.58),
                    )
                ),
                min_expected_edge_pct=float(
                    os.getenv(
                        "META_MODEL_RUNTIME_MIN_EDGE_PCT",
                        str(deployment_rules.get("min_expected_edge_pct", 0.10) or 0.10),
                    )
                ),
                min_edge_ratio=float(
                    os.getenv(
                        "META_MODEL_RUNTIME_MIN_EDGE_RATIO",
                        str(deployment_rules.get("min_edge_ratio", 0.12) or 0.12),
                    )
                ),
            )
        except Exception as exc:
            logger.warning(f"Meta model load failed: {exc}")
            return None

    def predict_from_dict(self, features: Dict[str, float]) -> Dict[str, float]:
        frame = pd.DataFrame([{col: float(features.get(col, 0.0) or 0.0) for col in self.feature_columns}])
        take_probability = float(self.classifier.predict_proba(frame)[0][1])
        expected_edge_pct = float(self.edge_model.predict(frame)[0])
        expected_drawdown_pct = float(abs(self.drawdown_model.predict(frame)[0]))
        return {
            "take_probability": round(take_probability, 4),
            "expected_edge_pct": round(expected_edge_pct, 3),
            "expected_drawdown_pct": round(max(expected_drawdown_pct, 0.1), 3),
            "decision_threshold": round(float(self.decision_threshold), 4),
            "min_expected_edge_pct": round(float(self.min_expected_edge_pct), 4),
            "min_edge_ratio": round(float(self.min_edge_ratio), 4),
        }


class EventAwareXGBoostRetrainer:
    def __init__(self, horizon: int = 5, train_split: float = 0.85):
        self.horizon = horizon
        self.train_split = train_split

    def _build_dataset(self, feature_matrices: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        from models.train_orchestrator import build_labels

        rows = []
        for symbol, frame in feature_matrices.items():
            if frame is None or frame.empty or len(frame) < 120:
                continue
            numeric = frame.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
            numeric = numeric.dropna(thresh=max(8, int(len(numeric.columns) * 0.6)))
            if numeric.empty:
                continue
            labels = build_labels(numeric, horizon=self.horizon).reindex(numeric.index).fillna(1).astype(int)
            numeric = numeric.copy()
            numeric["label"] = labels
            numeric["symbol"] = symbol
            numeric["timestamp"] = pd.to_datetime(numeric.index)
            rows.append(numeric.reset_index(drop=True))

        if not rows:
            return pd.DataFrame()

        dataset = pd.concat(rows, ignore_index=True)
        dataset = dataset.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        return dataset

    def retrain(self, feature_matrices: Dict[str, pd.DataFrame], run_optuna: bool = False, optuna_trials: int = 40) -> Dict:
        from models.xgboost_model import XGBoostOptimizer, XGBoostSignalModel, add_regime_interactions

        dataset = self._build_dataset(feature_matrices)
        if dataset.empty:
            raise ValueError("No feature history available for XGBoost retraining")

        feature_cols = [c for c in dataset.columns if c not in {"label", "symbol", "timestamp"}]
        X = dataset[feature_cols].fillna(0.0)
        y = dataset["label"].astype(int)

        unique_dates = dataset["timestamp"].sort_values().drop_duplicates().tolist()
        split_idx = max(1, int(len(unique_dates) * self.train_split))
        train_dates = set(unique_dates[:split_idx])
        train_mask = dataset["timestamp"].isin(train_dates)
        if train_mask.all() or (~train_mask).sum() < 200:
            train_mask.iloc[max(1, int(len(train_mask) * self.train_split)) :] = False

        X_train = X.loc[train_mask]
        y_train = y.loc[train_mask]
        X_val = X.loc[~train_mask]
        y_val = y.loc[~train_mask]
        if X_val.empty or y_val.empty:
            X_val = X_train.tail(min(1000, max(200, len(X_train) // 5)))
            y_val = y_train.loc[X_val.index]
            X_train = X_train.drop(index=X_val.index)
            y_train = y_train.drop(index=X_val.index)

        params = None
        if run_optuna:
            params = XGBoostOptimizer().optimize(X_train, y_train, n_trials=optuna_trials)

        model = XGBoostSignalModel(params=params)
        fit_metrics = model.fit(X_train, y_train, eval_set=(X_val, y_val))
        cv_metrics = model.time_series_cv_score(X, y)
        model_path = model.save(CHECKPOINTS_DIR / "xgboost_model.json")

        X_train_aug = add_regime_interactions(X_train)
        X_train_selected = model.selector.transform(X_train_aug).fillna(0.0)
        scaler = StandardScaler(with_mean=False, with_std=False)
        scaler.fit(X_train_selected)
        with open(CHECKPOINTS_DIR / "feature_cols.pkl", "wb") as f:
            pickle.dump(model.feature_names_in, f)
        with open(CHECKPOINTS_DIR / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        report = {
            "status": "ok",
            "train_rows": int(len(X_train)),
            "validation_rows": int(len(X_val)),
            "n_features": int(len(model.feature_names_in)),
            "selected_features": model.feature_names_in,
            "fit_metrics": fit_metrics,
            "cv_metrics": cv_metrics,
            "model_path": str(model_path),
        }
        XGB_REPORT_PATH.write_text(json.dumps(report, indent=2))
        logger.info(
            "Event-aware XGBoost retrained: "
            f"{report['train_rows']} train rows | {report['validation_rows']} val rows | "
            f"{report['n_features']} features"
        )
        return report


class MetaModelTrainer:
    def __init__(
        self,
        horizon_days: int = 5,
        take_threshold: float = 0.58,
        walk_forward_folds: int = 4,
        min_train_days: int = 120,
    ):
        self.horizon_days = horizon_days
        self.take_threshold = take_threshold
        self.walk_forward_folds = walk_forward_folds
        self.min_train_days = min_train_days

    def _estimate_risk_parameters(self, row: pd.Series) -> Dict[str, float]:
        atr_pct = float(row.get("atr_pct", 0.02) or 0.02)
        stop_loss_pct = float(np.clip(atr_pct * 100 * 1.3, 1.2, 6.0))
        take_profit_pct = float(np.clip(stop_loss_pct * 2.2, stop_loss_pct + 0.8, 12.0))
        return {
            "stop_loss_pct": round(stop_loss_pct, 3),
            "take_profit_pct": round(take_profit_pct, 3),
            "risk_reward_ratio": round(take_profit_pct / max(stop_loss_pct, 0.1), 3),
        }

    def _directional_path_stats(
        self,
        signal_direction: str,
        entry_close: float,
        future_close: pd.Series,
        stop_loss_pct: float,
        take_profit_pct: float,
    ) -> Dict[str, float]:
        if signal_direction == "sell":
            pnl_path = (entry_close / future_close.replace(0, np.nan) - 1.0) * 100.0
        else:
            pnl_path = (future_close / entry_close - 1.0) * 100.0

        pnl_path = pnl_path.replace([np.inf, -np.inf], np.nan).dropna()
        if pnl_path.empty:
            return {"edge_pct": 0.0, "drawdown_pct": 0.0, "hit": 0}

        stop_hit = pnl_path[pnl_path <= -abs(stop_loss_pct)]
        take_hit = pnl_path[pnl_path >= abs(take_profit_pct)]

        exit_edge = float(pnl_path.iloc[-1])
        hit = 1 if exit_edge > 0 else 0
        if not stop_hit.empty and (take_hit.empty or stop_hit.index[0] <= take_hit.index[0]):
            exit_edge = -abs(stop_loss_pct)
            hit = 0
        elif not take_hit.empty and (stop_hit.empty or take_hit.index[0] < stop_hit.index[0]):
            exit_edge = abs(take_profit_pct)
            hit = 1

        edge_pct = float(exit_edge)
        drawdown_pct = float(abs(min(float(pnl_path.min()), 0.0)))
        return {
            "edge_pct": round(edge_pct, 4),
            "drawdown_pct": round(drawdown_pct, 4),
            "hit": hit,
        }

    def _sample_weights(self, edge_pct: pd.Series, drawdown_pct: pd.Series, take_label: pd.Series) -> np.ndarray:
        edge_component = np.clip(np.abs(edge_pct.to_numpy(dtype=float)) / 3.0, 0.5, 3.0)
        draw_component = 1.0 / np.clip(drawdown_pct.to_numpy(dtype=float), 0.5, 8.0)
        label_component = np.where(take_label.to_numpy(dtype=int) == 1, 1.4, 1.0)
        return edge_component * (0.7 + draw_component) * label_component

    def _select_decision_rule(
        self,
        df: pd.DataFrame,
        take_prob: np.ndarray,
        edge_pred: np.ndarray,
        drawdown_pred: np.ndarray,
    ) -> Dict[str, float]:
        best = {
            "decision_threshold": self.take_threshold,
            "min_expected_edge_pct": 0.10,
            "min_edge_ratio": 0.12,
            "objective": -999.0,
        }
        realized_edge = df["edge_pct"].to_numpy(dtype=float)
        realized_draw = df["drawdown_pct"].to_numpy(dtype=float)
        hit_rate_array = (realized_edge > 0).astype(float)

        threshold_grid = [0.48, 0.52, 0.56, 0.60, 0.64]
        min_edge_grid = [0.05, 0.10, 0.20, 0.35]
        min_ratio_grid = [0.05, 0.10, 0.15, 0.20, 0.30]
        edge_ratio_pred = edge_pred / np.maximum(drawdown_pred, 0.35)

        for threshold in threshold_grid:
            for min_edge in min_edge_grid:
                for min_ratio in min_ratio_grid:
                    mask = (
                        (take_prob >= threshold)
                        & (edge_pred >= min_edge)
                        & (edge_ratio_pred >= min_ratio)
                    )
                    coverage = float(mask.mean() * 100.0)
                    count = int(mask.sum())
                    if count < 30 or coverage < 1.0 or coverage > 35.0:
                        continue
                    avg_edge = float(realized_edge[mask].mean())
                    avg_draw = float(realized_draw[mask].mean())
                    hit_rate = float(hit_rate_array[mask].mean() * 100.0)
                    objective = avg_edge - (0.32 * avg_draw) + ((hit_rate - 50.0) * 0.02)
                    if objective > best["objective"]:
                        best = {
                            "decision_threshold": threshold,
                            "min_expected_edge_pct": min_edge,
                            "min_edge_ratio": min_ratio,
                            "objective": objective,
                        }

        best.pop("objective", None)
        return best

    def build_training_dataset(self, feature_matrices: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        from core.signal_engine_v2 import MultiFactorScorer

        scorer = MultiFactorScorer()
        records = []

        for symbol, frame in feature_matrices.items():
            if frame is None or frame.empty or len(frame) < max(90, self.horizon_days + 30):
                continue

            ordered = frame.sort_index()
            last_usable_idx = len(ordered) - self.horizon_days
            for idx in range(60, last_usable_idx):
                row = ordered.iloc[idx]
                signal_score = scorer.score(row)
                direction = signal_score["direction"]
                if direction == "neutral":
                    synthetic_direction = (
                        0.45 * float(row.get("alpha_signal", 0.0) or 0.0)
                        + 0.35 * float(row.get("event_alpha_signal", 0.0) or 0.0)
                        + 0.20 * float(row.get("momentum_composite", 0.0) or 0.0)
                    )
                    if synthetic_direction > 0.10:
                        direction = "buy"
                    elif synthetic_direction < -0.10:
                        direction = "sell"
                    else:
                        continue
                    signal_score["direction"] = direction
                    signal_score["confidence"] = max(
                        float(signal_score.get("confidence", 0.0) or 0.0),
                        min(0.58, abs(synthetic_direction) * 1.75),
                    )
                    signal_score["conviction_score"] = max(
                        float(signal_score.get("conviction_score", 0.0) or 0.0),
                        round(min(6.6, abs(synthetic_direction) * 12.0), 1),
                    )

                future_window = ordered.iloc[idx + 1 : idx + 1 + self.horizon_days]
                if future_window.empty:
                    continue

                entry_close = float(row.get("close", 0.0) or 0.0)
                if entry_close <= 0:
                    continue

                risk_parameters = self._estimate_risk_parameters(row)
                path_stats = self._directional_path_stats(
                    direction,
                    entry_close,
                    future_window["close"],
                    stop_loss_pct=float(risk_parameters["stop_loss_pct"]),
                    take_profit_pct=float(risk_parameters["take_profit_pct"]),
                )
                feature_signal = {
                    "signal": direction,
                    "confidence": signal_score["confidence"],
                    "conviction_score": signal_score["conviction_score"],
                    "regime_multiplier": signal_score["regime_multiplier"],
                    "regime": signal_score["regime"],
                    "factor_scores": signal_score["factor_scores"],
                    "risk_parameters": risk_parameters,
                    "model_agreement": 0.0,
                    "xgb_alignment": "",
                }
                feature_values = MetaFeatureBuilder.build(symbol, feature_signal, row)

                edge_ratio = path_stats["edge_pct"] / max(path_stats["drawdown_pct"], 0.35)
                take_label = int(
                    path_stats["edge_pct"] > 0.30
                    and edge_ratio > 0.18
                    and path_stats["hit"] == 1
                )
                feature_values.update(
                    {
                        "symbol": symbol,
                        "timestamp": pd.Timestamp(ordered.index[idx]),
                        "take_label": take_label,
                        "edge_pct": path_stats["edge_pct"],
                        "drawdown_pct": path_stats["drawdown_pct"],
                        "edge_ratio": round(float(edge_ratio), 4),
                        "hit": path_stats["hit"],
                    }
                )
                records.append(feature_values)

        if not records:
            return pd.DataFrame()

        dataset = pd.DataFrame(records)
        dataset = dataset.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        return dataset

    def _fit_models(self, X: pd.DataFrame, y_take: pd.Series, y_edge: pd.Series, y_drawdown: pd.Series):
        classifier = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=4,
            max_iter=320,
            min_samples_leaf=28,
            l2_regularization=0.15,
            random_state=42,
        )
        edge_model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=5,
            max_iter=320,
            min_samples_leaf=24,
            l2_regularization=0.10,
            random_state=42,
        )
        drawdown_model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=5,
            max_iter=320,
            min_samples_leaf=24,
            l2_regularization=0.10,
            random_state=42,
            loss="absolute_error",
        )
        sample_weight = self._sample_weights(y_edge, y_drawdown, y_take)
        classifier.fit(X, y_take, sample_weight=sample_weight)
        edge_model.fit(X, y_edge, sample_weight=sample_weight)
        drawdown_model.fit(X, y_drawdown, sample_weight=sample_weight)
        return classifier, edge_model, drawdown_model

    def walk_forward_validate(self, dataset: pd.DataFrame) -> Dict:
        if dataset.empty:
            return {"status": "failed", "reason": "no_data"}

        dates = sorted(pd.to_datetime(dataset["timestamp"]).drop_duplicates().tolist())
        if len(dates) < (self.min_train_days + 20):
            return {"status": "failed", "reason": "insufficient_dates"}

        test_span = max(20, len(dates) // (self.walk_forward_folds + 2))
        folds = []
        for fold in range(self.walk_forward_folds):
            train_end = len(dates) - (self.walk_forward_folds - fold) * test_span
            test_end = min(len(dates), train_end + test_span)
            if train_end < self.min_train_days or test_end <= train_end:
                continue

            train_dates = set(dates[:train_end])
            test_dates = set(dates[train_end:test_end])
            train_df = dataset[dataset["timestamp"].isin(train_dates)]
            test_df = dataset[dataset["timestamp"].isin(test_dates)]
            if train_df.empty or test_df.empty:
                continue
            if train_df["take_label"].nunique() < 2 or test_df["take_label"].nunique() < 2:
                continue

            X_train = train_df[MetaFeatureBuilder.FEATURE_COLUMNS].fillna(0.0)
            X_test = test_df[MetaFeatureBuilder.FEATURE_COLUMNS].fillna(0.0)
            classifier, edge_model, drawdown_model = self._fit_models(
                X_train,
                train_df["take_label"],
                train_df["edge_pct"],
                train_df["drawdown_pct"],
            )
            train_take_prob = classifier.predict_proba(X_train)[:, 1]
            train_edge_pred = edge_model.predict(X_train)
            train_drawdown_pred = np.abs(drawdown_model.predict(X_train))
            decision_rule = self._select_decision_rule(
                train_df,
                train_take_prob,
                train_edge_pred,
                train_drawdown_pred,
            )
            take_prob = classifier.predict_proba(X_test)[:, 1]
            edge_pred = edge_model.predict(X_test)
            drawdown_pred = np.abs(drawdown_model.predict(X_test))
            pred_take = (
                (take_prob >= decision_rule["decision_threshold"])
                & (edge_pred >= decision_rule["min_expected_edge_pct"])
                & ((edge_pred / np.maximum(drawdown_pred, 0.35)) >= decision_rule["min_edge_ratio"])
            )

            realized_edges = test_df["edge_pct"].to_numpy(dtype=float)
            realized_drawdown = test_df["drawdown_pct"].to_numpy(dtype=float)
            taken_edges = realized_edges[pred_take]
            taken_drawdown = realized_drawdown[pred_take]

            folds.append(
                {
                    "fold": fold + 1,
                    "train_rows": int(len(train_df)),
                    "test_rows": int(len(test_df)),
                    "accuracy": round(float(accuracy_score(test_df["take_label"], pred_take.astype(int))), 4),
                    "precision": round(
                        float(precision_score(test_df["take_label"], pred_take.astype(int), zero_division=0)),
                        4,
                    ),
                    "recall": round(
                        float(recall_score(test_df["take_label"], pred_take.astype(int), zero_division=0)),
                        4,
                    ),
                    "coverage_pct": round(float(pred_take.mean() * 100.0), 2),
                    "decision_threshold": round(float(decision_rule["decision_threshold"]), 4),
                    "min_expected_edge_pct": round(float(decision_rule["min_expected_edge_pct"]), 4),
                    "min_edge_ratio": round(float(decision_rule["min_edge_ratio"]), 4),
                    "taken_avg_edge_pct": round(float(np.mean(taken_edges)) if len(taken_edges) else 0.0, 3),
                    "taken_avg_drawdown_pct": round(float(np.mean(taken_drawdown)) if len(taken_drawdown) else 0.0, 3),
                    "taken_hit_rate_pct": round(float(np.mean(taken_edges > 0) * 100.0) if len(taken_edges) else 0.0, 2),
                }
            )

        if not folds:
            return {"status": "failed", "reason": "no_valid_folds"}

        summary = pd.DataFrame(folds)
        return {
            "status": "ok",
            "folds": folds,
            "summary": {
                "mean_accuracy": round(float(summary["accuracy"].mean()), 4),
                "mean_precision": round(float(summary["precision"].mean()), 4),
                "mean_recall": round(float(summary["recall"].mean()), 4),
                "mean_coverage_pct": round(float(summary["coverage_pct"].mean()), 2),
                "mean_taken_edge_pct": round(float(summary["taken_avg_edge_pct"].mean()), 3),
                "mean_taken_drawdown_pct": round(float(summary["taken_avg_drawdown_pct"].mean()), 3),
                "mean_taken_hit_rate_pct": round(float(summary["taken_hit_rate_pct"].mean()), 2),
            },
        }

    def train(self, feature_matrices: Dict[str, pd.DataFrame]) -> Dict:
        dataset = self.build_training_dataset(feature_matrices)
        if dataset.empty:
            raise ValueError("No training rows available for meta model")
        if dataset["take_label"].nunique() < 2:
            raise ValueError("Meta labels do not contain both classes")

        walk_forward = self.walk_forward_validate(dataset)
        X = dataset[MetaFeatureBuilder.FEATURE_COLUMNS].fillna(0.0)
        classifier, edge_model, drawdown_model = self._fit_models(
            X,
            dataset["take_label"],
            dataset["edge_pct"],
            dataset["drawdown_pct"],
        )
        full_take_prob = classifier.predict_proba(X)[:, 1]
        full_edge_pred = edge_model.predict(X)
        full_drawdown_pred = np.abs(drawdown_model.predict(X))
        deployment_rules = self._select_decision_rule(
            dataset,
            full_take_prob,
            full_edge_pred,
            full_drawdown_pred,
        )

        joblib.dump(classifier, META_CLASSIFIER_PATH)
        joblib.dump(edge_model, META_EDGE_PATH)
        joblib.dump(drawdown_model, META_DRAWDOWN_PATH)
        with open(META_FEATURES_PATH, "wb") as f:
            pickle.dump(MetaFeatureBuilder.FEATURE_COLUMNS, f)

        report = {
            "status": "ok",
            "rows": int(len(dataset)),
            "positive_labels": int(dataset["take_label"].sum()),
            "feature_count": len(MetaFeatureBuilder.FEATURE_COLUMNS),
            "deployment_rules": deployment_rules,
            "walk_forward": walk_forward,
        }
        META_REPORT_PATH.write_text(json.dumps(report, indent=2))
        logger.info(
            "Meta model trained: "
            f"{report['rows']} rows | {report['positive_labels']} positive labels | "
            f"{report['feature_count']} features"
        )
        return report


class InstitutionalTrainingPipeline:
    def __init__(
        self,
        xgb_horizon: int = 5,
        meta_horizon: int = 5,
        meta_take_threshold: float = 0.58,
        meta_walk_forward_folds: int = 4,
    ):
        self.xgb_retrainer = EventAwareXGBoostRetrainer(horizon=xgb_horizon)
        self.meta_trainer = MetaModelTrainer(
            horizon_days=meta_horizon,
            take_threshold=meta_take_threshold,
            walk_forward_folds=meta_walk_forward_folds,
        )

    def train_all(self, feature_matrices: Dict[str, pd.DataFrame], run_optuna: bool = False) -> Dict:
        xgb_report = self.xgb_retrainer.retrain(feature_matrices, run_optuna=run_optuna)
        meta_report = self.meta_trainer.train(feature_matrices)
        return {
            "xgboost": xgb_report,
            "meta_model": meta_report,
        }
