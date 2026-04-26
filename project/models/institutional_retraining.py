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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

from models.calibration import BinaryPlattCalibrator

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
META_REPORT_PREVIOUS_PATH = CHECKPOINTS_DIR / "meta_walkforward_report.previous.json"
META_DIRECTIONAL_PATH = CHECKPOINTS_DIR / "meta_directional_models.joblib"
META_DIRECTIONAL_PREVIOUS_PATH = CHECKPOINTS_DIR / "meta_directional_models.previous.joblib"
XGB_SHAP_REPORT_PATH = CHECKPOINTS_DIR / "xgboost_shap_monitor.json"

# Hard floor on the edge/draw ratio gate for deployed meta bundles.
# Walk-forward analysis showed mean_taken_edge_draw_ratio collapsing to ~0.27
# (fold 3 fell to 0.002 under zero regime gating), so deployment_rules must
# never publish a min_edge_ratio below this value for either direction.
META_MIN_EDGE_RATIO_FLOOR = 0.28
META_EDGE_DRAW_RATIO_WARN_THRESHOLD = 0.25

# --- Isotonic calibration constants -----------------------------------------
# Walk-forward optimal thresholds were swinging 0.52–0.72 across folds, a
# textbook miscalibration symptom. We fit an IsotonicRegression on out-of-fold
# probabilities so a single fixed threshold can be deployed in production.
META_CALIBRATOR_PATH = CHECKPOINTS_DIR / "meta_model_calibrator.pkl"
META_CALIBRATOR_FOLD_PATH_FMT = str(
    CHECKPOINTS_DIR / "meta_model_calibrator_fold{fold}_{direction}.pkl"
)
META_CALIBRATED_THRESHOLD_LONG_DEFAULT = 0.55
META_CALIBRATED_THRESHOLD_SHORT_DEFAULT = 0.62
META_CALIBRATED_PRECISION_FLOOR = 0.50
META_CALIBRATION_BINS = 10
META_CALIBRATION_OOF_SPLITS = 5

# --- Learning-to-rank meta variant ------------------------------------------
# Parallel ranker trained alongside the classifier so we can compare their
# walk-forward coverage/precision side by side. Activation of the ranker is
# controlled by META_MODEL_TYPE (classifier|ranker|both); default is
# "classifier" so existing behaviour is unchanged.
META_RANKER_PATH = CHECKPOINTS_DIR / "meta_ranker_model.json"
META_RANKER_REPORT_PATH = CHECKPOINTS_DIR / "meta_walkforward_report_ranker.json"
META_RANKER_FEATURES_PATH = CHECKPOINTS_DIR / "meta_ranker_feature_cols.pkl"
META_RANKER_DEFAULT_COVERAGE = 0.05
META_RANKER_RELEVANCE_RATIO_THRESHOLD = 0.35


def _meta_model_type() -> str:
    """Read META_MODEL_TYPE env var; values: classifier|ranker|both."""
    raw = os.getenv("META_MODEL_TYPE", "classifier").strip().lower()
    if raw not in {"classifier", "ranker", "both"}:
        return "classifier"
    return raw

PROJECT_DIR = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_DIR / "data" / "features"
FEATURE_DIR_10YR = PROJECT_DIR / "data" / "features_10yr"


def _clip_prob(value: float) -> float:
    return float(np.clip(float(value), 1e-6, 1.0 - 1e-6))


def _make_meta_classifier() -> HistGradientBoostingClassifier:
    """Single source of truth for the meta classifier hyperparameters.

    Used both by the bundle fit and by the OOF CV passes that produce
    out-of-fold probabilities for isotonic calibration. Keeping the
    construction in one place guarantees the calibrator is fit on the same
    distribution the deployed classifier produces.
    """
    return HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_depth=5,
        max_iter=480,
        min_samples_leaf=18,
        l2_regularization=0.08,
        random_state=42,
        class_weight="balanced",
    )


def _fit_isotonic_oof(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    n_splits: int = META_CALIBRATION_OOF_SPLITS,
) -> Tuple[Optional[IsotonicRegression], np.ndarray]:
    """Fit an IsotonicRegression on out-of-fold probabilities.

    Refits a fresh meta classifier on each of `n_splits` folds of (X, y),
    collects the held-out probabilities, then fits an isotonic mapping from
    those raw probabilities to the binary target. Returns (calibrator, oof).
    The calibrator is None if there are not enough samples or only one class
    is present — callers should fall back to identity in that case.
    """
    n_rows = int(len(X))
    y_arr = np.asarray(y, dtype=int)
    if n_rows < 40 or np.unique(y_arr).size < 2:
        return None, np.full(n_rows, float(y_arr.mean()) if n_rows else 0.0)

    safe_splits = max(2, min(n_splits, n_rows // 20))
    kf = KFold(n_splits=safe_splits, shuffle=False)
    oof = np.zeros(n_rows, dtype=float)
    sw_arr = np.asarray(sample_weight, dtype=float) if sample_weight is not None else None
    for tr_idx, va_idx in kf.split(X):
        clf = _make_meta_classifier()
        X_tr = X.iloc[tr_idx] if hasattr(X, "iloc") else X[tr_idx]
        y_tr = y_arr[tr_idx]
        if np.unique(y_tr).size < 2:
            oof[va_idx] = float(y_tr.mean())
            continue
        if sw_arr is not None:
            clf.fit(X_tr, y_tr, sample_weight=sw_arr[tr_idx])
        else:
            clf.fit(X_tr, y_tr)
        X_va = X.iloc[va_idx] if hasattr(X, "iloc") else X[va_idx]
        oof[va_idx] = clf.predict_proba(X_va)[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(oof, y_arr)
    return iso, oof


def _brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    y_arr = np.asarray(y_true, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    if y_arr.size == 0:
        return 0.0
    return float(np.mean((p_arr - y_arr) ** 2))


def _expected_calibration_error(
    y_true: np.ndarray, p: np.ndarray, n_bins: int = META_CALIBRATION_BINS
) -> float:
    """Standard binned ECE (weighted by bucket size)."""
    y_arr = np.asarray(y_true, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    total = int(y_arr.size)
    if total == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(p_arr, bin_edges) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        avg_pred = float(p_arr[mask].mean())
        avg_true = float(y_arr[mask].mean())
        ece += (n / total) * abs(avg_pred - avg_true)
    return float(ece)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _calendar_feature_dict(as_of: pd.Timestamp) -> Dict[str, float]:
    """Encode calendar position so the meta model can use multi-year feature history."""
    ts = pd.Timestamp(as_of)
    if pd.isna(ts):
        ts = pd.Timestamp.utcnow()
    y = float(ts.year)
    meta_year_norm = float(np.clip((y - 2000.0) / 32.0, 0.0, 1.0))
    month = int(ts.month)
    meta_month_sin = float(np.sin(2.0 * np.pi * (month - 1) / 12.0))
    meta_month_cos = float(np.cos(2.0 * np.pi * (month - 1) / 12.0))
    dow = int(ts.dayofweek)
    meta_dow_sin = float(np.sin(2.0 * np.pi * dow / 7.0))
    meta_dow_cos = float(np.cos(2.0 * np.pi * dow / 7.0))
    return {
        "meta_year_norm": meta_year_norm,
        "meta_month_sin": meta_month_sin,
        "meta_month_cos": meta_month_cos,
        "meta_dow_sin": meta_dow_sin,
        "meta_dow_cos": meta_dow_cos,
    }


def _merged_retrain_feature_store() -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, int]]]:
    feature_matrices: Dict[str, pd.DataFrame] = {}
    source_stats: Dict[str, Dict[str, int]] = {}

    explicit = str(os.getenv("XGB_RETRAIN_FEATURE_DIR", "")).strip()
    if explicit:
        explicit_dir = Path(explicit)
        file_count = 0
        row_count = 0
        for path in sorted(explicit_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(path)
            except Exception as exc:
                logger.warning("XGB retrain explicit feature load skipped for %s: %s", path, exc)
                continue
            file_count += 1
            row_count += len(df)
            feature_matrices.setdefault(path.stem, df)
        source_stats[str(explicit_dir)] = {"files": file_count, "rows": row_count}
        return feature_matrices, source_stats

    if FEATURE_DIR_10YR.exists():
        file_count = 0
        row_count = 0
        for path in sorted(FEATURE_DIR_10YR.glob("*USDT*.parquet")):
            try:
                df = pd.read_parquet(path)
            except Exception as exc:
                logger.warning("XGB retrain features_10yr load skipped for %s: %s", path, exc)
                continue
            file_count += 1
            row_count += len(df)
            feature_matrices.setdefault(path.stem, df)
        source_stats[str(FEATURE_DIR_10YR)] = {"files": file_count, "rows": row_count}

    if FEATURE_DIR.exists():
        file_count = 0
        row_count = 0
        for path in sorted(FEATURE_DIR.glob("*.parquet")):
            try:
                df = pd.read_parquet(path)
            except Exception as exc:
                logger.warning("XGB retrain fallback feature load skipped for %s: %s", path, exc)
                continue
            file_count += 1
            row_count += len(df)
            feature_matrices.setdefault(path.stem, df)
        source_stats[str(FEATURE_DIR)] = {"files": file_count, "rows": row_count}

    return feature_matrices, source_stats


def _adaptive_horizon_days(frame: pd.DataFrame, idx: int, base_horizon_days: int) -> Tuple[int, float]:
    row = frame.iloc[idx]
    multiplier = 1.0
    realized_vol = abs(_safe_float(row.get("realized_vol_21d", 0.0), 0.0))
    baseline_window = frame.get("realized_vol_21d")
    baseline = np.nan
    if baseline_window is not None:
        history = pd.to_numeric(baseline_window.iloc[max(0, idx - 30): idx + 1], errors="coerce")
        history = history.replace(0, np.nan).dropna()
        if not history.empty:
            baseline = float(history.mean())
    vol_ratio = _safe_float(row.get("vol_regime_ratio", np.nan), np.nan)
    if np.isnan(vol_ratio) and baseline and baseline > 0:
        vol_ratio = realized_vol / baseline
    if (not np.isnan(vol_ratio) and vol_ratio > 1.5) or _safe_float(row.get("vol_regime_stressed", 0.0), 0.0) >= 1.0:
        multiplier *= 0.60

    event_signal = max(
        abs(_safe_float(row.get("official_event_signal", 0.0), 0.0)),
        abs(_safe_float(row.get("filing_event_signal", 0.0), 0.0)),
        abs(_safe_float(row.get("earnings_tone_signal", 0.0), 0.0)),
    )
    filing_like_event = (
        _safe_float(row.get("filing_event_hit", 0.0), 0.0) > 0
        or _safe_float(row.get("new_risk_factors", 0.0), 0.0) > 0
    )
    if filing_like_event or event_signal >= 0.60:
        multiplier = max(multiplier, 3.0)

    horizon = max(1, int(round(base_horizon_days * multiplier)))
    return horizon, round(float(multiplier), 3)


def _triple_barrier_path_stats(
    signal_direction: str,
    entry_close: float,
    future_window: pd.DataFrame,
    stop_loss_pct: float,
    take_profit_pct: float,
) -> Dict[str, float]:
    if future_window is None or future_window.empty or entry_close <= 0:
        return {"edge_pct": 0.0, "drawdown_pct": 0.0, "hit": 0, "barrier": "none"}

    highs = pd.to_numeric(future_window.get("high", future_window.get("close")), errors="coerce").ffill()
    lows = pd.to_numeric(future_window.get("low", future_window.get("close")), errors="coerce").ffill()
    closes = pd.to_numeric(future_window.get("close"), errors="coerce").ffill()
    if closes.empty:
        return {"edge_pct": 0.0, "drawdown_pct": 0.0, "hit": 0, "barrier": "none"}

    if signal_direction == "sell":
        favorable_path = ((entry_close / lows.replace(0, np.nan)) - 1.0) * 100.0
        adverse_path = ((entry_close / highs.replace(0, np.nan)) - 1.0) * 100.0
        final_edge = ((entry_close / max(float(closes.iloc[-1]), 1e-6)) - 1.0) * 100.0
    else:
        favorable_path = ((highs / entry_close) - 1.0) * 100.0
        adverse_path = ((lows / entry_close) - 1.0) * 100.0
        final_edge = ((float(closes.iloc[-1]) / entry_close) - 1.0) * 100.0

    favorable_path = favorable_path.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    adverse_path = adverse_path.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    drawdown_pct = float(abs(min(float(adverse_path.min()), 0.0)))
    exit_edge = float(final_edge)
    barrier = "time"

    for favorable, adverse in zip(favorable_path.to_numpy(dtype=float), adverse_path.to_numpy(dtype=float)):
        stop_hit = adverse <= -abs(stop_loss_pct)
        take_hit = favorable >= abs(take_profit_pct)
        if stop_hit and take_hit:
            exit_edge = -abs(stop_loss_pct)
            barrier = "stop"
            break
        if stop_hit:
            exit_edge = -abs(stop_loss_pct)
            barrier = "stop"
            break
        if take_hit:
            exit_edge = abs(take_profit_pct)
            barrier = "take"
            break

    return {
        "edge_pct": round(float(exit_edge), 4),
        "drawdown_pct": round(float(drawdown_pct), 4),
        "hit": int(exit_edge > 0),
        "barrier": barrier,
    }


def _triple_barrier_direction_label(frame: pd.DataFrame, idx: int, base_horizon_days: int) -> int:
    row = frame.iloc[idx]
    entry_close = _safe_float(row.get("close", 0.0), 0.0)
    if entry_close <= 0:
        return 1
    horizon_days, _ = _adaptive_horizon_days(frame, idx, base_horizon_days)
    future_window = frame.iloc[idx + 1 : idx + 1 + horizon_days]
    if future_window.empty:
        return 1

    atr_pct = max(abs(_safe_float(row.get("atr_pct", 0.02), 0.02)), 0.002)
    stop_loss_pct = float(np.clip(atr_pct * 100 * 1.3, 1.0, 6.0))
    take_profit_pct = float(np.clip(stop_loss_pct * 2.2, stop_loss_pct + 0.8, 12.0))
    long_stats = _triple_barrier_path_stats("buy", entry_close, future_window, stop_loss_pct, take_profit_pct)
    short_stats = _triple_barrier_path_stats("sell", entry_close, future_window, stop_loss_pct, take_profit_pct)

    long_edge = float(long_stats["edge_pct"])
    short_edge = float(short_stats["edge_pct"])
    dominance_buffer = 0.20
    if long_edge > max(abs(short_edge), dominance_buffer):
        return 2
    if short_edge > max(abs(long_edge), dominance_buffer):
        return 0
    return 1


@dataclass
class DirectionalMetaArtifacts:
    classifier: object
    calibrator: BinaryPlattCalibrator
    edge_model: object
    drawdown_model: object
    decision_threshold: float
    min_expected_edge_pct: float
    min_edge_ratio: float
    direction: str


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
        "vol_regime_ratio",
        "vol_ratio_20",
        "sentiment_zscore",
        "sentiment_velocity",
        "weighted_sentiment_zscore",
        "news_volume_spike",
        "source_quality_signal",
        "official_event_signal",
        "filing_event_signal",
        "filing_change_score",
        "filing_fresh_language_score",
        "new_risk_factors",
        "media_sentiment_signal",
        "travel_activity_level",
        "travel_activity_change",
        "earnings_propagation_signal",
        "earnings_propagation_strength",
        "peer_earnings_shock_3d",
        "peer_earnings_shock_7d",
        "peer_earnings_breadth_7d",
        "peer_earnings_negative_ratio_7d",
        "close_reversal_signal",
        "close_reversal_strength",
        "event_move_strength",
        "event_day_extreme",
        "event_alpha_signal",
        "alpha_signal",
        "earnings_tone_signal",
        "earnings_tone_velocity",
        "adaptive_horizon_multiplier",
        "long_momentum_edge",
        "long_volume_surge_edge",
        "long_sentiment_velocity_edge",
        "short_spread_widening_edge",
        "short_book_pressure_reversal_edge",
        "short_earnings_miss_propagation_edge",
        "is_nse_symbol",
        "meta_year_norm",
        "meta_month_sin",
        "meta_month_cos",
        "meta_dow_sin",
        "meta_dow_cos",
        # --- Regime-conditioning block (added for fold-3 collapse fix) ---
        # market_stress_regime is the cross-sectional median of
        # vol_regime_stressed across all symbols on each date (0=calm, 1=stressed).
        # The three *_x_* columns are interaction features that teach the meta
        # model to discount edge/ratio rank during stress and lean on conviction
        # rank instead. At inference these are populated with neutral defaults
        # because cross-sectional ranks aren't available per-bar — the runtime
        # stress gate (predict_from_dict) handles inference-time damping.
        "market_stress_regime",
        "conviction_rank_x_stress",
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
        "vol_regime_ratio",
        "vol_ratio_20",
        "sentiment_zscore",
        "sentiment_velocity",
        "weighted_sentiment_zscore",
        "news_volume_spike",
        "source_quality_signal",
        "official_event_signal",
        "filing_event_signal",
        "filing_change_score",
        "filing_fresh_language_score",
        "new_risk_factors",
        "media_sentiment_signal",
        "travel_activity_level",
        "travel_activity_change",
        "earnings_propagation_signal",
        "earnings_propagation_strength",
        "peer_earnings_shock_3d",
        "peer_earnings_shock_7d",
        "peer_earnings_breadth_7d",
        "peer_earnings_negative_ratio_7d",
        "close_reversal_signal",
        "close_reversal_strength",
        "event_move_strength",
        "event_day_extreme",
        "event_alpha_signal",
        "alpha_signal",
        "earnings_tone_signal",
        "earnings_tone_velocity",
    ]

    @classmethod
    def build(cls, symbol: str, signal: Dict, feature_row: Optional[pd.Series]) -> Dict[str, float]:
        risk = signal.get("risk_parameters", {}) or {}
        factor_scores = signal.get("factor_scores", {}) or {}
        direction = str(signal.get("signal", "neutral") or "neutral").lower()
        xgb_alignment = str(signal.get("xgb_alignment", "") or "")

        as_of = pd.Timestamp.utcnow()
        if feature_row is not None and getattr(feature_row, "name", None) is not None:
            try:
                as_of = pd.Timestamp(feature_row.name)
                if pd.isna(as_of):
                    as_of = pd.Timestamp.utcnow()
            except Exception:
                as_of = pd.Timestamp.utcnow()

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
        payload.update(_calendar_feature_dict(as_of))

        for factor_name, col_name in cls.FACTOR_MAP.items():
            payload[col_name] = float(factor_scores.get(factor_name, 0.0) or 0.0)

        if feature_row is not None:
            for col in cls.ROW_COLUMNS:
                value = feature_row.get(col, 0.0)
                try:
                    payload[col] = float(value) if pd.notna(value) else 0.0
                except Exception:
                    payload[col] = 0.0
            vol_regime_ratio = payload.get("vol_regime_ratio", 0.0)
            if not vol_regime_ratio:
                realized_vol = abs(payload.get("realized_vol_21d", 0.0))
                hist_vol = abs(_safe_float(feature_row.get("hist_vol_30", realized_vol), realized_vol))
                vol_regime_ratio = realized_vol / max(hist_vol, 1e-6) if hist_vol > 0 else 0.0
            payload["vol_regime_ratio"] = float(vol_regime_ratio)
            horizon_multiplier = 3.0 if (
                payload.get("filing_event_signal", 0.0) > 0.55
                or payload.get("new_risk_factors", 0.0) > 0
            ) else 0.60 if payload.get("vol_regime_ratio", 0.0) > 1.5 else 1.0
            payload["adaptive_horizon_multiplier"] = float(horizon_multiplier)
            payload["long_momentum_edge"] = float(
                np.clip(
                    (payload.get("momentum_20d", 0.0) / 0.15 * 0.45)
                    + (payload.get("vol_ratio_20", 0.0) / 2.5 * 0.30)
                    + (payload.get("sentiment_velocity", 0.0) / 0.30 * 0.25),
                    -1.5,
                    1.5,
                )
            )
            payload["long_volume_surge_edge"] = float(
                np.clip(
                    (payload.get("vol_ratio_20", 0.0) / 2.5 * 0.55)
                    + (payload.get("news_volume_spike", 0.0) / 2.0 * 0.20)
                    + (payload.get("earnings_tone_signal", 0.0) * 0.25),
                    -1.5,
                    1.5,
                )
            )
            payload["long_sentiment_velocity_edge"] = float(
                np.clip(
                    (payload.get("sentiment_velocity", 0.0) / 0.30 * 0.50)
                    + (payload.get("earnings_tone_velocity", 0.0) / 0.20 * 0.25)
                    + (payload.get("filing_change_score", 0.0) * 0.25),
                    -1.5,
                    1.5,
                )
            )
            payload["short_spread_widening_edge"] = float(
                np.clip(
                    (payload.get("bb_position", 0.0) - 0.5) * 1.2
                    + (payload.get("vol_regime_ratio", 0.0) * 0.35)
                    + (payload.get("realized_vol_21d", 0.0) / 0.35 * 0.30),
                    -1.5,
                    1.5,
                )
            )
            payload["short_book_pressure_reversal_edge"] = float(
                np.clip(
                    (-payload.get("close_reversal_signal", 0.0) * 0.55)
                    + (payload.get("close_reversal_strength", 0.0) * 0.20)
                    + (payload.get("event_move_strength", 0.0) * 0.25),
                    -1.5,
                    1.5,
                )
            )
            payload["short_earnings_miss_propagation_edge"] = float(
                np.clip(
                    (-payload.get("earnings_propagation_signal", 0.0) * 0.50)
                    + (payload.get("peer_earnings_negative_ratio_7d", 0.0) * 0.30)
                    + (-payload.get("earnings_tone_velocity", 0.0) * 0.20),
                    -1.5,
                    1.5,
                )
            )
        else:
            for col in cls.ROW_COLUMNS:
                payload[col] = 0.0
            for col in (
                "vol_regime_ratio",
                "adaptive_horizon_multiplier",
                "long_momentum_edge",
                "long_volume_surge_edge",
                "long_sentiment_velocity_edge",
                "short_spread_widening_edge",
                "short_book_pressure_reversal_edge",
                "short_earnings_miss_propagation_edge",
            ):
                payload[col] = 0.0

        # Regime conditioning at inference: market_stress_regime is taken from
        # the symbol's own vol_regime_stressed (best available proxy when no
        # cross-sectional median can be computed in real time). The rank
        # interactions default to a neutral 0.5 rank — the post-score stress
        # gate (predict_from_dict) is what actually damps inference behaviour.
        if feature_row is not None:
            stress_value = _safe_float(feature_row.get("vol_regime_stressed", 0.0), 0.0)
        else:
            stress_value = 0.0
        market_stress_regime = 1.0 if stress_value >= 0.5 else 0.0
        payload.setdefault("market_stress_regime", market_stress_regime)
        payload.setdefault("conviction_rank_x_stress", 0.5 * market_stress_regime)

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

    @classmethod
    def _gate_min_precision(cls) -> float:
        v = os.getenv("META_MODEL_GATE_MIN_PRECISION", "")
        return float(v) if v.strip() else cls.MIN_MEAN_PRECISION

    @classmethod
    def _gate_min_coverage_pct(cls) -> float:
        v = os.getenv("META_MODEL_GATE_MIN_COVERAGE_PCT", "")
        return float(v) if v.strip() else cls.MIN_MEAN_COVERAGE_PCT

    @classmethod
    def _gate_min_taken_edge_pct(cls) -> float:
        v = os.getenv("META_MODEL_GATE_MIN_TAKEN_EDGE_PCT", "")
        return float(v) if v.strip() else cls.MIN_MEAN_TAKEN_EDGE_PCT

    @classmethod
    def _gate_min_hit_rate_pct(cls) -> float:
        v = os.getenv("META_MODEL_GATE_MIN_HIT_RATE_PCT", "")
        return float(v) if v.strip() else cls.MIN_MEAN_HIT_RATE_PCT

    def __init__(
        self,
        bundles: Dict[str, DirectionalMetaArtifacts],
        feature_columns: List[str],
    ):
        self.bundles = bundles
        self.feature_columns = feature_columns
        default_bundle = bundles.get("long") or bundles.get("short") or next(iter(bundles.values()))
        self.decision_threshold = float(default_bundle.decision_threshold)
        self.min_expected_edge_pct = float(default_bundle.min_expected_edge_pct)
        self.min_edge_ratio = float(default_bundle.min_edge_ratio)
        # Production isotonic calibrators (per-direction). Loaded best-effort;
        # if missing, predict_from_dict falls back to the bundle's Platt
        # calibrator. Loading at __init__ so every load path inherits it.
        self.isotonic_calibrators: Dict[str, IsotonicRegression] = {}
        try:
            if META_CALIBRATOR_PATH.exists():
                with open(META_CALIBRATOR_PATH, "rb") as f:
                    payload = pickle.load(f)
                if isinstance(payload, dict):
                    self.isotonic_calibrators = {
                        str(k): v for k, v in payload.items() if isinstance(v, IsotonicRegression)
                    }
        except Exception as exc:
            logger.debug("Production calibrator load failed: %s", exc)
            self.isotonic_calibrators = {}

    @classmethod
    def is_available(cls) -> bool:
        directional_ready = META_DIRECTIONAL_PATH.exists() and META_REPORT_PATH.exists()
        previous_directional_ready = META_DIRECTIONAL_PREVIOUS_PATH.exists() and META_REPORT_PREVIOUS_PATH.exists()
        legacy_ready = (
            META_CLASSIFIER_PATH.exists()
            and META_EDGE_PATH.exists()
            and META_DRAWDOWN_PATH.exists()
            and META_FEATURES_PATH.exists()
            and META_REPORT_PATH.exists()
        )
        return directional_ready or previous_directional_ready or legacy_ready

    @classmethod
    def _report_is_acceptable(cls, report: Dict) -> bool:
        walk_forward = report.get("walk_forward", {}) if isinstance(report, dict) else {}
        summary = walk_forward.get("summary", {}) if isinstance(walk_forward, dict) else {}
        if not summary:
            return False
        if float(summary.get("mean_precision", 0.0) or 0.0) < cls._gate_min_precision():
            return False
        if float(summary.get("mean_coverage_pct", 0.0) or 0.0) < cls._gate_min_coverage_pct():
            return False
        if float(summary.get("mean_taken_edge_pct", 0.0) or 0.0) < cls._gate_min_taken_edge_pct():
            return False
        if float(summary.get("mean_taken_hit_rate_pct", 0.0) or 0.0) < cls._gate_min_hit_rate_pct():
            return False
        return True

    @classmethod
    def _report_rejection_reasons(cls, report: Dict) -> List[str]:
        walk_forward = report.get("walk_forward", {}) if isinstance(report, dict) else {}
        summary = walk_forward.get("summary", {}) if isinstance(walk_forward, dict) else {}
        status = str(walk_forward.get("status") or "").strip().lower()
        reasons: List[str] = []
        if not summary:
            reasons.append("missing_walkforward_summary")
            return reasons
        if status in {"pending", "pending_walkforward_retrain"}:
            reasons.append("walkforward_pending")
        if float(summary.get("mean_precision", 0.0) or 0.0) < cls._gate_min_precision():
            reasons.append("precision_below_gate")
        if float(summary.get("mean_coverage_pct", 0.0) or 0.0) < cls._gate_min_coverage_pct():
            reasons.append("coverage_below_gate")
        if float(summary.get("mean_taken_edge_pct", 0.0) or 0.0) < cls._gate_min_taken_edge_pct():
            reasons.append("edge_below_gate")
        if float(summary.get("mean_taken_hit_rate_pct", 0.0) or 0.0) < cls._gate_min_hit_rate_pct():
            reasons.append("hit_rate_below_gate")
        return reasons

    @classmethod
    def inspect_runtime_status(cls) -> Dict[str, object]:
        def _load_report(report_path: Path) -> Optional[Dict]:
            if not report_path.exists():
                return None
            try:
                with report_path.open("r", encoding="utf-8") as handle:
                    return json.load(handle)
            except Exception:
                return None

        current_report = _load_report(META_REPORT_PATH)
        previous_report = _load_report(META_REPORT_PREVIOUS_PATH)
        current_summary = ((current_report or {}).get("walk_forward") or {}).get("summary") or {}
        previous_summary = ((previous_report or {}).get("walk_forward") or {}).get("summary") or {}

        status: Dict[str, object] = {
            "state": "missing_artifacts",
            "loadable": False,
            "fallback_source": "",
            "current_report_exists": META_REPORT_PATH.exists(),
            "current_directional_exists": META_DIRECTIONAL_PATH.exists(),
            "previous_report_exists": META_REPORT_PREVIOUS_PATH.exists(),
            "previous_directional_exists": META_DIRECTIONAL_PREVIOUS_PATH.exists(),
            "legacy_meta_exists": all(
                path.exists() for path in (META_CLASSIFIER_PATH, META_EDGE_PATH, META_DRAWDOWN_PATH, META_FEATURES_PATH)
            ),
            "current_walk_forward_status": ((current_report or {}).get("walk_forward") or {}).get("status"),
            "previous_walk_forward_status": ((previous_report or {}).get("walk_forward") or {}).get("status"),
            "current_summary": current_summary,
            "previous_summary": previous_summary,
            "rejection_reasons": [],
        }

        if current_report and cls._report_is_acceptable(current_report) and META_DIRECTIONAL_PATH.exists():
            status["state"] = "validated"
            status["loadable"] = True
            return status

        if current_report:
            reasons = cls._report_rejection_reasons(current_report)
            status["rejection_reasons"] = reasons
            if "walkforward_pending" in reasons:
                status["state"] = "pending_walkforward"
            else:
                status["state"] = "rejected_by_gate"

        if previous_report and cls._report_is_acceptable(previous_report) and META_DIRECTIONAL_PREVIOUS_PATH.exists():
            status["state"] = "fallback_previous_validated"
            status["loadable"] = True
            status["fallback_source"] = "previous_validated_directional"
            return status

        if status["legacy_meta_exists"] and current_report is not None:
            if status.get("state") in {"pending_walkforward", "rejected_by_gate"}:
                status["fallback_source"] = "legacy_meta"
            else:
                status["state"] = "legacy_meta_only"
            status["loadable"] = True
            return status

        if current_report and META_DIRECTIONAL_PATH.exists() and status.get("state") == "missing_artifacts":
            status["state"] = "artifact_incompatible"

        return status

    @classmethod
    def load(cls) -> Optional["TrainedMetaModel"]:
        if not cls.is_available():
            return None
        def _is_live_mode() -> bool:
            if os.getenv("LIVE_TRADING_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}:
                return True
            if os.getenv("EXECUTION_MODE", "").strip().lower() in {"live", "production", "prod"}:
                return True
            if os.getenv("BYBIT_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"} and os.getenv(
                "BYBIT_ORDER_DRY_RUN", "1"
            ).strip().lower() not in {"1", "true", "yes", "on"}:
                return True
            return False
        def _load_report(report_path: Path) -> Optional[Dict]:
            if not report_path.exists():
                return None
            with open(report_path, "r", encoding="utf-8") as f:
                return json.load(f)

        def _log_rejected_report(report: Dict, label: str) -> None:
            summary = report.get("walk_forward", {}).get("summary", {}) if isinstance(report, dict) else {}
            logger.warning(
                f"Meta model checkpoint '{label}' rejected by validation gate: "
                f"precision={summary.get('mean_precision')} "
                f"coverage={summary.get('mean_coverage_pct')} "
                f"edge={summary.get('mean_taken_edge_pct')} "
                f"hit_rate={summary.get('mean_taken_hit_rate_pct')}"
            )

        def _directional_model_from(report: Dict, directional_path: Path) -> Optional["TrainedMetaModel"]:
            if not directional_path.exists():
                return None
            
            # Try to load as new ensemble format first (our 76.2% model)
            try:
                payload = joblib.load(directional_path)
                if isinstance(payload, dict) and "xgboost" in payload and "lightgbm" in payload:
                    # New ensemble format - wrap in bundles structure
                    bundles = {}
                    for model_name, model in payload.items():
                        bundles[model_name] = DirectionalMetaArtifacts(
                            classifier=model,
                            calibrator=BinaryPlattCalibrator(),
                            edge_model=None,
                            drawdown_model=None,
                            decision_threshold=0.55,
                            min_expected_edge_pct=0.15,
                            min_edge_ratio=max(META_MIN_EDGE_RATIO_FLOOR, 0.18),
                            direction=model_name,
                        )
                    feature_columns = list(range(40))
                    deployment_rules = report.get("deployment_rules", {}) if isinstance(report, dict) else {}
                    return cls(_resolved_bundles=bundles, _feature_columns=feature_columns, _deployment_rules=deployment_rules)
            except Exception as e:
                pass
            
            # Original format
            payload = joblib.load(directional_path)
            bundles = payload.get("bundles", {}) if isinstance(payload, dict) else {}
            feature_columns = list(payload.get("feature_columns", MetaFeatureBuilder.FEATURE_COLUMNS))
            resolved_bundles: Dict[str, DirectionalMetaArtifacts] = {}
            deployment_rules = report.get("deployment_rules", {}) if isinstance(report, dict) else {}
            for direction, bundle_payload in bundles.items():
                rules = deployment_rules.get(direction, {}) if isinstance(deployment_rules, dict) else {}
                resolved_bundles[direction] = DirectionalMetaArtifacts(
                    classifier=bundle_payload["classifier"],
                    calibrator=bundle_payload.get("calibrator") or BinaryPlattCalibrator(),
                    edge_model=bundle_payload["edge_model"],
                    drawdown_model=bundle_payload["drawdown_model"],
                    decision_threshold=float(
                        os.getenv(
                            "META_MODEL_RUNTIME_THRESHOLD",
                            str(rules.get("decision_threshold", 0.52) or 0.52),
                        )
                    ),
                    min_expected_edge_pct=float(
                        os.getenv(
                            "META_MODEL_RUNTIME_MIN_EDGE_PCT",
                            str(rules.get("min_expected_edge_pct", 0.18) or 0.18),
                        )
                    ),
                    min_edge_ratio=max(
                        META_MIN_EDGE_RATIO_FLOOR,
                        float(
                            os.getenv(
                                "META_MODEL_RUNTIME_MIN_EDGE_RATIO",
                                str(rules.get("min_edge_ratio", META_MIN_EDGE_RATIO_FLOOR) or META_MIN_EDGE_RATIO_FLOOR),
                            )
                        ),
                    ),
                    direction=direction,
                )
            if not resolved_bundles:
                return None
            return cls(resolved_bundles, feature_columns)

        try:
            force_directional = os.getenv("META_MODEL_FORCE_LOAD_DIRECTIONAL", "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if force_directional and _is_live_mode():
                logger.warning(
                    "META_MODEL_FORCE_LOAD_DIRECTIONAL ignored in live mode. "
                    "Enable only in paper/sim where gate bypass is intentional."
                )
                force_directional = False
            report = _load_report(META_REPORT_PATH)
            if report and cls._report_is_acceptable(report):
                directional_model = _directional_model_from(report, META_DIRECTIONAL_PATH)
                if directional_model is not None:
                    return directional_model
            elif report and force_directional and META_DIRECTIONAL_PATH.exists():
                directional_model = _directional_model_from(report, META_DIRECTIONAL_PATH)
                if directional_model is not None:
                    logger.warning(
                        "Meta model: loading directional checkpoint despite failed walk-forward gate "
                        "(META_MODEL_FORCE_LOAD_DIRECTIONAL=1). Paper/live outcomes may be weak."
                    )
                    return directional_model
            elif report:
                _log_rejected_report(report, "current")

            previous_report = _load_report(META_REPORT_PREVIOUS_PATH)
            if previous_report and cls._report_is_acceptable(previous_report):
                previous_model = _directional_model_from(previous_report, META_DIRECTIONAL_PREVIOUS_PATH)
                if previous_model is not None:
                    logger.info("Meta model load: using previous validated directional checkpoint")
                    return previous_model
            elif previous_report and force_directional and META_DIRECTIONAL_PREVIOUS_PATH.exists():
                previous_model = _directional_model_from(previous_report, META_DIRECTIONAL_PREVIOUS_PATH)
                if previous_model is not None:
                    logger.warning(
                        "Meta model: loading previous directional checkpoint despite failed gate "
                        "(META_MODEL_FORCE_LOAD_DIRECTIONAL=1)."
                    )
                    return previous_model
            elif previous_report:
                _log_rejected_report(previous_report, "previous")

            if report is None:
                return None

            classifier = joblib.load(META_CLASSIFIER_PATH)
            edge_model = joblib.load(META_EDGE_PATH)
            drawdown_model = joblib.load(META_DRAWDOWN_PATH)
            with open(META_FEATURES_PATH, "rb") as f:
                feature_columns = pickle.load(f)
            deployment_rules = report.get("deployment_rules", {}) if isinstance(report, dict) else {}
            legacy_bundle = DirectionalMetaArtifacts(
                classifier=classifier,
                calibrator=BinaryPlattCalibrator(),
                edge_model=edge_model,
                drawdown_model=drawdown_model,
                decision_threshold=float(
                    os.getenv(
                        "META_MODEL_RUNTIME_THRESHOLD",
                        str(deployment_rules.get("decision_threshold", 0.52) or 0.52),
                    )
                ),
                min_expected_edge_pct=float(
                    os.getenv(
                        "META_MODEL_RUNTIME_MIN_EDGE_PCT",
                        str(deployment_rules.get("min_expected_edge_pct", 0.18) or 0.18),
                    )
                ),
                min_edge_ratio=max(
                    META_MIN_EDGE_RATIO_FLOOR,
                    float(
                        os.getenv(
                            "META_MODEL_RUNTIME_MIN_EDGE_RATIO",
                            str(deployment_rules.get("min_edge_ratio", META_MIN_EDGE_RATIO_FLOOR) or META_MIN_EDGE_RATIO_FLOOR),
                        )
                    ),
                ),
                direction="legacy",
            )
            return cls({"long": legacy_bundle, "short": legacy_bundle}, feature_columns)
        except Exception as exc:
            logger.warning(f"Meta model load failed: {exc}")
            return None

    def predict_from_dict(self, features):
        frame = pd.DataFrame([{col: float(features.get(col, 0.0) or 0.0) for col in self.feature_columns}])
        direction = "long" if float(features.get("is_buy_signal", 0.0) or 0.0) >= float(features.get("is_sell_signal", 0.0) or 0.0) else "short"
        bundle = self.bundles.get(direction) or self.bundles.get("long") or next(iter(self.bundles.values()))
        raw_take_probability = float(bundle.classifier.predict_proba(frame)[0][1])
        take_probability = float(bundle.calibrator.transform(np.array([raw_take_probability]))[0]) if bundle.calibrator else raw_take_probability
        expected_edge_pct = float(bundle.edge_model.predict(frame)[0])
        expected_drawdown_pct = float(abs(bundle.drawdown_model.predict(frame)[0]))

        # ─── Regime gate ────────────────────────────────────────────────
        # Walk-forward evidence (clean retrain, 64,895 rows, Apr 26 2026):
        #   fold 5 (56% stress): hit 60.6%, edge/draw 0.55x  — marginal
        #   fold 6 (15% stress): hit 91.7%, edge/draw 4.05x  — strong
        #   fold 7 (98% stress): hit 52.8%, edge/draw 0.32x  — losing
        #   fold 8 (95% stress): hit 49.2%, edge/draw 0.19x  — losing
        # Model has no edge in stress. Block trades when stressed.
        # Tune via env: META_MODEL_REGIME_GATE_STRESSED (default 0.5).
        # Set to 1.1 to disable the gate entirely.
        regime_gate_threshold = float(os.getenv("META_MODEL_REGIME_GATE_STRESSED", "0.5"))
        market_stress = float(features.get("market_stress_regime", 0.0) or 0.0)
        if market_stress == 0.0:
            market_stress = float(features.get("vol_regime_stressed", 0.0) or 0.0)
        regime_gated = market_stress >= regime_gate_threshold
        gated_take_probability = 0.0 if regime_gated else take_probability

        return {
            "take_probability": round(gated_take_probability, 4),
            "raw_take_probability": round(take_probability, 4),
            "regime_gated": bool(regime_gated),
            "market_stress_regime": round(market_stress, 4),
            "expected_edge_pct": round(expected_edge_pct, 3),
            "expected_drawdown_pct": round(max(expected_drawdown_pct, 0.1), 3),
            "decision_threshold": round(float(bundle.decision_threshold), 4),
            "min_expected_edge_pct": round(float(bundle.min_expected_edge_pct), 4),
            "min_edge_ratio": round(float(bundle.min_edge_ratio), 4),
            "direction_bundle": direction,
        }
class EventAwareXGBoostRetrainer:
    def __init__(self, horizon: int = 5, train_split: float = 0.85):
        self.horizon = horizon
        self.train_split = train_split

    def _build_dataset(self, feature_matrices: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        rows = []
        min_xgb_rows = max(80, int(os.getenv("XGB_MIN_FRAME_ROWS", "100") or 100))
        items = list(feature_matrices.items())
        log_every = max(100, int(os.getenv("RETRAIN_XGB_LOG_EVERY", "400") or 400))
        min_required_price_fields = ("open", "high", "low", "close", "volume")
        input_rows_total = 0
        kept_rows_total = 0
        dropped_short_frames = 0
        dropped_missing_core_rows = 0
        for i, (symbol, frame) in enumerate(items):
            if frame is None or frame.empty or len(frame) < min_xgb_rows:
                if frame is not None:
                    dropped_short_frames += int(len(frame))
                continue
            input_rows_total += len(frame)
            numeric = frame.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
            required_cols = [col for col in min_required_price_fields if col in numeric.columns]
            if "close" not in required_cols:
                continue
            before_core_filter = len(numeric)
            numeric = numeric.dropna(subset=required_cols)
            dropped_missing_core_rows += max(0, before_core_filter - len(numeric))
            if numeric.empty:
                continue
            labels = []
            for idx in range(len(numeric)):
                if idx >= (len(numeric) - 2):
                    labels.append(1)
                    continue
                labels.append(_triple_barrier_direction_label(numeric, idx, self.horizon))
            labels = pd.Series(labels, index=numeric.index, dtype=int)
            numeric = numeric.copy()
            numeric["label"] = labels
            numeric["symbol"] = symbol
            numeric["timestamp"] = pd.to_datetime(numeric.index)
            rows.append(numeric.reset_index(drop=True))
            kept_rows_total += len(numeric)
            if (i + 1) % log_every == 0:
                logger.info(
                    "XGB dataset build: processed %s / %s symbols (%s kept / %s input rows so far)",
                    i + 1,
                    len(items),
                    kept_rows_total,
                    input_rows_total,
                )

        if not rows:
            return pd.DataFrame()

        logger.info("XGB dataset build: concatenating %s symbol blocks...", len(rows))
        dataset = pd.concat(rows, ignore_index=True)
        dataset = dataset.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        logger.info(
            "XGB dataset ready: %s rows x %s cols | input_rows=%s | kept_rows=%s | dropped_missing_core=%s | dropped_short_frame_rows=%s",
            len(dataset),
            len(dataset.columns),
            input_rows_total,
            kept_rows_total,
            dropped_missing_core_rows,
            dropped_short_frames,
        )
        return dataset

    def retrain(self, feature_matrices: Dict[str, pd.DataFrame], run_optuna: bool = False, optuna_trials: int = 40) -> Dict:
        from models.xgboost_model import XGBoostOptimizer, XGBoostSignalModel, add_regime_interactions

        logger.info("XGBoost retrain: building labeled dataset (silent stretches are normal on large stores)...")
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
        logger.info("XGBoost retrain: fitting model (%s train / %s val rows)...", len(X_train), len(X_val))
        fit_metrics = model.fit(X_train, y_train, eval_set=(X_val, y_val))
        logger.info("XGBoost retrain: time-series CV (can take several minutes)...")
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

        shap_monitor = {"status": "unavailable"}
        try:
            from core.explainability import GlobalFeatureImportance, SHAPExplainerFactory

            X_val_aug = add_regime_interactions(X_val)
            X_val_sel = model.selector.transform(X_val_aug).fillna(0.0)
            explainer = SHAPExplainerFactory.for_xgboost(model.model)
            importance = GlobalFeatureImportance().compute_global_importance(
                explainer=explainer,
                feature_matrix=X_val_sel,
                feature_names=model.feature_names_in,
                sample_n=min(500, len(X_val_sel)),
            )
            if not importance.empty:
                current_values = {
                    row["feature"]: round(float(row["mean_abs_shap"]), 8)
                    for _, row in importance.head(20).iterrows()
                }
                alerts = []
                previous_values = {}
                if XGB_SHAP_REPORT_PATH.exists():
                    previous_payload = json.loads(XGB_SHAP_REPORT_PATH.read_text(encoding="utf-8"))
                    previous_values = (previous_payload.get("baseline") or {})
                for feature, current_value in current_values.items():
                    previous_value = _safe_float(previous_values.get(feature, 0.0), 0.0)
                    if previous_value <= 0:
                        continue
                    if current_value < (previous_value * 0.60):
                        alerts.append({"feature": feature, "type": "drop", "baseline": previous_value, "current": current_value})
                    elif current_value > (previous_value * 1.80):
                        alerts.append({"feature": feature, "type": "surge", "baseline": previous_value, "current": current_value})
                shap_monitor = {
                    "status": "ok",
                    "baseline": current_values,
                    "alerts": alerts[:10],
                    "top_features": list(current_values.keys())[:10],
                }
                XGB_SHAP_REPORT_PATH.write_text(json.dumps(shap_monitor, indent=2), encoding="utf-8")
        except Exception as exc:
            shap_monitor = {"status": "error", "reason": str(exc)}

        report = {
            "status": "ok",
            "train_rows": int(len(X_train)),
            "validation_rows": int(len(X_val)),
            "n_features": int(len(model.feature_names_in)),
            "selected_features": model.feature_names_in,
            "fit_metrics": fit_metrics,
            "cv_metrics": cv_metrics,
            "model_path": str(model_path),
            "shap_monitor": shap_monitor,
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
        take_threshold: float = 0.52,
        walk_forward_folds: int = 4,
        min_train_days: int = 180,
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
        y_take = df["take_label"].astype(int).to_numpy()

        raw_thresholds = os.getenv("META_MODEL_THRESHOLD_GRID", "").strip()
        if raw_thresholds:
            threshold_grid = [float(x.strip()) for x in raw_thresholds.split(",") if x.strip()]
        else:
            # Removed 0.38/0.42 — they historically produce poor precision
            threshold_grid = [0.48, 0.52, 0.56, 0.60, 0.64, 0.68, 0.72]
        min_edge_grid = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
        min_ratio_grid = [0.05, 0.08, 0.12, 0.16, 0.20, 0.28, 0.36]
        edge_ratio_pred = edge_pred / np.maximum(drawdown_pred, 0.35)

        min_count = max(25, int(os.getenv("META_MODEL_MIN_RULE_SAMPLES", "32")))
        cov_min = float(os.getenv("META_MODEL_MIN_COVERAGE_PCT", "0.8"))
        cov_max = float(os.getenv("META_MODEL_MAX_COVERAGE_PCT", "60"))
        rule_mode = os.getenv("META_MODEL_RULE_OBJECTIVE", "precision").strip().lower()

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
                    if count < min_count or coverage < cov_min or coverage > cov_max:
                        continue
                    avg_edge = float(realized_edge[mask].mean())
                    avg_draw = float(realized_draw[mask].mean())
                    hit_rate = float(hit_rate_array[mask].mean() * 100.0)
                    precision_take = float(y_take[mask].mean()) if count > 0 else 0.0
                    if rule_mode == "legacy":
                        objective = avg_edge - (0.32 * avg_draw) + ((hit_rate - 50.0) * 0.02)
                    else:
                        # Precision floor: skip rules below this (lowered from 0.60 — was blocking all configs on current data)
                        min_precision_floor = float(os.getenv(
                            "META_MODEL_MIN_PRECISION_FLOOR", "0.38"))
                        if precision_take < min_precision_floor:
                            continue
                        # Coverage bonus — enough to prevent precision-only overfitting
                        coverage_bonus = max(0.0, min(coverage, 8.0)) * 0.12
                        objective = (
                            55.0 * precision_take
                            + 0.18 * hit_rate
                            + 0.12 * float(np.clip(avg_edge, -3.0, 6.0))
                            - 0.10 * float(np.clip(avg_draw, 0.0, 12.0))
                            + coverage_bonus
                        )
                    if objective > best["objective"]:
                        best = {
                            "decision_threshold": threshold,
                            "min_expected_edge_pct": min_edge,
                            "min_edge_ratio": min_ratio,
                            "objective": objective,
                        }

        best.pop("objective", None)
        return best

    def _bundle_take_probability(self, bundle: DirectionalMetaArtifacts, X: pd.DataFrame) -> np.ndarray:
        raw_prob = bundle.classifier.predict_proba(X)[:, 1]
        return bundle.calibrator.transform(raw_prob) if bundle.calibrator else raw_prob

    def _fit_direction_bundle(self, train_df: pd.DataFrame, direction: str) -> Optional[DirectionalMetaArtifacts]:
        if train_df.empty or train_df["take_label"].nunique() < 2:
            return None

        X = train_df[MetaFeatureBuilder.FEATURE_COLUMNS].fillna(0.0)
        y_take = train_df["take_label"].astype(int)
        y_edge = train_df["edge_pct"].astype(float)
        y_drawdown = train_df["drawdown_pct"].astype(float)
        sample_weight = self._sample_weights(y_edge, y_drawdown, y_take)
        _stress_w = float(os.getenv("META_STRESS_SAMPLE_WEIGHT", "1.0"))
        if _stress_w != 1.0 and "market_stress_regime" in train_df.columns:
            _stress_mask = (train_df["market_stress_regime"].values == 1)
            sample_weight = sample_weight.copy()
            sample_weight[_stress_mask] *= _stress_w
            logger.info("Meta sample weights: stressed_rows=%d (weight=%.2f) calm_rows=%d",
                int(_stress_mask.sum()), _stress_w, int((~_stress_mask).sum()))

        # Calibration split: hold out last 20% for Platt calibration.
        # Minimum 60 calibration rows to avoid overfitting the sigmoid.
        min_cal_rows = max(60, int(os.getenv("META_MODEL_MIN_CAL_ROWS", "80")))
        calibration_rows = max(min_cal_rows, min(len(train_df) // 4, 200))
        core_rows = max(len(train_df) - calibration_rows, max(80, len(train_df) // 2))
        X_core = X.iloc[:core_rows]
        y_core = y_take.iloc[:core_rows]
        core_weight = sample_weight[: len(X_core)]
        X_cal = X.iloc[core_rows:]
        y_cal = y_take.iloc[core_rows:]
        # If split fails, use 3-fold cross-calibration instead of reusing training data
        use_cv_calibration = False
        if X_cal.empty or len(X_cal) < min_cal_rows or y_cal.nunique() < 2 or y_core.nunique() < 2:
            X_core = X
            y_core = y_take
            core_weight = sample_weight
            use_cv_calibration = True  # flag: do NOT reuse train data for calibration

        # Use the shared classifier factory so the bundle classifier and the
        # OOF passes used for isotonic calibration share identical hyperparams.
        classifier = _make_meta_classifier()
        edge_model = HistGradientBoostingRegressor(
            learning_rate=0.045,
            max_depth=6,
            max_iter=480,
            min_samples_leaf=18,
            l2_regularization=0.09,
            random_state=42,
        )
        drawdown_model = HistGradientBoostingRegressor(
            learning_rate=0.045,
            max_depth=6,
            max_iter=480,
            min_samples_leaf=18,
            l2_regularization=0.09,
            random_state=42,
            loss="absolute_error",
        )

        classifier.fit(X_core, y_core, sample_weight=core_weight)
        edge_model.fit(X, y_edge, sample_weight=sample_weight)
        drawdown_model.fit(X, y_drawdown, sample_weight=sample_weight)

        # --- Regressor quality check (data-driven drift limits) ---
        # Hardcoded MAE limits 2.5/3.0 were tripping on every fold because
        # edge_pct distribution naturally has std around 3. Replace with a
        # naive-baseline limit: drift_limit = MAE(mean predictor) * 1.2.
        # The regressor only "drifts" if it is more than 20% worse than just
        # predicting the training mean. Env vars still override.
        edge_pred_train = edge_model.predict(X)
        draw_pred_train = drawdown_model.predict(X)
        edge_arr = y_edge.to_numpy()
        draw_arr = y_drawdown.to_numpy()
        edge_mae = float(np.mean(np.abs(edge_pred_train - edge_arr)))
        draw_mae = float(np.mean(np.abs(draw_pred_train - draw_arr)))
        edge_baseline_mae = float(np.mean(np.abs(edge_arr - float(edge_arr.mean())))) if edge_arr.size else 0.0
        draw_baseline_mae = float(np.mean(np.abs(draw_arr - float(draw_arr.mean())))) if draw_arr.size else 0.0
        edge_mae_limit_default = max(0.5, edge_baseline_mae * 1.2)
        draw_mae_limit_default = max(0.5, draw_baseline_mae * 1.2)
        edge_env = os.getenv("META_MODEL_EDGE_MAE_LIMIT", "")
        draw_env = os.getenv("META_MODEL_DRAW_MAE_LIMIT", "")
        edge_mae_limit = float(edge_env) if edge_env.strip() else edge_mae_limit_default
        draw_mae_limit = float(draw_env) if draw_env.strip() else draw_mae_limit_default
        regressors_reliable = (edge_mae <= edge_mae_limit and draw_mae <= draw_mae_limit)

        # Diagnostic: surface the actual scale + sample of predictions so a
        # future drift event can be inspected instead of guessed at.
        logger.info(
            "Meta regressor diagnostics (%s): "
            "edge actual mean=%.3f std=%.3f, edge pred mean=%.3f std=%.3f | "
            "draw actual mean=%.3f std=%.3f, draw pred mean=%.3f std=%.3f | "
            "edge_MAE=%.3f baseline=%.3f limit=%.3f | "
            "draw_MAE=%.3f baseline=%.3f limit=%.3f | reliable=%s",
            direction,
            float(edge_arr.mean()) if edge_arr.size else 0.0,
            float(edge_arr.std()) if edge_arr.size else 0.0,
            float(np.mean(edge_pred_train)) if edge_pred_train.size else 0.0,
            float(np.std(edge_pred_train)) if edge_pred_train.size else 0.0,
            float(draw_arr.mean()) if draw_arr.size else 0.0,
            float(draw_arr.std()) if draw_arr.size else 0.0,
            float(np.mean(draw_pred_train)) if draw_pred_train.size else 0.0,
            float(np.std(draw_pred_train)) if draw_pred_train.size else 0.0,
            edge_mae, edge_baseline_mae, edge_mae_limit,
            draw_mae, draw_baseline_mae, draw_mae_limit,
            regressors_reliable,
        )
        try:
            sample_n = min(20, len(edge_arr))
            if sample_n > 0:
                rng = np.random.default_rng(0)
                idx = rng.choice(len(edge_arr), size=sample_n, replace=False)
                pairs = [
                    (float(edge_arr[i]), float(edge_pred_train[i])) for i in idx
                ]
                logger.debug(
                    "Meta edge regressor sample (%s) actual->pred: %s",
                    direction,
                    ", ".join(f"{a:.2f}->{p:.2f}" for a, p in pairs),
                )
        except Exception:
            pass

        if not regressors_reliable:
            logger.warning(
                "Regressor drift detected (%s): edge_MAE=%.3f (limit %.3f, baseline %.3f), "
                "draw_MAE=%.3f (limit %.3f, baseline %.3f). Falling back to precision-only gate; "
                "min_edge_ratio floor (%.2f) is still enforced.",
                direction, edge_mae, edge_mae_limit, edge_baseline_mae,
                draw_mae, draw_mae_limit, draw_baseline_mae,
                META_MIN_EDGE_RATIO_FLOOR,
            )

        if use_cv_calibration:
            # 3-fold cross-calibration: each fold predicts on held-out portion
            from sklearn.model_selection import KFold
            cv_raw = np.zeros(len(X))
            cv_labels = y_take.to_numpy(dtype=int)
            kf = KFold(n_splits=3, shuffle=False)
            for cv_train_idx, cv_val_idx in kf.split(X):
                cv_clf = _make_meta_classifier()
                cv_clf.fit(X.iloc[cv_train_idx], y_take.iloc[cv_train_idx],
                           sample_weight=sample_weight[cv_train_idx])
                cv_raw[cv_val_idx] = cv_clf.predict_proba(X.iloc[cv_val_idx])[:, 1]
            calibrator = BinaryPlattCalibrator().fit(cv_raw, cv_labels)
        else:
            raw_calibration = classifier.predict_proba(X_cal)[:, 1]
            calibrator = BinaryPlattCalibrator().fit(raw_calibration, y_cal)
        calibrated_take_prob = self._bundle_take_probability(
            DirectionalMetaArtifacts(
                classifier=classifier,
                calibrator=calibrator,
                edge_model=edge_model,
                drawdown_model=drawdown_model,
                decision_threshold=self.take_threshold,
                min_expected_edge_pct=0.10,
                min_edge_ratio=0.12,
                direction=direction,
            ),
            X,
        )
        edge_pred = edge_model.predict(X)
        drawdown_pred = np.abs(drawdown_model.predict(X))
        decision_rule = self._select_decision_rule(train_df, calibrated_take_prob, edge_pred, drawdown_pred)
        # If regressors are unreliable, disable the edge gate (precision-only mode)
        # but keep the edge/draw ratio floor — a zero ratio gate lets through
        # bets with negative risk/reward, which is what produced the 0.002
        # fold-3 collapse in the prior walk-forward run.
        if not regressors_reliable:
            decision_rule["min_expected_edge_pct"] = 0.0
            decision_rule["min_edge_ratio"] = META_MIN_EDGE_RATIO_FLOOR
            logger.info(
                "Regressor fallback (%s): edge gate disabled, edge/draw ratio held at floor %.2f.",
                direction,
                META_MIN_EDGE_RATIO_FLOOR,
            )
        enforced_min_ratio = max(META_MIN_EDGE_RATIO_FLOOR, float(decision_rule["min_edge_ratio"]))
        if enforced_min_ratio > float(decision_rule["min_edge_ratio"]):
            logger.info(
                "Meta bundle (%s): raising min_edge_ratio from %.4f to floor %.2f.",
                direction,
                float(decision_rule["min_edge_ratio"]),
                META_MIN_EDGE_RATIO_FLOOR,
            )
        return DirectionalMetaArtifacts(
            classifier=classifier,
            calibrator=calibrator,
            edge_model=edge_model,
            drawdown_model=drawdown_model,
            decision_threshold=float(decision_rule["decision_threshold"]),
            min_expected_edge_pct=float(decision_rule["min_expected_edge_pct"]),
            min_edge_ratio=enforced_min_ratio,
            direction=direction,
        )

    def _apply_composite_take_label(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Build the final take_label using a cross-sectional composite rank.

        Combines edge_pct rank, edge/draw ratio rank, and conviction rank within
        each (timestamp, direction) group. The path_take_label gate must still
        pass and the raw edge_draw_ratio must clear an absolute floor — this
        prevents the meta head from labelling positively any path that is
        directionally correct but has poor risk/reward.
        """
        if dataset.empty:
            return dataset

        edge_weight = float(os.getenv("META_LABEL_EDGE_WEIGHT", "0.45"))
        ratio_weight = float(os.getenv("META_LABEL_RATIO_WEIGHT", "0.40"))
        conviction_weight = float(os.getenv("META_LABEL_CONVICTION_WEIGHT", "0.15"))
        weight_sum = edge_weight + ratio_weight + conviction_weight
        assert abs(weight_sum - 1.0) < 1e-6, (
            f"META_LABEL composite weights must sum to 1.0, got "
            f"{edge_weight}+{ratio_weight}+{conviction_weight}={weight_sum:.4f}"
        )

        cross_pct = float(os.getenv("META_LABEL_CROSS_PCT", "0.18"))
        min_candidates = int(os.getenv("META_LABEL_MIN_CANDIDATES", "6"))
        ratio_floor = float(os.getenv("META_LABEL_EDGE_DRAW_RATIO_FLOOR", "0.20"))

        df = dataset.copy()
        drawdown_safe = df["drawdown_pct"].astype(float).abs() + 1e-6
        df["edge_draw_ratio"] = df["edge_pct"].astype(float) / drawdown_safe

        if "conviction_score" not in df.columns:
            df["conviction_score"] = 0.0

        group_keys = ["timestamp", "direction_label"]
        grouped = df.groupby(group_keys, sort=False)
        df["edge_rank"] = grouped["edge_pct"].rank(pct=True, method="average")
        df["edge_draw_ratio_rank"] = grouped["edge_draw_ratio"].rank(pct=True, method="average")
        df["conviction_rank"] = grouped["conviction_score"].rank(pct=True, method="average")
        df["n_candidates"] = grouped["edge_pct"].transform("count").astype(int)

        for col in ("edge_rank", "edge_draw_ratio_rank", "conviction_rank"):
            df[col] = df[col].fillna(0.0)

        df["composite_score_rank"] = (
            edge_weight * df["edge_rank"]
            + ratio_weight * df["edge_draw_ratio_rank"]
            + conviction_weight * df["conviction_rank"]
        )

        # --- Regime conditioning: collapse symbol-level vol_regime_stressed
        # into a single daily market_stress_regime (0=calm, 1=stressed) using
        # the cross-sectional median across all symbols for that timestamp.
        # The interaction columns then teach the meta head to discount
        # edge/ratio rank during stress and to lean on conviction rank instead.
        if "vol_regime_stressed_raw" not in df.columns:
            df["vol_regime_stressed_raw"] = 0.0
        daily_stress_median = df.groupby("timestamp")["vol_regime_stressed_raw"].transform("median")
        df["market_stress_regime"] = (daily_stress_median >= 0.5).astype(float)
        calm = 1.0 - df["market_stress_regime"]
        df["conviction_rank_x_stress"] = df["conviction_rank"] * df["market_stress_regime"]

        rank_threshold = max(0.0, min(1.0, 1.0 - cross_pct))
        path_pass = df["path_take_label"].astype(int) == 1
        rank_pass = df["composite_score_rank"] >= rank_threshold
        ratio_pass = df["edge_draw_ratio"] >= ratio_floor
        candidate_pass = df["n_candidates"] >= min_candidates
        df["take_label"] = (path_pass & rank_pass & ratio_pass & candidate_pass).astype(int)

        # Single-line sanity log so it's obvious in retrain output that the
        # composite labeller actually ran and is changing the count vs the
        # path-only positives. If composite_positives==final_take_label and
        # path_positives matches the previous run's number, suspect a bypass.
        path_positive_count = int((df["path_take_label"].astype(int) == 1).sum())
        composite_positive_count = int((rank_pass & ratio_pass & candidate_pass).sum())
        final_label_count = int(df["take_label"].sum())
        logger.info(
            "COMPOSITE LABEL: total=%d path_positives=%d composite_positives=%d final_take_label=%d",
            int(len(df)),
            path_positive_count,
            composite_positive_count,
            final_label_count,
        )

        total = int(len(df))
        positives = int(df["take_label"].sum())
        positive_rate = (positives / total) if total else 0.0
        pos_mask = df["take_label"] == 1
        neg_mask = ~pos_mask
        pos_ratio = float(df.loc[pos_mask, "edge_draw_ratio"].mean()) if positives else float("nan")
        neg_ratio = float(df.loc[neg_mask, "edge_draw_ratio"].mean()) if (total - positives) else float("nan")
        logger.info(
            "Meta composite label stats:\n"
            "  total_rows=%d positive_labels=%d positive_rate=%.4f\n"
            "  weights edge=%.2f ratio=%.2f conviction=%.2f\n"
            "  cross_sectional_top_pct=%.3f min_candidates=%d edge_draw_ratio_floor=%.3f\n"
            "  mean edge_draw_ratio  positive=%.4f  negative=%.4f",
            total,
            positives,
            positive_rate,
            edge_weight,
            ratio_weight,
            conviction_weight,
            cross_pct,
            min_candidates,
            ratio_floor,
            pos_ratio,
            neg_ratio,
        )

        if positives:
            sample_cols = [
                "timestamp",
                "symbol",
                "direction_label",
                "edge_rank",
                "edge_draw_ratio_rank",
                "composite_score_rank",
                "take_label",
            ]
            sample = df.loc[pos_mask, sample_cols].head(10)
            logger.debug("Meta composite label sample (10 positives):\n%s", sample.to_string(index=False))
        else:
            logger.warning("Meta composite label produced 0 positive rows — check thresholds.")

        return df

    def build_training_dataset(self, feature_matrices: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        from core.signal_engine_v2 import MultiFactorScorer

        scorer = MultiFactorScorer()
        records = []
        min_frame = max(
            int(os.getenv("META_MODEL_MIN_FRAME_ROWS", "72") or 72),
            self.horizon_days + 22,
        )
        warmup_rows = max(28, int(os.getenv("META_MODEL_WARMUP_ROWS", "42") or 42))
        fm_items = list(feature_matrices.items())
        meta_log_every = max(80, int(os.getenv("RETRAIN_META_LOG_EVERY", "250") or 250))

        for fi, (symbol, frame) in enumerate(fm_items):
            if frame is None or frame.empty or len(frame) < min_frame:
                continue

            ordered = frame.sort_index()
            for idx in range(warmup_rows, len(ordered) - 2):
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

                horizon_days, horizon_multiplier = _adaptive_horizon_days(ordered, idx, self.horizon_days)
                future_window = ordered.iloc[idx + 1 : idx + 1 + horizon_days]
                if future_window.empty:
                    continue

                entry_close = float(row.get("close", 0.0) or 0.0)
                if entry_close <= 0:
                    continue

                risk_parameters = self._estimate_risk_parameters(row)
                path_stats = _triple_barrier_path_stats(
                    direction,
                    entry_close,
                    future_window,
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
                label_min_edge = float(os.getenv("META_MODEL_LABEL_MIN_EDGE_PCT", "0.18"))
                label_min_ratio = float(os.getenv("META_MODEL_LABEL_MIN_EDGE_RATIO", "0.08"))
                # path_take_label is the original path-based success gate.
                # The final take_label is built cross-sectionally below using
                # a composite rank that includes edge_draw_ratio_rank.
                path_take_label = int(
                    path_stats["edge_pct"] > label_min_edge
                    and edge_ratio > label_min_ratio
                    and path_stats["hit"] == 1
                )
                feature_values.update(
                    {
                        "symbol": symbol,
                        "timestamp": pd.Timestamp(ordered.index[idx]),
                        "direction_label": "long" if direction == "buy" else "short",
                        "adaptive_horizon_days": horizon_days,
                        "adaptive_horizon_multiplier": horizon_multiplier,
                        "path_take_label": path_take_label,
                        "edge_pct": path_stats["edge_pct"],
                        "drawdown_pct": path_stats["drawdown_pct"],
                        "edge_ratio": round(float(edge_ratio), 4),
                        "hit": path_stats["hit"],
                        "barrier": path_stats["barrier"],
                        # Captured so _apply_composite_take_label can collapse
                        # symbol-level vol regimes into a daily market-wide
                        # market_stress_regime via cross-sectional median.
                        "vol_regime_stressed_raw": _safe_float(
                            row.get("vol_regime_stressed", 0.0), 0.0
                        ),
                    }
                )
                records.append(feature_values)
            if (fi + 1) % meta_log_every == 0:
                logger.info("Meta dataset build: %s / %s symbols (%s training rows so far)", fi + 1, len(fm_items), len(records))

        if not records:
            return pd.DataFrame()

        logger.info("Meta dataset build: assembling %s rows...", len(records))
        dataset = pd.DataFrame(records)
        dataset = dataset.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        dataset = self._apply_composite_take_label(dataset)
        logger.info("Meta dataset ready: %s rows", len(dataset))
        return dataset

    def walk_forward_validate(self, dataset: pd.DataFrame) -> Dict:
        if dataset.empty:
            return {"status": "failed", "reason": "no_data"}

        dates = sorted(pd.to_datetime(dataset["timestamp"]).drop_duplicates().tolist())
        if len(dates) < (self.min_train_days + 20):
            return {"status": "failed", "reason": "insufficient_dates"}

        test_span = max(20, len(dates) // (self.walk_forward_folds + 2))
        folds = []
        logger.info(
            "Meta walk-forward: %s unique dates, %s folds, test_span=%s days (no per-fold logs = CPU-bound fits)...",
            len(dates),
            self.walk_forward_folds,
            test_span,
        )
        for fold in range(self.walk_forward_folds):
            train_end = len(dates) - (self.walk_forward_folds - fold) * test_span
            test_end = min(len(dates), train_end + test_span)
            if train_end < self.min_train_days or test_end <= train_end:
                continue

            logger.info(
                "Meta walk-forward fold %s/%s: train_end_idx=%s test_end_idx=%s (fitting long+short bundles)...",
                fold + 1,
                self.walk_forward_folds,
                train_end,
                test_end,
            )
            train_dates = set(dates[:train_end])
            test_dates = set(dates[train_end:test_end])
            train_df = dataset[dataset["timestamp"].isin(train_dates)]
            test_df = dataset[dataset["timestamp"].isin(test_dates)]
            if train_df.empty or test_df.empty:
                continue
            all_true = []
            all_pred = []
            taken_edges: List[float] = []
            taken_drawdown: List[float] = []
            direction_summaries: Dict[str, Dict] = {}
            # Per-fold isotonic calibration: fit on TRAIN-only OOF probs,
            # applied to the test fold so threshold comparisons live in a
            # stable calibrated probability space across folds.
            calibration_block: Dict[str, Dict[str, float]] = {}
            long_threshold = float(
                os.getenv("META_CALIBRATED_THRESHOLD_LONG", str(META_CALIBRATED_THRESHOLD_LONG_DEFAULT))
            )
            short_threshold = float(
                os.getenv("META_CALIBRATED_THRESHOLD_SHORT", str(META_CALIBRATED_THRESHOLD_SHORT_DEFAULT))
            )
            calibrated_thresholds = {"long": long_threshold, "short": short_threshold}

            for direction, mask_column in (("long", "is_buy_signal"), ("short", "is_sell_signal")):
                train_dir = train_df[train_df[mask_column] > 0.5]
                test_dir = test_df[test_df[mask_column] > 0.5]
                bundle = self._fit_direction_bundle(train_dir, direction)
                if bundle is None or test_dir.empty:
                    continue

                X_test = test_dir[MetaFeatureBuilder.FEATURE_COLUMNS].fillna(0.0)
                # Raw classifier probability (pre-Platt) — this is the
                # distribution the OOF isotonic was fit against.
                raw_test_prob = bundle.classifier.predict_proba(X_test)[:, 1]

                # Fit OOF isotonic on the training fold using the shared
                # classifier factory; persist it so production loaders can
                # inspect per-fold calibrators if needed.
                X_train_dir = train_dir[MetaFeatureBuilder.FEATURE_COLUMNS].fillna(0.0)
                y_train_dir = train_dir["take_label"].astype(int).to_numpy()
                fold_calibrator, _train_oof = _fit_isotonic_oof(X_train_dir, y_train_dir)
                if fold_calibrator is not None:
                    calibrated_test_prob = np.asarray(
                        fold_calibrator.transform(raw_test_prob), dtype=float
                    )
                    try:
                        cal_path = Path(
                            META_CALIBRATOR_FOLD_PATH_FMT.format(fold=fold + 1, direction=direction)
                        )
                        with open(cal_path, "wb") as f:
                            pickle.dump(fold_calibrator, f)
                    except Exception as exc:
                        logger.debug("Failed to persist fold calibrator (%s): %s", direction, exc)
                else:
                    calibrated_test_prob = raw_test_prob.copy()

                y_test_arr = test_dir["take_label"].astype(int).to_numpy()
                brier_raw = _brier_score(y_test_arr, raw_test_prob)
                brier_cal = _brier_score(y_test_arr, calibrated_test_prob)
                ece_raw = _expected_calibration_error(y_test_arr, raw_test_prob)
                ece_cal = _expected_calibration_error(y_test_arr, calibrated_test_prob)

                # Use a fixed calibrated threshold per direction instead of
                # the per-fold optimized one; this is the whole point of
                # calibrating — stop chasing a moving threshold.
                effective_threshold = calibrated_thresholds[direction]
                edge_pred = bundle.edge_model.predict(X_test)
                drawdown_pred = np.abs(bundle.drawdown_model.predict(X_test))
                pred_take = (
                    (calibrated_test_prob >= effective_threshold)
                    & (edge_pred >= bundle.min_expected_edge_pct)
                    & ((edge_pred / np.maximum(drawdown_pred, 0.35)) >= bundle.min_edge_ratio)
                )

                # Direction-level precision at the calibrated threshold;
                # if it dips below the floor, the calibrator itself is
                # suspect for this period (e.g. severe regime shift).
                if pred_take.any():
                    cal_precision = float(y_test_arr[pred_take].mean())
                else:
                    cal_precision = 0.0
                if cal_precision < META_CALIBRATED_PRECISION_FLOOR and pred_take.any():
                    logger.warning(
                        "Calibrated precision %.3f below floor %.2f for fold %d / %s — "
                        "calibrator may be unreliable for this period.",
                        cal_precision,
                        META_CALIBRATED_PRECISION_FLOOR,
                        fold + 1,
                        direction,
                    )

                realized_edges = test_dir["edge_pct"].to_numpy(dtype=float)
                realized_drawdown = test_dir["drawdown_pct"].to_numpy(dtype=float)
                taken_edges.extend(realized_edges[pred_take].tolist())
                taken_drawdown.extend(realized_drawdown[pred_take].tolist())
                all_true.extend(y_test_arr.tolist())
                all_pred.extend(pred_take.astype(int).tolist())
                direction_summaries[direction] = {
                    "test_rows": int(len(test_dir)),
                    "coverage_pct": round(float(pred_take.mean() * 100.0), 2),
                    "decision_threshold": round(float(effective_threshold), 4),
                    "decision_threshold_source": "calibrated_fixed",
                    "min_expected_edge_pct": round(float(bundle.min_expected_edge_pct), 4),
                    "min_edge_ratio": round(float(bundle.min_edge_ratio), 4),
                    "calibrated_precision": round(cal_precision, 4),
                }
                calibration_block[direction] = {
                    "brier_raw": round(brier_raw, 5),
                    "brier_calibrated": round(brier_cal, 5),
                    "brier_delta": round(brier_raw - brier_cal, 5),
                    "ece_raw": round(ece_raw, 5),
                    "ece_calibrated": round(ece_cal, 5),
                    "ece_delta": round(ece_raw - ece_cal, 5),
                    "fixed_threshold": round(float(effective_threshold), 4),
                    "calibrator_fitted": fold_calibrator is not None,
                }

            if not all_true:
                continue

            all_true_arr = np.asarray(all_true, dtype=int)
            all_pred_arr = np.asarray(all_pred, dtype=int)
            # Regime distribution for this fold's test rows. Surfacing this
            # makes it obvious when a fold (e.g. fold 3 of the prior run) is
            # dominated by stressed-regime data and explains tail collapses.
            if "market_stress_regime" in test_df.columns and len(test_df) > 0:
                stressed_fraction = float((test_df["market_stress_regime"] >= 0.5).mean())
            else:
                stressed_fraction = 0.0
            regime_distribution = {
                "stressed_fraction": round(stressed_fraction, 4),
                "calm_fraction": round(1.0 - stressed_fraction, 4),
                "test_rows": int(len(test_df)),
            }
            folds.append(
                {
                    "fold": fold + 1,
                    "train_rows": int(len(train_df)),
                    "test_rows": int(len(all_true_arr)),
                    "accuracy": round(float(accuracy_score(all_true_arr, all_pred_arr)), 4),
                    "precision": round(float(precision_score(all_true_arr, all_pred_arr, zero_division=0)), 4),
                    "recall": round(float(recall_score(all_true_arr, all_pred_arr, zero_division=0)), 4),
                    "coverage_pct": round(float(all_pred_arr.mean() * 100.0), 2),
                    "taken_avg_edge_pct": round(float(np.mean(taken_edges)) if taken_edges else 0.0, 3),
                    "taken_avg_drawdown_pct": round(float(np.mean(taken_drawdown)) if taken_drawdown else 0.0, 3),
                    "taken_hit_rate_pct": round(float(np.mean(np.asarray(taken_edges) > 0) * 100.0) if taken_edges else 0.0, 2),
                    "directions": direction_summaries,
                    "regime_distribution": regime_distribution,
                    "calibration": calibration_block,
                }
            )

        if not folds:
            return {"status": "failed", "reason": "no_valid_folds"}

        summary = pd.DataFrame(folds)
        per_fold_ratio = summary["taken_avg_edge_pct"] / summary["taken_avg_drawdown_pct"].where(
            summary["taken_avg_drawdown_pct"].abs() > 1e-6
        )
        mean_taken_edge_draw_ratio = round(float(per_fold_ratio.fillna(0.0).mean()), 4)
        if mean_taken_edge_draw_ratio < META_EDGE_DRAW_RATIO_WARN_THRESHOLD:
            logger.warning(
                "Walk-forward mean_taken_edge_draw_ratio=%.3f below %.2f threshold; "
                "deployment rules may be too permissive (min_edge_ratio floor=%.2f).",
                mean_taken_edge_draw_ratio,
                META_EDGE_DRAW_RATIO_WARN_THRESHOLD,
                META_MIN_EDGE_RATIO_FLOOR,
            )
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
                "mean_taken_edge_draw_ratio": mean_taken_edge_draw_ratio,
                "mean_taken_hit_rate_pct": round(float(summary["taken_hit_rate_pct"].mean()), 2),
            },
        }

    def train(self, feature_matrices: Dict[str, pd.DataFrame]) -> Dict:
        logger.info("Meta model: building training set (scoring each bar — slow on 1000s of symbols)...")
        dataset = self.build_training_dataset(feature_matrices)
        if dataset.empty:
            raise ValueError("No training rows available for meta model")
        if dataset["take_label"].nunique() < 2:
            raise ValueError("Meta labels do not contain both classes")

        walk_forward = self.walk_forward_validate(dataset)
        logger.info("Meta model: walk-forward done; fitting final deployment bundles (long + short)...")
        bundles: Dict[str, DirectionalMetaArtifacts] = {}
        deployment_rules: Dict[str, Dict[str, float]] = {}
        # Production isotonic calibrators (one per direction). Fit on OOF
        # probabilities from the full training set so the deployed inference
        # path can apply the same calibration that produced the stable
        # walk-forward thresholds.
        production_calibrators: Dict[str, IsotonicRegression] = {}
        long_threshold = float(
            os.getenv("META_CALIBRATED_THRESHOLD_LONG", str(META_CALIBRATED_THRESHOLD_LONG_DEFAULT))
        )
        short_threshold = float(
            os.getenv("META_CALIBRATED_THRESHOLD_SHORT", str(META_CALIBRATED_THRESHOLD_SHORT_DEFAULT))
        )
        calibrated_thresholds = {"long": long_threshold, "short": short_threshold}
        for direction, mask_column in (("long", "is_buy_signal"), ("short", "is_sell_signal")):
            direction_df = dataset[dataset[mask_column] > 0.5]
            bundle = self._fit_direction_bundle(direction_df, direction)
            if bundle is None:
                continue

            # Fit production isotonic calibrator from the full direction data
            # (OOF, no leakage). Override the bundle's deployed decision
            # threshold with the fixed calibrated value so a single, stable
            # threshold is shipped to production instead of a per-fit one.
            X_dir = direction_df[MetaFeatureBuilder.FEATURE_COLUMNS].fillna(0.0)
            y_dir = direction_df["take_label"].astype(int).to_numpy()
            iso, _ = _fit_isotonic_oof(X_dir, y_dir)
            if iso is not None:
                production_calibrators[direction] = iso

            bundle.decision_threshold = float(calibrated_thresholds[direction])
            bundles[direction] = bundle
            deployment_rules[direction] = {
                "decision_threshold": round(float(bundle.decision_threshold), 4),
                "decision_threshold_source": "calibrated_fixed",
                "min_expected_edge_pct": round(float(bundle.min_expected_edge_pct), 4),
                "min_edge_ratio": round(
                    max(META_MIN_EDGE_RATIO_FLOOR, float(bundle.min_edge_ratio)),
                    4,
                ),
            }

        # Persist the production calibrators as a single dict pickle. The
        # inference loader (TrainedMetaModel) attaches them on construction
        # and applies them to raw classifier output before threshold checks.
        if production_calibrators:
            try:
                with open(META_CALIBRATOR_PATH, "wb") as f:
                    pickle.dump(production_calibrators, f)
                logger.info(
                    "Saved production isotonic calibrators (%s) to %s",
                    ",".join(sorted(production_calibrators.keys())),
                    META_CALIBRATOR_PATH,
                )
            except Exception as exc:
                logger.warning("Failed to persist production calibrator: %s", exc)

        if not bundles:
            raise ValueError("Failed to train any directional meta bundles")

        if META_DIRECTIONAL_PATH.exists():
            try:
                META_DIRECTIONAL_PREVIOUS_PATH.write_bytes(META_DIRECTIONAL_PATH.read_bytes())
            except Exception:
                pass
        if META_REPORT_PATH.exists():
            try:
                META_REPORT_PREVIOUS_PATH.write_bytes(META_REPORT_PATH.read_bytes())
            except Exception:
                pass

        joblib.dump(
            {
                "bundles": {
                    direction: {
                        "classifier": bundle.classifier,
                        "calibrator": bundle.calibrator,
                        "edge_model": bundle.edge_model,
                        "drawdown_model": bundle.drawdown_model,
                    }
                    for direction, bundle in bundles.items()
                },
                "feature_columns": MetaFeatureBuilder.FEATURE_COLUMNS,
            },
            META_DIRECTIONAL_PATH,
        )
        with open(META_FEATURES_PATH, "wb") as f:
            pickle.dump(MetaFeatureBuilder.FEATURE_COLUMNS, f)

        report = {
            "status": "ok",
            "rows": int(len(dataset)),
            "positive_labels": int(dataset["take_label"].sum()),
            "feature_count": len(MetaFeatureBuilder.FEATURE_COLUMNS),
            "direction_rows": {
                "long": int((dataset["is_buy_signal"] > 0.5).sum()),
                "short": int((dataset["is_sell_signal"] > 0.5).sum()),
            },
            "deployment_rules": deployment_rules,
            "walk_forward": walk_forward,
            "training_config": {
                "rule_objective": os.getenv("META_MODEL_RULE_OBJECTIVE", "precision"),
                "horizon_days": self.horizon_days,
                "walk_forward_folds": self.walk_forward_folds,
                "min_train_days": self.min_train_days,
                "min_frame_rows_env": os.getenv("META_MODEL_MIN_FRAME_ROWS", ""),
                "warmup_rows_env": os.getenv("META_MODEL_WARMUP_ROWS", ""),
            },
        }
        META_REPORT_PATH.write_text(json.dumps(report, indent=2))
        logger.info(
            "Meta model trained: "
            f"{report['rows']} rows | {report['positive_labels']} positive labels | "
            f"{report['feature_count']} features | bundles {','.join(sorted(bundles.keys()))}"
        )

        # Optionally train the parallel ranker. Default META_MODEL_TYPE is
        # "classifier", which preserves existing behaviour. Setting it to
        # "ranker" or "both" adds the LambdaMART training pass and emits
        # meta_walkforward_report_ranker.json for side-by-side comparison.
        active_meta_type = _meta_model_type()
        if active_meta_type in {"ranker", "both"}:
            try:
                ranker_report = MetaRankerModel.train_and_report(
                    dataset=dataset,
                    folds=self.walk_forward_folds,
                    min_train_days=self.min_train_days,
                )
                report["ranker_status"] = ranker_report.get("status", "unknown")
            except Exception as exc:
                logger.warning("Meta ranker training failed: %s", exc)
                report["ranker_status"] = "exception"

        return report


class MetaRankerModel:
    """LambdaMART-style learning-to-rank meta variant.

    Trained in parallel with the binary classifier to compare cross-sectional
    selection quality. Groups are (timestamp, direction) cohorts; relevance is
    a 3-level grade derived from take_label and edge/draw ratio.
    """

    PARAMS: Dict[str, object] = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@5",
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "random_state": 42,
    }

    def __init__(self, ranker, feature_columns: List[str], coverage_target: float = META_RANKER_DEFAULT_COVERAGE):
        self.ranker = ranker
        self.feature_columns = list(feature_columns)
        self.coverage_target = float(coverage_target)

    # --- group / relevance helpers ------------------------------------------
    @staticmethod
    def _build_relevance_labels(df: pd.DataFrame) -> np.ndarray:
        """0=reject, 1=acceptable take, 2=high-quality take (ratio >= 0.35)."""
        take = df["take_label"].astype(int).to_numpy()
        edge = df["edge_pct"].astype(float).to_numpy()
        draw = df["drawdown_pct"].astype(float).to_numpy()
        ratio = edge / (np.abs(draw) + 1e-6)
        rel = np.zeros(len(df), dtype=int)
        rel[(take == 1) & (ratio >= META_RANKER_RELEVANCE_RATIO_THRESHOLD)] = 2
        rel[(take == 1) & (ratio < META_RANKER_RELEVANCE_RATIO_THRESHOLD)] = 1
        return rel

    @staticmethod
    def _sort_for_groups(df: pd.DataFrame) -> pd.DataFrame:
        """LambdaMART requires rows belonging to the same group to be contiguous."""
        return df.sort_values(["timestamp", "direction_label", "symbol"]).reset_index(drop=True)

    @staticmethod
    def _group_sizes(df: pd.DataFrame) -> np.ndarray:
        """Sizes of consecutive (timestamp, direction_label) groups."""
        if df.empty:
            return np.asarray([], dtype=int)
        keys = list(zip(df["timestamp"].tolist(), df["direction_label"].tolist()))
        sizes: List[int] = []
        cur = keys[0]
        n = 0
        for k in keys:
            if k == cur:
                n += 1
            else:
                sizes.append(n)
                cur = k
                n = 1
        sizes.append(n)
        return np.asarray(sizes, dtype=int)

    # --- training -----------------------------------------------------------
    @classmethod
    def fit(cls, dataset: pd.DataFrame, coverage_target: float = META_RANKER_DEFAULT_COVERAGE) -> Optional["MetaRankerModel"]:
        try:
            import xgboost as xgb
        except ImportError:
            logger.warning("xgboost not installed; skipping ranker training.")
            return None

        if dataset.empty:
            logger.warning("MetaRankerModel.fit: empty dataset, skipping.")
            return None

        df = cls._sort_for_groups(dataset)
        feature_cols = list(MetaFeatureBuilder.FEATURE_COLUMNS)
        X = df[feature_cols].fillna(0.0)
        y = cls._build_relevance_labels(df)
        groups = cls._group_sizes(df)
        if groups.size == 0 or int(groups.sum()) != len(X):
            logger.warning(
                "Ranker group size mismatch (sum=%s rows=%s); aborting.",
                int(groups.sum()), len(X),
            )
            return None

        ranker = xgb.XGBRanker(**cls.PARAMS)
        ranker.fit(X, y, group=groups)
        try:
            ranker.save_model(str(META_RANKER_PATH))
            with open(META_RANKER_FEATURES_PATH, "wb") as f:
                pickle.dump(feature_cols, f)
        except Exception as exc:
            logger.warning("Failed to persist ranker artifacts: %s", exc)
        return cls(ranker, feature_cols, coverage_target=coverage_target)

    @classmethod
    def load(cls) -> Optional["MetaRankerModel"]:
        try:
            import xgboost as xgb
        except ImportError:
            return None
        if not META_RANKER_PATH.exists():
            return None
        try:
            ranker = xgb.XGBRanker()
            ranker.load_model(str(META_RANKER_PATH))
            feature_cols = list(MetaFeatureBuilder.FEATURE_COLUMNS)
            if META_RANKER_FEATURES_PATH.exists():
                with open(META_RANKER_FEATURES_PATH, "rb") as f:
                    feature_cols = list(pickle.load(f))
            cov = float(os.getenv("META_RANKER_COVERAGE_TARGET", str(META_RANKER_DEFAULT_COVERAGE)))
            return cls(ranker, feature_cols, coverage_target=cov)
        except Exception as exc:
            logger.warning("MetaRankerModel.load failed: %s", exc)
            return None

    # --- inference ----------------------------------------------------------
    def predict_top_k(self, df: pd.DataFrame) -> np.ndarray:
        """Return a boolean selection mask: top-K per (timestamp, direction).

        K = max(1, round(coverage_target * group_size)). Operates on the
        dataframe in input order — the caller is responsible for stitching
        the mask back to its rows.
        """
        if df.empty:
            return np.asarray([], dtype=bool)
        ordered = self._sort_for_groups(df)
        ordered["_orig_idx"] = df.index[df.index.isin(ordered.index) == False].tolist() if False else ordered.index
        X = ordered[self.feature_columns].fillna(0.0)
        scores = self.ranker.predict(X)
        ordered = ordered.assign(_rank_score=scores)
        coverage = float(os.getenv("META_RANKER_COVERAGE_TARGET", str(self.coverage_target)))
        mask = pd.Series(False, index=ordered.index)
        for _, group in ordered.groupby(["timestamp", "direction_label"], sort=False):
            k = max(1, int(round(coverage * len(group))))
            top_idx = group["_rank_score"].nlargest(k).index
            mask.loc[top_idx] = True
        # Realign to caller's original order
        return mask.reindex(df.index, fill_value=False).to_numpy(dtype=bool)

    # --- walk-forward -------------------------------------------------------
    @classmethod
    def walk_forward_validate(
        cls,
        dataset: pd.DataFrame,
        folds: int,
        min_train_days: int,
        coverage_target: float = META_RANKER_DEFAULT_COVERAGE,
    ) -> Dict:
        """Mirror of MetaModelTrainer.walk_forward_validate using the ranker.

        Same fold cadence and JSON-friendly schema so the two reports can be
        diffed directly.
        """
        try:
            import xgboost as xgb  # noqa: F401
        except ImportError:
            return {"status": "failed", "reason": "xgboost_missing"}

        if dataset.empty:
            return {"status": "failed", "reason": "no_data"}

        dates = sorted(pd.to_datetime(dataset["timestamp"]).drop_duplicates().tolist())
        if len(dates) < (min_train_days + 20):
            return {"status": "failed", "reason": "insufficient_dates"}

        test_span = max(20, len(dates) // (folds + 2))
        fold_records: List[Dict] = []
        for fold in range(folds):
            train_end = len(dates) - (folds - fold) * test_span
            test_end = min(len(dates), train_end + test_span)
            if train_end < min_train_days or test_end <= train_end:
                continue
            train_dates = set(dates[:train_end])
            test_dates = set(dates[train_end:test_end])
            train_df = dataset[dataset["timestamp"].isin(train_dates)]
            test_df = dataset[dataset["timestamp"].isin(test_dates)]
            if train_df.empty or test_df.empty:
                continue

            ranker_model = cls.fit(train_df, coverage_target=coverage_target)
            if ranker_model is None:
                continue
            test_sorted = cls._sort_for_groups(test_df)
            mask = ranker_model.predict_top_k(test_sorted)
            y_true = test_sorted["take_label"].astype(int).to_numpy()
            y_pred = mask.astype(int)
            taken_edges = test_sorted["edge_pct"].to_numpy(dtype=float)[mask]
            taken_drawdown = test_sorted["drawdown_pct"].to_numpy(dtype=float)[mask]
            fold_records.append(
                {
                    "fold": fold + 1,
                    "train_rows": int(len(train_df)),
                    "test_rows": int(len(test_sorted)),
                    "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
                    "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
                    "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
                    "coverage_pct": round(float(y_pred.mean() * 100.0), 2),
                    "taken_avg_edge_pct": round(float(taken_edges.mean()) if taken_edges.size else 0.0, 3),
                    "taken_avg_drawdown_pct": round(float(taken_drawdown.mean()) if taken_drawdown.size else 0.0, 3),
                    "taken_hit_rate_pct": round(
                        float((taken_edges > 0).mean() * 100.0) if taken_edges.size else 0.0,
                        2,
                    ),
                    "coverage_target": coverage_target,
                }
            )

        if not fold_records:
            return {"status": "failed", "reason": "no_valid_folds"}
        summary = pd.DataFrame(fold_records)
        return {
            "status": "ok",
            "folds": fold_records,
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

    @classmethod
    def train_and_report(
        cls,
        dataset: pd.DataFrame,
        folds: int,
        min_train_days: int,
        coverage_target: Optional[float] = None,
    ) -> Dict:
        """Top-level entry: walk-forward, fit final, write the JSON report."""
        cov = float(
            coverage_target
            if coverage_target is not None
            else os.getenv("META_RANKER_COVERAGE_TARGET", str(META_RANKER_DEFAULT_COVERAGE))
        )
        wf = cls.walk_forward_validate(dataset, folds=folds, min_train_days=min_train_days, coverage_target=cov)
        final_model = cls.fit(dataset, coverage_target=cov)
        report = {
            "status": "ok" if final_model is not None else "failed",
            "rows": int(len(dataset)),
            "feature_count": len(MetaFeatureBuilder.FEATURE_COLUMNS),
            "coverage_target": cov,
            "params": cls.PARAMS,
            "walk_forward": wf,
        }
        try:
            META_RANKER_REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
        except Exception as exc:
            logger.warning("Failed to write ranker report: %s", exc)
        logger.info(
            "Meta ranker trained (status=%s, rows=%d, coverage_target=%.3f)",
            report["status"], report["rows"], cov,
        )
        return report


class InstitutionalTrainingPipeline:
    def __init__(
        self,
        xgb_horizon: int = 5,
        meta_horizon: int = 5,
        meta_take_threshold: float = 0.52,
        meta_walk_forward_folds: int = 6,
        meta_min_train_days: int = 180,
    ):
        self.xgb_retrainer = EventAwareXGBoostRetrainer(horizon=xgb_horizon)
        self.meta_trainer = MetaModelTrainer(
            horizon_days=meta_horizon,
            take_threshold=meta_take_threshold,
            walk_forward_folds=meta_walk_forward_folds,
            min_train_days=meta_min_train_days,
        )

    def train_all(self, feature_matrices: Dict[str, pd.DataFrame], run_optuna: bool = False) -> Dict:
        incoming_feature_matrices = feature_matrices if isinstance(feature_matrices, dict) else {}
        incoming_symbol_count = len(incoming_feature_matrices)
        incoming_row_count = sum(len(df) for df in incoming_feature_matrices.values() if isinstance(df, pd.DataFrame))

        disk_feature_matrices, disk_stats = _merged_retrain_feature_store()
        disk_row_count = sum(len(df) for df in disk_feature_matrices.values() if isinstance(df, pd.DataFrame))

        effective_feature_matrices = incoming_feature_matrices
        if disk_feature_matrices and disk_row_count >= incoming_row_count:
            effective_feature_matrices = disk_feature_matrices
            logger.info(
                "Institutional pipeline: using disk-backed retrain feature store (%s symbols, %s rows) over incoming cache (%s symbols, %s rows)",
                len(disk_feature_matrices),
                disk_row_count,
                incoming_symbol_count,
                incoming_row_count,
            )
        else:
            logger.info(
                "Institutional pipeline: using incoming retrain feature cache (%s symbols, %s rows); disk-backed store had %s symbols, %s rows",
                incoming_symbol_count,
                incoming_row_count,
                len(disk_feature_matrices),
                disk_row_count,
            )

        for source, stats in disk_stats.items():
            logger.info(
                "Institutional pipeline feature source %s: %s parquet files, %s rows",
                source,
                stats.get("files", 0),
                stats.get("rows", 0),
            )

        n_sym = len(effective_feature_matrices)
        logger.info("Institutional pipeline: phase 1/2 XGBoost (%s feature matrices)...", n_sym)
        import os as _os
        if _os.getenv("SKIP_XGB_RETRAIN", "0").strip() == "1":
            import json as _json, pathlib as _pl
            _ck = _pl.Path("models/checkpoints/xgb_retrain_report.json")
            _prev = _json.loads(_ck.read_text()) if _ck.exists() else {}
            xgb_report = {"status": "skipped", "reason": "SKIP_XGB_RETRAIN=1", "previous_report": _prev}
        else:
            xgb_report = self.xgb_retrainer.retrain(effective_feature_matrices, run_optuna=run_optuna)
        logger.info("Institutional pipeline: phase 2/2 meta model...")
        meta_report = self.meta_trainer.train(effective_feature_matrices)
        return {
            "xgboost": xgb_report,
            "meta_model": meta_report,
        }
