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
from sklearn.metrics import accuracy_score, precision_score, recall_score
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
PROJECT_DIR = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_DIR / "data" / "features"
FEATURE_DIR_10YR = PROJECT_DIR / "data" / "features_10yr"


def _clip_prob(value: float) -> float:
    return float(np.clip(float(value), 1e-6, 1.0 - 1e-6))


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


def _normalize_market_scope(raw_scope: Optional[str]) -> str:
    scope = str(raw_scope or "all").strip().lower()
    aliases = {
        "": "all",
        "stocks": "us",
        "equities": "us",
        "us-stocks": "us",
        "us_equities": "us",
        "usa": "us",
        "india": "nse",
        "ind": "nse",
        "nse-stocks": "nse",
        "crypto-only": "crypto",
    }
    return aliases.get(scope, scope if scope in {"all", "us", "nse", "crypto"} else "all")


def _market_bucket_for_symbol(symbol: str) -> str:
    raw = str(symbol or "").strip().upper()
    if raw.endswith(".NS"):
        return "nse"
    if "USDT" in raw or raw.endswith("-USD") or raw.endswith("USD"):
        return "crypto"
    return "us"


def _symbol_matches_market_scope(symbol: str, market_scope: str) -> bool:
    scope = _normalize_market_scope(market_scope)
    if scope == "all":
        return True
    return _market_bucket_for_symbol(symbol) == scope


def _filter_feature_matrices_for_market(
    feature_matrices: Dict[str, pd.DataFrame],
    market_scope: str,
) -> Dict[str, pd.DataFrame]:
    scope = _normalize_market_scope(market_scope)
    if scope == "all":
        return dict(feature_matrices)
    return {
        symbol: frame
        for symbol, frame in feature_matrices.items()
        if _symbol_matches_market_scope(symbol, scope)
    }


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
        "cross_sectional_candidate_count",
        "cross_sectional_selection_rank_pct",
        "cross_sectional_edge_rank_pct",
        "cross_sectional_edge_ratio_rank_pct",
        "cross_sectional_drawdown_rank_pct",
        "cross_sectional_conviction_rank_pct",
        "cross_sectional_score_rank_pct",
        "is_nse_symbol",
        "meta_year_norm",
        "meta_month_sin",
        "meta_month_cos",
        "meta_dow_sin",
        "meta_dow_cos",
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
                            min_edge_ratio=0.18,
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
                    min_edge_ratio=float(
                        os.getenv(
                            "META_MODEL_RUNTIME_MIN_EDGE_RATIO",
                            str(rules.get("min_edge_ratio", 0.08) or 0.08),
                        )
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
                min_edge_ratio=float(
                    os.getenv(
                        "META_MODEL_RUNTIME_MIN_EDGE_RATIO",
                        str(deployment_rules.get("min_edge_ratio", 0.08) or 0.08),
                    )
                ),
                direction="legacy",
            )
            return cls({"long": legacy_bundle, "short": legacy_bundle}, feature_columns)
        except Exception as exc:
            logger.warning(f"Meta model load failed: {exc}")
            return None

    def predict_from_dict(self, features: Dict[str, float]) -> Dict[str, float]:
        frame = pd.DataFrame([{col: float(features.get(col, 0.0) or 0.0) for col in self.feature_columns}])
        direction = "long" if float(features.get("is_buy_signal", 0.0) or 0.0) >= float(features.get("is_sell_signal", 0.0) or 0.0) else "short"
        bundle = self.bundles.get(direction) or self.bundles.get("long") or next(iter(self.bundles.values()))
        raw_take_probability = float(bundle.classifier.predict_proba(frame)[0][1])
        take_probability = float(bundle.calibrator.transform(np.array([raw_take_probability]))[0]) if bundle.calibrator else raw_take_probability
        expected_edge_pct = float(bundle.edge_model.predict(frame)[0])
        expected_drawdown_pct = float(abs(bundle.drawdown_model.predict(frame)[0]))
        return {
            "take_probability": round(take_probability, 4),
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
        market_scope: str = "all",
    ):
        self.horizon_days = horizon_days
        self.take_threshold = take_threshold
        self.walk_forward_folds = walk_forward_folds
        self.min_train_days = min_train_days
        self.market_scope = _normalize_market_scope(market_scope)

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

        classifier = HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_depth=5,
            max_iter=480,
            min_samples_leaf=18,
            l2_regularization=0.08,
            random_state=42,
            class_weight="balanced",
        )
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

        # --- Regressor quality check: MAE on training set ---
        edge_mae = float(np.mean(np.abs(edge_model.predict(X) - y_edge.to_numpy())))
        draw_mae = float(np.mean(np.abs(drawdown_model.predict(X) - y_drawdown.to_numpy())))
        edge_mae_limit = float(os.getenv("META_MODEL_EDGE_MAE_LIMIT", "2.5"))
        draw_mae_limit = float(os.getenv("META_MODEL_DRAW_MAE_LIMIT", "3.0"))
        regressors_reliable = (edge_mae <= edge_mae_limit and draw_mae <= draw_mae_limit)
        if not regressors_reliable:
            logger.warning(
                "Regressor drift detected (%s): edge_MAE=%.3f (limit %.1f), "
                "draw_MAE=%.3f (limit %.1f). Falling back to precision-only gate.",
                direction, edge_mae, edge_mae_limit, draw_mae, draw_mae_limit,
            )

        if use_cv_calibration:
            # 3-fold cross-calibration: each fold predicts on held-out portion
            from sklearn.model_selection import KFold
            cv_raw = np.zeros(len(X))
            cv_labels = y_take.to_numpy(dtype=int)
            kf = KFold(n_splits=3, shuffle=False)
            for cv_train_idx, cv_val_idx in kf.split(X):
                cv_clf = HistGradientBoostingClassifier(
                    learning_rate=0.06, max_depth=5, max_iter=480,
                    min_samples_leaf=18, l2_regularization=0.08,
                    random_state=42, class_weight="balanced",
                )
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
        # If regressors are unreliable, disable edge/ratio gates (precision-only mode)
        if not regressors_reliable:
            decision_rule["min_expected_edge_pct"] = 0.0
            decision_rule["min_edge_ratio"] = 0.0
            logger.info("Regressor fallback (%s): edge/ratio gates disabled, precision-only gate active.", direction)
        return DirectionalMetaArtifacts(
            classifier=classifier,
            calibrator=calibrator,
            edge_model=edge_model,
            drawdown_model=drawdown_model,
            decision_threshold=float(decision_rule["decision_threshold"]),
            min_expected_edge_pct=float(decision_rule["min_expected_edge_pct"]),
            min_edge_ratio=float(decision_rule["min_edge_ratio"]),
            direction=direction,
        )

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
            if not _symbol_matches_market_scope(symbol, self.market_scope):
                continue
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
                path_take_label = int(
                    path_stats["edge_pct"] > label_min_edge
                    and edge_ratio > label_min_ratio
                    and path_stats["hit"] == 1
                )
                feature_values.update(
                    {
                        "symbol": symbol,
                        "timestamp": pd.Timestamp(ordered.index[idx]),
                        "market_bucket": _market_bucket_for_symbol(symbol),
                        "direction_label": "long" if direction == "buy" else "short",
                        "adaptive_horizon_days": horizon_days,
                        "adaptive_horizon_multiplier": horizon_multiplier,
                        "path_take_label": path_take_label,
                        "edge_pct": path_stats["edge_pct"],
                        "drawdown_pct": path_stats["drawdown_pct"],
                        "edge_ratio": round(float(edge_ratio), 4),
                        "hit": path_stats["hit"],
                        "barrier": path_stats["barrier"],
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
        dataset = self._apply_cross_sectional_context(dataset)
        logger.info("Meta dataset ready: %s rows", len(dataset))
        return dataset

    def _apply_cross_sectional_context(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if dataset.empty:
            return dataset

        dataset = dataset.copy()
        dataset["market_bucket"] = dataset["market_bucket"].fillna("us").astype(str)
        group_keys = ["timestamp", "market_bucket", "direction_label"]
        group_size = dataset.groupby(group_keys)["symbol"].transform("count").astype(float)
        dataset["cross_sectional_candidate_count"] = group_size

        def _numeric_series(column: str) -> pd.Series:
            if column in dataset.columns:
                return pd.to_numeric(dataset[column], errors="coerce").fillna(0.0)
            return pd.Series(0.0, index=dataset.index, dtype=float)

        directional_alpha = (
            0.50 * _numeric_series("alpha_signal")
            + 0.35 * _numeric_series("event_alpha_signal")
            + 0.15 * _numeric_series("earnings_propagation_signal")
        )
        long_mask = dataset["direction_label"].eq("long")
        directional_alpha = directional_alpha.where(long_mask, -directional_alpha)

        dataset["__directional_alpha"] = directional_alpha
        dataset["cross_sectional_selection_rank_pct"] = dataset.groupby(group_keys)["__directional_alpha"].rank(
            pct=True,
            method="average",
        )
        dataset["cross_sectional_edge_rank_pct"] = dataset.groupby(group_keys)["edge_pct"].rank(
            pct=True,
            method="average",
        )
        dataset["cross_sectional_edge_ratio_rank_pct"] = dataset.groupby(group_keys)["edge_ratio"].rank(
            pct=True,
            method="average",
        )
        dataset["cross_sectional_drawdown_rank_pct"] = dataset.groupby(group_keys)["drawdown_pct"].transform(
            lambda series: (-series).rank(pct=True, method="average")
        )
        dataset["cross_sectional_conviction_rank_pct"] = dataset.groupby(group_keys)["conviction_score"].rank(
            pct=True,
            method="average",
        )

        dataset["__cross_sectional_score"] = (
            0.38 * dataset["cross_sectional_edge_rank_pct"]
            + 0.22 * dataset["cross_sectional_edge_ratio_rank_pct"]
            + 0.16 * dataset["cross_sectional_drawdown_rank_pct"]
            + 0.14 * dataset["cross_sectional_selection_rank_pct"]
            + 0.10 * dataset["cross_sectional_conviction_rank_pct"]
        )
        dataset["cross_sectional_score_rank_pct"] = dataset.groupby(group_keys)["__cross_sectional_score"].rank(
            pct=True,
            method="average",
        )

        top_pct = float(os.getenv("META_MODEL_CROSS_SECTIONAL_LABEL_PCT", "0.18") or 0.18)
        top_pct = float(np.clip(top_pct, 0.02, 0.50))
        min_candidates = max(3, int(os.getenv("META_MODEL_CROSS_SECTIONAL_MIN_CANDIDATES", "6") or 6))
        top_cutoff = 1.0 - top_pct
        small_group_mask = dataset["cross_sectional_candidate_count"] < float(min_candidates)
        selected_rank_mask = dataset["cross_sectional_score_rank_pct"] >= top_cutoff
        dataset["take_label"] = (
            dataset["path_take_label"].astype(int).eq(1)
            & (small_group_mask | selected_rank_mask)
        ).astype(int)

        dataset = dataset.drop(columns=["__directional_alpha", "__cross_sectional_score"], errors="ignore")
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

            for direction, mask_column in (("long", "is_buy_signal"), ("short", "is_sell_signal")):
                train_dir = train_df[train_df[mask_column] > 0.5]
                test_dir = test_df[test_df[mask_column] > 0.5]
                bundle = self._fit_direction_bundle(train_dir, direction)
                if bundle is None or test_dir.empty:
                    continue

                X_test = test_dir[MetaFeatureBuilder.FEATURE_COLUMNS].fillna(0.0)
                take_prob = self._bundle_take_probability(bundle, X_test)
                edge_pred = bundle.edge_model.predict(X_test)
                drawdown_pred = np.abs(bundle.drawdown_model.predict(X_test))
                pred_take = (
                    (take_prob >= bundle.decision_threshold)
                    & (edge_pred >= bundle.min_expected_edge_pct)
                    & ((edge_pred / np.maximum(drawdown_pred, 0.35)) >= bundle.min_edge_ratio)
                )

                realized_edges = test_dir["edge_pct"].to_numpy(dtype=float)
                realized_drawdown = test_dir["drawdown_pct"].to_numpy(dtype=float)
                taken_edges.extend(realized_edges[pred_take].tolist())
                taken_drawdown.extend(realized_drawdown[pred_take].tolist())
                all_true.extend(test_dir["take_label"].astype(int).tolist())
                all_pred.extend(pred_take.astype(int).tolist())
                direction_summaries[direction] = {
                    "test_rows": int(len(test_dir)),
                    "coverage_pct": round(float(pred_take.mean() * 100.0), 2),
                    "decision_threshold": round(float(bundle.decision_threshold), 4),
                    "min_expected_edge_pct": round(float(bundle.min_expected_edge_pct), 4),
                    "min_edge_ratio": round(float(bundle.min_edge_ratio), 4),
                }

            if not all_true:
                continue

            all_true_arr = np.asarray(all_true, dtype=int)
            all_pred_arr = np.asarray(all_pred, dtype=int)
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
        for direction, mask_column in (("long", "is_buy_signal"), ("short", "is_sell_signal")):
            direction_df = dataset[dataset[mask_column] > 0.5]
            bundle = self._fit_direction_bundle(direction_df, direction)
            if bundle is None:
                continue
            bundles[direction] = bundle
            deployment_rules[direction] = {
                "decision_threshold": round(float(bundle.decision_threshold), 4),
                "min_expected_edge_pct": round(float(bundle.min_expected_edge_pct), 4),
                "min_edge_ratio": round(float(bundle.min_edge_ratio), 4),
            }

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
            "path_positive_labels": int(dataset["path_take_label"].sum()) if "path_take_label" in dataset.columns else 0,
            "positive_labels": int(dataset["take_label"].sum()),
            "feature_count": len(MetaFeatureBuilder.FEATURE_COLUMNS),
            "direction_rows": {
                "long": int((dataset["is_buy_signal"] > 0.5).sum()),
                "short": int((dataset["is_sell_signal"] > 0.5).sum()),
            },
            "deployment_rules": deployment_rules,
            "walk_forward": walk_forward,
            "training_config": {
                "market_scope": self.market_scope,
                "rule_objective": os.getenv("META_MODEL_RULE_OBJECTIVE", "precision"),
                "horizon_days": self.horizon_days,
                "walk_forward_folds": self.walk_forward_folds,
                "min_train_days": self.min_train_days,
                "min_frame_rows_env": os.getenv("META_MODEL_MIN_FRAME_ROWS", ""),
                "warmup_rows_env": os.getenv("META_MODEL_WARMUP_ROWS", ""),
                "cross_sectional_label_pct": os.getenv("META_MODEL_CROSS_SECTIONAL_LABEL_PCT", "0.18"),
                "cross_sectional_min_candidates": os.getenv("META_MODEL_CROSS_SECTIONAL_MIN_CANDIDATES", "6"),
            },
        }
        META_REPORT_PATH.write_text(json.dumps(report, indent=2))
        logger.info(
            "Meta model trained: "
            f"{report['rows']} rows | {report['positive_labels']} positive labels | "
            f"{report['feature_count']} features | bundles {','.join(sorted(bundles.keys()))}"
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
        market_scope: str = "all",
    ):
        self.market_scope = _normalize_market_scope(market_scope)
        self.xgb_retrainer = EventAwareXGBoostRetrainer(horizon=xgb_horizon)
        self.meta_trainer = MetaModelTrainer(
            horizon_days=meta_horizon,
            take_threshold=meta_take_threshold,
            walk_forward_folds=meta_walk_forward_folds,
            min_train_days=meta_min_train_days,
            market_scope=self.market_scope,
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

        effective_feature_matrices = _filter_feature_matrices_for_market(
            effective_feature_matrices,
            self.market_scope,
        )
        if not effective_feature_matrices:
            raise ValueError(f"No feature matrices matched market_scope={self.market_scope}")

        n_sym = len(effective_feature_matrices)
        logger.info(
            "Institutional pipeline: market scope %s (%s feature matrices after filter)",
            self.market_scope,
            n_sym,
        )
        logger.info("Institutional pipeline: phase 1/2 XGBoost (%s feature matrices)...", n_sym)
        xgb_report = self.xgb_retrainer.retrain(effective_feature_matrices, run_optuna=run_optuna)
        logger.info("Institutional pipeline: phase 2/2 meta model...")
        meta_report = self.meta_trainer.train(effective_feature_matrices)
        return {
            "xgboost": xgb_report,
            "meta_model": meta_report,
        }
