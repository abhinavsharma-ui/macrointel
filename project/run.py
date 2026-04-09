"""
Master system orchestrator for the Macro Intelligence project.
"""

from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.example", override=False)

import os
import json

# Apply performance caps *before* importing NumPy/Pandas/XGBoost/etc.
from core.performance import apply_performance_profile

_PERF_SETTINGS = apply_performance_profile()

import logging
import math
import subprocess
import sys
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _get_primary_env_value(*env_names: str) -> str:
    for env_name in env_names:
        raw = os.getenv(env_name, "")
        for part in raw.replace(";", ",").split(","):
            value = part.strip()
            if value:
                return value
    return ""


def _resolve_root_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def _parse_symbol_blob(raw: str) -> List[str]:
    text = str(raw or "")
    values = []
    for line in text.splitlines():
        values.extend(line.replace(";", ",").split(","))
    cleaned = []
    for value in values:
        item = str(value).strip().upper()
        if not item or item.startswith("#"):
            continue
        cleaned.append(item)
    return list(dict.fromkeys(cleaned))


def _load_symbol_file(path_value: str) -> List[str]:
    path_text = str(path_value or "").strip()
    if not path_text:
        return []
    path = _resolve_root_path(path_text)
    if not path.exists():
        return []
    try:
        return _parse_symbol_blob(path.read_text(encoding="utf-8"))
    except Exception:
        return []


class MacroIntelligenceSystem:
    def __init__(self):
        self._signal_store: Dict = {}
        self._components: Dict = {}
        self._running = False
        self._data_versions = {"prices": 0, "sentiment": 0, "earnings": 0, "altdata": 0}
        self._last_inference_versions = dict(self._data_versions)
        self._last_seen_tick_ts: Dict[str, str] = {}
        self._api_keys = {
            "alpha_vantage": _get_primary_env_value("ALPHA_VANTAGE_API_KEYS", "ALPHA_VANTAGE_API_KEY"),
            "news_api": _get_primary_env_value("NEWS_API_KEYS", "NEWS_API_KEY"),
            "finnhub": _get_primary_env_value("FINNHUB_API_KEYS", "FINNHUB_API_KEY"),
            "upstox": os.getenv("UPSTOX_ACCESS_TOKEN", ""),
            "fred": os.getenv("FRED_API_KEY", ""),
        }
        self._universe_mode = os.getenv("UNIVERSE_MODE", "full").strip().lower() or "full"
        if self._universe_mode not in {
            "core",
            "full",
            "us",
            "nse",
            "daytrade",
            "daytrade_us",
            "daytrade_nse",
            "throughput",
            "throughput_us",
            "throughput_nse",
        }:
            logger.warning(f"Invalid UNIVERSE_MODE '{self._universe_mode}', defaulting to full")
            self._universe_mode = "full"
        self._day_trading_mode = (
            os.getenv("DAY_TRADING_MODE", "1").strip().lower() in {"1", "true", "yes", "on"}
            or self._universe_mode.startswith(("daytrade", "throughput"))
        )
        self._day_trade_force_daytrade_universe = (
            os.getenv("DAY_TRADE_FORCE_DAYTRADE_UNIVERSE", "0").strip().lower() in {"1", "true", "yes", "on"}
        )
        self._throughput_mode = os.getenv("THROUGHPUT_MODE", "1").strip().lower() in {"1", "true", "yes", "on"}
        self._universe_sync_summary: Dict[str, Any] = {}
        self._source_selection_summary: Dict[str, Dict[str, Any]] = {}
        self._pipeline_rotation_offsets: Dict[str, int] = defaultdict(int)
        self._data_pipeline_status: Dict[str, Any] = {}
        self._pipeline_priority_symbol_cap = max(0, int(os.getenv("PIPELINE_PRIORITY_SYMBOL_CAP", "0") or 0))
        self._source_symbol_caps = {
            "prices": max(0, int(os.getenv("PRICE_PIPELINE_SYMBOL_CAP", "0") or 0)),
            "sentiment": max(0, int(os.getenv("SENTIMENT_PIPELINE_SYMBOL_CAP", "0") or 0)),
            "earnings": max(0, int(os.getenv("EARNINGS_PIPELINE_SYMBOL_CAP", "0") or 0)),
            "altdata": max(0, int(os.getenv("ALTDATA_PIPELINE_SYMBOL_CAP", "0") or 0)),
        }
        self._binance_ws_symbols_per_connection = max(
            1,
            int(os.getenv("BINANCE_WS_SYMBOLS_PER_CONNECTION", "110") or 110),
        )
        self._bybit_ws_symbols_per_connection = max(
            1,
            int(os.getenv("BYBIT_WS_SYMBOLS_PER_CONNECTION", "60") or 60),
        )
        if self._throughput_mode or os.getenv("OFFICIAL_UNIVERSE_AUTO_SYNC", "1").strip().lower() in {"1", "true", "yes", "on"}:
            self._universe_sync_summary = self._sync_official_universes()
        self._crypto_depth_enabled = os.getenv("CRYPTO_DEPTH_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
        crypto_default = (
            "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,LINKUSDT,BNBUSDT,AVAXUSDT,"
            "LTCUSDT,BCHUSDT,DOTUSDT,TRXUSDT,ATOMUSDT,ETCUSDT,XLMUSDT,NEARUSDT,APTUSDT,"
            "ARBUSDT,OPUSDT,FILUSDT,ICPUSDT,INJUSDT,UNIUSDT,AAVEUSDT,SANDUSDT,MANAUSDT,"
            "ALGOUSDT,VETUSDT,EOSUSDT"
        )
        self._crypto_symbols = list(
            dict.fromkeys(
                _parse_symbol_blob(os.getenv("CRYPTO_DEPTH_SYMBOLS", crypto_default))
                + _load_symbol_file(os.getenv("CRYPTO_DEPTH_SYMBOLS_FILE", ""))
            )
        )
        self._crypto_feed_order = [
            name.strip().lower()
            for name in os.getenv("CRYPTO_FEED_ORDER", "bybit,binance").replace(";", ",").split(",")
            if name.strip()
        ] or ["bybit"]
        self._crypto_use_all_feeds = os.getenv("CRYPTO_FEED_FALLBACK_ENABLED", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._crypto_primary_feed_only_symbols = os.getenv(
            "CRYPTO_PRIMARY_FEED_ONLY_SYMBOLS",
            "1",
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._binance_crypto_symbols = list(
            dict.fromkeys(
                _parse_symbol_blob(os.getenv("BINANCE_CRYPTO_SYMBOLS", ""))
                + _load_symbol_file(os.getenv("BINANCE_CRYPTO_SYMBOLS_FILE", ""))
            )
        )
        self._bybit_crypto_symbols = list(
            dict.fromkeys(
                _parse_symbol_blob(os.getenv("BYBIT_CRYPTO_SYMBOLS", ""))
                + _load_symbol_file(os.getenv("BYBIT_CRYPTO_SYMBOLS_FILE", ""))
            )
        )
        if not self._binance_crypto_symbols:
            self._binance_crypto_symbols = list(self._crypto_symbols)
        if not self._bybit_crypto_symbols:
            self._bybit_crypto_symbols = list(self._crypto_symbols)
        if self._throughput_mode and len(self._crypto_symbols) < 24:
            self._crypto_symbols = list(dict.fromkeys(self._crypto_symbols + _parse_symbol_blob(crypto_default)))
        if self._throughput_mode and len(self._binance_crypto_symbols) < 24:
            self._binance_crypto_symbols = list(dict.fromkeys(self._binance_crypto_symbols + self._crypto_symbols))
        if self._throughput_mode and len(self._bybit_crypto_symbols) < 24:
            self._bybit_crypto_symbols = list(dict.fromkeys(self._bybit_crypto_symbols + self._crypto_symbols))
        if self._crypto_primary_feed_only_symbols and not self._crypto_use_all_feeds:
            primary_feed = self._crypto_feed_order[0] if self._crypto_feed_order else "bybit"
            if primary_feed == "bybit" and self._bybit_crypto_symbols:
                self._crypto_symbols = list(self._bybit_crypto_symbols)
            elif primary_feed == "binance" and self._binance_crypto_symbols:
                self._crypto_symbols = list(self._binance_crypto_symbols)
        self._crypto_signal_stale_seconds = max(15, int(os.getenv("CRYPTO_SIGNAL_STALE_SECONDS", "45")))
        self._crypto_signal_hold_grace_seconds = max(
            30,
            int(os.getenv("CRYPTO_SIGNAL_HOLD_GRACE_SECONDS", "420")),
        )
        self._crypto_signal_stale_retention_seconds = max(
            self._crypto_signal_hold_grace_seconds,
            int(
                os.getenv(
                    "CRYPTO_SIGNAL_STALE_RETENTION_SECONDS",
                    str(self._crypto_signal_hold_grace_seconds * 3),
                )
            ),
        )
        self._crypto_signal_refresh_seconds = max(2.0, float(os.getenv("CRYPTO_SIGNAL_REFRESH_SECONDS", "5")))
        self._allow_dual_lane_variants = os.getenv("DUAL_LANE_VARIANTS_ENABLED", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._live_feature_overlay_enabled = os.getenv("LIVE_FEATURE_OVERLAY_ENABLED", "1").strip().lower() not in {"0", "false", "off"}
        self._live_signal_max_tick_age_seconds = max(5, int(os.getenv("LIVE_SIGNAL_MAX_TICK_AGE_SECONDS", "120")))
        self._crypto_min_notional_usd = max(5.0, float(os.getenv("CRYPTO_MIN_NOTIONAL_USD", "75")))
        self._model_learning_enabled = os.getenv("MODEL_LEARNING_ENABLED", "1").strip().lower() not in {"0", "false", "off"}
        self._model_learning_refresh_seconds = max(
            900,
            int(os.getenv("MODEL_LEARNING_REFRESH_SECONDS", "21600")),
        )
        self._drift_recent_window_hours = max(1, int(os.getenv("MODEL_DRIFT_WINDOW_HOURS", "4")))
        self._drift_baseline_window_days = max(7, int(os.getenv("MODEL_DRIFT_BASELINE_DAYS", "30")))
        self._drift_error_trigger_mult = max(1.1, float(os.getenv("MODEL_DRIFT_ERROR_TRIGGER_MULT", "1.5")))
        self._lane_pause_win_rate_floor = min(0.90, max(0.10, float(os.getenv("LANE_PAUSE_WIN_RATE_FLOOR", "0.45"))))
        self._lane_pause_trade_window = max(5, int(os.getenv("LANE_PAUSE_TRADE_WINDOW", "20")))
        self._emergency_retrain_cooldown_seconds = max(
            900,
            int(os.getenv("EMERGENCY_RETRAIN_COOLDOWN_SECONDS", "3600")),
        )
        self._signal_halflife_exit_ratio = min(0.95, max(0.10, float(os.getenv("DAY_SIGNAL_HALFLIFE_EXIT_RATIO", "0.60"))))
        self._event_window_mode = os.getenv("EVENT_WINDOW_MODE", "0").strip().lower() in {"1", "true", "yes", "on"}
        self._event_window_days = max(1, int(os.getenv("EVENT_WINDOW_DAYS", "3")))
        if (
            self._day_trading_mode
            and self._day_trade_force_daytrade_universe
            and not self._universe_mode.startswith("daytrade")
        ):
            self._universe_mode = "daytrade"
        if self._throughput_mode and self._universe_mode in {"core", "full"}:
            self._universe_mode = "throughput"
        elif self._throughput_mode and self._universe_mode == "us":
            self._universe_mode = "throughput_us"
        elif self._throughput_mode and self._universe_mode == "nse":
            self._universe_mode = "throughput_nse"
        self._poll_batch_size = max(1, int(os.getenv("POLL_BATCH_SIZE", "20")))
        self._poll_interval_seconds = max(5, int(os.getenv("POLL_INTERVAL_SECONDS", "15")))
        self._sentiment_batch_size = max(1, int(os.getenv("SENTIMENT_BATCH_SIZE", "25")))
        self._sentiment_pause_seconds = max(0.0, float(os.getenv("SENTIMENT_BATCH_PAUSE_SECONDS", "1.0")))
        self._price_refresh_seconds = max(120, int(os.getenv("PRICE_REFRESH_SECONDS", "900")))
        self._sentiment_refresh_seconds = max(900, int(os.getenv("SENTIMENT_REFRESH_SECONDS", "3600")))
        self._earnings_refresh_seconds = max(1800, int(os.getenv("EARNINGS_REFRESH_SECONDS", "21600")))
        self._altdata_refresh_seconds = max(900, int(os.getenv("ALTDATA_REFRESH_SECONDS", "3600")))
        self._inference_refresh_seconds = max(15, int(os.getenv("INFERENCE_REFRESH_SECONDS", "120")))
        self._entry_readiness_enabled = os.getenv("ENTRY_READINESS_ENABLED", "1").strip().lower() not in {"0", "false", "off"}
        self._entry_stale_multiplier = max(1.0, float(os.getenv("ENTRY_STALE_MULTIPLIER", "2.0")))
        self._entry_require_market_open = os.getenv("ENTRY_REQUIRE_MARKET_OPEN", "1").strip().lower() not in {"0", "false", "off"}
        self._entry_require_live_price_normal = os.getenv("ENTRY_REQUIRE_LIVE_PRICE_NORMAL", "1").strip().lower() not in {"0", "false", "off"}
        self._entry_require_live_price_day = os.getenv("ENTRY_REQUIRE_LIVE_PRICE_DAY", "1").strip().lower() not in {"0", "false", "off"}
        self._entry_require_live_price_crypto = os.getenv("ENTRY_REQUIRE_LIVE_PRICE_CRYPTO", "1").strip().lower() not in {"0", "false", "off"}
        self._auto_trade_min_conviction = max(0.0, float(os.getenv("AUTO_TRADE_MIN_CONVICTION", "1.2")))
        self._auto_trade_top_k = max(1, int(os.getenv("AUTO_TRADE_TOP_K", "72")))
        self._auto_trade_max_new_per_cycle = max(1, int(os.getenv("AUTO_TRADE_MAX_NEW_PER_CYCLE", "18")))
        self._auto_trade_max_open_positions = max(1, int(os.getenv("AUTO_TRADE_MAX_OPEN_POSITIONS", "36")))
        self._auto_trade_max_sector_positions = max(1, int(os.getenv("AUTO_TRADE_MAX_SECTOR_POSITIONS", "6")))
        self._auto_trade_cooldown_seconds = max(0, int(os.getenv("AUTO_TRADE_COOLDOWN_SECONDS", "120")))
        self._auto_trade_min_hold_seconds = max(0, int(os.getenv("AUTO_TRADE_MIN_HOLD_SECONDS", "300")))
        self._day_trade_force_exit_seconds = max(300, int(os.getenv("DAY_TRADE_FORCE_EXIT_SECONDS", "3600")))
        self._day_trade_min_tick_count = max(6, int(os.getenv("DAY_TRADE_MIN_TICK_COUNT", "10")))
        self._day_trade_intraday_score_threshold = max(
            0.10,
            float(os.getenv("DAY_TRADE_INTRADAY_SCORE_THRESHOLD", "0.18")),
        )
        self._day_trade_intraday_weight = min(
            0.95,
            max(0.20, float(os.getenv("DAY_TRADE_INTRADAY_WEIGHT", "0.70"))),
        )
        self._day_trade_news_boost_threshold = max(
            0.0,
            float(os.getenv("DAY_TRADE_NEWS_BOOST_THRESHOLD", "0.15")),
        )
        self._day_trade_risk_per_trade_pct = min(
            0.02,
            max(0.001, float(os.getenv("DAY_TRADE_RISK_PER_TRADE_PCT", "0.005"))),
        )
        self._day_trade_max_daily_loss_pct = min(
            0.10,
            max(0.005, float(os.getenv("DAY_TRADE_MAX_DAILY_LOSS_PCT", "0.02"))),
        )
        self._day_trade_max_consecutive_losses = max(
            1,
            int(os.getenv("DAY_TRADE_MAX_CONSECUTIVE_LOSSES", "3")),
        )
        self._day_trade_opening_range_minutes = max(
            5,
            int(os.getenv("DAY_TRADE_OPENING_RANGE_MINUTES", "15")),
        )
        self._day_trade_active_open_minutes = max(
            self._day_trade_opening_range_minutes,
            int(os.getenv("DAY_TRADE_ACTIVE_OPEN_MINUTES", "90")),
        )
        self._day_trade_power_hour_minutes = max(
            15,
            int(os.getenv("DAY_TRADE_POWER_HOUR_MINUTES", "60")),
        )
        self._day_trade_vwap_pullback_atr_mult = max(
            0.05,
            float(os.getenv("DAY_TRADE_VWAP_PULLBACK_ATR_MULT", "0.08")),
        )
        self._day_trade_ensemble_threshold = max(
            0.15,
            float(os.getenv("DAY_TRADE_ENSEMBLE_THRESHOLD", "0.42")),
        )
        self._day_trade_trail_stop_pct = max(
            0.0025,
            float(os.getenv("DAY_TRADE_TRAIL_STOP_PCT", "0.0065")),
        )
        self._auto_trade_base_position_pct = max(0.005, float(os.getenv("AUTO_TRADE_BASE_POSITION_PCT", "0.03")))
        self._auto_trade_max_position_pct = max(
            self._auto_trade_base_position_pct,
            float(os.getenv("AUTO_TRADE_MAX_POSITION_PCT", "0.08")),
        )
        self._lane_target_open_positions = {
            "normal": max(0, int(os.getenv("NORMAL_LANE_TARGET_OPEN_POSITIONS", os.getenv("LANE_TARGET_OPEN_POSITIONS", "150")))),
            "day": max(0, int(os.getenv("DAY_LANE_TARGET_OPEN_POSITIONS", os.getenv("LANE_TARGET_OPEN_POSITIONS", "150")))),
            "crypto": max(0, int(os.getenv("CRYPTO_LANE_TARGET_OPEN_POSITIONS", os.getenv("LANE_TARGET_OPEN_POSITIONS", "150")))),
        }
        self._auto_trade_leader_min_conviction = max(
            0.0, float(os.getenv("AUTO_TRADE_LEADER_MIN_CONVICTION", "0.8"))
        )
        self._auto_trade_leader_min_take_probability = min(
            0.99, max(0.0, float(os.getenv("AUTO_TRADE_LEADER_MIN_TAKE_PROBABILITY", "0.18")))
        )
        self._auto_trade_leader_min_rank_score = max(
            0.0, float(os.getenv("AUTO_TRADE_LEADER_MIN_RANK_SCORE", "0.12"))
        )
        self._auto_trade_zero_weight_fallback_enabled = (
            os.getenv("AUTO_TRADE_ZERO_WEIGHT_FALLBACK_ENABLED", "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self._auto_trade_zero_weight_min_take_probability = min(
            0.99,
            max(0.0, float(os.getenv("AUTO_TRADE_ZERO_WEIGHT_MIN_TAKE_PROBABILITY", "0.16"))),
        )
        self._auto_trade_zero_weight_min_rank_score = max(
            0.0, float(os.getenv("AUTO_TRADE_ZERO_WEIGHT_MIN_RANK_SCORE", "0.08"))
        )
        self._auto_trade_replace_margin = max(0.0, float(os.getenv("AUTO_TRADE_REPLACE_MARGIN", "0.02")))
        self._auto_trade_stress_position_mult = min(
            1.0,
            max(0.1, float(os.getenv("AUTO_TRADE_STRESS_POSITION_MULT", "0.6"))),
        )
        self._auto_trade_corr_lookback_days = max(20, int(os.getenv("AUTO_TRADE_CORR_LOOKBACK_DAYS", "60")))
        self._auto_trade_max_pair_correlation = min(
            0.98,
            max(0.10, float(os.getenv("AUTO_TRADE_MAX_PAIR_CORRELATION", "0.78"))),
        )
        self._auto_trade_target_avg_correlation = min(
            self._auto_trade_max_pair_correlation,
            max(0.05, float(os.getenv("AUTO_TRADE_TARGET_AVG_CORRELATION", "0.55"))),
        )
        self._auto_trade_correlation_size_floor = min(
            1.0,
            max(0.10, float(os.getenv("AUTO_TRADE_CORRELATION_SIZE_FLOOR", "0.35"))),
        )
        self._portfolio_optimizer_enabled = os.getenv("PORTFOLIO_OPTIMIZER_ENABLED", "1").strip().lower() not in {"0", "false", "off"}
        self._portfolio_optimizer_gross_target_pct = min(
            0.95,
            max(0.10, float(os.getenv("PORTFOLIO_OPTIMIZER_GROSS_TARGET_PCT", "0.65"))),
        )
        self._portfolio_optimizer_max_names = max(
            self._auto_trade_max_open_positions,
            int(os.getenv("PORTFOLIO_OPTIMIZER_MAX_NAMES", str(max(self._auto_trade_max_open_positions, 14)))),
        )
        self._portfolio_optimizer_min_weight = max(0.0, float(os.getenv("PORTFOLIO_OPTIMIZER_MIN_WEIGHT", "0.03")))
        self._portfolio_optimizer_max_weight = min(
            1.0,
            max(self._portfolio_optimizer_min_weight, float(os.getenv("PORTFOLIO_OPTIMIZER_MAX_WEIGHT", "0.16"))),
        )
        self._portfolio_optimizer_covariance_shrinkage = min(
            0.95,
            max(0.0, float(os.getenv("PORTFOLIO_OPTIMIZER_COV_SHRINKAGE", "0.25"))),
        )
        self._portfolio_optimizer_factor_penalty = max(
            0.0, float(os.getenv("PORTFOLIO_OPTIMIZER_FACTOR_PENALTY", "0.30"))
        )
        self._portfolio_optimizer_beta_penalty = max(
            0.0, float(os.getenv("PORTFOLIO_OPTIMIZER_BETA_PENALTY", "0.45"))
        )
        self._portfolio_optimizer_target_beta = float(os.getenv("PORTFOLIO_OPTIMIZER_TARGET_BETA", "0.10"))
        self._feature_store_enabled = os.getenv("FEATURE_STORE_ENABLED", "1").strip().lower() not in {"0", "false", "off"}
        self._feature_store_save_seconds = max(300, int(os.getenv("FEATURE_STORE_SAVE_SECONDS", "1800")))
        self._feature_store_dir = _resolve_root_path(os.getenv("FEATURE_STORE_DIR", "data/features"))
        self._inference_feature_chunk_size = max(25, int(os.getenv("INFERENCE_FEATURE_CHUNK_SIZE", "250") or 250))
        self._inference_idle_wait_seconds = max(5, int(os.getenv("INFERENCE_IDLE_WAIT_SECONDS", "15") or 15))
        bootstrap_file_cap_default = self._source_symbol_caps.get("prices", 0) or 600
        self._signal_bootstrap_feature_file_cap = max(
            0,
            int(os.getenv("SIGNAL_BOOTSTRAP_FEATURE_FILE_CAP", str(bootstrap_file_cap_default)) or bootstrap_file_cap_default),
        )
        self._event_intel_enabled = os.getenv("EVENT_INTEL_ENABLED", "1").strip().lower() not in {"0", "false", "off"}
        self._event_intel_dir = _resolve_root_path(os.getenv("EVENT_INTEL_DIR", "data/event_intel"))
        self._auto_retrain_on_start = os.getenv("AUTO_RETRAIN_ON_START", "1").strip().lower() not in {"0", "false", "off"}
        self._auto_retrain_only_if_new_data = os.getenv("AUTO_RETRAIN_ONLY_IF_NEW_DATA", "1").strip().lower() not in {"0", "false", "off"}
        self._auto_retrain_min_updated_files = max(1, int(os.getenv("AUTO_RETRAIN_MIN_UPDATED_FILES", "5")))
        self._auto_retrain_timeout_seconds = max(60, int(os.getenv("AUTO_RETRAIN_TIMEOUT_SECONDS", "1800")))
        self._auto_retrain_use_optuna = os.getenv("AUTO_RETRAIN_USE_OPTUNA", "0").strip().lower() in {"1", "true", "yes", "on"}
        self._execution_trace_limit = max(100, int(os.getenv("EXECUTION_TRACE_LIMIT", "500")))
        self._runtime_state_path = _resolve_root_path(os.getenv("RUNTIME_STATE_PATH", "data/runtime_state.json"))
        self._execution_trace_log_path = _resolve_root_path(os.getenv("EXECUTION_TRACE_LOG_PATH", "data/execution_trace.jsonl"))
        self._execution_reconciliation_log_path = _resolve_root_path(
            os.getenv("EXECUTION_RECONCILIATION_LOG_PATH", "data/execution_reconciliation.jsonl")
        )
        self._system_health_path = _resolve_root_path(os.getenv("SYSTEM_HEALTH_PATH", "data/system_health.json"))
        self._runtime_state_save_seconds = max(5, int(os.getenv("RUNTIME_STATE_SAVE_SECONDS", "30")))
        self._execution_backtest_enabled = os.getenv("EXECUTION_BACKTEST_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
        self._execution_backtest_interval_seconds = max(
            300, int(os.getenv("EXECUTION_BACKTEST_INTERVAL_SECONDS", "1800"))
        )
        self._execution_backtest_lookback_days = max(
            30, int(os.getenv("EXECUTION_BACKTEST_LOOKBACK_DAYS", "120"))
        )
        self._execution_backtest_symbol_limit = max(
            30, int(os.getenv("EXECUTION_BACKTEST_SYMBOL_LIMIT", "160"))
        )
        self._execution_backtest_entry_score = float(os.getenv("EXECUTION_BACKTEST_ENTRY_SCORE", "0.35"))
        self._execution_backtest_exit_score = float(os.getenv("EXECUTION_BACKTEST_EXIT_SCORE", "0.10"))
        self._execution_backtest_min_hold_days = max(1, int(os.getenv("EXECUTION_BACKTEST_MIN_HOLD_DAYS", "2")))
        self._execution_backtest_tx_cost_bps = max(0.0, float(os.getenv("EXECUTION_BACKTEST_TX_COST_BPS", "8")))
        self._execution_backtest_monte_carlo_runs = max(
            100,
            int(os.getenv("EXECUTION_BACKTEST_MONTE_CARLO_RUNS", "500")),
        )
        self._execution_backtest_plateau_delta = max(
            0.05,
            float(os.getenv("EXECUTION_BACKTEST_PLATEAU_DELTA", "0.3")),
        )
        self._alpha_quality_enabled = os.getenv("ALPHA_QUALITY_ENABLED", "1").strip().lower() not in {"0", "false", "off"}
        self._alpha_quality_lookback_days = max(
            45,
            int(os.getenv("ALPHA_QUALITY_LOOKBACK_DAYS", str(max(120, self._execution_backtest_lookback_days)))),
        )
        self._alpha_quality_horizon_days = max(1, int(os.getenv("ALPHA_QUALITY_HORIZON_DAYS", os.getenv("META_MODEL_HORIZON_DAYS", "5"))))
        self._alpha_quality_bucket_count = min(10, max(3, int(os.getenv("ALPHA_QUALITY_BUCKETS", "5"))))
        self._alpha_quality_min_bucket_samples = max(5, int(os.getenv("ALPHA_QUALITY_MIN_BUCKET_SAMPLES", "12")))
        self._alpha_quality_report_path = _resolve_root_path(os.getenv("ALPHA_QUALITY_REPORT_PATH", "data/alpha_quality.json"))
        self._broker_execution_mode = str(os.getenv("BROKER_EXECUTION_MODE", "paper") or "paper").strip().lower()
        if self._broker_execution_mode not in {"paper", "shadow"}:
            logger.warning(f"Unsupported BROKER_EXECUTION_MODE '{self._broker_execution_mode}', falling back to paper")
            self._broker_execution_mode = "paper"
        self._shadow_router_enabled = self._broker_execution_mode == "shadow"
        self._shadow_reconciliation_log_path = str(
            os.getenv("SHADOW_RECONCILIATION_LOG_PATH", "data/shadow_execution_log.jsonl")
        ).strip() or "data/shadow_execution_log.jsonl"
        if self._day_trading_mode:
            self._inference_refresh_seconds = max(10, int(os.getenv("INFERENCE_REFRESH_SECONDS", "15")))
            self._auto_trade_top_k = max(self._auto_trade_top_k, int(os.getenv("DAY_TRADE_TOP_K", "36")))
            self._auto_trade_max_new_per_cycle = max(
                self._auto_trade_max_new_per_cycle, int(os.getenv("DAY_TRADE_MAX_NEW_PER_CYCLE", "8"))
            )
            self._auto_trade_max_open_positions = max(
                self._auto_trade_max_open_positions, int(os.getenv("DAY_TRADE_MAX_OPEN_POSITIONS", "18"))
            )
            self._auto_trade_max_sector_positions = max(
                self._auto_trade_max_sector_positions, int(os.getenv("DAY_TRADE_MAX_SECTOR_POSITIONS", "5"))
            )
            self._auto_trade_cooldown_seconds = min(
                self._auto_trade_cooldown_seconds, int(os.getenv("DAY_TRADE_COOLDOWN_SECONDS", "240"))
            )
            self._auto_trade_min_hold_seconds = min(
                self._auto_trade_min_hold_seconds, int(os.getenv("DAY_TRADE_MIN_HOLD_SECONDS", "600"))
            )
            self._auto_trade_base_position_pct = min(
                self._auto_trade_base_position_pct, float(os.getenv("DAY_TRADE_BASE_POSITION_PCT", "0.02"))
            )
            self._auto_trade_max_position_pct = min(
                self._auto_trade_max_position_pct, float(os.getenv("DAY_TRADE_MAX_POSITION_PCT", "0.05"))
            )
            self._portfolio_optimizer_gross_target_pct = max(
                self._portfolio_optimizer_gross_target_pct,
                float(os.getenv("DAY_TRADE_GROSS_TARGET_PCT", "0.82")),
            )
            self._portfolio_optimizer_max_names = max(
                self._portfolio_optimizer_max_names, int(os.getenv("DAY_TRADE_MAX_ALLOC_NAMES", "18"))
            )
        self._lane_engine_order = ("crypto", "day", "normal")
        self._lane_base_allocations = {
            "normal": max(0.05, float(os.getenv("NORMAL_LANE_BASE_ALLOC_PCT", "0.40"))),
            "day": max(0.05, float(os.getenv("DAY_LANE_BASE_ALLOC_PCT", "0.35"))),
            "crypto": max(0.05, float(os.getenv("CRYPTO_LANE_BASE_ALLOC_PCT", "0.25"))),
        }
        self._lane_min_allocations = {
            "normal": max(0.05, float(os.getenv("NORMAL_LANE_MIN_ALLOC_PCT", "0.15"))),
            "day": max(0.05, float(os.getenv("DAY_LANE_MIN_ALLOC_PCT", "0.15"))),
            "crypto": max(0.05, float(os.getenv("CRYPTO_LANE_MIN_ALLOC_PCT", "0.10"))),
        }
        self._lane_max_allocations = {
            "normal": min(0.90, max(self._lane_min_allocations["normal"], float(os.getenv("NORMAL_LANE_MAX_ALLOC_PCT", "0.55")))),
            "day": min(0.90, max(self._lane_min_allocations["day"], float(os.getenv("DAY_LANE_MAX_ALLOC_PCT", "0.50")))),
            "crypto": min(0.90, max(self._lane_min_allocations["crypto"], float(os.getenv("CRYPTO_LANE_MAX_ALLOC_PCT", "0.40")))),
        }
        self._governor_lookback_days = max(7, int(os.getenv("GOVERNOR_LOOKBACK_DAYS", "30")))
        self._governor_min_setup_trades = max(3, int(os.getenv("GOVERNOR_MIN_SETUP_TRADES", "6")))
        self._governor_min_time_bucket_trades = max(3, int(os.getenv("GOVERNOR_MIN_TIME_BUCKET_TRADES", "5")))
        self._governor_setup_pf_floor = max(0.5, float(os.getenv("GOVERNOR_SETUP_PF_FLOOR", "0.95")))
        self._governor_setup_review_pf = max(self._governor_setup_pf_floor, float(os.getenv("GOVERNOR_SETUP_REVIEW_PF", "1.10")))
        self._governor_time_bucket_pf_floor = max(0.4, float(os.getenv("GOVERNOR_TIME_BUCKET_PF_FLOOR", "0.90")))
        self._governor_time_bucket_throttle_mult = min(
            1.0,
            max(0.25, float(os.getenv("GOVERNOR_TIME_BUCKET_THROTTLE_MULT", "0.55"))),
        )
        self._governor_lane_throttle_mult = min(
            1.0,
            max(0.25, float(os.getenv("GOVERNOR_LANE_THROTTLE_MULT", "0.75"))),
        )
        self._crypto_scalper_max_spread_pct = max(0.001, float(os.getenv("CRYPTO_SCALPER_MAX_SPREAD_PCT", "0.08")))
        self._crypto_scalper_max_depth_age_seconds = max(
            0.2,
            float(os.getenv("CRYPTO_SCALPER_MAX_DEPTH_AGE_SECONDS", "2.0")),
        )
        self._crypto_scalper_max_tick_age_seconds = max(
            0.2,
            float(os.getenv("CRYPTO_SCALPER_MAX_TICK_AGE_SECONDS", "2.0")),
        )
        self._crypto_scalper_max_signal_age_seconds = max(
            0.5,
            float(os.getenv("CRYPTO_SCALPER_MAX_SIGNAL_AGE_SECONDS", "3.0")),
        )
        self._crypto_scalper_min_book_pressure = max(
            0.0,
            float(os.getenv("CRYPTO_SCALPER_MIN_BOOK_PRESSURE", "0.08")),
        )
        self._crypto_scalper_min_depth_imbalance = max(
            0.0,
            float(os.getenv("CRYPTO_SCALPER_MIN_DEPTH_IMBALANCE", "0.06")),
        )
        self._crypto_scalper_min_take_probability = min(
            0.99,
            max(0.0, float(os.getenv("CRYPTO_SCALPER_MIN_TAKE_PROBABILITY", "0.32"))),
        )
        self._crypto_quote_fallback_enabled = os.getenv("CRYPTO_QUOTE_FALLBACK_ENABLED", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._crypto_quote_fallback_spread_pct = min(
            max(self._crypto_scalper_max_spread_pct * 0.25, 0.005),
            max(0.005, float(os.getenv("CRYPTO_QUOTE_FALLBACK_SPREAD_PCT", "0.02"))),
        )
        self._crypto_obi_min_lifetime_ms = max(50.0, float(os.getenv("CRYPTO_OBI_MIN_LIFETIME_MS", "200")))
        self._crypto_obi_max_updates_per_second = max(
            0.5,
            float(os.getenv("CRYPTO_OBI_MAX_UPDATES_PER_SECOND", "5.0")),
        )
        self._crypto_obi_min_inter_update_delay_ms = max(
            0.0,
            float(os.getenv("CRYPTO_OBI_MIN_INTER_UPDATE_DELAY_MS", "20.0")),
        )
        self._crypto_prime_start_hour_utc = int(os.getenv("CRYPTO_PRIME_WINDOW_START_UTC", "9"))
        self._crypto_prime_end_hour_utc = int(os.getenv("CRYPTO_PRIME_WINDOW_END_UTC", "14"))
        self._crypto_prime_conviction_bias = float(os.getenv("CRYPTO_PRIME_MIN_CONVICTION_BIAS", "-0.2"))
        self._crypto_liquidity_hole_conviction_bias = float(os.getenv("CRYPTO_LIQUIDITY_HOLE_MIN_CONVICTION_BIAS", "0.3"))
        self._crypto_taker_book_pressure_threshold = max(
            0.0,
            float(os.getenv("CRYPTO_TAKER_BOOK_PRESSURE_THRESHOLD", "0.65")),
        )
        self._crypto_taker_spread_velocity_threshold = float(
            os.getenv("CRYPTO_TAKER_SPREAD_VELOCITY_THRESHOLD", "0.0")
        )
        self._crypto_inventory_skew_trigger = min(
            1.0,
            max(0.0, float(os.getenv("CRYPTO_INVENTORY_SKEW_TRIGGER", "0.60"))),
        )
        self._lane_engine_config = {
            "normal": {
                "top_k": max(1, int(os.getenv("NORMAL_LANE_TOP_K", "96"))),
                "max_new_per_cycle": max(1, int(os.getenv("NORMAL_LANE_MAX_NEW_PER_CYCLE", "20"))),
                "max_open_positions": max(1, int(os.getenv("NORMAL_LANE_MAX_OPEN_POSITIONS", "40"))),
                "cooldown_seconds": max(60, int(os.getenv("NORMAL_LANE_COOLDOWN_SECONDS", "300"))),
                "min_hold_seconds": max(300, int(os.getenv("NORMAL_LANE_MIN_HOLD_SECONDS", "1800"))),
                "force_exit_seconds": 0,
                "risk_per_trade_pct": min(0.02, max(0.001, float(os.getenv("NORMAL_LANE_RISK_PER_TRADE_PCT", "0.0075")))),
                "base_position_pct": max(0.005, float(os.getenv("NORMAL_LANE_BASE_POSITION_PCT", "0.04"))),
                "max_position_pct": max(0.01, float(os.getenv("NORMAL_LANE_MAX_POSITION_PCT", "0.10"))),
                "min_conviction": max(0.0, float(os.getenv("NORMAL_LANE_MIN_CONVICTION", "1.0"))),
                "min_take_probability": min(0.99, max(0.0, float(os.getenv("NORMAL_LANE_MIN_TAKE_PROBABILITY", "0.18")))),
                "sector_cap": max(1, int(os.getenv("NORMAL_LANE_MAX_SECTOR_POSITIONS", "12"))),
                "pair_corr_cap": min(0.99, max(0.10, float(os.getenv("NORMAL_LANE_MAX_PAIR_CORRELATION", "0.97")))),
            },
            "day": {
                "top_k": max(1, int(os.getenv("DAY_LANE_TOP_K", str(self._auto_trade_top_k)))),
                "max_new_per_cycle": max(1, int(os.getenv("DAY_LANE_MAX_NEW_PER_CYCLE", str(self._auto_trade_max_new_per_cycle)))),
                "max_open_positions": max(1, int(os.getenv("DAY_LANE_MAX_OPEN_POSITIONS", str(self._auto_trade_max_open_positions)))),
                "cooldown_seconds": max(15, int(os.getenv("DAY_LANE_COOLDOWN_SECONDS", str(self._auto_trade_cooldown_seconds)))),
                "min_hold_seconds": max(30, int(os.getenv("DAY_LANE_MIN_HOLD_SECONDS", str(self._auto_trade_min_hold_seconds)))),
                "force_exit_seconds": max(0, int(os.getenv("DAY_LANE_FORCE_EXIT_SECONDS", str(self._day_trade_force_exit_seconds)))),
                "risk_per_trade_pct": self._day_trade_risk_per_trade_pct,
                "base_position_pct": max(0.005, float(os.getenv("DAY_LANE_BASE_POSITION_PCT", str(self._auto_trade_base_position_pct)))),
                "max_position_pct": max(0.005, float(os.getenv("DAY_LANE_MAX_POSITION_PCT", str(self._auto_trade_max_position_pct)))),
                "min_conviction": max(0.0, float(os.getenv("DAY_LANE_MIN_CONVICTION", "0.75"))),
                "min_take_probability": min(0.99, max(0.0, float(os.getenv("DAY_LANE_MIN_TAKE_PROBABILITY", "0.14")))),
                "sector_cap": max(1, int(os.getenv("DAY_LANE_MAX_SECTOR_POSITIONS", str(self._auto_trade_max_sector_positions + 4)))),
                "pair_corr_cap": min(0.99, max(0.10, float(os.getenv("DAY_LANE_MAX_PAIR_CORRELATION", "0.98")))),
            },
            "crypto": {
                "top_k": max(1, int(os.getenv("CRYPTO_LANE_TOP_K", "32"))),
                "max_new_per_cycle": max(1, int(os.getenv("CRYPTO_LANE_MAX_NEW_PER_CYCLE", "12"))),
                "max_open_positions": max(1, int(os.getenv("CRYPTO_LANE_MAX_OPEN_POSITIONS", "16"))),
                "cooldown_seconds": max(5, int(os.getenv("CRYPTO_LANE_COOLDOWN_SECONDS", "15"))),
                "min_hold_seconds": max(15, int(os.getenv("CRYPTO_LANE_MIN_HOLD_SECONDS", "45"))),
                "force_exit_seconds": max(0, int(os.getenv("CRYPTO_LANE_FORCE_EXIT_SECONDS", "1800"))),
                "risk_per_trade_pct": min(0.02, max(0.001, float(os.getenv("CRYPTO_LANE_RISK_PER_TRADE_PCT", "0.004")))),
                "base_position_pct": max(0.0025, float(os.getenv("CRYPTO_LANE_BASE_POSITION_PCT", "0.025"))),
                "max_position_pct": max(0.005, float(os.getenv("CRYPTO_LANE_MAX_POSITION_PCT", "0.05"))),
                "min_conviction": max(0.0, float(os.getenv("CRYPTO_LANE_MIN_CONVICTION", "0.5"))),
                "min_take_probability": self._crypto_scalper_min_take_probability,
                "sector_cap": max(1, int(os.getenv("CRYPTO_LANE_MAX_SECTOR_POSITIONS", "12"))),
                "pair_corr_cap": min(0.99, max(0.10, float(os.getenv("CRYPTO_LANE_MAX_PAIR_CORRELATION", "0.98")))),
            },
        }
        for lane_name, target_value in self._lane_target_open_positions.items():
            if lane_name in self._lane_engine_config:
                self._lane_engine_config[lane_name]["target_open_positions"] = int(target_value)
        if self._throughput_mode:
            self._apply_throughput_mode()
        self._last_feature_signature: Dict[str, str] = {}
        self._feature_store_bootstrap_attempted = False
        self._crypto_last_signal_ts: Dict[str, float] = {}
        self._position_plans: Dict[str, Dict] = {}
        self._last_trade_timestamps: Dict[str, float] = {}
        self._last_execution_backtest_ts = 0.0
        self._execution_backtest_running = False
        self._last_feature_store_save_ts = 0.0
        self._last_runtime_state_save_ts = 0.0
        self._execution_trace: List[Dict] = []
        self._execution_reconciliation: List[Dict] = []
        self._intraday_state: Dict[str, Dict] = {}
        self._intraday_state_lock = threading.RLock()
        self._intraday_subscription_attached = False
        self._last_model_learning_ts = 0.0
        self._last_emergency_retrain_ts = 0.0
        self._model_learning_in_progress = False
        self._runtime_refresh_models = False
        self._components["learning_status"] = {}
        self._components["lane_allocator"] = {}
        self._components["governor"] = {}
        self._components["drift_status"] = {}
        self._components["signal_decay_library"] = {}
        self._components["execution_overfit"] = {}
        self._components["latest_feature_rows"] = {}
        self._components["execution_reconciliation"] = self._execution_reconciliation
        self._components["feature_matrices"] = {}
        self._components["event_feature_map"] = {}
        self._components["data_versions"] = dict(self._data_versions)
        if self._event_window_mode:
            self._apply_event_window_mode()
        self._data_health = self._seed_data_health()
        self._system_health: Dict[str, Any] = {}
        self._components["data_health"] = self._data_health
        self._components["system_health"] = self._system_health
        self._components["data_pipeline_status"] = self._data_pipeline_status
        self._load_runtime_state()
        self._refresh_system_health()

    def _source_refresh_seconds(self, source: str) -> int:
        return {
            "prices": self._price_refresh_seconds,
            "sentiment": self._sentiment_refresh_seconds,
            "earnings": self._earnings_refresh_seconds,
            "altdata": self._altdata_refresh_seconds,
        }.get(str(source or "").lower(), 0)

    def _source_stale_after_seconds(self, source: str) -> float:
        refresh_seconds = float(self._source_refresh_seconds(source) or 0.0)
        if refresh_seconds <= 0:
            return 0.0
        return max(refresh_seconds, refresh_seconds * self._entry_stale_multiplier)

    def _seed_data_health(self) -> Dict[str, Dict[str, Any]]:
        seeded: Dict[str, Dict[str, Any]] = {}
        for source in ("prices", "sentiment", "earnings", "altdata"):
            seeded[source] = {
                "updated_at": None,
                "updated_at_ts": 0.0,
                "refresh_seconds": self._source_refresh_seconds(source),
                "stale_after_seconds": self._source_stale_after_seconds(source),
                "status": "missing",
            }
        return seeded

    def _append_jsonl(self, path: Path, row: Dict[str, Any], *, log_label: str) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, default=str) + "\n")
        except Exception as exc:
            logger.warning(f"{log_label} save failed: {exc}")

    def _serialize_runtime_state(self) -> Dict[str, Any]:
        return {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "signal_store": self._signal_store,
            "position_plans": self._position_plans,
            "last_trade_timestamps": self._last_trade_timestamps,
            "execution_trace": self._execution_trace[-self._execution_trace_limit :],
            "execution_reconciliation": self._execution_reconciliation[-self._execution_trace_limit :],
            "data_health": self._data_health,
            "system_health": self._system_health,
        }

    def _persist_runtime_state(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_runtime_state_save_ts) < self._runtime_state_save_seconds:
            return
        self._refresh_system_health()
        payload = self._serialize_runtime_state()
        try:
            self._runtime_state_path.parent.mkdir(parents=True, exist_ok=True)
            self._runtime_state_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            self._system_health_path.parent.mkdir(parents=True, exist_ok=True)
            self._system_health_path.write_text(json.dumps(self._system_health, indent=2, default=str), encoding="utf-8")
            self._last_runtime_state_save_ts = now
        except Exception as exc:
            logger.warning(f"Runtime state save failed: {exc}")

    def _load_runtime_state(self) -> None:
        if not self._runtime_state_path.exists():
            return
        try:
            payload = json.loads(self._runtime_state_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"Runtime state load failed: {exc}")
            return

        self._execution_trace.clear()
        self._execution_trace.extend(
            [dict(row) for row in (payload.get("execution_trace") or []) if isinstance(row, dict)][-self._execution_trace_limit :]
        )
        self._signal_store.clear()
        for symbol_key, signal in (payload.get("signal_store") or {}).items():
            if not isinstance(signal, dict):
                continue
            symbol = str(signal.get("symbol") or symbol_key or "").strip().upper()
            if not symbol:
                continue
            restored_signal = self._mark_signal_warmup_only(symbol, signal, reason="runtime_restore")
            normal_lane = restored_signal.get("normal_lane_signal")
            if isinstance(normal_lane, dict):
                restored_signal["normal_lane_signal"] = self._mark_signal_warmup_only(
                    symbol,
                    normal_lane,
                    reason="runtime_restore",
                    lane_override="normal",
                )
            self._signal_store[symbol] = restored_signal
        self._execution_reconciliation.clear()
        self._execution_reconciliation.extend(
            [dict(row) for row in (payload.get("execution_reconciliation") or []) if isinstance(row, dict)][-self._execution_trace_limit :]
        )
        self._position_plans.clear()
        for position_key, plan in (payload.get("position_plans") or {}).items():
            if isinstance(plan, dict):
                self._position_plans[str(position_key)] = dict(plan)
        self._last_trade_timestamps.clear()
        for lane_key, value in (payload.get("last_trade_timestamps") or {}).items():
            try:
                self._last_trade_timestamps[str(lane_key)] = float(value)
            except Exception:
                continue

        restored_health = self._seed_data_health()
        loaded_health = payload.get("data_health") or {}
        if isinstance(loaded_health, dict):
            for source in restored_health:
                if isinstance(loaded_health.get(source), dict):
                    restored_health[source].update(loaded_health[source])
                restored_health[source]["refresh_seconds"] = self._source_refresh_seconds(source)
                restored_health[source]["stale_after_seconds"] = self._source_stale_after_seconds(source)
        self._data_health.clear()
        self._data_health.update(restored_health)

        self._system_health.clear()
        if isinstance(payload.get("system_health"), dict):
            self._system_health.update(payload["system_health"])

        logger.info(
            "Runtime state restored: %s signals | %s plans | %s execution events | %s reconciliations",
            len(self._signal_store),
            len(self._position_plans),
            len(self._execution_trace),
            len(self._execution_reconciliation),
        )

    def _mark_signal_warmup_only(
        self,
        symbol: str,
        signal: Optional[Dict[str, Any]],
        *,
        reason: str,
        lane_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = self._decorate_signal(str(symbol or "").upper(), signal or {}, lane_override=lane_override)
        payload["warmup_only"] = True
        payload["trade_eligible"] = False
        payload["stale_reason"] = reason
        payload["stale_seconds"] = None
        meta = payload.get("meta_decision")
        if isinstance(meta, dict):
            meta = dict(meta)
            meta["take_trade"] = False
            meta.setdefault("reason", reason)
            payload["meta_decision"] = meta
        return payload

    def _select_feature_store_bootstrap_paths(self) -> List[Tuple[str, Path]]:
        if self._feature_store_bootstrap_attempted:
            return []
        self._feature_store_bootstrap_attempted = True
        if not self._feature_store_enabled or not self._feature_store_dir.exists():
            return []

        feature_paths = {path.stem.upper(): path for path in self._feature_store_dir.glob("*.parquet")}
        if not feature_paths:
            return []

        ordered_symbols = [symbol for symbol in self._get_target_symbols() if symbol in feature_paths]
        if not ordered_symbols:
            ordered_symbols = sorted(
                feature_paths.keys(),
                key=lambda symbol: feature_paths[symbol].stat().st_mtime,
                reverse=True,
            )

        cap = self._signal_bootstrap_feature_file_cap
        if cap > 0:
            ordered_symbols = ordered_symbols[:cap]

        selected_paths = [(symbol, feature_paths[symbol]) for symbol in ordered_symbols if symbol in feature_paths]
        if selected_paths:
            logger.info(
                "Signal bootstrap queued %s feature files from %s",
                len(selected_paths),
                self._feature_store_dir,
            )
        return selected_paths

    def _sync_runtime_state_with_broker(self, broker) -> None:
        if broker is None:
            return
        positions = getattr(broker, "positions", {}) or {}
        open_position_keys = {
            str(position_key)
            for position_key, position in positions.items()
            if float(getattr(position, "quantity", 0.0) or 0.0) > 0.0
        }
        stale_plan_keys = [position_key for position_key in self._position_plans.keys() if position_key not in open_position_keys]
        for position_key in stale_plan_keys:
            self._position_plans.pop(position_key, None)

        restored_count = 0
        for position_key, position in positions.items():
            if float(getattr(position, "quantity", 0.0) or 0.0) <= 0.0:
                continue
            resolved_key = str(position_key)
            if resolved_key in self._position_plans:
                continue
            symbol = str(getattr(position, "symbol", resolved_key) or resolved_key)
            lane = self._lane_from_position_key(resolved_key)
            self._position_plans[resolved_key] = {
                "symbol": symbol,
                "signal_key": resolved_key,
                "lane": lane,
                "market": self._market_code_for_symbol(symbol),
                "entry_reason": "restored_state",
                "min_hold_seconds": int(self._lane_config(lane).get("min_hold_seconds", self._auto_trade_min_hold_seconds)),
                "stop_loss_pct": 2.0,
                "take_profit_pct": 5.0,
                "trailing_stop_pct": float(self._day_trade_trail_stop_pct * 100.0),
                "peak_price": float(getattr(position, "avg_cost", 0.0) or 0.0),
                "restored_from_broker_state": True,
            }
            restored_count += 1

        if stale_plan_keys or restored_count:
            logger.info(
                "Runtime broker sync: restored %s open-position plans | pruned %s stale plans",
                restored_count,
                len(stale_plan_keys),
            )

    def _record_source_refresh(self, source: str, payload: Optional[Dict[str, Any]]) -> None:
        source_key = str(source or "").lower()
        snapshot = self._data_health.get(source_key, {})
        entry = {
            "updated_at_ts": time.time(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "refresh_seconds": self._source_refresh_seconds(source_key),
            "stale_after_seconds": self._source_stale_after_seconds(source_key),
            "status": "ok",
        }
        if source_key == "prices":
            recent = (payload or {}).get("price_daily_recent", {}) if isinstance(payload, dict) else {}
            entry["symbols_covered"] = len(recent) if isinstance(recent, dict) else 0
        elif source_key == "sentiment":
            if isinstance(payload, dict):
                entry["symbols_covered"] = int(payload.get("symbols_covered", 0) or len(payload.get("headlines", {}) or {}))
                entry["provider_counts"] = dict((payload.get("news_summary", {}) or {}).get("provider_counts", {}) or {})
        elif source_key == "earnings":
            latest_by_symbol = (payload or {}).get("latest_earnings_by_symbol") if isinstance(payload, dict) else None
            earnings_events = (payload or {}).get("earnings_events") if isinstance(payload, dict) else None
            if isinstance(latest_by_symbol, dict):
                entry["symbols_covered"] = len(latest_by_symbol)
            elif hasattr(earnings_events, "empty"):
                try:
                    entry["symbols_covered"] = int(earnings_events["symbol"].nunique()) if not earnings_events.empty else 0
                except Exception:
                    entry["symbols_covered"] = 0
        elif source_key == "altdata":
            summary = (payload or {}).get("summary") if isinstance(payload, dict) else None
            if isinstance(summary, dict):
                entry["symbols_covered"] = int(summary.get("symbols_covered", 0) or 0)
        snapshot.update(entry)
        self._data_health[source_key] = snapshot

    def _source_health_snapshot(self, source: str, *, as_of_ts: Optional[float] = None) -> Dict[str, Any]:
        source_key = str(source or "").lower()
        snapshot = dict(self._data_health.get(source_key) or {})
        now_ts = float(as_of_ts if as_of_ts is not None else time.time())
        updated_at_ts = self._safe_number(snapshot.get("updated_at_ts"), 0.0)
        age_seconds = round(max(0.0, now_ts - updated_at_ts), 2) if updated_at_ts > 0 else None
        stale_after_seconds = self._source_stale_after_seconds(source_key)
        status = "missing"
        if updated_at_ts > 0:
            status = "ok" if age_seconds is not None and age_seconds <= stale_after_seconds else "stale"
        snapshot.update(
            {
                "source": source_key,
                "age_seconds": age_seconds,
                "refresh_seconds": self._source_refresh_seconds(source_key),
                "stale_after_seconds": stale_after_seconds,
                "status": status,
            }
        )
        return snapshot

    def _market_is_open(self, market: str, *, timestamp: Optional[datetime] = None) -> bool:
        if str(market or "").upper() == "CRYPTO":
            return True
        current_ts = timestamp if isinstance(timestamp, datetime) else datetime.now(timezone.utc)
        local_ts, open_local, close_local = self._market_session_clock(current_ts, market)
        return open_local <= local_ts <= close_local

    def _entry_readiness_gate(self, symbol: str, lane: str, signal: Optional[Dict] = None) -> Dict[str, Any]:
        lane_key = str(lane or "normal").lower()
        if not self._entry_readiness_enabled:
            return {"allow": True, "reason": "disabled", "lane": lane_key}

        resolved_signal = signal if isinstance(signal, dict) else {}
        market = str(resolved_signal.get("market") or self._market_code_for_symbol(symbol)).upper()
        reasons: List[str] = []
        prices_snapshot = self._source_health_snapshot("prices")
        sentiment_snapshot = self._source_health_snapshot("sentiment")
        earnings_snapshot = self._source_health_snapshot("earnings")
        altdata_snapshot = self._source_health_snapshot("altdata")

        if prices_snapshot.get("status") != "ok":
            reasons.append("price_data_stale")
        if lane_key in {"normal", "day"}:
            if sentiment_snapshot.get("status") != "ok":
                reasons.append("sentiment_missing" if sentiment_snapshot.get("status") == "missing" else "sentiment_stale")
            if earnings_snapshot.get("status") != "ok":
                reasons.append("earnings_missing" if earnings_snapshot.get("status") == "missing" else "earnings_stale")

        if self._entry_require_market_open and market != "CRYPTO" and not self._market_is_open(market):
            reasons.append("market_closed")

        live_required = (
            (lane_key == "normal" and self._entry_require_live_price_normal)
            or (lane_key == "day" and self._entry_require_live_price_day)
            or (lane_key == "crypto" and self._entry_require_live_price_crypto)
        )
        live_tick = self._latest_live_tick(symbol) if live_required else None
        live_tick_age_seconds = None
        live_source = ""
        if live_required:
            if live_tick is None:
                reasons.append("no_fresh_live_price")
            else:
                tick_ts = getattr(live_tick, "timestamp", None)
                if isinstance(tick_ts, datetime):
                    live_tick_age_seconds = round(
                        max(0.0, (datetime.now(timezone.utc) - tick_ts.astimezone(timezone.utc)).total_seconds()),
                        2,
                    )
                live_source = str(getattr(live_tick, "source", "") or "")

        return {
            "allow": len(reasons) == 0,
            "reason": reasons[0] if reasons else "ready",
            "reasons": reasons,
            "lane": lane_key,
            "market": market,
            "live_price_required": live_required,
            "live_tick_age_seconds": live_tick_age_seconds,
            "live_source": live_source,
            "sources": {
                "prices": prices_snapshot.get("status"),
                "sentiment": sentiment_snapshot.get("status"),
                "earnings": earnings_snapshot.get("status"),
                "altdata": altdata_snapshot.get("status"),
            },
        }

    def _refresh_system_health(self) -> None:
        price_buffer_stats: Dict[str, Any] = {}
        active_symbols = 0
        try:
            from core.realtime_engine import PRICE_BUFFER

            price_buffer_stats = dict(getattr(PRICE_BUFFER, "stats", {}) or {})
            active_symbols = len(PRICE_BUFFER.active_symbols())
        except Exception:
            price_buffer_stats = {}
            active_symbols = 0

        broker = self._components.get("broker")
        open_positions = sum(
            1
            for position in (getattr(broker, "positions", {}) or {}).values()
            if float(getattr(position, "quantity", 0.0) or 0.0) > 0.0
        )
        lane_counts = {lane: 0 for lane in self._lane_engine_order}
        if broker is not None:
            for _, position, _, lane, _ in self._iter_open_positions(broker):
                if float(getattr(position, "quantity", 0.0) or 0.0) > 0.0:
                    lane_counts[lane] = lane_counts.get(lane, 0) + 1
        source_snapshots = {source: self._source_health_snapshot(source) for source in ("prices", "sentiment", "earnings", "altdata")}
        blocking_reasons = [
            reason
            for reason, source in (
                ("price_data_stale", "prices"),
                ("sentiment_stale", "sentiment"),
                ("earnings_stale", "earnings"),
            )
            if source_snapshots[source].get("status") != "ok"
        ]
        self._system_health.clear()
        self._system_health.update(
            {
                "status": "ok" if not blocking_reasons else "blocked",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "entry_readiness_enabled": self._entry_readiness_enabled,
                "new_entries_enabled": self._entry_readiness_enabled and not blocking_reasons,
                "blocking_reasons": blocking_reasons,
                "data_sources": source_snapshots,
                "runtime": {
                    "signal_count": len(self._signal_store),
                    "open_positions": open_positions,
                    "lane_open_counts": lane_counts,
                    "lane_targets": {
                        lane: int((self._lane_engine_config.get(lane, {}) or {}).get("target_open_positions", 0) or 0)
                        for lane in self._lane_engine_order
                    },
                    "execution_trace_count": len(self._execution_trace),
                    "execution_reconciliation_count": len(self._execution_reconciliation),
                    "tick_count": int(price_buffer_stats.get("total_ticks", 0) or 0),
                    "active_symbols": active_symbols,
                    "universe_sync": dict(self._components.get("universe_sync", {}) or {}),
                    "pipeline_selection": dict(self._source_selection_summary),
                    "data_pipeline": dict(self._data_pipeline_status),
                },
            }
        )

    def _sync_official_universes(self, force: bool = False) -> Dict[str, Any]:
        try:
            from pipeline.universe_sync import sync_official_universes

            summary = sync_official_universes(force=force)
            if isinstance(summary, dict):
                self._components["universe_sync"] = dict(summary)
                status = str(summary.get("status") or "unknown")
                if status in {"ok", "fresh"}:
                    logger.info(
                        "Official universe sync %s | US=%s NSE=%s crypto=%s shared_crypto=%s",
                        status,
                        summary.get("us_symbols", 0),
                        summary.get("nse_symbols", 0),
                        summary.get("crypto_symbols", 0),
                        summary.get("shared_crypto_symbols", 0),
                    )
                elif status not in {"disabled"}:
                    logger.warning(f"Official universe sync returned status={status}: {summary}")
                return summary
        except Exception as exc:
            logger.warning(f"Official universe sync failed: {exc}")
        summary = {"status": "error"}
        self._components["universe_sync"] = summary
        return summary

    def _get_target_symbols(self, market: str = "full") -> List[str]:
        from pipeline.universe import get_universe

        mode = self._universe_mode
        if market == "us":
            if mode in {"daytrade", "daytrade_us"}:
                return get_universe("daytrade_us")
            if mode in {"throughput", "throughput_us"}:
                return get_universe("throughput_us")
            if mode == "daytrade_nse":
                return []
            if mode == "throughput_nse":
                return []
            return get_universe("us") if mode != "core" else [s for s in get_universe("core") if not s.endswith(".NS")]
        if market == "nse":
            if mode in {"daytrade", "daytrade_nse"}:
                return get_universe("daytrade_nse")
            if mode in {"throughput", "throughput_nse"}:
                return get_universe("throughput_nse")
            if mode == "daytrade_us":
                return []
            if mode == "throughput_us":
                return []
            return get_universe("nse") if mode != "core" else [s for s in get_universe("core") if s.endswith(".NS")]
        if mode in {"core", "full", "us", "nse", "daytrade", "daytrade_us", "daytrade_nse", "throughput", "throughput_us", "throughput_nse"}:
            return get_universe(mode)
        return get_universe("full")

    def _apply_throughput_mode(self) -> None:
        self._allow_dual_lane_variants = True
        self._auto_trade_top_k = max(self._auto_trade_top_k, int(os.getenv("THROUGHPUT_TOP_K", "480")))
        self._auto_trade_max_new_per_cycle = max(
            self._auto_trade_max_new_per_cycle,
            int(os.getenv("THROUGHPUT_MAX_NEW_PER_CYCLE", "120")),
        )
        self._auto_trade_max_open_positions = max(
            self._auto_trade_max_open_positions,
            int(os.getenv("THROUGHPUT_MAX_OPEN_POSITIONS", "520")),
        )
        self._auto_trade_max_sector_positions = max(
            self._auto_trade_max_sector_positions,
            int(os.getenv("THROUGHPUT_MAX_SECTOR_POSITIONS", "80")),
        )
        self._portfolio_optimizer_gross_target_pct = max(
            self._portfolio_optimizer_gross_target_pct,
            float(os.getenv("THROUGHPUT_GROSS_TARGET_PCT", "0.92")),
        )
        self._portfolio_optimizer_max_names = max(
            self._portfolio_optimizer_max_names,
            int(os.getenv("THROUGHPUT_MAX_ALLOC_NAMES", "600")),
        )
        self._portfolio_optimizer_min_weight = min(
            self._portfolio_optimizer_min_weight,
            float(os.getenv("THROUGHPUT_MIN_WEIGHT", "0.0015")),
        )
        self._portfolio_optimizer_max_weight = min(
            self._portfolio_optimizer_max_weight,
            max(self._portfolio_optimizer_min_weight, float(os.getenv("THROUGHPUT_MAX_WEIGHT", "0.03"))),
        )
        self._lane_base_allocations.update(
            {
                "normal": max(self._lane_base_allocations["normal"], float(os.getenv("THROUGHPUT_NORMAL_BASE_ALLOC_PCT", "0.34"))),
                "day": max(self._lane_base_allocations["day"], float(os.getenv("THROUGHPUT_DAY_BASE_ALLOC_PCT", "0.33"))),
                "crypto": max(self._lane_base_allocations["crypto"], float(os.getenv("THROUGHPUT_CRYPTO_BASE_ALLOC_PCT", "0.33"))),
            }
        )
        self._lane_min_allocations.update(
            {
                "normal": max(self._lane_min_allocations["normal"], float(os.getenv("THROUGHPUT_NORMAL_MIN_ALLOC_PCT", "0.20"))),
                "day": max(self._lane_min_allocations["day"], float(os.getenv("THROUGHPUT_DAY_MIN_ALLOC_PCT", "0.20"))),
                "crypto": max(self._lane_min_allocations["crypto"], float(os.getenv("THROUGHPUT_CRYPTO_MIN_ALLOC_PCT", "0.20"))),
            }
        )
        self._lane_max_allocations.update(
            {
                "normal": max(self._lane_max_allocations["normal"], float(os.getenv("THROUGHPUT_NORMAL_MAX_ALLOC_PCT", "0.45"))),
                "day": max(self._lane_max_allocations["day"], float(os.getenv("THROUGHPUT_DAY_MAX_ALLOC_PCT", "0.40"))),
                "crypto": max(self._lane_max_allocations["crypto"], float(os.getenv("THROUGHPUT_CRYPTO_MAX_ALLOC_PCT", "0.40"))),
            }
        )

        throughput_lane_defaults = {
            "normal": {"top_k": 520, "max_new_per_cycle": 140, "max_open_positions": 220, "sector_cap": 60, "pair_corr_cap": 0.995},
            "day": {"top_k": 520, "max_new_per_cycle": 160, "max_open_positions": 220, "sector_cap": 70, "pair_corr_cap": 0.997},
            "crypto": {"top_k": 240, "max_new_per_cycle": 120, "max_open_positions": 180, "sector_cap": 220, "pair_corr_cap": 0.999},
        }
        for lane_name, defaults in throughput_lane_defaults.items():
            lane_cfg = self._lane_engine_config.get(lane_name, {})
            env_prefix = lane_name.upper()
            lane_cfg["top_k"] = max(
                lane_cfg.get("top_k", 1),
                int(os.getenv(f"THROUGHPUT_{env_prefix}_LANE_TOP_K", str(defaults["top_k"]))),
            )
            lane_cfg["max_new_per_cycle"] = max(
                lane_cfg.get("max_new_per_cycle", 1),
                int(os.getenv(f"THROUGHPUT_{env_prefix}_LANE_MAX_NEW_PER_CYCLE", str(defaults["max_new_per_cycle"]))),
            )
            lane_cfg["max_open_positions"] = max(
                lane_cfg.get("max_open_positions", 1),
                int(os.getenv(f"THROUGHPUT_{env_prefix}_LANE_MAX_OPEN_POSITIONS", str(defaults["max_open_positions"]))),
            )
            lane_cfg["sector_cap"] = max(
                lane_cfg.get("sector_cap", 1),
                int(os.getenv(f"THROUGHPUT_{env_prefix}_LANE_MAX_SECTOR_POSITIONS", str(defaults["sector_cap"]))),
            )
            lane_cfg["pair_corr_cap"] = max(
                float(lane_cfg.get("pair_corr_cap", 0.10) or 0.10),
                float(os.getenv(f"THROUGHPUT_{env_prefix}_LANE_MAX_PAIR_CORRELATION", str(defaults["pair_corr_cap"]))),
            )
            lane_cfg["target_open_positions"] = max(
                int(lane_cfg.get("target_open_positions", 0) or 0),
                int(self._lane_target_open_positions.get(lane_name, 0)),
            )
            if lane_name == "normal":
                lane_cfg["base_position_pct"] = min(
                    float(lane_cfg.get("base_position_pct", 0.04) or 0.04),
                    float(os.getenv("THROUGHPUT_NORMAL_BASE_POSITION_PCT", "0.006")),
                )
                lane_cfg["max_position_pct"] = min(
                    float(lane_cfg.get("max_position_pct", 0.10) or 0.10),
                    float(os.getenv("THROUGHPUT_NORMAL_MAX_POSITION_PCT", "0.02")),
                )
            elif lane_name == "day":
                lane_cfg["base_position_pct"] = min(
                    float(lane_cfg.get("base_position_pct", 0.02) or 0.02),
                    float(os.getenv("THROUGHPUT_DAY_BASE_POSITION_PCT", "0.004")),
                )
                lane_cfg["max_position_pct"] = min(
                    float(lane_cfg.get("max_position_pct", 0.05) or 0.05),
                    float(os.getenv("THROUGHPUT_DAY_MAX_POSITION_PCT", "0.015")),
                )
            elif lane_name == "crypto":
                lane_cfg["base_position_pct"] = min(
                    float(lane_cfg.get("base_position_pct", 0.025) or 0.025),
                    float(os.getenv("THROUGHPUT_CRYPTO_BASE_POSITION_PCT", "0.003")),
                )
                lane_cfg["max_position_pct"] = min(
                    float(lane_cfg.get("max_position_pct", 0.05) or 0.05),
                    float(os.getenv("THROUGHPUT_CRYPTO_MAX_POSITION_PCT", "0.012")),
                )

    def _apply_event_window_mode(self) -> None:
        self._day_trading_mode = True
        self._crypto_depth_enabled = True
        if not self._crypto_symbols:
            self._crypto_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
        self._price_refresh_seconds = min(
            self._price_refresh_seconds,
            max(120, int(os.getenv("EVENT_WINDOW_PRICE_REFRESH_SECONDS", "300"))),
        )
        self._sentiment_refresh_seconds = min(
            self._sentiment_refresh_seconds,
            max(300, int(os.getenv("EVENT_WINDOW_SENTIMENT_REFRESH_SECONDS", "900"))),
        )
        self._earnings_refresh_seconds = min(
            self._earnings_refresh_seconds,
            max(1800, int(os.getenv("EVENT_WINDOW_EARNINGS_REFRESH_SECONDS", "7200"))),
        )
        self._altdata_refresh_seconds = min(
            self._altdata_refresh_seconds,
            max(900, int(os.getenv("EVENT_WINDOW_ALTDATA_REFRESH_SECONDS", "1800"))),
        )
        self._inference_refresh_seconds = min(
            self._inference_refresh_seconds,
            max(5, int(os.getenv("EVENT_WINDOW_INFERENCE_REFRESH_SECONDS", "10"))),
        )
        self._poll_interval_seconds = min(
            self._poll_interval_seconds,
            max(5, int(os.getenv("EVENT_WINDOW_POLL_INTERVAL_SECONDS", "5"))),
        )
        self._poll_batch_size = max(
            self._poll_batch_size,
            max(20, int(os.getenv("EVENT_WINDOW_POLL_BATCH_SIZE", "40"))),
        )
        self._sentiment_pause_seconds = min(
            self._sentiment_pause_seconds,
            max(0.0, float(os.getenv("EVENT_WINDOW_SENTIMENT_BATCH_PAUSE_SECONDS", "0.25"))),
        )
        self._feature_store_save_seconds = min(
            self._feature_store_save_seconds,
            max(60, int(os.getenv("EVENT_WINDOW_FEATURE_SAVE_SECONDS", "180"))),
        )

        self._auto_trade_min_conviction = min(
            self._auto_trade_min_conviction,
            max(0.0, float(os.getenv("EVENT_WINDOW_MIN_CONVICTION", "1.8"))),
        )
        self._auto_trade_top_k = max(
            self._auto_trade_top_k,
            max(24, int(os.getenv("EVENT_WINDOW_TOP_K", "128"))),
        )
        self._auto_trade_max_new_per_cycle = max(
            self._auto_trade_max_new_per_cycle,
            max(6, int(os.getenv("EVENT_WINDOW_MAX_NEW_PER_CYCLE", "32"))),
        )
        self._auto_trade_max_open_positions = max(
            self._auto_trade_max_open_positions,
            max(8, int(os.getenv("EVENT_WINDOW_MAX_OPEN_POSITIONS", "42"))),
        )
        self._auto_trade_max_sector_positions = max(
            self._auto_trade_max_sector_positions,
            max(4, int(os.getenv("EVENT_WINDOW_MAX_SECTOR_POSITIONS", "12"))),
        )
        self._auto_trade_cooldown_seconds = min(
            self._auto_trade_cooldown_seconds,
            max(0, int(os.getenv("EVENT_WINDOW_COOLDOWN_SECONDS", "20"))),
        )
        self._auto_trade_leader_min_conviction = min(
            self._auto_trade_leader_min_conviction,
            max(0.0, float(os.getenv("EVENT_WINDOW_LEADER_MIN_CONVICTION", "1.6"))),
        )
        self._auto_trade_leader_min_take_probability = min(
            self._auto_trade_leader_min_take_probability,
            min(0.99, max(0.0, float(os.getenv("EVENT_WINDOW_LEADER_MIN_TAKE_PROBABILITY", "0.24")))),
        )
        self._auto_trade_leader_min_rank_score = min(
            self._auto_trade_leader_min_rank_score,
            max(0.0, float(os.getenv("EVENT_WINDOW_LEADER_MIN_RANK_SCORE", "0.32"))),
        )
        self._auto_trade_zero_weight_min_take_probability = min(
            self._auto_trade_zero_weight_min_take_probability,
            min(0.99, max(0.0, float(os.getenv("EVENT_WINDOW_ZERO_WEIGHT_MIN_TAKE_PROBABILITY", "0.24")))),
        )
        self._auto_trade_zero_weight_min_rank_score = min(
            self._auto_trade_zero_weight_min_rank_score,
            max(0.0, float(os.getenv("EVENT_WINDOW_ZERO_WEIGHT_MIN_RANK_SCORE", "0.32"))),
        )
        self._auto_trade_replace_margin = min(
            self._auto_trade_replace_margin,
            max(0.01, float(os.getenv("EVENT_WINDOW_REPLACE_MARGIN", "0.03"))),
        )
        dual_lane_raw = os.getenv("EVENT_WINDOW_DUAL_LANE_VARIANTS_ENABLED")
        if dual_lane_raw is None:
            self._allow_dual_lane_variants = True
        else:
            self._allow_dual_lane_variants = dual_lane_raw.strip().lower() in {"1", "true", "yes", "on"}

        self._lane_base_allocations.update(
            {
                "normal": max(0.05, float(os.getenv("EVENT_WINDOW_NORMAL_BASE_ALLOC_PCT", "0.32"))),
                "day": max(0.05, float(os.getenv("EVENT_WINDOW_DAY_BASE_ALLOC_PCT", "0.42"))),
                "crypto": max(0.05, float(os.getenv("EVENT_WINDOW_CRYPTO_BASE_ALLOC_PCT", "0.26"))),
            }
        )

        normal_cfg = self._lane_engine_config["normal"]
        normal_cfg["top_k"] = max(normal_cfg["top_k"], int(os.getenv("EVENT_WINDOW_NORMAL_TOP_K", "96")))
        normal_cfg["max_new_per_cycle"] = max(normal_cfg["max_new_per_cycle"], int(os.getenv("EVENT_WINDOW_NORMAL_MAX_NEW", "20")))
        normal_cfg["max_open_positions"] = max(normal_cfg["max_open_positions"], int(os.getenv("EVENT_WINDOW_NORMAL_MAX_OPEN", "36")))
        normal_cfg["min_conviction"] = min(normal_cfg["min_conviction"], float(os.getenv("EVENT_WINDOW_NORMAL_MIN_CONVICTION", "2.3")))
        normal_cfg["min_take_probability"] = min(normal_cfg["min_take_probability"], float(os.getenv("EVENT_WINDOW_NORMAL_MIN_TAKE_PROBABILITY", "0.30")))

        day_cfg = self._lane_engine_config["day"]
        day_cfg["top_k"] = max(day_cfg["top_k"], int(os.getenv("EVENT_WINDOW_DAY_TOP_K", "128")))
        day_cfg["max_new_per_cycle"] = max(day_cfg["max_new_per_cycle"], int(os.getenv("EVENT_WINDOW_DAY_MAX_NEW", "32")))
        day_cfg["max_open_positions"] = max(day_cfg["max_open_positions"], int(os.getenv("EVENT_WINDOW_DAY_MAX_OPEN", "48")))
        day_cfg["cooldown_seconds"] = min(day_cfg["cooldown_seconds"], int(os.getenv("EVENT_WINDOW_DAY_COOLDOWN_SECONDS", "10")))
        day_cfg["min_hold_seconds"] = min(day_cfg["min_hold_seconds"], int(os.getenv("EVENT_WINDOW_DAY_MIN_HOLD_SECONDS", "90")))
        day_cfg["min_conviction"] = min(day_cfg["min_conviction"], float(os.getenv("EVENT_WINDOW_DAY_MIN_CONVICTION", "1.8")))
        day_cfg["min_take_probability"] = min(day_cfg["min_take_probability"], float(os.getenv("EVENT_WINDOW_DAY_MIN_TAKE_PROBABILITY", "0.22")))

        crypto_cfg = self._lane_engine_config["crypto"]
        crypto_cfg["top_k"] = max(crypto_cfg["top_k"], int(os.getenv("EVENT_WINDOW_CRYPTO_TOP_K", "64")))
        crypto_cfg["max_new_per_cycle"] = max(crypto_cfg["max_new_per_cycle"], int(os.getenv("EVENT_WINDOW_CRYPTO_MAX_NEW", "24")))
        crypto_cfg["max_open_positions"] = max(crypto_cfg["max_open_positions"], int(os.getenv("EVENT_WINDOW_CRYPTO_MAX_OPEN", "28")))
        crypto_cfg["cooldown_seconds"] = min(crypto_cfg["cooldown_seconds"], int(os.getenv("EVENT_WINDOW_CRYPTO_COOLDOWN_SECONDS", "5")))
        crypto_cfg["min_hold_seconds"] = min(crypto_cfg["min_hold_seconds"], int(os.getenv("EVENT_WINDOW_CRYPTO_MIN_HOLD_SECONDS", "20")))
        crypto_cfg["min_conviction"] = min(crypto_cfg["min_conviction"], float(os.getenv("EVENT_WINDOW_CRYPTO_MIN_CONVICTION", "1.6")))
        crypto_cfg["min_take_probability"] = min(crypto_cfg["min_take_probability"], float(os.getenv("EVENT_WINDOW_CRYPTO_MIN_TAKE_PROBABILITY", "0.16")))
        self._crypto_scalper_min_take_probability = min(
            self._crypto_scalper_min_take_probability,
            crypto_cfg["min_take_probability"],
        )
        self._crypto_signal_stale_seconds = max(
            self._crypto_signal_stale_seconds,
            max(30, int(os.getenv("EVENT_WINDOW_CRYPTO_SIGNAL_STALE_SECONDS", "240"))),
        )
        self._crypto_signal_hold_grace_seconds = max(
            self._crypto_signal_hold_grace_seconds,
            max(60, int(os.getenv("EVENT_WINDOW_CRYPTO_SIGNAL_HOLD_GRACE_SECONDS", "1800"))),
        )
        self._crypto_signal_stale_retention_seconds = max(
            self._crypto_signal_stale_retention_seconds,
            max(
                self._crypto_signal_hold_grace_seconds,
                int(os.getenv("EVENT_WINDOW_CRYPTO_SIGNAL_STALE_RETENTION_SECONDS", "21600")),
            ),
        )
        self._crypto_scalper_max_spread_pct = max(
            self._crypto_scalper_max_spread_pct,
            max(0.05, float(os.getenv("EVENT_WINDOW_CRYPTO_MAX_SPREAD_PCT", "0.24"))),
        )
        self._crypto_scalper_max_depth_age_seconds = max(
            self._crypto_scalper_max_depth_age_seconds,
            max(2.0, float(os.getenv("EVENT_WINDOW_CRYPTO_MAX_DEPTH_AGE_SECONDS", "20.0"))),
        )
        self._crypto_scalper_max_tick_age_seconds = max(
            self._crypto_scalper_max_tick_age_seconds,
            max(2.0, float(os.getenv("EVENT_WINDOW_CRYPTO_MAX_TICK_AGE_SECONDS", "20.0"))),
        )
        self._crypto_scalper_max_signal_age_seconds = max(
            self._crypto_scalper_max_signal_age_seconds,
            max(3.0, float(os.getenv("EVENT_WINDOW_CRYPTO_MAX_SIGNAL_AGE_SECONDS", "30.0"))),
        )
        self._crypto_scalper_min_book_pressure = min(
            self._crypto_scalper_min_book_pressure,
            max(0.0, float(os.getenv("EVENT_WINDOW_CRYPTO_MIN_BOOK_PRESSURE", "0.015"))),
        )
        self._crypto_scalper_min_depth_imbalance = min(
            self._crypto_scalper_min_depth_imbalance,
            max(0.0, float(os.getenv("EVENT_WINDOW_CRYPTO_MIN_DEPTH_IMBALANCE", "0.01"))),
        )

    def _chunk_symbols(self, symbols: List[str], batch_size: int) -> List[List[str]]:
        return [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]

    def _priority_pipeline_symbols(self, symbols: List[str]) -> List[str]:
        ordered = list(dict.fromkeys(str(symbol).strip().upper() for symbol in (symbols or []) if str(symbol).strip()))
        if not ordered:
            return []
        allowed = set(ordered)
        priority: List[str] = []
        broker = self._components.get("broker")
        if broker is not None:
            for _, position, symbol, _, _ in self._iter_open_positions(broker):
                if float(getattr(position, "quantity", 0.0) or 0.0) > 0.0 and symbol in allowed:
                    priority.append(symbol)
        try:
            from pipeline.universe import DAY_TRADE_EXPANDED_SYMBOLS, LEADER_SYMBOLS

            priority.extend(symbol for symbol in LEADER_SYMBOLS if symbol in allowed)
            priority.extend(symbol for symbol in DAY_TRADE_EXPANDED_SYMBOLS if symbol in allowed)
        except Exception:
            pass
        limit = self._pipeline_priority_symbol_cap
        deduped = list(dict.fromkeys(priority))
        return deduped[:limit] if limit > 0 else deduped

    def _select_pipeline_symbols(self, source_name: str, symbols: List[str]) -> List[str]:
        source_key = str(source_name or "").lower()
        ordered = list(dict.fromkeys(str(symbol).strip().upper() for symbol in (symbols or []) if str(symbol).strip()))
        cap = max(0, int(self._source_symbol_caps.get(source_key, 0) or 0))
        bootstrap_cap = self._source_bootstrap_symbol_cap(source_key, len(ordered), configured_cap=cap)
        effective_cap = cap
        if bootstrap_cap > 0:
            effective_cap = bootstrap_cap if effective_cap <= 0 else min(effective_cap, bootstrap_cap)
        if not ordered:
            self._source_selection_summary[source_key] = {
                "selected_symbols": 0,
                "total_symbols": 0,
                "priority_symbols": 0,
                "rotation_offset": 0,
                "configured_cap": cap,
                "effective_cap": 0,
                "bootstrap_active": False,
            }
            return []
        if effective_cap <= 0 or len(ordered) <= effective_cap:
            self._source_selection_summary[source_key] = {
                "selected_symbols": len(ordered),
                "total_symbols": len(ordered),
                "priority_symbols": min(len(ordered), len(self._priority_pipeline_symbols(ordered))),
                "rotation_offset": 0,
                "configured_cap": cap,
                "effective_cap": len(ordered),
                "bootstrap_active": False,
            }
            return ordered

        priority = self._priority_pipeline_symbols(ordered)
        selected = list(priority[:effective_cap])
        remaining_slots = max(0, effective_cap - len(selected))
        remainder = [symbol for symbol in ordered if symbol not in selected]
        next_offset = 0
        if remaining_slots > 0 and remainder:
            offset = self._pipeline_rotation_offsets.get(source_key, 0) % len(remainder)
            rotated = remainder[offset:] + remainder[:offset]
            selected.extend(rotated[:remaining_slots])
            next_offset = (offset + remaining_slots) % len(remainder)
        self._pipeline_rotation_offsets[source_key] = next_offset
        self._source_selection_summary[source_key] = {
            "selected_symbols": len(selected),
            "total_symbols": len(ordered),
            "priority_symbols": len(priority[:effective_cap]),
            "rotation_offset": next_offset,
            "configured_cap": cap,
            "effective_cap": effective_cap,
            "bootstrap_active": bootstrap_cap > 0 and effective_cap == bootstrap_cap,
        }
        return selected

    def _source_bootstrap_symbol_cap(self, source_name: str, total_symbols: int, *, configured_cap: int = 0) -> int:
        source_key = str(source_name or "").lower()
        if total_symbols <= 0:
            return 0
        if self._data_versions.get(source_key, 0) > 0:
            return 0
        default_caps = {
            "prices": 160,
            "sentiment": 80,
            "earnings": 160,
            "altdata": 48,
        }
        env_names = {
            "prices": "PRICE_PIPELINE_BOOTSTRAP_SYMBOL_CAP",
            "sentiment": "SENTIMENT_PIPELINE_BOOTSTRAP_SYMBOL_CAP",
            "earnings": "EARNINGS_PIPELINE_BOOTSTRAP_SYMBOL_CAP",
            "altdata": "ALTDATA_PIPELINE_BOOTSTRAP_SYMBOL_CAP",
        }
        bootstrap_env_name = env_names.get(source_key, f"{source_key.upper()}_PIPELINE_BOOTSTRAP_SYMBOL_CAP")
        source_bootstrap_env = os.getenv(bootstrap_env_name)
        global_bootstrap_env = os.getenv("PIPELINE_BOOTSTRAP_SYMBOL_CAP")
        if max(0, int(configured_cap or 0)) > 0 and source_bootstrap_env is None and global_bootstrap_env is None:
            return 0
        raw_cap = (
            source_bootstrap_env
            if source_bootstrap_env is not None
            else global_bootstrap_env
            if global_bootstrap_env is not None
            else str(default_caps.get(source_key, 0))
        )
        try:
            cap = max(0, int(float(raw_cap or 0)))
        except Exception:
            cap = default_caps.get(source_key, 0)
        candidate_total = min(total_symbols, max(0, int(configured_cap or 0))) if max(0, int(configured_cap or 0)) > 0 else total_symbols
        if cap <= 0 or candidate_total <= cap:
            return 0
        return min(candidate_total, cap)

    def _set_data_pipeline_status(self, **fields: Any) -> None:
        self._data_pipeline_status.clear()
        self._data_pipeline_status.update(
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                **fields,
            }
        )

    def _start_chunked_component(self, component_key: str, factory, symbols: List[str], chunk_size: int, **kwargs) -> int:
        ordered = list(dict.fromkeys(str(symbol).strip().upper() for symbol in (symbols or []) if str(symbol).strip()))
        if not ordered:
            return 0
        instances = []
        for chunk in self._chunk_symbols(ordered, max(1, chunk_size)):
            instance = factory(symbols=chunk, **kwargs)
            instance.start()
            instances.append(instance)
        self._components[component_key] = instances[0] if len(instances) == 1 else instances
        return len(instances)

    def _collect_sentiment_batched(self, symbols: List[str], days_back: int = 3, save: bool = True) -> Dict:
        import pandas as pd
        from pipeline.sentiment_collector import SentimentPipeline

        pipeline = SentimentPipeline()
        batches = self._chunk_symbols(symbols, self._sentiment_batch_size)
        merged_daily = []
        merged_headlines = {}
        merged_provider_map = {}
        summary_provider_counts: Dict[str, int] = {}
        total_symbols_with_news = 0
        total_official_symbols = 0
        total_fallback_symbols = 0
        no_news_symbols: List[str] = []

        for idx, batch in enumerate(batches, start=1):
            logger.info(f"Sentiment batch {idx}/{len(batches)}: {len(batch)} symbols")
            result = pipeline.run(symbols=batch, days_back=days_back, save=save)
            daily = result.get("symbol_sentiment_daily")
            if daily is not None and not getattr(daily, "empty", True):
                merged_daily.append(daily)
            merged_headlines.update(result.get("headlines", {}))
            merged_provider_map.update(result.get("news_provider_by_symbol", {}))
            summary = result.get("news_summary", {}) or {}
            total_symbols_with_news += int(summary.get("symbols_with_news", 0) or 0)
            total_official_symbols += int(summary.get("official_symbols", 0) or 0)
            total_fallback_symbols += int(summary.get("fallback_symbols", 0) or 0)
            no_news_symbols.extend(summary.get("no_news_symbols", []) or [])
            for provider, count in (summary.get("provider_counts", {}) or {}).items():
                summary_provider_counts[provider] = summary_provider_counts.get(provider, 0) + int(count)
            if idx < len(batches) and self._sentiment_pause_seconds > 0:
                time.sleep(self._sentiment_pause_seconds)

        daily_merged = pd.concat(merged_daily).sort_index() if merged_daily else pd.DataFrame()
        return {
            "symbol_sentiment_daily": daily_merged,
            "headlines": merged_headlines,
            "news_provider_by_symbol": merged_provider_map,
            "news_summary": {
                "symbols_with_news": total_symbols_with_news,
                "official_symbols": total_official_symbols,
                "fallback_symbols": total_fallback_symbols,
                "no_news_symbols": no_news_symbols,
                "provider_counts": summary_provider_counts,
            },
            "run_timestamp": pd.Timestamp.utcnow().isoformat(),
            "symbols_covered": len(merged_headlines),
        }

    def _collect_altdata(self, symbols: List[str]) -> Dict:
        from pipeline.altdata_collector import OpenSkyTravelFactorPipeline

        pipeline = OpenSkyTravelFactorPipeline()
        return pipeline.run(symbols=symbols)

    def _refresh_event_intelligence(self) -> Dict:
        if not self._event_intel_enabled:
            return {}
        sentiment_payload = self._components.get("sentiment_data")
        if not isinstance(sentiment_payload, dict) or not sentiment_payload.get("headlines"):
            return {}
        try:
            from pipeline.event_intel import build_event_intelligence

            event_payload = build_event_intelligence(
                sentiment_payload=sentiment_payload,
                signal_store=self._signal_store,
                output_dir=self._event_intel_dir,
                universe_mode=self._universe_mode,
            )
            self._components["event_intel"] = event_payload
            summary = event_payload.get("summary", {}) if isinstance(event_payload, dict) else {}
            logger.info(
                "Event intelligence refreshed: %s/%s symbols with news | %s official catalysts | saved to %s",
                summary.get("symbols_with_news", 0),
                summary.get("symbols_covered", 0),
                summary.get("official_symbols", 0),
                self._event_intel_dir,
            )
            return event_payload
        except Exception as exc:
            logger.warning(f"Event intelligence refresh failed: {exc}")
            return {}

    def start(
        self,
        run_dashboard: bool = True,
        run_realtime: bool = True,
        run_backtest: bool = True,
        run_security: bool = True,
        dashboard_port: int = int(os.getenv("DASHBOARD_PORT", "5050")),
    ):
        logger.info("=" * 60)
        logger.info("MACRO INTELLIGENCE SYSTEM - STARTING")
        logger.info("=" * 60)
        try:
            logger.info(
                "Performance: profile=%s | threads=%s | priority_applied=%s",
                _PERF_SETTINGS.get("profile"),
                _PERF_SETTINGS.get("threads"),
                _PERF_SETTINGS.get("process_priority_applied"),
            )
        except Exception:
            pass
        logger.info(f"Universe mode: {self._universe_mode}")
        if self._event_window_mode:
            logger.info(
                "Event window mode: active | %s-day high-volatility preset | event digests -> %s",
                self._event_window_days,
                self._event_intel_dir,
            )
            logger.info(
                "Event window lane split: dual-lane variants %s",
                "enabled" if self._allow_dual_lane_variants else "disabled",
            )
        if self._day_trading_mode:
            logger.info(
                "Day trading mode: active | "
                f"force exit {self._day_trade_force_exit_seconds}s | "
                f"intraday threshold {self._day_trade_intraday_score_threshold:.2f}"
            )
        if self._crypto_depth_enabled and self._crypto_symbols:
            logger.info(
                "Crypto scalper mode: active | "
                f"{len(self._crypto_symbols)} symbols | stale cutoff {self._crypto_signal_stale_seconds}s"
            )
        logger.info(f"Price provider order: {os.getenv('PRICE_PROVIDER_ORDER', 'yfinance,finnhub,alpha_vantage,polygon,eodhd')}")
        logger.info(
            "News provider order: "
            f"{os.getenv('NEWS_PROVIDER_ORDER', 'google_rss,sec_filings,press_releases,finnhub,alpha_vantage,eodhd,polygon,newsapi,gnews,rss,nse_announcements,bse_announcements')}"
        )
        logger.info(f"Earnings provider order: {os.getenv('EARNINGS_PROVIDER_ORDER', 'finnhub,alpha_vantage,eodhd')}")
        logger.info(
            f"Refresh cadence: prices {self._price_refresh_seconds}s | "
            f"sentiment {self._sentiment_refresh_seconds}s | "
            f"earnings {self._earnings_refresh_seconds}s | "
            f"altdata {self._altdata_refresh_seconds}s | "
            f"inference {self._inference_refresh_seconds}s"
        )
        logger.info(
            "Entry readiness guard: %s | stale multiplier x%.1f | market-open check %s",
            "enabled" if self._entry_readiness_enabled else "disabled",
            self._entry_stale_multiplier,
            "on" if self._entry_require_market_open else "off",
        )
        logger.info(
            f"Execution overlay: top {self._auto_trade_top_k} ideas | "
            f"max {self._auto_trade_max_open_positions} positions | "
            f"max {self._auto_trade_max_sector_positions} per sector"
        )
        if self._throughput_mode:
            logger.info(
                "Throughput mode: active | lane targets normal=%s day=%s crypto=%s",
                self._lane_target_open_positions.get("normal", 0),
                self._lane_target_open_positions.get("day", 0),
                self._lane_target_open_positions.get("crypto", 0),
            )
        if self._alpha_quality_enabled:
            logger.info(
                "Alpha quality monitor: enabled | lookback %sd | horizon %sd | buckets %s",
                self._alpha_quality_lookback_days,
                self._alpha_quality_horizon_days,
                self._alpha_quality_bucket_count,
            )
        logger.info(
            f"Portfolio correlation guard: max pair {self._auto_trade_max_pair_correlation:.2f} | "
            f"target avg {self._auto_trade_target_avg_correlation:.2f}"
        )
        if self._portfolio_optimizer_enabled:
            logger.info(
                f"Portfolio optimizer: gross {self._portfolio_optimizer_gross_target_pct:.2f} | "
                f"beta target {self._portfolio_optimizer_target_beta:.2f} | "
                f"max {self._portfolio_optimizer_max_names} names"
            )
        if self._feature_store_enabled:
            logger.info(
                f"Feature store autosave: {self._feature_store_dir} every {self._feature_store_save_seconds}s"
            )
        if self._auto_retrain_on_start:
            logger.info(
                f"Auto retrain on start: enabled | timeout {self._auto_retrain_timeout_seconds}s | "
                f"optuna {'on' if self._auto_retrain_use_optuna else 'off'} | "
                f"only-if-new-data {'on' if self._auto_retrain_only_if_new_data else 'off'}"
            )

        self._maybe_auto_retrain()

        self._running = True
        self._components["execution_trace"] = self._execution_trace
        self._components["execution_backtest"] = {}
        self._components["alpha_quality"] = {}
        self._components["portfolio_overlay"] = {}
        self._components["meta_model_status"] = {}
        self._components["stress_results"] = {}
        self._refresh_meta_model_status()
        self._refresh_learning_status()

        logger.info("[1/8] Initializing Point-in-Time database...")
        from core.point_in_time import PITDatabase

        self._components["pit_db"] = PITDatabase()

        logger.info("[2/8] Initializing execution broker...")
        from core.brokerages import ShadowBroker
        from core.external_execution import build_shadow_router
        from core.paper_trading import VirtualBroker

        paper_broker = VirtualBroker(
            initial_capital=float(os.getenv("PAPER_CAPITAL", "100000")),
            max_position_pct=self._auto_trade_max_position_pct,
            max_drawdown_pct=float(os.getenv("PAPER_MAX_DRAWDOWN_PCT", "0.20")),
            max_daily_loss_pct=self._day_trade_max_daily_loss_pct,
            max_consecutive_losses=self._day_trade_max_consecutive_losses,
            session_guardrails_enabled=os.getenv("PAPER_SESSION_GUARDRAILS_ENABLED", "0").strip().lower()
            in {"1", "true", "yes", "on"},
        )
        self._components["paper_broker"] = paper_broker

        broker = paper_broker
        if self._shadow_router_enabled:
            tracked_shadow_symbols = list(
                dict.fromkeys(self._get_target_symbols() + list(self._crypto_symbols))
            )
            shadow_router = build_shadow_router(api_keys=self._api_keys, tracked_symbols=tracked_shadow_symbols)
            if shadow_router is not None:
                broker = ShadowBroker(
                    paper_broker,
                    secondary=shadow_router,
                    reconciliation_path=self._shadow_reconciliation_log_path,
                )
                self._components["live_execution_router"] = shadow_router
                logger.info(
                    f"Execution mode: shadow | router={shadow_router.__class__.__name__} "
                    f"| configured={getattr(shadow_router, 'configured', False)} "
                    f"| log={self._shadow_reconciliation_log_path}"
                )
            else:
                logger.info(
                    f"Execution mode: shadow requested but no live secondary router is configured | "
                    f"log={self._shadow_reconciliation_log_path}"
                )
        else:
            logger.info("Execution mode: paper")

        self._components["broker"] = broker
        self._sync_runtime_state_with_broker(broker)
        self._persist_runtime_state(force=True)

        logger.info("[3/8] Starting data pipeline...")
        threading.Thread(target=self._data_pipeline_loop, daemon=True).start()

        if run_backtest:
            logger.info("[4/8] Running stress tests...")
            threading.Thread(target=self._run_stress_tests, daemon=True).start()

        if run_realtime:
            logger.info("[5/8] Starting real-time data feeds...")
            self._start_realtime_feeds()

        logger.info("[6/8] Starting signal inference loop...")
        threading.Thread(target=self._inference_loop, daemon=True).start()
        if self._crypto_depth_enabled and self._crypto_symbols:
            logger.info("Starting dedicated crypto signal loop...")
            threading.Thread(target=self._crypto_signal_loop, daemon=True).start()

        if run_security:
            logger.info("[7/8] Starting Humanizer Security Suite...")
            self._start_security_suite(dashboard_port)

        if run_dashboard:
            logger.info(f"[8/8] Starting dashboard on http://localhost:{dashboard_port}")
            self._start_dashboard(dashboard_port)

    def _start_realtime_feeds(self):
        from core.realtime_engine import (
            BinancePublicDepthWebSocket,
            BybitPublicDepthWebSocket,
            BybitTickerPollingFeed,
            FinnhubWebSocket,
            MetaTrader5PollingFeed,
            PollingFallback,
            UpstoxWebSocket,
            PRICE_BUFFER,
        )

        if not self._intraday_subscription_attached:
            PRICE_BUFFER.subscribe(self._on_realtime_tick)
            self._intraday_subscription_attached = True

        us = self._get_target_symbols("us")
        nse = self._get_target_symbols("nse")
        target_symbols = self._get_target_symbols()
        polling_enabled = os.getenv("POLLING_FALLBACK_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
        covered_by_ws = set()

        if self._api_keys.get("finnhub") and us:
            ws_symbols = us[:50]
            if len(us) > len(ws_symbols):
                logger.warning(f"Finnhub free-tier limit reached: streaming {len(ws_symbols)} of {len(us)} US symbols")
            ws = FinnhubWebSocket(api_key=self._api_keys["finnhub"], symbols=ws_symbols, buffer=PRICE_BUFFER)
            ws.start()
            self._components["finnhub_ws"] = ws
            covered_by_ws.update(ws_symbols)

        if self._api_keys.get("upstox") and nse:
            ws = UpstoxWebSocket(access_token=self._api_keys["upstox"], symbols=nse, buffer=PRICE_BUFFER)
            ws.start()
            self._components["upstox_ws"] = ws
            covered_by_ws.update(nse)

        if os.getenv("MT5_FEED_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"} and target_symbols:
            mt5_feed = MetaTrader5PollingFeed(
                symbols=target_symbols,
                buffer=PRICE_BUFFER,
                poll_interval_seconds=float(os.getenv("MT5_FEED_POLL_INTERVAL_SECONDS", "1.0")),
                symbol_map_path=os.getenv("MT5_SYMBOL_MAP_PATH", "data/mt5_symbols.json"),
                path=os.getenv("MT5_PATH", ""),
                login=os.getenv("MT5_LOGIN", ""),
                password=os.getenv("MT5_PASSWORD", ""),
                server=os.getenv("MT5_SERVER", ""),
            )
            mt5_feed.start()
            self._components["mt5_feed"] = mt5_feed

        if self._crypto_depth_enabled and self._crypto_symbols:
            started_crypto_feeds = []
            for feed_name in self._crypto_feed_order:
                if feed_name == "binance":
                    connection_count = self._start_chunked_component(
                        "binance_public_depth_ws",
                        BinancePublicDepthWebSocket,
                        self._binance_crypto_symbols,
                        self._binance_ws_symbols_per_connection,
                        buffer=PRICE_BUFFER,
                        testnet=os.getenv("BINANCE_WS_TESTNET", "0").strip().lower() in {"1", "true", "yes", "on"},
                    )
                    if connection_count:
                        started_crypto_feeds.append(f"binance x{connection_count}")
                elif feed_name == "bybit":
                    connection_count = self._start_chunked_component(
                        "bybit_public_depth_ws",
                        BybitPublicDepthWebSocket,
                        self._bybit_crypto_symbols,
                        self._bybit_ws_symbols_per_connection,
                        buffer=PRICE_BUFFER,
                        testnet=os.getenv("BYBIT_WS_TESTNET", "0").strip().lower() in {"1", "true", "yes", "on"},
                    )
                    if connection_count:
                        started_crypto_feeds.append(f"bybit x{connection_count}")
                if started_crypto_feeds and not self._crypto_use_all_feeds:
                    break
            if started_crypto_feeds:
                logger.info(
                    f"Crypto realtime feeds started: {', '.join(started_crypto_feeds)} | "
                    f"signal universe={len(self._crypto_symbols)} | primary_only={self._crypto_primary_feed_only_symbols}"
                )
            if os.getenv("BYBIT_TICKER_POLL_FALLBACK_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}:
                bybit_poll_symbols = self._bybit_crypto_symbols or self._crypto_symbols
                if bybit_poll_symbols:
                    bybit_poll = BybitTickerPollingFeed(
                        symbols=bybit_poll_symbols,
                        buffer=PRICE_BUFFER,
                        interval_seconds=float(os.getenv("BYBIT_TICKER_POLL_INTERVAL_SECONDS", "3.0")),
                        timeout_seconds=float(os.getenv("BYBIT_TICKER_POLL_TIMEOUT_SECONDS", "8.0")),
                        testnet=os.getenv("BYBIT_WS_TESTNET", "0").strip().lower() in {"1", "true", "yes", "on"},
                    )
                    bybit_poll.start()
                    self._components["bybit_ticker_poll"] = bybit_poll

        if not polling_enabled:
            logger.info("Polling fallback disabled (POLLING_FALLBACK_ENABLED=0)")
            return

        poll_symbols = [s for s in target_symbols if s not in covered_by_ws]
        if not poll_symbols:
            logger.info("Polling fallback skipped: all symbols covered by WebSockets")
            return

        poller = PollingFallback(
            symbols=poll_symbols,
            interval_seconds=self._poll_interval_seconds,
            buffer=PRICE_BUFFER,
            batch_size=self._poll_batch_size,
        )
        poller.start()
        self._components["poller"] = poller
        logger.info(
            f"Polling fallback configured for {len(poll_symbols)} symbols in batches of {self._poll_batch_size} "
            f"(covered by ws={len(covered_by_ws)})"
        )

    def _start_security_suite(self, port: int):
        def _boot():
            try:
                from security.camera_alerts import SecuritySuite

                enable_ngrok = os.getenv("SECURITY_ENABLE_NGROK", os.getenv("ENABLE_NGROK", "1")).strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
                suite = SecuritySuite(camera_index=int(os.getenv("CAMERA_INDEX", "0")), dashboard_port=port)
                result = suite.start(enable_ngrok=enable_ngrok)
                self._components["security"] = suite
                if result.get("ngrok_url"):
                    logger.info(f"Remote access: {result['ngrok_url']}")
                elif not enable_ngrok:
                    logger.info("Remote access: disabled by SECURITY_ENABLE_NGROK=0")
            except Exception as exc:
                logger.error(f"Security suite error: {exc}")

        threading.Thread(target=_boot, daemon=True).start()

    def _get_latest_close(self, symbol: str) -> Optional[float]:
        try:
            latest_tick = self._latest_live_tick(symbol)
            if latest_tick and float(latest_tick.price) > 0:
                return float(latest_tick.price)
        except Exception:
            pass
        price_data = self._components.get("price_data", {})
        recent = price_data.get("price_daily_recent", {}).get(symbol)
        if recent is None or recent.empty:
            return None
        current_price = float(recent["close"].iloc[-1])
        if current_price <= 0:
            return None
        return current_price

    def _get_recent_return_series(self, symbol: str, lookback_days: Optional[int] = None):
        live_recent = self._components.get("live_price_data", {}).get("price_daily_recent", {}).get(symbol)
        recent = live_recent if live_recent is not None else self._components.get("price_data", {}).get("price_daily_recent", {}).get(symbol)
        if recent is None or recent.empty or "close" not in recent.columns:
            return None
        closes = recent["close"].dropna()
        if len(closes) < 20:
            return None
        returns = closes.pct_change().dropna()
        if lookback_days:
            returns = returns.tail(lookback_days)
        if len(returns) < 15:
            return None
        returns.name = symbol
        return returns

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, float(value)))

    @staticmethod
    def _market_code_for_symbol(symbol: str) -> str:
        raw = str(symbol or "").upper()
        if raw.endswith(("USDT", "USDC", "BUSD")):
            return "CRYPTO"
        if raw.endswith(".NS") or raw.startswith("^NSE"):
            return "IN"
        return "US"

    @staticmethod
    def _lane_label(lane: str) -> str:
        return {
            "normal": "Normal Trading",
            "day": "Day Trading",
            "crypto": "Crypto Scalper",
        }.get(str(lane or "").lower(), "Normal Trading")

    def _time_bucket_label(self, timestamp: Optional[datetime], market: str) -> str:
        if not isinstance(timestamp, datetime):
            return "unknown"
        local_ts, open_local, close_local = self._market_session_clock(timestamp, market)
        minutes_from_open = max(0.0, (local_ts - open_local).total_seconds() / 60.0)
        minutes_to_close = max(0.0, (close_local - local_ts).total_seconds() / 60.0)
        if str(market).upper() == "CRYPTO":
            hour = local_ts.hour
            if 0 <= hour < 8:
                return "asia"
            if 8 <= hour < 13:
                return "europe"
            if 13 <= hour < 21:
                return "us"
            return "late_session"
        if minutes_from_open <= 30:
            return "open"
        if minutes_from_open <= 120:
            return "trend_window"
        if minutes_to_close <= 60:
            return "power_hour"
        if minutes_from_open >= 120:
            return "midday"
        return "session"

    def _signal_lane_meta(self, symbol: str, signal: Optional[Dict] = None, lane_override: Optional[str] = None) -> Dict:
        signal = signal or {}
        intraday = signal.get("intraday_overlay", {}) if isinstance(signal.get("intraday_overlay"), dict) else {}
        signal_style = str(signal.get("signal_style", "") or "")
        market = str(
            signal.get("market")
            or intraday.get("market")
            or self._market_code_for_symbol(symbol)
        ).upper()
        if lane_override:
            lane = lane_override
        elif market == "CRYPTO" or signal_style == "crypto_depth_intraday":
            lane = "crypto"
        elif signal_style == "day_trade_intraday":
            lane = "day"
        else:
            lane = "normal"
        setup_id = str(
            signal.get("setup_id")
            or intraday.get("setup")
            or ("crypto_microstructure" if lane == "crypto" else "event_multifactor")
        )
        timestamp_raw = signal.get("timestamp")
        timestamp = None
        if isinstance(timestamp_raw, datetime):
            timestamp = timestamp_raw
        elif timestamp_raw:
            try:
                timestamp = datetime.fromisoformat(str(timestamp_raw))
            except Exception:
                timestamp = None
        signal_key = f"{symbol}::{lane}"
        return {
            "lane": lane,
            "lane_label": self._lane_label(lane),
            "market": market,
            "setup_id": setup_id,
            "signal_key": signal_key,
            "time_bucket": self._time_bucket_label(timestamp or datetime.now(timezone.utc), market),
        }

    def _decorate_signal(self, symbol: str, signal: Optional[Dict], lane_override: Optional[str] = None) -> Dict:
        payload = dict(signal or {})
        payload["symbol"] = symbol
        payload.update(self._signal_lane_meta(symbol, payload, lane_override=lane_override))
        payload.setdefault("scenario", payload.get("lane_label"))
        return payload

    def _get_meta_engine(self):
        engine = self._components.get("meta_engine")
        if engine is None:
            from core.signal_engine_v2 import MetaDecisionEngine

            engine = MetaDecisionEngine()
            self._components["meta_engine"] = engine
        return engine

    def _latest_feature_row(self, symbol: str):
        rows = self._components.get("latest_feature_rows", {})
        if not isinstance(rows, dict):
            return None
        return rows.get(symbol)

    def _hydrate_signal_runtime_fields(self, symbol: str, signal: Optional[Dict]) -> Dict:
        payload = dict(signal or {})
        if not payload:
            return {}

        meta = payload.get("meta_decision")
        if not (isinstance(meta, dict) and "take_trade" in meta):
            feature_row = self._latest_feature_row(symbol)
            try:
                feature_rows = {symbol: feature_row} if feature_row is not None else None
                evaluated = self._get_meta_engine().evaluate_universe({symbol: payload}, feature_rows=feature_rows).get(symbol, {})
            except Exception as exc:
                logger.debug(f"Meta hydration failed for {symbol}: {exc}")
                evaluated = {}
            if evaluated:
                payload["meta_decision"] = evaluated
                payload["trade_eligible"] = evaluated.get("take_trade", False)
                payload["take_probability"] = evaluated.get("take_probability", 0.0)
                payload["skip_probability"] = evaluated.get("skip_probability", 1.0)
                payload["expected_edge_pct"] = evaluated.get("expected_edge_pct", 0.0)
                payload["expected_drawdown_pct"] = evaluated.get("expected_drawdown_pct", 0.0)
                payload["rank_score"] = evaluated.get("rank_score", 0.0)
                payload["rank_percentile"] = evaluated.get("rank_percentile", 0.0)
                payload["size_multiplier"] = evaluated.get("size_multiplier", 1.0)
                payload["meta_source"] = evaluated.get("source", "heuristic")

        lane = str(payload.get("lane") or self._signal_lane_meta(symbol, payload).get("lane") or "normal").lower()
        overlay = self._components.get("portfolio_overlay", {})
        if isinstance(overlay, dict):
            lane_allocator = (overlay.get("lane_allocator", {}) or {}).get(lane, {})
            governor_state = (overlay.get("governor", {}) or {}).get(lane, {})
            if lane_allocator and not isinstance(payload.get("lane_allocator"), dict):
                payload["lane_allocator"] = lane_allocator
            if governor_state and not isinstance(payload.get("governor_state"), dict):
                payload["governor_state"] = governor_state

        return payload

    def _build_order_metadata(self, symbol: str, signal: Optional[Dict], *, action: str, reason: str) -> Dict:
        payload = self._decorate_signal(symbol, signal or {})
        intraday = payload.get("intraday_overlay", {}) if isinstance(payload.get("intraday_overlay"), dict) else {}
        metadata = {
            "lane": payload.get("lane"),
            "lane_label": payload.get("lane_label"),
            "market": payload.get("market"),
            "setup_id": payload.get("setup_id"),
            "regime": str(payload.get("regime") or intraday.get("regime") or "normal"),
            "signal_style": payload.get("signal_style"),
            "time_bucket": payload.get("time_bucket"),
            "action_reason": reason,
            "signal_strength": round(float(payload.get("ensemble_score", payload.get("final_score", 0.0)) or 0.0), 4),
            "relative_volume": round(float(intraday.get("volume_spike", 0.0) or 0.0), 4),
            "news_flag": bool(abs(float(intraday.get("news_boost", 0.0) or 0.0)) >= self._day_trade_news_boost_threshold),
            "action": action,
        }
        for field in ("take_probability", "expected_edge_pct", "expected_drawdown_pct", "rank_score", "size_multiplier"):
            value = payload.get(field)
            if value in (None, ""):
                continue
            try:
                metadata[field] = round(float(value), 6)
            except Exception:
                metadata[field] = value
        execution_reference_price = self._get_latest_close(symbol)
        if execution_reference_price is not None:
            metadata["execution_reference_price"] = round(float(execution_reference_price), 8)
        for field in (
            "best_bid",
            "best_ask",
            "best_bid_qty",
            "best_ask_qty",
            "bid_depth_qty",
            "ask_depth_qty",
            "spread_pct",
            "tick_age_seconds",
            "depth_age_seconds",
            "signal_age_seconds",
            "book_pressure",
            "book_imbalance",
            "top_book_imbalance",
            "order_flow_imbalance",
            "spread_velocity",
            "microprice_bias",
            "flicker_filter_retention",
        ):
            value = intraday.get(field, payload.get(field))
            if value is None:
                continue
            try:
                metadata[field] = float(value)
            except Exception:
                metadata[field] = value
        execution_mode = payload.get("execution_mode")
        if execution_mode:
            metadata["execution_mode"] = str(execution_mode)
        return metadata

    @staticmethod
    def _safe_number(value, default: float = 0.0) -> float:
        try:
            if value in (None, ""):
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _cooldown_key(symbol: str, lane: Optional[str] = None) -> str:
        return f"{str(lane or 'shared').lower()}::{symbol}"

    @staticmethod
    def _position_key(symbol: str, lane: Optional[str] = None) -> str:
        normalized_lane = str(lane or "normal").lower()
        return f"{symbol}::{normalized_lane}"

    @staticmethod
    def _lane_from_position_key(position_key: str, default: str = "normal") -> str:
        text = str(position_key or "")
        if "::" in text:
            _, lane = text.rsplit("::", 1)
            if lane:
                return lane.lower()
        return str(default or "normal").lower()

    def _iter_open_positions(self, broker) -> List[Tuple[str, object, str, str, Dict]]:
        rows: List[Tuple[str, object, str, str, Dict]] = []
        if not broker:
            return rows
        for position_key, position in getattr(broker, "positions", {}).items():
            if getattr(position, "quantity", 0) <= 0:
                continue
            symbol = str(getattr(position, "symbol", position_key) or position_key)
            plan = self._position_plans.get(position_key, {})
            lane = str(plan.get("lane") or self._lane_from_position_key(position_key) or "normal").lower()
            rows.append((position_key, position, symbol, lane, plan))
        return rows

    def _lane_config(self, lane: str) -> Dict:
        return self._lane_engine_config.get(str(lane or "normal").lower(), self._lane_engine_config["normal"])

    def _get_lane_signal(self, symbol: str, lane: Optional[str] = None) -> Dict:
        raw = self._signal_store.get(symbol, {})
        if not isinstance(raw, dict):
            return {}
        target_lane = str(lane or raw.get("lane") or "normal").lower()
        if target_lane == "normal" and isinstance(raw.get("normal_lane_signal"), dict):
            variant = self._hydrate_signal_runtime_fields(symbol, raw["normal_lane_signal"])
            variant["symbol"] = symbol
            return self._decorate_signal(symbol, variant, lane_override="normal")
        if target_lane == "normal" and str(raw.get("lane", "")).lower() == "normal":
            return self._decorate_signal(symbol, self._hydrate_signal_runtime_fields(symbol, raw), lane_override="normal")
        if target_lane in {"day", "crypto"} and str(raw.get("lane", "")).lower() == target_lane:
            return self._decorate_signal(symbol, self._hydrate_signal_runtime_fields(symbol, raw), lane_override=target_lane)
        if lane is None:
            inferred = self._signal_lane_meta(symbol, raw).get("lane")
            return self._decorate_signal(symbol, self._hydrate_signal_runtime_fields(symbol, raw), lane_override=inferred)
        return {}

    def _iter_lane_signal_items(self) -> List[Tuple[str, str, Dict]]:
        items: List[Tuple[str, str, Dict]] = []
        seen = set()
        for symbol, raw in self._signal_store.items():
            if not isinstance(raw, dict):
                continue
            primary_lane = str(raw.get("lane") or self._signal_lane_meta(symbol, raw).get("lane") or "normal").lower()
            primary = self._get_lane_signal(symbol, primary_lane)
            if primary:
                key = (symbol, primary_lane)
                if key not in seen:
                    items.append((primary_lane, symbol, primary))
                    seen.add(key)
            normal_variant = self._get_lane_signal(symbol, "normal")
            include_normal_variant = bool(normal_variant) and (self._allow_dual_lane_variants or primary_lane != "normal")
            if normal_variant and include_normal_variant and (symbol, "normal") not in seen:
                items.append(("normal", symbol, normal_variant))
                seen.add((symbol, "normal"))
        return items

    def _recent_closed_trade_rows(self, broker, lookback_days: Optional[int] = None) -> List[Dict]:
        if not broker:
            return []
        rows: List[Dict] = []
        cutoff = None
        if lookback_days:
            cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        for trade in list(getattr(broker, "trade_log", []) or []):
            realized = self._safe_number(trade.get("realized_pnl"), 0.0)
            if abs(realized) < 1e-9:
                continue
            filled_at_raw = trade.get("filled_at")
            if not filled_at_raw:
                continue
            try:
                filled_at = datetime.fromisoformat(str(filled_at_raw))
            except Exception:
                continue
            filled_at = filled_at.astimezone(timezone.utc)
            if cutoff and filled_at < cutoff:
                continue
            rows.append({**trade, "filled_at_dt": filled_at})
        return rows

    def _lane_position_snapshot(self, broker, current_prices: Optional[Dict[str, float]] = None) -> Dict[str, Dict]:
        snapshot = {
            lane: {"open_positions": 0, "position_value": 0.0, "current_exposure_pct": 0.0}
            for lane in self._lane_engine_order
        }
        if not broker:
            return snapshot
        current_prices = current_prices or {}
        portfolio_value = max(self._safe_number(getattr(broker, "portfolio_value", 0.0), 0.0), 1.0)
        for position_key, position, symbol, lane, plan in self._iter_open_positions(broker):
            if lane not in snapshot:
                snapshot[lane] = {"open_positions": 0, "position_value": 0.0, "current_exposure_pct": 0.0}
            current_price = (
                current_prices.get(position_key)
                or current_prices.get(symbol)
                or self._get_latest_close(symbol)
                or getattr(position, "avg_cost", 0.0)
            )
            position_value = abs(getattr(position, "quantity", 0)) * self._safe_number(current_price, getattr(position, "avg_cost", 0.0))
            snapshot[lane]["open_positions"] += 1
            snapshot[lane]["position_value"] += position_value
        for lane, payload in snapshot.items():
            payload["current_exposure_pct"] = round(payload["position_value"] / portfolio_value, 6)
            payload["position_value"] = round(payload["position_value"], 2)
        return snapshot

    def _detect_allocation_regime(self) -> Dict[str, object]:
        target = self._components.get("allocation_regime")
        if not isinstance(target, dict):
            target = {}
            self._components["allocation_regime"] = target

        now = time.time()
        last_refresh_ts = self._safe_number(target.get("updated_at_ts"), 0.0)
        if target and (now - last_refresh_ts) < (4 * 3600):
            return dict(target)

        latest_feature_rows = self._components.get("latest_feature_rows", {})
        vol_ratios = []
        stressed_flags = []
        vix_values = []
        for row in latest_feature_rows.values():
            if row is None:
                continue
            vol_ratio = self._safe_number(row.get("vol_regime_ratio", np.nan), np.nan)
            if not np.isnan(vol_ratio):
                vol_ratios.append(vol_ratio)
            stressed_flags.append(self._safe_number(row.get("vol_regime_stressed", 0.0), 0.0))
            vix_value = self._safe_number(row.get("macro_vix_level", row.get("vix_level", np.nan)), np.nan)
            if not np.isnan(vix_value):
                vix_values.append(vix_value)

        mean_vol_ratio = float(np.nanmean(vol_ratios)) if vol_ratios else 1.0
        stressed_share = float(np.nanmean(stressed_flags)) if stressed_flags else 0.0
        avg_vix = float(np.nanmean(vix_values)) if vix_values else 18.0

        crypto_rows = [
            signal for signal in self._signal_store.values()
            if isinstance(signal, dict) and str(signal.get("lane") or "").lower() == "crypto"
        ]
        crypto_stress = 0.0
        if crypto_rows:
            crypto_stress = float(np.mean([
                abs(self._safe_number((signal.get("intraday_overlay") or {}).get("book_pressure"), 0.0))
                + abs(self._safe_number((signal.get("intraday_overlay") or {}).get("spread_velocity"), 0.0)) * 1000.0
                for signal in crypto_rows
            ]))

        regime = "trending_low_vol"
        base_allocations = {"normal": 0.50, "day": 0.30, "crypto": 0.20}
        if avg_vix >= 30 or stressed_share >= 0.60:
            regime = "risk_off_crisis"
            base_allocations = {"normal": 0.55, "day": 0.30, "crypto": 0.15}
        elif mean_vol_ratio >= 1.25 or stressed_share >= 0.35 or crypto_stress >= 0.55:
            regime = "high_vol_choppy"
            base_allocations = {"normal": 0.25, "day": 0.45, "crypto": 0.30}

        target.clear()
        target.update(
            {
                "regime": regime,
                "base_allocations": base_allocations,
                "mean_vol_ratio": round(mean_vol_ratio, 4),
                "stressed_share": round(stressed_share, 4),
                "avg_vix": round(avg_vix, 3),
                "crypto_stress": round(crypto_stress, 4),
                "updated_at_ts": now,
                "updated_at": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
            }
        )
        return dict(target)

    def _refresh_lane_allocator(self, broker, current_prices: Optional[Dict[str, float]] = None) -> Dict[str, Dict]:
        target = self._components.get("lane_allocator")
        if not isinstance(target, dict):
            target = {}
            self._components["lane_allocator"] = target
        if not broker:
            target.clear()
            return {}

        regime_profile = self._detect_allocation_regime()
        base_weights = dict(regime_profile.get("base_allocations") or self._lane_base_allocations)
        base_total = max(sum(base_weights.values()), 1e-9)
        normalized_base = {lane: weight / base_total for lane, weight in base_weights.items()}
        exposure = self._lane_position_snapshot(broker, current_prices=current_prices)
        closed_rows = self._recent_closed_trade_rows(broker, lookback_days=self._governor_lookback_days)
        drift_status = self._refresh_drift_status(broker)
        drift_lanes = drift_status.get("lanes", {}) if isinstance(drift_status, dict) else {}
        lane_rows: Dict[str, List[Dict]] = defaultdict(list)
        for trade in closed_rows:
            lane = str((trade.get("metadata") or {}).get("lane") or "normal").lower()
            lane_rows[lane].append(trade)

        raw_weights: Dict[str, float] = {}
        summaries: Dict[str, Dict] = {}
        for lane in self._lane_engine_order:
            rows = lane_rows.get(lane, [])
            wins = [self._safe_number(r.get("realized_pnl"), 0.0) for r in rows if self._safe_number(r.get("realized_pnl"), 0.0) > 0]
            losses = [abs(self._safe_number(r.get("realized_pnl"), 0.0)) for r in rows if self._safe_number(r.get("realized_pnl"), 0.0) < 0]
            gross_wins = sum(wins)
            gross_losses = sum(losses)
            profit_factor = gross_wins / gross_losses if gross_losses > 0 else (2.0 if gross_wins > 0 else 1.0)
            recent_pnl = sum(self._safe_number(r.get("realized_pnl"), 0.0) for r in rows)
            win_rate = (len(wins) / max(len(rows), 1)) if rows else 0.5
            pnl_pct = recent_pnl / max(self._safe_number(getattr(broker, "initial_capital", 0.0), 100000.0), 1.0)
            strength = 1.0
            if rows:
                strength += self._clamp((profit_factor - 1.0) * 0.35, -0.35, 0.45)
                strength += self._clamp((win_rate - 0.5) * 0.45, -0.18, 0.18)
                strength += self._clamp(pnl_pct * 8.0, -0.20, 0.20)
            raw_weights[lane] = normalized_base.get(lane, 0.0) * max(0.35, strength)
            summaries[lane] = {
                "recent_closed_trades": len(rows),
                "profit_factor": round(profit_factor, 3),
                "win_rate_pct": round(win_rate * 100.0, 1),
                "recent_pnl": round(recent_pnl, 2),
            }

        total_raw = max(sum(raw_weights.values()), 1e-9)
        provisional = {lane: raw_weights.get(lane, 0.0) / total_raw for lane in self._lane_engine_order}
        clamped = {
            lane: self._clamp(
                provisional.get(lane, normalized_base.get(lane, 0.0)),
                self._lane_min_allocations.get(lane, 0.05),
                self._lane_max_allocations.get(lane, 0.90),
            )
            for lane in self._lane_engine_order
        }
        clamped_total = max(sum(clamped.values()), 1e-9)
        final_weights = {lane: clamped[lane] / clamped_total for lane in self._lane_engine_order}

        target.clear()
        for lane in self._lane_engine_order:
            base_weight = normalized_base.get(lane, 0.0)
            target_weight = final_weights.get(lane, base_weight)
            current_exposure_pct = self._safe_number(exposure.get(lane, {}).get("current_exposure_pct"), 0.0)
            size_multiplier = self._clamp(
                target_weight / max(base_weight, 1e-6),
                0.50,
                1.60,
            )
            target[lane] = {
                "lane": lane,
                "lane_label": self._lane_label(lane),
                "base_capital_pct": round(base_weight, 4),
                "target_capital_pct": round(target_weight, 4),
                "size_multiplier": round(size_multiplier, 4),
                "open_positions": int(exposure.get(lane, {}).get("open_positions", 0)),
                "current_exposure_pct": round(current_exposure_pct, 4),
                "available_capital_pct": round(max(0.0, target_weight - current_exposure_pct), 4),
                "status": (
                    "boosted"
                    if size_multiplier > 1.06
                    else "throttled"
                    if size_multiplier < 0.94
                    else "neutral"
                ),
                "allocation_regime": regime_profile.get("regime"),
                **summaries.get(lane, {}),
            }
        return dict(target)

    def _refresh_governor_state(self, broker) -> Dict[str, Dict]:
        target = self._components.get("governor")
        if not isinstance(target, dict):
            target = {}
            self._components["governor"] = target
        if not broker:
            target.clear()
            return {}

        closed_rows = self._recent_closed_trade_rows(broker, lookback_days=self._governor_lookback_days)
        drift_status = self._refresh_drift_status(broker)
        drift_lanes = drift_status.get("lanes", {}) if isinstance(drift_status, dict) else {}
        lane_rows: Dict[str, List[Dict]] = defaultdict(list)
        for trade in closed_rows:
            lane = str((trade.get("metadata") or {}).get("lane") or "normal").lower()
            lane_rows[lane].append(trade)

        target.clear()
        for lane in self._lane_engine_order:
            rows = lane_rows.get(lane, [])
            wins = [self._safe_number(r.get("realized_pnl"), 0.0) for r in rows if self._safe_number(r.get("realized_pnl"), 0.0) > 0]
            losses = [abs(self._safe_number(r.get("realized_pnl"), 0.0)) for r in rows if self._safe_number(r.get("realized_pnl"), 0.0) < 0]
            gross_wins = sum(wins)
            gross_losses = sum(losses)
            lane_pf = gross_wins / gross_losses if gross_losses > 0 else (2.0 if gross_wins > 0 else 1.0)
            lane_pnl = sum(self._safe_number(r.get("realized_pnl"), 0.0) for r in rows)
            disabled_setups = set()
            time_bucket_actions: Dict[str, Dict] = {}
            setup_stats: Dict[str, Dict] = defaultdict(lambda: {"count": 0, "wins": 0, "pnl": 0.0, "gross_wins": 0.0, "gross_losses": 0.0})
            time_stats: Dict[str, Dict] = defaultdict(lambda: {"count": 0, "wins": 0, "pnl": 0.0, "gross_wins": 0.0, "gross_losses": 0.0})

            for trade in rows:
                metadata = trade.get("metadata") or {}
                realized = self._safe_number(trade.get("realized_pnl"), 0.0)
                setup_id = str(metadata.get("setup_id") or "unclassified")
                time_bucket = str(metadata.get("entry_time_bucket") or metadata.get("time_bucket") or "unknown")
                for bucket_name, bucket in ((setup_id, setup_stats[setup_id]), (time_bucket, time_stats[time_bucket])):
                    bucket["count"] += 1
                    bucket["pnl"] += realized
                    if realized > 0:
                        bucket["wins"] += 1
                        bucket["gross_wins"] += realized
                    elif realized < 0:
                        bucket["gross_losses"] += abs(realized)

            for setup_id, stat in setup_stats.items():
                if stat["count"] < self._governor_min_setup_trades:
                    continue
                pf = stat["gross_wins"] / stat["gross_losses"] if stat["gross_losses"] > 0 else (2.0 if stat["gross_wins"] > 0 else 1.0)
                if pf < self._governor_setup_pf_floor and stat["pnl"] < 0:
                    disabled_setups.add(setup_id)

            for bucket_name, stat in time_stats.items():
                if stat["count"] < self._governor_min_time_bucket_trades:
                    continue
                pf = stat["gross_wins"] / stat["gross_losses"] if stat["gross_losses"] > 0 else (2.0 if stat["gross_wins"] > 0 else 1.0)
                win_rate = stat["wins"] / max(stat["count"], 1)
                if pf < (self._governor_time_bucket_pf_floor * 0.75) and stat["pnl"] < 0 and win_rate < 0.35:
                    time_bucket_actions[bucket_name] = {"blocked": True, "size_multiplier": 0.0}
                elif pf < self._governor_time_bucket_pf_floor and stat["pnl"] < 0:
                    time_bucket_actions[bucket_name] = {
                        "blocked": False,
                        "size_multiplier": self._governor_time_bucket_throttle_mult,
                    }

            lane_status = "active"
            lane_throttle_multiplier = 1.0
            if len(rows) >= self._governor_min_setup_trades and lane_pf < self._governor_setup_review_pf and lane_pnl < 0:
                lane_status = "review"
                lane_throttle_multiplier = self._governor_lane_throttle_mult
            lane_drift = drift_lanes.get(lane, {}) if isinstance(drift_lanes, dict) else {}
            if lane_drift.get("paused"):
                lane_status = "paused"
                lane_throttle_multiplier = 0.0

            target[lane] = {
                "lane": lane,
                "lane_label": self._lane_label(lane),
                "status": lane_status,
                "lane_throttle_multiplier": round(lane_throttle_multiplier, 4),
                "recent_closed_trades": len(rows),
                "profit_factor": round(lane_pf, 3),
                "recent_pnl": round(lane_pnl, 2),
                "disabled_setups": sorted(disabled_setups),
                "time_bucket_actions": time_bucket_actions,
                "drift": lane_drift,
            }
        return dict(target)

    def _governor_decision(self, signal: Dict) -> Dict:
        lane = str(signal.get("lane") or "normal").lower()
        target = self._components.get("governor") if isinstance(self._components.get("governor"), dict) else {}
        lane_state = target.get(lane, {}) if isinstance(target, dict) else {}
        setup_id = str(signal.get("setup_id") or "unclassified")
        time_bucket = str(signal.get("time_bucket") or "unknown")
        if str(lane_state.get("status") or "").lower() == "paused":
            return {"allow": False, "reason": "lane_paused", "size_multiplier": 0.0, "status": "paused"}
        if setup_id in set(lane_state.get("disabled_setups", []) or []):
            return {"allow": False, "reason": "setup_retired", "size_multiplier": 0.0, "status": lane_state.get("status", "active")}
        size_multiplier = self._safe_number(lane_state.get("lane_throttle_multiplier"), 1.0)
        action = (lane_state.get("time_bucket_actions") or {}).get(time_bucket, {})
        if action.get("blocked"):
            return {"allow": False, "reason": "time_window_retired", "size_multiplier": 0.0, "status": lane_state.get("status", "active")}
        size_multiplier *= self._safe_number(action.get("size_multiplier"), 1.0)
        return {
            "allow": size_multiplier > 0.0,
            "reason": "lane_review_throttle" if size_multiplier < 0.99 else "active",
            "size_multiplier": round(size_multiplier, 4),
            "status": lane_state.get("status", "active"),
        }

    def _crypto_execution_filter(self, signal: Dict) -> Dict:
        intraday = signal.get("intraday_overlay", {}) if isinstance(signal.get("intraday_overlay"), dict) else {}
        best_bid = self._safe_number(intraday.get("best_bid"), 0.0)
        best_ask = self._safe_number(intraday.get("best_ask"), 0.0)
        mid_price = (best_bid + best_ask) / 2.0 if best_bid > 0 and best_ask > 0 else 0.0
        spread_pct = ((best_ask - best_bid) / mid_price) * 100.0 if mid_price > 0 else 999.0
        depth_age_seconds = self._safe_number(intraday.get("depth_age_seconds"), 999.0)
        tick_age_seconds = self._safe_number(intraday.get("tick_age_seconds"), 999.0)
        signal_age_seconds = max(depth_age_seconds, tick_age_seconds)
        book_pressure = abs(self._safe_number(intraday.get("book_pressure"), 0.0))
        depth_imbalance = abs(self._safe_number(intraday.get("book_imbalance"), 0.0))
        spread_velocity = self._safe_number(intraday.get("spread_velocity"), 0.0)
        flicker_filter_retention = self._safe_number(intraday.get("flicker_filter_retention"), 1.0)
        trade_window_active = bool(intraday.get("trade_window_active", True))
        if bool(intraday.get("quote_only_fallback", False)):
            return {"allow": False, "reason": "crypto_quote_only_fallback", "size_multiplier": 0.0}
        if spread_pct > self._crypto_scalper_max_spread_pct:
            return {"allow": False, "reason": "crypto_wide_spread", "size_multiplier": 0.0}
        if depth_age_seconds > self._crypto_scalper_max_depth_age_seconds:
            return {"allow": False, "reason": "crypto_stale_book", "size_multiplier": 0.0}
        if tick_age_seconds > self._crypto_scalper_max_tick_age_seconds:
            return {"allow": False, "reason": "crypto_stale_tick", "size_multiplier": 0.0}
        if signal_age_seconds > self._crypto_scalper_max_signal_age_seconds:
            return {"allow": False, "reason": "crypto_stale_signal", "size_multiplier": 0.0}
        if flicker_filter_retention < 0.10:
            return {"allow": False, "reason": "crypto_flickering_liquidity", "size_multiplier": 0.0}
        if book_pressure < self._crypto_scalper_min_book_pressure:
            return {"allow": False, "reason": "crypto_weak_book_pressure", "size_multiplier": 0.0}
        if depth_imbalance < self._crypto_scalper_min_depth_imbalance:
            return {"allow": False, "reason": "crypto_weak_depth_imbalance", "size_multiplier": 0.0}
        trade_window_confidence_floor = 0.62 if self._event_window_mode else 0.80
        if not trade_window_active and self._safe_number(intraday.get("confidence"), 0.0) < trade_window_confidence_floor:
            return {"allow": False, "reason": "crypto_outside_prime_window", "size_multiplier": 0.0}
        size_multiplier = 1.0
        if spread_pct > (self._crypto_scalper_max_spread_pct * 0.65):
            size_multiplier *= 0.75
        if signal_age_seconds > (self._crypto_scalper_max_signal_age_seconds * 0.65):
            size_multiplier *= 0.80
        if flicker_filter_retention < 0.45:
            size_multiplier *= 0.85
        execution_mode = (
            "taker"
            if book_pressure >= self._crypto_taker_book_pressure_threshold and spread_velocity > self._crypto_taker_spread_velocity_threshold
            else "maker"
        )
        return {
            "allow": True,
            "reason": "active",
            "size_multiplier": round(size_multiplier, 4),
            "spread_pct": round(spread_pct, 5),
            "signal_age_seconds": round(signal_age_seconds, 3),
            "spread_velocity": round(spread_velocity, 8),
            "flicker_filter_retention": round(flicker_filter_retention, 4),
            "execution_mode": execution_mode,
        }

    @staticmethod
    def _market_timezone(market: str) -> ZoneInfo:
        market_key = str(market).upper()
        if market_key == "IN":
            return ZoneInfo("Asia/Kolkata")
        if market_key == "CRYPTO":
            return ZoneInfo("UTC")
        return ZoneInfo("America/New_York")

    def _market_session_clock(self, timestamp: datetime, market: str) -> Tuple[datetime, datetime, datetime]:
        market_key = str(market).upper()
        local_ts = timestamp.astimezone(self._market_timezone(market))
        if market_key == "IN":
            open_local = local_ts.replace(hour=9, minute=15, second=0, microsecond=0)
            close_local = local_ts.replace(hour=15, minute=30, second=0, microsecond=0)
        elif market_key == "CRYPTO":
            open_local = local_ts.replace(hour=0, minute=0, second=0, microsecond=0)
            close_local = local_ts.replace(hour=23, minute=59, second=59, microsecond=0)
        else:
            open_local = local_ts.replace(hour=9, minute=30, second=0, microsecond=0)
            close_local = local_ts.replace(hour=16, minute=0, second=0, microsecond=0)
        return local_ts, open_local, close_local

    def _resolve_tick_volume_delta(self, tick, state: Dict) -> float:
        raw_volume = max(0.0, float(getattr(tick, "volume", 0) or 0.0))
        if str(getattr(tick, "market", "US")).upper() == "IN":
            previous_reported = float(state.get("last_reported_volume", 0.0) or 0.0)
            delta = raw_volume - previous_reported if raw_volume >= previous_reported else raw_volume
            state["last_reported_volume"] = raw_volume
            return max(delta, 1.0 if raw_volume > 0 else 0.0)
        state["last_reported_volume"] = raw_volume
        return max(raw_volume, 1.0)

    def _on_realtime_tick(self, tick) -> None:
        if not self._day_trading_mode or tick is None or float(getattr(tick, "price", 0.0) or 0.0) <= 0:
            return

        market = str(getattr(tick, "market", "US") or "US").upper()
        local_ts, open_local, close_local = self._market_session_clock(tick.timestamp, market)
        if local_ts < open_local or local_ts > close_local:
            return

        session_key = f"{market}:{local_ts.date().isoformat()}"
        with self._intraday_state_lock:
            state = self._intraday_state.get(tick.symbol)
            if not state or state.get("session_key") != session_key:
                state = {
                    "session_key": session_key,
                    "market": market,
                    "session_open_utc": open_local.astimezone(timezone.utc),
                    "session_close_utc": close_local.astimezone(timezone.utc),
                    "session_open_price": float(tick.price),
                    "session_high": float(tick.price),
                    "session_low": float(tick.price),
                    "opening_range_high": float(tick.price),
                    "opening_range_low": float(tick.price),
                    "opening_range_complete": False,
                    "last_price": float(tick.price),
                    "last_signed_direction": 0,
                    "last_reported_volume": 0.0,
                    "session_volume": 0.0,
                    "vwap_num": 0.0,
                    "vwap_den": 0.0,
                    "cum_delta": 0.0,
                    "recent_flow": deque(maxlen=720),
                }
                self._intraday_state[tick.symbol] = state

            volume_delta = self._resolve_tick_volume_delta(tick, state)
            last_price = float(state.get("last_price", tick.price) or tick.price)
            direction = 1 if float(tick.price) > last_price else -1 if float(tick.price) < last_price else int(
                state.get("last_signed_direction", 0) or 0
            )
            signed_volume = direction * volume_delta

            state["session_high"] = max(float(state.get("session_high", tick.price) or tick.price), float(tick.price))
            state["session_low"] = min(float(state.get("session_low", tick.price) or tick.price), float(tick.price))
            state["session_volume"] = float(state.get("session_volume", 0.0) or 0.0) + volume_delta
            state["vwap_num"] = float(state.get("vwap_num", 0.0) or 0.0) + (float(tick.price) * volume_delta)
            state["vwap_den"] = float(state.get("vwap_den", 0.0) or 0.0) + volume_delta
            state["cum_delta"] = float(state.get("cum_delta", 0.0) or 0.0) + signed_volume
            state["last_price"] = float(tick.price)
            state["last_signed_direction"] = direction
            state["last_tick_ts"] = tick.timestamp

            if local_ts <= open_local + timedelta(minutes=self._day_trade_opening_range_minutes):
                state["opening_range_high"] = max(
                    float(state.get("opening_range_high", tick.price) or tick.price),
                    float(tick.price),
                )
                state["opening_range_low"] = min(
                    float(state.get("opening_range_low", tick.price) or tick.price),
                    float(tick.price),
                )
            else:
                state["opening_range_complete"] = True

            state["recent_flow"].append(
                {
                    "timestamp": tick.timestamp,
                    "price": float(tick.price),
                    "volume": volume_delta,
                    "signed_volume": signed_volume,
                    "cum_delta": float(state["cum_delta"]),
                }
            )

    def _get_intraday_state_snapshot(self, symbol: str) -> Dict:
        with self._intraday_state_lock:
            raw = self._intraday_state.get(symbol)
            if not raw:
                return {}
            snapshot = {k: v for k, v in raw.items() if k != "recent_flow"}
            snapshot["recent_flow"] = list(raw.get("recent_flow", []))
            return snapshot

    def _compute_intraday_overlay(self, symbol: str, latest_row) -> Dict[str, float]:
        if not self._day_trading_mode:
            return {}

        try:
            from core.realtime_engine import DEPTH_BUFFER, PRICE_BUFFER
        except Exception:
            return {}

        bar_1m = PRICE_BUFFER.get_ohlcv(symbol, 60)
        bar_5m = PRICE_BUFFER.get_ohlcv(symbol, 300)
        bar_15m = PRICE_BUFFER.get_ohlcv(symbol, 900) or bar_5m
        if not bar_5m or int(bar_5m.get("tick_count", 0) or 0) < self._day_trade_min_tick_count:
            return {}

        def _bar_return(bar: Optional[Dict]) -> float:
            if not bar:
                return 0.0
            open_px = float(bar.get("open", 0.0) or 0.0)
            close_px = float(bar.get("close", 0.0) or 0.0)
            if open_px <= 0:
                return 0.0
            return (close_px / open_px) - 1.0

        def _row_value(name: str) -> float:
            if latest_row is None or name not in latest_row.index:
                return 0.0
            raw = latest_row.get(name, 0.0)
            if pd.isna(raw):
                return 0.0
            try:
                return float(raw)
            except Exception:
                return 0.0

        ret_1m = _bar_return(bar_1m)
        ret_5m = _bar_return(bar_5m)
        ret_15m = _bar_return(bar_15m)

        close_5m = float(bar_5m.get("close", 0.0) or 0.0)
        high_15m = float(bar_15m.get("high", close_5m) or close_5m)
        low_15m = float(bar_15m.get("low", close_5m) or close_5m)
        span = max(high_15m - low_15m, max(close_5m, 1.0) * 0.002)
        range_pos = (close_5m - low_15m) / span
        range_centered = self._clamp((range_pos - 0.5) * 2.0, -1.0, 1.0)

        vol_1m = float((bar_1m or {}).get("volume", 0.0) or 0.0)
        vol_15m = float((bar_15m or {}).get("volume", 0.0) or 0.0)
        expected_1m_volume = vol_15m / (15.0 if bar_15m else 5.0) if vol_15m > 0 else 0.0
        volume_spike = self._clamp(
            (vol_1m / max(expected_1m_volume, 1.0)) if vol_1m > 0 else 0.0,
            0.0,
            4.0,
        )

        trend_bias = self._clamp(_row_value("trend_composite"), -1.5, 1.5)
        momentum_bias = self._clamp(_row_value("momentum_composite"), -1.5, 1.5)
        official_event = self._clamp(_row_value("official_event_signal"), -1.0, 1.5)
        filing_event = self._clamp(_row_value("filing_event_signal"), -1.0, 1.5)
        sentiment_z = self._clamp(_row_value("weighted_sentiment_zscore"), -2.0, 2.0)
        source_quality = self._clamp(_row_value("source_quality_signal"), -1.0, 1.5)
        news_volume = self._clamp(_row_value("news_volume_spike"), -1.0, 2.0)
        travel_activity = self._clamp(_row_value("travel_activity_change"), -1.0, 1.0)

        news_boost = self._clamp(
            (official_event * 0.42)
            + (filing_event * 0.32)
            + (sentiment_z * 0.14)
            + (news_volume * 0.12)
            + (source_quality * 0.20)
            + (travel_activity * 0.12),
            -1.2,
            1.8,
        )

        legacy_score = (
            (ret_1m * 28.0)
            + (ret_5m * 96.0)
            + (ret_15m * 62.0)
            + (max(0.0, volume_spike - 1.0) * 0.26)
            + (range_centered * 0.24)
            + (trend_bias * 0.18)
            + (momentum_bias * 0.20)
            + (news_boost * 0.42)
        )
        if volume_spike < 0.60:
            legacy_score *= 0.82
        legacy_score = self._clamp(legacy_score, -2.5, 2.5)

        state = self._get_intraday_state_snapshot(symbol)
        if not state:
            direction = "neutral"
            if legacy_score >= self._day_trade_intraday_score_threshold:
                direction = "buy"
            elif legacy_score <= -self._day_trade_intraday_score_threshold:
                direction = "sell"
            confidence = self._clamp(abs(legacy_score) / 1.45, 0.0, 0.99)
            note_parts: List[str] = []
            if abs(ret_5m) >= 0.002:
                note_parts.append(f"5m move {ret_5m * 100:+.2f}%")
            if volume_spike >= 1.35:
                note_parts.append(f"RVOL {volume_spike:.1f}x")
            if news_boost >= self._day_trade_news_boost_threshold:
                note_parts.append("official/news catalyst")
            return {
                "score": round(legacy_score, 4),
                "direction": direction,
                "confidence": round(confidence, 4),
                "regime": "legacy_intraday",
                "setup": "legacy_momentum",
                "ensemble_long": round(max(legacy_score, 0.0), 4),
                "ensemble_short": round(max(-legacy_score, 0.0), 4),
                "volume_spike": round(volume_spike, 3),
                "ret_1m": round(ret_1m, 5),
                "ret_5m": round(ret_5m, 5),
                "ret_15m": round(ret_15m, 5),
                "range_position": round(range_pos, 4),
                "tick_count": int(bar_5m.get("tick_count", 0) or 0),
                "news_boost": round(news_boost, 4),
                "note": " | ".join(note_parts),
                "trade_window_active": True,
                "risk_adjustments": {},
            }

        depth_snapshot = DEPTH_BUFFER.filtered_snapshot(
            symbol,
            min_lifetime_ms=self._crypto_obi_min_lifetime_ms,
            max_updates_per_second=self._crypto_obi_max_updates_per_second,
            min_inter_update_delay_ms=self._crypto_obi_min_inter_update_delay_ms,
        )
        depth_updated_at = depth_snapshot.get("updated_at")
        depth_age_seconds = None
        if depth_updated_at:
            try:
                depth_age_seconds = max(
                    0.0,
                    (datetime.now(timezone.utc) - datetime.fromisoformat(str(depth_updated_at))).total_seconds(),
                )
            except Exception:
                depth_age_seconds = None
        depth_available = bool(depth_snapshot) and (
            depth_age_seconds is None or depth_age_seconds <= (self._crypto_signal_stale_seconds * 2)
        )

        def _safe_level_qty(level) -> float:
            if isinstance(level, (list, tuple)) and len(level) >= 2:
                try:
                    return float(level[1] or 0.0)
                except Exception:
                    return 0.0
            if isinstance(level, dict):
                raw = level.get("quantity", level.get("qty", 0.0))
                try:
                    return float(raw or 0.0)
                except Exception:
                    return 0.0
            return 0.0

        best_bid_qty = float(depth_snapshot.get("best_bid_qty", 0.0) or 0.0)
        best_ask_qty = float(depth_snapshot.get("best_ask_qty", 0.0) or 0.0)
        depth_bids = depth_snapshot.get("filtered_bids") or depth_snapshot.get("bids", []) or []
        depth_asks = depth_snapshot.get("filtered_asks") or depth_snapshot.get("asks", []) or []
        bid_depth_qty = sum(_safe_level_qty(level) for level in depth_bids[:5])
        ask_depth_qty = sum(_safe_level_qty(level) for level in depth_asks[:5])
        if bid_depth_qty <= 0 and best_bid_qty > 0:
            bid_depth_qty = best_bid_qty
        if ask_depth_qty <= 0 and best_ask_qty > 0:
            ask_depth_qty = best_ask_qty

        top_book_imbalance = 0.0
        if (best_bid_qty + best_ask_qty) > 0:
            top_book_imbalance = (best_bid_qty - best_ask_qty) / (best_bid_qty + best_ask_qty)
        depth_book_imbalance = 0.0
        if (bid_depth_qty + ask_depth_qty) > 0:
            depth_book_imbalance = (bid_depth_qty - ask_depth_qty) / (bid_depth_qty + ask_depth_qty)
        filtered_depth_imbalance = self._safe_number(depth_snapshot.get("filtered_book_imbalance"), depth_book_imbalance)
        if depth_bids or depth_asks:
            depth_book_imbalance = filtered_depth_imbalance

        best_bid = float(depth_snapshot.get("best_bid", 0.0) or 0.0)
        best_ask = float(depth_snapshot.get("best_ask", 0.0) or 0.0)
        spread_velocity = self._safe_number(depth_snapshot.get("spread_velocity"), 0.0)
        microprice_bias = 0.0
        if best_bid > 0 and best_ask > 0 and (best_bid_qty + best_ask_qty) > 0:
            microprice = ((best_ask * best_bid_qty) + (best_bid * best_ask_qty)) / (best_bid_qty + best_ask_qty)
            mid_price = (best_bid + best_ask) / 2.0
            if mid_price > 0:
                microprice_bias = self._clamp(((microprice / mid_price) - 1.0) / 0.0008, -1.0, 1.0)

        book_pressure = self._clamp(
            (depth_book_imbalance * 0.60) + (top_book_imbalance * 0.25) + (microprice_bias * 0.15),
            -1.2,
            1.2,
        )
        if not depth_available:
            top_book_imbalance = 0.0
            depth_book_imbalance = 0.0
            microprice_bias = 0.0
            book_pressure = 0.0

        recent_flow = state.get("recent_flow", []) or []
        last_tick_ts = state.get("last_tick_ts")
        market = str(state.get("market", "US") or "US").upper()
        trade_window_active = True
        session_age_minutes = 0.0
        minutes_to_close = 0.0
        if isinstance(last_tick_ts, datetime):
            local_ts, open_local, close_local = self._market_session_clock(last_tick_ts, market)
            session_age_minutes = max(0.0, (local_ts - open_local).total_seconds() / 60.0)
            minutes_to_close = max(0.0, (close_local - local_ts).total_seconds() / 60.0)
            trade_window_active = (
                session_age_minutes <= self._day_trade_active_open_minutes
                or minutes_to_close <= self._day_trade_power_hour_minutes
            )

        def _flow_window(seconds: int) -> List[Dict]:
            if not recent_flow:
                return []
            cutoff = recent_flow[-1]["timestamp"] - timedelta(seconds=seconds)
            return [item for item in recent_flow if item["timestamp"] >= cutoff]

        def _flow_stats(items: List[Dict]) -> Dict[str, float]:
            if not items:
                return {
                    "ofi": 0.0,
                    "delta_norm": 0.0,
                    "total_volume": 0.0,
                    "price_return": 0.0,
                    "high": close_5m,
                    "low": close_5m,
                }
            prices = [float(item.get("price", close_5m) or close_5m) for item in items]
            volumes = [float(item.get("volume", 0.0) or 0.0) for item in items]
            total_volume = sum(volumes)
            signed_volume = sum(float(item.get("signed_volume", 0.0) or 0.0) for item in items)
            delta_first = float(items[0].get("cum_delta", 0.0) or 0.0)
            delta_last = float(items[-1].get("cum_delta", 0.0) or 0.0)
            first_price = prices[0]
            price_return = (prices[-1] / first_price) - 1.0 if first_price > 0 else 0.0
            norm = max(total_volume, 1.0)
            return {
                "ofi": signed_volume / norm,
                "delta_norm": (delta_last - delta_first) / norm,
                "total_volume": total_volume,
                "price_return": price_return,
                "high": max(prices),
                "low": min(prices),
            }

        flow_short = _flow_stats(_flow_window(90))
        flow_medium = _flow_stats(_flow_window(300))
        flow_long = _flow_stats(_flow_window(900))
        prior_flow_items = _flow_window(900)
        exclude_count = max(3, len(_flow_window(120)) or 0)
        prior_reference = prior_flow_items[:-exclude_count] if len(prior_flow_items) > exclude_count else prior_flow_items[:-1]
        prior_high = max((float(item.get("price", close_5m) or close_5m) for item in prior_reference), default=flow_long["high"])
        prior_low = min((float(item.get("price", close_5m) or close_5m) for item in prior_reference), default=flow_long["low"])

        session_vwap = close_5m
        if float(state.get("vwap_den", 0.0) or 0.0) > 0:
            session_vwap = float(state.get("vwap_num", 0.0) or 0.0) / float(state.get("vwap_den", 1.0) or 1.0)
        session_open_price = float(state.get("session_open_price", close_5m) or close_5m)
        vwap_bias = self._clamp(((close_5m / max(session_vwap, 0.01)) - 1.0) / 0.0035, -1.5, 1.5)
        vwap_slope = self._clamp(((session_vwap / max(session_open_price, 0.01)) - 1.0) / 0.0045, -1.2, 1.2)
        atr_pct = max(abs(_row_value("atr_pct")), 0.002)
        realized_range_5m = (float(bar_5m.get("high", close_5m) or close_5m) - float(bar_5m.get("low", close_5m) or close_5m)) / max(close_5m, 0.01)
        volatility_expansion = self._clamp(
            (realized_range_5m / max(atr_pct * 0.35, 0.0015)) - 1.0,
            -1.0,
            1.5,
        )
        volatility_signed = volatility_expansion * (
            1.0 if (flow_medium["price_return"] >= 0 or vwap_bias >= 0) else -1.0
        )

        opening_high = float(state.get("opening_range_high", close_5m) or close_5m)
        opening_low = float(state.get("opening_range_low", close_5m) or close_5m)
        opening_range_complete = bool(state.get("opening_range_complete", False))
        orb_long = 1.0 if opening_range_complete and close_5m > (opening_high * 1.0007) else 0.0
        orb_short = 1.0 if opening_range_complete and close_5m < (opening_low * 0.9993) else 0.0
        orb_retest_long = 1.0 if (
            orb_long
            and flow_short["low"] <= (opening_high * 1.002)
            and flow_short["ofi"] > 0.04
            and close_5m > opening_high
        ) else 0.0
        orb_retest_short = 1.0 if (
            orb_short
            and flow_short["high"] >= (opening_low * 0.998)
            and flow_short["ofi"] < -0.04
            and close_5m < opening_low
        ) else 0.0

        sweep_long = 1.0 if (
            flow_short["low"] < (prior_low * 0.9994)
            and close_5m > prior_low
            and (flow_short["ofi"] > 0.03 or book_pressure > 0.10)
            and ret_1m > -0.0005
        ) else 0.0
        sweep_short = 1.0 if (
            flow_short["high"] > (prior_high * 1.0006)
            and close_5m < prior_high
            and (flow_short["ofi"] < -0.03 or book_pressure < -0.10)
            and ret_1m < 0.0005
        ) else 0.0

        vwap_pullback_buffer = max(close_5m * atr_pct * self._day_trade_vwap_pullback_atr_mult, close_5m * 0.0008)
        vwap_pullback_long = 1.0 if (
            close_5m > session_vwap
            and vwap_slope >= -0.02
            and abs(flow_short["low"] - session_vwap) <= vwap_pullback_buffer
            and flow_short["ofi"] > 0.02
            and ret_1m > -0.0003
        ) else 0.0
        vwap_pullback_short = 1.0 if (
            close_5m < session_vwap
            and vwap_slope <= 0.02
            and abs(flow_short["high"] - session_vwap) <= vwap_pullback_buffer
            and flow_short["ofi"] < -0.02
            and ret_1m < 0.0003
        ) else 0.0

        orderflow_long = self._clamp(
            (
                (max(flow_short["ofi"], 0.0) * 1.20)
                + max(flow_medium["delta_norm"], 0.0)
                + (max(book_pressure, 0.0) * 0.42)
            ) * 2.0,
            0.0,
            1.0,
        )
        orderflow_short = self._clamp(
            (
                (max(-flow_short["ofi"], 0.0) * 1.20)
                + max(-flow_medium["delta_norm"], 0.0)
                + (max(-book_pressure, 0.0) * 0.42)
            ) * 2.0,
            0.0,
            1.0,
        )

        momentum_long = self._clamp(
            (orb_retest_long * 0.55)
            + (max(vwap_bias, 0.0) * 0.16)
            + (max(flow_medium["price_return"], 0.0) / 0.0045 * 0.18)
            + (max(volume_spike - 1.0, 0.0) * 0.12),
            0.0,
            1.0,
        )
        momentum_short = self._clamp(
            (orb_retest_short * 0.55)
            + (max(-vwap_bias, 0.0) * 0.16)
            + (max(-flow_medium["price_return"], 0.0) / 0.0045 * 0.18)
            + (max(volume_spike - 1.0, 0.0) * 0.12),
            0.0,
            1.0,
        )
        news_long = self._clamp((max(news_boost, 0.0) * 0.55) + (max(volume_spike - 1.0, 0.0) * 0.12), 0.0, 1.0)
        news_short = self._clamp((max(-news_boost, 0.0) * 0.55) + (max(volume_spike - 1.0, 0.0) * 0.12), 0.0, 1.0)

        trend_score = self._clamp(
            (0.35 * vwap_bias)
            + (0.25 * flow_short["ofi"] * 1.6)
            + (0.20 * flow_medium["delta_norm"] * 1.4)
            + (0.12 * book_pressure)
            + (0.20 * volatility_signed),
            -1.5,
            1.5,
        )

        regime = "neutral"
        if trend_score >= 0.45:
            regime = "trend_up"
        elif trend_score <= -0.45:
            regime = "trend_down"
        elif sweep_long >= 0.55:
            regime = "mean_revert_long"
        elif sweep_short >= 0.55:
            regime = "mean_revert_short"

        long_score = (
            (0.30 * momentum_long)
            + (0.25 * sweep_long)
            + (0.20 * vwap_pullback_long)
            + (0.15 * orderflow_long)
            + (0.10 * news_long)
        )
        short_score = (
            (0.30 * momentum_short)
            + (0.25 * sweep_short)
            + (0.20 * vwap_pullback_short)
            + (0.15 * orderflow_short)
            + (0.10 * news_short)
        )
        if regime == "trend_up":
            long_score += (max(trend_score, 0.0) * 0.18) + (vwap_pullback_long * 0.08)
            short_score *= 0.72
        elif regime == "trend_down":
            short_score += (max(-trend_score, 0.0) * 0.18) + (vwap_pullback_short * 0.08)
            long_score *= 0.72
        elif regime == "mean_revert_long":
            long_score += sweep_long * 0.18
            short_score *= 0.82
        elif regime == "mean_revert_short":
            short_score += sweep_short * 0.18
            long_score *= 0.82

        if not trade_window_active:
            long_score *= 0.82
            short_score *= 0.82
            legacy_score *= 0.88

        ensemble_net = long_score - short_score
        intraday_score = self._clamp((legacy_score * 0.38) + (ensemble_net * 2.05) + (trend_score * 0.45), -2.5, 2.5)

        setup_scores = {
            "opening_range_momentum": max(momentum_long, momentum_short),
            "liquidity_sweep_reversal": max(sweep_long, sweep_short),
            "vwap_pullback_continuation": max(vwap_pullback_long, vwap_pullback_short),
            "orderflow_confirmation": max(orderflow_long, orderflow_short),
            "news_volatility": max(news_long, news_short),
        }
        dominant_setup = "hybrid_intraday"
        if setup_scores and max(setup_scores.values()) >= 0.05:
            dominant_setup = max(setup_scores, key=setup_scores.get)
        direction = "neutral"
        if intraday_score >= self._day_trade_intraday_score_threshold and long_score >= self._day_trade_ensemble_threshold:
            direction = "buy"
        elif intraday_score <= -self._day_trade_intraday_score_threshold and short_score >= self._day_trade_ensemble_threshold:
            direction = "sell"

        confidence = self._clamp(
            max(abs(ensemble_net), min(abs(trend_score), 1.0) * 0.85, abs(legacy_score) / 2.2),
            0.0,
            0.99,
        )
        confidence = self._clamp(confidence * (1.04 if trade_window_active else 0.92), 0.0, 0.99)

        base_stop_pct = self._clamp(atr_pct * 100.0 * 0.65, 0.35, 1.8)
        if dominant_setup == "liquidity_sweep_reversal":
            if direction == "buy":
                sweep_span_pct = abs((close_5m - flow_short["low"]) / max(close_5m, 0.01)) * 100.0
            elif direction == "sell":
                sweep_span_pct = abs((flow_short["high"] - close_5m) / max(close_5m, 0.01)) * 100.0
            else:
                sweep_span_pct = 0.0
            base_stop_pct = self._clamp(max(0.30, sweep_span_pct * 1.15), 0.30, 1.40)
        elif dominant_setup == "vwap_pullback_continuation":
            base_stop_pct = self._clamp(base_stop_pct * 0.85, 0.30, 1.45)
        elif dominant_setup == "opening_range_momentum":
            base_stop_pct = self._clamp(base_stop_pct * 0.95, 0.35, 1.60)
        reward_multiple = 2.2 if regime in {"trend_up", "trend_down"} else 1.7
        take_profit_pct = self._clamp(base_stop_pct * reward_multiple, max(base_stop_pct * 1.4, 0.70), 4.50)

        note_parts: List[str] = [f"{regime.replace('_', ' ')}"]
        if dominant_setup:
            note_parts.append(dominant_setup.replace("_", " "))
        if opening_range_complete and (orb_retest_long or orb_retest_short):
            note_parts.append("opening range retest")
        if vwap_pullback_long or vwap_pullback_short:
            note_parts.append("VWAP reclaim")
        if sweep_long or sweep_short:
            note_parts.append("liquidity sweep")
        if abs(flow_short["ofi"]) >= 0.05:
            note_parts.append(f"OFI {flow_short['ofi']:+.2f}")
        if depth_available and abs(book_pressure) >= 0.08:
            note_parts.append(f"book {book_pressure:+.2f}")
        if volume_spike >= 1.35:
            note_parts.append(f"RVOL {volume_spike:.1f}x")
        if news_boost >= self._day_trade_news_boost_threshold:
            note_parts.append("official/news catalyst")
        if not trade_window_active:
            note_parts.append("outside prime trade window")

        return {
            "score": round(intraday_score, 4),
            "direction": direction,
            "confidence": round(confidence, 4),
            "regime": regime,
            "setup": dominant_setup,
            "ensemble_long": round(long_score, 4),
            "ensemble_short": round(short_score, 4),
            "trend_score": round(trend_score, 4),
            "order_flow_imbalance": round(flow_short["ofi"], 4),
            "delta_slope": round(flow_medium["delta_norm"], 4),
            "book_imbalance": round(depth_book_imbalance, 4),
            "top_book_imbalance": round(top_book_imbalance, 4),
            "book_pressure": round(book_pressure, 4),
            "spread_velocity": round(spread_velocity, 8),
            "microprice_bias": round(microprice_bias, 4),
            "best_bid_qty": round(best_bid_qty, 6),
            "best_ask_qty": round(best_ask_qty, 6),
            "bid_depth_qty": round(bid_depth_qty, 6),
            "ask_depth_qty": round(ask_depth_qty, 6),
            "depth_available": depth_available,
            "flicker_filter_retention": round(self._safe_number(depth_snapshot.get("flicker_filter_retention"), 1.0), 4),
            "session_vwap": round(session_vwap, 4),
            "vwap_bias": round(vwap_bias, 4),
            "vwap_slope": round(vwap_slope, 4),
            "volume_spike": round(volume_spike, 3),
            "ret_1m": round(ret_1m, 5),
            "ret_5m": round(ret_5m, 5),
            "ret_15m": round(ret_15m, 5),
            "range_position": round(range_pos, 4),
            "tick_count": int(bar_5m.get("tick_count", 0) or 0),
            "news_boost": round(news_boost, 4),
            "opening_range_high": round(opening_high, 4),
            "opening_range_low": round(opening_low, 4),
            "trade_window_active": trade_window_active,
            "session_age_minutes": round(session_age_minutes, 1),
            "minutes_to_close": round(minutes_to_close, 1),
            "risk_adjustments": {
                "stop_loss_pct": round(base_stop_pct, 2),
                "take_profit_pct": round(take_profit_pct, 2),
                "trailing_stop_pct": round(self._day_trade_trail_stop_pct * 100.0, 2),
                "entry_note": " | ".join(note_parts),
            },
            "note": " | ".join(note_parts),
        }

    def _apply_day_trading_overlay(self, symbol: str, latest_row, scored: Dict[str, float]) -> None:
        overlay = self._compute_intraday_overlay(symbol, latest_row)
        if not overlay:
            return

        base_score = float(scored.get("final_score", 0.0) or 0.0)
        base_direction = str(scored.get("direction", "neutral") or "neutral")
        overlay_direction = str(overlay.get("direction", "neutral") or "neutral")
        overlay_confidence = float(overlay.get("confidence", 0.0) or 0.0)
        combined_score = (
            base_score * (1.0 - self._day_trade_intraday_weight)
            + float(overlay.get("score", 0.0)) * self._day_trade_intraday_weight
        )
        if overlay_direction != "neutral":
            should_override = (
                base_direction == overlay_direction
                or base_direction == "neutral"
                or abs(base_score) < 0.18
                or overlay_confidence >= 0.58
                or float(overlay.get("news_boost", 0.0) or 0.0) >= self._day_trade_news_boost_threshold
            )
            if should_override:
                scored["direction"] = overlay_direction
                scored["confidence"] = round(max(float(scored.get("confidence", 0.0) or 0.0), overlay_confidence), 4)
                scored["conviction_score"] = round(
                    min(10.0, max(float(scored.get("conviction_score", 0.0) or 0.0), overlay_confidence * 10.0)),
                    1,
                )
        elif str(scored.get("direction", "neutral")) == "neutral" and abs(combined_score) >= (
            self._day_trade_intraday_score_threshold * 0.75
        ):
            scored["direction"] = "buy" if combined_score > 0 else "sell"
            scored["confidence"] = round(max(float(scored.get("confidence", 0.0) or 0.0), overlay_confidence * 0.85), 4)
            scored["conviction_score"] = round(min(10.0, float(scored["confidence"]) * 10.0), 1)

        scored["final_score"] = round(combined_score, 4)
        scored["intraday_overlay"] = overlay

    def _build_crypto_intraday_signal(self, symbol: str) -> Optional[Dict]:
        try:
            from core.realtime_engine import DEPTH_BUFFER, PRICE_BUFFER
        except Exception:
            return None

        def _build_quote_only_fallback(depth_snapshot: Optional[Dict], note: str) -> Optional[Dict]:
            if not self._crypto_quote_fallback_enabled or latest_tick is None:
                return None
            last_price = float(getattr(latest_tick, "price", 0.0) or 0.0)
            if last_price <= 0:
                return None
            best_bid = float(getattr(latest_tick, "bid", 0.0) or 0.0)
            best_ask = float(getattr(latest_tick, "ask", 0.0) or 0.0)
            if best_bid <= 0 or best_ask <= 0 or best_ask < best_bid:
                half_spread = last_price * (self._crypto_quote_fallback_spread_pct / 100.0) / 2.0
                best_bid = max(last_price - half_spread, last_price * 0.999)
                best_ask = max(last_price + half_spread, best_bid)
            spread_pct = ((best_ask - best_bid) / max((best_bid + best_ask) / 2.0, 1e-9)) * 100.0
            tick_age = max(0.0, (datetime.now(timezone.utc) - latest_tick.timestamp).total_seconds())
            fallback_overlay = {
                "direction": "neutral",
                "confidence": 0.18,
                "setup": "quote_only_watch",
                "book_pressure": 0.0,
                "book_imbalance": 0.0,
                "top_book_imbalance": 0.0,
                "trend_score": 0.0,
                "volume_spike": 1.0,
                "score": 0.0,
                "news_boost": 0.0,
                "order_flow_imbalance": 0.0,
                "spread_velocity": round(self._safe_number((depth_snapshot or {}).get("spread_velocity"), 0.0), 8),
                "microprice_bias": 0.0,
                "best_bid_qty": round(float((depth_snapshot or {}).get("best_bid_qty", 0.0) or 0.0), 6),
                "best_ask_qty": round(float((depth_snapshot or {}).get("best_ask_qty", 0.0) or 0.0), 6),
                "bid_depth_qty": round(float((depth_snapshot or {}).get("best_bid_qty", 0.0) or 0.0), 6),
                "ask_depth_qty": round(float((depth_snapshot or {}).get("best_ask_qty", 0.0) or 0.0), 6),
                "flicker_filter_retention": round(self._safe_number((depth_snapshot or {}).get("flicker_filter_retention"), 1.0), 4),
                "vwap_bias": 0.0,
                "ret_5m": 0.0,
                "trade_window_active": True,
                "quote_only_fallback": True,
                "risk_adjustments": {
                    "stop_loss_pct": 0.9,
                    "take_profit_pct": 1.6,
                    "trailing_stop_pct": round(self._day_trade_trail_stop_pct * 100.0, 2),
                },
                "note": note,
            }
            return self._decorate_signal(
                symbol,
                {
                    "symbol": symbol,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "signal": "neutral",
                    "confidence": 0.18,
                    "conviction_score": 0.0,
                    "ensemble_score": 0.0,
                    "regime": "calm",
                    "regime_multiplier": 1.0,
                    "factor_scores": {
                        "trend": 0.0,
                        "momentum": 0.0,
                        "mean_revert": 0.0,
                        "volume": 0.0,
                        "sentiment": 0.0,
                        "earnings_propagation": 0.0,
                        "close_reversal": 0.0,
                    },
                    "factor_weights": {
                        "trend": 0.28,
                        "momentum": 0.24,
                        "mean_revert": 0.16,
                        "volume": 0.14,
                        "sentiment": 0.08,
                        "earnings_propagation": 0.0,
                        "close_reversal": 0.10,
                    },
                    "model_agreement": 0.0,
                    "model_breakdown": {},
                    "top_drivers": [{"feature": "Live Quote", "impact": round(spread_pct, 5), "direction": "mixed"}],
                    "waterfall_data": [],
                    "macro_warnings": ["depth feed unstable"],
                    "risk_parameters": {
                        "stop_loss_pct": 0.9,
                        "take_profit_pct": 1.6,
                        "trailing_stop_pct": round(self._day_trade_trail_stop_pct * 100.0, 2),
                        "risk_reward_ratio": round(1.6 / 0.9, 3),
                        "entry_note": note,
                    },
                    "intraday_overlay": {
                        **fallback_overlay,
                        "market": "CRYPTO",
                        "best_bid": round(best_bid, 8),
                        "best_ask": round(best_ask, 8),
                        "spread_pct": round(spread_pct, 5),
                        "tick_age_seconds": round(tick_age, 2),
                        "depth_age_seconds": None,
                        "signal_age_seconds": round(tick_age, 2),
                    },
                    "signal_style": "crypto_depth_intraday",
                    "xgb_alignment": None,
                },
                lane_override="crypto",
            )

        latest_tick = PRICE_BUFFER.latest(symbol)
        if latest_tick is None or float(getattr(latest_tick, "price", 0.0) or 0.0) <= 0:
            return None

        tick_age_seconds = max(0.0, (datetime.now(timezone.utc) - latest_tick.timestamp).total_seconds())
        if tick_age_seconds > self._crypto_signal_stale_seconds:
            return None

        depth_snapshot = DEPTH_BUFFER.filtered_snapshot(
            symbol,
            min_lifetime_ms=self._crypto_obi_min_lifetime_ms,
            max_updates_per_second=self._crypto_obi_max_updates_per_second,
            min_inter_update_delay_ms=self._crypto_obi_min_inter_update_delay_ms,
        ) or {}
        overlay = self._compute_intraday_overlay(symbol, latest_row=None)
        if not overlay:
            best_bid = float(depth_snapshot.get("best_bid", 0.0) or getattr(latest_tick, "bid", 0.0) or 0.0)
            best_ask = float(depth_snapshot.get("best_ask", 0.0) or getattr(latest_tick, "ask", 0.0) or 0.0)
            bid_qty = float(depth_snapshot.get("best_bid_qty", 0.0) or 0.0)
            ask_qty = float(depth_snapshot.get("best_ask_qty", 0.0) or 0.0)
            if best_bid <= 0 or best_ask <= 0:
                return _build_quote_only_fallback(depth_snapshot, "Quote-only fallback")
            book_imbalance = (bid_qty - ask_qty) / max(bid_qty + ask_qty, 1e-6)
            spread_pct = ((best_ask - best_bid) / ((best_bid + best_ask) / 2.0)) * 100.0
            confidence = self._clamp(
                0.22 + min(abs(book_imbalance) * 1.85, 0.42) + max(0.0, (0.08 - spread_pct)) * 2.0,
                0.18,
                0.84,
            )
            direction = "buy" if book_imbalance >= 0.04 else "sell" if book_imbalance <= -0.04 else "neutral"
            overlay = {
                "direction": direction,
                "confidence": round(confidence, 4),
                "setup": "depth_imbalance_breakout" if direction != "neutral" else "quote_balance_watch",
                "book_pressure": round(book_imbalance, 4),
                "book_imbalance": round(book_imbalance, 4),
                "top_book_imbalance": round(book_imbalance, 4),
                "trend_score": round(book_imbalance * 0.75, 4),
                "volume_spike": 1.0,
                "score": round(book_imbalance * 1.9, 4),
                "news_boost": 0.0,
                "order_flow_imbalance": round(book_imbalance * 0.85, 4),
                "spread_velocity": round(self._safe_number(depth_snapshot.get("spread_velocity"), 0.0), 8),
                "microprice_bias": 0.0,
                "best_bid_qty": round(bid_qty, 6),
                "best_ask_qty": round(ask_qty, 6),
                "bid_depth_qty": round(bid_qty, 6),
                "ask_depth_qty": round(ask_qty, 6),
                "flicker_filter_retention": round(self._safe_number(depth_snapshot.get("flicker_filter_retention"), 1.0), 4),
                "vwap_bias": 0.0,
                "ret_5m": 0.0,
                "trade_window_active": True,
                "risk_adjustments": {
                    "stop_loss_pct": 0.7,
                    "take_profit_pct": 1.5,
                    "trailing_stop_pct": round(self._day_trade_trail_stop_pct * 100.0, 2),
                },
                "note": "Depth snapshot fallback",
            }

        best_bid = float(depth_snapshot.get("best_bid", 0.0) or overlay.get("best_bid", 0.0) or 0.0)
        best_ask = float(depth_snapshot.get("best_ask", 0.0) or overlay.get("best_ask", 0.0) or 0.0)
        if (best_bid <= 0 or best_ask <= 0 or best_ask < best_bid) and self._crypto_quote_fallback_enabled:
            fallback_signal = _build_quote_only_fallback(depth_snapshot, "Quote-only fallback")
            if fallback_signal is not None:
                return fallback_signal
        depth_updated_at = depth_snapshot.get("updated_at")
        depth_age_seconds = None
        if depth_updated_at:
            try:
                depth_age_seconds = max(
                    0.0,
                    (datetime.now(timezone.utc) - datetime.fromisoformat(str(depth_updated_at))).total_seconds(),
                )
            except Exception:
                depth_age_seconds = None

        direction = str(overlay.get("direction", "neutral") or "neutral")
        setup = str(overlay.get("setup", "hybrid_intraday") or "hybrid_intraday")
        book_pressure = float(overlay.get("book_pressure", 0.0) or 0.0)
        book_imbalance = float(overlay.get("book_imbalance", 0.0) or 0.0)
        trend_score = float(overlay.get("trend_score", 0.0) or 0.0)
        volume_spike = float(overlay.get("volume_spike", 1.0) or 1.0)
        score_value = float(overlay.get("score", 0.0) or 0.0)
        spread_velocity = self._safe_number(overlay.get("spread_velocity"), self._safe_number(depth_snapshot.get("spread_velocity"), 0.0))

        trend_factor = self._clamp((trend_score * 0.75) + (book_pressure * 0.20), -1.0, 1.0)
        momentum_factor = self._clamp(score_value / 1.8, -1.0, 1.0)
        mean_revert_factor = 0.0
        if setup == "liquidity_sweep_reversal":
            mean_revert_factor = 0.70 if direction == "buy" else (-0.70 if direction == "sell" else 0.0)
        volume_factor = self._clamp(((volume_spike - 1.0) / 1.75) + (abs(book_imbalance) * 0.20), -1.0, 1.0)
        sentiment_factor = self._clamp(float(overlay.get("news_boost", 0.0) or 0.0) * 0.35, -1.0, 1.0)
        orderflow_factor = self._clamp(
            (float(overlay.get("order_flow_imbalance", 0.0) or 0.0) * 1.25) + (book_pressure * 0.35),
            -1.0,
            1.0,
        )

        factor_scores = {
            "trend": round(trend_factor, 4),
            "momentum": round(momentum_factor, 4),
            "mean_revert": round(mean_revert_factor, 4),
            "volume": round(volume_factor, 4),
            "sentiment": round(sentiment_factor, 4),
            "earnings_propagation": 0.0,
            "close_reversal": round(orderflow_factor * 0.35, 4),
        }
        factor_weights = {
            "trend": 0.28,
            "momentum": 0.24,
            "mean_revert": 0.16,
            "volume": 0.14,
            "sentiment": 0.08,
            "earnings_propagation": 0.0,
            "close_reversal": 0.10,
        }

        stop_loss_pct = float(overlay.get("risk_adjustments", {}).get("stop_loss_pct", 0.9) or 0.9)
        take_profit_pct = float(overlay.get("risk_adjustments", {}).get("take_profit_pct", 1.8) or 1.8)
        trailing_stop_pct = float(
            overlay.get("risk_adjustments", {}).get("trailing_stop_pct", self._day_trade_trail_stop_pct * 100.0)
            or (self._day_trade_trail_stop_pct * 100.0)
        )
        risk_reward_ratio = round(take_profit_pct / max(stop_loss_pct, 0.05), 3)

        confidence = self._clamp(float(overlay.get("confidence", 0.0) or 0.0), 0.0, 0.99)
        conviction_score = round(
            min(10.0, max(confidence * 10.0, abs(score_value) * 4.2, abs(book_pressure) * 5.0)),
            1,
        )
        model_agreement = round(
            self._clamp(
                (abs(trend_factor) * 0.35)
                + (abs(orderflow_factor) * 0.35)
                + (min(abs(score_value), 2.0) / 2.0 * 0.30),
                0.0,
                1.0,
            ),
            4,
        )

        regime = "normal"
        regime_multiplier = 0.95
        if volume_spike >= 2.6 or abs(float(overlay.get("ret_5m", 0.0) or 0.0)) >= 0.015:
            regime = "stressed"
            regime_multiplier = 0.70
        elif volume_spike <= 0.85 and abs(float(overlay.get("ret_5m", 0.0) or 0.0)) <= 0.0025:
            regime = "calm"
            regime_multiplier = 1.05

        driver_direction = "bullish" if direction == "buy" else ("bearish" if direction == "sell" else "mixed")
        top_drivers = [
            {"feature": "Order Book Pressure", "impact": round(book_pressure, 4), "direction": driver_direction},
            {"feature": "Session VWAP Bias", "impact": round(float(overlay.get("vwap_bias", 0.0) or 0.0), 4), "direction": driver_direction},
            {"feature": "Relative Volume", "impact": round(volume_factor, 4), "direction": "bullish" if volume_factor >= 0 else "bearish"},
        ]
        if setup == "liquidity_sweep_reversal":
            top_drivers.append({"feature": "Liquidity Sweep", "impact": round(mean_revert_factor, 4), "direction": driver_direction})

        signal = {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal": direction,
            "confidence": round(confidence, 4),
            "conviction_score": conviction_score,
            "ensemble_score": round(score_value, 4),
            "regime": regime,
            "regime_multiplier": regime_multiplier,
            "factor_scores": factor_scores,
            "factor_weights": factor_weights,
            "model_agreement": model_agreement,
            "model_breakdown": factor_scores,
            "top_drivers": top_drivers,
            "waterfall_data": [],
            "macro_warnings": [],
            "risk_parameters": {
                "stop_loss_pct": round(stop_loss_pct, 2),
                "take_profit_pct": round(take_profit_pct, 2),
                "trailing_stop_pct": round(trailing_stop_pct, 2),
                "risk_reward_ratio": risk_reward_ratio,
                "entry_note": str(overlay.get("note", "") or ""),
            },
            "intraday_overlay": {
                **overlay,
                "market": "CRYPTO",
                "spread_velocity": round(spread_velocity, 8),
                "flicker_filter_retention": round(self._safe_number(depth_snapshot.get("flicker_filter_retention"), 1.0), 4),
                "best_bid": round(best_bid, 8) if best_bid > 0 else None,
                "best_ask": round(best_ask, 8) if best_ask > 0 else None,
                "best_bid_qty": round(float(overlay.get("best_bid_qty", depth_snapshot.get("best_bid_qty", 0.0)) or 0.0), 6),
                "best_ask_qty": round(float(overlay.get("best_ask_qty", depth_snapshot.get("best_ask_qty", 0.0)) or 0.0), 6),
                "bid_depth_qty": round(float(overlay.get("bid_depth_qty", depth_snapshot.get("bid_depth_qty", 0.0)) or 0.0), 6),
                "ask_depth_qty": round(float(overlay.get("ask_depth_qty", depth_snapshot.get("ask_depth_qty", 0.0)) or 0.0), 6),
                "spread_pct": round((((best_ask - best_bid) / ((best_bid + best_ask) / 2.0)) * 100.0), 5)
                if best_bid > 0 and best_ask > 0
                else None,
                "tick_age_seconds": round(tick_age_seconds, 2),
                "depth_age_seconds": round(depth_age_seconds, 2) if depth_age_seconds is not None else None,
                "signal_age_seconds": round(max(tick_age_seconds, depth_age_seconds or 0.0), 2),
            },
            "signal_style": "crypto_depth_intraday",
            "xgb_alignment": None,
        }
        return self._decorate_signal(symbol, signal, lane_override="crypto")

    def _refresh_crypto_signals(self, latest_feature_rows: Optional[Dict[str, pd.Series]] = None) -> int:
        if not (self._crypto_depth_enabled and self._crypto_symbols):
            return 0

        broker = self._components.get("broker")
        held_symbols = {
            str(getattr(position, "symbol", "") or "")
            for position in getattr(broker, "positions", {}).values()
            if getattr(position, "quantity", 0) > 0
        } if broker else set()
        latest_feature_rows = latest_feature_rows or {}
        updated = 0
        now_ts = time.time()

        for symbol in self._crypto_symbols:
            try:
                signal = self._build_crypto_intraday_signal(symbol)
            except Exception as exc:
                logger.warning(f"Crypto signal build failed for {symbol}: {exc}")
                continue
            if signal is None:
                last_live_ts = self._crypto_last_signal_ts.get(symbol, 0.0)
                signal_age_seconds = now_ts - last_live_ts
                within_hold_grace = signal_age_seconds <= self._crypto_signal_hold_grace_seconds
                within_stale_retention = signal_age_seconds <= self._crypto_signal_stale_retention_seconds
                existing = self._signal_store.get(symbol)
                if within_stale_retention and isinstance(existing, dict):
                    if str(existing.get("lane", "")).lower() == "crypto":
                        meta = existing.get("meta_decision")
                        if isinstance(meta, dict):
                            if within_hold_grace:
                                meta.setdefault("reason", "crypto_refresh_gap_soft")
                            else:
                                meta["take_trade"] = False
                                meta["reason"] = "crypto_refresh_gap"
                        if not within_hold_grace:
                            existing["trade_eligible"] = False
                        existing["stale_reason"] = "crypto_refresh_gap_soft" if within_hold_grace else "crypto_refresh_gap"
                        existing["stale_seconds"] = round(max(signal_age_seconds, 0.0), 2)
                        continue
                if symbol not in held_symbols and symbol not in latest_feature_rows:
                    self._signal_store.pop(symbol, None)
                    self._crypto_last_signal_ts.pop(symbol, None)
                continue
            signal.pop("stale_reason", None)
            signal.pop("stale_seconds", None)
            self._signal_store[symbol] = signal
            self._crypto_last_signal_ts[symbol] = now_ts
            updated += 1

        return updated

    def _apply_live_meta_decisions(self, symbols: List[str], latest_feature_rows: Optional[Dict[str, pd.Series]] = None) -> int:
        if not symbols:
            return 0
        signal_subset = {
            symbol: self._signal_store.get(symbol)
            for symbol in symbols
            if isinstance(self._signal_store.get(symbol), dict)
        }
        if not signal_subset:
            return 0
        feature_rows = {}
        if isinstance(latest_feature_rows, dict):
            feature_rows = {
                symbol: latest_feature_rows[symbol]
                for symbol in signal_subset
                if symbol in latest_feature_rows
            }
        try:
            meta_decisions = self._get_meta_engine().evaluate_universe(signal_subset, feature_rows=feature_rows)
        except Exception as exc:
            logger.debug(f"Live meta evaluation failed: {exc}")
            return 0
        applied = 0
        for symbol, meta in meta_decisions.items():
            signal = self._signal_store.get(symbol)
            if not isinstance(signal, dict):
                continue
            signal["warmup_only"] = False
            signal["meta_decision"] = meta
            signal["trade_eligible"] = meta.get("take_trade", False)
            signal["take_probability"] = meta.get("take_probability", 0.0)
            signal["skip_probability"] = meta.get("skip_probability", 1.0)
            signal["expected_edge_pct"] = meta.get("expected_edge_pct", 0.0)
            signal["expected_drawdown_pct"] = meta.get("expected_drawdown_pct", 0.0)
            signal["rank_score"] = meta.get("rank_score", 0.0)
            signal["rank_percentile"] = meta.get("rank_percentile", 0.0)
            signal["size_multiplier"] = meta.get("size_multiplier", 0.0)
            signal["meta_source"] = meta.get("source", "heuristic")
            signal.pop("stale_reason", None)
            signal.pop("stale_seconds", None)
            applied += 1
        return applied

    def _crypto_signal_loop(self):
        while self._running:
            try:
                latest_feature_rows = self._components.get("latest_feature_rows")
                if not isinstance(latest_feature_rows, dict):
                    latest_feature_rows = {}
                updated = self._refresh_crypto_signals(latest_feature_rows)
                crypto_symbols = [
                    symbol
                    for symbol, signal in self._signal_store.items()
                    if isinstance(signal, dict) and str(signal.get("lane") or "").lower() == "crypto"
                ]
                applied = self._apply_live_meta_decisions(crypto_symbols, latest_feature_rows)
                if updated or applied:
                    logger.info(
                        "Crypto signals: %s updated | %s meta refreshed | total=%s",
                        updated,
                        applied,
                        len(crypto_symbols),
                    )
            except Exception as exc:
                logger.error(f"Crypto signal loop error: {exc}")
            time.sleep(self._crypto_signal_refresh_seconds)

    def _get_portfolio_correlation_snapshot(self, symbol: str, broker) -> Dict[str, float]:
        import pandas as pd

        base = self._get_recent_return_series(symbol, self._auto_trade_corr_lookback_days)
        if base is None:
            return {"avg_abs_corr": 0.0, "max_abs_corr": 0.0, "sample_size": 0}

        correlations = []
        seen_symbols = set()
        for _, position, held_symbol, _, _ in self._iter_open_positions(broker):
            if held_symbol == symbol or held_symbol in seen_symbols:
                continue
            seen_symbols.add(held_symbol)
            peer = self._get_recent_return_series(held_symbol, self._auto_trade_corr_lookback_days)
            if peer is None:
                continue
            aligned = pd.concat([base, peer], axis=1, join="inner").dropna()
            if len(aligned) < 15:
                continue
            corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
            if pd.notna(corr):
                correlations.append(abs(float(corr)))

        if not correlations:
            return {"avg_abs_corr": 0.0, "max_abs_corr": 0.0, "sample_size": 0}

        return {
            "avg_abs_corr": round(sum(correlations) / len(correlations), 4),
            "max_abs_corr": round(max(correlations), 4),
            "sample_size": len(correlations),
        }

    def _get_signal_score(self, signal: dict) -> float:
        intraday = signal.get("intraday_overlay", {}) if isinstance(signal, dict) else {}
        intraday_bonus = 0.0
        if isinstance(intraday, dict) and intraday:
            intraday_bonus = float(intraday.get("score", 0.0)) * 3.0 + float(intraday.get("confidence", 0.0))
        construction = signal.get("portfolio_construction", {}) if isinstance(signal, dict) else {}
        if isinstance(construction, dict) and construction:
            return round(
                float(construction.get("portfolio_score", 0.0))
                + float(construction.get("target_position_pct", 0.0)) * 100.0,
                4,
            ) + intraday_bonus
        meta = signal.get("meta_decision", {}) if isinstance(signal, dict) else {}
        if isinstance(meta, dict) and meta:
            return round(
                float(meta.get("rank_score", 0.0)) * 10.0
                + float(meta.get("take_probability", 0.0)) * 2.0,
                4,
            ) + intraday_bonus
        conviction = float(signal.get("conviction_score", 0.0))
        regime_multiplier = float(signal.get("regime_multiplier", 1.0))
        model_agreement = float(signal.get("model_agreement", 0.0))
        factor_scores = signal.get("factor_scores", {}) or {}
        event_edge = abs(float(factor_scores.get("earnings_propagation", 0.0))) + abs(
            float(factor_scores.get("close_reversal", 0.0))
        )
        xgb_alignment = signal.get("xgb_alignment")
        alignment_bonus = 0.4 if xgb_alignment == "confirmed" else 0.0
        return round(
            (conviction * regime_multiplier) + (event_edge * 2.5) + model_agreement + alignment_bonus + intraday_bonus,
            4,
        )

    def _record_execution_event(
        self,
        symbol: str,
        action: str,
        status: str,
        reason: str,
        signal: Optional[Dict] = None,
        position_key: Optional[str] = None,
        score: Optional[float] = None,
        conviction: Optional[float] = None,
        sector: Optional[str] = None,
        details: Optional[Dict] = None,
    ):
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "action": action,
            "status": status,
            "reason": reason,
        }
        resolved_signal = signal if isinstance(signal, dict) else {}
        if not resolved_signal:
            resolved_signal = self._signal_store.get(symbol, {}) if isinstance(self._signal_store.get(symbol), dict) else {}
        if not resolved_signal and position_key and position_key in self._position_plans:
            signal = {
                "lane": self._position_plans[position_key].get("lane"),
                "setup_id": self._position_plans[position_key].get("setup_id"),
                "market": self._position_plans[position_key].get("market"),
                "signal_style": self._position_plans[position_key].get("signal_style"),
                "intraday_overlay": {
                    "setup": self._position_plans[position_key].get("setup_id"),
                    "regime": self._position_plans[position_key].get("entry_regime"),
                },
            }
            resolved_signal = signal
        lane_meta = self._signal_lane_meta(symbol, resolved_signal)
        event.update(
            {
                "lane": lane_meta.get("lane"),
                "lane_label": lane_meta.get("lane_label"),
                "market": lane_meta.get("market"),
                "setup_id": lane_meta.get("setup_id"),
                "time_bucket": lane_meta.get("time_bucket"),
                "position_key": position_key or self._position_key(symbol, lane_meta.get("lane")),
            }
        )
        if score is not None:
            event["score"] = round(float(score), 4)
        if conviction is not None:
            event["conviction"] = round(float(conviction), 2)
        if sector:
            event["sector"] = sector
        if isinstance(details, dict):
            for key, value in details.items():
                if value in (None, "", [], {}):
                    continue
                event[key] = value
        self._execution_trace.append(event)
        if len(self._execution_trace) > self._execution_trace_limit:
            del self._execution_trace[: len(self._execution_trace) - self._execution_trace_limit]
        self._append_jsonl(self._execution_trace_log_path, event, log_label="Execution trace")
        self._persist_runtime_state()

    def _broker_result_details(self, result: Optional[Dict]) -> Dict:
        if not isinstance(result, dict):
            return {}
        payload: Dict[str, object] = {
            "broker_status": result.get("status"),
        }
        trade = result.get("trade") if isinstance(result.get("trade"), dict) else {}
        if trade:
            payload.update(
                {
                    "filled_quantity": trade.get("quantity"),
                    "requested_quantity": trade.get("requested_quantity", trade.get("quantity")),
                    "fill_price": trade.get("fill_price"),
                    "slippage_pct": trade.get("slippage_pct"),
                    "fill_ratio": trade.get("fill_ratio"),
                    "partial_fill": trade.get("partial_fill"),
                    "simulated_latency_ms": trade.get("simulated_latency_ms"),
                    "broker_market": trade.get("market"),
                }
            )
        shadow = result.get("shadow") if isinstance(result.get("shadow"), dict) else {}
        if shadow:
            payload["shadow_status"] = shadow.get("status")
            payload["shadow_reason"] = shadow.get("reason")
            payload["shadow_broker"] = shadow.get("broker")
        return {k: v for k, v in payload.items() if v not in (None, "", [], {})}

    def _capture_execution_reconciliation(
        self,
        symbol: str,
        action: str,
        signal: Optional[Dict],
        result: Optional[Dict],
        *,
        position_key: Optional[str] = None,
    ) -> None:
        details = self._broker_result_details(result)
        if not details:
            return
        resolved_signal = signal if isinstance(signal, dict) else {}
        lane_meta = self._signal_lane_meta(symbol, resolved_signal)
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "action": action,
            "position_key": position_key or self._position_key(symbol, lane_meta.get("lane")),
            "lane": lane_meta.get("lane"),
            **details,
        }
        self._execution_reconciliation.append(row)
        if len(self._execution_reconciliation) > self._execution_trace_limit:
            del self._execution_reconciliation[: len(self._execution_reconciliation) - self._execution_trace_limit]
        self._append_jsonl(self._execution_reconciliation_log_path, row, log_label="Execution reconciliation")
        self._persist_runtime_state()

    def _persist_feature_store(
        self,
        feature_matrices: Dict[str, object],
        force: bool = False,
        updated_symbols: Optional[List[str]] = None,
    ):
        if not self._feature_store_enabled:
            return
        now = time.time()
        if not force and (now - self._last_feature_store_save_ts) < self._feature_store_save_seconds:
            return

        self._feature_store_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        candidates = updated_symbols if updated_symbols else list(feature_matrices.keys())
        for symbol in candidates:
            frame = feature_matrices.get(symbol)
            if frame is None or getattr(frame, "empty", True):
                continue
            if not isinstance(frame, pd.DataFrame):
                continue
            try:
                path = self._feature_store_dir / f"{symbol}.parquet"
                frame.sort_index().to_parquet(path)
                saved += 1
            except Exception as exc:
                logger.warning(f"Feature store save failed for {symbol}: {exc}")

        if saved:
            self._last_feature_store_save_ts = now
            logger.info(f"Feature store updated: {saved} symbols saved to {self._feature_store_dir}")

    def _maybe_auto_retrain(self):
        if not self._auto_retrain_on_start:
            return

        retrain_script = Path(__file__).resolve().parent / "retrain_institutional_models.py"
        if not retrain_script.exists():
            logger.warning("Auto retrain skipped: retrain_institutional_models.py not found")
            return

        if self._auto_retrain_only_if_new_data:
            feature_files = sorted(self._feature_store_dir.glob("*.parquet")) if self._feature_store_dir.exists() else []
            if not feature_files:
                logger.info("Auto retrain skipped: no feature-store parquet files found yet")
                return

            checkpoint_dir = Path(__file__).resolve().parent / "models" / "checkpoints"
            checkpoint_refs = [
                checkpoint_dir / "xgboost_retrain_report.json",
                checkpoint_dir / "meta_walkforward_report.json",
                checkpoint_dir / "xgboost_model.json",
                checkpoint_dir / "meta_directional_models.joblib",
            ]
            existing_refs = [p for p in checkpoint_refs if p.exists()]
            if existing_refs:
                last_retrain_ts = max(p.stat().st_mtime for p in existing_refs)
                updated_files = [p for p in feature_files if p.stat().st_mtime > last_retrain_ts]
                if not updated_files:
                    logger.info(
                        "Auto retrain skipped: no feature-store files newer than last retrain"
                    )
                    return
                logger.info(
                    f"Auto retrain triggered: {len(updated_files)} feature files newer than last retrain"
                )
            else:
                logger.info("Auto retrain triggered: no prior retrain checkpoints found")

        logger.info("Auto retrain starting before live system boot")
        env = os.environ.copy()
        env["RUN_XGB_OPTUNA"] = "1" if self._auto_retrain_use_optuna else "0"
        try:
            subprocess.run(
                [sys.executable, str(retrain_script)],
                cwd=str(retrain_script.parent),
                env=env,
                check=True,
                timeout=self._auto_retrain_timeout_seconds,
            )
            logger.info("Auto retrain completed successfully")
        except subprocess.TimeoutExpired:
            logger.warning(
                f"Auto retrain timed out after {self._auto_retrain_timeout_seconds}s; continuing startup"
            )
        except subprocess.CalledProcessError as exc:
            logger.warning(f"Auto retrain failed with exit code {exc.returncode}; continuing startup")
        except Exception as exc:
            logger.warning(f"Auto retrain failed unexpectedly: {exc}; continuing startup")

    def _refresh_portfolio_overlay(self) -> Dict:
        target = self._components.get("portfolio_overlay")
        if not isinstance(target, dict):
            target = {}
            self._components["portfolio_overlay"] = target
        broker = self._components.get("broker")
        lane_allocator = self._refresh_lane_allocator(broker)
        governor = self._refresh_governor_state(broker)

        if not self._portfolio_optimizer_enabled:
            target.clear()
            for signal in self._signal_store.values():
                if isinstance(signal, dict):
                    signal["portfolio_construction"] = {}
                    lane = str(signal.get("lane") or "normal").lower()
                    signal["lane_allocator"] = lane_allocator.get(lane, {})
                    signal["governor_state"] = governor.get(lane, {})
            return {}

        price_recent = (
            self._components.get("live_price_data", {}).get("price_daily_recent")
            or self._components.get("price_data", {}).get("price_daily_recent", {})
        )
        if not broker or not price_recent:
            return {}

        from core.portfolio_construction import InstitutionalPortfolioConstructor

        constructor = InstitutionalPortfolioConstructor(
            lookback_days=self._auto_trade_corr_lookback_days,
            gross_target_pct=self._portfolio_optimizer_gross_target_pct,
            max_names=self._portfolio_optimizer_max_names,
            min_weight=self._portfolio_optimizer_min_weight,
            max_weight=min(self._portfolio_optimizer_max_weight, self._auto_trade_max_position_pct),
            covariance_shrinkage=self._portfolio_optimizer_covariance_shrinkage,
            factor_penalty=self._portfolio_optimizer_factor_penalty,
            beta_penalty=self._portfolio_optimizer_beta_penalty,
            target_beta=self._portfolio_optimizer_target_beta,
        )
        overlay = constructor.construct(self._signal_store, broker, price_recent)
        target.clear()
        if isinstance(overlay, dict):
            target.update(overlay)
        target["lane_allocator"] = lane_allocator
        target["governor"] = governor
        target["allocation_regime"] = self._detect_allocation_regime()

        allocations = target.get("allocations", {})
        for symbol, signal in self._signal_store.items():
            if not isinstance(signal, dict):
                continue
            allocation = allocations.get(symbol, {})
            signal["portfolio_construction"] = allocation
            signal["portfolio_overlay_summary"] = target.get("summary", {})
            signal["target_weight"] = allocation.get("target_weight", 0.0)
            signal["target_position_pct"] = allocation.get("target_position_pct", 0.0)
            signal["residual_alpha_score"] = allocation.get("residual_alpha_score", 0.0)
            signal["beta_exposure"] = allocation.get("beta", 0.0)
            signal["portfolio_score"] = allocation.get("portfolio_score", 0.0)
            lane = str(signal.get("lane") or "normal").lower()
            signal["lane_allocator"] = lane_allocator.get(lane, {})
            signal["governor_state"] = governor.get(lane, {})
            if isinstance(signal.get("normal_lane_signal"), dict):
                signal["normal_lane_signal"]["portfolio_construction"] = allocation
                signal["normal_lane_signal"]["portfolio_overlay_summary"] = target.get("summary", {})
                signal["normal_lane_signal"]["target_weight"] = allocation.get("target_weight", 0.0)
                signal["normal_lane_signal"]["target_position_pct"] = allocation.get("target_position_pct", 0.0)
                signal["normal_lane_signal"]["residual_alpha_score"] = allocation.get("residual_alpha_score", 0.0)
                signal["normal_lane_signal"]["beta_exposure"] = allocation.get("beta", 0.0)
                signal["normal_lane_signal"]["portfolio_score"] = allocation.get("portfolio_score", 0.0)
                signal["normal_lane_signal"]["lane_allocator"] = lane_allocator.get("normal", {})
                signal["normal_lane_signal"]["governor_state"] = governor.get("normal", {})
        return dict(target)

    def _refresh_meta_model_status(self) -> Dict:
        import json

        from models.institutional_retraining import META_REPORT_PATH, TrainedMetaModel

        status = {
            "active": False,
            "source": "heuristic_only",
        }

        if META_REPORT_PATH.exists():
            try:
                report = json.loads(META_REPORT_PATH.read_text(encoding="utf-8"))
                walk_forward = report.get("walk_forward", {}) if isinstance(report, dict) else {}
                summary = walk_forward.get("summary", {}) if isinstance(walk_forward, dict) else {}
                status.update(
                    {
                        "rows": report.get("rows"),
                        "positive_labels": report.get("positive_labels"),
                        "feature_count": report.get("feature_count"),
                        "walk_forward_status": walk_forward.get("status"),
                        "mean_precision": summary.get("mean_precision"),
                        "mean_recall": summary.get("mean_recall"),
                        "mean_coverage_pct": summary.get("mean_coverage_pct"),
                        "mean_taken_edge_pct": summary.get("mean_taken_edge_pct"),
                        "mean_taken_drawdown_pct": summary.get("mean_taken_drawdown_pct"),
                        "mean_taken_hit_rate_pct": summary.get("mean_taken_hit_rate_pct"),
                        "deployment_rules": report.get("deployment_rules", {}),
                    }
                )
            except Exception as exc:
                status["note"] = f"meta report read failed: {exc}"

        active_model = TrainedMetaModel.load()
        if active_model is not None:
            status["active"] = True
            status["source"] = "trained_meta"
            status["decision_threshold"] = active_model.decision_threshold
            status["min_expected_edge_pct"] = active_model.min_expected_edge_pct
            status["min_edge_ratio"] = active_model.min_edge_ratio

        target = self._components.get("meta_model_status")
        if not isinstance(target, dict):
            target = {}
            self._components["meta_model_status"] = target
        target.clear()
        target.update(status)
        return dict(target)

    def _refresh_learning_status(self) -> Dict:
        import json

        from models.institutional_retraining import META_DIRECTIONAL_PATH, META_REPORT_PATH, XGB_REPORT_PATH, XGB_SHAP_REPORT_PATH

        feature_files = sorted(self._feature_store_dir.glob("*.parquet")) if self._feature_store_dir.exists() else []
        status = {
            "enabled": self._model_learning_enabled,
            "learning_refresh_seconds": self._model_learning_refresh_seconds,
            "feature_store_files": len(feature_files),
            "feature_store_path": str(self._feature_store_dir),
            "retrain_in_progress": self._model_learning_in_progress,
            "last_check_at": datetime.fromtimestamp(self._last_model_learning_ts, tz=timezone.utc).isoformat()
            if self._last_model_learning_ts
            else None,
        }

        if XGB_REPORT_PATH.exists():
            try:
                xgb_report = json.loads(XGB_REPORT_PATH.read_text(encoding="utf-8"))
                status["xgb_last_retrain_at"] = datetime.fromtimestamp(XGB_REPORT_PATH.stat().st_mtime, tz=timezone.utc).isoformat()
                status["xgb_validation_accuracy"] = xgb_report.get("validation_accuracy")
                status["xgb_validation_rows"] = xgb_report.get("validation_rows")
                status["xgb_features"] = xgb_report.get("n_features")
                status["xgb_shap_monitor"] = xgb_report.get("shap_monitor")
            except Exception as exc:
                status["xgb_error"] = str(exc)

        if META_REPORT_PATH.exists():
            try:
                meta_report = json.loads(META_REPORT_PATH.read_text(encoding="utf-8"))
                summary = ((meta_report.get("walk_forward") or {}).get("summary") or {})
                status["meta_last_retrain_at"] = datetime.fromtimestamp(META_REPORT_PATH.stat().st_mtime, tz=timezone.utc).isoformat()
                status["meta_precision"] = summary.get("mean_precision")
                status["meta_coverage_pct"] = summary.get("mean_coverage_pct")
                status["meta_edge_pct"] = summary.get("mean_taken_edge_pct")
                status["meta_hit_rate_pct"] = summary.get("mean_taken_hit_rate_pct")
                status["meta_deployment_rules"] = meta_report.get("deployment_rules", {})
            except Exception as exc:
                status["meta_error"] = str(exc)
        status["meta_directional_ready"] = META_DIRECTIONAL_PATH.exists()
        if XGB_SHAP_REPORT_PATH.exists():
            try:
                status["xgb_shap_baseline"] = json.loads(XGB_SHAP_REPORT_PATH.read_text(encoding="utf-8"))
            except Exception as exc:
                status["xgb_shap_error"] = str(exc)
        drift_status = self._components.get("drift_status")
        if isinstance(drift_status, dict) and drift_status:
            status["drift_status"] = drift_status
        execution_overfit = self._components.get("execution_overfit")
        if isinstance(execution_overfit, dict) and execution_overfit:
            status["execution_overfit"] = execution_overfit
        decay_library = self._components.get("signal_decay_library")
        if isinstance(decay_library, dict) and decay_library:
            status["signal_decay_library"] = dict(list(decay_library.items())[:15])

        target = self._components.get("learning_status")
        if not isinstance(target, dict):
            target = {}
            self._components["learning_status"] = target
        target.clear()
        target.update(status)
        return dict(target)

    def _refresh_drift_status(self, broker) -> Dict:
        target = self._components.get("drift_status")
        if not isinstance(target, dict):
            target = {}
            self._components["drift_status"] = target
        if not broker:
            target.clear()
            return {}

        closed_rows = self._recent_closed_trade_rows(broker, lookback_days=self._drift_baseline_window_days)
        recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=self._drift_recent_window_hours)
        lanes = {}
        trigger_retrain = False

        for lane in self._lane_engine_order:
            lane_rows = [
                row for row in closed_rows
                if str((row.get("metadata") or {}).get("lane") or "normal").lower() == lane
            ]
            if not lane_rows:
                lanes[lane] = {
                    "baseline_log_loss": None,
                    "recent_log_loss": None,
                    "error_delta": 0.0,
                    "weight": 0.0,
                    "trigger_retrain": False,
                    "paused": False,
                    "recent_win_rate_pct": None,
                    "recent_trade_count": 0,
                }
                continue

            def _trade_log_loss(row: Dict) -> Optional[float]:
                metadata = row.get("metadata") or {}
                prob = self._safe_number(
                    metadata.get("entry_take_probability", metadata.get("take_probability")),
                    -1.0,
                )
                if prob < 0:
                    return None
                prob = float(np.clip(prob, 1e-6, 1.0 - 1e-6))
                label = 1.0 if self._safe_number(row.get("realized_pnl"), 0.0) > 0 else 0.0
                return float(-((label * math.log(prob)) + ((1.0 - label) * math.log(1.0 - prob))))

            baseline_losses = [loss for loss in (_trade_log_loss(row) for row in lane_rows) if loss is not None]
            recent_rows = [row for row in lane_rows if row.get("filled_at_dt") and row["filled_at_dt"] >= recent_cutoff]
            recent_losses = [loss for loss in (_trade_log_loss(row) for row in recent_rows) if loss is not None]
            baseline_error = float(np.mean(baseline_losses)) if baseline_losses else None
            recent_error = float(np.mean(recent_losses)) if recent_losses else None
            error_delta = 0.0
            weight = 0.0
            lane_trigger = False
            if (
                baseline_error is not None
                and baseline_error > 0
                and recent_error is not None
                and len(recent_losses) >= 3
                and recent_error > (baseline_error * self._drift_error_trigger_mult)
            ):
                error_delta = recent_error - baseline_error
                weight = min(1.0, max(0.0, error_delta / baseline_error))
                lane_trigger = True

            ordered_rows = sorted(
                lane_rows,
                key=lambda row: row.get("filled_at_dt") or datetime.min.replace(tzinfo=timezone.utc),
            )
            trailing_window = ordered_rows[-self._lane_pause_trade_window :]
            wins = sum(1 for row in trailing_window if self._safe_number(row.get("realized_pnl"), 0.0) > 0)
            recent_win_rate = wins / max(len(trailing_window), 1) if trailing_window else None
            paused = bool(
                trailing_window
                and len(trailing_window) >= self._lane_pause_trade_window
                and recent_win_rate is not None
                and recent_win_rate < self._lane_pause_win_rate_floor
            )
            trigger_retrain = trigger_retrain or lane_trigger or paused
            lanes[lane] = {
                "baseline_log_loss": round(baseline_error, 5) if baseline_error is not None else None,
                "recent_log_loss": round(recent_error, 5) if recent_error is not None else None,
                "error_delta": round(error_delta, 5),
                "weight": round(weight, 4),
                "trigger_retrain": lane_trigger,
                "paused": paused,
                "recent_win_rate_pct": round((recent_win_rate or 0.0) * 100.0, 1) if recent_win_rate is not None else None,
                "recent_trade_count": len(trailing_window),
            }

        target.clear()
        target.update(
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "recent_window_hours": self._drift_recent_window_hours,
                "baseline_window_days": self._drift_baseline_window_days,
                "error_trigger_multiple": self._drift_error_trigger_mult,
                "trigger_retrain": trigger_retrain,
                "lanes": lanes,
            }
        )
        return dict(target)

    def _maybe_periodic_retrain(self, feature_matrices: Dict[str, object]) -> None:
        if not self._model_learning_enabled or self._model_learning_in_progress:
            return
        now = time.time()
        if not feature_matrices:
            return
        broker = self._components.get("broker")
        drift_status = self._refresh_drift_status(broker) if broker else {}
        urgent_retrain = bool((drift_status or {}).get("trigger_retrain"))
        if urgent_retrain and self._last_emergency_retrain_ts:
            if (now - self._last_emergency_retrain_ts) < self._emergency_retrain_cooldown_seconds:
                return
        if not urgent_retrain and self._last_model_learning_ts and (now - self._last_model_learning_ts) < self._model_learning_refresh_seconds:
            return

        self._last_model_learning_ts = now
        if urgent_retrain:
            self._last_emergency_retrain_ts = now
        self._model_learning_in_progress = True
        self._refresh_learning_status()

        def _run_retrain(local_feature_matrices: Dict[str, object]):
            try:
                trigger_reason = "drift/emergency" if urgent_retrain else "scheduled"
                logger.info(f"Periodic model-learning retrain started ({trigger_reason})")
                report = self.retrain_institutional_models(
                    feature_matrices=local_feature_matrices,
                    run_optuna=self._auto_retrain_use_optuna,
                )
                logger.info("Periodic model-learning retrain completed")
                target = self._components.get("learning_status")
                if not isinstance(target, dict):
                    target = {}
                    self._components["learning_status"] = target
                target["last_runtime_retrain_at"] = datetime.now(timezone.utc).isoformat()
                target["last_runtime_report"] = report
                target["last_runtime_trigger"] = "drift/emergency" if urgent_retrain else "scheduled"
                self._runtime_refresh_models = True
                self._refresh_meta_model_status()
            except Exception as exc:
                logger.warning(f"Periodic model-learning retrain failed: {exc}")
                target = self._components.get("learning_status")
                if not isinstance(target, dict):
                    target = {}
                    self._components["learning_status"] = target
                target["last_runtime_error"] = str(exc)
            finally:
                self._model_learning_in_progress = False
                self._refresh_learning_status()

        threading.Thread(
            target=_run_retrain,
            args=(feature_matrices,),
            daemon=True,
            name="model-learning-retrain",
        ).start()

    def _row_backtest_score(self, row) -> float:
        import math

        def _f(name: str, default: float = 0.0) -> float:
            val = row.get(name, default)
            try:
                fv = float(val)
            except Exception:
                return default
            if math.isnan(fv):
                return default
            return fv

        earnings = _f("earnings_propagation_signal")
        reversal = _f("close_reversal_signal")
        momentum = _f("momentum_composite")
        trend = _f("trend_composite")
        sentiment = _f("compound_score")
        vol_penalty = abs(_f("vol_regime"))
        score = (
            (2.2 * earnings)
            + (1.8 * reversal)
            + (0.9 * momentum)
            + (0.7 * trend)
            + (0.4 * sentiment)
            - (0.4 * vol_penalty)
        )
        return float(score)

    def _simulate_overlay_backtest(
        self,
        feature_matrices: Dict[str, object],
        lookback_days: int,
        constrained: bool = True,
        entry_score_override: Optional[float] = None,
    ) -> Dict:
        import pandas as pd

        from pipeline.universe import get_sector, is_leader_symbol

        trade_cost_pct = self._execution_backtest_tx_cost_bps / 10_000.0
        entry_score = self._execution_backtest_entry_score if entry_score_override is None else float(entry_score_override)
        exit_score = self._execution_backtest_exit_score
        hold_days = self._execution_backtest_min_hold_days if constrained else 1
        max_positions = self._auto_trade_max_open_positions if constrained else max(5, self._auto_trade_max_open_positions * 2)
        max_sector_positions = self._auto_trade_max_sector_positions if constrained else max(8, self._auto_trade_max_sector_positions * 4)

        symbol_frames = {}
        latest_end = None
        for symbol, frame in feature_matrices.items():
            if frame is None or getattr(frame, "empty", True):
                continue
            if "close" not in frame.columns:
                continue
            ordered = frame.sort_index()
            if ordered.empty:
                continue
            if latest_end is None or ordered.index[-1] > latest_end:
                latest_end = ordered.index[-1]
            symbol_frames[symbol] = ordered

        if not symbol_frames or latest_end is None:
            return {"error": "insufficient_feature_history"}

        if len(symbol_frames) > self._execution_backtest_symbol_limit:
            ranked_symbols = sorted(
                symbol_frames.keys(),
                key=lambda sym: len(symbol_frames[sym]),
                reverse=True,
            )[: self._execution_backtest_symbol_limit]
            symbol_frames = {sym: symbol_frames[sym] for sym in ranked_symbols}

        start_cutoff = latest_end - pd.Timedelta(days=lookback_days)
        date_set = set()
        for frame in symbol_frames.values():
            date_set.update([idx for idx in frame.index if idx >= start_cutoff])
        dates = sorted(date_set)
        if len(dates) < 15:
            return {"error": "not_enough_days"}

        positions: Dict[str, int] = {}
        returns = []
        trade_returns: List[float] = []
        turnover = 0

        for i in range(1, len(dates) - 1):
            day = dates[i]
            next_day = dates[i + 1]
            candidates: List[Tuple[float, str]] = []
            day_returns = []
            sector_counts: Dict[str, int] = {}
            exits = 0
            entries = 0

            for symbol in list(positions.keys()):
                frame = symbol_frames.get(symbol)
                if frame is None or day not in frame.index or next_day not in frame.index:
                    continue
                row = frame.loc[day]
                score = self._row_backtest_score(row)
                positions[symbol] += 1
                if positions[symbol] >= hold_days and score < exit_score:
                    del positions[symbol]
                    exits += 1
                    continue
                sector = get_sector(symbol)
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
                c0 = float(frame.loc[day]["close"])
                c1 = float(frame.loc[next_day]["close"])
                if c0 > 0:
                    realized_ret = (c1 / c0) - 1.0
                    day_returns.append(realized_ret)
                    trade_returns.append(realized_ret)

            for symbol, frame in symbol_frames.items():
                if symbol in positions:
                    continue
                if day not in frame.index:
                    continue
                row = frame.loc[day]
                score = self._row_backtest_score(row)
                if score < entry_score:
                    continue
                candidates.append((score, symbol))

            candidates.sort(key=lambda t: t[0], reverse=True)
            for score, symbol in candidates:
                if len(positions) >= max_positions:
                    break
                sector = get_sector(symbol)
                if sector_counts.get(sector, 0) >= max_sector_positions:
                    continue
                positions[symbol] = 0
                entries += 1
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

            turnover += entries + exits
            gross = sum(day_returns) / len(day_returns) if day_returns else 0.0
            tx_penalty = trade_cost_pct * max(0, entries + exits)
            returns.append({"date": next_day, "ret": gross - tx_penalty})

        if not returns:
            return {"error": "no_returns"}

        rets = pd.Series(
            [item["ret"] for item in returns],
            index=pd.DatetimeIndex([item["date"] for item in returns]),
            dtype=float,
        )
        equity = (1.0 + rets).cumprod() * 100_000.0
        from core.backtesting import PerformanceMetrics

        metrics = PerformanceMetrics.full_report(rets, equity)
        return {
            "metrics": metrics,
            "days": len(rets),
            "avg_daily_turnover": round(turnover / max(len(rets), 1), 2),
            "constrained": constrained,
            "trade_returns": [round(float(value), 6) for value in trade_returns],
        }

    def _summarize_alpha_slice(self, frame: pd.DataFrame) -> Dict[str, Any]:
        if frame is None or frame.empty:
            return {
                "count": 0,
                "avg_return_pct": 0.0,
                "median_return_pct": 0.0,
                "hit_rate_pct": 0.0,
                "avg_take_probability_pct": 0.0,
                "avg_rank_score": 0.0,
                "avg_expected_edge_pct": 0.0,
            }

        returns = frame["forward_return"].astype(float)
        return {
            "count": int(len(frame)),
            "avg_return_pct": round(float(returns.mean() * 100.0), 3),
            "median_return_pct": round(float(returns.median() * 100.0), 3),
            "hit_rate_pct": round(float(frame["win"].mean() * 100.0), 2),
            "avg_take_probability_pct": round(float(frame["take_probability"].mean() * 100.0), 2),
            "avg_rank_score": round(float(frame["rank_score"].mean()), 4),
            "avg_expected_edge_pct": round(float(frame["expected_edge_pct"].mean()), 3),
        }

    def _alpha_bucket_summary(self, frame: pd.DataFrame, column: str) -> Dict[str, Any]:
        if frame is None or frame.empty or column not in frame.columns:
            return {"buckets": [], "monotonic_avg_return": False}

        working = frame[[column, "forward_return", "win", "take_probability"]].dropna().copy()
        if len(working) < max(self._alpha_quality_bucket_count * 2, self._alpha_quality_min_bucket_samples):
            return {"buckets": [], "monotonic_avg_return": False}

        ranked = working[column].rank(method="first", pct=True)
        labels = [f"Q{i}" for i in range(1, self._alpha_quality_bucket_count + 1)]
        working["bucket"] = pd.cut(
            ranked,
            bins=np.linspace(0.0, 1.0, self._alpha_quality_bucket_count + 1),
            labels=labels,
            include_lowest=True,
        )

        buckets = []
        for label in labels:
            bucket = working[working["bucket"] == label]
            if bucket.empty:
                continue
            buckets.append(
                {
                    "bucket": label,
                    "count": int(len(bucket)),
                    "min_value": round(float(bucket[column].min()), 4),
                    "max_value": round(float(bucket[column].max()), 4),
                    "avg_return_pct": round(float(bucket["forward_return"].mean() * 100.0), 3),
                    "median_return_pct": round(float(bucket["forward_return"].median() * 100.0), 3),
                    "hit_rate_pct": round(float(bucket["win"].mean() * 100.0), 2),
                    "avg_take_probability_pct": round(float(bucket["take_probability"].mean() * 100.0), 2),
                }
            )

        avg_returns = [bucket["avg_return_pct"] for bucket in buckets]
        monotonic = len(avg_returns) >= 2 and all(
            avg_returns[idx] <= avg_returns[idx + 1] + 1e-9 for idx in range(len(avg_returns) - 1)
        )
        return {"buckets": buckets, "monotonic_avg_return": monotonic}

    def _build_alpha_quality_dataset(
        self,
        feature_matrices: Dict[str, object],
        *,
        lookback_days: int,
        horizon_days: int,
    ) -> pd.DataFrame:
        from core.signal_engine_v2 import MetaDecisionEngine, MultiFactorScorer
        from pipeline.universe import is_day_trade_symbol

        scorer = MultiFactorScorer()
        meta_engine = MetaDecisionEngine()
        latest_end = None
        symbol_frames: Dict[str, pd.DataFrame] = {}
        for symbol, frame in feature_matrices.items():
            if frame is None or getattr(frame, "empty", True) or "close" not in frame.columns:
                continue
            ordered = frame.sort_index()
            if ordered.empty or len(ordered) <= horizon_days:
                continue
            symbol_frames[symbol] = ordered
            if latest_end is None or ordered.index[-1] > latest_end:
                latest_end = ordered.index[-1]

        if not symbol_frames or latest_end is None:
            return pd.DataFrame()

        start_cutoff = latest_end - pd.Timedelta(days=lookback_days)
        trade_cost_pct = self._execution_backtest_tx_cost_bps / 10_000.0
        factor_names = list(MultiFactorScorer.FACTOR_WEIGHTS.keys())
        rows: List[Dict[str, Any]] = []

        for symbol, ordered in symbol_frames.items():
            subset = ordered[ordered.index >= start_cutoff].copy()
            if subset.empty or len(subset) <= horizon_days:
                continue
            forward_returns = (subset["close"].shift(-horizon_days) / subset["close"]) - 1.0
            subset = subset.iloc[:-horizon_days]
            if subset.empty:
                continue

            market = self._market_code_for_symbol(symbol)
            lane = "crypto" if market == "CRYPTO" else "day" if self._day_trading_mode and is_day_trade_symbol(symbol) else "normal"
            lane_config = self._lane_config(lane)
            risk_defaults = {
                "stop_loss_pct": 2.0,
                "take_profit_pct": 5.0,
                "risk_reward_ratio": 2.0,
            }
            if lane == "crypto":
                risk_defaults["stop_loss_pct"] = 1.5
                risk_defaults["take_profit_pct"] = 3.0
            elif lane == "day":
                risk_defaults["stop_loss_pct"] = 1.2
                risk_defaults["take_profit_pct"] = 2.4

            for idx, feature_row in subset.iterrows():
                future_return = forward_returns.loc[idx]
                if pd.isna(future_return):
                    continue
                try:
                    scored = scorer.score(feature_row, symbol=symbol)
                except Exception:
                    continue
                if str(scored.get("direction", "neutral")).lower() != "buy":
                    continue

                signal_payload = {
                    **scored,
                    "lane": lane,
                    "market": market,
                    "model_agreement": self._safe_number(feature_row.get("model_agreement"), 0.0),
                    "risk_parameters": {
                        **risk_defaults,
                        "risk_reward_ratio": round(
                            risk_defaults["take_profit_pct"] / max(risk_defaults["stop_loss_pct"], 0.1),
                            3,
                        ),
                    },
                }
                decision = meta_engine._evaluate_one(symbol, signal_payload, feature_row)
                record = {
                    "date": pd.Timestamp(idx).isoformat(),
                    "symbol": symbol,
                    "lane": lane,
                    "market": market,
                    "leader_symbol": bool(scored.get("leader_symbol", False)),
                    "regime": str(scored.get("regime", "normal") or "normal"),
                    "final_score": float(scored.get("final_score", 0.0) or 0.0),
                    "confidence": float(scored.get("confidence", 0.0) or 0.0),
                    "conviction_score": float(scored.get("conviction_score", 0.0) or 0.0),
                    "take_trade": bool(decision.get("take_trade", False)),
                    "take_probability": float(decision.get("take_probability", 0.0) or 0.0),
                    "rank_score": float(decision.get("rank_score", 0.0) or 0.0),
                    "expected_edge_pct": float(decision.get("expected_edge_pct", 0.0) or 0.0),
                    "expected_drawdown_pct": float(decision.get("expected_drawdown_pct", 0.0) or 0.0),
                    "meta_reason": str(decision.get("reason", "") or ""),
                    "meta_source": str(decision.get("source", "") or ""),
                    "forward_return": float(future_return),
                    "win": bool(float(future_return) > trade_cost_pct),
                    "min_conviction_threshold": float(
                        lane_config.get(
                            "min_conviction",
                            self._auto_trade_min_conviction,
                        )
                    ),
                }
                for factor_name in factor_names:
                    record[f"factor_{factor_name}"] = float((scored.get("factor_scores", {}) or {}).get(factor_name, 0.0) or 0.0)
                rows.append(record)

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def _persist_alpha_quality_report(self, payload: Dict[str, Any]) -> None:
        try:
            self._alpha_quality_report_path.parent.mkdir(parents=True, exist_ok=True)
            self._alpha_quality_report_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        except Exception as exc:
            logger.warning(f"Alpha quality report save failed: {exc}")

    def _compute_alpha_quality_report(self, feature_matrices: Dict[str, object]) -> Dict[str, Any]:
        if not self._alpha_quality_enabled:
            return {"status": "disabled", "timestamp": datetime.now(timezone.utc).isoformat()}

        dataset = self._build_alpha_quality_dataset(
            feature_matrices,
            lookback_days=self._alpha_quality_lookback_days,
            horizon_days=self._alpha_quality_horizon_days,
        )
        if dataset.empty:
            return {
                "status": "insufficient_data",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "lookback_days": self._alpha_quality_lookback_days,
                "horizon_days": self._alpha_quality_horizon_days,
            }

        take_rows = dataset[dataset["take_trade"]].copy()
        skip_rows = dataset[~dataset["take_trade"]].copy()
        probability_buckets = self._alpha_bucket_summary(dataset, "take_probability")
        rank_buckets = self._alpha_bucket_summary(dataset, "rank_score")

        calibration_gap_pct = None
        bucket_rows = probability_buckets.get("buckets", [])
        if bucket_rows:
            total_weight = sum(int(bucket.get("count", 0) or 0) for bucket in bucket_rows)
            if total_weight > 0:
                weighted_gap = sum(
                    abs(float(bucket.get("avg_take_probability_pct", 0.0)) - float(bucket.get("hit_rate_pct", 0.0)))
                    * int(bucket.get("count", 0) or 0)
                    for bucket in bucket_rows
                )
                calibration_gap_pct = round(weighted_gap / total_weight, 3)

        threshold_candidates = sorted(
            {
                round(value, 2)
                for value in np.linspace(0.30, 0.80, max(self._alpha_quality_bucket_count + 1, 6))
            }
        )
        threshold_rows = []
        for threshold in threshold_candidates:
            subset = dataset[dataset["take_probability"] >= threshold]
            if len(subset) < self._alpha_quality_min_bucket_samples:
                continue
            threshold_rows.append(
                {
                    "take_probability_threshold": threshold,
                    **self._summarize_alpha_slice(subset),
                }
            )
        recommended_threshold = None
        if threshold_rows:
            recommended_threshold = max(
                threshold_rows,
                key=lambda item: (
                    float(item.get("avg_return_pct", 0.0)),
                    float(item.get("hit_rate_pct", 0.0)),
                    int(item.get("count", 0)),
                ),
            )

        factor_correlations = []
        for factor_name in (
            "trend",
            "momentum",
            "mean_revert",
            "volume",
            "sentiment",
            "earnings_propagation",
            "close_reversal",
        ):
            factor_col = f"factor_{factor_name}"
            if factor_col not in dataset.columns or dataset[factor_col].nunique() < 3:
                continue
            try:
                correlation = dataset[factor_col].corr(dataset["forward_return"], method="spearman")
            except Exception:
                correlation = None
            if correlation is None or pd.isna(correlation):
                continue
            factor_correlations.append({"factor": factor_name, "spearman_corr": round(float(correlation), 4)})
        factor_correlations.sort(key=lambda item: abs(float(item["spearman_corr"])), reverse=True)

        regime_summary = []
        for regime, group in dataset.groupby("regime"):
            regime_summary.append({"regime": regime, **self._summarize_alpha_slice(group)})
        regime_summary.sort(key=lambda item: item["regime"])

        reason_summary = []
        for reason, group in dataset.groupby("meta_reason"):
            reason_summary.append({"reason": reason, **self._summarize_alpha_slice(group)})
        reason_summary.sort(key=lambda item: item["count"], reverse=True)

        payload = {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "lookback_days": self._alpha_quality_lookback_days,
            "horizon_days": self._alpha_quality_horizon_days,
            "coverage": {
                "buy_signal_rows": int(len(dataset)),
                "take_rows": int(len(take_rows)),
                "skip_rows": int(len(skip_rows)),
                "symbols": int(dataset["symbol"].nunique()),
                "markets": sorted(dataset["market"].dropna().astype(str).unique().tolist()),
                "lanes": sorted(dataset["lane"].dropna().astype(str).unique().tolist()),
            },
            "summary": {
                "all_buy_signals": self._summarize_alpha_slice(dataset),
                "taken_signals": self._summarize_alpha_slice(take_rows),
                "skipped_signals": self._summarize_alpha_slice(skip_rows),
                "uplift_avg_return_pct": round(
                    float(self._summarize_alpha_slice(take_rows)["avg_return_pct"] - self._summarize_alpha_slice(skip_rows)["avg_return_pct"]),
                    3,
                ),
                "uplift_hit_rate_pct": round(
                    float(self._summarize_alpha_slice(take_rows)["hit_rate_pct"] - self._summarize_alpha_slice(skip_rows)["hit_rate_pct"]),
                    2,
                ),
                "calibration_gap_pct": calibration_gap_pct,
            },
            "take_probability_buckets": probability_buckets,
            "rank_score_buckets": rank_buckets,
            "threshold_sweep": threshold_rows,
            "recommended_take_probability_threshold": recommended_threshold,
            "factor_correlations": factor_correlations[:6],
            "regime_summary": regime_summary,
            "reason_summary": reason_summary[:8],
            "flags": {
                "take_probability_monotonic": bool(probability_buckets.get("monotonic_avg_return")),
                "rank_score_monotonic": bool(rank_buckets.get("monotonic_avg_return")),
                "calibration_warning": calibration_gap_pct is not None and calibration_gap_pct > 12.0,
                "negative_taken_expectancy": float(self._summarize_alpha_slice(take_rows)["avg_return_pct"]) <= 0.0,
            },
        }
        self._persist_alpha_quality_report(payload)
        return payload

    def _maybe_refresh_execution_backtest(self, feature_matrices: Dict[str, object]):
        if not self._execution_backtest_enabled:
            return
        now = time.time()
        if self._execution_backtest_running:
            return
        if (now - self._last_execution_backtest_ts) < self._execution_backtest_interval_seconds:
            return

        self._execution_backtest_running = True
        self._last_execution_backtest_ts = now

        def _refresh():
            try:
                constrained = self._simulate_overlay_backtest(
                    feature_matrices=feature_matrices,
                    lookback_days=self._execution_backtest_lookback_days,
                    constrained=True,
                )
                baseline = self._simulate_overlay_backtest(
                    feature_matrices=feature_matrices,
                    lookback_days=self._execution_backtest_lookback_days,
                    constrained=False,
                )
                constrained_m = constrained.get("metrics", {}) if isinstance(constrained, dict) else {}
                baseline_m = baseline.get("metrics", {}) if isinstance(baseline, dict) else {}
                uplift = {
                    "total_return_pct": round(
                        float(constrained_m.get("total_return_pct", 0.0))
                        - float(baseline_m.get("total_return_pct", 0.0)),
                        2,
                    ),
                    "sharpe_ratio": round(
                        float(constrained_m.get("sharpe_ratio", 0.0))
                        - float(baseline_m.get("sharpe_ratio", 0.0)),
                        3,
                    ),
                    "max_drawdown_pct": round(
                        float(constrained_m.get("max_drawdown_pct", 0.0))
                        - float(baseline_m.get("max_drawdown_pct", 0.0)),
                        2,
                    ),
                }
                trade_returns = np.asarray(constrained.get("trade_returns", []) or [], dtype=float)
                monte_carlo = {"status": "insufficient_data"}
                if trade_returns.size >= 12:
                    rng = np.random.default_rng(7)
                    mc_sharpes = []
                    annualizer = np.sqrt(252.0)
                    for _ in range(self._execution_backtest_monte_carlo_runs):
                        sample = rng.choice(trade_returns, size=trade_returns.size, replace=True)
                        sample_std = float(np.std(sample))
                        sharpe = 0.0 if sample_std <= 1e-9 else float(np.mean(sample) / sample_std * annualizer)
                        mc_sharpes.append(sharpe)
                    monte_carlo = {
                        "status": "ok",
                        "runs": self._execution_backtest_monte_carlo_runs,
                        "sharpe_p10": round(float(np.percentile(mc_sharpes, 10)), 4),
                        "sharpe_mean": round(float(np.mean(mc_sharpes)), 4),
                        "sharpe_std": round(float(np.std(mc_sharpes)), 4),
                    }
                    monte_carlo["overfit_flag"] = bool(
                        monte_carlo["sharpe_p10"] < 0
                        or (abs(monte_carlo["sharpe_mean"]) > 1e-9 and monte_carlo["sharpe_std"] > (abs(monte_carlo["sharpe_mean"]) * 2.0))
                    )

                plateau_results = []
                base_sharpe = float(constrained_m.get("sharpe_ratio", 0.0) or 0.0)
                for delta in (-self._execution_backtest_plateau_delta, self._execution_backtest_plateau_delta):
                    adjusted = self._simulate_overlay_backtest(
                        feature_matrices=feature_matrices,
                        lookback_days=self._execution_backtest_lookback_days,
                        constrained=True,
                        entry_score_override=self._execution_backtest_entry_score + delta,
                    )
                    adjusted_metrics = adjusted.get("metrics", {}) if isinstance(adjusted, dict) else {}
                    adjusted_sharpe = float(adjusted_metrics.get("sharpe_ratio", 0.0) or 0.0)
                    degradation = 0.0
                    if abs(base_sharpe) > 1e-9:
                        degradation = abs(adjusted_sharpe - base_sharpe) / abs(base_sharpe)
                    plateau_results.append(
                        {
                            "delta": round(float(delta), 3),
                            "sharpe_ratio": round(adjusted_sharpe, 4),
                            "degradation_pct": round(degradation * 100.0, 2),
                            "robust": degradation <= 0.20,
                        }
                    )
                plateau_flag = any(not item["robust"] for item in plateau_results)
                overfit_payload = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "monte_carlo": monte_carlo,
                    "parameter_plateau": plateau_results,
                    "flagged": bool(monte_carlo.get("overfit_flag")) or plateau_flag,
                }
                payload = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "lookback_days": self._execution_backtest_lookback_days,
                    "overlay": constrained,
                    "baseline": baseline,
                    "uplift": uplift,
                    "overfit": overfit_payload,
                }
                alpha_quality_payload = self._compute_alpha_quality_report(feature_matrices)
                target = self._components.get("execution_backtest")
                if isinstance(target, dict):
                    target.clear()
                    target.update(payload)
                else:
                    self._components["execution_backtest"] = payload
                alpha_target = self._components.get("alpha_quality")
                if isinstance(alpha_target, dict):
                    alpha_target.clear()
                    alpha_target.update(alpha_quality_payload)
                else:
                    self._components["alpha_quality"] = alpha_quality_payload
                overfit_target = self._components.get("execution_overfit")
                if isinstance(overfit_target, dict):
                    overfit_target.clear()
                    overfit_target.update(overfit_payload)
                else:
                    self._components["execution_overfit"] = overfit_payload
                logger.info(
                    "Execution backtest refreshed: overlay return "
                    f"{constrained_m.get('total_return_pct', 'n/a')}% vs baseline "
                    f"{baseline_m.get('total_return_pct', 'n/a')}%"
                )
                if isinstance(alpha_quality_payload, dict) and alpha_quality_payload.get("status") == "ok":
                    alpha_summary = alpha_quality_payload.get("summary", {}) or {}
                    logger.info(
                        "Alpha quality refreshed: taken avg return %s%% | calibration gap %s%%",
                        alpha_summary.get("taken_signals", {}).get("avg_return_pct", "n/a"),
                        alpha_summary.get("calibration_gap_pct", "n/a"),
                    )
            except Exception as exc:
                logger.warning(f"Execution backtest refresh failed: {exc}")
            finally:
                self._execution_backtest_running = False

        threading.Thread(target=_refresh, daemon=True, name="execution-backtest-refresh").start()

    def _get_position_age_seconds(self, position) -> float:
        opened_at = getattr(position, "opened_at", None)
        if opened_at is None:
            return float(self._auto_trade_min_hold_seconds)
        return max(0.0, (datetime.now(timezone.utc) - opened_at).total_seconds())

    def _day_fill_quality_adjustment(self, signal: dict, broker) -> Dict[str, float]:
        if not broker:
            return {"multiplier": 1.0, "fill_quality_score": 0.0}
        symbol = str(signal.get("symbol", "") or "")
        latest_row = self._latest_feature_row(symbol)
        atr_pct = self._safe_number((latest_row or {}).get("atr_pct"), 0.01)
        atr_pct_points = max(atr_pct * 100.0, 0.10)
        scores: List[float] = []
        for trade in reversed(list(getattr(broker, "trade_log", []) or [])):
            metadata = trade.get("metadata") or {}
            if str(metadata.get("lane") or "").lower() != "day":
                continue
            reference_price = self._safe_number(metadata.get("execution_reference_price"), 0.0)
            fill_price = self._safe_number(trade.get("fill_price"), reference_price)
            if reference_price <= 0 or fill_price <= 0:
                continue
            slip_pct = abs(fill_price - reference_price) / reference_price * 100.0
            scores.append(slip_pct / atr_pct_points)
            if len(scores) >= 5:
                break
        if not scores:
            return {"multiplier": 1.0, "fill_quality_score": 0.0}
        fill_quality_score = float(np.mean(scores))
        if fill_quality_score > 0.50:
            desired_base_pct = 0.008
            multiplier = min(1.0, desired_base_pct / max(self._lane_config("day").get("base_position_pct", 0.012), 1e-6))
        elif fill_quality_score < 0.10:
            multiplier = 1.6
        else:
            multiplier = 1.0
        return {
            "multiplier": round(float(multiplier), 4),
            "fill_quality_score": round(float(fill_quality_score), 4),
        }

    def _record_signal_halflife_observation(
        self,
        plan: Dict,
        *,
        age_seconds: float,
        current_take_probability: float,
        exit_reason: str,
    ) -> None:
        target = self._components.get("signal_decay_library")
        if not isinstance(target, dict):
            target = {}
            self._components["signal_decay_library"] = target
        sector = str(plan.get("sector") or "unknown")
        bucket = str(plan.get("entry_time_bucket") or "unknown")
        key = f"{sector}::{bucket}"
        payload = target.setdefault(
            key,
            {
                "sector": sector,
                "time_bucket": bucket,
                "observations": 0,
                "avg_exit_age_seconds": 0.0,
                "avg_entry_take_probability": 0.0,
                "avg_exit_take_probability": 0.0,
                "last_exit_reason": None,
            },
        )
        count = int(payload.get("observations", 0))
        entry_take_probability = self._safe_number(plan.get("entry_take_probability"), 0.0)
        payload["observations"] = count + 1
        payload["avg_exit_age_seconds"] = round(
            ((payload["avg_exit_age_seconds"] * count) + max(age_seconds, 0.0)) / max(count + 1, 1),
            2,
        )
        payload["avg_entry_take_probability"] = round(
            ((payload["avg_entry_take_probability"] * count) + entry_take_probability) / max(count + 1, 1),
            4,
        )
        payload["avg_exit_take_probability"] = round(
            ((payload["avg_exit_take_probability"] * count) + max(current_take_probability, 0.0)) / max(count + 1, 1),
            4,
        )
        payload["last_exit_reason"] = exit_reason

    def _compute_position_size(
        self,
        signal: dict,
        current_price: float,
        broker,
        corr_snapshot: Optional[Dict[str, float]] = None,
    ) -> float:
        lane = str(signal.get("lane") or "normal").lower()
        lane_config = self._lane_config(lane)
        risk = signal.get("risk_parameters", {}) or {}
        meta = signal.get("meta_decision", {}) if isinstance(signal, dict) else {}
        construction = signal.get("portfolio_construction", {}) if isinstance(signal, dict) else {}
        lane_allocator = signal.get("lane_allocator", {}) if isinstance(signal.get("lane_allocator"), dict) else {}
        governor_size_multiplier = self._safe_number(signal.get("governor_size_multiplier"), 1.0)
        execution_size_multiplier = self._safe_number(signal.get("execution_size_multiplier"), 1.0)
        suggested_pct = float(risk.get("suggested_position_size_pct", 0.0)) / 100.0
        stop_loss_pct = float(risk.get("stop_loss_pct", 2.0)) / 100.0
        portfolio_target_pct = float(construction.get("target_position_pct", 0.0) or 0.0)
        if portfolio_target_pct > 0:
            target_pct = portfolio_target_pct
        else:
            target_pct = max(lane_config.get("base_position_pct", self._auto_trade_base_position_pct), suggested_pct)
        target_pct = min(target_pct, lane_config.get("max_position_pct", self._auto_trade_max_position_pct))
        if lane == "day":
            fill_quality = self._day_fill_quality_adjustment(signal, broker)
            signal["fill_quality_score"] = fill_quality.get("fill_quality_score", 0.0)
            target_pct *= self._safe_number(fill_quality.get("multiplier"), 1.0)
        if portfolio_target_pct <= 0 and isinstance(meta, dict) and meta:
            target_pct *= float(meta.get("size_multiplier", 1.0) or 1.0)
        target_pct *= self._safe_number(lane_allocator.get("size_multiplier"), 1.0)
        target_pct *= governor_size_multiplier
        target_pct *= execution_size_multiplier
        if corr_snapshot and corr_snapshot.get("sample_size", 0) > 0:
            avg_corr = float(corr_snapshot.get("avg_abs_corr", 0.0))
            if avg_corr > self._auto_trade_target_avg_correlation:
                corr_span = max(0.01, 1.0 - self._auto_trade_target_avg_correlation)
                excess = min(1.0, (avg_corr - self._auto_trade_target_avg_correlation) / corr_span)
                corr_scale = max(self._auto_trade_correlation_size_floor, 1.0 - excess)
                target_pct *= corr_scale
        regime_value = signal.get("regime")
        stressed = False
        if isinstance(regime_value, dict):
            stressed = str(regime_value.get("volatility", "")).lower() in {"stressed", "crisis"}
        else:
            stressed = str(regime_value or "").lower() in {"stressed", "crisis", "risk-off"}
        if stressed:
            target_pct *= self._auto_trade_stress_position_mult

        max_position_value = broker.portfolio_value * target_pct
        intraday = signal.get("intraday_overlay", {}) if isinstance(signal.get("intraday_overlay"), dict) else {}
        trade_window_active = bool(intraday.get("trade_window_active", True))
        risk_per_trade_pct = lane_config.get("risk_per_trade_pct", self._day_trade_risk_per_trade_pct if self._day_trading_mode else 0.01)
        if lane in {"day", "crypto"} and not trade_window_active:
            risk_per_trade_pct *= 0.75
        risk_budget_value = broker.portfolio_value * risk_per_trade_pct
        stop_risk_value = current_price * max(stop_loss_pct, 0.01)
        risk_units = risk_budget_value / max(stop_risk_value, 0.01)
        size_from_target = max_position_value / current_price
        size_from_cash = max(broker.cash, 0.0) / current_price
        lane_target_pct = self._safe_number(lane_allocator.get("target_capital_pct"), 0.0)
        lane_current_pct = self._safe_number(lane_allocator.get("current_exposure_pct"), 0.0)
        lane_headroom_value = broker.portfolio_value * max(0.0, (lane_target_pct * 1.05) - lane_current_pct)
        size_from_lane_budget = (lane_headroom_value / current_price) if lane_headroom_value > 0 else 0.0
        sizing_inputs = [size_from_target, risk_units, size_from_cash]
        if lane_target_pct > 0:
            sizing_inputs.append(size_from_lane_budget)
        quantity = min(sizing_inputs)
        if lane == "crypto":
            if (quantity * current_price) < self._crypto_min_notional_usd:
                return 0.0
            if current_price >= 10_000:
                step = 0.0001
            elif current_price >= 1_000:
                step = 0.001
            elif current_price >= 100:
                step = 0.01
            elif current_price >= 1:
                step = 0.1
            else:
                step = 1.0
            quantity = math.floor(max(quantity, 0.0) / step) * step
            return round(max(quantity, 0.0), 6)
        return float(max(int(quantity), 0))

    def _can_enter_position(self, symbol: str, lane: Optional[str] = None) -> bool:
        lane_key = self._cooldown_key(symbol, lane)
        lane_config = self._lane_config(str(lane or "normal").lower())
        last_trade_ts = self._last_trade_timestamps.get(lane_key, 0.0)
        return (time.time() - last_trade_ts) >= lane_config.get("cooldown_seconds", self._auto_trade_cooldown_seconds)

    def _record_trade_timestamp(self, symbol: str, lane: Optional[str] = None):
        self._last_trade_timestamps[self._cooldown_key(symbol, lane)] = time.time()
        self._persist_runtime_state()

    def _select_replacement_candidate(self, broker, lane: Optional[str] = None) -> Optional[Tuple[str, float]]:
        weakest_position_key = None
        weakest_score = None
        target_lane = str(lane or "").lower()
        for position_key, position, symbol, symbol_lane, plan in self._iter_open_positions(broker):
            if target_lane and symbol_lane != target_lane:
                continue
            min_hold_seconds = int(plan.get("min_hold_seconds") or self._lane_config(symbol_lane).get("min_hold_seconds", self._auto_trade_min_hold_seconds))
            if self._get_position_age_seconds(position) < min_hold_seconds:
                continue
            signal = self._get_lane_signal(symbol, symbol_lane)
            score = self._get_signal_score(signal)
            if signal.get("signal") != "buy":
                score -= 2.0
            if weakest_score is None or score < weakest_score:
                weakest_position_key = position_key
                weakest_score = score
        if weakest_position_key is None or weakest_score is None:
            return None
        return weakest_position_key, weakest_score

    def _close_position(
        self,
        symbol: str,
        reason: str,
        signal_source: str = "EventMultiFactor",
        position_key: Optional[str] = None,
    ) -> bool:
        broker = self._components.get("broker")
        if not broker:
            return False

        from core.paper_trading import Order, OrderSide
        resolved_position_key = position_key
        if not resolved_position_key or resolved_position_key not in getattr(broker, "positions", {}):
            matches = [
                key
                for key, position, held_symbol, _, _ in self._iter_open_positions(broker)
                if held_symbol == symbol
            ]
            if len(matches) != 1:
                return False
            resolved_position_key = matches[0]

        existing = broker.positions.get(resolved_position_key)
        if not existing or existing.quantity <= 0:
            return False

        current_price = self._get_latest_close(symbol)
        if current_price is None:
            return False

        plan = self._position_plans.get(resolved_position_key, {})
        lane = str(plan.get("lane") or "normal").lower()
        signal = self._get_lane_signal(symbol, lane)
        age_seconds = self._get_position_age_seconds(existing)
        exit_signal = {
            **(signal if isinstance(signal, dict) else {}),
            "lane": plan.get("lane") or (signal.get("lane") if isinstance(signal, dict) else None),
            "setup_id": plan.get("setup_id") or (signal.get("setup_id") if isinstance(signal, dict) else None),
            "market": plan.get("market") or (signal.get("market") if isinstance(signal, dict) else None),
            "signal_style": plan.get("signal_style") or (signal.get("signal_style") if isinstance(signal, dict) else None),
            "intraday_overlay": {
                **((signal.get("intraday_overlay", {}) if isinstance(signal, dict) else {}) or {}),
                "setup": plan.get("setup_id") or ((signal.get("intraday_overlay", {}) if isinstance(signal, dict) else {}).get("setup")),
                "regime": plan.get("entry_regime") or ((signal.get("intraday_overlay", {}) if isinstance(signal, dict) else {}).get("regime")),
            },
        }
        lane_meta = self._signal_lane_meta(symbol, exit_signal, lane_override=plan.get("lane"))
        order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=existing.quantity,
            position_key=resolved_position_key,
            signal_source=signal_source,
            metadata={
                **self._build_order_metadata(symbol, exit_signal, action="sell", reason=reason),
                "exit_reason": reason,
                "entry_reason": plan.get("entry_reason"),
                "entry_regime": plan.get("entry_regime"),
                "entry_time_bucket": plan.get("entry_time_bucket"),
                "entry_take_probability": plan.get("entry_take_probability"),
                "entry_expected_edge_pct": plan.get("entry_expected_edge_pct"),
                "entry_expected_drawdown_pct": plan.get("entry_expected_drawdown_pct"),
                "scenario_label": lane_meta.get("lane_label"),
                "signal_key": resolved_position_key,
                "position_key": resolved_position_key,
            },
        )
        result = broker.submit_order(order, current_price)
        if result["status"] == "filled":
            if lane == "day":
                current_take_probability = self._safe_number(
                    exit_signal.get("take_probability", (exit_signal.get("meta_decision") or {}).get("take_probability")),
                    0.0,
                )
                self._record_signal_halflife_observation(
                    plan,
                    age_seconds=age_seconds,
                    current_take_probability=current_take_probability,
                    exit_reason=reason,
                )
            self._position_plans.pop(resolved_position_key, None)
            self._record_trade_timestamp(symbol, lane)
            self._capture_execution_reconciliation(symbol, "sell", exit_signal, result, position_key=resolved_position_key)
            self._record_execution_event(
                symbol,
                "sell",
                "filled",
                reason,
                signal=exit_signal,
                position_key=resolved_position_key,
                details=self._broker_result_details(result),
            )
            self._persist_runtime_state(force=True)
            logger.info(f"AUTO-SELL: {symbol} x{existing.quantity} @ ${current_price:.2f} | {reason}")
            return True
        self._capture_execution_reconciliation(symbol, "sell", exit_signal, result, position_key=resolved_position_key)
        self._record_execution_event(
            symbol,
            "sell",
            "rejected",
            result.get("reason", "broker_reject"),
            signal=exit_signal,
            position_key=resolved_position_key,
            details=self._broker_result_details(result),
        )
        return False

    def _manage_open_positions(self, broker, current_prices: Dict[str, float]) -> None:
        for position_key, position, symbol, lane, plan in list(self._iter_open_positions(broker)):
            current_price = current_prices.get(position_key) or current_prices.get(symbol)
            if current_price is None:
                continue

            signal = self._get_lane_signal(symbol, lane)
            if not isinstance(signal, dict):
                signal = {}
            meta = signal.get("meta_decision", {}) if isinstance(signal.get("meta_decision"), dict) else {}
            construction = signal.get("portfolio_construction", {}) if isinstance(signal.get("portfolio_construction"), dict) else {}
            intraday = signal.get("intraday_overlay", {}) if isinstance(signal.get("intraday_overlay"), dict) else {}
            age_seconds = self._get_position_age_seconds(position)
            lane_config = self._lane_config(lane)
            stop_loss_pct = float(plan.get("stop_loss_pct", 2.0)) / 100.0
            take_profit_pct = float(plan.get("take_profit_pct", stop_loss_pct * 2.5 * 100)) / 100.0
            stop_price = position.avg_cost * (1 - stop_loss_pct)
            take_profit_price = position.avg_cost * (1 + take_profit_pct)
            current_take_probability = self._safe_number(
                meta.get("take_probability", signal.get("take_probability", 0.0)),
                0.0,
            )

            if (
                lane != "day"
                and lane_config.get("force_exit_seconds", 0) > 0
                and age_seconds >= lane_config.get("force_exit_seconds", 0)
            ):
                self._close_position(symbol, "time exit", position_key=position_key)
                continue
            if current_price <= stop_price:
                self._close_position(symbol, "stop loss hit", position_key=position_key)
                continue
            if current_price >= take_profit_price:
                self._close_position(symbol, "take profit hit", position_key=position_key)
                continue

            if lane == "day":
                peak_price = max(float(plan.get("peak_price", position.avg_cost) or position.avg_cost), current_price)
                plan["peak_price"] = peak_price
                trail_stop_pct = float(plan.get("trailing_stop_pct", self._day_trade_trail_stop_pct * 100.0)) / 100.0
                trail_trigger = max(stop_loss_pct * 0.60, trail_stop_pct)
                if peak_price >= position.avg_cost * (1 + trail_trigger):
                    trailing_stop_price = peak_price * (1 - trail_stop_pct)
                    if current_price <= trailing_stop_price:
                        self._close_position(symbol, "adaptive trailing stop", position_key=position_key)
                        continue
                half_life_floor = max(
                    0.05,
                    self._safe_number(plan.get("entry_take_probability"), current_take_probability) * self._signal_halflife_exit_ratio,
                )
                day_min_halflife_hold = max(
                    120.0,
                    lane_config.get("min_hold_seconds", self._auto_trade_min_hold_seconds) * 0.50,
                )
                if age_seconds >= day_min_halflife_hold and current_take_probability <= half_life_floor:
                    self._close_position(symbol, "signal halflife decay", position_key=position_key)
                    continue
                if lane_config.get("force_exit_seconds", 0) > 0 and age_seconds >= lane_config.get("force_exit_seconds", 0):
                    self._close_position(symbol, "time exit", position_key=position_key)
                    continue
                if age_seconds >= max(120.0, lane_config.get("min_hold_seconds", self._auto_trade_min_hold_seconds) * 0.50):
                    if intraday.get("direction") == "sell" and float(intraday.get("score", 0.0) or 0.0) <= -0.10:
                        self._close_position(symbol, "intraday regime flip", position_key=position_key)
                        continue
                    if (
                        str(plan.get("entry_setup", "")) == "opening_range_momentum"
                        and float(intraday.get("order_flow_imbalance", 0.0) or 0.0) < -0.04
                        and current_price < float(intraday.get("session_vwap", current_price) or current_price)
                    ):
                        self._close_position(symbol, "opening range failure", position_key=position_key)
                        continue
                    if (
                        str(plan.get("entry_setup", "")) == "vwap_pullback_continuation"
                        and float(intraday.get("vwap_bias", 0.0) or 0.0) < -0.08
                        and intraday.get("direction") == "sell"
                    ):
                        self._close_position(symbol, "VWAP continuation failure", position_key=position_key)
                        continue
                    if (
                        str(plan.get("entry_setup", "")) == "liquidity_sweep_reversal"
                        and intraday.get("direction") == "sell"
                        and float(intraday.get("order_flow_imbalance", 0.0) or 0.0) < -0.02
                    ):
                        self._close_position(symbol, "sweep reversal failed", position_key=position_key)
                        continue

            if lane == "crypto":
                crypto_filter = self._crypto_execution_filter(signal)
                signal["execution_filter"] = crypto_filter
                if not crypto_filter.get("allow", True):
                    self._close_position(
                        symbol,
                        str(crypto_filter.get("reason", "crypto_execution_filter")),
                        signal_source="CryptoMicrostructure",
                        position_key=position_key,
                    )
                    continue

            if (
                age_seconds >= lane_config.get("min_hold_seconds", self._auto_trade_min_hold_seconds)
                and construction
                and float(construction.get("target_position_pct", 0.0) or 0.0)
                < (lane_config.get("base_position_pct", self._auto_trade_base_position_pct) * 0.25)
            ):
                self._close_position(symbol, "optimizer deallocated", position_key=position_key)
                continue

            conviction = float(signal.get("conviction_score", 0.0) or 0.0)
            direction = signal.get("signal")
            exit_floor = 4.6 if lane == "day" else 4.2 if lane == "crypto" else 5.8
            should_exit = direction == "sell" or conviction < exit_floor
            if lane == "day" and intraday.get("direction") == "sell":
                should_exit = True
            if isinstance(meta, dict) and meta:
                should_exit = should_exit or (
                    meta.get("take_trade") is False and float(meta.get("take_probability", 0.0) or 0.0) < 0.45
                )
            if age_seconds >= lane_config.get("min_hold_seconds", self._auto_trade_min_hold_seconds) and should_exit:
                self._close_position(symbol, "signal decay", position_key=position_key)

    def _build_lane_candidates(self) -> Dict[str, List[Tuple[float, str, Dict, float]]]:
        from pipeline.universe import is_leader_symbol

        candidates_by_lane: Dict[str, List[Tuple[float, str, Dict, float]]] = defaultdict(list)
        for lane, symbol, signal in self._iter_lane_signal_items():
            if not isinstance(signal, dict):
                continue

            lane = str(lane or signal.get("lane") or "normal").lower()
            lane_config = self._lane_config(lane)
            leader_symbol = lane in {"normal", "day"} and is_leader_symbol(symbol)
            meta = signal.get("meta_decision", {}) if isinstance(signal.get("meta_decision"), dict) else {}
            construction = signal.get("portfolio_construction", {}) if isinstance(signal.get("portfolio_construction"), dict) else {}
            intraday = signal.get("intraday_overlay", {}) if isinstance(signal.get("intraday_overlay"), dict) else {}
            conviction = float(signal.get("conviction_score", 0.0) or 0.0)
            take_probability = float(meta.get("take_probability", signal.get("take_probability", 0.0)) or 0.0)
            rank_score = float(meta.get("rank_score", signal.get("rank_score", 0.0)) or 0.0)
            meta_reason = str(meta.get("reason", "") or "")
            portfolio_target_pct = float(construction.get("target_position_pct", 0.0) or 0.0)
            lane_min_conviction = lane_config.get("min_conviction", self._auto_trade_min_conviction)
            if lane == "crypto":
                try:
                    signal_ts = datetime.fromisoformat(str(signal.get("timestamp") or datetime.now(timezone.utc).isoformat()))
                except Exception:
                    signal_ts = datetime.now(timezone.utc)
                signal_hour_utc = signal_ts.astimezone(timezone.utc).hour
                conviction_bias = 0.0
                if self._crypto_prime_start_hour_utc <= signal_hour_utc < self._crypto_prime_end_hour_utc:
                    conviction_bias += self._crypto_prime_conviction_bias
                if signal_hour_utc == 21:
                    conviction_bias += self._crypto_liquidity_hole_conviction_bias
                lane_min_conviction = max(0.0, lane_min_conviction + conviction_bias)
                signal["crypto_conviction_bias"] = round(conviction_bias, 4)
            min_conviction = min(lane_min_conviction, self._auto_trade_leader_min_conviction) if leader_symbol else lane_min_conviction
            leader_override = (
                leader_symbol
                and take_probability >= self._auto_trade_leader_min_take_probability
                and rank_score >= self._auto_trade_leader_min_rank_score
            )
            position_key = str(signal.get("signal_key") or self._position_key(symbol, lane))

            governor_decision = self._governor_decision(signal)
            signal["governor_decision"] = governor_decision
            signal["governor_size_multiplier"] = self._safe_number(governor_decision.get("size_multiplier"), 1.0)
            if not governor_decision.get("allow", True):
                self._record_execution_event(
                    symbol,
                    "buy",
                    "skipped",
                    str(governor_decision.get("reason", "governor_block")),
                    signal=signal,
                    position_key=position_key,
                    score=rank_score,
                    conviction=conviction,
                )
                continue

            execution_filter = {"allow": True, "reason": "active", "size_multiplier": 1.0}
            if lane == "crypto":
                execution_filter = self._crypto_execution_filter(signal)
                signal["execution_filter"] = execution_filter
                if not execution_filter.get("allow", True):
                    self._record_execution_event(
                        symbol,
                        "buy",
                        "skipped",
                        str(execution_filter.get("reason", "crypto_execution_filter")),
                        signal=signal,
                        position_key=position_key,
                        score=rank_score,
                        conviction=conviction,
                    )
                    continue
            signal["execution_size_multiplier"] = self._safe_number(execution_filter.get("size_multiplier"), 1.0)

            if signal.get("signal") != "buy":
                continue
            if conviction < min_conviction and not leader_override:
                self._record_execution_event(
                    symbol,
                    "buy",
                    "skipped",
                    "weak_conviction",
                    signal=signal,
                    position_key=position_key,
                    score=rank_score,
                    conviction=conviction,
                )
                continue
            if meta and take_probability < lane_config.get("min_take_probability", 0.0) and not leader_override:
                self._record_execution_event(
                    symbol,
                    "buy",
                    "skipped",
                    "low_take_probability",
                    signal=signal,
                    position_key=position_key,
                    score=rank_score,
                    conviction=conviction,
                )
                continue

            zero_weight_fallback = (
                construction
                and portfolio_target_pct <= 0
                and self._auto_trade_zero_weight_fallback_enabled
                and (bool(meta.get("take_trade", False)) or leader_override)
                and take_probability >= self._auto_trade_zero_weight_min_take_probability
                and rank_score >= self._auto_trade_zero_weight_min_rank_score
                and conviction >= (min_conviction * (0.8 if leader_symbol else 0.9))
            )
            qualified_edge_fallback = (
                construction
                and portfolio_target_pct <= 0
                and lane in {"normal", "day"}
                and meta_reason in {
                    "qualified_edge",
                    "qualified_edge_rescue",
                    "trained_qualified_edge",
                    "trained_qualified_edge_rescue",
                    "weak_conviction_rescue",
                }
                and take_probability >= max(0.10, lane_config.get("min_take_probability", 0.0) * 0.85)
                and rank_score >= max(0.04, self._auto_trade_zero_weight_min_rank_score * 0.75)
                and conviction >= max(0.5, min_conviction * 0.65)
            )
            intraday_daytrade_override = (
                lane == "day"
                and portfolio_target_pct <= 0
                and take_probability >= (0.28 if self._event_window_mode else 0.35)
                and conviction >= (min_conviction * 0.9)
                and float(intraday.get("confidence", 0.0) or 0.0) >= 0.42
                and intraday.get("direction") == "buy"
            )
            if construction and portfolio_target_pct <= 0 and not (
                zero_weight_fallback or qualified_edge_fallback or intraday_daytrade_override
            ):
                self._record_execution_event(
                    symbol,
                    "buy",
                    "skipped",
                    "optimizer_zero_weight",
                    signal=signal,
                    position_key=position_key,
                    score=float(construction.get("portfolio_score", 0.0)),
                    conviction=conviction,
                )
                continue
            if meta and not meta.get("take_trade", False):
                if (
                    lane == "crypto"
                    and self._event_window_mode
                    and take_probability >= (lane_config.get("min_take_probability", 0.0) * 0.85)
                    and conviction >= (min_conviction * 0.90)
                ):
                    signal["meta_override"] = "event_window_crypto"
                else:
                    self._record_execution_event(
                        symbol,
                        "buy",
                        "skipped",
                        str(meta.get("reason", "meta_skip")),
                        signal=signal,
                        position_key=position_key,
                        score=rank_score,
                        conviction=conviction,
                    )
                    continue
            if lane == "day":
                trade_window_active = bool(intraday.get("trade_window_active", True))
                exceptional_signal = (
                    float(intraday.get("confidence", 0.0) or 0.0) >= 0.78
                    or take_probability >= 0.68
                    or float(intraday.get("news_boost", 0.0) or 0.0) >= self._day_trade_news_boost_threshold
                )
                if not trade_window_active and not exceptional_signal:
                    self._record_execution_event(
                        symbol,
                        "buy",
                        "skipped",
                        "midday_chop_filter",
                        signal=signal,
                        position_key=position_key,
                        score=float(intraday.get("score", 0.0) or 0.0),
                        conviction=conviction,
                    )
                    continue

            entry_readiness = self._entry_readiness_gate(symbol, lane, signal)
            signal["entry_readiness"] = entry_readiness
            if not entry_readiness.get("allow", True):
                self._record_execution_event(
                    symbol,
                    "buy",
                    "skipped",
                    str(entry_readiness.get("reason", "entry_readiness_block")),
                    signal=signal,
                    position_key=position_key,
                    score=rank_score,
                    conviction=conviction,
                    details={
                        "readiness_reasons": entry_readiness.get("reasons"),
                        "live_tick_age_seconds": entry_readiness.get("live_tick_age_seconds"),
                        "live_source": entry_readiness.get("live_source"),
                    },
                )
                continue

            current_price = self._get_latest_close(symbol)
            if current_price is None:
                continue
            candidates_by_lane[lane].append((self._get_signal_score(signal), symbol, signal, current_price))
        return candidates_by_lane

    def _execute_lane_entries(self, broker, candidates_by_lane: Dict[str, List[Tuple[float, str, Dict, float]]]) -> None:
        from core.paper_trading import Order, OrderSide
        from pipeline.universe import get_sector

        lane_sector_counts: Dict[str, Dict[str, int]] = defaultdict(dict)
        lane_open_counts: Dict[str, int] = defaultdict(int)
        for _, _, symbol, lane, _ in self._iter_open_positions(broker):
            sector = get_sector(symbol)
            lane_sector_counts[lane][sector] = lane_sector_counts[lane].get(sector, 0) + 1
            lane_open_counts[lane] += 1

        for lane in self._lane_engine_order:
            lane_config = self._lane_config(lane)
            replacement_margin = self._auto_trade_replace_margin * (0.5 if lane in {"day", "crypto"} else 1.0)
            ranked_candidates = sorted(candidates_by_lane.get(lane, []), key=lambda item: item[0], reverse=True)
            ranked_candidates = ranked_candidates[: lane_config.get("top_k", self._auto_trade_top_k)]
            new_positions = 0
            sector_counts = lane_sector_counts[lane]

            for score, symbol, signal, current_price in ranked_candidates:
                position_key = str(signal.get("signal_key") or self._position_key(symbol, lane))
                if new_positions >= lane_config.get("max_new_per_cycle", self._auto_trade_max_new_per_cycle):
                    self._record_execution_event(symbol, "buy", "skipped", "cycle_entry_limit", signal=signal, position_key=position_key, score=score)
                    break
                if broker.has_open_position(position_key=position_key):
                    self._record_execution_event(symbol, "buy", "skipped", "already_in_portfolio", signal=signal, position_key=position_key, score=score)
                    continue
                if not self._can_enter_position(symbol, lane=lane):
                    self._record_execution_event(symbol, "buy", "skipped", "cooldown_active", signal=signal, position_key=position_key, score=score)
                    continue

                sector = get_sector(symbol)
                open_positions = lane_open_counts.get(lane, 0)
                corr_snapshot = self._get_portfolio_correlation_snapshot(symbol, broker)
                signal["portfolio_fit"] = corr_snapshot
                if sector_counts.get(sector, 0) >= lane_config.get("sector_cap", self._auto_trade_max_sector_positions):
                    self._record_execution_event(
                        symbol,
                        "buy",
                        "skipped",
                        "sector_cap_reached",
                        signal=signal,
                        position_key=position_key,
                        score=score,
                        sector=sector,
                    )
                    continue
                if (
                    corr_snapshot.get("sample_size", 0) > 0
                    and corr_snapshot.get("max_abs_corr", 0.0) >= lane_config.get("pair_corr_cap", self._auto_trade_max_pair_correlation)
                ):
                    self._record_execution_event(
                        symbol,
                        "buy",
                        "skipped",
                        "correlation_cap_reached",
                        signal=signal,
                        position_key=position_key,
                        score=score,
                        conviction=float(signal.get("conviction_score", 0.0)),
                        sector=sector,
                    )
                    continue

                if open_positions >= lane_config.get("max_open_positions", self._auto_trade_max_open_positions):
                    replacement = self._select_replacement_candidate(broker, lane=lane)
                    if not replacement:
                        self._record_execution_event(
                            symbol,
                            "buy",
                            "skipped",
                            "portfolio_full_no_replacement",
                            signal=signal,
                            position_key=position_key,
                            score=score,
                        )
                        break
                    weakest_position_key, weakest_score = replacement
                    if score < (weakest_score + replacement_margin):
                        self._record_execution_event(
                            symbol,
                            "buy",
                            "skipped",
                            "not_better_than_weakest",
                            signal=signal,
                            position_key=position_key,
                            score=score,
                        )
                        continue
                    weakest_position = broker.positions.get(weakest_position_key)
                    weakest_symbol = str(getattr(weakest_position, "symbol", weakest_position_key) or weakest_position_key)
                    if not self._close_position(
                        weakest_symbol,
                        f"rotating into stronger idea {symbol}",
                        position_key=weakest_position_key,
                    ):
                        self._record_execution_event(
                            symbol,
                            "buy",
                            "skipped",
                            "rotation_close_failed",
                            signal=signal,
                            position_key=position_key,
                            score=score,
                        )
                        continue
                    weakest_sector = get_sector(weakest_symbol)
                    sector_counts[weakest_sector] = max(0, sector_counts.get(weakest_sector, 1) - 1)
                    lane_open_counts[lane] = max(0, lane_open_counts.get(lane, 1) - 1)

                inventory_load = 0.0
                if lane == "crypto":
                    inventory_load = lane_open_counts.get(lane, 0) / max(lane_config.get("max_open_positions", 1), 1)
                    signal["inventory_skew"] = round(inventory_load, 4)
                    if inventory_load > self._crypto_inventory_skew_trigger:
                        signal["execution_size_multiplier"] = round(
                            self._safe_number(signal.get("execution_size_multiplier"), 1.0) * 0.85,
                            4,
                        )

                position_size = self._compute_position_size(signal, current_price, broker, corr_snapshot=corr_snapshot)
                min_position_size = 0.0001 if lane == "crypto" else 1.0
                if position_size < min_position_size:
                    self._record_execution_event(
                        symbol,
                        "buy",
                        "skipped",
                        "position_size_too_small",
                        signal=signal,
                        position_key=position_key,
                        score=score,
                    )
                    continue

                lane_meta = self._signal_lane_meta(symbol, signal, lane_override=lane)
                signal_source = {
                    "normal": "NormalTrading",
                    "day": "DayTrading",
                    "crypto": "CryptoMicrostructure",
                }.get(lane_meta.get("lane"), "EventMultiFactor")
                order_type = "market"
                limit_price = None
                if lane == "crypto":
                    intraday = signal.get("intraday_overlay", {}) if isinstance(signal.get("intraday_overlay"), dict) else {}
                    execution_filter = signal.get("execution_filter", {}) if isinstance(signal.get("execution_filter"), dict) else {}
                    execution_mode = str(execution_filter.get("execution_mode", "market") or "market").lower()
                    best_bid = self._safe_number(intraday.get("best_bid"), 0.0)
                    best_ask = self._safe_number(intraday.get("best_ask"), 0.0)
                    mid_price = (best_bid + best_ask) / 2.0 if best_bid > 0 and best_ask > 0 else 0.0
                    if execution_mode == "maker" and mid_price > 0 and best_bid > 0 and best_ask > 0:
                        tick_size = max((best_ask - best_bid) / 2.0, mid_price * 0.00005)
                        candidate_price = min(best_ask - tick_size, mid_price)
                        if inventory_load > self._crypto_inventory_skew_trigger:
                            candidate_price = max(best_bid, candidate_price - (tick_size * 0.50))
                        order_type = "limit"
                        limit_price = round(max(best_bid, candidate_price), 8)
                    signal["execution_mode"] = execution_mode
                order = Order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=position_size,
                    position_key=position_key,
                    order_type=order_type,
                    limit_price=limit_price,
                    signal_source=signal_source,
                    metadata={
                        **self._build_order_metadata(symbol, signal, action="buy", reason="ranked_entry"),
                        "entry_reason": "ranked_entry",
                        "scenario_label": lane_meta.get("lane_label"),
                        "signal_key": position_key,
                        "position_key": position_key,
                        "underlying_symbol": symbol,
                        "execution_mode": signal.get("execution_mode", "market"),
                        "inventory_skew": signal.get("inventory_skew", 0.0),
                    },
                )
                result = broker.submit_order(order, current_price)
                if result["status"] != "filled":
                    self._capture_execution_reconciliation(symbol, "buy", signal, result, position_key=position_key)
                    self._record_execution_event(
                        symbol,
                        "buy",
                        "rejected",
                        result.get("reason", "broker_reject"),
                        signal=signal,
                        position_key=position_key,
                        score=score,
                        details=self._broker_result_details(result),
                    )
                    continue

                risk = signal.get("risk_parameters", {}) or {}
                intraday = signal.get("intraday_overlay", {}) if isinstance(signal.get("intraday_overlay"), dict) else {}
                self._position_plans[position_key] = {
                    "symbol": symbol,
                    "signal_key": position_key,
                    "stop_loss_pct": float(risk.get("stop_loss_pct", 2.0)),
                    "take_profit_pct": float(risk.get("take_profit_pct", 5.0)),
                    "entry_score": score,
                    "sector": sector,
                    "entry_setup": str(intraday.get("setup", "") or ""),
                    "entry_regime": str(intraday.get("regime", "") or ""),
                    "setup_id": str(signal.get("setup_id") or intraday.get("setup", "") or "event_multifactor"),
                    "lane": str(signal.get("lane") or lane_meta.get("lane") or lane),
                    "market": str(signal.get("market") or lane_meta.get("market") or self._market_code_for_symbol(symbol)),
                    "signal_style": str(signal.get("signal_style") or ""),
                    "entry_reason": "ranked_entry",
                    "entry_time_bucket": str(signal.get("time_bucket") or lane_meta.get("time_bucket") or "unknown"),
                    "min_hold_seconds": int(lane_config.get("min_hold_seconds", self._auto_trade_min_hold_seconds)),
                    "trailing_stop_pct": float(risk.get("trailing_stop_pct", self._day_trade_trail_stop_pct * 100.0)),
                    "peak_price": float(current_price),
                    "entry_take_probability": float(signal.get("take_probability", 0.0) or 0.0),
                    "entry_expected_edge_pct": float(signal.get("expected_edge_pct", 0.0) or 0.0),
                    "entry_expected_drawdown_pct": float(signal.get("expected_drawdown_pct", 0.0) or 0.0),
                }
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
                lane_open_counts[lane] = lane_open_counts.get(lane, 0) + 1
                self._record_trade_timestamp(symbol, lane)
                self._capture_execution_reconciliation(symbol, "buy", signal, result, position_key=position_key)
                self._record_execution_event(
                    symbol,
                    "buy",
                    "filled",
                    "ranked_entry",
                    signal=signal,
                    position_key=position_key,
                    score=score,
                    conviction=float(signal.get("conviction_score", 0.0)),
                    sector=sector,
                    details=self._broker_result_details(result),
                )
                self._persist_runtime_state(force=True)
                new_positions += 1
                logger.info(
                    f"AUTO-BUY: {symbol} [{lane}] x{position_size} @ ${current_price:.2f} | "
                    f"score {score:.2f} | sector {sector}"
                )

    def _auto_trade(self):
        broker = self._components.get("broker")
        if not broker:
            return
        current_prices = {}
        for position_key, _, symbol, _, _ in self._iter_open_positions(broker):
            current_price = self._get_latest_close(symbol)
            if current_price is not None:
                current_prices[position_key] = current_price
                current_prices[symbol] = current_price
        if current_prices:
            broker.update_prices(current_prices)

        self._refresh_system_health()
        self._refresh_portfolio_overlay()
        self._manage_open_positions(broker, current_prices)
        self._execute_lane_entries(broker, self._build_lane_candidates())
        self._persist_runtime_state()

    def _start_dashboard(self, port: int):
        from core.realtime_engine import PRICE_BUFFER
        from dashboard.app import create_app

        app, socketio = create_app(
            price_buffer=PRICE_BUFFER,
            paper_broker=self._components.get("broker"),
            signal_store=self._signal_store,
            stress_results=self._components.get("stress_results", {}),
            execution_trace=self._components.get("execution_trace"),
            execution_backtest=self._components.get("execution_backtest"),
            alpha_quality=self._components.get("alpha_quality"),
            execution_reconciliation=self._components.get("execution_reconciliation"),
            portfolio_overlay=self._components.get("portfolio_overlay"),
            meta_model_status=self._components.get("meta_model_status"),
            learning_status=self._components.get("learning_status"),
            system_health=self._components.get("system_health"),
            security_suite=self._components.get("security"),
        )
        logger.info(f"\n{'=' * 52}")
        logger.info(f"  Dashboard : http://localhost:{port}")
        if self._components.get("security") is not None:
            logger.info(f"  Camera    : http://localhost:{port}/video_feed")
        logger.info(f"{'=' * 52}\n")
        socketio.run(
            app,
            host="0.0.0.0",
            port=port,
            debug=False,
            use_reloader=False,
            allow_unsafe_werkzeug=os.getenv("ALLOW_UNSAFE_WERKZEUG", "1").strip().lower() in {"1", "true", "yes", "on"},
        )

    def _data_pipeline_loop(self):
        price_pipeline_cls = None
        earnings_pipeline_cls = None
        earnings_pipeline = None
        last_price_refresh = 0.0
        last_sentiment_refresh = 0.0
        last_earnings_refresh = 0.0
        last_altdata_refresh = 0.0
        self._set_data_pipeline_status(status="starting", source="", selected_symbols=0, total_symbols=0)
        self._refresh_system_health()

        while self._running:
            try:
                if price_pipeline_cls is None:
                    from pipeline.price_collector import PriceDataPipeline as _PriceDataPipeline

                    price_pipeline_cls = _PriceDataPipeline
                if earnings_pipeline_cls is None:
                    from pipeline.earnings_collector import EarningsEventPipeline as _EarningsEventPipeline

                    earnings_pipeline_cls = _EarningsEventPipeline
                if earnings_pipeline is None:
                    earnings_pipeline = earnings_pipeline_cls(pit_db=self._components.get("pit_db"))

                target_symbols = self._get_target_symbols()
                now = time.time()
                refreshed = []

                if not self._components.get("price_data") or (now - last_price_refresh) >= self._price_refresh_seconds:
                    price_symbols = self._select_pipeline_symbols("prices", target_symbols)
                    self._set_data_pipeline_status(
                        status="refreshing",
                        source="prices",
                        selected_symbols=len(price_symbols),
                        total_symbols=len(target_symbols),
                        bootstrap_active=bool((self._source_selection_summary.get("prices") or {}).get("bootstrap_active")),
                    )
                    self._refresh_system_health()
                    self._components["price_data"] = price_pipeline_cls(symbols=price_symbols).run_incremental_update()
                    last_price_refresh = now
                    self._data_versions["prices"] += 1
                    self._record_source_refresh("prices", self._components["price_data"])
                    refreshed.append("prices")

                if not self._components.get("sentiment_data") or (now - last_sentiment_refresh) >= self._sentiment_refresh_seconds:
                    sentiment_symbols = self._select_pipeline_symbols("sentiment", target_symbols)
                    self._set_data_pipeline_status(
                        status="refreshing",
                        source="sentiment",
                        selected_symbols=len(sentiment_symbols),
                        total_symbols=len(target_symbols),
                        bootstrap_active=bool((self._source_selection_summary.get("sentiment") or {}).get("bootstrap_active")),
                    )
                    self._refresh_system_health()
                    self._components["sentiment_data"] = self._collect_sentiment_batched(symbols=sentiment_symbols, days_back=3, save=True)
                    last_sentiment_refresh = now
                    self._data_versions["sentiment"] += 1
                    self._record_source_refresh("sentiment", self._components["sentiment_data"])
                    self._refresh_event_intelligence()
                    refreshed.append("sentiment")

                if not self._components.get("earnings_data") or (now - last_earnings_refresh) >= self._earnings_refresh_seconds:
                    earnings_symbols = self._select_pipeline_symbols("earnings", target_symbols)
                    self._set_data_pipeline_status(
                        status="refreshing",
                        source="earnings",
                        selected_symbols=len(earnings_symbols),
                        total_symbols=len(target_symbols),
                        bootstrap_active=bool((self._source_selection_summary.get("earnings") or {}).get("bootstrap_active")),
                    )
                    self._refresh_system_health()
                    self._components["earnings_data"] = earnings_pipeline.run(symbols=earnings_symbols, save=True)
                    last_earnings_refresh = now
                    self._data_versions["earnings"] += 1
                    self._record_source_refresh("earnings", self._components["earnings_data"])
                    refreshed.append("earnings")

                if not self._components.get("altdata_data") or (now - last_altdata_refresh) >= self._altdata_refresh_seconds:
                    altdata_symbols = self._select_pipeline_symbols("altdata", target_symbols)
                    self._set_data_pipeline_status(
                        status="refreshing",
                        source="altdata",
                        selected_symbols=len(altdata_symbols),
                        total_symbols=len(target_symbols),
                        bootstrap_active=bool((self._source_selection_summary.get("altdata") or {}).get("bootstrap_active")),
                    )
                    self._refresh_system_health()
                    self._components["altdata_data"] = self._collect_altdata(symbols=altdata_symbols)
                    last_altdata_refresh = now
                    self._data_versions["altdata"] += 1
                    self._record_source_refresh("altdata", self._components["altdata_data"])
                    refreshed.append("altdata")

                if refreshed:
                    if self._event_intel_enabled and self._components.get("sentiment_data") and not self._components.get("event_intel"):
                        self._refresh_event_intelligence()
                    self._components["data_versions"] = dict(self._data_versions)
                    self._set_data_pipeline_status(
                        status="idle",
                        source="",
                        selected_symbols=0,
                        total_symbols=len(target_symbols),
                        refreshed_sources=list(refreshed),
                    )
                    self._refresh_system_health()
                    self._persist_runtime_state()
                    logger.info(
                        f"Data pipeline refresh complete for {len(target_symbols)} symbols in {self._universe_mode} mode "
                        f"({', '.join(refreshed)})"
                    )
            except Exception as exc:
                self._set_data_pipeline_status(
                    status="error",
                    source=str(self._data_pipeline_status.get("source") or ""),
                    selected_symbols=int(self._data_pipeline_status.get("selected_symbols", 0) or 0),
                    total_symbols=int(self._data_pipeline_status.get("total_symbols", 0) or 0),
                    error=str(exc),
                )
                logger.exception("Data pipeline error")
                price_pipeline_cls = None
                earnings_pipeline_cls = None
                earnings_pipeline = None
                self._refresh_system_health()
                self._persist_runtime_state()
            time.sleep(30)

    def _latest_live_tick(self, symbol: str):
        if not self._live_feature_overlay_enabled:
            return None
        try:
            from core.realtime_engine import PRICE_BUFFER
        except Exception:
            return None

        tick = PRICE_BUFFER.latest(symbol)
        if tick is None:
            return None
        price = float(getattr(tick, "price", 0.0) or 0.0)
        tick_ts = getattr(tick, "timestamp", None)
        if price <= 0 or not isinstance(tick_ts, datetime):
            return None
        try:
            age_seconds = max(0.0, (datetime.now(timezone.utc) - tick_ts.astimezone(timezone.utc)).total_seconds())
        except Exception:
            return None
        if age_seconds > self._live_signal_max_tick_age_seconds:
            return None
        return tick

    def _mark_price_frame_live(self, symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
        if not self._live_feature_overlay_enabled or frame is None or frame.empty:
            return frame

        tick = self._latest_live_tick(symbol)
        if tick is None:
            return frame

        try:
            base = frame.copy().sort_index()
            if "adj_close" not in base.columns and "close" in base.columns:
                base["adj_close"] = base["close"]

            tick_ts = tick.timestamp.astimezone(timezone.utc)
            live_day = pd.Timestamp(tick_ts.replace(tzinfo=None)).normalize()
            last_index = pd.Timestamp(base.index[-1])
            if last_index.tzinfo is not None:
                last_index = last_index.tz_localize(None)
            last_day = last_index.normalize()

            live_price = float(getattr(tick, "price", 0.0) or 0.0)
            tick_volume = max(0.0, float(getattr(tick, "volume", 0.0) or 0.0))
            state = self._get_intraday_state_snapshot(symbol)
            session_open = self._safe_number(state.get("session_open_price"), live_price)
            session_high = self._safe_number(state.get("session_high"), live_price)
            session_low = self._safe_number(state.get("session_low"), live_price)
            session_volume = max(self._safe_number(state.get("session_volume"), 0.0), tick_volume)

            if last_day == live_day:
                row = base.iloc[-1].copy()
                open_px = self._safe_number(row.get("open"), session_open if session_open > 0 else live_price)
                prior_high = self._safe_number(row.get("high"), open_px if open_px > 0 else live_price)
                prior_low = self._safe_number(row.get("low"), open_px if open_px > 0 else live_price)
                prior_volume = self._safe_number(row.get("volume"), 0.0)
                row["open"] = open_px
                row["high"] = max(prior_high, session_high, live_price)
                row["low"] = min(
                    value
                    for value in (prior_low, session_low if session_low > 0 else live_price, live_price)
                    if value > 0
                )
                row["close"] = live_price
                if "adj_close" in base.columns:
                    row["adj_close"] = live_price
                row["volume"] = max(prior_volume, session_volume)
                base.iloc[-1] = row
                return base

            prior_close = self._safe_number(base["close"].iloc[-1], live_price)
            open_px = session_open if session_open > 0 else (prior_close if prior_close > 0 else live_price)
            high_px = max(open_px, session_high if session_high > 0 else live_price, live_price)
            low_candidates = [value for value in (open_px, session_low if session_low > 0 else None, live_price) if value and value > 0]
            low_px = min(low_candidates) if low_candidates else live_price
            row = {
                "open": open_px,
                "high": high_px,
                "low": low_px,
                "close": live_price,
                "volume": max(session_volume, tick_volume),
                "adj_close": live_price,
            }
            base.loc[live_day] = row
            return base.sort_index()
        except Exception as exc:
            logger.debug(f"Live price mark failed for {symbol}: {exc}")
            return frame

    def _build_live_price_inputs(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        if not self._live_feature_overlay_enabled or not price_data:
            return price_data

        live_inputs: Dict[str, pd.DataFrame] = {}
        live_marks = 0
        for symbol, frame in price_data.items():
            marked = self._mark_price_frame_live(symbol, frame)
            live_inputs[symbol] = marked
            if marked is not frame:
                live_marks += 1

        target = self._components.get("live_price_data")
        if not isinstance(target, dict):
            target = {}
            self._components["live_price_data"] = target
        target.clear()
        target["price_daily_recent"] = live_inputs
        target["generated_at"] = datetime.now(timezone.utc).isoformat()
        target["live_marks"] = live_marks
        return live_inputs

    def _build_feature_signature(self, symbol: str, features, earnings_events) -> str:
        latest = features.iloc[-1]
        latest_ts = str(features.index[-1]) if len(features.index) else "na"
        parts = [
            latest_ts,
            f"{float(latest.get('close', 0.0)):.4f}",
            f"{float(latest.get('volume', 0.0)):.0f}",
            f"{float(latest.get('compound_score', 0.0)):.4f}",
            f"{float(latest.get('official_event_signal', 0.0)):.4f}",
            f"{float(latest.get('earnings_propagation_signal', 0.0)):.4f}",
            f"{float(latest.get('close_reversal_signal', 0.0)):.4f}",
            f"{float(latest.get('travel_activity_change', 0.0)):.4f}",
        ]
        if earnings_events is not None and not getattr(earnings_events, "empty", True):
            sym_events = earnings_events[earnings_events["symbol"] == symbol]
            if not sym_events.empty:
                event_row = sym_events.sort_values("reported_date").iloc[-1]
                parts.append(str(event_row.get("reported_date")))
                parts.append(f"{float(event_row.get('surprise_pct') or 0.0):.3f}")

        live_tick = self._latest_live_tick(symbol)
        if live_tick is not None:
            parts.append(str(live_tick.timestamp.astimezone(timezone.utc).isoformat()))
            parts.append(f"{float(getattr(live_tick, 'price', 0.0) or 0.0):.4f}")
            parts.append(f"{float(getattr(live_tick, 'bid', 0.0) or 0.0):.4f}")
            parts.append(f"{float(getattr(live_tick, 'ask', 0.0) or 0.0):.4f}")

        intraday_state = self._get_intraday_state_snapshot(symbol)
        if intraday_state:
            parts.append(str(intraday_state.get("session_key") or ""))
            parts.append(str(intraday_state.get("last_tick_ts") or ""))
            parts.append(f"{self._safe_number(intraday_state.get('session_volume'), 0.0):.0f}")
            parts.append(f"{self._safe_number(intraday_state.get('cum_delta'), 0.0):.1f}")
            parts.append(f"{self._safe_number(intraday_state.get('last_price'), 0.0):.4f}")

        try:
            from core.realtime_engine import DEPTH_BUFFER

            depth_snapshot = DEPTH_BUFFER.latest(symbol) or {}
        except Exception:
            depth_snapshot = {}
        if depth_snapshot:
            parts.append(str(depth_snapshot.get("updated_at") or ""))
            parts.append(f"{self._safe_number(depth_snapshot.get('best_bid'), 0.0):.6f}")
            parts.append(f"{self._safe_number(depth_snapshot.get('best_ask'), 0.0):.6f}")
        return "|".join(parts)

    def _run_stress_tests(self):
        try:
            from core.signal_engine_v2 import RegimeAwareStressTester

            tester = RegimeAwareStressTester()
            results = tester.run_regime_aware_stress_tests()
            target = self._components.get("stress_results")
            if not isinstance(target, dict):
                target = {}
                self._components["stress_results"] = target
            target.clear()
            if isinstance(results, dict):
                target.update(results)
            summary = results.get("summary", {})
            logger.info(
                f"Stress tests: Grade {summary.get('overall_grade')} | "
                f"{summary.get('stress_tests_passed')} passed | "
                f"Avg protection: {summary.get('avg_crisis_protection_pct')}%"
            )
        except Exception as exc:
            logger.error(f"Stress test error: {exc}")
            target = self._components.get("stress_results")
            if isinstance(target, dict):
                target.clear()
            else:
                self._components["stress_results"] = {}

    def retrain_institutional_models(self, feature_matrices: Optional[Dict[str, object]] = None, run_optuna: bool = False) -> Dict:
        from models.institutional_retraining import InstitutionalTrainingPipeline
        from pipeline.feature_engineering import FeaturePipeline

        if feature_matrices is None:
            price_data = self._components.get("price_data", {}).get("price_daily_recent", {})
            if not price_data:
                raise ValueError("No price data loaded. Run the data pipeline before retraining models.")
            feature_matrices = FeaturePipeline().build_feature_matrix(
                price_data=price_data,
                sentiment_data=self._components.get("sentiment_data", {}).get("symbol_sentiment_daily"),
                earnings_data=self._components.get("earnings_data", {}).get("earnings_events"),
                altdata_data=self._components.get("altdata_data", {}).get("symbol_altdata_daily"),
            )

        trainer = InstitutionalTrainingPipeline(
            xgb_horizon=max(1, int(os.getenv("XGB_RETRAIN_HORIZON_DAYS", "5"))),
            meta_horizon=max(1, int(os.getenv("META_MODEL_HORIZON_DAYS", "5"))),
            meta_take_threshold=float(os.getenv("META_MODEL_TAKE_THRESHOLD", "0.58")),
            meta_walk_forward_folds=max(2, int(os.getenv("META_MODEL_WALKFORWARD_FOLDS", "4"))),
        )
        return trainer.train_all(feature_matrices, run_optuna=run_optuna)

    def _inference_loop(self):
        from core.explainability import SignalExplainer
        from core.signal_engine_v2 import MetaDecisionEngine, MultiFactorScorer
        from models.xgboost_model import XGBoostSignalModel
        from pipeline.feature_bridge import FeatureBridge
        from pipeline.feature_engineering import FeaturePipeline
        from pipeline.universe import is_day_trade_symbol

        scorer = MultiFactorScorer()
        explainer = SignalExplainer()
        bridge = FeatureBridge()
        xgb_path = Path("models/checkpoints/xgboost_model.json")
        meta_engine = MetaDecisionEngine()
        self._components["meta_engine"] = meta_engine
        self._refresh_meta_model_status()
        self._refresh_learning_status()
        xgb_model = None
        xgb_mtime = 0.0
        meta_mtime = 0.0
        class_map = {0: "sell", 1: "neutral", 2: "buy"}

        def _reload_runtime_models(force: bool = False):
            nonlocal xgb_model, xgb_mtime, meta_engine, meta_mtime
            current_xgb_mtime = xgb_path.stat().st_mtime if xgb_path.exists() else 0.0
            meta_report_path = Path("models/checkpoints/meta_walkforward_report.json")
            current_meta_mtime = meta_report_path.stat().st_mtime if meta_report_path.exists() else 0.0
            should_reload_xgb = force or self._runtime_refresh_models or current_xgb_mtime != xgb_mtime
            should_reload_meta = force or self._runtime_refresh_models or current_meta_mtime != meta_mtime

            if should_reload_xgb:
                if xgb_path.exists():
                    try:
                        xgb_model = XGBoostSignalModel.load(xgb_path)
                        xgb_mtime = current_xgb_mtime
                        logger.info("XGBoost model loaded successfully")
                    except Exception as exc:
                        logger.warning(f"XGBoost load failed: {exc}")
                else:
                    xgb_model = None
                    xgb_mtime = 0.0

            if should_reload_meta:
                meta_engine = MetaDecisionEngine()
                self._components["meta_engine"] = meta_engine
                meta_mtime = current_meta_mtime
                self._refresh_meta_model_status()
                self._refresh_learning_status()

            if self._runtime_refresh_models and (should_reload_xgb or should_reload_meta):
                self._runtime_refresh_models = False

        _reload_runtime_models(force=True)

        while self._running:
            try:
                _reload_runtime_models()
                price_data = self._components.get("price_data", {})
                daily_prices = price_data.get("price_daily_recent", {}) if isinstance(price_data, dict) else {}

                latest_feature_rows = self._components.get("latest_feature_rows")
                if not isinstance(latest_feature_rows, dict):
                    latest_feature_rows = {}
                    self._components["latest_feature_rows"] = latest_feature_rows

                def _refresh_signal_for_symbol(symbol: str, features: Optional[pd.DataFrame], earnings_events) -> int:
                    if features is None or getattr(features, "empty", True):
                        latest_feature_rows.pop(symbol, None)
                        self._signal_store.pop(symbol, None)
                        self._last_feature_signature.pop(symbol, None)
                        return 0

                    latest_feature_rows[symbol] = features.iloc[-1]
                    signature = self._build_feature_signature(symbol, features, earnings_events)
                    if self._last_feature_signature.get(symbol) == signature and isinstance(self._signal_store.get(symbol), dict):
                        return 0

                    latest_row = features.iloc[-1]
                    scored = scorer.score(latest_row, symbol=symbol)
                    base_scored = {
                        "final_score": float(scored.get("final_score", 0.0) or 0.0),
                        "direction": str(scored.get("direction", "neutral") or "neutral"),
                        "confidence": float(scored.get("confidence", 0.0) or 0.0),
                        "conviction_score": float(scored.get("conviction_score", 0.0) or 0.0),
                        "regime": str(scored.get("regime", "normal") or "normal"),
                        "regime_multiplier": float(scored.get("regime_multiplier", 1.0) or 1.0),
                        "factor_scores": dict(scored.get("factor_scores", {}) or {}),
                        "factor_weights": dict(scored.get("factor_weights", {}) or {}),
                        "xgb_alignment": scored.get("xgb_alignment"),
                    }
                    symbol_day_mode = self._day_trading_mode and is_day_trade_symbol(symbol)
                    if symbol_day_mode:
                        self._apply_day_trading_overlay(symbol, latest_row, scored)

                    if xgb_model is not None and bridge._feature_cols is not None:
                        try:
                            ohlcv = daily_prices.get(symbol)
                            if ohlcv is not None and self._live_feature_overlay_enabled:
                                ohlcv = self._mark_price_frame_live(symbol, ohlcv)
                            X = bridge.prepare_latest(features, ohlcv)
                            if X is not None:
                                pred_class = int(xgb_model.predict_selected(X)[0])
                                pred_proba = xgb_model.predict_selected_proba(X)[0]
                                xgb_direction = class_map.get(pred_class, "neutral")
                                xgb_confidence = float(max(pred_proba))
                                event_edge = max(
                                    abs(scored["factor_scores"].get("earnings_propagation", 0)),
                                    abs(scored["factor_scores"].get("close_reversal", 0)),
                                )
                                if xgb_confidence > 0.60:
                                    if xgb_direction == scored["direction"] and xgb_direction != "neutral":
                                        blended_conf = min(1.0, (scored["confidence"] * 0.45) + (xgb_confidence * 0.55))
                                        scored["confidence"] = round(blended_conf, 4)
                                        scored["conviction_score"] = round(min(10, blended_conf * 10), 1)
                                        scored["xgb_alignment"] = "confirmed"
                                    elif xgb_confidence > 0.80 and abs(scored["final_score"]) < 0.22 and event_edge < 0.35:
                                        scored["direction"] = xgb_direction
                                        scored["confidence"] = xgb_confidence
                                        scored["conviction_score"] = round(min(10, xgb_confidence * 10), 1)
                                        scored["xgb_alignment"] = "override_weak_signal"
                                    else:
                                        scored["xgb_alignment"] = "non_blocking_disagreement"
                        except Exception as exc:
                            logger.debug(f"XGBoost inference error for {symbol}: {exc}")

                    feature_names = [c for c in features.columns if features[c].dtype != object]

                    def _build_signal_payload(scored_payload: Dict, signal_style: str, lane_override: Optional[str] = None) -> Dict:
                        built = explainer.explain_prediction(
                            symbol=symbol,
                            features=features.iloc[[-1]],
                            model_predictions=scored_payload["factor_scores"],
                            ensemble_score=scored_payload["final_score"],
                            feature_names=feature_names,
                        )
                        built["signal"] = scored_payload["direction"]
                        built["confidence"] = scored_payload["confidence"]
                        built["conviction_score"] = scored_payload["conviction_score"]
                        built["regime"] = scored_payload["regime"]
                        built["regime_multiplier"] = scored_payload["regime_multiplier"]
                        built["factor_scores"] = scored_payload["factor_scores"]
                        built["factor_weights"] = scored_payload["factor_weights"]
                        built["xgb_alignment"] = scored_payload.get("xgb_alignment")
                        built["intraday_overlay"] = scored_payload.get("intraday_overlay", {})
                        if (
                            signal_style == "day_trade_intraday"
                            and isinstance(built.get("risk_parameters"), dict)
                            and isinstance(built["intraday_overlay"], dict)
                        ):
                            risk_adjustments = built["intraday_overlay"].get("risk_adjustments", {})
                            if isinstance(risk_adjustments, dict) and risk_adjustments:
                                built["risk_parameters"] = {
                                    **built["risk_parameters"],
                                    **risk_adjustments,
                                }
                        built["signal_style"] = signal_style
                        return self._decorate_signal(symbol, built, lane_override=lane_override)

                    signal = _build_signal_payload(
                        scored,
                        "day_trade_intraday" if symbol_day_mode else "swing_event",
                        lane_override="day" if symbol_day_mode else "normal",
                    )
                    if symbol_day_mode:
                        signal["normal_lane_signal"] = _build_signal_payload(
                            base_scored,
                            "swing_event",
                            lane_override="normal",
                        )
                    self._signal_store[symbol] = signal
                    self._last_feature_signature[symbol] = signature
                    return 1

                def _apply_meta_decisions() -> None:
                    meta_decisions = meta_engine.evaluate_universe(self._signal_store, feature_rows=latest_feature_rows)
                    for symbol, meta in meta_decisions.items():
                        signal = self._signal_store.get(symbol)
                        if not isinstance(signal, dict):
                            continue
                        signal["warmup_only"] = False
                        signal["meta_decision"] = meta
                        signal["trade_eligible"] = meta.get("take_trade", False)
                        signal["take_probability"] = meta.get("take_probability", 0.0)
                        signal["skip_probability"] = meta.get("skip_probability", 1.0)
                        signal["expected_edge_pct"] = meta.get("expected_edge_pct", 0.0)
                        signal["expected_drawdown_pct"] = meta.get("expected_drawdown_pct", 0.0)
                        signal["rank_score"] = meta.get("rank_score", 0.0)
                        signal["rank_percentile"] = meta.get("rank_percentile", 0.0)
                        signal["size_multiplier"] = meta.get("size_multiplier", 0.0)
                        signal["meta_source"] = meta.get("source", "heuristic")
                        signal.pop("stale_reason", None)
                        signal.pop("stale_seconds", None)
                        normal_signal = signal.get("normal_lane_signal")
                        if isinstance(normal_signal, dict):
                            normal_signal["warmup_only"] = False
                            normal_meta = meta_engine.evaluate_universe(
                                {symbol: normal_signal},
                                feature_rows={symbol: latest_feature_rows.get(symbol)} if symbol in latest_feature_rows else {},
                            ).get(symbol, {})
                            normal_signal["meta_decision"] = normal_meta
                            normal_signal["trade_eligible"] = normal_meta.get("take_trade", False)
                            normal_signal["take_probability"] = normal_meta.get("take_probability", 0.0)
                            normal_signal["skip_probability"] = normal_meta.get("skip_probability", 1.0)
                            normal_signal["expected_edge_pct"] = normal_meta.get("expected_edge_pct", 0.0)
                            normal_signal["expected_drawdown_pct"] = normal_meta.get("expected_drawdown_pct", 0.0)
                            normal_signal["rank_score"] = normal_meta.get("rank_score", 0.0)
                            normal_signal["rank_percentile"] = normal_meta.get("rank_percentile", 0.0)
                            normal_signal["size_multiplier"] = normal_meta.get("size_multiplier", 0.0)
                            normal_signal["meta_source"] = normal_meta.get("source", "heuristic")
                            normal_signal.pop("stale_reason", None)
                            normal_signal.pop("stale_seconds", None)

                if not self._signal_store:
                    bootstrap_paths = self._select_feature_store_bootstrap_paths()
                    if bootstrap_paths:
                        feature_cache = self._components.get("feature_matrices")
                        if not isinstance(feature_cache, dict):
                            feature_cache = {}
                            self._components["feature_matrices"] = feature_cache
                        earnings_events = self._components.get("earnings_data", {}).get("earnings_events")
                        bootstrapped_signals = 0
                        bootstrap_failures = 0
                        bootstrap_map = {symbol: path for symbol, path in bootstrap_paths}
                        bootstrap_symbols = list(bootstrap_map.keys())
                        bootstrap_batches = self._chunk_symbols(bootstrap_symbols, self._inference_feature_chunk_size)
                        for batch_idx, batch_symbols in enumerate(bootstrap_batches, start=1):
                            batch_restored = 0
                            for symbol in batch_symbols:
                                path = bootstrap_map.get(symbol)
                                if path is None:
                                    continue
                                try:
                                    features = pd.read_parquet(path)
                                except Exception as exc:
                                    bootstrap_failures += 1
                                    if bootstrap_failures <= 5:
                                        logger.warning("Signal bootstrap read failed for %s: %s", path.name, exc)
                                    continue
                                if features is None or getattr(features, "empty", True):
                                    continue
                                feature_cache[symbol] = features
                                bootstrapped_signals += _refresh_signal_for_symbol(symbol, features, earnings_events)
                                signal = self._signal_store.get(symbol)
                                if isinstance(signal, dict):
                                    batch_restored += 1
                                    self._signal_store[symbol] = self._mark_signal_warmup_only(
                                        symbol,
                                        signal,
                                        reason="feature_store_bootstrap",
                                    )
                                    normal_signal = self._signal_store[symbol].get("normal_lane_signal")
                                    if isinstance(normal_signal, dict):
                                        self._signal_store[symbol]["normal_lane_signal"] = self._mark_signal_warmup_only(
                                            symbol,
                                            normal_signal,
                                            reason="feature_store_bootstrap",
                                            lane_override="normal",
                                        )
                            if len(bootstrap_batches) > 1:
                                logger.info(
                                    "Signal bootstrap progress: batch %s/%s | %s/%s files loaded | %s restored | %s failures",
                                    batch_idx,
                                    len(bootstrap_batches),
                                    min(batch_idx * self._inference_feature_chunk_size, len(bootstrap_symbols)),
                                    len(bootstrap_symbols),
                                    batch_restored,
                                    bootstrap_failures,
                                )
                        logger.info(
                            "Signal bootstrap ready: %s cached signals restored while live data warms (%s files queued, %s failures)",
                            len(self._signal_store),
                            len(bootstrap_symbols),
                            bootstrap_failures,
                        )

                if not price_data:
                    time.sleep(self._inference_idle_wait_seconds)
                    continue

                if not daily_prices:
                    time.sleep(self._inference_idle_wait_seconds)
                    continue

                current_versions = dict(self._data_versions)
                sources_changed = any(
                    current_versions.get(k, 0) != self._last_inference_versions.get(k, 0) for k in current_versions
                )
                prices_changed = current_versions.get("prices", 0) != self._last_inference_versions.get("prices", 0)
                earnings_changed = current_versions.get("earnings", 0) != self._last_inference_versions.get("earnings", 0)

                feature_cache = self._components.get("feature_matrices")
                if not isinstance(feature_cache, dict):
                    feature_cache = {}
                    self._components["feature_matrices"] = feature_cache

                event_feature_map = self._components.get("event_feature_map")
                if not isinstance(event_feature_map, dict):
                    event_feature_map = {}
                    self._components["event_feature_map"] = event_feature_map

                if prices_changed or earnings_changed or not event_feature_map:
                    try:
                        from pipeline.event_alpha import build_event_feature_matrices

                        event_feature_map.clear()
                        event_feature_map.update(
                            build_event_feature_matrices(
                                price_data=daily_prices,
                                earnings_events=self._components.get("earnings_data", {}).get("earnings_events"),
                            )
                        )
                    except Exception as exc:
                        logger.warning(f"Event alpha refresh failed: {exc}")

                symbols_to_rebuild: List[str] = []
                if sources_changed:
                    symbols_to_rebuild = list(daily_prices.keys())
                else:
                    for symbol in daily_prices.keys():
                        if symbol not in feature_cache:
                            symbols_to_rebuild.append(symbol)
                            continue
                        tick = self._latest_live_tick(symbol)
                        tick_ts = tick.timestamp.astimezone(timezone.utc).isoformat() if tick is not None else ""
                        prev_ts = self._last_seen_tick_ts.get(symbol, "")
                        if tick_ts != prev_ts:
                            symbols_to_rebuild.append(symbol)
                        self._last_seen_tick_ts[symbol] = tick_ts

                if symbols_to_rebuild:
                    tail_rows = max(0, int(float(os.getenv("FEATURE_ENGINEERING_TAIL_ROWS", "0") or 0)))
                    reversal_tail_rows = max(30, int(float(os.getenv("EVENT_REVERSAL_TAIL_ROWS", "80") or 80)))
                    price_subset: Dict[str, pd.DataFrame] = {}

                    for symbol in symbols_to_rebuild:
                        frame = daily_prices.get(symbol)
                        if frame is None or getattr(frame, "empty", True):
                            continue
                        tick = self._latest_live_tick(symbol)
                        self._last_seen_tick_ts[symbol] = (
                            tick.timestamp.astimezone(timezone.utc).isoformat() if tick is not None else ""
                        )
                        working = self._mark_price_frame_live(symbol, frame)
                        if tail_rows > 0 and len(working) > tail_rows:
                            working = working.tail(tail_rows)
                        price_subset[symbol] = working

                        # Update only the latest close-reversal columns when a live overlay is active.
                        try:
                            if self._live_feature_overlay_enabled:
                                from pipeline.event_alpha import compute_close_reversal_features

                                rev = compute_close_reversal_features(working.tail(reversal_tail_rows))
                                if rev is not None and not getattr(rev, "empty", True):
                                    last_idx = pd.Timestamp(working.index[-1])
                                    last_rev = rev.iloc[-1]
                                    existing = event_feature_map.get(symbol)
                                    if existing is None or getattr(existing, "empty", True):
                                        existing = pd.DataFrame(index=pd.DatetimeIndex(pd.to_datetime(working.index)))
                                    if last_idx not in existing.index:
                                        existing.loc[last_idx] = {col: 0.0 for col in existing.columns}
                                    for col in (
                                        "close_reversal_signal",
                                        "close_reversal_strength",
                                        "event_move_strength",
                                        "event_day_extreme",
                                    ):
                                        if col not in existing.columns:
                                            existing[col] = 0.0
                                        existing.at[last_idx, col] = float(last_rev.get(col, 0.0) or 0.0)
                                    event_feature_map[symbol] = existing
                        except Exception:
                            pass

                    if price_subset:
                        rebuild_symbols = list(price_subset.keys())
                        batches = self._chunk_symbols(rebuild_symbols, self._inference_feature_chunk_size)
                        total_batches = max(len(batches), 1)
                        updated_symbols = 0
                        earnings_events = self._components.get("earnings_data", {}).get("earnings_events")
                        for batch_idx, batch_symbols in enumerate(batches, start=1):
                            batch_prices = {symbol: price_subset[symbol] for symbol in batch_symbols}
                            refreshed = FeaturePipeline().build_feature_matrix(
                                price_data=batch_prices,
                                sentiment_data=self._components.get("sentiment_data", {}).get("symbol_sentiment_daily"),
                                earnings_data=earnings_events,
                                altdata_data=self._components.get("altdata_data", {}).get("symbol_altdata_daily"),
                                event_feature_map=event_feature_map,
                            )
                            for symbol in batch_symbols:
                                features = refreshed.get(symbol)
                                if features is None or getattr(features, "empty", True):
                                    feature_cache.pop(symbol, None)
                                else:
                                    feature_cache[symbol] = features
                                updated_symbols += _refresh_signal_for_symbol(symbol, features, earnings_events)
                            if total_batches > 1:
                                logger.info(
                                    "Inference refresh progress: batch %s/%s | %s/%s symbols rebuilt",
                                    batch_idx,
                                    total_batches,
                                    min(batch_idx * self._inference_feature_chunk_size, len(rebuild_symbols)),
                                    len(rebuild_symbols),
                                )
                    else:
                        updated_symbols = 0
                        earnings_events = self._components.get("earnings_data", {}).get("earnings_events")
                else:
                    updated_symbols = 0
                    earnings_events = self._components.get("earnings_data", {}).get("earnings_events")

                feature_matrices = feature_cache
                self._last_inference_versions = current_versions

                updated_symbols += self._refresh_crypto_signals(latest_feature_rows)

                _apply_meta_decisions()

                self._auto_trade()
                self._maybe_refresh_execution_backtest(feature_matrices)
                self._persist_feature_store(feature_matrices, updated_symbols=symbols_to_rebuild)
                self._maybe_periodic_retrain(feature_matrices)

                buys = sum(1 for s in self._signal_store.values() if s.get("signal") == "buy")
                sells = sum(1 for s in self._signal_store.values() if s.get("signal") == "sell")
                holds = len(self._signal_store) - buys - sells
                logger.info(f"Signals: {buys} BUY | {sells} SELL | {holds} HOLD | {updated_symbols} updated")
            except Exception as exc:
                logger.error(f"Inference loop error: {exc}")
            time.sleep(self._inference_refresh_seconds)


if __name__ == "__main__":
    from pipeline.alerts_scheduler import setup_production_logging

    setup_production_logging(log_dir=os.getenv("LOG_DIR", "logs"))
    print("Starting Macro Intelligence System...")

    def _env_flag(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    MacroIntelligenceSystem().start(
        run_dashboard=_env_flag("RUN_DASHBOARD", True),
        run_realtime=_env_flag("RUN_REALTIME", True),
        run_backtest=_env_flag("RUN_BACKTEST", False),
        run_security=_env_flag("RUN_SECURITY", False),
        dashboard_port=int(os.getenv("DASHBOARD_PORT", "5050")),
    )
