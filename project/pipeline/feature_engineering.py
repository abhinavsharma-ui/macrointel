"""
Feature Engineering Pipeline
============================
Converts raw OHLCV price data into feature matrices for live inference.

The pipeline now includes:
  - baseline technicals
  - scorer-compatible aliases used by MultiFactorScorer
  - sentiment velocity and z-score
  - event-driven alpha features for earnings propagation and close reversal
"""

import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
FEATURE_FLOAT_DTYPE = (os.getenv("FEATURE_FLOAT_DTYPE", "") or "").strip().lower()

EVENT_FEATURE_COLUMNS = [
    "peer_earnings_shock_3d",
    "peer_earnings_shock_7d",
    "peer_earnings_breadth_7d",
    "peer_earnings_event_count_7d",
    "peer_earnings_negative_ratio_7d",
    "earnings_propagation_signal",
    "earnings_propagation_strength",
    "close_reversal_signal",
    "close_reversal_strength",
    "event_move_strength",
    "event_day_extreme",
]

SENTIMENT_FEATURE_COLUMNS = [
    "compound_score",
    "weighted_compound_score",
    "media_sentiment",
    "official_sentiment",
    "filing_sentiment",
    "filing_change_score",
    "filing_fresh_language_score",
    "new_risk_factors",
    "earnings_tone_signal",
    "earnings_call_count",
    "article_count",
    "media_article_count",
    "official_article_count",
    "filing_article_count",
    "press_release_count",
    "official_event_hit",
    "filing_event_hit",
    "source_quality_score",
]

ALT_DATA_FEATURE_COLUMNS = [
    "travel_activity_level",
    "travel_activity_change",
]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def _bollinger_bands(close: pd.Series, period: int = 20, std_dev: int = 2):
    mid = close.rolling(period).mean()
    sigma = close.rolling(period).std()
    upper = mid + std_dev * sigma
    lower = mid - std_dev * sigma
    width = (upper - lower) / mid.replace(0, np.nan)
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return mid, upper, lower, width, pct_b


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * volume).cumsum()


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    highest = high.rolling(period).max()
    lowest = low.rolling(period).min()
    return -100 * (highest - close) / (highest - lowest).replace(0, np.nan)


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3):
    lowest = low.rolling(k).min()
    highest = high.rolling(k).max()
    k_pct = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    d_pct = k_pct.rolling(d).mean()
    return k_pct, d_pct


def _money_flow_index(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    direction = typical_price.diff()
    positive_flow = raw_money_flow.where(direction > 0, 0.0)
    negative_flow = raw_money_flow.where(direction < 0, 0.0).abs()
    pos_sum = positive_flow.rolling(period).sum()
    neg_sum = negative_flow.rolling(period).sum()
    money_ratio = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - (100 / (1 + money_ratio))


def _chaikin_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    fast: int = 3,
    slow: int = 10,
) -> pd.Series:
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mfv = mfm.fillna(0.0) * volume
    adl = mfv.cumsum()
    return adl.ewm(span=fast, adjust=False).mean() - adl.ewm(span=slow, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return plus_di, minus_di, adx


class FeaturePipeline:
    def build_feature_matrix(
        self,
        price_data: Dict[str, pd.DataFrame],
        sentiment_data: Optional[pd.DataFrame] = None,
        earnings_data: Optional[pd.DataFrame] = None,
        altdata_data: Optional[pd.DataFrame] = None,
        event_feature_map: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, pd.DataFrame]:
        results: Dict[str, pd.DataFrame] = {}
        resolved_event_map: Dict[str, pd.DataFrame] = {}

        if price_data:
            if isinstance(event_feature_map, dict) and event_feature_map:
                resolved_event_map = event_feature_map
            else:
                try:
                    from pipeline.event_alpha import build_event_feature_matrices

                    resolved_event_map = build_event_feature_matrices(
                        price_data=price_data,
                        earnings_events=earnings_data,
                    )
                except Exception as exc:
                    logger.error(f"Event alpha feature build failed: {exc}")

        for symbol, ohlcv in price_data.items():
            try:
                engineered = self._engineer_features(
                    df=ohlcv.copy(),
                    symbol=symbol,
                    sentiment_data=sentiment_data,
                    event_data=resolved_event_map.get(symbol),
                    altdata_data=altdata_data,
                )
                if engineered is not None and not engineered.empty:
                    results[symbol] = engineered
            except Exception as exc:
                logger.error(f"Feature engineering failed for {symbol}: {exc}")
        return results

    def _engineer_features(
        self,
        df: pd.DataFrame,
        symbol: str,
        sentiment_data: Optional[pd.DataFrame],
        event_data: Optional[pd.DataFrame],
        altdata_data: Optional[pd.DataFrame],
    ) -> Optional[pd.DataFrame]:
        if len(df) < 60:
            logger.warning(f"{symbol}: insufficient data ({len(df)} rows), need >=60")
            return None

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)
        feat = pd.DataFrame(index=df.index)
        feat["open"] = df["open"].astype(float)
        feat["high"] = high
        feat["low"] = low
        feat["close"] = close
        feat["volume"] = volume

        feat["rsi_14"] = _rsi(close, 14)
        feat["rsi_9"] = _rsi(close, 9)
        feat["rsi_21"] = _rsi(close, 21)
        macd_line, macd_signal, macd_hist = _macd(close)
        feat["macd"] = macd_line
        feat["macd_signal"] = macd_signal
        feat["macd_hist"] = macd_hist
        feat["roc_5"] = close.pct_change(5)
        feat["roc_10"] = close.pct_change(10)
        feat["roc_20"] = close.pct_change(20)
        feat["williams_r"] = _williams_r(high, low, close)
        stoch_k, stoch_d = _stochastic(high, low, close)
        feat["stoch_k"] = stoch_k
        feat["stoch_d"] = stoch_d

        for span in [9, 21, 50, 200]:
            feat[f"ema_{span}"] = close.ewm(span=span, adjust=False).mean()

        sma_9 = close.rolling(9).mean()
        sma_50 = close.rolling(50).mean()
        sma_200 = close.rolling(200).mean()
        feat["price_vs_ema9"] = (close / feat["ema_9"] - 1) * 100
        feat["price_vs_ema21"] = (close / feat["ema_21"] - 1) * 100
        feat["price_vs_ema50"] = (close / feat["ema_50"] - 1) * 100
        feat["price_vs_ema200"] = (close / feat["ema_200"] - 1) * 100
        feat["golden_cross"] = (feat["ema_50"] > feat["ema_200"]).astype(float)
        feat["ema_slope_9"] = feat["ema_9"].pct_change(3)
        feat["close_vs_sma_9"] = close / sma_9.replace(0, np.nan) - 1
        feat["close_vs_sma_50"] = close / sma_50.replace(0, np.nan) - 1
        feat["close_vs_sma_200"] = close / sma_200.replace(0, np.nan) - 1
        feat["momentum_20d"] = close.pct_change(20)
        feat["momentum_60d"] = close.pct_change(60)
        feat["momentum_squared"] = feat["momentum_20d"] ** 2
        feat["momentum_differencing"] = feat["momentum_20d"].diff()
        feat["price_acceleration"] = feat["momentum_20d"] - feat["momentum_60d"]
        feat["returns_1d"] = close.pct_change(1)

        feat["atr_14"] = _atr(high, low, close)
        feat["atr_pct"] = feat["atr_14"] / close
        _, _, _, bb_width, bb_pct = _bollinger_bands(close)
        feat["bb_width"] = bb_width
        feat["bb_pct"] = bb_pct
        feat["bb_position"] = bb_pct
        feat["hist_vol_10"] = close.pct_change().rolling(10).std() * np.sqrt(252)
        feat["hist_vol_30"] = close.pct_change().rolling(30).std() * np.sqrt(252)
        feat["vol_ratio"] = feat["hist_vol_10"] / feat["hist_vol_30"].replace(0, np.nan)
        mean_60 = close.rolling(60).mean()
        std_60 = close.rolling(60).std().replace(0, np.nan)
        feat["zscore_vs_60d"] = (close - mean_60) / std_60
        feat["realized_vol_21d"] = close.pct_change().rolling(21).std() * np.sqrt(252)
        feat["vol_regime_stressed"] = (
            feat["realized_vol_21d"] > feat["realized_vol_21d"].rolling(63).mean()
        ).astype(float)
        feat["vol_regime_ratio"] = (
            feat["realized_vol_21d"] / feat["realized_vol_21d"].rolling(30, min_periods=10).mean().replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.1, 5.0)

        vol_ma20 = volume.rolling(20).mean()
        feat["vol_ratio_20"] = volume / vol_ma20.replace(0, np.nan)
        feat["obv"] = _obv(close, volume)
        feat["obv_log"] = np.log1p(feat["obv"].abs())
        feat["obv_acceleration"] = feat["obv"].diff().diff()
        feat["obv_slope"] = feat["obv"].pct_change(5)
        feat["obv_trend"] = feat["obv"].diff(5)
        feat["vol_zscore"] = ((volume - vol_ma20) / volume.rolling(20).std().replace(0, np.nan)).clip(-3, 3)
        feat["mfi"] = _money_flow_index(high, low, close, volume)
        feat["chaikin_osc"] = _chaikin_oscillator(high, low, close, volume)
        di_plus, di_minus, adx_14 = _adx(high, low, close)
        feat["di_plus"] = di_plus
        feat["di_minus"] = di_minus
        feat["adx_14"] = adx_14

        feat["gap_up"] = ((df["open"] - df["close"].shift()) / df["close"].shift()).clip(-0.1, 0.1)
        feat["candle_body"] = (close - df["open"]) / df["open"]
        feat["upper_wick"] = (high - close.clip(lower=df["open"])) / close.clip(lower=1)
        feat["lower_wick"] = (close.clip(upper=df["open"]) - low) / close.clip(lower=1)
        feat["sma_50_squared"] = feat["close_vs_sma_50"] ** 2
        feat["sma_50_sign"] = np.sign(feat["close_vs_sma_50"])
        feat["sma_200_squared"] = feat["close_vs_sma_200"] ** 2
        feat["sma_200_sign"] = np.sign(feat["close_vs_sma_200"])
        feat["macd_hist_strength"] = feat["macd_hist"].abs()
        feat["stoch_cross"] = feat["stoch_k"] - feat["stoch_d"]
        feat["52w_high_ratio"] = close / close.rolling(252).max().replace(0, np.nan)
        feat["52w_low_ratio"] = close / close.rolling(252).min().replace(0, np.nan)
        feat["close_vs_sma_200_normalized"] = feat["close_vs_sma_200"].clip(-0.2, 0.2) / 0.2

        rsi_norm = (50 - feat["rsi_14"]) / 50
        macd_norm = feat["macd_hist"].clip(-2, 2) / 2
        roc_norm = feat["roc_10"].clip(-0.1, 0.1) / 0.1
        bb_norm = (0.5 - feat["bb_pct"]).clip(-0.5, 0.5) / 0.5
        obv_norm = feat["obv_slope"].clip(-0.1, 0.1) / 0.1
        feat["momentum_composite"] = (
            0.25 * rsi_norm
            + 0.25 * macd_norm
            + 0.20 * roc_norm
            + 0.15 * bb_norm
            + 0.15 * obv_norm
        ).fillna(0.0).clip(-1, 1)

        feat["trend_regime"] = (feat["ema_50"] > feat["ema_200"]).astype(int)
        feat["vol_regime"] = (
            feat["hist_vol_30"] > feat["hist_vol_30"].rolling(60).mean()
        ).astype(int)

        if sentiment_data is not None:
            try:
                sym_sent = (
                    sentiment_data.xs(symbol, level="symbol")
                    if "symbol" in sentiment_data.index.names
                    else None
                )
                if sym_sent is not None and not sym_sent.empty:
                    available_cols = [col for col in SENTIMENT_FEATURE_COLUMNS if col in sym_sent.columns]
                    feat = feat.join(sym_sent[available_cols], how="left")
                    missing_sent = [col for col in SENTIMENT_FEATURE_COLUMNS if col not in feat.columns]
                    if missing_sent:
                        feat = pd.concat(
                            [feat, pd.DataFrame({c: 0.0 for c in missing_sent}, index=feat.index)],
                            axis=1,
                        )
                    feat[SENTIMENT_FEATURE_COLUMNS] = feat[SENTIMENT_FEATURE_COLUMNS].fillna(0.0)
                else:
                    sent_zero = pd.DataFrame(
                        {col: 0.0 for col in SENTIMENT_FEATURE_COLUMNS}, index=feat.index
                    )
                    feat = feat.drop(columns=[c for c in SENTIMENT_FEATURE_COLUMNS if c in feat.columns], errors="ignore")
                    feat = pd.concat([feat, sent_zero], axis=1)
            except Exception:
                sent_zero = pd.DataFrame(
                    {col: 0.0 for col in SENTIMENT_FEATURE_COLUMNS}, index=feat.index
                )
                feat = feat.drop(columns=[c for c in SENTIMENT_FEATURE_COLUMNS if c in feat.columns], errors="ignore")
                feat = pd.concat([feat, sent_zero], axis=1)
        else:
            sent_zero = pd.DataFrame(
                {col: 0.0 for col in SENTIMENT_FEATURE_COLUMNS}, index=feat.index
            )
            feat = feat.drop(columns=[c for c in SENTIMENT_FEATURE_COLUMNS if c in feat.columns], errors="ignore")
            feat = pd.concat([feat, sent_zero], axis=1)

        sent_mean = feat["compound_score"].rolling(20, min_periods=5).mean()
        sent_std = feat["compound_score"].rolling(20, min_periods=5).std().replace(0, np.nan)
        feat["sentiment_zscore"] = (
            (feat["compound_score"] - sent_mean) / sent_std
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        feat["sentiment_velocity"] = feat["compound_score"].diff(3).fillna(0.0)
        feat["earnings_tone_velocity"] = (
            feat["earnings_tone_signal"]
            - feat["earnings_tone_signal"].rolling(63, min_periods=5).mean().shift(1)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.5, 1.5)
        weighted_sent_mean = feat["weighted_compound_score"].rolling(20, min_periods=5).mean()
        weighted_sent_std = feat["weighted_compound_score"].rolling(20, min_periods=5).std().replace(0, np.nan)
        feat["weighted_sentiment_zscore"] = (
            (feat["weighted_compound_score"] - weighted_sent_mean) / weighted_sent_std
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        news_count_mean = feat["article_count"].rolling(20, min_periods=5).mean()
        news_count_std = feat["article_count"].rolling(20, min_periods=5).std().replace(0, np.nan)
        feat["news_volume_spike"] = (
            (feat["article_count"] - news_count_mean) / news_count_std
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-3, 3)
        feat["source_quality_signal"] = (
            ((feat["source_quality_score"] - 0.50) / 0.35)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(-1, 1)
        )
        feat["official_event_signal"] = (
            0.60 * feat["official_sentiment"].clip(-1, 1)
            + 0.25 * feat["official_event_hit"].clip(0, 1)
            + 0.15 * feat["source_quality_signal"]
        ).clip(-1, 1)
        feat["filing_event_signal"] = (
            0.60 * feat["filing_sentiment"].clip(-1, 1)
            + 0.18 * feat["filing_change_score"].clip(-1, 1)
            + 0.12 * feat["filing_fresh_language_score"].clip(-1, 1)
            - 0.08 * feat["new_risk_factors"].clip(0, 1)
            + 0.30 * feat["filing_event_hit"].clip(0, 1)
            + 0.10 * feat["source_quality_signal"]
        ).clip(-1, 1)
        feat["media_sentiment_signal"] = (
            0.65 * feat["media_sentiment"].clip(-1, 1)
            + 0.20 * feat["weighted_sentiment_zscore"].clip(-2, 2) / 2
            + 0.15 * feat["news_volume_spike"].clip(-2, 2) / 2
        ).clip(-1, 1)

        # --- event features (batch assign to avoid DataFrame fragmentation) ---
        if event_data is not None and not event_data.empty:
            aligned_events = event_data.reindex(feat.index).fillna(0.0)
            event_cols_df = pd.DataFrame(
                {col: aligned_events.get(col, 0.0) for col in EVENT_FEATURE_COLUMNS},
                index=feat.index,
            )
        else:
            event_cols_df = pd.DataFrame(
                {col: 0.0 for col in EVENT_FEATURE_COLUMNS},
                index=feat.index,
            )
        # Drop any already-present event columns before concat to avoid dupes
        feat = feat.drop(columns=[c for c in EVENT_FEATURE_COLUMNS if c in feat.columns], errors="ignore")
        feat = pd.concat([feat, event_cols_df], axis=1)
        feat["reversal_strength"] = feat["close_reversal_signal"].abs()
        feat["reversal_sign"] = np.sign(feat["close_reversal_signal"])

        # --- alt-data features (batch assign) ---
        if altdata_data is not None:
            try:
                sym_alt = (
                    altdata_data.xs(symbol, level="symbol")
                    if "symbol" in altdata_data.index.names
                    else None
                )
                if sym_alt is not None and not sym_alt.empty:
                    available_cols = [col for col in ALT_DATA_FEATURE_COLUMNS if col in sym_alt.columns]
                    feat = feat.join(sym_alt[available_cols], how="left")
                    # Fill any missing alt columns with 0.0 in one shot
                    missing_alt = [col for col in ALT_DATA_FEATURE_COLUMNS if col not in feat.columns]
                    if missing_alt:
                        feat = pd.concat(
                            [feat, pd.DataFrame({c: 0.0 for c in missing_alt}, index=feat.index)],
                            axis=1,
                        )
                    feat[ALT_DATA_FEATURE_COLUMNS] = feat[ALT_DATA_FEATURE_COLUMNS].fillna(0.0)
                else:
                    alt_zero_df = pd.DataFrame(
                        {col: 0.0 for col in ALT_DATA_FEATURE_COLUMNS}, index=feat.index
                    )
                    feat = feat.drop(columns=[c for c in ALT_DATA_FEATURE_COLUMNS if c in feat.columns], errors="ignore")
                    feat = pd.concat([feat, alt_zero_df], axis=1)
            except Exception:
                alt_zero_df = pd.DataFrame(
                    {col: 0.0 for col in ALT_DATA_FEATURE_COLUMNS}, index=feat.index
                )
                feat = feat.drop(columns=[c for c in ALT_DATA_FEATURE_COLUMNS if c in feat.columns], errors="ignore")
                feat = pd.concat([feat, alt_zero_df], axis=1)
        else:
            alt_zero_df = pd.DataFrame(
                {col: 0.0 for col in ALT_DATA_FEATURE_COLUMNS}, index=feat.index
            )
            feat = feat.drop(columns=[c for c in ALT_DATA_FEATURE_COLUMNS if c in feat.columns], errors="ignore")
            feat = pd.concat([feat, alt_zero_df], axis=1)

        sent_norm = feat["weighted_compound_score"].clip(-1, 1)
        feat["event_alpha_signal"] = (
            0.60 * feat["earnings_propagation_signal"]
            + 0.40 * feat["close_reversal_signal"]
        ).fillna(0.0).clip(-1, 1)
        feat["adaptive_horizon_multiplier"] = np.where(
            feat["vol_regime_ratio"] > 1.5,
            0.60,
            1.0,
        )
        event_extension_mask = (
            feat["official_event_hit"].clip(0, 1) > 0
        ) | (feat["filing_event_hit"].clip(0, 1) > 0)
        feat.loc[event_extension_mask, "adaptive_horizon_multiplier"] = np.maximum(
            feat.loc[event_extension_mask, "adaptive_horizon_multiplier"],
            3.0,
        )
        feat["alpha_signal"] = (
            0.45 * feat["momentum_composite"]
            + 0.15 * sent_norm
            + 0.18 * feat["event_alpha_signal"]
            + 0.10 * feat["official_event_signal"]
            + 0.05 * feat["filing_event_signal"]
            + 0.04 * feat["earnings_tone_velocity"].clip(-1, 1)
            + 0.03 * feat["filing_change_score"].clip(-1, 1)
            + 0.05 * feat["travel_activity_change"].clip(-1, 1)
        ).fillna(0.0).clip(-1, 1)

        ema_cols = [col for col in feat.columns if col.startswith("ema_")]
        feat = feat.drop(columns=ema_cols)
        feat = feat.dropna(thresh=max(1, len(feat.columns) // 2))

        if FEATURE_FLOAT_DTYPE in {"float32", "f32"}:
            try:
                feat = feat.astype(np.float32)
            except Exception:
                pass

        logger.debug(f"{symbol}: {len(feat)} rows, {len(feat.columns)} features")
        return feat

    def get_feature_names(self) -> list:
        return [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "rsi_14",
            "rsi_9",
            "rsi_21",
            "macd",
            "macd_signal",
            "macd_hist",
            "roc_5",
            "roc_10",
            "roc_20",
            "williams_r",
            "stoch_k",
            "stoch_d",
            "price_vs_ema9",
            "price_vs_ema21",
            "price_vs_ema50",
            "price_vs_ema200",
            "golden_cross",
            "ema_slope_9",
            "close_vs_sma_50",
            "close_vs_sma_200",
            "momentum_20d",
            "momentum_60d",
            "price_acceleration",
            "returns_1d",
            "atr_14",
            "atr_pct",
            "bb_width",
            "bb_pct",
            "bb_position",
            "hist_vol_10",
            "hist_vol_30",
            "vol_ratio",
            "zscore_vs_60d",
            "realized_vol_21d",
            "vol_regime_stressed",
            "vol_regime_ratio",
            "vol_ratio_20",
            "obv",
            "obv_slope",
            "obv_trend",
            "vol_zscore",
            "mfi",
            "chaikin_osc",
            "di_plus",
            "di_minus",
            "adx_14",
            "gap_up",
            "candle_body",
            "upper_wick",
            "lower_wick",
            "52w_high_ratio",
            "52w_low_ratio",
            "momentum_composite",
            "trend_regime",
            "vol_regime",
            "compound_score",
            "weighted_compound_score",
            "media_sentiment",
            "official_sentiment",
            "filing_sentiment",
            "filing_change_score",
            "filing_fresh_language_score",
            "new_risk_factors",
            "earnings_tone_signal",
            "earnings_call_count",
            "article_count",
            "media_article_count",
            "official_article_count",
            "filing_article_count",
            "press_release_count",
            "official_event_hit",
            "filing_event_hit",
            "source_quality_score",
            "sentiment_zscore",
            "sentiment_velocity",
            "earnings_tone_velocity",
            "weighted_sentiment_zscore",
            "news_volume_spike",
            "source_quality_signal",
            "official_event_signal",
            "filing_event_signal",
            "media_sentiment_signal",
            "peer_earnings_shock_3d",
            "peer_earnings_shock_7d",
            "peer_earnings_breadth_7d",
            "peer_earnings_event_count_7d",
            "peer_earnings_negative_ratio_7d",
            "earnings_propagation_signal",
            "earnings_propagation_strength",
            "close_reversal_signal",
            "close_reversal_strength",
            "event_move_strength",
            "event_day_extreme",
            "travel_activity_level",
            "travel_activity_change",
            "event_alpha_signal",
            "adaptive_horizon_multiplier",
            "alpha_signal",
        ]
