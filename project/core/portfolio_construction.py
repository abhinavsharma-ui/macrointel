"""
Institutional Portfolio Construction
====================================
Builds target portfolio weights from the live signal set using:
  - covariance-aware risk penalties
  - factor crowding penalties
  - residual alpha estimates
  - beta-neutrality penalties

This is intentionally lightweight so it can run inside the live loop.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class InstitutionalPortfolioConstructor:
    FACTOR_ORDER = [
        "trend",
        "momentum",
        "mean_revert",
        "volume",
        "sentiment",
        "earnings_propagation",
        "close_reversal",
    ]

    def __init__(
        self,
        lookback_days: int = 60,
        gross_target_pct: float = 0.65,
        max_names: int = 12,
        min_weight: float = 0.03,
        max_weight: float = 0.18,
        covariance_shrinkage: float = 0.25,
        factor_penalty: float = 0.30,
        beta_penalty: float = 0.45,
        target_beta: float = 0.10,
    ):
        self.lookback_days = lookback_days
        self.gross_target_pct = min(0.95, max(0.10, gross_target_pct))
        self.max_names = max(1, max_names)
        self.min_weight = max(0.0, min_weight)
        self.max_weight = min(1.0, max(max_weight, self.min_weight))
        self.covariance_shrinkage = float(np.clip(covariance_shrinkage, 0.0, 0.95))
        self.factor_penalty = max(0.0, factor_penalty)
        self.beta_penalty = max(0.0, beta_penalty)
        self.target_beta = target_beta

    def _return_matrix(self, symbols, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        series_map = {}
        for symbol in symbols:
            frame = price_data.get(symbol)
            if frame is None or frame.empty or "close" not in frame.columns:
                continue
            close = frame["close"].dropna()
            if len(close) < 25:
                continue
            returns = close.pct_change().dropna().tail(self.lookback_days)
            if len(returns) < 15:
                continue
            series_map[symbol] = returns.rename(symbol)
        if not series_map:
            return pd.DataFrame()
        return pd.concat(series_map.values(), axis=1, join="inner").dropna(how="any")

    def _factor_matrix(self, symbols, signals: Dict[str, Dict]) -> np.ndarray:
        rows = []
        for symbol in symbols:
            factor_scores = (signals.get(symbol, {}) or {}).get("factor_scores", {}) or {}
            rows.append([float(factor_scores.get(name, 0.0) or 0.0) for name in self.FACTOR_ORDER])
        return np.array(rows, dtype=float) if rows else np.zeros((0, len(self.FACTOR_ORDER)))

    def _expected_residual_alpha(self, symbols, signals: Dict[str, Dict], returns: pd.DataFrame, market_proxy: pd.Series, betas: np.ndarray) -> np.ndarray:
        mu = []
        for idx, symbol in enumerate(symbols):
            signal = signals.get(symbol, {}) or {}
            meta = signal.get("meta_decision", {}) or {}
            edge_pct = float(signal.get("expected_edge_pct", meta.get("expected_edge_pct", 0.0)) or 0.0) / 100.0
            take_prob = float(signal.get("take_probability", meta.get("take_probability", 0.0)) or 0.0)
            rank_score = float(signal.get("rank_score", meta.get("rank_score", 0.0)) or 0.0)
            symbol_returns = returns[symbol]
            residual = symbol_returns - (betas[idx] * market_proxy)
            residual_momentum = float(residual.tail(min(10, len(residual))).mean()) if not residual.empty else 0.0
            event_strength = abs(float(signal.get("factor_scores", {}).get("earnings_propagation", 0.0) or 0.0)) + abs(
                float(signal.get("factor_scores", {}).get("close_reversal", 0.0) or 0.0)
            )
            alpha = (
                edge_pct * 0.55
                + residual_momentum * 8.0 * 0.20
                + take_prob * 0.15
                + rank_score * 0.08
                + min(event_strength, 1.0) * 0.02
            )
            mu.append(alpha)
        return np.array(mu, dtype=float)

    def _cap_weights(self, weights: np.ndarray) -> np.ndarray:
        weights = np.maximum(weights, 0.0)
        if weights.sum() <= 0:
            return weights
        weights = weights / weights.sum()

        for _ in range(10):
            over = weights > self.max_weight
            if not over.any():
                break
            excess = float((weights[over] - self.max_weight).sum())
            weights[over] = self.max_weight
            under = ~over
            if under.any() and excess > 0:
                under_weights = weights[under]
                if under_weights.sum() > 0:
                    weights[under] += excess * (under_weights / under_weights.sum())

        weights = np.maximum(weights, 0.0)
        if weights.sum() > 0:
            weights = weights / weights.sum()

        if self.min_weight > 0:
            keep = weights >= self.min_weight
            if keep.any() and keep.sum() < len(weights):
                weights = weights * keep
                if weights.sum() > 0:
                    weights = weights / weights.sum()

        return weights

    def construct(self, signals: Dict[str, Dict], broker, price_data: Dict[str, pd.DataFrame]) -> Dict:
        candidates = []
        for symbol, signal in signals.items():
            if not isinstance(signal, dict):
                continue
            has_open_symbol = (
                broker.has_open_position(symbol=symbol)
                if hasattr(broker, "has_open_position")
                else any(
                    getattr(position, "quantity", 0) > 0 and getattr(position, "symbol", "") == symbol
                    for position in getattr(broker, "positions", {}).values()
                )
            )
            if has_open_symbol:
                if signal.get("signal") != "sell":
                    candidates.append(symbol)
                continue
            if signal.get("signal") != "buy":
                continue
            if signal.get("trade_eligible") is False:
                continue
            candidates.append(symbol)

        if not candidates:
            return {"timestamp": datetime.now(timezone.utc).isoformat(), "allocations": {}, "summary": {"selected_symbols": 0}}

        raw_returns = self._return_matrix(candidates, price_data)
        if raw_returns.empty or raw_returns.shape[1] == 0:
            return {"timestamp": datetime.now(timezone.utc).isoformat(), "allocations": {}, "summary": {"selected_symbols": 0}}

        symbols = list(raw_returns.columns)
        if len(symbols) > self.max_names:
            symbols = sorted(
                symbols,
                key=lambda sym: float((signals.get(sym, {}) or {}).get("rank_score", 0.0) or 0.0),
                reverse=True,
            )[: self.max_names]
            raw_returns = raw_returns[symbols]

        market_proxy = raw_returns.mean(axis=1)
        market_var = float(market_proxy.var()) if not market_proxy.empty else 0.0
        betas = []
        for symbol in symbols:
            if market_var <= 1e-9:
                betas.append(1.0)
                continue
            cov = float(raw_returns[symbol].cov(market_proxy))
            betas.append(cov / market_var)
        betas = np.array(betas, dtype=float)

        mu = self._expected_residual_alpha(symbols, signals, raw_returns, market_proxy, betas)
        if np.all(mu <= 0):
            mu = np.array(
                [
                    max(
                        float((signals.get(sym, {}) or {}).get("take_probability", 0.0) or 0.0) * 0.10,
                        0.001,
                    )
                    for sym in symbols
                ],
                dtype=float,
            )

        cov = raw_returns.cov().to_numpy(dtype=float)
        diag = np.diag(np.diag(cov))
        cov_shrunk = ((1.0 - self.covariance_shrinkage) * cov) + (self.covariance_shrinkage * diag)
        factor_matrix = self._factor_matrix(symbols, signals)
        factor_penalty_matrix = (
            (factor_matrix @ factor_matrix.T) / max(1, factor_matrix.shape[1])
            if factor_matrix.size
            else np.zeros_like(cov_shrunk)
        )
        beta_gap = betas - self.target_beta
        beta_penalty_matrix = np.outer(beta_gap, beta_gap)

        risk_matrix = cov_shrunk + (self.factor_penalty * factor_penalty_matrix) + (self.beta_penalty * beta_penalty_matrix)
        try:
            raw_weights = np.linalg.pinv(risk_matrix) @ mu
        except Exception:
            raw_weights = mu.copy()

        raw_weights = np.maximum(raw_weights, 0.0)
        if raw_weights.sum() <= 0:
            raw_weights = np.maximum(mu, 0.0)
        if raw_weights.sum() <= 0:
            raw_weights = np.ones(len(symbols), dtype=float)

        weights = self._cap_weights(raw_weights)
        net_beta = float(np.dot(weights, betas)) if len(weights) else 0.0
        beta_gap_abs = abs(net_beta - self.target_beta)
        beta_scale = max(0.55, 1.0 - min(0.45, beta_gap_abs * 0.35))
        gross_target_pct = self.gross_target_pct * beta_scale

        factor_exposures = {}
        if factor_matrix.size:
            exposure_vector = weights @ factor_matrix
            factor_exposures = {
                name: round(float(exposure_vector[idx]), 4) for idx, name in enumerate(self.FACTOR_ORDER)
            }

        allocations = {}
        for idx, symbol in enumerate(symbols):
            target_weight = float(weights[idx])
            signal = signals.get(symbol, {}) or {}
            meta = signal.get("meta_decision", {}) or {}
            target_position_pct = target_weight * gross_target_pct
            portfolio_score = float(mu[idx]) * 100.0 + float(meta.get("rank_score", 0.0) or 0.0)
            allocations[symbol] = {
                "target_weight": round(target_weight, 4),
                "target_position_pct": round(target_position_pct, 4),
                "portfolio_score": round(portfolio_score, 4),
                "residual_alpha_score": round(float(mu[idx]), 5),
                "beta": round(float(betas[idx]), 4),
                "take_probability": round(float(meta.get("take_probability", signal.get("take_probability", 0.0)) or 0.0), 4),
            }

        summary = {
            "selected_symbols": len(symbols),
            "gross_target_pct": round(gross_target_pct, 4),
            "net_beta": round(net_beta, 4),
            "target_beta": round(self.target_beta, 4),
            "beta_gap": round(net_beta - self.target_beta, 4),
            "hedge_ratio": round(max(0.0, net_beta - self.target_beta), 4),
            "factor_exposures": factor_exposures,
        }

        logger.info(
            "Portfolio overlay refreshed: "
            f"{summary['selected_symbols']} names | gross {summary['gross_target_pct']:.2f} | "
            f"net beta {summary['net_beta']:.2f}"
        )
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "allocations": allocations,
            "summary": summary,
        }
