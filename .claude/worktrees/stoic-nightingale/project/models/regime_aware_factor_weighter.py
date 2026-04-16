"""
Regime-aware factor weighting.

This module adapts the factor mix by market regime while keeping the final
weights normalized.
"""

from __future__ import annotations

from typing import Dict


class RegimeAwareFactorWeighter:
    """
    Overrides factor weights based on regime.
    Returns adapted weights dict per regime.
    """

    BASE_WEIGHTS = {
        "trend": 0.16,
        "momentum": 0.14,
        "mean_revert": 0.10,
        "volume": 0.08,
        "sentiment": 0.06,
        "earnings_propagation": 0.10,
        "close_reversal": 0.08,
        "cross_asset_regime": 0.10,
        "institutional_flow": 0.08,
        "order_book_imbalance": 0.06,
        "supply_chain": 0.04,
    }

    REGIME_OVERRIDES = {
        "calm": {
            "mean_revert": 0.25,
            "trend": 0.10,
            "momentum": 0.08,
            "volume": 0.06,
            "institutional_flow": 0.12,
            "cross_asset_regime": 0.04,
        },
        "normal": {},
        "stressed": {
            "momentum": 0.28,
            "mean_revert": 0.02,
            "cross_asset_regime": 0.18,
            "trend": 0.12,
            "institutional_flow": 0.12,
            "volume": 0.10,
        },
        "crisis": {
            "cross_asset_regime": 0.35,
            "momentum": 0.20,
            "trend": 0.15,
            "institutional_flow": 0.15,
            "everything_else": 0.15,
        },
    }

    def get_weights(self, regime: str) -> Dict[str, float]:
        """
        Returns normalized weights for given regime.
        """
        normalized_regime = str(regime or "normal").strip().lower()
        if normalized_regime not in self.REGIME_OVERRIDES:
            normalized_regime = "normal"

        if normalized_regime == "normal":
            return self._normalize(self.BASE_WEIGHTS.copy())

        overrides = self.REGIME_OVERRIDES[normalized_regime]
        weights = self.BASE_WEIGHTS.copy()

        for factor, weight in overrides.items():
            if factor != "everything_else":
                weights[factor] = float(weight)

        if "everything_else" in overrides:
            catch_all = float(overrides["everything_else"])
            remaining = [factor for factor in weights if factor not in overrides]
            if remaining:
                per_factor = catch_all / len(remaining)
                for factor in remaining:
                    weights[factor] = per_factor

        return self._normalize(weights)

    def _normalize(self, weights: Dict[str, float]) -> Dict[str, float]:
        total = sum(float(value) for value in weights.values())
        if total <= 0:
            return self.BASE_WEIGHTS.copy()
        return {factor: float(value) / total for factor, value in weights.items()}
