#!/usr/bin/env python3
"""
Trade Confidence Score
=======================
Implements the bias-toward-execution framework from the upgrade spec.

Core idea: every feature is OPTIONAL at inference. Missing features reduce
the composite score rather than blocking the trade. Only two absolute hard
blocks exist system-wide:
    1. Available margin == 0
    2. Market closed (outside exchange hours)

Everything else is a soft gate with a score penalty.

Penalty ladder (cumulative, additive):
    residual_momentum filled w/ fallback     -> -0.05
    finbert_sentiment filled w/ neutral      -> -0.05
    options GEX/Vanna/Charm unavailable      -> -0.08
    stale websocket data used                -> -0.10
    feature_completeness_ratio < 0.6         -> -0.10
    sector features unavailable (India)      -> -0.05

Threshold -> size multiplier:
    final_score >= 0.65  -> 1.00 x intended size
    final_score >= 0.50  -> 0.70 x
    final_score >= 0.38  -> 0.40 x
    final_score <  0.38  -> skip

Vanna IV-falling rally bias (per spec): if net_vanna > threshold and IV falling,
add +0.05 to confidence score.

Standalone test:
    python -m project.core.confidence_score --demo
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration (env-tunable, sensible defaults)
# -----------------------------------------------------------------------------

PENALTY_RESIDUAL_FALLBACK = float(os.getenv("CONF_PENALTY_RESIDUAL", "0.05"))
PENALTY_FINBERT_NEUTRAL = float(os.getenv("CONF_PENALTY_FINBERT", "0.05"))
PENALTY_OPTIONS_MISSING = float(os.getenv("CONF_PENALTY_OPTIONS", "0.08"))
PENALTY_STALE_WS = float(os.getenv("CONF_PENALTY_STALE_WS", "0.10"))
PENALTY_LOW_COMPLETENESS = float(os.getenv("CONF_PENALTY_LOW_COMPLETENESS", "0.10"))
PENALTY_SECTOR_MISSING = float(os.getenv("CONF_PENALTY_SECTOR_MISSING", "0.05"))
COMPLETENESS_THRESHOLD = float(os.getenv("CONF_COMPLETENESS_THRESHOLD", "0.60"))

VANNA_RALLY_BONUS = float(os.getenv("CONF_VANNA_RALLY_BONUS", "0.05"))
VANNA_THRESHOLD = float(os.getenv("CONF_VANNA_THRESHOLD", "1e6"))

FULL_SIZE_THRESHOLD = float(os.getenv("CONF_FULL_SIZE_THRESHOLD", "0.65"))
SEVENTY_SIZE_THRESHOLD = float(os.getenv("CONF_70_SIZE_THRESHOLD", "0.50"))
FORTY_SIZE_THRESHOLD = float(os.getenv("CONF_40_SIZE_THRESHOLD", "0.38"))

PIN_ZONE_DISTANCE = float(os.getenv("CONF_PIN_ZONE_DISTANCE", "0.002"))
PIN_ZONE_SIZE_MULTIPLIER = float(os.getenv("CONF_PIN_SIZE_MULTIPLIER", "0.5"))


@dataclass
class ConfidenceInputs:
    """Inputs to trade_confidence_score()."""
    meta_probability: float
    residual_momentum_fallback: bool = False
    finbert_neutral: bool = False
    options_unavailable: bool = False
    stale_data: bool = False
    feature_completeness_ratio: float = 1.0
    sector_unavailable: bool = False

    # Optional bonuses
    net_vanna: float = 0.0
    iv_falling: bool = False
    distance_to_pin: float = 1.0  # absolute fraction; |0.01| far, |0.002| near

    # Context (not penalised, passed through for logging)
    symbol: str = ""
    lane: str = ""


@dataclass
class ConfidenceResult:
    final_score: float
    base_score: float
    penalties_applied: Dict[str, float]
    bonuses_applied: Dict[str, float]
    size_multiplier: float
    should_execute: bool
    reason: str
    symbol: str = ""
    lane: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def trade_confidence_score(inputs: ConfidenceInputs) -> ConfidenceResult:
    """
    Compute confidence & size multiplier. Pure function.

    Never raises; clamps score into [0, 1].
    """
    base = float(inputs.meta_probability or 0.0)
    if not math.isfinite(base):
        base = 0.0
    base = max(0.0, min(1.0, base))

    score = base
    penalties: Dict[str, float] = {}
    bonuses: Dict[str, float] = {}

    if inputs.residual_momentum_fallback:
        score -= PENALTY_RESIDUAL_FALLBACK
        penalties["residual_fallback"] = PENALTY_RESIDUAL_FALLBACK
    if inputs.finbert_neutral:
        score -= PENALTY_FINBERT_NEUTRAL
        penalties["finbert_neutral"] = PENALTY_FINBERT_NEUTRAL
    if inputs.options_unavailable:
        score -= PENALTY_OPTIONS_MISSING
        penalties["options_missing"] = PENALTY_OPTIONS_MISSING
    if inputs.stale_data:
        score -= PENALTY_STALE_WS
        penalties["stale_ws_data"] = PENALTY_STALE_WS
    try:
        ratio = float(inputs.feature_completeness_ratio)
    except (TypeError, ValueError):
        ratio = 1.0
    if not math.isfinite(ratio):
        ratio = 1.0
    ratio = max(0.0, min(1.0, ratio))
    if ratio < COMPLETENESS_THRESHOLD:
        score -= PENALTY_LOW_COMPLETENESS
        penalties["low_completeness"] = PENALTY_LOW_COMPLETENESS
    if inputs.sector_unavailable and (inputs.lane or "").lower() in ("day", "india", "day_india", "nse"):
        score -= PENALTY_SECTOR_MISSING
        penalties["sector_missing"] = PENALTY_SECTOR_MISSING

    # Vanna + IV-falling rally bonus (only when options data present)
    if (
        not inputs.options_unavailable
        and inputs.iv_falling
        and float(inputs.net_vanna or 0.0) > VANNA_THRESHOLD
    ):
        score += VANNA_RALLY_BONUS
        bonuses["vanna_rally"] = VANNA_RALLY_BONUS

    final = max(0.0, min(1.0, score))

    # Size multiplier from threshold ladder
    if final >= FULL_SIZE_THRESHOLD:
        size_mult = 1.0
        tier = "full"
    elif final >= SEVENTY_SIZE_THRESHOLD:
        size_mult = 0.70
        tier = "0.70x"
    elif final >= FORTY_SIZE_THRESHOLD:
        size_mult = 0.40
        tier = "0.40x"
    else:
        size_mult = 0.0
        tier = "skip"

    # Pin zone dampening: if distance_to_pin is tiny and options data is live,
    # halve the size (chop expected). Still execute.
    if (
        not inputs.options_unavailable
        and abs(float(inputs.distance_to_pin or 0.0)) < PIN_ZONE_DISTANCE
        and size_mult > 0.0
    ):
        size_mult *= PIN_ZONE_SIZE_MULTIPLIER
        tier = f"{tier}+pin_zone"
        bonuses["pin_zone_damp"] = -1.0 * (1 - PIN_ZONE_SIZE_MULTIPLIER)

    should_execute = size_mult > 0.0
    reason = f"tier={tier} base={base:.3f} final={final:.3f}"

    result = ConfidenceResult(
        final_score=final,
        base_score=base,
        penalties_applied=penalties,
        bonuses_applied=bonuses,
        size_multiplier=size_mult,
        should_execute=should_execute,
        reason=reason,
        symbol=inputs.symbol,
        lane=inputs.lane,
    )
    log_line = (
        f"[confidence] symbol={inputs.symbol or '-'} lane={inputs.lane or '-'} "
        f"base={base:.3f} final={final:.3f} size_mult={size_mult:.2f} "
        f"penalties={penalties} bonuses={bonuses} execute={should_execute}"
    )
    logger.info(log_line)
    return result


def compute_feature_completeness(
    available_features: Dict[str, Any],
    expected_features: Optional[list] = None,
) -> float:
    """
    Ratio of expected features that are present and non-null/non-NaN in
    `available_features`. If `expected_features` is None, treats every non-null
    value in `available_features` as present (and completeness=1.0 if dict
    non-empty).
    """
    if expected_features:
        total = len(expected_features)
        if total == 0:
            return 1.0
        present = 0
        for key in expected_features:
            val = available_features.get(key)
            if val is None:
                continue
            try:
                fval = float(val)
                if math.isnan(fval) or math.isinf(fval):
                    continue
            except (TypeError, ValueError):
                # Non-numeric values (e.g. regime strings) count as present
                pass
            present += 1
        return present / total if total else 1.0
    return 1.0 if available_features else 0.0


# -----------------------------------------------------------------------------
# __main__ demo
# -----------------------------------------------------------------------------


def _demo() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cases = [
        ("all clean, high prob",        ConfidenceInputs(meta_probability=0.78, symbol="AAPL", lane="normal")),
        ("all clean, medium prob",      ConfidenceInputs(meta_probability=0.55, symbol="AAPL", lane="normal")),
        ("missing sentiment",           ConfidenceInputs(meta_probability=0.62, finbert_neutral=True, symbol="AAPL", lane="normal")),
        ("missing options+residual",    ConfidenceInputs(meta_probability=0.62, options_unavailable=True, residual_momentum_fallback=True, symbol="AAPL", lane="normal")),
        ("stale data",                  ConfidenceInputs(meta_probability=0.66, stale_data=True, symbol="AAPL", lane="crypto")),
        ("low completeness",            ConfidenceInputs(meta_probability=0.60, feature_completeness_ratio=0.45, symbol="AAPL", lane="normal")),
        ("india + sector missing",      ConfidenceInputs(meta_probability=0.55, sector_unavailable=True, symbol="RELIANCE", lane="day")),
        ("vanna + IV falling bonus",    ConfidenceInputs(meta_probability=0.60, net_vanna=2e6, iv_falling=True, symbol="SPY", lane="normal")),
        ("pin zone dampening",          ConfidenceInputs(meta_probability=0.80, distance_to_pin=0.0005, symbol="SPY", lane="normal")),
        ("too low -> skip",             ConfidenceInputs(meta_probability=0.40, options_unavailable=True, finbert_neutral=True, residual_momentum_fallback=True, stale_data=True, symbol="X", lane="crypto")),
    ]
    for name, inp in cases:
        res = trade_confidence_score(inp)
        print(f"\n=== {name} ===")
        print(json.dumps(res.to_dict(), indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()
    _demo()
