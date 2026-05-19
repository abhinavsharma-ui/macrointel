from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REPORT_PATH = Path(os.getenv("FACTOR_OVERLAY_REPORT", "reports/factor_overlay_report.json"))


def _num(value, default=0.0) -> float:
    try:
        if value is None or value == "":
            return default
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def _rank_percentile(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if not higher_is_better:
        values = -values
    if values.notna().sum() <= 1:
        return pd.Series(0.5, index=series.index)
    return values.rank(pct=True, method="average").fillna(0.5).clip(0.0, 1.0)


def apply_factor_overlay(live: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Add conservative factor context without replacing the ML model.

    The overlay is intentionally small. It is a same-day tie-breaker around the
    ML probability, not a second model that can overpower the production score.
    """
    out = live.copy()
    enabled = os.getenv("SIG_FACTOR_OVERLAY_ENABLED", "1") != "0"
    weight = max(0.0, min(0.15, _num(os.getenv("SIG_FACTOR_OVERLAY_WEIGHT", "0.03"), 0.03)))
    tie_band = max(0.0, min(0.03, _num(os.getenv("SIG_FACTOR_TIE_BAND", "0.012"), 0.012)))

    if out.empty or "probability" not in out.columns:
        return out, {"enabled": enabled, "status": "empty_or_missing_probability"}

    if not enabled or weight <= 0:
        out["factor_composite"] = 0.5
        out["factor_rank_score"] = out["probability"]
        return out, {"enabled": enabled, "weight": weight, "status": "disabled_or_zero_weight"}

    momentum_cols = [c for c in ("momentum_60d", "momentum_20d", "return_20d", "roc_20") if c in out.columns]
    if momentum_cols:
        momentum_raw = sum(pd.to_numeric(out[c], errors="coerce").fillna(0.0) for c in momentum_cols) / len(momentum_cols)
    else:
        momentum_raw = pd.Series(0.0, index=out.index)

    vol_cols = [c for c in ("realized_vol_21d", "hist_vol_30", "atr_pct", "bb_width") if c in out.columns]
    if vol_cols:
        vol_raw = sum(pd.to_numeric(out[c], errors="coerce").fillna(np.nan) for c in vol_cols) / len(vol_cols)
    else:
        vol_raw = pd.Series(np.nan, index=out.index)

    quality_cols = [c for c in ("compound_score", "weighted_compound_score", "source_quality_score", "earnings_tone_signal") if c in out.columns]
    if quality_cols:
        quality_raw = sum(pd.to_numeric(out[c], errors="coerce").fillna(0.0) for c in quality_cols) / len(quality_cols)
    else:
        quality_raw = pd.Series(0.0, index=out.index)

    value_cols = [c for c in ("52w_low_ratio", "zscore_vs_60d", "bb_position") if c in out.columns]
    if value_cols:
        value_raw = -sum(pd.to_numeric(out[c], errors="coerce").fillna(0.0) for c in value_cols) / len(value_cols)
    else:
        value_raw = pd.Series(0.0, index=out.index)

    out["factor_momentum"] = _rank_percentile(momentum_raw, higher_is_better=True)
    out["factor_low_vol"] = _rank_percentile(vol_raw, higher_is_better=False)
    out["factor_quality"] = _rank_percentile(quality_raw, higher_is_better=True)
    out["factor_value"] = _rank_percentile(value_raw, higher_is_better=True)
    out["factor_composite"] = (
        0.35 * out["factor_momentum"]
        + 0.25 * out["factor_low_vol"]
        + 0.25 * out["factor_quality"]
        + 0.15 * out["factor_value"]
    ).clip(0.0, 1.0)

    adjustment = ((out["factor_composite"] - 0.5) * weight).clip(-tie_band / 2.0, tie_band / 2.0)
    out["factor_rank_score"] = pd.to_numeric(out["probability"], errors="coerce").fillna(0.0) + adjustment

    top = out.sort_values("factor_rank_score", ascending=False).head(20)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "enabled": True,
        "weight": weight,
        "tie_band": tie_band,
        "rows": int(len(out)),
        "top": [
            {
                "symbol": str(row.get("symbol", "")),
                "probability": round(_num(row.get("probability")), 6),
                "factor_composite": round(_num(row.get("factor_composite"), 0.5), 4),
                "factor_rank_score": round(_num(row.get("factor_rank_score")), 6),
            }
            for _, row in top.iterrows()
        ],
        "status": "ok",
    }
    try:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except Exception as exc:
        report["write_error"] = str(exc)[:180]
    return out, report
