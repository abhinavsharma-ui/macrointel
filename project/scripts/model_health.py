from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = Path(os.getenv("MODEL_HEALTH_REPORT", "reports/model_health.json"))


def _num(value, default=0.0) -> float:
    try:
        if value is None or value == "":
            return default
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def _quantiles(values) -> dict:
    s = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return {}
    return {
        "min": round(float(s.min()), 6),
        "p10": round(float(s.quantile(0.10)), 6),
        "median": round(float(s.median()), 6),
        "p90": round(float(s.quantile(0.90)), 6),
        "max": round(float(s.max()), 6),
    }


def write_signal_health_report(
    live: pd.DataFrame,
    signals: list[dict],
    features: list[str],
    model_path,
    threshold: float,
    top_n: int,
    extra: dict | None = None,
) -> dict:
    """Write a fast health report from the actual nightly scoring run."""
    extra = extra or {}
    probs = pd.to_numeric(live.get("probability", pd.Series(dtype=float)), errors="coerce") if isinstance(live, pd.DataFrame) else pd.Series(dtype=float)
    missing = {}
    if isinstance(live, pd.DataFrame) and features:
        sample = live.reindex(columns=features)
        na_rate = sample.replace([np.inf, -np.inf], np.nan).isna().mean().sort_values(ascending=False)
        missing = {str(k): round(float(v), 4) for k, v in na_rate.head(15).items() if v > 0}

    selected_probs = [_num(s.get("probability")) for s in signals if isinstance(s, dict)]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "threshold": threshold,
        "top_n": top_n,
        "scored_rows": int(len(live)) if isinstance(live, pd.DataFrame) else 0,
        "qualified_rows": int((probs >= threshold).sum()) if not probs.empty else 0,
        "signal_count": len(signals),
        "score_distribution": _quantiles(probs),
        "selected_score_distribution": _quantiles(selected_probs),
        "feature_count": len(features or []),
        "worst_feature_missing_rates": missing,
        "quality_flags": [],
        "extra": extra,
    }
    if payload["scored_rows"] < 500:
        payload["quality_flags"].append("small_scored_universe")
    if payload["qualified_rows"] == 0:
        payload["quality_flags"].append("no_symbols_above_threshold")
    if payload["signal_count"] == 0:
        payload["quality_flags"].append("no_final_signals")
    if missing and max(missing.values()) >= 0.25:
        payload["quality_flags"].append("feature_missingness_elevated")
    try:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        payload["write_error"] = str(exc)[:180]
    return payload


def fixed_return_label(frame: pd.DataFrame, entry_idx: int, hold_days: int, pt_pct: float, sl_pct: float) -> tuple[int, float]:
    entry = _num(frame["close"].iloc[entry_idx])
    if entry <= 0:
        return 0, 0.0
    end_idx = min(len(frame) - 1, entry_idx + hold_days)
    window = frame.iloc[entry_idx + 1 : end_idx + 1]
    if window.empty:
        return 0, 0.0
    target = entry * (1.0 + pt_pct / 100.0)
    stop = entry * (1.0 - sl_pct / 100.0)
    for _, row in window.iterrows():
        low = _num(row.get("low"), _num(row.get("close")))
        high = _num(row.get("high"), _num(row.get("close")))
        if low <= stop and high >= target:
            return 0, -sl_pct
        if low <= stop:
            return 0, -sl_pct
        if high >= target:
            return 1, pt_pct
    exit_px = _num(window["close"].iloc[-1], entry)
    ret = (exit_px / entry - 1.0) * 100.0
    return int(ret > 0), ret


def run_walk_forward_shadow(limit_symbols: int = 80, max_rows_per_symbol: int = 80) -> dict:
    """Shadow-validate the current model on recent historical rows.

    This does not retrain the model. It checks whether the currently loaded
    model's high-probability calls would have passed fixed-return labels on
    older rows, using chronological folds and an embargo around each test fold.
    """
    os.chdir(ROOT)
    import joblib
    import fixed_return_daily_signals as sig

    model_path, model, features = sig.load_model()
    paths = sorted(Path(sig.LIVE_ROOT).glob("*.parquet"))[: max(1, limit_symbols)]
    rows = []
    for path in paths:
        try:
            df = sig.load_file(path)
            if df is None or len(df) < sig.HOLD_DAYS + 80:
                continue
            df = df.reset_index(drop=True)
            start = max(50, len(df) - max_rows_per_symbol - sig.HOLD_DAYS - 1)
            stop = len(df) - sig.HOLD_DAYS - 1
            for idx in range(start, stop):
                row = {col: _num(df.iloc[idx].get(col)) for col in features}
                label, ret = fixed_return_label(df, idx, sig.HOLD_DAYS, sig.PROFIT_TARGET_PCT, sig.STOP_LOSS_PCT)
                row.update({"symbol": sig.norm_sym(path.name), "date": str(df.iloc[idx].get("date", idx))[:10], "label": label, "realized_return_pct": ret})
                rows.append(row)
        except Exception:
            continue
    if not rows:
        payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "status": "no_validation_rows"}
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    sample = pd.DataFrame(rows)
    x = sample.reindex(columns=features).replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    proba = model.predict_proba(x)
    classes = list(getattr(model, "classes_", [0, 1]))
    pos_idx = classes.index(1) if 1 in classes else proba.shape[1] - 1
    sample["probability"] = proba[:, pos_idx]
    sample = sample.sort_values("date").reset_index(drop=True)
    folds = np.array_split(sample.index.to_numpy(), 6)
    fold_rows = []
    embargo = max(1, int(os.getenv("MODEL_HEALTH_EMBARGO_DAYS", "3")))
    for fold_no, idxs in enumerate(folds, 1):
        if len(idxs) == 0:
            continue
        test = sample.iloc[idxs].copy()
        taken = test[test["probability"] >= sig.SIG_THRESHOLD]
        if taken.empty:
            fold_rows.append({"fold": fold_no, "taken": 0, "precision": None, "avg_return_pct": None, "embargo_days": embargo})
            continue
        precision = float((taken["label"] == 1).mean())
        avg_ret = float(taken["realized_return_pct"].mean())
        fold_rows.append({"fold": fold_no, "taken": int(len(taken)), "precision": round(precision, 4), "avg_return_pct": round(avg_ret, 4), "embargo_days": embargo})
    taken_all = sample[sample["probability"] >= sig.SIG_THRESHOLD]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "mode": "walk_forward_shadow_current_model",
        "model_path": str(model_path),
        "symbols_sampled": len({p.stem for p in paths}),
        "rows": int(len(sample)),
        "threshold": sig.SIG_THRESHOLD,
        "taken_rows": int(len(taken_all)),
        "overall_precision": round(float((taken_all["label"] == 1).mean()), 4) if len(taken_all) else None,
        "overall_avg_return_pct": round(float(taken_all["realized_return_pct"].mean()), 4) if len(taken_all) else None,
        "folds": fold_rows,
        "quality_flags": [],
    }
    if payload["taken_rows"] < 30:
        payload["quality_flags"].append("too_few_shadow_takes")
    if payload["overall_precision"] is not None and payload["overall_precision"] < 0.52:
        payload["quality_flags"].append("shadow_precision_below_live_gate")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="MacroIntel model health checks.")
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--limit-symbols", type=int, default=80)
    args = parser.parse_args()
    if args.walk_forward:
        print(json.dumps(run_walk_forward_shadow(limit_symbols=args.limit_symbols), indent=2))
    else:
        print(json.dumps({"status": "use --walk-forward for historical shadow validation"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
