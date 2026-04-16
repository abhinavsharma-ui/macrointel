# Fix: Meta Model Quality Guard — Prevent Bad Retraining from Overwriting Good Checkpoints

## Problem

Every restart triggers `_maybe_auto_retrain()` which runs `retrain_institutional_models.py`.
When retraining produces a bad meta model (0% edge, 0% hit, 0.00 precision), it unconditionally
overwrites the previous good checkpoint. On the next restart the dashboard shows
`trained meta - 0.00% edge - 0.0% hit - 0.00 precision`.

The `.previous` backup files exist but are themselves overwritten on the next retrain cycle,
so the good checkpoint is eventually lost entirely.

## Root Cause

In `MetaModelTrainer.train()` (project/models/institutional_retraining.py, lines ~1508–1562),
the new model is saved unconditionally — there is no comparison against the existing checkpoint
to verify the new model is actually better before overwriting.

---

## FILE — `project/models/institutional_retraining.py`

### Fix (HIGH): Add quality guard before saving new meta model checkpoint

Find the block that saves the new checkpoint (lines ~1508–1562). It currently looks like:

```python
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
            { ... },
            META_DIRECTIONAL_PATH,
        )
        ...
        report = { ... "walk_forward": walk_forward, ... }
        META_REPORT_PATH.write_text(json.dumps(report, indent=2))
```

Replace the entire save block (from `if META_DIRECTIONAL_PATH.exists()` through
`META_REPORT_PATH.write_text(...)`) with the following, which:
1. Builds the report dict first
2. Compares it against the existing checkpoint using `_report_is_acceptable` + a score
3. Only overwrites if the new model is at least as good as the existing one

```python
        # --- Build the report first so we can quality-gate it before saving ---
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

        # --- Quality gate: only save if new model passes AND beats the existing one ---
        def _report_score(r: dict) -> float:
            """Single scalar to compare two reports. Higher = better."""
            summary = (r.get("walk_forward") or {}).get("summary") or {}
            precision = float(summary.get("mean_precision") or 0.0)
            edge = float(summary.get("mean_taken_edge_pct") or 0.0)
            hit = float(summary.get("mean_taken_hit_rate_pct") or 0.0)
            return precision * 0.5 + (edge / 2.0) * 0.3 + (hit / 100.0) * 0.2

        new_passes = self._report_is_acceptable(report)
        new_score = _report_score(report)

        existing_score = 0.0
        if META_REPORT_PATH.exists():
            try:
                existing_report = json.loads(META_REPORT_PATH.read_text(encoding="utf-8"))
                existing_score = _report_score(existing_report)
            except Exception:
                pass  # Unreadable existing report — allow overwrite

        if not new_passes:
            logger.warning(
                f"Meta model quality gate FAILED — new model does not meet minimum thresholds "
                f"(score={new_score:.4f}). Keeping existing checkpoint (score={existing_score:.4f}). "
                f"New report NOT saved."
            )
            return report  # Return the report for logging but do not persist

        if new_score < existing_score * 0.95:
            # Allow up to 5% regression — anything worse keeps the old model
            logger.warning(
                f"Meta model quality gate REGRESSION — new score {new_score:.4f} is worse than "
                f"existing {existing_score:.4f}. Keeping existing checkpoint. New report NOT saved."
            )
            return report

        # New model passes and is at least as good — safe to save
        logger.info(
            f"Meta model quality gate PASSED — new score {new_score:.4f} vs existing {existing_score:.4f}. "
            f"Saving new checkpoint."
        )

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
                        "decision_threshold": float(bundle.decision_threshold),
                        "min_expected_edge_pct": float(bundle.min_expected_edge_pct),
                        "min_edge_ratio": float(bundle.min_edge_ratio),
                        "regressors_reliable": getattr(bundle, "regressors_reliable", True),
                    }
                    for direction, bundle in bundles.items()
                },
                "feature_columns": MetaFeatureBuilder.FEATURE_COLUMNS,
            },
            META_DIRECTIONAL_PATH,
        )
        with open(META_FEATURES_PATH, "wb") as f:
            pickle.dump(MetaFeatureBuilder.FEATURE_COLUMNS, f)

        META_REPORT_PATH.write_text(json.dumps(report, indent=2))
```

Also remove the original `report = { ... }` and `META_REPORT_PATH.write_text(...)` lines
that follow the existing save block — they are now replaced by the block above.

---

## Verification

After applying:

1. Confirm the quality gate log message appears on the next retrain:
   ```
   grep "quality gate" project/logs/system.jsonl | tail -5
   ```

2. Confirm a bad retrain no longer overwrites a good checkpoint:
   - Temporarily set `META_MODEL_GATE_MIN_PRECISION=0.99` in `.env` (impossibly high)
   - Restart the system and wait for auto-retrain
   - Dashboard should still show the old good stats
   - Log should show: `Meta model quality gate FAILED`
   - Reset `META_MODEL_GATE_MIN_PRECISION` to blank after testing

3. Confirm the existing good checkpoint (76.2% precision, 1.46% edge) is preserved
   across restarts:
   ```
   cat project/models/checkpoints/meta_walkforward_report.json | python3 -m json.tool | grep mean_taken_edge
   ```
   Should show `1.46`, not `0.0`.
