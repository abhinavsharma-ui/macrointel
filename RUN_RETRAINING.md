# Running the Improved Model Retraining

All 12 code fixes have been applied to your project. The changes are in:
- `project/models/xgboost_model.py`
- `project/models/institutional_retraining.py`
- `project/models/train_orchestrator.py`

## Quick Start

**Option 1: Full Institutional Retraining (Recommended)**
```bash
cd /path/to/macro_intelligence_complete/project
python retrain_institutional_models.py
```

This will:
1. Load feature matrices from `data/features/*.parquet`
2. Retrain XGBoost classifier with fixed directional accuracy metric
3. Build cleaner 3-class labels (dominance_buffer=0.60)
4. Train meta-model (take/skip classifier + regressors) with improved params
5. Generate walk-forward evaluation report

**Expected Duration:** 10-30 minutes depending on data size

**Option 2: Quick Test with Dummy Data**
If you want to test the fixes without full training data:
```python
import pandas as pd
import numpy as np
from models.xgboost_model import XGBoostSignalModel

# Create dummy data
X = pd.DataFrame(np.random.randn(1000, 50), columns=[f'feat_{i}' for i in range(50)])
y = pd.Series(np.random.choice([0, 1, 2], 1000))

# Test the fixed model
model = XGBoostSignalModel()
metrics = model.fit(X, y)
print(f"Train acc: {metrics['train_accuracy']:.3f}")

# Run CV with fixed directional accuracy
cv_results = model.time_series_cv_score(X, y, n_splits=3)
print(f"Mean CV acc: {cv_results['mean_accuracy']:.3f}")
print(f"Mean dir_acc: {cv_results['mean_directional_accuracy']:.3f}")
```

## What Changed

### XGBoost Model (`xgboost_model.py`)
- ✓ Balanced class weights now applied to training (was missing for multiclass)
- ✓ Directional accuracy metric now CORRECT (was inflated before)
- ✓ n_estimators in CV increased from 300 → 600 (more representative metrics)
- ✓ Removed scale_pos_weight (binary-only param, was doing nothing)

### Meta Model (`institutional_retraining.py`)
- ✓ Triple-barrier label dominance_buffer 0.20% → 0.60% (cleaner signals)
- ✓ Edge/drawdown regressors use 3-fold CV MAE (honest evaluation)
- ✓ Regressor regularization improved (better generalization)
- ✓ Added 7 new interaction features for edge/drawdown prediction
- ✓ MAE limits now adaptive (85% of null-model baseline)

### Label Building (`train_orchestrator.py`)
- ✓ Minimum threshold floor (±0.8%) prevents hold band collapse in low-vol

## Expected Results

| Metric | Before | After | Notes |
|--------|--------|-------|-------|
| Train Acc | 0.427 | ~0.45-0.48 | Better label signal + class weighting |
| CV Acc (3-class) | 0.376 | ~0.41-0.44 | More consistent folds |
| Dir Acc | ~0.556 | ~0.45-0.50 | **LOWER is CORRECT** — old metric was inflated |
| Edge MAE | 6.31 | ~3.5-4.5 | CV evaluation reveals true generalization |
| Draw MAE | 5.06 | ~2.5-3.5 | Improved regressor + better features |

**Important:** The directional accuracy will be LOWER. This is not a regression — the old metric was misleading. The new metric honestly measures how often the model predicts the correct direction (buy/sell/hold) on actual directional events. The decline proves the old metric was only counting "correct" predictions when the model had already decided to go directional.

## Monitoring the Retraining

Watch for these in the logs:

**Good signs:**
```
XGBoost trained: acc=0.45, f1=0.35, features=52
Fold 1: acc=0.42 dir_acc=0.48
Fold 2: acc=0.44 dir_acc=0.51
Fold 3: acc=0.46 dir_acc=0.49
...
Regressor quality OK (long): edge_CV_MAE=3.2 (null 3.8, limit 3.23)
Regressor quality OK (short): draw_CV_MAE=2.1 (null 2.5, limit 2.12)
```

**Warning signs** (regressors not ready):
```
Regressor drift detected (long): edge_CV_MAE=6.5 (null 3.8, limit 3.23)
→ Falling back to precision-only gate
```

If regressors are unreliable, the system will still work (just uses precision gate instead of edge/drawdown filtering). This is by design.

## Troubleshooting

**"ModuleNotFoundError: No module named 'joblib'"**
- Install: `pip install joblib scikit-learn xgboost optuna`

**"No feature matrices available in data/features/"**
- Ensure your feature parquets are in `project/data/features/`
- Check format: they should be indexed by symbol and have the required columns from MetaFeatureBuilder

**"Regressor drift detected" — regressors keep failing**
- This is normal if features are weak for predicting edge/drawdown
- The system falls back to precision-gating (safe mode)
- Consider tuning `META_MODEL_SKILL_THRESHOLD` env var (default 0.85):
  ```bash
  export META_MODEL_SKILL_THRESHOLD=0.80  # More lenient
  python retrain_institutional_models.py
  ```

## Environment Variables

Useful tuning parameters (set before running):
```bash
# Meta model training
export META_MODEL_HORIZON_DAYS=5
export META_MODEL_WALKFORWARD_FOLDS=4
export META_MODEL_MIN_TRAIN_DAYS=180

# Regressor reliability (new adaptive system)
export META_MODEL_SKILL_THRESHOLD=0.85  # Model must be 15% better than baseline

# Label building
export XGB_MIN_FRAME_ROWS=100

# Optional: override adaptive MAE limits (if you want fixed limits back)
# export META_MODEL_EDGE_MAE_LIMIT=2.5
# export META_MODEL_DRAW_MAE_LIMIT=3.0
```

## Next Steps After Retraining

1. Check the generated reports:
   - `project/models/checkpoints/xgboost_retrain_report.json` — XGBoost CV metrics
   - `project/models/checkpoints/meta_walkforward_report.json` — Meta model metrics

2. Compare metrics to baseline (saved in FIXES_APPLIED.txt)

3. If satisfied, deploy the new checkpoints:
   - `xgboost_model.json`
   - `meta_take_model.joblib`
   - `meta_edge_model.joblib`
   - `meta_drawdown_model.joblib`

## Support

All changes follow the comprehensive prompt in `model_improvement_prompt.md`.
Each fix is documented with the "why" and "how" in that file for reference.
