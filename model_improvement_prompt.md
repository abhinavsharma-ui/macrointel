# Model Improvement Prompt
## Raise XGBoost accuracy, directional accuracy, and balance edge/draw MAE

You are working on a production quant signal system. The primary model is a 3-class XGBoost
classifier (`sell=0, hold=1, buy=2`) trained via triple-barrier labels. A separate meta-model
stack (binary take/skip classifier + edge regressor + drawdown regressor) gates live signals.

Walk-forward CV across 6 folds on fold 6/6 shows:
- `edge_MAE=6.311` against a limit of 2.5 → regressor fallback triggered
- `draw_MAE=5.061` against a limit of 3.0 → regressor fallback triggered
- XGBoost fold accuracies: 0.384 / 0.357 / 0.366 / 0.394 (barely above 33.3% random)
- XGBoost directional accuracies: 0.569 / 0.512 / 0.512 / 0.568 (folds 2–3 near coin-flip)

Apply all fixes below in priority order. Do not change any other logic.

---

## FILE 1 — `project/models/xgboost_model.py`

### Fix 1 (HIGH): Remove duplicate dead `ENHANCED_PARAMS` block

The class `XGBoostSignalModel` defines `ENHANCED_PARAMS` twice (lines ~409 and ~432).
The first definition is completely overwritten. Delete the FIRST block (lines ~409–430)
so only one `ENHANCED_PARAMS` dict exists. No other change.

### Fix 2 (HIGH): Remove `scale_pos_weight` from multiclass params

`scale_pos_weight` is a binary-only XGBoost parameter and is silently ignored in
`multi:softprob`. It creates a false sense that class imbalance is handled. Remove it
from BOTH `DEFAULT_PARAMS` and `ENHANCED_PARAMS`.

### Fix 3 (HIGH): Add proper class-balanced sample weights for multiclass

In `XGBoostSignalModel.fit()`, after feature selection and before calling
`_fit_xgb_classifier`, compute per-sample class weights using sklearn and pass them
to the fit call:

```python
from sklearn.utils.class_weight import compute_sample_weight
sample_weight = compute_sample_weight(class_weight="balanced", y=y)
```

Pass `sample_weight=sample_weight` as an additional argument to `_fit_xgb_classifier`.
Update `_fit_xgb_classifier` to accept and forward an optional `sample_weight` kwarg
to `model.fit(...)`.

Do the same in `time_series_cv_score` for each fold's `fold_model.fit(...)` call —
compute `compute_sample_weight("balanced", y_train)` and pass it.

### Fix 4 (HIGH): Fix the directional accuracy metric — it is mathematically wrong

Current code (in `time_series_cv_score` and `XGBoostOptimizer.objective`):
```python
directional_mask = (y_test != 1) & (y_pred != 1)
```
This only measures accuracy on samples where BOTH the true label AND the predicted label
are non-hold. This inflates dir_acc by silently excluding all false-hold predictions
(where the model said "hold" but truth was "buy" or "sell"). It is not a measure of
directional correctness — it is a measure of directional correctness conditional on the
model having already made a directional bet.

Fix: the mask should only filter on the TRUE label:
```python
directional_mask = (y_test != 1)   # true directional events only
dir_acc = (
    accuracy_score(y_test[directional_mask], y_pred[directional_mask])
    if directional_mask.any() else 0.0
)
```
Apply this fix in BOTH `time_series_cv_score` AND inside `XGBoostOptimizer.objective`.
This makes dir_acc a true recall-weighted directional accuracy that penalises the model
for predicting "hold" on real buy/sell events.

### Fix 5 (MEDIUM): Increase n_estimators in CV fold models

In `time_series_cv_score`, each fold model is initialised with `n_estimators=300`.
The full model uses 1500. This inconsistency makes CV acc/dir_acc pessimistic.
Raise to `n_estimators=600` so CV metrics are more representative of the full model.

### Fix 6 (MEDIUM): Add `focal_loss` weighting alternative in Optuna search space

In `XGBoostOptimizer.objective`, add a trial parameter:
```python
use_focal = trial.suggest_categorical("use_focal_weights", [True, False])
```
If `use_focal=True`, compute focal sample weights before fitting:
```python
# Focal weighting: upweight hard examples (low confidence in last fold pred)
probs = np.clip(prev_proba, 1e-6, 1.0 - 1e-6)  # from previous fold if available
focal_w = (1.0 - probs.max(axis=1)) ** 2
sample_weight = focal_w / focal_w.mean()
```
If unavailable (first fold), fall back to balanced class weights.
This gives Optuna the ability to discover whether focal weighting improves dir_acc.

### Fix 7 (MEDIUM): Increase triple-barrier dominance buffer to reduce noisy labels

In `institutional_retraining.py`, function `_triple_barrier_direction_label` (line ~192):
```python
dominance_buffer = 0.20
```
This means long and short edges only need to differ by 0.20% for a directional label to
be assigned. At this level, the label is unreliable noise. Raise to:
```python
dominance_buffer = 0.60
```
This will reduce the proportion of buy/sell labels and increase hold, but the remaining
directional labels will be cleaner and more learnable. Expect class imbalance to increase
(more holds) — Fix 3 above compensates for this.

---

## FILE 2 — `project/models/institutional_retraining.py`

### Fix 8 (HIGH): Fix in-sample MAE evaluation for edge/drawdown regressors

In `MetaModelTrainer._fit_direction_bundle`, the regressors are evaluated on the same data
they were trained on (lines ~1114–1117):
```python
edge_pred_train = edge_model.predict(X)   # X = full training set
draw_pred_train = drawdown_model.predict(X)
edge_mae = float(np.mean(np.abs(edge_pred_train - y_edge.to_numpy())))
draw_mae = float(np.mean(np.abs(draw_pred_train - y_drawdown.to_numpy())))
```
In-sample MAE for a fitted regressor will always be lower than null_mae because the model
memorises training data. Yet the logs show edge_MAE=6.311 against null_mae=2.5, meaning
the model can't even fit its own training data — severe underfitting.

Fix in two parts:

**Part A — use out-of-fold MAE instead of in-sample MAE:**
Replace the in-sample evaluation with a 3-fold cross-validated MAE:
```python
from sklearn.model_selection import cross_val_score
edge_cv_mae = -cross_val_score(
    HistGradientBoostingRegressor(
        learning_rate=0.045, max_depth=6, max_iter=480,
        min_samples_leaf=18, l2_regularization=0.09, random_state=42
    ),
    X, y_edge, cv=3, scoring="neg_mean_absolute_error"
).mean()
draw_cv_mae = -cross_val_score(
    HistGradientBoostingRegressor(
        learning_rate=0.045, max_depth=6, max_iter=480,
        min_samples_leaf=18, l2_regularization=0.09, random_state=42,
        loss="absolute_error"
    ),
    X, y_drawdown, cv=3, scoring="neg_mean_absolute_error"
).mean()
```
Use `edge_cv_mae` and `draw_cv_mae` for the reliability check instead of in-sample.
Then fit the final edge_model and drawdown_model on all of X as before.

**Part B — add interaction features to MetaFeatureBuilder.FEATURE_COLUMNS:**
The regressor has only 65 features. Add the following derived features which carry direct
causal signal for actual edge size and drawdown depth. Compute them in
`MetaFeatureBuilder.build()` and append to the returned dict, and add the column names
to `FEATURE_COLUMNS`:

```python
# Volatility-adjusted risk parameters
"stop_x_vol_ratio":     risk.get("stop_loss_pct", 2.0) * vol_ratio_20,
"tp_x_momentum":        risk.get("take_profit_pct", 4.0) * abs(momentum_20d),
"rrr_x_confidence":     risk.get("risk_reward_ratio", 2.0) * confidence,
# Momentum × sentiment alignment
"momentum_sentiment_align": np.sign(momentum_20d) * sentiment_zscore,
# Conviction under stress
"conviction_under_stress":  conviction_score * vol_regime_ratio,
# Event strength × horizon multiplier
"event_x_horizon":      event_move_strength * adaptive_horizon_multiplier,
# Barrier tightness relative to recent vol
"barrier_vol_ratio":    stop_loss_pct / max(realized_vol_21d * 100, 0.1),
```

Where `vol_ratio_20`, `momentum_20d`, `sentiment_zscore`, etc. are read from
`feature_row` in the `build()` method (they already exist in ROW_COLUMNS).

### Fix 9 (HIGH): Regularise the edge/drawdown regressors to reduce underfitting

Replace the current `HistGradientBoostingRegressor` instantiation for `edge_model` with:
```python
edge_model = HistGradientBoostingRegressor(
    learning_rate=0.03,       # slower learning, better generalisation
    max_depth=4,              # shallower to prevent overfitting to training noise
    max_iter=700,             # more iterations to compensate for lower lr
    min_samples_leaf=25,      # larger leaves = more regularisation
    l2_regularization=0.15,   # stronger regularisation
    max_bins=128,             # reduce bin resolution to smooth predictions
    random_state=42,
)
```
And for `drawdown_model`:
```python
drawdown_model = HistGradientBoostingRegressor(
    learning_rate=0.03,
    max_depth=4,
    max_iter=700,
    min_samples_leaf=25,
    l2_regularization=0.15,
    max_bins=128,
    random_state=42,
    loss="absolute_error",
)
```

### Fix 10 (MEDIUM): Make the regressor reliability limits adaptive to data distribution

The env var `META_MODEL_EDGE_MAE_LIMIT` is currently set to 2.5 externally. Instead of a
fixed limit, use a fraction of the null-model MAE as the ceiling. Replace the limit
logic (lines ~1123–1129) with:

```python
env_edge_limit = float(os.getenv("META_MODEL_EDGE_MAE_LIMIT", "0"))
env_draw_limit = float(os.getenv("META_MODEL_DRAW_MAE_LIMIT", "0"))
# Default: model must beat null-model by at least 15% to be considered reliable
skill_threshold = float(os.getenv("META_MODEL_SKILL_THRESHOLD", "0.85"))
edge_mae_limit = env_edge_limit if env_edge_limit > 0 else null_edge_mae * skill_threshold
draw_mae_limit = env_draw_limit if env_draw_limit > 0 else null_draw_mae * skill_threshold
regressors_reliable = (edge_cv_mae <= edge_mae_limit and draw_cv_mae <= draw_mae_limit)
```

This means the regressors only need to be 15% better than predicting-the-mean, rather
than beating an arbitrary hardcoded constant. Adjust `META_MODEL_SKILL_THRESHOLD` env
var to tighten/loosen — 0.85 is a reasonable starting point.

### Fix 11 (LOW): Add per-fold regressor MAE logging in walk-forward

In `MetaModelTrainer._run_walk_forward` (the walk-forward loop), after each fold's
bundle is fitted, log the fold's edge_cv_mae and draw_cv_mae alongside the fold index.
This makes it easy to spot which specific folds are causing high MAE rather than
diagnosing from a single aggregate warning.

---

## FILE 3 — `project/models/train_orchestrator.py`

### Fix 12 (MEDIUM): Increase label binning threshold for cleaner 3-class labels

In `build_labels()` (line ~62):
```python
labels = pd.cut(fwd_ret, bins=[-np.inf, -threshold, threshold, np.inf], labels=[0, 1, 2])
```
`threshold` is derived from a rolling std. Add a minimum threshold floor so that in
low-vol regimes the hold band doesn't collapse to near-zero and create a nearly 50/50
buy/sell split with almost no holds:
```python
threshold = max(threshold, 0.008)   # minimum ±0.8% to be considered directional
```

---

## Verification steps after applying all changes

1. Run the XGBoost CV and confirm:
   - dir_acc is now LOWER than before (this is correct — the old metric was inflated)
   - The new dir_acc measures true directional recall; 0.55+ is a reasonable target
   - Fold-to-fold variance should decrease with balanced class weights

2. Run the meta model retraining on the long bundle and confirm:
   - `edge_cv_mae` is reported instead of in-sample MAE in the WARNING/INFO log
   - edge_cv_mae should be closer to `null_edge_mae * 0.85` with the new regressor params
   - Fallback to precision-only gate should occur less frequently

3. Confirm no regressions in:
   - `predict_proba()` output shape (still `n_samples × 3`)
   - Model save/load roundtrip (`XGBoostSignalModel.save()` / `.load()`)
   - `MetaFeatureBuilder.build()` still returns a valid dict for all signal types

4. Run `grep -n "scale_pos_weight" project/models/xgboost_model.py` and confirm zero results.

5. Run `grep -c "ENHANCED_PARAMS" project/models/xgboost_model.py` and confirm result is 1.
