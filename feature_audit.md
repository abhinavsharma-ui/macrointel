# Feature audit — macro_intelligence_complete

**Date:** 2026-05-13
**Scope:** the per-ticker parquet feature store at `project/data/features/*.parquet` and the XGBoost training pipeline in `project/models/xgboost_model.py`.

## TL;DR

Three problems, in order of severity:

1. **Twenty-seven features in the parquet store are constant zero.** The sentiment / news / filing / alt-data writers are not producing real values. They look fine in the schema but carry zero information. The FeatureSelector hides this by dropping them after fit.
2. **The original importance plot is misleading.** Built-in XGBoost importance splits credit between near-duplicate features, which is why `hist_vol_30`, `atr_pct`, and `realized_vol_21d` all looked large. Permutation importance shows `atr_pct` is doing roughly 2x as much work as anything else, and `hist_vol_30` adds essentially nothing on top of it.
3. **Eight features are net harmful** — scrambling them improves the model. Drop immediately.

After the cleanup the candidate feature set shrinks from 102 → ~64, and 19 new features are added in `project/pipeline/new_features.py`.

## What the analysis actually did

- Loaded 60 randomly sampled tickers (~17k rows) from `project/data/features/`.
- Built a 3-class target (top/bottom tercile of 5-day forward return).
- Trained `sklearn.HistGradientBoostingClassifier` (same family as XGBoost — uses histogrammed splits, same kind of feature importances) chronologically (70/30 train/test split).
- Test accuracy: **0.370** vs 0.333 random baseline — a real but modest edge.
- Ran `permutation_importance` on the test set. Numbers below are the mean accuracy drop when each feature is permuted (higher = more important).

Raw output is in `outputs/perm_importance.csv` and `outputs/perm_log.txt`.

## Constant-zero features (drop, fix upstream, or stop saving them)

These 27 columns have zero variance across every ticker — the upstream pipelines that write them are silently producing nothing:

`compound_score`, `weighted_compound_score`, `media_sentiment`, `official_sentiment`, `filing_sentiment`, `filing_change_score`, `filing_fresh_language_score`, `new_risk_factors`, `earnings_tone_signal`, `earnings_call_count`, `article_count`, `media_article_count`, `official_article_count`, `filing_article_count`, `press_release_count`, `official_event_hit`, `filing_event_hit`, `source_quality_score`, `sentiment_zscore`, `sentiment_velocity`, `earnings_tone_velocity`, `weighted_sentiment_zscore`, `news_volume_spike`, `source_quality_signal`, `media_sentiment_signal`, `travel_activity_level`, `travel_activity_change`.

Action item: trace each back to its writer in `project/pipeline/{sentiment_collector, enhanced_sentiment, altdata_collector, earnings_collector}.py`. Until they are producing real values, they are noise in the training pipeline.

## Features to drop (negative or zero permutation importance)

| feature | perm imp | reason |
|---|---|---|
| williams_r | -0.0070 | worst — pure noise, redundant with stoch_k |
| vol_zscore | -0.0060 | redundant with vol_ratio_20, vol_regime_ratio |
| candle_body | -0.0040 | too noisy at single-bar resolution |
| bb_pct | -0.0023 | duplicate of bb_position |
| hist_vol_10 | -0.0017 | redundant with atr_pct |
| rsi_9 | -0.0017 | redundant with rsi_14 |
| close_reversal_signal | -0.0017 | keep `close_reversal_strength` instead |
| vol_ratio | -0.0010 | redundant with vol_ratio_20 |
| roc_5 | -0.0003 | duplicate of returns + momentum |
| roc_10 | -0.0003 | duplicate of momentum_20d |
| gap_up | -0.0003 | rare, not informative on its own |
| peer_earnings_shock_7d, peer_earnings_breadth_7d, event_day_extreme, earnings_propagation_strength | 0.000 | populated <3% of the time; rebuild as `days_since_event` instead |

## Features to keep (real predictive value)

Top 15 by permutation importance:

```
atr_pct                  0.0373   *** dominant
vol_regime_ratio         0.0183
rsi_14                   0.0157
zscore_vs_60d            0.0107
obv_trend                0.0100
vol_ratio_20             0.0100
alpha_signal             0.0097
momentum_60d             0.0083
mfi                      0.0067
vol_regime               0.0063
roc_20                   0.0060
rsi_21                   0.0060
close_reversal_strength  0.0057
upper_wick               0.0053
hist_vol_30              0.0053
```

Note the divergence from the original chart: by built-in importance, `hist_vol_30` was the #1 feature at 0.1928. By permutation importance it ranks #15. The model can rebuild what `hist_vol_30` provides from `atr_pct` and `realized_vol_21d`.

## Redundancy clusters among survivors

Two high-correlation pairs in the top 25 (|r| > 0.85): `alpha_signal ~~ momentum_composite` and `close_reversal_strength ~~ event_move_strength`. Keep one of each pair — the code drops `momentum_composite` and `event_move_strength`.

## What was missing — added in `pipeline/new_features.py`

Per-ticker, no extra data needed:

- `mom_12_1` — Jegadeesh-Titman 12-1 momentum. Skips the last 21 days to avoid short-term reversal contamination.
- `risk_adj_mom_63` — rolling 63-day Sharpe of daily returns. Often beats raw momentum on quality-adjusted basis.
- `mom_dispersion` — gap between short and long momentum, captures trend acceleration.
- `vol_of_vol_21d`, `vol_zscore_252d`, `vol_accel` — volatility *dynamics*, not just levels.
- `ret_skew_63`, `ret_kurt_63`, `downside_var_21` — higher moments / tail risk.
- `amihud_illiq_21`, `dollar_vol_z60` — liquidity regime.
- `dd_from_63d_high`, `days_since_63d_high` — drawdown context.
- `dow`, `month`, `is_monday`, `is_friday`, `is_month_end_5d`, `is_month_start_5d` — calendar.
- `days_since_peer_earn` — rebuilds the sparse `peer_earnings_event_count_7d` as a dense feature.

Need extra data (the file accepts an optional `macro` DataFrame):

- `macro_vix`, `macro_vix_term_structure`, `macro_hy_oas`, `macro_ust_2y_10y_spread` — regime levels.
- 5-day changes in VIX / HY OAS / DXY / 10Y — regime moves.
- 21-day returns of oil / gold / DXY — cross-asset context.

Need the universe panel (also optional):

- `*_xs_rank` — cross-sectional percentile rank within each date for momentum / volatility / RSI. Cross-sectional features are the single biggest missing piece for a universe of 312 tickers.

## How to wire it in

```python
from pipeline.new_features import add_new_features

X = add_new_features(
    df,                       # existing per-ticker feature DataFrame
    macro=macro_df,           # optional
    universe_df=universe_df,  # optional
)
# Then pass X to XGBoostSignalModel.fit() as before.
```

`add_new_features` calls `drop_useless` at the end, so the 38 useless/harmful/redundant columns are removed even if you don't add a single new feature. That alone should give you a cleaner importance chart and a small accuracy lift.

## Recommended next steps, in order

1. **Pull `add_new_features` into your training loop and rerun your importance plot.** Use permutation importance, not built-in. The picture will be much clearer.
2. **Fix the sentiment/alt-data writers** — that's 27 features being saved as zeros today. Until they are real, you can't tell whether the model needs them.
3. **Add macro context** — even just VIX level and 5d-change, HY OAS, and the 2s10s spread. For a "macro_intelligence" project, this is the highest-leverage missing piece.
4. **Add cross-sectional ranks**. With a 312-ticker universe, percentile-within-date features routinely beat single-asset features.
5. **Increase the training history.** ~289 rows × 312 tickers = ~90k samples spanning only 14 months. Permutation importance is noisy with this little time-series. Extending to 5+ years of history will firm up the rankings.
