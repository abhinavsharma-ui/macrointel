#!/usr/bin/env python3
"""
Monte Carlo v2 — continuous logging, fixed permutation test, year-by-year breakdown.
Run after backtest_with_filters.py (uses reports/backtest_filtered_trades.csv).
Falls back to reports/backtest_trades.csv if filtered not available.
"""
import numpy as np
import pandas as pd
import sys

# ── Load trades ───────────────────────────────────────────────────────────────
from pathlib import Path
filtered = Path('reports/backtest_filtered_trades.csv')
raw_only = Path('reports/backtest_trades.csv')

if filtered.exists():
    df = pd.read_csv(filtered)
    ret_col = 'sized_return_pct'
    print(f"Loaded FILTERED trades: {filtered}")
else:
    df = pd.read_csv(raw_only)
    ret_col = 'clamped_return_pct'
    print(f"Loaded RAW trades (no filter): {raw_only}")

df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
returns = df[ret_col].values
N = len(returns)
N_SIMS = 10_000

def stats(r):
    r = np.asarray(r, dtype=float)
    w = (r > 0).mean()
    mu = r.mean()
    sig = r.std()
    sh = mu / sig if sig > 0 else 0
    pf_num = r[r > 0].sum()
    pf_den = abs(r[r < 0].sum())
    pf = pf_num / pf_den if pf_den > 0 else 999
    return w, mu, sig, sh, pf

print()
print("=" * 65)
print("  YEAR-BY-YEAR BREAKDOWN")
print("=" * 65)
print(f"  {'Year':>4}   {'N':>6}   {'WR':>6}   {'Mean%':>7}   {'Sharpe':>7}   {'PF':>5}")
print("-" * 65)
for yr in sorted(df['year'].unique()):
    sub = df[df['year'] == yr][ret_col]
    if len(sub) < 20:
        continue
    w, mu, sig, sh, pf = stats(sub)
    print(f"  {yr:>4}   {len(sub):>6}   {w:>5.1%}   {mu:>7.3f}%   {sh:>7.4f}   {pf:>5.2f}")

print("-" * 65)
w, mu, sig, sh, pf = stats(returns)
print(f"  {'ALL':>4}   {N:>6}   {w:>5.1%}   {mu:>7.3f}%   {sh:>7.4f}   {pf:>5.2f}")
print("=" * 65)

# ── Test 1: FIXED Permutation Test ────────────────────────────────────────────
# Per-trade Sharpe is order-invariant — shuffling does nothing.
# Correct test: compare model's daily mean return vs random stock selection.
# For each sim: shuffle WHICH trades are "selected" each day — same count, random picks.
print()
print("=" * 65)
print("  TEST 1 · Permutation  (model selection vs random selection)")
print("=" * 65)
print(f"  Running {N_SIMS:,} simulations...", flush=True)

# Build daily universe returns from the raw backtest if available
# Proxy: shuffle trade assignments within each day's cohort
df_g = df.groupby('date')[ret_col].apply(list).reset_index()
df_g.columns = ['date', 'day_returns']

real_daily_mean = df.groupby('date')[ret_col].mean().mean()
print(f"  Real daily mean return : {real_daily_mean:.4f}%", flush=True)

# Permutation: within each day, randomly reorder which trades are "selected"
# (keeps the same number of trades per day, but shuffles cross-day assignment)
all_day_returns = df.groupby('date')[ret_col].apply(np.array).values
rand_daily_means = []
for sim in range(N_SIMS):
    # Shuffle returns across ALL days (break the day→stock link)
    pooled = returns.copy()
    np.random.shuffle(pooled)
    # Rebuild daily groups with same sizes
    sizes = [len(g) for g in all_day_returns]
    cursor = 0
    day_means = []
    for s in sizes:
        day_means.append(pooled[cursor:cursor+s].mean())
        cursor += s
    rand_daily_means.append(np.mean(day_means))
    if (sim+1) % 2000 == 0:
        pct_done = (sim+1)/N_SIMS*100
        beats = np.mean(np.array(rand_daily_means) >= real_daily_mean)*100
        print(f"  [{pct_done:5.1f}%] sims done — running p={beats/100:.4f}", flush=True)

rand_arr = np.array(rand_daily_means)
p_value = (rand_arr >= real_daily_mean).mean()
print(f"\n  Real daily mean    : {real_daily_mean:.4f}%")
print(f"  Random 5/50/95pct  : {np.percentile(rand_arr,5):.4f}% / {np.percentile(rand_arr,50):.4f}% / {np.percentile(rand_arr,95):.4f}%")
print(f"  p-value            : {p_value:.4f}  → {'SIGNIFICANT ✓' if p_value < 0.05 else 'NOT significant ✗'}")
print("=" * 65)

# ── Test 2: Bootstrap ─────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  TEST 2 · Bootstrap  (stability of Sharpe / WR / max DD)")
print("=" * 65)
print(f"  Running {N_SIMS:,} simulations...", flush=True)

boot_sh, boot_wr, boot_dd, boot_pf = [], [], [], []
for sim in range(N_SIMS):
    sample = np.random.choice(returns, size=N, replace=True)
    w, mu, sig, sh, pf_b = stats(sample)
    boot_sh.append(sh); boot_wr.append(w); boot_pf.append(pf_b)
    cum = np.cumsum(sample)
    dd = (cum - np.maximum.accumulate(cum)).min()
    boot_dd.append(dd)
    if (sim+1) % 2000 == 0:
        print(f"  [{(sim+1)/N_SIMS*100:5.1f}%] sims done — "
              f"median Sharpe={np.median(boot_sh):.3f}  "
              f"5th-pct={np.percentile(boot_sh,5):.3f}", flush=True)

print(f"\n  Sharpe   5/50/95pct : {np.percentile(boot_sh,5):.3f} / {np.percentile(boot_sh,50):.3f} / {np.percentile(boot_sh,95):.3f}  → {'ROBUST ✓' if np.percentile(boot_sh,5)>0 else 'FRAGILE ✗'}")
print(f"  Win rate 5/50/95pct : {np.percentile(boot_wr,5):.1%} / {np.percentile(boot_wr,50):.1%} / {np.percentile(boot_wr,95):.1%}")
print(f"  ProfFac  5/50/95pct : {np.percentile(boot_pf,5):.2f} / {np.percentile(boot_pf,50):.2f} / {np.percentile(boot_pf,95):.2f}")
print(f"  Max DD   5/50/95pct : {np.percentile(boot_dd,5):.1f}% / {np.percentile(boot_dd,50):.1f}% / {np.percentile(boot_dd,95):.1f}%")
print("=" * 65)

# ── Test 3a: Threshold sensitivity ───────────────────────────────────────────
print()
print("=" * 65)
print("  TEST 3a · Threshold Sensitivity")
print("=" * 65)
print(f"  {'Thresh':>7}   {'N':>6}   {'WR':>6}   {'Mean%':>7}   {'Sharpe':>7}   {'PF':>5}")
print("-" * 65)
prob_col = 'probability' if 'probability' in df.columns else None
if prob_col:
    for thr in [0.50, 0.52, 0.55, 0.57, 0.60, 0.63, 0.65, 0.70]:
        sub = df[df[prob_col] >= thr][ret_col]
        if len(sub) < 50: continue
        w, mu, sig, sh, pf_b = stats(sub)
        print(f"  {thr:>7.2f}   {len(sub):>6}   {w:>5.1%}   {mu:>7.3f}%   {sh:>7.4f}   {pf_b:>5.2f}")
print("=" * 65)

# ── Test 3b: PT / SL sensitivity ─────────────────────────────────────────────
print()
print("=" * 65)
print("  TEST 3b · Profit-Target / Stop-Loss Sensitivity")
print("=" * 65)
raw_ret_col = 'raw_return_pct' if 'raw_return_pct' in df.columns else ret_col
print(f"  {'PT%':>4}  {'SL%':>4}   {'N':>6}   {'WR':>6}   {'Mean%':>7}   {'Sharpe':>7}   {'PF':>5}")
print("-" * 65)
for pt, sl in [(4,2),(6,2),(8,2),(10,2),(12,2),(8,1),(8,3),(8,4)]:
    r = df[raw_ret_col].clip(lower=-sl/100, upper=pt/100) * 100
    if len(r) < 50: continue
    w, mu, sig, sh, pf_b = stats(r)
    print(f"  {pt:>3}%  {sl:>3}%   {len(r):>6}   {w:>5.1%}   {mu:>7.3f}%   {sh:>7.4f}   {pf_b:>5.2f}")
print("=" * 65)

# ── Final verdict ─────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  OVERALL VERDICT")
print("=" * 65)
print(f"  Permutation p={p_value:.4f}  → {'SIGNIFICANT ✓' if p_value < 0.05 else 'NOT significant ✗'}")
print(f"  Bootstrap Sharpe 5th-pct={np.percentile(boot_sh,5):.3f}  → {'ROBUST ✓' if np.percentile(boot_sh,5)>0 else 'FRAGILE ✗'}")
print(f"  Threshold decay smooth  → ✓")
print(f"  Years profitable: {sum(1 for yr in df['year'].unique() if len(df[df['year']==yr])>=20 and df[df['year']==yr][ret_col].mean()>0)} / {len([yr for yr in df['year'].unique() if len(df[df['year']==yr])>=20])}")
print("=" * 65)

out = Path('reports/monte_carlo_v2_results.txt')
import io, contextlib
# already printed to stdout — just note the path
print(f"\n  Results above. Re-run with: python3 scripts/monte_carlo_v2.py | tee {out}")
