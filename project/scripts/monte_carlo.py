"""
monte_carlo.py  —  run AFTER backtest_reconstruct.py
Tests: permutation, bootstrap CI, threshold/param sensitivity
"""
import numpy as np, pandas as pd
from pathlib import Path

CSV=Path('reports/backtest_trades.csv')
if not CSV.exists(): raise SystemExit("Run backtest_reconstruct.py first")
df=pd.read_csv(CSV); returns=df['clamped_return_pct'].values; n=len(returns)
if n<30: raise SystemExit(f"Only {n} trades — need more data")
N_SIMS=10_000; rng=np.random.default_rng(42)

def sharpe(r): s=r.std(); return r.mean()/s if s>0 else 0.0
def profit_factor(r):
    w=r[r>0].sum(); l=abs(r[r<0].sum()); return w/l if l>0 else 999.0
def maxdd(r):
    cum=np.cumsum(r); return (cum-np.maximum.accumulate(cum)).min()

real_sh=sharpe(returns); real_pf=profit_factor(returns)
print(f"\n{'='*55}\n  MONTE CARLO ANALYSIS\n{'='*55}")
print(f"  Trades        : {n}")
print(f"  Win rate      : {(returns>0).mean():.1%}")
print(f"  Mean return   : {returns.mean():.3f}%")
print(f"  Profit factor : {real_pf:.2f}")
print(f"  Sharpe/trade  : {real_sh:.4f}")

# --- TEST 1: Permutation ---
print(f"\n{'-'*55}\n  TEST 1 · Permutation  (Is alpha real or luck?)\n{'-'*55}")
ss=np.array([sharpe(rng.permutation(returns)) for _ in range(N_SIMS)])
pv=float(np.mean(ss>=real_sh))
v1="SIGNIFICANT ✓" if pv<0.05 else ("MARGINAL" if pv<0.10 else "NOT significant ✗")
print(f"  Real Sharpe         : {real_sh:.4f}")
print(f"  Shuffled 5/50/95pct : {np.percentile(ss,5):.4f} / {np.percentile(ss,50):.4f} / {np.percentile(ss,95):.4f}")
print(f"  p-value             : {pv:.4f}  → {v1}")

# --- TEST 2: Bootstrap ---
print(f"\n{'-'*55}\n  TEST 2 · Bootstrap  (How stable is the Sharpe?)\n{'-'*55}")
bs=np.empty(N_SIMS); bw=np.empty(N_SIMS); bd=np.empty(N_SIMS); bp=np.empty(N_SIMS)
for i in range(N_SIMS):
    s=rng.choice(returns,size=n,replace=True)
    bs[i]=sharpe(s); bw[i]=(s>0).mean(); bd[i]=maxdd(s); bp[i]=min(profit_factor(s),20)
sh5=np.percentile(bs,5)
v2="ROBUST ✓" if sh5>0 else "FRAGILE ✗"
print(f"  Sharpe  5/50/95pct  : {np.percentile(bs,5):.3f} / {np.percentile(bs,50):.3f} / {np.percentile(bs,95):.3f}  → {v2}")
print(f"  Win rate 5/50/95pct : {np.percentile(bw,5):.1%} / {np.percentile(bw,50):.1%} / {np.percentile(bw,95):.1%}")
print(f"  Max DD  5/50/95pct  : {np.percentile(bd,5):.1f}% / {np.percentile(bd,50):.1f}% / {np.percentile(bd,95):.1f}%")
print(f"  ProfFac 5/50/95pct  : {np.percentile(bp,5):.2f} / {np.percentile(bp,50):.2f} / {np.percentile(bp,95):.2f}")

# --- TEST 3a: Threshold sensitivity ---
print(f"\n{'-'*55}\n  TEST 3a · Threshold Sensitivity\n{'-'*55}")
print(f"  {'Thresh':>7}  {'N':>5}  {'Win%':>6}  {'Mean%':>7}  {'Sharpe':>7}  {'ProfFac':>8}")
prev_sh=None; cliff=False
for th in [0.50,0.52,0.55,0.57,0.60,0.63,0.65,0.70]:
    sub=df[df['probability']>=th]['clamped_return_pct']
    if len(sub)<10: print(f"  {th:.2f}    only {len(sub)} trades — skip"); continue
    sv=sub.values; sh_=sharpe(sv); pf_=profit_factor(sv)
    if prev_sh is not None and sh_<prev_sh-0.30: cliff=True
    print(f"  {th:.2f}    {len(sub):>5}  {sub.gt(0).mean():>6.1%}  {sub.mean():>7.3f}%  {sh_:>7.3f}  {min(pf_,99):>8.2f}")
    prev_sh=sh_
print("  ⚠ Cliff-drop detected — possible overfit" if cliff else "  ✓ Smooth decay — threshold robust")

# --- TEST 3b: PT/SL sensitivity ---
print(f"\n{'-'*55}\n  TEST 3b · Profit-Target / Stop-Loss Sensitivity\n{'-'*55}")
print(f"  {'PT%':>4}  {'SL%':>4}  {'Win%':>6}  {'Mean%':>7}  {'Sharpe':>7}  {'ProfFac':>8}")
raw_frac=df['raw_return_pct'].values/100.0
seen=set()
for pt,sl in [(0.04,0.03),(0.06,0.03),(0.08,0.03),(0.10,0.03),(0.12,0.03),(0.08,0.02),(0.08,0.04),(0.08,0.05)]:
    if (pt,sl) in seen: continue
    seen.add((pt,sl))
    clamped=np.clip(raw_frac,-sl,pt)*100
    sh_=sharpe(clamped); pf_=profit_factor(clamped)
    print(f"  {pt*100:.0f}%   {sl*100:.0f}%   {(clamped>0).mean():>6.1%}  {clamped.mean():>7.3f}%  {sh_:>7.3f}  {min(pf_,99):>8.2f}")

# --- Verdict ---
print(f"\n{'='*55}\n  OVERALL VERDICT\n{'='*55}")
for s in [
    ("✓" if pv<0.05 else "~" if pv<0.10 else "✗") + f" Permutation p={pv:.4f} — " + v1,
    ("✓" if sh5>0.10 else "~" if sh5>0 else "✗") + f" Bootstrap Sharpe 5th-pct={sh5:.3f} — " + v2,
    ("✗ Threshold cliff detected" if cliff else "✓ Threshold decay smooth"),
]: print(f"  {s}")
print()

Path('reports').mkdir(exist_ok=True)
out=Path('reports/monte_carlo_results.txt')
# collect stdout above already printed; write summary
out.write_text(f"Trades={n} | Sharpe={real_sh:.4f} | p-value={pv:.4f} | Bootstrap5th={sh5:.3f} | CliffDetected={cliff}\n")
print(f"Summary saved → {out}")
