import csv, statistics
from pathlib import Path
from collections import defaultdict

f = Path("/home/abhinavsharma1359/macro_intelligence_complete/project/reports/wf_th060_trades.csv")
rows = list(csv.DictReader(open(f)))

def num(x, d=0.0):
    try: return float(x)
    except: return d

wins   = [r for r in rows if num(r.get("pnl")) > 0]
losses = [r for r in rows if num(r.get("pnl")) <= 0]
print(f"Total: {len(rows)}  Wins: {len(wins)} ({100*len(wins)/len(rows):.1f}%)  Losses: {len(losses)} ({100*len(losses)/len(rows):.1f}%)\n")
print("=== 1. Win rate by probability band ===")
for lo,hi in [(0.55,0.60),(0.60,0.65),(0.65,0.70),(0.70,0.75),(0.75,0.80),(0.80,1.01)]:
    sub = [r for r in rows if lo <= num(r["prob"]) < hi]
    w   = [r for r in sub if num(r["pnl"]) > 0]
    if sub:
        print(f"  prob [{lo:.2f},{hi:.2f}): n={len(sub):4d}  wr={100*len(w)/len(sub):5.1f}%  avg_pnl=${statistics.mean(num(r['pnl']) for r in sub):+.2f}")

print("\n=== 2. Win rate by SPY realized vol ===")
for lo,hi in [(0,10),(10,15),(15,20),(20,25),(25,30),(30,50),(50,999)]:
    sub = [r for r in rows if lo <= num(r["spy_realized_vol_pct"]) < hi]
    w   = [r for r in sub if num(r["pnl"]) > 0]
    if sub:
        print(f"  spyvol [{lo:2d},{hi:3d}): n={len(sub):4d}  wr={100*len(w)/len(sub):5.1f}%  avg_pnl=${statistics.mean(num(r['pnl']) for r in sub):+.2f}")

print("\n=== 3. Exit reason ===")
reasons = defaultdict(list)
for r in rows: reasons[r.get("exit_reason","?")].append(r)
for reason, sub in sorted(reasons.items(), key=lambda x: -len(x[1])):
    w = [r for r in sub if num(r["pnl"]) > 0]
    print(f"  {reason:<22s}: n={len(sub):4d}  wr={100*len(w)/len(sub):5.1f}%  avg_pnl=${statistics.mean(num(r['pnl']) for r in sub):+.2f}")
print("\n=== 4. Win rate by hold days ===")
for lo,hi in [(0,1),(1,2),(2,3),(3,5),(5,8),(8,12),(12,99)]:
    sub = [r for r in rows if lo <= num(r["exit_offset"]) < hi]
    w   = [r for r in sub if num(r["pnl"]) > 0]
    if sub:
        print(f"  hold [{lo},{hi}): n={len(sub):4d}  wr={100*len(w)/len(sub):5.1f}%  avg_pnl=${statistics.mean(num(r['pnl']) for r in sub):+.2f}")

print("\n=== 5. PT hit day (winners only) ===")
pt_days = defaultdict(list)
for r in wins:
    try: pt_days[int(float(r.get("pt_day","x")))].append(r)
    except: pass
for day in sorted(pt_days.keys()):
    sub = pt_days[day]
    print(f"  PT day {day:2d}: n={len(sub):4d}  avg_pnl=${statistics.mean(num(r['pnl']) for r in sub):+.2f}")

print("\n=== 6. Loss P&L distribution ===")
loss_pnls = sorted(num(r["pnl"]) for r in losses)
print(f"  min={min(loss_pnls):+.2f}  max={max(loss_pnls):+.2f}  mean={statistics.mean(loss_pnls):+.2f}  median={statistics.median(loss_pnls):+.2f}")
for lo,hi in [(-9999,-100),(-100,-50),(-50,-20),(-20,-10),(-10,-5),(-5,0)]:
    sub = [p for p in loss_pnls if lo <= p < hi]
    if sub: print(f"  [{lo:6.0f},{hi:5.0f}): n={len(sub):4d}  ({100*len(sub)/len(loss_pnls):.1f}%)")

print("\n=== 7. Top 15 symbols by total loss ===")
sym_pnl = defaultdict(float); sym_n = defaultdict(int); sym_w = defaultdict(int)
for r in rows:
    s=r["symbol"]; p=num(r["pnl"]); sym_pnl[s]+=p; sym_n[s]+=1
    if p>0: sym_w[s]+=1
for sym,total in sorted(sym_pnl.items(), key=lambda x:x[1])[:15]:
    n=sym_n[sym]; wr=100*sym_w[sym]/n if n else 0
    print(f"  {sym:<8s}: total_pnl=${total:+.2f}  n={n:3d}  wr={wr:5.1f}%")
print("\n=== 8. Win rate by Kelly multiplier ===")
for lo,hi in [(0,0.5),(0.5,0.75),(0.75,1.0),(1.0,1.25),(1.25,1.5),(1.5,2.01)]:
    sub = [r for r in rows if lo <= num(r.get("kelly_multiplier"),1.0) < hi]
    w   = [r for r in sub if num(r["pnl"]) > 0]
    if sub:
        print(f"  kelly [{lo:.2f},{hi:.2f}): n={len(sub):4d}  wr={100*len(w)/len(sub):5.1f}%  avg_pnl=${statistics.mean(num(r['pnl']) for r in sub):+.2f}")

print("\n=== 9. Win rate by vol_multiplier (market regime) ===")
for lo,hi in [(0,0.5),(0.5,0.76),(0.76,1.01),(1.01,1.26),(1.26,2.0)]:
    sub = [r for r in rows if lo <= num(r.get("vol_multiplier"),1.0) < hi]
    w   = [r for r in sub if num(r["pnl"]) > 0]
    if sub:
        print(f"  volmult [{lo:.2f},{hi:.2f}): n={len(sub):4d}  wr={100*len(w)/len(sub):5.1f}%  avg_pnl=${statistics.mean(num(r['pnl']) for r in sub):+.2f}")

print("\n=== 10. Near-miss losses (positive model_ret but pnl<=0) ===")
near3 = [r for r in losses if num(r.get("model_ret",0)) > 0.03]
near5 = [r for r in losses if num(r.get("model_ret",0)) > 0.05]
print(f"  Losses with model_ret >3%: {len(near3)} ({100*len(near3)/len(losses):.1f}% of losses)")
print(f"  Losses with model_ret >5%: {len(near5)} ({100*len(near5)/len(losses):.1f}% of losses)")
if near3:
    print(f"  Avg PnL of >3% model_ret losses: ${statistics.mean(num(r['pnl']) for r in near3):+.2f}")

print("\n=== 11. Consecutive loss/win streaks ===")
streak_max_loss=0; streak_cur_loss=0; streak_max_win=0; streak_cur_win=0
for r in sorted(rows, key=lambda r: r.get("signal_date","")):
    if num(r["pnl"])>0:
        streak_cur_win+=1; streak_max_win=max(streak_max_win,streak_cur_win); streak_cur_loss=0
    else:
        streak_cur_loss+=1; streak_max_loss=max(streak_max_loss,streak_cur_loss); streak_cur_win=0
print(f"  Max consecutive losses: {streak_max_loss}")
print(f"  Max consecutive wins:   {streak_max_win}")
