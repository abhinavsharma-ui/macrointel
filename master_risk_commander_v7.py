import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# 1. SETUP
TRADES_PATH = 'reports/fixed_return_h10_walkforward_trades.csv'
REPORT_DIR  = 'reports/final_calibrated_v7'
os.makedirs(REPORT_DIR, exist_ok=True)

df = pd.read_csv(TRADES_PATH)
def find_col(p_list):
    for p in p_list:
        if p in df.columns: return p
    return None

e_col = find_col(['signal_date', 'entry_date', 'date'])
ex_col = find_col(['exit_date', 'exit_time'])
prob_col = find_col(['prob', 'model_ret', 'score'])

df['entry_date'] = pd.to_datetime(df[e_col]).dt.normalize()
df['exit_date'] = pd.to_datetime(df[ex_col]).dt.normalize()

# 2. SLOT-BASED SIMULATION
dates = pd.date_range(df['entry_date'].min(), df['exit_date'].max(), freq='B')
initial_cap = 100000.0
equity = initial_cap
cash = initial_cap
pos_size_pct = 0.02   # 2% of equity per trade
max_slots = 50        # 50 concurrent positions
slots = [None] * max_slots
history = []

print(f"Running v7.2 Slot Tracker (50 positions | 2% Sizing)...")

for d in tqdm(dates):
    # Step A: Exit trades whose exit_date is TODAY or EARLIER
    for i in range(len(slots)):
        if slots[i] is not None and slots[i]['exit_date'] <= d:
            # Realize PnL (Scaling Fix: / 100)
            trade_pnl = (slots[i]['net_ret'] / 100.0) * (equity * pos_size_pct)
            cash += trade_pnl
            slots[i] = None

    # Step B: Fill empty slots with new signals starting today
    new_candidates = df[df['entry_date'] == d]
    if not new_candidates.empty:
        if prob_col:
            new_candidates = new_candidates.sort_values(by=prob_col, ascending=False)

        for _, trade in new_candidates.iterrows():
            # Find an empty slot
            for i in range(len(slots)):
                if slots[i] is None:
                    slots[i] = trade
                    break
            else:
                break # All slots full

    # Step C: Record Daily Equity (Realized Basis)
    # Fixed the active_count error by using a generator instead of .count()
    active_count = sum(1 for s in slots if s is not None)
    equity = cash
    history.append({'date': d, 'equity': equity, 'active_count': active_count})

stats = pd.DataFrame(history).set_index('date')
stats['dd'] = (stats['equity'] / stats['equity'].cummax() - 1) * 100

# 3. RESULTS
total_y = (stats.index[-1] - stats.index[0]).days / 365.25
cagr = ((stats['equity'].iloc[-1] / initial_cap)**(1/total_y)-1)*100

print("\n" + "="*50)
print(" CALIBRATED SLOT PERFORMANCE (v7.2)")
print("="*50)
print(f"Final Equity:      ${stats['equity'].iloc[-1]:,.2f}")
print(f"CAGR:              {cagr:.2f}%")
print(f"MAX DRAWDOWN:      {stats['dd'].min():.2f}%")
print(f"Avg Active Slots:  {stats['active_count'].mean():.1f}")

# Plot
plt.figure(figsize=(15, 7))
stats['equity'].plot(title='17-Year Equity Curve (v7.2 Slot Tracker)', color='teal')
plt.savefig(f'{REPORT_DIR}/v7_real_results.png')
