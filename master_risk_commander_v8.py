import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# 1. SETUP
TRADES_PATH = 'reports/fixed_return_h10_walkforward_trades.csv'
REPORT_DIR  = 'reports/slippage_stress_v8'
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

# 2. STRESS TEST PARAMETERS
dates = pd.date_range(df['entry_date'].min(), df['exit_date'].max(), freq='B')
initial_cap = 100000.0
equity = initial_cap
cash = initial_cap
pos_size_pct = 0.02
max_slots = 50
slots = [None] * max_slots
history = []

# 🚨 THE SLIPPAGE PENALTY (0.1% = 0.001)
SLIPPAGE = 0.001

print(f"Running v8.0 Stress Test (Slippage: {SLIPPAGE*100}% per trade)...")

for d in tqdm(dates):
    # Step A: Exit trades
    for i in range(len(slots)):
        if slots[i] is not None and slots[i]['exit_date'] <= d:
            # 🚨 APPLY SLIPPAGE TO THE PERCENTAGE RETURN 🚨
            # If the model says 4.75%, we treat it as 4.65% after slippage
            adjusted_ret = (slots[i]['net_ret'] / 100.0) - SLIPPAGE

            trade_pnl = adjusted_ret * (equity * pos_size_pct)
            cash += trade_pnl
            slots[i] = None

    # Step B: Fill slots
    new_candidates = df[df['entry_date'] == d]
    if not new_candidates.empty:
        if prob_col:
            new_candidates = new_candidates.sort_values(by=prob_col, ascending=False)

        for _, trade in new_candidates.iterrows():
            for i in range(len(slots)):
                if slots[i] is None:
                    slots[i] = trade
                    break
            else:
                break

    # Step C: Record Equity
    equity = cash
    history.append({'date': d, 'equity': equity})

stats = pd.DataFrame(history).set_index('date')
stats['dd'] = (stats['equity'] / stats['equity'].cummax() - 1) * 100

# 3. RESULTS
total_y = (stats.index[-1] - stats.index[0]).days / 365.25
cagr = ((stats['equity'].iloc[-1] / initial_cap)**(1/total_y)-1)*100

print("\n" + "="*50)
print(" SLIPPAGE STRESS TEST PERFORMANCE (v8.0)")
print("="*50)
print(f"Final Equity:      ${stats['equity'].iloc[-1]:,.2f}")
print(f"CAGR:              {cagr:.2f}%")
print(f"MAX DRAWDOWN:      {stats['dd'].min():.2f}%")
print(f"Total Friction:    -{SLIPPAGE*100}% per round-trip")

# Plot
plt.figure(figsize=(15, 7))
stats['equity'].plot(title='Equity Curve with 0.1% Slippage Penalty', color='crimson')
plt.savefig(f'{REPORT_DIR}/v8_stress_results.png')
