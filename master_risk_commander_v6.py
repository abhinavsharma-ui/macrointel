import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# 1. SETUP
TRADES_PATH = 'reports/fixed_return_h10_walkforward_trades.csv'
PRICES_DIR  = 'data/prices_full'
REPORT_DIR  = 'reports/sniper_report'
os.makedirs(REPORT_DIR, exist_ok=True)

# 2. LOAD & SNIPE
df = pd.read_csv(TRADES_PATH)
def find_col(p_list):
    for p in p_list:
        if p in df.columns: return p
    return None

e_col = find_col(['signal_date', 'entry_date', 'date'])
ex_col = find_col(['exit_date', 'exit_time'])
p_col = find_col(['entry_price', 'price'])
prob_col = find_col(['prob', 'model_ret', 'score'])

df['entry_date'] = pd.to_datetime(df[e_col]).dt.normalize()
df['exit_date'] = pd.to_datetime(df[ex_col]).dt.normalize()
df['entry_price'] = df[p_col]

# 3. PRICE ENGINE
class RiskDataEngine:
    def __init__(self, path):
        self.path = path
        self.cache = {}
    def get_price(self, sym, date):
        if sym not in self.cache:
            p = f"{self.path}/{sym}.parquet"
            if os.path.exists(p): self.cache[sym] = pd.read_parquet(p)
            else: return None
        try: return self.cache[sym].asof(date)['close']
        except: return None

rde = RiskDataEngine(PRICES_DIR)

# 4. SNIPER PARAMETERS
dates = pd.date_range(df['entry_date'].min(), df['exit_date'].max(), freq='B')
initial_cap = 100000.0
equity = initial_cap
cash = initial_cap
pos_size_pct = 0.0075  # 0.75% sizing
max_slots = 5          # Only 5 trades at a time
history = []

print(f"Running Sniper Simulation (5 slots @ 0.75% | No SL)...")

for d in tqdm(dates):
    # Get all potential trades for today and rank by probability
    active = df[(df['entry_date'] <= d) & (df['exit_date'] > d)].copy()

    # Filter for the Top 5 most "confident" signals
    if not active.empty and prob_col:
        active = active.sort_values(by=prob_col, ascending=False).head(max_slots)
    elif not active.empty:
        active = active.head(max_slots)

    if active.empty:
        history.append({'date': d, 'equity': equity, 'dd': (equity/initial_cap)-1})
        continue

    # Static sizing based on initial cap for 0.75%
    dollar_size = initial_cap * pos_size_pct

    # Mark-to-Market
    prices = {s: rde.get_price(s, d) for s in active['symbol'].unique()}
    active['curr_px'] = active['symbol'].map(prices)
    active = active.dropna(subset=['curr_px'])
    active['mtm_ret'] = (active['curr_px'] - active['entry_price']) / active['entry_price']

    # Standard Exit (8-day rule) - No SL here
    closed = active[active['exit_date'] == d]
    cash += (closed['net_ret'] * dollar_size).sum() if 'net_ret' in closed else (closed['return_pct']/100 * dollar_size).sum()

    equity = cash + (active['mtm_ret'] * dollar_size).sum()
    history.append({'date': d, 'equity': equity})

stats = pd.DataFrame(history).set_index('date')
stats['dd'] = (stats['equity'] / stats['equity'].cummax() - 1) * 100

# 5. FINAL RESULTS
total_y = (stats.index[-1] - stats.index[0]).days / 365.25
cagr = ((stats['equity'].iloc[-1] / initial_cap)**(1/total_y)-1)*100

print("\n" + "="*50)
print(" SNIPER SYSTEM PERFORMANCE (0.75% SIZE | No SL)")
print("="*50)
print(f"Final Equity:      ${stats['equity'].iloc[-1]:,.2f}")
print(f"CAGR:              {cagr:.2f}%")
print(f"MAX DRAWDOWN:      {stats['dd'].min():.2f}%")
print(f"Exposure Cap:      {max_slots * pos_size_pct * 100:.2f}% of Portfolio")

# Plotting
plt.figure(figsize=(15, 7))
stats['equity'].plot(title='Sniper Equity Curve (5 Trades @ 0.75% | No SL)', color='purple')
plt.savefig(f'{REPORT_DIR}/sniper_final_report.png')
