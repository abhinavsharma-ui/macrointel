import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 1. SETUP
TRADES_PATH = 'reports/fixed_return_h10_walkforward_trades.csv'
MAP_PATH    = 'data/altdata/sector_mapping.json'
PRICES_DIR  = 'data/prices_full'
REPORT_DIR  = 'reports/institutional_survival_v4'
os.makedirs(REPORT_DIR, exist_ok=True)

df = pd.read_csv(TRADES_PATH)
def find_col(p_list):
    for p in p_list:
        if p in df.columns: return p
    return None

e_col = find_col(['signal_date', 'entry_date', 'date'])
ex_col = find_col(['exit_date', 'exit_time'])
p_col = find_col(['entry_price', 'price'])

df['entry_date'] = pd.to_datetime(df[e_col]).dt.normalize()
df['exit_date'] = pd.to_datetime(df[ex_col]).dt.normalize()
df['entry_price'] = df[p_col]

with open(MAP_PATH, 'r') as f: meta_map = json.load(f)

class RiskDataEngine:
    def __init__(self, path):
        self.path = path
        self.cache = {}
    def get_row(self, sym, date):
        if sym not in self.cache:
            p = f"{self.path}/{sym}.parquet"
            if os.path.exists(p): self.cache[sym] = pd.read_parquet(p)
            else: return None
        try: return self.cache[sym].asof(date)
        except: return None

rde = RiskDataEngine(PRICES_DIR)

# 2. SURVIVAL PARAMETERS
dates = pd.date_range(df['entry_date'].min(), df['exit_date'].max(), freq='B')
initial_cap = 100000.0
equity = initial_cap
cash = initial_cap
pos_size_pct = 0.005 # 🚨 DROPPED TO 0.5% FOR SURVIVAL 🚨
sl_limit = -0.05      # 🚨 WIDENED TO 5% TO REDUCE NOISE 🚨
stopped_trades = set()
history = []

print(f"Running Survival Simulation (0.5% sizing)...")

for d in tqdm(dates):
    active = df[(df['entry_date'] <= d) & (df['exit_date'] > d)].copy()
    active = active[~active.index.isin(stopped_trades)]

    if equity <= 1000: # Circuit breaker
        break

    # Dynamic Sizing
    dollar_size = equity * pos_size_pct

    # Data Fetch
    prices = {}
    for s in active['symbol'].unique():
        row = rde.get_row(s, d)
        if row is not None: prices[s] = row['close']

    active['curr_px'] = active['symbol'].map(prices)
    active = active.dropna(subset=['curr_px'])
    active['mtm_ret'] = (active['curr_px'] - active['entry_price']) / active['entry_price']

    # SL Logic
    hits_sl = active[active['mtm_ret'] <= sl_limit].index
    for idx in hits_sl:
        cash += (sl_limit - 0.001) * dollar_size
        stopped_trades.add(idx)

    active = active[~active.index.isin(stopped_trades)]

    # Standard Exit
    closed = df[(df['exit_date'] == d) & (~df.index.isin(stopped_trades))]
    cash += (closed['net_ret'] * dollar_size).sum() if 'net_ret' in closed else (closed['return_pct']/100 * dollar_size).sum()

    equity = cash + (active['mtm_ret'] * dollar_size).sum()
    history.append({'date': d, 'equity': equity, 'count': len(active)})

stats = pd.DataFrame(history).set_index('date')
stats['dd'] = (stats['equity'] / stats['equity'].cummax() - 1) * 100

# 3. FINAL RESULTS
total_y = (stats.index[-1] - stats.index[0]).days / 365.25
cagr = ((stats['equity'].iloc[-1] / initial_cap)**(1/total_y)-1)*100

print("\n" + "="*50)
print(" INSTITUTIONAL SURVIVAL REPORT (v4.3)")
print("="*50)
print(f"Final Equity:      ${stats['equity'].iloc[-1]:,.2f}")
print(f"CAGR:              {cagr:.2f}%")
print(f"MAX DRAWDOWN:      {stats['dd'].min():.2f}%")
print(f"Stopped Trades:    {len(stopped_trades)}")
print(f"Survival Status:   {'✅ SURVIVED' if equity > 1000 else '💀 DECEASED'}")

# 4. PLOT
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
stats['equity'].plot(ax=ax1, title='Survival Equity Curve (0.5% Size)', color='blue')
stats['dd'].plot(ax=ax2, title='Max Drawdown %', kind='area', color='orange', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{REPORT_DIR}/survival_dashboard.png')
