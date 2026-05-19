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
REPORT_DIR  = 'reports/institutional_risk_report'
os.makedirs(REPORT_DIR, exist_ok=True)

df = pd.read_csv(TRADES_PATH)

# Universal Column Sniper
def find_col(possible_names):
    for name in possible_names:
        if name in df.columns: return name
    return None

entry_col = find_col(['signal_date', 'entry_date', 'entry_time', 'date'])
exit_col  = find_col(['exit_date', 'exit_time'])
price_col = find_col(['entry_price', 'price', 'Entry Price'])

df['entry_date'] = pd.to_datetime(df[entry_col]).dt.normalize()
df['exit_date'] = pd.to_datetime(df[exit_col]).dt.normalize()
df['entry_price'] = df[price_col]

# 2. DATA ENGINE
class DataEngine:
    def __init__(self, path):
        self.path = path
        self.cache = {}
    def get_data(self, sym, date):
        if sym not in self.cache:
            p = f"{self.path}/{sym}.parquet"
            if os.path.exists(p): self.cache[sym] = pd.read_parquet(p)
            else: return None
        try: return self.cache[sym].asof(date)['close']
        except: return None

de = DataEngine(PRICES_DIR)

# 3. SURVIVAL RECONSTRUCTION (3% STOP LOSS)
dates = pd.date_range(df['entry_date'].min(), df['exit_date'].max(), freq='B')
cap, cash = 100000.0, 100000.0
pos_size = 0.02
sl_threshold = -0.03
history = []

# Track trades that were stopped out to avoid double counting
stopped_trades = set()

print(f"Analyzing Survival Risk with 3% Stop Loss...")

for d in tqdm(dates):
    # Active trades (excluding those already stopped out)
    active = df[(df['entry_date'] <= d) & (df['exit_date'] > d)].copy()
    active = active[~active.index.isin(stopped_trades)]

    if active.empty:
        history.append({'date': d, 'equity': cash, 'dd': 0})
        continue

    # Mark-to-Market
    active['curr_px'] = active['symbol'].apply(lambda s: de.get_data(s, d))
    active = active.dropna(subset=['curr_px'])
    active['mtm_ret'] = (active['curr_px'] - active['entry_price']) / active['entry_price']

    # 🚨 STOP LOSS CHECK 🚨
    hits_sl = active[active['mtm_ret'] <= sl_threshold].index
    for idx in hits_sl:
        # Liquidate at -3% + 0.1% slippage buffer = -3.1%
        cash += (sl_threshold - 0.001) * (cap * pos_size)
        stopped_trades.add(idx)

    # Refresh active list after SL liquidations
    active = active[~active.index.isin(stopped_trades)]

    floating_pnl = (active['mtm_ret'] * (cap * pos_size)).sum()

    # Standard Exit (Reached Exit Date without hitting SL)
    closed_today = df[(df['exit_date'] == d) & (~df.index.isin(stopped_trades))]
    cash += (closed_today['net_ret'] * (cap * pos_size)).sum() if 'net_ret' in closed_today else (closed_today['return_pct']/100 * (cap * pos_size)).sum()

    equity = cash + floating_pnl
    history.append({'date': d, 'equity': equity})

stats = pd.DataFrame(history).set_index('date')
stats['dd'] = (stats['equity'] / stats['equity'].cummax() - 1) * 100

# 4. RESULTS
total_years = (stats.index[-1] - stats.index[0]).days / 365.25
cagr = ((stats['equity'].iloc[-1] / cap) ** (1/total_years) - 1) * 100

print("\n" + "="*40)
print(" SAFE SYSTEM PERFORMANCE (3% STOP LOSS)")
print("="*40)
print(f"Final Equity: ${stats['equity'].iloc[-1]:,.2f}")
print(f"CAGR:         {cagr:.2f}%")
print(f"MAX DRAWDOWN: {stats['dd'].min():.2f}%")
print(f"Trades Stopped Out: {len(stopped_trades)}")

# 5. PLOT
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
stats['equity'].plot(ax=ax1, title='Safe MTM Equity Curve', color='blue')
stats['dd'].plot(ax=ax2, title='Safe Drawdown % (3% SL Active)', kind='area', color='orange', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{REPORT_DIR}/safe_risk_report.png')
