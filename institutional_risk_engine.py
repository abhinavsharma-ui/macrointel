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

if not os.path.exists(TRADES_PATH):
    print(f"Error: {TRADES_PATH} not found.")
    exit()

# 2. UPDATED COLUMN SNIPER
df = pd.read_csv(TRADES_PATH)

def find_col(possible_names):
    for name in possible_names:
        if name in df.columns: return name
    return None

# Added 'signal_date' to the list here:
entry_col = find_col(['signal_date', 'entry_date', 'entry_time', 'date', 'timestamp'])
exit_col  = find_col(['exit_date', 'exit_time', 'Exit Date'])
price_col = find_col(['entry_price', 'price', 'Entry Price', 'avg_price'])

if not entry_col or not exit_col:
    print(f"Found columns: {list(df.columns)}")
    print("Error: Could not identify Date columns. Please check headers.")
    exit()

df['entry_date'] = pd.to_datetime(df[entry_col]).dt.normalize()
df['exit_date'] = pd.to_datetime(df[exit_col]).dt.normalize()
if price_col != 'entry_price': df['entry_price'] = df[price_col]

print(f"Success: Identified '{entry_col}' as Entry and '{exit_col}' as Exit.")

# 3. METADATA & DATA ENGINE
with open(MAP_PATH, 'r') as f: meta_map = json.load(f)

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

# 4. RECONSTRUCTION (NAKED / NO-SL)
dates = pd.date_range(df['entry_date'].min(), df['exit_date'].max(), freq='B')
cap, cash = 100000.0, 100000.0
pos_size = 0.02
history = []

print(f"Analyzing Naked Risk for {len(df)} trades at 2% sizing...")

for d in tqdm(dates):
    active = df[(df['entry_date'] <= d) & (df['exit_date'] > d)].copy()
    if active.empty:
        history.append({'date': d, 'equity': cash, 'dd': 0, 'count': 0})
        continue

    # MTM & Concentration
    active['curr_px'] = active['symbol'].apply(lambda s: de.get_data(s, d))
    active = active.dropna(subset=['curr_px'])

    floating_pnl = ((active['curr_px'] - active['entry_price']) / active['entry_price'] * (cap * pos_size)).sum()

    active['sector'] = active['symbol'].map(lambda x: meta_map.get(x, {}).get('sector', 'Unknown'))
    s_weights = (active.groupby('sector')['symbol'].count() * pos_size)
    top_3_s = s_weights.nlargest(3).sum()

    # Cash Realization
    closed = df[df['exit_date'] == d]
    cash += (closed['net_ret'] * (cap * pos_size)).sum() if 'net_ret' in closed else (closed['return_pct']/100 * (cap * pos_size)).sum()

    history.append({
        'date': d, 'equity': cash + floating_pnl,
        'top_3_s': top_3_s, 'count': len(active)
    })

stats = pd.DataFrame(history).set_index('date')
stats['dd'] = (stats['equity'] / stats['equity'].cummax() - 1) * 100

# 5. FINAL METRICS
total_years = (stats.index[-1] - stats.index[0]).days / 365.25
cagr = ((stats['equity'].iloc[-1] / cap) ** (1/total_years) - 1) * 100

print("\n" + "="*40)
print(" NAKED SYSTEM PERFORMANCE (2% SIZE)")
print("="*40)
print(f"Final Equity: ${stats['equity'].iloc[-1]:,.2f}")
print(f"CAGR:         {cagr:.2f}%")
print(f"MAX DRAWDOWN: {stats['dd'].min():.2f}%")
print(f"Peak Trades:  {stats['count'].max()}")

# 6. PLOT
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
stats['equity'].plot(ax=ax1, title='Mark-to-Market Equity Curve', color='green')
stats['dd'].plot(ax=ax2, title='True Drawdown % (Time Exit Only)', kind='area', color='red', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{REPORT_DIR}/universal_risk_report.png')
print(f"\nDashboard: {REPORT_DIR}/universal_risk_report.png")
