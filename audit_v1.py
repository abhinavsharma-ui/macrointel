import pandas as pd
import numpy as np
import os

# 1. LOAD DATA
TRADES_PATH = 'reports/fixed_return_h10_walkforward_trades.csv'
df = pd.read_csv(TRADES_PATH)

# Robust Column Finder
def get_col(options):
    return next((c for c in options if c in df.columns), None)

d_col = get_col(['entry_date', 'signal_date', 'date', 'Entry Date'])
s_col = get_col(['sector', 'Sector', 'industry'])
r_col = get_col(['net_ret', 'return_pct', 'Return'])
sym_col = get_col(['symbol', 'ticker', 'Symbol'])

if not d_col:
    print(f"Error: Could not find a date column. Available: {df.columns.tolist()}")
    exit()

df[d_col] = pd.to_datetime(df[d_col]).dt.normalize()
df['year'] = df[d_col].dt.year

print("\n" + "="*60)
print("             INSTITUTIONAL STRATEGY AUDIT v1.0")
print("="*60)

# --- 1. HIDDEN LEAKAGE (Look-ahead Check) ---
df_leak = df.sort_values([sym_col, d_col])
df_leak['prev_exit'] = df_leak.groupby(sym_col)[d_col].shift(-1)

# --- 2. CORRELATION CRISES (Sector Crowding) ---
if s_col:
    sector_crowding = df.groupby([d_col, s_col]).size().unstack().fillna(0)
    daily_max_sector = sector_crowding.div(sector_crowding.sum(axis=1), axis=0).max(axis=1)
    print(f"SECTOR CONCENTRATION:")
    print(f"  - Avg Daily Crowding:   {daily_max_sector.mean()*100:.2f}%")
    print(f"  - Peak Sector Heat:     {daily_max_sector.max()*100:.2f}%")
else:
    print("SECTOR CONCENTRATION: [SKIPPED] No sector column found.")

# --- 3. REGIME DEPENDENCY (Annual Breakdown) ---
df['real_ret'] = df[r_col] / 100.0 if df[r_col].mean() > 1 else df[r_col]

regime = df.groupby('year').agg(
    Win_Rate    = ('real_ret', lambda x: (x > 0).mean() * 100),
    Avg_Return  = ('real_ret', lambda x: x.mean() * 100),
    Trade_Count = (sym_col, 'count'),
    Volatility  = ('real_ret', lambda x: x.std() * 100)
)

print("\nANNUAL REGIME PERFORMANCE:")
print(regime.round(2))

# --- 4. CAPACITY SCALING ---
print("\nCAPACITY AUDIT:")
print(f"  - Total Trade Sample:   {len(df)}")
print(f"  - Avg Trades Per Year:  {len(df)/df['year'].nunique():.0f}")

print("="*60)
