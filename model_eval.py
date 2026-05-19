import pandas as pd
import numpy as np

# 1. LOAD THE EXISTING CHAMPION MODEL
# Replace with the path to your new model's CSV when testing the challenger
TRADES_PATH = 'reports/fixed_return_h10_walkforward_trades.csv'
df = pd.read_csv(TRADES_PATH)

# Robust column finding
def get_col(options): return next((c for c in options if c in df.columns), None)
r_col = get_col(['net_ret', 'return_pct', 'Return'])
prob_col = get_col(['prob', 'model_ret', 'score'])

if not r_col:
    print("Error: Could not find return column.")
    exit()

# Fix percentage scaling if necessary (4.75 -> 0.0475)
df['real_ret'] = df[r_col] / 100.0 if df[r_col].mean() > 1 else df[r_col]

# Define a "Win" (True Positive)
df['is_win'] = df['real_ret'] > 0

print("\n" + "="*50)
print(" CHAMPION MODEL BASELINE (EXISTING)")
print("="*50)

# --- OVERALL METRICS ---
total_trades = len(df)
wins = df['is_win'].sum()
precision = (wins / total_trades) * 100
avg_win = df[df['is_win']]['real_ret'].mean() * 100
avg_loss = df[~df['is_win']]['real_ret'].mean() * 100
expectancy = (precision/100 * avg_win) + ((1 - precision/100) * avg_loss)

print(f"Overall Precision (Win Rate): {precision:.2f}%")
print(f"Overall Expectancy:           {expectancy:.3f}% per trade")
print(f"Total Trades:                 {total_trades}")
print("-"*50)

# --- THRESHOLD ANALYSIS (If you have a probability column) ---
if prob_col:
    print("\nPRECISION BY CONFIDENCE THRESHOLD:")
    print("Prob > | Precision | Expectancy | Trades")
    print("-" * 45)

    # Test different confidence percentiles (e.g., top 10%, top 20%)
    thresholds = df[prob_col].quantile([0.5, 0.7, 0.9, 0.95]).values

    for t in thresholds:
        subset = df[df[prob_col] >= t]
        if len(subset) == 0: continue

        s_wins = subset['is_win'].sum()
        s_prec = (s_wins / len(subset)) * 100
        s_avg_w = subset[subset['is_win']]['real_ret'].mean() * 100
        s_avg_l = subset[~subset['is_win']]['real_ret'].mean() * 100
        s_exp = (s_prec/100 * s_avg_w) + ((1 - s_prec/100) * s_avg_l)

        print(f" > {t:.3f} |   {s_prec:.1f}%   |   {s_exp:>6.3f}%  | {len(subset)}")
else:
    print("\n[!] No probability/score column found for threshold analysis.")

print("="*50)
