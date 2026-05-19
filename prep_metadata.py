import pandas as pd
import yfinance as yf
import json
import os
import glob
from tqdm import tqdm

OUT_DIR = 'data/altdata'
os.makedirs(OUT_DIR, exist_ok=True)

# Get the symbols actually present in your data
files = glob.glob("data/prices_full/*.parquet")
symbols = [os.path.basename(f).replace('.parquet', '') for f in files]

mapping = {}
print(f"Scraping sector, industry, and factor metadata for {len(symbols)} symbols...")

for sym in tqdm(symbols):
    try:
        t = yf.Ticker(sym)
        i = t.info
        mapping[sym] = {
            "sector": i.get('sector', 'Unknown'),
            "industry": i.get('industry', 'Unknown'),
            "mkt_cap": i.get('marketCap', 0),
            "beta": i.get('beta', 1.0)
        }
    except Exception:
        mapping[sym] = {"sector": "Unknown", "industry": "Unknown", "mkt_cap": 0, "beta": 1.0}

with open(f'{OUT_DIR}/sector_mapping.json', 'w') as f:
    json.dump(mapping, f, indent=4)

print("\n--- METADATA PREP COMPLETE ---")
print(f"File saved to: {OUT_DIR}/sector_mapping.json")
