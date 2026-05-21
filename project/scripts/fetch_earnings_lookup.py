"""
Earnings Lookup Builder (yfinance)
====================================
Fetches historical earnings dates + EPS surprise for every symbol in the
26yr liquid feature store. Saves to data/earnings_lookup.csv.

This CSV is the data source for pead_calm_walkforward_sim.py — no Finnhub
API key required.

Usage:
    python scripts/fetch_earnings_lookup.py

Runtime: ~15-30 minutes for ~500 symbols (rate-limited to be polite)
Output:  data/earnings_lookup.csv
         columns: symbol, earnings_date, eps_estimate, reported_eps, surprise_pct
"""

import glob
import os
import time
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf

# ── Config ─────────────────────────────────────────────────────────────────────
ROOT        = Path(os.getenv("PEAD_ROOT", "data/features_26yr_liquid"))
OUT_CSV     = Path(os.getenv("EARNINGS_LOOKUP", "data/earnings_lookup.csv"))
LIMIT       = int(os.getenv("EARNINGS_LIMIT", "52"))      # ~13 years of quarters
MAX_WORKERS = int(os.getenv("EARNINGS_WORKERS", "4"))     # keep low to avoid rate-limit
SLEEP_MIN   = float(os.getenv("EARNINGS_SLEEP_MIN", "0.3"))
SLEEP_MAX   = float(os.getenv("EARNINGS_SLEEP_MAX", "0.8"))


def norm_sym(p: Path) -> str:
    return p.stem.replace("_US", "").replace(".US", "").upper()


def get_symbols() -> list:
    files = sorted(ROOT.glob("*.parquet"))
    syms = []
    for p in files:
        s = norm_sym(p)
        if s.endswith((".NS", ".BO", ".NSE", ".BSE", "SPY", "QQQ", "IWM")):
            continue
        syms.append(s)
    return syms


def fetch_one(sym: str) -> pd.DataFrame:
    try:
        time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))
        t  = yf.Ticker(sym)
        ed = t.get_earnings_dates(limit=LIMIT)
        if ed is None or ed.empty:
            return pd.DataFrame()
        ed = ed.reset_index()
        ed.columns = [c.strip() for c in ed.columns]
        # Normalise column names from yfinance
        col_map = {}
        for c in ed.columns:
            cl = c.lower().replace(" ", "_").replace("(%)", "pct").replace("(%)","pct")
            col_map[c] = cl
        ed = ed.rename(columns=col_map)
        # Expected cols after rename: earnings_date, eps_estimate, reported_eps, surprise_pct
        date_col = next((c for c in ed.columns if "date" in c.lower()), None)
        if date_col is None:
            return pd.DataFrame()
        ed = ed.rename(columns={date_col: "earnings_date"})
        ed["symbol"] = sym
        # Normalise date: strip timezone, keep date only
        ed["earnings_date"] = pd.to_datetime(ed["earnings_date"], utc=True).dt.tz_localize(None).dt.normalize()
        # Only keep rows with a valid reported EPS (future dates have NaN)
        rep_col = next((c for c in ed.columns if "reported" in c.lower()), None)
        if rep_col:
            ed = ed.dropna(subset=[rep_col])
        keep = ["symbol", "earnings_date"]
        for want in ["eps_estimate", "reported_eps", "surprise_pct",
                     "surprise(%)", "reported_eps", "eps_estimate"]:
            match = next((c for c in ed.columns if want.replace(" ","_") in c.lower()), None)
            if match and match not in keep:
                keep.append(match)
        ed = ed[[c for c in keep if c in ed.columns]].copy()
        # Standardise surprise column name
        for c in ed.columns:
            if "surprise" in c.lower() and c != "surprise_pct":
                ed = ed.rename(columns={c: "surprise_pct"})
        return ed
    except Exception as e:
        print(f"  WARN {sym}: {e}")
        return pd.DataFrame()


def main():
    syms = get_symbols()
    print(f"Fetching earnings for {len(syms)} symbols (limit={LIMIT} quarters each)...")
    print(f"Workers={MAX_WORKERS}, sleep={SLEEP_MIN}-{SLEEP_MAX}s per call")
    print(f"Estimated time: {len(syms) * (SLEEP_MIN + SLEEP_MAX) / 2 / MAX_WORKERS / 60:.0f} min")

    results = []
    done = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch_one, s): s for s in syms}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                df = fut.result()
                if len(df):
                    results.append(df)
                    done += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"  ERROR {sym}: {e}")
                failed += 1
            total = done + failed
            if total % 50 == 0:
                print(f"  Progress: {total}/{len(syms)} | ok={done} failed={failed}")

    if not results:
        print("ERROR: No earnings data fetched. Check yfinance / network.")
        return

    out = pd.concat(results, ignore_index=True)
    out = out.sort_values(["symbol", "earnings_date"]).reset_index(drop=True)

    # Ensure surprise_pct exists
    if "surprise_pct" not in out.columns:
        print("WARNING: surprise_pct column missing — adding NaN column")
        out["surprise_pct"] = float("nan")

    # Deduplicate
    out = out.drop_duplicates(subset=["symbol", "earnings_date"])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    print(f"\nDone. {len(out)} earnings events | {out.symbol.nunique()} symbols")
    print(f"Date range: {out.earnings_date.min().date()} → {out.earnings_date.max().date()}")
    print(f"Saved to {OUT_CSV}")

    # Quick sanity check
    yr_dist = out.groupby(out.earnings_date.dt.year).size()
    print("\nEvents per year (sample):")
    for yr in sorted(yr_dist.index)[-10:]:
        print(f"  {yr}: {yr_dist[yr]}")


if __name__ == "__main__":
    main()
