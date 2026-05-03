#!/usr/bin/env python
"""Refresh OHLCV + derived features for US symbols using yfinance.

Write rules per row:
  - OHLCV cols: always overwrite with fresh yfinance values.
  - Derived cols (adv20, return_*, sma_*): write only if currently NaN.
  - All other columns (sentiment, events, signals, etc.): never touched.
"""

import os
import sys
import time
import glob
from pathlib import Path

import pandas as pd
import yfinance as yf

EXCLUDE_TOKENS = (".NS", ".BO", ".NSE", ".BSE")
FEATURES_DIR = Path("data/features")
BATCH_SIZE = 50
SLEEP_BETWEEN_BATCHES = 1.0

OHLCV_COLS = ("open", "high", "low", "close", "volume")
DERIVED_COLS = (
    "adv20_dollar_vol",
    "return_1d", "return_5d", "return_20d",
    "sma_20", "sma_50", "sma_200",
)


def list_us_parquets(features_dir: Path):
    files = sorted(glob.glob(str(features_dir / "*.parquet")))
    out = []
    for f in files:
        name = os.path.basename(f)
        if any(tok in name for tok in EXCLUDE_TOKENS):
            continue
        out.append(f)
    return out


def symbol_from_filename(path: str) -> str:
    stem = Path(path).stem
    if stem.endswith("_US"):
        stem = stem[:-3]
    return stem


def extract_ticker_frame(data: pd.DataFrame, ticker: str, batch_len: int) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()
    if batch_len == 1:
        df = data.copy()
    else:
        if ticker not in data.columns.get_level_values(0):
            return pd.DataFrame()
        df = data[ticker].copy()
    df = df.dropna(how="all")
    if df.empty:
        return pd.DataFrame()
    df.columns = [str(c).lower() for c in df.columns]
    keep = [c for c in OHLCV_COLS if c in df.columns]
    return df[keep]


def update_parquet(parquet_path: str, fresh: pd.DataFrame) -> bool:
    if fresh.empty or "close" not in fresh.columns:
        return False

    existing = pd.read_parquet(parquet_path)
    if existing.empty:
        return False

    fresh = fresh.copy()
    fresh.index    = pd.to_datetime(fresh.index).tz_localize(None).normalize()
    existing.index = pd.to_datetime(existing.index).tz_localize(None).normalize()

    last_date   = existing.index[-1]
    latest_date = fresh.index[-1]
    latest_row  = fresh.iloc[-1]
    target_date = latest_date if latest_date >= last_date else last_date

    # --- OHLCV: always overwrite the target row with fresh market data ---
    for col in OHLCV_COLS:
        if col in latest_row.index and pd.notna(latest_row[col]):
            if col not in existing.columns:
                existing[col] = pd.NA
            existing.loc[target_date, col] = float(latest_row[col])

    # --- Derived: compute from full series, write only where target row is NaN ---
    close  = existing["close"].astype(float)
    volume = existing["volume"].astype(float)

    computed = {
        "adv20_dollar_vol": (close * volume).rolling(20, min_periods=1).mean(),
        "return_1d":        close.pct_change(1)  * 100,
        "return_5d":        close.pct_change(5)  * 100,
        "return_20d":       close.pct_change(20) * 100,
        "sma_20":           close.rolling(20,  min_periods=5).mean(),
        "sma_50":           close.rolling(50,  min_periods=10).mean(),
        "sma_200":          close.rolling(200, min_periods=50).mean(),
    }

    for col, series in computed.items():
        if col not in existing.columns:
            existing[col] = pd.NA
        current = existing.at[target_date, col]
        if pd.isna(current) and target_date in series.index and pd.notna(series.loc[target_date]):
            existing.loc[target_date, col] = series.loc[target_date]

    # All other columns (sentiment, events, signals, etc.) are NEVER touched.
    existing = existing.sort_index()
    existing.to_parquet(parquet_path)
    return True


def main() -> int:
    if not FEATURES_DIR.exists():
        print(f"REFRESH ERROR features dir missing: {FEATURES_DIR}", flush=True)
        return 1

    paths = list_us_parquets(FEATURES_DIR)
    if not paths:
        print("REFRESH DONE updated=0 failed=0 duration=0s", flush=True)
        return 0

    sym_to_path = {}
    for p in paths:
        sym = symbol_from_filename(p)
        sym_to_path.setdefault(sym, p)
    symbols = list(sym_to_path.keys())

    start = time.time()
    total_updated = 0
    total_failed  = 0
    batches = [symbols[i:i + BATCH_SIZE] for i in range(0, len(symbols), BATCH_SIZE)]

    for batch_idx, batch in enumerate(batches, start=1):
        batch_updated = 0
        batch_failed  = 0
        try:
            data = yf.download(
                batch,
                period="5d",
                interval="1d",
                progress=False,
                auto_adjust=False,
                group_by="ticker",
                threads=True,
            )
        except Exception as exc:
            print(f"REFRESH batch={batch_idx} download_error={exc!r}", flush=True)
            total_failed += len(batch)
            time.sleep(SLEEP_BETWEEN_BATCHES)
            continue

        for ticker in batch:
            try:
                df = extract_ticker_frame(data, ticker, len(batch))
                if df.empty:
                    batch_failed += 1
                    continue
                ok = update_parquet(sym_to_path[ticker], df)
                if ok:
                    batch_updated += 1
                else:
                    batch_failed += 1
            except Exception as exc:
                batch_failed += 1
                print(f"REFRESH symbol_error ticker={ticker} err={exc!r}", flush=True)

        total_updated += batch_updated
        total_failed  += batch_failed
        print(f"REFRESH batch={batch_idx} updated={batch_updated} failed={batch_failed}", flush=True)
        time.sleep(SLEEP_BETWEEN_BATCHES)

    duration = int(time.time() - start)
    print(f"REFRESH DONE updated={total_updated} failed={total_failed} duration={duration}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
