#!/usr/bin/env python3
"""
FRED Macro Feature Builder
==========================
Fetches macroeconomic regime features from the Federal Reserve (FRED API)
and joins them onto every symbol's feature parquet file.

These are regime/context features — they tell the model WHEN technical
signals are reliable vs noisy (e.g. RSI means something different in a
credit crisis vs calm bull market).

Series fetched (all free, no API key needed):
  VIX        - VIXCLS        - Market fear gauge
  Fed Funds  - FEDFUNDS      - Overnight rate
  10yr Yield - DGS10         - Long-term rate
  2yr Yield  - DGS2          - Short-term rate
  Yield Curve- T10Y2Y        - 10yr minus 2yr (inversion = recession signal)
  Corp Spread- BAA10Y        - Baa minus 10yr (credit stress)
  USD Index  - DTWEXBGS      - Broad USD index
  Inflation  - CPIAUCSL      - CPI (monthly, ffilled)
  Unemp Rate - UNRATE        - Unemployment (monthly, ffilled)
  M2 Supply  - M2SL          - Money supply (monthly, ffilled)

Usage:
    python build_fred_features.py              # enrich all symbols
    python build_fred_features.py --symbols AAPL MSFT
    python build_fred_features.py --fetch-only  # just download FRED data

Requirements:
    pip install pandas-datareader pandas pyarrow
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent
FEATURES_DIR = PROJECT_DIR / "data" / "features_10yr"
MACRO_CACHE = PROJECT_DIR / "data" / "altdata" / "fred_macro.parquet"
MACRO_CACHE.parent.mkdir(parents=True, exist_ok=True)

# FRED series to fetch (series_id → feature_name)
FRED_SERIES = {
    "VIXCLS":    "macro_vix",
    "FEDFUNDS":  "macro_fed_rate",
    "DGS10":     "macro_10yr_yield",
    "DGS2":      "macro_2yr_yield",
    "T10Y2Y":    "macro_yield_curve",   # positive = normal, negative = inverted
    "BAA10Y":    "macro_credit_spread",
    "DTWEXBGS":  "macro_usd_index",
    "CPIAUCSL":  "macro_cpi",
    "UNRATE":    "macro_unemployment",
    "M2SL":      "macro_m2",
}

# Derived features computed after fetching
DERIVED_FEATURES = [
    "macro_vix_regime",          # 0=calm(<15), 1=normal(15-25), 2=stressed(25-35), 3=crisis(35+)
    "macro_yield_curve_inverted",# 1 if inverted (recession risk)
    "macro_rate_rising",         # 1 if fed rate risen >0.25% in last 3mo
    "macro_credit_stress",       # 1 if credit spread > 2.0
    "macro_vix_zscore_1yr",      # VIX vs 1yr rolling mean (normalized fear)
    "macro_10yr_change_30d",     # rate change direction
]


# ── FRED fetcher ─────────────────────────────────────────────────────────────

def fetch_fred_series(series_id: str, start: str = "2010-01-01") -> Optional[pd.Series]:
    """Fetch a single FRED series using pandas_datareader."""
    try:
        import pandas_datareader.data as web
        s = web.DataReader(series_id, "fred", start=start)
        return s.iloc[:, 0].rename(series_id)
    except ImportError:
        # Fallback: direct FRED API (no key needed for public series)
        import requests
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&vintage_date="
        try:
            r = requests.get(
                f"https://fred.stlouisfed.org/graph/fredgraph.csv",
                params={"id": series_id},
                timeout=15
            )
            if r.status_code != 200:
                logger.warning(f"FRED {series_id}: HTTP {r.status_code}")
                return None
            from io import StringIO
            df = pd.read_csv(StringIO(r.text), index_col=0, parse_dates=True)
            df.columns = [series_id]
            s = df.iloc[:, 0]
            s = s[s.index >= pd.Timestamp(start)]
            s = pd.to_numeric(s, errors="coerce")
            return s
        except Exception as e:
            logger.warning(f"FRED {series_id} direct fetch failed: {e}")
            return None
    except Exception as e:
        logger.warning(f"pandas_datareader {series_id}: {e}")
        return None


def build_macro_df(start: str = "2010-01-01", force_refresh: bool = False) -> pd.DataFrame:
    """
    Download all FRED series, align to daily frequency, compute derived features.
    Caches result to MACRO_CACHE parquet.
    """
    if MACRO_CACHE.exists() and not force_refresh:
        cache_age = (pd.Timestamp.now() - pd.Timestamp(MACRO_CACHE.stat().st_mtime, unit="s"))
        if cache_age.days < 1:
            logger.info("Using cached FRED data (< 1 day old)")
            return pd.read_parquet(MACRO_CACHE)

    logger.info("Fetching FRED macro data...")
    series_dict = {}
    for series_id, feature_name in FRED_SERIES.items():
        logger.info(f"  Fetching {series_id} → {feature_name}")
        s = fetch_fred_series(series_id, start=start)
        if s is not None:
            series_dict[feature_name] = s
        else:
            logger.warning(f"  {series_id}: failed, will be NaN")
        time.sleep(0.2)  # be polite to FRED

    if not series_dict:
        logger.error("No FRED data fetched at all!")
        return pd.DataFrame()

    # Combine into daily dataframe
    macro = pd.DataFrame(series_dict)
    macro.index = pd.to_datetime(macro.index)
    macro = macro.sort_index()

    # Daily business day index
    bday_idx = pd.bdate_range(start=macro.index[0], end=pd.Timestamp.today())
    macro = macro.reindex(bday_idx)

    # Forward-fill (FRED publishes monthly/weekly, ffill to daily)
    macro = macro.ffill().bfill()

    # ── Derived features ──────────────────────────────────────────────────
    if "macro_vix" in macro.columns:
        macro["macro_vix_regime"] = pd.cut(
            macro["macro_vix"],
            bins=[0, 15, 25, 35, 999],
            labels=[0, 1, 2, 3],
        ).astype(float)
        # VIX z-score vs trailing 252 days
        macro["macro_vix_zscore_1yr"] = (
            (macro["macro_vix"] - macro["macro_vix"].rolling(252).mean())
            / (macro["macro_vix"].rolling(252).std() + 1e-9)
        ).clip(-4, 4)

    if "macro_yield_curve" in macro.columns:
        macro["macro_yield_curve_inverted"] = (macro["macro_yield_curve"] < 0).astype(float)

    if "macro_fed_rate" in macro.columns:
        fed_3mo_ago = macro["macro_fed_rate"].shift(63)  # ~3 months
        macro["macro_rate_rising"] = (
            (macro["macro_fed_rate"] - fed_3mo_ago) > 0.25
        ).astype(float)

    if "macro_credit_spread" in macro.columns:
        macro["macro_credit_stress"] = (macro["macro_credit_spread"] > 2.0).astype(float)

    if "macro_10yr_yield" in macro.columns:
        macro["macro_10yr_change_30d"] = macro["macro_10yr_yield"].diff(21)

    # Normalise continuous series (z-score vs 2yr rolling window)
    continuous = [
        "macro_vix", "macro_fed_rate", "macro_10yr_yield", "macro_2yr_yield",
        "macro_credit_spread", "macro_usd_index", "macro_cpi",
        "macro_unemployment", "macro_m2",
    ]
    for col in continuous:
        if col in macro.columns:
            roll_mean = macro[col].rolling(504).mean()  # ~2yr
            roll_std = macro[col].rolling(504).std() + 1e-9
            macro[f"{col}_zscore"] = ((macro[col] - roll_mean) / roll_std).clip(-4, 4)

    macro = macro.fillna(method="ffill").fillna(0)
    macro.to_parquet(MACRO_CACHE)
    logger.info(f"FRED macro data built: {len(macro)} rows × {len(macro.columns)} features")
    logger.info(f"Cached to {MACRO_CACHE}")
    return macro


# ── Enrichment ───────────────────────────────────────────────────────────────

def enrich_symbol(symbol: str, macro_df: pd.DataFrame) -> bool:
    """Join macro features onto a symbol's feature parquet."""
    parquet_path = FEATURES_DIR / f"{symbol}.parquet"
    if not parquet_path.exists():
        return False

    try:
        df = pd.read_parquet(parquet_path)
        df.index = pd.to_datetime(df.index)
    except Exception as e:
        logger.warning(f"{symbol}: read failed — {e}")
        return False

    # Join macro on date index
    macro_aligned = macro_df.reindex(df.index, method="ffill")
    macro_cols = [c for c in macro_df.columns if c not in df.columns]
    if not macro_cols:
        logger.debug(f"{symbol}: macro features already present")
        return True

    for col in macro_cols:
        df[col] = macro_aligned[col]

    df.to_parquet(parquet_path)
    return True


def main():
    parser = argparse.ArgumentParser(description="Enrich features with FRED macro data")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--fetch-only", action="store_true",
                        help="Only download/cache FRED data, don't enrich parquets")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Force re-download even if cache is fresh")
    args = parser.parse_args()

    # Step 1: Build macro dataframe
    macro_df = build_macro_df(force_refresh=args.force_refresh)
    if macro_df.empty:
        logger.error("Failed to build macro dataframe. Check internet connection.")
        return 1

    logger.info(f"Macro features: {list(macro_df.columns)}")

    if args.fetch_only:
        logger.info("--fetch-only: done.")
        return 0

    # Step 2: Enrich all symbol parquets
    if args.symbols:
        symbols = args.symbols
    else:
        symbols = [p.stem for p in sorted(FEATURES_DIR.glob("*.parquet"))]

    logger.info(f"Enriching {len(symbols)} symbols with {len(macro_df.columns)} macro features")

    success = 0
    for i, sym in enumerate(symbols, 1):
        ok = enrich_symbol(sym, macro_df)
        if ok:
            success += 1
        if i % 50 == 0:
            logger.info(f"Progress: {i}/{len(symbols)}")

    logger.info(f"Done — {success}/{len(symbols)} symbols enriched with macro features")
    logger.info("Run retrain now: python retrain_institutional_models.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
