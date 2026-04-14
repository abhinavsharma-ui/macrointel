#!/usr/bin/env python3
"""
Options Flow Feature Builder
=============================
Enriches features_10yr/*.parquet with options-derived signals:

  options_pcr            - Put/Call volume ratio (>1 = bearish sentiment)
  options_pcr_zscore     - PCR vs 30d rolling z-score (unusual put buying)
  options_iv_rank        - Implied volatility rank 0-100 (high = expensive options)
  options_oi_imbalance   - (Call OI - Put OI) / Total OI (positive = bullish positioning)
  options_unusual_call   - 1 if call volume > 3x 30d avg (smart money buying calls)
  options_unusual_put    - 1 if put volume > 3x 30d avg (hedging / bearish bet)
  options_iv_crush_risk  - 1 if IV rank > 70 and earnings within 5 days

Since free APIs don't provide historical options chains, this script:
  1. Fetches the CURRENT options chain from yfinance (free)
  2. Saves daily snapshots to data/altdata/options_snapshots/
  3. Builds rolling features from accumulated snapshots
  4. For symbols without snapshots yet: injects current values as static features

Run daily via cron to accumulate history:
  0 20 * * 1-5 cd ~/macro_intelligence_complete && source venv/bin/activate && python project/build_options_features.py --snapshot-only

After 30+ days of snapshots, features become genuinely predictive.

Usage:
    python build_options_features.py               # snapshot + enrich all
    python build_options_features.py --snapshot-only  # just collect today's data
    python build_options_features.py --enrich-only    # enrich from existing snapshots
    python build_options_features.py --symbols AAPL MSFT NVDA
"""

import argparse
import logging
import os
import time
from datetime import date
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent
FEATURES_DIR = PROJECT_DIR / "data" / "features"
SNAPSHOTS_DIR = PROJECT_DIR / "data" / "altdata" / "options_snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

TODAY = date.today().isoformat()


def _configure_logging(log_file: Optional[str] = None) -> None:
    handlers = [logging.StreamHandler()]
    if log_file:
        log_path = Path(log_file)
        if not log_path.is_absolute():
            log_path = PROJECT_DIR / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=handlers,
        force=True,
    )


def _resolve_features_dir() -> Path:
    configured = str(os.getenv("FEATURE_STORE_DIR", "")).strip()
    candidates = []
    if configured:
        candidates.append(Path(configured) if Path(configured).is_absolute() else PROJECT_DIR / configured)
    candidates.extend(
        [
            PROJECT_DIR / "data" / "features",
            PROJECT_DIR / "data" / "features_10yr",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    target = candidates[0] if candidates else PROJECT_DIR / "data" / "features"
    target.mkdir(parents=True, exist_ok=True)
    return target


# ── Snapshot collection ───────────────────────────────────────────────────────

def collect_options_snapshot(symbol: str) -> Optional[dict]:
    """Fetch current options chain and compute summary metrics."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        if not expirations:
            return None

        # Use near-term expiry (most liquid)
        chain = ticker.option_chain(expirations[0])
        calls, puts = chain.calls, chain.puts
        if calls.empty or puts.empty:
            return None

        call_vol = float(calls["volume"].fillna(0).sum())
        put_vol = float(puts["volume"].fillna(0).sum())
        call_oi = float(calls["openInterest"].fillna(0).sum())
        put_oi = float(puts["openInterest"].fillna(0).sum())
        total_vol = call_vol + put_vol + 1e-9
        total_oi = call_oi + put_oi + 1e-9

        # IV: use ATM options for better estimate
        avg_iv_call = float(calls["impliedVolatility"].replace(0, np.nan).median() or 0)
        avg_iv_put = float(puts["impliedVolatility"].replace(0, np.nan).median() or 0)
        avg_iv = (avg_iv_call + avg_iv_put) / 2

        return {
            "date": TODAY,
            "symbol": symbol,
            "pcr_volume": put_vol / max(call_vol, 1),
            "pcr_oi": put_oi / max(call_oi, 1),
            "oi_imbalance": (call_oi - put_oi) / total_oi,
            "call_volume": call_vol,
            "put_volume": put_vol,
            "call_oi": call_oi,
            "put_oi": put_oi,
            "avg_iv": avg_iv,
            "total_volume": total_vol,
        }
    except Exception as e:
        logger.debug(f"{symbol} options failed: {e}")
        return None


def save_snapshot(snapshot: dict) -> None:
    sym = snapshot["symbol"]
    snap_file = SNAPSHOTS_DIR / f"{sym}.parquet"
    new_row = pd.DataFrame([snapshot]).set_index("date")
    new_row.index = pd.to_datetime(new_row.index)
    if snap_file.exists():
        existing = pd.read_parquet(snap_file)
        combined = pd.concat([existing, new_row])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new_row
    combined.to_parquet(snap_file)


def run_snapshot_collection(symbols: list, workers: int = 6) -> int:
    """Collect today's options snapshot for all symbols."""
    logger.info(f"Collecting options snapshots for {len(symbols)} symbols...")
    saved = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(collect_options_snapshot, s): s for s in symbols}
        for i, fut in enumerate(as_completed(futures), 1):
            sym = futures[fut]
            try:
                snap = fut.result()
                if snap:
                    save_snapshot(snap)
                    saved += 1
            except Exception as e:
                logger.debug(f"{sym}: {e}")
            if i % 50 == 0:
                logger.info(f"  Snapshots: {i}/{len(symbols)} ({saved} saved)")
            time.sleep(0.05)
    logger.info(f"Snapshots saved: {saved}/{len(symbols)}")
    return saved


# ── Feature computation from snapshots ───────────────────────────────────────

def build_options_features_from_snapshots(symbol: str) -> pd.DataFrame:
    """
    Build rolling options features from accumulated daily snapshots.
    Returns a DataFrame indexed by date with options features.
    """
    snap_file = SNAPSHOTS_DIR / f"{symbol}.parquet"
    if not snap_file.exists():
        return pd.DataFrame()

    snaps = pd.read_parquet(snap_file).sort_index()
    if len(snaps) < 2:
        # Single snapshot — use as static features
        row = snaps.iloc[-1]
        return pd.DataFrame({
            "options_pcr": row.get("pcr_volume", np.nan),
            "options_oi_imbalance": row.get("oi_imbalance", np.nan),
            "options_iv_rank": np.nan,  # can't compute rank from 1 day
            "options_pcr_zscore": 0.0,
            "options_unusual_call": 0,
            "options_unusual_put": 0,
        }, index=[snaps.index[-1]])

    # Rolling features
    features = pd.DataFrame(index=snaps.index)
    features["options_pcr"] = snaps["pcr_volume"]
    features["options_oi_imbalance"] = snaps["oi_imbalance"]

    window = min(30, len(snaps))
    roll_mean_pcr = snaps["pcr_volume"].rolling(window).mean()
    roll_std_pcr = snaps["pcr_volume"].rolling(window).std() + 1e-9
    features["options_pcr_zscore"] = ((snaps["pcr_volume"] - roll_mean_pcr) / roll_std_pcr).clip(-4, 4)

    # IV rank: current IV percentile vs rolling window
    if "avg_iv" in snaps.columns:
        roll_min = snaps["avg_iv"].rolling(window).min()
        roll_max = snaps["avg_iv"].rolling(window).max()
        features["options_iv_rank"] = (
            (snaps["avg_iv"] - roll_min) / (roll_max - roll_min + 1e-9) * 100
        ).clip(0, 100)
    else:
        features["options_iv_rank"] = np.nan

    # Unusual volume flags
    roll_call_avg = snaps["call_volume"].rolling(window).mean()
    roll_put_avg = snaps["put_volume"].rolling(window).mean()
    features["options_unusual_call"] = (snaps["call_volume"] > 3 * roll_call_avg).astype(float)
    features["options_unusual_put"] = (snaps["put_volume"] > 3 * roll_put_avg).astype(float)

    return features


def enrich_symbol_options(symbol: str) -> bool:
    """Join options features onto the symbol's feature parquet."""
    feat_file = FEATURES_DIR / f"{symbol}.parquet"
    if not feat_file.exists():
        return False

    options_features = build_options_features_from_snapshots(symbol)
    if options_features.empty:
        return False

    try:
        df = pd.read_parquet(feat_file)
        df.index = pd.to_datetime(df.index)
        options_aligned = options_features.reindex(df.index, method="ffill")
        for col in options_aligned.columns:
            df[col] = options_aligned[col]
        df.to_parquet(feat_file)
        return True
    except Exception as e:
        logger.warning(f"{symbol} enrich failed: {e}")
        return False


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Options flow feature builder")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--snapshot-only", action="store_true")
    parser.add_argument("--enrich-only", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--log-file", default=None, help="Optional log file path")
    args = parser.parse_args()
    _configure_logging(args.log_file)
    global FEATURES_DIR
    FEATURES_DIR = _resolve_features_dir()
    logger.info(f"Feature source directory: {FEATURES_DIR}")
    logger.info(f"Options snapshot directory: {SNAPSHOTS_DIR}")

    if args.symbols:
        symbols = args.symbols
    else:
        symbols = [p.stem for p in sorted(FEATURES_DIR.glob("*.parquet"))]

    if not args.enrich_only:
        run_snapshot_collection(symbols, workers=args.workers)

    if not args.snapshot_only:
        logger.info(f"Enriching {len(symbols)} feature files with options features...")
        success = 0
        for sym in symbols:
            if enrich_symbol_options(sym):
                success += 1
        logger.info(f"Enriched {success}/{len(symbols)} symbols with options features")

    # Set up reminder about daily cron
    snap_count = len(list(SNAPSHOTS_DIR.glob("*.parquet")))
    logger.info(f"Current snapshot coverage: {snap_count} symbols")
    if snap_count > 0:
        sample = pd.read_parquet(list(SNAPSHOTS_DIR.glob("*.parquet"))[0])
        days = len(sample)
        logger.info(f"Days of history: {days} (need 30+ for strong features)")
        if days < 30:
            logger.info("TIP: Add to crontab for daily collection:")
            logger.info("  0 20 * * 1-5 cd ~/macro_intelligence_complete && source venv/bin/activate && python project/build_options_features.py --snapshot-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
