#!/usr/bin/env python3
"""
Earnings Drift + Insider Transaction Feature Builder
=====================================================
Two of the strongest documented alpha factors in academic literature:

PRE-EARNINGS DRIFT features (per trading day):
  earnings_days_to_next     - calendar days until next earnings (0-90, capped)
  earnings_pre_drift_zone   - 1 if within 5 trading days of earnings (historically drifts)
  earnings_historical_move  - avg abs % move on earnings day (last 8 quarters)
  earnings_beat_rate        - fraction of last 8 quarters that beat estimates
  earnings_surprise_momentum- avg surprise pct over last 3 quarters (trend)
  earnings_season           - 1 if current week is peak earnings season

INSIDER TRANSACTION features (rolling windows):
  insider_net_ratio_30d     - (buy_value - sell_value) / total_value, last 30d
  insider_net_ratio_90d     - same for 90 days
  insider_buy_count_30d     - number of insider buy transactions in 30d
  insider_cluster_buy       - 1 if 3+ insiders bought in last 30d (strong signal)
  insider_ceo_bought        - 1 if CEO/CFO specifically bought in last 90d
  insider_sell_pressure     - 1 if sell value > 5x buy value in 30d (warning)

Usage:
    python build_earnings_insider_features.py              # all symbols
    python build_earnings_insider_features.py --symbols AAPL MSFT
    python build_earnings_insider_features.py --skip-insider  # earnings only
    python build_earnings_insider_features.py --skip-earnings # insider only
"""

import argparse
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent
FEATURES_DIR = PROJECT_DIR / "data" / "features"


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

def _load_env() -> dict:
    env = {}
    env_path = PROJECT_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env

_ENV = _load_env()
FINNHUB_KEY = (_ENV.get("FINNHUB_API_KEYS", "") or _ENV.get("FINNHUB_API_KEY", "")).split(",")[0].strip()
BASE_URL = "https://finnhub.io/api/v1"
RATE_DELAY = 0.25


def _get(endpoint: str, params: dict, retries: int = 3) -> Optional[dict]:
    params["token"] = FINNHUB_KEY
    for attempt in range(retries):
        try:
            r = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=10)
            if r.status_code == 429:
                logger.warning("Rate limited — sleeping 60s")
                time.sleep(60)
                continue
            if r.status_code != 200:
                return None
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                logger.debug(f"{endpoint}: {e}")
            time.sleep(1)
    return None


# ── Earnings features ─────────────────────────────────────────────────────────

def build_earnings_features(symbol: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Build per-day earnings drift features using Finnhub historical earnings.
    """
    # Fetch historical earnings (up to 40 quarters)
    data = _get("stock/earnings", {"symbol": symbol, "limit": 40})
    time.sleep(RATE_DELAY)

    if not data:
        return pd.DataFrame(index=index)

    # Parse earnings dates and metrics
    earnings_records = []
    for q in data:
        try:
            date_str = q.get("period") or q.get("date")
            if not date_str:
                continue
            actual = float(q.get("actual") or 0)
            estimate = float(q.get("estimate") or 0)
            surprise_pct = (actual - estimate) / (abs(estimate) + 1e-9)
            beat = 1 if surprise_pct > 0.01 else (-1 if surprise_pct < -0.01 else 0)
            earnings_records.append({
                "date": pd.to_datetime(date_str),
                "actual": actual,
                "estimate": estimate,
                "surprise_pct": np.clip(surprise_pct, -5, 5),
                "beat": beat,
            })
        except Exception:
            continue

    if not earnings_records:
        return pd.DataFrame(index=index)

    eq = pd.DataFrame(earnings_records).sort_values("date")
    earnings_dates = pd.DatetimeIndex(eq["date"])

    features = pd.DataFrame(index=index)
    days_to_next = np.full(len(index), 90.0)
    pre_drift_zone = np.zeros(len(index))
    hist_move = np.full(len(index), np.nan)
    beat_rate = np.full(len(index), np.nan)
    surprise_momentum = np.full(len(index), np.nan)
    earnings_season = np.zeros(len(index))

    # For each trading day, compute earnings-based features
    for i, dt in enumerate(index):
        # Days to next earnings
        future_earnings = earnings_dates[earnings_dates > dt]
        if len(future_earnings) > 0:
            days_to_next[i] = float((future_earnings[0] - dt).days)
        else:
            days_to_next[i] = 90.0

        # Pre-drift zone: within 5 trading days (~7 calendar days) of earnings
        if days_to_next[i] <= 7:
            pre_drift_zone[i] = 1.0

        # Historical earnings stats from past 8 quarters before this date
        past_earnings = eq[eq["date"] < dt].tail(8)
        if len(past_earnings) >= 2:
            beat_rate[i] = past_earnings["beat"].apply(lambda x: 1 if x == 1 else 0).mean()
            surprise_momentum[i] = past_earnings.tail(3)["surprise_pct"].mean()

        # Earnings season: Jan/Feb, Apr/May, Jul/Aug, Oct/Nov
        month = dt.month
        if month in (1, 2, 4, 5, 7, 8, 10, 11):
            earnings_season[i] = 1.0

    features["earnings_days_to_next"] = np.clip(days_to_next, 0, 90)
    features["earnings_pre_drift_zone"] = pre_drift_zone
    features["earnings_beat_rate"] = beat_rate
    features["earnings_surprise_momentum"] = surprise_momentum
    features["earnings_season"] = earnings_season

    # Normalize days_to_next
    features["earnings_proximity_score"] = np.where(
        days_to_next < 90,
        1.0 - (days_to_next / 90.0),
        0.0,
    )

    return features


# ── Insider features ──────────────────────────────────────────────────────────

# C-suite title keywords → higher weight signals
EXEC_TITLES = {"ceo", "cfo", "coo", "president", "chairman", "chief"}

def _is_exec(name_or_title: str) -> bool:
    if not name_or_title:
        return False
    s = name_or_title.lower()
    return any(t in s for t in EXEC_TITLES)


def build_insider_features(symbol: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Build rolling insider transaction features using Finnhub insider data.
    """
    data = _get("stock/insider-transactions", {"symbol": symbol})
    time.sleep(RATE_DELAY)

    if not data or "data" not in data:
        return pd.DataFrame(index=index)

    rows = []
    for tx in data["data"]:
        try:
            dt = pd.to_datetime(tx.get("filingDate") or tx.get("transactionDate"))
            tx_type = str(tx.get("transactionCode") or "").upper()
            shares = float(tx.get("share") or 0)
            price = float(tx.get("price") or 0)
            value = abs(shares * price)
            name = str(tx.get("name") or "")
            title = str(tx.get("officerTitle") or "")
            is_exec = _is_exec(name) or _is_exec(title)

            # P = purchase (buy), S = sale (sell)
            is_buy = tx_type in ("P", "A")  # Purchase or Award
            is_sell = tx_type in ("S", "D")  # Sale or Disposition

            if not (is_buy or is_sell) or value < 1000:
                continue

            rows.append({
                "date": dt,
                "is_buy": int(is_buy),
                "is_sell": int(is_sell),
                "value": value,
                "is_exec": int(is_exec),
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(index=index)

    tx_df = pd.DataFrame(rows).sort_values("date")
    tx_df = tx_df.set_index("date")
    tx_df.index = pd.to_datetime(tx_df.index)

    features = pd.DataFrame(index=index)
    net_ratio_30 = np.zeros(len(index))
    net_ratio_90 = np.zeros(len(index))
    buy_count_30 = np.zeros(len(index))
    cluster_buy = np.zeros(len(index))
    exec_bought = np.zeros(len(index))
    sell_pressure = np.zeros(len(index))

    for i, dt in enumerate(index):
        for window_days, arr_net, arr_buys in [
            (30, net_ratio_30, buy_count_30),
            (90, net_ratio_90, None),
        ]:
            start_dt = dt - pd.Timedelta(days=window_days)
            window_tx = tx_df[(tx_df.index >= start_dt) & (tx_df.index <= dt)]
            if window_tx.empty:
                continue

            buy_val = window_tx[window_tx["is_buy"] == 1]["value"].sum()
            sell_val = window_tx[window_tx["is_sell"] == 1]["value"].sum()
            total_val = buy_val + sell_val + 1e-9

            arr_net[i] = np.clip((buy_val - sell_val) / total_val, -1, 1)
            if arr_buys is not None:
                arr_buys[i] = float(window_tx["is_buy"].sum())

            if window_days == 30:
                n_buyers = window_tx[window_tx["is_buy"] == 1].shape[0]
                cluster_buy[i] = 1.0 if n_buyers >= 3 else 0.0
                sell_pressure[i] = 1.0 if sell_val > 5 * buy_val + 1 else 0.0

        # C-suite buy in 90d
        start_90 = dt - pd.Timedelta(days=90)
        exec_window = tx_df[
            (tx_df.index >= start_90) & (tx_df.index <= dt) &
            (tx_df["is_buy"] == 1) & (tx_df["is_exec"] == 1)
        ]
        exec_bought[i] = 1.0 if len(exec_window) > 0 else 0.0

    features["insider_net_ratio_30d"] = net_ratio_30
    features["insider_net_ratio_90d"] = net_ratio_90
    features["insider_buy_count_30d"] = buy_count_30
    features["insider_cluster_buy"] = cluster_buy
    features["insider_ceo_bought"] = exec_bought
    features["insider_sell_pressure"] = sell_pressure

    return features


# ── Enrichment ────────────────────────────────────────────────────────────────

def enrich_symbol(symbol: str, skip_earnings: bool = False, skip_insider: bool = False) -> bool:
    feat_file = FEATURES_DIR / f"{symbol}.parquet"
    if not feat_file.exists():
        return False

    try:
        df = pd.read_parquet(feat_file)
        df.index = pd.to_datetime(df.index)
    except Exception as e:
        logger.warning(f"{symbol}: read error — {e}")
        return False

    added = 0

    if not skip_earnings:
        ef = build_earnings_features(symbol, df.index)
        for col in ef.columns:
            if col not in df.columns or df[col].isna().all():
                df[col] = ef[col]
                added += 1

    if not skip_insider:
        inf = build_insider_features(symbol, df.index)
        for col in inf.columns:
            if col not in df.columns or df[col].isna().all():
                df[col] = inf[col]
                added += 1

    if added > 0:
        df.to_parquet(feat_file)
        logger.info(f"{symbol}: +{added} features")
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=2)  # low for Finnhub rate limits
    parser.add_argument("--skip-earnings", action="store_true")
    parser.add_argument("--skip-insider", action="store_true")
    parser.add_argument("--log-file", default=None, help="Optional log file path")
    args = parser.parse_args()
    _configure_logging(args.log_file)
    global FEATURES_DIR
    FEATURES_DIR = _resolve_features_dir()
    logger.info(f"Feature source directory: {FEATURES_DIR}")

    if not FINNHUB_KEY:
        logger.error("FINNHUB_API_KEYS not set in project/.env")
        return 1

    symbols = args.symbols or [p.stem for p in sorted(FEATURES_DIR.glob("*.parquet"))]
    logger.info(f"Processing {len(symbols)} symbols — earnings={not args.skip_earnings}, insider={not args.skip_insider}")

    success = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(enrich_symbol, s, args.skip_earnings, args.skip_insider): s
            for s in symbols
        }
        for i, fut in enumerate(as_completed(futures), 1):
            sym = futures[fut]
            try:
                if fut.result():
                    success += 1
            except Exception as e:
                logger.debug(f"{sym}: {e}")
            if i % 25 == 0:
                logger.info(f"Progress: {i}/{len(symbols)} — {success} enriched")

    logger.info(f"Done — {success}/{len(symbols)} symbols enriched")
    logger.info("Ready to retrain: python retrain_institutional_models.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
