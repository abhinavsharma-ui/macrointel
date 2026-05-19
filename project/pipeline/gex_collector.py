from __future__ import annotations
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
GEX_HISTORY_PATH = Path("data/altdata/gex_history.csv")
CACHE_TTL_SECONDS = 86400
GEX_FEATURE_COLUMNS = ["gex_zscore_rolling_20d","distance_from_zero_gamma","dix_accumulation_trend","microstructure_edge_composite"]

class GEXCollector:
    def __init__(self, history_path=str(GEX_HISTORY_PATH)):
        self.history_path = Path(history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache = None
        self._cache_time = None

    def _is_cache_valid(self):
        if self._cache is None or self._cache_time is None:
            return False
        return (datetime.utcnow() - self._cache_time).total_seconds() < CACHE_TTL_SECONDS

    def _load_history(self):
        if not self.history_path.exists():
            return pd.DataFrame(columns=["date","dix","gex"])
        try:
            df = pd.read_csv(self.history_path, parse_dates=["date"])
            return df.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)
        except Exception as exc:
            logger.warning(f"GEX history load failed: {exc}")
            return pd.DataFrame(columns=["date","dix","gex"])

    def _persist_history(self, df):
        try:
            df.to_csv(self.history_path, index=False)
        except Exception as exc:
            logger.warning(f"GEX history save failed: {exc}")

    def _fetch_gex_from_options(self):
        """Compute GEX from SPY options chain via yfinance. More accurate than Squeezemetrics."""
        try:
            import yfinance as yf
            spy = yf.Ticker("SPY")
            spot = spy.fast_info.last_price
            if not spot or spot <= 0:
                logger.warning("GEX: could not get SPY spot price")
                return None

            total_gex = 0.0
            expiries = spy.options[:8]  # first 8 expiries capture most open interest
            for expiry in expiries:
                try:
                    chain = spy.option_chain(expiry)
                    calls = chain.calls.dropna(subset=["gamma","openInterest"])
                    puts  = chain.puts.dropna(subset=["gamma","openInterest"])
                    call_gex = (calls["gamma"] * calls["openInterest"] * 100 * spot**2 * 0.01).sum()
                    put_gex  = (puts["gamma"]  * puts["openInterest"]  * 100 * spot**2 * 0.01).sum()
                    total_gex += call_gex - put_gex
                except Exception:
                    continue

            gex_billions = total_gex / 1e9
            dix = self._fetch_dix_proxy()
            today = pd.Timestamp(datetime.utcnow().date())
            logger.info(f"GEX computed: {gex_billions:.2f}B, DIX proxy: {dix:.3f}")
            return pd.DataFrame([{"date": today, "gex": gex_billions, "dix": dix}])
        except Exception as exc:
            logger.warning(f"yfinance GEX compute failed: {exc}")
            return None

    def _fetch_dix_proxy(self) -> float:
        """
        DIX proxy using FINRA RegSHO short volume for SPY.
        High short volume in lit markets inversely tracks dark pool accumulation.
        Falls back to neutral 0.45 on failure.
        """
        try:
            import requests
            date = datetime.utcnow().date()
            # try today and last 3 trading days in case of weekend/holiday
            for delta in range(4):
                d = date - timedelta(days=delta)
                url = f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d.strftime('%Y%m%d')}.txt"
                resp = requests.get(url, timeout=10)
                if resp.status_code != 200:
                    continue
                lines = resp.text.strip().split("\n")
                for line in lines:
                    parts = line.split("|")
                    if len(parts) >= 4 and parts[0].strip() == "SPY":
                        short_vol = float(parts[1])
                        total_vol = float(parts[2])
                        if total_vol > 0:
                            ratio = short_vol / total_vol
                            # invert: high short vol on lit markets = more dark pool buying
                            dix_proxy = float(np.clip(1.0 - ratio + 0.10, 0.30, 0.70))
                            return dix_proxy
            return 0.45
        except Exception:
            return 0.45

    def _compute_features(self, df):
        result = df.copy().set_index("date").sort_index()
        gex_mean = result["gex"].rolling(20, min_periods=5).mean()
        gex_std  = result["gex"].rolling(20, min_periods=5).std().replace(0, np.nan)
        result["gex_zscore_rolling_20d"] = ((result["gex"] - gex_mean) / gex_std).clip(-3.0, 3.0).fillna(0.0)
        zero_gamma_proxy = result["gex"].rolling(60, min_periods=20).quantile(0.10)
        gex_abs_mean = result["gex"].abs().rolling(60, min_periods=20).mean().replace(0, np.nan)
        result["distance_from_zero_gamma"] = ((result["gex"] - zero_gamma_proxy) / gex_abs_mean).clip(-2.0, 2.0).fillna(0.0)
        dix_threshold = 0.45 if result["dix"].max() < 2.0 else 45.0
        result["dix_accumulation_trend"] = (result["dix"].rolling(5, min_periods=3).mean() > dix_threshold).astype(float)
        dix_centered = result["dix_accumulation_trend"] * 2.0 - 1.0
        result["microstructure_edge_composite"] = (
            0.45 * result["gex_zscore_rolling_20d"].clip(-1, 1)
            + 0.35 * result["distance_from_zero_gamma"].clip(-1, 1)
            + 0.20 * dix_centered
        ).clip(-1.0, 1.0).fillna(0.0)
        return result[["gex","dix","gex_zscore_rolling_20d","distance_from_zero_gamma","dix_accumulation_trend","microstructure_edge_composite"]]

    def run(self):
        if self._is_cache_valid():
            return self._cache
        history = self._load_history()
        needs_fetch = True
        if not history.empty:
            last_date = pd.Timestamp(history["date"].max())
            cutoff = pd.Timestamp(datetime.utcnow().date() - timedelta(days=1))
            if last_date >= cutoff:
                needs_fetch = False
        if needs_fetch:
            fresh = self._fetch_gex_from_options()
            if fresh is not None:
                history = pd.concat([history, fresh], ignore_index=True).drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
                self._persist_history(history)
            else:
                logger.warning("GEX fetch failed — falling back to cached history")
        if history.empty:
            logger.error("GEX: no data available, returning None")
            return None
        features = self._compute_features(history)
        self._cache = features
        self._cache_time = datetime.utcnow()
        return features

    def get_latest(self):
        df = self.run()
        if df is None or df.empty:
            return {col: 0.0 for col in GEX_FEATURE_COLUMNS}
        row = df.iloc[-1]
        return {col: float(row.get(col, 0.0)) for col in GEX_FEATURE_COLUMNS}
