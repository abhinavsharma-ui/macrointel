"""
Alternative Data Pipeline
=========================
Free-first alternative data collectors that can be joined into the live
feature pipeline. The first implemented factor is an OpenSky airline/travel
activity monitor for travel-sensitive symbols.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests

logger = logging.getLogger(__name__)

OPENSKY_BASE = "https://opensky-network.org/api/states/all"
HEADERS = {
    "User-Agent": "MacroIntel/1.0 contact: research@example.com",
}

TRAVEL_SYMBOL_WEIGHTS = {
    "DAL": 1.00,
    "UAL": 1.00,
    "AAL": 1.00,
    "LUV": 0.95,
    "ALK": 0.90,
    "ABNB": 0.85,
    "UBER": 0.65,
    "LYFT": 0.65,
    "UPS": 0.55,
    "FDX": 0.55,
    "INDIGO.NS": 1.00,
    "IRCTC.NS": 0.80,
    "DELHIVERY.NS": 0.55,
    "RATEGAIN.NS": 0.70,
}

REGION_PARAMS = {
    "global": {},
    "us": {"lamin": 24.0, "lomin": -125.0, "lamax": 49.5, "lomax": -66.0},
    "india": {"lamin": 6.0, "lomin": 68.0, "lamax": 37.5, "lomax": 97.5},
}


class OpenSkyTravelFactorPipeline:
    def __init__(self, history_path: str = "data/altdata/opensky_history.csv"):
        self.history_path = Path(history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

    def _fetch_region_count(self, region: str) -> int:
        params = REGION_PARAMS.get(region, {})
        try:
            resp = requests.get(OPENSKY_BASE, params=params, timeout=20, headers=HEADERS)
            resp.raise_for_status()
            payload = resp.json()
            states = payload.get("states") or []
            return int(len(states))
        except Exception as exc:
            logger.warning(f"OpenSky {region} fetch failed: {exc}")
            return 0

    def _load_history(self) -> pd.DataFrame:
        if not self.history_path.exists():
            return pd.DataFrame(columns=["timestamp", "global_count", "us_count", "india_count"])
        try:
            frame = pd.read_csv(self.history_path, parse_dates=["timestamp"])
            return frame.sort_values("timestamp")
        except Exception as exc:
            logger.warning(f"OpenSky history load failed: {exc}")
            return pd.DataFrame(columns=["timestamp", "global_count", "us_count", "india_count"])

    def _persist_history(self, frame: pd.DataFrame):
        frame.tail(720).to_csv(self.history_path, index=False)

    def _compute_signal(self, history: pd.DataFrame, market: str, latest_count: int) -> Dict[str, float]:
        if history.empty:
            return {"travel_activity_level": 0.0, "travel_activity_change": 0.0}

        count_col = "india_count" if market == "nse" else "us_count"
        recent = history[count_col].tail(48)
        baseline = float(recent.mean()) if not recent.empty else 0.0
        latest_prev = float(history[count_col].iloc[-1]) if len(history) else 0.0

        if baseline <= 0:
            level = 0.0
        else:
            level = max(-1.0, min(1.0, (latest_count - baseline) / max(baseline, 1.0)))

        if latest_prev <= 0:
            change = 0.0
        else:
            change = max(-1.0, min(1.0, (latest_count - latest_prev) / max(latest_prev, 1.0)))

        return {
            "travel_activity_level": round(level, 4),
            "travel_activity_change": round(change, 4),
        }

    def run(self, symbols: List[str]) -> Dict:
        timestamp = datetime.now(timezone.utc)
        global_count = self._fetch_region_count("global")
        us_count = self._fetch_region_count("us")
        india_count = self._fetch_region_count("india")

        history = self._load_history()
        updated_history = pd.concat(
            [
                history,
                pd.DataFrame(
                    [
                        {
                            "timestamp": timestamp,
                            "global_count": global_count,
                            "us_count": us_count,
                            "india_count": india_count,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        self._persist_history(updated_history)

        records = []
        for symbol in symbols:
            weight = TRAVEL_SYMBOL_WEIGHTS.get(symbol, 0.0)
            if weight <= 0:
                continue
            market = "nse" if symbol.endswith(".NS") else "us"
            signal = self._compute_signal(updated_history.iloc[:-1], market=market, latest_count=india_count if market == "nse" else us_count)
            records.append(
                {
                    "symbol": symbol,
                    "date": pd.Timestamp(timestamp.date()),
                    "travel_activity_level": signal["travel_activity_level"] * weight,
                    "travel_activity_change": signal["travel_activity_change"] * weight,
                }
            )

        if records:
            daily = pd.DataFrame(records).set_index(["symbol", "date"]).sort_index()
        else:
            daily = pd.DataFrame(
                columns=["travel_activity_level", "travel_activity_change"],
                index=pd.MultiIndex.from_tuples([], names=["symbol", "date"]),
            )

        summary = {
            "timestamp": timestamp.isoformat(),
            "global_count": global_count,
            "us_count": us_count,
            "india_count": india_count,
            "symbols_covered": len(records),
        }
        logger.info(
            "OpenSky alt data: "
            f"global={global_count} us={us_count} india={india_count} symbols={len(records)}"
        )
        return {
            "symbol_altdata_daily": daily,
            "summary": summary,
        }
