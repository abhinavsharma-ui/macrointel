from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

DEFAULT_PATH = Path("models/checkpoints/ohlcv_selector.pkl")
_SELECTOR = None

class OhlcvSelector:
    def __init__(self, path: str | Path = DEFAULT_PATH):
        self.path = Path(path)
        with self.path.open("rb") as f:
            payload = pickle.load(f)
        self.model = payload["model"]
        self.features = list(payload["features"])
        self.threshold = float(payload["threshold"])
        self.validation = payload.get("validation", {})
        self.label = payload.get("label", {})

    def vector_from(self, row: Any, signal: Optional[Dict[str, Any]] = None):
        signal = signal or {}
        values = {}
        missing = []
        for f in self.features:
            v = None
            if row is not None:
                try:
                    v = row.get(f) if hasattr(row, "get") else None
                except Exception:
                    v = None
            if v is None:
                aliases = {
                    "return_20d": ["momentum_20d"],
                    "return_60d": ["momentum_60d"],
                    "close_vs_sma_50": ["close_sma_50", "sma_50_gap"],
                    "close_vs_sma_200": ["close_sma_200", "sma_200_gap"],
                }.get(f, [])
                for alt in aliases:
                    if row is not None:
                        try:
                            v = row.get(alt) if hasattr(row, "get") else None
                        except Exception:
                            v = None
                    if v is None:
                        v = signal.get(alt)
                    if v is not None:
                        break
            if v is None:
                v = signal.get(f)
            try:
                values[f] = float(v)
            except Exception:
                values[f] = 0.0
                missing.append(f)
        x = pd.DataFrame([values], columns=self.features).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return x, missing

    def score(self, symbol: str, row: Any, signal: Optional[Dict[str, Any]] = None, threshold: Optional[float] = None) -> Dict[str, Any]:
        x, missing = self.vector_from(row, signal)
        prob = float(self.model.predict_proba(x)[0, 1])
        th = float(self.threshold if threshold is None else threshold)
        return {
            "symbol": str(symbol),
            "probability": round(prob, 6),
            "threshold": round(th, 6),
            "allow": bool(prob >= th and not missing),
            "missing_features": missing,
            "feature_count": len(self.features),
            "artifact": str(self.path),
        }

def get_ohlcv_selector(path: str | Path = DEFAULT_PATH) -> OhlcvSelector:
    global _SELECTOR
    p = Path(path)
    if _SELECTOR is None or _SELECTOR.path != p:
        _SELECTOR = OhlcvSelector(p)
    return _SELECTOR
