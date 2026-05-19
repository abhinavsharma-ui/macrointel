from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(os.getenv("PRICE_26YR_SOURCE", "/home/abhinavsharma1359/project/project/data/prices_10yr"))
OUT = Path(os.getenv("FEATURE_26YR_OUT", "data/features_26yr"))
WORKERS = int(os.getenv("FEATURE_26YR_WORKERS", "24") or 24)

def pick_col(df, names):
    lower = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None

def normalize_ohlcv(df):
    frame = df.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        for c in ("timestamp", "date", "datetime", "time"):
            if c in frame.columns:
                frame.index = pd.to_datetime(frame[c], errors="coerce", utc=True)
                break
    else:
        frame.index = pd.to_datetime(frame.index, errors="coerce", utc=True)
    frame = frame[~pd.isna(frame.index)].sort_index()
    try:
        frame.index = frame.index.tz_convert(None)
    except Exception:
        try:
            frame.index = frame.index.tz_localize(None)
        except Exception:
            pass

    mapping = {
        "open": pick_col(frame, ["open", "Open", "o"]),
        "high": pick_col(frame, ["high", "High", "h"]),
        "low": pick_col(frame, ["low", "Low", "l"]),
        "close": pick_col(frame, ["close", "Close", "adj_close", "Adj Close", "c"]),
        "volume": pick_col(frame, ["volume", "Volume", "v"]),
    }
    out = pd.DataFrame(index=frame.index)
    for k, c in mapping.items():
        out[k] = pd.to_numeric(frame[c], errors="coerce") if c is not None else np.nan
    out["close"] = out["close"].ffill()
    out["open"] = out["open"].fillna(out["close"])
    out["high"] = out["high"].fillna(out[["open","close"]].max(axis=1))
    out["low"] = out["low"].fillna(out[["open","close"]].min(axis=1))
    out["volume"] = out["volume"].fillna(0.0)
    out = out.dropna(subset=["close"])
    out = out[out["close"] > 0]
    return out

def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def build_one(path_str):
    path = Path(path_str)
    sym = path.stem
    try:
        raw = pd.read_parquet(path)
        df = normalize_ohlcv(raw)
        if len(df) < 260:
            return sym, 0, "short"

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        ret = close.pct_change()

        feat = df.copy()
        feat["return_1d"] = ret.fillna(0)
        feat["return_5d"] = close.pct_change(5).fillna(0)
        feat["return_20d"] = close.pct_change(20).fillna(0)
        feat["momentum_20d"] = feat["return_20d"]
        feat["momentum_60d"] = close.pct_change(60).fillna(0)
        feat["momentum_composite"] = (0.5*feat["momentum_20d"] + 0.3*feat["momentum_60d"] + 0.2*close.pct_change(120).fillna(0))
        feat["sma_20"] = close.rolling(20).mean()
        feat["sma_50"] = close.rolling(50).mean()
        feat["sma_200"] = close.rolling(200).mean()
        feat["close_vs_sma_50"] = (close / feat["sma_50"] - 1).replace([np.inf,-np.inf], np.nan).fillna(0)
        feat["close_vs_sma_200"] = (close / feat["sma_200"] - 1).replace([np.inf,-np.inf], np.nan).fillna(0)
        feat["price_acceleration"] = feat["momentum_20d"] - feat["momentum_60d"]
        feat["realized_vol_21d"] = ret.rolling(21).std().fillna(0) * np.sqrt(252)
        feat["hist_vol_30"] = ret.rolling(30).std().fillna(0) * np.sqrt(252)
        feat["vol_regime_ratio"] = (feat["realized_vol_21d"] / feat["hist_vol_30"].replace(0, np.nan)).replace([np.inf,-np.inf], np.nan).fillna(1.0)
        feat["vol_regime_stressed"] = (feat["vol_regime_ratio"] > 1.5).astype(float)
        tr = pd.concat([(high-low).abs(), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        feat["atr_pct"] = (atr / close).replace([np.inf,-np.inf], np.nan).fillna(0.02)
        feat["vol_ratio_20"] = (volume / volume.rolling(20).mean().replace(0, np.nan)).replace([np.inf,-np.inf], np.nan).fillna(1.0)
        feat["rsi_14"] = rsi(close, 14)
        mid = close.rolling(20).mean()
        sd = close.rolling(20).std()
        upper = mid + 2*sd
        lower = mid - 2*sd
        feat["bb_position"] = ((close - lower) / (upper - lower).replace(0, np.nan)).clip(0,1).fillna(0.5)
        feat["alpha_signal"] = (
            0.45*np.tanh(feat["momentum_20d"]*8)
            + 0.25*np.tanh(feat["close_vs_sma_50"]*5)
            + 0.20*np.tanh((feat["rsi_14"]-50)/20)
            + 0.10*np.tanh((feat["vol_ratio_20"]-1.0))
        )
        feat["event_alpha_signal"] = 0.0
        feat["official_event_signal"] = 0.0
        feat["filing_event_signal"] = 0.0
        feat["earnings_tone_signal"] = 0.0
        feat["earnings_tone_velocity"] = 0.0
        feat["sentiment_velocity"] = 0.0
        feat["weighted_sentiment_zscore"] = 0.0
        feat["earnings_propagation_signal"] = 0.0
        feat["filing_event_hit"] = 0.0
        feat["new_risk_factors"] = 0.0
        feat["news_volume_spike"] = 0.0
        feat["filing_change_score"] = 0.0
        feat["peer_earnings_negative_ratio_7d"] = 0.0
        feat["close_reversal_signal"] = -np.tanh((feat["rsi_14"]-50)/25)
        feat["close_reversal_strength"] = feat["close_reversal_signal"].abs()
        feat["event_move_strength"] = feat["return_5d"].abs()

        feat = feat.replace([np.inf,-np.inf], np.nan).dropna(subset=["close"])
        feat = feat.iloc[220:].copy()
        if feat.empty:
            return sym, 0, "empty_after_warmup"

        OUT.mkdir(parents=True, exist_ok=True)
        feat.to_parquet(OUT / f"{sym}.parquet")
        return sym, len(feat), "ok"
    except Exception as e:
        return sym, 0, type(e).__name__ + ": " + str(e)[:120]

def main():
    files = sorted(SRC.glob("*.parquet"))
    print("source", SRC, "files", len(files))
    print("out", OUT, "workers", WORKERS)
    OUT.mkdir(parents=True, exist_ok=True)

    total = 0
    ok = 0
    errors = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(build_one, str(f)) for f in files]
        for i, fut in enumerate(as_completed(futs), start=1):
            sym, rows, status = fut.result()
            if status == "ok":
                ok += 1
                total += rows
            else:
                errors[status] = errors.get(status, 0) + 1
            if i % 100 == 0 or i == len(futs):
                print(f"progress {i}/{len(futs)} ok={ok} rows={total} errors={sum(errors.values())}", flush=True)
    print("DONE", "ok", ok, "rows", total, "errors", errors)

if __name__ == "__main__":
    main()
