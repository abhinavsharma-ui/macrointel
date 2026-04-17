#!/usr/bin/env python3
"""
Relative Strength Residualization
==================================
Computes idiosyncratic (market-neutral) momentum per symbol:

    epsilon_i = R_i - (alpha_i + beta_i * R_market)

Where alpha and beta come from a rolling OLS regression (60-bar window) of
symbol return on the relevant market proxy return:
    * Crypto lane  -> BTC-USD return
    * US equities  -> SPY return
    * India lane   -> NSE_INDEX|Nifty 50 return

SOFT GATE: If fewer than 20 bars of history exist, fall back to raw symbol
return as the feature value, and set `residual_fallback=True` so the caller
can apply a 0.05 confidence penalty.

The module is stateless and pure: it takes aligned price/return series and
returns the residual-momentum series plus a boolean fallback flag. It does
not fetch any data on its own; caller supplies returns already aligned.

Standalone test:
    python -m project.core.residual_momentum --demo
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROLLING_WINDOW = int(os.getenv("RESIDUAL_MOMENTUM_WINDOW", "60"))
MIN_BARS_FOR_REGRESSION = int(os.getenv("RESIDUAL_MOMENTUM_MIN_BARS", "20"))


MARKET_PROXY_BY_LANE = {
    "crypto": "BTC-USD",
    "us":     "SPY",
    "normal": "SPY",
    "india":  "NSE_INDEX|Nifty 50",
    "day":    "NSE_INDEX|Nifty 50",
}


def proxy_for_lane(lane: str) -> str:
    return MARKET_PROXY_BY_LANE.get((lane or "").lower(), "SPY")


@dataclass
class ResidualResult:
    residual: float
    used_fallback: bool
    alpha: float
    beta: float
    r_squared: float
    n_bars_used: int


def rolling_residual(
    symbol_returns: pd.Series,
    market_returns: pd.Series,
    *,
    window: int = ROLLING_WINDOW,
    min_bars: int = MIN_BARS_FOR_REGRESSION,
) -> Tuple[pd.Series, pd.Series]:
    """
    Returns (residual_series, fallback_flags_series) aligned to symbol_returns.

    residual_series: R_i - (alpha + beta * R_market) computed on a rolling window;
    if the lookback is shorter than `min_bars`, we return the raw R_i and set
    the fallback flag True for that row.
    """
    df = pd.concat(
        [symbol_returns.rename("y"), market_returns.rename("x")],
        axis=1,
    ).dropna()
    n = len(df)
    residuals = np.full(n, np.nan)
    fallback = np.zeros(n, dtype=bool)

    if n == 0:
        return (
            pd.Series(residuals, index=symbol_returns.index, name="residual_momentum"),
            pd.Series(fallback, index=symbol_returns.index, name="residual_fallback"),
        )

    y = df["y"].to_numpy(dtype=np.float64)
    x = df["x"].to_numpy(dtype=np.float64)

    for i in range(n):
        start = max(0, i - window + 1)
        lookback_len = i - start + 1
        if lookback_len < min_bars:
            residuals[i] = y[i]
            fallback[i] = True
            continue
        xs = x[start : i + 1]
        ys = y[start : i + 1]
        x_mean = xs.mean()
        y_mean = ys.mean()
        var_x = ((xs - x_mean) ** 2).sum()
        if var_x <= 1e-12:
            residuals[i] = y[i]
            fallback[i] = True
            continue
        cov_xy = ((xs - x_mean) * (ys - y_mean)).sum()
        beta = cov_xy / var_x
        alpha = y_mean - beta * x_mean
        residuals[i] = y[i] - (alpha + beta * x[i])
        fallback[i] = False

    out_r = pd.Series(residuals, index=df.index, name="residual_momentum")
    out_f = pd.Series(fallback, index=df.index, name="residual_fallback")
    # Re-align to the original symbol_returns index (NaN where one of the two
    # series was missing)
    out_r = out_r.reindex(symbol_returns.index)
    out_f = out_f.reindex(symbol_returns.index).fillna(True).astype(bool)
    return out_r, out_f


def latest_residual(
    symbol_returns: pd.Series,
    market_returns: pd.Series,
    *,
    window: int = ROLLING_WINDOW,
    min_bars: int = MIN_BARS_FOR_REGRESSION,
) -> ResidualResult:
    """
    Online-style: compute only the latest residual with alpha/beta/r_squared
    diagnostics. Returns fallback if window is too short or x-variance is degenerate.
    """
    df = pd.concat(
        [symbol_returns.rename("y"), market_returns.rename("x")],
        axis=1,
    ).dropna()
    if df.empty:
        return ResidualResult(residual=0.0, used_fallback=True, alpha=0.0, beta=0.0, r_squared=0.0, n_bars_used=0)

    df = df.tail(window)
    n = len(df)
    y = df["y"].to_numpy(dtype=np.float64)
    x = df["x"].to_numpy(dtype=np.float64)

    if n < min_bars:
        return ResidualResult(residual=float(y[-1]), used_fallback=True, alpha=0.0, beta=0.0, r_squared=0.0, n_bars_used=n)

    x_mean = x.mean()
    y_mean = y.mean()
    var_x = ((x - x_mean) ** 2).sum()
    if var_x <= 1e-12:
        return ResidualResult(residual=float(y[-1]), used_fallback=True, alpha=0.0, beta=0.0, r_squared=0.0, n_bars_used=n)

    cov_xy = ((x - x_mean) * (y - y_mean)).sum()
    beta = float(cov_xy / var_x)
    alpha = float(y_mean - beta * x_mean)
    residual = float(y[-1] - (alpha + beta * x[-1]))

    y_pred = alpha + beta * x
    ss_res = float(((y - y_pred) ** 2).sum())
    ss_tot = float(((y - y_mean) ** 2).sum())
    r_sq = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    return ResidualResult(
        residual=residual,
        used_fallback=False,
        alpha=alpha,
        beta=beta,
        r_squared=max(0.0, min(1.0, r_sq)),
        n_bars_used=n,
    )


# -----------------------------------------------------------------------------
# Demo
# -----------------------------------------------------------------------------


def _demo() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rng = np.random.default_rng(42)

    # 120 bars. Market walks, symbol is 1.2*market + idio.
    market = pd.Series(rng.normal(0.0005, 0.012, size=120), name="spy_ret")
    true_beta = 1.2
    idio = pd.Series(rng.normal(0.0002, 0.006, size=120), name="idio")
    symbol = true_beta * market + idio
    symbol.name = "symbol_ret"

    ser, fb = rolling_residual(symbol, market, window=60, min_bars=20)
    print("rolling_residual: first 25 fallback rows:", fb.iloc[:25].sum(), "of 25")
    print("post-warmup residual mean (should be ~0):", ser.iloc[30:].mean())
    print("post-warmup residual std (should be ~idio std 0.006):", ser.iloc[30:].std())

    res = latest_residual(symbol, market)
    print(f"latest_residual: residual={res.residual:.5f} beta={res.beta:.3f} r2={res.r_squared:.3f} fallback={res.used_fallback} n={res.n_bars_used}")

    short_sym = symbol.iloc[:10]
    short_mkt = market.iloc[:10]
    short = latest_residual(short_sym, short_mkt)
    print(f"short-history fallback: residual={short.residual:.5f} fallback={short.used_fallback}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()
    _demo()
