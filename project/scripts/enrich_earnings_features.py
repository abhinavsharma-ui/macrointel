"""
Earnings Feature Enrichment
============================
Computes rolling earnings quality features from data/earnings_lookup.csv
and price data from the 26yr parquets. No API calls needed.

Output: data/earnings_features.csv
Columns added:
  beat_rate_8q          - fraction of last 8 quarters that beat (surprise > 0)
  surprise_momentum_3q  - avg surprise_pct over last 3 quarters
  surprise_momentum_8q  - avg surprise_pct over last 8 quarters
  hist_move_avg_4q      - avg abs price % move on earnings day (last 4 quarters)
  beat_streak           - consecutive quarters beating (0 if last was a miss)
  consistent_beater     - 1 if beat_rate_8q >= 0.75 (reliable compounder)

Runtime: ~2-3 minutes (reads parquets for price move calculation)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

EARNINGS_CSV  = Path(os.getenv("EARNINGS_LOOKUP",  "data/earnings_lookup.csv"))
FEATURES_CSV  = Path(os.getenv("EARNINGS_FEATURES","data/earnings_features.csv"))
ROOT          = Path(os.getenv("PEAD_ROOT",         "data/features_26yr_liquid"))


def log(*x):
    print("ENRICH", *x, flush=True)


# ── Load earnings lookup ───────────────────────────────────────────────────────
log("Loading earnings_lookup.csv...")
df = pd.read_csv(EARNINGS_CSV, parse_dates=["earnings_date"])
df["earnings_date"] = pd.to_datetime(df["earnings_date"]).dt.normalize()
df["symbol"] = df["symbol"].str.upper()
df = df.sort_values(["symbol", "earnings_date"]).reset_index(drop=True)

# Ensure surprise_pct is computed (fallback if NaN)
mask = df["surprise_pct"].isna() & df["eps_estimate"].notna() & df["reported_eps"].notna()
df.loc[mask, "surprise_pct"] = (
    (df.loc[mask, "reported_eps"] - df.loc[mask, "eps_estimate"])
    / df.loc[mask, "eps_estimate"].abs() * 100
).replace([np.inf, -np.inf], np.nan)

log(f"  {len(df)} events | {df.symbol.nunique()} symbols | "
    f"surprise non-null: {df.surprise_pct.notna().sum()}")


# ── Rolling earnings quality features (computed per-symbol) ───────────────────
def compute_rolling(grp: pd.DataFrame) -> pd.DataFrame:
    grp = grp.sort_values("earnings_date").copy()
    sp  = grp["surprise_pct"].values

    beat_rate_8q         = []
    surprise_momentum_3q = []
    surprise_momentum_8q = []
    beat_streak_vals     = []

    for i in range(len(sp)):
        past8 = sp[max(0, i-8):i]
        past3 = sp[max(0, i-3):i]

        # Beat rate: fraction of past quarters where surprise > 0
        if len(past8) >= 2:
            br = float(np.nansum(past8 > 0) / np.sum(~np.isnan(past8)))
        else:
            br = np.nan
        beat_rate_8q.append(br)

        # Surprise momentum: avg surprise of recent quarters
        surprise_momentum_3q.append(float(np.nanmean(past3)) if len(past3) >= 1 else np.nan)
        surprise_momentum_8q.append(float(np.nanmean(past8)) if len(past8) >= 2 else np.nan)

        # Beat streak: consecutive beats ending at most recent past quarter
        streak = 0
        for v in reversed(past8):
            if np.isnan(v):
                break
            if v > 0:
                streak += 1
            else:
                break
        beat_streak_vals.append(streak)

    grp["beat_rate_8q"]          = beat_rate_8q
    grp["surprise_momentum_3q"]  = surprise_momentum_3q
    grp["surprise_momentum_8q"]  = surprise_momentum_8q
    grp["beat_streak"]           = beat_streak_vals
    grp["consistent_beater"]     = (grp["beat_rate_8q"] >= 0.75).astype(float)
    return grp


log("Computing rolling features per symbol...")
df = df.groupby("symbol", group_keys=False).apply(compute_rolling)
log(f"  beat_rate_8q non-null: {df.beat_rate_8q.notna().sum()}")
log(f"  consistent_beaters: {(df.consistent_beater==1).sum()} events")


# ── Historical price move on earnings day (from parquets) ─────────────────────
log("Computing historical price moves from parquets...")

def norm_sym(p: Path) -> str:
    return p.stem.replace("_US", "").replace(".US", "").upper()


price_moves = {}  # {(symbol, earnings_date): abs_pct_move}

files = sorted(ROOT.glob("*.parquet"))
for i, p in enumerate(files, 1):
    sym = norm_sym(p)
    events = df[df.symbol == sym]
    if events.empty:
        continue
    try:
        px = pd.read_parquet(p, columns=["close"] + (["date"] if "date" in
             pd.read_parquet(p, columns=[]).columns else []))
        # Reconstruct date
        if "date" in px.columns:
            px["date"] = pd.to_datetime(px["date"]).dt.normalize()
        else:
            px["date"] = pd.to_datetime(px.index, errors="coerce").normalize()
        px = px.dropna(subset=["date", "close"]).sort_values("date")
        px["pct_chg"] = px["close"].pct_change().abs() * 100
        px_map = dict(zip(px["date"], px["pct_chg"]))
        for _, row in events.iterrows():
            move = px_map.get(row["earnings_date"], np.nan)
            price_moves[(sym, row["earnings_date"])] = move
    except Exception:
        pass
    if i % 100 == 0:
        log(f"  {i}/{len(files)} parquets processed")

df["hist_move_day"] = df.apply(
    lambda r: price_moves.get((r["symbol"], r["earnings_date"]), np.nan), axis=1
)

# Rolling avg abs move over last 4 quarters
def add_hist_move(grp):
    grp = grp.sort_values("earnings_date").copy()
    moves = grp["hist_move_day"].values
    avg4 = []
    for i in range(len(moves)):
        past = moves[max(0, i-4):i]
        avg4.append(float(np.nanmean(past)) if np.sum(~np.isnan(past)) >= 1 else np.nan)
    grp["hist_move_avg_4q"] = avg4
    return grp

df = df.groupby("symbol", group_keys=False).apply(add_hist_move)

# ── Save ───────────────────────────────────────────────────────────────────────
FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(FEATURES_CSV, index=False)

log(f"\nDone. Saved to {FEATURES_CSV}")
log(f"  Rows: {len(df)} | Symbols: {df.symbol.nunique()}")
log(f"  beat_rate_8q:         {df.beat_rate_8q.notna().sum()} non-null, mean={df.beat_rate_8q.mean():.2f}")
log(f"  surprise_momentum_3q: {df.surprise_momentum_3q.notna().sum()} non-null, mean={df.surprise_momentum_3q.mean():.2f}")
log(f"  hist_move_avg_4q:     {df.hist_move_avg_4q.notna().sum()} non-null, mean={df.hist_move_avg_4q.mean():.2f}%")

# Sample output for AAPL
sample = df[df.symbol=="AAPL"][
    ["earnings_date","surprise_pct","beat_rate_8q",
     "surprise_momentum_3q","beat_streak","hist_move_avg_4q"]
].tail(6)
log(f"\nAAPL sample:\n{sample.to_string()}")
