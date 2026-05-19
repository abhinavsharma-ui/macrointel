#!/usr/bin/env python3
"""
Train the production fixed_return_h10 model on all available data through today.

Mirrors fixed_return_h10_walkforward_sim.py exactly:
  - same data loader (data/features_26yr_liquid/*.parquet)
  - same runtime universe (data/runtime_state.json)
  - same filters (MIN_ADV, MIN_PRICE)
  - same label: (model_ret > COST).astype(int), where
        entry_open         = open.shift(-1)
        default_exit_close = close.shift(-(HOLD+1))
        model_ret          = (default_exit_close / entry_open - 1) * 100
  - same feature_cols() banned set + preference order
  - same model: HistGradientBoostingClassifier(
        max_iter=180, learning_rate=0.045, max_leaf_nodes=31,
        l2_regularization=0.1, random_state=42)
  - same MAX_TRAIN cap with random_state=42

Output:
  models/checkpoints/fixed_return_h10_model.joblib
  models/fixed_return_h10_model.joblib  (mirrored if it already exists)

The output preserves the format of any existing artifact at the target path
(bare estimator OR {"model": ..., "features": [...], ...} dict), so the live
loader keeps working without changes.

Run from the project root:
    python scripts/train_fixed_return_h10.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT)

# ── env knobs (identical defaults to the walkforward sim) ──────────────────
ROOT          = Path(os.getenv("SIM_ROOT", "data/features_26yr_liquid"))
RUNTIME_STATE = Path(os.getenv("SIM_RUNTIME_STATE", "data/runtime_state.json"))
HOLD          = int(os.getenv("SIM_HOLD_DAYS", "10"))
COST          = float(os.getenv("SIM_TOTAL_COST_PCT", "0.45"))
MIN_ADV       = float(os.getenv("SIM_MIN_ADV20_DOLLAR_VOL", "5000000"))
MIN_PRICE     = float(os.getenv("SIM_MIN_ENTRY_PRICE", "5"))
MAX_ABS_RET   = float(os.getenv("SIM_MAX_ABS_HONEST_RET_PCT", "100"))
MAX_TRAIN     = int(os.getenv("SIM_MAX_TRAIN_ROWS", "350000"))
USE_RUNTIME_UNIVERSE = os.getenv("SIM_USE_RUNTIME_UNIVERSE", "1").lower() in {"1", "true", "yes", "on"}

MODEL_PATH  = Path(os.getenv("FIXED_MODEL_PATH", "models/checkpoints/fixed_return_h10_model.joblib"))
MIRROR_PATH = Path("models/fixed_return_h10_model.joblib")


def log(*a):
    print("TRAIN", *a, flush=True)


# ── helpers copied verbatim from fixed_return_h10_walkforward_sim.py ───────
def norm_sym(p: Path) -> str:
    return p.stem.replace("_US", "").replace(".US", "").upper()


def read_price_file(p: Path):
    df = pd.read_parquet(p)
    if df.empty or not {"open", "close", "volume"}.issubset(df.columns):
        return None
    idx = pd.to_datetime(df.index, errors="coerce")
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df.index = pd.RangeIndex(len(df))
    if "date" in df.columns:
        dc = pd.to_datetime(df["date"], errors="coerce")
        df["date"] = dc if dc.notna().sum() >= idx.notna().sum() else idx
    else:
        df["date"] = idx
    df = df.dropna(subset=["date", "open", "close"]).sort_values("date").reset_index(drop=True)
    df["symbol"] = norm_sym(p)
    if "adv20_dollar_vol" not in df.columns:
        df["adv20_dollar_vol"] = (df["close"] * df["volume"]).rolling(20, min_periods=5).mean()
    return df


def runtime_symbols():
    """Mirror of sim's runtime_symbols() — base universe only (no EXPAND)."""
    if not (USE_RUNTIME_UNIVERSE and RUNTIME_STATE.exists()):
        return None
    try:
        j = json.load(open(RUNTIME_STATE))
        base = {str(k).upper() for k in (j.get("signal_store") or {}).keys()}
        return base or None
    except Exception:
        return None


ALLOWED = runtime_symbols()


def transform_one(x: pd.DataFrame):
    """Strip-down of sim's load_one — keeps only what's needed for training:
    apply MIN_ADV/MIN_PRICE filters, compute model_ret, drop bad rows."""
    x = x[x["adv20_dollar_vol"].fillna(0) >= MIN_ADV].copy()
    x = x[x["close"].fillna(0) >= MIN_PRICE].copy()
    if x.empty:
        return None
    x["entry_open"]         = x["open"].shift(-1)
    x["default_exit_close"] = x["close"].shift(-(HOLD + 1))
    x["model_ret"]          = ((x["default_exit_close"] / x["entry_open"]) - 1) * 100
    x = x.replace([np.inf, -np.inf], np.nan).dropna(subset=["model_ret"])
    x = x[x["model_ret"].abs() <= MAX_ABS_RET]
    return x


def feature_cols(df: pd.DataFrame):
    """Identical to fixed_return_h10_walkforward_sim.feature_cols."""
    banned = {
        "date", "symbol", "year", "open", "high", "low", "close", "volume", "adj_close",
        "entry_open", "default_exit_close", "effective_exit_close", "exit_close",
        "model_ret", "honest_ret", "adv20_dollar_vol", "pt_hit", "pt_day",
        "exit_reason", "exit_offset", "spy_realized_vol_pct",
    }
    out = []
    for c in df.columns:
        cl = c.lower()
        if (c in banned
                or cl.startswith("gross_ret_") or cl.startswith("exit_timestamp")
                or cl.startswith("future") or cl.startswith("next")
                or "target" in cl or "label" in cl):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    pref = [
        "return_20d", "return_60d", "momentum_composite", "vol_regime_ratio", "atr_pct",
        "price_acceleration", "close_vs_sma_50", "close_vs_sma_200", "rsi_14",
    ]
    return [c for c in pref if c in out] + [c for c in out if c not in pref]


# ── load ────────────────────────────────────────────────────────────────────
files = sorted(ROOT.glob("*.parquet"))
log("files", len(files), "universe", "all" if ALLOWED is None else len(ALLOWED))

rows = []
for i, p in enumerate(files, 1):
    sym = norm_sym(p)
    if ALLOWED is not None and sym not in ALLOWED:
        continue
    raw = read_price_file(p)
    if raw is None:
        continue
    x = transform_one(raw)
    if x is None or len(x) == 0:
        continue
    rows.append(x)
    if i % 100 == 0:
        log("loaded", i, "files", "kept_rows", sum(len(r) for r in rows))

if not rows:
    log("no data after filters — abort")
    sys.exit(2)

df = pd.concat(rows, ignore_index=True)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
log("dataset rows", len(df), "symbols", df["symbol"].nunique(),
    "date range", df["date"].min().date(), "→", df["date"].max().date())

# Belt-and-suspenders embargo: don't train on rows whose default_exit_close
# would land in the future (the shift(-) above already NaN'd these and dropna
# removed them, but if data extends past today, force the cutoff explicitly).
today = pd.Timestamp.today().normalize()
cutoff = today - pd.tseries.offsets.BDay(HOLD + 1)
before = len(df)
df = df[df["date"] <= cutoff].reset_index(drop=True)
log("post-embargo rows", len(df), "dropped", before - len(df), "cutoff", cutoff.date())

# ── features + label ───────────────────────────────────────────────────────
fs = feature_cols(df)
if not fs:
    log("no features found in dataframe — abort")
    sys.exit(3)
log("features", len(fs), "first8", fs[:8])

if len(df) > MAX_TRAIN:
    df = df.sample(MAX_TRAIN, random_state=42).reset_index(drop=True)
    log("subsampled to", MAX_TRAIN)

X = df[fs].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
y = (df["model_ret"] > COST).astype(int)
log("class balance pos", int(y.sum()), "/", len(y), "=", round(float(y.mean()), 4))

# ── train ──────────────────────────────────────────────────────────────────
model = HistGradientBoostingClassifier(
    max_iter=180,
    learning_rate=0.045,
    max_leaf_nodes=31,
    l2_regularization=0.1,
    random_state=42,
)
model.fit(X, y)
log("trained")

# ── persist (preserve existing artifact format) ────────────────────────────
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

existing_format = "bare"
if MODEL_PATH.exists():
    try:
        prev = joblib.load(MODEL_PATH)
        if isinstance(prev, dict) and "model" in prev:
            existing_format = "dict"
    except Exception as e:
        log("warn: could not introspect existing artifact:", repr(e))

bundle = {
    "model": model,
    "features": fs,
    "trained_at": datetime.now(timezone.utc).isoformat(),
    "cost_pct": COST,
    "hold_days": HOLD,
    "min_adv": MIN_ADV,
    "min_price": MIN_PRICE,
    "n_rows": int(len(df)),
    "n_symbols": int(df["symbol"].nunique()),
    "hyperparams": dict(
        estimator="HistGradientBoostingClassifier",
        max_iter=180, learning_rate=0.045, max_leaf_nodes=31,
        l2_regularization=0.1, random_state=42,
    ),
}

if MODEL_PATH.exists():
    bk = MODEL_PATH.with_suffix(f".joblib.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    shutil.copy2(MODEL_PATH, bk)
    log("backup", str(bk))

payload = bundle if existing_format == "dict" else model
joblib.dump(payload, MODEL_PATH)
log("wrote", str(MODEL_PATH), "format", existing_format)

if MIRROR_PATH.exists() or MIRROR_PATH.parent.exists():
    MIRROR_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(MODEL_PATH, MIRROR_PATH)
    log("mirrored to", str(MIRROR_PATH))

# Sidecar metadata is always handy, regardless of bundle/bare format
meta_path = MODEL_PATH.with_suffix(".meta.json")
meta_path.write_text(json.dumps({k: v for k, v in bundle.items() if k != "model"},
                                indent=2, default=str))
log("meta", str(meta_path))
log("DONE")
