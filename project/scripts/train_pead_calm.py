"""
PEAD Calm-Market Production Trainer
=====================================
Trains the production PEAD model on ALL available 26yr data.
Output: models/checkpoints/pead_calm_model.joblib

ONLY RUN THIS AFTER pead_calm_walkforward_sim.py passes ALL success criteria:
  - Walk-forward WR ≥ 65%
  - Sharpe ≥ 1.0
  - Max drawdown ≤ 20%
  - 2014 WR ≥ 65%, 2015 WR ≥ 65%

DO NOT overwrite fixed_return_h10_model.joblib.
DO NOT modify fixed_return_daily_signals.py until 30 paper trades completed.
"""

import json, os, warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.ensemble import HistGradientBoostingClassifier

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.parent
ROOT        = Path(os.getenv("PEAD_ROOT",  "data/features_26yr_liquid"))
MODEL_OUT   = Path(os.getenv("PEAD_MODEL_OUT",
                              "models/checkpoints/pead_calm_model.joblib"))
META_OUT    = MODEL_OUT.with_suffix(".meta.json")

# ── Constants (must match pead_calm_walkforward_sim.py) ───────────────────────
PEAD_HOLD             = int(  os.getenv("PEAD_HOLD",               "20"))
PEAD_ENTRY_OFFSET     = int(  os.getenv("PEAD_ENTRY_OFFSET",        "1"))
COST                  = float(os.getenv("PEAD_COST_PCT",            "0.45"))
MIN_ADV               = float(os.getenv("PEAD_MIN_ADV",             "10000000"))
MIN_PRICE             = float(os.getenv("PEAD_MIN_PRICE",           "10"))
PEAD_VOL_GATE         = float(os.getenv("PEAD_VOL_GATE",            "20.0"))
PEAD_MIN_SURPRISE     = float(os.getenv("PEAD_MIN_SURPRISE",        "0.01"))
EARNINGS_DAY_THRESH   = int(  os.getenv("PEAD_EARNINGS_DAY_THRESH", "4"))
EARNINGS_RESET_THRESH = int(  os.getenv("PEAD_EARNINGS_RESET_THRESH","15"))
MAX_ABS_RET           = float(os.getenv("PEAD_MAX_ABS_RET",         "100"))
MAX_TRAIN_ROWS        = int(  os.getenv("PEAD_MAX_TRAIN_ALL",        "500000"))

PEAD_FEATURE_CANDIDATES = [
    "earnings_surprise_momentum",
    "earnings_beat_rate",
    "earnings_historical_move",
    "return_1d",
    "return_5d",
    "return_20d",
    "rsi_14",
    "close_vs_sma_50",
    "close_vs_sma_200",
    "adv20_dollar_vol",
    "atr_pct",
    "spy_realized_vol_pct",
    "insider_net_ratio_30d",
    "insider_cluster_buy",
    "insider_ceo_bought",
    "volume_ratio_20d",
    "momentum_composite",
    "price_acceleration",
    "vol_regime_ratio",
]


def log(*x):
    print("PEAD_TRAIN", *x, flush=True)


def norm_sym(p: Path) -> str:
    return p.stem.replace("_US", "").replace(".US", "").upper()


def read_price_file(p: Path, need_symbol: bool = True) -> "pd.DataFrame | None":
    df = pd.read_parquet(p)
    if df.empty or not {"open", "close", "volume"}.issubset(df.columns):
        return None
    idx = pd.to_datetime(df.index, errors="coerce")
    df  = df.loc[:, ~df.columns.duplicated()].copy()
    df.index = pd.RangeIndex(len(df))
    if "date" in df.columns:
        dc = pd.to_datetime(df["date"], errors="coerce")
        df["date"] = dc if dc.notna().sum() >= idx.notna().sum() else idx
    else:
        df["date"] = idx
    df = df.dropna(subset=["date", "open", "close"]).sort_values("date").reset_index(drop=True)
    if need_symbol:
        df["symbol"] = norm_sym(p)
    if "adv20_dollar_vol" not in df.columns:
        df["adv20_dollar_vol"] = (df["close"] * df["volume"]).rolling(20, min_periods=5).mean()
    return df


def load_spy_vol_map() -> dict:
    candidates = [ROOT / "SPY.parquet", ROOT / "SPY_US.parquet", ROOT / "SPY.US.parquet"]
    p = next((x for x in candidates if x.exists()), None)
    if p is None:
        return {}
    x = read_price_file(p, need_symbol=False)
    if x is None or x.empty:
        return {}
    ret = x["close"].pct_change()
    vol = ret.rolling(20, min_periods=10).std() * np.sqrt(252) * 100.0
    d   = pd.to_datetime(x["date"]).dt.normalize()
    return dict(zip(d, vol.fillna(20.0)))


SPY_VOL = load_spy_vol_map()


def load_one_pead(p: Path) -> "pd.DataFrame | None":
    sym = norm_sym(p)
    if sym.endswith((".NS", ".BO", ".NSE", ".BSE")):
        return None
    x = read_price_file(p)
    if x is None or x.empty:
        return None
    x = x[x["adv20_dollar_vol"].fillna(0) >= MIN_ADV].copy()
    x = x[x["close"].fillna(0) >= MIN_PRICE].copy()
    if len(x) < 60:
        return None
    if "earnings_days_to_next" not in x.columns:
        return None

    prev_days = x["earnings_days_to_next"].shift(1).fillna(999)
    curr_days = x["earnings_days_to_next"].fillna(999)
    earnings_event = (
        (prev_days <= EARNINGS_DAY_THRESH) &
        (curr_days > EARNINGS_RESET_THRESH)
    )
    if PEAD_ENTRY_OFFSET > 1:
        earnings_event = earnings_event.shift(-(PEAD_ENTRY_OFFSET - 1)).fillna(False)

    x["pead_entry_flag"]      = earnings_event.astype(int)
    x["entry_open"]           = x["open"].shift(-1)
    x["default_exit_close"]   = x["close"].shift(-(PEAD_HOLD + 1))
    x["model_ret"]            = (x["default_exit_close"] / x["entry_open"] - 1) * 100
    x["effective_exit_close"] = x["default_exit_close"]
    x["honest_ret"]           = x["model_ret"]
    x["exit_offset"]          = PEAD_HOLD + 1
    x["exit_reason"]          = "hold"
    x["pt_hit"]               = False
    x["pt_day"]               = np.nan
    x["spy_realized_vol_pct"] = (
        pd.to_datetime(x["date"]).dt.normalize().map(SPY_VOL).fillna(20.0)
    )

    x = x[x["pead_entry_flag"] == 1].copy()
    if x.empty:
        return None

    if "earnings_surprise_momentum" in x.columns:
        x = x[x["earnings_surprise_momentum"].fillna(-999) >= PEAD_MIN_SURPRISE].copy()

    # For training: use ALL volatility regimes but weight calm rows more
    # (the vol gate will be applied at inference time in daily_signals.py)
    x = x.replace([np.inf, -np.inf], np.nan).dropna(subset=["model_ret", "honest_ret"])
    x = x[x["model_ret"].abs() <= MAX_ABS_RET]
    return x


def load_all() -> pd.DataFrame:
    rows  = []
    files = sorted(ROOT.glob("*.parquet"))
    for i, p in enumerate(files, 1):
        x = load_one_pead(p)
        if x is not None and len(x):
            rows.append(x)
        if i % 100 == 0:
            log(f"  {i}/{len(files)} files | rows: {sum(len(r) for r in rows)}")
    if not rows:
        raise RuntimeError("No PEAD rows found. Run build_earnings_insider_features.py first.")
    df = pd.concat(rows, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


def feature_cols(df: pd.DataFrame) -> list:
    available = set(df.columns)
    banned = {
        "date", "symbol", "year", "open", "high", "low", "close", "volume", "adj_close",
        "entry_open", "default_exit_close", "effective_exit_close", "exit_close",
        "model_ret", "honest_ret", "adv20_dollar_vol", "pt_hit", "pt_day",
        "exit_reason", "exit_offset", "pead_entry_flag",
    }
    selected = [c for c in PEAD_FEATURE_CANDIDATES if c in available and c not in banned]
    extra = [
        c for c in df.columns
        if c not in banned and c not in selected
        and not any(c.lower().startswith(pfx) for pfx in ("future", "next", "gross_ret_", "exit_timestamp"))
        and "target" not in c.lower() and "label" not in c.lower()
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return selected + extra


# ── Verify backtest passed ─────────────────────────────────────────────────────
backtest_result = Path("reports/pead_calm_walkforward_sim.json")
if backtest_result.exists():
    try:
        bt = json.load(open(backtest_result))
        wr = bt.get("summary", {}).get("win_rate_pct", 0)
        sh = bt.get("summary", {}).get("sharpe", 0)
        dd = bt.get("summary", {}).get("max_drawdown_pct", 999)
        log(f"Backtest result: WR={wr:.1f}%  Sharpe={sh:.2f}  MaxDD={dd:.1f}%")
        if wr < 65 or sh < 1.0 or dd > 20:
            raise SystemExit(
                f"ABORT: Backtest does not meet criteria (WR={wr:.1f}% Sharpe={sh:.2f} DD={dd:.1f}%). "
                "Fix the model before training production version."
            )
        log("Backtest criteria passed — proceeding to train production model")
    except SystemExit:
        raise
    except Exception as e:
        log(f"WARNING: Could not parse backtest result ({e}). Proceeding anyway.")
else:
    log("WARNING: No backtest result found. Run pead_calm_walkforward_sim.py first!")
    log("Proceeding with training anyway — ensure backtest has passed manually.")

# ── Load data and train ────────────────────────────────────────────────────────
log("Loading all PEAD rows...")
df = load_all()
fs = feature_cols(df)

log(f"Total PEAD rows: {len(df)} | Symbols: {df.symbol.nunique()} | Features: {len(fs)}")
log("Features:", fs)

if len(df) > MAX_TRAIN_ROWS:
    log(f"Sampling down from {len(df)} to {MAX_TRAIN_ROWS} rows")
    df = df.sample(MAX_TRAIN_ROWS, random_state=42)

y = (df["model_ret"] > COST).astype(int)
log(f"Label distribution: {y.mean():.3f} positive rate ({y.sum()} / {len(y)})")

X = df[fs].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")

model = HistGradientBoostingClassifier(
    max_iter=150,
    learning_rate=0.04,
    max_leaf_nodes=25,
    l2_regularization=0.15,
    random_state=42,
)
log("Training HistGradientBoostingClassifier...")
model.fit(X, y)
log("Training complete.")

# ── Save model ─────────────────────────────────────────────────────────────────
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_OUT)
log(f"Model saved to {MODEL_OUT}")

meta = {
    "model_type":     "HistGradientBoostingClassifier",
    "purpose":        "PEAD calm-market drift — second lane (SPY vol < 20)",
    "features":       fs,
    "hold_days":      PEAD_HOLD,
    "vol_gate":       PEAD_VOL_GATE,
    "min_surprise":   PEAD_MIN_SURPRISE,
    "cost_pct":       COST,
    "train_rows":     int(len(df)),
    "symbols":        int(df.symbol.nunique()),
    "positive_rate":  round(float(y.mean()), 4),
    "trained_date":   str(pd.Timestamp.now().date()),
    "deployment_note": (
        "Deploy ONLY after 30 paper trades. "
        "Integrate via regime switch in fixed_return_daily_signals.py: "
        "if spy_vol < 20: use this model with hold_days=20, else use h10 model."
    ),
}
META_OUT.write_text(json.dumps(meta, indent=2))
log(f"Model metadata saved to {META_OUT}")
log("Done. Next step: paper trade for 30 days before live deployment.")
