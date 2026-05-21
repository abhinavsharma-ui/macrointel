"""
PEAD Calm-Market Walkforward Backtest
======================================
Post-Earnings Announcement Drift model — the SECOND LANE of the regime-complete system.

Strategy:
  - Universe: only rows where earnings just happened (2 days after) + meaningful beat + SPY vol < 20
  - Entry: next day open after signal (i.e., ~2 trading days post-earnings)
  - Hold: PEAD_HOLD (20) days
  - Label: (exit_close / entry_open - 1) * 100 > COST → 1 else 0

Walk-forward discipline mirrors fixed_return_h10_walkforward_sim.py:
  - FOLDS=4, train on past, test on next fold only
  - train_end < test_start by HOLD+1 days (no lookahead)
  - MAX_TRAIN rows capped per fold

Outputs:
  reports/pead_calm_walkforward_sim.json
  reports/pead_calm_walkforward_trades.csv

DO NOT modify the h10 model or its training script.
DO NOT run this with train_fixed_return_h10.py or overwrite the production model.
"""

import json, os, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.ensemble import HistGradientBoostingClassifier

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT   = Path(os.getenv("PEAD_ROOT",   "data/features_26yr_liquid"))
OUT    = Path(os.getenv("PEAD_OUT",    "reports/pead_calm_walkforward_sim.json"))
TRADES = Path(os.getenv("PEAD_TRADES", "reports/pead_calm_walkforward_trades.csv"))

# ── PEAD-specific constants ────────────────────────────────────────────────────
PEAD_HOLD        = int(  os.getenv("PEAD_HOLD",        "20"))   # 20-day drift window
PEAD_ENTRY_OFFSET = int( os.getenv("PEAD_ENTRY_OFFSET", "1"))   # signal on day E+1; entry = E+2 open
COST             = float(os.getenv("PEAD_COST_PCT",     "0.45")) # round-trip cost bps
MIN_ADV          = float(os.getenv("PEAD_MIN_ADV",      "10000000"))  # $10M ADV
MIN_PRICE        = float(os.getenv("PEAD_MIN_PRICE",    "10"))
PEAD_VOL_GATE    = float(os.getenv("PEAD_VOL_GATE",     "20.0"))  # only trade when SPY vol < 20
PEAD_MIN_SURPRISE= float(os.getenv("PEAD_MIN_SURPRISE", "0.01"))  # earnings_surprise_momentum threshold
EARNINGS_DAY_THRESH   = int(os.getenv("PEAD_EARNINGS_DAY_THRESH",   "4"))   # earnings_days_to_next ≤ this = earnings week
EARNINGS_RESET_THRESH = int(os.getenv("PEAD_EARNINGS_RESET_THRESH", "15"))  # earnings_days_to_next > this = post-earnings reset confirmed
MAX_ABS_RET      = float(os.getenv("PEAD_MAX_ABS_RET",  "100"))
TH               = float(os.getenv("PEAD_THRESHOLD",    "0.55"))
TOP_N            = int(  os.getenv("PEAD_TOP_N",        "5"))
POS              = float(os.getenv("PEAD_POSITION_PCT", "0.002"))
MAX_OPEN         = int(  os.getenv("PEAD_MAX_OPEN",     "30"))
MAX_TRAIN        = int(  os.getenv("PEAD_MAX_TRAIN",    "200000"))
FOLDS            = int(  os.getenv("PEAD_FOLDS",        "4"))
INIT             = float(os.getenv("PEAD_INITIAL_CAP",  "100000"))

# ── PEAD feature list (explicit — not the banned-list approach) ────────────────
# spy_realized_vol_pct IS allowed here as a training feature (unlike h10)
PEAD_FEATURE_CANDIDATES = [
    "earnings_surprise_momentum",   # trend of earnings beats
    "earnings_beat_rate",           # historical reliability
    "earnings_historical_move",     # size of typical moves
    "return_1d",                    # day+0 reaction (strong predictor of drift direction)
    "return_5d",                    # 1-week drift already in motion
    "return_20d",                   # medium-term context
    "rsi_14",
    "close_vs_sma_50",
    "close_vs_sma_200",
    "adv20_dollar_vol",
    "atr_pct",
    "spy_realized_vol_pct",         # ALLOWED: used as both feature and regime gate
    "insider_net_ratio_30d",
    "insider_cluster_buy",
    "insider_ceo_bought",
    # additional context features likely in parquets
    "volume_ratio_20d",
    "momentum_composite",
    "price_acceleration",
    "vol_regime_ratio",
]


def log(*x):
    print("PEAD_SIM", *x, flush=True)


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


# ── SPY realized vol map ───────────────────────────────────────────────────────
def load_spy_vol_map() -> dict:
    candidates = [ROOT / "SPY.parquet", ROOT / "SPY_US.parquet", ROOT / "SPY.US.parquet"]
    p = next((x for x in candidates if x.exists()), None)
    if p is None:
        log("WARNING: SPY parquet not found — vol gate will use 20.0 default")
        return {}
    x = read_price_file(p, need_symbol=False)
    if x is None or x.empty:
        return {}
    ret = x["close"].pct_change()
    vol = ret.rolling(20, min_periods=10).std() * np.sqrt(252) * 100.0
    d   = pd.to_datetime(x["date"]).dt.normalize()
    return dict(zip(d, vol.fillna(20.0)))


SPY_VOL = load_spy_vol_map()
log("spy_vol_map loaded, dates:", len(SPY_VOL))


# ── Per-symbol loader with PEAD event detection ────────────────────────────────
def load_one_pead(p: Path) -> "pd.DataFrame | None":
    sym = norm_sym(p)
    if sym.endswith((".NS", ".BO", ".NSE", ".BSE")):
        return None

    x = read_price_file(p)
    if x is None or x.empty:
        return None

    # Basic liquidity / price filter
    x = x[x["adv20_dollar_vol"].fillna(0) >= MIN_ADV].copy()
    x = x[x["close"].fillna(0) >= MIN_PRICE].copy()
    if len(x) < 60:
        return None

    # ── Detect post-earnings rows ────────────────────────────────────────────
    # earnings_days_to_next counts DOWN to the next earnings (0 on earnings day).
    # After earnings, the counter RESETS to ~60-90 (next quarter).
    # Pattern: prev_row ≤ EARNINGS_DAY_THRESH AND current_row > EARNINGS_RESET_THRESH
    #          → current row = first trading day AFTER earnings (E+1).
    # We signal on E+1 and enter on E+2 open (via entry_open = open.shift(-1)).
    if "earnings_days_to_next" not in x.columns:
        return None

    prev_days = x["earnings_days_to_next"].shift(1).fillna(999)
    curr_days = x["earnings_days_to_next"].fillna(999)

    earnings_event = (
        (prev_days <= EARNINGS_DAY_THRESH) &
        (curr_days > EARNINGS_RESET_THRESH)
    )

    # If PEAD_ENTRY_OFFSET > 1, shift the flag forward by (PEAD_ENTRY_OFFSET - 1) rows
    # Default PEAD_ENTRY_OFFSET=1 → signal on E+1 (earnings_event row), enter E+2 open
    if PEAD_ENTRY_OFFSET > 1:
        earnings_event = earnings_event.shift(-(PEAD_ENTRY_OFFSET - 1)).fillna(False)

    x["pead_entry_flag"] = earnings_event.astype(int)

    # ── Label construction ─────────────────────────────────────────────────
    # Entry: next day open (E+2 open for default offset=1)
    # Exit:  PEAD_HOLD days later close (no intraperiod TP for simplicity)
    x["entry_open"]          = x["open"].shift(-1)
    x["default_exit_close"]  = x["close"].shift(-(PEAD_HOLD + 1))
    x["model_ret"]           = (x["default_exit_close"] / x["entry_open"] - 1) * 100
    x["honest_ret"]          = x["model_ret"]         # no TP/SL in PEAD base model
    x["exit_offset"]         = PEAD_HOLD + 1
    x["exit_reason"]         = "hold"
    x["pt_hit"]              = False
    x["pt_day"]              = np.nan
    x["effective_exit_close"] = x["default_exit_close"]

    # SPY vol mapped onto each row (used as feature + gate)
    x["spy_realized_vol_pct"] = (
        pd.to_datetime(x["date"]).dt.normalize().map(SPY_VOL).fillna(20.0)
    )

    # ── Apply PEAD filters ──────────────────────────────────────────────────
    # 1. Only post-earnings rows
    x = x[x["pead_entry_flag"] == 1].copy()
    if x.empty:
        return None

    # 2. Only meaningful earnings beats (earnings_surprise_momentum > threshold)
    #    If the column doesn't exist, skip this filter (will rely on model)
    if "earnings_surprise_momentum" in x.columns:
        x = x[x["earnings_surprise_momentum"].fillna(-999) >= PEAD_MIN_SURPRISE].copy()

    # 3. Only calm regime rows (SPY realized vol < PEAD_VOL_GATE)
    #    This keeps the training distribution aligned with deployment conditions
    x = x[x["spy_realized_vol_pct"].fillna(99) < PEAD_VOL_GATE].copy()

    if x.empty:
        return None

    # Drop rows with missing labels or extreme returns
    x = x.replace([np.inf, -np.inf], np.nan).dropna(subset=["model_ret", "honest_ret"])
    x = x[x["model_ret"].abs() <= MAX_ABS_RET]
    x = x[x["honest_ret"].abs() <= MAX_ABS_RET]

    return x


def load_all_pead() -> pd.DataFrame:
    rows  = []
    files = sorted(ROOT.glob("*.parquet"))
    for i, p in enumerate(files, 1):
        x = load_one_pead(p)
        if x is not None and len(x):
            rows.append(x)
        if i % 100 == 0:
            log("loaded", i, "files | pead_rows so far:", sum(len(r) for r in rows))
    if not rows:
        raise RuntimeError(
            "No PEAD rows found! Check that earnings_days_to_next is populated "
            "in the parquets (run build_earnings_insider_features.py on the VM first)."
        )
    df = pd.concat(rows, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


# ── Feature column selection ───────────────────────────────────────────────────
def feature_cols_pead(df: pd.DataFrame) -> list:
    """Return the PEAD feature set — only columns that actually exist in df."""
    available = set(df.columns)
    banned = {
        "date", "symbol", "year", "open", "high", "low", "close", "volume", "adj_close",
        "entry_open", "default_exit_close", "effective_exit_close", "exit_close",
        "model_ret", "honest_ret", "adv20_dollar_vol", "pt_hit", "pt_day",
        "exit_reason", "exit_offset", "pead_entry_flag",
    }
    # Start with explicit PEAD candidates (preserves priority order)
    selected = [c for c in PEAD_FEATURE_CANDIDATES if c in available and c not in banned]
    # Add any remaining numeric columns not banned or already selected
    extra = [
        c for c in df.columns
        if c not in banned and c not in selected
        and not any(c.lower().startswith(pfx) for pfx in ("future", "next", "gross_ret_", "exit_timestamp"))
        and "target" not in c.lower() and "label" not in c.lower()
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return selected + extra


# ── Simulation engine (same logic as h10 but with SPYVOL_MAX gate) ─────────────
def simulate(pred: pd.DataFrame, calendar_dates: pd.Index) -> "tuple[dict, pd.DataFrame]":
    pred = pred[pred["prob"] >= TH].copy()
    if pred.empty:
        return {}, pd.DataFrame()

    pred["row_id"] = pred["symbol"].astype(str) + "|" + pred["date"].astype(str)
    by_date = {d: g.sort_values("prob", ascending=False) for d, g in pred.groupby("date")}

    eq = INIT; peak = INIT; dd = 0.0; active = []; trades = []; skipped = 0

    for i, d in enumerate(calendar_dates):
        keep = []; active_syms = set()
        for a in active:
            if a["exit_i"] <= i:
                eq += a["pnl"]
                trades.append({**a, "exit_date": str(d), "status": "closed"})
            else:
                keep.append(a); active_syms.add(a["symbol"])
        active = keep
        peak = max(peak, eq)
        dd   = max(dd, (peak - eq) / peak * 100 if peak else 0)

        day = by_date.get(d)
        if day is None:
            continue

        new = 0
        for _, r in day.head(TOP_N * 3).iterrows():
            if new >= TOP_N:
                break
            sym = str(r.symbol)
            if sym in active_syms or len(active) >= MAX_OPEN:
                continue
            # PEAD calm-regime gate: skip if SPY vol ≥ PEAD_VOL_GATE
            spy_vol = float(r.get("spy_realized_vol_pct", 99.0))
            if spy_vol >= PEAD_VOL_GATE:
                skipped += 1
                continue

            net       = float(r.honest_ret) - COST
            notional  = eq * POS
            pnl       = notional * net / 100
            off       = int(r.get("exit_offset", PEAD_HOLD + 1) or PEAD_HOLD + 1)

            active.append({
                "row_id":      r.row_id,
                "symbol":      sym,
                "signal_date": str(d),
                "prob":        float(r.prob),
                "model_ret":   float(r.model_ret),
                "honest_ret":  float(r.honest_ret),
                "net_ret":     net,
                "notional":    notional,
                "pnl":         pnl,
                "entry_price": float(r.entry_open),
                "exit_price":  float(r.effective_exit_close),
                "exit_i":      i + off,
                "exit_offset": off,
                "exit_reason": str(r.exit_reason),
                "pt_hit":      bool(r.pt_hit),
                "pt_day":      None if pd.isna(r.pt_day) else int(r.pt_day),
                "spy_realized_vol_pct": spy_vol,
            })
            active_syms.add(sym); new += 1

    final_date = str(calendar_dates[-1]) if len(calendar_dates) else ""
    for a in active:
        eq += a["pnl"]
        trades.append({**a, "exit_date": final_date, "status": "closed_final"})

    t   = pd.DataFrame(trades)
    arr = t["net_ret"].to_numpy() if len(t) else np.array([])

    return {
        "trades":                int(len(pred)),
        "executed_trades":       int(len(t)),
        "skipped_vol_gate":      int(skipped),
        "final_equity":          round(float(eq), 2),
        "return_pct":            round((eq / INIT - 1) * 100, 4),
        "max_drawdown_pct":      round(float(dd), 4),
        "avg_trade_net_return":  round(float(arr.mean()) if len(arr) else 0, 4),
        "win_rate_pct":          round(float((arr > 0).mean() * 100) if len(arr) else 0, 2),
        "sharpe":                round(float(arr.mean() / (arr.std() + 1e-9)) if len(arr) >= 5 else 0, 4),
        "avg_hold_days":         round(float(t["exit_offset"].mean()) if len(t) else 0, 2),
    }, t


# ── Per-year breakdown (key: 2014/2015 gap analysis) ─────────────────────────
def per_year_stats(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {}
    trades_df = trades_df.copy()
    trades_df["year"] = pd.to_datetime(trades_df["signal_date"]).dt.year
    result = {}
    for yr, grp in trades_df.groupby("year"):
        arr = grp["net_ret"].to_numpy()
        result[int(yr)] = {
            "trades":    int(len(arr)),
            "win_rate":  round(float((arr > 0).mean() * 100) if len(arr) else 0, 2),
            "avg_ret":   round(float(arr.mean()) if len(arr) else 0, 4),
            "sharpe":    round(float(arr.mean() / (arr.std() + 1e-9)) if len(arr) >= 5 else 0, 4),
        }
    return result


# ── Main ───────────────────────────────────────────────────────────────────────
log("Loading PEAD rows from", ROOT)
log(f"Settings: hold={PEAD_HOLD}, cost={COST}, min_adv={MIN_ADV/1e6:.0f}M, "
    f"vol_gate={PEAD_VOL_GATE}, min_surprise={PEAD_MIN_SURPRISE}, threshold={TH}")

df   = load_all_pead()
fs   = feature_cols_pead(df)
dates = pd.Index(sorted(df["date"].drop_duplicates()))

log(f"Dataset: {len(df)} PEAD rows | {df.symbol.nunique()} symbols | {len(fs)} features")
log("Feature set:", fs[:20], "..." if len(fs) > 20 else "")

# Sanity check: confirm earnings features are present
earn_cols = [c for c in df.columns if "earn" in c.lower() or "insider" in c.lower()]
log("Earnings/insider cols found:", earn_cols)
if not earn_cols:
    log("WARNING: No earnings features found in parquets! "
        "Run build_earnings_insider_features.py on the VM first.")

# Confirm PEAD rows exist and are post-earnings (not lookahead check)
log("Year distribution of PEAD signal rows:")
yr_dist = df["date"].dt.year.value_counts().sort_index()
for yr, cnt in yr_dist.items():
    log(f"  {yr}: {cnt} rows")

# ── Walk-forward folds ─────────────────────────────────────────────────────────
# Same discipline as h10: start folds at 35% of history, test on future only
chunks    = np.array_split(dates[int(len(dates) * 0.35):], FOLDS)
all_preds = []
folds     = []

for fold, ch in enumerate(chunks, 1):
    test_start = pd.Timestamp(ch[0])
    test_end   = pd.Timestamp(ch[-1])
    start_i    = dates.get_loc(test_start)
    train_end_i = max(0, start_i - PEAD_HOLD - 1)  # no lookahead: gap of HOLD+1 days

    if train_end_i < 252:
        log(f"Fold {fold}: skipping — not enough training history ({train_end_i} days)")
        continue

    train_dates = set(dates[:train_end_i])
    test_dates  = pd.Index(ch)

    train = df[df.date.isin(train_dates)].copy()
    test  = df[df.date.isin(set(test_dates))].copy()

    if len(train) > MAX_TRAIN:
        train = train.sample(MAX_TRAIN, random_state=42 + fold)

    y = (train["model_ret"] > COST).astype(int)
    pos_rate = float(y.mean())

    if y.nunique() < 2:
        log(f"Fold {fold}: skipping — only one class in training labels")
        continue

    log(f"Fold {fold}: train={len(train)} rows, test={len(test)} rows, "
        f"positive_rate={pos_rate:.3f}, test={test_start.date()}→{test_end.date()}")

    # Train PEAD model
    model = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.04,
        max_leaf_nodes=25,
        l2_regularization=0.15,
        random_state=42 + fold,
    )
    X_train = train[fs].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    model.fit(X_train, y)

    # Score test set
    X_test = test[fs].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    test = test.copy()
    test["prob"] = model.predict_proba(X_test)[:, 1]

    res, tr = simulate(test, test_dates)
    log(f"Fold {fold} result:", json.dumps(res))

    keep_cols = [
        "date", "symbol", "model_ret", "honest_ret", "entry_open",
        "effective_exit_close", "prob", "exit_offset", "exit_reason",
        "pt_hit", "pt_day", "spy_realized_vol_pct", "atr_pct",
        "earnings_surprise_momentum", "earnings_beat_rate",
    ]
    keep_cols = [c for c in keep_cols if c in test.columns]
    all_preds.append(test[keep_cols])

    folds.append({
        "fold":          fold,
        "test_start":    str(test_start.date()),
        "test_end":      str(test_end.date()),
        "train_rows":    int(len(train)),
        "test_rows":     int(len(test)),
        "positive_rate": round(pos_rate, 4),
        "test":          res,
    })


# ── Aggregate results ──────────────────────────────────────────────────────────
pred     = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
calendar = pd.Index(sorted(pred["date"].drop_duplicates())) if len(pred) else pd.Index([])
summary, trades = simulate(pred, calendar)

# Stress tests (mirror h10 structure)
odd_pred = pred[pred.date.dt.year % 2 == 1] if len(pred) else pd.DataFrame()
odd, _   = (
    simulate(odd_pred, pd.Index(sorted(odd_pred["date"].drop_duplicates())))
    if len(odd_pred) else ({}, pd.DataFrame())
)

if len(trades):
    remove_ids = set(trades.sort_values("net_ret", ascending=False).head(50)["row_id"])
    rb  = pred[~(pred["symbol"].astype(str) + "|" + pred["date"].astype(str)).isin(remove_ids)]
    rb_sum, _ = simulate(rb, pd.Index(sorted(rb["date"].drop_duplicates())))
else:
    rb_sum = {}

# Per-year breakdown — critical for validating 2014/2015 gap
yr_breakdown = per_year_stats(trades) if len(trades) else {}

# ── Write output ───────────────────────────────────────────────────────────────
out = {
    "summary": summary,
    "folds":   folds,
    "per_year": yr_breakdown,
    "stress": {
        "odd_years_only":  odd,
        "remove_best_50":  rb_sum,
    },
    "settings": {
        "root":            str(ROOT),
        "symbols":         int(df.symbol.nunique()),
        "pead_hold_days":  PEAD_HOLD,
        "entry_offset":    PEAD_ENTRY_OFFSET,
        "vol_gate":        PEAD_VOL_GATE,
        "min_surprise":    PEAD_MIN_SURPRISE,
        "threshold":       TH,
        "top_n":           TOP_N,
        "position_pct":    POS,
        "cost_pct":        COST,
        "max_open":        MAX_OPEN,
        "min_adv":         MIN_ADV,
        "features":        fs,
        "earnings_cols":   earn_cols,
    },
    "success_criteria": {
        "wr_min":         65.0,
        "sharpe_min":     1.0,
        "max_drawdown":   20.0,
        "trades_per_calm_year": 30,
        "2014_wr_min":    65.0,
        "2015_wr_min":    65.0,
    },
    "contract": (
        "walk-forward only; train dates end before test by PEAD_HOLD+1; "
        "target next open to PEAD_HOLD+1 close; "
        "signal rows are first-trading-day post-earnings only (earnings_days_to_next reset detection); "
        "no lookahead into future prices; spy_vol_gate applied in simulation only"
    ),
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, indent=2, default=str))

if len(trades):
    TRADES.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(TRADES, index=False)

print(json.dumps(out["summary"], indent=2))
log("Per-year win rates:")
for yr in sorted(yr_breakdown.keys()):
    s = yr_breakdown[yr]
    flag = " *** BELOW TARGET" if s["win_rate"] < 65 and s["trades"] >= 10 else ""
    log(f"  {yr}: WR={s['win_rate']:.1f}%  trades={s['trades']}  sharpe={s['sharpe']:.2f}{flag}")

print(f"\nWrote {OUT} and {TRADES}")

# ── Final go/no-go check ───────────────────────────────────────────────────────
log("=" * 60)
log("FINAL GO / NO-GO CHECK")
log("=" * 60)
wr  = summary.get("win_rate_pct", 0)
sh  = summary.get("sharpe", 0)
dd  = summary.get("max_drawdown_pct", 999)
n14 = yr_breakdown.get(2014, {}).get("trades", 0)
w14 = yr_breakdown.get(2014, {}).get("win_rate", 0)
n15 = yr_breakdown.get(2015, {}).get("trades", 0)
w15 = yr_breakdown.get(2015, {}).get("win_rate", 0)
total_trades = summary.get("executed_trades", 0)

checks = [
    ("Overall WR ≥ 65%",        wr  >= 65,   f"{wr:.1f}%"),
    ("Sharpe ≥ 1.0",            sh  >= 1.0,  f"{sh:.2f}"),
    ("Max drawdown ≤ 20%",      dd  <= 20,   f"{dd:.1f}%"),
    ("Total trades ≥ 50",       total_trades >= 50, str(total_trades)),
    (f"2014 WR ≥ 65% (n={n14})", w14 >= 65 or n14 < 10, f"{w14:.1f}%"),
    (f"2015 WR ≥ 65% (n={n15})", w15 >= 65 or n15 < 10, f"{w15:.1f}%"),
]
all_pass = True
for name, ok, val in checks:
    status = "PASS" if ok else "FAIL"
    if not ok:
        all_pass = False
    log(f"  [{status}] {name}: {val}")

if all_pass:
    log("RESULT: GO — proceed to train_pead_calm.py and paper trade for 30 days")
else:
    log("RESULT: NO-GO — adjust hyperparams or feature set and rerun")
log("=" * 60)
