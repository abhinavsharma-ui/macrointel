"""
Calm-Market H10 Walkforward Backtest
======================================
Regime-filtered variant of the h10 engine — trained and simulated exclusively
on calm-market rows (SPY realized vol < CALM_VOL_MAX).

Goal: fill the performance gap in 2014 (59.9% WR) and 2015 (56.0% WR) where
the production h10 model underperforms. The h10 model is crisis alpha; this is
calm alpha.

Changes from fixed_return_h10_walkforward_sim.py:
  1. Load-time calm filter: only rows where spy_realized_vol_pct < CALM_VOL_MAX
  2. Feature set includes unique parquet columns h10 never sees:
       weighted_sentiment_zscore, filing_change_score, alpha_signal,
       event_move_strength, bb_position, vol_regime_stressed, hist_vol_30
  3. spy_realized_vol_pct is a PRIMARY feature (model learns calm-regime shape)
  4. Outputs to reports/calm_h10_walkforward_sim.json

DO NOT modify fixed_return_h10_model.joblib or its training script.
Deploy only after 30 paper trades showing ≥65% WR.
"""

import json, os, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.ensemble import HistGradientBoostingClassifier

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT   = Path(os.getenv("SIM_ROOT",   "data/features_26yr_liquid"))
OUT    = Path(os.getenv("SIM_OUT",    "reports/calm_h10_walkforward_sim.json"))
TRADES = Path(os.getenv("SIM_TRADES", "reports/calm_h10_walkforward_trades.csv"))

# ── Constants (mirrors h10 defaults unless noted) ──────────────────────────────
HOLD           = int(  os.getenv("SIM_HOLD_DAYS",           "10"))
COST           = float(os.getenv("SIM_TOTAL_COST_PCT",       "0.45"))
MIN_ADV        = float(os.getenv("SIM_MIN_ADV20_DOLLAR_VOL", "5000000"))
MIN_PRICE      = float(os.getenv("SIM_MIN_ENTRY_PRICE",      "5"))
MAX_ABS_RET    = float(os.getenv("SIM_MAX_ABS_HONEST_RET",   "100"))
TH             = float(os.getenv("SIM_THRESHOLD",            "0.55"))
TOP_N          = int(  os.getenv("SIM_TOP_N",                "5"))
POS            = float(os.getenv("SIM_POSITION_PCT",         "0.002"))
MAX_OPEN       = int(  os.getenv("SIM_MAX_OPEN_POSITIONS",   "50"))
MAX_TRAIN      = int(  os.getenv("SIM_MAX_TRAIN_ROWS",       "300000"))
FOLDS          = int(  os.getenv("SIM_FOLDS",                "4"))
INIT           = float(os.getenv("SIM_INITIAL_CAPITAL",      "100000"))
PROFIT_TARGET  = float(os.getenv("SIM_PROFIT_TARGET_PCT",    "0") or 0)
# ── Calm-specific ──────────────────────────────────────────────────────────────
CALM_VOL_MAX   = float(os.getenv("CALM_VOL_MAX", "20.0"))  # only trade when SPY vol < this


def log(*x):
    print("CALM_SIM", *x, flush=True)


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


# ── SPY vol map ────────────────────────────────────────────────────────────────
def load_spy_vol_map() -> dict:
    candidates = [ROOT / "SPY.parquet", ROOT / "SPY_US.parquet", ROOT / "SPY.US.parquet"]
    p = next((x for x in candidates if x.exists()), None)
    if p is None:
        log("WARNING: SPY parquet not found — defaulting to 20.0")
        return {}
    x = read_price_file(p, need_symbol=False)
    if x is None or x.empty:
        return {}
    ret = x["close"].pct_change()
    vol = ret.rolling(20, min_periods=10).std() * np.sqrt(252) * 100.0
    d   = pd.to_datetime(x["date"]).dt.normalize()
    return dict(zip(d, vol.fillna(20.0)))


SPY_VOL = load_spy_vol_map()


# ── Per-symbol loader ──────────────────────────────────────────────────────────
def load_one(p: Path) -> "pd.DataFrame | None":
    sym = norm_sym(p)
    if sym.endswith((".NS", ".BO", ".NSE", ".BSE")):
        return None

    x = read_price_file(p)
    if x is None:
        return None

    x = x[x["adv20_dollar_vol"].fillna(0) >= MIN_ADV].copy()
    x = x[x["close"].fillna(0) >= MIN_PRICE].copy()
    if x.empty:
        return None

    # Label construction (identical to h10)
    x["entry_open"]          = x["open"].shift(-1)
    x["default_exit_close"]  = x["close"].shift(-(HOLD + 1))
    x["model_ret"]           = (x["default_exit_close"] / x["entry_open"] - 1) * 100
    x["effective_exit_close"] = x["default_exit_close"]
    x["exit_offset"]         = HOLD + 1
    x["pt_hit"]              = False
    x["pt_day"]              = np.nan
    x["exit_reason"]         = "hold"

    if PROFIT_TARGET > 0:
        high_col = "high" if "high" in x.columns else "close"
        hit_so_far = pd.Series(False, index=x.index)
        for k in range(1, HOLD + 2):
            future_high  = x[high_col].shift(-k)
            future_close = x["close"].shift(-k)
            hit = ((future_high / x["entry_open"] - 1) * 100 >= PROFIT_TARGET).fillna(False)
            first = hit & ~hit_so_far
            if first.any():
                x.loc[first, "effective_exit_close"] = future_close.loc[first]
                x.loc[first, "exit_offset"]  = k
                x.loc[first, "pt_hit"]       = True
                x.loc[first, "pt_day"]       = k
                x.loc[first, "exit_reason"]  = "profit_target"
            hit_so_far = hit_so_far | hit

    x["honest_ret"] = (x["effective_exit_close"] / x["entry_open"] - 1) * 100
    x["spy_realized_vol_pct"] = (
        pd.to_datetime(x["date"]).dt.normalize().map(SPY_VOL).fillna(20.0)
    )

    # ── CALM FILTER: only keep rows in calm regime ─────────────────────────────
    x = x[x["spy_realized_vol_pct"] < CALM_VOL_MAX].copy()
    if x.empty:
        return None

    x = x.replace([np.inf, -np.inf], np.nan).dropna(subset=["model_ret", "honest_ret"])
    x = x[x["model_ret"].abs() <= MAX_ABS_RET]
    x = x[x["honest_ret"].abs() <= MAX_ABS_RET]
    return x


def load_all() -> pd.DataFrame:
    rows  = []
    files = sorted(ROOT.glob("*.parquet"))
    for i, p in enumerate(files, 1):
        x = load_one(p)
        if x is not None and len(x):
            rows.append(x)
        if i % 100 == 0:
            log(f"  {i}/{len(files)} files | calm_rows: {sum(len(r) for r in rows)}")
    if not rows:
        raise RuntimeError("No calm-regime rows found. Check SPY_VOL map and CALM_VOL_MAX.")
    df = pd.concat(rows, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


# ── Feature columns ────────────────────────────────────────────────────────────
def feature_cols(df: pd.DataFrame) -> list:
    banned = {
        "date", "symbol", "year", "open", "high", "low", "close", "volume", "adj_close",
        "entry_open", "default_exit_close", "effective_exit_close", "exit_close",
        "model_ret", "honest_ret", "adv20_dollar_vol", "pt_hit", "pt_day",
        "exit_reason", "exit_offset",
    }
    out = []
    for c in df.columns:
        cl = c.lower()
        if c in banned:
            continue
        if (cl.startswith("gross_ret_") or cl.startswith("exit_timestamp") or
                cl.startswith("future") or cl.startswith("next") or
                cl.startswith("sma_") or
                "target" in cl or "label" in cl):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)

    # Priority order — spy_realized_vol_pct leads as the regime-defining feature
    pref = [
        "spy_realized_vol_pct",       # regime definer — primary for calm model
        "weighted_sentiment_zscore",  # NLP sentiment — unique to these parquets
        "alpha_signal",               # pre-computed alpha
        "filing_change_score",        # filing activity signal
        "event_move_strength",        # event-driven move magnitude
        "bb_position",                # Bollinger Band position
        "vol_regime_stressed",        # stressed vol indicator
        "hist_vol_30",                # 30-day historical vol
        "vol_ratio_20",               # volume ratio
        "close_reversal_signal",      # reversal
        "close_reversal_strength",
        "news_volume_spike",
        "peer_earnings_negative_ratio_7d",
        # Standard h10 features below
        "return_20d", "return_60d", "return_5d", "return_1d",
        "momentum_composite", "momentum_20d", "momentum_60d",
        "vol_regime_ratio", "atr_pct", "price_acceleration",
        "close_vs_sma_50", "close_vs_sma_200", "rsi_14",
        "realized_vol_21d",
    ]
    return [c for c in pref if c in out] + [c for c in out if c not in pref]


# ── Simulation (identical to h10) ─────────────────────────────────────────────
def simulate(pred: pd.DataFrame, calendar_dates: pd.Index) -> "tuple[dict, pd.DataFrame]":
    pred = pred[pred["prob"] >= TH].copy()
    if pred.empty:
        return {}, pd.DataFrame()

    pred["row_id"] = pred["symbol"].astype(str) + "|" + pred["date"].astype(str)
    by_date = {d: g.sort_values("prob", ascending=False) for d, g in pred.groupby("date")}

    eq = INIT; peak = INIT; dd = 0.0; active = []; trades = []

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
            net      = float(r.honest_ret) - COST
            notional = eq * POS
            pnl      = notional * net / 100
            off      = int(r.get("exit_offset", HOLD + 1) or HOLD + 1)
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
                "spy_realized_vol_pct": float(r.get("spy_realized_vol_pct", 15.0)),
            })
            active_syms.add(sym); new += 1

    final_date = str(calendar_dates[-1]) if len(calendar_dates) else ""
    for a in active:
        eq += a["pnl"]
        trades.append({**a, "exit_date": final_date, "status": "closed_final"})

    t   = pd.DataFrame(trades)
    arr = t["net_ret"].to_numpy() if len(t) else np.array([])

    return {
        "trades":               int(len(pred)),
        "executed_trades":      int(len(t)),
        "final_equity":         round(float(eq), 2),
        "return_pct":           round((eq / INIT - 1) * 100, 4),
        "max_drawdown_pct":     round(float(dd), 4),
        "avg_trade_net_return": round(float(arr.mean()) if len(arr) else 0, 4),
        "win_rate_pct":         round(float((arr > 0).mean() * 100) if len(arr) else 0, 2),
        "sharpe":               round(float(arr.mean() / (arr.std() + 1e-9)) if len(arr) >= 5 else 0, 4),
        "avg_hold_days":        round(float(t["exit_offset"].mean()) if len(t) else 0, 2),
    }, t


def per_year_stats(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {}
    trades_df = trades_df.copy()
    trades_df["year"] = pd.to_datetime(trades_df["signal_date"]).dt.year
    result = {}
    for yr, grp in trades_df.groupby("year"):
        arr = grp["net_ret"].to_numpy()
        result[int(yr)] = {
            "trades":   int(len(arr)),
            "win_rate": round(float((arr > 0).mean() * 100) if len(arr) else 0, 2),
            "avg_ret":  round(float(arr.mean()) if len(arr) else 0, 4),
            "sharpe":   round(float(arr.mean() / (arr.std() + 1e-9)) if len(arr) >= 5 else 0, 4),
        }
    return result


# ── Main ───────────────────────────────────────────────────────────────────────
log(f"Settings: hold={HOLD}, cost={COST}, calm_vol_max={CALM_VOL_MAX}, "
    f"threshold={TH}, min_adv={MIN_ADV/1e6:.0f}M")

log("Loading calm-regime rows...")
df    = load_all()
fs    = feature_cols(df)
dates = pd.Index(sorted(df["date"].drop_duplicates()))

log(f"Dataset: {len(df)} calm rows | {df.symbol.nunique()} symbols | {len(fs)} features")
log("Top features:", fs[:15])

log("Year distribution:")
for yr, cnt in df["date"].dt.year.value_counts().sort_index().items():
    log(f"  {yr}: {cnt} rows")

# ── Walk-forward folds ─────────────────────────────────────────────────────────
chunks    = np.array_split(dates[int(len(dates) * 0.35):], FOLDS)
all_preds = []
folds     = []

for fold, ch in enumerate(chunks, 1):
    test_start  = pd.Timestamp(ch[0])
    test_end    = pd.Timestamp(ch[-1])
    start_i     = dates.get_loc(test_start)
    train_end_i = max(0, start_i - HOLD - 1)

    if train_end_i < 504:
        log(f"Fold {fold}: skipping — insufficient history")
        continue

    train_dates = set(dates[:train_end_i])
    test_dates  = pd.Index(ch)

    train = df[df.date.isin(train_dates)].copy()
    test  = df[df.date.isin(set(test_dates))].copy()

    if len(train) > MAX_TRAIN:
        train = train.sample(MAX_TRAIN, random_state=42 + fold)

    y        = (train["model_ret"] > COST).astype(int)
    pos_rate = float(y.mean())

    if y.nunique() < 2:
        log(f"Fold {fold}: skipping — single class (pos_rate={pos_rate:.3f})")
        continue

    log(f"Fold {fold}: train={len(train)} test={len(test)} "
        f"pos_rate={pos_rate:.3f} {test_start.date()}→{test_end.date()}")

    model = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.045,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=42 + fold,
    )
    X_tr = train[fs].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    model.fit(X_tr, y)

    X_te = test[fs].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    test = test.copy()
    test["prob"] = model.predict_proba(X_te)[:, 1]

    res, tr = simulate(test, test_dates)
    log(f"Fold {fold}: {json.dumps(res)}")

    keep = [c for c in [
        "date", "symbol", "model_ret", "honest_ret", "entry_open",
        "effective_exit_close", "prob", "exit_offset", "exit_reason",
        "pt_hit", "pt_day", "spy_realized_vol_pct", "atr_pct",
        "weighted_sentiment_zscore", "alpha_signal",
    ] if c in test.columns]
    all_preds.append(test[keep])

    folds.append({
        "fold":          fold,
        "test_start":    str(test_start.date()),
        "test_end":      str(test_end.date()),
        "train_rows":    int(len(train)),
        "test_rows":     int(len(test)),
        "positive_rate": round(pos_rate, 4),
        "test":          res,
    })

# ── Aggregate ─────────────────────────────────────────────────────────────────
pred     = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
calendar = pd.Index(sorted(pred["date"].drop_duplicates())) if len(pred) else pd.Index([])
summary, trades = simulate(pred, calendar)

odd_pred = pred[pred.date.dt.year % 2 == 1] if len(pred) else pd.DataFrame()
odd, _   = (
    simulate(odd_pred, pd.Index(sorted(odd_pred["date"].drop_duplicates())))
    if len(odd_pred) else ({}, pd.DataFrame())
)
if len(trades):
    remove_ids = set(trades.sort_values("net_ret", ascending=False).head(50)["row_id"])
    rb = pred[~(pred["symbol"].astype(str) + "|" + pred["date"].astype(str)).isin(remove_ids)]
    rb_sum, _ = simulate(rb, pd.Index(sorted(rb["date"].drop_duplicates())))
else:
    rb_sum = {}

yr_breakdown = per_year_stats(trades) if len(trades) else {}

# ── Output ────────────────────────────────────────────────────────────────────
out = {
    "summary":  summary,
    "folds":    folds,
    "per_year": yr_breakdown,
    "stress":   {"odd_years_only": odd, "remove_best_50": rb_sum},
    "settings": {
        "root":          str(ROOT),
        "calm_vol_max":  CALM_VOL_MAX,
        "symbols":       int(df.symbol.nunique()),
        "hold_days":     HOLD,
        "threshold":     TH,
        "top_n":         TOP_N,
        "position_pct":  POS,
        "cost_pct":      COST,
        "features":      fs,
    },
    "success_criteria": {
        "wr_min": 65.0, "sharpe_min": 1.0, "max_drawdown": 20.0,
        "2014_wr_min": 65.0, "2015_wr_min": 65.0,
    },
    "contract": (
        "walk-forward only; calm filter applied at load time (spy_vol < calm_vol_max); "
        "no h10 model modification; same label and fold structure as h10"
    ),
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, indent=2, default=str))
if len(trades):
    TRADES.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(TRADES, index=False)

print(json.dumps(out["summary"], indent=2))
log("Per-year breakdown:")
for yr in sorted(yr_breakdown.keys()):
    s = yr_breakdown[yr]
    flag = " *** BELOW TARGET" if s["win_rate"] < 65 and s["trades"] >= 15 else ""
    log(f"  {yr}: WR={s['win_rate']:.1f}%  n={s['trades']}  sharpe={s['sharpe']:.2f}{flag}")

log(f"Wrote {OUT} and {TRADES}")

# ── Go / No-Go ─────────────────────────────────────────────────────────────────
log("=" * 60)
wr  = summary.get("win_rate_pct", 0)
sh  = summary.get("sharpe", 0)
dd  = summary.get("max_drawdown_pct", 999)
n   = summary.get("executed_trades", 0)
w14 = yr_breakdown.get(2014, {}).get("win_rate", 0); n14 = yr_breakdown.get(2014, {}).get("trades", 0)
w15 = yr_breakdown.get(2015, {}).get("win_rate", 0); n15 = yr_breakdown.get(2015, {}).get("trades", 0)

checks = [
    ("Overall WR ≥ 65%",           wr >= 65,          f"{wr:.1f}%"),
    ("Sharpe ≥ 1.0",               sh >= 1.0,          f"{sh:.2f}"),
    ("Max DD ≤ 20%",               dd <= 20,           f"{dd:.1f}%"),
    ("Trades ≥ 100",               n  >= 100,          str(n)),
    (f"2014 WR ≥65% (n={n14})",    w14>=65 or n14<15,  f"{w14:.1f}%"),
    (f"2015 WR ≥65% (n={n15})",    w15>=65 or n15<15,  f"{w15:.1f}%"),
]
all_pass = True
for name, ok, val in checks:
    if not ok:
        all_pass = False
    log(f"  [{'PASS' if ok else 'FAIL'}] {name}: {val}")

log("RESULT:", "GO — run train_calm_h10.py then paper trade 30 days" if all_pass
    else "NO-GO — tune CALM_VOL_MAX, threshold, or feature set")
log("=" * 60)
