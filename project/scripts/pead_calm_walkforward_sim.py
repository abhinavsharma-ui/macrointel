"""
PEAD Calm-Market Walkforward Backtest  (v2 — yfinance earnings lookup)
=======================================================================
Post-Earnings Announcement Drift model — SECOND LANE of the regime-complete system.

Data source for earnings events: data/earnings_lookup.csv
  (built by scripts/fetch_earnings_lookup.py using yfinance — no Finnhub needed)

Strategy:
  - Universe: rows where earnings happened 1 trading day ago + beat by > PEAD_MIN_SURPRISE%
               + SPY realized vol < PEAD_VOL_GATE (calm regime)
  - Entry:  next day open after signal (E+2 open — ~2 trading days post-announcement)
  - Hold:   PEAD_HOLD (20) days
  - Label:  (exit_close / entry_open - 1) * 100 > COST  →  1 else 0

Walk-forward discipline mirrors fixed_return_h10_walkforward_sim.py exactly:
  - FOLDS=4, train on past only, test on strictly future fold
  - train_end < test_start by HOLD+1 days (zero lookahead)

Outputs:
  reports/pead_calm_walkforward_sim.json
  reports/pead_calm_walkforward_trades.csv

DO NOT modify fixed_return_h10_model.joblib or its training script.
"""

import json, os, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.ensemble import HistGradientBoostingClassifier

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT           = Path(os.getenv("PEAD_ROOT",    "data/features_26yr_liquid"))
EARNINGS_CSV   = Path(os.getenv("EARNINGS_LOOKUP", "data/earnings_lookup.csv"))
OUT            = Path(os.getenv("PEAD_OUT",     "reports/pead_calm_walkforward_sim.json"))
TRADES         = Path(os.getenv("PEAD_TRADES",  "reports/pead_calm_walkforward_trades.csv"))

# ── Constants ──────────────────────────────────────────────────────────────────
PEAD_HOLD         = int(  os.getenv("PEAD_HOLD",        "20"))
COST              = float(os.getenv("PEAD_COST_PCT",     "0.45"))
MIN_ADV           = float(os.getenv("PEAD_MIN_ADV",      "10000000"))
MIN_PRICE         = float(os.getenv("PEAD_MIN_PRICE",    "10"))
PEAD_VOL_GATE     = float(os.getenv("PEAD_VOL_GATE",     "20.0"))
PEAD_MIN_SURPRISE = float(os.getenv("PEAD_MIN_SURPRISE", "2.0"))   # EPS surprise % threshold
MAX_ABS_RET       = float(os.getenv("PEAD_MAX_ABS_RET",  "100"))
TH                = float(os.getenv("PEAD_THRESHOLD",    "0.55"))
TOP_N             = int(  os.getenv("PEAD_TOP_N",        "5"))
POS               = float(os.getenv("PEAD_POSITION_PCT", "0.002"))
MAX_OPEN          = int(  os.getenv("PEAD_MAX_OPEN",     "30"))
MAX_TRAIN         = int(  os.getenv("PEAD_MAX_TRAIN",    "200000"))
FOLDS             = int(  os.getenv("PEAD_FOLDS",        "4"))
INIT              = float(os.getenv("PEAD_INITIAL_CAP",  "100000"))

# ── Feature candidates (all available in parquets — no earnings stub cols) ────
PEAD_FEATURE_CANDIDATES = [
    # Post-earnings price reaction (the core PEAD driver)
    "return_1d",            # day+0 reaction — strong predictor of drift direction
    "return_5d",            # 1-week drift already in motion
    "return_20d",           # medium-term context
    # Earnings quality (from CSV join)
    "surprise_pct",         # EPS beat magnitude — the primary PEAD predictor
    # Momentum / trend
    "momentum_composite",
    "momentum_20d",
    "momentum_60d",
    "price_acceleration",
    "close_vs_sma_50",
    "close_vs_sma_200",
    "rsi_14",
    # Volatility / regime
    "spy_realized_vol_pct", # ALLOWED here — used as feature + gate
    "hist_vol_30",
    "vol_regime_ratio",
    "vol_regime_stressed",
    "atr_pct",
    # Volume confirmation
    "vol_ratio_20",
    "news_volume_spike",
    # Event strength signals (non-zero in parquets)
    "event_move_strength",
    "filing_change_score",
    "weighted_sentiment_zscore",
    "alpha_signal",
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


# ── Earnings lookup ────────────────────────────────────────────────────────────
def load_earnings_lookup() -> dict:
    """
    Returns dict: {(symbol_upper, date_normalized): surprise_pct}
    date is the earnings announcement date (day E).
    Signal fires on E+1, entry on E+2 open.
    """
    if not EARNINGS_CSV.exists():
        raise FileNotFoundError(
            f"Earnings lookup not found at {EARNINGS_CSV}. "
            "Run scripts/fetch_earnings_lookup.py first."
        )
    df = pd.read_csv(EARNINGS_CSV, parse_dates=["earnings_date"])
    df["earnings_date"] = pd.to_datetime(df["earnings_date"]).dt.normalize()
    df["symbol"] = df["symbol"].str.upper()
    # Only keep beats above threshold (or allow all for feature use)
    lookup = {}
    for _, row in df.iterrows():
        key = (row["symbol"], row["earnings_date"])
        lookup[key] = float(row.get("surprise_pct", 0) or 0)
    log(f"Earnings lookup: {len(lookup)} events | "
        f"{df.symbol.nunique()} symbols | "
        f"{df.earnings_date.min().date()} → {df.earnings_date.max().date()}")
    return lookup


# ── SPY realized vol map ───────────────────────────────────────────────────────
def load_spy_vol_map() -> dict:
    candidates = [ROOT / "SPY.parquet", ROOT / "SPY_US.parquet", ROOT / "SPY.US.parquet"]
    p = next((x for x in candidates if x.exists()), None)
    if p is None:
        log("WARNING: SPY parquet not found — defaulting vol to 20.0")
        return {}
    x = read_price_file(p, need_symbol=False)
    if x is None or x.empty:
        return {}
    ret = x["close"].pct_change()
    vol = ret.rolling(20, min_periods=10).std() * np.sqrt(252) * 100.0
    d   = pd.to_datetime(x["date"]).dt.normalize()
    return dict(zip(d, vol.fillna(20.0)))


# ── Per-symbol PEAD loader ─────────────────────────────────────────────────────
def load_one_pead(p: Path, earnings_lookup: dict, spy_vol: dict) -> "pd.DataFrame | None":
    sym = norm_sym(p)
    if sym.endswith((".NS", ".BO", ".NSE", ".BSE")):
        return None

    x = read_price_file(p)
    if x is None or x.empty:
        return None

    # Liquidity / price filter
    x = x[x["adv20_dollar_vol"].fillna(0) >= MIN_ADV].copy()
    x = x[x["close"].fillna(0) >= MIN_PRICE].copy()
    if len(x) < 60:
        return None

    # ── Detect post-earnings signal rows ─────────────────────────────────────
    # For each row, check if earnings were announced on the PREVIOUS trading day.
    # (yfinance earnings_date = announcement date = day E)
    # Signal row = E+1, entry_open = E+2 open (shift(-1))
    x["date_norm"] = pd.to_datetime(x["date"]).dt.normalize()
    prev_dates = x["date_norm"].shift(1)

    # Look up surprise for (sym, prev_date) — fires on E+1
    def get_surprise(prev_date):
        if pd.isna(prev_date):
            return float("nan")
        return earnings_lookup.get((sym, prev_date), float("nan"))

    x["surprise_pct"]   = prev_dates.map(get_surprise)
    x["pead_entry_flag"] = x["surprise_pct"].notna().astype(int)

    # ── Label construction ────────────────────────────────────────────────────
    x["entry_open"]           = x["open"].shift(-1)           # E+2 open
    x["default_exit_close"]   = x["close"].shift(-(PEAD_HOLD + 1))  # 20 days later
    x["model_ret"]            = (x["default_exit_close"] / x["entry_open"] - 1) * 100
    x["honest_ret"]           = x["model_ret"]
    x["effective_exit_close"] = x["default_exit_close"]
    x["exit_offset"]          = PEAD_HOLD + 1
    x["exit_reason"]          = "hold"
    x["pt_hit"]               = False
    x["pt_day"]               = np.nan

    # SPY vol per row
    x["spy_realized_vol_pct"] = x["date_norm"].map(spy_vol).fillna(20.0)

    # ── Apply PEAD filters ────────────────────────────────────────────────────
    # 1. Only post-earnings rows
    x = x[x["pead_entry_flag"] == 1].copy()
    if x.empty:
        return None

    # 2. Only meaningful EPS beats
    x = x[x["surprise_pct"].fillna(-999) >= PEAD_MIN_SURPRISE].copy()
    if x.empty:
        return None

    # 3. Calm regime only (aligns training distribution with deployment)
    x = x[x["spy_realized_vol_pct"].fillna(99) < PEAD_VOL_GATE].copy()
    if x.empty:
        return None

    # Clean up
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.dropna(subset=["model_ret", "honest_ret", "entry_open", "default_exit_close"])
    x = x[x["model_ret"].abs() <= MAX_ABS_RET]
    x = x[x["honest_ret"].abs() <= MAX_ABS_RET]

    return x


def load_all_pead(earnings_lookup: dict, spy_vol: dict) -> pd.DataFrame:
    rows  = []
    files = sorted(ROOT.glob("*.parquet"))
    for i, p in enumerate(files, 1):
        x = load_one_pead(p, earnings_lookup, spy_vol)
        if x is not None and len(x):
            rows.append(x)
        if i % 100 == 0:
            log(f"  {i}/{len(files)} files | pead_rows: {sum(len(r) for r in rows)}")
    if not rows:
        raise RuntimeError(
            "No PEAD rows found after applying all filters.\n"
            "Check: (1) earnings_lookup.csv has data for these symbols, "
            "(2) PEAD_MIN_SURPRISE is not too high, (3) PEAD_VOL_GATE not too tight."
        )
    df = pd.concat(rows, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


# ── Feature column selection ───────────────────────────────────────────────────
def feature_cols_pead(df: pd.DataFrame) -> list:
    available = set(df.columns)
    banned = {
        "date", "date_norm", "symbol", "open", "high", "low", "close", "volume", "adj_close",
        "entry_open", "default_exit_close", "effective_exit_close", "exit_close",
        "model_ret", "honest_ret", "adv20_dollar_vol", "pt_hit", "pt_day",
        "exit_reason", "exit_offset", "pead_entry_flag",
    }
    selected = [c for c in PEAD_FEATURE_CANDIDATES if c in available and c not in banned]
    extra = [
        c for c in df.columns
        if c not in banned and c not in selected
        and not any(c.lower().startswith(pfx) for pfx in
                    ("future", "next", "gross_ret_", "exit_timestamp", "sma_"))
        and "target" not in c.lower() and "label" not in c.lower()
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return selected + extra


# ── Simulation engine ──────────────────────────────────────────────────────────
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
            spy_vol = float(r.get("spy_realized_vol_pct", 99.0))
            if spy_vol >= PEAD_VOL_GATE:
                skipped += 1
                continue

            net      = float(r.honest_ret) - COST
            notional = eq * POS
            pnl      = notional * net / 100
            off      = int(r.get("exit_offset", PEAD_HOLD + 1) or PEAD_HOLD + 1)

            active.append({
                "row_id":      r.row_id,
                "symbol":      sym,
                "signal_date": str(d),
                "prob":        float(r.prob),
                "surprise_pct": float(r.get("surprise_pct", 0)),
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
        "trades":               int(len(pred)),
        "executed_trades":      int(len(t)),
        "skipped_vol_gate":     int(skipped),
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
log(f"Settings: hold={PEAD_HOLD}, cost={COST}, min_adv={MIN_ADV/1e6:.0f}M, "
    f"vol_gate={PEAD_VOL_GATE}, min_surprise={PEAD_MIN_SURPRISE}%, threshold={TH}")

earnings_lookup = load_earnings_lookup()
spy_vol         = load_spy_vol_map()
log(f"SPY vol map: {len(spy_vol)} dates")

log("Loading PEAD rows...")
df   = load_all_pead(earnings_lookup, spy_vol)
fs   = feature_cols_pead(df)
dates = pd.Index(sorted(df["date"].drop_duplicates()))

log(f"Dataset: {len(df)} PEAD rows | {df.symbol.nunique()} symbols | {len(fs)} features")
log("Features:", fs)

log("Year distribution of PEAD signal rows:")
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
    train_end_i = max(0, start_i - PEAD_HOLD - 1)

    if train_end_i < 252:
        log(f"Fold {fold}: skipping — insufficient training history")
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
        log(f"Fold {fold}: skipping — single class in labels (pos_rate={pos_rate:.3f})")
        continue

    log(f"Fold {fold}: train={len(train)}, test={len(test)}, "
        f"pos_rate={pos_rate:.3f}, {test_start.date()}→{test_end.date()}")

    model = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.04,
        max_leaf_nodes=25,
        l2_regularization=0.15,
        random_state=42 + fold,
    )
    X_tr = train[fs].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    model.fit(X_tr, y)

    X_te = test[fs].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    test = test.copy()
    test["prob"] = model.predict_proba(X_te)[:, 1]

    res, tr = simulate(test, test_dates)
    log(f"Fold {fold} result: {json.dumps(res)}")

    keep = [c for c in [
        "date", "symbol", "model_ret", "honest_ret", "entry_open",
        "effective_exit_close", "prob", "exit_offset", "exit_reason",
        "pt_hit", "pt_day", "spy_realized_vol_pct", "surprise_pct",
        "return_1d", "atr_pct",
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

# ── Write output ───────────────────────────────────────────────────────────────
out = {
    "summary":  summary,
    "folds":    folds,
    "per_year": yr_breakdown,
    "stress":   {"odd_years_only": odd, "remove_best_50": rb_sum},
    "settings": {
        "root":           str(ROOT),
        "earnings_csv":   str(EARNINGS_CSV),
        "symbols":        int(df.symbol.nunique()),
        "pead_hold_days": PEAD_HOLD,
        "vol_gate":       PEAD_VOL_GATE,
        "min_surprise":   PEAD_MIN_SURPRISE,
        "threshold":      TH,
        "top_n":          TOP_N,
        "position_pct":   POS,
        "cost_pct":       COST,
        "features":       fs,
    },
    "success_criteria": {
        "wr_min": 65.0, "sharpe_min": 1.0, "max_drawdown": 20.0,
        "2014_wr_min": 65.0, "2015_wr_min": 65.0,
    },
    "contract": (
        "walk-forward only; earnings events from yfinance CSV (no parquet lookahead); "
        "signal=E+1 row, entry=E+2 open, exit=20d close; "
        "spy_vol_gate applied at simulation time; no gross_ret leakage"
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
    flag = " *** BELOW TARGET" if s["win_rate"] < 65 and s["trades"] >= 10 else ""
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
    ("Overall WR ≥ 65%",   wr >= 65,          f"{wr:.1f}%"),
    ("Sharpe ≥ 1.0",       sh >= 1.0,          f"{sh:.2f}"),
    ("Max DD ≤ 20%",       dd <= 20,           f"{dd:.1f}%"),
    ("Trades ≥ 50",        n  >= 50,           str(n)),
    (f"2014 WR ≥65% n={n14}", w14>=65 or n14<10, f"{w14:.1f}%"),
    (f"2015 WR ≥65% n={n15}", w15>=65 or n15<10, f"{w15:.1f}%"),
]
all_pass = True
for name, ok, val in checks:
    if not ok: all_pass = False
    log(f"  [{'PASS' if ok else 'FAIL'}] {name}: {val}")

log("RESULT:", "GO — run train_pead_calm.py then paper trade 30 days" if all_pass else "NO-GO — tune params and rerun")
log("=" * 60)
