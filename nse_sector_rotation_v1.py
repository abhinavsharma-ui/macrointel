from pathlib import Path
import json
import math
import pandas as pd
import numpy as np

COST = 0.22
HOLD = 20
VIX_THRESHOLD = 20.0
TOP_K = 3
MAX_PER_SECTOR_WEIGHT = 1.0 / TOP_K

SECTOR_DIR = Path("data/sector_indices")
NIFTY_PATH = Path("data/prices_full/NIFTY50.parquet")
VIX_PATH = Path("data/prices_full/INDIAVIX.parquet")
REPORT = Path("reports/nse_sector_rotation_v1.json")
TRADES = Path("reports/nse_sector_rotation_v1_trades.csv")

FOLDS = [
    (1, "2013-01-01", "2015-01-01"),
    (2, "2017-01-01", "2019-01-01"),
    (3, "2021-01-01", "2023-01-01"),
    (4, "2024-01-01", "2026-01-01"),
]

def load_index(path, name):
    df = pd.read_parquet(path).copy()

    # Normalize yfinance/NSE parquet shapes safely.
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    cols = []
    seen = {}
    for c in df.columns:
        base = str(c).lower().replace(" ", "_")
        if base in seen:
            seen[base] += 1
            base = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
        cols.append(base)
    df.columns = cols

    date_col = None
    for c in df.columns:
        if c in {"date", "datetime", "index"} or "date" in c:
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    close_col = "adj_close" if "adj_close" in df.columns else "close"
    if close_col not in df.columns:
        close_candidates = [c for c in df.columns if "close" in c]
        if not close_candidates:
            raise ValueError(f"No close column in {path}: {list(df.columns)}")
        close_col = close_candidates[0]

    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None).dt.normalize(),
        name: pd.to_numeric(df[close_col], errors="coerce"),
    })
    out = out.dropna().sort_values("date").drop_duplicates("date", keep="last")
    return out

def build_panel():
    nifty = load_index(NIFTY_PATH, "nifty")
    vix = load_index(VIX_PATH, "vix")
    panel = nifty.merge(vix, on="date", how="inner")

    sector_frames = []
    for p in sorted(SECTOR_DIR.glob("*.parquet")):
        name = p.stem
        s = load_index(p, "close")
        s["sector"] = name
        sector_frames.append(s)
    if not sector_frames:
        raise SystemExit("No sector index parquet files found")

    sec = pd.concat(sector_frames, ignore_index=True)
    sec = sec.merge(panel, on="date", how="inner")
    sec = sec.sort_values(["sector", "date"])

    for d in (20, 60, 120):
        sec[f"sec_ret_{d}"] = sec.groupby("sector")["close"].pct_change(d) * 100.0
        sec[f"nifty_ret_{d}"] = sec["nifty"].pct_change(d) * 100.0
        sec[f"rel_mom_{d}"] = sec[f"sec_ret_{d}"] - sec[f"nifty_ret_{d}"]

    sec["fwd_sec_ret"] = sec.groupby("sector")["close"].shift(-HOLD) / sec["close"] * 100.0 - 100.0
    sec["fwd_nifty_ret"] = sec["nifty"].shift(-HOLD) / sec["nifty"] * 100.0 - 100.0
    sec["rel_ret"] = sec["fwd_sec_ret"] - sec["fwd_nifty_ret"] - COST

    sec["score"] = (
        0.50 * sec["rel_mom_60"].fillna(0)
        + 0.30 * sec["rel_mom_120"].fillna(0)
        + 0.20 * sec["rel_mom_20"].fillna(0)
    )
    return sec.dropna(subset=["rel_ret", "score", "vix"])

def eval_fold(df, fold, start, end):
    test = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] < pd.Timestamp(end))].copy()
    test = test[test["vix"] <= VIX_THRESHOLD].copy()

    rows = []
    for dt, g in test.groupby("date"):
        g = g.dropna(subset=["score", "rel_ret"])
        if len(g) < TOP_K:
            continue
        # choose top sectors by information available at close t;
        # result is t -> t+HOLD relative return.
        pick = g.nlargest(TOP_K, "score")
        for _, r in pick.iterrows():
            rows.append({
                "fold": fold,
                "date": dt.date().isoformat(),
                "sector": r["sector"],
                "score": round(float(r["score"]), 4),
                "rel_ret": float(r["rel_ret"]),
                "sec_ret": float(r["fwd_sec_ret"]),
                "nifty_ret": float(r["fwd_nifty_ret"]),
                "vix": float(r["vix"]),
            })

    trades = pd.DataFrame(rows)
    if trades.empty:
        return {"fold": fold, "trades": 0, "rel_wr": 0, "avg_rel": 0, "sharpe": 0, "equity": 100000}, trades

    avg_by_day = trades.groupby("date")["rel_ret"].mean()
    rel_wr = float((trades["rel_ret"] > 0).mean() * 100.0)
    avg_rel = float(trades["rel_ret"].mean())
    sharpe = float(avg_by_day.mean() / max(avg_by_day.std(ddof=1), 1e-9) * math.sqrt(252 / HOLD)) if len(avg_by_day) > 1 else 0.0
    equity = float(100000 * np.prod(1 + avg_by_day / 100.0))

    return {
        "fold": fold,
        "trades": int(len(trades)),
        "rel_wr": round(rel_wr, 2),
        "avg_rel": round(avg_rel, 4),
        "sharpe": round(sharpe, 3),
        "equity": round(equity, 2),
        "days": int(trades["date"].nunique()),
    }, trades

def main():
    print("NSE_SECTOR_ROTATION_V1 ACTIVE: VIX<=20, top-3 sector relative momentum")
    df = build_panel()
    print("rows", len(df), "sectors", df["sector"].nunique(), "dates", df["date"].nunique())
    print("sectors", sorted(df["sector"].unique()))

    results = []
    all_trades = []
    for fold, start, end in FOLDS:
        res, tr = eval_fold(df, fold, start, end)
        results.append(res)
        all_trades.append(tr)
        print(
            f"fold={fold} trades={res['trades']} rel_wr={res['rel_wr']}% "
            f"avg_rel={res['avg_rel']}% sharpe={res['sharpe']} equity={res['equity']}"
        )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({"strategy": "nse_sector_rotation_v1", "results": results}, indent=2))
    pd.concat(all_trades, ignore_index=True).to_csv(TRADES, index=False)
    print("saved", REPORT, TRADES)

if __name__ == "__main__":
    main()
