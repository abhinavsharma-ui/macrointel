from pathlib import Path
import json, math
import pandas as pd
import numpy as np

TRADES_PATH = Path("reports/nse_calm_residual_v1_PARTIAL_trades.csv")
PRICE_DIR = Path("data/prices_full")
NIFTY_PATH = PRICE_DIR / "NIFTY50.parquet"
OUT_JSON = Path("reports/nse_calm_hedged_narrow_gate_v1.json")
OUT_CSV = Path("reports/nse_calm_hedged_narrow_gate_v1_trades.csv")

HEDGE_BETA = 0.25
THRESHOLDS = [-5.0, -3.0, -2.0, -1.0, 0.0]

def load_px(path, name):
    df = pd.read_parquet(path).copy()

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

def build_narrow_features():
    nifty = load_px(NIFTY_PATH, "nifty").sort_values("date")
    nifty["nifty_ret60"] = nifty["nifty"].pct_change(60) * 100.0

    rows = []
    files = list(PRICE_DIR.glob("*.NS.parquet"))
    for p in files:
        try:
            df = load_px(p, "px").sort_values("date")
            df["symbol"] = p.stem.replace(".NS", "")
            df["ret60"] = df["px"].pct_change(60) * 100.0
            rows.append(df[["date", "symbol", "ret60"]])
        except Exception:
            pass

    panel = pd.concat(rows, ignore_index=True)
    panel = panel.merge(nifty[["date", "nifty_ret60"]], on="date", how="left")
    panel["rel60"] = panel["ret60"] - panel["nifty_ret60"]

    daily = panel.groupby("date").agg(
        median_rel60=("rel60", "median"),
        mean_rel60=("rel60", "mean"),
        pct_outperform_60=("rel60", lambda s: float((s.dropna() > 0).mean()) if len(s.dropna()) else np.nan),
        active=("rel60", "count"),
    ).reset_index()

    # Shift one day: close-t breadth determines next executable day.
    for c in ["median_rel60", "mean_rel60", "pct_outperform_60", "active"]:
        daily[c] = daily[c].shift(1)
    return daily

def eval_threshold(trades, daily, threshold):
    d = trades.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None).dt.normalize()
    d = d.merge(daily, on="date", how="left")
    d = d[d["median_rel60"] > threshold].copy()

    rel_col = "rel_net_ret" if "rel_net_ret" in d.columns else "rel_ret"
    nifty_col = "model_ret" if "model_ret" in d.columns else None

    d["hedged_rel_ret"] = pd.to_numeric(d[rel_col], errors="coerce")
    if nifty_col:
        d["hedged_rel_ret"] = d["hedged_rel_ret"] + (1.0 - HEDGE_BETA) * pd.to_numeric(d[nifty_col], errors="coerce")

    results = []
    for fold, g in d.groupby("fold"):
        g = g.dropna(subset=["hedged_rel_ret"])
        if g.empty:
            continue
        by_day = g.groupby("date")["hedged_rel_ret"].mean()
        results.append({
            "threshold": threshold,
            "fold": int(fold),
            "trades": int(len(g)),
            "rel_wr": round(float((g["hedged_rel_ret"] > 0).mean() * 100), 2),
            "avg_rel": round(float(g["hedged_rel_ret"].mean()), 4),
            "sharpe": round(float(by_day.mean() / max(by_day.std(ddof=1), 1e-9) * math.sqrt(252/20)), 3) if len(by_day) > 1 else 0,
            "equity": round(float(100000 * np.prod(1 + by_day / 100)), 2),
            "gate_days": int(g["date"].nunique()),
        })
    return results, d

def main():
    print("NSE_CALM_HEDGED_NARROW_GATE_V1 beta=0.25 gate=median_rel60 > threshold")
    trades = pd.read_csv(TRADES_PATH)
    daily = build_narrow_features()
    all_results = []
    best_saved = None

    for th in THRESHOLDS:
        rows, d = eval_threshold(trades, daily, th)
        all_results.extend(rows)
        print("\nTHRESHOLD", th)
        for r in rows:
            print(f"fold={r['fold']} trades={r['trades']} rel_wr={r['rel_wr']} avg_rel={r['avg_rel']} sharpe={r['sharpe']} equity={r['equity']}")
        if th == -2.0:
            best_saved = d

    OUT_JSON.write_text(json.dumps({"strategy": "nse_calm_hedged_narrow_gate_v1", "results": all_results}, indent=2))
    if best_saved is not None:
        best_saved.to_csv(OUT_CSV, index=False)
    print("saved", OUT_JSON, OUT_CSV)

if __name__ == "__main__":
    main()
