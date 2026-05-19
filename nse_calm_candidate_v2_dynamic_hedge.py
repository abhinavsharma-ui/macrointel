
from pathlib import Path
import json
import numpy as np
import pandas as pd

PRICE_DIR = Path("data/prices_full")
NIFTY_PATH = PRICE_DIR / "NIFTY50.parquet"
SRC_TRADES = Path("reports/nse_calm_residual_v1_PARTIAL_trades.csv")
OUT_JSON = Path("reports/nse_calm_candidate_v2_dynamic_hedge.json")
OUT_CSV = Path("reports/nse_calm_candidate_v2_dynamic_hedge_trades.csv")

PROB_FLOORS = [None, 0.54, 0.56]
HEDGE_BETAS = [0.15, 0.25, 0.35]

BLOCK_MEDIAN_REL60 = -3.0
BLOCK_BREADTH60 = 0.45

WARN_MEDIAN_REL60 = -1.0
WARN_BREADTH60 = 0.55

def load_px(path, symbol=None):
    df = pd.read_parquet(path)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    lower = {str(c).lower(): c for c in df.columns}

    if "date" in lower:
        d = df[lower["date"]]
    elif isinstance(df.index, pd.DatetimeIndex):
        d = pd.Series(df.index, index=df.index)
    else:
        df = df.reset_index()
        df = df.loc[:, ~df.columns.duplicated()].copy()
        lower = {str(c).lower(): c for c in df.columns}
        d = df[lower["date"]] if "date" in lower else df.iloc[:, 0]

    if isinstance(d, pd.DataFrame):
        d = d.iloc[:, 0]

    close_col = lower.get("adj_close") or lower.get("close")
    if close_col is None:
        return None

    out = pd.DataFrame({
        "date": pd.to_datetime(d, errors="coerce", utc=True).dt.tz_convert(None).dt.normalize(),
        "close": pd.to_numeric(df[close_col], errors="coerce"),
    }).dropna()

    out = out.reset_index(drop=True).drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    if symbol is not None:
        out["symbol"] = symbol
    return out

def build_daily_narrow_features():
    nifty = load_px(NIFTY_PATH, "NIFTY")
    nifty["nifty_ret60"] = nifty["close"].pct_change(60) * 100.0
    nifty = nifty[["date", "nifty_ret60"]]

    frames = []
    paths = sorted(PRICE_DIR.glob("*.NS.parquet"))
    print(f"loading {len(paths)} NSE stock files...")

    for i, path in enumerate(paths, 1):
        sym = path.stem.replace(".NS", "")
        df = load_px(path, sym)
        if df is None or len(df) < 90:
            continue
        df["stock_ret60"] = df["close"].pct_change(60) * 100.0
        frames.append(df[["date", "symbol", "stock_ret60"]])
        if i % 250 == 0:
            print(f"  loaded {i}/{len(paths)}")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.merge(nifty, on="date", how="left")
    panel["rel60"] = panel["stock_ret60"] - panel["nifty_ret60"]
    panel = panel.dropna(subset=["rel60"])

    daily = panel.groupby("date").agg(
        median_rel60=("rel60", "median"),
        pct_outperform_60=("rel60", lambda s: float((s > 0).mean())),
        n_symbols=("rel60", "count"),
    ).reset_index()

    daily = daily.merge(nifty, on="date", how="left").sort_values("date")

    # No lookahead: trade date uses yesterday's completed market breadth state.
    cols = ["median_rel60", "pct_outperform_60", "n_symbols", "nifty_ret60"]
    daily[cols] = daily[cols].shift(1)

    return daily

def normalize_trade_dates(trades):
    trades = trades.copy()
    trades["date"] = pd.to_datetime(trades["date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    for c in ["prob", "net_ret", "rel_net_ret", "fold"]:
        trades[c] = pd.to_numeric(trades[c], errors="coerce")
    return trades.dropna(subset=["date", "prob", "net_ret", "rel_net_ret", "fold"])

def summarize(sub):
    ret = sub["dyn_rel_ret"].astype(float)
    if len(ret) == 0:
        return {"trades": 0, "rel_win_rate_pct": 0.0, "avg_rel_ret": 0.0, "sharpe": 0.0}
    sd = ret.std(ddof=1)
    return {
        "trades": int(len(ret)),
        "rel_win_rate_pct": float((ret > 0).mean() * 100.0),
        "avg_rel_ret": float(ret.mean()),
        "sharpe": float((ret.mean() / sd) * np.sqrt(len(ret))) if sd and sd > 0 else 0.0,
    }

def main():
    if not SRC_TRADES.exists():
        raise SystemExit(f"Missing {SRC_TRADES}. Run/copy the residual v1 trade CSV first.")

    print("NSE_CALM_CANDIDATE_V2_DYNAMIC_HEDGE")
    daily = build_daily_narrow_features()
    trades = normalize_trade_dates(pd.read_csv(SRC_TRADES))
    trades = trades.merge(daily, on="date", how="left")

    trades["toxic_block"] = (
        (trades["nifty_ret60"] > 0) &
        (trades["median_rel60"] < BLOCK_MEDIAN_REL60) &
        (trades["pct_outperform_60"] < BLOCK_BREADTH60)
    ).fillna(False)

    trades["warning_narrow"] = (
        (trades["nifty_ret60"] > 0) &
        (trades["median_rel60"] < WARN_MEDIAN_REL60) &
        (trades["pct_outperform_60"] < WARN_BREADTH60)
    ).fillna(False)

    trades["nifty_trade_ret"] = trades["net_ret"] - trades["rel_net_ret"]

    all_rows = []
    all_trades = []

    for prob_floor in PROB_FLOORS:
        for hedge_beta in HEDGE_BETAS:
            t = trades[~trades["toxic_block"]].copy()
            if prob_floor is not None:
                t = t[t["prob"] >= prob_floor].copy()

            t["hedge_beta"] = np.where(t["warning_narrow"], hedge_beta, 0.0)
            t["dyn_rel_ret"] = t["net_ret"] - (t["hedge_beta"] * t["nifty_trade_ret"])
            t["prob_floor"] = "NONE" if prob_floor is None else prob_floor
            tag = f"prob={t['prob_floor'].iloc[0] if len(t) else prob_floor}_hedge={hedge_beta}"

            print(f"\n=== {tag} ===")
            for fold in [1, 2, 3, 4]:
                s = summarize(t[t["fold"] == fold])
                row = {"prob_floor": prob_floor, "hedge_beta": hedge_beta, "fold": fold, **s}
                all_rows.append(row)
                print(
                    f"fold={fold} trades={s['trades']} "
                    f"rel_wr={s['rel_win_rate_pct']:.2f} "
                    f"avg_rel={s['avg_rel_ret']:.4f} "
                    f"sharpe={s['sharpe']:.3f}"
                )

            all_trades.append(t)

    res = pd.DataFrame(all_rows)
    print("\n=== CANDIDATES: fold2 > 0 and fold4 > 0 ===")
    piv = res.pivot_table(index=["prob_floor", "hedge_beta"], columns="fold", values="avg_rel_ret")
    good = piv[(piv[2] > 0) & (piv[4] > 0)].sort_values([2, 4], ascending=False)
    print(good.to_string() if len(good) else "NONE")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"model": "nse_calm_candidate_v2_dynamic_hedge", "results": all_rows}, indent=2))
    pd.concat(all_trades, ignore_index=True).to_csv(OUT_CSV, index=False)
    print(f"\nsaved {OUT_JSON} {OUT_CSV}")

if __name__ == "__main__":
    main()
