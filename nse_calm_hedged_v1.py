from pathlib import Path
import json
import pandas as pd
import numpy as np
import math

TRADES_PATH = Path("reports/nse_calm_residual_v1_PARTIAL_trades.csv")
OUT_JSON = Path("reports/nse_calm_hedged_v1.json")
OUT_CSV = Path("reports/nse_calm_hedged_v1_trades.csv")

FOLDS = [1, 2, 3, 4]
HEDGE_BETAS = [0.25, 0.50, 0.75, 1.00, 1.25]

def find_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def eval_beta(df, beta):
    d = df.copy()
    rel_col = find_col(d, ["rel_ret", "rel_net_ret", "avg_rel", "relative_return"])
    nifty_col = find_col(d, ["nifty_ret", "fwd_nifty_ret", "benchmark_ret", "model_ret"])
    stock_col = find_col(d, ["stock_ret", "fwd_stock_ret", "return"])

    if rel_col is None:
        raise SystemExit(f"Cannot find rel_ret column. columns={list(d.columns)}")

    # If we have only rel_ret, beta=1 is exactly current relative-return metric.
    # For other betas, need nifty return. If absent, skip non-1 betas.
    if beta != 1.0 and nifty_col is None:
        return None, None

    if beta == 1.0:
        d["hedged_rel_ret"] = pd.to_numeric(d[rel_col], errors="coerce")
    elif stock_col is not None:
        d["hedged_rel_ret"] = pd.to_numeric(d[stock_col], errors="coerce") - beta * pd.to_numeric(d[nifty_col], errors="coerce")
    else:
        # rel_ret = stock - nifty - cost-ish, approximate beta adjustment.
        d["hedged_rel_ret"] = pd.to_numeric(d[rel_col], errors="coerce") + (1.0 - beta) * pd.to_numeric(d[nifty_col], errors="coerce")

    rows = []
    for fold, g in d.groupby("fold"):
        g = g.dropna(subset=["hedged_rel_ret"])
        if g.empty:
            continue
        if "date" in g.columns:
            by_day = g.groupby("date")["hedged_rel_ret"].mean()
        else:
            by_day = g["hedged_rel_ret"]
        rows.append({
            "beta": beta,
            "fold": int(fold),
            "trades": int(len(g)),
            "rel_wr": round(float((g["hedged_rel_ret"] > 0).mean() * 100), 2),
            "avg_rel": round(float(g["hedged_rel_ret"].mean()), 4),
            "sharpe": round(float(by_day.mean() / max(by_day.std(ddof=1), 1e-9) * math.sqrt(252/20)), 3) if len(by_day) > 1 else 0,
            "equity": round(float(100000 * np.prod(1 + by_day / 100)), 2),
        })
    return rows, d

def main():
    if not TRADES_PATH.exists():
        raise SystemExit(f"Missing {TRADES_PATH}. Archive residual v1 trades first.")
    df = pd.read_csv(TRADES_PATH)
    if "fold" not in df.columns:
        raise SystemExit(f"Missing fold column. columns={list(df.columns)}")

    all_rows = []
    saved = None
    for beta in HEDGE_BETAS:
        rows, d = eval_beta(df, beta)
        if rows is None:
            continue
        all_rows.extend(rows)
        if beta == 1.0:
            saved = d

    print("NSE_CALM_HEDGED_V1 from residual trades")
    for r in all_rows:
        print(f"beta={r['beta']} fold={r['fold']} trades={r['trades']} rel_wr={r['rel_wr']} avg_rel={r['avg_rel']} sharpe={r['sharpe']} equity={r['equity']}")

    OUT_JSON.write_text(json.dumps({"strategy":"nse_calm_hedged_v1", "results": all_rows}, indent=2))
    if saved is not None:
        saved.to_csv(OUT_CSV, index=False)
    print("saved", OUT_JSON, OUT_CSV)

if __name__ == "__main__":
    main()
