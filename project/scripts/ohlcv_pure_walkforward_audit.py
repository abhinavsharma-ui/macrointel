from __future__ import annotations

import argparse
import json
import os
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")
warnings.filterwarnings("ignore", category=UserWarning, message="no explicit representation of timezones available.*")

FEATURES = [
    "return_20d",
    "return_60d",
    "momentum_composite",
    "vol_regime_ratio",
    "atr_pct",
    "price_acceleration",
    "close_vs_sma_50",
    "close_vs_sma_200",
    "rsi_14",
]
POLICIES = [(1.5, 2.0), (1.5, 3.0), (2.0, 3.0), (2.0, 4.0), (2.5, 3.0), (2.5, 4.0), (3.0, 4.0), (3.0, 5.0)]
ROOT = Path(os.getenv("PURE_FEATURE_ROOT", "data/features_26yr_liquid"))
CACHE = Path(os.getenv("PURE_AUDIT_CACHE", "reports/ohlcv_pure_audit_dataset.parquet"))
OUT = Path(os.getenv("PURE_AUDIT_OUT", "reports/ohlcv_pure_walkforward_audit.json"))


def log(*x: object) -> None:
    print("PURE_AUDIT", *x, flush=True)


def fgrid(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def igrid(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def policy_col(sl: float, tp: float) -> str:
    return f"gross_ret_sl{str(sl).replace('.', 'p')}_tp{str(tp).replace('.', 'p')}"


def hold_col(sl: float, tp: float) -> str:
    return f"hold_days_sl{str(sl).replace('.', 'p')}_tp{str(tp).replace('.', 'p')}"


def pick_col(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def as_num(x: object, default: float = 0.0) -> pd.Series:
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)
    return pd.Series(default)


def parse_ts(df: pd.DataFrame) -> pd.Series:
    for col in ("timestamp", "date", "datetime", "time", "Date"):
        if col in df.columns:
            ts = pd.to_datetime(df[col], utc=True, errors="coerce")
            if ts.notna().any():
                return pd.Series(ts, index=df.index)
    return pd.Series(pd.to_datetime(df.index, utc=True, errors="coerce"), index=df.index)


def load_one(path: Path, horizon: int) -> Optional[pd.DataFrame]:
    try:
        raw = pd.read_parquet(path)
        if raw is None or raw.empty:
            return None
        close_col = pick_col(raw, ["close", "Close", "adj_close", "Adj Close", "c"])
        if close_col is None:
            return None
        open_col = pick_col(raw, ["open", "Open", "o"]) or close_col
        high_col = pick_col(raw, ["high", "High", "h"]) or close_col
        low_col = pick_col(raw, ["low", "Low", "l"]) or close_col
        volume_col = pick_col(raw, ["volume", "Volume", "v"])
        frame = pd.DataFrame(
            {
                "timestamp": parse_ts(raw),
                "open": as_num(raw[open_col]),
                "high": as_num(raw[high_col]),
                "low": as_num(raw[low_col]),
                "close": as_num(raw[close_col]),
                "volume": as_num(raw[volume_col]) if volume_col else 0.0,
            }
        )
        for f in FEATURES:
            if f in raw.columns:
                frame[f] = as_num(raw[f])
            elif f == "return_20d" and "momentum_20d" in raw.columns:
                frame[f] = as_num(raw["momentum_20d"])
            elif f == "return_60d" and "momentum_60d" in raw.columns:
                frame[f] = as_num(raw["momentum_60d"])
            else:
                frame[f] = 0.0
        frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp").reset_index(drop=True)
        frame = frame[(frame["open"] > 0) & (frame["close"] > 0)]
        if len(frame) <= horizon + 260:
            return None

        o = frame["open"].to_numpy(float)
        h = frame["high"].to_numpy(float)
        l = frame["low"].to_numpy(float)
        c = frame["close"].to_numpy(float)
        ts = pd.to_datetime(frame["timestamp"], utc=True).to_numpy()
        n = len(frame)
        idx = np.arange(n)
        fut_h = np.column_stack([np.r_[h[k:], np.full(k, np.nan)] for k in range(1, horizon + 1)])
        fut_l = np.column_stack([np.r_[l[k:], np.full(k, np.nan)] for k in range(1, horizon + 1)])
        entry_open = np.r_[o[1:], np.nan]
        entry_ts = np.r_[ts[1:], np.array([np.datetime64("NaT")])]
        exit_close = np.r_[c[horizon:], np.full(horizon, np.nan)]

        out = pd.DataFrame(
            {
                "symbol": path.stem,
                "timestamp": pd.to_datetime(ts, utc=True),
                "entry_timestamp": pd.to_datetime(entry_ts, utc=True),
                "close": c,
                "entry_open": entry_open,
                "volume": frame["volume"].to_numpy(float),
                "adv20_dollar_vol": (frame["close"] * frame["volume"]).rolling(20, min_periods=5).mean().to_numpy(float),
            }
        )
        for f in FEATURES:
            out[f] = frame[f].to_numpy(float)
        out["edge_pct"] = (np.nanmax(fut_h, axis=1) / entry_open - 1.0) * 100.0
        out["drawdown_pct"] = np.maximum(0.0, (1.0 - np.nanmin(fut_l, axis=1) / entry_open) * 100.0)

        for sl, tp in POLICIES:
            ret_name = policy_col(sl, tp)
            h_name = hold_col(sl, tp)
            stop = entry_open * (1.0 - sl / 100.0)
            target = entry_open * (1.0 + tp / 100.0)
            stop_hit = fut_l <= stop[:, None]
            target_hit = fut_h >= target[:, None]
            stop_any = stop_hit.any(axis=1)
            target_any = target_hit.any(axis=1)
            stop_first = np.where(stop_any, stop_hit.argmax(axis=1), 999)
            target_first = np.where(target_any, target_hit.argmax(axis=1), 999)
            gross = (exit_close / entry_open - 1.0) * 100.0
            hold = np.full(n, horizon, dtype=np.int16)
            stop_wins = stop_any & (stop_first <= target_first)
            target_wins = target_any & (target_first < stop_first)
            gross = np.where(target_wins, tp, gross)
            gross = np.where(stop_wins, -sl, gross)
            hold = np.where(target_wins, target_first + 1, hold)
            hold = np.where(stop_wins, stop_first + 1, hold)
            exit_idx = idx + hold
            ok = exit_idx < n
            exit_ts = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
            exit_ts[ok] = ts[exit_idx[ok]]
            out[ret_name] = gross
            out[h_name] = hold
            out[f"exit_timestamp_{ret_name}"] = pd.to_datetime(exit_ts, utc=True)

        ret_cols = [policy_col(sl, tp) for sl, tp in POLICIES]
        req = ["timestamp", "entry_timestamp", "entry_open", "edge_pct", "drawdown_pct", "adv20_dollar_vol"] + FEATURES + ret_cols
        out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=req)
        return out[out["entry_timestamp"].notna()]
    except Exception as exc:
        log("load_error", path.name, repr(exc))
        return None


def build_or_load(args: argparse.Namespace) -> pd.DataFrame:
    if CACHE.exists() and CACHE.stat().st_size > 50_000_000 and not args.rebuild_cache:
        log("loading_cache", CACHE, CACHE.stat().st_size)
        return pd.read_parquet(CACHE)
    files = sorted(ROOT.glob("*.parquet"))
    if args.max_symbols:
        files = files[: args.max_symbols]
    if not files:
        raise SystemExit(f"No parquet files under {ROOT}")
    parts: List[pd.DataFrame] = []
    log("building_cache", ROOT, "files", len(files), "horizon", args.horizon)
    for i, path in enumerate(files, 1):
        part = load_one(path, args.horizon)
        if part is not None and not part.empty:
            parts.append(part)
        if i % 50 == 0 or i == len(files):
            log("build_progress", i, "/", len(files), "ok", len(parts), "rows", sum(len(x) for x in parts))
    if not parts:
        raise SystemExit("No usable data")
    df = pd.concat(parts, ignore_index=True)
    if not args.max_symbols:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(CACHE, index=False)
        log("wrote_cache", CACHE, CACHE.stat().st_size, "rows", len(df), "symbols", df["symbol"].nunique())
    return df


def prep(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["entry_timestamp"] = pd.to_datetime(out["entry_timestamp"], utc=True, errors="coerce")
    for col in FEATURES + ["edge_pct", "drawdown_pct", "adv20_dollar_vol"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["timestamp", "entry_timestamp", "edge_pct", "drawdown_pct", "adv20_dollar_vol"] + FEATURES)
    before = len(out)
    out = out[out["adv20_dollar_vol"] >= args.min_adv20_dollar_vol].copy()
    draw = out["drawdown_pct"].clip(lower=0.35)
    out["take_label"] = ((out["edge_pct"] >= args.label_edge_pct) & (out["drawdown_pct"] <= args.label_max_drawdown_pct) & ((out["edge_pct"] / draw) >= args.label_min_ratio)).astype(int)
    log("liquidity_filter", "before", before, "after", len(out), "symbols", out["symbol"].nunique(), "positives", int(out["take_label"].sum()))
    out = apply_regime_filter(out, args)
    return out.sort_values("timestamp").reset_index(drop=True)


def apply_regime_filter(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    out["regime_allowed"] = True
    if getattr(args, "regime_filter", "none") != "spy_uptrend":
        return out
    spy = out[out["symbol"].astype(str).eq("SPY")].copy().sort_values("timestamp")
    if spy.empty:
        out["regime_allowed"] = False
        log("regime_filter", "spy_uptrend", "SPY missing; all rows blocked")
        return out
    close = pd.to_numeric(spy["close"], errors="coerce")
    spy["spy_sma200"] = close.rolling(200, min_periods=160).mean()
    spy["spy_return_20d"] = close.pct_change(20)
    spy["regime_allowed"] = (close > spy["spy_sma200"]) & (spy["spy_return_20d"] > 0)
    allowed = spy.set_index("timestamp")["regime_allowed"].to_dict()
    out["regime_allowed"] = out["timestamp"].map(allowed).fillna(False).astype(bool)
    log("regime_filter", "spy_uptrend", "allowed_rows", int(out["regime_allowed"].sum()), "total_rows", len(out))
    return out


def fit_model(train: pd.DataFrame, seed: int, shuffle_labels: bool) -> Optional[HistGradientBoostingClassifier]:
    y = train["take_label"].astype(int).to_numpy(copy=True)
    if shuffle_labels:
        rng = np.random.default_rng(seed)
        rng.shuffle(y)
    if len(np.unique(y)) < 2:
        return None
    model = HistGradientBoostingClassifier(max_iter=220, learning_rate=0.045, max_leaf_nodes=31, l2_regularization=0.05, random_state=seed)
    model.fit(train[FEATURES].fillna(0.0), y)
    return model


def score(model: HistGradientBoostingClassifier, frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["probability"] = model.predict_proba(out[FEATURES].fillna(0.0))[:, 1]
    out = out.sort_values(["timestamp", "probability"], ascending=[True, False]).reset_index(drop=True)
    out["prob_rank"] = out.groupby("timestamp", sort=False).cumcount().astype(int) + 1
    return out


def select_trades(scored: pd.DataFrame, threshold: float, max_new: int, ret_col: str, args: argparse.Namespace) -> pd.DataFrame:
    exit_col = f"exit_timestamp_{ret_col}"
    if scored.empty or ret_col not in scored.columns or exit_col not in scored.columns:
        return pd.DataFrame()
    keep = ["symbol", "timestamp", "entry_timestamp", "probability", "prob_rank", ret_col, exit_col]
    out = scored[keep].copy()
    if getattr(args, "regime_filter", "none") != "none" and "regime_allowed" in scored.columns:

        if "regime_allowed" in out.columns:
            out = out[out["regime_allowed"].fillna(False).to_numpy(dtype=bool)]

    if getattr(args, "regime_filter", "none") != "none" and "regime_allowed" in scored.columns:

        if "regime_allowed" in out.columns:
            out = out[out["regime_allowed"].fillna(False).to_numpy(dtype=bool)]

    out = out[(out["probability"] >= threshold) & (out["prob_rank"] <= max_new)]
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=[ret_col, "entry_timestamp", exit_col])
    if out.empty:
        return out
    out = out.rename(columns={ret_col: "gross_return_pct", exit_col: "exit_timestamp"})
    out["net_return_pct"] = pd.to_numeric(out["gross_return_pct"], errors="coerce") - args.roundtrip_cost_pct - args.slippage_pct
    return out.dropna(subset=["net_return_pct"])


def simulate(trades: pd.DataFrame, args: argparse.Namespace) -> Dict[str, object]:
    if trades.empty:
        return {"trades": 0, "executed_trades": 0, "skipped_capacity": 0, "final_equity": 100000.0, "return_pct": 0.0, "max_drawdown_pct": 0.0, "avg_trade_net_return_pct": 0.0, "win_rate_pct": 0.0}
    work = trades.copy()
    work["entry_timestamp"] = pd.to_datetime(work["entry_timestamp"], utc=True, errors="coerce")
    work["exit_timestamp"] = pd.to_datetime(work["exit_timestamp"], utc=True, errors="coerce")
    work = work.dropna(subset=["entry_timestamp", "exit_timestamp", "net_return_pct"]).sort_values(["entry_timestamp", "probability"], ascending=[True, False])
    cash = 100000.0
    open_pos: List[Tuple[pd.Timestamp, float, float]] = []
    peak = 100000.0
    max_dd = 0.0
    executed: List[float] = []
    skipped = 0
    groups = {k: v for k, v in work.groupby("entry_timestamp", sort=False)}
    dates = sorted(set(work["entry_timestamp"].tolist()) | set(work["exit_timestamp"].tolist()))
    for date in dates:
        if open_pos:
            rem = []
            for exit_date, notional, ret in open_pos:
                if exit_date <= date:
                    cash += notional * (1.0 + ret / 100.0)
                else:
                    rem.append((exit_date, notional, ret))
            open_pos = rem
        if date in groups:
            for _, row in groups[date].iterrows():
                equity = cash + sum(p[1] for p in open_pos)
                notional = equity * args.position_pct
                open_notional = sum(p[1] for p in open_pos)
                if cash < notional or open_notional + notional > equity * args.max_gross_exposure_pct:
                    skipped += 1
                    continue
                cash -= notional
                ret = float(row["net_return_pct"])
                open_pos.append((row["exit_timestamp"], notional, ret))
                executed.append(ret)
        equity = cash + sum(p[1] for p in open_pos)
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak * 100.0)
    for exit_date, notional, ret in sorted(open_pos, key=lambda x: x[0]):
        cash += notional * (1.0 + ret / 100.0)
        peak = max(peak, cash)
        max_dd = max(max_dd, (peak - cash) / peak * 100.0)
    arr = np.asarray(executed, dtype=float)
    return {
        "trades": int(len(work)),
        "executed_trades": int(len(arr)),
        "skipped_capacity": int(skipped),
        "final_equity": round(float(cash), 2),
        "return_pct": round(float((cash / 100000.0 - 1.0) * 100.0), 4),
        "max_drawdown_pct": round(float(max_dd), 4),
        "avg_trade_net_return_pct": round(float(arr.mean()) if len(arr) else 0.0, 4),
        "win_rate_pct": round(float((arr > 0).mean() * 100.0) if len(arr) else 0.0, 2),
    }


def objective(metrics: Dict[str, object], args: argparse.Namespace) -> float:
    if int(metrics["executed_trades"]) < args.min_val_trades:
        return -1e12
    if float(metrics["return_pct"]) <= 0 or float(metrics["avg_trade_net_return_pct"]) <= 0:
        return -1e12
    if float(metrics["win_rate_pct"]) < 50 or float(metrics["max_drawdown_pct"]) > args.max_drawdown_pct:
        return -1e12
    return float(metrics["return_pct"]) * 4 + float(metrics["avg_trade_net_return_pct"]) * 30 + float(metrics["win_rate_pct"]) * 0.1 - float(metrics["max_drawdown_pct"]) * 3


def bounds(frame: pd.DataFrame) -> Dict[str, object]:
    if frame.empty:
        return {"rows": 0, "start": None, "end": None}
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    return {"rows": int(len(frame)), "start": str(ts.min().date()), "end": str(ts.max().date())}


def run_strategy(df: pd.DataFrame, args: argparse.Namespace, name: str, shuffle_labels: bool = False, odd_years_only: bool = False, symbols: Optional[Sequence[str]] = None) -> Tuple[Dict[str, object], pd.DataFrame]:
    work = df[df["symbol"].isin(symbols)].copy() if symbols is not None else df.copy()
    dates = np.array(sorted(work["timestamp"].dropna().unique()))
    span = max(80, len(dates) // (args.folds + 2))
    thresholds = fgrid(args.thresholds)
    max_grid = igrid(args.max_new_grid)
    ret_cols = [policy_col(sl, tp) for sl, tp in POLICIES]
    fold_rows = []
    all_trades = []
    for fold in range(1, args.folds + 1):
        test_end = len(dates) - (args.folds - fold) * span
        test_start = test_end - span
        train_end = test_start
        if train_end < span:
            fold_rows.append({"fold": fold, "skipped": True, "reason": "too_little_history"})
            continue
        train_dates = dates[:train_end]
        val_start = max(0, int(len(train_dates) * 0.82))
        train = work[work["timestamp"].isin(train_dates[:val_start])]
        val = work[work["timestamp"].isin(train_dates[val_start:])]
        train_val = work[work["timestamp"].isin(train_dates)]
        test = work[work["timestamp"].isin(dates[test_start:test_end])]
        if odd_years_only:
            test = test[pd.to_datetime(test["timestamp"], utc=True).dt.year % 2 == 1]
        model = fit_model(train, 1000 + fold, shuffle_labels)
        if model is None:
            fold_rows.append({"fold": fold, "skipped": True, "reason": "single_class_train"})
            continue
        val_scored = score(model, val)
        best = None
        for ret_col in ret_cols:
            for th in thresholds:
                for mx in max_grid:
                    trades = select_trades(val_scored, th, mx, ret_col, args)
                    metrics = simulate(trades, args)
                    obj = objective(metrics, args)
                    if obj <= -1e11:
                        continue
                    cand = {"ret_col": ret_col, "threshold": th, "max_new_per_day": mx, "validation": metrics, "objective": round(float(obj), 4)}
                    if best is None or cand["objective"] > best["objective"]:
                        best = cand
        if best is None:
            fold_rows.append({"fold": fold, "skipped": True, "reason": "no_profitable_validation_policy", "train": bounds(train), "val": bounds(val), "test": bounds(test)})
            continue
        final = fit_model(train_val, 2000 + fold, shuffle_labels)
        if final is None:
            fold_rows.append({"fold": fold, "skipped": True, "reason": "single_class_train_val"})
            continue
        test_scored = score(final, test)
        trades = select_trades(test_scored, float(best["threshold"]), int(best["max_new_per_day"]), str(best["ret_col"]), args)
        metrics = simulate(trades, args)
        if not trades.empty:
            trades = trades.copy()
            trades["fold"] = fold
            all_trades.append(trades)
        rec = {"fold": fold, "train": bounds(train), "val": bounds(val), "test_window": bounds(test), "policy": best, "test": metrics}
        fold_rows.append(rec)
        log(name, "fold_done", json.dumps({"fold": fold, "policy": best, "test": metrics}, default=str))
    trades_all = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    summary = simulate(trades_all, args)
    summary["valid_folds"] = sum(1 for f in fold_rows if f.get("test") and not f.get("skipped"))
    summary["folds"] = args.folds
    report = {"name": name, "summary": summary, "folds": fold_rows, "trades_tail": trades_all.tail(50).to_dict("records") if not trades_all.empty else []}
    return report, trades_all


def spy_return(df: pd.DataFrame, trades: pd.DataFrame) -> Optional[float]:
    if trades.empty:
        return None
    spy = df[df["symbol"].astype(str).eq("SPY")].copy()
    if spy.empty:
        return None
    start = pd.to_datetime(trades["entry_timestamp"], utc=True).min()
    end = pd.to_datetime(trades["exit_timestamp"], utc=True).max()
    spy = spy[(pd.to_datetime(spy["timestamp"], utc=True) >= start) & (pd.to_datetime(spy["timestamp"], utc=True) <= end)].sort_values("timestamp")
    if len(spy) < 2:
        return None
    return round((float(spy["close"].iloc[-1]) / float(spy["close"].iloc[0]) - 1.0) * 100.0, 4)


def remove_best_test(df: pd.DataFrame, trades: pd.DataFrame, args: argparse.Namespace) -> Dict[str, object]:
    if trades.empty:
        return {"removed_best_trades": args.remove_best_n, **simulate(trades, args)}
    reduced = trades.copy()
    reduced["net_return_pct"] = pd.to_numeric(reduced["net_return_pct"], errors="coerce")
    reduced = reduced.sort_values("net_return_pct", ascending=False).iloc[args.remove_best_n :].copy()
    metrics = simulate(reduced, args)
    bench = spy_return(df, reduced)
    if bench is not None:
        metrics["spy_buy_hold_return_pct_same_window"] = bench
        metrics["beats_spy_same_window"] = bool(float(metrics["return_pct"]) > bench)
    metrics["removed_best_trades"] = args.remove_best_n
    return metrics


def dataset_report(df: pd.DataFrame, raw: pd.DataFrame, args: argparse.Namespace) -> Dict[str, object]:
    ts = pd.to_datetime(df["timestamp"], utc=True)
    adv = pd.to_numeric(raw["adv20_dollar_vol"], errors="coerce")
    return {
        "rows_after_liquidity": int(len(df)),
        "symbols_after_liquidity": int(df["symbol"].nunique()),
        "date_min": str(ts.min().date()),
        "date_max": str(ts.max().date()),
        "years": sorted(ts.dt.year.dropna().astype(int).unique().tolist()),
        "raw_symbols": int(raw["symbol"].nunique()),
        "adv20_quantiles_raw": {str(k): round(float(v), 4) for k, v in adv.quantile([0, 0.01, 0.05, 0.1, 0.5, 0.9, 0.99, 1]).to_dict().items()},
        "survivorship_warning": "Current parquet universe cannot prove delisted or bankrupt names are present. This audit removes lookahead and adds liquidity filters, but point-in-time delisted data is still required to fully remove survivorship bias.",
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--max-symbols", type=int, default=int(os.getenv("PURE_MAX_SYMBOLS", "0") or 0))
    ap.add_argument("--horizon", type=int, default=int(os.getenv("PURE_HORIZON_DAYS", "10")))
    ap.add_argument("--folds", type=int, default=int(os.getenv("PURE_FOLDS", "4")))
    ap.add_argument("--thresholds", default=os.getenv("PURE_THRESHOLDS", "0.55,0.58,0.61,0.64,0.65,0.66,0.67,0.70,0.73"))
    ap.add_argument("--max-new-grid", default=os.getenv("PURE_MAX_NEW_GRID", "1,2,3,5,8,12,20,30"))
    ap.add_argument("--position-pct", type=float, default=float(os.getenv("PURE_POSITION_PCT", "0.02")))
    ap.add_argument("--max-gross-exposure-pct", type=float, default=float(os.getenv("PURE_MAX_GROSS_EXPOSURE_PCT", "1.0")))
    ap.add_argument("--roundtrip-cost-pct", type=float, default=float(os.getenv("PURE_ROUNDTRIP_COST_PCT", "0.35")))
    ap.add_argument("--slippage-pct", type=float, default=float(os.getenv("PURE_SLIPPAGE_PCT", "0.10")))
    ap.add_argument("--min-adv20-dollar-vol", type=float, default=float(os.getenv("PURE_MIN_ADV20_DOLLAR_VOL", "5000000")))
    ap.add_argument("--label-edge-pct", type=float, default=float(os.getenv("PURE_LABEL_EDGE_PCT", "0.90")))
    ap.add_argument("--label-max-drawdown-pct", type=float, default=float(os.getenv("PURE_LABEL_MAX_DRAWDOWN_PCT", "2.50")))
    ap.add_argument("--label-min-ratio", type=float, default=float(os.getenv("PURE_LABEL_MIN_RATIO", "0.30")))
    ap.add_argument("--min-val-trades", type=int, default=int(os.getenv("PURE_MIN_VAL_TRADES", "120")))
    ap.add_argument("--max-drawdown-pct", type=float, default=float(os.getenv("PURE_MAX_DRAWDOWN_PCT", "12.0")))
    ap.add_argument("--run-stress-tests", action="store_true", default=os.getenv("PURE_RUN_STRESS_TESTS", "1") != "0")
    ap.add_argument("--remove-best-n", type=int, default=int(os.getenv("PURE_REMOVE_BEST_N", "50")))
    ap.add_argument("--random-universe-size", type=int, default=int(os.getenv("PURE_RANDOM_UNIVERSE_SIZE", "180")))
    ap.add_argument("--random-seed", type=int, default=int(os.getenv("PURE_RANDOM_SEED", "20260502")))
    ap.add_argument("--regime-filter", default=os.getenv("PURE_REGIME_FILTER", "none"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    raw = build_or_load(args)
    df = prep(raw, args)
    if args.max_symbols and not args.rebuild_cache:
        keep = sorted(df["symbol"].unique())[: args.max_symbols]
        df = df[df["symbol"].isin(keep)].copy()
    baseline, baseline_trades = run_strategy(df, args, "baseline_fold_local_next_open")
    tests: Dict[str, object] = {}
    if args.run_stress_tests:
        tests["shuffle_labels"], _ = run_strategy(df, args, "shuffle_labels", shuffle_labels=True)
        tests["odd_years_only"], _ = run_strategy(df, args, "odd_years_only", odd_years_only=True)
        tests["remove_best_50"] = remove_best_test(df, baseline_trades, args)
        syms = sorted(df["symbol"].dropna().astype(str).unique().tolist())
        rng = random.Random(args.random_seed)
        sample = sorted(rng.sample(syms, min(args.random_universe_size, len(syms))))
        random_result, _ = run_strategy(df, args, "random_universe", symbols=sample)
        tests["random_universe"] = {"symbol_count": len(sample), "symbols_head": sample[:50], "result": random_result}
    bench = spy_return(df, baseline_trades)
    if bench is not None:
        baseline["summary"]["spy_buy_hold_return_pct_same_window"] = bench
        baseline["summary"]["beats_spy_same_window"] = bool(float(baseline["summary"]["return_pct"]) > bench)
    result = {
        "integrity_contract": {
            "fold_local_selector_models": True,
            "full_history_selector_artifact_used": False,
            "entry_price": "next_bar_open",
            "regime_filter": args.regime_filter,
            "same_day_stop_and_target": "stop_loss_first",
            "costs": {"roundtrip_cost_pct": args.roundtrip_cost_pct, "slippage_pct": args.slippage_pct},
            "liquidity_filter_min_adv20_dollar_vol": args.min_adv20_dollar_vol,
            "position_model": {"position_pct_current_equity": args.position_pct, "max_gross_exposure_pct": args.max_gross_exposure_pct, "capital_capacity_skips_enabled": True},
            "grid": {"thresholds": args.thresholds, "max_new_grid": args.max_new_grid, "policies": [policy_col(sl, tp) for sl, tp in POLICIES]},
        },
        "dataset": dataset_report(df, raw, args),
        "baseline": baseline,
        "falsification_tests": tests,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, default=str) + "\n")
    log("wrote", OUT)
    print(json.dumps({
        "baseline": baseline["summary"],
        "shuffle_labels": tests.get("shuffle_labels", {}).get("summary") if tests else None,
        "odd_years_only": tests.get("odd_years_only", {}).get("summary") if tests else None,
        "remove_best_50": tests.get("remove_best_50") if tests else None,
        "random_universe": tests.get("random_universe", {}).get("result", {}).get("summary") if tests else None,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
