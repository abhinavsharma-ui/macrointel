#!/usr/bin/env python3
"""
Model-Wired Backtest
====================
Tests the trained stacking ensemble (XGB+LGB+CAT+meta) on crypto 4h data.
Simulates actual trades with fees, ATR-based exits, and P&L tracking.

Usage:
    python3 model_backtest.py
    python3 model_backtest.py --horizon 4 --confidence 0.55 --fee 0.00055
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_DIR   = Path(__file__).parent
CHECKPOINTS   = PROJECT_DIR / "project" / "models" / "checkpoints"
FEATURES_DIR  = PROJECT_DIR / "project" / "data" / "features_10yr"

# ── Fee & risk constants ─────────────────────────────────────────────────────
TAKER_FEE     = 0.00055   # Bybit taker fee per side (0.055%)
ATR_PERIOD    = 14
ATR_STOP_MULT = 1.5       # stop loss = 1.5 × ATR
ATR_TP_MULT   = 2.5       # take profit = 2.5 × ATR
MAX_HOLD_BARS = 10        # max bars to hold if neither SL nor TP hit


def load_models():
    """Load all stacking artifacts from checkpoints."""
    logger.info("Loading stacking models...")
    xgb_blob  = joblib.load(CHECKPOINTS / "stacking_xgb.joblib")
    lgb_blob  = joblib.load(CHECKPOINTS / "stacking_lgb.joblib")
    cat_blob  = joblib.load(CHECKPOINTS / "stacking_cat.joblib")
    meta_blob = joblib.load(CHECKPOINTS / "stacking_meta.joblib")

    xgb = xgb_blob.get("model", xgb_blob) if isinstance(xgb_blob, dict) else xgb_blob
    lgb = lgb_blob.get("model", lgb_blob) if isinstance(lgb_blob, dict) else lgb_blob
    cat = cat_blob.get("model", cat_blob) if isinstance(cat_blob, dict) else cat_blob
    meta = meta_blob.get("model", meta_blob) if isinstance(meta_blob, dict) else meta_blob

    meta_json = json.loads((CHECKPOINTS / "stacking_meta.json").read_text())
    feature_order = (
        meta_json.get("feature_order")
        or meta_json.get("features")
        or (xgb_blob.get("features") if isinstance(xgb_blob, dict) else None)
    )
    if not feature_order:
        raise KeyError("stacking_meta.json missing feature_order/features")
    meta_kind = meta_json.get("meta_kind") or (meta_blob.get("kind") if isinstance(meta_blob, dict) else None) or "rf"

    logger.info(f"Models loaded. Meta: {meta_kind}, Features: {len(feature_order)}")
    return xgb, lgb, cat, meta, feature_order, meta_kind


def predict_proba_take(model, X: pd.DataFrame) -> np.ndarray:
    """Return P(take=1) for base learners."""
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1]
    return model.predict(X).astype(float)


def meta_predict(meta, X_meta: np.ndarray, meta_kind: str):
    """Return (preds, proba) from meta model."""
    preds = meta.predict(X_meta)
    if hasattr(meta, "predict_proba"):
        p = meta.predict_proba(X_meta)
        if p.ndim == 2 and p.shape[1] >= 2:
            return preds, p[:, 1]
    try:
        sig = 1.0 / (1.0 + np.exp(-meta.decision_function(X_meta)))
        return preds, sig
    except Exception:
        return preds, preds.astype(float)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def build_take_labels(df: pd.DataFrame, horizon: int) -> pd.Series:
    close = df["close"]
    fwd_ret    = close.pct_change(horizon).shift(-horizon)
    daily_std  = close.pct_change().std()
    threshold  = (daily_std or 0.01) * 0.5 * (horizon ** 0.5)
    take = (fwd_ret.abs() >= threshold).astype(int)
    take.loc[fwd_ret.isna()] = 0
    return take


def backtest_symbol(
    symbol: str,
    xgb, lgb, cat, meta,
    feature_order: list,
    meta_kind: str,
    horizon: int,
    confidence_threshold: float,
    fee: float,
) -> list:
    """Run backtest for one symbol. Returns list of trade dicts."""
    path = FEATURES_DIR / f"{symbol}.parquet"
    if not path.exists():
        return []

    df = pd.read_parquet(path)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df = df[numeric_cols].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

    if len(df) < 200:
        return []

    # Align features to training order
    missing = [c for c in feature_order if c not in df.columns]
    for c in missing:
        df[c] = 0.0
    X_all = df[feature_order].copy()

    # ATR for exit sizing
    if "high" in df.columns and "low" in df.columns and "close" in df.columns:
        atr = compute_atr(df)
    else:
        atr = pd.Series(df["close"].pct_change().rolling(14).std() * df["close"]
                        if "close" in df.columns else 0.01, index=df.index)

    # Use last 20% as test set (same as training holdout)
    n = len(df)
    test_start = int(n * 0.80)
    X_test = X_all.iloc[test_start:].reset_index(drop=True)
    df_test = df.iloc[test_start:].reset_index(drop=True)
    atr_test = atr.iloc[test_start:].reset_index(drop=True)

    if len(X_test) < horizon + MAX_HOLD_BARS + 1:
        return []

    # Get model predictions
    p_xgb = predict_proba_take(xgb, X_test)
    p_lgb = predict_proba_take(lgb, X_test)
    p_cat = predict_proba_take(cat, X_test)

    completeness = np.isfinite(X_test.values) & (X_test.values != 0.0)
    comp_ratio   = completeness.mean(axis=1)

    X_meta = np.column_stack([p_xgb, p_lgb, p_cat, comp_ratio])
    preds, proba = meta_predict(meta, X_meta, meta_kind)

    trades = []
    close = df_test["close"].values if "close" in df_test.columns else None
    if close is None:
        return []

    for i in range(len(X_test) - MAX_HOLD_BARS - 1):
        if proba[i] < confidence_threshold:
            continue

        entry_price = close[i]
        atr_val     = atr_test.iloc[i] if not np.isnan(atr_test.iloc[i]) else entry_price * 0.01

        # Determine direction from forward return
        fwd_idx = min(i + horizon, len(close) - 1)
        direction = "long" if close[fwd_idx] > entry_price else "short"

        stop_loss   = atr_val * ATR_STOP_MULT
        take_profit = atr_val * ATR_TP_MULT

        # Simulate bar-by-bar exit
        exit_price  = close[min(i + MAX_HOLD_BARS, len(close) - 1)]
        exit_reason = "timeout"

        for j in range(i + 1, min(i + MAX_HOLD_BARS + 1, len(close))):
            price = close[j]
            if direction == "long":
                if price <= entry_price - stop_loss:
                    exit_price  = entry_price - stop_loss
                    exit_reason = "stop_loss"
                    break
                if price >= entry_price + take_profit:
                    exit_price  = entry_price + take_profit
                    exit_reason = "take_profit"
                    break
            else:
                if price >= entry_price + stop_loss:
                    exit_price  = entry_price + stop_loss
                    exit_reason = "stop_loss"
                    break
                if price <= entry_price - take_profit:
                    exit_price  = entry_price - take_profit
                    exit_reason = "take_profit"
                    break

        if direction == "long":
            gross_pnl = (exit_price - entry_price) / entry_price
        else:
            gross_pnl = (entry_price - exit_price) / entry_price

        net_pnl = gross_pnl - 2 * fee  # entry + exit fee

        trades.append({
            "symbol":      symbol,
            "bar":         i,
            "direction":   direction,
            "confidence":  round(float(proba[i]), 4),
            "entry_price": round(float(entry_price), 6),
            "exit_price":  round(float(exit_price), 6),
            "exit_reason": exit_reason,
            "gross_pnl":   round(float(gross_pnl), 6),
            "net_pnl":     round(float(net_pnl), 6),
        })

    return trades


def run_backtest(horizon: int, confidence: float, fee: float, max_symbols: int):
    xgb, lgb, cat, meta, feature_order, meta_kind = load_models()

    symbols = sorted([p.stem for p in FEATURES_DIR.glob("*.parquet")])
    if max_symbols:
        symbols = symbols[:max_symbols]

    logger.info(f"Backtesting {len(symbols)} symbols | horizon={horizon} | confidence>={confidence} | fee={fee*100:.4f}%")

    all_trades = []
    for i, sym in enumerate(symbols):
        trades = backtest_symbol(
            sym, xgb, lgb, cat, meta, feature_order, meta_kind,
            horizon, confidence, fee
        )
        all_trades.extend(trades)
        if (i + 1) % 100 == 0:
            logger.info(f"  {i+1}/{len(symbols)} symbols processed, {len(all_trades)} trades so far")

    if not all_trades:
        logger.error("No trades generated. Check confidence threshold or data.")
        return

    df = pd.DataFrame(all_trades)

    total      = len(df)
    winners    = df[df["net_pnl"] > 0]
    losers     = df[df["net_pnl"] <= 0]
    win_rate   = len(winners) / total * 100
    avg_win    = winners["net_pnl"].mean() * 100 if len(winners) else 0
    avg_loss   = losers["net_pnl"].mean()  * 100 if len(losers)  else 0
    total_pnl  = df["net_pnl"].sum() * 100
    pf         = abs(winners["net_pnl"].sum() / losers["net_pnl"].sum()) if len(losers) else 999

    # Equity curve & drawdown
    equity = np.cumprod(1 + df["net_pnl"].values)
    peak   = np.maximum.accumulate(equity)
    dd     = (peak - equity) / peak * 100
    max_dd = dd.max()

    # Exit reason breakdown
    exit_counts = df["exit_reason"].value_counts().to_dict()

    print("\n" + "=" * 60)
    print("  MODEL BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Horizon          : {horizon} bars ({horizon*4}h)")
    print(f"  Confidence cutoff: {confidence}")
    print(f"  Fee (per side)   : {fee*100:.4f}%")
    print(f"  Symbols tested   : {len(symbols)}")
    print()
    print(f"  Total trades     : {total:,}")
    print(f"  Win rate         : {win_rate:.1f}%")
    print(f"  Avg win          : +{avg_win:.3f}%")
    print(f"  Avg loss         : {avg_loss:.3f}%")
    print(f"  Profit factor    : {pf:.2f}")
    print(f"  Total P&L        : {total_pnl:+.2f}%  (sum across all symbols)")
    print(f"  Max drawdown     : {max_dd:.2f}%")
    print()
    print(f"  Exit reasons     : {exit_counts}")
    print("=" * 60)

    # Save results
    out_path = PROJECT_DIR / "project" / "models" / "checkpoints" / "backtest_results.json"
    result = {
        "horizon": horizon,
        "confidence": confidence,
        "fee": fee,
        "total_trades": total,
        "win_rate": round(win_rate, 2),
        "avg_win_pct": round(avg_win, 4),
        "avg_loss_pct": round(avg_loss, 4),
        "profit_factor": round(pf, 3),
        "total_pnl_pct": round(total_pnl, 4),
        "max_drawdown_pct": round(max_dd, 2),
        "exit_reasons": exit_counts,
    }
    out_path.write_text(json.dumps(result, indent=2))
    logger.info(f"Results saved to {out_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon",    type=int,   default=4)
    parser.add_argument("--confidence", type=float, default=0.55)
    parser.add_argument("--fee",        type=float, default=TAKER_FEE)
    parser.add_argument("--max-symbols",type=int,   default=0, help="0=all")
    args = parser.parse_args()

    run_backtest(args.horizon, args.confidence, args.fee, args.max_symbols)
