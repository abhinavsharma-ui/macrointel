from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, time as dt_time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
NY = ZoneInfo("America/New_York")
MARKET_OPEN = dt_time(9, 30)
MARKET_CLOSE = dt_time(16, 0)
DEFAULT_ROOT = Path(os.getenv("SIG_LIVE_ROOT", "data/features"))
REPORT_PATH = Path("reports/fixed_return_intraday_universe_refresh.json")


def num(value, default=0.0) -> float:
    try:
        if value is None or value == "":
            return default
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def norm_symbol(value: str) -> str:
    return Path(str(value)).stem.replace("_US", "").replace(".US", "").upper()


def is_us_symbol(symbol: str) -> bool:
    sym = norm_symbol(symbol)
    return bool(sym) and not sym.endswith((".NS", ".BO", ".NSE", ".BSE"))


def now_ny() -> datetime:
    return datetime.now(NY)


def is_market_time(ts: datetime) -> bool:
    return ts.weekday() < 5 and MARKET_OPEN <= ts.time() <= MARKET_CLOSE


def age_seconds(ts) -> float | None:
    try:
        now = datetime.now(ts.tzinfo) if getattr(ts, "tzinfo", None) else datetime.utcnow()
        return max(0.0, (now - ts).total_seconds())
    except Exception:
        return None


def load_symbols(root: Path, symbols_arg: str | None, limit: int) -> list[str]:
    if symbols_arg:
        path = Path(symbols_arg)
        if path.exists():
            raw = [line.strip() for line in path.read_text().splitlines()]
        else:
            raw = [part.strip() for part in symbols_arg.split(",")]
        symbols = [norm_symbol(s) for s in raw if is_us_symbol(s)]
    else:
        symbols = [norm_symbol(path.name) for path in sorted(root.glob("*.parquet")) if is_us_symbol(path.name)]
    unique = sorted(dict.fromkeys(symbols))
    return unique[:limit] if limit and limit > 0 else unique


def chunked(items: list[str], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def fetch_live_prices(symbols: list[str], batch_size: int) -> dict[str, dict]:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestBarRequest, StockLatestQuoteRequest, StockLatestTradeRequest

    client = StockHistoricalDataClient(
        api_key=os.getenv("ALPACA_API_KEY", ""),
        secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
    )
    quote_after_seconds = num(os.getenv("ALPACA_QUOTE_MID_AFTER_SECONDS"), 120.0)
    max_quote_spread_pct = num(os.getenv("ALPACA_QUOTE_MID_MAX_SPREAD_PCT"), 2.5)
    out: dict[str, dict] = {}

    for batch in chunked(symbols, max(1, batch_size)):
        trades, quotes, bars = {}, {}, {}
        errors = []
        try:
            trades = client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=batch))
        except Exception as exc:
            errors.append(f"trade:{exc}")
        try:
            quotes = client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=batch))
        except Exception as exc:
            errors.append(f"quote:{exc}")
        if not trades:
            try:
                bars = client.get_stock_latest_bar(StockLatestBarRequest(symbol_or_symbols=batch))
            except Exception as exc:
                errors.append(f"bar:{exc}")

        for sym in batch:
            price, timestamp, source, stale, spread = None, "", "", None, None
            trade = trades.get(sym) if hasattr(trades, "get") else None
            if trade is not None and getattr(trade, "price", None) is not None:
                price = float(trade.price)
                timestamp = str(getattr(trade, "timestamp", ""))
                source = "trade"
                stale = age_seconds(getattr(trade, "timestamp", None))

            quote = quotes.get(sym) if hasattr(quotes, "get") else None
            if quote is not None:
                bid = num(getattr(quote, "bid_price", None))
                ask = num(getattr(quote, "ask_price", None))
                if bid > 0 and ask > 0 and ask >= bid:
                    mid = (bid + ask) / 2.0
                    spread = (ask - bid) / mid * 100.0 if mid else None
                    use_quote = price is None or (
                        stale is not None
                        and stale >= quote_after_seconds
                        and spread is not None
                        and spread <= max_quote_spread_pct
                    )
                    if use_quote:
                        price = mid
                        timestamp = str(getattr(quote, "timestamp", ""))
                        source = "quote_mid"
                        stale = age_seconds(getattr(quote, "timestamp", None))

            if price is None:
                bar = bars.get(sym) if hasattr(bars, "get") else None
                if bar is not None and getattr(bar, "close", None) is not None:
                    price = float(bar.close)
                    timestamp = str(getattr(bar, "timestamp", ""))
                    source = "bar"
                    stale = age_seconds(getattr(bar, "timestamp", None))

            out[sym] = {
                "ok": price is not None,
                "price": round(float(price), 4) if price is not None else None,
                "source": source,
                "timestamp": timestamp,
                "stale_seconds": round(stale, 1) if stale is not None else None,
                "quote_spread_pct": round(spread, 3) if spread is not None else None,
                "error": "; ".join(errors) if price is None else "",
            }
        time.sleep(0.05)
    return out


def recompute_core_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "close" not in df.columns:
        return df
    c = pd.to_numeric(df["close"], errors="coerce")
    h = pd.to_numeric(df["high"], errors="coerce") if "high" in df.columns else c
    l = pd.to_numeric(df["low"], errors="coerce") if "low" in df.columns else c

    df["return_1d"] = c.pct_change(1) * 100
    df["returns_1d"] = c.pct_change(1)
    df["return_5d"] = c.pct_change(5) * 100
    df["return_20d"] = c.pct_change(20) * 100
    df["sma_20"] = c.rolling(20, min_periods=10).mean()
    df["sma_50"] = c.rolling(50, min_periods=25).mean()
    df["sma_200"] = c.rolling(200, min_periods=50).mean()
    df["sma_200_sign"] = (df["sma_200"].diff() > 0).astype(float)
    df["sma_50_sign"] = (df["sma_50"].diff() > 0).astype(float)
    df["close_vs_sma_50"] = (c - df["sma_50"]) / df["sma_50"].replace(0, np.nan) * 100
    df["close_vs_sma_200"] = (c - df["sma_200"]) / df["sma_200"].replace(0, np.nan) * 100

    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain_14 = gain.ewm(com=13, adjust=False).mean()
    avg_loss_14 = loss.ewm(com=13, adjust=False).mean().replace(0, np.nan)
    df["rsi_14"] = 100 - 100 / (1 + avg_gain_14 / avg_loss_14)
    avg_gain_21 = gain.ewm(com=20, adjust=False).mean()
    avg_loss_21 = loss.ewm(com=20, adjust=False).mean().replace(0, np.nan)
    df["rsi_21"] = 100 - 100 / (1 + avg_gain_21 / avg_loss_21)

    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    df["atr_14"] = tr.ewm(com=13, adjust=False).mean()
    df["atr_pct"] = df["atr_14"] / c.replace(0, np.nan) * 100

    bb_mid = c.rolling(20, min_periods=10).mean()
    bb_std = c.rolling(20, min_periods=10).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    df["bb_width"] = bb_range / bb_mid.replace(0, np.nan)
    df["bb_position"] = (c - bb_lower) / bb_range
    df["zscore_vs_60d"] = (c - c.rolling(60, min_periods=20).mean()) / c.rolling(60, min_periods=20).std().replace(0, np.nan)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["momentum_20d"] = c.pct_change(20) * 100
    df["momentum_60d"] = c.pct_change(60) * 100
    df["momentum_differencing"] = df["momentum_20d"] - df["momentum_60d"]
    df["momentum_squared"] = df["momentum_20d"] ** 2
    df["roc_20"] = c.pct_change(20) * 100

    daily_ret = c.pct_change()
    df["realized_vol_21d"] = daily_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
    vol_short = daily_ret.rolling(21, min_periods=10).std()
    vol_long = daily_ret.rolling(63, min_periods=30).std().replace(0, np.nan)
    df["vol_regime_ratio"] = vol_short / vol_long
    df["vol_regime_stressed"] = (df["vol_regime_ratio"] > 1.5).astype(float)
    df["price_acceleration"] = df["return_1d"] - df["return_1d"].shift(1)
    if "volume" in df.columns:
        df["adv20_dollar_vol"] = (c * pd.to_numeric(df["volume"], errors="coerce")).rolling(20, min_periods=5).mean()
    return df


def update_feature_file(path: Path, mark: dict, stamp: datetime, dry_run: bool) -> dict:
    sym = norm_symbol(path.name)
    if not mark.get("ok") or not mark.get("price"):
        return {"symbol": sym, "updated": False, "reason": mark.get("error") or "no_live_price"}
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        return {"symbol": sym, "updated": False, "reason": f"read_error:{exc}"}
    if df is None or df.empty:
        return {"symbol": sym, "updated": False, "reason": "empty_parquet"}

    df = df.loc[:, ~df.columns.duplicated()].copy()
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
    else:
        dates = pd.to_datetime(df.index, errors="coerce")
    df = df.reset_index(drop=True)
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    df["date"] = dates.to_numpy()
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    if df.empty:
        return {"symbol": sym, "updated": False, "reason": "no_valid_rows"}

    today = stamp.date()
    price = float(mark["price"])
    last_idx = len(df) - 1
    last_date = pd.Timestamp(df.loc[last_idx, "date"]).date()
    prev_close = num(df.loc[last_idx, "close"], price)

    if last_date == today:
        target_idx = last_idx
    else:
        row = df.iloc[last_idx].copy()
        row["date"] = pd.Timestamp(today)
        if "open" in df.columns:
            row["open"] = prev_close
        if "high" in df.columns:
            row["high"] = max(prev_close, price)
        if "low" in df.columns:
            row["low"] = min(prev_close, price)
        row["close"] = price
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        target_idx = len(df) - 1

    if "open" in df.columns and not num(df.loc[target_idx, "open"]):
        df.loc[target_idx, "open"] = prev_close
    if "high" in df.columns:
        df.loc[target_idx, "high"] = max(num(df.loc[target_idx, "high"], price), price)
    if "low" in df.columns:
        low = num(df.loc[target_idx, "low"], price)
        df.loc[target_idx, "low"] = min(low if low > 0 else price, price)
    df.loc[target_idx, "close"] = price
    df["intraday_live_price"] = np.nan
    df["intraday_live_source"] = ""
    df["intraday_live_timestamp"] = ""
    df.loc[target_idx, "intraday_live_price"] = price
    df.loc[target_idx, "intraday_live_source"] = mark.get("source", "")
    df.loc[target_idx, "intraday_live_timestamp"] = mark.get("timestamp", "")
    df = recompute_core_features(df)

    if not dry_run:
        df.to_parquet(path, index=False)
        os.utime(path, None)
    return {
        "symbol": sym,
        "updated": True,
        "price": round(price, 4),
        "source": mark.get("source", ""),
        "timestamp": mark.get("timestamp", ""),
        "stale_seconds": mark.get("stale_seconds"),
        "quote_spread_pct": mark.get("quote_spread_pct"),
        "path": str(path),
    }


def write_report(payload: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh US fixed-return feature parquets with Alpaca intraday marks.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="feature parquet root")
    parser.add_argument("--symbols", default="", help="comma-separated symbols or a file of symbols")
    parser.add_argument("--limit", type=int, default=0, help="limit symbols for smoke tests")
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("ALPACA_LATEST_BATCH_SIZE", "200")))
    parser.add_argument("--force", action="store_true", help="run outside US market hours")
    parser.add_argument("--dry-run", action="store_true", help="fetch and report without writing parquet files")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT / ".env.example", override=False)

    stamp = now_ny()
    root = Path(args.root)
    if not args.force and not is_market_time(stamp):
        payload = {
            "ok": True,
            "skipped": True,
            "reason": "outside_us_market_hours",
            "market_time_ny": stamp.isoformat(),
        }
        write_report(payload)
        print(json.dumps(payload, indent=2))
        return 0

    symbols = load_symbols(root, args.symbols or None, args.limit)
    marks = fetch_live_prices(symbols, args.batch_size)
    results = []
    for sym in symbols:
        path = root / f"{sym}.parquet"
        if not path.exists():
            path = root / f"{sym}_US.parquet"
        if not path.exists():
            results.append({"symbol": sym, "updated": False, "reason": "feature_file_missing"})
            continue
        results.append(update_feature_file(path, marks.get(sym, {}), stamp, args.dry_run))

    updated = sum(1 for row in results if row.get("updated"))
    payload = {
        "ok": True,
        "dry_run": bool(args.dry_run),
        "timestamp": stamp.astimezone(ZoneInfo("UTC")).isoformat(),
        "market_time_ny": stamp.isoformat(),
        "root": str(root),
        "symbols_requested": len(symbols),
        "quotes_ok": sum(1 for row in marks.values() if row.get("ok")),
        "files_updated": updated,
        "files_failed": len(results) - updated,
        "results": results[:500],
    }
    write_report(payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
