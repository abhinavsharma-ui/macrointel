from __future__ import annotations

import argparse
import json
from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv


POSITIONS_PATH = Path("reports/fixed_return_open_positions.json")
SIGNALS_PATH = Path("reports/fixed_return_daily_signals.json")
QUOTE_PATH = Path("reports/fixed_return_intraday_quotes.json")
SHADOW_SCORES_PATH = Path("reports/fixed_return_intraday_shadow_scores.json")
HISTORY_PATH = Path("reports/fixed_return_intraday_mtm_history.json")

INITIAL_CAPITAL = 100_000.0
NY = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

load_dotenv(Path(".env"))
load_dotenv(Path(".env.example"), override=False)


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def now_ny() -> datetime:
    return datetime.now(NY)


def is_market_time(ts: datetime) -> bool:
    if ts.weekday() >= 5:
        return False
    t = ts.time()
    return MARKET_OPEN <= t <= MARKET_CLOSE


def norm_symbol(symbol: str) -> str:
    return str(symbol or "").replace("_US", "").replace(".US", "").upper()


def yf_symbol(symbol: str) -> str:
    sym = norm_symbol(symbol)
    return sym.replace(".", "-")


def latest_close_frame(df: pd.DataFrame, symbol: str) -> tuple[float | None, str | None]:
    if df is None or df.empty:
        return None, None

    sub = df
    yf_sym = yf_symbol(symbol)
    if isinstance(df.columns, pd.MultiIndex):
        if yf_sym in df.columns.get_level_values(0):
            sub = df[yf_sym]
        elif symbol in df.columns.get_level_values(0):
            sub = df[symbol]
        elif yf_sym in df.columns.get_level_values(-1):
            sub = df.xs(yf_sym, axis=1, level=-1)
        elif symbol in df.columns.get_level_values(-1):
            sub = df.xs(symbol, axis=1, level=-1)
        else:
            return None, None

    close_col = None
    for candidate in ("Close", "Adj Close", "close", "adj_close"):
        if candidate in sub.columns:
            close_col = candidate
            break
    if close_col is None:
        return None, None

    close = pd.to_numeric(sub[close_col], errors="coerce").dropna()
    if close.empty:
        return None, None

    idx = close.index[-1]
    value = float(close.iloc[-1])
    try:
        ts = pd.Timestamp(idx)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        ts = ts.tz_convert(NY)
        stamp = ts.isoformat()
    except Exception:
        stamp = str(idx)
    return value, stamp


def download_quotes(symbols: list[str], interval: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    unique = sorted({norm_symbol(s) for s in symbols if norm_symbol(s)})
    if not unique:
        return out
    try:
        import os as _os
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestBarRequest
        client = StockHistoricalDataClient(
            api_key=_os.getenv("ALPACA_API_KEY", ""),
            secret_key=_os.getenv("ALPACA_SECRET_KEY", ""),
        )
        bars = client.get_stock_latest_bar(StockLatestBarRequest(symbol_or_symbols=unique))
        for sym in unique:
            bar = bars.get(sym)
            if bar is None:
                out[sym] = {"ok": False, "error": "no_data_alpaca"}
            else:
                out[sym] = {"ok": True, "price": round(float(bar.close), 4),
                            "timestamp": str(bar.timestamp), "interval": interval}
    except Exception as exc:
        return {sym: {"ok": False, "error": f"alpaca_error: {exc}"} for sym in unique}
    return out

def business_days_held(entry_date: str | None, current_date: datetime) -> int:
    if not entry_date:
        return 0
    entry = pd.to_datetime(entry_date, errors="coerce")
    if pd.isna(entry):
        return 0
    days = pd.bdate_range(entry.normalize(), pd.Timestamp(current_date.date()))
    return max(0, len(days) - 1)


def update_position(pos: dict, quote: dict | None, stamp_now: datetime, interval: str) -> dict:
    if str(pos.get("status", "open")).lower() not in {"open", "active"}:
        return pos

    sym = norm_symbol(pos.get("symbol"))
    quote = quote or {"ok": False, "error": "missing_quote"}
    pos["symbol"] = sym
    pos["last_intraday_mtm_at"] = stamp_now.astimezone(timezone.utc).isoformat()
    pos["intraday_mtm_source"] = "alpaca"
    pos["intraday_mtm_interval"] = interval

    if not quote.get("ok"):
        pos["intraday_mtm_error"] = quote.get("error", "quote_unavailable")
        return pos

    current = float(quote["price"])
    entry = float(pos.get("entry_price") or 0)
    pos_pct = float(pos.get("position_pct") or 0)
    ret = (current / entry - 1.0) * 100.0 if entry > 0 and current > 0 else 0.0
    pnl = INITIAL_CAPITAL * pos_pct * ret / 100.0

    pos["current_price"] = round(current, 4)
    pos["last_price"] = round(current, 4)
    pos["last_price_date"] = stamp_now.date().isoformat()
    pos["current_timestamp"] = quote.get("timestamp") or stamp_now.isoformat()
    pos["unrealized_pnl"] = round(pnl, 2)
    pos["unrealized_pnl_pct"] = round(ret, 4)
    pos["days_held"] = business_days_held(pos.get("entry_date"), stamp_now)
    pos.pop("intraday_mtm_error", None)
    return pos


def write_shadow_scores(quotes: dict[str, dict], stamp_now: datetime, interval: str) -> None:
    doc = load_json(SIGNALS_PATH, {})
    signals = doc.get("signals", []) if isinstance(doc, dict) else []
    rows = []
    for row in signals:
        if not isinstance(row, dict):
            continue
        sym = norm_symbol(row.get("symbol"))
        quote = quotes.get(sym) or {}
        entry = float(row.get("entry_price") or row.get("price") or row.get("close") or 0) or 0.0
        current = float(quote.get("price") or entry or 0.0)
        ret = (current / entry - 1.0) * 100.0 if entry > 0 and current > 0 else 0.0
        rows.append({
            "rank": row.get("rank"),
            "symbol": sym,
            "probability": row.get("probability") or row.get("prob") or row.get("score"),
            "entry_price": round(entry, 4),
            "current_price": round(current, 4),
            "intraday_return_pct": round(ret, 4),
            "quote_ok": bool(quote.get("ok")),
            "quote_timestamp": quote.get("timestamp"),
            "note": "shadow_only_no_orders",
        })

    write_json(SHADOW_SCORES_PATH, {
        "timestamp": stamp_now.astimezone(timezone.utc).isoformat(),
        "market_time_ny": stamp_now.isoformat(),
        "interval": interval,
        "signal_date": doc.get("signal_date") if isinstance(doc, dict) else None,
        "shadow_only": True,
        "orders_enabled": False,
        "scores": rows,
    })


def write_history(positions: list[dict], stamp_now: datetime, quotes: dict[str, dict]) -> dict:
    open_positions = [p for p in positions if str(p.get("status", "open")).lower() in {"open", "active"}]
    unrealized = sum(float(p.get("unrealized_pnl") or 0) for p in open_positions)
    gross = sum(INITIAL_CAPITAL * float(p.get("position_pct") or 0) for p in open_positions)
    row = {
        "timestamp": stamp_now.astimezone(timezone.utc).isoformat(),
        "date": stamp_now.date().isoformat(),
        "source": "intraday_alpaca",
        "quotes_ok": sum(1 for q in quotes.values() if q.get("ok")),
        "quotes_requested": len(quotes),
        "open_positions": len(open_positions),
        "gross_open_value": round(gross, 2),
        "unrealized_pnl": round(unrealized, 2),
        "portfolio_value_est": round(INITIAL_CAPITAL + unrealized, 2),
    }
    existing = load_json(HISTORY_PATH, [])
    if isinstance(existing, dict):
        existing = existing.get("history", [])
    history = [x for x in existing if isinstance(x, dict)]
    if history and history[-1].get("date") == row["date"]:
        history[-1] = row
    else:
        history.append(row)
    write_json(HISTORY_PATH, {"history": history[-250:]})
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Intraday US fixed-return mark-to-market from yfinance.")
    parser.add_argument("--interval", default="30m")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    stamp_now = now_ny()
    if not args.force and not is_market_time(stamp_now):
        payload = {
            "timestamp": stamp_now.astimezone(timezone.utc).isoformat(),
            "skipped": True,
            "reason": "outside_us_market_hours",
            "market_time_ny": stamp_now.isoformat(),
        }
        write_json(QUOTE_PATH, payload)
        print(json.dumps(payload, indent=2))
        return 0

    doc = load_json(POSITIONS_PATH, {"positions": []})
    positions = doc.get("positions", []) if isinstance(doc, dict) else []
    positions = [dict(p) for p in positions if isinstance(p, dict)]
    open_symbols = [norm_symbol(p.get("symbol")) for p in positions if str(p.get("status", "open")).lower() in {"open", "active"}]
    sig_doc = load_json(SIGNALS_PATH, {})
    signal_symbols = [norm_symbol(s.get("symbol")) for s in sig_doc.get("signals", [])] if isinstance(sig_doc, dict) else []
    symbols = sorted({s for s in open_symbols + signal_symbols if s})
    quotes = download_quotes(symbols, args.interval)

    updated = [update_position(dict(p), quotes.get(norm_symbol(p.get("symbol"))), stamp_now, args.interval) for p in positions]
    history_row = write_history(updated, stamp_now, quotes)
    write_shadow_scores(quotes, stamp_now, args.interval)
    write_json(POSITIONS_PATH, {"updated_at": stamp_now.astimezone(timezone.utc).isoformat(), "positions": updated})
    write_json(QUOTE_PATH, {
        "timestamp": stamp_now.astimezone(timezone.utc).isoformat(),
        "market_time_ny": stamp_now.isoformat(),
        "interval": args.interval,
        "symbols": symbols,
        "quotes": quotes,
        "summary": history_row,
    })

    print(json.dumps({
        "mtm_source": "intraday_alpaca",
        "quotes_ok": history_row["quotes_ok"],
        "quotes_requested": history_row["quotes_requested"],
        "open_positions": history_row["open_positions"],
        "unrealized_pnl": history_row["unrealized_pnl"],
        "portfolio_value_est": history_row["portfolio_value_est"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
