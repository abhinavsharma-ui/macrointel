from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.example", override=False)
PORT = int(os.getenv("DASHBOARD_ULTRA_PORT", "5055"))
INITIAL = float(os.getenv("DASHBOARD_US_INITIAL_CAPITAL", os.getenv("PAPER_CAPITAL", "100000")))
START = time.time()

app = Flask(__name__)
CORS(app)


BROKER = [
    ROOT / "data/paper_broker_state.json",
    BASE / "data/paper_broker_state.json",
    BASE / "project/data/paper_broker_state.json",
]
OPEN = [
    ROOT / "reports/fixed_return_open_positions.json",
    BASE / "reports/fixed_return_open_positions.json",
    BASE / "project/reports/fixed_return_open_positions.json",
]
SIGNALS = [
    ROOT / "reports/fixed_return_daily_signals.json",
    BASE / "reports/fixed_return_daily_signals.json",
    BASE / "project/reports/fixed_return_daily_signals.json",
    ROOT / "reports/fixed_return_live_paper.json",
    BASE / "reports/fixed_return_live_paper.json",
    BASE / "project/reports/fixed_return_live_paper.json",
]
SCORES = [
    ROOT / "reports/fixed_return_daily_scores.json",
    BASE / "reports/fixed_return_daily_scores.json",
    BASE / "project/reports/fixed_return_daily_scores.json",
]
MODEL_HEALTH = [
    ROOT / "reports/model_health.json",
    BASE / "reports/model_health.json",
    BASE / "project/reports/model_health.json",
]
UNIFIED_RISK = [
    ROOT / "reports/unified_risk_state.json",
    BASE / "reports/unified_risk_state.json",
    BASE / "project/reports/unified_risk_state.json",
]
SL_DECISIONS = [
    ROOT / "reports/sl_decisions.json",
    BASE / "reports/sl_decisions.json",
    BASE / "project/reports/sl_decisions.json",
]
TRADES = list(
    dict.fromkeys(
        Path(p).resolve()
        for p in [
            ROOT / "reports/fixed_return_paper_trades.csv",
            BASE / "reports/fixed_return_paper_trades.csv",
            BASE / "project/reports/fixed_return_paper_trades.csv",
        ]
    )
)
FEATURES = [
    ROOT / "data/features",
    BASE / "data/features",
    BASE / "project/data/features",
    ROOT / "data/features_26yr_liquid",
    BASE / "data/features_26yr_liquid",
    BASE / "project/data/features_26yr_liquid",
]
NSE_DIRS = [BASE / "reports/nse_runtime", ROOT / "reports/nse_runtime"]


def read_json(paths, default):
    for path in paths:
        try:
            path = Path(path)
            if path.exists():
                return json.loads(path.read_text())
        except Exception:
            pass
    return default


def broker_state():
    data = read_json(BROKER, {})
    return data if isinstance(data, dict) else {}


def num(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def parse_dt(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        try:
            return datetime.strptime(str(value)[:10], "%Y-%m-%d")
        except Exception:
            return None


def business_days(start: date, end: date) -> int:
    if end <= start:
        return 0
    out = 0
    cur = start
    while cur < end:
        cur += timedelta(days=1)
        if cur.weekday() < 5:
            out += 1
    return out


def current_price(symbol: str, fallback: float) -> float:
    for folder in FEATURES:
        for name in [f"{symbol}.parquet", f"{symbol}_US.parquet", f"{symbol}.US.parquet"]:
            path = folder / name
            if not path.exists():
                continue
            try:
                import pandas as pd

                df = pd.read_parquet(path)
                if "close" in df.columns and len(df):
                    return float(df["close"].iloc[-1])
            except Exception:
                pass
    return num(fallback)


def alpaca_live_prices(symbols: list) -> dict:
    """Alpaca latest trade/quote price per symbol. Never raises."""
    if not symbols:
        return {}
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestBarRequest, StockLatestQuoteRequest, StockLatestTradeRequest
        client = StockHistoricalDataClient(
            api_key=os.getenv("ALPACA_API_KEY", ""),
            secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
        )
        unique = sorted(set(s.upper() for s in symbols if s))
        trades, quotes, bars = {}, {}, {}
        errors = []
        try:
            trades = client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=unique))
        except Exception as exc:
            errors.append(f"trade:{exc}")
        try:
            quotes = client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=unique))
        except Exception as exc:
            errors.append(f"quote:{exc}")
        if not trades:
            try:
                bars = client.get_stock_latest_bar(StockLatestBarRequest(symbol_or_symbols=unique))
            except Exception as exc:
                errors.append(f"bar:{exc}")

        quote_after_seconds = num(os.getenv("ALPACA_QUOTE_MID_AFTER_SECONDS"), 120.0)
        max_quote_spread_pct = num(os.getenv("ALPACA_QUOTE_MID_MAX_SPREAD_PCT"), 2.5)

        def age_seconds(ts):
            try:
                now = datetime.now(ts.tzinfo) if getattr(ts, "tzinfo", None) else datetime.utcnow()
                return max(0.0, (now - ts).total_seconds())
            except Exception:
                return None

        out = {}
        for sym in unique:
            price, timestamp, source, stale_seconds, spread_pct = None, "", "", None, None
            trade = trades.get(sym) if hasattr(trades, "get") else None
            if trade is not None and getattr(trade, "price", None) is not None:
                price = float(trade.price)
                timestamp = str(getattr(trade, "timestamp", ""))
                source = "trade"
                stale_seconds = age_seconds(getattr(trade, "timestamp", None))

            quote = quotes.get(sym) if hasattr(quotes, "get") else None
            if quote is not None:
                bid = num(getattr(quote, "bid_price", None))
                ask = num(getattr(quote, "ask_price", None))
                if bid > 0 and ask > 0 and ask >= bid:
                    mid = (bid + ask) / 2.0
                    spread_pct = (ask - bid) / mid * 100.0 if mid else None
                    use_quote = price is None or (
                        stale_seconds is not None
                        and stale_seconds >= quote_after_seconds
                        and spread_pct is not None
                        and spread_pct <= max_quote_spread_pct
                    )
                    if use_quote:
                        price = mid
                        timestamp = str(getattr(quote, "timestamp", ""))
                        source = "quote_mid"
                        stale_seconds = age_seconds(getattr(quote, "timestamp", None))

            if price is None:
                bar = bars.get(sym) if hasattr(bars, "get") else None
                if bar is not None and getattr(bar, "close", None) is not None:
                    price = float(bar.close)
                    timestamp = str(getattr(bar, "timestamp", ""))
                    source = "bar"
                    stale_seconds = age_seconds(getattr(bar, "timestamp", None))

            if price is None:
                out[sym] = {"ok": False, "price": None, "source": "", "error": "; ".join(errors)}
            else:
                out[sym] = {
                    "ok": True,
                    "price": round(float(price), 4),
                    "source": source,
                    "timestamp": timestamp,
                    "stale_seconds": round(stale_seconds, 1) if stale_seconds is not None else None,
                    "quote_spread_pct": round(spread_pct, 3) if spread_pct is not None else None,
                }
        return out
    except Exception as exc:
        return {s.upper(): {"ok": False, "price": None, "error": str(exc)} for s in symbols}


def volatility_regime():
    vol = None
    for folder in FEATURES:
        path = folder / "SPY.parquet"
        if not path.exists():
            continue
        try:
            import pandas as pd

            df = pd.read_parquet(path)
            if "return_1d" in df.columns:
                ret = df["return_1d"].dropna()
            else:
                ret = df["close"].pct_change().dropna()
            vol = float(ret.tail(20).std() * math.sqrt(252) * 100)
            break
        except Exception:
            pass
    if vol is None or not math.isfinite(vol):
        vol = 0.0
    if vol < 15:
        label, mult, desc = "LOW", 0.75, "Low-vol sizing reduced"
    elif vol < 25:
        label, mult, desc = "MEDIUM", 1.0, "Normal volatility"
    elif vol < 40:
        label, mult, desc = "HIGH", 1.25, "High-vol edge regime"
    else:
        label, mult, desc = "EXTREME", 1.5, "Crisis alpha regime"
    return {"spy_realized_vol": round(vol, 2), "vol_regime": label, "vol_multiplier": mult, "description": desc}


def signal_payload():
    data = read_json(SIGNALS, {})
    raw = data.get("signals", []) if isinstance(data, dict) else []
    if not raw and isinstance(data, dict):
        raw = data.get("orders", []) or data.get("positions", []) or []
    rows = []
    for index, signal in enumerate(raw[:16], 1):
        trade = signal.get("result", {}).get("trade", {}) if isinstance(signal.get("result"), dict) else {}
        meta = trade.get("metadata", {}) if isinstance(trade, dict) else {}
        price = num(signal.get("entry_price") or signal.get("price") or signal.get("close") or trade.get("reference_price") or trade.get("fill_price"))
        raw_stop = signal.get("stop_loss_price")
        stop_loss_price = num(raw_stop) if raw_stop is not None else price * 0.97
        rows.append(
            {
                "rank": int(signal.get("rank") or index),
                "symbol": str(signal.get("symbol") or trade.get("symbol") or ""),
                "probability": num(signal.get("probability") or signal.get("prob") or signal.get("score") or meta.get("take_probability")),
                "entry_price": price,
                "position_pct": round(num(signal.get("position_pct") or meta.get("position_pct")) * 100.0, 4),
                "profit_target_price": num(signal.get("profit_target_price") or price * 1.05),
                "stop_loss_price": stop_loss_price,
                "stop_loss_enabled": bool(signal.get("stop_loss_enabled", stop_loss_price > 0)),
                "expected_exit_date": str(signal.get("expected_exit_date") or ""),
            }
        )
    return {
        "signal_date": str(data.get("signal_date") or date.today().isoformat()) if isinstance(data, dict) else date.today().isoformat(),
        "signals": rows,
    }


def _open_symbol_set() -> set:
    raw = read_json(OPEN, [])
    if isinstance(raw, dict):
        raw = raw.get("positions", [])
        raw = list(raw.values()) if isinstance(raw, dict) else raw
    if not isinstance(raw, list):
        raw = []
    out = set()
    for pos in raw:
        if not isinstance(pos, dict):
            continue
        if str(pos.get("status", "open")).lower() not in {"open", "active"}:
            continue
        sym = str(pos.get("symbol") or "").strip().upper()
        if sym:
            out.add(sym)
    return out


def diagnostic_examples(limit: int = 10) -> dict:
    scores = read_json(SCORES, {})
    rows = scores.get("scores", []) if isinstance(scores, dict) else []
    source = "score_cache" if rows else "daily_signals"
    if not rows:
        data = read_json(SIGNALS, {})
        rows = data.get("signals", []) if isinstance(data, dict) else []
        scores = data if isinstance(data, dict) else {}

    try:
        scripts_dir = str(ROOT / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import fixed_return_daily_signals as sig

        threshold = float(scores.get("threshold") or sig.SIG_THRESHOLD)
        is_allowed = sig.is_allowed_symbol
    except Exception:
        threshold = float(scores.get("threshold") or 0.61)
        is_allowed = lambda symbol: True

    open_symbols = _open_symbol_set()
    examples = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            continue
        sym = str(row.get("symbol") or "").strip().upper()
        if not sym or sym in open_symbols or not is_allowed(sym):
            continue
        probability = num(row.get("probability"))
        if probability < threshold:
            continue
        entry = num(row.get("entry_price") or row.get("price") or row.get("close"))
        examples.append(
            {
                "symbol": sym,
                "probability": round(probability, 6),
                "rank": int(row.get("ml_rank") or row.get("rank") or index),
                "entry_price": round(entry, 4),
                "source": source,
            }
        )
    examples = sorted(examples, key=lambda item: item["probability"], reverse=True)[: max(1, limit)]
    return {
        "ok": True,
        "source": source,
        "threshold": threshold,
        "open_symbols": sorted(open_symbols),
        "examples": examples,
    }


def open_positions():
    raw = read_json(OPEN, [])
    if isinstance(raw, dict):
        raw = raw.get("positions", [])
        raw = list(raw.values()) if isinstance(raw, dict) else raw
    if not isinstance(raw, list):
        raw = []
    broker = read_json(BROKER, {})
    rows = []
    for pos in raw:
        if str(pos.get("status", "open")).lower() not in {"open", "active"}:
            continue
        sym = str(pos.get("symbol", ""))
        entry = num(pos.get("entry_price"))
        current = num(pos.get("current_price") or pos.get("last_price"))
        if not current:
            current = current_price(sym, entry)
        entry_dt = parse_dt(pos.get("entry_date")) or datetime.now()
        qty = num(pos.get("quantity"))
        pos_pct = num(pos.get("position_pct"))
        raw_stop = pos.get("stop_loss_price")
        stop_loss_price = num(raw_stop) if raw_stop is not None else entry * 0.97
        ret = ((current / entry) - 1.0) * 100.0 if entry else 0.0
        pnl = (current - entry) * qty if qty else INITIAL * pos_pct * ret / 100.0
        rows.append(
            {
                "symbol": sym,
                "entry_date": entry_dt.date().isoformat(),
                "entry_price": round(entry, 4),
                "current_price": round(current, 4),
                "profit_target_price": round(num(pos.get("profit_target_price") or entry * 1.05), 4),
                "stop_loss_price": round(stop_loss_price, 4),
                "stop_loss_enabled": bool(pos.get("stop_loss_enabled", stop_loss_price > 0)),
                "days_held": business_days(entry_dt.date(), date.today()),
                "position_pct": round(pos_pct * 100.0, 4),
                "unrealized_pnl": round(pnl, 2),
                "unrealized_pnl_pct": round(ret, 2),
                "confidence": num(pos.get("probability") or pos.get("confidence")),
                "sl_grace": pos.get("sl_grace") if isinstance(pos.get("sl_grace"), dict) else None,
            }
        )
    for key, val in (broker.get("positions", {}) if isinstance(broker, dict) else {}).items():
        sym = str(val.get("symbol") or key.split("::")[0])
        if any(row["symbol"] == sym for row in rows):
            continue
        entry = num(val.get("avg_cost") or val.get("entry_price"))
        qty = num(val.get("quantity"))
        current = num(val.get("current_price") or val.get("last_price"))
        if not current:
            current = current_price(sym, entry)
        entry_dt = parse_dt(val.get("opened_at")) or datetime.now()
        ret = ((current / entry) - 1.0) * 100.0 if entry else 0.0
        rows.append(
            {
                "symbol": sym,
                "entry_date": entry_dt.date().isoformat(),
                "entry_price": round(entry, 4),
                "current_price": round(current, 4),
                "profit_target_price": round(entry * 1.05, 4),
                "stop_loss_price": round(entry * 0.97, 4),
                "days_held": business_days(entry_dt.date(), date.today()),
                "position_pct": round(qty * entry / INITIAL * 100 if entry else 0, 4),
                "unrealized_pnl": round((current - entry) * qty, 2),
                "unrealized_pnl_pct": round(ret, 2),
                "confidence": 0.0,
                "sl_grace": None,
            }
        )
    return sorted(rows, key=lambda row: row["days_held"], reverse=True)


def live_mark_positions(positions):
    rows = [dict(pos) for pos in (positions or [])]
    prices = alpaca_live_prices([row.get("symbol", "") for row in rows])
    for row in rows:
        sym = str(row.get("symbol", "")).upper()
        px = prices.get(sym, {})
        live_px = num(px.get("price")) or num(row.get("current_price") or row.get("entry_price"))
        entry = num(row.get("entry_price"))
        qty = num(row.get("quantity"))
        display_pos_pct = num(row.get("position_pct"))
        pos_fraction = display_pos_pct / 100.0
        ret = (live_px / entry - 1.0) if entry > 0 and live_px > 0 else 0.0
        pnl = (live_px - entry) * qty if qty else INITIAL * pos_fraction * ret
        row.update(
            {
                "current_price": round(live_px, 4),
                "unrealized_pnl": round(pnl, 2),
                "unrealized_pnl_pct": round(ret * 100.0, 3),
                "position_fraction": round(pos_fraction, 8),
                "live_price_ok": bool(px.get("ok")),
                "live_price_source": px.get("source", ""),
                "live_price_timestamp": px.get("timestamp", ""),
                "live_price_stale_seconds": px.get("stale_seconds"),
                "quote_spread_pct": px.get("quote_spread_pct"),
            }
        )
    return rows


def us_unrealized_attribution(positions):
    rows = []
    for pos in positions or []:
        pnl = num(pos.get("unrealized_pnl"))
        entry = num(pos.get("entry_price"))
        current = num(pos.get("current_price"))
        ret = num(pos.get("unrealized_pnl_pct"))
        rows.append(
            {
                "symbol": pos.get("symbol") or "",
                "position_pct": num(pos.get("position_pct")),
                "entry_price": entry,
                "current_price": current,
                "return_pct": ret,
                "pnl": pnl,
            }
        )
    losers = sorted([row for row in rows if row["pnl"] < 0], key=lambda row: row["pnl"])
    winners = sorted([row for row in rows if row["pnl"] > 0], key=lambda row: row["pnl"], reverse=True)
    gross_loss = abs(sum(row["pnl"] for row in losers))
    gross_gain = sum(row["pnl"] for row in winners)
    for row in losers:
        row["gross_loss_share_pct"] = abs(row["pnl"]) / gross_loss * 100.0 if gross_loss else 0.0
    return {
        "net_unrealized_pnl": round(sum(row["pnl"] for row in rows), 2),
        "gross_loss": round(gross_loss, 2),
        "gross_gain": round(gross_gain, 2),
        "losers": losers[:12],
        "winners": winners[:12],
    }


def closed_trades():
    rows = []
    for path in TRADES:
        if not path.exists():
            continue
        try:
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    entry = num(row.get("entry_price"))
                    exit_price = num(row.get("exit_price"))
                    qty = num(row.get("quantity") or row.get("qty"))
                    if not qty:
                        _pp = num(row.get("position_pct"))
                        if _pp and entry and entry > 0:
                            qty = round(_pp * 100_000 / entry, 2)
                    ret = num(row.get("return_pct") or row.get("net_ret"))
                    pnl = num(row.get("pnl") or row.get("realized_pnl"))
                    if not pnl and entry and exit_price and qty:
                        pnl = (exit_price - entry) * qty
                    if not ret and entry and exit_price:
                        ret = (exit_price / entry - 1.0) * 100.0
                    reason = str(row.get("exit_reason") or row.get("reason") or ("profit_target" if ret >= 4.5 else "stop_loss" if ret <= -2.5 else "time_exit"))
                    rows.append(
                        {
                            "exit_date": str(row.get("exit_date") or row.get("closed_at") or row.get("date") or row.get("timestamp") or "")[:10],
                            "symbol": row.get("symbol", ""),
                            "entry_price": round(entry, 4),
                            "exit_price": round(exit_price, 4),
                            "hold_days": (lambda ed, xd: ((__import__('datetime').date.fromisoformat(xd) - __import__('datetime').date.fromisoformat(ed)).days) if ed and xd and len(ed)>=10 and len(xd)>=10 else int(num(row.get("hold_days"))))(str(row.get("entry_date","") or "")[:10], str(row.get("exit_date","") or row.get("closed_at","") or "")[:10]),
                            "return_pct": round(ret, 2),
                            "pnl": round(pnl, 2),
                            "quantity": round(qty, 2) if qty else None,
                            "exit_reason": reason,
                        }
                    )
        except Exception:
            pass
    rows = [row for row in rows if row.get("symbol")]
    rows.sort(key=lambda row: row.get("exit_date", ""), reverse=True)
    return rows


def broker_account_snapshot():
    broker = broker_state()
    if not broker:
        return {}
    initial = num(broker.get("initial_capital"), INITIAL) or INITIAL
    cash = num(broker.get("cash"), None)
    peak = num(broker.get("peak_portfolio_value"), 0.0)
    explicit_value = num(
        broker.get("current_portfolio_value")
        or broker.get("portfolio_value")
        or broker.get("total_value"),
        0.0,
    )
    trade_log = broker.get("trade_log") or []
    realized_from_log = sum(num(row.get("realized_pnl")) for row in trade_log if isinstance(row, dict))
    explicit_realized = num(
        broker.get("realized_pnl")
        or broker.get("closed_pnl")
        or broker.get("cumulative_realized_pnl"),
        0.0,
    )
    realized = explicit_realized or realized_from_log
    if cash is not None and initial and cash >= initial:
        realized = cash - initial
    elif not realized and cash is not None and initial:
        realized = cash - initial
    return {
        "initial_capital": initial,
        "cash": cash,
        "portfolio_value": explicit_value,
        "peak_portfolio_value": peak,
        "realized_pnl": realized,
        "source": "paper_broker_state.json",
    }


def portfolio(use_live=False):
    positions = open_positions()
    if use_live:
        positions = live_mark_positions(positions)
    trades = closed_trades()
    broker_account = broker_account_snapshot()
    initial = num(broker_account.get("initial_capital"), INITIAL) or INITIAL
    realized = sum(num(row["pnl"]) for row in trades)
    cash = initial + realized
    open_notional = sum(initial * num(pos["position_pct"]) / 100.0 for pos in positions)
    open_unrealized = sum(num(pos["unrealized_pnl"]) for pos in positions)
    value = initial + realized + open_unrealized
    if value <= 0:
        value = initial + realized + open_unrealized
    peak = max(initial, value, num(broker_account.get("peak_portfolio_value"), 0.0))
    wins = [row for row in trades if num(row["pnl"]) > 0]
    win_rate = len(wins) / len(trades) * 100.0 if trades else None
    return {
        "initial_capital": initial,
        "cash": round(cash, 2),
        "portfolio_value": round(value, 2),
        "realized_pnl": round(realized, 2),
        "drawdown_from_peak_pct": round(max(0, (peak - value) / peak * 100.0), 3) if peak else 0,
        "open_positions_count": len(positions),
        "positions": positions,
        "closed_trades": trades[:60],
        "closed_trade_count": len(trades),
        "win_rate": round(win_rate, 2) if win_rate is not None else None,
        "total_return_pct": round((value / initial - 1.0) * 100.0, 3) if initial else 0.0,
        "account_source": broker_account.get("source") or "trade_csv",
    }


def analytics():
    data = portfolio()
    trades = data["closed_trades"]
    wins = [row for row in trades if num(row["pnl"]) > 0]
    losses = [row for row in trades if num(row["pnl"]) <= 0]
    pt = [row for row in trades if "profit" in str(row["exit_reason"]).lower()]
    sl = [row for row in trades if "stop" in str(row["exit_reason"]).lower()]
    n = data["closed_trade_count"]
    return {
        "closed_pnl": data["realized_pnl"],
        "closed_trades": n,
        "win_rate": data["win_rate"],
        "drawdown_from_peak_pct": data["drawdown_from_peak_pct"],
        "profit_target_hit_rate": round(len(pt) / n * 100.0, 2) if n else None,
        "avg_hold_days": round(sum(num(row["hold_days"]) for row in trades) / len(trades), 2) if trades else None,
        "avg_winner_pct": round(sum(num(row["return_pct"]) for row in wins) / len(wins), 2) if wins else None,
        "avg_loser_pct": round(sum(num(row["return_pct"]) for row in losses) / len(losses), 2) if losses else None,
        "outcomes": {"profit_target": len(pt), "stop_loss": len(sl), "time_exit": max(0, n - len(pt) - len(sl))},
        "trades": trades,
    }


def pnl_series():
    rows = sorted(closed_trades(), key=lambda row: (row.get("exit_date", ""), row.get("symbol", "")))
    labels, pnl, equity = [], [], []
    total = 0.0
    for row in rows:
        total += num(row["pnl"])
        labels.append(f"{row['symbol']} - {row['exit_date'][-5:]}")
        pnl.append(round(total, 2))
        equity.append(round(INITIAL + total, 2))
    return {
        "labels": labels,
        "cumulative_pnl": pnl,
        "portfolio_value": equity,
        "closed_pnl": round(total, 2),
        "closed_sample": len(rows),
        "source": "closed_trades()",
        "insufficient": len(labels) < 2,
    }

def nse_payload():
    def one(name, default):
        for folder in NSE_DIRS:
            path = folder / name
            try:
                if path.exists():
                    return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return default

    def vol_state(vix):
        value = num(vix)
        if value > 22:
            return {"label": "HIGH", "multiplier": 0.5, "description": "High India VIX: defensive sizing."}
        if value >= 16:
            return {"label": "MID", "multiplier": 1.0, "description": "Normal India VIX: base sizing."}
        return {"label": "LOW", "multiplier": 0.75, "description": "Low India VIX: reduced sizing for hidden narrow-risk."}

    def backtest_reference(regime):
        paths = [
            BASE / "reports/final_nse_dual_regime_v1/nse_dual_regime_v1_combined_report.json",
            ROOT / "reports/final_nse_dual_regime_v1/nse_dual_regime_v1_combined_report.json",
        ]
        scope = regime if regime in {"calm_v2", "crisis_v1"} else "combined"
        for path in paths:
            try:
                if not path.exists():
                    continue
                data = json.loads(path.read_text(encoding="utf-8"))
                rows = data.get("results", [])
                scoped = [row for row in rows if row.get("scope") == scope and int(row.get("fold", 0)) == 4]
                if not scoped and scope != "combined":
                    scoped = [row for row in rows if row.get("scope") == "combined" and int(row.get("fold", 0)) == 4]
                if scoped:
                    row = scoped[0]
                    return {
                        "scope": row.get("scope"),
                        "fold": row.get("fold"),
                        "trades": row.get("trades"),
                        "backtest_wr": round(num(row.get("rel_win_rate_pct")), 2),
                        "backtest_avg_rel_pct": round(num(row.get("avg_rel_ret_pct")), 4),
                    }
            except Exception:
                pass
        return {"scope": scope, "fold": 4, "trades": 0, "backtest_wr": None, "backtest_avg_rel_pct": None}

    def trade_row(row):
        entry = num(row.get("entry_price") or row.get("price"))
        exit_price = num(row.get("exit_price") or row.get("current_price"))
        qty = num(row.get("quantity"))
        pnl = num(row.get("realized_pnl_inr") or row.get("pnl_inr") or row.get("pnl"))
        if not pnl and entry and exit_price and qty:
            pnl = (exit_price - entry) * qty
        ret = num(row.get("return_pct"))
        if not ret and entry and exit_price:
            ret = (exit_price / entry - 1.0) * 100.0
        return {
            "date": str(row.get("exit_date") or row.get("current_date") or row.get("timestamp") or "")[:10],
            "symbol": row.get("symbol", ""),
            "entry": round(entry, 4),
            "exit": round(exit_price, 4),
            "hold": int(num(row.get("days_held") or row.get("hold_days"))),
            "return_pct": round(ret, 4),
            "pnl": round(pnl, 2),
            "reason": str(row.get("exit_reason") or row.get("reason") or "hold_20d"),
        }

    def closed_trades_from(status):
        raw = one("nse_closed_trades.json", [])
        if isinstance(raw, dict):
            raw = raw.get("trades", [])
        rows = [trade_row(row) for row in raw if isinstance(row, dict)]
        for row in status.get("recent_closed", []) if isinstance(status, dict) else []:
            if isinstance(row, dict):
                rows.append(trade_row(row))
        dedup = {}
        for row in rows:
            if row.get("symbol"):
                dedup[(row.get("symbol"), row.get("date"), row.get("entry"), row.get("exit"), row.get("reason"))] = row
        return sorted(dedup.values(), key=lambda row: row.get("date", ""), reverse=True)

    def risk_band(signal):
        eq = signal.get("execution_quality", {}) if isinstance(signal, dict) else {}
        day_range = num(eq.get("day_range_pct")) / 100.0
        if day_range > 0:
            return max(0.05, min(0.12, day_range * 2.0))
        return 0.07

    def enrich_signal(signal):
        entry = num(signal.get("close") or signal.get("entry_price") or signal.get("price"))
        band = risk_band(signal)
        return {
            **signal,
            "entry": round(entry, 4),
            "profit_target_price": round(entry * (1.0 + band), 4) if entry else 0,
            "stop_loss_price": round(entry * (1.0 - band), 4) if entry else 0,
            "risk_band_pct": round(band * 100.0, 2),
        }

    def enrich_position(position):
        entry = num(position.get("entry_price") or position.get("price"))
        current = num(position.get("current_price") or entry)
        qty = num(position.get("quantity"))
        notional = num(position.get("current_value_inr") or position.get("notional_inr") or (qty * entry))
        ret = num(position.get("return_pct"))
        if not ret and entry and current:
            ret = (current / entry - 1.0) * 100.0
        pnl = num(position.get("unrealized_pnl_inr") or position.get("pnl_inr"))
        if not pnl and qty and entry and current:
            pnl = (current - entry) * qty
        target = num(position.get("profit_target_price")) or (entry * 1.07 if entry else 0)
        stop = num(position.get("stop_loss_price")) or (entry * 0.93 if entry else 0)
        days = int(num(position.get("days_held")))
        flags = []
        if stop and current <= stop * 1.015:
            flags.append("within 1.5% of stop")
        if target and current >= target * 0.985:
            flags.append("within 1.5% of target")
        if ret <= -5:
            flags.append(f"drawdown {ret:.2f}%")
        if days > 15:
            flags.append("stale")
        return {
            **position,
            "entry_price": round(entry, 4),
            "current_price": round(current, 4),
            "current_value_inr": round(notional, 2),
            "unrealized_pnl_inr": round(pnl, 2),
            "return_pct": round(ret, 4),
            "profit_target_price": round(target, 4),
            "stop_loss_price": round(stop, 4),
            "alert_flags": flags,
        }

    def curves(status):
        history = one("nse_mtm_history.json", [])
        if isinstance(history, dict):
            history = history.get("history", [])
        if not history and isinstance(status, dict):
            history = [{
                "date": status.get("timestamp", "")[:10],
                "total_pnl_inr": status.get("total_pnl_inr", 0),
                "portfolio_value_est_inr": status.get("portfolio_value_est_inr", 1_000_000),
            }]
        labels, pnl, value = [], [], []
        for row in history[-80:]:
            labels.append(str(row.get("date") or row.get("timestamp") or "")[:10])
            pnl.append(round(num(row.get("total_pnl_inr")), 2))
            value.append(round(num(row.get("portfolio_value_est_inr") or 1_000_000), 2))
        return {"labels": labels, "cumulative_pnl": pnl, "portfolio_value": value}

    sig = one("latest_nse_signals.json", {})
    gate = one("gate_decision.json", {})
    pre_gate = one("signals_pre_gate.json", [])
    blocked = one("blocked_signals.json", [])
    status = one("nse_paper_status.json", {})
    pos_doc = one("nse_paper_positions.json", {})
    orders = one("nse_paper_orders.json", {})
    positions = status.get("positions") or pos_doc.get("positions") or []
    open_pos = [enrich_position(row) for row in positions if row.get("status", "open") == "open"]
    closed = closed_trades_from(status)
    notional = sum(num(row.get("current_value_inr") or row.get("notional_inr")) for row in open_pos)
    pnl = sum(num(row.get("unrealized_pnl_inr") or row.get("pnl_inr")) for row in open_pos)
    realized = sum(num(row.get("pnl")) for row in closed)
    wins = [row for row in closed if num(row.get("pnl")) > 0]
    pt_hits = [row for row in closed if "profit" in str(row.get("reason", "")).lower() or "target" in str(row.get("reason", "")).lower()]
    notionals = sorted([num(row.get("current_value_inr") or row.get("notional_inr")) for row in open_pos], reverse=True)
    capital = num(status.get("capital_inr")) or 1_000_000.0
    near_stop = sum(1 for row in open_pos if any("stop" in flag for flag in row.get("alert_flags", [])))
    losers = sum(1 for row in open_pos if num(row.get("return_pct")) < 0)
    attribution_rows = []
    for row in open_pos:
        pnl_inr = num(row.get("unrealized_pnl_inr") or row.get("pnl_inr"))
        entry = num(row.get("entry_price") or row.get("price"))
        current = num(row.get("current_price") or entry)
        attribution_rows.append({
            "symbol": row.get("symbol", ""),
            "quantity": int(num(row.get("quantity"))),
            "entry_price": round(entry, 4),
            "current_price": round(current, 4),
            "return_pct": round(num(row.get("return_pct")), 4),
            "pnl_inr": round(pnl_inr, 2),
            "status": "open_unrealized",
        })
    gross_loss = abs(sum(num(row.get("pnl_inr")) for row in attribution_rows if num(row.get("pnl_inr")) < 0))
    gross_gain = sum(num(row.get("pnl_inr")) for row in attribution_rows if num(row.get("pnl_inr")) > 0)
    for row in attribution_rows:
        pnl_inr = num(row.get("pnl_inr"))
        row["gross_loss_share_pct"] = round(abs(pnl_inr) / gross_loss * 100.0, 2) if pnl_inr < 0 and gross_loss else 0
    attribution_losers = sorted(
        [row for row in attribution_rows if num(row.get("pnl_inr")) < 0],
        key=lambda row: num(row.get("pnl_inr")),
    )
    attribution_winners = sorted(
        [row for row in attribution_rows if num(row.get("pnl_inr")) > 0],
        key=lambda row: num(row.get("pnl_inr")),
        reverse=True,
    )
    reference = backtest_reference(sig.get("regime"))
    enriched_signals = [enrich_signal(row) for row in sig.get("signals", [])]
    return {
        "ok": bool(sig or positions or orders),
        "asof_date": sig.get("asof_date"),
        "regime": sig.get("regime"),
        "india_vix": sig.get("india_vix"),
        "regime_state": gate.get("regime_state"),
        "narrow_score": gate.get("narrow_score"),
        "allow_new_entries": sig.get("allow_new_entries"),
        "hedge_beta": sig.get("hedge_beta", 0),
        "orders_enabled": orders.get("orders_enabled", False),
        "vol_state": vol_state(sig.get("india_vix")),
        "signals": enriched_signals,
        "signals_pre_gate": pre_gate if isinstance(pre_gate, list) else [],
        "blocked_signals": blocked if isinstance(blocked, list) else [],
        "gate_decision": gate if isinstance(gate, dict) else {},
        "positions": open_pos,
        "unrealized_attribution": {
            "net_unrealized_pnl_inr": round(pnl, 2),
            "gross_open_loss_inr": round(gross_loss, 2),
            "gross_open_gain_inr": round(gross_gain, 2),
            "losers": attribution_losers,
            "winners": attribution_winners,
            "explanation": "No NSE trades are closed; this is open-position mark-to-market from entry price to current price.",
        },
        "closed_trades": closed[:50],
        "orders": orders.get("new_orders", []),
        "curves": curves(status),
        "reality": {
            "live_wr": round(len(wins) / len(closed) * 100.0, 2) if closed else None,
            "backtest_wr": reference.get("backtest_wr"),
            "backtest_avg_rel_pct": reference.get("backtest_avg_rel_pct"),
            "backtest_scope": reference.get("scope"),
            "backtest_fold": reference.get("fold"),
            "live_closed_pnl": round(realized, 2),
            "pt_hit_rate": round(len(pt_hits) / len(closed) * 100.0, 2) if closed else None,
            "closed_sample": len(closed),
        },
        "risk_exposure": {
            "gross_exposure_pct": round(notional / capital * 100.0, 2) if capital else 0,
            "top5_exposure_pct": round(sum(notionals[:5]) / capital * 100.0, 2) if capital else 0,
            "biggest_position_pct": round((notionals[0] if notionals else 0) / capital * 100.0, 2) if capital else 0,
            "near_stop_count": near_stop,
            "open_losers_count": losers,
        },
        "live_summary": {
            "open_winners": sum(1 for row in open_pos if num(row.get("return_pct")) > 0),
            "open_losers": losers,
            "unrealized_pnl": round(pnl, 2),
            "pt_hit_rate": round(len(pt_hits) / len(closed) * 100.0, 2) if closed else None,
        },
        "metrics": {
            "signals": len(enriched_signals),
            "pre_gate_signals": len(pre_gate) if isinstance(pre_gate, list) else 0,
            "blocked_signals": len(blocked) if isinstance(blocked, list) else 0,
            "positions": len(open_pos),
            "notional": notional,
            "pnl": pnl,
            "realized_pnl": realized,
            "portfolio_value": capital + pnl + realized,
            "portfolio_value_est": capital + pnl + realized,
            "cash": max(0.0, capital - notional),
        },
    }


UTC = ZoneInfo("UTC")
NY = ZoneInfo("America/New_York")
IST = ZoneInfo("Asia/Kolkata")


def first_existing(paths):
    for path in paths:
        path = Path(path)
        if path.exists():
            return path
    return None


def file_age(path):
    try:
        path = Path(path)
        if path.exists():
            return max(0, int(time.time() - path.stat().st_mtime))
    except Exception:
        pass
    return None


def run_cmd(args, timeout=0.4):
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        text = (out.stdout or out.stderr or "").strip()
        return out.returncode, text
    except Exception as exc:
        return -1, str(exc)


def next_weekday_at(now, hour, minute):
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def next_intraday_at(now, start_h, start_m, end_h, end_m, step_minutes=30):
    day = now
    while day.weekday() >= 5:
        day = (day + timedelta(days=1)).replace(hour=start_h, minute=start_m, second=0, microsecond=0)
    start = day.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
    end = day.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
    if now <= start:
        return start
    if now > end:
        nxt = (day + timedelta(days=1)).replace(hour=start_h, minute=start_m, second=0, microsecond=0)
        while nxt.weekday() >= 5:
            nxt += timedelta(days=1)
        return nxt
    steps = int(((now - start).total_seconds() + step_minutes * 60 - 1) // (step_minutes * 60))
    candidate = start + timedelta(minutes=steps * step_minutes)
    if candidate > end:
        return next_intraday_at(end + timedelta(minutes=1), start_h, start_m, end_h, end_m, step_minutes)
    return candidate


def next_us_intraday_ist_at(now_ist):
    candidates = []
    for offset in range(8):
        day = (now_ist + timedelta(days=offset)).replace(second=0, microsecond=0)
        # US regular session in IST starts on the same Indian calendar day.
        if day.weekday() <= 4:
            start = day.replace(hour=19, minute=0)
            end = day.replace(hour=23, minute=30)
            if now_ist <= end:
                t = max(now_ist, start)
                minutes = ((t - start).total_seconds() + 29 * 60 + 59) // (30 * 60)
                candidate = start + timedelta(minutes=int(minutes) * 30)
                if start <= candidate <= end:
                    candidates.append(candidate)
        # The same US session continues after Indian midnight Tue-Sat.
        if 1 <= day.weekday() <= 5:
            start = day.replace(hour=0, minute=0)
            end = day.replace(hour=1, minute=30)
            if now_ist <= end:
                t = max(now_ist, start)
                minutes = ((t - start).total_seconds() + 29 * 60 + 59) // (30 * 60)
                candidate = start + timedelta(minutes=int(minutes) * 30)
                if start <= candidate <= end:
                    candidates.append(candidate)
    return min(candidates) if candidates else next_intraday_at(now_ist, 19, 0, 23, 30)


def clock_payload():
    now_utc = datetime.now(UTC)
    now_ny = now_utc.astimezone(NY)
    now_ist = now_utc.astimezone(IST)

    def is_open(now, start_h, start_m, end_h, end_m):
        if now.weekday() >= 5:
            return False
        start = now.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
        end = now.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
        return start <= now < end

    us_trade = next_weekday_at(now_ny, 20, 45)
    us_screen = next_weekday_at(now_ny, 20, 30)
    us_refresh = next_weekday_at(now_ny, 20, 0)
    us_intraday = next_us_intraday_ist_at(now_ist)
    nse_intraday = next_intraday_at(now_ist, 9, 15, 15, 30)
    nse_daily = next_weekday_at(now_ist, 16, 0)
    return {
        "now_ny": now_ny.strftime("%Y-%m-%d %H:%M %Z"),
        "now_ist": now_ist.strftime("%Y-%m-%d %H:%M %Z"),
        "us_market_open": is_open(now_ny, 9, 30, 16, 0),
        "nse_market_open": is_open(now_ist, 9, 15, 15, 30),
        "next_us_refresh": us_refresh.strftime("%a %H:%M %Z"),
        "next_us_screen": us_screen.strftime("%a %H:%M %Z"),
        "next_us_trade": us_trade.strftime("%a %H:%M %Z"),
        "next_us_intraday": us_intraday.strftime("%a %H:%M %Z"),
        "next_nse_intraday": nse_intraday.strftime("%a %H:%M %Z"),
        "next_nse_daily": nse_daily.strftime("%a %H:%M %Z"),
    }


def engine_status_payload():
    cron_code, cron_text = run_cmd(["systemctl", "is-active", "cron"])

    def proc(pattern):
        code, text = run_cmd(["pgrep", "-af", pattern])
        lines = [line for line in text.splitlines() if line.strip()]
        return {"running": code == 0 and bool(lines), "matches": lines[:4]}

    logs = {
        "us_refresh": BASE / "project/logs/daily_refresh.log",
        "us_trade": BASE / "project/logs/paper_execute.log",
        "us_screen": BASE / "project/logs/daily_signals.log",
        "us_intraday": BASE / "project/logs/us_intraday_mtm.log",
        "nse_intraday": BASE / "logs/nse_intraday_mtm.log",
        "nse_daily": BASE / "logs/nse_daily.log",
        "dashboard": BASE / "logs/dashboard_ultra.log",
    }
    return {
        "cron_active": cron_code == 0 and cron_text.strip() == "active",
        "cron_status": cron_text.strip() or "unavailable",
        "processes": {
            "us_trade": proc("fixed_return_paper_execute.py"),
            "us_screen": proc("fixed_return_daily_signals.py"),
            "us_intraday": proc("intraday_mark_to_market.py"),
            "nse_intraday": proc("run_nse_intraday_mtm.sh"),
            "dashboard": proc("dashboard_ultra.py"),
            "nse_daily": proc("run_nse_daily.sh"),
        },
        "log_age_seconds": {key: file_age(path) for key, path in logs.items()},
    }


def tail_lines(path, limit=3):
    try:
        path = Path(path)
        if not path.exists():
            return []
        lines = [line.strip() for line in path.read_text(errors="ignore").splitlines() if line.strip()]
        return lines[-limit:]
    except Exception:
        return []


def action_log_payload():
    items = []
    for label, path in [
        ("US refresh", BASE / "project/logs/daily_refresh.log"),
        ("US signals", BASE / "project/logs/daily_signals.log"),
        ("US execute", BASE / "project/logs/paper_execute.log"),
        ("US intraday", BASE / "project/logs/us_intraday_mtm.log"),
        ("NSE intraday", BASE / "logs/nse_intraday_mtm.log"),
        ("NSE daily", BASE / "logs/nse_daily.log"),
        ("Dashboard", BASE / "logs/dashboard_ultra.log"),
    ]:
        age = file_age(path)
        for line in tail_lines(path, 2):
            items.append({"source": label, "age_seconds": age, "text": line[-170:]})
    for folder in NSE_DIRS:
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:4]:
            items.append({"source": "NSE runtime", "age_seconds": file_age(path), "text": f"updated {path.name}"})
        break
    return items[:12]


def filter_pipeline_payload():
    sig = signal_payload()
    us_positions = open_positions()
    nse = nse_payload()
    gate = nse.get("gate_decision") or {}
    nse_in = int(gate.get("signals_in") or nse.get("metrics", {}).get("pre_gate_signals") or 0)
    nse_out = int(gate.get("signals_out") or nse_in)
    nse_post = int(nse.get("metrics", {}).get("signals") or 0)
    return {
        "us": {
            "raw_signals": len(sig.get("signals", [])),
            "open_positions": len(us_positions),
            "capacity_left": max(0, 50 - len(us_positions)),
            "note": "US dashboard source exposes accepted candidates and open book.",
        },
        "nse": {
            "pre_gate": nse_in,
            "gate_blocked": int(gate.get("blocked_signals") or max(0, nse_in - nse_out)),
            "post_gate": nse_out,
            "execution_filtered": max(0, nse_out - nse_post),
            "post_filter": nse_post,
            "blocked_preview": [
                str(row.get("symbol") or row.get("raw_symbol") or "") for row in (nse.get("blocked_signals") or [])[:8]
            ],
        },
    }


def risk_payload():
    positions = open_positions()
    nse = nse_payload()
    unified = read_json(UNIFIED_RISK, {})
    us_exposures = sorted([num(row.get("position_pct")) for row in positions], reverse=True)
    nse_exposures = sorted([num(row.get("current_value_inr") or row.get("notional_inr")) for row in nse.get("positions", [])], reverse=True)
    nse_notional = sum(nse_exposures)
    us_unrealized = sum(num(row.get("unrealized_pnl")) for row in positions)
    return {
        "us": {
            "gross_exposure_pct": round(sum(us_exposures), 2),
            "top5_exposure_pct": round(sum(us_exposures[:5]), 2),
            "biggest_position_pct": round(us_exposures[0], 2) if us_exposures else 0,
            "unrealized_pnl": round(us_unrealized, 2),
            "losers": len([row for row in positions if num(row.get("unrealized_pnl")) < 0]),
            "near_stop": len([row for row in positions if num(row.get("stop_loss_price")) > 0 and num(row.get("current_price")) <= num(row.get("stop_loss_price")) * 1.015]),
            "sl_grace": len([row for row in positions if row.get("sl_grace")]),
            "unified_state": unified.get("state") if isinstance(unified, dict) else None,
            "unified_messages": unified.get("messages", []) if isinstance(unified, dict) else [],
        },
        "nse": {
            "gross_notional_inr": round(nse_notional, 2),
            "gross_exposure_pct": round(nse_notional / 1_000_000.0 * 100.0, 2),
            "top5_exposure_pct": round(sum(nse_exposures[:5]) / 1_000_000.0 * 100.0, 2),
            "biggest_position_pct": round((nse_exposures[0] / 1_000_000.0 * 100.0) if nse_exposures else 0, 2),
            "cash_inr": round(num(nse.get("metrics", {}).get("cash")), 2),
        },
    }


def alerts_payload():
    positions = open_positions()
    alerts = []
    for row in positions:
        sym = row.get("symbol")
        current = num(row.get("current_price"))
        stop = num(row.get("stop_loss_price"))
        target = num(row.get("profit_target_price"))
        ret = num(row.get("unrealized_pnl_pct"))
        grace = row.get("sl_grace") if isinstance(row.get("sl_grace"), dict) else None
        if grace:
            verdict = grace.get("last_verdict") or "verifying"
            reason = grace.get("llm_reason") or grace.get("key_signal") or "below soft stop"
            alerts.append({"scope": "US", "level": "warn", "text": f"{sym} in SL verification: {verdict} - {reason}"})
        if stop and current <= stop * 1.015:
            alerts.append({"scope": "US", "level": "danger", "text": f"{sym} is within 1.5% of stop"})
        if target and current >= target * 0.985:
            alerts.append({"scope": "US", "level": "good", "text": f"{sym} is within 1.5% of target"})
        if ret <= -7:
            alerts.append({"scope": "US", "level": "danger", "text": f"{sym} drawdown {ret:.2f}%"})
        if num(row.get("days_held")) >= 7:
            alerts.append({"scope": "US", "level": "warn", "text": f"{sym} has held {row.get('days_held')} business days"})
    nse = nse_payload()
    gate = nse.get("gate_decision") or {}
    if gate.get("gate_fired"):
        alerts.append({"scope": "NSE", "level": "warn", "text": f"Gate fired: {gate.get('gate_reason') or 'toxic_narrow'}"})
    if num(nse.get("metrics", {}).get("blocked_signals")):
        alerts.append({"scope": "NSE", "level": "warn", "text": f"{int(num(nse.get('metrics', {}).get('blocked_signals')))} NSE signals blocked"})
    for row in nse.get("positions", [])[:20]:
        for flag in row.get("alert_flags", [])[:3]:
            level = "danger" if "stop" in flag or "drawdown" in flag else "good" if "target" in flag else "warn"
            alerts.append({"scope": "NSE", "level": level, "text": f"{row.get('symbol')} {flag}"})
    if nse.get("asof_date"):
        try:
            asof = datetime.strptime(nse["asof_date"], "%Y-%m-%d").date()
            today_ist = datetime.now(IST).date()
            if today_ist.weekday() < 5 and asof < today_ist:
                alerts.append({"scope": "NSE", "level": "warn", "text": f"NSE signal is still {nse['asof_date']}"})
        except Exception:
            pass
    return alerts[:16]


def reality_payload():
    p = portfolio()
    a = analytics()
    return {
        "us": {
            "live_win_rate": a.get("win_rate"),
            "backtest_win_rate": 60.6,
            "live_closed_pnl": a.get("closed_pnl"),
            "pt_hit_rate": a.get("profit_target_hit_rate"),
            "backtest_return": 52.9,
            "backtest_drawdown": 4.82,
            "sample_trades": a.get("closed_trades"),
            "open_positions": p.get("open_positions_count"),
        },
        "nse": {
            "asof_date": nse_payload().get("asof_date"),
            "shadow_gate_mode": "logging_only",
            "gate_policy": "live allocator unchanged",
        },
    }


def nse_shadow_payload():
    nse = nse_payload()
    gate = nse.get("gate_decision") or {}
    metrics = nse.get("metrics") or {}
    return {
        "state": gate.get("regime_state"),
        "gate_fired": bool(gate.get("gate_fired")),
        "reason": gate.get("gate_reason"),
        "pre_gate": metrics.get("pre_gate_signals"),
        "blocked": metrics.get("blocked_signals"),
        "post_filter": metrics.get("signals"),
        "narrow_score": gate.get("narrow_score"),
        "vix": gate.get("vix") or nse.get("india_vix"),
        "nifty_ret60": gate.get("nifty_ret60"),
        "median_rel60": gate.get("median_rel60"),
        "pct_outperform_60": gate.get("pct_outperform_60"),
    }


def operator_payload():
    clock = clock_payload()
    engine = engine_status_payload()
    filters = filter_pipeline_payload()
    risk = risk_payload()
    alerts = alerts_payload()
    p = portfolio()
    nse = nse_payload()
    health = read_json(MODEL_HEALTH, {})
    sl_log = read_json(SL_DECISIONS, [])
    sl_count = len(sl_log if isinstance(sl_log, list) else sl_log.get("decisions", [])) if sl_log else 0
    summary = [
        "US cron active" if engine["cron_active"] else "US cron not confirmed",
        f"US next MTM {clock['next_us_intraday']}",
        f"US next execute {clock['next_us_trade']}",
        f"NSE next MTM {clock['next_nse_intraday']}",
        f"NSE next daily {clock['next_nse_daily']}",
        f"US open positions {p.get('open_positions_count', 0)}",
        f"NSE open positions {nse.get('metrics', {}).get('positions', 0)}",
        f"model health {health.get('status', 'pending') if isinstance(health, dict) else 'pending'}",
        f"SL decisions {sl_count}",
        f"alerts {len(alerts)}",
    ]
    return {
        "clock": clock,
        "engine": engine,
        "actions": action_log_payload(),
        "filters": filters,
        "risk": risk,
        "alerts": alerts,
        "reality": reality_payload(),
        "model_health": health,
        "nse_shadow": nse_shadow_payload(),
        "summary": summary,
        "refresh": {"default_seconds": 30, "server_time": datetime.now(UTC).isoformat()},
    }


@app.after_request
def no_cache(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response




# DASHBOARD_INSTITUTIONAL_SNAPSHOT_OVERRIDE
def _dashboard_account_override():
    return read_json([BASE / "reports/dashboard_account_override.json"], {})

def _apply_dashboard_override(payload):
    override = _dashboard_account_override()
    if not isinstance(override, dict) or not override:
        return payload

    closed = round(num(override.get("closed_pnl") or override.get("closed_pnl_total")), 2)
    portfolio = payload.setdefault("portfolio", {})
    analytics = payload.setdefault("analytics", {})
    initial = num(portfolio.get("initial_capital"), INITIAL) or INITIAL

    analytics["closed_pnl"] = closed
    analytics["closed_pnl_total"] = closed

    portfolio["cash"] = round(initial + closed, 2)
    portfolio["realized_pnl"] = closed
    if "portfolio_value" in override:
        portfolio["portfolio_value"] = round(num(override.get("portfolio_value")), 2)
    else:
        old_value = num(portfolio.get("portfolio_value"), initial)
        old_realized = num(portfolio.get("realized_pnl"), 0.0)
        portfolio["portfolio_value"] = round(old_value - old_realized + closed, 2)
    portfolio["total_return_pct"] = round((portfolio["portfolio_value"] / initial - 1.0) * 100.0, 3)

    return payload

@app.route("/api/snapshot")
def api_snapshot():
    port = portfolio(use_live=True)
    return jsonify(
        {
            "portfolio": port,
            "signals": signal_payload(),
            "regime": volatility_regime(),
            "analytics": analytics(),
            "unrealized_attribution": us_unrealized_attribution(port.get("positions", [])),
            "status": {"uptime_seconds": int(time.time() - START)},
        }
    )


@app.route("/api/pnl")
def api_pnl():
    return jsonify(pnl_series())


@app.route("/api/nse")
def api_nse():
    return jsonify(nse_payload())


@app.route("/api/operator")
def api_operator():
    return jsonify(operator_payload())


@app.route("/api/live_prices")
def api_live_prices():
    from datetime import timezone
    raw = open_positions(); pos_list = raw.get("positions", []) if isinstance(raw, dict) else (raw or [])
    pos_list = live_mark_positions(pos_list)
    result = []
    for p in pos_list:
        sym = p["symbol"]
        live_px = p.get("current_price", p.get("entry_price", 0))
        entry = num(p.get("entry_price", 0))
        qty = num(p.get("quantity", 0))
        display_pos_pct = num(p.get("position_pct", 0))
        pos_fraction = display_pos_pct / 100.0
        ret = (live_px / entry - 1.0) if entry > 0 else 0.0
        pnl = round((live_px - entry) * qty if qty else INITIAL * pos_fraction * ret, 2)
        pnl_pct = round(ret * 100.0, 3)
        result.append({"symbol": sym, "live_price": round(float(live_px), 4),
                        "entry_price": entry, "quantity": qty,
                        "position_pct": round(display_pos_pct, 4),
                        "position_fraction": round(pos_fraction, 8),
                        "pnl": pnl, "pnl_pct": pnl_pct,
                        "alpaca_ok": p.get("live_price_ok", False),
                        "source": p.get("live_price_source", ""),
                        "stale_seconds": p.get("live_price_stale_seconds"),
                        "quote_spread_pct": p.get("quote_spread_pct"),
                        "timestamp": p.get("live_price_timestamp", "")})
    alpaca_port = None
    try:
        ak = os.getenv("ALPACA_API_KEY", "")
        sk = os.getenv("ALPACA_SECRET_KEY", "")
        if ak and sk:
            from alpaca.trading.client import TradingClient
            tc = TradingClient(api_key=ak, secret_key=sk,
                               paper=os.getenv("ALPACA_PAPER","true").lower()=="true")
            alpaca_port = round(float(tc.get_account().portfolio_value), 2)
    except Exception:
        pass
    return jsonify({"positions": result, "alpaca_portfolio": alpaca_port,
                    "timestamp": datetime.now(timezone.utc).isoformat()})


@app.route("/api/run_signals", methods=["POST"])
def api_run_signals():
    from datetime import timezone
    import subprocess as _sp, sys as _sys
    script = ROOT / "scripts" / "fixed_return_daily_signals.py"
    if not script.exists():
        return jsonify({"ok": False, "error": "script not found"}), 404
    try:
        proc = _sp.Popen([_sys.executable, str(script)], cwd=str(ROOT),
                         stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True)
        return jsonify({"ok": True, "pid": proc.pid,
                        "timestamp": datetime.now(timezone.utc).isoformat()})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/symbol_diagnostic", methods=["POST"])
def api_symbol_diagnostic():
    data = request.get_json(silent=True) or {}
    symbol = str(data.get("symbol") or "").strip().upper()
    if not symbol:
        return jsonify({"ok": False, "error": "symbol required"}), 400
    try:
        scripts_dir = str(ROOT / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from single_symbol_diagnostic import diagnose_symbol

        payload = diagnose_symbol(
            symbol,
            refresh=bool(data.get("refresh", True)),
            run_llm_filter=bool(data.get("run_llm", True)),
            history_days=int(data.get("history_days") or 950),
            force_llm=bool(data.get("force_llm", False)),
        )
        return jsonify(payload), 200 if payload.get("ok") else 400
    except Exception as exc:
        return jsonify({"ok": False, "symbol": symbol, "error": str(exc)}), 500


@app.route("/api/symbol_diagnostic_examples")
def api_symbol_diagnostic_examples():
    try:
        limit = int(request.args.get("limit") or 10)
        return jsonify(diagnostic_examples(limit=limit))
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


HTML = r"""
<!doctype html>
<html data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MacroIntel Institutional</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#edf1f5;--top:#f8fafc;--ink:#17212f;--muted:#5d6f84;--line:#cdd7e3;--soft:#e6edf4;--panel:#f9fbfd;--panel2:#f1f5f9;--hover:#eef4fb;--green:#087f5b;--red:#c92a2a;--blue:#1d5ed7;--amber:#a96808;--shadow:0 10px 24px rgba(31,45,61,.07);--grid:#d9e2ec}
:root[data-theme="dark"]{--bg:#090d13;--top:#080c12;--ink:#edf4fb;--muted:#91a5bd;--line:#223044;--soft:#172232;--panel:#0d131c;--panel2:#111a26;--hover:#131e2d;--green:#00d98b;--red:#ff4a63;--blue:#70a4ff;--amber:#f4b23d;--shadow:0 14px 34px rgba(0,0,0,.28);--grid:#223044}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,Segoe UI,Arial,sans-serif;font-size:13px;transition:background .22s ease,color .22s ease}.top{height:58px;display:flex;align-items:center;gap:14px;padding:0 22px;background:var(--top);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5;box-shadow:0 1px 0 rgba(255,255,255,.04)}.dot{width:9px;height:9px;border-radius:50%;background:var(--green);box-shadow:0 0 14px rgba(0,217,139,.55)}.brand{font-weight:850;font-size:18px}.chip,.themeBtn{font:12px ui-monospace,SFMono-Regular,Consolas,monospace;border:1px solid var(--line);border-radius:999px;padding:7px 11px;background:var(--panel);color:var(--muted)}.themeBtn{cursor:pointer;color:var(--ink);transition:transform .18s ease,border-color .18s ease,background .18s ease}.themeBtn:hover{transform:translateY(-1px);border-color:var(--blue)}.page{max-width:1900px;margin:0 auto;padding:16px 18px 24px}.tabs{display:flex;gap:8px;margin-bottom:14px}.tab{border:1px solid var(--line);background:var(--panel);border-radius:999px;padding:8px 14px;font-weight:750;color:var(--muted);cursor:pointer;transition:background .18s ease,color .18s ease,transform .18s ease}.tab:hover{transform:translateY(-1px);color:var(--ink)}.tab.active{background:var(--ink);color:var(--bg);border-color:var(--ink)}.view{display:none;animation:fadeIn .18s ease}.view.active{display:block}@keyframes fadeIn{from{opacity:.72;transform:translateY(3px)}to{opacity:1;transform:none}}
.kpis{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:12px;margin-bottom:12px}.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px;box-shadow:var(--shadow);transition:background .22s ease,border-color .22s ease,transform .18s ease,box-shadow .18s ease}.card:hover{border-color:color-mix(in srgb,var(--blue) 35%,var(--line));transform:translateY(-1px)}.kpi{min-height:88px}.label{font-size:10px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);font-weight:800}.value{font:800 24px ui-monospace,SFMono-Regular,Consolas,monospace;margin-top:8px}.sub{font-size:12px;color:var(--muted);margin-top:6px}.green{color:var(--green)!important}.red{color:var(--red)!important}.blue{color:var(--blue)!important}.amber{color:var(--amber)!important}
.grid{display:grid;gap:12px}.mainGrid{grid-template-columns:1.45fr .95fr}.workGrid{grid-template-columns:minmax(0,1.3fr) minmax(410px,.7fr)}.chartBox{height:235px}.titleRow{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}.title{font-size:12px;text-transform:uppercase;letter-spacing:.12em;color:var(--ink);font-weight:850}.badge{font:12px ui-monospace,SFMono-Regular,Consolas,monospace;color:var(--blue);background:var(--panel2);border:1px solid var(--line);border-radius:999px;padding:5px 10px}
table{width:100%;border-collapse:collapse}th{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.11em;text-align:left;padding:9px 10px;border-bottom:1px solid var(--line);background:var(--panel2)}td{font:13px ui-monospace,SFMono-Regular,Consolas,monospace;padding:9px 10px;border-bottom:1px solid var(--soft);white-space:nowrap}tr:hover td{background:var(--hover)}.bar{height:7px;border-radius:99px;background:var(--soft);overflow:hidden}.fill{height:100%;background:var(--blue);transition:width .22s ease}.rail{display:grid;gap:12px;align-content:start}.metricLine{display:flex;justify-content:space-between;align-items:baseline;padding:9px 0;border-bottom:1px solid var(--soft)}.metricLine:last-child{border-bottom:0}.big{font:850 44px ui-monospace,SFMono-Regular,Consolas,monospace}.nseGrid{grid-template-columns:repeat(5,minmax(0,1fr))}.empty{text-align:center;color:var(--muted);padding:28px!important}
.nseGrid{grid-template-columns:repeat(6,minmax(0,1fr))}.nseHero{display:grid;grid-template-columns:1.15fr .85fr;gap:12px;margin-bottom:12px}.nseChartGrid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.nseBlotter{display:grid;grid-template-columns:1.05fr .95fr;gap:12px}.moneyRows{display:grid;gap:10px}.moneyRow{display:grid;grid-template-columns:118px 1fr 110px;gap:10px;align-items:center}.moneyRow b{font:700 12px ui-monospace,SFMono-Regular,Consolas,monospace;color:var(--muted)}.moneyRow span{font:750 12px ui-monospace,SFMono-Regular,Consolas,monospace;text-align:right}.stateBox{display:grid;grid-template-columns:1fr 1fr;gap:10px}.stateCell{background:var(--panel2);border:1px solid var(--soft);border-radius:8px;padding:10px}.stateCell strong{display:block;font:800 19px ui-monospace,SFMono-Regular,Consolas,monospace;margin-top:5px}.chartMini{height:210px}.chartTiny{height:180px}
.opsGrid{display:grid;grid-template-columns:1.15fr .85fr .85fr;gap:12px;margin-bottom:12px}.opsWide{grid-column:span 1}.miniGrid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px}.pillGrid{display:flex;gap:8px;flex-wrap:wrap}.miniCell{background:var(--panel2);border:1px solid var(--soft);border-radius:8px;padding:10px}.miniCell strong{display:block;font:800 16px ui-monospace,SFMono-Regular,Consolas,monospace;margin-top:4px}.statusDot{display:inline-block;width:7px;height:7px;border-radius:99px;background:var(--muted);margin-right:6px}.statusDot.on{background:var(--green);box-shadow:0 0 10px color-mix(in srgb,var(--green) 55%,transparent)}.statusDot.warn{background:var(--amber)}.statusDot.bad{background:var(--red)}.timeline{display:grid;gap:8px;max-height:180px;overflow:hidden}.event{display:grid;grid-template-columns:78px 1fr;gap:8px;border-bottom:1px solid var(--soft);padding-bottom:7px}.event:last-child{border-bottom:0}.event b{font:800 10px ui-monospace,SFMono-Regular,Consolas,monospace;color:var(--blue);text-transform:uppercase}.event span{color:var(--muted);line-height:1.35}.controlRow{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-top:10px}.softBtn{border:1px solid var(--line);background:var(--panel2);color:var(--ink);border-radius:8px;padding:8px 10px;font-weight:800;cursor:pointer}.softBtn:hover{border-color:var(--blue)}.diagnosticGrid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:12px 0}.alertList{display:grid;gap:8px}.alertItem{border-left:3px solid var(--blue);background:var(--panel2);border-radius:7px;padding:9px 10px;color:var(--ink)}.alertItem.danger{border-left-color:var(--red)}.alertItem.warn{border-left-color:var(--amber)}.alertItem.good{border-left-color:var(--green)}.stackRows{display:grid;gap:7px}.stackRow{display:flex;align-items:center;justify-content:space-between;gap:10px;border-bottom:1px solid var(--soft);padding:7px 0}.stackRow:last-child{border-bottom:0}.stackRow span{color:var(--muted)}.stackRow b{font:800 12px ui-monospace,SFMono-Regular,Consolas,monospace}
@media(max-width:1200px){.kpis{grid-template-columns:repeat(3,minmax(0,1fr))}.mainGrid,.workGrid,.nseHero,.nseChartGrid,.nseBlotter,.opsGrid,.diagnosticGrid{grid-template-columns:1fr}.nseGrid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:720px){.kpis,.nseGrid,.stateBox,.miniGrid{grid-template-columns:1fr}.top{height:auto;flex-wrap:wrap;padding:12px}.page{padding:12px}td,th{padding:8px 7px}.value{font-size:22px}.moneyRow{grid-template-columns:1fr}}
</style>
<style>
:root{--accent:#087f5b;--accent2:#1d5ed7;--accent-soft:rgba(8,127,91,.13)}
:root[data-theme=dark]{--accent:#00d98b;--accent2:#70a4ff;--accent-soft:rgba(0,217,139,.13)}
:root[data-accent=emerald]{--accent:#00d98b;--accent2:#70a4ff;--accent-soft:rgba(0,217,139,.13);--green:#00d98b;--blue:#70a4ff}
:root[data-accent=blue]{--accent:#3b82f6;--accent2:#22d3ee;--accent-soft:rgba(59,130,246,.14);--green:#3b82f6;--blue:#22d3ee}
:root[data-accent=purple]{--accent:#8b5cf6;--accent2:#a78bfa;--accent-soft:rgba(139,92,246,.15);--green:#8b5cf6;--blue:#a78bfa}
:root[data-accent=pink]{--accent:#ec4899;--accent2:#f472b6;--accent-soft:rgba(236,72,153,.15);--green:#ec4899;--blue:#f472b6}
:root[data-accent=red]{--accent:#ef4444;--accent2:#fb7185;--accent-soft:rgba(239,68,68,.15);--green:#ef4444;--blue:#fb7185}
:root[data-accent=amber]{--accent:#f59e0b;--accent2:#fbbf24;--accent-soft:rgba(245,158,11,.16);--green:#f59e0b;--blue:#fbbf24}
.dot{background:var(--accent);box-shadow:0 0 18px color-mix(in srgb,var(--accent) 65%,transparent)}
.top{border-bottom-color:color-mix(in srgb,var(--accent) 24%,var(--line))}
.card{background:linear-gradient(180deg,color-mix(in srgb,var(--accent) 4%,var(--panel)),var(--panel));border-color:color-mix(in srgb,var(--accent) 18%,var(--line))}
.card:hover{border-color:color-mix(in srgb,var(--accent) 52%,var(--line));box-shadow:0 12px 32px color-mix(in srgb,var(--accent) 12%,transparent)}
.chip,.themeBtn,.badge{border-color:color-mix(in srgb,var(--accent) 26%,var(--line));background:color-mix(in srgb,var(--accent) 7%,var(--panel));color:color-mix(in srgb,var(--accent2) 82%,var(--ink))}
.tab.active{background:var(--accent);color:#061014;border-color:var(--accent);box-shadow:0 8px 20px color-mix(in srgb,var(--accent) 24%,transparent)}
.tab:hover,.themeBtn:hover,.softBtn:hover{border-color:color-mix(in srgb,var(--accent) 55%,var(--line));color:var(--accent2)}
.fill{background:linear-gradient(90deg,var(--accent),var(--accent2))}
th{background:color-mix(in srgb,var(--accent) 8%,var(--panel2));color:color-mix(in srgb,var(--accent2) 65%,var(--muted))}
tr:hover td{background:color-mix(in srgb,var(--accent) 7%,var(--hover))}
.stateCell,.miniCell,.alertItem{background:color-mix(in srgb,var(--accent) 6%,var(--panel2));border-color:color-mix(in srgb,var(--accent) 16%,var(--soft))}
.metricLine,.stackRow,td{border-bottom-color:color-mix(in srgb,var(--accent) 13%,var(--soft))}
.bar{background:color-mix(in srgb,var(--accent) 10%,var(--soft))}
.palette{display:flex;align-items:center;gap:7px}
.swatch{width:20px;height:20px;border-radius:999px;border:1px solid var(--line);cursor:pointer;background:var(--sw);box-shadow:inset 0 0 0 2px rgba(255,255,255,.10);transition:transform .15s ease,border-color .15s ease,box-shadow .15s ease}
.swatch:hover{transform:translateY(-1px)}
.swatch.active{border-color:var(--ink);box-shadow:0 0 0 3px color-mix(in srgb,var(--sw) 28%,transparent),0 0 18px color-mix(in srgb,var(--sw) 34%,transparent)}
.symbolSearch{display:grid;gap:10px}.symbolSearch.collapsed{padding-bottom:10px}.symbolSearch.collapsed .titleRow{margin-bottom:0}.symbolSearch.collapsed .diagBody{display:none}.titleActions{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.collapseBtn{width:34px;height:30px;padding:0;text-align:center;font:900 15px ui-monospace,SFMono-Regular,Consolas,monospace}.searchRow{display:grid;grid-template-columns:minmax(120px,220px) auto auto auto 1fr;gap:8px;align-items:center}.searchInput{height:36px;border:1px solid var(--line);border-radius:8px;background:var(--panel2);color:var(--ink);font:800 14px ui-monospace,SFMono-Regular,Consolas,monospace;padding:0 11px;text-transform:uppercase}.checkLabel{display:flex;align-items:center;gap:7px;color:var(--muted);font-weight:750}.exampleGrid{display:flex;gap:8px;flex-wrap:wrap}.exampleBtn{border:1px solid var(--soft);background:var(--panel2);color:var(--ink);border-radius:999px;padding:6px 9px;font:800 11px ui-monospace,SFMono-Regular,Consolas,monospace;cursor:pointer}.exampleBtn:hover{border-color:var(--accent2)}.diagHero{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px}.diagCell{background:color-mix(in srgb,var(--accent) 6%,var(--panel2));border:1px solid color-mix(in srgb,var(--accent) 16%,var(--soft));border-radius:8px;padding:10px}.diagCell strong{display:block;font:800 18px ui-monospace,SFMono-Regular,Consolas,monospace;margin-top:5px}.gateGrid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:7px}.gatePill{border:1px solid var(--soft);border-radius:8px;padding:8px;background:var(--panel2)}.gatePill b{display:block;font:800 11px ui-monospace,SFMono-Regular,Consolas,monospace}.gatePill span{display:block;color:var(--muted);font-size:11px;margin-top:4px;overflow:hidden;text-overflow:ellipsis}.diagText{line-height:1.45;color:var(--muted);word-break:break-word}.diagJson{max-height:360px;overflow:auto;white-space:pre-wrap;border:1px solid var(--soft);border-radius:8px;background:var(--panel2);padding:10px;color:var(--ink);font:11px ui-monospace,SFMono-Regular,Consolas,monospace}
@media(max-width:1200px){.diagHero,.gateGrid{grid-template-columns:repeat(2,minmax(0,1fr))}.searchRow{grid-template-columns:1fr auto auto auto}}@media(max-width:720px){.diagHero,.gateGrid,.searchRow{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="top"><div class="dot"></div><div class="brand">MacroIntel Institutional</div><div class="chip">Paper Trading</div><div class="chip" id="volChip">VOL --</div><button class="themeBtn" id="themeToggle" type="button">Dark</button><div class="palette" id="accentPalette"><button class="swatch" data-accent="emerald" title="Emerald" style="--sw:#00d98b" type="button"></button><button class="swatch" data-accent="blue" title="Blue" style="--sw:#3b82f6" type="button"></button><button class="swatch" data-accent="purple" title="Purple" style="--sw:#8b5cf6" type="button"></button><button class="swatch" data-accent="pink" title="Pink" style="--sw:#ec4899" type="button"></button><button class="swatch" data-accent="red" title="Red" style="--sw:#ef4444" type="button"></button><button class="swatch" data-accent="amber" title="Amber" style="--sw:#f59e0b" type="button"></button></div><div style="flex:1"></div><div class="chip" id="updated">loading</div></div>
<div class="page">
<div class="tabs"><button class="tab active" data-view="us">US Fixed Return</button><button class="tab" data-view="nse">India NSE</button></div>
<section class="opsGrid">
<div class="card"><div class="titleRow"><div><div class="title">Operator Summary</div><div class="sub">what matters right now</div></div><div class="badge" id="refreshState">auto 30s</div></div><div class="pillGrid" id="operatorSummary"></div><div class="controlRow"><button class="softBtn" id="refreshNow" type="button">Refresh now</button><button class="softBtn" id="autoRefresh" type="button">Auto refresh on</button><span class="sub" id="lastRefresh">not loaded</span></div></div>
<div class="card"><div class="titleRow"><div><div class="title">Market Clock</div><div class="sub">US/NSE sessions and next fire</div></div></div><div class="miniGrid" id="marketClock"></div></div>
<div class="card"><div class="titleRow"><div><div class="title">Trade Engine Status</div><div class="sub">cron, launchers, dashboard process</div></div></div><div class="stackRows" id="engineStatus"></div></div>
</section>
<section class="card" style="margin-bottom:12px"><div class="titleRow"><div><div class="title">Today's Action Log</div><div class="sub">latest launcher, NSE and dashboard events</div></div><div class="badge" id="actionCount">0 events</div></div><div class="timeline" id="actionLog"></div></section>

<section id="us" class="view active">
<div class="kpis">
<div class="card kpi"><div class="label">Portfolio Value</div><div class="value" id="pv">$0</div><div class="sub" id="ret">--</div></div>
<div class="card kpi"><div class="label">Cash</div><div class="value" id="cash">$0</div><div class="sub">paper broker cash</div></div>
<div class="card kpi"><div class="label">Realized P&L</div><div class="value" id="cpnl">$0</div><div class="sub">closed trades</div></div>
<div class="card kpi"><div class="label">Open Positions</div><div class="value blue" id="oc">0</div><div class="sub">live book</div></div>
<div class="card kpi"><div class="label">Win Rate</div><div class="value" id="wr">--</div><div class="sub" id="tc">backtest 60.6%</div></div>
<div class="card kpi"><div class="label">Drawdown</div><div class="value" id="dd">0%</div><div class="sub">from peak</div></div>
</div>
<div class="card symbolSearch" id="diagCard" style="margin-bottom:12px"><div class="titleRow"><div><div class="title">Stock Diagnostic</div><div class="sub">single-symbol model, gates and LLM judgment</div></div><div class="titleActions"><div class="badge" id="diagBadge">idle</div><button class="softBtn collapseBtn" id="diagToggle" type="button" onclick="toggleSymbolDiagnostic()" title="Minimize stock diagnostic" aria-label="Minimize stock diagnostic" aria-expanded="true">-</button></div></div><div class="diagBody"><div class="searchRow"><input class="searchInput" id="diagSymbol" placeholder="AAPL" autocomplete="off"><button class="softBtn" id="diagRun" type="button" onclick="runSymbolDiagnostic()">RUN CHECK</button><label class="checkLabel"><input id="diagLlm" type="checkbox" checked> LLM</label><label class="checkLabel"><input id="diagForceLlm" type="checkbox"> Force LLM</label><span class="sub" id="diagStatus" style="display:none"></span></div><div class="exampleGrid" id="diagExamples"></div><div id="diagResult" class="diagText">No symbol checked yet.</div></div></div>
<div class="grid mainGrid" style="margin-bottom:12px">
<div class="card"><div class="titleRow"><div><div class="title">Cumulative P&L</div><div class="sub">realized profit curve</div></div><div class="badge" id="pnlBadge">--</div></div><div class="chartBox"><canvas id="pnlChart"></canvas></div></div>
<div class="card"><div class="titleRow"><div><div class="title">Portfolio Value</div><div class="sub">paper equity path</div></div><div class="badge" id="portBadge">--</div></div><div class="chartBox"><canvas id="portChart"></canvas></div></div>
</div>
<div class="grid workGrid">
<div class="grid">
<div class="card"><div class="titleRow"><div><div class="title">Open Positions</div><div class="sub">entry, target, stop, progress and P&L</div></div><div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap"><span id="livePriceBadge" class="badge">LIVE --</span><div class="badge" id="pb">0 open</div><button class="softBtn" id="runSignalsBtn" type="button" onclick="runSignals()">RUN SIGNALS</button></div></div><table><thead><tr><th>Symbol</th><th>Entry</th><th>Current</th><th>PT</th><th>SL</th><th>Days</th><th>Progress</th><th>P&L</th><th>Return</th><th>Conf</th></tr></thead><tbody id="pos"></tbody></table></div>
<div class="card"><div class="titleRow"><div><div class="title">Unrealized P&L Attribution</div><div class="sub">open positions only: explains intraday MTM before exits</div></div><div class="badge" id="usAttrBadge">--</div></div><table><thead><tr><th>Symbol</th><th>Size</th><th>Entry</th><th>Current</th><th>Move</th><th>Open P&L</th><th>Loss Share</th></tr></thead><tbody id="usAttribution"></tbody></table></div>
<div class="card"><div class="titleRow"><div><div class="title">Closed Trades</div><div class="sub">settled exits</div></div><div class="badge" id="cb">0 closed</div></div><table><thead><tr><th>Date</th><th>Symbol</th><th>Qty</th><th>Entry</th><th>Exit</th><th>Return</th><th>P&L</th><th>Reason</th></tr></thead><tbody id="tr"></tbody></table></div>
</div>
<aside class="rail">
<div class="card"><div class="title">Volatility Regime</div><div class="big blue" id="spy">--%</div><div class="metricLine"><span class="label">State</span><b id="reg">--</b></div><div class="metricLine"><span class="label">Multiplier</span><b id="mul">--x</b></div><div class="sub" id="desc"></div></div>
<div class="card"><div class="titleRow"><div><div class="title">Today's Signals</div><div class="sub">next candidates</div></div><div class="badge" id="sd">--</div></div><table><thead><tr><th>#</th><th>Symbol</th><th>Prob</th><th>Entry</th><th>PT</th><th>SL</th></tr></thead><tbody id="sig"></tbody></table></div>
<div class="card"><div class="titleRow"><div class="title">Live Summary</div><div class="badge" id="outBadge">--</div></div><div id="summaryBox"></div></div>
</aside>
</div>
<div class="diagnosticGrid">
<div class="card"><div class="titleRow"><div><div class="title">Blocked / Filtered Signals</div><div class="sub">US handoff visibility</div></div></div><div class="stackRows" id="usFilter"></div></div>
<div class="card"><div class="titleRow"><div><div class="title">Risk Exposure</div><div class="sub">gross, concentration, stop pressure</div></div></div><div class="stackRows" id="usRisk"></div></div>
<div class="card"><div class="titleRow"><div><div class="title">Position Alert Flags</div><div class="sub">targets, stops, stale holds</div></div><div class="badge" id="usAlertCount">0</div></div><div class="alertList" id="usAlerts"></div></div>
<div class="card"><div class="titleRow"><div><div class="title">Paper vs Backtest Reality</div><div class="sub">live sample against fixed-return reference</div></div></div><div class="stackRows" id="usReality"></div></div>
</div>
</section>

<section id="nse" class="view">
<div class="kpis nseGrid">
<div class="card kpi"><div class="label">As Of</div><div class="value" id="nseDate">--</div><div class="sub">last NSE signal</div></div>
<div class="card kpi"><div class="label">Regime</div><div class="value blue" id="nseRegime">--</div><div class="sub">dual-regime router</div></div>
<div class="card kpi"><div class="label">India VIX</div><div class="value" id="nseVix">--</div><div class="sub">vol state</div></div>
<div class="card kpi"><div class="label">Gate State</div><div class="value" id="nseGate">--</div><div class="sub" id="nseGateSub">shadow logger</div></div>
<div class="card kpi"><div class="label">Open Positions</div><div class="value blue" id="nseOpen">0</div><div class="sub">paper only</div></div>
<div class="card kpi"><div class="label">Unrealized P&L</div><div class="value" id="nsePnl">Rs 0</div><div class="sub">NSE book</div></div>
</div>
<div class="nseHero">
<div class="card"><div class="titleRow"><div><div class="title">NSE Paper Money Tracker</div><div class="sub">Rs 10L paper allocation, cash, notional and mark-to-market</div></div><div class="badge" id="nseMoneyBadge">--</div></div><div class="moneyRows"><div class="moneyRow"><b>Invested</b><div class="bar"><div class="fill" id="nseInvestedBar"></div></div><span id="nseInvested">Rs 0</span></div><div class="moneyRow"><b>Cash</b><div class="bar"><div class="fill" id="nseCashBar"></div></div><span id="nseCash">Rs 0</span></div><div class="moneyRow"><b>P&L</b><div class="bar"><div class="fill" id="nsePnlBar"></div></div><span id="nsePnl2">Rs 0</span></div></div></div>
<div class="card"><div class="titleRow"><div><div class="title">Regime + Gate Decision</div><div class="sub">live router state and shadow evidence</div></div><div class="badge" id="nseShadowBadge">--</div></div><div class="stateBox"><div class="stateCell"><div class="label">State</div><strong id="nseState">--</strong></div><div class="stateCell"><div class="label">Gate Fired</div><strong id="nseGateFired">--</strong></div><div class="stateCell"><div class="label">Pre Gate</div><strong id="nsePreGate">0</strong></div><div class="stateCell"><div class="label">Blocked</div><strong id="nseBlocked">0</strong></div></div></div>
</div>
<div class="nseChartGrid" style="margin-bottom:12px">
<div class="card"><div class="titleRow"><div><div class="title">Signal Confidence</div><div class="sub">post-gate candidate probabilities</div></div><div class="badge" id="nseSignalsBadge">0</div></div><div class="chartMini"><canvas id="nseSignalChart"></canvas></div></div>
<div class="card"><div class="titleRow"><div><div class="title">Position Exposure</div><div class="sub">paper notional by symbol</div></div><div class="badge" id="nsePositionsBadge">0</div></div><div class="chartMini"><canvas id="nseExposureChart"></canvas></div></div>
</div>
<div class="nseChartGrid" style="margin-bottom:12px">
<div class="card"><div class="titleRow"><div><div class="title">Cumulative P&L</div><div class="sub">NSE mark-to-market history</div></div><div class="badge" id="nsePnlCurveBadge">--</div></div><div class="chartTiny"><canvas id="nsePnlCurveChart"></canvas></div></div>
<div class="card"><div class="titleRow"><div><div class="title">Portfolio Value</div><div class="sub">NSE paper equity path</div></div><div class="badge" id="nsePortfolioCurveBadge">--</div></div><div class="chartTiny"><canvas id="nsePortfolioCurveChart"></canvas></div></div>
</div>
<div class="nseBlotter">
<div class="card"><div class="titleRow"><div><div class="title">NSE Signals</div><div class="sub">next candidates with risk bands</div></div><div class="badge" id="nseSignalsBadge2">0</div></div><table><thead><tr><th>Symbol</th><th>Prob</th><th>Entry</th><th>PT</th><th>SL</th></tr></thead><tbody id="nseSignals"></tbody></table></div>
<div class="card"><div class="titleRow"><div><div class="title">NSE Positions</div><div class="sub">allocation, risk bands and alerts</div></div><div class="badge" id="nsePositionsBadge2">0</div></div><table><thead><tr><th>Symbol</th><th>Qty</th><th>Entry</th><th>Current</th><th>PT</th><th>SL</th><th>P&L</th><th>Flags</th></tr></thead><tbody id="nsePositions"></tbody></table></div>
</div>
<div class="card" style="margin:12px 0"><div class="titleRow"><div><div class="title">Unrealized P&L Attribution</div><div class="sub">open positions only: this explains MTM movement before any trade is closed</div></div><div class="badge" id="nseAttrBadge">--</div></div><table><thead><tr><th>Symbol</th><th>Qty</th><th>Entry</th><th>Current</th><th>Move</th><th>Open P&L</th><th>Loss Share</th></tr></thead><tbody id="nseAttribution"></tbody></table></div>
<div class="card" style="margin:12px 0"><div class="titleRow"><div><div class="title">NSE Closed Trades</div><div class="sub">settled NSE exits with reason codes</div></div><div class="badge" id="nseClosedBadge">0 closed</div></div><table><thead><tr><th>Date</th><th>Symbol</th><th>Entry</th><th>Exit</th><th>Return</th><th>P&L</th><th>Reason</th></tr></thead><tbody id="nseClosedTrades"></tbody></table></div>
<div class="diagnosticGrid">
<div class="card"><div class="titleRow"><div><div class="title">NSE Shadow Gate Tracker</div><div class="sub">logging only, allocator unchanged</div></div></div><div class="stackRows" id="nseShadow"></div></div>
<div class="card"><div class="titleRow"><div><div class="title">Blocked / Filtered Signals</div><div class="sub">pre-gate to post-execution stream</div></div></div><div class="stackRows" id="nseFilter"></div></div>
<div class="card"><div class="titleRow"><div><div class="title">Risk Exposure</div><div class="sub">Rs 10L paper book</div></div></div><div class="stackRows" id="nseRisk"></div></div>
<div class="card"><div class="titleRow"><div><div class="title">Position Alert Flags</div><div class="sub">gate, stale signal and blocked stream</div></div><div class="badge" id="nseAlertCount">0</div></div><div class="alertList" id="nseAlerts"></div></div>
<div class="card"><div class="titleRow"><div><div class="title">Live Summary</div><div class="sub">winners, losers and unrealized P&L</div></div></div><div class="stackRows" id="nseLiveSummary"></div></div>
<div class="card"><div class="titleRow"><div><div class="title">Paper vs Backtest Reality</div><div class="sub">NSE closed sample against fold reference</div></div></div><div class="stackRows" id="nseReality"></div></div>
</div>
</section>
</div>
<script>
const $=id=>document.getElementById(id);
const money=n=>{const v=Number(n);return n==null||isNaN(v)?'--':(v<0?'-':'')+'$'+Math.abs(v).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});};
const stopMoney=x=>Number(x&&x.stop_loss_enabled!==false?x.stop_loss_price:0)>0?money(x.stop_loss_price):'OFF';
const pct=n=>n==null||isNaN(n)?'--':(Number(n)>0?'+':'')+Number(n).toFixed(2)+'%';
const inr=n=>'Rs '+Number(n||0).toLocaleString('en-IN',{maximumFractionDigits:0});
function tone(el,n){el.classList.remove('green','red','amber'); if(n>0)el.classList.add('green'); if(n<0)el.classList.add('red')}
function css(name){return getComputedStyle(document.documentElement).getPropertyValue(name).trim()}
function setTheme(mode){document.documentElement.dataset.theme=mode;localStorage.setItem('mi_theme',mode);$('themeToggle').textContent=mode==='dark'?'Light':'Dark'; if(window.__lastSeries)renderCharts(window.__lastSeries); if(window.__lastNse)renderNseCharts(window.__lastNse.signals||[],window.__lastNse.positions||[],window.__lastNse)}
setTheme(localStorage.getItem('mi_theme')||'dark');
$('themeToggle').onclick=()=>setTheme(document.documentElement.dataset.theme==='dark'?'light':'dark');
function setAccent(name){document.documentElement.dataset.accent=name;localStorage.setItem('mi_accent',name);document.querySelectorAll('.swatch').forEach(x=>x.classList.toggle('active',x.dataset.accent===name)); if(window.__lastSeries)renderCharts(window.__lastSeries); if(window.__lastNse)renderNseCharts(window.__lastNse.signals||[],window.__lastNse.positions||[],window.__lastNse)}
document.querySelectorAll('.swatch').forEach(btn=>btn.onclick=()=>setAccent(btn.dataset.accent));
setAccent(localStorage.getItem('mi_accent')||'emerald');
let activeRegion='us';
function updateChromeRegime(){
 if(activeRegion==='nse'&&window.__lastNse){
  const vol=window.__lastNse.vol_state||{};
  $('volChip').textContent='NSE VOL '+(vol.label||'--')+' '+(vol.multiplier||'--')+'x';
  return;
 }
 const r=window.__usRegime||{};
 $('volChip').textContent='US VOL '+(r.vol_regime||'--')+' '+(r.vol_multiplier||'--')+'x';
}
document.querySelectorAll('.tab').forEach(btn=>btn.onclick=()=>{document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));document.querySelectorAll('.view').forEach(x=>x.classList.remove('active'));btn.classList.add('active');$(btn.dataset.view).classList.add('active');activeRegion=btn.dataset.view==='nse'?'nse':'us';updateChromeRegime()});
const age=s=>s==null?'--':s<60?s+'s ago':s<3600?Math.round(s/60)+'m ago':Math.round(s/3600)+'h ago';
function esc(v){return String(v==null?'':v).replace(/[&<>"']/g,function(ch){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]})}
function kv(rows){return rows.map(r=>`<div class=stackRow><span>${r[0]}</span><b class="${r[2]||''}">${r[1]}</b></div>`).join('')}
function dot(on,bad=false){return `<span class="statusDot ${bad?'bad':on?'on':'warn'}"></span>`}
function renderAlerts(id,rows){$(id).innerHTML=rows.length?rows.map(a=>`<div class="alertItem ${a.level||''}"><b>${a.scope}</b> ${a.text}</div>`).join(''):'<div class="alertItem good">No active flags</div>'}
function singleSymbolDiagJson(diag,llm){
 const clean={
  decision:diag.decision||{},
  data:diag.data||{},
  model:diag.model||{},
  risk_plan:diag.risk_plan||{},
  market:diag.market||{},
  gates:diag.gates||[],
  llm:{
   ran:!!(llm&&llm.ran),
   forced:!!(llm&&llm.forced),
   research_only:!!(llm&&llm.research_only),
   status:llm&&llm.status,
   mode:llm&&llm.mode,
   decision:llm&&llm.decision,
   reason:llm&&llm.reason,
   confidence:llm&&llm.confidence,
   duration_seconds:llm&&llm.duration_seconds,
   ignored_failed_gates:llm&&llm.ignored_failed_gates,
   blocking_failed_gates:llm&&llm.blocking_failed_gates
  }
 };
 const snap=(diag.features&&diag.features.snapshot)||{};
 clean.features={snapshot:snap};
 return JSON.stringify(clean,null,2);
}
function setDiagCollapsed(collapsed){
 const card=$('diagCard'), btn=$('diagToggle');
 if(!card||!btn)return;
 card.classList.toggle('collapsed',!!collapsed);
 btn.textContent=collapsed?'+':'-';
 const label=collapsed?'Expand stock diagnostic':'Minimize stock diagnostic';
 btn.title=label; btn.setAttribute('aria-label',label); btn.setAttribute('aria-expanded',collapsed?'false':'true');
 try{localStorage.setItem('macrointel_diag_collapsed',collapsed?'1':'0')}catch(e){}
}
function toggleSymbolDiagnostic(){
 const card=$('diagCard');
 setDiagCollapsed(!(card&&card.classList.contains('collapsed')));
}
function restoreSymbolDiagnosticState(){
 let collapsed=false;
 try{collapsed=localStorage.getItem('macrointel_diag_collapsed')==='1'}catch(e){}
 setDiagCollapsed(collapsed);
}
function renderSymbolDiagnostic(d){
 if(!d||!d.ok){$('diagResult').innerHTML=`<div class="alertItem danger">${esc(d&&d.error?d.error:'diagnostic failed')}</div>`;$('diagBadge').textContent='failed';return}
 const sig=d.signal||{}, rank=d.rank||{}, llm=d.llm||{}, feature=d.feature||{}, source=d.source||{}, fc=d.friend_context||{}, diag=d.diagnostics||{};
 const decision=diag.decision||{}, dataDiag=diag.data||{}, modelDiag=diag.model||{}, daily=diag.daily_context||{}, risk=diag.risk_plan||{}, market=diag.market||{};
 const verdict=d.verdict||'--', ok=!!d.would_trade;
 $('diagBadge').textContent=ok?'allowed':(llm.research_only?'research':'blocked');
 const mlMargin=(modelDiag.ml_margin??fc.ml_margin)==null?'--':(Number(modelDiag.ml_margin??fc.ml_margin)>0?'+':'')+Number(modelDiag.ml_margin??fc.ml_margin).toFixed(3);
 const tech=(diag.features&&diag.features.snapshot)||fc.technical_snapshot||{};
 const techBits=Object.keys(tech).slice(0,20).map(k=>`${k}: ${tech[k]}`).join(' | ')||'technical snapshot unavailable';
 const llmLabel=llm.forced?'Research LLM':llm.ran?'LLM':'LLM';
 const riskPlan=(risk.profit_target_pct??fc.target_pct??'--')+'% PT / '+(risk.stop_loss_pct??fc.stop_pct??'--')+'% SL / '+(risk.hold_days??fc.hold_days??'--')+'d';
 const dataRows=(dataDiag.rows??feature.rows??0)+' / '+(dataDiag.required_rows??feature.required_rows??'--');
 const sourceLabel=source.source==='fetched_from_alpaca'?'ALPACA':'SYSTEM';
 const systemRead=(fc.notes||[]).join(' | ')||decision.top_n_note||fc.pipeline_stage||'--';
 $('diagResult').innerHTML=[
  `<div class=diagHero>`,
  `<div class=diagCell><div class=label>Verdict</div><strong class="${ok?'green':'red'}">${esc(verdict)}</strong></div>`,
  `<div class=diagCell><div class=label>Probability</div><strong class="${Number(sig.probability||0)>=Number((d.model||{}).threshold||0)?'green':'red'}">${Number(sig.probability||0).toFixed(3)}</strong></div>`,
  `<div class=diagCell><div class=label>ML Threshold</div><strong>${Number((d.model||{}).threshold||0).toFixed(3)}</strong></div>`,
  `<div class=diagCell><div class=label>Entry</div><strong>${money(sig.entry_price)}</strong></div>`,
  `<div class=diagCell><div class=label>Source</div><strong>${sourceLabel}</strong></div>`,
  `</div>`,
  `<div class=gateGrid>${(d.gates||[]).map(g=>`<div class=gatePill><b class="${g.passed?'green':'red'}">${g.passed?'PASS':'FAIL'} ${esc(g.name)}</b><span>${esc(g.value!=null?g.value+' - '+(g.detail||''):g.detail||'')}</span></div>`).join('')}</div>`,
  `<div class=stackRows>`,
  `${kv([['threshold',(d.model||{}).threshold,'blue'],['ML margin',mlMargin,Number(modelDiag.ml_margin??fc.ml_margin??0)>=0?'green':'red'],['size',risk.position_pct_display==null?(fc.position_size_pct==null?'--':fc.position_size_pct+'%'):risk.position_pct_display+'%','blue'],['risk plan',riskPlan,'blue'],['data rows',dataRows,(dataDiag.passed??true)?'blue':'red'],['data meaning',dataDiag.meaning||'full feature history available','blue'],['missing model features',modelDiag.missing_model_feature_count??0,(modelDiag.missing_model_feature_count||0)?'amber':'green'],['feature date',dataDiag.feature_date||feature.feature_date||'--','blue'],['sector',market.sector||d.sector||'unknown','blue'],['regime',market.regime||'--','blue'],['ADV20 $',market.adv20_dollar_vol==null?'--':Number(market.adv20_dollar_vol).toLocaleString(),'blue'],['LLM',llm.ran?(llm.decision||llm.status||'ran'):(llm.status||'not run'),llm.decision==='skip'?'red':llm.ran?'green':'amber']])}`,
  `</div>`,
  `<div class="alertItem ${ok?'good':llm.research_only?'warn':'warn'}"><b>System read</b> ${esc(systemRead)}</div>`,
  `<div class="alertItem"><b>Feature snapshot</b> ${esc(techBits)}</div>`,
  `<div class="alertItem ${llm.decision==='skip'?'danger':llm.ran?'good':'warn'}"><b>${llmLabel}</b> ${esc(llm.reason||llm.status||'no LLM decision')}</div>`,
  `<pre class=diagJson>${esc(singleSymbolDiagJson(diag,llm))}</pre>`
 ].join('');
}
function pickDiagSymbol(sym){$('diagSymbol').value=sym;$('diagStatus').textContent='loaded example '+sym}
async function loadDiagnosticExamples(){
 try{
  const d=await (await fetch('/api/symbol_diagnostic_examples?limit=8&ts='+Date.now())).json();
  const rows=(d.examples||[]);
  $('diagExamples').innerHTML=rows.length?rows.map(x=>`<button class=exampleBtn type=button onclick="pickDiagSymbol('${esc(x.symbol)}')">${esc(x.symbol)} ${Number(x.probability||0).toFixed(3)}</button>`).join(''):'<span class=sub>No non-open ML-pass examples found</span>';
 }catch(e){$('diagExamples').innerHTML='<span class=sub>examples unavailable</span>'}
}
async function runSymbolDiagnostic(){
 const sym=($('diagSymbol').value||'').trim().toUpperCase();
 if(!sym){$('diagStatus').textContent='enter a symbol';return}
 setDiagCollapsed(false);
 const btn=$('diagRun'); btn.disabled=true; btn.textContent='RUNNING...'; $('diagBadge').textContent='working'; $('diagStatus').textContent='fetching and scoring '+sym; $('diagResult').innerHTML='<div class=empty>Running full diagnostic...</div>';
 try{
  const force=$('diagForceLlm').checked;
  const r=await fetch('/api/symbol_diagnostic',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({symbol:sym,refresh:true,run_llm:$('diagLlm').checked||force,force_llm:force})});
  const d=await r.json(); renderSymbolDiagnostic(d); $('diagStatus').textContent=r.ok?'done':'failed';
 }catch(e){
  renderSymbolDiagnostic({ok:false,error:String(e)}); $('diagStatus').textContent='failed';
 }finally{
  btn.disabled=false; btn.textContent='RUN CHECK';
 }
}
if($('diagSymbol')){$('diagSymbol').addEventListener('keydown',e=>{if(e.key==='Enter')runSymbolDiagnostic()})}
if($('diagToggle'))restoreSymbolDiagnosticState();
if($('diagExamples'))loadDiagnosticExamples();
async function loadOperator(){
 const ctrl=new AbortController();
 const timer=setTimeout(()=>ctrl.abort(),1800);
 let o;
 try{
  o=await (await fetch('/api/operator?ts='+Date.now(),{signal:ctrl.signal})).json();
 }catch(err){
  $('operatorSummary').innerHTML='<span class=badge>operator panel delayed</span>';
  $('refreshState').textContent=autoRefresh?'auto 30s':'manual';
  return;
 }finally{
  clearTimeout(timer);
 }
 const c=o.clock||{}, e=o.engine||{}, f=o.filters||{}, r=o.risk||{}, real=o.reality||{}, sh=o.nse_shadow||{};
 $('operatorSummary').innerHTML=(o.summary||[]).map(x=>`<span class=badge>${x}</span>`).join('');
 $('marketClock').innerHTML=[
  `<div class=miniCell><div class=label>US Market</div><strong>${dot(c.us_market_open)}${c.us_market_open?'Open':'Closed'}</strong><div class=sub>${c.now_ny||'--'}</div></div>`,
  `<div class=miniCell><div class=label>NSE Market</div><strong>${dot(c.nse_market_open)}${c.nse_market_open?'Open':'Closed'}</strong><div class=sub>${c.now_ist||'--'}</div></div>`,
  `<div class=miniCell><div class=label>US Refresh</div><strong>${c.next_us_refresh||'--'}</strong></div>`,
  `<div class=miniCell><div class=label>US Signals</div><strong>${c.next_us_screen||'--'}</strong></div>`,
  `<div class=miniCell><div class=label>US Execute</div><strong>${c.next_us_trade||'--'}</strong></div>`,
  `<div class=miniCell><div class=label>US Intraday MTM</div><strong>${c.next_us_intraday||'--'}</strong></div>`,
  `<div class=miniCell><div class=label>NSE Intraday MTM</div><strong>${c.next_nse_intraday||'--'}</strong></div>`,
  `<div class=miniCell><div class=label>NSE Daily</div><strong>${c.next_nse_daily||'--'}</strong></div>`
 ].join('');
 const p=e.processes||{}, la=e.log_age_seconds||{};
 $('engineStatus').innerHTML=kv([
  ['cron', e.cron_active?'active':(e.cron_status||'unknown'), e.cron_active?'green':'amber'],
  ['US trade proc', p.us_trade&&p.us_trade.running?'running':'scheduled/stopped', p.us_trade&&p.us_trade.running?'green':'amber'],
  ['US screen proc', p.us_screen&&p.us_screen.running?'running':'scheduled/stopped', p.us_screen&&p.us_screen.running?'green':'amber'],
  ['US intraday MTM', p.us_intraday&&p.us_intraday.running?'running':'scheduled/stopped', p.us_intraday&&p.us_intraday.running?'green':'blue'],
  ['NSE intraday MTM', p.nse_intraday&&p.nse_intraday.running?'running':'scheduled/stopped', p.nse_intraday&&p.nse_intraday.running?'green':'blue'],
  ['NSE daily proc', p.nse_daily&&p.nse_daily.running?'running':'scheduled/stopped', p.nse_daily&&p.nse_daily.running?'green':'amber'],
  ['dashboard log', age(la.dashboard), 'blue']
 ]);
 const actions=o.actions||[]; $('actionCount').textContent=actions.length+' events'; $('actionLog').innerHTML=actions.length?actions.map(x=>`<div class=event><b>${x.source}</b><span>${x.text}<br><small>${age(x.age_seconds)}</small></span></div>`).join(''):'<div class=empty>No recent events</div>';
 const usF=f.us||{}, nseF=f.nse||{}, usR=r.us||{}, nseR=r.nse||{}, usReal=real.us||{};
 $('usFilter').innerHTML=kv([['accepted signals', usF.raw_signals||0,'blue'],['open positions', usF.open_positions||0,'blue'],['capacity left', usF.capacity_left||0,'green'],['note', usF.note||'--','']]);
 const riskState=usR.unified_state||'normal', riskTone=(riskState==='normal'||riskState==='ok')?'green':'amber';
 $('usRisk').innerHTML=kv([['gross exposure', (usR.gross_exposure_pct||0).toFixed(2)+'%','blue'],['top 5 exposure', (usR.top5_exposure_pct||0).toFixed(2)+'%','blue'],['biggest position', (usR.biggest_position_pct||0).toFixed(2)+'%','amber'],['near stop', usR.near_stop||0,(usR.near_stop||0)>0?'red':'green'],['SL verify', usR.sl_grace||0,(usR.sl_grace||0)>0?'amber':'green'],['risk state', riskState,riskTone],['open losers', usR.losers||0,(usR.losers||0)>0?'amber':'green']]);
 $('usReality').innerHTML=kv([['live WR', usReal.live_win_rate==null?'--':Number(usReal.live_win_rate).toFixed(1)+'%','blue'],['backtest WR', '60.6%','blue'],['live closed P&L', money(usReal.live_closed_pnl||0),(usReal.live_closed_pnl||0)>=0?'green':'red'],['PT hit rate', usReal.pt_hit_rate==null?'--':Number(usReal.pt_hit_rate).toFixed(1)+'%','blue'],['closed sample', usReal.sample_trades||0,'amber']]);
 $('nseShadow').innerHTML=kv([['state', (sh.state||'--').toUpperCase(),'blue'],['gate fired', sh.gate_fired?'YES':'NO',sh.gate_fired?'red':'green'],['narrow score', sh.narrow_score==null?'--':Number(sh.narrow_score).toFixed(3),'amber'],['VIX', sh.vix==null?'--':Number(sh.vix).toFixed(2),'blue'],['Nifty 60d', sh.nifty_ret60==null?'--':Number(sh.nifty_ret60).toFixed(2)+'%',(sh.nifty_ret60||0)>=0?'green':'red']]);
 $('nseFilter').innerHTML=kv([['pre gate', nseF.pre_gate||0,'blue'],['gate blocked', nseF.gate_blocked||0,(nseF.gate_blocked||0)>0?'red':'green'],['post gate', nseF.post_gate||0,'blue'],['execution filtered', nseF.execution_filtered||0,(nseF.execution_filtered||0)>0?'amber':'green'],['post filter', nseF.post_filter||0,'green']]);
 $('nseRisk').innerHTML=kv([['gross notional', inr(nseR.gross_notional_inr||0),'blue'],['gross exposure', (nseR.gross_exposure_pct||0).toFixed(2)+'%','blue'],['top 5 exposure', (nseR.top5_exposure_pct||0).toFixed(2)+'%','amber'],['biggest position', (nseR.biggest_position_pct||0).toFixed(2)+'%','amber'],['cash', inr(nseR.cash_inr||0),'green']]);
 const alerts=o.alerts||[], usA=alerts.filter(x=>x.scope==='US'), nseA=alerts.filter(x=>x.scope==='NSE'); $('usAlertCount').textContent=usA.length; $('nseAlertCount').textContent=nseA.length; renderAlerts('usAlerts',usA); renderAlerts('nseAlerts',nseA);
 $('refreshState').textContent=autoRefresh?'auto 30s':'manual';
}
async function loadUS(){
 const snap=await (await fetch('/api/snapshot?ts='+Date.now())).json(); const p=snap.portfolio||{}, a=snap.analytics||{}, r=snap.regime||{}, s=snap.signals||{};
 window.__usRegime=r;
 $('pv').textContent=money(p.portfolio_value); tone($('pv'),p.total_return_pct); $('ret').textContent=pct(p.total_return_pct);
 $('cash').textContent=money(p.cash); $('cpnl').textContent=money(a.closed_pnl); tone($('cpnl'),a.closed_pnl); $('oc').textContent=p.open_positions_count||0;
 $('wr').textContent=a.win_rate==null?'--':Number(a.win_rate).toFixed(1)+'%'; $('tc').textContent=(a.closed_trades||0)+' closed - backtest 60.6%';
 $('dd').textContent=pct(p.drawdown_from_peak_pct); tone($('dd'),-Number(p.drawdown_from_peak_pct||0));
 updateChromeRegime(); $('spy').textContent=Number(r.spy_realized_vol||0).toFixed(2)+'%'; $('reg').textContent=r.vol_regime||'--'; $('mul').textContent=(r.vol_multiplier||'--')+'x'; $('desc').textContent=r.description||''; $('updated').textContent='updated '+new Date().toLocaleTimeString();
 $('sd').textContent=s.signal_date||'--'; const sig=s.signals||[]; $('sig').innerHTML=sig.length?sig.map(x=>`<tr><td>${x.rank}</td><td><b>${x.symbol}</b></td><td>${Number(x.probability||0).toFixed(3)}</td><td>${money(x.entry_price)}</td><td class=green>${money(x.profit_target_price)}</td><td class=red>${stopMoney(x)}</td></tr>`).join(''):'<tr><td colspan=6 class=empty>No signals</td></tr>';
 const pos=p.positions||[]; $('pb').textContent=pos.length+' open'; window.__lastPositions=pos; $('pos').innerHTML=pos.length?pos.map(x=>{let c=Number(x.unrealized_pnl||0)>=0?'green':'red',prog=Math.min(100,(Number(x.days_held||0)/8)*100),g=x.sl_grace||null,gb=g?` <span class=badge title="${esc(g.llm_reason||g.key_signal||'SL verification active')}">SL</span>`:'';return `<tr><td><b>${esc(x.symbol)}</b>${gb}</td><td>${money(x.entry_price)}</td><td>${money(x.current_price)}</td><td class=green>${money(x.profit_target_price)}</td><td class=red>${stopMoney(x)}</td><td>${x.days_held}d</td><td><div class=bar><div class=fill style="width:${prog}%"></div></div></td><td class=${c}>${money(x.unrealized_pnl)}</td><td class=${c}>${pct(x.unrealized_pnl_pct)}</td><td>${Number(x.confidence||0).toFixed(3)}</td></tr>`}).join(''):'<tr><td colspan=10 class=empty>No open positions</td></tr>';
 const ua=snap.unrealized_attribution||{}, ul=ua.losers||[], uw=ua.winners||[], urows=[...ul,...uw].slice(0,16);$('usAttrBadge').textContent=`net ${money(ua.net_unrealized_pnl||0)}`;$('usAttribution').innerHTML=urows.length?urows.map(x=>{const pnl=Number(x.pnl||0),ret=Number(x.return_pct||0);return `<tr><td><b>${x.symbol}</b></td><td>${Number(x.position_pct||0).toFixed(3)}%</td><td>${money(x.entry_price)}</td><td>${money(x.current_price)}</td><td class="${ret>=0?'green':'red'}">${pct(ret)}</td><td class="${pnl>=0?'green':'red'}">${money(pnl)}</td><td>${pnl<0?Number(x.gross_loss_share_pct||0).toFixed(1)+'%':'--'}</td></tr>`}).join(''):'<tr><td colspan=7 class=empty>No open P&L attribution yet</td></tr>';
 const tr=a.trades||[]; $('cb').textContent=(a.closed_trades||0)+' closed'; $('tr').innerHTML=tr.length?tr.map(x=>{let c=Number(x.pnl||0)>=0?'green':'red';return `<tr><td>${x.exit_date||'--'}</td><td><b>${x.symbol}</b></td><td>${x.quantity!=null?x.quantity.toFixed(0)+'sh':'--'}</td><td>${money(x.entry_price)}</td><td>${money(x.exit_price)}</td><td class=${c}>${pct(x.return_pct)}</td><td class=${c}>${money(x.pnl)}</td><td>${x.exit_reason||'--'}</td></tr>`}).join(''):'<tr><td colspan=9 class=empty>No closed trades</td></tr>';
 const up=pos.filter(x=>Number(x.unrealized_pnl||0)>0).length, down=pos.filter(x=>Number(x.unrealized_pnl||0)<0).length, total=pos.reduce((z,x)=>z+Number(x.unrealized_pnl||0),0); $('outBadge').textContent=(a.closed_trades||0)+' closed'; $('summaryBox').innerHTML=`<div class=metricLine><span class=label>Open Winners</span><b class=green>${up}</b></div><div class=metricLine><span class=label>Open Losers</span><b class=red>${down}</b></div><div class=metricLine><span class=label>Unrealized</span><b class="${total>=0?'green':'red'}">${money(total)}</b></div><div class=metricLine><span class=label>PT Hit Rate</span><b>${a.profit_target_hit_rate==null?'--':Number(a.profit_target_hit_rate).toFixed(1)+'%'}</b></div>`;
 const series=await (await fetch('/api/pnl?ts='+Date.now())).json(); renderCharts(series);
}
let pnlChart, portChart;
let chartSig='';
function renderCharts(d){ if(!window.Chart||!d.labels||d.labels.length<2)return; window.__lastSeries=d; const sig=JSON.stringify([d.labels,d.cumulative_pnl,d.portfolio_value,document.documentElement.dataset.theme,document.documentElement.dataset.accent]); $('pnlBadge').textContent=money(d.cumulative_pnl.at(-1)); $('portBadge').textContent=money(d.portfolio_value.at(-1)); if(sig===chartSig)return; chartSig=sig; if(pnlChart)pnlChart.destroy(); if(portChart)portChart.destroy(); const grid=css('--grid'), muted=css('--muted'), green=css('--accent'), blue=css('--accent2'); const common={responsive:true,maintainAspectRatio:false,animation:false,interaction:{mode:'index',intersect:false},plugins:{legend:{display:false},tooltip:{backgroundColor:css('--panel'),titleColor:css('--ink'),bodyColor:css('--ink'),borderColor:css('--line'),borderWidth:1}},scales:{x:{ticks:{color:muted,maxTicksLimit:6},grid:{color:grid}},y:{ticks:{color:muted},grid:{color:grid}}}}; pnlChart=new Chart($('pnlChart'),{type:'line',data:{labels:d.labels,datasets:[{data:d.cumulative_pnl,borderColor:green,backgroundColor:green+'22',fill:true,tension:.28,pointRadius:2,pointHoverRadius:5}]},options:common}); portChart=new Chart($('portChart'),{type:'line',data:{labels:d.labels,datasets:[{data:d.portfolio_value,borderColor:blue,backgroundColor:blue+'22',fill:true,tension:.28,pointRadius:2,pointHoverRadius:5}]},options:common});}
let nseSignalChart, nseExposureChart, nsePnlCurveChart, nsePortfolioCurveChart;
let nseChartSig='';
function renderNseCharts(signals, positions, payload={}){
 if(!window.Chart)return;
 const curves=payload.curves||{};
 const sig=JSON.stringify([signals.map(x=>[x.symbol,x.prob]),positions.map(x=>[x.symbol,x.current_value_inr||x.notional_inr]),curves.labels,curves.cumulative_pnl,curves.portfolio_value,document.documentElement.dataset.theme,document.documentElement.dataset.accent]);
 if(sig===nseChartSig)return;
 nseChartSig=sig;
 if(nseSignalChart)nseSignalChart.destroy();
 if(nseExposureChart)nseExposureChart.destroy();
 if(nsePnlCurveChart)nsePnlCurveChart.destroy();
 if(nsePortfolioCurveChart)nsePortfolioCurveChart.destroy();
 const grid=css('--grid'), muted=css('--muted'), blue=css('--accent2'), green=css('--accent'), amber=css('--amber');
 const common={responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{display:false}},scales:{x:{ticks:{color:muted,maxRotation:0},grid:{display:false}},y:{ticks:{color:muted},grid:{color:grid}}}};
 const s=signals.slice(0,12);
 nseSignalChart=new Chart($('nseSignalChart'),{type:'bar',data:{labels:s.map(x=>x.symbol),datasets:[{data:s.map(x=>Number(x.prob||0)),backgroundColor:s.map((_,i)=>i%2?blue:green),borderRadius:5}]},options:{...common,scales:{...common.scales,y:{min:.50,max:.60,ticks:{color:muted},grid:{color:grid}}}}});
 const p=positions.slice(0,10);
 nseExposureChart=new Chart($('nseExposureChart'),{type:'doughnut',data:{labels:p.map(x=>x.symbol),datasets:[{data:p.map(x=>Number(x.current_value_inr||x.notional_inr||0)),backgroundColor:[green,blue,amber,'#8b5cf6','#14b8a6','#f97316','#06b6d4','#84cc16','#ec4899','#64748b'],borderWidth:1,borderColor:css('--panel')}]},options:{responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{position:'right',labels:{color:muted,boxWidth:10,font:{size:11}}}},cutout:'64%'}});
 const labels=(curves.labels&&curves.labels.length?curves.labels:[payload.asof_date||'now']);
 const pnl=(curves.cumulative_pnl&&curves.cumulative_pnl.length?curves.cumulative_pnl:[Number(payload.metrics?.pnl||0)]);
 const pv=(curves.portfolio_value&&curves.portfolio_value.length?curves.portfolio_value:[Number(payload.metrics?.portfolio_value||payload.metrics?.portfolio_value_est||1000000)]);
 const curveCommon={responsive:true,maintainAspectRatio:false,animation:false,interaction:{mode:'index',intersect:false},plugins:{legend:{display:false}},scales:{x:{ticks:{color:muted,maxTicksLimit:5},grid:{color:grid}},y:{ticks:{color:muted},grid:{color:grid}}}};
 nsePnlCurveChart=new Chart($('nsePnlCurveChart'),{type:'line',data:{labels,datasets:[{data:pnl,borderColor:green,backgroundColor:green+'22',fill:true,tension:.32,pointRadius:2}]},options:curveCommon});
 nsePortfolioCurveChart=new Chart($('nsePortfolioCurveChart'),{type:'line',data:{labels,datasets:[{data:pv,borderColor:blue,backgroundColor:blue+'22',fill:true,tension:.32,pointRadius:2}]},options:curveCommon});
}
async function loadNSE(){try{
 const n=await (await fetch('/api/nse?ts='+Date.now())).json();
 const m=n.metrics||{}, gate=n.gate_decision||{}, state=gate.regime_state||((n.regime||'').includes('calm')?'clear':'crisis');
 window.__lastNse=n;
 $('nseDate').textContent=n.asof_date||'--';$('nseRegime').textContent=(n.regime||'--').toUpperCase();$('nseVix').textContent=Number(n.india_vix||0).toFixed(2);
 $('nseOpen').textContent=m.positions||0;$('nsePnl').textContent=inr(m.pnl||0);tone($('nsePnl'),m.pnl||0);$('nsePnl2').textContent=inr(m.pnl||0);tone($('nsePnl2'),m.pnl||0);
 $('nseGate').textContent=gate.gate_fired?'FIRED':'CLEAR';tone($('nseGate'),gate.gate_fired?-1:1);$('nseGateSub').textContent=gate.gate_reason||state||'shadow logger';
 $('nseMoneyBadge').textContent=inr(m.portfolio_value||m.portfolio_value_est||1000000);$('nseInvested').textContent=inr(m.notional||0);$('nseCash').textContent=inr(m.cash||0);
 $('nseInvestedBar').style.width=Math.min(100,(Number(m.notional||0)/1000000)*100)+'%';$('nseCashBar').style.width=Math.min(100,(Number(m.cash||0)/1000000)*100)+'%';$('nsePnlBar').style.width=Math.min(100,Math.abs(Number(m.pnl||0))/1000000*1000)+'%';
 $('nseState').textContent=(state||'--').toUpperCase();$('nseGateFired').textContent=gate.gate_fired?'YES':'NO';$('nsePreGate').textContent=m.pre_gate_signals||0;$('nseBlocked').textContent=m.blocked_signals||0;$('nseShadowBadge').textContent=(m.blocked_signals||0)+' blocked';
 const vol=n.vol_state||{}; updateChromeRegime();
 const sig=n.signals||[];$('nseSignalsBadge').textContent=sig.length+' signals';$('nseSignalsBadge2').textContent=sig.length+' signals';$('nseSignals').innerHTML=sig.length?sig.map(x=>`<tr><td><b>${x.symbol}</b></td><td>${Number(x.prob||0).toFixed(3)}</td><td>${inr(x.entry||x.close)}</td><td class=green>${inr(x.profit_target_price)}</td><td class=red>${inr(x.stop_loss_price)}</td></tr>`).join(''):'<tr><td colspan=5 class=empty>No signals</td></tr>';
 const pos=n.positions||[];$('nsePositionsBadge').textContent=pos.length+' open';$('nsePositionsBadge2').textContent=pos.length+' open';$('nsePositions').innerHTML=pos.length?pos.map(x=>{const pnl=Number(x.unrealized_pnl_inr||0);return `<tr><td><b>${x.symbol}</b></td><td>${x.quantity||0}</td><td>${inr(x.entry_price)}</td><td>${inr(x.current_price)}</td><td class=green>${inr(x.profit_target_price)}</td><td class=red>${inr(x.stop_loss_price)}</td><td class="${pnl>=0?'green':'red'}">${inr(pnl)}</td><td>${(x.alert_flags||[]).join(', ')||'ok'}</td></tr>`}).join(''):'<tr><td colspan=8 class=empty>No positions</td></tr>';
 const attr=n.unrealized_attribution||{}, losers=attr.losers||[], winners=attr.winners||[], attrRows=[...losers,...winners].slice(0,16);$('nseAttrBadge').textContent=`net ${inr(attr.net_unrealized_pnl_inr||m.pnl||0)}`;$('nseAttribution').innerHTML=attrRows.length?attrRows.map(x=>{const pnl=Number(x.pnl_inr||0),ret=Number(x.return_pct||0);return `<tr><td><b>${x.symbol}</b></td><td>${x.quantity||0}</td><td>${inr(x.entry_price)}</td><td>${inr(x.current_price)}</td><td class="${ret>=0?'green':'red'}">${ret.toFixed(2)}%</td><td class="${pnl>=0?'green':'red'}">${inr(pnl)}</td><td>${pnl<0?Number(x.gross_loss_share_pct||0).toFixed(1)+'%':'--'}</td></tr>`}).join(''):'<tr><td colspan=7 class=empty>No open P&L attribution yet</td></tr>';
 const closed=n.closed_trades||[];$('nseClosedBadge').textContent=closed.length+' closed';$('nseClosedTrades').innerHTML=closed.length?closed.map(x=>{const pnl=Number(x.pnl||0);return `<tr><td>${x.date||'--'}</td><td><b>${x.symbol||''}</b></td><td>${inr(x.entry)}</td><td>${inr(x.exit)}</td><td class="${Number(x.return_pct||0)>=0?'green':'red'}">${Number(x.return_pct||0).toFixed(2)}%</td><td class="${pnl>=0?'green':'red'}">${inr(pnl)}</td><td>${x.reason||'--'}</td></tr>`}).join(''):'<tr><td colspan=8 class=empty>No closed NSE trades yet</td></tr>';
 const curves=n.curves||{};$('nsePnlCurveBadge').textContent=inr((curves.cumulative_pnl||[]).at(-1)||m.pnl||0);$('nsePortfolioCurveBadge').textContent=inr((curves.portfolio_value||[]).at(-1)||m.portfolio_value||m.portfolio_value_est||1000000);
 const r=n.reality||{}, risk=n.risk_exposure||{}, live=n.live_summary||{};
 $('nseReality').innerHTML=kv([['live WR',r.live_wr==null?'--':Number(r.live_wr).toFixed(1)+'%','blue'],['backtest WR',r.backtest_wr==null?'--':Number(r.backtest_wr).toFixed(1)+'%','blue'],['backtest avg rel',r.backtest_avg_rel_pct==null?'--':Number(r.backtest_avg_rel_pct).toFixed(4)+'%','amber'],['live closed P&L',inr(r.live_closed_pnl||0),(r.live_closed_pnl||0)>=0?'green':'red'],['PT hit rate',r.pt_hit_rate==null?'--':Number(r.pt_hit_rate).toFixed(1)+'%','blue'],['closed sample',r.closed_sample||0,'amber']]);
 $('nseLiveSummary').innerHTML=kv([['winners',live.open_winners||0,'green'],['losers',live.open_losers||0,(live.open_losers||0)>0?'red':'green'],['unrealized',inr(live.unrealized_pnl||0),(live.unrealized_pnl||0)>=0?'green':'red'],['PT hit',live.pt_hit_rate==null?'--':Number(live.pt_hit_rate).toFixed(1)+'%','blue'],['near stop',risk.near_stop_count||0,(risk.near_stop_count||0)>0?'red':'green'],['sizing',Number(vol.multiplier||1).toFixed(2)+'x','blue']]);
 renderNseCharts(sig,pos,n);
}catch(e){console.warn('nse load failed',e)}}
let autoRefresh=true, refreshTimer=null;
async function load(){
 try{
  await Promise.allSettled([loadUS(),loadNSE()]);
  liveRefresh().catch(e=>console.warn('live refresh failed',e));
  loadOperator().catch(e=>console.warn('operator load failed',e));
  $('lastRefresh').textContent='last refresh '+new Date().toLocaleTimeString();
 }catch(e){console.error(e)}
}
function scheduleRefresh(){if(refreshTimer)clearInterval(refreshTimer);refreshTimer=setInterval(()=>{if(autoRefresh)load()},30000)}
$('refreshNow').onclick=()=>load();
$('autoRefresh').onclick=()=>{autoRefresh=!autoRefresh;$('autoRefresh').textContent=autoRefresh?'Auto refresh on':'Auto refresh off';$('refreshState').textContent=autoRefresh?'auto 30s':'manual'};
load(); scheduleRefresh();

function signedMoney(n){
  n=Number(n);
  if(!Number.isFinite(n))return'--';
  return(n<0?'-':'')+'$'+Math.abs(n).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
}
function setLiveTone(cell,n){
  if(!cell)return;
  cell.classList.remove('green','red');
  cell.classList.add(Number(n)>=0?'green':'red');
  cell.style.color='';
}
function applyLivePosition(pos){
  var tbody=document.getElementById('pos');
  if(!tbody||!pos||!pos.symbol)return false;
  var symbol=String(pos.symbol).trim().toUpperCase();
  var rows=tbody.querySelectorAll('tr');
  for(var i=0;i<rows.length;i++){
    var cells=rows[i].querySelectorAll('td');
    if(cells.length<9)continue;
    var rowSymbol=cells[0].textContent.trim().toUpperCase();
    if(rowSymbol.indexOf(symbol)<0)continue;
    var livePrice=Number(pos.live_price);
    var pnl=Number(pos.pnl);
    var pnlPct=Number(pos.pnl_pct);
    if(Number.isFinite(livePrice))cells[2].textContent=money(livePrice);
    if(Number.isFinite(pnl)){
      cells[7].textContent=signedMoney(pnl);
      setLiveTone(cells[7],pnl);
    }
    if(Number.isFinite(pnlPct)){
      cells[8].textContent=pct(pnlPct);
      setLiveTone(cells[8],pnlPct);
    }
    return true;
  }
  return false;
}
function liveTableSymbols(id){
  var tbody=document.getElementById(id);
  if(!tbody)return[];
  return Array.from(tbody.querySelectorAll('tr')).map(function(row){
    var cell=row.querySelector('td');
    return cell?cell.textContent.trim().toUpperCase():'';
  }).filter(Boolean);
}
function liveSymbolsChanged(positions){
  if(!Array.isArray(positions))return false;
  var latest=new Set(positions.map(function(pos){return String(pos.symbol||'').trim().toUpperCase();}).filter(Boolean));
  var current=liveTableSymbols('pos');
  if(current.length!==latest.size)return true;
  return current.some(function(sym){return !latest.has(sym);});
}
function applyLiveAttribution(positions){
  var tbody=document.getElementById('usAttribution');
  if(!tbody||!Array.isArray(positions))return;
  var bySymbol={};
  var total=0,grossLoss=0,winners=0,losers=0;
  positions.forEach(function(pos){
    var sym=String(pos.symbol||'').trim().toUpperCase();
    if(!sym)return;
    var pnl=Number(pos.pnl);
    bySymbol[sym]=pos;
    if(Number.isFinite(pnl)){
      total+=pnl;
      if(pnl>0)winners+=1;
      if(pnl<0){losers+=1;grossLoss+=Math.abs(pnl);}
    }
  });
  tbody.querySelectorAll('tr').forEach(function(row){
    var cells=row.querySelectorAll('td');
    if(cells.length<6)return;
    var sym=cells[0].textContent.trim().toUpperCase();
    var pos=bySymbol[sym];
    if(!pos)return;
    var livePrice=Number(pos.live_price);
    var pnl=Number(pos.pnl);
    var pnlPct=Number(pos.pnl_pct);
    if(Number.isFinite(livePrice))cells[3].textContent=money(livePrice);
    if(Number.isFinite(pnlPct)){
      cells[4].textContent=pct(pnlPct);
      setLiveTone(cells[4],pnlPct);
    }
    if(Number.isFinite(pnl)){
      cells[5].textContent=signedMoney(pnl);
      setLiveTone(cells[5],pnl);
      if(cells[6])cells[6].textContent=pnl<0&&grossLoss?((Math.abs(pnl)/grossLoss*100).toFixed(1)+'%'):'--';
    }
  });
  var attrBadge=document.getElementById('usAttrBadge');
  if(attrBadge)attrBadge.textContent='net '+signedMoney(total);
  updateLiveSummary(positions,winners,losers,total);
}
function updateLiveSummary(positions,winners,losers,total){
  var box=document.getElementById('summaryBox');
  if(!box)return;
  var oldPt=box.querySelector('.metricLine:last-child b');
  var ptText=oldPt?oldPt.textContent:'--';
  box.innerHTML=`<div class=metricLine><span class=label>Open Winners</span><b class=green>${winners}</b></div><div class=metricLine><span class=label>Open Losers</span><b class=red>${losers}</b></div><div class=metricLine><span class=label>Unrealized</span><b class="${total>=0?'green':'red'}">${signedMoney(total)}</b></div><div class=metricLine><span class=label>PT Hit Rate</span><b>${ptText}</b></div>`;
}
async function liveRefresh(){
  try{
    var r=await fetch('/api/live_prices?ts='+Date.now(),{cache:'no-store'});
    if(!r.ok)return;
    var d=await r.json();
    if(!d||!Array.isArray(d.positions))return;
    if(liveSymbolsChanged(d.positions)){
      await loadUS();
      return;
    }
    var openCount=d.positions.length;
    if(document.getElementById('pb'))document.getElementById('pb').textContent=openCount+' open';
    if(document.getElementById('oc'))document.getElementById('oc').textContent=openCount;
    d.positions.forEach(applyLivePosition);
    applyLiveAttribution(d.positions);
    var badge=document.getElementById('livePriceBadge');
    if(badge){
      badge.textContent='LIVE '+new Date().toLocaleTimeString();
      badge.title=d.positions.map(function(p){
        return p.symbol+': '+(p.source||'fallback')+(p.timestamp?' @ '+p.timestamp:'');
      }).join('\n');
    }
  }catch(e){
    console.warn('live price refresh failed',e);
  }
}
function startLivePolling(){
  if(window.__livePollingStarted)return;
  window.__livePollingStarted=true;
  var tick=function(){liveRefresh().finally(function(){setTimeout(tick,5000);});};
  setTimeout(tick,1000);
}
async function runSignals(){
  var btn=document.getElementById('runSignalsBtn');
  if(btn){btn.textContent='RUNNING...';btn.disabled=true;}
  try{
    var r=await fetch('/api/run_signals',{method:'POST'});
    var d=await r.json();
    if(btn)btn.textContent=d.ok?'PID '+d.pid:'FAILED';
  }catch(e){
    if(btn)btn.textContent='FAILED';
  }finally{
    setTimeout(function(){if(btn){btn.textContent='RUN SIGNALS';btn.disabled=false;}},8000);
  }
}
document.addEventListener('DOMContentLoaded',startLivePolling);

</script>
</body></html>
"""


@app.route("/")
def home():
    return render_template_string(HTML)


if __name__ == "__main__":
    print(f"MacroIntel Institutional Dashboard: http://0.0.0.0:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
