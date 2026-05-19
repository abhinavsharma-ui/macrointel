from __future__ import annotations

import csv
import json
import math
import os
import re
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    from dotenv import load_dotenv
    load_dotenv(Path(".env"), override=False)
    load_dotenv(Path(".env.example"), override=False)
except Exception:
    pass


REPORT_PATH = Path(os.getenv("SL_DECISIONS_PATH", "reports/sl_decisions.json"))
TRADES_CSV = Path(os.getenv("FR_TRADES_CSV", "reports/fixed_return_paper_trades.csv"))
INITIAL_CAPITAL = float(os.getenv("FR_INITIAL_CAPITAL", os.getenv("PAPER_CAPITAL", "100000")))
NY = ZoneInfo("America/New_York")

SOFT_STOP_PCT = float(os.getenv("SL_VERIFY_SOFT_STOP_PCT", os.getenv("SIG_STOP_LOSS_PCT", "3.0")))
HARD_KILL_PCT = float(os.getenv("SL_VERIFY_HARD_KILL_PCT", "7.0"))
MAX_DAYS_BELOW = int(os.getenv("SL_VERIFY_MAX_DAYS_BELOW", "3"))
MAX_HOLD_DAYS = int(os.getenv("SL_VERIFY_MAX_HOLD_DAYS", "10"))
_LLM_KEY_INDEX = 0
_LLM_DEAD_KEY_INDICES: set[int] = set()


def _num(value, default=0.0) -> float:
    try:
        if value is None or value == "":
            return default
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def norm_sym(value) -> str:
    return str(value or "").replace("_US", "").replace(".US", "").upper()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def business_days_held(entry_date: str | None, current_date: datetime) -> int:
    try:
        import pandas as pd

        entry = pd.to_datetime(entry_date, errors="coerce")
        if pd.isna(entry):
            return 0
        return max(0, len(pd.bdate_range(entry.normalize(), datetime.combine(current_date.date(), time.min))) - 1)
    except Exception:
        try:
            return int(_num((current_date.date() - date.fromisoformat(str(entry_date)[:10])).days, 0))
        except Exception:
            return 0


def _llm_key_order(keys: list[str]) -> list[int]:
    n = len(keys)
    return [
        idx
        for offset in range(n)
        for idx in [(_LLM_KEY_INDEX + offset) % n]
        if idx not in _LLM_DEAD_KEY_INDICES
    ]


def _retire_llm_key(idx: int, key_count: int) -> None:
    global _LLM_KEY_INDEX
    _LLM_DEAD_KEY_INDICES.add(idx)
    for offset in range(1, max(1, key_count) + 1):
        next_idx = (idx + offset) % max(1, key_count)
        if next_idx not in _LLM_DEAD_KEY_INDICES:
            _LLM_KEY_INDEX = next_idx
            return


def _client():
    from alpaca.trading.client import TradingClient

    return TradingClient(
        api_key=os.getenv("ALPACA_API_KEY", ""),
        secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
        paper=os.getenv("ALPACA_PAPER", "true").lower() == "true",
    )


def alpaca_enabled() -> bool:
    return bool(os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"))


def get_alpaca_qty(symbol: str, fallback=0.0) -> float:
    qty = _num(fallback)
    if qty > 0 or not alpaca_enabled():
        return qty
    try:
        pos = _client().get_open_position(symbol)
        return abs(_num(getattr(pos, "qty", 0)))
    except Exception:
        return 0.0


def place_backstop_order(symbol: str, qty: float, stop_price: float) -> dict:
    """Place a GTC stop backstop.

    The original design said "limit sell", but a sell limit below market can
    execute immediately. A stop order is the correct Alpaca hard backstop.
    """
    if os.getenv("SL_VERIFY_PLACE_BACKSTOP", "1") == "0":
        return {"ok": False, "status": "disabled"}
    if not alpaca_enabled():
        return {"ok": False, "status": "alpaca_keys_missing"}
    qty = get_alpaca_qty(symbol, qty)
    if qty <= 0:
        return {"ok": False, "status": "qty_unavailable"}
    try:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import StopOrderRequest

        order = StopOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            stop_price=round(stop_price, 2),
        )
        result = _client().submit_order(order)
        return {"ok": True, "status": "placed", "order_id": str(getattr(result, "id", "")), "stop_price": round(stop_price, 2), "qty": qty}
    except Exception as exc:
        return {"ok": False, "status": "place_failed", "error": str(exc)[:240], "stop_price": round(stop_price, 2), "qty": qty}


def cancel_backstop(order_id: str | None) -> dict:
    if not order_id or not alpaca_enabled():
        return {"ok": False, "status": "missing_order_or_keys"}
    try:
        _client().cancel_order_by_id(order_id)
        return {"ok": True, "status": "cancelled", "order_id": order_id}
    except Exception as exc:
        return {"ok": False, "status": "cancel_failed", "order_id": order_id, "error": str(exc)[:180]}


def close_alpaca_position(symbol: str, reason: str) -> dict:
    if not alpaca_enabled():
        return {"ok": False, "status": "alpaca_keys_missing"}
    try:
        import alpaca_bridge

        ok = alpaca_bridge.submit_sell(symbol, reason)
        return {"ok": bool(ok), "status": "submitted" if ok else "submit_failed"}
    except Exception as exc:
        return {"ok": False, "status": "sell_failed", "error": str(exc)[:240]}


def has_earnings_tomorrow(symbol: str, today_ny: date | None = None) -> bool:
    if os.getenv("SL_VERIFY_EARNINGS_CHECK", "1") == "0":
        return False
    today_ny = today_ny or datetime.now(NY).date()
    tomorrow = today_ny + timedelta(days=1)
    try:
        import yfinance as yf

        cal = yf.Ticker(symbol).calendar
        dates = cal.get("Earnings Date", []) if cal else []
        if not isinstance(dates, list):
            dates = [dates]
        for item in dates:
            if item is None:
                continue
            dt = item.date() if hasattr(item, "date") else item
            if dt == tomorrow:
                return True
    except Exception:
        return False
    return False


def fetch_intraday_context(symbol: str, entry: float, current: float, quote: dict | None, stamp_now: datetime) -> dict:
    context = {
        "open_today": None,
        "high_today": None,
        "low_today": None,
        "current": current,
        "volume_today": None,
        "avg_daily_volume": None,
        "volume_ratio": None,
        "spy_return_today": None,
        "sector_return_today": None,
        "already_recovered": current > entry * (1.0 - SOFT_STOP_PCT / 100.0) if entry else False,
        "source": "quote_only",
    }
    if quote:
        context["quote_timestamp"] = quote.get("timestamp")
    if not alpaca_enabled():
        return context
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        client = StockHistoricalDataClient(os.getenv("ALPACA_API_KEY", ""), os.getenv("ALPACA_SECRET_KEY", ""))
        start_ny = datetime.combine(stamp_now.astimezone(NY).date(), time(9, 30), tzinfo=NY)
        request = StockBarsRequest(symbol_or_symbols=[symbol, "SPY"], timeframe=TimeFrame.Minute, start=start_ny.astimezone(timezone.utc), end=stamp_now.astimezone(timezone.utc))
        bars = client.get_stock_bars(request).df
        if bars is None or bars.empty:
            return context
        import pandas as pd

        def one(sym):
            frame = bars
            if isinstance(frame.index, pd.MultiIndex):
                frame = frame.xs(sym, level="symbol")
            return frame.sort_index()

        frame = one(symbol)
        if not frame.empty:
            context.update(
                {
                    "open_today": round(_num(frame["open"].iloc[0]), 4),
                    "high_today": round(float(frame["high"].max()), 4),
                    "low_today": round(float(frame["low"].min()), 4),
                    "current": round(_num(frame["close"].iloc[-1], current), 4),
                    "volume_today": round(float(frame["volume"].sum()), 2) if "volume" in frame.columns else None,
                    "source": "alpaca_1min",
                }
            )
        spy = one("SPY")
        if not spy.empty:
            spy_open = _num(spy["open"].iloc[0])
            spy_last = _num(spy["close"].iloc[-1])
            context["spy_return_today"] = round((spy_last / spy_open - 1.0) * 100.0, 3) if spy_open else None
    except Exception as exc:
        context["intraday_error"] = str(exc)[:180]
    return context


def build_prompt(pos: dict, context: dict, ret_pct: float, sl_checks: int, days_held: int) -> str:
    entry = _num(pos.get("entry_price"))
    current = _num(context.get("current"), _num(pos.get("current_price") or pos.get("last_price")))
    sl_price = entry * (1.0 - SOFT_STOP_PCT / 100.0)
    days_remaining = max(0, MAX_HOLD_DAYS - days_held)
    return f"""You are the stop-loss verification module for a US equities mean-reversion system.

POSITION:
  Symbol: {norm_sym(pos.get("symbol"))}
  Entry: ${entry:.4f}
  Stop loss level: ${sl_price:.4f} (-{SOFT_STOP_PCT:.1f}%)
  Current price: ${current:.4f} ({ret_pct:.2f}%)
  Entry date: {pos.get("entry_date")}
  Days remaining in max hold: {days_remaining}

INTRADAY PRICE ACTION:
  Open today: {context.get("open_today")}
  High today: {context.get("high_today")}
  Low today: {context.get("low_today")}
  Current: {context.get("current")}
  Price has been below SL for: {sl_checks} checks
  Already recovered above SL since crossing: {"YES" if context.get("already_recovered") else "NO"}

VOLUME:
  Volume today so far: {context.get("volume_today")}
  20-day average daily volume: {context.get("avg_daily_volume")}
  Volume ratio: {context.get("volume_ratio")}

MARKET CONTEXT:
  SPY today: {context.get("spy_return_today")}
  Sector ETF today: {context.get("sector_return_today")}

PREVIOUS LLM CHECKS:
  Times LLM has said HOLD on this SL: {sl_checks}

YOUR TASK:
Determine whether this SL hit is:
(a) A temporary dip likely to recover -- the position should be held
(b) A genuine breakdown -- the position should be closed

VERDICTS:
  CLOSE
  HOLD_1
  HOLD_EOD
  HOLD_RECOVERY

RULES:
- If this is the 3rd or more consecutive HOLD verdict, default CLOSE unless recovery evidence is extremely compelling.
- Never recommend HOLD_RECOVERY unless current price is already moving back toward entry.
- A volume ratio above 2.0 with price still falling is almost always CLOSE.
- A long lower wick suggests a spike, bias toward HOLD.

Respond in this exact format:
VERDICT: <CLOSE|HOLD_1|HOLD_EOD|HOLD_RECOVERY>
CONFIDENCE: <LOW|MEDIUM|HIGH>
REASON: <2-3 sentences max>
KEY_SIGNAL: <the single most important factor in your decision>
"""


def parse_llm_response(text: str) -> dict:
    verdict = "CLOSE"
    confidence = "LOW"
    reason = ""
    key_signal = ""
    for line in (text or "").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().upper()
        value = value.strip()
        if key == "VERDICT":
            token = re.sub(r"[^A-Z0-9_]", "", value.upper())
            if token in {"CLOSE", "HOLD_1", "HOLD_EOD", "HOLD_RECOVERY"}:
                verdict = token
        elif key == "CONFIDENCE":
            token = re.sub(r"[^A-Z]", "", value.upper())
            if token in {"LOW", "MEDIUM", "HIGH"}:
                confidence = token
        elif key == "REASON":
            reason = value[:500]
        elif key == "KEY_SIGNAL":
            key_signal = value[:240]
    return {"verdict": verdict, "confidence": confidence, "reason": reason, "key_signal": key_signal, "raw": text[:2000]}


def call_llm(prompt: str) -> dict:
    global _LLM_KEY_INDEX
    if os.getenv("SL_VERIFY_LLM_ENABLED", "1") == "0":
        return {"ok": False, "status": "disabled"}
    keys = [k.strip() for k in os.getenv("GROQ_API_KEY", "").split(",") if k.strip()]
    if not keys:
        return {"ok": False, "status": "keys_missing"}
    try:
        import requests

        model = os.getenv("SL_VERIFY_LLM_MODEL", os.getenv("LLM_FILTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"))
        headers = {"Content-Type": "application/json", "HTTP-Referer": "https://macro-intelligence.local", "X-Title": "MacroIntelligence SL Verifier"}
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a disciplined stop-loss verifier. Obey hard risk rules and output exactly the requested fields."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.05,
        }
        timeout = float(os.getenv("SL_VERIFY_LLM_TIMEOUT_SECONDS", "35"))
        last_error = ""
        for idx in _llm_key_order(keys):
            key = keys[idx]
            try:
                headers["Authorization"] = f"Bearer {key}"
                resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body, timeout=timeout)
                if resp.status_code in (401, 402, 403, 429):
                    last_error = f"{resp.status_code} key[{idx}]"
                    _retire_llm_key(idx, len(keys))
                    continue
                resp.raise_for_status()
                _LLM_KEY_INDEX = idx
                msg = resp.json()["choices"][0]["message"]
                text = (msg.get("content") or msg.get("reasoning") or "").strip()
                parsed = parse_llm_response(text)
                parsed.update({"ok": True, "status": "ok", "key_index": idx, "model": model})
                return parsed
            except Exception as exc:
                last_error = str(exc)[:180]
                continue
        return {"ok": False, "status": "all_keys_failed", "error": last_error}
    except Exception as exc:
        return {"ok": False, "status": "requests_failed", "error": str(exc)[:180]}


def append_decision(row: dict) -> None:
    try:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if REPORT_PATH.exists():
            data = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
            existing = data if isinstance(data, list) else data.get("decisions", [])
        existing = [row] + [x for x in existing if isinstance(x, dict)][:499]
        REPORT_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    except Exception:
        pass


def write_trade(row: dict) -> None:
    TRADES_CSV.parent.mkdir(parents=True, exist_ok=True)
    exists = TRADES_CSV.exists()
    fields = ["closed_at", "symbol", "entry_date", "exit_date", "exit_reason", "entry_price", "exit_price", "position_pct", "pnl_pct", "pnl_contribution_pct", "pnl", "probability"]
    with open(TRADES_CSV, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def close_position(pos: dict, exit_price: float, reason: str, stamp_now: datetime) -> tuple[dict, dict]:
    sym = norm_sym(pos.get("symbol"))
    entry = _num(pos.get("entry_price"))
    pos_pct = _num(pos.get("position_pct"))
    pnl_pct = (exit_price / entry - 1.0) * 100.0 if entry else 0.0
    cancel = cancel_backstop((pos.get("sl_grace") or {}).get("backstop_order_id"))
    sell = close_alpaca_position(sym, reason)
    row = {
        **pos,
        "status": "closed",
        "closed_at": stamp_now.astimezone(timezone.utc).isoformat(),
        "exit_date": stamp_now.date().isoformat(),
        "exit_reason": reason,
        "exit_price": round(exit_price, 4),
        "pnl_pct": round(pnl_pct, 4),
        "pnl_contribution_pct": round(pnl_pct * pos_pct, 6),
        "pnl": round(pnl_pct * pos_pct * INITIAL_CAPITAL / 100.0, 2),
        "sl_backstop_cancel": cancel,
        "sl_alpaca_sell": sell,
    }
    row.pop("sl_grace", None)
    if os.getenv("SL_VERIFY_WRITE_TRADES", "1") != "0":
        write_trade(row)
    return row, {"symbol": sym, "action": "closed", "reason": reason, "exit_price": round(exit_price, 4), "sell": sell}


def should_skip_mechanical_stop(pos: dict, px: dict, today: str | None = None) -> bool:
    if SOFT_STOP_PCT <= 0:
        return False
    grace = pos.get("sl_grace")
    if not isinstance(grace, dict):
        return False
    entry = _num(pos.get("entry_price"))
    if entry <= 0:
        return False
    hard_price = entry * (1.0 - HARD_KILL_PCT / 100.0)
    if _num(px.get("low") or px.get("close")) <= hard_price:
        return False
    if today and str(today) >= str(pos.get("exit_date")):
        return False
    return True


def evaluate_position(pos: dict, quote: dict | None, stamp_now: datetime) -> tuple[dict, dict | None]:
    if SOFT_STOP_PCT <= 0:
        return pos, None
    if str(pos.get("status", "open")).lower() not in {"open", "active"}:
        return pos, None
    sym = norm_sym(pos.get("symbol"))
    entry = _num(pos.get("entry_price"))
    current = _num(pos.get("current_price") or pos.get("last_price") or (quote or {}).get("price"))
    if entry <= 0 or current <= 0:
        return pos, None
    ret_pct = (current / entry - 1.0) * 100.0
    if ret_pct > -SOFT_STOP_PCT:
        grace = pos.get("sl_grace")
        if isinstance(grace, dict):
            cancel = cancel_backstop(grace.get("backstop_order_id"))
            pos.pop("sl_grace", None)
            event = {"symbol": sym, "action": "grace_cleared_recovered", "return_pct": round(ret_pct, 4), "cancel": cancel}
            append_decision({"decision_time": stamp_now.astimezone(timezone.utc).isoformat(), **event})
            return pos, event
        return pos, None

    grace = pos.get("sl_grace") if isinstance(pos.get("sl_grace"), dict) else {}
    today = stamp_now.astimezone(NY).date().isoformat()
    if grace.get("last_below_date") != today:
        grace["sl_days_below"] = int(grace.get("sl_days_below", 0)) + 1
        grace["last_below_date"] = today
    days_held = int(pos.get("days_held") or business_days_held(pos.get("entry_date"), stamp_now))
    hard_price = entry * (1.0 - HARD_KILL_PCT / 100.0)
    hard_reason = None
    if ret_pct <= -HARD_KILL_PCT:
        hard_reason = "HARD_KILL_7PCT"
    elif int(grace.get("sl_days_below", 0)) >= MAX_DAYS_BELOW:
        hard_reason = "HARD_KILL_3DAYS"
    elif days_held >= MAX_HOLD_DAYS:
        hard_reason = "MAX_HOLD"
    elif has_earnings_tomorrow(sym, stamp_now.astimezone(NY).date()):
        hard_reason = "EARNINGS_RISK"
    if hard_reason:
        closed, event = close_position(pos, current, hard_reason, stamp_now)
        append_decision({"decision_time": stamp_now.astimezone(timezone.utc).isoformat(), "symbol": sym, "verdict": "CLOSE", "reason": hard_reason, "price_at_decision": current, "return_pct": round(ret_pct, 4), "sl_checks": int(grace.get("sl_checks", 0))})
        return closed, event

    if not pos.get("sl_grace"):
        backstop = place_backstop_order(sym, _num(pos.get("quantity")), hard_price)
        grace = {
            "first_triggered": stamp_now.astimezone(timezone.utc).isoformat(),
            "last_checked": None,
            "sl_checks": 0,
            "last_verdict": None,
            "sl_days_below": int(grace.get("sl_days_below", 1)),
            "last_below_date": today,
            "hard_backstop_price": round(hard_price, 4),
            "backstop_order_id": backstop.get("order_id"),
            "backstop_status": backstop,
        }
        pos["sl_grace"] = grace
    else:
        pos["sl_grace"] = grace

    context = fetch_intraday_context(sym, entry, current, quote, stamp_now)
    sl_checks = int(grace.get("sl_checks", 0))
    prompt = build_prompt(pos, context, ret_pct, sl_checks, days_held)
    llm = call_llm(prompt)
    if not llm.get("ok"):
        if not grace.get("backstop_order_id") and alpaca_enabled():
            closed, event = close_position(pos, current, "SL_LLM_FAILED_NO_BACKSTOP", stamp_now)
            append_decision({"decision_time": stamp_now.astimezone(timezone.utc).isoformat(), "symbol": sym, "verdict": "CLOSE", "reason": llm.get("status"), "price_at_decision": current, "return_pct": round(ret_pct, 4)})
            return closed, event
        llm = {"verdict": "HOLD_1", "confidence": "LOW", "reason": f"LLM unavailable ({llm.get('status')}); holding only because hard backstop is active or paper-only.", "key_signal": "llm_unavailable", **llm}
    verdict = llm.get("verdict", "CLOSE")
    if sl_checks >= 2 and verdict != "CLOSE" and not context.get("already_recovered"):
        verdict = "CLOSE"
        llm["reason"] = "Third SL check without recovery; forced close by progressive-conviction rule."
        llm["key_signal"] = "third_check_no_recovery"
    decision_row = {
        "symbol": sym,
        "decision_time": stamp_now.astimezone(timezone.utc).isoformat(),
        "entry_price": round(entry, 4),
        "sl_price": round(entry * (1.0 - SOFT_STOP_PCT / 100.0), 4),
        "hard_backstop_price": round(hard_price, 4),
        "price_at_decision": round(current, 4),
        "return_pct": round(ret_pct, 4),
        "pct_below_sl": round(ret_pct + SOFT_STOP_PCT, 4),
        "verdict": verdict,
        "confidence": llm.get("confidence"),
        "reason": llm.get("reason"),
        "key_signal": llm.get("key_signal"),
        "sl_checks": sl_checks + 1,
        "context": context,
        "llm_status": llm.get("status"),
    }
    append_decision(decision_row)
    if verdict == "CLOSE":
        closed, event = close_position(pos, current, "SL_VERIFIED_CLOSE", stamp_now)
        return closed, event
    grace["sl_checks"] = sl_checks + 1
    grace["last_verdict"] = verdict
    grace["last_checked"] = stamp_now.astimezone(timezone.utc).isoformat()
    grace["llm_reason"] = llm.get("reason")
    grace["key_signal"] = llm.get("key_signal")
    pos["sl_grace"] = grace
    return pos, {"symbol": sym, "action": "held_in_sl_grace", "verdict": verdict, "return_pct": round(ret_pct, 4)}


def process_positions(positions: list[dict], quotes: dict[str, dict] | None, stamp_now: datetime) -> tuple[list[dict], list[dict]]:
    if os.getenv("SL_VERIFY_ENABLED", "1") == "0" or SOFT_STOP_PCT <= 0:
        return positions, []
    out = []
    events = []
    quotes = quotes or {}
    for pos in positions:
        try:
            updated, event = evaluate_position(dict(pos), quotes.get(norm_sym(pos.get("symbol"))), stamp_now)
            out.append(updated)
            if event:
                events.append(event)
        except Exception as exc:
            p = dict(pos)
            p["sl_verify_error"] = str(exc)[:240]
            out.append(p)
            events.append({"symbol": norm_sym(pos.get("symbol")), "action": "error", "error": str(exc)[:240]})
    return out, events
