from __future__ import annotations

import csv
import json
import math
import os
from datetime import date, datetime, timezone
from pathlib import Path


POSITIONS_PATH = Path(os.getenv("FR_POSITIONS_PATH", "reports/fixed_return_open_positions.json"))
TRADES_CSV = Path(os.getenv("FR_TRADES_CSV", "reports/fixed_return_paper_trades.csv"))
REPORT_PATH = Path(os.getenv("UNIFIED_RISK_REPORT", "reports/unified_risk_state.json"))
INITIAL_CAPITAL = float(os.getenv("FR_INITIAL_CAPITAL", os.getenv("PAPER_CAPITAL", "100000")))


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


def load_positions(path: Path = POSITIONS_PATH) -> list[dict]:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = data.get("positions", []) if isinstance(data, dict) else data
            return [r for r in rows if isinstance(r, dict)]
    except Exception:
        pass
    return []


def load_closed_trades(path: Path = TRADES_CSV) -> list[dict]:
    if not path.exists():
        return []
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            return [dict(row) for row in csv.DictReader(fh)]
    except Exception:
        return []


def sector_for(symbol: str) -> str:
    try:
        import fixed_return_daily_signals as sig

        return sig.get_sector(symbol) or "unknown"
    except Exception:
        return "unknown"


def current_price(pos: dict) -> float:
    return _num(pos.get("current_price") or pos.get("last_price") or pos.get("entry_price"))


def build_risk_state(positions: list[dict] | None = None, trades: list[dict] | None = None) -> dict:
    positions = positions if positions is not None else load_positions()
    trades = trades if trades is not None else load_closed_trades()
    open_positions = [p for p in positions if str(p.get("status", "open")).lower() in {"open", "active"}]
    sector_exposure: dict[str, float] = {}
    gross = 0.0
    stop_pressure = []
    symbol_rows = []
    for pos in open_positions:
        sym = norm_sym(pos.get("symbol"))
        size = _num(pos.get("position_pct"))
        gross += size
        sec = sector_for(sym)
        sector_exposure[sec] = sector_exposure.get(sec, 0.0) + size
        entry = _num(pos.get("entry_price"))
        px = current_price(pos)
        stop = _num(pos.get("stop_loss_price") or (entry * 0.97 if entry else 0))
        ret = (px / entry - 1.0) * 100.0 if entry and px else 0.0
        dist_stop = (px / stop - 1.0) * 100.0 if stop and px else None
        if dist_stop is not None and dist_stop <= 1.5:
            stop_pressure.append(sym)
        symbol_rows.append(
            {
                "symbol": sym,
                "sector": sec,
                "position_pct": round(size * 100.0, 4),
                "return_pct": round(ret, 4),
                "distance_to_stop_pct": round(dist_stop, 4) if dist_stop is not None else None,
                "sl_grace": bool(pos.get("sl_grace")),
            }
        )

    today = date.today().isoformat()
    today_pnl = sum(_num(row.get("pnl")) for row in trades if str(row.get("exit_date") or row.get("closed_at") or "")[:10] == today)
    closed_pnl = sum(_num(row.get("pnl")) for row in trades)
    peak = INITIAL_CAPITAL
    equity = INITIAL_CAPITAL
    for row in trades:
        equity += _num(row.get("pnl"))
        peak = max(peak, equity)
    drawdown_pct = (peak - equity) / peak * 100.0 if peak else 0.0
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "initial_capital": INITIAL_CAPITAL,
        "open_positions": len(open_positions),
        "gross_exposure_pct": round(gross * 100.0, 4),
        "top_sector_exposure_pct": round(max(sector_exposure.values()) * 100.0, 4) if sector_exposure else 0.0,
        "sector_exposure_pct": {k: round(v * 100.0, 4) for k, v in sorted(sector_exposure.items())},
        "today_closed_pnl": round(today_pnl, 2),
        "closed_pnl": round(closed_pnl, 2),
        "drawdown_pct": round(drawdown_pct, 4),
        "stop_pressure_symbols": stop_pressure,
        "symbols": symbol_rows,
        "limits": {
            "max_gross_exposure_pct": _num(os.getenv("RISK_MAX_GROSS_EXPOSURE_PCT", "35"), 35.0),
            "max_sector_exposure_pct": _num(os.getenv("RISK_MAX_SECTOR_EXPOSURE_PCT", "12"), 12.0),
            "daily_loss_pause_pct": _num(os.getenv("RISK_DAILY_LOSS_PAUSE_PCT", "2.0"), 2.0),
            "drawdown_yellow_pct": _num(os.getenv("RISK_DRAWDOWN_YELLOW_PCT", "5.0"), 5.0),
            "drawdown_orange_pct": _num(os.getenv("RISK_DRAWDOWN_ORANGE_PCT", "8.0"), 8.0),
            "drawdown_red_pct": _num(os.getenv("RISK_DRAWDOWN_RED_PCT", "12.0"), 12.0),
        },
        "state": "ok",
        "messages": [],
    }
    limits = payload["limits"]
    if payload["gross_exposure_pct"] >= limits["max_gross_exposure_pct"]:
        payload["state"] = "block_new_entries"
        payload["messages"].append("gross exposure limit reached")
    if payload["drawdown_pct"] >= limits["drawdown_red_pct"]:
        payload["state"] = "red_halt"
        payload["messages"].append("red drawdown halt")
    elif payload["drawdown_pct"] >= limits["drawdown_orange_pct"]:
        payload["state"] = "orange_no_new_entries"
        payload["messages"].append("orange drawdown no-new-entry state")
    elif payload["drawdown_pct"] >= limits["drawdown_yellow_pct"]:
        payload["state"] = "yellow_half_size"
        payload["messages"].append("yellow drawdown size reduction state")
    try:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        payload["write_error"] = str(exc)[:180]
    return payload


def approve_new_signal(signal: dict, current_positions: list[dict], pending_positions: list[dict] | None = None) -> tuple[bool, str, dict]:
    pending_positions = pending_positions or []
    all_positions = list(current_positions) + list(pending_positions)
    state = build_risk_state(all_positions)
    limits = state.get("limits", {})
    if state["state"] in {"red_halt", "orange_no_new_entries", "block_new_entries"}:
        return False, state["state"], state
    size_pct = _num(signal.get("position_pct")) * 100.0
    sym = norm_sym(signal.get("symbol"))
    sec = sector_for(sym)
    sector_after = _num(state.get("sector_exposure_pct", {}).get(sec)) + size_pct
    if sector_after > _num(limits.get("max_sector_exposure_pct"), 12.0):
        return False, f"sector_exposure_limit_{sec}", state
    gross_after = _num(state.get("gross_exposure_pct")) + size_pct
    if gross_after > _num(limits.get("max_gross_exposure_pct"), 35.0):
        return False, "gross_exposure_limit", state
    return True, "ok", state
