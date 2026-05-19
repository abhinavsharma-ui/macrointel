from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from flask import Flask, jsonify, render_template_string
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
        rows.append(
            {
                "rank": int(signal.get("rank") or index),
                "symbol": str(signal.get("symbol") or trade.get("symbol") or ""),
                "probability": num(signal.get("probability") or signal.get("prob") or signal.get("score") or meta.get("take_probability")),
                "entry_price": price,
                "position_pct": num(signal.get("position_pct") or meta.get("position_pct")),
                "profit_target_price": num(signal.get("profit_target_price") or price * 1.05),
                "stop_loss_price": num(signal.get("stop_loss_price") or price * 0.97),
                "expected_exit_date": str(signal.get("expected_exit_date") or ""),
            }
        )
    return {
        "signal_date": str(data.get("signal_date") or date.today().isoformat()) if isinstance(data, dict) else date.today().isoformat(),
        "signals": rows,
    }


def open_positions():
    raw = read_json(OPEN, [])
    if isinstance(raw, dict):
        raw = raw.get("positions") or raw
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
        current = current_price(sym, entry)
        entry_dt = parse_dt(pos.get("entry_date")) or datetime.now()
        qty = num(pos.get("quantity"))
        pos_pct = num(pos.get("position_pct"))
        ret = ((current / entry) - 1.0) * 100.0 if entry else 0.0
        pnl = (current - entry) * qty if qty else INITIAL * pos_pct * ret / 10_000.0
        rows.append(
            {
                "symbol": sym,
                "entry_date": entry_dt.date().isoformat(),
                "entry_price": round(entry, 4),
                "current_price": round(current, 4),
                "profit_target_price": round(num(pos.get("profit_target_price") or entry * 1.05), 4),
                "stop_loss_price": round(num(pos.get("stop_loss_price") or entry * 0.97), 4),
                "days_held": business_days(entry_dt.date(), date.today()),
                "position_pct": round(pos_pct, 4),
                "unrealized_pnl": round(pnl, 2),
                "unrealized_pnl_pct": round(ret, 2),
                "confidence": num(pos.get("probability") or pos.get("confidence")),
            }
        )
    for key, val in (broker.get("positions", {}) if isinstance(broker, dict) else {}).items():
        sym = str(val.get("symbol") or key.split("::")[0])
        if any(row["symbol"] == sym for row in rows):
            continue
        entry = num(val.get("avg_cost") or val.get("entry_price"))
        qty = num(val.get("quantity"))
        current = current_price(sym, num(val.get("current_price") or entry))
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
            }
        )
    return sorted(rows, key=lambda row: row["days_held"], reverse=True)


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
                            "hold_days": int(num(row.get("hold_days"))),
                            "return_pct": round(ret, 2),
                            "pnl": round(pnl, 2),
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


def portfolio():
    positions = open_positions()
    trades = closed_trades()
    broker_account = broker_account_snapshot()
    initial = num(broker_account.get("initial_capital"), INITIAL) or INITIAL
    csv_realized = sum(num(row["pnl"]) for row in trades)
    broker_realized = num(broker_account.get("realized_pnl"), 0.0)
    realized = broker_realized if abs(broker_realized) > 1e-9 else csv_realized
    broker_cash = broker_account.get("cash")
    cash = num(broker_cash, initial + realized) if broker_cash is not None else initial + realized
    open_value = sum(initial * num(pos["position_pct"]) / 100.0 + num(pos["unrealized_pnl"]) for pos in positions)
    broker_value = num(broker_account.get("portfolio_value"), 0.0)
    value = broker_value if broker_value > 0 else cash + open_value
    if value <= 0:
        value = initial + realized + sum(num(pos["unrealized_pnl"]) for pos in positions)
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
    labels, pnl, equity = [], [], []
    total = 0.0
    for row in reversed(closed_trades()):
        total += num(row["pnl"])
        labels.append(f"{row['symbol']} - {row['exit_date'][-5:]}")
        pnl.append(round(total, 2))
        equity.append(round(INITIAL + total, 2))

    # Make chart badge match the same dashboard override used by the KPI.
    override = read_json([BASE / "reports/dashboard_account_override.json"], {})
    if isinstance(override, dict) and override:
        closed = round(num(override.get("closed_pnl") or override.get("closed_pnl_total")), 2)
        value = round(num(override.get("portfolio_value"), INITIAL + closed), 2)
        if labels:
            labels[-1] = "Dashboard override - now"
            pnl[-1] = closed
            equity[-1] = value
        else:
            labels = ["Dashboard override - now"]
            pnl = [closed]
            equity = [value]

    return {"labels": labels, "cumulative_pnl": pnl, "portfolio_value": equity, "insufficient": len(labels) < 2}

def nse_payload():
    def one(name, default):
        for folder in NSE_DIRS:
            path = folder / name
            try:
                if path.exists():
                    return json.loads(path.read_text())
            except Exception:
                pass
        return default

    sig = one("latest_nse_signals.json", {})
    gate = one("gate_decision.json", {})
    pre_gate = one("signals_pre_gate.json", [])
    blocked = one("blocked_signals.json", [])
    status = one("nse_paper_status.json", {})
    pos_doc = one("nse_paper_positions.json", {})
    orders = one("nse_paper_orders.json", {})
    positions = status.get("positions") or pos_doc.get("positions") or []
    open_pos = [row for row in positions if row.get("status", "open") == "open"]
    notional = sum(num(row.get("notional_inr")) for row in open_pos)
    pnl = sum(num(row.get("unrealized_pnl_inr", row.get("pnl_inr", 0))) for row in open_pos)
    return {
        "ok": bool(sig or positions or orders),
        "asof_date": sig.get("asof_date"),
        "regime": sig.get("regime"),
        "india_vix": sig.get("india_vix"),
        "allow_new_entries": sig.get("allow_new_entries"),
        "hedge_beta": sig.get("hedge_beta", 0),
        "signals": sig.get("signals", []),
        "signals_pre_gate": pre_gate if isinstance(pre_gate, list) else [],
        "blocked_signals": blocked if isinstance(blocked, list) else [],
        "gate_decision": gate if isinstance(gate, dict) else {},
        "positions": open_pos,
        "orders": orders.get("new_orders", []),
        "metrics": {
            "signals": len(sig.get("signals", [])),
            "pre_gate_signals": len(pre_gate) if isinstance(pre_gate, list) else 0,
            "blocked_signals": len(blocked) if isinstance(blocked, list) else 0,
            "positions": len(open_pos),
            "notional": notional,
            "pnl": pnl,
            "portfolio_value": 1_000_000.0 + pnl,
            "cash": max(0.0, 1_000_000.0 - notional),
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


def run_cmd(args, timeout=2):
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

    us_trade = next_weekday_at(now_ny, 9, 0)
    us_screen = next_weekday_at(now_ny, 8, 0)
    nse_daily = next_weekday_at(now_ist, 16, 0)
    return {
        "now_ny": now_ny.strftime("%Y-%m-%d %H:%M %Z"),
        "now_ist": now_ist.strftime("%Y-%m-%d %H:%M %Z"),
        "us_market_open": is_open(now_ny, 9, 30, 16, 0),
        "nse_market_open": is_open(now_ist, 9, 15, 15, 30),
        "next_us_screen": us_screen.strftime("%a %H:%M %Z"),
        "next_us_trade": us_trade.strftime("%a %H:%M %Z"),
        "next_nse_daily": nse_daily.strftime("%a %H:%M %Z"),
    }


def engine_status_payload():
    cron_code, cron_text = run_cmd(["systemctl", "is-active", "cron"])

    def proc(pattern):
        code, text = run_cmd(["pgrep", "-af", pattern])
        lines = [line for line in text.splitlines() if line.strip()]
        return {"running": code == 0 and bool(lines), "matches": lines[:4]}

    logs = {
        "us_trade": BASE / "logs/launch_trade.log",
        "us_screen": BASE / "logs/launch_screen.log",
        "nse_daily": BASE / "logs/nse_daily.log",
        "dashboard": BASE / "logs/dashboard_institutional.log",
    }
    return {
        "cron_active": cron_code == 0 and cron_text.strip() == "active",
        "cron_status": cron_text.strip() or "unavailable",
        "processes": {
            "us_trade": proc("launch.py --trade"),
            "us_screen": proc("launch.py --screen"),
            "dashboard": proc("dashboard_institutional.py"),
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
        ("US trade", BASE / "logs/launch_trade.log"),
        ("US screen", BASE / "logs/launch_screen.log"),
        ("NSE daily", BASE / "logs/nse_daily.log"),
        ("Dashboard", BASE / "logs/dashboard_institutional.log"),
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
    us_exposures = sorted([num(row.get("position_pct")) for row in positions], reverse=True)
    nse_exposures = sorted([num(row.get("notional_inr")) for row in nse.get("positions", [])], reverse=True)
    nse_notional = sum(nse_exposures)
    us_unrealized = sum(num(row.get("unrealized_pnl")) for row in positions)
    return {
        "us": {
            "gross_exposure_pct": round(sum(us_exposures), 2),
            "top5_exposure_pct": round(sum(us_exposures[:5]), 2),
            "biggest_position_pct": round(us_exposures[0], 2) if us_exposures else 0,
            "unrealized_pnl": round(us_unrealized, 2),
            "losers": len([row for row in positions if num(row.get("unrealized_pnl")) < 0]),
            "near_stop": len([row for row in positions if num(row.get("current_price")) <= num(row.get("stop_loss_price")) * 1.015]),
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
    summary = [
        "US cron active" if engine["cron_active"] else "US cron not confirmed",
        f"US next trade {clock['next_us_trade']}",
        f"NSE next daily {clock['next_nse_daily']}",
        f"US open positions {p.get('open_positions_count', 0)}",
        f"NSE open positions {nse.get('metrics', {}).get('positions', 0)}",
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
    return jsonify(_apply_dashboard_override({"portfolio": portfolio(), "signals": signal_payload(), "regime": volatility_regime(), "analytics": analytics(), "status": {"uptime_seconds": int(time.time() - START)}}))


@app.route("/api/pnl")
def api_pnl():
    return jsonify(pnl_series())


@app.route("/api/nse")
def api_nse():
    return jsonify(nse_payload())


@app.route("/api/operator")
def api_operator():
    return jsonify(operator_payload())


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
</head>
<body>
<div class="top"><div class="dot"></div><div class="brand">MacroIntel Institutional</div><div class="chip">Paper Trading</div><div class="chip" id="volChip">VOL --</div><button class="themeBtn" id="themeToggle" type="button">Dark</button><div style="flex:1"></div><div class="chip" id="updated">loading</div></div>
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
<div class="grid mainGrid" style="margin-bottom:12px">
<div class="card"><div class="titleRow"><div><div class="title">Cumulative P&L</div><div class="sub">realized profit curve</div></div><div class="badge" id="pnlBadge">--</div></div><div class="chartBox"><canvas id="pnlChart"></canvas></div></div>
<div class="card"><div class="titleRow"><div><div class="title">Portfolio Value</div><div class="sub">paper equity path</div></div><div class="badge" id="portBadge">--</div></div><div class="chartBox"><canvas id="portChart"></canvas></div></div>
</div>
<div class="grid workGrid">
<div class="grid">
<div class="card"><div class="titleRow"><div><div class="title">Open Positions</div><div class="sub">entry, target, stop, progress and P&L</div></div><div class="badge" id="pb">0 open</div></div><table><thead><tr><th>Symbol</th><th>Entry</th><th>Current</th><th>PT</th><th>SL</th><th>Days</th><th>Progress</th><th>P&L</th><th>Return</th><th>Conf</th></tr></thead><tbody id="pos"></tbody></table></div>
<div class="card"><div class="titleRow"><div><div class="title">Closed Trades</div><div class="sub">settled exits</div></div><div class="badge" id="cb">0 closed</div></div><table><thead><tr><th>Date</th><th>Symbol</th><th>Entry</th><th>Exit</th><th>Hold</th><th>Return</th><th>P&L</th><th>Reason</th></tr></thead><tbody id="tr"></tbody></table></div>
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
<div class="nseBlotter">
<div class="card"><div class="titleRow"><div><div class="title">NSE Signals</div><div class="sub">post-gate execution stream</div></div><div class="badge" id="nseSignalsBadge2">0</div></div><table><thead><tr><th>Symbol</th><th>Prob</th><th>Close</th><th>ADV</th></tr></thead><tbody id="nseSignals"></tbody></table></div>
<div class="card"><div class="titleRow"><div><div class="title">NSE Positions</div><div class="sub">paper allocation</div></div><div class="badge" id="nsePositionsBadge2">0</div></div><table><thead><tr><th>Symbol</th><th>Qty</th><th>Entry</th><th>Notional</th><th>Prob</th></tr></thead><tbody id="nsePositions"></tbody></table></div>
</div>
<div class="diagnosticGrid">
<div class="card"><div class="titleRow"><div><div class="title">NSE Shadow Gate Tracker</div><div class="sub">logging only, allocator unchanged</div></div></div><div class="stackRows" id="nseShadow"></div></div>
<div class="card"><div class="titleRow"><div><div class="title">Blocked / Filtered Signals</div><div class="sub">pre-gate to post-execution stream</div></div></div><div class="stackRows" id="nseFilter"></div></div>
<div class="card"><div class="titleRow"><div><div class="title">Risk Exposure</div><div class="sub">Rs 10L paper book</div></div></div><div class="stackRows" id="nseRisk"></div></div>
<div class="card"><div class="titleRow"><div><div class="title">Position Alert Flags</div><div class="sub">gate, stale signal and blocked stream</div></div><div class="badge" id="nseAlertCount">0</div></div><div class="alertList" id="nseAlerts"></div></div>
</div>
</section>
</div>
<script>
const $=id=>document.getElementById(id);
const money=n=>n==null||isNaN(n)?'--':'$'+Number(n).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
const pct=n=>n==null||isNaN(n)?'--':(Number(n)>0?'+':'')+Number(n).toFixed(2)+'%';
const inr=n=>'Rs '+Number(n||0).toLocaleString('en-IN',{maximumFractionDigits:0});
function tone(el,n){el.classList.remove('green','red','amber'); if(n>0)el.classList.add('green'); if(n<0)el.classList.add('red')}
function css(name){return getComputedStyle(document.documentElement).getPropertyValue(name).trim()}
function setTheme(mode){document.documentElement.dataset.theme=mode;localStorage.setItem('mi_theme',mode);$('themeToggle').textContent=mode==='dark'?'Light':'Dark'; if(window.__lastSeries)renderCharts(window.__lastSeries)}
setTheme(localStorage.getItem('mi_theme')||'dark');
$('themeToggle').onclick=()=>setTheme(document.documentElement.dataset.theme==='dark'?'light':'dark');
document.querySelectorAll('.tab').forEach(btn=>btn.onclick=()=>{document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));document.querySelectorAll('.view').forEach(x=>x.classList.remove('active'));btn.classList.add('active');$(btn.dataset.view).classList.add('active')});
const age=s=>s==null?'--':s<60?s+'s ago':s<3600?Math.round(s/60)+'m ago':Math.round(s/3600)+'h ago';
function kv(rows){return rows.map(r=>`<div class=stackRow><span>${r[0]}</span><b class="${r[2]||''}">${r[1]}</b></div>`).join('')}
function dot(on,bad=false){return `<span class="statusDot ${bad?'bad':on?'on':'warn'}"></span>`}
function renderAlerts(id,rows){$(id).innerHTML=rows.length?rows.map(a=>`<div class="alertItem ${a.level||''}"><b>${a.scope}</b> ${a.text}</div>`).join(''):'<div class="alertItem good">No active flags</div>'}
async function loadOperator(){
 const o=await (await fetch('/api/operator?ts='+Date.now())).json(); const c=o.clock||{}, e=o.engine||{}, f=o.filters||{}, r=o.risk||{}, real=o.reality||{}, sh=o.nse_shadow||{};
 $('operatorSummary').innerHTML=(o.summary||[]).map(x=>`<span class=badge>${x}</span>`).join('');
 $('marketClock').innerHTML=[
  `<div class=miniCell><div class=label>US Market</div><strong>${dot(c.us_market_open)}${c.us_market_open?'Open':'Closed'}</strong><div class=sub>${c.now_ny||'--'}</div></div>`,
  `<div class=miniCell><div class=label>NSE Market</div><strong>${dot(c.nse_market_open)}${c.nse_market_open?'Open':'Closed'}</strong><div class=sub>${c.now_ist||'--'}</div></div>`,
  `<div class=miniCell><div class=label>US Screen</div><strong>${c.next_us_screen||'--'}</strong></div>`,
  `<div class=miniCell><div class=label>US Trade</div><strong>${c.next_us_trade||'--'}</strong></div>`,
  `<div class=miniCell><div class=label>NSE Daily</div><strong>${c.next_nse_daily||'--'}</strong></div>`
 ].join('');
 const p=e.processes||{}, la=e.log_age_seconds||{};
 $('engineStatus').innerHTML=kv([
  ['cron', e.cron_active?'active':(e.cron_status||'unknown'), e.cron_active?'green':'amber'],
  ['US trade proc', p.us_trade&&p.us_trade.running?'running':'scheduled/stopped', p.us_trade&&p.us_trade.running?'green':'amber'],
  ['US screen proc', p.us_screen&&p.us_screen.running?'running':'scheduled/stopped', p.us_screen&&p.us_screen.running?'green':'amber'],
  ['NSE daily proc', p.nse_daily&&p.nse_daily.running?'running':'scheduled/stopped', p.nse_daily&&p.nse_daily.running?'green':'amber'],
  ['dashboard log', age(la.dashboard), 'blue']
 ]);
 const actions=o.actions||[]; $('actionCount').textContent=actions.length+' events'; $('actionLog').innerHTML=actions.length?actions.map(x=>`<div class=event><b>${x.source}</b><span>${x.text}<br><small>${age(x.age_seconds)}</small></span></div>`).join(''):'<div class=empty>No recent events</div>';
 const usF=f.us||{}, nseF=f.nse||{}, usR=r.us||{}, nseR=r.nse||{}, usReal=real.us||{};
 $('usFilter').innerHTML=kv([['accepted signals', usF.raw_signals||0,'blue'],['open positions', usF.open_positions||0,'blue'],['capacity left', usF.capacity_left||0,'green'],['note', usF.note||'--','']]);
 $('usRisk').innerHTML=kv([['gross exposure', (usR.gross_exposure_pct||0).toFixed(2)+'%','blue'],['top 5 exposure', (usR.top5_exposure_pct||0).toFixed(2)+'%','blue'],['biggest position', (usR.biggest_position_pct||0).toFixed(2)+'%','amber'],['near stop', usR.near_stop||0,(usR.near_stop||0)>0?'red':'green'],['open losers', usR.losers||0,(usR.losers||0)>0?'amber':'green']]);
 $('usReality').innerHTML=kv([['live WR', usReal.live_win_rate==null?'--':Number(usReal.live_win_rate).toFixed(1)+'%','blue'],['backtest WR', '60.6%','blue'],['live closed P&L', money(usReal.live_closed_pnl||0),(usReal.live_closed_pnl||0)>=0?'green':'red'],['PT hit rate', usReal.pt_hit_rate==null?'--':Number(usReal.pt_hit_rate).toFixed(1)+'%','blue'],['closed sample', usReal.sample_trades||0,'amber']]);
 $('nseShadow').innerHTML=kv([['state', (sh.state||'--').toUpperCase(),'blue'],['gate fired', sh.gate_fired?'YES':'NO',sh.gate_fired?'red':'green'],['narrow score', sh.narrow_score==null?'--':Number(sh.narrow_score).toFixed(3),'amber'],['VIX', sh.vix==null?'--':Number(sh.vix).toFixed(2),'blue'],['Nifty 60d', sh.nifty_ret60==null?'--':Number(sh.nifty_ret60).toFixed(2)+'%',(sh.nifty_ret60||0)>=0?'green':'red']]);
 $('nseFilter').innerHTML=kv([['pre gate', nseF.pre_gate||0,'blue'],['gate blocked', nseF.gate_blocked||0,(nseF.gate_blocked||0)>0?'red':'green'],['post gate', nseF.post_gate||0,'blue'],['execution filtered', nseF.execution_filtered||0,(nseF.execution_filtered||0)>0?'amber':'green'],['post filter', nseF.post_filter||0,'green']]);
 $('nseRisk').innerHTML=kv([['gross notional', inr(nseR.gross_notional_inr||0),'blue'],['gross exposure', (nseR.gross_exposure_pct||0).toFixed(2)+'%','blue'],['top 5 exposure', (nseR.top5_exposure_pct||0).toFixed(2)+'%','amber'],['biggest position', (nseR.biggest_position_pct||0).toFixed(2)+'%','amber'],['cash', inr(nseR.cash_inr||0),'green']]);
 const alerts=o.alerts||[], usA=alerts.filter(x=>x.scope==='US'), nseA=alerts.filter(x=>x.scope==='NSE'); $('usAlertCount').textContent=usA.length; $('nseAlertCount').textContent=nseA.length; renderAlerts('usAlerts',usA); renderAlerts('nseAlerts',nseA);
 $('refreshState').textContent=autoRefresh?'auto 30s':'manual';
}
async function loadUS(){
 const snap=await (await fetch('/api/snapshot?ts='+Date.now())).json(); const p=snap.portfolio||{}, a=snap.analytics||{}, r=snap.regime||{}, s=snap.signals||{};
 $('pv').textContent=money(p.portfolio_value); tone($('pv'),p.total_return_pct); $('ret').textContent=pct(p.total_return_pct);
 $('cash').textContent=money(p.cash); $('cpnl').textContent=money(a.closed_pnl); tone($('cpnl'),a.closed_pnl); $('oc').textContent=p.open_positions_count||0;
 $('wr').textContent=a.win_rate==null?'--':Number(a.win_rate).toFixed(1)+'%'; $('tc').textContent=(a.closed_trades||0)+' closed - backtest 60.6%';
 $('dd').textContent=pct(p.drawdown_from_peak_pct); tone($('dd'),-Number(p.drawdown_from_peak_pct||0));
 $('volChip').textContent='VOL '+(r.vol_regime||'--')+' '+(r.vol_multiplier||'--')+'x'; $('spy').textContent=Number(r.spy_realized_vol||0).toFixed(2)+'%'; $('reg').textContent=r.vol_regime||'--'; $('mul').textContent=(r.vol_multiplier||'--')+'x'; $('desc').textContent=r.description||''; $('updated').textContent='updated '+new Date().toLocaleTimeString();
 $('sd').textContent=s.signal_date||'--'; const sig=s.signals||[]; $('sig').innerHTML=sig.length?sig.slice(0,6).map(x=>`<tr><td>${x.rank}</td><td><b>${x.symbol}</b></td><td>${Number(x.probability||0).toFixed(3)}</td><td>${money(x.entry_price)}</td><td class=green>${money(x.profit_target_price)}</td><td class=red>${money(x.stop_loss_price)}</td></tr>`).join(''):'<tr><td colspan=6 class=empty>No signals</td></tr>';
 const pos=p.positions||[]; $('pb').textContent=pos.length+' open'; $('pos').innerHTML=pos.length?pos.map(x=>{let c=Number(x.unrealized_pnl||0)>=0?'green':'red',prog=Math.min(100,(Number(x.days_held||0)/8)*100);return `<tr><td><b>${x.symbol}</b></td><td>${money(x.entry_price)}</td><td>${money(x.current_price)}</td><td class=green>${money(x.profit_target_price)}</td><td class=red>${money(x.stop_loss_price)}</td><td>${x.days_held}d</td><td><div class=bar><div class=fill style="width:${prog}%"></div></div></td><td class=${c}>${money(x.unrealized_pnl)}</td><td class=${c}>${pct(x.unrealized_pnl_pct)}</td><td>${Number(x.confidence||0).toFixed(3)}</td></tr>`}).join(''):'<tr><td colspan=10 class=empty>No open positions</td></tr>';
 const tr=a.trades||[]; $('cb').textContent=(a.closed_trades||0)+' closed'; $('tr').innerHTML=tr.length?tr.map(x=>{let c=Number(x.pnl||0)>=0?'green':'red';return `<tr><td>${x.exit_date||'--'}</td><td><b>${x.symbol}</b></td><td>${money(x.entry_price)}</td><td>${money(x.exit_price)}</td><td>${x.hold_days||0}d</td><td class=${c}>${pct(x.return_pct)}</td><td class=${c}>${money(x.pnl)}</td><td>${x.exit_reason||'--'}</td></tr>`}).join(''):'<tr><td colspan=8 class=empty>No closed trades</td></tr>';
 const up=pos.filter(x=>Number(x.unrealized_pnl||0)>0).length, down=pos.filter(x=>Number(x.unrealized_pnl||0)<0).length, total=pos.reduce((z,x)=>z+Number(x.unrealized_pnl||0),0); $('outBadge').textContent=(a.closed_trades||0)+' closed'; $('summaryBox').innerHTML=`<div class=metricLine><span class=label>Open Winners</span><b class=green>${up}</b></div><div class=metricLine><span class=label>Open Losers</span><b class=red>${down}</b></div><div class=metricLine><span class=label>Unrealized</span><b class="${total>=0?'green':'red'}">${money(total)}</b></div><div class=metricLine><span class=label>PT Hit Rate</span><b>${a.profit_target_hit_rate==null?'--':Number(a.profit_target_hit_rate).toFixed(1)+'%'}</b></div>`;
 const series=await (await fetch('/api/pnl?ts='+Date.now())).json(); renderCharts(series);
}
let pnlChart, portChart;
let chartSig='';
function renderCharts(d){ if(!window.Chart||!d.labels||d.labels.length<2)return; window.__lastSeries=d; const sig=JSON.stringify([d.labels,d.cumulative_pnl,d.portfolio_value,document.documentElement.dataset.theme]); $('pnlBadge').textContent=money(d.cumulative_pnl.at(-1)); $('portBadge').textContent=money(d.portfolio_value.at(-1)); if(sig===chartSig)return; chartSig=sig; if(pnlChart)pnlChart.destroy(); if(portChart)portChart.destroy(); const grid=css('--grid'), muted=css('--muted'), green=css('--green'), blue=css('--blue'); const common={responsive:true,maintainAspectRatio:false,animation:false,interaction:{mode:'index',intersect:false},plugins:{legend:{display:false},tooltip:{backgroundColor:css('--panel'),titleColor:css('--ink'),bodyColor:css('--ink'),borderColor:css('--line'),borderWidth:1}},scales:{x:{ticks:{color:muted,maxTicksLimit:6},grid:{color:grid}},y:{ticks:{color:muted},grid:{color:grid}}}}; pnlChart=new Chart($('pnlChart'),{type:'line',data:{labels:d.labels,datasets:[{data:d.cumulative_pnl,borderColor:green,backgroundColor:green+'22',fill:true,tension:.28,pointRadius:2,pointHoverRadius:5}]},options:common}); portChart=new Chart($('portChart'),{type:'line',data:{labels:d.labels,datasets:[{data:d.portfolio_value,borderColor:blue,backgroundColor:blue+'22',fill:true,tension:.28,pointRadius:2,pointHoverRadius:5}]},options:common});}
let nseSignalChart, nseExposureChart;
let nseChartSig='';
function renderNseCharts(signals, positions){
 if(!window.Chart)return;
 const sig=JSON.stringify([signals.map(x=>[x.symbol,x.prob]),positions.map(x=>[x.symbol,x.notional_inr]),document.documentElement.dataset.theme]);
 if(sig===nseChartSig)return;
 nseChartSig=sig;
 if(nseSignalChart)nseSignalChart.destroy();
 if(nseExposureChart)nseExposureChart.destroy();
 const grid=css('--grid'), muted=css('--muted'), blue=css('--blue'), green=css('--green'), amber=css('--amber');
 const common={responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{display:false}},scales:{x:{ticks:{color:muted,maxRotation:0},grid:{display:false}},y:{ticks:{color:muted},grid:{color:grid}}}};
 const s=signals.slice(0,12);
 nseSignalChart=new Chart($('nseSignalChart'),{type:'bar',data:{labels:s.map(x=>x.symbol),datasets:[{data:s.map(x=>Number(x.prob||0)),backgroundColor:s.map((_,i)=>i%2?blue:green),borderRadius:5}]},options:{...common,scales:{...common.scales,y:{min:.50,max:.60,ticks:{color:muted},grid:{color:grid}}}}});
 const p=positions.slice(0,10);
 nseExposureChart=new Chart($('nseExposureChart'),{type:'doughnut',data:{labels:p.map(x=>x.symbol),datasets:[{data:p.map(x=>Number(x.notional_inr||0)),backgroundColor:[green,blue,amber,'#8b5cf6','#14b8a6','#f97316','#06b6d4','#84cc16','#ec4899','#64748b'],borderWidth:1,borderColor:css('--panel')}]},options:{responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{position:'right',labels:{color:muted,boxWidth:10,font:{size:11}}}},cutout:'64%'}});
}
async function loadNSE(){try{
 const n=await (await fetch('/api/nse?ts='+Date.now())).json();
 const m=n.metrics||{}, gate=n.gate_decision||{}, state=gate.regime_state||((n.regime||'').includes('calm')?'clear':'crisis');
 $('nseDate').textContent=n.asof_date||'--';$('nseRegime').textContent=(n.regime||'--').toUpperCase();$('nseVix').textContent=Number(n.india_vix||0).toFixed(2);
 $('nseOpen').textContent=m.positions||0;$('nsePnl').textContent=inr(m.pnl||0);tone($('nsePnl'),m.pnl||0);$('nsePnl2').textContent=inr(m.pnl||0);tone($('nsePnl2'),m.pnl||0);
 $('nseGate').textContent=gate.gate_fired?'FIRED':'CLEAR';tone($('nseGate'),gate.gate_fired?-1:1);$('nseGateSub').textContent=gate.gate_reason||state||'shadow logger';
 $('nseMoneyBadge').textContent=inr(m.portfolio_value||1000000);$('nseInvested').textContent=inr(m.notional||0);$('nseCash').textContent=inr(m.cash||0);
 $('nseInvestedBar').style.width=Math.min(100,(Number(m.notional||0)/1000000)*100)+'%';$('nseCashBar').style.width=Math.min(100,(Number(m.cash||0)/1000000)*100)+'%';$('nsePnlBar').style.width=Math.min(100,Math.abs(Number(m.pnl||0))/1000000*1000)+'%';
 $('nseState').textContent=(state||'--').toUpperCase();$('nseGateFired').textContent=gate.gate_fired?'YES':'NO';$('nsePreGate').textContent=m.pre_gate_signals||0;$('nseBlocked').textContent=m.blocked_signals||0;$('nseShadowBadge').textContent=(m.blocked_signals||0)+' blocked';
 const sig=n.signals||[];$('nseSignalsBadge').textContent=sig.length+' signals';$('nseSignalsBadge2').textContent=sig.length+' signals';$('nseSignals').innerHTML=sig.length?sig.map(x=>`<tr><td><b>${x.symbol}</b></td><td>${Number(x.prob||0).toFixed(3)}</td><td>${inr(x.close)}</td><td>${inr(x.adv20_dollar_vol)}</td></tr>`).join(''):'<tr><td colspan=4 class=empty>No signals</td></tr>';
 const pos=n.positions||[];$('nsePositionsBadge').textContent=pos.length+' open';$('nsePositionsBadge2').textContent=pos.length+' open';$('nsePositions').innerHTML=pos.length?pos.map(x=>`<tr><td><b>${x.symbol}</b></td><td>${x.quantity||0}</td><td>${inr(x.entry_price)}</td><td>${inr(x.notional_inr)}</td><td>${Number(x.prob||0).toFixed(3)}</td></tr>`).join(''):'<tr><td colspan=5 class=empty>No positions</td></tr>';
 renderNseCharts(sig,pos);
}catch(e){console.warn('nse load failed',e)}}
let autoRefresh=true, refreshTimer=null;
async function load(){try{await Promise.all([loadUS(),loadNSE(),loadOperator()]);$('lastRefresh').textContent='last refresh '+new Date().toLocaleTimeString()}catch(e){console.error(e)}}
function scheduleRefresh(){if(refreshTimer)clearInterval(refreshTimer);refreshTimer=setInterval(()=>{if(autoRefresh)load()},30000)}
$('refreshNow').onclick=()=>load();
$('autoRefresh').onclick=()=>{autoRefresh=!autoRefresh;$('autoRefresh').textContent=autoRefresh?'Auto refresh on':'Auto refresh off';$('refreshState').textContent=autoRefresh?'auto 30s':'manual'};
load(); scheduleRefresh();
</script>
</body></html>
"""


@app.route("/")
def home():
    return render_template_string(HTML)


if __name__ == "__main__":
    print(f"MacroIntel Institutional Dashboard: http://0.0.0.0:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
