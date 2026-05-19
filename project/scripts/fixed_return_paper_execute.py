import argparse, csv, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
try:
    from dotenv import load_dotenv
    load_dotenv(Path(".env"), override=False)
    load_dotenv(Path(".env.example"), override=False)
except Exception:
    pass
try:
    from alpaca_bridge import submit_buy_at_open, submit_sell as alpaca_sell
    ALPACA_ENABLED = True
except ImportError:
    ALPACA_ENABLED = False
    print("ALPACA BRIDGE: alpaca_bridge not found, orders skipped")
try:
    import sl_verifier
except Exception:
    sl_verifier = None
try:
    import portfolio_risk
except Exception:
    portfolio_risk = None

SIGNALS_PATH = Path(os.getenv("FR_SIGNALS_PATH", "reports/fixed_return_daily_signals.json"))
POSITIONS_PATH = Path(os.getenv("FR_POSITIONS_PATH", "reports/fixed_return_open_positions.json"))
INITIAL_CAPITAL_PNL = float(os.getenv('FR_INITIAL_CAPITAL', '100000'))
TRADES_CSV = Path(os.getenv("FR_TRADES_CSV", "reports/fixed_return_paper_trades.csv"))
LIVE_ROOT = Path(os.getenv("SIG_LIVE_ROOT", "data/features"))
SYMBOL_BLOCKLIST_FILES = [
    Path(os.getenv("SIG_SYMBOL_BLOCKLIST", "data/blocklist.txt")),
    Path(os.getenv("SIG_ETF_BLOCKLIST", "data/etf_blocklist.txt")),
    Path(os.getenv("SIG_LEVERAGED_ETF_BLACKLIST", "data/leveraged_etf_blacklist.txt")),
]
MAX_OPEN = int(os.getenv("FR_MAX_OPEN_POSITIONS", "50"))
FR_DISABLE_STOP_LOSS = os.getenv("FR_DISABLE_STOP_LOSS", "0").lower() in {"1", "true", "yes", "on"} or float(os.getenv("SIG_STOP_LOSS_PCT", "3") or 0) <= 0
NY = ZoneInfo("America/New_York")

def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)

def norm_sym(s):
    return str(s).replace("_US", "").replace(".US", "").upper()

def load_blocked_symbols():
    blocked = set()
    for path in SYMBOL_BLOCKLIST_FILES:
        try:
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                raw = line.split("#", 1)[0].strip()
                if raw:
                    blocked.add(norm_sym(raw))
        except Exception as exc:
            print(f"WARN blocklist read failed {path}: {exc}")
    return blocked

def read_feature_file(path):
    df = pd.read_parquet(path)
    if df.empty or "close" not in df.columns:
        return None
    idx = pd.to_datetime(df.index, errors="coerce")
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df.index = pd.RangeIndex(len(df))
    if "date" in df.columns:
        dc = pd.to_datetime(df["date"], errors="coerce")
        df["date"] = dc if dc.notna().sum() >= idx.notna().sum() else idx
    else:
        df["date"] = idx
    return df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

def latest_price_row(symbol):
    for p in [LIVE_ROOT/f"{symbol}.parquet", LIVE_ROOT/f"{symbol}_US.parquet", LIVE_ROOT/f"{symbol}.US.parquet"]:
        if p.exists():
            df = read_feature_file(p)
            if df is None or df.empty:
                return None
            r = df.iloc[-1]
            close = float(r.get("close", 0) or 0)
            return {
                "close": close,
                "high": float(r.get("high", close) or close),
                "low": float(r.get("low", close) or close),
                "date": str(pd.Timestamp(r["date"]).date()),
            }
    return None

def load_positions():
    if not POSITIONS_PATH.exists():
        return []
    try:
        return json.load(open(POSITIONS_PATH)).get("positions", [])
    except Exception:
        return []

def write_trade(row):
    TRADES_CSV.parent.mkdir(parents=True, exist_ok=True)
    exists = TRADES_CSV.exists()
    fields = ["closed_at","symbol","entry_date","exit_date","exit_reason","entry_price","exit_price","position_pct","pnl_pct","pnl_contribution_pct","pnl","probability"]
    if exists:
        try:
            with open(TRADES_CSV, newline="", encoding="utf-8") as fh:
                for old in csv.DictReader(fh):
                    same = (
                        norm_sym(old.get("symbol")) == norm_sym(row.get("symbol"))
                        and str(old.get("entry_date") or "") == str(row.get("entry_date") or "")
                        and str(old.get("exit_reason") or "") == str(row.get("exit_reason") or "")
                        and round(float(old.get("exit_price") or 0), 4) == round(float(row.get("exit_price") or 0), 4)
                    )
                    if same:
                        print(f"SKIP duplicate trade row {row.get('symbol')} {row.get('exit_reason')} {row.get('exit_price')}")
                        return
        except Exception as exc:
            print(f"WARN duplicate trade check failed: {exc}")
    with open(TRADES_CSV, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})

def safe_alpaca_sell(symbol, reason):
    if not ALPACA_ENABLED:
        return
    try:
        alpaca_sell(symbol, reason)
    except Exception as exc:
        print(f"WARN alpaca sell failed for {symbol}: {exc}")

def safe_alpaca_buy(position):
    if not ALPACA_ENABLED:
        return
    try:
        submit_buy_at_open(position["symbol"], position["position_pct"], position["entry_price"])
    except Exception as exc:
        print(f"WARN alpaca buy failed for {position.get('symbol')}: {exc}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    today = datetime.now(NY).date().isoformat()
    payload = {"signals": []}
    allow_new_entries = False
    if SIGNALS_PATH.exists():
        payload = json.load(open(SIGNALS_PATH))
        allow_new_entries = payload.get("signal_date") == today
        if not allow_new_entries:
            print(f"WARN stale signals: signal_date={payload.get('signal_date')} today={today}. Exits still active; new entries blocked.")
    else:
        print(f"WARN signals file not found: {SIGNALS_PATH}. Exits still active; new entries blocked.")

    positions = load_positions()
    blocked_symbols = load_blocked_symbols()
    open_positions = [p for p in positions if p.get("status") == "open"]
    closed_today, still_open = [], []

    for pos in open_positions:
        sym = norm_sym(pos["symbol"])
        px = latest_price_row(sym)
        if sym in blocked_symbols:
            exit_price = float((px or {}).get("close") or pos.get("entry_price") or 0)
            pnl_pct = (exit_price / float(pos["entry_price"]) - 1.0) * 100.0 if float(pos.get("entry_price") or 0) else 0.0
            row = {
                **pos, "status": "closed", "closed_at": datetime.now(timezone.utc).isoformat(),
                "exit_reason": "blocked_symbol_etf", "exit_price": round(exit_price, 4),
                "pnl_pct": round(pnl_pct, 4),
                "pnl_contribution_pct": round(pnl_pct * float(pos.get("position_pct") or 0), 6),
                "pnl": round(pnl_pct * float(pos.get("position_pct") or 0) * INITIAL_CAPITAL_PNL / 100, 2),
            }
            closed_today.append(row)
            if not args.dry_run:
                write_trade(row)
                safe_alpaca_sell(sym, "blocked_symbol_etf")
            continue
        if px is None:
            still_open.append(pos)
            continue
        reason = None
        exit_price = None
        entry_price = float(pos.get("entry_price") or 0)
        stop_price = float(pos.get("stop_loss_price") or 0)
        target_price = float(pos["profit_target_price"])
        stop_enabled = (not FR_DISABLE_STOP_LOSS) and stop_price > 0 and (entry_price <= 0 or stop_price < entry_price)
        stop_hit = stop_enabled and (px["low"] <= stop_price or px["close"] <= stop_price)
        if stop_hit and sl_verifier and sl_verifier.should_skip_mechanical_stop(pos, px, today):
            pos["last_price"] = px["close"]
            pos["last_price_date"] = px["date"]
            pos["sl_mechanical_stop_skipped"] = datetime.now(timezone.utc).isoformat()
            still_open.append(pos)
            continue
        if stop_enabled and px["low"] <= stop_price and px["high"] >= target_price:
            reason, exit_price = "both_hit_stop_first", stop_price
        elif stop_hit:
            reason, exit_price = "stop_loss", stop_price
        elif px["high"] >= target_price or px["close"] >= target_price:
            reason, exit_price = "profit_target", float(pos["profit_target_price"])
        elif today >= str(pos["exit_date"]):
            reason, exit_price = "time_exit", px["close"]

        if reason:
            pnl_pct = (exit_price / float(pos["entry_price"]) - 1.0) * 100.0
            row = {
                **pos, "status": "closed", "closed_at": datetime.now(timezone.utc).isoformat(),
                "exit_reason": reason, "exit_price": round(exit_price, 4),
                "pnl_pct": round(pnl_pct, 4),
                "pnl_contribution_pct": round(pnl_pct * float(pos["position_pct"]), 6),
            "pnl": round(pnl_pct * float(pos["position_pct"]) * INITIAL_CAPITAL_PNL / 100, 2),
            }
            closed_today.append(row)
            if not args.dry_run:
                write_trade(row)
                safe_alpaca_sell(norm_sym(pos["symbol"]), reason)
        else:
            pos["last_price"] = px["close"]
            pos["last_price_date"] = px["date"]
            still_open.append(pos)

    open_symbols = {norm_sym(p["symbol"]) for p in still_open if p.get("status") == "open"}
    closed_symbols = {norm_sym(p["symbol"]) for p in closed_today}
    new_positions = []
    if not allow_new_entries:
        payload = {"signals": []}
    for sig in payload.get("signals", []):
        sym = norm_sym(sig["symbol"])
        if sym in blocked_symbols:
            print(f"SKIP blocked signal {sym}")
            continue
        if sym in open_symbols or sym in closed_symbols:
            continue
        if len(still_open) + len(new_positions) >= MAX_OPEN:
            break
        candidate_position = {
            "symbol": sym, "entry_date": today, "entry_price": float(sig["entry_price"]),
            "profit_target_price": float(sig["profit_target_price"]),
            "stop_loss_price": float(sig.get("stop_loss_price") or 0),
            "stop_loss_enabled": (not FR_DISABLE_STOP_LOSS) and bool(sig.get("stop_loss_enabled", float(sig.get("stop_loss_price") or 0) > 0)),
            "exit_date": sig["expected_exit_date"],
            "position_pct": float(sig["position_pct"]),
            "probability": float(sig["probability"]),
            "llm_decision": sig.get("llm_decision"),
            "llm_reason": sig.get("llm_reason"),
            "event_type": sig.get("event_type"),
            "event_confidence": sig.get("event_confidence"),
            "factor_composite": sig.get("factor_composite"),
            "hold_days": int(sig["hold_days"]),
            "status": "open",
            "source": "fixed_return_h8_pt5_volscale",
        }
        if portfolio_risk:
            ok, risk_reason, _risk_state = portfolio_risk.approve_new_signal(sig, still_open, new_positions)
            if not ok:
                print(f"SKIP risk blocked {sym}: {risk_reason}")
                continue
        new_positions.append(candidate_position)

    updated = [p for p in positions if p.get("status") != "open"] + still_open + new_positions
    out = {"updated_at": datetime.now(timezone.utc).isoformat(), "positions": updated}

    print(f"OPEN POSITIONS BEFORE {len(open_positions)}")
    print(f"CLOSED TODAY {len(closed_today)}")
    print(f"NEW POSITIONS OPENED {len(new_positions)}")
    print(f"OPEN POSITIONS AFTER {len(still_open) + len(new_positions)}")
    print(f"RUNNING CLOSED PNL CONTRIBUTION PCT {sum(float(x.get('pnl_contribution_pct',0) or 0) for x in closed_today):.4f}")
    for p in new_positions:
        sl_text = f"{p['stop_loss_price']:.2f}" if p.get("stop_loss_enabled") else "OFF"
        print(f"OPEN {p['symbol']} entry={p['entry_price']:.2f} PT={p['profit_target_price']:.2f} SL={sl_text} pos={p['position_pct']:.4f}")

    if args.dry_run:
        print("DRY RUN: no files written")
        return
    POSITIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    POSITIONS_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    if ALPACA_ENABLED:
        for p in new_positions:
            safe_alpaca_buy(p)
    if portfolio_risk:
        try:
            portfolio_risk.build_risk_state(updated)
        except Exception as exc:
            print(f"WARN risk state write failed: {exc}")
    print(f"POSITIONS WRITTEN TO {POSITIONS_PATH}")

if __name__ == "__main__":
    main()
