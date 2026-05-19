import json, os, math
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from core.paper_trading import VirtualBroker, Order, OrderSide

OUT = Path("reports/force_paper_trade_smoke.json")
TOP_N = int(os.getenv("FORCE_PAPER_TOP_N", "5"))
NOTIONAL_PCT = float(os.getenv("FORCE_PAPER_NOTIONAL_PCT", "0.002"))
ROOTS = [Path("data/features"), Path("data/features_26yr_liquid")]

def sf(x, d=0.0):
    try:
        if x is None or x != x:
            return d
        return float(x)
    except Exception:
        return d

def score_row(r):
    score = 0.0
    score += max(-2, min(2, sf(r.get("return_20d")) / 10.0))
    score += max(-2, min(2, sf(r.get("momentum_20d")) / 10.0))
    score += max(-2, min(2, sf(r.get("momentum_60d")) / 20.0))
    score += max(-1, min(1, sf(r.get("close_vs_sma_50")) / 10.0))
    score += max(-1, min(1, sf(r.get("alpha_signal"))))
    score += max(-1, min(1, sf(r.get("event_alpha_signal"))))
    score += max(-1, min(1, sf(r.get("weighted_sentiment_zscore")) / 2.0))
    rsi = sf(r.get("rsi_14"), 50)
    if 35 <= rsi <= 70:
        score += 0.5
    if sf(r.get("close")) >= 5:
        score += 0.5
    return score

rows = []
used_root = None
for root in ROOTS:
    if not root.exists():
        continue
    files = sorted(root.glob("*.parquet"))
    if not files:
        continue
    used_root = str(root)
    for p in files:
        try:
            df = pd.read_parquet(p)
            if df.empty or "close" not in df.columns:
                continue
            r = df.iloc[-1].to_dict()
            price = sf(r.get("close"))
            if price <= 5 or not math.isfinite(price):
                continue
            rows.append({
                "symbol": p.stem.replace("_US", "").replace(".US", ""),
                "price": price,
                "score": score_row(r),
            })
        except Exception:
            continue
    if rows:
        break

rows = sorted(rows, key=lambda x: x["score"], reverse=True)

broker = VirtualBroker(
    initial_capital=float(os.getenv("PAPER_INITIAL_CAPITAL", "100000")),
    max_position_pct=0.10,
    session_guardrails_enabled=False,
)

equity = float(getattr(broker, "portfolio_value", broker.cash))
orders = []

for x in rows[:TOP_N]:
    sym = x["symbol"]
    price = x["price"]
    if hasattr(broker, "has_open_position") and broker.has_open_position(symbol=sym):
        continue
    qty = max(1, int((equity * NOTIONAL_PCT) / price))
    order = Order(
        symbol=sym,
        side=OrderSide.BUY,
        quantity=qty,
        position_key=f"{sym}::normal",
        signal_source="force_feature_paper_smoke",
        metadata={
            "lane": "normal",
            "market": "US",
            "forced_feature_smoke": True,
            "score": x["score"],
            "tick_age_seconds": 0,
            "signal_age_seconds": 0,
            "spread_pct": 0.05,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    res = broker.submit_order(order, current_price=price)
    orders.append({"symbol": sym, "price": price, "qty": qty, "score": x["score"], "result": res})

getattr(broker, "_persist_state", lambda: None)()

out = {
    "used_root": used_root,
    "feature_candidates": len(rows),
    "orders_attempted": len(orders),
    "orders": orders,
    "paper_state": "data/paper_broker_state.json",
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, indent=2, default=str))
print(json.dumps(out, indent=2, default=str))
