from __future__ import annotations
import os, traceback
from pathlib import Path

try:
    from dotenv import load_dotenv

    here = Path(__file__).resolve()
    seen = set()
    for env_path in (
        Path.cwd() / ".env",
        here.parent.parent / ".env",
        here.parent.parent.parent / ".env",
    ):
        env_path = env_path.resolve()
        if env_path in seen:
            continue
        seen.add(env_path)
        if env_path.exists():
            load_dotenv(env_path, override=False)
        if os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"):
            break
except Exception as exc:
    print(f"ALPACA BRIDGE: dotenv load skipped: {exc}")

def _client():
    from alpaca.trading.client import TradingClient
    return TradingClient(
        api_key=os.getenv("ALPACA_API_KEY", ""),
        secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
        paper=os.getenv("ALPACA_PAPER", "true").lower() == "true",
    )

def _enabled():
    return bool(os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"))

def submit_buy_at_open(symbol, position_pct, entry_price):
    if not _enabled():
        print(f"ALPACA BRIDGE: keys not set, skipping buy {symbol}")
        return False
    try:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        client = _client()
        portfolio_value = float(client.get_account().portfolio_value)
        dollar_amount = round(portfolio_value * position_pct, 2)
        qty = int(dollar_amount / entry_price) if entry_price > 0 else 0
        if qty >= 1:
            order = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.OPG)
        else:
            order = MarketOrderRequest(symbol=symbol, notional=dollar_amount, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
        result = client.submit_order(order)
        print(f"ALPACA BUY {symbol} qty={qty if qty >= 1 else f'${dollar_amount:.0f} notional'} order_id={result.id}")
        return True
    except Exception as e:
        print(f"ALPACA BUY {symbol} FAILED: {e}")
        traceback.print_exc()
        return False

def submit_sell(symbol, reason=""):
    if not _enabled():
        print(f"ALPACA BRIDGE: keys not set, skipping sell {symbol}")
        return False
    try:
        _client().close_position(symbol)
        print(f"ALPACA SELL {symbol} reason={reason or 'unknown'}")
        return True
    except Exception as e:
        if "422" in str(e) or "position does not exist" in str(e).lower():
            print(f"ALPACA SELL {symbol} — no open position (already flat)")
        else:
            print(f"ALPACA SELL {symbol} FAILED: {e}")
            traceback.print_exc()
        return False
