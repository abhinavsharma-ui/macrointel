"""
Alpaca order execution bridge for MacroIntel.

Submits buy/sell orders to Alpaca (paper or live) alongside the existing
JSON position tracking in fixed_return_paper_execute.py.

Buy orders use TimeInForce.OPG (opening auction) — submitted after market
close, they fill at the next morning's open, matching the backtest assumption.

Sell orders use TimeInForce.DAY — submitted after market close, they queue
for next morning's open.

All failures are logged but never propagate — Alpaca errors must never
break the position tracking or signal pipeline.
"""
from __future__ import annotations

import os
import traceback


def _client():
    from alpaca.trading.client import TradingClient
    return TradingClient(
        api_key=os.getenv("ALPACA_API_KEY", ""),
        secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
        paper=os.getenv("ALPACA_PAPER", "true").lower() == "true",
    )


def _enabled() -> bool:
    return bool(os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"))


def submit_buy_at_open(symbol: str, position_pct: float, entry_price: float) -> bool:
    """
    Submit a market buy order sized by position_pct of current portfolio value.
    Uses TimeInForce.OPG so it fills at the next morning's opening auction.
    Falls back to notional order if qty < 1.
    Returns True on success, False on any failure.
    """
    if not _enabled():
        print(f"ALPACA BRIDGE: keys not set, skipping buy {symbol}")
        return False
    try:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        client = _client()
        portfolio_value = float(client.get_account().portfolio_value)
        dollar_amount = round(portfolio_value * position_pct, 2)

        if entry_price > 0:
            qty = int(dollar_amount / entry_price)
        else:
            qty = 0

        if qty >= 1:
            order = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.OPG,
            )
        else:
            # Fractional / notional fallback (DAY — OPG requires whole shares)
            order = MarketOrderRequest(
                symbol=symbol,
                notional=dollar_amount,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )

        result = client.submit_order(order)
        print(
            f"ALPACA BUY  {symbol:6s} "
            f"qty={qty if qty >= 1 else f'${dollar_amount:.0f} notional':>10} "
            f"pos={position_pct:.4f}  order_id={result.id}"
        )
        return True

    except Exception as e:
        print(f"ALPACA BUY  {symbol} FAILED: {e}")
        traceback.print_exc()
        return False


def submit_sell(symbol: str, reason: str = "") -> bool:
    """
    Close entire Alpaca position in symbol.
    Returns True on success, False on any failure (including position not found).
    """
    if not _enabled():
        print(f"ALPACA BRIDGE: keys not set, skipping sell {symbol}")
        return False
    try:
        client = _client()
        client.close_position(symbol)
        print(f"ALPACA SELL {symbol:6s} reason={reason or 'unknown'}")
        return True

    except Exception as e:
        # 422 = no position to close (already flat) — not an error worth alarming on
        if "422" in str(e) or "position does not exist" in str(e).lower():
            print(f"ALPACA SELL {symbol} — no open position (already flat or never filled)")
        else:
            print(f"ALPACA SELL {symbol} FAILED: {e}")
            traceback.print_exc()
        return False


def get_portfolio_value() -> float | None:
    """Return current Alpaca portfolio value, or None on failure."""
    if not _enabled():
        return None
    try:
        return float(_client().get_account().portfolio_value)
    except Exception as e:
        print(f"ALPACA get_portfolio_value FAILED: {e}")
        return None
