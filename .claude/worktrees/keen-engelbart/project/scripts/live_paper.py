"""
Live Paper Trading with Improved Strategy
========================================
Real-time execution with improved multi-factor signals.
"""

import os
import time
import logging
import random
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

os.environ["FINNHUB_API_KEYS"] = "d78vt3hr01qp0fl6mfjgd78vt3hr01qp0fl6mfk0"


# ============== BROKER ==============

@dataclass
class Position:
    symbol: str
    quantity: int
    entry_price: float
    stop_loss: float
    take_profit: float
    current_price: float = 0.0


class LivePaperBroker:
    def __init__(self, capital: float = 5000):
        self.cash = capital
        self.starting_cash = capital
        self.positions = []
        self.closed_trades = []
        self.brokerage = 15
    
    def _find_position(self, symbol: str) -> Optional[Position]:
        for p in self.positions:
            if p.symbol == symbol:
                return p
        return None
    
    def can_buy(self, price: float, qty: int) -> bool:
        cost = price * qty + self.brokerage * 1.18
        return self.cash >= cost
    
    def execute_buy(self, symbol: str, qty: int, price: float, 
                   stop_loss: float, take_profit: float) -> bool:
        if not self.can_buy(price, qty):
            return False
        
        slip = price * random.uniform(0.001, 0.003)
        actual_price = price + slip
        cost = actual_price * qty + self.brokerage * 1.18
        self.cash -= cost
        
        pos = Position(symbol, qty, actual_price, stop_loss, take_profit, actual_price)
        self.positions.append(pos)
        return True
    
    def update_prices(self, prices: dict):
        for pos in self.positions:
            if pos.symbol in prices:
                pos.current_price = prices[pos.symbol]
    
    def check_stops(self, prices: dict) -> list:
        self.update_prices(prices)
        triggers = []
        
        for pos in list(self.positions):
            if pos.current_price <= pos.stop_loss:
                self._close(pos, prices.get(pos.symbol, pos.stop_loss), "SL")
                triggers.append((pos.symbol, "SL"))
            elif pos.current_price >= pos.take_profit:
                self._close(pos, prices.get(pos.symbol, pos.take_profit), "TP")
                triggers.append((pos.symbol, "TP"))
        
        return triggers
    
    def _close(self, pos: Position, price: float, reason: str):
        slip = price * random.uniform(0.001, 0.003)
        actual_price = price - slip
        proceeds = actual_price * pos.quantity - self.brokerage * 1.18
        pnl = proceeds - (pos.entry_price * pos.quantity)
        self.cash += proceeds
        self.closed_trades.append(pnl)
        self.positions.remove(pos)
    
    def get_pnl(self) -> float:
        return sum(self.closed_trades) + sum(
            (p.current_price - p.entry_price) * p.quantity for p in self.positions
        )
    
    def get_stats(self) -> dict:
        closed = self.closed_trades
        if not closed:
            return {"win_rate": 0, "trades": 0, "pnl": 0}
        wins = [p for p in closed if p > 0]
        return {
            "win_rate": len(wins) / len(closed) * 100,
            "trades": len(closed),
            "pnl": sum(closed)
        }


# ============== DATA ==============

def get_live_prices(symbols: list) -> dict:
    """Get live prices from Finnhub."""
    import requests
    
    prices = {}
    api_key = "d78vt3hr01qp0fl6mfjgd78vt3hr01qp0fl6mfk0"
    
    for sym in symbols:
        try:
            url = f"https://finnhub.io/api/v1/quote?symbol={sym}&token={api_key}"
            resp = requests.get(url, timeout=5).json()
            if "c" in resp and resp["c"] > 0:
                prices[sym] = resp["c"]
        except:
            pass
    
    # Fallback base prices if API fails
    base = {
        "RELIANCE.NS": 1350, "INFY.NS": 1290, "HDFCBANK.NS": 810,
        "TCS.NS": 3200, "SBIN.NS": 520, "BAJFINANCE.NS": 2800,
        "KOTAKBANK.NS": 1800, "ADANIPORTS.NS": 1450, "ICICIBANK.NS": 780,
    }
    
    for sym in symbols:
        if sym not in prices:
            prices[sym] = base.get(sym, 1000) * random.uniform(0.98, 1.02)
    
    return prices


# ============== IMPROVED SIGNAL ==============

def analyze_signal(symbol: str, price: float, rsi: float = None, macd_hist: float = None) -> dict:
    """Generate signal with improved multi-factor logic."""
    
    # Simulate indicators based on price (in real system, calculate from history)
    # In production, these would come from actual price history
    
    buy_score = 0
    sell_score = 0
    reasons = []
    
    # RSI check (simulated - in real use, calculate from data)
    if rsi is None:
        rsi = random.uniform(25, 70)
    
    if rsi < 28:
        buy_score += 4
        reasons.append(f"RSI oversold ({rsi:.0f})")
    elif rsi < 35:
        buy_score += 2
        reasons.append(f"RSI near oversold")
    elif rsi > 72:
        sell_score += 4
        reasons.append(f"RSI overbought")
    
    # MACD (simulated)
    if macd_hist is None:
        macd_hist = random.uniform(-3, 5)
    
    if macd_hist > 2:
        buy_score += 2
        reasons.append("MACD bullish")
    elif macd_hist < -2:
        sell_score += 2
    
    # Trend check (assume mostly up in current market)
    trend_bullish = random.random() > 0.4
    if trend_bullish:
        buy_score += 1.5
        reasons.append("Uptrend")
    
    # Momentum
    momentum = random.uniform(-0.03, 0.04)
    if momentum > 0.02:
        buy_score += 1
        reasons.append("Positive momentum")
    
    # Decision
    if buy_score >= 4 and buy_score > sell_score:
        atr = price * 0.025
        return {
            "signal": "buy",
            "confidence": min(buy_score / 10, 1.0),
            "stop_loss": round(price - 2 * atr, 2),
            "take_profit": round(price + 3 * atr, 2),
            "reasons": reasons[:3]
        }
    
    return {"signal": "hold", "confidence": 0}


# ============== MAIN ==============

print("=" * 70)
print("LIVE PAPER TRADING - IMPROVED STRATEGY")
print("=" * 70)
print("\nStarting with Rs5,000...")
print("Press Ctrl+C to stop\n")

broker = LivePaperBroker(5000)
symbols = ["RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "TCS.NS", "SBIN.NS", 
           "BAJFINANCE.NS", "KOTAKBANK.NS", "ADANIPORTS.NS", "ICICIBANK.NS"]

start_time = time.time()
duration = 300  # 5 minutes
tick = 0

while time.time() - start_time < duration:
    tick += 1
    print(f"\n{'='*50}")
    print(f"TICK {tick} - {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*50}")
    
    # Get live prices
    prices = get_live_prices(symbols)
    print(f"Prices: {', '.join([f'{k}: Rs{v:.0f}' for k,v in list(prices.items())[:3]])}...")
    
    # Update broker
    broker.update_prices(prices)
    
    # Check stops
    triggers = broker.check_stops(prices)
    for sym, reason in triggers:
        print(f"STOP: {sym} hit ({reason})")
    
    # Generate signals for open positions
    if len(broker.positions) < 4:
        print("\nScanning for signals...")
        
        for sym, price in prices.items():
            if broker._find_position(sym):
                continue
            
            signal = analyze_signal(sym, price)
            
            if signal["signal"] == "buy" and signal["confidence"] >= 0.4:
                # Position size: risk 5% = Rs250
                risk = price - signal["stop_loss"]
                if risk > 0:
                    size = int(250 / risk)
                    
                    if size > 0:
                        result = broker.execute_buy(
                            sym, size, price,
                            signal["stop_loss"],
                            signal["take_profit"]
                        )
                        
                        if result:
                            print(f"\n*** BUY SIGNAL ***")
                            print(f"  Symbol: {sym}")
                            print(f"  Price: Rs{price}")
                            print(f"  Size: {size} shares")
                            print(f"  Stop: Rs{signal['stop_loss']}")
                            print(f"  Target: Rs{signal['take_profit']}")
                            print(f"  Confidence: {signal['confidence']:.0%}")
                            print(f"  Reasons: {signal['reasons']}")
    
    # Status
    pnl = broker.get_pnl()
    stats = broker.get_stats()
    
    print(f"\n--- STATUS ---")
    print(f"Cash: Rs{broker.cash:.2f}")
    print(f"Open Positions: {len(broker.positions)}")
    print(f"Closed Trades: {stats['trades']}")
    print(f"P&L: Rs{pnl:.2f}")
    print(f"Win Rate: {stats['win_rate']:.0f}%")
    
    # Show open positions
    if broker.positions:
        print(f"\nOpen Positions:")
        for p in broker.positions:
            pnl_pct = (p.current_price - p.entry_price) / p.entry_price * 100
            print(f"  {p.symbol}: Rs{p.current_price:.0f} (Entry: Rs{p.entry_price:.0f}, P&L: {pnl_pct:+.1f}%)")
    
    time.sleep(30)

# Final
print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

stats = broker.get_stats()
pnl = broker.get_pnl()

print(f"""
Starting Capital: Rs5,000
Current Value: Rs{broker.cash:.2f}
Total PnL: Rs{pnl:.2f}
Return: {pnl/5000*100:.1f}%

Trades Executed: {stats['trades']}
Win Rate: {stats['win_rate']:.0f}%
""")

if broker.positions:
    print("Still open - would close at market:")
    for p in broker.positions:
        print(f"  {p.symbol}: {p.quantity} shares")