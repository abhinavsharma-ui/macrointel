"""
Live Paper Trading v7 - Simple Version
"""

import os
import time
import logging
import requests
import random
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

API_KEY = "d78vt3hr01qp0fl6mfjgd78vt3hr01qp0fl6mfk0"

@dataclass
class Position:
    symbol: str
    quantity: int
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_date: str
    current_price: float = 0.0

class LivePaperBroker:
    def __init__(self, capital: float = 5000):
        self.cash = capital
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
        pos = Position(symbol, qty, actual_price, stop_loss, take_profit, 
                       datetime.now().strftime("%Y-%m-%d"), actual_price)
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
    
    def get_stats(self) -> dict:
        closed = self.closed_trades
        if not closed:
            return {"win_rate": 0, "trades": 0, "pnl": 0}
        wins = [p for p in closed if p > 0]
        return {"win_rate": len(wins)/len(closed)*100, "trades": len(closed), "pnl": sum(closed)}


def get_live_price(symbol: str) -> float:
    """Get single live price."""
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={API_KEY}"
        resp = requests.get(url, timeout=3).json()
        if "c" in resp and resp["c"] > 0:
            return resp["c"]
    except:
        pass
    return None


def get_signal_data(symbol: str) -> dict:
    """Get signal for symbol."""
    import yfinance as yf
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="200d", interval="1d")
        df = df.dropna(subset=['Close'])
        
        if len(df) < 30:
            return None
        
        close = df["Close"]
        volume = df["Volume"]
        price = close.iloc[-1]
        
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_current = rsi.iloc[-1]
        
        if pd.isna(rsi_current) or rsi_current >= 38:
            return None
        
        vol_ma20 = volume.rolling(20).mean().iloc[-1]
        volume_ratio = volume.iloc[-1] / vol_ma20 if vol_ma20 > 0 else 1
        
        return {
            "price": price,
            "rsi": rsi_current,
            "volume_ratio": volume_ratio,
            "stop_loss": round(price * 0.94, 2),
            "take_profit": round(price * 1.18, 2)
        }
    except:
        return None


# ============== MAIN ==============

print("=" * 70)
print("LIVE PAPER TRADING - v7")
print("=" * 70)

broker = LivePaperBroker(5000)
symbols = ["AJANTPHARM.NS", "DRREDDY.NS", "SUNPHARMA.NS", "RELIANCE.NS", "INFY.NS"]

print("Starting with Rs5,000...\n")

# Pre-load signals
print("Analyzing stocks...")
stock_data = {}
for sym in symbols:
    data = get_signal_data(sym)
    if data:
        stock_data[sym] = data
        print(f"  {sym}: RSI={data['rsi']:.0f}, Price=Rs{data['price']:.0f}")

print(f"\nSignals found: {len(stock_data)}")

# Try to execute
print("\nTrying to execute trades...")

# Try live price first, then use historical
for sym, data in stock_data.items():
    live_price = get_live_price(sym)
    price = live_price if live_price else data['price']
    
    risk = price - data['stop_loss']
    if risk > 0:
        size = int(250 / risk)
        
        if size > 0:
            result = broker.execute_buy(sym, size, price, data['stop_loss'], data['take_profit'])
            if result:
                print(f"*** BOUGHT {sym} ***")
                print(f"  Price: Rs{price:.0f}, Size: {size}, Stop: Rs{data['stop_loss']}, Target: Rs{data['take_profit']}")

# Simulate some time passing
print(f"\nCash: Rs{broker.cash:.2f}")
print(f"Positions: {len(broker.positions)}")

if broker.positions:
    print("\nOpen Positions:")
    for p in broker.positions:
        print(f"  {p.symbol}: {p.quantity} shares @ Rs{p.entry_price}")

print("\nDone! Paper trading setup complete.")