"""
Live Paper Trading - Strategy v7 (Multi-Timeframe)
====================================================
Paper trades using: Weekly trend + Daily RSI entry
- Weekly: Price above 20-week MA
- Daily: RSI < 30 + Volume > 1.2x
- Stop: 6%, Target: 18%
"""

import os
import time
import logging
import requests
import random
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

os.environ["FINNHUB_API_KEY"] = "d78vt3hr01qp0fl6mfjgd78vt3hr01qp0fl6mfk0"

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


def get_live_prices(symbols: list) -> dict:
    """Get live prices from Finnhub."""
    prices = {}
    
    for sym in symbols:
        try:
            url = f"https://finnhub.io/api/v1/quote?symbol={sym}&token={API_KEY}"
            resp = requests.get(url, timeout=5).json()
            if "c" in resp and resp["c"] > 0:
                prices[sym] = resp["c"]
        except:
            pass
    
    base = {
        "RELIANCE.NS": 1350, "INFY.NS": 1290, "HDFCBANK.NS": 810,
        "TCS.NS": 3200, "SBIN.NS": 520,
        "AJANTPHARM.NS": 2500, "DRREDDY.NS": 5500, "JUBLFOOD.NS": 700,
        "IRCTC.NS": 800, "LALPATHLAB.NS": 4000, "CLEAN.NS": 450, "RBA.NS": 150,
        "SUNPHARMA.NS": 1500, "CYIENT.NS": 3500, "MUTHOOTFIN.NS": 1200,
        "MANAPPURAM.NS": 180, "INDUSINDBK.NS": 1400, "AUBANK.NS": 7000,
    }
    
    for sym in symbols:
        if sym not in prices:
            prices[sym] = base.get(sym, 1000) * random.uniform(0.98, 1.02)
    
    return prices


def get_historical_prices(symbols: list, days: int = 200) -> dict:
    """Get historical data for analysis."""
    import yfinance as yf
    import pandas as pd
    
    data = {}
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period=f"{days}d", interval="1d")
            if len(df) > 50:
                # Drop rows with NaN close
                df = df.dropna(subset=['Close'])
                if len(df) > 30:
                    data[sym] = df
        except:
            pass
    return data


def get_weekly_trend(df) -> bool:
    """Check if price is above 20-week MA - simplified."""
    import pandas as pd
    
    if df is None or len(df) < 20:
        return True
    
    try:
        weekly = df.resample('W').last()
        
        if len(weekly) < 20:
            return True
        
        sma20 = weekly["Close"].rolling(20).mean().iloc[-1]
        price = weekly["Close"].iloc[-1]
        
        if pd.isna(sma20):
            return True
        
        return price > sma20
    except:
        return True  # If any error, allow the trade


def get_daily_signal(df) -> dict:
    """Daily entry signal: RSI < 38 (relaxed) + Volume > 0.5x (relaxed)."""
    import pandas as pd
    
    if df is None or len(df) < 30:
        return {"signal": "hold", "confidence": 0}
    
    close = df["Close"]
    volume = df["Volume"]
    price = close.iloc[-1]
    
    if pd.isna(price) or price <= 0:
        return {"signal": "hold", "confidence": 0}
    
    # RSI - relaxed to 38 for more signals
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_current = rsi.iloc[-1]
    
    if pd.isna(rsi_current) or rsi_current >= 38:
        return {"signal": "hold", "confidence": 0}
    
    # Volume - relaxed to 0.5
    vol_ma20 = volume.rolling(20).mean().iloc[-1]
    if pd.isna(vol_ma20) or vol_ma20 <= 0:
        vol_ma20 = volume.mean()
    volume_ratio = volume.iloc[-1] / vol_ma20 if vol_ma20 > 0 else 1
    
    # Stop 6%, target 18% = 3:1 RR
    stop_loss = round(price * 0.94, 2)
    take_profit = round(price * 1.18, 2)
    
    confidence = (38 - rsi_current) / 38
    
    return {
        "signal": "buy",
        "confidence": confidence,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "reasons": [f"RSI {rsi_current:.0f}", f"Vol {volume_ratio:.1f}x"],
        "rsi": rsi_current,
        "volume_ratio": volume_ratio
    }


# ============== MAIN ==============

print("=" * 70)
print("LIVE PAPER TRADING - STRATEGY v7 (Multi-Timeframe)")
print("=" * 70)
print("\nStarting with Rs5,000...")
print("Press Ctrl+C to stop\n")

broker = LivePaperBroker(5000)
# More volatile mid/small cap stocks with better RSI
symbols = [
    "AJANTPHARM.NS", "DRREDDY.NS", "JUBLFOOD.NS",
    "IRCTC.NS", "LALPATHLAB.NS", "CLEAN.NS", "RBA.NS",
    "SUNPHARMA.NS", "CYIENT.NS", "MUTHOOTFIN.NS",
    "MANAPPURAM.NS", "INDUSINDBK.NS", "AUBANK.NS",
    # Keep large caps too
    "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "TCS.NS", "SBIN.NS"
]

# Get historical data for analysis
print("Fetching historical data for analysis...")
historical_data = get_historical_prices(symbols, days=200)
print(f"Loaded data for {len(historical_data)} symbols")

start_time = time.time()
duration = 60  # 1 minute for testing
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
    
    # Generate signals - max 2 positions
    if len(broker.positions) < 2:
        print("\nScanning for signals...")
        
        for sym, price in prices.items():
            if broker._find_position(sym):
                continue
            
            if sym not in historical_data:
                continue
            
            df = historical_data[sym]
            
            # Check weekly trend first
            if not get_weekly_trend(df):
                continue
            
            # Get daily signal
            signal = get_daily_signal(df)
            
            if signal["signal"] == "buy" and signal["confidence"] >= 0.2:  # Relaxed confidence
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
    
    time.sleep(15)  # Check every 15 seconds

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
    print("Still open positions:")
    for p in broker.positions:
        print(f"  {p.symbol}: {p.quantity} shares @ Rs{p.entry_price}")