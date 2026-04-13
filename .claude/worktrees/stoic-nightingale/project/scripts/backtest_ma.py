"""
MA Crossover + Trend Strategy
==============================
Classic profitable strategy:
1. Golden Cross: 50-day MA crosses above 200-day MA (long-term trend)
2. Entry: Price above 20-day MA + momentum
3. Stop: Below 20-day MA (simple trailing)
4. Target: 8%
"""

import os
import logging
import yfinance as yf
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")


@dataclass
class Position:
    symbol: str
    quantity: int
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_date: str
    current_price: float = 0.0


class BacktestBroker:
    def __init__(self, capital: float = 5000):
        self.cash = capital
        self.positions: List[Position] = []
        self.closed_trades: List[Dict] = []
        self.brokerage = 15
    
    def _find_position(self, symbol: str) -> Optional[Position]:
        for p in self.positions:
            if p.symbol == symbol:
                return p
        return None
    
    def execute_buy(self, symbol: str, qty: int, price: float,
                    stop_loss: float, take_profit: float, entry_date: str) -> bool:
        cost = price * qty + self.brokerage * 1.18
        if self.cash < cost:
            return False
        self.cash -= cost
        pos = Position(symbol, qty, price, stop_loss, take_profit, entry_date, price)
        self.positions.append(pos)
        return True
    
    def close_position(self, pos: Position, price: float, exit_date: str, reason: str):
        proceeds = price * pos.quantity - self.brokerage * 1.18
        pnl = proceeds - (pos.entry_price * pos.quantity)
        self.cash += proceeds
        self.closed_trades.append({
            "symbol": pos.symbol, "entry_date": pos.entry_date, "exit_date": exit_date,
            "entry_price": pos.entry_price, "exit_price": price, "quantity": pos.quantity,
            "pnl": pnl, "pnl_pct": pnl / (pos.entry_price * pos.quantity) * 100, "reason": reason
        })
        self.positions.remove(pos)
    
    def check_exits(self, prices: dict, current_date: str):
        for pos in list(self.positions):
            if pos.symbol in prices:
                price = prices[pos.symbol]
                if price <= pos.stop_loss:
                    self.close_position(pos, price, current_date, "SL")
                elif price >= pos.take_profit:
                    self.close_position(pos, price, current_date, "TP")
    
    def get_summary(self) -> dict:
        closed = self.closed_trades
        if not closed:
            return {"cash": self.cash, "total_value": self.cash, "total_pnl": 0,
                    "win_rate": 0, "trades": 0, "profit_factor": 0, "avg_win": 0, "avg_loss": 0,
                    "best_trade": 0, "worst_trade": 0}
        wins = [t for t in closed if t["pnl"] > 0]
        losses = [t for t in closed if t["pnl"] <= 0]
        win_rate = len(wins) / len(closed) * 100
        avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0
        pf = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        return {"cash": self.cash, "total_value": self.cash, "total_pnl": sum(t["pnl"] for t in closed),
                "win_rate": win_rate, "trades": len(closed), "profit_factor": pf,
                "avg_win": avg_win, "avg_loss": avg_loss, "wins": len(wins), "losses": len(losses),
                "best_trade": max(t["pnl"] for t in closed), "worst_trade": min(t["pnl"] for t in closed)}


def get_ma_crossover_signal(df: pd.DataFrame) -> dict:
    """
    MA Crossover Strategy:
    - Price above 20-day MA (short-term trend up)
    - 20-day MA above 50-day MA (medium-term trend up)
    - Not overbought (RSI < 65)
    - Simple stop below 20-day MA
    """
    
    if df is None or len(df) < 60:
        return {"signal": "hold", "confidence": 0}
    
    close = df["Close"]
    price = close.iloc[-1]
    
    # Moving averages
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    
    current_sma20 = sma20.iloc[-1]
    current_sma50 = sma50.iloc[-1]
    prev_sma20 = sma20.iloc[-2]
    prev_sma50 = sma50.iloc[-2]
    
    # Must have both valid
    if pd.isna(current_sma20) or pd.isna(current_sma50):
        return {"signal": "hold", "confidence": 0}
    
    # Price above 20-day MA
    if price <= current_sma20:
        return {"signal": "hold", "confidence": 0}
    
    # 20-day MA above 50-day MA (trend up)
    if current_sma20 <= current_sma50:
        return {"signal": "hold", "confidence": 0}
    
    # Golden cross happened recently (bonus)
    golden_cross = (prev_sma20 <= prev_sma50) and (current_sma20 > current_sma50)
    
    # RSI not overbought
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_current = rsi.iloc[-1]
    
    if rsi_current > 70:
        return {"signal": "hold", "confidence": 0}
    
    # Momentum
    returns_5d = (price / close.shift(5).iloc[-1] - 1) if not pd.isna(close.shift(5).iloc[-1]) else 0
    
    # Stop: below 20-day MA (5-7%)
    stop_loss = round(current_sma20 * 0.95, 2)
    
    # Target: 8%
    take_profit = round(price * 1.08, 2)
    
    # Calculate confidence
    score = 0
    reasons = []
    
    if price > current_sma20 * 1.03:
        score += 2
        reasons.append("Strong above SMA20")
    
    if golden_cross:
        score += 3
        reasons.append("Golden Cross!")
    
    if rsi_current < 50:
        score += 1
        reasons.append(f"RSI {rsi_current:.0f}")
    
    if returns_5d > 0.03:
        score += 2
        reasons.append(f"Mom {returns_5d*100:.1f}%")
    
    # Check if 200-day MA exists and price above it
    if not pd.isna(sma200.iloc[-1]) and price > sma200.iloc[-1]:
        score += 1
        reasons.append("Above SMA200")
    
    if score >= 4:
        confidence = min(score / 10, 1.0)
        
        return {
            "signal": "buy",
            "confidence": confidence,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "reasons": reasons,
            "price": price,
            "sma20": current_sma20,
            "golden_cross": golden_cross
        }
    
    return {"signal": "hold", "confidence": 0}


def run_backtest_ma():
    """Backtest MA crossover strategy."""
    
    symbols = ["RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "TCS.NS", "SBIN.NS",
               "BAJFINANCE.NS", "TITAN.NS", "SUNPHARMA.NS", "MARUTI.NS", "M&M.NS",
               "KOTAKBANK.NS", "ICICIBANK.NS", "ADANIPORTS.NS", "HINDUNILVR.NS", "NESTLEIND.NS"]
    
    print("=" * 70)
    print("MA CROSSOVER STRATEGY BACKTEST")
    print("=" * 70)
    
    broker = BacktestBroker(5000)
    
    print("\nFetching data...")
    all_data = {}
    
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period="2y", interval="1d")
            if len(df) > 100:
                all_data[sym] = df.dropna()
                print(f"  {sym}: {len(all_data[sym])} days")
        except:
            pass
    
    if not all_data:
        return
    
    # Get unique trading dates
    all_dates = []
    for sym, df in all_data.items():
        all_dates.extend(df.index.tolist())
    unique_dates = sorted(set(all_dates))
    
    print(f"\nSimulating {len(unique_dates)} days...")
    trades = 0
    
    for i, date in enumerate(unique_dates[60:], 1):
        if i % 50 == 0:
            print(f"  Day {i}")
        
        prices = {}
        for sym, df in all_data.items():
            if date in df.index:
                prices[sym] = df.loc[date, "Close"]
        
        if not prices:
            continue
        
        broker.check_exits(prices, str(date.date()))
        
        # Max 2 positions
        if len(broker.positions) < 2:
            for sym, price in prices.items():
                if broker._find_position(sym):
                    continue
                
                if sym not in all_data:
                    continue
                
                df = all_data[sym]
                df_subset = df[df.index <= date]
                
                if len(df_subset) < 60:
                    continue
                
                signal = get_ma_crossover_signal(df_subset)
                
                if signal["signal"] == "buy" and signal["confidence"] >= 0.4:
                    risk = price - signal["stop_loss"]
                    if risk > 0:
                        size = int(400 / risk)
                        
                        if size > 0:
                            result = broker.execute_buy(sym, size, price,
                                                        signal["stop_loss"],
                                                        signal["take_profit"],
                                                        str(date.date()))
                            if result:
                                trades += 1
                                gc = "GC!" if signal.get("golden_cross") else ""
                                print(f"  BUY: {sym} @ Rs{price:.0f} | {gc} Conf:{signal['confidence']:.0%}")
    
    print("\n" + "=" * 70)
    print("MA CROSSOVER STRATEGY RESULTS")
    print("=" * 70)
    
    summary = broker.get_summary()
    
    print(f"""
Strategy: MA Crossover
Entry:
  - Price above 20-day MA
  - 20-day MA above 50-day MA (trend up)
  - RSI < 70
  - Recent momentum
Stop: Below 20-day MA (~5%)
Target: 8%

=== Performance ===
Starting: Rs5,000 | Final: Rs{summary['total_value']:.2f}
P&L: Rs{summary['total_pnl']:.2f} | Return: {(summary['total_value']/5000-1)*100:.1f}%

Trades: {summary['trades']}
Win Rate: {summary['win_rate']:.1f}%
Wins: {summary.get('wins', 0)} | Losses: {summary.get('losses', 0)}
Avg Win: Rs{summary['avg_win']:.2f} | Avg Loss: Rs{summary['avg_loss']:.2f}
Profit Factor: {summary['profit_factor']:.2f}
""")
    
    if broker.closed_trades:
        print("Trades:")
        for t in broker.closed_trades:
            print(f"  {t['symbol']} | {t['entry_date']}->{t['exit_date']} | "
                  f"PnL:Rs{t['pnl']:.0f} ({t['pnl_pct']:+.1f}%) | {t['reason']}")
    
    return summary


if __name__ == "__main__":
    run_backtest_ma()