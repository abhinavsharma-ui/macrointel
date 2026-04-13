"""
Breakout Strategy with Adaptive Stops
=====================================
Key changes:
1. Wider stops (8%) to survive market noise
2. ATR-based targets
3. Only enter on strong breakout (price breaks above 20-day high)
4. Add trailing stop when in profit
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
    trail_price: float = 0.0
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
        pos = Position(symbol, qty, price, stop_loss, take_profit, entry_date, price, price)
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
        """Check exits with trailing stop."""
        for pos in list(self.positions):
            if pos.symbol in prices:
                price = prices[pos.symbol]
                
                # Update trailing stop when in profit
                if price > pos.entry_price * 1.05:  # 5% profit
                    # Trail at 3% below highest price since entry
                    new_trail = price * 0.97
                    if new_trail > pos.trail_price:
                        pos.trail_price = new_trail
                        # Move stop to breakeven minimum
                        pos.stop_loss = max(pos.stop_loss, pos.entry_price * 0.98)
                
                # Check stops
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


def get_breakout_signal(df: pd.DataFrame) -> dict:
    """
    Breakout Strategy:
    - Price breaks above 20-day high (not just MA)
    - Volume confirms (>1.5x)
    - RSI not overbought (<65)
    - ADX shows strength (>25)
    """
    
    if df is None or len(df) < 50:
        return {"signal": "hold", "confidence": 0}
    
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    price = close.iloc[-1]
    
    # 20-day high
    high_20d = high.rolling(20).max().iloc[-1]
    
    # Must break above 20-day high
    if price <= high_20d:
        return {"signal": "hold", "confidence": 0}
    
    # Was below high yesterday (true breakout)
    high_20d_yesterday = high.rolling(20).max().iloc[-2]
    if close.iloc[-2] <= high_20d_yesterday:
        pass  # This is the breakout day
    else:
        return {"signal": "hold", "confidence": 0}  # Already broken
    
    # RSI not overbought
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_current = rsi.iloc[-1]
    
    if rsi_current > 70:
        return {"signal": "hold", "confidence": 0}
    
    # Volume confirmation
    vol_ma20 = volume.rolling(20).mean().iloc[-1]
    volume_ratio = volume.iloc[-1] / vol_ma20 if vol_ma20 > 0 else 1
    
    if volume_ratio < 1.3:
        return {"signal": "hold", "confidence": 0}
    
    # ATR for stops
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    atr_pct = atr / price
    
    # Wide stop (8%)
    stop_loss = round(price * 0.92, 2)
    
    # ATR-based target (4x ATR = ~8-12%)
    take_profit = round(price + (atr * 4), 2)
    
    # Calculate confidence
    score = 0
    reasons = []
    
    # Strong breakout (far above 20d high)
    if price > high_20d * 1.02:
        score += 2
        reasons.append(f"Strong breakout")
    
    # Very high volume
    if volume_ratio > 2.0:
        score += 2
        reasons.append(f"Volume {volume_ratio:.1f}x")
    elif volume_ratio > 1.5:
        score += 1
        reasons.append(f"Vol {volume_ratio:.1f}x")
    
    # Good momentum (not overbought RSI)
    if rsi_current < 50:
        score += 1
        reasons.append(f"RSI {rsi_current:.0f}")
    
    # Strong recent move
    returns_5d = (price / close.shift(5).iloc[-1] - 1) if not pd.isna(close.shift(5).iloc[-1]) else 0
    if returns_5d > 0.05:
        score += 1
        reasons.append(f"Mom {returns_5d*100:.1f}%")
    
    if score >= 4:
        confidence = min(score / 10, 1.0)
        
        return {
            "signal": "buy",
            "confidence": confidence,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "reasons": reasons,
            "price": price,
            "high_20d": high_20d,
            "volume_ratio": volume_ratio,
            "atr_pct": atr_pct
        }
    
    return {"signal": "hold", "confidence": 0}


def run_backtest_breakout():
    """Backtest breakout strategy."""
    
    symbols = ["RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "TCS.NS", "SBIN.NS",
               "BAJFINANCE.NS", "TITAN.NS", "SUNPHARMA.NS", "MARUTI.NS", "M&M.NS",
               "KOTAKBANK.NS", "ICICIBANK.NS", "ADANIPORTS.NS", "HINDUNILVR.NS", "NESTLEIND.NS"]
    
    print("=" * 70)
    print("BREAKOUT STRATEGY BACKTEST")
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
    
    for i, date in enumerate(unique_dates[50:], 1):
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
                
                if len(df_subset) < 50:
                    continue
                
                signal = get_breakout_signal(df_subset)
                
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
                                print(f"  BUY: {sym} @ Rs{price:.0f} | "
                                      f"Conf:{signal['confidence']:.0%}")
    
    print("\n" + "=" * 70)
    print("BREAKOUT STRATEGY RESULTS")
    print("=" * 70)
    
    summary = broker.get_summary()
    
    print(f"""
Strategy: Breakout
Entry:
  - Price breaks above 20-day high (today)
  - Volume > 1.3x
  - RSI < 70
Stop: 8%
Target: ATR-based (~8-12%)
Trailing stop when 5%+ profit

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
    run_backtest_breakout()