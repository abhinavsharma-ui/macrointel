"""
Improved Strategy v3 - Mean Reversion Focus
============================================
Key changes:
1. Focus on mean reversion - buy when RSI very oversold
2. Fixed 5% stop loss (more realistic)
3. 10% profit target (2:1 RR)
4. Only enter when RSI < 25 and has been rising for 2+ days
5. Exit on RSI reaching 50 (mean) instead of price target
"""

import os
import sys
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

os.environ["FINNHUB_API_KEYS"] = "d78vt3hr01qp0fl6mfjgd78vt3hr01qp0fl6mfk0"


@dataclass
class Position:
    symbol: str
    quantity: int
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_date: str
    rsi_entry: float = 0.0
    current_price: float = 0.0


class BacktestBroker:
    def __init__(self, capital: float = 5000):
        self.cash = capital
        self.starting_cash = capital
        self.positions: List[Position] = []
        self.closed_trades: List[Dict] = []
        self.brokerage = 15
    
    def _find_position(self, symbol: str) -> Optional[Position]:
        for p in self.positions:
            if p.symbol == symbol:
                return p
        return None
    
    def execute_buy(self, symbol: str, qty: int, price: float, 
                    stop_loss: float, take_profit: float, entry_date: str, rsi_entry: float) -> bool:
        cost = price * qty + self.brokerage * 1.18
        if self.cash < cost:
            return False
        
        self.cash -= cost
        pos = Position(symbol, qty, price, stop_loss, take_profit, entry_date, rsi_entry, price)
        self.positions.append(pos)
        return True
    
    def close_position(self, pos: Position, price: float, exit_date: str, reason: str):
        proceeds = price * pos.quantity - self.brokerage * 1.18
        pnl = proceeds - (pos.entry_price * pos.quantity)
        
        self.cash += proceeds
        self.closed_trades.append({
            "symbol": pos.symbol,
            "entry_date": pos.entry_date,
            "exit_date": exit_date,
            "entry_price": pos.entry_price,
            "exit_price": price,
            "quantity": pos.quantity,
            "pnl": pnl,
            "pnl_pct": pnl / (pos.entry_price * pos.quantity) * 100,
            "reason": reason,
            "rsi_entry": pos.rsi_entry
        })
        self.positions.remove(pos)
    
    def check_exits(self, prices: dict, current_date: str, rsi_data: dict):
        """Check both stops AND RSI exits (mean reversion exit at RSI 50)."""
        for pos in list(self.positions):
            if pos.symbol in prices:
                price = prices[pos.symbol]
                rsi = rsi_data.get(pos.symbol, 50)
                
                # Stop loss
                if price <= pos.stop_loss:
                    self.close_position(pos, price, current_date, "SL")
                    continue
                
                # Take profit
                if price >= pos.take_profit:
                    self.close_position(pos, price, current_date, "TP")
                    continue
                
                # Mean reversion exit: if RSI crosses 50, exit
                if rsi >= 50 and pos.rsi_entry < 35:
                    self.close_position(pos, price, current_date, "RSI50_EXIT")
                    continue
    
    def get_summary(self) -> dict:
        closed = self.closed_trades
        if not closed:
            return {
                "cash": self.cash,
                "total_value": self.cash,
                "total_pnl": 0,
                "win_rate": 0,
                "trades": 0,
                "profit_factor": 0,
                "avg_win": 0,
                "avg_loss": 0
            }
        
        wins = [t for t in closed if t["pnl"] > 0]
        losses = [t for t in closed if t["pnl"] <= 0]
        
        win_rate = len(wins) / len(closed) * 100
        avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0
        pf = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        return {
            "cash": self.cash,
            "total_value": self.cash,
            "total_pnl": sum(t["pnl"] for t in closed),
            "win_rate": win_rate,
            "trades": len(closed),
            "profit_factor": pf,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "wins": len(wins),
            "losses": len(losses),
            "best_trade": max(t["pnl"] for t in closed) if closed else 0,
            "worst_trade": min(t["pnl"] for t in closed) if closed else 0
        }


def get_mean_reversion_signal(df: pd.DataFrame) -> dict:
    """Mean reversion strategy - buy very oversold, exit at mean."""
    
    if len(df) < 30:
        return {"signal": "hold", "confidence": 0}
    
    close = df["Close"]
    volume = df["Volume"]
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_current = rsi.iloc[-1]
    
    # RSI trend - has it been rising?
    rsi_1d_ago = rsi.iloc[-2] if len(rsi) >= 2 else rsi_current
    rsi_2d_ago = rsi.iloc[-3] if len(rsi) >= 3 else rsi_1d_ago
    
    rsi_rising = rsi_current > rsi_1d_ago > rsi_2d_ago
    
    # RSI must be < 25 AND rising
    if rsi_current >= 25:
        return {"signal": "hold", "confidence": 0}
    
    if not rsi_rising:
        return {"signal": "hold", "confidence": 0}
    
    # Volume check
    vol_ma20 = volume.rolling(20).mean()
    volume_ratio = volume.iloc[-1] / vol_ma20.iloc[-1] if vol_ma20.iloc[-1] > 0 else 1
    
    price = close.iloc[-1]
    
    # Calculate stop and target
    # Fixed 5% stop
    stop_loss = round(price * 0.95, 2)
    # 10% target (2:1 RR)
    take_profit = round(price * 1.10, 2)
    
    # Confidence based on how oversold
    confidence = (25 - rsi_current) / 25  # 0 to 1
    
    return {
        "signal": "buy",
        "confidence": confidence,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "reasons": [f"RSI oversold ({rsi_current:.0f})", "RSI rising"],
        "rsi": rsi_current
    }


def run_backtest_v3():
    """Run backtest with mean reversion strategy."""
    
    symbols = ["RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "TCS.NS", "SBIN.NS", "BAJFINANCE.NS"]
    
    print("=" * 70)
    print("STRATEGY v3 - MEAN REVERSION")
    print("=" * 70)
    
    broker = BacktestBroker(5000)
    
    print("\nFetching historical data...")
    all_data = {}
    
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period="2y", interval="1d")
            if len(df) > 100:
                # Calculate RSI for each row
                close = df["Close"]
                delta = close.diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                df["rsi"] = 100 - (100 / (1 + rs))
                
                all_data[sym] = df.dropna()
                print(f"  {sym}: {len(all_data[sym])} days")
        except Exception as e:
            print(f"  {sym}: Failed - {e}")
    
    if not all_data:
        print("No data loaded!")
        return
    
    # Get unique trading dates
    all_dates = []
    for sym, df in all_data.items():
        all_dates.extend(df.index.tolist())
    unique_dates = sorted(set(all_dates))
    
    print(f"\nSimulating {len(unique_dates)} trading days...")
    
    trades_executed = 0
    
    for i, date in enumerate(unique_dates[30:], 1):
        if i % 50 == 0:
            print(f"  Day {i}/{len(unique_dates)-30}")
        
        # Get prices and RSI
        prices = {}
        rsi_data = {}
        for sym, df in all_data.items():
            if date in df.index:
                prices[sym] = df.loc[date, "Close"]
                rsi_data[sym] = df.loc[date, "rsi"]
        
        if not prices:
            continue
        
        # Check exits
        broker.check_exits(prices, str(date.date()), rsi_data)
        
        # Generate signals
        if len(broker.positions) < 3:
            for sym, price in prices.items():
                if broker._find_position(sym):
                    continue
                
                if sym not in all_data:
                    continue
                
                df = all_data[sym]
                df_subset = df[df.index <= date]
                
                if len(df_subset) < 30:
                    continue
                
                signal = get_mean_reversion_signal(df_subset)
                
                if signal["signal"] == "buy":
                    # Position size (risk 5% = Rs250)
                    risk = price - signal["stop_loss"]
                    if risk > 0:
                        size = int(250 / risk)
                        
                        if size > 0:
                            result = broker.execute_buy(
                                sym, size, price,
                                signal["stop_loss"],
                                signal["take_profit"],
                                str(date.date()),
                                signal["rsi"]
                            )
                            if result:
                                trades_executed += 1
    
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS - v3 (Mean Reversion)")
    print("=" * 70)
    
    summary = broker.get_summary()
    
    print(f"""
Strategy: Mean Reversion
- Entry: RSI < 25 AND rising for 2+ days
- Exit 1: Stop loss at -5%
- Exit 2: Take profit at +10%
- Exit 3: RSI crosses 50 (mean reversion complete)

=== Performance ===
Starting Capital: Rs5,000
Final Value: Rs{summary['total_value']:.2f}
Total PnL: Rs{summary['total_pnl']:.2f}
Return: {(summary['total_value']/5000-1)*100:.1f}%

=== Trade Statistics ===
Trades Executed: {trades_executed}
Closed Trades: {summary['trades']}
Win Rate: {summary['win_rate']:.1f}%

Wins: {summary.get('wins', 0)}
Losses: {summary.get('losses', 0)}

Avg Win: Rs{summary['avg_win']:.2f}
Avg Loss: Rs{summary['avg_loss']:.2f}
Profit Factor: {summary['profit_factor']:.2f}
""")
    
    if broker.closed_trades:
        print("Trade Details:")
        for t in broker.closed_trades:
            print(f"  {t['symbol']} | Entry:{t['entry_date']} Exit:{t['exit_date']} | "
                  f"PnL:Rs{t['pnl']:.0f} ({t['pnl_pct']:+.1f}%) | RSI:{t['rsi_entry']:.0f}->50 | {t['reason']}")
    
    return summary


if __name__ == "__main__":
    run_backtest_v3()