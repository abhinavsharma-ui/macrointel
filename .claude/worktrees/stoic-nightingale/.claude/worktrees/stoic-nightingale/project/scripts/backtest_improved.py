"""
Improved Strategy Backtest with Multi-Factor Confirmation
===========================================================
Tests the actual improved strategy against historical data.
"""

import os
import sys
import logging
import random
from pathlib import Path
from datetime import datetime, timedelta
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
            "symbol": pos.symbol,
            "entry_date": pos.entry_date,
            "exit_date": exit_date,
            "entry_price": pos.entry_price,
            "exit_price": price,
            "quantity": pos.quantity,
            "pnl": pnl,
            "pnl_pct": pnl / (pos.entry_price * pos.quantity) * 100,
            "reason": reason
        })
        self.positions.remove(pos)
    
    def check_stops(self, prices: dict, current_date: str):
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
            "losses": len(losses)
        }


def get_improved_signal(df: pd.DataFrame) -> dict:
    """Multi-factor signal generation."""
    
    if len(df) < 50:
        return {"signal": "hold", "confidence": 0}
    
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_current = rsi.iloc[-1]
    
    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9).mean()
    macd_hist = macd - macd_signal
    
    macd_bullish = (macd.iloc[-1] > macd_signal.iloc[-1]) and (macd.iloc[-2] <= macd_signal.iloc[-2])
    macd_hist_positive = macd_hist.iloc[-1] > 0
    
    # Moving Averages
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    
    price_above_ma20 = close.iloc[-1] > sma20.iloc[-1] if not pd.isna(sma20.iloc[-1]) else False
    price_above_ma50 = close.iloc[-1] > sma50.iloc[-1] if not pd.isna(sma50.iloc[-1]) else False
    
    # Volume
    vol_ma20 = volume.rolling(20).mean()
    volume_ratio = volume.iloc[-1] / vol_ma20.iloc[-1] if vol_ma20.iloc[-1] > 0 else 1
    
    # Momentum
    returns_5d = close.pct_change(5).iloc[-1]
    
    # ATR
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    atr_pct = atr / close.iloc[-1] if close.iloc[-1] > 0 else 0.02
    
    # Scoring
    buy_score = 0
    sell_score = 0
    reasons = []
    
    # RSI (most important)
    if rsi_current < 25:
        buy_score += 4
        reasons.append(f"RSI oversold ({rsi_current:.0f})")
    elif rsi_current < 32:
        buy_score += 2
        reasons.append(f"RSI near oversold ({rsi_current:.0f})")
    
    if rsi_current > 75:
        sell_score += 4
    
    # MACD
    if macd_bullish:
        buy_score += 2.5
        reasons.append("MACD bullish cross")
    elif macd_hist_positive:
        buy_score += 1
    
    # Trend
    if price_above_ma20 and price_above_ma50:
        buy_score += 1.5
        reasons.append("Uptrend")
    
    # Volume
    if volume_ratio > 1.3:
        buy_score += 1
        reasons.append(f"High vol ({volume_ratio:.1f}x)")
    
    # Momentum
    if returns_5d > 0.02:
        buy_score += 1
    elif returns_5d < -0.02:
        sell_score += 1
    
    # High volatility penalty
    if atr_pct > 0.04:
        buy_score *= 0.7
    
    # Decision
    entry_price = close.iloc[-1]
    
    if buy_score >= 5 and buy_score > sell_score:
        return {
            "signal": "buy",
            "confidence": min(buy_score / 10, 1.0),
            "buy_score": buy_score,
            "stop_loss": round(entry_price * (1 - 2 * atr_pct), 2),
            "take_profit": round(entry_price * (1 + 3 * atr_pct), 2),
            "reasons": reasons[:3],
            "rsi": rsi_current,
            "atr_pct": atr_pct
        }
    
    return {"signal": "hold", "confidence": 0}


def run_backtest():
    """Run backtest with improved multi-factor strategy."""
    
    symbols = ["RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "TCS.NS", "SBIN.NS", "BAJFINANCE.NS"]
    
    print("=" * 70)
    print("IMPROVED STRATEGY BACKTEST - MULTI-FACTOR CONFIRMATION")
    print("=" * 70)
    
    broker = BacktestBroker(5000)
    
    # Fetch data
    print("\nFetching historical data...")
    all_data = {}
    
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period="2y", interval="1d")
            if len(df) > 100:
                df = df.dropna()
                all_data[sym] = df
                print(f"  {sym}: {len(df)} days")
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
    
    signals_generated = 0
    trades_executed = 0
    
    # Run through each day
    for i, date in enumerate(unique_dates[60:], 1):
        if i % 50 == 0:
            print(f"  Day {i}/{len(unique_dates)-60}")
        
        # Get prices
        prices = {}
        for sym, df in all_data.items():
            if date in df.index:
                prices[sym] = df.loc[date, "Close"]
        
        if not prices:
            continue
        
        # Check stops
        broker.check_stops(prices, str(date.date()))
        
        # Limit open positions
        if len(broker.positions) < 4:
            # Scan for signals
            for sym, price in prices.items():
                if broker._find_position(sym):
                    continue
                
                if sym not in all_data:
                    continue
                
                df = all_data[sym]
                df_subset = df[df.index <= date]
                
                if len(df_subset) < 50:
                    continue
                
                # Get signal
                signal = get_improved_signal(df_subset)
                
                if signal["signal"] == "buy" and signal["confidence"] >= 0.5:
                    signals_generated += 1
                    
                    # Check risk/reward
                    rr = (signal["take_profit"] - price) / (price - signal["stop_loss"])
                    
                    if rr >= 1.5:
                        # Position size (risk 5% = Rs250)
                        risk = price - signal["stop_loss"]
                        if risk > 0:
                            size = int(250 / risk)
                            
                            if size > 0:
                                result = broker.execute_buy(
                                    sym, size, price,
                                    signal["stop_loss"],
                                    signal["take_profit"],
                                    str(date.date())
                                )
                                if result:
                                    trades_executed += 1
    
    # Results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    
    summary = broker.get_summary()
    
    print(f"""
Strategy: Multi-Factor Confirmation
- Entry: RSI < 32 + MACD bullish + Uptrend + Volume > 1.3x
- Confidence: >= 50%
- Risk/Reward: >= 1.5 (2x ATR stop, 3x ATR target)
- Max positions: 4
- Risk per trade: 5%

=== Performance ===
Starting Capital: Rs5,000
Final Value: Rs{summary['total_value']:.2f}
Total PnL: Rs{summary['total_pnl']:.2f}
Return: {(summary['total_value']/5000-1)*100:.1f}%

=== Trade Statistics ===
Signals Generated: {signals_generated}
Trades Executed: {trades_executed}
Closed Trades: {summary['trades']}
Win Rate: {summary['win_rate']:.1f}%

Wins: {summary.get('wins', 0)}
Losses: {summary.get('losses', 0)}

Avg Win: Rs{summary['avg_win']:.2f}
Avg Loss: Rs{summary['avg_loss']:.2f}
Profit Factor: {summary['profit_factor']:.2f}

=== Precision Analysis ===
Total Signals: {signals_generated}
Trades Taken: {trades_executed}
Signal-to-Trade Rate: {trades_executed/signals_generated*100:.1f}%
Win Rate (Actual): {summary['win_rate']:.1f}%
""")
    
    # Show trade details
    if broker.closed_trades:
        print("Sample Trades:")
        for t in broker.closed_trades[:10]:
            print(f"  {t['symbol']} | {t['entry_date']} -> {t['exit_date']} | "
                  f"PnL: Rs{t['pnl']:.0f} ({t['pnl_pct']:+.1f}%) | {t['reason']}")
    
    return summary


if __name__ == "__main__":
    run_backtest()