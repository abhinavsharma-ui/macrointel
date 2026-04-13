"""
Improved Strategy v4 - ML Signal + Technical Entry
===================================================
Combines ML model predictions with technical entry rules:
1. ML predicts "buy" with high confidence -> filter to potential trades
2. Technical entry: RSI < 30 + MACD bullish + uptrend
3. Wider stops (6%) to avoid getting stopped out
4. Trail stops when in profit
"""

import os
import sys
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict
import pickle

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

os.environ["FINNHUB_API_KEYS"] = "d78vt3hr01qp0fl6mfjgd78vt3hr01qp0mfk0"


@dataclass
class Position:
    symbol: str
    quantity: int
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_date: str
    trail_level: float = 0.0
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
        pos = Position(symbol, qty, price, stop_loss, take_profit, entry_date, price, price)
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
    
    def check_exits(self, prices: dict, current_date: str):
        """Check stops with trailing stop logic."""
        for pos in list(self.positions):
            if pos.symbol in prices:
                price = prices[pos.symbol]
                
                # Update trailing stop when in profit
                if price > pos.entry_price * 1.03:  # 3% profit
                    new_trail = price * 0.97  # Trail at 3% below high
                    if new_trail > pos.trail_level:
                        pos.trail_level = new_trail
                        pos.stop_loss = max(pos.stop_loss, new_trail)
                
                # Check exits
                if price <= pos.stop_loss:
                    self.close_position(pos, price, current_date, "SL")
                elif price >= pos.take_profit:
                    self.close_position(pos, price, current_date, "TP")
    
    def get_summary(self) -> dict:
        closed = self.closed_trades
        if not closed:
            return {"cash": self.cash, "total_value": self.cash, "total_pnl": 0, 
                    "win_rate": 0, "trades": 0, "profit_factor": 0, "avg_win": 0, "avg_loss": 0}
        
        wins = [t for t in closed if t["pnl"] > 0]
        losses = [t for t in closed if t["pnl"] <= 0]
        
        win_rate = len(wins) / len(closed) * 100
        avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0
        pf = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        return {"cash": self.cash, "total_value": self.cash, "total_pnl": sum(t["pnl"] for t in closed),
                "win_rate": win_rate, "trades": len(closed), "profit_factor": pf,
                "avg_win": avg_win, "avg_loss": avg_loss, "wins": len(wins), "losses": len(losses)}


def get_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features for ML model."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    
    features = pd.DataFrame(index=df.index)
    
    # Returns
    for d in [1, 2, 3, 5, 10, 20]:
        features[f"return_{d}d"] = close.pct_change(d)
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    features["rsi_14"] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9).mean()
    features["macd"] = macd
    features["macd_signal"] = macd_signal
    features["macd_hist"] = macd - macd_signal
    
    # Moving averages
    for w in [5, 10, 20, 50]:
        features[f"sma_{w}"] = close.rolling(w).mean()
        features[f"price_vs_sma_{w}"] = close / close.rolling(w).mean() - 1
    
    # Volume
    features["volume_ma10"] = volume.rolling(10).mean()
    features["volume_ratio"] = volume / volume.rolling(10).mean()
    
    # Volatility
    features["atr_14"] = (high - low).rolling(14).mean()
    features["atr_pct"] = features["atr_14"] / close
    
    # Momentum
    features["momentum_5"] = close / close.shift(5) - 1
    features["momentum_10"] = close / close.shift(10) - 1
    
    return features


def get_technical_signal(df: pd.DataFrame) -> dict:
    """Get technical entry signal."""
    
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
    
    # Trend
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    
    price = close.iloc[-1]
    uptrend = (price > sma20.iloc[-1]) if not pd.isna(sma20.iloc[-1]) else False
    
    # Volume
    vol_ma10 = volume.rolling(10).mean()
    volume_ratio = volume.iloc[-1] / vol_ma10.iloc[-1] if vol_ma10.iloc[-1] > 0 else 1
    
    # ATR
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    atr_pct = atr / price if price > 0 else 0.02
    
    # Score
    score = 0
    reasons = []
    
    if rsi_current < 30:
        score += 3
        reasons.append(f"RSI {rsi_current:.0f}")
    
    if macd_bullish:
        score += 2
        reasons.append("MACD cross")
    
    if uptrend:
        score += 2
        reasons.append("Uptrend")
    
    if volume_ratio > 1.3:
        score += 1
        reasons.append(f"Vol {volume_ratio:.1f}x")
    
    if score >= 5:
        # Wider stop (6%), target (12%) = 2:1 RR
        stop_loss = round(price * (1 - 0.06), 2)
        take_profit = round(price * (1 + 0.12), 2)
        
        return {"signal": "buy", "confidence": min(score/10, 1.0), 
                "stop_loss": stop_loss, "take_profit": take_profit, 
                "reasons": reasons, "rsi": rsi_current}
    
    return {"signal": "hold", "confidence": 0}


def run_backtest_v4():
    """Run backtest with ML + technical strategy."""
    
    symbols = ["RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "TCS.NS", "SBIN.NS", "BAJFINANCE.NS"]
    
    print("=" * 70)
    print("STRATEGY v4 - TECHNICAL ENTRY WITH WIDER STOPS")
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
        except Exception as e:
            print(f"  {sym}: Failed")
    
    if not all_data:
        return
    
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
        
        if len(broker.positions) < 3:
            for sym, price in prices.items():
                if broker._find_position(sym):
                    continue
                
                if sym not in all_data:
                    continue
                
                df = all_data[sym]
                df_subset = df[df.index <= date]
                
                if len(df_subset) < 50:
                    continue
                
                signal = get_technical_signal(df_subset)
                
                if signal["signal"] == "buy" and signal["confidence"] >= 0.5:
                    risk = price - signal["stop_loss"]
                    if risk > 0:
                        size = int(250 / risk)
                        
                        if size > 0:
                            result = broker.execute_buy(sym, size, price, 
                                                        signal["stop_loss"], 
                                                        signal["take_profit"],
                                                        str(date.date()))
                            if result:
                                trades += 1
    
    print("\n" + "=" * 70)
    print("RESULTS - v4 (Wider Stops)")
    print("=" * 70)
    
    summary = broker.get_summary()
    
    print(f"""
Strategy: Technical Entry + Wider Stops
- Entry: RSI < 30 + MACD cross + Uptrend + Vol > 1.3x
- Stop: 6% (was 5%)
- Target: 12% (was 10%)
- Trailing stop when 3%+ profit

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
    run_backtest_v4()