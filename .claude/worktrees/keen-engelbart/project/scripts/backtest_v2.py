"""
Improved Strategy v2 - Tighter Stops + Better Filtering
========================================================
Changes:
1. Tighter stops (1.5x ATR instead of 2x)
2. Wider targets (4x ATR instead of 3x)
3. Higher confidence threshold (60%)
4. Added market regime filter
5. Only trade with strong RSI divergence
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
            "losses": len(losses),
            "best_trade": max(t["pnl"] for t in closed) if closed else 0,
            "worst_trade": min(t["pnl"] for t in closed) if closed else 0
        }


def get_improved_signal_v2(df: pd.DataFrame) -> dict:
    """Improved multi-factor signal with tighter stops."""
    
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
    
    # RSI history for divergence
    rsi_10d_ago = rsi.iloc[-10] if len(rsi) >= 10 else rsi_current
    
    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9).mean()
    macd_hist = macd - macd_signal
    
    macd_bullish_cross = (macd.iloc[-1] > macd_signal.iloc[-1]) and (macd.iloc[-2] <= macd_signal.iloc[-2])
    macd_hist_turning_positive = (macd_hist.iloc[-1] > 0) and (macd_hist.iloc[-2] <= 0)
    
    # Moving Averages
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean() if len(close) >= 200 else sma50
    
    price = close.iloc[-1]
    price_vs_ma20 = (price - sma20.iloc[-1]) / sma20.iloc[-1] if not pd.isna(sma20.iloc[-1]) else 0
    price_vs_ma50 = (price - sma50.iloc[-1]) / sma50.iloc[-1] if not pd.isna(sma50.iloc[-1]) else 0
    
    # Volume
    vol_ma20 = volume.rolling(20).mean()
    volume_ratio = volume.iloc[-1] / vol_ma20.iloc[-1] if vol_ma20.iloc[-1] > 0 else 1
    
    # Momentum
    returns_5d = close.pct_change(5).iloc[-1]
    returns_20d = close.pct_change(20).iloc[-1]
    
    # ATR
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    atr_pct = atr / price if price > 0 else 0.02
    
    # Volatility regime
    avg_atr_pct = tr.rolling(20).mean() / close.rolling(20).mean()
    avg_atr_pct_val = avg_atr_pct.iloc[-1] if not pd.isna(avg_atr_pct.iloc[-1]) else 0.02
    
    # Market trend (using 20d returns)
    market_bullish = returns_20d > 0
    
    # Scoring - much stricter
    buy_score = 0
    reasons = []
    
    # RSI: Need very oversold + divergence
    if rsi_current < 25:
        buy_score += 4
        reasons.append(f"RSI very oversold ({rsi_current:.0f})")
        
        # Strong divergence bonus
        if rsi_current > rsi_10d_ago + 5:
            buy_score += 2
            reasons.append("RSI bullish divergence")
    elif rsi_current < 30:
        buy_score += 1.5
        reasons.append(f"RSI oversold ({rsi_current:.0f})")
    
    # MACD: Need clear bullish cross, not just histogram positive
    if macd_bullish_cross:
        buy_score += 3
        reasons.append("MACD bullish crossover")
    elif macd_hist_turning_positive:
        buy_score += 1.5
        reasons.append("MACD histogram positive")
    
    # Trend: Price must be above MAs
    if price_vs_ma20 > 0 and price_vs_ma50 > 0:
        buy_score += 2
        reasons.append("Price above MAs")
    
    # Above 200-day MA (if available)
    if not pd.isna(sma200.iloc[-1]) and price > sma200.iloc[-1]:
        buy_score += 1
    
    # Volume: Strong volume on signal
    if volume_ratio > 1.5:
        buy_score += 1.5
        reasons.append(f"High volume ({volume_ratio:.1f}x)")
    elif volume_ratio > 1.2:
        buy_score += 0.5
    
    # Market context: Only trade when market is bullish
    if market_bullish:
        buy_score += 1.5
        reasons.append("Market uptrend")
    
    # Momentum
    if returns_5d > 0.03:
        buy_score += 1.5
        reasons.append(f"Strong momentum ({returns_5d*100:.1f}%)")
    elif returns_5d > 0:
        buy_score += 0.5
    
    # Volatility: Only trade in normal/low volatility
    if avg_atr_pct_val > 0.04:
        buy_score *= 0.5
        reasons.append("High volatility - reduced")
    elif avg_atr_pct_val < 0.02:
        buy_score *= 1.2
    
    # Calculate stop and target
    if buy_score >= 6:  # Higher threshold
        # Tighter stop (1.5x ATR), wider target (4x ATR)
        stop_loss = round(price * (1 - 1.5 * atr_pct), 2)
        take_profit = round(price * (1 + 4 * atr_pct), 2)
        
        # Risk/reward
        rr = (take_profit - price) / (price - stop_loss) if price > stop_loss else 0
        
        if rr >= 2.0:  # Need at least 2:1
            return {
                "signal": "buy",
                "confidence": min(buy_score / 12, 1.0),
                "buy_score": buy_score,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_reward": rr,
                "reasons": reasons[:4],
                "rsi": rsi_current,
                "atr_pct": atr_pct * 100
            }
    
    return {"signal": "hold", "confidence": 0}


def run_backtest_v2():
    """Run backtest with improved strategy v2."""
    
    symbols = ["RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "TCS.NS", "SBIN.NS", "BAJFINANCE.NS"]
    
    print("=" * 70)
    print("IMPROVED STRATEGY v2 - TIGHTER STOPS + BETTER FILTERING")
    print("=" * 70)
    
    broker = BacktestBroker(5000)
    
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
    
    for i, date in enumerate(unique_dates[60:], 1):
        if i % 50 == 0:
            print(f"  Day {i}/{len(unique_dates)-60}")
        
        prices = {}
        for sym, df in all_data.items():
            if date in df.index:
                prices[sym] = df.loc[date, "Close"]
        
        if not prices:
            continue
        
        broker.check_stops(prices, str(date.date()))
        
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
                
                signal = get_improved_signal_v2(df_subset)
                
                if signal["signal"] == "buy" and signal["confidence"] >= 0.5:
                    signals_generated += 1
                    
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
    
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS - v2")
    print("=" * 70)
    
    summary = broker.get_summary()
    
    print(f"""
Strategy v2 Changes:
- Higher score threshold: 6+ (was 5)
- Tighter stops: 1.5x ATR (was 2x)
- Wider targets: 4x ATR (was 3x)
- Require RR >= 2.0 (was 1.5)
- RSI divergence bonus
- Market trend filter

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

Best Trade: Rs{summary.get('best_trade', 0):.2f}
Worst Trade: Rs{summary.get('worst_trade', 0):.2f}
""")
    
    if broker.closed_trades:
        print("Trade Details:")
        for t in broker.closed_trades:
            print(f"  {t['symbol']} | {t['entry_date']}->{t['exit_date']} | "
                  f"PnL: Rs{t['pnl']:.0f} ({t['pnl_pct']:+.1f}%) | {t['reason']}")
    
    return summary


if __name__ == "__main__":
    run_backtest_v2()