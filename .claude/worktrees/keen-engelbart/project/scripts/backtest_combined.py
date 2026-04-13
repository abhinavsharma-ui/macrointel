"""
Combined Strategy - Mean Reversion + Momentum with Better Filters
===================================================================
Key improvements:
1. ADX trend strength filter (must be trending, not ranging)
2. Wider stops (5%) to avoid noise
3. Combined signals - either RSI oversold OR strong momentum breakout
4. Higher confidence threshold (50%)
5. Only trade in market uptrend
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


def calculate_adx(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate ADX (Average Directional Index) for trend strength."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    
    # +DM and -DM
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
    
    # True Range
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smoothed
    atr = tr.rolling(period).mean()
    plus_di = (plus_dm.rolling(period).mean() / atr) * 100
    minus_di = (minus_dm.rolling(period).mean() / atr) * 100
    
    # DX
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    
    # ADX
    adx = dx.rolling(period).mean()
    
    return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 25


def get_combined_signal(df: pd.DataFrame, market_bullish: bool = True) -> dict:
    """
    Combined Mean Reversion + Momentum Strategy:
    - Option 1: RSI oversold (<35) + recovering
    - Option 2: Strong momentum breakout (price > SMA20 + ADX > 25 + volume)
    - Must have market in uptrend
    - Wider stops (5%), target 15% (3:1 RR)
    """
    
    if df is None or len(df) < 50:
        return {"signal": "hold", "confidence": 0}
    
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    price = close.iloc[-1]
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_current = rsi.iloc[-1]
    rsi_3d_ago = rsi.iloc[-4] if len(rsi) >= 4 else rsi_current
    
    # Moving averages
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    
    # Trend strength
    adx = calculate_adx(df)
    
    # Volume
    vol_ma20 = volume.rolling(20).mean().iloc[-1]
    volume_ratio = volume.iloc[-1] / vol_ma20 if vol_ma20 > 0 else 1
    
    # Momentum
    returns_5d = (price / close.shift(5).iloc[-1] - 1) if not pd.isna(close.shift(5).iloc[-1]) else 0
    
    # === CHECK MARKET FILTER ===
    if not market_bullish:
        return {"signal": "hold", "confidence": 0}
    
    score = 0
    reasons = []
    signal_type = ""
    
    # === OPTION 1: MEAN REVERSION (RSI oversold recovering) ===
    rsi_buy = False
    if rsi_current < 35 and rsi_current > rsi_3d_ago:  # RSI recovering
        score += 3
        reasons.append(f"RSI recovering ({rsi_current:.0f})")
        rsi_buy = True
        signal_type = "mean_reversion"
    
    # === OPTION 2: MOMENTUM (trend breakout) ===
    momentum_buy = False
    if price > sma20 and adx > 20:  # Trending + above MA
        score += 2
        reasons.append(f"Trending (ADX:{adx:.0f})")
        
        if returns_5d > 0.03:
            score += 2
            reasons.append(f"Momentum {returns_5d*100:.1f}%")
            momentum_buy = True
        
        if volume_ratio > 1.3:
            score += 1
            reasons.append(f"Vol {volume_ratio:.1f}x")
        
        if price > sma20 * 1.03:  # Strong breakout
            score += 1
        
        if signal_type == "":
            signal_type = "momentum"
    
    # Apply if either condition met
    if not (rsi_buy or momentum_buy):
        return {"signal": "hold", "confidence": 0}
    
    # === STOP LOSS (5% below entry) ===
    stop_loss = round(price * 0.95, 2)
    
    # === TARGET (15% = 3:1 RR) ===
    take_profit = round(price * 1.15, 2)
    
    if score >= 5:  # High confidence
        confidence = min(score / 10, 1.0)
        
        return {
            "signal": "buy",
            "confidence": confidence,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "reasons": reasons,
            "price": price,
            "rsi": rsi_current,
            "adx": adx,
            "signal_type": signal_type
        }
    
    return {"signal": "hold", "confidence": 0}


def run_backtest_combined():
    """Backtest combined strategy."""
    
    symbols = ["RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "TCS.NS", "SBIN.NS",
               "BAJFINANCE.NS", "TITAN.NS", "SUNPHARMA.NS", "MARUTI.NS", "M&M.NS",
               "KOTAKBANK.NS", "ICICIBANK.NS", "ADANIPORTS.NS", "HINDUNILVR.NS", "NESTLEIND.NS"]
    
    print("=" * 70)
    print("COMBINED STRATEGY (Mean Reversion + Momentum)")
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
    
    # Calculate market trend
    print("\nCalculating market trends...")
    market_trends = {}
    try:
        nifty = yf.Ticker("^NSEI").history(period="2y", interval="1d")
        if len(nifty) > 50:
            nifty_sma20 = nifty["Close"].rolling(20).mean()
            for date in unique_dates:
                if date in nifty.index:
                    market_trends[date] = nifty["Close"].loc[date] > nifty_sma20.loc[date]
    except:
        pass
    
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
        
        # Max 2 positions with higher risk
        if len(broker.positions) < 2:
            market_bullish = market_trends.get(date, True)
            
            for sym, price in prices.items():
                if broker._find_position(sym):
                    continue
                
                if sym not in all_data:
                    continue
                
                df = all_data[sym]
                df_subset = df[df.index <= date]
                
                if len(df_subset) < 50:
                    continue
                
                signal = get_combined_signal(df_subset, market_bullish)
                
                if signal["signal"] == "buy" and signal["confidence"] >= 0.5:
                    risk = price - signal["stop_loss"]
                    if risk > 0:
                        size = int(400 / risk)  # Higher risk
                        
                        if size > 0:
                            result = broker.execute_buy(sym, size, price,
                                                        signal["stop_loss"],
                                                        signal["take_profit"],
                                                        str(date.date()))
                            if result:
                                trades += 1
                                stype = signal.get("signal_type", "unknown")
                                print(f"  BUY: {sym} @ Rs{price:.0f} | {stype} | "
                                      f"Conf:{signal['confidence']:.0%}")
    
    print("\n" + "=" * 70)
    print("COMBINED STRATEGY RESULTS")
    print("=" * 70)
    
    summary = broker.get_summary()
    
    print(f"""
Strategy: Combined (Mean Reversion + Momentum)
Entry Options:
  1. RSI < 35 + recovering (mean reversion)
  2. Price > SMA20 + ADX > 20 + momentum > 3% (momentum)
Market Filter: Must be in uptrend
Stop: 5%
Target: 15% (3:1 RR)
Confidence: >= 50%

=== Performance ===
Starting: Rs5,000 | Final: Rs{summary['total_value']:.2f}
P&L: Rs{summary['total_pnl']:.2f} | Return: {(summary['total_value']/5000-1)*100:.1f}%

Trades: {summary['trades']}
Win Rate: {summary['win_rate']:.1f}%
Wins: {summary.get('wins', 0)} | Losses: {summary.get('losses', 0)}
Avg Win: Rs{summary['avg_win']:.2f} | Avg Loss: Rs{summary['avg_loss']:.2f}
Profit Factor: {summary['profit_factor']:.2f}
Best: Rs{summary.get('best_trade', 0):.0f} | Worst: Rs{summary.get('worst_trade', 0):.0f}
""")
    
    if broker.closed_trades:
        print("Trades:")
        for t in broker.closed_trades:
            print(f"  {t['symbol']} | {t['entry_date']}->{t['exit_date']} | "
                  f"PnL:Rs{t['pnl']:.0f} ({t['pnl_pct']:+.1f}%) | {t['reason']}")
    
    return summary


if __name__ == "__main__":
    run_backtest_combined()