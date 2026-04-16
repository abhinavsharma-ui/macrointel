"""
Momentum/Trend-Following Strategy
================================
Key changes from RSI mean reversion:
1. Entry: Price breaking out above 20-day MA + MACD bullish + strong volume
2. Stop: Below recent support (lowest low of last 10 days)
3. Target: 3:1 RR (12% target, 4% stop)
4. Market filter: Only trade when market is bullish (Nifty > 20-day MA)
5. Multiple confirmations for higher precision
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


def get_momentum_signal(df: pd.DataFrame, market_bullish: bool = True) -> dict:
    """
    Momentum/Trend-Following Entry:
    - Price breaking above 20-day MA
    - MACD bullish crossover
    - Strong volume (>1.3x)
    - Price momentum positive (>2% in 5 days)
    - Market in uptrend (filter)
    """
    
    if df is None or len(df) < 50:
        return {"signal": "hold", "confidence": 0}
    
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    price = close.iloc[-1]
    
    # === TREND CHECK ===
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    
    if pd.isna(sma20) or price <= sma20:
        return {"signal": "hold", "confidence": 0}  # Must be above 20-day MA
    
    # === MOMENTUM CHECK ===
    returns_5d = (price / close.shift(5).iloc[-1] - 1) if not pd.isna(close.shift(5).iloc[-1]) else 0
    if returns_5d < 0.02:  # Must have positive momentum
        return {"signal": "hold", "confidence": 0}
    
    # === MACD CHECK ===
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9).mean()
    macd_hist = macd - macd_signal
    
    # Bullish cross OR histogram turning positive
    macd_bullish = (macd.iloc[-1] > macd_signal.iloc[-1]) and (macd.iloc[-2] <= macd_signal.iloc[-2])
    macd_positive = macd_hist.iloc[-1] > 0 and macd_hist.iloc[-2] < 0
    
    if not (macd_bullish or macd_positive):
        return {"signal": "hold", "confidence": 0}
    
    # === VOLUME CHECK ===
    vol_ma20 = volume.rolling(20).mean().iloc[-1]
    volume_ratio = volume.iloc[-1] / vol_ma20 if vol_ma20 > 0 else 1
    
    if volume_ratio < 1.2:
        return {"signal": "hold", "confidence": 0}
    
    # === MARKET FILTER ===
    if not market_bullish:
        return {"signal": "hold", "confidence": 0}
    
    # === STOP LOSS (below recent support) ===
    # Use lowest low of last 10 days
    recent_low = low.rolling(10).min().iloc[-1]
    stop_loss = round(recent_low * 0.98, 2)  # 2% below support
    
    # === TARGET (3:1 RR) ===
    risk = price - stop_loss
    take_profit = round(price + (risk * 3), 2)
    
    # === SCORE FOR CONFIDENCE ===
    score = 0
    reasons = []
    
    # Strong trend (price well above MA)
    if price > sma20 * 1.05:
        score += 2
        reasons.append("Strong uptrend")
    
    # Very strong momentum
    if returns_5d > 0.05:
        score += 2
        reasons.append(f"Momentum {returns_5d*100:.1f}%")
    elif returns_5d > 0.02:
        score += 1
    
    # Clear MACD cross
    if macd_bullish:
        score += 2
        reasons.append("MACD cross")
    elif macd_positive:
        score += 1
    
    # Strong volume
    if volume_ratio > 1.5:
        score += 1
        reasons.append(f"Vol {volume_ratio:.1f}x")
    
    # Above 50-day MA too
    if not pd.isna(sma50) and price > sma50:
        score += 1
        reasons.append("Above SMA50")
    
    if score >= 4:  # High confidence
        confidence = min(score / 10, 1.0)
        
        return {
            "signal": "buy",
            "confidence": confidence,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "reasons": reasons,
            "price": price,
            "sma20": sma20,
            "returns_5d": returns_5d,
            "volume_ratio": volume_ratio
        }
    
    return {"signal": "hold", "confidence": 0}


def run_backtest_momentum():
    """Backtest momentum strategy."""
    
    symbols = ["RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "TCS.NS", "SBIN.NS",
               "BAJFINANCE.NS", "TITAN.NS", "SUNPHARMA.NS", "MARUTI.NS", "M&M.NS",
               "KOTAKBANK.NS", "ICICIBANK.NS", "ADANIPORTS.NS", "HINDUNILVR.NS", "NESTLEIND.NS"]
    
    print("=" * 70)
    print("MOMENTUM/TREND-FOLLOWING STRATEGY BACKTEST")
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
    
    # Calculate market trend (using first symbol as market proxy)
    print("\nCalculating market trends...")
    market_trends = {}
    nifty_symbol = "^NSEI"  # Nifty 50 index
    try:
        nifty = yf.Ticker(nifty_symbol).history(period="2y", interval="1d")
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
        
        # Max 2 positions
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
                
                signal = get_momentum_signal(df_subset, market_bullish)
                
                if signal["signal"] == "buy" and signal["confidence"] >= 0.4:
                    risk = price - signal["stop_loss"]
                    if risk > 0:
                        size = int(400 / risk)  # Increased risk
                        
                        if size > 0:
                            result = broker.execute_buy(sym, size, price,
                                                        signal["stop_loss"],
                                                        signal["take_profit"],
                                                        str(date.date()))
                            if result:
                                trades += 1
                                print(f"  BUY: {sym} @ Rs{price:.0f} | "
                                      f"Mom:{signal['returns_5d']*100:.1f}% | "
                                      f"Conf:{signal['confidence']:.0%}")
    
    print("\n" + "=" * 70)
    print("MOMENTUM STRATEGY RESULTS")
    print("=" * 70)
    
    summary = broker.get_summary()
    
    print(f"""
Strategy: Momentum/Trend-Following
Entry:
  - Price above 20-day MA (breakout)
  - MACD bullish cross/positive
  - Volume > 1.2x
  - 5-day momentum > 2%
  - Market in uptrend
Stop: Below recent 10-day low (2%)
Target: 3:1 RR

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
    run_backtest_momentum()