"""
Strategy v5 - High Precision Setup Only
========================================
Key insight: We need >70% precision, so only take the highest probability setups.
- RSI < 20 (extreme oversold)
- MACD bullish cross TODAY (not just histogram positive)
- Price above 200-day MA (strong trend)
- Volume > 2x average (institutional interest)
- Market in uptrend (20d return > 0)
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

os.environ["FINNHUB_API_KEY"] = "d78vt3hr01qp0fl6mfjgd78vt3hr01qp0fl6mfk0"


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


def get_high_precision_signal(df: pd.DataFrame, market_trend: bool = True) -> dict:
    """Very selective - only extreme oversold with multiple confirmations."""
    
    if len(df) < 200:
        return {"signal": "hold", "confidence": 0}
    
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    price = close.iloc[-1]
    
    # RSI (must be extreme)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_current = rsi.iloc[-1]
    
    # RSI must be VERY oversold (< 20)
    if rsi_current >= 20:
        return {"signal": "hold", "confidence": 0}
    
    # MACD - MUST have bullish cross TODAY
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9).mean()
    
    macd_cross_today = (macd.iloc[-1] > macd_signal.iloc[-1]) and (macd.iloc[-2] <= macd_signal.iloc[-2])
    
    if not macd_cross_today:
        return {"signal": "hold", "confidence": 0}
    
    # Trend - price above 200-day MA
    sma200 = close.rolling(200).mean().iloc[-1]
    if pd.isna(sma200) or price < sma200:
        return {"signal": "hold", "confidence": 0}
    
    # Also check 50-day MA
    sma50 = close.rolling(50).mean().iloc[-1]
    if pd.isna(sma50) or price < sma50:
        return {"signal": "hold", "confidence": 0}
    
    # Volume - must be 2x average
    vol_ma20 = volume.rolling(20).mean().iloc[-1]
    volume_ratio = volume.iloc[-1] / vol_ma20 if vol_ma20 > 0 else 1
    
    if volume_ratio < 1.5:
        return {"signal": "hold", "confidence": 0}
    
    # Market trend check
    returns_20d = close.pct_change(20).iloc[-1]
    if market_trend and returns_20d < 0:
        return {"signal": "hold", "confidence": 0}
    
    # Calculate stops/targets
    # Tight stop (4%), target (12%) = 3:1 RR
    stop_loss = round(price * 0.96, 2)
    take_profit = round(price * 1.12, 2)
    
    # Confidence based on how extreme the setup is
    rsi_score = (20 - rsi_current) / 20  # 0 to 1
    confidence = 0.5 + (rsi_score * 0.5)
    
    return {
        "signal": "buy",
        "confidence": confidence,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "reasons": [f"RSI extreme ({rsi_current:.0f})", "MACD cross", f"Vol {volume_ratio:.1f}x"],
        "rsi": rsi_current
    }


def run_backtest_v5():
    """Run backtest with high precision strategy."""
    
    symbols = ["RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "TCS.NS", "SBIN.NS", "BAJFINANCE.NS"]
    
    print("=" * 70)
    print("STRATEGY v5 - HIGH PRECISION SETUP ONLY")
    print("=" * 70)
    
    broker = BacktestBroker(5000)
    
    print("\nFetching data...")
    all_data = {}
    
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period="2y", interval="1d")
            if len(df) > 200:
                all_data[sym] = df.dropna()
                print(f"  {sym}: {len(all_data[sym])} days")
        except:
            print(f"  {sym}: Failed")
    
    if not all_data:
        return
    
    all_dates = []
    for sym, df in all_data.items():
        all_dates.extend(df.index.tolist())
    unique_dates = sorted(set(all_dates))
    
    # Calculate market trend for each day
    print("\nCalculating market trends...")
    market_trends = {}
    for date in unique_dates:
        prices = {}
        for sym, df in all_data.items():
            if date in df.index:
                prices[sym] = df.loc[date, "Close"]
        if prices:
            avg_return = np.mean([(df.loc[date, "Close"] / df.loc[date - pd.Timedelta(days=20), "Close"] - 1) 
                                  for sym, df in all_data.items() if date - pd.Timedelta(days=20) in df.index])
            market_trends[date] = avg_return > 0 if not np.isnan(avg_return) else True
    
    print(f"\nSimulating {len(unique_dates)} days...")
    trades = 0
    
    for i, date in enumerate(unique_dates[200:], 1):
        if i % 50 == 0:
            print(f"  Day {i}")
        
        prices = {}
        for sym, df in all_data.items():
            if date in df.index:
                prices[sym] = df.loc[date, "Close"]
        
        if not prices:
            continue
        
        broker.check_exits(prices, str(date.date()))
        
        # Only 1 position at a time for high precision
        if len(broker.positions) < 1:
            market_bullish = market_trends.get(date, True)
            
            for sym, price in prices.items():
                if broker._find_position(sym):
                    continue
                
                if sym not in all_data:
                    continue
                
                df = all_data[sym]
                df_subset = df[df.index <= date]
                
                if len(df_subset) < 200:
                    continue
                
                signal = get_high_precision_signal(df_subset, market_bullish)
                
                if signal["signal"] == "buy":
                    risk = price - signal["stop_loss"]
                    if risk > 0:
                        # Risk 10% of capital for high conviction
                        size = int(500 / risk)
                        
                        if size > 0:
                            result = broker.execute_buy(sym, size, price, 
                                                        signal["stop_loss"], 
                                                        signal["take_profit"],
                                                        str(date.date()))
                            if result:
                                trades += 1
                                print(f"  BUY: {sym} @ Rs{price:.0f} | RSI:{signal['rsi']:.0f}")
    
    print("\n" + "=" * 70)
    print("RESULTS - v5 (High Precision)")
    print("=" * 70)
    
    summary = broker.get_summary()
    
    print(f"""
Strategy: High Precision Setup
- RSI < 20 (extreme oversold)
- MACD bullish cross TODAY
- Price above SMA50 & SMA200
- Volume > 1.5x average
- Market in uptrend
- Stop: 4%, Target: 12% (3:1 RR)
- Risk 10% per trade

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
    run_backtest_v5()