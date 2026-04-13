"""
Improved Trading Strategy - Better Multi-Factor Signals
========================================================
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

# Set API key
os.environ["FINNHUB_API_KEYS"] = "d78vt3hr01qp0fl6mfjgd78vt3hr01qp0fl6mfk0"

# Import components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.paper_trading_india import get_india_paper_broker
from core.risk_manager import get_risk_manager

# Try importing yfinance for data
try:
    import yfinance as yf
    YF_AVAILABLE = True
except:
    YF_AVAILABLE = False
    print("yfinance not available, using alternative")


def get_price_data(symbol: str, period: str = "60d") -> pd.DataFrame:
    """Get price data for symbol."""
    if YF_AVAILABLE:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval="1d")
            return df
        except:
            pass
    return pd.DataFrame()


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators."""
    if df.empty:
        return df
    
    close = df["Close"]
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    
    # Moving averages
    df["sma20"] = close.rolling(20).mean()
    df["sma50"] = close.rolling(50).mean()
    df["sma200"] = close.rolling(200).mean() if len(close) >= 200 else df["sma50"]
    
    # Volume
    df["volume_ma20"] = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_ma20"]
    
    # ATR
    high = df["High"]
    low = df["Low"]
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr"] / close
    
    # Momentum
    df["returns_5d"] = close.pct_change(5)
    
    return df


def generate_signal(df: pd.DataFrame) -> dict:
    """Generate trading signal with improved logic."""
    if df.empty or len(df) < 30:
        return {"signal": "hold", "confidence": 0}
    
    close = df["Close"]
    rsi = df["rsi"].iloc[-1]
    macd_hist = df["macd_hist"].iloc[-1]
    macd = df["macd"].iloc[-1]
    macd_signal = df["macd_signal"].iloc[-1]
    price = close.iloc[-1]
    
    sma20 = df["sma20"].iloc[-1] if not pd.isna(df["sma20"].iloc[-1]) else price
    sma50 = df["sma50"].iloc[-1] if not pd.isna(df["sma50"].iloc[-1]) else price
    sma200 = df["sma200"].iloc[-1] if not pd.isna(df["sma200"].iloc[-1]) else sma50
    
    volume_ratio = df["volume_ratio"].iloc[-1]
    returns_5d = df["returns_5d"].iloc[-1]
    atr_pct = df["atr_pct"].iloc[-1] if not pd.isna(df["atr_pct"].iloc[-1]) else 0.02
    
    # Scoring
    buy_score = 0
    sell_score = 0
    reasons = []
    
    # RSI (key indicator)
    if rsi < 25:
        buy_score += 4
        reasons.append(f"RSI deeply oversold ({rsi:.0f})")
    elif rsi < 30:
        buy_score += 2.5
        reasons.append(f"RSI oversold ({rsi:.0f})")
    elif rsi < 35:
        buy_score += 1
        reasons.append(f"RSI near oversold ({rsi:.0f})")
    
    if rsi > 75:
        sell_score += 4
        reasons.append(f"RSI deeply overbought ({rsi:.0f})")
    elif rsi > 70:
        sell_score += 2.5
        reasons.append(f"RSI overbought ({rsi:.0f})")
    elif rsi > 65:
        sell_score += 1
    
    # MACD (key indicator)
    macd_bullish_cross = (macd > macd_signal) and (df["macd"].iloc[-2] <= df["macd_signal"].iloc[-2])
    macd_bearish_cross = (macd < macd_signal) and (df["macd"].iloc[-2] >= df["macd_signal"].iloc[-2])
    
    if macd_bullish_cross:
        buy_score += 2.5
        reasons.append("MACD bullish cross")
    elif macd_hist > 0 and df["macd_hist"].iloc[-2] < 0:
        buy_score += 1.5
        reasons.append("MACD histogram turning positive")
    
    if macd_bearish_cross:
        sell_score += 2.5
        reasons.append("MACD bearish cross")
    
    # Trend (SMA)
    if price > sma20 > sma50:
        buy_score += 1.5
        reasons.append("Uptrend (price > SMA20 > SMA50)")
    elif price > sma20:
        buy_score += 0.5
    
    if price < sma20 < sma50:
        sell_score += 1.5
        reasons.append("Downtrend")
    
    if price > sma200:
        buy_score += 1
        reasons.append("Above 200-day MA")
    
    # Volume
    if volume_ratio > 1.5:
        buy_score += 1
        reasons.append(f"High volume ({volume_ratio:.1f}x)")
    
    # Momentum
    if returns_5d > 0.03:
        buy_score += 1.5
        reasons.append(f"Strong momentum ({returns_5d*100:.1f}%)")
    elif returns_5d > 0:
        buy_score += 0.5
    
    if returns_5d < -0.03:
        sell_score += 1.5
        reasons.append(f"Weak momentum")
    
    # High volatility reduction
    if atr_pct > 0.04:
        buy_score *= 0.7
        sell_score *= 0.7
        reasons.append("High vol - reduced")
    
    # Decision
    if buy_score >= 5 and buy_score > sell_score:
        confidence = min(buy_score / 10, 1.0)
        stop_loss = price * (1 - 2 * atr_pct)
        take_profit = price * (1 + 3 * atr_pct)
        
        return {
            "signal": "buy",
            "confidence": round(confidence, 2),
            "buy_score": round(buy_score, 1),
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "risk_reward": 1.5,
            "reasons": reasons[:4]
        }
    elif sell_score >= 5 and sell_score > buy_score:
        return {
            "signal": "sell",
            "confidence": min(sell_score / 10, 1.0),
            "sell_score": round(sell_score, 1),
        }
    
    return {"signal": "hold", "confidence": 0}


# Run Paper Trading Test
print("=" * 70)
print("IMPROVED STRATEGY - PAPER TRADING TEST")
print("=" * 70)

broker = get_india_paper_broker(5000)
risk_mgr = get_risk_manager(5000)

symbols = ["RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "TCS.NS", "SBIN.NS", 
           "BAJFINANCE.NS", "KOTAKBANK.NS", "ADANIPORTS.NS"]

print("\nLoading historical data...")
data = {}
for sym in symbols:
    df = get_price_data(sym, "60d")
    if not df.empty:
        df = calculate_indicators(df)
        data[sym] = df
        print(f"  {sym}: {len(df)} days")

# Use latest data for live simulation
print("\nStarting paper trading simulation...")

start_time = time.time()
duration = 180  # 3 minutes for demo

tick = 0
while time.time() - start_time < duration:
    tick += 1
    print(f"\n--- Tick {tick} ---")
    
    # Get latest prices (simulated by last available data)
    prices = {}
    for sym, df in data.items():
        if not df.empty:
            prices[sym] = df["Close"].iloc[-1]
    
    print(f"Prices loaded: {len(prices)}")
    
    broker.update_prices(prices)
    
    # Check stops
    triggers = broker.check_stop_loss_take_profit(prices)
    for t in triggers:
        print(f"STOP: {t['symbol']} - {t['reason']}")
    
    # Generate signals
    if len(broker.portfolio.positions) < 4:
        for sym, price in prices.items():
            if broker._find_position(sym):
                continue
            
            if sym not in data or data[sym].empty:
                continue
            
            signal = generate_signal(data[sym])
            
            if (signal["signal"] == "buy" and 
                signal["confidence"] >= 0.5 and
                signal.get("risk_reward", 0) >= 1.5):
                
                # Position sizing
                size = risk_mgr.calculate_position_size(
                    price, 
                    signal["stop_loss"],
                    250  # 5% risk
                )
                
                if size > 0:
                    result = broker.execute_buy(
                        sym, size, price,
                        stop_loss=signal["stop_loss"],
                        take_profit=signal["take_profit"]
                    )
                    
                    if result:
                        print(f"BUY: {sym} x{size} @ Rs{price}")
                        print(f"  Stop: {signal['stop_loss']}, Target: {signal['take_profit']}")
                        print(f"  Reasons: {signal['reasons']}")
    
    summary = broker.get_portfolio_summary()
    print(f"PnL: Rs{summary['total_pnl']:.2f} | Win: {summary['win_rate']}% | Trades: {summary['closed_trades']}")
    
    # Move time forward (simulate next day)
    for sym in data:
        if not data[sym].empty:
            last_idx = data[sym].index[-1]
            new_idx = last_idx + timedelta(days=1)
            data[sym].loc[new_idx] = data[sym].iloc[-1]
    
    time.sleep(30)

# Final results
print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

summary = broker.get_portfolio_summary()
print(f"""
Starting Capital: Rs5,000
Final Value: Rs{summary['total_value']:.2f}
Total PnL: Rs{summary['total_pnl']:.2f}
Return: {(summary['total_value']/5000-1)*100:.1f}%

Trades: {summary['closed_trades']}
Win Rate: {summary['win_rate']}%
Best Trade: Rs{summary['best_trade']:.2f}
Worst Trade: Rs{summary['worst_trade']:.2f}
Profit Factor: {summary['profit_factor']:.2f}
""")

print("\nStrategy: Multi-factor confirmation (RSI + MACD + Trend + Volume)")
print("Risk Management: 5% max risk, 2x ATR stops, 3x ATR targets")