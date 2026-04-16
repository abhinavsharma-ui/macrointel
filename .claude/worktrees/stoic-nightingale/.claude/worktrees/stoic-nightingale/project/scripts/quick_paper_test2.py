"""
Quick Paper Trading Test - Improved
=====================================
"""

import logging
import os
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

os.environ["FINNHUB_API_KEYS"] = "d78vt3hr01qp0fl6mfjgd78vt3hr01qp0fl6mfk0"

from core.paper_trading_india import get_india_paper_broker
from core.risk_manager import get_risk_manager
from pipeline.live_data_connector import get_live_data_connector
import pandas as pd
import numpy as np

def _rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0).rolling(p).mean()
    l = (-d.where(d < 0, 0)).rolling(p).mean()
    r = g / l
    return 100 - (100 / (1 + r))

def _macd(s):
    e12 = s.ewm(span=12).mean()
    e26 = s.ewm(span=26).mean()
    return e12 - e26

print("=" * 60)
print("PAPER TRADING TEST (5 minutes)")
print("=" * 60)

india_broker = get_india_paper_broker(5000)
risk_manager = get_risk_manager(5000)
data_connector = get_live_data_connector()

symbols = ["RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "TCS.NS", "SBIN.NS", "BAJFINANCE.NS", "KOTAKBANK.NS", "ADANIPORTS.NS"]

start_time = time.time()
duration = 5 * 60
tick = 0

while time.time() - start_time < duration:
    tick += 1
    print(f"\n--- Tick {tick} ---")
    
    prices = {}
    for sym in symbols:
        q = data_connector.get_quote(sym)
        if q:
            prices[sym] = q["price"]
    
    print(f"Prices: {len(prices)} loaded")
    
    india_broker.update_prices(prices)
    
    triggers = india_broker.check_stop_loss_take_profit(prices)
    for t in triggers:
        print(f"STOP: {t}")
    
    if len(india_broker.portfolio.positions) < 3:
        for sym, price in prices.items():
            if india_broker._find_position(sym):
                continue
            
            hist = data_connector.get_historical(sym, period="10d", interval="1h")
            if hist is not None and len(hist) > 20:
                close = hist["Close"]
                
                rsi_14 = _rsi(close, 14).iloc[-1]
                rsi_9 = _rsi(close, 9).iloc[-1]
                macd_val = _macd(close).iloc[-1]
                ma20 = close.rolling(20).mean().iloc[-1]
                ma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else ma20
                momentum = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]
                
                if (rsi_14 < 35 and rsi_9 < 40 and 
                    macd_val > 0 and close.iloc[-1] > ma20 and close.iloc[-1] > ma50 and
                    momentum > 0):
                    
                    size = risk_manager.calculate_position_size(price, price * 0.95, 250)
                    if size > 0:
                        result = india_broker.execute_buy(
                            sym, size, price,
                            stop_loss=price * 0.95,
                            take_profit=price * 1.12
                        )
                        if result:
                            print(f"BUY: {sym} x{size} @ Rs{price}")
    
    summary = india_broker.get_portfolio_summary()
    print(f"PnL: Rs{summary['total_pnl']:.2f} | Win: {summary['win_rate']}% | Trades: {summary['closed_trades']}")
    
    time.sleep(60)

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

summary = india_broker.get_portfolio_summary()
print(f"""
India Paper Trading (5 min test):

Starting Capital: Rs5,000
Final Value: Rs{summary['total_value']:.2f}
PnL: Rs{summary['total_pnl']:.2f}
Win Rate: {summary['win_rate']}%

Trades: {summary['closed_trades']}
Best Trade: Rs{summary['best_trade']:.2f}
Worst Trade: Rs{summary['worst_trade']:.2f}
Profit Factor: {summary['profit_factor']:.2f}
""")