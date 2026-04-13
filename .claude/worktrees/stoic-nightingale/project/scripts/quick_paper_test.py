"""
Quick Paper Trading Test
==========================
Runs a short paper trading session to test the system.
"""

import logging
import os
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# Set API keys
os.environ["FINNHUB_API_KEYS"] = "d78vt3hr01qp0fl6mfjgd78vt3hr01qp0fl6mfk0"

from core.paper_trading_india import get_india_paper_broker
from core.paper_trading_crypto import get_crypto_paper_broker
from core.risk_manager import get_risk_manager
from pipeline.live_data_connector import get_live_data_connector
from pipeline.india_features import get_india_feature_engineer

def run_test(duration_minutes: int = 5):
    """Run paper trading test."""
    logger.info(f"Starting {duration_minutes}-minute paper trading test")
    logger.info("=" * 50)
    
    # Initialize components
    india_broker = get_india_paper_broker(5000)
    crypto_broker = get_crypto_paper_broker(5000)
    risk_manager = get_risk_manager(10000)
    data_connector = get_live_data_connector()
    india_features = get_india_feature_engineer()
    
    # Test symbols
    india_symbols = ["RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "TCS.NS", "SBIN.NS"]
    crypto_symbols = ["BTC-USD", "ETH-USD"]
    
    # Track trades
    trades_executed = 0
    
    import time
    start_time = time.time()
    tick = 0
    
    while (time.time() - start_time) < duration_minutes * 60:
        tick += 1
        logger.info(f"\n--- Tick {tick} ---")
        
        # Fetch prices
        india_prices = {}
        for sym in india_symbols:
            quote = data_connector.get_quote(sym)
            if quote:
                india_prices[sym] = quote["price"]
        
        crypto_prices = {}
        for sym in crypto_symbols:
            quote = data_connector.get_quote(sym)
            if quote:
                crypto_prices[sym] = quote["price"]
        
        logger.info(f"India prices: {india_prices}")
        
        # Update broker prices
        india_broker.update_prices(india_prices)
        crypto_broker.update_prices(crypto_prices)
        
        # Check stops
        india_triggers = india_broker.check_stop_loss_take_profit(india_prices)
        crypto_triggers = crypto_broker.check_stops(crypto_prices)
        
        for t in india_triggers + crypto_triggers:
            logger.info(f"STOP TRIGGERED: {t}")
        
        # Generate simple signals
        risk_ok, _ = risk_manager.can_open_position("TEST", 0.05, [], {})
        
        if risk_ok and len(india_broker.portfolio.positions) < 3:
            # Simple momentum signal
            for symbol, price in india_prices.items():
                if india_broker._find_position(symbol):
                    continue
                
                history = data_connector.get_historical(symbol, period="5d", interval="1h")
                if history is not None and len(history) > 20:
                    close = history["Close"]
                    rsi = _calc_rsi(close)
                    ma20 = close.rolling(20).mean().iloc[-1]
                    
                    # Buy signal: RSI < 35 and price above MA20
                    if rsi < 35 and price > ma20:
                        # Calculate position size
                        size = risk_manager.calculate_position_size(price, price * 0.95, 500)
                        if size > 0:
                            result = india_broker.execute_buy(
                                symbol, size, price,
                                stop_loss=price * 0.95,
                                take_profit=price * 1.10
                            )
                            if result:
                                trades_executed += 1
                                logger.info(f"BUY {symbol}: {size} @ ₹{price}")
                                risk_manager.update_pnl(0)
        
        # Log status
        india_summary = india_broker.get_portfolio_summary()
        crypto_summary = crypto_broker.get_summary()
        
        logger.info(
            f"India: ₹{india_summary['total_pnl']:.2f} ({india_summary['win_rate']:.0f}% win), "
            f"Crypto: ${crypto_summary['total_pnl']:.2f} ({crypto_summary['win_rate']:.0f}% win)"
        )
        
        # Wait
        time.sleep(60)
    
    # Final report
    logger.info("\n" + "=" * 50)
    logger.info("FINAL REPORT")
    logger.info("=" * 50)
    
    india_summary = india_broker.get_portfolio_summary()
    crypto_summary = crypto_broker.get_summary()
    
    print(f"""
=== Paper Trading Results ===

India Market:
  Starting Capital: ₹5,000
  Final Value: ₹{india_summary['total_value']:.2f}
  P&L: ₹{india_summary['total_pnl']:.2f}
  Win Rate: {india_summary['win_rate']:.1f}%
  Total Trades: {india_summary['closed_trades']}
  Best Trade: ₹{india_summary['best_trade']:.2f}
  Worst Trade: ₹{india_summary['worst_trade']:.2f}

Crypto Market:
  Starting Capital: $5,000
  Final Value: ${crypto_summary['total_value']:.2f}
  P&L: ${crypto_summary['total_pnl']:.2f}
  Win Rate: {crypto_summary['win_rate']:.1f}%
  Total Trades: {crypto_summary['total_trades']}

Total Trades Executed: {trades_executed}
""")
    
    return india_summary, crypto_summary


def _calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50


if __name__ == "__main__":
    import numpy as np
    run_test(duration_minutes=5)