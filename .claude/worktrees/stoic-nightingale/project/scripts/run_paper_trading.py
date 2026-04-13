"""
Real-Time Paper Trading Runner
================================
Runs paper trading with live market data:
- Fetches live prices every interval
- Evaluates signals from signal engine
- Executes paper trades
- Tracks P&L in real-time
- Generates reports
"""

import logging
import os
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from core.unified_paper_trading import get_unified_paper_trading, UnifiedPaperTrading
from core.risk_manager import get_risk_manager, RiskManager
from pipeline.live_data_connector import get_live_data_connector, LiveDataConnector
from pipeline.india_features import get_india_feature_engineer


class PaperTradingRunner:
    """
    Real-time paper trading runner.
    
    Workflow:
    1. Fetch live prices
    2. Get signals from signal engine
    3. Check risk limits
    4. Execute paper trades
    5. Monitor positions (SL/TP)
    6. Track P&L
    7. Generate reports
    """
    
    def __init__(
        self,
        india_capital: float = 5000,
        crypto_capital: float = 5000,
        tick_interval: int = 60,
        symbols_india: List[str] = None,
        symbols_crypto: List[str] = None,
    ):
        self.india_capital = india_capital
        self.crypto_capital = crypto_capital
        self.tick_interval = tick_interval
        
        self.india_symbols = symbols_india or [
            "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
            "SBIN.NS", "BAJFINANCE.NS", "ADANIPORTS.NS", "KOTAKBANK.NS", "HINDUNILVR.NS",
        ]
        
        self.crypto_symbols = symbols_crypto or [
            "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
        ]
        
        self.paper_trading = get_unified_paper_trading(india_capital, crypto_capital)
        self.risk_manager = get_risk_manager(india_capital + crypto_capital)
        self.data_connector = get_live_data_connector()
        self.india_engineer = get_india_feature_engineer()
        
        self.running = False
        self.current_prices_india: Dict[str, float] = {}
        self.current_prices_crypto: Dict[str, float] = {}
        
        self.signals = {}
        self.trade_log = []
        
        self.save_dir = Path("data/paper_trading")
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def start(self, duration_minutes: Optional[int] = None):
        """Start paper trading."""
        logger.info(f"Starting paper trading for {duration_minutes or 'indefinite'} minutes")
        logger.info(f"India capital: ₹{self.india_capital}, Crypto: ${self.crypto_capital}")
        
        self.running = True
        
        start_time = time.time()
        tick_count = 0
        
        while self.running:
            try:
                self._tick(tick_count)
                tick_count += 1
                
                if duration_minutes and (time.time() - start_time) > duration_minutes * 60:
                    logger.info("Duration reached, stopping")
                    break
                    
            except Exception as e:
                logger.error(f"Tick error: {e}")
            
            time.sleep(self.tick_interval)
        
        self._generate_report()
    
    def _tick(self, tick_count: int):
        """Process one tick."""
        now = datetime.now()
        logger.info(f"Tick {tick_count} at {now.strftime('%H:%M:%S')}")
        
        self._fetch_prices()
        
        self._check_stops()
        
        self._generate_signals()
        
        self._execute_signals()
        
        self._log_status()
    
    def _fetch_prices(self):
        """Fetch live prices."""
        logger.info("Fetching live prices...")
        
        india_quotes = self.data_connector.get_quotes_batch(self.india_symbols)
        self.current_prices_india = {
            symbol: data["price"] 
            for symbol, data in india_quotes.items() 
            if data and data.get("price")
        }
        
        crypto_quotes = self.data_connector.get_quotes_batch(self.crypto_symbols)
        self.current_prices_crypto = {
            symbol: data["price"]
            for symbol, data in crypto_quotes.items()
            if data and data.get("price")
        }
        
        logger.info(f"India prices: {len(self.current_prices_india)} symbols")
        logger.info(f"Crypto prices: {len(self.current_prices_crypto)} symbols")
    
    def _check_stops(self):
        """Check stop loss / take profit."""
        india_triggers = self.paper_trading.check_all_stops(
            self.current_prices_india,
            self.current_prices_crypto
        )
        
        for trigger in india_triggers:
            logger.info(f"Stop triggered: {trigger['symbol']} - {trigger['reason']}")
            self.trade_log.append({
                "time": datetime.now().isoformat(),
                "type": "stop",
                **trigger
            })
    
    def _generate_signals(self):
        """Generate trading signals (simplified)."""
        self.signals = {}
        
        for symbol, price in self.current_prices_india.items():
            signal = self._generate_india_signal(symbol, price)
            if signal:
                self.signals[symbol] = signal
    
    def _generate_india_signal(self, symbol: str, price: float) -> Optional[Dict]:
        """Generate simplified signal for Indian stock."""
        
        history = self.data_connector.get_historical(symbol, period="5d", interval="1h")
        if history is None or len(history) < 10:
            return None
        
        close = history["Close"]
        rsi = self._calculate_rsi(close, 14)
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        
        if len(close) < 2:
            return None
        
        momentum = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]
        
        signal = "hold"
        if rsi < 35 and close.iloc[-1] > ma20.iloc[-1]:
            signal = "buy"
        elif rsi > 65 and close.iloc[-1] < ma20.iloc[-1]:
            signal = "sell"
        
        if signal == "hold":
            return None
        
        entry = price
        stop = entry * 0.95 if signal == "buy" else entry * 1.05
        target = entry * 1.10 if signal == "buy" else entry * 0.90
        
        return {
            "signal": signal,
            "price": entry,
            "stop_loss": stop,
            "take_profit": target,
            "rsi": rsi,
            "momentum": momentum,
            "above_ma20": close.iloc[-1] > ma20.iloc[-1] if not pd.isna(ma20.iloc[-1]) else False,
        }
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> float:
        """Calculate RSI."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    
    def _execute_signals(self):
        """Execute trading signals."""
        risk_summary = self.risk_manager.get_risk_summary([])
        
        if not risk_summary["can_trade"]:
            logger.warning(f"Cannot trade: {risk_summary}")
            return
        
        for symbol, signal_data in self.signals.items():
            if symbol in self.current_prices_india:
                self._execute_india_signal(symbol, signal_data)
    
    def _execute_india_signal(self, symbol: str, signal: Dict):
        """Execute India signal."""
        can_trade, reason = self.risk_manager.can_open_position(
            symbol, 0.05, [], {}
        )
        
        if not can_trade:
            logger.info(f"Skipping {symbol}: {reason}")
            return
        
        price = signal["price"]
        
        if signal["signal"] == "buy":
            quantity = int(self.risk_manager.calculate_position_size(
                price, signal["stop_loss"], self.india_capital * 0.02
            ))
            
            if quantity > 0:
                success = self.paper_trading.execute_buy_india(
                    symbol, quantity, price,
                    stop_loss=signal["stop_loss"],
                    take_profit=signal["take_profit"],
                )
                
                if success:
                    logger.info(f"Executed BUY {symbol}: {quantity} @ ₹{price}")
                    self.trade_log.append({
                        "time": datetime.now().isoformat(),
                        "type": "buy",
                        "symbol": symbol,
                        "quantity": quantity,
                        "price": price,
                    })
    
    def _log_status(self):
        """Log current status."""
        summary = self.paper_trading.get_summary()
        
        logger.info(
            f"Portfolio: ₹{summary['india_pnl']:.2f} (India), "
            f"${summary['crypto_pnl']:.2f} (Crypto), "
            f"Win Rate: {summary['win_rate']:.1f}%"
        )
    
    def _generate_report(self):
        """Generate final report."""
        summary = self.paper_trading.get_summary()
        
        report_path = self.save_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, "w") as f:
            import json
            json.dump({
                "summary": summary,
                "trades": self.trade_log,
                "duration_ticks": len(self.trade_log),
            }, f, indent=2)
        
        logger.info(f"Report saved to {report_path}")
        logger.info(f"Final P&L: India ₹{summary['india_pnl']}, Crypto ${summary['crypto_pnl']}")
        logger.info(f"Win Rate: {summary['win_rate']}%")
    
    def stop(self):
        """Stop paper trading."""
        self.running = False
        logger.info("Paper trading stopped")


def run_paper_trading(
    india_capital: float = 5000,
    crypto_capital: float = 5000,
    duration_minutes: int = 60,
    tick_interval: int = 60,
):
    """
    Run paper trading.
    
    Args:
        india_capital: Starting capital for India (₹)
        crypto_capital: Starting capital for Crypto ($)
        duration_minutes: How long to run (minutes)
        tick_interval: Seconds between ticks
    """
    runner = PaperTradingRunner(
        india_capital=india_capital,
        crypto_capital=crypto_capital,
        tick_interval=tick_interval,
    )
    
    try:
        runner.start(duration_minutes=duration_minutes)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        runner.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    print("Starting 10-minute paper trading test...")
    run_paper_trading(
        india_capital=5000,
        crypto_capital=5000,
        duration_minutes=10,
        tick_interval=60,
    )