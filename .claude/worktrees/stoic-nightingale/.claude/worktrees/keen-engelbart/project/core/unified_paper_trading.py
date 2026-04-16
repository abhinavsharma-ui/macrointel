"""
Unified Paper Trading System
==============================
Manages paper trading for both India and Crypto markets.
Consolidated portfolio tracking, risk management, and P&L.
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

from core.paper_trading_india import IndiaPaperBroker, get_india_paper_broker
from core.paper_trading_crypto import CryptoPaperBroker, get_crypto_paper_broker


@dataclass
class UnifiedPortfolio:
    india_broker: IndiaPaperBroker
    crypto_broker: CryptoPaperBroker
    
    @property
    def total_value(self) -> float:
        return self.india_broker.portfolio.total_value + self.crypto_broker.portfolio.total_value
    
    @property
    def total_pnl(self) -> float:
        return self.india_broker.portfolio.total_pnl + self.crypto_broker.portfolio.total_value - self.crypto_broker.portfolio.starting_balance
    
    @property
    def cash_india(self) -> float:
        return self.india_broker.portfolio.cash
    
    @property
    def cash_crypto(self) -> float:
        return self.crypto_broker.portfolio.usdt_balance
    
    @property
    def open_positions_india(self) -> int:
        return self.india_broker.portfolio.open_positions_count
    
    @property
    def open_positions_crypto(self) -> int:
        return len(self.crypto_broker.portfolio.positions)
    
    @property
    def total_trades(self) -> int:
        return len(self.india_broker.portfolio.closed_trades) + len(self.crypto_broker.portfolio.closed_trades)


class UnifiedPaperTrading:
    """
    Unified paper trading system for India + Crypto.
    
    Features:
    - Separate portfolios per market
    - Unified P&L tracking
    - Risk limits per market
    - Automatic SL/TP checking
    - Daily/weekly reports
    """
    
    def __init__(
        self,
        india_capital: float = 5000.0,
        crypto_capital: float = 5000.0,
        max_positions_per_market: int = 3,
        max_total_positions: int = 5,
    ):
        self.india_broker = get_india_paper_broker(india_capital)
        self.crypto_broker = get_crypto_paper_broker(crypto_capital)
        
        self.max_positions_per_market = max_positions_per_market
        self.max_total_positions = max_total_positions
        
        self.daily_loss_limit = 0.05
        self.weekly_loss_limit = 0.10
    
    def can_open_position(self, market: str) -> bool:
        """Check if new position can be opened."""
        if market == "india":
            current = self.india_broker.portfolio.open_positions_count
            return current < self.max_positions_per_market
        elif market == "crypto":
            current = len(self.crypto_broker.portfolio.positions)
            return current < self.max_positions_per_market
        
        total = self.india_broker.portfolio.open_positions_count + len(self.crypto_broker.portfolio.positions)
        return total < self.max_total_positions
    
    def get_available_capital(self, market: str) -> float:
        """Get available capital for market."""
        if market == "india":
            return self.india_broker.portfolio.cash
        return self.crypto_broker.portfolio.usdt_balance
    
    def execute_buy_india(
        self,
        symbol: str,
        quantity: int,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> bool:
        """Execute buy in India market."""
        if not self.can_open_position("india"):
            logger.warning("Max India positions reached")
            return False
        
        result = self.india_broker.execute_buy(symbol, quantity, price, stop_loss, take_profit)
        return result is not None
    
    def execute_buy_crypto(
        self,
        symbol: str,
        quantity: float,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> bool:
        """Execute buy in Crypto market."""
        if not self.can_open_position("crypto"):
            logger.warning("Max Crypto positions reached")
            return False
        
        result = self.crypto_broker.execute_buy(symbol, quantity, price, stop_loss, take_profit)
        return result is not None
    
    def execute_sell_india(self, symbol: str, quantity: int, price: float, reason: str = "signal") -> bool:
        """Execute sell in India market."""
        result = self.india_broker.execute_sell(symbol, quantity, price, reason)
        return result is not None
    
    def execute_sell_crypto(self, symbol: str, quantity: float, price: float, reason: str = "signal") -> bool:
        """Execute sell in Crypto market."""
        result = self.crypto_broker.execute_sell(symbol, quantity, price, reason)
        return result is not None
    
    def check_all_stops(self, india_prices: Dict[str, float], crypto_prices: Dict[str, float]):
        """Check SL/TP for all positions."""
        self.india_broker.check_stop_loss_take_profit(india_prices)
        self.crypto_broker.check_stops(crypto_prices)
    
    def update_india_prices(self, prices: Dict[str, float]):
        """Update India positions."""
        self.india_broker.update_prices(prices)
    
    def update_crypto_prices(self, prices: Dict[str, float]):
        """Update Crypto positions."""
        self.crypto_broker.update_prices(prices)
    
    def get_total_pnl(self) -> float:
        """Get total P&L across markets."""
        return (
            self.india_broker.portfolio.total_pnl 
            + self.crypto_broker.portfolio.total_value 
            - self.crypto_broker.portfolio.starting_balance
        )
    
    def get_summary(self) -> Dict:
        """Get unified summary."""
        india = self.india_broker.get_portfolio_summary()
        crypto = self.crypto_broker.get_summary()
        
        total_pnl = self.get_total_pnl()
        india_pnl = india.get("total_pnl", 0)
        crypto_pnl = crypto.get("total_pnl", 0)
        
        india_trades = india.get("closed_trades", 0)
        crypto_trades = crypto.get("total_trades", 0)
        total_trades = india_trades + crypto_trades
        
        india_wins = int(india_trades * india.get("win_rate", 0) / 100) if india_trades else 0
        crypto_wins = int(crypto_trades * crypto.get("win_rate", 0) / 100) if crypto_trades else 0
        total_wins = india_wins + crypto_wins
        
        win_rate = (total_wins / total_trades * 100) if total_trades else 0
        
        return {
            "total_pnl": round(total_pnl, 2),
            "india_pnl": round(india_pnl, 2),
            "crypto_pnl": round(crypto_pnl, 2),
            "cash_india": round(india.get("cash", 0), 2),
            "cash_crypto": round(crypto.get("balance", 0), 2),
            "india_positions": india.get("open_positions", 0),
            "crypto_positions": crypto.get("open_positions", 0),
            "total_trades": total_trades,
            "win_rate": round(win_rate, 2),
            "india_details": india,
            "crypto_details": crypto,
        }
    
    def check_risk_limits(self) -> Dict[str, bool]:
        """Check if risk limits are hit."""
        summary = self.get_summary()
        
        total_capital = 10000
        current_pnl_pct = summary["total_pnl"] / total_capital
        
        return {
            "daily_loss_hit": current_pnl_pct <= -self.daily_loss_limit,
            "weekly_loss_hit": current_pnl_pct <= -self.weekly_loss_limit,
            "can_trade": current_pnl_pct > -self.weekly_loss_limit,
        }
    
    def reset_all(self):
        """Reset both portfolios."""
        self.india_broker.reset()
        self.crypto_broker.reset()
        logger.info("Both paper trading portfolios reset")


def get_unified_paper_trading(
    india_capital: float = 5000.0,
    crypto_capital: float = 5000.0,
) -> UnifiedPaperTrading:
    """Factory function."""
    return UnifiedPaperTrading(
        india_capital=india_capital,
        crypto_capital=crypto_capital,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    paper = get_unified_paper_trading(5000, 5000)
    
    paper.execute_buy_india("RELIANCE.NS", 10, 2500, stop_loss=2375, take_profit=2750)
    paper.execute_buy_crypto("BTCUSDT", 0.01, 45000, stop_loss=42750, take_profit=49500)
    
    paper.update_india_prices({"RELIANCE.NS": 2600})
    paper.update_crypto_prices({"BTCUSDT": 47000})
    
    print("\n=== Unified Paper Trading Summary ===")
    for k, v in paper.get_summary().items():
        if k not in ["india_details", "crypto_details"]:
            print(f"  {k}: {v}")