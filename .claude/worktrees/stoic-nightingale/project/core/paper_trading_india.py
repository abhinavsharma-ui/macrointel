"""
Paper Trading Simulator - India
================================
Simulates India stock trading without real broker.
Uses NSE data + realistic execution modeling.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import random

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    direction: str
    quantity: int
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class Trade:
    symbol: str
    direction: str
    quantity: int
    price: float
    timestamp: datetime
    pnl: float = 0.0
    holding_period: int = 0
    exit_reason: str = ""


@dataclass
class Portfolio:
    cash: float
    positions: List[Position] = field(default_factory=list)
    closed_trades: List[Trade] = field(default_factory=list)
    starting_cash: float = 0.0
    
    @property
    def total_value(self) -> float:
        pos_value = sum(p.unrealized_pnl for p in self.positions)
        return self.cash + pos_value
    
    @property
    def total_pnl(self) -> float:
        closed_pnl = sum(t.pnl for t in self.closed_trades)
        open_pnl = sum(p.unrealized_pnl for p in self.positions)
        return closed_pnl + open_pnl
    
    @property
    def open_positions_count(self) -> int:
        return len(self.positions)


class IndiaPaperBroker:
    """
    Paper broker for Indian market.
    Simulates realistic NSE execution with:
    - Market hours (9:15-15:30 IST)
    - 0.1-0.5% slippage
    - Brokerage simulation (₹0-20 per trade)
    - GST on brokerage (18%)
    """
    
    MARKET_OPEN = 9 * 60 + 15
    MARKET_CLOSE = 15 * 60 + 30
    BROKERAGE_PER_TRADE = 15.0
    GST_RATE = 0.18
    
    def __init__(self, initial_cash: float = 5000.0):
        self.portfolio = Portfolio(cash=initial_cash, starting_cash=initial_cash)
        self._initialize_market_hours()
    
    def _initialize_market_hours(self):
        from zoneinfo import ZoneInfo
        self.ist = ZoneInfo("Asia/Kolkata")
    
    def _is_market_open(self) -> bool:
        now = datetime.now(self.ist)
        if now.weekday() >= 5:
            return False
        
        minutes = now.hour * 60 + now.minute
        return self.MARKET_OPEN <= minutes <= self.MARKET_CLOSE
    
    def _get_slippage(self, price: float, direction: str) -> float:
        base_slippage = random.uniform(0.001, 0.005)
        if direction == "buy":
            return price * (1 + base_slippage)
        return price * (1 - base_slippage)
    
    def _calculate_costs(self, price: float, quantity: int) -> Dict[str, float]:
        turnover = price * quantity
        brokerage = self.BROKERAGE_PER_TRADE
        gst = brokerage * self.GST_RATE
        stamp_duty = turnover * 0.0001
        total_cost = brokerage + gst + stamp_duty
        return {
            "brokerage": brokerage,
            "gst": gst,
            "stamp_duty": stamp_duty,
            "total_cost": total_cost,
        }
    
    def can_trade(self, symbol: str, direction: str, quantity: int, price: float) -> bool:
        """Check if trade is possible."""
        costs = self._calculate_costs(price, quantity)
        
        if direction == "buy":
            required = (price * quantity) + costs["total_cost"]
            return self.portfolio.cash >= required
        
        position = self._find_position(symbol)
        if not position or position.direction != direction:
            return False
        
        return position.quantity >= quantity
    
    def _find_position(self, symbol: str) -> Optional[Position]:
        for pos in self.portfolio.positions:
            if pos.symbol == symbol:
                return pos
        return None
    
    def execute_buy(
        self,
        symbol: str,
        quantity: int,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Optional[Trade]:
        """Execute buy order."""
        if not self.can_trade(symbol, "buy", quantity, price):
            logger.warning(f"Insufficient cash for buy: {symbol}")
            return None
        
        slippage_price = self._get_slippage(price, "buy")
        costs = self._calculate_costs(slippage_price, quantity)
        
        total_cost = (slippage_price * quantity) + costs["total_cost"]
        self.portfolio.cash -= total_cost
        
        position = Position(
            symbol=symbol,
            direction="buy",
            quantity=quantity,
            entry_price=slippage_price,
            entry_time=datetime.now(self.ist),
            stop_loss=stop_loss or (slippage_price * 0.95),
            take_profit=take_profit or (slippage_price * 1.10),
            current_price=slippage_price,
        )
        self.portfolio.positions.append(position)
        
        logger.info(f"BUY {quantity} {symbol} @ {slippage_price:.2f}, cost: ₹{total_cost:.2f}")
        
        return Trade(
            symbol=symbol,
            direction="buy",
            quantity=quantity,
            price=slippage_price,
            timestamp=position.entry_time,
        )
    
    def execute_sell(
        self,
        symbol: str,
        quantity: int,
        price: float,
        reason: str = "signal",
    ) -> Optional[Trade]:
        """Execute sell order."""
        position = self._find_position(symbol)
        if not position or position.direction != "buy":
            logger.warning(f"No long position to sell: {symbol}")
            return None
        
        quantity = min(quantity, position.quantity)
        slippage_price = self._get_slippage(price, "sell")
        costs = self._calculate_costs(slippage_price, quantity)
        
        pnl = (slippage_price - position.entry_price) * quantity - costs["total_cost"]
        self.portfolio.cash += (slippage_price * quantity) - costs["total_cost"]
        
        trade = Trade(
            symbol=symbol,
            direction="sell",
            quantity=quantity,
            price=slippage_price,
            timestamp=datetime.now(self.ist),
            pnl=pnl,
            holding_period=(datetime.now(self.ist) - position.entry_time).days,
            exit_reason=reason,
        )
        
        position.quantity -= quantity
        if position.quantity == 0:
            self.portfolio.positions.remove(position)
        
        self.portfolio.closed_trades.append(trade)
        
        logger.info(f"SELL {quantity} {symbol} @ {slippage_price:.2f}, PnL: ₹{pnl:.2f}")
        return trade
    
    def update_prices(self, prices: Dict[str, float]):
        """Update current prices and calculate P&L."""
        for position in self.portfolio.positions:
            if position.symbol in prices:
                position.current_price = prices[position.symbol]
                position.unrealized_pnl = (
                    position.current_price - position.entry_price
                ) * position.quantity
                
                if position.direction == "sell":
                    position.unrealized_pnl *= -1
    
    def check_stop_loss_take_profit(self, prices: Dict[str, float]) -> List[Dict]:
        """Check for SL/TP triggers."""
        self.update_prices(prices)
        triggers = []
        
        for position in list(self.portfolio.positions):
            if position.direction != "buy":
                continue
                
            current = position.current_price
            
            if current <= position.stop_loss:
                self.execute_sell(position.symbol, position.quantity, current, "stop_loss")
                triggers.append({
                    "symbol": position.symbol,
                    "reason": "stop_loss",
                    "price": current,
                })
                
            elif current >= position.take_profit:
                self.execute_sell(position.symbol, position.quantity, current, "take_profit")
                triggers.append({
                    "symbol": position.symbol,
                    "reason": "take_profit",
                    "price": current,
                })
        
        return triggers
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary."""
        closed_trades = self.portfolio.closed_trades
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        return {
            "cash": round(self.portfolio.cash, 2),
            "total_value": round(self.portfolio.total_value, 2),
            "total_pnl": round(self.portfolio.total_pnl, 2),
            "open_positions": self.portfolio.open_positions_count,
            "closed_trades": len(closed_trades),
            "win_rate": round(win_rate * 100, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else 0,
            "best_trade": round(max([t.pnl for t in closed_trades], default=0), 2),
            "worst_trade": round(min([t.pnl for t in closed_trades], default=0), 2),
        }
    
    def reset(self):
        """Reset portfolio."""
        self.portfolio = Portfolio(
            cash=self.portfolio.starting_cash,
            starting_cash=self.portfolio.starting_cash
        )
        logger.info("Portfolio reset")


def get_india_paper_broker(initial_cash: float = 5000.0) -> IndiaPaperBroker:
    """Factory function."""
    return IndiaPaperBroker(initial_cash=initial_cash)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    broker = get_india_paper_broker(5000)
    
    broker.execute_buy("RELIANCE.NS", 10, 2500, stop_loss=2375, take_profit=2750)
    
    broker.update_prices({"RELIANCE.NS": 2600})
    
    print("\nPortfolio Summary:")
    for k, v in broker.get_portfolio_summary().items():
        print(f"  {k}: {v}")