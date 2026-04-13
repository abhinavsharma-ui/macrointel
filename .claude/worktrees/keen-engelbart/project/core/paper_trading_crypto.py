"""
Paper Trading Simulator - Crypto
=================================
Simulates crypto trading without real exchange.
Realistic modeling of:
- 24/7 markets
- Funding rates
- Slippage (0.05-0.2%)
- Maker/taker fees
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import random

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CryptoPosition:
    symbol: str
    direction: str
    quantity: float
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class CryptoTrade:
    symbol: str
    direction: str
    quantity: float
    price: float
    timestamp: datetime
    pnl: float = 0.0
    holding_hours: float = 0.0
    exit_reason: str = ""


@dataclass
class CryptoPortfolio:
    usdt_balance: float
    positions: List[CryptoPosition] = field(default_factory=list)
    closed_trades: List[CryptoTrade] = field(default_factory=list)
    starting_balance: float = 0.0
    
    @property
    def total_value(self) -> float:
        pos_pnl = sum(p.unrealized_pnl for p in self.positions)
        return self.usdt_balance + pos_pnl
    
    @property
    def total_pnl(self) -> float:
        closed_pnl = sum(t.pnl for t in self.closed_trades)
        open_pnl = sum(p.unrealized_pnl for p in self.positions)
        return closed_pnl + open_pnl


class CryptoPaperBroker:
    """
    Paper broker for Crypto markets.
    Simulates realistic crypto execution:
    - Slippage: 0.05-0.2%
    - Maker fee: 0.02%
    - Taker fee: 0.06%
    - Funding rates (simulated)
    """
    
    MAKER_FEE = 0.0002
    TAKER_FEE = 0.0006
    SLIPPAGE_RANGE = (0.0005, 0.002)
    
    def __init__(self, initial_balance: float = 5000.0, base_currency: str = "USDT"):
        self.base_currency = base_currency
        self.portfolio = CryptoPortfolio(
            usdt_balance=initial_balance,
            starting_balance=initial_balance
        )
    
    def _get_slippage_price(self, price: float, direction: str) -> float:
        slippage = random.uniform(*self.SLIPPAGE_RANGE)
        if direction == "buy":
            return price * (1 + slippage)
        return price * (1 - slippage)
    
    def _calculate_fees(self, price: float, quantity: float, side: str) -> Dict[str, float]:
        turnover = price * quantity
        fee = self.MAKER_FEE if side == "buy" else self.TAKER_FEE
        fee_amount = turnover * fee
        return {
            "turnover": turnover,
            "fee": fee_amount,
            "fee_rate": fee,
        }
    
    def _find_position(self, symbol: str) -> Optional[CryptoPosition]:
        for pos in self.portfolio.positions:
            if pos.symbol == symbol:
                return pos
        return None
    
    def can_trade(self, symbol: str, direction: str, quantity: float, price: float) -> bool:
        """Check if trade is possible."""
        fees = self._calculate_fees(price, quantity, direction)
        
        if direction == "buy":
            required = (price * quantity) + fees["fee"]
            return self.portfolio.usdt_balance >= required
        
        position = self._find_position(symbol)
        return position is not None and position.direction == direction and position.quantity >= quantity
    
    def execute_buy(
        self,
        symbol: str,
        quantity: float,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Optional[CryptoTrade]:
        """Execute buy order."""
        if not self.can_trade(symbol, "buy", quantity, price):
            logger.warning(f"Insufficient balance for buy: {symbol}")
            return None
        
        slip_price = self._get_slippage_price(price, "buy")
        fees = self._calculate_fees(slip_price, quantity, "buy")
        
        total_cost = (slip_price * quantity) + fees["fee"]
        self.portfolio.usdt_balance -= total_cost
        
        position = CryptoPosition(
            symbol=symbol,
            direction="buy",
            quantity=quantity,
            entry_price=slip_price,
            entry_time=datetime.utcnow(),
            stop_loss=stop_loss or (slip_price * 0.95),
            take_profit=take_profit or (slip_price * 1.10),
            current_price=slip_price,
        )
        self.portfolio.positions.append(position)
        
        logger.info(f"BUY {quantity:.4f} {symbol} @ ${slip_price:.2f}, Total: ${total_cost:.2f}")
        
        return CryptoTrade(
            symbol=symbol,
            direction="buy",
            quantity=quantity,
            price=slip_price,
            timestamp=position.entry_time,
        )
    
    def execute_sell(
        self,
        symbol: str,
        quantity: float,
        price: float,
        reason: str = "signal",
    ) -> Optional[CryptoTrade]:
        """Execute sell order."""
        position = self._find_position(symbol)
        if not position or position.direction != "buy":
            logger.warning(f"No long position: {symbol}")
            return None
        
        quantity = min(quantity, position.quantity)
        slip_price = self._get_slippage_price(price, "sell")
        fees = self._calculate_fees(slip_price, quantity, "sell")
        
        proceeds = (slip_price * quantity) - fees["fee"]
        pnl = proceeds - (position.entry_price * quantity)
        
        self.portfolio.usdt_balance += proceeds
        
        holding_hours = (datetime.utcnow() - position.entry_time).total_seconds() / 3600
        
        trade = CryptoTrade(
            symbol=symbol,
            direction="sell",
            quantity=quantity,
            price=slip_price,
            timestamp=datetime.utcnow(),
            pnl=pnl,
            holding_hours=holding_hours,
            exit_reason=reason,
        )
        
        position.quantity -= quantity
        if position.quantity == 0:
            self.portfolio.positions.remove(position)
        
        self.portfolio.closed_trades.append(trade)
        
        logger.info(f"SELL {quantity:.4f} {symbol} @ ${slip_price:.2f}, PnL: ${pnl:.2f}")
        return trade
    
    def update_prices(self, prices: Dict[str, float]):
        """Update positions with current prices."""
        for pos in self.portfolio.positions:
            if pos.symbol in prices:
                pos.current_price = prices[pos.symbol]
                pos.unrealized_pnl = (pos.current_price - pos.entry_price) * pos.quantity
    
    def check_stops(self, prices: Dict[str, float]) -> List[Dict]:
        """Check SL/TP triggers."""
        self.update_prices(prices)
        triggers = []
        
        for pos in list(self.portfolio.positions):
            if pos.direction != "buy":
                continue
            
            current = pos.current_price
            
            if current <= pos.stop_loss:
                self.execute_sell(pos.symbol, pos.quantity, current, "stop_loss")
                triggers.append({"symbol": pos.symbol, "reason": "sl", "price": current})
                
            elif current >= pos.take_profit:
                self.execute_sell(pos.symbol, pos.quantity, current, "take_profit")
                triggers.append({"symbol": pos.symbol, "reason": "tp", "price": current})
        
        return triggers
    
    def get_summary(self) -> Dict:
        """Get portfolio summary."""
        trades = self.portfolio.closed_trades
        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl <= 0]
        
        win_rate = len(winners) / len(trades) if trades else 0
        avg_win = np.mean([t.pnl for t in winners]) if winners else 0
        avg_loss = np.mean([t.pnl for t in losers]) if losers else 0
        
        return {
            "balance": round(self.portfolio.usdt_balance, 2),
            "total_value": round(self.portfolio.total_value, 2),
            "total_pnl": round(self.portfolio.total_pnl, 2),
            "open_positions": len(self.portfolio.positions),
            "total_trades": len(trades),
            "win_rate": round(win_rate * 100, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else 0,
        }
    
    def reset(self):
        """Reset portfolio."""
        self.portfolio = CryptoPortfolio(
            usdt_balance=self.portfolio.starting_balance,
            starting_balance=self.portfolio.starting_balance
        )


def get_crypto_paper_broker(initial_balance: float = 5000.0) -> CryptoPaperBroker:
    """Factory function."""
    return CryptoPaperBroker(initial_balance=initial_balance)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    broker = get_crypto_paper_broker(5000)
    
    broker.execute_buy("BTCUSDT", 0.01, 45000, stop_loss=42750, take_profit=49500)
    broker.update_prices({"BTCUSDT": 47000})
    
    print("\nCrypto Portfolio:")
    for k, v in broker.get_summary().items():
        print(f"  {k}: {v}")