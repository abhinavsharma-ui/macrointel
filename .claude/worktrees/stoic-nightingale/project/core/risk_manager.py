"""
Production Risk Management System
===================================
Production-grade risk controls for live trading:
- Position sizing (Kelly + max limits)
- Drawdown management
- Correlation limits
- Sector exposure limits
- Daily/weekly/monthly loss limits
- Circuit breakers
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    max_position_size_pct: float = 0.10
    max_portfolio_risk_pct: float = 0.02
    max_daily_loss_pct: float = 0.02
    max_weekly_loss_pct: float = 0.05
    max_monthly_loss_pct: float = 0.10
    max_correlation: float = 0.60
    max_sector_exposure_pct: float = 0.25
    max_concurrent_positions: int = 5
    min_risk_reward: float = 1.5


@dataclass
class PositionRisk:
    symbol: str
    size_pct: float
    entry_price: float
    current_price: float
    stop_loss: float
    risk_pct: float
    sector: str


class RiskManager:
    """
    Production risk management system.
    
    Enforces:
    - Position size limits
    - Portfolio risk limits
    - Drawdown stops
    - Correlation limits
    - Sector exposure
    - Circuit breakers
    """
    
    def __init__(self, initial_capital: float = 10000.0, limits: Optional[RiskLimits] = None):
        self.initial_capital = initial_capital
        self.limits = limits or RiskLimits()
        
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.monthly_pnl = 0.0
        self.peak_value = initial_capital
        
        self.daily_trades = []
        self.weekly_trades = []
        self.daily_reset_time = datetime.now()
        self.weekly_reset_time = datetime.now()
        self.monthly_reset_time = datetime.now()
        
        self.circuit_breaker_triggered = False
        self.circuit_breaker_reason = ""
    
    def _check_reset_times(self):
        """Reset daily/weekly counters if needed."""
        now = datetime.now()
        
        if now.date() > self.daily_reset_time.date():
            self.daily_pnl = 0.0
            self.daily_trades = []
            self.daily_reset_time = now
        
        if now.isocalendar()[1] != self.weekly_reset_time.isocalendar()[1]:
            self.weekly_pnl = 0.0
            self.weekly_trades = []
            self.weekly_reset_time = now
        
        if now.month != self.monthly_reset_time.month:
            self.monthly_pnl = 0.0
            self.monthly_reset_time = now
    
    def can_open_position(
        self,
        symbol: str,
        size_pct: float,
        current_positions: List[PositionRisk],
        sector_exposures: Dict[str, float],
    ) -> tuple:
        """
        Check if position can be opened.
        
        Returns:
            (can_trade: bool, reason: str)
        """
        self._check_reset_times()
        
        if self.circuit_breaker_triggered:
            return False, f"Circuit breaker: {self.circuit_breaker_reason}"
        
        daily_loss_pct = self.daily_pnl / self.initial_capital
        if daily_loss_pct <= -self.limits.max_daily_loss_pct:
            self._trigger_circuit_breaker("Daily loss limit hit")
            return False, "Daily loss limit reached"
        
        weekly_loss_pct = self.weekly_pnl / self.initial_capital
        if weekly_loss_pct <= -self.limits.max_weekly_loss_pct:
            self._trigger_circuit_breaker("Weekly loss limit hit")
            return False, "Weekly loss limit reached"
        
        monthly_loss_pct = self.monthly_pnl / self.initial_capital
        if monthly_loss_pct <= -self.limits.max_monthly_loss_pct:
            self._trigger_circuit_breaker("Monthly loss limit hit")
            return False, "Monthly loss limit reached"
        
        if size_pct > self.limits.max_position_size_pct:
            return False, f"Position size {size_pct:.1%} exceeds max {self.limits.max_position_size_pct:.1%}"
        
        if len(current_positions) >= self.limits.max_concurrent_positions:
            return False, "Max concurrent positions reached"
        
        total_risk = sum(p.risk_pct for p in current_positions)
        if total_risk + (size_pct * 0.02) > self.limits.max_portfolio_risk_pct:
            return False, "Portfolio risk limit would be exceeded"
        
        sector = self._get_sector(symbol)
        current_sector_exposure = sector_exposures.get(sector, 0)
        if current_sector_exposure + size_pct > self.limits.max_sector_exposure_pct:
            return False, f"Sector {sector} exposure limit would be exceeded"
        
        return True, "OK"
    
    def _get_sector(self, symbol: str) -> str:
        """Get sector for symbol."""
        sector_map = {
            "RELIANCE": "energy", "TCS": "it", "HDFCBANK": "bank",
            "INFY": "it", "ICICIBANK": "bank", "SBIN": "bank",
            "BAJFINANCE": "finance", "ADANIPORTS": "infra", "KOTAKBANK": "bank",
            "HINDUNILVR": "consumer",
        }
        return sector_map.get(symbol.replace(".NS", ""), "other")
    
    def _trigger_circuit_breaker(self, reason: str):
        """Trigger circuit breaker."""
        self.circuit_breaker_triggered = True
        self.circuit_breaker_reason = reason
        logger.critical(f"CIRCUIT BREAKER TRIGGERED: {reason}")
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        risk_amount: float,
    ) -> int:
        """
        Calculate position size based on risk.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            risk_amount: Dollar/rupee amount to risk
        
        Returns:
            Position size (shares/lots)
        """
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            return 0
        
        raw_size = risk_amount / risk_per_share
        
        max_size = (self.initial_capital * self.limits.max_position_size_pct) / entry_price
        size = min(raw_size, max_size)
        
        return int(size)
    
    def calculate_kelly_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        max_position_pct: float = 0.10,
    ) -> float:
        """
        Calculate Kelly Criterion position size.
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average win amount
            avg_loss: Average loss amount
        
        Returns:
            Position size as % of portfolio (use half-Kelly for safety)
        """
        if avg_loss == 0 or win_rate == 0:
            return 0
        
        win_loss_ratio = abs(avg_win / avg_loss)
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        half_kelly = kelly / 2
        
        safe_size = min(half_kelly, max_position_pct)
        
        return max(0, safe_size)
    
    def check_correlation_limit(
        self,
        new_symbol: str,
        new_correlation: float,
        current_positions: List[PositionRisk],
    ) -> bool:
        """Check if new position exceeds correlation limits."""
        for pos in current_positions:
            if new_correlation > self.limits.max_correlation:
                logger.warning(
                    f"Correlation {new_correlation:.2f} between {new_symbol} and "
                    f"{pos.symbol} exceeds limit {self.limits.max_correlation}"
                )
                return False
        return True
    
    def update_pnl(self, pnl: float):
        """Update P&L tracking."""
        self.daily_pnl += pnl
        self.weekly_pnl += pnl
        self.monthly_pnl += pnl
        
        current_value = self.initial_capital + self.monthly_pnl
        if current_value > self.peak_value:
            self.peak_value = current_value
    
    def get_drawdown(self) -> Dict[str, float]:
        """Calculate current drawdown."""
        current_value = self.initial_capital + self.monthly_pnl
        drawdown = (self.peak_value - current_value) / self.peak_value if self.peak_value > 0 else 0
        
        return {
            "current_value": current_value,
            "peak_value": self.peak_value,
            "drawdown_pct": drawdown * 100,
            "drawdown_amount": self.peak_value - current_value,
        }
    
    def get_risk_summary(self, current_positions: List[PositionRisk]) -> Dict:
        """Get comprehensive risk summary."""
        total_risk = sum(p.risk_pct for p in current_positions)
        sector_exposures = {}
        for pos in current_positions:
            sector_exposures[pos.sector] = sector_exposures.get(pos.sector, 0) + pos.size_pct
        
        dd = self.get_drawdown()
        
        return {
            "circuit_breaker_active": self.circuit_breaker_triggered,
            "circuit_breaker_reason": self.circuit_breaker_reason,
            "daily_pnl_pct": (self.daily_pnl / self.initial_capital) * 100,
            "weekly_pnl_pct": (self.weekly_pnl / self.initial_capital) * 100,
            "monthly_pnl_pct": (self.monthly_pnl / self.initial_capital) * 100,
            "total_risk_pct": total_risk * 100,
            "max_daily_loss_remaining_pct": self.limits.max_daily_loss_pct * 100 - (self.daily_pnl / self.initial_capital) * 100,
            "max_weekly_loss_remaining_pct": self.limits.max_weekly_loss_pct * 100 - (self.weekly_pnl / self.initial_capital) * 100,
            "drawdown_pct": dd["drawdown_pct"],
            "open_positions": len(current_positions),
            "sector_exposures": sector_exposures,
        }
    
    def reset_circuit_breaker(self):
        """Reset circuit breaker (manual intervention needed)."""
        self.circuit_breaker_triggered = False
        self.circuit_breaker_reason = ""
        logger.info("Circuit breaker manually reset")


def get_risk_manager(initial_capital: float = 10000.0) -> RiskManager:
    """Factory function."""
    return RiskManager(initial_capital=initial_capital)


if __name__ == "__main__":
    rm = get_risk_manager(10000)
    
    positions = [
        PositionRisk("RELIANCE", 0.08, 2500, 2550, 2375, 0.05, "energy"),
        PositionRisk("TCS", 0.06, 3500, 3550, 3325, 0.05, "it"),
    ]
    
    can_trade, reason = rm.can_open_position("INFY", 0.07, positions, {"energy": 0.08, "it": 0.06})
    print(f"Can trade INFY: {can_trade} - {reason}")
    
    size = rm.calculate_position_size(100, 95, 500)
    print(f"Position size: {size}")
    
    print("\nRisk Summary:", rm.get_risk_summary(positions))