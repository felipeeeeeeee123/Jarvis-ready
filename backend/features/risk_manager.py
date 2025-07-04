"""Risk management system for JARVIS v3.0 trading."""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import math

from database.services import trade_service, memory_service, config_service
from utils.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


@dataclass
class RiskLimits:
    """Risk management limits configuration."""
    max_position_size: float = 0.05  # 5% of portfolio per position
    max_daily_loss: float = 0.02  # 2% max daily loss
    max_weekly_loss: float = 0.05  # 5% max weekly loss
    max_monthly_loss: float = 0.10  # 10% max monthly loss
    stop_loss_percentage: float = 0.03  # 3% stop loss
    take_profit_percentage: float = 0.06  # 6% take profit
    max_open_positions: int = 10  # Maximum open positions
    max_correlation: float = 0.7  # Max correlation between positions
    min_portfolio_value: float = 100.0  # Minimum portfolio value to trade
    trade_cooldown_minutes: int = 60  # Minutes between trades per symbol


@dataclass
class PositionInfo:
    """Information about an open position."""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    entry_time: datetime
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0


class RiskManager:
    """Advanced risk management for trading operations."""
    
    def __init__(self):
        self.limits = RiskLimits()
        self.load_risk_settings()
        logger.info("Risk manager initialized", extra={"limits": self.limits.__dict__})
    
    def load_risk_settings(self):
        """Load risk settings from database configuration."""
        try:
            # Load custom risk limits from config
            max_pos_size = config_service.get_config("risk.max_position_size", self.limits.max_position_size)
            max_daily_loss = config_service.get_config("risk.max_daily_loss", self.limits.max_daily_loss)
            stop_loss_pct = config_service.get_config("risk.stop_loss_percentage", self.limits.stop_loss_percentage)
            
            self.limits.max_position_size = max_pos_size
            self.limits.max_daily_loss = max_daily_loss
            self.limits.stop_loss_percentage = stop_loss_pct
            
            logger.info("Risk settings loaded from database")
        except Exception as e:
            logger.warning(f"Failed to load risk settings, using defaults: {e}")
    
    def save_risk_settings(self):
        """Save current risk settings to database."""
        try:
            config_service.set_config("risk.max_position_size", self.limits.max_position_size, "Maximum position size as % of portfolio")
            config_service.set_config("risk.max_daily_loss", self.limits.max_daily_loss, "Maximum daily loss as % of portfolio")
            config_service.set_config("risk.stop_loss_percentage", self.limits.stop_loss_percentage, "Stop loss percentage")
            config_service.set_config("risk.take_profit_percentage", self.limits.take_profit_percentage, "Take profit percentage")
            config_service.set_config("risk.max_open_positions", self.limits.max_open_positions, "Maximum open positions")
            
            logger.info("Risk settings saved to database")
        except Exception as e:
            logger.error(f"Failed to save risk settings: {e}")
    
    def validate_trade(self, symbol: str, action: str, quantity: int, price: float, 
                      portfolio_value: float) -> Tuple[bool, str]:
        """Validate if a trade meets risk management criteria."""
        try:
            # Basic validations
            if portfolio_value < self.limits.min_portfolio_value:
                return False, f"Portfolio value ${portfolio_value:.2f} below minimum ${self.limits.min_portfolio_value}"
            
            if quantity <= 0:
                return False, "Invalid quantity"
            
            trade_value = quantity * price
            
            # Position size check
            position_percent = trade_value / portfolio_value
            if position_percent > self.limits.max_position_size:
                return False, f"Position size {position_percent:.1%} exceeds limit {self.limits.max_position_size:.1%}"
            
            # Daily loss check
            daily_pnl = self.get_daily_pnl()
            if daily_pnl < 0 and abs(daily_pnl) / portfolio_value > self.limits.max_daily_loss:
                return False, f"Daily loss limit exceeded: {abs(daily_pnl)/portfolio_value:.1%} > {self.limits.max_daily_loss:.1%}"
            
            # Weekly loss check
            weekly_pnl = self.get_weekly_pnl()
            if weekly_pnl < 0 and abs(weekly_pnl) / portfolio_value > self.limits.max_weekly_loss:
                return False, f"Weekly loss limit exceeded: {abs(weekly_pnl)/portfolio_value:.1%} > {self.limits.max_weekly_loss:.1%}"
            
            # Cooldown check
            if not self.check_trade_cooldown(symbol):
                return False, f"Trade cooldown active for {symbol}"
            
            # Open positions check for buy orders
            if action.lower() == "buy":
                open_positions = self.get_open_positions_count()
                if open_positions >= self.limits.max_open_positions:
                    return False, f"Maximum open positions limit reached: {open_positions}/{self.limits.max_open_positions}"
            
            return True, "Trade approved"
            
        except Exception as e:
            logger.error(f"Trade validation failed: {e}")
            return False, f"Validation error: {e}"
    
    def calculate_position_size(self, symbol: str, price: float, portfolio_value: float,
                               risk_percentage: Optional[float] = None) -> int:
        """Calculate optimal position size based on risk management."""
        try:
            risk_pct = risk_percentage or self.limits.max_position_size
            
            # Calculate base position size
            max_trade_value = portfolio_value * risk_pct
            base_quantity = int(max_trade_value / price)
            
            # Apply additional constraints
            max_dollar_amount = min(max_trade_value, settings.TRADE_CAP)
            constrained_quantity = int(max_dollar_amount / price)
            
            final_quantity = min(base_quantity, constrained_quantity)
            
            logger.info(f"Position size calculated for {symbol}", extra={
                "price": price,
                "portfolio_value": portfolio_value,
                "risk_percentage": risk_pct,
                "calculated_quantity": final_quantity
            })
            
            return max(0, final_quantity)
            
        except Exception as e:
            logger.error(f"Position size calculation failed: {e}")
            return 0
    
    def calculate_stop_loss(self, entry_price: float, action: str) -> float:
        """Calculate stop loss price for a position."""
        if action.lower() == "buy":
            # For long positions, stop loss is below entry price
            return entry_price * (1 - self.limits.stop_loss_percentage)
        else:
            # For short positions, stop loss is above entry price
            return entry_price * (1 + self.limits.stop_loss_percentage)
    
    def calculate_take_profit(self, entry_price: float, action: str) -> float:
        """Calculate take profit price for a position."""
        if action.lower() == "buy":
            # For long positions, take profit is above entry price
            return entry_price * (1 + self.limits.take_profit_percentage)
        else:
            # For short positions, take profit is below entry price
            return entry_price * (1 - self.limits.take_profit_percentage)
    
    def check_stop_loss_trigger(self, symbol: str, current_price: float) -> Optional[Dict]:
        """Check if any positions should trigger stop loss."""
        try:
            # Get recent buy trades for this symbol (open positions)
            recent_trades = trade_service.get_recent_trades(symbol, limit=10)
            
            for trade in recent_trades:
                if trade.action == "buy" and trade.status == "filled":
                    # Calculate if stop loss should trigger
                    stop_loss_price = self.calculate_stop_loss(trade.price, "buy")
                    
                    if current_price <= stop_loss_price:
                        return {
                            "symbol": symbol,
                            "action": "sell",
                            "reason": "stop_loss",
                            "entry_price": trade.price,
                            "current_price": current_price,
                            "stop_loss_price": stop_loss_price,
                            "loss_amount": (trade.price - current_price) * trade.quantity,
                            "loss_percentage": (trade.price - current_price) / trade.price
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Stop loss check failed for {symbol}: {e}")
            return None
    
    def check_take_profit_trigger(self, symbol: str, current_price: float) -> Optional[Dict]:
        """Check if any positions should trigger take profit."""
        try:
            # Get recent buy trades for this symbol (open positions)
            recent_trades = trade_service.get_recent_trades(symbol, limit=10)
            
            for trade in recent_trades:
                if trade.action == "buy" and trade.status == "filled":
                    # Calculate if take profit should trigger
                    take_profit_price = self.calculate_take_profit(trade.price, "buy")
                    
                    if current_price >= take_profit_price:
                        return {
                            "symbol": symbol,
                            "action": "sell",
                            "reason": "take_profit",
                            "entry_price": trade.price,
                            "current_price": current_price,
                            "take_profit_price": take_profit_price,
                            "profit_amount": (current_price - trade.price) * trade.quantity,
                            "profit_percentage": (current_price - trade.price) / trade.price
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Take profit check failed for {symbol}: {e}")
            return None
    
    def get_daily_pnl(self) -> float:
        """Get today's profit/loss."""
        try:
            today = datetime.utcnow().date()
            trades = trade_service.get_recent_trades(limit=100)
            
            daily_pnl = 0.0
            for trade in trades:
                if trade.executed_at.date() == today:
                    if trade.action == "sell":
                        daily_pnl += trade.total_value
                    else:
                        daily_pnl -= trade.total_value
            
            return daily_pnl
            
        except Exception as e:
            logger.error(f"Daily PnL calculation failed: {e}")
            return 0.0
    
    def get_weekly_pnl(self) -> float:
        """Get this week's profit/loss."""
        try:
            week_start = datetime.utcnow() - timedelta(days=7)
            trades = trade_service.get_recent_trades(limit=500)
            
            weekly_pnl = 0.0
            for trade in trades:
                if trade.executed_at >= week_start:
                    if trade.action == "sell":
                        weekly_pnl += trade.total_value
                    else:
                        weekly_pnl -= trade.total_value
            
            return weekly_pnl
            
        except Exception as e:
            logger.error(f"Weekly PnL calculation failed: {e}")
            return 0.0
    
    def get_open_positions_count(self) -> int:
        """Get count of open positions."""
        try:
            # Simplified: count recent buy trades without corresponding sells
            # In a real system, this would track actual portfolio positions
            recent_trades = trade_service.get_recent_trades(limit=100)
            
            position_count = {}
            for trade in recent_trades:
                symbol = trade.symbol
                if symbol not in position_count:
                    position_count[symbol] = 0
                
                if trade.action == "buy":
                    position_count[symbol] += trade.quantity
                else:
                    position_count[symbol] -= trade.quantity
            
            # Count symbols with positive positions
            open_positions = sum(1 for qty in position_count.values() if qty > 0)
            return open_positions
            
        except Exception as e:
            logger.error(f"Open positions count failed: {e}")
            return 0
    
    def check_trade_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in trade cooldown period."""
        try:
            cooldown_minutes = self.limits.trade_cooldown_minutes
            cutoff_time = datetime.utcnow() - timedelta(minutes=cooldown_minutes)
            
            recent_trades = trade_service.get_recent_trades(symbol, limit=5)
            if recent_trades:
                last_trade = recent_trades[0]
                if last_trade.executed_at > cutoff_time:
                    logger.info(f"Trade cooldown active for {symbol}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Cooldown check failed for {symbol}: {e}")
            return True  # Allow trade if check fails
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk management summary."""
        try:
            return {
                "limits": self.limits.__dict__,
                "current_metrics": {
                    "daily_pnl": self.get_daily_pnl(),
                    "weekly_pnl": self.get_weekly_pnl(),
                    "open_positions": self.get_open_positions_count(),
                },
                "status": "active"
            }
        except Exception as e:
            logger.error(f"Risk summary generation failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def update_limits(self, **kwargs) -> bool:
        """Update risk management limits."""
        try:
            for key, value in kwargs.items():
                if hasattr(self.limits, key):
                    setattr(self.limits, key, value)
                    logger.info(f"Updated risk limit {key} to {value}")
            
            self.save_risk_settings()
            return True
            
        except Exception as e:
            logger.error(f"Failed to update risk limits: {e}")
            return False


# Global risk manager instance
risk_manager = RiskManager()