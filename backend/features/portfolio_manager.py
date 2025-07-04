"""Portfolio management and allocation system for JARVIS v3.0."""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import math

from database.services import trade_service, config_service
from database.models import TradeLog
from utils.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


@dataclass
class Position:
    """Represents a portfolio position."""
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    last_updated: datetime


@dataclass
class PortfolioAllocation:
    """Portfolio allocation configuration."""
    max_positions: int = 10
    max_sector_allocation: float = 0.30  # 30% max per sector
    max_single_position: float = 0.15    # 15% max per position
    min_cash_reserve: float = 0.10       # 10% minimum cash
    rebalance_threshold: float = 0.05    # 5% deviation triggers rebalance


class PortfolioManager:
    """Manages portfolio allocation, positions, and performance tracking."""
    
    def __init__(self):
        self.allocation_config = PortfolioAllocation()
        self.positions: Dict[str, Position] = {}
        self.cash_balance = 0.0
        self.total_value = 0.0
        self.daily_pnl = 0.0
        self.load_allocation_settings()
        logger.info("Portfolio manager initialized")
    
    def load_allocation_settings(self):
        """Load portfolio allocation settings from database."""
        try:
            max_positions = config_service.get_config("portfolio.max_positions", self.allocation_config.max_positions)
            max_single = config_service.get_config("portfolio.max_single_position", self.allocation_config.max_single_position)
            min_cash = config_service.get_config("portfolio.min_cash_reserve", self.allocation_config.min_cash_reserve)
            
            self.allocation_config.max_positions = max_positions
            self.allocation_config.max_single_position = max_single
            self.allocation_config.min_cash_reserve = min_cash
            
            logger.info("Portfolio allocation settings loaded")
        except Exception as e:
            logger.warning(f"Failed to load portfolio settings, using defaults: {e}")
    
    def save_allocation_settings(self):
        """Save portfolio allocation settings to database."""
        try:
            config_service.set_config("portfolio.max_positions", self.allocation_config.max_positions, 
                                    "Maximum number of positions")
            config_service.set_config("portfolio.max_single_position", self.allocation_config.max_single_position,
                                    "Maximum allocation per single position")
            config_service.set_config("portfolio.min_cash_reserve", self.allocation_config.min_cash_reserve,
                                    "Minimum cash reserve percentage")
            config_service.set_config("portfolio.max_sector_allocation", self.allocation_config.max_sector_allocation,
                                    "Maximum allocation per sector")
            
            logger.info("Portfolio allocation settings saved")
        except Exception as e:
            logger.error(f"Failed to save portfolio settings: {e}")
    
    def update_positions_from_trades(self):
        """Update current positions based on trade history."""
        try:
            # Get all trades from the last 90 days
            cutoff_date = datetime.utcnow() - timedelta(days=90)
            trades = trade_service.get_recent_trades(limit=1000)
            
            # Filter recent trades
            recent_trades = [t for t in trades if t.executed_at >= cutoff_date and t.status == "filled"]
            
            # Calculate positions by symbol
            position_data = {}
            
            for trade in recent_trades:
                symbol = trade.symbol
                if symbol not in position_data:
                    position_data[symbol] = {
                        "quantity": 0,
                        "total_cost": 0.0,
                        "trades": []
                    }
                
                position_data[symbol]["trades"].append(trade)
                
                if trade.action == "buy":
                    position_data[symbol]["quantity"] += trade.quantity
                    position_data[symbol]["total_cost"] += trade.total_value
                elif trade.action == "sell":
                    position_data[symbol]["quantity"] -= trade.quantity
                    position_data[symbol]["total_cost"] -= trade.total_value
            
            # Create Position objects for non-zero positions
            self.positions = {}
            for symbol, data in position_data.items():
                if data["quantity"] > 0:  # Only keep long positions
                    avg_cost = data["total_cost"] / data["quantity"] if data["quantity"] > 0 else 0
                    
                    # For now, use the avg_cost as current_price (would need real-time data)
                    current_price = avg_cost
                    market_value = data["quantity"] * current_price
                    unrealized_pnl = market_value - data["total_cost"]
                    unrealized_pnl_percent = (unrealized_pnl / data["total_cost"]) * 100 if data["total_cost"] > 0 else 0
                    
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        quantity=data["quantity"],
                        avg_cost=avg_cost,
                        current_price=current_price,
                        market_value=market_value,
                        unrealized_pnl=unrealized_pnl,
                        unrealized_pnl_percent=unrealized_pnl_percent,
                        last_updated=datetime.utcnow()
                    )
            
            logger.info(f"Updated {len(self.positions)} positions from trade history")
            
        except Exception as e:
            logger.error(f"Failed to update positions: {e}")
    
    def calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive portfolio performance metrics."""
        try:
            self.update_positions_from_trades()
            
            # Calculate total portfolio value
            total_market_value = sum(pos.market_value for pos in self.positions.values())
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            
            # Get trading statistics
            trade_stats = trade_service.get_performance_summary(30)  # Last 30 days
            
            # Calculate daily P&L
            self.daily_pnl = self._calculate_daily_pnl()
            
            # Portfolio composition
            position_allocations = {}
            for symbol, position in self.positions.items():
                allocation_percent = (position.market_value / total_market_value * 100) if total_market_value > 0 else 0
                position_allocations[symbol] = {
                    "quantity": position.quantity,
                    "market_value": position.market_value,
                    "allocation_percent": allocation_percent,
                    "unrealized_pnl": position.unrealized_pnl,
                    "unrealized_pnl_percent": position.unrealized_pnl_percent
                }
            
            return {
                "total_positions": len(self.positions),
                "total_market_value": total_market_value,
                "total_unrealized_pnl": total_unrealized_pnl,
                "daily_pnl": self.daily_pnl,
                "position_allocations": position_allocations,
                "trade_statistics": trade_stats,
                "allocation_limits": {
                    "max_positions": self.allocation_config.max_positions,
                    "max_single_position": self.allocation_config.max_single_position,
                    "min_cash_reserve": self.allocation_config.min_cash_reserve
                },
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Portfolio metrics calculation failed: {e}")
            return {"error": str(e)}
    
    def _calculate_daily_pnl(self) -> float:
        """Calculate today's realized P&L from trades."""
        try:
            today = datetime.utcnow().date()
            trades = trade_service.get_recent_trades(limit=100)
            
            daily_pnl = 0.0
            for trade in trades:
                if trade.executed_at.date() == today and trade.status == "filled":
                    if trade.action == "sell":
                        # This is simplified - would need to track cost basis properly
                        daily_pnl += trade.total_value
                    # Note: Buy trades reduce cash but don't affect P&L until sold
            
            return daily_pnl
            
        except Exception as e:
            logger.error(f"Daily P&L calculation failed: {e}")
            return 0.0
    
    def check_allocation_limits(self, symbol: str, proposed_value: float, 
                              total_portfolio_value: float) -> Tuple[bool, str]:
        """Check if a proposed trade violates allocation limits."""
        try:
            # Check if we're at max positions and this is a new position
            if symbol not in self.positions and len(self.positions) >= self.allocation_config.max_positions:
                return False, f"Maximum positions limit reached: {len(self.positions)}/{self.allocation_config.max_positions}"
            
            # Check single position allocation limit
            proposed_allocation = proposed_value / total_portfolio_value
            if proposed_allocation > self.allocation_config.max_single_position:
                return False, f"Single position allocation {proposed_allocation:.1%} exceeds limit {self.allocation_config.max_single_position:.1%}"
            
            # Check minimum cash reserve
            cash_after_trade = total_portfolio_value - proposed_value
            min_cash_required = total_portfolio_value * self.allocation_config.min_cash_reserve
            if cash_after_trade < min_cash_required:
                return False, f"Trade would violate minimum cash reserve requirement"
            
            return True, "Allocation limits satisfied"
            
        except Exception as e:
            logger.error(f"Allocation limit check failed: {e}")
            return False, f"Allocation check error: {e}"
    
    def suggest_position_size(self, symbol: str, price: float, portfolio_value: float,
                            target_allocation: float = 0.05) -> int:
        """Suggest optimal position size based on allocation strategy."""
        try:
            # Ensure target allocation doesn't exceed limits
            max_allocation = min(target_allocation, self.allocation_config.max_single_position)
            
            # Calculate target dollar amount
            target_value = portfolio_value * max_allocation
            
            # Account for existing position if any
            if symbol in self.positions:
                existing_value = self.positions[symbol].market_value
                additional_value = max(0, target_value - existing_value)
            else:
                additional_value = target_value
            
            # Calculate quantity
            suggested_quantity = int(additional_value / price)
            
            # Validate against allocation limits
            total_trade_value = suggested_quantity * price
            valid, message = self.check_allocation_limits(symbol, total_trade_value, portfolio_value)
            
            if not valid:
                # Reduce quantity to fit within limits
                max_allowed_value = portfolio_value * self.allocation_config.max_single_position * 0.9  # 90% of limit
                suggested_quantity = int(max_allowed_value / price)
            
            logger.info(f"Position size suggestion for {symbol}", extra={
                "suggested_quantity": suggested_quantity,
                "target_allocation": target_allocation,
                "price": price,
                "total_value": suggested_quantity * price
            })
            
            return max(0, suggested_quantity)
            
        except Exception as e:
            logger.error(f"Position size suggestion failed: {e}")
            return 0
    
    def identify_rebalancing_opportunities(self) -> List[Dict[str, Any]]:
        """Identify positions that need rebalancing."""
        try:
            opportunities = []
            total_value = sum(pos.market_value for pos in self.positions.values())
            
            if total_value == 0:
                return opportunities
            
            target_allocation = 1.0 / min(len(self.positions), self.allocation_config.max_positions)
            
            for symbol, position in self.positions.items():
                current_allocation = position.market_value / total_value
                allocation_deviation = abs(current_allocation - target_allocation)
                
                if allocation_deviation > self.allocation_config.rebalance_threshold:
                    target_value = total_value * target_allocation
                    rebalance_amount = target_value - position.market_value
                    
                    opportunities.append({
                        "symbol": symbol,
                        "current_allocation": current_allocation,
                        "target_allocation": target_allocation,
                        "deviation": allocation_deviation,
                        "rebalance_amount": rebalance_amount,
                        "action": "reduce" if rebalance_amount < 0 else "increase"
                    })
            
            # Sort by largest deviation first
            opportunities.sort(key=lambda x: x["deviation"], reverse=True)
            
            logger.info(f"Identified {len(opportunities)} rebalancing opportunities")
            return opportunities
            
        except Exception as e:
            logger.error(f"Rebalancing analysis failed: {e}")
            return []
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary."""
        try:
            metrics = self.calculate_portfolio_metrics()
            rebalancing = self.identify_rebalancing_opportunities()
            
            return {
                "metrics": metrics,
                "rebalancing_opportunities": rebalancing,
                "allocation_config": self.allocation_config.__dict__,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Portfolio summary generation failed: {e}")
            return {"error": str(e)}
    
    def update_allocation_config(self, **kwargs) -> bool:
        """Update portfolio allocation configuration."""
        try:
            for key, value in kwargs.items():
                if hasattr(self.allocation_config, key):
                    setattr(self.allocation_config, key, value)
                    logger.info(f"Updated allocation config {key} to {value}")
            
            self.save_allocation_settings()
            return True
            
        except Exception as e:
            logger.error(f"Failed to update allocation config: {e}")
            return False


# Global portfolio manager instance
portfolio_manager = PortfolioManager()