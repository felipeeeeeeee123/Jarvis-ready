"""Dynamic strategy management system for JARVIS v3.0."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum

from database.services import trade_service, config_service, memory_service
from utils.logging_config import get_logger
from config.settings import settings
from .strategies import rsi_strategy, ema_strategy, macd_strategy

logger = get_logger(__name__)


class MarketCondition(Enum):
    """Market condition classifications."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    LOW_VOLUME = "low_volume"


class StrategyPerformance:
    """Track performance metrics for trading strategies."""
    
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.max_drawdown = 0.0
        self.avg_trade_duration = 0.0
        self.sharpe_ratio = 0.0
        self.last_updated = datetime.utcnow()
    
    def update_from_trades(self, trades: List):
        """Update performance metrics from trade history."""
        if not trades:
            return
        
        self.total_trades = len(trades)
        profits = []
        
        for trade in trades:
            if trade.action == "sell":  # Assuming sells close positions
                # Calculate P&L (simplified)
                pnl = trade.total_value  # This would need proper P&L calculation
                if pnl > 0:
                    self.winning_trades += 1
                    self.total_profit += pnl
                else:
                    self.losing_trades += 1
                    self.total_loss += abs(pnl)
                profits.append(pnl)
        
        # Calculate additional metrics
        if profits:
            profits_array = np.array(profits)
            self.sharpe_ratio = np.mean(profits_array) / np.std(profits_array) if np.std(profits_array) > 0 else 0
            
            # Calculate max drawdown
            cumulative = np.cumsum(profits_array)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            self.max_drawdown = np.min(drawdown)
        
        self.last_updated = datetime.utcnow()
    
    def win_rate(self) -> float:
        """Calculate win rate percentage."""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100
    
    def profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if self.total_loss == 0:
            return float('inf') if self.total_profit > 0 else 1.0
        return self.total_profit / self.total_loss
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "strategy_name": self.strategy_name,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate(),
            "total_profit": self.total_profit,
            "total_loss": self.total_loss,
            "profit_factor": self.profit_factor(),
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "last_updated": self.last_updated.isoformat()
        }


class StrategyManager:
    """Manages dynamic strategy switching based on market conditions and performance."""
    
    def __init__(self):
        self.strategies = {
            "RSI": rsi_strategy,
            "EMA": ema_strategy,
            "MACD": macd_strategy
        }
        
        self.strategy_conditions = {
            "RSI": [MarketCondition.SIDEWAYS, MarketCondition.VOLATILE],
            "EMA": [MarketCondition.TRENDING_UP, MarketCondition.TRENDING_DOWN],
            "MACD": [MarketCondition.TRENDING_UP, MarketCondition.TRENDING_DOWN, MarketCondition.VOLATILE]
        }
        
        self.performance_tracker = {
            name: StrategyPerformance(name) for name in self.strategies.keys()
        }
        
        self.current_strategy = settings.STRATEGY
        self.strategy_switch_threshold = 0.7  # Switch if performance drops below 70%
        self.min_trades_for_switch = 10  # Minimum trades before considering switch
        
        self.load_performance_data()
        logger.info(f"Strategy manager initialized with {len(self.strategies)} strategies")
    
    def load_performance_data(self):
        """Load historical performance data for all strategies."""
        try:
            for strategy_name in self.strategies.keys():
                # Get trades for this strategy from the last 30 days
                cutoff_date = datetime.utcnow() - timedelta(days=30)
                trades = trade_service.get_recent_trades(limit=1000)
                
                # Filter trades by strategy
                strategy_trades = [
                    trade for trade in trades 
                    if trade.strategy_used == strategy_name and trade.executed_at >= cutoff_date
                ]
                
                self.performance_tracker[strategy_name].update_from_trades(strategy_trades)
                
            logger.info("Strategy performance data loaded")
        except Exception as e:
            logger.error(f"Failed to load strategy performance data: {e}")
    
    def analyze_market_condition(self, prices: pd.Series, volume: Optional[pd.Series] = None) -> MarketCondition:
        """Analyze current market conditions to determine best strategy."""
        try:
            if len(prices) < 20:
                return MarketCondition.SIDEWAYS  # Default for insufficient data
            
            # Calculate trend indicators
            sma_20 = prices.rolling(20).mean()
            sma_50 = prices.rolling(50).mean() if len(prices) >= 50 else sma_20
            
            # Current price relative to moving averages
            current_price = prices.iloc[-1]
            current_sma20 = sma_20.iloc[-1]
            current_sma50 = sma_50.iloc[-1] if len(sma_50) > 0 else current_sma20
            
            # Price volatility (using standard deviation)
            volatility = prices.pct_change().rolling(20).std().iloc[-1]
            avg_volatility = prices.pct_change().std()
            
            # Trend strength
            trend_up = current_price > current_sma20 > current_sma50
            trend_down = current_price < current_sma20 < current_sma50
            
            # Volume analysis (if available)
            low_volume = False
            if volume is not None and len(volume) >= 20:
                avg_volume = volume.rolling(20).mean().iloc[-1]
                recent_volume = volume.iloc[-5:].mean()
                low_volume = recent_volume < avg_volume * 0.7
            
            # Determine market condition
            if low_volume:
                return MarketCondition.LOW_VOLUME
            elif volatility > avg_volatility * 1.5:
                return MarketCondition.VOLATILE
            elif trend_up:
                return MarketCondition.TRENDING_UP
            elif trend_down:
                return MarketCondition.TRENDING_DOWN
            else:
                return MarketCondition.SIDEWAYS
                
        except Exception as e:
            logger.error(f"Market condition analysis failed: {e}")
            return MarketCondition.SIDEWAYS
    
    def select_best_strategy(self, market_condition: MarketCondition, 
                           performance_weight: float = 0.6) -> str:
        """Select the best strategy based on market condition and performance."""
        try:
            # Get strategies suitable for current market condition
            suitable_strategies = []
            for strategy_name, conditions in self.strategy_conditions.items():
                if market_condition in conditions:
                    suitable_strategies.append(strategy_name)
            
            if not suitable_strategies:
                suitable_strategies = list(self.strategies.keys())  # Fall back to all strategies
            
            # Score strategies based on performance and suitability
            strategy_scores = {}
            
            for strategy_name in suitable_strategies:
                performance = self.performance_tracker[strategy_name]
                
                # Base score from market condition suitability
                condition_score = 1.0 if market_condition in self.strategy_conditions[strategy_name] else 0.5
                
                # Performance score
                if performance.total_trades >= self.min_trades_for_switch:
                    win_rate = performance.win_rate() / 100.0
                    profit_factor = min(performance.profit_factor(), 3.0) / 3.0  # Cap at 3.0
                    sharpe_ratio = max(0, min(performance.sharpe_ratio + 1, 2.0)) / 2.0  # Normalize
                    
                    performance_score = (win_rate * 0.4 + profit_factor * 0.4 + sharpe_ratio * 0.2)
                else:
                    performance_score = 0.5  # Neutral score for insufficient data
                
                # Combined score
                strategy_scores[strategy_name] = (
                    condition_score * (1 - performance_weight) + 
                    performance_score * performance_weight
                )
            
            # Select strategy with highest score
            best_strategy = max(strategy_scores, key=strategy_scores.get)
            
            logger.info(f"Strategy selection", extra={
                "market_condition": market_condition.value,
                "strategy_scores": strategy_scores,
                "selected_strategy": best_strategy
            })
            
            return best_strategy
            
        except Exception as e:
            logger.error(f"Strategy selection failed: {e}")
            return self.current_strategy
    
    def should_switch_strategy(self, symbol: str, prices: pd.Series, 
                              volume: Optional[pd.Series] = None) -> Tuple[bool, str, str]:
        """Determine if strategy should be switched for given market data."""
        try:
            # Analyze market conditions
            market_condition = self.analyze_market_condition(prices, volume)
            
            # Get recommended strategy
            recommended_strategy = self.select_best_strategy(market_condition)
            
            # Check if switch is warranted
            should_switch = False
            reason = ""
            
            # Switch if recommended strategy is different and performs significantly better
            if recommended_strategy != self.current_strategy:
                current_perf = self.performance_tracker[self.current_strategy]
                recommended_perf = self.performance_tracker[recommended_strategy]
                
                # Check if we have enough data to make a decision
                if (current_perf.total_trades >= self.min_trades_for_switch or
                    current_perf.win_rate() < 40.0):  # Switch if performing very poorly
                    
                    should_switch = True
                    reason = f"Market condition: {market_condition.value}, Performance improvement expected"
            
            return should_switch, recommended_strategy, reason
            
        except Exception as e:
            logger.error(f"Strategy switch evaluation failed: {e}")
            return False, self.current_strategy, f"Error: {e}"
    
    def execute_strategy_switch(self, new_strategy: str, reason: str = "") -> bool:
        """Execute a strategy switch."""
        try:
            old_strategy = self.current_strategy
            self.current_strategy = new_strategy
            
            # Update configuration
            config_service.set_config("trading.current_strategy", new_strategy, 
                                    f"Current trading strategy (switched from {old_strategy})")
            
            # Log the switch
            logger.info(f"Strategy switched", extra={
                "old_strategy": old_strategy,
                "new_strategy": new_strategy,
                "reason": reason
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Strategy switch execution failed: {e}")
            return False
    
    def get_strategy_signal(self, symbol: str, prices: pd.Series, 
                           volume: Optional[pd.Series] = None) -> Tuple[str, str, Dict]:
        """Get trading signal using the current strategy with dynamic switching."""
        try:
            # Check if strategy should be switched
            should_switch, recommended_strategy, switch_reason = self.should_switch_strategy(
                symbol, prices, volume
            )
            
            if should_switch:
                self.execute_strategy_switch(recommended_strategy, switch_reason)
            
            # Get signal from current strategy
            strategy_func = self.strategies[self.current_strategy]
            signal = strategy_func(prices)
            
            # Additional metadata
            metadata = {
                "strategy_used": self.current_strategy,
                "market_condition": self.analyze_market_condition(prices, volume).value,
                "strategy_switched": should_switch,
                "switch_reason": switch_reason if should_switch else None,
                "signal_strength": self._calculate_signal_strength(signal, prices)
            }
            
            return signal, self.current_strategy, metadata
            
        except Exception as e:
            logger.error(f"Strategy signal generation failed: {e}")
            return "hold", self.current_strategy, {"error": str(e)}
    
    def _calculate_signal_strength(self, signal: str, prices: pd.Series) -> float:
        """Calculate signal strength/confidence (0.0 to 1.0)."""
        try:
            if signal == "hold":
                return 0.5
            
            # Simple signal strength based on price momentum
            if len(prices) >= 5:
                recent_change = (prices.iloc[-1] - prices.iloc[-5]) / prices.iloc[-5]
                momentum = abs(recent_change)
                
                # Normalize to 0.5-1.0 range
                strength = 0.5 + min(momentum * 10, 0.5)
                return strength
            
            return 0.6  # Default medium confidence
            
        except Exception as e:
            logger.warning(f"Signal strength calculation failed: {e}")
            return 0.5
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary for all strategies."""
        summary = {
            "current_strategy": self.current_strategy,
            "last_updated": datetime.utcnow().isoformat(),
            "strategies": {}
        }
        
        for name, performance in self.performance_tracker.items():
            summary["strategies"][name] = performance.to_dict()
        
        return summary
    
    def update_strategy_performance(self, strategy_name: str, symbol: str):
        """Update performance tracking for a specific strategy."""
        try:
            if strategy_name in self.performance_tracker:
                # Get recent trades for this strategy
                recent_trades = trade_service.get_recent_trades(symbol, limit=50)
                strategy_trades = [t for t in recent_trades if t.strategy_used == strategy_name]
                
                self.performance_tracker[strategy_name].update_from_trades(strategy_trades)
                
        except Exception as e:
            logger.error(f"Strategy performance update failed: {e}")


# Global strategy manager instance
strategy_manager = StrategyManager()