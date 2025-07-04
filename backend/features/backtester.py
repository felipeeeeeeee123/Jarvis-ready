"""Backtesting system for JARVIS v3.0 trading strategies."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from utils.logging_config import get_logger
from config.settings import settings
from .strategies import rsi_strategy, ema_strategy, macd_strategy
from .risk_manager import RiskLimits

logger = get_logger(__name__)


@dataclass
class BacktestTrade:
    """Represents a trade in backtesting."""
    symbol: str
    action: str  # buy/sell
    quantity: int
    price: float
    timestamp: datetime
    strategy: str
    signal_strength: float = 0.5
    metadata: Dict = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Comprehensive backtesting results."""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_percent: float
    max_drawdown: float
    max_drawdown_percent: float
    sharpe_ratio: float
    sortino_ratio: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    avg_winning_trade: float
    avg_losing_trade: float
    largest_win: float
    largest_loss: float
    trades: List[BacktestTrade] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)


class BacktestEngine:
    """Backtesting engine for trading strategies."""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.strategies = {
            "RSI": rsi_strategy,
            "EMA": ema_strategy,
            "MACD": macd_strategy
        }
        self.risk_limits = RiskLimits()
        logger.info(f"Backtest engine initialized with ${initial_capital:,.2f} initial capital")
    
    def fetch_historical_data(self, symbol: str, start_date: datetime, 
                            end_date: datetime, timeframe: str = "1Day") -> Optional[pd.DataFrame]:
        """Fetch historical data for backtesting."""
        try:
            # This would integrate with Alpaca API for real historical data
            # For now, create sample data for demonstration
            
            # Generate sample OHLCV data
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Simple random walk for price generation (for demo)
            np.random.seed(42)  # For reproducible results
            returns = np.random.normal(0.001, 0.02, len(date_range))  # Daily returns
            
            # Start with a base price
            base_price = 100.0
            prices = [base_price]
            
            for ret in returns[1:]:
                new_price = prices[-1] * (1 + ret)
                prices.append(new_price)
            
            # Create OHLCV data
            data = []
            for i, (date, price) in enumerate(zip(date_range, prices)):
                # Generate OHLC from close price
                daily_volatility = abs(np.random.normal(0, 0.01))
                high = price * (1 + daily_volatility)
                low = price * (1 - daily_volatility)
                open_price = prices[i-1] if i > 0 else price
                volume = np.random.randint(100000, 1000000)
                
                data.append({
                    'timestamp': date,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': price,
                    'volume': volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Generated sample historical data for {symbol}: {len(df)} days")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            return None
    
    def simulate_trade_execution(self, symbol: str, action: str, quantity: int, 
                               price: float, timestamp: datetime, 
                               cash_balance: float) -> Tuple[bool, float, str]:
        """Simulate trade execution with slippage and fees."""
        try:
            # Apply slippage (0.01% for market orders)
            slippage = 0.0001
            if action == "buy":
                execution_price = price * (1 + slippage)
            else:
                execution_price = price * (1 - slippage)
            
            # Apply commission (simplified)
            commission = 0.0  # Most brokers are commission-free now
            
            trade_value = quantity * execution_price
            
            if action == "buy":
                total_cost = trade_value + commission
                if total_cost > cash_balance:
                    return False, 0, "Insufficient funds"
                return True, total_cost, "Executed"
            else:  # sell
                proceeds = trade_value - commission
                return True, proceeds, "Executed"
                
        except Exception as e:
            return False, 0, f"Execution error: {e}"
    
    def calculate_position_size(self, price: float, cash_balance: float, 
                              portfolio_value: float) -> int:
        """Calculate position size for backtesting."""
        # Use risk management position sizing
        max_position_value = portfolio_value * self.risk_limits.max_position_size
        affordable_shares = int(cash_balance / price)
        risk_based_shares = int(max_position_value / price)
        
        return min(affordable_shares, risk_based_shares)
    
    def run_backtest(self, symbol: str, strategy_name: str, start_date: datetime,
                    end_date: datetime, benchmark_symbol: Optional[str] = None) -> BacktestResult:
        """Run a comprehensive backtest for a strategy."""
        try:
            logger.info(f"Starting backtest: {strategy_name} on {symbol} from {start_date} to {end_date}")
            
            # Fetch historical data
            data = self.fetch_historical_data(symbol, start_date, end_date)
            if data is None or data.empty:
                raise ValueError("No historical data available")
            
            # Get strategy function
            if strategy_name not in self.strategies:
                raise ValueError(f"Unknown strategy: {strategy_name}")
            
            strategy_func = self.strategies[strategy_name]
            
            # Initialize portfolio state
            cash = self.initial_capital
            positions = {}  # symbol -> quantity
            portfolio_values = []
            daily_returns = []
            trades = []
            
            # Backtest loop
            for i in range(20, len(data)):  # Start after enough data for indicators
                current_date = data.index[i]
                current_price = data.iloc[i]['close']
                
                # Get price history for strategy
                price_history = data.iloc[max(0, i-50):i+1]['close']
                
                # Get strategy signal
                signal = strategy_func(price_history)
                
                # Calculate current portfolio value
                position_value = positions.get(symbol, 0) * current_price
                portfolio_value = cash + position_value
                portfolio_values.append(portfolio_value)
                
                # Calculate daily return
                if len(portfolio_values) > 1:
                    daily_return = (portfolio_value - portfolio_values[-2]) / portfolio_values[-2]
                    daily_returns.append(daily_return)
                
                # Execute trades based on signal
                if signal == "buy" and symbol not in positions:
                    # Calculate position size
                    quantity = self.calculate_position_size(current_price, cash, portfolio_value)
                    
                    if quantity > 0:
                        # Execute buy order
                        executed, cost, status = self.simulate_trade_execution(
                            symbol, "buy", quantity, current_price, current_date, cash
                        )
                        
                        if executed:
                            cash -= cost
                            positions[symbol] = quantity
                            
                            trade = BacktestTrade(
                                symbol=symbol,
                                action="buy",
                                quantity=quantity,
                                price=current_price,
                                timestamp=current_date,
                                strategy=strategy_name
                            )
                            trades.append(trade)
                            
                elif signal == "sell" and symbol in positions:
                    quantity = positions[symbol]
                    
                    # Execute sell order
                    executed, proceeds, status = self.simulate_trade_execution(
                        symbol, "sell", quantity, current_price, current_date, cash
                    )
                    
                    if executed:
                        cash += proceeds
                        del positions[symbol]
                        
                        trade = BacktestTrade(
                            symbol=symbol,
                            action="sell",
                            quantity=quantity,
                            price=current_price,
                            timestamp=current_date,
                            strategy=strategy_name
                        )
                        trades.append(trade)
            
            # Calculate final portfolio value
            final_position_value = sum(qty * data.iloc[-1]['close'] for qty in positions.values())
            final_capital = cash + final_position_value
            
            # Calculate performance metrics
            result = self._calculate_performance_metrics(
                start_date, end_date, self.initial_capital, final_capital,
                portfolio_values, daily_returns, trades
            )
            
            logger.info(f"Backtest completed: {result.total_return_percent:.2f}% return, {result.total_trades} trades")
            return result
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
    
    def _calculate_performance_metrics(self, start_date: datetime, end_date: datetime,
                                     initial_capital: float, final_capital: float,
                                     portfolio_values: List[float], daily_returns: List[float],
                                     trades: List[BacktestTrade]) -> BacktestResult:
        """Calculate comprehensive performance metrics."""
        try:
            # Basic returns
            total_return = final_capital - initial_capital
            total_return_percent = (total_return / initial_capital) * 100
            
            # Drawdown calculations
            equity_curve = np.array(portfolio_values)
            running_max = np.maximum.accumulate(equity_curve)
            drawdown = equity_curve - running_max
            drawdown_percent = (drawdown / running_max) * 100
            
            max_drawdown = np.min(drawdown)
            max_drawdown_percent = np.min(drawdown_percent)
            
            # Risk-adjusted returns
            if len(daily_returns) > 1:
                returns_array = np.array(daily_returns)
                mean_return = np.mean(returns_array)
                std_return = np.std(returns_array)
                
                # Sharpe ratio (assuming 2% risk-free rate)
                risk_free_rate = 0.02 / 252  # Daily risk-free rate
                sharpe_ratio = (mean_return - risk_free_rate) / std_return if std_return > 0 else 0
                
                # Sortino ratio (downside deviation)
                downside_returns = returns_array[returns_array < 0]
                downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
                sortino_ratio = (mean_return - risk_free_rate) / downside_std if downside_std > 0 else 0
            else:
                sharpe_ratio = 0
                sortino_ratio = 0
            
            # Trade analysis
            total_trades = len(trades)
            
            if total_trades > 0:
                # Calculate trade P&L (simplified)
                trade_returns = []
                buy_trades = [t for t in trades if t.action == "buy"]
                sell_trades = [t for t in trades if t.action == "sell"]
                
                # Match buy/sell pairs
                for i in range(min(len(buy_trades), len(sell_trades))):
                    buy_trade = buy_trades[i]
                    sell_trade = sell_trades[i]
                    trade_return = (sell_trade.price - buy_trade.price) * buy_trade.quantity
                    trade_returns.append(trade_return)
                
                winning_trades = len([r for r in trade_returns if r > 0])
                losing_trades = len([r for r in trade_returns if r < 0])
                win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
                
                gross_profit = sum(r for r in trade_returns if r > 0)
                gross_loss = abs(sum(r for r in trade_returns if r < 0))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                
                avg_trade_return = np.mean(trade_returns) if trade_returns else 0
                avg_winning_trade = np.mean([r for r in trade_returns if r > 0]) if winning_trades > 0 else 0
                avg_losing_trade = np.mean([r for r in trade_returns if r < 0]) if losing_trades > 0 else 0
                
                largest_win = max(trade_returns) if trade_returns else 0
                largest_loss = min(trade_returns) if trade_returns else 0
            else:
                winning_trades = losing_trades = 0
                win_rate = profit_factor = 0
                avg_trade_return = avg_winning_trade = avg_losing_trade = 0
                largest_win = largest_loss = 0
            
            return BacktestResult(
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                final_capital=final_capital,
                total_return=total_return,
                total_return_percent=total_return_percent,
                max_drawdown=max_drawdown,
                max_drawdown_percent=max_drawdown_percent,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                profit_factor=profit_factor,
                avg_trade_return=avg_trade_return,
                avg_winning_trade=avg_winning_trade,
                avg_losing_trade=avg_losing_trade,
                largest_win=largest_win,
                largest_loss=largest_loss,
                trades=trades,
                daily_returns=daily_returns,
                equity_curve=portfolio_values,
                drawdown_curve=drawdown.tolist()
            )
            
        except Exception as e:
            logger.error(f"Performance metrics calculation failed: {e}")
            raise
    
    def compare_strategies(self, symbol: str, start_date: datetime, end_date: datetime,
                          strategies: Optional[List[str]] = None) -> Dict[str, BacktestResult]:
        """Compare multiple strategies on the same data."""
        try:
            strategies = strategies or list(self.strategies.keys())
            results = {}
            
            logger.info(f"Comparing {len(strategies)} strategies on {symbol}")
            
            for strategy_name in strategies:
                try:
                    result = self.run_backtest(symbol, strategy_name, start_date, end_date)
                    results[strategy_name] = result
                except Exception as e:
                    logger.error(f"Strategy {strategy_name} backtest failed: {e}")
            
            # Log comparison summary
            if results:
                best_strategy = max(results.keys(), key=lambda k: results[k].total_return_percent)
                logger.info(f"Best performing strategy: {best_strategy} ({results[best_strategy].total_return_percent:.2f}%)")
            
            return results
            
        except Exception as e:
            logger.error(f"Strategy comparison failed: {e}")
            return {}
    
    def optimize_strategy_parameters(self, symbol: str, strategy_name: str,
                                   start_date: datetime, end_date: datetime,
                                   parameter_ranges: Dict[str, List]) -> Dict[str, Any]:
        """Optimize strategy parameters through backtesting."""
        # This would implement parameter optimization
        # For now, return a placeholder
        logger.info("Parameter optimization not yet implemented")
        return {
            "best_parameters": {},
            "best_return": 0.0,
            "optimization_results": []
        }
    
    def generate_backtest_report(self, result: BacktestResult) -> str:
        """Generate a formatted backtest report."""
        try:
            report = f"""
JARVIS v3.0 Backtest Report
==========================

Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}
Initial Capital: ${result.initial_capital:,.2f}
Final Capital: ${result.final_capital:,.2f}

Performance Metrics:
-------------------
Total Return: ${result.total_return:,.2f} ({result.total_return_percent:.2f}%)
Max Drawdown: ${result.max_drawdown:,.2f} ({result.max_drawdown_percent:.2f}%)
Sharpe Ratio: {result.sharpe_ratio:.3f}
Sortino Ratio: {result.sortino_ratio:.3f}

Trade Statistics:
----------------
Total Trades: {result.total_trades}
Winning Trades: {result.winning_trades}
Losing Trades: {result.losing_trades}
Win Rate: {result.win_rate:.1f}%
Profit Factor: {result.profit_factor:.2f}

Trade Analysis:
--------------
Average Trade Return: ${result.avg_trade_return:.2f}
Average Winning Trade: ${result.avg_winning_trade:.2f}
Average Losing Trade: ${result.avg_losing_trade:.2f}
Largest Win: ${result.largest_win:.2f}
Largest Loss: ${result.largest_loss:.2f}

Risk Assessment:
---------------
Maximum Drawdown: {result.max_drawdown_percent:.2f}%
Risk-Adjusted Return (Sharpe): {result.sharpe_ratio:.3f}
Downside Risk (Sortino): {result.sortino_ratio:.3f}
"""
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return f"Error generating report: {e}"


# Global backtester instance
backtester = BacktestEngine()