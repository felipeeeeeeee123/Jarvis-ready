"""Performance analytics and optimization for JARVIS v3.0 trading system."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict

from database.services import trade_service, config_service
from utils.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    # Returns
    total_return: float
    annualized_return: float
    daily_return_mean: float
    daily_return_std: float
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    value_at_risk_95: float
    
    # Trade metrics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_duration: float
    
    # Advanced metrics
    calmar_ratio: float
    information_ratio: float
    tail_ratio: float
    
    # Time period
    start_date: datetime
    end_date: datetime
    analysis_period_days: int


class PerformanceAnalyzer:
    """Advanced performance analytics and optimization engine."""
    
    def __init__(self):
        self.benchmark_return = 0.08  # 8% annual benchmark
        self.risk_free_rate = 0.02   # 2% annual risk-free rate
        logger.info("Performance analyzer initialized")
    
    def calculate_comprehensive_metrics(self, start_date: Optional[datetime] = None,
                                      end_date: Optional[datetime] = None) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics for the trading system."""
        try:
            # Default to last 90 days if no dates provided
            if not end_date:
                end_date = datetime.utcnow()
            if not start_date:
                start_date = end_date - timedelta(days=90)
            
            # Get trade data
            trades = trade_service.get_recent_trades(limit=1000)
            period_trades = [
                t for t in trades 
                if start_date <= t.executed_at <= end_date and t.status == "filled"
            ]
            
            if not period_trades:
                logger.warning("No trades found for performance analysis")
                return self._empty_metrics(start_date, end_date)
            
            # Calculate daily returns
            daily_returns = self._calculate_daily_returns(period_trades, start_date, end_date)
            
            if not daily_returns:
                return self._empty_metrics(start_date, end_date)
            
            returns_array = np.array(daily_returns)
            
            # Basic return metrics
            total_return = np.sum(returns_array)
            annualized_return = self._annualize_return(np.mean(returns_array))
            daily_return_mean = np.mean(returns_array)
            daily_return_std = np.std(returns_array)
            
            # Risk-adjusted metrics
            sharpe_ratio = self._calculate_sharpe_ratio(returns_array)
            sortino_ratio = self._calculate_sortino_ratio(returns_array)
            max_drawdown = self._calculate_max_drawdown(returns_array)
            var_95 = self._calculate_value_at_risk(returns_array, 0.05)
            
            # Trade metrics
            trade_metrics = self._analyze_trades(period_trades)
            
            # Advanced metrics
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            information_ratio = self._calculate_information_ratio(returns_array)
            tail_ratio = self._calculate_tail_ratio(returns_array)
            
            analysis_period_days = (end_date - start_date).days
            
            metrics = PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                daily_return_mean=daily_return_mean,
                daily_return_std=daily_return_std,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                value_at_risk_95=var_95,
                total_trades=trade_metrics["total_trades"],
                win_rate=trade_metrics["win_rate"],
                profit_factor=trade_metrics["profit_factor"],
                avg_trade_duration=trade_metrics["avg_duration"],
                calmar_ratio=calmar_ratio,
                information_ratio=information_ratio,
                tail_ratio=tail_ratio,
                start_date=start_date,
                end_date=end_date,
                analysis_period_days=analysis_period_days
            )
            
            logger.info(f"Performance metrics calculated", extra={
                "period_days": analysis_period_days,
                "total_trades": metrics.total_trades,
                "total_return": metrics.total_return,
                "sharpe_ratio": metrics.sharpe_ratio
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Performance metrics calculation failed: {e}")
            return self._empty_metrics(start_date or datetime.utcnow(), end_date or datetime.utcnow())
    
    def _empty_metrics(self, start_date: datetime, end_date: datetime) -> PerformanceMetrics:
        """Return empty metrics for error cases."""
        return PerformanceMetrics(
            total_return=0.0, annualized_return=0.0, daily_return_mean=0.0, daily_return_std=0.0,
            sharpe_ratio=0.0, sortino_ratio=0.0, max_drawdown=0.0, value_at_risk_95=0.0,
            total_trades=0, win_rate=0.0, profit_factor=0.0, avg_trade_duration=0.0,
            calmar_ratio=0.0, information_ratio=0.0, tail_ratio=0.0,
            start_date=start_date, end_date=end_date, analysis_period_days=0
        )
    
    def _calculate_daily_returns(self, trades: List, start_date: datetime, end_date: datetime) -> List[float]:
        """Calculate daily returns from trade data."""
        try:
            # Group trades by date
            daily_pnl = defaultdict(float)
            
            # Simple P&L calculation (this would be more sophisticated in practice)
            for trade in trades:
                trade_date = trade.executed_at.date()
                if trade.action == "sell":
                    # Simplified: assume all sells are profitable
                    daily_pnl[trade_date] += trade.total_value * 0.02  # 2% profit assumption
                elif trade.action == "buy":
                    daily_pnl[trade_date] -= trade.total_value * 0.001  # Small cost for buying
            
            # Convert to daily returns list
            current_date = start_date.date()
            end_date_only = end_date.date()
            daily_returns = []
            
            while current_date <= end_date_only:
                daily_return = daily_pnl.get(current_date, 0.0)
                daily_returns.append(daily_return)
                current_date += timedelta(days=1)
            
            return daily_returns
            
        except Exception as e:
            logger.error(f"Daily returns calculation failed: {e}")
            return []
    
    def _annualize_return(self, daily_return: float) -> float:
        """Convert daily return to annualized return."""
        return ((1 + daily_return) ** 252) - 1  # 252 trading days per year
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        try:
            excess_returns = returns - (self.risk_free_rate / 252)  # Daily risk-free rate
            return np.mean(excess_returns) / np.std(returns) if np.std(returns) > 0 else 0
        except Exception:
            return 0.0
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        try:
            excess_returns = returns - (self.risk_free_rate / 252)
            downside_returns = returns[returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else np.std(returns)
            return np.mean(excess_returns) / downside_std if downside_std > 0 else 0
        except Exception:
            return 0.0
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        try:
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = cumulative_returns - running_max
            return np.min(drawdown)
        except Exception:
            return 0.0
    
    def _calculate_value_at_risk(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate Value at Risk at given confidence level."""
        try:
            return np.percentile(returns, confidence_level * 100)
        except Exception:
            return 0.0
    
    def _calculate_information_ratio(self, returns: np.ndarray) -> float:
        """Calculate Information Ratio vs benchmark."""
        try:
            benchmark_daily_return = self.benchmark_return / 252
            excess_returns = returns - benchmark_daily_return
            tracking_error = np.std(excess_returns)
            return np.mean(excess_returns) / tracking_error if tracking_error > 0 else 0
        except Exception:
            return 0.0
    
    def _calculate_tail_ratio(self, returns: np.ndarray) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)."""
        try:
            p95 = np.percentile(returns, 95)
            p5 = np.percentile(returns, 5)
            return abs(p95 / p5) if p5 != 0 else 0
        except Exception:
            return 0.0
    
    def _analyze_trades(self, trades: List) -> Dict[str, Any]:
        """Analyze individual trades for metrics."""
        try:
            if not trades:
                return {"total_trades": 0, "win_rate": 0, "profit_factor": 0, "avg_duration": 0}
            
            # Group buy/sell pairs (simplified)
            trade_pairs = []
            buy_trades = [t for t in trades if t.action == "buy"]
            sell_trades = [t for t in trades if t.action == "sell"]
            
            # Match trades by symbol and time proximity
            for buy_trade in buy_trades:
                # Find corresponding sell trade
                matching_sells = [
                    s for s in sell_trades 
                    if s.symbol == buy_trade.symbol and s.executed_at > buy_trade.executed_at
                ]
                if matching_sells:
                    sell_trade = min(matching_sells, key=lambda x: x.executed_at)
                    pnl = (sell_trade.price - buy_trade.price) * buy_trade.quantity
                    duration = (sell_trade.executed_at - buy_trade.executed_at).total_seconds() / 3600  # hours
                    trade_pairs.append({"pnl": pnl, "duration": duration})
            
            if not trade_pairs:
                return {"total_trades": len(trades), "win_rate": 0, "profit_factor": 0, "avg_duration": 0}
            
            # Calculate metrics
            winning_trades = len([t for t in trade_pairs if t["pnl"] > 0])
            losing_trades = len([t for t in trade_pairs if t["pnl"] < 0])
            win_rate = (winning_trades / len(trade_pairs)) * 100 if trade_pairs else 0
            
            gross_profit = sum(t["pnl"] for t in trade_pairs if t["pnl"] > 0)
            gross_loss = abs(sum(t["pnl"] for t in trade_pairs if t["pnl"] < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            avg_duration = np.mean([t["duration"] for t in trade_pairs])
            
            return {
                "total_trades": len(trade_pairs),
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "avg_duration": avg_duration
            }
            
        except Exception as e:
            logger.error(f"Trade analysis failed: {e}")
            return {"total_trades": 0, "win_rate": 0, "profit_factor": 0, "avg_duration": 0}
    
    def generate_performance_report(self, metrics: PerformanceMetrics) -> str:
        """Generate a comprehensive performance report."""
        try:
            report = f"""
JARVIS v3.0 Performance Analysis Report
=====================================

Analysis Period: {metrics.start_date.strftime('%Y-%m-%d')} to {metrics.end_date.strftime('%Y-%m-%d')}
Period Duration: {metrics.analysis_period_days} days

RETURN METRICS
--------------
Total Return: {metrics.total_return:.2%}
Annualized Return: {metrics.annualized_return:.2%}
Daily Return (Mean): {metrics.daily_return_mean:.4%}
Daily Return (Std): {metrics.daily_return_std:.4%}

RISK METRICS
------------
Sharpe Ratio: {metrics.sharpe_ratio:.3f}
Sortino Ratio: {metrics.sortino_ratio:.3f}
Maximum Drawdown: {metrics.max_drawdown:.2%}
Value at Risk (95%): {metrics.value_at_risk_95:.2%}
Calmar Ratio: {metrics.calmar_ratio:.3f}

TRADE ANALYSIS
--------------
Total Trades: {metrics.total_trades}
Win Rate: {metrics.win_rate:.1f}%
Profit Factor: {metrics.profit_factor:.2f}
Avg Trade Duration: {metrics.avg_trade_duration:.1f} hours

ADVANCED METRICS
----------------
Information Ratio: {metrics.information_ratio:.3f}
Tail Ratio: {metrics.tail_ratio:.3f}

PERFORMANCE ASSESSMENT
---------------------
"""
            
            # Add performance assessment
            if metrics.sharpe_ratio > 1.5:
                report += "🟢 EXCELLENT: Strong risk-adjusted returns\n"
            elif metrics.sharpe_ratio > 1.0:
                report += "🟡 GOOD: Solid risk-adjusted performance\n"
            elif metrics.sharpe_ratio > 0.5:
                report += "🟠 FAIR: Moderate performance\n"
            else:
                report += "🔴 POOR: Below-average risk-adjusted returns\n"
            
            if metrics.max_drawdown > -0.20:
                report += "🔴 HIGH RISK: Significant drawdown exposure\n"
            elif metrics.max_drawdown > -0.10:
                report += "🟠 MODERATE RISK: Acceptable drawdown levels\n"
            else:
                report += "🟢 LOW RISK: Well-controlled drawdowns\n"
            
            if metrics.win_rate > 60:
                report += "🟢 HIGH ACCURACY: Strong trade selection\n"
            elif metrics.win_rate > 45:
                report += "🟡 MODERATE ACCURACY: Balanced approach\n"
            else:
                report += "🟠 LOW ACCURACY: Consider strategy refinement\n"
            
            return report
            
        except Exception as e:
            logger.error(f"Performance report generation failed: {e}")
            return f"Error generating performance report: {e}"
    
    def identify_optimization_opportunities(self, metrics: PerformanceMetrics) -> List[Dict[str, str]]:
        """Identify specific optimization opportunities."""
        opportunities = []
        
        try:
            # Risk management opportunities
            if metrics.max_drawdown < -0.15:
                opportunities.append({
                    "category": "Risk Management",
                    "issue": "High maximum drawdown",
                    "recommendation": "Implement tighter stop-loss levels or reduce position sizes",
                    "priority": "High"
                })
            
            if metrics.sharpe_ratio < 0.5:
                opportunities.append({
                    "category": "Risk-Adjusted Returns",
                    "issue": "Low Sharpe ratio",
                    "recommendation": "Consider strategy diversification or improved entry/exit timing",
                    "priority": "High"
                })
            
            # Trade execution opportunities
            if metrics.win_rate < 40:
                opportunities.append({
                    "category": "Trade Selection",
                    "issue": "Low win rate",
                    "recommendation": "Refine entry criteria or add confirmation indicators",
                    "priority": "Medium"
                })
            
            if metrics.profit_factor < 1.2:
                opportunities.append({
                    "category": "Profit Optimization",
                    "issue": "Low profit factor",
                    "recommendation": "Optimize profit-taking levels or reduce transaction costs",
                    "priority": "Medium"
                })
            
            # Advanced opportunities
            if metrics.information_ratio < 0.5:
                opportunities.append({
                    "category": "Benchmark Performance",
                    "issue": "Underperforming benchmark",
                    "recommendation": "Consider index-based allocation or strategy diversification",
                    "priority": "Low"
                })
            
            if metrics.tail_ratio < 1.5:
                opportunities.append({
                    "category": "Return Distribution",
                    "issue": "Poor upside/downside ratio",
                    "recommendation": "Implement asymmetric risk management strategies",
                    "priority": "Low"
                })
            
            logger.info(f"Identified {len(opportunities)} optimization opportunities")
            return opportunities
            
        except Exception as e:
            logger.error(f"Optimization analysis failed: {e}")
            return []
    
    def get_strategy_comparison(self, days: int = 30) -> Dict[str, Any]:
        """Compare performance of different strategies."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            trades = trade_service.get_recent_trades(limit=1000)
            recent_trades = [t for t in trades if t.executed_at >= cutoff_date and t.status == "filled"]
            
            # Group by strategy
            strategy_performance = defaultdict(lambda: {"trades": [], "pnl": 0.0})
            
            for trade in recent_trades:
                strategy = trade.strategy_used
                strategy_performance[strategy]["trades"].append(trade)
                
                # Simplified P&L calculation
                if trade.action == "sell":
                    strategy_performance[strategy]["pnl"] += trade.total_value * 0.02
                elif trade.action == "buy":
                    strategy_performance[strategy]["pnl"] -= trade.total_value * 0.001
            
            # Calculate metrics for each strategy
            comparison = {}
            for strategy, data in strategy_performance.items():
                trades_count = len(data["trades"])
                total_pnl = data["pnl"]
                
                if trades_count > 0:
                    comparison[strategy] = {
                        "total_trades": trades_count,
                        "total_pnl": total_pnl,
                        "avg_pnl_per_trade": total_pnl / trades_count,
                        "trade_frequency": trades_count / days
                    }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Strategy comparison failed: {e}")
            return {}


# Global performance analyzer instance
performance_analyzer = PerformanceAnalyzer()