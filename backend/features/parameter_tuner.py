"""Performance-based parameter tuning system for JARVIS v3.0."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

from database.services import config_service, trade_service, memory_service
from utils.logging_config import get_logger
from config.settings import settings
from .backtester import BacktestEngine, BacktestResult
from .performance_analyzer import PerformanceAnalyzer

logger = get_logger(__name__)


class OptimizationType(Enum):
    """Types of parameter optimization."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"


@dataclass
class ParameterRange:
    """Defines a parameter range for optimization."""
    name: str
    min_value: float
    max_value: float
    step: float = 0.1
    param_type: str = "float"  # float, int, bool
    current_value: float = 0.0


@dataclass
class OptimizationResult:
    """Results from parameter optimization."""
    parameter_set: Dict[str, Any]
    performance_score: float
    metrics: Dict[str, float]
    backtest_result: Optional[BacktestResult] = None
    execution_time: float = 0.0


@dataclass
class TuningSession:
    """Represents a parameter tuning session."""
    id: str
    strategy_name: str
    optimization_type: OptimizationType
    parameter_ranges: List[ParameterRange]
    started_at: datetime
    completed_at: Optional[datetime] = None
    best_result: Optional[OptimizationResult] = None
    all_results: List[OptimizationResult] = field(default_factory=list)
    status: str = "running"
    error_message: Optional[str] = None


class ParameterOptimizer:
    """Handles different parameter optimization algorithms."""
    
    def __init__(self):
        self.backtester = BacktestEngine()
        self.performance_analyzer = PerformanceAnalyzer()
        
    def grid_search(self, parameter_ranges: List[ParameterRange], 
                   evaluation_function: Callable, max_iterations: int = 100) -> List[OptimizationResult]:
        """Perform grid search optimization."""
        try:
            logger.info("Starting grid search optimization")
            
            # Generate all parameter combinations
            param_combinations = self._generate_grid_combinations(parameter_ranges)
            
            # Limit combinations if too many
            if len(param_combinations) > max_iterations:
                param_combinations = param_combinations[:max_iterations]
                logger.info(f"Limited grid search to {max_iterations} combinations")
            
            results = []
            for i, param_set in enumerate(param_combinations):
                try:
                    logger.info(f"Evaluating parameter set {i+1}/{len(param_combinations)}")
                    result = evaluation_function(param_set)
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error evaluating parameter set {i+1}: {e}")
                    continue
            
            # Sort by performance score
            results.sort(key=lambda x: x.performance_score, reverse=True)
            
            logger.info(f"Grid search completed with {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Grid search failed: {e}")
            return []
    
    def random_search(self, parameter_ranges: List[ParameterRange], 
                     evaluation_function: Callable, max_iterations: int = 50) -> List[OptimizationResult]:
        """Perform random search optimization."""
        try:
            logger.info("Starting random search optimization")
            
            results = []
            for i in range(max_iterations):
                try:
                    # Generate random parameter set
                    param_set = self._generate_random_parameters(parameter_ranges)
                    
                    logger.info(f"Evaluating random parameter set {i+1}/{max_iterations}")
                    result = evaluation_function(param_set)
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error evaluating random parameter set {i+1}: {e}")
                    continue
            
            # Sort by performance score
            results.sort(key=lambda x: x.performance_score, reverse=True)
            
            logger.info(f"Random search completed with {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Random search failed: {e}")
            return []
    
    def bayesian_optimization(self, parameter_ranges: List[ParameterRange], 
                            evaluation_function: Callable, max_iterations: int = 30) -> List[OptimizationResult]:
        """Perform Bayesian optimization (simplified version)."""
        try:
            logger.info("Starting Bayesian optimization")
            
            # For now, implement a simplified version using random search with exploration/exploitation
            results = []
            best_score = float('-inf')
            exploration_factor = 0.3
            
            for i in range(max_iterations):
                try:
                    if i < 5 or np.random.random() < exploration_factor:
                        # Exploration: random parameters
                        param_set = self._generate_random_parameters(parameter_ranges)
                    else:
                        # Exploitation: parameters around best known result
                        if results:
                            best_params = max(results, key=lambda x: x.performance_score).parameter_set
                            param_set = self._generate_parameters_around_best(parameter_ranges, best_params)
                        else:
                            param_set = self._generate_random_parameters(parameter_ranges)
                    
                    logger.info(f"Evaluating Bayesian parameter set {i+1}/{max_iterations}")
                    result = evaluation_function(param_set)
                    results.append(result)
                    
                    if result.performance_score > best_score:
                        best_score = result.performance_score
                        logger.info(f"New best score: {best_score:.4f}")
                    
                    # Reduce exploration over time
                    exploration_factor *= 0.95
                    
                except Exception as e:
                    logger.error(f"Error evaluating Bayesian parameter set {i+1}: {e}")
                    continue
            
            # Sort by performance score
            results.sort(key=lambda x: x.performance_score, reverse=True)
            
            logger.info(f"Bayesian optimization completed with {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Bayesian optimization failed: {e}")
            return []
    
    def _generate_grid_combinations(self, parameter_ranges: List[ParameterRange]) -> List[Dict[str, Any]]:
        """Generate all combinations for grid search."""
        param_values = {}
        
        for param in parameter_ranges:
            if param.param_type == "float":
                values = np.arange(param.min_value, param.max_value + param.step, param.step)
            elif param.param_type == "int":
                values = np.arange(int(param.min_value), int(param.max_value) + 1, max(1, int(param.step)))
            elif param.param_type == "bool":
                values = [True, False]
            else:
                values = [param.current_value]
            
            param_values[param.name] = values
        
        # Generate all combinations
        combinations = []
        param_names = list(param_values.keys())
        
        for combination in itertools.product(*param_values.values()):
            param_set = dict(zip(param_names, combination))
            combinations.append(param_set)
        
        return combinations
    
    def _generate_random_parameters(self, parameter_ranges: List[ParameterRange]) -> Dict[str, Any]:
        """Generate random parameter values."""
        param_set = {}
        
        for param in parameter_ranges:
            if param.param_type == "float":
                value = np.random.uniform(param.min_value, param.max_value)
            elif param.param_type == "int":
                value = np.random.randint(int(param.min_value), int(param.max_value) + 1)
            elif param.param_type == "bool":
                value = np.random.choice([True, False])
            else:
                value = param.current_value
            
            param_set[param.name] = value
        
        return param_set
    
    def _generate_parameters_around_best(self, parameter_ranges: List[ParameterRange], 
                                       best_params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate parameters around the best known parameters."""
        param_set = {}
        
        for param in parameter_ranges:
            if param.name in best_params:
                best_value = best_params[param.name]
                
                if param.param_type == "float":
                    # Add noise around best value
                    noise_range = (param.max_value - param.min_value) * 0.1
                    value = best_value + np.random.normal(0, noise_range)
                    value = np.clip(value, param.min_value, param.max_value)
                elif param.param_type == "int":
                    # Add small integer noise
                    noise = np.random.randint(-2, 3)
                    value = int(best_value) + noise
                    value = np.clip(value, int(param.min_value), int(param.max_value))
                elif param.param_type == "bool":
                    # Mostly keep best value, sometimes flip
                    value = best_value if np.random.random() < 0.8 else not best_value
                else:
                    value = best_value
                
                param_set[param.name] = value
            else:
                # If parameter not in best params, use random value
                param_set[param.name] = self._generate_random_parameters([param])[param.name]
        
        return param_set


class ParameterTuner:
    """Main parameter tuning system."""
    
    def __init__(self):
        self.optimizer = ParameterOptimizer()
        self.active_sessions: Dict[str, TuningSession] = {}
        self.tuning_history: List[TuningSession] = []
        self.auto_tuning_enabled = True
        self.tuning_thread = None
        self.running = False
        
        # Configuration
        self.auto_tune_interval = 86400  # Daily auto-tuning
        self.performance_threshold = 0.6  # Tune if performance drops below 60%
        self.min_trades_for_tuning = 100  # Minimum trades before tuning
        
        self.load_configuration()
        logger.info("Parameter tuner initialized")
    
    def load_configuration(self):
        """Load configuration from database."""
        try:
            self.auto_tuning_enabled = config_service.get_config("tuning.auto_enabled", True)
            self.auto_tune_interval = config_service.get_config("tuning.auto_interval", 86400)
            self.performance_threshold = config_service.get_config("tuning.performance_threshold", 0.6)
            self.min_trades_for_tuning = config_service.get_config("tuning.min_trades", 100)
            
            logger.info("Parameter tuner configuration loaded")
        except Exception as e:
            logger.warning(f"Failed to load parameter tuner configuration: {e}")
    
    def save_configuration(self):
        """Save configuration to database."""
        try:
            config_service.set_config("tuning.auto_enabled", self.auto_tuning_enabled, "Enable automatic parameter tuning")
            config_service.set_config("tuning.auto_interval", self.auto_tune_interval, "Auto-tuning interval in seconds")
            config_service.set_config("tuning.performance_threshold", self.performance_threshold, "Performance threshold for tuning")
            config_service.set_config("tuning.min_trades", self.min_trades_for_tuning, "Minimum trades before tuning")
            
            logger.info("Parameter tuner configuration saved")
        except Exception as e:
            logger.error(f"Failed to save parameter tuner configuration: {e}")
    
    def create_strategy_parameters(self, strategy_name: str) -> List[ParameterRange]:
        """Create parameter ranges for a specific strategy."""
        try:
            if strategy_name.upper() == "RSI":
                return [
                    ParameterRange("rsi_period", 10, 20, 1, "int", 14),
                    ParameterRange("rsi_overbought", 70, 85, 1, "int", 70),
                    ParameterRange("rsi_oversold", 15, 30, 1, "int", 30),
                    ParameterRange("volume_threshold", 0.5, 2.0, 0.1, "float", 1.0)
                ]
            elif strategy_name.upper() == "EMA":
                return [
                    ParameterRange("ema_short", 5, 15, 1, "int", 12),
                    ParameterRange("ema_long", 20, 30, 1, "int", 26),
                    ParameterRange("signal_threshold", 0.01, 0.05, 0.005, "float", 0.02),
                    ParameterRange("volume_confirmation", 0.8, 1.5, 0.1, "float", 1.0)
                ]
            elif strategy_name.upper() == "MACD":
                return [
                    ParameterRange("macd_fast", 8, 16, 1, "int", 12),
                    ParameterRange("macd_slow", 20, 30, 1, "int", 26),
                    ParameterRange("macd_signal", 7, 12, 1, "int", 9),
                    ParameterRange("divergence_threshold", 0.001, 0.01, 0.001, "float", 0.005)
                ]
            else:
                # Generic parameters
                return [
                    ParameterRange("lookback_period", 10, 30, 1, "int", 20),
                    ParameterRange("threshold", 0.01, 0.1, 0.01, "float", 0.05),
                    ParameterRange("volume_factor", 0.5, 2.0, 0.1, "float", 1.0)
                ]
                
        except Exception as e:
            logger.error(f"Error creating strategy parameters: {e}")
            return []
    
    def evaluate_parameters(self, strategy_name: str, param_set: Dict[str, Any]) -> OptimizationResult:
        """Evaluate a parameter set using backtesting."""
        try:
            start_time = time.time()
            
            # Create a modified strategy function with new parameters
            # This would need to be implemented based on your strategy structure
            
            # For now, simulate backtesting with the parameters
            start_date = datetime.utcnow() - timedelta(days=30)
            end_date = datetime.utcnow()
            
            # Run backtest with modified parameters
            backtest_result = self.optimizer.backtester.run_backtest(
                symbol="AAPL",  # Test symbol
                strategy_name=strategy_name,
                start_date=start_date,
                end_date=end_date
            )
            
            # Calculate performance score
            performance_score = self._calculate_performance_score(backtest_result)
            
            # Extract key metrics
            metrics = {
                "total_return": backtest_result.total_return_percent,
                "sharpe_ratio": backtest_result.sharpe_ratio,
                "max_drawdown": backtest_result.max_drawdown_percent,
                "win_rate": backtest_result.win_rate,
                "profit_factor": backtest_result.profit_factor
            }
            
            execution_time = time.time() - start_time
            
            return OptimizationResult(
                parameter_set=param_set,
                performance_score=performance_score,
                metrics=metrics,
                backtest_result=backtest_result,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Error evaluating parameters: {e}")
            return OptimizationResult(
                parameter_set=param_set,
                performance_score=0.0,
                metrics={},
                execution_time=0.0
            )
    
    def _calculate_performance_score(self, backtest_result: BacktestResult) -> float:
        """Calculate a composite performance score."""
        try:
            # Weighted combination of different metrics
            weights = {
                "return": 0.3,
                "sharpe": 0.3,
                "drawdown": 0.2,
                "win_rate": 0.1,
                "profit_factor": 0.1
            }
            
            # Normalize metrics to 0-1 scale
            return_score = max(0, min(1, (backtest_result.total_return_percent + 10) / 20))
            sharpe_score = max(0, min(1, (backtest_result.sharpe_ratio + 1) / 3))
            drawdown_score = max(0, min(1, 1 - abs(backtest_result.max_drawdown_percent) / 50))
            win_rate_score = backtest_result.win_rate / 100
            profit_factor_score = max(0, min(1, backtest_result.profit_factor / 3))
            
            # Calculate weighted score
            score = (
                return_score * weights["return"] +
                sharpe_score * weights["sharpe"] +
                drawdown_score * weights["drawdown"] +
                win_rate_score * weights["win_rate"] +
                profit_factor_score * weights["profit_factor"]
            )
            
            return score
            
        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 0.0
    
    def run_optimization(self, strategy_name: str, optimization_type: OptimizationType = OptimizationType.RANDOM_SEARCH,
                        max_iterations: int = 30) -> str:
        """Run parameter optimization for a strategy."""
        try:
            session_id = f"{strategy_name}_{optimization_type.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            logger.info(f"Starting parameter optimization: {session_id}")
            
            # Create parameter ranges
            parameter_ranges = self.create_strategy_parameters(strategy_name)
            
            # Create tuning session
            session = TuningSession(
                id=session_id,
                strategy_name=strategy_name,
                optimization_type=optimization_type,
                parameter_ranges=parameter_ranges,
                started_at=datetime.utcnow()
            )
            
            self.active_sessions[session_id] = session
            
            # Create evaluation function
            def evaluation_function(param_set):
                return self.evaluate_parameters(strategy_name, param_set)
            
            # Run optimization
            if optimization_type == OptimizationType.GRID_SEARCH:
                results = self.optimizer.grid_search(parameter_ranges, evaluation_function, max_iterations)
            elif optimization_type == OptimizationType.RANDOM_SEARCH:
                results = self.optimizer.random_search(parameter_ranges, evaluation_function, max_iterations)
            elif optimization_type == OptimizationType.BAYESIAN:
                results = self.optimizer.bayesian_optimization(parameter_ranges, evaluation_function, max_iterations)
            else:
                raise ValueError(f"Unsupported optimization type: {optimization_type}")
            
            # Update session with results
            session.all_results = results
            session.best_result = results[0] if results else None
            session.completed_at = datetime.utcnow()
            session.status = "completed"
            
            # Move to history
            self.tuning_history.append(session)
            del self.active_sessions[session_id]
            
            # Apply best parameters if significantly better
            if session.best_result and session.best_result.performance_score > 0.7:
                self.apply_optimized_parameters(strategy_name, session.best_result.parameter_set)
            
            logger.info(f"Parameter optimization completed: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Parameter optimization failed: {e}")
            if session_id in self.active_sessions:
                self.active_sessions[session_id].status = "failed"
                self.active_sessions[session_id].error_message = str(e)
            return ""
    
    def apply_optimized_parameters(self, strategy_name: str, parameters: Dict[str, Any]):
        """Apply optimized parameters to a strategy."""
        try:
            logger.info(f"Applying optimized parameters for {strategy_name}: {parameters}")
            
            # Save parameters to database
            for param_name, value in parameters.items():
                config_key = f"strategy.{strategy_name.lower()}.{param_name}"
                config_service.set_config(config_key, value, f"Optimized parameter for {strategy_name}")
            
            # Update strategy configuration timestamp
            config_service.set_config(
                f"strategy.{strategy_name.lower()}.last_optimized", 
                datetime.utcnow().isoformat(), 
                f"Last optimization timestamp for {strategy_name}"
            )
            
            logger.info(f"Successfully applied optimized parameters for {strategy_name}")
            
        except Exception as e:
            logger.error(f"Error applying optimized parameters: {e}")
    
    def should_tune_strategy(self, strategy_name: str) -> Tuple[bool, str]:
        """Determine if a strategy should be tuned."""
        try:
            # Check if enough trades have been executed
            recent_trades = trade_service.get_recent_trades(limit=self.min_trades_for_tuning)
            strategy_trades = [t for t in recent_trades if t.strategy_used == strategy_name]
            
            if len(strategy_trades) < self.min_trades_for_tuning:
                return False, f"Insufficient trades ({len(strategy_trades)} < {self.min_trades_for_tuning})"
            
            # Check performance
            metrics = self.optimizer.performance_analyzer.calculate_comprehensive_metrics()
            if metrics.sharpe_ratio < self.performance_threshold:
                return True, f"Poor performance (Sharpe: {metrics.sharpe_ratio:.3f})"
            
            # Check if strategy hasn't been optimized recently
            last_optimized = config_service.get_config(f"strategy.{strategy_name.lower()}.last_optimized", None)
            if last_optimized:
                try:
                    last_opt_date = datetime.fromisoformat(last_optimized)
                    days_since_opt = (datetime.utcnow() - last_opt_date).days
                    if days_since_opt > 30:  # Optimize every 30 days
                        return True, f"Strategy not optimized for {days_since_opt} days"
                except:
                    pass
            else:
                return True, "Strategy never optimized"
            
            return False, "Strategy performing well"
            
        except Exception as e:
            logger.error(f"Error checking if strategy should be tuned: {e}")
            return False, f"Error: {e}"
    
    def auto_tune_loop(self):
        """Automatic tuning loop."""
        logger.info("Auto-tuning loop started")
        
        while self.running:
            try:
                if not self.auto_tuning_enabled:
                    time.sleep(self.auto_tune_interval)
                    continue
                
                # Check each strategy
                strategies = ["RSI", "EMA", "MACD"]
                for strategy_name in strategies:
                    try:
                        should_tune, reason = self.should_tune_strategy(strategy_name)
                        
                        if should_tune:
                            logger.info(f"Auto-tuning {strategy_name}: {reason}")
                            session_id = self.run_optimization(strategy_name, OptimizationType.RANDOM_SEARCH, 20)
                            
                            if session_id:
                                logger.info(f"Auto-tuning completed for {strategy_name}")
                            else:
                                logger.error(f"Auto-tuning failed for {strategy_name}")
                        
                    except Exception as e:
                        logger.error(f"Error auto-tuning {strategy_name}: {e}")
                
                # Sleep until next check
                time.sleep(self.auto_tune_interval)
                
            except Exception as e:
                logger.error(f"Auto-tuning loop error: {e}")
                time.sleep(self.auto_tune_interval)
    
    def start_auto_tuning(self):
        """Start automatic parameter tuning."""
        if self.running:
            logger.warning("Auto-tuning already running")
            return
        
        self.running = True
        self.tuning_thread = threading.Thread(target=self.auto_tune_loop, daemon=True)
        self.tuning_thread.start()
        
        logger.info("Auto-tuning started")
    
    def stop_auto_tuning(self):
        """Stop automatic parameter tuning."""
        if not self.running:
            logger.warning("Auto-tuning not running")
            return
        
        self.running = False
        if self.tuning_thread:
            self.tuning_thread.join(timeout=10)
        
        logger.info("Auto-tuning stopped")
    
    def get_tuning_status(self) -> Dict:
        """Get comprehensive tuning status."""
        try:
            return {
                "auto_tuning_enabled": self.auto_tuning_enabled,
                "auto_tuning_running": self.running,
                "active_sessions": len(self.active_sessions),
                "completed_sessions": len(self.tuning_history),
                "recent_sessions": [
                    {
                        "id": session.id,
                        "strategy_name": session.strategy_name,
                        "optimization_type": session.optimization_type.value,
                        "started_at": session.started_at.isoformat(),
                        "completed_at": session.completed_at.isoformat() if session.completed_at else None,
                        "status": session.status,
                        "best_score": session.best_result.performance_score if session.best_result else 0.0,
                        "error_message": session.error_message
                    }
                    for session in self.tuning_history[-5:]  # Last 5 sessions
                ],
                "last_check": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting tuning status: {e}")
            return {"error": str(e)}


# Global parameter tuner instance
parameter_tuner = ParameterTuner()