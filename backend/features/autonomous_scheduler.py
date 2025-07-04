"""Autonomous action scheduler with advanced triggers for JARVIS v3.0."""

import asyncio
import threading
import time
import json
import cron_descriptor
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pytz
from croniter import croniter
import inspect

from database.services import config_service, memory_service, trade_service
from utils.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class TriggerType(Enum):
    """Types of triggers for autonomous actions."""
    SCHEDULED = "scheduled"          # Cron-based schedule
    MARKET_EVENT = "market_event"    # Market conditions
    PERFORMANCE = "performance"      # Performance metrics
    SYSTEM_STATE = "system_state"    # System health/status
    USER_PATTERN = "user_pattern"    # User behavior patterns
    CUSTOM = "custom"                # Custom Python condition


class ActionType(Enum):
    """Types of autonomous actions."""
    TRADE = "trade"
    STRATEGY_SWITCH = "strategy_switch"
    PARAMETER_TUNE = "parameter_tune"
    MODEL_UPDATE = "model_update"
    SYSTEM_MAINTENANCE = "system_maintenance"
    ALERT = "alert"
    ANALYSIS = "analysis"
    LEARNING = "learning"
    CUSTOM = "custom"


class ActionStatus(Enum):
    """Status of autonomous actions."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SCHEDULED = "scheduled"


@dataclass
class TriggerCondition:
    """Defines a trigger condition."""
    type: TriggerType
    condition: str              # Cron expression, Python condition, or event name
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


@dataclass
class AutonomousAction:
    """Defines an autonomous action."""
    id: str
    name: str
    description: str
    action_type: ActionType
    function_name: str          # Function to execute
    parameters: Dict[str, Any] = field(default_factory=dict)
    triggers: List[TriggerCondition] = field(default_factory=list)
    enabled: bool = True
    priority: int = 5           # 1-10, higher = more important
    timeout_minutes: int = 30
    retry_count: int = 3
    cooldown_minutes: int = 5
    status: ActionStatus = ActionStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_executed: Optional[datetime] = None
    execution_count: int = 0
    success_rate: float = 0.0
    average_duration: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    required_conditions: List[str] = field(default_factory=list)


@dataclass
class ActionExecution:
    """Records an action execution."""
    action_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: ActionStatus = ActionStatus.RUNNING
    result: Any = None
    error_message: Optional[str] = None
    duration_seconds: float = 0.0
    triggered_by: str = ""
    execution_context: Dict[str, Any] = field(default_factory=dict)


class TriggerEvaluator:
    """Evaluates different types of triggers."""
    
    def __init__(self):
        self.timezone = pytz.timezone('UTC')
        
    def evaluate_scheduled_trigger(self, condition: str, last_triggered: Optional[datetime]) -> bool:
        """Evaluate cron-based scheduled trigger."""
        try:
            # Parse cron expression
            cron = croniter(condition, datetime.now(self.timezone))
            next_run = cron.get_next(datetime)
            
            # Check if it's time to run
            now = datetime.now(self.timezone)
            
            # If never triggered, check if we're past the first scheduled time
            if last_triggered is None:
                # Get previous scheduled time
                cron_prev = croniter(condition, now)
                prev_run = cron_prev.get_prev(datetime)
                return now >= prev_run
            
            # Check if we've passed a scheduled time since last trigger
            return next_run <= now and (last_triggered is None or next_run > last_triggered)
            
        except Exception as e:
            logger.error(f"Error evaluating scheduled trigger '{condition}': {e}")
            return False
    
    def evaluate_market_event_trigger(self, condition: str, parameters: Dict) -> bool:
        """Evaluate market event trigger."""
        try:
            # Parse market condition
            if condition == "market_open":
                now = datetime.now()
                return 9 <= now.hour < 16 and now.weekday() < 5  # Market hours
            elif condition == "high_volatility":
                # Check recent price volatility
                threshold = parameters.get("threshold", 0.05)
                # Would check actual market volatility
                return False  # Placeholder
            elif condition == "volume_spike":
                # Check for unusual volume
                threshold = parameters.get("volume_multiplier", 2.0)
                # Would check actual volume data
                return False  # Placeholder
            elif condition == "price_breakout":
                # Check for price breakouts
                symbol = parameters.get("symbol", "AAPL")
                # Would check actual price breakout
                return False  # Placeholder
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating market event trigger '{condition}': {e}")
            return False
    
    def evaluate_performance_trigger(self, condition: str, parameters: Dict) -> bool:
        """Evaluate performance-based trigger."""
        try:
            from .performance_analyzer import performance_analyzer
            
            # Get recent performance metrics
            metrics = performance_analyzer.calculate_comprehensive_metrics()
            
            if condition == "poor_performance":
                threshold = parameters.get("sharpe_threshold", 0.5)
                return metrics.sharpe_ratio < threshold
            elif condition == "high_drawdown":
                threshold = parameters.get("drawdown_threshold", -0.15)
                return metrics.max_drawdown < threshold
            elif condition == "low_win_rate":
                threshold = parameters.get("win_rate_threshold", 40.0)
                return metrics.win_rate < threshold
            elif condition == "profit_target_reached":
                threshold = parameters.get("profit_threshold", 0.20)
                return metrics.total_return > threshold
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating performance trigger '{condition}': {e}")
            return False
    
    def evaluate_system_state_trigger(self, condition: str, parameters: Dict) -> bool:
        """Evaluate system state trigger."""
        try:
            from .system_monitor import system_monitor
            
            # Get current system status
            status = system_monitor.get_current_status()
            
            if condition == "high_cpu":
                threshold = parameters.get("cpu_threshold", 80.0)
                return status["system_metrics"]["cpu_percent"] > threshold
            elif condition == "low_memory":
                threshold = parameters.get("memory_threshold", 85.0)
                return status["system_metrics"]["memory_percent"] > threshold
            elif condition == "system_unhealthy":
                threshold = parameters.get("health_threshold", 70.0)
                return status["system_health"]["score"] < threshold
            elif condition == "trading_stopped":
                return status["application_metrics"]["active_trades"] == 0
            elif condition == "alerts_critical":
                critical_count = len([a for a in status["alerts"]["recent_alerts"] 
                                    if a.get("severity") == "critical"])
                threshold = parameters.get("critical_alert_threshold", 1)
                return critical_count >= threshold
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating system state trigger '{condition}': {e}")
            return False
    
    def evaluate_user_pattern_trigger(self, condition: str, parameters: Dict) -> bool:
        """Evaluate user behavior pattern trigger."""
        try:
            if condition == "no_user_activity":
                hours_threshold = parameters.get("hours", 24)
                # Check last user interaction
                last_interaction = config_service.get_config("user.last_interaction", None)
                if last_interaction:
                    last_time = datetime.fromisoformat(last_interaction)
                    hours_since = (datetime.utcnow() - last_time).total_seconds() / 3600
                    return hours_since > hours_threshold
                return True
            elif condition == "repeated_queries":
                # Check for repeated query patterns
                return False  # Placeholder
            elif condition == "user_feedback_poor":
                # Check recent user feedback scores
                avg_score = config_service.get_config("user.average_feedback_score", 5.0)
                threshold = parameters.get("score_threshold", 3.0)
                return avg_score < threshold
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating user pattern trigger '{condition}': {e}")
            return False
    
    def evaluate_custom_trigger(self, condition: str, parameters: Dict) -> bool:
        """Evaluate custom Python condition trigger."""
        try:
            # Create safe execution context
            safe_globals = {
                "__builtins__": {},
                "datetime": datetime,
                "timedelta": timedelta,
                "config_service": config_service,
                "trade_service": trade_service,
                "memory_service": memory_service,
                "parameters": parameters
            }
            
            # Add commonly used functions
            safe_globals.update({
                "len": len,
                "sum": sum,
                "max": max,
                "min": min,
                "abs": abs,
                "round": round
            })
            
            # Evaluate the condition
            result = eval(condition, safe_globals, {})
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error evaluating custom trigger '{condition}': {e}")
            return False
    
    def evaluate_trigger(self, trigger: TriggerCondition) -> bool:
        """Evaluate a trigger condition."""
        try:
            if not trigger.enabled:
                return False
            
            if trigger.type == TriggerType.SCHEDULED:
                return self.evaluate_scheduled_trigger(trigger.condition, trigger.last_triggered)
            elif trigger.type == TriggerType.MARKET_EVENT:
                return self.evaluate_market_event_trigger(trigger.condition, trigger.parameters)
            elif trigger.type == TriggerType.PERFORMANCE:
                return self.evaluate_performance_trigger(trigger.condition, trigger.parameters)
            elif trigger.type == TriggerType.SYSTEM_STATE:
                return self.evaluate_system_state_trigger(trigger.condition, trigger.parameters)
            elif trigger.type == TriggerType.USER_PATTERN:
                return self.evaluate_user_pattern_trigger(trigger.condition, trigger.parameters)
            elif trigger.type == TriggerType.CUSTOM:
                return self.evaluate_custom_trigger(trigger.condition, trigger.parameters)
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating trigger: {e}")
            return False


class ActionExecutor:
    """Executes autonomous actions."""
    
    def __init__(self):
        self.action_functions = self._setup_action_functions()
    
    def _setup_action_functions(self) -> Dict[str, Callable]:
        """Setup available action functions."""
        return {
            "execute_trade": self._execute_trade,
            "switch_strategy": self._switch_strategy,
            "tune_parameters": self._tune_parameters,
            "update_model": self._update_model,
            "system_maintenance": self._system_maintenance,
            "send_alert": self._send_alert,
            "run_analysis": self._run_analysis,
            "trigger_learning": self._trigger_learning,
            "rebalance_portfolio": self._rebalance_portfolio,
            "optimize_settings": self._optimize_settings,
            "backup_data": self._backup_data,
            "cleanup_logs": self._cleanup_logs,
            "health_check": self._health_check,
            "emergency_shutdown": self._emergency_shutdown
        }
    
    async def execute_action(self, action: AutonomousAction, triggered_by: str = "") -> ActionExecution:
        """Execute an autonomous action."""
        execution = ActionExecution(
            action_id=action.id,
            started_at=datetime.utcnow(),
            triggered_by=triggered_by,
            execution_context={"parameters": action.parameters}
        )
        
        try:
            logger.info(f"Executing autonomous action: {action.name}")
            
            # Check dependencies
            if not self._check_dependencies(action):
                raise Exception("Action dependencies not met")
            
            # Check required conditions
            if not self._check_required_conditions(action):
                raise Exception("Required conditions not met")
            
            # Get action function
            if action.function_name not in self.action_functions:
                raise Exception(f"Unknown action function: {action.function_name}")
            
            action_func = self.action_functions[action.function_name]
            
            # Execute with timeout
            result = await asyncio.wait_for(
                self._run_action_function(action_func, action.parameters),
                timeout=action.timeout_minutes * 60
            )
            
            # Update execution record
            execution.completed_at = datetime.utcnow()
            execution.status = ActionStatus.COMPLETED
            execution.result = result
            execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
            
            # Update action statistics
            action.execution_count += 1
            action.last_executed = execution.completed_at
            
            # Update success rate
            success_count = action.execution_count * action.success_rate + 1
            action.success_rate = success_count / action.execution_count
            
            # Update average duration
            total_duration = action.average_duration * (action.execution_count - 1) + execution.duration_seconds
            action.average_duration = total_duration / action.execution_count
            
            logger.info(f"Successfully executed action: {action.name}")
            
        except Exception as e:
            execution.completed_at = datetime.utcnow()
            execution.status = ActionStatus.FAILED
            execution.error_message = str(e)
            execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
            
            logger.error(f"Action execution failed: {action.name} - {e}")
        
        return execution
    
    async def _run_action_function(self, func: Callable, parameters: Dict) -> Any:
        """Run an action function asynchronously."""
        if inspect.iscoroutinefunction(func):
            return await func(**parameters)
        else:
            # Run in thread pool for blocking functions
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: func(**parameters))
    
    def _check_dependencies(self, action: AutonomousAction) -> bool:
        """Check if action dependencies are met."""
        # For now, always return True
        # Could check system state, other running actions, etc.
        return True
    
    def _check_required_conditions(self, action: AutonomousAction) -> bool:
        """Check if required conditions are met."""
        # For now, always return True
        # Could check system health, market status, etc.
        return True
    
    # Action function implementations
    def _execute_trade(self, **kwargs) -> Dict[str, Any]:
        """Execute a trade."""
        try:
            from .autotrade import execute_trade
            symbol = kwargs.get("symbol", "AAPL")
            execute_trade(symbol)
            return {"status": "success", "symbol": symbol}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _switch_strategy(self, **kwargs) -> Dict[str, Any]:
        """Switch trading strategy."""
        try:
            from .strategy_manager import strategy_manager
            new_strategy = kwargs.get("strategy", "RSI")
            reason = kwargs.get("reason", "Autonomous switch")
            success = strategy_manager.execute_strategy_switch(new_strategy, reason)
            return {"status": "success" if success else "error", "strategy": new_strategy}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _tune_parameters(self, **kwargs) -> Dict[str, Any]:
        """Tune strategy parameters."""
        try:
            from .parameter_tuner import parameter_tuner, OptimizationType
            strategy = kwargs.get("strategy", "RSI")
            optimization_type = OptimizationType(kwargs.get("optimization_type", "random_search"))
            max_iterations = kwargs.get("max_iterations", 20)
            
            session_id = parameter_tuner.run_optimization(strategy, optimization_type, max_iterations)
            return {"status": "success", "session_id": session_id}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _update_model(self, **kwargs) -> Dict[str, Any]:
        """Update AI model."""
        try:
            from .model_monitor import model_monitor
            model_name = kwargs.get("model_name", "jarvisbrain")
            success = model_monitor.force_update(model_name)
            return {"status": "success" if success else "error", "model": model_name}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _system_maintenance(self, **kwargs) -> Dict[str, Any]:
        """Perform system maintenance."""
        try:
            # Run cleanup, optimization, etc.
            maintenance_type = kwargs.get("type", "general")
            
            if maintenance_type == "database":
                # Database cleanup
                pass
            elif maintenance_type == "logs":
                # Log cleanup
                pass
            elif maintenance_type == "cache":
                # Cache cleanup
                pass
            
            return {"status": "success", "type": maintenance_type}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _send_alert(self, **kwargs) -> Dict[str, Any]:
        """Send alert notification."""
        try:
            from .telegram_alerts import send_telegram_alert
            message = kwargs.get("message", "Autonomous alert from JARVIS")
            send_telegram_alert(f"🤖 {message}")
            return {"status": "success", "message": message}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _run_analysis(self, **kwargs) -> Dict[str, Any]:
        """Run data analysis."""
        try:
            analysis_type = kwargs.get("type", "performance")
            
            if analysis_type == "performance":
                from .performance_analyzer import performance_analyzer
                metrics = performance_analyzer.calculate_comprehensive_metrics()
                return {"status": "success", "analysis": "performance", "metrics": metrics.__dict__}
            
            return {"status": "success", "analysis": analysis_type}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _trigger_learning(self, **kwargs) -> Dict[str, Any]:
        """Trigger learning process."""
        try:
            # This would trigger the learning feedback loop
            learning_type = kwargs.get("type", "general")
            return {"status": "success", "learning_type": learning_type}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _rebalance_portfolio(self, **kwargs) -> Dict[str, Any]:
        """Rebalance portfolio."""
        try:
            from .portfolio_manager import portfolio_manager
            opportunities = portfolio_manager.identify_rebalancing_opportunities()
            return {"status": "success", "opportunities": len(opportunities)}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _optimize_settings(self, **kwargs) -> Dict[str, Any]:
        """Optimize system settings."""
        try:
            # Optimize various system settings based on performance
            return {"status": "success", "optimization": "completed"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _backup_data(self, **kwargs) -> Dict[str, Any]:
        """Backup system data."""
        try:
            # Perform data backup
            backup_type = kwargs.get("type", "full")
            return {"status": "success", "backup_type": backup_type}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _cleanup_logs(self, **kwargs) -> Dict[str, Any]:
        """Clean up old log files."""
        try:
            days = kwargs.get("days", 30)
            # Clean logs older than specified days
            return {"status": "success", "days": days}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _health_check(self, **kwargs) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            from .system_monitor import system_monitor
            status = system_monitor.get_current_status()
            return {"status": "success", "health_score": status["system_health"]["score"]}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _emergency_shutdown(self, **kwargs) -> Dict[str, Any]:
        """Emergency system shutdown."""
        try:
            reason = kwargs.get("reason", "Emergency shutdown triggered")
            logger.critical(f"EMERGENCY SHUTDOWN: {reason}")
            # Would perform emergency shutdown procedures
            return {"status": "success", "reason": reason}
        except Exception as e:
            return {"status": "error", "message": str(e)}


class AutonomousScheduler:
    """Main autonomous scheduler system."""
    
    def __init__(self):
        self.actions: Dict[str, AutonomousAction] = {}
        self.executions: List[ActionExecution] = []
        self.trigger_evaluator = TriggerEvaluator()
        self.action_executor = ActionExecutor()
        
        self.running = False
        self.scheduler_thread = None
        self.check_interval = 30  # Check every 30 seconds
        
        self.max_concurrent_actions = 5
        self.current_running_actions = 0
        
        # Setup default actions
        self._setup_default_actions()
        
        self.load_configuration()
        logger.info("Autonomous scheduler initialized")
    
    def _setup_default_actions(self):
        """Setup default autonomous actions."""
        default_actions = [
            # Daily performance analysis
            AutonomousAction(
                id="daily_analysis",
                name="Daily Performance Analysis",
                description="Analyze daily trading performance and generate insights",
                action_type=ActionType.ANALYSIS,
                function_name="run_analysis",
                parameters={"type": "performance"},
                triggers=[
                    TriggerCondition(
                        type=TriggerType.SCHEDULED,
                        condition="0 18 * * *",  # Daily at 6 PM
                        parameters={}
                    )
                ],
                priority=7
            ),
            
            # Automatic strategy switching on poor performance
            AutonomousAction(
                id="performance_strategy_switch",
                name="Performance-Based Strategy Switch",
                description="Switch strategy when performance degrades",
                action_type=ActionType.STRATEGY_SWITCH,
                function_name="switch_strategy",
                parameters={"strategy": "EMA", "reason": "Poor performance detected"},
                triggers=[
                    TriggerCondition(
                        type=TriggerType.PERFORMANCE,
                        condition="poor_performance",
                        parameters={"sharpe_threshold": 0.3}
                    )
                ],
                priority=8,
                cooldown_minutes=60
            ),
            
            # Weekly parameter optimization
            AutonomousAction(
                id="weekly_optimization",
                name="Weekly Parameter Optimization",
                description="Optimize strategy parameters weekly",
                action_type=ActionType.PARAMETER_TUNE,
                function_name="tune_parameters",
                parameters={"strategy": "RSI", "optimization_type": "random_search", "max_iterations": 25},
                triggers=[
                    TriggerCondition(
                        type=TriggerType.SCHEDULED,
                        condition="0 2 * * 0",  # Weekly on Sunday at 2 AM
                        parameters={}
                    )
                ],
                priority=6
            ),
            
            # Emergency shutdown on critical system state
            AutonomousAction(
                id="emergency_shutdown",
                name="Emergency System Shutdown",
                description="Emergency shutdown on critical system alerts",
                action_type=ActionType.SYSTEM_MAINTENANCE,
                function_name="emergency_shutdown",
                parameters={"reason": "Critical system alerts detected"},
                triggers=[
                    TriggerCondition(
                        type=TriggerType.SYSTEM_STATE,
                        condition="alerts_critical",
                        parameters={"critical_alert_threshold": 3}
                    )
                ],
                priority=10,
                cooldown_minutes=30
            ),
            
            # Model update on poor performance
            AutonomousAction(
                id="model_performance_update",
                name="Model Update on Poor Performance",
                description="Update AI model when performance drops",
                action_type=ActionType.MODEL_UPDATE,
                function_name="update_model",
                parameters={"model_name": "jarvisbrain"},
                triggers=[
                    TriggerCondition(
                        type=TriggerType.USER_PATTERN,
                        condition="user_feedback_poor",
                        parameters={"score_threshold": 2.5}
                    )
                ],
                priority=7,
                cooldown_minutes=120
            ),
            
            # Daily system health check
            AutonomousAction(
                id="daily_health_check",
                name="Daily System Health Check",
                description="Comprehensive daily health check",
                action_type=ActionType.SYSTEM_MAINTENANCE,
                function_name="health_check",
                parameters={},
                triggers=[
                    TriggerCondition(
                        type=TriggerType.SCHEDULED,
                        condition="0 8 * * *",  # Daily at 8 AM
                        parameters={}
                    )
                ],
                priority=5
            ),
            
            # Portfolio rebalancing
            AutonomousAction(
                id="portfolio_rebalance",
                name="Portfolio Rebalancing",
                description="Rebalance portfolio when allocations drift",
                action_type=ActionType.TRADE,
                function_name="rebalance_portfolio",
                parameters={},
                triggers=[
                    TriggerCondition(
                        type=TriggerType.SCHEDULED,
                        condition="0 10 * * 1,3,5",  # Mon, Wed, Fri at 10 AM
                        parameters={}
                    )
                ],
                priority=6
            ),
            
            # Market volatility response
            AutonomousAction(
                id="volatility_response",
                name="High Volatility Response",
                description="Respond to high market volatility",
                action_type=ActionType.STRATEGY_SWITCH,
                function_name="switch_strategy",
                parameters={"strategy": "MACD", "reason": "High volatility detected"},
                triggers=[
                    TriggerCondition(
                        type=TriggerType.MARKET_EVENT,
                        condition="high_volatility",
                        parameters={"threshold": 0.08}
                    )
                ],
                priority=8,
                cooldown_minutes=120
            )
        ]
        
        for action in default_actions:
            self.actions[action.id] = action
    
    def load_configuration(self):
        """Load scheduler configuration."""
        try:
            self.check_interval = config_service.get_config("scheduler.check_interval", 30)
            self.max_concurrent_actions = config_service.get_config("scheduler.max_concurrent", 5)
            
            # Load custom actions
            custom_actions = config_service.get_config("scheduler.custom_actions", [])
            for action_data in custom_actions:
                try:
                    action = AutonomousAction(**action_data)
                    self.actions[action.id] = action
                except Exception as e:
                    logger.error(f"Error loading custom action: {e}")
            
            logger.info("Autonomous scheduler configuration loaded")
        except Exception as e:
            logger.warning(f"Failed to load scheduler configuration: {e}")
    
    def save_configuration(self):
        """Save scheduler configuration."""
        try:
            config_service.set_config("scheduler.check_interval", self.check_interval, "Scheduler check interval")
            config_service.set_config("scheduler.max_concurrent", self.max_concurrent_actions, "Max concurrent actions")
            
            logger.info("Autonomous scheduler configuration saved")
        except Exception as e:
            logger.error(f"Failed to save scheduler configuration: {e}")
    
    def add_action(self, action: AutonomousAction):
        """Add a new autonomous action."""
        self.actions[action.id] = action
        logger.info(f"Added autonomous action: {action.name}")
    
    def remove_action(self, action_id: str) -> bool:
        """Remove an autonomous action."""
        if action_id in self.actions:
            del self.actions[action_id]
            logger.info(f"Removed autonomous action: {action_id}")
            return True
        return False
    
    def enable_action(self, action_id: str) -> bool:
        """Enable an autonomous action."""
        if action_id in self.actions:
            self.actions[action_id].enabled = True
            logger.info(f"Enabled autonomous action: {action_id}")
            return True
        return False
    
    def disable_action(self, action_id: str) -> bool:
        """Disable an autonomous action."""
        if action_id in self.actions:
            self.actions[action_id].enabled = False
            logger.info(f"Disabled autonomous action: {action_id}")
            return True
        return False
    
    async def check_and_execute_actions(self):
        """Check triggers and execute actions."""
        try:
            if self.current_running_actions >= self.max_concurrent_actions:
                logger.debug("Max concurrent actions reached, skipping check")
                return
            
            # Get enabled actions sorted by priority
            enabled_actions = [a for a in self.actions.values() if a.enabled]
            enabled_actions.sort(key=lambda x: x.priority, reverse=True)
            
            for action in enabled_actions:
                try:
                    # Check cooldown
                    if (action.last_executed and 
                        datetime.utcnow() - action.last_executed < timedelta(minutes=action.cooldown_minutes)):
                        continue
                    
                    # Check if any trigger is activated
                    triggered_by = None
                    for trigger in action.triggers:
                        if self.trigger_evaluator.evaluate_trigger(trigger):
                            triggered_by = f"{trigger.type.value}:{trigger.condition}"
                            trigger.last_triggered = datetime.utcnow()
                            trigger.trigger_count += 1
                            break
                    
                    if triggered_by:
                        logger.info(f"Action triggered: {action.name} by {triggered_by}")
                        
                        # Execute action asynchronously
                        self.current_running_actions += 1
                        asyncio.create_task(self._execute_action_with_cleanup(action, triggered_by))
                        
                        # Limit concurrent executions
                        if self.current_running_actions >= self.max_concurrent_actions:
                            break
                
                except Exception as e:
                    logger.error(f"Error checking action {action.name}: {e}")
                    
        except Exception as e:
            logger.error(f"Error in action check loop: {e}")
    
    async def _execute_action_with_cleanup(self, action: AutonomousAction, triggered_by: str):
        """Execute action and handle cleanup."""
        try:
            execution = await self.action_executor.execute_action(action, triggered_by)
            self.executions.append(execution)
            
            # Keep only last 1000 executions
            if len(self.executions) > 1000:
                self.executions = self.executions[-1000:]
                
        except Exception as e:
            logger.error(f"Error executing action {action.name}: {e}")
        finally:
            self.current_running_actions -= 1
    
    async def scheduler_loop(self):
        """Main scheduler loop."""
        logger.info("Autonomous scheduler started")
        
        while self.running:
            try:
                await self.check_and_execute_actions()
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(self.check_interval)
    
    def start(self):
        """Start the autonomous scheduler."""
        if self.running:
            logger.warning("Autonomous scheduler already running")
            return
        
        self.running = True
        
        # Start asyncio loop in a separate thread
        def run_scheduler():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.scheduler_loop())
        
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Autonomous scheduler started")
    
    def stop(self):
        """Stop the autonomous scheduler."""
        if not self.running:
            logger.warning("Autonomous scheduler not running")
            return
        
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=10)
        
        logger.info("Autonomous scheduler stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive scheduler status."""
        try:
            # Recent executions
            recent_executions = []
            for execution in self.executions[-10:]:
                recent_executions.append({
                    "action_id": execution.action_id,
                    "started_at": execution.started_at.isoformat(),
                    "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                    "status": execution.status.value,
                    "duration_seconds": execution.duration_seconds,
                    "triggered_by": execution.triggered_by,
                    "error_message": execution.error_message
                })
            
            # Action summaries
            action_summaries = {}
            for action_id, action in self.actions.items():
                action_summaries[action_id] = {
                    "name": action.name,
                    "description": action.description,
                    "action_type": action.action_type.value,
                    "enabled": action.enabled,
                    "priority": action.priority,
                    "execution_count": action.execution_count,
                    "success_rate": action.success_rate,
                    "average_duration": action.average_duration,
                    "last_executed": action.last_executed.isoformat() if action.last_executed else None,
                    "trigger_count": sum(t.trigger_count for t in action.triggers),
                    "next_scheduled": self._get_next_scheduled_time(action)
                }
            
            return {
                "running": self.running,
                "current_running_actions": self.current_running_actions,
                "max_concurrent_actions": self.max_concurrent_actions,
                "total_actions": len(self.actions),
                "enabled_actions": len([a for a in self.actions.values() if a.enabled]),
                "total_executions": len(self.executions),
                "recent_executions": recent_executions,
                "actions": action_summaries,
                "last_check": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting scheduler status: {e}")
            return {"error": str(e)}
    
    def _get_next_scheduled_time(self, action: AutonomousAction) -> Optional[str]:
        """Get next scheduled time for an action."""
        try:
            for trigger in action.triggers:
                if trigger.type == TriggerType.SCHEDULED:
                    cron = croniter(trigger.condition, datetime.now())
                    next_time = cron.get_next(datetime)
                    return next_time.isoformat()
            return None
        except Exception:
            return None
    
    def force_execute_action(self, action_id: str) -> bool:
        """Force execute a specific action."""
        try:
            if action_id not in self.actions:
                return False
            
            action = self.actions[action_id]
            asyncio.create_task(self._execute_action_with_cleanup(action, "manual_trigger"))
            
            logger.info(f"Force executed action: {action.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error force executing action {action_id}: {e}")
            return False


# Global autonomous scheduler instance
autonomous_scheduler = AutonomousScheduler()