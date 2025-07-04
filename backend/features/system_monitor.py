"""Comprehensive system monitoring and analytics for JARVIS v3.0."""

import psutil
import platform
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import subprocess
import requests

from database.services import config_service, memory_service, trade_service
from utils.logging_config import get_logger
from config.settings import settings
from .self_healing import self_healing_daemon
from .model_monitor import model_monitor
from .parameter_tuner import parameter_tuner
from .plugin_manager import plugin_manager
from .performance_analyzer import performance_analyzer

logger = get_logger(__name__)


@dataclass
class SystemMetrics:
    """System performance metrics snapshot."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_sent: int
    network_received: int
    process_count: int
    uptime_seconds: int
    load_average: List[float] = field(default_factory=list)
    gpu_percent: Optional[float] = None
    gpu_memory_percent: Optional[float] = None


@dataclass
class ApplicationMetrics:
    """Application-specific metrics."""
    timestamp: datetime
    active_trades: int
    portfolio_value: float
    daily_pnl: float
    api_requests_count: int
    database_connections: int
    memory_usage_mb: float
    active_strategies: List[str] = field(default_factory=list)
    plugin_count: int = 0
    model_performance: Dict[str, float] = field(default_factory=dict)


@dataclass
class AlertRule:
    """System alert rule definition."""
    name: str
    condition: str  # Python expression to evaluate
    threshold: float
    severity: str  # low, medium, high, critical
    enabled: bool = True
    cooldown_minutes: int = 15
    last_triggered: Optional[datetime] = None
    action: Optional[str] = None  # Action to take when triggered


class MetricsCollector:
    """Collects various system and application metrics."""
    
    def __init__(self):
        self.start_time = time.time()
        self.network_baseline = self._get_network_stats()
        
    def _get_network_stats(self) -> Dict[str, int]:
        """Get network I/O statistics."""
        try:
            net_io = psutil.net_io_counters()
            return {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv
            }
        except Exception:
            return {"bytes_sent": 0, "bytes_recv": 0}
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect system-level metrics."""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            
            # Network
            current_net = self._get_network_stats()
            net_sent = current_net["bytes_sent"] - self.network_baseline["bytes_sent"]
            net_recv = current_net["bytes_recv"] - self.network_baseline["bytes_recv"]
            
            # Processes
            process_count = len(psutil.pids())
            
            # System uptime
            uptime_seconds = int(time.time() - self.start_time)
            
            # Load average (Unix-like systems)
            load_average = []
            try:
                if hasattr(os, 'getloadavg'):
                    load_average = list(os.getloadavg())
            except Exception:
                pass
            
            # GPU metrics (if available)
            gpu_percent = None
            gpu_memory_percent = None
            try:
                # This would integrate with nvidia-smi or similar
                # For now, skip GPU metrics
                pass
            except Exception:
                pass
            
            return SystemMetrics(
                timestamp=datetime.utcnow(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                network_sent=net_sent,
                network_received=net_recv,
                process_count=process_count,
                uptime_seconds=uptime_seconds,
                load_average=load_average,
                gpu_percent=gpu_percent,
                gpu_memory_percent=gpu_memory_percent
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.utcnow(),
                cpu_percent=0, memory_percent=0, disk_percent=0,
                network_sent=0, network_received=0, process_count=0,
                uptime_seconds=0
            )
    
    def collect_application_metrics(self) -> ApplicationMetrics:
        """Collect application-specific metrics."""
        try:
            # Trading metrics
            recent_trades = trade_service.get_recent_trades(limit=100)
            active_trades = len([t for t in recent_trades if t.status == "pending"])
            
            # Portfolio value (simplified)
            portfolio_value = 10000.0  # Would get from actual portfolio
            daily_pnl = 0.0  # Would calculate from trades
            
            # API metrics
            api_requests_count = 0  # Would track API calls
            
            # Database connections
            database_connections = 1  # Would get from connection pool
            
            # Memory usage of current process
            process = psutil.Process()
            memory_usage_mb = process.memory_info().rss / (1024 * 1024)
            
            # Active strategies
            active_strategies = ["RSI", "EMA", "MACD"]  # Would get from strategy manager
            
            # Plugin count
            plugin_count = len(plugin_manager.get_plugin_list()) if plugin_manager else 0
            
            # Model performance
            model_performance = {}
            if model_monitor:
                status = model_monitor.get_status()
                for model_name, model_data in status.get("models", {}).items():
                    model_performance[model_name] = model_data.get("performance_score", 0.0)
            
            return ApplicationMetrics(
                timestamp=datetime.utcnow(),
                active_trades=active_trades,
                portfolio_value=portfolio_value,
                daily_pnl=daily_pnl,
                api_requests_count=api_requests_count,
                database_connections=database_connections,
                memory_usage_mb=memory_usage_mb,
                active_strategies=active_strategies,
                plugin_count=plugin_count,
                model_performance=model_performance
            )
            
        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
            return ApplicationMetrics(
                timestamp=datetime.utcnow(),
                active_trades=0, portfolio_value=0, daily_pnl=0,
                api_requests_count=0, database_connections=0,
                memory_usage_mb=0
            )


class AlertManager:
    """Manages system alerts and notifications."""
    
    def __init__(self):
        self.alert_rules: List[AlertRule] = []
        self.triggered_alerts: deque = deque(maxlen=1000)
        self.notification_handlers: List[Callable] = []
        
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default alert rules."""
        default_rules = [
            AlertRule(
                name="High CPU Usage",
                condition="cpu_percent > 85",
                threshold=85.0,
                severity="high",
                cooldown_minutes=10
            ),
            AlertRule(
                name="High Memory Usage",
                condition="memory_percent > 90",
                threshold=90.0,
                severity="critical",
                cooldown_minutes=5
            ),
            AlertRule(
                name="Low Disk Space",
                condition="disk_percent > 95",
                threshold=95.0,
                severity="critical",
                cooldown_minutes=15
            ),
            AlertRule(
                name="Trading System Down",
                condition="active_trades == 0 and portfolio_value > 0",
                threshold=0,
                severity="high",
                cooldown_minutes=30
            ),
            AlertRule(
                name="Model Performance Degraded",
                condition="any(score < 0.5 for score in model_performance.values())",
                threshold=0.5,
                severity="medium",
                cooldown_minutes=60
            )
        ]
        
        self.alert_rules.extend(default_rules)
    
    def add_notification_handler(self, handler: Callable):
        """Add a notification handler function."""
        self.notification_handlers.append(handler)
    
    def evaluate_alerts(self, system_metrics: SystemMetrics, app_metrics: ApplicationMetrics):
        """Evaluate all alert rules against current metrics."""
        try:
            # Create evaluation context
            context = {
                "cpu_percent": system_metrics.cpu_percent,
                "memory_percent": system_metrics.memory_percent,
                "disk_percent": system_metrics.disk_percent,
                "process_count": system_metrics.process_count,
                "active_trades": app_metrics.active_trades,
                "portfolio_value": app_metrics.portfolio_value,
                "daily_pnl": app_metrics.daily_pnl,
                "memory_usage_mb": app_metrics.memory_usage_mb,
                "plugin_count": app_metrics.plugin_count,
                "model_performance": app_metrics.model_performance,
                "datetime": datetime,
                "timedelta": timedelta
            }
            
            for rule in self.alert_rules:
                if not rule.enabled:
                    continue
                
                # Check cooldown
                if (rule.last_triggered and 
                    datetime.utcnow() - rule.last_triggered < timedelta(minutes=rule.cooldown_minutes)):
                    continue
                
                try:
                    # Evaluate the condition
                    if eval(rule.condition, {"__builtins__": {}}, context):
                        self._trigger_alert(rule, context)
                except Exception as e:
                    logger.error(f"Error evaluating alert rule '{rule.name}': {e}")
                    
        except Exception as e:
            logger.error(f"Error evaluating alerts: {e}")
    
    def _trigger_alert(self, rule: AlertRule, context: Dict):
        """Trigger an alert."""
        try:
            alert_data = {
                "rule_name": rule.name,
                "severity": rule.severity,
                "condition": rule.condition,
                "threshold": rule.threshold,
                "triggered_at": datetime.utcnow(),
                "context": {k: v for k, v in context.items() if isinstance(v, (int, float, str, list, dict))}
            }
            
            # Update rule
            rule.last_triggered = datetime.utcnow()
            
            # Store alert
            self.triggered_alerts.append(alert_data)
            
            # Log alert
            logger.warning(f"ALERT TRIGGERED: {rule.name} (severity: {rule.severity})")
            
            # Send notifications
            for handler in self.notification_handlers:
                try:
                    handler(alert_data)
                except Exception as e:
                    logger.error(f"Error sending alert notification: {e}")
                    
        except Exception as e:
            logger.error(f"Error triggering alert: {e}")
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent alerts within specified hours."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            recent = [
                alert for alert in self.triggered_alerts
                if alert["triggered_at"] >= cutoff_time
            ]
            
            return recent
        except Exception as e:
            logger.error(f"Error getting recent alerts: {e}")
            return []


class SystemMonitor:
    """Main system monitoring coordinator."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.running = False
        self.monitor_thread = None
        
        # Data storage
        self.system_metrics_history: deque = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        self.app_metrics_history: deque = deque(maxlen=1440)
        
        # Configuration
        self.collection_interval = 60  # Collect every minute
        self.retention_hours = 24
        
        # Setup notification handlers
        self._setup_notification_handlers()
        
        self.load_configuration()
        logger.info("System monitor initialized")
    
    def _setup_notification_handlers(self):
        """Setup notification handlers for alerts."""
        def log_alert(alert_data):
            logger.critical(f"SYSTEM ALERT: {alert_data['rule_name']} - {alert_data['severity']}")
        
        def telegram_alert(alert_data):
            try:
                # This would send to Telegram
                from .telegram_alerts import send_telegram_alert
                message = f"🚨 JARVIS ALERT: {alert_data['rule_name']} ({alert_data['severity']})"
                send_telegram_alert(message)
            except Exception as e:
                logger.error(f"Failed to send Telegram alert: {e}")
        
        self.alert_manager.add_notification_handler(log_alert)
        # Uncomment to enable Telegram alerts
        # self.alert_manager.add_notification_handler(telegram_alert)
    
    def load_configuration(self):
        """Load monitoring configuration."""
        try:
            self.collection_interval = config_service.get_config("monitor.collection_interval", 60)
            self.retention_hours = config_service.get_config("monitor.retention_hours", 24)
            
            # Load custom alert rules
            custom_rules = config_service.get_config("monitor.custom_alert_rules", [])
            for rule_data in custom_rules:
                try:
                    rule = AlertRule(**rule_data)
                    self.alert_manager.alert_rules.append(rule)
                except Exception as e:
                    logger.error(f"Error loading custom alert rule: {e}")
            
            logger.info("System monitor configuration loaded")
        except Exception as e:
            logger.warning(f"Failed to load system monitor configuration: {e}")
    
    def save_configuration(self):
        """Save monitoring configuration."""
        try:
            config_service.set_config("monitor.collection_interval", self.collection_interval, "Metrics collection interval")
            config_service.set_config("monitor.retention_hours", self.retention_hours, "Metrics retention period")
            
            logger.info("System monitor configuration saved")
        except Exception as e:
            logger.error(f"Failed to save system monitor configuration: {e}")
    
    def collect_and_store_metrics(self):
        """Collect and store current metrics."""
        try:
            # Collect metrics
            system_metrics = self.metrics_collector.collect_system_metrics()
            app_metrics = self.metrics_collector.collect_application_metrics()
            
            # Store in memory
            self.system_metrics_history.append(system_metrics)
            self.app_metrics_history.append(app_metrics)
            
            # Evaluate alerts
            self.alert_manager.evaluate_alerts(system_metrics, app_metrics)
            
            # Store in database (optional - for persistence)
            try:
                metrics_data = {
                    "system": {
                        "cpu_percent": system_metrics.cpu_percent,
                        "memory_percent": system_metrics.memory_percent,
                        "disk_percent": system_metrics.disk_percent,
                        "network_sent": system_metrics.network_sent,
                        "network_received": system_metrics.network_received
                    },
                    "application": {
                        "active_trades": app_metrics.active_trades,
                        "portfolio_value": app_metrics.portfolio_value,
                        "daily_pnl": app_metrics.daily_pnl,
                        "memory_usage_mb": app_metrics.memory_usage_mb
                    }
                }
                
                memory_service.store_memory(
                    "system_metrics",
                    f"System metrics at {system_metrics.timestamp.isoformat()}",
                    tags=["monitoring", "metrics"],
                    metadata=metrics_data
                )
                
            except Exception as e:
                logger.warning(f"Failed to store metrics in database: {e}")
                
        except Exception as e:
            logger.error(f"Error collecting and storing metrics: {e}")
    
    def monitor_loop(self):
        """Main monitoring loop."""
        logger.info("System monitoring started")
        
        while self.running:
            try:
                self.collect_and_store_metrics()
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"System monitor loop error: {e}")
                time.sleep(self.collection_interval)
    
    def start(self):
        """Start system monitoring."""
        if self.running:
            logger.warning("System monitoring already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("System monitoring started")
    
    def stop(self):
        """Stop system monitoring."""
        if not self.running:
            logger.warning("System monitoring not running")
            return
        
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        
        logger.info("System monitoring stopped")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current system status."""
        try:
            # Get latest metrics
            system_metrics = self.metrics_collector.collect_system_metrics()
            app_metrics = self.metrics_collector.collect_application_metrics()
            
            # Get component statuses
            component_status = {}
            
            # Self-healing daemon
            if self_healing_daemon:
                component_status["self_healing"] = self_healing_daemon.get_health_status()
            
            # Model monitor
            if model_monitor:
                component_status["model_monitor"] = model_monitor.get_status()
            
            # Parameter tuner
            if parameter_tuner:
                component_status["parameter_tuner"] = parameter_tuner.get_tuning_status()
            
            # Plugin manager
            if plugin_manager:
                component_status["plugin_manager"] = plugin_manager.get_status()
            
            # Performance analyzer
            performance_metrics = None
            if performance_analyzer:
                try:
                    performance_metrics = performance_analyzer.calculate_comprehensive_metrics()
                except Exception as e:
                    logger.error(f"Error getting performance metrics: {e}")
            
            # Recent alerts
            recent_alerts = self.alert_manager.get_recent_alerts(hours=24)
            
            return {
                "monitoring_active": self.running,
                "timestamp": datetime.utcnow().isoformat(),
                "system_metrics": {
                    "cpu_percent": system_metrics.cpu_percent,
                    "memory_percent": system_metrics.memory_percent,
                    "disk_percent": system_metrics.disk_percent,
                    "process_count": system_metrics.process_count,
                    "uptime_seconds": system_metrics.uptime_seconds,
                    "load_average": system_metrics.load_average
                },
                "application_metrics": {
                    "active_trades": app_metrics.active_trades,
                    "portfolio_value": app_metrics.portfolio_value,
                    "daily_pnl": app_metrics.daily_pnl,
                    "memory_usage_mb": app_metrics.memory_usage_mb,
                    "active_strategies": app_metrics.active_strategies,
                    "plugin_count": app_metrics.plugin_count
                },
                "component_status": component_status,
                "performance_metrics": {
                    "total_return": performance_metrics.total_return if performance_metrics else 0,
                    "sharpe_ratio": performance_metrics.sharpe_ratio if performance_metrics else 0,
                    "max_drawdown": performance_metrics.max_drawdown if performance_metrics else 0,
                    "win_rate": performance_metrics.win_rate if performance_metrics else 0
                } if performance_metrics else {},
                "alerts": {
                    "active_rules": len([r for r in self.alert_manager.alert_rules if r.enabled]),
                    "recent_count": len(recent_alerts),
                    "recent_alerts": recent_alerts[-5:]  # Last 5 alerts
                },
                "system_health": self._calculate_overall_health(system_metrics, app_metrics, recent_alerts)
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"error": str(e)}
    
    def _calculate_overall_health(self, system_metrics: SystemMetrics, 
                                app_metrics: ApplicationMetrics, recent_alerts: List) -> Dict[str, Any]:
        """Calculate overall system health score."""
        try:
            health_score = 100.0
            issues = []
            
            # System resource penalties
            if system_metrics.cpu_percent > 80:
                health_score -= min(20, (system_metrics.cpu_percent - 80) * 2)
                issues.append("High CPU usage")
            
            if system_metrics.memory_percent > 85:
                health_score -= min(25, (system_metrics.memory_percent - 85) * 2)
                issues.append("High memory usage")
            
            if system_metrics.disk_percent > 90:
                health_score -= min(30, (system_metrics.disk_percent - 90) * 3)
                issues.append("Low disk space")
            
            # Application penalties
            if app_metrics.memory_usage_mb > 1000:  # 1GB
                health_score -= min(15, (app_metrics.memory_usage_mb - 1000) / 100)
                issues.append("High application memory usage")
            
            # Recent alerts penalty
            critical_alerts = len([a for a in recent_alerts if a.get("severity") == "critical"])
            high_alerts = len([a for a in recent_alerts if a.get("severity") == "high"])
            
            health_score -= critical_alerts * 15
            health_score -= high_alerts * 10
            
            if critical_alerts > 0:
                issues.append(f"{critical_alerts} critical alerts")
            if high_alerts > 0:
                issues.append(f"{high_alerts} high priority alerts")
            
            # Ensure score doesn't go below 0
            health_score = max(0, health_score)
            
            # Determine health status
            if health_score >= 90:
                status = "excellent"
            elif health_score >= 75:
                status = "good"
            elif health_score >= 50:
                status = "fair"
            elif health_score >= 25:
                status = "poor"
            else:
                status = "critical"
            
            return {
                "score": round(health_score, 1),
                "status": status,
                "issues": issues,
                "last_calculated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating system health: {e}")
            return {
                "score": 0,
                "status": "error",
                "issues": ["Health calculation error"],
                "last_calculated": datetime.utcnow().isoformat()
            }
    
    def get_metrics_history(self, hours: int = 6) -> Dict[str, List]:
        """Get metrics history for specified hours."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            system_history = [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "cpu_percent": m.cpu_percent,
                    "memory_percent": m.memory_percent,
                    "disk_percent": m.disk_percent,
                    "network_sent": m.network_sent,
                    "network_received": m.network_received
                }
                for m in self.system_metrics_history
                if m.timestamp >= cutoff_time
            ]
            
            app_history = [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "active_trades": m.active_trades,
                    "portfolio_value": m.portfolio_value,
                    "daily_pnl": m.daily_pnl,
                    "memory_usage_mb": m.memory_usage_mb
                }
                for m in self.app_metrics_history
                if m.timestamp >= cutoff_time
            ]
            
            return {
                "system_metrics": system_history,
                "application_metrics": app_history
            }
            
        except Exception as e:
            logger.error(f"Error getting metrics history: {e}")
            return {"system_metrics": [], "application_metrics": []}


# Global system monitor instance
system_monitor = SystemMonitor()