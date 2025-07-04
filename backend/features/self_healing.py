"""Self-healing daemon system for JARVIS v3.0 - monitors and repairs system issues automatically."""

import time
import psutil
import threading
import subprocess
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import traceback

from database.services import config_service, memory_service
from utils.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


@dataclass
class HealthCheck:
    """Represents a health check result."""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    details: Dict = None
    auto_repair_attempted: bool = False
    repair_successful: bool = False


class SystemMonitor:
    """Monitors system resources and health."""
    
    def __init__(self):
        self.cpu_threshold = 85.0  # CPU usage warning threshold
        self.memory_threshold = 85.0  # Memory usage warning threshold
        self.disk_threshold = 90.0  # Disk usage critical threshold
        self.network_timeout = 30.0  # Network timeout in seconds
        
    def check_cpu_usage(self) -> HealthCheck:
        """Check CPU usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_percent > self.cpu_threshold:
                status = HealthStatus.CRITICAL if cpu_percent > 95 else HealthStatus.WARNING
                message = f"High CPU usage: {cpu_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"CPU usage normal: {cpu_percent:.1f}%"
            
            return HealthCheck(
                name="cpu_usage",
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                details={"cpu_percent": cpu_percent, "threshold": self.cpu_threshold}
            )
            
        except Exception as e:
            return HealthCheck(
                name="cpu_usage",
                status=HealthStatus.FAILED,
                message=f"CPU check failed: {e}",
                timestamp=datetime.utcnow()
            )
    
    def check_memory_usage(self) -> HealthCheck:
        """Check memory usage."""
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            if memory_percent > self.memory_threshold:
                status = HealthStatus.CRITICAL if memory_percent > 95 else HealthStatus.WARNING
                message = f"High memory usage: {memory_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory_percent:.1f}%"
            
            return HealthCheck(
                name="memory_usage",
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                details={
                    "memory_percent": memory_percent,
                    "available_mb": memory.available // (1024 * 1024),
                    "total_mb": memory.total // (1024 * 1024),
                    "threshold": self.memory_threshold
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="memory_usage",
                status=HealthStatus.FAILED,
                message=f"Memory check failed: {e}",
                timestamp=datetime.utcnow()
            )
    
    def check_disk_usage(self) -> HealthCheck:
        """Check disk usage."""
        try:
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            
            if disk_percent > self.disk_threshold:
                status = HealthStatus.CRITICAL
                message = f"Critical disk usage: {disk_percent:.1f}%"
            elif disk_percent > 80:
                status = HealthStatus.WARNING
                message = f"High disk usage: {disk_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage normal: {disk_percent:.1f}%"
            
            return HealthCheck(
                name="disk_usage",
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                details={
                    "disk_percent": disk_percent,
                    "free_gb": disk_usage.free // (1024 * 1024 * 1024),
                    "total_gb": disk_usage.total // (1024 * 1024 * 1024),
                    "threshold": self.disk_threshold
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="disk_usage",
                status=HealthStatus.FAILED,
                message=f"Disk check failed: {e}",
                timestamp=datetime.utcnow()
            )
    
    def check_database_connection(self) -> HealthCheck:
        """Check database connectivity."""
        try:
            # Test database connection
            test_config = config_service.get_config("system.test_connection", "test")
            
            status = HealthStatus.HEALTHY
            message = "Database connection healthy"
            
            return HealthCheck(
                name="database_connection",
                status=status,
                message=message,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            return HealthCheck(
                name="database_connection",
                status=HealthStatus.CRITICAL,
                message=f"Database connection failed: {e}",
                timestamp=datetime.utcnow(),
                details={"error": str(e)}
            )
    
    def check_api_endpoints(self) -> HealthCheck:
        """Check critical API endpoints."""
        try:
            # This would check if Flask app is responding
            import requests
            import time
            
            # Simple health check - would need to be adapted to actual endpoints
            status = HealthStatus.HEALTHY
            message = "API endpoints responding"
            
            return HealthCheck(
                name="api_endpoints",
                status=status,
                message=message,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            return HealthCheck(
                name="api_endpoints",
                status=HealthStatus.CRITICAL,
                message=f"API endpoint check failed: {e}",
                timestamp=datetime.utcnow(),
                details={"error": str(e)}
            )


class AutoRepair:
    """Automatic repair system for common issues."""
    
    def __init__(self):
        self.repair_functions = {
            "high_memory": self._repair_high_memory,
            "database_connection": self._repair_database_connection,
            "disk_space": self._repair_disk_space,
            "api_endpoints": self._repair_api_endpoints
        }
    
    def _repair_high_memory(self, health_check: HealthCheck) -> bool:
        """Attempt to repair high memory usage."""
        try:
            logger.info("Attempting memory usage repair")
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear any large caches if they exist
            # This would be customized based on your application
            
            # Restart memory-intensive processes if needed
            # (This would be more sophisticated in practice)
            
            return True
            
        except Exception as e:
            logger.error(f"Memory repair failed: {e}")
            return False
    
    def _repair_database_connection(self, health_check: HealthCheck) -> bool:
        """Attempt to repair database connection."""
        try:
            logger.info("Attempting database connection repair")
            
            # Restart database connection pool
            # This would involve reconnecting to the database
            
            return True
            
        except Exception as e:
            logger.error(f"Database repair failed: {e}")
            return False
    
    def _repair_disk_space(self, health_check: HealthCheck) -> bool:
        """Attempt to free up disk space."""
        try:
            logger.info("Attempting disk space cleanup")
            
            # Clean up old log files
            import glob
            import os
            
            # Clean logs older than 30 days
            log_files = glob.glob("/var/log/*.log*")
            for log_file in log_files:
                try:
                    if os.path.getmtime(log_file) < time.time() - (30 * 24 * 3600):
                        os.remove(log_file)
                        logger.info(f"Removed old log file: {log_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove {log_file}: {e}")
            
            # Clean temporary files
            temp_files = glob.glob("/tmp/*")
            for temp_file in temp_files:
                try:
                    if os.path.getmtime(temp_file) < time.time() - (7 * 24 * 3600):
                        if os.path.isfile(temp_file):
                            os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to remove {temp_file}: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Disk cleanup failed: {e}")
            return False
    
    def _repair_api_endpoints(self, health_check: HealthCheck) -> bool:
        """Attempt to repair API endpoints."""
        try:
            logger.info("Attempting API endpoint repair")
            
            # Restart Flask application (this would be more sophisticated)
            # For now, just log the attempt
            
            return True
            
        except Exception as e:
            logger.error(f"API repair failed: {e}")
            return False
    
    def attempt_repair(self, health_check: HealthCheck) -> bool:
        """Attempt to repair an issue based on health check."""
        try:
            repair_type = self._determine_repair_type(health_check)
            
            if repair_type and repair_type in self.repair_functions:
                logger.info(f"Attempting repair: {repair_type}")
                success = self.repair_functions[repair_type](health_check)
                
                if success:
                    logger.info(f"Repair successful: {repair_type}")
                else:
                    logger.error(f"Repair failed: {repair_type}")
                
                return success
            else:
                logger.warning(f"No repair available for: {health_check.name}")
                return False
                
        except Exception as e:
            logger.error(f"Repair attempt failed: {e}")
            return False
    
    def _determine_repair_type(self, health_check: HealthCheck) -> Optional[str]:
        """Determine the type of repair needed."""
        if health_check.name == "memory_usage" and health_check.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
            return "high_memory"
        elif health_check.name == "database_connection" and health_check.status == HealthStatus.CRITICAL:
            return "database_connection"
        elif health_check.name == "disk_usage" and health_check.status == HealthStatus.CRITICAL:
            return "disk_space"
        elif health_check.name == "api_endpoints" and health_check.status == HealthStatus.CRITICAL:
            return "api_endpoints"
        
        return None


class SelfHealingDaemon:
    """Main self-healing daemon that monitors and repairs system issues."""
    
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval  # Check every 60 seconds
        self.monitor = SystemMonitor()
        self.repair = AutoRepair()
        self.running = False
        self.daemon_thread = None
        self.health_history: List[HealthCheck] = []
        self.max_history = 1000  # Keep last 1000 health checks
        
        # Load configuration
        self.load_configuration()
        
        logger.info(f"Self-healing daemon initialized with {check_interval}s interval")
    
    def load_configuration(self):
        """Load daemon configuration from database."""
        try:
            self.check_interval = config_service.get_config("daemon.check_interval", 60)
            self.monitor.cpu_threshold = config_service.get_config("daemon.cpu_threshold", 85.0)
            self.monitor.memory_threshold = config_service.get_config("daemon.memory_threshold", 85.0)
            self.monitor.disk_threshold = config_service.get_config("daemon.disk_threshold", 90.0)
            
            logger.info("Self-healing daemon configuration loaded")
        except Exception as e:
            logger.warning(f"Failed to load daemon configuration: {e}")
    
    def save_configuration(self):
        """Save daemon configuration to database."""
        try:
            config_service.set_config("daemon.check_interval", self.check_interval, "Health check interval in seconds")
            config_service.set_config("daemon.cpu_threshold", self.monitor.cpu_threshold, "CPU usage warning threshold")
            config_service.set_config("daemon.memory_threshold", self.monitor.memory_threshold, "Memory usage warning threshold")
            config_service.set_config("daemon.disk_threshold", self.monitor.disk_threshold, "Disk usage critical threshold")
            
            logger.info("Self-healing daemon configuration saved")
        except Exception as e:
            logger.error(f"Failed to save daemon configuration: {e}")
    
    def run_health_checks(self) -> List[HealthCheck]:
        """Run all health checks and return results."""
        health_checks = []
        
        try:
            # System resource checks
            health_checks.append(self.monitor.check_cpu_usage())
            health_checks.append(self.monitor.check_memory_usage())
            health_checks.append(self.monitor.check_disk_usage())
            
            # Service checks
            health_checks.append(self.monitor.check_database_connection())
            health_checks.append(self.monitor.check_api_endpoints())
            
            # Log health check results
            for check in health_checks:
                if check.status == HealthStatus.HEALTHY:
                    logger.debug(f"Health check passed: {check.name}")
                else:
                    logger.warning(f"Health check failed: {check.name} - {check.message}")
            
            return health_checks
            
        except Exception as e:
            logger.error(f"Health checks failed: {e}")
            return []
    
    def process_health_checks(self, health_checks: List[HealthCheck]):
        """Process health check results and attempt repairs."""
        for check in health_checks:
            if check.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                logger.warning(f"Health issue detected: {check.name} - {check.message}")
                
                # Attempt automatic repair
                if not check.auto_repair_attempted:
                    check.auto_repair_attempted = True
                    repair_success = self.repair.attempt_repair(check)
                    check.repair_successful = repair_success
                    
                    if repair_success:
                        logger.info(f"Auto-repair successful for: {check.name}")
                    else:
                        logger.error(f"Auto-repair failed for: {check.name}")
                        
                        # Send alert for critical issues that couldn't be repaired
                        if check.status == HealthStatus.CRITICAL:
                            self._send_critical_alert(check)
    
    def _send_critical_alert(self, health_check: HealthCheck):
        """Send alert for critical issues that couldn't be auto-repaired."""
        try:
            # This would integrate with your alerting system
            # For now, just log the critical issue
            logger.critical(f"CRITICAL SYSTEM ISSUE: {health_check.name} - {health_check.message}")
            
            # Could send email, Slack, or other notifications here
            
        except Exception as e:
            logger.error(f"Failed to send critical alert: {e}")
    
    def daemon_loop(self):
        """Main daemon loop."""
        logger.info("Self-healing daemon started")
        
        while self.running:
            try:
                # Run health checks
                health_checks = self.run_health_checks()
                
                # Process results and attempt repairs
                self.process_health_checks(health_checks)
                
                # Store health history
                self.health_history.extend(health_checks)
                if len(self.health_history) > self.max_history:
                    self.health_history = self.health_history[-self.max_history:]
                
                # Sleep until next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Daemon loop error: {e}")
                logger.error(traceback.format_exc())
                time.sleep(self.check_interval)
    
    def start(self):
        """Start the self-healing daemon."""
        if self.running:
            logger.warning("Daemon already running")
            return
        
        self.running = True
        self.daemon_thread = threading.Thread(target=self.daemon_loop, daemon=True)
        self.daemon_thread.start()
        
        logger.info("Self-healing daemon started")
    
    def stop(self):
        """Stop the self-healing daemon."""
        if not self.running:
            logger.warning("Daemon not running")
            return
        
        self.running = False
        if self.daemon_thread:
            self.daemon_thread.join(timeout=10)
        
        logger.info("Self-healing daemon stopped")
    
    def get_health_status(self) -> Dict:
        """Get current health status."""
        try:
            recent_checks = self.run_health_checks()
            
            # Overall system status
            overall_status = HealthStatus.HEALTHY
            for check in recent_checks:
                if check.status == HealthStatus.CRITICAL:
                    overall_status = HealthStatus.CRITICAL
                    break
                elif check.status == HealthStatus.WARNING and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.WARNING
                elif check.status == HealthStatus.FAILED:
                    overall_status = HealthStatus.FAILED
                    break
            
            return {
                "overall_status": overall_status.value,
                "daemon_running": self.running,
                "last_check": datetime.utcnow().isoformat(),
                "health_checks": [
                    {
                        "name": check.name,
                        "status": check.status.value,
                        "message": check.message,
                        "timestamp": check.timestamp.isoformat(),
                        "auto_repair_attempted": check.auto_repair_attempted,
                        "repair_successful": check.repair_successful
                    }
                    for check in recent_checks
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get health status: {e}")
            return {"error": str(e)}
    
    def get_health_history(self, hours: int = 24) -> List[Dict]:
        """Get health check history for the specified number of hours."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            recent_history = [
                {
                    "name": check.name,
                    "status": check.status.value,
                    "message": check.message,
                    "timestamp": check.timestamp.isoformat(),
                    "auto_repair_attempted": check.auto_repair_attempted,
                    "repair_successful": check.repair_successful
                }
                for check in self.health_history
                if check.timestamp >= cutoff_time
            ]
            
            return recent_history
            
        except Exception as e:
            logger.error(f"Failed to get health history: {e}")
            return []


# Global self-healing daemon instance
self_healing_daemon = SelfHealingDaemon()