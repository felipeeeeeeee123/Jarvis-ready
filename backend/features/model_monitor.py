"""Model version monitoring and automatic updates for JARVIS v3.0."""

import os
import hashlib
import subprocess
import requests
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import threading
import time

from database.services import config_service, memory_service
from utils.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class ModelStatus(Enum):
    """Model status classifications."""
    ACTIVE = "active"
    UPDATING = "updating"
    FAILED = "failed"
    OUTDATED = "outdated"
    UNAVAILABLE = "unavailable"


@dataclass
class ModelVersion:
    """Represents a model version."""
    name: str
    version: str
    hash: str
    size_mb: float
    last_updated: datetime
    status: ModelStatus
    performance_score: float = 0.0
    error_message: Optional[str] = None


@dataclass
class ModelUpdate:
    """Represents a model update operation."""
    model_name: str
    old_version: str
    new_version: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    success: bool = False
    error_message: Optional[str] = None


class OllamaMonitor:
    """Monitor and manage Ollama models."""
    
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.models: Dict[str, ModelVersion] = {}
        self.update_history: List[ModelUpdate] = []
        self.check_interval = 300  # Check every 5 minutes
        
    def check_ollama_availability(self) -> bool:
        """Check if Ollama is available."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return False
    
    def get_installed_models(self) -> List[Dict]:
        """Get list of installed Ollama models."""
        try:
            if not self.check_ollama_availability():
                return []
            
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=30)
            if response.status_code == 200:
                data = response.json()
                return data.get("models", [])
            else:
                logger.error(f"Failed to get models: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting installed models: {e}")
            return []
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get detailed information about a specific model."""
        try:
            if not self.check_ollama_availability():
                return None
            
            response = requests.post(
                f"{self.ollama_url}/api/show",
                json={"name": model_name},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get model info for {model_name}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting model info for {model_name}: {e}")
            return None
    
    def update_model_registry(self):
        """Update the internal model registry."""
        try:
            installed_models = self.get_installed_models()
            
            for model_data in installed_models:
                model_name = model_data.get("name", "")
                model_size = model_data.get("size", 0) / (1024 * 1024)  # Convert to MB
                modified_at = model_data.get("modified_at", "")
                
                # Parse modified_at timestamp
                try:
                    last_updated = datetime.fromisoformat(modified_at.replace('Z', '+00:00'))
                except:
                    last_updated = datetime.utcnow()
                
                # Calculate hash for version tracking
                model_hash = hashlib.md5(f"{model_name}_{modified_at}".encode()).hexdigest()
                
                # Get performance score from database
                performance_score = config_service.get_config(f"model.{model_name}.performance", 0.0)
                
                self.models[model_name] = ModelVersion(
                    name=model_name,
                    version=modified_at,
                    hash=model_hash,
                    size_mb=model_size,
                    last_updated=last_updated,
                    status=ModelStatus.ACTIVE,
                    performance_score=performance_score
                )
            
            logger.info(f"Updated model registry with {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Failed to update model registry: {e}")
    
    def check_for_updates(self, model_name: str) -> bool:
        """Check if a model has updates available."""
        try:
            # This would check against a model repository for updates
            # For now, simulate checking for updates
            
            # Check if model was updated more than 7 days ago
            if model_name in self.models:
                model = self.models[model_name]
                days_old = (datetime.utcnow() - model.last_updated).days
                
                # Consider model outdated if older than 7 days and performance is low
                if days_old > 7 and model.performance_score < 0.7:
                    logger.info(f"Model {model_name} may need updating (age: {days_old} days, performance: {model.performance_score:.2f})")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking for updates for {model_name}: {e}")
            return False
    
    def pull_model_update(self, model_name: str) -> bool:
        """Pull an updated version of a model."""
        try:
            logger.info(f"Pulling update for model: {model_name}")
            
            # Start update tracking
            update = ModelUpdate(
                model_name=model_name,
                old_version=self.models.get(model_name, ModelVersion("", "", "", 0, datetime.utcnow(), ModelStatus.UNAVAILABLE)).version,
                new_version="updating",
                started_at=datetime.utcnow()
            )
            self.update_history.append(update)
            
            # Set model status to updating
            if model_name in self.models:
                self.models[model_name].status = ModelStatus.UPDATING
            
            # Pull the model using Ollama API
            response = requests.post(
                f"{self.ollama_url}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=3600  # 1 hour timeout for large models
            )
            
            if response.status_code == 200:
                # Monitor pull progress
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if data.get("status") == "success":
                                break
                        except json.JSONDecodeError:
                            continue
                
                # Update completed successfully
                update.completed_at = datetime.utcnow()
                update.success = True
                
                # Update model registry
                self.update_model_registry()
                
                logger.info(f"Successfully updated model: {model_name}")
                return True
            else:
                update.completed_at = datetime.utcnow()
                update.error_message = f"Pull failed with status {response.status_code}"
                
                if model_name in self.models:
                    self.models[model_name].status = ModelStatus.FAILED
                    self.models[model_name].error_message = update.error_message
                
                logger.error(f"Failed to pull model {model_name}: {response.status_code}")
                return False
                
        except Exception as e:
            # Update failed
            if self.update_history:
                self.update_history[-1].completed_at = datetime.utcnow()
                self.update_history[-1].error_message = str(e)
            
            if model_name in self.models:
                self.models[model_name].status = ModelStatus.FAILED
                self.models[model_name].error_message = str(e)
            
            logger.error(f"Error pulling model update for {model_name}: {e}")
            return False
    
    def get_model_performance(self, model_name: str) -> float:
        """Get model performance score."""
        try:
            # Get performance metrics from memory/database
            performance_data = memory_service.get_memories_by_tag(f"model_performance_{model_name}")
            
            if performance_data:
                # Calculate average performance from recent interactions
                scores = []
                for memory in performance_data[-10:]:  # Last 10 interactions
                    if "performance_score" in memory.metadata:
                        scores.append(memory.metadata["performance_score"])
                
                if scores:
                    return sum(scores) / len(scores)
            
            return 0.5  # Default neutral score
            
        except Exception as e:
            logger.error(f"Error getting model performance for {model_name}: {e}")
            return 0.5


class ModelMonitor:
    """Main model monitoring and update system."""
    
    def __init__(self):
        self.ollama_monitor = OllamaMonitor()
        self.monitoring_enabled = True
        self.auto_update_enabled = True
        self.monitor_thread = None
        self.running = False
        
        # Configuration
        self.check_interval = 300  # Check every 5 minutes
        self.performance_threshold = 0.6  # Update if performance drops below 60%
        self.max_model_age_days = 14  # Update models older than 14 days
        
        self.load_configuration()
        logger.info("Model monitor initialized")
    
    def load_configuration(self):
        """Load configuration from database."""
        try:
            self.monitoring_enabled = config_service.get_config("model_monitor.enabled", True)
            self.auto_update_enabled = config_service.get_config("model_monitor.auto_update", True)
            self.check_interval = config_service.get_config("model_monitor.check_interval", 300)
            self.performance_threshold = config_service.get_config("model_monitor.performance_threshold", 0.6)
            self.max_model_age_days = config_service.get_config("model_monitor.max_age_days", 14)
            
            logger.info("Model monitor configuration loaded")
        except Exception as e:
            logger.warning(f"Failed to load model monitor configuration: {e}")
    
    def save_configuration(self):
        """Save configuration to database."""
        try:
            config_service.set_config("model_monitor.enabled", self.monitoring_enabled, "Enable model monitoring")
            config_service.set_config("model_monitor.auto_update", self.auto_update_enabled, "Enable automatic model updates")
            config_service.set_config("model_monitor.check_interval", self.check_interval, "Model check interval in seconds")
            config_service.set_config("model_monitor.performance_threshold", self.performance_threshold, "Performance threshold for updates")
            config_service.set_config("model_monitor.max_age_days", self.max_model_age_days, "Maximum model age in days")
            
            logger.info("Model monitor configuration saved")
        except Exception as e:
            logger.error(f"Failed to save model monitor configuration: {e}")
    
    def evaluate_model_health(self, model_name: str) -> Dict[str, Any]:
        """Evaluate overall health of a model."""
        try:
            if model_name not in self.ollama_monitor.models:
                return {
                    "status": "not_found",
                    "health_score": 0.0,
                    "recommendations": ["Model not found in registry"]
                }
            
            model = self.ollama_monitor.models[model_name]
            recommendations = []
            
            # Calculate health score
            health_score = 1.0
            
            # Age factor
            age_days = (datetime.utcnow() - model.last_updated).days
            if age_days > self.max_model_age_days:
                age_penalty = min(0.5, (age_days - self.max_model_age_days) / 30)
                health_score -= age_penalty
                recommendations.append(f"Model is {age_days} days old, consider updating")
            
            # Performance factor
            performance_score = self.ollama_monitor.get_model_performance(model_name)
            if performance_score < self.performance_threshold:
                performance_penalty = (self.performance_threshold - performance_score) * 0.5
                health_score -= performance_penalty
                recommendations.append(f"Performance below threshold ({performance_score:.2f})")
            
            # Status factor
            if model.status != ModelStatus.ACTIVE:
                health_score -= 0.3
                recommendations.append(f"Model status: {model.status.value}")
            
            # Determine overall status
            if health_score >= 0.8:
                status = "excellent"
            elif health_score >= 0.6:
                status = "good"
            elif health_score >= 0.4:
                status = "fair"
            else:
                status = "poor"
                recommendations.append("Consider immediate update or replacement")
            
            return {
                "status": status,
                "health_score": max(0.0, health_score),
                "age_days": age_days,
                "performance_score": performance_score,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model health for {model_name}: {e}")
            return {
                "status": "error",
                "health_score": 0.0,
                "recommendations": [f"Error evaluating model: {e}"]
            }
    
    def should_update_model(self, model_name: str) -> Tuple[bool, str]:
        """Determine if a model should be updated."""
        try:
            health_data = self.evaluate_model_health(model_name)
            
            # Update if health score is low
            if health_data["health_score"] < 0.5:
                return True, "Low health score"
            
            # Update if performance is below threshold
            if health_data["performance_score"] < self.performance_threshold:
                return True, "Performance below threshold"
            
            # Update if model is too old
            if health_data["age_days"] > self.max_model_age_days:
                return True, "Model too old"
            
            # Check for available updates
            if self.ollama_monitor.check_for_updates(model_name):
                return True, "Updates available"
            
            return False, "Model is healthy"
            
        except Exception as e:
            logger.error(f"Error determining if model should update: {e}")
            return False, f"Error: {e}"
    
    def monitor_loop(self):
        """Main monitoring loop."""
        logger.info("Model monitor started")
        
        while self.running:
            try:
                if not self.monitoring_enabled:
                    time.sleep(self.check_interval)
                    continue
                
                # Update model registry
                self.ollama_monitor.update_model_registry()
                
                # Check each model
                for model_name in self.ollama_monitor.models:
                    try:
                        should_update, reason = self.should_update_model(model_name)
                        
                        if should_update:
                            logger.info(f"Model {model_name} needs update: {reason}")
                            
                            if self.auto_update_enabled:
                                logger.info(f"Auto-updating model: {model_name}")
                                success = self.ollama_monitor.pull_model_update(model_name)
                                
                                if success:
                                    logger.info(f"Successfully updated model: {model_name}")
                                else:
                                    logger.error(f"Failed to update model: {model_name}")
                            else:
                                logger.info(f"Auto-update disabled, skipping update for: {model_name}")
                        
                    except Exception as e:
                        logger.error(f"Error monitoring model {model_name}: {e}")
                
                # Sleep until next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Model monitor loop error: {e}")
                time.sleep(self.check_interval)
    
    def start(self):
        """Start the model monitor."""
        if self.running:
            logger.warning("Model monitor already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Model monitor started")
    
    def stop(self):
        """Stop the model monitor."""
        if not self.running:
            logger.warning("Model monitor not running")
            return
        
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        
        logger.info("Model monitor stopped")
    
    def get_status(self) -> Dict:
        """Get comprehensive model monitoring status."""
        try:
            self.ollama_monitor.update_model_registry()
            
            models_status = {}
            for model_name, model in self.ollama_monitor.models.items():
                health_data = self.evaluate_model_health(model_name)
                
                models_status[model_name] = {
                    "version": model.version,
                    "size_mb": model.size_mb,
                    "last_updated": model.last_updated.isoformat(),
                    "status": model.status.value,
                    "health_score": health_data["health_score"],
                    "performance_score": health_data["performance_score"],
                    "age_days": health_data["age_days"],
                    "recommendations": health_data["recommendations"]
                }
            
            return {
                "monitor_running": self.running,
                "monitoring_enabled": self.monitoring_enabled,
                "auto_update_enabled": self.auto_update_enabled,
                "ollama_available": self.ollama_monitor.check_ollama_availability(),
                "models": models_status,
                "recent_updates": [
                    {
                        "model_name": update.model_name,
                        "old_version": update.old_version,
                        "new_version": update.new_version,
                        "started_at": update.started_at.isoformat(),
                        "completed_at": update.completed_at.isoformat() if update.completed_at else None,
                        "success": update.success,
                        "error_message": update.error_message
                    }
                    for update in self.ollama_monitor.update_history[-5:]  # Last 5 updates
                ],
                "last_check": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting model monitor status: {e}")
            return {"error": str(e)}
    
    def force_update(self, model_name: str) -> bool:
        """Force update a specific model."""
        try:
            logger.info(f"Forcing update for model: {model_name}")
            return self.ollama_monitor.pull_model_update(model_name)
        except Exception as e:
            logger.error(f"Error forcing update for {model_name}: {e}")
            return False
    
    def update_model_performance(self, model_name: str, performance_score: float):
        """Update model performance score."""
        try:
            if model_name in self.ollama_monitor.models:
                self.ollama_monitor.models[model_name].performance_score = performance_score
                
                # Store in database
                config_service.set_config(
                    f"model.{model_name}.performance", 
                    performance_score, 
                    f"Performance score for {model_name}"
                )
                
                logger.info(f"Updated performance score for {model_name}: {performance_score}")
                
        except Exception as e:
            logger.error(f"Error updating model performance: {e}")


# Global model monitor instance
model_monitor = ModelMonitor()