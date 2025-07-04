"""Dynamic plugin management and upgrade system for JARVIS v3.0."""

import os
import sys
import importlib
import inspect
import hashlib
import shutil
import requests
import zipfile
from typing import Dict, List, Optional, Any, Callable, Type
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
import time
from pathlib import Path

from database.services import config_service, memory_service
from utils.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class PluginStatus(Enum):
    """Plugin status classifications."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    LOADING = "loading"
    ERROR = "error"
    UPDATING = "updating"
    OUTDATED = "outdated"


@dataclass
class PluginInfo:
    """Information about a plugin."""
    name: str
    version: str
    description: str
    author: str
    file_path: str
    class_name: str
    status: PluginStatus
    loaded_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    error_message: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    performance_score: float = 1.0


@dataclass
class PluginUpdate:
    """Information about a plugin update."""
    plugin_name: str
    current_version: str
    new_version: str
    download_url: str
    changelog: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    success: bool = False
    error_message: Optional[str] = None


class BasePlugin:
    """Base class for all JARVIS plugins."""
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.version = "1.0.0"
        self.description = "Base plugin"
        self.author = "JARVIS"
        self.enabled = True
        self.config = {}
        
    def initialize(self) -> bool:
        """Initialize the plugin. Return True if successful."""
        return True
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute the plugin's main functionality."""
        raise NotImplementedError("Plugin must implement execute method")
    
    def shutdown(self):
        """Clean shutdown of the plugin."""
        pass
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Return configuration schema for the plugin."""
        return {}
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate plugin configuration."""
        return True
    
    def get_health_status(self) -> Dict[str, Any]:
        """Return plugin health status."""
        return {
            "status": "healthy",
            "last_executed": None,
            "error_count": 0
        }


class PluginLoader:
    """Handles loading and managing individual plugins."""
    
    def __init__(self, plugin_directory: str):
        self.plugin_directory = Path(plugin_directory)
        self.loaded_plugins: Dict[str, Any] = {}
        self.plugin_info: Dict[str, PluginInfo] = {}
        
        # Ensure plugin directory exists
        self.plugin_directory.mkdir(exist_ok=True)
        
        # Add plugin directory to Python path
        if str(self.plugin_directory) not in sys.path:
            sys.path.insert(0, str(self.plugin_directory))
    
    def discover_plugins(self) -> List[str]:
        """Discover available plugin files."""
        try:
            plugin_files = []
            
            # Look for Python files in plugin directory
            for file_path in self.plugin_directory.glob("*.py"):
                if file_path.name.startswith("__"):
                    continue
                plugin_files.append(file_path.stem)
            
            # Look for plugin packages
            for dir_path in self.plugin_directory.iterdir():
                if dir_path.is_dir() and not dir_path.name.startswith("__"):
                    init_file = dir_path / "__init__.py"
                    if init_file.exists():
                        plugin_files.append(dir_path.name)
            
            logger.info(f"Discovered {len(plugin_files)} plugins: {plugin_files}")
            return plugin_files
            
        except Exception as e:
            logger.error(f"Error discovering plugins: {e}")
            return []
    
    def load_plugin(self, plugin_name: str) -> bool:
        """Load a specific plugin."""
        try:
            logger.info(f"Loading plugin: {plugin_name}")
            
            # Create plugin info entry
            plugin_info = PluginInfo(
                name=plugin_name,
                version="unknown",
                description="",
                author="",
                file_path=str(self.plugin_directory / f"{plugin_name}.py"),
                class_name="",
                status=PluginStatus.LOADING
            )
            
            # Import the plugin module
            try:
                # Remove from cache if already loaded
                module_name = f"plugins.{plugin_name}" if "plugins" in sys.modules else plugin_name
                if module_name in sys.modules:
                    importlib.reload(sys.modules[module_name])
                    module = sys.modules[module_name]
                else:
                    module = importlib.import_module(plugin_name)
            except ImportError as e:
                logger.error(f"Failed to import plugin {plugin_name}: {e}")
                plugin_info.status = PluginStatus.ERROR
                plugin_info.error_message = str(e)
                self.plugin_info[plugin_name] = plugin_info
                return False
            
            # Find plugin class (should inherit from BasePlugin)
            plugin_class = None
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BasePlugin) and obj != BasePlugin:
                    plugin_class = obj
                    plugin_info.class_name = name
                    break
            
            if not plugin_class:
                error_msg = f"No valid plugin class found in {plugin_name}"
                logger.error(error_msg)
                plugin_info.status = PluginStatus.ERROR
                plugin_info.error_message = error_msg
                self.plugin_info[plugin_name] = plugin_info
                return False
            
            # Instantiate the plugin
            try:
                plugin_instance = plugin_class()
                
                # Update plugin info
                plugin_info.version = getattr(plugin_instance, "version", "1.0.0")
                plugin_info.description = getattr(plugin_instance, "description", "")
                plugin_info.author = getattr(plugin_instance, "author", "")
                plugin_info.dependencies = getattr(plugin_instance, "dependencies", [])
                
                # Initialize the plugin
                if plugin_instance.initialize():
                    self.loaded_plugins[plugin_name] = plugin_instance
                    plugin_info.status = PluginStatus.ACTIVE
                    plugin_info.loaded_at = datetime.utcnow()
                    
                    logger.info(f"Successfully loaded plugin: {plugin_name} v{plugin_info.version}")
                else:
                    plugin_info.status = PluginStatus.ERROR
                    plugin_info.error_message = "Plugin initialization failed"
                    logger.error(f"Plugin initialization failed: {plugin_name}")
                    
            except Exception as e:
                plugin_info.status = PluginStatus.ERROR
                plugin_info.error_message = str(e)
                logger.error(f"Error instantiating plugin {plugin_name}: {e}")
                
            self.plugin_info[plugin_name] = plugin_info
            return plugin_info.status == PluginStatus.ACTIVE
            
        except Exception as e:
            logger.error(f"Error loading plugin {plugin_name}: {e}")
            return False
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a specific plugin."""
        try:
            if plugin_name in self.loaded_plugins:
                plugin = self.loaded_plugins[plugin_name]
                
                # Call shutdown method
                try:
                    plugin.shutdown()
                except Exception as e:
                    logger.warning(f"Error during plugin shutdown {plugin_name}: {e}")
                
                # Remove from loaded plugins
                del self.loaded_plugins[plugin_name]
                
                # Update status
                if plugin_name in self.plugin_info:
                    self.plugin_info[plugin_name].status = PluginStatus.INACTIVE
                
                logger.info(f"Unloaded plugin: {plugin_name}")
                return True
            else:
                logger.warning(f"Plugin not loaded: {plugin_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_name}: {e}")
            return False
    
    def execute_plugin(self, plugin_name: str, *args, **kwargs) -> Any:
        """Execute a plugin's main functionality."""
        try:
            if plugin_name not in self.loaded_plugins:
                raise ValueError(f"Plugin not loaded: {plugin_name}")
            
            plugin = self.loaded_plugins[plugin_name]
            
            # Update last used timestamp
            if plugin_name in self.plugin_info:
                self.plugin_info[plugin_name].last_used = datetime.utcnow()
            
            result = plugin.execute(*args, **kwargs)
            
            logger.debug(f"Executed plugin: {plugin_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error executing plugin {plugin_name}: {e}")
            raise
    
    def get_plugin_health(self, plugin_name: str) -> Dict[str, Any]:
        """Get health status of a plugin."""
        try:
            if plugin_name not in self.loaded_plugins:
                return {"status": "not_loaded"}
            
            plugin = self.loaded_plugins[plugin_name]
            return plugin.get_health_status()
            
        except Exception as e:
            logger.error(f"Error getting plugin health {plugin_name}: {e}")
            return {"status": "error", "error": str(e)}


class PluginRegistry:
    """Manages plugin discovery and updates from remote sources."""
    
    def __init__(self):
        self.registry_url = "https://api.github.com/repos/jarvis-plugins/registry/contents/plugins"
        self.available_plugins: Dict[str, Dict] = {}
        self.update_cache_duration = 3600  # 1 hour
        self.last_update_check = None
    
    def fetch_available_plugins(self) -> Dict[str, Dict]:
        """Fetch available plugins from remote registry."""
        try:
            logger.info("Fetching available plugins from registry")
            
            # This would fetch from a real plugin registry
            # For now, return a mock registry
            mock_plugins = {
                "sentiment_analyzer": {
                    "name": "sentiment_analyzer",
                    "version": "1.2.0",
                    "description": "Advanced sentiment analysis for trading signals",
                    "author": "JARVIS Team",
                    "download_url": "https://example.com/plugins/sentiment_analyzer.zip",
                    "dependencies": ["textblob", "vaderSentiment"],
                    "changelog": "Added support for multiple languages"
                },
                "news_aggregator": {
                    "name": "news_aggregator",
                    "version": "2.1.0",
                    "description": "Aggregates news from multiple sources",
                    "author": "JARVIS Team",
                    "download_url": "https://example.com/plugins/news_aggregator.zip",
                    "dependencies": ["feedparser", "newspaper3k"],
                    "changelog": "Improved RSS feed parsing"
                },
                "technical_indicators": {
                    "name": "technical_indicators",
                    "version": "1.5.0",
                    "description": "Additional technical indicators for trading",
                    "author": "JARVIS Team",
                    "download_url": "https://example.com/plugins/technical_indicators.zip",
                    "dependencies": ["talib"],
                    "changelog": "Added Bollinger Bands and Stochastic indicators"
                }
            }
            
            self.available_plugins = mock_plugins
            self.last_update_check = datetime.utcnow()
            
            logger.info(f"Found {len(mock_plugins)} available plugins")
            return mock_plugins
            
        except Exception as e:
            logger.error(f"Error fetching available plugins: {e}")
            return {}
    
    def check_for_updates(self, installed_plugins: Dict[str, PluginInfo]) -> List[PluginUpdate]:
        """Check for updates to installed plugins."""
        try:
            updates = []
            
            # Refresh available plugins if needed
            if (not self.last_update_check or 
                datetime.utcnow() - self.last_update_check > timedelta(seconds=self.update_cache_duration)):
                self.fetch_available_plugins()
            
            for plugin_name, plugin_info in installed_plugins.items():
                if plugin_name in self.available_plugins:
                    available_version = self.available_plugins[plugin_name]["version"]
                    current_version = plugin_info.version
                    
                    # Simple version comparison (would be more sophisticated in practice)
                    if available_version != current_version:
                        update = PluginUpdate(
                            plugin_name=plugin_name,
                            current_version=current_version,
                            new_version=available_version,
                            download_url=self.available_plugins[plugin_name]["download_url"],
                            changelog=self.available_plugins[plugin_name].get("changelog", "")
                        )
                        updates.append(update)
            
            logger.info(f"Found {len(updates)} plugin updates")
            return updates
            
        except Exception as e:
            logger.error(f"Error checking for plugin updates: {e}")
            return []
    
    def download_plugin(self, plugin_name: str, download_url: str, target_directory: Path) -> bool:
        """Download and extract a plugin."""
        try:
            logger.info(f"Downloading plugin: {plugin_name}")
            
            # Download the plugin
            response = requests.get(download_url, timeout=30)
            response.raise_for_status()
            
            # Save to temporary file
            temp_file = target_directory / f"{plugin_name}_temp.zip"
            with open(temp_file, "wb") as f:
                f.write(response.content)
            
            # Extract the plugin
            with zipfile.ZipFile(temp_file, "r") as zip_ref:
                zip_ref.extractall(target_directory)
            
            # Clean up temporary file
            temp_file.unlink()
            
            logger.info(f"Successfully downloaded plugin: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading plugin {plugin_name}: {e}")
            return False


class PluginManager:
    """Main plugin management system."""
    
    def __init__(self, plugin_directory: str = "plugins"):
        self.plugin_directory = Path(plugin_directory)
        self.loader = PluginLoader(self.plugin_directory)
        self.registry = PluginRegistry()
        self.auto_update_enabled = True
        self.update_check_interval = 86400  # Daily
        self.monitor_thread = None
        self.running = False
        
        # Initialize plugin directory structure
        self._setup_plugin_directory()
        
        # Load configuration
        self.load_configuration()
        
        logger.info(f"Plugin manager initialized with directory: {self.plugin_directory}")
    
    def _setup_plugin_directory(self):
        """Setup plugin directory structure."""
        try:
            self.plugin_directory.mkdir(exist_ok=True)
            
            # Create __init__.py for Python package
            init_file = self.plugin_directory / "__init__.py"
            if not init_file.exists():
                init_file.write_text("# JARVIS Plugins Package\n")
            
            logger.info("Plugin directory structure setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up plugin directory: {e}")
    
    def load_configuration(self):
        """Load plugin manager configuration."""
        try:
            self.auto_update_enabled = config_service.get_config("plugins.auto_update", True)
            self.update_check_interval = config_service.get_config("plugins.update_interval", 86400)
            
            logger.info("Plugin manager configuration loaded")
        except Exception as e:
            logger.warning(f"Failed to load plugin manager configuration: {e}")
    
    def save_configuration(self):
        """Save plugin manager configuration."""
        try:
            config_service.set_config("plugins.auto_update", self.auto_update_enabled, "Enable automatic plugin updates")
            config_service.set_config("plugins.update_interval", self.update_check_interval, "Plugin update check interval")
            
            logger.info("Plugin manager configuration saved")
        except Exception as e:
            logger.error(f"Failed to save plugin manager configuration: {e}")
    
    def discover_and_load_all(self) -> Dict[str, bool]:
        """Discover and load all available plugins."""
        try:
            plugin_files = self.loader.discover_plugins()
            results = {}
            
            for plugin_name in plugin_files:
                success = self.loader.load_plugin(plugin_name)
                results[plugin_name] = success
            
            logger.info(f"Loaded {sum(results.values())}/{len(results)} plugins")
            return results
            
        except Exception as e:
            logger.error(f"Error discovering and loading plugins: {e}")
            return {}
    
    def install_plugin(self, plugin_name: str) -> bool:
        """Install a plugin from the registry."""
        try:
            logger.info(f"Installing plugin: {plugin_name}")
            
            # Fetch available plugins
            available = self.registry.fetch_available_plugins()
            
            if plugin_name not in available:
                logger.error(f"Plugin not found in registry: {plugin_name}")
                return False
            
            plugin_data = available[plugin_name]
            
            # Download and install
            success = self.registry.download_plugin(
                plugin_name, 
                plugin_data["download_url"], 
                self.plugin_directory
            )
            
            if success:
                # Load the plugin
                return self.loader.load_plugin(plugin_name)
            
            return False
            
        except Exception as e:
            logger.error(f"Error installing plugin {plugin_name}: {e}")
            return False
    
    def update_plugin(self, plugin_name: str) -> bool:
        """Update an existing plugin."""
        try:
            logger.info(f"Updating plugin: {plugin_name}")
            
            # Unload current version
            self.loader.unload_plugin(plugin_name)
            
            # Install new version (this would be more sophisticated)
            success = self.install_plugin(plugin_name)
            
            if success:
                logger.info(f"Successfully updated plugin: {plugin_name}")
            else:
                logger.error(f"Failed to update plugin: {plugin_name}")
                # Try to reload old version
                self.loader.load_plugin(plugin_name)
            
            return success
            
        except Exception as e:
            logger.error(f"Error updating plugin {plugin_name}: {e}")
            return False
    
    def uninstall_plugin(self, plugin_name: str) -> bool:
        """Uninstall a plugin."""
        try:
            logger.info(f"Uninstalling plugin: {plugin_name}")
            
            # Unload the plugin
            self.loader.unload_plugin(plugin_name)
            
            # Remove plugin files
            plugin_file = self.plugin_directory / f"{plugin_name}.py"
            plugin_dir = self.plugin_directory / plugin_name
            
            if plugin_file.exists():
                plugin_file.unlink()
            
            if plugin_dir.exists():
                shutil.rmtree(plugin_dir)
            
            # Remove from plugin info
            if plugin_name in self.loader.plugin_info:
                del self.loader.plugin_info[plugin_name]
            
            logger.info(f"Successfully uninstalled plugin: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error uninstalling plugin {plugin_name}: {e}")
            return False
    
    def monitor_and_update_loop(self):
        """Monitor plugins and perform automatic updates."""
        logger.info("Plugin monitor started")
        
        while self.running:
            try:
                if not self.auto_update_enabled:
                    time.sleep(self.update_check_interval)
                    continue
                
                logger.info("Checking for plugin updates")
                
                # Check for updates
                updates = self.registry.check_for_updates(self.loader.plugin_info)
                
                for update in updates:
                    try:
                        logger.info(f"Auto-updating plugin: {update.plugin_name}")
                        success = self.update_plugin(update.plugin_name)
                        
                        if success:
                            logger.info(f"Successfully auto-updated: {update.plugin_name}")
                        else:
                            logger.error(f"Failed to auto-update: {update.plugin_name}")
                            
                    except Exception as e:
                        logger.error(f"Error auto-updating plugin {update.plugin_name}: {e}")
                
                time.sleep(self.update_check_interval)
                
            except Exception as e:
                logger.error(f"Plugin monitor loop error: {e}")
                time.sleep(self.update_check_interval)
    
    def start_monitoring(self):
        """Start plugin monitoring and auto-updates."""
        if self.running:
            logger.warning("Plugin monitoring already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self.monitor_and_update_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Plugin monitoring started")
    
    def stop_monitoring(self):
        """Stop plugin monitoring."""
        if not self.running:
            logger.warning("Plugin monitoring not running")
            return
        
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        
        logger.info("Plugin monitoring stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive plugin status."""
        try:
            # Check for available updates
            updates = self.registry.check_for_updates(self.loader.plugin_info)
            
            return {
                "monitoring_enabled": self.running,
                "auto_update_enabled": self.auto_update_enabled,
                "total_plugins": len(self.loader.plugin_info),
                "active_plugins": len([p for p in self.loader.plugin_info.values() if p.status == PluginStatus.ACTIVE]),
                "available_updates": len(updates),
                "plugins": {
                    name: {
                        "version": info.version,
                        "status": info.status.value,
                        "description": info.description,
                        "author": info.author,
                        "loaded_at": info.loaded_at.isoformat() if info.loaded_at else None,
                        "last_used": info.last_used.isoformat() if info.last_used else None,
                        "error_message": info.error_message,
                        "performance_score": info.performance_score
                    }
                    for name, info in self.loader.plugin_info.items()
                },
                "available_updates": [
                    {
                        "plugin_name": update.plugin_name,
                        "current_version": update.current_version,
                        "new_version": update.new_version,
                        "changelog": update.changelog
                    }
                    for update in updates
                ],
                "last_check": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting plugin status: {e}")
            return {"error": str(e)}
    
    def execute_plugin(self, plugin_name: str, *args, **kwargs) -> Any:
        """Execute a plugin."""
        return self.loader.execute_plugin(plugin_name, *args, **kwargs)
    
    def get_plugin_list(self) -> List[str]:
        """Get list of loaded plugin names."""
        return list(self.loader.loaded_plugins.keys())
    
    def is_plugin_loaded(self, plugin_name: str) -> bool:
        """Check if a plugin is loaded."""
        return plugin_name in self.loader.loaded_plugins


# Global plugin manager instance
plugin_manager = PluginManager()