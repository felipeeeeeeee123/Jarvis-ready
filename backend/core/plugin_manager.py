"""Plugin management system for JARVIS v3.0."""

import os
import sys
import importlib
import inspect
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Type, Callable
from pathlib import Path
from utils.logging_config import get_logger

logger = get_logger(__name__)


class JARVISPlugin(ABC):
    """Base class for JARVIS plugins."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Plugin description."""
        pass
    
    @property
    def commands(self) -> Dict[str, Callable]:
        """Commands provided by this plugin."""
        return {}
    
    @property
    def triggers(self) -> List[str]:
        """Keywords that trigger this plugin."""
        return []
    
    def initialize(self) -> bool:
        """Initialize the plugin. Return True if successful."""
        return True
    
    def shutdown(self) -> None:
        """Clean up plugin resources."""
        pass
    
    def can_handle(self, prompt: str) -> bool:
        """Check if this plugin can handle the given prompt."""
        if not self.triggers:
            return False
        return any(trigger.lower() in prompt.lower() for trigger in self.triggers)
    
    @abstractmethod
    def process(self, prompt: str, context: Optional[Dict] = None) -> str:
        """Process the prompt and return a response."""
        pass


class PluginManager:
    """Manages JARVIS plugins."""
    
    def __init__(self, plugins_dir: Optional[str] = None):
        self.plugins_dir = Path(plugins_dir or "plugins")
        self.plugins: Dict[str, JARVISPlugin] = {}
        self.commands: Dict[str, JARVISPlugin] = {}
        self.enabled_plugins: Dict[str, bool] = {}
        logger.info(f"Plugin manager initialized with directory: {self.plugins_dir}")
    
    def discover_plugins(self) -> List[str]:
        """Discover available plugins in the plugins directory."""
        discovered = []
        
        if not self.plugins_dir.exists():
            logger.warning(f"Plugins directory does not exist: {self.plugins_dir}")
            return discovered
        
        # Add plugins directory to Python path
        if str(self.plugins_dir) not in sys.path:
            sys.path.insert(0, str(self.plugins_dir))
        
        for item in self.plugins_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Check for __init__.py or plugin.py
                if (item / "__init__.py").exists() or (item / "plugin.py").exists():
                    discovered.append(item.name)
                    logger.info(f"Discovered plugin directory: {item.name}")
            elif item.is_file() and item.suffix == '.py' and item.name != '__init__.py':
                discovered.append(item.stem)
                logger.info(f"Discovered plugin file: {item.name}")
        
        return discovered
    
    def load_plugin(self, plugin_name: str) -> bool:
        """Load a specific plugin."""
        try:
            # Try importing the plugin
            if (self.plugins_dir / plugin_name / "plugin.py").exists():
                # Plugin in directory with plugin.py
                module_name = f"{plugin_name}.plugin"
            elif (self.plugins_dir / plugin_name / "__init__.py").exists():
                # Plugin in directory with __init__.py
                module_name = plugin_name
            elif (self.plugins_dir / f"{plugin_name}.py").exists():
                # Single file plugin
                module_name = plugin_name
            else:
                logger.error(f"Plugin file not found for: {plugin_name}")
                return False
            
            # Import the module
            try:
                module = importlib.import_module(module_name)
            except ImportError as e:
                # Try reloading if already imported
                if module_name in sys.modules:
                    module = importlib.reload(sys.modules[module_name])
                else:
                    raise e
            
            # Find plugin classes
            plugin_classes = []
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, JARVISPlugin) and 
                    obj != JARVISPlugin):
                    plugin_classes.append(obj)
            
            if not plugin_classes:
                logger.error(f"No plugin classes found in {module_name}")
                return False
            
            # Initialize the first plugin class found
            plugin_class = plugin_classes[0]
            plugin_instance = plugin_class()
            
            # Initialize the plugin
            if not plugin_instance.initialize():
                logger.error(f"Plugin initialization failed: {plugin_name}")
                return False
            
            # Register the plugin
            self.plugins[plugin_name] = plugin_instance
            self.enabled_plugins[plugin_name] = True
            
            # Register commands
            for command_name, command_func in plugin_instance.commands.items():
                self.commands[command_name] = plugin_instance
            
            logger.info(f"Successfully loaded plugin: {plugin_instance.name} v{plugin_instance.version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_name}: {e}")
            return False
    
    def load_all_plugins(self) -> Dict[str, bool]:
        """Load all discovered plugins."""
        discovered = self.discover_plugins()
        results = {}
        
        for plugin_name in discovered:
            results[plugin_name] = self.load_plugin(plugin_name)
        
        loaded_count = sum(results.values())
        logger.info(f"Loaded {loaded_count}/{len(discovered)} plugins successfully")
        return results
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a specific plugin."""
        if plugin_name not in self.plugins:
            logger.warning(f"Plugin not loaded: {plugin_name}")
            return False
        
        try:
            # Clean up plugin
            self.plugins[plugin_name].shutdown()
            
            # Remove commands
            commands_to_remove = []
            for cmd_name, plugin in self.commands.items():
                if plugin == self.plugins[plugin_name]:
                    commands_to_remove.append(cmd_name)
            
            for cmd_name in commands_to_remove:
                del self.commands[cmd_name]
            
            # Remove plugin
            del self.plugins[plugin_name]
            self.enabled_plugins[plugin_name] = False
            
            logger.info(f"Unloaded plugin: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload plugin {plugin_name}: {e}")
            return False
    
    def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a specific plugin."""
        if plugin_name in self.plugins:
            self.unload_plugin(plugin_name)
        return self.load_plugin(plugin_name)
    
    def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a loaded plugin."""
        if plugin_name not in self.plugins:
            logger.warning(f"Plugin not loaded: {plugin_name}")
            return False
        
        self.enabled_plugins[plugin_name] = True
        logger.info(f"Enabled plugin: {plugin_name}")
        return True
    
    def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a loaded plugin."""
        if plugin_name not in self.plugins:
            logger.warning(f"Plugin not loaded: {plugin_name}")
            return False
        
        self.enabled_plugins[plugin_name] = False
        logger.info(f"Disabled plugin: {plugin_name}")
        return True
    
    def get_plugin_for_prompt(self, prompt: str) -> Optional[JARVISPlugin]:
        """Find the best plugin to handle a prompt."""
        for plugin_name, plugin in self.plugins.items():
            if self.enabled_plugins.get(plugin_name, False) and plugin.can_handle(prompt):
                return plugin
        return None
    
    def execute_command(self, command: str, args: List[str], context: Optional[Dict] = None) -> Optional[str]:
        """Execute a plugin command."""
        if command not in self.commands:
            return None
        
        plugin = self.commands[command]
        plugin_name = self._get_plugin_name(plugin)
        
        if not self.enabled_plugins.get(plugin_name, False):
            logger.warning(f"Plugin disabled for command: {command}")
            return None
        
        try:
            # Build prompt from command and args
            prompt = f"{command} {' '.join(args)}".strip()
            return plugin.process(prompt, context)
        except Exception as e:
            logger.error(f"Command execution failed for {command}: {e}")
            return f"Error executing command: {e}"
    
    def process_with_plugins(self, prompt: str, context: Optional[Dict] = None) -> Optional[str]:
        """Try to process prompt with plugins."""
        plugin = self.get_plugin_for_prompt(prompt)
        if plugin:
            plugin_name = self._get_plugin_name(plugin)
            logger.info(f"Processing with plugin: {plugin_name}")
            try:
                return plugin.process(prompt, context)
            except Exception as e:
                logger.error(f"Plugin processing failed for {plugin_name}: {e}")
                return f"Plugin error: {e}"
        return None
    
    def _get_plugin_name(self, plugin: JARVISPlugin) -> str:
        """Get plugin name from instance."""
        for name, instance in self.plugins.items():
            if instance == plugin:
                return name
        return "unknown"
    
    def list_plugins(self) -> Dict[str, Dict[str, Any]]:
        """List all plugins with their status."""
        plugin_list = {}
        for plugin_name, plugin in self.plugins.items():
            plugin_list[plugin_name] = {
                "name": plugin.name,
                "version": plugin.version,
                "description": plugin.description,
                "enabled": self.enabled_plugins.get(plugin_name, False),
                "commands": list(plugin.commands.keys()),
                "triggers": plugin.triggers
            }
        return plugin_list
    
    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific plugin."""
        if plugin_name not in self.plugins:
            return None
        
        plugin = self.plugins[plugin_name]
        return {
            "name": plugin.name,
            "version": plugin.version,
            "description": plugin.description,
            "enabled": self.enabled_plugins.get(plugin_name, False),
            "commands": list(plugin.commands.keys()),
            "triggers": plugin.triggers,
            "class_name": plugin.__class__.__name__,
            "module": plugin.__class__.__module__
        }


# Global plugin manager instance
plugin_manager = PluginManager()