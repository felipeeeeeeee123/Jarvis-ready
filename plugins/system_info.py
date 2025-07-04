"""System Information Plugin for JARVIS v3.0."""

import sys
import os
import platform
import psutil
from datetime import datetime
from typing import Dict, List, Callable, Optional

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from core.plugin_manager import JARVISPlugin


class SystemInfoPlugin(JARVISPlugin):
    """Plugin for system information and monitoring."""
    
    @property
    def name(self) -> str:
        return "System Info"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Provides system information including CPU, memory, disk usage, and platform details"
    
    @property
    def commands(self) -> Dict[str, Callable]:
        return {
            "sysinfo": self.get_system_info,
            "cpu": self.get_cpu_info,
            "memory": self.get_memory_info,
            "disk": self.get_disk_info,
            "uptime": self.get_uptime
        }
    
    @property
    def triggers(self) -> List[str]:
        return ["system", "cpu", "memory", "disk", "uptime", "platform", "sysinfo"]
    
    def initialize(self) -> bool:
        """Initialize the plugin."""
        try:
            # Test if psutil is available
            psutil.cpu_percent()
            return True
        except Exception:
            return False
    
    def can_handle(self, prompt: str) -> bool:
        """Check if this plugin can handle the given prompt."""
        prompt_lower = prompt.lower()
        return any(trigger in prompt_lower for trigger in self.triggers)
    
    def process(self, prompt: str, context: Optional[Dict] = None) -> str:
        """Process the prompt and return system information."""
        prompt_lower = prompt.lower()
        
        if "cpu" in prompt_lower:
            return self.get_cpu_info()
        elif "memory" in prompt_lower or "ram" in prompt_lower:
            return self.get_memory_info()
        elif "disk" in prompt_lower or "storage" in prompt_lower:
            return self.get_disk_info()
        elif "uptime" in prompt_lower:
            return self.get_uptime()
        else:
            return self.get_system_info()
    
    def get_system_info(self) -> str:
        """Get comprehensive system information."""
        try:
            # Platform info
            platform_info = {
                "System": platform.system(),
                "Release": platform.release(),
                "Version": platform.version(),
                "Machine": platform.machine(),
                "Processor": platform.processor(),
                "Python Version": platform.python_version()
            }
            
            # CPU info
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Memory info
            memory = psutil.virtual_memory()
            
            # Disk info
            disk = psutil.disk_usage('/')
            
            # Format response
            response = "🖥️ **System Information**\n\n"
            
            response += "**Platform:**\n"
            for key, value in platform_info.items():
                response += f"• {key}: {value}\n"
            
            response += f"\n**CPU:**\n"
            response += f"• Cores: {cpu_count}\n"
            response += f"• Usage: {cpu_percent}%\n"
            if cpu_freq:
                response += f"• Frequency: {cpu_freq.current:.2f} MHz\n"
            
            response += f"\n**Memory:**\n"
            response += f"• Total: {self._bytes_to_gb(memory.total):.2f} GB\n"
            response += f"• Available: {self._bytes_to_gb(memory.available):.2f} GB\n"
            response += f"• Used: {memory.percent}%\n"
            
            response += f"\n**Disk:**\n"
            response += f"• Total: {self._bytes_to_gb(disk.total):.2f} GB\n"
            response += f"• Free: {self._bytes_to_gb(disk.free):.2f} GB\n"
            response += f"• Used: {((disk.total - disk.free) / disk.total * 100):.1f}%\n"
            
            return response
            
        except Exception as e:
            return f"Error getting system info: {e}"
    
    def get_cpu_info(self) -> str:
        """Get CPU information."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            response = f"🔧 **CPU Information**\n\n"
            response += f"• Physical Cores: {psutil.cpu_count(logical=False)}\n"
            response += f"• Total Cores: {cpu_count}\n"
            response += f"• Current Usage: {cpu_percent}%\n"
            
            if cpu_freq:
                response += f"• Current Frequency: {cpu_freq.current:.2f} MHz\n"
                response += f"• Max Frequency: {cpu_freq.max:.2f} MHz\n"
            
            # Per-core usage
            per_cpu = psutil.cpu_percent(percpu=True, interval=1)
            response += f"\n**Per-Core Usage:**\n"
            for i, percent in enumerate(per_cpu):
                response += f"• Core {i}: {percent}%\n"
            
            return response
            
        except Exception as e:
            return f"Error getting CPU info: {e}"
    
    def get_memory_info(self) -> str:
        """Get memory information."""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            response = f"💾 **Memory Information**\n\n"
            response += f"**Virtual Memory:**\n"
            response += f"• Total: {self._bytes_to_gb(memory.total):.2f} GB\n"
            response += f"• Available: {self._bytes_to_gb(memory.available):.2f} GB\n"
            response += f"• Used: {self._bytes_to_gb(memory.used):.2f} GB ({memory.percent}%)\n"
            response += f"• Free: {self._bytes_to_gb(memory.free):.2f} GB\n"
            
            response += f"\n**Swap Memory:**\n"
            response += f"• Total: {self._bytes_to_gb(swap.total):.2f} GB\n"
            response += f"• Used: {self._bytes_to_gb(swap.used):.2f} GB ({swap.percent}%)\n"
            response += f"• Free: {self._bytes_to_gb(swap.free):.2f} GB\n"
            
            return response
            
        except Exception as e:
            return f"Error getting memory info: {e}"
    
    def get_disk_info(self) -> str:
        """Get disk information."""
        try:
            response = f"💿 **Disk Information**\n\n"
            
            # Get all disk partitions
            partitions = psutil.disk_partitions()
            
            for partition in partitions:
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    response += f"**{partition.device}** ({partition.mountpoint})\n"
                    response += f"• File System: {partition.fstype}\n"
                    response += f"• Total: {self._bytes_to_gb(usage.total):.2f} GB\n"
                    response += f"• Used: {self._bytes_to_gb(usage.used):.2f} GB ({(usage.used/usage.total*100):.1f}%)\n"
                    response += f"• Free: {self._bytes_to_gb(usage.free):.2f} GB\n\n"
                except PermissionError:
                    response += f"**{partition.device}** - Permission denied\n\n"
            
            return response
            
        except Exception as e:
            return f"Error getting disk info: {e}"
    
    def get_uptime(self) -> str:
        """Get system uptime."""
        try:
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            uptime = datetime.now() - boot_time
            
            days = uptime.days
            hours, remainder = divmod(uptime.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            response = f"⏰ **System Uptime**\n\n"
            response += f"• Boot Time: {boot_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            response += f"• Uptime: {days} days, {hours} hours, {minutes} minutes, {seconds} seconds\n"
            
            return response
            
        except Exception as e:
            return f"Error getting uptime: {e}"
    
    def _bytes_to_gb(self, bytes_value: int) -> float:
        """Convert bytes to gigabytes."""
        return bytes_value / (1024 ** 3)