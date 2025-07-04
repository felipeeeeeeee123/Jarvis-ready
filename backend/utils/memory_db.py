"""Database-backed memory manager for JARVIS v3.0."""

import time
from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta
from database.services import memory_service, trade_service
from utils.logging_config import get_logger

logger = get_logger(__name__)


class DatabaseMemoryManager:
    """Database-backed memory manager replacing JSON storage."""

    def __init__(self):
        self.memory_service = memory_service
        self.trade_service = trade_service
        self._trade_cooldowns = {}  # In-memory cache for trade cooldowns
        logger.info("Database memory manager initialized")

    @property
    def memory(self) -> Dict[str, Any]:
        """Get all general memory as a dictionary (for backward compatibility)."""
        return self.memory_service.get_by_category("general")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a memory value."""
        return self.memory_service.get(key, default)

    def set(self, key: str, value: Any, category: str = "general") -> None:
        """Set a memory value."""
        self.memory_service.set(key, value, category)

    def save(self) -> None:
        """Save method for backward compatibility (no-op since DB auto-saves)."""
        pass

    def should_trade(self, symbol: str, cooldown_seconds: int) -> bool:
        """Check if we should trade a symbol (respects cooldown)."""
        # Check in-memory cache first
        if symbol in self._trade_cooldowns:
            last_trade_time = self._trade_cooldowns[symbol]
            if time.time() - last_trade_time < cooldown_seconds:
                return False

        # Check database for recent trades
        recent_trades = self.trade_service.get_recent_trades(symbol, limit=1)
        if recent_trades:
            last_trade = recent_trades[0]
            time_since_last = datetime.utcnow() - last_trade.executed_at
            if time_since_last.total_seconds() < cooldown_seconds:
                # Update in-memory cache
                self._trade_cooldowns[symbol] = time.time()
                return False

        return True

    def set_cooldown(self, symbol: str) -> None:
        """Set cooldown for a symbol."""
        self._trade_cooldowns[symbol] = time.time()
        logger.info(f"Set trade cooldown for {symbol}")

    def get_trading_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get trading statistics for the last N days."""
        return self.trade_service.get_performance_summary(days)

    def get_recent_trades(self, symbol: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent trades."""
        trades = self.trade_service.get_recent_trades(symbol, limit)
        return [
            {
                "symbol": trade.symbol,
                "action": trade.action,
                "quantity": trade.quantity,
                "price": trade.price,
                "total_value": trade.total_value,
                "strategy": trade.strategy_used,
                "executed_at": trade.executed_at.isoformat(),
                "status": trade.status
            }
            for trade in trades
        ]

    def log_trade(self, symbol: str, action: str, quantity: int, price: float,
                  strategy: str, order_id: Optional[str] = None) -> None:
        """Log a trade execution."""
        self.trade_service.log_trade(
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=price,
            strategy_used=strategy,
            order_id=order_id
        )

    # Backward compatibility methods
    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access."""
        return self.memory.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Dictionary-style setting."""
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return self.get(key) is not None

    def keys(self):
        """Get all keys."""
        return self.memory.keys()

    def items(self):
        """Get all items."""
        return self.memory.items()

    def values(self):
        """Get all values."""
        return self.memory.values()

    def update(self, other: Dict[str, Any]) -> None:
        """Update with dictionary."""
        for key, value in other.items():
            self.set(key, value)

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and stats."""
        return {
            "memory_entries": len(self.memory),
            "recent_trades": len(self.get_recent_trades(limit=100)),
            "trading_stats": self.get_trading_stats(7),  # Last 7 days
            "cooldowns_active": len(self._trade_cooldowns),
            "uptime": self.get("system_start_time", "unknown")
        }