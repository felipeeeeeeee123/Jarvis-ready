"""Database services for JARVIS v3.0."""

import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, or_, func
from .config import get_session
from .models import Memory, QAEntry, TradeLog, SystemConfig, PerformanceMetric, EngineeringFormula, ConversationHistory
from utils.logging_config import get_logger

logger = get_logger(__name__)


class MemoryService:
    """Service for managing core memory storage."""

    def __init__(self, db: Optional[Session] = None):
        self.db = db

    def _get_db(self) -> Session:
        return self.db or get_session()

    def set(self, key: str, value: Any, category: str = "general", metadata: Optional[Dict] = None) -> Memory:
        """Set a memory value."""
        db = self._get_db()
        try:
            # Determine value type and serialize if needed
            if isinstance(value, (dict, list)):
                value_str = json.dumps(value)
                value_type = "json"
            elif isinstance(value, bool):
                value_str = str(value)
                value_type = "bool"
            elif isinstance(value, (int, float)):
                value_str = str(value)
                value_type = "float" if isinstance(value, float) else "int"
            else:
                value_str = str(value)
                value_type = "string"

            # Check if memory exists
            memory = db.query(Memory).filter(Memory.key == key).first()
            if memory:
                memory.value = value_str
                memory.value_type = value_type
                memory.category = category
                memory.metadata = metadata
                memory.updated_at = datetime.utcnow()
            else:
                memory = Memory(
                    key=key,
                    value=value_str,
                    value_type=value_type,
                    category=category,
                    metadata=metadata
                )
                db.add(memory)

            db.commit()
            db.refresh(memory)
            return memory
        finally:
            if not self.db:
                db.close()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a memory value."""
        db = self._get_db()
        try:
            memory = db.query(Memory).filter(Memory.key == key).first()
            if not memory:
                return default

            # Deserialize based on type
            if memory.value_type == "json":
                return json.loads(memory.value)
            elif memory.value_type == "bool":
                return memory.value.lower() == "true"
            elif memory.value_type == "int":
                return int(memory.value)
            elif memory.value_type == "float":
                return float(memory.value)
            else:
                return memory.value
        finally:
            if not self.db:
                db.close()

    def get_by_category(self, category: str) -> Dict[str, Any]:
        """Get all memories in a category."""
        db = self._get_db()
        try:
            memories = db.query(Memory).filter(Memory.category == category).all()
            result = {}
            for memory in memories:
                result[memory.key] = self._deserialize_value(memory)
            return result
        finally:
            if not self.db:
                db.close()

    def delete(self, key: str) -> bool:
        """Delete a memory entry."""
        db = self._get_db()
        try:
            memory = db.query(Memory).filter(Memory.key == key).first()
            if memory:
                db.delete(memory)
                db.commit()
                return True
            return False
        finally:
            if not self.db:
                db.close()

    def _deserialize_value(self, memory: Memory) -> Any:
        """Deserialize a memory value based on its type."""
        if memory.value_type == "json":
            return json.loads(memory.value)
        elif memory.value_type == "bool":
            return memory.value.lower() == "true"
        elif memory.value_type == "int":
            return int(memory.value)
        elif memory.value_type == "float":
            return float(memory.value)
        else:
            return memory.value


class QAService:
    """Service for managing Q&A entries."""

    def __init__(self, db: Optional[Session] = None):
        self.db = db

    def _get_db(self) -> Session:
        return self.db or get_session()

    def add_entry(self, question: str, answer: str, source: str, confidence_score: float = 0.0,
                  context: Optional[str] = None, tags: Optional[List[str]] = None) -> QAEntry:
        """Add a new Q&A entry."""
        db = self._get_db()
        try:
            entry = QAEntry(
                question=question,
                answer=answer,
                source=source,
                confidence_score=confidence_score,
                token_count=len(answer.split()),
                context=context,
                tags=tags
            )
            db.add(entry)
            db.commit()
            db.refresh(entry)
            logger.info(f"Added Q&A entry", extra={"source": source, "confidence": confidence_score})
            return entry
        finally:
            if not self.db:
                db.close()

    def get_recent_entries(self, limit: int = 100) -> List[QAEntry]:
        """Get recent Q&A entries."""
        db = self._get_db()
        try:
            return db.query(QAEntry).order_by(desc(QAEntry.created_at)).limit(limit).all()
        finally:
            if not self.db:
                db.close()

    def get_low_confidence_entries(self, threshold: float = 0.5, limit: int = 50) -> List[QAEntry]:
        """Get entries with low confidence scores for review."""
        db = self._get_db()
        try:
            return (db.query(QAEntry)
                   .filter(QAEntry.confidence_score < threshold)
                   .filter(QAEntry.is_reviewed == False)
                   .order_by(QAEntry.created_at)
                   .limit(limit)
                   .all())
        finally:
            if not self.db:
                db.close()

    def update_review(self, entry_id: int, review_score: float) -> Optional[QAEntry]:
        """Update the review score for an entry."""
        db = self._get_db()
        try:
            entry = db.query(QAEntry).filter(QAEntry.id == entry_id).first()
            if entry:
                entry.review_score = review_score
                entry.is_reviewed = True
                entry.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(entry)
            return entry
        finally:
            if not self.db:
                db.close()

    def prune_old_entries(self, keep_count: int = 1000) -> int:
        """Prune old entries, keeping only the most recent ones."""
        db = self._get_db()
        try:
            # Get IDs of entries to keep
            keep_ids = (db.query(QAEntry.id)
                       .order_by(desc(QAEntry.created_at))
                       .limit(keep_count)
                       .subquery())

            # Delete entries not in keep list
            deleted = (db.query(QAEntry)
                      .filter(~QAEntry.id.in_(keep_ids))
                      .delete(synchronize_session=False))

            db.commit()
            logger.info(f"Pruned {deleted} old Q&A entries")
            return deleted
        finally:
            if not self.db:
                db.close()


class TradeService:
    """Service for managing trade logs."""

    def __init__(self, db: Optional[Session] = None):
        self.db = db

    def _get_db(self) -> Session:
        return self.db or get_session()

    def log_trade(self, symbol: str, action: str, quantity: int, price: float,
                  strategy_used: str, signal_strength: Optional[float] = None,
                  order_id: Optional[str] = None, metadata: Optional[Dict] = None) -> TradeLog:
        """Log a trade execution."""
        db = self._get_db()
        try:
            trade = TradeLog(
                symbol=symbol,
                action=action,
                quantity=quantity,
                price=price,
                total_value=quantity * price,
                strategy_used=strategy_used,
                signal_strength=signal_strength,
                order_id=order_id,
                metadata=metadata
            )
            db.add(trade)
            db.commit()
            db.refresh(trade)
            logger.info(f"Logged trade", extra={
                "symbol": symbol, "action": action, "quantity": quantity, "price": price
            })
            return trade
        finally:
            if not self.db:
                db.close()

    def get_recent_trades(self, symbol: Optional[str] = None, limit: int = 100) -> List[TradeLog]:
        """Get recent trades, optionally filtered by symbol."""
        db = self._get_db()
        try:
            query = db.query(TradeLog)
            if symbol:
                query = query.filter(TradeLog.symbol == symbol)
            return query.order_by(desc(TradeLog.executed_at)).limit(limit).all()
        finally:
            if not self.db:
                db.close()

    def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get trading performance summary for the last N days."""
        db = self._get_db()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            trades = (db.query(TradeLog)
                     .filter(TradeLog.executed_at >= cutoff_date)
                     .filter(TradeLog.status == "filled")
                     .all())

            total_trades = len(trades)
            buy_trades = [t for t in trades if t.action == "buy"]
            sell_trades = [t for t in trades if t.action == "sell"]
            total_volume = sum(t.total_value for t in trades)

            return {
                "total_trades": total_trades,
                "buy_trades": len(buy_trades),
                "sell_trades": len(sell_trades),
                "total_volume": total_volume,
                "avg_trade_size": total_volume / total_trades if total_trades > 0 else 0,
                "symbols_traded": list(set(t.symbol for t in trades)),
                "strategies_used": list(set(t.strategy_used for t in trades))
            }
        finally:
            if not self.db:
                db.close()


class SystemConfigService:
    """Service for managing system configuration."""

    def __init__(self, db: Optional[Session] = None):
        self.db = db

    def _get_db(self) -> Session:
        return self.db or get_session()

    def set_config(self, key: str, value: Any, description: Optional[str] = None,
                   is_sensitive: bool = False) -> SystemConfig:
        """Set a configuration value."""
        db = self._get_db()
        try:
            # Serialize value
            if isinstance(value, (dict, list)):
                value_str = json.dumps(value)
                value_type = "json"
            elif isinstance(value, bool):
                value_str = str(value)
                value_type = "bool"
            elif isinstance(value, (int, float)):
                value_str = str(value)
                value_type = "float" if isinstance(value, float) else "int"
            else:
                value_str = str(value)
                value_type = "string"

            config = db.query(SystemConfig).filter(SystemConfig.key == key).first()
            if config:
                config.value = value_str
                config.value_type = value_type
                config.description = description
                config.is_sensitive = is_sensitive
                config.updated_at = datetime.utcnow()
            else:
                config = SystemConfig(
                    key=key,
                    value=value_str,
                    value_type=value_type,
                    description=description,
                    is_sensitive=is_sensitive
                )
                db.add(config)

            db.commit()
            db.refresh(config)
            
            if not is_sensitive:
                logger.info(f"Updated config: {key}")
            return config
        finally:
            if not self.db:
                db.close()

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        db = self._get_db()
        try:
            config = db.query(SystemConfig).filter(SystemConfig.key == key).first()
            if not config:
                return default

            # Deserialize based on type
            if config.value_type == "json":
                return json.loads(config.value)
            elif config.value_type == "bool":
                return config.value.lower() == "true"
            elif config.value_type == "int":
                return int(config.value)
            elif config.value_type == "float":
                return float(config.value)
            else:
                return config.value
        finally:
            if not self.db:
                db.close()


# Create service instances for easy import
memory_service = MemoryService()
qa_service = QAService()
trade_service = TradeService()
config_service = SystemConfigService()