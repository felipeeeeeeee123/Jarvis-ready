"""Database models for JARVIS v3.0."""

from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON, Index
from sqlalchemy.sql import func
from .config import Base


class Memory(Base):
    """Core memory storage for JARVIS."""
    __tablename__ = "memory"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(255), unique=True, index=True, nullable=False)
    value = Column(Text, nullable=True)
    value_type = Column(String(50), default="string")  # string, json, float, int, bool
    category = Column(String(100), index=True, default="general")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    metadata = Column(JSON, nullable=True)  # Additional metadata

    def __repr__(self):
        return f"<Memory(key='{self.key}', category='{self.category}')>"


class QAEntry(Base):
    """Question-Answer pairs with scoring."""
    __tablename__ = "qa_entries"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    source = Column(String(100), nullable=False)  # Ollama, OpenAI, DuckDuckGo, etc.
    confidence_score = Column(Float, default=0.0)
    token_count = Column(Integer, default=0)
    context = Column(Text, nullable=True)  # Additional context used
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_reviewed = Column(Boolean, default=False)
    review_score = Column(Float, nullable=True)
    tags = Column(JSON, nullable=True)  # List of tags

    # Indexes for better query performance
    __table_args__ = (
        Index('idx_qa_source', 'source'),
        Index('idx_qa_confidence', 'confidence_score'),
        Index('idx_qa_created', 'created_at'),
    )

    def __repr__(self):
        return f"<QAEntry(id={self.id}, source='{self.source}', score={self.confidence_score})>"


class TradeLog(Base):
    """Trading activity log."""
    __tablename__ = "trade_logs"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    action = Column(String(10), nullable=False)  # buy, sell, hold
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    total_value = Column(Float, nullable=False)
    strategy_used = Column(String(50), nullable=False)  # RSI, EMA, MACD
    signal_strength = Column(Float, nullable=True)  # Confidence in the signal
    executed_at = Column(DateTime(timezone=True), server_default=func.now())
    order_id = Column(String(100), nullable=True)  # Alpaca order ID
    status = Column(String(20), default="pending")  # pending, filled, cancelled, rejected
    error_message = Column(Text, nullable=True)
    portfolio_value_before = Column(Float, nullable=True)
    portfolio_value_after = Column(Float, nullable=True)
    metadata = Column(JSON, nullable=True)  # Additional trade data

    # Indexes for performance
    __table_args__ = (
        Index('idx_trade_symbol_date', 'symbol', 'executed_at'),
        Index('idx_trade_strategy', 'strategy_used'),
        Index('idx_trade_status', 'status'),
    )

    def __repr__(self):
        return f"<TradeLog(symbol='{self.symbol}', action='{self.action}', quantity={self.quantity})>"


class SystemConfig(Base):
    """System configuration and settings."""
    __tablename__ = "system_config"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(255), unique=True, index=True, nullable=False)
    value = Column(Text, nullable=False)
    value_type = Column(String(50), default="string")  # string, json, float, int, bool
    description = Column(Text, nullable=True)
    is_sensitive = Column(Boolean, default=False)  # Don't log sensitive configs
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<SystemConfig(key='{self.key}', type='{self.value_type}')>"


class PerformanceMetric(Base):
    """System performance metrics."""
    __tablename__ = "performance_metrics"

    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(50), nullable=True)  # seconds, bytes, count, etc.
    component = Column(String(100), nullable=False)  # ai_brain, autotrade, web_search, etc.
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())
    metadata = Column(JSON, nullable=True)

    # Indexes for time-series queries
    __table_args__ = (
        Index('idx_metrics_name_time', 'metric_name', 'recorded_at'),
        Index('idx_metrics_component', 'component'),
    )

    def __repr__(self):
        return f"<PerformanceMetric(name='{self.metric_name}', value={self.metric_value}, component='{self.component}')>"


class EngineeringFormula(Base):
    """Engineering formulas and solutions cache."""
    __tablename__ = "engineering_formulas"

    id = Column(Integer, primary_key=True, index=True)
    formula = Column(Text, nullable=False)
    keywords = Column(JSON, nullable=False)  # List of keywords for search
    solution_steps = Column(Text, nullable=False)
    domain = Column(String(100), nullable=False)  # mechanical, electrical, civil, etc.
    difficulty = Column(String(20), default="medium")  # easy, medium, hard
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    used_count = Column(Integer, default=0)
    last_used = Column(DateTime(timezone=True), nullable=True)

    # Indexes for search
    __table_args__ = (
        Index('idx_formula_domain', 'domain'),
        Index('idx_formula_difficulty', 'difficulty'),
        Index('idx_formula_used', 'used_count'),
    )

    def __repr__(self):
        return f"<EngineeringFormula(id={self.id}, domain='{self.domain}', used={self.used_count})>"


class ConversationHistory(Base):
    """Conversation history for multi-turn context."""
    __tablename__ = "conversation_history"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), nullable=False, index=True)
    user_message = Column(Text, nullable=False)
    assistant_response = Column(Text, nullable=False)
    message_type = Column(String(50), default="chat")  # chat, command, trade, search
    context_used = Column(JSON, nullable=True)  # Context that influenced the response
    response_time = Column(Float, nullable=True)  # Response time in seconds
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    metadata = Column(JSON, nullable=True)

    # Indexes for session queries
    __table_args__ = (
        Index('idx_conversation_session_time', 'session_id', 'created_at'),
        Index('idx_conversation_type', 'message_type'),
    )

    def __repr__(self):
        return f"<ConversationHistory(session='{self.session_id}', type='{self.message_type}')>"