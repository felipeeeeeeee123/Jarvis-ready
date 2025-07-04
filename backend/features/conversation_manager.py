"""Conversation memory and session management for JARVIS v3.0."""

import uuid
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from database.services import get_session
from database.models import ConversationHistory
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_
from utils.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class ConversationManager:
    """Manages conversation sessions and multi-turn context."""
    
    def __init__(self, max_context_turns: int = 10, session_timeout_hours: int = 24):
        self.max_context_turns = max_context_turns
        self.session_timeout_hours = session_timeout_hours
        self.current_session_id = None
        self.context_cache = {}  # In-memory cache for active sessions
        logger.info("Conversation manager initialized")
    
    def start_new_session(self, session_type: str = "chat") -> str:
        """Start a new conversation session."""
        session_id = f"{session_type}_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        self.current_session_id = session_id
        self.context_cache[session_id] = []
        logger.info(f"Started new conversation session: {session_id}")
        return session_id
    
    def get_current_session(self) -> Optional[str]:
        """Get current session ID, create new if none exists."""
        if not self.current_session_id:
            return self.start_new_session()
        return self.current_session_id
    
    def add_interaction(self, user_message: str, assistant_response: str, 
                       message_type: str = "chat", context_used: Optional[Dict] = None,
                       response_time: Optional[float] = None, metadata: Optional[Dict] = None) -> None:
        """Add a user-assistant interaction to the current session."""
        session_id = self.get_current_session()
        
        db = get_session()
        try:
            # Store in database
            conversation = ConversationHistory(
                session_id=session_id,
                user_message=user_message,
                assistant_response=assistant_response,
                message_type=message_type,
                context_used=context_used,
                response_time=response_time,
                metadata=metadata
            )
            db.add(conversation)
            db.commit()
            
            # Update cache
            if session_id not in self.context_cache:
                self.context_cache[session_id] = []
            
            self.context_cache[session_id].append({
                "user_message": user_message,
                "assistant_response": assistant_response,
                "timestamp": datetime.utcnow(),
                "message_type": message_type
            })
            
            # Keep cache within limits
            if len(self.context_cache[session_id]) > self.max_context_turns:
                self.context_cache[session_id] = self.context_cache[session_id][-self.max_context_turns:]
            
            logger.info(f"Added interaction to session {session_id}", extra={
                "session_id": session_id,
                "message_type": message_type,
                "response_time": response_time
            })
            
        finally:
            db.close()
    
    def get_conversation_context(self, session_id: Optional[str] = None, 
                               max_turns: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get conversation context for the specified session."""
        if not session_id:
            session_id = self.get_current_session()
        
        max_turns = max_turns or self.max_context_turns
        
        # Try cache first
        if session_id in self.context_cache:
            cached_context = self.context_cache[session_id][-max_turns:]
            if cached_context:
                return cached_context
        
        # Fallback to database
        db = get_session()
        try:
            conversations = (db.query(ConversationHistory)
                           .filter(ConversationHistory.session_id == session_id)
                           .order_by(desc(ConversationHistory.created_at))
                           .limit(max_turns)
                           .all())
            
            # Convert to cache format and reverse to chronological order
            context = []
            for conv in reversed(conversations):
                context.append({
                    "user_message": conv.user_message,
                    "assistant_response": conv.assistant_response,
                    "timestamp": conv.created_at,
                    "message_type": conv.message_type
                })
            
            # Update cache
            self.context_cache[session_id] = context
            return context
            
        finally:
            db.close()
    
    def build_context_prompt(self, current_prompt: str, session_id: Optional[str] = None,
                           include_system_info: bool = True) -> str:
        """Build a context-aware prompt including conversation history."""
        context = self.get_conversation_context(session_id)
        
        if not context:
            return current_prompt
        
        # Build context string
        context_parts = []
        
        if include_system_info:
            context_parts.append(f"You are JARVIS v{settings.APP_VERSION}, an advanced AI assistant.")
            context_parts.append("You have access to trading, engineering analysis, web search, and memory systems.")
        
        # Add recent conversation history
        if context:
            context_parts.append("\nRecent conversation:")
            for interaction in context[-5:]:  # Last 5 interactions for context
                timestamp = interaction["timestamp"].strftime("%H:%M")
                context_parts.append(f"[{timestamp}] User: {interaction['user_message'][:100]}...")
                context_parts.append(f"[{timestamp}] You: {interaction['assistant_response'][:100]}...")
        
        context_parts.append(f"\nCurrent question: {current_prompt}")
        
        return "\n".join(context_parts)
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary information about a conversation session."""
        db = get_session()
        try:
            conversations = (db.query(ConversationHistory)
                           .filter(ConversationHistory.session_id == session_id)
                           .order_by(ConversationHistory.created_at)
                           .all())
            
            if not conversations:
                return {"error": "Session not found"}
            
            first_message = conversations[0]
            last_message = conversations[-1]
            
            # Analyze message types
            message_types = {}
            total_response_time = 0
            response_count = 0
            
            for conv in conversations:
                msg_type = conv.message_type
                message_types[msg_type] = message_types.get(msg_type, 0) + 1
                
                if conv.response_time:
                    total_response_time += conv.response_time
                    response_count += 1
            
            avg_response_time = total_response_time / response_count if response_count > 0 else 0
            session_duration = (last_message.created_at - first_message.created_at).total_seconds()
            
            return {
                "session_id": session_id,
                "start_time": first_message.created_at.isoformat(),
                "end_time": last_message.created_at.isoformat(),
                "duration_seconds": session_duration,
                "total_interactions": len(conversations),
                "message_types": message_types,
                "avg_response_time": round(avg_response_time, 2),
                "first_message": first_message.user_message[:100],
                "last_message": last_message.user_message[:100]
            }
            
        finally:
            db.close()
    
    def cleanup_old_sessions(self, older_than_hours: Optional[int] = None) -> int:
        """Clean up old conversation sessions."""
        older_than_hours = older_than_hours or (self.session_timeout_hours * 7)  # Default: 7x timeout
        cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
        
        db = get_session()
        try:
            # Get session IDs to clean up
            old_sessions = (db.query(ConversationHistory.session_id)
                          .filter(ConversationHistory.created_at < cutoff_time)
                          .distinct()
                          .all())
            
            session_ids = [s[0] for s in old_sessions]
            
            # Delete old conversations
            deleted = (db.query(ConversationHistory)
                      .filter(ConversationHistory.created_at < cutoff_time)
                      .delete())
            
            db.commit()
            
            # Clean up cache
            for session_id in session_ids:
                self.context_cache.pop(session_id, None)
            
            logger.info(f"Cleaned up {deleted} conversation entries from {len(session_ids)} old sessions")
            return deleted
            
        finally:
            db.close()
    
    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get list of recent conversation sessions."""
        db = get_session()
        try:
            # Get recent unique sessions
            recent_sessions = (db.query(ConversationHistory.session_id)
                             .distinct()
                             .order_by(desc(ConversationHistory.created_at))
                             .limit(limit)
                             .all())
            
            session_summaries = []
            for session_tuple in recent_sessions:
                session_id = session_tuple[0]
                summary = self.get_session_summary(session_id)
                if "error" not in summary:
                    session_summaries.append(summary)
            
            return session_summaries
            
        finally:
            db.close()
    
    def switch_session(self, session_id: str) -> bool:
        """Switch to an existing session."""
        db = get_session()
        try:
            # Verify session exists
            session_exists = (db.query(ConversationHistory)
                            .filter(ConversationHistory.session_id == session_id)
                            .first() is not None)
            
            if session_exists:
                self.current_session_id = session_id
                logger.info(f"Switched to session: {session_id}")
                return True
            else:
                logger.warning(f"Attempted to switch to non-existent session: {session_id}")
                return False
                
        finally:
            db.close()


# Global conversation manager instance
conversation_manager = ConversationManager()