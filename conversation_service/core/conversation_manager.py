"""
Conversation Manager for maintaining conversation context and history.

This manager handles conversation state, context tracking, and history
management for the conversation service. It provides a simple but effective
memory system for maintaining conversation continuity.

Classes:
    - ConversationManager: Main conversation context manager
    - ConversationStore: Storage interface for conversation data
    - MemoryStore: In-memory implementation of conversation storage

Author: Conversation Service Team
Created: 2025-01-31
Version: 1.0.0 MVP - Simple Conversation Management
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict, deque
from abc import ABC, abstractmethod

from ..models.conversation_models import ConversationContext, ConversationTurn

logger = logging.getLogger(__name__)


class ConversationStore(ABC):
    """Abstract base class for conversation storage backends."""
    
    @abstractmethod
    async def get_context(self, conversation_id: str) -> Optional[ConversationContext]:
        """Get conversation context by ID."""
        pass
    
    @abstractmethod
    async def save_context(self, context: ConversationContext) -> None:
        """Save conversation context."""
        pass
    
    @abstractmethod
    async def add_turn(self, conversation_id: str, turn: ConversationTurn) -> None:
        """Add a conversation turn."""
        pass
    
    @abstractmethod
    async def clear_context(self, conversation_id: str) -> None:
        """Clear conversation context."""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        pass


class MemoryStore(ConversationStore):
    """
    In-memory implementation of conversation storage.
    
    This is a simple, fast storage implementation suitable for MVP
    and development. Data is lost when the service restarts.
    
    Attributes:
        conversations: Dictionary storing conversation contexts
        max_conversations: Maximum number of conversations to keep
        max_turns_per_conversation: Maximum turns per conversation
        cleanup_interval: Interval for automatic cleanup
    """
    
    def __init__(self, max_conversations: int = 1000, 
                 max_turns_per_conversation: int = 50,
                 cleanup_interval_hours: int = 24):
        """
        Initialize memory store.
        
        Args:
            max_conversations: Maximum conversations to keep in memory
            max_turns_per_conversation: Maximum turns per conversation
            cleanup_interval_hours: Hours between automatic cleanup
        """
        self.conversations: Dict[str, ConversationContext] = {}
        self.max_conversations = max_conversations
        self.max_turns_per_conversation = max_turns_per_conversation
        self.cleanup_interval = timedelta(hours=cleanup_interval_hours)
        
        # Statistics
        self.stats = {
            "total_conversations": 0,
            "total_turns": 0,
            "conversations_created": 0,
            "conversations_cleaned": 0,
            "last_cleanup": datetime.utcnow()
        }
        
        logger.info(f"Initialized MemoryStore with max {max_conversations} conversations")
    
    async def get_context(self, conversation_id: str) -> Optional[ConversationContext]:
        """Get conversation context by ID."""
        return self.conversations.get(conversation_id)
    
    async def save_context(self, context: ConversationContext) -> None:
        """Save conversation context."""
        # Check if this is a new conversation
        if context.conversation_id not in self.conversations:
            self.stats["conversations_created"] += 1
            
            # Check if we need to clean up old conversations
            if len(self.conversations) >= self.max_conversations:
                await self._cleanup_old_conversations()
        
        self.conversations[context.conversation_id] = context
        self.stats["total_conversations"] = len(self.conversations)
        
        logger.debug(f"Saved context for conversation {context.conversation_id}")
    
    async def add_turn(self, conversation_id: str, turn: ConversationTurn) -> None:
        """Add a conversation turn."""
        context = await self.get_context(conversation_id)
        
        if context is None:
            # Create new conversation context
            context = ConversationContext(
                conversation_id=conversation_id,
                user_id=1,  # Default user ID for MVP
                turns=[],
                current_turn=1,
                status="active",
                language="fr"
            )
        
        # Add the turn
        context.turns.append(turn)
        context.current_turn = len(context.turns)
        context.updated_at = datetime.utcnow()
        
        # Limit turns per conversation
        if len(context.turns) > self.max_turns_per_conversation:
            context.turns = context.turns[-self.max_turns_per_conversation:]
            context.current_turn = len(context.turns)
        
        # Update context summary for efficiency
        context.context_summary = self._generate_context_summary(context)
        
        # Save updated context
        await self.save_context(context)
        
        self.stats["total_turns"] += 1
        
        logger.debug(f"Added turn to conversation {conversation_id}")
    
    async def clear_context(self, conversation_id: str) -> None:
        """Clear conversation context."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            self.stats["total_conversations"] = len(self.conversations)
            logger.debug(f"Cleared context for conversation {conversation_id}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        # Calculate memory usage estimate
        total_turns = sum(len(conv.turns) for conv in self.conversations.values())
        avg_turns_per_conversation = total_turns / len(self.conversations) if self.conversations else 0
        
        return {
            "storage_type": "memory",
            "total_conversations": len(self.conversations),
            "total_turns": total_turns,
            "average_turns_per_conversation": round(avg_turns_per_conversation, 2),
            "max_conversations": self.max_conversations,
            "max_turns_per_conversation": self.max_turns_per_conversation,
            "memory_usage_estimate_mb": self._estimate_memory_usage(),
            "conversations_created": self.stats["conversations_created"],
            "conversations_cleaned": self.stats["conversations_cleaned"],
            "last_cleanup": self.stats["last_cleanup"].isoformat()
        }
    
    def _generate_context_summary(self, context: ConversationContext) -> str:
        """Generate a summary of conversation context."""
        if not context.turns:
            return "Nouvelle conversation"
        
        recent_turns = context.turns[-3:]  # Last 3 turns
        summary_parts = []
        
        # Extract key topics and intents
        intents = [turn.intent_detected for turn in recent_turns if turn.intent_detected]
        if intents:
            summary_parts.append(f"Intentions: {', '.join(set(intents))}")
        
        # Extract entities if available
        entities = []
        for turn in recent_turns:
            if turn.entities_extracted:
                for entity in turn.entities_extracted:
                    if entity.get('entity_type') and entity.get('normalized_value'):
                        entities.append(f"{entity['entity_type']}:{entity['normalized_value']}")
        
        if entities:
            summary_parts.append(f"Entités: {', '.join(entities[:5])}")  # Limit to 5 entities
        
        # Add turn count
        summary_parts.append(f"Tours: {len(context.turns)}")
        
        return " | ".join(summary_parts)
    
    async def _cleanup_old_conversations(self) -> None:
        """Clean up old conversations to free memory."""
        if not self.conversations:
            return
        
        # Sort conversations by last update time
        sorted_conversations = sorted(
            self.conversations.items(),
            key=lambda x: x[1].updated_at
        )
        
        # Remove oldest 20% of conversations
        num_to_remove = max(1, len(sorted_conversations) // 5)
        
        for i in range(num_to_remove):
            conversation_id = sorted_conversations[i][0]
            del self.conversations[conversation_id]
            self.stats["conversations_cleaned"] += 1
        
        self.stats["last_cleanup"] = datetime.utcnow()
        self.stats["total_conversations"] = len(self.conversations)
        
        logger.info(f"Cleaned up {num_to_remove} old conversations")
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        try:
            import sys
            total_size = sys.getsizeof(self.conversations)
            
            for conv in self.conversations.values():
                total_size += sys.getsizeof(conv)
                total_size += sum(sys.getsizeof(turn) for turn in conv.turns)
            
            return total_size / (1024 * 1024)  # Convert to MB
        except Exception:
            return 0.0  # Fallback if sys.getsizeof fails


class ConversationManager:
    """
    Main conversation context manager.
    
    This manager provides high-level APIs for conversation context
    management, including context tracking, history management,
    and user session handling.
    
    Attributes:
        store: Storage backend for conversation data
        user_sessions: Active user session tracking
        auto_cleanup_enabled: Whether automatic cleanup is enabled
        cleanup_task: Background cleanup task
    """
    
    def __init__(self, storage_backend: str = "memory", 
                 max_conversations: int = 1000):
        """
        Initialize conversation manager.
        
        Args:
            storage_backend: Storage backend type ("memory" for MVP)
            max_conversations: Maximum conversations to maintain
        """
        self.storage_backend = storage_backend
        self.max_conversations = max_conversations
        
        # Initialize storage
        if storage_backend == "memory":
            self.store = MemoryStore(max_conversations=max_conversations)
        else:
            raise ValueError(f"Unsupported storage backend: {storage_backend}")
        
        # User session tracking
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Background tasks
        self.auto_cleanup_enabled = True
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Manager statistics
        self.manager_stats = {
            "context_retrievals": 0,
            "turns_added": 0,
            "contexts_cleared": 0,
            "user_sessions_created": 0,
            "start_time": datetime.utcnow()
        }
        
        logger.info(f"Initialized ConversationManager with {storage_backend} backend")
    
    async def initialize(self) -> None:
        """Initialize the conversation manager and start background tasks."""
        try:
            # Start background cleanup task if enabled
            if self.auto_cleanup_enabled:
                self.cleanup_task = asyncio.create_task(self._background_cleanup())
            
            logger.info("ConversationManager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ConversationManager: {e}")
            raise
    
    async def get_context(
        self, conversation_id: str, user_id: Optional[int] = None
    ) -> ConversationContext:
        """Get conversation context, creating if it doesn't exist.

        Args:
            conversation_id: Conversation identifier
            user_id: Optional user identifier

        Returns:
            ConversationContext object
        """
        self.manager_stats["context_retrievals"] += 1

        context = await self.store.get_context(conversation_id)

        if context is None:
            # Create new conversation context
            context = ConversationContext(
                conversation_id=conversation_id,
                user_id=user_id if user_id is not None else 1,
                turns=[],
                current_turn=0,
                status="active",
                language="fr",
                domain="financial"
            )

            await self.store.save_context(context)
            logger.debug(f"Created new conversation context: {conversation_id}")
        elif user_id is not None and context.user_id != user_id:
            context.user_id = user_id
            await self.store.save_context(context)

        return context
    
    async def add_turn(self, conversation_id: str, user_id: int, user_msg: str,
                      assistant_msg: str, intent_detected: Optional[str] = None,
                      entities_extracted: Optional[List[Dict]] = None,
                      processing_time_ms: Optional[float] = None,
                      agent_chain: Optional[List[str]] = None,
                      search_results_count: Optional[int] = None,
                      confidence_score: Optional[float] = None) -> None:
        """
        Add a conversation turn.

        Args:
            conversation_id: Conversation identifier
            user_id: User identifier
            user_msg: User's message
            assistant_msg: Assistant's response
            intent_detected: Detected intent (optional)
            entities_extracted: Extracted entities (optional)
            processing_time_ms: Processing time (optional)
            agent_chain: Chain of agents involved (optional)
            search_results_count: Number of search results returned (optional)
            confidence_score: Confidence score for the response (optional)
        """
        self.manager_stats["turns_added"] += 1

        # Get current context or create a new one with provided user_id
        context = await self.get_context(conversation_id, user_id=user_id)

        # Create conversation turn
        metadata: Dict[str, Any] = {}
        if search_results_count is not None:
            metadata["search_results_count"] = search_results_count

        turn = ConversationTurn(
            user_message=user_msg,
            assistant_response=assistant_msg,
            turn_number=context.current_turn + 1,
            processing_time_ms=processing_time_ms or 0.0,
            intent_detected=intent_detected,
            entities_extracted=entities_extracted or [],
            confidence_score=confidence_score or 0.8,
            error_occurred=False,
            agent_chain=agent_chain or ["orchestrator_agent"],
            metadata=metadata
        )

        # Add turn to storage
        await self.store.add_turn(conversation_id, turn)

        logger.debug(f"Added turn to conversation {conversation_id}")
    
    async def update_user_context(self, conversation_id: str, user_id: int,
                                user_message: str) -> None:
        """
        Update user context for a conversation.
        
        Args:
            conversation_id: Conversation identifier
            user_id: User identifier
            user_message: Current user message
        """
        context = await self.get_context(conversation_id, user_id=user_id)
        
        # Update user information
        context.user_id = user_id
        context.updated_at = datetime.utcnow()
        
        # Track user session
        if conversation_id not in self.user_sessions:
            self.user_sessions[conversation_id] = {
                "user_id": user_id,
                "start_time": datetime.utcnow(),
                "last_activity": datetime.utcnow(),
                "message_count": 0
            }
            self.manager_stats["user_sessions_created"] += 1
        
        # Update session activity
        session = self.user_sessions[conversation_id]
        session["last_activity"] = datetime.utcnow()
        session["message_count"] += 1
        
        # Update active entities based on current message
        context.active_entities = self._extract_active_entities(user_message, context)
        
        # Save updated context
        await self.store.save_context(context)
    
    async def clear_context(self, conversation_id: str) -> None:
        """
        Clear conversation context and session data.
        
        Args:
            conversation_id: Conversation identifier
        """
        await self.store.clear_context(conversation_id)
        
        # Clear user session
        if conversation_id in self.user_sessions:
            del self.user_sessions[conversation_id]
        
        self.manager_stats["contexts_cleared"] += 1
        
        logger.debug(f"Cleared context for conversation {conversation_id}")
    
    async def get_conversation_summary(self, conversation_id: str, 
                                     max_turns: int = 5) -> str:
        """
        Get a summary of recent conversation history.
        
        Args:
            conversation_id: Conversation identifier
            max_turns: Maximum turns to include in summary
            
        Returns:
            Formatted conversation summary
        """
        context = await self.store.get_context(conversation_id)
        
        if not context or not context.turns:
            return "Aucun historique de conversation disponible."
        
        recent_turns = context.turns[-max_turns:]
        summary_lines = []
        
        for i, turn in enumerate(recent_turns, 1):
            user_preview = turn.user_message[:50] + "..." if len(turn.user_message) > 50 else turn.user_message
            assistant_preview = turn.assistant_response[:50] + "..." if len(turn.assistant_response) > 50 else turn.assistant_response
            
            summary_lines.append(f"{i}. Utilisateur: {user_preview}")
            summary_lines.append(f"   Assistant: {assistant_preview}")
            
            if turn.intent_detected:
                summary_lines.append(f"   Intent: {turn.intent_detected}")
        
        return "\n".join(summary_lines)
    
    async def get_active_conversations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get list of active conversations.
        
        Args:
            limit: Maximum number of conversations to return
            
        Returns:
            List of conversation summaries
        """
        active_conversations = []
        
        # Get recent conversations from user sessions
        recent_sessions = sorted(
            self.user_sessions.items(),
            key=lambda x: x[1]["last_activity"],
            reverse=True
        )[:limit]
        
        for conversation_id, session in recent_sessions:
            context = await self.store.get_context(conversation_id)
            
            if context:
                active_conversations.append({
                    "conversation_id": conversation_id,
                    "user_id": session["user_id"],
                    "start_time": session["start_time"].isoformat(),
                    "last_activity": session["last_activity"].isoformat(),
                    "message_count": session["message_count"],
                    "turn_count": len(context.turns),
                    "status": context.status,
                    "language": context.language,
                    "context_summary": context.context_summary
                })
        
        return active_conversations
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive conversation manager statistics.
        
        Returns:
            Dictionary containing all manager statistics
        """
        store_stats = await self.store.get_stats()
        
        # Calculate uptime
        uptime_seconds = (datetime.utcnow() - self.manager_stats["start_time"]).total_seconds()
        
        # Active session statistics
        active_sessions = len(self.user_sessions)
        recent_sessions = len([
            s for s in self.user_sessions.values()
            if (datetime.utcnow() - s["last_activity"]).total_seconds() < 3600  # Active in last hour
        ])
        
        return {
            "manager_overview": {
                "storage_backend": self.storage_backend,
                "max_conversations": self.max_conversations,
                "auto_cleanup_enabled": self.auto_cleanup_enabled,
                "uptime_seconds": round(uptime_seconds, 2)
            },
            "manager_statistics": {
                "context_retrievals": self.manager_stats["context_retrievals"],
                "turns_added": self.manager_stats["turns_added"],
                "contexts_cleared": self.manager_stats["contexts_cleared"],
                "user_sessions_created": self.manager_stats["user_sessions_created"],
                "active_sessions": active_sessions,
                "recent_sessions": recent_sessions
            },
            "storage_statistics": store_stats,
            "performance_metrics": {
                "operations_per_hour": round(
                    (self.manager_stats["context_retrievals"] + self.manager_stats["turns_added"]) / (uptime_seconds / 3600),
                    2
                ) if uptime_seconds > 0 else 0.0
            }
        }
    
    def _extract_active_entities(self, user_message: str, 
                               context: ConversationContext) -> Dict[str, Any]:
        """
        Extract entities that should remain active in context.
        
        Args:
            user_message: Current user message
            context: Current conversation context
            
        Returns:
            Dictionary of active entities
        """
        active_entities = {}
        
        # Simple entity extraction - in production, use more sophisticated methods
        message_lower = user_message.lower()
        
        # Look for date references
        date_keywords = ["janvier", "février", "mars", "avril", "mai", "juin",
                        "juillet", "août", "septembre", "octobre", "novembre", "décembre",
                        "mois", "année", "semaine", "jour", "hier", "aujourd'hui", "demain"]
        
        for keyword in date_keywords:
            if keyword in message_lower:
                active_entities["date_context"] = keyword
                break
        
        # Look for amount references
        if any(char.isdigit() for char in user_message):
            active_entities["amount_mentioned"] = True
        
        # Look for category references
        category_keywords = ["alimentation", "transport", "logement", "shopping", 
                           "restaurant", "essence", "courses", "factures"]
        
        for keyword in category_keywords:
            if keyword in message_lower:
                active_entities["category_context"] = keyword
                break
        
        return active_entities
    
    async def _background_cleanup(self) -> None:
        """Background task for periodic cleanup."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean up old user sessions
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                sessions_to_remove = [
                    conv_id for conv_id, session in self.user_sessions.items()
                    if session["last_activity"] < cutoff_time
                ]
                
                for conv_id in sessions_to_remove:
                    del self.user_sessions[conv_id]
                
                if sessions_to_remove:
                    logger.info(f"Cleaned up {len(sessions_to_remove)} old user sessions")
                
            except asyncio.CancelledError:
                logger.info("Background cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Background cleanup failed: {e}")
    
    async def close(self) -> None:
        """Close the conversation manager and clean up resources."""
        try:
            # Cancel background tasks
            if self.cleanup_task and not self.cleanup_task.done():
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Clear all data
            self.user_sessions.clear()
            
            logger.info("ConversationManager closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing ConversationManager: {e}")
    
    async def export_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Export conversation data for analysis or backup.
        
        Args:
            conversation_id: Conversation to export
            
        Returns:
            Complete conversation data or None if not found
        """
        context = await self.store.get_context(conversation_id)
        
        if not context:
            return None
        
        return {
            "conversation_id": conversation_id,
            "export_timestamp": datetime.utcnow().isoformat(),
            "context": context.dict(),
            "session_data": self.user_sessions.get(conversation_id),
            "statistics": {
                "total_turns": len(context.turns),
                "conversation_duration": (context.updated_at - context.created_at).total_seconds() if context.updated_at else 0,
                "intents_detected": [turn.intent_detected for turn in context.turns if turn.intent_detected],
                "average_processing_time": sum(turn.processing_time_ms for turn in context.turns) / len(context.turns) if context.turns else 0
            }
        }