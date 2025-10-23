"""
Redis-based conversation memory cache for conversation_service_v3
Manages short-term conversation history with token limit control
"""
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
import redis
from redis.exceptions import RedisError

from ..utils.token_counter import TokenCounter

logger = logging.getLogger(__name__)


class ConversationMessage:
    """Single message in conversation history"""

    def __init__(
        self,
        role: str,
        content: str,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.role = role  # "user" or "assistant"
        self.content = content
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationMessage":
        """Create from dictionary"""
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=data.get("timestamp"),
            metadata=data.get("metadata", {})
        )

    def to_llm_format(self) -> Dict[str, str]:
        """Convert to OpenAI chat format"""
        return {
            "role": self.role,
            "content": self.content
        }


class RedisConversationCache:
    """
    Redis-based cache for conversation history

    Features:
    - Stores conversation history with automatic TTL
    - Token limit control to prevent context overflow
    - Sliding window of N recent messages
    - Efficient retrieval for LLM context
    """

    def __init__(
        self,
        redis_url: str,
        max_messages_per_conversation: int = 10,
        max_context_tokens: int = 4000,
        cache_ttl_seconds: int = 86400,  # 24 hours
        token_counter: Optional[TokenCounter] = None
    ):
        """
        Initialize Redis conversation cache

        Args:
            redis_url: Redis connection URL (redis://...)
            max_messages_per_conversation: Maximum messages to keep per conversation
            max_context_tokens: Maximum tokens for LLM context
            cache_ttl_seconds: TTL for cached conversations (default: 24h)
            token_counter: Optional token counter instance
        """
        self.redis_url = redis_url
        self.max_messages = max_messages_per_conversation
        self.max_context_tokens = max_context_tokens
        self.cache_ttl_seconds = cache_ttl_seconds

        # Initialize Redis client
        try:
            self.redis_client = redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"✅ Redis conversation cache connected: {redis_url}")
        except RedisError as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise

        # Token counter for context size management
        self.token_counter = token_counter or TokenCounter()

    def _get_cache_key(self, user_id: int, conversation_id: Optional[int] = None) -> str:
        """
        Generate Redis key for conversation

        Args:
            user_id: User ID
            conversation_id: Optional conversation ID

        Returns:
            Redis key string
        """
        if conversation_id:
            return f"conv_history:{user_id}:{conversation_id}"
        else:
            # Fallback: user's default/latest conversation
            return f"conv_history:{user_id}:latest"

    def add_message(
        self,
        user_id: int,
        role: str,
        content: str,
        conversation_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a message to conversation history

        Args:
            user_id: User ID
            role: Message role ("user" or "assistant")
            content: Message content
            conversation_id: Optional conversation ID
            metadata: Optional metadata

        Returns:
            True if successful
        """
        try:
            key = self._get_cache_key(user_id, conversation_id)

            # Create message
            message = ConversationMessage(
                role=role,
                content=content,
                metadata=metadata
            )

            # Get current history
            history = self._get_history_raw(key)

            # Add new message
            history.append(message.to_dict())

            # Enforce message limit (sliding window)
            if len(history) > self.max_messages:
                history = history[-self.max_messages:]

            # Save back to Redis with TTL
            self.redis_client.setex(
                key,
                self.cache_ttl_seconds,
                json.dumps(history)
            )

            logger.debug(f"📝 Message added to cache: {key} (total: {len(history)} messages)")
            return True

        except RedisError as e:
            logger.error(f"❌ Failed to add message to cache: {e}")
            return False

    def get_conversation_history(
        self,
        user_id: int,
        conversation_id: Optional[int] = None,
        max_tokens: Optional[int] = None,
        include_system_message: bool = False,
        system_message_content: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Get conversation history for LLM context

        Args:
            user_id: User ID
            conversation_id: Optional conversation ID
            max_tokens: Override default max_tokens
            include_system_message: Add system message at start
            system_message_content: Custom system message content

        Returns:
            List of messages in OpenAI format [{"role": "user", "content": "..."}]
        """
        try:
            key = self._get_cache_key(user_id, conversation_id)
            history_raw = self._get_history_raw(key)

            if not history_raw:
                logger.debug(f"📭 No history found for {key}")
                return self._create_system_messages(include_system_message, system_message_content)

            # Convert to ConversationMessage objects
            messages = [ConversationMessage.from_dict(msg) for msg in history_raw]

            # Convert to LLM format
            llm_messages = [msg.to_llm_format() for msg in messages]

            # Add system message if requested
            if include_system_message:
                system_msg = {
                    "role": "system",
                    "content": system_message_content or self._get_default_system_message()
                }
                llm_messages.insert(0, system_msg)

            # Apply token limit
            token_limit = max_tokens or self.max_context_tokens
            truncated_messages = self.token_counter.truncate_messages_to_limit(
                messages=llm_messages,
                max_tokens=token_limit,
                keep_system_message=include_system_message
            )

            total_tokens = self.token_counter.count_messages_tokens(truncated_messages)
            logger.debug(
                f"📖 Retrieved {len(truncated_messages)} messages "
                f"({total_tokens}/{token_limit} tokens) for {key}"
            )

            return truncated_messages

        except RedisError as e:
            logger.error(f"❌ Failed to get conversation history: {e}")
            return self._create_system_messages(include_system_message, system_message_content)

    def _get_history_raw(self, key: str) -> List[Dict[str, Any]]:
        """Get raw history from Redis"""
        try:
            data = self.redis_client.get(key)
            if data:
                return json.loads(data)
            return []
        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"❌ Failed to parse history from Redis: {e}")
            return []

    def _create_system_messages(
        self,
        include_system_message: bool,
        system_message_content: Optional[str]
    ) -> List[Dict[str, str]]:
        """Create system message list"""
        if include_system_message:
            return [{
                "role": "system",
                "content": system_message_content or self._get_default_system_message()
            }]
        return []

    def _get_default_system_message(self) -> str:
        """Default system message for Harena"""
        return (
            "Vous êtes Harena, un assistant financier intelligent qui aide les utilisateurs "
            "à comprendre et gérer leurs finances personnelles. Vous analysez leurs transactions "
            "et fournissez des insights clairs et actionnables."
        )

    def clear_conversation(
        self,
        user_id: int,
        conversation_id: Optional[int] = None
    ) -> bool:
        """
        Clear conversation history from cache

        Args:
            user_id: User ID
            conversation_id: Optional conversation ID

        Returns:
            True if successful
        """
        try:
            key = self._get_cache_key(user_id, conversation_id)
            self.redis_client.delete(key)
            logger.info(f"🗑️ Cleared conversation cache: {key}")
            return True
        except RedisError as e:
            logger.error(f"❌ Failed to clear conversation: {e}")
            return False

    def get_conversation_stats(
        self,
        user_id: int,
        conversation_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get statistics about cached conversation

        Args:
            user_id: User ID
            conversation_id: Optional conversation ID

        Returns:
            Dictionary with stats (message_count, total_tokens, ttl_seconds)
        """
        try:
            key = self._get_cache_key(user_id, conversation_id)
            history = self._get_history_raw(key)

            if not history:
                return {
                    "message_count": 0,
                    "total_tokens": 0,
                    "ttl_seconds": 0,
                    "exists": False
                }

            # Convert to LLM format for token counting
            messages = [ConversationMessage.from_dict(msg).to_llm_format() for msg in history]
            total_tokens = self.token_counter.count_messages_tokens(messages)

            # Get TTL
            ttl = self.redis_client.ttl(key)

            return {
                "message_count": len(history),
                "total_tokens": total_tokens,
                "ttl_seconds": ttl,
                "exists": True,
                "within_limit": total_tokens <= self.max_context_tokens
            }

        except RedisError as e:
            logger.error(f"❌ Failed to get conversation stats: {e}")
            return {"error": str(e)}

    def update_ttl(
        self,
        user_id: int,
        conversation_id: Optional[int] = None,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """
        Update TTL for conversation cache

        Args:
            user_id: User ID
            conversation_id: Optional conversation ID
            ttl_seconds: New TTL in seconds (default: use cache_ttl_seconds)

        Returns:
            True if successful
        """
        try:
            key = self._get_cache_key(user_id, conversation_id)
            ttl = ttl_seconds or self.cache_ttl_seconds
            self.redis_client.expire(key, ttl)
            logger.debug(f"⏱️ Updated TTL for {key}: {ttl}s")
            return True
        except RedisError as e:
            logger.error(f"❌ Failed to update TTL: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """
        Check Redis connection health

        Returns:
            Health status dictionary
        """
        try:
            self.redis_client.ping()
            return {
                "status": "healthy",
                "redis_connected": True,
                "max_messages": self.max_messages,
                "max_context_tokens": self.max_context_tokens,
                "cache_ttl_seconds": self.cache_ttl_seconds
            }
        except RedisError as e:
            return {
                "status": "unhealthy",
                "redis_connected": False,
                "error": str(e)
            }
