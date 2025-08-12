"""
Base Financial Agent class for AutoGen v0.4 integration.

This module provides the foundational class for all specialized financial agents
in the conversation service. It includes common functionality like metrics tracking,
error handling, and DeepSeek client integration.

Classes:
    - BaseFinancialAgent: Base class extending AutoGen AssistantAgent
    - AgentMetrics: Performance metrics tracking for agents

Author: Conversation Service Team
Created: 2025-01-31
Version: 1.0.0 MVP - AutoGen v0.4 + DeepSeek
"""

import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict, deque

try:
    from autogen import AssistantAgent
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    # Fallback base class for when AutoGen is not available
    class AssistantAgent:
        def __init__(self, *args, **kwargs):
            pass

from ..models.agent_models import AgentConfig, AgentResponse
from ..core.deepseek_client import DeepSeekClient

logger = logging.getLogger(__name__)


class AgentMetrics:
    """
    Performance metrics tracking for financial agents.
    
    Tracks execution times, success rates, token usage, and error patterns
    to enable monitoring and optimization of agent performance.
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize metrics tracking.
        
        Args:
            window_size: Number of recent operations to keep in memory
        """
        self.window_size = window_size
        self.reset_metrics()
    
    def reset_metrics(self) -> None:
        """Reset all metrics to initial state."""
        self.total_operations = 0
        self.successful_operations = 0
        self.failed_operations = 0
        self.total_execution_time = 0.0
        self.total_tokens_used = 0
        
        # Recent operations for rolling averages
        self.recent_execution_times = deque(maxlen=self.window_size)
        self.recent_tokens = deque(maxlen=self.window_size)
        self.recent_errors = deque(maxlen=self.window_size)
        
        # Error tracking
        self.error_counts = defaultdict(int)
        self.last_error = None
        self.last_success_time = None
        
        # Performance tracking
        self.min_execution_time = float('inf')
        self.max_execution_time = 0.0
        self.start_time = datetime.utcnow()
    
    def record_operation(self, execution_time: float, success: bool, 
                        tokens_used: int = 0, error_type: Optional[str] = None) -> None:
        """
        Record a completed operation.
        
        Args:
            execution_time: Time taken to execute operation in milliseconds
            success: Whether the operation succeeded
            tokens_used: Number of tokens consumed
            error_type: Type of error if operation failed
        """
        self.total_operations += 1
        self.total_execution_time += execution_time
        self.total_tokens_used += tokens_used
        
        # Update min/max execution times
        self.min_execution_time = min(self.min_execution_time, execution_time)
        self.max_execution_time = max(self.max_execution_time, execution_time)
        
        # Record in recent windows
        self.recent_execution_times.append(execution_time)
        self.recent_tokens.append(tokens_used)
        
        if success:
            self.successful_operations += 1
            self.last_success_time = datetime.utcnow()
        else:
            self.failed_operations += 1
            if error_type:
                self.error_counts[error_type] += 1
                self.last_error = error_type
                self.recent_errors.append(error_type)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive performance statistics.
        
        Returns:
            Dictionary containing all performance metrics
        """
        uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
        
        # Calculate averages
        avg_execution_time = (
            self.total_execution_time / self.total_operations 
            if self.total_operations > 0 else 0.0
        )
        
        recent_avg_time = (
            sum(self.recent_execution_times) / len(self.recent_execution_times)
            if self.recent_execution_times else 0.0
        )
        
        recent_avg_tokens = (
            sum(self.recent_tokens) / len(self.recent_tokens)
            if self.recent_tokens else 0.0
        )
        
        success_rate = (
            self.successful_operations / self.total_operations
            if self.total_operations > 0 else 0.0
        )
        
        operations_per_hour = (
            self.total_operations / (uptime_seconds / 3600)
            if uptime_seconds > 0 else 0.0
        )
        
        return {
            "total_operations": self.total_operations,
            "successful_operations": self.successful_operations,
            "failed_operations": self.failed_operations,
            "success_rate": round(success_rate, 4),
            "uptime_seconds": round(uptime_seconds, 2),
            "operations_per_hour": round(operations_per_hour, 2),
            "execution_times": {
                "average_ms": round(avg_execution_time, 2),
                "recent_average_ms": round(recent_avg_time, 2),
                "min_ms": round(self.min_execution_time, 2) if self.min_execution_time != float('inf') else 0.0,
                "max_ms": round(self.max_execution_time, 2)
            },
            "token_usage": {
                "total_tokens": self.total_tokens_used,
                "average_tokens": round(self.total_tokens_used / self.total_operations, 2) if self.total_operations > 0 else 0.0,
                "recent_average_tokens": round(recent_avg_tokens, 2)
            },
            "errors": {
                "error_counts": dict(self.error_counts),
                "last_error": self.last_error,
                "recent_error_rate": len(self.recent_errors) / len(self.recent_execution_times) if self.recent_execution_times else 0.0
            },
            "last_success_time": self.last_success_time.isoformat() if self.last_success_time else None
        }


class BaseFinancialAgent(AssistantAgent):
    """
    Base class for all financial agents in the conversation service.
    
    This class extends AutoGen's AssistantAgent with financial domain-specific
    functionality, including DeepSeek integration, metrics tracking, and
    standardized error handling.
    
    Attributes:
        name: Unique identifier for the agent
        config: Agent configuration
        deepseek_client: DeepSeek LLM client
        metrics: Performance metrics tracker
        domain: Financial domain context
    """
    
    def __init__(self, name: str, config: AgentConfig, deepseek_client: DeepSeekClient):
        """
        Initialize the base financial agent.
        
        Args:
            name: Unique identifier for the agent
            config: Agent configuration including model settings
            deepseek_client: Configured DeepSeek client instance
        """
        if not AUTOGEN_AVAILABLE:
            raise ImportError("AutoGen is required but not available")
        
        # Initialize AutoGen AssistantAgent
        super().__init__(
            name=name,
            system_message=config.system_message,
            llm_config={
                "model": config.model_client_config["model"],
                "api_key": config.model_client_config["api_key"],
                "base_url": config.model_client_config["base_url"],
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "timeout": config.timeout_seconds
            },
            max_consecutive_auto_reply=config.max_consecutive_auto_reply,
            description=config.description
        )
        
        self.config = config
        self.deepseek_client = deepseek_client
        self.metrics = AgentMetrics()
        self.domain = "financial"
        
        logger.info(f"Initialized {self.__class__.__name__} agent: {name}")
    
    async def execute_with_metrics(self, input_data: Dict[str, Any], user_id: int) -> AgentResponse:
        """
        Execute agent operation with comprehensive metrics tracking.

        This method wraps the agent's core functionality with timing,
        error handling, and metrics collection.

        Args:
            input_data: Input data for the agent operation
            user_id: ID of the requesting user

        Returns:
            AgentResponse with result and metadata

        Raises:
            Exception: If agent execution fails and retry limit exceeded
        """
        start_time = time.perf_counter()
        success = False
        error_message = None
        tokens_used = 0

        try:
            # Execute the agent's specific logic
            result = await self._execute_operation(input_data, user_id)
            
            # Extract token usage if available
            if isinstance(result, dict) and "token_usage" in result:
                tokens_used = result["token_usage"].get("total_tokens", 0)
            
            success = True
            execution_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
            
            # Record successful operation
            self.metrics.record_operation(
                execution_time=execution_time,
                success=True,
                tokens_used=tokens_used
            )
            
            return AgentResponse(
                agent_name=self.name,
                content=result.get("content", str(result)) if isinstance(result, dict) else str(result),
                metadata=result.get("metadata", {}) if isinstance(result, dict) else {},
                execution_time_ms=execution_time,
                success=True,
                token_usage=result.get("token_usage") if isinstance(result, dict) else None,
                confidence_score=result.get("confidence_score") if isinstance(result, dict) else None
            )
            
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            error_message = str(e)
            error_type = type(e).__name__
            
            # Record failed operation
            self.metrics.record_operation(
                execution_time=execution_time,
                success=False,
                error_type=error_type
            )
            
            logger.error(f"Agent {self.name} execution failed: {error_message}")
            
            return AgentResponse(
                agent_name=self.name,
                content="",
                metadata={"error": error_message, "error_type": error_type},
                execution_time_ms=execution_time,
                success=False,
                error_message=error_message
            )
    
    async def _execute_operation(self, input_data: Dict[str, Any], user_id: int) -> Any:
        """
        Execute the agent's specific operation logic.

        This method should be overridden by concrete agent implementations
        to provide their specialized functionality.

        Args:
            input_data: Input data for the operation
            user_id: ID of the requesting user

        Returns:
            Result of the agent operation

        Raises:
            NotImplementedError: If not overridden by concrete class
        """
        raise NotImplementedError("Concrete agents must implement _execute_operation")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get agent performance statistics.
        
        Returns:
            Comprehensive performance metrics for this agent
        """
        base_stats = self.metrics.get_performance_stats()
        base_stats.update({
            "agent_name": self.name,
            "agent_type": self.__class__.__name__,
            "domain": self.domain,
            "config": {
                "model": self.config.model_client_config["model"],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "timeout_seconds": self.config.timeout_seconds
            }
        })
        return base_stats
    
    def _log_execution(self, operation: str, duration: float, success: bool = True, 
                      additional_info: Optional[Dict] = None) -> None:
        """
        Log agent execution details.
        
        Args:
            operation: Name of the operation performed
            duration: Execution duration in milliseconds
            success: Whether the operation succeeded
            additional_info: Additional information to log
        """
        log_data = {
            "agent": self.name,
            "operation": operation,
            "duration_ms": round(duration, 2),
            "success": success
        }
        
        if additional_info:
            log_data.update(additional_info)
        
        if success:
            logger.info(f"Agent execution completed: {log_data}")
        else:
            logger.warning(f"Agent execution failed: {log_data}")
    
    def reset_metrics(self) -> None:
        """Reset agent performance metrics."""
        self.metrics.reset_metrics()
        logger.info(f"Reset metrics for agent: {self.name}")
    
    def is_healthy(self) -> bool:
        """
        Check if agent is healthy based on recent performance.

        The agent starts in a healthy state and remains so until it
        completes its first successful operation. This avoids the agent
        being marked as unhealthy immediately after startup before any
        executions have occurred.

        Returns:
            True if agent is performing well, False otherwise
        """
        stats = self.metrics.get_performance_stats()

        # Consider the agent healthy until at least one operation succeeds
        if stats["successful_operations"] == 0:
            return True

        # Check basic health indicators
        recent_success_rate = 1.0 - stats["errors"]["recent_error_rate"]
        recent_avg_time = stats["execution_times"]["recent_average_ms"]

        # Agent is healthy if:
        # - Recent success rate > 90%
        # - Recent average execution time < 10 seconds
        return (
            recent_success_rate > 0.9 and
            recent_avg_time < 10000
        )
