"""
MVP Team Manager for coordinating the complete agent team.

This manager orchestrates the entire conversation service team, including
the orchestrator agent and all specialized agents. It provides high-level
API for conversation processing with comprehensive error handling, 
monitoring, and performance optimization.

Classes:
    - MVPTeamManager: Main team coordination manager
    - TeamHealth: Team health monitoring
    - TeamConfiguration: Team setup configuration

Author: Conversation Service Team
Created: 2025-01-31
Version: 1.0.0 MVP - Complete Team Management
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from datetime import datetime
from dataclasses import dataclass

from ..core.deepseek_client import DeepSeekClient
from ..agents.hybrid_intent_agent import HybridIntentAgent
from ..agents.search_query_agent import SearchQueryAgent
from ..agents.response_agent import ResponseAgent
from .conversation_manager import ConversationManager

if TYPE_CHECKING:
    from ..agents.orchestrator_agent import OrchestratorAgent
    from ..models.agent_models import AgentResponse

logger = logging.getLogger(__name__)


@dataclass
class TeamHealth:
    """Team health status information."""
    overall_healthy: bool
    agent_statuses: Dict[str, bool]
    last_health_check: datetime
    issues: List[str]
    performance_summary: Dict[str, Any]
    disabled_agents: List[str]
    agent_failure_counts: Dict[str, int]


@dataclass
class TeamConfiguration:
    """Team configuration parameters."""
    search_service_url: str
    enable_intent_caching: bool = True
    enable_response_caching: bool = True
    max_conversation_history: int = 100
    workflow_timeout_seconds: int = 60
    health_check_interval_seconds: int = 300
    auto_recovery_enabled: bool = True


class MVPTeamManager:
    """
    MVP Team Manager for the complete conversation service.
    
    This manager provides the high-level interface for conversation processing,
    coordinating all agents and providing comprehensive monitoring, error
    handling, and performance optimization.
    
    Attributes:
        config: Application settings
        team_config: Team-specific configuration
        deepseek_client: DeepSeek LLM client
        agents: Dictionary of initialized agents
        orchestrator: Main orchestrator agent
        conversation_manager: Conversation context manager
        team_health: Current team health status
        is_initialized: Whether the team is fully initialized
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, team_config: Optional[TeamConfiguration] = None):
        """
        Initialize the MVP team manager.
        
        Args:
            config: Optional configuration dictionary (from environment variables)
            team_config: Optional team configuration (uses defaults if None)
        """
        # Load configuration from environment variables
        self.config = config or self._load_config_from_env()
        self.team_config = team_config or TeamConfiguration(
            search_service_url=self.config.get('SEARCH_SERVICE_URL', 'http://localhost:8000/api/v1/search')
        )
        
        # Core components
        self.deepseek_client: Optional[DeepSeekClient] = None
        self.agents: Dict[str, Any] = {}
        self.orchestrator: Optional["OrchestratorAgent"] = None
        self.conversation_manager: Optional[ConversationManager] = None

        # Team status
        self.team_health: Optional[TeamHealth] = None
        self.is_initialized = False
        self.initialization_error: Optional[str] = None
        self.last_health_check = datetime.utcnow()
        self._delayed_health_check_task: Optional[asyncio.Task] = None
        self._initial_health_check_pending = False

        # Agent health tracking
        self.agent_failure_counts: Dict[str, int] = {}
        self.disabled_agents = set()
        self.failure_threshold = int(self.config.get('AGENT_FAILURE_THRESHOLD', 3))
        
        # Performance tracking
        self.team_stats = {
            "total_conversations": 0,
            "successful_conversations": 0,
            "failed_conversations": 0,
            "avg_response_time_ms": 0.0,
            "uptime_start": datetime.utcnow()
        }
        
        logger.info("Initialized MVPTeamManager")
    
    async def initialize_agents(self, initial_health_check: Optional[bool] = None) -> None:
        """
        Initialize all agents and team components.

        Args:
            initial_health_check: If ``True`` perform a health check at startup.
                If ``None``, the value is taken from the ``INITIAL_HEALTH_CHECK``
                environment variable (defaults to ``False``).

        Raises:
            Exception: If initialization fails or health check reports issues
        """
        try:
            logger.info("Starting team initialization...")

            # Step 1: Initialize DeepSeek client
            await self._initialize_deepseek_client()

            # Step 2: Initialize conversation manager
            await self._initialize_conversation_manager()

            # Step 3: Initialize specialized agents
            await self._initialize_specialized_agents()

            # Step 4: Initialize orchestrator
            await self._initialize_orchestrator()

            if initial_health_check is None:
                initial_health_check = os.getenv("INITIAL_HEALTH_CHECK", "false").lower() == "true"

            if initial_health_check:
                # Step 5: Initial health check
                await self._perform_health_check()

                # Fail fast if any component is unhealthy
                if not self.team_health or not self.team_health.overall_healthy:
                    issues = self.team_health.issues if self.team_health else ["Unknown health check failure"]
                    error_msg = f"Team health check failed: {issues}"
                    self.initialization_error = error_msg
                    logger.error(error_msg)
                    raise Exception(error_msg)
            else:
                # Schedule health check after first successful operation or delay
                self._initial_health_check_pending = True
                delay = int(self.config.get("INITIAL_HEALTH_CHECK_DELAY_SECONDS", 60))
                self._delayed_health_check_task = asyncio.create_task(
                    self._delayed_health_check(delay)
                )

            self.is_initialized = True
            logger.info("Team initialization completed successfully")

        except Exception as e:
            self.initialization_error = str(e)
            logger.error(f"Team initialization failed: {e}")
            raise Exception(f"Failed to initialize team: {e}")
    
    async def process_user_message(self, user_message: str, user_id: int,
                                 conversation_id: str) -> "AgentResponse":
        """
        Process a user message through the complete agent team.

        Args:
            user_message: User's input message
            user_id: User identifier
            conversation_id: Conversation identifier

        Returns:
            AgentResponse from the orchestrator containing content and metadata

        Raises:
            Exception: If team is not initialized or processing fails
        """
        if not self.is_initialized:
            raise Exception("Team not initialized. Call initialize_agents() first.")

        start_time = asyncio.get_event_loop().time()

        try:
            # Update conversation context with user info
            await self.conversation_manager.update_user_context(
                conversation_id, user_id, user_message
            )

            # Process through orchestrator
            response_data = await self.orchestrator.execute_with_metrics(
                {
                    "user_message": user_message,
                    "conversation_id": conversation_id,
                },
                user_id,
            )

            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            if response_data.success:
                self._update_team_stats(True, execution_time)
                logger.info(f"Successfully processed message for conversation {conversation_id}")
                self._trigger_initial_health_check_if_needed()
                await self._perform_health_check()
                return response_data
            elif response_data.content:
                # Workflow failure but a response was generated; keep team healthy
                logger.warning(
                    "Workflow reported failure but returned content; maintaining healthy state"
                )
                self._update_team_stats(True, execution_time)
                self._trigger_initial_health_check_if_needed()
                await self._perform_health_check()
                return response_data
            else:
                # Handle orchestrator failure without response content
                error_msg = response_data.error_message or "Orchestrator execution failed"
                logger.error(f"Orchestrator failed: {error_msg}")

                if not user_message.strip():
                    # Empty request: reset health and avoid marking as failure
                    self._update_team_stats(True, execution_time)
                    response_data.content = self._generate_error_response(user_message, error_msg)
                    await self._perform_health_check()
                    return response_data

                self._update_team_stats(False, execution_time)

                # Return graceful error response while preserving metadata
                response_data.content = self._generate_error_response(user_message, error_msg)
                await self._perform_health_check()
                return response_data

        except Exception as e:
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            self._update_team_stats(False, execution_time)

            logger.error(f"Message processing failed: {e}")

            from ..models.agent_models import AgentResponse

            return AgentResponse(
                agent_name="orchestrator_agent",
                content=self._generate_error_response(user_message, str(e)),
                metadata={},
                execution_time_ms=execution_time,
                success=False,
                error_message=str(e),
                confidence_score=0.1,
            )

    async def process_user_message_with_metadata(
        self, user_message: str, user_id: int, conversation_id: str
    ) -> Dict[str, Any]:
        """Process a user message and return response with agent metadata."""

        response = await self.process_user_message(
            user_message=user_message,
            user_id=user_id,
            conversation_id=conversation_id,
        )

        return {
            "content": response.content,
            "success": response.success,
            "confidence_score": response.confidence_score,
            "error_message": response.error_message,
            "metadata": {
                "intent_detected": response.metadata.get("intent_detected"),
                "entities_extracted": response.metadata.get("entities_extracted"),
                "agent_chain": response.metadata.get("agent_chain"),
                "search_results_count": response.metadata.get("search_results_count"),
            },
        }
    
    async def get_team_performance(self) -> Dict[str, Any]:
        """
        Get comprehensive team performance metrics.
        
        Returns:
            Dictionary containing all team performance data
        """
        if not self.is_initialized:
            return {"error": "Team not initialized"}
        
        # Aggregate metrics from all agents
        agent_metrics = {}
        for agent_name, agent in self.agents.items():
            try:
                agent_metrics[agent_name] = agent.get_performance_stats()
            except Exception as e:
                agent_metrics[agent_name] = {"error": str(e)}
        
        # Orchestrator metrics
        orchestrator_metrics = {}
        if self.orchestrator:
            try:
                orchestrator_metrics = self.orchestrator.get_workflow_stats()
            except Exception as e:
                orchestrator_metrics = {"error": str(e)}
        
        # Conversation manager metrics
        conversation_metrics = {}
        if self.conversation_manager:
            try:
                conversation_metrics = await self.conversation_manager.get_stats()
            except Exception as e:
                conversation_metrics = {"error": str(e)}
        
        # Team-level metrics
        uptime_seconds = (datetime.utcnow() - self.team_stats["uptime_start"]).total_seconds()
        total_conversations = self.team_stats["total_conversations"]
        success_rate = (
            self.team_stats["successful_conversations"] / total_conversations
            if total_conversations > 0 else 0.0
        )
        
        return {
            "team_overview": {
                "is_initialized": self.is_initialized,
                "initialization_error": self.initialization_error,
                "uptime_seconds": round(uptime_seconds, 2),
                "last_health_check": self.last_health_check.isoformat(),
                "team_healthy": self.team_health.overall_healthy if self.team_health else False
            },
            "team_statistics": {
                "total_conversations": total_conversations,
                "successful_conversations": self.team_stats["successful_conversations"],
                "failed_conversations": self.team_stats["failed_conversations"],
                "success_rate": round(success_rate, 4),
                "avg_response_time_ms": round(self.team_stats["avg_response_time_ms"], 2),
                "conversations_per_hour": round(total_conversations / (uptime_seconds / 3600), 2) if uptime_seconds > 0 else 0.0
            },
            "agent_performance": agent_metrics,
            "orchestrator_performance": orchestrator_metrics,
            "conversation_manager_performance": conversation_metrics,
            "team_health": self.team_health.__dict__ if self.team_health else None,
            "configuration": {
                "search_service_url": self.team_config.search_service_url,
                "workflow_timeout_seconds": self.team_config.workflow_timeout_seconds,
                "max_conversation_history": self.team_config.max_conversation_history
            }
        }
    
    async def shutdown(self) -> None:
        """
        Gracefully shutdown the team and all components.
        """
        logger.info("Starting team shutdown...")
        
        try:
            if self._delayed_health_check_task:
                self._delayed_health_check_task.cancel()
            # Close agents with async cleanup
            if self.agents.get("search_query_agent"):
                await self.agents["search_query_agent"].close()
            
            # Close conversation manager
            if self.conversation_manager:
                await self.conversation_manager.close()
            
            # Close DeepSeek client
            if self.deepseek_client:
                await self.deepseek_client.close()
            
            self.is_initialized = False
            logger.info("Team shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during team shutdown: {e}")
    
    def _load_config_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        return {
            'DEEPSEEK_API_KEY': os.getenv('DEEPSEEK_API_KEY', ''),
            'DEEPSEEK_BASE_URL': os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com'),
            'DEEPSEEK_TIMEOUT': int(os.getenv('DEEPSEEK_TIMEOUT', '30')),
            'SEARCH_SERVICE_URL': os.getenv('SEARCH_SERVICE_URL', 'http://localhost:8000/api/v1/search'),
            'MAX_CONVERSATION_HISTORY': int(os.getenv('MAX_CONVERSATION_HISTORY', '100')),
            'WORKFLOW_TIMEOUT_SECONDS': int(os.getenv('WORKFLOW_TIMEOUT_SECONDS', '60')),
            'HEALTH_CHECK_INTERVAL_SECONDS': int(os.getenv('HEALTH_CHECK_INTERVAL_SECONDS', '300')),
            'AUTO_RECOVERY_ENABLED': os.getenv('AUTO_RECOVERY_ENABLED', 'true').lower() == 'true',
            'INITIAL_HEALTH_CHECK_DELAY_SECONDS': int(os.getenv('INITIAL_HEALTH_CHECK_DELAY_SECONDS', '60')),
            'INITIAL_HEALTH_CHECK': os.getenv('INITIAL_HEALTH_CHECK', 'false').lower() == 'true',
            'AGENT_FAILURE_THRESHOLD': int(os.getenv('AGENT_FAILURE_THRESHOLD', '3'))
        }
    
    async def _initialize_deepseek_client(self) -> None:
        """Initialize the DeepSeek client."""
        try:
            self.deepseek_client = DeepSeekClient(
                api_key=self.config['DEEPSEEK_API_KEY'],
                base_url=self.config['DEEPSEEK_BASE_URL'],
                timeout=self.config['DEEPSEEK_TIMEOUT']
            )
            
            # Test connection
            await self.deepseek_client.health_check()
            logger.info("DeepSeek client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize DeepSeek client: {e}")
            raise
    
    async def _initialize_conversation_manager(self) -> None:
        """Initialize the conversation manager."""
        try:
            self.conversation_manager = ConversationManager(
                storage_backend="memory",  # MVP uses memory storage
                max_conversations=self.config['MAX_CONVERSATION_HISTORY']
            )
            
            await self.conversation_manager.initialize()
            logger.info("Conversation manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize conversation manager: {e}")
            raise
    
    async def _initialize_specialized_agents(self) -> None:
        """Initialize all specialized agents."""
        try:
            # Intent detection agent
            self.agents["intent_agent"] = HybridIntentAgent(
                deepseek_client=self.deepseek_client
            )
            
            # Search query agent
            self.agents["search_query_agent"] = SearchQueryAgent(
                deepseek_client=self.deepseek_client,
                search_service_url=self.team_config.search_service_url
            )
            
            # Response generation agent
            self.agents["response_agent"] = ResponseAgent(
                deepseek_client=self.deepseek_client
            )
            
            logger.info("Specialized agents initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize specialized agents: {e}")
            raise
    
    async def _initialize_orchestrator(self) -> None:
        """Initialize the orchestrator agent."""
        try:
            from ..agents.orchestrator_agent import OrchestratorAgent

            self.orchestrator = OrchestratorAgent(
                intent_agent=self.agents["intent_agent"],
                search_agent=self.agents["search_query_agent"],
                response_agent=self.agents["response_agent"]
            )

            logger.info("Orchestrator agent initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            raise
    
    async def _perform_health_check(self) -> None:
        """Perform comprehensive health check of all team components."""
        try:
            agent_statuses: Dict[str, bool] = {}
            issues: List[str] = []

            # Check each agent
            for agent_name, agent in self.agents.items():
                try:
                    is_healthy = agent.is_healthy()
                except Exception as e:
                    is_healthy = False
                    issues.append(f"Agent {agent_name} health check failed: {e}")

                if is_healthy:
                    self.agent_failure_counts[agent_name] = 0
                    if agent_name in self.disabled_agents:
                        self.disabled_agents.remove(agent_name)
                        logger.info(f"Agent {agent_name} reactivated after health recovery")
                    agent_statuses[agent_name] = True
                else:
                    count = self.agent_failure_counts.get(agent_name, 0) + 1
                    self.agent_failure_counts[agent_name] = count
                    if count >= self.failure_threshold:
                        agent_statuses[agent_name] = False
                        if agent_name not in self.disabled_agents:
                            self.disabled_agents.add(agent_name)
                            issues.append(
                                f"Agent {agent_name} disabled after {count} consecutive failures"
                            )
                    else:
                        agent_statuses[agent_name] = True
                        issues.append(
                            f"Agent {agent_name} failed health check ({count}/{self.failure_threshold})"
                        )

            # Check orchestrator separately
            if self.orchestrator:
                try:
                    orchestrator_healthy = self.orchestrator.is_healthy()
                except Exception as e:
                    orchestrator_healthy = False
                    issues.append(f"Orchestrator health check failed: {e}")

                name = "orchestrator"
                if orchestrator_healthy:
                    self.agent_failure_counts[name] = 0
                    if name in self.disabled_agents:
                        self.disabled_agents.remove(name)
                        logger.info("Orchestrator reactivated after health recovery")
                    agent_statuses[name] = True
                else:
                    count = self.agent_failure_counts.get(name, 0) + 1
                    self.agent_failure_counts[name] = count
                    if count >= self.failure_threshold:
                        agent_statuses[name] = False
                        if name not in self.disabled_agents:
                            self.disabled_agents.add(name)
                            issues.append(
                                f"Orchestrator disabled after {count} consecutive failures"
                            )
                    else:
                        agent_statuses[name] = True
                        issues.append(
                            f"Orchestrator failed health check ({count}/{self.failure_threshold})"
                        )

            # Overall health determined by disabled agents
            overall_healthy = len(self.disabled_agents) == 0

            # Performance summary
            performance_summary = {
                "total_agents": len(self.agents),
                "healthy_agents": sum(1 for status in agent_statuses.values() if status),
                "response_time_ms": self.team_stats["avg_response_time_ms"]
            }

            self.team_health = TeamHealth(
                overall_healthy=overall_healthy,
                agent_statuses=agent_statuses,
                last_health_check=datetime.utcnow(),
                issues=issues,
                performance_summary=performance_summary,
                disabled_agents=list(self.disabled_agents),
                agent_failure_counts=dict(self.agent_failure_counts)
            )

            self.last_health_check = datetime.utcnow()

            if overall_healthy:
                logger.info("Team health check passed")
            else:
                logger.warning(f"Team health check failed with issues: {issues}")

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self.team_health = TeamHealth(
                overall_healthy=False,
                agent_statuses={},
                last_health_check=datetime.utcnow(),
                issues=[f"Health check system error: {e}"],
                performance_summary={},
                disabled_agents=list(self.disabled_agents),
                agent_failure_counts=dict(self.agent_failure_counts)
            )
    
    async def _delayed_health_check(self, delay_seconds: int) -> None:
        """Run health check after a delay if still pending."""
        try:
            await asyncio.sleep(delay_seconds)
            if self._initial_health_check_pending:
                await self._perform_health_check()
                self._initial_health_check_pending = False
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Delayed health check failed: {e}")

    def _trigger_initial_health_check_if_needed(self) -> None:
        """Trigger pending initial health check after first successful operation."""
        if self._initial_health_check_pending:
            self._initial_health_check_pending = False
            if self._delayed_health_check_task:
                self._delayed_health_check_task.cancel()
            asyncio.create_task(self._perform_health_check())

    def _update_team_stats(self, success: bool, execution_time_ms: float) -> None:
        """Update team-level statistics."""
        self.team_stats["total_conversations"] += 1
        
        if success:
            self.team_stats["successful_conversations"] += 1
        else:
            self.team_stats["failed_conversations"] += 1
        
        # Update running average of response time
        total = self.team_stats["total_conversations"]
        current_avg = self.team_stats["avg_response_time_ms"]
        self.team_stats["avg_response_time_ms"] = (
            (current_avg * (total - 1) + execution_time_ms) / total
        )
    
    def _generate_error_response(self, user_message: str, error: str) -> str:
        """Generate a graceful error response for users."""
        return f"Je rencontre actuellement des difficultés techniques pour traiter votre demande concernant '{user_message[:50]}...'. Veuillez réessayer dans quelques instants. Si le problème persiste, contactez le support technique."
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform and return current health status.

        Returns:
            Dictionary containing current health status
        """
        await self._perform_health_check()

        details = self.team_health.__dict__.copy() if self.team_health else None
        if details and details.get("last_health_check"):
            details["last_health_check"] = details["last_health_check"].isoformat()
            details["failure_threshold"] = self.failure_threshold
            details["recovery_policy"] = (
                f"Agents disabled after {self.failure_threshold} consecutive failures"
                " and automatically reactivated when healthy."
            )

        return {
            "healthy": self.team_health.overall_healthy if self.team_health else False,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details
        }

    async def get_health_status(self) -> Dict[str, Any]:
        """
        Backward compatible wrapper for :meth:`health_check`.

        Returns:
            Current health status information.
        """
        return await self.health_check()
    
    def is_healthy(self) -> bool:
        """
        Quick health check.
        
        Returns:
            True if team is healthy, False otherwise
        """
        return (
            self.is_initialized and
            (
                self.team_health is None or
                self.team_health.overall_healthy
            )
        )
