"""
Orchestrator Agent for coordinating multi-agent workflows.

This agent coordinates the execution of the three specialized agents
(Intent, Search Query, and Response) to provide a complete conversation
processing pipeline with error handling and fallback mechanisms.

Classes:
    - OrchestratorAgent: Main workflow coordination agent
    - WorkflowExecutor: Helper class for workflow execution
    - WorkflowStep: Individual workflow step representation

Author: Conversation Service Team
Created: 2025-01-31
Version: 1.0.0 MVP - Multi-Agent Orchestration
"""

import time
import logging
from typing import Dict, Any, Optional, List, Deque
from dataclasses import dataclass
from enum import Enum
from collections import deque

from .base_financial_agent import BaseFinancialAgent
from .llm_intent_agent import LLMIntentAgent
from .search_query_agent import SearchQueryAgent
from .response_agent import ResponseAgent
from ..models.agent_models import AgentConfig, AgentResponse
from types import SimpleNamespace
from ..models.conversation_models import ConversationContext
from ..core.deepseek_client import DeepSeekClient
from ..core.conversation_manager import ConversationManager
from ..utils.metrics import get_default_metrics_collector

logger = logging.getLogger(__name__)


class WorkflowStepStatus(Enum):
    """Status of individual workflow steps."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """Represents a single step in the workflow."""
    name: str
    agent_name: str
    status: WorkflowStepStatus = WorkflowStepStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[AgentResponse] = None
    error: Optional[str] = None
    
    @property
    def duration_ms(self) -> float:
        """Get step duration in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0


class WorkflowExecutor:
    """Helper class for executing multi-agent workflows."""

    def __init__(self, intent_agent: LLMIntentAgent,
                 search_agent: SearchQueryAgent,
                 response_agent: ResponseAgent):
        """Initialize workflow executor."""
        self.intent_agent = intent_agent
        self.search_agent = search_agent
        self.response_agent = response_agent
        self.conversation_manager = ConversationManager()

    def _build_performance_summary(self, steps: List[WorkflowStep]) -> Dict[str, int]:
        """Create a performance summary from workflow steps."""
        return {
            "completed_steps": len([s for s in steps if s.status == WorkflowStepStatus.COMPLETED]),
            "failed_steps": len([s for s in steps if s.status == WorkflowStepStatus.FAILED]),
            "skipped_steps": len([s for s in steps if s.status == WorkflowStepStatus.SKIPPED]),
            "total_steps": len(steps),
        }
        
    async def execute_workflow(
        self, user_message: str, conversation_id: str, user_id: int
    ) -> Dict[str, Any]:
        """
        Execute the complete 3-agent workflow.

        Args:
            user_message: User's input message
            conversation_id: Conversation identifier
            user_id: ID of the requesting user

        Returns:
            Dict with workflow execution results including:
                - ``success``: Overall workflow success flag
                - ``final_response``: Final response string (may be fallback)
                - ``workflow_data``: Intermediate data collected during steps
                - ``execution_details``: Timing and status for each step
                - ``performance_summary``: Summary of completed and failed
                  steps. Always present, even when the workflow fails
                  entirely.
        """
        workflow_start = time.perf_counter()
        metrics = get_default_metrics_collector()
        
        # Initialize workflow steps
        steps = [
            WorkflowStep("intent_detection", self.intent_agent.name),
            WorkflowStep("search_query", self.search_agent.name),
            WorkflowStep("response_generation", self.response_agent.name)
        ]
        
        workflow_data = {
            "user_message": user_message,
            "conversation_id": conversation_id,
            "intent_result": None,
            "search_results": None,
            "final_response": None
        }
        
        try:
            # Step 1: Intent Detection
            intent_step = steps[0]
            logger.info("Starting intent_detection step")
            intent_step.status = WorkflowStepStatus.RUNNING
            intent_step.start_time = time.perf_counter()
            intent_timer = metrics.performance_monitor.start_timer("intent_detection")
            
            try:
                intent_response = await self.intent_agent.execute_with_metrics(
                    {"user_message": user_message}, user_id
                )
                intent_result = (
                    intent_response.metadata.get("intent_result")
                    if intent_response.metadata
                    else None
                )
                logger.info(
                    f"Intent: {intent_result.intent_type}, "
                    f"Entities: {[e.model_dump() for e in getattr(intent_result, 'entities', [])]}"
                )

                if intent_response.success:
                    workflow_data["intent_result"] = intent_result
                    intent_step.status = WorkflowStepStatus.COMPLETED
                    intent_step.result = intent_response
                else:
                    raise Exception(intent_response.error_message or "Intent detection failed")
                    
            except Exception as e:
                intent_step.status = WorkflowStepStatus.FAILED
                intent_step.error = str(e)
                logger.error(f"Intent detection step failed: {e}")
                
                # Try to continue with fallback intent
                workflow_data["intent_result"] = self._create_fallback_intent()
            
            finally:
                intent_step.end_time = time.perf_counter()
                duration_ms = metrics.performance_monitor.end_timer(intent_timer)
                metrics.record_timer("intent_detection_duration_ms", duration_ms)
                for alert in metrics.performance_monitor.check_performance_alerts("intent_detection"):
                    logger.warning(alert.message)
                entities_count = 0
                if workflow_data["intent_result"] and getattr(workflow_data["intent_result"], "entities", None):
                    entities_count = len(workflow_data["intent_result"].entities)
                logger.info(
                    "Finished intent_detection step in %.2f ms with %d entities",
                    intent_step.duration_ms,
                    entities_count,
                )
            
            # Step 2: Search Query (only if we have intent)
            search_step = steps[1]
            response_step = steps[2]
            intent_result = workflow_data["intent_result"]
            if intent_result:
                search_required = getattr(intent_result, "search_required", True)
                if search_required:
                    logger.info("Starting search_query step")
                    search_step.status = WorkflowStepStatus.RUNNING
                    search_step.start_time = time.perf_counter()
                    search_timer = metrics.performance_monitor.start_timer("search_query")
                    try:
                        search_response = await self.search_agent.execute_with_metrics(
                            {"intent_result": intent_result, "user_message": user_message},
                            user_id,
                        )
                        if search_response.success:
                            # Store complete search response object for downstream agents
                            workflow_data["search_results"] = search_response
                            search_step.status = WorkflowStepStatus.COMPLETED
                            search_step.result = search_response
                        else:
                            raise Exception(search_response.error_message or "Search query failed")
                    except Exception as e:
                        search_step.status = WorkflowStepStatus.FAILED
                        search_step.error = str(e)
                        logger.error(f"Search query step failed: {e}")
                        workflow_data["search_results"] = self._create_empty_search_results()
                    finally:
                        search_step.end_time = time.perf_counter()
                        duration_ms = metrics.performance_monitor.end_timer(search_timer)
                        metrics.record_timer("search_query_duration_ms", duration_ms)
                        for alert in metrics.performance_monitor.check_performance_alerts("search_query"):
                            logger.warning(alert.message)
                        results_count = 0
                        if search_step.result and getattr(search_step.result, "metadata", None):
                            results_count = search_step.result.metadata.get("search_results_count", 0)
                        logger.info(
                            "Finished search_query step in %.2f ms with %d results",
                            search_step.duration_ms,
                            results_count,
                        )
                else:
                    search_step.status = WorkflowStepStatus.SKIPPED
                    search_step.error = "Search not required for intent"
                    search_step.start_time = time.perf_counter()
                    search_step.end_time = time.perf_counter()
                    workflow_data["search_results"] = self._create_empty_search_results()
                    if getattr(intent_result, "suggested_actions", None):
                        workflow_data["final_response"] = intent_result.suggested_actions[0]
                    else:
                        workflow_data["final_response"] = "Bonjour !"
                    response_step.status = WorkflowStepStatus.SKIPPED
                    response_step.error = "Search not required for intent"
                    response_step.start_time = time.perf_counter()
                    response_step.end_time = time.perf_counter()
                    logger.info("Search_query step skipped")
                    logger.info("Response_generation step skipped")
            else:
                search_step.status = WorkflowStepStatus.SKIPPED
                search_step.error = "No valid intent result"
                workflow_data["search_results"] = self._create_empty_search_results()
                logger.info("Search_query step skipped - no intent result")

            # Step 3: Response Generation
            if response_step.status == WorkflowStepStatus.PENDING:
                logger.info("Starting response_generation step")
                response_step.status = WorkflowStepStatus.RUNNING
                response_step.start_time = time.perf_counter()
                response_timer = metrics.performance_monitor.start_timer("response_generation")
                try:
                    context = await self._create_conversation_context(
                        conversation_id
                    )
                    logger.info(
                        "Conversation context size before response: %d",
                        len(context.turns) if context else 0,
                    )
                    response_response = await self.response_agent.execute_with_metrics(
                        {
                            "user_message": user_message,
                            "search_results": workflow_data["search_results"],
                            "context": context,
                        },
                        user_id,
                    )
                    if response_response.success:
                        workflow_data["final_response"] = response_response.content
                        response_step.status = WorkflowStepStatus.COMPLETED
                        response_step.result = response_response
                    else:
                        raise Exception(response_response.error_message or "Response generation failed")
                except Exception as e:
                    response_step.status = WorkflowStepStatus.FAILED
                    response_step.error = str(e)
                    logger.error(f"Response generation step failed: {e}")
                    workflow_data["final_response"] = self._create_fallback_response(user_message)
                finally:
                    response_step.end_time = time.perf_counter()
                    duration_ms = metrics.performance_monitor.end_timer(response_timer)
                    metrics.record_timer("response_generation_duration_ms", duration_ms)
                    for alert in metrics.performance_monitor.check_performance_alerts("response_generation"):
                        logger.warning(alert.message)
                    results_count = 0
                    if workflow_data.get("search_results") and getattr(workflow_data["search_results"], "metadata", None):
                        results_count = workflow_data["search_results"].metadata.get("search_results_count", 0)
                    logger.info(
                        "Finished response_generation step in %.2f ms using %d results",
                        response_step.duration_ms,
                        results_count,
                    )
            
            # Compile workflow results
            workflow_end = time.perf_counter()
            total_duration = (workflow_end - workflow_start) * 1000
            
            return {
                "success": any(step.status == WorkflowStepStatus.COMPLETED for step in steps),
                "final_response": workflow_data["final_response"],
                "workflow_data": workflow_data,
                "execution_details": {
                    "total_duration_ms": total_duration,
                    "steps": [
                        {
                            "name": step.name,
                            "agent": step.agent_name,
                            "status": step.status.value,
                            "duration_ms": step.duration_ms,
                            "error": step.error
                        }
                        for step in steps
                    ]
                },
                "performance_summary": self._build_performance_summary(steps)
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                "success": False,
                "final_response": f"Je rencontre des difficultés techniques. Erreur: {str(e)}",
                "workflow_data": workflow_data,
                "error": str(e),
                "execution_details": {
                    "total_duration_ms": (time.perf_counter() - workflow_start) * 1000,
                    "steps": [
                        {
                            "name": step.name,
                            "agent": step.agent_name,
                            "status": step.status.value,
                            "duration_ms": step.duration_ms,
                            "error": step.error
                        }
                        for step in steps
                    ]
                },
                "performance_summary": self._build_performance_summary(steps)
            }
    
    def _create_fallback_intent(self) -> Dict[str, Any]:
        """Create a fallback intent when detection fails."""
        from ..models.financial_models import (
            IntentResult,
            IntentCategory,
            DetectionMethod,
        )
        
        return IntentResult(
            intent_type="GENERAL",
            intent_category=IntentCategory.GENERAL_QUESTION,
            confidence=0.3,
            entities=[],
            method=DetectionMethod.FALLBACK,
            processing_time_ms=0.0
        )
    
    def _create_empty_search_results(self) -> AgentResponse:
        """Create empty search results for fallback."""
        empty_search_response = {
            "response_metadata": {
                "query_id": f"fallback_{int(time.time())}",
                "processing_time_ms": 0,
                "returned_results": 0,
                "total_results": 0,
                "has_more_results": False,
                "cache_hit": False,
                "search_strategy_used": "none",
            },
            "results": [],
            "aggregations": None,
        }

        metadata = {
            "search_response": empty_search_response,
            "search_query": None,
            "enhanced_entities": {},
            "execution_time_ms": 0,
            "search_results_count": 0,
        }

        return AgentResponse(
            agent_name=self.search_agent.name,
            content="Search completed: 0 results",
            metadata=metadata,
            execution_time_ms=0,
            success=True,
        )
    
    async def _create_conversation_context(
        self, conversation_id: str
    ) -> Optional[ConversationContext]:
        """Retrieve conversation context using ConversationManager."""
        try:
            context = await self.conversation_manager.get_context(conversation_id)
            return context
        except Exception as e:
            logger.warning(f"Failed to create conversation context: {e}")
            return None
    
    def _create_fallback_response(self, user_message: str) -> str:
        """Create a fallback response when all else fails."""
        return f"Je comprends que vous me demandez quelque chose concernant '{user_message[:50]}...', mais je rencontre des difficultés techniques pour traiter votre demande complètement. Pouvez-vous reformuler votre question ?"


class OrchestratorAgent(BaseFinancialAgent):
    """
    Orchestrator agent for coordinating multi-agent workflows.
    
    This agent manages the execution of specialized agents in sequence,
    handling failures gracefully and providing comprehensive workflow
    tracking and metrics with percentile distribution tracking. The default
    performance threshold is 30 seconds.
    
    Attributes:
        intent_agent: Intent detection agent
        search_agent: Search query agent  
        response_agent: Response generation agent
        workflow_executor: Workflow execution helper
        workflow_stats: Workflow performance statistics
            (includes p95 and p99 workflow duration metrics)
    """
    
    def __init__(self, intent_agent: LLMIntentAgent,
                 search_agent: SearchQueryAgent,
                 response_agent: ResponseAgent,
                 config: Optional[AgentConfig] = None,
                 performance_threshold_ms: int = 30000):
        """
        Initialize the orchestrator agent.
        
        Args:
            intent_agent: Intent detection agent
            search_agent: Search query agent
            response_agent: Response generation agent
            config: Optional agent configuration
            performance_threshold_ms: Maximum allowed workflow duration in milliseconds
                (default 30000ms)
        """
        if config is None:
            try:
                config = AgentConfig(
                    name="orchestrator_agent",
                    model_client_config={
                        "model": "deepseek-chat",
                        "api_key": intent_agent.deepseek_client.api_key,
                        "base_url": intent_agent.deepseek_client.base_url,
                    },
                    system_message=self._get_system_message(),
                    max_consecutive_auto_reply=1,
                    description="Multi-agent workflow orchestration agent",
                    temperature=0.1,
                    max_tokens=150,
                    timeout_seconds=30,  # Higher timeout for workflow coordination
                )
            except TypeError:
                # Fallback for environments with simplified AgentConfig stubs
                config = SimpleNamespace(
                    name="orchestrator_agent",
                    model_client_config={
                        "model": "deepseek-chat",
                        "api_key": getattr(intent_agent.deepseek_client, "api_key", ""),
                        "base_url": getattr(intent_agent.deepseek_client, "base_url", ""),
                    },
                    system_message=self._get_system_message(),
                    max_consecutive_auto_reply=1,
                    description="Multi-agent workflow orchestration agent",
                    temperature=0.1,
                    max_tokens=150,
                    timeout_seconds=30,
                )

        try:
            super().__init__(
                name=config.name,
                config=config,
                deepseek_client=intent_agent.deepseek_client,
            )
        except TypeError:
            # Minimal initialization when BaseFinancialAgent is not fully functional
            self.name = config.name
            self.config = config
            self.deepseek_client = intent_agent.deepseek_client
            self.metrics = None
            self.domain = "financial"
        
        self.intent_agent = intent_agent
        self.search_agent = search_agent
        self.response_agent = response_agent
        self.workflow_executor = WorkflowExecutor(intent_agent, search_agent, response_agent)

        # Performance monitoring
        self.performance_threshold_ms = performance_threshold_ms  # 30s default threshold
        self.recent_workflow_times: Deque[float] = deque(maxlen=1000)

        # Workflow statistics
        self.workflow_stats = {
            "total_workflows": 0,
            "successful_workflows": 0,
            "failed_workflows": 0,
            "avg_workflow_time_ms": 0.0,
            "p95_workflow_time_ms": 0.0,
            "p99_workflow_time_ms": 0.0,
            "step_success_rates": {
                "intent_detection": 0.0,
                "search_query": 0.0,
                "response_generation": 0.0
            }
        }
        
        logger.info("Initialized OrchestratorAgent with 3-agent workflow")
    
    async def _execute_operation(self, input_data: Dict[str, Any], user_id: int) -> Dict[str, Any]:
        """
        Execute orchestration operation.

        Args:
            input_data: Dict containing 'user_message' and optional 'conversation_id'
            user_id: ID of the requesting user

        Returns:
            Dict with workflow execution results
        """
        user_message = input_data.get("user_message", "")
        conversation_id = input_data.get("conversation_id", f"conv_{int(time.time())}")

        if not user_message:
            logger.warning(
                "Received empty user message; returning validation error response"
            )
            return {
                "content": "Je n'ai pas reçu de requête. Veuillez formuler votre demande.",
                "metadata": {
                    "workflow_success": False,
                    "error": "empty_user_message",
                    "conversation_id": conversation_id,
                },
                "confidence_score": 0.0,
            }

        return await self.process_conversation(user_message, conversation_id, user_id)

    async def process_conversation(
        self, user_message: str, conversation_id: str, user_id: int
    ) -> Dict[str, Any]:
        """
        Process a conversation through the complete agent workflow.
        
        Args:
            user_message: User's input message
            conversation_id: Conversation identifier
            user_id: ID of the requesting user
            
        Returns:
            Dictionary containing final response and workflow details
        """
        start_time = time.perf_counter()
        
        try:
            # Execute the complete workflow
            workflow_result = await self._execute_workflow(
                user_message, conversation_id, user_id
            )
            performance_summary = workflow_result.get("performance_summary", {})

            # Performance alerting
            execution_details = workflow_result.get("execution_details", {})
            total_duration = execution_details.get("total_duration_ms")
            if (
                isinstance(total_duration, (int, float))
                and total_duration > self.performance_threshold_ms
            ):
                logger.warning(
                    "Workflow execution exceeded %d ms: %.2f ms",
                    self.performance_threshold_ms,
                    total_duration,
                )
            
            # Extract workflow data for additional metadata
            workflow_data = workflow_result.get("workflow_data", {})
            intent_result = workflow_data.get("intent_result")
            search_metadata = workflow_data.get("search_results") or {}
            metadata = (
                search_metadata.get("metadata", {})
                if isinstance(search_metadata, dict)
                else {}
            )
            search_results_count = metadata.get("search_results_count")
            if search_results_count is None:
                sr_meta = metadata.get("search_response", {}) if isinstance(metadata, dict) else {}
                rm = sr_meta.get("response_metadata", {}) if isinstance(sr_meta, dict) else {}
                search_results_count = rm.get("returned_results", 0)

            entities_extracted = []
            intent_detected = None
            if intent_result is not None:
                intent_detected = getattr(intent_result, "intent_type", None)
                entities_extracted = [
                    e.model_dump() for e in getattr(intent_result, "entities", [])
                ]

            steps = execution_details.get("steps", []) if isinstance(execution_details, dict) else []
            agent_chain = [
                "orchestrator_agent",
                *[step.get("agent") for step in steps if step.get("status") != "skipped"],
            ]

            # Update workflow statistics
            self._update_workflow_stats(workflow_result, time.perf_counter() - start_time)

            return {
                "content": workflow_result.get(
                    "final_response",
                    "Je rencontre des difficultés techniques."
                ),
                "metadata": {
                    "workflow_success": workflow_result.get("success", False),
                    "execution_details": execution_details,
                    "performance_summary": performance_summary,
                    "conversation_id": conversation_id,
                    "intent_detected": intent_detected,
                    "entities_extracted": entities_extracted,
                    "agent_chain": agent_chain,
                    "search_results_count": search_results_count,
                },
                "confidence_score": self._calculate_workflow_confidence(workflow_result),
                "token_usage": self._aggregate_token_usage(workflow_result),
            }
            
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            self.workflow_stats["failed_workflows"] += 1
            
            logger.error(f"Conversation processing failed: {e}")
            
            return {
                "content": f"Je rencontre des difficultés pour traiter votre demande: {str(e)}",
                "metadata": {
                    "workflow_success": False,
                    "error": str(e),
                    "execution_time_ms": execution_time,
                    "conversation_id": conversation_id,
                    "intent_detected": None,
                    "entities_extracted": [],
                    "agent_chain": ["orchestrator_agent"],
                    "search_results_count": 0,
                },
                "confidence_score": 0.1
            }
    
    async def _execute_workflow(
        self, user_message: str, conversation_id: str, user_id: int
    ) -> Dict[str, Any]:
        """Execute the workflow using the workflow executor."""
        return await self.workflow_executor.execute_workflow(
            user_message, conversation_id, user_id
        )

    @staticmethod
    def _percentile(values: Deque[float], percentile: float) -> float:
        """Compute a percentile for workflow durations."""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        k = int(round((len(sorted_vals) - 1) * percentile))
        return sorted_vals[k]

    def _update_workflow_stats(self, workflow_result: Dict[str, Any], duration_seconds: float) -> None:
        """Update workflow performance statistics and percentile metrics."""
        self.workflow_stats["total_workflows"] += 1
        
        if workflow_result["success"]:
            self.workflow_stats["successful_workflows"] += 1
        else:
            self.workflow_stats["failed_workflows"] += 1
        
        # Update average workflow time
        total = self.workflow_stats["total_workflows"]
        current_avg = self.workflow_stats["avg_workflow_time_ms"]
        new_time_ms = duration_seconds * 1000

        self.workflow_stats["avg_workflow_time_ms"] = (
            (current_avg * (total - 1) + new_time_ms) / total
        )
        self.recent_workflow_times.append(new_time_ms)
        self.workflow_stats["p95_workflow_time_ms"] = round(
            self._percentile(self.recent_workflow_times, 0.95), 2
        )
        self.workflow_stats["p99_workflow_time_ms"] = round(
            self._percentile(self.recent_workflow_times, 0.99), 2
        )
        
        # Update step success rates
        steps = workflow_result.get("execution_details", {}).get("steps", [])
        for step in steps:
            step_name = step["name"]
            if step_name in self.workflow_stats["step_success_rates"]:
                if step["status"] == "skipped":
                    continue
                current_rate = self.workflow_stats["step_success_rates"][step_name]
                success = 1.0 if step["status"] == "completed" else 0.0
                new_rate = (current_rate * (total - 1) + success) / total
                self.workflow_stats["step_success_rates"][step_name] = new_rate
    
    def _calculate_workflow_confidence(self, workflow_result: Dict[str, Any]) -> float:
        """Calculate overall confidence score for the workflow."""
        if not workflow_result["success"]:
            return 0.2
        
        performance = workflow_result.get("performance_summary", {})
        completed_steps = performance.get("completed_steps", 0)
        total_steps = performance.get("total_steps", 3)
        
        # Base confidence on step completion rate
        base_confidence = completed_steps / total_steps
        
        # Boost if all steps completed
        if completed_steps == total_steps:
            base_confidence = min(base_confidence + 0.1, 1.0)
        
        return round(base_confidence, 2)
    
    def _aggregate_token_usage(self, workflow_result: Dict[str, Any]) -> Dict[str, int]:
        """Aggregate token usage across all workflow steps."""
        # Estimate token usage - in production, collect from actual agent responses
        total_tokens = 0
        
        steps = workflow_result.get("execution_details", {}).get("steps", [])
        for step in steps:
            if step["status"] == "completed":
                # Rough estimates based on agent type
                if step["name"] == "intent_detection":
                    total_tokens += 70
                elif step["name"] == "search_query":
                    total_tokens += 120
                elif step["name"] == "response_generation":
                    total_tokens += 230
        
        return {
            "prompt_tokens": int(total_tokens * 0.7),
            "completion_tokens": int(total_tokens * 0.3),
            "total_tokens": total_tokens
        }
    
    def get_workflow_stats(self) -> Dict[str, Any]:
        """Get comprehensive workflow performance statistics."""
        base_stats = self.get_performance_stats()
        
        # Calculate success rate
        total = self.workflow_stats["total_workflows"]
        success_rate = (
            self.workflow_stats["successful_workflows"] / total 
            if total > 0 else 0.0
        )
        
        workflow_specific_stats = {
            "workflow_statistics": {
                "total_workflows": self.workflow_stats["total_workflows"],
                "successful_workflows": self.workflow_stats["successful_workflows"],
                "failed_workflows": self.workflow_stats["failed_workflows"],
                "success_rate": round(success_rate, 3),
                "avg_workflow_time_ms": round(self.workflow_stats["avg_workflow_time_ms"], 2)
            },
            "step_performance": self.workflow_stats["step_success_rates"],
            "agent_health": {
                "intent_agent_healthy": self.intent_agent.is_healthy(),
                "search_agent_healthy": self.search_agent.is_healthy(),
                "response_agent_healthy": self.response_agent.is_healthy()
            }
        }
        
        base_stats.update(workflow_specific_stats)
        return base_stats
    
    def reset_workflow_stats(self) -> None:
        """Reset workflow statistics."""
        self.workflow_stats = {
            "total_workflows": 0,
            "successful_workflows": 0,
            "failed_workflows": 0,
            "avg_workflow_time_ms": 0.0,
            "step_success_rates": {
                "intent_detection": 0.0,
                "search_query": 0.0,
                "response_generation": 0.0
            }
        }
        logger.info("Reset workflow statistics")
    
    def _get_system_message(self) -> str:
        """Get system message for the orchestrator agent."""
        return """Tu es un agent orchestrateur qui coordonne l'exécution de workflows multi-agents pour les conversations financières.

Ton rôle est de:
1. Coordonner l'exécution séquentielle des agents spécialisés
2. Gérer les erreurs et les fallbacks gracieusement
3. Suivre les performances et métriques de workflow
4. Assurer la continuité conversationnelle

Workflow standard:
1. Agent de détection d'intention (règles + IA)
2. Agent de requête de recherche (génération + exécution)
3. Agent de génération de réponse (contextuelle)

Tu dois optimiser pour la fiabilité, la performance et l'expérience utilisateur."""