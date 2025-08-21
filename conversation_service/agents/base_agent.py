"""
Base Financial Agent for Harena Conversation Service.

This module implements the base class for all financial agents using AutoGen v0.4
framework. It provides common functionality including caching, metrics, error handling,
and OpenAI integration for specialized financial domain agents.

Key Features:
- AutoGen AssistantAgent integration with OpenAI
- Intelligent caching system with TTL and LRU eviction
- Comprehensive metrics collection and performance monitoring
- Structured error handling with fallback mechanisms
- Few-shot prompt management and optimization
- Cost tracking and budget management

Author: Harena Conversation Team
Created: 2025-01-31
Version: 1.0.0 - AutoGen v0.4 Base Infrastructure
"""

import asyncio
import hashlib
import json
import time
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from dataclasses import dataclass

# AutoGen imports
from autogen import AssistantAgent

from ..clients.openai_client import OpenAIClient

# Local imports
from ..models.agent_models import AgentConfig, AgentResponse
from ..models.core_models import FinancialEntity
from ..core.cache_manager import CacheManager
from ..core.metrics_collector import MetricsCollector
from ..core.validators import HarenaValidators
from ..utils.logging import get_structured_logger

__all__ = ["BaseFinancialAgent", "AgentPerformanceTracker", "PromptOptimizer"]

# Configure logging
logger = get_structured_logger(__name__)

# Default cache TTL (seconds) per agent
AGENT_CACHE_TTLS = {
    "intent_classifier": 600,
    "entity_extractor": 900,
    "query_generator": 120,
    "response_generator": 60,
}

# ================================
# PERFORMANCE TRACKING SYSTEM
# ================================

@dataclass
class AgentPerformanceMetrics:
    """Performance metrics for agent monitoring."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_processing_time_ms: int = 0
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0
    average_confidence: float = 0.0
    last_activity: Optional[datetime] = None

class AgentPerformanceTracker:
    """
    Performance tracker for agent monitoring and optimization.
    
    Tracks key metrics for performance analysis, cost optimization,
    and quality assessment of financial agents.
    """
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.metrics = AgentPerformanceMetrics()
        self.hourly_stats = {}  # Hour -> metrics
        self.daily_stats = {}   # Date -> metrics
        
        # Performance thresholds
        self.thresholds = {
            "max_processing_time_ms": 10000,  # 10 seconds
            "min_success_rate": 0.95,         # 95%
            "max_cost_per_call": 0.10,       # $0.10
            "min_cache_hit_rate": 0.30       # 30%
        }
    
    def record_call(
        self,
        success: bool,
        processing_time_ms: int,
        tokens_used: int = 0,
        cost_usd: float = 0.0,
        confidence: Optional[float] = None,
        cached: bool = False
    ):
        """Record a single agent call for performance tracking."""
        
        # Update main metrics
        self.metrics.total_calls += 1
        self.metrics.last_activity = datetime.now()
        
        if success:
            self.metrics.successful_calls += 1
        else:
            self.metrics.failed_calls += 1
        
        if cached:
            self.metrics.cache_hits += 1
        else:
            self.metrics.cache_misses += 1
        
        self.metrics.total_processing_time_ms += processing_time_ms
        self.metrics.total_tokens_used += tokens_used
        self.metrics.total_cost_usd += cost_usd
        
        if confidence is not None:
            # Running average of confidence
            total_successful = self.metrics.successful_calls
            if total_successful > 0:
                self.metrics.average_confidence = (
                    (self.metrics.average_confidence * (total_successful - 1) + confidence) 
                    / total_successful
                )
        
        # Update hourly stats
        hour_key = datetime.now().strftime("%Y-%m-%d-%H")
        if hour_key not in self.hourly_stats:
            self.hourly_stats[hour_key] = {"calls": 0, "successes": 0, "total_time": 0}
        
        self.hourly_stats[hour_key]["calls"] += 1
        if success:
            self.hourly_stats[hour_key]["successes"] += 1
        self.hourly_stats[hour_key]["total_time"] += processing_time_ms
        
        # Log performance alerts
        self._check_performance_alerts(processing_time_ms, cost_usd)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        
        if self.metrics.total_calls == 0:
            return {"agent_name": self.agent_name, "status": "no_activity"}
        
        success_rate = self.metrics.successful_calls / self.metrics.total_calls
        cache_hit_rate = self.metrics.cache_hits / self.metrics.total_calls
        avg_processing_time = self.metrics.total_processing_time_ms / self.metrics.total_calls
        avg_cost_per_call = self.metrics.total_cost_usd / self.metrics.total_calls
        avg_tokens_per_call = self.metrics.total_tokens_used / self.metrics.total_calls
        
        # Performance assessment
        performance_score = self._calculate_performance_score(
            success_rate, cache_hit_rate, avg_processing_time, avg_cost_per_call
        )
        
        return {
            "agent_name": self.agent_name,
            "total_calls": self.metrics.total_calls,
            "success_rate": success_rate,
            "cache_hit_rate": cache_hit_rate,
            "average_processing_time_ms": avg_processing_time,
            "average_cost_per_call_usd": avg_cost_per_call,
            "average_tokens_per_call": avg_tokens_per_call,
            "average_confidence": self.metrics.average_confidence,
            "total_cost_usd": self.metrics.total_cost_usd,
            "performance_score": performance_score,
            "last_activity": self.metrics.last_activity.isoformat() if self.metrics.last_activity else None,
            "status": self._get_health_status()
        }
    
    def _calculate_performance_score(
        self, 
        success_rate: float, 
        cache_hit_rate: float, 
        avg_processing_time: float, 
        avg_cost: float
    ) -> float:
        """Calculate overall performance score (0-100)."""
        
        # Weighted scoring
        success_score = success_rate * 40  # 40% weight
        speed_score = max(0, (10000 - avg_processing_time) / 10000) * 25  # 25% weight
        cost_score = max(0, (0.10 - avg_cost) / 0.10) * 20  # 20% weight
        cache_score = cache_hit_rate * 15  # 15% weight
        
        return success_score + speed_score + cost_score + cache_score
    
    def _get_health_status(self) -> str:
        """Determine agent health status."""
        
        if self.metrics.total_calls == 0:
            return "no_activity"
        
        success_rate = self.metrics.successful_calls / self.metrics.total_calls
        avg_processing_time = self.metrics.total_processing_time_ms / self.metrics.total_calls
        
        if success_rate < 0.85 or avg_processing_time > 15000:
            return "critical"
        elif success_rate < 0.95 or avg_processing_time > 8000:
            return "warning"
        else:
            return "healthy"
    
    def _check_performance_alerts(self, processing_time_ms: int, cost_usd: float):
        """Check for performance threshold breaches."""
        
        if processing_time_ms > self.thresholds["max_processing_time_ms"]:
            logger.warning(
                "Agent performance alert: high processing time",
                agent_name=self.agent_name,
                processing_time_ms=processing_time_ms,
                threshold=self.thresholds["max_processing_time_ms"]
            )
        
        if cost_usd > self.thresholds["max_cost_per_call"]:
            logger.warning(
                "Agent cost alert: high cost per call",
                agent_name=self.agent_name,
                cost_usd=cost_usd,
                threshold=self.thresholds["max_cost_per_call"]
            )

# ================================
# PROMPT OPTIMIZATION SYSTEM
# ================================

class PromptOptimizer:
    """
    Intelligent prompt optimizer for financial agents.
    
    Optimizes prompts based on performance metrics, token usage,
    and success rates to reduce costs and improve accuracy.
    """
    
    def __init__(self):
        self.prompt_performance = {}  # prompt_hash -> performance metrics
        self.optimization_rules = {
            "min_examples_for_confidence": 3,
            "max_examples_for_efficiency": 8,
            "token_budget_per_call": 1500,
            "confidence_threshold": 0.85
        }
    
    def optimize_few_shot_examples(
        self,
        base_examples: List[Dict[str, str]],
        recent_performance: Dict[str, Any],
        intent_type: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Optimize few-shot examples based on performance data."""
        
        success_rate = recent_performance.get("success_rate", 0.0)
        avg_confidence = recent_performance.get("average_confidence", 0.0)
        avg_tokens = recent_performance.get("average_tokens_per_call", 0)
        
        # If performance is good and token usage is reasonable, keep current examples
        if (success_rate >= 0.95 and 
            avg_confidence >= 0.85 and 
            avg_tokens <= self.optimization_rules["token_budget_per_call"]):
            return base_examples
        
        # If performance is poor, add more examples (up to max)
        if success_rate < 0.90 or avg_confidence < 0.80:
            if len(base_examples) < self.optimization_rules["max_examples_for_efficiency"]:
                additional_examples = self._get_additional_examples(intent_type)
                return base_examples + additional_examples[:2]  # Add 2 more examples
        
        # If token usage is too high, reduce examples
        if avg_tokens > self.optimization_rules["token_budget_per_call"]:
            min_examples = self.optimization_rules["min_examples_for_confidence"]
            if len(base_examples) > min_examples:
                # Keep the most effective examples
                return base_examples[:min_examples]
        
        return base_examples
    
    def _get_additional_examples(self, intent_type: Optional[str]) -> List[Dict[str, str]]:
        """Get additional examples for specific intent types."""
        
        additional_examples = {
            "CATEGORY_ANALYSIS": [
                {
                    "input": "Mes achats shopping en décembre",
                    "output": '{"intent": "CATEGORY_ANALYSIS", "confidence": 0.94, "entities": [{"type": "CATEGORY", "value": "shopping"}, {"type": "DATE_RANGE", "value": "décembre"}]}'
                }
            ],
            "MERCHANT_ANALYSIS": [
                {
                    "input": "Combien chez Leclerc cette semaine ?",
                    "output": '{"intent": "MERCHANT_ANALYSIS", "confidence": 0.92, "entities": [{"type": "MERCHANT", "value": "Leclerc"}, {"type": "DATE_RANGE", "value": "this_week"}]}'
                }
            ]
        }
        
        return additional_examples.get(intent_type, [])

# ================================
# BASE FINANCIAL AGENT
# ================================

class BaseFinancialAgent(ABC):
    """
    Base class for all financial agents in Harena Conversation Service.
    
    Provides common functionality including AutoGen integration, caching,
    metrics collection, error handling, and OpenAI communication for
    specialized financial domain agents.
    """
    
    def __init__(
        self,
        config: AgentConfig,
        openai_client: OpenAIClient,
        cache_manager: Optional[CacheManager] = None,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """Initialize base financial agent."""
        
        self.config = config
        self.openai_client = openai_client
        self.cache_manager = cache_manager
        self.metrics_collector = metrics_collector
        
        # Performance tracking
        self.performance_tracker = AgentPerformanceTracker(config.name)
        self.prompt_optimizer = PromptOptimizer()
        
        # Create AutoGen AssistantAgent
        self.agent = AssistantAgent(
            name=config.name,
            system_message=config.system_message,
            llm_config={
                "model": config.model_name,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "timeout": config.timeout_seconds
            }
        )
        
        # Agent state
        self.is_initialized = True
        self.last_error = None
        self.circuit_breaker_failures = 0
        self.circuit_breaker_reset_time = None
        
        logger.info(
            "Base Financial Agent initialized",
            agent_name=config.name,
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Main processing method with comprehensive error handling and metrics.
        
        This method provides the standard interface for all financial agents,
        including caching, metrics collection, and circuit breaker patterns.
        """
        
        start_time = time.time()
        processing_time_ms = 0
        cached_result = False
        
        try:
            # Input validation
            self._validate_input_data(input_data)
            
            # Check circuit breaker
            if self._is_circuit_breaker_open():
                raise Exception("Circuit breaker is open - agent temporarily unavailable")
            
            # Check cache first
            cache_key = None
            user_id = str(input_data.get("user_id", "anonymous"))
            if self.cache_manager:
                cache_key = self._generate_cache_key(input_data)
                cached_result_data = await self.cache_manager.get(cache_key, user_id)
                
                if cached_result_data:
                    processing_time_ms = int((time.time() - start_time) * 1000)
                    cached_result = True
                    
                    # Record cache hit
                    self.performance_tracker.record_call(
                        success=True,
                        processing_time_ms=processing_time_ms,
                        cached=True
                    )
                    
                    logger.debug(
                        "Cache hit for agent",
                        agent_name=self.config.name,
                        cache_key=cache_key[:16] + "...",
                        processing_time_ms=processing_time_ms
                    )
                    
                    return AgentResponse(
                        agent_name=self.config.name,
                        success=True,
                        result=cached_result_data,
                        processing_time_ms=processing_time_ms,
                        tokens_used=0,
                        cached=True
                    )
            
            # Process with implementation
            result = await self._process_implementation(input_data)
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Cache successful result
            if self.cache_manager and result and cache_key:
                cache_ttl = AGENT_CACHE_TTLS.get(self.config.name, 300)
                await self.cache_manager.set(cache_key, result, user_id=user_id, ttl=cache_ttl)
            
            # Record successful call
            confidence = result.get("confidence") if isinstance(result, dict) else None
            tokens_used = result.get("tokens_used", 0) if isinstance(result, dict) else 0
            
            self.performance_tracker.record_call(
                success=True,
                processing_time_ms=processing_time_ms,
                tokens_used=tokens_used,
                confidence=confidence,
                cached=False
            )
            
            # Reset circuit breaker on success
            self.circuit_breaker_failures = 0
            self.circuit_breaker_reset_time = None
            
            # Record metrics
            if self.metrics_collector:
                await self.metrics_collector.record_agent_call(
                    agent_name=self.config.name,
                    success=True,
                    processing_time_ms=processing_time_ms,
                    tokens_used=tokens_used
                )
            
            logger.debug(
                "Agent processing successful",
                agent_name=self.config.name,
                processing_time_ms=processing_time_ms,
                tokens_used=tokens_used,
                confidence=confidence
            )
            
            return AgentResponse(
                agent_name=self.config.name,
                success=True,
                result=result,
                processing_time_ms=processing_time_ms,
                tokens_used=tokens_used,
                cached=False
            )
            
        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            error_message = str(e)
            
            # Record failure
            self.performance_tracker.record_call(
                success=False,
                processing_time_ms=processing_time_ms,
                cached=False
            )
            
            # Update circuit breaker
            self.circuit_breaker_failures += 1
            if self.circuit_breaker_failures >= 5:  # 5 failures trigger circuit breaker
                self.circuit_breaker_reset_time = datetime.now() + timedelta(minutes=5)
            
            # Record metrics
            if self.metrics_collector:
                await self.metrics_collector.record_agent_call(
                    agent_name=self.config.name,
                    success=False,
                    processing_time_ms=processing_time_ms,
                    error=error_message
                )
            
            logger.error(
                "Agent processing failed",
                agent_name=self.config.name,
                error=error_message,
                processing_time_ms=processing_time_ms,
                circuit_breaker_failures=self.circuit_breaker_failures,
                exc_info=True
            )
            
            return AgentResponse(
                agent_name=self.config.name,
                success=False,
                result=None,
                error_message=error_message,
                processing_time_ms=processing_time_ms,
                tokens_used=0,
                cached=False
            )
    
    @abstractmethod
    async def _process_implementation(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Abstract method to be implemented by specific agent types.
        
        This method contains the core business logic for each specialized agent.
        """
        pass
    
    async def _call_openai(
        self, 
        prompt: str, 
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call OpenAI with optimized prompts and few-shot examples.
        
        Includes prompt optimization, token management, and cost tracking.
        """
        
        # Optimize few-shot examples based on performance
        performance_summary = self.performance_tracker.get_performance_summary()
        optimized_examples = self.prompt_optimizer.optimize_few_shot_examples(
            few_shot_examples or self.config.few_shot_examples,
            performance_summary,
            intent_type=kwargs.get("intent_type")
        )
        
        # Build messages
        messages = []
        
        # System message
        messages.append({
            "role": "system",
            "content": self.config.system_message
        })
        
        # Few-shot examples
        for example in optimized_examples:
            messages.append({"role": "user", "content": example["input"]})
            messages.append({"role": "assistant", "content": example["output"]})
        
        # User prompt
        messages.append({"role": "user", "content": prompt})
        
        # Call OpenAI
        try:
            response = await self.openai_client.chat_completion(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout_seconds,
                **kwargs,
            )
            
            return {
                "content": response.choices[0].message.content,
                "tokens_used": response.usage.total_tokens if response.usage else 0,
                "model": response.model,
                "finish_reason": response.choices[0].finish_reason
            }
            
        except Exception as e:
            logger.error(
                "OpenAI API call failed",
                agent_name=self.config.name,
                model=self.config.model_name,
                error=str(e),
                exc_info=True
            )
            raise
    
    def _generate_cache_key(self, input_data: Dict[str, Any]) -> str:
        """Generate unique cache key for input data."""
        
        # Create deterministic cache key
        cache_data = {
            "agent": self.config.name,
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "input": input_data,
            "version": "1.0"
        }
        
        # Convert to JSON string and hash
        cache_string = json.dumps(cache_data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(cache_string.encode()).hexdigest()
    
    def _validate_input_data(self, input_data: Dict[str, Any]):
        """Validate input data structure and content."""
        
        if not isinstance(input_data, dict):
            raise ValueError("input_data must be a dictionary")
        
        # Agent-specific validation can be overridden
        required_fields = getattr(self, 'required_input_fields', ['user_message'])
        
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Required field '{field}' missing from input_data")
    
    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open (agent temporarily disabled)."""
        
        if self.circuit_breaker_reset_time is None:
            return False
        
        if datetime.now() > self.circuit_breaker_reset_time:
            # Reset circuit breaker
            self.circuit_breaker_failures = 0
            self.circuit_breaker_reset_time = None
            return False
        
        return True
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status and health information."""
        
        performance_summary = self.performance_tracker.get_performance_summary()
        
        return {
            "agent_name": self.config.name,
            "model": self.config.model_name,
            "is_initialized": self.is_initialized,
            "circuit_breaker_open": self._is_circuit_breaker_open(),
            "circuit_breaker_failures": self.circuit_breaker_failures,
            "performance": performance_summary,
            "configuration": {
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "timeout_seconds": self.config.timeout_seconds,
                "cache_enabled": self.cache_manager is not None,
                "metrics_enabled": self.metrics_collector is not None
            },
            "capabilities": {
                "caching": self.cache_manager is not None,
                "metrics_collection": self.metrics_collector is not None,
                "circuit_breaker": True,
                "prompt_optimization": True,
                "performance_tracking": True
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform agent health check with basic functionality test."""
        
        try:
            # Simple test call
            test_input = {"user_message": "test", "health_check": True}
            start_time = time.time()
            
            # Skip cache for health checks
            result = await self._process_implementation(test_input)
            response_time = int((time.time() - start_time) * 1000)
            
            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "circuit_breaker_open": self._is_circuit_breaker_open(),
                "last_error": self.last_error,
                "performance_score": self.performance_tracker.get_performance_summary().get("performance_score", 0)
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker_open": self._is_circuit_breaker_open(),
                "last_error": str(e)
            }