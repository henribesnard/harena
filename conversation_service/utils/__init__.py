"""
Utilities package for Conversation Service MVP.

This module provides essential utility components for the AutoGen-based
conversation service, including caching, validation, metrics collection,
and performance monitoring.

Exports:
    Cache Components:
        - LRUCache: Simple LRU cache with TTL
        - MultiLevelCache: Multi-level cache (L1 + L2)
        - create_cache: Cache factory function
        - get_default_cache: Default cache instance
    
    Validation:
        - ContractValidator: Service contract validation
        - ValidationError: Custom validation exception
    
    Metrics:
        - MetricsCollector: Agent and system metrics
        - PerformanceMonitor: Performance timing and monitoring
        - MetricsAggregator: Metrics aggregation and reporting

Author: Conversation Service Team
Created: 2025-01-31
Version: 1.0.0 MVP
"""

# Import cache components
from .cache import (
    LRUCache,
    MultiLevelCache,
    CacheStats,
    CacheEntry,
    create_cache,
    get_default_cache,
    generate_cache_key
)

# Import validators
from .validators import (
    ContractValidator,
    ValidationError,
    validate_search_query_contract,
    validate_search_response_contract,
    validate_intent_result_contract
)

# Import metrics
from .metrics import (
    MetricsCollector,
    PerformanceMonitor,
    MetricsAggregator,
    TimerContext,
    get_default_metrics_collector
)

# Package metadata
__version__ = "1.0.0"
__author__ = "Conversation Service Team"

# Export all utility components
__all__ = [
    # Cache Components
    "LRUCache",
    "MultiLevelCache", 
    "CacheStats",
    "CacheEntry",
    "create_cache",
    "get_default_cache",
    "generate_cache_key",
    
    # Validation
    "ContractValidator",
    "ValidationError",
    "validate_search_query_contract",
    "validate_search_response_contract", 
    "validate_intent_result_contract",
    
    # Metrics
    "MetricsCollector",
    "PerformanceMonitor",
    "MetricsAggregator",
    "TimerContext",
    "get_default_metrics_collector"
]

# Configuration par défaut
DEFAULT_UTILS_CONFIG = {
    "cache_enabled": True,
    "metrics_enabled": True,
    "validation_strict": True,
    "performance_monitoring": True
}