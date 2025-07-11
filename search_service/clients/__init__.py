"""
Module clients pour le service de recherche.

Ce module fournit les clients pour les services externes utilisés
par le service de recherche, notamment Elasticsearch/Bonsai.
"""

from .base_client import (
    BaseClient,
    ClientStatus,
    RetryConfig,
    CircuitBreakerConfig,
    HealthCheckConfig,
    CircuitBreaker,
    CircuitBreakerState
)
from .elasticsearch_client import ElasticsearchClient

__all__ = [
    "BaseClient",
    "ClientStatus", 
    "RetryConfig",
    "CircuitBreakerConfig",
    "HealthCheckConfig",
    "CircuitBreaker",
    "CircuitBreakerState",
    "ElasticsearchClient"
]