"""
Clients externes pour le Search Service.

Ce module expose les clients optimisés pour l'interaction avec les services
externes utilisés par le Search Service : Elasticsearch (Bonsai) principalement.

ARCHITECTURE CLIENTS:
- BaseClient : Classe de base avec retry, circuit breaker, health checks
- ElasticsearchClient : Client spécialisé Bonsai avec optimisations financières
- Extensible pour futurs clients (Redis cache, etc.)

RESPONSABILITÉS:
✅ Abstraction des services externes
✅ Gestion robuste des erreurs et retry
✅ Circuit breaker et health monitoring
✅ Métriques et observabilité
✅ Configuration centralisée
✅ Optimisations performance

USAGE:
    from search_service.clients import ElasticsearchClient, BaseClient
    from search_service.clients.base_client import RetryConfig, CircuitBreakerConfig
    
    # Client Elasticsearch optimisé
    es_client = ElasticsearchClient(
        bonsai_url="https://your-cluster.bonsaisearch.net",
        index_name="harena_transactions"
    )
    await es_client.start()
    
    # Recherche avec validation défensive
    results = await es_client.search(
        index="harena_transactions",
        body={"query": {"match_all": {}}}
    )
"""

import logging
from typing import Dict, Any, Optional

# Configuration centralisée
from config_service.config import settings

# Clients disponibles
from .base_client import (
    BaseClient,
    RetryConfig,
    CircuitBreakerConfig,
    HealthCheckConfig,
    ClientError,
    ConnectionError,
    TimeoutError,
    RetryExhaustedError,
    CircuitBreakerOpenError
)

from .elasticsearch_client import ElasticsearchClient

# Logger pour ce module
logger = logging.getLogger(__name__)

# ==================== FACTORY ET UTILITAIRES ====================

def create_elasticsearch_client(
    bonsai_url: Optional[str] = None,
    index_name: Optional[str] = None,
    timeout: Optional[float] = None,
    **kwargs
) -> ElasticsearchClient:
    """
    Factory pour créer un client Elasticsearch configuré.
    
    Args:
        bonsai_url: URL Bonsai (par défaut depuis config_service)
        index_name: Nom de l'index (par défaut depuis config_service)
        timeout: Timeout des requêtes
        **kwargs: Configuration supplémentaire
        
    Returns:
        Client Elasticsearch configuré
        
    Raises:
        ValueError: Si la configuration est invalide
    """
    # Valeurs par défaut depuis la configuration centralisée
    if bonsai_url is None:
        bonsai_url = settings.BONSAI_URL
        if not bonsai_url:
            raise ValueError("BONSAI_URL must be configured")
    
    if index_name is None:
        index_name = getattr(settings, 'ELASTICSEARCH_INDEX', 'harena_transactions')
    
    if timeout is None:
        timeout = getattr(settings, 'ELASTICSEARCH_TIMEOUT', 5.0)
    
    # Configuration retry et circuit breaker depuis settings
    retry_config = kwargs.get('retry_config')
    if retry_config is None:
        retry_config = RetryConfig(
            max_attempts=getattr(settings, 'ELASTICSEARCH_RETRY_ATTEMPTS', 3),
            base_delay=getattr(settings, 'ELASTICSEARCH_RETRY_DELAY', 1.0),
            max_delay=getattr(settings, 'ELASTICSEARCH_RETRY_MAX_DELAY', 10.0),
            backoff_factor=getattr(settings, 'ELASTICSEARCH_BACKOFF_FACTOR', 2.0)
        )
    
    circuit_breaker_config = kwargs.get('circuit_breaker_config')
    if circuit_breaker_config is None:
        circuit_breaker_config = CircuitBreakerConfig(
            failure_threshold=getattr(settings, 'ELASTICSEARCH_CB_FAILURE_THRESHOLD', 5),
            timeout_threshold=getattr(settings, 'ELASTICSEARCH_CB_TIMEOUT_THRESHOLD', 10.0),
            recovery_timeout=getattr(settings, 'ELASTICSEARCH_CB_RECOVERY_TIMEOUT', 60.0)
        )
    
    logger.info(f"Creating Elasticsearch client for index: {index_name}")
    
    return ElasticsearchClient(
        bonsai_url=bonsai_url,
        index_name=index_name,
        timeout=timeout,
        retry_config=retry_config,
        circuit_breaker_config=circuit_breaker_config,
        **{k: v for k, v in kwargs.items() if k not in ['retry_config', 'circuit_breaker_config']}
    )

# ==================== EXPORTS ====================

__all__ = [
    # Classes de base
    "BaseClient",
    "RetryConfig", 
    "CircuitBreakerConfig",
    "HealthCheckConfig",
    
    # Exceptions
    "ClientError",
    "ConnectionError",
    "TimeoutError", 
    "RetryExhaustedError",
    "CircuitBreakerOpenError",
    
    # Clients spécialisés
    "ElasticsearchClient",
    
    # Factory
    "create_elasticsearch_client"
]