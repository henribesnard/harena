"""
Module core pour le service d'enrichissement.

Ce module contient les composants centraux partagés par l'ensemble
du service d'enrichissement : configuration, sécurité, logging, exceptions.
"""

from enrichment_service.core.config import enrichment_settings
from enrichment_service.core.logging import setup_enrichment_logging, get_contextual_logger
from enrichment_service.core.security import (
    get_current_user, 
    get_current_active_user, 
    get_current_superuser,
    require_permission,
    EnrichmentPermissions
)
from enrichment_service.core.exceptions import (
    EnrichmentServiceError,
    EmbeddingServiceError,
    QdrantServiceError,
    TransactionEnrichmentError,
    PatternDetectionError,
    InsightGenerationError,
    SummaryGenerationError,
    ValidationError,
    ConfigurationError
)

__all__ = [
    # Configuration
    'enrichment_settings',
    
    # Logging
    'setup_enrichment_logging', 
    'get_contextual_logger',
    
    # Security
    'get_current_user',
    'get_current_active_user',
    'get_current_superuser',
    'require_permission',
    'EnrichmentPermissions',
    
    # Exceptions
    'EnrichmentServiceError',
    'EmbeddingServiceError',
    'QdrantServiceError', 
    'TransactionEnrichmentError',
    'PatternDetectionError',
    'InsightGenerationError',
    'SummaryGenerationError',
    'ValidationError',
    'ConfigurationError'
]