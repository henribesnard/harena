"""
Package des modèles pour le service de recherche.
VERSION CORRIGÉE - Suppression de SearchQuery inexistant
"""

from .requests import (
    SearchRequest,
    ReindexRequest,
    BulkIndexRequest,
    DeleteUserDataRequest,
    QueryExpansionRequest,
    UserStatsRequest,
    DebugSearchRequest,
    IndexManagementRequest,
    BaseRequest
)

from .responses import (
    SearchResultItem,
    SearchResponse,
    ReindexResponse,
    BulkIndexResponse,
    DeleteUserDataResponse,
    UserStatsResponse,
    HealthResponse,
    QueryExpansionResponse,
    DebugClientResponse,
    DebugSearchResponse,
    ErrorResponse,
    ValidationErrorResponse,
    IndexInfoResponse,
    ServiceMetricsResponse,
    BatchOperationResponse,
    IndexManagementResponse,
    SimpleTestResponse,
    ConnectionStatusResponse,
    SearchCapabilitiesResponse,
    SystemInfoResponse,
    ConfigurationResponse,
    DiagnosticResponse,
    BenchmarkResponse,
    BaseResponse
)

# Export de tous les modèles - SANS SearchQuery qui n'existe pas
__all__ = [
    # Modèles de requêtes
    'SearchRequest',
    'ReindexRequest',
    'BulkIndexRequest',
    'DeleteUserDataRequest',
    'QueryExpansionRequest',
    'UserStatsRequest',
    'DebugSearchRequest',
    'IndexManagementRequest',
    'BaseRequest',
    
    # Modèles de réponses
    'SearchResultItem',
    'SearchResponse',
    'ReindexResponse',
    'BulkIndexResponse',
    'DeleteUserDataResponse',
    'UserStatsResponse',
    'HealthResponse',
    'QueryExpansionResponse',
    'DebugClientResponse',
    'DebugSearchResponse',
    'ErrorResponse',
    'ValidationErrorResponse',
    'IndexInfoResponse',
    'ServiceMetricsResponse',
    'BatchOperationResponse',
    'IndexManagementResponse',
    'SimpleTestResponse',
    'ConnectionStatusResponse',
    'SearchCapabilitiesResponse',
    'SystemInfoResponse',
    'ConfigurationResponse',
    'DiagnosticResponse',
    'BenchmarkResponse',
    'BaseResponse'
]

# Métadonnées du package
__version__ = "1.0.0"
__author__ = "Search Service Team"
__description__ = "Modèles de données pour le service de recherche Harena"