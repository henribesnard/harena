"""
Package des modèles pour le service de recherche.
"""

from .requests import (
    SearchRequest,
    ReindexRequest,
    BulkIndexRequest,
    DeleteUserDataRequest,
    QueryExpansionRequest,
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
    BaseResponse
)

# Export de tous les modèles
__all__ = [
    # Modèles de requêtes
    'SearchRequest',
    'ReindexRequest',
    'BulkIndexRequest',
    'DeleteUserDataRequest',
    'QueryExpansionRequest',
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
    'BaseResponse'
]

# Métadonnées du package
__version__ = "1.0.0"
__author__ = "Search Service Team"
__description__ = "Modèles de données pour le service de recherche Harena"

