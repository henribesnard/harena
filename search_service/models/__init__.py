"""
Package des modèles pour le service de recherche.
VERSION COMPLÈTE - Inclut SearchQuery, SearchResult, SearchType et tous les modèles
"""

from .requests import (
    # Enum
    SearchType,
    
    # Modèles principaux (pour compatibilité)
    SearchQuery,  # IMPORTANT : Modèle original pour rétrocompatibilité
    
    # Nouveaux modèles de requêtes
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
    # Modèles principaux (pour compatibilité)
    SearchResult,  # IMPORTANT : Modèle original pour rétrocompatibilité
    SearchResultItem,
    SearchResponse,
    
    # Modèles d'opérations
    ReindexResponse,
    BulkIndexResponse,
    DeleteUserDataResponse,
    BatchOperationResponse,
    IndexManagementResponse,
    
    # Modèles de statistiques et informations
    UserStatsResponse,
    HealthResponse,
    IndexInfoResponse,
    ServiceMetricsResponse,
    SystemInfoResponse,
    ConfigurationResponse,
    
    # Modèles de debug et diagnostic
    QueryExpansionResponse,
    DebugClientResponse,
    DebugSearchResponse,
    DiagnosticResponse,
    BenchmarkResponse,
    
    # Modèles de test et statut
    SimpleTestResponse,
    ConnectionStatusResponse,
    SearchCapabilitiesResponse,
    
    # Modèles d'erreur
    ErrorResponse,
    ValidationErrorResponse,
    
    # Classe de base
    BaseResponse
)

# Export de tous les modèles - AVEC SearchQuery et SearchResult pour compatibilité
__all__ = [
    # Enum (ESSENTIEL)
    'SearchType',
    
    # Modèles principaux pour compatibilité (CRITIQUES)
    'SearchQuery',     # Requis par search_engine.py, cache.py, etc.
    'SearchResult',    # Requis par search_engine.py
    'SearchResponse',  # Requis partout
    
    # Modèles de requêtes
    'SearchRequest',   # Nouveau système
    'ReindexRequest',
    'BulkIndexRequest',
    'DeleteUserDataRequest',
    'QueryExpansionRequest',
    'UserStatsRequest',
    'DebugSearchRequest',
    'IndexManagementRequest',
    'BaseRequest',
    
    # Modèles de réponses
    'SearchResultItem', # Nouveau système
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
__version__ = "2.0.0"
__author__ = "Search Service Team"
__description__ = "Modèles de données pour le service de recherche Harena - Migration complète"

# Note de migration
__migration_note__ = """
MIGRATION VERS NOUVEAU SYSTÈME COMPLÉTÉE

✅ ANCIENS MODÈLES (pour compatibilité) :
- SearchQuery  -> Utilisé par search_engine.py, cache.py
- SearchResult -> Utilisé par search_engine.py  
- SearchType   -> Enum partagé

✅ NOUVEAUX MODÈLES :
- SearchRequest   -> Remplace SearchQuery à terme
- SearchResultItem -> Remplace SearchResult à terme

⚠️  PROCHAINES ÉTAPES :
1. Tester que tous les imports fonctionnent
2. Migrer progressivement vers SearchRequest/SearchResultItem
3. Supprimer search_service/models.py
4. Valider que l'erreur d'injection est résolue
"""