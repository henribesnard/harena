"""
Modèles de réponses pour le service de recherche.
VERSION COMPLÈTE - Inclut SearchResult et tous les modèles nécessaires
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


class BaseResponse(BaseModel):
    """Classe de base pour toutes les réponses."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    timestamp: float = Field(..., description="Timestamp de la réponse")
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        super().__init__(**data)


class SearchResult(BaseModel):
    """Modèle de résultat de recherche - ORIGINAL pour compatibilité."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    transaction_id: int = Field(..., description="ID de la transaction")
    user_id: int = Field(..., description="ID de l'utilisateur")
    score: float = Field(..., description="Score de pertinence", ge=0.0)
    transaction: Dict[str, Any] = Field(..., description="Données de la transaction")
    highlights: Optional[Dict[str, List[str]]] = Field(None, description="Texte mis en évidence")
    search_type: Optional[str] = Field(None, description="Type de recherche utilisé")
    explanation: Optional[Dict[str, Any]] = Field(None, description="Explication du score")


class SearchResultItem(BaseModel):
    """Modèle pour un résultat de recherche individuel - NOUVEAU système."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    id: str = Field(..., description="ID unique du résultat")
    score: float = Field(..., description="Score de pertinence", ge=0.0)
    transaction: Dict[str, Any] = Field(..., description="Données de la transaction")
    highlights: Optional[Dict[str, List[str]]] = Field(None, description="Texte mis en évidence")
    search_type: Optional[str] = Field(None, description="Type de recherche utilisé")
    explanation: Optional[Dict[str, Any]] = Field(None, description="Explication du score")


class SearchResponse(BaseResponse):
    """Modèle de réponse pour la recherche de transactions - Compatible ancien/nouveau."""
    
    # Champs pour compatibilité avec l'ancien système
    query: str = Field(..., description="Requête originale")
    search_type: str = Field(..., description="Type de recherche utilisé")
    results: List[SearchResult] = Field(..., description="Liste des résultats")
    total_found: int = Field(..., description="Nombre total de résultats trouvés", ge=0)
    limit: int = Field(..., description="Limite de résultats demandée", ge=1)
    offset: int = Field(..., description="Décalage pour pagination", ge=0)
    has_more: bool = Field(..., description="Y a-t-il plus de résultats")
    processing_time: float = Field(..., description="Temps de traitement en secondes", ge=0.0)
    
    # Champs pour compatibilité avec le nouveau système (API routes)
    total: Optional[int] = Field(None, description="Nombre total de résultats (alias)", ge=0)
    query_time: Optional[float] = Field(None, description="Temps de requête (alias)", ge=0.0)
    user_id: Optional[int] = Field(None, description="ID de l'utilisateur")
    
    # Champs avancés
    timings: Optional[Dict[str, float]] = Field(None, description="Détails des temps de traitement")
    filters_applied: Optional[Dict[str, Any]] = Field(None, description="Filtres appliqués")
    suggestions: Optional[List[str]] = Field(None, description="Suggestions de requêtes")
    error: Optional[str] = Field(None, description="Message d'erreur éventuel")
    
    def __init__(self, **data):
        # Synchroniser total et total_found
        if 'total' in data and 'total_found' not in data:
            data['total_found'] = data['total']
        elif 'total_found' in data and 'total' not in data:
            data['total'] = data['total_found']
        
        # Synchroniser query_time et processing_time
        if 'query_time' in data and 'processing_time' not in data:
            data['processing_time'] = data['query_time']
        elif 'processing_time' in data and 'query_time' not in data:
            data['query_time'] = data['processing_time']
        
        # Valeurs par défaut pour compatibilité
        if 'has_more' not in data:
            total = data.get('total_found', data.get('total', 0))
            limit = data.get('limit', 20)
            offset = data.get('offset', 0)
            data['has_more'] = (offset + len(data.get('results', []))) < total
        
        super().__init__(**data)


class ReindexResponse(BaseResponse):
    """Modèle de réponse pour la réindexation."""
    
    success: bool = Field(..., description="Succès de l'opération")
    processed: int = Field(..., description="Nombre de documents traités", ge=0)
    indexed: int = Field(..., description="Nombre de documents indexés", ge=0)
    errors: int = Field(..., description="Nombre d'erreurs", ge=0)
    reindex_time: float = Field(..., description="Temps de réindexation en secondes", ge=0.0)
    user_id: int = Field(..., description="ID de l'utilisateur")
    error: Optional[str] = Field(None, description="Message d'erreur éventuel")
    details: Optional[Dict[str, Any]] = Field(None, description="Détails additionnels")


class BulkIndexResponse(BaseResponse):
    """Modèle de réponse pour l'indexation en lot."""
    
    success: bool = Field(..., description="Succès global de l'opération")
    total_users: int = Field(..., description="Nombre total d'utilisateurs traités", ge=0)
    successful_users: int = Field(..., description="Utilisateurs traités avec succès", ge=0)
    failed_users: int = Field(..., description="Utilisateurs en échec", ge=0)
    total_documents: int = Field(..., description="Total de documents traités", ge=0)
    processing_time: float = Field(..., description="Temps total de traitement", ge=0.0)
    details: Dict[str, Any] = Field(..., description="Détails par utilisateur")
    errors: List[str] = Field(default_factory=list, description="Liste des erreurs")


class DeleteUserDataResponse(BaseResponse):
    """Modèle de réponse pour la suppression des données utilisateur."""
    
    success: bool = Field(..., description="Succès de l'opération")
    user_id: int = Field(..., description="ID de l'utilisateur")
    deleted_from_search: bool = Field(..., description="Supprimé des index de recherche")
    deleted_from_cache: bool = Field(..., description="Supprimé du cache")
    documents_removed: int = Field(..., description="Nombre de documents supprimés", ge=0)
    processing_time: float = Field(..., description="Temps de traitement", ge=0.0)
    error: Optional[str] = Field(None, description="Message d'erreur éventuel")


class UserStatsResponse(BaseResponse):
    """Modèle de réponse pour les statistiques utilisateur."""
    
    user_id: int = Field(..., description="ID de l'utilisateur")
    search_count: int = Field(..., description="Nombre de recherches", ge=0)
    cache_hits: int = Field(..., description="Nombre de hits cache", ge=0)
    cache_misses: int = Field(..., description="Nombre de miss cache", ge=0)
    average_response_time: float = Field(..., description="Temps de réponse moyen", ge=0.0)
    most_searched_terms: List[str] = Field(..., description="Termes les plus recherchés")
    date_range: Dict[str, str] = Field(..., description="Plage de dates analysée")


class HealthResponse(BaseResponse):
    """Modèle de réponse pour le health check."""
    
    status: str = Field(..., description="Statut global du service")
    elasticsearch: Dict[str, Any] = Field(..., description="État d'Elasticsearch")
    qdrant: Dict[str, Any] = Field(..., description="État de Qdrant")
    search_engine: Dict[str, Any] = Field(..., description="État du moteur de recherche")
    cache: Optional[Dict[str, Any]] = Field(None, description="État du cache")
    uptime: float = Field(..., description="Temps de fonctionnement", ge=0.0)


class QueryExpansionResponse(BaseResponse):
    """Modèle de réponse pour l'expansion de requête."""
    
    original_query: str = Field(..., description="Requête originale")
    expanded_terms: List[str] = Field(..., description="Termes étendus")
    synonyms: List[str] = Field(..., description="Synonymes trouvés")
    financial_terms: List[str] = Field(..., description="Termes financiers")
    processing_time: float = Field(..., description="Temps de traitement", ge=0.0)


class DebugClientResponse(BaseResponse):
    """Modèle de réponse pour le debug des clients."""
    
    elasticsearch_status: Dict[str, Any] = Field(..., description="État Elasticsearch")
    qdrant_status: Dict[str, Any] = Field(..., description="État Qdrant")
    search_engine_status: Dict[str, Any] = Field(..., description="État du moteur")
    injection_status: Dict[str, Any] = Field(..., description="État de l'injection")
    connectivity_tests: Dict[str, Any] = Field(..., description="Tests de connectivité")


class DebugSearchResponse(BaseResponse):
    """Modèle de réponse pour le debug de recherche."""
    
    query_analysis: Dict[str, Any] = Field(..., description="Analyse de la requête")
    search_execution: Dict[str, Any] = Field(..., description="Exécution de la recherche")
    results_processing: Dict[str, Any] = Field(..., description="Traitement des résultats")
    performance_metrics: Dict[str, float] = Field(..., description="Métriques de performance")
    raw_results: Optional[List[Dict[str, Any]]] = Field(None, description="Résultats bruts")


class ErrorResponse(BaseResponse):
    """Modèle de réponse pour les erreurs."""
    
    error: str = Field(..., description="Message d'erreur")
    error_code: str = Field(..., description="Code d'erreur")
    details: Optional[Dict[str, Any]] = Field(None, description="Détails de l'erreur")
    request_id: Optional[str] = Field(None, description="ID de la requête")


class ValidationErrorResponse(BaseResponse):
    """Modèle de réponse pour les erreurs de validation."""
    
    message: str = Field(..., description="Message de validation")
    field_errors: List[Dict[str, Any]] = Field(..., description="Erreurs par champ")
    invalid_fields: List[str] = Field(..., description="Champs invalides")


class IndexInfoResponse(BaseResponse):
    """Modèle de réponse pour les informations d'index."""
    
    index_name: str = Field(..., description="Nom de l'index")
    document_count: int = Field(..., description="Nombre de documents", ge=0)
    index_size: str = Field(..., description="Taille de l'index")
    settings: Dict[str, Any] = Field(..., description="Paramètres de l'index")
    mappings: Dict[str, Any] = Field(..., description="Mappings de l'index")


class ServiceMetricsResponse(BaseResponse):
    """Modèle de réponse pour les métriques du service."""
    
    search_count: int = Field(..., description="Nombre total de recherches", ge=0)
    average_response_time: float = Field(..., description="Temps de réponse moyen", ge=0.0)
    cache_hit_rate: float = Field(..., description="Taux de hit cache", ge=0.0, le=1.0)
    error_rate: float = Field(..., description="Taux d'erreur", ge=0.0, le=1.0)
    uptime: float = Field(..., description="Temps de fonctionnement", ge=0.0)
    memory_usage: Dict[str, Any] = Field(..., description="Utilisation mémoire")


class BatchOperationResponse(BaseResponse):
    """Modèle de réponse pour les opérations en lot."""
    
    operation_type: str = Field(..., description="Type d'opération")
    total_items: int = Field(..., description="Nombre total d'éléments", ge=0)
    successful_items: int = Field(..., description="Éléments traités avec succès", ge=0)
    failed_items: int = Field(..., description="Éléments en échec", ge=0)
    processing_time: float = Field(..., description="Temps de traitement", ge=0.0)
    errors: List[str] = Field(default_factory=list, description="Liste des erreurs")


class IndexManagementResponse(BaseResponse):
    """Modèle de réponse pour la gestion des index."""
    
    action: str = Field(..., description="Action effectuée")
    success: bool = Field(..., description="Succès de l'opération")
    index_type: str = Field(..., description="Type d'index traité")
    details: Dict[str, Any] = Field(..., description="Détails de l'opération")
    processing_time: float = Field(..., description="Temps de traitement", ge=0.0)
    error: Optional[str] = Field(None, description="Message d'erreur éventuel")


class SimpleTestResponse(BaseResponse):
    """Modèle de réponse pour les tests simples."""
    
    status: str = Field(..., description="Statut du test")
    message: str = Field(..., description="Message du test")
    version: str = Field(..., description="Version du service")
    test_data: Optional[Dict[str, Any]] = Field(None, description="Données de test")


class ConnectionStatusResponse(BaseResponse):
    """Modèle de réponse pour le statut des connexions."""
    
    service_name: str = Field(..., description="Nom du service")
    connections: Dict[str, Any] = Field(..., description="État des connexions")
    total_connections: int = Field(..., description="Nombre total de connexions", ge=0)
    healthy_connections: int = Field(..., description="Connexions saines", ge=0)
    failed_connections: int = Field(..., description="Connexions échouées", ge=0)


class SearchCapabilitiesResponse(BaseResponse):
    """Modèle de réponse pour les capacités de recherche."""
    
    lexical_search: bool = Field(..., description="Recherche lexicale disponible")
    semantic_search: bool = Field(..., description="Recherche sémantique disponible")
    hybrid_search: bool = Field(..., description="Recherche hybride disponible")
    reranking: bool = Field(..., description="Reranking disponible")
    caching: bool = Field(..., description="Cache disponible")
    supported_languages: List[str] = Field(..., description="Langues supportées")


class SystemInfoResponse(BaseResponse):
    """Modèle de réponse pour les informations système."""
    
    service_name: str = Field(..., description="Nom du service")
    version: str = Field(..., description="Version du service")
    python_version: str = Field(..., description="Version Python")
    dependencies: Dict[str, str] = Field(..., description="Versions des dépendances")
    environment: str = Field(..., description="Environnement (dev/prod)")
    uptime: float = Field(..., description="Temps de fonctionnement", ge=0.0)


class ConfigurationResponse(BaseResponse):
    """Modèle de réponse pour la configuration."""
    
    search_settings: Dict[str, Any] = Field(..., description="Paramètres de recherche")
    cache_settings: Dict[str, Any] = Field(..., description="Paramètres de cache")
    elasticsearch_config: Dict[str, Any] = Field(..., description="Configuration Elasticsearch")
    qdrant_config: Dict[str, Any] = Field(..., description="Configuration Qdrant")


class DiagnosticResponse(BaseResponse):
    """Modèle de réponse pour le diagnostic complet."""
    
    service_status: str = Field(..., description="Statut global du service")
    components: Dict[str, Dict[str, Any]] = Field(..., description="État des composants")
    performance_metrics: Dict[str, float] = Field(..., description="Métriques de performance")
    error_summary: Dict[str, int] = Field(..., description="Résumé des erreurs")
    recommendations: List[str] = Field(default_factory=list, description="Recommandations")
    detailed_logs: List[Dict[str, Any]] = Field(default_factory=list, description="Logs détaillés")
    diagnostic_duration: float = Field(..., description="Durée du diagnostic", ge=0.0)


class BenchmarkResponse(BaseResponse):
    """Modèle de réponse pour les benchmarks."""
    
    test_name: str = Field(..., description="Nom du test")
    iterations: int = Field(..., description="Nombre d'itérations", ge=1)
    average_time: float = Field(..., description="Temps moyen", ge=0.0)
    min_time: float = Field(..., description="Temps minimum", ge=0.0)
    max_time: float = Field(..., description="Temps maximum", ge=0.0)
    success_rate: float = Field(..., description="Taux de succès", ge=0.0, le=1.0)
    details: Dict[str, Any] = Field(..., description="Détails du benchmark")


# Export de tous les modèles
__all__ = [
    # Modèles de résultats principaux (pour compatibilité)
    'SearchResult',  # IMPORTANT : Modèle original pour compatibilité
    'SearchResultItem',
    'SearchResponse',
    
    # Modèles d'opérations
    'ReindexResponse',
    'BulkIndexResponse',
    'DeleteUserDataResponse',
    'BatchOperationResponse',
    'IndexManagementResponse',
    
    # Modèles de statistiques et informations
    'UserStatsResponse',
    'HealthResponse',
    'IndexInfoResponse',
    'ServiceMetricsResponse',
    'SystemInfoResponse',
    'ConfigurationResponse',
    
    # Modèles de debug et diagnostic
    'QueryExpansionResponse',
    'DebugClientResponse',
    'DebugSearchResponse',
    'DiagnosticResponse',
    'BenchmarkResponse',
    
    # Modèles de test et statut
    'SimpleTestResponse',
    'ConnectionStatusResponse',
    'SearchCapabilitiesResponse',
    
    # Modèles d'erreur
    'ErrorResponse',
    'ValidationErrorResponse',
    
    # Classe de base
    'BaseResponse'
]