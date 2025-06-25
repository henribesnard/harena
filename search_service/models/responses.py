"""
Modèles de réponses pour le service de recherche.
VERSION CORRIGÉE - Compatible Pydantic V2 avec toutes les classes requises
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


class SearchResultItem(BaseModel):
    """Modèle pour un résultat de recherche individuel."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    id: str = Field(..., description="ID unique du résultat")
    score: float = Field(..., description="Score de pertinence", ge=0.0)
    transaction: Dict[str, Any] = Field(..., description="Données de la transaction")
    highlights: Optional[Dict[str, List[str]]] = Field(None, description="Texte mis en évidence")
    search_type: Optional[str] = Field(None, description="Type de recherche utilisé")


class SearchResponse(BaseModel):
    """Modèle de réponse pour la recherche de transactions."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    results: List[SearchResultItem] = Field(..., description="Liste des résultats")
    total: int = Field(..., description="Nombre total de résultats", ge=0)
    query_time: float = Field(..., description="Temps de requête en secondes", ge=0.0)
    search_type: str = Field(..., description="Type de recherche utilisé")
    user_id: int = Field(..., description="ID de l'utilisateur")
    query: str = Field(..., description="Requête originale")
    timestamp: Optional[float] = Field(None, description="Timestamp de la réponse")
    error: Optional[str] = Field(None, description="Message d'erreur éventuel")
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        super().__init__(**data)


class ReindexResponse(BaseModel):
    """Modèle de réponse pour la réindexation."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    success: bool = Field(..., description="Succès de l'opération")
    processed: int = Field(..., description="Nombre de documents traités", ge=0)
    indexed: int = Field(..., description="Nombre de documents indexés", ge=0)
    errors: int = Field(..., description="Nombre d'erreurs", ge=0)
    reindex_time: float = Field(..., description="Temps de réindexation en secondes", ge=0.0)
    user_id: int = Field(..., description="ID de l'utilisateur")
    timestamp: Optional[float] = Field(None, description="Timestamp de la réponse")
    error: Optional[str] = Field(None, description="Message d'erreur éventuel")
    details: Optional[Dict[str, Any]] = Field(None, description="Détails additionnels")
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        super().__init__(**data)


class BulkIndexResponse(BaseModel):
    """Modèle de réponse pour l'indexation en lot."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    success: bool = Field(..., description="Succès de l'opération")
    total_submitted: int = Field(..., description="Nombre total de documents soumis", ge=0)
    elasticsearch_indexed: int = Field(..., description="Documents indexés dans Elasticsearch", ge=0)
    qdrant_indexed: int = Field(..., description="Documents indexés dans Qdrant", ge=0)
    errors: int = Field(..., description="Nombre d'erreurs", ge=0)
    processing_time: float = Field(..., description="Temps de traitement en secondes", ge=0.0)
    user_id: int = Field(..., description="ID de l'utilisateur")
    timestamp: Optional[float] = Field(None, description="Timestamp de la réponse")
    error_details: Optional[List[str]] = Field(None, description="Détails des erreurs")
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        super().__init__(**data)


class DeleteUserDataResponse(BaseModel):
    """Modèle de réponse pour la suppression des données utilisateur."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    success: bool = Field(..., description="Succès de l'opération")
    user_id: int = Field(..., description="ID de l'utilisateur")
    elasticsearch_deleted: int = Field(..., description="Documents supprimés d'Elasticsearch", ge=0)
    qdrant_deleted: int = Field(..., description="Vecteurs supprimés de Qdrant", ge=0)
    delete_time: float = Field(..., description="Temps de suppression en secondes", ge=0.0)
    timestamp: Optional[float] = Field(None, description="Timestamp de la réponse")
    error: Optional[str] = Field(None, description="Message d'erreur éventuel")
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        super().__init__(**data)


class UserStatsResponse(BaseModel):
    """Modèle de réponse pour les statistiques utilisateur."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    user_id: int = Field(..., description="ID de l'utilisateur")
    elasticsearch: Dict[str, Any] = Field(..., description="Statistiques Elasticsearch")
    qdrant: Dict[str, Any] = Field(..., description="Statistiques Qdrant")
    last_update: Optional[str] = Field(None, description="Dernière mise à jour")
    timestamp: Optional[float] = Field(None, description="Timestamp de la réponse")
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        super().__init__(**data)


class QueryExpansionResponse(BaseModel):
    """Modèle de réponse pour l'expansion de requête."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    original_query: str = Field(..., description="Requête originale")
    query_type: str = Field(..., description="Type de la requête originale")
    expanded_terms: List[str] = Field(..., description="Termes expandus")
    expanded_count: int = Field(..., description="Nombre de termes expandus", ge=0)
    search_string: str = Field(..., description="Chaîne de recherche construite")
    timestamp: Optional[float] = Field(None, description="Timestamp de la réponse")
    debug_info: Optional[Dict[str, Any]] = Field(None, description="Informations de debug")
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        super().__init__(**data)


class HealthResponse(BaseModel):
    """Modèle de réponse pour le health check."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    status: str = Field(..., description="Statut global du service")
    elasticsearch: Dict[str, Any] = Field(..., description="Statut Elasticsearch")
    qdrant: Dict[str, Any] = Field(..., description="Statut Qdrant")
    search_engine: Dict[str, Any] = Field(..., description="Statut du moteur de recherche")
    timestamp: float = Field(..., description="Timestamp de la vérification")
    uptime: Optional[float] = Field(None, description="Temps de fonctionnement en secondes")
    version: Optional[str] = Field(None, description="Version du service")


class ErrorResponse(BaseModel):
    """Modèle de réponse pour les erreurs."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    error: str = Field(..., description="Type d'erreur")
    detail: str = Field(..., description="Description détaillée de l'erreur")
    timestamp: float = Field(..., description="Timestamp de l'erreur")
    request_id: Optional[str] = Field(None, description="ID de la requête")
    bug_detected: Optional[str] = Field(None, description="Bug spécifique détecté")
    solution: Optional[str] = Field(None, description="Solution suggérée")
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        super().__init__(**data)


class ValidationErrorResponse(BaseModel):
    """Modèle de réponse pour les erreurs de validation spécifiques."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    error_type: str = Field(..., description="Type d'erreur de validation")
    field: str = Field(..., description="Champ en erreur")
    received_type: str = Field(..., description="Type reçu")
    expected_type: str = Field(..., description="Type attendu")
    received_value: Optional[str] = Field(None, description="Valeur reçue (tronquée)")
    message: str = Field(..., description="Message d'erreur détaillé")
    timestamp: float = Field(..., description="Timestamp de l'erreur")
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        super().__init__(**data)


class DebugClientResponse(BaseModel):
    """Modèle de réponse pour le debug des clients."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    elastic_client: Dict[str, Any] = Field(..., description="Informations client Elasticsearch")
    qdrant_client: Dict[str, Any] = Field(..., description="Informations client Qdrant")
    search_engine: Dict[str, Any] = Field(..., description="Informations moteur de recherche")
    timestamp: float = Field(..., description="Timestamp de la vérification")
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        super().__init__(**data)


class DebugSearchResponse(BaseModel):
    """Modèle de réponse pour le debug de recherche."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    client_type: str = Field(..., description="Type de client utilisé")
    user_id: int = Field(..., description="ID de l'utilisateur")
    query: str = Field(..., description="Requête de recherche")
    query_type: str = Field(..., description="Type de la requête")
    results_count: int = Field(..., description="Nombre de résultats", ge=0)
    results: List[Dict[str, Any]] = Field(..., description="Résultats de recherche")
    query_time: float = Field(..., description="Temps de requête en secondes", ge=0.0)
    timestamp: float = Field(..., description="Timestamp de la réponse")
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        super().__init__(**data)


class IndexInfoResponse(BaseModel):
    """Modèle de réponse pour les informations d'index."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    index_name: str = Field(..., description="Nom de l'index")
    elasticsearch: Dict[str, Any] = Field(..., description="Informations Elasticsearch")
    qdrant: Dict[str, Any] = Field(..., description="Informations Qdrant")
    timestamp: float = Field(..., description="Timestamp de la vérification")
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        super().__init__(**data)


class ServiceMetricsResponse(BaseModel):
    """Modèle de réponse pour les métriques du service."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    uptime: float = Field(..., description="Temps de fonctionnement en secondes")
    total_searches: int = Field(..., description="Nombre total de recherches", ge=0)
    total_indexations: int = Field(..., description="Nombre total d'indexations", ge=0)
    average_query_time: float = Field(..., description="Temps moyen de requête", ge=0.0)
    error_rate: float = Field(..., description="Taux d'erreur", ge=0.0, le=1.0)
    last_24h: Dict[str, Any] = Field(..., description="Métriques des 24 dernières heures")
    timestamp: float = Field(..., description="Timestamp des métriques")
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        super().__init__(**data)


class BatchOperationResponse(BaseModel):
    """Modèle de réponse pour les opérations par lots."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    operation_type: str = Field(..., description="Type d'opération")
    total_items: int = Field(..., description="Nombre total d'éléments", ge=0)
    successful: int = Field(..., description="Nombre de succès", ge=0)
    failed: int = Field(..., description="Nombre d'échecs", ge=0)
    processing_time: float = Field(..., description="Temps de traitement total", ge=0.0)
    items_per_second: float = Field(..., description="Vitesse de traitement", ge=0.0)
    errors: List[str] = Field(default_factory=list, description="Liste des erreurs")
    timestamp: float = Field(..., description="Timestamp de l'opération")
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        
        # Calculer items_per_second automatiquement
        if 'items_per_second' not in data and data.get('processing_time', 0) > 0:
            total = data.get('total_items', 0)
            time_taken = data.get('processing_time', 1)
            data['items_per_second'] = round(total / time_taken, 2)
        
        super().__init__(**data)


class IndexManagementResponse(BaseModel):
    """Modèle de réponse pour la gestion des index."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    action: str = Field(..., description="Action effectuée")
    success: bool = Field(..., description="Succès de l'opération")
    index_type: str = Field(..., description="Type d'index traité")
    details: Dict[str, Any] = Field(..., description="Détails de l'opération")
    processing_time: float = Field(..., description="Temps de traitement", ge=0.0)
    timestamp: float = Field(..., description="Timestamp de l'opération")
    error: Optional[str] = Field(None, description="Message d'erreur éventuel")
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        super().__init__(**data)


class SimpleTestResponse(BaseModel):
    """Modèle de réponse pour les tests simples."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    status: str = Field(..., description="Statut du test")
    message: str = Field(..., description="Message du test")
    timestamp: float = Field(..., description="Timestamp du test")
    version: str = Field(..., description="Version du service")
    test_data: Optional[Dict[str, Any]] = Field(None, description="Données de test")
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        super().__init__(**data)


class ConnectionStatusResponse(BaseModel):
    """Modèle de réponse pour le statut des connexions."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    service_name: str = Field(..., description="Nom du service")
    connections: Dict[str, Any] = Field(..., description="État des connexions")
    total_connections: int = Field(..., description="Nombre total de connexions", ge=0)
    healthy_connections: int = Field(..., description="Connexions saines", ge=0)
    failed_connections: int = Field(..., description="Connexions échouées", ge=0)
    timestamp: float = Field(..., description="Timestamp de la vérification")
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        super().__init__(**data)


class SearchCapabilitiesResponse(BaseModel):
    """Modèle de réponse pour les capacités de recherche."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    available_search_types: List[str] = Field(..., description="Types de recherche disponibles")
    elasticsearch_enabled: bool = Field(..., description="Elasticsearch disponible")
    qdrant_enabled: bool = Field(..., description="Qdrant disponible")
    features: Dict[str, bool] = Field(..., description="Fonctionnalités disponibles")
    limitations: List[str] = Field(default_factory=list, description="Limitations")
    performance_info: Dict[str, Any] = Field(..., description="Informations de performance")
    timestamp: float = Field(..., description="Timestamp de la vérification")
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        super().__init__(**data)


class SystemInfoResponse(BaseModel):
    """Modèle de réponse pour les informations système."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    service_name: str = Field(..., description="Nom du service")
    version: str = Field(..., description="Version du service")
    uptime: float = Field(..., description="Temps de fonctionnement", ge=0.0)
    environment: str = Field(..., description="Environnement")
    python_version: str = Field(..., description="Version Python")
    dependencies: Dict[str, str] = Field(..., description="Versions des dépendances")
    configuration: Dict[str, Any] = Field(..., description="Configuration active")
    timestamp: float = Field(..., description="Timestamp de la réponse")
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        super().__init__(**data)


# Classe de base corrigée pour Pydantic V2
class BaseResponse(BaseModel):
    """Classe de base pour toutes les réponses avec timestamp automatique."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True
    )
    
    timestamp: Optional[float] = Field(None, description="Timestamp de la réponse")
    
    def __init__(self, **data):
        if 'timestamp' not in data or data['timestamp'] is None:
            data['timestamp'] = datetime.now().timestamp()
        super().__init__(**data)


# Modèles de réponse pour les opérations spécialisées
class BenchmarkResponse(BaseModel):
    """Modèle de réponse pour les benchmarks de performance."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    test_name: str = Field(..., description="Nom du test de performance")
    total_queries: int = Field(..., description="Nombre total de requêtes", ge=0)
    successful_queries: int = Field(..., description="Requêtes réussies", ge=0)
    failed_queries: int = Field(..., description="Requêtes échouées", ge=0)
    average_response_time: float = Field(..., description="Temps de réponse moyen", ge=0.0)
    min_response_time: float = Field(..., description="Temps minimum", ge=0.0)
    max_response_time: float = Field(..., description="Temps maximum", ge=0.0)
    queries_per_second: float = Field(..., description="Requêtes par seconde", ge=0.0)
    benchmark_duration: float = Field(..., description="Durée du benchmark", ge=0.0)
    timestamp: float = Field(..., description="Timestamp du benchmark")
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        super().__init__(**data)


class ConfigurationResponse(BaseModel):
    """Modèle de réponse pour la configuration du service."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    service_name: str = Field(..., description="Nom du service")
    elasticsearch_config: Dict[str, Any] = Field(..., description="Configuration Elasticsearch")
    qdrant_config: Dict[str, Any] = Field(..., description="Configuration Qdrant")
    search_settings: Dict[str, Any] = Field(..., description="Paramètres de recherche")
    performance_settings: Dict[str, Any] = Field(..., description="Paramètres de performance")
    security_settings: Dict[str, Any] = Field(..., description="Paramètres de sécurité")
    timestamp: float = Field(..., description="Timestamp de la configuration")
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        super().__init__(**data)


class DiagnosticResponse(BaseModel):
    """Modèle de réponse pour le diagnostic approfondi."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    service_status: str = Field(..., description="Statut global du service")
    components: Dict[str, Dict[str, Any]] = Field(..., description="État des composants")
    performance_metrics: Dict[str, float] = Field(..., description="Métriques de performance")
    error_summary: Dict[str, int] = Field(..., description="Résumé des erreurs")
    recommendations: List[str] = Field(default_factory=list, description="Recommandations")
    detailed_logs: List[Dict[str, Any]] = Field(default_factory=list, description="Logs détaillés")
    diagnostic_duration: float = Field(..., description="Durée du diagnostic", ge=0.0)
    timestamp: float = Field(..., description="Timestamp du diagnostic")
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        super().__init__(**data)


# Export de tous les modèles
__all__ = [
    # Modèles de résultats principaux
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