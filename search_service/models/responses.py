"""
Modèles de réponses pour le service de recherche.
VERSION CORRIGÉE - Compatible Pydantic V2
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


# Export de tous les modèles
__all__ = [
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
    'BaseResponse'
]