"""
Modèles de réponses pour le service de recherche.
Définit les structures de données pour les réponses API sortantes.
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime


class SearchResultItem(BaseModel):
    """Modèle pour un résultat de recherche individuel."""
    
    id: str = Field(..., description="ID unique du résultat")
    score: float = Field(..., description="Score de pertinence", ge=0.0)
    transaction: Dict[str, Any] = Field(..., description="Données de la transaction")
    highlights: Optional[Dict[str, List[str]]] = Field(None, description="Texte mis en évidence")
    search_type: Optional[str] = Field(None, description="Type de recherche utilisé")
    
    class Config:
        """Configuration du modèle."""
        schema_extra = {
            "example": {
                "id": "tx_123456",
                "score": 0.95,
                "transaction": {
                    "id": "tx_123456",
                    "user_id": 34,
                    "description": "Virement SEPA EDF",
                    "amount": -150.00,
                    "date": "2024-06-25",
                    "merchant_name": "EDF",
                    "category": "utilities"
                },
                "highlights": {
                    "description": ["<mark>Virement</mark> <mark>SEPA</mark> EDF"],
                    "merchant_name": ["<mark>EDF</mark>"]
                },
                "search_type": "hybrid"
            }
        }


class SearchResponse(BaseModel):
    """Modèle de réponse pour la recherche de transactions."""
    
    results: List[SearchResultItem] = Field(..., description="Liste des résultats")
    total: int = Field(..., description="Nombre total de résultats", ge=0)
    query_time: float = Field(..., description="Temps de requête en secondes", ge=0.0)
    search_type: str = Field(..., description="Type de recherche utilisé")
    user_id: int = Field(..., description="ID de l'utilisateur")
    query: str = Field(..., description="Requête originale")
    timestamp: Optional[float] = Field(None, description="Timestamp de la réponse")
    error: Optional[str] = Field(None, description="Message d'erreur éventuel")
    
    def __init__(self, **data):
        """Initialise la réponse avec timestamp automatique."""
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        super().__init__(**data)
    
    class Config:
        """Configuration du modèle."""
        schema_extra = {
            "example": {
                "results": [
                    {
                        "id": "tx_123456",
                        "score": 0.95,
                        "transaction": {
                            "id": "tx_123456",
                            "description": "Virement SEPA EDF",
                            "amount": -150.00,
                            "date": "2024-06-25"
                        },
                        "highlights": {
                            "description": ["<mark>Virement</mark> <mark>SEPA</mark>"]
                        }
                    }
                ],
                "total": 1,
                "query_time": 0.045,
                "search_type": "hybrid",
                "user_id": 34,
                "query": "virement sepa",
                "timestamp": 1719504000.0
            }
        }


class ReindexResponse(BaseModel):
    """Modèle de réponse pour la réindexation."""
    
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
        """Initialise la réponse avec timestamp automatique."""
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        super().__init__(**data)
    
    class Config:
        """Configuration du modèle."""
        schema_extra = {
            "example": {
                "success": True,
                "processed": 250,
                "indexed": 248,
                "errors": 2,
                "reindex_time": 12.5,
                "user_id": 34,
                "timestamp": 1719504000.0,
                "details": {
                    "elasticsearch_indexed": 248,
                    "qdrant_indexed": 248,
                    "batch_size": 100
                }
            }
        }


class BulkIndexResponse(BaseModel):
    """Modèle de réponse pour l'indexation en lot."""
    
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
        """Initialise la réponse avec timestamp automatique."""
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        super().__init__(**data)
    
    class Config:
        """Configuration du modèle."""
        schema_extra = {
            "example": {
                "success": True,
                "total_submitted": 100,
                "elasticsearch_indexed": 98,
                "qdrant_indexed": 98,
                "errors": 2,
                "processing_time": 3.2,
                "user_id": 34,
                "timestamp": 1719504000.0,
                "error_details": [
                    "Document tx_456: Invalid date format",
                    "Document tx_789: Missing required field"
                ]
            }
        }


class DeleteUserDataResponse(BaseModel):
    """Modèle de réponse pour la suppression des données utilisateur."""
    
    success: bool = Field(..., description="Succès de l'opération")
    user_id: int = Field(..., description="ID de l'utilisateur")
    elasticsearch_deleted: int = Field(..., description="Documents supprimés d'Elasticsearch", ge=0)
    qdrant_deleted: int = Field(..., description="Vecteurs supprimés de Qdrant", ge=0)
    delete_time: float = Field(..., description="Temps de suppression en secondes", ge=0.0)
    timestamp: Optional[float] = Field(None, description="Timestamp de la réponse")
    error: Optional[str] = Field(None, description="Message d'erreur éventuel")
    
    def __init__(self, **data):
        """Initialise la réponse avec timestamp automatique."""
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        super().__init__(**data)
    
    class Config:
        """Configuration du modèle."""
        schema_extra = {
            "example": {
                "success": True,
                "user_id": 34,
                "elasticsearch_deleted": 250,
                "qdrant_deleted": 250,
                "delete_time": 2.1,
                "timestamp": 1719504000.0
            }
        }


class UserStatsResponse(BaseModel):
    """Modèle de réponse pour les statistiques utilisateur."""
    
    user_id: int = Field(..., description="ID de l'utilisateur")
    elasticsearch: Dict[str, Any] = Field(..., description="Statistiques Elasticsearch")
    qdrant: Dict[str, Any] = Field(..., description="Statistiques Qdrant")
    last_update: Optional[str] = Field(None, description="Dernière mise à jour")
    timestamp: Optional[float] = Field(None, description="Timestamp de la réponse")
    
    def __init__(self, **data):
        """Initialise la réponse avec timestamp automatique."""
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        super().__init__(**data)
    
    class Config:
        """Configuration du modèle."""
        schema_extra = {
            "example": {
                "user_id": 34,
                "elasticsearch": {
                    "total_documents": 250,
                    "available": True,
                    "last_indexed": "2024-06-25T10:30:00Z"
                },
                "qdrant": {
                    "total_vectors": 250,
                    "available": True,
                    "last_indexed": "2024-06-25T10:30:00Z"
                },
                "last_update": "2024-06-25T10:30:00Z",
                "timestamp": 1719504000.0
            }
        }


class QueryExpansionResponse(BaseModel):
    """Modèle de réponse pour l'expansion de requête (debug)."""
    
    original_query: str = Field(..., description="Requête originale")
    query_type: str = Field(..., description="Type de la requête originale")
    expanded_terms: List[str] = Field(..., description="Termes expandus")
    expanded_count: int = Field(..., description="Nombre de termes expandus", ge=0)
    search_string: str = Field(..., description="Chaîne de recherche construite")
    timestamp: Optional[float] = Field(None, description="Timestamp de la réponse")
    debug_info: Optional[Dict[str, Any]] = Field(None, description="Informations de debug")
    
    def __init__(self, **data):
        """Initialise la réponse avec timestamp automatique."""
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        super().__init__(**data)
    
    class Config:
        """Configuration du modèle."""
        schema_extra = {
            "example": {
                "original_query": "vir sepa ce mois-ci",
                "query_type": "str",
                "expanded_terms": [
                    "vir", "sepa", "ce", "mois-ci", "virement", "virements", 
                    "paiement", "transfer", "mois-cis", "europeen"
                ],
                "expanded_count": 10,
                "search_string": "vir sepa ce mois-ci virement virements paiement transfer mois-cis europeen",
                "timestamp": 1719504000.0,
                "debug_info": {
                    "steps": ["validation", "tokenization", "expansion"],
                    "tokens": ["vir", "sepa", "ce", "mois-ci"],
                    "expansions_applied": ["financial_terms", "temporal_terms"]
                }
            }
        }


class HealthResponse(BaseModel):
    """Modèle de réponse pour le health check."""
    
    status: str = Field(..., description="Statut global du service")
    elasticsearch: Dict[str, Any] = Field(..., description="Statut Elasticsearch")
    qdrant: Dict[str, Any] = Field(..., description="Statut Qdrant")
    search_engine: Dict[str, Any] = Field(..., description="Statut du moteur de recherche")
    timestamp: float = Field(..., description="Timestamp de la vérification")
    uptime: Optional[float] = Field(None, description="Temps de fonctionnement en secondes")
    version: Optional[str] = Field(None, description="Version du service")
    
    class Config:
        """Configuration du modèle."""
        schema_extra = {
            "example": {
                "status": "healthy",
                "elasticsearch": {
                    "available": True,
                    "initialized": True,
                    "healthy": True,
                    "client_type": "bonsai",
                    "error": None
                },
                "qdrant": {
                    "available": True,
                    "initialized": True,
                    "healthy": True,
                    "error": None
                },
                "search_engine": {
                    "available": True,
                    "elasticsearch_enabled": True,
                    "qdrant_enabled": True
                },
                "timestamp": 1719504000.0,
                "uptime": 3600.0,
                "version": "1.0.0"
            }
        }


class ErrorResponse(BaseModel):
    """Modèle de réponse pour les erreurs."""
    
    error: str = Field(..., description="Type d'erreur")
    detail: str = Field(..., description="Description détaillée de l'erreur")
    timestamp: float = Field(..., description="Timestamp de l'erreur")
    request_id: Optional[str] = Field(None, description="ID de la requête")
    bug_detected: Optional[str] = Field(None, description="Bug spécifique détecté")
    solution: Optional[str] = Field(None, description="Solution suggérée")
    
    def __init__(self, **data):
        """Initialise la réponse avec timestamp automatique."""
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        super().__init__(**data)
    
    class Config:
        """Configuration du modèle."""
        schema_extra = {
            "example": {
                "error": "Query type validation error",
                "detail": "Query parameter must be a string, received a dict object",
                "timestamp": 1719504000.0,
                "bug_detected": "dict.lower() bug",
                "solution": "Ensure query parameter is properly validated as string"
            }
        }


class DebugClientResponse(BaseModel):
    """Modèle de réponse pour le debug des clients."""
    
    elastic_client: Dict[str, Any] = Field(..., description="Informations client Elasticsearch")
    qdrant_client: Dict[str, Any] = Field(..., description="Informations client Qdrant")
    search_engine: Dict[str, Any] = Field(..., description="Informations moteur de recherche")
    timestamp: float = Field(..., description="Timestamp de la vérification")
    
    def __init__(self, **data):
        """Initialise la réponse avec timestamp automatique."""
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        super().__init__(**data)
    
    class Config:
        """Configuration du modèle."""
        schema_extra = {
            "example": {
                "elastic_client": {
                    "available": True,
                    "type": "HybridElasticClient",
                    "initialized": True,
                    "client_type": "bonsai"
                },
                "qdrant_client": {
                    "available": True,
                    "type": "QdrantClient",
                    "initialized": True
                },
                "search_engine": {
                    "available": True,
                    "type": "SearchEngine",
                    "elasticsearch_enabled": True,
                    "qdrant_enabled": True
                },
                "timestamp": 1719504000.0
            }
        }


class DebugSearchResponse(BaseModel):
    """Modèle de réponse pour le debug de recherche."""
    
    client_type: str = Field(..., description="Type de client utilisé")
    user_id: int = Field(..., description="ID de l'utilisateur")
    query: str = Field(..., description="Requête de recherche")
    query_type: str = Field(..., description="Type de la requête")
    results_count: int = Field(..., description="Nombre de résultats", ge=0)
    results: List[Dict[str, Any]] = Field(..., description="Résultats de recherche")
    query_time: float = Field(..., description="Temps de requête en secondes", ge=0.0)
    timestamp: float = Field(..., description="Timestamp de la réponse")
    
    def __init__(self, **data):
        """Initialise la réponse avec timestamp automatique."""
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        super().__init__(**data)
    
    class Config:
        """Configuration du modèle."""
        schema_extra = {
            "example": {
                "client_type": "elasticsearch",
                "user_id": 34,
                "query": "virement sepa",
                "query_type": "str",
                "results_count": 5,
                "results": [
                    {
                        "id": "tx_123",
                        "score": 0.95,
                        "source": {
                            "description": "Virement SEPA EDF",
                            "amount": -150.00
                        }
                    }
                ],
                "query_time": 0.025,
                "timestamp": 1719504000.0
            }
        }


# Classe de base pour toutes les réponses
class BaseResponse(BaseModel):
    """Classe de base pour toutes les réponses avec timestamp automatique."""
    
    timestamp: Optional[float] = Field(None, description="Timestamp de la réponse")
    
    def __init__(self, **data):
        """Initialise la réponse avec timestamp automatique si non fourni."""
        if 'timestamp' not in data or data['timestamp'] is None:
            data['timestamp'] = datetime.now().timestamp()
        super().__init__(**data)
    
    class Config:
        """Configuration commune."""
        validate_assignment = True
        use_enum_values = True
        allow_population_by_field_name = True


# Modèles de réponse pour les opérations d'administration
class IndexInfoResponse(BaseModel):
    """Modèle de réponse pour les informations d'index."""
    
    index_name: str = Field(..., description="Nom de l'index")
    elasticsearch: Dict[str, Any] = Field(..., description="Informations Elasticsearch")
    qdrant: Dict[str, Any] = Field(..., description="Informations Qdrant")
    timestamp: float = Field(..., description="Timestamp de la vérification")
    
    def __init__(self, **data):
        """Initialise la réponse avec timestamp automatique."""
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        super().__init__(**data)
    
    class Config:
        """Configuration du modèle."""
        schema_extra = {
            "example": {
                "index_name": "harena_transactions",
                "elasticsearch": {
                    "exists": True,
                    "document_count": 1250,
                    "size_in_bytes": 2048576,
                    "health": "green"
                },
                "qdrant": {
                    "exists": True,
                    "vector_count": 1250,
                    "collection_config": {
                        "vector_size": 1536,
                        "distance": "Cosine"
                    }
                },
                "timestamp": 1719504000.0
            }
        }


class ServiceMetricsResponse(BaseModel):
    """Modèle de réponse pour les métriques du service."""
    
    uptime: float = Field(..., description="Temps de fonctionnement en secondes")
    total_searches: int = Field(..., description="Nombre total de recherches", ge=0)
    total_indexations: int = Field(..., description="Nombre total d'indexations", ge=0)
    average_query_time: float = Field(..., description="Temps moyen de requête", ge=0.0)
    error_rate: float = Field(..., description="Taux d'erreur", ge=0.0, le=1.0)
    last_24h: Dict[str, Any] = Field(..., description="Métriques des 24 dernières heures")
    timestamp: float = Field(..., description="Timestamp des métriques")
    
    def __init__(self, **data):
        """Initialise la réponse avec timestamp automatique."""
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        super().__init__(**data)
    
    class Config:
        """Configuration du modèle."""
        schema_extra = {
            "example": {
                "uptime": 86400.0,
                "total_searches": 15420,
                "total_indexations": 2340,
                "average_query_time": 0.087,
                "error_rate": 0.02,
                "last_24h": {
                    "searches": 1250,
                    "indexations": 150,
                    "errors": 25,
                    "avg_query_time": 0.089
                },
                "timestamp": 1719504000.0
            }
        }


class ValidationErrorResponse(BaseModel):
    """Modèle de réponse pour les erreurs de validation spécifiques."""
    
    error_type: str = Field(..., description="Type d'erreur de validation")
    field: str = Field(..., description="Champ en erreur")
    received_type: str = Field(..., description="Type reçu")
    expected_type: str = Field(..., description="Type attendu")
    received_value: Optional[str] = Field(None, description="Valeur reçue (tronquée)")
    message: str = Field(..., description="Message d'erreur détaillé")
    timestamp: float = Field(..., description="Timestamp de l'erreur")
    
    def __init__(self, **data):
        """Initialise la réponse avec timestamp automatique."""
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        super().__init__(**data)
    
    class Config:
        """Configuration du modèle."""
        schema_extra = {
            "example": {
                "error_type": "type_validation_error",
                "field": "query",
                "received_type": "dict",
                "expected_type": "str",
                "received_value": "{'search': 'virement'}",
                "message": "Field 'query' must be a string, received dict object",
                "timestamp": 1719504000.0
            }
        }


# Modèles de réponse pour les opérations par lots
class BatchOperationResponse(BaseModel):
    """Modèle de réponse pour les opérations par lots."""
    
    operation_type: str = Field(..., description="Type d'opération")
    total_items: int = Field(..., description="Nombre total d'éléments", ge=0)
    successful: int = Field(..., description="Nombre de succès", ge=0)
    failed: int = Field(..., description="Nombre d'échecs", ge=0)
    processing_time: float = Field(..., description="Temps de traitement total", ge=0.0)
    items_per_second: float = Field(..., description="Vitesse de traitement", ge=0.0)
    errors: List[str] = Field(default_factory=list, description="Liste des erreurs")
    timestamp: float = Field(..., description="Timestamp de l'opération")
    
    def __init__(self, **data):
        """Initialise la réponse avec timestamp et calculs automatiques."""
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        
        # Calculer items_per_second automatiquement
        if 'items_per_second' not in data and data.get('processing_time', 0) > 0:
            total = data.get('total_items', 0)
            time_taken = data.get('processing_time', 1)
            data['items_per_second'] = round(total / time_taken, 2)
        
        super().__init__(**data)
    
    class Config:
        """Configuration du modèle."""
        schema_extra = {
            "example": {
                "operation_type": "bulk_index",
                "total_items": 1000,
                "successful": 998,
                "failed": 2,
                "processing_time": 12.5,
                "items_per_second": 80.0,
                "errors": [
                    "Item 245: Invalid date format",
                    "Item 756: Missing required field"
                ],
                "timestamp": 1719504000.0
            }
        }


# Export de tous les modèles
__all__ = [
    # Modèles de résultats
    'SearchResultItem',
    'SearchResponse',
    
    # Modèles d'opérations
    'ReindexResponse',
    'BulkIndexResponse',
    'DeleteUserDataResponse',
    'BatchOperationResponse',
    
    # Modèles de statistiques et informations
    'UserStatsResponse',
    'HealthResponse',
    'IndexInfoResponse',
    'ServiceMetricsResponse',
    
    # Modèles de debug
    'QueryExpansionResponse',
    'DebugClientResponse',
    'DebugSearchResponse',
    
    # Modèles d'erreur
    'ErrorResponse',
    'ValidationErrorResponse',
    
    # Classe de base
    'BaseResponse'
]