"""
Modèles de requêtes pour le service de recherche.
Définit les structures de données pour les requêtes API entrantes.
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator


class SearchRequest(BaseModel):
    """Modèle de requête pour la recherche de transactions."""
    
    user_id: int = Field(..., description="ID de l'utilisateur", gt=0)
    query: str = Field(..., description="Requête de recherche", min_length=1)
    type: Optional[str] = Field("hybrid", description="Type de recherche: lexical, semantic, hybrid")
    limit: Optional[int] = Field(10, description="Nombre maximum de résultats", ge=1, le=50)
    use_reranking: Optional[bool] = Field(True, description="Utiliser le reranking des résultats")
    filters: Optional[Dict[str, Any]] = Field(None, description="Filtres additionnels")
    include_highlights: Optional[bool] = Field(True, description="Inclure la mise en évidence")
    
    @validator('query')
    def validate_query(cls, v):
        """Valide que la requête est une string non vide."""
        if not isinstance(v, str):
            raise ValueError(f"Query must be a string, got {type(v).__name__}")
        
        stripped = v.strip()
        if not stripped:
            raise ValueError("Query cannot be empty or only whitespace")
        
        if len(stripped) > 500:
            raise ValueError("Query too long (max 500 characters)")
        
        return stripped
    
    @validator('type')
    def validate_search_type(cls, v):
        """Valide le type de recherche."""
        if v is None:
            return "hybrid"
        
        if not isinstance(v, str):
            v = str(v)
        
        v = v.lower().strip()
        valid_types = ["lexical", "semantic", "hybrid"]
        
        if v not in valid_types:
            raise ValueError(f"Search type must be one of: {', '.join(valid_types)}")
        
        return v
    
    @validator('user_id')
    def validate_user_id(cls, v):
        """Valide l'ID utilisateur."""
        if not isinstance(v, int):
            try:
                v = int(v)
            except (ValueError, TypeError):
                raise ValueError(f"user_id must be an integer, got {type(v).__name__}")
        
        if v <= 0:
            raise ValueError("user_id must be positive")
        
        return v
    
    @validator('limit')
    def validate_limit(cls, v):
        """Valide la limite de résultats."""
        if v is None:
            return 10
        
        if not isinstance(v, int):
            try:
                v = int(v)
            except (ValueError, TypeError):
                raise ValueError(f"limit must be an integer, got {type(v).__name__}")
        
        if v <= 0:
            return 10
        
        return min(v, 50)  # Maximum 50 résultats
    
    @validator('filters')
    def validate_filters(cls, v):
        """Valide les filtres."""
        if v is None:
            return {}
        
        if not isinstance(v, dict):
            raise ValueError(f"filters must be a dict, got {type(v).__name__}")
        
        # Valider que les clés sont des strings et les valeurs sont sérialisables
        validated_filters = {}
        for key, value in v.items():
            if not isinstance(key, str):
                raise ValueError(f"Filter key must be string, got {type(key).__name__}")
            
            # Permettre None, string, int, float, bool
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise ValueError(f"Filter value for '{key}' must be string, int, float, bool or None")
            
            validated_filters[key] = value
        
        return validated_filters
    
    class Config:
        """Configuration du modèle."""
        schema_extra = {
            "example": {
                "user_id": 34,
                "query": "virement sepa ce mois-ci",
                "type": "hybrid",
                "limit": 10,
                "use_reranking": True,
                "filters": {
                    "category": "transfers",
                    "amount_min": 50.0
                },
                "include_highlights": True
            }
        }


class ReindexRequest(BaseModel):
    """Modèle de requête pour la réindexation."""
    
    user_id: int = Field(..., description="ID de l'utilisateur", gt=0)
    force_refresh: Optional[bool] = Field(False, description="Forcer le refresh complet")
    index_type: Optional[str] = Field("both", description="Type d'index: elasticsearch, qdrant, both")
    batch_size: Optional[int] = Field(100, description="Taille des lots pour l'indexation", ge=1, le=1000)
    
    @validator('user_id')
    def validate_user_id(cls, v):
        """Valide l'ID utilisateur."""
        if not isinstance(v, int):
            try:
                v = int(v)
            except (ValueError, TypeError):
                raise ValueError(f"user_id must be an integer, got {type(v).__name__}")
        
        if v <= 0:
            raise ValueError("user_id must be positive")
        
        return v
    
    @validator('index_type')
    def validate_index_type(cls, v):
        """Valide le type d'index."""
        if v is None:
            return "both"
        
        if not isinstance(v, str):
            v = str(v)
        
        v = v.lower().strip()
        valid_types = ["elasticsearch", "qdrant", "both"]
        
        if v not in valid_types:
            raise ValueError(f"index_type must be one of: {', '.join(valid_types)}")
        
        return v
    
    @validator('batch_size')
    def validate_batch_size(cls, v):
        """Valide la taille des lots."""
        if v is None:
            return 100
        
        if not isinstance(v, int):
            try:
                v = int(v)
            except (ValueError, TypeError):
                raise ValueError(f"batch_size must be an integer, got {type(v).__name__}")
        
        if v <= 0:
            return 100
        
        return min(v, 1000)  # Maximum 1000 pour éviter les timeouts
    
    class Config:
        """Configuration du modèle."""
        schema_extra = {
            "example": {
                "user_id": 34,
                "force_refresh": False,
                "index_type": "both",
                "batch_size": 100
            }
        }


class BulkIndexRequest(BaseModel):
    """Modèle de requête pour l'indexation en lot."""
    
    user_id: int = Field(..., description="ID de l'utilisateur", gt=0)
    transactions: List[Dict[str, Any]] = Field(..., description="Liste des transactions à indexer")
    index_type: Optional[str] = Field("both", description="Type d'index: elasticsearch, qdrant, both")
    refresh_after: Optional[bool] = Field(True, description="Refresh l'index après indexation")
    
    @validator('user_id')
    def validate_user_id(cls, v):
        """Valide l'ID utilisateur."""
        if not isinstance(v, int):
            try:
                v = int(v)
            except (ValueError, TypeError):
                raise ValueError(f"user_id must be an integer, got {type(v).__name__}")
        
        if v <= 0:
            raise ValueError("user_id must be positive")
        
        return v
    
    @validator('transactions')
    def validate_transactions(cls, v):
        """Valide la liste des transactions."""
        if not isinstance(v, list):
            raise ValueError(f"transactions must be a list, got {type(v).__name__}")
        
        if len(v) == 0:
            raise ValueError("transactions list cannot be empty")
        
        if len(v) > 1000:
            raise ValueError("Maximum 1000 transactions per bulk request")
        
        # Valider que chaque transaction est un dict avec les champs requis
        required_fields = ['id', 'description', 'amount', 'date']
        
        for i, transaction in enumerate(v):
            if not isinstance(transaction, dict):
                raise ValueError(f"Transaction {i} must be a dict, got {type(transaction).__name__}")
            
            for field in required_fields:
                if field not in transaction:
                    raise ValueError(f"Transaction {i} missing required field: {field}")
        
        return v
    
    @validator('index_type')
    def validate_index_type(cls, v):
        """Valide le type d'index."""
        if v is None:
            return "both"
        
        if not isinstance(v, str):
            v = str(v)
        
        v = v.lower().strip()
        valid_types = ["elasticsearch", "qdrant", "both"]
        
        if v not in valid_types:
            raise ValueError(f"index_type must be one of: {', '.join(valid_types)}")
        
        return v
    
    class Config:
        """Configuration du modèle."""
        schema_extra = {
            "example": {
                "user_id": 34,
                "transactions": [
                    {
                        "id": "tx_123",
                        "description": "Virement SEPA",
                        "amount": -150.00,
                        "date": "2024-06-25",
                        "merchant": "EDF",
                        "category": "utilities"
                    }
                ],
                "index_type": "both",
                "refresh_after": True
            }
        }


class DeleteUserDataRequest(BaseModel):
    """Modèle de requête pour la suppression des données utilisateur."""
    
    user_id: int = Field(..., description="ID de l'utilisateur", gt=0)
    confirm: bool = Field(..., description="Confirmation de suppression")
    index_type: Optional[str] = Field("both", description="Type d'index: elasticsearch, qdrant, both")
    
    @validator('user_id')
    def validate_user_id(cls, v):
        """Valide l'ID utilisateur."""
        if not isinstance(v, int):
            try:
                v = int(v)
            except (ValueError, TypeError):
                raise ValueError(f"user_id must be an integer, got {type(v).__name__}")
        
        if v <= 0:
            raise ValueError("user_id must be positive")
        
        return v
    
    @validator('confirm')
    def validate_confirm(cls, v):
        """Valide la confirmation."""
        if not isinstance(v, bool):
            raise ValueError("confirm must be a boolean")
        
        if not v:
            raise ValueError("confirm must be True to delete user data")
        
        return v
    
    @validator('index_type')
    def validate_index_type(cls, v):
        """Valide le type d'index."""
        if v is None:
            return "both"
        
        if not isinstance(v, str):
            v = str(v)
        
        v = v.lower().strip()
        valid_types = ["elasticsearch", "qdrant", "both"]
        
        if v not in valid_types:
            raise ValueError(f"index_type must be one of: {', '.join(valid_types)}")
        
        return v
    
    class Config:
        """Configuration du modèle."""
        schema_extra = {
            "example": {
                "user_id": 34,
                "confirm": True,
                "index_type": "both"
            }
        }


class QueryExpansionRequest(BaseModel):
    """Modèle de requête pour tester l'expansion de requête (debug)."""
    
    query: str = Field(..., description="Requête à expanser")
    include_debug: Optional[bool] = Field(False, description="Inclure les informations de debug")
    
    @validator('query')
    def validate_query(cls, v):
        """Valide la requête."""
        if not isinstance(v, str):
            # Pour les tests, on accepte la conversion
            v = str(v) if v is not None else ""
        
        if not v.strip():
            raise ValueError("Query cannot be empty")
        
        return v.strip()
    
    class Config:
        """Configuration du modèle."""
        schema_extra = {
            "example": {
                "query": "virement sepa ce mois-ci",
                "include_debug": True
            }
        }


# Classe de base pour toutes les requêtes
class BaseRequest(BaseModel):
    """Classe de base pour toutes les requêtes avec validation commune."""
    
    class Config:
        """Configuration commune."""
        validate_assignment = True
        use_enum_values = True
        allow_population_by_field_name = True
        
    @validator('*', pre=True)
    def prevent_dict_queries(cls, v, field):
        """Empêche l'utilisation d'objets dict pour les champs query."""
        if field.name == 'query' and isinstance(v, dict):
            raise ValueError("Query parameter cannot be a dict object")
        return v


# Export des modèles
__all__ = [
    'SearchRequest',
    'ReindexRequest', 
    'BulkIndexRequest',
    'DeleteUserDataRequest',
    'QueryExpansionRequest',
    'BaseRequest'
]