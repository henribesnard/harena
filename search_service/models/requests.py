"""
Modèles de requêtes pour le service de recherche.
VERSION CORRIGÉE - Compatible Pydantic V2
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator, ConfigDict


class SearchRequest(BaseModel):
    """Modèle de requête pour la recherche de transactions."""
    
    # Configuration Pydantic V2
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        str_strip_whitespace=True
    )
    
    user_id: int = Field(..., description="ID de l'utilisateur", gt=0)
    query: str = Field(..., description="Requête de recherche", min_length=1)
    type: Optional[str] = Field("hybrid", description="Type de recherche: lexical, semantic, hybrid")
    limit: Optional[int] = Field(10, description="Nombre maximum de résultats", ge=1, le=50)
    use_reranking: Optional[bool] = Field(True, description="Utiliser le reranking des résultats")
    filters: Optional[Dict[str, Any]] = Field(None, description="Filtres additionnels")
    include_highlights: Optional[bool] = Field(True, description="Inclure la mise en évidence")
    
    @field_validator('query')
    @classmethod
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
    
    @field_validator('type')
    @classmethod
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
    
    @field_validator('user_id')
    @classmethod
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
    
    @field_validator('limit')
    @classmethod
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
        
        return min(v, 50)
    
    @field_validator('filters')
    @classmethod
    def validate_filters(cls, v):
        """Valide les filtres."""
        if v is None:
            return {}
        
        if not isinstance(v, dict):
            raise ValueError(f"filters must be a dict, got {type(v).__name__}")
        
        validated_filters = {}
        for key, value in v.items():
            if not isinstance(key, str):
                raise ValueError(f"Filter key must be string, got {type(key).__name__}")
            
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise ValueError(f"Filter value for '{key}' must be string, int, float, bool or None")
            
            validated_filters[key] = value
        
        return validated_filters


class ReindexRequest(BaseModel):
    """Modèle de requête pour la réindexation."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True
    )
    
    user_id: int = Field(..., description="ID de l'utilisateur", gt=0)
    force_refresh: Optional[bool] = Field(False, description="Forcer le refresh complet")
    index_type: Optional[str] = Field("both", description="Type d'index: elasticsearch, qdrant, both")
    batch_size: Optional[int] = Field(100, description="Taille des lots", ge=1, le=1000)
    
    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v):
        if not isinstance(v, int):
            try:
                v = int(v)
            except (ValueError, TypeError):
                raise ValueError(f"user_id must be an integer, got {type(v).__name__}")
        
        if v <= 0:
            raise ValueError("user_id must be positive")
        
        return v
    
    @field_validator('index_type')
    @classmethod
    def validate_index_type(cls, v):
        if v is None:
            return "both"
        
        if not isinstance(v, str):
            v = str(v)
        
        v = v.lower().strip()
        valid_types = ["elasticsearch", "qdrant", "both"]
        
        if v not in valid_types:
            raise ValueError(f"index_type must be one of: {', '.join(valid_types)}")
        
        return v
    
    @field_validator('batch_size')
    @classmethod
    def validate_batch_size(cls, v):
        if v is None:
            return 100
        
        if not isinstance(v, int):
            try:
                v = int(v)
            except (ValueError, TypeError):
                raise ValueError(f"batch_size must be an integer, got {type(v).__name__}")
        
        if v <= 0:
            return 100
        
        return min(v, 1000)


class BulkIndexRequest(BaseModel):
    """Modèle de requête pour l'indexation en lot."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    user_id: int = Field(..., description="ID de l'utilisateur", gt=0)
    transactions: List[Dict[str, Any]] = Field(..., description="Liste des transactions")
    index_type: Optional[str] = Field("both", description="Type d'index")
    refresh_after: Optional[bool] = Field(True, description="Refresh après indexation")
    
    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v):
        if not isinstance(v, int):
            try:
                v = int(v)
            except (ValueError, TypeError):
                raise ValueError(f"user_id must be an integer, got {type(v).__name__}")
        
        if v <= 0:
            raise ValueError("user_id must be positive")
        
        return v
    
    @field_validator('transactions')
    @classmethod
    def validate_transactions(cls, v):
        if not isinstance(v, list):
            raise ValueError(f"transactions must be a list, got {type(v).__name__}")
        
        if len(v) == 0:
            raise ValueError("transactions list cannot be empty")
        
        if len(v) > 1000:
            raise ValueError("Maximum 1000 transactions per bulk request")
        
        required_fields = ['id', 'description', 'amount', 'date']
        
        for i, transaction in enumerate(v):
            if not isinstance(transaction, dict):
                raise ValueError(f"Transaction {i} must be a dict, got {type(transaction).__name__}")
            
            for field in required_fields:
                if field not in transaction:
                    raise ValueError(f"Transaction {i} missing required field: {field}")
        
        return v
    
    @field_validator('index_type')
    @classmethod
    def validate_index_type(cls, v):
        if v is None:
            return "both"
        
        if not isinstance(v, str):
            v = str(v)
        
        v = v.lower().strip()
        valid_types = ["elasticsearch", "qdrant", "both"]
        
        if v not in valid_types:
            raise ValueError(f"index_type must be one of: {', '.join(valid_types)}")
        
        return v


class DeleteUserDataRequest(BaseModel):
    """Modèle de requête pour la suppression des données utilisateur."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    user_id: int = Field(..., description="ID de l'utilisateur", gt=0)
    confirm: bool = Field(..., description="Confirmation de suppression")
    index_type: Optional[str] = Field("both", description="Type d'index")
    
    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v):
        if not isinstance(v, int):
            try:
                v = int(v)
            except (ValueError, TypeError):
                raise ValueError(f"user_id must be an integer, got {type(v).__name__}")
        
        if v <= 0:
            raise ValueError("user_id must be positive")
        
        return v
    
    @field_validator('confirm')
    @classmethod
    def validate_confirm(cls, v):
        if not isinstance(v, bool):
            raise ValueError("confirm must be a boolean")
        
        if not v:
            raise ValueError("confirm must be True to delete user data")
        
        return v
    
    @field_validator('index_type')
    @classmethod
    def validate_index_type(cls, v):
        if v is None:
            return "both"
        
        if not isinstance(v, str):
            v = str(v)
        
        v = v.lower().strip()
        valid_types = ["elasticsearch", "qdrant", "both"]
        
        if v not in valid_types:
            raise ValueError(f"index_type must be one of: {', '.join(valid_types)}")
        
        return v


class QueryExpansionRequest(BaseModel):
    """Modèle de requête pour tester l'expansion de requête."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    query: str = Field(..., description="Requête à expanser")
    include_debug: Optional[bool] = Field(False, description="Inclure les informations de debug")
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        if not isinstance(v, str):
            v = str(v) if v is not None else ""
        
        if not v.strip():
            raise ValueError("Query cannot be empty")
        
        return v.strip()


# Classe de base corrigée pour Pydantic V2
class BaseRequest(BaseModel):
    """Classe de base pour toutes les requêtes avec validation commune."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        str_strip_whitespace=True
    )
    
    @field_validator('*', mode='before')
    @classmethod
    def prevent_dict_queries(cls, v, info):
        """Empêche l'utilisation d'objets dict pour les champs query."""
        if info.field_name == 'query' and isinstance(v, dict):
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
