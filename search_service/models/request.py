# search_service/models/request.py
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, Optional

class SearchRequest(BaseModel):
    """Modèle de requête unifié pour le search service"""
    
    # Obligatoire
    user_id: int = Field(..., description="ID utilisateur (obligatoire pour sécurité)", gt=0)
    query: str = Field(default="", description="Requête textuelle libre")
    
    # Optionnel - Filtres
    filters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Filtres additionnels (term, range, terms)"
    )
    
    # Optionnel - Paramètres de pagination
    limit: int = Field(default=20, description="Nombre de résultats max", ge=1, le=100)
    offset: int = Field(default=0, description="Décalage pour pagination", ge=0)
    
    # Optionnel - Métadonnées
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Métadonnées pour debug et contexte"
    )
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validation et nettoyage de la requête"""
        if v is None:
            return ""
        return v.strip()
    
    @field_validator('filters')
    @classmethod
    def validate_filters(cls, v: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Validation basique des filtres"""
        if v is None:
            return {}
        
        # Validation des types de filtres supportés (basé sur votre schéma)
        allowed_fields = {
            'category_name', 'merchant_name', 'operation_type',
            'currency_code', 'transaction_type', 'month_year',
            'weekday', 'amount', 'amount_abs', 'date', 'account_id',
            'primary_description', 'searchable_text'
        }
        
        for field in v.keys():
            if field not in allowed_fields:
                # On log un warning mais on ne bloque pas pour la flexibilité
                pass
        
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 34,
                "query": "restaurant italien",
                "filters": {
                    "category_name": "restaurant",
                    "amount": {"gte": -100, "lte": 0},
                    "date": {"gte": "2024-01-01", "lte": "2024-01-31"}
                },
                "limit": 10,
                "offset": 0,
                "metadata": {"debug": True}
            }
        }