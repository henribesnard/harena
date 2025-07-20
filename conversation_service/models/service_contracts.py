"""
Contrats d'interface avec les services externes
Standardisation des échanges Search Service
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum


class SearchFilterOperator(str, Enum):
    """Opérateurs de filtrage"""
    EQUALS = "equals"
    CONTAINS = "contains"
    RANGE = "range"
    GT = "gt"
    LT = "lt"
    IN = "in"


class SearchFilter(BaseModel):
    """Filtre de recherche standardisé"""
    field: str = Field(..., description="Champ à filtrer")
    operator: SearchFilterOperator = Field(..., description="Opérateur de filtrage")
    value: Any = Field(..., description="Valeur de filtrage")
    boost: Optional[float] = Field(1.0, description="Boost de pertinence")
    
    class Config:
        schema_extra = {
            "example": {
                "field": "category",
                "operator": "equals",
                "value": "restaurant",
                "boost": 1.5
            }
        }


class SearchAggregation(BaseModel):
    """Agrégation de recherche"""
    name: str = Field(..., description="Nom de l'agrégation")
    type: str = Field(..., description="Type d'agrégation (sum, avg, count, etc.)")
    field: str = Field(..., description="Champ à agréger")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "name": "total_amount",
                "type": "sum",
                "field": "amount",
                "parameters": {"currency": "EUR"}
            }
        }


class SearchServiceMetadata(BaseModel):
    """Métadonnées requête Search Service"""
    user_id: int = Field(..., description="ID utilisateur pour sécurité")
    intent_type: str = Field(..., description="Intention détectée")
    confidence_score: float = Field(..., description="Score confiance intention")
    detection_level: str = Field(..., description="Niveau détection (L0/L1/L2)")
    agent_name: str = Field(default="conversation_service", description="Service émetteur")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": 12345,
                "intent_type": "expense_analysis",
                "confidence_score": 0.92,
                "detection_level": "L1_lightweight",
                "agent_name": "conversation_service",
                "timestamp": "2025-01-20T10:30:00Z"
            }
        }


class SearchServiceQuery(BaseModel):
    """Requête standardisée pour Search Service"""
    query_text: Optional[str] = Field(None, description="Texte de recherche")
    filters: List[SearchFilter] = Field(default_factory=list, description="Filtres de recherche")
    aggregations: List[SearchAggregation] = Field(default_factory=list, description="Agrégations demandées")
    limit: int = Field(default=20, ge=1, le=100, description="Nombre de résultats")
    offset: int = Field(default=0, ge=0, description="Offset pagination")
    sort_by: Optional[str] = Field(None, description="Champ de tri")
    sort_order: str = Field(default="desc", description="Ordre de tri")
    timeout_ms: int = Field(default=5000, description="Timeout requête")
    query_metadata: SearchServiceMetadata = Field(..., description="Métadonnées requête")
    
    @validator('filters')
    def validate_mandatory_user_filter(cls, v, values):
        """Validation sécurité: filtre user_id obligatoire"""
        user_filter_exists = any(
            f.field == "user_id" for f in v
        )
        if not user_filter_exists and 'query_metadata' in values:
            # Ajout automatique filtre sécurité
            user_filter = SearchFilter(
                field="user_id",
                operator=SearchFilterOperator.EQUALS,
                value=values['query_metadata'].user_id
            )
            v.append(user_filter)
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "query_text": "restaurant",
                "filters": [
                    {
                        "field": "user_id",
                        "operator": "equals",
                        "value": 12345
                    },
                    {
                        "field": "category",
                        "operator": "equals", 
                        "value": "restaurant"
                    }
                ],
                "aggregations": [
                    {
                        "name": "total_spent",
                        "type": "sum",
                        "field": "amount"
                    }
                ],
                "limit": 20,
                "query_metadata": {
                    "user_id": 12345,
                    "intent_type": "expense_analysis",
                    "confidence_score": 0.92,
                    "detection_level": "L1_lightweight"
                }
            }
        }


class SearchServiceResponse(BaseModel):
    """Réponse standardisée Search Service"""
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Résultats de recherche")
    aggregations: Optional[Dict[str, Any]] = Field(None, description="Résultats agrégations")
    total_hits: int = Field(..., description="Nombre total de résultats")
    returned_hits: int = Field(..., description="Nombre de résultats retournés")
    execution_time_ms: int = Field(..., description="Temps d'exécution")
    timed_out: bool = Field(default=False, description="Timeout atteint")
    query_id: str = Field(..., description="ID unique de requête")
    
    @validator('returned_hits')
    def validate_returned_hits(cls, v, values):
        if 'results' in values and v != len(values['results']):
            raise ValueError('returned_hits must match results length')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "results": [
                    {
                        "transaction_id": "tx_123",
                        "amount": 25.50,
                        "merchant": "Restaurant Le Bistrot",
                        "date": "2025-01-19",
                        "category": "restaurant"
                    }
                ],
                "aggregations": {
                    "total_spent": 156.30,
                    "transaction_count": 8
                },
                "total_hits": 25,
                "returned_hits": 1,
                "execution_time_ms": 45,
                "timed_out": False,
                "query_id": "search_456789"
            }
        }
