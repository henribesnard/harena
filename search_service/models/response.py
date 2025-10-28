"""Schémas de réponse pour le service de recherche."""

from pydantic import BaseModel, Field, ConfigDict, field_serializer
from typing import List, Optional, Dict, Any, Union

class SearchResult(BaseModel):
    """Modèle d'un résultat de transaction basé sur votre schéma existant"""
    
    # Champs principaux
    transaction_id: str = Field(..., description="ID unique de la transaction")
    user_id: int = Field(..., description="ID utilisateur")
    account_id: Optional[int] = Field(None, description="ID du compte")

    # Informations du compte (dénormalisées pour performance)
    account_name: Optional[str] = Field(None, description="Nom du compte")
    account_type: Optional[str] = Field(None, description="Type du compte (checking, savings, etc.)")

    # Montants
    amount: float = Field(..., description="Montant avec signe")
    amount_abs: float = Field(..., description="Montant en valeur absolue")
    currency_code: str = Field(default="EUR", description="Code devise")
    transaction_type: str = Field(..., description="Type: debit/credit")
    
    # Informations temporelles
    date: str = Field(..., description="Date de la transaction (YYYY-MM-DD)")
    month_year: Optional[str] = Field(None, description="Format YYYY-MM")
    weekday: Optional[str] = Field(None, description="Jour de la semaine")
    
    # Descriptions et catégories
    primary_description: str = Field(..., description="Description principale")
    merchant_name: Optional[str] = Field(None, description="Nom du marchand")
    category_name: Optional[str] = Field(None, description="Catégorie")
    operation_type: Optional[str] = Field(None, description="Type d'opération")

    # ✅ CORRECTION CRITIQUE : Métadonnées de recherche avec garanties
    score: float = Field(
        default=0.0, 
        description="Score de pertinence", 
        alias="_score"
    )
    highlights: Optional[Dict[str, List[str]]] = Field(
        default=None, 
        description="Surlignade des termes"
    )

    # ✅ NOUVEAU : Garantir que _score apparaît toujours dans le JSON
    @field_serializer('score', when_used='always')
    def serialize_score(self, value: float) -> float:
        """Garantit que _score apparaît toujours, même si 0.0"""
        return value if value is not None else 0.0

    model_config = ConfigDict(
        populate_by_name=True,
        # ✅ CORRECTION : Garantir inclusion des champs par défaut
        use_enum_values=True,
        validate_assignment=True,
        json_schema_extra={
            "example": {
                "transaction_id": "user_34_tx_12345",
                "user_id": 34,
                "account_id": 101,
                "account_name": "Compte Courant",
                "account_type": "checking",
                "amount": -45.67,
                "amount_abs": 45.67,
                "currency_code": "EUR",
                "transaction_type": "debit",
                "date": "2024-01-15",
                "month_year": "2024-01",
                "weekday": "Monday",
                "primary_description": "RESTAURANT LE BISTROT",
                "merchant_name": "Le Bistrot",
                "category_name": "Restaurant",
                "operation_type": "card_payment",
                "_score": 1.0,
                "highlights": {"primary_description": ["<em>bistrot</em>"]}
            }
        }
    )

class AccountResult(BaseModel):
    """Modèle d'un résultat de compte pour les recherches sur l'index accounts"""
    
    # Champs obligatoires pour les comptes
    user_id: int = Field(..., description="ID utilisateur")
    account_id: int = Field(..., description="ID du compte")
    account_name: str = Field(..., description="Nom du compte")
    account_type: str = Field(..., description="Type de compte")
    account_balance: float = Field(..., description="Solde du compte")
    account_currency: str = Field(default="EUR", description="Devise du compte")
    
    # Métadonnées de recherche
    score: float = Field(
        default=0.0, 
        description="Score de pertinence", 
        alias="_score"
    )
    highlights: Optional[Dict[str, List[str]]] = Field(
        default=None, 
        description="Surlignade des termes"
    )

    @field_serializer('score', when_used='always')
    def serialize_score(self, value: float) -> float:
        """Garantit que _score apparaît toujours, même si 0.0"""
        return value if value is not None else 0.0

    model_config = ConfigDict(
        populate_by_name=True,
        use_enum_values=True,
        validate_assignment=True,
        json_schema_extra={
            "example": {
                "user_id": 34,
                "account_id": 101,
                "account_name": "Compte courant",
                "account_type": "checking",
                "account_balance": 1000.0,
                "account_currency": "EUR",
                "_score": 1.0,
                "highlights": {"account_name": ["<em>courant</em>"]}
            }
        }
    )

class SearchResponse(BaseModel):
    """Réponse standardisée du service de recherche"""
    
    # Résultats
    results: List[SearchResult] = Field(default_factory=list, description="Liste des résultats")
    
    # Métadonnées de la recherche
    total_results: int = Field(..., description="Nombre total de résultats")
    returned_results: int = Field(..., description="Nombre de résultats retournés")
    page: int = Field(..., description="Numéro de page courant (1-indexé)")
    page_size: int = Field(..., description="Nombre de résultats par page")
    total_pages: int = Field(..., description="Nombre total de pages disponibles")
    has_more_results: bool = Field(..., description="Indique s'il reste d'autres résultats")
    processing_time_ms: int = Field(..., description="Temps de traitement en ms")
    
    # Informations Elasticsearch
    elasticsearch_took: Optional[int] = Field(None, description="Temps Elasticsearch en ms")
    cache_hit: bool = Field(default=False, description="Résultat depuis le cache")
    
    # Optionnel - Agrégations simples
    aggregations: Optional[Dict[str, Any]] = Field(None, description="Agrégations si demandées")
    
    # Debug et contexte
    query_info: Optional[Dict[str, Any]] = Field(None, description="Informations de debug")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "results": [
                    {
                        "transaction_id": "user_34_tx_12345",
                        "user_id": 34,
                        "account_id": 101,
                        "amount": -45.67,
                        "amount_abs": 45.67,
                        "currency_code": "EUR",
                        "transaction_type": "debit",
                        "date": "2024-01-15",
                        "month_year": "2024-01",
                        "weekday": "Monday",
                        "primary_description": "RESTAURANT LE BISTROT",
                        "merchant_name": "Le Bistrot",
                        "category_name": "Restaurant",
                        "operation_type": "card_payment",
                        "_score": 1.0,
                        "highlights": {"primary_description": ["<em>bistrot</em>"]}
                    }
                ],
                "total_results": 156,
                "returned_results": 1,
                "page": 1,
                "page_size": 10,
                "total_pages": 16,
                "has_more_results": True,
                "processing_time_ms": 45,
                "elasticsearch_took": 23,
                "cache_hit": False,
                "aggregations": {"category_name": {"buckets": [{"key": "Restaurant", "doc_count": 120}]}},
                "query_info": {"raw_query": "restaurant italien"}
            }
        }
    )