from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class SearchResult(BaseModel):
    """Modèle d'un résultat de transaction basé sur votre schéma existant"""
    
    # Champs principaux
    transaction_id: str = Field(..., description="ID unique de la transaction")
    user_id: int = Field(..., description="ID utilisateur")
    account_id: Optional[int] = Field(None, description="ID du compte")
    
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
    
    # Métadonnées de recherche
    score: Optional[float] = Field(None, description="Score de pertinence")
    highlights: Optional[Dict[str, List[str]]] = Field(None, description="Surlignade des termes")

class SearchResponse(BaseModel):
    """Réponse standardisée du service de recherche"""
    
    # Résultats
    results: List[SearchResult] = Field(default_factory=list, description="Liste des résultats")
    
    # Métadonnées de la recherche
    total_hits: int = Field(..., description="Nombre total de résultats")
    returned_hits: int = Field(..., description="Nombre de résultats retournés")
    execution_time_ms: int = Field(..., description="Temps d'exécution en ms")
    
    # Informations Elasticsearch
    elasticsearch_took: Optional[int] = Field(None, description="Temps Elasticsearch en ms")
    cache_hit: bool = Field(default=False, description="Résultat depuis le cache")
    
    # Optionnel - Agrégations simples
    aggregations: Optional[Dict[str, Any]] = Field(None, description="Agrégations si demandées")
    
    # Debug et contexte
    query_info: Optional[Dict[str, Any]] = Field(None, description="Informations de debug")
    
    class Config:
        json_schema_extra = {
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
                        "score": 1.0
                    }
                ],
                "total_hits": 156,
                "returned_hits": 1,
                "execution_time_ms": 45,
                "elasticsearch_took": 23,
                "cache_hit": False
            }
        }