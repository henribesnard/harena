"""
Modèles de données pour le service de recherche.

Ce module définit les structures de données utilisées pour les requêtes
et réponses du service de recherche hybride.
"""
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from enum import Enum


class SearchType(str, Enum):
    """Types de recherche disponibles."""
    HYBRID = "hybrid"
    LEXICAL = "lexical"
    SEMANTIC = "semantic"


class SearchQuery(BaseModel):
    """Modèle de requête de recherche."""
    user_id: int = Field(..., description="ID de l'utilisateur")
    query: str = Field(..., min_length=1, description="Texte de recherche")
    search_type: SearchType = Field(default=SearchType.HYBRID, description="Type de recherche")
    
    # Pagination
    limit: int = Field(default=20, ge=1, le=100, description="Nombre de résultats")
    offset: int = Field(default=0, ge=0, description="Décalage pour pagination")
    
    # Filtres optionnels
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Filtres additionnels")
    date_from: Optional[datetime] = Field(default=None, description="Date de début")
    date_to: Optional[datetime] = Field(default=None, description="Date de fin")
    amount_min: Optional[float] = Field(default=None, description="Montant minimum")
    amount_max: Optional[float] = Field(default=None, description="Montant maximum")
    categories: Optional[List[int]] = Field(default=None, description="IDs de catégories")
    account_ids: Optional[List[int]] = Field(default=None, description="IDs de comptes")
    transaction_types: Optional[List[str]] = Field(default=None, description="Types de transaction")
    
    # Paramètres de recherche hybride
    lexical_weight: float = Field(default=0.5, ge=0, le=1, description="Poids recherche lexicale")
    semantic_weight: float = Field(default=0.5, ge=0, le=1, description="Poids recherche sémantique")
    
    # Options avancées
    use_reranking: bool = Field(default=True, description="Utiliser le reranking")
    include_highlights: bool = Field(default=True, description="Inclure les highlights")
    include_explanations: bool = Field(default=False, description="Inclure les explications de score")
    
    @validator('semantic_weight')
    def weights_sum_to_one(cls, v, values):
        """Vérifie que les poids somment à 1."""
        if 'lexical_weight' in values:
            if abs(values['lexical_weight'] + v - 1.0) > 0.001:
                raise ValueError('lexical_weight + semantic_weight must equal 1.0')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 123,
                "query": "restaurant paris",
                "search_type": "hybrid",
                "limit": 20,
                "filters": {
                    "merchant_type": "restaurant"
                },
                "date_from": "2024-01-01T00:00:00",
                "amount_min": 10.0,
                "amount_max": 100.0
            }
        }


class SearchResult(BaseModel):
    """Résultat individuel de recherche."""
    # Identifiants
    transaction_id: int = Field(..., description="ID de la transaction")
    user_id: int = Field(..., description="ID de l'utilisateur")
    
    # Données principales
    description: str = Field(..., description="Description de la transaction")
    amount: float = Field(..., description="Montant")
    date: datetime = Field(..., description="Date de la transaction")
    currency: str = Field(default="EUR", description="Devise")
    
    # Catégorisation
    category_id: Optional[int] = Field(default=None, description="ID de catégorie")
    category_name: Optional[str] = Field(default=None, description="Nom de catégorie")
    merchant_name: Optional[str] = Field(default=None, description="Nom du marchand")
    
    # Scores et métadonnées
    score: float = Field(..., description="Score de pertinence global")
    lexical_score: Optional[float] = Field(default=None, description="Score lexical")
    semantic_score: Optional[float] = Field(default=None, description="Score sémantique")
    rerank_score: Optional[float] = Field(default=None, description="Score de reranking")
    
    # Highlights
    highlights: Optional[Dict[str, List[str]]] = Field(
        default=None, 
        description="Texte surligné par champ"
    )
    
    # Explications (optionnel)
    explanations: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Explications détaillées du scoring"
    )
    
    # Métadonnées additionnelles
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Métadonnées supplémentaires"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": 12345,
                "user_id": 123,
                "description": "RESTAURANT LE PETIT PARIS",
                "amount": -45.50,
                "date": "2024-01-15T19:30:00",
                "currency": "EUR",
                "category_id": 15,
                "category_name": "Restaurants",
                "merchant_name": "Le Petit Paris",
                "score": 0.95,
                "lexical_score": 0.90,
                "semantic_score": 0.88,
                "rerank_score": 0.95,
                "highlights": {
                    "description": ["<em>RESTAURANT</em> LE PETIT <em>PARIS</em>"]
                }
            }
        }


class SearchResponse(BaseModel):
    """Réponse complète de recherche."""
    # Requête originale
    query: str = Field(..., description="Requête originale")
    search_type: SearchType = Field(..., description="Type de recherche effectué")
    
    # Résultats
    results: List[SearchResult] = Field(..., description="Liste des résultats")
    total_found: int = Field(..., description="Nombre total de résultats trouvés")
    
    # Pagination
    limit: int = Field(..., description="Limite demandée")
    offset: int = Field(..., description="Décalage appliqué")
    has_more: bool = Field(..., description="Indique s'il y a plus de résultats")
    
    # Performance
    processing_time: float = Field(..., description="Temps de traitement en secondes")
    timings: Optional[Dict[str, float]] = Field(
        default=None,
        description="Détail des temps par étape"
    )
    
    # Métadonnées
    filters_applied: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Filtres effectivement appliqués"
    )
    suggestions: Optional[List[str]] = Field(
        default=None,
        description="Suggestions de recherche"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "restaurant paris",
                "search_type": "hybrid",
                "results": [
                    {
                        "transaction_id": 12345,
                        "user_id": 123,
                        "description": "RESTAURANT LE PETIT PARIS",
                        "amount": -45.50,
                        "date": "2024-01-15T19:30:00",
                        "score": 0.95
                    }
                ],
                "total_found": 42,
                "limit": 20,
                "offset": 0,
                "has_more": True,
                "processing_time": 0.125,
                "timings": {
                    "lexical_search": 0.045,
                    "semantic_search": 0.060,
                    "reranking": 0.020
                }
            }
        }


class HealthStatus(BaseModel):
    """État de santé du service."""
    status: Literal["healthy", "degraded", "unhealthy"] = Field(..., description="État global")
    
    # Statut des composants
    elasticsearch_status: bool = Field(..., description="État Elasticsearch")
    qdrant_status: bool = Field(..., description="État Qdrant")
    cohere_status: bool = Field(..., description="État Cohere")
    openai_status: bool = Field(..., description="État OpenAI")
    
    # Métriques
    response_time_ms: Optional[float] = Field(default=None, description="Temps de réponse moyen")
    requests_per_minute: Optional[float] = Field(default=None, description="Requêtes par minute")
    cache_hit_rate: Optional[float] = Field(default=None, description="Taux de cache hit")
    
    # Informations système
    version: str = Field(..., description="Version du service")
    uptime_seconds: float = Field(..., description="Temps de fonctionnement")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp")