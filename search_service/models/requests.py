"""
Modèles de requêtes entrantes pour le service de recherche.

Ce module définit les modèles Pydantic pour toutes les requêtes
acceptées par l'API de recherche.
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
from search_service.models.search_types import (
    SearchType, SortOrder, TransactionType, 
    DEFAULT_SEARCH_LIMIT, MAX_SEARCH_LIMIT,
    DEFAULT_SIMILARITY_THRESHOLD, MIN_SIMILARITY_THRESHOLD, MAX_SIMILARITY_THRESHOLD,
    DEFAULT_LEXICAL_WEIGHT, DEFAULT_SEMANTIC_WEIGHT
)
from search_service.models.filters import AdvancedFilters


class SearchRequest(BaseModel):
    """Requête de recherche standard."""
    query: str = Field(..., min_length=1, max_length=500, description="Terme de recherche")
    user_id: int = Field(..., description="ID de l'utilisateur")
    search_type: SearchType = Field(default=SearchType.HYBRID, description="Type de recherche")
    limit: int = Field(default=DEFAULT_SEARCH_LIMIT, ge=1, le=MAX_SEARCH_LIMIT, description="Nombre max de résultats")
    offset: int = Field(default=0, ge=0, description="Décalage pour pagination")
    sort_order: SortOrder = Field(default=SortOrder.RELEVANCE, description="Ordre de tri")
    
    # Paramètres spécifiques à la recherche sémantique
    similarity_threshold: float = Field(
        default=DEFAULT_SIMILARITY_THRESHOLD,
        ge=MIN_SIMILARITY_THRESHOLD,
        le=MAX_SIMILARITY_THRESHOLD,
        description="Seuil de similarité pour recherche sémantique"
    )
    
    # Paramètres pour la recherche hybride
    lexical_weight: float = Field(
        default=DEFAULT_LEXICAL_WEIGHT,
        ge=0.0,
        le=1.0,
        description="Poids de la recherche lexicale (0-1)"
    )
    semantic_weight: float = Field(
        default=DEFAULT_SEMANTIC_WEIGHT,
        ge=0.0,
        le=1.0,
        description="Poids de la recherche sémantique (0-1)"
    )
    
    # Filtres simples
    transaction_type: TransactionType = Field(default=TransactionType.ALL, description="Type de transaction")
    account_ids: Optional[List[int]] = Field(default=None, description="IDs de comptes à inclure")
    category_ids: Optional[List[int]] = Field(default=None, description="IDs de catégories à inclure")
    
    # Options d'affichage
    include_highlights: bool = Field(default=True, description="Inclure le highlighting des résultats")
    include_metadata: bool = Field(default=True, description="Inclure les métadonnées détaillées")
    
    @validator("semantic_weight")
    def validate_weights_sum(cls, v, values):
        """Valide que les poids lexical + sémantique = 1.0 pour recherche hybride."""
        lexical_weight = values.get("lexical_weight", DEFAULT_LEXICAL_WEIGHT)
        search_type = values.get("search_type", SearchType.HYBRID)
        
        if search_type == SearchType.HYBRID:
            total_weight = lexical_weight + v
            if abs(total_weight - 1.0) > 0.01:  # Tolérance pour erreurs de floating point
                raise ValueError("lexical_weight + semantic_weight must equal 1.0 for hybrid search")
        
        return v
    
    @validator("query")
    def validate_query_content(cls, v):
        """Valide le contenu de la requête."""
        # Supprimer les espaces en début/fin
        v = v.strip()
        
        # Vérifier qu'il ne s'agit pas que d'espaces
        if not v:
            raise ValueError("Query cannot be empty or only whitespace")
        
        # Interdire certains caractères potentiellement dangereux
        forbidden_chars = ["<", ">", "&", "'", '"']
        if any(char in v for char in forbidden_chars):
            raise ValueError("Query contains forbidden characters")
        
        return v


class AdvancedSearchRequest(SearchRequest):
    """Requête de recherche avancée avec filtres complexes."""
    filters: Optional[AdvancedFilters] = Field(default=None, description="Filtres avancés")
    
    # Options de recherche textuelle avancée
    exact_phrase: Optional[str] = Field(default=None, description="Phrase exacte à rechercher")
    exclude_terms: Optional[List[str]] = Field(default=None, description="Termes à exclure")
    required_terms: Optional[List[str]] = Field(default=None, description="Termes obligatoires")
    
    # Options de recherche par période
    date_from: Optional[str] = Field(default=None, description="Date de début (YYYY-MM-DD)")
    date_to: Optional[str] = Field(default=None, description="Date de fin (YYYY-MM-DD)")
    
    # Options de recherche par montant
    amount_min: Optional[float] = Field(default=None, description="Montant minimum")
    amount_max: Optional[float] = Field(default=None, description="Montant maximum")
    
    # Options de performance
    use_cache: bool = Field(default=True, description="Utiliser le cache de résultats")
    explain_scoring: bool = Field(default=False, description="Inclure explication du scoring")
    
    @validator("date_from", "date_to")
    def validate_date_format(cls, v):
        """Valide le format des dates."""
        if v is not None:
            from datetime import datetime
            try:
                datetime.strptime(v, "%Y-%m-%d")
            except ValueError:
                raise ValueError("Date must be in YYYY-MM-DD format")
        return v
    
    @validator("date_to")
    def validate_date_range(cls, v, values):
        """Valide que date_to > date_from."""
        if v is not None and values.get("date_from") is not None:
            from datetime import datetime
            date_from = datetime.strptime(values["date_from"], "%Y-%m-%d")
            date_to = datetime.strptime(v, "%Y-%m-%d")
            
            if date_to <= date_from:
                raise ValueError("date_to must be after date_from")
        return v
    
    @validator("amount_max")
    def validate_amount_range(cls, v, values):
        """Valide que amount_max > amount_min."""
        if v is not None and values.get("amount_min") is not None:
            if v <= values["amount_min"]:
                raise ValueError("amount_max must be greater than amount_min")
        return v


class SuggestionsRequest(BaseModel):
    """Requête pour l'auto-complétion et suggestions."""
    partial_query: str = Field(..., min_length=1, max_length=100, description="Début de requête")
    user_id: int = Field(..., description="ID de l'utilisateur")
    max_suggestions: int = Field(default=10, ge=1, le=20, description="Nombre max de suggestions")
    
    # Types de suggestions
    include_merchants: bool = Field(default=True, description="Inclure suggestions de marchands")
    include_categories: bool = Field(default=True, description="Inclure suggestions de catégories") 
    include_descriptions: bool = Field(default=True, description="Inclure suggestions de descriptions")
    include_recent_searches: bool = Field(default=True, description="Inclure recherches récentes")
    
    @validator("partial_query")
    def validate_partial_query(cls, v):
        """Valide le contenu de la requête partielle."""
        v = v.strip()
        if not v:
            raise ValueError("partial_query cannot be empty")
        return v


class StatsRequest(BaseModel):
    """Requête pour les statistiques de recherche utilisateur."""
    user_id: int = Field(..., description="ID de l'utilisateur")
    period_days: int = Field(default=30, ge=1, le=365, description="Période en jours")
    include_query_stats: bool = Field(default=True, description="Inclure stats des requêtes")
    include_result_stats: bool = Field(default=True, description="Inclure stats des résultats")
    include_performance_stats: bool = Field(default=False, description="Inclure stats de performance")


class BulkSearchRequest(BaseModel):
    """Requête pour la recherche en lot (pour analyses)."""
    queries: List[str] = Field(..., min_items=1, max_items=10, description="Liste de requêtes")
    user_id: int = Field(..., description="ID de l'utilisateur")
    search_type: SearchType = Field(default=SearchType.HYBRID, description="Type de recherche")
    common_filters: Optional[AdvancedFilters] = Field(default=None, description="Filtres communs")
    
    @validator("queries")
    def validate_queries_unique(cls, v):
        """Valide que les requêtes sont uniques."""
        if len(v) != len(set(v)):
            raise ValueError("Queries must be unique")
        return v


class ExplainRequest(BaseModel):
    """Requête pour expliquer le scoring d'un résultat."""
    query: str = Field(..., min_length=1, description="Requête originale")
    user_id: int = Field(..., description="ID de l'utilisateur")
    transaction_id: int = Field(..., description="ID de la transaction à expliquer")
    search_type: SearchType = Field(default=SearchType.HYBRID, description="Type de recherche utilisé")


class HealthCheckRequest(BaseModel):
    """Requête pour vérifier la santé du service."""
    include_detailed: bool = Field(default=False, description="Inclure diagnostics détaillés")
    check_connectivity: bool = Field(default=True, description="Vérifier connectivité aux services")
    check_performance: bool = Field(default=False, description="Vérifier performances")
    timeout_seconds: int = Field(default=10, ge=1, le=30, description="Timeout pour les checks")