"""
Modèles de requêtes pour le service de recherche.
VERSION COMPLÈTE - Inclut SearchQuery et tous les modèles nécessaires
"""
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class SearchType(str, Enum):
    """Types de recherche disponibles."""
    HYBRID = "hybrid"
    LEXICAL = "lexical"
    SEMANTIC = "semantic"


class BaseRequest(BaseModel):
    """Classe de base pour toutes les requêtes."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        str_strip_whitespace=True
    )
    
    @field_validator('*', mode='before')
    @classmethod
    def prevent_dict_queries(cls, v, info):
        """Empêche l'utilisation d'objets dict pour les champs query."""
        if hasattr(info, 'field_name') and info.field_name == 'query' and isinstance(v, dict):
            raise ValueError("Query parameter cannot be a dict object")
        return v


class SearchQuery(BaseModel):
    """Modèle de requête de recherche - ORIGINAL pour compatibilité."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        str_strip_whitespace=True
    )
    
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
    
    @field_validator('semantic_weight')
    @classmethod
    def weights_sum_to_one(cls, v, values):
        """Vérifie que les poids somment à 1."""
        if 'lexical_weight' in values and values['lexical_weight'] + v != 1.0:
            # Ajuster automatiquement le poids lexical
            values['lexical_weight'] = 1.0 - v
        return v


class SearchRequest(BaseRequest):
    """Modèle de requête de recherche - NOUVEAU système."""
    
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


class ReindexRequest(BaseRequest):
    """Modèle de requête pour la réindexation."""
    
    user_id: int = Field(..., description="ID de l'utilisateur")
    force_refresh: bool = Field(default=False, description="Forcer la réindexation complète")
    batch_size: int = Field(default=1000, ge=1, le=10000, description="Taille des lots")
    include_deleted: bool = Field(default=False, description="Inclure les transactions supprimées")


class BulkIndexRequest(BaseRequest):
    """Modèle de requête pour l'indexation en lot."""
    
    user_ids: List[int] = Field(..., description="Liste des IDs utilisateurs")
    batch_size: int = Field(default=1000, ge=1, le=10000, description="Taille des lots")
    parallel_workers: int = Field(default=2, ge=1, le=10, description="Nombre de workers parallèles")
    force_refresh: bool = Field(default=False, description="Forcer la réindexation complète")


class DeleteUserDataRequest(BaseRequest):
    """Modèle de requête pour la suppression des données utilisateur."""
    
    user_id: int = Field(..., description="ID de l'utilisateur")
    confirm_deletion: bool = Field(..., description="Confirmation de suppression")
    remove_from_search: bool = Field(default=True, description="Supprimer des index de recherche")
    remove_from_cache: bool = Field(default=True, description="Supprimer du cache")


class QueryExpansionRequest(BaseRequest):
    """Modèle de requête pour l'expansion de requête."""
    
    query: str = Field(..., min_length=1, description="Requête à étendre")
    max_terms: int = Field(default=10, ge=1, le=50, description="Nombre maximum de termes")
    include_synonyms: bool = Field(default=True, description="Inclure les synonymes")
    include_financial_terms: bool = Field(default=True, description="Inclure les termes financiers")


class UserStatsRequest(BaseRequest):
    """Modèle de requête pour les statistiques utilisateur."""
    
    user_id: int = Field(..., description="ID de l'utilisateur")
    date_from: Optional[datetime] = Field(default=None, description="Date de début")
    date_to: Optional[datetime] = Field(default=None, description="Date de fin")
    include_cache_stats: bool = Field(default=True, description="Inclure les stats de cache")


class DebugSearchRequest(BaseRequest):
    """Modèle de requête pour le debug de recherche."""
    
    user_id: int = Field(..., description="ID de l'utilisateur")
    query: str = Field(..., min_length=1, description="Texte de recherche")
    search_type: SearchType = Field(default=SearchType.HYBRID, description="Type de recherche")
    include_explanation: bool = Field(default=True, description="Inclure l'explication détaillée")
    include_timings: bool = Field(default=True, description="Inclure les temps de traitement")
    include_raw_results: bool = Field(default=False, description="Inclure les résultats bruts")


class IndexManagementRequest(BaseRequest):
    """Modèle de requête pour la gestion des index."""
    
    action: str = Field(..., description="Action à effectuer (create, delete, refresh, optimize)")
    index_type: str = Field(..., description="Type d'index (transactions, users, categories)")
    force: bool = Field(default=False, description="Forcer l'action même si risqué")
    backup_before: bool = Field(default=True, description="Faire une sauvegarde avant l'action")


# Export des modèles
__all__ = [
    # Enums
    'SearchType',
    
    # Modèles principaux (pour compatibilité)
    'SearchQuery',  # IMPORTANT : Modèle original pour compatibilité
    
    # Nouveaux modèles
    'SearchRequest',
    'ReindexRequest', 
    'BulkIndexRequest',
    'DeleteUserDataRequest',
    'QueryExpansionRequest',
    'UserStatsRequest',
    'DebugSearchRequest',
    'IndexManagementRequest',
    'BaseRequest'
]