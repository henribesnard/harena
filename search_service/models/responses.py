"""
Modèles de réponses sortantes pour le service de recherche.

Ce module définit les modèles Pydantic pour toutes les réponses
renvoyées par l'API de recherche.
"""
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from pydantic import BaseModel, Field
from search_service.models.search_types import SearchType, SearchQuality, SortOrder


class SearchResultItem(BaseModel):
    """Un résultat individuel de recherche."""
    # Identifiants
    transaction_id: int = Field(..., description="ID de la transaction")
    user_id: int = Field(..., description="ID de l'utilisateur")
    account_id: Optional[int] = Field(default=None, description="ID du compte")
    
    # Scores et pertinence
    score: float = Field(..., description="Score de pertinence (0-1)")
    lexical_score: Optional[float] = Field(default=None, description="Score lexical")
    semantic_score: Optional[float] = Field(default=None, description="Score sémantique")
    combined_score: Optional[float] = Field(default=None, description="Score combiné final")
    
    # Contenu de la transaction
    primary_description: str = Field(..., description="Description principale")
    searchable_text: Optional[str] = Field(default=None, description="Texte enrichi searchable")
    merchant_name: Optional[str] = Field(default=None, description="Nom du marchand")
    
    # Données financières
    amount: float = Field(..., description="Montant de la transaction")
    currency_code: str = Field(default="EUR", description="Code devise")
    transaction_type: str = Field(..., description="Type de transaction (debit/credit)")
    
    # Dates
    transaction_date: str = Field(..., description="Date de transaction (YYYY-MM-DD)")
    created_at: Optional[str] = Field(default=None, description="Date de création")
    
    # Catégorisation
    category_id: Optional[int] = Field(default=None, description="ID catégorie")
    operation_type: Optional[str] = Field(default=None, description="Type d'opération")
    
    # Highlighting et métadonnées
    highlights: Optional[Dict[str, List[str]]] = Field(default=None, description="Texte mis en évidence")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Métadonnées additionnelles")
    
    # Informations de debug (optionnel)
    explanation: Optional[Dict[str, Any]] = Field(default=None, description="Explication du scoring")


class SearchResponse(BaseModel):
    """Réponse complète d'une recherche."""
    # Métadonnées de la requête
    query: str = Field(..., description="Requête originale")
    search_type: SearchType = Field(..., description="Type de recherche utilisé")
    user_id: int = Field(..., description="ID de l'utilisateur")
    
    # Résultats
    results: List[SearchResultItem] = Field(..., description="Liste des résultats")
    total_found: int = Field(..., description="Nombre total de résultats trouvés")
    returned_count: int = Field(..., description="Nombre de résultats retournés")
    
    # Pagination
    offset: int = Field(default=0, description="Décalage appliqué")
    limit: int = Field(..., description="Limite appliquée")
    has_more: bool = Field(..., description="Plus de résultats disponibles")
    
    # Performance et qualité
    processing_time_ms: float = Field(..., description="Temps de traitement en ms")
    search_quality: SearchQuality = Field(..., description="Qualité estimée de la recherche")
    
    # Détails par moteur de recherche
    lexical_results_count: Optional[int] = Field(default=None, description="Résultats lexicaux trouvés")
    semantic_results_count: Optional[int] = Field(default=None, description="Résultats sémantiques trouvés")
    cache_hit: bool = Field(default=False, description="Résultat servi depuis le cache")
    
    # Suggestions et corrections
    suggestions: Optional[List[str]] = Field(default=None, description="Suggestions de requêtes alternatives")
    corrected_query: Optional[str] = Field(default=None, description="Requête corrigée automatiquement")
    
    # Métadonnées
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Timestamp de la réponse")
    
    @property
    def average_score(self) -> float:
        """Calcule le score moyen des résultats."""
        if not self.results:
            return 0.0
        return sum(result.score for result in self.results) / len(self.results)


class SuggestionItem(BaseModel):
    """Une suggestion d'auto-complétion."""
    text: str = Field(..., description="Texte suggéré")
    type: str = Field(..., description="Type de suggestion (merchant, category, description, recent)")
    frequency: Optional[int] = Field(default=None, description="Fréquence d'utilisation")
    category: Optional[str] = Field(default=None, description="Catégorie de la suggestion")


class SuggestionsResponse(BaseModel):
    """Réponse d'auto-complétion."""
    partial_query: str = Field(..., description="Requête partielle originale")
    suggestions: List[SuggestionItem] = Field(..., description="Liste des suggestions")
    processing_time_ms: float = Field(..., description="Temps de traitement en ms")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class SearchStats(BaseModel):
    """Statistiques de recherche utilisateur."""
    user_id: int = Field(..., description="ID de l'utilisateur")
    period_days: int = Field(..., description="Période analysée en jours")
    
    # Statistiques générales
    total_searches: int = Field(..., description="Nombre total de recherches")
    unique_queries: int = Field(..., description="Nombre de requêtes uniques")
    average_results_per_search: float = Field(..., description="Moyenne de résultats par recherche")
    
    # Répartition par type de recherche
    lexical_searches: int = Field(default=0, description="Recherches lexicales")
    semantic_searches: int = Field(default=0, description="Recherches sémantiques")
    hybrid_searches: int = Field(default=0, description="Recherches hybrides")
    
    # Performance
    average_response_time_ms: float = Field(..., description="Temps de réponse moyen")
    cache_hit_rate: float = Field(..., description="Taux de succès du cache (0-1)")
    
    # Requêtes populaires
    top_queries: List[Dict[str, Union[str, int]]] = Field(default=[], description="Requêtes les plus fréquentes")
    recent_queries: List[str] = Field(default=[], description="Requêtes récentes")
    
    # Qualité des résultats
    quality_distribution: Dict[str, int] = Field(default={}, description="Distribution de la qualité")


class StatsResponse(BaseModel):
    """Réponse des statistiques utilisateur."""
    stats: SearchStats = Field(..., description="Statistiques calculées")
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class ServiceHealth(BaseModel):
    """Statut de santé d'un service."""
    service_name: str = Field(..., description="Nom du service")
    status: str = Field(..., description="Statut (healthy, degraded, unhealthy)")
    response_time_ms: Optional[float] = Field(default=None, description="Temps de réponse")
    version: Optional[str] = Field(default=None, description="Version du service")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Détails additionnels")
    last_check: str = Field(default_factory=lambda: datetime.now().isoformat())


class HealthResponse(BaseModel):
    """Réponse de vérification de santé."""
    overall_status: str = Field(..., description="Statut global (healthy, degraded, unhealthy)")
    services: List[ServiceHealth] = Field(..., description="Statut des services individuels")
    
    # Métriques globales
    total_services: int = Field(..., description="Nombre total de services")
    healthy_services: int = Field(..., description="Services en bonne santé")
    degraded_services: int = Field(..., description="Services dégradés")
    unhealthy_services: int = Field(..., description="Services non fonctionnels")
    
    # Performance
    search_engine_ready: bool = Field(..., description="Moteur de recherche prêt")
    cache_operational: bool = Field(..., description="Cache opérationnel")
    
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class BulkSearchResponse(BaseModel):
    """Réponse pour la recherche en lot."""
    results: List[SearchResponse] = Field(..., description="Résultats pour chaque requête")
    total_queries: int = Field(..., description="Nombre total de requêtes")
    successful_queries: int = Field(..., description="Requêtes réussies")
    failed_queries: int = Field(..., description="Requêtes échouées")
    total_processing_time_ms: float = Field(..., description="Temps total de traitement")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ExplanationDetail(BaseModel):
    """Détail d'explication du scoring."""
    component: str = Field(..., description="Composant de scoring (lexical, semantic, boost, etc.)")
    score: float = Field(..., description="Score de ce composant")
    weight: float = Field(..., description="Poids appliqué")
    final_contribution: float = Field(..., description="Contribution finale au score")
    explanation: str = Field(..., description="Explication textuelle")


class ScoringExplanation(BaseModel):
    """Explication détaillée du scoring d'un résultat."""
    transaction_id: int = Field(..., description="ID de la transaction")
    final_score: float = Field(..., description="Score final")
    components: List[ExplanationDetail] = Field(..., description="Détails par composant")
    query_analysis: Dict[str, Any] = Field(..., description="Analyse de la requête")
    matching_terms: List[str] = Field(..., description="Termes correspondants")
    boosting_factors: Dict[str, float] = Field(default={}, description="Facteurs de boost appliqués")


class ExplainResponse(BaseModel):
    """Réponse d'explication du scoring."""
    query: str = Field(..., description="Requête originale")
    explanation: ScoringExplanation = Field(..., description="Explication détaillée")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ErrorResponse(BaseModel):
    """Réponse d'erreur standardisée."""
    error: str = Field(..., description="Type d'erreur")
    message: str = Field(..., description="Message d'erreur")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Détails additionnels")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    request_id: Optional[str] = Field(default=None, description="ID de la requête pour traçabilité")


class MetricsResponse(BaseModel):
    """Réponse des métriques du service."""
    # Métriques de performance
    total_searches_24h: int = Field(..., description="Recherches dans les 24h")
    average_response_time_ms: float = Field(..., description="Temps de réponse moyen")
    cache_hit_rate: float = Field(..., description="Taux de succès du cache")
    error_rate: float = Field(..., description="Taux d'erreur")
    
    # Métriques par type de recherche
    search_type_distribution: Dict[str, int] = Field(..., description="Répartition par type")
    quality_distribution: Dict[str, int] = Field(..., description="Répartition par qualité")
    
    # Métriques système
    memory_usage_mb: Optional[float] = Field(default=None, description="Utilisation mémoire")
    cpu_usage_percent: Optional[float] = Field(default=None, description="Utilisation CPU")
    
    # Services externes
    elasticsearch_health: str = Field(..., description="Santé Elasticsearch")
    qdrant_health: str = Field(..., description="Santé Qdrant")
    openai_health: str = Field(..., description="Santé OpenAI")
    
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())