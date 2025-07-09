"""
Modèles de réponse pour l'API du Search Service.

Ces modèles définissent les structures de données pour toutes les réponses
de l'API REST du Search Service, avec enrichissement contextuel et
métriques de performance.

ARCHITECTURE:
- LexicalSearchResponse: Réponse de recherche lexicale principale
- Enrichissement contextuel automatique
- Métriques de performance intégrées
- Support des agrégations et suggestions
- Gestion des erreurs standardisée

CONFIGURATION CENTRALISÉE:
- Formatage via config_service
- Métriques configurables
- Enrichissement adaptatif
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Literal
from uuid import UUID
from enum import Enum

from pydantic import BaseModel, Field, validator
from pydantic.types import PositiveInt, NonNegativeInt, NonNegativeFloat

# Configuration centralisée
from config_service.config import settings

# Import des contrats pour cohérence
from .service_contracts import (
    SearchResult, AggregationMetrics, PerformanceMetrics,
    ResponseMetadata, ContextEnrichment
)

# ==================== ENUMS ET STATUTS ====================

class ResponseStatus(str, Enum):
    """Statuts de réponse."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    ERROR = "error"
    TIMEOUT = "timeout"
    EMPTY = "empty"

class QualityLevel(str, Enum):
    """Niveaux de qualité des résultats."""
    EXCELLENT = "excellent"
    GOOD = "good"
    MEDIUM = "medium"
    POOR = "poor"

class EnrichmentType(str, Enum):
    """Types d'enrichissement disponibles."""
    SUGGESTIONS = "suggestions"
    RELATED_SEARCHES = "related_searches"
    CATEGORY_INSIGHTS = "category_insights"
    SPENDING_PATTERNS = "spending_patterns"
    TEMPORAL_ANALYSIS = "temporal_analysis"

# ==================== MÉTRIQUES ET QUALITÉ ====================

class QualityMetrics(BaseModel):
    """Métriques de qualité des résultats."""
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Score global de qualité")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Score de pertinence")
    coverage_score: float = Field(..., ge=0.0, le=1.0, description="Score de couverture")
    diversity_score: float = Field(..., ge=0.0, le=1.0, description="Score de diversité")
    
    quality_level: QualityLevel = Field(..., description="Niveau de qualité global")
    confidence_interval: tuple[float, float] = Field(..., description="Intervalle de confiance")
    
    # Détails par dimension
    precision: Optional[float] = Field(None, ge=0.0, le=1.0, description="Précision estimée")
    recall: Optional[float] = Field(None, ge=0.0, le=1.0, description="Rappel estimé")
    f1_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Score F1")
    
    # Facteurs de qualité
    result_density: float = Field(..., ge=0.0, description="Densité des résultats")
    query_complexity_match: float = Field(..., ge=0.0, le=1.0, description="Adéquation complexité")
    temporal_relevance: float = Field(..., ge=0.0, le=1.0, description="Pertinence temporelle")
    
    class Config:
        use_enum_values = True

class ResponseEnrichment(BaseModel):
    """Enrichissement avancé des réponses."""
    # Suggestions intelligentes
    suggested_queries: List[str] = Field(default=[], description="Requêtes suggérées")
    suggested_filters: List[Dict[str, Any]] = Field(default=[], description="Filtres suggérés")
    suggested_refinements: List[str] = Field(default=[], description="Raffinements suggérés")
    
    # Insights contextuels
    category_insights: Dict[str, Any] = Field(default={}, description="Insights par catégorie")
    spending_insights: Dict[str, Any] = Field(default={}, description="Insights de dépenses")
    temporal_insights: Dict[str, Any] = Field(default={}, description="Insights temporels")
    
    # Patterns détectés
    detected_patterns: List[str] = Field(default=[], description="Patterns détectés")
    anomalies: List[Dict[str, Any]] = Field(default=[], description="Anomalies détectées")
    trends: List[Dict[str, Any]] = Field(default=[], description="Tendances identifiées")
    
    # Recommandations
    recommendations: List[str] = Field(default=[], description="Recommandations")
    next_actions: List[str] = Field(default=[], description="Actions suivantes suggérées")
    
    # Métadonnées d'enrichissement
    enrichment_types: List[EnrichmentType] = Field(default=[], description="Types d'enrichissement")
    enrichment_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confiance enrichissement")
    enrichment_time_ms: NonNegativeInt = Field(default=0, description="Temps d'enrichissement (ms)")

class ResultEnrichment(BaseModel):
    """Enrichissement individuel des résultats."""
    # Scores détaillés
    relevance_breakdown: Dict[str, float] = Field(default={}, description="Détail pertinence")
    quality_indicators: Dict[str, float] = Field(default={}, description="Indicateurs qualité")
    
    # Contexte additionnel
    related_transactions: List[str] = Field(default=[], description="Transactions liées")
    category_context: Dict[str, Any] = Field(default={}, description="Contexte catégorie")
    merchant_context: Dict[str, Any] = Field(default={}, description="Contexte marchand")
    
    # Explications
    why_relevant: List[str] = Field(default=[], description="Pourquoi pertinent")
    matching_factors: List[str] = Field(default=[], description="Facteurs de correspondance")
    
    # Enrichissement temporel
    recency_factor: float = Field(default=1.0, ge=0.0, description="Facteur de récence")
    seasonal_context: Optional[str] = Field(None, description="Contexte saisonnier")

# ==================== RÉPONSES DE BASE ====================

class BaseResponse(BaseModel):
    """Classe de base pour toutes les réponses."""
    status: ResponseStatus = Field(..., description="Statut de la réponse")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp de réponse")
    request_id: Optional[UUID] = Field(None, description="ID de la requête")
    execution_time_ms: NonNegativeInt = Field(..., description="Temps d'exécution (ms)")
    
    class Config:
        use_enum_values = True

class SuccessResponse(BaseResponse):
    """Réponse de succès générique."""
    status: Literal[ResponseStatus.SUCCESS] = ResponseStatus.SUCCESS
    message: str = Field(default="Opération réussie", description="Message de succès")
    data: Optional[Dict[str, Any]] = Field(None, description="Données de réponse")

class ErrorResponse(BaseResponse):
    """Réponse d'erreur standardisée."""
    status: Literal[ResponseStatus.ERROR] = ResponseStatus.ERROR
    error_code: str = Field(..., description="Code d'erreur")
    error_message: str = Field(..., description="Message d'erreur")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Détails de l'erreur")
    
    # Contexte de l'erreur
    error_category: str = Field(..., description="Catégorie d'erreur")
    is_retryable: bool = Field(default=False, description="Erreur récupérable")
    retry_after_seconds: Optional[int] = Field(None, description="Délai avant retry")
    
    # Aide et suggestions
    help_message: Optional[str] = Field(None, description="Message d'aide")
    suggested_actions: List[str] = Field(default=[], description="Actions suggérées")

# ==================== RÉPONSES SPÉCIALISÉES ====================

class LexicalSearchResponse(BaseResponse):
    """
    Réponse de recherche lexicale principale.
    
    Cette réponse est retournée par l'endpoint POST /search/lexical
    et contient tous les résultats avec enrichissement contextuel.
    """
    status: ResponseStatus = Field(..., description="Statut de la recherche")
    
    # Métadonnées de réponse
    response_metadata: ResponseMetadata = Field(..., description="Métadonnées de réponse")
    
    # Résultats principaux
    results: List[SearchResult] = Field(default=[], description="Résultats de recherche")
    
    # Agrégations et statistiques
    aggregations: Optional[AggregationMetrics] = Field(None, description="Résultats d'agrégation")
    
    # Performance et métriques
    performance: PerformanceMetrics = Field(..., description="Métriques de performance")
    quality: QualityMetrics = Field(..., description="Métriques de qualité")
    
    # Enrichissement contextuel
    enrichment: ResponseEnrichment = Field(
        default_factory=ResponseEnrichment, 
        description="Enrichissement contextuel"
    )
    
    # Pagination et navigation
    pagination: Dict[str, Any] = Field(default={}, description="Informations de pagination")
    
    # Debug et observabilité
    debug_info: Optional[Dict[str, Any]] = Field(None, description="Informations de debug")
    
    @validator('results')
    def validate_results_consistency(cls, v, values):
        """Valide la cohérence des résultats."""
        metadata = values.get('response_metadata')
        if metadata and len(v) != metadata.returned_hits:
            # Log l'incohérence mais ne fait pas échouer la validation
            # car cela peut arriver en cas de déduplication
            pass
        return v

class QueryValidationResponse(BaseResponse):
    """Réponse de validation de requête."""
    status: ResponseStatus = Field(..., description="Statut de validation")
    
    is_valid: bool = Field(..., description="Requête valide")
    validation_errors: List[str] = Field(default=[], description="Erreurs de validation")
    validation_warnings: List[str] = Field(default=[], description="Avertissements")
    
    # Analyse de la requête
    query_analysis: Dict[str, Any] = Field(default={}, description="Analyse de la requête")
    estimated_performance: Dict[str, Any] = Field(default={}, description="Performance estimée")
    
    # Suggestions d'optimisation
    optimization_suggestions: List[str] = Field(default=[], description="Suggestions d'optimisation")
    rewritten_query: Optional[Dict[str, Any]] = Field(None, description="Requête réécrite")

class TemplateListResponse(BaseResponse):
    """Réponse de liste des templates."""
    status: ResponseStatus = Field(..., description="Statut de la liste")
    
    templates: List[Dict[str, Any]] = Field(default=[], description="Templates disponibles")
    categories: List[str] = Field(default=[], description="Catégories de templates")
    
    # Métadonnées des templates
    total_templates: NonNegativeInt = Field(..., description="Nombre total de templates")
    template_metadata: Dict[str, Any] = Field(default={}, description="Métadonnées des templates")

class HealthCheckResponse(BaseResponse):
    """Réponse de vérification de santé."""
    status: ResponseStatus = Field(..., description="Statut de santé")
    
    # Statuts des composants
    elasticsearch_status: str = Field(..., description="Statut Elasticsearch")
    cache_status: str = Field(..., description="Statut du cache")
    
    # Métriques de santé
    health_metrics: Dict[str, Any] = Field(default={}, description="Métriques de santé")
    
    # Diagnostics détaillés
    component_details: Dict[str, Any] = Field(default={}, description="Détails des composants")
    performance_indicators: Dict[str, float] = Field(default={}, description="Indicateurs de performance")
    
    # Alertes et recommandations
    alerts: List[str] = Field(default=[], description="Alertes actives")
    recommendations: List[str] = Field(default=[], description="Recommandations")

class MetricsResponse(BaseResponse):
    """Réponse des métriques système."""
    status: ResponseStatus = Field(..., description="Statut des métriques")
    
    # Métriques par catégorie
    performance_metrics: Dict[str, Any] = Field(default={}, description="Métriques de performance")
    usage_metrics: Dict[str, Any] = Field(default={}, description="Métriques d'usage")
    error_metrics: Dict[str, Any] = Field(default={}, description="Métriques d'erreur")
    
    # Métriques temporelles
    time_series_data: List[Dict[str, Any]] = Field(default=[], description="Données temporelles")
    aggregated_metrics: Dict[str, Any] = Field(default={}, description="Métriques agrégées")
    
    # Métadonnées des métriques
    metrics_metadata: Dict[str, Any] = Field(default={}, description="Métadonnées des métriques")
    collection_period: str = Field(..., description="Période de collecte")

# ==================== RÉPONSES SPÉCIALISÉES AVANCÉES ====================

class BulkSearchResponse(BaseResponse):
    """Réponse de recherche en lot."""
    status: ResponseStatus = Field(..., description="Statut global du lot")
    
    # Résultats par recherche
    search_results: List[LexicalSearchResponse] = Field(default=[], description="Résultats par recherche")
    
    # Statistiques globales
    total_searches: PositiveInt = Field(..., description="Nombre total de recherches")
    successful_searches: NonNegativeInt = Field(..., description="Recherches réussies")
    failed_searches: NonNegativeInt = Field(..., description="Recherches échouées")
    
    # Performance globale
    total_execution_time_ms: NonNegativeInt = Field(..., description="Temps total d'exécution")
    average_execution_time_ms: float = Field(..., description="Temps moyen par recherche")
    
    # Erreurs et avertissements
    errors: List[ErrorResponse] = Field(default=[], description="Erreurs rencontrées")
    warnings: List[str] = Field(default=[], description="Avertissements")

class AggregationOnlyResponse(BaseResponse):
    """Réponse d'agrégation sans résultats détaillés."""
    status: ResponseStatus = Field(..., description="Statut de l'agrégation")
    
    # Métadonnées basiques
    query_id: UUID = Field(..., description="ID de la requête")
    total_documents: NonNegativeInt = Field(..., description="Nombre total de documents")
    
    # Agrégations principales
    aggregations: AggregationMetrics = Field(..., description="Résultats d'agrégation")
    
    # Performance spécialisée
    aggregation_performance: Dict[str, Any] = Field(default={}, description="Performance des agrégations")
    
    # Insights d'agrégation
    aggregation_insights: Dict[str, Any] = Field(default={}, description="Insights des agrégations")

class AutocompleteResponse(BaseResponse):
    """Réponse d'autocomplétion."""
    status: ResponseStatus = Field(..., description="Statut de l'autocomplétion")
    
    # Suggestions
    suggestions: List[str] = Field(default=[], description="Suggestions d'autocomplétion")
    
    # Métadonnées des suggestions
    suggestion_metadata: List[Dict[str, Any]] = Field(default=[], description="Métadonnées par suggestion")
    
    # Contexte d'autocomplétion
    completion_context: Dict[str, Any] = Field(default={}, description="Contexte de complétion")

# ==================== VALIDATION ET HELPERS ====================

class ResponseValidationError(Exception):
    """Exception pour les erreurs de validation de réponse."""
    pass

class ResponseValidator:
    """Validateur pour les réponses de recherche."""
    
    @staticmethod
    def validate_search_response(response: LexicalSearchResponse) -> bool:
        """
        Valide une réponse de recherche lexicale.
        
        Args:
            response: La réponse à valider
            
        Returns:
            True si valide
            
        Raises:
            ResponseValidationError: Si la validation échoue
        """
        try:
            # Validation de base Pydantic
            response.dict()
            
            # Validations métier spécifiques
            if response.response_metadata.execution_time_ms < 0:
                raise ResponseValidationError("execution_time_ms ne peut pas être négatif")
            
            if response.response_metadata.returned_hits > response.response_metadata.total_hits:
                raise ResponseValidationError("returned_hits > total_hits impossible")
            
            # Validation cohérence qualité
            if response.quality.overall_score < 0 or response.quality.overall_score > 1:
                raise ResponseValidationError("overall_score doit être entre 0 et 1")
            
            # Validation résultats
            for result in response.results:
                if result.amount_abs < 0:
                    raise ResponseValidationError("amount_abs ne peut pas être négatif")
            
            return True
            
        except Exception as e:
            raise ResponseValidationError(f"Validation échouée: {str(e)}")

# ==================== FACTORY FUNCTIONS ====================

def create_search_response(
    request_id: UUID,
    results: List[Dict[str, Any]],
    total_hits: int,
    execution_time_ms: int,
    **kwargs
) -> LexicalSearchResponse:
    """
    Factory pour créer une LexicalSearchResponse.
    
    Args:
        request_id: ID de la requête
        results: Résultats de recherche
        total_hits: Nombre total de résultats
        execution_time_ms: Temps d'exécution
        **kwargs: Paramètres additionnels
        
    Returns:
        LexicalSearchResponse configurée
    """
    # Conversion des résultats
    search_results = [SearchResult(**result) for result in results]
    
    # Métadonnées de réponse
    response_metadata = ResponseMetadata(
        query_id=request_id,
        execution_time_ms=execution_time_ms,
        total_hits=total_hits,
        returned_hits=len(search_results),
        has_more=total_hits > len(search_results),
        elasticsearch_took=kwargs.get('elasticsearch_took', execution_time_ms // 2)
    )
    
    # Métriques de performance
    performance = PerformanceMetrics(
        query_complexity=kwargs.get('query_complexity', 'simple'),
        optimization_applied=kwargs.get('optimization_applied', []),
        index_used=kwargs.get('index_used', settings.ELASTICSEARCH_INDEX),
        elasticsearch_took=kwargs.get('elasticsearch_took', execution_time_ms // 2)
    )
    
    # Calcul automatique de la qualité
    quality = calculate_quality_metrics(search_results, total_hits, execution_time_ms)
    
    return LexicalSearchResponse(
        status=ResponseStatus.SUCCESS,
        request_id=request_id,
        execution_time_ms=execution_time_ms,
        response_metadata=response_metadata,
        results=search_results,
        performance=performance,
        quality=quality,
        aggregations=kwargs.get('aggregations'),
        enrichment=kwargs.get('enrichment', ResponseEnrichment()),
        debug_info=kwargs.get('debug_info')
    )

def create_error_response(
    request_id: Optional[UUID],
    error_code: str,
    error_message: str,
    execution_time_ms: int,
    **kwargs
) -> ErrorResponse:
    """
    Factory pour créer une ErrorResponse.
    
    Args:
        request_id: ID de la requête
        error_code: Code d'erreur
        error_message: Message d'erreur
        execution_time_ms: Temps d'exécution
        **kwargs: Paramètres additionnels
        
    Returns:
        ErrorResponse configurée
    """
    return ErrorResponse(
        request_id=request_id,
        execution_time_ms=execution_time_ms,
        error_code=error_code,
        error_message=error_message,
        error_details=kwargs.get('error_details'),
        error_category=kwargs.get('error_category', 'unknown'),
        is_retryable=kwargs.get('is_retryable', False),
        retry_after_seconds=kwargs.get('retry_after_seconds'),
        help_message=kwargs.get('help_message'),
        suggested_actions=kwargs.get('suggested_actions', [])
    )

# ==================== UTILITAIRES DE QUALITÉ ====================

def calculate_quality_metrics(
    results: List[SearchResult],
    total_hits: int,
    execution_time_ms: int
) -> QualityMetrics:
    """
    Calcule automatiquement les métriques de qualité.
    
    Args:
        results: Résultats de recherche
        total_hits: Nombre total de résultats
        execution_time_ms: Temps d'exécution
        
    Returns:
        QualityMetrics calculées
    """
    if not results:
        return QualityMetrics(
            overall_score=0.0,
            relevance_score=0.0,
            coverage_score=0.0,
            diversity_score=0.0,
            quality_level=QualityLevel.POOR,
            confidence_interval=(0.0, 0.0),
            result_density=0.0,
            query_complexity_match=0.0,
            temporal_relevance=0.0
        )
    
    # Calcul des scores
    relevance_score = calculate_relevance_score(results)
    coverage_score = calculate_coverage_score(len(results), total_hits)
    diversity_score = calculate_diversity_score(results)
    
    # Score global pondéré
    overall_score = (
        relevance_score * 0.4 +
        coverage_score * 0.3 +
        diversity_score * 0.3
    )
    
    # Niveau de qualité
    if overall_score >= settings.QUALITY_EXCELLENT_THRESHOLD:
        quality_level = QualityLevel.EXCELLENT
    elif overall_score >= settings.QUALITY_GOOD_THRESHOLD:
        quality_level = QualityLevel.GOOD
    elif overall_score >= settings.QUALITY_MEDIUM_THRESHOLD:
        quality_level = QualityLevel.MEDIUM
    else:
        quality_level = QualityLevel.POOR
    
    # Métriques additionnelles
    result_density = len(results) / max(total_hits, 1)
    query_complexity_match = 1.0 if execution_time_ms < 1000 else 0.8
    temporal_relevance = calculate_temporal_relevance(results)
    
    return QualityMetrics(
        overall_score=overall_score,
        relevance_score=relevance_score,
        coverage_score=coverage_score,
        diversity_score=diversity_score,
        quality_level=quality_level,
        confidence_interval=(max(0.0, overall_score - 0.1), min(1.0, overall_score + 0.1)),
        result_density=result_density,
        query_complexity_match=query_complexity_match,
        temporal_relevance=temporal_relevance
    )

def calculate_relevance_score(results: List[SearchResult]) -> float:
    """Calcule le score de pertinence moyen."""
    if not results:
        return 0.0
    
    scores = [result.score for result in results]
    avg_score = sum(scores) / len(scores)
    
    # Normalisation et ajustement
    normalized_score = min(avg_score, 1.0)
    
    # Pénalité si trop peu de résultats avec score élevé
    high_score_ratio = sum(1 for score in scores if score > 0.8) / len(scores)
    adjustment = high_score_ratio * 0.2
    
    return min(normalized_score + adjustment, 1.0)

def calculate_coverage_score(returned: int, total: int) -> float:
    """Calcule le score de couverture."""
    if total == 0:
        return 1.0
    
    ratio = returned / total
    
    # Score optimal pour 10-50 résultats
    if 10 <= returned <= 50:
        return min(ratio * 1.2, 1.0)
    elif returned < 10:
        return ratio * 0.8
    else:
        return ratio

def calculate_diversity_score(results: List[SearchResult]) -> float:
    """Calcule le score de diversité."""
    if len(results) < 2:
        return 1.0 if results else 0.0
    
    # Diversité par catégorie
    categories = set(result.category_name for result in results)
    category_diversity = len(categories) / len(results)
    
    # Diversité par marchand
    merchants = set(result.merchant_name for result in results if result.merchant_name)
    merchant_diversity = len(merchants) / len(results) if merchants else 0.5
    
    # Diversité temporelle
    dates = set(result.date for result in results)
    date_diversity = min(len(dates) / len(results), 1.0)
    
    # Score global de diversité
    return (category_diversity * 0.4 + merchant_diversity * 0.4 + date_diversity * 0.2)

def calculate_temporal_relevance(results: List[SearchResult]) -> float:
    """Calcule la pertinence temporelle."""
    if not results:
        return 0.0
    
    from datetime import datetime, timedelta
    
    now = datetime.now()
    total_score = 0.0
    
    for result in results:
        try:
            result_date = datetime.strptime(result.date, "%Y-%m-%d")
            days_ago = (now - result_date).days
            
            # Score décroissant avec l'âge
            if days_ago <= 7:
                score = 1.0
            elif days_ago <= 30:
                score = 0.8
            elif days_ago <= 90:
                score = 0.6
            elif days_ago <= 365:
                score = 0.4
            else:
                score = 0.2
            
            total_score += score
        except:
            total_score += 0.5  # Score par défaut si date invalide
    
    return total_score / len(results)

# ==================== ENRICHISSEMENT AUTOMATIQUE ====================

def enrich_response_automatically(
    response: LexicalSearchResponse,
    original_query: Optional[str] = None
) -> LexicalSearchResponse:
    """
    Enrichit automatiquement une réponse avec du contexte intelligent.
    
    Args:
        response: Réponse à enrichir
        original_query: Requête originale de l'utilisateur
        
    Returns:
        Réponse enrichie
    """
    if not response.results:
        return response
    
    enrichment = ResponseEnrichment()
    
    # Génération de suggestions basées sur les résultats
    enrichment.suggested_queries = generate_suggested_queries(response.results, original_query)
    enrichment.suggested_refinements = generate_suggested_refinements(response.results)
    
    # Analyse des patterns
    enrichment.detected_patterns = detect_spending_patterns(response.results)
    enrichment.category_insights = analyze_category_insights(response.results)
    enrichment.spending_insights = analyze_spending_insights(response.results)
    
    # Recommandations intelligentes
    enrichment.recommendations = generate_recommendations(response.results, response.quality)
    enrichment.next_actions = generate_next_actions(response.results, original_query)
    
    # Métadonnées d'enrichissement
    enrichment.enrichment_types = [
        EnrichmentType.SUGGESTIONS,
        EnrichmentType.CATEGORY_INSIGHTS,
        EnrichmentType.SPENDING_PATTERNS
    ]
    enrichment.enrichment_confidence = 0.8
    
    response.enrichment = enrichment
    return response

def generate_suggested_queries(results: List[SearchResult], original_query: Optional[str]) -> List[str]:
    """Génère des requêtes suggérées basées sur les résultats."""
    suggestions = []
    
    if not results:
        return suggestions
    
    # Suggestions basées sur les catégories trouvées
    categories = list(set(result.category_name for result in results))
    for category in categories[:3]:
        suggestions.append(f"Toutes mes dépenses {category.lower()}")
    
    # Suggestions basées sur les marchands
    merchants = list(set(result.merchant_name for result in results if result.merchant_name))
    for merchant in merchants[:2]:
        suggestions.append(f"Mes achats chez {merchant}")
    
    # Suggestions temporelles
    suggestions.extend([
        "Évolution de mes dépenses ce mois",
        "Comparaison avec le mois dernier",
        "Mes plus grosses dépenses"
    ])
    
    return suggestions[:5]

def generate_suggested_refinements(results: List[SearchResult]) -> List[str]:
    """Génère des raffinements suggérés."""
    refinements = []
    
    if not results:
        return refinements
    
    # Analyse des montants pour suggérer des plages
    amounts = [abs(result.amount) for result in results]
    if amounts:
        avg_amount = sum(amounts) / len(amounts)
        refinements.append(f"Montants supérieurs à {avg_amount:.0f}€")
        refinements.append(f"Montants inférieurs à {avg_amount:.0f}€")
    
    # Suggestions temporelles
    refinements.extend([
        "Cette semaine seulement",
        "Derniers 30 jours",
        "Transactions récentes"
    ])
    
    return refinements[:4]

def detect_spending_patterns(results: List[SearchResult]) -> List[str]:
    """Détecte des patterns de dépenses."""
    patterns = []
    
    if len(results) < 3:
        return patterns
    
    # Analyse des catégories dominantes
    category_counts = {}
    for result in results:
        category_counts[result.category_name] = category_counts.get(result.category_name, 0) + 1
    
    dominant_category = max(category_counts, key=category_counts.get)
    if category_counts[dominant_category] / len(results) > 0.5:
        patterns.append(f"Concentration sur {dominant_category.lower()}")
    
    # Analyse des montants
    amounts = [abs(result.amount) for result in results]
    avg_amount = sum(amounts) / len(amounts)
    
    high_amounts = [a for a in amounts if a > avg_amount * 1.5]
    if len(high_amounts) / len(amounts) > 0.3:
        patterns.append("Plusieurs dépenses importantes")
    
    # Pattern temporel
    from collections import Counter
    weekdays = [result.weekday for result in results]
    weekday_counts = Counter(weekdays)
    most_common_day = weekday_counts.most_common(1)[0]
    if most_common_day[1] / len(results) > 0.4:
        patterns.append(f"Activité concentrée le {most_common_day[0]}")
    
    return patterns

def analyze_category_insights(results: List[SearchResult]) -> Dict[str, Any]:
    """Analyse les insights par catégorie."""
    insights = {}
    
    category_data = {}
    for result in results:
        cat = result.category_name
        if cat not in category_data:
            category_data[cat] = {"count": 0, "total": 0.0, "amounts": []}
        
        category_data[cat]["count"] += 1
        category_data[cat]["total"] += abs(result.amount)
        category_data[cat]["amounts"].append(abs(result.amount))
    
    for category, data in category_data.items():
        insights[category] = {
            "transaction_count": data["count"],
            "total_amount": data["total"],
            "average_amount": data["total"] / data["count"],
            "percentage_of_total": (data["count"] / len(results)) * 100
        }
    
    return insights

def analyze_spending_insights(results: List[SearchResult]) -> Dict[str, Any]:
    """Analyse les insights de dépenses."""
    if not results:
        return {}
    
    amounts = [abs(result.amount) for result in results]
    total = sum(amounts)
    
    insights = {
        "total_amount": total,
        "average_transaction": total / len(amounts),
        "highest_transaction": max(amounts),
        "lowest_transaction": min(amounts),
        "transaction_count": len(results)
    }
    
    # Distribution des montants
    high_amounts = [a for a in amounts if a > 100]
    medium_amounts = [a for a in amounts if 20 <= a <= 100]
    low_amounts = [a for a in amounts if a < 20]
    
    insights["distribution"] = {
        "high_amounts": {"count": len(high_amounts), "total": sum(high_amounts)},
        "medium_amounts": {"count": len(medium_amounts), "total": sum(medium_amounts)},
        "low_amounts": {"count": len(low_amounts), "total": sum(low_amounts)}
    }
    
    return insights

def generate_recommendations(results: List[SearchResult], quality: QualityMetrics) -> List[str]:
    """Génère des recommandations intelligentes."""
    recommendations = []
    
    # Recommandations basées sur la qualité
    if quality.quality_level == QualityLevel.POOR:
        recommendations.append("Essayez des termes de recherche plus spécifiques")
        recommendations.append("Utilisez des filtres pour affiner votre recherche")
    elif quality.quality_level == QualityLevel.EXCELLENT:
        recommendations.append("Résultats très pertinents trouvés")
    
    # Recommandations basées sur les résultats
    if len(results) > 50:
        recommendations.append("Beaucoup de résultats - considérez ajouter des filtres")
    elif len(results) < 5:
        recommendations.append("Peu de résultats - essayez d'élargir votre recherche")
    
    # Recommandations contextuelles
    if results:
        categories = set(result.category_name for result in results)
        if len(categories) == 1:
            recommendations.append("Explorez d'autres catégories similaires")
    
    return recommendations[:3]

def generate_next_actions(results: List[SearchResult], original_query: Optional[str]) -> List[str]:
    """Génère des actions suivantes suggérées."""
    actions = []
    
    if not results:
        actions.extend([
            "Modifier les critères de recherche",
            "Essayer une recherche plus large",
            "Vérifier les filtres appliqués"
        ])
        return actions
    
    # Actions basées sur le contexte
    actions.extend([
        "Voir plus de détails sur ces transactions",
        "Analyser les tendances de dépenses",
        "Comparer avec les périodes précédentes",
        "Exporter ces résultats"
    ])
    
    # Actions spécifiques selon les résultats
    if len(results) >= 10:
        actions.append("Créer un rapport détaillé")
    
    categories = set(result.category_name for result in results)
    if len(categories) > 1:
        actions.append("Analyser par catégorie")
    
    return actions[:4]

# ==================== UTILITAIRES DE FORMATAGE ====================

def format_response_for_agent(response: LexicalSearchResponse, agent_context: Optional[str] = None) -> Dict[str, Any]:
    """
    Formate une réponse pour un agent AutoGen spécifique.
    
    Args:
        response: Réponse à formater
        agent_context: Contexte de l'agent destinataire
        
    Returns:
        Réponse formatée pour l'agent
    """
    formatted = {
        "status": response.status,
        "results_count": len(response.results),
        "total_hits": response.response_metadata.total_hits,
        "execution_time": response.execution_time_ms,
        "quality_score": response.quality.overall_score,
        "has_more": response.response_metadata.has_more
    }
    
    # Ajout conditionnel selon le contexte de l'agent
    if agent_context == "response_generator":
        formatted["enrichment"] = response.enrichment.dict()
        formatted["suggestions"] = response.enrichment.suggested_queries
        formatted["insights"] = response.enrichment.category_insights
    
    elif agent_context == "query_optimizer":
        formatted["performance"] = response.performance.dict()
        formatted["optimization_opportunities"] = response.enrichment.suggested_refinements
    
    elif agent_context == "analytics":
        formatted["aggregations"] = response.aggregations.dict() if response.aggregations else {}
        formatted["patterns"] = response.enrichment.detected_patterns
    
    # Toujours inclure les résultats principaux
    formatted["results"] = [result.dict() for result in response.results]
    
    return formatted

# ==================== CONSTANTES ET EXPORTS ====================

# Seuils de qualité par défaut
DEFAULT_QUALITY_THRESHOLDS = {
    QualityLevel.EXCELLENT: 0.9,
    QualityLevel.GOOD: 0.7,
    QualityLevel.MEDIUM: 0.5,
    QualityLevel.POOR: 0.0
}

# Codes d'erreur standardisés
ERROR_CODES = {
    "VALIDATION_ERROR": "VALIDATION_ERROR",
    "ELASTICSEARCH_ERROR": "ELASTICSEARCH_ERROR",
    "TIMEOUT_ERROR": "TIMEOUT_ERROR",
    "PERMISSION_ERROR": "PERMISSION_ERROR",
    "RATE_LIMIT_ERROR": "RATE_LIMIT_ERROR",
    "INTERNAL_ERROR": "INTERNAL_ERROR"
}

# Types d'enrichissement par défaut
DEFAULT_ENRICHMENT_TYPES = [
    EnrichmentType.SUGGESTIONS,
    EnrichmentType.CATEGORY_INSIGHTS,
    EnrichmentType.SPENDING_PATTERNS
]

__all__ = [
    # Réponses principales
    "LexicalSearchResponse",
    "QueryValidationResponse",
    "TemplateListResponse",
    "HealthCheckResponse",
    "MetricsResponse",
    
    # Réponses spécialisées
    "BulkSearchResponse",
    "AggregationOnlyResponse",
    "AutocompleteResponse",
    
    # Réponses de base
    "SuccessResponse",
    "ErrorResponse",
    
    # Métriques et qualité
    "QualityMetrics",
    "ResponseEnrichment",
    "ResultEnrichment",
    
    # Enums
    "ResponseStatus",
    "QualityLevel",
    "EnrichmentType",
    
    # Validation
    "ResponseValidationError",
    "ResponseValidator",
    
    # Factory functions
    "create_search_response",
    "create_error_response",
    
    # Utilitaires de qualité
    "calculate_quality_metrics",
    "calculate_relevance_score",
    "calculate_coverage_score",
    "calculate_diversity_score",
    
    # Enrichissement
    "enrich_response_automatically",
    "generate_suggested_queries",
    "detect_spending_patterns",
    "analyze_category_insights",
    
    # Formatage
    "format_response_for_agent",
    
    # Constantes
    "DEFAULT_QUALITY_THRESHOLDS",
    "ERROR_CODES",
    "DEFAULT_ENRICHMENT_TYPES"
]