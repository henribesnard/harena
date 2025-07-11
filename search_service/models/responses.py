"""
📤 Modèles Réponses Search Service - API Endpoints
=================================================

Modèles Pydantic pour les réponses API du Search Service. Ces modèles définissent
la structure standardisée des réponses pour tous les endpoints FastAPI.

Responsabilités:
- Modèles réponses endpoints API
- Sérialisation données sortie
- Documentation API automatique
- Cohérence format réponses
- Métriques et métadonnées
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum

from .service_contracts import SearchServiceResponse, TransactionResult


# =============================================================================
# 🎯 ÉNUMÉRATIONS RÉPONSES
# =============================================================================

class ResponseStatus(str, Enum):
    """Statuts de réponse."""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    PARTIAL = "partial"

class QueryComplexity(str, Enum):
    """Niveaux complexité requête."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


# =============================================================================
# 🏗️ RÉPONSES BASE
# =============================================================================

class BaseResponse(BaseModel):
    """Réponse de base pour tous les endpoints."""
    status: ResponseStatus = Field(..., description="Statut réponse")
    message: str = Field(..., description="Message descriptif")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp réponse")
    execution_time_ms: int = Field(..., ge=0, description="Temps exécution (ms)")
    request_id: Optional[str] = Field(None, description="ID requête pour traçage")
    
    class Config:
        """Configuration Pydantic."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ErrorResponse(BaseModel):
    """Réponse d'erreur standardisée."""
    status: ResponseStatus = Field(default=ResponseStatus.ERROR, description="Statut erreur")
    error_code: str = Field(..., description="Code erreur")
    error_message: str = Field(..., description="Message erreur")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Détails erreur")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp erreur")
    request_id: Optional[str] = Field(None, description="ID requête")
    
    class Config:
        """Configuration Pydantic."""
        schema_extra = {
            "example": {
                "status": "error",
                "error_code": "VALIDATION_ERROR",
                "error_message": "Invalid user_id parameter",
                "error_details": {
                    "field": "user_id",
                    "value": -1,
                    "constraint": "must be positive"
                },
                "request_id": "req_12345"
            }
        }


# =============================================================================
# 🔍 RÉPONSES RECHERCHE
# =============================================================================

class SearchResultSummary(BaseModel):
    """Résumé résultats recherche."""
    total_found: int = Field(..., ge=0, description="Total résultats trouvés")
    returned: int = Field(..., ge=0, description="Nombre retourné")
    offset: int = Field(..., ge=0, description="Offset pagination")
    has_more: bool = Field(..., description="Plus de résultats disponibles")
    max_score: Optional[float] = Field(None, description="Score maximum")
    avg_score: Optional[float] = Field(None, description="Score moyen")

class SearchPerformanceMetrics(BaseModel):
    """Métriques performance recherche."""
    query_time_ms: int = Field(..., ge=0, description="Temps requête (ms)")
    elasticsearch_time_ms: int = Field(..., ge=0, description="Temps Elasticsearch (ms)")
    processing_time_ms: int = Field(..., ge=0, description="Temps traitement (ms)")
    cache_hit: bool = Field(..., description="Hit cache")
    query_complexity: QueryComplexity = Field(..., description="Complexité requête")
    shards_queried: int = Field(..., ge=1, description="Shards interrogés")
    optimizations_applied: List[str] = Field(default_factory=list, description="Optimisations appliquées")

class SimpleLexicalSearchResponse(BaseResponse):
    """Réponse recherche lexicale simple."""
    results: List[TransactionResult] = Field(..., description="Résultats transactions")
    summary: SearchResultSummary = Field(..., description="Résumé résultats")
    performance: SearchPerformanceMetrics = Field(..., description="Métriques performance")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions recherche")
    
    @validator('results')
    def validate_results_consistency(cls, v, values):
        """Validation cohérence résultats."""
        summary = values.get('summary')
        if summary and len(v) != summary.returned:
            raise ValueError("Results count must match summary.returned")
        return v

class CategorySearchResponse(BaseResponse):
    """Réponse recherche par catégorie."""
    results: List[TransactionResult] = Field(..., description="Transactions de la catégorie")
    summary: SearchResultSummary = Field(..., description="Résumé résultats")
    performance: SearchPerformanceMetrics = Field(..., description="Métriques performance")
    category_stats: Dict[str, Any] = Field(default_factory=dict, description="Statistiques catégorie")
    related_categories: List[str] = Field(default_factory=list, description="Catégories liées")

class MerchantSearchResponse(BaseResponse):
    """Réponse recherche par marchand."""
    results: List[TransactionResult] = Field(..., description="Transactions du marchand")
    summary: SearchResultSummary = Field(..., description="Résumé résultats")
    performance: SearchPerformanceMetrics = Field(..., description="Métriques performance")
    merchant_info: Dict[str, Any] = Field(default_factory=dict, description="Info marchand")
    similar_merchants: List[str] = Field(default_factory=list, description="Marchands similaires")


# =============================================================================
# 📊 RÉPONSES ANALYSE ET AGRÉGATIONS
# =============================================================================

class CategoryStats(BaseModel):
    """Statistiques catégorie."""
    category_name: str = Field(..., description="Nom catégorie")
    transaction_count: int = Field(..., ge=0, description="Nombre transactions")
    total_amount: float = Field(..., description="Montant total")
    avg_amount: float = Field(..., description="Montant moyen")
    min_amount: float = Field(..., description="Montant minimum")
    max_amount: float = Field(..., description="Montant maximum")
    percentage_of_total: float = Field(..., ge=0, le=100, description="Pourcentage du total")

class MerchantStats(BaseModel):
    """Statistiques marchand."""
    merchant_name: str = Field(..., description="Nom marchand")
    transaction_count: int = Field(..., ge=0, description="Nombre transactions")
    total_amount: float = Field(..., description="Montant total")
    avg_amount: float = Field(..., description="Montant moyen")
    first_transaction: Optional[str] = Field(None, description="Première transaction")
    last_transaction: Optional[str] = Field(None, description="Dernière transaction")
    frequency_days: Optional[float] = Field(None, description="Fréquence moyenne (jours)")

class TemporalDataPoint(BaseModel):
    """Point de données temporel."""
    period: str = Field(..., description="Période (YYYY-MM, YYYY-MM-DD, etc.)")
    transaction_count: int = Field(..., ge=0, description="Nombre transactions")
    total_amount: float = Field(..., description="Montant total")
    avg_amount: float = Field(..., description="Montant moyen")
    categories: Dict[str, float] = Field(default_factory=dict, description="Répartition catégories")

class CategoryAnalysisResponse(BaseResponse):
    """Réponse analyse par catégories."""
    top_categories: List[CategoryStats] = Field(..., description="Top catégories")
    total_categories: int = Field(..., ge=0, description="Nombre total catégories")
    analysis_period: Dict[str, str] = Field(..., description="Période analyse")
    summary_stats: Dict[str, float] = Field(..., description="Statistiques résumé")
    performance: SearchPerformanceMetrics = Field(..., description="Métriques performance")

class MerchantAnalysisResponse(BaseResponse):
    """Réponse analyse par marchands."""
    top_merchants: List[MerchantStats] = Field(..., description="Top marchands")
    total_merchants: int = Field(..., ge=0, description="Nombre total marchands")
    analysis_period: Dict[str, str] = Field(..., description="Période analyse")
    category_breakdown: Dict[str, List[MerchantStats]] = Field(default_factory=dict, description="Répartition par catégorie")
    performance: SearchPerformanceMetrics = Field(..., description="Métriques performance")

class TemporalAnalysisResponse(BaseResponse):
    """Réponse analyse temporelle."""
    time_series: List[TemporalDataPoint] = Field(..., description="Série temporelle")
    trends: Dict[str, Any] = Field(..., description="Analyse tendances")
    seasonality: Dict[str, Any] = Field(default_factory=dict, description="Analyse saisonnalité")
    forecasts: List[Dict[str, Any]] = Field(default_factory=list, description="Prévisions")
    granularity: str = Field(..., description="Granularité données")
    performance: SearchPerformanceMetrics = Field(..., description="Métriques performance")


# =============================================================================
# 🔧 RÉPONSES UTILITAIRES
# =============================================================================

class ValidationResult(BaseModel):
    """Résultat validation."""
    valid: bool = Field(..., description="Validation réussie")
    errors: List[str] = Field(default_factory=list, description="Erreurs validation")
    warnings: List[str] = Field(default_factory=list, description="Avertissements")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions amélioration")

class QueryValidationResponse(BaseResponse):
    """Réponse validation requête."""
    validation: ValidationResult = Field(..., description="Résultat validation")
    optimized_query: Optional[Dict[str, Any]] = Field(None, description="Requête optimisée")
    estimated_performance: Optional[Dict[str, Any]] = Field(None, description="Performance estimée")
    explain_plan: Optional[Dict[str, Any]] = Field(None, description="Plan exécution")

class HealthStatus(BaseModel):
    """Statut santé composant."""
    component: str = Field(..., description="Nom composant")
    status: str = Field(..., description="Statut (healthy/unhealthy/degraded)")
    response_time_ms: Optional[int] = Field(None, description="Temps réponse (ms)")
    details: Dict[str, Any] = Field(default_factory=dict, description="Détails statut")
    last_check: datetime = Field(default_factory=datetime.utcnow, description="Dernière vérification")

class HealthCheckResponse(BaseResponse):
    """Réponse health check."""
    overall_status: str = Field(..., description="Statut global")
    components: List[HealthStatus] = Field(..., description="Statuts composants")
    uptime_seconds: int = Field(..., ge=0, description="Uptime en secondes")
    version: str = Field(..., description="Version service")
    
    class Config:
        """Configuration Pydantic."""
        schema_extra = {
            "example": {
                "status": "success",
                "message": "Service healthy",
                "overall_status": "healthy",
                "components": [
                    {
                        "component": "elasticsearch",
                        "status": "healthy",
                        "response_time_ms": 15,
                        "details": {"cluster_status": "green"}
                    },
                    {
                        "component": "redis",
                        "status": "healthy", 
                        "response_time_ms": 2,
                        "details": {"memory_usage": "45%"}
                    }
                ],
                "uptime_seconds": 86400,
                "version": "1.0.0"
            }
        }

class MetricValue(BaseModel):
    """Valeur métrique."""
    name: str = Field(..., description="Nom métrique")
    value: Union[int, float, str] = Field(..., description="Valeur métrique")
    unit: Optional[str] = Field(None, description="Unité mesure")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp mesure")

class MetricsResponse(BaseResponse):
    """Réponse métriques système."""
    metrics: List[MetricValue] = Field(..., description="Métriques collectées")
    time_range: str = Field(..., description="Période métriques")
    summary: Dict[str, Any] = Field(..., description="Résumé métriques")
    alerts: List[Dict[str, Any]] = Field(default_factory=list, description="Alertes actives")


# =============================================================================
# 🤝 RÉPONSES CONTRATS
# =============================================================================

class ContractSearchResponse(BaseResponse):
    """Réponse recherche par contrat."""
    contract_response: SearchServiceResponse = Field(..., description="Réponse contrat standardisé")
    validation_results: Optional[ValidationResult] = Field(None, description="Résultats validation contrat")
    contract_version: str = Field(..., description="Version contrat utilisée")
    debug_info: Optional[Dict[str, Any]] = Field(None, description="Informations debug")
    
    class Config:
        """Configuration Pydantic."""
        schema_extra = {
            "example": {
                "status": "success",
                "message": "Contract search completed successfully",
                "execution_time_ms": 45,
                "contract_response": {
                    "response_metadata": {
                        "query_id": "uuid-v4",
                        "execution_time_ms": 45,
                        "total_hits": 156,
                        "returned_hits": 20,
                        "has_more": True,
                        "cache_hit": False,
                        "elasticsearch_took": 23
                    },
                    "results": [],
                    "performance": {
                        "query_complexity": "simple",
                        "optimization_applied": ["user_filter", "category_filter"],
                        "index_used": "harena_transactions",
                        "shards_queried": 1
                    },
                    "context_enrichment": {
                        "search_intent_matched": True,
                        "result_quality_score": 0.95,
                        "suggested_followup_questions": []
                    }
                },
                "contract_version": "v1.0"
            }
        }


# =============================================================================
# 📋 RÉPONSES BATCH ET BULK
# =============================================================================

class BatchSearchResult(BaseModel):
    """Résultat recherche batch individuelle."""
    query_index: int = Field(..., ge=0, description="Index requête dans batch")
    success: bool = Field(..., description="Succès exécution")
    response: Optional[Union[SimpleLexicalSearchResponse, Dict[str, Any]]] = Field(None, description="Réponse si succès")
    error: Optional[ErrorResponse] = Field(None, description="Erreur si échec")
    execution_time_ms: int = Field(..., ge=0, description="Temps exécution individuel")

class BatchSearchResponse(BaseResponse):
    """Réponse recherche batch."""
    results: List[BatchSearchResult] = Field(..., description="Résultats individuels")
    total_queries: int = Field(..., ge=0, description="Nombre total requêtes")
    successful_queries: int = Field(..., ge=0, description="Requêtes réussies")
    failed_queries: int = Field(..., ge=0, description="Requêtes échouées")
    parallel_execution: bool = Field(..., description="Exécution parallèle utilisée")
    total_execution_time_ms: int = Field(..., ge=0, description="Temps total exécution")

class BulkValidationResult(BaseModel):
    """Résultat validation bulk individuelle."""
    query_index: int = Field(..., ge=0, description="Index requête")
    validation: ValidationResult = Field(..., description="Résultat validation")
    contract_valid: bool = Field(..., description="Contrat valide")

class BulkValidationResponse(BaseResponse):
    """Réponse validation bulk."""
    results: List[BulkValidationResult] = Field(..., description="Résultats validation individuels")
    total_queries: int = Field(..., ge=0, description="Nombre total requêtes")
    valid_queries: int = Field(..., ge=0, description="Requêtes valides")
    invalid_queries: int = Field(..., ge=0, description="Requêtes invalides")
    summary: ValidationResult = Field(..., description="Résumé validation global")


# =============================================================================
# 🎯 RÉPONSES SPÉCIALISÉES FINANCIÈRES
# =============================================================================

class RecurringTransaction(BaseModel):
    """Transaction récurrente détectée."""
    pattern_id: str = Field(..., description="ID pattern récurrent")
    merchant_name: str = Field(..., description="Nom marchand")
    category_name: str = Field(..., description="Catégorie")
    avg_amount: float = Field(..., description="Montant moyen")
    frequency_days: int = Field(..., ge=1, description="Fréquence en jours")
    occurrences: int = Field(..., ge=2, description="Nombre occurrences")
    confidence_score: float = Field(..., ge=0, le=1, description="Score confiance")
    next_predicted_date: Optional[str] = Field(None, description="Prochaine date prédite")
    transactions: List[TransactionResult] = Field(..., description="Transactions du pattern")

class RecurringTransactionResponse(BaseResponse):
    """Réponse détection transactions récurrentes."""
    recurring_patterns: List[RecurringTransaction] = Field(..., description="Patterns récurrents détectés")
    total_patterns: int = Field(..., ge=0, description="Nombre total patterns")
    analysis_period_months: int = Field(..., ge=1, description="Période analyse (mois)")
    detection_settings: Dict[str, Any] = Field(..., description="Paramètres détection")
    performance: SearchPerformanceMetrics = Field(..., description="Métriques performance")

class SuspiciousActivity(BaseModel):
    """Activité suspecte détectée."""
    alert_id: str = Field(..., description="ID alerte")
    alert_type: str = Field(..., description="Type alerte")
    severity: str = Field(..., description="Sévérité (low/medium/high)")
    description: str = Field(..., description="Description activité")
    confidence_score: float = Field(..., ge=0, le=1, description="Score confiance")
    triggered_rules: List[str] = Field(..., description="Règles déclenchées")
    related_transactions: List[TransactionResult] = Field(..., description="Transactions liées")
    recommended_actions: List[str] = Field(default_factory=list, description="Actions recommandées")

class SuspiciousActivityResponse(BaseResponse):
    """Réponse détection activité suspecte."""
    suspicious_activities: List[SuspiciousActivity] = Field(..., description="Activités suspectes")
    total_alerts: int = Field(..., ge=0, description="Nombre total alertes")
    severity_breakdown: Dict[str, int] = Field(..., description="Répartition par sévérité")
    detection_settings: Dict[str, Any] = Field(..., description="Paramètres détection")
    analysis_period_days: int = Field(..., ge=1, description="Période analyse (jours)")
    performance: SearchPerformanceMetrics = Field(..., description="Métriques performance")

class BudgetCategory(BaseModel):
    """Catégorie budget."""
    category_name: str = Field(..., description="Nom catégorie")
    budgeted_amount: Optional[float] = Field(None, description="Montant budgété")
    actual_amount: float = Field(..., description="Montant réel")
    variance_amount: float = Field(..., description="Écart montant")
    variance_percentage: float = Field(..., description="Écart pourcentage")
    status: str = Field(..., description="Statut (under/over/on_target)")
    projection: Optional[float] = Field(None, description="Projection fin période")

class BudgetAnalysisResponse(BaseResponse):
    """Réponse analyse budget."""
    budget_categories: List[BudgetCategory] = Field(..., description="Catégories budget")
    total_budgeted: Optional[float] = Field(None, description="Total budgété")
    total_actual: float = Field(..., description="Total réel")
    overall_variance: float = Field(..., description="Écart global")
    budget_period: str = Field(..., description="Période budget")
    projections: Dict[str, float] = Field(default_factory=dict, description="Projections")
    recommendations: List[str] = Field(default_factory=list, description="Recommandations")
    performance: SearchPerformanceMetrics = Field(..., description="Métriques performance")


# =============================================================================
# 📈 RÉPONSES ANALYTICS AVANCÉES
# =============================================================================

class SpendingPattern(BaseModel):
    """Pattern de dépense."""
    pattern_type: str = Field(..., description="Type pattern")
    pattern_name: str = Field(..., description="Nom pattern")
    description: str = Field(..., description="Description pattern")
    strength: float = Field(..., ge=0, le=1, description="Force du pattern")
    frequency: str = Field(..., description="Fréquence")
    related_categories: List[str] = Field(..., description="Catégories liées")
    avg_amount: float = Field(..., description="Montant moyen")
    examples: List[TransactionResult] = Field(..., description="Exemples transactions")

class SpendingPatternResponse(BaseResponse):
    """Réponse analyse patterns de dépenses."""
    patterns: List[SpendingPattern] = Field(..., description="Patterns détectés")
    total_patterns: int = Field(..., ge=0, description="Nombre total patterns")
    pattern_strength_distribution: Dict[str, int] = Field(..., description="Distribution force patterns")
    analysis_settings: Dict[str, Any] = Field(..., description="Paramètres analyse")
    insights: List[str] = Field(default_factory=list, description="Insights générés")
    performance: SearchPerformanceMetrics = Field(..., description="Métriques performance")

class TrendAnalysis(BaseModel):
    """Analyse de tendance."""
    metric: str = Field(..., description="Métrique analysée")
    trend_direction: str = Field(..., description="Direction tendance (up/down/stable)")
    trend_strength: float = Field(..., ge=0, le=1, description="Force tendance")
    change_percentage: float = Field(..., description="Pourcentage changement")
    change_amount: float = Field(..., description="Montant changement")
    time_period: str = Field(..., description="Période analyse")
    confidence: float = Field(..., ge=0, le=1, description="Confiance tendance")

class TrendAnalysisResponse(BaseResponse):
    """Réponse analyse tendances."""
    trends: List[TrendAnalysis] = Field(..., description="Tendances détectées")
    overall_trend: str = Field(..., description="Tendance globale")
    significant_changes: List[Dict[str, Any]] = Field(..., description="Changements significatifs")
    forecasts: List[Dict[str, Any]] = Field(default_factory=list, description="Prévisions")
    recommendations: List[str] = Field(default_factory=list, description="Recommandations")
    performance: SearchPerformanceMetrics = Field(..., description="Métriques performance")


# =============================================================================
# 🛠️ FACTORY RÉPONSES
# =============================================================================

class ResponseFactory:
    """Factory pour créer réponses standardisées."""
    
    @staticmethod
    def create_success_response(
        message: str, 
        execution_time_ms: int, 
        data: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ) -> BaseResponse:
        """Créer réponse succès."""
        response_data = {
            "status": ResponseStatus.SUCCESS,
            "message": message,
            "execution_time_ms": execution_time_ms,
        }
        if request_id:
            response_data["request_id"] = request_id
        if data:
            response_data.update(data)
        
        return BaseResponse(**response_data)
    
    @staticmethod
    def create_error_response(
        error_code: str,
        error_message: str,
        error_details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ) -> ErrorResponse:
        """Créer réponse erreur."""
        return ErrorResponse(
            error_code=error_code,
            error_message=error_message,
            error_details=error_details,
            request_id=request_id
        )
    
    @staticmethod
    def create_validation_response(
        valid: bool,
        errors: List[str] = None,
        warnings: List[str] = None,
        execution_time_ms: int = 0
    ) -> QueryValidationResponse:
        """Créer réponse validation."""
        validation_result = ValidationResult(
            valid=valid,
            errors=errors or [],
            warnings=warnings or []
        )
        
        return QueryValidationResponse(
            status=ResponseStatus.SUCCESS if valid else ResponseStatus.ERROR,
            message="Validation completed",
            execution_time_ms=execution_time_ms,
            validation=validation_result
        )
    
    @staticmethod
    def create_search_response(
        results: List[TransactionResult],
        total_found: int,
        execution_time_ms: int,
        elasticsearch_time_ms: int,
        cache_hit: bool = False,
        query_complexity: QueryComplexity = QueryComplexity.SIMPLE
    ) -> SimpleLexicalSearchResponse:
        """Créer réponse recherche."""
        summary = SearchResultSummary(
            total_found=total_found,
            returned=len(results),
            offset=0,
            has_more=len(results) < total_found,
            max_score=max([r.score for r in results]) if results else None,
            avg_score=sum([r.score for r in results]) / len(results) if results else None
        )
        
        performance = SearchPerformanceMetrics(
            query_time_ms=execution_time_ms,
            elasticsearch_time_ms=elasticsearch_time_ms,
            processing_time_ms=execution_time_ms - elasticsearch_time_ms,
            cache_hit=cache_hit,
            query_complexity=query_complexity,
            shards_queried=1
        )
        
        return SimpleLexicalSearchResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Found {total_found} results",
            execution_time_ms=execution_time_ms,
            results=results,
            summary=summary,
            performance=performance
        )
    
    @staticmethod
    def create_health_response(
        overall_status: str,
        components: List[HealthStatus],
        uptime_seconds: int,
        version: str = "1.0.0"
    ) -> HealthCheckResponse:
        """Créer réponse health check."""
        return HealthCheckResponse(
            status=ResponseStatus.SUCCESS,
            message="Health check completed",
            execution_time_ms=10,
            overall_status=overall_status,
            components=components,
            uptime_seconds=uptime_seconds,
            version=version
        )
    
    @staticmethod
    def create_contract_response(
        contract_response: SearchServiceResponse,
        validation_results: Optional[ValidationResult] = None,
        contract_version: str = "v1.0",
        debug_info: Optional[Dict[str, Any]] = None
    ) -> ContractSearchResponse:
        """Créer réponse contrat."""
        return ContractSearchResponse(
            status=ResponseStatus.SUCCESS,
            message="Contract search completed successfully",
            execution_time_ms=contract_response.response_metadata.execution_time_ms,
            contract_response=contract_response,
            validation_results=validation_results,
            contract_version=contract_version,
            debug_info=debug_info
        )


# =============================================================================
# 📋 EXPORTS
# =============================================================================

__all__ = [
    # Énumérations
    "ResponseStatus", "QueryComplexity",
    # Réponses de base
    "BaseResponse", "ErrorResponse",
    # Modèles support
    "SearchResultSummary", "SearchPerformanceMetrics", "ValidationResult", "HealthStatus", "MetricValue",
    # Réponses recherche
    "SimpleLexicalSearchResponse", "CategorySearchResponse", "MerchantSearchResponse",
    # Réponses analyse
    "CategoryStats", "MerchantStats", "TemporalDataPoint", 
    "CategoryAnalysisResponse", "MerchantAnalysisResponse", "TemporalAnalysisResponse",
    # Réponses utilitaires
    "QueryValidationResponse", "HealthCheckResponse", "MetricsResponse",
    # Réponses contrats
    "ContractSearchResponse",
    # Réponses batch
    "BatchSearchResult", "BatchSearchResponse", "BulkValidationResult", "BulkValidationResponse",
    # Réponses spécialisées
    "RecurringTransaction", "RecurringTransactionResponse",
    "SuspiciousActivity", "SuspiciousActivityResponse",
    "BudgetCategory", "BudgetAnalysisResponse",
    # Réponses analytics
    "SpendingPattern", "SpendingPatternResponse", "TrendAnalysis", "TrendAnalysisResponse",
    # Factory
    "ResponseFactory",
]