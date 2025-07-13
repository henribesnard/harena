"""
Modèles de réponses internes et API du Search Service
====================================================

Structures optimisées pour le traitement interne avant conversion vers contrats
et modèles de réponses API spécialisés pour les endpoints REST.

Contient :
- Modèles internes (InternalSearchResponse, RawTransaction...)
- Modèles API REST (ValidationResponse, HealthResponse...)
- Transformateurs (ResponseTransformer)
- Builders (ResponseBuilder)
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, validator
from dataclasses import dataclass, field

from .service_contracts import SearchServiceResponse, AggregationType


# === ENUMS INTERNES ===

class ExecutionStatus(str, Enum):
    """Statuts d'exécution des requêtes"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    TIMEOUT = "timeout"
    ERROR = "error"
    CACHED = "cached"


class OptimizationType(str, Enum):
    """Types d'optimisations appliquées"""
    CACHE_HIT = "cache_hit"
    INDEX_OPTIMIZATION = "index_optimization"
    QUERY_REWRITE = "query_rewrite"
    FIELD_FILTERING = "field_filtering"
    AGGREGATION_CACHE = "aggregation_cache"
    RESULT_DEDUPLICATION = "result_deduplication"
    BOOST_OPTIMIZATION = "boost_optimization"


class QualityIndicator(str, Enum):
    """Indicateurs de qualité des résultats"""
    EXCELLENT = "excellent"    # Score > 0.9, résultats très pertinents
    GOOD = "good"             # Score > 0.7, résultats pertinents
    AVERAGE = "average"       # Score > 0.5, résultats moyens
    POOR = "poor"             # Score > 0.3, résultats faibles
    VERY_POOR = "very_poor"   # Score ≤ 0.3, résultats peu pertinents


class ComponentStatus(str, Enum):
    """Statuts des composants pour health check"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ValidationSeverity(str, Enum):
    """Niveaux de sévérité pour validation"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# === STRUCTURES DE DONNÉES INTERNES ===

@dataclass
class RawTransaction:
    """Transaction brute d'Elasticsearch"""
    source: Dict[str, Any]
    score: float
    index: str
    id: str
    highlights: Optional[Dict[str, List[str]]] = None
    explanation: Optional[Dict[str, Any]] = None
    
    def to_standardized_result(self) -> Dict[str, Any]:
        """Convertit vers format standardisé"""
        return {
            "transaction_id": self.source.get("transaction_id", self.id),
            "user_id": self.source.get("user_id"),
            "account_id": self.source.get("account_id"),
            "amount": self.source.get("amount", 0.0),
            "amount_abs": self.source.get("amount_abs", abs(self.source.get("amount", 0.0))),
            "transaction_type": self.source.get("transaction_type", "unknown"),
            "currency_code": self.source.get("currency_code", "EUR"),
            "date": self.source.get("date", ""),
            "primary_description": self.source.get("primary_description", ""),
            "merchant_name": self.source.get("merchant_name"),
            "category_name": self.source.get("category_name"),
            "operation_type": self.source.get("operation_type", ""),
            "month_year": self.source.get("month_year", ""),
            "weekday": self.source.get("weekday"),
            "score": self.score,
            "highlights": self.highlights
        }


@dataclass
class AggregationBucketInternal:
    """Bucket d'agrégation interne optimisé"""
    key: Union[str, int, float]
    doc_count: int
    metrics: Dict[str, float] = field(default_factory=dict)
    sub_buckets: List['AggregationBucketInternal'] = field(default_factory=list)
    
    def get_metric(self, metric_name: str, default: float = 0.0) -> float:
        """Récupère une métrique avec valeur par défaut"""
        return self.metrics.get(metric_name, default)
    
    def add_metric(self, name: str, value: float):
        """Ajoute une métrique"""
        self.metrics[name] = value
    
    def to_external_bucket(self) -> Dict[str, Any]:
        """Convertit vers format externe"""
        return {
            "key": self.key,
            "doc_count": self.doc_count,
            "total_amount": self.get_metric("sum_amount"),
            "avg_amount": self.get_metric("avg_amount"),
            "min_amount": self.get_metric("min_amount"),
            "max_amount": self.get_metric("max_amount")
        }


@dataclass 
class InternalAggregationResult:
    """Résultat d'agrégation interne"""
    name: str
    aggregation_type: AggregationType
    buckets: List[AggregationBucketInternal] = field(default_factory=list)
    total_count: int = 0
    total_amount: float = 0.0
    avg_amount: float = 0.0
    min_amount: Optional[float] = None
    max_amount: Optional[float] = None
    percentiles: Dict[str, float] = field(default_factory=dict)
    
    def add_bucket(self, bucket: AggregationBucketInternal):
        """Ajoute un bucket"""
        self.buckets.append(bucket)
    
    def calculate_totals(self):
        """Calcule les totaux à partir des buckets"""
        if not self.buckets:
            return
        
        self.total_count = sum(b.doc_count for b in self.buckets)
        amounts = [b.get_metric("sum_amount") for b in self.buckets if b.get_metric("sum_amount") > 0]
        
        if amounts:
            self.total_amount = sum(amounts)
            self.avg_amount = self.total_amount / len(amounts)
            self.min_amount = min(amounts)
            self.max_amount = max(amounts)
    
    def get_top_buckets(self, limit: int = 10) -> List[AggregationBucketInternal]:
        """Retourne les top buckets par doc_count"""
        return sorted(self.buckets, key=lambda b: b.doc_count, reverse=True)[:limit]


@dataclass
class ExecutionMetrics:
    """Métriques d'exécution internes"""
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    elasticsearch_took: int = 0
    total_execution_time: int = 0
    cache_lookup_time: int = 0
    query_build_time: int = 0
    result_processing_time: int = 0
    optimizations_applied: List[OptimizationType] = field(default_factory=list)
    
    def mark_completed(self):
        """Marque l'exécution comme terminée"""
        self.end_time = datetime.now(timezone.utc)
        self.total_execution_time = int((self.end_time - self.start_time).total_seconds() * 1000)
    
    def add_optimization(self, optimization: OptimizationType):
        """Ajoute une optimisation appliquée"""
        if optimization not in self.optimizations_applied:
            self.optimizations_applied.append(optimization)
    
    def get_breakdown(self) -> Dict[str, int]:
        """Retourne la répartition des temps"""
        return {
            "elasticsearch_took": self.elasticsearch_took,
            "cache_lookup_time": self.cache_lookup_time,
            "query_build_time": self.query_build_time,
            "result_processing_time": self.result_processing_time,
            "other_time": max(0, self.total_execution_time - (
                self.elasticsearch_took + self.cache_lookup_time + 
                self.query_build_time + self.result_processing_time
            ))
        }


# === MODÈLES INTERNES ===

class InternalSearchResponse(BaseModel):
    """Réponse de recherche interne complète"""
    
    # Identification et statut
    request_id: str = Field(..., description="ID de la requête")
    user_id: int = Field(..., description="ID utilisateur")
    status: ExecutionStatus = Field(..., description="Statut d'exécution")
    
    # Résultats
    raw_results: List[RawTransaction] = Field(default_factory=list, description="Résultats bruts ES")
    total_hits: int = Field(default=0, description="Total hits Elasticsearch")
    max_score: Optional[float] = Field(default=None, description="Score maximum")
    
    # Agrégations
    aggregations: List[InternalAggregationResult] = Field(default_factory=list, description="Résultats agrégations")
    
    # Métriques de performance
    execution_metrics: ExecutionMetrics = Field(default_factory=ExecutionMetrics, description="Métriques d'exécution")
    
    # Qualité et contexte
    quality_score: float = Field(default=0.0, description="Score qualité global")
    quality_indicator: QualityIndicator = Field(default=QualityIndicator.AVERAGE, description="Indicateur qualité")
    
    # Cache et optimisations
    served_from_cache: bool = Field(default=False, description="Servi depuis cache")
    cache_key: Optional[str] = Field(default=None, description="Clé de cache utilisée")
    
    # Elasticsearch raw (optionnel)
    elasticsearch_response: Optional[Dict[str, Any]] = Field(default=None, description="Réponse ES brute")
    
    # Contexte pour enrichissement
    suggested_followups: List[str] = Field(default_factory=list, description="Questions suggérées")
    related_categories: List[str] = Field(default_factory=list, description="Catégories liées")
    
    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True


# === MODÈLES API REST ===

class ValidationError(BaseModel):
    """Erreur de validation avec détails"""
    field: str = Field(..., description="Champ concerné")
    error_type: str = Field(..., description="Type d'erreur")
    message: str = Field(..., description="Message d'erreur")
    severity: ValidationSeverity = Field(default=ValidationSeverity.ERROR, description="Sévérité")
    suggested_fix: Optional[str] = Field(default=None, description="Correction suggérée")


class SecurityCheckResult(BaseModel):
    """Résultat de vérification sécurité"""
    passed: bool = Field(..., description="Vérification réussie")
    user_id_check: bool = Field(..., description="Vérification user_id")
    data_isolation_check: bool = Field(..., description="Vérification isolation données")
    permissions_check: bool = Field(..., description="Vérification permissions")
    warnings: List[str] = Field(default_factory=list, description="Avertissements sécurité")


class PerformanceAnalysis(BaseModel):
    """Analyse de performance d'une requête"""
    complexity: str = Field(..., description="Complexité estimée")
    estimated_time_ms: int = Field(..., description="Temps estimé en ms")
    warnings: List[str] = Field(default_factory=list, description="Avertissements performance")
    optimization_suggestions: List[str] = Field(default_factory=list, description="Suggestions d'optimisation")
    cache_eligible: bool = Field(default=True, description="Éligible au cache")
    field_count: int = Field(default=0, description="Nombre de champs recherchés")
    filter_count: int = Field(default=0, description="Nombre de filtres")


class ValidationResponse(BaseModel):
    """Réponse de validation d'une requête"""
    valid: bool = Field(..., description="Requête valide")
    errors: List[ValidationError] = Field(default_factory=list, description="Erreurs de validation")
    warnings: List[ValidationError] = Field(default_factory=list, description="Avertissements")
    security_check: SecurityCheckResult = Field(..., description="Résultat vérification sécurité")
    performance_analysis: PerformanceAnalysis = Field(..., description="Analyse de performance")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Métadonnées validation")


class TemplateInfo(BaseModel):
    """Informations sur un template de requête"""
    name: str = Field(..., description="Nom du template")
    category: str = Field(..., description="Catégorie")
    intent_type: str = Field(..., description="Type d'intention")
    description: str = Field(..., description="Description")
    complexity: str = Field(..., description="Complexité")
    usage_count: int = Field(default=0, description="Nombre d'utilisations")
    avg_execution_time_ms: float = Field(default=0.0, description="Temps d'exécution moyen")
    cache_hit_rate: float = Field(default=0.0, description="Taux de cache hit")
    last_used: Optional[datetime] = Field(default=None, description="Dernière utilisation")


class TemplateListResponse(BaseModel):
    """Réponse listant les templates disponibles"""
    templates: Dict[str, TemplateInfo] = Field(..., description="Templates disponibles")
    total_count: int = Field(..., description="Nombre total de templates")
    categories: List[str] = Field(..., description="Catégories disponibles")
    intent_types: List[str] = Field(..., description="Types d'intention disponibles")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Métadonnées de réponse")


# === MODÈLES CORRECTS POUR HEALTH CHECK ===

class ComponentHealthInfo(BaseModel):
    """Informations de santé d'un composant - Structure compatible avec routes.py"""
    name: str = Field(..., description="Nom du composant")
    status: str = Field(..., description="Statut du composant (healthy/degraded/unhealthy)")
    last_check: datetime = Field(..., description="Dernière vérification")
    response_time_ms: Optional[float] = Field(default=None, description="Temps de réponse en ms")
    error_message: Optional[str] = Field(default=None, description="Message d'erreur si unhealthy")
    dependencies: List[str] = Field(default_factory=list, description="Dépendances")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Métriques spécifiques")

    class Config:
        use_enum_values = True


class SystemHealth(BaseModel):
    """Santé globale du système - Structure compatible avec routes.py"""
    overall_status: str = Field(..., description="Statut global (healthy/degraded/unhealthy)")
    uptime_seconds: float = Field(..., description="Uptime en secondes")
    memory_usage_mb: float = Field(..., description="Usage mémoire en MB")
    cpu_usage_percent: float = Field(..., description="Usage CPU en %")
    active_connections: int = Field(default=0, description="Connexions actives")
    total_requests: int = Field(default=0, description="Total requêtes")
    error_rate_percent: float = Field(default=0.0, description="Taux d'erreur %")

    class Config:
        use_enum_values = True


class HealthResponse(BaseModel):
    """Réponse détaillée de santé du service - Structure compatible avec routes.py"""
    system: SystemHealth = Field(..., description="Santé système globale")
    components: List[ComponentHealthInfo] = Field(..., description="Santé des composants")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp")
    service_version: str = Field(default="1.0.0", description="Version du service")
    environment: str = Field(default="production", description="Environnement")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Métadonnées")

    class Config:
        use_enum_values = True


class MetricsResponse(BaseModel):
    """Réponse d'export des métriques"""
    format: str = Field(..., description="Format des métriques")
    content: str = Field(..., description="Contenu des métriques")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Date génération")
    metrics_count: int = Field(..., description="Nombre de métriques")
    time_range_hours: int = Field(default=1, description="Période couverte en heures")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Métadonnées")


# === TRANSFORMATEUR DE RÉPONSES ===

class ResponseTransformer:
    """Transformateur de réponses internes vers contrats externes"""
    
    @staticmethod
    def to_service_contract(internal_response: InternalSearchResponse, 
                          original_query_id: str) -> SearchServiceResponse:
        """Convertit une réponse interne vers le contrat externe"""
        
        # Finaliser les calculs
        internal_response.calculate_quality_score()
        internal_response.generate_followup_suggestions()
        internal_response.mark_completed()
        
        # Construire les métadonnées de réponse
        response_metadata = {
            "query_id": original_query_id,
            "execution_time_ms": internal_response.execution_metrics.total_execution_time,
            "total_hits": internal_response.total_hits,
            "returned_hits": len(internal_response.raw_results),
            "has_more": internal_response.total_hits > len(internal_response.raw_results),
            "cache_hit": internal_response.served_from_cache,
            "elasticsearch_took": internal_response.execution_metrics.elasticsearch_took,
            "agent_context": {
                "requesting_agent": "search_service",
                "next_suggested_agent": "response_generator_agent" if internal_response.raw_results else "query_optimizer_agent"
            },
            "timestamp": datetime.now(timezone.utc)
        }
        
        # Convertir les résultats
        standardized_results = internal_response.get_standardized_results()
        
        # Construire les agrégations externes
        external_aggregations = None
        if internal_response.aggregations:
            external_aggregations = ResponseTransformer._build_external_aggregations(
                internal_response.aggregations
            )
        
        # Métriques de performance
        performance = {
            "query_complexity": ResponseTransformer._determine_complexity(internal_response),
            "optimization_applied": [opt.value for opt in internal_response.execution_metrics.optimizations_applied],
            "index_used": "harena_transactions",  # TODO: récupérer depuis config
            "shards_queried": 1,  # TODO: récupérer depuis ES response
            "cache_hit": internal_response.served_from_cache
        }
        
        # Enrichissement contextuel
        context_enrichment = {
            "search_intent_matched": internal_response.quality_score > 0.5,
            "result_quality_score": internal_response.quality_score,
            "suggested_followup_questions": internal_response.suggested_followups,
            "next_suggested_agent": "response_generator_agent" if internal_response.raw_results else None
        }
        
        # Debug info (optionnel)
        debug_info = None
        if internal_response.elasticsearch_response:
            debug_info = {
                "elasticsearch_query": "available_on_request",
                "performance_breakdown": internal_response.execution_metrics.get_breakdown(),
                "quality_breakdown": {
                    "quality_indicator": internal_response.quality_indicator.value,
                    "max_score": internal_response.max_score,
                    "avg_score": sum(r.score for r in internal_response.raw_results) / len(internal_response.raw_results) if internal_response.raw_results else 0
                }
            }
        
        # Construire le contrat de réponse
        return SearchServiceResponse(
            response_metadata=response_metadata,
            results=standardized_results,
            aggregations=external_aggregations,
            performance=performance,
            context_enrichment=context_enrichment,
            debug=debug_info
        )
    
    @staticmethod
    def _build_external_aggregations(internal_aggs: List[InternalAggregationResult]) -> Dict[str, Any]:
        """Construit les agrégations pour le format externe"""
        result = {
            "total_amount": 0.0,
            "transaction_count": 0,
            "average_amount": 0.0,
            "by_month": [],
            "by_category": [],
            "by_merchant": [],
            "statistics": {}
        }
        
        for agg in internal_aggs:
            if agg.name == "monthly":
                result["by_month"] = [bucket.to_external_bucket() for bucket in agg.buckets]
            elif agg.name == "category":
                result["by_category"] = [bucket.to_external_bucket() for bucket in agg.buckets]
            elif agg.name == "merchant":
                result["by_merchant"] = [bucket.to_external_bucket() for bucket in agg.buckets]
            
            # Accumulation des totaux
            if agg.total_amount:
                result["total_amount"] += agg.total_amount
            if agg.total_count:
                result["transaction_count"] += agg.total_count
        
        # Calculs finaux
        if result["transaction_count"] > 0:
            result["average_amount"] = result["total_amount"] / result["transaction_count"]
        
        # Statistiques globales
        all_amounts = []
        for agg in internal_aggs:
            for bucket in agg.buckets:
                if bucket.get_metric("sum_amount") > 0:
                    all_amounts.append(bucket.get_metric("sum_amount"))
        
        if all_amounts:
            result["statistics"] = {
                "min_amount": min(all_amounts),
                "max_amount": max(all_amounts),
                "std_deviation": ResponseTransformer._calculate_std_dev(all_amounts)
            }
        
        return result
    
    @staticmethod
    def _determine_complexity(internal_response: InternalSearchResponse) -> str:
        """Détermine la complexité de la requête exécutée"""
        complexity_score = 0
        
        # Score basé sur le temps d'exécution
        exec_time = internal_response.execution_metrics.total_execution_time
        if exec_time > 200:
            complexity_score += 2
        elif exec_time > 100:
            complexity_score += 1
        
        # Score basé sur les optimisations
        complexity_score += len(internal_response.execution_metrics.optimizations_applied)
        
        # Score basé sur les agrégations
        complexity_score += len(internal_response.aggregations)
        
        # Score basé sur le volume de résultats
        if internal_response.total_hits > 1000:
            complexity_score += 2
        elif internal_response.total_hits > 100:
            complexity_score += 1
        
        if complexity_score <= 2:
            return "simple"
        elif complexity_score <= 5:
            return "medium"
        else:
            return "complex"
    
    @staticmethod
    def _calculate_std_dev(values: List[float]) -> float:
        """Calcule l'écart-type"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5


# === AJOUT DES MÉTHODES MANQUANTES POUR InternalSearchResponse ===

# Extension de la classe InternalSearchResponse avec les méthodes manquantes
def _calculate_quality_score(self):
    """Calcule le score de qualité global"""
    if not self.raw_results:
        self.quality_score = 0.0
        self.quality_indicator = QualityIndicator.VERY_POOR
        return
    
    # Score basé sur plusieurs facteurs
    scores = []
    
    # 1. Score moyen des résultats
    avg_score = sum(r.score for r in self.raw_results) / len(self.raw_results)
    scores.append(avg_score)
    
    # 2. Distribution des scores (éviter trop de scores faibles)
    high_score_ratio = len([r for r in self.raw_results if r.score > 0.7]) / len(self.raw_results)
    scores.append(high_score_ratio)
    
    # 3. Couverture des résultats vs demande
    if self.total_hits > 0:
        coverage_ratio = min(len(self.raw_results) / min(20, self.total_hits), 1.0)
        scores.append(coverage_ratio)
    
    # Score final pondéré
    self.quality_score = sum(scores) / len(scores)
    
    # Détermination de l'indicateur
    if self.quality_score > 0.9:
        self.quality_indicator = QualityIndicator.EXCELLENT
    elif self.quality_score > 0.7:
        self.quality_indicator = QualityIndicator.GOOD
    elif self.quality_score > 0.5:
        self.quality_indicator = QualityIndicator.AVERAGE
    elif self.quality_score > 0.3:
        self.quality_indicator = QualityIndicator.POOR
    else:
        self.quality_indicator = QualityIndicator.VERY_POOR


def _generate_followup_suggestions(self):
    """Génère des suggestions de questions de suivi"""
    suggestions = []
    
    # Basé sur les résultats trouvés
    if self.raw_results:
        # Suggestions basées sur les catégories trouvées
        categories = list(set(r.source.get("category_name") for r in self.raw_results 
                            if r.source.get("category_name")))
        if categories:
            self.related_categories = categories[:5]
            for cat in categories[:3]:
                suggestions.append(f"Voir plus de transactions {cat}")
        
        # Suggestions basées sur les marchands
        merchants = list(set(r.source.get("merchant_name") for r in self.raw_results 
                           if r.source.get("merchant_name")))
        if merchants:
            for merchant in merchants[:2]:
                suggestions.append(f"Historique complet {merchant}")
        
        # Suggestions temporelles
        if len(self.raw_results) > 5:
            suggestions.append("Comparer avec le mois précédent")
            suggestions.append("Évolution sur 6 mois")
    
    # Suggestions d'agrégation si pas déjà fait
    if not self.aggregations and len(self.raw_results) > 3:
        suggestions.append("Voir le total par catégorie")
        suggestions.append("Répartition par mois")
    
    self.suggested_followups = suggestions[:5]  # Limite à 5 suggestions


def _get_standardized_results(self) -> List[Dict[str, Any]]:
    """Retourne les résultats au format standardisé"""
    return [result.to_standardized_result() for result in self.raw_results]


def _mark_completed(self):
    """Marque la réponse comme terminée"""
    self.execution_metrics.mark_completed()


def _add_raw_result(self, result: RawTransaction):
    """Ajoute un résultat brut"""
    self.raw_results.append(result)
    
    # Mise à jour du score max
    if self.max_score is None or result.score > self.max_score:
        self.max_score = result.score


def _add_aggregation(self, aggregation: InternalAggregationResult):
    """Ajoute un résultat d'agrégation"""
    aggregation.calculate_totals()
    self.aggregations.append(aggregation)


def _get_performance_summary(self) -> Dict[str, Any]:
    """Retourne un résumé de performance"""
    return {
        "status": self.status.value,
        "total_time_ms": self.execution_metrics.total_execution_time,
        "elasticsearch_time_ms": self.execution_metrics.elasticsearch_took,
        "served_from_cache": self.served_from_cache,
        "optimizations_count": len(self.execution_metrics.optimizations_applied),
        "quality_score": round(self.quality_score, 3),
        "quality_indicator": self.quality_indicator.value,
        "results_count": len(self.raw_results),
        "total_hits": self.total_hits
    }


# Ajouter les méthodes à la classe InternalSearchResponse
InternalSearchResponse.calculate_quality_score = _calculate_quality_score
InternalSearchResponse.generate_followup_suggestions = _generate_followup_suggestions
InternalSearchResponse.get_standardized_results = _get_standardized_results
InternalSearchResponse.mark_completed = _mark_completed
InternalSearchResponse.add_raw_result = _add_raw_result
InternalSearchResponse.add_aggregation = _add_aggregation
InternalSearchResponse.get_performance_summary = _get_performance_summary


# === BUILDERS ET HELPERS ===

class ResponseBuilder:
    """Builder pour construire facilement des réponses internes"""
    
    def __init__(self, request_id: str, user_id: int):
        self.response = InternalSearchResponse(
            request_id=request_id,
            user_id=user_id,
            status=ExecutionStatus.SUCCESS
        )
    
    def add_elasticsearch_results(self, es_response: Dict[str, Any]) -> 'ResponseBuilder':
        """Ajoute les résultats Elasticsearch"""
        self.response.elasticsearch_response = es_response
        self.response.total_hits = es_response.get("hits", {}).get("total", {}).get("value", 0)
        self.response.execution_metrics.elasticsearch_took = es_response.get("took", 0)
        
        # Convertir les hits
        for hit in es_response.get("hits", {}).get("hits", []):
            raw_result = RawTransaction(
                source=hit.get("_source", {}),
                score=hit.get("_score", 0.0),
                index=hit.get("_index", ""),
                id=hit.get("_id", ""),
                highlights=hit.get("highlight"),
                explanation=hit.get("_explanation")
            )
            self.response.add_raw_result(raw_result)
        
        return self
    
    def add_cache_info(self, cache_hit: bool, cache_key: str = None) -> 'ResponseBuilder':
        """Ajoute les informations de cache"""
        self.response.served_from_cache = cache_hit
        self.response.cache_key = cache_key
        if cache_hit:
            self.response.execution_metrics.add_optimization(OptimizationType.CACHE_HIT)
        return self
    
    def add_optimization(self, optimization: OptimizationType) -> 'ResponseBuilder':
        """Ajoute une optimisation"""
        self.response.execution_metrics.add_optimization(optimization)
        return self
    
    def set_status(self, status: ExecutionStatus) -> 'ResponseBuilder':
        """Définit le statut"""
        self.response.status = status
        return self
    
    def build(self) -> InternalSearchResponse:
        """Construit la réponse finale"""
        self.response.mark_completed()
        return self.response


# === HELPERS POUR LES MODÈLES API ===

class ValidationResponseBuilder:
    """Builder pour les réponses de validation"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.security_result = SecurityCheckResult(
            passed=True,
            user_id_check=True,
            data_isolation_check=True,
            permissions_check=True
        )
        self.performance_analysis = PerformanceAnalysis(
            complexity="simple",
            estimated_time_ms=50
        )
    
    def add_error(self, field: str, error_type: str, message: str, 
                  severity: ValidationSeverity = ValidationSeverity.ERROR,
                  suggested_fix: str = None) -> 'ValidationResponseBuilder':
        """Ajoute une erreur de validation"""
        self.errors.append(ValidationError(
            field=field,
            error_type=error_type,
            message=message,
            severity=severity,
            suggested_fix=suggested_fix
        ))
        return self
    
    def add_warning(self, field: str, message: str, 
                   suggested_fix: str = None) -> 'ValidationResponseBuilder':
        """Ajoute un avertissement"""
        self.warnings.append(ValidationError(
            field=field,
            error_type="warning",
            message=message,
            severity=ValidationSeverity.WARNING,
            suggested_fix=suggested_fix
        ))
        return self
    
    def set_security_check(self, passed: bool, **kwargs) -> 'ValidationResponseBuilder':
        """Configure la vérification sécurité"""
        self.security_result = SecurityCheckResult(passed=passed, **kwargs)
        return self
    
    def set_performance_analysis(self, complexity: str, estimated_time_ms: int,
                               **kwargs) -> 'ValidationResponseBuilder':
        """Configure l'analyse de performance"""
        self.performance_analysis = PerformanceAnalysis(
            complexity=complexity,
            estimated_time_ms=estimated_time_ms,
            **kwargs
        )
        return self
    
    def build(self, metadata: Dict[str, Any] = None) -> ValidationResponse:
        """Construit la réponse de validation"""
        return ValidationResponse(
            valid=len(self.errors) == 0,
            errors=self.errors,
            warnings=self.warnings,
            security_check=self.security_result,
            performance_analysis=self.performance_analysis,
            metadata=metadata or {}
        )


class HealthResponseBuilder:
    """Builder pour les réponses de santé - Compatible avec routes.py"""
    
    def __init__(self):
        self.components = []
        self.system_health = SystemHealth(
            overall_status="healthy",
            uptime_seconds=0.0,
            memory_usage_mb=0.0,
            cpu_usage_percent=0.0
        )
    
    def add_component(self, name: str, status: str,
                     response_time_ms: float = None, error_message: str = None,
                     dependencies: List[str] = None, 
                     metrics: Dict[str, Any] = None) -> 'HealthResponseBuilder':
        """Ajoute un composant avec status en string"""
        self.components.append(ComponentHealthInfo(
            name=name,
            status=status,
            last_check=datetime.now(timezone.utc),
            response_time_ms=response_time_ms,
            error_message=error_message,
            dependencies=dependencies or [],
            metrics=metrics or {}
        ))
        return self
    
    def set_system_health(self, **kwargs) -> 'HealthResponseBuilder':
        """Configure la santé système"""
        for key, value in kwargs.items():
            if hasattr(self.system_health, key):
                setattr(self.system_health, key, value)
        return self
    
    def calculate_overall_status(self) -> 'HealthResponseBuilder':
        """Calcule le statut global basé sur les composants"""
        if not self.components:
            self.system_health.overall_status = "unknown"
            return self
        
        statuses = [comp.status for comp in self.components]
        
        if "unhealthy" in statuses:
            self.system_health.overall_status = "unhealthy"
        elif "degraded" in statuses:
            self.system_health.overall_status = "degraded"
        elif all(status == "healthy" for status in statuses):
            self.system_health.overall_status = "healthy"
        else:
            self.system_health.overall_status = "unknown"
        
        return self
    
    def build(self, service_version: str = "1.0.0", environment: str = "production",
             metadata: Dict[str, Any] = None) -> HealthResponse:
        """Construit la réponse de santé"""
        self.calculate_overall_status()
        
        return HealthResponse(
            system=self.system_health,
            components=self.components,
            service_version=service_version,
            environment=environment,
            metadata=metadata or {}
        )


class TemplateResponseBuilder:
    """Builder pour les réponses de templates"""
    
    def __init__(self):
        self.templates = {}
        self.categories = set()
        self.intent_types = set()
    
    def add_template(self, name: str, category: str, intent_type: str,
                    description: str, complexity: str = "medium",
                    usage_count: int = 0, avg_execution_time_ms: float = 0.0,
                    cache_hit_rate: float = 0.0, 
                    last_used: datetime = None) -> 'TemplateResponseBuilder':
        """Ajoute un template"""
        self.templates[name] = TemplateInfo(
            name=name,
            category=category,
            intent_type=intent_type,
            description=description,
            complexity=complexity,
            usage_count=usage_count,
            avg_execution_time_ms=avg_execution_time_ms,
            cache_hit_rate=cache_hit_rate,
            last_used=last_used
        )
        
        self.categories.add(category)
        self.intent_types.add(intent_type)
        return self
    
    def build(self, metadata: Dict[str, Any] = None) -> TemplateListResponse:
        """Construit la réponse des templates"""
        return TemplateListResponse(
            templates=self.templates,
            total_count=len(self.templates),
            categories=sorted(list(self.categories)),
            intent_types=sorted(list(self.intent_types)),
            metadata=metadata or {}
        )


# === UTILITAIRES DE CONVERSION ===

class ResponseConverter:
    """Convertisseur entre différents formats de réponse"""
    
    @staticmethod
    def dict_to_template_info(template_dict: Dict[str, Any], name: str) -> TemplateInfo:
        """Convertit un dictionnaire en TemplateInfo"""
        return TemplateInfo(
            name=name,
            category=template_dict.get("category", "unknown"),
            intent_type=template_dict.get("intent_type", "unknown"),
            description=template_dict.get("description", ""),
            complexity=template_dict.get("complexity", "medium"),
            usage_count=template_dict.get("usage_count", 0),
            avg_execution_time_ms=template_dict.get("avg_execution_time_ms", 0.0),
            cache_hit_rate=template_dict.get("cache_hit_rate", 0.0),
            last_used=template_dict.get("last_used")
        )
    
    @staticmethod
    def metrics_dict_to_response(metrics_dict: Dict[str, Any], 
                               format_type: str = "json") -> MetricsResponse:
        """Convertit des métriques en MetricsResponse"""
        import json
        
        if format_type == "json":
            content = json.dumps(metrics_dict, indent=2, default=str)
        else:
            content = str(metrics_dict)
        
        return MetricsResponse(
            format=format_type,
            content=content,
            metrics_count=len(metrics_dict.get("metrics", {})),
            metadata={
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "content_length": len(content)
            }
        )


# === VALIDATEURS SPÉCIALISÉS ===

class ResponseValidator:
    """Validateur pour les modèles de réponse"""
    
    @staticmethod
    def validate_internal_response(response: InternalSearchResponse) -> bool:
        """Valide une réponse interne"""
        try:
            # Vérification cohérence des hits
            if response.total_hits < len(response.raw_results):
                raise ValueError("total_hits ne peut pas être inférieur au nombre de résultats")
            
            # Vérification scores
            if response.max_score is not None and response.raw_results:
                max_result_score = max(r.score for r in response.raw_results)
                if response.max_score < max_result_score:
                    raise ValueError("max_score incohérent avec les scores des résultats")
            
            # Vérification quality_score
            if not 0 <= response.quality_score <= 1:
                raise ValueError("quality_score doit être entre 0 et 1")
            
            return True
        except Exception as e:
            raise ValueError(f"Validation réponse interne échouée: {e}")
    
    @staticmethod
    def validate_api_response(response: Union[ValidationResponse, HealthResponse, 
                                           TemplateListResponse]) -> bool:
        """Valide une réponse API"""
        try:
            if isinstance(response, ValidationResponse):
                # La validation ne peut pas être valid s'il y a des erreurs
                if response.valid and response.errors:
                    raise ValueError("Réponse ne peut pas être valid avec des erreurs")
                
                # Les erreurs critiques doivent rendre la validation invalid
                critical_errors = [e for e in response.errors 
                                 if e.severity == ValidationSeverity.CRITICAL]
                if critical_errors and response.valid:
                    raise ValueError("Erreurs critiques doivent rendre la validation invalid")
            
            elif isinstance(response, HealthResponse):
                # Vérification cohérence statut global vs composants
                unhealthy_components = [c for c in response.components 
                                      if c.status == "unhealthy"]
                if unhealthy_components and response.system.overall_status == "healthy":
                    raise ValueError("Statut global incohérent avec composants unhealthy")
            
            elif isinstance(response, TemplateListResponse):
                # Vérification cohérence count
                if response.total_count != len(response.templates):
                    raise ValueError("total_count incohérent avec le nombre de templates")
            
            return True
        except Exception as e:
            raise ValueError(f"Validation réponse API échouée: {e}")


# === FACTORIES ET HELPERS SUPPLÉMENTAIRES ===

class HealthResponseFactory:
    """Factory pour créer facilement des réponses de santé"""
    
    @staticmethod
    def create_healthy_response(
        uptime_seconds: float = 0.0,
        memory_mb: float = 0.0,
        cpu_percent: float = 0.0
    ) -> HealthResponse:
        """Crée une réponse de santé OK"""
        
        system = SystemHealth(
            overall_status="healthy",
            uptime_seconds=uptime_seconds,
            memory_usage_mb=memory_mb,
            cpu_usage_percent=cpu_percent,
            active_connections=0,
            total_requests=0,
            error_rate_percent=0.0
        )
        
        components = [
            ComponentHealthInfo(
                name="core_engine",
                status="healthy",
                last_check=datetime.now(timezone.utc),
                response_time_ms=5.0,
                dependencies=["elasticsearch"],
                metrics={"status": "operational"}
            ),
            ComponentHealthInfo(
                name="elasticsearch",
                status="healthy",
                last_check=datetime.now(timezone.utc),
                response_time_ms=10.0,
                dependencies=[],
                metrics={"cluster_status": "green"}
            )
        ]
        
        return HealthResponse(
            system=system,
            components=components,
            service_version="1.0.0",
            environment="production",
            metadata={
                "created_by": "factory",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    
    @staticmethod
    def create_unhealthy_response(error_message: str = "Service unavailable") -> HealthResponse:
        """Crée une réponse de santé en erreur"""
        
        system = SystemHealth(
            overall_status="unhealthy",
            uptime_seconds=0.0,
            memory_usage_mb=0.0,
            cpu_usage_percent=0.0,
            active_connections=0,
            total_requests=0,
            error_rate_percent=100.0
        )
        
        components = [
            ComponentHealthInfo(
                name="system",
                status="unhealthy",
                last_check=datetime.now(timezone.utc),
                error_message=error_message,
                dependencies=[],
                metrics={}
            )
        ]
        
        return HealthResponse(
            system=system,
            components=components,
            service_version="1.0.0",
            environment="unknown",
            metadata={
                "created_by": "factory",
                "error": error_message,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )


class ValidationResponseFactory:
    """Factory pour créer facilement des réponses de validation"""
    
    @staticmethod
    def create_valid_response(metadata: Dict[str, Any] = None) -> ValidationResponse:
        """Crée une réponse de validation valide"""
        
        security_check = SecurityCheckResult(
            passed=True,
            user_id_check=True,
            data_isolation_check=True,
            permissions_check=True,
            warnings=[]
        )
        
        performance_analysis = PerformanceAnalysis(
            complexity="simple",
            estimated_time_ms=25,
            warnings=[],
            optimization_suggestions=[],
            cache_eligible=True,
            field_count=3,
            filter_count=1
        )
        
        return ValidationResponse(
            valid=True,
            errors=[],
            warnings=[],
            security_check=security_check,
            performance_analysis=performance_analysis,
            metadata=metadata or {}
        )
    
    @staticmethod
    def create_invalid_response(
        field: str, 
        error_message: str, 
        metadata: Dict[str, Any] = None
    ) -> ValidationResponse:
        """Crée une réponse de validation invalide"""
        
        error = ValidationError(
            field=field,
            error_type="validation_error",
            message=error_message,
            severity=ValidationSeverity.ERROR,
            suggested_fix=f"Please correct the {field} field"
        )
        
        security_check = SecurityCheckResult(
            passed=False,
            user_id_check=True,
            data_isolation_check=True,
            permissions_check=True,
            warnings=["Request validation failed"]
        )
        
        performance_analysis = PerformanceAnalysis(
            complexity="unknown",
            estimated_time_ms=0,
            warnings=["Cannot estimate performance for invalid request"],
            optimization_suggestions=[],
            cache_eligible=False,
            field_count=0,
            filter_count=0
        )
        
        return ValidationResponse(
            valid=False,
            errors=[error],
            warnings=[],
            security_check=security_check,
            performance_analysis=performance_analysis,
            metadata=metadata or {}
        )


# === UTILITAIRES DE SÉRIALISATION ===

class ResponseSerializer:
    """Utilitaires pour sérialiser les réponses"""
    
    @staticmethod
    def serialize_health_response(response: HealthResponse) -> Dict[str, Any]:
        """Sérialise une HealthResponse en dictionnaire"""
        return {
            "system": {
                "overall_status": response.system.overall_status,
                "uptime_seconds": response.system.uptime_seconds,
                "memory_usage_mb": response.system.memory_usage_mb,
                "cpu_usage_percent": response.system.cpu_usage_percent,
                "active_connections": response.system.active_connections,
                "total_requests": response.system.total_requests,
                "error_rate_percent": response.system.error_rate_percent
            },
            "components": [
                {
                    "name": comp.name,
                    "status": comp.status,
                    "last_check": comp.last_check.isoformat(),
                    "response_time_ms": comp.response_time_ms,
                    "error_message": comp.error_message,
                    "dependencies": comp.dependencies,
                    "metrics": comp.metrics
                }
                for comp in response.components
            ],
            "timestamp": response.timestamp.isoformat(),
            "service_version": response.service_version,
            "environment": response.environment,
            "metadata": response.metadata
        }
    
    @staticmethod
    def deserialize_health_response(data: Dict[str, Any]) -> HealthResponse:
        """Désérialise un dictionnaire en HealthResponse"""
        
        system = SystemHealth(**data["system"])
        
        components = [
            ComponentHealthInfo(
                name=comp["name"],
                status=comp["status"],
                last_check=datetime.fromisoformat(comp["last_check"]),
                response_time_ms=comp.get("response_time_ms"),
                error_message=comp.get("error_message"),
                dependencies=comp.get("dependencies", []),
                metrics=comp.get("metrics", {})
            )
            for comp in data["components"]
        ]
        
        return HealthResponse(
            system=system,
            components=components,
            timestamp=datetime.fromisoformat(data["timestamp"]),
            service_version=data["service_version"],
            environment=data["environment"],
            metadata=data.get("metadata", {})
        )


# === HELPERS POUR TESTS ===

class ResponseTestHelpers:
    """Helpers pour les tests des modèles de réponse"""
    
    @staticmethod
    def create_mock_raw_transaction(
        transaction_id: str = "txn_123",
        amount: float = 100.0,
        score: float = 0.8
    ) -> RawTransaction:
        """Crée une transaction de test"""
        
        source = {
            "transaction_id": transaction_id,
            "user_id": 1,
            "account_id": "acc_123",
            "amount": amount,
            "amount_abs": abs(amount),
            "transaction_type": "debit" if amount < 0 else "credit",
            "currency_code": "EUR",
            "date": "2024-01-15",
            "primary_description": "Test transaction",
            "merchant_name": "Test Merchant",
            "category_name": "Test Category",
            "operation_type": "card_payment",
            "month_year": "2024-01",
            "weekday": "Monday"
        }
        
        return RawTransaction(
            source=source,
            score=score,
            index="test_index",
            id=transaction_id,
            highlights={"primary_description": ["Test <em>transaction</em>"]},
            explanation={"value": score, "description": "BM25 score"}
        )
    
    @staticmethod
    def create_mock_internal_response(
        request_id: str = "req_123",
        user_id: int = 1,
        num_results: int = 5
    ) -> InternalSearchResponse:
        """Crée une réponse interne de test"""
        
        response = InternalSearchResponse(
            request_id=request_id,
            user_id=user_id,
            status=ExecutionStatus.SUCCESS
        )
        
        # Ajouter des résultats de test
        for i in range(num_results):
            transaction = ResponseTestHelpers.create_mock_raw_transaction(
                transaction_id=f"txn_{i}",
                amount=100.0 * (i + 1),
                score=0.9 - (i * 0.1)
            )
            response.add_raw_result(transaction)
        
        response.total_hits = num_results
        response.served_from_cache = False
        
        # Simuler les métriques
        response.execution_metrics.elasticsearch_took = 15
        response.execution_metrics.total_execution_time = 45
        
        return response
    
    @staticmethod
    def assert_health_response_valid(response: HealthResponse):
        """Valide qu'une HealthResponse est correcte"""
        assert response.system is not None
        assert response.components is not None
        assert len(response.components) > 0
        assert response.service_version is not None
        assert response.timestamp is not None
        
        # Vérifier la cohérence du statut
        if response.system.overall_status == "healthy":
            unhealthy_components = [c for c in response.components if c.status == "unhealthy"]
            assert len(unhealthy_components) == 0, "Statut global healthy avec composants unhealthy"


# === EXPORTS ===

__all__ = [
    # === ENUMS ===
    "ExecutionStatus",
    "OptimizationType", 
    "QualityIndicator",
    "ComponentStatus",
    "ValidationSeverity",
    
    # === STRUCTURES INTERNES ===
    "RawTransaction",
    "AggregationBucketInternal",
    "InternalAggregationResult",
    "ExecutionMetrics",
    
    # === MODÈLES INTERNES ===
    "InternalSearchResponse",
    
    # === MODÈLES API REST ===
    "ValidationError",
    "SecurityCheckResult",
    "PerformanceAnalysis",
    "ValidationResponse",
    "TemplateInfo",
    "TemplateListResponse",
    "ComponentHealthInfo",
    "SystemHealth",
    "HealthResponse",
    "MetricsResponse",
    
    # === TRANSFORMATEURS ET BUILDERS ===
    "ResponseTransformer",
    "ResponseBuilder",
    "ValidationResponseBuilder",
    "HealthResponseBuilder",
    "TemplateResponseBuilder",
    
    # === FACTORIES ===
    "HealthResponseFactory",
    "ValidationResponseFactory",
    
    # === UTILITAIRES ===
    "ResponseConverter",
    "ResponseValidator",
    "ResponseSerializer",
    "ResponseTestHelpers"
]


# === INFORMATIONS DU MODULE ===

__version__ = "1.0.0"
__author__ = "Search Service Team"  
__description__ = "Modèles de réponses internes et API avec corrections pour compatibilité routes.py"

# Log de chargement
import logging
logger = logging.getLogger(__name__)
logger.info(f"Module models.responses chargé - version {__version__} (avec corrections health check)")