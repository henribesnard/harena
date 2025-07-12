"""
Modèles de réponses internes du Search Service
Structures optimisées pour le traitement interne avant conversion vers contrats
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, field_validator
from dataclasses import dataclass, field

from .service_contracts import SearchServiceResponse, AggregationType


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
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    elasticsearch_took: int = 0
    total_execution_time: int = 0
    cache_lookup_time: int = 0
    query_build_time: int = 0
    result_processing_time: int = 0
    optimizations_applied: List[OptimizationType] = field(default_factory=list)
    
    def mark_completed(self):
        """Marque l'exécution comme terminée"""
        self.end_time = datetime.utcnow()
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
    
    def mark_completed(self):
        """Marque la réponse comme terminée"""
        self.execution_metrics.mark_completed()
    
    def add_raw_result(self, result: RawTransaction):
        """Ajoute un résultat brut"""
        self.raw_results.append(result)
        
        # Mise à jour du score max
        if self.max_score is None or result.score > self.max_score:
            self.max_score = result.score
    
    def add_aggregation(self, aggregation: InternalAggregationResult):
        """Ajoute un résultat d'agrégation"""
        aggregation.calculate_totals()
        self.aggregations.append(aggregation)
    
    def calculate_quality_score(self):
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
    
    def generate_followup_suggestions(self):
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
    
    def get_standardized_results(self) -> List[Dict[str, Any]]:
        """Retourne les résultats au format standardisé"""
        return [result.to_standardized_result() for result in self.raw_results]
    
    def get_performance_summary(self) -> Dict[str, Any]:
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
            "timestamp": datetime.utcnow()
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


# === EXPORTS ===

__all__ = [
    # Enums
    "ExecutionStatus",
    "OptimizationType", 
    "QualityIndicator",
    
    # Dataclasses
    "RawTransaction",
    "AggregationBucketInternal",
    "InternalAggregationResult",
    "ExecutionMetrics",
    
    # Modèle principal
    "InternalSearchResponse",
    
    # Transformateur
    "ResponseTransformer",
    
    # Builder
    "ResponseBuilder"
]