"""
Optimiseur de performance pour les requêtes Elasticsearch
Gère les optimisations adaptatives, la surveillance des performances et l'amélioration continue
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import time
import statistics
from collections import defaultdict, deque

from models.service_contracts import SearchServiceQuery, SearchServiceResponse
from config import settings


logger = logging.getLogger(__name__)


class OptimizationLevel(str, Enum):
    """Niveaux d'optimisation de performance"""
    CONSERVATIVE = "conservative"  # Optimisations sûres uniquement
    BALANCED = "balanced"         # Équilibre performance/précision
    AGGRESSIVE = "aggressive"     # Maximum de performance
    ADAPTIVE = "adaptive"         # S'adapte selon le contexte


class PerformanceProfile(str, Enum):
    """Profils de performance prédéfinis"""
    LATENCY_FIRST = "latency_first"      # Priorité à la vitesse
    PRECISION_FIRST = "precision_first"   # Priorité à la précision
    THROUGHPUT_FIRST = "throughput_first" # Priorité au débit
    COST_OPTIMIZED = "cost_optimized"     # Optimisation coût/performance


class QueryComplexity(str, Enum):
    """Niveaux de complexité des requêtes"""
    SIMPLE = "simple"        # Filtres simples, peu de champs
    MODERATE = "moderate"    # Recherche textuelle basique
    COMPLEX = "complex"      # Multi-champs avec agrégations
    VERY_COMPLEX = "very_complex"  # Requêtes sophistiquées


class OptimizationType(str, Enum):
    """Types d'optimisations disponibles"""
    QUERY_REWRITE = "query_rewrite"
    INDEX_SELECTION = "index_selection"
    CACHE_STRATEGY = "cache_strategy"
    SHARD_ROUTING = "shard_routing"
    FIELD_SELECTION = "field_selection"
    AGGREGATION_OPT = "aggregation_optimization"
    TIMEOUT_TUNING = "timeout_tuning"
    BATCH_PROCESSING = "batch_processing"
    ASYNC_OPTIMIZATION = "async_optimization"


@dataclass
class PerformanceMetrics:
    """Métriques de performance détaillées"""
    execution_time_ms: float
    elasticsearch_time_ms: float
    processing_time_ms: float
    cache_hit: bool
    cache_miss: bool
    query_complexity: QueryComplexity
    optimization_level: OptimizationLevel
    optimizations_applied: List[OptimizationType] = field(default_factory=list)
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    network_io_bytes: int = 0
    elasticsearch_shards: int = 1
    results_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OptimizationRule:
    """Règle d'optimisation conditionnelle"""
    name: str
    condition: str  # Expression évaluable
    optimization_type: OptimizationType
    parameters: Dict[str, Any]
    priority: int = 1
    enabled: bool = True
    success_count: int = 0
    failure_count: int = 0
    avg_improvement_percent: float = 0.0


class PerformanceHistoryEntry(NamedTuple):
    """Entrée d'historique de performance"""
    timestamp: datetime
    query_signature: str
    metrics: PerformanceMetrics
    optimizations: List[OptimizationType]
    result_quality: float


@dataclass
class PerformanceConfiguration:
    """Configuration globale de performance"""
    # Limites de performance
    max_query_time_ms: int = 2000
    max_cache_size_mb: int = 512
    max_concurrent_queries: int = 100
    
    # Seuils d'optimisation
    slow_query_threshold_ms: int = 500
    cache_hit_target_percent: float = 80.0
    memory_usage_threshold_mb: float = 1024
    
    # Stratégies adaptatives
    adaptive_optimization: bool = True
    auto_tune_timeouts: bool = True
    dynamic_cache_sizing: bool = True
    
    # Monitoring
    metrics_retention_hours: int = 24
    performance_logging: bool = True
    detailed_profiling: bool = False


class PerformanceOptimizer:
    """Optimiseur principal de performance pour les requêtes Elasticsearch"""
    
    def __init__(self, config: Optional[PerformanceConfiguration] = None):
        self.config = config or PerformanceConfiguration()
        self.metrics_history: deque = deque(maxlen=10000)
        self.optimization_rules: List[OptimizationRule] = []
        self.query_signatures_cache: Dict[str, Any] = {}
        self.performance_stats: Dict[str, Any] = defaultdict(list)
        
        # Gestionnaires spécialisés
        self.cache_optimizer = CacheOptimizer()
        self.query_rewriter = QueryRewriter()
        self.timeout_manager = TimeoutManager()
        self.batch_processor = BatchProcessor()
        
        self._initialize_default_rules()
        self._start_background_tasks()
        
        logger.info("PerformanceOptimizer initialisé")
    
    async def optimize_query(
        self, 
        query_body: Dict[str, Any],
        search_request: SearchServiceQuery,
        performance_profile: PerformanceProfile = PerformanceProfile.BALANCED
    ) -> Tuple[Dict[str, Any], List[OptimizationType]]:
        """
        Optimise une requête selon le profil de performance
        
        Returns:
            Tuple[requête optimisée, optimisations appliquées]
        """
        start_time = time.time()
        optimizations_applied = []
        
        try:
            # Analyser la complexité de la requête
            complexity = self._analyze_query_complexity(query_body, search_request)
            
            # Sélectionner le niveau d'optimisation
            optimization_level = self._select_optimization_level(
                complexity, performance_profile
            )
            
            # Appliquer les optimisations par ordre de priorité
            optimized_query = query_body.copy()
            
            # 1. Optimisations de cache
            if self._should_apply_cache_optimization(search_request):
                optimized_query = await self.cache_optimizer.optimize_for_cache(
                    optimized_query, search_request
                )
                optimizations_applied.append(OptimizationType.CACHE_STRATEGY)
            
            # 2. Réécriture de requête
            if optimization_level in [OptimizationLevel.BALANCED, OptimizationLevel.AGGRESSIVE]:
                rewritten_query = await self.query_rewriter.rewrite_for_performance(
                    optimized_query, complexity, performance_profile
                )
                if rewritten_query != optimized_query:
                    optimized_query = rewritten_query
                    optimizations_applied.append(OptimizationType.QUERY_REWRITE)
            
            # 3. Optimisation des champs
            if self._should_optimize_field_selection(search_request):
                optimized_query = self._optimize_field_selection(
                    optimized_query, search_request
                )
                optimizations_applied.append(OptimizationType.FIELD_SELECTION)
            
            # 4. Optimisation des agrégations
            if "aggs" in optimized_query and optimized_query["aggs"]:
                optimized_query = self._optimize_aggregations(
                    optimized_query, optimization_level
                )
                optimizations_applied.append(OptimizationType.AGGREGATION_OPT)
            
            # 5. Gestion des timeouts
            timeout_ms = self.timeout_manager.calculate_optimal_timeout(
                complexity, performance_profile
            )
            optimized_query["timeout"] = f"{timeout_ms}ms"
            optimizations_applied.append(OptimizationType.TIMEOUT_TUNING)
            
            # 6. Routage des shards
            if optimization_level == OptimizationLevel.AGGRESSIVE:
                optimized_query = self._optimize_shard_routing(
                    optimized_query, search_request
                )
                optimizations_applied.append(OptimizationType.SHARD_ROUTING)
            
            # Enregistrer les métriques d'optimisation
            optimization_time = (time.time() - start_time) * 1000
            self._record_optimization_metrics(
                search_request, complexity, optimization_level, 
                optimizations_applied, optimization_time
            )
            
            return optimized_query, optimizations_applied
            
        except Exception as e:
            logger.error(f"Erreur lors de l'optimisation de requête: {e}")
            return query_body, []
    
    async def analyze_performance(
        self, 
        metrics: PerformanceMetrics,
        search_request: SearchServiceQuery,
        response: SearchServiceResponse
    ) -> Dict[str, Any]:
        """Analyse les performances d'une requête exécutée"""
        
        # Enregistrer dans l'historique
        query_signature = self._generate_query_signature(search_request)
        history_entry = PerformanceHistoryEntry(
            timestamp=datetime.now(),
            query_signature=query_signature,
            metrics=metrics,
            optimizations=metrics.optimizations_applied,
            result_quality=response.context_enrichment.result_quality_score
        )
        self.metrics_history.append(history_entry)
        
        # Analyser les tendances
        analysis = {
            "performance_trend": self._analyze_performance_trend(query_signature),
            "optimization_effectiveness": self._analyze_optimization_effectiveness(metrics),
            "recommendations": self._generate_performance_recommendations(metrics),
            "anomalies": self._detect_performance_anomalies(metrics),
            "resource_usage": self._analyze_resource_usage(metrics)
        }
        
        # Mettre à jour les règles d'optimisation adaptatives
        if self.config.adaptive_optimization:
            await self._update_optimization_rules(metrics, analysis)
        
        return analysis
    
    def _analyze_query_complexity(
        self, 
        query_body: Dict[str, Any],
        search_request: SearchServiceQuery
    ) -> QueryComplexity:
        """Analyse la complexité d'une requête"""
        
        complexity_score = 0
        
        # Analyser la structure de la requête
        if "query" in query_body:
            query_part = query_body["query"]
            
            # Recherche textuelle
            if self._has_text_search(query_part):
                complexity_score += 2
            
            # Filtres multiples
            filter_count = self._count_filters(query_part)
            complexity_score += min(filter_count, 5)
            
            # Requêtes booléennes complexes
            if self._has_complex_boolean(query_part):
                complexity_score += 3
        
        # Agrégations
        if "aggs" in query_body and query_body["aggs"]:
            agg_complexity = self._analyze_aggregation_complexity(query_body["aggs"])
            complexity_score += agg_complexity
        
        # Tri personnalisé
        if "sort" in query_body and len(query_body["sort"]) > 1:
            complexity_score += 1
        
        # Taille de résultats
        size = query_body.get("size", 10)
        if size > 100:
            complexity_score += 2
        elif size > 1000:
            complexity_score += 4
        
        # Déterminer le niveau de complexité
        if complexity_score <= 3:
            return QueryComplexity.SIMPLE
        elif complexity_score <= 7:
            return QueryComplexity.MODERATE
        elif complexity_score <= 12:
            return QueryComplexity.COMPLEX
        else:
            return QueryComplexity.VERY_COMPLEX
    
    def _select_optimization_level(
        self,
        complexity: QueryComplexity,
        performance_profile: PerformanceProfile
    ) -> OptimizationLevel:
        """Sélectionne le niveau d'optimisation approprié"""
        
        if performance_profile == PerformanceProfile.PRECISION_FIRST:
            return OptimizationLevel.CONSERVATIVE
        
        elif performance_profile == PerformanceProfile.LATENCY_FIRST:
            if complexity in [QueryComplexity.SIMPLE, QueryComplexity.MODERATE]:
                return OptimizationLevel.AGGRESSIVE
            else:
                return OptimizationLevel.BALANCED
        
        elif performance_profile == PerformanceProfile.THROUGHPUT_FIRST:
            return OptimizationLevel.AGGRESSIVE
        
        elif performance_profile == PerformanceProfile.COST_OPTIMIZED:
            return OptimizationLevel.BALANCED
        
        else:  # Défaut : adaptatif basé sur la complexité
            if complexity == QueryComplexity.SIMPLE:
                return OptimizationLevel.CONSERVATIVE
            elif complexity == QueryComplexity.MODERATE:
                return OptimizationLevel.BALANCED
            else:
                return OptimizationLevel.AGGRESSIVE
    
    def _should_apply_cache_optimization(
        self, 
        search_request: SearchServiceQuery
    ) -> bool:
        """Détermine si l'optimisation de cache doit être appliquée"""
        
        # Cache activé pour les requêtes avec filtres
        if search_request.search_parameters.filters:
            return True
        
        # Cache pour les recherches répétitives
        query_signature = self._generate_query_signature(search_request)
        recent_queries = [
            entry for entry in self.metrics_history 
            if entry.query_signature == query_signature 
            and entry.timestamp > datetime.now() - timedelta(minutes=5)
        ]
        
        return len(recent_queries) > 1
    
    def _optimize_field_selection(
        self,
        query_body: Dict[str, Any],
        search_request: SearchServiceQuery
    ) -> Dict[str, Any]:
        """Optimise la sélection des champs retournés"""
        
        optimized = query_body.copy()
        
        # Limiter les champs source selon le contexte
        if not search_request.search_parameters.include_raw_data:
            essential_fields = [
                "transaction_id", "user_id", "account_id",
                "amount", "amount_abs", "transaction_type",
                "date", "primary_description", "merchant_name",
                "category_name", "operation_type"
            ]
            optimized["_source"] = essential_fields
        
        # Désactiver _source pour les agrégations uniquement
        if ("aggs" in optimized and optimized["aggs"] and 
            optimized.get("size", 10) == 0):
            optimized["_source"] = False
        
        return optimized
    
    def _optimize_aggregations(
        self,
        query_body: Dict[str, Any],
        optimization_level: OptimizationLevel
    ) -> Dict[str, Any]:
        """Optimise les agrégations pour la performance"""
        
        optimized = query_body.copy()
        aggs = optimized.get("aggs", {})
        
        if not aggs:
            return optimized
        
        # Limiter la taille des buckets
        if optimization_level == OptimizationLevel.AGGRESSIVE:
            for agg_name, agg_config in aggs.items():
                if "terms" in agg_config:
                    if "size" not in agg_config["terms"]:
                        agg_config["terms"]["size"] = 50
                    elif agg_config["terms"]["size"] > 100:
                        agg_config["terms"]["size"] = 100
        
        # Optimiser les agrégations de dates
        for agg_name, agg_config in aggs.items():
            if "date_histogram" in agg_config:
                # Utiliser un intervalle optimal selon la période
                if "calendar_interval" not in agg_config["date_histogram"]:
                    agg_config["date_histogram"]["calendar_interval"] = "1d"
        
        optimized["aggs"] = aggs
        return optimized
    
    def _optimize_shard_routing(
        self,
        query_body: Dict[str, Any],
        search_request: SearchServiceQuery
    ) -> Dict[str, Any]:
        """Optimise le routage vers les shards"""
        
        optimized = query_body.copy()
        
        # Utiliser user_id pour le routage si disponible
        if search_request.query_metadata.user_id:
            optimized["preference"] = f"_shards:0|_local"
            # Routing basé sur user_id pour regrouper les données
            optimized["routing"] = str(search_request.query_metadata.user_id)
        
        return optimized
    
    def _generate_query_signature(self, search_request: SearchServiceQuery) -> str:
        """Génère une signature unique pour une requête"""
        
        signature_parts = [
            f"intent:{search_request.query_metadata.intent_type}",
            f"user:{search_request.query_metadata.user_id}",
            f"limit:{search_request.search_parameters.limit}"
        ]
        
        # Ajouter les filtres normalisés
        if search_request.search_parameters.filters:
            filters_str = json.dumps(
                search_request.search_parameters.filters,
                sort_keys=True
            )
            signature_parts.append(f"filters:{hash(filters_str)}")
        
        return "|".join(signature_parts)
    
    def _analyze_performance_trend(self, query_signature: str) -> Dict[str, Any]:
        """Analyse la tendance de performance pour un type de requête"""
        
        relevant_entries = [
            entry for entry in self.metrics_history
            if entry.query_signature == query_signature
            and entry.timestamp > datetime.now() - timedelta(hours=1)
        ]
        
        if len(relevant_entries) < 2:
            return {"trend": "insufficient_data", "sample_size": len(relevant_entries)}
        
        execution_times = [entry.metrics.execution_time_ms for entry in relevant_entries]
        
        # Calculer la tendance
        recent_avg = statistics.mean(execution_times[-5:]) if len(execution_times) >= 5 else statistics.mean(execution_times)
        overall_avg = statistics.mean(execution_times)
        
        trend = "stable"
        if recent_avg > overall_avg * 1.2:
            trend = "degrading"
        elif recent_avg < overall_avg * 0.8:
            trend = "improving"
        
        return {
            "trend": trend,
            "sample_size": len(relevant_entries),
            "recent_avg_ms": recent_avg,
            "overall_avg_ms": overall_avg,
            "std_deviation": statistics.stdev(execution_times) if len(execution_times) > 1 else 0
        }
    
    def _analyze_optimization_effectiveness(
        self, 
        metrics: PerformanceMetrics
    ) -> Dict[str, Any]:
        """Analyse l'efficacité des optimisations appliquées"""
        
        effectiveness = {}
        
        for optimization in metrics.optimizations_applied:
            # Comparer avec les performances sans cette optimisation
            baseline_metrics = self._get_baseline_metrics(optimization)
            
            if baseline_metrics:
                improvement = (
                    (baseline_metrics["avg_time_ms"] - metrics.execution_time_ms) /
                    baseline_metrics["avg_time_ms"] * 100
                )
                effectiveness[optimization.value] = {
                    "improvement_percent": improvement,
                    "baseline_ms": baseline_metrics["avg_time_ms"],
                    "current_ms": metrics.execution_time_ms
                }
        
        return effectiveness
    
    def _generate_performance_recommendations(
        self, 
        metrics: PerformanceMetrics
    ) -> List[str]:
        """Génère des recommandations pour améliorer les performances"""
        
        recommendations = []
        
        # Recommandations basées sur le temps d'exécution
        if metrics.execution_time_ms > self.config.slow_query_threshold_ms:
            recommendations.append(
                f"Requête lente ({metrics.execution_time_ms:.0f}ms). "
                "Considérez l'ajout de filtres ou la réduction de la taille des résultats."
            )
        
        # Recommandations basées sur l'usage de cache
        if not metrics.cache_hit and metrics.query_complexity in [
            QueryComplexity.SIMPLE, QueryComplexity.MODERATE
        ]:
            recommendations.append(
                "Cette requête pourrait bénéficier de la mise en cache. "
                "Activez le cache pour des performances améliorées."
            )
        
        # Recommandations basées sur l'usage mémoire
        if metrics.memory_usage_mb > self.config.memory_usage_threshold_mb:
            recommendations.append(
                f"Usage mémoire élevé ({metrics.memory_usage_mb:.0f}MB). "
                "Réduisez la taille des résultats ou utilisez la pagination."
            )
        
        # Recommandations pour les agrégations
        if (metrics.query_complexity == QueryComplexity.VERY_COMPLEX and 
            OptimizationType.AGGREGATION_OPT not in metrics.optimizations_applied):
            recommendations.append(
                "Requête complexe avec agrégations. "
                "L'optimisation des agrégations pourrait améliorer les performances."
            )
        
        return recommendations
    
    def _detect_performance_anomalies(
        self, 
        metrics: PerformanceMetrics
    ) -> List[str]:
        """Détecte les anomalies de performance"""
        
        anomalies = []
        
        # Temps d'exécution anormalement élevé
        if metrics.execution_time_ms > self.config.max_query_time_ms:
            anomalies.append(
                f"Temps d'exécution critique: {metrics.execution_time_ms:.0f}ms "
                f"(seuil: {self.config.max_query_time_ms}ms)"
            )
        
        # Ratio Elasticsearch/traitement déséquilibré
        es_ratio = metrics.elasticsearch_time_ms / metrics.execution_time_ms
        if es_ratio < 0.3:  # Trop de temps en post-traitement
            anomalies.append(
                "Temps de post-traitement élevé par rapport à Elasticsearch. "
                "Optimisez le traitement des résultats."
            )
        elif es_ratio > 0.9:  # Elasticsearch trop lent
            anomalies.append(
                "Elasticsearch prend la majorité du temps d'exécution. "
                "Optimisez la requête ou vérifiez l'état du cluster."
            )
        
        return anomalies
    
    def _analyze_resource_usage(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Analyse l'usage des ressources"""
        
        return {
            "memory_usage": {
                "current_mb": metrics.memory_usage_mb,
                "threshold_mb": self.config.memory_usage_threshold_mb,
                "usage_percent": (metrics.memory_usage_mb / self.config.memory_usage_threshold_mb) * 100
            },
            "cpu_usage": {
                "current_percent": metrics.cpu_usage_percent,
                "status": "high" if metrics.cpu_usage_percent > 80 else "normal"
            },
            "network_io": {
                "bytes": metrics.network_io_bytes,
                "mb": metrics.network_io_bytes / (1024 * 1024)
            },
            "elasticsearch_shards": metrics.elasticsearch_shards
        }
    
    async def _update_optimization_rules(
        self, 
        metrics: PerformanceMetrics,
        analysis: Dict[str, Any]
    ):
        """Met à jour les règles d'optimisation adaptatives"""
        
        # Identifier les optimisations efficaces
        effectiveness = analysis.get("optimization_effectiveness", {})
        
        for optimization_type, result in effectiveness.items():
            improvement = result.get("improvement_percent", 0)
            
            # Mettre à jour les statistiques des règles
            for rule in self.optimization_rules:
                if rule.optimization_type.value == optimization_type:
                    if improvement > 5:  # Amélioration significative
                        rule.success_count += 1
                        rule.avg_improvement_percent = (
                            (rule.avg_improvement_percent * (rule.success_count - 1) + improvement) /
                            rule.success_count
                        )
                    else:
                        rule.failure_count += 1
                    break
    
    def _initialize_default_rules(self):
        """Initialise les règles d'optimisation par défaut"""
        
        default_rules = [
            OptimizationRule(
                name="Cache pour requêtes répétitives",
                condition="query_frequency > 2 and complexity in ['simple', 'moderate']",
                optimization_type=OptimizationType.CACHE_STRATEGY,
                parameters={"cache_ttl": 300},
                priority=1
            ),
            OptimizationRule(
                name="Limitation des champs pour grandes réponses",
                condition="expected_results > 100",
                optimization_type=OptimizationType.FIELD_SELECTION,
                parameters={"essential_fields_only": True},
                priority=2
            ),
            OptimizationRule(
                name="Timeout adaptatif pour requêtes complexes",
                condition="complexity in ['complex', 'very_complex']",
                optimization_type=OptimizationType.TIMEOUT_TUNING,
                parameters={"timeout_multiplier": 2.0},
                priority=3
            ),
            OptimizationRule(
                name="Routage par utilisateur",
                condition="user_id is not None",
                optimization_type=OptimizationType.SHARD_ROUTING,
                parameters={"routing_field": "user_id"},
                priority=4
            )
        ]
        
        self.optimization_rules.extend(default_rules)
    
    def _start_background_tasks(self):
        """Démarre les tâches de maintenance en arrière-plan"""
        
        async def cleanup_old_metrics():
            """Nettoie les anciennes métriques"""
            while True:
                try:
                    cutoff_time = datetime.now() - timedelta(
                        hours=self.config.metrics_retention_hours
                    )
                    
                    # Filtrer les entrées trop anciennes
                    self.metrics_history = deque([
                        entry for entry in self.metrics_history
                        if entry.timestamp > cutoff_time
                    ], maxlen=10000)
                    
                    await asyncio.sleep(3600)  # Nettoyer toutes les heures
                    
                except Exception as e:
                    logger.error(f"Erreur lors du nettoyage des métriques: {e}")
                    await asyncio.sleep(3600)
        
        # Démarrer la tâche de nettoyage
        asyncio.create_task(cleanup_old_metrics())
    
    def _has_text_search(self, query_part: Dict[str, Any]) -> bool:
        """Vérifie si la requête contient une recherche textuelle"""
        if isinstance(query_part, dict):
            for key, value in query_part.items():
                if key in ["match", "multi_match", "query_string", "simple_query_string"]:
                    return True
                elif isinstance(value, (dict, list)):
                    if self._has_text_search(value):
                        return True
        elif isinstance(query_part, list):
            for item in query_part:
                if self._has_text_search(item):
                    return True
        return False
    
    def _count_filters(self, query_part: Dict[str, Any]) -> int:
        """Compte le nombre de filtres dans la requête"""
        count = 0
        if isinstance(query_part, dict):
            if "filter" in query_part and isinstance(query_part["filter"], list):
                count += len(query_part["filter"])
            elif "bool" in query_part:
                bool_query = query_part["bool"]
                for clause in ["must", "filter", "should", "must_not"]:
                    if clause in bool_query and isinstance(bool_query[clause], list):
                        count += len(bool_query[clause])
        return count
    
    def _has_complex_boolean(self, query_part: Dict[str, Any]) -> bool:
        """Vérifie si la requête contient des structures booléennes complexes"""
        if isinstance(query_part, dict):
            if "bool" in query_part:
                bool_query = query_part["bool"]
                clause_count = sum(
                    1 for clause in ["must", "filter", "should", "must_not"]
                    if clause in bool_query and bool_query[clause]
                )
                return clause_count > 2
        return False
    
    def _analyze_aggregation_complexity(self, aggs: Dict[str, Any]) -> int:
        """Analyse la complexité des agrégations"""
        complexity = 0
        
        def count_agg_levels(agg_dict, level=0):
            nonlocal complexity
            complexity += level + 1
            
            for agg_name, agg_config in agg_dict.items():
                if "aggs" in agg_config:
                    count_agg_levels(agg_config["aggs"], level + 1)
        
        count_agg_levels(aggs)
        return min(complexity, 10)  # Limiter le score
    
    def _get_baseline_metrics(self, optimization_type: OptimizationType) -> Optional[Dict[str, Any]]:
        """Récupère les métriques de base pour un type d'optimisation"""
        
        # Filtrer les entrées sans cette optimisation
        baseline_entries = [
            entry for entry in self.metrics_history
            if optimization_type not in entry.optimizations
            and entry.timestamp > datetime.now() - timedelta(hours=1)
        ]
        
        if len(baseline_entries) < 3:
            return None
        
        execution_times = [entry.metrics.execution_time_ms for entry in baseline_entries]
        
        return {
            "avg_time_ms": statistics.mean(execution_times),
            "median_time_ms": statistics.median(execution_times),
            "sample_size": len(baseline_entries)
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Retourne un résumé complet des performances"""
        
        recent_entries = [
            entry for entry in self.metrics_history
            if entry.timestamp > datetime.now() - timedelta(hours=1)
        ]
        
        if not recent_entries:
            return {"status": "no_recent_data"}
        
        execution_times = [entry.metrics.execution_time_ms for entry in recent_entries]
        cache_hits = sum(1 for entry in recent_entries if entry.metrics.cache_hit)
        
        return {
            "total_queries": len(recent_entries),
            "avg_execution_time_ms": statistics.mean(execution_times),
            "median_execution_time_ms": statistics.median(execution_times),
            "p95_execution_time_ms": statistics.quantiles(execution_times, n=20)[18] if len(execution_times) > 10 else max(execution_times),
            "cache_hit_rate": (cache_hits / len(recent_entries)) * 100,
            "slow_queries": sum(1 for t in execution_times if t > self.config.slow_query_threshold_ms),
            "complexity_distribution": self._get_complexity_distribution(recent_entries),
            "optimization_usage": self._get_optimization_usage_stats(recent_entries)
        }
    
    def _get_complexity_distribution(self, entries: List[PerformanceHistoryEntry]) -> Dict[str, int]:
        """Analyse la distribution de complexité des requêtes"""
        
        distribution = defaultdict(int)
        for entry in entries:
            distribution[entry.metrics.query_complexity.value] += 1
        
        return dict(distribution)
    
    def _get_optimization_usage_stats(self, entries: List[PerformanceHistoryEntry]) -> Dict[str, Any]:
        """Statistiques d'usage des optimisations"""
        
        optimization_counts = defaultdict(int)
        total_queries = len(entries)
        
        for entry in entries:
            for optimization in entry.optimizations:
                optimization_counts[optimization.value] += 1
        
        return {
            optimization: {
                "count": count,
                "usage_percent": (count / total_queries) * 100
            }
            for optimization, count in optimization_counts.items()
        }


class CacheOptimizer:
    """Optimiseur spécialisé pour les stratégies de cache"""
    
    def __init__(self):
        self.cache_stats = defaultdict(int)
    
    async def optimize_for_cache(
        self,
        query_body: Dict[str, Any],
        search_request: SearchServiceQuery
    ) -> Dict[str, Any]:
        """Optimise une requête pour maximiser l'efficacité du cache"""
        
        optimized = query_body.copy()
        
        # Activer le cache de requête Elasticsearch
        optimized["request_cache"] = True
        
        # Normaliser les paramètres pour améliorer les cache hits
        if "preference" not in optimized:
            optimized["preference"] = "_local"
        
        # Arrondir les timestamps pour regrouper les requêtes similaires
        if "query" in optimized and "bool" in optimized["query"]:
            self._normalize_date_filters(optimized["query"]["bool"])
        
        self.cache_stats["optimizations_applied"] += 1
        return optimized
    
    def _normalize_date_filters(self, bool_query: Dict[str, Any]):
        """Normalise les filtres de date pour améliorer le cache"""
        
        for clause in ["filter", "must"]:
            if clause in bool_query and isinstance(bool_query[clause], list):
                for filter_item in bool_query[clause]:
                    if "range" in filter_item:
                        for field, range_config in filter_item["range"].items():
                            if field == "date" and isinstance(range_config, dict):
                                # Arrondir les dates à l'heure la plus proche
                                for date_key in ["gte", "lte", "gt", "lt"]:
                                    if date_key in range_config:
                                        date_value = range_config[date_key]
                                        if isinstance(date_value, str):
                                            try:
                                                dt = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                                                rounded_dt = dt.replace(minute=0, second=0, microsecond=0)
                                                range_config[date_key] = rounded_dt.isoformat()
                                            except ValueError:
                                                pass  # Garder la valeur originale


class QueryRewriter:
    """Réécrivain de requêtes pour optimiser les performances"""
    
    def __init__(self):
        self.rewrite_patterns = self._initialize_rewrite_patterns()
    
    async def rewrite_for_performance(
        self,
        query_body: Dict[str, Any],
        complexity: QueryComplexity,
        performance_profile: PerformanceProfile
    ) -> Dict[str, Any]:
        """Réécrit une requête pour améliorer les performances"""
        
        rewritten = query_body.copy()
        
        # Simplification pour les profils orientés latence
        if performance_profile == PerformanceProfile.LATENCY_FIRST:
            rewritten = self._simplify_for_latency(rewritten)
        
        # Optimisation pour le débit
        elif performance_profile == PerformanceProfile.THROUGHPUT_FIRST:
            rewritten = self._optimize_for_throughput(rewritten)
        
        # Réduction des coûts
        elif performance_profile == PerformanceProfile.COST_OPTIMIZED:
            rewritten = self._optimize_for_cost(rewritten)
        
        return rewritten
    
    def _simplify_for_latency(self, query_body: Dict[str, Any]) -> Dict[str, Any]:
        """Simplifie la requête pour réduire la latence"""
        
        simplified = query_body.copy()
        
        # Limiter la taille des résultats
        if simplified.get("size", 10) > 50:
            simplified["size"] = 50
        
        # Simplifier les agrégations
        if "aggs" in simplified:
            simplified["aggs"] = self._simplify_aggregations(simplified["aggs"])
        
        # Utiliser constant_score pour les filtres simples
        if "query" in simplified and "bool" in simplified["query"]:
            bool_query = simplified["query"]["bool"]
            if (not bool_query.get("must") and not bool_query.get("should") and 
                bool_query.get("filter")):
                simplified["query"] = {
                    "constant_score": {
                        "filter": {"bool": {"filter": bool_query["filter"]}},
                        "boost": 1.0
                    }
                }
        
        return simplified
    
    def _optimize_for_throughput(self, query_body: Dict[str, Any]) -> Dict[str, Any]:
        """Optimise pour maximiser le débit"""
        
        optimized = query_body.copy()
        
        # Désactiver le scoring pour les requêtes de filtrage
        if self._is_filter_only_query(optimized):
            optimized["query"] = {
                "constant_score": {
                    "filter": optimized["query"],
                    "boost": 1.0
                }
            }
        
        # Réduire la précision des agrégations
        if "aggs" in optimized:
            self._reduce_aggregation_precision(optimized["aggs"])
        
        return optimized
    
    def _optimize_for_cost(self, query_body: Dict[str, Any]) -> Dict[str, Any]:
        """Optimise pour réduire les coûts de calcul"""
        
        cost_optimized = query_body.copy()
        
        # Limiter track_total_hits
        cost_optimized["track_total_hits"] = min(
            cost_optimized.get("track_total_hits", True), 1000
        )
        
        # Réduire les champs retournés
        if "_source" not in cost_optimized:
            cost_optimized["_source"] = [
                "transaction_id", "user_id", "amount", 
                "date", "primary_description"
            ]
        
        return cost_optimized
    
    def _simplify_aggregations(self, aggs: Dict[str, Any]) -> Dict[str, Any]:
        """Simplifie les agrégations pour la performance"""
        
        simplified = {}
        
        for agg_name, agg_config in aggs.items():
            simplified_agg = agg_config.copy()
            
            # Limiter la taille des buckets terms
            if "terms" in simplified_agg:
                simplified_agg["terms"]["size"] = min(
                    simplified_agg["terms"].get("size", 10), 10
                )
            
            # Simplifier les sous-agrégations
            if "aggs" in simplified_agg:
                simplified_agg["aggs"] = self._simplify_aggregations(simplified_agg["aggs"])
            
            simplified[agg_name] = simplified_agg
        
        return simplified
    
    def _is_filter_only_query(self, query_body: Dict[str, Any]) -> bool:
        """Vérifie si la requête ne fait que du filtrage"""
        
        if "query" not in query_body:
            return True
        
        query_part = query_body["query"]
        
        # Vérifier s'il n'y a que des filtres
        if "bool" in query_part:
            bool_query = query_part["bool"]
            has_scoring = bool_query.get("must") or bool_query.get("should")
            has_filters = bool_query.get("filter") or bool_query.get("must_not")
            return not has_scoring and has_filters
        
        return False
    
    def _reduce_aggregation_precision(self, aggs: Dict[str, Any]):
        """Réduit la précision des agrégations pour le débit"""
        
        for agg_name, agg_config in aggs.items():
            # Réduire precision pour cardinality
            if "cardinality" in agg_config:
                agg_config["cardinality"]["precision_threshold"] = 1000
            
            # Simplifier les histogrammes
            elif "histogram" in agg_config:
                current_interval = agg_config["histogram"].get("interval", 1)
                agg_config["histogram"]["interval"] = max(current_interval, 10)
            
            # Traiter les sous-agrégations
            if "aggs" in agg_config:
                self._reduce_aggregation_precision(agg_config["aggs"])
    
    def _initialize_rewrite_patterns(self) -> List[Dict[str, Any]]:
        """Initialise les patterns de réécriture"""
        
        return [
            {
                "name": "term_to_constant_score",
                "condition": lambda q: self._has_only_term_filters(q),
                "transformation": self._convert_to_constant_score
            },
            {
                "name": "multi_match_simplification",
                "condition": lambda q: self._has_complex_multi_match(q),
                "transformation": self._simplify_multi_match
            }
        ]
    
    def _has_only_term_filters(self, query_body: Dict[str, Any]) -> bool:
        """Vérifie si la requête n'a que des filtres term"""
        # Implémentation simplifiée
        return False
    
    def _has_complex_multi_match(self, query_body: Dict[str, Any]) -> bool:
        """Vérifie si la requête a des multi_match complexes"""
        # Implémentation simplifiée
        return False
    
    def _convert_to_constant_score(self, query_body: Dict[str, Any]) -> Dict[str, Any]:
        """Convertit en constant_score"""
        return query_body
    
    def _simplify_multi_match(self, query_body: Dict[str, Any]) -> Dict[str, Any]:
        """Simplifie les requêtes multi_match"""
        return query_body


class TimeoutManager:
    """Gestionnaire intelligent des timeouts"""
    
    def __init__(self):
        self.timeout_stats = defaultdict(list)
        self.base_timeouts = {
            QueryComplexity.SIMPLE: 100,
            QueryComplexity.MODERATE: 300,
            QueryComplexity.COMPLEX: 800,
            QueryComplexity.VERY_COMPLEX: 2000
        }
    
    def calculate_optimal_timeout(
        self,
        complexity: QueryComplexity,
        performance_profile: PerformanceProfile
    ) -> int:
        """Calcule le timeout optimal pour une requête"""
        
        base_timeout = self.base_timeouts[complexity]
        
        # Ajustement selon le profil de performance
        if performance_profile == PerformanceProfile.LATENCY_FIRST:
            multiplier = 0.7
        elif performance_profile == PerformanceProfile.PRECISION_FIRST:
            multiplier = 2.0
        elif performance_profile == PerformanceProfile.THROUGHPUT_FIRST:
            multiplier = 0.8
        else:  # COST_OPTIMIZED ou défaut
            multiplier = 1.0
        
        # Ajustement adaptatif basé sur l'historique
        historical_avg = self._get_historical_average(complexity)
        if historical_avg:
            adaptive_multiplier = max(0.5, min(2.0, historical_avg / base_timeout))
            multiplier *= adaptive_multiplier
        
        return int(base_timeout * multiplier)
    
    def _get_historical_average(self, complexity: QueryComplexity) -> Optional[float]:
        """Récupère la moyenne historique pour une complexité donnée"""
        
        if complexity.value in self.timeout_stats:
            recent_times = self.timeout_stats[complexity.value][-10:]  # 10 dernières
            if recent_times:
                return statistics.mean(recent_times)
        
        return None
    
    def record_execution_time(self, complexity: QueryComplexity, execution_time_ms: float):
        """Enregistre un temps d'exécution pour ajustement adaptatif"""
        
        self.timeout_stats[complexity.value].append(execution_time_ms)
        
        # Garder seulement les 50 dernières mesures
        if len(self.timeout_stats[complexity.value]) > 50:
            self.timeout_stats[complexity.value] = self.timeout_stats[complexity.value][-50:]


class BatchProcessor:
    """Processeur de requêtes par batch pour optimiser le débit"""
    
    def __init__(self):
        self.batch_queue = asyncio.Queue()
        self.batch_size = 10
        self.batch_timeout_ms = 100
        self.processing = False
    
    async def process_batch_query(
        self,
        queries: List[Tuple[Dict[str, Any], SearchServiceQuery]]
    ) -> List[Dict[str, Any]]:
        """Traite un batch de requêtes de manière optimisée"""
        
        if not queries:
            return []
        
        # Grouper les requêtes similaires
        grouped_queries = self._group_similar_queries(queries)
        
        results = []
        for group in grouped_queries:
            # Optimiser le groupe comme une seule requête avec multi-search
            if len(group) > 1:
                batch_results = await self._execute_multi_search(group)
                results.extend(batch_results)
            else:
                # Traitement individuel pour les requêtes uniques
                single_result = await self._execute_single_query(group[0])
                results.append(single_result)
        
        return results
    
    def _group_similar_queries(
        self,
        queries: List[Tuple[Dict[str, Any], SearchServiceQuery]]
    ) -> List[List[Tuple[Dict[str, Any], SearchServiceQuery]]]:
        """Groupe les requêtes similaires pour optimisation batch"""
        
        groups = defaultdict(list)
        
        for query_body, search_request in queries:
            # Créer une clé de groupement basée sur la structure
            group_key = self._generate_group_key(query_body, search_request)
            groups[group_key].append((query_body, search_request))
        
        return list(groups.values())
    
    def _generate_group_key(
        self,
        query_body: Dict[str, Any],
        search_request: SearchServiceQuery
    ) -> str:
        """Génère une clé de groupement pour les requêtes similaires"""
        
        key_parts = [
            f"intent:{search_request.query_metadata.intent_type}",
            f"user:{search_request.query_metadata.user_id}",
            f"has_aggs:{'aggs' in query_body}",
            f"size:{query_body.get('size', 10)}"
        ]
        
        return "|".join(key_parts)
    
    async def _execute_multi_search(
        self,
        query_group: List[Tuple[Dict[str, Any], SearchServiceQuery]]
    ) -> List[Dict[str, Any]]:
        """Exécute un groupe de requêtes via multi-search"""
        
        # Préparer le multi-search body
        msearch_body = []
        for query_body, search_request in query_group:
            # Header pour chaque requête
            msearch_body.append({"index": settings.ELASTICSEARCH_INDEX})
            # Corps de la requête
            msearch_body.append(query_body)
        
        # Ici on simule l'exécution - dans l'implémentation réelle,
        # on utiliserait le client Elasticsearch
        results = []
        for _ in query_group:
            results.append({"hits": {"hits": [], "total": {"value": 0}}})
        
        return results
    
    async def _execute_single_query(
        self,
        query_tuple: Tuple[Dict[str, Any], SearchServiceQuery]
    ) -> Dict[str, Any]:
        """Exécute une requête unique"""
        
        query_body, search_request = query_tuple
        
        # Simulation d'exécution
        return {"hits": {"hits": [], "total": {"value": 0}}}


# === EXPORTS ===

__all__ = [
    "PerformanceOptimizer",
    "PerformanceMetrics",
    "OptimizationLevel",
    "PerformanceProfile",
    "QueryComplexity",
    "OptimizationType",
    "PerformanceConfiguration",
    "CacheOptimizer",
    "QueryRewriter", 
    "TimeoutManager",
    "BatchProcessor"
]