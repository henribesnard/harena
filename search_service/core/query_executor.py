"""
Exécuteur de requêtes Elasticsearch haute performance
Responsable de la construction et exécution optimisée des requêtes
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import time

from models.requests import InternalSearchRequest, RequestValidator
from models.responses import InternalSearchResponse, ExecutionMetrics, OptimizationType
from models.elasticsearch_queries import ESSearchQuery, optimize_es_query, extract_query_metadata
from models.service_contracts import SearchServiceQuery, SearchServiceResponse
from clients.elasticsearch_client import ElasticsearchClient
from templates import (
    template_manager, build_query_from_intent, process_financial_query,
    before_query_execution, after_query_execution
)
from utils.cache import LRUCache
from utils.metrics import QueryMetrics
from utils.elasticsearch_helpers import ElasticsearchQueryBuilder
from config import settings


logger = logging.getLogger(__name__)


class ExecutionStrategy(str, Enum):
    """Stratégies d'exécution des requêtes"""
    SINGLE = "single"                    # Exécution simple
    PARALLEL = "parallel"                # Exécution parallèle multiple
    SEQUENTIAL = "sequential"            # Exécution séquentielle
    BATCH = "batch"                      # Exécution en lot
    STREAMING = "streaming"              # Exécution avec streaming


class QueryComplexity(str, Enum):
    """Niveaux de complexité des requêtes"""
    SIMPLE = "simple"                    # <50ms - filtres simples
    MODERATE = "moderate"                # 50-200ms - recherche textuelle
    COMPLEX = "complex"                  # 200-500ms - agrégations
    HEAVY = "heavy"                      # >500ms - requêtes lourdes


class ExecutionPriority(str, Enum):
    """Priorités d'exécution"""
    LOW = "low"
    NORMAL = "normal" 
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ExecutionContext:
    """Contexte d'exécution d'une requête"""
    request_id: str
    user_id: int
    strategy: ExecutionStrategy = ExecutionStrategy.SINGLE
    priority: ExecutionPriority = ExecutionPriority.NORMAL
    timeout_ms: int = 5000
    retry_count: int = 0
    max_retries: int = 2
    cache_enabled: bool = True
    debug_mode: bool = False
    agent_context: Dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    
    def get_elapsed_ms(self) -> int:
        """Retourne le temps écoulé en millisecondes"""
        return int((datetime.now() - self.start_time).total_seconds() * 1000)
    
    def should_retry(self) -> bool:
        """Détermine si un retry est possible"""
        return self.retry_count < self.max_retries
    
    def increment_retry(self):
        """Incrémente le compteur de retry"""
        self.retry_count += 1


@dataclass 
class ExecutionResult:
    """Résultat d'exécution avec métadonnées complètes"""
    success: bool
    response: Optional[InternalSearchResponse] = None
    error: Optional[Exception] = None
    execution_time_ms: int = 0
    strategy_used: ExecutionStrategy = ExecutionStrategy.SINGLE
    cache_hit: bool = False
    retries_attempted: int = 0
    elasticsearch_took: int = 0
    optimizations_applied: List[OptimizationType] = field(default_factory=list)
    debug_info: Optional[Dict[str, Any]] = None


class QueryExecutionEngine:
    """Moteur d'exécution de requêtes avec optimisations avancées"""
    
    def __init__(self, elasticsearch_client: ElasticsearchClient):
        self.es_client = elasticsearch_client
        self.query_builder = ElasticsearchQueryBuilder()
        self.cache = LRUCache(max_size=settings.QUERY_CACHE_SIZE)
        self.metrics = QueryMetrics()
        self.active_executions: Dict[str, ExecutionContext] = {}
        
        # Configuration d'exécution
        self.max_concurrent_queries = settings.MAX_CONCURRENT_QUERIES
        self.default_timeout_ms = settings.DEFAULT_QUERY_TIMEOUT_MS
        self.semaphore = asyncio.Semaphore(self.max_concurrent_queries)
        
        logger.info(f"QueryExecutionEngine initialisé - Max concurrent: {self.max_concurrent_queries}")
    
    async def execute_query(self, 
                           request: InternalSearchRequest,
                           context: Optional[ExecutionContext] = None) -> ExecutionResult:
        """
        Exécute une requête avec gestion complète des erreurs et optimisations
        """
        # Créer le contexte si non fourni
        if not context:
            context = ExecutionContext(
                request_id=request.request_id,
                user_id=request.user_id,
                timeout_ms=request.timeout_ms or self.default_timeout_ms
            )
        
        # Enregistrer l'exécution active
        self.active_executions[context.request_id] = context
        
        try:
            # Validation préalable
            if not self._validate_execution_request(request, context):
                return ExecutionResult(
                    success=False,
                    error=ValueError("Requête invalide"),
                    execution_time_ms=context.get_elapsed_ms()
                )
            
            # Hooks avant exécution
            await self._before_execution_hooks(request, context)
            
            # Sélection de la stratégie d'exécution
            strategy = self._determine_execution_strategy(request, context)
            context.strategy = strategy
            
            # Exécution selon la stratégie
            if strategy == ExecutionStrategy.SINGLE:
                result = await self._execute_single_query(request, context)
            elif strategy == ExecutionStrategy.PARALLEL:
                result = await self._execute_parallel_queries(request, context)
            elif strategy == ExecutionStrategy.BATCH:
                result = await self._execute_batch_query(request, context)
            else:
                result = await self._execute_single_query(request, context)  # Fallback
            
            # Hooks après exécution
            await self._after_execution_hooks(request, context, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution de la requête {context.request_id}: {str(e)}")
            return ExecutionResult(
                success=False,
                error=e,
                execution_time_ms=context.get_elapsed_ms(),
                strategy_used=context.strategy,
                retries_attempted=context.retry_count
            )
        finally:
            # Nettoyer l'exécution active
            self.active_executions.pop(context.request_id, None)
    
    async def _execute_single_query(self, 
                                   request: InternalSearchRequest,
                                   context: ExecutionContext) -> ExecutionResult:
        """Exécute une seule requête avec optimisations"""
        
        # Vérifier le cache d'abord
        if context.cache_enabled:
            cache_key = self._generate_cache_key(request)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.debug(f"Cache hit pour {context.request_id}")
                return ExecutionResult(
                    success=True,
                    response=cached_result,
                    execution_time_ms=1,  # Cache quasi-instantané
                    cache_hit=True,
                    strategy_used=ExecutionStrategy.SINGLE
                )
        
        # Construire la requête Elasticsearch
        try:
            es_query_dict = self._build_elasticsearch_query(request, context)
        except Exception as e:
            logger.error(f"Erreur construction requête ES: {str(e)}")
            return ExecutionResult(
                success=False,
                error=e,
                execution_time_ms=context.get_elapsed_ms()
            )
        
        # Limiter la concurrence
        async with self.semaphore:
            return await self._execute_with_retry(request, context, es_query_dict)
    
    async def _execute_with_retry(self,
                                 request: InternalSearchRequest,
                                 context: ExecutionContext,
                                 es_query_dict: Dict[str, Any]) -> ExecutionResult:
        """Exécute avec logique de retry"""
        
        last_error = None
        
        while True:
            try:
                start_time = time.time()
                
                # Exécution Elasticsearch
                es_response = await self.es_client.search(
                    index=settings.ELASTICSEARCH_INDEX,
                    body=es_query_dict,
                    timeout=f"{context.timeout_ms}ms"
                )
                
                execution_time_ms = int((time.time() - start_time) * 1000)
                
                # Traitement de la réponse
                internal_response = self._process_elasticsearch_response(
                    es_response, request, context, execution_time_ms
                )
                
                # Mise en cache si approprié
                if context.cache_enabled and self._should_cache_result(internal_response):
                    cache_key = self._generate_cache_key(request)
                    self.cache.set(cache_key, internal_response, ttl=settings.CACHE_TTL_SECONDS)
                
                # Enregistrement des métriques
                self.metrics.record_query_execution(
                    request.query_type.value,
                    execution_time_ms,
                    len(internal_response.raw_results),
                    context.cache_enabled
                )
                
                return ExecutionResult(
                    success=True,
                    response=internal_response,
                    execution_time_ms=execution_time_ms,
                    strategy_used=context.strategy,
                    cache_hit=False,
                    retries_attempted=context.retry_count,
                    elasticsearch_took=es_response.get("took", 0),
                    optimizations_applied=self.query_builder.get_applied_optimizations(),
                    debug_info={"es_query": es_query_dict} if context.debug_mode else None
                )
                
            except Exception as e:
                last_error = e
                logger.warning(f"Tentative {context.retry_count + 1} échouée pour {context.request_id}: {str(e)}")
                
                if context.should_retry():
                    context.increment_retry()
                    # Attendre avant retry (backoff exponentiel)
                    await asyncio.sleep(0.1 * (2 ** context.retry_count))
                    continue
                else:
                    break
        
        # Échec après tous les retries
        return ExecutionResult(
            success=False,
            error=last_error,
            execution_time_ms=context.get_elapsed_ms(),
            strategy_used=context.strategy,
            retries_attempted=context.retry_count
        )
    
    async def _execute_parallel_queries(self,
                                       request: InternalSearchRequest,
                                       context: ExecutionContext) -> ExecutionResult:
        """Exécute plusieurs requêtes en parallèle (pour agrégations complexes)"""
        
        # Décomposer la requête en sous-requêtes
        sub_requests = self._decompose_into_parallel_requests(request)
        
        if len(sub_requests) <= 1:
            # Fallback vers exécution simple
            return await self._execute_single_query(request, context)
        
        logger.info(f"Exécution parallèle de {len(sub_requests)} sous-requêtes")
        
        # Exécuter toutes les sous-requêtes en parallèle
        start_time = time.time()
        tasks = []
        
        for i, sub_request in enumerate(sub_requests):
            sub_context = ExecutionContext(
                request_id=f"{context.request_id}_sub_{i}",
                user_id=context.user_id,
                strategy=ExecutionStrategy.SINGLE,
                timeout_ms=context.timeout_ms // len(sub_requests),  # Répartir le timeout
                cache_enabled=context.cache_enabled
            )
            task = self._execute_single_query(sub_request, sub_context)
            tasks.append(task)
        
        # Attendre toutes les tâches
        try:
            sub_results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=e,
                execution_time_ms=context.get_elapsed_ms(),
                strategy_used=ExecutionStrategy.PARALLEL
            )
        
        execution_time_ms = int((time.time() - start_time) * 1000)
        
        # Combiner les résultats
        combined_response = self._combine_parallel_results(sub_results, request, context)
        
        return ExecutionResult(
            success=True,
            response=combined_response,
            execution_time_ms=execution_time_ms,
            strategy_used=ExecutionStrategy.PARALLEL,
            retries_attempted=context.retry_count
        )
    
    async def _execute_batch_query(self,
                                  request: InternalSearchRequest,
                                  context: ExecutionContext) -> ExecutionResult:
        """Exécute une requête en mode batch (pour gros volumes)"""
        
        # Pour l'instant, fallback vers exécution simple
        # TODO: Implémenter scroll API pour gros datasets
        logger.warning("Batch execution pas encore implémenté, fallback vers single")
        return await self._execute_single_query(request, context)
    
    def _build_elasticsearch_query(self,
                                  request: InternalSearchRequest,
                                  context: ExecutionContext) -> Dict[str, Any]:
        """Construit la requête Elasticsearch optimisée"""
        
        # Utiliser le builder d'helpers
        es_query_dict = self.query_builder.build_query(request)
        
        # Optimisations supplémentaires
        es_query_dict = optimize_es_query(es_query_dict)
        
        # Ajouter métadonnées pour debugging
        if context.debug_mode:
            es_query_dict["_debug"] = {
                "request_id": context.request_id,
                "user_id": context.user_id,
                "strategy": context.strategy.value,
                "priority": context.priority.value
            }
        
        logger.debug(f"Requête ES construite pour {context.request_id}")
        return es_query_dict
    
    def _process_elasticsearch_response(self,
                                       es_response: Dict[str, Any],
                                       request: InternalSearchRequest,
                                       context: ExecutionContext,
                                       execution_time_ms: int) -> InternalSearchResponse:
        """Traite la réponse Elasticsearch en réponse interne"""
        
        # Extraire les métadonnées de base
        total_hits = es_response.get("hits", {}).get("total", {})
        if isinstance(total_hits, dict):
            total_count = total_hits.get("value", 0)
        else:
            total_count = total_hits
        
        # Extraire les documents
        raw_results = []
        for hit in es_response.get("hits", {}).get("hits", []):
            document = hit["_source"]
            document["_score"] = hit.get("_score", 0.0)
            document["_id"] = hit["_id"]
            
            # Ajouter highlighting si présent
            if "highlight" in hit:
                document["_highlights"] = hit["highlight"]
            
            raw_results.append(document)
        
        # Extraire les agrégations
        aggregations = []
        if "aggregations" in es_response:
            aggregations = self._extract_aggregations(es_response["aggregations"])
        
        # Calculer la qualité des résultats
        quality_score = self._calculate_result_quality(es_response, request)
        
        # Créer les métriques d'exécution
        execution_metrics = ExecutionMetrics(
            total_time_ms=execution_time_ms,
            elasticsearch_took=es_response.get("took", 0),
            query_complexity=self._determine_query_complexity(request),
            optimizations_applied=self.query_builder.get_applied_optimizations(),
            cache_used=False,  # Sera mis à jour si cache utilisé
            parallel_executions=1
        )
        
        # Créer la réponse interne
        internal_response = InternalSearchResponse(
            request_id=context.request_id,
            user_id=context.user_id,
            total_hits=total_count,
            returned_hits=len(raw_results),
            raw_results=raw_results,
            aggregations=aggregations,
            execution_metrics=execution_metrics,
            quality_score=quality_score,
            elasticsearch_response=es_response if context.debug_mode else None,
            served_from_cache=False
        )
        
        return internal_response
    
    def _extract_aggregations(self, es_aggregations: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extrait et normalise les agrégations Elasticsearch"""
        aggregations = []
        
        for agg_name, agg_data in es_aggregations.items():
            if "buckets" in agg_data:
                # Agrégation à buckets
                aggregations.append({
                    "name": agg_name,
                    "type": "terms",
                    "buckets": agg_data["buckets"]
                })
            elif "value" in agg_data:
                # Agrégation métrique simple
                aggregations.append({
                    "name": agg_name,
                    "type": "metric",
                    "value": agg_data["value"]
                })
            else:
                # Autres types d'agrégations
                aggregations.append({
                    "name": agg_name,
                    "type": "complex",
                    "data": agg_data
                })
        
        return aggregations
    
    def _calculate_result_quality(self,
                                 es_response: Dict[str, Any],
                                 request: InternalSearchRequest) -> float:
        """Calcule un score de qualité des résultats"""
        
        hits = es_response.get("hits", {}).get("hits", [])
        if not hits:
            return 0.0
        
        # Score basé sur la pertinence Elasticsearch
        total_score = sum(hit.get("_score", 0.0) for hit in hits)
        max_possible_score = hits[0].get("_score", 1.0) * len(hits)
        
        if max_possible_score == 0:
            return 0.5  # Score neutre pour requêtes sans scoring
        
        relevance_score = total_score / max_possible_score
        
        # Ajuster selon le type de requête
        if request.text_query:
            # Recherche textuelle: favoriser les scores élevés
            return min(1.0, relevance_score * 1.2)
        else:
            # Recherche par filtres: score plus uniforme
            return min(1.0, relevance_score + 0.3)
    
    def _determine_query_complexity(self, request: InternalSearchRequest) -> QueryComplexity:
        """Détermine la complexité d'une requête"""
        complexity_score = 0
        
        # Filtres (+1 point par filtre)
        complexity_score += len(request.term_filters)
        
        # Recherche textuelle (+3 points)
        if request.text_query:
            complexity_score += 3
            complexity_score += len(request.text_query.fields)
        
        # Agrégations (+5 points par type)
        complexity_score += len(request.aggregation_types) * 5
        
        # Pagination profonde (+2 points si offset > 1000)
        if request.offset > 1000:
            complexity_score += 2
        
        # Déterminer la complexité
        if complexity_score <= 3:
            return QueryComplexity.SIMPLE
        elif complexity_score <= 8:
            return QueryComplexity.MODERATE
        elif complexity_score <= 15:
            return QueryComplexity.COMPLEX
        else:
            return QueryComplexity.HEAVY
    
    def _determine_execution_strategy(self,
                                    request: InternalSearchRequest,
                                    context: ExecutionContext) -> ExecutionStrategy:
        """Détermine la stratégie d'exécution optimale"""
        
        # Forcer si spécifié dans le contexte
        if context.strategy != ExecutionStrategy.SINGLE:
            return context.strategy
        
        # Analyse automatique
        complexity = self._determine_query_complexity(request)
        
        # Requêtes simples: exécution simple
        if complexity == QueryComplexity.SIMPLE:
            return ExecutionStrategy.SINGLE
        
        # Requêtes avec plusieurs agrégations: parallélisation possible
        if len(request.aggregation_types) > 2:
            return ExecutionStrategy.PARALLEL
        
        # Requêtes très lourdes: batch
        if complexity == QueryComplexity.HEAVY:
            return ExecutionStrategy.BATCH
        
        # Par défaut: exécution simple
        return ExecutionStrategy.SINGLE
    
    def _decompose_into_parallel_requests(self,
                                        request: InternalSearchRequest) -> List[InternalSearchRequest]:
        """Décompose une requête en sous-requêtes parallèles"""
        
        # Stratégie simple: séparer les agrégations
        if len(request.aggregation_types) <= 1:
            return [request]
        
        sub_requests = []
        
        # Requête principale sans agrégations (pour les hits)
        main_request = request.copy()
        main_request.aggregation_types = []
        main_request.aggregation_fields = []
        sub_requests.append(main_request)
        
        # Une requête par type d'agrégation
        for agg_type in request.aggregation_types:
            agg_request = request.copy()
            agg_request.aggregation_types = [agg_type]
            agg_request.limit = 0  # Pas de hits pour les agrégations
            sub_requests.append(agg_request)
        
        return sub_requests
    
    def _combine_parallel_results(self,
                                 sub_results: List[ExecutionResult],
                                 original_request: InternalSearchRequest,
                                 context: ExecutionContext) -> InternalSearchResponse:
        """Combine les résultats de requêtes parallèles"""
        
        # Vérifier les succès
        successful_results = [r for r in sub_results if isinstance(r, ExecutionResult) and r.success]
        
        if not successful_results:
            # Créer une réponse vide en cas d'échec total
            return InternalSearchResponse(
                request_id=context.request_id,
                user_id=context.user_id,
                total_hits=0,
                returned_hits=0,
                raw_results=[],
                aggregations=[],
                execution_metrics=ExecutionMetrics(
                    total_time_ms=context.get_elapsed_ms(),
                    query_complexity=QueryComplexity.COMPLEX,
                    parallel_executions=len(sub_results)
                ),
                quality_score=0.0
            )
        
        # Combiner les résultats
        main_result = successful_results[0].response
        combined_aggregations = []
        
        # Collecter toutes les agrégations
        for result in successful_results:
            if result.response and result.response.aggregations:
                combined_aggregations.extend(result.response.aggregations)
        
        # Calculer les métriques combinées
        total_execution_time = max(r.execution_time_ms for r in successful_results)
        total_elasticsearch_time = sum(r.elasticsearch_took for r in successful_results)
        
        execution_metrics = ExecutionMetrics(
            total_time_ms=total_execution_time,
            elasticsearch_took=total_elasticsearch_time,
            query_complexity=QueryComplexity.COMPLEX,
            parallel_executions=len(successful_results)
        )
        
        # Créer la réponse combinée
        combined_response = InternalSearchResponse(
            request_id=context.request_id,
            user_id=context.user_id,
            total_hits=main_result.total_hits,
            returned_hits=main_result.returned_hits,
            raw_results=main_result.raw_results,
            aggregations=combined_aggregations,
            execution_metrics=execution_metrics,
            quality_score=main_result.quality_score
        )
        
        return combined_response
    
    def _generate_cache_key(self, request: InternalSearchRequest) -> str:
        """Génère une clé de cache unique pour la requête"""
        
        # Éléments stables pour le cache
        cache_elements = [
            f"user:{request.user_id}",
            f"type:{request.query_type.value}",
            f"limit:{request.limit}",
            f"offset:{request.offset}"
        ]
        
        # Ajouter les filtres
        for term_filter in request.term_filters:
            cache_elements.append(f"filter:{term_filter.field}:{term_filter.operator}:{term_filter.value}")
        
        # Ajouter la recherche textuelle
        if request.text_query:
            cache_elements.append(f"text:{request.text_query.query}")
            cache_elements.append(f"fields:{','.join(request.text_query.fields)}")
        
        # Ajouter les agrégations
        if request.aggregation_types:
            agg_str = ','.join(sorted([agg.value for agg in request.aggregation_types]))
            cache_elements.append(f"aggs:{agg_str}")
        
        return "|".join(cache_elements)
    
    def _should_cache_result(self, response: InternalSearchResponse) -> bool:
        """Détermine si un résultat doit être mis en cache"""
        
        # Ne pas cacher les requêtes en temps réel
        if response.execution_metrics.total_time_ms < 10:
            return False
        
        # Ne pas cacher les résultats vides
        if response.total_hits == 0:
            return False
        
        # Ne pas cacher les résultats de faible qualité
        if response.quality_score < 0.3:
            return False
        
        # Cacher par défaut
        return True
    
    def _validate_execution_request(self,
                                   request: InternalSearchRequest,
                                   context: ExecutionContext) -> bool:
        """Valide une demande d'exécution"""
        
        try:
            # Validation de base du request
            RequestValidator.validate_request(request)
            
            # Validation du contexte
            if context.timeout_ms <= 0 or context.timeout_ms > 30000:
                logger.error(f"Timeout invalide: {context.timeout_ms}")
                return False
            
            if not context.request_id:
                logger.error("request_id manquant dans le contexte")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation échouée: {str(e)}")
            return False
    
    async def _before_execution_hooks(self,
                                     request: InternalSearchRequest,
                                     context: ExecutionContext):
        """Hooks exécutés avant l'exécution"""
        
        # Hook de monitoring global
        try:
            # Note: before_query_execution attend un SearchServiceQuery, 
            # on pourrait adapter ou créer un hook spécifique
            logger.debug(f"Début d'exécution: {context.request_id}")
        except Exception as e:
            logger.warning(f"Erreur dans before_execution_hooks: {str(e)}")
    
    async def _after_execution_hooks(self,
                                    request: InternalSearchRequest,
                                    context: ExecutionContext,
                                    result: ExecutionResult):
        """Hooks exécutés après l'exécution"""
        
        try:
            # Enregistrement des métriques
            self.metrics.record_execution_result(
                request.query_type.value,
                result.execution_time_ms,
                result.success,
                result.cache_hit
            )
            
            logger.debug(f"Fin d'exécution: {context.request_id}, "
                        f"succès: {result.success}, temps: {result.execution_time_ms}ms")
            
        except Exception as e:
            logger.warning(f"Erreur dans after_execution_hooks: {str(e)}")
    
    # === MÉTHODES D'ADMINISTRATION ===
    
    def get_active_executions(self) -> Dict[str, Dict[str, Any]]:
        """Retourne les exécutions actives"""
        return {
            request_id: {
                "user_id": context.user_id,
                "strategy": context.strategy.value,
                "priority": context.priority.value,
                "elapsed_ms": context.get_elapsed_ms(),
                "retry_count": context.retry_count
            }
            for request_id, context in self.active_executions.items()
        }
    
    def get_execution_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques d'exécution"""
        return {
            "cache_stats": self.cache.get_stats(),
            "query_metrics": self.metrics.get_summary(),
            "active_executions_count": len(self.active_executions),
            "max_concurrent_queries": self.max_concurrent_queries,
            "default_timeout_ms": self.default_timeout_ms
        }
    
    def clear_cache(self):
        """Vide le cache de requêtes"""
        self.cache.clear()
        logger.info("Cache de requêtes vidé")
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérification de santé du moteur d'exécution"""
        
        try:
            # Test de connectivité Elasticsearch
            es_health = await self.es_client.cluster.health()
            es_status = es_health.get("status", "unknown")
            
            # Métriques de performance
            cache_stats = self.cache.get_stats()
            query_stats = self.metrics.get_summary()
            
            # Déterminer la santé globale
            is_healthy = (
                es_status in ["green", "yellow"] and
                len(self.active_executions) < self.max_concurrent_queries * 0.8 and
                cache_stats["hit_rate"] > 0.3
            )
            
            return {
                "status": "healthy" if is_healthy else "degraded",
                "elasticsearch": {
                    "status": es_status,
                    "cluster_name": es_health.get("cluster_name", "unknown")
                },
                "execution_engine": {
                    "active_executions": len(self.active_executions),
                    "max_concurrent": self.max_concurrent_queries,
                    "utilization_percent": len(self.active_executions) / self.max_concurrent_queries * 100
                },
                "cache": cache_stats,
                "performance": {
                    "total_queries": query_stats.get("total_executions", 0),
                    "average_time_ms": query_stats.get("average_execution_time", 0),
                    "success_rate": query_stats.get("success_rate", 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur lors du health check: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


class HighLevelQueryExecutor:
    """
    Exécuteur de requêtes de haut niveau avec intégration des templates
    Interface principale pour l'exécution de requêtes depuis les contrats
    """
    
    def __init__(self, elasticsearch_client: ElasticsearchClient):
        self.execution_engine = QueryExecutionEngine(elasticsearch_client)
        self.template_manager = template_manager
        
    async def execute_from_contract(self, 
                                   contract: SearchServiceQuery,
                                   debug_mode: bool = False) -> SearchServiceResponse:
        """
        Exécute une requête à partir d'un contrat SearchServiceQuery
        """
        try:
            # Hooks avant exécution
            before_query_execution(contract)
            
            start_time = time.time()
            
            # 1. Construire la requête Elasticsearch depuis le template
            try:
                es_query = process_financial_query(contract)
            except Exception as e:
                logger.error(f"Erreur construction template: {str(e)}")
                return self._create_error_response(contract, f"Template construction failed: {str(e)}")
            
            # 2. Convertir en requête interne
            from models.requests import RequestTransformer
            internal_request = RequestTransformer.from_contract(contract)
            
            # 3. Créer le contexte d'exécution
            context = ExecutionContext(
                request_id=contract.query_metadata.query_id,
                user_id=contract.query_metadata.user_id,
                timeout_ms=contract.search_parameters.timeout_ms,
                debug_mode=debug_mode,
                agent_context={
                    "agent_name": contract.query_metadata.agent_name,
                    "team_name": contract.query_metadata.team_name,
                    "intent_type": contract.query_metadata.intent_type
                }
            )
            
            # 4. Exécuter la requête
            execution_result = await self.execution_engine.execute_query(internal_request, context)
            
            # 5. Transformer en réponse de contrat
            if execution_result.success:
                from models.responses import ResponseTransformer
                service_response = ResponseTransformer.to_contract(execution_result.response)
                
                # Enrichir avec les métriques d'exécution
                service_response.response_metadata.execution_time_ms = execution_result.execution_time_ms
                service_response.response_metadata.cache_hit = execution_result.cache_hit
                service_response.performance.query_complexity = execution_result.response.execution_metrics.query_complexity.value
                
                total_time = int((time.time() - start_time) * 1000)
                
                # Hooks après exécution
                after_query_execution(contract, total_time, True)
                
                return service_response
            else:
                # Gestion d'erreur
                error_msg = str(execution_result.error) if execution_result.error else "Unknown execution error"
                total_time = int((time.time() - start_time) * 1000)
                
                after_query_execution(contract, total_time, False)
                
                return self._create_error_response(contract, error_msg)
                
        except Exception as e:
            logger.error(f"Erreur dans execute_from_contract: {str(e)}")
            total_time = int((time.time() - start_time) * 1000) if 'start_time' in locals() else 0
            
            if 'contract' in locals():
                after_query_execution(contract, total_time, False)
            
            return self._create_error_response(contract, str(e))
    
    async def execute_aggregation(self,
                                 intent: str,
                                 user_id: int,
                                 filters: Optional[Dict[str, Any]] = None,
                                 use_cache: bool = True) -> Dict[str, Any]:
        """
        Exécute une agrégation directement par intention
        """
        try:
            from templates.aggregation_templates import AggregationIntent, execute_financial_aggregation
            
            # Convertir l'intention
            agg_intent = AggregationIntent(intent)
            
            # Exécuter via le système d'agrégation
            return execute_financial_aggregation(agg_intent, user_id, filters, use_cache)
            
        except Exception as e:
            logger.error(f"Erreur dans execute_aggregation: {str(e)}")
            return {
                "error": str(e),
                "intent": intent,
                "user_id": user_id
            }
    
    def _create_error_response(self, 
                              contract: SearchServiceQuery, 
                              error_message: str) -> SearchServiceResponse:
        """Crée une réponse d'erreur standardisée"""
        
        from models.service_contracts import (
            SearchServiceResponse, ResponseMetadata, PerformanceMetrics,
            ContextEnrichment
        )
        
        return SearchServiceResponse(
            response_metadata=ResponseMetadata(
                query_id=contract.query_metadata.query_id,
                execution_time_ms=0,
                total_hits=0,
                returned_hits=0,
                has_more=False,
                cache_hit=False,
                elasticsearch_took=0,
                agent_context={
                    "requesting_agent": contract.query_metadata.agent_name,
                    "error_occurred": True,
                    "error_message": error_message
                }
            ),
            results=[],
            aggregations=None,
            performance=PerformanceMetrics(
                query_complexity="error",
                optimization_applied=["error_handling"],
                index_used="",
                shards_queried=0,
                cache_hit=False
            ),
            context_enrichment=ContextEnrichment(
                search_intent_matched=False,
                result_quality_score=0.0,
                suggested_followup_questions=[],
                next_suggested_agent="error_handler_agent"
            )
        )
    
    async def batch_execute(self, 
                           contracts: List[SearchServiceQuery],
                           max_parallel: int = 5) -> List[SearchServiceResponse]:
        """
        Exécute plusieurs contrats en parallèle avec limitation de concurrence
        """
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def execute_single(contract):
            async with semaphore:
                return await self.execute_from_contract(contract)
        
        tasks = [execute_single(contract) for contract in contracts]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Traiter les exceptions
        processed_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                error_response = self._create_error_response(
                    contracts[i], 
                    f"Batch execution error: {str(response)}"
                )
                processed_responses.append(error_response)
            else:
                processed_responses.append(response)
        
        return processed_responses
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques d'exécution détaillées"""
        
        engine_metrics = self.execution_engine.get_execution_metrics()
        template_metrics = self.template_manager.get_unified_performance_report()
        
        return {
            "execution_engine": engine_metrics,
            "template_system": template_metrics,
            "integration_health": {
                "template_engine_status": "active",
                "elasticsearch_client_status": "active",
                "cache_system_status": "active"
            }
        }
    
    async def optimize_for_user(self, user_id: int) -> Dict[str, Any]:
        """
        Optimise les performances pour un utilisateur spécifique
        """
        try:
            # Analyser l'historique des requêtes de l'utilisateur
            user_metrics = self.execution_engine.metrics.get_user_metrics(user_id)
            
            # Suggestions d'optimisation
            optimizations = []
            
            if user_metrics.get("average_execution_time", 0) > 500:
                optimizations.append("Consider using more specific filters")
            
            if user_metrics.get("cache_hit_rate", 0) < 0.3:
                optimizations.append("Enable aggressive caching for repeated queries")
            
            frequent_intentions = user_metrics.get("frequent_intentions", [])
            if frequent_intentions:
                optimizations.append(f"Pre-warm cache for intentions: {', '.join(frequent_intentions[:3])}")
            
            return {
                "user_id": user_id,
                "current_performance": user_metrics,
                "optimization_suggestions": optimizations,
                "recommended_settings": {
                    "cache_strategy": "aggressive" if user_metrics.get("cache_hit_rate", 0) < 0.5 else "standard",
                    "timeout_ms": min(10000, max(1000, user_metrics.get("average_execution_time", 3000) * 2))
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur dans optimize_for_user: {str(e)}")
            return {"error": str(e), "user_id": user_id}


class QueryExecutorManager:
    """
    Gestionnaire global pour l'exécution de requêtes
    Point d'entrée principal pour le module
    """
    
    def __init__(self):
        self._elasticsearch_client = None
        self._high_level_executor = None
        self._initialized = False
    
    def initialize(self, elasticsearch_client: ElasticsearchClient):
        """Initialise le gestionnaire avec un client Elasticsearch"""
        self._elasticsearch_client = elasticsearch_client
        self._high_level_executor = HighLevelQueryExecutor(elasticsearch_client)
        self._initialized = True
        
        logger.info("QueryExecutorManager initialisé avec succès")
    
    @property
    def executor(self) -> HighLevelQueryExecutor:
        """Accès à l'exécuteur de haut niveau"""
        if not self._initialized:
            raise RuntimeError("QueryExecutorManager non initialisé. Appelez initialize() d'abord.")
        return self._high_level_executor
    
    @property
    def engine(self) -> QueryExecutionEngine:
        """Accès au moteur d'exécution bas niveau"""
        if not self._initialized:
            raise RuntimeError("QueryExecutorManager non initialisé. Appelez initialize() d'abord.")
        return self._high_level_executor.execution_engine
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérification de santé globale"""
        if not self._initialized:
            return {
                "status": "not_initialized",
                "error": "QueryExecutorManager not initialized"
            }
        
        try:
            engine_health = await self.engine.health_check()
            executor_stats = self.executor.get_execution_statistics()
            
            return {
                "status": engine_health["status"],
                "components": {
                    "execution_engine": engine_health,
                    "high_level_executor": {
                        "status": "healthy",
                        "statistics": executor_stats
                    }
                },
                "initialized": self._initialized
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "initialized": self._initialized
            }
    
    def shutdown(self):
        """Arrêt propre du gestionnaire"""
        if self._initialized:
            try:
                # Attendre la fin des exécutions actives
                active_count = len(self.engine.active_executions)
                if active_count > 0:
                    logger.info(f"Attente de la fin de {active_count} exécutions actives...")
                    # En production, implémenter une attente réelle
                
                # Vider les caches
                self.engine.clear_cache()
                
                logger.info("QueryExecutorManager arrêté proprement")
                
            except Exception as e:
                logger.error(f"Erreur lors de l'arrêt: {str(e)}")
            finally:
                self._initialized = False
                self._high_level_executor = None
                self._elasticsearch_client = None


# === INSTANCE GLOBALE ===
query_executor_manager = QueryExecutorManager()


# === FONCTIONS D'UTILITÉ PRINCIPALES ===

async def execute_search_query(contract: SearchServiceQuery, 
                              debug_mode: bool = False) -> SearchServiceResponse:
    """
    Fonction principale pour exécuter une requête de recherche
    """
    return await query_executor_manager.executor.execute_from_contract(contract, debug_mode)


async def execute_aggregation_query(intent: str,
                                   user_id: int,
                                   filters: Optional[Dict[str, Any]] = None,
                                   use_cache: bool = True) -> Dict[str, Any]:
    """
    Fonction principale pour exécuter une agrégation
    """
    return await query_executor_manager.executor.execute_aggregation(intent, user_id, filters, use_cache)


async def batch_execute_queries(contracts: List[SearchServiceQuery],
                               max_parallel: int = 5) -> List[SearchServiceResponse]:
    """
    Fonction principale pour exécuter plusieurs requêtes en lot
    """
    return await query_executor_manager.executor.batch_execute(contracts, max_parallel)


def initialize_query_executor(elasticsearch_client: ElasticsearchClient):
    """
    Initialise le système d'exécution de requêtes
    """
    query_executor_manager.initialize(elasticsearch_client)


async def get_execution_health() -> Dict[str, Any]:
    """
    Vérifie la santé du système d'exécution
    """
    return await query_executor_manager.health_check()


def shutdown_query_executor():
    """
    Arrêt propre du système d'exécution
    """
    query_executor_manager.shutdown()


# === EXPORTS PRINCIPAUX ===

__all__ = [
    # === CLASSES PRINCIPALES ===
    "QueryExecutionEngine",
    "HighLevelQueryExecutor",
    "QueryExecutorManager",
    
    # === ENUMS ===
    "ExecutionStrategy",
    "QueryComplexity", 
    "ExecutionPriority",
    
    # === MODÈLES ===
    "ExecutionContext",
    "ExecutionResult",
    
    # === FONCTIONS PRINCIPALES ===
    "execute_search_query",
    "execute_aggregation_query",
    "batch_execute_queries",
    "initialize_query_executor",
    "get_execution_health",
    "shutdown_query_executor",
    
    # === INSTANCE GLOBALE ===
    "query_executor_manager"
]


# === HELPERS D'INTÉGRATION ===

def get_execution_components():
    """Retourne les composants du système d'exécution"""
    return {
        "manager": query_executor_manager,
        "executor": query_executor_manager.executor if query_executor_manager._initialized else None,
        "engine": query_executor_manager.engine if query_executor_manager._initialized else None
    }


def get_performance_summary() -> Dict[str, Any]:
    """Retourne un résumé des performances d'exécution"""
    if not query_executor_manager._initialized:
        return {"error": "System not initialized"}
    
    try:
        return query_executor_manager.executor.get_execution_statistics()
    except Exception as e:
        return {"error": str(e)}


async def optimize_user_queries(user_id: int) -> Dict[str, Any]:
    """Optimise les requêtes pour un utilisateur"""
    if not query_executor_manager._initialized:
        return {"error": "System not initialized"}
    
    return await query_executor_manager.executor.optimize_for_user(user_id)


# === CONFIGURATION ET LOGGING ===

# Configuration du logging spécialisé
execution_logger = logging.getLogger(f"{__name__}.execution")
performance_logger = logging.getLogger(f"{__name__}.performance")

# Métriques globales pour debugging
_global_execution_stats = {
    "total_queries": 0,
    "successful_queries": 0,
    "failed_queries": 0,
    "cache_hits": 0,
    "average_execution_time": 0.0
}


def update_global_stats(execution_time_ms: int, success: bool, cache_hit: bool):
    """Met à jour les statistiques globales"""
    global _global_execution_stats
    
    _global_execution_stats["total_queries"] += 1
    
    if success:
        _global_execution_stats["successful_queries"] += 1
    else:
        _global_execution_stats["failed_queries"] += 1
    
    if cache_hit:
        _global_execution_stats["cache_hits"] += 1
    
    # Moyenne mobile simple
    current_avg = _global_execution_stats["average_execution_time"]
    total = _global_execution_stats["total_queries"]
    new_avg = ((current_avg * (total - 1)) + execution_time_ms) / total
    _global_execution_stats["average_execution_time"] = new_avg


def get_global_stats() -> Dict[str, Any]:
    """Retourne les statistiques globales"""
    return _global_execution_stats.copy()


# === INITIALISATION CONDITIONNELLE ===

def auto_initialize_if_possible():
    """Tentative d'initialisation automatique si client ES disponible"""
    try:
        from clients.elasticsearch_client import get_default_client
        
        es_client = get_default_client()
        if es_client:
            initialize_query_executor(es_client)
            logger.info("QueryExecutor auto-initialisé avec succès")
            return True
    except ImportError:
        logger.debug("Client Elasticsearch non disponible pour auto-initialisation")
    except Exception as e:
        logger.warning(f"Échec auto-initialisation QueryExecutor: {str(e)}")
    
    return False


# Tentative d'auto-initialisation
# auto_initialize_if_possible()