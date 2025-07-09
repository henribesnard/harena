# search_service/core/query_executor.py
"""
⚡ Exécuteur requêtes Elasticsearch haute performance
Responsabilité : Construction et exécution requêtes
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum

from ..models.service_contracts import SearchServiceQuery, SearchFilter, AggregationRequest
from ..clients.elasticsearch_client import ElasticsearchClient
from ..utils.metrics import SearchMetrics
from ..config.settings import SearchSettings

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types de requêtes supportées"""
    SIMPLE_SEARCH = "simple_search"
    FILTERED_SEARCH = "filtered_search"
    TEXT_SEARCH = "text_search"
    AGGREGATED_SEARCH = "aggregated_search"
    COMPLEX_SEARCH = "complex_search"

@dataclass
class QueryExecutionPlan:
    """Plan d'exécution pour une requête"""
    query_type: QueryType
    estimated_time_ms: int
    complexity_score: float
    parallel_execution: bool
    optimization_hints: List[str]

@dataclass
class ExecutionResult:
    """Résultat d'exécution"""
    success: bool
    data: Optional[Dict[str, Any]]
    execution_time_ms: int
    error: Optional[str]
    query_stats: Dict[str, Any]

class QueryExecutor:
    """
    Solution : Exécuteur requêtes Elasticsearch haute performance
    Responsabilité : Construction et exécution requêtes
    """
    
    def __init__(
        self,
        elasticsearch_client: ElasticsearchClient,
        metrics: SearchMetrics,
        settings: SearchSettings
    ):
        self.es_client = elasticsearch_client
        self.metrics = metrics
        self.settings = settings
        
        # Thread pool pour exécution parallèle
        self.executor = ThreadPoolExecutor(max_workers=settings.max_concurrent_queries)
        
        # Cache plans d'exécution
        self._execution_plans: Dict[str, QueryExecutionPlan] = {}
        
        # Mapping types de requêtes
        self._query_builders = {
            QueryType.SIMPLE_SEARCH: self._build_simple_search,
            QueryType.FILTERED_SEARCH: self._build_filtered_search,
            QueryType.TEXT_SEARCH: self._build_text_search,
            QueryType.AGGREGATED_SEARCH: self._build_aggregated_search,
            QueryType.COMPLEX_SEARCH: self._build_complex_search
        }
    
    async def execute_query(self, query: SearchServiceQuery) -> ExecutionResult:
        """
        Exécution requête avec plan d'exécution optimisé
        """
        start_time = datetime.now()
        
        try:
            # 1. Analyse et planification
            execution_plan = await self._analyze_and_plan(query)
            
            # 2. Construction requête Elasticsearch
            es_query = await self._build_elasticsearch_query(query, execution_plan)
            
            # 3. Exécution selon plan
            if execution_plan.parallel_execution:
                result = await self._execute_parallel(es_query, query)
            else:
                result = await self._execute_sequential(es_query, query)
            
            # 4. Métriques
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            self.metrics.record_query_execution(execution_time, query.query_metadata.intent_type)
            
            return ExecutionResult(
                success=True,
                data=result,
                execution_time_ms=int(execution_time),
                error=None,
                query_stats=self._extract_query_stats(result)
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Erreur exécution requête: {e}", extra={
                "query_id": query.query_metadata.query_id,
                "execution_time_ms": execution_time
            })
            
            return ExecutionResult(
                success=False,
                data=None,
                execution_time_ms=int(execution_time),
                error=str(e),
                query_stats={}
            )
    
    async def execute_multiple_queries(self, queries: List[SearchServiceQuery]) -> List[ExecutionResult]:
        """
        Exécution multiple requêtes en parallèle
        """
        if not queries:
            return []
        
        # Création futures pour parallélisation
        futures = []
        for query in queries:
            future = asyncio.create_task(self.execute_query(query))
            futures.append(future)
        
        # Exécution parallèle avec timeout global
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*futures, return_exceptions=True),
                timeout=self.settings.multi_query_timeout
            )
            
            # Traitement résultats
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append(ExecutionResult(
                        success=False,
                        data=None,
                        execution_time_ms=0,
                        error=str(result),
                        query_stats={}
                    ))
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except asyncio.TimeoutError:
            logger.warning(f"Timeout exécution multiple requêtes: {self.settings.multi_query_timeout}s")
            return [ExecutionResult(
                success=False,
                data=None,
                execution_time_ms=0,
                error="Timeout",
                query_stats={}
            ) for _ in queries]
    
    async def _analyze_and_plan(self, query: SearchServiceQuery) -> QueryExecutionPlan:
        """
        Analyse requête et génération plan d'exécution
        """
        # Clé cache pour plan
        plan_key = self._generate_plan_key(query)
        
        # Vérification cache
        if plan_key in self._execution_plans:
            return self._execution_plans[plan_key]
        
        # Analyse complexité
        complexity_score = self._calculate_complexity(query)
        query_type = self._determine_query_type(query)
        
        # Estimation temps
        estimated_time = self._estimate_execution_time(query, complexity_score)
        
        # Décision parallélisation
        parallel_execution = (
            complexity_score > self.settings.parallel_threshold or
            query.aggregations.enabled and len(query.aggregations.group_by) > 2
        )
        
        # Optimisations suggérées
        optimization_hints = self._generate_optimization_hints(query)
        
        # Création plan
        plan = QueryExecutionPlan(
            query_type=query_type,
            estimated_time_ms=estimated_time,
            complexity_score=complexity_score,
            parallel_execution=parallel_execution,
            optimization_hints=optimization_hints
        )
        
        # Mise en cache
        self._execution_plans[plan_key] = plan
        
        return plan
    
    def _calculate_complexity(self, query: SearchServiceQuery) -> float:
        """
        Calcul score complexité requête
        """
        complexity = 0.0
        
        # Facteurs base
        complexity += len(query.filters.required) * 0.5
        complexity += len(query.filters.optional) * 0.3
        complexity += len(query.filters.ranges) * 1.0
        
        # Recherche textuelle
        if query.filters.text_search:
            complexity += 2.0
            complexity += len(query.filters.text_search.fields) * 0.5
        
        # Agrégations
        if query.aggregations.enabled:
            complexity += len(query.aggregations.group_by) * 2.0
            complexity += len(query.aggregations.types) * 0.5
            complexity += len(query.aggregations.metrics) * 0.3
        
        # Taille résultats
        if query.search_parameters.limit > 100:
            complexity += 1.0
        
        return complexity
    
    def _determine_query_type(self, query: SearchServiceQuery) -> QueryType:
        """
        Détermination type requête
        """
        has_text_search = query.filters.text_search is not None
        has_filters = (
            query.filters.required or 
            query.filters.optional or 
            query.filters.ranges
        )
        has_aggregations = query.aggregations.enabled
        
        # Logique de détermination
        if has_aggregations and has_filters and has_text_search:
            return QueryType.COMPLEX_SEARCH
        elif has_aggregations:
            return QueryType.AGGREGATED_SEARCH
        elif has_text_search:
            return QueryType.TEXT_SEARCH
        elif has_filters:
            return QueryType.FILTERED_SEARCH
        else:
            return QueryType.SIMPLE_SEARCH
    
    def _estimate_execution_time(self, query: SearchServiceQuery, complexity: float) -> int:
        """
        Estimation temps d'exécution
        """
        # Temps base par type
        base_times = {
            QueryType.SIMPLE_SEARCH: 10,
            QueryType.FILTERED_SEARCH: 25,
            QueryType.TEXT_SEARCH: 50,
            QueryType.AGGREGATED_SEARCH: 100,
            QueryType.COMPLEX_SEARCH: 200
        }
        
        query_type = self._determine_query_type(query)
        base_time = base_times.get(query_type, 50)
        
        # Facteur complexité
        complexity_factor = 1.0 + (complexity / 10.0)
        
        # Facteur taille
        size_factor = 1.0 + (query.search_parameters.limit / 1000.0)
        
        return int(base_time * complexity_factor * size_factor)
    
    def _generate_optimization_hints(self, query: SearchServiceQuery) -> List[str]:
        """
        Génération conseils optimisation
        """
        hints = []
        
        # Filtres utilisateur obligatoires
        if not any(f.field == "user_id" for f in query.filters.required):
            hints.append("add_user_filter")
        
        # Limite raisonnable
        if query.search_parameters.limit > 500:
            hints.append("reduce_limit")
        
        # Cache activation
        if not query.options.cache_enabled:
            hints.append("enable_cache")
        
        # Champs spécifiques
        if len(query.search_parameters.fields) > 20:
            hints.append("reduce_fields")
        
        # Agrégations optimisées
        if query.aggregations.enabled and len(query.aggregations.group_by) > 3:
            hints.append("optimize_aggregations")
        
        return hints
    
    async def _build_elasticsearch_query(
        self, 
        query: SearchServiceQuery, 
        plan: QueryExecutionPlan
    ) -> Dict[str, Any]:
        """
        Construction requête Elasticsearch selon plan
        """
        # Délégation au builder approprié
        builder = self._query_builders.get(plan.query_type, self._build_simple_search)
        return await builder(query, plan)
    
    async def _build_simple_search(
        self, 
        query: SearchServiceQuery, 
        plan: QueryExecutionPlan
    ) -> Dict[str, Any]:
        """
        Construction requête simple
        """
        return {
            "query": {
                "bool": {
                    "filter": [
                        {"term": {f.field: f.value}} 
                        for f in query.filters.required
                    ]
                }
            },
            "size": query.search_parameters.limit,
            "from": query.search_parameters.offset,
            "_source": query.search_parameters.fields
        }
    
    async def _build_filtered_search(
        self, 
        query: SearchServiceQuery, 
        plan: QueryExecutionPlan
    ) -> Dict[str, Any]:
        """
        Construction requête avec filtres
        """
        bool_query = {
            "bool": {
                "filter": [],
                "should": [],
                "must_not": []
            }
        }
        
        # Filtres obligatoires
        for filter_item in query.filters.required:
            bool_query["bool"]["filter"].append({
                "term": {f"{filter_item.field}": filter_item.value}
            })
        
        # Filtres optionnels
        for filter_item in query.filters.optional:
            bool_query["bool"]["should"].append({
                "term": {f"{filter_item.field}": filter_item.value}
            })
        
        # Filtres range
        for range_filter in query.filters.ranges:
            range_clause = {}
            if range_filter.operator == "between":
                range_clause[range_filter.field] = {
                    "gte": range_filter.value[0],
                    "lte": range_filter.value[1]
                }
            elif range_filter.operator == "gt":
                range_clause[range_filter.field] = {"gt": range_filter.value}
            elif range_filter.operator == "gte":
                range_clause[range_filter.field] = {"gte": range_filter.value}
            elif range_filter.operator == "lt":
                range_clause[range_filter.field] = {"lt": range_filter.value}
            elif range_filter.operator == "lte":
                range_clause[range_filter.field] = {"lte": range_filter.value}
            
            bool_query["bool"]["filter"].append({"range": range_clause})
        
        return {
            "query": bool_query,
            "size": query.search_parameters.limit,
            "from": query.search_parameters.offset,
            "_source": query.search_parameters.fields,
            "sort": [{"date": {"order": "desc"}}]
        }
    
    async def _build_text_search(
        self, 
        query: SearchServiceQuery, 
        plan: QueryExecutionPlan
    ) -> Dict[str, Any]:
        """
        Construction requête recherche textuelle
        """
        bool_query = {
            "bool": {
                "must": [],
                "filter": []
            }
        }
        
        # Recherche textuelle principale
        if query.filters.text_search:
            text_query = {
                "multi_match": {
                    "query": query.filters.text_search.query,
                    "fields": query.filters.text_search.fields,
                    "type": "best_fields",
                    "operator": "or",
                    "fuzziness": "AUTO",
                    "boost": 1.0
                }
            }
            bool_query["bool"]["must"].append(text_query)
        
        # Filtres obligatoires
        for filter_item in query.filters.required:
            bool_query["bool"]["filter"].append({
                "term": {f"{filter_item.field}": filter_item.value}
            })
        
        # Filtres range
        for range_filter in query.filters.ranges:
            if range_filter.operator == "between":
                bool_query["bool"]["filter"].append({
                    "range": {
                        range_filter.field: {
                            "gte": range_filter.value[0],
                            "lte": range_filter.value[1]
                        }
                    }
                })
        
        es_query = {
            "query": bool_query,
            "size": query.search_parameters.limit,
            "from": query.search_parameters.offset,
            "_source": query.search_parameters.fields
        }
        
        # Highlights pour recherche textuelle
        if query.options.include_highlights:
            es_query["highlight"] = {
                "fields": {
                    field: {
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"],
                        "fragment_size": 150,
                        "number_of_fragments": 3
                    } for field in query.filters.text_search.fields
                }
            }
        
        return es_query
    
    async def _build_aggregated_search(
        self, 
        query: SearchServiceQuery, 
        plan: QueryExecutionPlan
    ) -> Dict[str, Any]:
        """
        Construction requête avec agrégations
        """
        # Query base
        bool_query = {
            "bool": {
                "filter": [
                    {"term": {f.field: f.value}} 
                    for f in query.filters.required
                ]
            }
        }
        
        # Ajout filtres range
        for range_filter in query.filters.ranges:
            if range_filter.operator == "between":
                bool_query["bool"]["filter"].append({
                    "range": {
                        range_filter.field: {
                            "gte": range_filter.value[0],
                            "lte": range_filter.value[1]
                        }
                    }
                })
        
        es_query = {
            "query": bool_query,
            "size": 0,  # Pas besoin de documents pour agrégations
            "aggs": await self._build_aggregations_advanced(query.aggregations)
        }
        
        return es_query
    
    async def _build_complex_search(
        self, 
        query: SearchServiceQuery, 
        plan: QueryExecutionPlan
    ) -> Dict[str, Any]:
        """
        Construction requête complexe
        """
        bool_query = {
            "bool": {
                "must": [],
                "filter": [],
                "should": []
            }
        }
        
        # Recherche textuelle
        if query.filters.text_search:
            text_query = {
                "multi_match": {
                    "query": query.filters.text_search.query,
                    "fields": query.filters.text_search.fields,
                    "type": "cross_fields",
                    "operator": "and",
                    "fuzziness": "AUTO"
                }
            }
            bool_query["bool"]["must"].append(text_query)
        
        # Filtres obligatoires
        for filter_item in query.filters.required:
            bool_query["bool"]["filter"].append({
                "term": {f"{filter_item.field}": filter_item.value}
            })
        
        # Filtres optionnels avec boost
        for filter_item in query.filters.optional:
            bool_query["bool"]["should"].append({
                "term": {
                    f"{filter_item.field}": {
                        "value": filter_item.value,
                        "boost": 0.5
                    }
                }
            })
        
        # Filtres range
        for range_filter in query.filters.ranges:
            if range_filter.operator == "between":
                bool_query["bool"]["filter"].append({
                    "range": {
                        range_filter.field: {
                            "gte": range_filter.value[0],
                            "lte": range_filter.value[1]
                        }
                    }
                })
        
        es_query = {
            "query": bool_query,
            "size": query.search_parameters.limit,
            "from": query.search_parameters.offset,
            "_source": query.search_parameters.fields,
            "sort": [
                {"_score": {"order": "desc"}},
                {"date": {"order": "desc"}}
            ]
        }
        
        # Agrégations si demandées
        if query.aggregations.enabled:
            es_query["aggs"] = await self._build_aggregations_advanced(query.aggregations)
        
        # Highlights
        if query.options.include_highlights and query.filters.text_search:
            es_query["highlight"] = {
                "fields": {
                    field: {
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"],
                        "fragment_size": 200,
                        "number_of_fragments": 2
                    } for field in query.filters.text_search.fields
                }
            }
        
        return es_query
    
    async def _build_aggregations_advanced(self, agg_config: AggregationRequest) -> Dict[str, Any]:
        """
        Construction agrégations avancées
        """
        aggregations = {}
        
        # Métriques globales
        if "sum" in agg_config.types:
            for metric in agg_config.metrics:
                aggregations[f"total_{metric}"] = {
                    "sum": {"field": metric}
                }
        
        if "avg" in agg_config.types:
            for metric in agg_config.metrics:
                aggregations[f"avg_{metric}"] = {
                    "avg": {"field": metric}
                }
        
        if "count" in agg_config.types:
            aggregations["transaction_count"] = {
                "value_count": {"field": "transaction_id"}
            }
        
        if "min" in agg_config.types:
            for metric in agg_config.metrics:
                aggregations[f"min_{metric}"] = {
                    "min": {"field": metric}
                }
        
        if "max" in agg_config.types:
            for metric in agg_config.metrics:
                aggregations[f"max_{metric}"] = {
                    "max": {"field": metric}
                }
        
        # Statistiques avancées
        if "stats" in agg_config.types:
            for metric in agg_config.metrics:
                aggregations[f"stats_{metric}"] = {
                    "stats": {"field": metric}
                }
        
        # Groupements avec sous-agrégations
        for group_field in agg_config.group_by:
            field_name = f"{group_field}.keyword" if group_field in ["category_name", "merchant_name"] else group_field
            
            sub_aggs = {}
            for metric in agg_config.metrics:
                sub_aggs[f"sum_{metric}"] = {"sum": {"field": metric}}
                sub_aggs[f"avg_{metric}"] = {"avg": {"field": metric}}
            
            sub_aggs["doc_count"] = {"value_count": {"field": "transaction_id"}}
            
            aggregations[f"by_{group_field}"] = {
                "terms": {
                    "field": field_name,
                    "size": 100,
                    "order": {"sum_amount_abs": "desc"}
                },
                "aggs": sub_aggs
            }
        
        # Agrégations temporelles spéciales
        if "date" in agg_config.group_by:
            aggregations["by_date_histogram"] = {
                "date_histogram": {
                    "field": "date",
                    "calendar_interval": "day",
                    "format": "yyyy-MM-dd"
                },
                "aggs": {
                    "daily_sum": {"sum": {"field": "amount_abs"}},
                    "daily_count": {"value_count": {"field": "transaction_id"}}
                }
            }
        
        if "month_year" in agg_config.group_by:
            aggregations["by_month"] = {
                "terms": {
                    "field": "month_year",
                    "size": 24,
                    "order": {"_key": "desc"}
                },
                "aggs": {
                    "monthly_sum": {"sum": {"field": "amount_abs"}},
                    "monthly_count": {"value_count": {"field": "transaction_id"}},
                    "monthly_avg": {"avg": {"field": "amount_abs"}}
                }
            }
        
        return aggregations
    
    async def _execute_sequential(self, es_query: Dict[str, Any], query: SearchServiceQuery) -> Dict[str, Any]:
        """
        Exécution séquentielle
        """
        try:
            response = await asyncio.wrap_future(
                self.executor.submit(
                    self.es_client.search,
                    index=self.settings.index_name,
                    body=es_query,
                    timeout=f"{query.search_parameters.timeout_ms}ms"
                )
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Erreur exécution séquentielle: {e}")
            raise
    
    async def _execute_parallel(self, es_query: Dict[str, Any], query: SearchServiceQuery) -> Dict[str, Any]:
        """
        Exécution parallèle pour requêtes complexes
        """
        try:
            # Séparation requête en parties parallélisables
            if query.aggregations.enabled and "aggs" in es_query:
                # Exécution séparée documents et agrégations
                doc_query = {k: v for k, v in es_query.items() if k != "aggs"}
                agg_query = {
                    "query": es_query["query"],
                    "aggs": es_query["aggs"],
                    "size": 0
                }
                
                # Exécution parallèle
                doc_future = self.executor.submit(
                    self.es_client.search,
                    index=self.settings.index_name,
                    body=doc_query
                )
                
                agg_future = self.executor.submit(
                    self.es_client.search,
                    index=self.settings.index_name,
                    body=agg_query
                )
                
                # Attente résultats
                doc_response, agg_response = await asyncio.gather(
                    asyncio.wrap_future(doc_future),
                    asyncio.wrap_future(agg_future)
                )
                
                # Fusion résultats
                merged_response = doc_response.copy()
                if "aggregations" in agg_response:
                    merged_response["aggregations"] = agg_response["aggregations"]
                
                return merged_response
            
            else:
                # Exécution normale si pas de parallélisation possible
                return await self._execute_sequential(es_query, query)
                
        except Exception as e:
            logger.error(f"Erreur exécution parallèle: {e}")
            # Fallback séquentiel
            return await self._execute_sequential(es_query, query)
    
    def _extract_query_stats(self, es_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extraction statistiques requête
        """
        return {
            "took": es_response.get("took", 0),
            "timed_out": es_response.get("timed_out", False),
            "total_hits": es_response.get("hits", {}).get("total", {}).get("value", 0),
            "max_score": es_response.get("hits", {}).get("max_score", 0),
            "shards": es_response.get("_shards", {}),
            "has_aggregations": "aggregations" in es_response
        }
    
    def _generate_plan_key(self, query: SearchServiceQuery) -> str:
        """
        Génération clé unique pour plan d'exécution
        """
        key_parts = [
            f"intent_{query.query_metadata.intent_type}",
            f"filters_{len(query.filters.required)}_{len(query.filters.optional)}_{len(query.filters.ranges)}",
            f"text_search_{query.filters.text_search is not None}",
            f"aggs_{query.aggregations.enabled}_{len(query.aggregations.group_by) if query.aggregations.enabled else 0}",
            f"limit_{query.search_parameters.limit}"
        ]
        
        return ":".join(key_parts)
    
    async def optimize_query(self, query: SearchServiceQuery) -> Dict[str, Any]:
        """
        Optimisation requête avec suggestions
        """
        plan = await self._analyze_and_plan(query)
        
        optimizations = {
            "current_complexity": plan.complexity_score,
            "estimated_time_ms": plan.estimated_time_ms,
            "parallel_execution": plan.parallel_execution,
            "suggestions": []
        }
        
        # Suggestions d'optimisation
        for hint in plan.optimization_hints:
            if hint == "add_user_filter":
                optimizations["suggestions"].append({
                    "type": "security",
                    "message": "Ajouter filtre user_id obligatoire",
                    "impact": "high"
                })
            elif hint == "reduce_limit":
                optimizations["suggestions"].append({
                    "type": "performance",
                    "message": f"Réduire limite de {query.search_parameters.limit} à 100",
                    "impact": "medium"
                })
            elif hint == "enable_cache":
                optimizations["suggestions"].append({
                    "type": "performance",
                    "message": "Activer cache pour améliorer performances",
                    "impact": "high"
                })
        
        return optimizations
    
    async def explain_query(self, query: SearchServiceQuery) -> Dict[str, Any]:
        """
        Explication détaillée requête
        """
        plan = await self._analyze_and_plan(query)
        es_query = await self._build_elasticsearch_query(query, plan)
        
        # Explication Elasticsearch
        try:
            explain_response = await asyncio.wrap_future(
                self.executor.submit(
                    self.es_client.explain_query,
                    index=self.settings.index_name,
                    body=es_query
                )
            )
        except:
            explain_response = {"error": "Could not generate explanation"}
        
        return {
            "query_metadata": {
                "intent_type": query.query_metadata.intent_type,
                "query_type": plan.query_type.value,
                "complexity_score": plan.complexity_score,
                "estimated_time_ms": plan.estimated_time_ms
            },
            "elasticsearch_query": es_query,
            "execution_plan": {
                "parallel_execution": plan.parallel_execution,
                "optimization_hints": plan.optimization_hints
            },
            "explanation": explain_response
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Vérification santé exécuteur
        """
        try:
            # Test requête simple
            test_query = {
                "query": {"match_all": {}},
                "size": 1
            }
            
            start_time = datetime.now()
            response = await asyncio.wrap_future(
                self.executor.submit(
                    self.es_client.search,
                    index=self.settings.index_name,
                    body=test_query
                )
            )
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                "status": "healthy",
                "test_query_time_ms": execution_time,
                "executor_threads": self.executor._threads,
                "cached_plans": len(self._execution_plans),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }