"""
🔍 Moteur de recherche lexicale SEUL (plus d'hybride)
Responsabilité : Exécution recherches Elasticsearch optimisées
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..models.service_contracts import (
    SearchServiceQuery, 
    SearchServiceResponse,
    QueryMetadata,
    ResponseMetadata
)
from ..clients.elasticsearch_client import ElasticsearchClient
from ..templates.query_templates import QueryTemplateManager
from ..utils.cache import SearchCache
from ..utils.metrics import SearchMetrics
from ..utils.validators import QueryValidator
from ..config.settings import SearchSettings

logger = logging.getLogger(__name__)

class LexicalSearchEngine:
    """
    Solution : Moteur recherche lexicale SEUL (plus d'hybride)
    Responsabilité : Exécution recherches Elasticsearch optimisées
    """
    
    def __init__(
        self,
        elasticsearch_client: ElasticsearchClient,
        template_manager: QueryTemplateManager,
        cache: SearchCache,
        metrics: SearchMetrics,
        settings: SearchSettings
    ):
        self.es_client = elasticsearch_client
        self.template_manager = template_manager
        self.cache = cache
        self.metrics = metrics
        self.settings = settings
        self.validator = QueryValidator()
        
        # Thread pool pour requêtes parallèles
        self.executor = ThreadPoolExecutor(max_workers=settings.max_concurrent_queries)
        
    async def search_lexical(self, query: SearchServiceQuery) -> SearchServiceResponse:
        """
        Exécution recherche lexicale avec requête structurée
        """
        start_time = datetime.now()
        
        try:
            # 1. Validation stricte requête
            self.validator.validate_search_query(query)
            
            # 2. Vérification cache
            cache_key = self._generate_cache_key(query)
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                self.metrics.increment_cache_hit()
                return self._add_cache_metadata(cached_result, True)
            
            # 3. Construction requête Elasticsearch
            es_query = await self._build_elasticsearch_query(query)
            
            # 4. Exécution requête avec timeout
            es_response = await self._execute_with_timeout(es_query, query.search_parameters.timeout_ms)
            
            # 5. Traitement résultats
            response = await self._process_elasticsearch_response(es_response, query)
            
            # 6. Mise en cache
            await self.cache.set(cache_key, response, ttl=self.settings.cache_ttl)
            
            # 7. Métriques
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            self.metrics.record_query_execution(execution_time, query.query_metadata.intent_type)
            
            return response
            
        except Exception as e:
            logger.error(f"Erreur recherche lexicale: {e}", extra={
                "query_id": query.query_metadata.query_id,
                "intent_type": query.query_metadata.intent_type
            })
            self.metrics.increment_error()
            raise
    
    async def _build_elasticsearch_query(self, query: SearchServiceQuery) -> Dict[str, Any]:
        """
        Construction requête Elasticsearch à partir du contrat
        """
        # Sélection template basé sur l'intention
        template = self.template_manager.get_template(query.query_metadata.intent_type)
        
        # Construction query bool
        bool_query = {
            "bool": {
                "must": [],
                "filter": [],
                "should": [],
                "must_not": []
            }
        }
        
        # Ajout filtres obligatoires
        for filter_item in query.filters.required:
            bool_query["bool"]["filter"].append({
                "term": {f"{filter_item.field}": filter_item.value}
            })
        
        # Ajout filtres optionnels
        for filter_item in query.filters.optional:
            bool_query["bool"]["should"].append({
                "term": {f"{filter_item.field}": filter_item.value}
            })
        
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
            elif range_filter.operator == "gt":
                bool_query["bool"]["filter"].append({
                    "range": {range_filter.field: {"gt": range_filter.value}}
                })
            elif range_filter.operator == "lt":
                bool_query["bool"]["filter"].append({
                    "range": {range_filter.field: {"lt": range_filter.value}}
                })
        
        # Ajout recherche textuelle
        if query.filters.text_search:
            text_query = {
                "multi_match": {
                    "query": query.filters.text_search.query,
                    "fields": query.filters.text_search.fields,
                    "type": "best_fields",
                    "operator": "or"
                }
            }
            bool_query["bool"]["must"].append(text_query)
        
        # Construction requête finale
        es_query = {
            "query": bool_query,
            "size": query.search_parameters.limit,
            "from": query.search_parameters.offset,
            "_source": query.search_parameters.fields
        }
        
        # Ajout agrégations si demandées
        if query.aggregations.enabled:
            es_query["aggs"] = await self._build_aggregations(query.aggregations)
        
        # Ajout highlights si demandé
        if query.options.include_highlights:
            es_query["highlight"] = {
                "fields": {
                    field: {} for field in query.filters.text_search.fields
                }
            }
        
        return es_query
    
    async def _build_aggregations(self, agg_config) -> Dict[str, Any]:
        """
        Construction agrégations Elasticsearch
        """
        aggregations = {}
        
        # Métriques simples
        if "sum" in agg_config.types:
            for metric in agg_config.metrics:
                aggregations[f"sum_{metric}"] = {
                    "sum": {"field": metric}
                }
        
        if "count" in agg_config.types:
            aggregations["doc_count"] = {
                "value_count": {"field": "transaction_id"}
            }
        
        if "avg" in agg_config.types:
            for metric in agg_config.metrics:
                aggregations[f"avg_{metric}"] = {
                    "avg": {"field": metric}
                }
        
        # Groupements
        for group_field in agg_config.group_by:
            aggregations[f"group_by_{group_field}"] = {
                "terms": {
                    "field": f"{group_field}.keyword" if group_field in ["category_name", "merchant_name"] else group_field,
                    "size": 100
                },
                "aggs": {
                    "total_amount": {"sum": {"field": "amount_abs"}},
                    "doc_count": {"value_count": {"field": "transaction_id"}}
                }
            }
        
        return aggregations
    
    async def _execute_with_timeout(self, es_query: Dict[str, Any], timeout_ms: int) -> Dict[str, Any]:
        """
        Exécution requête avec timeout
        """
        try:
            # Exécution asynchrone avec timeout
            future = self.executor.submit(
                self.es_client.search,
                index=self.settings.index_name,
                body=es_query,
                timeout=f"{timeout_ms}ms"
            )
            
            response = await asyncio.wait_for(
                asyncio.wrap_future(future),
                timeout=timeout_ms / 1000
            )
            
            return response
            
        except asyncio.TimeoutError:
            logger.warning(f"Timeout requête Elasticsearch: {timeout_ms}ms")
            raise
        except Exception as e:
            logger.error(f"Erreur exécution Elasticsearch: {e}")
            raise
    
    async def _process_elasticsearch_response(
        self, 
        es_response: Dict[str, Any], 
        query: SearchServiceQuery
    ) -> SearchServiceResponse:
        """
        Traitement et formatage résultats Elasticsearch
        """
        hits = es_response.get("hits", {})
        total_hits = hits.get("total", {}).get("value", 0)
        took = es_response.get("took", 0)
        
        # Formatage résultats
        results = []
        for hit in hits.get("hits", []):
            result = hit["_source"]
            result["score"] = hit["_score"]
            
            # Ajout highlights si présents
            if query.options.include_highlights and "highlight" in hit:
                result["highlights"] = hit["highlight"]
            
            results.append(result)
        
        # Traitement agrégations
        aggregations = None
        if query.aggregations.enabled and "aggregations" in es_response:
            aggregations = await self._process_aggregations(es_response["aggregations"])
        
        # Construction réponse
        response = SearchServiceResponse(
            response_metadata=ResponseMetadata(
                query_id=query.query_metadata.query_id,
                execution_time_ms=took,
                total_hits=total_hits,
                returned_hits=len(results),
                has_more=total_hits > (query.search_parameters.offset + len(results)),
                cache_hit=False,
                elasticsearch_took=took,
                agent_context={
                    "requesting_agent": query.query_metadata.agent_name,
                    "requesting_team": query.query_metadata.get("team_name"),
                    "next_suggested_agent": "response_generator_agent"
                }
            ),
            results=results,
            aggregations=aggregations,
            performance={
                "query_complexity": self._assess_query_complexity(query),
                "optimization_applied": self._get_optimizations_applied(query),
                "index_used": self.settings.index_name,
                "shards_queried": 1
            },
            context_enrichment={
                "search_intent_matched": True,
                "result_quality_score": self._calculate_quality_score(results, query),
                "suggested_followup_questions": self._generate_followup_questions(results, query)
            }
        )
        
        return response
    
    async def _process_aggregations(self, agg_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Traitement agrégations Elasticsearch
        """
        processed_aggs = {}
        
        for agg_name, agg_data in agg_response.items():
            if agg_name.startswith("sum_"):
                processed_aggs[agg_name] = agg_data["value"]
            elif agg_name.startswith("avg_"):
                processed_aggs[agg_name] = agg_data["value"]
            elif agg_name.startswith("group_by_"):
                processed_aggs[agg_name] = [
                    {
                        "key": bucket["key"],
                        "doc_count": bucket["doc_count"],
                        "total_amount": bucket.get("total_amount", {}).get("value", 0)
                    }
                    for bucket in agg_data["buckets"]
                ]
        
        return processed_aggs
    
    def _generate_cache_key(self, query: SearchServiceQuery) -> str:
        """
        Génération clé cache basée sur requête
        """
        key_parts = [
            f"user_{query.query_metadata.user_id}",
            f"intent_{query.query_metadata.intent_type}",
            f"limit_{query.search_parameters.limit}",
            f"offset_{query.search_parameters.offset}"
        ]
        
        # Ajout filtres
        for filter_item in query.filters.required:
            key_parts.append(f"{filter_item.field}:{filter_item.value}")
        
        # Ajout recherche textuelle
        if query.filters.text_search:
            key_parts.append(f"text:{query.filters.text_search.query}")
        
        return ":".join(key_parts)
    
    def _assess_query_complexity(self, query: SearchServiceQuery) -> str:
        """
        Évaluation complexité requête
        """
        complexity_score = 0
        
        # Facteurs complexité
        complexity_score += len(query.filters.required)
        complexity_score += len(query.filters.optional) * 0.5
        complexity_score += len(query.filters.ranges) * 2
        
        if query.filters.text_search:
            complexity_score += 3
        
        if query.aggregations.enabled:
            complexity_score += len(query.aggregations.group_by) * 2
            complexity_score += len(query.aggregations.types)
        
        if complexity_score <= 5:
            return "simple"
        elif complexity_score <= 15:
            return "medium"
        else:
            return "complex"
    
    def _get_optimizations_applied(self, query: SearchServiceQuery) -> List[str]:
        """
        Liste optimisations appliquées
        """
        optimizations = []
        
        # Optimisations filtres
        if any(f.field == "user_id" for f in query.filters.required):
            optimizations.append("user_filter")
        
        if any(f.field == "category_name" for f in query.filters.required):
            optimizations.append("category_filter")
        
        # Optimisations cache
        if query.options.cache_enabled:
            optimizations.append("cache_enabled")
        
        return optimizations
    
    def _calculate_quality_score(self, results: List[Dict], query: SearchServiceQuery) -> float:
        """
        Calcul score qualité résultats
        """
        if not results:
            return 0.0
        
        # Score basé sur pertinence BM25
        avg_score = sum(r.get("score", 0) for r in results) / len(results)
        
        # Normalisation score (BM25 généralement 0-10)
        normalized_score = min(avg_score / 10.0, 1.0)
        
        # Ajustement selon couverture
        coverage_bonus = min(len(results) / query.search_parameters.limit, 1.0) * 0.1
        
        return min(normalized_score + coverage_bonus, 1.0)
    
    def _generate_followup_questions(self, results: List[Dict], query: SearchServiceQuery) -> List[str]:
        """
        Génération questions de suivi
        """
        followups = []
        
        if query.query_metadata.intent_type == "SEARCH_BY_CATEGORY" and results:
            followups.append("Voir détails de ces transactions")
            followups.append("Comparer avec période précédente")
        
        if query.query_metadata.intent_type == "SEARCH_BY_MERCHANT" and results:
            followups.append("Analyser fréquence de visite")
            followups.append("Voir évolution montants")
        
        if query.aggregations.enabled and results:
            followups.append("Voir répartition détaillée")
            followups.append("Exporter en rapport")
        
        return followups
    
    def _add_cache_metadata(self, response: SearchServiceResponse, cache_hit: bool) -> SearchServiceResponse:
        """
        Ajout métadonnées cache
        """
        response.response_metadata.cache_hit = cache_hit
        return response

    async def validate_query(self, query: SearchServiceQuery) -> Dict[str, Any]:
        """
        Validation requête Elasticsearch
        """
        try:
            # Validation syntaxique
            self.validator.validate_search_query(query)
            
            # Test construction requête
            es_query = await self._build_elasticsearch_query(query)
            
            # Validation Elasticsearch
            validation_response = await self.es_client.validate_query(
                index=self.settings.index_name,
                body={"query": es_query["query"]}
            )
            
            return {
                "valid": validation_response.get("valid", False),
                "query": es_query,
                "explanation": validation_response.get("explanations", [])
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "query": None
            }
    
    async def get_available_templates(self) -> List[str]:
        """
        Liste des templates disponibles
        """
        return self.template_manager.get_available_templates()
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Vérification santé moteur
        """
        try:
            # Test connexion Elasticsearch
            es_health = await self.es_client.cluster_health()
            
            # Test cache
            cache_health = await self.cache.health_check()
            
            return {
                "status": "healthy",
                "elasticsearch": {
                    "status": es_health.get("status", "unknown"),
                    "cluster_name": es_health.get("cluster_name", "unknown"),
                    "number_of_nodes": es_health.get("number_of_nodes", 0)
                },
                "cache": cache_health,
                "templates": len(await self.get_available_templates()),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }