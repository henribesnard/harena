"""
Exécuteur de requêtes Elasticsearch - Composant Core #1
Construit et exécute les requêtes Elasticsearch optimisées
"""

import logging
from typing import Dict, List, Any, Optional, Union
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ElasticsearchException, RequestError, NotFoundError

from search_service.models.service_contracts import SearchServiceQuery, QueryMetadata
from search_service.models.elasticsearch_queries import ElasticsearchQuery
from search_service.templates.query_templates import QueryTemplateManager
from search_service.utils.validators import QueryValidator
from search_service.utils.elasticsearch_helpers import ElasticsearchHelper

logger = logging.getLogger(__name__)


class QueryExecutor:
    """
    Exécuteur de requêtes Elasticsearch haute performance
    
    Responsabilités:
    - Construction de requêtes à partir de templates
    - Optimisation des requêtes bool complexes
    - Gestion des agrégations financières
    - Parallélisation des requêtes multiples
    """
    
    def __init__(self, elasticsearch_client: Elasticsearch, template_manager: QueryTemplateManager):
        self.es_client = elasticsearch_client
        self.template_manager = template_manager
        self.validator = QueryValidator()
        self.helper = ElasticsearchHelper()
        
        # Configuration par défaut
        self.default_size = 10
        self.max_size = 100
        self.timeout = "30s"
        
        logger.info("✅ QueryExecutor initialisé")
    
    async def execute_query(self, query: SearchServiceQuery) -> Dict[str, Any]:
        """
        Exécute une requête de recherche
        
        Args:
            query: Requête de recherche standardisée
            
        Returns:
            Résultats Elasticsearch bruts
        """
        try:
            # Validation de la requête
            self.validator.validate_search_query(query)
            
            # Construction de la requête Elasticsearch
            es_query = await self._build_elasticsearch_query(query)
            
            # Exécution de la requête
            results = await self._execute_elasticsearch_query(es_query, query.metadata)
            
            logger.info(f"✅ Requête exécutée avec succès - {results.get('hits', {}).get('total', {}).get('value', 0)} résultats")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'exécution de la requête: {e}")
            raise
    
    async def execute_multi_query(self, queries: List[SearchServiceQuery]) -> List[Dict[str, Any]]:
        """
        Exécute plusieurs requêtes en parallèle
        
        Args:
            queries: Liste des requêtes à exécuter
            
        Returns:
            Liste des résultats
        """
        try:
            # Validation des requêtes
            for query in queries:
                self.validator.validate_search_query(query)
            
            # Construction des requêtes Elasticsearch
            es_queries = []
            for query in queries:
                es_query = await self._build_elasticsearch_query(query)
                es_queries.append(es_query)
            
            # Exécution en parallèle via msearch
            results = await self._execute_multi_search(es_queries)
            
            logger.info(f"✅ {len(queries)} requêtes exécutées en parallèle")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'exécution des requêtes multiples: {e}")
            raise
    
    async def execute_aggregation(self, query: SearchServiceQuery) -> Dict[str, Any]:
        """
        Exécute une requête avec agrégations
        
        Args:
            query: Requête avec agrégations
            
        Returns:
            Résultats d'agrégation
        """
        try:
            # Validation
            self.validator.validate_search_query(query)
            
            # Construction avec focus sur les agrégations
            es_query = await self._build_aggregation_query(query)
            
            # Exécution
            results = await self._execute_elasticsearch_query(es_query, query.metadata)
            
            logger.info(f"✅ Requête d'agrégation exécutée - {len(results.get('aggregations', {}))} agrégations")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'exécution de l'agrégation: {e}")
            raise
    
    async def _build_elasticsearch_query(self, query: SearchServiceQuery) -> ElasticsearchQuery:
        """
        Construit une requête Elasticsearch à partir d'une requête standardisée
        
        Args:
            query: Requête de recherche standardisée
            
        Returns:
            Requête Elasticsearch optimisée
        """
        try:
            # Sélection du template approprié
            template = self.template_manager.get_template_for_query(query)
            
            # Construction de la requête de base
            es_query = ElasticsearchQuery(
                index=query.index or "financial_documents",
                size=min(query.size or self.default_size, self.max_size),
                from_=query.from_,
                timeout=self.timeout
            )
            
            # Construction de la requête bool
            bool_query = {
                "must": [],
                "filter": [],
                "should": [],
                "must_not": []
            }
            
            # Ajout des termes de recherche
            if query.query:
                if query.search_type == "phrase":
                    # Recherche de phrase exacte
                    bool_query["must"].append({
                        "match_phrase": {
                            "content": {
                                "query": query.query,
                                "boost": 2.0
                            }
                        }
                    })
                elif query.search_type == "fuzzy":
                    # Recherche floue
                    bool_query["must"].append({
                        "multi_match": {
                            "query": query.query,
                            "fields": ["title^3", "content^2", "summary"],
                            "fuzziness": "AUTO",
                            "prefix_length": 2
                        }
                    })
                else:
                    # Recherche standard
                    bool_query["must"].append({
                        "multi_match": {
                            "query": query.query,
                            "fields": ["title^3", "content^2", "summary"],
                            "type": "best_fields"
                        }
                    })
            
            # Ajout des filtres
            if query.filters:
                for filter_item in query.filters:
                    filter_clause = self._build_filter_clause(filter_item)
                    if filter_clause:
                        bool_query["filter"].append(filter_clause)
            
            # Ajout des highlights
            highlight = {
                "fields": {
                    "title": {"number_of_fragments": 1},
                    "content": {"number_of_fragments": 3, "fragment_size": 150},
                    "summary": {"number_of_fragments": 1}
                },
                "pre_tags": ["<mark>"],
                "post_tags": ["</mark>"]
            }
            
            # Construction de la requête finale
            es_query.body = {
                "query": {
                    "bool": bool_query
                },
                "highlight": highlight,
                "sort": self._build_sort_clause(query),
                "_source": query.source_fields or True
            }
            
            # Ajout des agrégations si nécessaire
            if query.aggregations:
                es_query.body["aggs"] = self._build_aggregations(query.aggregations)
            
            return es_query
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la construction de la requête: {e}")
            raise
    
    async def _build_aggregation_query(self, query: SearchServiceQuery) -> ElasticsearchQuery:
        """
        Construit une requête d'agrégation optimisée
        
        Args:
            query: Requête avec agrégations
            
        Returns:
            Requête d'agrégation Elasticsearch
        """
        try:
            # Template spécialisé pour les agrégations
            template = self.template_manager.get_aggregation_template(query)
            
            # Construction de la requête d'agrégation
            es_query = ElasticsearchQuery(
                index=query.index or "financial_documents",
                size=0,  # Pas besoin de résultats pour les agrégations pures
                timeout=self.timeout
            )
            
            # Construction des agrégations
            aggregations = {}
            if query.aggregations:
                for agg_request in query.aggregations:
                    agg_name = agg_request.name
                    agg_config = self._build_aggregation_config(agg_request)
                    aggregations[agg_name] = agg_config
            
            # Requête de base avec filtres
            base_query = {"match_all": {}}
            if query.filters:
                bool_query = {"filter": []}
                for filter_item in query.filters:
                    filter_clause = self._build_filter_clause(filter_item)
                    if filter_clause:
                        bool_query["filter"].append(filter_clause)
                base_query = {"bool": bool_query}
            
            es_query.body = {
                "query": base_query,
                "aggs": aggregations
            }
            
            return es_query
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la construction de l'agrégation: {e}")
            raise
    
    def _build_filter_clause(self, filter_item) -> Optional[Dict[str, Any]]:
        """
        Construit une clause de filtre Elasticsearch
        
        Args:
            filter_item: Élément de filtre
            
        Returns:
            Clause de filtre Elasticsearch
        """
        try:
            if filter_item.field and filter_item.value:
                if filter_item.operator == "equals":
                    return {"term": {filter_item.field: filter_item.value}}
                elif filter_item.operator == "range":
                    return {"range": {filter_item.field: filter_item.value}}
                elif filter_item.operator == "exists":
                    return {"exists": {"field": filter_item.field}}
                elif filter_item.operator == "in":
                    return {"terms": {filter_item.field: filter_item.value}}
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la construction du filtre: {e}")
            return None
    
    def _build_sort_clause(self, query: SearchServiceQuery) -> List[Dict[str, Any]]:
        """
        Construit la clause de tri
        
        Args:
            query: Requête de recherche
            
        Returns:
            Clause de tri Elasticsearch
        """
        try:
            if query.sort_by:
                sort_order = query.sort_order or "desc"
                return [{query.sort_by: {"order": sort_order}}]
            
            # Tri par défaut par score de pertinence
            return ["_score"]
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la construction du tri: {e}")
            return ["_score"]
    
    def _build_aggregations(self, aggregations: List) -> Dict[str, Any]:
        """
        Construit les agrégations Elasticsearch
        
        Args:
            aggregations: Liste des agrégations demandées
            
        Returns:
            Configuration des agrégations
        """
        try:
            aggs = {}
            for agg_request in aggregations:
                agg_name = agg_request.name
                agg_config = self._build_aggregation_config(agg_request)
                aggs[agg_name] = agg_config
            
            return aggs
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la construction des agrégations: {e}")
            return {}
    
    def _build_aggregation_config(self, agg_request) -> Dict[str, Any]:
        """
        Construit la configuration d'une agrégation
        
        Args:
            agg_request: Demande d'agrégation
            
        Returns:
            Configuration de l'agrégation
        """
        try:
            if agg_request.type == "terms":
                return {
                    "terms": {
                        "field": agg_request.field,
                        "size": agg_request.size or 10
                    }
                }
            elif agg_request.type == "date_histogram":
                return {
                    "date_histogram": {
                        "field": agg_request.field,
                        "calendar_interval": agg_request.interval or "month"
                    }
                }
            elif agg_request.type == "stats":
                return {
                    "stats": {
                        "field": agg_request.field
                    }
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la configuration de l'agrégation: {e}")
            return {}
    
    async def _execute_elasticsearch_query(self, es_query: ElasticsearchQuery, metadata: Optional[QueryMetadata] = None) -> Dict[str, Any]:
        """
        Exécute une requête Elasticsearch
        
        Args:
            es_query: Requête Elasticsearch
            metadata: Métadonnées de la requête
            
        Returns:
            Résultats Elasticsearch
        """
        try:
            # Logging de la requête pour debug
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"🔍 Exécution requête ES: {es_query.body}")
            
            # Exécution de la requête
            response = await self.es_client.search(
                index=es_query.index,
                body=es_query.body,
                size=es_query.size,
                from_=es_query.from_,
                timeout=es_query.timeout
            )
            
            return response
            
        except RequestError as e:
            logger.error(f"❌ Erreur de requête Elasticsearch: {e}")
            raise
        except NotFoundError as e:
            logger.error(f"❌ Index non trouvé: {e}")
            raise
        except ElasticsearchException as e:
            logger.error(f"❌ Erreur Elasticsearch: {e}")
            raise
    
    async def _execute_multi_search(self, es_queries: List[ElasticsearchQuery]) -> List[Dict[str, Any]]:
        """
        Exécute plusieurs requêtes via msearch
        
        Args:
            es_queries: Liste des requêtes Elasticsearch
            
        Returns:
            Liste des résultats
        """
        try:
            # Construction du body msearch
            body = []
            for es_query in es_queries:
                header = {"index": es_query.index}
                body.append(header)
                body.append(es_query.body)
            
            # Exécution msearch
            response = await self.es_client.msearch(body=body)
            
            # Extraction des résultats
            results = []
            for response_item in response.get("responses", []):
                if "error" in response_item:
                    logger.error(f"❌ Erreur dans msearch: {response_item['error']}")
                    results.append({})
                else:
                    results.append(response_item)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'exécution msearch: {e}")
            raise
    
    def get_query_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques d'utilisation
        
        Returns:
            Statistiques du query executor
        """
        return {
            "elasticsearch_client": str(self.es_client),
            "default_size": self.default_size,
            "max_size": self.max_size,
            "timeout": self.timeout
        }