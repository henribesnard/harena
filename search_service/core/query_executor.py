"""
Query Executor - Construction et exécution des requêtes Elasticsearch

Responsabilité : Transforme les contrats SearchServiceQuery en requêtes Elasticsearch
optimisées et gère leur exécution asynchrone avec gestion d'erreurs robuste.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import ElasticsearchException, NotFoundError

from ..models.service_contracts import SearchServiceQuery, QueryFilter
from ..models.elasticsearch_queries import ElasticsearchQuery, BoolQuery
from ..templates.query_templates import QueryTemplateManager
from ..utils.elasticsearch_helpers import ElasticsearchQueryBuilder
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class QueryExecutor:
    """
    Exécuteur de requêtes Elasticsearch haute performance.
    
    Responsabilités:
    - Construction requêtes Elasticsearch depuis contrats
    - Validation syntaxique avant exécution
    - Exécution asynchrone avec timeouts
    - Gestion erreurs et retry logic
    - Optimisation requêtes complexes
    """
    
    def __init__(self, elasticsearch_client: AsyncElasticsearch):
        self.client = elasticsearch_client
        self.settings = get_settings()
        self.template_manager = QueryTemplateManager()
        self.query_builder = ElasticsearchQueryBuilder()
        
        # Configuration performance
        self.default_timeout = self.settings.ELASTICSEARCH_TIMEOUT_MS
        self.max_retry_attempts = 3
        self.retry_delay = 0.5
        
        # Cache requêtes construites pour optimisation
        self._query_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info("QueryExecutor initialisé avec client Elasticsearch")
    
    async def execute_search(
        self, 
        query_contract: SearchServiceQuery
    ) -> Dict[str, Any]:
        """
        Exécute une recherche Elasticsearch depuis un contrat.
        
        Args:
            query_contract: Contrat de requête standardisé
            
        Returns:
            Dict: Réponse brute Elasticsearch
            
        Raises:
            ElasticsearchException: Erreur Elasticsearch
            ValueError: Contrat invalide
            TimeoutError: Timeout dépassé
        """
        try:
            # 1. Construction de la requête Elasticsearch
            es_query = await self._build_elasticsearch_query(query_contract)
            
            # 2. Validation syntaxique
            self._validate_query(es_query)
            
            # 3. Optimisation basée sur le type de requête
            optimized_query = await self._optimize_query(es_query, query_contract)
            
            # 4. Exécution avec retry logic
            result = await self._execute_with_retry(
                optimized_query,
                query_contract.search_parameters.timeout_ms or self.default_timeout
            )
            
            logger.info(
                f"Requête exécutée avec succès: {result.get('took', 0)}ms, "
                f"{result.get('hits', {}).get('total', {}).get('value', 0)} résultats"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution de la requête: {str(e)}")
            raise
    
    async def execute_multi_search(
        self, 
        query_contracts: List[SearchServiceQuery]
    ) -> List[Dict[str, Any]]:
        """
        Exécute plusieurs recherches en parallèle pour optimiser les performances.
        
        Args:
            query_contracts: Liste des contrats de requête
            
        Returns:
            List[Dict]: Liste des réponses Elasticsearch
        """
        if not query_contracts:
            return []
        
        try:
            # Construction de toutes les requêtes en parallèle
            es_queries = await asyncio.gather(
                *[self._build_elasticsearch_query(contract) for contract in query_contracts]
            )
            
            # Préparation pour multi-search
            body = []
            for i, (es_query, contract) in enumerate(zip(es_queries, query_contracts)):
                # Header pour chaque requête
                index_header = {
                    "index": self.settings.ELASTICSEARCH_INDEX,
                    "timeout": f"{contract.search_parameters.timeout_ms or self.default_timeout}ms"
                }
                body.append(index_header)
                body.append(es_query)
            
            # Exécution multi-search
            result = await self.client.msearch(body=body)
            
            # Traitement des réponses
            responses = []
            for i, response in enumerate(result.get("responses", [])):
                if "error" in response:
                    logger.error(f"Erreur dans la requête {i}: {response['error']}")
                    responses.append({"error": response["error"]})
                else:
                    responses.append(response)
            
            logger.info(f"Multi-search exécuté: {len(responses)} requêtes")
            return responses
            
        except Exception as e:
            logger.error(f"Erreur lors du multi-search: {str(e)}")
            raise
    
    async def validate_query_syntax(
        self, 
        query_contract: SearchServiceQuery
    ) -> Dict[str, Any]:
        """
        Valide la syntaxe d'une requête sans l'exécuter.
        
        Args:
            query_contract: Contrat de requête à valider
            
        Returns:
            Dict: Résultat de validation avec détails
        """
        try:
            # Construction de la requête
            es_query = await self._build_elasticsearch_query(query_contract)
            
            # Validation via Elasticsearch validate API
            validation_result = await self.client.indices.validate_query(
                index=self.settings.ELASTICSEARCH_INDEX,
                body={"query": es_query.get("query", {})},
                explain=True
            )
            
            return {
                "valid": validation_result.get("valid", False),
                "explanation": validation_result.get("explanations", []),
                "query": es_query
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la validation: {str(e)}")
            return {
                "valid": False,
                "error": str(e),
                "query": None
            }
    
    async def _build_elasticsearch_query(
        self, 
        query_contract: SearchServiceQuery
    ) -> Dict[str, Any]:
        """
        Construit une requête Elasticsearch depuis un contrat.
        
        Args:
            query_contract: Contrat de requête standardisé
            
        Returns:
            Dict: Requête Elasticsearch complète
        """
        # Vérification du cache
        cache_key = self._generate_cache_key(query_contract)
        if cache_key in self._query_cache:
            logger.debug("Requête trouvée dans le cache")
            return self._query_cache[cache_key].copy()
        
        # Sélection du template basé sur l'intention
        template = self.template_manager.get_template(
            query_contract.query_metadata.intent_type,
            query_contract.search_parameters.query_type
        )
        
        if not template:
            # Fallback: construction manuelle
            es_query = await self._build_manual_query(query_contract)
        else:
            # Utilisation du template
            es_query = self.query_builder.build_from_template(
                template, query_contract
            )
        
        # Ajout des paramètres de recherche
        self._add_search_parameters(es_query, query_contract)
        
        # Ajout des agrégations si demandées
        if query_contract.aggregations and query_contract.aggregations.enabled:
            self._add_aggregations(es_query, query_contract)
        
        # Cache de la requête construite
        self._query_cache[cache_key] = es_query.copy()
        
        return es_query
    
    async def _build_manual_query(
        self, 
        query_contract: SearchServiceQuery
    ) -> Dict[str, Any]:
        """
        Construction manuelle d'une requête Elasticsearch.
        Utilisé comme fallback quand aucun template n'est disponible.
        """
        query = {
            "query": {
                "bool": {
                    "must": [],
                    "filter": [],
                    "should": [],
                    "must_not": []
                }
            }
        }
        
        # Ajout des filtres obligatoires
        for filter_item in query_contract.filters.required:
            term_query = self._build_filter_query(filter_item)
            query["query"]["bool"]["filter"].append(term_query)
        
        # Ajout des filtres optionnels
        for filter_item in query_contract.filters.optional:
            term_query = self._build_filter_query(filter_item)
            query["query"]["bool"]["should"].append(term_query)
        
        # Ajout des filtres de plage
        for range_filter in query_contract.filters.ranges:
            range_query = self._build_range_query(range_filter)
            query["query"]["bool"]["filter"].append(range_query)
        
        # Ajout de la recherche textuelle
        if query_contract.filters.text_search:
            text_query = self._build_text_search_query(
                query_contract.filters.text_search
            )
            query["query"]["bool"]["must"].append(text_query)
        
        return query
    
    def _build_filter_query(self, filter_item: QueryFilter) -> Dict[str, Any]:
        """Construit une requête de filtre selon l'opérateur."""
        if filter_item.operator == "eq":
            return {"term": {filter_item.field: filter_item.value}}
        elif filter_item.operator == "in":
            return {"terms": {filter_item.field: filter_item.value}}
        elif filter_item.operator == "exists":
            return {"exists": {"field": filter_item.field}}
        else:
            raise ValueError(f"Opérateur de filtre non supporté: {filter_item.operator}")
    
    def _build_range_query(self, range_filter: QueryFilter) -> Dict[str, Any]:
        """Construit une requête de plage."""
        range_query = {"range": {range_filter.field: {}}}
        
        if range_filter.operator == "gt":
            range_query["range"][range_filter.field]["gt"] = range_filter.value
        elif range_filter.operator == "gte":
            range_query["range"][range_filter.field]["gte"] = range_filter.value
        elif range_filter.operator == "lt":
            range_query["range"][range_filter.field]["lt"] = range_filter.value
        elif range_filter.operator == "lte":
            range_query["range"][range_filter.field]["lte"] = range_filter.value
        elif range_filter.operator == "between":
            if isinstance(range_filter.value, list) and len(range_filter.value) == 2:
                range_query["range"][range_filter.field]["gte"] = range_filter.value[0]
                range_query["range"][range_filter.field]["lte"] = range_filter.value[1]
            else:
                raise ValueError("L'opérateur 'between' nécessite une liste de 2 valeurs")
        
        return range_query
    
    def _build_text_search_query(
        self, 
        text_search: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Construit une requête de recherche textuelle."""
        query_text = text_search.get("query", "")
        fields = text_search.get("fields", ["searchable_text"])
        operator = text_search.get("operator", "match")
        
        if operator == "match":
            return {
                "multi_match": {
                    "query": query_text,
                    "fields": fields,
                    "type": "best_fields"
                }
            }
        elif operator == "phrase":
            return {
                "multi_match": {
                    "query": query_text,
                    "fields": fields,
                    "type": "phrase"
                }
            }
        elif operator == "fuzzy":
            return {
                "multi_match": {
                    "query": query_text,
                    "fields": fields,
                    "fuzziness": "AUTO"
                }
            }
        else:
            raise ValueError(f"Opérateur de recherche textuelle non supporté: {operator}")
    
    def _add_search_parameters(
        self, 
        es_query: Dict[str, Any], 
        query_contract: SearchServiceQuery
    ) -> None:
        """Ajoute les paramètres de recherche à la requête."""
        params = query_contract.search_parameters
        
        # Pagination
        es_query["from"] = params.offset
        es_query["size"] = params.limit
        
        # Champs à retourner
        if params.fields:
            es_query["_source"] = params.fields
        
        # Tri par pertinence par défaut
        es_query["sort"] = [{"_score": {"order": "desc"}}]
        
        # Options d'highlighting si demandées
        if query_contract.options.include_highlights:
            es_query["highlight"] = {
                "fields": {
                    "searchable_text": {},
                    "primary_description": {},
                    "merchant_name": {}
                }
            }
    
    def _add_aggregations(
        self, 
        es_query: Dict[str, Any], 
        query_contract: SearchServiceQuery
    ) -> None:
        """Ajoute les agrégations à la requête."""
        aggs = query_contract.aggregations
        es_aggs = {}
        
        # Agrégations par groupement
        for group_field in aggs.group_by:
            es_aggs[f"by_{group_field}"] = {
                "terms": {
                    "field": f"{group_field}.keyword" if group_field in [
                        "category_name", "merchant_name"
                    ] else group_field,
                    "size": 50
                }
            }
            
            # Sous-agrégations pour les métriques
            if aggs.metrics:
                sub_aggs = {}
                for metric_field in aggs.metrics:
                    if "sum" in aggs.types:
                        sub_aggs[f"total_{metric_field}"] = {
                            "sum": {"field": metric_field}
                        }
                    if "avg" in aggs.types:
                        sub_aggs[f"avg_{metric_field}"] = {
                            "avg": {"field": metric_field}
                        }
                    if "count" in aggs.types:
                        sub_aggs[f"count_{metric_field}"] = {
                            "value_count": {"field": metric_field}
                        }
                
                if sub_aggs:
                    es_aggs[f"by_{group_field}"]["aggs"] = sub_aggs
        
        # Agrégations globales
        if aggs.metrics:
            for metric_field in aggs.metrics:
                if "sum" in aggs.types:
                    es_aggs[f"total_{metric_field}"] = {
                        "sum": {"field": metric_field}
                    }
                if "avg" in aggs.types:
                    es_aggs[f"avg_{metric_field}"] = {
                        "avg": {"field": metric_field}
                    }
                if "min" in aggs.types:
                    es_aggs[f"min_{metric_field}"] = {
                        "min": {"field": metric_field}
                    }
                if "max" in aggs.types:
                    es_aggs[f"max_{metric_field}"] = {
                        "max": {"field": metric_field}
                    }
        
        if es_aggs:
            es_query["aggs"] = es_aggs
    
    async def _optimize_query(
        self, 
        es_query: Dict[str, Any], 
        query_contract: SearchServiceQuery
    ) -> Dict[str, Any]:
        """
        Optimise la requête selon le type et la complexité.
        """
        # Optimisation pour requêtes de filtrage simple
        if query_contract.search_parameters.query_type == "filtered_search":
            # Utilisez filter context au lieu de query context pour les booléens
            query = es_query.get("query", {})
            if "bool" in query and not query["bool"].get("must"):
                # Pas de scoring nécessaire, utiliser filter context seulement
                es_query["query"] = {"bool": {"filter": query["bool"]["filter"]}}
        
        # Optimisation pour agrégations
        if query_contract.aggregations and query_contract.aggregations.enabled:
            # Réduire la taille des résultats si on ne fait que des agrégations
            if not query_contract.search_parameters.fields:
                es_query["size"] = 0  # Pas besoin des documents
        
        return es_query
    
    async def _execute_with_retry(
        self, 
        es_query: Dict[str, Any], 
        timeout_ms: int
    ) -> Dict[str, Any]:
        """
        Exécute la requête avec retry logic en cas d'erreur temporaire.
        """
        last_exception = None
        
        for attempt in range(self.max_retry_attempts):
            try:
                result = await self.client.search(
                    index=self.settings.ELASTICSEARCH_INDEX,
                    body=es_query,
                    timeout=f"{timeout_ms}ms",
                    request_timeout=timeout_ms / 1000.0
                )
                return result
                
            except (ConnectionError, TimeoutError) as e:
                last_exception = e
                if attempt < self.max_retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    logger.warning(f"Tentative {attempt + 1} échouée, retry en cours...")
                continue
            except ElasticsearchException as e:
                # Erreurs Elasticsearch non temporaires
                logger.error(f"Erreur Elasticsearch: {str(e)}")
                raise
        
        # Toutes les tentatives ont échoué
        logger.error(f"Échec après {self.max_retry_attempts} tentatives")
        raise last_exception
    
    def _validate_query(self, es_query: Dict[str, Any]) -> None:
        """
        Valide la syntaxe de base d'une requête Elasticsearch.
        """
        if not isinstance(es_query, dict):
            raise ValueError("La requête doit être un dictionnaire")
        
        # Validation de la structure minimale
        if "query" not in es_query:
            raise ValueError("La requête doit contenir une clé 'query'")
        
        # Validation des limites
        size = es_query.get("size", 10)
        if size > self.settings.MAX_SEARCH_SIZE:
            raise ValueError(f"Taille de résultat trop grande: {size} > {self.settings.MAX_SEARCH_SIZE}")
        
        from_param = es_query.get("from", 0)
        if from_param > self.settings.MAX_SEARCH_OFFSET:
            raise ValueError(f"Offset trop grand: {from_param} > {self.settings.MAX_SEARCH_OFFSET}")
    
    def _generate_cache_key(self, query_contract: SearchServiceQuery) -> str:
        """
        Génère une clé de cache pour la requête.
        """
        import hashlib
        import json
        
        # Sérialisation déterministe pour le cache
        contract_dict = query_contract.model_dump()
        # Exclu les métadonnées variables (timestamp, query_id)
        contract_dict.pop("query_metadata", None)
        
        contract_str = json.dumps(contract_dict, sort_keys=True)
        return hashlib.md5(contract_str.encode()).hexdigest()
    
    async def clear_cache(self) -> None:
        """Vide le cache des requêtes."""
        self._query_cache.clear()
        logger.info("Cache des requêtes vidé")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache."""
        return {
            "cached_queries": len(self._query_cache),
            "cache_size_bytes": sum(
                len(str(query)) for query in self._query_cache.values()
            )
        }