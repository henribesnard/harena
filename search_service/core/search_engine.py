import logging
import time
import json
import os
import asyncio
from datetime import datetime
from collections import deque, defaultdict
from typing import Dict, Any

from conversation_service.utils.cache import (
    MultiLevelCache,
    generate_cache_key,
)
from search_service.models.request import SearchRequest
from search_service.models.response import SearchResult
from .query_builder import QueryBuilder


class RateLimitExceeded(Exception):
    """Raised when the rate limit for a user is exceeded."""

    def __init__(self, limit: int, window: int):
        self.limit = limit
        self.window = window
        super().__init__(
            f"Rate limit exceeded: {limit} requests per {window} seconds"
        )


class ElasticsearchHTTPError(Exception):
    """Represents an HTTP error from Elasticsearch."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Elasticsearch error {status_code}: {message}")

logger = logging.getLogger(__name__)

class SearchEngine:
    """Moteur de recherche unifié utilisant le client Elasticsearch existant"""

    def __init__(self, elasticsearch_client=None, cache_enabled: bool = True):
        self.elasticsearch_client = elasticsearch_client
        self.query_builder = QueryBuilder()
        self.index_name = "harena_transactions"  # Basé sur votre config existante

        # Cache multi-niveaux pour réponses de recherche
        self.cache_enabled = cache_enabled
        self.cache = MultiLevelCache()
        self.cache_hits = 0
        self.cache_misses = 0

        # Rate limiting (par utilisateur)
        self.requests_per_minute = int(
            os.getenv("SEARCH_RATE_LIMIT_REQUESTS_PER_MINUTE", "60")
        )
        self.rate_limit_window = int(
            os.getenv("SEARCH_RATE_LIMIT_WINDOW_SECONDS", "60")
        )
        self._rate_limit_storage: Dict[int, deque] = defaultdict(deque)
    
    def set_elasticsearch_client(self, client):
        """Définit le client Elasticsearch à utiliser"""
        self.elasticsearch_client = client
        if hasattr(client, 'index_name'):
            self.index_name = client.index_name

    def _generate_cache_key(self, request: SearchRequest) -> str:
        """Génère une clé de cache basée sur les paramètres de la requête."""
        return generate_cache_key(
            "search",
            user_id=request.user_id,
            query=request.query,
            filters=json.dumps(request.filters, sort_keys=True),
            offset=request.offset,
            limit=request.limit,
        )

    def _check_rate_limit(self, user_id: int) -> None:
        """Applique un rate limiting simple par utilisateur."""
        requests = self._rate_limit_storage[user_id]
        now = time.time()
        while requests and now - requests[0] >= self.rate_limit_window:
            requests.popleft()
        if len(requests) >= self.requests_per_minute:
            raise RateLimitExceeded(
                self.requests_per_minute, self.rate_limit_window
            )
        requests.append(now)

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de cache et de rate limiting."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0
        return {
            "cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": hit_rate,
            },
            "rate_limit": {
                "requests_per_minute": self.requests_per_minute,
                "window_seconds": self.rate_limit_window,
            },
        }

    async def clear_cache(self) -> None:
        """Vide le cache et réinitialise les statistiques associées."""
        if self.cache:
            await self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    async def search(self, request: SearchRequest) -> Dict[str, Any]:
        """Execute a search and always return a structured response.

        This method now guarantees that even in error scenarios a predictable
        response structure is returned.  This structure contains a
        ``response_metadata`` block with default timings and counts so that
        higher level services (e.g. ``SearchServiceResponse``) can safely
        parse the output.
        """

        start_time = time.time()

        try:
            if not self.elasticsearch_client:
                raise RuntimeError("Client Elasticsearch non initialisé")

            # Rate limit check
            self._check_rate_limit(request.user_id)

            # Vérification cache
            cache_key = None
            if self.cache_enabled:
                cache_key = self._generate_cache_key(request)
                cached = await self.cache.get(cache_key)
                if cached:
                    self.cache_hits += 1
                    cached["response_metadata"]["cache_hit"] = True
                    return cached
                else:
                    self.cache_misses += 1

            # Construction requête Elasticsearch
            if request.aggregations:
                es_query = self.query_builder.build_aggregation_query(
                    request, request.aggregations
                )
            else:
                es_query = self.query_builder.build_query(request)

            logger.debug(
                f"Executing search for user {request.user_id} with query: '{request.query}'"
            )

            # Exécution via le client existant avec retry
            es_response = await self._execute_search(es_query, request)

            # Traitement des résultats
            results = self._process_results(es_response)

            # Calcul temps d'exécution
            execution_time = int((time.time() - start_time) * 1000)

            total_results = self._get_total_hits(es_response)
            returned_results = len(results)

            response = {
                "results": [r.model_dump() for r in results],
                "aggregations": es_response.get("aggregations"),
                "success": True,
                "error_message": None,
                "response_metadata": {
                    "query_id": (request.metadata or {}).get("query_id", "unknown"),
                    "response_timestamp": datetime.utcnow().isoformat(),
                    "processing_time_ms": execution_time,
                    "total_results": total_results,
                    "returned_results": returned_results,
                    "has_more_results": total_results > (returned_results + request.offset),
                    "search_strategy_used": (request.metadata or {}).get(
                        "search_strategy", "standard"
                    ),
                    "elasticsearch_took": es_response.get("took", 0),
                    "cache_hit": False,
                },
            }

            if request.metadata.get("debug"):
                response["response_metadata"]["debug_info"] = self._build_debug_info(
                    request, es_query
                )

            # Mise en cache du résultat
            if self.cache_enabled:
                cache_ttl = int(os.getenv("SEARCH_CACHE_TTL", "30"))
                await self.cache.set(cache_key, response, ttl=cache_ttl)

            logger.info(
                f"Search completed: {returned_results}/{total_results} results in {execution_time}ms"
            )
            return response

        except Exception as e:
            logger.error(f"Search failed for user {request.user_id}: {str(e)}")
            execution_time = int((time.time() - start_time) * 1000)

            return {
                "results": [],
                "aggregations": None,
                "success": False,
                "error_message": str(e),
                "response_metadata": {
                    "query_id": (request.metadata or {}).get("query_id", "unknown"),
                    "response_timestamp": datetime.utcnow().isoformat(),
                    "processing_time_ms": execution_time,
                    "total_results": 0,
                    "returned_results": 0,
                    "has_more_results": False,
                    "search_strategy_used": (request.metadata or {}).get(
                        "search_strategy", "standard"
                    ),
                    "elasticsearch_took": 0,
                    "cache_hit": False,
                },
            }
    
    async def _execute_search(self, es_query: Dict[str, Any], request: SearchRequest) -> Dict[str, Any]:
        """Exécute la recherche via le client Elasticsearch avec retries."""

        max_retries = int(os.getenv("SEARCH_MAX_RETRIES", "3"))
        backoff_base = float(os.getenv("SEARCH_BACKOFF_BASE", "0.5"))
        retry_statuses = {429, 500, 502, 503, 504}

        for attempt in range(max_retries):
            try:
                # Utiliser la méthode search du client existant
                if hasattr(self.elasticsearch_client, "search"):
                    response = await self.elasticsearch_client.search(
                        index=self.index_name,
                        body=es_query,
                        size=request.limit,
                        from_=request.offset,
                    )
                else:
                    # Fallback: requête HTTP directe
                    search_url = f"/{self.index_name}/_search"
                    es_query["size"] = request.limit
                    es_query["from"] = request.offset
                    async with self.elasticsearch_client.session.post(
                        f"{self.elasticsearch_client.base_url}{search_url}",
                        json=es_query,
                    ) as resp:
                        if resp.status != 200:
                            error_text = await resp.text()
                            raise ElasticsearchHTTPError(resp.status, error_text)
                        response = await resp.json()

                return response

            except ElasticsearchHTTPError as e:
                if e.status_code in retry_statuses and attempt < max_retries - 1:
                    wait = backoff_base * (2 ** attempt)
                    logger.warning(
                        f"Elasticsearch HTTP error {e.status_code}, retrying in {wait:.1f}s"
                    )
                    await asyncio.sleep(wait)
                    continue
                raise
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = backoff_base * (2 ** attempt)
                    logger.warning(
                        f"Elasticsearch request failed: {e}, retrying in {wait:.1f}s"
                    )
                    await asyncio.sleep(wait)
                    continue
                raise
    
    def _process_results(self, es_response: Dict[str, Any]) -> list[SearchResult]:
        """
        Traite les résultats Elasticsearch en objets SearchResult
        VERSION CORRIGÉE - Robuste aux données manquantes/nulles
        """
        results = []
        
        hits = es_response.get('hits', {}).get('hits', [])
        logger.debug(f"Processing {len(hits)} hits from Elasticsearch")
        
        for i, hit in enumerate(hits):
            source = hit.get('_source', {})
            score = hit.get('_score')
            
            try:
                # ✅ CORRECTION : Gestion robuste de tous les champs avec valeurs par défaut sécurisées
                result = SearchResult(
                    # Champs obligatoires avec fallbacks robustes
                    transaction_id=str(source.get('transaction_id', f'tx_{i}_{int(time.time())}')),
                    user_id=int(source.get('user_id', 0)),
                    amount=float(source.get('amount', 0.0)),
                    amount_abs=float(source.get('amount_abs', abs(float(source.get('amount', 0.0))))),
                    currency_code=str(source.get('currency_code', 'EUR')),
                    
                    # ✅ CORRECTION CRITIQUE : Gérer les champs obligatoires qui peuvent être vides
                    transaction_type=str(source.get('transaction_type', 'unknown')),
                    date=str(source.get('date', '')),
                    primary_description=str(source.get('primary_description', 'Description non disponible')),
                    
                    # Champs optionnels - gestion explicite des None
                    account_id=source.get('account_id'),  # Peut être None
                    month_year=source.get('month_year'),  # Peut être None
                    weekday=source.get('weekday'),        # Peut être None
                    merchant_name=source.get('merchant_name'),      # Peut être None ou ""
                    category_name=source.get('category_name'),      # Peut être None ou ""
                    operation_type=source.get('operation_type'),    # Peut être None ou ""
                    
                    # Métadonnées de recherche
                    score=float(score) if score is not None else 0.0,
                    highlights=hit.get('highlight')
                )
                results.append(result)
                
                # Log de succès pour debug
                logger.debug(f"✅ Successfully processed result {i+1}: {result.transaction_id} - {result.primary_description[:50]}")
                
            except ValueError as ve:
                # Erreur de conversion de type (int, float)
                logger.error(f"❌ ValueError processing search result {i+1}: {str(ve)}")
                logger.error(f"   Problematic source data: {json.dumps(source, indent=2, default=str)}")
                logger.error(f"   Score: {score}")
                
                # Essayer de créer un résultat minimal avec des types corrects
                try:
                    minimal_result = SearchResult(
                        transaction_id=str(source.get('transaction_id', f'error_tx_{i}')),
                        user_id=int(source.get('user_id', 0)) if source.get('user_id') is not None else 0,
                        amount=0.0,
                        amount_abs=0.0,
                        currency_code='EUR',
                        transaction_type='error',
                        date='',
                        primary_description=f'Erreur conversion: {str(ve)[:100]}',
                        score=0.0
                    )
                    results.append(minimal_result)
                    logger.warning(f"⚠️ Created minimal result for failed conversion {i+1}")
                except Exception as e2:
                    logger.error(f"❌ Failed to create minimal result: {str(e2)}")
                    continue
                    
            except Exception as e:
                # Autres erreurs (validation Pydantic, etc.)
                logger.error(f"❌ General error processing search result {i+1}: {str(e)}")
                logger.error(f"   Exception type: {type(e).__name__}")
                logger.error(f"   Source data: {json.dumps(source, indent=2, default=str)}")
                logger.error(f"   Score: {score}")
                
                # Log des champs spécifiques pour debug
                logger.error(f"   transaction_id: {repr(source.get('transaction_id'))}")
                logger.error(f"   user_id: {repr(source.get('user_id'))}")
                logger.error(f"   transaction_type: {repr(source.get('transaction_type'))}")
                logger.error(f"   primary_description: {repr(source.get('primary_description'))}")
                logger.error(f"   amount: {repr(source.get('amount'))}")
                
                continue
        
        success_count = len(results)
        total_count = len(hits)
        
        if success_count < total_count:
            logger.warning(f"⚠️ Processed only {success_count}/{total_count} results due to errors")
        else:
            logger.info(f"✅ Successfully processed all {success_count}/{total_count} Elasticsearch hits")
        
        return results
    
    def _get_total_hits(self, es_response: Dict[str, Any]) -> int:
        """Extrait le nombre total de résultats"""
        hits = es_response.get('hits', {})
        total = hits.get('total', 0)
        
        # Gestion des différents formats de total d'Elasticsearch
        if isinstance(total, dict):
            return total.get('value', 0)
        else:
            return int(total)
    
    def _build_debug_info(self, request: SearchRequest, es_query: Dict[str, Any]) -> Dict[str, Any]:
        """Construit les informations de debug"""
        return {
            "original_request": {
                "user_id": request.user_id,
                "query": request.query,
                "filters": request.filters,
                "limit": request.limit,
                "offset": request.offset
            },
            "elasticsearch_query": es_query,
            "index_used": self.index_name
        }
    
    async def count(self, request: SearchRequest) -> int:
        """Compte le nombre total de résultats sans les récupérer"""
        if not self.elasticsearch_client:
            raise RuntimeError("Client Elasticsearch non initialisé")
        
        try:
            # Construction requête de comptage
            count_query = {
                "query": self.query_builder.build_query(request)["query"]
            }
            
            # Exécution via le client
            if hasattr(self.elasticsearch_client, 'count'):
                response = await self.elasticsearch_client.count(
                    index=self.index_name,
                    body=count_query
                )
                return response.get('count', 0)
            else:
                # Fallback: utiliser _count endpoint
                count_url = f"/{self.index_name}/_count"
                async with self.elasticsearch_client.session.post(
                    f"{self.elasticsearch_client.base_url}{count_url}",
                    json=count_query
                ) as resp:
                    if resp.status != 200:
                        return 0
                    response = await resp.json()
                    return response.get('count', 0)
                    
        except Exception as e:
            logger.error(f"Count failed: {e}")
            return 0
