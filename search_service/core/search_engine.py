import logging
import time
import json
import asyncio
import math
from datetime import datetime
from collections import deque, defaultdict
from typing import Dict, Any, Optional, List, Union

from search_service.utils.cache import (
    MultiLevelCache,
    generate_cache_key,
)
from config_service.config import settings
from search_service.models.request import SearchRequest
from search_service.models.response import SearchResult, AccountResult
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

# Debug logs supprimés pour réduire la verbosité


class SearchEngine:
    """
    Moteur de recherche unifié utilisant le client Elasticsearch existant
    🔥 VERSION FINALE CORRIGÉE - Support complet _score et highlights avec debug forcing
    """

    def __init__(self, elasticsearch_client=None, cache_enabled: bool = True):
        logger.debug("SearchEngine initialized")
        
        self.elasticsearch_client = elasticsearch_client
        self.query_builder = QueryBuilder()
        self.index_name = "harena_transactions"  # Basé sur votre config existante

        # Cache multi-niveaux pour réponses de recherche
        self.cache_enabled = cache_enabled
        self.cache = MultiLevelCache()
        self.cache_hits = 0
        self.cache_misses = 0

        # Rate limiting (par utilisateur)
        self.requests_per_minute = getattr(
            settings,
            "SEARCH_RATE_LIMIT_REQUESTS_PER_MINUTE",
            getattr(settings, "RATE_LIMIT_REQUESTS_PER_MINUTE", 60),
        )
        self.rate_limit_window = getattr(
            settings,
            "SEARCH_RATE_LIMIT_WINDOW_SECONDS",
            getattr(settings, "RATE_LIMIT_PERIOD", 60),
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
            query=request.query,
            filters=json.dumps(request.filters, sort_keys=True),
            aggregations=json.dumps(request.aggregations, sort_keys=True),
            highlight=json.dumps(request.highlight, sort_keys=True),
            aggregation_only=request.aggregation_only,
            offset=request.offset,
            page_size=request.page_size,
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
        """
        Execute a search and always return a structured response.
        
        🔥 VERSION FINALE CORRIGÉE - Support complet _score et highlights avec debug forcing
        """
        start_time = time.time()
        logger.debug(f"Search request: query='{request.query}', highlights={bool(request.highlight)}")

        try:
            if not self.elasticsearch_client:
                raise RuntimeError("Client Elasticsearch non initialisé")

            # Rate limit check
            self._check_rate_limit(request.user_id)

            # Vérification cache
            cache_key = None
            if self.cache_enabled:
                cache_key = self._generate_cache_key(request)
                cached = await self.cache.get(request.user_id, cache_key)
                if cached:
                    self.cache_hits += 1
                    cached["response_metadata"]["cache_hit"] = True
                    logger.debug("Cache hit - returning cached result")
                    return cached
                else:
                    self.cache_misses += 1

            # Construction requête Elasticsearch avec détection d'index
            if request.aggregations:
                query_info = self.query_builder.build_aggregation_query(
                    request, request.aggregations
                )
                # Pour les agrégations, on récupère les infos d'index aussi
                if isinstance(query_info, dict) and "query" in query_info:
                    es_query = query_info["query"]
                    target_index = query_info.get("target_index", "harena_transactions")
                    search_type = query_info.get("search_type", "transactions")
                else:
                    # Rétrocompatibilité si la méthode n'est pas encore adaptée
                    es_query = query_info
                    target_index = "harena_transactions"
                    search_type = "transactions"
            else:
                # Utiliser build_aggregation_query si des agrégations sont présentes
                if request.aggregations:
                    query_info = self.query_builder.build_aggregation_query(request, request.aggregations)
                    es_query = query_info["query"] if "query" in query_info else query_info
                    # Récupérer les métadonnées depuis build_query
                    base_info = self.query_builder.build_query(request)
                    target_index = base_info["target_index"]
                    search_type = base_info["search_type"]
                else:
                    query_info = self.query_builder.build_query(request)
                    es_query = query_info["query"]
                    target_index = query_info["target_index"]
                    search_type = query_info["search_type"]

            logger.debug(f"🎯 Search type: {search_type}, target index: {target_index}")
            logger.debug(f"Elasticsearch query: {json.dumps(es_query, indent=2)}")

            # Exécution via le client existant avec retry sur l'index approprié
            es_response = await self._execute_search(es_query, request, target_index)
            logger.debug(f"🔥 ES_RESPONSE REÇUE: hits={es_response.get('hits', {}).get('total', {}).get('value', 0)}")

            # 🔥 CORRECTION CRITIQUE : Traitement des résultats avec debug et type de recherche
            processed = self._process_results(es_response, request, search_type)
            logger.debug(f"🔥 RÉSULTATS PROCESSÉS: {len(processed)} objets ({search_type})")
            
            # Sécurité supplémentaire : filtrer par user_id côté application
            results = [r for r in processed if r.user_id == request.user_id]
            logger.debug(f"🔥 RÉSULTATS FILTRÉS PAR USER: {len(results)} objets")

            # Extraire le nombre total de résultats rapporté par Elasticsearch
            total_hits = self._get_total_hits(es_response)
            es_took = es_response.get("took", 0)

            # Logique complexe d'agrégation (inchangée)
            if request.aggregations and not request.aggregation_only:
                current_offset = request.offset + request.limit
                while len(results) + request.offset < total_hits:
                    next_req_data = request.model_dump()
                    next_req_data["offset"] = current_offset
                    next_request = SearchRequest(**next_req_data)
                    page_response = await self._execute_search(es_query, next_request, target_index)
                    es_took += page_response.get("took", 0)
                    page_processed = self._process_results(page_response, next_request, search_type)
                    page_results = [r for r in page_processed if r.user_id == request.user_id]
                    if not page_results:
                        break
                    results.extend(page_results)
                    current_offset += request.limit
            
            aggregations = es_response.get("aggregations")

            # Récupération complète des résultats si agrégations demandées (logique complexe inchangée)
            if request.aggregations and not request.aggregation_only:
                next_offset = request.offset + request.limit
                while request.offset + len(results) < total_hits:
                    next_request = request.model_copy(update={"offset": next_offset, "aggregations": None})
                    es_query_page = self.query_builder.build_query(next_request)
                    es_response_page = await self._execute_search(es_query_page["query"], next_request, target_index)
                    processed_page = self._process_results(es_response_page, next_request, search_type)
                    page_results = [r for r in processed_page if r.user_id == request.user_id]
                    if not page_results:
                        break
                    results.extend(page_results)
                    next_offset += request.limit
                total_results = total_hits
            else:
                total_results = len(results)

            # Calcul temps d'exécution
            execution_time = int((time.time() - start_time) * 1000)

            total_results = total_hits
            returned_results = len(results)
            page_size = request.page_size
            page = (request.offset // page_size) + 1 if page_size else 1
            total_pages = math.ceil(total_results / page_size) if page_size else 0
            has_more_results = (request.offset + returned_results) < total_results

            # 🔥 CORRECTION FINALE CRITIQUE : Sérialisation avec debug forcing
            logger.debug(f"🔥 AVANT SÉRIALISATION: {len(results)} résultats à sérialiser")
            serialized_results = self._serialize_results_with_score_and_highlights_FINAL(results, request)
            logger.debug(f"🔥 APRÈS SÉRIALISATION: {len(serialized_results)} résultats sérialisés")

            response = {
                "results": serialized_results,
                "aggregations": aggregations,
                "success": True,
                "error_message": None,
                "response_metadata": {
                    "query_id": (request.metadata or {}).get("query_id", "unknown"),
                    "response_timestamp": datetime.utcnow().isoformat(),
                    "processing_time_ms": execution_time,
                    "total_results": total_results,
                    "returned_results": returned_results,
                    "has_more_results": has_more_results,
                    "total_pages": total_pages,
                    "page": page,
                    "page_size": page_size,
                    "search_strategy_used": (request.metadata or {}).get(
                        "search_strategy", "standard"
                    ),
                    "elasticsearch_took": es_took,
                    "cache_hit": False,
                },
            }

            if request.metadata and request.metadata.get("debug"):
                response["response_metadata"]["debug_info"] = self._build_debug_info(
                    request, es_query
                )

            # Mise en cache du résultat
            if self.cache_enabled and cache_key:
                await self.cache.set(request.user_id, cache_key, response, ttl=settings.SEARCH_CACHE_TTL)

            logger.debug(f"🔥 SEARCH TERMINÉE: {returned_results}/{total_results} results in {execution_time}ms")
            return response

        except Exception as e:
            logger.error(f"Search failed for user {request.user_id}: {str(e)}")
            logger.error(f"Stacktrace:", exc_info=True)
            execution_time = int((time.time() - start_time) * 1000)
            page_size = request.limit
            page = (request.offset // page_size) + 1

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
                    "page": page,
                    "page_size": page_size,
                    "total_pages": 0,
                    "has_more_results": False,
                    "search_strategy_used": (request.metadata or {}).get(
                        "search_strategy", "standard"
                    ),
                    "elasticsearch_took": 0,
                    "cache_hit": False,
                },
            }

    def _serialize_results_with_score_and_highlights_FINAL(
        self, 
        results: List[Union[SearchResult, AccountResult]], 
        request: SearchRequest
    ) -> List[Dict[str, Any]]:
        """
        🔥 MÉTHODE FINALE - Sérialisation robuste avec _score et highlights + debug forcing
        
        Cette méthode corrige définitivement les problèmes :
        1. _score toujours présent même si score=0
        2. highlights correctement propagées
        3. Debug logging complet pour diagnostiquer
        4. Force les valeurs même si Pydantic a des problèmes
        """
        logger.debug(f"🔥 _serialize_results_with_score_and_highlights_FINAL() - {len(results)} résultats à traiter")
        
        serialized_results = []
        
        for i, result in enumerate(results):
            try:
                # 🔥 CORRECTION : Sérialisation standard avec alias
                result_data = result.model_dump(by_alias=True)
                logger.debug(f"🔥 Résultat {i} sérialisé - keys avant corrections: {list(result_data.keys())}")
                
                # 🔥 CORRECTION CRITIQUE 1: Assurer que _score existe TOUJOURS
                if "_score" not in result_data:
                    score_value = getattr(result, 'score', 0.0)
                    result_data["_score"] = float(score_value) if score_value is not None else 0.0
                    logger.debug(f"🔥 Résultat {i} - _score manquant, ajouté: {result_data['_score']}")
                else:
                    logger.debug(f"🔥 Résultat {i} - _score présent: {result_data['_score']}")
                
                # 🔥 CORRECTION CRITIQUE 2: Assurer que highlights est correctement présent si demandé
                if request.highlight:
                    if "highlights" not in result_data:
                        highlights_value = getattr(result, 'highlights', None)
                        result_data["highlights"] = highlights_value
                        logger.debug(f"🔥 Résultat {i} - highlights manquant, ajouté: {result_data['highlights']}")
                    else:
                        logger.debug(f"🔥 Résultat {i} - highlights présent: {result_data['highlights']}")
                else:
                    logger.debug(f"🔥 Résultat {i} - highlights non demandés")
                
                # 🔥 FORCE DEBUG : Vérification finale
                final_score = result_data.get("_score")
                final_highlights = result_data.get("highlights")
                logger.debug(f"🔥 Résultat {i} FINAL - _score: {final_score}, highlights: {final_highlights}")
                
                serialized_results.append(result_data)
                
            except Exception as e:
                logger.error(f"❌ Failed to serialize result {i}: {str(e)}")
                logger.error(f"   Result type: {type(result)}")
                logger.error(f"   Result attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
                
                # Créer un résultat minimal en cas d'erreur de sérialisation
                if isinstance(result, AccountResult):
                    minimal_result = {
                        "user_id": getattr(result, 'user_id', 0),
                        "account_id": getattr(result, 'account_id', 0),
                        "account_name": f"Serialization error: {str(e)[:50]}",
                        "account_type": "error",
                        "account_balance": 0.0,
                        "account_currency": "EUR",
                        "_score": 0.0,
                        "highlights": None
                    }
                else:
                    minimal_result = {
                        "transaction_id": getattr(result, 'transaction_id', f'error_{i}'),
                        "user_id": getattr(result, 'user_id', 0),
                        "amount": 0.0,
                        "amount_abs": 0.0,
                        "currency_code": "EUR",
                        "transaction_type": "error",
                        "date": "",
                        "primary_description": f"Serialization error: {str(e)[:100]}",
                        "_score": 0.0,
                        "highlights": None
                    }
                serialized_results.append(minimal_result)
                logger.debug(f"🔥 Résultat {i} - Erreur sérialisation, résultat minimal créé")
        
        logger.debug(f"🔥 SÉRIALISATION TERMINÉE: {len(serialized_results)}/{len(results)} résultats traités avec succès")
        return serialized_results
    
    async def _execute_search(self, es_query: Dict[str, Any], request: SearchRequest, target_index: str = None) -> Dict[str, Any]:
        """Exécute la recherche via le client Elasticsearch avec retries et support multi-index."""

        # Utiliser l'index détecté ou fallback sur l'index par défaut
        index_name = target_index or self.index_name
        
        max_retries = settings.INTENT_MAX_RETRIES
        backoff_base = settings.INTENT_BACKOFF_BASE
        retry_statuses = {429, 500, 502, 503, 504}

        for attempt in range(max_retries):
            try:
                es_size = 0 if request.aggregation_only else request.page_size
                es_from = 0 if request.aggregation_only else request.offset

                # Utiliser la méthode search du client existant
                if hasattr(self.elasticsearch_client, "search"):
                    response = await self.elasticsearch_client.search(
                        index=index_name,  # 🎯 Index dynamique
                        body=es_query,
                        size=es_size,
                        from_=es_from,
                    )
                else:
                    # Fallback: requête HTTP directe
                    search_url = f"/{index_name}/_search"  # 🎯 Index dynamique
                    es_query["size"] = es_size
                    es_query["from"] = es_from
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
    
    def _process_results(
        self, 
        es_response: Dict[str, Any], 
        request: Optional[SearchRequest] = None,
        search_type: str = "transactions"
    ) -> List[Union[SearchResult, AccountResult]]:
        """
        🔥 VERSION FINALE - Traite les résultats Elasticsearch en objets SearchResult
        Support robuste _score et highlights avec debug forcing
        """
        results = []

        hits = es_response.get('hits', {}).get('hits', [])
        if not hits:
            logger.debug("No hits returned from Elasticsearch")
            return []
        
        logger.debug(f"🔥 _process_results() - Processing {len(hits)} hits from Elasticsearch (search_type: {search_type})")
        
        for i, hit in enumerate(hits):
            source = hit.get('_source', {})
            score = hit.get('_score')
            highlights = hit.get('highlight')  # ✅ Extraction highlighting
            
            logger.debug(f"🔥 Hit {i} - _score: {score}, highlight: {'present' if highlights else 'none'}")
            
            try:
                if search_type == "accounts":
                    # 🔥 CORRECTION : Créer un AccountResult pour les recherches de comptes
                    result = AccountResult(
                        user_id=int(source.get('user_id', 0)),
                        account_id=int(source.get('account_id', 0)),
                        account_name=str(source.get('account_name', 'Compte inconnu')),
                        account_type=str(source.get('account_type', 'unknown')),
                        account_balance=float(source.get('account_balance', 0.0)),
                        account_currency=str(source.get('account_currency', 'EUR')),
                        
                        # 🔥 CORRECTION SCORE : Métadonnées de recherche
                        score=float(score) if score is not None else 0.0,
                        
                        # 🔥 CORRECTION HIGHLIGHTS : Gestion robuste highlights
                        highlights=highlights if highlights else None
                    )
                    results.append(result)
                    logger.debug(f"🔥 Successfully processed account result {i+1}: {result.account_name} (ID: {result.account_id}) - score={result.score}")
                else:
                    # 🔥 CORRECTION : Gestion robuste de tous les champs pour les transactions
                    result = SearchResult(
                        # Champs obligatoires avec fallbacks robustes
                        transaction_id=str(source.get('transaction_id', f'tx_{i}_{int(time.time())}')),
                        user_id=int(source.get('user_id', 0)),
                        amount=float(source.get('amount', 0.0)),
                        amount_abs=float(source.get('amount_abs', abs(float(source.get('amount', 0.0))))),
                        currency_code=str(source.get('currency_code', 'EUR')),
                        
                        # 🔥 CORRECTION CRITIQUE : Gérer les champs obligatoires qui peuvent être vides
                        transaction_type=str(source.get('transaction_type', 'unknown')),
                        date=str(source.get('date', '')),
                        primary_description=str(source.get('primary_description', 'Description non disponible')),
                        
                        # Champs optionnels - gestion explicite des None
                        account_id=source.get('account_id'),  # Peut être None
                        account_name=source.get('account_name'),
                        account_type=source.get('account_type'),
                        account_balance=source.get('account_balance'),
                        account_currency=source.get('account_currency'),
                        month_year=source.get('month_year'),  # Peut être None
                        weekday=source.get('weekday'),        # Peut être None
                        merchant_name=source.get('merchant_name'),      # Peut être None ou ""
                        category_name=source.get('category_name'),      # Peut être None ou ""
                        operation_type=source.get('operation_type'),    # Peut être None ou ""
                        
                        # 🔥 CORRECTION SCORE : Métadonnées de recherche avec gestion robuste
                        score=float(score) if score is not None else 0.0,
                        
                        # 🔥 CORRECTION HIGHLIGHTS : Gestion robuste highlights
                        highlights=highlights if highlights else None
                    )
                    results.append(result)
                    logger.debug(f"🔥 Successfully processed transaction result {i+1}: {result.transaction_id} - score={result.score} - highlights={'present' if result.highlights else 'none'}")
                
            except ValueError as ve:
                # Erreur de conversion de type (int, float)
                logger.error(f"❌ ValueError processing search result {i+1}: {str(ve)}")
                logger.error(f"   Problematic source data: {json.dumps(source, indent=2, default=str)}")
                logger.error(f"   Score: {score}, Highlights: {highlights}")
                
                # Essayer de créer un résultat minimal avec des types corrects
                try:
                    if search_type == "accounts":
                        minimal_result = AccountResult(
                            user_id=int(source.get('user_id', 0)) if source.get('user_id') is not None else 0,
                            account_id=int(source.get('account_id', 0)) if source.get('account_id') is not None else 0,
                            account_name=f'Erreur conversion: {str(ve)[:50]}',
                            account_type='error',
                            account_balance=0.0,
                            account_currency='EUR',
                            score=0.0,
                            highlights=None
                        )
                    else:
                        minimal_result = SearchResult(
                            transaction_id=str(source.get('transaction_id', f'error_tx_{i}')),
                            user_id=int(source.get('user_id', 0)) if source.get('user_id') is not None else 0,
                            amount=0.0,
                            amount_abs=0.0,
                            currency_code='EUR',
                            transaction_type='error',
                            date='',
                            primary_description=f'Erreur conversion: {str(ve)[:100]}',
                            score=0.0,
                            highlights=None
                        )
                    results.append(minimal_result)
                    logger.warning(f"⚠️ Created minimal {search_type} result for failed conversion {i+1}")
                except Exception as e2:
                    logger.error(f"❌ Failed to create minimal result: {str(e2)}")
                    continue
                    
            except Exception as e:
                # Autres erreurs (validation Pydantic, etc.)
                logger.error(f"❌ General error processing search result {i+1}: {str(e)}")
                logger.error(f"   Exception type: {type(e).__name__}")
                logger.error(f"   Source data: {json.dumps(source, indent=2, default=str)}")
                logger.error(f"   Score: {score}, Highlights: {highlights}")
                
                continue
        
        success_count = len(results)
        total_count = len(hits)
        
        if success_count < total_count:
            logger.warning(f"⚠️ Processed only {success_count}/{total_count} results due to errors")
        else:
            logger.debug(f"🔥 Successfully processed all {success_count}/{total_count} Elasticsearch hits")
        
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
                "page": request.page,
                "page_size": request.page_size,
                "offset": request.offset,
                "highlight_requested": bool(request.highlight),
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