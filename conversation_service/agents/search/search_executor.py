"""
Agent Search Executor - Exécution des requêtes search_service avec résilience
Patterns de protection et gestion d'erreurs robuste
"""
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from conversation_service.core.search_service_client import (
    SearchServiceClient, SearchServiceConfig, SearchServiceError
)
from conversation_service.core.circuit_breaker import CircuitBreakerError
from conversation_service.models.contracts.search_service import (
    SearchQuery, SearchResponse, QueryValidationResult
)
from conversation_service.models.responses.conversation_responses import (
    ProcessingSteps
)

logger = logging.getLogger("conversation_service.search_executor")


class SearchExecutorRequest(BaseModel):
    """Requête pour l'agent Search Executor"""
    
    search_query: SearchQuery
    user_id: int
    request_id: str
    timeout_seconds: Optional[float] = 30.0
    enable_fallback: bool = True
    validate_before_search: bool = True
    auth_token: Optional[str] = None  # Token JWT pour authentification search_service
    
    # Contexte pour monitoring
    context: Dict[str, Any] = Field(default_factory=dict)


class SearchExecutorResponse(BaseModel):
    """Réponse de l'agent Search Executor"""
    
    success: bool
    search_results: Optional[SearchResponse] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    
    # Métriques exécution
    execution_time_ms: int
    fallback_used: bool = False
    circuit_breaker_triggered: bool = False
    retry_attempts: int = 0
    
    # Traçabilité
    request_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Performance
    estimated_performance: str = "unknown"  # optimal, good, poor, failed
    actual_results_count: Optional[int] = None


class SearchResultsCache:
    """Cache simple pour résultats de recherche"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
    
    def _generate_cache_key(self, query: SearchQuery) -> str:
        """Génère une clé de cache pour la requête"""
        # Simplification: hash des éléments principaux
        key_data = {
            "user_id": query.user_id,
            "filters": query.filters,
            "page_size": query.page_size,
            "sort": query.sort
        }
        # Conversion en string stable
        return str(hash(str(sorted(key_data.items()))))
    
    def get(self, query: SearchQuery) -> Optional[SearchResponse]:
        """Récupère du cache si valide"""
        cache_key = self._generate_cache_key(query)
        
        if cache_key in self._cache:
            current_time = time.time()
            cached_time = self._access_times.get(cache_key, 0)
            
            # Vérifier TTL
            if current_time - cached_time <= self.ttl_seconds:
                self._access_times[cache_key] = current_time  # Mise à jour accès
                cached_data = self._cache[cache_key]
                return SearchResponse.model_validate(cached_data["response"])
            else:
                # Expiration - nettoyage
                self._remove_entry(cache_key)
        
        return None
    
    def put(self, query: SearchQuery, response: SearchResponse):
        """Stocke en cache"""
        cache_key = self._generate_cache_key(query)
        current_time = time.time()
        
        # Nettoyage si cache plein
        if len(self._cache) >= self.max_size:
            self._cleanup_oldest()
        
        self._cache[cache_key] = {
            "response": response.model_dump(),
            "cached_at": current_time
        }
        self._access_times[cache_key] = current_time
        
        logger.debug(f"Cached search results for key: {cache_key}")
    
    def _remove_entry(self, cache_key: str):
        """Supprime une entrée du cache"""
        self._cache.pop(cache_key, None)
        self._access_times.pop(cache_key, None)
    
    def _cleanup_oldest(self):
        """Nettoie les entrées les plus anciennes"""
        if not self._access_times:
            return
        
        # Supprimer 20% des plus anciennes
        sorted_keys = sorted(self._access_times.items(), key=lambda x: x[1])
        keys_to_remove = [key for key, _ in sorted_keys[:len(sorted_keys)//5]]
        
        for key in keys_to_remove:
            self._remove_entry(key)
        
        logger.debug(f"Cleaned {len(keys_to_remove)} cache entries")
    
    def clear(self):
        """Vide le cache"""
        self._cache.clear()
        self._access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Statistiques du cache"""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "oldest_entry_age": (
                time.time() - min(self._access_times.values())
                if self._access_times else 0
            )
        }


class SearchExecutor:
    """
    Agent exécuteur de recherches avec résilience complète
    
    Fonctionnalités:
    - Exécution requêtes search_service
    - Circuit breaker et retry automatique
    - Cache intelligent des résultats
    - Fallbacks configurables
    - Métriques détaillées
    """
    
    def __init__(
        self,
        search_service_config: Optional[SearchServiceConfig] = None,
        cache_enabled: bool = True,
        cache_ttl_seconds: int = 300
    ):
        
        # Client search_service
        self.search_client = SearchServiceClient(search_service_config)
        
        # Cache résultats
        self.cache_enabled = cache_enabled
        self.results_cache = SearchResultsCache(ttl_seconds=cache_ttl_seconds) if cache_enabled else None
        
        # Métriques agent
        self.total_requests = 0
        self.cached_responses = 0
        self.successful_searches = 0
        self.failed_searches = 0
        self.fallback_responses = 0
        
        # Configuration fallbacks
        self.fallback_strategies = [
            self._fallback_cached_results,
            self._fallback_simplified_query,
            self._fallback_empty_results
        ]
        
        logger.info("Search Executor agent initialisé avec résilience complète")
    
    async def handle_search_request(
        self,
        message: SearchExecutorRequest,
        ctx: Optional[Any] = None
    ) -> SearchExecutorResponse:
        """
        Traite une demande de recherche
        
        Args:
            message: Requête de recherche
            ctx: Contexte optionnel (non utilisé)
            
        Returns:
            Résultats de recherche ou erreur
        """
        start_time = time.time()
        self.total_requests += 1
        
        logger.info(
            f"Search request pour user_id={message.user_id}, "
            f"request_id={message.request_id}"
        )
        
        try:
            # Vérification cache si activé
            cached_result = None
            if self.cache_enabled and self.results_cache:
                cached_result = self.results_cache.get(message.search_query)
                if cached_result:
                    self.cached_responses += 1
                    logger.info(f"Cache hit pour request_id={message.request_id}")
                    
                    return SearchExecutorResponse(
                        success=True,
                        search_results=cached_result,
                        execution_time_ms=int((time.time() - start_time) * 1000),
                        request_id=message.request_id,
                        estimated_performance="optimal"
                    )
            
            # Exécution recherche avec récupération complète si nécessaire
            async with self.search_client:
                search_response = await self.search_client.search(
                    query=message.search_query,
                    validate_query=message.validate_before_search,
                    enable_fallback=message.enable_fallback,
                    auth_token=message.auth_token
                )
                
                # Si il y a plus de résultats que retournés, récupérer tous les résultats
                search_response = await self._ensure_complete_results(
                    search_response, message, start_time
                )
            
            # Cache résultat si réussi
            if self.cache_enabled and self.results_cache:
                self.results_cache.put(message.search_query, search_response)
            
            self.successful_searches += 1
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Estimation performance basée sur temps de réponse
            performance = self._estimate_performance(execution_time_ms, search_response)
            
            logger.info(
                f"Search successful pour request_id={message.request_id}, "
                f"results={search_response.total_hits}, time={execution_time_ms}ms"
            )
            
            return SearchExecutorResponse(
                success=True,
                search_results=search_response,
                execution_time_ms=execution_time_ms,
                request_id=message.request_id,
                estimated_performance=performance,
                actual_results_count=search_response.total_hits
            )
        
        except CircuitBreakerError as e:
            # Circuit breaker ouvert
            self.failed_searches += 1
            logger.warning(f"Circuit breaker ouvert pour request_id={message.request_id}")
            
            # Tentative fallback
            fallback_result = await self._execute_fallback_strategies(
                message, "circuit_breaker_open", start_time
            )
            if fallback_result:
                return fallback_result
            
            return SearchExecutorResponse(
                success=False,
                error_message="Search service temporarily unavailable (circuit breaker open)",
                error_type="circuit_breaker_open",
                execution_time_ms=int((time.time() - start_time) * 1000),
                circuit_breaker_triggered=True,
                request_id=message.request_id,
                estimated_performance="failed"
            )
        
        except SearchServiceError as e:
            # Erreur search service
            self.failed_searches += 1
            logger.error(f"Search service error pour request_id={message.request_id}: {str(e)}")
            
            # Tentative fallback
            fallback_result = await self._execute_fallback_strategies(
                message, type(e).__name__.lower(), start_time
            )
            if fallback_result:
                return fallback_result
            
            return SearchExecutorResponse(
                success=False,
                error_message=str(e),
                error_type=type(e).__name__,
                execution_time_ms=int((time.time() - start_time) * 1000),
                request_id=message.request_id,
                estimated_performance="failed"
            )
        
        except Exception as e:
            # Erreur inattendue
            self.failed_searches += 1
            logger.error(f"Unexpected error pour request_id={message.request_id}: {str(e)}")
            
            return SearchExecutorResponse(
                success=False,
                error_message=f"Unexpected error: {str(e)}",
                error_type="unexpected_error",
                execution_time_ms=int((time.time() - start_time) * 1000),
                request_id=message.request_id,
                estimated_performance="failed"
            )
    
    async def _execute_fallback_strategies(
        self,
        original_request: SearchExecutorRequest,
        error_context: str,
        start_time: float
    ) -> Optional[SearchExecutorResponse]:
        """Exécute les stratégies de fallback"""
        
        if not original_request.enable_fallback:
            return None
        
        logger.info(f"Executing fallback strategies for error: {error_context}")
        
        for strategy in self.fallback_strategies:
            try:
                fallback_result = await strategy(original_request, error_context)
                if fallback_result:
                    self.fallback_responses += 1
                    fallback_result.fallback_used = True
                    fallback_result.execution_time_ms = int((time.time() - start_time) * 1000)
                    
                    logger.info(f"Fallback successful avec {strategy.__name__}")
                    return fallback_result
            except Exception as e:
                logger.warning(f"Fallback strategy {strategy.__name__} failed: {str(e)}")
                continue
        
        logger.warning("Toutes les stratégies de fallback ont échoué")
        return None
    
    async def _fallback_cached_results(
        self,
        request: SearchExecutorRequest,
        error_context: str
    ) -> Optional[SearchExecutorResponse]:
        """Fallback: Résultats en cache"""
        
        if not self.cache_enabled or not self.results_cache:
            return None
        
        # Recherche cache élargi (sans TTL strict)
        cache_key = self.results_cache._generate_cache_key(request.search_query)
        if cache_key in self.results_cache._cache:
            cached_data = self.results_cache._cache[cache_key]
            cached_response = SearchResponse.model_validate(cached_data["response"])
            
            return SearchExecutorResponse(
                success=True,
                search_results=cached_response,
                request_id=request.request_id,
                estimated_performance="good",  # Cache donc rapide
                actual_results_count=cached_response.total_hits
            )
        
        return None
    
    async def _fallback_simplified_query(
        self,
        request: SearchExecutorRequest,
        error_context: str
    ) -> Optional[SearchExecutorResponse]:
        """Fallback: Requête simplifiée"""
        
        # Créer requête minimale
        simplified_query = SearchQuery(
            user_id=request.search_query.user_id,
            filters={"user_id": request.search_query.user_id},
            sort=[{"date": {"order": "desc"}}],
            page_size=min(request.search_query.page_size or 20, 10)
        )
        
        try:
            async with self.search_client:
                search_response = await self.search_client.search(
                    query=simplified_query,
                    validate_query=False,
                    enable_fallback=False,  # Pas de fallback récursif
                    auth_token=original_request.auth_token
                )
            
            return SearchExecutorResponse(
                success=True,
                search_results=search_response,
                request_id=request.request_id,
                estimated_performance="poor",  # Requête simplifiée
                actual_results_count=search_response.total_hits
            )
        
        except Exception:
            return None
    
    async def _fallback_empty_results(
        self,
        request: SearchExecutorRequest,
        error_context: str
    ) -> Optional[SearchExecutorResponse]:
        """Fallback: Résultats vides mais valides"""
        
        empty_response = SearchResponse(
            hits=[],
            total_hits=0,
            aggregations={},
            took_ms=0,
            query_id=f"fallback_{request.request_id}",
            timestamp=datetime.now(timezone.utc)
        )
        
        return SearchExecutorResponse(
            success=True,
            search_results=empty_response,
            request_id=request.request_id,
            estimated_performance="poor",
            actual_results_count=0,
            execution_time_ms=0,
            fallback_used=True
        )
    
    def _estimate_performance(
        self,
        execution_time_ms: int,
        search_response: SearchResponse
    ) -> str:
        """Estime la performance basée sur les métriques"""
        
        if execution_time_ms < 500:  # < 500ms
            return "optimal"
        elif execution_time_ms < 2000:  # < 2s
            return "good"
        else:
            return "poor"
    
    async def get_health_status(self) -> Dict[str, Any]:
        """État de santé de l'agent"""
        
        # Santé du client search service
        search_health = await self.search_client.health_check()
        
        return {
            "agent_status": "healthy",
            "search_service": search_health,
            "cache_enabled": self.cache_enabled,
            "cache_stats": self.results_cache.get_stats() if self.results_cache else {},
            "metrics": self.get_metrics()
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Métriques détaillées de l'agent"""
        
        base_metrics = {
            "total_requests": self.total_requests,
            "cached_responses": self.cached_responses,
            "successful_searches": self.successful_searches,
            "failed_searches": self.failed_searches,
            "fallback_responses": self.fallback_responses,
            "cache_hit_rate": (
                self.cached_responses / self.total_requests 
                if self.total_requests > 0 else 0.0
            ),
            "success_rate": (
                self.successful_searches / self.total_requests 
                if self.total_requests > 0 else 0.0
            ),
            "fallback_rate": (
                self.fallback_responses / self.total_requests 
                if self.total_requests > 0 else 0.0
            )
        }
        
        # Ajouter métriques search client
        search_client_metrics = self.search_client.get_metrics()
        base_metrics["search_client"] = search_client_metrics
        
        return base_metrics
    
    async def _ensure_complete_results(
        self, 
        initial_response: SearchResponse, 
        message: SearchExecutorRequest,
        start_time: float
    ) -> SearchResponse:
        """
        S'assure que tous les résultats sont récupérés si nécessaire
        
        Logique adaptative pour récupérer tous les résultats disponibles
        dans la limite des 128K tokens de DeepSeek.
        """
        # Si on n'a pas d'agrégation uniquement et qu'il y a plus de résultats
        if (not message.search_query.aggregation_only and 
            initial_response.total_hits > len(initial_response.hits)):
            
            total_available = initial_response.total_hits
            current_hits = len(initial_response.hits)
            
            logger.info(f"Résultats partiels détectés: {current_hits}/{total_available} - récupération complète")
            
            # Estimer si on peut récupérer tous les résultats dans la limite de tokens
            # Estimation conservatrice: ~100 tokens par transaction
            estimated_tokens = total_available * 100
            
            # Limite DeepSeek: 128K tokens, gardons une marge pour le prompt système
            max_safe_tokens = 120000
            
            if estimated_tokens <= max_safe_tokens and total_available <= 1000:
                # Récupérer tous les résultats en une fois
                logger.info(f"Récupération complète de {total_available} résultats (estimation: {estimated_tokens} tokens)")
                
                # Créer nouvelle requête avec taille augmentée
                complete_query = message.search_query.model_copy(deep=True)
                complete_query.page_size = min(total_available, 1000)  # Limite à 1000 max
                
                try:
                    complete_request = SearchExecutorRequest(
                        search_query=complete_query,
                        user_id=message.user_id,
                        request_id=f"{message.request_id}_complete",
                        timeout_seconds=message.timeout_seconds,
                        enable_fallback=message.enable_fallback,
                        validate_before_search=False,  # Déjà validé
                        context=message.context
                    )
                    
                    # Faire la requête complète
                    complete_response = await self.search_client.search(
                        query=complete_query,
                        validate_query=False,
                        enable_fallback=message.enable_fallback,
                        auth_token=message.auth_token
                    )
                    
                    logger.info(f"✅ Récupération complète réussie: {len(complete_response.hits)} résultats")
                    return complete_response
                    
                except Exception as e:
                    logger.warning(f"⚠️ Échec récupération complète: {str(e)} - utilisation résultats partiels")
                    # Retourner la réponse initiale en cas d'échec
                    return initial_response
            else:
                logger.info(f"Trop de résultats pour récupération complète ({total_available}, ~{estimated_tokens} tokens) - utilisation pagination intelligente")
                
                # TODO: Implémenter pagination intelligente si nécessaire
                # Pour l'instant, on augmente juste la page_size à 500
                if current_hits < 500 and total_available > current_hits:
                    try:
                        complete_query = message.search_query.model_copy(deep=True)
                        complete_query.page_size = min(500, total_available)
                        
                        complete_response = await self.search_client.search(
                            query=complete_query,
                            validate_query=False,
                            enable_fallback=message.enable_fallback,
                            auth_token=message.auth_token
                        )
                        
                        logger.info(f"✅ Récupération partielle étendue: {len(complete_response.hits)}/{total_available} résultats")
                        return complete_response
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Échec récupération étendue: {str(e)}")
                        
                return initial_response
        
        return initial_response

    async def clear_cache(self):
        """Vide le cache de résultats"""
        if self.results_cache:
            self.results_cache.clear()
            logger.info("Cache de résultats vidé")
    
    async def cleanup(self):
        """Nettoyage des ressources"""
        await self.search_client.close()
        if self.results_cache:
            self.results_cache.clear()
        logger.info("Search Executor agent nettoyé")