"""
Query Executor - Agent Logique Phase 3
Architecture v2.0 - Composant déterministe

Responsabilité : Exécution des requêtes search_service
- Évolution de search_executor.py sans AutoGen
- Gestion connexions et timeout
- Traitement des réponses et erreurs
- Cache des résultats avec TTL
"""

import logging
import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

@dataclass
class QueryExecutionRequest:
    """Requête d'exécution de query"""
    query: Dict[str, Any]
    user_id: int
    timeout_ms: int = 5000
    cache_ttl_seconds: Optional[int] = 300  # 5 minutes
    retry_count: int = 2
    jwt_token: Optional[str] = None

@dataclass
class QueryExecutionResult:
    """Résultat d'exécution de query"""
    success: bool
    results: List[Dict[str, Any]]
    total_hits: int
    aggregations: Optional[Dict[str, Any]]
    processing_time_ms: int
    search_service_time_ms: int
    cached: bool = False
    error_message: Optional[str] = None
    retry_count: int = 0

@dataclass
class CacheEntry:
    """Entrée de cache pour les résultats"""
    result: QueryExecutionResult
    cached_at: datetime
    ttl_seconds: int
    
    @property
    def is_expired(self) -> bool:
        return datetime.now() > self.cached_at + timedelta(seconds=self.ttl_seconds)

class QueryExecutor:
    """
    Agent logique pour exécution de requêtes search_service
    
    Gère les connexions, cache et traitement des réponses
    """
    
    def __init__(
        self,
        search_service_url: str,
        timeout_seconds: int = 10,
        max_concurrent_requests: int = 50,
        enable_cache: bool = True
    ):
        self.search_service_url = search_service_url.rstrip('/')
        self.timeout_seconds = timeout_seconds
        self.max_concurrent_requests = max_concurrent_requests
        self.enable_cache = enable_cache
        
        # Cache des résultats
        self._result_cache: Dict[str, CacheEntry] = {}
        
        # Semaphore pour limiter les requêtes concurrentes
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        # Session HTTP réutilisable
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Statistiques
        self.stats = {
            "queries_executed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "timeouts": 0,
            "retries": 0
        }
        
        logger.info(f"QueryExecutor initialisé - URL: {self.search_service_url}")
    
    async def initialize(self) -> bool:
        """Initialise la session HTTP"""
        try:
            connector = aiohttp.TCPConnector(
                limit=self.max_concurrent_requests,
                limit_per_host=self.max_concurrent_requests,
                ttl_dns_cache=300,
                use_dns_cache=True
            )
            
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'conversation-service/2.0'
                }
            )
            
            logger.info("QueryExecutor session HTTP initialisée")
            return True
            
        except Exception as e:
            logger.error(f"Erreur initialisation QueryExecutor: {str(e)}")
            return False
    
    async def execute_query(self, request: QueryExecutionRequest) -> QueryExecutionResult:
        """
        Exécute une requête sur search_service avec cache et retry
        
        Args:
            request: Requête d'exécution avec query et paramètres
            
        Returns:
            QueryExecutionResult avec résultats ou erreur
        """
        start_time = datetime.now()
        
        try:
            # 1. Vérification cache
            if self.enable_cache and request.cache_ttl_seconds:
                cache_key = self._generate_cache_key(request.query, request.user_id)
                cached_result = self._get_cached_result(cache_key)
                
                if cached_result:
                    self.stats["cache_hits"] += 1
                    cached_result.cached = True
                    cached_result.processing_time_ms = self._get_processing_time(start_time)
                    return cached_result
                
                self.stats["cache_misses"] += 1
            
            # 2. Exécution avec retry
            result = await self._execute_with_retry(request, start_time)
            
            # 3. Mise en cache si succès
            if (result.success and self.enable_cache and 
                request.cache_ttl_seconds and cache_key):
                self._cache_result(cache_key, result, request.cache_ttl_seconds)
            
            # 4. Statistiques
            self.stats["queries_executed"] += 1
            if not result.success:
                self.stats["errors"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur inattendue exécution query: {str(e)}")
            return QueryExecutionResult(
                success=False,
                results=[],
                total_hits=0,
                aggregations=None,
                processing_time_ms=self._get_processing_time(start_time),
                search_service_time_ms=0,
                error_message=f"Unexpected error: {str(e)}"
            )
    
    async def _execute_with_retry(
        self, 
        request: QueryExecutionRequest,
        start_time: datetime
    ) -> QueryExecutionResult:
        """Exécution avec retry automatique"""
        
        last_error = None
        
        for attempt in range(request.retry_count + 1):
            try:
                async with self._semaphore:
                    result = await self._execute_single_query(request, start_time)
                    
                    if result.success:
                        result.retry_count = attempt
                        return result
                    
                    last_error = result.error_message
                    
                    # Retry seulement sur certaines erreurs
                    if not self._should_retry(result.error_message):
                        return result
                    
                    # Backoff exponentiel
                    if attempt < request.retry_count:
                        wait_time = (2 ** attempt) * 0.1  # 100ms, 200ms, 400ms...
                        await asyncio.sleep(wait_time)
                        self.stats["retries"] += 1
                        logger.debug(f"Retry attempt {attempt + 1} après {wait_time}s")
                        
            except asyncio.TimeoutError:
                self.stats["timeouts"] += 1
                last_error = "Request timeout"
                logger.warning(f"Timeout attempt {attempt + 1}")
                
            except Exception as e:
                last_error = str(e)
                logger.error(f"Error attempt {attempt + 1}: {last_error}")
        
        # Tous les attempts ont échoué
        return QueryExecutionResult(
            success=False,
            results=[],
            total_hits=0,
            aggregations=None,
            processing_time_ms=self._get_processing_time(start_time),
            search_service_time_ms=0,
            error_message=f"All retry attempts failed. Last error: {last_error}",
            retry_count=request.retry_count
        )
    
    async def _execute_single_query(
        self,
        request: QueryExecutionRequest,
        start_time: datetime
    ) -> QueryExecutionResult:
        """Exécution d'une requête HTTP avec récupération de toutes les pages"""
        
        if not self._session:
            await self.initialize()
        
        search_start = datetime.now()
        
        try:
            # Construction URL
            search_url = urljoin(self.search_service_url, '/api/v1/search/search')
            
            # Transformation du format template vers SearchRequest
            query_data = request.query
            
            # Vérification : si la query est None (intents conversationnels), retourner résultat vide
            if query_data is None:
                return QueryExecutionResult(
                    success=True,
                    results=[],
                    total_hits=0,
                    aggregations=None,
                    processing_time_ms=self._get_processing_time(start_time),
                    search_service_time_ms=0
                )
            
            # Extraire les éléments de la query générée par le template
            base_payload = {
                "user_id": request.user_id,
                "query": "",  # Requête textuelle vide par défaut
                "filters": query_data.get("filters", {}),
                "sort": query_data.get("sort", []),
                "page_size": query_data.get("page_size", 50),
                "page": 1,  # Commencer à la page 1
                "metadata": {"source": "conversation_service"}
            }
            
            # Ajouter les agrégations si présentes dans la query du template
            if "aggregations" in query_data:
                base_payload["aggregations"] = query_data["aggregations"]
            
            # Si il y a une requête textuelle dans le template
            if "query" in query_data and isinstance(query_data["query"], str):
                base_payload["query"] = query_data["query"]
            
            # Headers avec authentification JWT
            headers = {}
            if request.jwt_token:
                headers["Authorization"] = f"Bearer {request.jwt_token}"
            
            # Log de la query pour debugging
            logger.info(f"Query envoyée au search service: {json.dumps(base_payload, indent=2)}")
            
            # Variables pour collecter tous les résultats
            all_results = []
            total_hits = 0
            aggregations = None
            current_page = 1
            total_search_time = 0
            
            while True:
                # Préparer payload pour la page courante
                current_payload = base_payload.copy()
                current_payload["page"] = current_page
                
                # Exécution requête HTTP pour la page courante
                async with self._session.post(search_url, json=current_payload, headers=headers) as response:
                    page_search_time = self._get_processing_time(search_start)
                    total_search_time += page_search_time
                    
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Search service error {response.status}: {error_text}")
                        
                        return QueryExecutionResult(
                            success=False,
                            results=[],
                            total_hits=0,
                            aggregations=None,
                            processing_time_ms=self._get_processing_time(start_time),
                            search_service_time_ms=total_search_time,
                            error_message=f"Search service error {response.status}: {error_text}"
                        )
                    
                    page_data = await response.json()
                    
                    # Récupérer les résultats de cette page
                    page_results = page_data.get("results", [])
                    all_results.extend(page_results)
                    logger.info(f"Page {current_page}: {len(page_results)} résultats récupérés, total accumulé: {len(all_results)}")
                    
                    # Mettre à jour les métadonnées (seulement à la première page)
                    if current_page == 1:
                        total_hits = page_data.get("total_hits", 0)
                        aggregations = page_data.get("aggregations")
                    
                    # Vérifier s'il y a plus de pages
                    has_more_results = page_data.get("has_more_results", False)
                    total_pages = page_data.get("total_pages", 1)
                    total_hits_page = page_data.get("total_hits", 0)
                    returned_results = page_data.get("returned_results", len(page_results))
                    
                    logger.info(f"Page {current_page}/{total_pages}: {len(page_results)} résultats récupérés, has_more: {has_more_results}, total_hits: {total_hits_page}, returned_results: {returned_results}")
                    
                    # Si plus de résultats et pas trop de pages (protection contre boucles infinies)
                    if has_more_results and current_page < total_pages and current_page < 100:  # Max 100 pages
                        current_page += 1
                        search_start = datetime.now()  # Reset timer pour la page suivante
                        logger.info(f"Récupération page suivante: {current_page}")
                    else:
                        logger.info(f"Arrêt récupération - has_more: {has_more_results}, current_page: {current_page}, total_pages: {total_pages}")
                        break
            
            logger.info(f"Récupération terminée: {len(all_results)} résultats sur {current_page} pages")
            
            return QueryExecutionResult(
                success=True,
                results=all_results,
                total_hits=total_hits,
                aggregations=aggregations,
                processing_time_ms=self._get_processing_time(start_time),
                search_service_time_ms=total_search_time
            )
        
        except asyncio.TimeoutError:
            raise  # Re-raise pour gestion retry
            
        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error: {str(e)}")
            return QueryExecutionResult(
                success=False,
                results=[],
                total_hits=0,
                aggregations=None,
                processing_time_ms=self._get_processing_time(start_time),
                search_service_time_ms=0,
                error_message=f"HTTP client error: {str(e)}"
            )
    
    def _should_retry(self, error_message: str) -> bool:
        """Détermine si une erreur mérite un retry"""
        
        if not error_message:
            return True
            
        # Retry sur erreurs temporaires
        retry_patterns = [
            "timeout",
            "connection",
            "502",
            "503",
            "504",
            "temporarily unavailable"
        ]
        
        error_lower = error_message.lower()
        return any(pattern in error_lower for pattern in retry_patterns)
    
    def _generate_cache_key(self, query: Dict[str, Any], user_id: int) -> str:
        """Génère une clé de cache pour une query"""
        
        # Serialization déterministe
        query_str = json.dumps(query, sort_keys=True, separators=(',', ':'))
        
        # Hash pour clé compacte
        import hashlib
        query_hash = hashlib.sha256(f"{query_str}:{user_id}".encode()).hexdigest()[:16]
        
        return f"query_result:{user_id}:{query_hash}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[QueryExecutionResult]:
        """Récupère un résultat du cache s'il est valide"""
        
        entry = self._result_cache.get(cache_key)
        
        if entry and not entry.is_expired:
            return entry.result
        
        # Nettoyer entrée expirée
        if entry:
            del self._result_cache[cache_key]
        
        return None
    
    def _cache_result(
        self, 
        cache_key: str, 
        result: QueryExecutionResult, 
        ttl_seconds: int
    ) -> None:
        """Met en cache un résultat"""
        
        # Ne pas cacher les erreurs
        if not result.success:
            return
        
        entry = CacheEntry(
            result=result,
            cached_at=datetime.now(),
            ttl_seconds=ttl_seconds
        )
        
        self._result_cache[cache_key] = entry
        
        # Nettoyage périodique du cache (limite mémoire)
        if len(self._result_cache) > 1000:
            self._cleanup_cache()
    
    def _cleanup_cache(self) -> None:
        """Nettoie les entrées expirées du cache"""
        
        expired_keys = [
            key for key, entry in self._result_cache.items()
            if entry.is_expired
        ]
        
        for key in expired_keys:
            del self._result_cache[key]
        
        logger.debug(f"Cache cleanup: {len(expired_keys)} entrées supprimées")
    
    def _get_processing_time(self, start_time: datetime) -> int:
        """Calcule le temps de traitement en ms"""
        return int((datetime.now() - start_time).total_seconds() * 1000)
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check du search_service"""
        
        try:
            if not self._session:
                await self.initialize()
            
            # Test simple ping
            health_url = urljoin(self.search_service_url, '/health')
            
            async with self._session.get(health_url) as response:
                if response.status == 200:
                    health_data = await response.json()
                    
                    return {
                        "status": "healthy",
                        "component": "query_executor",
                        "search_service": health_data,
                        "cache_size": len(self._result_cache),
                        "stats": self.stats,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "component": "query_executor",
                        "error": f"Search service returned {response.status}",
                        "timestamp": datetime.now().isoformat()
                    }
                    
        except Exception as e:
            return {
                "status": "unhealthy",
                "component": "query_executor",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques de l'executor"""
        
        cache_stats = {
            "cache_size": len(self._result_cache),
            "cache_hit_rate": 0.0
        }
        
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        if total_requests > 0:
            cache_stats["cache_hit_rate"] = self.stats["cache_hits"] / total_requests
        
        return {
            **self.stats,
            **cache_stats
        }
    
    async def clear_cache(self) -> None:
        """Vide le cache des résultats"""
        
        self._result_cache.clear()
        logger.info("QueryExecutor cache cleared")
    
    async def close(self) -> None:
        """Ferme proprement les connexions"""
        
        if self._session:
            await self._session.close()
            self._session = None
            logger.info("QueryExecutor session fermée")