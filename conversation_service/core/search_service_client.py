"""
Client search_service avec circuit breaker et retry intelligent
Intégration complète des patterns de résilience
"""
import asyncio
import logging
import time
import aiohttp
from typing import Any, Dict, Optional, List, Union
from datetime import datetime, timezone

from conversation_service.core.circuit_breaker import CircuitBreaker, CircuitBreakerError
from conversation_service.utils.retry_handler import SearchServiceRetryHandler
from conversation_service.models.contracts.search_service import (
    SearchQuery, SearchResponse, QueryValidationResult, SearchError
)

logger = logging.getLogger("conversation_service.search_service_client")


class SearchServiceConfig:
    """Configuration du client search_service"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000/api/v1/search",
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        circuit_breaker_enabled: bool = True,
        fallback_enabled: bool = True
    ):
        self.base_url = base_url.rstrip('/')
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.circuit_breaker_enabled = circuit_breaker_enabled
        self.fallback_enabled = fallback_enabled
        
        # Endpoints
        self.search_endpoint = f"{self.base_url}/search"
        self.health_endpoint = f"{self.base_url}/health"


class SearchServiceClient:
    """
    Client search_service avec patterns de résilience
    
    Fonctionnalités:
    - Circuit breaker pour protection
    - Retry intelligent avec backoff
    - Validation requêtes automatique
    - Fallbacks configurables
    - Métriques détaillées
    """
    
    def __init__(self, config: Optional[SearchServiceConfig] = None):
        self.config = config or SearchServiceConfig()
        
        # Circuit breaker spécialisé
        self.circuit_breaker = CircuitBreaker(
            name="search_service",
            failure_threshold=5,
            recovery_timeout=60.0,
            success_threshold=2,
            timeout_threshold=self.config.timeout_seconds,
            expected_exceptions=(
                aiohttp.ClientError,
                aiohttp.ServerTimeoutError,
                asyncio.TimeoutError,
                SearchServiceError
            )
        ) if self.config.circuit_breaker_enabled else None
        
        # Retry handler spécialisé
        self.retry_handler = SearchServiceRetryHandler()
        
        # Session HTTP réutilisable
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_timeout = aiohttp.ClientTimeout(
            total=self.config.timeout_seconds,
            connect=10.0
        )
        
        # Métriques client
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.fallback_requests = 0
        self.average_response_time = 0.0
        self._response_times: List[float] = []
        
        logger.info(f"Search service client initialisé - Base URL: {self.config.base_url}")
    
    async def __aenter__(self):
        """Context manager entry"""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.close()
    
    async def _ensure_session(self):
        """Assure qu'une session HTTP est active"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self._session_timeout,
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'conversation-service/1.0'
                }
            )
    
    async def close(self):
        """Fermeture propre des ressources"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    async def search(
        self,
        query: SearchQuery,
        validate_query: bool = True,
        enable_fallback: bool = None
    ) -> SearchResponse:
        """
        Exécute une recherche avec protection complète
        
        Args:
            query: Requête search_service
            validate_query: Validation automatique avant envoi
            enable_fallback: Override fallback config
            
        Returns:
            Résultats de recherche
            
        Raises:
            SearchServiceError: Erreur search_service
            CircuitBreakerError: Circuit ouvert
            ValidationError: Requête invalide
        """
        start_time = time.time()
        self.total_requests += 1
        
        try:
            # Validation requête si demandée
            if validate_query:
                validation_result = self._validate_query(query)
                if not validation_result.schema_valid:
                    raise SearchServiceValidationError(
                        f"Requête invalide: {validation_result.errors}"
                    )
            
            # Exécution avec protection
            if self.circuit_breaker:
                result = await self.circuit_breaker.call(
                    self._execute_search_request,
                    query.dict(exclude_none=True)
                )
            else:
                result = await self.retry_handler.execute_search_request(
                    self._execute_search_request,
                    query.dict(exclude_none=True),
                    error_context="direct_search"
                )
            
            # Métriques succès
            response_time = time.time() - start_time
            self._record_success(response_time)
            
            return result
        
        except (CircuitBreakerError, SearchServiceError) as e:
            # Tentative fallback si autorisé
            enable_fb = enable_fallback if enable_fallback is not None else self.config.fallback_enabled
            
            if enable_fb:
                logger.warning(f"Search failed, attempting fallback: {str(e)}")
                result = await self._execute_fallback(query, e)
                if result:
                    self.fallback_requests += 1
                    return result
            
            # Métriques échec
            response_time = time.time() - start_time
            self._record_failure(response_time)
            raise
        
        except Exception as e:
            response_time = time.time() - start_time
            self._record_failure(response_time)
            logger.error(f"Search request unexpected error: {type(e).__name__}: {str(e)}")
            raise SearchServiceError(f"Unexpected error: {str(e)}")
    
    async def _execute_search_request(self, query_dict: Dict[str, Any]) -> SearchResponse:
        """Exécution requête HTTP search_service"""
        await self._ensure_session()
        
        try:
            async with self._session.post(
                self.config.search_endpoint,
                json=query_dict
            ) as response:
                
                # Vérification statut
                if response.status == 422:
                    error_data = await response.json()
                    raise SearchServiceValidationError(
                        f"Validation error: {error_data.get('detail', 'Unknown validation error')}"
                    )
                elif response.status == 429:
                    raise SearchServiceRateLimitError("Rate limit exceeded")
                elif response.status == 503:
                    raise SearchServiceUnavailableError("Service temporarily unavailable")
                elif response.status >= 500:
                    raise SearchServiceError(f"Server error: {response.status}")
                elif response.status >= 400:
                    raise SearchServiceError(f"Client error: {response.status}")
                
                # Parsing réponse
                response_data = await response.json()
                return self._convert_search_service_response(response_data)
                
        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error: {str(e)}")
            raise SearchServiceConnectionError(f"Connection error: {str(e)}")
        except asyncio.TimeoutError:
            logger.error("Search request timeout")
            raise SearchServiceTimeoutError("Request timeout")
    
    def _validate_query(self, query: SearchQuery) -> QueryValidationResult:
        """Validation basique de la requête"""
        errors = []
        
        # Validation user_id
        if not query.user_id or query.user_id <= 0:
            errors.append("user_id manquant ou invalide")
        
        # Validation filters - peut être dict ou SearchFilters BaseModel
        if query.filters:
            if not (isinstance(query.filters, dict) or hasattr(query.filters, '__dict__')):
                errors.append("filters doit être un dictionnaire ou objet SearchFilters")
        
        # Validation aggregations
        if query.aggregations:
            if not isinstance(query.aggregations, dict):
                errors.append("aggregations doit être un dictionnaire")
        
        # Validation page_size
        if query.page_size and (query.page_size < 1 or query.page_size > 1000):
            errors.append("page_size doit être entre 1 et 1000")
        
        return QueryValidationResult(
            schema_valid=len(errors) == 0,
            contract_compliant=len(errors) == 0,  # Simplifié pour cette validation
            errors=errors,
            estimated_performance="unknown",
            optimization_applied=[],
            potential_issues=[],
            warnings=[]
        )
    
    async def _execute_fallback(
        self,
        original_query: SearchQuery,
        original_error: Exception
    ) -> Optional[SearchResponse]:
        """
        Stratégies de fallback en cas d'échec
        
        Args:
            original_query: Requête originale
            original_error: Erreur originale
            
        Returns:
            Réponse fallback ou None
        """
        logger.info(f"Executing fallback for error: {type(original_error).__name__}")
        
        # Stratégie 1: Simplification requête
        if isinstance(original_error, SearchServiceValidationError):
            simplified_query = self._create_simplified_query(original_query)
            if simplified_query:
                try:
                    return await self._execute_search_request(simplified_query.dict(exclude_none=True))
                except Exception as e:
                    logger.warning(f"Simplified query fallback failed: {str(e)}")
        
        # Stratégie 2: Requête vide avec pagination
        if isinstance(original_error, (SearchServiceTimeoutError, SearchServiceError)):
            empty_query = self._create_empty_query(original_query.user_id)
            try:
                return await self._execute_search_request(empty_query.dict(exclude_none=True))
            except Exception as e:
                logger.warning(f"Empty query fallback failed: {str(e)}")
        
        # Stratégie 3: Cache ou réponse vide
        return self._create_fallback_response(original_query)
    
    def _create_simplified_query(self, original_query: SearchQuery) -> Optional[SearchQuery]:
        """Crée une requête simplifiée"""
        try:
            return SearchQuery(
                user_id=original_query.user_id,
                filters={"user_id": original_query.user_id},  # Filtre minimum
                sort=[{"date": {"order": "desc"}}],
                page_size=20
            )
        except Exception:
            return None
    
    def _create_empty_query(self, user_id: int) -> SearchQuery:
        """Crée une requête minimale"""
        return SearchQuery(
            user_id=user_id,
            filters={"user_id": user_id},
            page_size=10
        )
    
    def _create_fallback_response(self, original_query: SearchQuery) -> SearchResponse:
        """Crée une réponse fallback vide"""
        return SearchResponse(
            hits=[],
            total_hits=0,
            aggregations={},
            took_ms=0,
            query_id=f"fallback_{int(time.time())}",
            timestamp=datetime.now(timezone.utc)
        )
    
    def _record_success(self, response_time: float):
        """Enregistre un succès"""
        self.successful_requests += 1
        self._response_times.append(response_time)
        
        # Garder fenêtre glissante de 100 mesures
        if len(self._response_times) > 100:
            self._response_times = self._response_times[-100:]
        
        self.average_response_time = sum(self._response_times) / len(self._response_times)
    
    def _record_failure(self, response_time: float):
        """Enregistre un échec"""
        self.failed_requests += 1
        if response_time > 0:
            self._response_times.append(response_time)
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérification santé du service search"""
        await self._ensure_session()
        
        try:
            start_time = time.time()
            async with self._session.get(self.config.health_endpoint) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    health_data = await response.json()
                    return {
                        "status": "healthy",
                        "response_time_ms": response_time * 1000,
                        "service_info": health_data
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "response_time_ms": response_time * 1000,
                        "error": f"HTTP {response.status}"
                    }
        except Exception as e:
            return {
                "status": "unreachable",
                "error": str(e)
            }
    
    def _convert_search_service_response(self, response_data: Dict[str, Any]) -> SearchResponse:
        """
        Convertit la réponse brute du search_service en SearchResponse
        
        Le search_service retourne:
        {
            "results": [{"transaction_id": "...", "amount": 123, "_score": 0.5, ...}],
            "aggregations": {...} | None,
            "response_metadata": {...}
        }
        
        SearchResponse attend:
        {
            "hits": [{"_id": "...", "_score": 0.5, "_source": {...}}],
            "total_hits": 123,
            "aggregations": {...} | None
        }
        """
        from conversation_service.models.contracts.search_service import SearchHit
        
        results = response_data.get('results', [])
        
        # Convertir chaque résultat au format SearchHit
        hits = []
        for result in results:
            # Extraire _score s'il existe, sinon 0.0
            score = result.get('_score', 0.0)
            
            # Utiliser transaction_id comme _id
            doc_id = str(result.get('transaction_id', ''))
            
            # Le reste va dans _source (sans _score)
            source = {k: v for k, v in result.items() if k != '_score'}
            
            # Créer le SearchHit
            hit = SearchHit(
                id=doc_id,
                score=score, 
                source=source
            )
            hits.append(hit)
        
        # Récupérer les métadonnées
        metadata = response_data.get('response_metadata', {})
        total_hits = metadata.get('total_results', len(results))
        took_ms = metadata.get('took_ms', 0)
        
        # Créer la SearchResponse
        return SearchResponse(
            hits=hits,
            total_hits=total_hits,
            aggregations=response_data.get('aggregations'),
            took_ms=took_ms
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques client"""
        base_metrics = {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "fallback_requests": self.fallback_requests,
            "success_rate": (
                self.successful_requests / self.total_requests
                if self.total_requests > 0 else 0.0
            ),
            "average_response_time_ms": self.average_response_time * 1000,
            "configuration": {
                "base_url": self.config.base_url,
                "timeout_seconds": self.config.timeout_seconds,
                "circuit_breaker_enabled": self.config.circuit_breaker_enabled,
                "fallback_enabled": self.config.fallback_enabled
            }
        }
        
        # Ajouter métriques circuit breaker si activé
        if self.circuit_breaker:
            base_metrics["circuit_breaker"] = self.circuit_breaker.get_metrics()
        
        # Ajouter métriques retry handler
        base_metrics["retry_handler"] = self.retry_handler.get_metrics()
        
        return base_metrics


# Exceptions spécialisées search_service
class SearchServiceError(Exception):
    """Erreur générale search_service"""
    pass

class SearchServiceConnectionError(SearchServiceError):
    """Erreur de connexion"""
    pass

class SearchServiceTimeoutError(SearchServiceError):
    """Timeout de requête"""
    pass

class SearchServiceValidationError(SearchServiceError):
    """Erreur de validation"""
    pass

class SearchServiceRateLimitError(SearchServiceError):
    """Rate limiting"""
    pass

class SearchServiceUnavailableError(SearchServiceError):
    """Service indisponible"""
    pass