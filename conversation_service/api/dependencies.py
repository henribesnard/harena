"""
Dépendances FastAPI optimisées pour conversation service avec gestion avancée
"""
import logging
import time
import asyncio
from typing import Optional, Dict, Any, Callable
from functools import wraps
from dataclasses import dataclass
from datetime import datetime, timezone
from fastapi import Request, HTTPException, Depends
from conversation_service.clients.deepseek_client import DeepSeekClient
from conversation_service.core.cache_manager import CacheManager
from conversation_service.api.middleware.auth_middleware import get_current_user_id, verify_user_id_match
from conversation_service.utils.metrics_collector import metrics_collector

# Configuration du logger
logger = logging.getLogger("conversation_service.dependencies")


@dataclass
class ServiceHealth:
    """État de santé d'un service"""
    service_name: str
    healthy: bool
    last_check: datetime
    error_message: Optional[str] = None
    response_time_ms: Optional[float] = None


class ServiceHealthChecker:
    """Vérificateur de santé des services avec cache"""
    
    def __init__(self, cache_ttl: int = 30):
        self.cache_ttl = cache_ttl
        self._health_cache: Dict[str, ServiceHealth] = {}
        
    async def check_service_health(self, service_name: str, health_func: Callable) -> ServiceHealth:
        """Vérifie la santé d'un service avec cache"""
        
        # Vérifier le cache
        cached_health = self._health_cache.get(service_name)
        if cached_health:
            age = (datetime.now(timezone.utc) - cached_health.last_check).total_seconds()
            if age < self.cache_ttl:
                return cached_health
        
        # Nouvelle vérification
        start_time = time.time()
        try:
            is_healthy = await health_func() if asyncio.iscoroutinefunction(health_func) else health_func()
            response_time = (time.time() - start_time) * 1000
            
            health = ServiceHealth(
                service_name=service_name,
                healthy=bool(is_healthy),
                last_check=datetime.now(timezone.utc),
                response_time_ms=response_time
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            health = ServiceHealth(
                service_name=service_name,
                healthy=False,
                last_check=datetime.now(timezone.utc),
                error_message=str(e),
                response_time_ms=response_time
            )
        
        # Mise en cache
        self._health_cache[service_name] = health
        return health


# Instance globale du vérificateur
health_checker = ServiceHealthChecker()


async def get_deepseek_client(request: Request) -> DeepSeekClient:
    """
    Récupération client DeepSeek depuis app state avec vérification santé
    
    Args:
        request: Requête FastAPI
        
    Returns:
        DeepSeekClient: Client DeepSeek opérationnel
        
    Raises:
        HTTPException: Si client non disponible ou défaillant
    """
    try:
        deepseek_client = getattr(request.app.state, 'deepseek_client', None)
        
        if not deepseek_client:
            metrics_collector.increment_counter("dependencies.errors.deepseek_missing")
            logger.error("DeepSeek client non disponible dans app state")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Service DeepSeek temporairement indisponible",
                    "service": "deepseek",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        
        # Vérification santé avec cache
        health = await health_checker.check_service_health(
            "deepseek", 
            deepseek_client.health_check
        )
        
        if not health.healthy:
            metrics_collector.increment_counter("dependencies.errors.deepseek_unhealthy")
            logger.error(f"DeepSeek client non opérationnel: {health.error_message}")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Service DeepSeek en maintenance",
                    "service": "deepseek",
                    "health_check_failed": True,
                    "error_details": health.error_message,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        
        # Métriques succès
        metrics_collector.increment_counter("dependencies.success.deepseek")
        if health.response_time_ms:
            metrics_collector.record_histogram("dependencies.deepseek.response_time", health.response_time_ms)
        
        return deepseek_client
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        metrics_collector.increment_counter("dependencies.errors.deepseek_unexpected")
        logger.error(f"Erreur inattendue récupération DeepSeek: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Erreur interne récupération service DeepSeek",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )


async def get_cache_manager(request: Request) -> Optional[CacheManager]:
    """
    Récupération cache manager depuis app state (optionnel)
    
    Args:
        request: Requête FastAPI
        
    Returns:
        CacheManager ou None: Cache manager si disponible
        
    Note:
        Ne lève pas d'exception si cache indisponible (non critique)
    """
    try:
        cache_manager = getattr(request.app.state, 'cache_manager', None)
        
        if not cache_manager:
            metrics_collector.increment_counter("dependencies.cache_disabled")
            logger.debug("Cache manager non disponible - mode sans cache")
            return None
        
        # Vérification santé cache (non bloquante)
        try:
            health = await health_checker.check_service_health(
                "cache",
                cache_manager.health_check
            )
            
            if health.healthy:
                metrics_collector.increment_counter("dependencies.success.cache")
                if health.response_time_ms:
                    metrics_collector.record_histogram("dependencies.cache.response_time", health.response_time_ms)
                return cache_manager
            else:
                metrics_collector.increment_counter("dependencies.cache_unhealthy")
                logger.warning(f"Cache manager non opérationnel: {health.error_message}")
                return None
                
        except Exception as e:
            metrics_collector.increment_counter("dependencies.cache_check_failed")
            logger.debug(f"Vérification santé cache échouée: {str(e)}")
            # Retourner le cache manager même si health check échoue (dégradation gracieuse)
            return cache_manager
        
    except Exception as e:
        metrics_collector.increment_counter("dependencies.errors.cache_unexpected")
        logger.debug(f"Erreur récupération cache manager: {str(e)}")
        return None


async def get_conversation_service_status(request: Request) -> Dict[str, Any]:
    """
    Vérification statut conversation service avec diagnostic détaillé
    
    Args:
        request: Requête FastAPI
        
    Returns:
        Dict: Statut détaillé du service
        
    Raises:
        HTTPException: Si service non opérationnel
    """
    try:
        conversation_service = getattr(request.app.state, 'conversation_service', None)
        
        if not conversation_service:
            metrics_collector.increment_counter("dependencies.errors.service_missing")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Conversation service non initialisé",
                    "service": "conversation_service",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        
        if not getattr(conversation_service, 'service_healthy', False):
            metrics_collector.increment_counter("dependencies.errors.service_unhealthy")
            error_details = {
                "error": "Conversation service en maintenance",
                "service": "conversation_service", 
                "initialization_error": getattr(conversation_service, 'initialization_error', 'Erreur inconnue'),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            logger.warning(f"Service unhealthy: {error_details}")
            raise HTTPException(status_code=503, detail=error_details)
        
        # Construction statut détaillé
        service_config = getattr(request.app.state, 'service_config', {})
        service_start_time = getattr(request.app.state, 'service_start_time', datetime.now(timezone.utc))
        uptime_seconds = (datetime.now(timezone.utc) - service_start_time).total_seconds()
        
        status = {
            "status": "healthy",
            "service": "conversation_service",
            "uptime_seconds": uptime_seconds,
            "phase": service_config.get("phase", 1),
            "version": service_config.get("version", "1.0.0"),
            "features": service_config.get("features", []),
            "json_output_enforced": service_config.get("json_output_enforced", True),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        metrics_collector.increment_counter("dependencies.success.service_status")
        return status
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        metrics_collector.increment_counter("dependencies.errors.service_unexpected")
        logger.error(f"Erreur vérification statut service: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Erreur interne vérification statut",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )


async def validate_path_user_id(
    request: Request,
    path_user_id: int,
    token_user_id: int = Depends(get_current_user_id)
) -> int:
    """
    Validation user_id du path vs token JWT avec logging sécurité
    
    Args:
        request: Requête FastAPI
        path_user_id: User ID depuis l'URL
        token_user_id: User ID depuis le token JWT
        
    Returns:
        int: User ID validé
        
    Raises:
        HTTPException: Si validation échoue
    """
    try:
        # Validation de base
        if path_user_id <= 0:
            metrics_collector.increment_counter("dependencies.validation.invalid_user_id")
            raise HTTPException(
                status_code=400,
                detail="User ID invalide dans l'URL"
            )
        
        # Vérification correspondance avec token
        await verify_user_id_match(request, path_user_id)
        
        # Logging sécurité (sans données sensibles)
        client_ip = getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
        logger.info(
            f"User validation successful - User: {path_user_id}, "
            f"IP: {client_ip}, "
            f"Path: {request.url.path}"
        )
        
        metrics_collector.increment_counter("dependencies.validation.success")
        return path_user_id
        
    except HTTPException:
        metrics_collector.increment_counter("dependencies.validation.failed")
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        metrics_collector.increment_counter("dependencies.errors.validation_unexpected")
        logger.error(f"Erreur validation user ID: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Erreur interne validation utilisateur"
        )


async def get_user_context(
    request: Request,
    user_id: int = Depends(get_current_user_id)
) -> Dict[str, Any]:
    """
    Récupération contexte utilisateur enrichi (Phase 1: basique, sera étendu)
    
    Args:
        request: Requête FastAPI
        user_id: ID utilisateur authentifié
        
    Returns:
        Dict: Contexte utilisateur
    """
    try:
        # Contexte de session
        session_context = {
            "user_id": user_id,
            "request_id": f"req_{int(time.time())}_{user_id}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "client_ip": getattr(request.client, 'host', 'unknown') if request.client else 'unknown',
            "user_agent": request.headers.get("user-agent", "unknown")
        }
        
        # Contexte requête
        request_context = {
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers_count": len(request.headers)
        }
        
        # Métriques d'utilisation (basique pour Phase 1)
        usage_context = {
            "session_start": getattr(request.state, "session_start", datetime.now(timezone.utc).isoformat()),
            "request_count": getattr(request.state, "request_count", 1)
        }
        
        # Phase 1: contexte basique
        # TODO Phase 2+: Ajouter historique conversations, préférences utilisateur, patterns
        user_context = {
            **session_context,
            "request_details": request_context,
            "usage_stats": usage_context,
            "phase": 1,
            "features_available": ["intent_classification"],
        }
        
        # Mise à jour compteur requêtes pour session
        request.state.request_count = getattr(request.state, "request_count", 0) + 1
        
        metrics_collector.increment_counter("dependencies.context.generated")
        logger.debug(f"Contexte généré pour user {user_id}")
        
        return user_context
        
    except Exception as e:
        metrics_collector.increment_counter("dependencies.errors.context_generation")
        logger.error(f"Erreur génération contexte utilisateur: {str(e)}")
        
        # Contexte minimal en cas d'erreur
        return {
            "user_id": user_id,
            "error": "Contexte partiel généré",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


class AdvancedRateLimitDependency:
    """Dépendance rate limiting avancée avec fenêtres glissantes"""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_allowance: int = 10
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_allowance = burst_allowance
        
        # Stockage des requêtes par utilisateur
        self.user_requests: Dict[int, Dict[str, Any]] = {}
        self._cleanup_interval = 300  # Nettoyage toutes les 5 minutes
        self._last_cleanup = time.time()
        
        logger.info(f"RateLimit configuré: {requests_per_minute}/min, {requests_per_hour}/h, burst: {burst_allowance}")
    
    async def __call__(
        self,
        request: Request,
        user_id: int = Depends(get_current_user_id)
    ) -> None:
        """
        Vérification rate limiting avec fenêtres multiples
        
        Args:
            request: Requête FastAPI
            user_id: ID utilisateur authentifié
            
        Raises:
            HTTPException: Si rate limit dépassé
        """
        try:
            current_time = time.time()
            current_minute = int(current_time // 60)
            current_hour = int(current_time // 3600)
            
            # Nettoyage périodique
            await self._periodic_cleanup(current_time)
            
            # Initialisation données utilisateur
            if user_id not in self.user_requests:
                self.user_requests[user_id] = {
                    "minute_requests": {},
                    "hour_requests": {},
                    "last_burst": 0,
                    "burst_count": 0
                }
            
            user_data = self.user_requests[user_id]
            
            # Vérification burst (requêtes rapides consécutives)
            if current_time - user_data["last_burst"] < 1:  # 1 seconde
                user_data["burst_count"] += 1
                if user_data["burst_count"] > self.burst_allowance:
                    metrics_collector.increment_counter("dependencies.rate_limit.burst_exceeded")
                    await self._log_rate_limit_violation(request, user_id, "burst")
                    raise HTTPException(
                        status_code=429,
                        detail={
                            "error": "Trop de requêtes rapides consécutives",
                            "limit_type": "burst",
                            "limit": self.burst_allowance,
                            "retry_after": 5,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    )
            else:
                user_data["burst_count"] = 0
                user_data["last_burst"] = current_time
            
            # Vérification limite par minute
            minute_count = user_data["minute_requests"].get(current_minute, 0)
            if minute_count >= self.requests_per_minute:
                metrics_collector.increment_counter("dependencies.rate_limit.minute_exceeded")
                await self._log_rate_limit_violation(request, user_id, "minute")
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": f"Limite de {self.requests_per_minute} requêtes par minute dépassée",
                        "limit_type": "minute",
                        "limit": self.requests_per_minute,
                        "current": minute_count,
                        "retry_after": 60,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
            
            # Vérification limite par heure
            hour_count = user_data["hour_requests"].get(current_hour, 0)
            if hour_count >= self.requests_per_hour:
                metrics_collector.increment_counter("dependencies.rate_limit.hour_exceeded")
                await self._log_rate_limit_violation(request, user_id, "hour")
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": f"Limite de {self.requests_per_hour} requêtes par heure dépassée",
                        "limit_type": "hour",
                        "limit": self.requests_per_hour,
                        "current": hour_count,
                        "retry_after": 3600,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
            
            # Incrémentation compteurs
            user_data["minute_requests"][current_minute] = minute_count + 1
            user_data["hour_requests"][current_hour] = hour_count + 1
            
            # Métriques succès
            metrics_collector.increment_counter("dependencies.rate_limit.allowed")
            
            # Headers informatifs
            if hasattr(request, 'state'):
                request.state.rate_limit_info = {
                    "minute_remaining": self.requests_per_minute - minute_count - 1,
                    "hour_remaining": self.requests_per_hour - hour_count - 1,
                    "reset_minute": (current_minute + 1) * 60,
                    "reset_hour": (current_hour + 1) * 3600
                }
            
        except HTTPException:
            raise  # Re-raise rate limit exceptions
        except Exception as e:
            metrics_collector.increment_counter("dependencies.errors.rate_limit")
            logger.error(f"Erreur rate limiting: {str(e)}")
            # Continue sans rate limiting en cas d'erreur
    
    async def _periodic_cleanup(self, current_time: float) -> None:
        """Nettoyage périodique des données anciennes"""
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
        
        try:
            cleaned_users = 0
            cleaned_entries = 0
            
            current_minute = int(current_time // 60)
            current_hour = int(current_time // 3600)
            
            for user_id, user_data in list(self.user_requests.items()):
                # Nettoyage minutes anciennes (garde dernières 2 minutes)
                old_minutes = [
                    minute for minute in user_data["minute_requests"].keys()
                    if minute < current_minute - 1
                ]
                for minute in old_minutes:
                    del user_data["minute_requests"][minute]
                    cleaned_entries += 1
                
                # Nettoyage heures anciennes (garde dernières 2 heures)
                old_hours = [
                    hour for hour in user_data["hour_requests"].keys()
                    if hour < current_hour - 1
                ]
                for hour in old_hours:
                    del user_data["hour_requests"][hour]
                    cleaned_entries += 1
                
                # Suppression utilisateurs inactifs
                if (not user_data["minute_requests"] and 
                    not user_data["hour_requests"] and
                    current_time - user_data["last_burst"] > 3600):
                    del self.user_requests[user_id]
                    cleaned_users += 1
            
            self._last_cleanup = current_time
            
            if cleaned_entries > 0 or cleaned_users > 0:
                logger.debug(f"Rate limit cleanup: {cleaned_entries} entries, {cleaned_users} users")
                
        except Exception as e:
            logger.error(f"Erreur nettoyage rate limit: {str(e)}")
    
    async def _log_rate_limit_violation(
        self,
        request: Request,
        user_id: int,
        limit_type: str
    ) -> None:
        """Log des violations de rate limit pour sécurité"""
        client_ip = getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
        user_agent = request.headers.get("user-agent", "unknown")
        
        logger.warning(
            f"Rate limit exceeded - User: {user_id}, "
            f"Type: {limit_type}, "
            f"IP: {client_ip}, "
            f"UA: {user_agent[:50]}..., "
            f"Path: {request.url.path}"
        )


# Instances des dépendances
rate_limit_dependency = AdvancedRateLimitDependency(
    requests_per_minute=60,
    requests_per_hour=1000,
    burst_allowance=10
)

# Dépendance rate limiting flexible par environnement
def get_rate_limit_dependency():
    """Récupère la dépendance rate limiting selon l'environnement"""
    try:
        from config_service.config import settings
        environment = getattr(settings, 'ENVIRONMENT', 'production')
        
        if environment == 'development':
            # Limites plus souples en développement
            return AdvancedRateLimitDependency(
                requests_per_minute=120,
                requests_per_hour=2000,
                burst_allowance=20
            )
        elif environment == 'testing':
            # Limites très souples pour tests
            return AdvancedRateLimitDependency(
                requests_per_minute=300,
                requests_per_hour=5000,
                burst_allowance=50
            )
        else:
            # Production: limites standard
            return rate_limit_dependency
            
    except ImportError:
        logger.warning("Configuration non disponible, utilisation limites par défaut")
        return rate_limit_dependency

# Fonction utilitaire pour décorateur rate limiting
def with_rate_limit(dependency_func: Callable = None):
    """Décorateur pour appliquer rate limiting à une fonction"""
    if dependency_func is None:
        dependency_func = get_rate_limit_dependency()
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Recherche de l'objet Request dans les arguments
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if request:
                # Application du rate limiting
                await dependency_func(request)
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator