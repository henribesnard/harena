"""
Dépendances FastAPI optimisées pour conversation service - Version compatible JWT
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
    """Vérificateur de santé des services avec cache optimisé"""
    
    def __init__(self, cache_ttl: int = 30):
        self.cache_ttl = cache_ttl
        self._health_cache: Dict[str, ServiceHealth] = {}
        
    async def check_service_health(self, service_name: str, health_func: Callable) -> ServiceHealth:
        """Vérifie la santé d'un service avec cache intelligent"""
        
        # Vérifier le cache
        cached_health = self._health_cache.get(service_name)
        if cached_health:
            age = (datetime.now(timezone.utc) - cached_health.last_check).total_seconds()
            if age < self.cache_ttl:
                return cached_health
        
        # Nouvelle vérification avec timeout
        start_time = time.time()
        try:
            # Ajout timeout pour éviter les blocages
            if asyncio.iscoroutinefunction(health_func):
                is_healthy = await asyncio.wait_for(health_func(), timeout=5.0)
            else:
                is_healthy = health_func()
                
            response_time = (time.time() - start_time) * 1000
            
            health = ServiceHealth(
                service_name=service_name,
                healthy=bool(is_healthy),
                last_check=datetime.now(timezone.utc),
                response_time_ms=response_time
            )
            
        except asyncio.TimeoutError:
            response_time = (time.time() - start_time) * 1000
            health = ServiceHealth(
                service_name=service_name,
                healthy=False,
                last_check=datetime.now(timezone.utc),
                error_message="Timeout health check",
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
        
        # Mise en cache avec limitation de taille
        if len(self._health_cache) > 10:
            # Nettoyer les entrées les plus anciennes
            oldest_key = min(self._health_cache.keys(), 
                           key=lambda k: self._health_cache[k].last_check)
            del self._health_cache[oldest_key]
        
        self._health_cache[service_name] = health
        return health

# Instance globale du vérificateur
health_checker = ServiceHealthChecker()

async def get_deepseek_client(request: Request) -> DeepSeekClient:
    """
    Récupération client DeepSeek depuis app state avec validation renforcée
    
    Args:
        request: Requête FastAPI
        
    Returns:
        DeepSeekClient: Client DeepSeek opérationnel
        
    Raises:
        HTTPException: Si client non disponible ou défaillant
    """
    try:
        # Vérification présence dans app state
        if not hasattr(request.app.state, 'deepseek_client'):
            metrics_collector.increment_counter("dependencies.errors.deepseek_missing_state")
            logger.error("DeepSeek client absent de app state")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Service DeepSeek non initialisé",
                    "service": "deepseek",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        
        deepseek_client = request.app.state.deepseek_client
        
        if not deepseek_client:
            metrics_collector.increment_counter("dependencies.errors.deepseek_missing")
            logger.error("DeepSeek client None dans app state")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Service DeepSeek temporairement indisponible",
                    "service": "deepseek",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        
        # Vérification santé avec cache et timeout
        try:
            health = await health_checker.check_service_health(
                "deepseek", 
                deepseek_client.health_check
            )
            
            if not health.healthy:
                metrics_collector.increment_counter("dependencies.errors.deepseek_unhealthy")
                logger.warning(f"DeepSeek client non opérationnel: {health.error_message}")
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
        except Exception as health_error:
            # Health check a échoué mais on peut essayer le client quand même
            logger.warning(f"Health check DeepSeek échoué, utilisation directe: {str(health_error)}")
            metrics_collector.increment_counter("dependencies.warnings.deepseek_health_failed")
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
    Récupération cache manager depuis app state (toujours optionnel)
    
    Args:
        request: Requête FastAPI
        
    Returns:
        CacheManager ou None: Cache manager si disponible
        
    Note:
        Ne lève jamais d'exception - retourne None si indisponible
    """
    try:
        # Vérification présence dans app state
        if not hasattr(request.app.state, 'cache_manager'):
            metrics_collector.increment_counter("dependencies.cache_not_configured")
            logger.debug("Cache manager non configuré dans app state")
            return None
        
        cache_manager = request.app.state.cache_manager
        
        if not cache_manager:
            metrics_collector.increment_counter("dependencies.cache_disabled")
            logger.debug("Cache manager désactivé - mode sans cache")
            return None
        
        # Vérification santé cache (non bloquante avec timeout court)
        try:
            health = await asyncio.wait_for(
                health_checker.check_service_health("cache", cache_manager.health_check),
                timeout=2.0  # Timeout court pour le cache
            )
            
            if health.healthy:
                metrics_collector.increment_counter("dependencies.success.cache")
                if health.response_time_ms:
                    metrics_collector.record_histogram("dependencies.cache.response_time", health.response_time_ms)
                return cache_manager
            else:
                metrics_collector.increment_counter("dependencies.cache_unhealthy")
                logger.debug(f"Cache manager non opérationnel: {health.error_message}")
                return None
                
        except asyncio.TimeoutError:
            metrics_collector.increment_counter("dependencies.cache_timeout")
            logger.debug("Health check cache timeout - utilisation directe")
            # Retourner le cache manager malgré le timeout (dégradation gracieuse)
            return cache_manager
        except Exception as e:
            metrics_collector.increment_counter("dependencies.cache_check_failed")
            logger.debug(f"Vérification santé cache échouée: {str(e)}")
            # Retourner le cache manager même si health check échoue
            return cache_manager
        
    except Exception as e:
        metrics_collector.increment_counter("dependencies.errors.cache_unexpected")
        logger.debug(f"Erreur récupération cache manager: {str(e)}")
        return None  # Toujours retourner None en cas d'erreur (non critique)

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
        # Vérification présence service dans app state
        if not hasattr(request.app.state, 'conversation_service'):
            metrics_collector.increment_counter("dependencies.errors.service_missing")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Conversation service non initialisé",
                    "service": "conversation_service",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        
        conversation_service = request.app.state.conversation_service
        
        if not conversation_service:
            metrics_collector.increment_counter("dependencies.errors.service_null")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Conversation service null",
                    "service": "conversation_service",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        
        # Vérification santé du service
        if not getattr(conversation_service, 'service_healthy', False):
            metrics_collector.increment_counter("dependencies.errors.service_unhealthy")
            initialization_error = getattr(conversation_service, 'initialization_error', 'Erreur inconnue')
            error_details = {
                "error": "Conversation service en maintenance",
                "service": "conversation_service", 
                "initialization_error": initialization_error,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            logger.warning(f"Service unhealthy: {error_details}")
            raise HTTPException(status_code=503, detail=error_details)
        
        # Construction statut détaillé avec gestion d'erreur
        try:
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
                "jwt_compatible": service_config.get("jwt_compatible", False),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as status_error:
            logger.warning(f"Erreur construction statut détaillé: {str(status_error)}")
            # Statut minimal en cas d'erreur
            status = {
                "status": "healthy",
                "service": "conversation_service",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": "Statut partiel disponible"
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
    Validation user_id du path vs token JWT avec logging sécurité renforcé
    
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
        # Validation de base avec logs
        if path_user_id <= 0:
            metrics_collector.increment_counter("dependencies.validation.invalid_user_id")
            logger.warning(f"User ID invalide dans l'URL: {path_user_id}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "User ID invalide dans l'URL",
                    "provided_user_id": path_user_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        
        if path_user_id > 1000000000:  # Limite de sécurité
            metrics_collector.increment_counter("dependencies.validation.suspicious_user_id")
            logger.warning(f"User ID suspicieusement élevé: {path_user_id}")
        
        # Vérification correspondance avec token
        try:
            await verify_user_id_match(request, path_user_id)
        except HTTPException as auth_error:
            # Re-log pour les métriques de sécurité
            client_ip = getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
            logger.warning(
                f"User ID mismatch - Path: {path_user_id}, Token: {token_user_id}, "
                f"IP: {client_ip}, URL: {request.url.path}"
            )
            raise auth_error
        
        # Logging sécurité succès (debug uniquement)
        client_ip = getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
        logger.debug(
            f"User validation OK - User: {path_user_id}, "
            f"IP: {client_ip}, Path: {request.url.path}"
        )
        
        metrics_collector.increment_counter("dependencies.validation.success")
        return path_user_id
        
    except HTTPException:
        metrics_collector.increment_counter("dependencies.validation.failed")
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        metrics_collector.increment_counter("dependencies.errors.validation_unexpected")
        logger.error(f"Erreur inattendue validation user ID: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Erreur interne validation utilisateur",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

async def get_user_context(
    request: Request,
    user_id: int = Depends(get_current_user_id)
) -> Dict[str, Any]:
    """
    Récupération contexte utilisateur enrichi avec gestion d'erreur robuste
    
    Args:
        request: Requête FastAPI
        user_id: ID utilisateur authentifié
        
    Returns:
        Dict: Contexte utilisateur (toujours retourne quelque chose)
    """
    try:
        # Contexte de session de base (toujours disponible)
        base_context = {
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        # Informations de requête (avec gestion d'erreur)
        try:
            request_context = {
                "request_id": f"req_{int(time.time())}_{user_id}",
                "client_ip": getattr(request.client, 'host', 'unknown') if request.client else 'unknown',
                "user_agent": request.headers.get("user-agent", "unknown")[:100],  # Tronquer pour sécurité
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params) if request.query_params else {},
            }
        except Exception as req_error:
            logger.debug(f"Erreur extraction contexte requête: {str(req_error)}")
            request_context = {
                "request_id": f"req_{int(time.time())}_{user_id}",
                "error": "Contexte requête partiel"
            }
        
        # Métriques d'utilisation (avec gestion d'état de session)
        try:
            # Utiliser request.state pour persister des infos de session
            if not hasattr(request.state, 'session_start'):
                request.state.session_start = datetime.now(timezone.utc).isoformat()
            
            if not hasattr(request.state, 'request_count'):
                request.state.request_count = 0
            
            request.state.request_count += 1
            
            usage_context = {
                "session_start": request.state.session_start,
                "request_count": request.state.request_count,
                "session_duration_seconds": (
                    datetime.now(timezone.utc) - 
                    datetime.fromisoformat(request.state.session_start.replace('Z', '+00:00'))
                ).total_seconds()
            }
        except Exception as usage_error:
            logger.debug(f"Erreur métriques usage: {str(usage_error)}")
            usage_context = {
                "session_start": datetime.now(timezone.utc).isoformat(),
                "request_count": 1
            }
        
        # Contexte technique (avec informations de service)
        try:
            service_config = getattr(request.app.state, 'service_config', {})
            technical_context = {
                "phase": service_config.get("phase", 1),
                "features_available": service_config.get("features", ["intent_classification"]),
                "json_output_enforced": service_config.get("json_output_enforced", True),
                "jwt_compatible": service_config.get("jwt_compatible", False)
            }
        except Exception as tech_error:
            logger.debug(f"Erreur contexte technique: {str(tech_error)}")
            technical_context = {
                "phase": 1,
                "features_available": ["intent_classification"]
            }
        
        # Assemblage contexte final
        user_context = {
            **base_context,
            "request_details": request_context,
            "usage_stats": usage_context,
            "technical_info": technical_context
        }
        
        metrics_collector.increment_counter("dependencies.context.generated")
        logger.debug(f"Contexte généré pour user {user_id} - {len(user_context)} sections")
        
        return user_context
        
    except Exception as e:
        metrics_collector.increment_counter("dependencies.errors.context_generation")
        logger.error(f"Erreur génération contexte utilisateur: {str(e)}")
        
        # Contexte minimal en cas d'erreur complète
        return {
            "user_id": user_id,
            "error": "Contexte minimal généré",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_details": {
                "request_id": f"req_{int(time.time())}_{user_id}",
                "error": "Contexte d'erreur"
            }
        }

class AdvancedRateLimitDependency:
    """Dépendance rate limiting avancée avec fenêtres glissantes et gestion d'erreur robuste"""
    
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
        
        logger.info(f"RateLimit initialisé: {requests_per_minute}/min, {requests_per_hour}/h, burst: {burst_allowance}")
    
    async def __call__(
        self,
        request: Request,
        user_id: int = Depends(get_current_user_id)
    ) -> None:
        """
        Vérification rate limiting avec gestion d'erreur robuste
        """
        try:
            current_time = time.time()
            current_minute = int(current_time // 60)
            current_hour = int(current_time // 3600)
            
            # Nettoyage périodique avec gestion d'erreur
            try:
                await self._periodic_cleanup(current_time)
            except Exception as cleanup_error:
                logger.debug(f"Erreur nettoyage rate limit: {str(cleanup_error)}")
            
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
                        },
                        headers={"Retry-After": "5"}
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
                    },
                    headers={"Retry-After": "60"}
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
                    },
                    headers={"Retry-After": "3600"}
                )
            
            # Incrémentation compteurs
            user_data["minute_requests"][current_minute] = minute_count + 1
            user_data["hour_requests"][current_hour] = hour_count + 1
            
            # Métriques succès
            metrics_collector.increment_counter("dependencies.rate_limit.allowed")
            
            # Headers informatifs dans request state
            if hasattr(request, 'state'):
                request.state.rate_limit_info = {
                    "minute_remaining": max(0, self.requests_per_minute - minute_count - 1),
                    "hour_remaining": max(0, self.requests_per_hour - hour_count - 1),
                    "reset_minute": (current_minute + 1) * 60,
                    "reset_hour": (current_hour + 1) * 3600
                }
            
        except HTTPException:
            raise  # Re-raise rate limit exceptions
        except Exception as e:
            metrics_collector.increment_counter("dependencies.errors.rate_limit")
            logger.error(f"Erreur rate limiting: {str(e)}")
            # Continue sans rate limiting en cas d'erreur (fail open)
            logger.warning(f"Rate limiting désactivé temporairement pour user {user_id} à cause d'erreur")
    
    async def _periodic_cleanup(self, current_time: float) -> None:
        """Nettoyage périodique des données anciennes avec gestion d'erreur"""
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
        
        try:
            cleaned_users = 0
            cleaned_entries = 0
            
            current_minute = int(current_time // 60)
            current_hour = int(current_time // 3600)
            
            for user_id, user_data in list(self.user_requests.items()):
                try:
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
                        
                except Exception as user_cleanup_error:
                    logger.debug(f"Erreur nettoyage user {user_id}: {str(user_cleanup_error)}")
            
            self._last_cleanup = current_time
            
            if cleaned_entries > 0 or cleaned_users > 0:
                logger.debug(f"Rate limit cleanup: {cleaned_entries} entries, {cleaned_users} users")
                
        except Exception as e:
            logger.error(f"Erreur nettoyage rate limit global: {str(e)}")
    
    async def _log_rate_limit_violation(
        self,
        request: Request,
        user_id: int,
        limit_type: str
    ) -> None:
        """Log des violations de rate limit pour sécurité avec gestion d'erreur"""
        try:
            client_ip = getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
            user_agent = request.headers.get("user-agent", "unknown")
            
            logger.warning(
                f"Rate limit dépassé - User: {user_id}, "
                f"Type: {limit_type}, IP: {client_ip}, "
                f"UA: {user_agent[:50]}..., Path: {request.url.path}"
            )
        except Exception as log_error:
            logger.debug(f"Erreur logging violation rate limit: {str(log_error)}")

# Instances configurables selon l'environnement
def create_rate_limit_dependency():
    """Crée une dépendance rate limiting selon l'environnement"""
    try:
        from config_service.config import settings
        environment = getattr(settings, 'ENVIRONMENT', 'production')
        
        if environment == 'development':
            return AdvancedRateLimitDependency(
                requests_per_minute=120,
                requests_per_hour=2000,
                burst_allowance=20
            )
        elif environment in ['testing', 'test']:
            return AdvancedRateLimitDependency(
                requests_per_minute=300,
                requests_per_hour=5000,
                burst_allowance=50
            )
        else:
            # Production: limites standard
            return AdvancedRateLimitDependency(
                requests_per_minute=60,
                requests_per_hour=1000,
                burst_allowance=10
            )
            
    except ImportError:
        logger.warning("Configuration non disponible, utilisation limites par défaut")
        return AdvancedRateLimitDependency()

# Instance globale
rate_limit_dependency = create_rate_limit_dependency()

# Fonction utilitaire pour décorateur rate limiting
def with_rate_limit(dependency_func: Callable = None):
    """Décorateur pour appliquer rate limiting à une fonction"""
    if dependency_func is None:
        dependency_func = rate_limit_dependency
    
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
                try:
                    # Application du rate limiting
                    await dependency_func(request)
                except Exception as rate_limit_error:
                    logger.warning(f"Rate limiting échoué, continuation: {str(rate_limit_error)}")
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

# Utilitaires de debug
def get_dependency_health_status() -> Dict[str, Any]:
    """Retourne le statut de santé des dépendances"""
    return {
        "health_checker": {
            "cache_size": len(health_checker._health_cache),
            "cache_ttl": health_checker.cache_ttl
        },
        "rate_limiter": {
            "active_users": len(rate_limit_dependency.user_requests),
            "requests_per_minute": rate_limit_dependency.requests_per_minute,
            "requests_per_hour": rate_limit_dependency.requests_per_hour
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

# Export des principales dépendances
__all__ = [
    'get_deepseek_client',
    'get_cache_manager', 
    'get_conversation_service_status',
    'validate_path_user_id',
    'get_user_context',
    'rate_limit_dependency',
    'with_rate_limit',
    'get_dependency_health_status'
]