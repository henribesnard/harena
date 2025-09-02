"""
Dépendances FastAPI optimisées pour conversation service - Version compatible JWT
Health check avec correction de l'erreur await et dégradation gracieuse
"""
import logging
import time
import asyncio
import inspect
from typing import Optional, Dict, Any, Callable, Literal
from functools import wraps
from dataclasses import dataclass
from datetime import datetime, timezone
from fastapi import Request, HTTPException, Depends
from conversation_service.clients.deepseek_client import DeepSeekClient
from conversation_service.core.cache_manager import CacheManager
from conversation_service.api.middleware.auth_middleware import get_current_user_id, verify_user_id_match
from conversation_service.utils.metrics_collector import metrics_collector
from conversation_service.teams import MultiAgentFinancialTeam
from conversation_service.services.conversation_persistence import ConversationPersistenceService

# Configuration du logger
logger = logging.getLogger("conversation_service.dependencies")

@dataclass
class ServiceHealth:
    """État de santé d'un service avec niveaux de dégradation"""
    service_name: str
    status: Literal["healthy", "degraded", "critical", "unknown"]
    last_check: datetime
    error_message: Optional[str] = None
    response_time_ms: Optional[float] = None
    consecutive_failures: int = 0
    last_success: Optional[datetime] = None

class AdvancedServiceHealthChecker:
    """
    Vérificateur de santé des services avec dégradation gracieuse
    
    Évite les blocages de service complet en cas de problème d'un composant.
    Implémente une logique de santé à plusieurs niveaux.
    Gère correctement les fonctions synchrones ET asynchrones.
    """
    
    def __init__(self, cache_ttl: int = 600):  # 10 minutes par défaut
        self.cache_ttl = cache_ttl
        self._health_cache: Dict[str, ServiceHealth] = {}
        self._failure_thresholds = {
            "critical": 10,  # 10 échecs consécutifs = critique
            "degraded": 3    # 3 échecs consécutifs = dégradé
        }
        
    async def check_service_health(
        self, 
        service_name: str, 
        health_func: Callable,
        critical_service: bool = False
    ) -> ServiceHealth:
        """
        Vérifie la santé d'un service avec logique de dégradation intelligente
        
        Gère automatiquement les fonctions synchrones et asynchrones.
        
        Args:
            service_name: Nom du service
            health_func: Fonction de vérification santé (sync ou async)
            critical_service: Si True, échec bloque le service complet
        """
        
        # Vérifier le cache avec TTL adaptatif
        cached_health = self._health_cache.get(service_name)
        if cached_health:
            age = (datetime.now(timezone.utc) - cached_health.last_check).total_seconds()
            
            # TTL adaptatif selon le statut
            effective_ttl = self._get_adaptive_ttl(cached_health.status)
            if age < effective_ttl:
                logger.debug(f"Health cache hit pour {service_name}: {cached_health.status}")
                return cached_health
        
        # Nouvelle vérification avec timeout adaptatif
        start_time = time.time()
        consecutive_failures = cached_health.consecutive_failures if cached_health else 0
        last_success = cached_health.last_success if cached_health else None
        
        try:
            # Timeout adaptatif selon l'historique
            timeout = self._get_adaptive_timeout(service_name, consecutive_failures)
            
            # Détection automatique sync/async avec gestion d'erreur robuste
            try:
                if inspect.iscoroutinefunction(health_func):
                    # Fonction asynchrone
                    logger.debug(f"Health check async pour {service_name}")
                    is_healthy = await asyncio.wait_for(health_func(), timeout=timeout)
                elif asyncio.iscoroutinefunction(health_func):
                    # Double vérification pour asyncio.iscoroutinefunction
                    logger.debug(f"Health check async (asyncio) pour {service_name}")
                    is_healthy = await asyncio.wait_for(health_func(), timeout=timeout)
                else:
                    # Fonction synchrone
                    logger.debug(f"Health check sync pour {service_name}")
                    # Exécution synchrone avec timeout via thread pool
                    is_healthy = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, health_func),
                        timeout=timeout
                    )
                    
            except Exception as detection_error:
                # Fallback: essayer d'abord sync, puis async
                logger.warning(f"Erreur détection type fonction pour {service_name}: {str(detection_error)}")
                try:
                    # Tentative synchrone
                    is_healthy = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, health_func),
                        timeout=timeout
                    )
                except Exception:
                    # Tentative asynchrone
                    is_healthy = await asyncio.wait_for(health_func(), timeout=timeout)
                
            response_time = (time.time() - start_time) * 1000
            
            if is_healthy:
                health = ServiceHealth(
                    service_name=service_name,
                    status="healthy",
                    last_check=datetime.now(timezone.utc),
                    response_time_ms=response_time,
                    consecutive_failures=0,
                    last_success=datetime.now(timezone.utc)
                )
                logger.debug(f"Health check OK pour {service_name} ({response_time:.1f}ms)")
            else:
                # Échec mais on détermine le niveau de sévérité
                consecutive_failures += 1
                status = self._determine_health_status(consecutive_failures, last_success)
                
                health = ServiceHealth(
                    service_name=service_name,
                    status=status,
                    last_check=datetime.now(timezone.utc),
                    error_message="Health check failed",
                    response_time_ms=response_time,
                    consecutive_failures=consecutive_failures,
                    last_success=last_success
                )
                logger.warning(f"Health check échec pour {service_name}: {status}")
            
        except asyncio.TimeoutError:
            response_time = (time.time() - start_time) * 1000
            consecutive_failures += 1
            status = self._determine_health_status(consecutive_failures, last_success)
            
            health = ServiceHealth(
                service_name=service_name,
                status=status,
                last_check=datetime.now(timezone.utc),
                error_message=f"Timeout health check ({timeout}s)",
                response_time_ms=response_time,
                consecutive_failures=consecutive_failures,
                last_success=last_success
            )
            logger.warning(f"Health check timeout pour {service_name}: {status}")
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            consecutive_failures += 1
            status = self._determine_health_status(consecutive_failures, last_success)
            
            health = ServiceHealth(
                service_name=service_name,
                status=status,
                last_check=datetime.now(timezone.utc),
                error_message=f"Health check error: {str(e)}",
                response_time_ms=response_time,
                consecutive_failures=consecutive_failures,
                last_success=last_success
            )
            logger.warning(f"Health check erreur pour {service_name}: {status} - {str(e)}")
        
        # Mise en cache avec limitation de taille
        self._update_cache(health)
        return health
    
    def _get_adaptive_ttl(self, status: str) -> int:
        """TTL adaptatif selon le statut de santé"""
        ttl_map = {
            "healthy": self.cache_ttl,      # TTL complet si sain
            "degraded": self.cache_ttl // 3, # TTL réduit si dégradé
            "critical": self.cache_ttl // 6, # TTL très court si critique
            "unknown": self.cache_ttl // 2   # TTL moyen si inconnu
        }
        return ttl_map.get(status, self.cache_ttl // 2)
    
    def _get_adaptive_timeout(self, service_name: str, consecutive_failures: int) -> float:
        """Timeout adaptatif selon l'historique d'échecs"""
        base_timeout = 10.0  # Base 10 secondes
        
        if consecutive_failures == 0:
            return base_timeout
        elif consecutive_failures <= 2:
            return base_timeout * 1.5  # 15 secondes
        elif consecutive_failures <= 5:
            return base_timeout * 2.0  # 20 secondes
        else:
            return base_timeout * 2.5  # 25 secondes max
    
    def _determine_health_status(
        self, 
        consecutive_failures: int, 
        last_success: Optional[datetime]
    ) -> Literal["degraded", "critical", "unknown"]:
        """Détermine le statut de santé selon l'historique"""
        
        if consecutive_failures >= self._failure_thresholds["critical"]:
            return "critical"
        elif consecutive_failures >= self._failure_thresholds["degraded"]:
            return "degraded"
        else:
            # Si pas encore assez d'échecs mais pas de succès récent
            if last_success:
                time_since_success = (datetime.now(timezone.utc) - last_success).total_seconds()
                if time_since_success > 3600:  # 1 heure
                    return "degraded"
            return "unknown"
    
    def _update_cache(self, health: ServiceHealth) -> None:
        """Met à jour le cache avec limitation de taille"""
        if len(self._health_cache) > 20:
            # Nettoyer les entrées les plus anciennes
            oldest_key = min(self._health_cache.keys(), 
                           key=lambda k: self._health_cache[k].last_check)
            del self._health_cache[oldest_key]
        
        self._health_cache[health.service_name] = health
    
    def get_overall_health(self) -> Dict[str, Any]:
        """État de santé global de tous les services"""
        if not self._health_cache:
            return {"status": "unknown", "services": {}}
        
        services_status = {}
        critical_count = 0
        degraded_count = 0
        healthy_count = 0
        
        for service_name, health in self._health_cache.items():
            services_status[service_name] = {
                "status": health.status,
                "last_check": health.last_check.isoformat(),
                "consecutive_failures": health.consecutive_failures,
                "response_time_ms": health.response_time_ms
            }
            
            if health.status == "critical":
                critical_count += 1
            elif health.status == "degraded":
                degraded_count += 1
            elif health.status == "healthy":
                healthy_count += 1
        
        # Déterminer statut global
        if critical_count > 0:
            overall_status = "critical"
        elif degraded_count > 0:
            overall_status = "degraded"
        elif healthy_count > 0:
            overall_status = "healthy"
        else:
            overall_status = "unknown"
        
        return {
            "status": overall_status,
            "services": services_status,
            "summary": {
                "healthy": healthy_count,
                "degraded": degraded_count,
                "critical": critical_count,
                "total": len(self._health_cache)
            }
        }

# Instance globale du vérificateur amélioré
health_checker = AdvancedServiceHealthChecker(cache_ttl=600)  # 10 minutes

async def get_deepseek_client(request: Request) -> DeepSeekClient:
    """
    Récupération client DeepSeek depuis app state avec dégradation gracieuse
    
    N'interrompt JAMAIS le service pour un problème de health check.
    Utilise une logique de dégradation intelligente.
    
    Args:
        request: Requête FastAPI
        
    Returns:
        DeepSeekClient: Client DeepSeek (potentiellement dégradé mais fonctionnel)
        
    Raises:
        HTTPException: Seulement si client complètement indisponible
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
        
        # Vérification santé avec dégradation gracieuse
        try:
            # Utilisation de la méthode synchrone health_check() pour éviter l'erreur await
            health = await health_checker.check_service_health(
                "deepseek", 
                deepseek_client.health_check,  # Méthode synchrone
                critical_service=False  # DeepSeek n'est pas critique pour le démarrage
            )
            
            # Logique de dégradation au lieu de blocage complet
            if health.status == "critical":
                # Même en critique, on essaie de continuer avec warnings
                logger.error(
                    f"DeepSeek en état critique ({health.consecutive_failures} échecs) "
                    f"mais service continue en mode dégradé"
                )
                metrics_collector.increment_counter("dependencies.warnings.deepseek_critical")
                
                # Ajouter headers d'avertissement pour le client
                if hasattr(request, 'state'):
                    request.state.service_degraded = True
                    request.state.degradation_reason = "DeepSeek critical"
                
                return deepseek_client  # On continue quand même
                
            elif health.status == "degraded":
                logger.warning(
                    f"DeepSeek dégradé ({health.consecutive_failures} échecs récents) "
                    f"mais service opérationnel"
                )
                metrics_collector.increment_counter("dependencies.warnings.deepseek_degraded")
                
                if hasattr(request, 'state'):
                    request.state.service_degraded = True
                    request.state.degradation_reason = "DeepSeek degraded"
            
            elif health.status == "healthy":
                metrics_collector.increment_counter("dependencies.success.deepseek")
                if hasattr(request, 'state'):
                    request.state.service_degraded = False
            
            # Métriques de performance si disponibles
            if health.response_time_ms:
                metrics_collector.record_histogram("dependencies.deepseek.response_time", health.response_time_ms)
            
            return deepseek_client
            
        except Exception as health_error:
            # Health check a échoué mais on peut utiliser le client directement
            logger.warning(f"Health check DeepSeek échoué, utilisation directe: {str(health_error)}")
            metrics_collector.increment_counter("dependencies.warnings.deepseek_health_bypass")
            
            if hasattr(request, 'state'):
                request.state.service_degraded = True
                request.state.degradation_reason = "Health check failed"
            
            return deepseek_client  # Fail open - on essaie quand même
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions seulement pour les cas critiques
    except Exception as e:
        metrics_collector.increment_counter("dependencies.errors.deepseek_unexpected")
        logger.error(f"Erreur inattendue récupération DeepSeek: {str(e)}", exc_info=True)
        
        # En cas d'erreur inattendue, essayer de continuer si client existe
        if (hasattr(request.app.state, 'deepseek_client') and 
            request.app.state.deepseek_client):
            logger.warning("Utilisation DeepSeek en mode dégradé après erreur inattendue")
            return request.app.state.deepseek_client
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Erreur interne récupération service DeepSeek",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

async def get_cache_manager(request: Request) -> Optional[CacheManager]:
    """
    Récupération cache manager depuis app state avec dégradation gracieuse
    
    Le cache est toujours optionnel - ne bloque JAMAIS le service.
    
    Args:
        request: Requête FastAPI
        
    Returns:
        CacheManager ou None: Cache manager si disponible
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
        
        # Vérification santé cache (non bloquante avec timeout très court)
        try:
            # Utilisation de health_check si disponible, sinon skip
            cache_health_func = getattr(cache_manager, 'health_check', None)
            
            if cache_health_func:
                health = await asyncio.wait_for(
                    health_checker.check_service_health(
                        "cache", 
                        cache_health_func,
                        critical_service=False
                    ),
                    timeout=3.0  # Timeout court pour le cache
                )
                
                if health.status in ["healthy", "degraded"]:
                    metrics_collector.increment_counter("dependencies.success.cache")
                    if health.response_time_ms:
                        metrics_collector.record_histogram("dependencies.cache.response_time", health.response_time_ms)
                    
                    if health.status == "degraded":
                        logger.debug("Cache dégradé mais opérationnel")
                        metrics_collector.increment_counter("dependencies.warnings.cache_degraded")
                    
                    return cache_manager
                else:
                    logger.debug(f"Cache en état {health.status} - désactivation temporaire")
                    metrics_collector.increment_counter("dependencies.cache_unhealthy")
                    return None
            else:
                # Pas de health check, on assume que le cache fonctionne
                logger.debug("Cache sans health check - utilisation directe")
                return cache_manager
                
        except asyncio.TimeoutError:
            logger.debug("Health check cache timeout - utilisation directe en mode dégradé")
            metrics_collector.increment_counter("dependencies.cache_timeout")
            # Retourner le cache manager malgré le timeout (dégradation gracieuse)
            return cache_manager
            
        except Exception as e:
            logger.debug(f"Vérification santé cache échouée: {str(e)}")
            metrics_collector.increment_counter("dependencies.cache_check_failed")
            # Retourner le cache manager même si health check échoue
            return cache_manager
        
    except Exception as e:
        metrics_collector.increment_counter("dependencies.errors.cache_unexpected")
        logger.debug(f"Erreur récupération cache manager: {str(e)}")
        return None  # Toujours retourner None en cas d'erreur (non critique)

async def get_conversation_service_status(request: Request) -> Dict[str, Any]:
    """
    Vérification statut conversation service avec diagnostic détaillé
    
    Retourne toujours un statut, même en cas de problème.
    """
    try:
        # Vérification présence service dans app state
        if not hasattr(request.app.state, 'conversation_service'):
            metrics_collector.increment_counter("dependencies.errors.service_missing")
            logger.warning("Conversation service absent de app state - mode minimal")
            return {
                "status": "degraded",
                "service": "conversation_service",
                "error": "Service non initialisé",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        conversation_service = request.app.state.conversation_service
        
        if not conversation_service:
            metrics_collector.increment_counter("dependencies.errors.service_null")
            logger.warning("Conversation service null - mode minimal")
            return {
                "status": "degraded",
                "service": "conversation_service",
                "error": "Service null",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        # Vérification santé du service (non bloquante)
        service_healthy = getattr(conversation_service, 'service_healthy', True)
        if not service_healthy:
            initialization_error = getattr(conversation_service, 'initialization_error', 'Erreur inconnue')
            logger.warning(f"Service unhealthy mais continuation: {initialization_error}")
            metrics_collector.increment_counter("dependencies.warnings.service_degraded")
            
            return {
                "status": "degraded",
                "service": "conversation_service",
                "warning": "Service en mode dégradé",
                "initialization_error": initialization_error,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        # Construction statut détaillé avec gestion d'erreur
        try:
            service_config = getattr(request.app.state, 'service_config', {})
            service_start_time = getattr(request.app.state, 'service_start_time', datetime.now(timezone.utc))
            uptime_seconds = (datetime.now(timezone.utc) - service_start_time).total_seconds()
            
            # Intégrer l'état des sous-services
            overall_health = health_checker.get_overall_health()
            
            status = {
                "status": "healthy" if overall_health["status"] in ["healthy", "unknown"] else "degraded",
                "service": "conversation_service",
                "uptime_seconds": uptime_seconds,
                "phase": service_config.get("phase", 1),
                "version": service_config.get("version", "1.1.0"),
                "features": service_config.get("features", []),
                "json_output_enforced": service_config.get("json_output_enforced", True),
                "jwt_compatible": service_config.get("jwt_compatible", True),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sub_services": overall_health["summary"]
            }
            
        except Exception as status_error:
            logger.warning(f"Erreur construction statut détaillé: {str(status_error)}")
            # Statut minimal en cas d'erreur
            status = {
                "status": "degraded",
                "service": "conversation_service",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "warning": "Statut partiel disponible",
                "error": str(status_error)
            }
        
        metrics_collector.increment_counter("dependencies.success.service_status")
        return status
        
    except Exception as e:
        metrics_collector.increment_counter("dependencies.errors.service_unexpected")
        logger.error(f"Erreur vérification statut service: {str(e)}")
        
        # Retourner un statut d'erreur plutôt que de lever une exception
        return {
            "status": "critical",
            "service": "conversation_service",
            "error": "Erreur interne vérification statut",
            "details": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

async def validate_path_user_id(
    request: Request,
    path_user_id: int,
    token_user_id: int = Depends(get_current_user_id)
) -> int:
    """
    Validation user_id du path vs token JWT avec logging sécurité renforcé
    
    Cette fonction reste stricte car la sécurité ne peut pas être dégradée.
    
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
        raise  # Re-raise HTTP exceptions (sécurité stricte)
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
    
    Retourne toujours un contexte, même minimal en cas de problème.
    
    Args:
        request: Requête FastAPI
        user_id: ID utilisateur authentifié
        
    Returns:
        Dict: Contexte utilisateur
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
                ).total_seconds(),
                "service_degraded": getattr(request.state, 'service_degraded', False),
                "degradation_reason": getattr(request.state, 'degradation_reason', None)
            }
        except Exception as usage_error:
            logger.debug(f"Erreur métriques usage: {str(usage_error)}")
            usage_context = {
                "session_start": datetime.now(timezone.utc).isoformat(),
                "request_count": 1,
                "service_degraded": False
            }
        
        # Contexte technique (avec informations de service)
        try:
            service_config = getattr(request.app.state, 'service_config', {})
            overall_health = health_checker.get_overall_health()
            
            technical_context = {
                "phase": service_config.get("phase", 1),
                "features_available": service_config.get("features", ["intent_classification"]),
                "json_output_enforced": service_config.get("json_output_enforced", True),
                "jwt_compatible": service_config.get("jwt_compatible", True),
                "service_health": overall_health["status"],
                "healthy_services": overall_health["summary"].get("healthy", 0),
                "total_services": overall_health["summary"].get("total", 0)
            }
        except Exception as tech_error:
            logger.debug(f"Erreur contexte technique: {str(tech_error)}")
            technical_context = {
                "phase": 1,
                "features_available": ["intent_classification"],
                "service_health": "unknown"
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
        
        En cas d'erreur interne, laisse passer la requête (fail open).
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


async def get_multi_agent_team(request: Request) -> Optional[MultiAgentFinancialTeam]:
    """
    Récupération équipe multi-agents depuis app state avec fallback gracieux
    
    L'équipe est optionnelle - ne bloque JAMAIS le service.
    
    Args:
        request: Requête FastAPI
        
    Returns:
        MultiAgentFinancialTeam ou None: Équipe si disponible
    """
    try:
        import os
        
        # Vérification feature flag
        team_enabled = os.getenv("MULTI_AGENT_TEAM_ENABLED", "true").lower() == "true"
        if not team_enabled:
            logger.debug("Équipe multi-agents désactivée par configuration")
            return None
        
        # Vérification présence dans app state
        if not hasattr(request.app.state, 'multi_agent_team'):
            metrics_collector.increment_counter("dependencies.team_not_configured")
            logger.debug("Équipe multi-agents non configurée dans app state")
            return None
        
        multi_agent_team = request.app.state.multi_agent_team
        
        if not multi_agent_team:
            metrics_collector.increment_counter("dependencies.team_disabled")
            logger.debug("Équipe multi-agents désactivée - mode agents individuels")
            return None
        
        # Vérification santé équipe (non bloquante avec timeout court)
        try:
            team_health = await asyncio.wait_for(
                multi_agent_team.health_check(),
                timeout=2.0  # Timeout court pour l'équipe
            )
            
            if team_health["overall_status"] in ["healthy", "degraded"]:
                metrics_collector.increment_counter("dependencies.success.team")
                
                if team_health["overall_status"] == "degraded":
                    logger.debug("Équipe multi-agents dégradée mais opérationnelle")
                    metrics_collector.increment_counter("dependencies.warnings.team_degraded")
                
                return multi_agent_team
            else:
                logger.debug(f"Équipe multi-agents en état {team_health['overall_status']} - fallback agents individuels")
                metrics_collector.increment_counter("dependencies.team_unhealthy")
                return None
                
        except asyncio.TimeoutError:
            logger.debug("Health check équipe timeout - utilisation en mode dégradé")
            metrics_collector.increment_counter("dependencies.team_timeout")
            # Retourner l'équipe malgré le timeout (dégradation gracieuse)
            return multi_agent_team
            
        except Exception as e:
            logger.debug(f"Vérification santé équipe échouée: {str(e)}")
            metrics_collector.increment_counter("dependencies.team_check_failed")
            # Retourner l'équipe même si health check échoue (dégradation gracieuse)
            return multi_agent_team
        
    except Exception as e:
        metrics_collector.increment_counter("dependencies.errors.team_unexpected")
        logger.debug(f"Erreur récupération équipe multi-agents: {str(e)}")
        return None  # Toujours retourner None en cas d'erreur (non critique)


async def get_conversation_persistence(request: Request) -> Optional[ConversationPersistenceService]:
    """
    Récupération service persistence conversations avec gestion d'erreur robuste
    
    La persistence est optionnelle - ne bloque JAMAIS le service.
    
    Args:
        request: Requête FastAPI
        
    Returns:
        ConversationPersistenceService ou None: Service persistence si disponible
    """
    try:
        # Créer une session DB directe pour le service de persistence
        from db_service.session import SessionLocal
        
        try:
            db_session = SessionLocal()
            
            # Créer le service de persistence
            persistence_service = ConversationPersistenceService(db_session)
            
            # Vérification santé persistence (non bloquante avec timeout très court)
            try:
                # Test simple de connexion DB
                from sqlalchemy import text
                health = await health_checker.check_service_health(
                    "conversation_persistence",
                    lambda: db_session.execute(text("SELECT 1")).scalar() == 1,
                    critical_service=False
                )
                
                if health.status in ["healthy", "degraded"]:
                    metrics_collector.increment_counter("dependencies.success.persistence")
                    if health.response_time_ms:
                        metrics_collector.record_histogram("dependencies.persistence.response_time", health.response_time_ms)
                    
                    if health.status == "degraded":
                        logger.debug("Persistence dégradée mais opérationnelle")
                        metrics_collector.increment_counter("dependencies.warnings.persistence_degraded")
                    
                    return persistence_service
                else:
                    logger.debug(f"Persistence en état {health.status} - désactivation temporaire")
                    metrics_collector.increment_counter("dependencies.persistence_unhealthy")
                    db_session.close()
                    return None
                    
            except Exception as health_error:
                logger.debug(f"Vérification santé persistence échouée: {str(health_error)}")
                metrics_collector.increment_counter("dependencies.persistence_check_failed")
                # Retourner le service malgré l'échec du health check (dégradation gracieuse)
                return persistence_service
                
        except Exception as session_error:
            logger.debug(f"Erreur création session DB: {str(session_error)}")
            metrics_collector.increment_counter("dependencies.persistence_db_unavailable")
            return None
        
    except Exception as e:
        metrics_collector.increment_counter("dependencies.errors.persistence_unexpected")
        logger.debug(f"Erreur récupération service persistence: {str(e)}")
        return None  # Toujours retourner None en cas d'erreur (non critique)


async def get_conversation_processor(
    request: Request,
    deepseek_client: DeepSeekClient = Depends(get_deepseek_client),
    multi_agent_team: Optional[MultiAgentFinancialTeam] = Depends(get_multi_agent_team)
) -> Dict[str, Any]:
    """
    Processeur conversation avec choix automatique entre modes
    
    Détermine automatiquement le meilleur mode de traitement:
    - multi_agent_team: Si équipe AutoGen disponible et saine
    - single_agent: Fallback sur agents individuels
    
    Args:
        request: Requête FastAPI
        deepseek_client: Client DeepSeek (toujours requis)
        multi_agent_team: Équipe multi-agents (optionnelle)
        
    Returns:
        Dict contenant le processeur et métadonnées de mode
    """
    try:
        # Déterminer mode de traitement optimal
        if multi_agent_team:
            processing_mode = "multi_agent_team"
            metrics_collector.increment_counter("dependencies.mode_selection.multi_agent")
            logger.debug("Mode multi-agent sélectionné pour traitement")
        else:
            processing_mode = "single_agent"
            metrics_collector.increment_counter("dependencies.mode_selection.single_agent")
            logger.debug("Mode agent unique sélectionné (fallback ou préférence)")
        
        # Enrichir request.state avec informations mode
        if hasattr(request, 'state'):
            request.state.processing_mode = processing_mode
            request.state.multi_agent_available = multi_agent_team is not None
            request.state.fallback_available = deepseek_client is not None
        
        # Contexte processeur avec toutes les options disponibles
        processor_context = {
            "processing_mode": processing_mode,
            "multi_agent_team": multi_agent_team,
            "deepseek_client": deepseek_client,
            "fallback_chain": [
                "multi_agent_team" if multi_agent_team else None,
                "single_agent" if deepseek_client else None,
                "error"
            ],
            "capabilities": {
                "intent_classification": True,
                "entity_extraction": multi_agent_team is not None,
                "coherence_validation": multi_agent_team is not None,
                "team_context": multi_agent_team is not None,
                "caching": hasattr(request.app.state, 'cache_manager') and request.app.state.cache_manager is not None
            },
            "performance_expectations": {
                "multi_agent_team": {"target_ms": 2000, "features": "full"},
                "single_agent": {"target_ms": 800, "features": "intent_only"}
            }
        }
        
        metrics_collector.increment_counter("dependencies.processor_context_generated")
        return processor_context
        
    except Exception as e:
        metrics_collector.increment_counter("dependencies.errors.processor_selection")
        logger.error(f"Erreur sélection processeur conversation: {str(e)}")
        
        # Fallback critique: au minimum le client DeepSeek
        if deepseek_client:
            logger.warning("Fallback sur mode single_agent après erreur sélection processeur")
            return {
                "processing_mode": "single_agent",
                "multi_agent_team": None,
                "deepseek_client": deepseek_client,
                "fallback_chain": ["single_agent", "error"],
                "capabilities": {"intent_classification": True},
                "error": "Mode dégradé après erreur sélection"
            }
        else:
            logger.error("Aucun processeur disponible - service critique")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Service conversation temporairement indisponible",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )


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

# Utilitaires de debug et monitoring
def get_dependency_health_status() -> Dict[str, Any]:
    """Retourne le statut de santé des dépendances"""
    overall_health = health_checker.get_overall_health()
    
    return {
        "health_checker": {
            "cache_size": len(health_checker._health_cache),
            "cache_ttl": health_checker.cache_ttl,
            "overall_status": overall_health["status"],
            "services": overall_health["services"]
        },
        "rate_limiter": {
            "active_users": len(rate_limit_dependency.user_requests),
            "requests_per_minute": rate_limit_dependency.requests_per_minute,
            "requests_per_hour": rate_limit_dependency.requests_per_hour
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

def get_detailed_service_health() -> Dict[str, Any]:
    """Statut de santé détaillé de tous les services"""
    return health_checker.get_overall_health()

# Export des principales dépendances
__all__ = [
    'get_deepseek_client',
    'get_cache_manager', 
    'get_conversation_service_status',
    'validate_path_user_id',
    'get_user_context',
    'rate_limit_dependency',
    'with_rate_limit',
    'get_dependency_health_status',
    'get_detailed_service_health',
    'get_multi_agent_team',
    'get_conversation_processor',
    'get_conversation_persistence'
]