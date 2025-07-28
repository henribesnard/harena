"""
🔗 Injection de Dépendances FastAPI

Système d'injection de dépendances pour l'API FastAPI avec gestion
du cycle de vie des services et instances singleton.
"""

import logging
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from conversation_service.services.intent_detection.detector import get_intent_service_sync, OptimizedIntentService
from conversation_service.clients.cache.memory_cache import get_memory_cache, IntelligentMemoryCache
from conversation_service.utils.monitoring.intent_metrics import get_metrics_collector, IntentMetricsCollector
from conversation_service.config import config
from conversation_service.models.exceptions import ServiceUnavailableError

logger = logging.getLogger(__name__)

# Schéma d'authentification optionnel
security = HTTPBearer(auto_error=False)


async def get_intent_service() -> OptimizedIntentService:
    """
    Dépendance : Service principal de détection d'intention
    
    Returns:
        Instance du service optimisé
        
    Raises:
        HTTPException: Si service indisponible
    """
    try:
        service = get_intent_service_sync()
        
        # Vérification santé du service
        if not hasattr(service, 'rule_engine') or service.rule_engine is None:
            raise ServiceUnavailableError(
                "Service de détection d'intention non initialisé",
                service_name="intent_detection"
            )
        
        return service
        
    except Exception as e:
        logger.error(f"Erreur récupération service intention: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Service indisponible",
                "message": "Le service de détection d'intention est temporairement indisponible",
                "retry_after": 30
            }
        )


def get_cache_manager() -> IntelligentMemoryCache:
    """
    Dépendance : Gestionnaire de cache mémoire
    
    Returns:
        Instance du gestionnaire de cache
    """
    try:
        return get_memory_cache()
    except Exception as e:
        logger.warning(f"Erreur récupération cache: {e}")
        # Retourner cache par défaut si erreur
        from conversation_service.clients.cache.memory_cache import IntelligentMemoryCache
        return IntelligentMemoryCache(max_size=100)


def get_metrics_manager() -> IntentMetricsCollector:
    """
    Dépendance : Collecteur de métriques
    
    Returns:
        Instance du collecteur de métriques
    """
    try:
        return get_metrics_collector()
    except Exception as e:
        logger.warning(f"Erreur récupération métriques: {e}")
        # Retourner collecteur par défaut si erreur
        from conversation_service.utils.monitoring.intent_metrics import IntentMetricsCollector
        return IntentMetricsCollector()


async def validate_request_size(request: Request) -> bool:
    """
    Dépendance : Validation taille des requêtes
    
    Args:
        request: Requête FastAPI
        
    Returns:
        True si taille valide
        
    Raises:
        HTTPException: Si requête trop grande
    """
    content_length = request.headers.get('content-length')
    
    if content_length:
        content_length = int(content_length)
        max_size = 1024 * 1024  # 1MB max
        
        if content_length > max_size:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "Requête trop grande",
                    "message": f"Taille maximale autorisée: {max_size} bytes",
                    "received_size": content_length
                }
            )
    
    return True


async def get_user_context(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """
    Dépendance : Extraction contexte utilisateur
    
    Args:
        request: Requête FastAPI
        credentials: Credentials optionnels
        
    Returns:
        Contexte utilisateur avec métadonnées
    """
    context = {
        "user_id": None,
        "ip_address": request.client.host if request.client else "unknown",
        "user_agent": request.headers.get("user-agent", "unknown"),
        "authenticated": False,
        "request_id": None,
        "session_info": {}
    }
    
    # Extraction user_id depuis headers ou auth
    if credentials and credentials.credentials:
        try:
            # Ici on pourrait valider le token et extraire user_id
            # Pour l'instant, on accepte tout token comme user_id
            context["user_id"] = credentials.credentials[:50]  # Limitation sécurité
            context["authenticated"] = True
        except Exception as e:
            logger.warning(f"Erreur validation token: {e}")
    
    # User ID depuis header custom
    user_id_header = request.headers.get("X-User-ID")
    if user_id_header and not context["user_id"]:
        context["user_id"] = user_id_header[:50]
    
    # Request ID pour traçabilité
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        import uuid
        request_id = str(uuid.uuid4())[:8]
    context["request_id"] = request_id
    
    # Informations de session
    session_id = request.headers.get("X-Session-ID")
    if session_id:
        context["session_info"]["session_id"] = session_id
    
    return context


async def check_rate_limit(
    user_context: Dict[str, Any] = Depends(get_user_context),
    cache_manager: IntelligentMemoryCache = Depends(get_cache_manager)
) -> bool:
    """
    Dépendance : Vérification rate limiting
    
    Args:
        user_context: Contexte utilisateur
        cache_manager: Gestionnaire de cache
        
    Returns:
        True si sous la limite
        
    Raises:
        HTTPException: Si limite dépassée
    """
    # Rate limiting basique basé sur IP + user_id
    rate_limit_key = f"rate_limit:{user_context['ip_address']}:{user_context.get('user_id', 'anonymous')}"
    
    # Pour l'instant, rate limiting simple (pourrait être enrichi)
    # Limite: 100 requêtes par minute par IP/user
    import time
    current_minute = int(time.time() // 60)
    minute_key = f"{rate_limit_key}:{current_minute}"
    
    # Utilisation cache pour compter requêtes
    # (Implémentation simplifiée - en production utiliser Redis)
    requests_this_minute = 1  # Placeholder
    max_requests_per_minute = 100
    
    if requests_this_minute > max_requests_per_minute:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit dépassé",
                "message": f"Maximum {max_requests_per_minute} requêtes par minute",
                "retry_after": 60,
                "current_usage": requests_this_minute
            }
        )
    
    return True


async def validate_service_health(
    intent_service: OptimizedIntentService = Depends(get_intent_service)
) -> Dict[str, str]:
    """
    Dépendance : Validation santé globale des services
    
    Args:
        intent_service: Service principal
        
    Returns:
        Status des composants
        
    Raises:
        HTTPException: Si services critiques défaillants
    """
    try:
        # Vérification santé complète
        health_status = await intent_service.health_check()
        
        # Vérification composants critiques
        critical_components = ["rule_engine", "entity_extractor", "text_cleaner"]
        failed_critical = [
            comp for comp in critical_components
            if health_status["components"].get(comp, "error").startswith("error")
        ]
        
        if failed_critical:
            logger.error(f"Composants critiques défaillants: {failed_critical}")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Services critiques indisponibles",
                    "failed_components": failed_critical,
                    "retry_after": 60
                }
            )
        
        return health_status["components"]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur vérification santé: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Erreur interne",
                "message": "Impossible de vérifier l'état des services"
            }
        )


def get_request_validator(
    require_auth: bool = False,
    require_health_check: bool = True,
    enable_rate_limiting: bool = True
):
    """
    Factory de dépendances : Créateur de validateur de requête configurable
    
    Args:
        require_auth: Authentification requise
        require_health_check: Vérification santé requise
        enable_rate_limiting: Rate limiting activé
        
    Returns:
        Fonction de validation configurée
    """
    async def request_validator(
        request: Request,
        user_context: Dict[str, Any] = Depends(get_user_context),
        request_size_ok: bool = Depends(validate_request_size),
        rate_limit_ok: bool = Depends(check_rate_limit) if enable_rate_limiting else None,
        service_health: Dict[str, str] = Depends(validate_service_health) if require_health_check else None
    ) -> Dict[str, Any]:
        """Validateur de requête configuré"""
        
        # Vérification authentification si requise
        if require_auth and not user_context["authenticated"]:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "Authentification requise",
                    "message": "Cette endpoint nécessite une authentification"
                }
            )
        
        # Logging requête validée
        logger.info(
            f"Requête validée: {request.method} {request.url.path} "
            f"from {user_context['ip_address']} "
            f"(user: {user_context.get('user_id', 'anonymous')})"
        )
        
        return {
            "validated": True,
            "user_context": user_context,
            "service_health": service_health,
            "request_metadata": {
                "method": request.method,
                "path": request.url.path,
                "timestamp": time.time()
            }
        }
    
    return request_validator


# Dépendances pré-configurées pour différents niveaux de sécurité
public_endpoint_validator = get_request_validator(
    require_auth=False,
    require_health_check=False,
    enable_rate_limiting=True
)

protected_endpoint_validator = get_request_validator(
    require_auth=True,
    require_health_check=True,
    enable_rate_limiting=True
)

admin_endpoint_validator = get_request_validator(
    require_auth=True,
    require_health_check=True,
    enable_rate_limiting=False  # Admins sans limite
)


async def get_configuration() -> Dict[str, Any]:
    """
    Dépendance : Configuration actuelle du service
    
    Returns:
        Configuration active
    """
    return {
        "service_name": config.service.service_name,
        "version": config.service.version,
        "deepseek_enabled": config.service.enable_deepseek_fallback,
        "cache_enabled": config.rule_engine.enable_cache,
        "performance_targets": {
            "latency_ms": config.performance.target_latency_ms,
            "accuracy": config.performance.target_accuracy
        },
        "feature_flags": {
            "batch_processing": config.service.enable_batch_processing,
            "metrics_collection": config.service.enable_metrics_collection
        }
    }


# Import time pour time.time()
import time

# Exports publics
__all__ = [
    "get_intent_service",
    "get_cache_manager", 
    "get_metrics_manager",
    "validate_request_size",
    "get_user_context",
    "check_rate_limit",
    "validate_service_health",
    "get_request_validator",
    "public_endpoint_validator",
    "protected_endpoint_validator", 
    "admin_endpoint_validator",
    "get_configuration"
]