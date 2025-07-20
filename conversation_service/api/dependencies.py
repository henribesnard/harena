"""
Dépendances FastAPI avec intégration architecture hybride
Injection services Intent Detection Engine et Redis
"""

import time
from typing import Dict, Any, Optional
from fastapi import Depends, HTTPException, Request, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from conversation_service.config import settings
from conversation_service.intent_detection import IntentDetectionEngine
from conversation_service.cache import RedisManager
from conversation_service.utils.logging import get_logger
from conversation_service.utils.validators import RequestValidator
from conversation_service.utils.performance import PerformanceMonitor

logger = get_logger(__name__)

# Security pour authentification optionnelle
security = HTTPBearer(auto_error=False)


class RequestTimer:
    """Context manager pour timing requêtes"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
    
    @property
    def duration_ms(self) -> int:
        if self.start_time and self.end_time:
            return int((self.end_time - self.start_time) * 1000)
        return 0


# === DÉPENDANCES DE BASE ===

async def get_request_timer() -> RequestTimer:
    """Timer pour mesurer performance requêtes"""
    return RequestTimer()


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict[str, Any]]:
    """
    Extraction utilisateur depuis token (optionnel)
    En production: intégration avec système auth
    """
    
    if not credentials:
        # Mode anonyme autorisé
        return {
            "id": 1,  # User ID anonymous par défaut
            "anonymous": True,
            "tier": "free"
        }
    
    # TODO: Validation JWT token en production
    # Pour l'instant: validation simple
    token = credentials.credentials
    
    if token == "demo_token":
        return {
            "id": 12345,
            "username": "demo_user",
            "anonymous": False,
            "tier": "premium"
        }
    
    # Token invalide = mode anonyme
    return {
        "id": 1,
        "anonymous": True,
        "tier": "free"
    }


async def validate_request_size(
    request: Request,
    content_length: Optional[str] = Header(None)
):
    """Validation taille requête"""
    if content_length:
        size = int(content_length)
        if size > settings.MAX_REQUEST_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Request too large: {size} bytes (max: {settings.MAX_REQUEST_SIZE})"
            )


async def rate_limit_check(request: Request):
    """Rate limiting basique (amélioration avec Redis en production)"""
    # TODO: Implémentation rate limiting avec Redis
    # Pour l'instant: validation basique
    pass


async def validate_headers(request: Request):
    """Validation headers requis"""
    # Validation Content-Type pour POST/PUT
    if request.method in ["POST", "PUT"]:
        content_type = request.headers.get("content-type", "")
        if not content_type.startswith("application/json"):
            raise HTTPException(
                status_code=415,
                detail="Content-Type must be application/json"
            )


# === DÉPENDANCES SERVICES ===

async def get_intent_engine(request: Request) -> IntentDetectionEngine:
    """Injection Intent Detection Engine"""
    if hasattr(request.app.state, 'intent_engine'):
        return request.app.state.intent_engine
    else:
        raise HTTPException(
            status_code=503,
            detail="Intent Detection Engine not available"
        )


async def get_redis_manager(request: Request) -> RedisManager:
    """Injection Redis Manager"""
    if hasattr(request.app.state, 'redis_manager'):
        return request.app.state.redis_manager
    else:
        raise HTTPException(
            status_code=503,
            detail="Redis Manager not available"
        )


async def get_performance_monitor(request: Request) -> PerformanceMonitor:
    """Injection Performance Monitor"""
    if hasattr(request.state, 'performance_monitor'):
        return request.state.performance_monitor
    else:
        return PerformanceMonitor()


# === DÉPENDANCES COMPOSÉES ===

async def common_dependencies(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
    _: None = Depends(validate_request_size),
    __: None = Depends(rate_limit_check),
    ___: None = Depends(validate_headers)
):
    """
    Dépendances communes pour tous les endpoints
    
    Returns:
        dict: Informations de contexte de la requête
    """
    
    return {
        "request": request,
        "user": current_user,
        "client_ip": request.client.host if request.client else "unknown",
        "user_agent": request.headers.get("user-agent", ""),
        "timestamp": time.time()
    }


async def conversation_dependencies(
    context: dict = Depends(common_dependencies),
    intent_engine: IntentDetectionEngine = Depends(get_intent_engine),
    redis_manager: RedisManager = Depends(get_redis_manager)
):
    """
    Dépendances spécifiques aux endpoints de conversation
    
    Returns:
        dict: Contexte enrichi pour la conversation
    """
    
    # Vérifications additionnelles pour les conversations
    request = context["request"]
    
    # Validation message conversation si applicable
    if request.method == "POST":
        # Le middleware de validation s'occupe déjà de la validation basique
        pass
    
    # Log de la requête de conversation
    logger.info(f"Conversation request from {context['client_ip']}")
    
    # Ajout services au contexte
    context["intent_engine"] = intent_engine
    context["redis_manager"] = redis_manager
    
    return context


async def system_dependencies(
    request: Request,
    _: None = Depends(validate_request_size)
):
    """
    Dépendances allégées pour les endpoints système
    
    Returns:
        dict: Contexte minimal pour les endpoints système
    """
    
    return {
        "request": request,
        "client_ip": request.client.host if request.client else "unknown",
        "timestamp": time.time()
    }


async def admin_dependencies(
    context: dict = Depends(common_dependencies)
):
    """
    Dépendances pour endpoints admin (avec vérification permissions)
    """
    
    user = context["user"]
    
    # Vérification permissions admin
    if user.get("anonymous", True) or user.get("tier") != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin permissions required"
        )
    
    return context


# === HELPERS VALIDATION ===

async def validate_conversation_request(request_data: dict) -> dict:
    """Validation spécialisée requête conversation"""
    
    message = request_data.get("message", "")
    validation_result = RequestValidator.validate_conversation_message(message)
    
    if not validation_result["valid"]:
        raise HTTPException(
            status_code=400,
            detail={
                "type": "validation_error",
                "errors": validation_result["errors"],
                "warnings": validation_result.get("warnings", [])
            }
        )
    
    # Retourner message sanitized
    request_data["message"] = validation_result["sanitized_message"]
    return request_data


# Export dépendances principales
__all__ = [
    "RequestTimer",
    "get_request_timer",
    "get_current_user",
    "get_intent_engine",
    "get_redis_manager",
    "get_performance_monitor",
    "common_dependencies",
    "conversation_dependencies",
    "system_dependencies",
    "admin_dependencies",
    "validate_conversation_request"
]