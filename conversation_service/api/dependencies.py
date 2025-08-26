"""
Dépendances FastAPI pour conversation service
"""
import logging
from typing import Optional
from fastapi import Request, HTTPException, Depends
from conversation_service.clients.deepseek_client import DeepSeekClient
from conversation_service.core.cache_manager import CacheManager
from conversation_service.api.middleware.auth_middleware import get_current_user_id, verify_user_id_match

# Configuration du logger
logger = logging.getLogger("conversation_service.dependencies")

async def get_deepseek_client(request: Request) -> DeepSeekClient:
    """Récupération client DeepSeek depuis app state"""
    deepseek_client = getattr(request.app.state, 'deepseek_client', None)
    
    if not deepseek_client:
        logger.error("DeepSeek client non disponible dans app state")
        raise HTTPException(
            status_code=503,
            detail="Service DeepSeek temporairement indisponible"
        )
    
    return deepseek_client

async def get_cache_manager(request: Request) -> CacheManager:
    """Récupération cache manager depuis app state"""
    cache_manager = getattr(request.app.state, 'cache_manager', None)
    
    if not cache_manager:
        logger.error("Cache manager non disponible dans app state")
        raise HTTPException(
            status_code=503,
            detail="Service cache temporairement indisponible"
        )
    
    return cache_manager

async def get_conversation_service_status(request: Request) -> dict:
    """Vérification statut conversation service"""
    conversation_service = getattr(request.app.state, 'conversation_service', None)
    
    if not conversation_service:
        raise HTTPException(
            status_code=503,
            detail="Conversation service non initialisé"
        )
    
    if not conversation_service.service_healthy:
        raise HTTPException(
            status_code=503,
            detail="Conversation service en maintenance"
        )
    
    return {"status": "healthy", "service": "conversation_service"}

async def validate_path_user_id(
    request: Request,
    path_user_id: int,
    token_user_id: int = Depends(get_current_user_id)
) -> int:
    """Validation user_id du path vs token JWT"""
    await verify_user_id_match(request, path_user_id)
    return path_user_id

async def get_user_context(
    request: Request,
    user_id: int = Depends(get_current_user_id)
) -> Optional[dict]:
    """Récupération contexte utilisateur (Phase 1: vide, sera enrichi phases suivantes)"""
    
    # Phase 1: pas de contexte persistant
    # Sera enrichi dans les phases suivantes avec:
    # - Historique conversations récentes
    # - Préférences utilisateur  
    # - Patterns comportementaux
    
    basic_context = {
        "user_id": user_id,
        "session_start": request.state.__dict__.get("session_start"),
        "request_count": getattr(request.state, "request_count", 1)
    }
    
    return basic_context

class RateLimitDependency:
    """Dépendance pour rate limiting par utilisateur"""
    
    def __init__(self, max_requests_per_minute: int = 60):
        self.max_requests = max_requests_per_minute
        self.user_requests = {}  # Simple in-memory counter (Phase 1)
        
    async def __call__(self, user_id: int = Depends(get_current_user_id)) -> None:
        """Vérification rate limit utilisateur"""
        
        # Phase 1: rate limiting simple en mémoire
        # Sera remplacé par Redis dans phases suivantes
        
        current_minute = int(time.time() // 60)
        user_key = f"{user_id}:{current_minute}"
        
        current_count = self.user_requests.get(user_key, 0)
        
        if current_count >= self.max_requests:
            logger.warning(f"Rate limit exceeded for user {user_id}: {current_count} req/min")
            raise HTTPException(
                status_code=429,
                detail=f"Trop de requêtes. Limite: {self.max_requests} par minute"
            )
        
        self.user_requests[user_key] = current_count + 1
        
        # Nettoyage périodique (garder seulement minute courante et précédente)
        keys_to_remove = [
            key for key in self.user_requests.keys() 
            if int(key.split(':')[1]) < current_minute - 1
        ]
        for key in keys_to_remove:
            del self.user_requests[key]

# Instances des dépendances
rate_limit_dependency = RateLimitDependency(max_requests_per_minute=60)

# Import pour faciliter l'usage
import time