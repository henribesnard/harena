"""
Dépendances FastAPI pour le Search Service.

Ce module contient toutes les dépendances injectées dans FastAPI pour :
- Validation des requêtes
- Authentification et sécurité
- Rate limiting
- Formatage des erreurs
- Injection des services
"""

from typing import Optional, Dict, Any, Annotated
from fastapi import Depends, HTTPException, Request, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import ValidationError
import time
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import redis
from functools import lru_cache

from ..config.settings import get_settings
from ..core.lexical_engine import LexicalEngine
from ..clients.elasticsearch_client import ElasticsearchClient
from ..models.requests import (
    LexicalSearchRequest, 
    SearchOptions, 
    QueryOptions, 
    ResultOptions
)
from ..models.responses import ErrorResponse, ErrorDetail
from ..utils.validators import RequestValidator
from ..utils.cache import CacheManager

logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================

@lru_cache()
def get_settings_cached():
    """Récupère les paramètres de configuration (mise en cache)."""
    return get_settings()

@lru_cache()
def get_redis_client():
    """Créé un client Redis pour le rate limiting (mise en cache)."""
    settings = get_settings_cached()
    if not settings.redis_url:
        return None
    
    try:
        return redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5
        )
    except Exception as e:
        logger.warning(f"Impossible de connecter à Redis: {e}")
        return None

# ==================== AUTHENTIFICATION ====================

security = HTTPBearer(auto_error=False)

class AuthenticationError(HTTPException):
    """Exception d'authentification personnalisée."""
    def __init__(self, detail: str = "Authentication required"):
        super().__init__(
            status_code=401,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"}
        )

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_api_key: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """
    Authentification de l'utilisateur courant.
    
    Supporte Bearer token et API key.
    """
    settings = get_settings_cached()
    
    # Mode développement - pas d'authentification
    if not settings.auth_enabled:
        return {"user_id": "dev_user", "role": "developer"}
    
    # Vérification API Key
    if x_api_key:
        if x_api_key in settings.api_keys:
            return {
                "user_id": settings.api_keys[x_api_key].get("user_id", "api_user"),
                "role": settings.api_keys[x_api_key].get("role", "user"),
                "auth_method": "api_key"
            }
        else:
            raise AuthenticationError("Invalid API key")
    
    # Vérification Bearer token
    if credentials:
        # Ici, vous pourriez valider le JWT token
        # Pour l'exemple, nous acceptons tout token non vide
        if len(credentials.credentials) > 10:
            return {
                "user_id": "token_user",
                "role": "user",
                "auth_method": "bearer_token"
            }
        else:
            raise AuthenticationError("Invalid bearer token")
    
    raise AuthenticationError("No authentication provided")

# ==================== RATE LIMITING ====================

class RateLimitError(HTTPException):
    """Exception de rate limiting."""
    def __init__(self, detail: str = "Rate limit exceeded"):
        super().__init__(
            status_code=429,
            detail=detail,
            headers={"Retry-After": "60"}
        )

class RateLimiter:
    """Gestionnaire de rate limiting avec Redis ou mémoire."""
    
    def __init__(self):
        self.redis_client = get_redis_client()
        self.memory_store = defaultdict(list)
        self.settings = get_settings_cached()
    
    def _get_key(self, identifier: str, window: str) -> str:
        """Génère une clé pour le rate limiting."""
        return f"rate_limit:{identifier}:{window}"
    
    def _check_redis_limit(self, key: str, limit: int, window: int) -> bool:
        """Vérifie la limite avec Redis."""
        try:
            pipe = self.redis_client.pipeline()
            pipe.incr(key)
            pipe.expire(key, window)
            results = pipe.execute()
            return results[0] <= limit
        except Exception as e:
            logger.warning(f"Erreur Redis rate limiting: {e}")
            return True  # Fail open
    
    def _check_memory_limit(self, identifier: str, limit: int, window: int) -> bool:
        """Vérifie la limite en mémoire."""
        now = time.time()
        window_start = now - window
        
        # Nettoie les anciens enregistrements
        self.memory_store[identifier] = [
            timestamp for timestamp in self.memory_store[identifier]
            if timestamp > window_start
        ]
        
        # Vérifie la limite
        if len(self.memory_store[identifier]) >= limit:
            return False
        
        # Ajoute l'enregistrement courant
        self.memory_store[identifier].append(now)
        return True
    
    def check_limit(self, identifier: str, limit: int, window: int) -> bool:
        """Vérifie si la limite est respectée."""
        if self.redis_client:
            key = self._get_key(identifier, f"{window}s")
            return self._check_redis_limit(key, limit, window)
        else:
            return self._check_memory_limit(identifier, limit, window)

rate_limiter = RateLimiter()

async def check_rate_limit(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> None:
    """
    Vérifie les limites de taux par utilisateur.
    
    Différentes limites selon le rôle utilisateur.
    """
    settings = get_settings_cached()
    
    if not settings.rate_limit_enabled:
        return
    
    # Identifier l'utilisateur
    user_id = current_user["user_id"]
    role = current_user.get("role", "user")
    
    # Limites par rôle
    limits = {
        "developer": (1000, 3600),  # 1000 req/h
        "premium": (500, 3600),     # 500 req/h
        "user": (100, 3600),        # 100 req/h
        "free": (50, 3600)          # 50 req/h
    }
    
    limit, window = limits.get(role, limits["user"])
    
    if not rate_limiter.check_limit(user_id, limit, window):
        raise RateLimitError(
            f"Rate limit exceeded for role '{role}': {limit} requests per hour"
        )

# ==================== VALIDATION ====================

class ValidationError(HTTPException):
    """Exception de validation personnalisée."""
    def __init__(self, errors: list):
        detail = ErrorResponse(
            error=ErrorDetail(
                type="validation_error",
                message="Request validation failed",
                details={"validation_errors": errors}
            ),
            request_id=None,
            timestamp=datetime.utcnow()
        )
        super().__init__(
            status_code=422,
            detail=detail.dict()
        )

async def validate_search_request(
    request: LexicalSearchRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> LexicalSearchRequest:
    """
    Valide une requête de recherche lexicale.
    
    Vérifie la structure, les limites et la sécurité.
    """
    validator = RequestValidator()
    
    try:
        # Validation de base
        validator.validate_query_text(request.query)
        validator.validate_filters(request.filters)
        validator.validate_pagination(request.from_, request.size)
        
        # Validation des options
        if request.options:
            validator.validate_search_options(request.options)
        
        # Validation basée sur le rôle
        role = current_user.get("role", "user")
        validator.validate_role_permissions(request, role)
        
        return request
        
    except ValidationError as e:
        logger.warning(f"Validation error for user {current_user['user_id']}: {e}")
        raise ValidationError(e.errors if hasattr(e, 'errors') else [str(e)])
    except Exception as e:
        logger.error(f"Unexpected validation error: {e}")
        raise ValidationError([f"Validation failed: {str(e)}"])

# ==================== SERVICES ====================

@lru_cache()
def get_elasticsearch_client() -> ElasticsearchClient:
    """Créé et retourne le client Elasticsearch (singleton)."""
    settings = get_settings_cached()
    return ElasticsearchClient(settings)

@lru_cache()
def get_cache_manager() -> CacheManager:
    """Créé et retourne le gestionnaire de cache (singleton)."""
    settings = get_settings_cached()
    return CacheManager(settings)

@lru_cache()
def get_lexical_engine() -> LexicalEngine:
    """Créé et retourne le moteur lexical (singleton)."""
    es_client = get_elasticsearch_client()
    cache_manager = get_cache_manager()
    settings = get_settings_cached()
    
    return LexicalEngine(
        elasticsearch_client=es_client,
        cache_manager=cache_manager,
        settings=settings
    )

# ==================== UTILITAIRES ====================

def get_request_context(request: Request) -> Dict[str, Any]:
    """Extrait le contexte de la requête HTTP."""
    return {
        "client_ip": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
        "request_id": request.headers.get("x-request-id"),
        "timestamp": datetime.utcnow(),
        "method": request.method,
        "path": request.url.path,
        "query_params": dict(request.query_params)
    }

def format_error_response(
    error: Exception,
    request_id: Optional[str] = None
) -> ErrorResponse:
    """Formate une réponse d'erreur standardisée."""
    error_type = type(error).__name__
    
    if isinstance(error, HTTPException):
        error_detail = ErrorDetail(
            type=error_type,
            message=str(error.detail),
            code=error.status_code
        )
    else:
        error_detail = ErrorDetail(
            type=error_type,
            message=str(error),
            code=500
        )
    
    return ErrorResponse(
        error=error_detail,
        request_id=request_id,
        timestamp=datetime.utcnow()
    )

# ==================== DÉPENDANCES COMMUNES ====================

# Alias pour faciliter l'utilisation
CurrentUser = Annotated[Dict[str, Any], Depends(get_current_user)]
ValidatedRequest = Annotated[LexicalSearchRequest, Depends(validate_search_request)]
LexicalEngineService = Annotated[LexicalEngine, Depends(get_lexical_engine)]
RequestContext = Annotated[Dict[str, Any], Depends(get_request_context)]
RateLimited = Annotated[None, Depends(check_rate_limit)]