"""
Package API pour le Search Service.

Ce module expose les composants principaux de l'API :
- Routes et endpoints
- Middlewares
- Dépendances
- Configuration
"""

from .routes import get_router
from .middleware import setup_middleware
from .dependencies import (
    CurrentUser,
    ValidatedRequest,
    LexicalEngineService,
    RequestContext,
    RateLimited,
    get_current_user,
    validate_search_request,
    get_lexical_engine,
    format_error_response
)

__all__ = [
    # Router principal
    "get_router",
    
    # Configuration middleware
    "setup_middleware",
    
    # Dépendances FastAPI
    "CurrentUser",
    "ValidatedRequest", 
    "LexicalEngineService",
    "RequestContext",
    "RateLimited",
    
    # Fonctions de dépendances
    "get_current_user",
    "validate_search_request",
    "get_lexical_engine",
    
    # Utilitaires
    "format_error_response"
]