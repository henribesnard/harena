"""
API module pour le Conversation Service

Ce module contient toute l'interface REST du service :
- Routes FastAPI organisées
- Dépendances et middleware
- Gestion des erreurs HTTP
- Documentation automatique
"""

from .routes import (
    router,
    conversation_router,
    system_router
)

from .dependencies import (
    common_dependencies,
    conversation_dependencies,
    system_dependencies,
    get_current_user,
    validate_request_size,
    rate_limit_check,
    validate_headers,
    RequestTimer,
    get_request_timer
)

__version__ = "1.0.0"
__all__ = [
    # Routers
    "router",                    # Router principal (inclut tous les autres)
    "conversation_router",       # Router pour endpoints de conversation
    "system_router",            # Router pour endpoints système
    
    # Dépendances principales
    "common_dependencies",
    "conversation_dependencies", 
    "system_dependencies",
    
    # Dépendances spécialisées
    "get_current_user",
    "validate_request_size",
    "rate_limit_check", 
    "validate_headers",
    "RequestTimer",
    "get_request_timer"
]

# Configuration des endpoints
API_PREFIX = "/api/v1"
CONVERSATION_PREFIX = f"{API_PREFIX}/conversation"
SYSTEM_ENDPOINTS = ["/health", "/metrics"]

# Tags pour la documentation
CONVERSATION_TAG = "conversation"
SYSTEM_TAG = "system"
MONITORING_TAG = "monitoring"

def get_api_info():
    """Informations sur l'API"""
    return {
        "version": __version__,
        "prefix": API_PREFIX,
        "conversation_prefix": CONVERSATION_PREFIX,
        "endpoints": {
            "conversation": [
                f"{CONVERSATION_PREFIX}/chat",
                f"{CONVERSATION_PREFIX}/config",
                f"{CONVERSATION_PREFIX}/clear-cache"
            ],
            "system": [
                "/health",
                "/metrics",
                "/"
            ]
        },
        "documentation": "/docs",
        "openapi": "/openapi.json"
    }

def get_available_routes():
    """Liste toutes les routes disponibles"""
    routes = []
    for route in router.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            routes.append({
                "path": route.path,
                "methods": list(route.methods),
                "name": getattr(route, 'name', 'unnamed')
            })
    return routes