"""
API module pour le Conversation Service

Ce module contient toute l'interface REST du service :
- Routes FastAPI organisées avec router unique
- Dépendances et middleware
- Gestion des erreurs HTTP
- Documentation automatique
"""

from .routes import router

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
    # Router principal (architecture simplifiée)
    "router",                    # Router unique avec tous les endpoints
    
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

# Tags pour la documentation
CONVERSATION_TAG = "conversation"
SYSTEM_TAG = "system"
MONITORING_TAG = "monitoring"

def get_api_info():
    """Informations sur l'API"""
    return {
        "version": __version__,
        "architecture": "simplified_single_router",
        "prefix": API_PREFIX,
        "conversation_prefix": CONVERSATION_PREFIX,
        "endpoints": {
            "conversation": [
                f"{CONVERSATION_PREFIX}/chat"
            ],
            "system": [
                f"{CONVERSATION_PREFIX}/health",
                f"{CONVERSATION_PREFIX}/metrics",
                f"{CONVERSATION_PREFIX}/config",
                f"{CONVERSATION_PREFIX}/clear-cache"
            ]
        },
        "documentation": "/docs",
        "openapi": "/openapi.json"
    }

def get_available_routes():
    """Liste toutes les routes disponibles du router principal"""
    routes = []
    for route in router.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            routes.append({
                "path": route.path,
                "methods": list(route.methods),
                "name": getattr(route, 'name', 'unnamed'),
                "tags": getattr(route, 'tags', [])
            })
    return routes

def get_router_info():
    """Informations détaillées sur le router"""
    return {
        "router_type": "single_unified_router",
        "total_routes": len(router.routes),
        "routes_by_tag": _group_routes_by_tag(),
        "available_methods": _get_available_methods()
    }

def _group_routes_by_tag():
    """Groupe les routes par tag"""
    routes_by_tag = {}
    for route in router.routes:
        tags = getattr(route, 'tags', ['untagged'])
        for tag in tags:
            if tag not in routes_by_tag:
                routes_by_tag[tag] = []
            routes_by_tag[tag].append({
                "path": route.path,
                "methods": list(getattr(route, 'methods', []))
            })
    return routes_by_tag

def _get_available_methods():
    """Récupère toutes les méthodes HTTP disponibles"""
    methods = set()
    for route in router.routes:
        if hasattr(route, 'methods'):
            methods.update(route.methods)
    return sorted(list(methods))