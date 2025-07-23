"""
📦 API package pour conversation service

Ce module centralise les exports de l'API conversation service
pour garantir la compatibilité avec le système de chargement
dynamique des routers dans local_app.py et heroku_app.py.

Pattern standardisé identique à search_service.
"""

from .routes import router, initialize_intent_engine

# Export explicite pour load_service_router() et initialisation
__all__ = ["router", "initialize_intent_engine"]

# Métadonnées du module API
__version__ = "1.0.0"
__description__ = "API REST pour conversation service avec Intent Detection"
__author__ = "Harena Finance Platform"

# Configuration des endpoints disponibles
API_ENDPOINTS = {
    "chat": {
        "path": "/chat",
        "method": "POST",
        "description": "Classification d'intentions avec pipeline L0→L1→L2"
    },
    "health": {
        "path": "/health", 
        "method": "GET",
        "description": "Health check avec métriques performance"
    },
    "metrics": {
        "path": "/metrics",
        "method": "GET", 
        "description": "Métriques détaillées du service"
    },
    "status": {
        "path": "/status",
        "method": "GET",
        "description": "Status et informations du service"
    },
    "debug": {
        "path": "/debug/test-levels",
        "method": "POST",
        "description": "Test debug des niveaux L0/L1/L2"
    }
}

# Informations pour le chargement dynamique
ROUTER_INFO = {
    "name": "conversation_service",
    "version": __version__,
    "description": __description__,
    "endpoints_count": len(API_ENDPOINTS),
    "prefix": "/api/v1/conversation",
    "tags": ["conversation", "intent-detection", "deepseek"],
    "dependencies": [
        "conversation_service.agents.intent_classifier",
        "conversation_service.utils"
    ]
}