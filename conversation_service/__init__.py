"""
📦 Conversation Service Package

Service de conversation financière avec détection d'intentions optimisée
utilisant un pipeline L0→L1→L2 pour des performances sub-secondes.

Architecture:
- L0: Pattern Matching (<10ms, 85% hit rate)  
- L1: TinyBERT Classification (15-30ms, 12% usage)
- L2: DeepSeek LLM Fallback (200-500ms, 3% usage)

Version: 1.0.0
Auteur: Harena Finance Platform
"""

from .main import app, get_intent_engine

# Export principal pour le chargement par local_app.py
__all__ = ["app", "get_intent_engine"]

# Métadonnées du package
__version__ = "1.0.0"
__title__ = "Conversation Service"
__description__ = "Service de conversation financière avec IA optimisée"
__author__ = "Harena Finance Platform"

# Configuration pour le chargement dynamique
SERVICE_INFO = {
    "name": "conversation_service",
    "version": __version__,
    "description": __description__,
    "app_module": "conversation_service.main",
    "app_factory": "app",
    "health_endpoint": "/health",
    "api_prefix": "/api/v1",
    "dependencies": [
        "conversation_service.intent_detection.engine",
        "conversation_service.agents.intent_classifier",
        "conversation_service.utils"
    ],
    "required_config": [
        "DEEPSEEK_API_KEY",
        "MIN_CONFIDENCE_THRESHOLD"
    ]
}
