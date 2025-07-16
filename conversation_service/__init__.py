"""
🤖 Conversation Service - Service de classification d'intentions financières

Ce service fournit une API REST pour classifier les intentions des utilisateurs
dans un contexte financier en utilisant DeepSeek comme modèle de langage.

Architecture MVP:
- Agent de classification d'intentions (intent_classifier)
- Client DeepSeek optimisé avec cache et retry
- API REST FastAPI avec métriques intégrées
- Configuration centralisée et validation

Fonctionnalités principales:
- Classification de 8 types d'intentions financières
- Extraction automatique d'entités (marchands, catégories, montants, dates)
- Scoring de confiance avec seuils configurables
- Cache intelligent pour optimiser les coûts DeepSeek
- Métriques complètes et monitoring en temps réel
- Tests automatisés et collection Postman

Usage:
    from conversation_service import app, classify_message
    
    # Démarrage du service
    uvicorn.run(app, host="0.0.0.0", port=8001)
    
    # Classification directe
    result = await classify_message("mes restaurants ce mois")
"""

# Métadonnées du package
__version__ = "1.0.0"
__title__ = "Conversation Service"
__description__ = "Service de classification d'intentions financières avec DeepSeek"
__author__ = "Harena Team"
__license__ = "MIT"

# Imports principaux
from .main import app
from .config import settings
from .agents import intent_classifier, classify_message
from .clients import deepseek_client
from .models import (
    FinancialIntent,
    ChatRequest,
    ChatResponse,
    IntentResult,
    EntityHints,
    get_supported_intents
)

# Export de l'API publique
__all__ = [
    # Application principale
    "app",
    
    # Configuration
    "settings",
    
    # Agents et classification
    "intent_classifier",
    "classify_message",
    
    # Clients
    "deepseek_client",
    
    # Modèles de données principaux
    "FinancialIntent",
    "ChatRequest", 
    "ChatResponse",
    "IntentResult",
    "EntityHints",
    
    # Helpers
    "get_supported_intents",
    "get_service_info",
    "health_check",
    "get_metrics"
]

# Helpers publics
async def health_check():
    """
    Vérification rapide de santé du service
    
    Returns:
        dict: Statut de santé du service et ses dépendances
    """
    from .clients import health_check_all_clients
    from .agents import get_agents_status
    
    try:
        clients_health = await health_check_all_clients()
        agents_status = get_agents_status()
        
        # Déterminer le statut global
        all_healthy = all(
            client.get("status") == "healthy" 
            for client in clients_health.values()
        )
        
        global_status = "healthy" if all_healthy else "degraded"
        
        return {
            "status": global_status,
            "version": __version__,
            "clients": clients_health,
            "agents": agents_status
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "version": __version__,
            "error": str(e)
        }

def get_metrics():
    """
    Récupère les métriques globales du service
    
    Returns:
        dict: Métriques consolidées de tous les composants
    """
    from .clients import get_all_clients_metrics
    from .agents import get_all_agents_metrics
    
    try:
        clients_metrics = get_all_clients_metrics()
        agents_metrics = get_all_agents_metrics()
        
        # Métriques consolidées
        total_classifications = agents_metrics.get("intent_classifier", {}).get("total_classifications", 0)
        total_requests = clients_metrics.get("deepseek", {}).get("total_requests", 0)
        
        return {
            "service": {
                "version": __version__,
                "total_classifications": total_classifications,
                "total_deepseek_requests": total_requests
            },
            "agents": agents_metrics,
            "clients": clients_metrics
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "service": {"version": __version__}
        }

def get_service_info():
    """
    Informations complètes sur le service
    
    Returns:
        dict: Informations détaillées du service
    """
    from .api import get_api_info
    
    return {
        "name": __title__,
        "version": __version__,
        "description": __description__,
        "author": __author__,
        "license": __license__,
        
        # Configuration
        "config": {
            "confidence_threshold": settings.MIN_CONFIDENCE_THRESHOLD,
            "deepseek_model": settings.DEEPSEEK_CHAT_MODEL,
            "debug_mode": settings.DEBUG,
            "port": settings.PORT
        },
        
        # Capacités
        "capabilities": {
            "supported_intents": get_supported_intents(),
            "intent_count": len(get_supported_intents()),
            "cache_enabled": True,
            "metrics_enabled": settings.ENABLE_METRICS
        },
        
        # API
        "api": get_api_info()
    }

# Version check et avertissements
def _check_environment():
    """Vérifications d'environnement au démarrage"""
    import os
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Vérification des variables critiques
    required_vars = ["DEEPSEEK_API_KEY"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        logger.warning(f"Variables d'environnement manquantes: {missing_vars}")
    
    # Validation de la configuration
    try:
        validation = settings.validate_configuration()
        if not validation["valid"]:
            logger.error(f"Configuration invalide: {validation['errors']}")
        if validation["warnings"]:
            logger.warning(f"Avertissements: {validation['warnings']}")
    except Exception as e:
        logger.error(f"Erreur validation configuration: {e}")

# Exécution des vérifications au moment de l'import
try:
    _check_environment()
except Exception:
    # Ne pas faire échouer l'import si les vérifications échouent
    pass

# Message de bienvenue pour les développeurs
def _welcome_message():
    """Message informatif pour les développeurs"""
    return f"""
🤖 {__title__} v{__version__} chargé avec succès!

📚 Documentation:
   - API Docs: http://localhost:{settings.PORT}/docs
   - Health: http://localhost:{settings.PORT}/health
   - Metrics: http://localhost:{settings.PORT}/metrics

🎯 Intentions supportées: {len(get_supported_intents())}
   {', '.join(get_supported_intents())}

🚀 Démarrage rapide:
   cd conversation_service && python run.py

💡 Aide:
   from conversation_service import get_service_info
   print(get_service_info())
"""

# Export du message pour usage optionnel
__doc__ += _welcome_message()