"""
Module API du Search Service - Version simplifiée
================================================

Expose seulement les composants essentiels qui existent vraiment.
"""

import logging

logger = logging.getLogger(__name__)

# === IMPORT SÉCURISÉ DU ROUTER ===
try:
    from .routes import router
    logger.info("✅ Router importé depuis routes.py")
except ImportError as e:
    logger.error(f"❌ Erreur import router: {e}")
    # Fallback router vide
    from fastapi import APIRouter
    router = APIRouter()
    logger.warning("⚠️ Utilisation d'un router fallback vide")

# === API MANAGER SIMPLIFIÉ ===
class APIManager:
    """Gestionnaire API minimal pour heroku_app.py"""
    
    def __init__(self):
        self.router = router
        self._initialized = False
        logger.info(f"APIManager créé avec {len(router.routes) if hasattr(router, 'routes') else 0} routes")
    
    async def initialize(self):
        """Initialise l'API manager"""
        self._initialized = True
        logger.info("✅ API Manager initialisé")
    
    async def shutdown(self):
        """Arrêt propre de l'API manager"""
        logger.info("✅ API Manager arrêté")

# === INSTANCE GLOBALE ===
api_manager = APIManager()

# === EXPORTS MINIMALISTES ===
__all__ = ["router", "api_manager", "APIManager"]

logger.info(f"Module search_service.api chargé - {len(router.routes) if hasattr(router, 'routes') else 0} routes disponibles")