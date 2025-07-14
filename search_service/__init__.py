"""
Search Service - Service de recherche lexicale haute performance
===============================================================

Version simplifiée - imports seulement des modules qui existent.
"""

import logging

__version__ = "1.0.0"
__description__ = "Service de recherche lexicale haute performance"
__author__ = "Search Service Team"

logger = logging.getLogger(__name__)

# === IMPORTS SÉCURISÉS ===

# Application principale (existe)
try:
    from .main import create_app, app as default_app
    logger.info("✅ Module main importé")
except ImportError as e:
    logger.error(f"❌ Erreur import main: {e}")
    create_app = None
    default_app = None

# API (existe maintenant)
try:
    from .api import api_manager, router
    logger.info("✅ Module api importé")
except ImportError as e:
    logger.error(f"❌ Erreur import api: {e}")
    api_manager = None
    router = None

# Configuration (si existe)
try:
    from .config import settings
    logger.info("✅ Configuration importée")
except ImportError as e:
    logger.warning(f"⚠️ Configuration non disponible: {e}")
    # Fallback settings minimal
    class Settings:
        environment = "development"
        elasticsearch_host = "localhost"
        elasticsearch_port = 9200
    settings = Settings()

# === EXPORTS MINIMALISTES ===
__all__ = [
    "create_app",
    "default_app", 
    "api_manager",
    "router",
    "settings",
    "__version__",
    "__description__",
    "__author__"
]

logger.info(f"Search Service v{__version__} - Module principal chargé")
logger.info(f"Environnement: {settings.environment}")

# Éviter la pollution du namespace
del logging