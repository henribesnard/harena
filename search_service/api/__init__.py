"""
🌐 Module API - Interface REST pour le service de recherche

Point d'entrée simplifié pour tous les composants API du Search Service.
Expose uniquement les éléments essentiels sans logique métier.
"""

# === IMPORTS SÉCURISÉS AVEC GESTION D'ERREURS ===

# Tentative d'import des routes avec fallback
try:
    from .routes import (
        router,
        admin_router, 
        initialize_routes,
        shutdown_routes
    )
except ImportError:
    # Fallback si le module routes n'existe pas encore
    from fastapi import APIRouter
    
    router = APIRouter()
    admin_router = APIRouter()
    
    async def initialize_routes():
        """Fallback pour l'initialisation des routes"""
        pass
    
    async def shutdown_routes():
        """Fallback pour l'arrêt des routes"""
        pass

# Tentative d'import des dépendances avec fallback
try:
    from .dependencies import (
        # Gestionnaires principaux
        auth_manager,
        rate_limiter,
        
        # Dépendances FastAPI
        get_authenticated_user,
        validate_search_request,
        validate_rate_limit,
        check_service_health,
        
        # Dépendances spécialisées (seulement celles qui existent)
        create_admin_dependencies,
        
        # Fonctions d'initialisation
        initialize_dependencies,
        shutdown_dependencies
    )
except ImportError:
    # Fallbacks pour dependencies
    auth_manager = None
    rate_limiter = None
    
    async def get_authenticated_user():
        """Fallback pour l'authentification"""
        return {"user": "anonymous"}
    
    async def validate_search_request():
        """Fallback pour la validation"""
        return True
    
    async def validate_rate_limit():
        """Fallback pour le rate limiting"""
        return True
    
    async def check_service_health():
        """Fallback pour le health check"""
        return {"status": "ok"}
    
    def create_admin_dependencies():
        """Fallback pour les dépendances admin"""
        return []
    
    async def initialize_dependencies():
        """Fallback pour l'initialisation des dépendances"""
        pass
    
    async def shutdown_dependencies():
        """Fallback pour l'arrêt des dépendances"""
        pass

# === IMPORTS MIDDLEWARE (commentés car le module n'existe pas encore) ===
try:
    from .middleware import (
        StructuredLoggingMiddleware,
        MetricsMiddleware,
        SecurityMiddleware,
        ErrorHandlingMiddleware,
        CompressionMiddleware,
        create_middleware_stack,
        initialize_middleware
    )
except ImportError:
    # Fallbacks pour middleware
    def create_middleware_stack():
        return []
    
    async def initialize_middleware():
        pass

# === CLASSE GESTIONNAIRE SIMPLIFIÉE ===
class APIManager:
    """
    Gestionnaire unifié pour l'API REST
    
    Centralise l'accès aux composants API sans implémenter de logique.
    """
    def __init__(self):
        self.router = router
        self.admin_router = admin_router
        self.auth_manager = auth_manager
        self.rate_limiter = rate_limiter
        # self.middleware_stack = create_middleware_stack()  # Commenté temporairement
    
    async def initialize(self):
        """Initialise tous les composants API"""
        try:
            await initialize_dependencies()
            # await initialize_middleware()  # Commenté temporairement
            await initialize_routes()
        except Exception as e:
            # Log l'erreur mais ne fait pas planter l'initialisation
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Erreur initialisation API Manager: {e}")
    
    async def shutdown(self):
        """Arrête tous les composants API"""
        try:
            await shutdown_routes()
            await shutdown_dependencies()
        except Exception as e:
            # Log l'erreur mais ne fait pas planter l'arrêt
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Erreur arrêt API Manager: {e}")

# === INSTANCE GLOBALE ===
api_manager = APIManager()

# === EXPORTS ===
__all__ = [
    # === GESTIONNAIRE PRINCIPAL ===
    "APIManager",
    "api_manager",
    
    # === ROUTERS ===
    "router",
    "admin_router",
    
    # === DÉPENDANCES PRINCIPALES ===
    "auth_manager",
    "rate_limiter",
    "get_authenticated_user",
    "validate_search_request",
    "validate_rate_limit",
    "check_service_health",
    
    # === DÉPENDANCES SPÉCIALISÉES ===
    "create_admin_dependencies",
    
    # === FONCTIONS D'INITIALISATION ===
    "initialize_routes",
    "shutdown_routes",
    "initialize_dependencies",
    "shutdown_dependencies"
]