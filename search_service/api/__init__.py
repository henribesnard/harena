"""
🌐 Module API - Interface REST pour le service de recherche

Point d'entrée simplifié pour tous les composants API du Search Service.
Expose uniquement les éléments essentiels sans logique métier.
"""

# === IMPORTS ROUTES ===
from .routes import (
    router,
    admin_router, 
    initialize_routes,
    shutdown_routes
)

# === IMPORTS DEPENDENCIES ===
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

# === IMPORTS MIDDLEWARE (commentés car le module n'existe pas encore) ===
# try:
#     from .middleware import (
#         StructuredLoggingMiddleware,
#         MetricsMiddleware,
#         SecurityMiddleware,
#         ErrorHandlingMiddleware,
#         CompressionMiddleware,
#         create_middleware_stack,
#         initialize_middleware
#     )
# except ImportError:
#     # Fallbacks pour middleware
#     def create_middleware_stack():
#         return []
#     
#     async def initialize_middleware():
#         pass

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
        await initialize_dependencies()
        # await initialize_middleware()  # Commenté temporairement
        await initialize_routes()
    
    async def shutdown(self):
        """Arrête tous les composants API"""
        await shutdown_routes()
        await shutdown_dependencies()

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