"""
Search Service - Service de recherche lexicale pour Harena.

Ce package implémente un service de recherche spécialisé dans les transactions
financières, utilisant Elasticsearch pour des recherches lexicales optimisées.

ARCHITECTURE:
- API REST FastAPI pour les requêtes de recherche
- Moteur de recherche lexicale pure (sans IA)
- Cache LRU pour les performances
- Templates de requêtes optimisés
- Métriques et monitoring

MODULES PRINCIPAUX:
- api/          : Interface REST API
- core/         : Moteur de recherche lexicale
- models/       : Modèles de données et contrats
- clients/      : Clients Elasticsearch
- templates/    : Templates de requêtes
- utils/        : Utilitaires et cache
- config/       : Configuration centralisée
- tests/        : Tests unitaires et d'intégration

USAGE:
    from search_service import SearchService
    from search_service.core import LexicalEngine
    from search_service.models import SearchRequest, SearchResponse
    
    # Service principal
    service = SearchService()
    
    # Recherche lexicale
    results = await service.search_lexical(query="restaurant", user_id=12345)

RESPONSABILITÉS:
✅ Recherche lexicale pure dans les transactions
✅ Filtrage avancé par dates, montants, catégories
✅ Cache des résultats pour les performances
✅ Métriques et monitoring
✅ Validation et sécurité des requêtes
✅ Templates de requêtes optimisés

AUTHORS: Équipe Harena
VERSION: 1.0.0
"""

import logging
from typing import Optional, Dict, Any

# Version du package
__version__ = "1.0.0"
__author__ = "Équipe Harena"
__email__ = "dev@harena.fr"

# Configuration du logging pour le package
logger = logging.getLogger(__name__)

# ==================== IMPORTS PRINCIPAUX AVEC GESTION D'ERREURS ====================

# Configuration avec gestion d'erreur robuste
try:
    from .config.settings import settings, get_settings
    CONFIG_AVAILABLE = True
    logger.info("✅ Configuration chargée")
except ImportError as e:
    logger.warning(f"Configuration non disponible: {e}")
    CONFIG_AVAILABLE = False
    
    # Fallback settings
    class FallbackSettings:
        PROJECT_NAME = "Search Service"
        ELASTICSEARCH_HOST = "localhost"
        ELASTICSEARCH_PORT = 9200
        SEARCH_CACHE_SIZE = 1000
        MAX_SEARCH_RESULTS = 1000
    
    settings = FallbackSettings()
    get_settings = lambda: settings

# Core - Moteur de recherche principal
try:
    from .core.lexical_engine import LexicalEngine, LexicalEngineFactory
    from .core.query_executor import QueryExecutor
    from .core.result_processor import ResultProcessor
    from .core.performance_optimizer import PerformanceOptimizer
    CORE_AVAILABLE = True
    logger.info("✅ Modules core chargés")
except ImportError as e:
    logger.warning(f"Modules core non disponibles: {e}")
    CORE_AVAILABLE = False
    LexicalEngine = None
    LexicalEngineFactory = None
    QueryExecutor = None
    ResultProcessor = None
    PerformanceOptimizer = None

# API avec gestion d'erreur
try:
    from .api.routes import router as api_router
    from .api.dependencies import get_current_user, validate_search_request
    from .api.middleware import setup_middleware
    API_AVAILABLE = True
    logger.info("✅ Modules API chargés")
except ImportError as e:
    logger.warning(f"Modules API non disponibles: {e}")
    API_AVAILABLE = False
    api_router = None
    get_current_user = None
    validate_search_request = None
    setup_middleware = lambda app: None

# Models avec gestion d'erreur
try:
    from .models.requests import LexicalSearchRequest, SearchOptions
    from .models.responses import SearchResponse, ErrorResponse
    from .models.service_contracts import SearchServiceQuery, SearchServiceResponse
    MODELS_AVAILABLE = True
    logger.info("✅ Modules models chargés")
except ImportError as e:
    logger.warning(f"Modules models non disponibles: {e}")
    MODELS_AVAILABLE = False
    LexicalSearchRequest = None
    SearchOptions = None
    SearchResponse = None
    ErrorResponse = None
    SearchServiceQuery = None
    SearchServiceResponse = None

# Clients avec gestion d'erreur
try:
    from .clients.elasticsearch_client import ElasticsearchClient
    CLIENTS_AVAILABLE = True
    logger.info("✅ Clients chargés")
except ImportError as e:
    logger.warning(f"Clients non disponibles: {e}")
    CLIENTS_AVAILABLE = False
    ElasticsearchClient = None

# Utils avec gestion d'erreur et protection contre GlobalSettings
try:
    # Importer utils en protégeant contre les erreurs de GlobalSettings
    import sys
    if 'config_service.config' in sys.modules:
        # Si config_service est déjà importé, on évite les conflits
        from .utils import *
    else:
        from .utils import *
    UTILS_AVAILABLE = True
    logger.info("✅ Utils chargés")
except Exception as e:
    logger.warning(f"Utils non disponibles: {e}")
    UTILS_AVAILABLE = False

# ==================== CLASSE PRINCIPALE ====================

class SearchService:
    """
    Classe principale du service de recherche.
    
    Orchestre tous les composants pour fournir une interface
    unifiée de recherche lexicale.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialise le service de recherche.
        
        Args:
            config: Configuration personnalisée (optionnel)
        """
        # Configuration sécurisée
        if config:
            self.config = config
        elif CONFIG_AVAILABLE:
            self.config = get_settings().__dict__ if hasattr(get_settings(), '__dict__') else {}
        else:
            self.config = {
                'ELASTICSEARCH_HOST': 'localhost',
                'ELASTICSEARCH_PORT': 9200,
                'SEARCH_CACHE_SIZE': 1000,
                'MAX_SEARCH_RESULTS': 1000
            }
        
        self.lexical_engine = None
        self.elasticsearch_client = None
        
        # Initialisation différée des composants
        self._initialized = False
        
        logger.info(f"SearchService v{__version__} initialisé")
    
    async def initialize(self):
        """Initialise les composants du service."""
        if self._initialized:
            return
        
        try:
            # Initialisation du client Elasticsearch
            if CLIENTS_AVAILABLE and ElasticsearchClient:
                self.elasticsearch_client = ElasticsearchClient(self.config)
                await self.elasticsearch_client.connect()
            
            # Initialisation du moteur lexical
            if CORE_AVAILABLE and LexicalEngineFactory:
                self.lexical_engine = LexicalEngineFactory.create(
                    elasticsearch_client=self.elasticsearch_client,
                    config=self.config
                )
            
            self._initialized = True
            logger.info("SearchService complètement initialisé")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation: {e}")
            # Ne pas lever l'exception pour permettre un fonctionnement dégradé
    
    async def search_lexical(
        self, 
        query: str, 
        user_id: int, 
        filters: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Effectue une recherche lexicale.
        
        Args:
            query: Texte de recherche
            user_id: ID de l'utilisateur
            filters: Filtres optionnels
            options: Options de recherche
            
        Returns:
            Résultats de recherche formatés
        """
        if not self._initialized:
            await self.initialize()
        
        if not self.lexical_engine:
            # Mode fallback sans moteur
            return {
                "results": [],
                "total": 0,
                "query": query,
                "user_id": user_id,
                "status": "no_engine_available",
                "message": "Moteur de recherche non disponible - mode dégradé"
            }
        
        try:
            # Construction de la requête
            search_request = {
                "query": query,
                "user_id": user_id,
                "filters": filters or {},
                "options": options or {}
            }
            
            # Exécution de la recherche
            results = await self.lexical_engine.search(search_request)
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche: {e}")
            # Retourner une réponse d'erreur plutôt que lever l'exception
            return {
                "results": [],
                "total": 0,
                "query": query,
                "user_id": user_id,
                "status": "error",
                "error": str(e)
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Vérifie l'état de santé du service.
        
        Returns:
            Statut de santé des composants
        """
        health = {
            "service": "search_service",
            "version": __version__,
            "status": "healthy",
            "components": {
                "config": CONFIG_AVAILABLE,
                "core": CORE_AVAILABLE,
                "api": API_AVAILABLE,
                "models": MODELS_AVAILABLE,
                "clients": CLIENTS_AVAILABLE,
                "utils": UTILS_AVAILABLE
            }
        }
        
        # Vérification des composants critiques
        if self.elasticsearch_client:
            try:
                es_health = await self.elasticsearch_client.health_check()
                health["components"]["elasticsearch"] = es_health.get("status") == "healthy"
            except:
                health["components"]["elasticsearch"] = False
                health["status"] = "degraded"
        else:
            health["components"]["elasticsearch"] = False
            health["status"] = "degraded"
        
        return health
    
    async def close(self):
        """Ferme les connexions et nettoie les ressources."""
        try:
            if self.elasticsearch_client:
                await self.elasticsearch_client.close()
        except Exception as e:
            logger.error(f"Erreur lors de la fermeture: {e}")
        
        logger.info("SearchService fermé proprement")

# ==================== FACTORY FUNCTION ====================

def create_search_service(config: Optional[Dict[str, Any]] = None) -> SearchService:
    """
    Factory function pour créer une instance du service de recherche.
    
    Args:
        config: Configuration personnalisée
        
    Returns:
        Instance configurée du SearchService
    """
    return SearchService(config)

# ==================== EXPORTS ====================

__all__ = [
    # Version
    "__version__",
    "__author__",
    
    # Classe principale
    "SearchService",
    "create_search_service",
    
    # Composants core (si disponibles)
    "LexicalEngine",
    "LexicalEngineFactory",
    "QueryExecutor",
    "ResultProcessor",
    "PerformanceOptimizer",
    
    # API (si disponible)
    "api_router",
    "setup_middleware",
    
    # Models (si disponibles)
    "LexicalSearchRequest",
    "SearchResponse",
    "SearchServiceQuery",
    "SearchServiceResponse",
    
    # Clients (si disponibles)
    "ElasticsearchClient",
    
    # Configuration
    "settings",
    "get_settings",
    
    # Flags de disponibilité
    "CONFIG_AVAILABLE",
    "CORE_AVAILABLE",
    "API_AVAILABLE",
    "MODELS_AVAILABLE",
    "CLIENTS_AVAILABLE",
    "UTILS_AVAILABLE"
]

# ==================== INITIALISATION ====================

# Log de l'état du package au chargement
logger.info(f"Search Service v{__version__} package chargé")
logger.info(f"Modules disponibles: "
           f"config={CONFIG_AVAILABLE}, "
           f"core={CORE_AVAILABLE}, "
           f"api={API_AVAILABLE}, "
           f"models={MODELS_AVAILABLE}, "
           f"clients={CLIENTS_AVAILABLE}, "
           f"utils={UTILS_AVAILABLE}")

# Avertissement si des modules critiques manquent
if not CORE_AVAILABLE:
    logger.warning("Modules core manquants - fonctionnalités de recherche limitées")

if not API_AVAILABLE:
    logger.warning("Modules API manquants - interface REST non disponible")

if not CONFIG_AVAILABLE:
    logger.warning("Configuration utilise le mode fallback")