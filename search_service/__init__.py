"""
🔍 Search Service - Service de recherche lexicale haute performance

Ce module fournit un service de recherche spécialisé basé sur Elasticsearch,
optimisé pour les requêtes financières et la performance.

Architecture:
    - API REST avec FastAPI
    - Moteur de recherche lexicale Elasticsearch
    - Cache intelligent pour les performances
    - Validation stricte des requêtes
    - Métriques et observabilité

Usage:
    from search_service import SearchService, create_app
    
    # Création de l'application
    app = create_app()
    
    # Ou utilisation directe du service
    service = SearchService()
    results = await service.search(query)
"""

import logging
from typing import Optional, Dict, Any

# === IMPORTS PRINCIPAUX ===

# Configuration
from .config import settings

# API et application
from .main import (
    create_app, create_development_app, create_production_app, create_testing_app,
    app as default_app
)

# Gestionnaires principaux
from .api import APIManager, api_manager
from .core import CoreManager, core_manager

# Modèles et contrats
from .models import (
    # Contrats d'interface
    SearchServiceQuery, SearchServiceResponse,
    QueryType, FilterOperator, AggregationType,
    SearchFilter, SearchFilters, SearchParameters,
    QueryMetadata, SearchResult, ResponseMetadata,
    
    # Modèles internes
    InternalSearchRequest, InternalSearchResponse,
    
    # Validateurs
    ContractValidator, RequestValidator
)

# Clients
from .clients import ElasticsearchClient, BaseClient

# Utilitaires principaux
from .utils import (
    CacheManager, MetricsCollector, RequestValidator as UtilValidator,
    ElasticsearchHelper
)


# === CONFIGURATION LOGGING ===
logger = logging.getLogger(__name__)


# === CLASSE SERVICE PRINCIPALE ===

class SearchService:
    """
    Service de recherche principal - Interface haut niveau
    
    Fournit une interface simplifiée pour effectuer des recherches
    en utilisant tous les composants internes de manière coordonnée.
    """
    
    def __init__(
        self,
        elasticsearch_client: Optional[ElasticsearchClient] = None,
        cache_enabled: bool = True,
        metrics_enabled: bool = True
    ):
        """
        Initialise le service de recherche
        
        Args:
            elasticsearch_client: Client Elasticsearch personnalisé
            cache_enabled: Active/désactive le cache
            metrics_enabled: Active/désactive les métriques
        """
        self._elasticsearch_client = elasticsearch_client
        self._cache_enabled = cache_enabled
        self._metrics_enabled = metrics_enabled
        self._initialized = False
        
        logger.info("SearchService créé")
    
    async def initialize(self):
        """Initialise tous les composants du service"""
        if self._initialized:
            logger.warning("SearchService déjà initialisé")
            return
        
        try:
            # Initialisation des composants core
            await core_manager.initialize()
            
            # Initialisation de l'API
            await api_manager.initialize()
            
            self._initialized = True
            logger.info("✅ SearchService initialisé avec succès")
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation SearchService: {e}")
            raise
    
    async def shutdown(self):
        """Ferme proprement tous les composants"""
        if not self._initialized:
            return
        
        try:
            await api_manager.shutdown()
            await core_manager.shutdown()
            
            self._initialized = False
            logger.info("✅ SearchService fermé proprement")
            
        except Exception as e:
            logger.error(f"❌ Erreur fermeture SearchService: {e}")
            raise
    
    async def search(self, query: SearchServiceQuery) -> SearchServiceResponse:
        """
        Effectue une recherche avec le contrat standardisé
        
        Args:
            query: Requête au format SearchServiceQuery
            
        Returns:
            SearchServiceResponse: Résultats de la recherche
            
        Raises:
            ValueError: Si la requête n'est pas valide
            RuntimeError: Si le service n'est pas initialisé
        """
        if not self._initialized:
            raise RuntimeError("SearchService non initialisé - appelez initialize() d'abord")
        
        # Validation du contrat
        ContractValidator.validate_search_query(query)
        
        # Délégation au moteur de recherche core
        lexical_engine = core_manager.lexical_engine
        if not lexical_engine:
            raise RuntimeError("Moteur lexical non disponible")
        
        return await lexical_engine.search(query)
    
    async def validate_query(self, query: SearchServiceQuery) -> Dict[str, Any]:
        """
        Valide une requête sans l'exécuter
        
        Args:
            query: Requête à valider
            
        Returns:
            Dict[str, Any]: Résultat de la validation
        """
        try:
            ContractValidator.validate_search_query(query)
            return {
                "valid": True,
                "message": "Requête valide"
            }
        except Exception as e:
            return {
                "valid": False,
                "error": str(e)
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Vérification de santé du service
        
        Returns:
            Dict[str, Any]: Statut de santé
        """
        if not self._initialized:
            return {
                "healthy": False,
                "error": "Service non initialisé"
            }
        
        # Délégation aux gestionnaires
        core_health = await core_manager.health_check()
        api_health = await api_manager.health_check() if api_manager._initialized else {"healthy": True}
        
        overall_healthy = core_health.get("healthy", False) and api_health.get("healthy", False)
        
        return {
            "healthy": overall_healthy,
            "components": {
                "core": core_health,
                "api": api_health
            },
            "initialized": self._initialized
        }
    
    @property
    def is_initialized(self) -> bool:
        """Retourne True si le service est initialisé"""
        return self._initialized


# === INSTANCE GLOBALE ===

# Instance globale du service pour usage simple
_default_service: Optional[SearchService] = None


def get_service() -> SearchService:
    """
    Retourne l'instance globale du service
    
    Returns:
        SearchService: Instance globale
    """
    global _default_service
    if _default_service is None:
        _default_service = SearchService()
    return _default_service


async def initialize_service(**kwargs) -> SearchService:
    """
    Initialise et retourne le service global
    
    Args:
        **kwargs: Arguments pour SearchService
        
    Returns:
        SearchService: Service initialisé
    """
    global _default_service
    if _default_service is None:
        _default_service = SearchService(**kwargs)
    
    if not _default_service.is_initialized:
        await _default_service.initialize()
    
    return _default_service


async def shutdown_service():
    """Ferme le service global"""
    global _default_service
    if _default_service and _default_service.is_initialized:
        await _default_service.shutdown()


# === FONCTIONS UTILITAIRES ===

async def quick_search(
    text: str,
    user_id: int,
    filters: Optional[Dict] = None,
    limit: int = 20
) -> SearchServiceResponse:
    """
    Fonction utilitaire pour une recherche rapide
    
    Args:
        text: Texte à rechercher
        user_id: ID utilisateur (obligatoire pour sécurité)
        filters: Filtres optionnels
        limit: Nombre de résultats
        
    Returns:
        SearchServiceResponse: Résultats de la recherche
    """
    service = get_service()
    if not service.is_initialized:
        await service.initialize()
    
    # Construction requête simple
    query = SearchServiceQuery(
        query_metadata=QueryMetadata(
            user_id=user_id,
            intent_type="search",
            agent_name="quick_search"
        ),
        text_search=text,
        search_parameters=SearchParameters(limit=limit),
        filters=SearchFilters(
            required=[
                SearchFilter(field="user_id", value=user_id, operator=FilterOperator.EQUALS)
            ]
        )
    )
    
    # Ajout filtres supplémentaires
    if filters:
        for field, value in filters.items():
            query.filters.required.append(
                SearchFilter(field=field, value=value, operator=FilterOperator.EQUALS)
            )
    
    return await service.search(query)


def get_version() -> str:
    """Retourne la version du module"""
    return __version__


def get_info() -> Dict[str, Any]:
    """Retourne les informations du module"""
    return {
        "name": __name__,
        "version": __version__,
        "description": __doc__.strip() if __doc__ else "",
        "author": __author__,
        "settings": {
            "elasticsearch_host": settings.elasticsearch_host,
            "cache_enabled": settings.cache_enabled,
            "metrics_enabled": settings.metrics_enabled,
            "environment": settings.environment
        }
    }


# === EXPORTS PRINCIPAUX ===

__all__ = [
    # === SERVICE PRINCIPAL ===
    "SearchService",
    "get_service",
    "initialize_service",
    "shutdown_service",
    
    # === APPLICATIONS FASTAPI ===
    "create_app",
    "create_development_app", 
    "create_production_app",
    "create_testing_app",
    "default_app",
    
    # === GESTIONNAIRES ===
    "APIManager",
    "api_manager",
    "CoreManager", 
    "core_manager",
    
    # === MODÈLES ET CONTRATS ===
    "SearchServiceQuery",
    "SearchServiceResponse",
    "QueryType",
    "FilterOperator", 
    "AggregationType",
    "SearchFilter",
    "SearchFilters",
    "SearchParameters",
    "QueryMetadata",
    "SearchResult",
    "ResponseMetadata",
    
    # === MODÈLES INTERNES ===
    "InternalSearchRequest",
    "InternalSearchResponse",
    
    # === VALIDATEURS ===
    "ContractValidator",
    "RequestValidator",
    
    # === CLIENTS ===
    "ElasticsearchClient",
    "BaseClient",
    
    # === UTILITAIRES ===
    "CacheManager",
    "MetricsCollector", 
    "UtilValidator",
    "ElasticsearchHelper",
    
    # === FONCTIONS UTILITAIRES ===
    "quick_search",
    "get_version",
    "get_info",
    
    # === CONFIGURATION ===
    "settings"
]


# === MÉTADONNÉES MODULE ===

__version__ = "1.0.0"
__author__ = "Search Service Team"
__email__ = "search-team@company.com"
__description__ = "Service de recherche lexicale haute performance avec Elasticsearch"
__license__ = "Proprietary"

# === INITIALISATION MODULE ===

logger.info(f"Module search_service chargé - version {__version__}")
logger.info(f"Configuration: Elasticsearch {settings.elasticsearch_host}:{settings.elasticsearch_port}")
logger.info(f"Environnement: {settings.environment}")


# === NETTOYAGE NAMESPACE ===

# Suppression des imports internes pour garder un namespace propre
del logging, Optional, Dict, Any