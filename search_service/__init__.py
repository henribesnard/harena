"""
🔍 Search Service - Service de recherche lexicale haute performance

Service de recherche spécialisé basé sur Elasticsearch, optimisé pour
les requêtes financières avec cache intelligent et validation stricte.

Architecture:
    - API REST avec FastAPI
    - Moteur de recherche lexicale Elasticsearch  
    - Cache LRU pour optimisation performance
    - Validation et sécurité des requêtes
    - Métriques et observabilité intégrées

Usage:
    from search_service import SearchService, create_app
    
    # Application FastAPI
    app = create_app()
    
    # Service direct
    service = SearchService()
    results = await service.search(query)
"""

import logging
from typing import Optional, Dict, Any, List

# === CONFIGURATION ET SETTINGS ===
from .config import settings

# === APPLICATIONS FASTAPI ===
from .main import (
    create_app,
    create_development_app, 
    create_production_app,
    create_testing_app,
    app as default_app
)

# === GESTIONNAIRES PRINCIPAUX ===
from .api import APIManager, api_manager
from .core import CoreManager

# === MODÈLES ET CONTRATS ===
from .models import (
    # Contrats d'interface publics
    SearchServiceQuery,
    SearchServiceResponse,
    QueryType,
    FilterOperator,
    AggregationType,
    
    # Modèles de requête
    SearchFilter,
    SearchFilters,
    SearchParameters,
    QueryMetadata,
    
    # Modèles de réponse
    SearchResult,
    ResponseMetadata,
    
    # Modèles internes
    InternalSearchRequest,
    InternalSearchResponse,
    
    # Validateurs
    ContractValidator,
    RequestValidator
)

# === CLIENTS ===
from .clients import ElasticsearchClient, BaseClient

# === TEMPLATES ===
from .templates import QueryTemplateEngine, FinancialAggregationEngine

# === UTILITAIRES PRINCIPAUX ===
from .utils import (
    CacheManager,
    MetricsCollector,
    ElasticsearchQueryBuilder,
    get_system_metrics,
    cleanup_old_metrics,
    get_utils_health
)

# === CONFIGURATION LOGGING ===
logger = logging.getLogger(__name__)


# === SERVICE PRINCIPAL ===

class SearchService:
    """
    Service de recherche principal - Interface haut niveau
    
    Fournit une interface simplifiée pour effectuer des recherches
    en coordonnant tous les composants internes.
    """
    
    def __init__(self):
        """Initialise le service avec tous ses composants"""
        self.core_manager = CoreManager()
        self.api_manager = api_manager
        self.template_engine = QueryTemplateEngine()
        self.aggregation_engine = FinancialAggregationEngine()
        self._initialized = False
    
    async def initialize(self):
        """Initialise tous les composants du service"""
        if self._initialized:
            return
        
        try:
            # Initialiser les gestionnaires
            await self.api_manager.initialize()
            
            # Initialiser les templates
            await self.template_engine.initialize()
            await self.aggregation_engine.initialize()
            
            self._initialized = True
            logger.info("✅ SearchService initialisé avec succès")
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation SearchService: {e}")
            raise
    
    async def search(self, query: SearchServiceQuery) -> SearchServiceResponse:
        """
        Execute une recherche complète
        
        Args:
            query: Requête de recherche validée
            
        Returns:
            SearchServiceResponse: Résultats formatés
        """
        if not self._initialized:
            await self.initialize()
        
        return await self.core_manager.lexical_engine.search(query)
    
    async def validate_query(self, query: SearchServiceQuery) -> Dict[str, Any]:
        """
        Valide une requête de recherche
        
        Args:
            query: Requête à valider
            
        Returns:
            Dict: Résultat de validation détaillé
        """
        return ContractValidator.validate_search_query(query)
    
    async def get_templates(self, intent_type: str = None) -> List[Dict[str, Any]]:
        """
        Récupère les templates disponibles
        
        Args:
            intent_type: Type d'intention spécifique (optionnel)
            
        Returns:
            List: Templates disponibles
        """
        return await self.template_engine.get_available_templates(intent_type)
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Vérifie la santé du service
        
        Returns:
            Dict: État de santé détaillé
        """
        return get_utils_health()
    
    async def shutdown(self):
        """Arrêt propre du service"""
        try:
            await self.api_manager.shutdown()
            cleanup_old_metrics(hours=1)
            self._initialized = False
            logger.info("✅ SearchService arrêté proprement")
        except Exception as e:
            logger.error(f"❌ Erreur arrêt SearchService: {e}")


# === INSTANCE GLOBALE ===
_service_instance: Optional[SearchService] = None


def get_service() -> SearchService:
    """
    Retourne l'instance globale du service (singleton)
    
    Returns:
        SearchService: Instance globale
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = SearchService()
    return _service_instance


async def initialize_service() -> SearchService:
    """
    Initialise et retourne le service global
    
    Returns:
        SearchService: Service initialisé
    """
    service = get_service()
    await service.initialize()
    return service


async def shutdown_service():
    """Arrêt propre du service global"""
    global _service_instance
    if _service_instance:
        await _service_instance.shutdown()
        _service_instance = None


# === FONCTIONS UTILITAIRES RAPIDES ===

async def quick_search(user_id: int, query_text: str, limit: int = 20) -> SearchServiceResponse:
    """
    Recherche rapide avec paramètres minimaux
    
    Args:
        user_id: ID utilisateur
        query_text: Texte de recherche
        limit: Nombre max de résultats
        
    Returns:
        SearchServiceResponse: Résultats de recherche
    """
    service = get_service()
    
    # Construire requête minimale
    query = SearchServiceQuery(
        query_metadata=QueryMetadata(
            user_id=user_id,
            intent_type="TEXT_SEARCH"
        ),
        search_parameters=SearchParameters(
            query_type=QueryType.TEXT_SEARCH,
            limit=limit
        ),
        filters=SearchFilters(
            required=[SearchFilter(field="user_id", operator=FilterOperator.EQ, value=user_id)]
        ),
        text_search={
            "query": query_text,
            "fields": ["searchable_text", "primary_description", "merchant_name"]
        }
    )
    
    return await service.search(query)


def get_version() -> str:
    """Retourne la version du service"""
    return __version__


def get_info() -> Dict[str, Any]:
    """Retourne les informations du service"""
    return {
        "name": "Search Service",
        "version": __version__,
        "description": __description__,
        "author": __author__,
        "environment": settings.environment,
        "elasticsearch_host": settings.elasticsearch_host,
        "elasticsearch_port": settings.elasticsearch_port
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
    
    # === TEMPLATES ===
    "QueryTemplateEngine",
    "FinancialAggregationEngine",
    
    # === UTILITAIRES ===
    "CacheManager",
    "MetricsCollector",
    "ElasticsearchQueryBuilder",
    "get_system_metrics",
    "cleanup_old_metrics",
    "get_utils_health",
    
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
__license__ = "MIT"


# === INITIALISATION MODULE ===

logger.info(f"Module search_service chargé - version {__version__}")
logger.info(f"Configuration: Elasticsearch {settings.elasticsearch_host}:{settings.elasticsearch_port}")
logger.info(f"Environnement: {settings.environment}")

# Nettoyage du namespace
del logging, Optional, Dict, Any