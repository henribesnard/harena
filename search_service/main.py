"""
Point d'entrée principal du search_service - VERSION RÉÉCRITE.

Cette version corrige les problèmes d'initialisation et de gestion des dépendances
qui causaient l'erreur 'str' object has no attribute 'generate_embedding'.

Améliorations:
- Initialisation séquentielle stricte avec validation à chaque étape
- Injection de dépendances FastAPI native au lieu du système manuel
- Gestion d'erreurs robuste avec rollback
- Mode dégradé gracieux pour les composants optionnels
- Validation complète des services avant utilisation
"""
import asyncio
import logging
import logging.config
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Configuration globale importée en premier
try:
    from config_service.config import settings as global_settings
except ImportError:
    print("❌ Impossible d'importer la configuration globale")
    sys.exit(1)

# Configuration locale du search_service
from search_service.config import (
    get_search_settings, get_logging_config, get_elasticsearch_config,
    get_qdrant_config, get_embedding_config, get_hybrid_search_config
)

# Clients
from search_service.clients.elasticsearch_client import ElasticsearchClient
from search_service.clients.qdrant_client import QdrantClient

# Core services
from search_service.core.embeddings import EmbeddingService, EmbeddingManager, EmbeddingConfig
from search_service.core.query_processor import QueryProcessor
from search_service.core.lexical_engine import LexicalSearchEngine, LexicalSearchConfig
from search_service.core.semantic_engine import SemanticSearchEngine, SemanticSearchConfig
from search_service.core.result_merger import ResultMerger, FusionConfig
from search_service.core.search_engine import HybridSearchEngine, HybridSearchConfig

# Setup logging
logging.config.dictConfig(get_logging_config())
logger = logging.getLogger(__name__)

# ========== MODÈLE DE DÉPENDANCES ==========

@dataclass
class ServiceDependencies:
    """Container pour toutes les dépendances du service."""
    
    # Clients de base de données
    elasticsearch_client: Optional[ElasticsearchClient] = None
    qdrant_client: Optional[QdrantClient] = None
    
    # Services d'embeddings
    embedding_service: Optional[EmbeddingService] = None
    embedding_manager: Optional[EmbeddingManager] = None
    
    # Moteurs de recherche
    query_processor: Optional[QueryProcessor] = None
    lexical_engine: Optional[LexicalSearchEngine] = None
    semantic_engine: Optional[SemanticSearchEngine] = None
    result_merger: Optional[ResultMerger] = None
    hybrid_engine: Optional[HybridSearchEngine] = None
    
    # État d'initialisation
    initialization_results: Dict[str, Any] = field(default_factory=dict)
    startup_time: Optional[float] = None
    
    def is_fully_initialized(self) -> bool:
        """Vérifie si tous les composants essentiels sont initialisés."""
        essential_components = [
            self.elasticsearch_client,
            self.qdrant_client,
            self.query_processor,
            self.lexical_engine,
            self.result_merger,
            self.hybrid_engine
        ]
        return all(comp is not None for comp in essential_components)
    
    def is_semantic_available(self) -> bool:
        """Vérifie si la recherche sémantique est disponible."""
        return all([
            self.embedding_service is not None,
            self.embedding_manager is not None,
            self.semantic_engine is not None,
            hasattr(self.embedding_manager, 'generate_embedding')
        ])

# Instance globale des dépendances
app_dependencies = ServiceDependencies()

# ========== INITIALISATION SÉQUENTIELLE ==========

class InitializationError(Exception):
    """Exception pour les erreurs d'initialisation critiques."""
    pass

async def initialize_step_by_step() -> ServiceDependencies:
    """
    Initialisation séquentielle avec validation à chaque étape.
    Chaque étape peut échouer de manière contrôlée.
    """
    deps = ServiceDependencies()
    deps.startup_time = time.time()
    
    logger.info("🚀 Démarrage de l'initialisation séquentielle du Search Service")
    
    # ÉTAPE 1: Configuration
    try:
        logger.info("📋 ÉTAPE 1: Validation de la configuration...")
        await validate_configuration()
        deps.initialization_results["configuration"] = {"status": "success"}
        logger.info("✅ Configuration validée")
    except Exception as e:
        logger.error(f"❌ ÉTAPE 1 ÉCHOUÉE: {e}")
        deps.initialization_results["configuration"] = {"status": "failed", "error": str(e)}
        raise InitializationError(f"Configuration validation failed: {e}")
    
    # ÉTAPE 2: Clients de base de données
    try:
        logger.info("🗄️ ÉTAPE 2: Initialisation des clients de base de données...")
        deps.elasticsearch_client = await initialize_elasticsearch_client()
        deps.qdrant_client = await initialize_qdrant_client()
        deps.initialization_results["clients"] = {"status": "success"}
        logger.info("✅ Clients de base de données initialisés")
    except Exception as e:
        logger.error(f"❌ ÉTAPE 2 ÉCHOUÉE: {e}")
        deps.initialization_results["clients"] = {"status": "failed", "error": str(e)}
        raise InitializationError(f"Database clients initialization failed: {e}")
    
    # ÉTAPE 3: Services d'embeddings (optionnel, peut échouer gracieusement)
    try:
        logger.info("🤖 ÉTAPE 3: Initialisation des services d'embeddings...")
        deps.embedding_service, deps.embedding_manager = await initialize_embedding_services_robust()
        
        if deps.embedding_service and deps.embedding_manager:
            # Test de validation
            await validate_embedding_services(deps.embedding_service, deps.embedding_manager)
            deps.initialization_results["embeddings"] = {"status": "success"}
            logger.info("✅ Services d'embeddings initialisés et validés")
        else:
            deps.initialization_results["embeddings"] = {"status": "disabled", "reason": "API key not configured"}
            logger.warning("⚠️ Services d'embeddings désactivés (mode dégradé)")
            
    except Exception as e:
        logger.warning(f"⚠️ ÉTAPE 3 EN MODE DÉGRADÉ: {e}")
        deps.embedding_service = None
        deps.embedding_manager = None
        deps.initialization_results["embeddings"] = {"status": "degraded", "error": str(e)}
        logger.info("🔄 Continuant en mode dégradé sans recherche sémantique")
    
    # ÉTAPE 4: Moteurs de recherche
    try:
        logger.info("🎯 ÉTAPE 4: Initialisation des moteurs de recherche...")
        deps.query_processor = await initialize_query_processor()
        deps.lexical_engine = await initialize_lexical_engine(deps.elasticsearch_client)
        
        # Moteur sémantique uniquement si les embeddings sont disponibles
        if deps.embedding_manager:
            deps.semantic_engine = await initialize_semantic_engine(deps.qdrant_client, deps.embedding_manager)
            logger.info("✅ Moteur sémantique initialisé")
        else:
            logger.info("⚠️ Moteur sémantique désactivé (embeddings non disponibles)")
        
        deps.initialization_results["engines"] = {"status": "success"}
        logger.info("✅ Moteurs de recherche initialisés")
        
    except Exception as e:
        logger.error(f"❌ ÉTAPE 4 ÉCHOUÉE: {e}")
        deps.initialization_results["engines"] = {"status": "failed", "error": str(e)}
        raise InitializationError(f"Search engines initialization failed: {e}")
    
    # ÉTAPE 5: Fusion et moteur hybride
    try:
        logger.info("🔀 ÉTAPE 5: Initialisation du système de fusion...")
        deps.result_merger = await initialize_result_merger()
        deps.hybrid_engine = await initialize_hybrid_engine(
            deps.lexical_engine, 
            deps.semantic_engine, 
            deps.result_merger
        )
        deps.initialization_results["hybrid"] = {"status": "success"}
        logger.info("✅ Système hybride initialisé")
        
    except Exception as e:
        logger.error(f"❌ ÉTAPE 5 ÉCHOUÉE: {e}")
        deps.initialization_results["hybrid"] = {"status": "failed", "error": str(e)}
        raise InitializationError(f"Hybrid system initialization failed: {e}")
    
    # ÉTAPE 6: Validation finale
    try:
        logger.info("🔍 ÉTAPE 6: Validation finale du système...")
        await perform_final_validation(deps)
        logger.info("✅ Validation finale réussie")
        
    except Exception as e:
        logger.error(f"❌ ÉTAPE 6 ÉCHOUÉE: {e}")
        raise InitializationError(f"Final validation failed: {e}")
    
    total_time = time.time() - deps.startup_time
    logger.info(f"🎉 Initialisation complète terminée en {total_time:.2f}s")
    
    # Log du résumé
    semantic_status = "✅ Activée" if deps.is_semantic_available() else "⚠️ Désactivée"
    logger.info(f"📊 Résumé: Recherche lexicale ✅ | Recherche sémantique {semantic_status}")
    
    return deps

# ========== FONCTIONS D'INITIALISATION INDIVIDUELLES ==========

async def validate_configuration():
    """Valide la configuration requise."""
    # Vérifier les configurations essentielles
    elasticsearch_config = get_elasticsearch_config()
    qdrant_config = get_qdrant_config()
    
    if not elasticsearch_config.get("hosts"):
        raise ValueError("Configuration Elasticsearch manquante")
    
    if not qdrant_config.get("host"):
        raise ValueError("Configuration Qdrant manquante")

async def initialize_elasticsearch_client() -> ElasticsearchClient:
    """Initialise le client Elasticsearch."""
    config = get_elasticsearch_config()
    client = ElasticsearchClient(**config)
    await client.connect()
    return client

async def initialize_qdrant_client() -> QdrantClient:
    """Initialise le client Qdrant."""
    config = get_qdrant_config()
    client = QdrantClient(**config)
    await client.connect()
    return client

async def initialize_embedding_services_robust() -> tuple[Optional[EmbeddingService], Optional[EmbeddingManager]]:
    """
    Initialise les services d'embeddings de manière robuste.
    Retourne (None, None) si impossible, sans lever d'exception.
    """
    # Vérifier la clé API
    if not global_settings.OPENAI_API_KEY:
        logger.info("🔑 OPENAI_API_KEY non configurée, désactivation des embeddings")
        return None, None
    
    try:
        # Configuration
        embedding_config = EmbeddingConfig(**get_embedding_config())
        
        # Créer le service
        embedding_service = EmbeddingService(
            api_key=global_settings.OPENAI_API_KEY,
            config=embedding_config
        )
        
        # Validation du type
        if not isinstance(embedding_service, EmbeddingService):
            raise TypeError(f"Expected EmbeddingService, got {type(embedding_service)}")
        
        # Test de fonctionnement
        test_embedding = await embedding_service.generate_embedding("test", use_cache=False)
        if not test_embedding or len(test_embedding) == 0:
            raise ValueError("Test embedding failed")
        
        # Créer le manager
        embedding_manager = EmbeddingManager(embedding_service)
        
        # Validation du manager
        if not isinstance(embedding_manager, EmbeddingManager):
            raise TypeError(f"Expected EmbeddingManager, got {type(embedding_manager)}")
        
        if not hasattr(embedding_manager, 'primary_service'):
            raise AttributeError("EmbeddingManager missing primary_service")
        
        if not hasattr(embedding_manager.primary_service, 'generate_embedding'):
            raise AttributeError("Primary service missing generate_embedding method")
        
        logger.info(f"✅ Services d'embeddings créés et testés ({len(test_embedding)} dims)")
        return embedding_service, embedding_manager
        
    except Exception as e:
        logger.warning(f"⚠️ Impossible d'initialiser les embeddings: {e}")
        return None, None

async def validate_embedding_services(embedding_service: EmbeddingService, embedding_manager: EmbeddingManager):
    """Valide que les services d'embeddings fonctionnent correctement."""
    
    # Test du service principal
    test_result = await embedding_service.generate_embedding("validation test")
    if not test_result:
        raise ValueError("Embedding service validation failed")
    
    # Test du manager
    manager_result = await embedding_manager.generate_embedding("manager test")
    if not manager_result:
        raise ValueError("Embedding manager validation failed")
    
    logger.info("✅ Services d'embeddings validés avec succès")

async def initialize_query_processor() -> QueryProcessor:
    """Initialise le processeur de requêtes."""
    return QueryProcessor()

async def initialize_lexical_engine(elasticsearch_client: ElasticsearchClient) -> LexicalSearchEngine:
    """Initialise le moteur de recherche lexicale."""
    config = LexicalSearchConfig()
    return LexicalSearchEngine(elasticsearch_client, config)

async def initialize_semantic_engine(qdrant_client: QdrantClient, embedding_manager: EmbeddingManager) -> SemanticSearchEngine:
    """Initialise le moteur de recherche sémantique."""
    config = SemanticSearchConfig()
    
    # Validation préalable pour éviter l'erreur
    if not hasattr(embedding_manager, 'generate_embedding'):
        raise AttributeError(f"EmbeddingManager {type(embedding_manager)} missing generate_embedding method")
    
    return SemanticSearchEngine(qdrant_client, embedding_manager, config)

async def initialize_result_merger() -> ResultMerger:
    """Initialise le système de fusion des résultats."""
    config = FusionConfig()
    return ResultMerger(config)

async def initialize_hybrid_engine(
    lexical_engine: LexicalSearchEngine, 
    semantic_engine: Optional[SemanticSearchEngine], 
    result_merger: ResultMerger
) -> HybridSearchEngine:
    """Initialise le moteur de recherche hybride."""
    config = HybridSearchConfig()
    return HybridSearchEngine(lexical_engine, semantic_engine, result_merger, config)

async def perform_final_validation(deps: ServiceDependencies):
    """Effectue une validation finale complète du système."""
    
    # Validation des composants essentiels
    if not deps.is_fully_initialized():
        raise ValueError("Composants essentiels non initialisés")
    
    # Test de recherche lexicale
    try:
        lexical_test = await deps.lexical_engine.search("test", user_id=1, limit=1)
        logger.info("✅ Test recherche lexicale réussi")
    except Exception as e:
        logger.warning(f"⚠️ Test recherche lexicale échoué: {e}")
    
    # Test de recherche sémantique (si disponible)
    if deps.semantic_engine:
        try:
            semantic_test = await deps.semantic_engine.search("test", user_id=1, limit=1)
            logger.info("✅ Test recherche sémantique réussi")
        except Exception as e:
            logger.warning(f"⚠️ Test recherche sémantique échoué: {e}")
    
    # Test du moteur hybride
    try:
        hybrid_test = await deps.hybrid_engine.search("test", user_id=1, limit=1)
        logger.info("✅ Test recherche hybride réussi")
    except Exception as e:
        raise ValueError(f"Test hybride échoué: {e}")

# ========== GESTION DU CYCLE DE VIE ==========

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie avec initialisation robuste."""
    global app_dependencies
    
    logger.info("🚀 Démarrage du Search Service...")
    
    try:
        # Initialisation complète
        app_dependencies = await initialize_step_by_step()
        logger.info("✅ Search Service démarré avec succès")
        
    except InitializationError as e:
        logger.error(f"❌ Erreur critique d'initialisation: {e}")
        # En production, on pourrait choisir de continuer en mode très dégradé
        # ou de s'arrêter complètement selon la criticité
        raise
        
    except Exception as e:
        logger.error(f"❌ Erreur inattendue: {e}", exc_info=True)
        raise
    
    try:
        yield  # L'application s'exécute ici
    finally:
        # Nettoyage
        logger.info("🛑 Arrêt du Search Service...")
        await cleanup_resources()
        logger.info("✅ Search Service arrêté proprement")

async def cleanup_resources():
    """Nettoie les ressources à l'arrêt."""
    global app_dependencies
    
    if app_dependencies.elasticsearch_client:
        await app_dependencies.elasticsearch_client.close()
    
    if app_dependencies.qdrant_client:
        await app_dependencies.qdrant_client.close()
    
    if app_dependencies.embedding_service:
        await app_dependencies.embedding_service.close()

# ========== INJECTION DE DÉPENDANCES FASTAPI ==========

def get_elasticsearch_client() -> ElasticsearchClient:
    """Dependency provider pour le client Elasticsearch."""
    if not app_dependencies.elasticsearch_client:
        raise HTTPException(status_code=503, detail="Elasticsearch client not available")
    return app_dependencies.elasticsearch_client

def get_qdrant_client() -> QdrantClient:
    """Dependency provider pour le client Qdrant."""
    if not app_dependencies.qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant client not available")
    return app_dependencies.qdrant_client

def get_embedding_manager() -> Optional[EmbeddingManager]:
    """Dependency provider pour le gestionnaire d'embeddings."""
    return app_dependencies.embedding_manager

def get_query_processor() -> QueryProcessor:
    """Dependency provider pour le processeur de requêtes."""
    if not app_dependencies.query_processor:
        raise HTTPException(status_code=503, detail="Query processor not available")
    return app_dependencies.query_processor

def get_lexical_engine() -> LexicalSearchEngine:
    """Dependency provider pour le moteur lexical."""
    if not app_dependencies.lexical_engine:
        raise HTTPException(status_code=503, detail="Lexical engine not available")
    return app_dependencies.lexical_engine

def get_semantic_engine() -> Optional[SemanticSearchEngine]:
    """Dependency provider pour le moteur sémantique."""
    return app_dependencies.semantic_engine

def get_result_merger() -> ResultMerger:
    """Dependency provider pour le merger."""
    if not app_dependencies.result_merger:
        raise HTTPException(status_code=503, detail="Result merger not available")
    return app_dependencies.result_merger

def get_hybrid_engine() -> HybridSearchEngine:
    """Dependency provider pour le moteur hybride."""
    if not app_dependencies.hybrid_engine:
        raise HTTPException(status_code=503, detail="Hybrid engine not available")
    return app_dependencies.hybrid_engine

# ========== APPLICATION FASTAPI ==========

# Création de l'application avec cycle de vie
app = FastAPI(
    title="Harena Search Service",
    description="Service de recherche hybride (lexicale + sémantique) pour les transactions financières",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À configurer selon l'environnement
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== ROUTES DE SANTÉ ET DEBUG ==========

@app.get("/health")
async def health_check():
    """Vérification de santé détaillée."""
    uptime = time.time() - app_dependencies.startup_time if app_dependencies.startup_time else 0
    
    return {
        "status": "healthy" if app_dependencies.is_fully_initialized() else "degraded",
        "uptime_seconds": round(uptime, 2),
        "components": {
            "elasticsearch": app_dependencies.elasticsearch_client is not None,
            "qdrant": app_dependencies.qdrant_client is not None,
            "embeddings": app_dependencies.is_semantic_available(),
            "lexical_search": app_dependencies.lexical_engine is not None,
            "semantic_search": app_dependencies.semantic_engine is not None,
            "hybrid_search": app_dependencies.hybrid_engine is not None,
        },
        "initialization_results": app_dependencies.initialization_results,
        "timestamp": time.time()
    }

@app.get("/debug/dependencies")
async def debug_dependencies():
    """Debug détaillé des dépendances pour résoudre les problèmes."""
    return {
        "embedding_service": {
            "exists": app_dependencies.embedding_service is not None,
            "type": str(type(app_dependencies.embedding_service)) if app_dependencies.embedding_service else None,
            "has_generate_method": hasattr(app_dependencies.embedding_service, 'generate_embedding') if app_dependencies.embedding_service else False,
        },
        "embedding_manager": {
            "exists": app_dependencies.embedding_manager is not None,
            "type": str(type(app_dependencies.embedding_manager)) if app_dependencies.embedding_manager else None,
            "has_primary_service": hasattr(app_dependencies.embedding_manager, 'primary_service') if app_dependencies.embedding_manager else False,
            "primary_service_type": str(type(app_dependencies.embedding_manager.primary_service)) if app_dependencies.embedding_manager and hasattr(app_dependencies.embedding_manager, 'primary_service') else None,
            "has_generate_method": hasattr(app_dependencies.embedding_manager, 'generate_embedding') if app_dependencies.embedding_manager else False,
        },
        "semantic_engine": {
            "exists": app_dependencies.semantic_engine is not None,
            "type": str(type(app_dependencies.semantic_engine)) if app_dependencies.semantic_engine else None,
        },
        "full_status": {
            "fully_initialized": app_dependencies.is_fully_initialized(),
            "semantic_available": app_dependencies.is_semantic_available(),
        }
    }

# ========== ROUTES PRINCIPALES ==========

# Import et inclusion des routes principales (à adapter selon votre structure)
from search_service.api.routes import router

# Inclusion du router avec injection de dépendances automatique
app.include_router(router, prefix="/api/v1")

# ========== POINT D'ENTRÉE ==========

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )