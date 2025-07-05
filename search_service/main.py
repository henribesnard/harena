"""
Point d'entrée principal du search_service.

Ce module initialise et configure l'application FastAPI pour le service
de recherche hybride de transactions financières avec tous les composants
nécessaires : clients, moteurs, fusion et cache.
"""
import asyncio
import logging
import logging.config
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
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

# API
from search_service.api.routes import router
from search_service.api.dependencies import (
    get_current_user, validate_search_request, rate_limit
)

# Setup logging
logging.config.dictConfig(get_logging_config())
logger = logging.getLogger(__name__)

# Variables globales pour les services (injectées dans les routes)
elasticsearch_client: Optional[ElasticsearchClient] = None
qdrant_client: Optional[QdrantClient] = None
embedding_service: Optional[EmbeddingService] = None
embedding_manager: Optional[EmbeddingManager] = None
query_processor: Optional[QueryProcessor] = None
lexical_engine: Optional[LexicalSearchEngine] = None
semantic_engine: Optional[SemanticSearchEngine] = None
result_merger: Optional[ResultMerger] = None
hybrid_engine: Optional[HybridSearchEngine] = None

# Métriques de démarrage
startup_time: Optional[float] = None
initialization_results: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application."""
    global startup_time
    startup_time = time.time()
    
    logger.info("🚀 Starting Search Service...")
    
    # Startup
    try:
        await startup_event()
        logger.info("✅ Search Service démarré avec succès")
    except Exception as e:
        logger.error(f"❌ Échec du démarrage: {e}", exc_info=True)
        raise
    
    try:
        yield
    finally:
        # Shutdown
        await shutdown_event()
        logger.info("🛑 Search Service arrêté")


async def startup_event():
    """Initialisation complète au démarrage de l'application."""
    global elasticsearch_client, qdrant_client, embedding_service, embedding_manager
    global query_processor, lexical_engine, semantic_engine, result_merger, hybrid_engine
    
    try:
        # 1. Charger et valider la configuration
        await initialize_configuration()
        
        # 2. Initialiser les clients de base de données
        await initialize_clients()
        
        # 3. Initialiser les services d'embeddings (CORRIGÉ)
        await initialize_embedding_services()
        
        # 4. Validation des services d'embeddings (NOUVEAU)
        validate_embedding_injection()
        
        # 5. Initialiser les moteurs de recherche
        await initialize_search_engines()
        
        # 6. Initialiser le moteur hybride
        await initialize_hybrid_engine()
        
        # 7. Injecter les dépendances dans les routes (CORRIGÉ)
        inject_dependencies_into_routes()
        
        # 8. Effectuer les vérifications de santé
        await perform_health_checks()
        
        # 9. Optionnel: Warmup du système
        if getattr(global_settings, 'SEARCH_WARMUP_ENABLED', False):
            await warmup_search_engines()
        
        logger.info("🎉 Initialisation complète du Search Service terminée")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation: {e}", exc_info=True)
        raise RuntimeError(f"Search Service initialization failed: {e}")


async def initialize_configuration():
    """Charge et valide la configuration."""
    logger.info("⚙️ Initialisation de la configuration...")
    
    try:
        # Charger la configuration du search service
        search_settings = get_search_settings()
        validation = search_settings.validate_config()
        
        if not validation["valid"]:
            raise RuntimeError(f"Configuration invalide: {validation['errors']}")
        
        if validation["warnings"]:
            for warning in validation["warnings"]:
                logger.warning(f"⚠️ Configuration warning: {warning}")
        
        initialization_results["configuration"] = {
            "status": "success",
            "warnings": validation["warnings"],
            "settings_loaded": True
        }
        
        logger.info("✅ Configuration validée et chargée")
        
    except Exception as e:
        initialization_results["configuration"] = {
            "status": "failed",
            "error": str(e)
        }
        raise


async def initialize_clients():
    """Initialise les clients Elasticsearch et Qdrant."""
    global elasticsearch_client, qdrant_client
    
    logger.info("🔌 Initialisation des clients de base de données...")
    
    # Initialiser le client Elasticsearch
    try:
        if global_settings.BONSAI_URL:
            es_config = get_elasticsearch_config()
            elasticsearch_client = ElasticsearchClient(
                url=global_settings.BONSAI_URL,
                **es_config
            )
            
            # Test de connectivité
            await elasticsearch_client.health()
            logger.info("✅ Client Elasticsearch initialisé et connecté")
            
            initialization_results["elasticsearch"] = {
                "status": "success",
                "url": global_settings.BONSAI_URL,
                "connected": True
            }
        else:
            logger.warning("⚠️ BONSAI_URL non configuré, Elasticsearch désactivé")
            initialization_results["elasticsearch"] = {
                "status": "disabled",
                "reason": "BONSAI_URL not configured"
            }
        
    except Exception as e:
        logger.error(f"❌ Échec initialisation Elasticsearch: {e}")
        initialization_results["elasticsearch"] = {
            "status": "failed",
            "error": str(e)
        }
        elasticsearch_client = None
    
    # Initialiser le client Qdrant
    try:
        if global_settings.QDRANT_URL and global_settings.QDRANT_API_KEY:
            qdrant_config = get_qdrant_config()
            qdrant_client = QdrantClient(
                url=global_settings.QDRANT_URL,
                api_key=global_settings.QDRANT_API_KEY,
                **qdrant_config
            )
            
            # Test de connectivité
            await qdrant_client.health_check()
            logger.info("✅ Client Qdrant initialisé et connecté")
            
            initialization_results["qdrant"] = {
                "status": "success",
                "url": global_settings.QDRANT_URL,
                "connected": True
            }
        else:
            logger.warning("⚠️ QDRANT_URL ou QDRANT_API_KEY non configuré, Qdrant désactivé")
            initialization_results["qdrant"] = {
                "status": "disabled",
                "reason": "QDRANT_URL or QDRANT_API_KEY not configured"
            }
        
    except Exception as e:
        logger.error(f"❌ Échec initialisation Qdrant: {e}")
        initialization_results["qdrant"] = {
            "status": "failed",
            "error": str(e)
        }
        qdrant_client = None
    
    # Vérifier qu'au moins un client est disponible
    if not elasticsearch_client and not qdrant_client:
        raise RuntimeError("Aucun client de base de données disponible")


async def initialize_embedding_services():
    """Initialise les services d'embeddings avec vérifications renforcées."""
    global embedding_service, embedding_manager
    
    logger.info("🤖 Initialisation des services d'embeddings...")
    
    try:
        # Vérifier la clé API OpenAI
        if not global_settings.OPENAI_API_KEY:
            logger.warning("⚠️ OPENAI_API_KEY non définie, service d'embeddings désactivé")
            initialization_results["embeddings"] = {
                "status": "disabled",
                "reason": "OPENAI_API_KEY not configured"
            }
            embedding_service = None
            embedding_manager = None
            return
        
        # Log de debug pour la clé API (masquée)
        api_key_preview = f"{global_settings.OPENAI_API_KEY[:10]}..." if global_settings.OPENAI_API_KEY else "None"
        logger.info(f"🔑 OPENAI_API_KEY trouvée: {api_key_preview}")
        
        # Configuration des embeddings
        embedding_config = EmbeddingConfig(
            **get_embedding_config()
        )
        
        logger.info(f"📋 Configuration embeddings: model={embedding_config.model.value}, dimensions={embedding_config.dimensions}")
        
        # Créer le service d'embeddings principal
        embedding_service = EmbeddingService(
            api_key=global_settings.OPENAI_API_KEY,
            config=embedding_config
        )
        
        # Vérification que l'instance est bien créée
        if not isinstance(embedding_service, EmbeddingService):
            raise Exception(f"Expected EmbeddingService instance, got {type(embedding_service)}")
        
        logger.info(f"✅ EmbeddingService créé: {type(embedding_service)}")
        
        # Créer le gestionnaire avec l'instance vérifiée
        embedding_manager = EmbeddingManager(embedding_service)
        
        # Vérification que le gestionnaire est bien créé
        if not isinstance(embedding_manager, EmbeddingManager):
            raise Exception(f"Expected EmbeddingManager instance, got {type(embedding_manager)}")
        
        # Vérification que l'injection s'est bien passée
        if not hasattr(embedding_manager, 'primary_service'):
            raise Exception("EmbeddingManager.primary_service not set correctly")
            
        if not isinstance(embedding_manager.primary_service, EmbeddingService):
            raise Exception(f"Primary service has wrong type: {type(embedding_manager.primary_service)}")
        
        logger.info(f"✅ EmbeddingManager créé avec primary_service: {type(embedding_manager.primary_service)}")
        
        # Test de génération d'embedding
        test_embedding = await embedding_service.generate_embedding(
            "test query for initialization",
            use_cache=False
        )
        
        if test_embedding and len(test_embedding) > 0:
            logger.info(f"✅ Test d'embedding réussi: {len(test_embedding)} dimensions")
            initialization_results["embeddings"] = {
                "status": "success",
                "model": embedding_config.model.value,
                "dimensions": embedding_config.dimensions,
                "test_successful": True,
                "service_type": str(type(embedding_service)),
                "manager_type": str(type(embedding_manager)),
                "primary_service_type": str(type(embedding_manager.primary_service)),
                "test_embedding_dimensions": len(test_embedding)
            }
        else:
            raise Exception("Test d'embedding échoué: embedding vide ou None")
        
    except Exception as e:
        logger.error(f"❌ Échec initialisation embeddings: {e}")
        logger.error(f"❌ OPENAI_API_KEY présente: {bool(global_settings.OPENAI_API_KEY)}")
        logger.error(f"❌ Type embedding_service: {type(embedding_service) if 'embedding_service' in locals() else 'undefined'}")
        logger.error(f"❌ Type embedding_manager: {type(embedding_manager) if 'embedding_manager' in locals() else 'undefined'}")
        
        initialization_results["embeddings"] = {
            "status": "failed",
            "error": str(e),
            "debug_info": {
                "api_key_present": bool(global_settings.OPENAI_API_KEY),
                "embedding_service_type": str(type(embedding_service)) if 'embedding_service' in locals() else None,
                "embedding_manager_type": str(type(embedding_manager)) if 'embedding_manager' in locals() else None
            }
        }
        embedding_service = None
        embedding_manager = None
        # Ne pas lever l'exception pour permettre au service de fonctionner en mode dégradé


def validate_embedding_injection():
    """Valide que l'injection des services d'embeddings s'est bien passée."""
    global embedding_service, embedding_manager
    
    logger.info("🔍 Validation de l'injection des services d'embeddings...")
    
    errors = []
    warnings = []
    
    # Validation embedding_service
    if embedding_service is None:
        warnings.append("embedding_service is None - recherche sémantique désactivée")
    elif not isinstance(embedding_service, EmbeddingService):
        errors.append(f"embedding_service has wrong type: {type(embedding_service)}")
    else:
        logger.info(f"✅ embedding_service valid: {type(embedding_service)}")
    
    # Validation embedding_manager
    if embedding_manager is None:
        warnings.append("embedding_manager is None - recherche sémantique désactivée")
    elif not isinstance(embedding_manager, EmbeddingManager):
        errors.append(f"embedding_manager has wrong type: {type(embedding_manager)}")
    elif not hasattr(embedding_manager, 'primary_service'):
        errors.append("embedding_manager has no primary_service attribute")
    elif not isinstance(embedding_manager.primary_service, EmbeddingService):
        errors.append(f"embedding_manager.primary_service has wrong type: {type(embedding_manager.primary_service)}")
    elif not hasattr(embedding_manager.primary_service, 'generate_embedding'):
        errors.append("embedding_manager.primary_service has no generate_embedding method")
    else:
        logger.info(f"✅ embedding_manager valid with primary_service: {type(embedding_manager.primary_service)}")
    
    # Affichage des résultats
    if errors:
        logger.error("🚨 EMBEDDING INJECTION VALIDATION FAILED:")
        for error in errors:
            logger.error(f"   ❌ {error}")
        raise RuntimeError(f"Embedding injection validation failed: {'; '.join(errors)}")
    
    if warnings:
        logger.warning("⚠️ EMBEDDING INJECTION WARNINGS:")
        for warning in warnings:
            logger.warning(f"   ⚠️ {warning}")
    
    logger.info("✅ Validation de l'injection des embeddings réussie")
    
    return len(errors) == 0


async def initialize_search_engines():
    """Initialise les moteurs de recherche lexical et sémantique."""
    global query_processor, lexical_engine, semantic_engine, result_merger
    
    logger.info("🔍 Initialisation des moteurs de recherche...")
    
    # Initialiser le processeur de requêtes
    try:
        query_processor = QueryProcessor()
        logger.info("✅ Query processor initialisé")
        
        initialization_results["query_processor"] = {"status": "success"}
        
    except Exception as e:
        logger.error(f"❌ Échec initialisation query processor: {e}")
        initialization_results["query_processor"] = {
            "status": "failed",
            "error": str(e)
        }
        raise
    
    # Initialiser le moteur lexical
    if elasticsearch_client:
        try:
            lexical_config = LexicalSearchConfig()
            lexical_engine = LexicalSearchEngine(
                elasticsearch_client=elasticsearch_client,
                query_processor=query_processor,
                config=lexical_config
            )
            
            # Test de santé
            health = await lexical_engine.health_check()
            if health["status"] == "healthy":
                logger.info("✅ Moteur de recherche lexical initialisé")
                initialization_results["lexical_engine"] = {
                    "status": "success",
                    "health": health
                }
            else:
                raise Exception(f"Lexical engine unhealthy: {health}")
                
        except Exception as e:
            logger.error(f"❌ Échec initialisation moteur lexical: {e}")
            initialization_results["lexical_engine"] = {
                "status": "failed",
                "error": str(e)
            }
            lexical_engine = None
    else:
        logger.warning("⚠️ Moteur lexical désactivé (Elasticsearch non disponible)")
        initialization_results["lexical_engine"] = {
            "status": "disabled",
            "reason": "Elasticsearch not available"
        }
    
    # Initialiser le moteur sémantique
    if qdrant_client and embedding_manager:
        try:
            semantic_config = SemanticSearchConfig()
            semantic_engine = SemanticSearchEngine(
                qdrant_client=qdrant_client,
                embedding_manager=embedding_manager,
                query_processor=query_processor,
                config=semantic_config
            )
            
            # Test de santé
            health = await semantic_engine.health_check()
            if health["status"] == "healthy":
                logger.info("✅ Moteur de recherche sémantique initialisé")
                initialization_results["semantic_engine"] = {
                    "status": "success",
                    "health": health
                }
            else:
                logger.warning(f"⚠️ Moteur sémantique en état dégradé: {health}")
                initialization_results["semantic_engine"] = {
                    "status": "degraded",
                    "health": health
                }
                # Ne pas désactiver complètement, garder en mode dégradé
                
        except Exception as e:
            logger.error(f"❌ Échec initialisation moteur sémantique: {e}")
            initialization_results["semantic_engine"] = {
                "status": "failed",
                "error": str(e)
            }
            semantic_engine = None
    else:
        reason_parts = []
        if not qdrant_client:
            reason_parts.append("Qdrant not available")
        if not embedding_manager:
            reason_parts.append("Embeddings not available")
        reason = " and ".join(reason_parts)
        
        logger.warning(f"⚠️ Moteur sémantique désactivé ({reason})")
        initialization_results["semantic_engine"] = {
            "status": "disabled",
            "reason": reason
        }
    
    # Initialiser le fusionneur de résultats
    try:
        fusion_config = FusionConfig()
        result_merger = ResultMerger(config=fusion_config)
        
        logger.info("✅ Result merger initialisé")
        initialization_results["result_merger"] = {"status": "success"}
        
    except Exception as e:
        logger.error(f"❌ Échec initialisation result merger: {e}")
        initialization_results["result_merger"] = {
            "status": "failed",
            "error": str(e)
        }
        raise


async def initialize_hybrid_engine():
    """Initialise le moteur de recherche hybride principal."""
    global hybrid_engine
    
    logger.info("🎯 Initialisation du moteur hybride...")
    
    try:
        # Vérifier qu'au moins un moteur est disponible
        if not lexical_engine and not semantic_engine:
            raise RuntimeError("Aucun moteur de recherche disponible pour le mode hybride")
        
        # Configuration hybride
        hybrid_config = HybridSearchConfig(
            **get_hybrid_search_config()
        )
        
        # Créer le moteur hybride
        hybrid_engine = HybridSearchEngine(
            lexical_engine=lexical_engine,
            semantic_engine=semantic_engine,
            query_processor=query_processor,
            result_merger=result_merger,
            config=hybrid_config
        )
        
        # Test de santé
        health = await hybrid_engine.health_check()
        
        available_engines = sum(
            1 for engine_health in health["engines"].values()
            if engine_health.get("status") in ["healthy", "degraded"]
        )
        
        if available_engines > 0:
            logger.info(f"✅ Moteur hybride initialisé ({available_engines} moteurs disponibles)")
            initialization_results["hybrid_engine"] = {
                "status": "success",
                "available_engines": available_engines,
                "health": health
            }
        else:
            raise Exception("Aucun moteur sous-jacent disponible")
        
    except Exception as e:
        logger.error(f"❌ Échec initialisation moteur hybride: {e}")
        initialization_results["hybrid_engine"] = {
            "status": "failed",
            "error": str(e)
        }
        raise


def inject_dependencies_into_routes():
    """Injecte les services dans le module routes pour utilisation globale."""
    
    logger.info("💉 Injection des dépendances dans les routes...")
    
    try:
        # Import du module routes pour injection
        from search_service.api import routes
        
        # Injecter toutes les dépendances avec vérification
        routes.elasticsearch_client = elasticsearch_client
        routes.qdrant_client = qdrant_client
        routes.embedding_manager = embedding_manager
        routes.query_processor = query_processor
        routes.lexical_engine = lexical_engine
        routes.semantic_engine = semantic_engine
        routes.result_merger = result_merger
        routes.hybrid_engine = hybrid_engine
        
        # Vérification de l'injection
        injection_success = {
            "elasticsearch_client": routes.elasticsearch_client is not None,
            "qdrant_client": routes.qdrant_client is not None,
            "embedding_manager": routes.embedding_manager is not None,
            "query_processor": routes.query_processor is not None,
            "lexical_engine": routes.lexical_engine is not None,
            "semantic_engine": routes.semantic_engine is not None,
            "result_merger": routes.result_merger is not None,
            "hybrid_engine": routes.hybrid_engine is not None
        }
        
        successful_injections = sum(injection_success.values())
        total_injections = len(injection_success)
        
        logger.info(f"✅ Dépendances injectées dans les routes: {successful_injections}/{total_injections}")
        
        # Log détaillé des injections
        for component, success in injection_success.items():
            status = "✅" if success else "❌"
            logger.info(f"   {status} {component}: {success}")
        
        initialization_results["dependency_injection"] = {
            "status": "success",
            "successful_injections": successful_injections,
            "total_injections": total_injections,
            "details": injection_success
        }
        
    except Exception as e:
        logger.error(f"❌ Échec injection des dépendances: {e}")
        initialization_results["dependency_injection"] = {
            "status": "failed",
            "error": str(e)
        }
        raise


async def perform_health_checks():
    """Effectue des vérifications de santé sur tous les composants."""
    logger.info("🏥 Vérifications de santé des composants...")
    
    health_results = {}
    
    # Check des clients
    if elasticsearch_client:
        try:
            es_health = await elasticsearch_client.health()
            health_results["elasticsearch"] = es_health
        except Exception as e:
            health_results["elasticsearch"] = {"status": "unhealthy", "error": str(e)}
    
    if qdrant_client:
        try:
            qdrant_health = await qdrant_client.health_check()
            health_results["qdrant"] = qdrant_health
        except Exception as e:
            health_results["qdrant"] = {"status": "unhealthy", "error": str(e)}
    
    # Check des services
    if embedding_service:
        try:
            # Test simple sans health_check s'il n'existe pas
            test_embedding = await embedding_service.generate_embedding("test", use_cache=False)
            health_results["embeddings"] = {
                "status": "healthy" if test_embedding else "unhealthy",
                "test_successful": bool(test_embedding)
            }
        except Exception as e:
            health_results["embeddings"] = {"status": "unhealthy", "error": str(e)}
    
    # Check du moteur hybride
    if hybrid_engine:
        try:
            hybrid_health = await hybrid_engine.health_check()
            health_results["hybrid_engine"] = hybrid_health
        except Exception as e:
            health_results["hybrid_engine"] = {"status": "unhealthy", "error": str(e)}
    
    initialization_results["health_checks"] = health_results
    
    # Compter les composants sains
    healthy_components = sum(
        1 for health in health_results.values()
        if health.get("status") in ["healthy", "degraded"]
    )
    
    logger.info(f"✅ Health checks terminés: {healthy_components}/{len(health_results)} composants sains")


async def warmup_search_engines():
    """Réchauffe les moteurs de recherche avec des requêtes prédéfinies."""
    logger.info("🔥 Warmup des moteurs de recherche...")
    
    if not hybrid_engine:
        logger.warning("⚠️ Pas de moteur hybride disponible pour le warmup")
        return
    
    try:
        # Requêtes de warmup simples
        warmup_queries = [
            "test",
            "paiement",
            "achat",
            "virement",
            "carte bancaire"
        ]
        
        successful_warmups = 0
        total_time = 0
        
        for query in warmup_queries:
            try:
                start_time = time.time()
                # Test simple de recherche
                await hybrid_engine.search(query, user_id=1, limit=1)
                end_time = time.time()
                
                successful_warmups += 1
                total_time += (end_time - start_time) * 1000
                
            except Exception as e:
                logger.warning(f"Warmup failed for query '{query}': {e}")
        
        initialization_results["warmup"] = {
            "status": "completed",
            "successful_queries": successful_warmups,
            "total_queries": len(warmup_queries),
            "total_time_ms": total_time
        }
        
        logger.info(f"✅ Warmup terminé: {successful_warmups}/{len(warmup_queries)} requêtes")
        
    except Exception as e:
        logger.error(f"❌ Échec du warmup: {e}")
        initialization_results["warmup"] = {
            "status": "failed",
            "error": str(e)
        }


async def shutdown_event():
    """Nettoyage lors de l'arrêt de l'application."""
    logger.info("🛑 Arrêt du Search Service...")
    
    # Fermer les clients
    if elasticsearch_client:
        try:
            await elasticsearch_client.close()
            logger.info("✅ Client Elasticsearch fermé")
        except Exception as e:
            logger.error(f"❌ Erreur fermeture Elasticsearch: {e}")
    
    if qdrant_client:
        try:
            await qdrant_client.close()
            logger.info("✅ Client Qdrant fermé")
        except Exception as e:
            logger.error(f"❌ Erreur fermeture Qdrant: {e}")
    
    # Vider les caches
    if hybrid_engine:
        try:
            hybrid_engine.clear_cache()
            logger.info("✅ Caches vidés")
        except Exception as e:
            logger.error(f"❌ Erreur vidage cache: {e}")
    
    logger.info("✅ Arrêt propre du Search Service terminé")


def create_search_app() -> FastAPI:
    """
    Crée et configure l'application FastAPI pour le search service.
    
    Returns:
        Application FastAPI configurée
    """
    
    # Créer l'application avec cycle de vie
    app = FastAPI(
        title="Harena Search Service",
        description="Service de recherche hybride pour transactions financières",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if getattr(global_settings, 'DEBUG', False) else None,
        redoc_url="/redoc" if getattr(global_settings, 'DEBUG', False) else None
    )
    
    # Configuration CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if getattr(global_settings, 'DEBUG', False) else ["https://harena.app"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"]
    )
    
    # Inclure les routes principales
    app.include_router(router, prefix="/api/v1", tags=["search"])
    
    # Route de santé globale
    @app.get("/health")
    async def health_check():
        """Point de santé global du service."""
        uptime = time.time() - startup_time if startup_time else 0
        
        return {
            "status": "healthy",
            "service": "search_service",
            "version": "1.0.0",
            "uptime_seconds": round(uptime, 2),
            "initialization": initialization_results,
            "components": {
                "elasticsearch": elasticsearch_client is not None,
                "qdrant": qdrant_client is not None,
                "embeddings": embedding_manager is not None,
                "lexical_engine": lexical_engine is not None,
                "semantic_engine": semantic_engine is not None,
                "hybrid_engine": hybrid_engine is not None
            }
        }
    
    # Route de debugging pour les embeddings
    @app.get("/debug/embedding")
    async def debug_embedding():
        """Informations de debug sur les services d'embeddings."""
        return {
            "embedding_service": {
                "exists": embedding_service is not None,
                "type": str(type(embedding_service)) if embedding_service else None,
                "has_generate_method": hasattr(embedding_service, 'generate_embedding') if embedding_service else False
            },
            "embedding_manager": {
                "exists": embedding_manager is not None,
                "type": str(type(embedding_manager)) if embedding_manager else None,
                "has_primary_service": hasattr(embedding_manager, 'primary_service') if embedding_manager else False,
                "primary_service_type": str(type(embedding_manager.primary_service)) if embedding_manager and hasattr(embedding_manager, 'primary_service') else None,
                "can_generate": hasattr(embedding_manager.primary_service, 'generate_embedding') if embedding_manager and hasattr(embedding_manager, 'primary_service') else False
            },
            "openai_api_key": {
                "configured": bool(global_settings.OPENAI_API_KEY),
                "preview": f"{global_settings.OPENAI_API_KEY[:10]}..." if global_settings.OPENAI_API_KEY else None
            },
            "initialization_results": initialization_results.get("embeddings", {})
        }
    
    # Route d'information détaillée (admin)
    @app.get("/info")
    async def service_info():
        """Informations détaillées du service (admin)."""
        if not getattr(global_settings, 'DEBUG', False):
            raise HTTPException(status_code=404, detail="Not found")
        
        metrics = {}
        if hybrid_engine:
            metrics = hybrid_engine.get_metrics()
        
        return {
            "service": "search_service",
            "initialization_results": initialization_results,
            "metrics": metrics,
            "performance_summary": hybrid_engine.get_performance_summary() if hybrid_engine else None,
            "components_status": {
                "elasticsearch_client": {
                    "available": elasticsearch_client is not None,
                    "type": str(type(elasticsearch_client)) if elasticsearch_client else None
                },
                "qdrant_client": {
                    "available": qdrant_client is not None,
                    "type": str(type(qdrant_client)) if qdrant_client else None
                },
                "embedding_service": {
                    "available": embedding_service is not None,
                    "type": str(type(embedding_service)) if embedding_service else None
                },
                "embedding_manager": {
                    "available": embedding_manager is not None,
                    "type": str(type(embedding_manager)) if embedding_manager else None,
                    "primary_service_type": str(type(embedding_manager.primary_service)) if embedding_manager and hasattr(embedding_manager, 'primary_service') else None
                },
                "lexical_engine": {
                    "available": lexical_engine is not None,
                    "type": str(type(lexical_engine)) if lexical_engine else None
                },
                "semantic_engine": {
                    "available": semantic_engine is not None,
                    "type": str(type(semantic_engine)) if semantic_engine else None
                },
                "hybrid_engine": {
                    "available": hybrid_engine is not None,
                    "type": str(type(hybrid_engine)) if hybrid_engine else None
                }
            }
        }
    
    # Route de test pour les embeddings
    @app.post("/test/embedding")
    async def test_embedding_endpoint(text: str = "test query"):
        """Test de génération d'embedding (admin/debug)."""
        if not getattr(global_settings, 'DEBUG', False):
            raise HTTPException(status_code=404, detail="Not found")
        
        if not embedding_manager:
            raise HTTPException(status_code=503, detail="Embedding service not available")
        
        try:
            start_time = time.time()
            embedding = await embedding_manager.generate_embedding(text, use_cache=False)
            end_time = time.time()
            
            return {
                "success": True,
                "text": text,
                "embedding_dimensions": len(embedding) if embedding else 0,
                "processing_time_ms": (end_time - start_time) * 1000,
                "embedding_preview": embedding[:5] if embedding else None
            }
        except Exception as e:
            return {
                "success": False,
                "text": text,
                "error": str(e),
                "error_type": str(type(e))
            }
    
    # Gestionnaire d'erreurs global
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"Erreur non gérée: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc) if getattr(global_settings, 'DEBUG', False) else "Une erreur est survenue",
                "service": "search_service"
            }
        )
    
    return app


# Point d'entrée principal
if __name__ == "__main__":
    app = create_search_app()
    
    # Configuration de développement
    if getattr(global_settings, 'DEBUG', False):
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8003,  # Port spécifique au search service
            reload=True,
            log_level="info"
        )
    else:
        # Configuration de production
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=int(getattr(global_settings, 'PORT', 8003)),
            log_level="warning"
        )