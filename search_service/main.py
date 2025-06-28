"""
Point d'entrée principal du search_service.

Ce module initialise et configure l'application FastAPI pour le service
de recherche hybride de transactions financières.
"""
import asyncio
import logging
import logging.config
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any

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
    get_qdrant_config, get_embedding_config
)

# Clients et composants core
from search_service.clients import ElasticsearchClient, QdrantClient
from search_service.core.embeddings import EmbeddingService, EmbeddingManager, EmbeddingConfig
from search_service.core.query_processor import QueryProcessor
from search_service.core.lexical_engine import LexicalSearchEngine

# API
from search_service.api import routes, dependencies

# Setup logging
logging.config.dictConfig(get_logging_config())
logger = logging.getLogger(__name__)

# Variables globales pour les services
elasticsearch_client: ElasticsearchClient = None
qdrant_client: QdrantClient = None
embedding_manager: EmbeddingManager = None
query_processor: QueryProcessor = None
lexical_engine: LexicalSearchEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application."""
    logger.info("🚀 Starting Search Service...")
    
    # Startup
    await startup_event()
    
    try:
        yield
    finally:
        # Shutdown
        await shutdown_event()


async def startup_event():
    """Initialisation au démarrage de l'application."""
    global elasticsearch_client, qdrant_client, embedding_manager, query_processor, lexical_engine
    
    try:
        # 1. Charger et valider la configuration
        search_settings = get_search_settings()
        validation = search_settings.validate_config()
        
        if not validation["valid"]:
            logger.error(f"❌ Configuration invalide: {validation['errors']}")
            raise RuntimeError("Invalid configuration")
        
        if validation["warnings"]:
            for warning in validation["warnings"]:
                logger.warning(f"⚠️ Configuration warning: {warning}")
        
        logger.info("✅ Configuration validée")
        
        # 2. Initialiser les clients de base
        await initialize_clients()
        
        # 3. Initialiser les services d'embeddings
        await initialize_embedding_services()
        
        # 4. Initialiser les moteurs de recherche
        await initialize_search_engines()
        
        # 5. Warmup optionnel
        if search_settings.performance.warmup_enabled:
            await perform_warmup()
        
        # 6. Injecter les dépendances dans les routes
        routes.elasticsearch_client = elasticsearch_client
        routes.qdrant_client = qdrant_client
        routes.embedding_manager = embedding_manager
        routes.query_processor = query_processor
        routes.lexical_engine = lexical_engine
        
        logger.info("🎉 Search Service démarré avec succès!")
        
    except Exception as e:
        logger.error(f"💥 Erreur lors du démarrage: {e}")
        raise


async def initialize_clients():
    """Initialise les clients Elasticsearch et Qdrant."""
    global elasticsearch_client, qdrant_client
    
    # Configuration Elasticsearch
    es_config = get_elasticsearch_config()
    
    if global_settings.BONSAI_URL and "your-" not in global_settings.BONSAI_URL:
        elasticsearch_client = ElasticsearchClient(
            bonsai_url=global_settings.BONSAI_URL,
            index_name="harena_transactions",
            timeout=es_config["timeout"]
        )
        
        await elasticsearch_client.start()
        
        # Test de connectivité
        if await elasticsearch_client.test_connection():
            logger.info("✅ Client Elasticsearch initialisé et connecté")
        else:
            logger.warning("⚠️ Client Elasticsearch initialisé mais connexion échouée")
    else:
        logger.warning("⚠️ BONSAI_URL non configurée - recherche lexicale indisponible")
    
    # Configuration Qdrant
    qdrant_config = get_qdrant_config()
    
    if global_settings.QDRANT_URL and "your-" not in global_settings.QDRANT_URL:
        qdrant_client = QdrantClient(
            qdrant_url=global_settings.QDRANT_URL,
            api_key=global_settings.QDRANT_API_KEY,
            collection_name="financial_transactions",
            timeout=qdrant_config["timeout"]
        )
        
        await qdrant_client.start()
        
        # Test de connectivité
        if await qdrant_client.test_connection():
            logger.info("✅ Client Qdrant initialisé et connecté")
        else:
            logger.warning("⚠️ Client Qdrant initialisé mais connexion échouée")
    else:
        logger.warning("⚠️ QDRANT_URL non configurée - recherche sémantique indisponible")


async def initialize_embedding_services():
    """Initialise les services d'embeddings."""
    global embedding_manager
    
    embedding_config = get_embedding_config()
    
    if global_settings.OPENAI_API_KEY and global_settings.OPENAI_API_KEY.startswith("sk-"):
        # Service principal OpenAI
        embedding_service_config = EmbeddingConfig(
            model=embedding_config["model"],
            dimensions=embedding_config["dimensions"],
            batch_size=embedding_config["batch_size"],
            cache_ttl_seconds=embedding_config["cache"]["ttl"],
            max_cache_size=embedding_config["cache"]["max_size"]
        )
        
        primary_embedding_service = EmbeddingService(
            api_key=global_settings.OPENAI_API_KEY,
            config=embedding_service_config,
            timeout=embedding_config["timeout"]
        )
        
        await primary_embedding_service.start()
        
        # Test de connectivité
        if await primary_embedding_service.test_connection():
            logger.info("✅ Service d'embeddings OpenAI initialisé")
            
            # Créer le gestionnaire d'embeddings
            embedding_manager = EmbeddingManager(primary_embedding_service)
            await embedding_manager.start_all_services()
            
        else:
            logger.warning("⚠️ Service d'embeddings initialisé mais connexion échouée")
    else:
        logger.warning("⚠️ OPENAI_API_KEY non configurée - embeddings indisponibles")


async def initialize_search_engines():
    """Initialise les moteurs de recherche."""
    global query_processor, lexical_engine
    
    # Processeur de requêtes (toujours disponible)
    query_processor = QueryProcessor()
    logger.info("✅ Processeur de requêtes initialisé")
    
    # Moteur de recherche lexicale
    if elasticsearch_client:
        lexical_engine = LexicalSearchEngine(
            elasticsearch_client=elasticsearch_client,
            query_processor=query_processor
        )
        logger.info("✅ Moteur de recherche lexicale initialisé")
    else:
        logger.warning("⚠️ Moteur lexical non initialisé (Elasticsearch indisponible)")


async def perform_warmup():
    """Effectue le warmup des services."""
    search_settings = get_search_settings()
    warmup_queries = search_settings.performance.warmup_queries
    
    if not warmup_queries:
        logger.info("⚡ Aucune requête de warmup configurée")
        return
    
    logger.info(f"⚡ Démarrage du warmup avec {len(warmup_queries)} requêtes...")
    
    # Test user_id pour le warmup
    test_user_id = 34
    
    success_count = 0
    
    # Warmup du moteur lexical
    if lexical_engine:
        try:
            warmup_success = await lexical_engine.warmup(test_user_id)
            if warmup_success:
                success_count += 1
                logger.info("✅ Warmup moteur lexical réussi")
            else:
                logger.warning("⚠️ Warmup moteur lexical partiellement échoué")
        except Exception as e:
            logger.warning(f"⚠️ Erreur warmup moteur lexical: {e}")
    
    # Warmup des embeddings
    if embedding_manager:
        try:
            test_embeddings = await embedding_manager.generate_embeddings_batch(
                warmup_queries[:3],  # Première moitié pour éviter les limits
                use_cache=True
            )
            
            successful_embeddings = sum(1 for emb in test_embeddings if emb is not None)
            if successful_embeddings > 0:
                success_count += 1
                logger.info(f"✅ Warmup embeddings réussi ({successful_embeddings}/{len(warmup_queries[:3])})")
            else:
                logger.warning("⚠️ Warmup embeddings échoué")
        except Exception as e:
            logger.warning(f"⚠️ Erreur warmup embeddings: {e}")
    
    logger.info(f"⚡ Warmup terminé: {success_count} services opérationnels")


async def shutdown_event():
    """Nettoyage à l'arrêt de l'application."""
    logger.info("🛑 Arrêt du Search Service...")
    
    try:
        # Fermer les clients
        if elasticsearch_client:
            await elasticsearch_client.close()
            logger.info("✅ Client Elasticsearch fermé")
        
        if qdrant_client:
            await qdrant_client.close()
            logger.info("✅ Client Qdrant fermé")
        
        if embedding_manager:
            await embedding_manager.close_all_services()
            logger.info("✅ Services d'embeddings fermés")
        
        logger.info("🏁 Search Service arrêté proprement")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'arrêt: {e}")


def create_search_app() -> FastAPI:
    """
    Crée et configure l'application FastAPI pour le search_service.
    
    Returns:
        Application FastAPI configurée
    """
    search_settings = get_search_settings()
    
    # Créer l'application avec gestion du cycle de vie
    app = FastAPI(
        title="Harena Search Service",
        description="Service de recherche hybride pour transactions financières",
        version=search_settings.search_service.version,
        debug=search_settings.search_service.debug,
        lifespan=lifespan
    )
    
    # Configuration CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # À restreindre en production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Inclure les routes avec préfixe
    app.include_router(
        routes.router,
        prefix="/api/v1",
        tags=["search"]
    )
    
    # Route de santé globale
    @app.get("/health")
    async def health_check():
        """Vérification de santé rapide."""
        return {
            "service": "search_service",
            "status": "healthy",
            "version": search_settings.search_service.version,
            "timestamp": "2025-06-28T07:15:21.601152"
        }
    
    # Route d'informations sur la configuration
    @app.get("/config")
    async def get_config_info():
        """Informations sur la configuration (sans secrets)."""
        config_info = search_settings.get_environment_info()
        
        # Masquer les clés sensibles
        if "environment_variables" in config_info:
            sensitive_vars = ["OPENAI_API_KEY", "QDRANT_API_KEY", "BONSAI_URL", "QDRANT_URL"]
            for var in sensitive_vars:
                if var in config_info["environment_variables"]:
                    value = config_info["environment_variables"][var]
                    if value:
                        config_info["environment_variables"][var] = f"{value[:8]}...{value[-4:]}"
        
        return config_info
    
    # Gestionnaire d'erreurs global
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        """Gestionnaire d'erreurs HTTP personnalisé."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "search_service_error",
                "message": exc.detail,
                "status_code": exc.status_code,
                "timestamp": "2025-06-28T07:15:21.601152"
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        """Gestionnaire d'erreurs générales."""
        logger.error(f"Erreur non gérée: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "Une erreur interne s'est produite",
                "status_code": 500,
                "timestamp": "2025-06-28T07:15:21.601152"
            }
        )
    
    return app


# Point d'entrée principal
app = create_search_app()


if __name__ == "__main__":
    """Démarrage en mode développement."""
    search_settings = get_search_settings()
    
    # Configuration du serveur
    uvicorn_config = {
        "app": "search_service.main:app",
        "host": "0.0.0.0",
        "port": 8003,  # Port spécifique au search_service
        "reload": search_settings.search_service.debug,
        "log_level": "debug" if search_settings.search_service.debug else "info",
        "access_log": True
    }
    
    logger.info("🚀 Démarrage du Search Service en mode développement")
    logger.info(f"📊 Configuration: {search_settings.search_service.service_name} v{search_settings.search_service.version}")
    logger.info(f"🌐 Serveur: http://localhost:{uvicorn_config['port']}")
    logger.info(f"📚 Documentation: http://localhost:{uvicorn_config['port']}/docs")
    
    # Afficher les services disponibles
    logger.info("🔧 Services configurés:")
    if hasattr(settings, 'BONSAI_URL') and settings.BONSAI_URL and "your-" not in settings.BONSAI_URL:
        logger.info("  ✅ Elasticsearch (recherche lexicale)")
    else:
        logger.info("  ❌ Elasticsearch non configuré")
    
    if hasattr(settings, 'QDRANT_URL') and settings.QDRANT_URL and "your-" not in settings.QDRANT_URL:
        logger.info("  ✅ Qdrant (recherche sémantique)")
    else:
        logger.info("  ❌ Qdrant non configuré")
    
    if hasattr(settings, 'OPENAI_API_KEY') and settings.OPENAI_API_KEY and settings.OPENAI_API_KEY.startswith("sk-"):
        logger.info("  ✅ OpenAI (embeddings)")
    else:
        logger.info("  ❌ OpenAI non configuré")
    
    # Démarrer le serveur
    uvicorn.run(**uvicorn_config)


def get_application() -> FastAPI:
    """
    Fonction utilitaire pour récupérer l'application configurée.
    Utilisée par Heroku et autres déploiements.
    """
    return app


# Fonctions d'assistance pour les tests et le debugging

async def test_search_functionality():
    """Teste les fonctionnalités de recherche de base."""
    test_results = {
        "elasticsearch": {"available": False, "tested": False, "results": 0},
        "qdrant": {"available": False, "tested": False, "results": 0},
        "embeddings": {"available": False, "tested": False, "generated": 0},
        "overall": {"status": "unknown", "details": []}
    }
    
    test_user_id = 34
    test_query = "restaurant"
    
    # Test Elasticsearch
    if elasticsearch_client:
        test_results["elasticsearch"]["available"] = True
        try:
            es_results = await elasticsearch_client.search_transactions(
                query=test_query,
                user_id=test_user_id,
                limit=5
            )
            hits = es_results.get("hits", {}).get("hits", [])
            test_results["elasticsearch"]["tested"] = True
            test_results["elasticsearch"]["results"] = len(hits)
            test_results["overall"]["details"].append(f"Elasticsearch: {len(hits)} résultats")
        except Exception as e:
            test_results["overall"]["details"].append(f"Elasticsearch error: {e}")
    
    # Test Qdrant
    if qdrant_client:
        test_results["qdrant"]["available"] = True
        try:
            # Générer un embedding de test d'abord
            if embedding_manager:
                test_embedding = await embedding_manager.generate_embedding(test_query)
                if test_embedding:
                    qdrant_results = await qdrant_client.search_similar_transactions(
                        query_vector=test_embedding,
                        user_id=test_user_id,
                        limit=5
                    )
                    results = qdrant_results.get("result", [])
                    test_results["qdrant"]["tested"] = True
                    test_results["qdrant"]["results"] = len(results)
                    test_results["overall"]["details"].append(f"Qdrant: {len(results)} résultats")
                else:
                    test_results["overall"]["details"].append("Qdrant: échec génération embedding")
            else:
                test_results["overall"]["details"].append("Qdrant: embedding manager indisponible")
        except Exception as e:
            test_results["overall"]["details"].append(f"Qdrant error: {e}")
    
    # Test Embeddings
    if embedding_manager:
        test_results["embeddings"]["available"] = True
        try:
            test_embeddings = await embedding_manager.generate_embeddings_batch([
                "restaurant", "virement", "carte bancaire"
            ])
            successful = sum(1 for emb in test_embeddings if emb is not None)
            test_results["embeddings"]["tested"] = True
            test_results["embeddings"]["generated"] = successful
            test_results["overall"]["details"].append(f"Embeddings: {successful}/3 générés")
        except Exception as e:
            test_results["overall"]["details"].append(f"Embeddings error: {e}")
    
    # Déterminer le statut global
    available_services = sum([
        test_results["elasticsearch"]["available"],
        test_results["qdrant"]["available"], 
        test_results["embeddings"]["available"]
    ])
    
    tested_services = sum([
        test_results["elasticsearch"]["tested"],
        test_results["qdrant"]["tested"],
        test_results["embeddings"]["tested"]
    ])
    
    if tested_services == available_services and available_services > 0:
        test_results["overall"]["status"] = "all_working"
    elif tested_services > 0:
        test_results["overall"]["status"] = "partial_working"
    else:
        test_results["overall"]["status"] = "not_working"
    
    return test_results


async def get_service_diagnostics() -> Dict[str, Any]:
    """Retourne un diagnostic complet des services."""
    diagnostics = {
        "timestamp": "2025-06-28T07:15:21.601152",
        "search_service": {
            "status": "running",
            "version": get_search_settings().search_service.version,
            "config_valid": get_search_settings().validate_config()["valid"]
        },
        "clients": {},
        "engines": {},
        "performance": {}
    }
    
    # Diagnostics des clients
    if elasticsearch_client:
        try:
            es_health = await elasticsearch_client.health_check()
            diagnostics["clients"]["elasticsearch"] = es_health
        except Exception as e:
            diagnostics["clients"]["elasticsearch"] = {"status": "error", "error": str(e)}
    else:
        diagnostics["clients"]["elasticsearch"] = {"status": "not_configured"}
    
    if qdrant_client:
        try:
            qdrant_health = await qdrant_client.health_check()
            diagnostics["clients"]["qdrant"] = qdrant_health
        except Exception as e:
            diagnostics["clients"]["qdrant"] = {"status": "error", "error": str(e)}
    else:
        diagnostics["clients"]["qdrant"] = {"status": "not_configured"}
    
    # Diagnostics des moteurs
    if lexical_engine:
        diagnostics["engines"]["lexical"] = lexical_engine.get_engine_stats()
    else:
        diagnostics["engines"]["lexical"] = {"status": "not_available"}
    
    # Diagnostics des embeddings
    if embedding_manager:
        diagnostics["engines"]["embeddings"] = embedding_manager.get_manager_stats()
    else:
        diagnostics["engines"]["embeddings"] = {"status": "not_available"}
    
    # Test de fonctionnalité
    try:
        functionality_test = await test_search_functionality()
        diagnostics["functionality_test"] = functionality_test
    except Exception as e:
        diagnostics["functionality_test"] = {"error": str(e)}
    
    return diagnostics


# Variables pour l'import global
from config_service.config import settings