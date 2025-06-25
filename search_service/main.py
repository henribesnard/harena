"""
Point d'entrée principal pour le service de recherche.
VERSION CORRIGÉE - Gestion propre des connexions et cycle de vie
"""
import asyncio
import logging
import signal
import sys
import time
import atexit
from contextlib import asynccontextmanager
from typing import Optional, Any, List

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config_service.config import settings

# Configuration du logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("search_service.main")

# Variables globales pour les services
elastic_client: Optional[Any] = None
qdrant_client: Optional[Any] = None
search_engine: Optional[Any] = None
initialization_errors: List[str] = []
cleanup_tasks: List[asyncio.Task] = []
app_start_time = time.time()

# Gestionnaire de fermeture propre
shutdown_event = asyncio.Event()


def register_cleanup_task(coro_func, *args, **kwargs):
    """Enregistre une tâche de nettoyage."""
    try:
        task = asyncio.create_task(coro_func(*args, **kwargs))
        cleanup_tasks.append(task)
        logger.debug(f"Tâche de nettoyage enregistrée: {coro_func.__name__}")
    except Exception as e:
        logger.error(f"Erreur enregistrement tâche nettoyage: {e}")


async def cleanup_on_shutdown():
    """Nettoie les ressources lors de l'arrêt."""
    global elastic_client, qdrant_client, search_engine
    
    logger.info("🧹 Début du nettoyage des ressources...")
    start_cleanup = time.time()
    
    cleanup_errors = []
    
    # Fermer les clients individuellement avec timeout
    if elastic_client:
        try:
            logger.info("🔒 Fermeture client Elasticsearch...")
            await asyncio.wait_for(elastic_client.close(), timeout=5.0)
            logger.info("✅ Client Elasticsearch fermé")
        except asyncio.TimeoutError:
            logger.warning("⚠️ Timeout fermeture Elasticsearch")
            cleanup_errors.append("Elasticsearch timeout")
        except Exception as e:
            logger.error(f"❌ Erreur fermeture Elasticsearch: {e}")
            cleanup_errors.append(f"Elasticsearch: {e}")
        finally:
            elastic_client = None
    
    if qdrant_client:
        try:
            logger.info("🔒 Fermeture client Qdrant...")
            if hasattr(qdrant_client, 'close'):
                await asyncio.wait_for(qdrant_client.close(), timeout=5.0)
            logger.info("✅ Client Qdrant fermé")
        except asyncio.TimeoutError:
            logger.warning("⚠️ Timeout fermeture Qdrant")
            cleanup_errors.append("Qdrant timeout")
        except Exception as e:
            logger.error(f"❌ Erreur fermeture Qdrant: {e}")
            cleanup_errors.append(f"Qdrant: {e}")
        finally:
            qdrant_client = None
    
    # Nettoyer le moteur de recherche
    if search_engine:
        try:
            logger.info("🔒 Fermeture moteur de recherche...")
            if hasattr(search_engine, 'close'):
                await asyncio.wait_for(search_engine.close(), timeout=5.0)
            logger.info("✅ Moteur de recherche fermé")
        except Exception as e:
            logger.error(f"❌ Erreur fermeture moteur: {e}")
            cleanup_errors.append(f"SearchEngine: {e}")
        finally:
            search_engine = None
    
    # Attendre les tâches de nettoyage enregistrées
    if cleanup_tasks:
        try:
            logger.info(f"🔄 Attente de {len(cleanup_tasks)} tâches de nettoyage...")
            await asyncio.wait_for(
                asyncio.gather(*cleanup_tasks, return_exceptions=True),
                timeout=10.0
            )
            logger.info("✅ Tâches de nettoyage terminées")
        except asyncio.TimeoutError:
            logger.warning("⚠️ Timeout tâches de nettoyage")
            cleanup_errors.append("Cleanup tasks timeout")
        except Exception as e:
            logger.error(f"❌ Erreur tâches de nettoyage: {e}")
            cleanup_errors.append(f"Cleanup tasks: {e}")
    
    cleanup_time = time.time() - start_cleanup
    
    if cleanup_errors:
        logger.warning(f"🟡 Nettoyage terminé avec erreurs en {cleanup_time:.2f}s: {cleanup_errors}")
    else:
        logger.info(f"✅ Nettoyage terminé proprement en {cleanup_time:.2f}s")
    
    # Marquer l'arrêt comme terminé
    shutdown_event.set()


def signal_handler(signum, frame):
    """Gestionnaire de signal pour arrêt propre."""
    logger.info(f"🛑 Signal {signum} reçu, arrêt en cours...")
    
    # Créer la tâche de nettoyage dans la boucle d'événements actuelle
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(cleanup_on_shutdown())
    except RuntimeError:
        # Pas de boucle active, créer une nouvelle
        asyncio.run(cleanup_on_shutdown())


def setup_signal_handlers():
    """Configure les gestionnaires de signaux."""
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Gestionnaire atexit comme backup
    atexit.register(lambda: asyncio.run(cleanup_on_shutdown()))
    
    logger.info("🛡️ Gestionnaires de signaux configurés")


async def safe_initialize_elasticsearch() -> Optional[Any]:
    """Initialise Elasticsearch de manière sécurisée avec timeout."""
    global elastic_client
    
    logger.info("🔍 Initialisation d'Elasticsearch...")
    
    try:
        # Import avec gestion d'erreur
        try:
            from search_service.storage.elastic_client_hybrid import HybridElasticClient
        except ImportError as e:
            logger.error(f"❌ Import Elasticsearch échoué: {e}")
            initialization_errors.append(f"Elasticsearch import: {e}")
            return None
        
        # Vérification de la configuration
        if not settings.BONSAI_URL:
            logger.warning("⚠️ BONSAI_URL non configurée - Elasticsearch ignoré")
            return None
        
        # Initialisation avec timeout strict
        logger.info("🔄 Création du client Elasticsearch...")
        elastic_client = HybridElasticClient()
        
        logger.info("🔄 Initialisation du client Elasticsearch...")
        try:
            success = await asyncio.wait_for(elastic_client.initialize(), timeout=45.0)
            if not success:
                logger.error("❌ Elasticsearch: initialize() retourné False")
                initialization_errors.append("Elasticsearch initialization returned False")
                await elastic_client.close()
                elastic_client = None
                return None
        except asyncio.TimeoutError:
            logger.error("❌ Timeout (45s) lors de l'initialisation d'Elasticsearch")
            initialization_errors.append("Elasticsearch timeout during initialization")
            if elastic_client:
                await elastic_client.close()
                elastic_client = None
            return None
        
        # Vérification de l'état
        if hasattr(elastic_client, '_initialized') and elastic_client._initialized:
            logger.info("✅ Elasticsearch initialisé avec succès")
            
            # Test de santé rapide
            try:
                if hasattr(elastic_client, 'is_healthy'):
                    is_healthy = await asyncio.wait_for(elastic_client.is_healthy(), timeout=10.0)
                    if is_healthy:
                        logger.info("✅ Elasticsearch est en bonne santé")
                    else:
                        logger.warning("⚠️ Elasticsearch initialisé mais pas en bonne santé")
            except Exception as health_error:
                logger.warning(f"⚠️ Test de santé Elasticsearch échoué: {health_error}")
            
            # Enregistrer pour nettoyage
            register_cleanup_task(elastic_client.close)
            
            return elastic_client
        else:
            logger.error("❌ Elasticsearch non initialisé (flag _initialized absent/false)")
            initialization_errors.append("Elasticsearch initialization flag not set")
            if elastic_client:
                await elastic_client.close()
                elastic_client = None
            return None
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation d'Elasticsearch: {e}")
        initialization_errors.append(f"Elasticsearch error: {e}")
        if elastic_client:
            try:
                await elastic_client.close()
            except:
                pass
            elastic_client = None
        return None


async def safe_initialize_qdrant() -> Optional[Any]:
    """Initialise Qdrant de manière sécurisée avec timeout."""
    global qdrant_client
    
    logger.info("🔍 Initialisation de Qdrant...")
    
    try:
        # Import avec gestion d'erreur
        try:
            from search_service.utils.initialization import initialize_qdrant
        except ImportError as e:
            logger.error(f"❌ Import Qdrant échoué: {e}")
            initialization_errors.append(f"Qdrant import: {e}")
            return None
        
        # Vérification de la configuration
        if not settings.QDRANT_URL:
            logger.warning("⚠️ QDRANT_URL non configurée - Qdrant ignoré")
            return None
        
        # Initialisation avec timeout
        logger.info("🔄 Initialisation du client Qdrant...")
        try:
            success, client, diagnostic = await asyncio.wait_for(
                initialize_qdrant(), 
                timeout=30.0
            )
            
            if success and client:
                logger.info("✅ Qdrant initialisé avec succès")
                logger.info(f"   Diagnostic: {diagnostic.get('connection_time', 'N/A')}s")
                
                # Enregistrer pour nettoyage
                if hasattr(client, 'close'):
                    register_cleanup_task(client.close)
                
                qdrant_client = client
                return client
            else:
                logger.error(f"❌ Qdrant non initialisé: {diagnostic.get('error', 'Unknown error')}")
                initialization_errors.append(f"Qdrant error: {diagnostic.get('error', 'Unknown')}")
                return None
                
        except asyncio.TimeoutError:
            logger.error("❌ Timeout (30s) lors de l'initialisation de Qdrant")
            initialization_errors.append("Qdrant timeout during initialization")
            return None
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation de Qdrant: {e}")
        initialization_errors.append(f"Qdrant error: {e}")
        return None


async def initialize_search_engine() -> Optional[Any]:
    """Initialise le moteur de recherche avec les clients disponibles."""
    global search_engine
    
    logger.info("🔍 Initialisation du moteur de recherche...")
    
    try:
        # Import avec gestion d'erreur
        try:
            from search_service.core.search_engine import SearchEngine
        except ImportError as e:
            logger.error(f"❌ Import SearchEngine échoué: {e}")
            initialization_errors.append(f"SearchEngine import: {e}")
            return None
        
        # Vérifier qu'au moins un client est disponible
        if not elastic_client and not qdrant_client:
            logger.error("❌ Aucun client disponible pour créer SearchEngine")
            initialization_errors.append("No clients available for SearchEngine")
            return None
        
        # Créer le moteur de recherche
        logger.info("🔄 Création du SearchEngine...")
        search_engine = SearchEngine(
            elastic_client=elastic_client,
            qdrant_client=qdrant_client
        )
        
        # Enregistrer pour nettoyage
        if hasattr(search_engine, 'close'):
            register_cleanup_task(search_engine.close)
        
        logger.info("✅ SearchEngine créé avec succès")
        logger.info(f"   Elasticsearch: {'✅' if elastic_client else '❌'}")
        logger.info(f"   Qdrant: {'✅' if qdrant_client else '❌'}")
        
        return search_engine
        
    except Exception as e:
        logger.error(f"❌ Erreur création SearchEngine: {e}")
        initialization_errors.append(f"SearchEngine error: {e}")
        return None


async def initialize_services():
    """Initialise tous les services de manière séquentielle."""
    global elastic_client, qdrant_client, search_engine
    
    logger.info("🚀 === INITIALISATION DES SERVICES ===")
    start_time = time.time()
    
    # Réinitialiser les erreurs
    initialization_errors.clear()
    
    # 1. Initialiser Elasticsearch
    elastic_client = await safe_initialize_elasticsearch()
    
    # 2. Initialiser Qdrant
    qdrant_client = await safe_initialize_qdrant()
    
    # 3. Initialiser le moteur de recherche
    search_engine = await initialize_search_engine()
    
    # 4. Configurer les routes avec les clients
    try:
        from search_service.api.routes import set_clients
        set_clients(elastic=elastic_client, qdrant=qdrant_client)
        logger.info("✅ Routes configurées avec les clients")
    except Exception as e:
        logger.error(f"❌ Erreur configuration routes: {e}")
        initialization_errors.append(f"Routes configuration: {e}")
    
    total_time = time.time() - start_time
    
    # Résumé de l'initialisation
    services_count = 0
    services_status = []
    
    if elastic_client:
        services_count += 1
        services_status.append("Elasticsearch")
    else:
        services_status.append("❌ Elasticsearch")
    
    if qdrant_client:
        services_count += 1
        services_status.append("Qdrant")
    else:
        services_status.append("❌ Qdrant")
    
    if search_engine:
        services_count += 1
        services_status.append("SearchEngine")
    else:
        services_status.append("❌ SearchEngine")
    
    logger.info(f"🎯 Initialisation terminée en {total_time:.2f}s")
    logger.info(f"📊 Services: {services_count}/3 opérationnels")
    logger.info(f"📋 Statut: {', '.join(services_status)}")
    
    if initialization_errors:
        logger.warning(f"⚠️ Erreurs d'initialisation: {len(initialization_errors)}")
        for i, error in enumerate(initialization_errors, 1):
            logger.warning(f"   {i}. {error}")
    
    # Le service peut démarrer même avec des erreurs partielles
    return services_count > 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie de l'application."""
    # Démarrage
    logger.info("🚀 Démarrage de l'application...")
    
    # Configurer les gestionnaires de signaux
    setup_signal_handlers()
    
    # Initialiser les services
    services_initialized = await initialize_services()
    
    if not services_initialized:
        logger.error("❌ Aucun service initialisé - arrêt de l'application")
        raise RuntimeError("Failed to initialize any services")
    
    logger.info("🎉 Application prête")
    
    try:
        yield
    finally:
        # Arrêt
        logger.info("🛑 Arrêt de l'application...")
        await cleanup_on_shutdown()
        
        # Attendre que l'arrêt soit terminé
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=15.0)
            logger.info("✅ Arrêt terminé")
        except asyncio.TimeoutError:
            logger.warning("⚠️ Timeout arrêt - forçage")


# Création de l'application FastAPI
app = FastAPI(
    title="Harena Search Service",
    description="Service de recherche pour transactions financières",
    version="1.0.0",
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


# Gestionnaire d'erreur global pour les connexions fermées
@app.exception_handler(RuntimeError)
async def runtime_error_handler(request, exc):
    """Gestionnaire d'erreur pour les RuntimeError (connexions fermées, etc.)."""
    error_msg = str(exc).lower()
    
    if "non initialisé" in error_msg or "fermé" in error_msg or "closed" in error_msg:
        logger.error(f"🔌 Erreur de connexion: {exc}")
        return JSONResponse(
            status_code=503,
            content={
                "error": "Service temporarily unavailable",
                "detail": "Search service connections are being reestablished",
                "retry_after": 30,
                "timestamp": time.time()
            }
        )
    
    # Autres RuntimeError
    logger.error(f"❌ RuntimeError: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": time.time()
        }
    )


# Gestionnaire d'erreur pour les timeouts
@app.exception_handler(asyncio.TimeoutError)
async def timeout_error_handler(request, exc):
    """Gestionnaire d'erreur pour les TimeoutError."""
    logger.error(f"⏰ Timeout: {exc}")
    return JSONResponse(
        status_code=504,
        content={
            "error": "Request timeout",
            "detail": "The request took too long to process",
            "timestamp": time.time()
        }
    )


# Routes de base
@app.get("/")
async def root():
    """Route racine avec informations de base."""
    uptime = time.time() - app_start_time
    
    return {
        "service": "Harena Search Service",
        "version": "1.0.0",
        "status": "running",
        "uptime": round(uptime, 2),
        "services": {
            "elasticsearch": elastic_client is not None,
            "qdrant": qdrant_client is not None,
            "search_engine": search_engine is not None
        },
        "timestamp": time.time()
    }


@app.get("/health")
async def health_check():
    """Health check détaillé."""
    uptime = time.time() - app_start_time
    
    # Vérifier l'état des services
    elasticsearch_healthy = False
    qdrant_healthy = False
    
    if elastic_client:
        try:
            elasticsearch_healthy = await asyncio.wait_for(
                elastic_client.is_healthy(), 
                timeout=5.0
            )
        except Exception as e:
            logger.warning(f"⚠️ Health check Elasticsearch échoué: {e}")
    
    if qdrant_client:
        try:
            qdrant_healthy = await asyncio.wait_for(
                qdrant_client.is_healthy() if hasattr(qdrant_client, 'is_healthy') else True,
                timeout=5.0
            )
        except Exception as e:
            logger.warning(f"⚠️ Health check Qdrant échoué: {e}")
    
    # Déterminer le statut global
    healthy_services = sum([elasticsearch_healthy, qdrant_healthy])
    total_services = sum([elastic_client is not None, qdrant_client is not None])
    
    if healthy_services == 0:
        status = "unhealthy"
        status_code = 503
    elif healthy_services < total_services:
        status = "degraded"
        status_code = 200
    else:
        status = "healthy"
        status_code = 200
    
    response_data = {
        "status": status,
        "uptime": round(uptime, 2),
        "services": {
            "elasticsearch": {
                "available": elastic_client is not None,
                "healthy": elasticsearch_healthy,
                "type": getattr(elastic_client, 'client_type', None) if elastic_client else None
            },
            "qdrant": {
                "available": qdrant_client is not None,
                "healthy": qdrant_healthy
            },
            "search_engine": {
                "available": search_engine is not None
            }
        },
        "metrics": {
            "healthy_services": healthy_services,
            "total_services": total_services,
            "initialization_errors": len(initialization_errors)
        },
        "timestamp": time.time()
    }
    
    return JSONResponse(content=response_data, status_code=status_code)


# Inclure les routes de l'API
try:
    from search_service.api.routes import router as search_router
    app.include_router(search_router, prefix="/api/v1/search", tags=["search"])
    logger.info("✅ Routes de recherche incluses")
except Exception as e:
    logger.error(f"❌ Erreur inclusion routes: {e}")


# Route de debug pour les erreurs d'initialisation
@app.get("/debug/initialization")
async def debug_initialization():
    """Endpoint de debug pour les erreurs d'initialisation."""
    return {
        "initialization_errors": initialization_errors,
        "services": {
            "elasticsearch": {
                "client": elastic_client is not None,
                "type": type(elastic_client).__name__ if elastic_client else None,
                "initialized": getattr(elastic_client, '_initialized', False) if elastic_client else False
            },
            "qdrant": {
                "client": qdrant_client is not None,
                "type": type(qdrant_client).__name__ if qdrant_client else None,
                "initialized": getattr(qdrant_client, '_initialized', False) if qdrant_client else False
            },
            "search_engine": {
                "engine": search_engine is not None,
                "type": type(search_engine).__name__ if search_engine else None
            }
        },
        "cleanup_tasks": len(cleanup_tasks),
        "timestamp": time.time()
    }


if __name__ == "__main__":
    logger.info("🚀 Démarrage du serveur Search Service...")
    
    # Configuration du serveur
    uvicorn_config = {
        "host": "0.0.0.0",
        "port": int(settings.PORT) if hasattr(settings, 'PORT') else 8000,
        "log_level": settings.LOG_LEVEL.lower(),
        "access_log": True,
        "loop": "asyncio",
        "timeout_keep_alive": 30,
        "timeout_notify": 25,
        "limit_max_requests": 1000,
        "limit_concurrency": 100
    }
    
    logger.info(f"📡 Serveur configuré sur {uvicorn_config['host']}:{uvicorn_config['port']}")
    
    try:
        uvicorn.run(app, **uvicorn_config)
    except KeyboardInterrupt:
        logger.info("🛑 Arrêt demandé par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur serveur: {e}")
    finally:
        logger.info("👋 Serveur arrêté")