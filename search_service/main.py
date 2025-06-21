"""
Service de recherche Harena - Point d'entrée principal (VERSION ROBUSTE).

Ce module configure et démarre le service de recherche hybride combinant
Elasticsearch (Bonsai) pour la recherche lexicale et Qdrant pour la recherche sémantique.

Version simplifiée et robuste pour Heroku avec gestion d'erreur améliorée.
"""

import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Configuration du logging avant les autres imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
    stream=sys.stdout
)

logger = logging.getLogger("search_service.main")

# ==================== VARIABLES GLOBALES ====================

# Clients principaux
elastic_client = None
qdrant_client = None

# Services IA
embedding_service = None
reranker_service = None

# Utilitaires
search_cache = None
metrics_collector = None

# Diagnostics
startup_diagnostics = {}
startup_time = None
initialization_errors = []

# ==================== FONCTIONS UTILITAIRES ====================

def log_startup_banner():
    """Affiche la bannière de démarrage."""
    print("=" * 100)
    print("🔍 HARENA SEARCH SERVICE - DÉMARRAGE ROBUSTE")
    print("=" * 100)
    print(f"🕐 Heure de démarrage: {datetime.now().isoformat()}")
    print(f"🐍 Python: {sys.version}")
    print("=" * 100)

def check_environment_variables():
    """Vérifie les variables d'environnement critiques."""
    logger.info("📋 Vérification des variables d'environnement...")
    
    env_status = {
        "BONSAI_URL": bool(os.environ.get("BONSAI_URL")),
        "QDRANT_URL": bool(os.environ.get("QDRANT_URL")),
        "QDRANT_API_KEY": bool(os.environ.get("QDRANT_API_KEY")),
        "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
        "COHERE_KEY": bool(os.environ.get("COHERE_KEY")),
    }
    
    for var, present in env_status.items():
        status_icon = "✅" if present else "❌"
        logger.info(f"   {status_icon} {var}: {'Configuré' if present else 'Manquant'}")
    
    critical_missing = []
    if not env_status["BONSAI_URL"]:
        critical_missing.append("BONSAI_URL")
    if not env_status["QDRANT_URL"]:
        critical_missing.append("QDRANT_URL")
    
    if critical_missing:
        logger.warning(f"⚠️ Variables critiques manquantes: {critical_missing}")
        logger.warning("💡 Le service démarrera en mode dégradé")
    
    return env_status

# ==================== INITIALISATION DES CLIENTS ====================

async def safe_initialize_elasticsearch() -> Optional[Any]:
    """Initialise Elasticsearch de manière sécurisée avec timeout."""
    global elastic_client
    
    logger.info("🔍 Initialisation d'Elasticsearch...")
    
    try:
        # Import avec gestion d'erreur
        try:
            from search_service.storage.elastic_client_hybrid import HybridElasticClient
            from config_service.config import settings
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
            await asyncio.wait_for(elastic_client.initialize(), timeout=45.0)
        except asyncio.TimeoutError:
            logger.error("❌ Timeout (45s) lors de l'initialisation d'Elasticsearch")
            initialization_errors.append("Elasticsearch timeout during initialization")
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
            
            return elastic_client
        else:
            logger.error("❌ Elasticsearch non initialisé (flag _initialized absent/false)")
            initialization_errors.append("Elasticsearch initialization flag not set")
            return None
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation d'Elasticsearch: {e}")
        initialization_errors.append(f"Elasticsearch error: {e}")
        return None

async def safe_initialize_qdrant() -> Optional[Any]:
    """Initialise Qdrant de manière sécurisée avec timeout."""
    global qdrant_client
    
    logger.info("🎯 Initialisation de Qdrant...")
    
    try:
        # Import avec gestion d'erreur
        try:
            from search_service.storage.qdrant_client import QdrantClient
            from config_service.config import settings
        except ImportError as e:
            logger.error(f"❌ Import Qdrant échoué: {e}")
            initialization_errors.append(f"Qdrant import: {e}")
            return None
        
        # Vérification de la configuration
        if not settings.QDRANT_URL:
            logger.warning("⚠️ QDRANT_URL non configurée - Qdrant ignoré")
            return None
        
        # Initialisation avec timeout strict
        logger.info("🔄 Création du client Qdrant...")
        qdrant_client = QdrantClient()
        
        logger.info("🔄 Initialisation du client Qdrant...")
        try:
            await asyncio.wait_for(qdrant_client.initialize(), timeout=45.0)
        except asyncio.TimeoutError:
            logger.error("❌ Timeout (45s) lors de l'initialisation de Qdrant")
            initialization_errors.append("Qdrant timeout during initialization")
            return None
        
        # Vérification de l'état
        if hasattr(qdrant_client, '_initialized') and qdrant_client._initialized:
            logger.info("✅ Qdrant initialisé avec succès")
            
            # Test de santé rapide
            try:
                if hasattr(qdrant_client, 'get_collections'):
                    collections = await asyncio.wait_for(qdrant_client.get_collections(), timeout=10.0)
                    logger.info(f"✅ Qdrant - Collections disponibles: {len(collections) if collections else 0}")
            except Exception as health_error:
                logger.warning(f"⚠️ Test de santé Qdrant échoué: {health_error}")
            
            return qdrant_client
        else:
            logger.error("❌ Qdrant non initialisé (flag _initialized absent/false)")
            initialization_errors.append("Qdrant initialization flag not set")
            return None
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation de Qdrant: {e}")
        initialization_errors.append(f"Qdrant error: {e}")
        return None

async def create_qdrant_collections():
    """Crée les collections Qdrant si nécessaire."""
    if not qdrant_client or not hasattr(qdrant_client, '_initialized') or not qdrant_client._initialized:
        logger.info("⚠️ Qdrant non disponible - création de collections ignorée")
        return False
    
    logger.info("🏗️ Vérification/création des collections Qdrant...")
    
    try:
        # Import de la fonction de création
        try:
            from search_service.utils.initialization import create_collections_if_needed
        except ImportError:
            logger.warning("⚠️ Fonction create_collections_if_needed non disponible")
            return False
        
        # Création avec timeout
        success = await asyncio.wait_for(
            create_collections_if_needed(qdrant_client), 
            timeout=30.0
        )
        
        if success:
            logger.info("✅ Collections Qdrant prêtes")
        else:
            logger.warning("⚠️ Problème avec les collections Qdrant")
        
        return success
        
    except asyncio.TimeoutError:
        logger.error("❌ Timeout lors de la création des collections Qdrant")
        return False
    except Exception as e:
        logger.error(f"❌ Erreur lors de la création des collections: {e}")
        return False

# ==================== INITIALISATION DES SERVICES OPTIONNELS ====================

def safe_initialize_ai_services() -> Dict[str, bool]:
    """Initialise les services IA de manière sécurisée."""
    global embedding_service, reranker_service
    
    logger.info("🤖 Initialisation des services IA...")
    
    results = {
        "embedding_service": False,
        "reranker_service": False
    }
    
    # Service d'embeddings
    try:
        from search_service.core.embeddings import EmbeddingService
        embedding_service = EmbeddingService()
        results["embedding_service"] = True
        logger.info("✅ Service d'embeddings initialisé")
    except ImportError:
        logger.info("ℹ️ Service d'embeddings non disponible (module non trouvé)")
    except Exception as e:
        logger.warning(f"⚠️ Erreur service d'embeddings: {e}")
        embedding_service = None
    
    # Service de reranking
    try:
        from search_service.core.reranker import RerankerService
        reranker_service = RerankerService()
        results["reranker_service"] = True
        logger.info("✅ Service de reranking initialisé")
    except ImportError:
        logger.info("ℹ️ Service de reranking non disponible (module non trouvé)")
    except Exception as e:
        logger.warning(f"⚠️ Erreur service de reranking: {e}")
        reranker_service = None
    
    return results

def safe_initialize_utilities() -> Dict[str, bool]:
    """Initialise les utilitaires de manière sécurisée."""
    global search_cache, metrics_collector
    
    logger.info("🛠️ Initialisation des utilitaires...")
    
    results = {
        "cache": False,
        "metrics": False
    }
    
    # Cache de recherche
    try:
        from search_service.utils.cache import SearchCache
        search_cache = SearchCache()
        results["cache"] = True
        logger.info("✅ Cache de recherche initialisé")
    except ImportError:
        logger.info("ℹ️ Cache de recherche non disponible (module non trouvé)")
    except Exception as e:
        logger.warning(f"⚠️ Erreur cache de recherche: {e}")
        search_cache = None
    
    # Collecteur de métriques (placeholder)
    try:
        # Pour l'instant, on simule le collecteur de métriques
        metrics_collector = None  # Sera implémenté plus tard
        logger.info("ℹ️ Collecteur de métriques désactivé (non implémenté)")
    except Exception as e:
        logger.warning(f"⚠️ Erreur collecteur de métriques: {e}")
        metrics_collector = None
    
    return results

# ==================== INJECTION DES DÉPENDANCES ====================

def force_inject_dependencies() -> bool:
    """Force l'injection des dépendances dans les routes."""
    logger.info("🔗 Injection forcée des dépendances...")
    
    try:
        # Import du module routes
        try:
            import search_service.api.routes as routes
            logger.info("✅ Module routes importé avec succès")
        except ImportError as e:
            logger.error(f"❌ Impossible d'importer le module routes: {e}")
            initialization_errors.append(f"Routes import failed: {e}")
            return False
        
        # Injection avec validation
        injection_results = {}
        
        # Clients principaux
        routes.elastic_client = elastic_client
        injection_results["elasticsearch"] = elastic_client is not None
        logger.info(f"{'✅' if elastic_client else '⚠️'} elastic_client injecté: {type(elastic_client).__name__ if elastic_client else 'None'}")
        
        routes.qdrant_client = qdrant_client
        injection_results["qdrant"] = qdrant_client is not None
        logger.info(f"{'✅' if qdrant_client else '⚠️'} qdrant_client injecté: {type(qdrant_client).__name__ if qdrant_client else 'None'}")
        
        # Services IA
        routes.embedding_service = embedding_service
        injection_results["embeddings"] = embedding_service is not None
        logger.info(f"{'✅' if embedding_service else 'ℹ️'} embedding_service injecté: {type(embedding_service).__name__ if embedding_service else 'None'}")
        
        routes.reranker_service = reranker_service
        injection_results["reranking"] = reranker_service is not None
        logger.info(f"{'✅' if reranker_service else 'ℹ️'} reranker_service injecté: {type(reranker_service).__name__ if reranker_service else 'None'}")
        
        # Utilitaires
        routes.search_cache = search_cache
        injection_results["cache"] = search_cache is not None
        logger.info(f"{'✅' if search_cache else 'ℹ️'} search_cache injecté: {type(search_cache).__name__ if search_cache else 'None'}")
        
        routes.metrics_collector = metrics_collector
        injection_results["metrics"] = metrics_collector is not None
        logger.info(f"{'ℹ️'} metrics_collector injecté: {type(metrics_collector).__name__ if metrics_collector else 'None'}")
        
        # Vérification post-injection
        verification_passed = True
        required_attrs = ['elastic_client', 'qdrant_client', 'embedding_service', 'reranker_service', 'search_cache', 'metrics_collector']
        
        for attr in required_attrs:
            if not hasattr(routes, attr):
                logger.error(f"❌ Attribut {attr} manquant après injection")
                verification_passed = False
            else:
                logger.debug(f"✅ Attribut {attr} présent")
        
        # Compter les services critiques injectés
        critical_services = sum([
            injection_results.get("elasticsearch", False),
            injection_results.get("qdrant", False)
        ])
        
        total_services = sum(injection_results.values())
        
        logger.info(f"📊 Résumé injection: {total_services}/6 services injectés ({critical_services}/2 critiques)")
        
        if critical_services == 0:
            logger.error("❌ Aucun service critique injecté - fonctionnalité limitée")
            return False
        elif critical_services == 1:
            logger.warning("⚠️ Un seul service critique injecté - mode dégradé")
            return True
        else:
            logger.info("✅ Tous les services critiques injectés - mode optimal")
            return True
            
    except Exception as e:
        logger.error(f"💥 Erreur critique lors de l'injection: {e}")
        initialization_errors.append(f"Injection failed: {e}")
        
        # Tentative d'injection de fallback
        try:
            import search_service.api.routes as routes
            routes.elastic_client = None
            routes.qdrant_client = None
            routes.embedding_service = None
            routes.reranker_service = None
            routes.search_cache = None
            routes.metrics_collector = None
            logger.warning("⚠️ Injection de fallback (None) effectuée")
        except Exception as fallback_error:
            logger.error(f"💥 Impossible d'effectuer l'injection de fallback: {fallback_error}")
        
        return False

# ==================== CYCLE DE VIE DE L'APPLICATION ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie de l'application."""
    global startup_time, startup_diagnostics
    
    try:
        # === PHASE DE DÉMARRAGE ===
        log_startup_banner()
        startup_time = time.time()
        
        logger.info("🚀 Début de l'initialisation robuste...")
        
        # 1. Vérification de l'environnement
        env_status = check_environment_variables()
        
        # 2. Initialisation parallèle des clients avec timeout global
        logger.info("🔄 Initialisation des clients en parallèle...")
        
        try:
            # Lancer les deux initializations en parallèle avec timeout global
            elasticsearch_task = asyncio.create_task(safe_initialize_elasticsearch())
            qdrant_task = asyncio.create_task(safe_initialize_qdrant())
            
            # Attendre les deux avec timeout global
            done, pending = await asyncio.wait_for(
                asyncio.wait([elasticsearch_task, qdrant_task], return_when=asyncio.ALL_COMPLETED),
                timeout=90.0  # Timeout global de 90 secondes
            )
            
            # Récupérer les résultats
            elasticsearch_result = elasticsearch_task.result() if elasticsearch_task.done() else None
            qdrant_result = qdrant_task.result() if qdrant_task.done() else None
            
        except asyncio.TimeoutError:
            logger.error("❌ Timeout global (90s) lors de l'initialisation des clients")
            initialization_errors.append("Global client initialization timeout")
            elasticsearch_result = None
            qdrant_result = None
        
        # Assigner les résultats aux variables globales
        global elastic_client, qdrant_client
        elastic_client = elasticsearch_result
        qdrant_client = qdrant_result
        
        # 3. Vérifier qu'au moins un service fonctionne
        services_ready = sum([
            elastic_client is not None,
            qdrant_client is not None
        ])
        
        if services_ready == 0:
            logger.error("🚨 AUCUN service de recherche disponible")
            logger.error("💡 Le service démarrera quand même pour permettre le debugging")
        elif services_ready == 1:
            logger.warning("⚠️ Un seul service de recherche disponible - mode dégradé")
        else:
            logger.info("🎉 Tous les services de recherche disponibles - mode optimal")
        
        # 4. Création des collections Qdrant si disponible
        collections_ready = False
        if qdrant_client:
            collections_ready = await create_qdrant_collections()
        
        # 5. Initialisation des services optionnels
        ai_services_status = safe_initialize_ai_services()
        utilities_status = safe_initialize_utilities()
        
        # 6. Injection des dépendances
        injection_success = force_inject_dependencies()
        
        # 7. Calcul du temps d'initialisation
        initialization_time = time.time() - startup_time
        
        # 8. Résumé final
        startup_diagnostics = {
            "startup_time": startup_time,
            "initialization_time": initialization_time,
            "environment": env_status,
            "services": {
                "elasticsearch": elastic_client is not None,
                "qdrant": qdrant_client is not None,
                "collections_ready": collections_ready,
                "total_ready": services_ready
            },
            "ai_services": ai_services_status,
            "utilities": utilities_status,
            "injection_success": injection_success,
            "initialization_errors": initialization_errors,
            "status": "SUCCESS" if services_ready > 0 and injection_success else "DEGRADED"
        }
        
        # Log du résumé final
        logger.info("=" * 60)
        logger.info("📊 RÉSUMÉ DE L'INITIALISATION")
        logger.info("=" * 60)
        logger.info(f"⏱️ Temps d'initialisation: {initialization_time:.2f}s")
        logger.info(f"🔍 Elasticsearch: {'✅ Prêt' if elastic_client else '❌ Indisponible'}")
        logger.info(f"🎯 Qdrant: {'✅ Prêt' if qdrant_client else '❌ Indisponible'}")
        logger.info(f"🏗️ Collections: {'✅ Prêtes' if collections_ready else '⚠️ Non créées'}")
        logger.info(f"🤖 Services IA: {sum(ai_services_status.values())}/2")
        logger.info(f"🛠️ Utilitaires: {sum(utilities_status.values())}/2")
        logger.info(f"🔗 Injection: {'✅ Réussie' if injection_success else '❌ Échouée'}")
        logger.info(f"🎯 Statut global: {startup_diagnostics['status']}")
        
        if initialization_errors:
            logger.warning("⚠️ Erreurs d'initialisation:")
            for error in initialization_errors:
                logger.warning(f"   - {error}")
        
        logger.info("=" * 60)
        logger.info("🎉 Service de recherche Harena démarré")
        logger.info("=" * 60)
        
        # === PHASE D'EXÉCUTION ===
        yield
        
    except Exception as e:
        logger.error(f"💥 Erreur critique durant l'initialisation: {e}", exc_info=True)
        initialization_errors.append(f"Critical startup error: {e}")
        
        # Créer un diagnostic d'urgence
        startup_diagnostics = {
            "status": "FAILED",
            "error": str(e),
            "initialization_time": time.time() - startup_time if startup_time else 0,
            "initialization_errors": initialization_errors
        }
        
        # Continuer le démarrage malgré l'erreur pour permettre le debugging
        yield
    
    finally:
        # === PHASE D'ARRÊT ===
        logger.info("🔄 Arrêt du service de recherche...")
        
        # Liste des tâches de nettoyage
        cleanup_tasks = []
        
        if elastic_client and hasattr(elastic_client, 'close'):
            cleanup_tasks.append(("Elasticsearch", elastic_client.close()))
        
        if qdrant_client and hasattr(qdrant_client, 'close'):
            cleanup_tasks.append(("Qdrant", qdrant_client.close()))
        
        # Exécution du nettoyage avec timeout individuel
        for service_name, cleanup_coro in cleanup_tasks:
            try:
                await asyncio.wait_for(cleanup_coro, timeout=10.0)
                logger.info(f"✅ {service_name} fermé proprement")
            except asyncio.TimeoutError:
                logger.warning(f"⚠️ Timeout lors de la fermeture de {service_name}")
            except Exception as e:
                logger.warning(f"⚠️ Erreur fermeture {service_name}: {e}")
        
        logger.info("🏁 Service de recherche arrêté proprement")

# ==================== CRÉATION DE L'APPLICATION ====================

def create_app() -> FastAPI:
    """Crée l'application FastAPI avec la configuration robuste."""
    
    logger.info("🏗️ Création de l'application FastAPI...")
    
    # Création de l'app avec le cycle de vie robuste
    app = FastAPI(
        title="Harena Search Service",
        description="Service de recherche hybride pour Harena Finance (Version Robuste)",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Configuration CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # ==================== ROUTES ESSENTIELLES ====================
    
    @app.get("/health")
    async def health_check():
        """Check de santé complet."""
        current_time = time.time()
        uptime = current_time - startup_time if startup_time else 0
        
        # Statut des services
        services_status = {
            "elasticsearch": elastic_client is not None and hasattr(elastic_client, '_initialized') and elastic_client._initialized,
            "qdrant": qdrant_client is not None and hasattr(qdrant_client, '_initialized') and qdrant_client._initialized,
            "embedding_service": embedding_service is not None,
            "search_cache": search_cache is not None
        }
        
        healthy_services = sum(services_status.values())
        core_services = sum([services_status["elasticsearch"], services_status["qdrant"]])
        
        overall_status = "healthy" if core_services > 0 else "degraded"
        
        return {
            "status": overall_status,
            "service": "search_service",
            "timestamp": current_time,
            "uptime_seconds": uptime,
            "services": {
                "total_healthy": healthy_services,
                "core_services": core_services,
                "details": services_status
            },
            "initialization": {
                "status": startup_diagnostics.get("status", "UNKNOWN"),
                "initialization_time": startup_diagnostics.get("initialization_time", 0),
                "errors_count": len(initialization_errors)
            }
        }
    
    @app.get("/diagnostics")
    async def full_diagnostics():
        """Diagnostics complets du service."""
        return {
            "service": "search_service",
            "timestamp": time.time(),
            "startup_diagnostics": startup_diagnostics,
            "initialization_errors": initialization_errors,
            "runtime_info": {
                "uptime_seconds": time.time() - startup_time if startup_time else 0,
                "clients_status": {
                    "elasticsearch": {
                        "available": elastic_client is not None,
                        "initialized": elastic_client is not None and hasattr(elastic_client, '_initialized') and elastic_client._initialized,
                        "type": type(elastic_client).__name__ if elastic_client else None
                    },
                    "qdrant": {
                        "available": qdrant_client is not None,
                        "initialized": qdrant_client is not None and hasattr(qdrant_client, '_initialized') and qdrant_client._initialized,
                        "type": type(qdrant_client).__name__ if qdrant_client else None
                    }
                }
            }
        }
    
    @app.get("/")
    async def root():
        """Endpoint racine avec informations de base."""
        return {
            "service": "Harena Search Service",
            "version": "1.0.0",
            "status": "online",
            "documentation": "/docs",
            "health_check": "/health",
            "diagnostics": "/diagnostics"
        }
    
    # ==================== ROUTES PRINCIPALES ====================
    
    # Tentative d'ajout des routes principales
    try:
        from search_service.api.routes import router
        app.include_router(router, prefix="/api/v1")
        logger.info("✅ Routes principales ajoutées avec succès")
        
        # Note: Routes WebSocket non implémentées pour le moment
        logger.info("ℹ️ Routes WebSocket non implémentées")
        
    except Exception as e:
        logger.error(f"❌ Impossible d'ajouter les routes principales: {e}")
        initialization_errors.append(f"Routes loading failed: {e}")
        
        # Route de diagnostic d'urgence
        @app.get("/emergency-diagnostics")
        async def emergency_diagnostics():
            return {
                "error": "Routes principales indisponibles",
                "reason": str(e),
                "fallback_mode": True,
                "startup_diagnostics": startup_diagnostics,
                "initialization_errors": initialization_errors,
                "available_endpoints": [
                    "GET / - Informations de base",
                    "GET /health - Check de santé",
                    "GET /diagnostics - Diagnostics complets",
                    "GET /emergency-diagnostics - Ce diagnostic d'urgence"
                ]
            }
    
    logger.info("✅ Application FastAPI créée avec succès")
    return app

# ==================== POINT D'ENTRÉE ====================

# Instance de l'application
app = create_app()

# Point d'entrée pour le développement local
if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Démarrage en mode développement local...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        log_level="info",
        reload=False  # Désactivé pour éviter les conflits avec le cycle de vie
    )