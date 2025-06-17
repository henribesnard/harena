"""
Module principal du service de recherche avec diagnostic complet au démarrage.

Ce module initialise et configure le service de recherche hybride de Harena,
avec vérifications détaillées des services externes (Bonsai Elasticsearch + Qdrant).
"""
import logging
import time
import sys
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from search_service.api.routes import router
from search_service.core.embeddings import embedding_service
from search_service.core.reranker import reranker_service
from search_service.storage.elastic_client import ElasticClient
from search_service.storage.qdrant_client import QdrantClient
from search_service.utils.cache import SearchCache
from search_service.utils.metrics import MetricsCollector
from config_service.config import settings

# Configuration du logging avec format détaillé
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("search_service.main")

# Réduire le bruit des bibliothèques externes
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ==================== VARIABLES GLOBALES ====================

# Instances globales des services
elastic_client = None
qdrant_client = None
search_cache = None
metrics_collector = None
startup_time = None
startup_diagnostics = {}

# ==================== FONCTIONS DE DIAGNOSTIC ====================

def log_startup_banner():
    """Affiche la bannière de démarrage."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        🔍 HARENA SEARCH SERVICE 🔍                          ║
║                     Service de Recherche Hybride v1.0.0                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)
    logger.info("🚀 === DÉMARRAGE DU SERVICE DE RECHERCHE HYBRIDE ===")

def check_environment_configuration() -> Dict[str, Any]:
    """Vérifie et diagnostique la configuration des variables d'environnement."""
    logger.info("🔧 === VÉRIFICATION DE LA CONFIGURATION ===")
    
    config_status = {
        "critical_services": {},
        "optional_services": {},
        "summary": {}
    }
    
    # Services critiques pour le fonctionnement
    critical_configs = {
        "BONSAI_URL": {
            "description": "Elasticsearch via Bonsai (recherche lexicale)",
            "value": settings.BONSAI_URL,
            "required": True
        },
        "QDRANT_URL": {
            "description": "Qdrant (recherche sémantique)",
            "value": settings.QDRANT_URL,
            "required": True
        }
    }
    
    # Services optionnels mais recommandés
    optional_configs = {
        "OPENAI_API_KEY": {
            "description": "OpenAI (génération d'embeddings)",
            "value": settings.OPENAI_API_KEY,
            "impact": "Pas d'embeddings automatiques"
        },
        "COHERE_KEY": {
            "description": "Cohere (reranking des résultats)",
            "value": settings.COHERE_KEY,
            "impact": "Pas de reranking intelligent"
        },
        "QDRANT_API_KEY": {
            "description": "Qdrant API Key (authentification)",
            "value": settings.QDRANT_API_KEY,
            "impact": "Connexion sans authentification"
        }
    }
    
    # Vérification des services critiques
    critical_ok_count = 0
    for key, info in critical_configs.items():
        is_configured = bool(info["value"])
        config_status["critical_services"][key] = {
            "configured": is_configured,
            "description": info["description"],
            "required": info["required"]
        }
        
        if is_configured:
            critical_ok_count += 1
            # Masquer les URLs sensibles pour l'affichage
            if "URL" in key:
                safe_value = info["value"].split('@')[-1] if '@' in info["value"] else info["value"]
                logger.info(f"✅ {key}: {safe_value}")
            else:
                logger.info(f"✅ {key}: Configuré")
        else:
            logger.error(f"❌ {key}: NON CONFIGURÉ - {info['description']}")
    
    # Vérification des services optionnels
    optional_ok_count = 0
    for key, info in optional_configs.items():
        is_configured = bool(info["value"])
        config_status["optional_services"][key] = {
            "configured": is_configured,
            "description": info["description"],
            "impact": info["impact"]
        }
        
        if is_configured:
            optional_ok_count += 1
            if "KEY" in key:
                masked_value = f"{info['value'][:8]}...{info['value'][-4:]}" if len(info["value"]) > 12 else "***"
                logger.info(f"✅ {key}: {masked_value}")
            else:
                logger.info(f"✅ {key}: Configuré")
        else:
            logger.warning(f"⚠️ {key}: Non configuré - {info['impact']}")
    
    # Résumé de la configuration
    config_status["summary"] = {
        "critical_configured": critical_ok_count,
        "critical_total": len(critical_configs),
        "optional_configured": optional_ok_count,
        "optional_total": len(optional_configs),
        "critical_percentage": (critical_ok_count / len(critical_configs)) * 100,
        "optional_percentage": (optional_ok_count / len(optional_configs)) * 100
    }
    
    if critical_ok_count == len(critical_configs):
        logger.info("🎉 CONFIGURATION: Tous les services critiques sont configurés")
    elif critical_ok_count > 0:
        logger.warning("⚠️ CONFIGURATION: Services critiques PARTIELLEMENT configurés")
    else:
        logger.error("🚨 CONFIGURATION: AUCUN service critique configuré")
    
    return config_status

async def initialize_elasticsearch() -> tuple[bool, Optional[ElasticClient], Dict[str, Any]]:
    """Initialise et teste la connexion Elasticsearch (Bonsai)."""
    logger.info("🔍 === INITIALISATION ELASTICSEARCH (BONSAI) ===")
    diagnostic = {
        "service": "elasticsearch_bonsai",
        "configured": False,
        "connected": False,
        "healthy": False,
        "error": None,
        "connection_time": None,
        "cluster_info": {},
        "indices_info": {}
    }
    
    if not settings.BONSAI_URL:
        logger.error("❌ BONSAI_URL non configurée")
        diagnostic["error"] = "BONSAI_URL not configured"
        return False, None, diagnostic
    
    diagnostic["configured"] = True
    
    # Masquer les credentials pour l'affichage
    safe_url = settings.BONSAI_URL.split('@')[-1] if '@' in settings.BONSAI_URL else settings.BONSAI_URL
    logger.info(f"🔗 Connexion à Bonsai Elasticsearch: {safe_url}")
    
    try:
        start_time = time.time()
        client = ElasticClient()
        
        logger.info("⏱️ Test de connexion Elasticsearch...")
        await client.initialize()
        
        connection_time = time.time() - start_time
        diagnostic["connection_time"] = round(connection_time, 3)
        diagnostic["connected"] = True
        
        logger.info(f"✅ Connexion établie en {connection_time:.3f}s")
        
        # Test de santé
        logger.info("🩺 Vérification de la santé du cluster...")
        is_healthy = await client.is_healthy()
        diagnostic["healthy"] = is_healthy
        
        if is_healthy:
            logger.info("🟢 Cluster Elasticsearch en bonne santé")
            
            # Récupérer les informations du cluster
            try:
                cluster_info = await client.get_cluster_info()
                diagnostic["cluster_info"] = cluster_info
                logger.info(f"📊 Cluster: {cluster_info.get('cluster_name', 'Unknown')}")
                logger.info(f"📊 Version: {cluster_info.get('version', {}).get('number', 'Unknown')}")
                logger.info(f"📊 Nœuds: {cluster_info.get('nodes', 'Unknown')}")
            except Exception as e:
                logger.warning(f"⚠️ Impossible de récupérer les infos cluster: {e}")
            
            # Vérifier les indices
            try:
                indices_info = await client.get_indices_info()
                diagnostic["indices_info"] = indices_info
                if indices_info:
                    logger.info(f"📁 Indices disponibles: {len(indices_info)}")
                    for index_name, info in indices_info.items():
                        logger.info(f"   - {index_name}: {info.get('docs', {}).get('count', 0)} documents")
                else:
                    logger.info("📁 Aucun indice trouvé")
            except Exception as e:
                logger.warning(f"⚠️ Impossible de lister les indices: {e}")
            
            return True, client, diagnostic
        else:
            logger.error("🔴 Cluster Elasticsearch non opérationnel")
            diagnostic["error"] = "Cluster unhealthy"
            return False, client, diagnostic
            
    except Exception as e:
        connection_time = time.time() - start_time
        diagnostic["connection_time"] = round(connection_time, 3)
        diagnostic["error"] = str(e)
        
        logger.error(f"💥 Erreur Elasticsearch après {connection_time:.3f}s:")
        logger.error(f"   Type: {type(e).__name__}")
        logger.error(f"   Message: {str(e)}")
        
        # Diagnostic spécifique des erreurs
        error_str = str(e).lower()
        if "connection" in error_str or "timeout" in error_str:
            logger.error("🔌 DIAGNOSTIC: Problème de connectivité réseau")
            logger.error("   - Vérifiez l'URL Bonsai")
            logger.error("   - Vérifiez la connectivité réseau")
        elif "401" in str(e) or "403" in str(e) or "auth" in error_str:
            logger.error("🔑 DIAGNOSTIC: Problème d'authentification")
            logger.error("   - Vérifiez les credentials dans BONSAI_URL")
        elif "ssl" in error_str or "certificate" in error_str:
            logger.error("🔒 DIAGNOSTIC: Problème SSL/TLS")
            logger.error("   - Vérifiez les certificats SSL")
        
        return False, None, diagnostic

async def initialize_qdrant() -> tuple[bool, Optional[QdrantClient], Dict[str, Any]]:
    """Initialise et teste la connexion Qdrant."""
    logger.info("🎯 === INITIALISATION QDRANT ===")
    diagnostic = {
        "service": "qdrant",
        "configured": False,
        "connected": False,
        "healthy": False,
        "error": None,
        "connection_time": None,
        "collections_info": {},
        "version_info": {}
    }
    
    if not settings.QDRANT_URL:
        logger.error("❌ QDRANT_URL non configurée")
        diagnostic["error"] = "QDRANT_URL not configured"
        return False, None, diagnostic
    
    diagnostic["configured"] = True
    logger.info(f"🔗 Connexion à Qdrant: {settings.QDRANT_URL}")
    
    if settings.QDRANT_API_KEY:
        logger.info("🔑 Authentification par API Key activée")
    else:
        logger.info("🔓 Connexion sans authentification")
    
    try:
        start_time = time.time()
        client = QdrantClient()
        
        logger.info("⏱️ Test de connexion Qdrant...")
        await client.initialize()
        
        connection_time = time.time() - start_time
        diagnostic["connection_time"] = round(connection_time, 3)
        diagnostic["connected"] = True
        
        logger.info(f"✅ Connexion établie en {connection_time:.3f}s")
        
        # Test de santé
        logger.info("🩺 Vérification de la santé de Qdrant...")
        is_healthy = await client.is_healthy()
        diagnostic["healthy"] = is_healthy
        
        if is_healthy:
            logger.info("🟢 Service Qdrant en bonne santé")
            
            # Récupérer les informations des collections
            try:
                collections_info = await client.get_collections_info()
                diagnostic["collections_info"] = collections_info
                if collections_info:
                    logger.info(f"📊 Collections disponibles: {len(collections_info)}")
                    for collection_name, info in collections_info.items():
                        points_count = info.get('points_count', 0)
                        vectors_count = info.get('vectors_count', 0)
                        logger.info(f"   - {collection_name}: {points_count} points, {vectors_count} vecteurs")
                else:
                    logger.warning("📊 Aucune collection trouvée")
            except Exception as e:
                logger.warning(f"⚠️ Impossible de récupérer les infos collections: {e}")
            
            # Vérifier la collection des transactions
            try:
                collection_exists = await client.collection_exists("financial_transactions")
                if collection_exists:
                    logger.info("✅ Collection 'financial_transactions' trouvée")
                else:
                    logger.warning("⚠️ Collection 'financial_transactions' non trouvée")
                    logger.warning("   - Lancez l'enrichment_service pour créer la collection")
            except Exception as e:
                logger.warning(f"⚠️ Impossible de vérifier la collection: {e}")
            
            return True, client, diagnostic
        else:
            logger.error("🔴 Service Qdrant non opérationnel")
            diagnostic["error"] = "Service unhealthy"
            return False, client, diagnostic
            
    except Exception as e:
        connection_time = time.time() - start_time
        diagnostic["connection_time"] = round(connection_time, 3)
        diagnostic["error"] = str(e)
        
        logger.error(f"💥 Erreur Qdrant après {connection_time:.3f}s:")
        logger.error(f"   Type: {type(e).__name__}")
        logger.error(f"   Message: {str(e)}")
        
        # Diagnostic spécifique des erreurs
        error_str = str(e).lower()
        if "connection" in error_str or "timeout" in error_str:
            logger.error("🔌 DIAGNOSTIC: Problème de connectivité réseau")
            logger.error("   - Vérifiez l'URL Qdrant")
            logger.error("   - Vérifiez la connectivité réseau")
        elif "401" in str(e) or "403" in str(e) or "auth" in error_str:
            logger.error("🔑 DIAGNOSTIC: Problème d'authentification")
            logger.error("   - Vérifiez QDRANT_API_KEY")
        elif "404" in str(e):
            logger.error("📂 DIAGNOSTIC: Collection non trouvée")
            logger.error("   - La collection 'financial_transactions' doit être créée")
            logger.error("   - Lancez d'abord l'enrichment_service")
        
        return False, None, diagnostic

async def initialize_ai_services() -> Dict[str, Dict[str, Any]]:
    """Initialise les services IA (embeddings et reranking)."""
    logger.info("🤖 === INITIALISATION DES SERVICES IA ===")
    ai_diagnostics = {}
    
    # Service d'embeddings (OpenAI)
    embedding_diagnostic = {
        "service": "openai_embeddings",
        "configured": bool(settings.OPENAI_API_KEY),
        "initialized": False,
        "error": None,
        "init_time": None
    }
    
    if settings.OPENAI_API_KEY:
        try:
            logger.info("🤖 Initialisation du service d'embeddings OpenAI...")
            start_time = time.time()
            await embedding_service.initialize()
            init_time = time.time() - start_time
            
            embedding_diagnostic["initialized"] = True
            embedding_diagnostic["init_time"] = round(init_time, 3)
            logger.info(f"✅ Service d'embeddings initialisé en {init_time:.3f}s")
        except Exception as e:
            embedding_diagnostic["error"] = str(e)
            logger.error(f"❌ Échec initialisation embeddings: {type(e).__name__}: {str(e)}")
            logger.error("📍 Vérifiez OPENAI_API_KEY et la connectivité internet")
    else:
        logger.warning("⚠️ OPENAI_API_KEY non configurée - embeddings indisponibles")
    
    ai_diagnostics["embeddings"] = embedding_diagnostic
    
    # Service de reranking (Cohere)
    reranking_diagnostic = {
        "service": "cohere_reranking",
        "configured": bool(settings.COHERE_KEY),
        "initialized": False,
        "error": None,
        "init_time": None
    }
    
    if settings.COHERE_KEY:
        try:
            logger.info("🎯 Initialisation du service de reranking Cohere...")
            start_time = time.time()
            await reranker_service.initialize()
            init_time = time.time() - start_time
            
            reranking_diagnostic["initialized"] = True
            reranking_diagnostic["init_time"] = round(init_time, 3)
            logger.info(f"✅ Service de reranking initialisé en {init_time:.3f}s")
        except Exception as e:
            reranking_diagnostic["error"] = str(e)
            logger.error(f"❌ Échec initialisation reranking: {type(e).__name__}: {str(e)}")
            logger.error("📍 Vérifiez COHERE_KEY et la connectivité internet")
    else:
        logger.warning("⚠️ COHERE_KEY non configurée - reranking indisponible")
    
    ai_diagnostics["reranking"] = reranking_diagnostic
    
    return ai_diagnostics

def initialize_cache_and_metrics() -> Dict[str, Dict[str, Any]]:
    """Initialise le cache et les métriques."""
    logger.info("🗃️ === INITIALISATION CACHE ET MÉTRIQUES ===")
    utils_diagnostics = {}
    
    # Cache de recherche
    cache_diagnostic = {
        "service": "search_cache",
        "initialized": False,
        "error": None
    }
    
    global search_cache
    try:
        search_cache = SearchCache()
        cache_diagnostic["initialized"] = True
        logger.info("✅ Cache de recherche initialisé")
    except Exception as e:
        cache_diagnostic["error"] = str(e)
        logger.error(f"❌ Erreur initialisation cache: {e}")
        search_cache = None
    
    utils_diagnostics["cache"] = cache_diagnostic
    
    # Collecteur de métriques
    metrics_diagnostic = {
        "service": "metrics_collector",
        "initialized": False,
        "error": None
    }
    
    global metrics_collector
    try:
        metrics_collector = MetricsCollector()
        metrics_diagnostic["initialized"] = True
        logger.info("✅ Collecteur de métriques initialisé")
    except Exception as e:
        metrics_diagnostic["error"] = str(e)
        logger.error(f"❌ Erreur initialisation métriques: {e}")
        metrics_collector = None
    
    utils_diagnostics["metrics"] = metrics_diagnostic
    
    return utils_diagnostics

def inject_dependencies():
    """Injecte les dépendances dans le module routes."""
    logger.info("🔗 === INJECTION DES DÉPENDANCES ===")
    
    import search_service.api.routes as routes
    routes.elastic_client = elastic_client
    routes.qdrant_client = qdrant_client
    routes.search_cache = search_cache
    routes.metrics_collector = metrics_collector
    
    logger.info("✅ Dépendances injectées dans les routes")
    logger.info(f"   🔍 Elasticsearch client: {'✅ Injecté' if elastic_client else '❌ None'}")
    logger.info(f"   🎯 Qdrant client: {'✅ Injecté' if qdrant_client else '❌ None'}")
    logger.info(f"   🗃️ Search cache: {'✅ Injecté' if search_cache else '❌ None'}")
    logger.info(f"   📊 Metrics collector: {'✅ Injecté' if metrics_collector else '❌ None'}")

def generate_startup_summary() -> Dict[str, Any]:
    """Génère un résumé complet du démarrage."""
    global startup_diagnostics
    
    startup_duration = time.time() - startup_time
    
    # Compter les services opérationnels
    operational_services = []
    failed_services = []
    
    # Services critiques
    if elastic_client:
        operational_services.append("Elasticsearch (Bonsai)")
    else:
        failed_services.append("Elasticsearch (Bonsai)")
    
    if qdrant_client:
        operational_services.append("Qdrant")
    else:
        failed_services.append("Qdrant")
    
    # Services IA
    ai_diag = startup_diagnostics.get("ai_services", {})
    if ai_diag.get("embeddings", {}).get("initialized"):
        operational_services.append("OpenAI Embeddings")
    else:
        failed_services.append("OpenAI Embeddings")
    
    if ai_diag.get("reranking", {}).get("initialized"):
        operational_services.append("Cohere Reranking")
    else:
        failed_services.append("Cohere Reranking")
    
    # Services utilitaires
    if search_cache:
        operational_services.append("Cache")
    if metrics_collector:
        operational_services.append("Métriques")
    
    # Déterminer l'état global du service
    critical_services_ok = elastic_client and qdrant_client
    
    if critical_services_ok:
        if len(operational_services) >= 4:  # Elasticsearch + Qdrant + au moins 2 autres
            status = "FULLY_OPERATIONAL"
            status_icon = "🎉"
            status_msg = "COMPLÈTEMENT OPÉRATIONNEL"
        else:
            status = "OPERATIONAL"
            status_icon = "✅"
            status_msg = "OPÉRATIONNEL"
    elif elastic_client or qdrant_client:
        status = "DEGRADED"
        status_icon = "⚠️"
        status_msg = "OPÉRATIONNEL DÉGRADÉ"
    else:
        status = "FAILED"
        status_icon = "🚨"
        status_msg = "NON OPÉRATIONNEL"
    
    summary = {
        "status": status,
        "status_icon": status_icon,
        "status_message": status_msg,
        "startup_duration": round(startup_duration, 2),
        "operational_services": operational_services,
        "failed_services": failed_services,
        "operational_count": len(operational_services),
        "total_services": len(operational_services) + len(failed_services),
        "critical_services_operational": critical_services_ok,
        "search_capabilities": {
            "lexical_search": bool(elastic_client),
            "semantic_search": bool(qdrant_client),
            "hybrid_search": bool(elastic_client and qdrant_client),
            "ai_reranking": ai_diag.get("reranking", {}).get("initialized", False)
        }
    }
    
    # Affichage du résumé
    logger.info("=" * 100)
    logger.info(f"{status_icon} SERVICE DE RECHERCHE HYBRIDE - ÉTAT: {status_msg}")
    logger.info(f"⏱️ Temps de démarrage: {startup_duration:.2f}s")
    logger.info(f"🎯 Services opérationnels ({len(operational_services)}/{len(operational_services) + len(failed_services)}): {', '.join(operational_services) if operational_services else 'Aucun'}")
    
    if failed_services:
        logger.warning(f"❌ Services échoués: {', '.join(failed_services)}")
    
    # Messages informatifs selon l'état
    if status == "FULLY_OPERATIONAL":
        logger.info("🚀 Toutes les fonctionnalités de recherche avancée sont disponibles")
        logger.info("   ✓ Recherche lexicale (Elasticsearch/Bonsai)")
        logger.info("   ✓ Recherche sémantique (Qdrant)")
        logger.info("   ✓ Recherche hybride avec fusion des scores")
        logger.info("   ✓ Reranking intelligent des résultats")
    elif status == "OPERATIONAL":
        logger.info("✅ Fonctionnalités de recherche de base disponibles")
        logger.info("   ✓ Recherche lexicale ET sémantique")
        logger.warning("   ⚠️ Certaines fonctionnalités avancées peuvent être limitées")
    elif status == "DEGRADED":
        logger.warning("⚠️ Service de recherche en mode dégradé")
        if elastic_client and not qdrant_client:
            logger.warning("   ✓ Recherche lexicale disponible")
            logger.warning("   ❌ Recherche sémantique indisponible")
        elif qdrant_client and not elastic_client:
            logger.warning("   ❌ Recherche lexicale indisponible")
            logger.warning("   ✓ Recherche sémantique disponible")
    else:
        logger.error("🚨 Service de recherche indisponible")
        logger.error("   ❌ Vérifiez la configuration BONSAI_URL et QDRANT_URL")
        logger.error("   ❌ Vérifiez la connectivité réseau")
    
    logger.info("📊 Endpoints disponibles:")
    logger.info("   GET  /health - Vérification de santé détaillée")
    logger.info("   POST /api/v1/search - Recherche de transactions")
    logger.info("   GET  /api/v1/search/suggest - Suggestions de recherche")
    logger.info("=" * 100)
    
    return summary

# ==================== CYCLE DE VIE DE L'APPLICATION ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie de l'application avec diagnostics complets."""
    global startup_time, startup_diagnostics, elastic_client, qdrant_client
    
    try:
        # === PHASE DE DÉMARRAGE ===
        log_startup_banner()
        startup_time = time.time()
        
        # 1. Vérification de la configuration
        config_status = check_environment_configuration()
        startup_diagnostics["configuration"] = config_status
        
        # 2. Initialisation d'Elasticsearch (Bonsai)
        elastic_success, elastic_client, elastic_diag = await initialize_elasticsearch()
        startup_diagnostics["elasticsearch"] = elastic_diag
        
        # 3. Initialisation de Qdrant
        qdrant_success, qdrant_client, qdrant_diag = await initialize_qdrant()
        startup_diagnostics["qdrant"] = qdrant_diag
        
        # 4. Initialisation des services IA
        ai_diagnostics = await initialize_ai_services()
        startup_diagnostics["ai_services"] = ai_diagnostics
        
        # 5. Initialisation du cache et des métriques
        utils_diagnostics = initialize_cache_and_metrics()
        startup_diagnostics["utilities"] = utils_diagnostics
        
        # 6. Injection des dépendances APRÈS initialisation
        inject_dependencies()
        
        # 7. Génération du résumé de démarrage
        startup_summary = generate_startup_summary()
        startup_diagnostics["summary"] = startup_summary
        
        # Application prête
        yield
        
    except Exception as e:
        logger.critical(f"💥 ERREUR CRITIQUE au démarrage: {type(e).__name__}: {str(e)}")
        import traceback
        logger.critical(f"Stack trace:\n{traceback.format_exc()}")
        raise
    
    finally:
        # === PHASE D'ARRÊT ===
        logger.info("⏹️ Arrêt du service de recherche...")
        
        # Fermeture des services IA
        try:
            if hasattr(embedding_service, 'close'):
                await embedding_service.close()
                logger.info("✅ Service d'embeddings fermé")
        except Exception as e:
            logger.error(f"❌ Erreur fermeture embeddings: {e}")
        
        try:
            if hasattr(reranker_service, 'close'):
                await reranker_service.close()
                logger.info("✅ Service de reranking fermé")
        except Exception as e:
            logger.error(f"❌ Erreur fermeture reranking: {e}")
        
        logger.info("🔚 Service de recherche arrêté proprement")

# ==================== APPLICATION FASTAPI ====================

def create_app() -> FastAPI:
    """Crée et configure l'application FastAPI."""
    app = FastAPI(
        title="Harena Search Service",
        description="Service de recherche hybride combinant recherche lexicale (Bonsai) et sémantique (Qdrant)",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # Configuration CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # À restreindre en production
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    # Endpoints de santé étendus
    @app.get("/health")
    async def health_check():
        """Endpoint de santé détaillé avec diagnostics complets."""
        try:
            # Test en temps réel des services
            current_elastic_health = False
            current_qdrant_health = False
            
            if elastic_client:
                try:
                    current_elastic_health = await elastic_client.is_healthy()
                except:
                    current_elastic_health = False
            
            if qdrant_client:
                try:
                    current_qdrant_health = await qdrant_client.is_healthy()
                except:
                    current_qdrant_health = False
            
            # Calcul de l'état global
            services_status = {
                "elasticsearch": {
                    "available": elastic_client is not None,
                    "healthy": current_elastic_health,
                    "provider": "Bonsai"
                },
                "qdrant": {
                    "available": qdrant_client is not None,
                    "healthy": current_qdrant_health
                },
                "embeddings": {
                    "available": hasattr(embedding_service, 'client'),
                    "healthy": hasattr(embedding_service, 'client')
                },
                "reranking": {
                    "available": hasattr(reranker_service, 'client'),
                    "healthy": hasattr(reranker_service, 'client')
                },
                "cache": {
                    "available": search_cache is not None,
                    "healthy": search_cache is not None
                },
                "metrics": {
                    "available": metrics_collector is not None,
                    "healthy": metrics_collector is not None
                }
            }
            
            # Calculer les capacités de recherche
            search_capabilities = {
                "lexical_search": services_status["elasticsearch"]["healthy"],
                "semantic_search": services_status["qdrant"]["healthy"],
                "hybrid_search": (services_status["elasticsearch"]["healthy"] and 
                                services_status["qdrant"]["healthy"]),
                "ai_reranking": services_status["reranking"]["healthy"]
            }
            
            # Déterminer l'état global
            critical_healthy = (services_status["elasticsearch"]["healthy"] and 
                              services_status["qdrant"]["healthy"])
            
            if critical_healthy and search_capabilities["ai_reranking"]:
                overall_status = "fully_operational"
            elif critical_healthy:
                overall_status = "operational"
            elif services_status["elasticsearch"]["healthy"] or services_status["qdrant"]["healthy"]:
                overall_status = "degraded"
            else:
                overall_status = "unhealthy"
            
            # Calcul de l'uptime
            uptime_seconds = time.time() - startup_time if startup_time else 0
            
            return {
                "service": "search_service",
                "version": "1.0.0",
                "status": overall_status,
                "timestamp": datetime.now().isoformat(),
                "uptime": {
                    "seconds": round(uptime_seconds, 2),
                    "human": str(timedelta(seconds=int(uptime_seconds)))
                },
                "services": services_status,
                "search_capabilities": search_capabilities,
                "startup_diagnostics": startup_diagnostics.get("summary", {}),
                "performance": {
                    "startup_time": startup_diagnostics.get("summary", {}).get("startup_duration"),
                    "operational_services": startup_diagnostics.get("summary", {}).get("operational_count", 0),
                    "total_services": startup_diagnostics.get("summary", {}).get("total_services", 0)
                },
                "clients_injected": {
                    "elasticsearch": elastic_client is not None,
                    "qdrant": qdrant_client is not None,
                    "cache": search_cache is not None,
                    "metrics": metrics_collector is not None
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur lors du health check: {e}")
            return {
                "service": "search_service",
                "version": "1.0.0",
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "clients_injected": {
                    "elasticsearch": elastic_client is not None,
                    "qdrant": qdrant_client is not None,
                    "cache": search_cache is not None,
                    "metrics": metrics_collector is not None
                }
            }
    
    @app.get("/")
    async def root():
        """Point d'entrée racine du service."""
        uptime_seconds = time.time() - startup_time if startup_time else 0
        summary = startup_diagnostics.get("summary", {})
        
        return {
            "service": "Harena Search Service",
            "description": "Service de recherche hybride (lexicale + sémantique)",
            "version": "1.0.0",
            "status": summary.get("status", "unknown"),
            "uptime_seconds": round(uptime_seconds, 2),
            "search_engines": {
                "elasticsearch": "Bonsai (recherche lexicale)",
                "qdrant": "Stockage vectoriel (recherche sémantique)"
            },
            "capabilities": summary.get("search_capabilities", {}),
            "endpoints": {
                "health": "GET /health",
                "search": "POST /api/v1/search",
                "suggestions": "GET /api/v1/search/suggest"
            },
            "clients_status": {
                "elasticsearch_injected": elastic_client is not None,
                "qdrant_injected": qdrant_client is not None,
                "cache_injected": search_cache is not None,
                "metrics_injected": metrics_collector is not None
            },
            "timestamp": datetime.now().isoformat()
        }
    
    @app.get("/startup-diagnostics")
    async def get_startup_diagnostics():
        """Retourne les diagnostics de démarrage complets."""
        return {
            "timestamp": datetime.now().isoformat(),
            "startup_time": startup_time,
            "diagnostics": startup_diagnostics,
            "current_clients_status": {
                "elasticsearch": {
                    "injected": elastic_client is not None,
                    "type": type(elastic_client).__name__ if elastic_client else None
                },
                "qdrant": {
                    "injected": qdrant_client is not None,
                    "type": type(qdrant_client).__name__ if qdrant_client else None
                },
                "cache": {
                    "injected": search_cache is not None,
                    "type": type(search_cache).__name__ if search_cache else None
                },
                "metrics": {
                    "injected": metrics_collector is not None,
                    "type": type(metrics_collector).__name__ if metrics_collector else None
                }
            }
        }
    
    @app.get("/clients-debug")
    async def debug_clients():
        """Debug détaillé des clients pour diagnostic."""
        try:
            # Import du module routes pour vérifier l'injection
            import search_service.api.routes as routes_module
            
            return {
                "timestamp": datetime.now().isoformat(),
                "global_clients": {
                    "elastic_client": {
                        "exists": elastic_client is not None,
                        "type": type(elastic_client).__name__ if elastic_client else None,
                        "id": id(elastic_client) if elastic_client else None
                    },
                    "qdrant_client": {
                        "exists": qdrant_client is not None,
                        "type": type(qdrant_client).__name__ if qdrant_client else None,
                        "id": id(qdrant_client) if qdrant_client else None
                    },
                    "search_cache": {
                        "exists": search_cache is not None,
                        "type": type(search_cache).__name__ if search_cache else None,
                        "id": id(search_cache) if search_cache else None
                    },
                    "metrics_collector": {
                        "exists": metrics_collector is not None,
                        "type": type(metrics_collector).__name__ if metrics_collector else None,
                        "id": id(metrics_collector) if metrics_collector else None
                    }
                },
                "routes_module_clients": {
                    "elastic_client": {
                        "exists": hasattr(routes_module, 'elastic_client') and routes_module.elastic_client is not None,
                        "type": type(getattr(routes_module, 'elastic_client', None)).__name__ if hasattr(routes_module, 'elastic_client') and routes_module.elastic_client else None,
                        "id": id(getattr(routes_module, 'elastic_client', None)) if hasattr(routes_module, 'elastic_client') and routes_module.elastic_client else None,
                        "same_as_global": (hasattr(routes_module, 'elastic_client') and 
                                         routes_module.elastic_client is elastic_client)
                    },
                    "qdrant_client": {
                        "exists": hasattr(routes_module, 'qdrant_client') and routes_module.qdrant_client is not None,
                        "type": type(getattr(routes_module, 'qdrant_client', None)).__name__ if hasattr(routes_module, 'qdrant_client') and routes_module.qdrant_client else None,
                        "id": id(getattr(routes_module, 'qdrant_client', None)) if hasattr(routes_module, 'qdrant_client') and routes_module.qdrant_client else None,
                        "same_as_global": (hasattr(routes_module, 'qdrant_client') and 
                                         routes_module.qdrant_client is qdrant_client)
                    }
                },
                "injection_success": {
                    "elastic_client": (hasattr(routes_module, 'elastic_client') and 
                                     routes_module.elastic_client is elastic_client and
                                     elastic_client is not None),
                    "qdrant_client": (hasattr(routes_module, 'qdrant_client') and 
                                    routes_module.qdrant_client is qdrant_client and
                                    qdrant_client is not None)
                }
            }
        except Exception as e:
            return {
                "error": f"Debug failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    # Inclure les routes de recherche
    app.include_router(router, prefix="/api/v1/search", tags=["search"])
    
    return app

# ==================== CRÉATION DE L'APPLICATION ====================

app = create_app()

# ==================== POINT D'ENTRÉE ====================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🔧 Démarrage en mode développement local")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
"""
Module principal du service de recherche avec diagnostic complet au démarrage.

Ce module initialise et configure le service de recherche hybride de Harena,
avec vérifications détaillées des services externes (Bonsai Elasticsearch + Qdrant).
"""
import logging
import time
import sys
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from search_service.api.routes import router
from search_service.core.embeddings import embedding_service
from search_service.core.reranker import reranker_service
from search_service.storage.elastic_client import ElasticClient
from search_service.storage.qdrant_client import QdrantClient
from search_service.utils.cache import SearchCache
from search_service.utils.metrics import MetricsCollector
from config_service.config import settings

# Configuration du logging avec format détaillé
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("search_service.main")

# Réduire le bruit des bibliothèques externes
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ==================== VARIABLES GLOBALES ====================

# Instances globales des services
elastic_client = None
qdrant_client = None
search_cache = None
metrics_collector = None
startup_time = None
startup_diagnostics = {}

# ==================== FONCTIONS DE DIAGNOSTIC ====================

def log_startup_banner():
    """Affiche la bannière de démarrage."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        🔍 HARENA SEARCH SERVICE 🔍                          ║
║                     Service de Recherche Hybride v1.0.0                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)
    logger.info("🚀 === DÉMARRAGE DU SERVICE DE RECHERCHE HYBRIDE ===")

def check_environment_configuration() -> Dict[str, Any]:
    """Vérifie et diagnostique la configuration des variables d'environnement."""
    logger.info("🔧 === VÉRIFICATION DE LA CONFIGURATION ===")
    
    config_status = {
        "critical_services": {},
        "optional_services": {},
        "summary": {}
    }
    
    # Services critiques pour le fonctionnement
    critical_configs = {
        "BONSAI_URL": {
            "description": "Elasticsearch via Bonsai (recherche lexicale)",
            "value": settings.BONSAI_URL,
            "required": True
        },
        "QDRANT_URL": {
            "description": "Qdrant (recherche sémantique)",
            "value": settings.QDRANT_URL,
            "required": True
        }
    }
    
    # Services optionnels mais recommandés
    optional_configs = {
        "OPENAI_API_KEY": {
            "description": "OpenAI (génération d'embeddings)",
            "value": settings.OPENAI_API_KEY,
            "impact": "Pas d'embeddings automatiques"
        },
        "COHERE_KEY": {
            "description": "Cohere (reranking des résultats)",
            "value": settings.COHERE_KEY,
            "impact": "Pas de reranking intelligent"
        },
        "QDRANT_API_KEY": {
            "description": "Qdrant API Key (authentification)",
            "value": settings.QDRANT_API_KEY,
            "impact": "Connexion sans authentification"
        }
    }
    
    # Vérification des services critiques
    critical_ok_count = 0
    for key, info in critical_configs.items():
        is_configured = bool(info["value"])
        config_status["critical_services"][key] = {
            "configured": is_configured,
            "description": info["description"],
            "required": info["required"]
        }
        
        if is_configured:
            critical_ok_count += 1
            # Masquer les URLs sensibles pour l'affichage
            if "URL" in key:
                safe_value = info["value"].split('@')[-1] if '@' in info["value"] else info["value"]
                logger.info(f"✅ {key}: {safe_value}")
            else:
                logger.info(f"✅ {key}: Configuré")
        else:
            logger.error(f"❌ {key}: NON CONFIGURÉ - {info['description']}")
    
    # Vérification des services optionnels
    optional_ok_count = 0
    for key, info in optional_configs.items():
        is_configured = bool(info["value"])
        config_status["optional_services"][key] = {
            "configured": is_configured,
            "description": info["description"],
            "impact": info["impact"]
        }
        
        if is_configured:
            optional_ok_count += 1
            if "KEY" in key:
                masked_value = f"{info['value'][:8]}...{info['value'][-4:]}" if len(info["value"]) > 12 else "***"
                logger.info(f"✅ {key}: {masked_value}")
            else:
                logger.info(f"✅ {key}: Configuré")
        else:
            logger.warning(f"⚠️ {key}: Non configuré - {info['impact']}")
    
    # Résumé de la configuration
    config_status["summary"] = {
        "critical_configured": critical_ok_count,
        "critical_total": len(critical_configs),
        "optional_configured": optional_ok_count,
        "optional_total": len(optional_configs),
        "critical_percentage": (critical_ok_count / len(critical_configs)) * 100,
        "optional_percentage": (optional_ok_count / len(optional_configs)) * 100
    }
    
    if critical_ok_count == len(critical_configs):
        logger.info("🎉 CONFIGURATION: Tous les services critiques sont configurés")
    elif critical_ok_count > 0:
        logger.warning("⚠️ CONFIGURATION: Services critiques PARTIELLEMENT configurés")
    else:
        logger.error("🚨 CONFIGURATION: AUCUN service critique configuré")
    
    return config_status

async def initialize_elasticsearch() -> tuple[bool, Optional[ElasticClient], Dict[str, Any]]:
    """Initialise et teste la connexion Elasticsearch (Bonsai)."""
    logger.info("🔍 === INITIALISATION ELASTICSEARCH (BONSAI) ===")
    diagnostic = {
        "service": "elasticsearch_bonsai",
        "configured": False,
        "connected": False,
        "healthy": False,
        "error": None,
        "connection_time": None,
        "cluster_info": {},
        "indices_info": {}
    }
    
    if not settings.BONSAI_URL:
        logger.error("❌ BONSAI_URL non configurée")
        diagnostic["error"] = "BONSAI_URL not configured"
        return False, None, diagnostic
    
    diagnostic["configured"] = True
    
    # Masquer les credentials pour l'affichage
    safe_url = settings.BONSAI_URL.split('@')[-1] if '@' in settings.BONSAI_URL else settings.BONSAI_URL
    logger.info(f"🔗 Connexion à Bonsai Elasticsearch: {safe_url}")
    
    try:
        start_time = time.time()
        client = ElasticClient()
        
        logger.info("⏱️ Test de connexion Elasticsearch...")
        await client.initialize()
        
        connection_time = time.time() - start_time
        diagnostic["connection_time"] = round(connection_time, 3)
        diagnostic["connected"] = True
        
        logger.info(f"✅ Connexion établie en {connection_time:.3f}s")
        
        # Test de santé
        logger.info("🩺 Vérification de la santé du cluster...")
        is_healthy = await client.is_healthy()
        diagnostic["healthy"] = is_healthy
        
        if is_healthy:
            logger.info("🟢 Cluster Elasticsearch en bonne santé")
            
            # Récupérer les informations du cluster
            try:
                cluster_info = await client.get_cluster_info()
                diagnostic["cluster_info"] = cluster_info
                logger.info(f"📊 Cluster: {cluster_info.get('cluster_name', 'Unknown')}")
                logger.info(f"📊 Version: {cluster_info.get('version', {}).get('number', 'Unknown')}")
                logger.info(f"📊 Nœuds: {cluster_info.get('nodes', 'Unknown')}")
            except Exception as e:
                logger.warning(f"⚠️ Impossible de récupérer les infos cluster: {e}")
            
            # Vérifier les indices
            try:
                indices_info = await client.get_indices_info()
                diagnostic["indices_info"] = indices_info
                if indices_info:
                    logger.info(f"📁 Indices disponibles: {len(indices_info)}")
                    for index_name, info in indices_info.items():
                        logger.info(f"   - {index_name}: {info.get('docs', {}).get('count', 0)} documents")
                else:
                    logger.info("📁 Aucun indice trouvé")
            except Exception as e:
                logger.warning(f"⚠️ Impossible de lister les indices: {e}")
            
            return True, client, diagnostic
        else:
            logger.error("🔴 Cluster Elasticsearch non opérationnel")
            diagnostic["error"] = "Cluster unhealthy"
            return False, client, diagnostic
            
    except Exception as e:
        connection_time = time.time() - start_time
        diagnostic["connection_time"] = round(connection_time, 3)
        diagnostic["error"] = str(e)
        
        logger.error(f"💥 Erreur Elasticsearch après {connection_time:.3f}s:")
        logger.error(f"   Type: {type(e).__name__}")
        logger.error(f"   Message: {str(e)}")
        
        # Diagnostic spécifique des erreurs
        error_str = str(e).lower()
        if "connection" in error_str or "timeout" in error_str:
            logger.error("🔌 DIAGNOSTIC: Problème de connectivité réseau")
            logger.error("   - Vérifiez l'URL Bonsai")
            logger.error("   - Vérifiez la connectivité réseau")
        elif "401" in str(e) or "403" in str(e) or "auth" in error_str:
            logger.error("🔑 DIAGNOSTIC: Problème d'authentification")
            logger.error("   - Vérifiez les credentials dans BONSAI_URL")
        elif "ssl" in error_str or "certificate" in error_str:
            logger.error("🔒 DIAGNOSTIC: Problème SSL/TLS")
            logger.error("   - Vérifiez les certificats SSL")
        
        return False, None, diagnostic

async def initialize_qdrant() -> tuple[bool, Optional[QdrantClient], Dict[str, Any]]:
    """Initialise et teste la connexion Qdrant."""
    logger.info("🎯 === INITIALISATION QDRANT ===")
    diagnostic = {
        "service": "qdrant",
        "configured": False,
        "connected": False,
        "healthy": False,
        "error": None,
        "connection_time": None,
        "collections_info": {},
        "version_info": {}
    }
    
    if not settings.QDRANT_URL:
        logger.error("❌ QDRANT_URL non configurée")
        diagnostic["error"] = "QDRANT_URL not configured"
        return False, None, diagnostic
    
    diagnostic["configured"] = True
    logger.info(f"🔗 Connexion à Qdrant: {settings.QDRANT_URL}")
    
    if settings.QDRANT_API_KEY:
        logger.info("🔑 Authentification par API Key activée")
    else:
        logger.info("🔓 Connexion sans authentification")
    
    try:
        start_time = time.time()
        client = QdrantClient()
        
        logger.info("⏱️ Test de connexion Qdrant...")
        await client.initialize()
        
        connection_time = time.time() - start_time
        diagnostic["connection_time"] = round(connection_time, 3)
        diagnostic["connected"] = True
        
        logger.info(f"✅ Connexion établie en {connection_time:.3f}s")
        
        # Test de santé
        logger.info("🩺 Vérification de la santé de Qdrant...")
        is_healthy = await client.is_healthy()
        diagnostic["healthy"] = is_healthy
        
        if is_healthy:
            logger.info("🟢 Service Qdrant en bonne santé")
            
            # Récupérer les informations des collections
            try:
                collections_info = await client.get_collections_info()
                diagnostic["collections_info"] = collections_info
                if collections_info:
                    logger.info(f"📊 Collections disponibles: {len(collections_info)}")
                    for collection_name, info in collections_info.items():
                        points_count = info.get('points_count', 0)
                        vectors_count = info.get('vectors_count', 0)
                        logger.info(f"   - {collection_name}: {points_count} points, {vectors_count} vecteurs")
                else:
                    logger.warning("📊 Aucune collection trouvée")
            except Exception as e:
                logger.warning(f"⚠️ Impossible de récupérer les infos collections: {e}")
            
            # Vérifier la collection des transactions
            try:
                collection_exists = await client.collection_exists("financial_transactions")
                if collection_exists:
                    logger.info("✅ Collection 'financial_transactions' trouvée")
                else:
                    logger.warning("⚠️ Collection 'financial_transactions' non trouvée")
                    logger.warning("   - Lancez l'enrichment_service pour créer la collection")
            except Exception as e:
                logger.warning(f"⚠️ Impossible de vérifier la collection: {e}")
            
            return True, client, diagnostic
        else:
            logger.error("🔴 Service Qdrant non opérationnel")
            diagnostic["error"] = "Service unhealthy"
            return False, client, diagnostic
            
    except Exception as e:
        connection_time = time.time() - start_time
        diagnostic["connection_time"] = round(connection_time, 3)
        diagnostic["error"] = str(e)
        
        logger.error(f"💥 Erreur Qdrant après {connection_time:.3f}s:")
        logger.error(f"   Type: {type(e).__name__}")
        logger.error(f"   Message: {str(e)}")
        
        # Diagnostic spécifique des erreurs
        error_str = str(e).lower()
        if "connection" in error_str or "timeout" in error_str:
            logger.error("🔌 DIAGNOSTIC: Problème de connectivité réseau")
            logger.error("   - Vérifiez l'URL Qdrant")
            logger.error("   - Vérifiez la connectivité réseau")
        elif "401" in str(e) or "403" in str(e) or "auth" in error_str:
            logger.error("🔑 DIAGNOSTIC: Problème d'authentification")
            logger.error("   - Vérifiez QDRANT_API_KEY")
        elif "404" in str(e):
            logger.error("📂 DIAGNOSTIC: Collection non trouvée")
            logger.error("   - La collection 'financial_transactions' doit être créée")
            logger.error("   - Lancez d'abord l'enrichment_service")
        
        return False, None, diagnostic

async def initialize_ai_services() -> Dict[str, Dict[str, Any]]:
    """Initialise les services IA (embeddings et reranking)."""
    logger.info("🤖 === INITIALISATION DES SERVICES IA ===")
    ai_diagnostics = {}
    
    # Service d'embeddings (OpenAI)
    embedding_diagnostic = {
        "service": "openai_embeddings",
        "configured": bool(settings.OPENAI_API_KEY),
        "initialized": False,
        "error": None,
        "init_time": None
    }
    
    if settings.OPENAI_API_KEY:
        try:
            logger.info("🤖 Initialisation du service d'embeddings OpenAI...")
            start_time = time.time()
            await embedding_service.initialize()
            init_time = time.time() - start_time
            
            embedding_diagnostic["initialized"] = True
            embedding_diagnostic["init_time"] = round(init_time, 3)
            logger.info(f"✅ Service d'embeddings initialisé en {init_time:.3f}s")
        except Exception as e:
            embedding_diagnostic["error"] = str(e)
            logger.error(f"❌ Échec initialisation embeddings: {type(e).__name__}: {str(e)}")
            logger.error("📍 Vérifiez OPENAI_API_KEY et la connectivité internet")
    else:
        logger.warning("⚠️ OPENAI_API_KEY non configurée - embeddings indisponibles")
    
    ai_diagnostics["embeddings"] = embedding_diagnostic
    
    # Service de reranking (Cohere)
    reranking_diagnostic = {
        "service": "cohere_reranking",
        "configured": bool(settings.COHERE_KEY),
        "initialized": False,
        "error": None,
        "init_time": None
    }
    
    if settings.COHERE_KEY:
        try:
            logger.info("🎯 Initialisation du service de reranking Cohere...")
            start_time = time.time()
            await reranker_service.initialize()
            init_time = time.time() - start_time
            
            reranking_diagnostic["initialized"] = True
            reranking_diagnostic["init_time"] = round(init_time, 3)
            logger.info(f"✅ Service de reranking initialisé en {init_time:.3f}s")
        except Exception as e:
            reranking_diagnostic["error"] = str(e)
            logger.error(f"❌ Échec initialisation reranking: {type(e).__name__}: {str(e)}")
            logger.error("📍 Vérifiez COHERE_KEY et la connectivité internet")
    else:
        logger.warning("⚠️ COHERE_KEY non configurée - reranking indisponible")
    
    ai_diagnostics["reranking"] = reranking_diagnostic
    
    return ai_diagnostics

def initialize_cache_and_metrics() -> Dict[str, Dict[str, Any]]:
    """Initialise le cache et les métriques."""
    logger.info("🗃️ === INITIALISATION CACHE ET MÉTRIQUES ===")
    utils_diagnostics = {}
    
    # Cache de recherche
    cache_diagnostic = {
        "service": "search_cache",
        "initialized": False,
        "error": None
    }
    
    global search_cache
    try:
        search_cache = SearchCache()
        cache_diagnostic["initialized"] = True
        logger.info("✅ Cache de recherche initialisé")
    except Exception as e:
        cache_diagnostic["error"] = str(e)
        logger.error(f"❌ Erreur initialisation cache: {e}")
        search_cache = None
    
    utils_diagnostics["cache"] = cache_diagnostic
    
    # Collecteur de métriques
    metrics_diagnostic = {
        "service": "metrics_collector",
        "initialized": False,
        "error": None
    }
    
    global metrics_collector
    try:
        metrics_collector = MetricsCollector()
        metrics_diagnostic["initialized"] = True
        logger.info("✅ Collecteur de métriques initialisé")
    except Exception as e:
        metrics_diagnostic["error"] = str(e)
        logger.error(f"❌ Erreur initialisation métriques: {e}")
        metrics_collector = None
    
    utils_diagnostics["metrics"] = metrics_diagnostic
    
    return utils_diagnostics

def inject_dependencies():
    """Injecte les dépendances dans le module routes."""
    logger.info("🔗 === INJECTION DES DÉPENDANCES ===")
    
    import search_service.api.routes as routes
    routes.elastic_client = elastic_client
    routes.qdrant_client = qdrant_client
    routes.search_cache = search_cache
    routes.metrics_collector = metrics_collector
    
    logger.info("✅ Dépendances injectées dans les routes")

def generate_startup_summary() -> Dict[str, Any]:
    """Génère un résumé complet du démarrage."""
    global startup_diagnostics
    
    startup_duration = time.time() - startup_time
    
    # Compter les services opérationnels
    operational_services = []
    failed_services = []
    
    # Services critiques
    if elastic_client:
        operational_services.append("Elasticsearch (Bonsai)")
    else:
        failed_services.append("Elasticsearch (Bonsai)")
    
    if qdrant_client:
        operational_services.append("Qdrant")
    else:
        failed_services.append("Qdrant")
    
    # Services IA
    ai_diag = startup_diagnostics.get("ai_services", {})
    if ai_diag.get("embeddings", {}).get("initialized"):
        operational_services.append("OpenAI Embeddings")
    else:
        failed_services.append("OpenAI Embeddings")
    
    if ai_diag.get("reranking", {}).get("initialized"):
        operational_services.append("Cohere Reranking")
    else:
        failed_services.append("Cohere Reranking")
    
    # Services utilitaires
    if search_cache:
        operational_services.append("Cache")
    if metrics_collector:
        operational_services.append("Métriques")
    
    # Déterminer l'état global du service
    critical_services_ok = elastic_client and qdrant_client
    
    if critical_services_ok:
        if len(operational_services) >= 4:  # Elasticsearch + Qdrant + au moins 2 autres
            status = "FULLY_OPERATIONAL"
            status_icon = "🎉"
            status_msg = "COMPLÈTEMENT OPÉRATIONNEL"
        else:
            status = "OPERATIONAL"
            status_icon = "✅"
            status_msg = "OPÉRATIONNEL"
    elif elastic_client or qdrant_client:
        status = "DEGRADED"
        status_icon = "⚠️"
        status_msg = "OPÉRATIONNEL DÉGRADÉ"
    else:
        status = "FAILED"
        status_icon = "🚨"
        status_msg = "NON OPÉRATIONNEL"
    
    summary = {
        "status": status,
        "status_icon": status_icon,
        "status_message": status_msg,
        "startup_duration": round(startup_duration, 2),
        "operational_services": operational_services,
        "failed_services": failed_services,
        "operational_count": len(operational_services),
        "total_services": len(operational_services) + len(failed_services),
        "critical_services_operational": critical_services_ok,
        "search_capabilities": {
            "lexical_search": bool(elastic_client),
            "semantic_search": bool(qdrant_client),
            "hybrid_search": bool(elastic_client and qdrant_client),
            "ai_reranking": ai_diag.get("reranking", {}).get("initialized", False)
        }
    }
    
    # Affichage du résumé
    logger.info("=" * 100)
    logger.info(f"{status_icon} SERVICE DE RECHERCHE HYBRIDE - ÉTAT: {status_msg}")
    logger.info(f"⏱️ Temps de démarrage: {startup_duration:.2f}s")
    logger.info(f"🎯 Services opérationnels ({len(operational_services)}/{len(operational_services) + len(failed_services)}): {', '.join(operational_services) if operational_services else 'Aucun'}")
    
    if failed_services:
        logger.warning(f"❌ Services échoués: {', '.join(failed_services)}")
    
    # Messages informatifs selon l'état
    if status == "FULLY_OPERATIONAL":
        logger.info("🚀 Toutes les fonctionnalités de recherche avancée sont disponibles")
        logger.info("   ✓ Recherche lexicale (Elasticsearch/Bonsai)")
        logger.info("   ✓ Recherche sémantique (Qdrant)")
        logger.info("   ✓ Recherche hybride avec fusion des scores")
        logger.info("   ✓ Reranking intelligent des résultats")
    elif status == "OPERATIONAL":
        logger.info("✅ Fonctionnalités de recherche de base disponibles")
        logger.info("   ✓ Recherche lexicale ET sémantique")
        logger.warning("   ⚠️ Certaines fonctionnalités avancées peuvent être limitées")
    elif status == "DEGRADED":
        logger.warning("⚠️ Service de recherche en mode dégradé")
        if elastic_client and not qdrant_client:
            logger.warning("   ✓ Recherche lexicale disponible")
            logger.warning("   ❌ Recherche sémantique indisponible")
        elif qdrant_client and not elastic_client:
            logger.warning("   ❌ Recherche lexicale indisponible")
            logger.warning("   ✓ Recherche sémantique disponible")
    else:
        logger.error("🚨 Service de recherche indisponible")
        logger.error("   ❌ Vérifiez la configuration BONSAI_URL et QDRANT_URL")
        logger.error("   ❌ Vérifiez la connectivité réseau")
    
    logger.info("📊 Endpoints disponibles:")
    logger.info("   GET  /health - Vérification de santé détaillée")
    logger.info("   POST /api/v1/search - Recherche de transactions")
    logger.info("   GET  /api/v1/search/suggest - Suggestions de recherche")
    logger.info("=" * 100)
    
    return summary

# ==================== CYCLE DE VIE DE L'APPLICATION ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie de l'application avec diagnostics complets."""
    global startup_time, startup_diagnostics, elastic_client, qdrant_client
    
    try:
        # === PHASE DE DÉMARRAGE ===
        log_startup_banner()
        startup_time = time.time()
        
        # 1. Vérification de la configuration
        config_status = check_environment_configuration()
        startup_diagnostics["configuration"] = config_status
        
        # 2. Initialisation d'Elasticsearch (Bonsai)
        elastic_success, elastic_client, elastic_diag = await initialize_elasticsearch()
        startup_diagnostics["elasticsearch"] = elastic_diag
        
        # 3. Initialisation de Qdrant
        qdrant_success, qdrant_client, qdrant_diag = await initialize_qdrant()
        startup_diagnostics["qdrant"] = qdrant_diag
        
        # 4. Initialisation des services IA
        ai_diagnostics = await initialize_ai_services()
        startup_diagnostics["ai_services"] = ai_diagnostics
        
        # 5. Initialisation du cache et des métriques
        utils_diagnostics = initialize_cache_and_metrics()
        startup_diagnostics["utilities"] = utils_diagnostics
        
        # 6. Injection des dépendances
        inject_dependencies()
        
        # 7. Génération du résumé de démarrage
        startup_summary = generate_startup_summary()
        startup_diagnostics["summary"] = startup_summary
        
        # Application prête
        yield
        
    except Exception as e:
        logger.critical(f"💥 ERREUR CRITIQUE au démarrage: {type(e).__name__}: {str(e)}")
        import traceback
        logger.critical(f"Stack trace:\n{traceback.format_exc()}")
        raise
    
    finally:
        # === PHASE D'ARRÊT ===
        logger.info("⏹️ Arrêt du service de recherche...")
        
        # Fermeture des services IA
        try:
            if hasattr(embedding_service, 'close'):
                await embedding_service.close()
                logger.info("✅ Service d'embeddings fermé")
        except Exception as e:
            logger.error(f"❌ Erreur fermeture embeddings: {e}")
        
        try:
            if hasattr(reranker_service, 'close'):
                await reranker_service.close()
                logger.info("✅ Service de reranking fermé")
        except Exception as e:
            logger.error(f"❌ Erreur fermeture reranking: {e}")
        
        logger.info("🔚 Service de recherche arrêté proprement")

# ==================== APPLICATION FASTAPI ====================

def create_app() -> FastAPI:
    """Crée et configure l'application FastAPI."""
    app = FastAPI(
        title="Harena Search Service",
        description="Service de recherche hybride combinant recherche lexicale (Bonsai) et sémantique (Qdrant)",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # Configuration CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # À restreindre en production
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    # Endpoints de santé étendus
    @app.get("/health")
    async def health_check():
        """Endpoint de santé détaillé avec diagnostics complets."""
        try:
            # Test en temps réel des services
            current_elastic_health = elastic_client.is_healthy() if elastic_client else False
            current_qdrant_health = qdrant_client.is_healthy() if qdrant_client else False
            
            # Calcul de l'état global
            services_status = {
                "elasticsearch": {
                    "available": elastic_client is not None,
                    "healthy": await current_elastic_health if elastic_client else False,
                    "provider": "Bonsai"
                },
                "qdrant": {
                    "available": qdrant_client is not None,
                    "healthy": await current_qdrant_health if qdrant_client else False
                },
                "embeddings": {
                    "available": hasattr(embedding_service, 'client'),
                    "healthy": hasattr(embedding_service, 'client')
                },
                "reranking": {
                    "available": hasattr(reranker_service, 'client'),
                    "healthy": hasattr(reranker_service, 'client')
                },
                "cache": {
                    "available": search_cache is not None,
                    "healthy": search_cache is not None
                },
                "metrics": {
                    "available": metrics_collector is not None,
                    "healthy": metrics_collector is not None
                }
            }
            
            # Calculer les capacités de recherche
            search_capabilities = {
                "lexical_search": services_status["elasticsearch"]["healthy"],
                "semantic_search": services_status["qdrant"]["healthy"],
                "hybrid_search": (services_status["elasticsearch"]["healthy"] and 
                                services_status["qdrant"]["healthy"]),
                "ai_reranking": services_status["reranking"]["healthy"]
            }
            
            # Déterminer l'état global
            critical_healthy = (services_status["elasticsearch"]["healthy"] and 
                              services_status["qdrant"]["healthy"])
            
            if critical_healthy and search_capabilities["ai_reranking"]:
                overall_status = "fully_operational"
            elif critical_healthy:
                overall_status = "operational"
            elif services_status["elasticsearch"]["healthy"] or services_status["qdrant"]["healthy"]:
                overall_status = "degraded"
            else:
                overall_status = "unhealthy"
            
            # Calcul de l'uptime
            uptime_seconds = time.time() - startup_time if startup_time else 0
            
            return {
                "service": "search_service",
                "version": "1.0.0",
                "status": overall_status,
                "timestamp": datetime.now().isoformat(),
                "uptime": {
                    "seconds": round(uptime_seconds, 2),
                    "human": str(timedelta(seconds=int(uptime_seconds)))
                },
                "services": services_status,
                "search_capabilities": search_capabilities,
                "startup_diagnostics": startup_diagnostics.get("summary", {}),
                "performance": {
                    "startup_time": startup_diagnostics.get("summary", {}).get("startup_duration"),
                    "operational_services": startup_diagnostics.get("summary", {}).get("operational_count", 0),
                    "total_services": startup_diagnostics.get("summary", {}).get("total_services", 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur lors du health check: {e}")
            return {
                "service": "search_service",
                "version": "1.0.0",
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    @app.get("/")
    async def root():
        """Point d'entrée racine du service."""
        uptime_seconds = time.time() - startup_time if startup_time else 0
        summary = startup_diagnostics.get("summary", {})
        
        return {
            "service": "Harena Search Service",
            "description": "Service de recherche hybride (lexicale + sémantique)",
            "version": "1.0.0",
            "status": summary.get("status", "unknown"),
            "uptime_seconds": round(uptime_seconds, 2),
            "search_engines": {
                "elasticsearch": "Bonsai (recherche lexicale)",
                "qdrant": "Stockage vectoriel (recherche sémantique)"
            },
            "capabilities": summary.get("search_capabilities", {}),
            "endpoints": {
                "health": "GET /health",
                "search": "POST /api/v1/search",
                "suggestions": "GET /api/v1/search/suggest"
            },
            "timestamp": datetime.now().isoformat()
        }
    
    @app.get("/startup-diagnostics")
    async def get_startup_diagnostics():
        """Retourne les diagnostics de démarrage complets."""
        return {
            "timestamp": datetime.now().isoformat(),
            "startup_time": startup_time,
            "diagnostics": startup_diagnostics
        }
    
    # Inclure les routes de recherche
    app.include_router(router, prefix="/api/v1/search", tags=["search"])
    
    return app

# ==================== CRÉATION DE L'APPLICATION ====================

app = create_app()

# ==================== POINT D'ENTRÉE ====================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🔧 Démarrage en mode développement local")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )