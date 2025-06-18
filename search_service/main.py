"""
Service de recherche Harena - Point d'entrée principal.

Ce module configure et démarre le service de recherche hybride combinant
Elasticsearch (Bonsai) pour la recherche lexicale et Qdrant pour la recherche sémantique.
"""

import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configuration du logging avant les autres imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
    stream=sys.stdout
)

logger = logging.getLogger("search_service.main")

# ==================== IMPORTS DES MODULES ====================

try:
    from config_service.config import settings
    from search_service.storage.elastic_client import ElasticClient
    from search_service.storage.qdrant_client import QdrantClient
    from search_service.core.embedding_service import EmbeddingService
    from search_service.core.reranker import RerankerService
    from search_service.utils.cache import SearchCache
    from search_service.monitoring.metrics_collector import MetricsCollector
    from search_service.api.routes import router
    
    logger.info("✅ Imports réussis")
    
except ImportError as e:
    logger.critical(f"💥 Erreur d'import critique: {e}")
    raise

# ==================== VARIABLES GLOBALES ====================

# Variables de l'application
startup_time = None
startup_diagnostics = {}

# Clients de stockage
elastic_client: Optional[ElasticClient] = None
qdrant_client: Optional[QdrantClient] = None

# Services IA
embedding_service: Optional[EmbeddingService] = None
reranker_service: Optional[RerankerService] = None

# Services utilitaires
search_cache: Optional[SearchCache] = None
metrics_collector: Optional[MetricsCollector] = None

# ==================== FONCTIONS D'INITIALISATION ====================

def log_startup_banner():
    """Affiche une bannière de démarrage."""
    logger.info("=" * 100)
    logger.info("🚀 DÉMARRAGE DU SERVICE DE RECHERCHE HARENA")
    logger.info("=" * 100)
    logger.info("🔍 Service: Recherche hybride (lexicale + sémantique)")
    logger.info("📊 Moteurs: Elasticsearch (Bonsai) + Qdrant")
    logger.info("🤖 IA: OpenAI Embeddings + Cohere Reranking")
    logger.info("=" * 100)

def check_environment_configuration() -> Dict[str, Any]:
    """Vérifie la configuration de l'environnement."""
    logger.info("⚙️ === VÉRIFICATION DE LA CONFIGURATION ===")
    
    config_status = {
        "critical_services": {},
        "optional_services": {},
        "summary": {}
    }
    
    # Services critiques
    critical_configs = {
        "BONSAI_URL": {
            "value": settings.BONSAI_URL,
            "description": "URL Elasticsearch (Bonsai)",
            "required": True
        },
        "QDRANT_URL": {
            "value": settings.QDRANT_URL,
            "description": "URL Qdrant",
            "required": True
        }
    }
    
    # Services optionnels
    optional_configs = {
        "OPENAI_API_KEY": {
            "value": settings.OPENAI_API_KEY,
            "description": "Clé OpenAI pour embeddings",
            "impact": "Désactivation de la recherche sémantique"
        },
        "COHERE_KEY": {
            "value": settings.COHERE_KEY,
            "description": "Clé Cohere pour reranking",
            "impact": "Pas de reranking intelligent"
        },
        "QDRANT_API_KEY": {
            "value": settings.QDRANT_API_KEY,
            "description": "Clé API Qdrant",
            "impact": "Connexion non sécurisée à Qdrant"
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
        # IMPORTANT: Attendre le retour de initialize()
        initialization_success = await client.initialize()
        
        connection_time = time.time() - start_time
        diagnostic["connection_time"] = round(connection_time, 3)
        
        if initialization_success and client._initialized:
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
                    logger.info(f"📊 Version: {cluster_info.get('version', 'Unknown')}")
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
                            doc_count = info.get('docs', {}).get('count', 0)
                            logger.info(f"   - {index_name}: {doc_count} documents")
                    else:
                        logger.info("📁 Aucun indice trouvé")
                except Exception as e:
                    logger.warning(f"⚠️ Impossible de lister les indices: {e}")
                
                return True, client, diagnostic
            else:
                logger.error("🔴 Cluster Elasticsearch non opérationnel")
                diagnostic["error"] = "Cluster unhealthy"
                return False, client, diagnostic
        else:
            logger.error("🔴 Échec d'initialisation du client Elasticsearch")
            diagnostic["error"] = "Client initialization failed"
            return False, None, diagnostic
            
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
        initialization_success = await client.initialize()
        
        connection_time = time.time() - start_time
        diagnostic["connection_time"] = round(connection_time, 3)
        
        if initialization_success and client._initialized:
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
                        logger.info(f"📁 Collections disponibles: {len(collections_info)}")
                        for collection_name, info in collections_info.items():
                            vectors_count = info.get('vectors_count', 0)
                            logger.info(f"   - {collection_name}: {vectors_count} vecteurs")
                    else:
                        logger.info("📁 Aucune collection trouvée")
                except Exception as e:
                    logger.warning(f"⚠️ Impossible de lister les collections: {e}")
                
                return True, client, diagnostic
            else:
                logger.error("🔴 Service Qdrant non opérationnel")
                diagnostic["error"] = "Qdrant unhealthy"
                return False, client, diagnostic
        else:
            logger.error("🔴 Échec d'initialisation du client Qdrant")
            diagnostic["error"] = "Client initialization failed"
            return False, None, diagnostic
            
    except Exception as e:
        connection_time = time.time() - start_time
        diagnostic["connection_time"] = round(connection_time, 3)
        diagnostic["error"] = str(e)
        
        logger.error(f"💥 Erreur Qdrant après {connection_time:.3f}s:")
        logger.error(f"   Type: {type(e).__name__}")
        logger.error(f"   Message: {str(e)}")
        
        return False, None, diagnostic

async def initialize_ai_services() -> Dict[str, Any]:
    """Initialise les services IA (embeddings et reranking)."""
    global embedding_service, reranker_service
    
    logger.info("🤖 === INITIALISATION DES SERVICES IA ===")
    diagnostics = {
        "embeddings": {"initialized": False, "error": None},
        "reranking": {"initialized": False, "error": None}
    }
    
    # Service d'embeddings (OpenAI)
    if settings.OPENAI_API_KEY:
        try:
            logger.info("🧠 Initialisation du service d'embeddings OpenAI...")
            embedding_service = EmbeddingService()
            await embedding_service.initialize()
            diagnostics["embeddings"]["initialized"] = True
            logger.info("✅ Service d'embeddings initialisé")
        except Exception as e:
            logger.error(f"❌ Erreur initialisation embeddings: {e}")
            diagnostics["embeddings"]["error"] = str(e)
            embedding_service = None
    else:
        logger.warning("⚠️ OPENAI_API_KEY non configurée - embeddings désactivés")
    
    # Service de reranking (Cohere)
    if settings.COHERE_KEY:
        try:
            logger.info("🎯 Initialisation du service de reranking Cohere...")
            reranker_service = RerankerService()
            await reranker_service.initialize()
            diagnostics["reranking"]["initialized"] = True
            logger.info("✅ Service de reranking initialisé")
        except Exception as e:
            logger.error(f"❌ Erreur initialisation reranking: {e}")
            diagnostics["reranking"]["error"] = str(e)
            reranker_service = None
    else:
        logger.warning("⚠️ COHERE_KEY non configurée - reranking désactivé")
    
    return diagnostics

def initialize_cache_and_metrics() -> Dict[str, Any]:
    """Initialise le cache et le collecteur de métriques."""
    global search_cache, metrics_collector
    
    logger.info("⚡ === INITIALISATION CACHE ET MÉTRIQUES ===")
    diagnostics = {
        "cache": {"initialized": False, "error": None},
        "metrics": {"initialized": False, "error": None}
    }
    
    # Cache de recherche
    try:
        logger.info("💾 Initialisation du cache de recherche...")
        search_cache = SearchCache()
        diagnostics["cache"]["initialized"] = True
        logger.info("✅ Cache de recherche initialisé")
    except Exception as e:
        logger.error(f"❌ Erreur initialisation cache: {e}")
        diagnostics["cache"]["error"] = str(e)
        search_cache = None
    
    # Collecteur de métriques
    try:
        logger.info("📊 Initialisation du collecteur de métriques...")
        metrics_collector = MetricsCollector()
        diagnostics["metrics"]["initialized"] = True
        logger.info("✅ Collecteur de métriques initialisé")
    except Exception as e:
        logger.error(f"❌ Erreur initialisation métriques: {e}")
        diagnostics["metrics"]["error"] = str(e)
        metrics_collector = None
    
    return diagnostics

def inject_dependencies():
    """Injecte les clients dans les modules qui en ont besoin - APRÈS initialisation complète."""
    logger.info("🔗 === INJECTION DES DÉPENDANCES ===")
    
    try:
        import search_service.api.routes as routes
        
        # Injecter seulement les clients qui ont été initialisés avec succès
        if elastic_client and hasattr(elastic_client, '_initialized') and elastic_client._initialized:
            routes.elastic_client = elastic_client
            logger.info("✅ ElasticClient injecté dans routes")
        else:
            routes.elastic_client = None
            logger.warning("⚠️ ElasticClient non disponible - non injecté")
        
        if qdrant_client and hasattr(qdrant_client, '_initialized') and qdrant_client._initialized:
            routes.qdrant_client = qdrant_client
            logger.info("✅ QdrantClient injecté dans routes")
        else:
            routes.qdrant_client = None
            logger.warning("⚠️ QdrantClient non disponible - non injecté")
        
        # Injecter les services IA
        routes.embedding_service = embedding_service
        routes.reranker_service = reranker_service
        
        # Injecter les services utilitaires
        routes.search_cache = search_cache
        routes.metrics_collector = metrics_collector
        
        logger.info("✅ Injection des dépendances terminée")
        
        # Log du statut final des clients
        clients_status = {
            "elasticsearch": elastic_client is not None and getattr(elastic_client, '_initialized', False),
            "qdrant": qdrant_client is not None and getattr(qdrant_client, '_initialized', False),
            "embeddings": embedding_service is not None,
            "reranking": reranker_service is not None,
            "cache": search_cache is not None,
            "metrics": metrics_collector is not None
        }
        
        active_clients = sum(clients_status.values())
        logger.info(f"📊 Services actifs: {active_clients}/6")
        
        for service_name, is_active in clients_status.items():
            status_icon = "✅" if is_active else "❌"
            logger.info(f"   {status_icon} {service_name}: {'Actif' if is_active else 'Inactif'}")
            
    except Exception as e:
        logger.error(f"💥 Erreur lors de l'injection des dépendances: {e}")
        # En cas d'erreur, injecter au moins None pour éviter les erreurs d'import
        try:
            import search_service.api.routes as routes
            routes.elastic_client = None
            routes.qdrant_client = None
            routes.embedding_service = None
            routes.reranker_service = None
            routes.search_cache = None
            routes.metrics_collector = None
            logger.warning("⚠️ Services par défaut (None) injectés suite à l'erreur")
        except Exception as fallback_error:
            logger.error(f"💥 Impossible d'injecter même les valeurs par défaut: {fallback_error}")

def generate_startup_summary() -> Dict[str, Any]:
    """Génère un résumé complet du démarrage."""
    global startup_diagnostics
    
    startup_duration = time.time() - startup_time
    
    # Compter les services opérationnels
    operational_services = []
    failed_services = []
    
    # Services critiques
    if elastic_client and getattr(elastic_client, '_initialized', False):
        operational_services.append("Elasticsearch (Bonsai)")
    else:
        failed_services.append("Elasticsearch (Bonsai)")
    
    if qdrant_client and getattr(qdrant_client, '_initialized', False):
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
    critical_services_ok = (elastic_client and getattr(elastic_client, '_initialized', False) and 
                           qdrant_client and getattr(qdrant_client, '_initialized', False))
    
    if critical_services_ok:
        if len(operational_services) >= 4:  # Elasticsearch + Qdrant + au moins 2 autres
            status = "FULLY_OPERATIONAL"
            status_icon = "🎉"
            status_msg = "COMPLÈTEMENT OPÉRATIONNEL"
        else:
            status = "OPERATIONAL"
            status_icon = "✅"
            status_msg = "OPÉRATIONNEL"
    elif (elastic_client and getattr(elastic_client, '_initialized', False)) or \
         (qdrant_client and getattr(qdrant_client, '_initialized', False)):
        status = "DEGRADED"
        status_icon = "⚠️"
        status_msg = "DÉGRADÉ"
    else:
        status = "FAILED"
        status_icon = "🚨"
        status_msg = "DÉFAILLANT"
    
    # Capacités de recherche
    search_capabilities = {
        "lexical_search": elastic_client is not None and getattr(elastic_client, '_initialized', False),
        "semantic_search": (qdrant_client is not None and getattr(qdrant_client, '_initialized', False) and 
                           embedding_service is not None),
        "hybrid_search": (elastic_client is not None and getattr(elastic_client, '_initialized', False) and 
                         qdrant_client is not None and getattr(qdrant_client, '_initialized', False) and 
                         embedding_service is not None),
        "intelligent_reranking": reranker_service is not None,
        "caching": search_cache is not None,
        "metrics": metrics_collector is not None
    }
    
    summary = {
        "status": status,
        "status_message": status_msg,
        "startup_duration": round(startup_duration, 2),
        "operational_count": len(operational_services),
        "failed_count": len(failed_services),
        "total_services": len(operational_services) + len(failed_services),
        "operational_services": operational_services,
        "failed_services": failed_services,
        "search_capabilities": search_capabilities,
        "timestamp": datetime.now().isoformat()
    }
    
    # Affichage du résumé
    logger.info("=" * 100)
    logger.info(f"{status_icon} RÉSUMÉ DU DÉMARRAGE - {status_msg}")
    logger.info("=" * 100)
    logger.info(f"⏱️ Durée: {startup_duration:.2f}s")
    logger.info(f"📊 Services: {len(operational_services)}/{len(operational_services) + len(failed_services)} opérationnels")
    
    if operational_services:
        logger.info("✅ Services opérationnels:")
        for service in operational_services:
            logger.info(f"   ✓ {service}")
    
    if failed_services:
        logger.info("❌ Services défaillants:")
        for service in failed_services:
            logger.info(f"   ✗ {service}")
    
    # Affichage des capacités
    if status == "FULLY_OPERATIONAL":
        logger.info("🎉 CAPACITÉS COMPLÈTES DISPONIBLES:")
        logger.info("   ✓ Recherche lexicale avancée (Elasticsearch/Bonsai)")
        logger.info("   ✓ Recherche sémantique (Qdrant)")
        logger.info("   ✓ Recherche hybride avec fusion des scores")
        logger.info("   ✓ Reranking intelligent des résultats")
    elif status == "OPERATIONAL":
        logger.info("✅ Fonctionnalités de recherche de base disponibles")
        logger.info("   ✓ Recherche lexicale ET sémantique")
        logger.warning("   ⚠️ Certaines fonctionnalités avancées peuvent être limitées")
    elif status == "DEGRADED":
        logger.warning("⚠️ Service de recherche en mode dégradé")
        if elastic_client and getattr(elastic_client, '_initialized', False) and not (qdrant_client and getattr(qdrant_client, '_initialized', False)):
            logger.warning("   ✓ Recherche lexicale disponible")
            logger.warning("   ❌ Recherche sémantique indisponible")
        elif qdrant_client and getattr(qdrant_client, '_initialized', False) and not (elastic_client and getattr(elastic_client, '_initialized', False)):
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
        
        # Fermeture des clients de stockage
        try:
            if elastic_client:
                await elastic_client.close()
                logger.info("✅ Client Elasticsearch fermé")
        except Exception as e:
            logger.error(f"❌ Erreur fermeture Elasticsearch: {e}")
        
        try:
            if qdrant_client:
                await qdrant_client.close()
                logger.info("✅ Client Qdrant fermé")
        except Exception as e:
            logger.error(f"❌ Erreur fermeture Qdrant: {e}")
        
        # Fermeture des services IA
        try:
            if embedding_service and hasattr(embedding_service, 'close'):
                await embedding_service.close()
                logger.info("✅ Service d'embeddings fermé")
        except Exception as e:
            logger.error(f"❌ Erreur fermeture embeddings: {e}")
        
        try:
            if reranker_service and hasattr(reranker_service, 'close'):
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
            uptime_seconds = time.time() - startup_time if startup_time else 0
            
            # Statut des services
            services_status = {
                "elasticsearch": {
                    "healthy": elastic_client is not None and getattr(elastic_client, '_initialized', False) and await elastic_client.is_healthy() if elastic_client else False,
                    "initialized": elastic_client is not None and getattr(elastic_client, '_initialized', False)
                },
                "qdrant": {
                    "healthy": qdrant_client is not None and getattr(qdrant_client, '_initialized', False) and await qdrant_client.is_healthy() if qdrant_client else False,
                    "initialized": qdrant_client is not None and getattr(qdrant_client, '_initialized', False)
                },
                "embeddings": {
                    "available": embedding_service is not None
                },
                "reranking": {
                    "available": reranker_service is not None
                },
                "cache": {
                    "available": search_cache is not None
                },
                "metrics": {
                    "available": metrics_collector is not None
                }
            }
            
            # Capacités de recherche
            search_capabilities = {
                "lexical_search": services_status["elasticsearch"]["healthy"],
                "semantic_search": services_status["qdrant"]["healthy"] and services_status["embeddings"]["available"],
                "hybrid_search": services_status["elasticsearch"]["healthy"] and services_status["qdrant"]["healthy"] and services_status["embeddings"]["available"],
                "intelligent_reranking": services_status["reranking"]["available"]
            }
            
            # Déterminer le statut global
            if services_status["elasticsearch"]["healthy"] and services_status["qdrant"]["healthy"]:
                overall_status = "healthy"
            elif services_status["elasticsearch"]["healthy"] or services_status["qdrant"]["healthy"]:
                overall_status = "degraded"
            else:
                overall_status = "unhealthy"
            
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
                        "id": id(elastic_client) if elastic_client else None,
                        "initialized": getattr(elastic_client, '_initialized', False) if elastic_client else False
                    },
                    "qdrant_client": {
                        "exists": qdrant_client is not None,
                        "type": type(qdrant_client).__name__ if qdrant_client else None,
                        "id": id(qdrant_client) if qdrant_client else None,
                        "initialized": getattr(qdrant_client, '_initialized', False) if qdrant_client else False
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
                    },
                    "search_cache": {
                        "exists": hasattr(routes_module, 'search_cache') and routes_module.search_cache is not None,
                        "type": type(getattr(routes_module, 'search_cache', None)).__name__ if hasattr(routes_module, 'search_cache') and routes_module.search_cache else None,
                        "id": id(getattr(routes_module, 'search_cache', None)) if hasattr(routes_module, 'search_cache') and routes_module.search_cache else None,
                        "same_as_global": (hasattr(routes_module, 'search_cache') and 
                                         routes_module.search_cache is search_cache)
                    },
                    "metrics_collector": {
                        "exists": hasattr(routes_module, 'metrics_collector') and routes_module.metrics_collector is not None,
                        "type": type(getattr(routes_module, 'metrics_collector', None)).__name__ if hasattr(routes_module, 'metrics_collector') and routes_module.metrics_collector else None,
                        "id": id(getattr(routes_module, 'metrics_collector', None)) if hasattr(routes_module, 'metrics_collector') and routes_module.metrics_collector else None,
                        "same_as_global": (hasattr(routes_module, 'metrics_collector') and 
                                         routes_module.metrics_collector is metrics_collector)
                    }
                },
                "injection_summary": {
                    "elasticsearch_injected_correctly": (
                        hasattr(routes_module, 'elastic_client') and 
                        routes_module.elastic_client is elastic_client and
                        elastic_client is not None and
                        getattr(elastic_client, '_initialized', False)
                    ),
                    "qdrant_injected_correctly": (
                        hasattr(routes_module, 'qdrant_client') and 
                        routes_module.qdrant_client is qdrant_client and
                        qdrant_client is not None and
                        getattr(qdrant_client, '_initialized', False)
                    )
                }
            }
            
        except Exception as e:
            return {
                "error": f"Erreur lors du debug des clients: {e}",
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