"""
Fonctions d'initialisation améliorées pour les services de recherche.

Ce module contient les fonctions d'initialisation corrigées pour les clients
Elasticsearch (Bonsai) et Qdrant avec gestion d'erreur robuste.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Tuple, List

from config_service.config import settings

logger = logging.getLogger("search_service.initialization")


async def initialize_elasticsearch_hybrid() -> Tuple[bool, Optional[Any], Dict[str, Any]]:
    """
    Initialise et teste la connexion Elasticsearch hybride (client standard + Bonsai HTTP).
    
    Returns:
        Tuple[bool, Optional[HybridElasticClient], Dict[str, Any]]: (succès, client, diagnostic)
    """
    logger.info("🔍 === INITIALISATION ELASTICSEARCH HYBRIDE ===")
    diagnostic = {
        "service": "elasticsearch_hybrid",
        "configured": False,
        "connected": False,
        "healthy": False,
        "error": None,
        "connection_time": None,
        "client_type": None,
        "cluster_info": {},
        "bonsai_url_format": None
    }
    
    # Vérifier la configuration BONSAI_URL
    if not settings.BONSAI_URL:
        logger.error("❌ BONSAI_URL non configurée")
        diagnostic["error"] = "BONSAI_URL not configured"
        return False, None, diagnostic
    
    diagnostic["configured"] = True
    
    # Analyser l'URL Bonsai
    bonsai_url = settings.BONSAI_URL.strip()
    
    # Vérifier le format de l'URL
    if not bonsai_url.startswith(('http://', 'https://')):
        logger.error(f"❌ Format URL invalide: {bonsai_url[:50]}...")
        diagnostic["error"] = "Invalid URL format"
        return False, None, diagnostic
    
    # Extraire les informations pour le diagnostic (sans exposer les credentials)
    if '@' in bonsai_url:
        parts = bonsai_url.split('@')
        if len(parts) == 2:
            host_part = parts[1]
            diagnostic["bonsai_url_format"] = f"https://user:pass@{host_part}"
            safe_url = f"https://***:***@{host_part}"
        else:
            safe_url = "URL malformée"
    else:
        safe_url = bonsai_url
        diagnostic["bonsai_url_format"] = "URL sans authentification"
    
    logger.info(f"🔗 Connexion à Bonsai Elasticsearch: {safe_url}")
    
    try:
        start_time = time.time()
        
        # Créer le client hybride
        from search_service.storage.elastic_client_hybrid import HybridElasticClient
        client = HybridElasticClient()
        
        logger.info("⏱️ Test de connexion Elasticsearch hybride...")
        
        # Initialiser le client (cette méthode retourne maintenant un booléen)
        initialization_success = await client.initialize()
        
        connection_time = time.time() - start_time
        diagnostic["connection_time"] = round(connection_time, 3)
        
        if initialization_success and client._initialized:
            diagnostic["connected"] = True
            diagnostic["client_type"] = client.client_type
            logger.info(f"✅ Connexion établie en {connection_time:.3f}s via {client.client_type}")
            
            # Test de santé du cluster
            logger.info("🩺 Vérification de la santé du service...")
            try:
                is_healthy = await client.is_healthy()
                diagnostic["healthy"] = is_healthy
                
                if is_healthy:
                    logger.info("💚 Service Elasticsearch sain et opérationnel")
                    
                    # Récupérer les informations pour le diagnostic
                    try:
                        client_info = client.get_client_info()
                        diagnostic["cluster_info"] = client_info
                            
                    except Exception as info_error:
                        logger.warning(f"⚠️ Impossible de récupérer les infos: {info_error}")
                    
                    return True, client, diagnostic
                else:
                    logger.error("💔 Service Elasticsearch en mauvaise santé")
                    diagnostic["error"] = "Service unhealthy"
                    return False, client, diagnostic
            except Exception as health_error:
                logger.error(f"❌ Impossible de vérifier la santé: {health_error}")
                diagnostic["error"] = f"Health check failed: {health_error}"
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


async def initialize_qdrant() -> Tuple[bool, Optional[Any], Dict[str, Any]]:
    """
    Initialise et teste la connexion Qdrant avec gestion d'erreur améliorée.
    
    Returns:
        Tuple[bool, Optional[QdrantClient], Dict[str, Any]]: (succès, client, diagnostic)
    """
    logger.info("🎯 === INITIALISATION QDRANT ===")
    diagnostic = {
        "service": "qdrant",
        "configured": False,
        "connected": False,
        "healthy": False,
        "error": None,
        "connection_time": None,
        "collections_info": {},
        "version_info": {},
        "api_key_configured": False
    }
    
    # Vérifier la configuration QDRANT_URL
    if not settings.QDRANT_URL:
        logger.error("❌ QDRANT_URL non configurée")
        diagnostic["error"] = "QDRANT_URL not configured"
        return False, None, diagnostic
    
    diagnostic["configured"] = True
    diagnostic["api_key_configured"] = bool(settings.QDRANT_API_KEY)
    
    # Log de la configuration
    qdrant_url = settings.QDRANT_URL.strip()
    logger.info(f"🔗 Connexion à Qdrant: {qdrant_url}")
    
    if settings.QDRANT_API_KEY:
        logger.info("🔑 Authentification par API Key activée")
        api_key_masked = f"{settings.QDRANT_API_KEY[:8]}...{settings.QDRANT_API_KEY[-4:]}" if len(settings.QDRANT_API_KEY) > 12 else "***"
        logger.info(f"🔐 API Key: {api_key_masked}")
    else:
        logger.info("🔓 Connexion sans authentification")
    
    try:
        start_time = time.time()
        
        # Créer le client Qdrant
        from search_service.storage.qdrant_client import QdrantClient
        client = QdrantClient()
        
        logger.info("⏱️ Test de connexion Qdrant...")
        
        # Initialiser le client (cette méthode retourne maintenant un booléen)
        initialization_success = await client.initialize()
        
        connection_time = time.time() - start_time
        diagnostic["connection_time"] = round(connection_time, 3)
        
        if initialization_success and client._initialized:
            diagnostic["connected"] = True
            logger.info(f"✅ Connexion établie en {connection_time:.3f}s")
            
            # Test de santé
            logger.info("🩺 Vérification de la santé de Qdrant...")
            try:
                is_healthy = await client.is_healthy()
                diagnostic["healthy"] = is_healthy
                
                if is_healthy:
                    logger.info("💚 Service Qdrant sain et opérationnel")
                    
                    # Récupérer les informations pour le diagnostic
                    try:
                        collection_info = await client.get_collection_info()
                        diagnostic["collections_info"] = collection_info
                        
                        # Informations sur les collections disponibles
                        if client.client:
                            collections = await client.client.get_collections()
                            collection_names = [col.name for col in collections.collections]
                            diagnostic["collections_info"]["available_collections"] = collection_names
                            diagnostic["collections_info"]["target_collection"] = client.collection_name
                            diagnostic["collections_info"]["target_exists"] = client.collection_name in collection_names
                            
                            # Informations du cluster
                            cluster_info = await client.get_cluster_info()
                            diagnostic["version_info"] = cluster_info
                            
                    except Exception as info_error:
                        logger.warning(f"⚠️ Impossible de récupérer les infos collections: {info_error}")
                    
                    return True, client, diagnostic
                else:
                    logger.error("💔 Service Qdrant en mauvaise santé")
                    diagnostic["error"] = "Service unhealthy"
                    return False, client, diagnostic
            except Exception as health_error:
                logger.error(f"❌ Impossible de vérifier la santé: {health_error}")
                diagnostic["error"] = f"Health check failed: {health_error}"
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
        
        # Diagnostic spécifique des erreurs
        error_str = str(e).lower()
        if "connection" in error_str or "timeout" in error_str:
            logger.error("🔌 DIAGNOSTIC: Problème de connectivité réseau")
            logger.error("   - Vérifiez l'URL Qdrant")
            logger.error("   - Vérifiez la connectivité réseau")
        elif "401" in str(e) or "403" in str(e) or "auth" in error_str:
            logger.error("🔑 DIAGNOSTIC: Problème d'authentification")
            logger.error("   - Vérifiez QDRANT_API_KEY")
        elif "ssl" in error_str or "certificate" in error_str:
            logger.error("🔒 DIAGNOSTIC: Problème SSL/TLS")
        
        return False, None, diagnostic


async def test_clients_connectivity() -> Dict[str, Any]:
    """
    Teste la connectivité des deux clients de manière indépendante.
    
    Returns:
        Dict[str, Any]: Rapport de test de connectivité
    """
    logger.info("🧪 === TEST DE CONNECTIVITÉ ===")
    
    test_results = {
        "timestamp": time.time(),
        "elasticsearch": {
            "tested": False,
            "success": False,
            "response_time": None,
            "error": None
        },
        "qdrant": {
            "tested": False,
            "success": False,
            "response_time": None,
            "error": None
        }
    }
    
    # Test Elasticsearch
    if settings.BONSAI_URL:
        logger.info("🔍 Test de connectivité Elasticsearch...")
        try:
            start_time = time.time()
            
            # Test basique avec aiohttp pour valider l'URL
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(settings.BONSAI_URL, timeout=10) as response:
                    response_time = time.time() - start_time
                    test_results["elasticsearch"]["tested"] = True
                    test_results["elasticsearch"]["response_time"] = round(response_time, 3)
                    test_results["elasticsearch"]["success"] = response.status == 200
                    
                    if response.status == 200:
                        logger.info(f"✅ Elasticsearch répond en {response_time:.3f}s")
                    else:
                        logger.warning(f"⚠️ Elasticsearch répond avec status {response.status}")
                        
        except Exception as e:
            test_results["elasticsearch"]["tested"] = True
            test_results["elasticsearch"]["error"] = str(e)
            logger.error(f"❌ Test Elasticsearch échoué: {e}")
    else:
        test_results["elasticsearch"]["error"] = "BONSAI_URL not configured"
        logger.warning("⚠️ BONSAI_URL non configurée - test ignoré")
    
    # Test Qdrant
    if settings.QDRANT_URL:
        logger.info("🎯 Test de connectivité Qdrant...")
        try:
            start_time = time.time()
            
            # Test basique avec aiohttp
            import aiohttp
            headers = {}
            if settings.QDRANT_API_KEY:
                headers["api-key"] = settings.QDRANT_API_KEY
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{settings.QDRANT_URL}/collections", headers=headers, timeout=10) as response:
                    response_time = time.time() - start_time
                    test_results["qdrant"]["tested"] = True
                    test_results["qdrant"]["response_time"] = round(response_time, 3)
                    test_results["qdrant"]["success"] = response.status == 200
                    
                    if response.status == 200:
                        logger.info(f"✅ Qdrant répond en {response_time:.3f}s")
                    else:
                        logger.warning(f"⚠️ Qdrant répond avec status {response.status}")
                        
        except Exception as e:
            test_results["qdrant"]["tested"] = True
            test_results["qdrant"]["error"] = str(e)
            logger.error(f"❌ Test Qdrant échoué: {e}")
    else:
        test_results["qdrant"]["error"] = "QDRANT_URL not configured"
        logger.warning("⚠️ QDRANT_URL non configurée - test ignoré")
    
    # Résumé
    elasticsearch_ok = test_results["elasticsearch"]["success"]
    qdrant_ok = test_results["qdrant"]["success"]
    
    if elasticsearch_ok and qdrant_ok:
        logger.info("🎉 Les deux services sont accessibles")
    elif elasticsearch_ok:
        logger.warning("⚠️ Seul Elasticsearch est accessible")
    elif qdrant_ok:
        logger.warning("⚠️ Seul Qdrant est accessible")
    else:
        logger.error("🚨 Aucun service n'est accessible")
    
    return test_results


def validate_environment_configuration() -> Dict[str, Any]:
    """
    Valide la configuration de l'environnement avant l'initialisation.
    
    Returns:
        Dict[str, Any]: Rapport de validation
    """
    logger.info("⚙️ === VALIDATION DE LA CONFIGURATION ===")
    
    validation = {
        "bonsai_url": {
            "configured": bool(settings.BONSAI_URL),
            "format_valid": False,
            "has_credentials": False,
            "error": None
        },
        "qdrant_url": {
            "configured": bool(settings.QDRANT_URL),
            "format_valid": False,
            "has_api_key": bool(settings.QDRANT_API_KEY),
            "error": None
        },
        "ai_services": {
            "openai_configured": bool(settings.OPENAI_API_KEY),
            "cohere_configured": bool(getattr(settings, 'COHERE_KEY', None)),
            "deepseek_configured": bool(getattr(settings, 'DEEPSEEK_API_KEY', None))
        },
        "summary": {
            "ready_for_elasticsearch": False,
            "ready_for_qdrant": False,
            "critical_issues": []
        }
    }
    
    # Validation BONSAI_URL
    if settings.BONSAI_URL:
        url = settings.BONSAI_URL.strip()
        if url.startswith(('http://', 'https://')):
            validation["bonsai_url"]["format_valid"] = True
            validation["bonsai_url"]["has_credentials"] = '@' in url
            validation["summary"]["ready_for_elasticsearch"] = True
            logger.info("✅ BONSAI_URL configurée correctement")
        else:
            validation["bonsai_url"]["error"] = "Format URL invalide"
            validation["summary"]["critical_issues"].append("BONSAI_URL format invalide")
            logger.error("❌ BONSAI_URL format invalide")
    else:
        validation["bonsai_url"]["error"] = "Non configurée"
        validation["summary"]["critical_issues"].append("BONSAI_URL manquante")
        logger.error("❌ BONSAI_URL non configurée")
    
    # Validation QDRANT_URL
    if settings.QDRANT_URL:
        url = settings.QDRANT_URL.strip()
        if url.startswith(('http://', 'https://')):
            validation["qdrant_url"]["format_valid"] = True
            validation["summary"]["ready_for_qdrant"] = True
            logger.info("✅ QDRANT_URL configurée correctement")
            
            if settings.QDRANT_API_KEY:
                logger.info("✅ QDRANT_API_KEY configurée")
            else:
                logger.warning("⚠️ QDRANT_API_KEY non configurée (connexion non sécurisée)")
        else:
            validation["qdrant_url"]["error"] = "Format URL invalide"
            validation["summary"]["critical_issues"].append("QDRANT_URL format invalide")
            logger.error("❌ QDRANT_URL format invalide")
    else:
        validation["qdrant_url"]["error"] = "Non configurée"
        validation["summary"]["critical_issues"].append("QDRANT_URL manquante")
        logger.error("❌ QDRANT_URL non configurée")
    
    # Validation des services IA
    if validation["ai_services"]["openai_configured"]:
        logger.info("✅ OpenAI API configurée (embeddings disponibles)")
    else:
        logger.warning("⚠️ OpenAI API non configurée (pas d'embeddings)")
    
    if validation["ai_services"]["cohere_configured"]:
        logger.info("✅ Cohere API configurée (reranking disponible)")
    else:
        logger.warning("⚠️ Cohere API non configurée (pas de reranking)")
    
    if validation["ai_services"]["deepseek_configured"]:
        logger.info("✅ DeepSeek API configurée")
    else:
        logger.warning("⚠️ DeepSeek API non configurée")
    
    # Résumé
    if validation["summary"]["ready_for_elasticsearch"] and validation["summary"]["ready_for_qdrant"]:
        logger.info("🎉 Configuration valide pour les deux services")
    elif validation["summary"]["critical_issues"]:
        logger.error(f"🚨 Problèmes critiques détectés: {len(validation['summary']['critical_issues'])}")
        for issue in validation["summary"]["critical_issues"]:
            logger.error(f"   - {issue}")
    
    return validation


async def initialize_search_clients() -> Dict[str, Any]:
    """
    Initialise les clients de recherche avec gestion d'erreur complète.
    
    Returns:
        Dict[str, Any]: Rapport d'initialisation complet
    """
    logger.info("🚀 === INITIALISATION DES CLIENTS DE RECHERCHE ===")
    
    initialization_report = {
        "timestamp": time.time(),
        "validation": {},
        "connectivity_test": {},
        "elasticsearch": {},
        "qdrant": {},
        "summary": {
            "elasticsearch_ready": False,
            "qdrant_ready": False,
            "total_ready": 0,
            "status": "FAILED"
        }
    }
    
    try:
        # 1. Validation de la configuration
        logger.info("📋 Étape 1: Validation de la configuration")
        validation = validate_environment_configuration()
        initialization_report["validation"] = validation
        
        if validation["summary"]["critical_issues"]:
            logger.error("❌ Arrêt à cause de problèmes de configuration critiques")
            return initialization_report
        
        # 2. Test de connectivité basique
        logger.info("🌐 Étape 2: Test de connectivité basique")
        connectivity_test = await test_clients_connectivity()
        initialization_report["connectivity_test"] = connectivity_test
        
        # 3. Initialisation d'Elasticsearch
        logger.info("🔍 Étape 3: Initialisation d'Elasticsearch")
        elastic_success, elastic_client, elastic_diag = await initialize_elasticsearch_hybrid()
        initialization_report["elasticsearch"] = {
            "success": elastic_success,
            "client_initialized": elastic_client is not None,
            "diagnostic": elastic_diag
        }
        initialization_report["summary"]["elasticsearch_ready"] = elastic_success
        
        # 4. Initialisation de Qdrant
        logger.info("🎯 Étape 4: Initialisation de Qdrant")
        qdrant_success, qdrant_client, qdrant_diag = await initialize_qdrant()
        initialization_report["qdrant"] = {
            "success": qdrant_success,
            "client_initialized": qdrant_client is not None,
            "diagnostic": qdrant_diag
        }
        initialization_report["summary"]["qdrant_ready"] = qdrant_success
        
        # 5. Calcul du statut final
        ready_count = sum([
            initialization_report["summary"]["elasticsearch_ready"],
            initialization_report["summary"]["qdrant_ready"]
        ])
        initialization_report["summary"]["total_ready"] = ready_count
        
        if ready_count == 2:
            initialization_report["summary"]["status"] = "OPTIMAL"
            logger.info("🎉 Initialisation OPTIMALE: Tous les services prêts")
        elif ready_count == 1:
            initialization_report["summary"]["status"] = "DEGRADED"
            logger.warning("⚠️ Initialisation DÉGRADÉE: Service partiel")
        else:
            initialization_report["summary"]["status"] = "FAILED"
            logger.error("🚨 Initialisation ÉCHOUÉE: Aucun service prêt")
        
        # 6. Retourner les clients initialisés
        initialization_report["clients"] = {
            "elasticsearch": elastic_client if elastic_success else None,
            "qdrant": qdrant_client if qdrant_success else None
        }
        
        return initialization_report
        
    except Exception as e:
        logger.error(f"💥 Erreur lors de l'initialisation: {e}", exc_info=True)
        initialization_report["summary"]["status"] = "ERROR"
        initialization_report["error"] = str(e)
        return initialization_report


def log_initialization_summary(report: Dict[str, Any]):
    """
    Affiche un résumé détaillé de l'initialisation.
    
    Args:
        report: Rapport d'initialisation
    """
    logger.info("=" * 100)
    logger.info("📊 RÉSUMÉ DE L'INITIALISATION")
    logger.info("=" * 100)
    
    # Statut global
    status = report["summary"]["status"]
    status_icons = {
        "OPTIMAL": "🎉",
        "DEGRADED": "⚠️",
        "FAILED": "🚨",
        "ERROR": "💥"
    }
    icon = status_icons.get(status, "❓")
    logger.info(f"{icon} Statut global: {status}")
    
    # Détails par service
    logger.info("📋 Détails par service:")
    
    # Elasticsearch
    elastic_ready = report["summary"]["elasticsearch_ready"]
    elastic_icon = "✅" if elastic_ready else "❌"
    logger.info(f"   {elastic_icon} Elasticsearch: {'PRÊT' if elastic_ready else 'ÉCHEC'}")
    
    if "elasticsearch" in report and "diagnostic" in report["elasticsearch"]:
        diag = report["elasticsearch"]["diagnostic"]
        if diag.get("connection_time"):
            logger.info(f"      ⏱️ Temps de connexion: {diag['connection_time']}s")
        if diag.get("client_type"):
            logger.info(f"      🔧 Client utilisé: {diag['client_type']}")
        if diag.get("error"):
            logger.info(f"      ❌ Erreur: {diag['error']}")
    
    # Qdrant
    qdrant_ready = report["summary"]["qdrant_ready"]
    qdrant_icon = "✅" if qdrant_ready else "❌"
    logger.info(f"   {qdrant_icon} Qdrant: {'PRÊT' if qdrant_ready else 'ÉCHEC'}")
    
    if "qdrant" in report and "diagnostic" in report["qdrant"]:
        diag = report["qdrant"]["diagnostic"]
        if diag.get("connection_time"):
            logger.info(f"      ⏱️ Temps de connexion: {diag['connection_time']}s")
        if diag.get("collections_info", {}).get("available_collections"):
            collections = diag["collections_info"]["available_collections"]
            logger.info(f"      📚 Collections: {len(collections)} disponibles")
        if diag.get("error"):
            logger.info(f"      ❌ Erreur: {diag['error']}")
    
    # Capacités disponibles
    logger.info("🔧 Capacités disponibles:")
    if elastic_ready and qdrant_ready:
        logger.info("   ✅ Recherche hybride (lexicale + sémantique)")
        logger.info("   ✅ Reranking intelligent")
        logger.info("   ✅ Toutes les fonctionnalités")
    elif elastic_ready:
        logger.info("   ✅ Recherche lexicale uniquement")
        logger.info("   ❌ Recherche sémantique indisponible")
    elif qdrant_ready:
        logger.info("   ❌ Recherche lexicale indisponible")
        logger.info("   ✅ Recherche sémantique uniquement")
    else:
        logger.info("   ❌ Aucune recherche disponible")
    
    # Recommandations
    logger.info("💡 Recommandations:")
    if status == "OPTIMAL":
        logger.info("   🎯 Service opérationnel - Aucune action requise")
    elif status == "DEGRADED":
        logger.info("   🔧 Vérifier la configuration du service défaillant")
        logger.info("   📞 Contacter l'équipe infrastructure si nécessaire")
    elif status == "FAILED":
        logger.info("   🚨 Vérifier les variables d'environnement")
        logger.info("   🌐 Tester la connectivité réseau")
        logger.info("   🔑 Valider les credentials d'authentification")
    
    logger.info("=" * 100)


async def create_collections_if_needed(qdrant_client) -> bool:
    """
    Crée les collections Qdrant nécessaires si elles n'existent pas.
    
    Args:
        qdrant_client: Client Qdrant initialisé
        
    Returns:
        bool: Succès de la création/vérification
    """
    if not qdrant_client or not qdrant_client._initialized:
        logger.error("❌ Client Qdrant non disponible pour la création de collections")
        return False
    
    logger.info("🔧 === CRÉATION DES COLLECTIONS QDRANT ===")
    
    try:
        # Vérifier si la collection principale existe
        exists = await qdrant_client.collection_exists()
        
        if not exists:
            logger.info(f"🏗️ Création de la collection '{qdrant_client.collection_name}'...")
            
            # Créer la collection avec les paramètres par défaut
            success = await qdrant_client.create_collection_if_not_exists(
                vector_size=1536,  # Taille pour text-embedding-3-small d'OpenAI
                distance_metric="Cosine"
            )
            
            if success:
                logger.info("✅ Collection créée avec succès")
                return True
            else:
                logger.error("❌ Échec de la création de la collection")
                return False
        else:
            logger.info(f"✅ Collection '{qdrant_client.collection_name}' existe déjà")
            return True
            
    except Exception as e:
        logger.error(f"💥 Erreur lors de la création des collections: {e}")
        return False


async def validate_clients_functionality(elastic_client, qdrant_client) -> Dict[str, Any]:
    """
    Valide la fonctionnalité des clients avec des tests simples.
    
    Args:
        elastic_client: Client Elasticsearch initialisé
        qdrant_client: Client Qdrant initialisé
        
    Returns:
        Dict[str, Any]: Rapport de validation
    """
    logger.info("🧪 === VALIDATION DE LA FONCTIONNALITÉ ===")
    
    validation_results = {
        "timestamp": time.time(),
        "elasticsearch": {
            "tested": False,
            "functional": False,
            "capabilities": [],
            "error": None
        },
        "qdrant": {
            "tested": False,
            "functional": False,
            "capabilities": [],
            "error": None
        },
        "overall_status": "FAILED"
    }
    
    # Test Elasticsearch
    if elastic_client and elastic_client._initialized:
        logger.info("🔍 Test fonctionnalité Elasticsearch...")
        validation_results["elasticsearch"]["tested"] = True
        
        try:
            # Test de santé
            is_healthy = await elastic_client.is_healthy()
            if is_healthy:
                validation_results["elasticsearch"]["capabilities"].append("health_check")
            
            # Test d'informations sur l'index
            index_info = await elastic_client.get_index_info()
            if not index_info.get("error"):
                validation_results["elasticsearch"]["capabilities"].append("index_management")
            
            # Test de comptage (même si l'index est vide)
            count = await elastic_client.count_documents()
            if count >= 0:  # Même 0 est valide
                validation_results["elasticsearch"]["capabilities"].append("document_counting")
            
            # Si au moins une capacité fonctionne
            if validation_results["elasticsearch"]["capabilities"]:
                validation_results["elasticsearch"]["functional"] = True
                logger.info("✅ Elasticsearch fonctionnel")
            else:
                logger.warning("⚠️ Elasticsearch partiellement fonctionnel")
                
        except Exception as e:
            validation_results["elasticsearch"]["error"] = str(e)
            logger.error(f"❌ Test Elasticsearch échoué: {e}")
    else:
        logger.warning("⚠️ Elasticsearch non disponible pour les tests")
    
    # Test Qdrant
    if qdrant_client and qdrant_client._initialized:
        logger.info("🎯 Test fonctionnalité Qdrant...")
        validation_results["qdrant"]["tested"] = True
        
        try:
            # Test de santé
            is_healthy = await qdrant_client.is_healthy()
            if is_healthy:
                validation_results["qdrant"]["capabilities"].append("health_check")
            
            # Test de liste des collections
            collections = await qdrant_client.get_collections_list()
            if isinstance(collections, list):
                validation_results["qdrant"]["capabilities"].append("collection_management")
            
            # Test d'existence de collection
            exists = await qdrant_client.collection_exists()
            validation_results["qdrant"]["capabilities"].append("collection_verification")
            
            # Test de comptage (même si la collection est vide)
            if exists:
                count = await qdrant_client.count_points()
                if count >= 0:  # Même 0 est valide
                    validation_results["qdrant"]["capabilities"].append("point_counting")
            
            # Si au moins une capacité fonctionne
            if validation_results["qdrant"]["capabilities"]:
                validation_results["qdrant"]["functional"] = True
                logger.info("✅ Qdrant fonctionnel")
            else:
                logger.warning("⚠️ Qdrant partiellement fonctionnel")
                
        except Exception as e:
            validation_results["qdrant"]["error"] = str(e)
            logger.error(f"❌ Test Qdrant échoué: {e}")
    else:
        logger.warning("⚠️ Qdrant non disponible pour les tests")
    
    # Déterminer le statut global
    elasticsearch_ok = validation_results["elasticsearch"]["functional"]
    qdrant_ok = validation_results["qdrant"]["functional"]
    
    if elasticsearch_ok and qdrant_ok:
        validation_results["overall_status"] = "OPTIMAL"
        logger.info("🎉 Tous les clients sont fonctionnels")
    elif elasticsearch_ok or qdrant_ok:
        validation_results["overall_status"] = "DEGRADED"
        logger.warning("⚠️ Au moins un client est fonctionnel")
    else:
        validation_results["overall_status"] = "FAILED"
        logger.error("🚨 Aucun client n'est fonctionnel")
    
    return validation_results


async def run_startup_diagnostics() -> Dict[str, Any]:
    """
    Lance un diagnostic complet au démarrage.
    
    Returns:
        Dict[str, Any]: Rapport de diagnostic complet
    """
    logger.info("🔬 === DIAGNOSTIC COMPLET DE DÉMARRAGE ===")
    
    diagnostic_start = time.time()
    
    # Rapport principal
    full_diagnostic = {
        "timestamp": time.time(),
        "version": "2.0.0",
        "environment": {
            "python_version": None,
            "dependencies": {},
            "system_info": {}
        },
        "configuration_validation": {},
        "connectivity_tests": {},
        "client_initialization": {},
        "functionality_validation": {},
        "recommendations": [],
        "duration_seconds": 0,
        "overall_status": "UNKNOWN"
    }
    
    try:
        # 1. Informations système
        import sys
        import platform
        
        full_diagnostic["environment"]["python_version"] = sys.version
        full_diagnostic["environment"]["system_info"] = {
            "platform": platform.platform(),
            "architecture": platform.architecture(),
            "processor": platform.processor()
        }
        
        # 2. Vérification des dépendances
        dependencies_status = {}
        required_packages = ["elasticsearch", "qdrant_client", "aiohttp", "fastapi"]
        
        for package in required_packages:
            try:
                __import__(package)
                dependencies_status[package] = "available"
            except ImportError:
                dependencies_status[package] = "missing"
        
        full_diagnostic["environment"]["dependencies"] = dependencies_status
        
        # 3. Validation de la configuration
        config_validation = validate_environment_configuration()
        full_diagnostic["configuration_validation"] = config_validation
        
        # 4. Tests de connectivité
        connectivity_tests = await test_clients_connectivity()
        full_diagnostic["connectivity_tests"] = connectivity_tests
        
        # 5. Initialisation des clients
        client_init = await initialize_search_clients()
        full_diagnostic["client_initialization"] = client_init
        
        # 6. Validation de la fonctionnalité
        if client_init.get("clients"):
            elastic_client = client_init["clients"].get("elasticsearch")
            qdrant_client = client_init["clients"].get("qdrant")
            
            functionality_validation = await validate_clients_functionality(elastic_client, qdrant_client)
            full_diagnostic["functionality_validation"] = functionality_validation
        
        # 7. Génération des recommandations
        recommendations = generate_recommendations(full_diagnostic)
        full_diagnostic["recommendations"] = recommendations
        
        # 8. Détermination du statut global
        overall_status = determine_overall_status(full_diagnostic)
        full_diagnostic["overall_status"] = overall_status
        
        # 9. Durée totale
        diagnostic_duration = time.time() - diagnostic_start
        full_diagnostic["duration_seconds"] = round(diagnostic_duration, 2)
        
        logger.info(f"✅ Diagnostic complet terminé en {diagnostic_duration:.2f}s")
        logger.info(f"📊 Statut global: {overall_status}")
        
        return full_diagnostic
        
    except Exception as e:
        logger.error(f"💥 Erreur lors du diagnostic: {e}", exc_info=True)
        full_diagnostic["error"] = str(e)
        full_diagnostic["overall_status"] = "ERROR"
        full_diagnostic["duration_seconds"] = round(time.time() - diagnostic_start, 2)
        return full_diagnostic


def generate_recommendations(diagnostic: Dict[str, Any]) -> List[str]:
    """
    Génère des recommandations basées sur le diagnostic.
    
    Args:
        diagnostic: Rapport de diagnostic complet
        
    Returns:
        List[str]: Liste de recommandations
    """
    recommendations = []
    
    # Recommandations de configuration
    config_validation = diagnostic.get("configuration_validation", {})
    if config_validation.get("summary", {}).get("critical_issues"):
        recommendations.append("🔧 Corriger les problèmes de configuration critiques")
        recommendations.append("📋 Vérifier les variables d'environnement BONSAI_URL et QDRANT_URL")
    
    # Recommandations de connectivité
    connectivity = diagnostic.get("connectivity_tests", {})
    if not connectivity.get("elasticsearch", {}).get("success"):
        recommendations.append("🌐 Vérifier la connectivité réseau vers Elasticsearch/Bonsai")
        recommendations.append("🔑 Valider les credentials d'authentification Bonsai")
    
    if not connectivity.get("qdrant", {}).get("success"):
        recommendations.append("🌐 Vérifier la connectivité réseau vers Qdrant")
        recommendations.append("🔑 Valider QDRANT_API_KEY si requis")
    
    # Recommandations d'initialisation
    client_init = diagnostic.get("client_initialization", {})
    if not client_init.get("summary", {}).get("elasticsearch_ready"):
        recommendations.append("🔍 Investiguer les problèmes d'initialisation Elasticsearch")
    
    if not client_init.get("summary", {}).get("qdrant_ready"):
        recommendations.append("🎯 Investiguer les problèmes d'initialisation Qdrant")
        recommendations.append("📚 Vérifier que les collections Qdrant sont créées")
    
    # Recommandations de fonctionnalité
    functionality = diagnostic.get("functionality_validation", {})
    if functionality.get("overall_status") == "DEGRADED":
        recommendations.append("⚙️ Tester les fonctionnalités en mode dégradé")
        recommendations.append("📈 Surveiller les performances avec un seul moteur")
    
    # Recommandations générales
    if not recommendations:
        recommendations.append("✅ Configuration optimale - aucune action requise")
        recommendations.append("📊 Surveiller les métriques de performance")
        recommendations.append("🔄 Effectuer des tests périodiques")
    
    return recommendations


def determine_overall_status(diagnostic: Dict[str, Any]) -> str:
    """
    Détermine le statut global basé sur tous les tests.
    
    Args:
        diagnostic: Rapport de diagnostic complet
        
    Returns:
        str: Statut global (OPTIMAL, DEGRADED, FAILED, ERROR)
    """
    # Vérifier s'il y a eu une erreur
    if "error" in diagnostic:
        return "ERROR"
    
    # Vérifier l'initialisation des clients
    client_init = diagnostic.get("client_initialization", {})
    init_status = client_init.get("summary", {}).get("status", "FAILED")
    
    # Vérifier la fonctionnalité
    functionality = diagnostic.get("functionality_validation", {})
    func_status = functionality.get("overall_status", "FAILED")
    
    # Combiner les statuts
    if init_status == "OPTIMAL" and func_status == "OPTIMAL":
        return "OPTIMAL"
    elif init_status in ["OPTIMAL", "DEGRADED"] and func_status in ["OPTIMAL", "DEGRADED"]:
        return "DEGRADED"
    elif init_status == "ERROR" or func_status == "ERROR":
        return "ERROR"
    else:
        return "FAILED"


def log_diagnostic_summary(diagnostic: Dict[str, Any]):
    """
    Affiche un résumé du diagnostic complet.
    
    Args:
        diagnostic: Rapport de diagnostic complet
    """
    logger.info("=" * 120)
    logger.info("🔬 RAPPORT DE DIAGNOSTIC COMPLET")
    logger.info("=" * 120)
    
    # En-tête
    status = diagnostic.get("overall_status", "UNKNOWN")
    duration = diagnostic.get("duration_seconds", 0)
    
    status_icons = {
        "OPTIMAL": "🎉",
        "DEGRADED": "⚠️",
        "FAILED": "🚨",
        "ERROR": "💥",
        "UNKNOWN": "❓"
    }
    
    icon = status_icons.get(status, "❓")
    logger.info(f"{icon} Statut global: {status}")
    logger.info(f"⏱️ Durée du diagnostic: {duration}s")
    logger.info(f"📅 Timestamp: {diagnostic.get('timestamp', 'N/A')}")
    
    # Environnement
    env = diagnostic.get("environment", {})
    if env:
        logger.info("🖥️ Environnement:")
        logger.info(f"   🐍 Python: {env.get('python_version', 'N/A')}")
        
        deps = env.get("dependencies", {})
        for pkg, status in deps.items():
            dep_icon = "✅" if status == "available" else "❌"
            logger.info(f"   {dep_icon} {pkg}: {status}")
    
    # Configuration
    config = diagnostic.get("configuration_validation", {})
    if config:
        issues = config.get("summary", {}).get("critical_issues", [])
        if issues:
            logger.info("❌ Problèmes de configuration:")
            for issue in issues:
                logger.info(f"   - {issue}")
        else:
            logger.info("✅ Configuration valide")
    
    # Connectivité
    connectivity = diagnostic.get("connectivity_tests", {})
    if connectivity:
        logger.info("🌐 Tests de connectivité:")
        
        for service in ["elasticsearch", "qdrant"]:
            test = connectivity.get(service, {})
            if test.get("tested"):
                success = test.get("success", False)
                time_ms = test.get("response_time", 0) * 1000
                icon = "✅" if success else "❌"
                logger.info(f"   {icon} {service.capitalize()}: {time_ms:.0f}ms")
            else:
                logger.info(f"   ⚪ {service.capitalize()}: Non testé")
    
    # Recommandations
    recommendations = diagnostic.get("recommendations", [])
    if recommendations:
        logger.info("💡 Recommandations:")
        for rec in recommendations[:5]:  # Limiter à 5 recommandations
            logger.info(f"   {rec}")
        
        if len(recommendations) > 5:
            logger.info(f"   ... et {len(recommendations) - 5} autres recommandations")
    
    logger.info("=" * 120)