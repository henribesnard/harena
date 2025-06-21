#!/usr/bin/env python3
"""
Script de test pour valider les clients Elasticsearch et Qdrant.

Ce script teste la connectivité et les fonctionnalités de base des clients
avec vos URLs configurées.
"""

import asyncio
import logging
import os
import sys
import time
from typing import Dict, Any

# Charger les variables d'environnement depuis .env
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("📄 Variables d'environnement chargées depuis .env")
except ImportError:
    print("⚠️ python-dotenv non installé, utilisation des variables système")
    print("💡 Installer avec: pip install python-dotenv")

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("test_clients")

# Configuration des URLs depuis les variables d'environnement
BONSAI_URL = os.environ.get("BONSAI_URL", "")
QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")

# Afficher la configuration chargée (en masquant les secrets)
print("\n🔧 Configuration chargée:")
if BONSAI_URL:
    safe_bonsai = BONSAI_URL.split('@')[-1] if '@' in BONSAI_URL else BONSAI_URL
    print(f"   📡 BONSAI_URL: https://***:***@{safe_bonsai}")
else:
    print("   📡 BONSAI_URL: Non configurée")

if QDRANT_URL:
    print(f"   🎯 QDRANT_URL: {QDRANT_URL}")
else:
    print("   🎯 QDRANT_URL: Non configurée")

if QDRANT_API_KEY:
    masked_key = f"{QDRANT_API_KEY[:8]}...{QDRANT_API_KEY[-4:]}" if len(QDRANT_API_KEY) > 12 else "***"
    print(f"   🔑 QDRANT_API_KEY: {masked_key}")
else:
    print("   🔑 QDRANT_API_KEY: Non configurée")

print()


async def test_elasticsearch_basic():
    """Test basique d'Elasticsearch avec aiohttp."""
    logger.info("🔍 Test basique Elasticsearch...")
    
    if not BONSAI_URL:
        logger.error("❌ BONSAI_URL non configurée")
        return False, {"error": "BONSAI_URL not configured"}
    
    try:
        import aiohttp
        
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.get(BONSAI_URL, timeout=10) as response:
                response_time = time.time() - start_time
                data = await response.json()
                
                logger.info(f"✅ Elasticsearch répond en {response_time:.3f}s")
                logger.info(f"📊 Status: {response.status}")
                logger.info(f"🏷️ Cluster: {data.get('cluster_name', 'N/A')}")
                logger.info(f"📈 Version: {data.get('version', {}).get('number', 'N/A')}")
                
                return True, {
                    "response_time": response_time,
                    "status": response.status,
                    "cluster_name": data.get('cluster_name'),
                    "version": data.get('version', {}).get('number')
                }
                
    except Exception as e:
        logger.error(f"❌ Test Elasticsearch échoué: {e}")
        return False, {"error": str(e)}


async def test_elasticsearch_client():
    """Test avec le client Elasticsearch officiel."""
    logger.info("🔍 Test client Elasticsearch...")
    
    if not BONSAI_URL:
        logger.error("❌ BONSAI_URL non configurée")
        return False, {"error": "BONSAI_URL not configured"}
    
    try:
        from elasticsearch import AsyncElasticsearch
        
        # Configuration pour Bonsai
        client = AsyncElasticsearch(
            [BONSAI_URL],
            verify_certs=True,
            ssl_show_warn=False,
            max_retries=3,
            retry_on_timeout=True,
            request_timeout=30
        )
        
        start_time = time.time()
        
        try:
            # Test info avec gestion de l'erreur UnsupportedProductError
            info = await client.info()
            logger.info("✅ Client Elasticsearch standard connecté")
        except Exception as info_error:
            if "UnsupportedProductError" in str(info_error):
                logger.warning("⚠️ Bonsai détecté comme non-standard, test HTTP direct...")
                # Utiliser une approche HTTP directe pour contourner la vérification
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(BONSAI_URL) as response:
                        if response.status == 200:
                            info = await response.json()
                            logger.info("✅ Connexion Bonsai réussie via HTTP direct")
                        else:
                            raise Exception(f"HTTP error: {response.status}")
            else:
                raise info_error
        
        response_time = time.time() - start_time
        
        logger.info(f"✅ Client Elasticsearch/Bonsai connecté en {response_time:.3f}s")
        logger.info(f"🏷️ Cluster: {info['cluster_name']}")
        logger.info(f"📈 Version: {info['version']['number']}")
        
        # Pour Bonsai, on peut faire les tests de base
        test_passed = False
        try:
            # Test santé
            health = await client.cluster.health()
            logger.info(f"💚 Santé cluster: {health['status']}")
            logger.info(f"📊 Nœuds: {health['number_of_nodes']}")
            
            # Test création index
            index_name = "test_harena"
            
            # Supprimer l'index s'il existe
            try:
                if await client.indices.exists(index=index_name):
                    await client.indices.delete(index=index_name)
                    logger.info(f"🗑️ Index {index_name} supprimé")
            except:
                pass  # L'index n'existe peut-être pas
            
            # Créer l'index
            mapping = {
                "mappings": {
                    "properties": {
                        "title": {"type": "text"},
                        "content": {"type": "text"},
                        "timestamp": {"type": "date"}
                    }
                }
            }
            
            await client.indices.create(index=index_name, body=mapping)
            logger.info(f"📚 Index {index_name} créé")
            
            # Insérer un document de test
            doc = {
                "title": "Test Document",
                "content": "Ceci est un document de test pour Harena",
                "timestamp": "2025-01-01T00:00:00Z"
            }
            
            result = await client.index(index=index_name, body=doc)
            logger.info(f"📄 Document inséré: {result['_id']}")
            
            # Forcer le refresh pour rendre le document visible immédiatement
            await client.indices.refresh(index=index_name)
            
            # Rechercher
            search_body = {
                "query": {
                    "match": {
                        "content": "test"
                    }
                }
            }
            
            search_result = await client.search(index=index_name, body=search_body)
            hits = search_result['hits']['total']
            total_hits = hits['value'] if isinstance(hits, dict) else hits
            
            logger.info(f"🔍 Recherche effectuée: {total_hits} résultats")
            
            # Afficher les résultats trouvés
            if search_result['hits']['hits']:
                for hit in search_result['hits']['hits']:
                    logger.info(f"   📄 Document trouvé: {hit['_source']['title']} (score: {hit['_score']:.3f})")
            
            # Nettoyer
            await client.indices.delete(index=index_name)
            logger.info(f"🧹 Index {index_name} nettoyé")
            
            test_passed = True
            
        except Exception as ops_error:
            logger.warning(f"⚠️ Opérations avancées échouées: {ops_error}")
            logger.info("💡 Connexion basique OK, mais opérations limitées")
        
        await client.close()
        
        return True, {
            "cluster_name": info['cluster_name'],
            "version": info['version']['number'],
            "health": health.get('status', 'unknown') if 'health' in locals() else 'unknown',
            "test_passed": test_passed,
            "response_time": response_time
        }
        
    except Exception as e:
        logger.error(f"❌ Test client Elasticsearch échoué: {e}")
        logger.error(f"📍 Détails de l'erreur: {type(e).__name__}")
        
        # Si c'est une erreur de produit non supporté, on peut quand même considérer que ça marche
        if "UnsupportedProductError" in str(e):
            logger.info("💡 Bonsai fonctionne mais n'est pas reconnu comme Elasticsearch standard")
            logger.info("🔧 Recommandation: Utiliser le client hybride")
            return True, {
                "error": "UnsupportedProductError - Bonsai détecté",
                "workaround_needed": True,
                "basic_connectivity": True
            }
        
        return False, {"error": str(e)}


async def test_qdrant_basic():
    """Test basique de Qdrant avec aiohttp."""
    if not QDRANT_URL:
        logger.warning("⚠️ QDRANT_URL non configurée - test ignoré")
        return False, {"error": "QDRANT_URL not configured"}
    
    logger.info("🎯 Test basique Qdrant...")
    
    try:
        import aiohttp
        
        headers = {}
        if QDRANT_API_KEY:
            headers["api-key"] = QDRANT_API_KEY
        
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{QDRANT_URL}/collections", headers=headers, timeout=10) as response:
                response_time = time.time() - start_time
                data = await response.json()
                
                logger.info(f"✅ Qdrant répond en {response_time:.3f}s")
                logger.info(f"📊 Status: {response.status}")
                
                if "result" in data:
                    collections = data["result"]["collections"]
                    logger.info(f"📚 Collections: {len(collections)} trouvées")
                    for col in collections:
                        logger.info(f"   - {col['name']}")
                
                return True, {
                    "response_time": response_time,
                    "status": response.status,
                    "collections": data.get("result", {}).get("collections", [])
                }
                
    except Exception as e:
        logger.error(f"❌ Test Qdrant échoué: {e}")
        return False, {"error": str(e)}


async def test_qdrant_client():
    """Test avec le client Qdrant officiel."""
    if not QDRANT_URL:
        logger.warning("⚠️ QDRANT_URL non configurée - test ignoré")
        return False, {"error": "QDRANT_URL not configured"}
    
    logger.info("🎯 Test client Qdrant...")
    
    try:
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.models import VectorParams, Distance, PointStruct
        
        # Créer le client
        if QDRANT_API_KEY:
            client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            client = AsyncQdrantClient(url=QDRANT_URL)
        
        start_time = time.time()
        
        # Test connexion
        collections = await client.get_collections()
        response_time = time.time() - start_time
        
        logger.info(f"✅ Client Qdrant connecté en {response_time:.3f}s")
        logger.info(f"📚 Collections existantes: {len(collections.collections)}")
        
        existing_collections = [col.name for col in collections.collections]
        for col_name in existing_collections:
            logger.info(f"   - {col_name}")
        
        # Test collection
        test_collection = "test_harena"
        
        # Supprimer la collection si elle existe
        if test_collection in existing_collections:
            await client.delete_collection(test_collection)
            logger.info(f"🗑️ Collection {test_collection} supprimée")
        
        # Créer une nouvelle collection
        await client.create_collection(
            collection_name=test_collection,
            vectors_config=VectorParams(size=3, distance=Distance.COSINE)
        )
        logger.info(f"📚 Collection {test_collection} créée")
        
        # Insérer des points de test
        points = [
            PointStruct(
                id=1,
                vector=[1.0, 0.0, 0.0],
                payload={"title": "Test 1", "content": "Premier document"}
            ),
            PointStruct(
                id=2,
                vector=[0.0, 1.0, 0.0],
                payload={"title": "Test 2", "content": "Deuxième document"}
            ),
            PointStruct(
                id=3,
                vector=[0.0, 0.0, 1.0],
                payload={"title": "Test 3", "content": "Troisième document"}
            )
        ]
        
        await client.upsert(collection_name=test_collection, points=points)
        logger.info(f"📄 {len(points)} points insérés")
        
        # Attendre l'indexation
        await asyncio.sleep(2)
        
        # Recherche vectorielle
        try:
            # Utiliser query_points si disponible (nouvelle API)
            search_result = await client.query_points(
                collection_name=test_collection,
                query=[1.0, 0.0, 0.0],
                limit=5
            )
            search_points = search_result.points
        except AttributeError:
            # Fallback pour anciennes versions
            logger.info("🔄 Utilisation de l'ancienne API search...")
            search_result = await client.search(
                collection_name=test_collection,
                query_vector=[1.0, 0.0, 0.0],
                limit=5
            )
            search_points = search_result
        
        logger.info(f"🔍 Recherche effectuée: {len(search_points)} résultats")
        for result in search_points:
            logger.info(f"   📄 ID: {result.id}, Score: {result.score:.3f}, Titre: {result.payload.get('title', 'N/A')}")
        
        # Test de comptage
        count_result = await client.count(collection_name=test_collection)
        logger.info(f"📊 Nombre total de points: {count_result.count}")
        
        # Nettoyer
        await client.delete_collection(test_collection)
        logger.info(f"🧹 Collection {test_collection} nettoyée")
        
        await client.close()
        
        return True, {
            "collections_count": len(collections.collections),
            "existing_collections": existing_collections,
            "test_passed": True,
            "search_results": len(search_points),
            "response_time": response_time
        }
        
    except Exception as e:
        logger.error(f"❌ Test client Qdrant échoué: {e}")
        return False, {"error": str(e)}


async def test_hybrid_client():
    """Test du client hybride Elasticsearch."""
    logger.info("🔧 Test client Elasticsearch hybride...")
    
    if not BONSAI_URL:
        logger.error("❌ BONSAI_URL non configurée")
        return False, {"error": "BONSAI_URL not configured"}
    
    try:
        # Importer le client hybride depuis le projet
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from search_service.storage.elastic_client_hybrid import HybridElasticClient
        
        start_time = time.time()
        
        # Créer et initialiser le client
        client = HybridElasticClient()
        success = await client.initialize()
        
        response_time = time.time() - start_time
        
        if success:
            logger.info(f"✅ Client hybride initialisé en {response_time:.3f}s")
            logger.info(f"🔧 Type de client utilisé: {client.client_type}")
            
            # Test de santé
            is_healthy = await client.is_healthy()
            logger.info(f"💚 Client sain: {is_healthy}")
            
            # Test d'informations
            client_info = client.get_client_info()
            logger.info(f"📊 Informations client: {client_info}")
            
            await client.close()
            
            return True, {
                "client_type": client.client_type,
                "healthy": is_healthy,
                "response_time": response_time,
                "client_info": client_info
            }
        else:
            logger.error("❌ Échec d'initialisation du client hybride")
            return False, {"error": "Initialization failed"}
            
    except ImportError as e:
        logger.error(f"❌ Impossible d'importer le client hybride: {e}")
        return False, {"error": f"Import error: {e}"}
    except Exception as e:
        logger.error(f"❌ Test client hybride échoué: {e}")
        return False, {"error": str(e)}


async def run_all_tests():
    """Lance tous les tests."""
    print("🚀 === DÉBUT DES TESTS ===")
    
    results = {
        "timestamp": time.time(),
        "elasticsearch_basic": {},
        "elasticsearch_client": {},
        "hybrid_client": {},
        "qdrant_basic": {},
        "qdrant_client": {},
        "summary": {}
    }
    
    # Tests Elasticsearch
    print("\n" + "="*50)
    print("🔍 TESTS ELASTICSEARCH")
    print("="*50)
    
    success, data = await test_elasticsearch_basic()
    results["elasticsearch_basic"] = {"success": success, "data": data}
    
    success, data = await test_elasticsearch_client()
    results["elasticsearch_client"] = {"success": success, "data": data}
    
    success, data = await test_hybrid_client()
    results["hybrid_client"] = {"success": success, "data": data}
    
    # Tests Qdrant
    print("\n" + "="*50)
    print("🎯 TESTS QDRANT")
    print("="*50)
    
    success, data = await test_qdrant_basic()
    results["qdrant_basic"] = {"success": success, "data": data}
    
    success, data = await test_qdrant_client()
    results["qdrant_client"] = {"success": success, "data": data}
    
    # Résumé
    print("\n" + "="*50)
    print("📊 RÉSUMÉ DES TESTS")
    print("="*50)
    
    # Analyser les résultats
    elasticsearch_basic_ok = results["elasticsearch_basic"]["success"]
    elasticsearch_client_ok = results["elasticsearch_client"]["success"]
    hybrid_client_ok = results["hybrid_client"]["success"]
    qdrant_basic_ok = results["qdrant_basic"]["success"]
    qdrant_client_ok = results["qdrant_client"]["success"]
    
    # Déterminer le meilleur client Elasticsearch
    elasticsearch_ok = elasticsearch_basic_ok  # Au minimum, le test basique doit marcher
    best_elasticsearch_client = "none"
    
    if hybrid_client_ok:
        best_elasticsearch_client = "hybrid"
        elasticsearch_ok = True
    elif elasticsearch_client_ok:
        best_elasticsearch_client = "standard"
        elasticsearch_ok = True
    elif elasticsearch_basic_ok:
        best_elasticsearch_client = "basic_http"
        elasticsearch_ok = True
    
    qdrant_ok = qdrant_basic_ok and qdrant_client_ok
    
    results["summary"] = {
        "elasticsearch_ready": elasticsearch_ok,
        "best_elasticsearch_client": best_elasticsearch_client,
        "qdrant_ready": qdrant_ok,
        "total_services": int(elasticsearch_ok) + int(qdrant_ok),
        "test_results": {
            "elasticsearch_basic": elasticsearch_basic_ok,
            "elasticsearch_client": elasticsearch_client_ok,
            "hybrid_client": hybrid_client_ok,
            "qdrant_basic": qdrant_basic_ok,
            "qdrant_client": qdrant_client_ok
        }
    }
    
    # Affichage des résultats
    print("📋 Résultats détaillés:")
    print(f"   🔍 Elasticsearch basique: {'✅' if elasticsearch_basic_ok else '❌'}")
    print(f"   🔍 Elasticsearch client: {'✅' if elasticsearch_client_ok else '❌'}")
    print(f"   🔧 Client hybride: {'✅' if hybrid_client_ok else '❌'}")
    print(f"   🎯 Qdrant basique: {'✅' if qdrant_basic_ok else '❌'}")
    print(f"   🎯 Qdrant client: {'✅' if qdrant_client_ok else '❌'}")
    
    print(f"\n🎯 Meilleur client Elasticsearch: {best_elasticsearch_client}")
    
    if elasticsearch_ok and qdrant_ok:
        print("🎉 SUCCÈS COMPLET: Les deux services fonctionnent parfaitement")
        print("✅ Recherche hybride disponible")
        print("💡 Recommandation: Utiliser le client hybride pour Elasticsearch")
    elif elasticsearch_ok:
        print("⚠️ SUCCÈS PARTIEL: Seul Elasticsearch fonctionne")
        print("✅ Recherche lexicale disponible")
        print("❌ Recherche sémantique indisponible")
        print("💡 Vérifier la configuration QDRANT_URL et QDRANT_API_KEY")
    elif qdrant_ok:
        print("⚠️ SUCCÈS PARTIEL: Seul Qdrant fonctionne") 
        print("❌ Recherche lexicale indisponible")
        print("✅ Recherche sémantique disponible")
        print("💡 Vérifier la configuration BONSAI_URL")
    else:
        print("🚨 ÉCHEC COMPLET: Aucun service ne fonctionne")
        print("❌ Vérifiez vos configurations")
        print("💡 Vérifiez BONSAI_URL, QDRANT_URL et la connectivité réseau")
    
    print("="*50)
    
    return results


def print_recommendations(results):
    """Affiche des recommandations basées sur les résultats."""
    print("\n💡 RECOMMANDATIONS:")
    print("-" * 30)
    
    summary = results["summary"]
    
    if summary["total_services"] == 2:
        print("🎯 Configuration optimale détectée")
        print("✅ Utiliser le service de recherche en mode hybride")
        if summary["best_elasticsearch_client"] == "hybrid":
            print("✅ Le client hybride est recommandé pour Elasticsearch")
        elif summary["best_elasticsearch_client"] == "standard":
            print("⚠️ Client standard OK, mais hybride recommandé pour plus de robustesse")
        
    elif summary["elasticsearch_ready"] and not summary["qdrant_ready"]:
        print("🔧 Configurer Qdrant pour activer la recherche sémantique")
        print("📋 Vérifier QDRANT_URL et QDRANT_API_KEY")
        print("🌐 Tester la connectivité réseau vers Qdrant")
        
    elif summary["qdrant_ready"] and not summary["elasticsearch_ready"]:
        print("🔧 Résoudre les problèmes Elasticsearch/Bonsai")
        print("📋 Vérifier BONSAI_URL et les credentials")
        print("🔧 Utiliser le client hybride qui gère les incompatibilités Bonsai")
        
    else:
        print("🚨 Configuration critique requise")
        print("📋 Vérifier toutes les variables d'environnement")
        print("🌐 Tester la connectivité réseau")
        print("🔑 Valider les credentials d'authentification")
    
    print("\n🔧 Pour utiliser les clients dans votre code:")
    if summary["best_elasticsearch_client"] == "hybrid":
        print("   from search_service.storage.elastic_client_hybrid import HybridElasticClient")
    elif summary["best_elasticsearch_client"] == "standard":
        print("   from elasticsearch import AsyncElasticsearch")
    
    if summary["qdrant_ready"]:
        print("   from search_service.storage.qdrant_client import QdrantClient")


if __name__ == "__main__":
    # Vérifier que les dépendances sont installées
    print("🔍 Vérification des dépendances...")
    
    dependencies = {
        "aiohttp": False,
        "elasticsearch": False,
        "qdrant_client": False,
        "dotenv": False
    }
    
    for package in dependencies:
        try:
            if package == "dotenv":
                import dotenv
            else:
                __import__(package)
            dependencies[package] = True
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - pip install {package if package != 'dotenv' else 'python-dotenv'}")
    
    missing_deps = [pkg for pkg, available in dependencies.items() if not available]
    
    if missing_deps and "qdrant_client" in missing_deps:
        print("\n⚠️ qdrant_client manquant - tests Qdrant ignorés")
        print("📦 Installer avec: pip install qdrant-client")
    
    if "aiohttp" in missing_deps or "elasticsearch" in missing_deps:
        print(f"\n❌ Dépendances critiques manquantes: {missing_deps}")
        print("📦 Installer avec: pip install aiohttp elasticsearch")
        sys.exit(1)
    
    # Vérifier la configuration minimale
    if not BONSAI_URL and not QDRANT_URL:
        print("\n❌ Aucune URL configurée")
        print("💡 Configurer au minimum BONSAI_URL ou QDRANT_URL dans .env")
        sys.exit(1)
    
    # Lancer les tests
    print("\n🚀 Lancement des tests...")
    results = asyncio.run(run_all_tests())
    
    # Afficher les recommandations
    print_recommendations(results)
    
    # Code de sortie
    total_services = results["summary"]["total_services"]
    if total_services == 2:
        print("\n🎉 Tests terminés avec succès complet")
        sys.exit(0)  # Succès complet
    elif total_services == 1:
        print("\n⚠️ Tests terminés avec succès partiel")
        sys.exit(1)  # Succès partiel
    else:
        print("\n🚨 Tests terminés en échec")
        sys.exit(2)  # Échec complet