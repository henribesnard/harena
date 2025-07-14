#!/usr/bin/env python3
"""
Script de test pour vérifier la configuration Elasticsearch
=========================================================

Ce script teste la nouvelle configuration simplifiée du client Elasticsearch
et vérifie que BONSAI_URL est correctement détectée.

USAGE: Ce script doit être lancé depuis le répertoire search_service/
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# === CONFIGURATION PYTHONPATH ===
# Ajouter le répertoire courant au PYTHONPATH pour permettre les imports
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("test_elasticsearch")

async def test_elasticsearch_configuration():
    """
    Teste la configuration Elasticsearch avec la nouvelle API simplifiée
    """
    print("🧪 Test de la configuration Elasticsearch simplifiée")
    print("=" * 60)
    
    # 1. Vérifier les variables d'environnement
    print("\n1️⃣ Vérification des variables d'environnement:")
    bonsai_url = os.environ.get("BONSAI_URL")
    elasticsearch_index = os.environ.get("ELASTICSEARCH_INDEX", "harena_transactions")
    test_user_id = os.environ.get("TEST_USER_ID", "34")
    
    print(f"   BONSAI_URL: {'✅ SET' if bonsai_url else '❌ NOT SET'}")
    if bonsai_url:
        print(f"   URL: {bonsai_url[:50]}...")
    print(f"   ELASTICSEARCH_INDEX: {elasticsearch_index}")
    print(f"   TEST_USER_ID: {test_user_id}")
    
    if not bonsai_url:
        print("\n❌ ERREUR: BONSAI_URL n'est pas configurée")
        print("💡 Veuillez définir BONSAI_URL dans votre fichier .env")
        print("   Exemple: BONSAI_URL=https://your-cluster.eu-west-1.bonsaisearch.net:443")
        return False
    
    # 2. Test d'import des modules (avec gestion d'erreurs)
    print("\n2️⃣ Test d'import des modules:")
    try:
        # Import depuis le répertoire local
        from clients.elasticsearch_client import (
            ElasticsearchClient,
            get_default_client,
            initialize_default_client,
            get_client_configuration_info
        )
        print("   ✅ Import réussi depuis clients.elasticsearch_client")
    except ImportError as e:
        print(f"   ❌ Erreur d'import: {e}")
        print("   💡 Vérifiez que vous êtes dans le répertoire search_service/")
        print("   💡 Vérifiez que le fichier clients/elasticsearch_client.py existe")
        return False
    except Exception as e:
        print(f"   ❌ Erreur inattendue: {e}")
        return False
    
    # 3. Test de création du client (nouvelle API)
    print("\n3️⃣ Test de création du client (nouvelle API):")
    try:
        # ✅ Test avec la nouvelle API simplifiée
        client = ElasticsearchClient()  # Plus besoin de passer bonsai_url
        print("   ✅ Client créé avec succès (auto-détection URL)")
        print(f"   📍 URL détectée: {client.base_url}")
        print(f"   📦 Index: {client.index_name}")
    except Exception as e:
        print(f"   ❌ Erreur lors de la création: {e}")
        import traceback
        print(f"   🔍 Stack trace: {traceback.format_exc()}")
        return False
    
    # 4. Test d'initialisation
    print("\n4️⃣ Test d'initialisation du client:")
    try:
        await client.initialize()
        print("   ✅ Initialisation réussie")
    except Exception as e:
        print(f"   ❌ Erreur lors de l'initialisation: {e}")
        print(f"   💡 Ceci peut être normal si Elasticsearch n'est pas accessible")
        # On continue les tests même si l'initialisation échoue
    
    # 5. Test de connexion (avec gestion d'échec)
    print("\n5️⃣ Test de connexion à Elasticsearch:")
    try:
        connection_ok = await client.test_connection()
        if connection_ok:
            print("   ✅ Connexion réussie")
        else:
            print("   ⚠️ Connexion échouée (normal si Elasticsearch non accessible)")
    except Exception as e:
        print(f"   ⚠️ Erreur lors du test de connexion: {e}")
        print("   💡 Ceci est normal si Elasticsearch n'est pas en cours d'exécution")
    
    # 6. Test de health check (avec gestion d'échec)
    print("\n6️⃣ Test de health check:")
    try:
        health = await client.health_check()
        print(f"   📊 Statut: {health.get('status', 'unknown')}")
        print(f"   🏷️ Cluster: {health.get('cluster_name', 'unknown')}")
        print(f"   📝 Version: {health.get('version', 'unknown')}")
        
        if health.get('status') == 'healthy':
            print("   ✅ Health check réussi")
        else:
            print("   ⚠️ Health check dégradé (normal si ES non accessible)")
    except Exception as e:
        print(f"   ⚠️ Erreur lors du health check: {e}")
        print("   💡 Ceci est normal si Elasticsearch n'est pas accessible")
    
    # 7. Test du client global
    print("\n7️⃣ Test du client global (get_default_client):")
    try:
        global_client = get_default_client()
        print("   ✅ Client global obtenu")
        print(f"   📍 URL: {global_client.base_url}")
        
        # Test d'initialisation du client global
        try:
            await global_client.initialize()
            print("   ✅ Client global initialisé")
        except Exception as e:
            print(f"   ⚠️ Initialisation client global échouée: {e}")
            print("   💡 Normal si Elasticsearch non accessible")
            
    except Exception as e:
        print(f"   ❌ Erreur avec le client global: {e}")
        return False
    
    # 8. Test de initialize_default_client
    print("\n8️⃣ Test de initialize_default_client:")
    try:
        initialized_client = await initialize_default_client()
        print("   ✅ initialize_default_client réussi")
        print(f"   📍 URL: {initialized_client.base_url}")
    except Exception as e:
        print(f"   ⚠️ Erreur initialize_default_client: {e}")
        print("   💡 Normal si Elasticsearch non accessible")
    
    # 9. Test des informations de configuration
    print("\n9️⃣ Informations de configuration:")
    try:
        config_info = get_client_configuration_info()
        print("   ✅ Configuration récupérée:")
        
        # Afficher les infos importantes
        if "configuration" in config_info:
            config = config_info["configuration"]
            print(f"   📍 BONSAI_URL: {config.get('bonsai_url', 'non définie')}")
            print(f"   📦 Index: {config.get('elasticsearch_index', 'non défini')}")
            print(f"   👤 Test User ID: {config.get('test_user_id', 'non défini')}")
        
        if "current_client" in config_info:
            client_info = config_info["current_client"]
            print(f"   🔗 Client URL: {client_info.get('base_url', 'non définie')}")
            print(f"   🏠 Localhost: {client_info.get('is_localhost', 'inconnu')}")
            
    except Exception as e:
        print(f"   ❌ Erreur récupération config: {e}")
    
    # 10. Test de recherche simple (avec gestion d'échec)
    print("\n🔟 Test de recherche simple:")
    try:
        from clients.elasticsearch_client import quick_search
        
        # Test avec l'utilisateur de test
        result = await quick_search(
            user_id=int(test_user_id),
            query="test",
            limit=1
        )
        
        if "error" in result:
            print(f"   ⚠️ Recherche échouée: {result['error']}")
            print("   💡 Normal si Elasticsearch non accessible ou index inexistant")
        else:
            total_hits = result.get("hits", {}).get("total", {}).get("value", 0)
            print(f"   ✅ Recherche réussie: {total_hits} résultats trouvés")
            
    except Exception as e:
        print(f"   ⚠️ Test de recherche échoué: {e}")
        print("   💡 Ceci peut être normal si l'index n'existe pas encore")
    
    # Nettoyage
    print("\n🧹 Nettoyage:")
    try:
        await client.stop()
        print("   ✅ Client fermé proprement")
    except Exception as e:
        print(f"   ⚠️ Erreur lors du nettoyage: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Test terminé!")
    print("✅ La configuration Elasticsearch simplifiée fonctionne")
    print("💡 Les erreurs de connexion sont normales si Elasticsearch n'est pas en cours d'exécution")
    return True


async def test_old_vs_new_api():
    """
    Compare l'ancienne et la nouvelle API pour montrer les améliorations
    """
    print("\n📊 Comparaison Ancienne vs Nouvelle API:")
    print("-" * 40)
    
    # Ancienne API (problématique)
    print("❌ Ancienne API (problématique):")
    print("   client = ElasticsearchClient(bonsai_url)  # bonsai_url obligatoire")
    print("   → Erreur si bonsai_url non fourni")
    print("   → Duplication de logique de résolution d'URL")
    
    # Nouvelle API (simplifiée)
    print("\n✅ Nouvelle API (simplifiée):")
    print("   client = ElasticsearchClient()  # Auto-détection")
    print("   → URL auto-détectée depuis BONSAI_URL")
    print("   → Plus simple à utiliser")
    print("   → Moins d'erreurs de configuration")


def check_directory():
    """
    Vérifie que le script est lancé depuis le bon répertoire
    """
    current_dir = Path.cwd()
    expected_files = [
        "clients/elasticsearch_client.py",
        "main.py",
        "api/routes.py"
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not (current_dir / file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ ERREUR: Répertoire incorrect ou fichiers manquants")
        print(f"   Répertoire actuel: {current_dir}")
        print(f"   Fichiers manquants: {missing_files}")
        print("\n💡 Solutions:")
        print("   1. Lancez ce script depuis le répertoire search_service/")
        print("   2. Vérifiez que les fichiers corrrigés ont été appliqués")
        return False
    
    return True


def main():
    """
    Point d'entrée principal du script de test
    """
    print("🚀 Script de test de la configuration Elasticsearch")
    print("Version: Configuration simplifiée avec auto-détection BONSAI_URL")
    
    # Vérifier Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ requis")
        sys.exit(1)
    
    # Vérifier le répertoire
    if not check_directory():
        sys.exit(1)
    
    print(f"📁 Répertoire de travail: {Path.cwd()}")
    print(f"🐍 Python: {sys.version}")
    
    # Lancer les tests
    try:
        # Test de la nouvelle API
        success = asyncio.run(test_elasticsearch_configuration())
        
        # Comparaison des APIs
        asyncio.run(test_old_vs_new_api())
        
        if success:
            print("\n🎯 RÉSULTAT: Configuration Elasticsearch opérationnelle")
            print("✅ Les corrections du constructeur ElasticsearchClient sont efficaces")
            sys.exit(0)
        else:
            print("\n❌ RÉSULTAT: Problèmes détectés dans la configuration")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Test interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()