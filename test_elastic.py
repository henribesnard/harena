import requests
import json

# Test direct sur l'endpoint Elasticsearch (si accessible)
import requests
import json
import aiohttp
import asyncio
import ssl

# Configuration Bonsai
bonsai_url = "https://37r8v9zfzn:4o7ydjkcc8@fir-178893546.eu-west-1.bonsaisearch.net:443"
index_name = "harena_transactions"
test_user_id = 34

print("=== TEST ELASTICSEARCH DIRECT (SYNC) ===")

# 1. TEST AVEC REQUESTS (Plus simple)
def test_elasticsearch_sync():
    """Test synchrone avec requests"""
    
    # Test 1: Vérifier que l'index existe
    print("\n1. Vérification de l'existence de l'index...")
    try:
        response = requests.head(f"{bonsai_url}/{index_name}")
        if response.status_code == 200:
            print(f"✅ Index '{index_name}' existe")
        else:
            print(f"❌ Index '{index_name}' introuvable (status: {response.status_code})")
            return
    except Exception as e:
        print(f"❌ Erreur de connexion: {e}")
        return
    
    # Test 2: Compter les transactions de l'utilisateur
    print(f"\n2. Comptage des transactions pour user_id={test_user_id}...")
    count_query = {
        "query": {
            "term": {
                "user_id": test_user_id
            }
        }
    }
    
    try:
        response = requests.post(
            f"{bonsai_url}/{index_name}/_count", 
            json=count_query,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            count = result.get("count", 0)
            print(f"✅ {count} transactions trouvées pour user_id={test_user_id}")
        else:
            print(f"❌ Erreur comptage: {response.status_code} - {response.text}")
            return
    except Exception as e:
        print(f"❌ Erreur comptage: {e}")
        return
    
    # Test 3: Récupérer quelques transactions brutes
    print(f"\n3. Récupération de 3 transactions brutes...")
    search_query = {
        "query": {
            "term": {
                "user_id": test_user_id
            }
        },
        "size": 3
    }
    
    try:
        response = requests.post(
            f"{bonsai_url}/{index_name}/_search", 
            json=search_query,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            hits = result.get("hits", {}).get("hits", [])
            total = result.get("hits", {}).get("total", {})
            
            print(f"✅ Requête réussie:")
            print(f"   - Total: {total}")
            print(f"   - Hits retournés: {len(hits)}")
            
            # Afficher les premières transactions
            for i, hit in enumerate(hits):
                source = hit.get("_source", {})
                print(f"\n📄 Transaction {i+1}:")
                print(f"   - ID: {source.get('transaction_id', 'N/A')}")
                print(f"   - Description: {source.get('primary_description', 'N/A')}")
                print(f"   - Montant: {source.get('amount', 'N/A')}")
                print(f"   - Date: {source.get('date', 'N/A')}")
                print(f"   - User ID: {source.get('user_id', 'N/A')}")
        else:
            print(f"❌ Erreur recherche: {response.status_code} - {response.text}")
            return
    except Exception as e:
        print(f"❌ Erreur recherche: {e}")
        return
    
    # Test 4: Recherche "netflix" spécifique
    print(f"\n4. Recherche spécifique 'netflix'...")
    netflix_query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"user_id": test_user_id}},
                    {"multi_match": {
                        "query": "netflix",
                        "fields": ["searchable_text^2.0", "primary_description^1.5", "merchant_name^1.8", "category_name^1.0"],
                        "type": "best_fields",
                        "fuzziness": "AUTO",
                        "minimum_should_match": "75%"
                    }}
                ]
            }
        },
        "size": 5
    }
    
    try:
        response = requests.post(
            f"{bonsai_url}/{index_name}/_search", 
            json=netflix_query,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            hits = result.get("hits", {}).get("hits", [])
            total = result.get("hits", {}).get("total", {})
            
            print(f"✅ Recherche Netflix:")
            print(f"   - Total trouvé: {total}")
            print(f"   - Hits retournés: {len(hits)}")
            
            # Afficher les résultats Netflix
            for i, hit in enumerate(hits):
                source = hit.get("_source", {})
                score = hit.get("_score", 0)
                print(f"\n🎬 Netflix {i+1} (score: {score}):")
                print(f"   - ID: {source.get('transaction_id', 'N/A')}")
                print(f"   - Description: {source.get('primary_description', 'N/A')}")
                print(f"   - Merchant: {source.get('merchant_name', 'N/A')}")
                print(f"   - Searchable: {source.get('searchable_text', 'N/A')[:100]}...")
                print(f"   - Montant: {source.get('amount', 'N/A')}")
                print(f"   - User ID: {source.get('user_id', 'N/A')}")
                
        else:
            print(f"❌ Erreur recherche Netflix: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Erreur recherche Netflix: {e}")

# 2. TEST AVEC AIOHTTP (Async comme dans votre code)
async def test_elasticsearch_async():
    """Test asynchrone avec aiohttp comme dans votre architecture"""
    
    print("\n\n=== TEST ELASTICSEARCH DIRECT (ASYNC) ===")
    
    ssl_context = ssl.create_default_context()
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    timeout = aiohttp.ClientTimeout(total=30)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        
        # Test connexion cluster
        print("\n1. Test connexion cluster...")
        try:
            async with session.get(bonsai_url) as response:
                if response.status == 200:
                    cluster_info = await response.json()
                    print(f"✅ Cluster connecté: {cluster_info.get('cluster_name', 'unknown')}")
                else:
                    print(f"❌ Cluster: {response.status}")
        except Exception as e:
            print(f"❌ Erreur cluster: {e}")
        
        # Test comptage async
        print(f"\n2. Comptage async user_id={test_user_id}...")
        count_query = {"query": {"term": {"user_id": test_user_id}}}
        
        try:
            async with session.post(
                f"{bonsai_url}/{index_name}/_count",
                json=count_query
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ {result.get('count', 0)} transactions trouvées")
                else:
                    text = await response.text()
                    print(f"❌ Erreur: {response.status} - {text}")
        except Exception as e:
            print(f"❌ Erreur: {e}")
        
        # Test recherche Netflix async
        print(f"\n3. Recherche Netflix async...")
        netflix_query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"user_id": test_user_id}},
                        {"multi_match": {
                            "query": "netflix",
                            "fields": ["searchable_text^2.0", "primary_description^1.5", "merchant_name^1.8"]
                        }}
                    ]
                }
            },
            "size": 3
        }
        
        try:
            async with session.post(
                f"{bonsai_url}/{index_name}/_search",
                json=netflix_query
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    hits = result.get("hits", {}).get("hits", [])
                    total = result.get("hits", {}).get("total", {})
                    print(f"✅ Netflix async: {total} total, {len(hits)} retournés")
                    
                    for hit in hits:
                        source = hit.get("_source", {})
                        print(f"   - {source.get('primary_description', 'N/A')}")
                else:
                    text = await response.text()
                    print(f"❌ Erreur Netflix: {response.status} - {text}")
        except Exception as e:
            print(f"❌ Erreur Netflix: {e}")

if __name__ == "__main__":
    print("🔍 Test de diagnostic Elasticsearch Harena")
    print(f"🔗 URL: {bonsai_url}")
    print(f"📋 Index: {index_name}")
    print(f"👤 User ID: {test_user_id}")
    
    # Test synchrone
    test_elasticsearch_sync()
    
    # Test asynchrone
    asyncio.run(test_elasticsearch_async())
    
    print("\n=== RÉSUMÉ DES TESTS ===")
    print("Si les tests directs trouvent des données mais votre API retourne 0 results:")
    print("➡️  Le problème est dans search_service/core/search_engine.py _process_results()")
    print("➡️  Elasticsearch fonctionne, mais la conversion Python échoue")
    print("➡️  Vérifiez les logs d'erreur dans _process_results()")