import requests
import json

# Configuration
BASE_URL = "https://harenabackend-ab1b255e55c6.herokuapp.com/api/v1"
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NTMxMTIxMzYsInN1YiI6IjM0In0.QSD_UKfGHQpG-47KzxnwnTuJ3pSteEZUQ-WUg7OtWoE"

def test_health_endpoint():
    """Test 1: Vérifier l'état de santé du service"""
    print("🩺 Test 1: Health check du service")
    
    url = f"{BASE_URL}/search/health"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {TOKEN}'
    }
    
    try:
        response = requests.get(url, headers=headers)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'healthy':
                print("✅ Service is healthy")
                return True
            else:
                print(f"⚠️ Service status: {data.get('status')}")
                print(f"Details: {data.get('details', {})}")
                return False
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Exception during health check: {e}")
        return False

def test_root_endpoint():
    """Test 2: Vérifier l'endpoint racine"""
    print("\n🏠 Test 2: Root endpoint")
    
    url = f"{BASE_URL}/"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {TOKEN}'
    }
    
    try:
        response = requests.get(url, headers=headers)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
    except Exception as e:
        print(f"❌ Exception: {e}")

def test_config_endpoint():
    """Test 3: Vérifier la configuration"""
    print("\n⚙️ Test 3: Configuration endpoint")
    
    url = f"{BASE_URL}/search/config"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {TOKEN}'
    }
    
    try:
        response = requests.get(url, headers=headers)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
    except Exception as e:
        print(f"❌ Exception: {e}")

def test_minimal_search():
    """Test 4: Recherche minimale pour diagnostiquer"""
    print("\n🔍 Test 4: Recherche minimale")
    
    url = f"{BASE_URL}/search/search"
    payload = {
        "user_id": 34
    }
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {TOKEN}'
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 503:
            print("❌ Service unavailable - problème d'infrastructure")
            print("💡 Suggestions:")
            print("   - Vérifier les variables d'environnement (BONSAI_URL)")
            print("   - Vérifier la connexion à Elasticsearch")
            print("   - Redémarrer le service sur Heroku")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

def test_elasticsearch_endpoints():
    """Test 5: Tester d'autres endpoints de diagnostic"""
    print("\n📊 Test 5: Endpoints de diagnostic")
    
    # Essayer différents endpoints qui pourraient exister
    endpoints_to_test = [
        "/health",
        "/search/health", 
        "/api/health",
        "/status",
        "/metrics",
        "/search/status"
    ]
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {TOKEN}'
    }
    
    for endpoint in endpoints_to_test:
        url = f"{BASE_URL}{endpoint}"
        try:
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code < 500:
                print(f"✅ {endpoint}: {response.status_code}")
                if response.status_code == 200:
                    try:
                        data = response.json()
                        print(f"   Data: {json.dumps(data, indent=4)}")
                    except:
                        print(f"   Text: {response.text[:200]}")
            else:
                print(f"❌ {endpoint}: {response.status_code}")
                
        except requests.exceptions.Timeout:
            print(f"⏰ {endpoint}: Timeout")
        except Exception as e:
            print(f"❌ {endpoint}: {e}")

def debug_initialization_status():
    """Test 6: Analyser les détails d'initialisation"""
    print("\n🔧 Test 6: Analyse de l'initialisation")
    
    # Recherche avec plus de détails pour comprendre l'état
    url = f"{BASE_URL}/search/search"
    payload = {
        "user_id": 34,
        "query": "test"
    }
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {TOKEN}'
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
        # Analyser la réponse d'erreur
        if response.status_code == 503:
            try:
                error_data = response.json()
                detail = error_data.get('detail', '')
                
                if 'Service not initialized' in detail:
                    print("🔍 Diagnostic: Service d'initialisation échoué")
                    print("💡 Solutions possibles:")
                    print("   1. Variables d'environnement manquantes")
                    print("   2. Problème de connexion Elasticsearch/Bonsai")
                    print("   3. Timeout d'initialisation")
                    
                elif 'Search engine not available' in detail:
                    print("🔍 Diagnostic: Core manager non initialisé")
                    print("💡 Solutions possibles:")
                    print("   1. Redémarrer l'application Heroku")
                    print("   2. Vérifier les logs de démarrage")
                    print("   3. Vérifier la configuration Elasticsearch")
                    
            except:
                pass
                
    except Exception as e:
        print(f"❌ Exception: {e}")

def run_full_diagnosis():
    """Exécute tous les tests de diagnostic"""
    print("="*60)
    print("🔬 DIAGNOSTIC COMPLET DU SEARCH SERVICE")
    print("="*60)
    
    # Exécuter tous les tests
    health_ok = test_health_endpoint()
    test_root_endpoint()
    test_config_endpoint()
    test_minimal_search()
    test_elasticsearch_endpoints()
    debug_initialization_status()
    
    print("\n" + "="*60)
    print("📋 RÉSUMÉ DU DIAGNOSTIC")
    print("="*60)
    
    if health_ok:
        print("✅ Service en bonne santé - problème probablement temporaire")
    else:
        print("❌ Service non initialisé correctement")
        print("\n🛠️ ACTIONS RECOMMANDÉES:")
        print("1. Vérifier les variables d'environnement sur Heroku:")
        print("   heroku config:show --app harenabackend")
        print("2. Vérifier les logs de démarrage:")
        print("   heroku logs --tail --app harenabackend")
        print("3. Redémarrer le service:")
        print("   heroku restart --app harenabackend")
        print("4. Vérifier la connectivité Elasticsearch/Bonsai")

if __name__ == "__main__":
    run_full_diagnosis()