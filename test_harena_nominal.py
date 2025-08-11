"""
Script de test automatique pour Harena Finance Platform - Chemin nominal.

Ce script teste automatiquement la chaîne complète :
1. Login utilisateur
2. Récupération profil utilisateur
3. Synchronisation enrichment Elasticsearch
4. Health check enrichment service
5. Recherche de transactions
6. Health check conversation service
7. Status conversation service
8. Chat conversation

Usage:
    python test_harena_nominal.py
    python test_harena_nominal.py --username test@example.com --password mypass
    python test_harena_nominal.py --base-url https://api.harena.com/api/v1
"""

import requests
import json
import argparse
import sys
from datetime import datetime
from typing import Dict, Any, Optional

# ===== CONFIGURATION INITIALE =====
DEFAULT_BASE_URL = "http://localhost:8000/api/v1"
DEFAULT_USERNAME = "test2@example.com"
DEFAULT_PASSWORD = "password123"

# Timeout pour les requêtes
REQUEST_TIMEOUT = 30

class HarenaTestClient:
    """Client de test pour Harena Finance Platform."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.token: Optional[str] = None
        self.user_id: Optional[int] = None
        self.session = requests.Session()
        self.session.timeout = REQUEST_TIMEOUT
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Fait une requête HTTP avec gestion d'erreurs."""
        url = f"{self.base_url}{endpoint}"
        
        # Ajouter l'Authorization header si token disponible
        if self.token and 'headers' not in kwargs:
            kwargs['headers'] = {}
        if self.token:
            kwargs['headers'] = kwargs.get('headers', {})
            kwargs['headers']['Authorization'] = f'Bearer {self.token}'
        
        try:
            response = self.session.request(method, url, **kwargs)
            return response
        except requests.exceptions.RequestException as e:
            print(f"❌ Erreur requête {method} {endpoint}: {e}")
            return None
    
    def _print_step(self, step_num: int, title: str, status: str = ""):
        """Affiche une étape du test."""
        print(f"\n{'='*60}")
        print(f"ÉTAPE {step_num}: {title}")
        if status:
            print(f"Status: {status}")
        print('='*60)
    
    def _print_response(self, response: Optional[requests.Response], expected_status: int = 200):
        """Affiche les détails de la réponse.

        Retourne toujours un tuple (success, json_data)."""
        if response is None:
            print("❌ Pas de réponse reçue")
            return False, None
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == expected_status:
            print("✅ Status Code OK")
        else:
            print(f"❌ Status Code attendu: {expected_status}, reçu: {response.status_code}")
        
        try:
            json_response = response.json()
            print("Réponse JSON:")
            print(json.dumps(json_response, indent=2, ensure_ascii=False))
            return response.status_code == expected_status, json_response
        except json.JSONDecodeError:
            print("❌ Réponse non JSON:")
            print(response.text[:500])
            return False, None
    
    def test_login(self, username: str, password: str) -> bool:
        """Test 1: Login utilisateur."""
        self._print_step(1, "LOGIN UTILISATEUR")
        
        payload = f'username={username}&password={password}'
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        response = self._make_request("POST", "/users/auth/login", headers=headers, data=payload)
        success, json_data = self._print_response(response)
        
        if success and json_data and 'access_token' in json_data:
            self.token = json_data['access_token']
            print(f"✅ Token récupéré: {self.token[:50]}...")
            return True
        else:
            print("❌ Échec récupération token")
            return False
    
    def test_user_profile(self) -> bool:
        """Test 2: Récupération du profil utilisateur."""
        self._print_step(2, "PROFIL UTILISATEUR")
        
        if not self.token:
            print("❌ Pas de token disponible")
            return False
        
        response = self._make_request("GET", "/users/me")
        success, json_data = self._print_response(response)
        
        if success and json_data and 'id' in json_data:
            self.user_id = json_data['id']
            print(f"✅ User ID récupéré: {self.user_id}")
            print(f"✅ Email: {json_data.get('email', 'N/A')}")
            print(f"✅ Nom: {json_data.get('first_name', '')} {json_data.get('last_name', '')}")
            return True
        else:
            print("❌ Échec récupération profil")
            return False
    
    def test_enrichment_sync(self) -> bool:
        """Test 3: Synchronisation enrichment Elasticsearch."""
        self._print_step(3, "SYNCHRONISATION ENRICHMENT ELASTICSEARCH")
        
        if not self.user_id:
            print("❌ User ID non disponible")
            return False
        
        response = self._make_request("POST", f"/enrichment/elasticsearch/sync-user/{self.user_id}")
        success, json_data = self._print_response(response)
        
        if success and json_data:
            total_tx = json_data.get('total_transactions', 0)
            indexed = json_data.get('indexed', 0)
            errors = json_data.get('errors', 0)
            processing_time = json_data.get('processing_time', 0)
            
            print(f"✅ Transactions totales: {total_tx}")
            print(f"✅ Transactions indexées: {indexed}")
            print(f"✅ Erreurs: {errors}")
            print(f"✅ Temps de traitement: {processing_time:.3f}s")
            
            if json_data.get('status') == 'success':
                print("✅ Synchronisation réussie")
                return True
            else:
                print("❌ Synchronisation échouée")
                return False
        else:
            print("❌ Échec synchronisation")
            return False
    
    def test_enrichment_health(self) -> bool:
        """Test 4: Health check enrichment service."""
        self._print_step(4, "HEALTH CHECK ENRICHMENT SERVICE")
        
        response = self._make_request("GET", "/enrichment/elasticsearch/health")
        success, json_data = self._print_response(response)
        
        if success and json_data:
            service_status = json_data.get('status', 'unknown')
            version = json_data.get('version', 'unknown')
            elasticsearch_available = json_data.get('elasticsearch', {}).get('available', False)
            
            print(f"✅ Service: {json_data.get('service', 'unknown')}")
            print(f"✅ Version: {version}")
            print(f"✅ Status: {service_status}")
            print(f"✅ Elasticsearch disponible: {elasticsearch_available}")
            
            if elasticsearch_available:
                cluster_info = json_data.get('elasticsearch', {}).get('cluster_info', {})
                print(f"✅ Cluster: {cluster_info.get('cluster_name', 'unknown')}")
                print(f"✅ ES Version: {cluster_info.get('version', {}).get('number', 'unknown')}")
            
            capabilities = json_data.get('capabilities', {})
            print(f"✅ Capabilities: {len([k for k, v in capabilities.items() if v])}/{len(capabilities)}")
            
            if service_status == 'healthy':
                print("✅ Service en bonne santé")
                return True
            else:
                print("⚠️ Service en mode dégradé")
                return False
        else:
            print("❌ Échec health check")
            return False
    
    def test_search(self) -> bool:
        """Test 5: Recherche de transactions."""
        self._print_step(5, "RECHERCHE DE TRANSACTIONS")
        
        if not self.user_id:
            print("❌ User ID non disponible")
            return False
        
        # Requête de recherche Netflix avec filtres
        search_payload = {
            "user_id": self.user_id,
            "query": "netflix",
            "limit": 25,
            "offset": 0,
            "filters": {
                "amount": {
                    "gte": -25,
                    "lte": -10
                },
                "date": {
                    "gte": "2025-01-01",
                    "lte": "2025-12-31"
                },
                "currency_code": "EUR",
                "transaction_type": "debit",
                "operation_type": "card"
            },
            "metadata": {
                "debug": True,
                "include_highlights": True,
                "cache_enabled": True,
                "explain_score": False
            }
        }
        
        headers = {'Content-Type': 'application/json'}
        response = self._make_request("POST", "/search/search",
                                    headers=headers,
                                    data=json.dumps(search_payload))

        success, json_data = self._print_response(response)
        
        if success and json_data:
            total_hits = json_data.get('total_hits', 0)
            returned_hits = json_data.get('returned_hits', 0)
            execution_time = json_data.get('execution_time_ms', 0)
            es_took = json_data.get('elasticsearch_took', 0)
            
            print(f"✅ Résultats trouvés: {total_hits}")
            print(f"✅ Résultats retournés: {returned_hits}")
            print(f"✅ Temps d'exécution: {execution_time}ms")
            print(f"✅ Temps Elasticsearch: {es_took}ms")
            
            if total_hits > 0:
                print("✅ Recherche fonctionnelle - Résultats trouvés")
                
                # Afficher quelques détails des résultats
                results = json_data.get('results', [])
                for i, result in enumerate(results[:3]):  # Afficher max 3 résultats
                    print(f"  📄 Résultat {i+1}:")
                    print(f"     • Description: {result.get('primary_description', 'N/A')}")
                    print(f"     • Montant: {result.get('amount', 'N/A')} {result.get('currency_code', '')}")
                    print(f"     • Date: {result.get('date', 'N/A')}")
                    print(f"     • Score: {result.get('score', 'N/A')}")
                
                return True
            else:
                print("⚠️ Aucun résultat trouvé pour 'netflix'")
                return True  # Ce n'est pas forcément une erreur
        else:
            print("❌ Échec recherche")
            return False

    def test_conversation_health(self) -> bool:
        """Test 6: Health check conversation service."""
        self._print_step(6, "HEALTH CHECK CONVERSATION SERVICE")

        response = self._make_request("GET", "/conversation/health")
        success, json_data = self._print_response(response)

        if success and json_data:
            status = json_data.get('status', 'unknown')
            print(f"✅ Service: {json_data.get('service', 'unknown')}")
            print(f"✅ Status: {status}")
            if status in ("healthy", "degraded"):
                return True
            else:
                print("❌ Statut inattendu")
                return False
        else:
            print("❌ Échec health check conversation")
            return False

    def test_conversation_status(self) -> bool:
        """Test 7: Status conversation service."""
        self._print_step(7, "STATUS CONVERSATION SERVICE")

        response = self._make_request("GET", "/conversation/status")
        success, json_data = self._print_response(response)

        if success and json_data:
            service = json_data.get('service')
            status = json_data.get('status')
            version = json_data.get('version')
            print(f"✅ Service: {service}")
            print(f"✅ Status: {status}")
            print(f"✅ Version: {version}")
            if service and status and version:
                return True
        print("❌ Échec status conversation")
        return False

    def test_conversation_chat(self) -> bool:
        """Test 8: Chat conversation."""
        self._print_step(8, "CHAT CONVERSATION")

        # Liste de messages représentatifs à envoyer séquentiellement
        test_cases = [
            {"message": "Bonjour", "expected_intent": "GREETING"},
            {
                "message": "Recherche mes dépenses Netflix",
                "expected_intent": "TRANSACTION_SEARCH",
            },
            {"message": "Quel est mon solde ?", "expected_intent": "BALANCE_INQUIRY"},
            {
                "message": "Analyse mes dépenses alimentaires",
                "expected_intent": "SPENDING_ANALYSIS",
            },
        ]

        headers = {"Content-Type": "application/json"}
        conversation_id = "test-conversation"
        all_passed = True

        for idx, case in enumerate(test_cases, start=1):
            payload = {
                "conversation_id": conversation_id,
                "message": case["message"],
            }

            print(f"\n--- Chat message {idx}: {case['message']} ---")
            response = self._make_request(
                "POST",
                "/conversation/chat",
                headers=headers,
                data=json.dumps(payload),
            )

            success, json_data = self._print_response(response)

            if not (
                success
                and json_data
                and json_data.get("success") is True
                and json_data.get("message")
            ):
                print(f"❌ Échec chat pour le message {idx}")
                all_passed = False
                continue

            # Vérification facultative de l'intention si renvoyée
            expected_intent = case.get("expected_intent")
            if expected_intent:
                metadata = json_data.get("metadata", {})
                detected_intent = metadata.get("intent") or metadata.get("intent_type")
                if detected_intent:
                    if detected_intent == expected_intent:
                        print(f"✅ Intention détectée: {detected_intent}")
                    else:
                        print(
                            f"❌ Intention attendue: {expected_intent}, reçue: {detected_intent}"
                        )
                        all_passed = False
                else:
                    print("ℹ️ Intention non fournie dans la réponse")

        if all_passed:
            print("✅ Chat sequence success")
        else:
            print("❌ Chat sequence failed")
        return all_passed
    
    def run_full_test(self, username: str, password: str) -> bool:
        """Lance le test complet."""
        print("🚀 DÉBUT DU TEST AUTOMATIQUE HARENA FINANCE PLATFORM")
        print(f"Base URL: {self.base_url}")
        print(f"Username: {username}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        # Compteur de succès
        tests_passed = 0
        total_tests = 8
        
        # Test 1: Login
        if self.test_login(username, password):
            tests_passed += 1
        else:
            print("\n❌ TEST ARRÊTÉ - Impossible de se connecter")
            return False
        
        # Test 2: Profil utilisateur
        if self.test_user_profile():
            tests_passed += 1
        else:
            print("\n❌ TEST ARRÊTÉ - Impossible de récupérer le profil")
            return False
        
        # Test 3: Synchronisation enrichment
        if self.test_enrichment_sync():
            tests_passed += 1
        
        # Test 4: Health check enrichment
        if self.test_enrichment_health():
            tests_passed += 1
        
        # Test 5: Recherche
        if self.test_search():
            tests_passed += 1

        # Test 6: Conversation health
        if self.test_conversation_health():
            tests_passed += 1

        # Test 7: Conversation status
        if self.test_conversation_status():
            tests_passed += 1

        # Test 8: Conversation chat
        if self.test_conversation_chat():
            tests_passed += 1

        # Résumé final
        print(f"\n{'='*60}")
        print("📊 RÉSUMÉ DU TEST")
        print('='*60)
        print(f"Tests réussis: {tests_passed}/{total_tests}")
        print(f"Pourcentage de réussite: {(tests_passed/total_tests)*100:.1f}%")
        
        if tests_passed == total_tests:
            print("✅ TOUS LES TESTS SONT PASSÉS - PLATEFORME OPÉRATIONNELLE")
            return True
        else:
            print("❌ CERTAINS TESTS ONT ÉCHOUÉ - VÉRIFIER LA CONFIGURATION")
            return False

def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Test automatique Harena Finance Platform")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, 
                       help=f"URL de base de l'API (défaut: {DEFAULT_BASE_URL})")
    parser.add_argument("--username", default=DEFAULT_USERNAME,
                       help=f"Nom d'utilisateur (défaut: {DEFAULT_USERNAME})")
    parser.add_argument("--password", default=DEFAULT_PASSWORD,
                       help="Mot de passe (défaut: password123)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Mode verbose")
    
    args = parser.parse_args()
    
    # Créer le client de test
    client = HarenaTestClient(args.base_url)
    
    # Lancer le test complet
    try:
        success = client.run_full_test(args.username, args.password)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrompu par l'utilisateur")
        sys.exit(2)
    except Exception as e:
        print(f"\n\n❌ Erreur inattendue: {e}")
        sys.exit(3)

if __name__ == "__main__":
    main()
