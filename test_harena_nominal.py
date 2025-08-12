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
    
    def _make_request(self, method: str, endpoint: str, use_auth: bool = True, **kwargs) -> requests.Response:
        """Fait une requête HTTP avec gestion d'erreurs."""
        url = f"{self.base_url}{endpoint}"

        # Ajouter l'Authorization header si token disponible et requis
        if self.token and use_auth:
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
            if 'permissions' in json_data:
                print(f"✅ Permissions: {json_data['permissions']}")
                return True
            else:
                print("❌ Permissions manquantes dans la réponse")
                return False
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
            metadata = json_data.get("response_metadata", {})
            total_results = metadata.get("total_results", 0)
            returned_results = metadata.get("returned_results", 0)
            processing_time = metadata.get("processing_time_ms", 0)
            es_took = metadata.get("elasticsearch_took", 0)

            print(f"✅ Résultats trouvés: {total_results}")
            print(f"✅ Résultats retournés: {returned_results}")
            print(f"✅ Temps de traitement: {processing_time}ms")
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

        response = self._make_request("GET", "/conversation/health", use_auth=False)
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

        response = self._make_request("GET", "/conversation/status", use_auth=False)
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

        headers = {"Content-Type": "application/json"}
        conversation_id = "test-conversation"

        # 1) Appel sans jeton - doit retourner 401
        payload = {"conversation_id": conversation_id, "message": "Test sans token"}
        response = self._make_request(
            "POST", "/conversation/chat", headers=headers, data=json.dumps(payload), use_auth=False
        )
        success, _ = self._print_response(response, expected_status=401)
        if not success:
            print("❌ L'appel sans jeton n'a pas retourné 401")
            return False

        # 2) Appel authentifié avec message de recherche Netflix
        payload = {
            "conversation_id": conversation_id,
            "message": "Recherche mes dépenses Netflix",
        }
        response = self._make_request(
            "POST", "/conversation/chat", headers=headers, data=json.dumps(payload)
        )
        success, json_data = self._print_response(response)

        if not (success and json_data and json_data.get("success") is True):
            print("❌ Échec chat conversation authentifié")
            return False

        returned_conv_id = json_data.get("conversation_id")
        if returned_conv_id != conversation_id:
            print("❌ conversation_id incohérent")
            return False

        # 3) Vérification cohérence user_id via métriques
        if not self.user_id:
            print("❌ user_id non disponible")
            return False

        metrics_resp = self._make_request("GET", "/conversation/metrics", use_auth=False)
        metrics_ok, metrics_json = self._print_response(metrics_resp)
        if metrics_ok and metrics_json:
            counters = metrics_json.get("service_metrics", {}).get("counters", {})
            key = f"requests_total{{endpoint=chat,user_id={self.user_id}}}"
            if counters.get(key, 0) < 1:
                print("❌ Métriques user_id incohérentes")
                return False
        else:
            print("❌ Impossible de vérifier les métriques de conversation")
            return False

        print("✅ Chat conversation authentifiée et cohérente")
        return True

    def test_conversation_intents(self) -> bool:
        """Test 9: Conversation avec détection d'intentions."""
        self._print_step(9, "CONVERSATION INTENTS")

        headers = {"Content-Type": "application/json"}
        conversation_id = "test-conversation-intents"

        intents = [
            ("recherche pizza", "SEARCH_BY_TEXT"),
            ("combien d'opérations ce mois", "COUNT_TRANSACTIONS"),
            ("tendance budget 2025", "ANALYZE_TRENDS"),
            ("bonjour", "SMALL_TALK"),
        ]

        records = []
        for message, expected_intent in intents:
            payload = {"conversation_id": conversation_id, "message": message}
            response = self._make_request(
                "POST", "/conversation/chat", headers=headers, data=json.dumps(payload)
            )
            success, json_data = self._print_response(response)
            if not (success and json_data and json_data.get("success") is True):
                print("❌ Échec conversation pour le message envoyé")
                return False

            metadata = json_data.get("metadata", {})
            detected_intent = metadata.get("intent_detected")
            x_process = response.headers.get("X-Process-Time")
            proc_json = json_data.get("processing_time_ms")
            records.append((expected_intent, detected_intent, x_process, proc_json))

        # Affichage tableau récapitulatif
        print("\nIntent détecté vs temps de traitement")
        header = f"{'Attendu':25} | {'Détecté':25} | {'Header(ms)':12} | {'JSON(ms)':10}"
        print(header)
        print("-" * len(header))
        for exp_intent, det_intent, x_proc, proc_json in records:
            print(
                f"{exp_intent:25} | {str(det_intent):25} | {str(x_proc):12} | {str(proc_json):10}"
            )

        # Vérification cloisonnement
        turns_endpoint = f"/conversation/conversations/{conversation_id}/turns"
        resp = self._make_request("GET", turns_endpoint, use_auth=False)
        success, _ = self._print_response(resp, expected_status=401)
        if not success:
            print("❌ L'historique sans token n'a pas retourné 401")
            return False

        # Contrôle persistance
        resp = self._make_request("GET", turns_endpoint)
        success, json_data = self._print_response(resp)
        if not (
            success
            and json_data
            and json_data.get("conversation_id") == conversation_id
            and isinstance(json_data.get("turns"), list)
        ):
            print("❌ Échec récupération des turns")
            return False

        turns = json_data.get("turns", [])
        if len(turns) != len(intents):
            print("❌ Nombre de turns incohérent")
            return False
        turn_numbers = {t.get("turn_number") for t in turns}
        if turn_numbers != set(range(1, len(intents) + 1)):
            print("❌ turn_number manquant dans les turns")
            return False

        print("✅ Intents détectés et persistance vérifiée")
        return True
    
    def run_full_test(self, username: str, password: str) -> bool:
        """Lance le test complet."""
        print("🚀 DÉBUT DU TEST AUTOMATIQUE HARENA FINANCE PLATFORM")
        print(f"Base URL: {self.base_url}")
        print(f"Username: {username}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        # Compteur de succès
        tests_passed = 0
        total_tests = 9
        
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

        # Test 9: Conversation intents
        if self.test_conversation_intents():
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
