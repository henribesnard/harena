"""
Script de test automatique pour Harena Finance Platform - Avec données réelles utilisateur.

Ce script teste automatiquement la chaîne complète avec des questions basées sur 
les vraies transactions de l'utilisateur 34:
1. Login utilisateur
2. Récupération profil utilisateur
3. Synchronisation enrichment Elasticsearch
4. Health check enrichment service
5. Recherche de transactions

Usage:
    python test_harena_real_data.py
    python test_harena_real_data.py --username test2@example.com --password mypass
    python test_harena_real_data.py --base-url https://api.harena.com/api/v1
"""

import requests
import json
import argparse
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

# ===== CONFIGURATION INITIALE =====
DEFAULT_BASE_URL = "http://localhost:8000/api/v1"
DEFAULT_USERNAME = "test2@example.com"
DEFAULT_PASSWORD = "password123"

# Timeout pour les requêtes
REQUEST_TIMEOUT = 30

class HarenaRealDataTestClient:
    """Client de test pour Harena Finance Platform avec données réelles."""

    def __init__(self, base_url: str, logger: Optional[logging.Logger] = None):
        self.base_url = base_url.rstrip('/')
        self.token: Optional[str] = None
        self.user_id: Optional[int] = None
        self.session = requests.Session()
        self.session.timeout = REQUEST_TIMEOUT
        self.logger = logger or logging.getLogger(__name__)
        
        # Questions basées sur le jeu de données du MockIntentAgent
        self.real_data_questions = [
            {
                "question": "Mes transactions Netflix ce mois",
                "expected_intent": "TRANSACTION_SEARCH",
                "expected_data": None,
                "should_find": True,
            },
            {
                "question": "Combien j'ai dépensé chez Carrefour ?",
                "expected_intent": "SPENDING_ANALYSIS",
                "expected_data": None,
                "should_find": True,
            },
            {
                "question": "Mes achats Amazon janvier 2025",
                "expected_intent": "TRANSACTION_SEARCH",
                "expected_data": None,
                "should_find": True,
            },
            {
                "question": "Transactions supérieures à 100 euros",
                "expected_intent": "TRANSACTION_SEARCH",
                "expected_data": None,
                "should_find": True,
            },
            {
                "question": "Mes dépenses restaurant cette semaine",
                "expected_intent": "SPENDING_ANALYSIS",
                "expected_data": None,
                "should_find": True,
            },
            {
                "question": "Analyse mes courses alimentaires",
                "expected_intent": "SPENDING_ANALYSIS",
                "expected_data": None,
                "should_find": True,
            },
            {
                "question": "Combien je dépense en transport par mois ?",
                "expected_intent": "SPENDING_ANALYSIS",
                "expected_data": None,
                "should_find": True,
            },
            {
                "question": "Évolution de mes dépenses ces 3 derniers mois",
                "expected_intent": "TREND_ANALYSIS",
                "expected_data": None,
                "should_find": True,
            },
            {
                "question": "Tendance dépenses loisirs depuis janvier",
                "expected_intent": "TREND_ANALYSIS",
                "expected_data": None,
                "should_find": True,
            },
        ]
    
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
            self.logger.error(f"❌ Erreur requête {method} {endpoint}: {e}")
            return None
    
    def _print_step(self, step_num: int, title: str, status: str = ""):
        """Affiche une étape du test."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"ÉTAPE {step_num}: {title}")
        if status:
            self.logger.info(f"Status: {status}")
        self.logger.info('='*60)
    
    def _print_response(self, response: Optional[requests.Response], expected_status: int = 200):
        """Affiche les détails de la réponse."""
        if response is None:
            self.logger.error("❌ Pas de réponse reçue")
            return False, None

        self.logger.info(f"Status Code: {response.status_code}")

        if response.status_code == expected_status:
            self.logger.info("✅ Status Code OK")
        else:
            self.logger.error(f"❌ Status Code attendu: {expected_status}, reçu: {response.status_code}")

        try:
            json_response = response.json()
            self.logger.info("Réponse JSON:")
            self.logger.info(json.dumps(json_response, indent=2, ensure_ascii=False))
            return response.status_code == expected_status, json_response
        except json.JSONDecodeError:
            self.logger.error("❌ Réponse non JSON:")
            self.logger.error(response.text[:500])
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
            self.logger.info(f"✅ Token récupéré: {self.token[:50]}...")
            return True
        else:
            self.logger.error("❌ Échec récupération token")
            return False
    
    def test_user_profile(self) -> bool:
        """Test 2: Récupération du profil utilisateur."""
        self._print_step(2, "PROFIL UTILISATEUR")
        
        if not self.token:
            self.logger.error("❌ Pas de token disponible")
            return False
        
        response = self._make_request("GET", "/users/me")
        success, json_data = self._print_response(response)
        
        if success and json_data and 'id' in json_data:
            self.user_id = json_data['id']
            self.logger.info(f"✅ User ID récupéré: {self.user_id}")
            self.logger.info(f"✅ Email: {json_data.get('email', 'N/A')}")
            self.logger.info(f"✅ Nom: {json_data.get('first_name', '')} {json_data.get('last_name', '')}")
            if 'permissions' in json_data:
                self.logger.info(f"✅ Permissions: {json_data['permissions']}")
                return True
            else:
                self.logger.error("❌ Permissions manquantes dans la réponse")
                return False
        else:
            self.logger.error("❌ Échec récupération profil")
            return False
    
    def test_enrichment_sync(self) -> bool:
        """Test 3: Synchronisation enrichment Elasticsearch."""
        self._print_step(3, "SYNCHRONISATION ENRICHMENT ELASTICSEARCH")
        
        if not self.user_id:
            self.logger.error("❌ User ID non disponible")
            return False
        
        response = self._make_request("POST", f"/enrichment/elasticsearch/sync-user/{self.user_id}")
        success, json_data = self._print_response(response)
        
        if success and json_data:
            total_tx = json_data.get('total_transactions', 0)
            indexed = json_data.get('indexed', 0)
            errors = json_data.get('errors', 0)
            processing_time = json_data.get('processing_time', 0)
            
            self.logger.info(f"✅ Transactions totales: {total_tx}")
            self.logger.info(f"✅ Transactions indexées: {indexed}")
            self.logger.info(f"✅ Erreurs: {errors}")
            self.logger.info(f"✅ Temps de traitement: {processing_time:.3f}s")

            if json_data.get('status') == 'success':
                self.logger.info("✅ Synchronisation réussie")
                return True
            else:
                self.logger.error("❌ Synchronisation échouée")
                return False
        else:
            self.logger.error("❌ Échec synchronisation")
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

            self.logger.info(f"✅ Service: {json_data.get('service', 'unknown')}")
            self.logger.info(f"✅ Version: {version}")
            self.logger.info(f"✅ Status: {service_status}")
            self.logger.info(f"✅ Elasticsearch disponible: {elasticsearch_available}")

            if elasticsearch_available:
                cluster_info = json_data.get('elasticsearch', {}).get('cluster_info', {})
                self.logger.info(f"✅ Cluster: {cluster_info.get('cluster_name', 'unknown')}")
                self.logger.info(f"✅ ES Version: {cluster_info.get('version', {}).get('number', 'unknown')}")

            capabilities = json_data.get('capabilities', {})
            self.logger.info(f"✅ Capabilities: {len([k for k, v in capabilities.items() if v])}/{len(capabilities)}")

            if service_status == 'healthy':
                self.logger.info("✅ Service en bonne santé")
                return True
            else:
                self.logger.warning("⚠️ Service en mode dégradé")
                return False
        else:
            self.logger.error("❌ Échec health check")
            return False
    
    def test_search_validation(self) -> bool:
        """Test 5: Validation recherche Netflix."""
        self._print_step(5, "VALIDATION RECHERCHE NETFLIX")
        
        if not self.user_id:
            self.logger.error("❌ User ID non disponible")
            return False
        
        # Recherche Netflix pour valider que les données sont bien indexées
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
            processing_time_ms = metadata.get("processing_time_ms", 0)
            elasticsearch_took = metadata.get("elasticsearch_took", 0)

            self.logger.info(f"✅ Résultats trouvés: {total_results}")
            self.logger.info(f"✅ Résultats retournés: {returned_results}")
            self.logger.info(f"✅ Temps de traitement: {processing_time_ms}ms")
            self.logger.info(f"✅ Temps Elasticsearch: {elasticsearch_took}ms")

            if total_results > 0:
                self.logger.info("✅ Recherche fonctionnelle - Résultats trouvés")
                results = json_data.get('results', [])
                for i, result in enumerate(results[:3]):
                    self.logger.info(f"  📄 Résultat {i+1}:")
                    self.logger.info(f"     • Description: {result.get('primary_description', 'N/A')}")
                    self.logger.info(f"     • Montant: {result.get('amount', 'N/A')} {result.get('currency_code', '')}")
                    self.logger.info(f"     • Date: {result.get('date', 'N/A')}")
                    self.logger.info(f"     • Score: {result.get('score', 'N/A')}")
                return True
            else:
                self.logger.error("❌ Aucun résultat Netflix trouvé dans la recherche directe")
                return False
        else:
            self.logger.error("❌ Échec recherche validation")
            return False
    
    def run_full_test(self, username: str, password: str) -> bool:
        """Lance le test complet avec données réelles."""
        self.logger.info("🚀 DÉBUT DU TEST AUTOMATIQUE HARENA - DONNÉES RÉELLES")
        self.logger.info(f"Base URL: {self.base_url}")
        self.logger.info(f"Username: {username}")
        self.logger.info(f"Timestamp: {datetime.now().isoformat()}")
        self.logger.info(f"📊 Questions basées sur données réelles utilisateur 34")
        
        # Compteur de succès
        tests_passed = 0
        total_tests = 5
        
        # Test 1: Login
        if self.test_login(username, password):
            tests_passed += 1
        else:
            self.logger.error("\n❌ TEST ARRÊTÉ - Impossible de se connecter")
            return False
        
        # Test 2: Profil utilisateur
        if self.test_user_profile():
            tests_passed += 1
        else:
            self.logger.error("\n❌ TEST ARRÊTÉ - Impossible de récupérer le profil")
            return False
        
        # Test 3: Synchronisation enrichment
        if self.test_enrichment_sync():
            tests_passed += 1
        
        # Test 4: Health check enrichment
        if self.test_enrichment_health():
            tests_passed += 1
        
        # Test 5: Validation recherche
        if self.test_search_validation():
            tests_passed += 1

        # Résumé final
        self.logger.info(f"\n{'='*60}")
        self.logger.info("📊 RÉSUMÉ DU TEST AVEC DONNÉES RÉELLES")
        self.logger.info('='*60)
        self.logger.info(f"Tests réussis: {tests_passed}/{total_tests}")
        self.logger.info(f"Pourcentage de réussite: {(tests_passed/total_tests)*100:.1f}%")
        
        # Analyse des performances
        self.logger.info(f"\n🎯 ANALYSE QUESTIONS DONNÉES RÉELLES:")
        for i, q in enumerate(self.real_data_questions[:5]):
            should_find_text = "✅ Devrait trouver" if q["should_find"] else "❌ Test négatif"
            self.logger.info(f"   {i+1}. {q['question']} ({should_find_text})")
            self.logger.info(f"      📊 {q['expected_data']}")

        if tests_passed == total_tests:
            self.logger.info("✅ TOUS LES TESTS SONT PASSÉS - PLATEFORME OPÉRATIONNELLE AVEC DONNÉES RÉELLES")
            return True
        else:
            self.logger.error("❌ CERTAINS TESTS ONT ÉCHOUÉ - VÉRIFIER LA COHÉRENCE DONNÉES RÉELLES")
            return False

def main():
    """Fonction principale."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("harena_test_real_data.log", mode="w", encoding="utf-8")
        ],
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Test automatique Harena Finance Platform - Données Réelles")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                       help=f"URL de base de l'API (défaut: {DEFAULT_BASE_URL})")
    parser.add_argument("--username", default=DEFAULT_USERNAME,
                       help=f"Nom d'utilisateur (défaut: {DEFAULT_USERNAME})")
    parser.add_argument("--password", default=DEFAULT_PASSWORD,
                       help="Mot de passe (défaut: password123)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Mode verbose")

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Créer le client de test
    client = HarenaRealDataTestClient(args.base_url, logger=logger)
    
    # Lancer le test complet
    try:
        success = client.run_full_test(args.username, args.password)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️ Test interrompu par l'utilisateur")
        sys.exit(2)
    except Exception as e:
        logger.error(f"\n\n❌ Erreur inattendue: {e}")
        sys.exit(3)

if __name__ == "__main__":
    main()