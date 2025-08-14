"""
Script de test automatique pour Harena Finance Platform - Avec données réelles utilisateur.

Ce script teste automatiquement la chaîne complète avec des questions basées sur 
les vraies transactions de l'utilisateur 34:
1. Login utilisateur
2. Récupération profil utilisateur
3. Synchronisation enrichment Elasticsearch
4. Health check enrichment service
5. Recherche de transactions
6. Health check conversation service
7. Status conversation service
8. Chat conversation avec questions réelles
9. Tests d'intentions avec données réelles

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
from typing import Dict, Any, Optional, List, Tuple
import logging
import uuid

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
        
        # Questions basées sur les vraies données de l'utilisateur 34
        self.real_data_questions = [
            # Netflix - Nous savons qu'il y a 3 transactions Netflix dans les données
            {
                "question": "Combien j'ai dépensé pour Netflix ce mois ?",
                "expected_intent": "SEARCH_BY_MERCHANT",
                "expected_data": "3 transactions Netflix -17.99€ chacune (avril, mai, juin 2025)",
                "should_find": True
            },
            # Salaire ACME - 2302.2€ mensuel
            {
                "question": "Quel est le montant de mon salaire ACME ?",
                "expected_intent": "SEARCH_BY_MERCHANT", 
                "expected_data": "2302.2€ par mois (avril, mai 2025)",
                "should_find": True
            },
            # McDonald's - Transactions multiples
            {
                "question": "Mes dépenses McDonald's récentes",
                "expected_intent": "SEARCH_BY_MERCHANT",
                "expected_data": "McDonald's 3.7€ (juin), Maroc 5.71€ x2 (avril, mai)",
                "should_find": True
            },
            # Uber Eats - Livraisons
            {
                "question": "Combien j'ai dépensé en livraison Uber Eats ?", 
                "expected_intent": "SEARCH_BY_MERCHANT",
                "expected_data": "25.12€ x2 (mai, juin 2025)",
                "should_find": True
            },
            # Orange - Forfait télé
            {
                "question": "Mes factures Orange télécom",
                "expected_intent": "SEARCH_BY_MERCHANT",
                "expected_data": "19.99€ et 29.99€ par mois (prélèvements)",
                "should_find": True
            },
            # Virement important John Doe
            {
                "question": "Les gros virements vers John Doe",
                "expected_intent": "SEARCH_BY_PERSON",
                "expected_data": "Virement 3000€ (avril, mai 2025)",
                "should_find": True
            },
            # Carrefour/courses
            {
                "question": "Budget courses alimentaires",
                "expected_intent": "SEARCH_BY_CATEGORY",
                "expected_data": "Carrefour, E.Leclerc, Franprix, Naturalia, Monoprix",
                "should_find": True
            },
            # Air France - Voyage
            {
                "question": "Billets d'avion Air France",
                "expected_intent": "SEARCH_BY_MERCHANT",
                "expected_data": "248.71€ (avril, mai 2025)",
                "should_find": True
            },
            # Retraits DAB
            {
                "question": "Combien je retire au distributeur par mois ?",
                "expected_intent": "SEARCH_BY_TYPE",
                "expected_data": "40€-140€ par mois en retraits",
                "should_find": True
            },
            # Recherche qui ne devrait rien donner
            {
                "question": "Mes dépenses chez Tesla",
                "expected_intent": "SEARCH_BY_MERCHANT", 
                "expected_data": "Aucune transaction Tesla",
                "should_find": False
            }
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
        """Test 5: Validation recherche Netflix pour comparaison avec conversation."""
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

    def test_conversation_health(self) -> bool:
        """Test 6: Health check conversation service."""
        self._print_step(6, "HEALTH CHECK CONVERSATION SERVICE")

        response = self._make_request("GET", "/conversation/health", use_auth=False)
        success, json_data = self._print_response(response)

        if success and json_data:
            status = json_data.get('status', 'unknown')
            self.logger.info(f"✅ Service: {json_data.get('service', 'unknown')}")
            self.logger.info(f"✅ Status: {status}")
            if status in ("healthy", "degraded"):
                return True
            else:
                self.logger.error("❌ Statut inattendu")
                return False
        else:
            self.logger.error("❌ Échec health check conversation")
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
            self.logger.info(f"✅ Service: {service}")
            self.logger.info(f"✅ Status: {status}")
            self.logger.info(f"✅ Version: {version}")
            if service and status and version:
                return True
        self.logger.error("❌ Échec status conversation")
        return False

    def test_conversation_real_data(self) -> bool:
        """Test 8: Chat conversation avec données réelles."""
        self._print_step(8, "CHAT CONVERSATION AVEC DONNÉES RÉELLES")

        headers = {"Content-Type": "application/json"}
        conversation_id = f"test-real-data-{uuid.uuid4()}"

        # Test sécurité d'abord
        self.logger.info("🔒 Test d'accès sans jeton: doit retourner 401")
        payload = {"conversation_id": conversation_id, "message": "Test sans token"}
        response = self._make_request(
            "POST", "/conversation/chat", headers=headers, data=json.dumps(payload), use_auth=False
        )
        success, _ = self._print_response(response, expected_status=401)
        if not success:
            self.logger.error("❌ L'appel sans jeton n'a pas retourné 401")
            return False

        # Test avec première question réelle sur Netflix
        netflix_question = self.real_data_questions[0]
        self.logger.info(f"🤖 Question Netflix: {netflix_question['question']}")
        
        payload = {
            "conversation_id": conversation_id,
            "message": netflix_question['question'],
        }
        response = self._make_request(
            "POST", "/conversation/chat", headers=headers, data=json.dumps(payload)
        )
        success, json_data = self._print_response(response)

        if not (success and json_data and json_data.get("success") is True):
            self.logger.error("❌ Échec chat conversation avec données réelles")
            return False

        # Analyser la réponse Netflix
        conversation_response = json_data.get("message", "")
        metadata = json_data.get("metadata", {})
        search_results_count = metadata.get("search_results_count", 0)
        
        self.logger.info(f"📊 Résultats de recherche conversation: {search_results_count}")
        self.logger.info(f"📝 Réponse conversation: {conversation_response[:200]}...")
        
        # Vérifier cohérence avec la recherche directe
        if "Netflix" in conversation_response and search_results_count > 0:
            self.logger.info("✅ Cohérence Search-Conversation VALIDÉE")
        elif "aucune transaction" in conversation_response.lower() and search_results_count == 0:
            self.logger.warning("⚠️ Aucune transaction trouvée par la conversation")
        else:
            self.logger.error("❌ INCOHÉRENCE Search-Conversation détectée")

        # Vérification métriques utilisateur
        if not self.user_id:
            self.logger.error("❌ user_id non disponible")
            return False

        metrics_resp = self._make_request("GET", "/conversation/metrics", use_auth=False)
        metrics_ok, metrics_json = self._print_response(metrics_resp)
        if metrics_ok and metrics_json:
            counters = metrics_json.get("service_metrics", {}).get("counters", {})
            key = f"requests_total{{endpoint=chat,user_id={self.user_id}}}"
            if counters.get(key, 0) < 1:
                self.logger.error("❌ Métriques user_id incohérentes")
                return False
        else:
            self.logger.error("❌ Impossible de vérifier les métriques de conversation")
            return False

        self.logger.info("✅ Chat conversation avec données réelles validé")
        return True

    def test_conversation_real_intents(self) -> bool:
        """Test 9: Tests d'intentions avec données réelles utilisateur."""
        self._print_step(9, "TESTS INTENTIONS AVEC DONNÉES RÉELLES")

        headers = {"Content-Type": "application/json"}
        conversation_id = f"test-real-intents-{uuid.uuid4()}"

        # Sélectionner 5 questions représentatives
        test_questions = self.real_data_questions[:5]
        
        records = []
        
        for i, question_data in enumerate(test_questions):
            question = question_data["question"]
            expected_intent = question_data["expected_intent"] 
            expected_data = question_data["expected_data"]
            should_find = question_data["should_find"]
            
            self.logger.info(f"\n🤖 Question {i+1}: {question}")
            self.logger.info(f"📊 Données attendues: {expected_data}")
            self.logger.info(f"🎯 Intention attendue: {expected_intent}")
            self.logger.info(f"🔍 Devrait trouver: {should_find}")
            
            payload = {"conversation_id": conversation_id, "message": question}
            response = self._make_request(
                "POST", "/conversation/chat", headers=headers, data=json.dumps(payload)
            )
            success, json_data = self._print_response(response)
            
            if not (success and json_data and json_data.get("success") is True):
                self.logger.error(f"❌ Échec conversation pour: {question}")
                return False

            # Analyser la réponse
            conversation_response = json_data.get("message", "")
            metadata = json_data.get("metadata", {})
            intent_result = metadata.get("intent_result", {}) or {}
            detected_intent = intent_result.get("intent_type", "UNKNOWN")
            search_results_count = metadata.get("search_results_count", 0)
            processing_time = json_data.get("processing_time_ms", 0)
            
            # Vérification cohérence résultats/attentes
            coherence_status = "✅"
            if should_find and search_results_count == 0:
                coherence_status = "❌ Devrait trouver"
            elif not should_find and search_results_count > 0:
                coherence_status = "⚠️ Ne devrait pas trouver"
            
            records.append({
                "question": question,
                "expected_intent": expected_intent,
                "detected_intent": detected_intent,
                "expected_data": expected_data,
                "search_results": search_results_count,
                "should_find": should_find,
                "processing_time": processing_time,
                "coherence": coherence_status,
                "response_preview": conversation_response[:100] + "..." if len(conversation_response) > 100 else conversation_response
            })

        # Affichage tableau récapitulatif détaillé
        self.logger.info("\n" + "="*120)
        self.logger.info("📊 RAPPORT DÉTAILLÉ TESTS DONNÉES RÉELLES")
        self.logger.info("="*120)
        
        for i, record in enumerate(records):
            self.logger.info(f"\n🤖 TEST {i+1}: {record['question']}")
            self.logger.info(f"   📊 Données attendues: {record['expected_data']}")
            self.logger.info(f"   🎯 Intention: {record['expected_intent']} → {record['detected_intent']}")
            self.logger.info(f"   🔍 Résultats: {record['search_results']} ({'✅ Devrait trouver' if record['should_find'] else '❌ Ne devrait pas trouver'})")
            self.logger.info(f"   ⏱️ Temps: {record['processing_time']}ms")
            self.logger.info(f"   🎭 Cohérence: {record['coherence']}")
            self.logger.info(f"   💬 Réponse: {record['response_preview']}")

        # Test sécurité historique
        turns_endpoint = f"/conversation/conversations/{conversation_id}/turns"
        resp = self._make_request("GET", turns_endpoint, use_auth=False)
        success, _ = self._print_response(resp, expected_status=401)
        if not success:
            self.logger.error("❌ L'historique sans token n'a pas retourné 401")
            return False

        # Vérification persistance
        resp = self._make_request("GET", turns_endpoint)
        success, json_data = self._print_response(resp)
        if not (
            success
            and json_data
            and json_data.get("conversation_id") == conversation_id
            and isinstance(json_data.get("turns"), list)
        ):
            self.logger.error("❌ Échec récupération des turns")
            return False

        turns = json_data.get("turns", [])
        if len(turns) != len(test_questions):
            self.logger.error("❌ Nombre de turns incohérent")
            return False

        self.logger.info("✅ Tests d'intentions avec données réelles validés")
        return True
    
    def run_full_test(self, username: str, password: str) -> bool:
        """Lance le test complet avec données réelles."""
        self.logger.info("🚀 DÉBUT DU TEST AUTOMATIQUE HARENA - DONNÉES RÉELLES")
        self.logger.info(f"Base URL: {self.base_url}")
        self.logger.info(f"Username: {username}")
        self.logger.info(f"Timestamp: {datetime.now().isoformat()}")
        self.logger.info(f"📊 Questions basées sur données réelles utilisateur 34")
        
        # Compteur de succès
        tests_passed = 0
        total_tests = 9
        
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

        # Test 6: Conversation health
        if self.test_conversation_health():
            tests_passed += 1

        # Test 7: Conversation status
        if self.test_conversation_status():
            tests_passed += 1

        # Test 8: Conversation avec données réelles
        if self.test_conversation_real_data():
            tests_passed += 1

        # Test 9: Intentions avec données réelles
        if self.test_conversation_real_intents():
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