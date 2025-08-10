#!/usr/bin/env python3
"""
🧪 Test Complet Conversation Service

Suite de tests exhaustive pour vérifier que tous les endpoints du conversation_service
sont opérationnels et que le service fonctionne correctement dans son état actuel.

Usage:
    python test_conversation_service_complete.py
    ou
    pytest test_conversation_service_complete.py -v
"""

import asyncio
import json
import time
import uuid
import requests
import pytest
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from conversation_service.models.financial_models import FinancialEntity

# Configuration logging pour les tests
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TestConfig:
    """Configuration des tests"""
    base_url: str = "http://localhost:8001"  # Port par défaut conversation_service
    timeout: int = 10
    max_retries: int = 3
    test_user_id: str = "test_user_123"
    admin_token: str = "admin_test_token"


@dataclass
class EndpointTestResult:
    """Résultat d'un test d'endpoint"""
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    success: bool
    error_message: Optional[str] = None
    response_data: Optional[Dict] = None


class ConversationServiceTester:
    """Testeur complet du service de conversation"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.session = requests.Session()
        self.session.timeout = config.timeout
        self.test_results: List[EndpointTestResult] = []
        
    def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> EndpointTestResult:
        """
        Effectue une requête HTTP et retourne le résultat
        """
        url = f"{self.config.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, params=params, headers=headers)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data, headers=headers, params=params)
            elif method.upper() == "PUT":
                response = self.session.put(url, json=data, headers=headers)
            elif method.upper() == "DELETE":
                response = self.session.delete(url, headers=headers)
            else:
                raise ValueError(f"Méthode HTTP non supportée: {method}")
            
            response_time_ms = (time.time() - start_time) * 1000
            
            # Tentative de parsing JSON
            try:
                response_data = response.json()
            except:
                response_data = {"raw_content": response.text[:500]}
            
            return EndpointTestResult(
                endpoint=endpoint,
                method=method,
                status_code=response.status_code,
                response_time_ms=response_time_ms,
                success=200 <= response.status_code < 300,
                response_data=response_data
            )
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            return EndpointTestResult(
                endpoint=endpoint,
                method=method,
                status_code=0,
                response_time_ms=response_time_ms,
                success=False,
                error_message=str(e)
            )
    
    # =====================================
    # TESTS ENDPOINTS SYSTÈME
    # =====================================
    
    def test_root_endpoint(self) -> EndpointTestResult:
        """Test endpoint racine /"""
        logger.info("🏠 Test endpoint racine")
        result = self._make_request("GET", "/")
        
        if result.success and result.response_data:
            expected_keys = ["service", "version", "endpoints", "features"]
            missing_keys = [k for k in expected_keys if k not in result.response_data]
            
            if missing_keys:
                result.success = False
                result.error_message = f"Clés manquantes: {missing_keys}"
            else:
                logger.info(f"✅ Service: {result.response_data.get('service')}")
                logger.info(f"✅ Version: {result.response_data.get('version')}")
        
        return result
    
    def test_api_status_endpoint(self) -> EndpointTestResult:
        """Test endpoint /api/v1/status"""
        logger.info("📊 Test endpoint /api/v1/status")
        result = self._make_request("GET", "/api/v1/status")
        
        if result.success and result.response_data:
            required_fields = ["status", "timestamp"]
            missing_fields = [f for f in required_fields if f not in result.response_data]
            
            if missing_fields:
                result.success = False
                result.error_message = f"Champs manquants: {missing_fields}"
            elif result.response_data.get("status") != "ok":
                result.success = False
                result.error_message = f"Status incorrect: {result.response_data.get('status')}"
        
        return result
    
    def test_api_version_endpoint(self) -> EndpointTestResult:
        """Test endpoint /api/v1/version"""
        logger.info("🔢 Test endpoint /api/v1/version")
        result = self._make_request("GET", "/api/v1/version")
        
        if result.success and result.response_data:
            required_fields = ["version", "service"]
            missing_fields = [f for f in required_fields if f not in result.response_data]
            
            if missing_fields:
                result.success = False
                result.error_message = f"Champs manquants: {missing_fields}"
        
        return result
    
    def test_legacy_status_endpoint(self) -> EndpointTestResult:
        """Test endpoint legacy /status"""
        logger.info("📊 Test endpoint legacy /status")
        result = self._make_request("GET", "/status")
        
        if result.success and result.response_data:
            # Vérifier que l'endpoint legacy inclut un avertissement de dépréciation
            if not result.response_data.get("deprecated"):
                result.error_message = "Endpoint legacy devrait inclure deprecated=true"
        
        return result
    
    def test_legacy_version_endpoint(self) -> EndpointTestResult:
        """Test endpoint legacy /version"""
        logger.info("🔢 Test endpoint legacy /version")
        return self._make_request("GET", "/version")
    
    # =====================================
    # TESTS ENDPOINTS CORE BUSINESS
    # =====================================
    
    def test_health_endpoint(self) -> EndpointTestResult:
        """Test endpoint /api/v1/health"""
        logger.info("🏥 Test endpoint health")
        result = self._make_request("GET", "/api/v1/health")
        
        if result.success and result.response_data:
            required_fields = ["status", "components", "timestamp"]
            missing_fields = [f for f in required_fields if f not in result.response_data]
            
            if missing_fields:
                result.success = False
                result.error_message = f"Champs manquants: {missing_fields}"
            else:
                # Vérifier statut service
                status = result.response_data.get("status")
                if status not in ["healthy", "degraded", "unhealthy"]:
                    result.success = False
                    result.error_message = f"Status santé invalide: {status}"
                else:
                    logger.info(f"✅ Statut santé: {status}")
        
        return result
    
    def test_metrics_endpoint(self) -> EndpointTestResult:
        """Test endpoint /api/v1/metrics"""
        logger.info("📈 Test endpoint metrics")
        result = self._make_request("GET", "/api/v1/metrics")
        
        if result.success and result.response_data:
            # Vérifier présence des métriques de base
            expected_sections = ["service_metrics", "performance_metrics"]
            
            for section in expected_sections:
                if section not in result.response_data:
                    result.error_message = f"Section métriques manquante: {section}"
                    break
        
        return result
    
    def test_detect_intent_endpoint(self) -> EndpointTestResult:
        """Test endpoint principal /api/v1/detect-intent"""
        logger.info("🎯 Test endpoint detect-intent")
        
        test_payload = {
            "query": "bonjour comment allez-vous",
            "user_id": self.config.test_user_id,
            "use_deepseek_fallback": False,
            "context": {"session_id": "test_session"}
        }
        
        result = self._make_request("POST", "/api/v1/detect-intent", data=test_payload)
        
        if result.success and result.response_data:
            required_fields = [
                "intent_type",
                "intent_category",
                "confidence",
                "entities",
                "processing_time_ms",
            ]
            missing_fields = [f for f in required_fields if f not in result.response_data]

            if missing_fields:
                result.success = False
                result.error_message = f"Champs réponse manquants: {missing_fields}"
            else:
                intent_type = result.response_data.get("intent_type")
                intent_category = result.response_data.get("intent_category")
                confidence = result.response_data.get("confidence")
                latency = result.response_data.get("processing_time_ms")
                entities = result.response_data.get("entities", [])

                logger.info(
                    f"✅ Intent détecté: {intent_type} ({intent_category})"
                )
                logger.info(f"✅ Confiance: {confidence:.3f}")
                logger.info(f"✅ Latence: {latency:.1f}ms")

                if not isinstance(entities, list):
                    result.success = False
                    result.error_message = "Entities should be a list"
                else:
                    for entity in entities:
                        FinancialEntity(**entity)

                # Vérifications basiques
                if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                    result.success = False
                    result.error_message = f"Confiance invalide: {confidence}"
                elif latency > 5000:  # 5 secondes max
                    result.success = False
                    result.error_message = f"Latence trop élevée: {latency}ms"
        
        return result
    
    def test_batch_detect_intent_endpoint(self) -> EndpointTestResult:
        """Test endpoint /api/v1/batch-detect"""
        logger.info("📦 Test endpoint batch-detect")
        
        test_payload = {
            "requests": [
                {
                    "query": "bonjour",
                    "user_id": self.config.test_user_id
                },
                {
                    "query": "quel est mon solde",
                    "user_id": self.config.test_user_id
                },
                {
                    "query": "au revoir",
                    "user_id": self.config.test_user_id
                }
            ]
        }
        
        result = self._make_request("POST", "/api/v1/batch-detect", data=test_payload)
        
        if result.success and result.response_data:
            required_fields = ["results", "batch_metadata"]
            missing_fields = [f for f in required_fields if f not in result.response_data]
            
            if missing_fields:
                result.success = False
                result.error_message = f"Champs batch manquants: {missing_fields}"
            else:
                results = result.response_data.get("results", [])
                if len(results) != 3:
                    result.success = False
                    result.error_message = f"Nombre résultats incorrect: {len(results)} != 3"
                else:
                    logger.info(f"✅ Batch traité: {len(results)} requêtes")
        
        return result
    
    def test_supported_intents_endpoint(self) -> EndpointTestResult:
        """Test endpoint /api/v1/supported-intents"""
        logger.info("📋 Test endpoint supported-intents")
        result = self._make_request("GET", "/api/v1/supported-intents")
        
        if result.success and result.response_data:
            required_fields = ["supported_intents", "intent_details"]
            missing_fields = [f for f in required_fields if f not in result.response_data]
            
            if missing_fields:
                result.success = False
                result.error_message = f"Champs intentions manquants: {missing_fields}"
            else:
                intents = result.response_data.get("supported_intents", [])
                logger.info(f"✅ Intentions supportées: {len(intents)}")
        
        return result
    
    # =====================================
    # TESTS ENDPOINTS ADMIN
    # =====================================
    
    def test_cache_clear_endpoint(self) -> EndpointTestResult:
        """Test endpoint admin /api/v1/cache/clear"""
        logger.info("🗑️ Test endpoint cache/clear")
        
        headers = {"Authorization": f"Bearer {self.config.admin_token}"}
        result = self._make_request("POST", "/api/v1/cache/clear", headers=headers)
        
        # Note: Peut échouer si pas d'auth admin, c'est normal
        if result.status_code == 401:
            logger.info("ℹ️ Endpoint admin nécessite authentification (normal)")
            result.success = True  # On considère ça comme un succès
        
        return result
    
    def test_debug_test_endpoint(self) -> EndpointTestResult:
        """Test endpoint debug /api/v1/debug/test"""
        logger.info("🔧 Test endpoint debug/test")
        
        headers = {"Authorization": f"Bearer {self.config.admin_token}"}
        result = self._make_request("GET", "/api/v1/debug/test", headers=headers)
        
        # Endpoint debug peut être désactivé en production
        if result.status_code == 404:
            logger.info("ℹ️ Endpoint debug désactivé (normal en production)")
            result.success = True
        
        return result
    
    def test_debug_config_endpoint(self) -> EndpointTestResult:
        """Test endpoint debug /api/v1/debug/config"""
        logger.info("⚙️ Test endpoint debug/config")
        result = self._make_request("GET", "/api/v1/debug/config")
        
        # Endpoint debug peut être désactivé
        if result.status_code == 404:
            logger.info("ℹ️ Endpoint debug/config désactivé (normal)")
            result.success = True
        
        return result
    
    # =====================================
    # TESTS DE CHARGE ET PERFORMANCE
    # =====================================
    
    def test_concurrent_requests(self, num_requests: int = 10) -> List[EndpointTestResult]:
        """Test de charge avec requêtes concurrentes"""
        logger.info(f"⚡ Test charge: {num_requests} requêtes concurrentes")
        
        test_queries = [
            "bonjour",
            "quel est mon solde",
            "mes dernières transactions",
            "virement 100 euros",
            "bloquer ma carte",
            "au revoir",
            "aide moi",
            "historique janvier",
            "budget mensuel",
            "dépenses restaurant"
        ]
        
        def make_concurrent_request(i):
            query = test_queries[i % len(test_queries)]
            payload = {
                "query": query,
                "user_id": f"concurrent_user_{i}",
                "use_deepseek_fallback": False
            }
            return self._make_request("POST", "/api/v1/detect-intent", data=payload)
        
        results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_concurrent_request, i) for i in range(num_requests)]
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
        
        # Analyse des résultats
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        avg_latency = sum(r.response_time_ms for r in successful) / len(successful) if successful else 0
        
        logger.info(f"✅ Requêtes réussies: {len(successful)}/{num_requests}")
        logger.info(f"❌ Requêtes échouées: {len(failed)}")
        logger.info(f"⏱️ Latence moyenne: {avg_latency:.1f}ms")
        
        return results
    
    def test_different_query_types(self) -> List[EndpointTestResult]:
        """Test avec différents types de requêtes"""
        logger.info("🎭 Test variété de requêtes")
        
        test_cases = [
            # Salutations
            {"query": "bonjour", "expected_intent_category": "conversational"},
            {"query": "salut comment ça va", "expected_intent_category": "conversational"},
            {"query": "bonsoir", "expected_intent_category": "conversational"},

            # Finance - Solde
            {"query": "quel est mon solde", "expected_intent_category": "financial"},
            {"query": "combien j'ai sur mon compte", "expected_intent_category": "financial"},
            {"query": "mon argent disponible", "expected_intent_category": "financial"},

            # Finance - Transactions
            {"query": "mes derniers achats", "expected_intent_category": "financial"},
            {"query": "historique janvier 2024", "expected_intent_category": "financial"},
            {"query": "dépenses restaurant ce mois", "expected_intent_category": "financial"},

            # Finance - Virements
            {"query": "virer 50 euros à Paul", "expected_intent_category": "financial"},
            {"query": "envoyer de l'argent", "expected_intent_category": "financial"},

            # Finance - Cartes
            {"query": "bloquer ma carte", "expected_intent_category": "financial"},
            {"query": "faire opposition", "expected_intent_category": "financial"},

            # Aide
            {"query": "aide moi", "expected_intent_category": "conversational"},
            {"query": "je ne comprends pas", "expected_intent_category": "conversational"},

            # Au revoir
            {"query": "au revoir", "expected_intent_category": "conversational"},
            {"query": "à bientôt", "expected_intent_category": "conversational"},

            # Requêtes ambigües/inconnues
            {"query": "aksjdhaksjdh", "expected_intent_category": "unknown"},
            {"query": "xyz 123 test", "expected_intent_category": "unknown"}
        ]
        
        results = []
        for i, test_case in enumerate(test_cases):
            payload = {
                "query": test_case["query"],
                "user_id": f"test_variety_{i}",
                "use_deepseek_fallback": False
            }
            
            result = self._make_request("POST", "/api/v1/detect-intent", data=payload)
            result.test_case = test_case
            results.append(result)
            
            if result.success and result.response_data:
                intent_type = result.response_data.get("intent_type")
                confidence = result.response_data.get("confidence", 0)
                logger.info(
                    f"  '{test_case['query'][:30]}' -> {intent_type} ({confidence:.3f})"
                )
        
        return results
    
    # =====================================
    # TESTS EDGE CASES
    # =====================================
    
    def test_edge_cases(self) -> List[EndpointTestResult]:
        """Test cas limites et edge cases"""
        logger.info("🔬 Test cas limites")
        
        edge_cases = [
            # Requête vide
            {"query": "", "description": "requête vide"},
            
            # Requête très courte
            {"query": "a", "description": "requête très courte"},
            
            # Requête très longue
            {"query": "a" * 1000, "description": "requête très longue"},
            
            # Caractères spéciaux
            {"query": "éàçñü特殊字符", "description": "caractères spéciaux"},
            
            # Uniquement espaces
            {"query": "   ", "description": "uniquement espaces"},
            
            # Chiffres uniquement
            {"query": "123456789", "description": "chiffres uniquement"},
            
            # Ponctuation uniquement
            {"query": "!@#$%^&*()", "description": "ponctuation uniquement"}
        ]
        
        results = []
        for i, case in enumerate(edge_cases):
            payload = {
                "query": case["query"],
                "user_id": f"edge_case_{i}",
                "use_deepseek_fallback": False
            }
            
            result = self._make_request("POST", "/api/v1/detect-intent", data=payload)
            result.edge_case = case
            results.append(result)
            
            logger.info(f"  {case['description']}: {'✅' if result.success else '❌'}")
        
        return results
    
    def test_invalid_payloads(self) -> List[EndpointTestResult]:
        """Test avec payloads invalides"""
        logger.info("🚫 Test payloads invalides")
        
        invalid_payloads = [
            # Payload vide
            {},
            
            # Query manquante
            {"user_id": "test"},
            
            # Types incorrects
            {"query": 123, "user_id": "test"},
            {"query": "test", "user_id": 123},
            {"query": "test", "confidence_threshold": "invalid"},
            
            # Valeurs nulles
            {"query": None, "user_id": "test"},
            {"query": "test", "user_id": None},
        ]
        
        results = []
        for i, payload in enumerate(invalid_payloads):
            result = self._make_request("POST", "/api/v1/detect-intent", data=payload)
            result.invalid_payload = payload
            results.append(result)
            
            # Pour les payloads invalides, on s'attend à une erreur 422
            if result.status_code == 422:
                result.success = True  # C'est le comportement attendu
                logger.info(f"  Payload {i}: ✅ (erreur 422 attendue)")
            else:
                logger.info(f"  Payload {i}: ❌ (attendu 422, reçu {result.status_code})")
        
        return results
    
    # =====================================
    # ORCHESTRATEUR DE TESTS
    # =====================================
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Exécute tous les tests et retourne un rapport complet"""
        logger.info("🚀 Démarrage de la suite de tests complète")
        start_time = time.time()
        
        all_results = []
        test_sections = {}
        
        # Tests endpoints système
        logger.info("\n📁 Section: Endpoints système")
        system_tests = [
            self.test_root_endpoint(),
            self.test_api_status_endpoint(),
            self.test_api_version_endpoint(),
            self.test_legacy_status_endpoint(),
            self.test_legacy_version_endpoint()
        ]
        test_sections["system"] = system_tests
        all_results.extend(system_tests)
        
        # Tests endpoints business
        logger.info("\n📁 Section: Endpoints métier")
        business_tests = [
            self.test_health_endpoint(),
            self.test_metrics_endpoint(),
            self.test_detect_intent_endpoint(),
            self.test_batch_detect_intent_endpoint(),
            self.test_supported_intents_endpoint()
        ]
        test_sections["business"] = business_tests
        all_results.extend(business_tests)
        
        # Tests endpoints admin
        logger.info("\n📁 Section: Endpoints admin")
        admin_tests = [
            self.test_cache_clear_endpoint(),
            self.test_debug_test_endpoint(),
            self.test_debug_config_endpoint()
        ]
        test_sections["admin"] = admin_tests
        all_results.extend(admin_tests)
        
        # Tests de performance
        logger.info("\n📁 Section: Tests de performance")
        concurrent_results = self.test_concurrent_requests(10)
        test_sections["performance"] = concurrent_results
        all_results.extend(concurrent_results)
        
        # Tests variété de requêtes
        logger.info("\n📁 Section: Variété de requêtes")
        variety_results = self.test_different_query_types()
        test_sections["variety"] = variety_results
        all_results.extend(variety_results)
        
        # Tests edge cases
        logger.info("\n📁 Section: Cas limites")
        edge_results = self.test_edge_cases()
        test_sections["edge_cases"] = edge_results
        all_results.extend(edge_results)
        
        # Tests payloads invalides
        logger.info("\n📁 Section: Payloads invalides")
        invalid_results = self.test_invalid_payloads()
        test_sections["invalid_payloads"] = invalid_results
        all_results.extend(invalid_results)
        
        # Calcul des statistiques finales
        total_time = time.time() - start_time
        successful_tests = [r for r in all_results if r.success]
        failed_tests = [r for r in all_results if not r.success]
        
        avg_response_time = sum(r.response_time_ms for r in successful_tests) / len(successful_tests) if successful_tests else 0
        
        report = {
            "summary": {
                "total_tests": len(all_results),
                "successful": len(successful_tests),
                "failed": len(failed_tests),
                "success_rate": len(successful_tests) / len(all_results) if all_results else 0,
                "total_duration_seconds": total_time,
                "average_response_time_ms": avg_response_time
            },
            "sections": {
                section: {
                    "total": len(results),
                    "successful": len([r for r in results if r.success]),
                    "failed": len([r for r in results if not r.success]),
                    "results": results
                }
                for section, results in test_sections.items()
            },
            "failed_tests": [
                {
                    "endpoint": r.endpoint,
                    "method": r.method,
                    "error": r.error_message,
                    "status_code": r.status_code
                }
                for r in failed_tests
            ],
            "service_status": "OPERATIONAL" if len(successful_tests) >= len(all_results) * 0.8 else "DEGRADED"
        }
        
        return report
    
    def print_report(self, report: Dict[str, Any]):
        """Affiche le rapport de tests de façon lisible"""
        print("\n" + "="*80)
        print("🧪 RAPPORT DE TESTS CONVERSATION SERVICE")
        print("="*80)
        
        summary = report["summary"]
        print(f"\n📊 RÉSUMÉ GLOBAL:")
        print(f"   Total tests: {summary['total_tests']}")
        print(f"   ✅ Réussis: {summary['successful']}")
        print(f"   ❌ Échoués: {summary['failed']}")
        print(f"   📈 Taux de réussite: {summary['success_rate']:.1%}")
        print(f"   ⏱️ Durée totale: {summary['total_duration_seconds']:.2f}s")
        print(f"   🚀 Temps de réponse moyen: {summary['average_response_time_ms']:.1f}ms")
        print(f"   🎯 Statut service: {report['service_status']}")
        
        print(f"\n📁 DÉTAIL PAR SECTION:")
        for section_name, section_data in report["sections"].items():
            success_rate = section_data["successful"] / section_data["total"] if section_data["total"] > 0 else 0
            print(f"   {section_name.title()}: {section_data['successful']}/{section_data['total']} ({success_rate:.1%})")
        
        if report["failed_tests"]:
            print(f"\n❌ TESTS ÉCHOUÉS:")
            for failed in report["failed_tests"]:
                print(f"   {failed['method']} {failed['endpoint']} (HTTP {failed['status_code']})")
                if failed["error"]:
                    print(f"      Erreur: {failed['error']}")
        
        print(f"\n{'🎉 SERVICE OPÉRATIONNEL' if report['service_status'] == 'OPERATIONAL' else '⚠️ SERVICE DÉGRADÉ'}")
        print("="*80)


def main():
    """Fonction principale pour exécuter les tests"""
    config = TestConfig()
    tester = ConversationServiceTester(config)
    
    print("🧪 Démarrage des tests complets du Conversation Service")
    print(f"🎯 URL de test: {config.base_url}")
    print("="*60)
    
    try:
        # Exécution de tous les tests
        report = tester.run_all_tests()
        
        # Affichage du rapport
        tester.print_report(report)
        
        # Sauvegarde du rapport en JSON
        with open("conversation_service_test_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 Rapport sauvegardé: conversation_service_test_report.json")
        
        # Code de sortie selon le statut
        return 0 if report["service_status"] == "OPERATIONAL" else 1
        
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrompus par l'utilisateur")
        return 2
    except Exception as e:
        print(f"\n💥 Erreur lors des tests: {e}")
        logger.error(f"Erreur critique: {e}", exc_info=True)
        return 3


# =====================================
# TESTS PYTEST (optionnel)
# =====================================

@pytest.mark.skip("requires live conversation service")
class TestConversationServicePytest:
    """
    Classe de tests compatible pytest pour intégration CI/CD
    """
    
    @classmethod
    def setup_class(cls):
        """Setup global des tests pytest"""
        cls.config = TestConfig()
        cls.tester = ConversationServiceTester(cls.config)
    
    def test_service_availability(self):
        """Test disponibilité générale du service"""
        result = self.tester.test_root_endpoint()
        assert result.success, f"Service indisponible: {result.error_message}"
        assert result.response_time_ms < 1000, f"Temps de réponse trop élevé: {result.response_time_ms}ms"
    
    def test_health_check(self):
        """Test endpoint de santé"""
        result = self.tester.test_health_endpoint()
        assert result.success, f"Health check échoué: {result.error_message}"
        
        if result.response_data:
            status = result.response_data.get("status")
            assert status in ["healthy", "degraded"], f"Statut santé invalide: {status}"
    
    def test_intent_detection_basic(self):
        """Test détection d'intention basique"""
        result = self.tester.test_detect_intent_endpoint()
        assert result.success, f"Détection intention échouée: {result.error_message}"
        
        if result.response_data:
            confidence = result.response_data.get("confidence")
            assert isinstance(confidence, (int, float)), "Confiance doit être numérique"
            assert 0 <= confidence <= 1, f"Confiance hors limites: {confidence}"
    
    def test_batch_processing(self):
        """Test traitement par batch"""
        result = self.tester.test_batch_detect_intent_endpoint()
        assert result.success, f"Batch processing échoué: {result.error_message}"
    
    def test_api_consistency(self):
        """Test cohérence des endpoints API"""
        # Test que tous les endpoints /api/v1 sont cohérents
        endpoints_to_test = [
            "/api/v1/status",
            "/api/v1/version", 
            "/api/v1/health",
            "/api/v1/metrics",
            "/api/v1/supported-intents"
        ]
        
        for endpoint in endpoints_to_test:
            result = self.tester._make_request("GET", endpoint)
            assert result.success or result.status_code in [401, 403], f"Endpoint {endpoint} inaccessible: {result.status_code}"
    
    def test_performance_baseline(self):
        """Test performance baseline"""
        concurrent_results = self.tester.test_concurrent_requests(5)
        successful = [r for r in concurrent_results if r.success]
        
        assert len(successful) >= len(concurrent_results) * 0.8, "Trop d'échecs en concurrence"
        
        if successful:
            avg_latency = sum(r.response_time_ms for r in successful) / len(successful)
            assert avg_latency < 2000, f"Latence moyenne trop élevée: {avg_latency}ms"
    
    def test_error_handling(self):
        """Test gestion d'erreurs"""
        invalid_results = self.tester.test_invalid_payloads()
        
        # La plupart des payloads invalides doivent retourner 422
        valid_error_responses = [r for r in invalid_results if r.status_code == 422]
        assert len(valid_error_responses) >= len(invalid_results) * 0.7, "Gestion d'erreurs insuffisante"


# =====================================
# UTILITAIRES ET HELPERS
# =====================================

def check_service_dependencies():
    """
    Vérifie les dépendances du service avant les tests
    """
    print("🔍 Vérification des dépendances...")
    
    dependencies = {
        "requests": "pour les appels HTTP",
        "json": "pour le parsing JSON",
        "concurrent.futures": "pour les tests concurrents"
    }
    
    missing_deps = []
    for dep, description in dependencies.items():
        try:
            __import__(dep)
            print(f"  ✅ {dep}: OK")
        except ImportError:
            print(f"  ❌ {dep}: MANQUANT ({description})")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\n⚠️ Dépendances manquantes: {', '.join(missing_deps)}")
        print("Installez avec: pip install requests pytest")
        return False
    
    return True


def generate_test_data():
    """
    Génère des données de test pour les différents scénarios
    """
    return {
        "financial_queries": [
            "quel est mon solde",
            "mes dernières transactions",
            "virement 100 euros à Paul",
            "dépenses restaurant janvier",
            "bloquer ma carte",
            "historique des virements",
            "budget mensuel",
            "mes revenus",
            "analyse des dépenses",
            "recherche par catégorie alimentation"
        ],
        "conversational_queries": [
            "bonjour",
            "salut comment ça va",
            "bonsoir",
            "aide moi",
            "je ne comprends pas",
            "peux tu m'aider",
            "au revoir",
            "à bientôt",
            "merci beaucoup",
            "comment ça marche"
        ],
        "edge_case_queries": [
            "",
            "a",
            "?",
            "123",
            "éàç",
            "très très très longue requête " * 50,
            "!@#$%",
            "   ",
            "\n\t",
            "query with\nnewlines"
        ]
    }


def benchmark_service_performance():
    """
    Benchmark rapide de performance du service
    """
    print("⚡ Benchmark de performance...")
    
    config = TestConfig()
    tester = ConversationServiceTester(config)
    
    # Test de latence simple
    single_request_times = []
    for i in range(10):
        result = tester._make_request("GET", "/api/v1/health")
        if result.success:
            single_request_times.append(result.response_time_ms)
    
    if single_request_times:
        avg_single = sum(single_request_times) / len(single_request_times)
        print(f"  📊 Latence moyenne (health): {avg_single:.1f}ms")
        print(f"  📊 Latence min/max: {min(single_request_times):.1f}ms / {max(single_request_times):.1f}ms")
    
    # Test de charge
    concurrent_results = tester.test_concurrent_requests(20)
    successful_concurrent = [r for r in concurrent_results if r.success]
    
    if successful_concurrent:
        avg_concurrent = sum(r.response_time_ms for r in successful_concurrent) / len(successful_concurrent)
        success_rate = len(successful_concurrent) / len(concurrent_results)
        print(f"  ⚡ Latence sous charge: {avg_concurrent:.1f}ms")
        print(f"  ⚡ Taux de succès concurrent: {success_rate:.1%}")
    
    return {
        "single_request_avg_ms": avg_single if single_request_times else 0,
        "concurrent_avg_ms": avg_concurrent if successful_concurrent else 0,
        "concurrent_success_rate": success_rate if successful_concurrent else 0
    }


def create_monitoring_dashboard_data(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Crée des données pour un dashboard de monitoring
    """
    return {
        "timestamp": time.time(),
        "service_status": report["service_status"],
        "overall_health": {
            "success_rate": report["summary"]["success_rate"],
            "avg_response_time_ms": report["summary"]["average_response_time_ms"],
            "total_tests": report["summary"]["total_tests"]
        },
        "endpoint_health": {
            section: {
                "availability": data["successful"] / data["total"] if data["total"] > 0 else 0,
                "test_count": data["total"]
            }
            for section, data in report["sections"].items()
        },
        "alerts": [
            f"Test échoué: {test['method']} {test['endpoint']}"
            for test in report["failed_tests"][:5]  # Limite à 5 alertes
        ],
        "performance_metrics": {
            "response_time_threshold_ms": 1000,
            "success_rate_threshold": 0.95,
            "current_success_rate": report["summary"]["success_rate"],
            "current_avg_response_time": report["summary"]["average_response_time_ms"]
        }
    }


if __name__ == "__main__":
    # Vérification des dépendances
    if not check_service_dependencies():
        print("❌ Impossible de continuer sans les dépendances requises")
        exit(1)
    
    # Benchmark rapide optionnel
    if "--benchmark" in __import__("sys").argv:
        benchmark_results = benchmark_service_performance()
        print("\n📊 Résultats benchmark:")
        for key, value in benchmark_results.items():
            print(f"   {key}: {value}")
        print()
    
    # Exécution des tests principaux
    exit_code = main()
    exit(exit_code)