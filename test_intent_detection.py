"""
Test complet pour le nouveau conversation_service TinyBERT
Teste spécifiquement l'endpoint /detect-intent avec métriques détaillées

Focus sur:
- Latence TinyBERT (<50ms objectif)
- Précision intentions financières (>70% objectif)
- Robustesse sur requêtes variées
- Analyse performance détaillée
"""

import requests
import json
import time
import statistics
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
import asyncio
import concurrent.futures

# Configuration URLs
BASE_URL = "http://localhost:8000/api/v1/conversation"
DETECT_INTENT_URL = f"{BASE_URL}/detect-intent"
HEALTH_URL = f"{BASE_URL}/health"
ROOT_URL = f"{BASE_URL}/"

@dataclass
class IntentTestResult:
    """Résultat de test détection intention TinyBERT"""
    query: str
    intent_detected: str
    confidence: float
    processing_time_ms: float
    total_latency_ms: float
    success: bool
    expected_intent: Optional[str] = None
    intent_correct: Optional[bool] = None
    model_used: str = "TinyBERT"
    timestamp: float = 0.0
    error_message: Optional[str] = None

class TinyBERTIntentTester:
    """Testeur spécialisé pour TinyBERT détection intentions"""
    
    def __init__(self):
        self.results: List[IntentTestResult] = []
        self.intent_distribution = Counter()
        self.confidence_ranges = {"low": 0, "medium": 0, "high": 0}
        self.latency_stats = []
        self.accuracy_stats = {"correct": 0, "incorrect": 0, "unknown": 0}
        
    def test_single_intent(self, query: str, expected_intent: str = None, timeout: float = 10.0) -> IntentTestResult:
        """Test une requête unique avec TinyBERT"""
        
        payload = {
            "query": query,
            "user_id": f"test_user_{int(time.time())}"
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(
                DETECT_INTENT_URL,
                headers={'Content-Type': 'application/json'},
                json=payload,
                timeout=timeout
            )
            
            total_latency_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                
                # Extraction données réponse
                intent_detected = data.get("intent", "UNKNOWN")
                confidence = float(data.get("confidence", 0.0))
                processing_time_ms = float(data.get("processing_time_ms", 0.0))
                model_used = data.get("model", "TinyBERT")
                timestamp = data.get("timestamp", time.time())
                
                # Validation intention si attendue
                intent_correct = None
                if expected_intent:
                    intent_correct = (intent_detected == expected_intent)
                    if intent_correct:
                        self.accuracy_stats["correct"] += 1
                    elif intent_detected == "UNKNOWN":
                        self.accuracy_stats["unknown"] += 1
                    else:
                        self.accuracy_stats["incorrect"] += 1
                
                # Classification confiance
                if confidence >= 0.8:
                    self.confidence_ranges["high"] += 1
                elif confidence >= 0.5:
                    self.confidence_ranges["medium"] += 1
                else:
                    self.confidence_ranges["low"] += 1
                
                # Statistiques
                self.intent_distribution[intent_detected] += 1
                self.latency_stats.append(processing_time_ms)
                
                result = IntentTestResult(
                    query=query,
                    intent_detected=intent_detected,
                    confidence=confidence,
                    processing_time_ms=processing_time_ms,
                    total_latency_ms=total_latency_ms,
                    success=True,
                    expected_intent=expected_intent,
                    intent_correct=intent_correct,
                    model_used=model_used,
                    timestamp=timestamp
                )
                
                return result
                
            else:
                return IntentTestResult(
                    query=query,
                    intent_detected="ERROR",
                    confidence=0.0,
                    processing_time_ms=0.0,
                    total_latency_ms=total_latency_ms,
                    success=False,
                    expected_intent=expected_intent,
                    error_message=f"HTTP {response.status_code}: {response.text[:200]}"
                )
                
        except Exception as e:
            return IntentTestResult(
                query=query,
                intent_detected="ERROR",
                confidence=0.0,
                processing_time_ms=0.0,
                total_latency_ms=(time.time() - start_time) * 1000,
                success=False,
                expected_intent=expected_intent,
                error_message=f"Exception: {str(e)}"
            )
    
    def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Suite de tests complète pour TinyBERT"""
        
        print("🤖 TESTS COMPLETS TINYBERT DÉTECTION INTENTIONS")
        print("="*80)
        print(f"🎯 Endpoint: {DETECT_INTENT_URL}")
        print(f"📊 Objectifs: <50ms latence, >70% précision")
        print("="*80)
        
        # Test santé service d'abord
        self._test_service_health()
        
        # ==========================================
        # 1. TESTS INTENTIONS FINANCIÈRES DE BASE
        # ==========================================
        print("\n💰 TESTS INTENTIONS FINANCIÈRES DE BASE")
        print("-" * 60)
        
        basic_financial_tests = [
            ("solde", "BALANCE_CHECK"),
            ("mon solde", "BALANCE_CHECK"),
            ("quel est mon solde", "BALANCE_CHECK"),
            ("combien j'ai sur mon compte", "BALANCE_CHECK"),
            ("argent disponible", "BALANCE_CHECK"),
            
            ("virement", "TRANSFER"),
            ("faire un virement", "TRANSFER"),
            ("transférer de l'argent", "TRANSFER"),
            ("envoyer 100 euros", "TRANSFER"),
            ("virer 50€ à Paul", "TRANSFER"),
            
            ("dépenses", "EXPENSE_ANALYSIS"),
            ("mes dépenses", "EXPENSE_ANALYSIS"),
            ("dépenses restaurant", "EXPENSE_ANALYSIS"),
            ("combien j'ai dépensé", "EXPENSE_ANALYSIS"),
            ("analyse de mes dépenses", "EXPENSE_ANALYSIS"),
            
            ("carte", "CARD_MANAGEMENT"),
            ("ma carte", "CARD_MANAGEMENT"),
            ("bloquer ma carte", "CARD_MANAGEMENT"),
            ("activer carte", "CARD_MANAGEMENT"),
            ("opposition carte", "CARD_MANAGEMENT"),
        ]
        
        for query, expected_intent in basic_financial_tests:
            result = self.test_single_intent(query, expected_intent)
            self.results.append(result)
            self._print_test_result(result)
        
        # ==========================================
        # 2. TESTS INTENTIONS CONVERSATIONNELLES
        # ==========================================
        print("\n💬 TESTS INTENTIONS CONVERSATIONNELLES")
        print("-" * 60)
        
        conversational_tests = [
            ("bonjour", "GREETING"),
            ("salut", "GREETING"),
            ("bonsoir", "GREETING"),
            ("hello", "GREETING"),
            
            ("aide", "HELP"),
            ("help", "HELP"),
            ("comment ça marche", "HELP"),
            ("j'ai besoin d'aide", "HELP"),
            
            ("au revoir", "GOODBYE"),
            ("bye", "GOODBYE"),
            ("à bientôt", "GOODBYE"),
            ("merci et au revoir", "GOODBYE"),
        ]
        
        for query, expected_intent in conversational_tests:
            result = self.test_single_intent(query, expected_intent)
            self.results.append(result)
            self._print_test_result(result)
        
        # ==========================================
        # 3. TESTS REQUÊTES COMPLEXES ET VARIATIONS
        # ==========================================
        print("\n🔄 TESTS REQUÊTES COMPLEXES ET VARIATIONS")
        print("-" * 60)
        
        complex_tests = [
            ("Peux-tu me dire mon solde s'il te plaît ?", "BALANCE_CHECK"),
            ("J'aimerais connaître le montant sur mon compte courant", "BALANCE_CHECK"),
            ("Est-ce que tu peux m'aider à faire un virement urgent ?", "TRANSFER"),
            ("Comment puis-je bloquer ma carte bancaire immédiatement ?", "CARD_MANAGEMENT"),
            ("Montre-moi un résumé de toutes mes dépenses du mois", "EXPENSE_ANALYSIS"),
            ("Bonjour, comment allez-vous aujourd'hui ?", "GREETING"),
            ("Pourriez-vous m'expliquer comment utiliser cette application ?", "HELP"),
        ]
        
        for query, expected_intent in complex_tests:
            result = self.test_single_intent(query, expected_intent)
            self.results.append(result)
            self._print_test_result(result)
        
        # ==========================================
        # 4. TESTS EDGE CASES ET ROBUSTESSE
        # ==========================================
        print("\n🔍 TESTS EDGE CASES ET ROBUSTESSE")
        print("-" * 60)
        
        edge_cases = [
            ("", "UNKNOWN"),
            ("   ", "UNKNOWN"),
            ("a", "UNKNOWN"),
            ("123", "UNKNOWN"),
            ("🏦💰📊", "UNKNOWN"),
            ("qwertyuiop", "UNKNOWN"),
            ("Blablabla test xyz", "UNKNOWN"),
            ("What is the weather today?", "UNKNOWN"),
            ("Comment cuisiner des pâtes ?", "UNKNOWN"),
            ("SOLDE SOLDE SOLDE", "BALANCE_CHECK"),
            ("Solde? Solde! SOLDE.", "BALANCE_CHECK"),
        ]
        
        for query, expected_intent in edge_cases:
            result = self.test_single_intent(query, expected_intent)
            self.results.append(result)
            self._print_test_result(result)
        
        # ==========================================
        # 5. TESTS DE PERFORMANCE ET CHARGE
        # ==========================================
        print(f"\n⚡ TESTS DE PERFORMANCE")
        print("-" * 60)
        
        self._run_performance_tests()
        
        # ==========================================
        # 6. GÉNÉRATION RAPPORT FINAL
        # ==========================================
        return self._generate_comprehensive_report()
    
    def _test_service_health(self):
        """Test santé du service TinyBERT"""
        print("\n🏥 SANTÉ DU SERVICE")
        print("-" * 30)
        
        try:
            response = requests.get(HEALTH_URL, timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                status = health_data.get("status", "unknown")
                model_loaded = health_data.get("model_loaded", False)
                total_requests = health_data.get("total_requests", 0)
                avg_latency = health_data.get("average_latency_ms", 0.0)
                
                print(f"✅ Status: {status}")
                print(f"🤖 Modèle chargé: {'Oui' if model_loaded else 'Non'}")
                print(f"📊 Requêtes totales: {total_requests}")
                print(f"⚡ Latence moyenne: {avg_latency:.2f}ms")
                
                if status != "healthy" or not model_loaded:
                    print("⚠️  Service en mode dégradé - résultats peuvent être affectés")
                    
            else:
                print(f"❌ Health check échoué: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Erreur health check: {e}")
            print("⚠️  Impossible de vérifier l'état du service")
    
    def _run_performance_tests(self):
        """Tests de performance spécialisés"""
        
        # Test latence sur requête standard
        standard_query = "quel est mon solde"
        latencies = []
        
        print(f"🔥 Test latence sur '{standard_query}' (20 requêtes):")
        
        for i in range(20):
            result = self.test_single_intent(standard_query, "BALANCE_CHECK")
            if result.success:
                latencies.append(result.processing_time_ms)
        
        if latencies:
            avg_latency = statistics.mean(latencies)
            median_latency = statistics.median(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            print(f"   Moyenne: {avg_latency:.2f}ms")
            print(f"   Médiane: {median_latency:.2f}ms")  
            print(f"   Min: {min_latency:.2f}ms")
            print(f"   Max: {max_latency:.2f}ms")
            
            # Évaluation performance
            if avg_latency < 30:
                print("   ✅ Performance EXCELLENTE (<30ms)")
            elif avg_latency < 50:
                print("   ✅ Performance BONNE (<50ms)")
            elif avg_latency < 100:
                print("   ⚠️  Performance ACCEPTABLE (<100ms)")
            else:
                print("   ❌ Performance FAIBLE (>100ms)")
        
        # Test concurrence
        print(f"\n🔄 Test concurrence (10 requêtes simultanées):")
        concurrent_results = self._test_concurrent_requests(standard_query, 10)
        
        if concurrent_results:
            successful = [r for r in concurrent_results if r.success]
            success_rate = len(successful) / len(concurrent_results) * 100
            
            if successful:
                concurrent_latencies = [r.processing_time_ms for r in successful]
                avg_concurrent_latency = statistics.mean(concurrent_latencies)
                
                print(f"   Taux succès: {success_rate:.1f}%")
                print(f"   Latence moyenne: {avg_concurrent_latency:.2f}ms")
                
                if success_rate >= 95:
                    print("   ✅ Robustesse concurrence EXCELLENTE")
                elif success_rate >= 80:
                    print("   ✅ Robustesse concurrence BONNE")
                else:
                    print("   ⚠️  Problèmes de concurrence détectés")
    
    def _test_concurrent_requests(self, query: str, num_concurrent: int) -> List[IntentTestResult]:
        """Test requêtes simultanées"""
        
        def make_request():
            return self.test_single_intent(query, "BALANCE_CHECK")
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(make_request) for _ in range(num_concurrent)]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=10)
                    results.append(result)
                except Exception as e:
                    # Créer un résultat d'erreur
                    error_result = IntentTestResult(
                        query=query,
                        intent_detected="ERROR",
                        confidence=0.0,
                        processing_time_ms=0.0,
                        total_latency_ms=0.0,
                        success=False,
                        error_message=f"Concurrent test error: {str(e)}"
                    )
                    results.append(error_result)
        
        return results
    
    def _print_test_result(self, result: IntentTestResult):
        """Affichage formaté du résultat"""
        
        # Emojis selon intention
        intent_emojis = {
            "BALANCE_CHECK": "💰",
            "TRANSFER": "💸", 
            "EXPENSE_ANALYSIS": "📊",
            "CARD_MANAGEMENT": "💳",
            "GREETING": "👋",
            "HELP": "❓",
            "GOODBYE": "👋",
            "UNKNOWN": "❔",
            "ERROR": "❌"
        }
        
        emoji = intent_emojis.get(result.intent_detected, "❓")
        
        # Status intention
        if result.intent_correct is True:
            intent_status = "✅"
        elif result.intent_correct is False:
            intent_status = "❌"
        else:
            intent_status = "➖"
        
        # Status performance
        if result.success and result.processing_time_ms < 50:
            perf_status = "⚡"
        elif result.success and result.processing_time_ms < 100:
            perf_status = "🔄"
        else:
            perf_status = "⏱️"
        
        # Affichage principal
        query_display = result.query[:40].ljust(40)
        intent_display = result.intent_detected.ljust(15)
        
        print(f"{emoji} {query_display} → {intent_display} "
              f"({result.confidence:.3f}) {result.processing_time_ms:6.1f}ms "
              f"{intent_status}{perf_status}")
        
        # Détails erreur si nécessaire
        if not result.success and result.error_message:
            print(f"    ❌ Erreur: {result.error_message[:60]}")
    
    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Génération du rapport complet"""
        
        total_tests = len(self.results)
        successful_tests = [r for r in self.results if r.success]
        failed_tests = [r for r in self.results if not r.success]
        
        print("\n" + "="*80)
        print("📊 RAPPORT COMPLET TINYBERT DÉTECTION INTENTIONS")
        print("="*80)
        
        # ==========================================
        # MÉTRIQUES GLOBALES
        # ==========================================
        print(f"\n📈 MÉTRIQUES GLOBALES:")
        print(f"   Tests total: {total_tests}")
        print(f"   Tests réussis: {len(successful_tests)}")
        print(f"   Tests échoués: {len(failed_tests)}")
        print(f"   Taux de succès: {len(successful_tests)/total_tests*100:.1f}%")
        
        # ==========================================
        # PERFORMANCE LATENCE
        # ==========================================
        if self.latency_stats:
            avg_latency = statistics.mean(self.latency_stats)
            median_latency = statistics.median(self.latency_stats)
            min_latency = min(self.latency_stats)
            max_latency = max(self.latency_stats)
            
            print(f"\n⚡ PERFORMANCE LATENCE:")
            print(f"   Latence moyenne: {avg_latency:.2f}ms")
            print(f"   Latence médiane: {median_latency:.2f}ms")
            print(f"   Latence min: {min_latency:.2f}ms")
            print(f"   Latence max: {max_latency:.2f}ms")
            
            # Évaluation objectif <50ms
            fast_requests = sum(1 for lat in self.latency_stats if lat < 50)
            fast_percentage = fast_requests / len(self.latency_stats) * 100
            print(f"   Requêtes <50ms: {fast_requests}/{len(self.latency_stats)} ({fast_percentage:.1f}%)")
            
            if avg_latency < 50:
                print("   ✅ OBJECTIF LATENCE ATTEINT (<50ms)")
            else:
                print("   ❌ OBJECTIF LATENCE NON ATTEINT (>50ms)")
        
        # ==========================================
        # PRÉCISION INTENTIONS
        # ==========================================
        total_with_expected = sum(1 for r in self.results if r.expected_intent)
        
        if total_with_expected > 0:
            accuracy_rate = self.accuracy_stats["correct"] / total_with_expected * 100
            
            print(f"\n🎯 PRÉCISION INTENTIONS:")
            print(f"   Intentions correctes: {self.accuracy_stats['correct']}")
            print(f"   Intentions incorrectes: {self.accuracy_stats['incorrect']}")
            print(f"   Intentions inconnues: {self.accuracy_stats['unknown']}")
            print(f"   Taux de précision: {accuracy_rate:.1f}%")
            
            if accuracy_rate >= 70:
                print("   ✅ OBJECTIF PRÉCISION ATTEINT (≥70%)")
            else:
                print("   ❌ OBJECTIF PRÉCISION NON ATTEINT (<70%)")
        
        # ==========================================
        # DISTRIBUTION INTENTIONS
        # ==========================================
        print(f"\n📊 DISTRIBUTION INTENTIONS:")
        total_successful = len(successful_tests)
        for intent, count in self.intent_distribution.most_common():
            percentage = count / total_successful * 100 if total_successful > 0 else 0
            print(f"   {intent}: {count} ({percentage:.1f}%)")
        
        # ==========================================
        # DISTRIBUTION CONFIANCE
        # ==========================================
        total_confidence = sum(self.confidence_ranges.values())
        if total_confidence > 0:
            print(f"\n🎲 DISTRIBUTION CONFIANCE:")
            high_pct = self.confidence_ranges["high"] / total_confidence * 100
            medium_pct = self.confidence_ranges["medium"] / total_confidence * 100
            low_pct = self.confidence_ranges["low"] / total_confidence * 100
            
            print(f"   Confiance haute (≥0.8): {self.confidence_ranges['high']} ({high_pct:.1f}%)")
            print(f"   Confiance moyenne (0.5-0.8): {self.confidence_ranges['medium']} ({medium_pct:.1f}%)")
            print(f"   Confiance faible (<0.5): {self.confidence_ranges['low']} ({low_pct:.1f}%)")
        
        # ==========================================
        # ANALYSE DES ÉCHECS
        # ==========================================
        if failed_tests:
            print(f"\n❌ ANALYSE DES ÉCHECS:")
            error_types = Counter(r.error_message.split(':')[0] if r.error_message else "Unknown" 
                                for r in failed_tests)
            
            for error_type, count in error_types.items():
                print(f"   {error_type}: {count}")
        
        # ==========================================
        # RECOMMANDATIONS
        # ==========================================
        recommendations = self._generate_recommendations()
        print(f"\n💡 RECOMMANDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        # ==========================================
        # VERDICT FINAL
        # ==========================================
        print(f"\n" + "="*80)
        print("🏆 VERDICT FINAL")
        print("="*80)
        
        success_rate = len(successful_tests) / total_tests * 100
        meets_latency = statistics.mean(self.latency_stats) < 50 if self.latency_stats else False
        meets_accuracy = (self.accuracy_stats["correct"] / max(total_with_expected, 1) * 100) >= 70
        
        if success_rate >= 95 and meets_latency and meets_accuracy:
            print("🟢 EXCELLENT - TinyBERT prêt pour production")
        elif success_rate >= 80 and (meets_latency or meets_accuracy):
            print("🟡 BON - TinyBERT nécessite optimisations mineures")
        else:
            print("🔴 INSUFFISANT - TinyBERT nécessite améliorations importantes")
        
        return {
            "summary": {
                "total_tests": total_tests,
                "successful_tests": len(successful_tests),
                "failed_tests": len(failed_tests),
                "success_rate": success_rate,
                "avg_latency_ms": statistics.mean(self.latency_stats) if self.latency_stats else 0,
                "accuracy_rate": self.accuracy_stats["correct"] / max(total_with_expected, 1) * 100,
                "meets_latency_target": meets_latency,
                "meets_accuracy_target": meets_accuracy
            },
            "performance": {
                "latency_stats": {
                    "mean": statistics.mean(self.latency_stats) if self.latency_stats else 0,
                    "median": statistics.median(self.latency_stats) if self.latency_stats else 0,
                    "min": min(self.latency_stats) if self.latency_stats else 0,
                    "max": max(self.latency_stats) if self.latency_stats else 0,
                },
                "fast_requests_percentage": sum(1 for lat in self.latency_stats if lat < 50) / len(self.latency_stats) * 100 if self.latency_stats else 0
            },
            "accuracy": dict(self.accuracy_stats),
            "intent_distribution": dict(self.intent_distribution),
            "confidence_distribution": dict(self.confidence_ranges),
            "recommendations": recommendations,
            "detailed_results": [
                {
                    "query": r.query,
                    "intent_detected": r.intent_detected,
                    "expected_intent": r.expected_intent,
                    "confidence": r.confidence,
                    "processing_time_ms": r.processing_time_ms,
                    "success": r.success,
                    "intent_correct": r.intent_correct
                }
                for r in self.results
            ]
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Génération recommandations basées sur l'analyse"""
        recommendations = []
        
        # Recommandations latence
        if self.latency_stats:
            avg_latency = statistics.mean(self.latency_stats)
            if avg_latency > 100:
                recommendations.append("Optimiser TinyBERT - latence trop élevée (>100ms)")
            elif avg_latency > 50:
                recommendations.append("Améliorer performance TinyBERT - objectif <50ms non atteint")
        
        # Recommandations précision
        total_with_expected = sum(1 for r in self.results if r.expected_intent)
        if total_with_expected > 0:
            accuracy_rate = self.accuracy_stats["correct"] / total_with_expected * 100
            if accuracy_rate < 50:
                recommendations.append("Fine-tuner TinyBERT sur données françaises - précision critique")
            elif accuracy_rate < 70:
                recommendations.append("Améliorer dataset d'entraînement TinyBERT - objectif 70% non atteint")
        
        # Recommandations UNKNOWN
        unknown_rate = self.intent_distribution.get("UNKNOWN", 0) / len(self.results) * 100
        if unknown_rate > 30:
            recommendations.append("Réduire taux UNKNOWN - ajouter plus d'exemples d'entraînement")
        
        # Recommandations confiance
        low_confidence = self.confidence_ranges.get("low", 0)
        total_confidence = sum(self.confidence_ranges.values())
        if total_confidence > 0 and low_confidence / total_confidence > 0.5:
            recommendations.append("Améliorer confiance modèle - trop de prédictions incertaines")
        
        # Recommandations échecs
        failed_tests = [r for r in self.results if not r.success]
        if len(failed_tests) > len(self.results) * 0.1:
            recommendations.append("Résoudre problèmes techniques - taux d'échec élevé")
        
        if not recommendations:
            recommendations.append("TinyBERT fonctionne correctement - prêt pour mise en production")
        
        return recommendations

def test_service_availability():
    """Test disponibilité du service"""
    print("🔍 VÉRIFICATION DISPONIBILITÉ SERVICE")
    print("-" * 50)
    
    try:
        # Test endpoint root
        response = requests.get(ROOT_URL, timeout=5)
        if response.status_code == 200:
            print("✅ Service disponible")
            data = response.json()
            print(f"   Service: {data.get('service', 'Unknown')}")
            print(f"   Version: {data.get('version', 'Unknown')}")
            print(f"   Modèle: {data.get('model', 'Unknown')}")
        else:
            print(f"❌ Service indisponible: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur connexion service: {e}")
        return False
    
    return True

def main():
    """Fonction principale des tests TinyBERT"""
    print("🤖 TESTS COMPLETS TINYBERT DÉTECTION INTENTIONS")
    print("="*80)
    print(f"⏰ Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Endpoint: {DETECT_INTENT_URL}")
    print(f"📊 Objectifs: <50ms latence, >70% précision intentions")
    
    # Vérifier disponibilité service
    if not test_service_availability():
        print("\n❌ Service indisponible - arrêt des tests")
        return None
    
    # Lancer suite de tests
    tester = TinyBERTIntentTester()
    report = tester.run_comprehensive_test_suite()
    
    # Sauvegarder rapport
    timestamp = int(time.time())
    report_filename = f"tinybert_test_report_{timestamp}.json"
    
    try:
        with open(report_filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n💾 Rapport sauvegardé: {report_filename}")
    except Exception as e:
        print(f"\n❌ Erreur sauvegarde rapport: {e}")
    
    # Conclusion finale
    print("\n" + "="*80)
    print("🎯 CONCLUSION ET ACTIONS RECOMMANDÉES")
    print("="*80)
    
    summary = report["summary"]
    success_rate = summary["success_rate"]
    avg_latency = summary["avg_latency_ms"]
    accuracy_rate = summary["accuracy_rate"]
    meets_latency = summary["meets_latency_target"]
    meets_accuracy = summary["meets_accuracy_target"]
    
    # Status global
    if success_rate >= 95 and meets_latency and meets_accuracy:
        print("🟢 STATUS: EXCELLENT")
        print("   TinyBERT est prêt pour la production")
        print("   ✅ Latence cible atteinte")
        print("   ✅ Précision cible atteinte")
        print("   ✅ Taux de succès élevé")
        action = "DÉPLOYER EN PRODUCTION"
        
    elif success_rate >= 80 and (meets_latency or meets_accuracy):
        print("🟡 STATUS: BON")
        print("   TinyBERT nécessite optimisations mineures")
        if not meets_latency:
            print("   ⚠️  Objectif latence non atteint")
        if not meets_accuracy:
            print("   ⚠️  Objectif précision non atteint")
        action = "OPTIMISER PUIS DÉPLOYER"
        
    else:
        print("🔴 STATUS: INSUFFISANT")
        print("   TinyBERT nécessite améliorations importantes")
        print(f"   ❌ Taux succès: {success_rate:.1f}% (objectif: >95%)")
        if not meets_latency:
            print(f"   ❌ Latence moyenne: {avg_latency:.1f}ms (objectif: <50ms)")
        if not meets_accuracy:
            print(f"   ❌ Précision: {accuracy_rate:.1f}% (objectif: >70%)")
        action = "AMÉLIORER AVANT DÉPLOIEMENT"
    
    print(f"\n🚀 ACTION RECOMMANDÉE: {action}")
    
    # Métriques clés finales
    print(f"\n📊 MÉTRIQUES CLÉS:")
    print(f"   Taux de succès: {success_rate:.1f}%")
    print(f"   Latence moyenne: {avg_latency:.1f}ms")
    print(f"   Précision intentions: {accuracy_rate:.1f}%")
    print(f"   Tests réussis: {summary['successful_tests']}/{summary['total_tests']}")
    
    # Prochaines étapes
    print(f"\n📋 PROCHAINES ÉTAPES:")
    if meets_latency and meets_accuracy:
        print("   1. Valider avec données réelles utilisateurs")
        print("   2. Tester charge en production")
        print("   3. Monitorer métriques en continu")
        print("   4. Optimiser patterns si nécessaire")
    else:
        print("   1. Analyser cas d'échec détaillés")
        print("   2. Fine-tuner modèle sur données françaises")
        print("   3. Optimiser architecture si latence élevée")
        print("   4. Re-tester jusqu'à atteinte objectifs")
    
    return report

def run_quick_test():
    """Test rapide pour vérification de base"""
    print("⚡ TEST RAPIDE TINYBERT")
    print("-" * 30)
    
    tester = TinyBERTIntentTester()
    
    # Quelques tests de base
    quick_tests = [
        ("bonjour", "GREETING"),
        ("quel est mon solde", "BALANCE_CHECK"),
        ("faire un virement", "TRANSFER"),
        ("mes dépenses", "EXPENSE_ANALYSIS"),
        ("bloquer ma carte", "CARD_MANAGEMENT"),
        ("aide", "HELP"),
        ("au revoir", "GOODBYE"),
        ("test xyz", "UNKNOWN")
    ]
    
    print("Tests en cours...")
    for query, expected in quick_tests:
        result = tester.test_single_intent(query, expected)
        tester.results.append(result)
        
        status = "✅" if result.success and result.intent_correct else "❌"
        print(f"{status} '{query}' → {result.intent_detected} ({result.confidence:.2f}) {result.processing_time_ms:.1f}ms")
    
    # Métriques rapides
    successful = [r for r in tester.results if r.success]
    correct_intents = [r for r in tester.results if r.intent_correct]
    
    if successful:
        avg_latency = sum(r.processing_time_ms for r in successful) / len(successful)
        accuracy = len(correct_intents) / len(tester.results) * 100
        
        print(f"\n📊 Résultats rapides:")
        print(f"   Succès: {len(successful)}/{len(tester.results)} ({len(successful)/len(tester.results)*100:.1f}%)")
        print(f"   Précision: {accuracy:.1f}%")
        print(f"   Latence moyenne: {avg_latency:.1f}ms")
        
        if avg_latency < 50 and accuracy > 70:
            print("✅ Test rapide RÉUSSI - TinyBERT fonctionne correctement")
        else:
            print("⚠️  Test rapide PARTIEL - optimisations recommandées")
    else:
        print("❌ Test rapide ÉCHOUÉ - problème technique")

def run_load_test():
    """Test de charge spécialisé"""
    print("🔥 TEST DE CHARGE TINYBERT")
    print("-" * 40)
    
    test_query = "quel est mon solde"
    num_requests = 50
    
    print(f"🎯 Requête test: '{test_query}'")
    print(f"📊 Nombre de requêtes: {num_requests}")
    print("⏱️  Test en cours...")
    
    tester = TinyBERTIntentTester()
    start_time = time.time()
    
    # Test séquentiel
    sequential_latencies = []
    for i in range(num_requests):
        result = tester.test_single_intent(test_query, "BALANCE_CHECK")
        if result.success:
            sequential_latencies.append(result.processing_time_ms)
    
    sequential_duration = time.time() - start_time
    
    # Test concurrent
    print("🔄 Test concurrent...")
    concurrent_results = tester._test_concurrent_requests(test_query, min(num_requests, 20))
    concurrent_latencies = [r.processing_time_ms for r in concurrent_results if r.success]
    
    # Résultats
    print(f"\n📊 RÉSULTATS TEST DE CHARGE:")
    
    if sequential_latencies:
        print(f"🔄 Séquentiel ({len(sequential_latencies)} requêtes):")
        print(f"   Latence moyenne: {statistics.mean(sequential_latencies):.2f}ms")
        print(f"   Latence médiane: {statistics.median(sequential_latencies):.2f}ms")
        print(f"   Temps total: {sequential_duration:.2f}s")
        print(f"   Débit: {len(sequential_latencies)/sequential_duration:.1f} req/s")
    
    if concurrent_latencies:
        success_rate = len(concurrent_latencies) / len(concurrent_results) * 100
        print(f"⚡ Concurrent ({len(concurrent_results)} requêtes):")
        print(f"   Taux succès: {success_rate:.1f}%")
        print(f"   Latence moyenne: {statistics.mean(concurrent_latencies):.2f}ms")
        print(f"   Latence médiane: {statistics.median(concurrent_latencies):.2f}ms")
        
        if success_rate >= 95:
            print("   ✅ Excellente robustesse concurrentielle")
        elif success_rate >= 80:
            print("   ✅ Bonne robustesse concurrentielle")
        else:
            print("   ⚠️  Problèmes de concurrence détectés")
    
    # Recommandations performance
    if sequential_latencies:
        p95_latency = sorted(sequential_latencies)[int(0.95 * len(sequential_latencies))]
        print(f"\n💡 Analyse performance:")
        print(f"   P95 latence: {p95_latency:.2f}ms")
        
        if p95_latency < 50:
            print("   ✅ Performance P95 excellente (<50ms)")
        elif p95_latency < 100:
            print("   ✅ Performance P95 acceptable (<100ms)")
        else:
            print("   ⚠️  Performance P95 à améliorer (>100ms)")

if __name__ == "__main__":
    import sys
    
    # Options de test
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            run_quick_test()
        elif sys.argv[1] == "--load":
            run_load_test()
        elif sys.argv[1] == "--help":
            print("🤖 Tests TinyBERT Détection Intentions")
            print("")
            print("Usage:")
            print("  python test_tinybert_intentions.py           # Test complet")
            print("  python test_tinybert_intentions.py --quick   # Test rapide")
            print("  python test_tinybert_intentions.py --load    # Test de charge")
            print("  python test_tinybert_intentions.py --help    # Cette aide")
        else:
            print(f"❌ Option inconnue: {sys.argv[1]}")
            print("Utilisez --help pour voir les options disponibles")
    else:
        # Test complet par défaut
        report = main()
        
        # Code de sortie selon résultats
        if report:
            summary = report["summary"]
            if summary["meets_latency_target"] and summary["meets_accuracy_target"] and summary["success_rate"] >= 95:
                sys.exit(0)  # Succès complet
            elif summary["success_rate"] >= 80:
                sys.exit(1)  # Succès partiel
            else:
                sys.exit(2)  # Échec
        else:
            sys.exit(3)  # Erreur technique