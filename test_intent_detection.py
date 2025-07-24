import requests
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter

# Configuration
BASE_URL = "http://localhost:8000/api/v1/conversation"
CHAT_URL = f"{BASE_URL}/chat"
HEALTH_URL = f"{BASE_URL}/health"
METRICS_URL = f"{BASE_URL}/metrics"
DEBUG_URL = f"{BASE_URL}/debug/test-levels"

@dataclass
class TestResult:
    """Résultat de test avec diagnostic détaillé"""
    message: str
    intent: str
    confidence: float
    level_used: str
    processing_time_ms: float
    cache_hit: bool
    success: bool
    expected_level: Optional[str] = None
    expected_intent: Optional[str] = None
    fallback_reason: Optional[str] = None
    diagnostic: Optional[Dict[str, Any]] = None

class ConversationServiceTester:
    """Testeur avancé avec diagnostic des fallbacks"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.fallback_reasons = Counter()
        self.level_distribution = Counter()
        self.intent_accuracy = {"correct": 0, "incorrect": 0, "unknown": 0}
        
    def test_query(
        self, 
        message: str, 
        expected_level: str = None, 
        expected_intent: str = None,
        force_level: str = None
    ) -> TestResult:
        """Test une requête avec diagnostic fallback"""
        
        payload = {
            "message": message,
            "user_id": 34,
            "conversation_id": f"test_{int(time.time())}"
        }
        
        # URL selon force level
        url = DEBUG_URL if force_level else CHAT_URL
        if force_level:
            payload["force_level"] = force_level
        
        try:
            start_time = time.time()
            response = requests.post(
                url,
                headers={'Content-Type': 'application/json'},
                json=payload,
                timeout=10
            )
            total_latency = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                
                # Extraction données base
                intent = data.get("intent", "UNKNOWN")
                confidence = float(data.get("confidence", 0.0))
                processing_metadata = data.get("processing_metadata", {})
                level_used = processing_metadata.get("level_used", "UNKNOWN")
                processing_time = processing_metadata.get("processing_time_ms", total_latency)
                cache_hit = processing_metadata.get("cache_hit", False)
                
                # Diagnostic fallback
                fallback_reason = None
                diagnostic = {}
                
                if level_used == "ERROR_FALLBACK":
                    fallback_reason = self._diagnose_fallback(message, data, expected_level)
                    diagnostic = self._detailed_diagnostic(message, data)
                
                # Validation intention
                intent_correct = None
                if expected_intent:
                    intent_correct = (intent == expected_intent)
                    if intent_correct:
                        self.intent_accuracy["correct"] += 1
                    elif intent == "UNKNOWN":
                        self.intent_accuracy["unknown"] += 1
                    else:
                        self.intent_accuracy["incorrect"] += 1
                
                result = TestResult(
                    message=message,
                    intent=intent,
                    confidence=confidence,
                    level_used=level_used,
                    processing_time_ms=processing_time,
                    cache_hit=cache_hit,
                    success=True,
                    expected_level=expected_level,
                    expected_intent=expected_intent,
                    fallback_reason=fallback_reason,
                    diagnostic=diagnostic
                )
                
                # Comptage statistiques
                self.level_distribution[level_used] += 1
                if fallback_reason:
                    self.fallback_reasons[fallback_reason] += 1
                
                return result
                
            else:
                return TestResult(
                    message=message,
                    intent="ERROR",
                    confidence=0.0,
                    level_used="ERROR_HTTP",
                    processing_time_ms=total_latency,
                    cache_hit=False,
                    success=False,
                    fallback_reason=f"HTTP {response.status_code}",
                    diagnostic={"http_error": response.text[:200]}
                )
                
        except Exception as e:
            return TestResult(
                message=message,
                intent="ERROR",
                confidence=0.0,
                level_used="ERROR_EXCEPTION",
                processing_time_ms=0.0,
                cache_hit=False,
                success=False,
                fallback_reason=f"Exception: {str(e)}",
                diagnostic={"exception": str(e)}
            )
    
    def _diagnose_fallback(self, message: str, response_data: Dict, expected_level: str) -> str:
        """Diagnostic spécifique pourquoi la requête tombe en fallback"""
        
        # Analyse du message
        message_length = len(message.strip())
        has_special_chars = any(c in message for c in "!@#$%^&*()[]{}|\\:;\"'<>?")
        has_numbers = any(c.isdigit() for c in message)
        word_count = len(message.split())
        
        # Patterns de diagnostic
        if message_length == 0:
            return "message_empty"
        elif message_length < 3:
            return "message_too_short"
        elif message_length > 200:
            return "message_too_long"
        elif word_count == 1:
            if expected_level == "L0_PATTERN":
                return "l0_single_word_not_matched"
            else:
                return "single_word_ambiguous"
        elif has_special_chars and not has_numbers:
            return "special_characters_interference"
        elif "?" in message and expected_level == "L0_PATTERN":
            return "l0_question_pattern_missing"
        elif any(word in message.lower() for word in ["restaurant", "solde", "virement", "carte"]):
            if expected_level == "L0_PATTERN":
                return "l0_financial_keywords_not_matched"
            else:
                return "l1_classification_failed"
        elif expected_level == "L0_PATTERN":
            return "l0_pattern_not_found"
        elif expected_level == "L1_LIGHTWEIGHT":
            return "l1_confidence_too_low"
        elif expected_level == "L2_LLM":
            return "l2_llm_unavailable_or_failed"
        else:
            return "unknown_fallback_reason"
    
    def _detailed_diagnostic(self, message: str, response_data: Dict) -> Dict[str, Any]:
        """Diagnostic détaillé pour debugging"""
        
        processing_metadata = response_data.get("processing_metadata", {})
        
        diagnostic = {
            "message_analysis": {
                "length": len(message),
                "word_count": len(message.split()),
                "has_financial_keywords": any(word in message.lower() for word in [
                    "solde", "compte", "virement", "transfert", "dépenses", "restaurant", 
                    "carte", "facture", "budget", "épargne", "argent"
                ]),
                "has_question_words": any(word in message.lower() for word in [
                    "quel", "combien", "comment", "pourquoi", "quand", "où"
                ]),
                "has_action_words": any(word in message.lower() for word in [
                    "voir", "faire", "bloquer", "activer", "payer", "envoyer"
                ]),
                "normalized": message.lower().strip()
            },
            "response_analysis": {
                "processing_time_ms": processing_metadata.get("processing_time_ms", 0),
                "engine_latency_ms": processing_metadata.get("engine_latency_ms", 0),
                "cache_hit": processing_metadata.get("cache_hit", False),
                "timestamp": processing_metadata.get("timestamp", 0)
            },
            "potential_issues": []
        }
        
        # Identification problèmes potentiels
        if len(message) < 5:
            diagnostic["potential_issues"].append("Message trop court pour classification fiable")
        
        if not diagnostic["message_analysis"]["has_financial_keywords"]:
            diagnostic["potential_issues"].append("Aucun mot-clé financier détecté")
        
        if "ERROR_FALLBACK" in str(response_data):
            diagnostic["potential_issues"].append("Système en mode fallback - vérifier santé services")
        
        return diagnostic
    
    def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Suite de tests complète avec diagnostic"""
        
        print("🚀 Démarrage tests diagnostiques conversation_service")
        print(f"📡 URL: {CHAT_URL}")
        print("="*80)
        
        # ==========================================
        # 1. TESTS L0 - PATTERNS SIMPLES
        # ==========================================
        print("\n⚡ TESTS L0 - PATTERNS SIMPLES")
        print("-" * 50)
        
        l0_tests = [
            ("solde", "L0_PATTERN", "BALANCE_CHECK"),
            ("compte", "L0_PATTERN", "BALANCE_CHECK"),
            ("virement", "L0_PATTERN", "TRANSFER"),
            ("carte", "L0_PATTERN", "CARD_MANAGEMENT"),
            ("dépenses", "L0_PATTERN", "EXPENSE_ANALYSIS"),
            ("bonjour", "L0_PATTERN", "GREETING"),
            ("aide", "L0_PATTERN", "HELP"),
        ]
        
        for message, expected_level, expected_intent in l0_tests:
            result = self.test_query(message, expected_level, expected_intent)
            self.results.append(result)
            self._print_test_result(result)
        
        # ==========================================
        # 2. TESTS L1 - REQUÊTES STRUCTURÉES
        # ==========================================
        print("\n🔥 TESTS L1 - REQUÊTES STRUCTURÉES")
        print("-" * 50)
        
        l1_tests = [
            ("quel est mon solde", "L1_LIGHTWEIGHT", "BALANCE_CHECK"),
            ("mes dépenses restaurant", "L1_LIGHTWEIGHT", "EXPENSE_ANALYSIS"),
            ("faire un virement", "L1_LIGHTWEIGHT", "TRANSFER"),
            ("bloquer ma carte", "L1_LIGHTWEIGHT", "CARD_MANAGEMENT"),
            ("combien j'ai dépensé", "L1_LIGHTWEIGHT", "EXPENSE_ANALYSIS"),
            ("voir mes comptes", "L1_LIGHTWEIGHT", "BALANCE_CHECK"),
        ]
        
        for message, expected_level, expected_intent in l1_tests:
            result = self.test_query(message, expected_level, expected_intent)
            self.results.append(result)
            self._print_test_result(result)
        
        # ==========================================
        # 3. TESTS L2 - REQUÊTES COMPLEXES
        # ==========================================
        print("\n🧠 TESTS L2 - REQUÊTES COMPLEXES")
        print("-" * 50)
        
        l2_tests = [
            ("Analyse mes dépenses et donne-moi des recommandations", "L2_LLM", "EXPENSE_ANALYSIS"),
            ("Comment optimiser mon budget mensuel", "L2_LLM", "BUDGET_PLANNING"),
            ("Quelle stratégie d'épargne me conseilles-tu", "L2_LLM", "SAVINGS_GOAL"),
        ]
        
        for message, expected_level, expected_intent in l2_tests:
            result = self.test_query(message, expected_level, expected_intent)
            self.results.append(result)
            self._print_test_result(result)
        
        # ==========================================
        # 4. TESTS EDGE CASES
        # ==========================================
        print("\n🔄 TESTS EDGE CASES")
        print("-" * 50)
        
        edge_cases = [
            ("", None, "UNKNOWN"),
            ("a", None, "UNKNOWN"),
            ("🏦💰📊", None, "UNKNOWN"),
            ("xyz abc 123", None, "UNKNOWN"),
            ("Hello how are you?", None, "GREETING"),
        ]
        
        for message, expected_level, expected_intent in edge_cases:
            result = self.test_query(message, expected_level, expected_intent)
            self.results.append(result)
            self._print_test_result(result)
        
        # ==========================================
        # 5. ANALYSE GLOBALE
        # ==========================================
        return self._generate_comprehensive_report()
    
    def _print_test_result(self, result: TestResult):
        """Affichage formaté résultat test"""
        
        level_emoji = {
            "L0_PATTERN": "⚡",
            "L1_LIGHTWEIGHT": "🔥", 
            "L2_LLM": "🧠",
            "ERROR_FALLBACK": "🔄",
            "ERROR_HTTP": "❌",
            "ERROR_EXCEPTION": "💥"
        }
        
        emoji = level_emoji.get(result.level_used, "❓")
        
        # Couleur selon succès level attendu
        level_match = "✅" if result.expected_level and result.level_used == result.expected_level else "❌" if result.expected_level else "➖"
        intent_match = "✅" if result.expected_intent and result.intent == result.expected_intent else "❌" if result.expected_intent else "➖"
        
        print(f"{emoji} {result.message[:35]:<35} → {result.intent:<15} "
              f"({result.confidence:.2f}) [{result.level_used}] "
              f"{result.processing_time_ms:.1f}ms {level_match}{intent_match}")
        
        # Diagnostic fallback
        if result.fallback_reason:
            print(f"    🔍 Fallback: {result.fallback_reason}")
        
        if result.diagnostic and result.diagnostic.get("potential_issues"):
            for issue in result.diagnostic["potential_issues"]:
                print(f"    ⚠️  {issue}")
    
    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Génération rapport complet avec diagnostics"""
        
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        
        # Calculs de base
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        avg_latency = sum(r.processing_time_ms for r in self.results if r.success) / max(successful_tests, 1)
        
        # Distribution niveaux
        level_percentages = {
            level: (count / total_tests) * 100 
            for level, count in self.level_distribution.items()
        }
        
        # Analyse fallbacks
        fallback_rate = (self.level_distribution.get("ERROR_FALLBACK", 0) / total_tests) * 100
        
        # Analyse précision intentions
        total_with_expected = sum(1 for r in self.results if r.expected_intent)
        intent_accuracy_rate = (self.intent_accuracy["correct"] / max(total_with_expected, 1)) * 100
        
        print("\n" + "="*80)
        print("📊 RAPPORT DIAGNOSTIC COMPLET")
        print("="*80)
        
        # Métriques globales
        print(f"\n📈 MÉTRIQUES GLOBALES:")
        print(f"   Tests totaux: {total_tests}")
        print(f"   Tests réussis: {successful_tests}")
        print(f"   Taux de succès: {success_rate:.1%}")
        print(f"   Latence moyenne: {avg_latency:.2f}ms")
        
        # Distribution niveaux
        print(f"\n🎯 DISTRIBUTION NIVEAUX:")
        for level, percentage in sorted(level_percentages.items()):
            count = self.level_distribution[level]
            target = self._get_level_target(level)
            status = "✅" if target and percentage >= target else "❌" if target else "➖"
            print(f"   {level}: {count} ({percentage:.1f}%) {status}")
            if target:
                print(f"      Target: {target}%")
        
        # Analyse fallbacks
        print(f"\n🔄 ANALYSE FALLBACKS ({fallback_rate:.1f}%):")
        if self.fallback_reasons:
            for reason, count in self.fallback_reasons.most_common():
                percentage = (count / total_tests) * 100
                print(f"   {reason}: {count} ({percentage:.1f}%)")
        else:
            print("   Aucun fallback détecté")
        
        # Précision intentions
        print(f"\n🎯 PRÉCISION INTENTIONS:")
        print(f"   Intentions correctes: {self.intent_accuracy['correct']}")
        print(f"   Intentions incorrectes: {self.intent_accuracy['incorrect']}")
        print(f"   Intentions inconnues: {self.intent_accuracy['unknown']}")
        print(f"   Taux de précision: {intent_accuracy_rate:.1%}")
        
        # Problèmes identifiés
        print(f"\n⚠️  PROBLÈMES IDENTIFIÉS:")
        issues = self._identify_main_issues()
        for issue in issues:
            print(f"   • {issue}")
        
        # Recommandations
        print(f"\n💡 RECOMMANDATIONS:")
        recommendations = self._generate_recommendations()
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        # Détails diagnostiques
        print(f"\n🔍 DÉTAILS TECHNIQUES:")
        problematic_queries = [r for r in self.results if r.fallback_reason]
        if problematic_queries:
            print(f"   Requêtes problématiques: {len(problematic_queries)}")
            for result in problematic_queries[:5]:  # Top 5
                print(f"      '{result.message}' → {result.fallback_reason}")
        
        return {
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": success_rate,
                "avg_latency_ms": avg_latency,
                "fallback_rate": fallback_rate,
                "intent_accuracy_rate": intent_accuracy_rate
            },
            "level_distribution": dict(self.level_distribution),
            "level_percentages": level_percentages,
            "fallback_analysis": dict(self.fallback_reasons),
            "intent_accuracy": dict(self.intent_accuracy),
            "issues_identified": issues,
            "recommendations": recommendations,
            "detailed_results": [
                {
                    "message": r.message,
                    "intent": r.intent,
                    "level_used": r.level_used,
                    "expected_level": r.expected_level,
                    "fallback_reason": r.fallback_reason,
                    "confidence": r.confidence,
                    "processing_time_ms": r.processing_time_ms
                }
                for r in self.results
            ]
        }
    
    def _get_level_target(self, level: str) -> Optional[float]:
        """Cibles en pourcentage pour chaque niveau"""
        targets = {
            "L0_PATTERN": 85.0,
            "L1_LIGHTWEIGHT": 12.0,
            "L2_LLM": 3.0,
            "ERROR_FALLBACK": 0.0
        }
        return targets.get(level)
    
    def _identify_main_issues(self) -> List[str]:
        """Identification problèmes principaux"""
        issues = []
        
        total_tests = len(self.results)
        fallback_rate = (self.level_distribution.get("ERROR_FALLBACK", 0) / total_tests) * 100
        
        # Problème fallback majeur
        if fallback_rate > 50:
            issues.append(f"Taux de fallback critique: {fallback_rate:.1f}% (>50%)")
        elif fallback_rate > 20:
            issues.append(f"Taux de fallback élevé: {fallback_rate:.1f%} (>20%)")
        
        # Problème L0 patterns
        l0_rate = (self.level_distribution.get("L0_PATTERN", 0) / total_tests) * 100
        if l0_rate < 50:
            issues.append(f"Usage L0 trop bas: {l0_rate:.1f}% (target: 85%)")
        
        # Problème précision intentions
        intent_accuracy_rate = (self.intent_accuracy["correct"] / max(sum(self.intent_accuracy.values()), 1)) * 100
        if intent_accuracy_rate < 80:
            issues.append(f"Précision intentions faible: {intent_accuracy_rate:.1f}% (<80%)")
        
        # Analyse fallbacks spécifiques
        top_fallback = self.fallback_reasons.most_common(1)
        if top_fallback and top_fallback[0][1] > 5:
            issues.append(f"Fallback récurrent: {top_fallback[0][0]} ({top_fallback[0][1]} occurrences)")
        
        return issues
    
    def _generate_recommendations(self) -> List[str]:
        """Génération recommandations basées sur l'analyse"""
        recommendations = []
        
        fallback_rate = (self.level_distribution.get("ERROR_FALLBACK", 0) / len(self.results)) * 100
        
        # Recommandations fallback
        if fallback_rate > 80:
            recommendations.append("Vérifier santé Intent Detection Engine - la plupart des requêtes échouent")
            recommendations.append("Contrôler initialisation composants L0/L1/L2")
        
        # Recommandations L0
        if "l0_pattern_not_found" in self.fallback_reasons:
            recommendations.append("Enrichir patterns L0 avec mots-clés financiers manquants")
        
        if "l0_single_word_not_matched" in self.fallback_reasons:
            recommendations.append("Ajouter patterns L0 pour mots-clés simples (solde, compte, etc.)")
        
        # Recommandations L1
        if "l1_confidence_too_low" in self.fallback_reasons:
            recommendations.append("Réduire seuil confiance L1 ou améliorer modèle TinyBERT")
        
        # Recommandations générales
        if self.intent_accuracy["unknown"] > 5:
            recommendations.append("Améliorer classification intentions - trop de UNKNOWN")
        
        # Recommandations performance
        avg_latency = sum(r.processing_time_ms for r in self.results if r.success) / max(len([r for r in self.results if r.success]), 1)
        if avg_latency > 100:
            recommendations.append(f"Optimiser performance - latence moyenne {avg_latency:.1f}ms > 100ms")
        
        if not recommendations:
            recommendations.append("Architecture fonctionne correctement - optimisations mineures possibles")
        
        return recommendations

def test_specific_components():
    """Tests spécifiques des composants"""
    print("\n🔧 TESTS COMPOSANTS SPÉCIFIQUES")
    print("="*80)
    
    # Test santé service
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Health check: {health.get('status', 'unknown')}")
            
            if 'performance' in health:
                perf = health['performance']
                print(f"   Total requests: {perf.get('total_requests', 0)}")
                print(f"   Success rate: {perf.get('success_rate', 0):.1%}")
        else:
            print(f"❌ Health check failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Test métriques
    try:
        response = requests.get(METRICS_URL, timeout=5)
        if response.status_code == 200:
            metrics = response.json()
            print(f"✅ Métriques disponibles")
            
            if 'performance' in metrics:
                perf = metrics['performance']
                print(f"   Status: {perf.get('status', 'unknown')}")
                print(f"   Avg latency: {perf.get('avg_latency_ms', 0):.1f}ms")
        else:
            print(f"❌ Métriques failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Métriques error: {e}")

def test_performance_stress():
    """Test de charge avec diagnostic"""
    print("\n🔥 TEST DE CHARGE")
    print("-" * 50)
    
    test_message = "mes dépenses restaurant"
    num_requests = 20
    results = []
    
    for i in range(num_requests):
        start = time.time()
        try:
            response = requests.post(CHAT_URL, json={
                "message": test_message,
                "user_id": 34,
                "conversation_id": f"stress_{i}"
            }, timeout=5)
            
            latency = (time.time() - start) * 1000
            
            if response.status_code == 200:
                data = response.json()
                results.append({
                    "latency_ms": latency,
                    "processing_time_ms": data.get("processing_metadata", {}).get("processing_time_ms", 0),
                    "level": data.get("processing_metadata", {}).get("level_used", "UNKNOWN"),
                    "intent": data.get("intent", "UNKNOWN"),
                    "confidence": data.get("confidence", 0.0),
                    "success": True
                })
            else:
                results.append({
                    "latency_ms": latency,
                    "success": False,
                    "error": f"HTTP {response.status_code}"
                })
        except Exception as e:
            results.append({
                "latency_ms": (time.time() - start) * 1000,
                "success": False,
                "error": str(e)
            })
    
    # Analyse résultats
    successful = [r for r in results if r.get("success", False)]
    failed = [r for r in results if not r.get("success", False)]
    
    if successful:
        latencies = [r["latency_ms"] for r in successful]
        processing_times = [r["processing_time_ms"] for r in successful]
        levels = [r["level"] for r in successful]
        
        print(f"📊 {len(successful)}/{num_requests} requêtes réussies:")
        print(f"   Latence réseau: {sum(latencies)/len(latencies):.2f}ms (min: {min(latencies):.2f}, max: {max(latencies):.2f})")
        print(f"   Traitement interne: {sum(processing_times)/len(processing_times):.2f}ms")
        
        # Distribution niveaux
        level_counts = Counter(levels)
        for level, count in level_counts.items():
            percentage = (count / len(successful)) * 100
            print(f"   {level}: {count}/{len(successful)} ({percentage:.1f}%)")
        
        # Cohérence intentions
        intents = [r["intent"] for r in successful]
        intent_counts = Counter(intents)
        print(f"   Intentions: {dict(intent_counts)}")
    
    if failed:
        print(f"❌ {len(failed)} requêtes échouées:")
        error_counts = Counter(r.get("error", "unknown") for r in failed)
        for error, count in error_counts.items():
            print(f"   {error}: {count}")

def main():
    """Fonction principale test diagnostic"""
    print("🚀 DIAGNOSTIC COMPLET CONVERSATION SERVICE")
    print("="*80)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target URL: {CHAT_URL}")
    
    # Tests composants de base
    test_specific_components()
    
    # Suite de tests principale
    tester = ConversationServiceTester()
    report = tester.run_comprehensive_test_suite()
    
    # Test de charge
    test_performance_stress()
    
    # Conclusion
    print("\n" + "="*80)
    print("🎯 CONCLUSION ET ACTIONS")
    print("="*80)
    
    fallback_rate = report["summary"]["fallback_rate"]
    intent_accuracy = report["summary"]["intent_accuracy_rate"]
    
    if fallback_rate > 80:
        print("🔴 STATUT: CRITIQUE - Service en mode fallback constant")
        print("   Action requise: Vérifier initialisation Intent Detection Engine")
    elif fallback_rate > 50:
        print("🟡 STATUT: DÉGRADÉ - Taux de fallback élevé")
        print("   Action recommandée: Diagnostiquer composants L0/L1")
    elif intent_accuracy > 85:
        print("🟢 STATUT: BON - Intentions détectées correctement malgré fallbacks")
        print("   Action suggérée: Optimiser pour réduire fallbacks")
    else:
        print("🟡 STATUT: ACCEPTABLE - Amélioration possible")
    
    print(f"\nRésumé: {fallback_rate:.1f}% fallbacks, {intent_accuracy:.1f}% précision intentions")
    
    return report

if __name__ == "__main__":
    report = main()