"""
💡 FICHIER SÉPARÉ: test_conversation_phase1.py
Script pour valider la Phase 1 L0 Pattern Matching

✅ CORRECTIONS APPLIQUÉES:
- URLs corrigées: http://localhost:8000 (local_app.py port)
- Endpoints corrigés: /api/v1/conversation/* (prefix conversation)
- Tests adaptés aux vrais modèles Pydantic
- Validation cohérente avec routes.py et main.py
"""

import asyncio
import aiohttp
import json
import time
import sys
from typing import Dict, Any, List

async def test_conversation_phase1_complete():
    """
    🧪 Test complet Phase 1 - L0 Pattern Matching
    
    Vérifie:
    1. Service Phase 1 démarré correctement
    2. Pattern Matcher L0 fonctionnel 
    3. Performance <10ms sur patterns simples
    4. Taux succès >85% sur requêtes financières
    5. Cohérence avec architecture Phase 1
    """
    
    print("🧪 Test Validation Phase 1 - L0 Pattern Matching")
    print("=" * 70)
    
    # ✅ CORRIGÉ: Port 8000 pour local_app.py, prefix /api/v1/conversation
    base_url = "http://localhost:8000"
    conversation_url = f"{base_url}/api/v1/conversation"
    
    results = {"tests": [], "performance": [], "errors": [], "l0_metrics": {}}
    
    async with aiohttp.ClientSession() as session:
        
        # ===== TEST 0: Health Check Global =====
        print("\n0️⃣ Test Health Check Global...")
        try:
            async with session.get(f"{base_url}/health", timeout=5) as response:
                if response.status == 200:
                    health_data = await response.json()
                    print(f"   ✅ Plateforme: {health_data['status']}")
                    
                    # Vérifier conversation_service dans health global
                    conv_service = health_data.get('conversation_service', {})
                    print(f"   🤖 Conversation Service: {conv_service.get('status', 'unknown')}")
                    print(f"   🔧 Phase: {conv_service.get('phase', 'unknown')}")
                    print(f"   📦 Version: {conv_service.get('version', 'unknown')}")
                    
                    if conv_service.get('status') not in ['ok', 'degraded']:
                        print(f"   ❌ Conversation Service non disponible")
                        results["errors"].append({"test": "global_health", "error": "conversation_service not available"})
                        return results
                    
                    results["tests"].append({"test": "global_health", "status": "pass"})
                else:
                    print(f"   ❌ Health global failed: {response.status}")
                    results["tests"].append({"test": "global_health", "status": "fail", "code": response.status})
                    return results
        except Exception as e:
            print(f"   ❌ Health global error: {e}")
            results["errors"].append({"test": "global_health", "error": str(e)})
            return results
        
        # ===== TEST 1: Health Check Conversation Service =====
        print("\n1️⃣ Test Health Check Conversation Service...")
        try:
            async with session.get(f"{conversation_url}/health", timeout=5) as response:
                if response.status == 200:
                    health_data = await response.json()
                    print(f"   ✅ Health: {health_data['status']}")
                    print(f"   🔧 Phase: {health_data.get('phase', 'unknown')}")
                    print(f"   ⚡ Latence Health: {health_data.get('latency_ms', 0):.1f}ms")
                    
                    # Vérifier Pattern Matcher
                    pattern_matcher = health_data.get('pattern_matcher', {})
                    print(f"   🎯 Pattern Matcher: {pattern_matcher.get('status', 'unknown')}")
                    print(f"   📊 Patterns: {pattern_matcher.get('patterns_loaded', 0)}")
                    
                    # Vérifier L0 Performance dans health
                    l0_perf = health_data.get('l0_performance', {})
                    if l0_perf:
                        print(f"   📈 L0 Success Rate: {l0_perf.get('success_rate', 0):.1%}")
                        print(f"   ⏱️ L0 Latence: {l0_perf.get('avg_latency_ms', 0):.1f}ms")
                    
                    # Vérifier targets
                    targets = health_data.get('targets_status', {})
                    if targets:
                        print(f"   🎯 Latence target: {'✅' if targets.get('latency_met', False) else '❌'}")
                        print(f"   🎯 Success target: {'✅' if targets.get('success_met', False) else '❌'}")
                    
                    results["tests"].append({"test": "conversation_health", "status": "pass"})
                else:
                    print(f"   ❌ Health conversation failed: {response.status}")
                    results["tests"].append({"test": "conversation_health", "status": "fail", "code": response.status})
                    return results
        except Exception as e:
            print(f"   ❌ Health conversation error: {e}")
            results["errors"].append({"test": "conversation_health", "error": str(e)})
            return results
        
        # ===== TEST 2: Patterns L0 Basiques =====
        print("\n2️⃣ Test Patterns L0 Basiques...")
        
        l0_test_queries = [
            # ✅ Patterns haute confiance Phase 1
            ("solde", "BALANCE_CHECK", True),
            ("virement", "TRANSFER", True), 
            ("bloquer carte", "CARD_MANAGEMENT", True),
            ("dépenses", "EXPENSE_ANALYSIS", True),
            ("bonjour", "GREETING", True),
            ("aide", "HELP", True),
            
            # ✅ Patterns avec entités
            ("virer 100€", "TRANSFER", True),
            ("dépenses restaurant", "EXPENSE_ANALYSIS", True),
            ("quel est mon solde", "BALANCE_CHECK", True),
            ("combien j'ai", "BALANCE_CHECK", True),
            
            # ✅ Patterns système
            ("salut", "GREETING", True),
            ("au revoir", "GOODBYE", True),
            
            # ✅ Cas qui ne devraient pas matcher L0
            ("météo aujourd'hui", "UNKNOWN", False),
            ("recette de cuisine", "UNKNOWN", False),
            ("comment aller à Paris", "UNKNOWN", False)
        ]
        
        l0_success = 0
        l0_latencies = []
        l0_detailed_results = []
        
        for i, (message, expected_intent, should_match) in enumerate(l0_test_queries, 1):
            try:
                start_time = time.time()
                
                # ✅ Format ChatRequest cohérent avec conversation_models.py
                payload = {
                    "message": message, 
                    "user_id": 100 + i,
                    "conversation_id": f"test_phase1_{i}",
                    "enable_cache": True,
                    "debug_mode": False
                }
                
                async with session.post(
                    f"{conversation_url}/chat",  # ✅ CORRIGÉ: /api/v1/conversation/chat
                    json=payload,
                    timeout=2  # Timeout court pour L0
                ) as response:
                    
                    request_latency = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        chat_data = await response.json()
                        
                        # ✅ Validation format ChatResponse
                        processing_metadata = chat_data.get("processing_metadata", {})
                        processing_latency = processing_metadata.get("processing_time_ms", 0)
                        level_used = processing_metadata.get("level_used", "unknown")
                        pattern_used = processing_metadata.get("pattern_matched", "unknown")
                        
                        result_detail = {
                            "query": message,
                            "expected": expected_intent,
                            "actual": chat_data.get('intent', 'UNKNOWN'),
                            "success": chat_data.get("success", False),
                            "confidence": chat_data.get("confidence", 0),
                            "latency_ms": processing_latency,
                            "level_used": level_used,
                            "pattern_used": pattern_used
                        }
                        
                        if should_match and chat_data["success"] and level_used == "L0_PATTERN":
                            # ✅ Devrait matcher L0 et a matché L0
                            print(f"   {i:2d}. ✅ '{message}' → {chat_data['intent']} ({pattern_used})")
                            print(f"        ⏱️ L0: {processing_latency:.1f}ms, Total: {request_latency:.1f}ms, Conf: {chat_data['confidence']:.2f}")
                            
                            l0_success += 1
                            l0_latencies.append(processing_latency)
                            
                            # Vérification intention attendue
                            if expected_intent and chat_data['intent'] != expected_intent:
                                print(f"        ⚠️ Intention différente: attendu {expected_intent}")
                                result_detail["warning"] = f"Expected {expected_intent}, got {chat_data['intent']}"
                            
                            # Alerte performance
                            if processing_latency > 10:
                                print(f"        ⚠️ LATENCE L0 ÉLEVÉE: {processing_latency:.1f}ms > 10ms")
                                result_detail["performance_warning"] = True
                            
                        elif not should_match and (not chat_data["success"] or chat_data['intent'] == 'UNKNOWN'):
                            # ✅ Ne devrait pas matcher et n'a pas matché (ou UNKNOWN)
                            print(f"   {i:2d}. ✅ '{message}' → Correctement non matché ({chat_data.get('intent', 'UNKNOWN')})")
                            l0_success += 1
                            
                        else:
                            # ❌ Résultat inattendu
                            print(f"   {i:2d}. ❌ '{message}' → Résultat inattendu")
                            print(f"        Attendu match: {should_match}, Résultat: {chat_data['success']}")
                            print(f"        Intent: {chat_data.get('intent', 'UNKNOWN')}, Level: {level_used}")
                            result_detail["unexpected"] = True
                        
                        l0_detailed_results.append(result_detail)
                            
                    else:
                        print(f"   {i:2d}. ❌ '{message}' → HTTP {response.status}")
                        error_text = await response.text()
                        print(f"        Error: {error_text[:100]}")
                        
                        l0_detailed_results.append({
                            "query": message,
                            "error": f"HTTP {response.status}",
                            "error_detail": error_text[:200]
                        })
                        
            except Exception as e:
                print(f"   {i:2d}. ❌ '{message}' → Exception: {e}")
                l0_detailed_results.append({
                    "query": message,
                    "exception": str(e)
                })
        
        # ===== ANALYSE PERFORMANCE L0 =====
        print(f"\n📊 Analyse Performance L0:")
        print(f"   - Tests réussis: {l0_success}/{len(l0_test_queries)}")
        success_rate = l0_success/len(l0_test_queries)
        print(f"   - Taux succès: {success_rate*100:.1f}%")
        
        if l0_latencies:
            avg_l0_latency = sum(l0_latencies) / len(l0_latencies)
            max_l0_latency = max(l0_latencies)
            min_l0_latency = min(l0_latencies)
            
            print(f"   - Latence L0 moyenne: {avg_l0_latency:.1f}ms")
            print(f"   - Latence L0 min/max: {min_l0_latency:.1f}/{max_l0_latency:.1f}ms")
            
            # ✅ Évaluation targets Phase 1
            latency_target_met = avg_l0_latency < 10.0
            success_target_met = success_rate >= 0.85
            
            print(f"   🎯 Target latence (<10ms): {'✅' if latency_target_met else '❌'}")
            print(f"   🎯 Target succès (>85%): {'✅' if success_target_met else '❌'}")
            
            results["l0_metrics"] = {
                "success_rate": success_rate,
                "avg_latency_ms": avg_l0_latency,
                "min_latency_ms": min_l0_latency,
                "max_latency_ms": max_l0_latency,
                "latency_target_met": latency_target_met,
                "success_target_met": success_target_met,
                "detailed_results": l0_detailed_results
            }
        
        # ===== TEST 3: Métriques Détaillées =====
        print(f"\n3️⃣ Test Métriques L0...")
        try:
            async with session.get(f"{conversation_url}/metrics", timeout=5) as response:
                if response.status == 200:
                    metrics_data = await response.json()
                    
                    # ✅ Structure cohérente avec routes.py corrigé
                    l0_perf = metrics_data.get("l0_performance", {})
                    targets_validation = metrics_data.get("targets_validation", {})
                    system_info = metrics_data.get("system_info", {})
                    
                    print(f"   ✅ Métriques L0 disponibles")
                    print(f"   📊 Requêtes totales: {l0_perf.get('total_requests', 0)}")
                    print(f"   📊 Taux succès L0: {l0_perf.get('success_rate', 0):.1%}")
                    print(f"   📊 Latence L0: {l0_perf.get('avg_latency_ms', 0):.1f}ms")
                    print(f"   📊 Usage L0: {l0_perf.get('usage_percent', 0):.1f}%")
                    print(f"   📊 Cache hit rate: {l0_perf.get('cache_hit_rate', 0):.1%}")
                    
                    # Validation targets depuis métriques
                    targets_met = sum(1 for k in ['latency_target_met', 'success_rate_met', 'usage_target_met'] 
                                    if targets_validation.get(k, False))
                    print(f"   🎯 Targets met: {targets_met}/3")
                    
                    # Ready for L1?
                    ready_for_l1 = system_info.get("ready_for_l1", False)
                    print(f"   🚀 Ready for Phase 2: {'✅' if ready_for_l1 else '❌'}")
                    
                    results["tests"].append({"test": "metrics_l0", "status": "pass"})
                    results["metrics_details"] = {
                        "l0_performance": l0_perf,
                        "targets_validation": targets_validation,
                        "ready_for_l1": ready_for_l1
                    }
                else:
                    print(f"   ⚠️ Métriques unavailable: {response.status}")
                    results["tests"].append({"test": "metrics_l0", "status": "degraded"})
        except Exception as e:
            print(f"   ⚠️ Métriques error: {e}")
            results["errors"].append({"test": "metrics_l0", "error": str(e)})
        
        # ===== TEST 4: Status Service =====
        print(f"\n4️⃣ Test Status Service...")
        try:
            async with session.get(f"{conversation_url}/status", timeout=5) as response:
                if response.status == 200:
                    status_data = await response.json()
                    
                    print(f"   ✅ Status disponible")
                    print(f"   🔧 Phase: {status_data.get('phase', 'unknown')}")
                    print(f"   📦 Version: {status_data.get('version', 'unknown')}")
                    
                    # Architecture Phase 1
                    architecture = status_data.get('architecture', {})
                    if architecture:
                        print(f"   🏗️ L0 enabled: {architecture.get('l0_enabled', False)}")
                        print(f"   🏗️ L1 enabled: {architecture.get('l1_enabled', False)}")
                        print(f"   🏗️ L2 enabled: {architecture.get('l2_enabled', False)}")
                    
                    # Pattern Matcher info
                    pm_info = status_data.get('pattern_matcher', {})
                    if pm_info:
                        print(f"   🎯 Patterns loaded: {pm_info.get('loaded', 0)}")
                    
                    results["tests"].append({"test": "status_service", "status": "pass"})
                else:
                    print(f"   ⚠️ Status unavailable: {response.status}")
                    results["tests"].append({"test": "status_service", "status": "degraded"})
        except Exception as e:
            print(f"   ⚠️ Status error: {e}")
            results["errors"].append({"test": "status_service", "error": str(e)})
        
        # ===== TEST 5: Debug Patterns =====
        print(f"\n5️⃣ Test Debug Patterns...")
        try:
            debug_payload = {
                "message": "solde compte courant", 
                "user_id": 999,
                "debug_mode": True
            }
            async with session.post(
                f"{conversation_url}/debug/test-patterns",
                json=debug_payload,
                timeout=5
            ) as response:
                if response.status == 200:
                    debug_data = await response.json()
                    
                    # ✅ Validation structure debug response
                    if debug_data.get("status") == "success" and debug_data.get("pattern_match"):
                        match_info = debug_data["pattern_match"]
                        print(f"   ✅ Debug patterns fonctionnel")
                        print(f"   🎯 Pattern: {match_info.get('pattern_name', 'unknown')}")
                        print(f"   📊 Confiance: {match_info.get('confidence', 0):.2f}")
                        print(f"   ⏱️ Latence: {debug_data.get('processing_time_ms', 0):.1f}ms")
                        
                        results["tests"].append({"test": "debug_patterns", "status": "pass"})
                    else:
                        print(f"   ⚠️ Debug patterns no match")
                        results["tests"].append({"test": "debug_patterns", "status": "partial"})
                else:
                    print(f"   ❌ Debug patterns failed: {response.status}")
                    results["tests"].append({"test": "debug_patterns", "status": "fail"})
        except Exception as e:
            print(f"   ❌ Debug patterns error: {e}")
            results["errors"].append({"test": "debug_patterns", "error": str(e)})
        
        # ===== TEST 6: Validation Phase 1 =====
        print(f"\n6️⃣ Test Validation Phase 1...")
        try:
            async with session.get(f"{conversation_url}/validate-phase1", timeout=10) as response:
                if response.status == 200:
                    validation_data = await response.json()
                    
                    phase1_validation = validation_data.get("phase1_validation", {})
                    overall_status = phase1_validation.get("overall_status", "unknown")
                    phase1_success = phase1_validation.get("phase1_success", False)
                    next_action = phase1_validation.get("next_action", "unknown")
                    
                    print(f"   ✅ Validation Phase 1 disponible")
                    print(f"   🎯 Status: {overall_status}")
                    print(f"   🎉 Phase 1 Success: {'✅' if phase1_success else '❌'}")
                    print(f"   🚀 Next Action: {next_action}")
                    
                    # Détails targets
                    targets_validation = validation_data.get("targets_validation", {})
                    if targets_validation:
                        all_targets_met = targets_validation.get("all_targets_met", False)
                        print(f"   🎯 All Targets Met: {'✅' if all_targets_met else '❌'}")
                    
                    # Performance actuelle
                    current_perf = validation_data.get("current_performance", {})
                    if current_perf:
                        print(f"   📊 Current Latency: {current_perf.get('latency_ms', 0):.1f}ms")
                        print(f"   📊 Current Success: {current_perf.get('success_rate', 0):.1%}")
                    
                    results["tests"].append({"test": "validate_phase1", "status": "pass"})
                    results["phase1_validation"] = validation_data
                else:
                    print(f"   ❌ Validation Phase 1 failed: {response.status}")
                    results["tests"].append({"test": "validate_phase1", "status": "fail"})
        except Exception as e:
            print(f"   ❌ Validation Phase 1 error: {e}")
            results["errors"].append({"test": "validate_phase1", "error": str(e)})
    
    # ===== RÉSUMÉ FINAL PHASE 1 =====
    print(f"\n🎯 RÉSUMÉ FINAL PHASE 1")
    print("=" * 70)
    
    total_tests = len(results["tests"])
    passed_tests = len([t for t in results["tests"] if t["status"] == "pass"])
    
    print(f"Tests réussis: {passed_tests}/{total_tests}")
    print(f"Erreurs: {len(results['errors'])}")
    
    # ✅ Status global Phase 1 basé sur métriques réelles
    l0_metrics = results.get("l0_metrics", {})
    validation_data = results.get("phase1_validation", {})
    
    # Critères Phase 1
    tests_passed = passed_tests >= total_tests * 0.75  # 75% tests passed
    latency_ok = l0_metrics.get("latency_target_met", False)
    success_ok = l0_metrics.get("success_target_met", False)
    
    # Validation officielle si disponible
    official_validation = validation_data.get("phase1_validation", {}).get("phase1_success", None)
    
    if official_validation is True:
        print("🎉 PHASE 1 OFFICIELLEMENT VALIDÉE!")
        print("🚀 Prêt pour Phase 2 (L1 TinyBERT)")
        results["overall_status"] = "phase1_complete_official"
    elif tests_passed and latency_ok and success_ok:
        print("🎉 PHASE 1 VALIDÉE - L0 Pattern Matching opérationnel!")
        print("🚀 Prêt pour Phase 2 (L1 TinyBERT)")
        results["overall_status"] = "phase1_complete"
    elif passed_tests >= total_tests * 0.5:
        print("⚠️ PHASE 1 PARTIELLE - Optimisations nécessaires")
        print("🔧 Recommandations: Améliorer patterns, optimiser latence")
        results["overall_status"] = "phase1_partial"
    else:
        print("❌ PHASE 1 ÉCHOUÉE - Problèmes critiques")
        results["overall_status"] = "phase1_failed"
    
    # ✅ Performance summary détaillé
    if l0_metrics:
        print(f"\n📈 Performance L0:")
        print(f"   - Succès: {l0_metrics['success_rate']:.1%}")
        print(f"   - Latence moyenne: {l0_metrics['avg_latency_ms']:.1f}ms")
        print(f"   - Latence min/max: {l0_metrics['min_latency_ms']:.1f}/{l0_metrics['max_latency_ms']:.1f}ms")
        print(f"   - Targets: {'✅' if l0_metrics.get('latency_target_met') and l0_metrics.get('success_target_met') else '❌'}")
    
    # Recommandations
    if results["overall_status"] in ["phase1_partial", "phase1_failed"]:
        print(f"\n🔧 Recommandations:")
        if not latency_ok:
            print(f"   - Optimiser patterns pour réduire latence <10ms")
        if not success_ok:
            print(f"   - Ajouter patterns pour couvrir plus de cas d'usage")
        if len(results["errors"]) > 0:
            print(f"   - Corriger erreurs techniques identifiées")
    
    return results

# ===== FONCTION MAIN =====
async def main():
    """Point d'entrée principal du test Phase 1"""
    try:
        print("🚀 Lancement test Phase 1...")
        print("📋 Configuration:")
        print("   - URL: http://localhost:8000")
        print("   - Service: conversation_service")
        print("   - Phase: L0_PATTERN_MATCHING")
        print("   - Endpoints: /api/v1/conversation/*")
        print()
        
        results = await test_conversation_phase1_complete()
        
        # ✅ Sauvegarde résultats détaillés
        timestamp = int(time.time())
        results_file = f"test_results_phase1_{timestamp}.json"
        
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"\n💾 Résultats sauvegardés: {results_file}")
        
        # ✅ Code de sortie basé sur status
        overall_status = results.get("overall_status", "unknown")
        
        if overall_status in ["phase1_complete", "phase1_complete_official"]:
            print("✅ EXIT CODE 0: Phase 1 validée")
            return 0
        elif overall_status == "phase1_partial":
            print("⚠️ EXIT CODE 1: Phase 1 partielle")
            return 1
        else:
            print("❌ EXIT CODE 2: Phase 1 échouée")
            return 2
            
    except KeyboardInterrupt:
        print("\n🛑 Test interrompu par utilisateur")
        return 130
    except Exception as e:
        print(f"\n💥 Erreur critique test: {e}")
        import traceback
        traceback.print_exc()
        return 3

if __name__ == "__main__":
    # ✅ Gestion propre des codes de sortie
    exit_code = asyncio.run(main())
    sys.exit(exit_code)