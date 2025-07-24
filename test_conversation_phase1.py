
"""
💡 FICHIER SÉPARÉ: test_conversation_phase1.py
Script pour valider la Phase 1 L0 Pattern Matching
"""

import asyncio
import aiohttp
import json
import time

async def test_conversation_phase1_complete():
    """
    🧪 Test complet Phase 1 - L0 Pattern Matching
    
    Vérifie:
    1. Service Phase 1 démarré correctement
    2. Pattern Matcher L0 fonctionnel
    3. Performance <10ms sur patterns simples
    4. Taux succès >85% sur requêtes financières
    """
    
    print("🧪 Test Validation Phase 1 - L0 Pattern Matching")
    print("=" * 70)
    
    base_url = "http://localhost:8001"
    results = {"tests": [], "performance": [], "errors": [], "l0_metrics": {}}
    
    async with aiohttp.ClientSession() as session:
        
        # ===== TEST 1: Health Check Phase 1 =====
        print("\n1️⃣ Test Health Check Phase 1...")
        try:
            async with session.get(f"{base_url}/health", timeout=5) as response:
                if response.status == 200:
                    health_data = await response.json()
                    print(f"   ✅ Health: {health_data['status']}")
                    print(f"   🔧 Phase: {health_data.get('phase', 'unknown')}")
                    print(f"   ⚡ Pattern Matcher: {health_data.get('pattern_matcher', {}).get('status', 'unknown')}")
                    
                    # Vérification targets
                    targets = health_data.get('targets_status', {})
                    print(f"   🎯 Latence target: {targets.get('latency_met', False)}")
                    print(f"   🎯 Success target: {targets.get('success_met', False)}")
                    
                    results["tests"].append({"test": "health_phase1", "status": "pass"})
                else:
                    print(f"   ❌ Health failed: {response.status}")
                    results["tests"].append({"test": "health_phase1", "status": "fail", "code": response.status})
                    return results
        except Exception as e:
            print(f"   ❌ Health error: {e}")
            results["errors"].append({"test": "health_phase1", "error": str(e)})
            return results
        
        # ===== TEST 2: Patterns L0 Basiques =====
        print("\n2️⃣ Test Patterns L0 Basiques...")
        
        l0_test_queries = [
            # Patterns haute confiance
            ("solde", "BALANCE_CHECK", True),
            ("virement", "TRANSFER", True),
            ("bloquer carte", "CARD_MANAGEMENT", True),
            ("dépenses", "EXPENSE_ANALYSIS", True),
            ("bonjour", "GREETING", True),
            
            # Patterns avec entités
            ("virer 100€", "TRANSFER", True),
            ("dépenses restaurant", "EXPENSE_ANALYSIS", True),
            ("quel est mon solde", "BALANCE_CHECK", True),
            
            # Cas qui ne devraient pas matcher L0
            ("météo aujourd'hui", None, False),
            ("recette de cuisine", None, False)
        ]
        
        l0_success = 0
        l0_latencies = []
        
        for i, (message, expected_intent, should_match) in enumerate(l0_test_queries, 1):
            try:
                start_time = time.time()
                
                payload = {"message": message, "user_id": 100 + i}
                async with session.post(
                    f"{base_url}/api/v1/chat",
                    json=payload,
                    timeout=1  # Timeout court pour L0
                ) as response:
                    
                    request_latency = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        chat_data = await response.json()
                        
                        if should_match and chat_data["success"]:
                            # Devrait matcher et a matché
                            processing_latency = chat_data["processing_metadata"]["latency_ms"]
                            level_used = chat_data["processing_metadata"]["level_used"]
                            pattern_used = chat_data["processing_metadata"].get("pattern_matched", "unknown")
                            
                            print(f"   {i:2d}. ✅ '{message}' → {chat_data['intent']} ({pattern_used})")
                            print(f"        ⏱️ L0: {processing_latency:.1f}ms, Total: {request_latency:.1f}ms")
                            
                            l0_success += 1
                            l0_latencies.append(processing_latency)
                            
                            # Vérification intention attendue
                            if expected_intent and chat_data['intent'] != expected_intent:
                                print(f"        ⚠️ Intention différente: attendu {expected_intent}")
                            
                            # Alerte performance
                            if processing_latency > 10:
                                print(f"        ⚠️ LATENCE L0 ÉLEVÉE: {processing_latency:.1f}ms > 10ms")
                            
                        elif not should_match and not chat_data["success"]:
                            # Ne devrait pas matcher et n'a pas matché
                            print(f"   {i:2d}. ✅ '{message}' → Correctement non matché")
                            l0_success += 1
                            
                        else:
                            # Résultat inattendu
                            print(f"   {i:2d}. ❌ '{message}' → Résultat inattendu")
                            print(f"        Attendu match: {should_match}, Résultat: {chat_data['success']}")
                            
                    else:
                        print(f"   {i:2d}. ❌ '{message}' → HTTP {response.status}")
                        
            except Exception as e:
                print(f"   {i:2d}. ❌ '{message}' → Exception: {e}")
        
        # ===== ANALYSE PERFORMANCE L0 =====
        print(f"\n📊 Analyse Performance L0:")
        print(f"   - Tests réussis: {l0_success}/{len(l0_test_queries)}")
        print(f"   - Taux succès: {(l0_success/len(l0_test_queries)*100):.1f}%")
        
        if l0_latencies:
            avg_l0_latency = sum(l0_latencies) / len(l0_latencies)
            max_l0_latency = max(l0_latencies)
            min_l0_latency = min(l0_latencies)
            
            print(f"   - Latence L0 moyenne: {avg_l0_latency:.1f}ms")
            print(f"   - Latence L0 min/max: {min_l0_latency:.1f}/{max_l0_latency:.1f}ms")
            
            # Évaluation targets Phase 1
            latency_target_met = avg_l0_latency < 10.0
            success_target_met = (l0_success/len(l0_test_queries)) >= 0.85
            
            print(f"   🎯 Target latence (<10ms): {'✅' if latency_target_met else '❌'}")
            print(f"   🎯 Target succès (>85%): {'✅' if success_target_met else '❌'}")
            
            results["l0_metrics"] = {
                "success_rate": l0_success/len(l0_test_queries),
                "avg_latency_ms": avg_l0_latency,
                "latency_target_met": latency_target_met,
                "success_target_met": success_target_met
            }
        
        # ===== TEST 3: Métriques Détaillées =====
        print(f"\n3️⃣ Test Métriques L0...")
        try:
            async with session.get(f"{base_url}/api/v1/metrics", timeout=5) as response:
                if response.status == 200:
                    metrics_data = await response.json()
                    l0_perf = metrics_data.get("l0_performance", {})
                    targets = metrics_data.get("targets_validation", {})
                    
                    print(f"   ✅ Métriques L0 disponibles")
                    print(f"   📊 Requêtes totales: {l0_perf.get('total_requests', 0)}")
                    print(f"   📊 Taux succès L0: {l0_perf.get('success_rate', 0):.1%}")
                    print(f"   📊 Latence L0: {l0_perf.get('avg_latency_ms', 0):.1f}ms")
                    print(f"   🎯 Targets: {sum(targets.get(k+'_met', False) for k in ['latency', 'success_rate', 'usage'])}/3")
                    
                    results["tests"].append({"test": "metrics_l0", "status": "pass"})
                else:
                    print(f"   ⚠️ Métriques unavailable: {response.status}")
                    results["tests"].append({"test": "metrics_l0", "status": "degraded"})
        except Exception as e:
            print(f"   ⚠️ Métriques error: {e}")
            results["errors"].append({"test": "metrics_l0", "error": str(e)})
        
        # ===== TEST 4: Debug Patterns =====
        print(f"\n4️⃣ Test Debug Patterns...")
        try:
            debug_payload = {"message": "solde compte courant", "user_id": 999}
            async with session.post(
                f"{base_url}/api/v1/debug/test-patterns",
                json=debug_payload,
                timeout=5
            ) as response:
                if response.status == 200:
                    debug_data = await response.json()
                    
                    if debug_data.get("best_match"):
                        match_info = debug_data["best_match"]
                        print(f"   ✅ Debug patterns fonctionnel")
                        print(f"   🎯 Pattern: {match_info['pattern_name']}")
                        print(f"   📊 Confiance: {match_info['confidence']:.2f}")
                        print(f"   ⏱️ Latence: {debug_data['performance']['latency_ms']:.1f}ms")
                        
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
    
    # ===== RÉSUMÉ FINAL PHASE 1 =====
    print(f"\n🎯 RÉSUMÉ FINAL PHASE 1")
    print("=" * 70)
    
    total_tests = len(results["tests"])
    passed_tests = len([t for t in results["tests"] if t["status"] == "pass"])
    
    print(f"Tests réussis: {passed_tests}/{total_tests}")
    print(f"Erreurs: {len(results['errors'])}")
    
    # Status global Phase 1
    l0_metrics = results.get("l0_metrics", {})
    phase1_success = (
        passed_tests >= total_tests * 0.75 and  # 75% tests passed
        l0_metrics.get("latency_target_met", False) and  # <10ms
        l0_metrics.get("success_target_met", False)  # >85% success
    )
    
    if phase1_success:
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
    
    # Performance summary
    if l0_metrics:
        print(f"\n📈 Performance L0:")
        print(f"   - Succès: {l0_metrics['success_rate']:.1%}")
        print(f"   - Latence: {l0_metrics['avg_latency_ms']:.1f}ms")
        print(f"   - Targets: {'✅' if l0_metrics.get('latency_target_met') and l0_metrics.get('success_target_met') else '❌'}")
    
    return results

# ===== FONCTION MAIN =====
async def main():
    """Point d'entrée principal du test Phase 1"""
    try:
        results = await test_conversation_phase1_complete()
        
        # Sauvegarde résultats
        with open("test_results_phase1.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Résultats sauvegardés: test_results_phase1.json")
        
        # Code de sortie
        if results.get("overall_status") == "phase1_complete":
            exit(0)
        elif results.get("overall_status") == "phase1_partial":
            exit(1)
        else:
            exit(2)
            
    except KeyboardInterrupt:
        print("\n🛑 Test interrompu")
        exit(130)
    except Exception as e:
        print(f"\n💥 Erreur test: {e}")
        exit(3)

if __name__ == "__main__":
    print("🚀 Lancement test Phase 1...")
    asyncio.run(main())