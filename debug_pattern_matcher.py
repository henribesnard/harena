"""
🔧 Script de debug Pattern Matcher L0 Phase 1

Diagnostique pourquoi les patterns simples comme "solde" retournent SYSTEM_ERROR
au lieu de matcher correctement.
"""

import asyncio
import aiohttp
import json
import sys
from typing import Dict, Any

async def debug_pattern_matcher():
    """Debug approfondi du Pattern Matcher L0"""
    
    print("🔧 DEBUG PATTERN MATCHER L0 - Phase 1")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    conversation_url = f"{base_url}/api/v1/conversation"
    
    async with aiohttp.ClientSession() as session:
        
        # ===== 1. VÉRIFICATION HEALTH DÉTAILLÉE =====
        print("\n1️⃣ Health Check Détaillé...")
        try:
            async with session.get(f"{conversation_url}/health", timeout=5) as response:
                if response.status == 200:
                    health_data = await response.json()
                    print(f"✅ Health Response:")
                    print(json.dumps(health_data, indent=2))
                else:
                    print(f"❌ Health failed: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Health error: {e}")
            return False
        
        # ===== 2. TEST PATTERN SIMPLE AVEC DEBUG =====
        print(f"\n2️⃣ Test Pattern Simple 'solde' avec Debug...")
        try:
            payload = {
                "message": "solde",
                "user_id": 999,
                "debug_mode": True,  # Mode debug activé
                "enable_cache": False  # Cache désactivé pour debug
            }
            
            async with session.post(
                f"{conversation_url}/chat",
                json=payload,
                timeout=10
            ) as response:
                
                print(f"📡 Response Status: {response.status}")
                
                if response.status == 200:
                    chat_data = await response.json()
                    print(f"✅ Chat Response:")
                    print(json.dumps(chat_data, indent=2))
                    
                    # Analyse détaillée de la réponse
                    print(f"\n🔍 Analyse Réponse:")
                    print(f"   - Success: {chat_data.get('success', 'unknown')}")
                    print(f"   - Intent: {chat_data.get('intent', 'unknown')}")
                    print(f"   - Confidence: {chat_data.get('confidence', 0)}")
                    print(f"   - Error: {chat_data.get('error', 'none')}")
                    
                    # Métadonnées de processing
                    metadata = chat_data.get('processing_metadata', {})
                    print(f"   - Level Used: {metadata.get('level_used', 'unknown')}")
                    print(f"   - Processing Time: {metadata.get('processing_time_ms', 0)}ms")
                    print(f"   - Pattern Matched: {metadata.get('pattern_matched', 'none')}")
                    print(f"   - Error Detail: {metadata.get('error', 'none')}")
                    print(f"   - Fallback Reason: {metadata.get('fallback_reason', 'none')}")
                    
                else:
                    error_text = await response.text()
                    print(f"❌ Chat Error Response:")
                    print(error_text)
                    
        except Exception as e:
            print(f"❌ Chat test error: {e}")
            import traceback
            traceback.print_exc()
        
        # ===== 3. TEST DEBUG ENDPOINT =====
        print(f"\n3️⃣ Test Debug Endpoint...")
        try:
            debug_payload = {
                "message": "solde",
                "user_id": 999,
                "debug_mode": True
            }
            
            async with session.post(
                f"{conversation_url}/debug/test-patterns",
                json=debug_payload,
                timeout=10
            ) as response:
                
                print(f"📡 Debug Response Status: {response.status}")
                
                if response.status == 200:
                    debug_data = await response.json()
                    print(f"✅ Debug Response:")
                    print(json.dumps(debug_data, indent=2))
                else:
                    error_text = await response.text()
                    print(f"❌ Debug Error:")
                    print(error_text)
                    
        except Exception as e:
            print(f"❌ Debug test error: {e}")
        
        # ===== 4. INFO PATTERNS =====
        print(f"\n4️⃣ Info Patterns...")
        try:
            async with session.get(f"{conversation_url}/debug/patterns-info", timeout=10) as response:
                if response.status == 200:
                    patterns_data = await response.json()
                    
                    summary = patterns_data.get('summary', {})
                    print(f"✅ Patterns Summary:")
                    print(f"   - Total patterns: {summary.get('total_patterns', 0)}")
                    print(f"   - By intent: {summary.get('by_intent_count', {})}")
                    
                    # Patterns par intention
                    by_intent = patterns_data.get('patterns_by_intent', {})
                    for intent, patterns in by_intent.items():
                        print(f"\n   📋 {intent}:")
                        for pattern in patterns[:3]:  # Premiers 3 seulement
                            print(f"      - {pattern.get('name', 'unknown')}: {pattern.get('confidence', 0):.2f}")
                    
                else:
                    print(f"❌ Patterns info failed: {response.status}")
                    
        except Exception as e:
            print(f"❌ Patterns info error: {e}")
        
        # ===== 5. MÉTRIQUES DÉTAILLÉES =====
        print(f"\n5️⃣ Métriques Détaillées...")
        try:
            async with session.get(f"{conversation_url}/metrics", timeout=5) as response:
                if response.status == 200:
                    metrics_data = await response.json()
                    
                    # L0 Performance
                    l0_perf = metrics_data.get('l0_performance', {})
                    print(f"✅ L0 Performance:")
                    for key, value in l0_perf.items():
                        print(f"   - {key}: {value}")
                    
                    # Patterns usage
                    pattern_analysis = metrics_data.get('pattern_analysis', {})
                    print(f"\n📊 Pattern Analysis:")
                    print(f"   - Patterns available: {pattern_analysis.get('total_patterns_available', 0)}")
                    print(f"   - Patterns used: {pattern_analysis.get('patterns_used', 0)}")
                    print(f"   - Patterns unused: {pattern_analysis.get('patterns_unused', 0)}")
                    
                    # Top patterns
                    top_patterns = pattern_analysis.get('top_patterns', [])
                    if top_patterns:
                        print(f"   - Top patterns: {top_patterns[:5]}")
                    
                    # Recommendations
                    recommendations = metrics_data.get('recommendations', [])
                    if recommendations:
                        print(f"\n🔧 Recommandations:")
                        for rec in recommendations:
                            print(f"   - {rec}")
                    
                else:
                    print(f"❌ Metrics failed: {response.status}")
                    
        except Exception as e:
            print(f"❌ Metrics error: {e}")
        
        # ===== 6. TEST AUTRES PATTERNS =====
        print(f"\n6️⃣ Test Autres Patterns de Base...")
        
        test_patterns = ["virement", "bonjour", "aide", "dépenses", "carte"]
        
        for pattern_text in test_patterns:
            try:
                payload = {
                    "message": pattern_text,
                    "user_id": 888,
                    "debug_mode": False
                }
                
                async with session.post(
                    f"{conversation_url}/chat",
                    json=payload,
                    timeout=5
                ) as response:
                    
                    if response.status == 200:
                        chat_data = await response.json()
                        success = chat_data.get('success', False)
                        intent = chat_data.get('intent', 'UNKNOWN')
                        level = chat_data.get('processing_metadata', {}).get('level_used', 'unknown')
                        error = chat_data.get('error', 'none')
                        
                        status = "✅" if success else "❌"
                        print(f"   {status} '{pattern_text}' → {intent} ({level}) {error if not success else ''}")
                    else:
                        print(f"   ❌ '{pattern_text}' → HTTP {response.status}")
                        
            except Exception as e:
                print(f"   ❌ '{pattern_text}' → Exception: {e}")
        
        # ===== 7. DIAGNOSTIC FINAL =====
        print(f"\n🎯 DIAGNOSTIC FINAL")
        print("=" * 60)
        
        print(f"🔍 Problèmes identifiés:")
        print(f"   1. Les patterns simples retournent SYSTEM_ERROR")
        print(f"   2. Cela indique des exceptions dans le Pattern Matcher")
        print(f"   3. Possible problème d'initialisation ou de configuration")
        
        print(f"\n🔧 Actions recommandées:")
        print(f"   1. Vérifier les logs du serveur pour les stack traces")
        print(f"   2. Vérifier l'initialisation du Pattern Matcher")
        print(f"   3. Vérifier que les patterns sont correctement chargés")
        print(f"   4. Tester directement le Pattern Matcher sans HTTP")
        
        print(f"\n📋 Commandes debug:")
        print(f"   - Logs serveur: Vérifier la console du serveur local_app.py")
        print(f"   - Health check: curl {conversation_url}/health")
        print(f"   - Patterns info: curl {conversation_url}/debug/patterns-info")
        print(f"   - Métriques: curl {conversation_url}/metrics")

async def main():
    """Point d'entrée principal"""
    try:
        await debug_pattern_matcher()
        print(f"\n✅ Debug terminé - Analyser les logs du serveur")
        return 0
    except KeyboardInterrupt:
        print(f"\n🛑 Debug interrompu")
        return 130
    except Exception as e:
        print(f"\n💥 Erreur debug: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)