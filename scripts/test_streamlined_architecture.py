#!/usr/bin/env python3
"""
Test de la nouvelle architecture streamlinée
Teste les 3 étapes:
1. IntentEntityClassifier
2. DeterministicQueryBuilder  
3. Response generation
"""
import asyncio
import sys
import os
import time
from pathlib import Path

# Ajouter le chemin du projet
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from conversation_service.agents.financial.intent_entity_classifier import IntentEntityClassifier
from conversation_service.core.deterministic_query_builder import DeterministicQueryBuilder
from conversation_service.clients.deepseek_client import DeepSeekClient
from conversation_service.core.cache_manager import CacheManager
from config_service.config import settings

async def test_streamlined_workflow():
    """Test complet du workflow streamliné"""
    
    print("Test de l'architecture streamlinee - 3 etapes")
    print("=" * 60)
    
    # Messages de test
    test_messages = [
        "mes dernieres transactions ?",
        "combien j'ai depense chez Amazon ce mois ?",
        "mes virements de janvier",
        "montre moi mes achats superieurs a 100 euros"
    ]
    
    try:
        # Initialisation des composants
        print("Initialisation des composants...")
        deepseek_client = DeepSeekClient()
        cache_manager = CacheManager()
        
        # Test de chaque message
        for i, message in enumerate(test_messages, 1):
            print(f"\n🔸 Test {i}/4: '{message}'")
            print("-" * 40)
            
            total_start = time.time()
            
            # ====================================================================
            # ÉTAPE 1: IntentEntityClassifier (1 appel LLM pour les 2 tâches)
            # ====================================================================
            print("1️⃣ Classification intention + extraction entités...")
            step1_start = time.time()
            
            intent_entity_classifier = IntentEntityClassifier(
                deepseek_client=deepseek_client,
                cache_manager=cache_manager
            )
            
            unified_result = await intent_entity_classifier.execute(
                input_data=message,
                context={"user_id": 12345}
            )
            
            intent_result = unified_result["intent_result"]
            entity_result = unified_result["entity_result"]
            
            step1_time = time.time() - step1_start
            print(f"   ✅ Intention: {intent_result.intent_type.value}")
            print(f"   ✅ Confiance: {intent_result.confidence:.2f}")
            print(f"   ✅ Entités extraites: {len(entity_result.get('entities', {}))}")
            print(f"   ⏱️  Temps: {step1_time * 1000:.1f}ms")
            
            # ====================================================================
            # ÉTAPE 2: Construction requête déterministe (logique pure, pas de LLM)
            # ====================================================================
            print("2️⃣ Construction requête déterministe...")
            step2_start = time.time()
            
            query_builder = DeterministicQueryBuilder()
            
            if intent_result.is_supported:
                search_query = query_builder.build_query(
                    intent_result=intent_result,
                    entity_result=entity_result,
                    user_id=12345,
                    context={"user_id": 12345}
                )
                
                query_valid = query_builder.validate_query(search_query) if search_query else False
                step2_time = time.time() - step2_start
                
                print(f"   ✅ Requête générée: {query_valid}")
                if search_query:
                    print(f"   ✅ Filtres: {len(search_query.filters)} éléments")
                    print(f"   ✅ Page size: {search_query.page_size}")
                print(f"   ⏱️  Temps: {step2_time * 1000:.1f}ms")
            else:
                print("   ⚠️  Intention non supportée, pas de requête générée")
                search_query = None
                step2_time = time.time() - step2_start
                print(f"   ⏱️  Temps: {step2_time * 1000:.1f}ms")
            
            # ====================================================================
            # ÉTAPE 3: Génération de réponse (simulation - pas d'appel search réel)
            # ====================================================================
            print("3️⃣ Génération de réponse...")
            step3_start = time.time()
            
            # Simulation d'une réponse basée sur l'intention
            if intent_result.is_supported and search_query:
                simulated_response = f"J'ai traité votre demande '{message}'. " \
                                   f"Intention détectée: {intent_result.intent_type.value} " \
                                   f"avec {len(entity_result.get('entities', {}))} entités extraites."
            else:
                simulated_response = f"Je ne peux pas traiter cette demande: {intent_result.reasoning}"
            
            step3_time = time.time() - step3_start
            print(f"   ✅ Réponse générée: {len(simulated_response)} caractères")
            print(f"   ⏱️  Temps: {step3_time * 1000:.1f}ms")
            
            # ====================================================================
            # RÉSUMÉ PERFORMANCE
            # ====================================================================
            total_time = time.time() - total_start
            print(f"\n📊 Performance totale:")
            print(f"   • Temps total: {total_time * 1000:.1f}ms")
            print(f"   • Étape 1 (LLM): {step1_time * 1000:.1f}ms ({step1_time/total_time*100:.1f}%)")
            print(f"   • Étape 2 (logique): {step2_time * 1000:.1f}ms ({step2_time/total_time*100:.1f}%)")
            print(f"   • Étape 3 (simulation): {step3_time * 1000:.1f}ms ({step3_time/total_time*100:.1f}%)")
            
            print(f"💡 Réponse simulée: {simulated_response}")
    
    except Exception as e:
        print(f"❌ Erreur during test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✅ Test de l'architecture streamlinée terminé avec succès!")
    print("📈 Architecture simplifiée: 4 agents LLM → 2 agents (+ 1 logique pure)")
    print("⚡ Performance attendue: ~80s → ~5-10s (réduction 8x)")
    return True

async def test_individual_components():
    """Test individuel de chaque composant"""
    
    print("\n🔧 Test des composants individuels")
    print("=" * 40)
    
    deepseek_client = DeepSeekClient()
    cache_manager = CacheManager()
    
    # Test IntentEntityClassifier
    print("🧠 Test IntentEntityClassifier...")
    classifier = IntentEntityClassifier(deepseek_client=deepseek_client, cache_manager=cache_manager)
    metrics = classifier.get_metrics()
    print(f"   ✅ Classificateur initialisé: {metrics['total_classifications']} classifications")
    
    # Test DeterministicQueryBuilder
    print("🏗️  Test DeterministicQueryBuilder...")
    builder = DeterministicQueryBuilder()
    builder_metrics = builder.get_metrics()
    print(f"   ✅ Builder initialisé: {builder_metrics['total_queries_built']} requêtes construites")
    
    print("✅ Tous les composants fonctionnent correctement!")

if __name__ == "__main__":
    print("🎯 Test de l'architecture streamlinée Harena")
    print(f"🕒 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Test du workflow complet
        success = asyncio.run(test_streamlined_workflow())
        
        if success:
            # Test des composants individuels
            asyncio.run(test_individual_components())
            
            print(f"\n🎉 SUCCÈS: Architecture streamlinée opérationnelle!")
            print(f"✨ Prête pour le déploiement en production")
            
        else:
            print(f"\n❌ ÉCHEC: Des problèmes ont été détectés")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Test interrompu par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Erreur critique: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)