#!/usr/bin/env python3
"""
Test simple de la nouvelle architecture streamlinee
Teste les 3 etapes sans emojis pour compatibilite Windows
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
    """Test complet du workflow streamline"""
    
    print("=== TEST ARCHITECTURE STREAMLINEE ===")
    print("Workflow: 3 etapes au lieu de 5")
    print("1. IntentEntityClassifier (1 LLM)")
    print("2. DeterministicQueryBuilder (logique)")  
    print("3. ResponseGenerator (1 LLM)")
    print("=" * 50)
    
    test_messages = [
        "mes dernieres transactions ?",
        "combien j'ai depense chez Amazon ce mois ?",
        "mes virements de janvier"
    ]
    
    try:
        print("Initialisation des composants...")
        deepseek_client = DeepSeekClient()
        cache_manager = CacheManager()
        
        total_workflow_start = time.time()
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n--- Test {i}/{len(test_messages)}: '{message}' ---")
            
            message_start = time.time()
            
            # ETAPE 1: Classification + extraction unifiee
            print("ETAPE 1: Classification intention + extraction entites...")
            step1_start = time.time()
            
            classifier = IntentEntityClassifier(
                deepseek_client=deepseek_client,
                cache_manager=cache_manager
            )
            
            unified_result = await classifier.execute(
                input_data=message,
                context={"user_id": 12345}
            )
            
            intent_result = unified_result["intent_result"]
            entity_result = unified_result["entity_result"]
            
            step1_time = time.time() - step1_start
            print(f"  -> Intention: {intent_result.intent_type.value}")
            print(f"  -> Confiance: {intent_result.confidence:.2f}")
            print(f"  -> Entites: {len(entity_result.get('entities', {}))}")
            print(f"  -> Temps: {step1_time * 1000:.0f}ms")
            
            # ETAPE 2: Construction requete deterministe
            print("ETAPE 2: Construction requete deterministe...")
            step2_start = time.time()
            
            builder = DeterministicQueryBuilder()
            search_query = None
            
            if intent_result.is_supported:
                search_query = builder.build_query(
                    intent_result=intent_result,
                    entity_result=entity_result,
                    user_id=12345,
                    context={"user_id": 12345}
                )
                
                query_valid = builder.validate_query(search_query) if search_query else False
                
                print(f"  -> Requete generee: {'OUI' if query_valid else 'NON'}")
                if search_query:
                    if isinstance(search_query.filters, dict):
                        print(f"  -> Filtres: {len(search_query.filters)}")
                    else:
                        print(f"  -> Filtres: type {type(search_query.filters)}")
                    print(f"  -> Page size: {search_query.page_size}")
            else:
                print("  -> Intention non supportee")
            
            step2_time = time.time() - step2_start
            print(f"  -> Temps: {step2_time * 1000:.0f}ms")
            
            # ETAPE 3: Simulation generation reponse
            print("ETAPE 3: Generation reponse (simulee)...")
            step3_start = time.time()
            
            if intent_result.is_supported and search_query:
                response = f"Demande traitee: {intent_result.intent_type.value}"
            else:
                response = f"Ne peut pas traiter: {intent_result.reasoning}"
            
            step3_time = time.time() - step3_start
            print(f"  -> Reponse: {len(response)} caracteres")
            print(f"  -> Temps: {step3_time * 1000:.0f}ms")
            
            # Bilan message
            message_time = time.time() - message_start
            print(f"TOTAL MESSAGE: {message_time * 1000:.0f}ms")
            print(f"  - LLM (etape 1): {step1_time/message_time*100:.1f}%")
            print(f"  - Logique (etape 2): {step2_time/message_time*100:.1f}%") 
            print(f"  - Simulation (etape 3): {step3_time/message_time*100:.1f}%")
        
        total_time = time.time() - total_workflow_start
        print(f"\n" + "=" * 50)
        print(f"RESULTAT GLOBAL:")
        print(f"  - Temps total: {total_time * 1000:.0f}ms")
        print(f"  - Moyenne par message: {total_time * 1000 / len(test_messages):.0f}ms")
        print(f"  - Performance cible atteinte: {'OUI' if total_time < 30 else 'NON'}")
        
        return True
        
    except Exception as e:
        print(f"ERREUR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("TEST ARCHITECTURE STREAMLINEE HARENA")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        success = asyncio.run(test_streamlined_workflow())
        
        if success:
            print("\nSUCCES: Architecture streamlinee operationnelle!")
            print("Reduction prevue: 80s -> 5-10s (8x plus rapide)")
        else:
            print("\nECHEC: Des problemes ont ete detectes")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nTest interrompu par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        print(f"\nErreur critique: {str(e)}")
        sys.exit(1)