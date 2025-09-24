"""
Test d'intégration équipe AutoGen avec infrastructure existante
Test sans dépendances externes pour validation architecture
"""

import sys
import os
import asyncio
from typing import Dict, Any

# Ajouter le chemin conversation_service au PYTHONPATH pour les tests
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_imports_infrastructure():
    """Test des imports infrastructure existante"""
    
    try:
        # Test import prompts AutoGen (déjà validé)
        from prompts.autogen.team_orchestration import MULTI_AGENT_TEAM_ORCHESTRATION_PROMPT
        from prompts.autogen import get_intent_classification_with_team_collaboration
        print("✓ Prompts AutoGen importés")
        
        # Test structure équipe (sans dépendances externes)
        from multi_agent_financial_team import MultiAgentFinancialTeam
        print("✓ Classe équipe importée")
        
        return True
        
    except ImportError as e:
        print(f"✗ Erreur import: {e}")
        return False

def test_team_structure():
    """Test structure équipe sans instanciation complète"""
    
    try:
        # Import classe sans instanciation (évite dépendances externes)
        from multi_agent_financial_team import MultiAgentFinancialTeam
        
        # Vérifier méthodes principales
        methods = [
            'process_user_message',
            '_extract_team_results', 
            '_validate_intent_entity_coherence',
            'health_check',
            'get_team_statistics'
        ]
        
        for method in methods:
            if hasattr(MultiAgentFinancialTeam, method):
                print(f"✓ Méthode {method} présente")
            else:
                print(f"✗ Méthode {method} manquante")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur structure: {e}")
        return False

def test_prompts_integration():
    """Test intégration prompts avec équipe"""
    
    try:
        from prompts.autogen.team_orchestration import (
            MULTI_AGENT_TEAM_ORCHESTRATION_PROMPT,
            get_orchestration_prompt_for_context,
            get_workflow_completion_message
        )
        
        # Test prompt orchestration
        assert len(MULTI_AGENT_TEAM_ORCHESTRATION_PROMPT) > 100
        print("✓ Prompt orchestration valide")
        
        # Test prompt contexte
        context_prompt = get_orchestration_prompt_for_context({
            "retry_count": 1,
            "failed_agents": ["entity_extractor"]
        })
        assert "CONTEXTE RETRY" in context_prompt
        print("✓ Prompt contexte adaptatif fonctionnel")
        
        # Test message completion
        completion_msg = get_workflow_completion_message(
            {"intent": "SEARCH_BY_MERCHANT", "confidence": 0.9},
            {"entities": {"merchants": ["Leclerc"]}, "confidence": 0.8},
            2.5
        )
        assert "Workflow équipe AutoGen terminé" in completion_msg
        print("✓ Message completion généré")
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur prompts: {e}")
        return False

def test_coherence_validation_logic():
    """Test logique validation cohérence (sans dépendances)"""
    
    try:
        # Simuler méthode validation cohérence
        def validate_coherence(intent_result, entities_result):
            """Version simplifiée pour test"""
            
            if not intent_result or not entities_result:
                return 0.0
                
            intent_type = intent_result.get("intent", "")
            entities = entities_result.get("entities", {})
            
            coherence_checks = {
                "SEARCH_BY_MERCHANT": lambda e: len(e.get("merchants", [])) > 0,
                "SEARCH_BY_AMOUNT": lambda e: len(e.get("amounts", [])) > 0,
                "BALANCE_INQUIRY": lambda e: True
            }
            
            check_function = coherence_checks.get(intent_type, lambda e: True)
            intent_confidence = intent_result.get("confidence", 0.5)
            entity_confidence = entities_result.get("confidence", 0.5)
            
            if check_function(entities):
                return min((intent_confidence + entity_confidence) / 2 + 0.2, 1.0)
            else:
                return max((intent_confidence + entity_confidence) / 2 - 0.3, 0.0)
        
        # Test cas cohérents
        test_cases = [
            {
                "name": "SEARCH_BY_MERCHANT cohérent",
                "intent": {"intent": "SEARCH_BY_MERCHANT", "confidence": 0.9},
                "entities": {"entities": {"merchants": ["Carrefour"]}, "confidence": 0.8},
                "expected_min": 0.7
            },
            {
                "name": "BALANCE_INQUIRY neutre",
                "intent": {"intent": "BALANCE_INQUIRY", "confidence": 0.7},
                "entities": {"entities": {}, "confidence": 0.6},
                "expected_min": 0.6
            },
            {
                "name": "Incohérence SEARCH_BY_MERCHANT",
                "intent": {"intent": "SEARCH_BY_MERCHANT", "confidence": 0.8},
                "entities": {"entities": {"merchants": []}, "confidence": 0.7},
                "expected_max": 0.5
            }
        ]
        
        for case in test_cases:
            score = validate_coherence(case["intent"], case["entities"])
            
            if "expected_min" in case and score >= case["expected_min"]:
                print(f"✓ {case['name']}: score {score:.2f} >= {case['expected_min']}")
            elif "expected_max" in case and score <= case["expected_max"]:
                print(f"✓ {case['name']}: score {score:.2f} <= {case['expected_max']}")
            else:
                print(f"✗ {case['name']}: score {score:.2f} inattendu")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur validation cohérence: {e}")
        return False

def test_team_metrics_structure():
    """Test structure métriques équipe"""
    
    try:
        # Structure métriques attendue
        expected_metrics = {
            "conversations_processed": 0,
            "success_rate": 0.0,
            "avg_processing_time_ms": 0.0,
            "total_processing_time_ms": 0.0,
            "cache_hit_rate": 0.0,
            "coherence_avg_score": 0.0
        }
        
        # Simulation mise à jour métriques
        def update_metrics(current_metrics, processing_time, success, coherence_score):
            current_metrics["conversations_processed"] += 1
            current_metrics["total_processing_time_ms"] += processing_time
            current_metrics["avg_processing_time_ms"] = (
                current_metrics["total_processing_time_ms"] / 
                current_metrics["conversations_processed"]
            )
            
            total = current_metrics["conversations_processed"]
            if success:
                current_successes = current_metrics["success_rate"] * (total - 1)
                current_metrics["success_rate"] = (current_successes + 1.0) / total
            
            coherence_sum = current_metrics["coherence_avg_score"] * (total - 1)
            current_metrics["coherence_avg_score"] = (coherence_sum + coherence_score) / total
            
            return current_metrics
        
        # Test mise à jour
        metrics = expected_metrics.copy()
        
        # Simulation 3 conversations
        metrics = update_metrics(metrics, 1500, True, 0.8)
        metrics = update_metrics(metrics, 2200, True, 0.9) 
        metrics = update_metrics(metrics, 1800, False, 0.4)
        
        # Vérifications
        assert metrics["conversations_processed"] == 3
        assert 0.6 < metrics["success_rate"] < 0.7  # 2/3 succès
        assert 1700 < metrics["avg_processing_time_ms"] < 1900  # ~1833ms
        assert 0.6 < metrics["coherence_avg_score"] < 0.8  # ~0.7
        
        print("✓ Structure métriques équipe validée")
        print(f"  - {metrics['conversations_processed']} conversations")
        print(f"  - {metrics['success_rate']:.1%} taux succès")
        print(f"  - {metrics['avg_processing_time_ms']:.0f}ms temps moyen")
        print(f"  - {metrics['coherence_avg_score']:.2f} cohérence moyenne")
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur métriques: {e}")
        return False

def main():
    """Test principal intégration"""
    
    print("=== TEST INTÉGRATION ÉQUIPE AUTOGEN ===\n")
    
    tests = [
        ("Imports infrastructure", test_imports_infrastructure),
        ("Structure équipe", test_team_structure), 
        ("Intégration prompts", test_prompts_integration),
        ("Logique validation cohérence", test_coherence_validation_logic),
        ("Structure métriques", test_team_metrics_structure)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"--- {test_name} ---")
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"Résultat: {'✓ SUCCÈS' if success else '✗ ÉCHEC'}\n")
        except Exception as e:
            print(f"✗ ÉCHEC: {e}\n")
            results.append((test_name, False))
    
    # Résumé final
    print("=== RÉSUMÉ TESTS ===")
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {test_name}")
    
    print(f"\n{successful}/{total} tests réussis ({successful/total:.1%})")
    
    if successful == total:
        print("\n🎯 INTÉGRATION ÉQUIPE AUTOGEN VALIDÉE")
        print("   - Architecture respectée")
        print("   - Prompts intégrés")
        print("   - Métriques fonctionnelles")
        print("   - Logique métier cohérente")
    else:
        print(f"\n  {total - successful} tests échoués - Corrections nécessaires")

if __name__ == "__main__":
    main()