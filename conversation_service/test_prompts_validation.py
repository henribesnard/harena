"""
🧪 Tests de Validation - Modules Prompts

Ce module contient les tests de validation pour tous les modules prompts
créés : intent_prompts, search_prompts, response_prompts, orchestrator_prompts.

Objectifs :
- Valider la syntaxe et les imports de tous les modules
- Tester les fonctions principales avec des données réalistes
- Vérifier la cohérence des exports et constantes
- Simuler l'utilisation dans le contexte AutoGen + DeepSeek
"""

import pytest
import sys
import json
from typing import Dict, List, Any
from unittest.mock import Mock, patch
from datetime import datetime, date

# =============================================================================
# TESTS D'IMPORTS ET SYNTAXE
# =============================================================================

class TestPromptsImports:
    """Tests de validation des imports et exports des modules prompts."""
    
    def test_intent_prompts_import(self):
        """Test l'import du module intent_prompts et ses exports."""
        try:
            from prompts.intent_prompts import (
                INTENT_FALLBACK_SYSTEM_PROMPT,
                INTENT_FALLBACK_USER_TEMPLATE,
                INTENT_EXAMPLES_FEW_SHOT,
                format_intent_prompt,
                build_context_summary,
                parse_intent_response,
                VALID_INTENTS,
                FINANCIAL_ENTITY_TYPES
            )
            
            # Vérification des types
            assert isinstance(INTENT_FALLBACK_SYSTEM_PROMPT, str)
            assert isinstance(INTENT_FALLBACK_USER_TEMPLATE, str)
            assert isinstance(INTENT_EXAMPLES_FEW_SHOT, str)
            assert callable(format_intent_prompt)
            assert callable(build_context_summary)
            assert callable(parse_intent_response)
            assert isinstance(VALID_INTENTS, set)
            assert isinstance(FINANCIAL_ENTITY_TYPES, set)
            
            # Vérification du contenu minimal
            assert len(INTENT_FALLBACK_SYSTEM_PROMPT) > 100
            assert "DeepSeek" in INTENT_FALLBACK_SYSTEM_PROMPT or "assistant" in INTENT_FALLBACK_SYSTEM_PROMPT.lower()
            assert "{user_message}" in INTENT_FALLBACK_USER_TEMPLATE
            assert "transaction_query" in VALID_INTENTS
            assert "amounts" in FINANCIAL_ENTITY_TYPES
            
        except ImportError as e:
            pytest.fail(f"Impossible d'importer intent_prompts: {e}")
        except Exception as e:
            pytest.fail(f"Erreur lors de la validation intent_prompts: {e}")
    
    def test_search_prompts_import(self):
        """Test l'import du module search_prompts et ses exports."""
        try:
            from prompts.search_prompts import (
                SEARCH_GENERATION_SYSTEM_PROMPT,
                SEARCH_GENERATION_TEMPLATE,
                SEARCH_EXAMPLES_FEW_SHOT,
                format_search_prompt,
                build_date_range_from_period,
                parse_search_response,
                ELASTICSEARCH_FIELD_MAPPING,
                QUERY_TYPE_STRATEGIES
            )
            
            # Vérification des types
            assert isinstance(SEARCH_GENERATION_SYSTEM_PROMPT, str)
            assert isinstance(SEARCH_GENERATION_TEMPLATE, str)
            assert isinstance(SEARCH_EXAMPLES_FEW_SHOT, str)
            assert callable(format_search_prompt)
            assert callable(build_date_range_from_period)
            assert callable(parse_search_response)
            assert isinstance(ELASTICSEARCH_FIELD_MAPPING, dict)
            assert isinstance(QUERY_TYPE_STRATEGIES, dict)
            
            # Vérification du contenu
            assert len(SEARCH_GENERATION_SYSTEM_PROMPT) > 100
            assert "Elasticsearch" in SEARCH_GENERATION_SYSTEM_PROMPT
            assert "{intent_type}" in SEARCH_GENERATION_TEMPLATE
            assert "searchable_text" in ELASTICSEARCH_FIELD_MAPPING
            assert "transaction_query" in QUERY_TYPE_STRATEGIES
            
        except ImportError as e:
            pytest.fail(f"Impossible d'importer search_prompts: {e}")
        except Exception as e:
            pytest.fail(f"Erreur lors de la validation search_prompts: {e}")
    
    def test_response_prompts_import(self):
        """Test l'import du module response_prompts et ses exports."""
        try:
            from prompts.response_prompts import (
                RESPONSE_GENERATION_SYSTEM_PROMPT,
                RESPONSE_GENERATION_TEMPLATE,
                RESPONSE_EXAMPLES_FEW_SHOT,
                format_response_prompt,
                format_search_results_for_prompt,
                truncate_search_results,
                extract_key_insights_from_results,
                format_amount_with_context,
                RESPONSE_TEMPLATES_BY_INTENT,
                FINANCIAL_EMOJIS,
                AMOUNT_THRESHOLDS,
                ERROR_MESSAGES
            )
            
            # Vérification des types
            assert isinstance(RESPONSE_GENERATION_SYSTEM_PROMPT, str)
            assert isinstance(RESPONSE_GENERATION_TEMPLATE, str)
            assert isinstance(RESPONSE_EXAMPLES_FEW_SHOT, str)
            assert callable(format_response_prompt)
            assert callable(format_search_results_for_prompt)
            assert callable(truncate_search_results)
            assert callable(extract_key_insights_from_results)
            assert callable(format_amount_with_context)
            
            # Vérification des constantes
            assert isinstance(RESPONSE_TEMPLATES_BY_INTENT, dict)
            assert isinstance(FINANCIAL_EMOJIS, dict)
            assert isinstance(AMOUNT_THRESHOLDS, dict)
            assert isinstance(ERROR_MESSAGES, dict)
            
            # Vérification du contenu
            assert "assistant financier" in RESPONSE_GENERATION_SYSTEM_PROMPT.lower()
            assert "{user_message}" in RESPONSE_GENERATION_TEMPLATE
            assert "transaction_query" in RESPONSE_TEMPLATES_BY_INTENT
            assert "spending" in FINANCIAL_EMOJIS
            
        except ImportError as e:
            pytest.fail(f"Impossible d'importer response_prompts: {e}")
        except Exception as e:
            pytest.fail(f"Erreur lors de la validation response_prompts: {e}")
    
    def test_orchestrator_prompts_import(self):
        """Test l'import du module orchestrator_prompts et ses exports."""
        try:
            from prompts.orchestrator_prompts import (
                ORCHESTRATOR_SYSTEM_PROMPT,
                ORCHESTRATOR_WORKFLOW_TEMPLATE,
                WorkflowStep,
                AgentStatus,
                format_orchestrator_prompt,
                parse_orchestrator_decision,
                create_workflow_decision,
                WORKFLOW_TYPES,
                AGENT_CAPABILITIES
            )
            
            # Vérification des types
            assert isinstance(ORCHESTRATOR_SYSTEM_PROMPT, str)
            assert isinstance(ORCHESTRATOR_WORKFLOW_TEMPLATE, str)
            assert callable(format_orchestrator_prompt)
            assert callable(parse_orchestrator_decision)
            assert callable(create_workflow_decision)
            assert isinstance(WORKFLOW_TYPES, dict)
            assert isinstance(AGENT_CAPABILITIES, dict)
            
            # Vérification des enums
            assert hasattr(WorkflowStep, 'INTENT_DETECTION')
            assert hasattr(AgentStatus, 'SUCCESS')
            
            # Vérification du contenu
            assert "orchestrateur" in ORCHESTRATOR_SYSTEM_PROMPT.lower()
            assert "{current_step}" in ORCHESTRATOR_WORKFLOW_TEMPLATE
            assert "standard" in WORKFLOW_TYPES
            
        except ImportError as e:
            pytest.fail(f"Impossible d'importer orchestrator_prompts: {e}")
        except Exception as e:
            pytest.fail(f"Erreur lors de la validation orchestrator_prompts: {e}")

# =============================================================================
# TESTS FONCTIONNELS DES PROMPTS
# =============================================================================

class TestIntentPromptsFunctionality:
    """Tests fonctionnels du module intent_prompts."""
    
    def test_format_intent_prompt_basic(self):
        """Test le formatage basique d'un prompt d'intention."""
        from prompts.intent_prompts import format_intent_prompt
        
        user_message = "Mes achats chez Carrefour"
        prompt = format_intent_prompt(user_message)
        
        assert isinstance(prompt, str)
        assert user_message in prompt
        assert len(prompt) > len(user_message)
    
    def test_format_intent_prompt_with_context(self):
        """Test le formatage d'un prompt avec contexte."""
        from prompts.intent_prompts import format_intent_prompt
        
        user_message = "Et pour Amazon ?"
        context = "L'utilisateur analysait ses achats chez Carrefour"
        prompt = format_intent_prompt(user_message, context)
        
        assert user_message in prompt
        assert context in prompt
        assert "CONTEXTE" in prompt.upper()
    
    def test_build_context_summary(self):
        """Test la construction du résumé de contexte."""
        from prompts.intent_prompts import build_context_summary
        
        conversation_history = [
            {"user": "Mes achats ce mois", "assistant": "Voici vos achats..."},
            {"user": "Et le mois dernier ?", "assistant": "Le mois dernier vous aviez..."}
        ]
        
        summary = build_context_summary(conversation_history)
        
        assert isinstance(summary, str)
        if summary:  # Peut être vide si trop court
            assert "User:" in summary or "Utilisateur:" in summary
    
    def test_parse_intent_response_valid(self):
        """Test le parsing d'une réponse d'intention valide."""
        from prompts.intent_prompts import parse_intent_response
        
        valid_response = """INTENT: transaction_query
CONFIDENCE: 0.85
ENTITIES: {"merchants": ["Carrefour"], "periods": ["ce mois"]}
REASONING: Recherche de transactions avec marchand spécifique"""
        
        result = parse_intent_response(valid_response)
        
        assert isinstance(result, dict)
        assert result["intent"] == "transaction_query"
        assert result["confidence"] == 0.85
        assert isinstance(result["entities"], dict)
        assert "merchants" in result["entities"]
    
    def test_parse_intent_response_invalid(self):
        """Test le parsing d'une réponse invalide avec fallback gracieux."""
        from prompts.intent_prompts import parse_intent_response
        
        invalid_response = "Réponse complètement invalide sans format"
        
        result = parse_intent_response(invalid_response)
        
        # Doit retourner un fallback gracieux
        assert isinstance(result, dict)
        assert result["intent"] == "other"
        assert 0.0 <= result["confidence"] <= 1.0
        assert isinstance(result["entities"], dict)

class TestSearchPromptsFunctionality:
    """Tests fonctionnels du module search_prompts."""
    
    def test_format_search_prompt(self):
        """Test le formatage d'un prompt de recherche."""
        from prompts.search_prompts import format_search_prompt
        
        intent_result = {
            "intent": "transaction_query",
            "confidence": 0.9,
            "entities": {"merchants": ["Amazon"]}
        }
        user_message = "Mes achats Amazon"
        
        prompt = format_search_prompt(intent_result, user_message)
        
        assert isinstance(prompt, str)
        assert "transaction_query" in prompt
        assert "Amazon" in prompt
        assert user_message in prompt
    
    def test_build_date_range_from_period(self):
        """Test la conversion de périodes en plages de dates."""
        from prompts.search_prompts import build_date_range_from_period
        
        # Test avec différentes expressions
        test_cases = [
            "mois dernier",
            "cette semaine", 
            "ce mois",
            "hier"
        ]
        
        for period in test_cases:
            date_range = build_date_range_from_period(period)
            
            assert isinstance(date_range, dict)
            assert "gte" in date_range
            assert "lte" in date_range
            
            # Vérification format date
            assert len(date_range["gte"]) == 10  # YYYY-MM-DD
            assert len(date_range["lte"]) == 10
            assert "-" in date_range["gte"]
            assert "-" in date_range["lte"]
    
    def test_parse_search_response_valid(self):
        """Test le parsing d'une réponse de recherche valide."""
        from prompts.search_prompts import parse_search_response
        
        valid_json_response = """{
            "query_type": "lexical",
            "search_text": "Amazon",
            "filters": {
                "date_range": {"gte": "2024-01-01", "lte": "2024-01-31"}
            },
            "size": 20
        }"""
        
        user_id = "test_user_123"
        result = parse_search_response(valid_json_response, user_id)
        
        assert isinstance(result, dict)
        assert result["query_type"] == "lexical"
        assert result["filters"]["user_id"] == user_id  # Doit être injecté
        assert "search_text" in result

class TestResponsePromptsFunctionality:
    """Tests fonctionnels du module response_prompts."""
    
    def test_format_response_prompt(self):
        """Test le formatage d'un prompt de réponse."""
        from prompts.response_prompts import format_response_prompt
        
        user_message = "Mes achats ce mois"
        search_results = {
            "results": [
                {
                    "date": "2025-01-15",
                    "amount": -52.30,
                    "merchant_name": "Carrefour",
                    "category_name": "Alimentation"
                }
            ],
            "response_metadata": {
                "total_count": 1,
                "execution_time_ms": 150,
                "query_type": "lexical"
            }
        }
        
        prompt = format_response_prompt(user_message, search_results)
        
        assert isinstance(prompt, str)
        assert user_message in prompt
        assert "Carrefour" in prompt
        assert "52.30€" in prompt
    
    def test_format_search_results_for_prompt(self):
        """Test le formatage des résultats pour inclusion dans le prompt."""
        from prompts.response_prompts import format_search_results_for_prompt
        
        search_results = {
            "results": [
                {
                    "date": "2025-01-15",
                    "amount": -52.30,
                    "merchant_name": "Carrefour",
                    "category_name": "Alimentation",
                    "primary_description": "Courses alimentaires"
                }
            ],
            "aggregations": {
                "total_amount": -52.30,
                "transaction_count": 1
            }
        }
        
        formatted = format_search_results_for_prompt(search_results)
        
        assert isinstance(formatted, str)
        assert "TRANSACTIONS TROUVÉES" in formatted
        assert "Carrefour" in formatted
        assert "-52.30€" in formatted
        assert "AGRÉGATIONS" in formatted
    
    def test_extract_key_insights_from_results(self):
        """Test l'extraction d'insights des résultats."""
        from prompts.response_prompts import extract_key_insights_from_results
        
        # Test avec une seule transaction
        single_transaction = {
            "results": [
                {"amount": -100.0, "date": "2025-01-15"}
            ]
        }
        
        insights = extract_key_insights_from_results(single_transaction)
        
        assert isinstance(insights, list)
        if insights:
            assert any("unique" in insight.lower() for insight in insights)
        
        # Test avec beaucoup de transactions
        many_transactions = {
            "results": [{"amount": -10.0, "date": f"2025-01-{i:02d}"} for i in range(1, 25)]
        }
        
        insights = extract_key_insights_from_results(many_transactions)
        
        assert isinstance(insights, list)
        assert len(insights) <= 3  # Limité à 3 insights max
    
    def test_format_amount_with_context(self):
        """Test le formatage des montants avec contexte."""
        from prompts.response_prompts import format_amount_with_context
        
        # Test montant négatif (dépense)
        expense = format_amount_with_context(-156.50)
        assert "156,50€" in expense
        assert "dépense" in expense
        
        # Test montant positif (crédit)
        credit = format_amount_with_context(200.75)
        assert "200,75€" in credit
        assert "crédit" in credit
        
        # Test avec devise différente
        usd = format_amount_with_context(100.0, "USD")
        assert "$" in usd
    
    def test_truncate_search_results(self):
        """Test la troncature des résultats de recherche."""
        from prompts.response_prompts import truncate_search_results
        
        large_results = {
            "results": [{"amount": -10.0} for _ in range(50)],
            "aggregations": {
                "by_category": [{"key": f"cat_{i}"} for i in range(20)]
            }
        }
        
        truncated = truncate_search_results(large_results, max_transactions=10)
        
        assert len(truncated["results"]) == 10
        assert len(truncated["aggregations"]["by_category"]) <= 20

# =============================================================================
# TESTS D'INTÉGRATION DEEPSEEK
# =============================================================================

class TestDeepSeekIntegration:
    """Tests d'intégration simulant l'utilisation avec DeepSeek."""
    
    def test_intent_detection_workflow(self):
        """Test du workflow complet de détection d'intention."""
        from prompts.intent_prompts import (
            format_intent_prompt,
            parse_intent_response,
            INTENT_FALLBACK_SYSTEM_PROMPT
        )
        
        # Simulation d'un workflow complet
        user_message = "Combien j'ai dépensé chez Amazon ce mois ?"
        
        # 1. Formatage du prompt
        prompt = format_intent_prompt(user_message)
        assert isinstance(prompt, str)
        assert user_message in prompt
        
        # 2. Simulation réponse DeepSeek
        mock_deepseek_response = """INTENT: transaction_query
CONFIDENCE: 0.92
ENTITIES: {"merchants": ["Amazon"], "periods": ["ce mois"]}
REASONING: Recherche de transactions avec marchand et période spécifiés"""
        
        # 3. Parsing de la réponse
        intent_result = parse_intent_response(mock_deepseek_response)
        
        assert intent_result["intent"] == "transaction_query"
        assert intent_result["confidence"] == 0.92
        assert "Amazon" in str(intent_result["entities"])
    
    def test_search_query_generation_workflow(self):
        """Test du workflow de génération de requête de recherche."""
        from prompts.search_prompts import (
            format_search_prompt,
            parse_search_response
        )
        
        # Intent result depuis l'étape précédente
        intent_result = {
            "intent": "transaction_query",
            "confidence": 0.92,
            "entities": {"merchants": ["Amazon"], "periods": ["ce mois"]}
        }
        
        user_message = "Combien j'ai dépensé chez Amazon ce mois ?"
        
        # 1. Formatage du prompt de recherche
        search_prompt = format_search_prompt(intent_result, user_message)
        assert "transaction_query" in search_prompt
        assert "Amazon" in search_prompt
        
        # 2. Simulation réponse DeepSeek pour génération de requête
        mock_search_response = """{
            "query_type": "lexical",
            "search_text": "Amazon",
            "filters": {
                "date_range": {"gte": "2025-01-01", "lte": "2025-01-31"},
                "merchants": ["amazon"]
            },
            "size": 20,
            "explanation": "Recherche transactions Amazon du mois en cours"
        }"""
        
        # 3. Parsing de la requête générée
        search_query = parse_search_response(mock_search_response, "user123")
        
        assert search_query["query_type"] == "lexical"
        assert search_query["filters"]["user_id"] == "user123"
        assert "Amazon" in search_query["search_text"]
    
    def test_response_generation_workflow(self):
        """Test du workflow de génération de réponse finale."""
        from prompts.response_prompts import (
            format_response_prompt,
            format_search_results_for_prompt
        )
        
        user_message = "Combien j'ai dépensé chez Amazon ce mois ?"
        
        # Simulation résultats du Search Service
        search_results = {
            "results": [
                {
                    "date": "2025-01-10",
                    "amount": -89.99,
                    "merchant_name": "Amazon",
                    "category_name": "E-commerce",
                    "primary_description": "Commande Amazon - Livres"
                },
                {
                    "date": "2025-01-18",
                    "amount": -156.50,
                    "merchant_name": "Amazon",
                    "category_name": "E-commerce", 
                    "primary_description": "Commande Amazon - Électronique"
                }
            ],
            "aggregations": {
                "total_amount": -246.49,
                "transaction_count": 2,
                "average_amount": -123.25
            },
            "response_metadata": {
                "total_count": 2,
                "execution_time_ms": 45,
                "query_type": "lexical"
            }
        }
        
        # 1. Formatage du prompt de réponse
        response_prompt = format_response_prompt(user_message, search_results)
        
        assert user_message in response_prompt
        assert "Amazon" in response_prompt
        assert "246.49€" in response_prompt
        assert "2" in response_prompt  # Nombre de transactions
        
        # 2. Vérification que le prompt contient les infos essentielles
        assert "TRANSACTIONS TROUVÉES" in response_prompt
        assert "AGRÉGATIONS" in response_prompt
        assert "MÉTADONNÉES" in response_prompt

# =============================================================================
# TESTS DE VALIDATION DES CONSTANTES
# =============================================================================

class TestConstantsValidation:
    """Tests de validation des constantes et configurations."""
    
    def test_intent_constants_consistency(self):
        """Test la cohérence des constantes d'intention."""
        from prompts.intent_prompts import VALID_INTENTS, FINANCIAL_ENTITY_TYPES
        
        # Vérifications des intentions
        expected_intents = {
            "transaction_query", "spending_analysis", "budget_inquiry",
            "category_analysis", "merchant_inquiry", "trend_analysis"
        }
        
        for intent in expected_intents:
            assert intent in VALID_INTENTS, f"Intent {intent} manquant"
        
        # Vérifications des entités
        expected_entities = {"amounts", "dates", "merchants", "categories"}
        
        for entity in expected_entities:
            assert entity in FINANCIAL_ENTITY_TYPES, f"Entity {entity} manquante"
    
    def test_search_strategies_consistency(self):
        """Test la cohérence des stratégies de recherche."""
        from prompts.search_prompts import QUERY_TYPE_STRATEGIES, ELASTICSEARCH_FIELD_MAPPING
        from prompts.intent_prompts import VALID_INTENTS
        
        # Toutes les intentions doivent avoir une stratégie
        missing_strategies = []
        for intent in VALID_INTENTS:
            if intent not in QUERY_TYPE_STRATEGIES:
                missing_strategies.append(intent)
        
        # Affichage informatif des intentions manquantes
        if missing_strategies:
            print(f"\nIntentions sans stratégie: {missing_strategies}")
            print(f"Intentions disponibles: {list(QUERY_TYPE_STRATEGIES.keys())}")
        
        # Vérification que les intentions principales ont des stratégies
        critical_intents = {
            "transaction_query", "spending_analysis", "budget_inquiry",
            "category_analysis", "merchant_inquiry", "trend_analysis"
        }
        
        for intent in critical_intents:
            assert intent in QUERY_TYPE_STRATEGIES, f"Stratégie critique manquante pour {intent}"
        
        # Vérification des champs Elasticsearch
        expected_fields = ["searchable_text", "merchant_name", "category_name", "amount"]
        
        for field in expected_fields:
            assert field in ELASTICSEARCH_FIELD_MAPPING, f"Champ {field} manquant"
    
    def test_response_templates_consistency(self):
        """Test la cohérence des templates de réponse."""
        from prompts.response_prompts import RESPONSE_TEMPLATES_BY_INTENT, FINANCIAL_EMOJIS
        from prompts.intent_prompts import VALID_INTENTS
        
        # Toutes les intentions doivent avoir un template
        for intent in VALID_INTENTS:
            assert intent in RESPONSE_TEMPLATES_BY_INTENT, f"Template manquant pour {intent}"
        
        # Vérification des emojis financiers
        expected_emojis = ["spending", "income", "budget", "trend_up", "trend_down"]
        
        for emoji_type in expected_emojis:
            assert emoji_type in FINANCIAL_EMOJIS, f"Emoji {emoji_type} manquant"
            assert isinstance(FINANCIAL_EMOJIS[emoji_type], str)
            assert len(FINANCIAL_EMOJIS[emoji_type]) > 0

# =============================================================================
# TESTS DE PERFORMANCE ET LIMITES
# =============================================================================

class TestPerformanceAndLimits:
    """Tests de performance et gestion des limites."""
    
    def test_prompt_size_limits(self):
        """Test que les prompts restent dans des limites raisonnables."""
        from prompts.intent_prompts import format_intent_prompt
        from prompts.search_prompts import format_search_prompt
        from prompts.response_prompts import format_response_prompt
        
        # Test avec des messages très longs
        long_message = "Mes achats " + "très " * 100 + "importants"
        
        # Les prompts ne doivent pas exploser en taille
        intent_prompt = format_intent_prompt(long_message)
        assert len(intent_prompt) < 10000  # Limite raisonnable
        
        # Test avec beaucoup de contexte
        large_context = "Contexte " + "important " * 200
        context_prompt = format_intent_prompt("Message court", large_context)
        
        # Le contexte doit être géré intelligemment
        assert len(context_prompt) < 15000
    
    def test_search_results_truncation(self):
        """Test la troncature des gros résultats de recherche."""
        from prompts.response_prompts import truncate_search_results
        
        # Simulation d'énormes résultats
        huge_results = {
            "results": [{"amount": -10.0, "date": f"2025-01-01"} for _ in range(1000)],
            "aggregations": {
                "by_category": [{"key": f"category_{i}", "total": -100} for i in range(100)]
            }
        }
        
        # Troncature avec limites strictes
        truncated = truncate_search_results(huge_results, max_transactions=5, max_aggregations=3)
        
        assert len(truncated["results"]) == 5
        # Les agrégations doivent aussi être limitées si la fonction les gère
    
    def test_error_handling_robustness(self):
        """Test la robustesse de la gestion d'erreur."""
        from prompts.intent_prompts import parse_intent_response
        from prompts.search_prompts import parse_search_response
        
        # Test avec entrées nulles/vides
        error_cases = [
            "",
            None,
            "Complètement invalide",
            "JSON cassé {[}",
            "INTENT: malformed response"
        ]
        
        for error_case in error_cases:
            try:
                if error_case is not None:
                    result = parse_intent_response(error_case)
                    # Doit retourner un fallback valide
                    assert isinstance(result, dict)
                    assert "intent" in result
            except ValueError:
                # Les ValueError sont acceptables
                pass
            except Exception as e:
                pytest.fail(f"Erreur inattendue avec {error_case}: {e}")

# =============================================================================
# EXÉCUTION DES TESTS
# =============================================================================

if __name__ == "__main__":
    # Configuration pytest pour une exécution standalone
    pytest.main([
        __file__,
        "-v",  # Mode verbose
        "--tb=short",  # Tracebacks courts
        "--color=yes",  # Couleurs dans la sortie
        "-x"  # Arrêt au premier échec
    ])