"""
🧪 Tests d'Intégration Complets - Architecture intent_rules

Ce module teste l'intégration complète de tous les composants :
- RuleLoader + PatternMatcher + RuleEngine
- Fonctions package-level (quick_detect_intent, etc.)
- Performance end-to-end
- Cas d'usage réels
"""

import pytest
import json
import tempfile
import shutil
import time
from pathlib import Path
from typing import Dict, List

# Import complet de l'architecture
import sys
sys.path.append(str(Path(__file__).parent.parent))

try:
    from intent_rules import (
        # Classes principales
        RuleLoader, PatternMatcher, RuleEngine,
        
        # Factory functions
        create_rule_loader, create_pattern_matcher, create_rule_engine,
        
        # Package functions
        get_default_loader, get_default_pattern_matcher, get_default_rule_engine,
        quick_detect_intent, extract_entities_quick,
        get_package_info, get_performance_report,
        reload_default_rules,
        
        # Types et structures
        IntentCategory, RuleMatch, EntityMatch, ExtractionResult
    )
    INTEGRATION_AVAILABLE = True
    print("✅ All intent_rules components imported successfully")
except ImportError as e:
    print(f"❌ Integration test failed - missing components: {e}")
    INTEGRATION_AVAILABLE = False


class TestIntegrationComplete:
    """Tests d'intégration complets pour toute l'architecture"""
    
    @pytest.fixture
    def realistic_rules_dir(self):
        """Fixture avec des règles réalistes pour tests d'intégration"""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        self._create_realistic_rules(temp_path)
        
        yield temp_path
        
        shutil.rmtree(temp_dir)
    
    def _create_realistic_rules(self, temp_path: Path):
        """Crée des règles réalistes pour tests end-to-end"""
        
        # Financial patterns - RÈGLES RÉALISTES
        financial_config = {
            "version": "1.0-integration-test",
            "last_updated": "2025-01-30",
            "description": "Realistic financial patterns for integration testing",
            "intents": {
                "SEARCH_BY_MERCHANT": {
                    "description": "Search transactions by merchant",
                    "intent_category": "SEARCH",
                    "confidence": 0.90,
                    "priority": 2,
                    "patterns": [
                        {
                            "regex": "\\b(amazon|netflix|carrefour|uber|sncf)\\b",
                            "case_sensitive": False,
                            "weight": 1.0,
                            "entity_extract": {"type": "merchant", "normalize": "uppercase"}
                        },
                        {
                            "regex": "\\b(mes\\s+achats?)\\s+(chez\\s+)?(amazon|netflix|carrefour)\\b",
                            "case_sensitive": False,
                            "weight": 0.95,
                            "extract_group": 3
                        }
                    ],
                    "exact_matches": [
                        "amazon", "netflix", "carrefour", "uber", "sncf",
                        "mes achats amazon", "achats netflix"
                    ],
                    "search_parameters": {
                        "query_type": "text_search_with_filters",
                        "primary_fields": ["merchant_name", "primary_description"]
                    },
                    "examples": [
                        "mes achats Amazon ce mois",
                        "transactions Netflix",
                        "dépenses chez Carrefour"
                    ]
                },
                
                "SEARCH_BY_CATEGORY": {
                    "description": "Search by expense category",
                    "intent_category": "SEARCH",
                    "confidence": 0.85,
                    "priority": 3,
                    "patterns": [
                        {
                            "regex": "\\b(restaurant|resto|repas|dîner|déjeuner)\\b",
                            "case_sensitive": False,
                            "weight": 1.0,
                            "entity_extract": {"type": "category", "value": "restaurant"}
                        },
                        {
                            "regex": "\\b(courses|alimentation|supermarché|épicerie)\\b",
                            "case_sensitive": False,
                            "weight": 1.0,
                            "entity_extract": {"type": "category", "value": "alimentation"}
                        },
                        {
                            "regex": "\\b(transport|essence|carburant|taxi)\\b",
                            "case_sensitive": False,
                            "weight": 1.0,
                            "entity_extract": {"type": "category", "value": "transport"}
                        }
                    ],
                    "exact_matches": [
                        "restaurant", "mes restaurants", "resto",
                        "courses", "mes courses", "alimentation",
                        "transport", "mes transports"
                    ],
                    "examples": [
                        "mes restaurants",
                        "dépenses restaurant janvier",
                        "courses ce mois"
                    ]
                },
                
                "ANALYZE_SPENDING": {
                    "description": "Analyze spending patterns",
                    "intent_category": "ANALYZE",
                    "confidence": 0.88,
                    "priority": 2,
                    "patterns": [
                        {
                            "regex": "\\b(combien|montant|total)\\s+(j'ai\\s+)?dépensé\\b",
                            "case_sensitive": False,
                            "weight": 1.0
                        },
                        {
                            "regex": "\\b(budget|dépenses?)\\s+(total|mensuel|annuel)\\b",
                            "case_sensitive": False,
                            "weight": 0.9
                        }
                    ],
                    "exact_matches": [
                        "combien j'ai dépensé",
                        "budget total",
                        "mes dépenses"
                    ],
                    "search_parameters": {
                        "query_type": "aggregation_search",
                        "aggregations_enabled": True
                    },
                    "examples": [
                        "combien j'ai dépensé ce mois",
                        "budget total restaurant",
                        "analyse de mes dépenses"
                    ]
                }
            },
            "global_settings": {
                "default_limit": 25,
                "boost_settings": {
                    "primary_description": 1.5,
                    "merchant_name": 1.8,
                    "searchable_text": 2.0
                }
            }
        }
        
        # Conversational patterns - COMPLET
        conversational_config = {
            "version": "1.0-integration-test", 
            "intents": {
                "GREETING": {
                    "description": "User greetings",
                    "intent_category": "CONVERSATIONAL",
                    "confidence": 0.98,
                    "priority": 1,
                    "patterns": [
                        {
                            "regex": "^\\s*(bonjour|salut|hello|hi|bonsoir)\\s*[!.]*\\s*$",
                            "case_sensitive": False,
                            "weight": 1.0
                        }
                    ],
                    "exact_matches": ["bonjour", "salut", "hello", "hi", "bonsoir"],
                    "examples": ["bonjour", "salut", "hello"]
                },
                
                "HELP": {
                    "description": "Help requests",
                    "intent_category": "CONVERSATIONAL",
                    "confidence": 0.95,
                    "priority": 2,
                    "patterns": [
                        {
                            "regex": "\\b(aide|help)\\b",
                            "case_sensitive": False,
                            "weight": 1.0
                        }
                    ],
                    "exact_matches": ["aide", "help"],
                    "examples": ["aide", "help"]
                }
            }
        }
        
        # Entity patterns - RÉALISTES
        entity_config = {
            "version": "1.0-integration-test",
            "entity_types": {
                "amount": {
                    "description": "Monetary amounts",
                    "priority": 1,
                    "patterns": [
                        {
                            "regex": "\\b(\\d+(?:[,.]\\d+)?)\\s*(?:euros?|€)\\b",
                            "case_sensitive": False,
                            "extract_group": 1,
                            "normalize": "float",
                            "weight": 1.0
                        }
                    ],
                    "exact_values": {
                        "cent euros": {"value": 100.0, "currency": "EUR"}
                    }
                },
                
                "period": {
                    "description": "Time periods",
                    "priority": 2,
                    "patterns": [
                        {
                            "regex": "\\b(ce\\s+mois|mois\\s+dernier)\\b",
                            "case_sensitive": False,
                            "normalize": "period",
                            "weight": 1.0
                        }
                    ],
                    "exact_values": {
                        "ce mois": "current_month",
                        "mois dernier": "last_month"
                    }
                },
                
                "merchant": {
                    "description": "Merchant names",
                    "priority": 3,
                    "patterns": [
                        {
                            "regex": "\\b(amazon|netflix|carrefour)\\b",
                            "case_sensitive": False,
                            "normalize": "uppercase",
                            "weight": 1.0
                        }
                    ],
                    "exact_values": {
                        "amazon": "AMAZON",
                        "netflix": "NETFLIX"
                    }
                }
            }
        }
        
        # Écriture des fichiers
        with open(temp_path / "financial_patterns.json", 'w', encoding='utf-8') as f:
            json.dump(financial_config, f, indent=2, ensure_ascii=False)
        
        with open(temp_path / "conversational_patterns.json", 'w', encoding='utf-8') as f:
            json.dump(conversational_config, f, indent=2, ensure_ascii=False)
        
        with open(temp_path / "entity_patterns.json", 'w', encoding='utf-8') as f:
            json.dump(entity_config, f, indent=2, ensure_ascii=False)
    
    @pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Integration components not available")
    def test_full_pipeline_integration(self, realistic_rules_dir):
        """Test du pipeline complet RuleLoader -> PatternMatcher -> RuleEngine"""
        print("\n🔄 Testing full pipeline integration...")
        
        # 1. Création du pipeline complet
        loader = create_rule_loader(realistic_rules_dir)
        pattern_matcher = create_pattern_matcher(loader)
        engine = create_rule_engine(loader, pattern_matcher)
        
        # 2. Test de cas réalistes
        test_cases = [
            # (input, expected_intent, expected_entities_count)
            ("bonjour", "GREETING", 0),
            ("mes achats Amazon ce mois", "SEARCH_BY_MERCHANT", 2),  # merchant + period
            ("combien j'ai dépensé en restaurant", "ANALYZE_SPENDING", 1),  # category
            ("mes courses", "SEARCH_BY_CATEGORY", 0),
            ("netflix", "SEARCH_BY_MERCHANT", 1),  # merchant
            ("aide", "HELP", 0)
        ]
        
        results = []
        for text, expected_intent, expected_entities in test_cases:
            match = engine.match_intent(text)
            
            if match:
                entity_count = sum(len(entities) for entities in match.entities.values())
                success = (match.intent == expected_intent)
                results.append({
                    "text": text,
                    "expected": expected_intent,
                    "actual": match.intent,
                    "confidence": match.confidence,
                    "entities_found": entity_count,
                    "entities_expected": expected_entities,
                    "success": success,
                    "execution_time": match.execution_time_ms
                })
                
                print(f"  {'✅' if success else '❌'} '{text}' -> {match.intent} (conf: {match.confidence:.3f}, {entity_count} entities, {match.execution_time_ms:.1f}ms)")
            else:
                results.append({
                    "text": text,
                    "expected": expected_intent,
                    "actual": None,
                    "success": False
                })
                print(f"  ❌ '{text}' -> NO MATCH")
        
        # 3. Validation des résultats
        successful = sum(1 for r in results if r["success"])
        success_rate = successful / len(results)
        avg_time = sum(r.get("execution_time", 0) for r in results if r.get("execution_time")) / len([r for r in results if r.get("execution_time")])
        
        print(f"\n📊 Pipeline Results: {successful}/{len(results)} successful ({success_rate:.1%})")
        print(f"⚡ Average execution time: {avg_time:.2f}ms")
        
        # Assertions
        assert success_rate >= 0.8, f"Success rate {success_rate:.1%} below 80% threshold"
        assert avg_time < 50, f"Average execution time {avg_time:.1f}ms above 50ms threshold"
    
    @pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Integration components not available")
    def test_entity_extraction_integration(self, realistic_rules_dir):
        """Test d'intégration extraction d'entités avec pipeline complet"""
        print("\n🔍 Testing entity extraction integration...")
        
        loader = create_rule_loader(realistic_rules_dir)
        pattern_matcher = create_pattern_matcher(loader)
        
        # Test cas complexes avec entités multiples
        complex_cases = [
            "j'ai dépensé 150 euros chez Amazon ce mois",
            "mes achats Netflix mois dernier",
            "budget restaurant 50 euros",
            "courses Carrefour ce mois 75€"
        ]
        
        for text in complex_cases:
            result = pattern_matcher.extract_entities(text)
            
            print(f"  📝 '{text}':")
            print(f"     → {result.total_matches} entities in {result.extraction_time_ms:.1f}ms")
            
            for entity_type, entities in result.entities.items():
                for entity in entities:
                    print(f"     → {entity_type}: '{entity.raw_value}' -> {entity.normalized_value}")
            
            # Validation performance
            assert result.extraction_time_ms < 30, f"Entity extraction took {result.extraction_time_ms:.1f}ms (> 30ms)"
    
    @pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Integration components not available")
    def test_package_functions_integration(self):
        """Test des fonctions package-level avec composants réels"""
        print("\n📦 Testing package-level functions...")
        
        # Test quick_detect_intent avec différents patterns
        test_queries = [
            "bonjour",
            "aide",
            "hello",
            "help"
        ]
        
        detected_count = 0
        for query in test_queries:
            match = quick_detect_intent(query, confidence_threshold=0.7)
            if match:
                detected_count += 1
                print(f"  ✅ '{query}' -> {match.intent} (confidence: {match.confidence:.3f})")
            else:
                print(f"  ⚠️  '{query}' -> NO MATCH")
        
        print(f"  📊 Detection rate: {detected_count}/{len(test_queries)} ({detected_count/len(test_queries):.1%})")
        
        # Test extract_entities_quick
        entity_text = "50 euros Amazon ce mois"
        entities = extract_entities_quick(entity_text)
        print(f"  🔍 Entity extraction: {entities.total_matches} entities from '{entity_text}'")
        
        # Test get_package_info
        info = get_package_info()
        assert "package" in info
        assert "rules" in info
        print(f"  📊 Package info: {info['rules']['financial_count']} financial, {info['rules']['conversational_count']} conversational rules")
        
        # Test get_performance_report
        try:
            report = get_performance_report()
            if "performance_stats" in report:
                cache_rate = report["performance_stats"]["cache_stats"]["hit_rate_percent"]
                print(f"  💾 Performance: {cache_rate:.1f}% cache hit rate")
            else:
                print("  ⚠️  Performance report not available (insufficient data)")
        except Exception as e:
            print(f"  ⚠️  Performance report failed: {e}")
    
    @pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Integration components not available")
    def test_performance_integration(self, realistic_rules_dir):
        """Test de performance end-to-end"""
        print("\n⚡ Testing end-to-end performance...")
        
        loader = create_rule_loader(realistic_rules_dir)
        engine = create_rule_engine(loader)
        
        # Warm-up pour initialiser les caches
        engine.match_intent("bonjour")
        engine.match_intent("aide")
        
        # Test de performance avec volume
        test_queries = [
            "bonjour", "salut", "hello",
            "aide", "help",
            "mes achats amazon", "netflix", "carrefour",
            "mes restaurants", "courses", "transport",
            "combien j'ai dépensé", "budget total"
        ] * 10  # 130 requêtes total
        
        start_time = time.time()
        successful_matches = 0
        total_execution_time = 0
        
        for query in test_queries:
            match = engine.match_intent(query)
            if match:
                successful_matches += 1
                total_execution_time += match.execution_time_ms
        
        end_time = time.time()
        
        # Calculs de performance
        total_time_ms = (end_time - start_time) * 1000
        avg_time_per_query = total_time_ms / len(test_queries)
        avg_engine_time = total_execution_time / successful_matches if successful_matches > 0 else 0
        queries_per_second = len(test_queries) / (total_time_ms / 1000)
        
        print(f"  📊 Volume test: {len(test_queries)} queries")
        print(f"  ✅ Successful matches: {successful_matches} ({successful_matches/len(test_queries):.1%})")
        print(f"  ⚡ Total time: {total_time_ms:.1f}ms")
        print(f"  ⚡ Avg time per query: {avg_time_per_query:.2f}ms")
        print(f"  ⚡ Avg engine time: {avg_engine_time:.2f}ms")
        print(f"  🚀 Throughput: {queries_per_second:.1f} queries/second")
        
        # Assertions de performance
        assert avg_time_per_query < 10, f"Average query time {avg_time_per_query:.2f}ms too high"
        assert queries_per_second > 50, f"Throughput {queries_per_second:.1f} QPS too low"
        
        # Test statistiques du moteur
        stats = engine.get_performance_stats()
        cache_hit_rate = stats["cache_stats"]["hit_rate_percent"]
        print(f"  💾 Cache hit rate: {cache_hit_rate:.1f}%")
        
        assert cache_hit_rate > 70, f"Cache hit rate {cache_hit_rate:.1f}% too low"
    
    @pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Integration components not available")
    def test_error_resilience_integration(self, realistic_rules_dir):
        """Test de résilience aux erreurs du système intégré"""
        print("\n🛡️  Testing error resilience...")
        
        engine = create_rule_engine(create_rule_loader(realistic_rules_dir))
        
        # Test cas d'erreur potentiels
        error_cases = [
            "",  # Chaîne vide
            "   ",  # Espaces uniquement
            "a" * 1000,  # Texte très long
            "caractères spéciaux !@#$%^&*()",  # Caractères spéciaux
            "🚀🎯💡",  # Emojis
            "texto con acentos àáâãäçèéêë",  # Accents
        ]
        
        handled_gracefully = 0
        for test_case in error_cases:
            try:
                match = engine.match_intent(test_case)
                handled_gracefully += 1
                status = "MATCH" if match else "NO_MATCH"
                print(f"  ✅ '{test_case[:30]}...' -> {status}")
            except Exception as e:
                print(f"  ❌ '{test_case[:30]}...' -> ERROR: {e}")
        
        resilience_rate = handled_gracefully / len(error_cases)
        print(f"  🛡️  Resilience: {handled_gracefully}/{len(error_cases)} ({resilience_rate:.1%})")
        
        assert resilience_rate >= 0.9, f"Error resilience {resilience_rate:.1%} below 90%"


def run_integration_tests():
    """Exécute tous les tests d'intégration avec rapport détaillé"""
    
    if not INTEGRATION_AVAILABLE:
        print("❌ Cannot run integration tests - components not available")
        return 1
    
    print("🧪 Starting comprehensive integration tests...")
    print("=" * 70)
    
    # Exécution des tests avec rapport détaillé  
    result = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--color=yes",
        "-s"  # Afficher les prints
    ])
    
    print("=" * 70)
    if result == 0:
        print("🎉 All integration tests passed successfully!")
        print("✅ Architecture fully validated:")
        print("   • RuleLoader ✅")
        print("   • PatternMatcher ✅")  
        print("   • RuleEngine ✅")
        print("   • Package functions ✅")
        print("   • Performance ✅")
        print("   • Error resilience ✅")
    else:
        print("❌ Some integration tests failed")
    
    return result


if __name__ == "__main__":
    exit_code = run_integration_tests()
    exit(exit_code)