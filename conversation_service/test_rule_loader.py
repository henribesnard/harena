"""
🧪 Tests pour RuleLoader - Validation complète du chargeur de règles

Ce module teste toutes les fonctionnalités du RuleLoader :
- Chargement des fichiers JSON
- Parsing des règles et patterns
- Validation des configurations
- Gestion des erreurs
- Performance des règles
"""
import os
import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, mock_open
from typing import Dict, List

# Import du module à tester
import sys
sys.path.append(str(Path(__file__).parent.parent))

from intent_rules.rule_loader import (
    RuleLoader,
    IntentRule,
    PatternRule,
    EntityPattern,
    IntentCategory,
    RuleMatch,
    create_rule_loader
)


class TestRuleLoader:
    """Tests complets pour la classe RuleLoader"""
    
    @pytest.fixture
    def temp_rules_dir(self):
        """Fixture créant un répertoire temporaire avec des fichiers de test"""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        # Création des fichiers JSON de test
        self._create_test_files(temp_path)
        
        yield temp_path
        
        # Nettoyage
        shutil.rmtree(temp_dir)
    
    def _create_test_files(self, temp_path: Path):
        """Crée des fichiers JSON de test dans le répertoire temporaire"""
        
        # Financial patterns test - VALIDES
        financial_config = {
            "version": "1.0-test",
            "last_updated": "2025-01-30",
            "intents": {
                "SEARCH_BY_MERCHANT": {
                    "description": "Test merchant search",
                    "intent_category": "SEARCH",
                    "confidence": 0.90,
                    "priority": 2,
                    "patterns": [
                        {
                            "regex": "\\b(amazon|netflix)\\b",
                            "case_sensitive": False,
                            "weight": 1.0,
                            "entity_extract": {"type": "merchant", "normalize": "uppercase"}
                        }
                    ],
                    "exact_matches": ["amazon", "netflix"],
                    "search_parameters": {
                        "query_type": "text_search_with_filters",
                        "primary_fields": ["merchant_name"]
                    },
                    "examples": ["mes achats Amazon"]
                }
            },
            "global_settings": {
                "default_limit": 25
            }
        }
        
        # Conversational patterns test
        conversational_config = {
            "version": "1.0-test",
            "intents": {
                "GREETING": {
                    "description": "Test greeting",
                    "intent_category": "CONVERSATIONAL",
                    "confidence": 0.98,
                    "priority": 1,
                    "patterns": [
                        {
                            "regex": "^\\s*(bonjour|salut)\\s*$",
                            "case_sensitive": False,
                            "weight": 1.0
                        }
                    ],
                    "exact_matches": ["bonjour", "salut"],
                    "examples": ["bonjour"]
                }
            }
        }
        
        # Entity patterns test
        entity_config = {
            "version": "1.0-test",
            "entity_types": {
                "amount": {
                    "description": "Test amounts",
                    "patterns": [
                        {
                            "regex": "\\b(\\d+)\\s*euros?\\b",
                            "case_sensitive": False,
                            "extract_group": 1,
                            "normalize": "float",
                            "weight": 1.0
                        }
                    ],
                    "exact_values": {
                        "cent euros": {"value": 100.0, "currency": "EUR"}
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
    
    def test_rule_loader_initialization(self, temp_rules_dir):
        """Test l'initialisation correcte du RuleLoader"""
        loader = RuleLoader(temp_rules_dir)
        
        # Vérifications de base
        assert loader.rules_dir == temp_rules_dir
        assert len(loader.financial_rules) > 0
        assert len(loader.conversational_rules) > 0
        assert len(loader.entity_patterns) > 0
    
    def test_financial_rules_loading(self, temp_rules_dir):
        """Test le chargement des règles financières"""
        loader = RuleLoader(temp_rules_dir)
        
        # Vérification règle SEARCH_BY_MERCHANT
        assert "SEARCH_BY_MERCHANT" in loader.financial_rules
        
        rule = loader.financial_rules["SEARCH_BY_MERCHANT"]
        assert rule.intent == "SEARCH_BY_MERCHANT"
        assert rule.intent_category == IntentCategory.SEARCH
        assert rule.confidence == 0.90
        assert rule.priority == 2
        assert len(rule.patterns) == 1
        assert "amazon" in rule.exact_matches
        assert rule.search_parameters is not None
    
    def test_conversational_rules_loading(self, temp_rules_dir):
        """Test le chargement des règles conversationnelles"""
        loader = RuleLoader(temp_rules_dir)
        
        # Vérification règle GREETING
        assert "GREETING" in loader.conversational_rules
        
        rule = loader.conversational_rules["GREETING"]
        assert rule.intent == "GREETING"
        assert rule.intent_category == IntentCategory.CONVERSATIONAL
        assert rule.confidence == 0.98
        assert "bonjour" in rule.exact_matches
    
    def test_entity_patterns_loading(self, temp_rules_dir):
        """Test le chargement des patterns d'entités"""
        loader = RuleLoader(temp_rules_dir)
        
        # Vérification patterns amount
        assert "amount" in loader.entity_patterns
        
        patterns = loader.entity_patterns["amount"]
        assert len(patterns) == 1
        assert patterns[0].entity_type == "amount"
        assert patterns[0].extract_group == 1
        assert patterns[0].normalize == "float"
    
    def test_invalid_confidence_validation(self, temp_rules_dir):
        """Test la validation des valeurs de confidence invalides"""
        # Création d'un fichier avec confidence invalide
        invalid_config = {
            "version": "1.0-test",
            "intents": {
                "INVALID_CONFIDENCE": {
                    "description": "Test invalid confidence",
                    "intent_category": "SEARCH", 
                    "confidence": 1.5,  # Invalid: > 1.0
                    "priority": 1,
                    "patterns": [],
                    "exact_matches": ["test"]
                }
            }
        }
        
        with open(temp_rules_dir / "financial_patterns.json", 'w', encoding='utf-8') as f:
            json.dump(invalid_config, f)
        
        # Cette règle devrait lever une exception à cause de confidence > 1.0
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            loader = RuleLoader(temp_rules_dir)
    
    def test_get_methods(self, temp_rules_dir):
        """Test les méthodes getter du RuleLoader"""
        # Modification du fichier pour éviter l'erreur de validation
        financial_config = {
            "version": "1.0-test",
            "intents": {
                "SEARCH_BY_MERCHANT": {
                    "description": "Test merchant search",
                    "intent_category": "SEARCH",
                    "confidence": 0.90,
                    "priority": 2,
                    "patterns": [
                        {
                            "regex": "\\b(amazon)\\b",
                            "case_sensitive": False,
                            "weight": 1.0
                        }
                    ],
                    "exact_matches": ["amazon"]
                }
            }
        }
        
        with open(temp_rules_dir / "financial_patterns.json", 'w', encoding='utf-8') as f:
            json.dump(financial_config, f)
        
        loader = RuleLoader(temp_rules_dir)
        
        # Test get_financial_rules
        financial_rules = loader.get_financial_rules()
        assert len(financial_rules) == 1
        assert "SEARCH_BY_MERCHANT" in financial_rules
        
        # Test get_conversational_rules
        conversational_rules = loader.get_conversational_rules()
        assert len(conversational_rules) == 1
        assert "GREETING" in conversational_rules
        
        # Test get_entity_patterns
        entity_patterns = loader.get_entity_patterns()
        assert "amount" in entity_patterns
        
        # Test get_entity_patterns avec type spécifique
        amount_patterns = loader.get_entity_patterns("amount")
        assert len(amount_patterns) == 1
        
        # Test get_all_rules
        all_rules = loader.get_all_rules()
        assert len(all_rules) == 2  # 1 financial + 1 conversational
    
    def test_get_rules_by_category(self, temp_rules_dir):
        """Test la récupération des règles par catégorie"""
        # Modification du fichier pour éviter l'erreur
        financial_config = {
            "version": "1.0-test",
            "intents": {
                "SEARCH_BY_MERCHANT": {
                    "description": "Test merchant search",
                    "intent_category": "SEARCH",
                    "confidence": 0.90,
                    "priority": 2,
                    "patterns": [],
                    "exact_matches": ["amazon"]
                }
            }
        }
        
        with open(temp_rules_dir / "financial_patterns.json", 'w', encoding='utf-8') as f:
            json.dump(financial_config, f)
        
        loader = RuleLoader(temp_rules_dir)
        
        # Test récupération par catégorie SEARCH
        search_rules = loader.get_rules_by_category(IntentCategory.SEARCH)
        assert len(search_rules) == 1
        assert "SEARCH_BY_MERCHANT" in search_rules
        
        # Test récupération par catégorie CONVERSATIONAL
        conv_rules = loader.get_rules_by_category(IntentCategory.CONVERSATIONAL)
        assert len(conv_rules) == 1
        assert "GREETING" in conv_rules
    
    def test_get_rules_by_priority(self, temp_rules_dir):
        """Test la récupération des règles par priorité"""
        # Modification du fichier
        financial_config = {
            "version": "1.0-test",
            "intents": {
                "HIGH_PRIORITY": {
                    "description": "High priority rule",
                    "intent_category": "SEARCH",
                    "confidence": 0.90,
                    "priority": 1,
                    "patterns": [],
                    "exact_matches": ["test"]
                },
                "LOW_PRIORITY": {
                    "description": "Low priority rule",
                    "intent_category": "SEARCH",
                    "confidence": 0.80,
                    "priority": 5,
                    "patterns": [],
                    "exact_matches": ["test2"]
                }
            }
        }
        
        with open(temp_rules_dir / "financial_patterns.json", 'w', encoding='utf-8') as f:
            json.dump(financial_config, f)
        
        loader = RuleLoader(temp_rules_dir)
        
        # Test récupération par priorité
        high_priority_rules = loader.get_rules_by_priority(1, 2)
        assert len(high_priority_rules) == 2  # HIGH_PRIORITY + GREETING (priority 1)
        
        # Vérification du tri par priorité
        priorities = [rule.priority for rule in high_priority_rules]
        assert priorities == sorted(priorities)
    
    def test_validation_rules(self, temp_rules_dir):
        """Test la validation des règles"""
        # Création d'un fichier avec des erreurs mais qui peut être chargé
        bad_financial_config = {
            "version": "1.0-test",
            "intents": {
                "NO_PATTERNS": {
                    "description": "Rule with no patterns", 
                    "intent_category": "SEARCH",
                    "confidence": 0.80,
                    "priority": 1,
                    "patterns": [],
                    "exact_matches": []  # Aucun pattern ni exact match
                },
                "VALID_RULE": {
                    "description": "Valid rule to avoid complete failure",
                    "intent_category": "SEARCH",
                    "confidence": 0.80,
                    "priority": 1,
                    "patterns": [
                        {
                            "regex": "\\btest\\b",  # Regex valide
                            "weight": 1.0
                        }
                    ],
                    "exact_matches": ["test"]
                }
            }
        }
        
        with open(temp_rules_dir / "financial_patterns.json", 'w', encoding='utf-8') as f:
            json.dump(bad_financial_config, f)
        
        loader = RuleLoader(temp_rules_dir)
        
        # Test de validation
        errors = loader.validate_rules()
        
        # Vérification des erreurs détectées
        assert len(errors['financial']) > 0
        
        # Recherche de l'erreur spécifique
        error_messages = ' '.join(errors['financial'])
        assert "No patterns or exact matches defined" in error_messages
    
    def test_version_info_and_settings(self, temp_rules_dir):
        """Test la récupération des informations de version et paramètres"""
        # Modification du fichier pour éviter l'erreur
        financial_config = {
            "version": "1.0-test",
            "last_updated": "2025-01-30",
            "intents": {
                "TEST_RULE": {
                    "description": "Test rule",
                    "intent_category": "SEARCH",
                    "confidence": 0.80,
                    "priority": 1,
                    "patterns": [],
                    "exact_matches": ["test"]
                }
            },
            "global_settings": {
                "default_limit": 25,
                "test_setting": "test_value"
            }
        }
        
        with open(temp_rules_dir / "financial_patterns.json", 'w', encoding='utf-8') as f:
            json.dump(financial_config, f)
        
        loader = RuleLoader(temp_rules_dir)
        
        # Test version info
        version_info = loader.get_version_info()
        assert "financial" in version_info
        assert version_info["financial"]["version"] == "1.0-test"
        assert version_info["financial"]["last_updated"] == "2025-01-30"
        
        # Test global settings
        settings = loader.get_global_settings()
        assert settings["default_limit"] == 25
        assert settings["test_setting"] == "test_value"
    
    def test_export_rules_summary(self, temp_rules_dir):
        """Test l'export du résumé des règles"""
        # Modification du fichier
        financial_config = {
            "version": "1.0-test",
            "intents": {
                "TEST_RULE": {
                    "description": "Test rule",
                    "intent_category": "SEARCH", 
                    "confidence": 0.80,
                    "priority": 1,
                    "patterns": [
                        {"regex": "test", "weight": 1.0}
                    ],
                    "exact_matches": ["test"],
                    "examples": ["exemple 1", "exemple 2"]
                }
            }
        }
        
        with open(temp_rules_dir / "financial_patterns.json", 'w', encoding='utf-8') as f:
            json.dump(financial_config, f)
        
        loader = RuleLoader(temp_rules_dir)
        
        # Test export summary
        summary = loader.export_rules_summary()
        
        # Vérifications
        assert "version_info" in summary
        assert "financial_rules_count" in summary
        assert "conversational_rules_count" in summary
        assert "entity_types_count" in summary
        assert "financial_rules" in summary
        assert "conversational_rules" in summary
        assert "entity_patterns" in summary
        
        # Vérification détails règle financière
        assert "TEST_RULE" in summary["financial_rules"]
        test_rule_summary = summary["financial_rules"]["TEST_RULE"]
        assert test_rule_summary["category"] == "SEARCH"
        assert test_rule_summary["confidence"] == 0.80
        assert test_rule_summary["priority"] == 1
        assert test_rule_summary["patterns_count"] == 1
        assert test_rule_summary["exact_matches_count"] == 1
        assert len(test_rule_summary["examples"]) <= 3
    
    def test_reload_rules(self, temp_rules_dir):
        """Test le rechargement des règles"""
        loader = RuleLoader(temp_rules_dir)
        
        # Comptage initial (avec la règle invalide, cela devrait échouer)
        # Modifions d'abord le fichier
        financial_config = {
            "version": "1.0-test",
            "intents": {
                "INITIAL_RULE": {
                    "description": "Initial rule",
                    "intent_category": "SEARCH",
                    "confidence": 0.80,
                    "priority": 1,
                    "patterns": [],
                    "exact_matches": ["initial"]
                }
            }
        }
        
        with open(temp_rules_dir / "financial_patterns.json", 'w', encoding='utf-8') as f:
            json.dump(financial_config, f)
        
        # Nouveau loader avec fichier corrigé
        loader = RuleLoader(temp_rules_dir)
        initial_count = len(loader.financial_rules)
        
        # Modification du fichier pour ajouter une règle
        financial_config["intents"]["NEW_RULE"] = {
            "description": "New rule",
            "intent_category": "SEARCH",
            "confidence": 0.85,
            "priority": 2,
            "patterns": [],
            "exact_matches": ["new"]
        }
        
        with open(temp_rules_dir / "financial_patterns.json", 'w', encoding='utf-8') as f:
            json.dump(financial_config, f)
        
        # Rechargement
        loader.reload_rules()
        
        # Vérification
        new_count = len(loader.financial_rules)
        assert new_count == initial_count + 1
        assert "NEW_RULE" in loader.financial_rules
    
    def test_missing_files_handling(self):
        """Test la gestion des fichiers manquants"""
        # Répertoire vide
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        try:
            # Le loader devrait fonctionner même sans fichiers
            loader = RuleLoader(temp_path)
            
            # Vérifications
            assert len(loader.financial_rules) == 0
            assert len(loader.conversational_rules) == 0
            assert len(loader.entity_patterns) == 0
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_factory_function(self, temp_rules_dir):
        """Test la fonction factory create_rule_loader"""
        loader = create_rule_loader(temp_rules_dir)
        
        assert isinstance(loader, RuleLoader)
        assert loader.rules_dir == temp_rules_dir
    
    def test_pattern_rule_validation(self):
        """Test la validation des PatternRule"""
        # Test création PatternRule valide
        import re
        pattern = PatternRule(
            regex=re.compile("test"),
            weight=1.0,
            entity_extract={"type": "test"},
            extract_group=1
        )
        
        assert pattern.regex.pattern == "test"
        assert pattern.weight == 1.0
        assert pattern.entity_extract == {"type": "test"}
        assert pattern.extract_group == 1
    
    def test_intent_rule_validation(self):
        """Test la validation des IntentRule"""
        import re
        
        # Test règle valide
        rule = IntentRule(
            intent="TEST_INTENT",
            description="Test intent",
            intent_category=IntentCategory.SEARCH,
            confidence=0.8,
            priority=1,
            patterns=[],
            exact_matches=set(["test"])
        )
        
        assert rule.intent == "TEST_INTENT"
        assert rule.confidence == 0.8
        assert rule.priority == 1
        
        # Test confidence invalide
        with pytest.raises(ValueError):
            IntentRule(
                intent="INVALID",
                description="Invalid",
                intent_category=IntentCategory.SEARCH,
                confidence=1.5,  # > 1.0
                priority=1,
                patterns=[],
                exact_matches=set()
            )
        
        # Test priority invalide
        with pytest.raises(ValueError):
            IntentRule(
                intent="INVALID",
                description="Invalid", 
                intent_category=IntentCategory.SEARCH,
                confidence=0.8,
                priority=0,  # < 1
                patterns=[],
                exact_matches=set()
            )


    def test_regex_validation_separate(self):
        """Test séparé pour la validation des regex invalides"""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        try:
            # Configuration avec regex invalide
            bad_regex_config = {
                "version": "1.0-test",
                "intents": {
                    "BAD_REGEX": {
                        "description": "Rule with invalid regex",
                        "intent_category": "SEARCH",
                        "confidence": 0.80,
                        "priority": 1,
                        "patterns": [
                            {
                                "regex": "[invalid regex",  # Regex invalide
                                "weight": 1.0
                            }
                        ],
                        "exact_matches": []
                    }
                }
            }
            
            # Fichiers minimaux pour éviter autres erreurs
            conversational_config = {
                "version": "1.0-test",
                "intents": {
                    "GREETING": {
                        "description": "Test greeting",
                        "intent_category": "CONVERSATIONAL",
                        "confidence": 0.98,
                        "priority": 1,
                        "patterns": [],
                        "exact_matches": ["bonjour"]
                    }
                }
            }
            
            entity_config = {
                "version": "1.0-test",
                "entity_types": {
                    "amount": {
                        "description": "Test amounts",
                        "patterns": []
                    }
                }
            }
            
            with open(temp_path / "financial_patterns.json", 'w', encoding='utf-8') as f:
                json.dump(bad_regex_config, f)
            
            with open(temp_path / "conversational_patterns.json", 'w', encoding='utf-8') as f:
                json.dump(conversational_config, f)
                
            with open(temp_path / "entity_patterns.json", 'w', encoding='utf-8') as f:
                json.dump(entity_config, f)
            
            # Le chargement devrait échouer à cause de la regex invalide
            with pytest.raises(Exception):  # re.error ou autre exception
                loader = RuleLoader(temp_path)
                
        finally:
            shutil.rmtree(temp_dir)
    """Tests de performance du RuleLoader"""
    
    @pytest.fixture
    def temp_rules_dir_perf(self):
        """Fixture dédiée aux tests de performance"""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        yield temp_path
        
        shutil.rmtree(temp_dir)
    
    def test_loading_performance(self, temp_rules_dir_perf):
        """Test la performance de chargement des règles"""
        import time
        
        # Configuration de base pour éviter les erreurs
        financial_config = {
            "version": "1.0-test",
            "intents": {}
        }
        
        # Création de nombreuses règles
        for i in range(100):
            financial_config["intents"][f"RULE_{i}"] = {
                "description": f"Rule {i}",
                "intent_category": "SEARCH",
                "confidence": 0.8,
                "priority": 1,
                "patterns": [
                    {
                        "regex": f"test{i}",
                        "weight": 1.0
                    }
                ],
                "exact_matches": [f"test{i}"]
            }
        
        # Fichiers conversationnels et entités minimaux pour éviter les erreurs
        conversational_config = {
            "version": "1.0-test",
            "intents": {
                "GREETING": {
                    "description": "Test greeting",
                    "intent_category": "CONVERSATIONAL",
                    "confidence": 0.98,
                    "priority": 1,
                    "patterns": [],
                    "exact_matches": ["bonjour"]
                }
            }
        }
        
        entity_config = {
            "version": "1.0-test",
            "entity_types": {
                "amount": {
                    "description": "Test amounts",
                    "patterns": [
                        {
                            "regex": "\\b(\\d+)\\s*euros?\\b",
                            "extract_group": 1,
                            "weight": 1.0
                        }
                    ]
                }
            }
        }
        
        with open(temp_rules_dir_perf / "financial_patterns.json", 'w', encoding='utf-8') as f:
            json.dump(financial_config, f)
        
        with open(temp_rules_dir_perf / "conversational_patterns.json", 'w', encoding='utf-8') as f:
            json.dump(conversational_config, f)
            
        with open(temp_rules_dir_perf / "entity_patterns.json", 'w', encoding='utf-8') as f:
            json.dump(entity_config, f)
        
        # Test de performance
        start_time = time.time()
        loader = RuleLoader(temp_rules_dir_perf)
        loading_time = time.time() - start_time
        
        # Vérification
        assert len(loader.financial_rules) == 100
        assert loading_time < 1.0  # Chargement en moins d'1 seconde
        
        print(f"Loaded 100 rules in {loading_time:.3f} seconds")


# Fonction utilitaire pour les tests
def run_all_tests():
    """Exécute tous les tests du RuleLoader"""
    import pytest
    
    # Exécution des tests avec rapport détaillé
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--color=yes"
    ])


if __name__ == "__main__":
    # Exécution directe des tests
    run_all_tests()