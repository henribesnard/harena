"""
Tests unitaires pour la validation des configurations
"""

import pytest
import yaml
import asyncio
from pathlib import Path
from typing import Dict, Any

import sys
from pathlib import Path
# Ajouter le répertoire parent pour imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from conversation_service.config.config_validator import (
    ConfigValidator, ValidationLevel, validate_full_configuration
)
from conversation_service.config.settings import ConfigManager

class TestConfigValidation:
    """Tests de validation des configurations"""
    
    @pytest.fixture
    def validator(self):
        return ConfigValidator()
    
    @pytest.fixture
    def config_dir(self):
        return Path(__file__).parent.parent.parent.parent / "config"
    
    @pytest.fixture
    def sample_intentions_config(self):
        return {
            "metadata": {"version": "2.0.0"},
            "intent_groups": {
                "TRANSACTION_SEARCH": {
                    "description": "Test group",
                    "patterns_generiques": {
                        "simple_search": {
                            "few_shot_examples": [
                                {
                                    "input": "mes transactions de janvier",
                                    "output": {
                                        "intent_group": "TRANSACTION_SEARCH",
                                        "intent_subtype": "by_date", 
                                        "entities": {"date": {"gte": "2025-01-01T00:00:00Z"}},
                                        "query_strategy": "date_filter"
                                    }
                                }
                            ]
                        }
                    }
                }
            },
            "classification_strategies": {},
            "query_strategies": {}
        }
    
    @pytest.fixture
    def sample_entities_config(self):
        return {
            "metadata": {"version": "2.0.0"},
            "search_service_fields": {
                "temporal_fields": ["date", "timestamp"],
                "monetary_fields": ["amount", "amount_abs"],
                "text_fields": ["merchant_name", "searchable_text"]
            },
            "temporal_entities": {
                "periode_temporelle": {
                    "target_field": "date",
                    "few_shot_examples": []
                }
            },
            "monetary_entities": {},
            "textual_entities": {},
            "analytical_entities": {}
        }
    
    @pytest.mark.asyncio
    async def test_validate_intentions_basic_structure(self, validator, sample_intentions_config):
        """Test validation structure de base des intentions"""
        results = await validator.validate_intentions_config(sample_intentions_config)
        
        # Ne devrait pas y avoir d'erreurs critiques
        critical_errors = [r for r in results if r.level == ValidationLevel.CRITICAL]
        assert len(critical_errors) == 0, f"Critical errors found: {[r.message for r in critical_errors]}"
    
    @pytest.mark.asyncio 
    async def test_validate_entities_basic_structure(self, validator, sample_entities_config):
        """Test validation structure de base des entités"""
        results = await validator.validate_entities_config(sample_entities_config)
        
        # Ne devrait pas y avoir d'erreurs critiques
        critical_errors = [r for r in results if r.level == ValidationLevel.CRITICAL]
        assert len(critical_errors) == 0, f"Critical errors found: {[r.message for r in critical_errors]}"
    
    @pytest.mark.asyncio
    async def test_validate_missing_required_fields(self, validator):
        """Test validation avec champs obligatoires manquants"""
        invalid_config = {"metadata": {"version": "2.0.0"}}
        
        results = await validator.validate_intentions_config(invalid_config)
        
        # Devrait y avoir des erreurs pour les champs manquants
        assert len(results) > 0
        critical_errors = [r for r in results if r.level == ValidationLevel.CRITICAL]
        assert len(critical_errors) > 0
    
    @pytest.mark.asyncio
    async def test_validate_unknown_elasticsearch_field(self, validator):
        """Test validation avec champ Elasticsearch inexistant"""
        invalid_entities = {
            "search_service_fields": {
                "temporal_fields": ["unknown_field", "date"]
            }
        }
        
        results = await validator.validate_entities_config(invalid_entities)
        
        # Devrait détecter le champ inexistant
        field_errors = [r for r in results if "unknown_field" in r.message]
        assert len(field_errors) > 0
    
    @pytest.mark.asyncio
    async def test_validate_temporal_format(self, validator):
        """Test validation du format des dates"""
        entities_config = {
            "temporal_entities": {
                "test_entity": {
                    "few_shot_examples": [
                        {
                            "output": {
                                "date": {"gte": "invalid-date-format"}
                            }
                        }
                    ]
                }
            }
        }
        
        results = await validator.validate_entities_config(entities_config)
        
        # Devrait détecter le format de date invalide
        date_format_errors = [r for r in results if "ISO 8601" in r.message]
        assert len(date_format_errors) > 0
    
    @pytest.mark.asyncio
    async def test_cross_compatibility_validation(self, validator, sample_intentions_config, sample_entities_config):
        """Test validation croisée intentions-entités"""
        # Ajouter un champ inconnu dans les intentions
        sample_intentions_config["intent_groups"]["TRANSACTION_SEARCH"]["patterns_generiques"]["simple_search"]["few_shot_examples"][0]["output"]["entities"]["unknown_field"] = "test"
        
        results = await validator.validate_cross_compatibility(sample_intentions_config, sample_entities_config)
        
        # Devrait détecter le champ manquant
        compatibility_errors = [r for r in results if r.category == "cross_compatibility"]
        assert len(compatibility_errors) > 0
    
    @pytest.mark.asyncio
    async def test_search_service_compatibility(self, validator, sample_entities_config):
        """Test compatibilité avec search_service"""
        results = await validator.validate_search_service_compatibility(sample_entities_config)
        
        # Tous les champs dans l'exemple devraient être valides
        critical_errors = [r for r in results if r.level == ValidationLevel.CRITICAL]
        assert len(critical_errors) == 0
    
    @pytest.mark.asyncio
    async def test_full_configuration_validation(self, sample_intentions_config, sample_entities_config):
        """Test validation complète"""
        is_valid, results = await validate_full_configuration(
            sample_intentions_config, 
            sample_entities_config
        )
        
        # La configuration de test devrait être valide
        assert is_valid, f"Configuration should be valid. Errors: {[r.message for r in results if r.level == ValidationLevel.CRITICAL]}"
        
        # Afficher les avertissements pour information
        warnings = [r for r in results if r.level == ValidationLevel.WARNING]
        if warnings:
            print("Warnings found:")
            for warning in warnings:
                print(f"  - {warning.message}")


class TestConfigManager:
    """Tests du gestionnaire de configuration"""
    
    @pytest.fixture
    def config_dir(self):
        return Path(__file__).parent.parent.parent.parent / "config"
    
    @pytest.mark.asyncio
    async def test_config_manager_initialization(self, config_dir):
        """Test initialisation du gestionnaire"""
        manager = ConfigManager(config_dir)
        success = await manager.initialize()
        
        assert success, "Config manager should initialize successfully"
        assert manager.state.is_valid, f"Configuration should be valid. Errors: {manager.state.validation_errors}"
    
    @pytest.mark.asyncio
    async def test_load_intentions_config(self, config_dir):
        """Test chargement des intentions"""
        manager = ConfigManager(config_dir)
        await manager.initialize()
        
        intentions = manager.get_intentions_config()
        assert intentions is not None
        assert "intent_groups" in intentions
        assert "TRANSACTION_SEARCH" in intentions["intent_groups"]
    
    @pytest.mark.asyncio
    async def test_load_entities_config(self, config_dir):
        """Test chargement des entités"""
        manager = ConfigManager(config_dir)
        await manager.initialize()
        
        entities = manager.get_entities_config()
        assert entities is not None
        assert "search_service_fields" in entities
        assert "temporal_entities" in entities
    
    @pytest.mark.asyncio
    async def test_configuration_status(self, config_dir):
        """Test statut de la configuration"""
        manager = ConfigManager(config_dir)
        await manager.initialize()
        
        status = manager.get_configuration_status()
        assert "is_valid" in status
        assert "version" in status
        assert "loaded_at" in status
        assert "config_files_status" in status
    
    @pytest.mark.asyncio
    async def test_missing_config_files(self, tmp_path):
        """Test comportement avec fichiers manquants"""
        manager = ConfigManager(tmp_path)
        success = await manager.initialize()
        
        # Devrait échouer avec fichiers manquants
        assert not success or not manager.state.is_valid


class TestRealConfigFiles:
    """Tests avec les vrais fichiers de configuration"""
    
    @pytest.fixture
    def config_dir(self):
        return Path(__file__).parent.parent.parent.parent / "config"
    
    @pytest.mark.asyncio
    async def test_real_intentions_yaml_valid(self, config_dir):
        """Test que le vrai fichier intentions_v2.yaml est valide"""
        intentions_file = config_dir / "intentions_v2.yaml"
        
        if not intentions_file.exists():
            pytest.skip("intentions_v2.yaml not found")
        
        with open(intentions_file, 'r', encoding='utf-8') as f:
            intentions_config = yaml.safe_load(f)
        
        validator = ConfigValidator()
        results = await validator.validate_intentions_config(intentions_config)
        
        # Afficher les résultats pour debug
        for result in results:
            print(f"{result.level.value}: {result.message}")
        
        # Ne devrait pas y avoir d'erreurs critiques
        critical_errors = [r for r in results if r.level == ValidationLevel.CRITICAL]
        assert len(critical_errors) == 0, f"Critical errors in intentions_v2.yaml: {[r.message for r in critical_errors]}"
    
    @pytest.mark.asyncio
    async def test_real_entities_yaml_valid(self, config_dir):
        """Test que le vrai fichier entities_v2.yaml est valide"""
        entities_file = config_dir / "entities_v2.yaml"
        
        if not entities_file.exists():
            pytest.skip("entities_v2.yaml not found")
        
        with open(entities_file, 'r', encoding='utf-8') as f:
            entities_config = yaml.safe_load(f)
        
        validator = ConfigValidator()
        results = await validator.validate_entities_config(entities_config)
        
        # Afficher les résultats pour debug
        for result in results:
            print(f"{result.level.value}: {result.message}")
        
        # Ne devrait pas y avoir d'erreurs critiques  
        critical_errors = [r for r in results if r.level == ValidationLevel.CRITICAL]
        assert len(critical_errors) == 0, f"Critical errors in entities_v2.yaml: {[r.message for r in critical_errors]}"
    
    @pytest.mark.asyncio
    async def test_real_search_service_field_mapping(self, config_dir):
        """Test mapping des champs search_service réels"""
        entities_file = config_dir / "entities_v2.yaml"
        
        if not entities_file.exists():
            pytest.skip("entities_v2.yaml not found")
        
        with open(entities_file, 'r', encoding='utf-8') as f:
            entities_config = yaml.safe_load(f)
        
        validator = ConfigValidator()
        results = await validator.validate_search_service_compatibility(entities_config)
        
        # Tous les champs devraient exister dans enrichment_service
        critical_field_errors = [
            r for r in results 
            if r.level == ValidationLevel.CRITICAL and "not found in Elasticsearch schema" in r.message
        ]
        
        if critical_field_errors:
            print("Unknown fields found:")
            for error in critical_field_errors:
                print(f"  - {error.message}")
        
        assert len(critical_field_errors) == 0, "All fields should exist in enrichment_service schema"

# Pour exécuter les tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])