"""
Tests unitaires pour le Template Engine
"""

import pytest
import json
import tempfile
import sys
from pathlib import Path
from typing import Dict, Any

# Ajouter le répertoire parent pour imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from conversation_service.core.template_engine import (
    TemplateEngine, CompiledTemplate, TemplateCompilationResult
)

class TestTemplateEngine:
    """Tests du moteur de templates"""
    
    @pytest.fixture
    def temp_templates_dir(self):
        """Crée un dossier temporaire avec des templates de test"""
        with tempfile.TemporaryDirectory() as temp_dir:
            templates_dir = Path(temp_dir) / "query"
            templates_dir.mkdir(parents=True)
            
            # Template simple valide
            simple_template = {
                "template_name": "test_simple",
                "description": "Template de test simple",
                "target_intention": "TEST.simple",
                "parameters": {
                    "user_id": "{{user_id}}",
                    "query": "{{search_query}}"
                },
                "parameter_mappings": {
                    "user_id": {
                        "required": True,
                        "type": "integer",
                        "source": "context.user_id"
                    },
                    "search_query": {
                        "required": False,
                        "type": "string",
                        "source": "entities.query",
                        "default": ""
                    }
                }
            }
            
            with open(templates_dir / "simple.json", 'w') as f:
                json.dump(simple_template, f)
            
            # Template avec agrégations
            aggregation_template = {
                "template_name": "test_aggregation",
                "description": "Template avec agrégations",
                "target_intention": "TEST.aggregation",
                "parameters": {
                    "user_id": "{{user_id}}",
                    "aggregations": {
                        "spending_total": {
                            "sum": {"field": "amount_abs"}
                        }
                    },
                    "aggregation_only": True
                },
                "parameter_mappings": {
                    "user_id": {
                        "required": True,
                        "type": "integer",
                        "source": "context.user_id"
                    }
                }
            }
            
            with open(templates_dir / "aggregation.json", 'w') as f:
                json.dump(aggregation_template, f)
            
            yield Path(temp_dir)

    @pytest.fixture
    def template_engine(self, temp_templates_dir):
        """Instance du moteur de templates avec templates temporaires"""
        return TemplateEngine(temp_templates_dir / "query")

    @pytest.mark.asyncio
    async def test_engine_initialization(self, template_engine):
        """Test initialisation du moteur"""
        success = await template_engine.initialize()
        assert success
        assert len(template_engine.compiled_templates) == 2

    @pytest.mark.asyncio
    async def test_compile_valid_template(self, template_engine, temp_templates_dir):
        """Test compilation d'un template valide"""
        template_file = temp_templates_dir / "query" / "simple.json"
        result = await template_engine.compile_template(template_file)
        
        assert result.success
        assert result.template is not None
        assert result.template.name == "test_simple"
        assert result.compilation_time_ms > 0

    @pytest.mark.asyncio
    async def test_compile_invalid_json(self, template_engine, temp_templates_dir):
        """Test compilation avec JSON invalide"""
        invalid_file = temp_templates_dir / "query" / "invalid.json"
        with open(invalid_file, 'w') as f:
            f.write("{ invalid json }")
        
        result = await template_engine.compile_template(invalid_file)
        assert not result.success
        assert "Invalid JSON" in result.error_message

    @pytest.mark.asyncio
    async def test_template_rendering(self, template_engine):
        """Test rendu d'un template"""
        await template_engine.initialize()
        
        # Récupérer un template compilé
        template = list(template_engine.compiled_templates.values())[0]
        
        # Paramètres de test
        parameters = {
            "context": {"user_id": 42},
            "entities": {"query": "test search"}
        }
        
        rendered = await template_engine.render_template(template, parameters)
        
        assert "user_id" in rendered
        assert rendered["user_id"] == 42

    @pytest.mark.asyncio
    async def test_parameter_extraction(self, template_engine):
        """Test extraction des paramètres"""
        await template_engine.initialize()
        
        template = list(template_engine.compiled_templates.values())[0]
        
        # Test avec paramètre requis manquant
        parameters = {"entities": {"query": "test"}}
        
        with pytest.raises(ValueError, match="Required parameter missing"):
            await template_engine.render_template(template, parameters)

    @pytest.mark.asyncio
    async def test_parameter_default_values(self, template_engine):
        """Test valeurs par défaut des paramètres"""
        await template_engine.initialize()
        
        # Trouver le template simple qui a le paramètre search_query
        template = None
        for tmpl in template_engine.compiled_templates.values():
            if "search_query" in tmpl.parameter_mappings:
                template = tmpl
                break
        
        assert template is not None, "Template simple avec search_query non trouvé"
        
        # Paramètres minimaux (sans query)
        parameters = {
            "context": {"user_id": 42},
            "entities": {}
        }
        
        rendered = await template_engine.render_template(template, parameters)
        
        # Le paramètre query devrait avoir sa valeur par défaut
        assert rendered["query"] == ""

    @pytest.mark.asyncio
    async def test_nested_parameter_extraction(self, template_engine):
        """Test extraction de paramètres imbriqués"""
        # Test avec chemin nested comme "entities.date.gte"
        test_data = {
            "context": {"user_id": 42},
            "entities": {
                "date": {
                    "gte": "2025-01-01T00:00:00Z",
                    "lte": "2025-01-31T23:59:59Z"
                }
            }
        }
        
        # Test de la méthode _get_nested_value
        value = template_engine._get_nested_value(test_data, "entities.date.gte")
        assert value == "2025-01-01T00:00:00Z"
        
        # Test avec chemin inexistant
        value = template_engine._get_nested_value(test_data, "entities.nonexistent")
        assert value is None

    @pytest.mark.asyncio 
    async def test_template_cache(self, template_engine):
        """Test du cache des templates"""
        await template_engine.initialize()
        
        # Premier chargement
        template1 = await template_engine.load_template("TEST", "simple")
        assert template1 is not None
        
        # Deuxième chargement - devrait venir du cache
        template2 = await template_engine.load_template("TEST", "simple")
        assert template1 is template2  # Même instance
        
        # Vérifier les stats de cache
        stats = template_engine.get_cache_stats()
        assert stats["cache_hits"] > 0

    @pytest.mark.asyncio
    async def test_filter_cleanup(self, template_engine):
        """Test nettoyage des filtres"""
        # Template avec nettoyage activé
        template_data = {
            "template_name": "test_cleanup",
            "description": "Test cleanup",
            "target_intention": "TEST.cleanup", 
            "parameters": {
                "user_id": "{{user_id}}",
                "filters": {
                    "date": "{{date_range}}",
                    "merchant": "{{merchant_name}}"
                }
            },
            "parameter_mappings": {
                "user_id": {
                    "required": True,
                    "type": "integer",
                    "source": "context.user_id"
                },
                "date_range": {
                    "required": False,
                    "source": "entities.date"
                },
                "merchant_name": {
                    "required": False,
                    "source": "entities.merchant"
                }
            },
            "filter_cleanup": {
                "remove_null_values": True,
                "remove_empty_objects": True
            }
        }
        
        # Simuler compilation et rendu
        from jinja2 import Template
        jinja_template = Template(json.dumps(template_data["parameters"]))
        
        compiled_template = CompiledTemplate(
            name="test_cleanup",
            template_data=template_data,
            jinja_template=jinja_template,
            parameter_mappings=template_data["parameter_mappings"],
            compiled_at=template_engine._parse_cache_duration("1h"),
            cache_duration=template_engine._parse_cache_duration("1h"),
            file_path=Path("test")
        )
        
        # Paramètres avec valeurs nulles
        parameters = {
            "context": {"user_id": 42},
            "entities": {
                "date": {"gte": "2025-01-01T00:00:00Z"},
                "merchant": None  # Sera supprimé
            }
        }
        
        rendered = await template_engine.render_template(compiled_template, parameters)
        
        # Vérifier que les valeurs nulles sont supprimées
        assert "merchant" not in rendered["filters"]
        assert "date" in rendered["filters"]

    def test_parse_cache_duration(self, template_engine):
        """Test parsing des durées de cache"""
        # Test différents formats
        assert template_engine._parse_cache_duration("30s").total_seconds() == 30
        assert template_engine._parse_cache_duration("15m").total_seconds() == 900
        assert template_engine._parse_cache_duration("2h").total_seconds() == 7200
        assert template_engine._parse_cache_duration("1d").total_seconds() == 86400
        
        # Format invalide - devrait retourner défaut (1h)
        assert template_engine._parse_cache_duration("invalid").total_seconds() == 3600

    def test_parameter_type_casting(self, template_engine):
        """Test casting des types de paramètres"""
        # Test casting vers integer
        assert template_engine._cast_parameter_type("42", "integer") == 42
        assert template_engine._cast_parameter_type(42.5, "integer") == 42
        
        # Test casting vers float
        assert template_engine._cast_parameter_type("42.5", "float") == 42.5
        
        # Test casting vers boolean
        assert template_engine._cast_parameter_type(1, "boolean") is True
        assert template_engine._cast_parameter_type(0, "boolean") is False
        
        # Test casting vers string
        assert template_engine._cast_parameter_type(42, "string") == "42"
        
        # Test avec valeur None
        assert template_engine._cast_parameter_type(None, "integer") is None

    def test_remove_null_values(self, template_engine):
        """Test suppression des valeurs nulles"""
        test_data = {
            "user_id": 42,
            "filters": {
                "date": {"gte": "2025-01-01"},
                "merchant": None,
                "amount": {"gte": 100, "lte": None}
            },
            "empty_list": [],
            "null_value": None
        }
        
        cleaned = template_engine._remove_null_values(test_data)
        
        assert "null_value" not in cleaned
        assert "merchant" not in cleaned["filters"] 
        assert "lte" not in cleaned["filters"]["amount"]
        assert cleaned["filters"]["date"]["gte"] == "2025-01-01"

    @pytest.mark.asyncio
    async def test_performance_requirements(self, template_engine):
        """Test des exigences de performance Phase 2"""
        await template_engine.initialize()
        
        # Test compilation < 50ms
        template_file = list(template_engine.templates_dir.rglob("*.json"))[0]
        result = await template_engine.compile_template(template_file)
        
        assert result.success
        assert result.compilation_time_ms < 50, f"Compilation trop lente: {result.compilation_time_ms}ms"

    @pytest.mark.asyncio
    async def test_aggregation_template(self, template_engine):
        """Test template avec agrégations"""
        await template_engine.initialize()
        
        # Trouver le template d'agrégation
        agg_template = None
        for template in template_engine.compiled_templates.values():
            if "aggregation" in template.name:
                agg_template = template
                break
        
        assert agg_template is not None
        
        parameters = {
            "context": {"user_id": 42}
        }
        
        rendered = await template_engine.render_template(agg_template, parameters)
        
        assert "aggregations" in rendered
        assert "spending_total" in rendered["aggregations"]
        assert rendered["aggregation_only"] is True

# Tests d'intégration avec de vrais templates
class TestRealTemplates:
    """Tests avec les vrais templates du projet"""
    
    @pytest.mark.asyncio
    async def test_real_templates_compilation(self):
        """Test compilation des vrais templates"""
        real_templates_dir = Path(__file__).parent.parent.parent.parent / "templates" / "query"
        
        if not real_templates_dir.exists():
            pytest.skip("Real templates directory not found")
        
        engine = TemplateEngine(real_templates_dir)
        success = await engine.initialize()
        
        assert success
        assert len(engine.compiled_templates) > 0
        
        # Vérifier que tous les templates sont valides
        for template_name, template in engine.compiled_templates.items():
            assert template.name is not None
            assert template.template_data is not None
            assert template.jinja_template is not None

    @pytest.mark.asyncio
    async def test_transaction_search_template(self):
        """Test template de recherche de transactions"""
        real_templates_dir = Path(__file__).parent.parent.parent.parent / "templates" / "query"
        
        if not real_templates_dir.exists():
            pytest.skip("Real templates directory not found")
            
        engine = TemplateEngine(real_templates_dir)
        await engine.initialize()
        
        # Test template by_date
        template_file = real_templates_dir / "transaction_search" / "by_date.json"
        if template_file.exists():
            result = await engine.compile_template(template_file)
            assert result.success
            
            # Test rendu
            parameters = {
                "context": {"user_id": 42},
                "entities": {
                    "periode_temporelle": {
                        "date": {"gte": "2025-01-01T00:00:00Z", "lte": "2025-01-31T23:59:59Z"}
                    }
                }
            }
            
            rendered = await engine.render_template(result.template, parameters)
            assert rendered["user_id"] == 42
            assert "filters" in rendered
            assert "date" in rendered["filters"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])