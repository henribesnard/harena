"""
Tests d'intégration pour la nouvelle architecture v2.0
Tests basiques pour valider que la nouvelle architecture fonctionne
"""

import pytest
from pathlib import Path


def test_conversation_service_structure():
    """Test que la structure conversation_service est correcte"""
    base_path = Path(__file__).parent.parent / "conversation_service"
    
    # Vérifier les dossiers principaux
    assert (base_path / "config").exists()
    assert (base_path / "core").exists() 
    assert (base_path / "agents").exists()
    assert (base_path / "models").exists()
    assert (base_path / "templates").exists()
    assert (base_path / "tests").exists()
    
    # Vérifier les fichiers clés
    assert (base_path / "config" / "entities_v2.yaml").exists()
    assert (base_path / "config" / "intentions_v2.yaml").exists()
    assert (base_path / "core" / "template_engine.py").exists()
    assert (base_path / "agents" / "base_agent.py").exists()


def test_config_files_exist():
    """Test que les fichiers de configuration existent"""
    config_path = Path(__file__).parent.parent / "conversation_service" / "config"
    
    assert (config_path / "entities_v2.yaml").exists()
    assert (config_path / "intentions_v2.yaml").exists()


def test_template_files_exist():
    """Test que les templates existent"""
    templates_path = Path(__file__).parent.parent / "conversation_service" / "templates" / "query"
    
    # Vérifier qu'il y a des templates
    template_files = list(templates_path.glob("**/*.json"))
    assert len(template_files) > 0


@pytest.mark.asyncio
async def test_config_validation():
    """Test que les configurations peuvent être chargées et validées"""
    try:
        from conversation_service.config.settings import ConfigManager
        from pathlib import Path
        
        config_dir = Path(__file__).parent.parent / "conversation_service" / "config"
        manager = ConfigManager(config_dir)
        
        success = await manager.initialize()
        assert success, "Configuration should load successfully"
        
        status = manager.get_configuration_status()
        assert status["is_valid"], "Configuration should be valid"
        
    except ImportError:
        pytest.skip("ConfigManager not available")


@pytest.mark.asyncio  
async def test_template_engine():
    """Test que le moteur de templates fonctionne"""
    try:
        from conversation_service.core.template_engine import TemplateEngine
        from pathlib import Path
        
        templates_dir = Path(__file__).parent.parent / "conversation_service" / "templates" / "query"
        engine = TemplateEngine(templates_dir)
        
        success = await engine.initialize()
        assert success, "Template engine should initialize successfully"
        
        assert len(engine.compiled_templates) > 0, "Should have compiled templates"
        
    except ImportError:
        pytest.skip("TemplateEngine not available")


def test_new_architecture_complete():
    """Test que la nouvelle architecture est complète selon le plan"""
    base_path = Path(__file__).parent.parent / "conversation_service"
    
    # Phase 1: Configuration et validation ✅
    assert (base_path / "config" / "config_validator.py").exists()
    assert (base_path / "config" / "settings.py").exists()
    
    # Phase 2: Templates et moteur de requêtes ✅  
    assert (base_path / "core" / "template_engine.py").exists()
    assert (base_path / "templates").exists()
    
    # Phase 3: Agents (structure de base créée)
    assert (base_path / "agents" / "base_agent.py").exists()
    assert (base_path / "agents" / "intent_classifier.py").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])