#!/usr/bin/env python3
"""
Script de validation rapide de l'architecture Conversation Service MVP.

Ce script vérifie rapidement que tous les composants sont correctement
installés et configurés sans exécuter de tests longs.

Usage:
    python validate_architecture.py
    
Author: Conversation Service Team
Created: 2025-01-31
Version: 1.0.0 MVP - Quick Architecture Validation
"""

import sys
import os
import importlib
import logging
from typing import Dict, List, Tuple
from pathlib import Path

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class ArchitectureValidator:
    """Validateur rapide de l'architecture du service."""
    
    def __init__(self):
        self.results = {
            'modules': [],
            'config': [],
            'structure': [],
            'dependencies': [],
            'errors': [],
            'warnings': []
        }
    
    def validate_all(self) -> bool:
        """Valide tous les aspects de l'architecture."""
        logger.info("🚀 Validation de l'architecture Conversation Service MVP")
        logger.info("=" * 60)
        
        success = True
        
        # 1. Structure des dossiers
        if not self.validate_folder_structure():
            success = False
        
        # 2. Imports des modules
        if not self.validate_module_imports():
            success = False
        
        # 3. Configuration
        if not self.validate_configuration():
            success = False
        
        # 4. Dépendances
        if not self.validate_dependencies():
            success = False
        
        # 5. Résumé
        self.print_summary()
        
        return success
    
    def validate_folder_structure(self) -> bool:
        """Valide la structure des dossiers."""
        logger.info("📁 Validation de la structure des dossiers...")
        
        expected_structure = {
            'models/': [
                '__init__.py',
                'agent_models.py', 
                'conversation_models.py',
                'financial_models.py',
                'service_contracts.py'
            ],
            'core/': [
                '__init__.py',
                'deepseek_client.py',
                'conversation_manager.py',
                'mvp_team_manager.py'
            ],
            'agents/': [
                '__init__.py',
                'base_financial_agent.py',
                'hybrid_intent_agent.py',
                'search_query_agent.py',
                'response_agent.py',
                'orchestrator_agent.py'
            ],
            'intent_rules/': [
                '__init__.py',
                'rule_loader.py',
                'pattern_matcher.py',
                'rule_engine.py'
            ],
            'utils/': [
                '__init__.py',
                'validators.py'
            ]
        }
        
        success = True
        for folder, files in expected_structure.items():
            folder_path = Path(folder)
            
            if not folder_path.exists():
                self.results['errors'].append(f"Dossier manquant: {folder}")
                success = False
                continue
            
            self.results['structure'].append(f"✅ {folder}")
            
            for file in files:
                file_path = folder_path / file
                if file_path.exists():
                    self.results['structure'].append(f"  ✅ {file}")
                else:
                    self.results['errors'].append(f"Fichier manquant: {folder}{file}")
                    success = False
        
        return success
    
    def validate_module_imports(self) -> bool:
        """Valide les imports des modules principaux."""
        logger.info("📦 Validation des imports de modules...")
        
        modules_to_test = [
            # Models
            ('models.agent_models', ['AgentConfig', 'AgentResponse', 'TeamWorkflow']),
            ('models.conversation_models', ['ConversationTurn', 'ConversationContext']),
            ('models.financial_models', ['FinancialEntity', 'IntentResult', 'EntityType']),
            ('models.service_contracts', ['SearchServiceQuery', 'SearchServiceResponse']),
            
            # Core (conditional)
            ('core', ['check_core_dependencies', 'get_core_config']),
            
            # Intent Rules
            ('intent_rules.rule_loader', ['RuleLoader']),
            ('intent_rules.rule_engine', ['RuleEngine']),
            
            # Utils
            ('utils.validators', ['ContractValidator']),
        ]
        
        success = True
        for module_name, expected_exports in modules_to_test:
            try:
                module = importlib.import_module(module_name)
                
                # Vérifier les exports
                missing_exports = []
                for export in expected_exports:
                    if not hasattr(module, export):
                        missing_exports.append(export)
                
                if missing_exports:
                    self.results['warnings'].append(
                        f"⚠️ {module_name}: exports manquants {missing_exports}"
                    )
                else:
                    self.results['modules'].append(f"✅ {module_name}")
                
            except ImportError as e:
                self.results['errors'].append(f"❌ {module_name}: {str(e)}")
                success = False
            except Exception as e:
                self.results['warnings'].append(f"⚠️ {module_name}: {str(e)}")
        
        return success
    
    def validate_configuration(self) -> bool:
        """Valide la configuration et les variables d'environnement."""
        logger.info("⚙️ Validation de la configuration...")
        
        # Variables d'environnement critiques
        critical_env_vars = [
            'DEEPSEEK_API_KEY'
        ]
        
        # Variables d'environnement optionnelles mais recommandées
        optional_env_vars = [
            'DEEPSEEK_BASE_URL',
            'SEARCH_SERVICE_URL',
            'MAX_CONVERSATION_HISTORY',
            'WORKFLOW_TIMEOUT_SECONDS'
        ]
        
        success = True
        
        # Vérifier variables critiques
        for var in critical_env_vars:
            value = os.getenv(var)
            if value:
                # Masquer la clé API pour la sécurité
                if 'API_KEY' in var:
                    display_value = f"{value[:8]}..." if len(value) > 8 else "***"
                else:
                    display_value = value
                self.results['config'].append(f"✅ {var}={display_value}")
            else:
                self.results['errors'].append(f"❌ Variable critique manquante: {var}")
                success = False
        
        # Vérifier variables optionnelles
        for var in optional_env_vars:
            value = os.getenv(var)
            if value:
                self.results['config'].append(f"✅ {var}={value}")
            else:
                self.results['warnings'].append(f"⚠️ Variable optionnelle: {var} (valeur par défaut utilisée)")
        
        # Tester chargement de la configuration core si disponible
        try:
            from core import get_core_config
            config = get_core_config()
            self.results['config'].append(f"✅ Configuration core chargée: {len(config)} paramètres")
        except ImportError:
            self.results['warnings'].append("⚠️ Configuration core non disponible")
        except Exception as e:
            self.results['errors'].append(f"❌ Erreur configuration core: {str(e)}")
            success = False
        
        return success
    
    def validate_dependencies(self) -> bool:
        """Valide les dépendances externes."""
        logger.info("📚 Validation des dépendances...")
        
        required_packages = [
            'pydantic',
            'httpx',
            'asyncio'
        ]
        
        optional_packages = [
            'autogen',
            'pytest',
            'redis'
        ]
        
        success = True
        
        # Dépendances requises
        for package in required_packages:
            try:
                importlib.import_module(package)
                self.results['dependencies'].append(f"✅ {package}")
            except ImportError:
                self.results['errors'].append(f"❌ Dépendance requise manquante: {package}")
                success = False
        
        # Dépendances optionnelles
        for package in optional_packages:
            try:
                importlib.import_module(package)
                self.results['dependencies'].append(f"✅ {package} (optionnel)")
            except ImportError:
                self.results['warnings'].append(f"⚠️ Dépendance optionnelle: {package}")
        
        return success
    
    def print_summary(self):
        """Affiche le résumé de la validation."""
        logger.info("=" * 60)
        logger.info("📊 RÉSUMÉ DE LA VALIDATION")
        logger.info("=" * 60)
        
        # Statistiques
        total_checks = (
            len(self.results['modules']) + 
            len(self.results['config']) + 
            len(self.results['structure']) + 
            len(self.results['dependencies'])
        )
        
        errors_count = len(self.results['errors'])
        warnings_count = len(self.results['warnings'])
        
        logger.info(f"📈 Statistiques:")
        logger.info(f"   ✅ Vérifications réussies: {total_checks}")
        logger.info(f"   ❌ Erreurs: {errors_count}")
        logger.info(f"   ⚠️ Avertissements: {warnings_count}")
        
        # Détails des erreurs
        if self.results['errors']:
            logger.info("\n❌ ERREURS À CORRIGER:")
            for error in self.results['errors']:
                logger.info(f"   {error}")
        
        # Détails des avertissements
        if self.results['warnings']:
            logger.info("\n⚠️ AVERTISSEMENTS:")
            for warning in self.results['warnings']:
                logger.info(f"   {warning}")
        
        # Succès par catégorie
        logger.info("\n📦 MODULES:")
        for module in self.results['modules']:
            logger.info(f"   {module}")
        
        logger.info("\n⚙️ CONFIGURATION:")
        for config in self.results['config']:
            logger.info(f"   {config}")
        
        logger.info("\n📁 STRUCTURE:")
        for structure in self.results['structure']:
            logger.info(f"   {structure}")
        
        logger.info("\n📚 DÉPENDANCES:")
        for dep in self.results['dependencies']:
            logger.info(f"   {dep}")
        
        # Verdict final
        logger.info("=" * 60)
        if errors_count == 0:
            logger.info("🎉 ARCHITECTURE VALIDÉE AVEC SUCCÈS! 🎉")
            logger.info("Le Conversation Service MVP est prêt à être utilisé.")
            if warnings_count > 0:
                logger.info(f"Note: {warnings_count} avertissement(s) à considérer pour l'optimisation.")
        else:
            logger.info("❌ VALIDATION ÉCHOUÉE")
            logger.info(f"Corrigez les {errors_count} erreur(s) avant de continuer.")
        logger.info("=" * 60)

def create_sample_env_file():
    """Crée un fichier .env.example avec les variables nécessaires."""
    env_content = """# Configuration Conversation Service MVP
# Copiez ce fichier vers .env et remplissez les valeurs

# === DEEPSEEK CONFIGURATION (REQUIS) ===
DEEPSEEK_API_KEY=your-deepseek-api-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_TIMEOUT=30

# === SEARCH SERVICE CONFIGURATION ===
SEARCH_SERVICE_URL=http://localhost:8000

# === CONVERSATION CONFIGURATION ===
MAX_CONVERSATION_HISTORY=100
WORKFLOW_TIMEOUT_SECONDS=60
HEALTH_CHECK_INTERVAL_SECONDS=300
AUTO_RECOVERY_ENABLED=true

# === CACHE CONFIGURATION ===
CACHE_ENABLED=true
METRICS_ENABLED=true
RATE_LIMIT_ENABLED=true

# === LOGGING CONFIGURATION ===
LOG_LEVEL=INFO
LOG_TO_FILE=false
"""
    
    with open('.env.example', 'w') as f:
        f.write(env_content)
    
    logger.info("📝 Fichier .env.example créé avec les variables nécessaires")

def main():
    """Point d'entrée principal."""
    validator = ArchitectureValidator()
    
    # Créer fichier .env.example si inexistant
    if not Path('.env.example').exists():
        create_sample_env_file()
    
    # Valider l'architecture
    success = validator.validate_all()
    
    # Suggestions d'amélioration
    if not success:
        logger.info("\n🔧 SUGGESTIONS POUR CORRIGER:")
        logger.info("1. Vérifiez que tous les fichiers sont présents")
        logger.info("2. Installez les dépendances: pip install pydantic httpx")
        logger.info("3. Configurez les variables d'environnement (voir .env.example)")
        logger.info("4. Assurez-vous que tous les imports fonctionnent")
    else:
        logger.info("\n🚀 PROCHAINES ÉTAPES:")
        logger.info("1. Exécutez les tests d'intégration: python test_integration_complete.py")
        logger.info("2. Configurez le Search Service externe")
        logger.info("3. Testez le workflow complet avec des données réelles")
        logger.info("4. Déployez en production avec monitoring")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)