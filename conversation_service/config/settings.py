"""
Configuration Manager pour Conversation Service v2.0
Chargement dynamique avec hot-reload des configurations YAML
"""

import os
import yaml
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)

@dataclass
class ConfigurationState:
    """État actuel de la configuration"""
    intentions: Dict[str, Any] = field(default_factory=dict)
    entities: Dict[str, Any] = field(default_factory=dict)
    llm_providers: Dict[str, Any] = field(default_factory=dict)
    query_templates: Dict[str, Any] = field(default_factory=dict)
    loaded_at: Optional[datetime] = None
    version: Optional[str] = None
    is_valid: bool = False
    validation_errors: List[str] = field(default_factory=list)

class ConfigFileHandler(FileSystemEventHandler):
    """Handler pour détecter les changements de fichiers de configuration"""
    
    def __init__(self, config_manager: 'ConfigManager'):
        self.config_manager = config_manager
        
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.yaml'):
            logger.info(f"Configuration file modified: {event.src_path}")
            asyncio.create_task(self.config_manager.reload_configuration())

class ConfigManager:
    """Gestionnaire principal des configurations avec hot-reload"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path(__file__).parent
        self.state = ConfigurationState()
        self.observer = None
        self.file_handler = ConfigFileHandler(self)
        
        # Fichiers de configuration à surveiller
        self.config_files = {
            'intentions': 'intentions_v2.yaml',
            'entities': 'entities_v2.yaml', 
            'llm_providers': 'llm_providers.yaml',
            'query_templates': 'query_templates.yaml'
        }
        
        logger.info(f"ConfigManager initialized with directory: {self.config_dir}")

    async def initialize(self) -> bool:
        """Initialise le gestionnaire de configuration"""
        try:
            # Charger les configurations initiales
            success = await self.load_configurations()
            if success:
                # Démarrer la surveillance des fichiers
                await self.start_file_watching()
                logger.info("Configuration manager initialized successfully")
            return success
        except Exception as e:
            logger.error(f"Failed to initialize configuration manager: {e}")
            return False

    async def load_configurations(self) -> bool:
        """Charge toutes les configurations depuis les fichiers YAML"""
        try:
            new_state = ConfigurationState()
            
            # Charger intentions (obligatoire)
            intentions_path = self.config_dir / self.config_files['intentions']
            if intentions_path.exists():
                with open(intentions_path, 'r', encoding='utf-8') as f:
                    new_state.intentions = yaml.safe_load(f)
                logger.info(f"Loaded intentions config: {len(new_state.intentions)} groups")
            else:
                logger.warning(f"Intentions file not found: {intentions_path}")
                
            # Charger entités (obligatoire)
            entities_path = self.config_dir / self.config_files['entities']
            if entities_path.exists():
                with open(entities_path, 'r', encoding='utf-8') as f:
                    new_state.entities = yaml.safe_load(f)
                logger.info(f"Loaded entities config")
            else:
                logger.warning(f"Entities file not found: {entities_path}")
                
            # Charger providers LLM (optionnel)
            llm_path = self.config_dir / self.config_files['llm_providers']
            if llm_path.exists():
                with open(llm_path, 'r', encoding='utf-8') as f:
                    new_state.llm_providers = yaml.safe_load(f)
                logger.info("Loaded LLM providers config")
                # Compléter avec les variables d'environnement
                new_state.llm_providers = self._merge_with_env_vars(new_state.llm_providers)
            else:
                # Configuration par défaut
                new_state.llm_providers = self._get_default_llm_config()
                logger.info("Using default LLM providers config")
                
            # Charger templates de requêtes (optionnel)
            templates_path = self.config_dir / self.config_files['query_templates']
            if templates_path.exists():
                with open(templates_path, 'r', encoding='utf-8') as f:
                    new_state.query_templates = yaml.safe_load(f)
                logger.info("Loaded query templates config")
            else:
                new_state.query_templates = {}
                logger.info("No query templates config found")
            
            # Valider la configuration
            is_valid, errors = await self._validate_configuration(new_state)
            new_state.is_valid = is_valid
            new_state.validation_errors = errors
            new_state.loaded_at = datetime.now()
            new_state.version = new_state.intentions.get('metadata', {}).get('version', 'unknown')
            
            if is_valid:
                # Remplacer l'état actuel
                self.state = new_state
                logger.info(f"Configuration loaded successfully (version: {new_state.version})")
                return True
            else:
                logger.error(f"Configuration validation failed: {errors}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load configurations: {e}")
            return False

    async def reload_configuration(self) -> bool:
        """Recharge la configuration (hot-reload)"""
        logger.info("Reloading configuration...")
        return await self.load_configurations()

    async def start_file_watching(self):
        """Démarre la surveillance des fichiers de configuration"""
        if self.observer is None:
            self.observer = Observer()
            self.observer.schedule(
                self.file_handler,
                str(self.config_dir),
                recursive=False
            )
            self.observer.start()
            logger.info("File watching started for configuration directory")

    async def stop_file_watching(self):
        """Arrête la surveillance des fichiers"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            logger.info("File watching stopped")

    def get_intentions_config(self) -> Dict[str, Any]:
        """Retourne la configuration des intentions"""
        return self.state.intentions

    def get_entities_config(self) -> Dict[str, Any]:
        """Retourne la configuration des entités"""
        return self.state.entities

    def get_llm_providers_config(self) -> Dict[str, Any]:
        """Retourne la configuration des providers LLM"""
        return self.state.llm_providers

    def get_query_templates_config(self) -> Dict[str, Any]:
        """Retourne la configuration des templates de requêtes"""
        return self.state.query_templates

    def get_configuration_status(self) -> Dict[str, Any]:
        """Retourne le statut de la configuration"""
        return {
            'is_valid': self.state.is_valid,
            'version': self.state.version,
            'loaded_at': self.state.loaded_at.isoformat() if self.state.loaded_at else None,
            'validation_errors': self.state.validation_errors,
            'config_files_status': self._get_files_status()
        }

    def _get_files_status(self) -> Dict[str, Dict[str, Any]]:
        """Retourne le statut des fichiers de configuration"""
        status = {}
        for config_name, filename in self.config_files.items():
            filepath = self.config_dir / filename
            status[config_name] = {
                'exists': filepath.exists(),
                'path': str(filepath),
                'size': filepath.stat().st_size if filepath.exists() else 0,
                'modified_at': datetime.fromtimestamp(
                    filepath.stat().st_mtime
                ).isoformat() if filepath.exists() else None
            }
        return status

    def _get_default_llm_config(self) -> Dict[str, Any]:
        """Configuration par défaut des providers LLM"""
        import os
        from dotenv import load_dotenv
        
        # S'assurer que le .env est chargé
        load_dotenv()
        
        deepseek_key = os.getenv('DEEPSEEK_API_KEY')
        openai_key = os.getenv('OPENAI_API_KEY')
        
        
        return {
            'providers': {
                'deepseek': {
                    'enabled': bool(deepseek_key),
                    'priority': 1,
                    'model': 'deepseek-chat',
                    'api_key': deepseek_key or '',
                    'base_url': 'https://api.deepseek.com',
                    'temperature': 0.1,
                    'max_tokens': 1500,
                    'timeout': 30
                },
                'openai': {
                    'enabled': bool(openai_key),
                    'priority': 2,
                    'model': 'gpt-3.5-turbo',
                    'api_key': openai_key or '',
                    'base_url': 'https://api.openai.com/v1',
                    'temperature': 0.1,
                    'max_tokens': 1500,
                    'timeout': 30
                },
                'local': {
                    'enabled': False,
                    'priority': 3,
                    'model': 'llama2',
                    'api_key': '',
                    'temperature': 0.1,
                    'max_tokens': 1500,
                    'timeout': 60
                }
            },
            'fallback_strategy': 'sequential',
            'retry_attempts': 2
        }
    
    def _merge_with_env_vars(self, yaml_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge la config YAML avec les variables d'environnement pour sécurité"""
        import os
        from dotenv import load_dotenv
        
        # S'assurer que le .env est chargé
        load_dotenv()
        
        # Copier la config YAML
        merged_config = yaml_config.copy()
        
        # Ajouter les API keys et base URLs depuis les env vars
        if 'providers' in merged_config:
            providers = merged_config['providers']
            
            # DeepSeek
            if 'deepseek' in providers:
                deepseek_key = os.getenv('DEEPSEEK_API_KEY')
                if deepseek_key:
                    providers['deepseek']['api_key'] = deepseek_key
                    providers['deepseek']['base_url'] = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
            
            # OpenAI
            if 'openai' in providers:
                openai_key = os.getenv('OPENAI_API_KEY')
                if openai_key:
                    providers['openai']['api_key'] = openai_key
                    providers['openai']['base_url'] = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
            
            # Local (pas d'API key nécessaire)
            if 'local' in providers:
                providers['local']['base_url'] = os.getenv('LOCAL_LLM_URL', 'http://localhost:11434')
        
        return merged_config

    async def _validate_configuration(self, state: ConfigurationState) -> tuple[bool, List[str]]:
        """Valide la configuration chargée"""
        errors = []
        
        try:
            # Validation intentions
            if not state.intentions:
                errors.append("Missing intentions configuration")
            elif 'intent_groups' not in state.intentions:
                errors.append("Missing 'intent_groups' in intentions config")
                
            # Validation entités
            if not state.entities:
                errors.append("Missing entities configuration")
            elif 'search_service_fields' not in state.entities:
                errors.append("Missing 'search_service_fields' in entities config")
                
            # Validation providers LLM
            if state.llm_providers and 'providers' in state.llm_providers:
                for provider_name, config in state.llm_providers['providers'].items():
                    if not isinstance(config.get('enabled'), bool):
                        errors.append(f"Invalid 'enabled' setting for provider {provider_name}")
                    if not isinstance(config.get('priority'), int):
                        errors.append(f"Invalid 'priority' setting for provider {provider_name}")
                        
            # Validation cohérence inter-configurations
            await self._validate_cross_references(state, errors)
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Configuration validation exception: {str(e)}")
            return False, errors

    async def _validate_cross_references(self, state: ConfigurationState, errors: List[str]):
        """Valide la cohérence entre les configurations"""
        try:
            # Vérifier que les champs mentionnés dans intentions existent dans entities
            if state.intentions and state.entities:
                intent_groups = state.intentions.get('intent_groups', {})
                available_fields = set()
                
                # Collecter tous les champs disponibles
                entities_config = state.entities.get('search_service_fields', {})
                for field_group in entities_config.values():
                    if isinstance(field_group, list):
                        available_fields.update(field_group)
                        
                # Vérifier les examples d'intentions
                for group_name, group_config in intent_groups.items():
                    patterns = group_config.get('patterns_generiques', {})
                    for pattern_name, pattern_config in patterns.items():
                        examples = pattern_config.get('few_shot_examples', [])
                        for example in examples:
                            # Validation basique de la structure
                            if 'input' not in example or 'output' not in example:
                                errors.append(f"Invalid example structure in {group_name}.{pattern_name}")
                                
        except Exception as e:
            errors.append(f"Cross-reference validation failed: {str(e)}")

# Instance globale partagée
_config_manager: Optional[ConfigManager] = None

async def get_config_manager() -> ConfigManager:
    """Retourne l'instance globale du gestionnaire de configuration"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
        await _config_manager.initialize()
    return _config_manager

async def initialize_config(config_dir: Optional[Path] = None) -> bool:
    """Initialise la configuration globale"""
    global _config_manager
    _config_manager = ConfigManager(config_dir)
    return await _config_manager.initialize()

def get_config() -> ConfigManager:
    """Retourne l'instance du gestionnaire (synchrone)"""
    global _config_manager
    if _config_manager is None:
        raise RuntimeError("Configuration not initialized. Call initialize_config() first.")
    return _config_manager