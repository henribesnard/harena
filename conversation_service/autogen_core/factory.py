import logging
from typing import Any, Dict, List, Optional, Type

try:
    from autogen_core import AgentRuntime
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    AgentRuntime = None

from config_service.config import settings as global_settings


logger = logging.getLogger(__name__)


class AutoGenAgentFactory:
    def __init__(self, runtime: Optional[Any], settings=None):
        self.runtime = runtime
        self.settings = settings or global_settings
        self._agent_registry: Dict[str, Type] = {}
        self._register_default_agents()
        
    def _register_default_agents(self):
        if not AUTOGEN_AVAILABLE:
            logger.warning("AutoGen non disponible - Registry d'agents vide")
            return
            
        try:
            from ..agents.financial.intent_classifier import IntentClassifierAgent
            from ..agents.financial.entity_extractor import EntityExtractorAgent
            from ..agents.base_agent import BaseAgent
            
            self._agent_registry.update({
                "IntentClassifierAgent": IntentClassifierAgent,
                "EntityExtractorAgent": EntityExtractorAgent,
                "ConversationManagerAgent": BaseAgent,  # Fallback
                "FinancialAdvisorAgent": BaseAgent,     # Fallback
                "BaseAgent": BaseAgent
            })
            
            logger.debug(f"Agents enregistrés: {list(self._agent_registry.keys())}")
            
        except ImportError as e:
            logger.warning(f"Impossible d'importer certains agents: {e}")

    def register_agent_type(self, agent_type: str, agent_class: Type) -> bool:
        try:
            self._agent_registry[agent_type] = agent_class
            logger.debug(f"Type d'agent {agent_type} enregistré")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement du type {agent_type}: {e}")
            return False

    async def create_agent(self, agent_type: str, agent_id: str, config: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        if not AUTOGEN_AVAILABLE:
            logger.warning(f"AutoGen non disponible - Création de mock agent pour {agent_id}")
            return self._create_mock_agent(agent_type, agent_id, config)
            
        if not self.runtime:
            logger.error("Runtime AutoGen non disponible")
            return None
            
        agent_class = self._agent_registry.get(agent_type)
        if not agent_class:
            logger.error(f"Type d'agent {agent_type} non trouvé dans le registry")
            return None
            
        try:
            agent_config = self._prepare_agent_config(agent_type, config)
            
            # Paramètres spéciaux pour agents AutoGen
            if agent_type == "IntentClassifierAgent":
                # Configuration pour mode AutoGen avec fallback classique
                init_params = {
                    "name": agent_id,
                    "autogen_mode": True,
                    "deepseek_client": getattr(self.runtime, '_deepseek_client', None),
                    "cache_manager": getattr(self.runtime, '_cache_manager', None),
                    **agent_config
                }
                agent = agent_class(**init_params)
                
                # Activation collaboration équipe
                if hasattr(agent, 'activate_team_collaboration'):
                    agent.activate_team_collaboration()
                    logger.debug(f"Collaboration équipe activée pour {agent_id}")
                    
            elif agent_type == "EntityExtractorAgent":
                # Configuration pour EntityExtractorAgent en mode AutoGen
                init_params = {
                    "name": agent_id,
                    "autogen_mode": True,
                    "deepseek_client": getattr(self.runtime, '_deepseek_client', None),
                    "cache_manager": getattr(self.runtime, '_cache_manager', None),
                    **agent_config
                }
                agent = agent_class(**init_params)
                
                # Activation collaboration équipe
                if hasattr(agent, 'activate_team_collaboration'):
                    agent.activate_team_collaboration()
                    logger.debug(f"Collaboration équipe activée pour {agent_id}")
                    
            else:
                # Autres types d'agents - configuration standard
                init_params = self._get_init_parameters(agent_class, agent_id, agent_config)
                agent = agent_class(**init_params)
                
            # Enregistrement dans le runtime si supporté
            if hasattr(self.runtime, 'register_agent') and hasattr(agent, 'register'):
                await agent.register(self.runtime, agent_id)
                
            logger.debug(f"Agent {agent_id} de type {agent_type} créé avec succès en mode AutoGen")
            return agent
            
        except Exception as e:
            logger.error(f"Erreur lors de la création de l'agent {agent_id}: {e}")
            logger.debug("Tentative création mock agent comme fallback")
            return self._create_mock_agent(agent_type, agent_id, config)

    def _prepare_agent_config(self, agent_type: str, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        base_config = {
            "deepseek_api_key": self.settings.DEEPSEEK_API_KEY,
            "deepseek_base_url": self.settings.DEEPSEEK_BASE_URL,
            "model_name": self.settings.DEEPSEEK_MODEL,
        }
        
        type_specific_configs = {
            "IntentClassifierAgent": {
                "temperature": 0.3,
                "max_tokens": 150,
                "timeout": 30.0
            },
            "EntityExtractorAgent": {
                "temperature": 0.05,
                "max_tokens": 200,
                "timeout": 35.0
            },
            "ConversationManagerAgent": {
                "temperature": 0.7,
                "max_tokens": 500,
                "timeout": 45.0
            },
            "FinancialAdvisorAgent": {
                "temperature": 0.5,
                "max_tokens": 800,
                "timeout": 60.0
            }
        }
        
        agent_config = base_config.copy()
        agent_config.update(type_specific_configs.get(agent_type, {}))
        
        if config:
            agent_config.update(config)
            
        return agent_config

    def _get_init_parameters(self, agent_class: Type, agent_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        import inspect
        
        try:
            sig = inspect.signature(agent_class.__init__)
            params = {}
            
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                elif param_name == 'name' or param_name == 'agent_id':
                    params[param_name] = agent_id
                elif param_name in config:
                    params[param_name] = config[param_name]
                elif param.default != inspect.Parameter.empty:
                    continue
                else:
                    logger.warning(f"Paramètre requis {param_name} manquant pour {agent_class.__name__}")
                    
            return params
            
        except Exception as e:
            logger.warning(f"Impossible d'analyser la signature de {agent_class.__name__}: {e}")
            return {"name": agent_id}

    def _create_mock_agent(self, agent_type: str, agent_id: str, config: Optional[Dict[str, Any]]) -> Any:
        class MockAutoGenAgent:
            def __init__(self, agent_id: str, agent_type: str):
                self.name = agent_id
                self.agent_type = agent_type
                self.logger = logging.getLogger(f"MockAutoGenAgent.{agent_id}")
                
            async def handle_message(self, message: Any, sender_id: Optional[str] = None) -> Dict[str, Any]:
                self.logger.debug(f"Mock agent {self.name} reçoit message de {sender_id}")
                
                if self.agent_type == "IntentClassifierAgent":
                    return {
                        "intent": "UNKNOWN",
                        "confidence": 0.5,
                        "entities": {},
                        "suggested_team": "general"
                    }
                elif self.agent_type == "EntityExtractorAgent":
                    return {
                        "entities": {
                            "amounts": [],
                            "dates": [],
                            "merchants": [],
                            "categories": [],
                            "operation_types": [],
                            "text_search": []
                        },
                        "confidence": 0.5,
                        "reasoning": f"Extraction mock par {self.name}",
                        "team_context": {
                            "ready_for_query_generation": True,
                            "mock_mode": True
                        }
                    }
                else:
                    return {
                        "response": f"Réponse mock de {self.name}",
                        "agent_type": self.agent_type,
                        "processed": True
                    }
                    
            async def register(self, runtime: Any, agent_id: str):
                self.logger.debug(f"Mock agent {agent_id} enregistré")
                
        return MockAutoGenAgent(agent_id, agent_type)

    def get_available_agent_types(self) -> List[str]:
        return list(self._agent_registry.keys())

    def get_factory_status(self) -> Dict[str, Any]:
        return {
            "autogen_available": AUTOGEN_AVAILABLE,
            "runtime_available": self.runtime is not None,
            "registered_types": len(self._agent_registry),
            "agent_types": list(self._agent_registry.keys())
        }