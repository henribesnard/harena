import asyncio
import logging
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

try:
    from autogen_core import AgentRuntime
    from autogen_core.application import SingleThreadedAgentRuntime
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    AgentRuntime = None
    SingleThreadedAgentRuntime = None

from config_service.config import settings as global_settings
from .factory import AutoGenAgentFactory


logger = logging.getLogger(__name__)


class ConversationServiceRuntime:
    def __init__(self, settings=None):
        self.settings = settings or global_settings
        self._runtime: Optional[Any] = None
        self._factory: Optional[AutoGenAgentFactory] = None
        self._is_initialized = False
        self._agents: Dict[str, Any] = {}
        
    @property
    def is_available(self) -> bool:
        return AUTOGEN_AVAILABLE
        
    @property  
    def is_initialized(self) -> bool:
        return self._is_initialized
        
    @property
    def runtime(self) -> Optional[Any]:
        return self._runtime
        
    @property
    def factory(self) -> Optional[AutoGenAgentFactory]:
        return self._factory

    async def initialize(self) -> bool:
        if not AUTOGEN_AVAILABLE:
            logger.warning("AutoGen Core n'est pas disponible - Runtime désactivé")
            return False
            
        if self._is_initialized:
            logger.debug("Runtime AutoGen déjà initialisé")
            return True
            
        try:
            self._runtime = SingleThreadedAgentRuntime()
            self._factory = AutoGenAgentFactory(self._runtime, self.settings)
            
            # Ajout références clients pour la factory
            self._setup_external_clients_references()
            
            await self._register_core_agents()
            
            self._is_initialized = True
            logger.info("Runtime AutoGen initialisé avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du runtime AutoGen: {e}")
            self._runtime = None
            self._factory = None
            return False
            
    def _setup_external_clients_references(self):
        """Configure les références vers les clients externes pour la factory"""
        try:
            # Import dynamique pour éviter les dépendances circulaires
            from ..clients.deepseek_client import DeepSeekClient
            from ..core.cache_manager import CacheManager
            
            # Tentative récupération clients existants du contexte global
            # En production ces clients seraient passés via app.state
            deepseek_client = getattr(self.settings, '_deepseek_client', None)
            cache_manager = getattr(self.settings, '_cache_manager', None)
            
            # Stockage pour la factory
            self._factory._deepseek_client = deepseek_client
            self._factory._cache_manager = cache_manager
            
            logger.debug("Références clients externes configurées pour factory")
            
        except Exception as e:
            logger.warning(f"Impossible de configurer références clients: {e}")
            # Non critique - les agents peuvent fonctionner en mode dégradé

    async def _register_core_agents(self):
        if not self._factory:
            return
            
        core_agents = [
            ("intent_classifier", "IntentClassifierAgent"),
            ("entity_extractor", "EntityExtractorAgent"),
            ("conversation_manager", "ConversationManagerAgent"),
            ("financial_advisor", "FinancialAdvisorAgent")
        ]
        
        for agent_id, agent_type in core_agents:
            try:
                agent = await self._factory.create_agent(agent_type, agent_id)
                if agent:
                    self._agents[agent_id] = agent
                    logger.debug(f"Agent {agent_id} enregistré avec succès")
            except Exception as e:
                logger.warning(f"Impossible d'enregistrer l'agent {agent_id}: {e}")

    async def get_agent(self, agent_id: str) -> Optional[Any]:
        if not self._is_initialized:
            logger.warning("Runtime non initialisé - get_agent retourne None")
            return None
            
        return self._agents.get(agent_id)

    async def create_agent(self, agent_type: str, agent_id: str, config: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        if not self._is_initialized or not self._factory:
            logger.warning("Runtime non initialisé - create_agent retourne None")
            return None
            
        try:
            agent = await self._factory.create_agent(agent_type, agent_id, config)
            if agent:
                self._agents[agent_id] = agent
            return agent
        except Exception as e:
            logger.error(f"Erreur lors de la création de l'agent {agent_id}: {e}")
            return None

    async def send_message(self, agent_id: str, message: Any, sender_id: Optional[str] = None) -> Optional[Any]:
        if not self._is_initialized:
            logger.warning("Runtime non initialisé - send_message retourne None")
            return None
            
        agent = self._agents.get(agent_id)
        if not agent:
            logger.error(f"Agent {agent_id} introuvable")
            return None
            
        try:
            response = await agent.handle_message(message, sender_id)
            return response
        except Exception as e:
            logger.error(f"Erreur lors de l'envoi du message à {agent_id}: {e}")
            return None

    async def shutdown(self):
        if not self._is_initialized:
            return
            
        try:
            if self._runtime and hasattr(self._runtime, 'stop'):
                await self._runtime.stop()
                
            self._agents.clear()
            self._runtime = None
            self._factory = None
            self._is_initialized = False
            
            logger.info("Runtime AutoGen arrêté avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'arrêt du runtime: {e}")

    def get_status(self) -> Dict[str, Any]:
        return {
            "available": self.is_available,
            "initialized": self.is_initialized,
            "agents_count": len(self._agents),
            "agents": list(self._agents.keys()) if self._is_initialized else [],
            "runtime_type": type(self._runtime).__name__ if self._runtime else None
        }

    @asynccontextmanager
    async def agent_context(self, agent_id: str):
        agent = await self.get_agent(agent_id)
        if not agent:
            raise RuntimeError(f"Agent {agent_id} indisponible")
        try:
            yield agent
        finally:
            pass