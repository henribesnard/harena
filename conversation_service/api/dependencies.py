"""
Dependances FastAPI pour conversation service - Architecture v2.0 complete  
Injection de dependances pour pipeline conversation lineaire
"""
import logging
import time
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from fastapi import Request, HTTPException, Depends

# Architecture v2.0 imports
from ..config.settings import ConfigManager
from ..core.template_engine import TemplateEngine
from ..core.context_manager import ContextManager
from ..core.query_builder import QueryBuilder
from ..core.query_executor import QueryExecutor
from ..agents.llm import (
    LLMProviderManager, ProviderConfig, ProviderType,
    IntentClassifier,
    ResponseGenerator
)
from ..core.conversation_orchestrator import ConversationOrchestrator

# Legacy imports pour compatibilite
from conversation_service.clients.deepseek_client import DeepSeekClient
from conversation_service.core.cache_manager import CacheManager
from conversation_service.api.middleware.auth_middleware import get_current_user_id, verify_user_id_match
from conversation_service.utils.metrics_collector import metrics_collector
from conversation_service.services.conversation_persistence import ConversationPersistenceService

logger = logging.getLogger(__name__)

# === APPLICATION STATE v2.0 ===

class ApplicationState:
    """Etat global de l'application avec composants v2.0 initialises"""
    
    def __init__(self):
        # Configuration
        self.config_manager: Optional[ConfigManager] = None
        
        # Phase 2: Template Engine
        self.template_engine: Optional[TemplateEngine] = None
        
        # Phase 3: Agents Logiques
        self.context_manager: Optional[ContextManager] = None
        self.query_builder: Optional[QueryBuilder] = None
        self.query_executor: Optional[QueryExecutor] = None
        
        # Phase 4: Agents LLM
        self.llm_provider_manager: Optional[LLMProviderManager] = None
        self.intent_classifier: Optional[IntentClassifier] = None
        self.response_generator: Optional[ResponseGenerator] = None
        
        # Phase 5: Orchestrateur
        self.conversation_orchestrator: Optional[ConversationOrchestrator] = None
        
        # Legacy components
        self.deepseek_client: Optional[DeepSeekClient] = None
        self.cache_manager: Optional[CacheManager] = None
        
        # Etat d'initialisation
        self.initialized = False
        self.initialization_error: Optional[str] = None
    
    async def initialize(self) -> bool:
        """Initialise tous les composants du pipeline v2.0"""
        
        try:
            logger.info("Demarrage initialisation pipeline v2.0...")
            
            # === PHASE 1: CONFIGURATION ===
            self.config_manager = ConfigManager()
            await self.config_manager.load_configurations()
            
            # === PHASE 2: TEMPLATE ENGINE ===
            from pathlib import Path
            self.template_engine = TemplateEngine()  # Utilise le chemin par défaut avec /query/
            await self.template_engine.initialize()
            logger.info("OK Template Engine initialise")
            
            # === PHASE 3: AGENTS LOGIQUES ===
            
            # Context Manager
            self.context_manager = ContextManager(
                max_context_turns=50,
                max_total_tokens=8000,
                context_ttl_hours=24
            )
            
            # Query Builder
            self.query_builder = QueryBuilder(
                template_engine=self.template_engine,
                config_manager=self.config_manager
            )
            
            # Query Executor
            from config_service.config import settings
            search_service_url = settings.SEARCH_SERVICE_URL
            self.query_executor = QueryExecutor(
                search_service_url=search_service_url,
                timeout_seconds=10,
                max_concurrent_requests=50
            )
            await self.query_executor.initialize()
            logger.info(f"✅ Query Executor configured with search_service_url: {search_service_url}")
            
            logger.info("OK Agents Logiques initialises")
            
            # === PHASE 4: AGENTS LLM ===
            
            llm_config = self.config_manager.get_llm_providers_config()
            provider_configs = {}
            
            # Access providers from the structure returned by ConfigManager
            providers = llm_config.get("providers", {})
            
            # DeepSeek
            if providers.get("deepseek", {}).get("enabled", False):
                deepseek_config = providers["deepseek"]
                provider_configs[ProviderType.DEEPSEEK] = ProviderConfig(
                    api_key=deepseek_config.get("api_key", ""),
                    base_url=deepseek_config.get("base_url", "https://api.deepseek.com/v1"),
                    models=[deepseek_config.get("model", "deepseek-chat")],
                    capabilities=[],
                    rate_limit_rpm=deepseek_config.get("rate_limit", 60),
                    priority=deepseek_config.get("priority", 1)
                )
            
            # OpenAI
            if providers.get("openai", {}).get("enabled", False):
                openai_config = providers["openai"]
                provider_configs[ProviderType.OPENAI] = ProviderConfig(
                    api_key=openai_config.get("api_key", ""),
                    base_url=openai_config.get("base_url", "https://api.openai.com/v1"),
                    models=[openai_config.get("model", "gpt-3.5-turbo")],
                    capabilities=[],
                    rate_limit_rpm=openai_config.get("rate_limit", 60),
                    priority=openai_config.get("priority", 2)
                )
            
            # Local/Ollama
            if providers.get("local", {}).get("enabled", False):
                local_config = providers["local"]
                provider_configs[ProviderType.LOCAL] = ProviderConfig(
                    api_key="",
                    base_url=local_config.get("base_url", "http://localhost:11434"),
                    models=[local_config.get("model", "llama2")],
                    capabilities=[],
                    rate_limit_rpm=0,
                    priority=local_config.get("priority", 3)
                )
            
            if not provider_configs:
                # Configuration par defaut pour developpement
                provider_configs[ProviderType.LOCAL] = ProviderConfig(
                    api_key="",
                    base_url="http://localhost:11434",
                    models=["llama2"],
                    capabilities=[],
                    rate_limit_rpm=0,
                    priority=1
                )
                logger.warning("Aucun provider LLM configure, utilisation du provider local par defaut")
            
            # LLM Provider Manager
            self.llm_provider_manager = LLMProviderManager(provider_configs)
            await self.llm_provider_manager.initialize()
            
            # Intent Classifier
            self.intent_classifier = IntentClassifier(
                llm_manager=self.llm_provider_manager,
                few_shot_examples_path=None
            )
            await self.intent_classifier.initialize()
            
            # Response Generator
            task_configs = llm_config.get("task_configs", {})
            response_config = task_configs.get("response_generation", {})

            self.response_generator = ResponseGenerator(
                llm_manager=self.llm_provider_manager,
                response_templates_path=None,
                model=response_config.get("model", "deepseek-chat"),
                max_tokens=response_config.get("max_tokens", 8000),
                temperature=response_config.get("temperature", 0.7),
                enable_analytics=True,  # Sprint 1.1: Analytics Agent
                enable_visualizations=True  # Sprint 1.3: Visualizations
            )
            await self.response_generator.initialize()
            
            logger.info("OK Agents LLM initialises")
            
            # === PHASE 5: ORCHESTRATEUR ===
            
            self.conversation_orchestrator = ConversationOrchestrator(
                context_manager=self.context_manager,
                intent_classifier=self.intent_classifier,
                query_builder=self.query_builder,
                query_executor=self.query_executor,
                response_generator=self.response_generator,
                config_manager=self.config_manager
            )
            
            await self.conversation_orchestrator.initialize()
            
            logger.info("OK Orchestrateur initialise")
            
            self.initialized = True
            logger.info("OK Pipeline v2.0 completement initialise !")
            
            return True
            
        except Exception as e:
            self.initialization_error = str(e)
            logger.error(f"ERROR Erreur initialisation pipeline: {str(e)}")
            return False
    
    async def close(self):
        """Ferme proprement tous les composants"""
        
        try:
            if self.conversation_orchestrator:
                await self.conversation_orchestrator.close()
                
            if self.query_executor:
                await self.query_executor.close()
                
            if self.llm_provider_manager:
                await self.llm_provider_manager.close()
                
            logger.info("Application fermee proprement")
            
        except Exception as e:
            logger.error(f"Erreur fermeture application: {str(e)}")

# Instance globale de l'etat de l'application
app_state = ApplicationState()

@asynccontextmanager
async def get_application_lifespan():
    """Context manager pour le cycle de vie de l'application"""
    
    # Startup
    initialization_success = await app_state.initialize()
    
    if not initialization_success:
        logger.error(f"ERROR Echec initialisation: {app_state.initialization_error}")
        raise RuntimeError(f"Application initialization failed: {app_state.initialization_error}")
    
    try:
        yield app_state
    finally:
        # Shutdown
        await app_state.close()

# === DEPENDENCY INJECTION FUNCTIONS v2.0 ===

def get_config_manager() -> ConfigManager:
    """Injection ConfigManager"""
    if not app_state.initialized or not app_state.config_manager:
        raise RuntimeError("ConfigManager not initialized")
    return app_state.config_manager

def get_template_engine() -> TemplateEngine:
    """Injection TemplateEngine"""
    if not app_state.initialized or not app_state.template_engine:
        raise RuntimeError("TemplateEngine not initialized")
    return app_state.template_engine

def get_context_manager() -> ContextManager:
    """Injection ContextManager"""
    if not app_state.initialized or not app_state.context_manager:
        raise RuntimeError("ContextManager not initialized")
    return app_state.context_manager

def get_query_builder() -> QueryBuilder:
    """Injection QueryBuilder"""
    if not app_state.initialized or not app_state.query_builder:
        raise RuntimeError("QueryBuilder not initialized")
    return app_state.query_builder

def get_query_executor() -> QueryExecutor:
    """Injection QueryExecutor"""
    if not app_state.initialized or not app_state.query_executor:
        raise RuntimeError("QueryExecutor not initialized")
    return app_state.query_executor

def get_llm_provider_manager() -> LLMProviderManager:
    """Injection LLMProviderManager"""
    if not app_state.initialized or not app_state.llm_provider_manager:
        raise RuntimeError("LLMProviderManager not initialized")
    return app_state.llm_provider_manager

def get_intent_classifier() -> IntentClassifier:
    """Injection IntentClassifier"""
    if not app_state.initialized or not app_state.intent_classifier:
        raise RuntimeError("IntentClassifier not initialized")
    return app_state.intent_classifier

def get_response_generator() -> ResponseGenerator:
    """Injection ResponseGenerator"""
    if not app_state.initialized or not app_state.response_generator:
        raise RuntimeError("ResponseGenerator not initialized")
    return app_state.response_generator

def get_conversation_orchestrator() -> ConversationOrchestrator:
    """Injection ConversationOrchestrator - Composant principal"""
    if not app_state.initialized or not app_state.conversation_orchestrator:
        raise RuntimeError("ConversationOrchestrator not initialized")
    return app_state.conversation_orchestrator

# === HEALTH CHECK DEPENDENCIES v2.0 ===

async def get_application_health() -> Dict[str, Any]:
    """Health check complet de l'application v2.0"""
    
    if not app_state.initialized:
        return {
            "status": "unhealthy",
            "error": "Application not initialized",
            "initialization_error": app_state.initialization_error
        }
    
    try:
        # Health check de l'orchestrateur (qui check tous les composants)
        return await app_state.conversation_orchestrator.health_check()
        
    except Exception as e:
        return {
            "status": "unhealthy", 
            "error": f"Health check failed: {str(e)}"
        }

async def get_pipeline_stats() -> Dict[str, Any]:
    """Statistiques completes du pipeline v2.0"""
    
    if not app_state.initialized or not app_state.conversation_orchestrator:
        return {"error": "Pipeline not initialized"}
    
    return app_state.conversation_orchestrator.get_pipeline_stats()

def require_initialization():
    """Decorator pour verifier l'initialisation avant execution"""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not app_state.initialized:
                raise RuntimeError("Application components not initialized")
            return func(*args, **kwargs)
        return wrapper
    return decorator

# === LEGACY DEPENDENCIES (COMPATIBILITE RETROGRADE) ===

async def get_deepseek_client(request: Request) -> Optional[DeepSeekClient]:
    """Injection DeepSeek client (legacy)"""
    
    # Priorite v2.0: utiliser le nouveau provider manager
    if app_state.initialized and app_state.llm_provider_manager:
        logger.debug("Using v2.0 LLM Provider Manager instead of legacy DeepSeek client")
        return None  # Indique l'utilisation du nouveau systeme
    
    # Fallback: legacy deepseek client
    if hasattr(request.app.state, 'deepseek_client'):
        return request.app.state.deepseek_client
    
    return None

async def get_cache_manager(request: Request) -> Optional[CacheManager]:
    """Injection cache manager (legacy)"""
    
    # Priorite v2.0: utiliser le cache du nouveau systeme
    if app_state.initialized and app_state.template_engine:
        logger.debug("Cache managed by v2.0 TemplateEngine")
        return None  # Le cache est gere par le nouveau systeme
    
    # Fallback: legacy cache manager
    if hasattr(request.app.state, 'cache_manager'):
        return request.app.state.cache_manager
    
    return None

async def get_conversation_service_status(request: Request) -> Dict[str, Any]:
    """Status du service conversation"""
    
    # Priorite v2.0: utiliser le nouvel orchestrateur
    if app_state.initialized and app_state.conversation_orchestrator:
        try:
            health = await app_state.conversation_orchestrator.health_check()
            return {
                "status": health.get("status", "healthy"),
                "service": "conversation_service_v2",
                "version": "2.0",
                "architecture": "complete",
                "pipeline_stages": 5,
                "components_healthy": health.get("components_healthy", 0),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as v2_error:
            logger.warning(f"v2.0 health check failed, fallback to legacy: {v2_error}")
    
    # Fallback: legacy service status
    if hasattr(request.app.state, 'conversation_service'):
        return {
            "status": "degraded",
            "service": "conversation_service",
            "version": "1.0",
            "architecture": "legacy",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    return {
        "status": "unknown",
        "service": "conversation_service",
        "error": "Service non initialise",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

async def validate_path_user_id(
    request: Request,
    path_user_id: int,
    token_user_id: int = Depends(get_current_user_id)
) -> int:
    """Validation user_id du path vs token JWT"""
    
    try:
        if path_user_id <= 0:
            metrics_collector.increment_counter("dependencies.validation.invalid_user_id")
            logger.warning(f"User ID invalide dans l'URL: {path_user_id}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "User ID invalide dans l'URL",
                    "provided_user_id": path_user_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        
        # Vérification correspondance avec token
        await verify_user_id_match(request, path_user_id)
        
        metrics_collector.increment_counter("dependencies.validation.success")
        return path_user_id
        
    except HTTPException:
        metrics_collector.increment_counter("dependencies.validation.failed")
        raise
    except Exception as e:
        metrics_collector.increment_counter("dependencies.errors.validation_unexpected")
        logger.error(f"Erreur inattendue validation user ID: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Erreur interne validation utilisateur",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

async def get_user_context(
    request: Request,
    user_id: int = Depends(get_current_user_id)
) -> Dict[str, Any]:
    """Récupération contexte utilisateur"""
    
    try:
        return {
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": f"req_{int(time.time())}_{user_id}",
            "client_ip": getattr(request.client, 'host', 'unknown') if request.client else 'unknown',
            "user_agent": request.headers.get("user-agent", "unknown")[:100],
            "method": request.method,
            "path": request.url.path
        }
        
    except Exception as e:
        logger.error(f"Erreur génération contexte utilisateur: {str(e)}")
        return {
            "user_id": user_id,
            "error": "Contexte minimal généré",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# Simple rate limiting (basic implementation)
class SimpleRateLimit:
    def __init__(self):
        self.requests = {}
    
    async def __call__(self, request: Request, user_id: int = Depends(get_current_user_id)):
        # Implementation basique - peut etre enrichie
        current_time = time.time()
        self.requests[user_id] = current_time
        return None

rate_limit_dependency = SimpleRateLimit()

async def get_conversation_persistence(request: Request) -> Optional[ConversationPersistenceService]:
    """Service persistence conversations"""
    
    try:
        from db_service.session import SessionLocal
        db_session = SessionLocal()
        return ConversationPersistenceService(db_session)
        
    except Exception as e:
        logger.debug(f"Persistence service non disponible: {str(e)}")
        return None

# === LEGACY COMPATIBILITY ===

# Fonctions legacy pour compatibilite
async def get_multi_agent_team(request: Request) -> None:
    """DEPRECATED: Multi-agent team functionality removed"""
    logger.debug("Multi-agent team deprecated - using v2.0 orchestrator")
    return None

async def get_conversation_processor(request: Request) -> Dict[str, Any]:
    """Processeur conversation avec choix automatique entre modes"""
    
    # Priorite v2.0: utiliser le nouvel orchestrateur
    if app_state.initialized and app_state.conversation_orchestrator:
        return {
            "processing_mode": "conversation_orchestrator_v2",
            "conversation_orchestrator": app_state.conversation_orchestrator,
            "capabilities": {
                "intent_classification": True,
                "entity_extraction": True,
                "context_management": True,
                "query_building": True,
                "search_execution": True,
                "response_generation": True,
                "streaming": True,
                "websocket": True
            },
            "architecture": "v2.0"
        }
    
    # Fallback legacy
    return {
        "processing_mode": "legacy",
        "capabilities": {"basic": True},
        "architecture": "legacy"
    }

# Export des principales dependances
__all__ = [
    # === v2.0 Dependencies ===
    'get_config_manager',
    'get_template_engine',
    'get_context_manager', 
    'get_query_builder',
    'get_query_executor',
    'get_llm_provider_manager',
    'get_intent_classifier',
    'get_response_generator',
    'get_conversation_orchestrator',
    'get_application_health',
    'get_pipeline_stats',
    'get_application_lifespan',
    'app_state',
    
    # === Legacy Dependencies ===
    'get_deepseek_client',
    'get_cache_manager', 
    'get_conversation_service_status',
    'validate_path_user_id',
    'get_user_context',
    'rate_limit_dependency',
    'get_multi_agent_team',
    'get_conversation_processor',
    'get_conversation_persistence'
]