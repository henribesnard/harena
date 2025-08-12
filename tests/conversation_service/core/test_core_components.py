import os
import sys
import types
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

# Ensure project root on path for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# ---- Stub external dependencies -------------------------------------------------
# Minimal conversation models to avoid heavy pydantic dependency
conv_models = types.ModuleType("conversation_service.models.conversation_models")

@dataclass
class ConversationTurn:
    user_message: str
    assistant_response: str
    turn_number: int
    processing_time_ms: float = 0.0
    intent_detected: Optional[str] = None
    entities_extracted: Optional[List[Dict[str, Any]]] = None
    confidence_score: float = 0.0
    error_occurred: bool = False
    agent_chain: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConversationContext:
    conversation_id: str
    user_id: int
    turns: List[ConversationTurn] = field(default_factory=list)
    current_turn: int = 0
    status: str = "active"
    language: str = "fr"
    context_summary: str = ""
    updated_at: datetime = field(default_factory=datetime.utcnow)
    active_entities: Optional[List[Any]] = None
    domain: str = "financial"

conv_models.ConversationTurn = ConversationTurn
conv_models.ConversationContext = ConversationContext
sys.modules['conversation_service.models.conversation_models'] = conv_models

# Minimal agent models
agent_models = types.ModuleType("conversation_service.models.agent_models")

@dataclass
class AgentConfig:
    name: str = ""
    model_client_config: Dict[str, Any] = field(default_factory=dict)
    system_message: str = ""
    max_consecutive_auto_reply: int = 1
    description: str = ""
    temperature: float = 0.0
    max_tokens: int = 0
    timeout_seconds: int = 0

@dataclass
class AgentResponse:
    success: bool = True
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

agent_models.AgentConfig = AgentConfig
agent_models.AgentResponse = AgentResponse
sys.modules['conversation_service.models.agent_models'] = agent_models

# Stub agent modules used by MVPTeamManager
for module_name, class_name in [
    ('hybrid_intent_agent', 'HybridIntentAgent'),
    ('search_query_agent', 'SearchQueryAgent'),
    ('response_agent', 'ResponseAgent'),
]:
    mod = types.ModuleType(f"conversation_service.agents.{module_name}")
    setattr(mod, class_name, type(class_name, (), {}))
    sys.modules[f'conversation_service.agents.{module_name}'] = mod

# Stub openai module used by DeepSeekClient
openai_mod = types.ModuleType("openai")


class DeepSeekError(Exception):
    pass


class DeepSeekTimeoutError(DeepSeekError):
    pass

class _DummyOpenAI:
    handler = None  # set per test

    def __init__(self, *args, **kwargs):
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create)
        )

    async def _create(self, *args, **kwargs):
        if _DummyOpenAI.handler is None:
            raise NotImplementedError("No handler set for AsyncOpenAI")
        return await _DummyOpenAI.handler(*args, **kwargs)
openai_mod.DeepSeekError = DeepSeekError
openai_mod.DeepSeekTimeoutError = DeepSeekTimeoutError
openai_mod.AsyncOpenAI = _DummyOpenAI

openai_types = types.ModuleType("openai.types")
openai_chat = types.ModuleType("openai.types.chat")

@dataclass
class ChatCompletion:
    usage: Any = None

openai_chat.ChatCompletion = ChatCompletion
openai_types.chat = openai_chat
sys.modules['openai'] = openai_mod
sys.modules['openai.types'] = openai_types
sys.modules['openai.types.chat'] = openai_chat

# Stub httpx module required by deepseek_client
httpx_mod = types.ModuleType("httpx")
class HTTPStatusError(Exception):
    pass
class TimeoutException(Exception):
    pass
httpx_mod.HTTPStatusError = HTTPStatusError
httpx_mod.TimeoutException = TimeoutException
httpx_mod.RequestError = Exception
sys.modules['httpx'] = httpx_mod

# ---- Import the modules under test ---------------------------------------------
from conversation_service.core.conversation_manager import ConversationManager
from conversation_service.core.mvp_team_manager import MVPTeamManager, TeamConfiguration
from conversation_service.core.deepseek_client import (
    DeepSeekClient,
    DeepSeekError,
    DeepSeekTimeoutError,
)

# ---- Tests ---------------------------------------------------------------------

def test_conversation_manager_creates_and_updates_context():
    async def run_test():
        manager = ConversationManager()
        await manager.initialize()

        ctx = await manager.get_context("conv1", user_id=42)
        assert ctx.conversation_id == "conv1"
        assert ctx.current_turn == 0

        await manager.add_turn("conv1", 42, "bonjour", "salut")
        ctx2 = await manager.get_context("conv1", user_id=42)
        assert ctx2.current_turn == 1
        assert ctx2.turns[0].user_message == "bonjour"
        assert ctx2.turns[0].assistant_response == "salut"

    asyncio.run(run_test())


def test_mvp_team_manager_initialization_and_recovery(monkeypatch):
    async def run_test():
        manager = MVPTeamManager()

        async def fake_init_deepseek():
            manager.deepseek_client = object()
        async def fake_init_conv():
            manager.conversation_manager = "ok"
        async def fake_init_agents():
            manager.agents = {}
        async def fake_init_orchestrator():
            manager.orchestrator = object()

        monkeypatch.setattr(manager, "_initialize_deepseek_client", fake_init_deepseek)
        monkeypatch.setattr(manager, "_initialize_conversation_manager", fake_init_conv)
        monkeypatch.setattr(manager, "_initialize_specialized_agents", fake_init_agents)
        monkeypatch.setattr(manager, "_initialize_orchestrator", fake_init_orchestrator)

        await manager.initialize_agents(initial_health_check=False)
        assert manager.is_initialized
        assert manager.conversation_manager == "ok"

        class DummyAgent:
            def __init__(self, healthy):
                self.healthy = healthy
            def is_healthy(self):
                return self.healthy

        agent = DummyAgent(False)
        manager.agents = {"dummy": agent}
        manager.orchestrator = DummyAgent(True)
        manager.failure_threshold = 1

        await manager._perform_health_check()
        assert "dummy" in manager.disabled_agents
        assert not manager.team_health.overall_healthy

        agent.healthy = True
        status = await manager.health_check()
        assert status["healthy"]
        assert "dummy" not in manager.disabled_agents

    asyncio.run(run_test())


def test_deepseek_client_cache_and_timeout(monkeypatch):
    async def run_test():
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
        monkeypatch.setenv("REDIS_CACHE_ENABLED", "false")

        calls: List[int] = []
        async def success_handler(*args, **kwargs):
            calls.append(1)
            usage = types.SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
            return ChatCompletion(usage=usage)

        _DummyOpenAI.handler = success_handler
        client = DeepSeekClient(cache_enabled=True)
        messages = [{"role": "user", "content": "hi"}]
        res1 = await client.create_completion(messages)
        res2 = await client.create_completion(messages)
        assert res1 is res2
        assert len(calls) == 1  # second call used cache

        async def timeout_handler(*args, **kwargs):
            raise asyncio.TimeoutError

        _DummyOpenAI.handler = timeout_handler
        client_no_cache = DeepSeekClient(cache_enabled=False)
        with pytest.raises(DeepSeekTimeoutError):
            await client_no_cache.create_completion(messages)

    import pytest  # local import to avoid global dependency before stubs
    asyncio.run(run_test())
