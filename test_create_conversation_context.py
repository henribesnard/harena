import sys, types
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Stub dependencies to avoid heavy imports
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

# Stub ConversationContext
conversation_models = types.ModuleType("conversation_service.models.conversation_models")

@dataclass
class ConversationContext:
    conversation_id: str
    user_id: int
    turns: List = field(default_factory=list)
    current_turn: int = 0
    status: str = "active"
    language: str = "fr"

conversation_models.ConversationContext = ConversationContext
sys.modules['conversation_service.models.conversation_models'] = conversation_models

# Stub DeepSeekClient
core_ds = types.ModuleType("conversation_service.core.deepseek_client")
class DeepSeekClient: ...
core_ds.DeepSeekClient = DeepSeekClient
sys.modules['conversation_service.core.deepseek_client'] = core_ds

# Stub agent modules (only specific submodules)
for module_name, class_name in [
    ('hybrid_intent_agent', 'HybridIntentAgent'),
    ('search_query_agent', 'SearchQueryAgent'),
    ('response_agent', 'ResponseAgent'),
]:
    mod = types.ModuleType(f"conversation_service.agents.{module_name}")
    setattr(mod, class_name, type(class_name, (), {}))
    sys.modules[f'conversation_service.agents.{module_name}'] = mod

from conversation_service.agents.orchestrator_agent import WorkflowExecutor


def test_conversation_context_starts_at_zero_when_no_turns():
    executor = WorkflowExecutor.__new__(WorkflowExecutor)
    context = executor._create_conversation_context('conv1', 'hello', 1)
    assert context.turns == []
    assert context.current_turn == 0
