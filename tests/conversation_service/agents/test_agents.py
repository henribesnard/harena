import sys
import types
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
from types import SimpleNamespace
from collections import deque
import asyncio
import os
import importlib

import pytest

# Ensure repository root on path and import base package
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT_DIR)
importlib.import_module("conversation_service")

# --- Stub external dependencies to avoid heavy imports ---

# Stub autogen
autogen_mod = types.ModuleType("autogen")
class AssistantAgent:  # minimal stub
    def __init__(self, name=None, **kwargs):
        self.name = name
autogen_mod.AssistantAgent = AssistantAgent
sys.modules["autogen"] = autogen_mod

# Stub DeepSeekClient
core_ds = types.ModuleType("conversation_service.core.deepseek_client")
class DeepSeekClient: ...
core_ds.DeepSeekClient = DeepSeekClient
sys.modules["conversation_service.core.deepseek_client"] = core_ds

# Stub agent_models
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
    agent_name: str = ""
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    token_usage: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = None

@dataclass
class TeamWorkflow:
    agents: List[str] = field(default_factory=list)

agent_models.AgentConfig = AgentConfig
agent_models.AgentResponse = AgentResponse
agent_models.TeamWorkflow = TeamWorkflow
sys.modules["conversation_service.models.agent_models"] = agent_models

# Stub financial_models
financial_models = types.ModuleType("conversation_service.models.financial_models")
class EntityType(str, Enum):
    MERCHANT = "MERCHANT"
    CATEGORY = "CATEGORY"
    DATE_RANGE = "DATE_RANGE"
    AMOUNT = "AMOUNT"
    TRANSACTION_TYPE = "TRANSACTION_TYPE"
    CURRENCY = "CURRENCY"
    DATE = "DATE"
    OTHER = "OTHER"
class IntentCategory(str, Enum):
    TRANSACTION_SEARCH = "TRANSACTION_SEARCH"
    GENERAL_QUESTION = "GENERAL_QUESTION"
class DetectionMethod(str, Enum):
    AI_DETECTION = "AI_DETECTION"
    FALLBACK = "FALLBACK"
    AI_ERROR_FALLBACK = "AI_ERROR_FALLBACK"
    AI_PARSE_FALLBACK = "AI_PARSE_FALLBACK"
    LLM_BASED = "LLM_BASED"
@dataclass
class FinancialEntity:
    entity_type: Any
    raw_value: Any
    normalized_value: Any
    confidence: float
    start_position: Optional[int] = None
    end_position: Optional[int] = None
    detection_method: Any = None
    def model_dump(self):
        return {
            "entity_type": getattr(self.entity_type, "value", self.entity_type),
            "raw_value": self.raw_value,
            "normalized_value": self.normalized_value,
            "confidence": self.confidence,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "detection_method": getattr(self.detection_method, "value", self.detection_method),
        }
@dataclass
class IntentResult:
    intent_type: str
    intent_category: Any
    confidence: float
    entities: List[Any] = field(default_factory=list)
    method: Any = None
    processing_time_ms: float = 0.0
    def model_dump(self):
        return {
            "intent_type": self.intent_type,
            "intent_category": getattr(self.intent_category, "value", self.intent_category),
            "confidence": self.confidence,
            "entities": [e.model_dump() for e in self.entities],
            "method": getattr(self.method, "value", self.method),
            "processing_time_ms": self.processing_time_ms,
        }
financial_models.EntityType = EntityType
financial_models.IntentCategory = IntentCategory
financial_models.DetectionMethod = DetectionMethod
financial_models.FinancialEntity = FinancialEntity
financial_models.IntentResult = IntentResult
sys.modules["conversation_service.models.financial_models"] = financial_models

# Stub conversation_models
conv_models = types.ModuleType("conversation_service.models.conversation_models")
@dataclass
class ConversationTurn:
    user_message: str = ""
    assistant_response: str = ""
    turn_number: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def model_dump(self):
        return {
            "user_message": self.user_message,
            "assistant_response": self.assistant_response,
            "turn_number": self.turn_number,
            "metadata": self.metadata,
        }

@dataclass
class ConversationContext:
    conversation_id: str
    user_id: int
    turns: List[Any] = field(default_factory=list)
    current_turn: int = 0
    status: str = "active"
    language: str = "fr"
    context_summary: Optional[str] = None
    active_entities: Optional[List[str]] = None

    def model_dump(self):
        return {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "turns": [t.model_dump() if hasattr(t, "model_dump") else t for t in self.turns],
            "current_turn": self.current_turn,
            "status": self.status,
            "language": self.language,
            "context_summary": self.context_summary,
            "active_entities": self.active_entities,
        }

@dataclass
class ConversationOut:
    conversation_id: str
    title: Optional[str] = None
    status: str = "active"
    total_turns: int = 0
    last_activity_at: Optional[Any] = None

    def model_dump(self):
        return {
            "conversation_id": self.conversation_id,
            "title": self.title,
            "status": self.status,
            "total_turns": self.total_turns,
            "last_activity_at": self.last_activity_at,
        }

@dataclass
class ConversationTurnsResponse:
    conversation_id: str
    turns: List[ConversationTurn] = field(default_factory=list)

    def model_dump(self):
        return {
            "conversation_id": self.conversation_id,
            "turns": [t.model_dump() for t in self.turns],
        }

conv_models.ConversationTurn = ConversationTurn
conv_models.ConversationContext = ConversationContext
conv_models.ConversationOut = ConversationOut
conv_models.ConversationTurnsResponse = ConversationTurnsResponse
sys.modules["conversation_service.models.conversation_models"] = conv_models

# Stub service_contracts
service_contracts = types.ModuleType("conversation_service.models.service_contracts")
@dataclass
class QueryMetadata:
    conversation_id: str
    user_id: int
    intent_type: str
    language: str
    priority: str
    source_agent: Optional[str] = None
@dataclass
class SearchParameters:
    search_text: str
    max_results: int
    include_highlights: bool
    boost_recent: bool
    fuzzy_matching: bool
@dataclass
class SearchFilters:
    categories: List[str] = field(default_factory=list)
    merchants: List[str] = field(default_factory=list)
    user_id: Optional[int] = None
    date_from: Any = None
    date_to: Any = None
    month_year: Any = None
    amount_min: Any = None
    amount_max: Any = None
@dataclass
class SearchServiceQuery:
    query_metadata: QueryMetadata
    search_parameters: SearchParameters
    filters: SearchFilters
    def dict(self):
        return {
            "query_metadata": self.query_metadata.__dict__,
            "search_parameters": self.search_parameters.__dict__,
            "filters": self.filters.__dict__,
        }
@dataclass
class SearchServiceResponse:
    response_metadata: Dict[str, Any] = field(default_factory=dict)
    def dict(self):
        return {"response_metadata": self.response_metadata}
service_contracts.QueryMetadata = QueryMetadata
service_contracts.SearchParameters = SearchParameters
service_contracts.SearchFilters = SearchFilters
service_contracts.SearchServiceQuery = SearchServiceQuery
service_contracts.SearchServiceResponse = SearchServiceResponse
sys.modules["conversation_service.models.service_contracts"] = service_contracts

# Stub validators
validators_mod = types.ModuleType("conversation_service.utils.validators")
class ContractValidator:
    def validate_search_query(self, *args, **kwargs):
        return []
validators_mod.ContractValidator = ContractValidator
sys.modules["conversation_service.utils.validators"] = validators_mod

# -----------------------------------------------------------------


@pytest.fixture
def agent_classes(httpx_stub):
    from conversation_service.agents.hybrid_intent_agent import HybridIntentAgent
    from conversation_service.agents.search_query_agent import SearchQueryAgent
    from conversation_service.agents.response_agent import ResponseAgent
    from conversation_service.agents.orchestrator_agent import OrchestratorAgent, WorkflowExecutor
    from conversation_service.agents.base_financial_agent import BaseFinancialAgent

    return {
        "HybridIntentAgent": HybridIntentAgent,
        "SearchQueryAgent": SearchQueryAgent,
        "ResponseAgent": ResponseAgent,
        "OrchestratorAgent": OrchestratorAgent,
        "WorkflowExecutor": WorkflowExecutor,
        "BaseFinancialAgent": BaseFinancialAgent,
    }


class MockDeepSeekClient:
    api_key = "test"
    base_url = "http://deepseek.test"

    async def generate_response(self, *args, **kwargs):
        return SimpleNamespace(content="")


def test_hybrid_intent_agent_ai_parsing(agent_classes):
    HybridIntentAgent = agent_classes["HybridIntentAgent"]
    agent = HybridIntentAgent.__new__(HybridIntentAgent)
    ai_content = (
        "Intention: TRANSACTION_SEARCH\n"
        "Confiance: 0.92\n"
        "Entités: {\"merchant\": \"Amazon\", \"amount\": 23.5}"
    )
    result = HybridIntentAgent._parse_ai_response(agent, ai_content, "msg")
    assert result.intent_type == "TRANSACTION_SEARCH"
    ents = {e.entity_type: e for e in result.entities}
    assert ents[EntityType.MERCHANT].normalized_value == "Amazon"
    assert ents[EntityType.AMOUNT].normalized_value == 23.5


def test_search_query_agent_filters_generation(agent_classes):
    SearchQueryAgent = agent_classes["SearchQueryAgent"]
    agent = SearchQueryAgent(MockDeepSeekClient(), "http://search.test")

    entities = [
        FinancialEntity(EntityType.MERCHANT, "Amazon", "Amazon", 0.9),
        FinancialEntity(
            EntityType.DATE_RANGE,
            "janvier",
            {"start_date": "2024-01-01", "end_date": "2024-01-31"},
            0.9,
        ),
        FinancialEntity(EntityType.AMOUNT, "20", 20.0, 0.9),
    ]
    intent_result = IntentResult(
        intent_type="TRANSACTION_SEARCH",
        intent_category=IntentCategory.TRANSACTION_SEARCH,
        confidence=0.9,
        entities=entities,
        method=DetectionMethod.AI_DETECTION,
        processing_time_ms=1.0,
    )

    query = asyncio.run(agent._generate_search_contract(intent_result, "message", user_id=5))

    assert query.filters.merchants == ["Amazon"]
    assert query.filters.date_from == "2024-01-01"
    assert query.filters.date_to == "2024-01-31"
    assert query.filters.amount_min < 20.0 < query.filters.amount_max
    assert query.filters.user_id == 5


def test_response_agent_format_message(agent_classes, monkeypatch):
    ResponseAgent = agent_classes["ResponseAgent"]

    class FakeSearchServiceResponse:
        def __init__(self, **data):
            self.response_metadata = SimpleNamespace(**data.get("response_metadata", {}))
            self.results = data.get("results", [])
            agg = data.get("aggregations")
            self.aggregations = SimpleNamespace(**agg) if agg else None

    monkeypatch.setattr(
        "conversation_service.agents.response_agent.SearchServiceResponse",
        FakeSearchServiceResponse,
    )

    agent = ResponseAgent(MockDeepSeekClient())

    async def fake_ai(self, user_message, formatted_results, conversation_context, user_id):
        return "Réponse générée"

    agent._generate_ai_response = fake_ai.__get__(agent, ResponseAgent)

    search_results = {
        "metadata": {
            "search_response": {
                "response_metadata": {"returned_results": 2, "processing_time_ms": 5},
                "results": [
                    SimpleNamespace(
                        source={
                            "date": "2024-01-05",
                            "amount": -50,
                            "merchant_name": "Store",
                            "category_name": "food",
                        }
                    ),
                    SimpleNamespace(
                        source={
                            "date": "2024-01-10",
                            "amount": -20,
                            "merchant_name": "Cafe",
                            "category_name": "food",
                        }
                    ),
                ],
                "aggregations": {
                    "transaction_count": 2,
                    "total_amount": -70,
                    "average_amount": -35,
                    "date_range": {"start_date": "2024-01-01", "end_date": "2024-01-31"},
                    "category_breakdown": [
                        {"category": "food", "count": 2, "total_amount": -70}
                    ],
                },
            }
        }
    }

    response = asyncio.run(agent.generate_response("résumé", search_results, user_id=1))
    assert response["content"] == "Réponse générée"
    assert "transactions trouvées" in response["metadata"]["formatted_results"]


def test_orchestrator_agent_pipeline(agent_classes):
    OrchestratorAgent = agent_classes["OrchestratorAgent"]
    WorkflowExecutor = agent_classes["WorkflowExecutor"]
    intent_result = IntentResult(
        intent_type="TRANSACTION_SEARCH",
        intent_category=IntentCategory.TRANSACTION_SEARCH,
        confidence=0.9,
        entities=[],
        method=DetectionMethod.AI_DETECTION,
        processing_time_ms=1.0,
    )
    intent_response = AgentResponse(
        agent_name="intent_agent",
        content="",
        metadata={"intent_result": intent_result},
        execution_time_ms=1.0,
        success=True,
    )
    search_response = AgentResponse(
        agent_name="search_query_agent",
        content="",
        metadata={
            "search_results_count": 0,
            "search_response": {},
        },
        execution_time_ms=1.0,
        success=True,
    )
    final_response = AgentResponse(
        agent_name="response_agent",
        content="Réponse finale",
        metadata={},
        execution_time_ms=1.0,
        success=True,
    )

    class StubAgent:
        def __init__(self, name, response):
            self.name = name
            self.deepseek_client = MockDeepSeekClient()
            self._response = response

        async def execute_with_metrics(self, *args, **kwargs):
            return self._response

        def is_healthy(self):
            return True

    intent_agent = StubAgent("intent_agent", intent_response)
    search_agent = StubAgent("search_query_agent", search_response)
    response_agent = StubAgent("response_agent", final_response)

    orchestrator = OrchestratorAgent.__new__(OrchestratorAgent)
    orchestrator.intent_agent = intent_agent
    orchestrator.search_agent = search_agent
    orchestrator.response_agent = response_agent
    orchestrator.workflow_executor = WorkflowExecutor(intent_agent, search_agent, response_agent)
    orchestrator.performance_threshold_ms = 30000
    orchestrator.recent_workflow_times = deque(maxlen=1000)
    orchestrator.workflow_stats = {
        "total_workflows": 0,
        "successful_workflows": 0,
        "failed_workflows": 0,
        "avg_workflow_time_ms": 0.0,
        "p95_workflow_time_ms": 0.0,
        "p99_workflow_time_ms": 0.0,
        "step_success_rates": {
            "intent_detection": 0.0,
            "search_query": 0.0,
            "response_generation": 0.0,
        },
    }

    result = asyncio.run(orchestrator.process_conversation("bonjour", "conv1", user_id=1))
    assert result["content"] == "Réponse finale"
    assert result["metadata"]["workflow_success"] is True
    assert result["metadata"]["agent_chain"] == [
        "orchestrator_agent",
        "intent_agent",
        "search_query_agent",
        "response_agent",
    ]


def test_base_financial_agent_metrics_and_health(agent_classes):
    BaseFinancialAgent = agent_classes["BaseFinancialAgent"]

    class DummyAgent(BaseFinancialAgent):
        def __init__(self):
            config = AgentConfig(
                name="dummy",
                model_client_config={"model": "m", "api_key": "k", "base_url": "u"},
                system_message="sys",
                max_consecutive_auto_reply=1,
                description="",
                temperature=0.0,
                max_tokens=10,
                timeout_seconds=1,
            )
            super().__init__("dummy", config, MockDeepSeekClient())

        async def _execute_operation(self, input_data, user_id):
            return {"content": "ok"}

    agent = DummyAgent()
    asyncio.run(agent.execute_with_metrics({}, user_id=1))
    assert agent.metrics.total_operations == 1
    assert agent.is_healthy() is True

    agent.metrics.record_operation(31000, success=False, error_type="Timeout")
    assert agent.is_healthy() is False
