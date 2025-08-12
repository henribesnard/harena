import sys
import types
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
from types import SimpleNamespace
from importlib.machinery import ModuleSpec
from pathlib import Path
import importlib.util
import importlib
import asyncio

import pytest

# Stub httpx to avoid dependency during import
sys.modules.setdefault("httpx", types.ModuleType("httpx"))

# Stub pydantic to avoid dependency during import
pydantic_stub = types.ModuleType("pydantic")

class BaseModel:
    ...


def Field(*args, **kwargs):
    return None


def field_validator(*args, **kwargs):
    def decorator(func):
        return func
    return decorator


def model_validator(*args, **kwargs):
    def decorator(func):
        return func
    return decorator


class ValidationError(Exception):
    ...


pydantic_stub.BaseModel = BaseModel
pydantic_stub.Field = Field
pydantic_stub.field_validator = field_validator
pydantic_stub.model_validator = model_validator
pydantic_stub.ValidationError = ValidationError

sys.modules.setdefault("pydantic", pydantic_stub)

# Ensure repository root is first on sys.path so we import the real package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conversation_service  # Ensure parent package is loaded for module stubbing

# Save original DeepSeek client module to restore after tests
_original_ds_module = sys.modules.get("conversation_service.core.deepseek_client")

# Stub DeepSeekClient and related errors
core_ds = types.ModuleType("conversation_service.core.deepseek_client")

class DeepSeekClient:
    ...


class DeepSeekError(Exception):
    ...


class DeepSeekTimeoutError(DeepSeekError):
    ...


core_ds.DeepSeekClient = DeepSeekClient
core_ds.DeepSeekError = DeepSeekError
core_ds.DeepSeekTimeoutError = DeepSeekTimeoutError

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
    success: bool = True
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
agent_models.AgentConfig = AgentConfig
agent_models.AgentResponse = AgentResponse
sys.modules["conversation_service.models.agent_models"] = agent_models

# Stub financial_models
financial_models = types.ModuleType("conversation_service.models.financial_models")
class EntityType(str, Enum):
    MERCHANT = "MERCHANT"
    CATEGORY = "CATEGORY"
    DATE_RANGE = "DATE_RANGE"
    AMOUNT = "AMOUNT"
@dataclass
class FinancialEntity:
    entity_type: Any
    normalized_value: Any
@dataclass
class IntentResult:
    intent_type: str
    entities: List[Any] = field(default_factory=list)
financial_models.EntityType = EntityType
financial_models.FinancialEntity = FinancialEntity
financial_models.IntentResult = IntentResult
sys.modules["conversation_service.models.financial_models"] = financial_models

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
    query_id: str = "q"
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

# Stub models package to avoid importing actual models (e.g., pydantic dependency)
models_pkg = types.ModuleType("conversation_service.models")
models_pkg.__path__ = []  # Mark as package
models_pkg.__spec__ = ModuleSpec("conversation_service.models", loader=None, is_package=True)
models_pkg.agent_models = agent_models
models_pkg.financial_models = financial_models
models_pkg.service_contracts = service_contracts
sys.modules["conversation_service.models"] = models_pkg
conversation_service.models = models_pkg

# Stub validators
validators_mod = types.ModuleType("conversation_service.utils.validators")
class ContractValidator:
    def validate_search_query(self, *args, **kwargs):
        return []
validators_mod.ContractValidator = ContractValidator
sys.modules["conversation_service.utils.validators"] = validators_mod

@pytest.fixture(scope="module", autouse=True)
def restore_deepseek_client_module():
    yield
    if _original_ds_module is not None:
        sys.modules["conversation_service.core.deepseek_client"] = _original_ds_module
    else:
        sys.modules.pop("conversation_service.core.deepseek_client", None)


# Import SearchQueryAgent and dependencies directly from file to avoid package path issues
agents_pkg = types.ModuleType("conversation_service.agents")
agents_pkg.__path__ = []
sys.modules["conversation_service.agents"] = agents_pkg

base_spec = importlib.util.spec_from_file_location(
    "conversation_service.agents.base_financial_agent",
    Path(__file__).resolve().parents[2]
    / "conversation_service"
    / "agents"
    / "base_financial_agent.py",
)
base_module = importlib.util.module_from_spec(base_spec)
base_spec.loader.exec_module(base_module)
sys.modules["conversation_service.agents.base_financial_agent"] = base_module
agents_pkg.base_financial_agent = base_module

spec = importlib.util.spec_from_file_location(
    "conversation_service.agents.search_query_agent",
    Path(__file__).resolve().parents[2]
    / "conversation_service"
    / "agents"
    / "search_query_agent.py",
)
search_query_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(search_query_module)
sys.modules["conversation_service.agents.search_query_agent"] = search_query_module
agents_pkg.search_query_agent = search_query_module

QueryOptimizer = search_query_module.QueryOptimizer
SearchQueryAgent = search_query_module.SearchQueryAgent


def test_optimize_search_text_with_string_entities():
    optimizer = QueryOptimizer()
    entities = [
        SimpleNamespace(entity_type="MERCHANT", normalized_value="Amazon"),
        SimpleNamespace(entity_type="CATEGORY", normalized_value="Shopping"),
    ]
    intent_result = SimpleNamespace(entities=entities)
    text = optimizer.optimize_search_text("J'ai payé", intent_result)
    assert "Amazon" in text
    assert "Shopping" in text


def test_generate_search_contract_handles_string_entities():
    agent = SearchQueryAgent.__new__(SearchQueryAgent)
    agent.query_optimizer = QueryOptimizer()
    agent.name = "test_agent"

    entities = [
        SimpleNamespace(entity_type="MERCHANT", normalized_value="Amazon"),
        SimpleNamespace(entity_type="CATEGORY", normalized_value="Shopping"),
    ]
    intent_result = SimpleNamespace(intent_type="TRANSACTION_SEARCH", entities=entities)
    query = asyncio.run(agent._generate_search_contract(intent_result, "message", user_id=1))
    assert query.filters.merchants == ["Amazon"]
    assert query.filters.categories == ["Shopping"]
