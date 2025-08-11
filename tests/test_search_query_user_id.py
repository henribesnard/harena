import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock
import sys
from pathlib import Path
import types

# Ensure project root is on sys.path for module imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Provide a minimal stub for httpx to avoid external dependency
httpx_stub = types.SimpleNamespace(
    AsyncClient=SimpleNamespace,
    HTTPStatusError=Exception,
    RequestError=Exception,
)
sys.modules.setdefault("httpx", httpx_stub)

# Minimal pydantic stub for tests
class _BaseModel:
    def __init__(self, **data):
        for k, v in data.items():
            setattr(self, k, v)

    def dict(self):
        result = {}
        for k, v in self.__dict__.items():
            if hasattr(v, "dict"):
                result[k] = v.dict()
            elif isinstance(v, list):
                result[k] = [item.dict() if hasattr(item, "dict") else item for item in v]
            else:
                result[k] = v
        return result


pydantic_stub = types.SimpleNamespace(
    BaseModel=_BaseModel,
    Field=lambda *args, **kwargs: None,
    field_validator=lambda *args, **kwargs: (lambda f: f),
    model_validator=lambda *args, **kwargs: (lambda f: f),
    ValidationError=Exception,
)
sys.modules.setdefault("pydantic", pydantic_stub)

# Stub for openai client used in DeepSeekClient
openai_types_chat_stub = types.SimpleNamespace(ChatCompletion=SimpleNamespace)
openai_types_stub = types.SimpleNamespace(chat=openai_types_chat_stub)
openai_stub = types.SimpleNamespace(AsyncOpenAI=SimpleNamespace, types=openai_types_stub)
sys.modules.setdefault("openai", openai_stub)
sys.modules.setdefault("openai.types", openai_types_stub)
sys.modules.setdefault("openai.types.chat", openai_types_chat_stub)

from conversation_service.agents.search_query_agent import SearchQueryAgent
from conversation_service.models.financial_models import (
    IntentResult,
    IntentCategory,
    DetectionMethod,
)
from conversation_service.models.service_contracts import (
    SearchServiceResponse,
    ResponseMetadata,
    TransactionResult,
)

import conversation_service.agents.base_financial_agent as base_agent


class DummyAssistantAgent:
    def __init__(self, name: str, *args, **kwargs):
        self.name = name


base_agent.AssistantAgent = DummyAssistantAgent
base_agent.AUTOGEN_AVAILABLE = True


class DummyDeepSeekClient(SimpleNamespace):
    api_key: str = "test"
    base_url: str = "http://test"


def test_search_query_agent_uses_provided_user_id():
    async def run_test():
        client = DummyDeepSeekClient()
        agent = SearchQueryAgent(client, search_service_url="http://search")
        agent.name = "search_query_agent"

        agent._extract_additional_entities = AsyncMock(return_value=[])

        from conversation_service.utils.validators import ContractValidator

        ContractValidator.validate_search_query = lambda self, query: []

        captured_query = {}

        async def fake_execute_search_query(self, query):
            captured_query["query"] = query
            metadata = ResponseMetadata(
                query_id="q1",
                processing_time_ms=1.0,
                total_results=1,
                returned_results=1,
                returned_hits=1,
                has_more_results=False,
                search_strategy_used="lexical",
            )
            tx = TransactionResult(
                transaction_id="t1",
                date="2024-01-01",
                amount=10.0,
                currency="EUR",
                description="test",
                account_id="a1",
                transaction_type="debit",
                metadata={"user_id": query.query_metadata.user_id},
            )
            return SearchServiceResponse(response_metadata=metadata, results=[tx])

        agent._execute_search_query = fake_execute_search_query.__get__(agent, SearchQueryAgent)

        intent = IntentResult(
            intent_type="TRANSACTION_SEARCH",
            intent_category=IntentCategory.TRANSACTION_SEARCH,
            confidence=0.8,
            entities=[],
            method=DetectionMethod.RULE_BASED,
            processing_time_ms=1.0,
        )

        result = await agent.process_search_request(intent, "test message", user_id=42)

        assert captured_query["query"].query_metadata.user_id == 42
        assert (
            result["metadata"]["search_response"]["results"][0]["metadata"]["user_id"]
            == 42
        )

    import asyncio

    asyncio.run(run_test())


def test_execute_search_query_sends_user_id_in_request():
    async def run_test():
        client = DummyDeepSeekClient()
        agent = SearchQueryAgent(client, search_service_url="http://search")
        agent.name = "search_query_agent"

        agent._extract_additional_entities = AsyncMock(return_value=[])

        from conversation_service.utils.validators import ContractValidator

        ContractValidator.validate_search_query = lambda self, query: []

        captured = {}

        class DummyHTTPClient:
            async def post(self, url, json, headers):
                captured["url"] = url
                captured["json"] = json

                class Resp:
                    def raise_for_status(self):
                        pass

                    def json(self):
                        return {
                            "response_metadata": {
                                "query_id": "q1",
                                "processing_time_ms": 1.0,
                                "total_results": 0,
                                "returned_results": 0,
                                "returned_hits": 0,
                                "has_more_results": False,
                                "search_strategy_used": "lexical",
                            },
                            "results": [],
                        }

                return Resp()

        agent.http_client = DummyHTTPClient()

        intent = IntentResult(
            intent_type="TRANSACTION_SEARCH",
            intent_category=IntentCategory.TRANSACTION_SEARCH,
            confidence=0.8,
            entities=[],
            method=DetectionMethod.RULE_BASED,
            processing_time_ms=1.0,
        )

        await agent.process_search_request(intent, "test message", user_id=2)

        assert captured["url"].endswith("/search")
        assert captured["json"]["user_id"] == 2
        assert captured["json"]["filters"]["user_id"] == 2

    import asyncio

    asyncio.run(run_test())
