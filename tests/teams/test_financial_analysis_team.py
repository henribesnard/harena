import json
import sys
import types
import asyncio
import pytest


class DummyAssistantAgent:
    def __init__(self, *args, **kwargs):
        pass

    async def a_generate_reply(self, *args, **kwargs):  # pragma: no cover - stub
        return "{}"

    def add_capability(self, *args, **kwargs):  # pragma: no cover - stub
        pass


class DummyGroupChat:
    def __init__(self, agents=None, messages=None, max_round=None, speaker_selection_method=None):
        self.agents = agents or []
        self.messages = messages or []
        self.max_round = max_round


class DummyGroupChatManager:
    def __init__(self, groupchat, llm_config=None):
        self.groupchat = groupchat
        self.llm_config = llm_config


autogen_module = types.ModuleType("autogen")
autogen_module.GroupChat = DummyGroupChat
autogen_module.GroupChatManager = DummyGroupChatManager
autogen_module.AssistantAgent = DummyAssistantAgent
sys.modules["autogen"] = autogen_module


class DummyCacheManager:
    async def get_semantic_cache(self, *args, **kwargs):
        return None

    async def set_semantic_cache(self, *args, **kwargs):
        pass

    async def health_check(self):
        return {"status": "ok"}


cache_module = types.ModuleType("conversation_service.core.cache_manager")
cache_module.CacheManager = DummyCacheManager
sys.modules["conversation_service.core.cache_manager"] = cache_module


from conversation_service.teams.financial_analysis_team import FinancialAnalysisTeam


class StubCacheManager:
    def __init__(self):
        self.store = {}
        self.metrics = {"hits": 0, "misses": 0}

    async def get_semantic_cache(self, key, cache_type="response"):
        if key in self.store:
            self.metrics["hits"] += 1
            return self.store[key]
        self.metrics["misses"] += 1
        return None

    async def set_semantic_cache(self, key, value, cache_type="response"):
        self.store[key] = value

    async def health_check(self):
        return {"status": "ok", **self.metrics}


class StubMetricsCollector:
    def __init__(self):
        self.histograms = []
        self.counters = {}

    def record_histogram(self, name, value):
        self.histograms.append((name, value))

    def increment_counter(self, name):
        self.counters[name] = self.counters.get(name, 0) + 1


class StubIntentAgent:
    name = "intent_classifier"

    def __init__(self):
        self.success_count = 0
        self.error_count = 0
        self.call_count = 0

    async def a_initiate_chat(self, manager, message):
        self.call_count += 1
        self.success_count += 1
        manager.groupchat.messages.append(
            {"name": self.name, "content": json.dumps({"intent": "BALANCE_INQUIRY", "confidence": 0.9})}
        )
        manager.groupchat.messages.append(
            {
                "name": "entity_extractor",
                "content": json.dumps(
                    {
                        "entities": [{"type": "account", "value": "123"}],
                        "extraction_success": True,
                    }
                ),
            }
        )


class StubEntityExtractorAgent:
    name = "entity_extractor"

    def __init__(self):
        self.success_count = 0
        self.error_count = 0


@pytest.fixture
def team():
    team = FinancialAnalysisTeam()
    team.cache_manager = StubCacheManager()
    team.metrics_collector = StubMetricsCollector()
    team.intent_classifier = StubIntentAgent()
    team.entity_extractor = StubEntityExtractorAgent()
    team.group_chat.agents = [team.intent_classifier, team.entity_extractor]
    return team


def test_process_user_message_sequence(team):
    result = asyncio.run(team.process_user_message("What's my balance?", user_id=1))
    assert result["intent"]["intent"] == "BALANCE_INQUIRY"
    assert result["entities"][0]["type"] == "account"
    assert team.group_chat.messages[1]["name"] == "intent_classifier"
    assert team.group_chat.messages[2]["name"] == "entity_extractor"
    assert team.intent_classifier.call_count == 1
    assert team.metrics_collector.counters["financial_team.success"] == 1


def test_process_user_message_uses_cache(team):
    msg = "Check balance"
    uid = 2
    first = asyncio.run(team.process_user_message(msg, uid))
    assert team.cache_manager.metrics["misses"] == 1
    second = asyncio.run(team.process_user_message(msg, uid))
    assert second == first
    assert team.cache_manager.metrics["hits"] == 1
    assert team.intent_classifier.call_count == 1


def test_process_user_message_fallback(team):
    async def failing_chat(manager, message):
        raise RuntimeError("boom")

    team.intent_classifier.a_initiate_chat = failing_chat

    result = asyncio.run(team.process_user_message("Hi", user_id=3))
    assert result["intent"]["intent"] == "GENERAL_INQUIRY"
    assert any("boom" in err for err in result["errors"])
    assert team.team_metrics["failures"] == 1
    assert team.metrics_collector.counters["financial_team.failure"] == 1


def test_health_check_structure(team):
    team.intent_classifier.success_count = 2
    team.entity_extractor.error_count = 1

    status = asyncio.run(team.health_check())
    assert "agents" in status
    assert status["agents"]["intent_classifier"]["success"] == 2
    assert status["agents"]["entity_extractor"]["errors"] == 1
    assert "team_metrics" in status
    assert "cache" in status and status["cache"]["status"] == "ok"
    assert status["config"]["llm_model"] == "deepseek-chat"
    assert "timestamp" in status
