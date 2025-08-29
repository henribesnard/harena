import sys
import types
import json
import pytest


class DummyAssistantAgent:
    def __init__(self, name="agent", **_):
        self.name = name

    def add_capability(self, *_args, **_kwargs):
        pass

    async def a_initiate_chat(self, *_args, **_kwargs):
        pass

    async def a_generate_reply(self, *_args, **_kwargs):
        return ""


class DummyGroupChat:
    def __init__(self, agents=None, messages=None, max_round=1, speaker_selection_method=""):
        self.agents = agents or []
        self.messages = messages or []
        self.max_round = max_round
        self.speaker_selection_method = speaker_selection_method


class DummyGroupChatManager:
    def __init__(self, groupchat, llm_config=None):
        self.groupchat = groupchat


autogen_stub = types.SimpleNamespace(
    AssistantAgent=DummyAssistantAgent,
    GroupChat=DummyGroupChat,
    GroupChatManager=DummyGroupChatManager,
)

sys.modules["autogen"] = autogen_stub

from conversation_service.teams.financial_analysis_team import FinancialAnalysisTeam


@pytest.mark.asyncio
async def test_workflow_complete_success(monkeypatch):
    team = FinancialAnalysisTeam()

    cache: dict = {}

    async def fake_get(key, cache_type="response"):
        return cache.get(key)

    async def fake_set(key, value, cache_type="response"):
        cache[key] = value

    monkeypatch.setattr(team.cache_manager, "get_semantic_cache", fake_get)
    monkeypatch.setattr(team.cache_manager, "set_semantic_cache", fake_set)

    metrics_calls = {"increment": [], "histogram": []}

    def fake_increment(name):
        metrics_calls["increment"].append(name)

    def fake_histogram(name, value):
        metrics_calls["histogram"].append((name, value))

    monkeypatch.setattr(team.metrics_collector, "increment_counter", fake_increment)
    monkeypatch.setattr(team.metrics_collector, "record_histogram", fake_histogram)

    async def fake_intent_reply(message):
        return json.dumps({"intent": "BALANCE_INQUIRY", "confidence": 0.9})

    async def fake_entity_reply(message):
        return json.dumps(
            {
                "extraction_success": True,
                "entities": [{"type": "account", "value": "123"}],
                "errors": [],
            }
        )

    async def fake_initiate_chat(self, manager, message):
        intent_msg = await fake_intent_reply(message)
        manager.groupchat.messages.append({"name": self.name, "content": intent_msg})
        entity_msg = await fake_entity_reply(message)
        manager.groupchat.messages.append(
            {"name": team.entity_extractor.name, "content": entity_msg}
        )

    monkeypatch.setattr(team.intent_classifier, "a_generate_reply", fake_intent_reply)
    monkeypatch.setattr(team.entity_extractor, "a_generate_reply", fake_entity_reply)
    monkeypatch.setattr(team.intent_classifier, "a_initiate_chat", types.MethodType(fake_initiate_chat, team.intent_classifier))

    result = await team.process_user_message("What is my balance?", user_id=1)

    assert result["intent"]["intent"] == "BALANCE_INQUIRY"
    assert result["entities"] == [{"type": "account", "value": "123"}]
    assert result["errors"] == []

    assert "financial_team.success" in metrics_calls["increment"]
    assert metrics_calls["histogram"]
    assert team.team_metrics["total_requests"] == 1
    assert team.team_metrics["failures"] == 0
    assert team.team_metrics["avg_processing_time_ms"] > 0


def test_intent_entity_coherence():
    team = FinancialAnalysisTeam()

    assert team._validate_intent_entity_coherence(
        {"intent": "BALANCE_INQUIRY"}, {"extraction_success": True}
    )
    assert not team._validate_intent_entity_coherence(
        {"intent": "GENERAL_INQUIRY"}, {"extraction_success": True}
    )


@pytest.mark.asyncio
async def test_shared_cache_usage(monkeypatch):
    team = FinancialAnalysisTeam()

    cache: dict = {}

    async def fake_get(key, cache_type="response"):
        return cache.get(key)

    async def fake_set(key, value, cache_type="response"):
        cache[key] = value

    monkeypatch.setattr(team.cache_manager, "get_semantic_cache", fake_get)
    monkeypatch.setattr(team.cache_manager, "set_semantic_cache", fake_set)

    call_counts = {"intent": 0, "entities": 0}

    async def fake_intent_reply(message):
        call_counts["intent"] += 1
        return json.dumps({"intent": "BALANCE_INQUIRY", "confidence": 0.9})

    async def fake_entity_reply(message):
        call_counts["entities"] += 1
        return json.dumps({"extraction_success": True, "entities": []})

    async def fake_initiate_chat(self, manager, message):
        intent_msg = await fake_intent_reply(message)
        manager.groupchat.messages.append({"name": self.name, "content": intent_msg})
        entity_msg = await fake_entity_reply(message)
        manager.groupchat.messages.append(
            {"name": team.entity_extractor.name, "content": entity_msg}
        )

    monkeypatch.setattr(team.intent_classifier, "a_generate_reply", fake_intent_reply)
    monkeypatch.setattr(team.entity_extractor, "a_generate_reply", fake_entity_reply)
    monkeypatch.setattr(team.intent_classifier, "a_initiate_chat", types.MethodType(fake_initiate_chat, team.intent_classifier))

    await team.process_user_message("Show my balance", user_id=2)
    await team.process_user_message("Show my balance", user_id=2)

    assert call_counts["intent"] == 1
    assert call_counts["entities"] == 1


def test_internal_metrics_update():
    team = FinancialAnalysisTeam()

    team._update_team_metrics(100.0, True)
    assert team.team_metrics["total_requests"] == 1
    assert team.team_metrics["failures"] == 0
    assert team.team_metrics["avg_processing_time_ms"] == 100.0

    team._update_team_metrics(200.0, False)
    assert team.team_metrics["total_requests"] == 2
    assert team.team_metrics["failures"] == 1
    assert team.team_metrics["avg_processing_time_ms"] == 150.0


@pytest.mark.asyncio
async def test_agent_failure_fallback(monkeypatch):
    team = FinancialAnalysisTeam()

    async def fake_get(key, cache_type="response"):
        return None

    async def fake_set(key, value, cache_type="response"):
        pass

    monkeypatch.setattr(team.cache_manager, "get_semantic_cache", fake_get)
    monkeypatch.setattr(team.cache_manager, "set_semantic_cache", fake_set)
    monkeypatch.setattr(team.metrics_collector, "increment_counter", lambda *a, **k: None)
    monkeypatch.setattr(team.metrics_collector, "record_histogram", lambda *a, **k: None)

    async def failing_intent_reply(message):
        raise RuntimeError("intent failure")

    async def fake_initiate_chat(self, manager, message):
        intent_msg = await self.a_generate_reply(message)
        manager.groupchat.messages.append({"name": self.name, "content": intent_msg})

    monkeypatch.setattr(team.intent_classifier, "a_generate_reply", failing_intent_reply)
    monkeypatch.setattr(team.intent_classifier, "a_initiate_chat", types.MethodType(fake_initiate_chat, team.intent_classifier))

    result = await team.process_user_message("Hello", user_id=3)

    assert result["intent"]["intent"] == "GENERAL_INQUIRY"
    assert result["entities"] == []
    assert any("intent failure" in err for err in result["errors"])
    assert team.team_metrics["failures"] == 1
