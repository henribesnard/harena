import asyncio
import logging
import types
import sys

import pytest

from conversation_service.autogen_core.agent_runtime import ConversationServiceRuntime


@pytest.fixture
def runtime():
    ConversationServiceRuntime._instance = None
    runtime = ConversationServiceRuntime()
    yield runtime
    ConversationServiceRuntime._instance = None


@pytest.fixture
def stub_autogen_client():
    """Stub autogen_ext client to avoid real imports."""
    openai_module = types.ModuleType("autogen_ext.models.openai")

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        async def create(self, *args, **kwargs):
            pass

    openai_module.OpenAIChatCompletionClient = DummyClient

    autogen_ext_module = types.ModuleType("autogen_ext")
    models_module = types.ModuleType("autogen_ext.models")
    sys.modules["autogen_ext"] = autogen_ext_module
    sys.modules["autogen_ext.models"] = models_module
    sys.modules["autogen_ext.models.openai"] = openai_module
    return DummyClient


@pytest.fixture
def stub_team_module():
    """Stub team module to test dynamic import."""
    teams_pkg = types.ModuleType("conversation_service.teams")
    sys.modules["conversation_service.teams"] = teams_pkg
    team_module = types.ModuleType(
        "conversation_service.teams.financial_analysis_team_phase2"
    )
    exec(
        "from conversation_service.autogen_core.agent_runtime import ConversationServiceRuntime\n"
        "class FinancialAnalysisTeamPhase2:\n"
        "    pass\n",
        team_module.__dict__,
    )
    sys.modules[
        "conversation_service.teams.financial_analysis_team_phase2"
    ] = team_module
    return team_module.FinancialAnalysisTeamPhase2


def test_initialization_without_env_vars_raises_and_logs(runtime, monkeypatch, caplog):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError) as excinfo:
            asyncio.run(runtime.initialize())
    assert "DEEPSEEK_API_KEY" in str(excinfo.value)
    assert "DEEPSEEK_BASE_URL" in str(excinfo.value)
    # No logs should be emitted when env vars are missing
    assert caplog.records == []


def test_initialization_with_invalid_key(runtime, stub_autogen_client, stub_team_module, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "invalid")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "http://example")

    async def fake_test(self):
        raise RuntimeError("Invalid API key")

    monkeypatch.setattr(ConversationServiceRuntime, "_test_deepseek_connection", fake_test)

    with pytest.raises(RuntimeError, match="Invalid API key"):
        asyncio.run(runtime.initialize())


def test_health_check_returns_status(runtime, stub_autogen_client, stub_team_module, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "valid")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "http://example")

    async def ok_test(self):
        return None

    monkeypatch.setattr(ConversationServiceRuntime, "_test_deepseek_connection", ok_test)

    asyncio.run(runtime.initialize())
    status = runtime.health_check()
    assert status["deepseek_client"] == "initialized"
    assert "phase2" in status["loaded_teams"]
    assert status["timestamp"]


def test_team_import_success(runtime, stub_autogen_client, stub_team_module, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "valid")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "http://example")

    async def ok_test(self):
        return None

    monkeypatch.setattr(ConversationServiceRuntime, "_test_deepseek_connection", ok_test)

    asyncio.run(runtime.initialize())
    team_cls = runtime.get_team("phase2")
    assert team_cls.__name__ == "FinancialAnalysisTeamPhase2"


def test_get_team_missing_name(runtime):
    with pytest.raises(KeyError, match="Équipe 'unknown' introuvable"):
        runtime.get_team("unknown")
