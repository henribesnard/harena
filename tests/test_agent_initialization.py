import asyncio
import logging
from datetime import datetime, timedelta

import conversation_service.agents.base_financial_agent as base_financial_agent
base_financial_agent.AUTOGEN_AVAILABLE = True

from conversation_service.core.mvp_team_manager import MVPTeamManager
from conversation_service.core.deepseek_client import DeepSeekClient


def test_agents_initialize_without_exception(monkeypatch):
    async def run_test():
        async def fake_health_check(self):
            return {"status": "healthy"}

        def fake_init(self, api_key=None, base_url=None, cache_enabled=True, timeout=None):
            self.api_key = api_key
            self.base_url = base_url
            self.timeout = timeout
            self.cache_enabled = cache_enabled

        monkeypatch.setattr(DeepSeekClient, "__init__", fake_init)
        monkeypatch.setattr(DeepSeekClient, "health_check", fake_health_check)

        config = {
            "DEEPSEEK_API_KEY": "test-key",
            "DEEPSEEK_BASE_URL": "http://localhost",
            "DEEPSEEK_TIMEOUT": 30,
            "SEARCH_SERVICE_URL": "http://search.example.com",
            "MAX_CONVERSATION_HISTORY": 100,
            "WORKFLOW_TIMEOUT_SECONDS": 45,
            "HEALTH_CHECK_INTERVAL_SECONDS": 300,
            "AUTO_RECOVERY_ENABLED": True,
            "INITIAL_HEALTH_CHECK_DELAY_SECONDS": 1,
            "INITIAL_HEALTH_CHECK": False,
            "AGENT_FAILURE_THRESHOLD": 3,
            "ORCHESTRATOR_PERFORMANCE_THRESHOLD_MS": 30000,
        }

        manager = MVPTeamManager(config=config)
        await manager.initialize_agents(initial_health_check=False)

        if manager._delayed_health_check_task:
            manager._delayed_health_check_task.cancel()
            try:
                await manager._delayed_health_check_task
            except asyncio.CancelledError:
                pass

        assert manager.is_initialized
        await manager.shutdown()

    asyncio.run(run_test())


def test_agent_reactivation_cooldown(monkeypatch, caplog):
    async def run_test():
        async def fake_health_check(self):
            return {"status": "healthy"}

        def fake_init(self, api_key=None, base_url=None, cache_enabled=True, timeout=None):
            self.api_key = api_key
            self.base_url = base_url
            self.timeout = timeout
            self.cache_enabled = cache_enabled

        monkeypatch.setattr(DeepSeekClient, "__init__", fake_init)
        monkeypatch.setattr(DeepSeekClient, "health_check", fake_health_check)

        config = {
            "DEEPSEEK_API_KEY": "test-key",
            "DEEPSEEK_BASE_URL": "http://localhost",
            "DEEPSEEK_TIMEOUT": 30,
            "SEARCH_SERVICE_URL": "http://search.example.com",
            "MAX_CONVERSATION_HISTORY": 100,
            "WORKFLOW_TIMEOUT_SECONDS": 45,
            "HEALTH_CHECK_INTERVAL_SECONDS": 300,
            "AUTO_RECOVERY_ENABLED": True,
            "INITIAL_HEALTH_CHECK_DELAY_SECONDS": 1,
            "INITIAL_HEALTH_CHECK": False,
            "AGENT_FAILURE_THRESHOLD": 1,
            "ORCHESTRATOR_PERFORMANCE_THRESHOLD_MS": 30000,
            "AGENT_REACTIVATION_COOLDOWN_SECONDS": 5,
        }

        manager = MVPTeamManager(config=config)
        await manager.initialize_agents(initial_health_check=False)

        if manager._delayed_health_check_task:
            manager._delayed_health_check_task.cancel()
            try:
                await manager._delayed_health_check_task
            except asyncio.CancelledError:
                pass

        intent_agent = manager.agents["intent_agent"]

        calls = {"count": 0}

        def fake_is_healthy():
            calls["count"] += 1
            return calls["count"] > 1

        monkeypatch.setattr(intent_agent, "is_healthy", fake_is_healthy)

        with caplog.at_level(logging.INFO):
            await manager._perform_health_check()
            assert "disabled after" in caplog.text
            caplog.clear()

            await manager._perform_health_check()
            assert "cooldown" in caplog.text

            manager.agent_disabled_since["intent_agent"] = (
                datetime.utcnow() - timedelta(seconds=manager.reactivation_cooldown_seconds + 1)
            )
            caplog.clear()

            await manager._perform_health_check()
            assert "reactivated" in caplog.text

        await manager.shutdown()

    asyncio.run(run_test())
