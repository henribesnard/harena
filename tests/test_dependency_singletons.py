import asyncio
from types import SimpleNamespace

from conversation_service.api import dependencies


def test_get_team_manager_returns_singleton(monkeypatch):
    class DummyTeamManager:
        def __init__(self, metrics_collector=None, cache_manager=None):
            self.metrics = metrics_collector
            self.cache = cache_manager
            self.team_health = SimpleNamespace(overall_healthy=True)

        async def initialize_agents(self):
            pass

        async def shutdown(self):
            pass

    def fake_load_team_manager():
        return DummyTeamManager, None

    monkeypatch.setattr(dependencies, "load_team_manager", fake_load_team_manager)
    asyncio.run(dependencies.cleanup_dependencies())
    tm1 = asyncio.run(dependencies.get_team_manager())
    tm2 = asyncio.run(dependencies.get_team_manager())
    assert tm1 is tm2
    assert isinstance(tm1, DummyTeamManager)
    asyncio.run(dependencies.cleanup_dependencies())


def test_get_conversation_manager_returns_singleton():
    asyncio.run(dependencies.cleanup_dependencies())
    cm1 = asyncio.run(dependencies.get_conversation_manager())
    cm2 = asyncio.run(dependencies.get_conversation_manager())
    from conversation_service.core.conversation_manager import ConversationManager
    assert cm1 is cm2
    assert isinstance(cm1, ConversationManager)
    asyncio.run(dependencies.cleanup_dependencies())
