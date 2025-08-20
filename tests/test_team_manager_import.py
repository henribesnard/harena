from conversation_service.core import load_team_manager


def test_load_team_manager_returns_none_when_unavailable():
    manager, config_cls = load_team_manager()
    assert manager is None
    assert config_cls is None
