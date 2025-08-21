import builtins
import sys

import conversation_service.core as core


def _reset_core_state() -> None:
    """Reset cached team manager components in the core module."""
    core.MVPTeamManager = None
    core.TeamConfiguration = None
    core.TEAM_MANAGER_AVAILABLE = False


def test_load_team_manager_returns_none_when_unavailable(monkeypatch):
    """``load_team_manager`` should gracefully handle missing dependency."""

    _reset_core_state()
    sys.modules.pop("conversation_service.core.mvp_team_manager", None)
    sys.modules.pop("conversation_service.mvp_team_manager", None)

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.endswith("mvp_team_manager"):
            raise ImportError("mocked missing module")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    manager, config_cls = core.load_team_manager()
    assert manager is None
    assert config_cls is None


def test_load_team_manager_loads_manager_when_available():
    """When the team manager is present it should be returned."""

    _reset_core_state()
    manager, config_cls = core.load_team_manager()

    from conversation_service.core.mvp_team_manager import (
        MVPTeamManager,
        TeamConfiguration,
    )

    assert manager is MVPTeamManager
    assert config_cls is TeamConfiguration

