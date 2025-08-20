"""Core team orchestration utilities.

This module provides a lightweight orchestrator that keeps a registry of
available teams and can route messages to them. Teams are registered
by name and lazily instantiated on first use. This design allows new
teams to be added dynamically at runtime.
"""

from __future__ import annotations

from typing import Any, Dict, List, Type


class TeamOrchestrator:
    """Registry and router for agent teams."""

    def __init__(self) -> None:
        self._registry: Dict[str, Type] = {}
        self._instances: Dict[str, Any] = {}

    def register_team(self, name: str, team_cls: Type) -> None:
        """Register a new team class.

        Args:
            name: Identifier for the team.
            team_cls: Class implementing the team.

        Raises:
            ValueError: If the name is already registered.
        """
        if name in self._registry:
            raise ValueError(f"Team '{name}' already registered")
        self._registry[name] = team_cls

    def available_teams(self) -> List[str]:
        """Return the list of registered team names."""
        return list(self._registry.keys())

    def get_team(self, name: str) -> Any:
        """Return an instance of the requested team.

        Instances are created lazily and cached for subsequent calls.

        Args:
            name: Team identifier.

        Returns:
            Instantiated team object.

        Raises:
            KeyError: If the team has not been registered.
        """
        if name not in self._registry:
            raise KeyError(f"Team '{name}' is not registered")
        if name not in self._instances:
            self._instances[name] = self._registry[name]()
        return self._instances[name]

    def process(self, team_name: str, message: str) -> Any:
        """Process a message using the specified team."""
        team = self.get_team(team_name)
        if not hasattr(team, "process"):
            raise AttributeError(f"Team '{team_name}' has no 'process' method")
        return team.process(message)


# Global orchestrator instance for convenience
_orchestrator = TeamOrchestrator()


def register_team(name: str, team_cls: Type) -> None:
    """Register a team on the global orchestrator."""
    _orchestrator.register_team(name, team_cls)


def get_team(name: str) -> Any:
    """Retrieve a team from the global orchestrator."""
    return _orchestrator.get_team(name)


def available_teams() -> List[str]:
    """List team names registered on the global orchestrator."""
    return _orchestrator.available_teams()


# Public alias for the global orchestrator
orchestrator = _orchestrator
