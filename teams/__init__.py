"""Team package with lazy imports to avoid heavy dependencies."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["FinancialTeam", "TeamOrchestrator"]


def __getattr__(name: str) -> Any:  # pragma: no cover - simple loader
    if name == "FinancialTeam":
        return import_module(".financial_team", __name__).FinancialTeam
    if name == "TeamOrchestrator":
        return import_module(".team_orchestrator", __name__).TeamOrchestrator
    raise AttributeError(name)

