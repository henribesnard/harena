"""Team modules for coordinating conversation agents.

This package exposes utilities to build specialised agent teams and
orchestrate conversations.  The :class:`FinancialTeam` bundles the
financial agents while :class:`TeamOrchestrator` manages conversation
sessions for the API layer.
"""

from .financial_team import FinancialTeam
from .team_orchestrator import TeamOrchestrator

__all__ = ["FinancialTeam", "TeamOrchestrator"]
