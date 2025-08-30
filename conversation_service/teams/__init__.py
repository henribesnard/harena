"""
Module teams - Orchestration équipes AutoGen pour conversation service
Intégration complète avec infrastructure existante
"""

from .multi_agent_financial_team import MultiAgentFinancialTeam

__all__ = [
    "MultiAgentFinancialTeam"
]

# Version et metadata
__version__ = "1.0.0"
__description__ = "Équipes AutoGen multi-agents intégrées avec infrastructure conversation service"