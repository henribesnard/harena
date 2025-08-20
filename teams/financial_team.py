"""Example financial analysis team.

The financial team demonstrates how specialized agents can be coordinated
through the :class:`~teams.team_orchestrator.TeamOrchestrator`. The team
uses two simple placeholder agents: one performs an "analysis" step and
the other generates a final report based on that analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class SimpleAgent:
    """Minimal agent placeholder used for demonstration."""

    name: str

    def run(self, message: str) -> str:  # pragma: no cover - trivial
        return f"{self.name}: {message}"


class AnalystAgent(SimpleAgent):
    def run(self, message: str) -> str:
        return f"analysis of {message}"


class ReportAgent(SimpleAgent):
    def run(self, message: str) -> str:
        return f"report based on {message}"


class FinancialTeam:
    """Simple financial team chaining two agents."""

    def __init__(self) -> None:
        self.analyst = AnalystAgent("financial_analyst")
        self.reporter = ReportAgent("report_generator")

    def process(self, message: str) -> Dict[str, str]:
        """Run the message through the analysis and reporting agents."""
        analysis = self.analyst.run(message)
        report = self.reporter.run(analysis)
        return {"analysis": analysis, "report": report}


# Register this team with the global orchestrator on import
from .team_orchestrator import register_team

register_team("financial", FinancialTeam)
