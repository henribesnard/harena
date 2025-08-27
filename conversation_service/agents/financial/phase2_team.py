from __future__ import annotations
"""Phase 2 AutoGen financial analysis team placeholder"""
from typing import Any, Dict


class FinancialAnalysisTeamPhase2:
    """Simple placeholder for the AutoGen financial analysis team.

    In real implementation this would orchestrate multiple agents using
    the AutoGen framework.  For the tests we provide a minimal async
    interface that mimics the expected behaviour.
    """

    async def run(self, message: str, user_context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Generate a dummy response for the provided message.

        Parameters
        ----------
        message: str
            User message to analyse.
        user_context: dict | None
            Optional context about the user.

        Returns
        -------
        dict
            A dictionary emulating a complex AutoGen response.
        """
        return {
            "final_answer": f"Analyse financière simulée pour: {message}",
            "context": user_context or {},
            "intermediate_steps": [
                {"agent": "analyst", "content": f"Analysing: {message}"}
            ],
        }
