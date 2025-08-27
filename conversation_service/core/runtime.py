from __future__ import annotations
"""Runtime container for conversation service agents."""
from typing import Any, Dict

from .cache_manager import CacheManager  # type: ignore
from ..clients.deepseek_client import DeepSeekClient  # type: ignore
from ..agents.financial.phase2_team import FinancialAnalysisTeamPhase2


class ConversationServiceRuntime:
    """Runtime holding long lived objects for the conversation service.

    The runtime is initialised once at application start and then injected
    in routes through the FastAPI application state.  Only a very small
    subset of the real behaviour is implemented here because tests rely on
    a lightweight mockable structure.
    """

    def __init__(self) -> None:
        self.deepseek_client: DeepSeekClient | None = None
        self.cache_manager: CacheManager | None = None
        self.financial_team = FinancialAnalysisTeamPhase2()

    async def initialize(self) -> None:
        """Initialise external clients if they exist."""
        if self.deepseek_client and hasattr(self.deepseek_client, "initialize"):
            await self.deepseek_client.initialize()
        if self.cache_manager and hasattr(self.cache_manager, "initialize"):
            await self.cache_manager.initialize()

    async def shutdown(self) -> None:
        """Close external resources."""
        if self.deepseek_client and hasattr(self.deepseek_client, "close"):
            await self.deepseek_client.close()
        if self.cache_manager and hasattr(self.cache_manager, "close"):
            await self.cache_manager.close()

    async def run_financial_team(self, message: str, user_context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Helper used by routes to call the financial analysis team."""
        return await self.financial_team.run(message, user_context or {})
