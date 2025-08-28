import os
import asyncio
import importlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict

logger = logging.getLogger("conversation_service.autogen")


class ConversationServiceRuntime:
    """Runtime pour le service de conversation basé sur AutoGen."""

    _instance: "ConversationServiceRuntime | None" = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
            cls._instance.client = None
            cls._instance.teams: Dict[str, Any] = {}
        return cls._instance

    async def initialize(self) -> None:
        if self._initialized:
            return

        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL")
        missing = []
        if not api_key:
            missing.append("DEEPSEEK_API_KEY")
        if not base_url:
            missing.append("DEEPSEEK_BASE_URL")
        if missing:
            raise RuntimeError(
                f"Variables d'environnement manquantes: {', '.join(missing)}"
            )

        try:
            from autogen_ext.models.openai import OpenAIChatCompletionClient
        except Exception as exc:  # pragma: no cover - import error path
            logger.error("Impossible d'importer OpenAIChatCompletionClient: %s", exc)
            raise

        self.client = OpenAIChatCompletionClient(
            api_key=api_key,
            base_url=base_url,
            model="deepseek-chat",
        )

        try:
            await asyncio.wait_for(self._test_deepseek_connection(), 30)
        except Exception as exc:
            logger.error("Échec du test de connexion DeepSeek: %s", exc)
            raise

        # Chargement dynamique des équipes
        try:
            module = importlib.import_module(
                "conversation_service.teams.financial_analysis_team_phase2"
            )
            team_cls = getattr(module, "FinancialAnalysisTeamPhase2")
            self.teams["phase2"] = team_cls
        except Exception as exc:  # pragma: no cover - import error path
            logger.error(
                "Erreur lors du chargement de l'équipe FinancialAnalysisTeamPhase2: %s",
                exc,
            )

        self._initialized = True

    async def _test_deepseek_connection(self) -> None:
        from autogen_core.models import UserMessage

        if self.client is None:
            raise RuntimeError("Client DeepSeek non initialisé")

        await self.client.create(
            messages=[UserMessage(content="ping", source="runtime")]
        )

    def get_team(self, team_name: str):
        try:
            return self.teams[team_name]
        except KeyError as exc:
            raise KeyError(f"Équipe '{team_name}' introuvable") from exc

    def health_check(self) -> Dict[str, Any]:
        return {
            "deepseek_client": "initialized" if self.client else "uninitialized",
            "loaded_teams": list(self.teams.keys()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
