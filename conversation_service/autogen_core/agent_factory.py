"""Fabrique d'agents AutoGen"""
from __future__ import annotations

import logging
from typing import Any

from .deepseek_integration import DeepSeekAutoGenClient

logger = logging.getLogger("conversation_service.autogen.factory")


class AutoGenAgentFactory:
    """Crée des agents AutoGen préconfigurés"""

    def __init__(self, deepseek: DeepSeekAutoGenClient | None = None) -> None:
        self.deepseek = deepseek or DeepSeekAutoGenClient()

    async def create_assistant(self, name: str, system_message: str) -> Any:
        """Crée un assistant AutoGen basé sur DeepSeek."""
        try:
            from autogen import AssistantAgent
        except Exception as exc:  # pragma: no cover - dépendance externe
            raise RuntimeError("Autogen n'est pas installé") from exc

        await self.deepseek.initialize()
        llm_config = self.deepseek.get_llm_config()
        return AssistantAgent(name=name, llm_config=llm_config, system_message=system_message)

    async def create_user_proxy(self, name: str) -> Any:
        """Crée un agent utilisateur proxy."""
        try:
            from autogen import UserProxyAgent
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Autogen n'est pas installé") from exc

        return UserProxyAgent(name=name, human_input_mode="NEVER")
