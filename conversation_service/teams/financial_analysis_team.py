"""Équipe d'analyse financière Phase 2 utilisant AutoGen.

Cette équipe orchestre un workflow en deux étapes :
1. Classification de l'intention utilisateur
2. Extraction des entités financières pertinentes

Le tout est géré par un ``GroupChat`` et son ``GroupChatManager``.
"""

from __future__ import annotations

from typing import Any, Dict

try:  # pragma: no cover - fallback lorsque autogen est indisponible en tests
    from autogen import AssistantAgent, GroupChat, GroupChatManager
except Exception:  # pragma: no cover
    class AssistantAgent:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            """Stub minimal d'``AssistantAgent``."""
            pass

    class GroupChat:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            """Stub minimal de ``GroupChat``."""
            self.agents = kwargs.get("agents", [])
            self.messages = kwargs.get("messages", [])
            self.max_round = kwargs.get("max_round", 1)

    class GroupChatManager:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            """Stub minimal de ``GroupChatManager``."""
            self.groupchat = kwargs.get("groupchat")
            self.llm_config = kwargs.get("llm_config", {})

        async def run(self, *_, **__) -> Dict[str, Any]:
            """Retourne un dictionnaire vide pour compatibilité."""
            return {}


class FinancialAnalysisTeamPhase2:
    """Orchestration AutoGen pour la phase 2 d'analyse financière."""

    def __init__(self, llm_config: Dict[str, Any] | None = None) -> None:
        self.llm_config = llm_config or {}

        # Agents spécialisés
        self.intent_agent = AssistantAgent(
            name="intent_classifier",
            system_message=(
                "Tu es un assistant expert qui détecte l'intention "
                "financière d'un message utilisateur."
            ),
            llm_config=self.llm_config,
            description="Phase 1 - Classification d'intentions",
        )

        self.entity_agent = AssistantAgent(
            name="entity_extractor",
            system_message=(
                "En te basant sur l'intention détectée, extrais les entités "
                "financières pertinentes."
            ),
            llm_config=self.llm_config,
            description="Phase 2 - Extraction d'entités",
        )

        # Configuration du GroupChat
        self.groupchat = GroupChat(
            agents=[self.intent_agent, self.entity_agent],
            messages=[],
            max_round=2,
        )

        # Manager orchestrant les échanges entre agents
        self.manager = GroupChatManager(
            groupchat=self.groupchat,
            llm_config=self.llm_config,
        )

    async def run(self, message: str) -> Dict[str, Any]:
        """Exécute le workflow Intent → Entities pour un message donné."""
        # Lancement du gestionnaire de conversation
        result = await self.manager.run(message)

        # Extraction des réponses si disponibles
        intent = None
        entities = None
        if isinstance(result, dict):
            intent = result.get("intent")
            entities = result.get("entities")

        return {"intent": intent, "entities": entities}
