"""Équipe d'analyse financière Phase 2 utilisant AutoGen.

Cette équipe orchestre un workflow en deux étapes :
1. Classification de l'intention utilisateur
2. Extraction des entités financières pertinentes

Le tout est géré par un ``GroupChat`` et son ``GroupChatManager``.
"""

from __future__ import annotations

from typing import Any, Dict
import json
import re

try:  # pragma: no cover - les classes réelles peuvent être indisponibles
    from autogen import (
        AssistantAgent as RealAssistantAgent,
        GroupChat as RealGroupChat,
        GroupChatManager as RealGroupChatManager,
    )
except Exception:  # pragma: no cover
    RealAssistantAgent = None
    RealGroupChat = None
    RealGroupChatManager = None


class StubAssistantAgent:
    """Stub minimal d'``AssistantAgent``."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401
        pass


class StubGroupChat:
    """Stub minimal de ``GroupChat``."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401
        self.agents = kwargs.get("agents", [])
        self.messages = kwargs.get("messages", [])
        self.max_round = kwargs.get("max_round", 1)


class StubGroupChatManager:
    """Stub minimal de ``GroupChatManager`` avec logique simple."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401
        self.groupchat = kwargs.get("groupchat")
        self.llm_config = kwargs.get("llm_config", {})

    async def run(self, message: str, *_, **__) -> Dict[str, Any]:
        """Analyse basique du message pour générer intent et entités."""
        intent = "unknown"
        entities: Dict[str, Any] = {}

        # Détection très simple de tickers (suite de lettres majuscules)
        tickers = re.findall(r"\b[A-Z]{2,5}\b", message)
        if tickers:
            entities["tickers"] = tickers
            intent = "price_query"

        # Détection de montants numériques
        amounts = re.findall(r"\b\d+(?:\.\d+)?\b", message)
        if amounts:
            entities["amounts"] = [float(a) for a in amounts]
            intent = "transaction_amount"

        return {"intent": intent, "entities": entities}


class FinancialAnalysisTeamPhase2:
    """Orchestration AutoGen pour la phase 2 d'analyse financière."""

    def __init__(self, llm_config: Dict[str, Any] | None = None) -> None:
        if llm_config:
            self.llm_config = llm_config
            AgentCls = RealAssistantAgent or StubAssistantAgent
            ChatCls = RealGroupChat or StubGroupChat
            ManagerCls = RealGroupChatManager or StubGroupChatManager
        else:  # aucun llm_config fourni -> utilisation des stubs locaux
            self.llm_config = {"model": "stub"}
            AgentCls = StubAssistantAgent
            ChatCls = StubGroupChat
            ManagerCls = StubGroupChatManager

        # Agents spécialisés
        self.intent_agent = AgentCls(
            name="intent_classifier",
            system_message=(
                "Tu es un assistant expert qui détecte l'intention "
                "financière d'un message utilisateur."
            ),
            llm_config=self.llm_config,
            description="Phase 1 - Classification d'intentions",
        )

        self.entity_agent = AgentCls(
            name="entity_extractor",
            system_message=(
                "En te basant sur l'intention détectée, extrais les entités "
                "financières pertinentes."
            ),
            llm_config=self.llm_config,
            description="Phase 2 - Extraction d'entités",
        )

        # Configuration du GroupChat
        self.groupchat = ChatCls(
            agents=[self.intent_agent, self.entity_agent],
            messages=[],
            max_round=2,
        )

        # Manager orchestrant les échanges entre agents
        self.manager = ManagerCls(
            groupchat=self.groupchat,
            llm_config=self.llm_config,
        )

    async def run(self, message: str) -> Dict[str, Any]:
        """Exécute le workflow Intent → Entities pour un message donné."""
        # Lancement du gestionnaire de conversation
        result = await self.manager.run(message)

        # Extraction des réponses si disponibles
        intent: Any = "unknown"
        entities: Dict[str, Any] = {}
        if isinstance(result, dict):
            intent = result.get("intent", intent)
            entities = result.get("entities", entities) or {}
        elif isinstance(result, str):
            # Si la réponse est une chaîne JSON, on tente de la parser
            try:
                data = json.loads(result)
                intent = data.get("intent", intent)
                entities = data.get("entities", entities) or {}
            except Exception:  # pragma: no cover - cas d'échec de parsing
                pass

        return {"intent": intent, "entities": entities}
