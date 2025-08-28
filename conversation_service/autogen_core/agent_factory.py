from __future__ import annotations

"""Fabrique centralisée pour les agents AutoGen."""

from typing import Any, Dict


class AutoGenAgentFactory:
    """Fabrique d'agents basés sur AutoGen."""

    @staticmethod
    def create_standard_llm_config() -> Dict[str, Any]:
        """Retourne la configuration LLM standard compatible DeepSeek."""
        return {
            "model": "deepseek-chat",
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 0.95,
            "timeout": 30,
        }

    @staticmethod
    def create_financial_agent(client: Any):
        """Crée et retourne l'agent financier en injectant le client DeepSeek."""
        config = AutoGenAgentFactory.create_standard_llm_config()
        try:
            from conversation_service.agents.financial import FinancialAgent
        except Exception as exc:  # pragma: no cover - agent non implémenté
            raise NotImplementedError(
                "La classe FinancialAgent doit être définie dans conversation_service.agents.financial"
            ) from exc

        return FinancialAgent(deepseek_client=client, llm_config=config)

    @staticmethod
    def create_sales_agent(client: Any):
        """Placeholder pour créer un agent commercial."""
        raise NotImplementedError("Agent commercial non implémenté")

    @staticmethod
    def create_marketing_agent(client: Any):
        """Placeholder pour créer un agent marketing."""
        raise NotImplementedError("Agent marketing non implémenté")
