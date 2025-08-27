"""Base financial agent with shared utilities.

This module provides a light wrapper around :class:`AssistantAgent` with
helpers common to financial agents, such as currency normalization.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

try:  # pragma: no cover - handled in tests via stub
    from autogen import AssistantAgent
except Exception:  # pragma: no cover - fallback when autogen isn't installed
    class AssistantAgent:  # type: ignore
        """Minimal stub used when the autogen library is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401 - simple stub
            pass

        def add_capability(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
            pass


class BaseFinancialAgent(AssistantAgent):
    """Assistant de base pour les opérations financières.

    Parameters
    ----------
    name:
        Nom de l'agent.
    system_message:
        Message système envoyé au LLM. Un message par défaut adapté au domaine
        financier est utilisé si aucun n'est fourni.
    llm_config:
        Configuration du modèle sous-jacent.
    default_currency:
        Devise par défaut pour la normalisation des montants.
    """

    def __init__(
        self,
        name: str = "base_financial",
        system_message: Optional[str] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        default_currency: str = "EUR",
        **kwargs: Any,
    ) -> None:
        system_message = system_message or (
            "Tu es un assistant financier aidant les utilisateurs à comprendre"
            " leurs données monétaires."
        )
        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            **kwargs,
        )
        self.default_currency = default_currency

    # ------------------------------------------------------------------
    # Helpers financiers
    # ------------------------------------------------------------------
    def normalize_amount(self, value: Any, currency: Optional[str] = None) -> Dict[str, Any]:
        """Normalise un montant en valeur flottante et code devise.

        Parameters
        ----------
        value:
            Valeur brute à normaliser.
        currency:
            Code devise (ex: ``"EUR"``). La devise par défaut est utilisée si
            aucune n'est fournie.

        Returns
        -------
        dict
            Dictionnaire contenant ``amount`` (float) et ``currency`` (str).
        """
        try:
            amount = float(value)
        except (TypeError, ValueError):  # pragma: no cover - validation simple
            amount = 0.0

        return {"amount": amount, "currency": (currency or self.default_currency).upper()}

    def format_amount(self, value: Any, currency: Optional[str] = None) -> str:
        """Formate un montant normalisé pour affichage."""
        normalized = self.normalize_amount(value, currency)
        return f"{normalized['amount']:.2f} {normalized['currency']}"
