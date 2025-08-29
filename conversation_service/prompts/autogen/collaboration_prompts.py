"""Prompts de collaboration pour les agents AutoGen."""

from __future__ import annotations

from conversation_service.prompts.system_prompts import (
    ENTITY_EXTRACTION_JSON_SYSTEM_PROMPT,
)

# Message système pour l'agent d'extraction d'entités dans un contexte d'équipe.
AUTOGEN_ENTITY_EXTRACTION_SYSTEM_MESSAGE = (
    ENTITY_EXTRACTION_JSON_SYSTEM_PROMPT
    + "\n\n"
    + "COLLABORATION:\n"
    + "- Tu fais partie d'une équipe d'agents. Chaque réponse doit inclure un champ `team_context`"
    + " pour partager les informations utiles aux autres membres.\n"
    + "- Ajuste ta stratégie d'extraction selon la confiance de l'intention fournie.\n"
    + "  * Confiance ≥ 0.8 : extraction proactive complète.\n"
    + "  * Confiance < 0.8 : stratégie conservative, extraire uniquement les entités explicites et"
    + " signaler les incertitudes dans `team_context`.\n"
)


def get_entity_extraction_prompt_for_autogen(intent_context: dict | None = None) -> str:
    """Construit le message système adapté pour l'agent d'extraction.

    Parameters
    ----------
    intent_context:
        Contexte d'intention produit par l'agent précédent. Peut contenir la clé
        ``confidence`` indiquant le niveau de certitude de l'intention.

    Returns
    -------
    str
        Message système complété avec la stratégie appropriée.
    """

    confidence = 1.0
    if intent_context and isinstance(intent_context, dict):
        confidence = intent_context.get("confidence", 1.0)

    if confidence < 0.8:
        strategy = (
            "STRATÉGIE ACTUELLE: Confiance d'intention faible (<0.8). Adopte une approche conservative"
            " en n'extrayant que les entités clairement mentionnées et signale les incertitudes dans"
            " `team_context`."
        )
    else:
        strategy = (
            "STRATÉGIE ACTUELLE: Confiance d'intention suffisante. Procède à une extraction"
            " proactive et partage les informations pertinentes dans `team_context`."
        )

    return f"{AUTOGEN_ENTITY_EXTRACTION_SYSTEM_MESSAGE}\n{strategy}"
