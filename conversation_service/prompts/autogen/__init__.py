"""Utilitaires de prompts pour l'intégration avec AutoGen.

Ce module rassemble les différents messages système et fonctions d'aide
nécessaires aux agents AutoGen.
"""

from .entity_extraction_prompts import (
    AUTOGEN_ENTITY_EXTRACTION_SYSTEM_MESSAGE,
    get_entity_extraction_prompt_for_autogen,
)
from .team_orchestration_prompts import TEAM_ORCHESTRATION_PHASE2_MESSAGE

__all__ = [
    "get_entity_extraction_prompt_for_autogen",
    "AUTOGEN_ENTITY_EXTRACTION_SYSTEM_MESSAGE",
    "TEAM_ORCHESTRATION_PHASE2_MESSAGE",
]
