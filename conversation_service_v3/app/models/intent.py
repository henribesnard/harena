"""
Modèles pour la classification des intentions utilisateur
"""
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class IntentCategory(Enum):
    """Catégories d'intentions utilisateur"""

    # Intentions nécessitant une recherche financière
    FINANCIAL_QUERY = "financial_query"
    FINANCIAL_ANALYSIS = "financial_analysis"
    FINANCIAL_STATS = "financial_stats"

    # Intentions conversationnelles (pas de recherche)
    GREETING = "greeting"
    FAREWELL = "farewell"
    GRATITUDE = "gratitude"
    SMALL_TALK = "small_talk"

    # Intentions métacognitives (à propos du service)
    SERVICE_INFO = "service_info"
    CAPABILITY_QUERY = "capability_query"
    HELP_REQUEST = "help_request"

    # Intentions hors scope
    OUT_OF_SCOPE = "out_of_scope"
    UNCLEAR = "unclear"


@dataclass
class IntentClassification:
    """Résultat de la classification d'intention"""
    category: IntentCategory
    confidence: float
    requires_search: bool
    reasoning: str
    suggested_response: Optional[str] = None
