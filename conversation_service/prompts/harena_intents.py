"""
Types d'intentions Harena pour la nouvelle architecture v2.0
Basé sur la configuration YAML intentions_v2.yaml
"""

from enum import Enum
from typing import Dict, List


class HarenaIntentType(Enum):
    """Types d'intentions supportés par Harena"""
    
    # Intentions principales selon intentions_v2.yaml
    TRANSACTION_SEARCH = "TRANSACTION_SEARCH"
    BALANCE_INQUIRY = "BALANCE_INQUIRY" 
    SPENDING_ANALYSIS = "SPENDING_ANALYSIS"
    ACCOUNT_MANAGEMENT = "ACCOUNT_MANAGEMENT"
    CATEGORY_ANALYSIS = "CATEGORY_ANALYSIS"
    
    # Intentions sociales
    GREETING = "GREETING"
    GOODBYE = "GOODBYE"
    THANKS = "THANKS"
    CONVERSATIONAL = "CONVERSATIONAL"
    
    # Intentions spéciales
    UNCLEAR_INTENT = "UNCLEAR_INTENT"
    UNKNOWN = "UNKNOWN"
    UNSUPPORTED = "UNSUPPORTED"
    
    # Gestion d'erreurs
    ERROR_TECHNICAL = "ERROR_TECHNICAL"
    ERROR_DATA = "ERROR_DATA"


# Mapping des catégories d'intentions
INTENT_CATEGORIES: Dict[HarenaIntentType, str] = {
    HarenaIntentType.TRANSACTION_SEARCH: "search",
    HarenaIntentType.BALANCE_INQUIRY: "inquiry", 
    HarenaIntentType.SPENDING_ANALYSIS: "analysis",
    HarenaIntentType.ACCOUNT_MANAGEMENT: "management",
    HarenaIntentType.CATEGORY_ANALYSIS: "analysis",
    HarenaIntentType.GREETING: "social",
    HarenaIntentType.GOODBYE: "social",
    HarenaIntentType.THANKS: "social",
    HarenaIntentType.CONVERSATIONAL: "social",
    HarenaIntentType.UNCLEAR_INTENT: "system",
    HarenaIntentType.UNKNOWN: "system",
    HarenaIntentType.UNSUPPORTED: "system",
    HarenaIntentType.ERROR_TECHNICAL: "error",
    HarenaIntentType.ERROR_DATA: "error",
}


# Intentions supportées (excluant les erreurs et système)
SUPPORTED_INTENTS: List[HarenaIntentType] = [
    HarenaIntentType.TRANSACTION_SEARCH,
    HarenaIntentType.BALANCE_INQUIRY,
    HarenaIntentType.SPENDING_ANALYSIS,
    HarenaIntentType.ACCOUNT_MANAGEMENT,
    HarenaIntentType.CATEGORY_ANALYSIS,
    HarenaIntentType.GREETING,
    HarenaIntentType.GOODBYE,
    HarenaIntentType.THANKS,
    HarenaIntentType.CONVERSATIONAL,
]

# Intentions nécessitant une recherche
SEARCH_REQUIRED_INTENTS: List[HarenaIntentType] = [
    HarenaIntentType.TRANSACTION_SEARCH,
    HarenaIntentType.BALANCE_INQUIRY,
    HarenaIntentType.SPENDING_ANALYSIS,
    HarenaIntentType.ACCOUNT_MANAGEMENT,
    HarenaIntentType.CATEGORY_ANALYSIS,
]

# Intentions pouvant aller directement à la génération de réponse
DIRECT_RESPONSE_INTENTS: List[HarenaIntentType] = [
    HarenaIntentType.GREETING,
    HarenaIntentType.GOODBYE,
    HarenaIntentType.THANKS,
    HarenaIntentType.CONVERSATIONAL,
    HarenaIntentType.UNCLEAR_INTENT,
    HarenaIntentType.UNKNOWN,
    HarenaIntentType.UNSUPPORTED,
]


def is_intent_supported(intent_type: HarenaIntentType) -> bool:
    """Vérifie si une intention est supportée"""
    return intent_type in SUPPORTED_INTENTS


def get_intent_category(intent_type: HarenaIntentType) -> str:
    """Récupère la catégorie d'une intention"""
    return INTENT_CATEGORIES.get(intent_type, "unknown")


def get_all_intent_types() -> List[HarenaIntentType]:
    """Récupère tous les types d'intention"""
    return list(HarenaIntentType)


def get_supported_intent_types() -> List[HarenaIntentType]:
    """Récupère les types d'intention supportés"""
    return SUPPORTED_INTENTS.copy()


def requires_search(intent_type: HarenaIntentType) -> bool:
    """Vérifie si une intention nécessite une recherche"""
    return intent_type in SEARCH_REQUIRED_INTENTS


def can_direct_response(intent_type: HarenaIntentType) -> bool:
    """Vérifie si une intention peut aller directement à la génération de réponse"""
    return intent_type in DIRECT_RESPONSE_INTENTS