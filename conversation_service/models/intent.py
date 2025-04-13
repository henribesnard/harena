"""
Modèles pour la classification des intentions et leur configuration.

Ce module définit les modèles pour représenter les intentions des utilisateurs
et la configuration associée.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class IntentType(str, Enum):
    """Types d'intentions supportées par le système."""
    CHECK_BALANCE = "CHECK_BALANCE"
    SEARCH_TRANSACTION = "SEARCH_TRANSACTION"
    ANALYZE_SPENDING = "ANALYZE_SPENDING"
    FORECAST_BALANCE = "FORECAST_BALANCE"
    SUBSCRIPTION_MANAGEMENT = "SUBSCRIPTION_MANAGEMENT"
    SAVINGS_GOAL = "SAVINGS_GOAL"
    BUDGET_TRACKING = "BUDGET_TRACKING"
    GENERAL_QUERY = "GENERAL_QUERY"
    ACCOUNT_INFO = "ACCOUNT_INFO"
    HELP = "HELP"


class IntentConfig(BaseModel):
    """Configuration d'une intention."""
    description: str
    examples: List[str]
    requires_auth: bool = True
    required_entities: List[str] = []
    optional_entities: List[str] = []
    handler_config: Optional[Dict[str, Any]] = None


class Intent(BaseModel):
    """Représentation complète d'une intention."""
    type: IntentType
    config: IntentConfig


class Entity(BaseModel):
    """Entité extraite d'une requête utilisateur."""
    name: str
    value: Any
    confidence: float = 1.0


class IntentClassification(BaseModel):
    """Résultat de la classification d'intention."""
    intent: IntentType
    confidence: float
    entities: Dict[str, Any] = Field(default_factory=dict)
    raw_response: Optional[str] = None