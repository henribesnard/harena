from enum import Enum


class IntentType(str, Enum):
    """Types d'intention pris en charge par la plateforme."""

    TRANSACTION_SEARCH = "TRANSACTION_SEARCH"
    """Rechercher toutes transactions sans filtre."""

    SEARCH_BY_DATE = "SEARCH_BY_DATE"
    """Transactions pour une date ou une période."""

    SEARCH_BY_AMOUNT = "SEARCH_BY_AMOUNT"
    """Transactions par montant."""

    BALANCE_INQUIRY = "BALANCE_INQUIRY"
    """Solde général actuel."""

    GREETING = "GREETING"
    """Message de salutation utilisateur."""

    UNKNOWN = "UNKNOWN"
    """Intention non reconnue."""


class EntityType(str, Enum):
    """Catégories d'entités extraites d'un message."""

    ACCOUNT = "ACCOUNT"
    """Identifiant de compte bancaire."""

    TRANSACTION = "TRANSACTION"
    """Identifiant de transaction."""

    MERCHANT = "MERCHANT"
    """Nom d'un marchand."""

    CATEGORY = "CATEGORY"
    """Catégorie de dépense."""

    DATE = "DATE"
    """Date explicite."""

    AMOUNT = "AMOUNT"
    """Valeur monétaire."""


class QueryType(str, Enum):
    """Types de requêtes supportées par le service."""

    TRANSACTION = "TRANSACTION"
    """Requête portant sur des transactions."""

    SPENDING = "SPENDING"
    """Analyse des dépenses."""

    BALANCE = "BALANCE"
    """Interrogation sur le solde."""

    GENERAL = "GENERAL"
    """Question générale sans catégorie précise."""


class ConfidenceThreshold(float, Enum):
    """Seuils de confiance utilisés pour les classifications."""

    LOW = 0.3
    """Seuil de confiance faible."""

    MEDIUM = 0.6
    """Seuil de confiance moyen."""

    HIGH = 0.9
    """Seuil de confiance élevé."""

    def __new__(cls, value: float):
        if not 0.0 <= value <= 1.0:
            raise ValueError("Le seuil doit être compris entre 0.0 et 1.0")
        obj = float.__new__(cls, value)
        obj._value_ = value
        return obj
