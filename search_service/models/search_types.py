"""
Types et énumérations pour le service de recherche.

Ce module définit tous les types, énumérations et constantes
utilisés par le service de recherche.
"""
from enum import Enum
from typing import Literal


class SearchType(str, Enum):
    """Types de recherche disponibles."""
    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class SortOrder(str, Enum):
    """Ordres de tri disponibles."""
    RELEVANCE = "relevance"
    DATE_DESC = "date_desc"
    DATE_ASC = "date_asc"
    AMOUNT_DESC = "amount_desc"
    AMOUNT_ASC = "amount_asc"


class TransactionType(str, Enum):
    """Types de transactions pour filtrage."""
    ALL = "all"
    DEBIT = "debit"
    CREDIT = "credit"


class FilterOperator(str, Enum):
    """Opérateurs pour les filtres."""
    EQ = "eq"           # Égal
    GT = "gt"           # Supérieur
    GTE = "gte"         # Supérieur ou égal
    LT = "lt"           # Inférieur
    LTE = "lte"         # Inférieur ou égal
    BETWEEN = "between" # Entre deux valeurs
    IN = "in"           # Dans une liste


class CategoryFilterType(str, Enum):
    """Types de filtres de catégorie."""
    CATEGORY_ID = "category_id"
    OPERATION_TYPE = "operation_type"


class SearchQuality(str, Enum):
    """Niveaux de qualité des résultats."""
    EXCELLENT = "excellent"
    GOOD = "good"
    MEDIUM = "medium"
    POOR = "poor"
    FAILED = "failed"


# Types littéraux pour la validation
SearchMethodType = Literal["lexical", "semantic", "hybrid"]
SortOrderType = Literal["relevance", "date_desc", "date_asc", "amount_desc", "amount_asc"]
TransactionTypeFilter = Literal["all", "debit", "credit"]

# Constantes de configuration
DEFAULT_SEARCH_LIMIT = 20
MAX_SEARCH_LIMIT = 100
DEFAULT_SIMILARITY_THRESHOLD = 0.55
MIN_SIMILARITY_THRESHOLD = 0.15
MAX_SIMILARITY_THRESHOLD = 0.95

# Pondérations par défaut pour la recherche hybride
DEFAULT_LEXICAL_WEIGHT = 0.6
DEFAULT_SEMANTIC_WEIGHT = 0.4

# Seuils de qualité
EXCELLENT_THRESHOLD = 0.9
GOOD_THRESHOLD = 0.7
MEDIUM_THRESHOLD = 0.5
POOR_THRESHOLD = 0.3

# Timeouts en secondes
ELASTICSEARCH_TIMEOUT = 5.0
QDRANT_TIMEOUT = 8.0
OPENAI_TIMEOUT = 10.0

# Cache settings
CACHE_TTL_SECONDS = 300  # 5 minutes
MAX_CACHE_SIZE = 1000

# Synonymes financiers pour expansion de requêtes
FINANCIAL_SYNONYMS = {
    "restaurant": ["restaurant", "resto", "brasserie", "cafeteria", "fast", "food"],
    "courses": ["courses", "achats", "shopping", "supermarché", "hypermarché"],
    "supermarché": ["supermarché", "hypermarché", "grande", "surface", "magasin"],
    "pharmacie": ["pharmacie", "parapharmacie", "medical", "santé"],
    "essence": ["essence", "carburant", "station", "service", "petrole", "gazole"],
    "virement": ["virement", "transfer", "transfert", "salaire", "paie"],
    "carte": ["carte", "cb", "paiement", "achat", "bancaire"],
    "abonnement": ["abonnement", "subscription", "souscription", "mensuel"]
}