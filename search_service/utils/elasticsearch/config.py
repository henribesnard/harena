"""
Configuration et constantes pour les helpers Elasticsearch.

Ce module centralise toutes les constantes, enums et configurations
utilisées par les différents helpers Elasticsearch.
"""

from enum import Enum
from typing import Dict, List

# ==================== ENUMS ====================

class QueryStrategy(str, Enum):
    """Stratégies de requête Elasticsearch."""
    EXACT = "exact"                # Correspondance exacte uniquement
    FUZZY = "fuzzy"               # Recherche floue avec tolérance d'erreurs
    WILDCARD = "wildcard"         # Recherche partielle avec patterns
    SEMANTIC = "semantic"         # Recherche sémantique avec synonymes
    HYBRID = "hybrid"             # Combinaison optimale de toutes les stratégies

class SortStrategy(str, Enum):
    """Stratégies de tri des résultats."""
    RELEVANCE = "relevance"       # Par score de pertinence (défaut)
    DATE_DESC = "date_desc"       # Par date décroissante
    DATE_ASC = "date_asc"         # Par date croissante
    AMOUNT_DESC = "amount_desc"   # Par montant décroissant
    AMOUNT_ASC = "amount_asc"     # Par montant croissant
    MERCHANT_ASC = "merchant_asc" # Par nom de marchand alphabétique

class BoostType(str, Enum):
    """Types de boost de scoring."""
    EXACT_PHRASE = "exact_phrase"
    MERCHANT_NAME = "merchant_name"
    RECENT_TRANSACTION = "recent_transaction"
    FREQUENT_MERCHANT = "frequent_merchant"
    AMOUNT_RELEVANCE = "amount_relevance"

class AggregationType(str, Enum):
    """Types d'agrégations financières."""
    CATEGORIES = "categories"
    MERCHANTS = "merchants"
    AMOUNTS = "amounts"
    TIME_SERIES = "time_series"
    TRANSACTION_TYPES = "transaction_types"
    ACCOUNTS = "accounts"

# ==================== CONSTANTES FINANCIÈRES ====================

# Synonymes financiers complets pour expansion sémantique
FINANCIAL_SYNONYMS = {
    # Opérations bancaires
    "virement": [
        "transfer", "transfert", "wire", "transfer bancaire", "vir", "virmt",
        "wire transfer", "bank transfer", "sepa", "instant transfer"
    ],
    "carte": [
        "card", "cb", "credit card", "debit card", "visa", "mastercard", 
        "carte bancaire", "payment card", "bank card", "contactless"
    ],
    "retrait": [
        "withdrawal", "cash", "atm", "distributeur", "retrait especes",
        "cash withdrawal", "atm withdrawal", "especes", "liquide"
    ],
    "depot": [
        "deposit", "dépôt", "versement", "depot especes", "credit",
        "cash deposit", "bank deposit", "versement especes"
    ],
    "prelevement": [
        "direct debit", "prélèvement automatique", "debit", "prelev",
        "automatic debit", "monthly debit", "subscription"
    ],
    "cheque": [
        "check", "chèque", "cheque bancaire", "bank check", "check payment"
    ],
    
    # Commerces et services
    "cafe": [
        "coffee", "café", "cafeteria", "cafétéria", "starbucks", "costa",
        "coffee shop", "espresso", "cappuccino", "bar", "bistrot"
    ],
    "restaurant": [
        "resto", "food", "meal", "dining", "restauration", "brasserie",
        "fast food", "takeaway", "delivery", "uber eats", "deliveroo"
    ],
    "essence": [
        "gas", "fuel", "station", "petrol", "shell", "total", "bp",
        "carburant", "gas station", "fuel station", "diesel", "sans plomb"
    ],
    "pharmacie": [
        "pharmacy", "drug store", "medication", "medicament", "parapharmacie",
        "apotheke", "chemist", "drugs", "medicine", "health"
    ],
    "supermarche": [
        "supermarket", "grocery", "courses", "carrefour", "leclerc", "auchan",
        "hypermarket", "shopping", "groceries", "food shopping", "monoprix"
    ],
    "transport": [
        "metro", "bus", "train", "taxi", "uber", "sncf", "ratp",
        "transport public", "subway", "tramway", "navigo", "public transport"
    ],
    
    # Types de paiements
    "achat": [
        "purchase", "buy", "shopping", "payment", "transaction", "sale"
    ],
    "remboursement": [
        "refund", "reimbursement", "credit", "return", "chargeback"
    ],
    "commission": [
        "fee", "charge", "commission", "cost", "frais", "tarif"
    ],
    "salaire": [
        "salary", "wage", "pay", "payroll", "paie", "revenus", "income"
    ]
}

# Champs de recherche optimisés avec boost adaptatif
FINANCIAL_SEARCH_FIELDS = {
    "primary": [
        "searchable_text^5.0",
        "primary_description^4.0", 
        "merchant_name^4.5"
    ],
    "secondary": [
        "clean_description^3.0",
        "provider_description^2.5",
        "category_description^2.0"
    ],
    "metadata": [
        "transaction_type^1.5",
        "operation_type^1.2"
    ]
}

# Champs pour highlighting optimisé
HIGHLIGHT_FIELDS = {
    "searchable_text": {
        "fragment_size": 150,
        "number_of_fragments": 3,
        "fragmenter": "span"
    },
    "primary_description": {
        "fragment_size": 100,
        "number_of_fragments": 2,
        "fragmenter": "simple"
    },
    "merchant_name": {
        "fragment_size": 50,
        "number_of_fragments": 1,
        "fragmenter": "simple"
    }
}

# Configuration des boost par défaut
DEFAULT_BOOST_VALUES = {
    BoostType.EXACT_PHRASE: 12.0,
    BoostType.MERCHANT_NAME: 8.0,
    BoostType.RECENT_TRANSACTION: 1.5,
    BoostType.FREQUENT_MERCHANT: 2.0,
    BoostType.AMOUNT_RELEVANCE: 1.2
}

# Buckets pour agrégations de montants
AMOUNT_AGGREGATION_BUCKETS = [
    {"key": "micro", "from": 0, "to": 5, "label": "< 5€"},
    {"key": "small", "from": 5, "to": 25, "label": "5€ - 25€"},
    {"key": "medium", "from": 25, "to": 100, "label": "25€ - 100€"},
    {"key": "large", "from": 100, "to": 500, "label": "100€ - 500€"},
    {"key": "xlarge", "from": 500, "to": 2000, "label": "500€ - 2000€"},
    {"key": "huge", "from": 2000, "label": "2000€+"}
]

# Champs source par défaut pour les transactions
DEFAULT_SOURCE_FIELDS = [
    "transaction_id", "primary_description", "merchant_name",
    "amount", "transaction_date", "searchable_text",
    "category_id", "account_id", "transaction_type",
    "currency_code", "operation_type", "provider_description"
]