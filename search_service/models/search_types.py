"""
Types et énumérations pour le service de recherche - VERSION CENTRALISÉE.

Ce module définit tous les types, énumérations et constantes
utilisés par le service de recherche.

AMÉLIORATION:
- Suppression des constantes hardcodées
- Utilisation de la configuration centralisée
- Fonctions d'accès pour obtenir les valeurs depuis config_service
"""
from enum import Enum
from typing import Literal, Dict, List


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


# ==========================================
# 🎯 TYPES LITTÉRAUX POUR LA VALIDATION
# ==========================================

SearchMethodType = Literal["lexical", "semantic", "hybrid"]
SortOrderType = Literal["relevance", "date_desc", "date_asc", "amount_desc", "amount_asc"]
TransactionTypeFilter = Literal["all", "debit", "credit"]


# ==========================================
# 🔧 FONCTIONS D'ACCÈS À LA CONFIGURATION CENTRALISÉE
# ==========================================

def get_default_search_limit() -> int:
    """Retourne la limite de recherche par défaut depuis la config centralisée."""
    from config_service.config import settings
    return settings.DEFAULT_SEARCH_LIMIT


def get_max_search_limit() -> int:
    """Retourne la limite maximale de recherche depuis la config centralisée."""
    from config_service.config import settings
    return settings.MAX_SEARCH_LIMIT


def get_default_similarity_threshold() -> float:
    """
    🚨 FONCTION CRITIQUE - Retourne le seuil de similarité par défaut.
    
    C'est LE paramètre qui cause vos problèmes de recherche vide !
    """
    from config_service.config import settings
    return settings.SIMILARITY_THRESHOLD_DEFAULT


def get_min_similarity_threshold() -> float:
    """Retourne le seuil de similarité minimum."""
    from config_service.config import settings
    return settings.SIMILARITY_THRESHOLD_LOOSE


def get_max_similarity_threshold() -> float:
    """Retourne le seuil de similarité maximum."""
    from config_service.config import settings
    return settings.SIMILARITY_THRESHOLD_STRICT


def get_similarity_threshold_range() -> Dict[str, float]:
    """Retourne tous les seuils de similarité disponibles."""
    from config_service.config import settings
    return {
        "min": settings.SIMILARITY_THRESHOLD_LOOSE,
        "default": settings.SIMILARITY_THRESHOLD_DEFAULT,
        "max": settings.SIMILARITY_THRESHOLD_STRICT
    }


def get_default_weights() -> Dict[str, float]:
    """Retourne les poids par défaut pour la recherche hybride."""
    from config_service.config import settings
    return {
        "lexical": settings.DEFAULT_LEXICAL_WEIGHT,
        "semantic": settings.DEFAULT_SEMANTIC_WEIGHT
    }


def get_quality_thresholds() -> Dict[str, float]:
    """Retourne les seuils de qualité depuis la config centralisée."""
    from config_service.config import settings
    return {
        "excellent": settings.QUALITY_EXCELLENT_THRESHOLD,
        "good": settings.QUALITY_GOOD_THRESHOLD,
        "medium": settings.QUALITY_MEDIUM_THRESHOLD,
        "poor": settings.QUALITY_POOR_THRESHOLD
    }


def get_timeout_config() -> Dict[str, float]:
    """Retourne la configuration des timeouts."""
    from config_service.config import settings
    return {
        "elasticsearch": settings.ELASTICSEARCH_TIMEOUT,
        "qdrant": settings.QDRANT_TIMEOUT,
        "openai": settings.OPENAI_TIMEOUT,
        "search": settings.SEARCH_TIMEOUT
    }


def get_cache_config() -> Dict[str, int]:
    """Retourne la configuration du cache."""
    from config_service.config import settings
    return {
        "search_ttl": settings.SEARCH_CACHE_TTL,
        "search_max_size": settings.SEARCH_CACHE_MAX_SIZE,
        "embedding_ttl": settings.EMBEDDING_CACHE_TTL,
        "embedding_max_size": settings.EMBEDDING_CACHE_MAX_SIZE
    }


# ==========================================
# 📚 SYNONYMES FINANCIERS (STATIQUES - OK)
# ==========================================

# Ces données sont statiques et ne nécessitent pas de configuration
FINANCIAL_SYNONYMS = {
    "restaurant": ["restaurant", "resto", "brasserie", "cafeteria", "fast", "food"],
    "courses": ["courses", "achats", "shopping", "supermarché", "hypermarché"],
    "supermarché": ["supermarché", "hypermarché", "grande", "surface", "magasin"],
    "pharmacie": ["pharmacie", "parapharmacie", "medical", "santé"],
    "essence": ["essence", "carburant", "station", "service", "petrole", "gazole"],
    "virement": ["virement", "transfer", "transfert", "salaire", "paie"],
    "carte": ["carte", "cb", "paiement", "achat", "bancaire"],
    "abonnement": ["abonnement", "subscription", "souscription", "mensuel"],
    "transport": ["transport", "metro", "bus", "train", "taxi", "uber"],
    "loisirs": ["loisirs", "cinema", "theatre", "sport", "divertissement"],
    "banque": ["banque", "frais", "commission", "agios", "bancaires"],
    "assurance": ["assurance", "mutuelle", "garantie", "protection"],
    "impots": ["impots", "taxes", "fiscal", "tresor", "public"],
    "electricite": ["electricite", "edf", "energie", "gaz", "engie"],
    "internet": ["internet", "box", "wifi", "abonnement", "telecom"],
    "telephone": ["telephone", "mobile", "forfait", "operateur"]
}


# ==========================================
# 🎯 FONCTIONS UTILITAIRES
# ==========================================

def validate_search_type(search_type: str) -> SearchType:
    """Valide et convertit un type de recherche."""
    try:
        return SearchType(search_type)
    except ValueError:
        return SearchType.HYBRID  # Fallback par défaut


def validate_sort_order(sort_order: str) -> SortOrder:
    """Valide et convertit un ordre de tri."""
    try:
        return SortOrder(sort_order)
    except ValueError:
        return SortOrder.RELEVANCE  # Fallback par défaut


def validate_similarity_threshold(threshold: float) -> float:
    """
    Valide un seuil de similarité selon les limites configurées.
    
    Args:
        threshold: Seuil à valider
        
    Returns:
        Seuil validé dans les limites autorisées
    """
    from config_service.config import settings
    
    min_threshold = settings.SIMILARITY_THRESHOLD_LOOSE
    max_threshold = settings.SIMILARITY_THRESHOLD_STRICT
    
    if threshold < min_threshold:
        return min_threshold
    elif threshold > max_threshold:
        return max_threshold
    else:
        return threshold


def validate_search_limit(limit: int) -> int:
    """
    Valide une limite de recherche selon la configuration.
    
    Args:
        limit: Limite à valider
        
    Returns:
        Limite validée dans les bornes autorisées
    """
    from config_service.config import settings
    
    if limit <= 0:
        return settings.DEFAULT_SEARCH_LIMIT
    elif limit > settings.MAX_SEARCH_LIMIT:
        return settings.MAX_SEARCH_LIMIT
    else:
        return limit


def get_search_quality_from_score(score: float) -> SearchQuality:
    """
    Détermine la qualité de recherche basée sur un score.
    
    Args:
        score: Score de qualité (0-1)
        
    Returns:
        Niveau de qualité correspondant
    """
    thresholds = get_quality_thresholds()
    
    if score >= thresholds["excellent"]:
        return SearchQuality.EXCELLENT
    elif score >= thresholds["good"]:
        return SearchQuality.GOOD
    elif score >= thresholds["medium"]:
        return SearchQuality.MEDIUM
    elif score >= thresholds["poor"]:
        return SearchQuality.POOR
    else:
        return SearchQuality.FAILED


def get_financial_synonyms(term: str) -> List[str]:
    """
    Retourne les synonymes d'un terme financier.
    
    Args:
        term: Terme à rechercher
        
    Returns:
        Liste des synonymes (inclut le terme original)
    """
    term_lower = term.lower()
    
    # Chercher dans les clés principales
    if term_lower in FINANCIAL_SYNONYMS:
        return FINANCIAL_SYNONYMS[term_lower]
    
    # Chercher dans les valeurs
    for key, synonyms in FINANCIAL_SYNONYMS.items():
        if term_lower in [s.lower() for s in synonyms]:
            return synonyms
    
    # Retourner le terme original si pas de synonymes
    return [term]


def expand_query_with_synonyms(query: str) -> List[str]:
    """
    Expand une requête avec des synonymes financiers.
    
    Args:
        query: Requête à étendre
        
    Returns:
        Liste des termes étendus avec synonymes
    """
    words = query.lower().split()
    expanded_terms = []
    
    for word in words:
        synonyms = get_financial_synonyms(word)
        expanded_terms.extend(synonyms)
    
    # Supprimer les doublons en gardant l'ordre
    seen = set()
    unique_terms = []
    for term in expanded_terms:
        if term not in seen:
            seen.add(term)
            unique_terms.append(term)
    
    return unique_terms


# ==========================================
# 🔧 FONCTIONS DE COMPATIBILITÉ
# ==========================================

# Ces fonctions permettent la migration progressive depuis les anciennes constantes

def get_legacy_constant(constant_name: str):
    """
    Fonction de compatibilité pour l'ancien code utilisant les constantes.
    
    ⚠️ DEPRECATED: Utilisez les fonctions get_* spécifiques à la place.
    """
    mapping = {
        "DEFAULT_SEARCH_LIMIT": get_default_search_limit,
        "MAX_SEARCH_LIMIT": get_max_search_limit,
        "DEFAULT_SIMILARITY_THRESHOLD": get_default_similarity_threshold,
        "MIN_SIMILARITY_THRESHOLD": get_min_similarity_threshold,
        "MAX_SIMILARITY_THRESHOLD": get_max_similarity_threshold,
        "DEFAULT_LEXICAL_WEIGHT": lambda: get_default_weights()["lexical"],
        "DEFAULT_SEMANTIC_WEIGHT": lambda: get_default_weights()["semantic"],
        "EXCELLENT_THRESHOLD": lambda: get_quality_thresholds()["excellent"],
        "GOOD_THRESHOLD": lambda: get_quality_thresholds()["good"],
        "MEDIUM_THRESHOLD": lambda: get_quality_thresholds()["medium"],
        "POOR_THRESHOLD": lambda: get_quality_thresholds()["poor"],
        "ELASTICSEARCH_TIMEOUT": lambda: get_timeout_config()["elasticsearch"],
        "QDRANT_TIMEOUT": lambda: get_timeout_config()["qdrant"],
        "OPENAI_TIMEOUT": lambda: get_timeout_config()["openai"],
        "CACHE_TTL_SECONDS": lambda: get_cache_config()["search_ttl"],
        "MAX_CACHE_SIZE": lambda: get_cache_config()["search_max_size"]
    }
    
    if constant_name in mapping:
        return mapping[constant_name]()
    else:
        raise ValueError(f"Unknown legacy constant: {constant_name}")


# ==========================================
# 🎯 EXPORTS POUR COMPATIBILITÉ
# ==========================================

# Pour faciliter la migration, on peut temporairement exposer les valeurs
# via des propriétés qui lisent la config centralisée

class _LegacyConstants:
    """Classe pour exposer les anciennes constantes via des propriétés."""
    
    @property
    def DEFAULT_SEARCH_LIMIT(self) -> int:
        return get_default_search_limit()
    
    @property 
    def MAX_SEARCH_LIMIT(self) -> int:
        return get_max_search_limit()
    
    @property
    def DEFAULT_SIMILARITY_THRESHOLD(self) -> float:
        return get_default_similarity_threshold()
    
    @property
    def MIN_SIMILARITY_THRESHOLD(self) -> float:
        return get_min_similarity_threshold()
    
    @property
    def MAX_SIMILARITY_THRESHOLD(self) -> float:
        return get_max_similarity_threshold()
    
    @property
    def DEFAULT_LEXICAL_WEIGHT(self) -> float:
        return get_default_weights()["lexical"]
    
    @property
    def DEFAULT_SEMANTIC_WEIGHT(self) -> float:
        return get_default_weights()["semantic"]
    
    @property
    def EXCELLENT_THRESHOLD(self) -> float:
        return get_quality_thresholds()["excellent"]
    
    @property
    def GOOD_THRESHOLD(self) -> float:
        return get_quality_thresholds()["good"]
    
    @property
    def MEDIUM_THRESHOLD(self) -> float:
        return get_quality_thresholds()["medium"]
    
    @property
    def POOR_THRESHOLD(self) -> float:
        return get_quality_thresholds()["poor"]


# Instance pour compatibilité temporaire - UNIQUE DÉFINITION
_constants = _LegacyConstants()

# Exposer TOUTES les constantes pour l'ancien code (migration progressive)
DEFAULT_SEARCH_LIMIT = _constants.DEFAULT_SEARCH_LIMIT
MAX_SEARCH_LIMIT = _constants.MAX_SEARCH_LIMIT
DEFAULT_SIMILARITY_THRESHOLD = _constants.DEFAULT_SIMILARITY_THRESHOLD
MIN_SIMILARITY_THRESHOLD = _constants.MIN_SIMILARITY_THRESHOLD  
MAX_SIMILARITY_THRESHOLD = _constants.MAX_SIMILARITY_THRESHOLD  
DEFAULT_LEXICAL_WEIGHT = _constants.DEFAULT_LEXICAL_WEIGHT
DEFAULT_SEMANTIC_WEIGHT = _constants.DEFAULT_SEMANTIC_WEIGHT
EXCELLENT_THRESHOLD = _constants.EXCELLENT_THRESHOLD
GOOD_THRESHOLD = _constants.GOOD_THRESHOLD
MEDIUM_THRESHOLD = _constants.MEDIUM_THRESHOLD
POOR_THRESHOLD = _constants.POOR_THRESHOLD