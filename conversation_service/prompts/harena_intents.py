"""
Taxonomie optimisée des intentions Harena avec métadonnées enrichies
Version corrigée avec SEARCH_BY_OPERATION_TYPE amélioré
"""
from enum import Enum
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from functools import lru_cache


class HarenaIntentType(str, Enum):
    """
    Énumération complète des intentions financières Harena
    Basée sur INTENTS.md avec optimisations et métadonnées
    """
    
    # === TRANSACTIONS ET RECHERCHE ===
    TRANSACTION_SEARCH = "TRANSACTION_SEARCH"
    SEARCH_BY_DATE = "SEARCH_BY_DATE" 
    SEARCH_BY_AMOUNT = "SEARCH_BY_AMOUNT"
    SEARCH_BY_MERCHANT = "SEARCH_BY_MERCHANT"
    SEARCH_BY_CATEGORY = "SEARCH_BY_CATEGORY"
    SEARCH_BY_AMOUNT_AND_DATE = "SEARCH_BY_AMOUNT_AND_DATE"
    SEARCH_BY_OPERATION_TYPE = "SEARCH_BY_OPERATION_TYPE"
    SEARCH_BY_TEXT = "SEARCH_BY_TEXT"
    COUNT_TRANSACTIONS = "COUNT_TRANSACTIONS"
    MERCHANT_INQUIRY = "MERCHANT_INQUIRY"
    FILTER_REQUEST = "FILTER_REQUEST"
    
    # === ANALYSE ET INSIGHTS ===
    SPENDING_ANALYSIS = "SPENDING_ANALYSIS"
    SPENDING_ANALYSIS_BY_CATEGORY = "SPENDING_ANALYSIS_BY_CATEGORY"
    SPENDING_ANALYSIS_BY_PERIOD = "SPENDING_ANALYSIS_BY_PERIOD"
    SPENDING_COMPARISON = "SPENDING_COMPARISON"
    TREND_ANALYSIS = "TREND_ANALYSIS"
    CATEGORY_ANALYSIS = "CATEGORY_ANALYSIS"
    COMPARISON_QUERY = "COMPARISON_QUERY"
    
    # === SOLDES ET COMPTES ===
    BALANCE_INQUIRY = "BALANCE_INQUIRY"
    ACCOUNT_BALANCE_SPECIFIC = "ACCOUNT_BALANCE_SPECIFIC"
    BALANCE_EVOLUTION = "BALANCE_EVOLUTION"
    
    # === CONVERSATIONNEL ===
    GREETING = "GREETING"
    CONFIRMATION = "CONFIRMATION"
    CLARIFICATION = "CLARIFICATION"
    GENERAL_QUESTION = "GENERAL_QUESTION"
    
    # === NON SUPPORTÉES (importantes pour training) ===
    TRANSFER_REQUEST = "TRANSFER_REQUEST"
    PAYMENT_REQUEST = "PAYMENT_REQUEST"
    CARD_BLOCK = "CARD_BLOCK"
    BUDGET_INQUIRY = "BUDGET_INQUIRY"
    GOAL_TRACKING = "GOAL_TRACKING"
    EXPORT_REQUEST = "EXPORT_REQUEST"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    
    # === AMBIGUËS ET ERREURS ===
    UNCLEAR_INTENT = "UNCLEAR_INTENT"
    UNKNOWN = "UNKNOWN"
    TEST_INTENT = "TEST_INTENT"
    ERROR = "ERROR"


@dataclass(frozen=True)
class IntentMetadata:
    """Métadonnées enrichies pour chaque intention"""
    description: str
    category: str
    is_supported: bool
    complexity: str  # "simple", "medium", "complex"
    frequency: str   # "very_high", "high", "medium", "low", "very_low"
    keywords: Tuple[str, ...] = field(default_factory=tuple)
    examples: Tuple[str, ...] = field(default_factory=tuple)
    related_intents: Tuple[str, ...] = field(default_factory=tuple)
    processing_hints: Dict[str, any] = field(default_factory=dict)
    confidence_threshold: float = 0.5
    
    def __post_init__(self):
        """Validation post-initialisation"""
        if self.confidence_threshold < 0.0 or self.confidence_threshold > 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")


# Métadonnées détaillées pour chaque intention
INTENT_METADATA: Dict[HarenaIntentType, IntentMetadata] = {
    
    # === TRANSACTIONS ET RECHERCHE ===
    HarenaIntentType.TRANSACTION_SEARCH: IntentMetadata(
        description="Rechercher toutes transactions sans filtre spécifique",
        category="FINANCIAL_QUERY",
        is_supported=True,
        complexity="simple",
        frequency="medium",
        keywords=("transactions", "historique", "toutes", "liste"),
        examples=("Mes transactions", "Historique complet", "Toutes mes opérations"),
        related_intents=("SEARCH_BY_DATE", "FILTER_REQUEST"),
        confidence_threshold=0.7
    ),
    
    HarenaIntentType.SEARCH_BY_DATE: IntentMetadata(
        description="Rechercher transactions pour une date ou période précise",
        category="FINANCIAL_QUERY",
        is_supported=True,
        complexity="medium",
        frequency="very_high",
        keywords=("hier", "aujourd'hui", "semaine", "mois", "janvier", "février", "date"),
        examples=("Mes dépenses d'hier", "Transactions ce mois", "Achats en janvier"),
        related_intents=("SEARCH_BY_AMOUNT_AND_DATE", "SPENDING_ANALYSIS_BY_PERIOD"),
        processing_hints={"requires_date_parsing": True},
        confidence_threshold=0.8
    ),
    
    HarenaIntentType.SEARCH_BY_AMOUNT: IntentMetadata(
        description="Rechercher transactions par montant ou plage de montants",
        category="FINANCIAL_QUERY",
        is_supported=True,
        complexity="medium",
        frequency="medium",
        keywords=("gros", "petites", "supérieur", "inférieur", "entre", "euros", "€"),
        examples=("Mes gros achats", "Dépenses > 100€", "Achats entre 50 et 200€"),
        related_intents=("SEARCH_BY_AMOUNT_AND_DATE",),
        processing_hints={"requires_amount_parsing": True},
        confidence_threshold=0.75
    ),
    
    HarenaIntentType.SEARCH_BY_MERCHANT: IntentMetadata(
        description="Rechercher transactions liées à un marchand précis",
        category="FINANCIAL_QUERY", 
        is_supported=True,
        complexity="simple",
        frequency="very_high",
        keywords=("amazon", "carrefour", "mcdonald", "uber", "netflix", "chez"),
        examples=("Mes achats Amazon", "Dépenses Carrefour", "Transactions Netflix"),
        related_intents=("MERCHANT_INQUIRY",),
        processing_hints={"requires_merchant_extraction": True},
        confidence_threshold=0.85
    ),
    
    HarenaIntentType.SEARCH_BY_CATEGORY: IntentMetadata(
        description="Rechercher transactions par catégorie de dépense",
        category="FINANCIAL_QUERY",
        is_supported=True,
        complexity="simple", 
        frequency="high",
        keywords=("restaurant", "courses", "transport", "essence", "vêtements", "santé"),
        examples=("Mes dépenses restaurants", "Frais de transport", "Achats vêtements"),
        related_intents=("CATEGORY_ANALYSIS", "SPENDING_ANALYSIS_BY_CATEGORY"),
        processing_hints={"requires_category_mapping": True},
        confidence_threshold=0.8
    ),
    
    HarenaIntentType.SEARCH_BY_AMOUNT_AND_DATE: IntentMetadata(
        description="Rechercher transactions combinant critères montant et date",
        category="FINANCIAL_QUERY",
        is_supported=True,
        complexity="complex",
        frequency="low",
        keywords=("gros", "achats", "hier", "semaine", "mois", "supérieur", "date"),
        examples=("Gros achats ce mois", "Dépenses > 50€ cette semaine"),
        related_intents=("SEARCH_BY_DATE", "SEARCH_BY_AMOUNT"),
        processing_hints={"requires_amount_parsing": True, "requires_date_parsing": True},
        confidence_threshold=0.7
    ),
    
    HarenaIntentType.SEARCH_BY_OPERATION_TYPE: IntentMetadata(
        description="Rechercher transactions existantes par type d'opération (virements, prélèvements, CB, etc.)",
        category="FINANCIAL_QUERY",
        is_supported=True,
        complexity="medium",
        frequency="medium",  # Augmenté de "low" à "medium" car plus fréquent que prévu
        keywords=("carte", "virement", "prélèvement", "chèque", "espèces", "cb", "combien", "mes", "historique", "liste", "nombre", "quels"),
        examples=(
            "Combien ai-je fait de virements en mai ?", 
            "Mes virements du mois dernier",
            "Quels sont mes prélèvements automatiques ?",
            "Historique de mes paiements par carte",
            "Paiements par carte", 
            "Virements reçus", 
            "Prélèvements automatiques"
        ),
        related_intents=("TRANSACTION_SEARCH", "COUNT_TRANSACTIONS"),
        processing_hints={"requires_operation_type_filtering": True, "supports_search_only": True},
        confidence_threshold=0.8  # Augmenté de 0.75 à 0.8 pour plus de précision
    ),
    
    HarenaIntentType.SEARCH_BY_TEXT: IntentMetadata(
        description="Recherche textuelle libre dans les descriptions",
        category="FINANCIAL_QUERY",
        is_supported=True,
        complexity="simple",
        frequency="medium",
        keywords=("contenant", "avec", "description", "libellé"),
        examples=("Transactions avec 'subscription'", "Libellés contenant 'Paris'"),
        related_intents=("TRANSACTION_SEARCH",),
        processing_hints={"requires_text_search": True},
        confidence_threshold=0.7
    ),
    
    HarenaIntentType.COUNT_TRANSACTIONS: IntentMetadata(
        description="Compter les transactions correspondant à des critères",
        category="FINANCIAL_QUERY",
        is_supported=True,
        complexity="simple",
        frequency="medium",  # Augmenté de "low" à "medium"
        keywords=("combien", "nombre", "compteur", "total", "transactions"),
        examples=("Combien de transactions Amazon ?", "Nombre d'achats ce mois", "Combien de virements ?"),
        related_intents=("TRANSACTION_SEARCH", "SEARCH_BY_OPERATION_TYPE"),
        processing_hints={"requires_counting": True},
        confidence_threshold=0.8
    ),
    
    HarenaIntentType.MERCHANT_INQUIRY: IntentMetadata(
        description="Analyse détaillée des transactions avec un marchand",
        category="FINANCIAL_QUERY",
        is_supported=True,
        complexity="medium",
        frequency="medium",
        keywords=("analyse", "détails", "statistiques", "chez", "avec"),
        examples=("Analyse Amazon", "Mes statistiques Netflix", "Détails Carrefour"),
        related_intents=("SEARCH_BY_MERCHANT", "SPENDING_ANALYSIS"),
        processing_hints={"requires_aggregation": True},
        confidence_threshold=0.75
    ),
    
    HarenaIntentType.FILTER_REQUEST: IntentMetadata(
        description="Raffiner une requête existante avec des filtres",
        category="FINANCIAL_QUERY",
        is_supported=True,
        complexity="complex",
        frequency="low",
        keywords=("filtrer", "seulement", "exclure", "sans", "uniquement"),
        examples=("Seulement les gros montants", "Exclure les remboursements"),
        related_intents=("TRANSACTION_SEARCH",),
        confidence_threshold=0.6
    ),
    
    # === ANALYSE ET INSIGHTS ===
    HarenaIntentType.SPENDING_ANALYSIS: IntentMetadata(
        description="Analyse globale des dépenses sur une période",
        category="SPENDING_ANALYSIS",
        is_supported=True,
        complexity="medium",
        frequency="high",
        keywords=("analyse", "dépenses", "combien", "dépensé", "bilan", "résumé"),
        examples=("Analyse de mes dépenses", "Combien j'ai dépensé ?", "Bilan mensuel"),
        related_intents=("SPENDING_ANALYSIS_BY_CATEGORY", "SPENDING_ANALYSIS_BY_PERIOD"),
        processing_hints={"requires_aggregation": True, "requires_insights": True},
        confidence_threshold=0.8
    ),
    
    HarenaIntentType.SPENDING_ANALYSIS_BY_CATEGORY: IntentMetadata(
        description="Répartition des dépenses par catégorie",
        category="SPENDING_ANALYSIS",
        is_supported=True,
        complexity="medium",
        frequency="high",
        keywords=("répartition", "catégories", "breakdown", "par type", "distribution"),
        examples=("Répartition par catégories", "Mes dépenses par type", "Distribution"),
        related_intents=("CATEGORY_ANALYSIS", "SPENDING_ANALYSIS"),
        processing_hints={"requires_categorization": True, "requires_aggregation": True},
        confidence_threshold=0.8
    ),
    
    HarenaIntentType.SPENDING_ANALYSIS_BY_PERIOD: IntentMetadata(
        description="Analyse des dépenses par période temporelle",
        category="SPENDING_ANALYSIS",
        is_supported=True,
        complexity="medium",
        frequency="medium",
        keywords=("mois", "semaine", "période", "évolution", "mensuel", "hebdomadaire"),
        examples=("Dépenses par mois", "Évolution hebdomadaire", "Analyse mensuelle"),
        related_intents=("TREND_ANALYSIS", "SPENDING_COMPARISON"),
        processing_hints={"requires_time_grouping": True},
        confidence_threshold=0.75
    ),
    
    HarenaIntentType.SPENDING_COMPARISON: IntentMetadata(
        description="Comparaison de dépenses entre périodes ou catégories",
        category="SPENDING_ANALYSIS",
        is_supported=True,
        complexity="complex",
        frequency="medium",
        keywords=("comparaison", "vs", "contre", "différence", "par rapport"),
        examples=("Comparaison avec le mois dernier", "Différence vs année passée"),
        related_intents=("TREND_ANALYSIS", "COMPARISON_QUERY"),
        processing_hints={"requires_comparison": True},
        confidence_threshold=0.7
    ),
    
    HarenaIntentType.TREND_ANALYSIS: IntentMetadata(
        description="Analyse des tendances d'évolution des dépenses",
        category="SPENDING_ANALYSIS",
        is_supported=True,
        complexity="complex",
        frequency="low",
        keywords=("tendance", "évolution", "progression", "augmentation", "diminution"),
        examples=("Tendance de mes dépenses", "Évolution sur 6 mois"),
        related_intents=("SPENDING_COMPARISON",),
        processing_hints={"requires_trend_calculation": True},
        confidence_threshold=0.7
    ),
    
    HarenaIntentType.CATEGORY_ANALYSIS: IntentMetadata(
        description="Analyse détaillée par catégories de dépenses",
        category="SPENDING_ANALYSIS",
        is_supported=True,
        complexity="medium",
        frequency="medium",
        keywords=("catégorie", "restaurant", "transport", "détail", "analyse"),
        examples=("Analyse détaillée restaurants", "Catégorie transport"),
        related_intents=("SPENDING_ANALYSIS_BY_CATEGORY", "SEARCH_BY_CATEGORY"),
        confidence_threshold=0.75
    ),
    
    HarenaIntentType.COMPARISON_QUERY: IntentMetadata(
        description="Comparaison ciblée entre éléments spécifiques",
        category="SPENDING_ANALYSIS",
        is_supported=True,
        complexity="complex",
        frequency="low",
        keywords=("comparer", "vs", "différence", "mieux", "plus", "moins"),
        examples=("Comparer Amazon vs autres", "Différence restaurants/courses"),
        related_intents=("SPENDING_COMPARISON",),
        confidence_threshold=0.65
    ),
    
    # === SOLDES ET COMPTES ===
    HarenaIntentType.BALANCE_INQUIRY: IntentMetadata(
        description="Consulter le solde général actuel",
        category="ACCOUNT_BALANCE",
        is_supported=True,
        complexity="simple",
        frequency="very_high",
        keywords=("solde", "combien", "compte", "argent", "reste", "disponible"),
        examples=("Mon solde", "Combien j'ai ?", "Solde actuel"),
        related_intents=("ACCOUNT_BALANCE_SPECIFIC",),
        confidence_threshold=0.9
    ),
    
    HarenaIntentType.ACCOUNT_BALANCE_SPECIFIC: IntentMetadata(
        description="Consulter le solde d'un compte précis",
        category="ACCOUNT_BALANCE",
        is_supported=True,
        complexity="medium",
        frequency="medium",
        keywords=("compte", "courant", "épargne", "livret", "solde"),
        examples=("Solde compte courant", "Mon livret A"),
        related_intents=("BALANCE_INQUIRY",),
        confidence_threshold=0.8
    ),
    
    HarenaIntentType.BALANCE_EVOLUTION: IntentMetadata(
        description="Historique et évolution du solde",
        category="ACCOUNT_BALANCE",
        is_supported=True,
        complexity="medium",
        frequency="low",
        keywords=("évolution", "historique", "progression", "solde", "temps"),
        examples=("Évolution de mon solde", "Historique du compte"),
        related_intents=("BALANCE_INQUIRY", "TREND_ANALYSIS"),
        confidence_threshold=0.7
    ),
    
    # === CONVERSATIONNEL ===
    HarenaIntentType.GREETING: IntentMetadata(
        description="Salutations et politesses d'ouverture",
        category="CONVERSATIONAL",
        is_supported=True,
        complexity="simple",
        frequency="very_high",
        keywords=("bonjour", "salut", "hello", "bonsoir", "hey", "coucou"),
        examples=("Bonjour", "Salut Harena", "Hello"),
        related_intents=(),
        confidence_threshold=0.95
    ),
    
    HarenaIntentType.CONFIRMATION: IntentMetadata(
        description="Confirmations et remerciements",
        category="CONVERSATIONAL",
        is_supported=True,
        complexity="simple",
        frequency="high",
        keywords=("merci", "parfait", "ok", "bien", "super", "nickel"),
        examples=("Merci", "Parfait", "C'est bon"),
        related_intents=(),
        confidence_threshold=0.9
    ),
    
    HarenaIntentType.CLARIFICATION: IntentMetadata(
        description="Demandes de précision ou d'explication",
        category="CONVERSATIONAL",
        is_supported=True,
        complexity="medium",
        frequency="medium",
        keywords=("comment", "pourquoi", "expliquer", "préciser", "détail"),
        examples=("Comment ça marche ?", "Peux-tu expliquer ?"),
        related_intents=("GENERAL_QUESTION",),
        confidence_threshold=0.7
    ),
    
    HarenaIntentType.GENERAL_QUESTION: IntentMetadata(
        description="Questions générales sur l'utilisation",
        category="CONVERSATIONAL",
        is_supported=True,
        complexity="medium",
        frequency="medium",
        keywords=("question", "aide", "help", "utiliser", "fonctionner"),
        examples=("Comment utiliser ?", "Aide générale"),
        related_intents=("CLARIFICATION",),
        confidence_threshold=0.6
    ),
    
    # === NON SUPPORTÉES ===
    HarenaIntentType.TRANSFER_REQUEST: IntentMetadata(
        description="Demande d'exécution de virement (non supporté - action bancaire)",
        category="UNSUPPORTED",
        is_supported=False,
        complexity="simple",
        frequency="medium",
        keywords=("faire", "effectuer", "virer", "transférer", "envoyer", "argent"),
        examples=("Faire un virement", "Virer 100€ à Paul", "Effectuer un transfert"),
        related_intents=(),
        processing_hints={"action_request": True, "requires_banking_operation": True},
        confidence_threshold=0.9
    ),
    
    HarenaIntentType.PAYMENT_REQUEST: IntentMetadata(
        description="Demande d'exécution de paiement (non supporté - action bancaire)",
        category="UNSUPPORTED",
        is_supported=False,
        complexity="simple",
        frequency="medium",
        keywords=("payer", "effectuer", "paiement", "facture", "régler"),
        examples=("Payer ma facture", "Effectuer un paiement"),
        related_intents=(),
        processing_hints={"action_request": True, "requires_banking_operation": True},
        confidence_threshold=0.9
    ),
    
    HarenaIntentType.CARD_BLOCK: IntentMetadata(
        description="Blocage de carte (non supporté - action sécuritaire)",
        category="UNSUPPORTED",
        is_supported=False,
        complexity="simple",
        frequency="low",
        keywords=("bloquer", "carte", "cb", "suspendre", "annuler"),
        examples=("Bloquer ma carte", "Suspendre ma CB"),
        related_intents=(),
        processing_hints={"action_request": True, "requires_security_operation": True},
        confidence_threshold=0.95
    ),
    
    HarenaIntentType.BUDGET_INQUIRY: IntentMetadata(
        description="Consultation budget (non supporté)",
        category="UNSUPPORTED",
        is_supported=False,
        complexity="medium",
        frequency="low",
        keywords=("budget", "enveloppe", "allocation", "limite"),
        examples=("Mon budget", "Où en est mon budget ?"),
        related_intents=(),
        confidence_threshold=0.8
    ),
    
    HarenaIntentType.GOAL_TRACKING: IntentMetadata(
        description="Suivi objectifs (non supporté)",
        category="UNSUPPORTED",
        is_supported=False,
        complexity="medium",
        frequency="low",
        keywords=("objectif", "épargne", "but", "cible", "économiser"),
        examples=("Mon objectif épargne", "Suivi de mes buts"),
        related_intents=(),
        confidence_threshold=0.8
    ),
    
    HarenaIntentType.EXPORT_REQUEST: IntentMetadata(
        description="Export de données (non supporté)",
        category="UNSUPPORTED",
        is_supported=False,
        complexity="simple",
        frequency="very_low",
        keywords=("exporter", "télécharger", "csv", "pdf", "fichier"),
        examples=("Exporter mes données", "Télécharger en CSV"),
        related_intents=(),
        confidence_threshold=0.85
    ),
    
    HarenaIntentType.OUT_OF_SCOPE: IntentMetadata(
        description="Hors domaine financier",
        category="UNSUPPORTED",
        is_supported=False,
        complexity="simple",
        frequency="low",
        keywords=(),  # Pas de mots-clés spécifiques
        examples=("Météo aujourd'hui", "Recette de cuisine"),
        related_intents=(),
        confidence_threshold=0.8
    ),
    
    # === AMBIGUËS ET ERREURS ===
    HarenaIntentType.UNCLEAR_INTENT: IntentMetadata(
        description="Intention ambiguë ou non claire",
        category="UNCLEAR_INTENT",
        is_supported=False,
        complexity="simple",
        frequency="medium",
        keywords=("euh", "help", "aide", "sais pas", "comprends pas"),
        examples=("Euh... aide moi", "Je sais pas", "Comment ça marche ?"),
        related_intents=(),
        confidence_threshold=0.6
    ),
    
    HarenaIntentType.UNKNOWN: IntentMetadata(
        description="Message incompréhensible",
        category="UNCLEAR_INTENT",
        is_supported=False,
        complexity="simple",
        frequency="low",
        keywords=(),
        examples=("azerty qwerty", "123 456 !!!", "jdhgkjdhgk"),
        related_intents=(),
        confidence_threshold=0.95
    ),
    
    HarenaIntentType.TEST_INTENT: IntentMetadata(
        description="Message de test",
        category="UNCLEAR_INTENT",
        is_supported=False,
        complexity="simple",
        frequency="very_low",
        keywords=("test", "testing", "debug", "check"),
        examples=("Test", "Testing 123", "Debug mode"),
        related_intents=(),
        confidence_threshold=0.9
    ),
    
    HarenaIntentType.ERROR: IntentMetadata(
        description="Erreur de traitement",
        category="UNCLEAR_INTENT",
        is_supported=False,
        complexity="simple",
        frequency="very_low",
        keywords=(),
        examples=(),
        related_intents=(),
        confidence_threshold=0.99
    )
}


# Catégories organisées avec métadonnées
INTENT_CATEGORIES: Dict[str, List[HarenaIntentType]] = {
    "FINANCIAL_QUERY": [
        HarenaIntentType.TRANSACTION_SEARCH,
        HarenaIntentType.SEARCH_BY_DATE,
        HarenaIntentType.SEARCH_BY_AMOUNT,
        HarenaIntentType.SEARCH_BY_MERCHANT,
        HarenaIntentType.SEARCH_BY_CATEGORY,
        HarenaIntentType.SEARCH_BY_AMOUNT_AND_DATE,
        HarenaIntentType.SEARCH_BY_OPERATION_TYPE,
        HarenaIntentType.SEARCH_BY_TEXT,
        HarenaIntentType.COUNT_TRANSACTIONS,
        HarenaIntentType.MERCHANT_INQUIRY,
        HarenaIntentType.FILTER_REQUEST
    ],
    "SPENDING_ANALYSIS": [
        HarenaIntentType.SPENDING_ANALYSIS,
        HarenaIntentType.SPENDING_ANALYSIS_BY_CATEGORY,
        HarenaIntentType.SPENDING_ANALYSIS_BY_PERIOD,
        HarenaIntentType.SPENDING_COMPARISON,
        HarenaIntentType.TREND_ANALYSIS,
        HarenaIntentType.CATEGORY_ANALYSIS,
        HarenaIntentType.COMPARISON_QUERY
    ],
    "ACCOUNT_BALANCE": [
        HarenaIntentType.BALANCE_INQUIRY,
        HarenaIntentType.ACCOUNT_BALANCE_SPECIFIC,
        HarenaIntentType.BALANCE_EVOLUTION
    ],
    "CONVERSATIONAL": [
        HarenaIntentType.GREETING,
        HarenaIntentType.CONFIRMATION,
        HarenaIntentType.CLARIFICATION,
        HarenaIntentType.GENERAL_QUESTION
    ],
    "UNSUPPORTED": [
        HarenaIntentType.TRANSFER_REQUEST,
        HarenaIntentType.PAYMENT_REQUEST,
        HarenaIntentType.CARD_BLOCK,
        HarenaIntentType.BUDGET_INQUIRY,
        HarenaIntentType.GOAL_TRACKING,
        HarenaIntentType.EXPORT_REQUEST,
        HarenaIntentType.OUT_OF_SCOPE
    ],
    "UNCLEAR_INTENT": [
        HarenaIntentType.UNCLEAR_INTENT,
        HarenaIntentType.UNKNOWN,
        HarenaIntentType.TEST_INTENT,
        HarenaIntentType.ERROR
    ]
}


# Fonctions utilitaires optimisées avec cache
@lru_cache(maxsize=128)
def get_intent_metadata(intent: HarenaIntentType) -> IntentMetadata:
    """Récupère les métadonnées d'une intention (avec cache LRU)"""
    return INTENT_METADATA.get(intent, INTENT_METADATA[HarenaIntentType.ERROR])


@lru_cache(maxsize=64)
def get_intents_by_category(category: str) -> List[HarenaIntentType]:
    """Récupère les intentions d'une catégorie (avec cache LRU)"""
    return INTENT_CATEGORIES.get(category, [])


@lru_cache(maxsize=128)
def get_intent_category(intent: HarenaIntentType) -> str:
    """Trouve la catégorie d'une intention (avec cache LRU)"""
    for category, intents in INTENT_CATEGORIES.items():
        if intent in intents:
            return category
    return "UNKNOWN_CATEGORY"


@lru_cache(maxsize=128)
def is_intent_supported(intent: HarenaIntentType) -> bool:
    """Vérifie si une intention est supportée (avec cache LRU)"""
    metadata = get_intent_metadata(intent)
    return metadata.is_supported


@lru_cache(maxsize=64)
def get_supported_intents() -> List[HarenaIntentType]:
    """Récupère toutes les intentions supportées"""
    return [intent for intent in HarenaIntentType if is_intent_supported(intent)]


@lru_cache(maxsize=64)
def get_unsupported_intents() -> List[HarenaIntentType]:
    """Récupère toutes les intentions non supportées"""
    return [intent for intent in HarenaIntentType if not is_intent_supported(intent)]


def get_related_intents(intent: HarenaIntentType) -> List[HarenaIntentType]:
    """Récupère les intentions liées à une intention donnée"""
    metadata = get_intent_metadata(intent)
    related = []
    
    for related_name in metadata.related_intents:
        try:
            related_intent = HarenaIntentType(related_name)
            related.append(related_intent)
        except ValueError:
            continue
    
    return related


def get_intent_keywords(intent: HarenaIntentType) -> Tuple[str, ...]:
    """Récupère les mots-clés d'une intention"""
    metadata = get_intent_metadata(intent)
    return metadata.keywords


def get_intent_examples(intent: HarenaIntentType) -> Tuple[str, ...]:
    """Récupère les exemples d'une intention"""
    metadata = get_intent_metadata(intent)
    return metadata.examples


def get_intents_by_frequency(frequency: str) -> List[HarenaIntentType]:
    """Récupère les intentions par niveau de fréquence"""
    return [
        intent for intent in HarenaIntentType 
        if get_intent_metadata(intent).frequency == frequency
    ]


def get_intents_by_complexity(complexity: str) -> List[HarenaIntentType]:
    """Récupère les intentions par niveau de complexité"""
    return [
        intent for intent in HarenaIntentType 
        if get_intent_metadata(intent).complexity == complexity
    ]


def search_intents_by_keyword(keyword: str) -> List[HarenaIntentType]:
    """Recherche d'intentions par mot-clé"""
    keyword_lower = keyword.lower()
    matching_intents = []
    
    for intent in HarenaIntentType:
        metadata = get_intent_metadata(intent)
        
        # Recherche dans les mots-clés
        if any(keyword_lower in kw.lower() for kw in metadata.keywords):
            matching_intents.append(intent)
            continue
        
        # Recherche dans la description
        if keyword_lower in metadata.description.lower():
            matching_intents.append(intent)
            continue
        
        # Recherche dans les exemples
        if any(keyword_lower in ex.lower() for ex in metadata.examples):
            matching_intents.append(intent)
    
    return matching_intents


def get_high_confidence_intents(threshold: float = 0.8) -> List[HarenaIntentType]:
    """Récupère les intentions avec un seuil de confiance élevé"""
    return [
        intent for intent in HarenaIntentType 
        if get_intent_metadata(intent).confidence_threshold >= threshold
    ]


def validate_intent_taxonomy() -> Dict[str, List[str]]:
    """Valide la cohérence de la taxonomie"""
    issues = {
        "missing_metadata": [],
        "invalid_categories": [],
        "orphaned_intents": [],
        "invalid_confidence": []
    }
    
    # Vérification métadonnées manquantes
    for intent in HarenaIntentType:
        if intent not in INTENT_METADATA:
            issues["missing_metadata"].append(intent.value)
    
    # Vérification catégories
    all_categorized_intents = set()
    for category, intents in INTENT_CATEGORIES.items():
        all_categorized_intents.update(intents)
    
    for intent in HarenaIntentType:
        if intent not in all_categorized_intents:
            issues["orphaned_intents"].append(intent.value)
    
    # Vérification cohérence confiance
    for intent, metadata in INTENT_METADATA.items():
        if not (0.0 <= metadata.confidence_threshold <= 1.0):
            issues["invalid_confidence"].append(intent.value)
    
    return issues


# Statistiques et informations sur la taxonomie
def get_taxonomy_statistics() -> Dict[str, Any]:
    """Statistiques complètes de la taxonomie"""
    stats = {
        "total_intents": len(HarenaIntentType),
        "supported_intents": len(get_supported_intents()),
        "unsupported_intents": len(get_unsupported_intents()),
        "categories": len(INTENT_CATEGORIES),
        "category_distribution": {},
        "frequency_distribution": {},
        "complexity_distribution": {},
        "avg_confidence_threshold": 0.0
    }
    
    # Distribution par catégorie
    for category, intents in INTENT_CATEGORIES.items():
        stats["category_distribution"][category] = len(intents)
    
    # Distribution par fréquence
    frequencies = ["very_high", "high", "medium", "low", "very_low"]
    for freq in frequencies:
        stats["frequency_distribution"][freq] = len(get_intents_by_frequency(freq))
    
    # Distribution par complexité
    complexities = ["simple", "medium", "complex"]
    for comp in complexities:
        stats["complexity_distribution"][comp] = len(get_intents_by_complexity(comp))
    
    # Seuil de confiance moyen
    total_confidence = sum(metadata.confidence_threshold for metadata in INTENT_METADATA.values())
    stats["avg_confidence_threshold"] = total_confidence / len(INTENT_METADATA)
    
    return stats


# Constantes pour compatibilité
INTENT_DESCRIPTIONS: Dict[HarenaIntentType, str] = {
    intent: metadata.description 
    for intent, metadata in INTENT_METADATA.items()
}