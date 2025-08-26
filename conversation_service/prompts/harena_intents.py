"""
Taxonomie des intentions Harena basée sur INTENTS.md
"""
from enum import Enum
from typing import Dict, List

class HarenaIntentType(str, Enum):
    # TRANSACTIONS
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
    
    # ANALYSE DÉPENSES
    SPENDING_ANALYSIS = "SPENDING_ANALYSIS"
    SPENDING_ANALYSIS_BY_CATEGORY = "SPENDING_ANALYSIS_BY_CATEGORY"
    SPENDING_ANALYSIS_BY_PERIOD = "SPENDING_ANALYSIS_BY_PERIOD"
    SPENDING_COMPARISON = "SPENDING_COMPARISON"
    TREND_ANALYSIS = "TREND_ANALYSIS"
    CATEGORY_ANALYSIS = "CATEGORY_ANALYSIS"
    COMPARISON_QUERY = "COMPARISON_QUERY"
    
    # SOLDES COMPTES
    BALANCE_INQUIRY = "BALANCE_INQUIRY"
    ACCOUNT_BALANCE_SPECIFIC = "ACCOUNT_BALANCE_SPECIFIC"
    BALANCE_EVOLUTION = "BALANCE_EVOLUTION"
    
    # CONVERSATIONNEL
    GREETING = "GREETING"
    CONFIRMATION = "CONFIRMATION"
    CLARIFICATION = "CLARIFICATION"
    GENERAL_QUESTION = "GENERAL_QUESTION"
    
    # NON SUPPORTÉES
    TRANSFER_REQUEST = "TRANSFER_REQUEST"
    PAYMENT_REQUEST = "PAYMENT_REQUEST"
    CARD_BLOCK = "CARD_BLOCK"
    BUDGET_INQUIRY = "BUDGET_INQUIRY"
    GOAL_TRACKING = "GOAL_TRACKING"
    EXPORT_REQUEST = "EXPORT_REQUEST"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    
    # AMBIGUËS/ERREURS
    UNCLEAR_INTENT = "UNCLEAR_INTENT"
    UNKNOWN = "UNKNOWN"
    TEST_INTENT = "TEST_INTENT"
    ERROR = "ERROR"

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

INTENT_DESCRIPTIONS: Dict[HarenaIntentType, str] = {
    # TRANSACTIONS
    HarenaIntentType.TRANSACTION_SEARCH: "Rechercher toutes transactions sans filtre spécifique",
    HarenaIntentType.SEARCH_BY_DATE: "Rechercher transactions pour une date ou période précise",
    HarenaIntentType.SEARCH_BY_AMOUNT: "Rechercher transactions par montant ou plage de montants",
    HarenaIntentType.SEARCH_BY_MERCHANT: "Rechercher transactions liées à un marchand précis",
    HarenaIntentType.SEARCH_BY_CATEGORY: "Rechercher transactions par catégorie de dépense",
    HarenaIntentType.SEARCH_BY_AMOUNT_AND_DATE: "Rechercher transactions combinant critères montant et date",
    HarenaIntentType.SEARCH_BY_OPERATION_TYPE: "Rechercher transactions par type d'opération (débit/crédit/carte)",
    HarenaIntentType.SEARCH_BY_TEXT: "Recherche textuelle libre dans les descriptions",
    HarenaIntentType.COUNT_TRANSACTIONS: "Compter les transactions correspondant à des critères",
    HarenaIntentType.MERCHANT_INQUIRY: "Analyse détaillée des transactions avec un marchand",
    HarenaIntentType.FILTER_REQUEST: "Raffiner une requête existante avec des filtres",
    
    # ANALYSE DÉPENSES  
    HarenaIntentType.SPENDING_ANALYSIS: "Analyse globale des dépenses sur une période",
    HarenaIntentType.SPENDING_ANALYSIS_BY_CATEGORY: "Répartition des dépenses par catégorie",
    HarenaIntentType.SPENDING_ANALYSIS_BY_PERIOD: "Analyse des dépenses par période temporelle",
    HarenaIntentType.SPENDING_COMPARISON: "Comparaison de dépenses entre périodes ou catégories",
    HarenaIntentType.TREND_ANALYSIS: "Analyse des tendances d'évolution des dépenses",
    HarenaIntentType.CATEGORY_ANALYSIS: "Analyse détaillée par catégories de dépenses",
    HarenaIntentType.COMPARISON_QUERY: "Comparaison ciblée entre éléments spécifiques",
    
    # SOLDES
    HarenaIntentType.BALANCE_INQUIRY: "Consulter le solde général actuel",
    HarenaIntentType.ACCOUNT_BALANCE_SPECIFIC: "Consulter le solde d'un compte précis",
    HarenaIntentType.BALANCE_EVOLUTION: "Historique et évolution du solde",
    
    # CONVERSATIONNEL
    HarenaIntentType.GREETING: "Salutations et politesses d'ouverture",
    HarenaIntentType.CONFIRMATION: "Confirmations et remerciements",
    HarenaIntentType.CLARIFICATION: "Demandes de précision ou d'explication",
    HarenaIntentType.GENERAL_QUESTION: "Questions générales sur l'utilisation",
    
    # NON SUPPORTÉES
    HarenaIntentType.TRANSFER_REQUEST: "Demande de virement (non supporté)",
    HarenaIntentType.PAYMENT_REQUEST: "Demande de paiement (non supporté)",
    HarenaIntentType.CARD_BLOCK: "Blocage de carte (non supporté)",
    HarenaIntentType.BUDGET_INQUIRY: "Consultation budget (non supporté)",
    HarenaIntentType.GOAL_TRACKING: "Suivi objectifs (non supporté)",
    HarenaIntentType.EXPORT_REQUEST: "Export de données (non supporté)",
    HarenaIntentType.OUT_OF_SCOPE: "Hors domaine financier",
    
    # ERREURS
    HarenaIntentType.UNCLEAR_INTENT: "Intention ambiguë ou non claire",
    HarenaIntentType.UNKNOWN: "Message incompréhensible",
    HarenaIntentType.TEST_INTENT: "Message de test",
    HarenaIntentType.ERROR: "Erreur de traitement"
}